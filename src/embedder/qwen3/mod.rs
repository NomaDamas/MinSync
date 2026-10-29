//! Qwen3 text-embedding model for the in-process native embedder.
//!
//! Ported from fastembed-rs 7.0.1 (`src/models/qwen3.rs`, Apache-2.0,
//! <https://github.com/Anush008/fastembed-rs>); see `THIRD_PARTY_NOTICES.md`.
//! The architecture and weight names are unchanged, so the same Hugging Face
//! checkpoints load. What differs from the original:
//!
//! * attention runs on a per-device backend (see [`attention`]) instead of one
//!   eager implementation, which removes the quadratic score-matrix memory;
//! * RMSNorm uses candle's fused kernel;
//! * the padding mask is built on the host in one pass, and the rotary tables
//!   are computed once at load time;
//! * only the last token is normalized, since it is the only one pooled;
//! * the final L2 normalization runs in f32 whatever the model dtype.

mod attention;

pub use attention::{AttentionBackend, AttentionRequest};

use crate::embedder::hub::{hf_repo, safetensors_weight_files};
use crate::error::{MinSyncError, Result as MinSyncResult};
use attention::{attend, BatchMask};
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{linear, linear_no_bias, Activation, Linear, Module, VarBuilder};
use serde::Deserialize;
use std::path::PathBuf;
use tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

#[derive(Deserialize, Debug, Clone)]
pub struct Config {
    #[serde(default)]
    pub attention_bias: bool,
    pub head_dim: Option<usize>,
    pub hidden_act: Activation,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub vocab_size: usize,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    fn num_kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}

/// Qwen3-VL embedding checkpoints nest the language model's config and weights.
#[derive(Deserialize)]
struct VisionLanguageConfig {
    text_config: Config,
}

fn parse_config(bytes: &[u8]) -> MinSyncResult<(Config, Option<&'static str>)> {
    if let Ok(config) = serde_json::from_slice::<Config>(bytes) {
        return Ok((config, None));
    }
    if let Ok(config) = serde_json::from_slice::<VisionLanguageConfig>(bytes) {
        return Ok((config.text_config, Some("model.language_model")));
    }
    Err(MinSyncError::Embedding(
        "native model config is neither a Qwen3 nor a Qwen3-VL text config".to_string(),
    ))
}

struct RmsNorm {
    weight: Tensor,
    eps: f32,
}

impl RmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get(dim, "weight")?,
            eps: eps as f32,
        })
    }
}

impl Module for RmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        candle_nn::ops::rms_norm(&xs.contiguous()?, &self.weight, self.eps)
    }
}

struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let up = xs.apply(&self.up_proj)?;
        (gate * up)?.apply(&self.down_proj)
    }
}

/// `cos` / `sin` of every position below the tokenizer's maximum length,
/// `[positions, head_dim]` in the model dtype.
struct RotaryTable {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryTable {
    fn new(cfg: &Config, positions: usize, device: &Device, dtype: DType) -> Result<Self> {
        let dim = cfg.head_dim();
        if !dim.is_multiple_of(2) {
            candle_core::bail!("head_dim must be even, got {dim}");
        }
        let half = dim / 2;
        let inv_freq: Vec<f64> = (0..half)
            .map(|i| 1.0 / cfg.rope_theta.powf((2 * i) as f64 / dim as f64))
            .collect();
        let mut cos = Vec::with_capacity(positions * dim);
        let mut sin = Vec::with_capacity(positions * dim);
        for position in 0..positions {
            for index in 0..dim {
                let angle = position as f64 * inv_freq[index % half];
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }
        let table =
            |values: Vec<f32>| Tensor::from_vec(values, (positions, dim), device)?.to_dtype(dtype);
        Ok(Self {
            cos: table(cos)?,
            sin: table(sin)?,
        })
    }

    /// `[1, 1, seq, head_dim]` views that broadcast over batch and heads.
    fn rows(&self, seq: usize) -> Result<(Tensor, Tensor)> {
        let rows = |t: &Tensor| t.narrow(0, 0, seq)?.unsqueeze(0)?.unsqueeze(0);
        Ok((rows(&self.cos)?, rows(&self.sin)?))
    }
}

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let half = x.dim(D::Minus1)? / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    Tensor::cat(&[&x2.neg()?, &x1], x.rank() - 1)
}

fn apply_rotary(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    x.broadcast_mul(cos)? + rotate_half(x)?.broadcast_mul(sin)?
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    scale: f32,
}

impl Attention {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.head_dim();
        if !cfg
            .num_attention_heads
            .is_multiple_of(cfg.num_key_value_heads)
        {
            candle_core::bail!("num_attention_heads must be a multiple of num_key_value_heads");
        }
        let q_out = cfg.num_attention_heads * head_dim;
        let kv_out = cfg.num_key_value_heads * head_dim;
        let projection = |inputs: usize, outputs: usize, name: &str| -> Result<Linear> {
            if cfg.attention_bias {
                linear(inputs, outputs, vb.pp(name))
            } else {
                linear_no_bias(inputs, outputs, vb.pp(name))
            }
        };
        Ok(Self {
            q_proj: projection(cfg.hidden_size, q_out, "q_proj")?,
            k_proj: projection(cfg.hidden_size, kv_out, "k_proj")?,
            v_proj: projection(cfg.hidden_size, kv_out, "v_proj")?,
            o_proj: projection(q_out, cfg.hidden_size, "o_proj")?,
            q_norm: RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            num_kv_groups: cfg.num_kv_groups(),
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        rotary: (&Tensor, &Tensor),
        mask: &BatchMask,
        backend: AttentionBackend,
    ) -> Result<Tensor> {
        let (batch, seq, _) = hidden.dims3()?;
        let dim = self.head_dim;
        let q = hidden
            .apply(&self.q_proj)?
            .reshape((batch, seq, self.num_heads, dim))?;
        let q = q.apply(&self.q_norm)?.transpose(1, 2)?;
        let k = hidden
            .apply(&self.k_proj)?
            .reshape((batch, seq, self.num_kv_heads, dim))?;
        let k = k.apply(&self.k_norm)?.transpose(1, 2)?;
        let v = hidden
            .apply(&self.v_proj)?
            .reshape((batch, seq, self.num_kv_heads, dim))?
            .transpose(1, 2)?;
        let q = apply_rotary(&q, rotary.0, rotary.1)?;
        let k = apply_rotary(&k, rotary.0, rotary.1)?;
        let out = attend(backend, &q, &k, &v, mask, self.scale, self.num_kv_groups)?;
        out.transpose(1, 2)?
            .reshape((batch, seq, self.num_heads * dim))?
            .apply(&self.o_proj)
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn: Attention::new(cfg, vb.pp("self_attn"))?,
            mlp: Mlp::new(cfg, vb.pp("mlp"))?,
            input_layernorm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        rotary: (&Tensor, &Tensor),
        mask: &BatchMask,
        backend: AttentionBackend,
    ) -> Result<Tensor> {
        let attended =
            self.self_attn
                .forward(&hidden.apply(&self.input_layernorm)?, rotary, mask, backend)?;
        let hidden = (hidden + attended)?;
        let mlp = hidden
            .apply(&self.post_attention_layernorm)?
            .apply(&self.mlp)?;
        hidden + mlp
    }
}

pub struct Qwen3Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    rotary: RotaryTable,
    cfg: Config,
    backend: AttentionBackend,
    device: Device,
    dtype: DType,
}

impl Qwen3Model {
    fn new(
        cfg: Config,
        vb: VarBuilder,
        backend: AttentionBackend,
        positions: usize,
    ) -> Result<Self> {
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        let layers = (0..cfg.num_hidden_layers)
            .map(|index| DecoderLayer::new(&cfg, vb.pp(format!("layers.{index}"))))
            .collect::<Result<Vec<_>>>()?;
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        let rotary = RotaryTable::new(&cfg, positions, vb.device(), vb.dtype())?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary,
            cfg,
            backend,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Final hidden state of the last position of every row, `[batch, hidden]`.
    /// Rows are left padded by `pad_lens`, so the last position is always a real token.
    fn forward_last(&self, input_ids: &Tensor, pad_lens: &[usize]) -> Result<Tensor> {
        let (batch, seq) = input_ids.dims2()?;
        if pad_lens.len() != batch {
            candle_core::bail!("expected {batch} padding lengths, got {}", pad_lens.len());
        }
        let mask = BatchMask::new(
            self.backend,
            pad_lens.to_vec(),
            seq,
            &self.device,
            self.dtype,
        )?;
        let (cos, sin) = self.rotary.rows(seq)?;
        let mut hidden = self.embed_tokens.forward(input_ids)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, (&cos, &sin), &mask, self.backend)?;
        }
        self.norm
            .forward(&hidden.narrow(1, seq - 1, 1)?.squeeze(1)?)
    }
}

pub struct Qwen3Embedder {
    model: Qwen3Model,
    tokenizer: Tokenizer,
}

/// Where and how to load a model.
pub struct LoadOptions<'a> {
    pub repo_id: &'a str,
    /// Cache used when `HF_HOME` is not set.
    pub default_cache_dir: PathBuf,
    pub device: &'a Device,
    pub dtype: DType,
    pub max_length: usize,
    pub attention: AttentionRequest,
}

impl Qwen3Embedder {
    pub fn from_hf(options: &LoadOptions<'_>) -> MinSyncResult<Self> {
        let load_error = |what: &str, error: &dyn std::fmt::Display| {
            MinSyncError::Embedding(format!("native model {what}: {error}"))
        };
        let repo = hf_repo(options.repo_id, options.default_cache_dir.clone())?;
        let config_path = repo
            .get("config.json")
            .map_err(|e| load_error("config download failed", &e))?;
        let config_bytes = std::fs::read(config_path)?;
        let (config, weight_prefix) = parse_config(&config_bytes)?;
        let backend = AttentionBackend::resolve(
            options.attention,
            options.device,
            options.dtype,
            config.head_dim(),
        )?;
        let weights = safetensors_weight_files(&repo)?;
        // SAFETY: the safetensors files are not modified while the model is loaded.
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&weights, options.dtype, options.device) }
                .map_err(|e| load_error("weight load failed", &e))?;
        let vb = match weight_prefix {
            Some(prefix) => vb.pp(prefix),
            None => vb,
        };
        let model = Qwen3Model::new(config, vb, backend, options.max_length)
            .map_err(|e| load_error("build failed", &e))?;
        let tokenizer_path = repo
            .get("tokenizer.json")
            .map_err(|e| load_error("tokenizer download failed", &e))?;
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| load_error("tokenizer parse failed", &e))?;
        Self::new(model, tokenizer, options.max_length)
    }

    fn new(model: Qwen3Model, mut tokenizer: Tokenizer, max_length: usize) -> MinSyncResult<Self> {
        // Left padding keeps the last position a real token for last-token pooling.
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            direction: PaddingDirection::Left,
            ..Default::default()
        }));
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length,
                ..Default::default()
            }))
            .map_err(|e| {
                MinSyncError::Embedding(format!("native tokenizer truncation failed: {e}"))
            })?;
        Ok(Self { model, tokenizer })
    }

    pub fn config(&self) -> &Config {
        &self.model.cfg
    }

    pub fn backend(&self) -> AttentionBackend {
        self.model.backend
    }

    /// Unit-length embeddings, one per input.
    pub fn embed<S: AsRef<str>>(&self, texts: &[S]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let encodings = self
            .tokenizer
            .encode_batch(texts.iter().map(AsRef::as_ref).collect::<Vec<_>>(), true)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let batch = encodings.len();
        let seq = encodings[0].len();
        if seq == 0 {
            candle_core::bail!("tokenizer produced no tokens");
        }
        let mut ids = Vec::with_capacity(batch * seq);
        let mut pad_lens = Vec::with_capacity(batch);
        for encoding in &encodings {
            ids.extend_from_slice(encoding.get_ids());
            pad_lens.push(
                encoding
                    .get_attention_mask()
                    .iter()
                    .filter(|&&m| m == 0)
                    .count(),
            );
        }
        let input_ids = Tensor::from_vec(ids, (batch, seq), &self.model.device)?;
        let pooled = self
            .model
            .forward_last(&input_ids, &pad_lens)?
            .to_dtype(DType::F32)?;
        let norm = (pooled.sqr()?.sum_keepdim(1)? + 1e-12)?.sqrt()?;
        pooled.broadcast_div(&norm)?.to_vec2::<f32>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Deterministic pseudo-random values in [-0.5, 0.5).
    fn pseudo(len: usize, seed: u32) -> Vec<f32> {
        let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(7);
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
            })
            .collect()
    }

    fn tiny_config() -> Config {
        Config {
            attention_bias: false,
            head_dim: Some(32),
            hidden_act: Activation::Silu,
            hidden_size: 48,
            intermediate_size: 96,
            num_attention_heads: 4,
            num_hidden_layers: 2,
            num_key_value_heads: 2,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            vocab_size: 24,
        }
    }

    /// Random weights with the checkpoint's tensor names.
    fn tiny_weights(cfg: &Config, device: &Device) -> HashMap<String, Tensor> {
        let mut weights = HashMap::new();
        let mut seed = 0;
        let mut put = |name: String, shape: &[usize], norm: bool| {
            seed += 1;
            let len: usize = shape.iter().product();
            let fan_in = *shape.last().unwrap() as f32;
            let values: Vec<f32> = pseudo(len, seed)
                .into_iter()
                .map(|v| {
                    if norm {
                        1.0 + 0.2 * v
                    } else {
                        v * 2.0 / fan_in.sqrt()
                    }
                })
                .collect();
            weights.insert(
                name,
                Tensor::from_vec(values, shape.to_vec(), device).unwrap(),
            );
        };
        let head_dim = cfg.head_dim();
        put(
            "embed_tokens.weight".into(),
            &[cfg.vocab_size, cfg.hidden_size],
            false,
        );
        for layer in 0..cfg.num_hidden_layers {
            let p = format!("layers.{layer}");
            let q_out = cfg.num_attention_heads * head_dim;
            let kv_out = cfg.num_key_value_heads * head_dim;
            put(
                format!("{p}.self_attn.q_proj.weight"),
                &[q_out, cfg.hidden_size],
                false,
            );
            put(
                format!("{p}.self_attn.k_proj.weight"),
                &[kv_out, cfg.hidden_size],
                false,
            );
            put(
                format!("{p}.self_attn.v_proj.weight"),
                &[kv_out, cfg.hidden_size],
                false,
            );
            put(
                format!("{p}.self_attn.o_proj.weight"),
                &[cfg.hidden_size, q_out],
                false,
            );
            put(format!("{p}.self_attn.q_norm.weight"), &[head_dim], true);
            put(format!("{p}.self_attn.k_norm.weight"), &[head_dim], true);
            put(
                format!("{p}.mlp.gate_proj.weight"),
                &[cfg.intermediate_size, cfg.hidden_size],
                false,
            );
            put(
                format!("{p}.mlp.up_proj.weight"),
                &[cfg.intermediate_size, cfg.hidden_size],
                false,
            );
            put(
                format!("{p}.mlp.down_proj.weight"),
                &[cfg.hidden_size, cfg.intermediate_size],
                false,
            );
            put(
                format!("{p}.input_layernorm.weight"),
                &[cfg.hidden_size],
                true,
            );
            put(
                format!("{p}.post_attention_layernorm.weight"),
                &[cfg.hidden_size],
                true,
            );
        }
        put("norm.weight".into(), &[cfg.hidden_size], true);
        weights
    }

    fn tiny_model(device: &Device, dtype: DType, backend: AttentionBackend) -> Qwen3Model {
        let cfg = tiny_config();
        let vb = VarBuilder::from_tensors(tiny_weights(&cfg, device), dtype, device);
        Qwen3Model::new(cfg, vb, backend, 64).expect("tiny model")
    }

    fn tokenizer() -> Tokenizer {
        Tokenizer::from_bytes(
            br#"{
                "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
                "normalizer": null, "pre_tokenizer": {"type": "Whitespace"},
                "post_processor": null, "decoder": null,
                "model": {"type": "WordLevel",
                          "vocab": {"[UNK]": 0, "a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6},
                          "unk_token": "[UNK]"}
            }"#,
        )
        .expect("tokenizer")
    }

    fn embedder(device: &Device, dtype: DType, backend: AttentionBackend) -> Qwen3Embedder {
        Qwen3Embedder::new(tiny_model(device, dtype, backend), tokenizer(), 64).expect("embedder")
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (norm(a) * norm(b))
    }

    #[test]
    fn every_cpu_backend_produces_the_same_embeddings() {
        let device = Device::Cpu;
        let texts = ["a b c d e f a b c d", "a b", "c d e", "f"];
        let reference = embedder(&device, DType::F32, AttentionBackend::Eager)
            .embed(&texts)
            .unwrap();
        for backend in [AttentionBackend::Tiled, AttentionBackend::CpuFlash] {
            let vectors = embedder(&device, DType::F32, backend)
                .embed(&texts)
                .unwrap();
            for (row, (a, b)) in reference.iter().zip(&vectors).enumerate() {
                let diff = a
                    .iter()
                    .zip(b)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0, f32::max);
                assert!(
                    diff < 1e-4,
                    "{backend:?} row {row} differs from eager by {diff}"
                );
            }
        }
    }

    #[test]
    fn embeddings_are_unit_length_and_independent_of_batch_padding() {
        let device = Device::Cpu;
        for backend in [
            AttentionBackend::Eager,
            AttentionBackend::Tiled,
            AttentionBackend::CpuFlash,
        ] {
            let embedder = embedder(&device, DType::F32, backend);
            let alone = embedder.embed(&["c d e"]).unwrap().remove(0);
            let norm: f32 = alone.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "{backend:?}: norm {norm}");
            // Batched with a longer text, so the short one is left padded.
            let batched = embedder
                .embed(&["a b c d e f a b c d", "c d e"])
                .unwrap()
                .remove(1);
            let similarity = cosine(&alone, &batched);
            assert!(
                similarity > 0.99999,
                "{backend:?}: padding changed the embedding ({similarity})"
            );
        }
    }

    #[test]
    fn empty_input_yields_no_embeddings() {
        let embedder = embedder(&Device::Cpu, DType::F32, AttentionBackend::CpuFlash);
        assert!(embedder.embed::<&str>(&[]).unwrap().is_empty());
    }

    #[test]
    fn fused_rms_norm_matches_the_textbook_definition() {
        let device = Device::Cpu;
        let values = Tensor::from_vec(pseudo(3 * 48, 9), (3, 48), &device).unwrap();
        let weight = Tensor::from_vec(pseudo(48, 4), 48, &device).unwrap();
        let norm = RmsNorm {
            weight: weight.clone(),
            eps: 1e-6,
        };
        let fused = norm.forward(&values).unwrap();
        let variance = values.sqr().unwrap().mean_keepdim(D::Minus1).unwrap();
        let expected = values
            .broadcast_div(&(variance + 1e-6).unwrap().sqrt().unwrap())
            .unwrap()
            .broadcast_mul(&weight)
            .unwrap();
        let diff = (fused - expected)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-5, "fused rms_norm differs by {diff}");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_fused_attention_matches_the_reference_in_f32_and_f16() {
        let Ok(device) = Device::new_metal(0) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let texts = ["a b c d e f a b c d", "a b", "c d e", "f"];
        let reference = embedder(&Device::Cpu, DType::F32, AttentionBackend::Eager)
            .embed(&texts)
            .unwrap();
        for (dtype, floor) in [(DType::F32, 0.999_99), (DType::F16, 0.998)] {
            let vectors = embedder(&device, dtype, AttentionBackend::MetalSdpa)
                .embed(&texts)
                .unwrap();
            for (row, (a, b)) in reference.iter().zip(&vectors).enumerate() {
                let similarity = cosine(a, b);
                assert!(
                    similarity > floor,
                    "{dtype:?} row {row}: cosine {similarity}"
                );
            }
        }
    }

    #[test]
    fn config_parses_plain_and_vision_language_layouts() {
        let text = br#"{"hidden_act":"silu","hidden_size":8,"intermediate_size":16,
            "num_attention_heads":2,"num_hidden_layers":1,"num_key_value_heads":1,
            "rms_norm_eps":1e-6,"rope_theta":1000000.0,"vocab_size":10,"head_dim":4}"#;
        let (config, prefix) = parse_config(text).unwrap();
        assert_eq!((config.head_dim(), prefix), (4, None));
        let nested = format!(
            r#"{{"text_config":{}}}"#,
            std::str::from_utf8(text).unwrap()
        );
        assert_eq!(
            parse_config(nested.as_bytes()).unwrap().1,
            Some("model.language_model")
        );
        assert!(parse_config(b"{}").is_err());
    }
}
