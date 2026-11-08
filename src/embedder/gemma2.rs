//! EmbeddingGemma 2 text tower for the in-process native embedder.
//!
//! Written against the reference implementation of
//! `google/embeddinggemma-2` (`transformers` `EmbeddingGemma2TextModel`,
//! Apache-2.0) and the checkpoint's Sentence-Transformers pipeline: masked mean
//! pooling, then L2 normalization. Only the text modality is loaded; the
//! vision and audio towers and their projections in the checkpoint are never
//! read. What the architecture does:
//!
//! * a bidirectional encoder: no causal mask, no KV cache;
//! * 24 layers, every sixth one `full_attention` (wider heads, one KV head, a
//!   1e6 rotary base) and the rest `sliding_attention` (a symmetric window of
//!   `|q - kv| <= sliding_window` tokens, a 1e4 base);
//! * attention scores are not scaled (`scaling = 1`): q and k are RMS
//!   normalized per head instead, and v is normalized without a weight;
//! * four norms per layer around attention and the MLP, then a third residual
//!   block (the per-layer embedding gate) and a learned per-layer scalar;
//! * per-layer embeddings are projection-only: derived from the scaled token
//!   embeddings, with no extra lookup table;
//! * the final hidden state is projected from `hidden_size` to
//!   `embedding_dim`. A linear map commutes with mean pooling, so the pooled
//!   vector is projected once instead of every token.
//!
//! The checkpoint overflows float16, so only f32 and bf16 are accepted.

use crate::embedder::hub::ModelSource;
use crate::embedder::qwen3::{AttentionBackend, AttentionRequest, LoadOptions};
use crate::error::{MinSyncError, Result as MinSyncResult};
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{linear_no_bias, Activation, Linear, Module, VarBuilder};
use serde::Deserialize;
use std::collections::HashMap;
use tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

/// Added to masked attention scores. Finite on purpose: a row whose every key
/// is masked (a padding query beyond the window of every real token) then
/// softmaxes to a uniform row instead of NaN, and a NaN would reach the real
/// rows through `0 * NaN` in the value product.
const MASKED: f32 = -1e9;

/// `model_type` of the checkpoint's `config.json`.
pub const MODEL_TYPE: &str = "embedding_gemma2";

#[derive(Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
enum LayerKind {
    #[serde(rename = "sliding_attention")]
    Sliding,
    #[serde(rename = "full_attention")]
    Full,
}

#[derive(Deserialize, Debug, Clone)]
struct RopeParameters {
    rope_theta: f64,
    #[serde(default = "default_rope_type")]
    rope_type: String,
}

fn default_rope_type() -> String {
    "default".to_string()
}

#[derive(Deserialize, Debug, Clone)]
struct RopeByKind {
    full_attention: RopeParameters,
    sliding_attention: RopeParameters,
}

/// Per-layer overrides of the head shape (the full-attention layers).
#[derive(Deserialize, Debug, Clone)]
struct LayerOverride {
    head_dim: Option<usize>,
    num_key_value_heads: Option<usize>,
}

/// The `text_config` of the checkpoint.
#[derive(Deserialize, Debug, Clone)]
pub struct Config {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    hidden_activation: Activation,
    rms_norm_eps: f64,
    #[serde(default)]
    attention_bias: bool,
    /// Inclusive radius of the symmetric window of the sliding layers.
    sliding_window: usize,
    layer_types: Vec<LayerKind>,
    hidden_size_per_layer_input: usize,
    embedding_dim: usize,
    rope_parameters: RopeByKind,
    /// Keyed by zero-padded layer index (`"05"`).
    #[serde(default)]
    per_layer_config: HashMap<String, LayerOverride>,
}

/// The checkpoint nests the text config next to the vision and audio ones.
#[derive(Deserialize)]
struct CheckpointConfig {
    model_type: String,
    text_config: Config,
}

#[derive(Debug, Clone, Copy)]
struct LayerShape {
    kind: LayerKind,
    head_dim: usize,
    kv_heads: usize,
}

impl Config {
    /// Parse a checkpoint's `config.json`.
    pub fn parse(bytes: &[u8]) -> MinSyncResult<Self> {
        let checkpoint: CheckpointConfig = serde_json::from_slice(bytes).map_err(|error| {
            MinSyncError::Embedding(format!(
                "native model config is not an EmbeddingGemma 2 config: {error}"
            ))
        })?;
        if checkpoint.model_type != MODEL_TYPE {
            return Err(MinSyncError::Embedding(format!(
                "native model config has model_type {:?}, expected {MODEL_TYPE:?}",
                checkpoint.model_type
            )));
        }
        let config = checkpoint.text_config;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> MinSyncResult<()> {
        let invalid = |what: String| {
            Err(MinSyncError::Embedding(format!(
                "unsupported EmbeddingGemma 2 config: {what}"
            )))
        };
        if self.layer_types.len() != self.num_hidden_layers {
            return invalid(format!(
                "layer_types has {} entries for {} layers",
                self.layer_types.len(),
                self.num_hidden_layers
            ));
        }
        if self.hidden_activation != Activation::GeluPytorchTanh {
            return invalid(format!(
                "hidden_activation {:?}, expected gelu_pytorch_tanh",
                self.hidden_activation
            ));
        }
        if self.attention_bias {
            return invalid("attention_bias is not supported".to_string());
        }
        for (name, rope) in [
            ("full_attention", &self.rope_parameters.full_attention),
            ("sliding_attention", &self.rope_parameters.sliding_attention),
        ] {
            if rope.rope_type != "default" {
                return invalid(format!("rope_type {:?} for {name}", rope.rope_type));
            }
        }
        for key in self.per_layer_config.keys() {
            match key.parse::<usize>() {
                Ok(layer) if layer < self.num_hidden_layers => {}
                _ => return invalid(format!("per_layer_config key {key:?}")),
            }
        }
        for layer in 0..self.num_hidden_layers {
            let shape = self.layer(layer);
            if shape.head_dim == 0 || !shape.head_dim.is_multiple_of(2) {
                return invalid(format!("head_dim {} of layer {layer}", shape.head_dim));
            }
            if shape.kv_heads == 0 || !self.num_attention_heads.is_multiple_of(shape.kv_heads) {
                return invalid(format!(
                    "{} attention heads over {} KV heads in layer {layer}",
                    self.num_attention_heads, shape.kv_heads
                ));
            }
        }
        Ok(())
    }

    fn layer(&self, index: usize) -> LayerShape {
        let overrides = self
            .per_layer_config
            .iter()
            .find(|(key, _)| key.parse::<usize>() == Ok(index))
            .map(|(_, value)| value);
        LayerShape {
            kind: self.layer_types[index],
            head_dim: overrides
                .and_then(|value| value.head_dim)
                .unwrap_or(self.head_dim),
            kv_heads: overrides
                .and_then(|value| value.num_key_value_heads)
                .unwrap_or(self.num_key_value_heads),
        }
    }

    /// Length of the embedding the model returns.
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    pub fn num_attention_heads(&self) -> usize {
        self.num_attention_heads
    }
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
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = xs.apply(&self.gate_proj)?.gelu()?;
        (gate * xs.apply(&self.up_proj)?)?.apply(&self.down_proj)
    }
}

/// The per-layer embedding gate: a bottleneck down to
/// `hidden_size_per_layer_input`, multiplied elementwise by this layer's slice
/// of the per-layer embeddings, and back.
struct PleBlock {
    input_gate: Linear,
    projection: Linear,
    post_norm: RmsNorm,
}

impl PleBlock {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let ple = cfg.hidden_size_per_layer_input;
        Ok(Self {
            input_gate: linear_no_bias(cfg.hidden_size, ple, vb.pp("per_layer_input_gate"))?,
            projection: linear_no_bias(ple, cfg.hidden_size, vb.pp("per_layer_projection"))?,
            post_norm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_per_layer_input_norm"),
            )?,
        })
    }

    fn forward(&self, hidden: &Tensor, per_layer_input: &Tensor) -> Result<Tensor> {
        let gated = (hidden.apply(&self.input_gate)?.gelu()? * per_layer_input)?;
        hidden + gated.apply(&self.projection)?.apply(&self.post_norm)?
    }
}

/// `cos` / `sin` of every position below the context, `[positions, head_dim]`
/// in the model dtype.
struct RotaryTable {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryTable {
    fn new(
        head_dim: usize,
        theta: f64,
        positions: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let half = head_dim / 2;
        let inv_freq: Vec<f64> = (0..half)
            .map(|i| 1.0 / theta.powf((2 * i) as f64 / head_dim as f64))
            .collect();
        let mut cos = Vec::with_capacity(positions * head_dim);
        let mut sin = Vec::with_capacity(positions * head_dim);
        for position in 0..positions {
            for index in 0..head_dim {
                let angle = position as f64 * inv_freq[index % half];
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }
        let table = |values: Vec<f32>| {
            Tensor::from_vec(values, (positions, head_dim), device)?.to_dtype(dtype)
        };
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
    /// All-ones weight of the weightless value norm.
    v_ones: Tensor,
    eps: f32,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn new(cfg: &Config, shape: LayerShape, vb: VarBuilder) -> Result<Self> {
        let q_out = cfg.num_attention_heads * shape.head_dim;
        let kv_out = shape.kv_heads * shape.head_dim;
        Ok(Self {
            q_proj: linear_no_bias(cfg.hidden_size, q_out, vb.pp("q_proj"))?,
            k_proj: linear_no_bias(cfg.hidden_size, kv_out, vb.pp("k_proj"))?,
            v_proj: linear_no_bias(cfg.hidden_size, kv_out, vb.pp("v_proj"))?,
            o_proj: linear_no_bias(q_out, cfg.hidden_size, vb.pp("o_proj"))?,
            q_norm: RmsNorm::new(shape.head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: RmsNorm::new(shape.head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            v_ones: Tensor::ones(shape.head_dim, vb.dtype(), vb.device())?,
            eps: cfg.rms_norm_eps as f32,
            heads: cfg.num_attention_heads,
            kv_heads: shape.kv_heads,
            head_dim: shape.head_dim,
        })
    }

    /// `mask` is additive, `[batch, 1, seq, seq]` in f32; `None` lets every
    /// position attend to every other.
    fn forward(
        &self,
        hidden: &Tensor,
        rotary: (&Tensor, &Tensor),
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch, seq, _) = hidden.dims3()?;
        let dim = self.head_dim;
        let q = hidden
            .apply(&self.q_proj)?
            .reshape((batch, seq, self.heads, dim))?
            .apply(&self.q_norm)?
            .transpose(1, 2)?;
        let k = hidden
            .apply(&self.k_proj)?
            .reshape((batch, seq, self.kv_heads, dim))?
            .apply(&self.k_norm)?
            .transpose(1, 2)?;
        let v = hidden
            .apply(&self.v_proj)?
            .reshape((batch, seq, self.kv_heads, dim))?
            .contiguous()?;
        let v = candle_nn::ops::rms_norm(&v, &self.v_ones, self.eps)?.transpose(1, 2)?;
        let q = apply_rotary(&q, rotary.0, rotary.1)?.contiguous()?;
        let k = apply_rotary(&k, rotary.0, rotary.1)?.contiguous()?;
        let v = v.contiguous()?;

        // Grouped-query attention: query head `h` reads KV head `h / groups`.
        // The KV heads are repeated rather than the query rows stacked, which
        // keeps every tensor 4-D: Metal's broadcast of a 5-D mask over the
        // batch returns wrong rows.
        let groups = self.heads / self.kv_heads;
        let repeat = |x: &Tensor| -> Result<Tensor> {
            if groups == 1 {
                return Ok(x.clone());
            }
            let x = x.unsqueeze(2)?;
            let copies = vec![&x; groups];
            Tensor::cat(&copies, 2)?.reshape((batch, self.heads, seq, dim))
        };
        let (k, v) = (repeat(&k)?, repeat(&v)?);
        let mut scores = q.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        let dtype = scores.dtype();
        if dtype != DType::F32 {
            scores = scores.to_dtype(DType::F32)?;
        }
        if let Some(mask) = mask {
            scores = scores.broadcast_add(mask)?;
        }
        let mut weights = candle_nn::ops::softmax_last_dim(&scores)?;
        if dtype != DType::F32 {
            weights = weights.to_dtype(dtype)?;
        }
        weights
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((batch, seq, self.heads * dim))?
            .apply(&self.o_proj)
    }
}

struct Layer {
    kind: LayerKind,
    attention: Attention,
    mlp: Mlp,
    input_norm: RmsNorm,
    post_attention_norm: RmsNorm,
    pre_feedforward_norm: RmsNorm,
    post_feedforward_norm: RmsNorm,
    ple: PleBlock,
    scalar: Tensor,
}

impl Layer {
    fn new(cfg: &Config, index: usize, vb: VarBuilder) -> Result<Self> {
        let shape = cfg.layer(index);
        let norm = |name: &str| RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp(name));
        Ok(Self {
            kind: shape.kind,
            attention: Attention::new(cfg, shape, vb.pp("self_attn"))?,
            mlp: Mlp::new(cfg, vb.pp("mlp"))?,
            input_norm: norm("input_layernorm")?,
            post_attention_norm: norm("post_attention_layernorm")?,
            pre_feedforward_norm: norm("pre_feedforward_layernorm")?,
            post_feedforward_norm: norm("post_feedforward_layernorm")?,
            ple: PleBlock::new(cfg, vb.pp("ple_block"))?,
            scalar: vb.get(1, "layer_scalar")?,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        per_layer_input: &Tensor,
        rotary: (&Tensor, &Tensor),
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let attended = self
            .attention
            .forward(&hidden.apply(&self.input_norm)?, rotary, mask)?
            .apply(&self.post_attention_norm)?;
        let hidden = (hidden + attended)?;
        let mlp = hidden
            .apply(&self.pre_feedforward_norm)?
            .apply(&self.mlp)?
            .apply(&self.post_feedforward_norm)?;
        let hidden = (&hidden + mlp)?;
        self.ple
            .forward(&hidden, per_layer_input)?
            .broadcast_mul(&self.scalar)
    }
}

/// Additive attention mask for one layer kind: a position attends to the real
/// tokens of its own row, and for a sliding layer only to those within `window`
/// positions. `None` when nothing is masked at all.
fn attention_mask(
    lens: &[usize],
    seq: usize,
    window: Option<usize>,
    device: &Device,
) -> Result<Option<Tensor>> {
    let padded = lens.iter().any(|&len| len < seq);
    let windowed = window.is_some_and(|radius| radius + 1 < seq);
    if !padded && !windowed {
        return Ok(None);
    }
    let mut values = vec![0f32; lens.len() * seq * seq];
    for (row, &len) in lens.iter().enumerate() {
        let base = row * seq * seq;
        for query in 0..seq {
            for key in 0..seq {
                let outside = window.is_some_and(|radius| query.abs_diff(key) > radius);
                if key >= len || outside {
                    values[base + query * seq + key] = MASKED;
                }
            }
        }
    }
    Tensor::from_vec(values, (lens.len(), 1, seq, seq), device).map(Some)
}

pub struct Gemma2Model {
    embed_tokens: candle_nn::Embedding,
    embed_scale: f64,
    layers: Vec<Layer>,
    norm: RmsNorm,
    ple_projection: Linear,
    ple_norm: RmsNorm,
    ple_scale: f64,
    embedding_projection: Linear,
    sliding_rotary: RotaryTable,
    full_rotary: RotaryTable,
    cfg: Config,
    device: Device,
    dtype: DType,
}

impl Gemma2Model {
    fn new(cfg: Config, vb: VarBuilder, positions: usize) -> Result<Self> {
        let dtype = vb.dtype();
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        let layers = (0..cfg.num_hidden_layers)
            .map(|index| Layer::new(&cfg, index, vb.pp(format!("layers.{index}"))))
            .collect::<Result<Vec<_>>>()?;
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        let ple = vb.pp("ple");
        let ple_projection = linear_no_bias(
            cfg.hidden_size,
            cfg.num_hidden_layers * cfg.hidden_size_per_layer_input,
            ple.pp("per_layer_model_projection"),
        )?;
        let ple_norm = RmsNorm::new(
            cfg.hidden_size_per_layer_input,
            cfg.rms_norm_eps,
            ple.pp("per_layer_projection_norm"),
        )?;
        let embedding_projection = linear_no_bias(
            cfg.hidden_size,
            cfg.embedding_dim,
            vb.pp("embedding_projection"),
        )?;
        let table = |kind: LayerKind, rope: &RopeParameters| -> Result<RotaryTable> {
            let head_dim = (0..cfg.num_hidden_layers)
                .map(|index| cfg.layer(index))
                .find(|shape| shape.kind == kind)
                .map_or(cfg.head_dim, |shape| shape.head_dim);
            RotaryTable::new(head_dim, rope.rope_theta, positions, vb.device(), dtype)
        };
        let sliding_rotary = table(LayerKind::Sliding, &cfg.rope_parameters.sliding_attention)?;
        let full_rotary = table(LayerKind::Full, &cfg.rope_parameters.full_attention)?;
        // The reference multiplies by the scale cast to the weight dtype, so in
        // bf16 sqrt(512) = 22.627 rounds to 22.625.
        let exact = (cfg.hidden_size as f64).sqrt();
        let embed_scale = if dtype == DType::BF16 {
            f64::from(
                Tensor::new(exact as f32, &Device::Cpu)?
                    .to_dtype(DType::BF16)?
                    .to_dtype(DType::F32)?
                    .to_scalar::<f32>()?,
            )
        } else {
            exact
        };
        Ok(Self {
            embed_tokens,
            embed_scale,
            layers,
            norm,
            ple_projection,
            ple_norm,
            ple_scale: (cfg.hidden_size as f64).powf(-0.5),
            embedding_projection,
            sliding_rotary,
            full_rotary,
            device: vb.device().clone(),
            dtype,
            cfg,
        })
    }

    /// The unit-length embedding of every row, `[batch, embedding_dim]`.
    /// `lens` are the real token counts; rows are right padded to `seq`.
    fn forward_pooled(&self, input_ids: &Tensor, lens: &[usize]) -> Result<Tensor> {
        let (batch, seq) = input_ids.dims2()?;
        if lens.len() != batch {
            candle_core::bail!("expected {batch} row lengths, got {}", lens.len());
        }
        let layers = self.cfg.num_hidden_layers;
        let embeds = (self.embed_tokens.forward(input_ids)? * self.embed_scale)?;
        let per_layer = (embeds.apply(&self.ple_projection)? * self.ple_scale)?
            .reshape((batch, seq, layers, self.cfg.hidden_size_per_layer_input))?
            .apply(&self.ple_norm)?;

        let full_mask = attention_mask(lens, seq, None, &self.device)?;
        let sliding_mask = attention_mask(lens, seq, Some(self.cfg.sliding_window), &self.device)?;
        let sliding_rows = self.sliding_rotary.rows(seq)?;
        let full_rows = self.full_rotary.rows(seq)?;

        let mut hidden = embeds;
        for (index, layer) in self.layers.iter().enumerate() {
            let per_layer_input = per_layer.narrow(2, index, 1)?.squeeze(2)?;
            let ((cos, sin), mask) = match layer.kind {
                LayerKind::Sliding => (&sliding_rows, sliding_mask.as_ref()),
                LayerKind::Full => (&full_rows, full_mask.as_ref()),
            };
            hidden = layer.forward(&hidden, &per_layer_input, (cos, sin), mask)?;
        }
        let hidden = hidden.apply(&self.norm)?;

        // Mean over each row's real tokens, as one batched matmul.
        let mut weights = vec![0f32; batch * seq];
        for (row, &len) in lens.iter().enumerate() {
            let share = 1.0 / len.max(1) as f32;
            weights[row * seq..row * seq + len.min(seq)].fill(share);
        }
        let weights =
            Tensor::from_vec(weights, (batch, 1, seq), &self.device)?.to_dtype(self.dtype)?;
        let pooled = weights.matmul(&hidden)?.squeeze(1)?;
        let projected = pooled
            .apply(&self.embedding_projection)?
            .to_dtype(DType::F32)?;
        let norm = (projected.sqr()?.sum_keepdim(1)? + 1e-12)?.sqrt()?;
        projected.broadcast_div(&norm)
    }
}

pub struct Gemma2Embedder {
    model: Gemma2Model,
    tokenizer: Tokenizer,
}

impl Gemma2Embedder {
    pub fn from_hf(options: &LoadOptions<'_>) -> MinSyncResult<Self> {
        let load_error = |what: &str, error: &dyn std::fmt::Display| {
            MinSyncError::Embedding(format!("native model {what}: {error}"))
        };
        match options.dtype {
            DType::F32 | DType::BF16 => {}
            other => {
                return Err(MinSyncError::Embedding(format!(
                    "EmbeddingGemma 2 overflows float16 and returns NaN: use dtype f32 or bf16, \
                     not {}",
                    other.as_str()
                )))
            }
        }
        if options.dtype == DType::BF16 && options.device.is_cpu() {
            return Err(MinSyncError::Embedding(
                "EmbeddingGemma 2 in bf16 needs a GPU: the CPU matmul has no bf16 kernel; \
                 use dtype f32 on the CPU"
                    .to_string(),
            ));
        }
        if !matches!(
            options.attention,
            AttentionRequest::Auto | AttentionRequest::Eager
        ) {
            return Err(MinSyncError::Embedding(
                "EmbeddingGemma 2 supports only attention = \"auto\" or \"eager\"".to_string(),
            ));
        }
        let source = ModelSource::resolve(options.repo_id, options.default_cache_dir.clone())?;
        let config_path = source.fetch("config.json")?;
        let config = Config::parse(&std::fs::read(config_path)?)?;
        let weights = source.weight_files()?;
        // SAFETY: the safetensors files are not modified while the model is loaded.
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&weights, options.dtype, options.device) }
                .map_err(|e| load_error("weight load failed", &e))?;
        let model = Gemma2Model::new(config, vb.pp("language_model"), options.max_length)
            .map_err(|e| load_error("build failed", &e))?;
        let tokenizer = Tokenizer::from_file(source.fetch("tokenizer.json")?)
            .map_err(|e| load_error("tokenizer parse failed", &e))?;
        Self::new(model, tokenizer, options.max_length)
    }

    fn new(model: Gemma2Model, mut tokenizer: Tokenizer, max_length: usize) -> MinSyncResult<Self> {
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            direction: PaddingDirection::Right,
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
        AttentionBackend::Eager
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
        let mut lens = Vec::with_capacity(batch);
        for encoding in &encodings {
            ids.extend_from_slice(encoding.get_ids());
            lens.push(
                encoding
                    .get_attention_mask()
                    .iter()
                    .filter(|&&mask| mask == 1)
                    .count(),
            );
        }
        let input_ids = Tensor::from_vec(ids, (batch, seq), &self.model.device)?;
        self.model
            .forward_pooled(&input_ids, &lens)?
            .to_vec2::<f32>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    fn fixture_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_embeddinggemma2")
    }

    #[derive(Deserialize)]
    struct Case {
        max_length: usize,
        batched: Vec<Vec<f32>>,
        alone: Vec<Vec<f32>>,
    }

    #[derive(Deserialize)]
    struct Golden {
        dim: usize,
        texts: Vec<String>,
        cases: Vec<Case>,
    }

    fn golden() -> Golden {
        serde_json::from_slice(&std::fs::read(fixture_dir().join("golden.json")).unwrap()).unwrap()
    }

    fn embedder(device: &Device, dtype: DType, max_length: usize) -> Gemma2Embedder {
        Gemma2Embedder::from_hf(&LoadOptions {
            repo_id: fixture_dir().to_str().unwrap(),
            default_cache_dir: std::env::temp_dir(),
            device,
            dtype,
            max_length,
            attention: AttentionRequest::Auto,
        })
        .expect("tiny checkpoint loads")
    }

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (norm(a) * norm(b))
    }

    #[test]
    fn matches_the_reference_implementation_batched_and_alone() {
        let golden = golden();
        for case in &golden.cases {
            let model = embedder(&Device::Cpu, DType::F32, case.max_length);
            let batched = model.embed(&golden.texts).unwrap();
            assert_eq!(batched.len(), golden.texts.len());
            for (row, expected) in case.batched.iter().enumerate() {
                assert_eq!(batched[row].len(), golden.dim);
                let diff = max_diff(&batched[row], expected);
                assert!(
                    diff < 2e-5,
                    "max_length {} batched row {row} differs by {diff}",
                    case.max_length
                );
            }
            for (row, text) in golden.texts.iter().enumerate() {
                let alone = model.embed(&[text.as_str()]).unwrap().remove(0);
                let diff = max_diff(&alone, &case.alone[row]);
                assert!(
                    diff < 2e-5,
                    "max_length {} alone row {row} differs by {diff}",
                    case.max_length
                );
            }
        }
    }

    #[test]
    fn embeddings_are_unit_length_and_independent_of_batch_padding() {
        let model = embedder(&Device::Cpu, DType::F32, 64);
        let long: String = (0..30).map(|i| format!("w{i} ")).collect();
        let alone = model.embed(&["w3 w4 w5"]).unwrap().remove(0);
        let norm: f32 = alone.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm {norm}");
        let batched = model.embed(&[long.as_str(), "w3 w4 w5"]).unwrap().remove(1);
        assert!(
            cosine(&alone, &batched) > 0.99999,
            "padding changed the embedding"
        );
    }

    #[test]
    fn empty_input_yields_no_embeddings() {
        let model = embedder(&Device::Cpu, DType::F32, 64);
        assert!(model.embed::<&str>(&[]).unwrap().is_empty());
    }

    #[test]
    fn mask_blocks_padding_and_the_window_and_is_absent_when_nothing_is_masked() {
        let device = Device::Cpu;
        assert!(attention_mask(&[4, 4], 4, None, &device).unwrap().is_none());
        assert!(attention_mask(&[4], 4, Some(3), &device).unwrap().is_none());
        let mask = attention_mask(&[2, 4], 4, Some(1), &device)
            .unwrap()
            .unwrap();
        let rows = mask.reshape((2, 4, 4)).unwrap().to_vec3::<f32>().unwrap();
        let allowed = |row: usize, query: usize, key: usize| rows[row][query][key] == 0.0;
        // Row 0 has two real tokens: keys 2 and 3 are padding.
        assert!(allowed(0, 0, 1) && !allowed(0, 0, 2) && !allowed(0, 1, 3));
        // Row 1 is full length: only the radius-1 window applies.
        assert!(allowed(1, 2, 1) && allowed(1, 2, 3) && !allowed(1, 0, 2) && !allowed(1, 3, 1));
    }

    #[test]
    fn half_precision_float_is_rejected() {
        let result = Gemma2Embedder::from_hf(&LoadOptions {
            repo_id: fixture_dir().to_str().unwrap(),
            default_cache_dir: std::env::temp_dir(),
            device: &Device::Cpu,
            dtype: DType::F16,
            max_length: 64,
            attention: AttentionRequest::Auto,
        });
        let message = result.err().expect("f16 must be refused").to_string();
        assert!(message.contains("float16"), "{message}");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_bf16_stays_close_to_f32() {
        let Ok(device) = Device::new_metal(0) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let golden = golden();
        let reference = embedder(&Device::Cpu, DType::F32, 64)
            .embed(&golden.texts)
            .unwrap();
        let low = embedder(&device, DType::BF16, 64)
            .embed(&golden.texts)
            .unwrap();
        for (row, (a, b)) in reference.iter().zip(&low).enumerate() {
            assert!(cosine(a, b) > 0.98, "row {row}: cosine {}", cosine(a, b));
        }
    }

    #[test]
    fn bf16_on_cpu_is_rejected() {
        let result = Gemma2Embedder::from_hf(&LoadOptions {
            repo_id: fixture_dir().to_str().unwrap(),
            default_cache_dir: std::env::temp_dir(),
            device: &Device::Cpu,
            dtype: DType::BF16,
            max_length: 64,
            attention: AttentionRequest::Auto,
        });
        let message = result
            .err()
            .expect("bf16 on CPU must be refused")
            .to_string();
        assert!(message.contains("CPU"), "{message}");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_f32_matches_the_reference() {
        let Ok(device) = Device::new_metal(0) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let golden = golden();
        let case = &golden.cases[0];
        let vectors = embedder(&device, DType::F32, case.max_length)
            .embed(&golden.texts)
            .unwrap();
        for (row, expected) in case.batched.iter().enumerate() {
            let similarity = cosine(&vectors[row], expected);
            assert!(similarity > 0.99999, "row {row}: cosine {similarity}");
        }
    }

    #[test]
    fn config_parses_the_checkpoint_layout_and_rejects_others() {
        let bytes = std::fs::read(fixture_dir().join("config.json")).unwrap();
        let config = Config::parse(&bytes).unwrap();
        assert_eq!(config.embedding_dim(), 24);
        // Layers 2 and 5 are the wide full-attention ones.
        assert_eq!(config.layer(0).head_dim, 16);
        assert_eq!(config.layer(2).head_dim, 32);
        assert_eq!(config.layer(2).kv_heads, 1);
        assert_eq!(config.layer(5).kind, LayerKind::Full);
        assert!(Config::parse(b"{}").is_err());
        let wrong = String::from_utf8(bytes)
            .unwrap()
            .replace(MODEL_TYPE, "qwen3");
        assert!(Config::parse(wrong.as_bytes()).is_err());
    }
}
