//! ModernBERT masked-LM + SPLADE pooling, in candle, for the in-process
//! sparse encoder.
//!
//! The network is ModernBERT as shipped in `nlpai-lab/KURE-Sparse-nano`
//! (`ModernBertForMaskedLM`, bidirectional attention, alternating global and
//! 128-token sliding-window layers, GeGLU MLP, tied decoder). The pooling is
//! KURE's `ShiftedTopKSpladePooling`: per position
//! `log(1 + relu(logit - shift))`, prefix tokens (`[Q] ` / `[D] `) masked out
//! of both the positions and the vocabulary, max over positions, then a max
//! fold of the vocabulary onto canonical terms. Weight names and numerics
//! follow the Hugging Face reference; the golden test reproduces its output.
//!
//! The implementation is written against this checkpoint family rather than
//! reusing `candle_transformers::models::modernbert`, which is not a
//! dependency of this crate, applies no padding mask to the sliding window
//! of padded keys, and has no MLM-logit chunking.

use crate::error::{MinSyncError, Result};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{LayerNorm, Linear, Module, VarBuilder};
use serde::Deserialize;

/// `config.json` of a ModernBERT checkpoint (the fields the forward pass uses).
#[derive(Debug, Clone, Deserialize)]
pub struct ModernBertConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    /// What the Hugging Face model reads. Older exports only have
    /// `layer_norm_eps`; the shipped KURE config has both.
    #[serde(default)]
    norm_eps: Option<f64>,
    #[serde(default)]
    layer_norm_eps: Option<f64>,
    #[serde(default = "default_local_attention")]
    pub local_attention: usize,
    #[serde(default = "default_global_every")]
    pub global_attn_every_n_layers: usize,
    #[serde(default = "default_max_positions")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,
    #[serde(default)]
    pub global_rope_theta: Option<f64>,
    #[serde(default)]
    pub local_rope_theta: Option<f64>,
    #[serde(default = "default_true")]
    pub decoder_bias: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
    #[serde(default)]
    pub norm_bias: bool,
    #[serde(default)]
    pub classifier_bias: bool,
    #[serde(default = "default_gelu")]
    pub hidden_activation: String,
    #[serde(default = "default_gelu")]
    pub classifier_activation: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    pub full_attention: RopeTheta,
    pub sliding_attention: RopeTheta,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeTheta {
    pub rope_theta: f64,
}

fn default_local_attention() -> usize {
    128
}
fn default_global_every() -> usize {
    3
}
fn default_max_positions() -> usize {
    8192
}
fn default_true() -> bool {
    true
}
fn default_gelu() -> String {
    "gelu".to_string()
}

impl ModernBertConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// LayerNorm epsilon: `norm_eps`, else `layer_norm_eps`, else 1e-5.
    pub fn eps(&self) -> f64 {
        self.norm_eps.or(self.layer_norm_eps).unwrap_or(1e-5)
    }

    /// Reject configurations whose weights this implementation would
    /// silently mis-read (it supports the layout the checkpoint family uses).
    pub fn validate(&self) -> Result<()> {
        let unsupported = |what: &str| {
            Err(MinSyncError::Embedding(format!(
                "unsupported ModernBERT config: {what}"
            )))
        };
        if self.num_attention_heads == 0
            || !self.hidden_size.is_multiple_of(self.num_attention_heads)
        {
            return unsupported("hidden_size must be a multiple of num_attention_heads");
        }
        if !self.head_dim().is_multiple_of(2) {
            return unsupported("head_dim must be even");
        }
        if self.attention_bias || self.mlp_bias || self.norm_bias || self.classifier_bias {
            return unsupported("biased attention/mlp/norm/classifier weights");
        }
        if self.hidden_activation != "gelu" || self.classifier_activation != "gelu" {
            return unsupported("activation other than gelu");
        }
        if self.global_attn_every_n_layers == 0 {
            return unsupported("global_attn_every_n_layers must be positive");
        }
        if let Some(types) = &self.layer_types {
            if types.len() != self.num_hidden_layers {
                return unsupported("layer_types length differs from num_hidden_layers");
            }
            for (index, kind) in types.iter().enumerate() {
                let expected = self.layer_is_global(index);
                let global = kind == "full_attention";
                if (kind != "full_attention" && kind != "sliding_attention") || global != expected {
                    return unsupported("layer_types disagrees with global_attn_every_n_layers");
                }
            }
        }
        self.thetas().map(|_| ())
    }

    fn layer_is_global(&self, index: usize) -> bool {
        index.is_multiple_of(self.global_attn_every_n_layers)
    }

    /// `(global, local)` RoPE bases.
    fn thetas(&self) -> Result<(f64, f64)> {
        if let Some(rope) = &self.rope_parameters {
            return Ok((
                rope.full_attention.rope_theta,
                rope.sliding_attention.rope_theta,
            ));
        }
        match (self.global_rope_theta, self.local_rope_theta) {
            (Some(global), Some(local)) => Ok((global, local)),
            _ => Err(MinSyncError::Embedding(
                "unsupported ModernBERT config: no rope_parameters or global/local_rope_theta"
                    .to_string(),
            )),
        }
    }
}

type CResult<T> = candle_core::Result<T>;

/// LayerNorm without bias. candle's generic path is about ten small kernels
/// per call (mean, subtract, square, sum, divide, sqrt, ...) and the model
/// has 47 of them; on Metal the fused kernel does it in one launch, with a
/// zero bias standing in for the absent one. Other devices keep candle's
/// reference implementation, whose numerics the golden tests pin.
struct Norm {
    reference: LayerNorm,
    fused: Option<(Tensor, f32)>,
}

impl Norm {
    fn new(weight: Tensor, eps: f64) -> CResult<Self> {
        let fused = if weight.device().is_metal() {
            Some((Tensor::zeros_like(&weight)?, eps as f32))
        } else {
            None
        };
        Ok(Self {
            reference: LayerNorm::new_no_bias(weight, eps),
            fused,
        })
    }

    fn load(size: usize, eps: f64, vb: VarBuilder) -> CResult<Self> {
        Self::new(vb.get(size, "weight")?, eps)
    }

    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        match &self.fused {
            Some((zero_bias, eps)) => candle_nn::ops::layer_norm(
                &x.contiguous()?,
                self.reference.weight(),
                zero_bias,
                *eps,
            ),
            None => self.reference.forward(x),
        }
    }
}

/// Query rows per attention block. Scores are materialized as
/// `batch x heads x QUERY_TILE x seq`, never `seq x seq`, so an 8192-token
/// input costs ~100 MB of scratch instead of ~3 GB.
const QUERY_TILE: usize = 256;

/// Tokens (rows x positions) the decoder projection and the pooling handle at
/// once. The projection is `tokens x vocab` f32, 200 KB per token for this
/// vocabulary, and pooling holds several temporaries of that size, so this
/// number alone sets the scratch memory of the encoder, whatever the batch.
///
/// Measured on KURE-Sparse-nano with a full sync: 128 tokens encode as fast as
/// 1024 (SciFact 15.6 vs 15.2-15.7 chunks/s, Korean 31.7-35.7 vs 32.8) and cut
/// the Metal peak from 2.2 GB to 1.56 GB and the CPU peak from 3.4 GB to 1.0 GB.
const LOGIT_TOKENS: usize = 128;

/// Rows and positions of one decoder/pooling block: at most `chunk` positions
/// (the pooling module's `chunk_size`, the whole sequence when unset) and at
/// most `cap` tokens (rows x positions) in total; never below 1 x 1.
pub(crate) fn logit_shape(seq: usize, chunk: Option<usize>, cap: usize) -> (usize, usize) {
    let cap = cap.max(1);
    let positions = chunk
        .filter(|&size| size > 0)
        .unwrap_or(seq)
        .min(seq)
        .min(cap)
        .max(1);
    (cap / positions, positions)
}

struct Rotary {
    cos: Tensor,
    sin: Tensor,
}

impl Rotary {
    /// `cos` / `sin` of every position below `positions`, `[positions, head_dim / 2]`.
    /// Computed in f64 then stored in the model dtype.
    fn new(
        head_dim: usize,
        theta: f64,
        positions: usize,
        device: &Device,
        dtype: DType,
    ) -> CResult<Self> {
        let half = head_dim / 2;
        let inv_freq: Vec<f64> = (0..half)
            .map(|i| 1.0 / theta.powf((2 * i) as f64 / head_dim as f64))
            .collect();
        let mut cos = Vec::with_capacity(positions * half);
        let mut sin = Vec::with_capacity(positions * half);
        for position in 0..positions {
            for freq in &inv_freq {
                let angle = position as f64 * freq;
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }
        let table =
            |values: Vec<f32>| Tensor::from_vec(values, (positions, half), device)?.to_dtype(dtype);
        Ok(Self {
            cos: table(cos)?,
            sin: table(sin)?,
        })
    }

    /// Rotate `x: [batch, heads, seq, head_dim]` (half-split pairing, as HF).
    fn apply(&self, x: &Tensor) -> CResult<Tensor> {
        let seq = x.dim(2)?;
        let cos = self.cos.narrow(0, 0, seq)?;
        let sin = self.sin.narrow(0, 0, seq)?;
        candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin)
    }
}

struct Attention {
    wqkv: Linear,
    wo: Linear,
    heads: usize,
    head_dim: usize,
}

impl Attention {
    fn load(cfg: &ModernBertConfig, vb: VarBuilder) -> CResult<Self> {
        Ok(Self {
            wqkv: candle_nn::linear_no_bias(cfg.hidden_size, 3 * cfg.hidden_size, vb.pp("Wqkv"))?,
            wo: candle_nn::linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("Wo"))?,
            heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim(),
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        rotary: &Rotary,
        mask: &Tensor,
        backend: AttentionBackend,
        tile: usize,
    ) -> CResult<Tensor> {
        let (batch, seq, _) = hidden.dims3()?;
        let qkv = self
            .wqkv
            .forward(hidden)?
            .reshape((batch, seq, 3, self.heads, self.head_dim))?;
        let part = |index: usize| -> CResult<Tensor> {
            qkv.i((.., .., index))?.transpose(1, 2)?.contiguous()
        };
        let (q, k, v) = (part(0)?, part(1)?, part(2)?);
        let (q, k) = (rotary.apply(&q)?, rotary.apply(&k)?);
        let scale = (self.head_dim as f32).powf(-0.5);
        let out = attend(backend, &q, &k, &v, mask, scale, tile)?;
        self.wo.forward(
            &out.transpose(1, 2)?
                .reshape((batch, seq, self.heads * self.head_dim))?,
        )
    }
}

/// How attention scores are computed. Both compute the same function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttentionBackend {
    /// candle's fused scaled-dot-product kernel (Metal): one launch per layer
    /// instead of about five per query tile, and the score matrix is never
    /// materialized.
    Sdpa,
    /// Matmul / mask / softmax / matmul over blocks of query rows, bounded
    /// scratch on every device.
    Tiled,
}

impl AttentionBackend {
    /// Head sizes the Metal SDPA kernel is instantiated for.
    const SDPA_HEAD_DIMS: [usize; 8] = [32, 64, 72, 80, 96, 128, 256, 512];

    /// The fused kernel where `device` has one that supports `head_dim`;
    /// otherwise tiled. A single query row takes the kernel's vector path,
    /// which does not read a mask, so it stays tiled.
    fn resolve(device: &Device, head_dim: usize, seq: usize) -> Self {
        if device.is_metal() && seq > 1 && Self::SDPA_HEAD_DIMS.contains(&head_dim) {
            Self::Sdpa
        } else {
            Self::Tiled
        }
    }
}

/// Attention over `q`, `k`, `v` (`[batch, heads, seq, head_dim]`) with the
/// additive `mask` (`[batch, 1, seq, seq]`, model dtype); `scale` multiplies
/// the scores before the mask.
fn attend(
    backend: AttentionBackend,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: &Tensor,
    scale: f32,
    tile: usize,
) -> CResult<Tensor> {
    let (batch, heads, seq, _) = q.dims4()?;
    match backend {
        AttentionBackend::Sdpa => {
            let mask = mask.expand((batch, heads, seq, seq))?;
            candle_nn::ops::sdpa(q, k, v, Some(&mask), false, scale, 1.0)
        }
        AttentionBackend::Tiled => {
            let q = (q * f64::from(scale))?;
            let k_t = k.transpose(2, 3)?.contiguous()?;
            let mut blocks = Vec::with_capacity(seq.div_ceil(tile));
            let mut start = 0;
            while start < seq {
                let rows = tile.min(seq - start);
                let scores = q.narrow(2, start, rows)?.matmul(&k_t)?;
                let scores = scores.broadcast_add(&mask.narrow(2, start, rows)?)?;
                blocks.push(candle_nn::ops::softmax_last_dim(&scores)?.matmul(v)?);
                start += rows;
            }
            if blocks.len() == 1 {
                Ok(blocks.swap_remove(0))
            } else {
                Tensor::cat(&blocks, 2)
            }
        }
    }
}

struct Mlp {
    wi: Linear,
    wo: Linear,
    intermediate: usize,
}

impl Mlp {
    fn load(cfg: &ModernBertConfig, vb: VarBuilder) -> CResult<Self> {
        Ok(Self {
            wi: candle_nn::linear_no_bias(cfg.hidden_size, 2 * cfg.intermediate_size, vb.pp("Wi"))?,
            wo: candle_nn::linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("Wo"))?,
            intermediate: cfg.intermediate_size,
        })
    }

    /// GeGLU: `gelu(first half) * second half`.
    fn forward(&self, hidden: &Tensor) -> CResult<Tensor> {
        let projected = self.wi.forward(hidden)?;
        let input = projected.narrow(D::Minus1, 0, self.intermediate)?;
        let gate = projected.narrow(D::Minus1, self.intermediate, self.intermediate)?;
        self.wo.forward(&(input.gelu_erf()? * gate)?)
    }
}

struct Layer {
    attn_norm: Option<Norm>,
    attn: Attention,
    mlp_norm: Norm,
    mlp: Mlp,
    global: bool,
}

pub struct ModernBertMlm {
    tok_embeddings: candle_nn::Embedding,
    embedding_norm: Norm,
    layers: Vec<Layer>,
    final_norm: Norm,
    head_dense: Linear,
    head_norm: Norm,
    decoder: Linear,
    global_rotary: Rotary,
    local_rotary: Rotary,
    half_window: usize,
    positions: usize,
    /// Query rows per attention block ([`QUERY_TILE`]); only the tiled backend reads it.
    tile: usize,
    head_dim: usize,
    /// Tokens per decoder/pooling block ([`LOGIT_TOKENS`]).
    logit_tokens: usize,
    device: Device,
    dtype: DType,
    vocab_size: usize,
}

impl ModernBertMlm {
    /// Build from checkpoint tensors (`model.layers.N.attn.Wqkv.weight`, ...).
    /// `positions` bounds the sequence length the rotary tables cover.
    pub fn load(cfg: &ModernBertConfig, vb: VarBuilder, positions: usize) -> Result<Self> {
        cfg.validate()?;
        Self::build(cfg, vb, positions).map_err(|error| {
            MinSyncError::Embedding(format!("native sparse model build failed: {error}"))
        })
    }

    fn build(cfg: &ModernBertConfig, vb: VarBuilder, positions: usize) -> CResult<Self> {
        let hidden = cfg.hidden_size;
        let model = vb.pp("model");
        let embeddings = model.pp("embeddings");
        let tok_embeddings =
            candle_nn::embedding(cfg.vocab_size, hidden, embeddings.pp("tok_embeddings"))?;
        let embedding_norm = Norm::load(hidden, cfg.eps(), embeddings.pp("norm"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for index in 0..cfg.num_hidden_layers {
            let layer = model.pp(format!("layers.{index}"));
            layers.push(Layer {
                // The first layer's pre-attention norm is the identity (absent from the checkpoint).
                attn_norm: if index == 0 {
                    None
                } else {
                    Some(Norm::load(hidden, cfg.eps(), layer.pp("attn_norm"))?)
                },
                attn: Attention::load(cfg, layer.pp("attn"))?,
                mlp_norm: Norm::load(hidden, cfg.eps(), layer.pp("mlp_norm"))?,
                mlp: Mlp::load(cfg, layer.pp("mlp"))?,
                global: cfg.layer_is_global(index),
            });
        }
        let final_norm = Norm::load(hidden, cfg.eps(), model.pp("final_norm"))?;
        let head = vb.pp("head");
        let head_dense = candle_nn::linear_no_bias(hidden, hidden, head.pp("dense"))?;
        let head_norm = Norm::load(hidden, cfg.eps(), head.pp("norm"))?;
        // The decoder is tied to the token embeddings; only its bias is stored.
        let decoder_bias = if cfg.decoder_bias {
            Some(vb.get(cfg.vocab_size, "decoder.bias")?)
        } else {
            None
        };
        let decoder = Linear::new(tok_embeddings.embeddings().clone(), decoder_bias);
        let (global_theta, local_theta) = cfg
            .thetas()
            .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        let positions = positions.max(1);
        Ok(Self {
            device: vb.device().clone(),
            dtype: vb.dtype(),
            tok_embeddings,
            embedding_norm,
            layers,
            final_norm,
            head_dense,
            head_norm,
            decoder,
            global_rotary: Rotary::new(
                cfg.head_dim(),
                global_theta,
                positions,
                vb.device(),
                vb.dtype(),
            )?,
            local_rotary: Rotary::new(
                cfg.head_dim(),
                local_theta,
                positions,
                vb.device(),
                vb.dtype(),
            )?,
            half_window: cfg.local_attention / 2,
            positions,
            tile: QUERY_TILE,
            head_dim: cfg.head_dim(),
            logit_tokens: LOGIT_TOKENS,
            vocab_size: cfg.vocab_size,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Query rows per attention block; tests shrink it to exercise tiling.
    #[cfg(test)]
    pub(crate) fn with_tile(mut self, tile: usize) -> Self {
        self.tile = tile.max(1);
        self
    }

    /// Tokens per decoder/pooling block; tests shrink it to exercise blocking.
    #[cfg(test)]
    pub(crate) fn with_logit_tokens(mut self, tokens: usize) -> Self {
        self.logit_tokens = tokens.max(1);
        self
    }

    pub fn positions(&self) -> usize {
        self.positions
    }

    /// Attention kernel this model runs for any batch longer than one token:
    /// `"sdpa"` (fused, Metal) or `"tiled"`.
    pub fn attention_name(&self) -> &'static str {
        match AttentionBackend::resolve(&self.device, self.head_dim, 2) {
            AttentionBackend::Sdpa => "sdpa",
            AttentionBackend::Tiled => "tiled",
        }
    }

    /// Additive masks `[batch, 1, seq, seq]`: `(global, sliding)`. A key that
    /// is padding is masked in both; the sliding mask also masks keys more
    /// than `half_window` positions from the query.
    fn masks(&self, lengths: &[usize], seq: usize) -> CResult<(Tensor, Tensor)> {
        const NEG: f32 = -1e4;
        let batch = lengths.len();
        let mut global = vec![0f32; batch * seq * seq];
        let mut local = vec![0f32; batch * seq * seq];
        for (row, &length) in lengths.iter().enumerate() {
            let plane = row * seq * seq;
            for query in 0..seq {
                for key in 0..seq {
                    let at = plane + query * seq + key;
                    if key >= length {
                        global[at] = NEG;
                        local[at] = NEG;
                    } else if key.abs_diff(query) > self.half_window {
                        local[at] = NEG;
                    }
                }
            }
        }
        let build = |values: Vec<f32>| {
            Tensor::from_vec(values, (batch, 1, seq, seq), &self.device)?.to_dtype(self.dtype)
        };
        Ok((build(global)?, build(local)?))
    }

    /// Final hidden state through the MLM head, `[batch, seq, hidden]`, for
    /// right-padded `ids` (`[batch, seq]`, u32) with real lengths `lengths`.
    fn head_states(&self, ids: &Tensor, lengths: &[usize]) -> CResult<Tensor> {
        let (batch, seq) = ids.dims2()?;
        if lengths.len() != batch {
            candle_core::bail!("expected {batch} lengths, got {}", lengths.len());
        }
        if seq > self.positions {
            candle_core::bail!(
                "sequence of {seq} tokens exceeds the {} positions this model was loaded for",
                self.positions
            );
        }
        let backend = AttentionBackend::resolve(&self.device, self.head_dim, seq);
        let (global_mask, local_mask) = self.masks(lengths, seq)?;
        let mut hidden = self
            .embedding_norm
            .forward(&self.tok_embeddings.forward(ids)?)?;
        for layer in &self.layers {
            let normed = match &layer.attn_norm {
                Some(norm) => norm.forward(&hidden)?,
                None => hidden.clone(),
            };
            let (rotary, mask) = if layer.global {
                (&self.global_rotary, &global_mask)
            } else {
                (&self.local_rotary, &local_mask)
            };
            hidden = (&hidden
                + layer
                    .attn
                    .forward(&normed, rotary, mask, backend, self.tile)?)?;
            hidden = (&hidden + layer.mlp.forward(&layer.mlp_norm.forward(&hidden)?)?)?;
        }
        let hidden = self.final_norm.forward(&hidden)?;
        self.head_norm
            .forward(&self.head_dense.forward(&hidden)?.gelu_erf()?)
    }

    /// Full MLM logits `[batch, seq, vocab]`. Materializes the whole tensor
    /// (`batch x seq x vocab`), so it is for tests and short inputs; the
    /// encoder uses [`Self::splade`], which never holds more than one chunk.
    pub fn logits(&self, ids: &Tensor, lengths: &[usize]) -> Result<Tensor> {
        self.head_states(ids, lengths)
            .and_then(|states| self.decoder.forward(&states))
            .map_err(|error| {
                MinSyncError::Embedding(format!("native sparse forward failed: {error}"))
            })
    }

    /// SPLADE term weights, one dense `[dimension]` row per input.
    ///
    /// `rows` are the real token ids of each input (the same ones that were
    /// right-padded into `ids`). The decoder projection and the pooling work
    /// on blocks of at most `logit_tokens` (rows x positions) at a time, so
    /// the logit tensors and the temporaries derived from them stay bounded
    /// whatever the batch size or the sequence length. The max over positions
    /// and the per-row result are exact across any blocking.
    pub fn splade(
        &self,
        ids: &Tensor,
        rows: &[Vec<u32>],
        pooling: &SpladePooling,
    ) -> Result<Vec<Vec<f32>>> {
        let lengths: Vec<usize> = rows.iter().map(Vec::len).collect();
        let inner = || -> CResult<Vec<Vec<f32>>> {
            let states = self.head_states(ids, &lengths)?;
            let (batch, seq, _) = states.dims3()?;
            let (block_rows, block_positions) =
                logit_shape(seq, pooling.chunk_size, self.logit_tokens);
            let mask = PoolMask::new(pooling, rows, seq, self.vocab_size, &self.device)?;
            let mut out = Vec::with_capacity(batch);
            let mut first = 0;
            while first < batch {
                let take_rows = block_rows.min(batch - first);
                let block = states.narrow(0, first, take_rows)?;
                let keep = mask.keep.narrow(0, first, take_rows)?;
                let mut pooled =
                    Tensor::zeros((take_rows, self.vocab_size), DType::F32, &self.device)?;
                let mut start = 0;
                while start < seq {
                    let take = block_positions.min(seq - start);
                    let logits = self.decoder.forward(&block.narrow(1, start, take)?)?;
                    pooled = pooling.accumulate(&pooled, &logits, &keep.narrow(1, start, take)?)?;
                    start += take;
                }
                out.extend(pooling.fold(&pooling.finish(&pooled, &mask.open)?)?);
                first += take_rows;
            }
            Ok(out)
        };
        inner().map_err(|error| {
            MinSyncError::Embedding(format!("native sparse forward failed: {error}"))
        })
    }
}

/// Which positions may speak and which vocabulary entries may fire.
struct PoolMask {
    /// `[batch, seq, 1]`: 1 for a real, non-prefix position.
    keep: Tensor,
    /// `[1, vocab]`: 0 for a masked vocabulary entry.
    open: Tensor,
}

impl PoolMask {
    fn new(
        pooling: &SpladePooling,
        rows: &[Vec<u32>],
        seq: usize,
        vocab: usize,
        device: &Device,
    ) -> CResult<Self> {
        let mut keep = vec![0f32; rows.len() * seq];
        for (row, ids) in rows.iter().enumerate() {
            for (position, id) in ids.iter().enumerate() {
                if !pooling.prefix_token_ids.contains(id) {
                    keep[row * seq + position] = 1.0;
                }
            }
        }
        let mut open = vec![1f32; vocab];
        for &id in &pooling.masked_vocab_ids {
            open[id as usize] = 0.0;
        }
        Ok(Self {
            keep: Tensor::from_vec(keep, (rows.len(), seq, 1), device)?,
            open: Tensor::from_vec(open, (1, vocab), device)?,
        })
    }
}

/// KURE's `ShiftedTopKSpladePooling` (`pooling_strategy = "max"`,
/// `activation_function = "relu"`, `top_k` unset), from the module's
/// `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct SpladePooling {
    #[serde(default = "default_max")]
    pub pooling_strategy: String,
    #[serde(default = "default_relu")]
    pub activation_function: String,
    #[serde(default)]
    pub chunk_size: Option<usize>,
    #[serde(default)]
    pub logit_shift: f64,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub prefix_token_ids: Vec<u32>,
    #[serde(default)]
    pub masked_vocab_ids: Vec<u32>,
    #[serde(default)]
    pub canonical_ids: Option<Vec<u32>>,
}

fn default_max() -> String {
    "max".to_string()
}
fn default_relu() -> String {
    "relu".to_string()
}

impl SpladePooling {
    /// Refuse a pooling whose behavior this port does not reproduce.
    pub fn validate(&self, vocab_size: usize) -> Result<()> {
        let unsupported = |what: String| {
            Err(MinSyncError::Embedding(format!(
                "unsupported SPLADE pooling: {what}"
            )))
        };
        if self.pooling_strategy != "max" {
            return unsupported(format!("pooling_strategy {:?}", self.pooling_strategy));
        }
        if self.activation_function != "relu" {
            return unsupported(format!(
                "activation_function {:?}",
                self.activation_function
            ));
        }
        if self.top_k.is_some_and(|k| k > 0) {
            return unsupported("top_k".to_string());
        }
        if let Some(canonical) = &self.canonical_ids {
            if canonical.len() != vocab_size {
                return unsupported(format!(
                    "canonical_ids has {} entries for a {vocab_size}-token vocabulary",
                    canonical.len()
                ));
            }
        }
        if self
            .prefix_token_ids
            .iter()
            .chain(&self.masked_vocab_ids)
            .any(|&id| id as usize >= vocab_size)
        {
            return unsupported("a prefix/masked token id is outside the vocabulary".to_string());
        }
        Ok(())
    }

    /// Number of output terms (`max(canonical_ids) + 1`, or the vocabulary).
    pub fn dimension(&self, vocab_size: usize) -> usize {
        self.canonical_ids
            .as_ref()
            .and_then(|ids| ids.iter().max())
            .map_or(vocab_size, |max| *max as usize + 1)
    }

    /// Fold one block of `logits` (`[rows, take, vocab]`) into the running
    /// per-term max `pooled` (`[rows, vocab]`, f32) of the *shifted* logit
    /// of the positions that may speak. `keep` is the block's `[rows, take,
    /// 1]` slice of the position mask. Only the reduction touches the
    /// per-position tensor: `relu`, `ln(1 + x)` and the vocabulary mask `open`
    /// (0 or 1) are monotone, so they commute with the max over positions and
    /// run once per row in [`Self::finish`]. `max` is exact across any
    /// blocking, and the accumulator starts at zero because a masked position
    /// contributes zero.
    fn accumulate(&self, pooled: &Tensor, logits: &Tensor, keep: &Tensor) -> CResult<Tensor> {
        // The reference multiplies the *shifted* logit by the mask before relu.
        let shifted = (logits.to_dtype(DType::F32)? - self.logit_shift)?.broadcast_mul(keep)?;
        pooled.maximum(&shifted.max(1)?)
    }

    /// Turn the accumulated max of shifted logits into term weights
    /// `ln(1 + relu(max))`, zero for a masked vocabulary entry (`open`:
    /// `[1, vocab]`).
    fn finish(&self, pooled: &Tensor, open: &Tensor) -> CResult<Tensor> {
        (pooled.broadcast_mul(open)?.relu()? + 1.0)?.log()
    }

    /// Fold the vocabulary onto canonical terms (max per term), on the host.
    fn fold(&self, pooled: &Tensor) -> CResult<Vec<Vec<f32>>> {
        let dense = pooled.to_vec2::<f32>()?;
        let Some(canonical) = &self.canonical_ids else {
            return Ok(dense);
        };
        let width = self.dimension(canonical.len());
        Ok(dense
            .into_iter()
            .map(|row| {
                let mut folded = vec![0f32; width];
                for (weight, &term) in row.iter().zip(canonical) {
                    let slot = &mut folded[term as usize];
                    if *weight > *slot {
                        *slot = *weight;
                    }
                }
                folded
            })
            .collect())
    }

    /// Pool already materialized `logits` (`[batch, seq, vocab]`), for tests.
    pub fn pool(
        &self,
        logits: &Tensor,
        rows: &[Vec<u32>],
        vocab_size: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let inner = || -> CResult<Vec<Vec<f32>>> {
            let (batch, seq, _) = logits.dims3()?;
            let mask = PoolMask::new(self, rows, seq, vocab_size, logits.device())?;
            let zeros = Tensor::zeros((batch, vocab_size), DType::F32, logits.device())?;
            self.fold(&self.finish(&self.accumulate(&zeros, logits, &mask.keep)?, &mask.open)?)
        };
        inner().map_err(|error| {
            MinSyncError::Embedding(format!("native sparse pooling failed: {error}"))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::SparseVector;
    use candle_core::safetensors;
    use serde::Deserialize;
    use std::path::{Path, PathBuf};

    pub(crate) fn fixture_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_splade")
    }

    #[derive(Deserialize)]
    struct Golden {
        dim: usize,
        cases: Vec<Case>,
    }

    #[derive(Deserialize)]
    struct Case {
        kind: String,
        max_length: usize,
        batched: Vec<Row>,
        alone: Vec<Row>,
    }

    #[derive(Deserialize)]
    struct Row {
        ids: Vec<u32>,
        indices: Vec<u32>,
        values: Vec<f32>,
    }

    fn load_model(dtype: DType) -> (ModernBertMlm, SpladePooling, ModernBertConfig) {
        let dir = fixture_dir();
        let cfg: ModernBertConfig =
            serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
        let pooling: SpladePooling = serde_json::from_slice(
            &std::fs::read(dir.join("1_ShiftedTopKSpladePooling/config.json")).unwrap(),
        )
        .unwrap();
        let tensors = safetensors::load(dir.join("model.safetensors"), &Device::Cpu).unwrap();
        let vb = VarBuilder::from_tensors(tensors, dtype, &Device::Cpu);
        let model = ModernBertMlm::load(&cfg, vb, 64).unwrap();
        pooling.validate(cfg.vocab_size).unwrap();
        (model, pooling, cfg)
    }

    /// Right-pad `rows` into an id tensor.
    fn pad(model: &ModernBertMlm, rows: &[Vec<u32>]) -> Tensor {
        let seq = rows.iter().map(Vec::len).max().unwrap();
        let mut flat = Vec::new();
        for row in rows {
            flat.extend_from_slice(row);
            flat.extend(std::iter::repeat_n(4u32, seq - row.len()));
        }
        Tensor::from_vec(flat, (rows.len(), seq), model.device()).unwrap()
    }

    fn sparse_rows(dense: Vec<Vec<f32>>) -> Vec<SparseVector> {
        dense
            .into_iter()
            .map(|row| {
                SparseVector::from_pairs(
                    row.iter()
                        .enumerate()
                        .map(|(term, &weight)| (term as u32, weight)),
                )
            })
            .collect()
    }

    /// Right-pad `rows` and encode them with the production path
    /// ([`ModernBertMlm::splade`]: chunked decoder + pooling).
    fn run(
        model: &ModernBertMlm,
        pooling: &SpladePooling,
        _vocab: usize,
        rows: &[Vec<u32>],
    ) -> Vec<SparseVector> {
        sparse_rows(model.splade(&pad(model, rows), rows, pooling).unwrap())
    }

    fn assert_matches(actual: &SparseVector, expected: &Row, tolerance: f32, label: &str) {
        assert_eq!(
            actual.indices(),
            expected.indices.as_slice(),
            "{label}: active terms differ"
        );
        for ((index, got), want) in actual
            .indices()
            .iter()
            .zip(actual.values())
            .zip(&expected.values)
        {
            assert!(
                (got - want).abs() <= tolerance,
                "{label}: term {index} weight {got} vs reference {want}"
            );
        }
    }

    #[test]
    fn reproduces_the_huggingface_reference_batched_and_alone() {
        let (model, pooling, cfg) = load_model(DType::F32);
        let golden: Golden =
            serde_json::from_slice(&std::fs::read(fixture_dir().join("golden.json")).unwrap())
                .unwrap();
        assert_eq!(pooling.dimension(cfg.vocab_size), golden.dim);
        for case in &golden.cases {
            let label = format!("{}/{}", case.kind, case.max_length);
            let batched: Vec<Vec<u32>> = case.batched.iter().map(|row| row.ids.clone()).collect();
            let together = run(&model, &pooling, cfg.vocab_size, &batched);
            for (index, (got, want)) in together.iter().zip(&case.batched).enumerate() {
                assert_matches(got, want, 1e-4, &format!("{label} batched[{index}]"));
            }
            for (index, want) in case.alone.iter().enumerate() {
                let got = run(
                    &model,
                    &pooling,
                    cfg.vocab_size,
                    std::slice::from_ref(&want.ids),
                );
                assert_matches(&got[0], want, 1e-4, &format!("{label} alone[{index}]"));
            }
        }
    }

    #[test]
    fn sliding_window_changes_the_result_for_a_long_input() {
        // Sanity that the fixture exercises the window: the 40-token row is
        // longer than local_attention (8), so a model treating every layer as
        // global must disagree with the reference on it.
        let (model, pooling, cfg) = load_model(DType::F32);
        let golden: Golden =
            serde_json::from_slice(&std::fs::read(fixture_dir().join("golden.json")).unwrap())
                .unwrap();
        let long = &golden.cases[0].alone[1];
        assert!(long.ids.len() > 2 * (cfg.local_attention / 2));
        let reference = run(
            &model,
            &pooling,
            cfg.vocab_size,
            std::slice::from_ref(&long.ids),
        );
        let mut global = cfg.clone();
        global.local_attention = 10_000;
        let tensors =
            safetensors::load(fixture_dir().join("model.safetensors"), &Device::Cpu).unwrap();
        let widened = ModernBertMlm::load(
            &global,
            VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu),
            64,
        )
        .unwrap();
        let widened = run(
            &widened,
            &pooling,
            cfg.vocab_size,
            std::slice::from_ref(&long.ids),
        );
        assert_ne!(
            reference[0].values(),
            widened[0].values(),
            "window must be observable in the fixture"
        );
    }

    #[test]
    fn pooling_is_independent_of_batch_padding() {
        let (model, pooling, cfg) = load_model(DType::F32);
        let golden: Golden =
            serde_json::from_slice(&std::fs::read(fixture_dir().join("golden.json")).unwrap())
                .unwrap();
        let short = golden.cases[0].alone[3].ids.clone();
        let long = golden.cases[0].alone[1].ids.clone();
        let alone = run(
            &model,
            &pooling,
            cfg.vocab_size,
            std::slice::from_ref(&short),
        );
        let padded = run(&model, &pooling, cfg.vocab_size, &[long, short]);
        assert_eq!(alone[0].indices(), padded[1].indices());
        for (a, b) in alone[0].values().iter().zip(padded[1].values()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn attention_tiling_and_logit_chunking_do_not_change_the_result() {
        // Tile of 3 rows and a pooling chunk of 3 positions both split the
        // 40-token row into many blocks; the reference ran it in one piece.
        let (model, pooling, cfg) = load_model(DType::F32);
        let golden: Golden =
            serde_json::from_slice(&std::fs::read(fixture_dir().join("golden.json")).unwrap())
                .unwrap();
        let rows: Vec<Vec<u32>> = golden.cases[0]
            .alone
            .iter()
            .map(|row| row.ids.clone())
            .collect();
        let whole = SpladePooling {
            chunk_size: None,
            ..pooling.clone()
        };
        let reference = run(&model, &whole, cfg.vocab_size, &rows);
        let tiled = load_model(DType::F32).0.with_tile(3);
        for chunk in [Some(1), Some(3), Some(7), Some(10_000)] {
            let chunked = SpladePooling {
                chunk_size: chunk,
                ..pooling.clone()
            };
            let got = run(&tiled, &chunked, cfg.vocab_size, &rows);
            for (index, (a, b)) in got.iter().zip(&reference).enumerate() {
                assert_eq!(a.indices(), b.indices(), "chunk {chunk:?} row {index}");
                for (x, y) in a.values().iter().zip(b.values()) {
                    assert!(
                        (x - y).abs() < 1e-4,
                        "chunk {chunk:?} row {index}: {x} vs {y}"
                    );
                }
            }
        }
    }

    #[test]
    fn rejects_input_longer_than_the_rotary_tables() {
        let (model, pooling, _cfg) = load_model(DType::F32);
        let too_long: Vec<u32> = std::iter::repeat_n(11u32, model.positions() + 1).collect();
        let ids = pad(&model, std::slice::from_ref(&too_long));
        let error = model
            .splade(&ids, std::slice::from_ref(&too_long), &pooling)
            .unwrap_err();
        assert!(error.to_string().contains("positions"), "{error}");
    }
    /// Logits `[1, seq, vocab]` from host rows.
    fn logits_of(rows: &[&[f32]]) -> Tensor {
        let vocab = rows[0].len();
        let flat: Vec<f32> = rows.iter().flat_map(|row| row.iter().copied()).collect();
        Tensor::from_vec(flat, (1, rows.len(), vocab), &Device::Cpu).unwrap()
    }

    fn pooling_with(
        shift: f64,
        prefix: &[u32],
        masked: &[u32],
        canon: Option<Vec<u32>>,
    ) -> SpladePooling {
        SpladePooling {
            pooling_strategy: "max".into(),
            activation_function: "relu".into(),
            chunk_size: None,
            logit_shift: shift,
            top_k: None,
            prefix_token_ids: prefix.to_vec(),
            masked_vocab_ids: masked.to_vec(),
            canonical_ids: canon,
        }
    }

    #[test]
    fn pooling_applies_shift_relu_log1p_and_takes_the_max_over_positions() {
        // vocab of 3, two positions; shift 1. Term 0: max(3-1, 0-1) -> log(3).
        // Term 1 never clears the shift -> absent. Term 2: max(1.5-1, 4-1) -> log(4).
        let pooling = pooling_with(1.0, &[], &[], None);
        let logits = logits_of(&[&[3.0, 0.5, 1.5], &[0.0, 0.9, 4.0]]);
        let dense = pooling.pool(&logits, &[vec![5, 6]], 3).unwrap();
        assert!((dense[0][0] - 3f32.ln()).abs() < 1e-6);
        assert_eq!(dense[0][1], 0.0);
        assert!((dense[0][2] - 4f32.ln()).abs() < 1e-6);
    }

    #[test]
    fn prefix_positions_contribute_nothing_and_masked_terms_never_fire() {
        // Position 0 holds the prefix token (id 9) with huge logits everywhere;
        // term 2 is masked out of the vocabulary. Only position 1 may speak,
        // and only through terms 0 and 1.
        let pooling = pooling_with(0.0, &[9], &[2], None);
        let logits = logits_of(&[&[50.0, 50.0, 50.0], &[1.0, 0.0, 30.0]]);
        let dense = pooling.pool(&logits, &[vec![9, 5]], 3).unwrap();
        assert!((dense[0][0] - 2f32.ln()).abs() < 1e-6, "{dense:?}");
        assert_eq!(dense[0][1], 0.0);
        assert_eq!(dense[0][2], 0.0, "a masked term must stay silent");
    }

    #[test]
    fn positions_beyond_the_real_length_are_ignored() {
        // Row is 1 real token + 1 padding position whose logits are huge.
        let pooling = pooling_with(0.0, &[], &[], None);
        let logits = logits_of(&[&[1.0, 0.0], &[99.0, 99.0]]);
        let dense = pooling.pool(&logits, &[vec![5]], 2).unwrap();
        assert!((dense[0][0] - 2f32.ln()).abs() < 1e-6);
        assert_eq!(dense[0][1], 0.0);
    }

    #[test]
    fn canonical_fold_takes_the_max_weight_per_term() {
        // Tokens 0 and 2 fold onto term 0, token 1 is term 1 (dimension 2).
        let pooling = pooling_with(0.0, &[], &[], Some(vec![0, 1, 0]));
        let logits = logits_of(&[&[1.0, 5.0, 3.0]]);
        let dense = pooling.pool(&logits, &[vec![5]], 3).unwrap();
        assert_eq!(dense[0].len(), 2);
        assert!((dense[0][0] - 4f32.ln()).abs() < 1e-6, "max(log2, log4)");
        assert!((dense[0][1] - 6f32.ln()).abs() < 1e-6);
    }

    #[test]
    fn pooling_validation_refuses_unreproduced_variants() {
        let base = SpladePooling {
            pooling_strategy: "max".into(),
            activation_function: "relu".into(),
            chunk_size: None,
            logit_shift: 0.0,
            top_k: None,
            prefix_token_ids: vec![],
            masked_vocab_ids: vec![],
            canonical_ids: None,
        };
        assert!(base.validate(10).is_ok());
        for broken in [
            SpladePooling {
                pooling_strategy: "sum".into(),
                ..base.clone()
            },
            SpladePooling {
                activation_function: "log1p_relu".into(),
                ..base.clone()
            },
            SpladePooling {
                top_k: Some(32),
                ..base.clone()
            },
            SpladePooling {
                canonical_ids: Some(vec![0; 3]),
                ..base.clone()
            },
            SpladePooling {
                prefix_token_ids: vec![99],
                ..base.clone()
            },
        ] {
            assert!(broken.validate(10).is_err(), "{broken:?}");
        }
    }

    #[test]
    fn parses_the_real_kure_sparse_nano_config() {
        // The shipped config carries both `norm_eps` and `layer_norm_eps`.
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/kure_sparse_nano/config.json");
        let cfg: ModernBertConfig = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        cfg.validate().unwrap();
        assert_eq!(
            (cfg.vocab_size, cfg.hidden_size, cfg.num_hidden_layers),
            (50002, 768, 22)
        );
        assert_eq!(cfg.num_attention_heads, 12);
        assert_eq!(cfg.local_attention, 128);
        assert_eq!(cfg.thetas().unwrap(), (160_000.0, 10_000.0));
        assert_eq!(cfg.eps(), 1e-5);
    }

    #[test]
    fn config_validation_rejects_layouts_the_port_would_misread() {
        let cfg: ModernBertConfig =
            serde_json::from_slice(&std::fs::read(fixture_dir().join("config.json")).unwrap())
                .unwrap();
        assert!(cfg.validate().is_ok());
        let mut biased = cfg.clone();
        biased.attention_bias = true;
        assert!(biased.validate().is_err());
        let mut odd = cfg.clone();
        odd.hidden_size = 30;
        assert!(odd.validate().is_err());
        let mut types = cfg.clone();
        types.layer_types = Some(vec!["sliding_attention".into(); cfg.num_hidden_layers]);
        assert!(types.validate().is_err());
    }

    #[test]
    fn pooling_matches_a_plain_reference_on_random_logits_and_masks() {
        // Ragged rows (padding positions), a prefix token and masked terms,
        // checked term by term against the formula written out in plain Rust:
        // max over real, non-prefix positions of ln(1 + relu(logit - shift)),
        // and nothing at all for a masked term.
        let (batch, seq, vocab) = (3usize, 9usize, 17usize);
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 8.0 - 3.0
        };
        let flat: Vec<f32> = (0..batch * seq * vocab).map(|_| next()).collect();
        let logits = Tensor::from_vec(flat.clone(), (batch, seq, vocab), &Device::Cpu).unwrap();
        let rows: Vec<Vec<u32>> = vec![vec![9, 5, 6, 7, 8, 5, 6, 7, 8], vec![9, 5, 6], vec![5]];
        let masked = [3u32, 11];
        let pooling = pooling_with(1.5, &[9], &masked, None);
        let got = pooling.pool(&logits, &rows, vocab).unwrap();
        for (row, ids) in rows.iter().enumerate() {
            for term in 0..vocab {
                let mut want = 0f32;
                if !masked.contains(&(term as u32)) {
                    for (position, id) in ids.iter().enumerate() {
                        if *id == 9 {
                            continue;
                        }
                        let logit = flat[(row * seq + position) * vocab + term];
                        want = want.max(((logit - 1.5).max(0.0) + 1.0).ln());
                    }
                }
                assert!(
                    (got[row][term] - want).abs() < 1e-5,
                    "row {row} term {term}: {} vs {want}",
                    got[row][term]
                );
            }
        }
    }

    #[test]
    fn logit_blocks_never_exceed_the_token_cap_and_cover_the_batch() {
        for (seq, chunk, cap) in [
            (40, Some(128), 1000),
            (4000, Some(128), 256),
            (4000, None, 256),
            (10, Some(0), 1),
            (1, None, 1),
            (7, Some(3), 100),
            (5, Some(2), 0),
        ] {
            let (rows, positions) = logit_shape(seq, chunk, cap);
            assert!(rows >= 1 && positions >= 1, "{seq} {chunk:?} {cap}");
            assert!(
                positions <= seq.max(1),
                "{seq} {chunk:?} {cap}: {positions}"
            );
            assert!(
                rows * positions <= cap.max(1),
                "{seq} {chunk:?} {cap}: {rows} x {positions}"
            );
        }
        assert_eq!(logit_shape(40, Some(128), 1000), (25, 40));
        assert_eq!(logit_shape(4000, Some(128), 256), (2, 128));
        assert_eq!(logit_shape(4000, None, 256), (1, 256));
    }

    #[test]
    fn row_blocking_of_the_logit_working_set_does_not_change_the_result() {
        // Ragged rows in one batch, then the decoder + pooling cut into blocks of
        // 1, 2, 5, 16 tokens and one block that holds everything.
        let (model, pooling, cfg) = load_model(DType::F32);
        let golden: Golden =
            serde_json::from_slice(&std::fs::read(fixture_dir().join("golden.json")).unwrap())
                .unwrap();
        let rows: Vec<Vec<u32>> = golden.cases[0]
            .alone
            .iter()
            .map(|row| row.ids.clone())
            .collect();
        assert!(rows.len() >= 3, "the fixture needs several rows to block");
        let reference = run(&model, &pooling, cfg.vocab_size, &rows);
        for tokens in [1, 2, 5, 16, 10_000] {
            let blocked = load_model(DType::F32).0.with_logit_tokens(tokens);
            let got = run(&blocked, &pooling, cfg.vocab_size, &rows);
            for (index, (a, b)) in got.iter().zip(&reference).enumerate() {
                assert_eq!(a.indices(), b.indices(), "tokens {tokens} row {index}");
                for (x, y) in a.values().iter().zip(b.values()) {
                    assert!(
                        (x - y).abs() < 1e-5,
                        "tokens {tokens} row {index}: {x} vs {y}"
                    );
                }
            }
        }
    }

    fn lcg(seed: u64) -> impl FnMut() -> f32 {
        let mut state = seed;
        move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
    }

    fn random(shape: &[usize], next: &mut impl FnMut() -> f32, device: &Device) -> Tensor {
        let count: usize = shape.iter().product();
        Tensor::from_vec(
            (0..count).map(|_| next()).collect::<Vec<_>>(),
            shape,
            device,
        )
        .unwrap()
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        (a.to_device(&Device::Cpu).unwrap() - b.to_device(&Device::Cpu).unwrap())
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    #[test]
    fn fused_attention_matches_tiled_attention_with_padding_and_a_sliding_window() {
        // Real head count and head_dim, a sequence longer than the 256-row
        // tile, right padding on the second row and a sliding-window mask.
        let Ok(device) = Device::new_metal(0) else {
            eprintln!("no Metal device: the fused kernel is not available here");
            return;
        };
        let (batch, heads, seq, dim) = (2usize, 12usize, 300usize, 64usize);
        let mut next = lcg(0x9E37_79B9_7F4A_7C15);
        let q = random(&[batch, heads, seq, dim], &mut next, &device);
        let k = random(&[batch, heads, seq, dim], &mut next, &device);
        let v = random(&[batch, heads, seq, dim], &mut next, &device);
        let mut mask = vec![0f32; batch * seq * seq];
        for row in 0..batch {
            let length = if row == 0 { seq } else { 211 };
            for query in 0..seq {
                for key in 0..seq {
                    if key >= length || key.abs_diff(query) > 64 {
                        mask[(row * seq + query) * seq + key] = -1e4;
                    }
                }
            }
        }
        let mask = Tensor::from_vec(mask, (batch, 1, seq, seq), &device).unwrap();
        let scale = (dim as f32).powf(-0.5);
        let tiled = attend(AttentionBackend::Tiled, &q, &k, &v, &mask, scale, 256).unwrap();
        let fused = attend(AttentionBackend::Sdpa, &q, &k, &v, &mask, scale, 256).unwrap();
        assert_eq!(tiled.dims(), fused.dims());
        // Rows at or beyond a row's length are padding; only real rows are read.
        let real = |t: &Tensor| t.narrow(2, 0, 211).unwrap();
        let diff = max_abs_diff(&real(&tiled), &real(&fused));
        assert!(diff < 1e-4, "fused vs tiled attention differ by {diff}");
    }

    #[test]
    fn the_fused_kernel_is_only_chosen_where_it_can_honor_the_mask() {
        let metal = Device::new_metal(0);
        if let Ok(metal) = &metal {
            assert_eq!(
                AttentionBackend::resolve(metal, 64, 128),
                AttentionBackend::Sdpa
            );
            // Head sizes the kernel is not built for.
            assert_eq!(
                AttentionBackend::resolve(metal, 8, 128),
                AttentionBackend::Tiled
            );
            // A single query row takes the kernel's vector path, which has no mask.
            assert_eq!(
                AttentionBackend::resolve(metal, 64, 1),
                AttentionBackend::Tiled
            );
        }
        assert_eq!(
            AttentionBackend::resolve(&Device::Cpu, 64, 128),
            AttentionBackend::Tiled
        );
    }

    #[test]
    fn the_reported_attention_name_follows_the_backend_actually_used() {
        // The tiny fixture has head_dim 8 and runs on CPU: tiled either way.
        assert_eq!(load_model(DType::F32).0.attention_name(), "tiled");
    }

    #[test]
    fn layer_norm_matches_the_textbook_formula_on_every_device() {
        let weight_values: Vec<f32> = (0..24).map(|i| 0.5 + i as f32 * 0.05).collect();
        let mut devices = vec![Device::Cpu];
        if let Ok(metal) = Device::new_metal(0) {
            devices.push(metal);
        }
        let mut next = lcg(0xD1B5_4A32_D192_ED03);
        let input: Vec<f32> = (0..5 * 24).map(|_| next() * 6.0 + 1.0).collect();
        let expected: Vec<f32> = input
            .chunks(24)
            .flat_map(|row| {
                let mean = row.iter().sum::<f32>() / 24.0;
                let var = row.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / 24.0;
                row.iter()
                    .zip(&weight_values)
                    .map(|(x, w)| (x - mean) / (var + 1e-5).sqrt() * w)
                    .collect::<Vec<_>>()
            })
            .collect();
        for device in devices {
            let weight = Tensor::from_vec(weight_values.clone(), 24, &device).unwrap();
            let norm = Norm::new(weight, 1e-5).unwrap();
            let x = Tensor::from_vec(input.clone(), (1, 5, 24), &device).unwrap();
            let got = norm.forward(&x).unwrap();
            let want = Tensor::from_vec(expected.clone(), (1, 5, 24), &Device::Cpu).unwrap();
            let diff = max_abs_diff(&got, &want);
            assert!(
                diff < 1e-4,
                "{:?}: layer norm off by {diff}",
                device.location()
            );
        }
    }
}
