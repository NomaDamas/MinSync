//! Attention backends for the native Qwen3 embedder.
//!
//! The reference ("eager") attention materializes a `[batch, heads, seq, seq]`
//! score tensor and then makes several same-sized copies of it per layer
//! (scale, mask, softmax). That tensor, not the model weights, dominates peak
//! memory once chunks reach a few hundred tokens: one pass of four 2048-token
//! inputs holds roughly 5 GiB of attention temporaries on top of the weights.
//!
//! Every backend computes the same function; they differ in how much of the
//! score matrix exists at once:
//!
//! * [`AttentionBackend::MetalSdpa`] - candle's fused scaled-dot-product
//!   attention kernel. The score matrix is never materialized.
//! * [`AttentionBackend::CpuFlash`] - candle's CPU flash attention, run on one
//!   unpadded sequence at a time. The score matrix is never materialized.
//! * [`AttentionBackend::Tiled`] - eager attention over blocks of
//!   [`QUERY_TILE`] query rows, so scores are bounded by
//!   `batch x heads x QUERY_TILE x seq`. Works on every device.
//! * [`AttentionBackend::Eager`] - the reference implementation.

use crate::error::{MinSyncError, Result as MinSyncResult};
use candle_core::{DType, Device, Result, Tensor, D};

/// Query rows processed per block by [`AttentionBackend::Tiled`].
pub(super) const QUERY_TILE: usize = 256;

/// Additive mask value for a masked key (matches the reference implementation;
/// finite so that fully masked padding rows stay finite after softmax).
const MASK_VALUE: f32 = -1e4;

/// Head sizes the Metal SDPA kernel is instantiated for.
const METAL_SDPA_HEAD_DIMS: [usize; 8] = [32, 64, 72, 80, 96, 128, 256, 512];

/// What the user asked for (`embedder.attention`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AttentionRequest {
    /// Fastest backend the device supports.
    #[default]
    Auto,
    /// Reference implementation (materializes the full score matrix).
    Eager,
    /// Eager attention over query tiles; any device.
    Tiled,
    /// The device's fused kernel; an error when the device has none.
    Fused,
}

impl AttentionRequest {
    pub fn parse(value: Option<&str>) -> MinSyncResult<Self> {
        match value {
            None | Some("auto") => Ok(Self::Auto),
            Some("eager") => Ok(Self::Eager),
            Some("tiled") => Ok(Self::Tiled),
            Some("fused") => Ok(Self::Fused),
            Some(other) => Err(MinSyncError::Config(format!(
                "invalid native attention '{other}': expected auto, eager, tiled, or fused"
            ))),
        }
    }
}

/// The implementation chosen for a loaded model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionBackend {
    MetalSdpa,
    CpuFlash,
    Tiled,
    Eager,
}

impl AttentionBackend {
    /// Pick the backend for `request` on `device`.
    pub fn resolve(
        request: AttentionRequest,
        device: &Device,
        dtype: DType,
        head_dim: usize,
    ) -> MinSyncResult<Self> {
        match request {
            AttentionRequest::Eager => Ok(Self::Eager),
            AttentionRequest::Tiled => Ok(Self::Tiled),
            AttentionRequest::Auto => {
                Ok(Self::fused(device, dtype, head_dim).unwrap_or(Self::Tiled))
            }
            AttentionRequest::Fused => Self::fused(device, dtype, head_dim).map_err(|reason| {
                MinSyncError::Config(format!("attention = \"fused\" is unavailable: {reason}"))
            }),
        }
    }

    fn fused(device: &Device, dtype: DType, head_dim: usize) -> std::result::Result<Self, String> {
        if device.is_metal() {
            if METAL_SDPA_HEAD_DIMS.contains(&head_dim) {
                Ok(Self::MetalSdpa)
            } else {
                Err(format!(
                    "the Metal kernel does not support head_dim {head_dim}"
                ))
            }
        } else if device.is_cpu() {
            if dtype == DType::F32 {
                Ok(Self::CpuFlash)
            } else {
                Err(format!("the CPU kernel requires dtype f32, got {dtype:?}"))
            }
        } else {
            Err(format!(
                "no fused kernel for device {:?}",
                device.location()
            ))
        }
    }

    /// Query rows of the score matrix alive at once for a batch padded to
    /// `padded` tokens; the planner multiplies this by
    /// `rows x heads x dtype_bytes x padded` to bound attention scratch.
    pub fn score_rows(self, padded: usize) -> usize {
        match self {
            Self::Eager => padded,
            Self::Tiled => padded.min(QUERY_TILE),
            Self::MetalSdpa | Self::CpuFlash => 0,
        }
    }

    fn needs_additive_mask(self) -> bool {
        !matches!(self, Self::CpuFlash)
    }
}

/// Padding description of one forward pass.
pub struct BatchMask {
    /// Left-padding length of every row.
    pad_lens: Vec<usize>,
    /// `[batch, 1, seq, seq]` additive causal + padding mask in the model dtype,
    /// only built for backends that read it.
    additive: Option<Tensor>,
}

impl BatchMask {
    pub fn new(
        backend: AttentionBackend,
        pad_lens: Vec<usize>,
        seq_len: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let additive = if backend.needs_additive_mask() {
            Some(additive_mask(&pad_lens, seq_len, device, dtype)?)
        } else {
            None
        };
        Ok(Self { pad_lens, additive })
    }

    fn additive(&self) -> Result<&Tensor> {
        self.additive
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("attention mask was not built".to_string()))
    }
}

/// Causal mask plus a mask for the keys that are left padding. Built on the
/// host in one pass; the reference implementation built the same values with
/// about six full-size device tensors.
fn additive_mask(
    pad_lens: &[usize],
    seq_len: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let batch = pad_lens.len();
    let mut data = vec![0f32; batch * seq_len * seq_len];
    for (row, &pad) in pad_lens.iter().enumerate() {
        let plane = &mut data[row * seq_len * seq_len..(row + 1) * seq_len * seq_len];
        for (query, scores) in plane.chunks_mut(seq_len).enumerate() {
            for (key, slot) in scores.iter_mut().enumerate() {
                let mut value = 0.0;
                if key > query {
                    value += MASK_VALUE;
                }
                if key < pad {
                    value += MASK_VALUE;
                }
                *slot = value;
            }
        }
    }
    Tensor::from_vec(data, (batch, 1, seq_len, seq_len), device)?.to_dtype(dtype)
}

/// Grouped-query attention: `[B, Nkv, T, D]` -> `[B, Nkv * n_rep, T, D]`.
fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (batch, n_kv, seq, dim) = x.dims4()?;
    x.unsqueeze(2)?
        .broadcast_as((batch, n_kv, n_rep, seq, dim))?
        .reshape((batch, n_kv * n_rep, seq, dim))
}

/// Attention over `q: [B, Nh, T, D]` and `k, v: [B, Nkv, T, D]`; returns
/// `[B, Nh, T, D]`. Rows of left padding are unspecified (the caller only reads
/// real tokens).
pub fn attend(
    backend: AttentionBackend,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: &BatchMask,
    scale: f32,
    n_rep: usize,
) -> Result<Tensor> {
    match backend {
        AttentionBackend::Eager => eager(q, k, v, mask.additive()?, scale, n_rep),
        AttentionBackend::Tiled => tiled(q, k, v, mask.additive()?, scale, n_rep, QUERY_TILE),
        AttentionBackend::MetalSdpa => {
            let (batch, heads, seq, _) = q.dims4()?;
            let mask = mask.additive()?.expand((batch, heads, seq, seq))?;
            candle_nn::ops::sdpa(q, k, v, Some(&mask), false, scale, 1.0)
        }
        AttentionBackend::CpuFlash => cpu_flash(q, k, v, &mask.pad_lens, scale),
    }
}

fn eager(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: &Tensor,
    scale: f32,
    n_rep: usize,
) -> Result<Tensor> {
    let k = repeat_kv(k, n_rep)?;
    let v = repeat_kv(v, n_rep)?;
    let scores = (q.matmul(&k.transpose(2, 3)?)? * f64::from(scale))?;
    let scores = scores.broadcast_add(mask)?;
    candle_nn::ops::softmax(&scores, D::Minus1)?.matmul(&v)
}

fn tiled(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: &Tensor,
    scale: f32,
    n_rep: usize,
    tile: usize,
) -> Result<Tensor> {
    let seq = q.dim(2)?;
    let k = repeat_kv(k, n_rep)?;
    let v = repeat_kv(v, n_rep)?;
    let kt = k.transpose(2, 3)?.contiguous()?;
    let q = (q * f64::from(scale))?;
    let mut blocks = Vec::with_capacity(seq.div_ceil(tile));
    let mut start = 0;
    while start < seq {
        let rows = tile.min(seq - start);
        let scores = q.narrow(2, start, rows)?.matmul(&kt)?;
        let scores = scores.broadcast_add(&mask.narrow(2, start, rows)?)?;
        blocks.push(candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v)?);
        start += rows;
    }
    if blocks.len() == 1 {
        return Ok(blocks.swap_remove(0));
    }
    Tensor::cat(&blocks, 2)
}

/// Flash attention is called per sequence on its real tokens only, which also
/// removes the padding mask.
fn cpu_flash(q: &Tensor, k: &Tensor, v: &Tensor, pad_lens: &[usize], scale: f32) -> Result<Tensor> {
    let (batch, heads, seq, dim) = q.dims4()?;
    let mut rows = Vec::with_capacity(batch);
    for (row, &pad) in pad_lens.iter().enumerate() {
        let real = seq - pad;
        let take = |t: &Tensor| -> Result<Tensor> {
            t.narrow(0, row, 1)?
                .narrow(2, pad, real)?
                .transpose(1, 2)?
                .contiguous()
        };
        let out = candle_nn::attention::flash_attn::<f32>(
            &take(q)?,
            &take(k)?,
            &take(v)?,
            scale,
            candle_nn::attention::AttnMask::causal(),
            None,
            None,
        )?;
        rows.push(if pad == 0 {
            out
        } else {
            Tensor::cat(
                &[
                    Tensor::zeros((1, heads, pad, dim), out.dtype(), out.device())?,
                    out,
                ],
                2,
            )?
        });
    }
    Tensor::cat(&rows, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEADS: usize = 4;
    const KV_HEADS: usize = 2;
    const DIM: usize = 32;

    /// Deterministic pseudo-random values in [-0.5, 0.5).
    fn pseudo(len: usize, seed: u32) -> Vec<f32> {
        let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
            })
            .collect()
    }

    fn tensor(shape: (usize, usize, usize, usize), seed: u32, device: &Device) -> Tensor {
        let len = shape.0 * shape.1 * shape.2 * shape.3;
        Tensor::from_vec(pseudo(len, seed), shape, device).expect("tensor")
    }

    struct Case {
        q: Tensor,
        k: Tensor,
        v: Tensor,
        pads: Vec<usize>,
        seq: usize,
    }

    fn case(device: &Device, pads: &[usize], seq: usize) -> Case {
        let batch = pads.len();
        Case {
            q: tensor((batch, HEADS, seq, DIM), 1, device),
            k: tensor((batch, KV_HEADS, seq, DIM), 2, device),
            v: tensor((batch, KV_HEADS, seq, DIM), 3, device),
            pads: pads.to_vec(),
            seq,
        }
    }

    fn scale() -> f32 {
        (DIM as f32).powf(-0.5)
    }

    fn run(backend: AttentionBackend, case: &Case, device: &Device, dtype: DType) -> Tensor {
        let mask =
            BatchMask::new(backend, case.pads.clone(), case.seq, device, dtype).expect("mask");
        let cast = |t: &Tensor| t.to_dtype(dtype).expect("cast");
        attend(
            backend,
            &cast(&case.q),
            &cast(&case.k),
            &cast(&case.v),
            &mask,
            scale(),
            HEADS / KV_HEADS,
        )
        .expect("attend")
        .to_dtype(DType::F32)
        .expect("to f32")
    }

    /// Largest absolute difference over the rows that carry real tokens.
    fn max_diff_on_real_rows(a: &Tensor, b: &Tensor, pads: &[usize]) -> f32 {
        pads.iter()
            .enumerate()
            .map(|(row, &pad)| {
                let seq = a.dim(2).expect("seq");
                let take = |t: &Tensor| {
                    t.narrow(0, row, 1)
                        .and_then(|t| t.narrow(2, pad, seq - pad))
                        .expect("narrow")
                };
                (take(a) - take(b))
                    .and_then(|d| d.abs())
                    .and_then(|d| d.max_all())
                    .and_then(|d| d.to_scalar::<f32>())
                    .expect("diff")
            })
            .fold(0.0, f32::max)
    }

    #[test]
    fn tiled_and_flash_match_eager_on_padded_batches() {
        let device = Device::Cpu;
        // Mixed padding, including a row with none and a row that is almost all padding.
        let case = case(&device, &[0, 5, 11, 2], 13);
        let eager = run(AttentionBackend::Eager, &case, &device, DType::F32);
        for backend in [AttentionBackend::Tiled, AttentionBackend::CpuFlash] {
            let other = run(backend, &case, &device, DType::F32);
            let diff = max_diff_on_real_rows(&eager, &other, &case.pads);
            assert!(diff < 1e-5, "{backend:?} differs from eager by {diff}");
        }
    }

    #[test]
    fn tiling_is_exact_across_several_blocks() {
        let device = Device::Cpu;
        let case = case(&device, &[3, 0], 17);
        let mask = BatchMask::new(
            AttentionBackend::Tiled,
            case.pads.clone(),
            case.seq,
            &device,
            DType::F32,
        )
        .expect("mask");
        let reference = eager(
            &case.q,
            &case.k,
            &case.v,
            mask.additive().expect("mask"),
            scale(),
            HEADS / KV_HEADS,
        )
        .expect("eager");
        // 17 rows in blocks of 5: three full blocks and a remainder.
        let blocks = tiled(
            &case.q,
            &case.k,
            &case.v,
            mask.additive().expect("mask"),
            scale(),
            HEADS / KV_HEADS,
            5,
        )
        .expect("tiled");
        let diff = max_diff_on_real_rows(&reference, &blocks, &case.pads);
        assert!(diff < 1e-5, "tiled blocks differ from eager by {diff}");
    }

    #[test]
    fn padding_does_not_change_real_rows() {
        // The same sequence alone and inside a batch where it is left padded must agree.
        let device = Device::Cpu;
        let long = case(&device, &[0], 9);
        let padded = Case {
            q: Tensor::cat(
                &[
                    Tensor::zeros((1, HEADS, 4, DIM), DType::F32, &device).unwrap(),
                    long.q.clone(),
                ],
                2,
            )
            .unwrap(),
            k: Tensor::cat(
                &[
                    Tensor::zeros((1, KV_HEADS, 4, DIM), DType::F32, &device).unwrap(),
                    long.k.clone(),
                ],
                2,
            )
            .unwrap(),
            v: Tensor::cat(
                &[
                    Tensor::zeros((1, KV_HEADS, 4, DIM), DType::F32, &device).unwrap(),
                    long.v.clone(),
                ],
                2,
            )
            .unwrap(),
            pads: vec![4],
            seq: 13,
        };
        for backend in [
            AttentionBackend::Eager,
            AttentionBackend::Tiled,
            AttentionBackend::CpuFlash,
        ] {
            let alone = run(backend, &long, &device, DType::F32);
            let inside = run(backend, &padded, &device, DType::F32)
                .narrow(2, 4, 9)
                .unwrap();
            let diff = max_diff_on_real_rows(&alone, &inside, &[0]);
            assert!(
                diff < 1e-5,
                "{backend:?}: padding moved real rows by {diff}"
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_sdpa_matches_eager() {
        let Ok(device) = Device::new_metal(0) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let case = case(&device, &[0, 7, 3], 16);
        for (dtype, tolerance) in [(DType::F32, 1e-4), (DType::F16, 2e-2)] {
            let eager = run(AttentionBackend::Eager, &case, &device, dtype);
            let fused = run(AttentionBackend::MetalSdpa, &case, &device, dtype);
            let diff = max_diff_on_real_rows(&eager, &fused, &case.pads);
            assert!(
                diff < tolerance,
                "{dtype:?}: sdpa differs from eager by {diff}"
            );
        }
    }

    #[test]
    fn auto_picks_the_fused_kernel_where_one_exists() {
        let cpu = Device::Cpu;
        assert_eq!(
            AttentionBackend::resolve(AttentionRequest::Auto, &cpu, DType::F32, 128).unwrap(),
            AttentionBackend::CpuFlash
        );
        // The CPU kernel only handles f32; half precision falls back to tiling.
        assert_eq!(
            AttentionBackend::resolve(AttentionRequest::Auto, &cpu, DType::F16, 128).unwrap(),
            AttentionBackend::Tiled
        );
        assert!(AttentionBackend::resolve(AttentionRequest::Fused, &cpu, DType::F16, 128).is_err());
        assert_eq!(
            AttentionBackend::resolve(AttentionRequest::Eager, &cpu, DType::F32, 128).unwrap(),
            AttentionBackend::Eager
        );
    }

    #[test]
    fn request_parsing_rejects_unknown_values() {
        assert_eq!(
            AttentionRequest::parse(None).unwrap(),
            AttentionRequest::Auto
        );
        assert_eq!(
            AttentionRequest::parse(Some("tiled")).unwrap(),
            AttentionRequest::Tiled
        );
        assert!(AttentionRequest::parse(Some("flash")).is_err());
    }

    #[test]
    fn score_rows_bound_attention_scratch_per_backend() {
        assert_eq!(AttentionBackend::Eager.score_rows(2048), 2048);
        assert_eq!(AttentionBackend::Tiled.score_rows(2048), QUERY_TILE);
        assert_eq!(AttentionBackend::Tiled.score_rows(100), 100);
        assert_eq!(AttentionBackend::MetalSdpa.score_rows(2048), 0);
        assert_eq!(AttentionBackend::CpuFlash.score_rows(2048), 0);
    }
}
