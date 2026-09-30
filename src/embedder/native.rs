//! In-process native embedder backed by fastembed-rs (Qwen3 / candle).

use crate::config::EmbedderConfig;
use crate::embedder::Embedder;
use crate::error::{MinSyncError, Result};
use async_trait::async_trait;
use candle_core::{DType, Device};
use fastembed::Qwen3TextEmbedding;
use hf_hub::api::sync::ApiBuilder;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer};
use tokio::sync::OnceCell;

const DEFAULT_MAX_LENGTH: usize = 2048;

#[cfg(test)]
#[test]
fn tokenized_length_detects_qwen3_limit() {
    let tokenizer = test_tokenizer();
    let text = "a ".repeat(DEFAULT_MAX_LENGTH);
    assert!(tokenized_length(&tokenizer, &text) >= DEFAULT_MAX_LENGTH);
}

#[cfg(test)]
#[test]
fn tokenized_length_short_control_stays_below_limit() {
    let tokenizer = test_tokenizer();
    assert!(tokenized_length(&tokenizer, "short control text") < DEFAULT_MAX_LENGTH);
}

/// Upper bound on the attention-score tensor (`batch x heads x seq x seq`)
/// materialized per forward pass. On Metal, a tensor above 4 GiB makes the
/// forward return NaN or silently wrong (finite) vectors, e.g. 17 inputs
/// padded to 2048 tokens with Qwen3-0.6B's 16 heads is ~4.6 GiB. 1 GiB keeps
/// a wide margin and bounds peak memory; smaller batches of long inputs cost
/// no throughput because attention is already compute-bound there.
const ATTENTION_BYTES_BUDGET: usize = 1 << 30;

/// Qwen3 uses byte-level BPE, so a text never tokenizes to more tokens than
/// its UTF-8 byte length. This slack covers template/special tokens.
const SPECIAL_TOKEN_SLACK: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NativeDevice {
    Auto,
    Cpu,
    Metal,
    Cuda,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NativeDtype {
    F32,
    F16,
    Bf16,
}

pub struct NativeEmbedder {
    model_id: String,
    device: NativeDevice,
    dtype: NativeDtype,
    max_length: usize,
    batch_size: usize,
    cache_dir: Option<PathBuf>,
    query_prefix: Option<String>,
    passage_prefix: Option<String>,
    inner: OnceCell<Arc<Mutex<Qwen3TextEmbedding>>>,
    tokenizer: OnceCell<Tokenizer>,
}

impl NativeEmbedder {
    pub fn from_config(settings: &EmbedderConfig) -> Result<Self> {
        let model_id = parse_native_model_id(&settings.id)?.to_string();
        Ok(Self {
            model_id,
            device: parse_device(settings.device.as_deref())?,
            dtype: parse_dtype(settings.dtype.as_deref())?,
            max_length: settings.max_length.unwrap_or(DEFAULT_MAX_LENGTH),
            batch_size: settings.batch_size,
            cache_dir: settings.model_cache_dir.as_ref().map(PathBuf::from),
            query_prefix: settings.query_prefix.clone(),
            passage_prefix: settings.passage_prefix.clone(),
            inner: OnceCell::new(),
            tokenizer: OnceCell::new(),
        })
    }

    async fn tokenizer(&self) -> Result<&Tokenizer> {
        self.tokenizer
            .get_or_try_init(|| async {
                let repo = self.model_id.clone();
                let cache_dir = self.cache_dir.clone();
                tokio::task::spawn_blocking(move || {
                    apply_cache_dir(cache_dir.as_deref());
                    let api = if std::env::var_os("HF_HOME").is_some() {
                        ApiBuilder::from_env()
                    } else {
                        ApiBuilder::new().with_cache_dir(PathBuf::from(fastembed::get_cache_dir()))
                    }
                    .with_progress(false)
                    .build()
                    .map_err(|error| {
                        MinSyncError::Embedding(format!(
                            "native tokenizer API initialization failed: {error}"
                        ))
                    })?;
                    let path = api.model(repo).get("tokenizer.json").map_err(|error| {
                        MinSyncError::Embedding(format!("native tokenizer load failed: {error}"))
                    })?;
                    let mut tokenizer = Tokenizer::from_file(path).map_err(|error| {
                        MinSyncError::Embedding(format!("native tokenizer parse failed: {error}"))
                    })?;
                    let _ = tokenizer.with_padding(Some(PaddingParams {
                        strategy: PaddingStrategy::BatchLongest,
                        direction: tokenizers::PaddingDirection::Left,
                        ..Default::default()
                    }));
                    Ok(tokenizer)
                })
                .await
                .map_err(|error| {
                    MinSyncError::Embedding(format!("native tokenizer join failed: {error}"))
                })?
            })
            .await
    }

    async fn count_truncated_texts(&self, texts: &[String]) -> Result<usize> {
        let texts: Vec<String> = texts
            .iter()
            .map(|text| match &self.passage_prefix {
                Some(prefix) => format!("{prefix}{text}"),
                None => text.clone(),
            })
            .collect();
        let candidates: Vec<_> = texts
            .iter()
            // Qwen3's byte-level BPE cannot produce more tokens than UTF-8
            // bytes; this gate avoids tokenizing clearly short chunks while
            // retaining multibyte text as a conservative candidate.
            .filter(|text| text.len() >= self.max_length)
            .collect();
        if candidates.is_empty() {
            return Ok(0);
        }
        let tokenizer = self.tokenizer().await?;
        candidates
            .iter()
            .map(|text| {
                tokenizer
                    .encode(text.as_str(), true)
                    .map(|encoding| usize::from(encoding.len() >= self.max_length))
                    .map_err(|error| {
                        MinSyncError::Embedding(format!("native tokenizer encode failed: {error}"))
                    })
            })
            .sum()
    }

    async fn model(&self) -> Result<Arc<Mutex<Qwen3TextEmbedding>>> {
        self.inner
            .get_or_try_init(|| async {
                let repo = self.model_id.clone();
                let device = self.device;
                let dtype = self.dtype;
                let max_length = self.max_length;
                let cache_dir = self.cache_dir.clone();
                let loaded = tokio::task::spawn_blocking(move || {
                    apply_cache_dir(cache_dir.as_deref());
                    let device = resolve_device(device)?;
                    Qwen3TextEmbedding::from_hf(&repo, &device, dtype.to_candle(), max_length)
                        .map_err(|error| {
                            MinSyncError::Embedding(format!("native model load failed: {error}"))
                        })
                })
                .await
                .map_err(|error| {
                    MinSyncError::Embedding(format!("native model load join failed: {error}"))
                })??;
                Ok(Arc::new(Mutex::new(loaded)))
            })
            .await
            .cloned()
    }

    async fn embed_texts(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        if self.batch_size == 0 {
            return Err(MinSyncError::Embedding(
                "batch_size must be greater than 0".to_string(),
            ));
        }
        let model = self.model().await?;
        let batch_size = self.batch_size;
        let max_length = self.max_length;
        let dtype_bytes = self.dtype.to_candle().size_in_bytes();
        tokio::task::spawn_blocking(move || {
            let model = model.lock().map_err(|error| {
                MinSyncError::Embedding(format!("native model lock poisoned: {error}"))
            })?;
            let per_row_attention_bytes = model.config().num_attention_heads.max(1) * dtype_bytes;
            let batches = plan_batches(
                &texts,
                batch_size,
                max_length,
                per_row_attention_bytes,
                ATTENTION_BYTES_BUDGET,
            );
            let mut all = Vec::with_capacity(texts.len());
            for range in batches {
                let chunk = &texts[range];
                let vectors = model.embed(chunk).map_err(|error| {
                    MinSyncError::Embedding(format!("native embedding failed: {error}"))
                })?;
                if vectors.len() != chunk.len() {
                    return Err(MinSyncError::Embedding(format!(
                        "native embedder returned {} embeddings for {} inputs",
                        vectors.len(),
                        chunk.len()
                    )));
                }
                all.extend(vectors);
            }
            Ok(all)
        })
        .await
        .map_err(|error| {
            MinSyncError::Embedding(format!("native embedding join failed: {error}"))
        })?
    }
}

#[async_trait]
impl Embedder for NativeEmbedder {
    fn id(&self) -> &str {
        &self.model_id
    }

    async fn count_truncated(&self, texts: &[String]) -> Result<usize> {
        self.count_truncated_texts(texts).await
    }

    fn max_length(&self) -> Option<usize> {
        Some(self.max_length)
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let inputs = texts
            .iter()
            .map(|text| match &self.passage_prefix {
                Some(prefix) => format!("{prefix}{text}"),
                None => text.clone(),
            })
            .collect();
        self.embed_texts(inputs).await
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let input = match &self.query_prefix {
            Some(prefix) => format!("{prefix}{text}"),
            None => text.to_string(),
        };
        self.embed_texts(vec![input])
            .await?
            .into_iter()
            .next()
            .ok_or_else(|| MinSyncError::Embedding("empty response".to_string()))
    }
}

/// Split `texts` into consecutive batches of at most `batch_size` inputs whose
/// attention-score tensor stays within `budget_bytes`.
///
/// Inputs are left-padded to the longest member, so a batch costs
/// `len * per_row_bytes * padded^2` where `per_row_bytes` is
/// `heads * dtype_size` and `padded` is bounded by the byte length of the
/// longest text (capped at `max_length`). A single input always forms a
/// batch on its own even when it alone exceeds the budget.
fn plan_batches(
    texts: &[String],
    batch_size: usize,
    max_length: usize,
    per_row_bytes: usize,
    budget_bytes: usize,
) -> Vec<std::ops::Range<usize>> {
    let token_bound = |text: &String| (text.len() + SPECIAL_TOKEN_SLACK).min(max_length.max(1));
    let mut batches = Vec::new();
    let mut start = 0;
    let mut padded = 0;
    for (index, text) in texts.iter().enumerate() {
        let candidate_padded = padded.max(token_bound(text));
        let candidate_len = index - start + 1;
        let attention_bytes = candidate_len
            .saturating_mul(per_row_bytes)
            .saturating_mul(candidate_padded)
            .saturating_mul(candidate_padded);
        if index > start && (candidate_len > batch_size || attention_bytes > budget_bytes) {
            batches.push(start..index);
            start = index;
            padded = token_bound(text);
        } else {
            padded = candidate_padded;
        }
    }
    if start < texts.len() {
        batches.push(start..texts.len());
    }
    batches
}

fn parse_native_model_id(id: &str) -> Result<&str> {
    let model = id.strip_prefix("native:").unwrap_or(id);
    if model.is_empty() {
        return Err(MinSyncError::Config(
            "native embedder id is missing a model name".to_string(),
        ));
    }
    Ok(model)
}

#[cfg(test)]
fn tokenized_length(tokenizer: &Tokenizer, text: &str) -> usize {
    tokenizer
        .encode(text, true)
        .expect("Qwen3 tokenizer encodes unit test input")
        .len()
}

#[cfg(test)]
fn test_tokenizer() -> Tokenizer {
    Tokenizer::from_bytes(
        br#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {"[UNK]": 0, "a": 1},
                "unk_token": "[UNK]"
            }
        }"#,
    )
    .expect("build test tokenizer")
}

fn parse_device(value: Option<&str>) -> Result<NativeDevice> {
    match value {
        None | Some("auto") => Ok(NativeDevice::Auto),
        Some("cpu") => Ok(NativeDevice::Cpu),
        Some("metal") => Ok(NativeDevice::Metal),
        Some("cuda") => Ok(NativeDevice::Cuda),
        Some(other) => Err(MinSyncError::Config(format!(
            "invalid native device '{other}': expected auto, cpu, metal, or cuda"
        ))),
    }
}

fn parse_dtype(value: Option<&str>) -> Result<NativeDtype> {
    match value {
        None | Some("f32") => Ok(NativeDtype::F32),
        Some("f16") => Ok(NativeDtype::F16),
        Some("bf16") => Ok(NativeDtype::Bf16),
        Some(other) => Err(MinSyncError::Config(format!(
            "invalid native dtype '{other}': expected f32, f16, or bf16"
        ))),
    }
}

impl NativeDtype {
    fn to_candle(self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::F16 => DType::F16,
            Self::Bf16 => DType::BF16,
        }
    }
}

fn resolve_device(spec: NativeDevice) -> Result<Device> {
    match spec {
        NativeDevice::Cpu => Ok(Device::Cpu),
        NativeDevice::Auto => Ok(auto_device()),
        NativeDevice::Metal => metal_device(),
        NativeDevice::Cuda => Err(MinSyncError::Config(
            "device cuda requires a CUDA build of minsync".to_string(),
        )),
    }
}

fn auto_device() -> Device {
    #[cfg(target_os = "macos")]
    {
        Device::new_metal(0).unwrap_or(Device::Cpu)
    }
    #[cfg(not(target_os = "macos"))]
    {
        Device::Cpu
    }
}

fn metal_device() -> Result<Device> {
    #[cfg(target_os = "macos")]
    {
        Device::new_metal(0)
            .map_err(|error| MinSyncError::Config(format!("failed to open metal device: {error}")))
    }
    #[cfg(not(target_os = "macos"))]
    {
        Err(MinSyncError::Config(
            "device metal is only available on macOS".to_string(),
        ))
    }
}

fn apply_cache_dir(configured: Option<&std::path::Path>) {
    if let Some(dir) = configured {
        let _ = std::fs::create_dir_all(dir);
        std::env::set_var("FASTEMBED_CACHE_DIR", dir);
        return;
    }
    if std::env::var_os("HF_HOME").is_some() || std::env::var_os("FASTEMBED_CACHE_DIR").is_some() {
        return;
    }
    let dir = default_model_cache_dir();
    let _ = std::fs::create_dir_all(&dir);
    std::env::set_var("FASTEMBED_CACHE_DIR", &dir);
}

fn default_model_cache_dir() -> PathBuf {
    #[cfg(windows)]
    {
        let root = std::env::var("LOCALAPPDATA").unwrap_or_else(|_| ".".to_string());
        PathBuf::from(root).join("minsync").join("models")
    }
    #[cfg(not(windows))]
    {
        let root = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        PathBuf::from(root)
            .join(".cache")
            .join("minsync")
            .join("models")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Qwen3-Embedding-0.6B: 16 heads, f32.
    const QWEN3_F32_ROW: usize = 16 * 4;

    fn texts(lengths: &[usize]) -> Vec<String> {
        lengths.iter().map(|&len| "a".repeat(len)).collect()
    }

    fn max_attention_bytes(texts: &[String], batches: &[std::ops::Range<usize>]) -> usize {
        batches
            .iter()
            .map(|range| {
                let padded = texts[range.clone()]
                    .iter()
                    .map(|text| (text.len() + SPECIAL_TOKEN_SLACK).min(DEFAULT_MAX_LENGTH))
                    .max()
                    .unwrap();
                range.len() * QWEN3_F32_ROW * padded * padded
            })
            .max()
            .unwrap()
    }

    #[test]
    fn short_texts_keep_configured_batch_size() {
        let inputs = texts(&[40; 130]);
        let batches = plan_batches(
            &inputs,
            64,
            DEFAULT_MAX_LENGTH,
            QWEN3_F32_ROW,
            ATTENTION_BYTES_BUDGET,
        );
        assert_eq!(batches, vec![0..64, 64..128, 128..130]);
    }

    #[test]
    fn max_length_inputs_stay_under_budget_and_below_metal_4gib_limit() {
        // Reproduces the failing shape: 20 chunks that all pad to 2048 tokens
        // (~5.4 GiB of attention scores in one batch before the fix).
        let inputs = texts(&[200_000; 20]);
        let batches = plan_batches(
            &inputs,
            64,
            DEFAULT_MAX_LENGTH,
            QWEN3_F32_ROW,
            ATTENTION_BYTES_BUDGET,
        );
        assert!(max_attention_bytes(&inputs, &batches) <= ATTENTION_BYTES_BUDGET);
        assert!(max_attention_bytes(&inputs, &batches) < 1 << 32);
        assert_eq!(batches.iter().map(|range| range.len()).sum::<usize>(), 20);
    }

    #[test]
    fn batches_are_contiguous_and_cover_every_input_in_order() {
        let inputs = texts(&[10, 5000, 10, 10, 3000, 10, 90_000, 10, 10]);
        let batches = plan_batches(
            &inputs,
            4,
            DEFAULT_MAX_LENGTH,
            QWEN3_F32_ROW,
            ATTENTION_BYTES_BUDGET,
        );
        let mut next = 0;
        for range in &batches {
            assert_eq!(range.start, next);
            assert!(!range.is_empty() && range.len() <= 4);
            next = range.end;
        }
        assert_eq!(next, inputs.len());
        assert!(max_attention_bytes(&inputs, &batches) <= ATTENTION_BYTES_BUDGET);
    }

    #[test]
    fn single_oversized_input_still_forms_its_own_batch() {
        let inputs = texts(&[100_000, 100_000]);
        let batches = plan_batches(&inputs, 64, 32_768, QWEN3_F32_ROW, ATTENTION_BYTES_BUDGET);
        assert_eq!(batches, vec![0..1, 1..2]);
    }

    #[test]
    fn empty_input_plans_no_batches() {
        let batches = plan_batches(
            &[],
            64,
            DEFAULT_MAX_LENGTH,
            QWEN3_F32_ROW,
            ATTENTION_BYTES_BUDGET,
        );
        assert!(batches.is_empty());
    }
}
