//! In-process native sparse encoder: KURE-Sparse-nano (ModernBERT MLM + SPLADE
//! pooling) on candle. Device/dtype selection, the load slot, batching and the
//! per-batch deadline mirror [`crate::embedder::native::NativeEmbedder`]; the
//! model itself lives in [`super::modernbert`].

use super::modernbert::{ModernBertConfig, ModernBertMlm, SpladePooling};
use super::{SparseEncoder, SparseVector};
use crate::config::{SparseConfig, DEFAULT_SPARSE_MAX_LENGTH};
use crate::embedder::deadline::BatchDeadline;
use crate::embedder::hub::ModelSource;
use crate::embedder::native::{
    apply_cache_dir, device_kind, device_label, dtype_name, effective_batch_timeout,
    model_cache_dir, parse_device, parse_dtype, resolve_device, LoadSlot, NativeDevice,
    NativeDtype,
};
use crate::error::{MinSyncError, Result};
use async_trait::async_trait;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokenizers::{Tokenizer, TruncationParams};

/// Default cap on padded tokens (`inputs x longest input`) per forward pass.
/// Bounds activations and keeps one pass short enough to finish inside the
/// per-batch deadline on a loaded CPU.
///
/// Measured on KURE-Sparse-nano, inputs sorted longest first, Metal and CPU
/// f32: throughput falls as the budget grows (SciFact on Metal: 17.7 chunks/s
/// at 512 tokens, 13.7 at 4096, 12.2 at 8192) because the forward pass costs
/// more per token the larger the batch. 1024 is within 5% of the best value
/// on every corpus measured and does not shrink batches of short chunks the
/// way 128-256 does.
const MAX_BATCH_TOKENS: usize = 1024;

/// Process-wide, like the dense engine: a timed-out forward pass strands its
/// thread (and the model) until exit, so later calls must fail fast.
static SPARSE_DEADLINE: BatchDeadline = BatchDeadline::new();

/// Everything resident once the model is loaded.
struct Loaded {
    model: ModernBertMlm,
    pooling: SpladePooling,
    tokenizer: Tokenizer,
    /// Id of the pad token the right-padded id tensor is filled with.
    pad_id: u32,
    device: String,
    dtype: DType,
    batch_timeout: Duration,
}

/// What a host needs to record which sparse space the vectors belong to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseIdentity {
    pub id: String,
    /// Number of terms in the output space.
    pub dimension: usize,
    pub dtype: &'static str,
    pub device: &'static str,
    pub max_length: usize,
}

pub struct NativeSparseEncoder {
    model_id: String,
    device: NativeDevice,
    dtype: Option<NativeDtype>,
    max_length: usize,
    batch_size: usize,
    max_batch_tokens: usize,
    batch_timeout: Duration,
    batch_timeout_explicit: bool,
    cache_dir: Option<PathBuf>,
    query_prefix: Option<String>,
    passage_prefix: Option<String>,
    inner: LoadSlot<Arc<Loaded>>,
    gate: Arc<tokio::sync::Mutex<()>>,
    forward: Arc<Mutex<()>>,
}

impl NativeSparseEncoder {
    pub fn from_config(settings: &SparseConfig) -> Result<Self> {
        let model_id = parse_native_model_id(&settings.id)?.to_string();
        let seconds = settings
            .per_batch_timeout_seconds
            .unwrap_or(settings.timeout_seconds);
        if seconds == 0 {
            return Err(MinSyncError::Config(
                "sparse.timeout_seconds / per_batch_timeout_seconds must be greater than 0 for the native encoder"
                    .to_string(),
            ));
        }
        if settings.batch_size == 0 {
            return Err(MinSyncError::Config(
                "sparse.batch_size must be greater than 0".to_string(),
            ));
        }
        Ok(Self {
            model_id,
            device: parse_device(settings.device.as_deref())?,
            dtype: parse_dtype(settings.dtype.as_deref())?,
            max_length: settings.max_length.unwrap_or(DEFAULT_SPARSE_MAX_LENGTH),
            batch_size: settings.batch_size,
            max_batch_tokens: settings.max_batch_tokens.unwrap_or(MAX_BATCH_TOKENS),
            batch_timeout: Duration::from_secs(seconds),
            batch_timeout_explicit: settings.per_batch_timeout_seconds.is_some(),
            cache_dir: settings.model_cache_dir.as_ref().map(PathBuf::from),
            query_prefix: settings.query_prefix.clone(),
            passage_prefix: settings.passage_prefix.clone(),
            inner: LoadSlot::new(),
            gate: Arc::new(tokio::sync::Mutex::new(())),
            forward: Arc::new(Mutex::new(())),
        })
    }

    async fn loaded(&self) -> Result<Arc<Loaded>> {
        self.inner
            .get_or_load(|| async {
                let model_id = self.model_id.clone();
                let device = self.device;
                let dtype = self.dtype;
                let max_length = self.max_length;
                let cache_dir = self.cache_dir.clone();
                let (timeout, explicit) = (self.batch_timeout, self.batch_timeout_explicit);
                tokio::task::spawn_blocking(move || {
                    apply_cache_dir(cache_dir.as_deref());
                    let device = resolve_device(device)?;
                    // f32 on every device. Metal f16 measured no faster for this
                    // model (19.0 vs 18.8 chunks/s) but flips which terms clear the
                    // SPLADE threshold (query weights off by up to 0.0185, three terms
                    // dropped), and a sparse index is decided by which terms exist.
                    let dtype = dtype.map_or(DType::F32, NativeDtype::to_candle);
                    let source = ModelSource::resolve(&model_id, PathBuf::from(model_cache_dir()))?;
                    let loaded = load_model(&source, &device, dtype, max_length)?;
                    let batch_timeout = effective_batch_timeout(timeout, explicit, device.is_cpu());
                    tracing::debug!(
                        device = ?device.location(),
                        ?dtype,
                        batch_timeout_seconds = batch_timeout.as_secs(),
                        "native sparse encoder ready"
                    );
                    Ok(Arc::new(Loaded {
                        device: device_label(&device),
                        dtype,
                        batch_timeout,
                        ..loaded
                    }))
                })
                .await
                .map_err(|error| {
                    MinSyncError::Embedding(format!(
                        "native sparse model load join failed: {error}"
                    ))
                })?
            })
            .await
    }

    /// The sparse space this encoder produces. Loads the model if needed.
    pub async fn identity(&self) -> Result<SparseIdentity> {
        let loaded = self.loaded().await?;
        Ok(SparseIdentity {
            id: format!("native:{}", self.model_id),
            dimension: loaded.pooling.dimension(loaded.model.vocab_size()),
            dtype: dtype_name(loaded.dtype),
            device: device_kind(&loaded.device),
            max_length: self.max_length,
        })
    }

    /// Release the model weights; the next call reloads them.
    pub async fn unload(&self) -> bool {
        self.inner.unload().await
    }

    pub fn is_loaded(&self) -> bool {
        self.inner.is_loaded()
    }

    fn prefixed(&self, prefix: &Option<String>, text: &str) -> String {
        match prefix {
            Some(prefix) => format!("{prefix}{text}"),
            None => text.to_string(),
        }
    }

    async fn encode_inputs(&self, inputs: Vec<String>) -> Result<Vec<SparseVector>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        SPARSE_DEADLINE.ensure_healthy("native sparse encoding")?;
        let loaded = self.loaded().await?;
        // Tokenize once, up front: the plan needs real lengths, the forward needs ids.
        let tokenized = {
            let loaded = Arc::clone(&loaded);
            tokio::task::spawn_blocking(move || tokenize_all(&loaded.tokenizer, &inputs))
                .await
                .map_err(|error| {
                    MinSyncError::Embedding(format!("native sparse tokenize join failed: {error}"))
                })??
        };
        let lengths: Vec<usize> = tokenized.iter().map(Vec::len).collect();
        // Longest first, so the chunks of one batch have similar lengths and
        // padding stays small; `order[position]` is the input that sits at
        // `position` in the sorted run. Results go back to input order below.
        let order = longest_first(&lengths);
        let mut slots: Vec<Option<Vec<u32>>> = tokenized.into_iter().map(Some).collect();
        let sorted: Vec<Vec<u32>> = order
            .iter()
            .map(|&index| slots[index].take().unwrap_or_default())
            .collect();
        let sorted_lengths: Vec<usize> = sorted.iter().map(Vec::len).collect();
        let batches = plan_batches(&sorted_lengths, self.batch_size, self.max_batch_tokens);
        let sorted = Arc::new(sorted);
        let mut encoded = Vec::with_capacity(sorted.len());
        for range in batches {
            let expected = range.len();
            let _turn = self.gate.lock().await;
            let forward = Arc::clone(&self.forward);
            let model = Arc::clone(&loaded);
            let rows = Arc::clone(&sorted);
            let vectors = SPARSE_DEADLINE
                .run(
                    "native sparse encoding batch",
                    loaded.batch_timeout,
                    move || {
                        let _guard = forward.lock().map_err(|error| {
                            MinSyncError::Embedding(format!(
                                "native sparse model lock poisoned: {error}"
                            ))
                        })?;
                        encode_batch(&model, &rows[range])
                    },
                )
                .await?;
            if vectors.len() != expected {
                return Err(MinSyncError::Embedding(format!(
                    "native sparse encoder returned {} vectors for {expected} inputs",
                    vectors.len()
                )));
            }
            encoded.extend(vectors);
        }
        let mut all: Vec<Option<SparseVector>> = (0..encoded.len()).map(|_| None).collect();
        for (position, vector) in encoded.into_iter().enumerate() {
            all[order[position]] = Some(vector);
        }
        all.into_iter()
            .map(|vector| {
                vector.ok_or_else(|| {
                    MinSyncError::Embedding("native sparse encoder lost an input".to_string())
                })
            })
            .collect()
    }
}

fn parse_native_model_id(id: &str) -> Result<&str> {
    let model = id.strip_prefix("native:").unwrap_or(id);
    if model.is_empty() {
        return Err(MinSyncError::Config(
            "native sparse encoder id is missing a model name".to_string(),
        ));
    }
    Ok(model)
}

fn load_error(what: &str, error: &dyn std::fmt::Display) -> MinSyncError {
    MinSyncError::Embedding(format!("native sparse model {what}: {error}"))
}

/// Read the checkpoint: `config.json`, `modules.json` -> the SPLADE pooling
/// module's `config.json`, weights, `tokenizer.json`.
fn load_model(
    source: &ModelSource,
    device: &Device,
    dtype: DType,
    max_length: usize,
) -> Result<Loaded> {
    let config_bytes = std::fs::read(source.fetch("config.json")?)?;
    let config: ModernBertConfig = serde_json::from_slice(&config_bytes)
        .map_err(|error| load_error("config.json is not a ModernBERT config", &error))?;
    config.validate()?;
    let pooling = load_pooling(source)?;
    pooling.validate(config.vocab_size)?;

    let weights = source.weight_files()?;
    // SAFETY: the safetensors files are not modified while the model is loaded.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weights, dtype, device) }
        .map_err(|error| load_error("weight load failed", &error))?;
    let positions = max_length.min(config.max_position_embeddings);
    let model = ModernBertMlm::load(&config, vb, positions)?;

    let tokenizer_path = source.fetch("tokenizer.json")?;
    let mut tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|error| load_error("tokenizer parse failed", &error))?;
    tokenizer.with_padding(None);
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: positions,
            ..Default::default()
        }))
        .map_err(|error| load_error("tokenizer truncation setup failed", &error))?;
    let pad_id = tokenizer
        .get_padding()
        .map(|padding| padding.pad_id)
        .or_else(|| tokenizer.token_to_id("<pad>"))
        .unwrap_or(0);
    Ok(Loaded {
        model,
        pooling,
        tokenizer,
        pad_id,
        device: String::new(),
        dtype,
        batch_timeout: Duration::ZERO,
    })
}

/// The SPLADE pooling module is the entry of `modules.json` whose `type`
/// names a SPLADE pooling class; its config sits in that entry's `path`.
fn load_pooling(source: &ModelSource) -> Result<SpladePooling> {
    let modules: Vec<serde_json::Value> =
        serde_json::from_slice(&std::fs::read(source.fetch("modules.json")?)?)
            .map_err(|error| load_error("modules.json is invalid", &error))?;
    let module = modules
        .iter()
        .find(|module| {
            module["type"]
                .as_str()
                .is_some_and(|kind| kind.to_ascii_lowercase().contains("splade"))
        })
        .ok_or_else(|| {
            MinSyncError::Embedding(
                "native sparse model modules.json has no SPLADE pooling module".to_string(),
            )
        })?;
    let dir = module["path"].as_str().unwrap_or("");
    let path = if dir.is_empty() {
        "config.json".to_string()
    } else {
        format!("{}/config.json", dir.trim_end_matches('/'))
    };
    serde_json::from_slice(&std::fs::read(source.fetch(&path)?)?)
        .map_err(|error| load_error("SPLADE pooling config is invalid", &error))
}

/// Token ids of every input (special tokens added, cut at the context).
fn tokenize_all(tokenizer: &Tokenizer, inputs: &[String]) -> Result<Vec<Vec<u32>>> {
    inputs
        .iter()
        .map(|text| {
            tokenizer
                .encode(text.as_str(), true)
                .map(|encoding| encoding.get_ids().to_vec())
                .map_err(|error| {
                    MinSyncError::Embedding(format!(
                        "native sparse tokenizer encode failed: {error}"
                    ))
                })
        })
        .collect()
}

/// Right-pad `rows`, run the model and turn each dense row into a sparse vector.
fn encode_batch(loaded: &Loaded, rows: &[Vec<u32>]) -> Result<Vec<SparseVector>> {
    let seq = rows.iter().map(Vec::len).max().unwrap_or(0).max(1);
    let mut flat = Vec::with_capacity(rows.len() * seq);
    for row in rows {
        flat.extend_from_slice(row);
        flat.extend(std::iter::repeat_n(loaded.pad_id, seq - row.len()));
    }
    let ids = Tensor::from_vec(flat, (rows.len(), seq), loaded.model.device())
        .map_err(|error| MinSyncError::Embedding(format!("native sparse input failed: {error}")))?;
    let dense = loaded.model.splade(&ids, rows, &loaded.pooling)?;
    dense
        .into_iter()
        .map(|row| {
            let vector = SparseVector::from_pairs(
                row.iter()
                    .enumerate()
                    .map(|(term, &weight)| (term as u32, weight)),
            );
            // `from_pairs` drops non-finite weights silently; a vector that lost
            // terms to NaN would index wrongly, so fail the batch instead.
            if row.iter().any(|weight| !weight.is_finite()) {
                return Err(MinSyncError::Embedding(
                    "native sparse encoder produced non-finite weights".to_string(),
                ));
            }
            Ok(vector)
        })
        .collect()
}

/// Input indices ordered by token length, longest first; equal lengths keep
/// their input order, so the result is deterministic.
pub(crate) fn longest_first(lengths: &[usize]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..lengths.len()).collect();
    order.sort_by_key(|&index| std::cmp::Reverse(lengths[index]));
    order
}

/// Split inputs (by token length, in order) into consecutive batches of at
/// most `batch_size` inputs and `max_batch_tokens` padded tokens
/// (`inputs x longest`). A single input always forms a batch on its own.
pub(crate) fn plan_batches(
    lengths: &[usize],
    batch_size: usize,
    max_batch_tokens: usize,
) -> Vec<std::ops::Range<usize>> {
    let mut batches = Vec::new();
    let (mut start, mut padded) = (0, 0);
    for (index, &length) in lengths.iter().enumerate() {
        let candidate_padded = padded.max(length.max(1));
        let candidate_len = index - start + 1;
        if index > start
            && (candidate_len > batch_size
                || candidate_len.saturating_mul(candidate_padded) > max_batch_tokens)
        {
            batches.push(start..index);
            start = index;
            padded = length.max(1);
        } else {
            padded = candidate_padded;
        }
    }
    if start < lengths.len() {
        batches.push(start..lengths.len());
    }
    batches
}

#[async_trait]
impl SparseEncoder for NativeSparseEncoder {
    fn id(&self) -> &str {
        &self.model_id
    }

    fn max_length(&self) -> Option<usize> {
        Some(self.max_length)
    }

    fn is_deterministic(&self) -> bool {
        true
    }

    async fn count_truncated(&self, texts: &[String]) -> Result<usize> {
        // A chunk can only reach the limit if its UTF-8 byte length does; skip
        // tokenizing the clearly short ones.
        let candidates: Vec<String> = texts
            .iter()
            .filter(|text| text.len() >= self.max_length)
            .map(|text| self.prefixed(&self.passage_prefix, text))
            .collect();
        if candidates.is_empty() {
            return Ok(0);
        }
        let loaded = self.loaded().await?;
        let limit = self.max_length.min(loaded.model.positions());
        let counted = tokio::task::spawn_blocking(move || -> Result<usize> {
            // The loaded tokenizer already truncates, so a text that hit the
            // limit comes back exactly `limit` long.
            Ok(tokenize_all(&loaded.tokenizer, &candidates)?
                .iter()
                .filter(|ids| ids.len() >= limit)
                .count())
        })
        .await
        .map_err(|error| {
            MinSyncError::Embedding(format!("native sparse tokenize join failed: {error}"))
        })??;
        Ok(counted)
    }

    async fn encode(&self, texts: &[String]) -> Result<Vec<SparseVector>> {
        let inputs = texts
            .iter()
            .map(|text| self.prefixed(&self.passage_prefix, text))
            .collect();
        self.encode_inputs(inputs).await
    }

    async fn encode_query(&self, text: &str) -> Result<SparseVector> {
        self.encode_inputs(vec![self.prefixed(&self.query_prefix, text)])
            .await?
            .into_iter()
            .next()
            .ok_or_else(|| MinSyncError::Embedding("empty response".to_string()))
    }
}

/// `minsync serve --sparse` runs the same engine, batch planner and deadline
/// as `sync`; only the prefixes move to the caller.
#[async_trait]
impl crate::serve::SparseServeEngine for NativeSparseEncoder {
    async fn load(&self) -> Result<crate::serve::EngineInfo> {
        SPARSE_DEADLINE.ensure_healthy("native sparse encoding")?;
        let loaded = self.loaded().await?;
        Ok(crate::serve::EngineInfo {
            model_id: self.model_id.clone(),
            dimension: loaded.pooling.dimension(loaded.model.vocab_size()),
            dtype: dtype_name(loaded.dtype).to_string(),
            device: loaded.device.clone(),
            attention: "tiled".to_string(),
            max_length: self.max_length,
        })
    }

    async fn token_counts(&self, texts: &[String]) -> Result<Vec<usize>> {
        let loaded = self.loaded().await?;
        // Count without the context cut, so the server can tell an over-long input.
        let texts = texts.to_vec();
        tokio::task::spawn_blocking(move || -> Result<Vec<usize>> {
            let mut counter = loaded.tokenizer.clone();
            counter
                .with_truncation(None)
                .map_err(|error| load_error("tokenizer setup failed", &error))?;
            Ok(tokenize_all(&counter, &texts)?
                .iter()
                .map(Vec::len)
                .collect())
        })
        .await
        .map_err(|error| {
            MinSyncError::Embedding(format!("native sparse tokenize join failed: {error}"))
        })?
    }

    async fn encode(&self, texts: Vec<String>) -> Result<Vec<SparseVector>> {
        self.encode_inputs(texts).await
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use serde::Deserialize;
    use std::path::Path;

    pub(crate) fn tiny_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_splade")
    }

    pub(crate) fn tiny_settings() -> SparseConfig {
        SparseConfig {
            id: format!("native:{}", tiny_dir().display()),
            device: Some("cpu".to_string()),
            dtype: Some("f32".to_string()),
            ..SparseConfig::default()
        }
    }

    #[derive(Deserialize)]
    struct Golden {
        cases: Vec<Case>,
    }

    #[derive(Deserialize)]
    struct Case {
        kind: String,
        max_length: usize,
        texts: Vec<String>,
        batched: Vec<Row>,
    }

    #[derive(Deserialize)]
    struct Row {
        indices: Vec<u32>,
        values: Vec<f32>,
    }

    fn golden() -> Golden {
        serde_json::from_slice(&std::fs::read(tiny_dir().join("golden.json")).unwrap()).unwrap()
    }

    fn assert_close(got: &SparseVector, want: &Row, label: &str) {
        assert_eq!(got.indices(), want.indices.as_slice(), "{label}: terms");
        for (a, b) in got.values().iter().zip(&want.values) {
            assert!((a - b).abs() < 1e-4, "{label}: {a} vs {b}");
        }
    }

    #[tokio::test]
    async fn encodes_text_end_to_end_like_the_huggingface_reference() {
        // Real tokenizer + [D]/[Q] prefixes + model + pooling, from raw strings.
        for case in golden().cases.iter().filter(|c| c.max_length == 64) {
            let encoder = NativeSparseEncoder::from_config(&tiny_settings()).unwrap();
            let got = if case.kind == "document" {
                encoder.encode(&case.texts).await.unwrap()
            } else {
                let mut out = Vec::new();
                for text in &case.texts {
                    out.push(encoder.encode_query(text).await.unwrap());
                }
                out
            };
            assert_eq!(got.len(), case.batched.len());
            for (index, (got, want)) in got.iter().zip(&case.batched).enumerate() {
                assert_close(got, want, &format!("{} text {index}", case.kind));
            }
        }
    }

    #[tokio::test]
    async fn small_batches_and_token_budgets_do_not_change_vectors() {
        let case = golden().cases.into_iter().next().unwrap();
        let reference = NativeSparseEncoder::from_config(&tiny_settings())
            .unwrap()
            .encode(&case.texts)
            .await
            .unwrap();
        for (batch_size, tokens) in [(1, 8192), (2, 8192), (64, 20), (3, 1)] {
            let settings = SparseConfig {
                batch_size,
                max_batch_tokens: Some(tokens),
                ..tiny_settings()
            };
            let got = NativeSparseEncoder::from_config(&settings)
                .unwrap()
                .encode(&case.texts)
                .await
                .unwrap();
            for (index, (a, b)) in got.iter().zip(&reference).enumerate() {
                assert_eq!(
                    a.indices(),
                    b.indices(),
                    "batch {batch_size}/{tokens} text {index}"
                );
                for (x, y) in a.values().iter().zip(b.values()) {
                    assert!((x - y).abs() < 1e-4);
                }
            }
        }
    }

    #[tokio::test]
    async fn max_length_truncation_matches_the_reference_and_is_counted() {
        let golden = golden();
        let case = golden
            .cases
            .iter()
            .find(|c| c.kind == "document" && c.max_length == 12)
            .unwrap();
        let settings = SparseConfig {
            max_length: Some(12),
            ..tiny_settings()
        };
        let encoder = NativeSparseEncoder::from_config(&settings).unwrap();
        let got = encoder.encode(&case.texts).await.unwrap();
        for (index, (got, want)) in got.iter().zip(&case.batched).enumerate() {
            assert_close(got, want, &format!("truncated text {index}"));
        }
        // Only the 40-word text (index 1) exceeds 12 tokens once "[D] " and the
        // special tokens are counted; the 3-word text must not be reported.
        assert_eq!(encoder.count_truncated(&case.texts).await.unwrap(), 1);
        assert_eq!(encoder.count_truncated(&case.texts[..1]).await.unwrap(), 0);
    }

    #[tokio::test]
    async fn empty_input_loads_nothing_and_returns_nothing() {
        let encoder = NativeSparseEncoder::from_config(&tiny_settings()).unwrap();
        assert!(encoder.encode(&[]).await.unwrap().is_empty());
        assert!(!encoder.is_loaded(), "empty input must not load the model");
    }

    #[tokio::test]
    async fn identity_reports_the_term_space_and_unload_releases_the_model() {
        let encoder = NativeSparseEncoder::from_config(&tiny_settings()).unwrap();
        let identity = encoder.identity().await.unwrap();
        assert_eq!(identity.dimension, 62);
        assert_eq!((identity.dtype, identity.device), ("f32", "cpu"));
        assert!(encoder.is_loaded());
        assert!(encoder.unload().await);
        assert!(!encoder.is_loaded());
        // Reloads on demand.
        assert!(!encoder.encode_query("w1").await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn a_missing_model_is_a_clear_error_not_a_panic() {
        let settings = SparseConfig {
            id: "native:/definitely/not/here".to_string(),
            model_cache_dir: Some(tempfile::tempdir().unwrap().path().display().to_string()),
            ..tiny_settings()
        };
        let encoder = NativeSparseEncoder::from_config(&settings).unwrap();
        let error = encoder.encode_query("x").await.unwrap_err();
        assert!(matches!(error, MinSyncError::Embedding(_)), "{error}");
    }

    #[test]
    fn rejects_unusable_settings_up_front() {
        for settings in [
            SparseConfig {
                batch_size: 0,
                ..tiny_settings()
            },
            SparseConfig {
                timeout_seconds: 0,
                per_batch_timeout_seconds: None,
                ..tiny_settings()
            },
            SparseConfig {
                id: "native:".to_string(),
                ..tiny_settings()
            },
            SparseConfig {
                device: Some("tpu".to_string()),
                ..tiny_settings()
            },
            SparseConfig {
                dtype: Some("int4".to_string()),
                ..tiny_settings()
            },
        ] {
            assert!(
                NativeSparseEncoder::from_config(&settings).is_err(),
                "{settings:?}"
            );
        }
    }

    #[test]
    fn plan_batches_covers_inputs_in_order_within_both_budgets() {
        let lengths = [5, 5, 200, 5, 5, 5, 90, 90, 90];
        let batches = plan_batches(&lengths, 4, 100);
        let mut next = 0;
        for range in &batches {
            assert_eq!(range.start, next);
            next = range.end;
            let longest = lengths[range.clone()].iter().max().unwrap();
            assert!(range.len() <= 4);
            assert!(
                range.len() == 1 || range.len() * longest <= 100,
                "{range:?} pads to {}",
                range.len() * longest
            );
        }
        assert_eq!(next, lengths.len());
        assert!(plan_batches(&[], 4, 100).is_empty());
        // A lone over-budget input still gets a batch of its own.
        assert_eq!(plan_batches(&[500], 4, 100), vec![0..1]);
    }

    #[test]
    fn longest_first_orders_by_token_length_and_keeps_ties_in_input_order() {
        assert_eq!(longest_first(&[3, 9, 3, 7, 9]), vec![1, 4, 3, 0, 2]);
        assert!(longest_first(&[]).is_empty());
    }

    #[test]
    fn sorting_before_batching_cuts_padding() {
        // Short and long inputs alternate: in input order every batch pads its
        // short members up to a long one.
        let lengths: Vec<usize> = (0..64).map(|i| if i % 2 == 0 { 20 } else { 400 }).collect();
        let padded = |lengths: &[usize]| -> usize {
            plan_batches(lengths, 64, 2048)
                .iter()
                .map(|range| range.len() * lengths[range.clone()].iter().max().unwrap())
                .sum()
        };
        let sorted: Vec<usize> = longest_first(&lengths)
            .iter()
            .map(|&index| lengths[index])
            .collect();
        let (before, after) = (padded(&lengths), padded(&sorted));
        assert!(after * 10 < before * 7, "padded tokens {before} -> {after}");
    }

    #[tokio::test]
    async fn results_follow_input_order_when_batches_are_sorted_by_length() {
        // The golden texts differ in length, so sorting moves them; every vector
        // must still land on the text it came from.
        let case = golden()
            .cases
            .into_iter()
            .find(|c| c.kind == "document" && c.max_length == 64)
            .unwrap();
        let alone = NativeSparseEncoder::from_config(&SparseConfig {
            batch_size: 1,
            ..tiny_settings()
        })
        .unwrap();
        let mut want = Vec::new();
        for text in &case.texts {
            want.push(
                alone
                    .encode(std::slice::from_ref(text))
                    .await
                    .unwrap()
                    .remove(0),
            );
        }
        let got = NativeSparseEncoder::from_config(&tiny_settings())
            .unwrap()
            .encode(&case.texts)
            .await
            .unwrap();
        assert_eq!(got.len(), want.len());
        for (index, (got, want)) in got.iter().zip(&want).enumerate() {
            assert_eq!(got.indices(), want.indices(), "text {index}: terms");
            for (a, b) in got.values().iter().zip(want.values()) {
                assert!((a - b).abs() < 1e-4, "text {index}: {a} vs {b}");
            }
        }
    }
}
