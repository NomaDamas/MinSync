//! In-process native embedder (Qwen3 on candle). The model lives in
//! [`super::qwen3`]; this module owns device/dtype selection, batching and the
//! per-batch deadline.

use crate::config::EmbedderConfig;
use crate::embedder::deadline::BatchDeadline;
use crate::embedder::qwen3::{AttentionBackend, AttentionRequest, LoadOptions, Qwen3Embedder};
use crate::embedder::Embedder;
use crate::error::{MinSyncError, Result};
use async_trait::async_trait;
use candle_core::{DType, Device};
use hf_hub::api::sync::ApiBuilder;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer};
use tokio::sync::{Mutex as AsyncMutex, OnceCell};

pub(crate) const DEFAULT_MAX_LENGTH: usize = 2048;

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

/// Default cap on padded tokens (`inputs x longest input`) per forward pass.
///
/// The attention budget alone still lets 64 inputs of ~500 tokens (32k padded
/// tokens) share one pass, which on a CPU or a contended GPU can outlast the
/// per-batch deadline even though nothing is stuck. Passes of at most this
/// many tokens finish in seconds, so the deadline only fires on a real hang.
/// Splitting costs little throughput: a pass this size already saturates the
/// device.
const MAX_BATCH_TOKENS: usize = 4096;

/// Qwen3 uses byte-level BPE, so a text never tokenizes to more tokens than
/// its UTF-8 byte length. This slack covers template/special tokens.
const SPECIAL_TOKEN_SLACK: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NativeDevice {
    Auto,
    Cpu,
    Metal,
    Cuda,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NativeDtype {
    F32,
    F16,
    Bf16,
}

#[derive(Clone)]
struct LoadedModel {
    model: Arc<Mutex<Qwen3Embedder>>,
    /// Serializes forward passes ahead of the deadline; see [`embed_in_batches`].
    gate: Arc<tokio::sync::Mutex<()>>,
    attention_heads: usize,
    /// Output vector length (`hidden_size`).
    dimension: usize,
    /// `cpu`, `metal:N` or `cuda:N`.
    device: String,
    dtype: DType,
    backend: AttentionBackend,
    /// Deadline for one forward pass on the device the model loaded onto.
    batch_timeout: Duration,
}

/// A lazily loaded value that can be dropped again and reloaded on demand.
///
/// Concurrent callers share one load; a failed load is not cached.
pub(crate) struct LoadSlot<T> {
    value: AsyncMutex<Option<T>>,
}

impl<T: Clone> LoadSlot<T> {
    pub(crate) fn new() -> Self {
        Self {
            value: AsyncMutex::new(None),
        }
    }

    pub(crate) async fn get_or_load<F, Fut>(&self, load: F) -> Result<T>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        let mut slot = self.value.lock().await;
        if let Some(value) = slot.as_ref() {
            return Ok(value.clone());
        }
        let value = load().await?;
        *slot = Some(value.clone());
        Ok(value)
    }

    /// Drop the held value; true when one was resident. Waits for an
    /// in-progress load first, so the load's value is dropped too.
    pub(crate) async fn unload(&self) -> bool {
        self.value.lock().await.take().is_some()
    }

    /// True when a value is resident. A load or unload in progress reports
    /// false.
    pub(crate) fn is_loaded(&self) -> bool {
        self.value
            .try_lock()
            .map(|slot| slot.is_some())
            .unwrap_or(false)
    }
}

/// What a host needs to record which embedding space the vectors belong to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeIdentity {
    /// The embedder id as written in `config.toml`, e.g.
    /// `native:Qwen/Qwen3-Embedding-0.6B`.
    pub id: String,
    pub dimension: usize,
    /// Resolved precision: `f32`, `f16` or `bf16`.
    pub dtype: &'static str,
    /// Resolved device: `cpu`, `metal` or `cuda`.
    pub device: &'static str,
    pub max_length: usize,
}

pub struct NativeEmbedder {
    model_id: String,
    device: NativeDevice,
    /// `None` picks the device's default (see [`default_dtype`]).
    dtype: Option<NativeDtype>,
    attention: AttentionRequest,
    max_length: usize,
    batch_size: usize,
    max_batch_tokens: usize,
    /// Deadline from the config (`per_batch_timeout_seconds`, else
    /// `timeout_seconds`); see [`effective_batch_timeout`] for the CPU default.
    batch_timeout: Duration,
    batch_timeout_explicit: bool,
    cache_dir: Option<PathBuf>,
    query_prefix: Option<String>,
    passage_prefix: Option<String>,
    inner: LoadSlot<LoadedModel>,
    tokenizer: OnceCell<Tokenizer>,
    deadline: &'static BatchDeadline,
}

/// Process-wide: a timed-out native forward pass strands a thread (and the
/// model it holds) until exit, so every `NativeEmbedder` in the process —
/// including ones rebuilt per call by `DeferredEmbedder` — must fail fast
/// instead of loading another model.
pub(crate) static NATIVE_DEADLINE: BatchDeadline = BatchDeadline::new();

impl NativeEmbedder {
    pub fn from_config(settings: &EmbedderConfig) -> Result<Self> {
        let model_id = parse_native_model_id(&settings.id)?.to_string();
        Ok(Self {
            model_id,
            device: parse_device(settings.device.as_deref())?,
            dtype: parse_dtype(settings.dtype.as_deref())?,
            attention: AttentionRequest::parse(settings.attention.as_deref())?,
            max_length: settings.max_length.unwrap_or(DEFAULT_MAX_LENGTH),
            batch_size: settings.batch_size,
            max_batch_tokens: settings.max_batch_tokens.unwrap_or(MAX_BATCH_TOKENS),
            batch_timeout: batch_timeout(settings)?,
            batch_timeout_explicit: settings.per_batch_timeout_seconds.is_some(),
            cache_dir: settings.model_cache_dir.as_ref().map(PathBuf::from),
            query_prefix: settings.query_prefix.clone(),
            passage_prefix: settings.passage_prefix.clone(),
            inner: LoadSlot::new(),
            tokenizer: OnceCell::new(),
            deadline: &NATIVE_DEADLINE,
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
                        ApiBuilder::new().with_cache_dir(PathBuf::from(model_cache_dir()))
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

    async fn model(&self) -> Result<LoadedModel> {
        self.inner
            .get_or_load(|| async {
                let repo = self.model_id.clone();
                let device = self.device;
                let dtype = self.dtype;
                let attention = self.attention;
                let max_length = self.max_length;
                let cache_dir = self.cache_dir.clone();
                let batch_timeout = self.batch_timeout;
                let batch_timeout_explicit = self.batch_timeout_explicit;
                tokio::task::spawn_blocking(move || {
                    apply_cache_dir(cache_dir.as_deref());
                    let device = resolve_device(device)?;
                    let dtype =
                        dtype.map_or_else(|| default_dtype(&device), NativeDtype::to_candle);
                    let model = Qwen3Embedder::from_hf(&LoadOptions {
                        repo_id: &repo,
                        default_cache_dir: PathBuf::from(model_cache_dir()),
                        device: &device,
                        dtype,
                        max_length,
                        attention,
                    })?;
                    let batch_timeout = effective_batch_timeout(
                        batch_timeout,
                        batch_timeout_explicit,
                        device.is_cpu(),
                    );
                    tracing::debug!(
                        device = ?device.location(),
                        ?dtype,
                        attention = ?model.backend(),
                        batch_timeout_seconds = batch_timeout.as_secs(),
                        "native embedder ready"
                    );
                    Ok(LoadedModel {
                        attention_heads: model.config().num_attention_heads.max(1),
                        dimension: model.config().hidden_size,
                        device: device_label(&device),
                        dtype,
                        backend: model.backend(),
                        batch_timeout,
                        model: Arc::new(Mutex::new(model)),
                        gate: Arc::new(tokio::sync::Mutex::new(())),
                    })
                })
                .await
                .map_err(|error| {
                    MinSyncError::Embedding(format!("native model load join failed: {error}"))
                })?
            })
            .await
    }

    /// The embedding space this embedder produces. Loads the model if it is
    /// not resident, because the device, dtype and dimension are only known
    /// once it is.
    pub async fn identity(&self) -> Result<NativeIdentity> {
        let loaded = self.model().await?;
        Ok(NativeIdentity {
            id: format!("native:{}", self.model_id),
            dimension: loaded.dimension,
            dtype: dtype_name(loaded.dtype),
            device: device_kind(&loaded.device),
            max_length: self.max_length,
        })
    }

    /// Release the model weights. The next embed reloads them. A forward pass
    /// already running keeps its own handle and finishes first. Returns true
    /// when a model was resident.
    pub async fn unload(&self) -> bool {
        self.inner.unload().await
    }

    /// True while the model weights are resident.
    pub fn is_loaded(&self) -> bool {
        self.inner.is_loaded()
    }

    /// Tokens per text as the model sees it before truncation.
    async fn token_counts_of(&self, texts: &[String]) -> Result<Vec<usize>> {
        let tokenizer = self.tokenizer().await?;
        texts
            .iter()
            .map(|text| {
                tokenizer
                    .encode(text.as_str(), true)
                    .map(|encoding| encoding.len())
                    .map_err(|error| {
                        MinSyncError::Embedding(format!("native tokenizer encode failed: {error}"))
                    })
            })
            .collect()
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
        self.deadline.ensure_healthy("native embedding")?;
        let loaded = self.model().await?;
        let per_row_attention_bytes = loaded.attention_heads * loaded.dtype.size_in_bytes();
        let backend = loaded.backend;
        let batch_timeout = loaded.batch_timeout;
        let model = loaded.model;
        let gate = loaded.gate;
        let batches = plan_batches(
            &texts,
            self.batch_size,
            self.max_length,
            per_row_attention_bytes,
            ATTENTION_BYTES_BUDGET,
            self.max_batch_tokens,
            backend,
        );
        embed_in_batches(
            self.deadline,
            &gate,
            batch_timeout,
            texts,
            batches,
            move |chunk: &[String]| {
                let model = model.lock().map_err(|error| {
                    MinSyncError::Embedding(format!("native model lock poisoned: {error}"))
                })?;
                model.embed(chunk).map_err(|error| {
                    MinSyncError::Embedding(format!("native embedding failed: {error}"))
                })
            },
        )
        .await
    }
}

/// Run `embed_batch` over each planned batch on the blocking pool, each
/// forward pass under its own `batch_timeout` so a stuck kernel surfaces as
/// [`MinSyncError::Timeout`] instead of hanging the caller.
///
/// `gate` admits one forward pass at a time. Callers queue on it *before*
/// the deadline starts, so a pass only times out on its own runtime, not
/// while another caller (e.g. a concurrent server request) holds the model.
async fn embed_in_batches<F>(
    deadline: &BatchDeadline,
    gate: &tokio::sync::Mutex<()>,
    batch_timeout: Duration,
    texts: Vec<String>,
    batches: Vec<std::ops::Range<usize>>,
    embed_batch: F,
) -> Result<Vec<Vec<f32>>>
where
    F: Fn(&[String]) -> Result<Vec<Vec<f32>>> + Send + Sync + 'static,
{
    let texts = Arc::new(texts);
    let embed_batch = Arc::new(embed_batch);
    let mut all = Vec::with_capacity(texts.len());
    for range in batches {
        let batch_texts = Arc::clone(&texts);
        let embed_batch = Arc::clone(&embed_batch);
        let expected = range.len();
        let _turn = gate.lock().await;
        let vectors = deadline
            .run("native embedding batch", batch_timeout, move || {
                embed_batch(&batch_texts[range])
            })
            .await?;
        if vectors.len() != expected {
            return Err(MinSyncError::Embedding(format!(
                "native embedder returned {} embeddings for {expected} inputs",
                vectors.len(),
            )));
        }
        all.extend(vectors);
    }
    Ok(all)
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

    fn is_deterministic(&self) -> bool {
        true
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

/// `minsync serve` runs the same engine, batch planner and deadline as
/// `sync`; only the prefixes move to the caller.
#[async_trait]
impl crate::serve::ServeEngine for NativeEmbedder {
    async fn load(&self) -> Result<crate::serve::EngineInfo> {
        self.deadline.ensure_healthy("native embedding")?;
        let loaded = self.model().await?;
        // The tokenizer guards the context per request; load it up front too.
        self.tokenizer().await?;
        Ok(crate::serve::EngineInfo {
            model_id: self.model_id.clone(),
            dimension: loaded.dimension,
            dtype: loaded.dtype.as_str().to_string(),
            device: loaded.device,
            attention: format!("{:?}", loaded.backend).to_ascii_lowercase(),
            max_length: self.max_length,
        })
    }

    async fn token_counts(&self, texts: &[String]) -> Result<Vec<usize>> {
        self.token_counts_of(texts).await
    }

    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        self.embed_texts(texts).await
    }
}

/// Split `texts` into consecutive batches of at most `batch_size` inputs whose
/// attention scratch stays within `budget_bytes` and whose padded token count
/// stays within `max_batch_tokens`.
///
/// Inputs are left-padded to the longest member, so a batch costs `len * padded`
/// tokens, where `padded` is bounded by the byte length of the longest text
/// (capped at `max_length`), and
/// `len * per_row_bytes * padded * backend.score_rows(padded)` bytes of
/// attention scores, where `per_row_bytes` is `heads * dtype_size`. The
/// eager backend keeps `padded` score rows alive, the tiled one a fixed tile,
/// and the fused kernels none, so they differ in how much the budget binds.
/// A single input always forms a batch on its own even when it alone exceeds
/// either budget.
fn plan_batches(
    texts: &[String],
    batch_size: usize,
    max_length: usize,
    per_row_bytes: usize,
    budget_bytes: usize,
    max_batch_tokens: usize,
    backend: AttentionBackend,
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
            .saturating_mul(backend.score_rows(candidate_padded));
        let padded_tokens = candidate_len.saturating_mul(candidate_padded);
        if index > start
            && (candidate_len > batch_size
                || attention_bytes > budget_bytes
                || padded_tokens > max_batch_tokens)
        {
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

/// Deadline for one forward pass in the config: `per_batch_timeout_seconds`,
/// else `timeout_seconds`.
pub(crate) fn batch_timeout(settings: &EmbedderConfig) -> Result<Duration> {
    let seconds = settings
        .per_batch_timeout_seconds
        .unwrap_or(settings.timeout_seconds);
    if seconds == 0 {
        return Err(MinSyncError::Config(
            "embedder.timeout_seconds / per_batch_timeout_seconds must be greater than 0 for the native embedder".to_string(),
        ));
    }
    Ok(Duration::from_secs(seconds))
}

/// Per-pass deadline on CPU when `per_batch_timeout_seconds` is not set. A CPU
/// pass is slow but does not hang: a 2-chunk pass of long inputs needs minutes
/// on a mid-range desktop CPU, which the 60 s default meant for a stuck GPU
/// kernel would abort as a timeout.
pub(crate) const CPU_BATCH_TIMEOUT: Duration = Duration::from_secs(900);

/// The deadline a loaded model runs under. An explicit
/// `per_batch_timeout_seconds` always wins; otherwise CPU gets at least
/// [`CPU_BATCH_TIMEOUT`] (a larger `timeout_seconds` is kept) and GPUs keep the
/// configured value.
pub(crate) fn effective_batch_timeout(
    configured: Duration,
    explicit: bool,
    on_cpu: bool,
) -> Duration {
    if on_cpu && !explicit {
        configured.max(CPU_BATCH_TIMEOUT)
    } else {
        configured
    }
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

pub(crate) fn parse_device(value: Option<&str>) -> Result<NativeDevice> {
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

pub(crate) fn parse_dtype(value: Option<&str>) -> Result<Option<NativeDtype>> {
    match value {
        None | Some("auto") => Ok(None),
        Some("f32") => Ok(Some(NativeDtype::F32)),
        Some("f16") => Ok(Some(NativeDtype::F16)),
        Some("bf16") => Ok(Some(NativeDtype::Bf16)),
        Some(other) => Err(MinSyncError::Config(format!(
            "invalid native dtype '{other}': expected auto, f32, f16, or bf16"
        ))),
    }
}

/// Weight/activation precision when the config does not name one: half
/// precision on Metal, where it cuts the model's footprint by a third at no
/// measurable retrieval cost; f32 elsewhere, where half-precision kernels are
/// either slow (CPU, pre-Volta CUDA) or not worth the precision.
pub(crate) fn default_dtype(device: &Device) -> DType {
    if device.is_metal() {
        DType::F16
    } else {
        DType::F32
    }
}

impl NativeDtype {
    pub(crate) fn to_candle(self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::F16 => DType::F16,
            Self::Bf16 => DType::BF16,
        }
    }
}

pub(crate) fn dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        other => other.as_str(),
    }
}

/// `cpu`, `metal` or `cuda`, from a [`device_label`] such as `metal:0`.
pub(crate) fn device_kind(label: &str) -> &'static str {
    match label.split(':').next() {
        Some("metal") => "metal",
        Some("cuda") => "cuda",
        _ => "cpu",
    }
}

pub(crate) fn resolve_device(spec: NativeDevice) -> Result<Device> {
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

/// Cache directory for model files. `FASTEMBED_CACHE_DIR` is the name earlier
/// releases used (and documented), so it stays the override.
pub(crate) fn model_cache_dir() -> String {
    std::env::var("FASTEMBED_CACHE_DIR").unwrap_or_else(|_| ".fastembed_cache".to_string())
}

pub(crate) fn apply_cache_dir(configured: Option<&std::path::Path>) {
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
    minsync_cache_root().join("models")
}

/// `~/.cache/minsync` (`%LOCALAPPDATA%\minsync` on Windows).
pub(crate) fn minsync_cache_root() -> PathBuf {
    #[cfg(windows)]
    {
        let root = std::env::var("LOCALAPPDATA").unwrap_or_else(|_| ".".to_string());
        PathBuf::from(root).join("minsync")
    }
    #[cfg(not(windows))]
    {
        let root = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        PathBuf::from(root).join(".cache").join("minsync")
    }
}

pub(crate) fn device_label(device: &Device) -> String {
    // Metal's `gpu_id` is an IORegistry id, not an ordinal; candle always
    // opens device 0, so the label carries no index.
    match device.location() {
        candle_core::DeviceLocation::Cpu => "cpu".to_string(),
        candle_core::DeviceLocation::Metal { .. } => "metal".to_string(),
        candle_core::DeviceLocation::Cuda { gpu_id } => format!("cuda:{gpu_id}"),
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
            MAX_BATCH_TOKENS,
            AttentionBackend::Eager,
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
            MAX_BATCH_TOKENS,
            AttentionBackend::Eager,
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
            MAX_BATCH_TOKENS,
            AttentionBackend::Eager,
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
        let batches = plan_batches(
            &inputs,
            64,
            32_768,
            QWEN3_F32_ROW,
            ATTENTION_BYTES_BUDGET,
            MAX_BATCH_TOKENS,
            AttentionBackend::Eager,
        );
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
            MAX_BATCH_TOKENS,
            AttentionBackend::Eager,
        );
        assert!(batches.is_empty());
    }

    fn max_padded_tokens(texts: &[String], batches: &[std::ops::Range<usize>]) -> usize {
        batches
            .iter()
            .map(|range| {
                let padded = texts[range.clone()]
                    .iter()
                    .map(|text| (text.len() + SPECIAL_TOKEN_SLACK).min(DEFAULT_MAX_LENGTH))
                    .max()
                    .unwrap();
                range.len() * padded
            })
            .max()
            .unwrap()
    }

    #[test]
    fn mid_length_chunks_are_capped_by_padded_token_budget_not_just_batch_size() {
        // 64 chunks of ~1000 tokens fit the 1 GiB attention budget (64 MB per
        // row of attention x 16 rows) yet are 64k padded tokens in one pass.
        let inputs = texts(&[1_000; 64]);
        let batches = plan_batches(
            &inputs,
            64,
            DEFAULT_MAX_LENGTH,
            QWEN3_F32_ROW,
            ATTENTION_BYTES_BUDGET,
            MAX_BATCH_TOKENS,
            AttentionBackend::Eager,
        );
        assert!(
            max_padded_tokens(&inputs, &batches) <= MAX_BATCH_TOKENS,
            "{batches:?}"
        );
        assert!(batches.iter().all(|range| range.len() <= 8), "{batches:?}");
        assert_eq!(batches.iter().map(|range| range.len()).sum::<usize>(), 64);
    }

    #[test]
    fn token_budget_is_checked_against_the_padded_longest_member() {
        // One long text in a run of short ones raises the padded length of
        // every row of its batch.
        let mut lengths = vec![40; 20];
        lengths[10] = 4_000;
        let inputs = texts(&lengths);
        let batches = plan_batches(
            &inputs,
            64,
            DEFAULT_MAX_LENGTH,
            QWEN3_F32_ROW,
            ATTENTION_BYTES_BUDGET,
            MAX_BATCH_TOKENS,
            AttentionBackend::Eager,
        );
        assert!(max_padded_tokens(&inputs, &batches) <= MAX_BATCH_TOKENS);
    }

    #[test]
    fn single_input_over_token_budget_still_forms_its_own_batch() {
        let inputs = texts(&[100_000, 100_000]);
        let batches = plan_batches(
            &inputs,
            64,
            32_768,
            QWEN3_F32_ROW,
            usize::MAX,
            4_096,
            AttentionBackend::Eager,
        );
        assert_eq!(batches, vec![0..1, 1..2]);
    }

    #[test]
    fn fused_kernels_are_not_limited_by_the_attention_budget() {
        // Four 2048-token inputs: eager attention scores are 256 MiB each, so a
        // 256 MiB budget splits them one per batch; a fused kernel holds none.
        let inputs = texts(&[200_000; 4]);
        let plan = |backend| {
            plan_batches(
                &inputs,
                64,
                DEFAULT_MAX_LENGTH,
                QWEN3_F32_ROW,
                ATTENTION_BYTES_BUDGET / 4,
                usize::MAX,
                backend,
            )
        };
        assert_eq!(plan(AttentionBackend::Eager).len(), 4);
        assert_eq!(plan(AttentionBackend::MetalSdpa), vec![0..4]);
        assert_eq!(plan(AttentionBackend::CpuFlash), vec![0..4]);
    }

    #[test]
    fn tiled_attention_scratch_binds_far_later_than_eager() {
        let inputs = texts(&[200_000; 16]);
        let plan = |backend| {
            plan_batches(
                &inputs,
                64,
                DEFAULT_MAX_LENGTH,
                QWEN3_F32_ROW,
                ATTENTION_BYTES_BUDGET,
                usize::MAX,
                backend,
            )
        };
        let eager = plan(AttentionBackend::Eager);
        let tiled = plan(AttentionBackend::Tiled);
        assert!(eager.iter().all(|range| range.len() <= 4), "{eager:?}");
        // 256 rows of 2048 keys per tile keep scratch at 2048 x 256 x 64 B = 32 MiB per input.
        assert!(tiled.iter().all(|range| range.len() <= 32), "{tiled:?}");
        assert!(tiled.len() < eager.len());
    }

    #[test]
    fn dtype_and_attention_settings_are_validated() {
        assert_eq!(parse_dtype(None).unwrap(), None);
        assert_eq!(parse_dtype(Some("auto")).unwrap(), None);
        assert_eq!(parse_dtype(Some("f16")).unwrap(), Some(NativeDtype::F16));
        assert!(parse_dtype(Some("int8")).is_err());
    }

    #[test]
    fn default_dtype_is_f32_unless_the_device_is_metal() {
        assert_eq!(default_dtype(&Device::Cpu), DType::F32);
        #[cfg(target_os = "macos")]
        if let Ok(metal) = Device::new_metal(0) {
            assert_eq!(default_dtype(&metal), DType::F16);
        }
    }
}

#[cfg(test)]
mod deadline_tests {
    use super::*;
    use std::time::Instant;

    fn settings(timeout: u64, per_batch: Option<u64>) -> EmbedderConfig {
        let mut config = crate::config::Config::default_for("s").embedder;
        config.timeout_seconds = timeout;
        config.per_batch_timeout_seconds = per_batch;
        config
    }

    #[test]
    fn batch_timeout_defaults_to_timeout_seconds() {
        assert_eq!(
            batch_timeout(&settings(42, None)).unwrap(),
            Duration::from_secs(42)
        );
    }

    #[test]
    fn per_batch_timeout_overrides_timeout_seconds() {
        assert_eq!(
            batch_timeout(&settings(42, Some(3))).unwrap(),
            Duration::from_secs(3)
        );
    }

    #[test]
    fn zero_timeout_is_rejected() {
        assert!(matches!(
            batch_timeout(&settings(0, None)),
            Err(MinSyncError::Config(_))
        ));
        assert!(matches!(
            batch_timeout(&settings(42, Some(0))),
            Err(MinSyncError::Config(_))
        ));
    }

    #[test]
    fn cpu_defaults_to_a_longer_deadline_unless_one_is_configured() {
        let configured = Duration::from_secs(60);
        // CPU without per_batch_timeout_seconds: 900 s.
        assert_eq!(
            effective_batch_timeout(configured, false, true),
            Duration::from_secs(900)
        );
        // A larger timeout_seconds is not shortened.
        assert_eq!(
            effective_batch_timeout(Duration::from_secs(1200), false, true),
            Duration::from_secs(1200)
        );
        // An explicit per_batch_timeout_seconds wins, even when shorter.
        assert_eq!(
            effective_batch_timeout(Duration::from_secs(30), true, true),
            Duration::from_secs(30)
        );
        // GPUs keep the configured value (60 s by default).
        assert_eq!(
            effective_batch_timeout(configured, false, false),
            Duration::from_secs(60)
        );
        assert_eq!(
            effective_batch_timeout(configured, true, false),
            Duration::from_secs(60)
        );
    }

    #[tokio::test]
    async fn stuck_batch_returns_timeout_within_deadline() {
        let deadline = BatchDeadline::new();
        let texts: Vec<String> = (0..4).map(|i| format!("t{i}")).collect();
        let started = Instant::now();

        let outcome = embed_in_batches(
            &deadline,
            &tokio::sync::Mutex::new(()),
            Duration::from_millis(100),
            texts,
            vec![0..2, 2..4],
            |chunk: &[String]| {
                if chunk[0] == "t2" {
                    std::thread::sleep(Duration::from_millis(800));
                }
                Ok(chunk.iter().map(|_| vec![1.0]).collect())
            },
        )
        .await;

        assert!(
            matches!(outcome, Err(MinSyncError::Timeout(_))),
            "{outcome:?}"
        );
        assert!(started.elapsed() < Duration::from_millis(600));
    }

    #[tokio::test]
    async fn healthy_batches_return_vectors_in_input_order() {
        let deadline = BatchDeadline::new();
        let texts: Vec<String> = (0..5).map(|i| i.to_string()).collect();

        let vectors = embed_in_batches(
            &deadline,
            &tokio::sync::Mutex::new(()),
            Duration::from_secs(5),
            texts,
            vec![0..2, 2..5],
            |chunk: &[String]| {
                Ok(chunk
                    .iter()
                    .map(|text| vec![text.parse::<f32>().unwrap()])
                    .collect())
            },
        )
        .await
        .unwrap();

        assert_eq!(
            vectors,
            vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0]]
        );
    }

    #[tokio::test]
    #[allow(clippy::single_range_in_vec_init)]
    async fn vector_count_mismatch_is_an_embedding_error() {
        let deadline = BatchDeadline::new();

        let outcome = embed_in_batches(
            &deadline,
            &tokio::sync::Mutex::new(()),
            Duration::from_secs(5),
            vec!["a".into(), "b".into()],
            vec![0..2],
            |_: &[String]| Ok(vec![vec![1.0]]),
        )
        .await;

        assert!(matches!(outcome, Err(MinSyncError::Embedding(_))));
    }

    /// A server runs several requests at once against one model. Waiting
    /// for another request's forward pass must not count against this
    /// request's per-pass deadline, or a busy server would declare itself
    /// wedged although every pass finished in time.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[allow(clippy::single_range_in_vec_init)]
    async fn waiting_for_another_callers_pass_does_not_count_against_the_deadline() {
        let deadline = Arc::new(BatchDeadline::new());
        let gate = Arc::new(tokio::sync::Mutex::new(()));
        let model = Arc::new(Mutex::new(()));
        let run = |name: &'static str| {
            let deadline = Arc::clone(&deadline);
            let gate = Arc::clone(&gate);
            let model = Arc::clone(&model);
            tokio::spawn(async move {
                embed_in_batches(
                    &deadline,
                    &gate,
                    Duration::from_millis(400),
                    vec![name.to_string()],
                    vec![0..1],
                    move |chunk: &[String]| {
                        let _model = model.lock().unwrap();
                        std::thread::sleep(Duration::from_millis(250));
                        Ok(chunk.iter().map(|_| vec![1.0]).collect())
                    },
                )
                .await
            })
        };
        let handles: Vec<_> = ["a", "b", "c"].into_iter().map(run).collect();
        for handle in handles {
            let outcome = handle.await.unwrap();
            assert!(outcome.is_ok(), "{outcome:?}");
        }
    }
}

#[cfg(test)]
mod lifecycle_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[tokio::test]
    async fn concurrent_callers_share_one_load() {
        let slot: LoadSlot<Arc<u32>> = LoadSlot::new();
        let loads = AtomicUsize::new(0);
        let load = || async {
            loads.fetch_add(1, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(30)).await;
            Ok(Arc::new(7))
        };

        let (a, b, c) = tokio::join!(
            slot.get_or_load(load),
            slot.get_or_load(load),
            slot.get_or_load(load)
        );

        assert_eq!((*a.unwrap(), *b.unwrap(), *c.unwrap()), (7, 7, 7));
        assert_eq!(loads.load(Ordering::SeqCst), 1);
        assert!(slot.is_loaded());
    }

    #[tokio::test]
    async fn unload_releases_the_value_and_the_next_call_reloads() {
        let slot: LoadSlot<Arc<u32>> = LoadSlot::new();
        let loads = AtomicUsize::new(0);
        let load = || async {
            loads.fetch_add(1, Ordering::SeqCst);
            Ok(Arc::new(1))
        };

        let first = slot.get_or_load(load).await.unwrap();
        assert_eq!(Arc::strong_count(&first), 2);

        assert!(slot.unload().await);
        assert!(!slot.is_loaded());
        assert_eq!(Arc::strong_count(&first), 1, "slot must drop its handle");
        assert!(!slot.unload().await, "second unload is a no-op");

        slot.get_or_load(load).await.unwrap();
        assert_eq!(loads.load(Ordering::SeqCst), 2);
        assert!(slot.is_loaded());
    }

    #[tokio::test]
    async fn failed_load_is_not_cached() {
        let slot: LoadSlot<u32> = LoadSlot::new();

        let failed = slot
            .get_or_load(|| async { Err(MinSyncError::Embedding("no weights".to_string())) })
            .await;
        assert!(matches!(failed, Err(MinSyncError::Embedding(_))));
        assert!(!slot.is_loaded());

        assert_eq!(slot.get_or_load(|| async { Ok(5) }).await.unwrap(), 5);
    }

    #[test]
    fn identity_names_match_the_config_vocabulary() {
        assert_eq!(dtype_name(DType::F32), "f32");
        assert_eq!(dtype_name(DType::F16), "f16");
        assert_eq!(dtype_name(DType::BF16), "bf16");
        assert_eq!(device_kind(&device_label(&Device::Cpu)), "cpu");
        assert_eq!(device_kind("metal:0"), "metal");
        assert_eq!(device_kind("cuda:1"), "cuda");
        #[cfg(target_os = "macos")]
        if let Ok(metal) = Device::new_metal(0) {
            assert_eq!(device_kind(&device_label(&metal)), "metal");
        }
    }
}
