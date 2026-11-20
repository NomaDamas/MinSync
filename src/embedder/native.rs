//! The `native:` dense embedder. All inference (device and dtype selection,
//! batching, the per-batch deadline, the models themselves) is
//! [`inl_text::TextEmbedder`] from `im-not-llm`; this module only maps
//! `[embedder]` settings onto its options and its answers onto [`Embedder`].

use crate::config::EmbedderConfig;
use crate::embedder::Embedder;
use crate::error::{MinSyncError, Result};
use async_trait::async_trait;
use inl_core::attention::AttentionRequest;
use inl_core::device::{device_kind, DeviceSpec, DtypeSpec};
use inl_core::{DenseEngine, EngineInfo};
use inl_text::{TextEmbedder, TextOptions};
use std::path::PathBuf;
use std::sync::Arc;

pub(crate) const DEFAULT_MAX_LENGTH: usize = 2048;

/// What a host needs to record which embedding space the vectors belong to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeIdentity {
    /// The embedder id as written in `config.toml`, e.g.
    /// `native:Qwen/Qwen3-Embedding-0.6B`.
    pub id: String,
    pub dimension: usize,
    /// Resolved precision: `f32`, `f16` or `bf16`.
    pub dtype: &'static str,
    /// Resolved device: `cpu`, `metal`, `cuda` or `ane`.
    pub device: &'static str,
    pub max_length: usize,
}

impl NativeIdentity {
    /// The identity of `model` as the engine that loaded it reports itself.
    pub fn of(model: &str, info: &EngineInfo) -> Self {
        Self {
            id: format!("native:{model}"),
            dimension: info.dimension,
            dtype: match info.dtype.as_str() {
                "f32" => "f32",
                "f16" => "f16",
                "bf16" => "bf16",
                _ => "unknown",
            },
            device: if info.device == "ane" {
                "ane"
            } else {
                device_kind(&info.device)
            },
            max_length: info.max_length,
        }
    }
}

impl From<inl_text::TextIdentity> for NativeIdentity {
    fn from(identity: inl_text::TextIdentity) -> Self {
        Self {
            id: format!("native:{}", identity.id),
            dimension: identity.dimension,
            dtype: identity.dtype,
            device: identity.device,
            max_length: identity.max_length,
        }
    }
}

/// The model of a `native:<model>` id (the prefix is optional).
pub(crate) fn parse_native_model_id(id: &str) -> Result<&str> {
    let model = id.strip_prefix("native:").unwrap_or(id);
    if model.is_empty() {
        return Err(MinSyncError::Config(
            "native embedder id is missing a model name".to_string(),
        ));
    }
    Ok(model)
}

/// Where model files are cached: the configured `model_cache_dir`, else
/// im-not-llm's `INL_CACHE_DIR`, else `<cache root>/models`
/// (`~/.cache/im-not-llm/models`). The directory is the one `inl serve`
/// uses, so MinSync and the daemon share a single weight cache. `HF_HOME`,
/// when set, still wins inside the hub client, as it always has.
pub(crate) fn model_cache_dir(configured: Option<&str>) -> PathBuf {
    inl_core::device::model_cache_dir(configured.map(std::path::Path::new))
}

/// The engine options `settings` describes.
pub fn text_options(settings: &EmbedderConfig) -> Result<TextOptions> {
    let defaults = TextOptions::default();
    Ok(TextOptions {
        id: parse_native_model_id(&settings.id)?.to_string(),
        device: DeviceSpec::parse(settings.device.as_deref())?,
        dtype: DtypeSpec::parse(settings.dtype.as_deref())?,
        attention: AttentionRequest::parse(settings.attention.as_deref())?,
        max_length: settings.max_length.unwrap_or(DEFAULT_MAX_LENGTH),
        batch_size: settings.batch_size,
        max_batch_tokens: settings
            .max_batch_tokens
            .unwrap_or(defaults.max_batch_tokens),
        timeout_seconds: settings.timeout_seconds,
        per_batch_timeout_seconds: settings.per_batch_timeout_seconds,
        cache_dir: Some(model_cache_dir(settings.model_cache_dir.as_deref())),
        query_prefix: settings.query_prefix.clone(),
        passage_prefix: settings.passage_prefix.clone(),
        output_dim: None,
    })
}

/// Options of an Apple Neural Engine package for `model`.
#[cfg(target_os = "macos")]
pub(crate) fn ane_options(
    model: &str,
    cache_dir: PathBuf,
    max_length: usize,
    timeout_seconds: u64,
    per_batch_timeout_seconds: Option<u64>,
) -> inl_ane::AneOptions {
    inl_ane::AneOptions {
        id: model.to_string(),
        cache_dir: Some(cache_dir),
        max_length,
        // An explicit deadline wins; otherwise a Neural Engine call gets at
        // least the engine's own default, as a CPU pass does on the candle path.
        timeout_seconds: per_batch_timeout_seconds
            .unwrap_or_else(|| timeout_seconds.max(inl_ane::DEFAULT_TIMEOUT_SECONDS)),
    }
}

fn dense_engine(settings: &EmbedderConfig, model: &str) -> Result<(Arc<dyn DenseEngine>, usize)> {
    match resolve_backend(
        model,
        false,
        settings.device.as_deref(),
        settings.dtype.as_deref(),
    )? {
        Backend::Candle => {
            let options = text_options(settings)?;
            let max_length = options.max_length;
            Ok((Arc::new(TextEmbedder::new(options)?), max_length))
        }
        Backend::Ane => {
            #[cfg(target_os = "macos")]
            {
                let max_length = settings.max_length.unwrap_or(DEFAULT_MAX_LENGTH);
                let options = ane_options(
                    model,
                    model_cache_dir(settings.model_cache_dir.as_deref()),
                    max_length,
                    settings.timeout_seconds,
                    settings.per_batch_timeout_seconds,
                );
                Ok((Arc::new(inl_ane::AneEmbedder::new(options)?), max_length))
            }
            #[cfg(not(target_os = "macos"))]
            Err(MinSyncError::Config(
                "the Apple Neural Engine needs an Apple silicon Mac".to_string(),
            ))
        }
    }
}

pub struct NativeEmbedder {
    model_id: String,
    max_length: usize,
    query_prefix: Option<String>,
    passage_prefix: Option<String>,
    engine: Arc<dyn DenseEngine>,
}

impl NativeEmbedder {
    pub fn from_config(settings: &EmbedderConfig) -> Result<Self> {
        let model_id = parse_native_model_id(&settings.id)?.to_string();
        let (engine, max_length) = dense_engine(settings, &model_id)?;
        Ok(Self {
            model_id,
            max_length,
            query_prefix: settings.query_prefix.clone(),
            passage_prefix: settings.passage_prefix.clone(),
            engine,
        })
    }

    /// The embedding space this embedder produces. Loads the model if it is
    /// not resident, because the device, dtype and dimension are only known
    /// once it is.
    pub async fn identity(&self) -> Result<NativeIdentity> {
        Ok(NativeIdentity::of(
            &self.model_id,
            &self.engine.load().await?,
        ))
    }

    /// The engine, for a server that drives it directly. It embeds exactly
    /// the text it is given: the server applies the prefixes.
    pub fn into_engine(self) -> Arc<dyn DenseEngine> {
        self.engine
    }

    fn prefixed(prefix: &Option<String>, text: &str) -> String {
        match prefix {
            Some(prefix) => format!("{prefix}{text}"),
            None => text.to_string(),
        }
    }
}

#[async_trait]
impl Embedder for NativeEmbedder {
    fn id(&self) -> &str {
        &self.model_id
    }

    async fn count_truncated(&self, texts: &[String]) -> Result<usize> {
        // No text reaches the context in tokens without reaching it in bytes
        // (byte-level and sentencepiece vocabularies emit at most one token
        // per byte), so only the long ones are tokenized.
        let candidates: Vec<String> = texts
            .iter()
            .filter(|text| text.len() >= self.max_length)
            .map(|text| Self::prefixed(&self.passage_prefix, text))
            .collect();
        if candidates.is_empty() {
            return Ok(0);
        }
        let counts = self.engine.token_counts(&candidates).await?;
        Ok(counts.iter().filter(|&&n| n >= self.max_length).count())
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
            .map(|text| Self::prefixed(&self.passage_prefix, text))
            .collect();
        Ok(self.engine.embed(inputs).await?)
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let input = Self::prefixed(&self.query_prefix, text);
        self.engine
            .embed(vec![input])
            .await?
            .into_iter()
            .next()
            .ok_or_else(|| MinSyncError::Embedding("empty response".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn settings() -> EmbedderConfig {
        Config::default_for("t").embedder
    }

    #[test]
    fn settings_map_onto_engine_options() {
        let mut settings = settings();
        settings.id = "native:org/model".to_string();
        settings.device = Some("cpu".to_string());
        settings.dtype = Some("bf16".to_string());
        settings.attention = Some("tiled".to_string());
        settings.max_length = Some(512);
        settings.batch_size = 8;
        settings.max_batch_tokens = Some(999);
        settings.timeout_seconds = 11;
        settings.per_batch_timeout_seconds = Some(5);
        settings.query_prefix = Some("Q: ".to_string());
        settings.passage_prefix = Some("P: ".to_string());
        settings.model_cache_dir = Some("/some/cache".to_string());

        let options = text_options(&settings).unwrap();
        assert_eq!(options.id, "org/model");
        assert_eq!(options.device, DeviceSpec::Cpu);
        assert_eq!(options.dtype, Some(DtypeSpec::Bf16));
        assert_eq!(options.attention, AttentionRequest::Tiled);
        assert_eq!(options.max_length, 512);
        assert_eq!(options.batch_size, 8);
        assert_eq!(options.max_batch_tokens, 999);
        assert_eq!(options.timeout_seconds, 11);
        assert_eq!(options.per_batch_timeout_seconds, Some(5));
        assert_eq!(options.query_prefix.as_deref(), Some("Q: "));
        assert_eq!(options.passage_prefix.as_deref(), Some("P: "));
        assert_eq!(options.cache_dir, Some(PathBuf::from("/some/cache")));
        assert_eq!(options.output_dim, None);
    }

    #[test]
    fn unset_limits_take_the_documented_defaults() {
        let options = text_options(&settings()).unwrap();
        assert_eq!(options.id, "Qwen/Qwen3-Embedding-0.6B");
        assert_eq!(options.max_length, DEFAULT_MAX_LENGTH);
        assert_eq!(options.max_batch_tokens, 4096);
        assert_eq!(options.device, DeviceSpec::Auto);
        assert_eq!(options.dtype, None);
        assert_eq!(options.attention, AttentionRequest::Auto);
    }

    #[test]
    fn the_provider_prefix_is_optional_and_a_bare_one_is_refused() {
        let mut settings = settings();
        settings.id = "org/model".to_string();
        assert_eq!(text_options(&settings).unwrap().id, "org/model");
        settings.id = "native:".to_string();
        assert!(matches!(
            text_options(&settings),
            Err(MinSyncError::Config(_))
        ));
    }

    #[test]
    fn bad_device_dtype_and_attention_are_config_errors() {
        for (device, dtype, attention) in [
            (Some("tpu"), None, None),
            (None, Some("int8"), None),
            (None, None, Some("flash")),
        ] {
            let mut settings = settings();
            settings.device = device.map(str::to_string);
            settings.dtype = dtype.map(str::to_string);
            settings.attention = attention.map(str::to_string);
            let error = text_options(&settings).unwrap_err();
            assert!(matches!(error, MinSyncError::Config(_)), "{error:?}");
        }
    }

    #[test]
    fn zero_timeouts_and_batch_sizes_are_refused_at_construction() {
        let mut settings = settings();
        settings.timeout_seconds = 0;
        assert!(NativeEmbedder::from_config(&settings).is_err());
    }

    /// The cache directory is `model_cache_dir`, then im-not-llm's
    /// `INL_CACHE_DIR`, then `<inl cache root>/models`: the same precedence
    /// `inl serve` uses, so MinSync and the daemon share one weight cache.
    #[test]
    fn cache_dir_precedence_is_config_then_inl_env_then_default() {
        // One test owns the env var: tests in a binary share the process.
        let saved = std::env::var_os(inl_core::device::CACHE_DIR_ENV);
        std::env::set_var(inl_core::device::CACHE_DIR_ENV, "/from/env");
        assert_eq!(model_cache_dir(Some("/cfg")), PathBuf::from("/cfg"));
        assert_eq!(model_cache_dir(None), PathBuf::from("/from/env"));
        std::env::remove_var(inl_core::device::CACHE_DIR_ENV);
        assert_eq!(
            model_cache_dir(None),
            inl_core::device::cache_root().join("models"),
            "default location"
        );
        if let Some(saved) = saved {
            std::env::set_var(inl_core::device::CACHE_DIR_ENV, saved);
        }
    }
}

/// Which engine runs a `native:` model.
///
/// The Apple Neural Engine (`inl-ane`) holds a model in a fraction of the
/// memory of the GPU path (measured on `inl serve`: KURE-Sparse-nano 354 MB vs
/// 1262 MB, EmbeddingGemma 2 233 MB vs 1995 MB, Qwen3 184 MB vs 2907 MB) at the
/// cost of fp16 graphs and a one-time compile of the Core ML package (minutes
/// for the dense models). So the sparse (SPLADE) encoder uses it by default on
/// Apple silicon, while a dense embedder uses it only when asked for with
/// `device = "ane"`: the default embedder must not start with a multi-minute
/// compile or change the vectors of existing indexes. `device = "metal"` /
/// `"cpu"` / `"cuda"` or a dtype other than `f16` keeps the candle path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Candle,
    Ane,
}

/// The engine `model` runs on, as `sparse` or dense, for these settings; an
/// error when `device = "ane"` was asked for but cannot be honoured.
pub fn resolve_backend(
    model: &str,
    sparse: bool,
    device: Option<&str>,
    dtype: Option<&str>,
) -> Result<Backend> {
    let package =
        inl_ane::known_package(model).filter(|package| package.family.is_sparse() == sparse);
    let fp16_ok = matches!(dtype, None | Some("auto" | "f16"));
    if device == Some("ane") {
        if !cfg!(target_os = "macos") {
            return Err(MinSyncError::Config(
                "device \"ane\" needs an Apple silicon Mac".to_string(),
            ));
        }
        if !fp16_ok {
            return Err(MinSyncError::Config(
                "device \"ane\" runs fp16 graphs: dtype must be auto or f16".to_string(),
            ));
        }
        if package.is_none() {
            return Err(MinSyncError::Config(format!(
                "no Apple Neural Engine package for {model}: device \"ane\" serves {}",
                inl_ane::KNOWN_PACKAGES
                    .iter()
                    .filter(|known| known.family.is_sparse() == sparse)
                    .map(|known| known.model)
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
        return Ok(Backend::Ane);
    }
    let by_default = sparse
        && cfg!(all(target_os = "macos", target_arch = "aarch64"))
        && matches!(device, None | Some("auto"))
        && fp16_ok
        && package.is_some_and(|package| package.default_on_apple);
    Ok(if by_default {
        Backend::Ane
    } else {
        Backend::Candle
    })
}

#[cfg(test)]
mod backend_tests {
    use super::*;

    const APPLE_SILICON: bool = cfg!(all(target_os = "macos", target_arch = "aarch64"));
    const QWEN: &str = "Qwen/Qwen3-Embedding-0.6B";
    const GEMMA: &str = "google/embeddinggemma-2";
    const KURE: &str = "nlpai-lab/KURE-Sparse-nano";

    fn pick(model: &str, sparse: bool, device: Option<&str>, dtype: Option<&str>) -> Backend {
        resolve_backend(model, sparse, device, dtype).unwrap()
    }

    #[test]
    fn apple_silicon_defaults_to_the_neural_engine_for_splade_only() {
        let ane_if_apple = if APPLE_SILICON {
            Backend::Ane
        } else {
            Backend::Candle
        };
        assert_eq!(pick(KURE, true, None, None), ane_if_apple);
        assert_eq!(pick(KURE, true, Some("auto"), Some("auto")), ane_if_apple);
        assert_eq!(pick(KURE, true, None, Some("f16")), ane_if_apple);
        // A dense embedder is opt-in: the default one must not begin with a
        // minutes-long Core ML compile nor change an existing index's vectors.
        assert_eq!(pick(QWEN, false, None, None), Backend::Candle);
        assert_eq!(pick(QWEN, false, None, Some("f16")), Backend::Candle);
        assert_eq!(pick(GEMMA, false, None, None), Backend::Candle);
    }

    #[test]
    fn an_explicit_other_device_or_dtype_keeps_the_candle_path() {
        for device in ["cpu", "metal", "cuda"] {
            assert_eq!(pick(KURE, true, Some(device), None), Backend::Candle);
        }
        for dtype in ["f32", "bf16"] {
            assert_eq!(pick(KURE, true, None, Some(dtype)), Backend::Candle);
        }
    }

    #[test]
    fn a_model_without_a_package_or_of_the_other_kind_stays_on_candle() {
        assert_eq!(pick("org/unknown", false, None, None), Backend::Candle);
        // A dense id is never a sparse package, and vice versa.
        assert_eq!(pick(QWEN, true, None, None), Backend::Candle);
        assert_eq!(pick(KURE, false, None, None), Backend::Candle);
        // A local checkpoint directory is not a known package either.
        assert_eq!(pick("/tmp/some/dir", true, None, None), Backend::Candle);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn device_ane_is_honoured_for_every_package_including_gemma2() {
        assert_eq!(pick(GEMMA, false, Some("ane"), None), Backend::Ane);
        assert_eq!(pick(KURE, true, Some("ane"), Some("f16")), Backend::Ane);
        assert_eq!(pick(QWEN, false, Some("ane"), Some("auto")), Backend::Ane);
    }

    #[test]
    fn device_ane_that_cannot_be_honoured_is_a_config_error_not_a_silent_fallback() {
        let bad_dtype = resolve_backend(QWEN, false, Some("ane"), Some("f32")).unwrap_err();
        let unknown = resolve_backend("org/unknown", false, Some("ane"), None).unwrap_err();
        let wrong_kind = resolve_backend(QWEN, true, Some("ane"), None).unwrap_err();
        for error in [bad_dtype, unknown, wrong_kind] {
            assert!(matches!(error, MinSyncError::Config(_)), "{error:?}");
        }
    }
}
