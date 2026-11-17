//! The `native:` dense embedder. All inference (device and dtype selection,
//! batching, the per-batch deadline, the models themselves) is
//! [`inl_text::TextEmbedder`] from `im-not-llm`; this module only maps
//! `[embedder]` settings onto its options and its answers onto [`Embedder`].

use crate::config::EmbedderConfig;
use crate::embedder::Embedder;
use crate::error::{MinSyncError, Result};
use async_trait::async_trait;
use inl_core::attention::AttentionRequest;
use inl_core::device::{DeviceSpec, DtypeSpec};
use inl_text::{TextEmbedder, TextOptions};
use std::path::PathBuf;

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
    /// Resolved device: `cpu`, `metal` or `cuda`.
    pub device: &'static str,
    pub max_length: usize,
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

/// `~/.cache/minsync` (`%LOCALAPPDATA%\minsync` on Windows).
pub fn minsync_cache_root() -> PathBuf {
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

/// Where model files are cached: the configured `model_cache_dir`, else
/// `FASTEMBED_CACHE_DIR` (the name earlier releases documented), else
/// `~/.cache/minsync/models`. `HF_HOME`, when set, still wins inside the hub
/// client, as it always has.
pub(crate) fn model_cache_dir(configured: Option<&str>) -> PathBuf {
    match configured {
        Some(dir) => PathBuf::from(dir),
        None => std::env::var_os("FASTEMBED_CACHE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| minsync_cache_root().join("models")),
    }
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

pub struct NativeEmbedder {
    model_id: String,
    max_length: usize,
    engine: TextEmbedder,
}

impl NativeEmbedder {
    pub fn from_config(settings: &EmbedderConfig) -> Result<Self> {
        let options = text_options(settings)?;
        Ok(Self {
            model_id: options.id.clone(),
            max_length: options.max_length,
            engine: TextEmbedder::new(options)?,
        })
    }

    /// The embedding space this embedder produces. Loads the model if it is
    /// not resident, because the device, dtype and dimension are only known
    /// once it is.
    pub async fn identity(&self) -> Result<NativeIdentity> {
        Ok(self.engine.identity().await?.into())
    }

    /// The engine, for a server that drives it directly.
    pub fn into_engine(self) -> TextEmbedder {
        self.engine
    }
}

#[async_trait]
impl Embedder for NativeEmbedder {
    fn id(&self) -> &str {
        &self.model_id
    }

    async fn count_truncated(&self, texts: &[String]) -> Result<usize> {
        Ok(self.engine.count_truncated(texts).await?)
    }

    fn max_length(&self) -> Option<usize> {
        Some(self.max_length)
    }

    fn is_deterministic(&self) -> bool {
        true
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(self.engine.embed(texts).await?)
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        Ok(self.engine.embed_query(text).await?)
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

    /// The cache directory is `model_cache_dir`, then `FASTEMBED_CACHE_DIR`,
    /// then `~/.cache/minsync/models`: the order earlier releases documented.
    #[test]
    fn cache_dir_precedence_is_config_then_env_then_default() {
        // One test owns the env var: tests in a binary share the process.
        let saved = std::env::var_os("FASTEMBED_CACHE_DIR");
        std::env::set_var("FASTEMBED_CACHE_DIR", "/from/env");
        assert_eq!(model_cache_dir(Some("/cfg")), PathBuf::from("/cfg"));
        assert_eq!(model_cache_dir(None), PathBuf::from("/from/env"));
        std::env::remove_var("FASTEMBED_CACHE_DIR");
        assert_eq!(
            model_cache_dir(None),
            minsync_cache_root().join("models"),
            "default location"
        );
        if let Some(saved) = saved {
            std::env::set_var("FASTEMBED_CACHE_DIR", saved);
        }
    }
}
