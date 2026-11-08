//! In-process native embedder: a thin adapter over [`inl_text::TextEmbedder`].
//!
//! The engine — device/dtype selection, batching, the padded-token cap, the
//! per-batch deadline and the models themselves — lives in im-not-llm. This
//! module maps MinSync's [`EmbedderConfig`] onto [`TextOptions`] and exposes the
//! engine through MinSync's [`Embedder`] trait, keeping MinSync's semantics:
//! the configured passage/query prefixes, the configured timeouts, and
//! `count_truncated` / `is_deterministic` as the CLI relies on them.

use crate::config::EmbedderConfig;
use crate::embedder::Embedder;
use crate::error::{MinSyncError, Result};
use async_trait::async_trait;
use inl_core::attention::AttentionRequest;
use inl_core::device::{DeviceSpec, DtypeSpec};
use inl_text::{TextEmbedder, TextOptions};
use std::path::PathBuf;
use std::sync::Arc;

/// Model context length in tokens when `[embedder].max_length` is unset.
pub(crate) const DEFAULT_MAX_LENGTH: usize = 2048;

/// Per-pass deadline on CPU when `per_batch_timeout_seconds` is not set (the
/// engine applies it; 900 s).
pub(crate) use inl_core::device::CPU_BATCH_TIMEOUT;

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

/// MinSync's view of a native model id: the hub repo id or local directory that
/// follows the `native:` provider prefix.
fn parse_native_model_id(id: &str) -> Result<&str> {
    let model = id.strip_prefix("native:").unwrap_or(id);
    if model.is_empty() {
        return Err(MinSyncError::Config(
            "native embedder id is missing a model name".to_string(),
        ));
    }
    Ok(model)
}

/// The engine options `settings` describes: the one place MinSync's
/// [`EmbedderConfig`] is translated into the engine's [`TextOptions`].
///
/// Unset keys keep the engine defaults, which match the CLI defaults (context
/// 2048, 64 inputs per pass, 4096 padded tokens, 60 s per pass). `device`,
/// `dtype` and `attention` are validated here so an unusable setting is a
/// [`MinSyncError::Config`] before any model is loaded.
pub fn text_options(settings: &EmbedderConfig) -> Result<TextOptions> {
    let mut options = TextOptions::default().with_id(parse_native_model_id(&settings.id)?);
    options.device = DeviceSpec::parse(settings.device.as_deref())?;
    options.dtype = DtypeSpec::parse(settings.dtype.as_deref())?;
    options.attention = AttentionRequest::parse(settings.attention.as_deref())?;
    options.max_length = settings.max_length.unwrap_or(DEFAULT_MAX_LENGTH);
    options.batch_size = settings.batch_size;
    if let Some(tokens) = settings.max_batch_tokens {
        options.max_batch_tokens = tokens;
    }
    options.timeout_seconds = settings.timeout_seconds;
    options.per_batch_timeout_seconds = settings.per_batch_timeout_seconds;
    options.cache_dir = settings.model_cache_dir.as_ref().map(PathBuf::from);
    options.query_prefix = settings.query_prefix.clone();
    options.passage_prefix = settings.passage_prefix.clone();
    Ok(options)
}

/// MinSync's dense native embedder. Inference is [`TextEmbedder`]; this wrapper
/// only maps config and errors, and reports the identity MinSync records.
pub struct NativeEmbedder {
    model_id: String,
    max_length: usize,
    inner: Arc<TextEmbedder>,
}

impl NativeEmbedder {
    pub fn from_config(settings: &EmbedderConfig) -> Result<Self> {
        let model_id = parse_native_model_id(&settings.id)?.to_string();
        let options = text_options(settings)?;
        let max_length = options.max_length;
        Ok(Self {
            model_id,
            max_length,
            inner: Arc::new(TextEmbedder::new(options)?),
        })
    }

    /// The engine behind this embedder, for `minsync serve` (`inl_server`
    /// drives engines through [`inl_core::DenseEngine`]).
    pub fn engine(&self) -> Arc<dyn inl_core::DenseEngine> {
        let engine: Arc<dyn inl_core::DenseEngine> = self.inner.clone();
        engine
    }

    /// The embedding space this embedder produces. Loads the model if it is
    /// not resident, because the device, dtype and dimension are only known
    /// once it is.
    pub async fn identity(&self) -> Result<NativeIdentity> {
        let identity = self.inner.identity().await?;
        Ok(NativeIdentity {
            id: format!("native:{}", self.model_id),
            dimension: identity.dimension,
            dtype: identity.dtype,
            device: identity.device,
            max_length: identity.max_length,
        })
    }

    /// Release the model weights. The next embed reloads them. True when a
    /// model was resident.
    pub async fn unload(&self) -> bool {
        self.inner.unload().await
    }

    /// True while the model weights are resident.
    pub fn is_loaded(&self) -> bool {
        self.inner.is_loaded()
    }
}

#[async_trait]
impl Embedder for NativeEmbedder {
    fn id(&self) -> &str {
        &self.model_id
    }

    async fn count_truncated(&self, texts: &[String]) -> Result<usize> {
        Ok(self.inner.count_truncated(texts).await?)
    }

    fn max_length(&self) -> Option<usize> {
        Some(self.max_length)
    }

    fn is_deterministic(&self) -> bool {
        true
    }

    /// Embed passages: the engine applies the configured passage prefix.
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(self.inner.embed(texts).await?)
    }

    /// Embed a search query: the engine applies the configured query prefix.
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        Ok(self.inner.embed_query(text).await?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn settings() -> EmbedderConfig {
        let mut config = Config::default_for("12345678-1234-4234-9234-123456789abc");
        config.embedder.id = "native:Qwen/Qwen3-Embedding-0.6B".to_string();
        config.embedder
    }

    #[test]
    fn config_maps_onto_the_engine_options() {
        let mut settings = settings();
        settings.device = Some("cpu".to_string());
        settings.dtype = Some("f32".to_string());
        settings.attention = Some("eager".to_string());
        settings.max_length = Some(512);
        settings.max_batch_tokens = Some(128);
        settings.timeout_seconds = 42;
        settings.per_batch_timeout_seconds = Some(3);
        settings.query_prefix = Some("q: ".to_string());
        settings.passage_prefix = Some("p: ".to_string());
        settings.model_cache_dir = Some("/tmp/models".to_string());

        let options = text_options(&settings).expect("map options");
        assert_eq!(options.id, "Qwen/Qwen3-Embedding-0.6B");
        assert_eq!(options.device, DeviceSpec::Cpu);
        assert_eq!(options.dtype, Some(DtypeSpec::F32));
        assert_eq!(options.attention, AttentionRequest::Eager);
        assert_eq!(options.max_length, 512);
        assert_eq!(options.batch_size, settings.batch_size);
        assert_eq!(options.max_batch_tokens, 128);
        assert_eq!(options.timeout_seconds, 42);
        assert_eq!(options.per_batch_timeout_seconds, Some(3));
        assert_eq!(options.query_prefix.as_deref(), Some("q: "));
        assert_eq!(options.passage_prefix.as_deref(), Some("p: "));
        assert_eq!(
            options.cache_dir.as_deref(),
            Some(std::path::Path::new("/tmp/models"))
        );
    }

    #[test]
    fn unset_keys_keep_the_cli_defaults() {
        let options = text_options(&settings()).expect("map options");
        assert_eq!(options.max_length, DEFAULT_MAX_LENGTH);
        assert_eq!(options.batch_size, 64);
        assert_eq!(options.max_batch_tokens, 4096);
        assert_eq!(options.timeout_seconds, 60);
        assert_eq!(options.per_batch_timeout_seconds, None);
        assert_eq!(options.device, DeviceSpec::Auto);
        assert_eq!(options.dtype, None);
    }

    #[test]
    fn unusable_settings_are_config_errors() {
        for tweak in [
            (|s: &mut EmbedderConfig| s.id = "native:".to_string()) as fn(&mut EmbedderConfig),
            |s: &mut EmbedderConfig| s.device = Some("tpu".to_string()),
            |s: &mut EmbedderConfig| s.dtype = Some("int8".to_string()),
            |s: &mut EmbedderConfig| s.attention = Some("psychic".to_string()),
            |s: &mut EmbedderConfig| s.timeout_seconds = 0,
            |s: &mut EmbedderConfig| s.per_batch_timeout_seconds = Some(0),
        ] {
            let mut settings = settings();
            tweak(&mut settings);
            let error = NativeEmbedder::from_config(&settings)
                .err()
                .unwrap_or_else(|| panic!("{settings:?} must be rejected"));
            assert!(matches!(error, MinSyncError::Config(_)), "{error}");
        }
    }

    #[test]
    fn identity_is_the_id_as_written_in_config() {
        let settings = settings();
        let embedder = NativeEmbedder::from_config(&settings).expect("build embedder");
        assert_eq!(embedder.id(), "Qwen/Qwen3-Embedding-0.6B");
        assert_eq!(embedder.max_length(), Some(DEFAULT_MAX_LENGTH));
        assert!(embedder.is_deterministic());
    }
}
