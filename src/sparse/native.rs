//! In-process native sparse encoder: a thin adapter over
//! [`inl_sparse::SparseEncoder`] (KURE-Sparse-nano: ModernBERT MLM + SPLADE
//! pooling on candle).
//!
//! The engine — device/dtype selection, batching, the per-batch deadline and
//! the model — lives in im-not-llm. This module maps MinSync's [`SparseConfig`]
//! onto [`SparseOptions`] and exposes the engine through MinSync's
//! [`SparseEncoder`](super::SparseEncoder) trait, keeping the configured
//! `[Q] `/`[D] ` markers, timeouts and `count_truncated` semantics.

use super::{SparseEncoder, SparseVector};
use crate::config::{SparseConfig, DEFAULT_SPARSE_MAX_LENGTH};
use crate::error::{MinSyncError, Result};
use async_trait::async_trait;
use inl_sparse::SparseOptions;
use std::path::PathBuf;
use std::sync::Arc;

/// MinSync's view of a native sparse model id: the hub repo id or local
/// directory that follows the `native:` provider prefix.
fn parse_native_model_id(id: &str) -> Result<&str> {
    let model = id.strip_prefix("native:").unwrap_or(id);
    if model.is_empty() {
        return Err(MinSyncError::Config(
            "native sparse encoder id is missing a model name".to_string(),
        ));
    }
    Ok(model)
}

/// The engine options `settings` describes: the one place MinSync's
/// [`SparseConfig`] is translated into the engine's [`SparseOptions`].
///
/// Unset keys keep the engine defaults, which match the CLI defaults (context
/// 2048, 64 inputs per pass, 1024 padded tokens, 60 s per pass, `[Q] `/`[D] `
/// markers). `device` and `dtype` are validated by the engine when the encoder
/// is built, so an unusable setting is a [`MinSyncError::Config`].
pub fn sparse_options(settings: &SparseConfig) -> Result<SparseOptions> {
    let mut options = SparseOptions::default().with_id(parse_native_model_id(&settings.id)?);
    options.device = settings.device.clone();
    options.dtype = settings.dtype.clone();
    options.max_length = settings.max_length.unwrap_or(DEFAULT_SPARSE_MAX_LENGTH);
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

/// What a host needs to record which sparse space the vectors belong to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseIdentity {
    /// The encoder id as written in `config.toml`, e.g.
    /// `native:nlpai-lab/KURE-Sparse-nano`.
    pub id: String,
    /// Number of terms in the output space.
    pub dimension: usize,
    pub dtype: String,
    pub device: String,
    pub max_length: usize,
}

/// MinSync's sparse native encoder. Encoding is
/// [`inl_sparse::SparseEncoder`]; this wrapper only maps config and errors, and
/// reports the identity MinSync records.
pub struct NativeSparseEncoder {
    model_id: String,
    max_length: usize,
    inner: Arc<inl_sparse::SparseEncoder>,
}

impl NativeSparseEncoder {
    pub fn from_config(settings: &SparseConfig) -> Result<Self> {
        let model_id = parse_native_model_id(&settings.id)?.to_string();
        let options = sparse_options(settings)?;
        let max_length = options.max_length;
        Ok(Self {
            model_id,
            max_length,
            inner: Arc::new(inl_sparse::SparseEncoder::new(options)?),
        })
    }

    /// The engine behind this encoder, for `minsync serve --sparse`
    /// (`inl_server` drives engines through [`inl_sparse::SparseEngine`]).
    pub fn engine(&self) -> Arc<dyn inl_sparse::SparseEngine> {
        let engine: Arc<dyn inl_sparse::SparseEngine> = self.inner.clone();
        engine
    }

    /// The sparse space this encoder produces. Loads the model if needed.
    pub async fn identity(&self) -> Result<SparseIdentity> {
        let identity = self.inner.identity().await?;
        Ok(SparseIdentity {
            id: format!("native:{}", self.model_id),
            dimension: identity.dimension,
            dtype: identity.dtype,
            device: identity.device,
            max_length: identity.max_length,
        })
    }

    /// Release the model weights; the next call reloads them.
    pub async fn unload(&self) -> bool {
        self.inner.unload().await
    }

    pub fn is_loaded(&self) -> bool {
        self.inner.is_loaded()
    }
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
        Ok(self.inner.count_truncated(texts).await?)
    }

    /// Encode passages: the engine applies the configured passage marker.
    async fn encode(&self, texts: &[String]) -> Result<Vec<SparseVector>> {
        Ok(self.inner.encode(texts).await?)
    }

    /// Encode one search query: the engine applies the configured query marker.
    async fn encode_query(&self, text: &str) -> Result<SparseVector> {
        Ok(self.inner.encode_query(text).await?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SparseConfig;

    fn settings() -> SparseConfig {
        SparseConfig::default()
    }

    #[test]
    fn config_maps_onto_the_engine_options() {
        let mut settings = settings();
        settings.device = Some("cpu".to_string());
        settings.dtype = Some("f32".to_string());
        settings.max_length = Some(512);
        settings.max_batch_tokens = Some(128);
        settings.timeout_seconds = 42;
        settings.per_batch_timeout_seconds = Some(3);
        settings.query_prefix = Some("Q ".to_string());
        settings.passage_prefix = Some("D ".to_string());
        settings.model_cache_dir = Some("/tmp/models".to_string());

        let options = sparse_options(&settings).expect("map options");
        assert_eq!(options.id, "nlpai-lab/KURE-Sparse-nano");
        assert_eq!(options.device.as_deref(), Some("cpu"));
        assert_eq!(options.dtype.as_deref(), Some("f32"));
        assert_eq!(options.max_length, 512);
        assert_eq!(options.batch_size, settings.batch_size);
        assert_eq!(options.max_batch_tokens, 128);
        assert_eq!(options.timeout_seconds, 42);
        assert_eq!(options.per_batch_timeout_seconds, Some(3));
        assert_eq!(options.query_prefix.as_deref(), Some("Q "));
        assert_eq!(options.passage_prefix.as_deref(), Some("D "));
        assert_eq!(
            options.cache_dir.as_deref(),
            Some(std::path::Path::new("/tmp/models"))
        );
    }

    #[test]
    fn unset_keys_keep_the_cli_defaults() {
        let options = sparse_options(&settings()).expect("map options");
        assert_eq!(options.max_length, DEFAULT_SPARSE_MAX_LENGTH);
        assert_eq!(options.batch_size, 64);
        assert_eq!(options.max_batch_tokens, 1024);
        assert_eq!(options.timeout_seconds, 60);
        assert_eq!(options.per_batch_timeout_seconds, None);
        assert_eq!(options.device, None);
        assert_eq!(options.dtype, None);
        assert_eq!(options.query_prefix.as_deref(), Some("[Q] "));
        assert_eq!(options.passage_prefix.as_deref(), Some("[D] "));
    }

    #[test]
    fn unusable_settings_are_config_errors() {
        for tweak in [
            (|s: &mut SparseConfig| s.id = "native:".to_string()) as fn(&mut SparseConfig),
            |s: &mut SparseConfig| s.batch_size = 0,
            |s: &mut SparseConfig| s.timeout_seconds = 0,
            |s: &mut SparseConfig| s.per_batch_timeout_seconds = Some(0),
            |s: &mut SparseConfig| s.device = Some("tpu".to_string()),
            |s: &mut SparseConfig| s.dtype = Some("int4".to_string()),
        ] {
            let mut settings = settings();
            tweak(&mut settings);
            let error = NativeSparseEncoder::from_config(&settings)
                .err()
                .unwrap_or_else(|| panic!("{settings:?} must be rejected"));
            assert!(matches!(error, MinSyncError::Config(_)), "{error}");
        }
    }
}
