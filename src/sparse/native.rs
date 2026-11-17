//! The `native:` sparse encoder. The model (ModernBERT masked LM with SPLADE
//! pooling), its tokenizer, batching and the per-batch deadline are
//! [`inl_sparse::SparseEncoder`] from `im-not-llm`; this module only maps
//! `[sparse]` settings onto its options and its answers onto
//! [`SparseEncoder`].

use super::{SparseEncoder, SparseVector};
use crate::config::{SparseConfig, DEFAULT_SPARSE_MAX_LENGTH};
use crate::embedder::native::{model_cache_dir, parse_native_model_id};
use crate::error::Result;
use async_trait::async_trait;
use inl_core::EngineInfo;
use inl_sparse::SparseOptions;

/// The engine options `settings` describes.
pub fn sparse_options(settings: &SparseConfig) -> Result<SparseOptions> {
    let defaults = SparseOptions::default();
    Ok(SparseOptions {
        id: parse_native_model_id(&settings.id)?.to_string(),
        device: settings.device.clone(),
        dtype: settings.dtype.clone(),
        max_length: settings.max_length.unwrap_or(DEFAULT_SPARSE_MAX_LENGTH),
        batch_size: settings.batch_size,
        max_batch_tokens: settings
            .max_batch_tokens
            .unwrap_or(defaults.max_batch_tokens),
        timeout_seconds: settings.timeout_seconds,
        per_batch_timeout_seconds: settings.per_batch_timeout_seconds,
        cache_dir: Some(model_cache_dir(settings.model_cache_dir.as_deref())),
        query_prefix: settings.query_prefix.clone(),
        passage_prefix: settings.passage_prefix.clone(),
    })
}

pub struct NativeSparseEncoder {
    model_id: String,
    max_length: usize,
    engine: inl_sparse::SparseEncoder,
}

impl NativeSparseEncoder {
    pub fn from_config(settings: &SparseConfig) -> Result<Self> {
        let options = sparse_options(settings)?;
        Ok(Self {
            model_id: options.id.clone(),
            max_length: options.max_length,
            engine: inl_sparse::SparseEncoder::new(options)?,
        })
    }

    /// The sparse space this encoder produces (`dimension` is the number of
    /// terms). Loads the model if it is not resident.
    pub async fn identity(&self) -> Result<EngineInfo> {
        Ok(self.engine.identity().await?)
    }

    /// The engine, for a server that drives it directly.
    pub fn into_engine(self) -> inl_sparse::SparseEncoder {
        self.engine
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
        Ok(self.engine.count_truncated(texts).await?)
    }

    async fn encode(&self, texts: &[String]) -> Result<Vec<SparseVector>> {
        Ok(self.engine.encode(texts).await?)
    }

    async fn encode_query(&self, text: &str) -> Result<SparseVector> {
        Ok(self.engine.encode_query(text).await?)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::error::MinSyncError;
    use std::path::{Path, PathBuf};

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

    #[test]
    fn settings_map_onto_engine_options() {
        let settings = SparseConfig {
            id: "native:org/splade".to_string(),
            device: Some("cpu".to_string()),
            dtype: Some("f16".to_string()),
            max_length: Some(256),
            batch_size: 7,
            max_batch_tokens: Some(333),
            timeout_seconds: 9,
            per_batch_timeout_seconds: Some(4),
            query_prefix: Some("[Q] ".to_string()),
            passage_prefix: Some("[D] ".to_string()),
            model_cache_dir: Some("/some/cache".to_string()),
            ..SparseConfig::default()
        };
        let options = sparse_options(&settings).unwrap();
        assert_eq!(options.id, "org/splade");
        assert_eq!(options.device.as_deref(), Some("cpu"));
        assert_eq!(options.dtype.as_deref(), Some("f16"));
        assert_eq!(options.max_length, 256);
        assert_eq!(options.batch_size, 7);
        assert_eq!(options.max_batch_tokens, 333);
        assert_eq!(options.timeout_seconds, 9);
        assert_eq!(options.per_batch_timeout_seconds, Some(4));
        assert_eq!(options.query_prefix.as_deref(), Some("[Q] "));
        assert_eq!(options.passage_prefix.as_deref(), Some("[D] "));
        assert_eq!(options.cache_dir, Some(PathBuf::from("/some/cache")));
    }

    #[test]
    fn unset_limits_take_the_documented_defaults() {
        let options = sparse_options(&SparseConfig::default()).unwrap();
        assert_eq!(options.id, "nlpai-lab/KURE-Sparse-nano");
        assert_eq!(options.max_length, DEFAULT_SPARSE_MAX_LENGTH);
        assert_eq!(options.max_batch_tokens, 1024);
        assert_eq!(options.query_prefix.as_deref(), Some("[Q] "));
        assert_eq!(options.passage_prefix.as_deref(), Some("[D] "));
    }

    #[test]
    fn invalid_settings_are_config_errors_at_construction() {
        for settings in [
            SparseConfig {
                id: "native:".to_string(),
                ..tiny_settings()
            },
            SparseConfig {
                device: Some("tpu".to_string()),
                ..tiny_settings()
            },
            SparseConfig {
                dtype: Some("int8".to_string()),
                ..tiny_settings()
            },
            SparseConfig {
                batch_size: 0,
                ..tiny_settings()
            },
            SparseConfig {
                timeout_seconds: 0,
                ..tiny_settings()
            },
        ] {
            let error = NativeSparseEncoder::from_config(&settings)
                .err()
                .unwrap_or_else(|| panic!("{settings:?} must be refused"));
            assert!(matches!(error, MinSyncError::Config(_)), "{error:?}");
        }
    }

    /// The tiny checkpoint through the adapter, as `minsync sync` drives it:
    /// the identity is the bare model id and the vectors satisfy the
    /// SparseVector invariants the store relies on.
    #[tokio::test]
    async fn the_adapter_encodes_through_the_engine() {
        let encoder = NativeSparseEncoder::from_config(&tiny_settings()).unwrap();
        assert_eq!(encoder.id(), tiny_dir().display().to_string());
        assert!(encoder.is_deterministic());
        assert_eq!(encoder.max_length(), Some(DEFAULT_SPARSE_MAX_LENGTH));
        let info = encoder.identity().await.unwrap();
        assert_eq!((info.dtype.as_str(), info.device.as_str()), ("f32", "cpu"));

        let texts = vec!["w0 w1 w2".to_string(), "w3 w4 w5 w6".to_string()];
        let docs = encoder.encode(&texts).await.unwrap();
        assert_eq!(docs.len(), 2);
        assert!(docs.iter().all(|doc| !doc.is_empty()));
        let query = encoder.encode_query("w0 w1").await.unwrap();
        assert!(!query.is_empty());
        assert!(query.dot(&docs[0]) > 0.0);
        assert_eq!(encoder.count_truncated(&texts).await.unwrap(), 0);
    }

    /// A load failure is an embedding error (exit 5), not a panic or config.
    #[tokio::test]
    async fn a_missing_checkpoint_is_an_embedding_error() {
        let empty = tempfile::tempdir().unwrap();
        let settings = SparseConfig {
            id: format!("native:{}", empty.path().display()),
            model_cache_dir: Some(tempfile::tempdir().unwrap().path().display().to_string()),
            ..tiny_settings()
        };
        let encoder = NativeSparseEncoder::from_config(&settings).unwrap();
        let error = encoder.encode(&["x".to_string()]).await.unwrap_err();
        assert!(matches!(error, MinSyncError::Embedding(_)), "{error:?}");
    }
}
