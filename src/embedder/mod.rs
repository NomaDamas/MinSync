pub mod context_limit;
pub mod deadline;
pub mod hub;
pub mod native;
pub mod openai;
mod qwen3;
pub mod retry;
pub mod served;
pub mod session;
pub mod tei;

use crate::config::Config;
use crate::error::{MinSyncError, Result};
use async_trait::async_trait;

#[async_trait]
pub trait Embedder: Send + Sync {
    fn id(&self) -> &str;

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;

    async fn count_truncated(&self, _texts: &[String]) -> Result<usize> {
        Ok(0)
    }

    fn max_length(&self) -> Option<usize> {
        None
    }

    /// True for the reserved `none` embedder of a sparse-only workspace. Sync
    /// then stores a placeholder vector instead of calling [`Embedder::embed`],
    /// and `check` has no dense model to probe.
    fn is_sparse_only(&self) -> bool {
        false
    }

    /// True when re-running the same input on this embedder is guaranteed to
    /// produce the same output (in-process inference on a fixed device). A
    /// retry of an identical single-chunk request can then never help.
    fn is_deterministic(&self) -> bool {
        false
    }

    async fn embed_single(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.embed(&[text.to_string()]).await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| crate::error::MinSyncError::Embedding("empty response".to_string()))
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_single(text).await
    }
}

/// Defers provider construction until an embedding request is made.
///
/// Watch mode uses this only with `--watch-on-sync-error`, allowing the
/// watcher to remain alive while credentials or a local backend recover.
pub struct DeferredEmbedder {
    config: Config,
}

impl DeferredEmbedder {
    pub fn new(config: Config) -> Self {
        Self { config }
    }
}

/// The `none` embedder: sparse-only workspaces have no dense model. Sync and
/// check bypass it via [`Embedder::is_sparse_only`]; any other use (vector or
/// hybrid queries) fails with a pointer to `--mode sparse`.
pub struct SparseOnlyEmbedder;

#[async_trait]
impl Embedder for SparseOnlyEmbedder {
    fn id(&self) -> &str {
        crate::config::NONE_EMBEDDER_ID
    }

    fn is_sparse_only(&self) -> bool {
        true
    }

    async fn embed(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Err(MinSyncError::Embedding(
            "this workspace has no dense embedder (`embedder.id = \"none\"`), so it only \
             supports sparse search: use `--mode sparse`, or re-run `minsync init --force \
             --embedder <id>` to enable vector search"
                .to_string(),
        ))
    }
}

#[async_trait]
impl Embedder for DeferredEmbedder {
    fn id(&self) -> &str {
        self.config
            .embedder
            .id
            .split_once(':')
            .map_or(self.config.embedder.id.as_str(), |(_, id)| id)
    }

    fn is_deterministic(&self) -> bool {
        self.config.embedder.id.starts_with("native:")
    }

    fn is_sparse_only(&self) -> bool {
        self.config.is_sparse_only()
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        create_embedder(&self.config)?.embed(texts).await
    }
}

/// Construct an [`Embedder`] implementation from the embedder id in `config`.
///
/// Returns [`MinSyncError::Config`] when the id has no known provider prefix.
pub fn create_embedder(config: &Config) -> Result<Box<dyn Embedder>> {
    let settings = &config.embedder;
    let id = settings.id.as_str();
    if config.is_sparse_only() {
        return Ok(Box::new(SparseOnlyEmbedder));
    }
    let timeout = std::time::Duration::from_secs(settings.timeout_seconds);
    if id.starts_with("openai:") {
        let mut embedder = openai::OpenAiEmbedder::from_env(id, settings.batch_size)?
            .with_timeout(timeout)
            .with_max_retries(settings.max_retries)
            .with_max_concurrent(settings.max_concurrent)
            .with_max_length(settings.max_length);
        if let Some(base_url) = &settings.base_url {
            embedder = embedder.with_base_url(base_url);
        }
        Ok(Box::new(embedder))
    } else if id.starts_with("tei:") {
        let base_url = settings
            .base_url
            .clone()
            .unwrap_or_else(|| "http://localhost:8080".to_string());
        let embedder = tei::TeiEmbedder::new(id, &base_url, settings.batch_size)
            .with_timeout(timeout)
            .with_max_retries(settings.max_retries)
            .with_max_concurrent(settings.max_concurrent)
            .with_query_prefix(config.embedder.query_prefix.clone())
            .with_passage_prefix(config.embedder.passage_prefix.clone())
            .with_truncate(settings.truncate)
            .with_max_length(settings.max_length);
        Ok(Box::new(embedder))
    } else if id.starts_with("native:") {
        if served::server_disabled() {
            Ok(Box::new(native::NativeEmbedder::from_config(settings)?))
        } else {
            Ok(Box::new(served::ServedEmbedder::from_config(settings)?))
        }
    } else {
        Err(MinSyncError::Config(format!("unknown embedder id: {id}")))
    }
}

#[cfg(test)]
mod tests {
    use super::Embedder;
    use crate::error::Result;

    struct TestEmbedder;

    #[async_trait::async_trait]
    impl Embedder for TestEmbedder {
        fn id(&self) -> &str {
            "test"
        }

        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|text| vec![text.len() as f32, 1.0])
                .collect())
        }
    }

    #[tokio::test]
    async fn embed_query_defaults_to_embed_single() {
        let embedder = TestEmbedder;
        let single = embedder.embed_single("x").await.unwrap();
        let query = embedder.embed_query("x").await.unwrap();

        assert_eq!(query, single);
    }
}

#[cfg(test)]
mod factory_tests {
    use super::*;
    use crate::config::Config;

    #[test]
    fn test_create_embedder_tei() {
        let mut config = Config::default_for("12345678-1234-4234-9234-123456789abc");
        config.embedder.id = "tei:intfloat/multilingual-e5-small".to_string();
        config.embedder.base_url = Some("http://localhost:9999".to_string());
        config.embedder.query_prefix = Some("query: ".to_string());
        config.embedder.passage_prefix = Some("passage: ".to_string());

        let embedder = create_embedder(&config).expect("create tei embedder");
        assert_eq!(embedder.id(), "intfloat/multilingual-e5-small");
    }

    #[test]
    fn test_only_in_process_embedders_are_deterministic() {
        let mut config = Config::default_for("12345678-1234-4234-9234-123456789abc");
        config.embedder.id = "tei:foo".to_string();
        assert!(!create_embedder(&config).unwrap().is_deterministic());
        assert!(!DeferredEmbedder::new(config.clone()).is_deterministic());

        config.embedder.id = "native:Qwen/Qwen3-Embedding-0.6B".to_string();
        assert!(DeferredEmbedder::new(config).is_deterministic());
    }

    #[tokio::test]
    async fn test_create_embedder_none_is_sparse_only_and_refuses_to_embed() {
        let mut config = Config::default_for("12345678-1234-4234-9234-123456789abc");
        config.embedder.id = "none".to_string();

        let embedder = create_embedder(&config).expect("none needs no model or key");
        assert!(embedder.is_sparse_only());
        assert!(DeferredEmbedder::new(config.clone()).is_sparse_only());
        let message = embedder.embed_query("x").await.unwrap_err().to_string();
        assert!(message.contains("--mode sparse"), "{message}");

        config.embedder.id = "tei:foo".to_string();
        assert!(!create_embedder(&config).unwrap().is_sparse_only());
    }

    #[test]
    fn test_create_embedder_tei_defaults_base_url() {
        let mut config = Config::default_for("12345678-1234-4234-9234-123456789abc");
        config.embedder.id = "tei:foo".to_string();
        config.embedder.base_url = None;

        let embedder = create_embedder(&config).expect("create tei embedder with default base_url");
        assert_eq!(embedder.id(), "foo");
    }

    #[test]
    fn test_create_embedder_unknown() {
        let mut config = Config::default_for("12345678-1234-4234-9234-123456789abc");
        config.embedder.id = "bogus:foo".to_string();
        let result = create_embedder(&config);
        assert!(matches!(result, Err(MinSyncError::Config(_))));
    }

    #[tokio::test]
    async fn test_create_embedder_honors_max_retries_from_config() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embed"))
            .respond_with(ResponseTemplate::new(503).set_body_string("unavailable"))
            .mount(&server)
            .await;

        let mut config = Config::default_for("12345678-1234-4234-9234-123456789abc");
        config.embedder.id = "tei:foo".to_string();
        config.embedder.base_url = Some(server.uri());
        config.embedder.max_retries = 1;

        let embedder = create_embedder(&config).expect("create tei embedder");
        let error = embedder.embed(&["hello".to_string()]).await.unwrap_err();

        assert!(error.to_string().contains("retries exhausted"));
        assert_eq!(server.received_requests().await.unwrap().len(), 2);
    }

    #[test]
    fn test_create_embedder_openai_requires_key() {
        let mut config = Config::default_for("12345678-1234-4234-9234-123456789abc");
        config.embedder.id = "openai:text-embedding-3-small".to_string();

        std::env::set_var("OPENAI_API_KEY", "test");
        let result = create_embedder(&config);
        std::env::remove_var("OPENAI_API_KEY");

        let embedder = result.expect("create openai embedder with key set");
        assert_eq!(embedder.id(), "text-embedding-3-small");
    }

    #[test]
    fn test_create_embedder_native_constructs_without_http_or_key() {
        let mut config = Config::default_for("12345678-1234-4234-9234-123456789abc");
        config.embedder.id = "native:Qwen/Qwen3-Embedding-0.6B".to_string();
        config.embedder.base_url = None;

        let embedder =
            create_embedder(&config).expect("native embedder constructs without TEI or API key");
        assert_eq!(embedder.id(), "Qwen/Qwen3-Embedding-0.6B");
    }

    #[test]
    fn test_create_embedder_native_rejects_empty_model() {
        let mut config = Config::default_for("12345678-1234-4234-9234-123456789abc");
        config.embedder.id = "native:".to_string();

        let result = create_embedder(&config);
        assert!(matches!(result, Err(MinSyncError::Config(_))));
    }

    #[test]
    fn test_create_embedder_native_rejects_invalid_device() {
        let mut config = Config::default_for("12345678-1234-4234-9234-123456789abc");
        config.embedder.id = "native:Qwen/Qwen3-Embedding-0.6B".to_string();
        config.embedder.device = Some("tpu".to_string());

        let result = create_embedder(&config);
        assert!(matches!(result, Err(MinSyncError::Config(_))));
    }

    #[test]
    fn test_create_embedder_native_rejects_invalid_dtype() {
        let mut config = Config::default_for("12345678-1234-4234-9234-123456789abc");
        config.embedder.id = "native:Qwen/Qwen3-Embedding-0.6B".to_string();
        config.embedder.dtype = Some("fp64".to_string());

        let result = create_embedder(&config);
        assert!(matches!(result, Err(MinSyncError::Config(_))));
    }

    #[test]
    fn test_create_embedder_passes_configured_max_length_to_remote_embedders() {
        let mut config = Config::default_for("12345678-1234-4234-9234-123456789abc");
        config.embedder.id = "tei:acme/mystery-embed-v9".to_string();
        config.embedder.base_url = Some("http://localhost:9999".to_string());
        assert_eq!(create_embedder(&config).unwrap().max_length(), Some(8192));

        config.embedder.max_length = Some(384);
        assert_eq!(create_embedder(&config).unwrap().max_length(), Some(384));

        std::env::set_var("OPENAI_API_KEY", "test");
        config.embedder.id = "openai:acme/mystery-embed-v9".to_string();
        let embedder = create_embedder(&config);
        std::env::remove_var("OPENAI_API_KEY");
        assert_eq!(embedder.unwrap().max_length(), Some(384));
    }
}
