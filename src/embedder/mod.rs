pub mod openai;
pub mod retry;
pub mod tei;

use crate::config::Config;
use crate::error::{MinSyncError, Result};
use async_trait::async_trait;

#[async_trait]
pub trait Embedder: Send + Sync {
    fn id(&self) -> &str;

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;

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

/// Construct an [`Embedder`] implementation from the embedder id in `config`.
///
/// Returns [`MinSyncError::Config`] when the id has no known provider prefix.
pub fn create_embedder(config: &Config) -> Result<Box<dyn Embedder>> {
    let settings = &config.embedder;
    let id = settings.id.as_str();
    let timeout = std::time::Duration::from_secs(settings.timeout_seconds);
    if id.starts_with("openai:") {
        let mut embedder = openai::OpenAiEmbedder::from_env(id, settings.batch_size)?
            .with_timeout(timeout)
            .with_max_retries(settings.max_retries)
            .with_max_concurrent(settings.max_concurrent);
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
            .with_passage_prefix(config.embedder.passage_prefix.clone());
        Ok(Box::new(embedder))
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
}
