use crate::embedder::Embedder;
use crate::error::{MinSyncError, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

pub struct OpenAiEmbedder {
    model: String,
    api_key: String,
    batch_size: usize,
    client: reqwest::Client,
    base_url: String,
}

#[derive(Serialize)]
struct EmbedRequest {
    input: Vec<String>,
    model: String,
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedData>,
}

#[derive(Deserialize)]
struct EmbedData {
    embedding: Vec<f32>,
}

impl OpenAiEmbedder {
    pub fn new(model: &str, api_key: &str, batch_size: usize) -> Self {
        // Config stores the embedder id with a provider scheme (e.g.
        // "openai:text-embedding-3-small"). Strip the leading "openai:" prefix
        // so the actual OpenAI API model name is sent in the request.
        let model = model.strip_prefix("openai:").unwrap_or(model);
        Self {
            model: model.to_string(),
            api_key: api_key.to_string(),
            batch_size,
            client: reqwest::Client::new(),
            base_url: "https://api.openai.com".to_string(),
        }
    }

    pub fn with_base_url(mut self, url: &str) -> Self {
        self.base_url = url.to_string();
        self
    }

    pub fn from_env(model: &str, batch_size: usize) -> Result<Self> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| MinSyncError::Embedding("OPENAI_API_KEY not set".to_string()))?;
        Ok(Self::new(model, &api_key, batch_size))
    }
}

#[async_trait]
impl Embedder for OpenAiEmbedder {
    fn id(&self) -> &str {
        &self.model
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        if self.batch_size == 0 {
            return Err(MinSyncError::Embedding(
                "batch_size must be greater than 0".to_string(),
            ));
        }

        let mut all_embeddings = Vec::with_capacity(texts.len());

        for batch in texts.chunks(self.batch_size) {
            let request = EmbedRequest {
                input: batch.to_vec(),
                model: self.model.clone(),
            };

            let response = self
                .client
                .post(format!("{}/v1/embeddings", self.base_url))
                .bearer_auth(&self.api_key)
                .json(&request)
                .send()
                .await
                .map_err(|e| MinSyncError::Embedding(e.to_string()))?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                return Err(MinSyncError::Embedding(format!(
                    "OpenAI API error {}: {}",
                    status, body
                )));
            }

            let embed_response: EmbedResponse = response
                .json()
                .await
                .map_err(|e| MinSyncError::Embedding(e.to_string()))?;

            for data in embed_response.data {
                all_embeddings.push(data.embedding);
            }
        }

        Ok(all_embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    async fn setup_mock(n_embeddings: usize) -> (MockServer, String) {
        let server = MockServer::start().await;
        let data: Vec<_> = (0..n_embeddings)
            .map(|i| json!({"embedding": [0.1 * (i as f32 + 1.0), 0.2, 0.3], "index": i}))
            .collect();

        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .and(header("authorization", "Bearer test-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": data,
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 10, "total_tokens": 10}
            })))
            .mount(&server)
            .await;

        let uri = server.uri();
        (server, uri)
    }

    #[tokio::test]
    async fn test_embed_single() {
        let (_server, url) = setup_mock(1).await;
        let embedder =
            OpenAiEmbedder::new("text-embedding-3-small", "test-key", 10).with_base_url(&url);

        let embedding = embedder.embed_single("hello").await.unwrap();

        assert_eq!(embedding, vec![0.1, 0.2, 0.3]);
    }

    #[tokio::test]
    async fn test_embed_multiple() {
        let (_server, url) = setup_mock(3).await;
        let embedder =
            OpenAiEmbedder::new("text-embedding-3-small", "test-key", 10).with_base_url(&url);
        let texts = vec!["one".to_string(), "two".to_string(), "three".to_string()];

        let embeddings = embedder.embed(&texts).await.unwrap();

        assert_eq!(embeddings.len(), 3);
        assert_eq!(embeddings[0], vec![0.1, 0.2, 0.3]);
        assert_eq!(embeddings[1], vec![0.2, 0.2, 0.3]);
        assert_eq!(embeddings[2], vec![0.3, 0.2, 0.3]);
    }

    #[tokio::test]
    async fn test_embed_empty() {
        let embedder = OpenAiEmbedder::new("text-embedding-3-small", "test-key", 10);
        let texts: Vec<String> = Vec::new();

        let embeddings = embedder.embed(&texts).await.unwrap();

        assert!(embeddings.is_empty());
    }

    #[test]
    fn test_strips_openai_provider_prefix() {
        let embedder = OpenAiEmbedder::new("openai:text-embedding-3-small", "test-key", 1);
        assert_eq!(embedder.id(), "text-embedding-3-small");

        let bare = OpenAiEmbedder::new("text-embedding-3-small", "test-key", 1);
        assert_eq!(bare.id(), "text-embedding-3-small");
    }

    #[tokio::test]
    async fn test_embed_batch_splitting() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .and(header("authorization", "Bearer test-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [
                    {"embedding": [0.1, 0.2, 0.3], "index": 0},
                    {"embedding": [0.4, 0.5, 0.6], "index": 1}
                ],
                "model": "text-embedding-3-small"
            })))
            .expect(3)
            .mount(&server)
            .await;

        let embedder = OpenAiEmbedder::new("text-embedding-3-small", "test-key", 2)
            .with_base_url(&server.uri());
        let texts = vec![
            "one".to_string(),
            "two".to_string(),
            "three".to_string(),
            "four".to_string(),
            "five".to_string(),
        ];

        let embeddings = embedder.embed(&texts).await.unwrap();

        assert_eq!(embeddings.len(), 6);
    }

    #[tokio::test]
    async fn test_embed_api_error() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .and(header("authorization", "Bearer test-key"))
            .respond_with(ResponseTemplate::new(401).set_body_string("unauthorized"))
            .mount(&server)
            .await;

        let embedder = OpenAiEmbedder::new("text-embedding-3-small", "test-key", 10)
            .with_base_url(&server.uri());
        let texts = vec!["hello".to_string()];

        let error = embedder.embed(&texts).await.unwrap_err();

        assert!(matches!(error, MinSyncError::Embedding(_)));
        assert!(error.to_string().contains("OpenAI API error 401"));
    }

    #[tokio::test]
    async fn test_from_env_missing_key() {
        std::env::remove_var("OPENAI_API_KEY");

        let error = match OpenAiEmbedder::from_env("text-embedding-3-small", 10) {
            Ok(_) => panic!("expected missing OPENAI_API_KEY to return an error"),
            Err(error) => error,
        };

        assert!(matches!(error, MinSyncError::Embedding(_)));
        assert!(error.to_string().contains("OPENAI_API_KEY not set"));
    }

    #[tokio::test]
    async fn test_embedder_id() {
        let embedder = OpenAiEmbedder::new("text-embedding-3-small", "test-key", 10);

        assert_eq!(embedder.id(), "text-embedding-3-small");
    }
}
