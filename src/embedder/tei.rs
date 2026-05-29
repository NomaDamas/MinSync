use crate::embedder::Embedder;
use crate::error::{MinSyncError, Result};
use async_trait::async_trait;
use serde::Serialize;

pub struct TeiEmbedder {
    model: String,
    base_url: String,
    batch_size: usize,
    query_prefix: Option<String>,
    passage_prefix: Option<String>,
    client: reqwest::Client,
}

#[derive(Serialize)]
struct EmbedRequest {
    inputs: Vec<String>,
    normalize: bool,
    truncate: Option<()>,
}

impl TeiEmbedder {
    pub fn new(model: &str, base_url: &str, batch_size: usize) -> Self {
        // Config stores the embedder id with a provider scheme (e.g.
        // "tei:intfloat/multilingual-e5-small"). Strip the leading "tei:"
        // prefix so the informational model id matches the actual TEI model.
        let model = model.strip_prefix("tei:").unwrap_or(model);
        Self {
            model: model.to_string(),
            base_url: base_url.trim_end_matches('/').to_string(),
            batch_size,
            query_prefix: None,
            passage_prefix: None,
            client: reqwest::Client::new(),
        }
    }

    pub fn with_query_prefix(mut self, p: Option<String>) -> Self {
        self.query_prefix = p;
        self
    }

    pub fn with_passage_prefix(mut self, p: Option<String>) -> Self {
        self.passage_prefix = p;
        self
    }

    async fn embed_inputs(&self, inputs: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let expected_count = inputs.len();
        let request = EmbedRequest {
            inputs,
            normalize: true,
            truncate: None,
        };

        let response = self
            .client
            .post(format!("{}/embed", self.base_url))
            .json(&request)
            .send()
            .await
            .map_err(|e| MinSyncError::Embedding(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(MinSyncError::Embedding(format!(
                "TEI API error {}: {}",
                status, body
            )));
        }

        let embeddings: Vec<Vec<f32>> = response
            .json()
            .await
            .map_err(|e| MinSyncError::Embedding(e.to_string()))?;

        if embeddings.len() != expected_count {
            return Err(MinSyncError::Embedding(format!(
                "TEI API returned {} embeddings for {} inputs",
                embeddings.len(),
                expected_count
            )));
        }

        Ok(embeddings)
    }
}

#[async_trait]
impl Embedder for TeiEmbedder {
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
            let inputs: Vec<String> = batch
                .iter()
                .map(|text| match &self.passage_prefix {
                    Some(prefix) => format!("{}{}", prefix, text),
                    None => text.clone(),
                })
                .collect();

            all_embeddings.extend(self.embed_inputs(inputs).await?);
        }

        Ok(all_embeddings)
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let input = match &self.query_prefix {
            Some(prefix) => format!("{}{}", prefix, text),
            None => text.to_string(),
        };
        let mut embeddings = self.embed_inputs(vec![input]).await?;
        embeddings
            .pop()
            .ok_or_else(|| MinSyncError::Embedding("empty response".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::matchers::{body_json, body_string_contains, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn test_embed_single_and_multi() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/embed"))
            .and(body_json(json!({
                "inputs": ["a", "b"],
                "normalize": true,
                "truncate": null
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!([
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6]
            ])))
            .mount(&server)
            .await;

        let embedder = TeiEmbedder::new("intfloat/multilingual-e5-small", &server.uri(), 10);
        let texts = vec!["a".to_string(), "b".to_string()];

        let embeddings = embedder.embed(&texts).await.unwrap();

        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0], vec![0.1, 0.2, 0.3]);
        assert_eq!(embeddings[1], vec![0.4, 0.5, 0.6]);
    }

    #[tokio::test]
    async fn test_embed_empty_returns_empty() {
        let server = MockServer::start().await;
        let embedder = TeiEmbedder::new("intfloat/multilingual-e5-small", &server.uri(), 10);
        let texts: Vec<String> = Vec::new();

        let embeddings = embedder.embed(&texts).await.unwrap();

        assert!(embeddings.is_empty());
        assert!(server.received_requests().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_batch_splitting() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/embed"))
            .and(body_json(json!({
                "inputs": ["one", "two"],
                "normalize": true,
                "truncate": null
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!([[0.1], [0.2]])))
            .expect(1)
            .mount(&server)
            .await;

        Mock::given(method("POST"))
            .and(path("/embed"))
            .and(body_json(json!({
                "inputs": ["three"],
                "normalize": true,
                "truncate": null
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!([[0.3]])))
            .expect(1)
            .mount(&server)
            .await;

        let embedder = TeiEmbedder::new("intfloat/multilingual-e5-small", &server.uri(), 2);
        let texts = vec!["one".to_string(), "two".to_string(), "three".to_string()];

        let embeddings = embedder.embed(&texts).await.unwrap();

        assert_eq!(embeddings, vec![vec![0.1], vec![0.2], vec![0.3]]);
    }

    #[tokio::test]
    async fn test_non_2xx_errors() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/embed"))
            .respond_with(ResponseTemplate::new(500).set_body_string("server error"))
            .mount(&server)
            .await;

        let embedder = TeiEmbedder::new("intfloat/multilingual-e5-small", &server.uri(), 10);
        let texts = vec!["hello".to_string()];

        let error = embedder.embed(&texts).await.unwrap_err();

        assert!(matches!(error, MinSyncError::Embedding(_)));
        assert!(error.to_string().contains("TEI API error 500"));
        assert!(error.to_string().contains("server error"));
    }

    #[tokio::test]
    async fn test_passage_prefix_applied() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/embed"))
            .and(body_string_contains("passage: doc"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!([[0.1, 0.2]])))
            .mount(&server)
            .await;

        let embedder = TeiEmbedder::new("intfloat/multilingual-e5-small", &server.uri(), 10)
            .with_passage_prefix(Some("passage: ".to_string()));
        let texts = vec!["doc".to_string()];

        let embeddings = embedder.embed(&texts).await.unwrap();

        assert_eq!(embeddings, vec![vec![0.1, 0.2]]);
    }

    #[tokio::test]
    async fn test_query_prefix_applied() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/embed"))
            .and(body_string_contains("query: hello"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!([[0.3, 0.4]])))
            .mount(&server)
            .await;

        let embedder = TeiEmbedder::new("intfloat/multilingual-e5-small", &server.uri(), 10)
            .with_query_prefix(Some("query: ".to_string()));

        let embedding = embedder.embed_query("hello").await.unwrap();

        assert_eq!(embedding, vec![0.3, 0.4]);
    }

    #[tokio::test]
    async fn test_no_prefix_sends_raw() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/embed"))
            .and(body_json(json!({
                "inputs": ["doc"],
                "normalize": true,
                "truncate": null
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!([[0.1]])))
            .expect(1)
            .mount(&server)
            .await;

        Mock::given(method("POST"))
            .and(path("/embed"))
            .and(body_json(json!({
                "inputs": ["hello"],
                "normalize": true,
                "truncate": null
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!([[0.2]])))
            .expect(1)
            .mount(&server)
            .await;

        let embedder = TeiEmbedder::new("intfloat/multilingual-e5-small", &server.uri(), 10);
        let texts = vec!["doc".to_string()];

        let embeddings = embedder.embed(&texts).await.unwrap();
        let query = embedder.embed_query("hello").await.unwrap();

        assert_eq!(embeddings, vec![vec![0.1]]);
        assert_eq!(query, vec![0.2]);
    }

    #[test]
    fn test_id_strips_tei_prefix() {
        let embedder = TeiEmbedder::new("tei:intfloat/multilingual-e5-small", "http://localhost:8080", 1);

        assert_eq!(embedder.id(), "intfloat/multilingual-e5-small");
    }

    #[tokio::test]
    async fn test_no_auth_header() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/embed"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!([[0.1, 0.2]])))
            .mount(&server)
            .await;

        let embedder = TeiEmbedder::new("intfloat/multilingual-e5-small", &server.uri(), 10);
        let texts = vec!["hello".to_string()];

        let embeddings = embedder.embed(&texts).await.unwrap();

        assert_eq!(embeddings, vec![vec![0.1, 0.2]]);
        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 1);
        assert!(!requests[0].headers.contains_key("authorization"));
    }
}
