use crate::embedder::openai::{build_client, DEFAULT_REQUEST_TIMEOUT};
use crate::embedder::retry::{classify_send_error, classify_status, RequestError, RetryPolicy};
use crate::embedder::Embedder;
use crate::error::{MinSyncError, Result};
use async_trait::async_trait;
use futures::StreamExt;
use serde::Serialize;
use std::time::Duration;

pub struct TeiEmbedder {
    model: String,
    base_url: String,
    batch_size: usize,
    max_concurrent: usize,
    retry: RetryPolicy,
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
            max_concurrent: 1,
            retry: RetryPolicy::default(),
            query_prefix: None,
            passage_prefix: None,
            client: build_client(DEFAULT_REQUEST_TIMEOUT),
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.client = build_client(timeout);
        self
    }

    pub fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.retry.max_retries = max_retries;
        self
    }

    pub fn with_backoff(mut self, base_delay: Duration, max_delay: Duration) -> Self {
        self.retry.base_delay = base_delay;
        self.retry.max_delay = max_delay;
        self
    }

    pub fn with_max_concurrent(mut self, max_concurrent: usize) -> Self {
        self.max_concurrent = max_concurrent.max(1);
        self
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
        self.retry
            .run(|| async {
                let request = EmbedRequest {
                    inputs: inputs.clone(),
                    normalize: true,
                    truncate: None,
                };

                let response = self
                    .client
                    .post(format!("{}/embed", self.base_url))
                    .json(&request)
                    .send()
                    .await
                    .map_err(|error| classify_send_error(&error, "TEI"))?;

                let status = response.status();
                if !status.is_success() {
                    let body = response.text().await.unwrap_or_default();
                    return Err(classify_status(status, "TEI", &body));
                }

                let embeddings: Vec<Vec<f32>> = response.json().await.map_err(|error| {
                    RequestError::Fatal(format!("TEI malformed response: {error}"))
                })?;

                if embeddings.len() != expected_count {
                    return Err(RequestError::Fatal(format!(
                        "TEI API returned {} embeddings for {} inputs",
                        embeddings.len(),
                        expected_count
                    )));
                }

                Ok(embeddings)
            })
            .await
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

        let batches: Vec<Vec<String>> = texts
            .chunks(self.batch_size)
            .map(|batch| {
                batch
                    .iter()
                    .map(|text| match &self.passage_prefix {
                        Some(prefix) => format!("{}{}", prefix, text),
                        None => text.clone(),
                    })
                    .collect()
            })
            .collect();

        let batch_results: Vec<Result<Vec<Vec<f32>>>> =
            futures::stream::iter(batches.into_iter().map(|batch| self.embed_inputs(batch)))
                .buffered(self.max_concurrent)
                .collect()
                .await;

        let mut all_embeddings = Vec::with_capacity(texts.len());
        for batch in batch_results {
            all_embeddings.extend(batch?);
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
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])),
            )
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

        let embedder = fast_retry_embedder(&server.uri(), 0);
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
        let embedder = TeiEmbedder::new(
            "tei:intfloat/multilingual-e5-small",
            "http://localhost:8080",
            1,
        );

        assert_eq!(embedder.id(), "intfloat/multilingual-e5-small");
    }

    fn fast_retry_embedder(url: &str, max_retries: usize) -> TeiEmbedder {
        TeiEmbedder::new("intfloat/multilingual-e5-small", url, 10)
            .with_max_retries(max_retries)
            .with_backoff(
                std::time::Duration::from_millis(1),
                std::time::Duration::from_millis(4),
            )
    }

    #[tokio::test]
    async fn test_retry_on_503_then_success() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embed"))
            .respond_with(ResponseTemplate::new(503).set_body_string("loading"))
            .up_to_n_times(2)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/embed"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!([[0.1, 0.2]])))
            .mount(&server)
            .await;

        let embedder = fast_retry_embedder(&server.uri(), 3);
        let embeddings = embedder.embed(&["hello".to_string()]).await.unwrap();

        assert_eq!(embeddings, vec![vec![0.1, 0.2]]);
        assert_eq!(server.received_requests().await.unwrap().len(), 3);
    }

    #[tokio::test]
    async fn test_retry_exhaustion_on_503() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embed"))
            .respond_with(ResponseTemplate::new(503).set_body_string("loading"))
            .mount(&server)
            .await;

        let embedder = fast_retry_embedder(&server.uri(), 1);
        let error = embedder.embed(&["hello".to_string()]).await.unwrap_err();

        assert!(error.to_string().contains("retries exhausted"));
        assert_eq!(server.received_requests().await.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn test_fail_fast_on_400() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embed"))
            .respond_with(ResponseTemplate::new(400).set_body_string("bad request"))
            .mount(&server)
            .await;

        let embedder = fast_retry_embedder(&server.uri(), 3);
        let error = embedder.embed(&["hello".to_string()]).await.unwrap_err();

        assert!(error.to_string().contains("TEI API error 400"));
        assert_eq!(server.received_requests().await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_count_mismatch_fails_fast() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embed"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!([[0.1], [0.2]])))
            .mount(&server)
            .await;

        let embedder = fast_retry_embedder(&server.uri(), 3);
        let error = embedder.embed(&["only-one".to_string()]).await.unwrap_err();

        assert!(error.to_string().contains("2 embeddings for 1 inputs"));
        assert_eq!(server.received_requests().await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_timeout_is_retried_then_exhausted() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embed"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_delay(std::time::Duration::from_millis(500))
                    .set_body_json(json!([[0.1]])),
            )
            .mount(&server)
            .await;

        let embedder = fast_retry_embedder(&server.uri(), 1)
            .with_timeout(std::time::Duration::from_millis(50));
        let error = embedder.embed(&["hello".to_string()]).await.unwrap_err();

        assert!(error.to_string().contains("retries exhausted"));
        assert_eq!(server.received_requests().await.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn test_concurrent_batches_keep_order() {
        let server = MockServer::start().await;
        for (text, value) in [("t-one", 0.1f32), ("t-two", 0.2), ("t-three", 0.3)] {
            Mock::given(method("POST"))
                .and(path("/embed"))
                .and(body_string_contains(text))
                .respond_with(
                    ResponseTemplate::new(200)
                        .set_delay(std::time::Duration::from_millis(200))
                        .set_body_json(json!([[value]])),
                )
                .mount(&server)
                .await;
        }

        let embedder = TeiEmbedder::new("intfloat/multilingual-e5-small", &server.uri(), 1)
            .with_max_concurrent(3);
        let texts = vec![
            "t-one".to_string(),
            "t-two".to_string(),
            "t-three".to_string(),
        ];

        let started = std::time::Instant::now();
        let embeddings = embedder.embed(&texts).await.unwrap();
        let elapsed = started.elapsed();

        assert_eq!(embeddings, vec![vec![0.1], vec![0.2], vec![0.3]]);
        assert!(
            elapsed < std::time::Duration::from_millis(550),
            "3 batches x 200ms should overlap with max_concurrent=3, took {elapsed:?}"
        );
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
