use crate::embedder::retry::{classify_send_error, classify_status, RequestError, RetryPolicy};
use crate::embedder::Embedder;
use crate::error::{MinSyncError, Result};
use async_trait::async_trait;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::time::Duration;

pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

pub struct OpenAiEmbedder {
    model: String,
    api_key: String,
    batch_size: usize,
    max_concurrent: usize,
    retry: RetryPolicy,
    client: reqwest::Client,
    base_url: String,
}

pub(crate) fn build_client(timeout: Duration) -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(timeout)
        .build()
        .unwrap_or_default()
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
            max_concurrent: 1,
            retry: RetryPolicy::default(),
            client: build_client(DEFAULT_REQUEST_TIMEOUT),
            base_url: "https://api.openai.com".to_string(),
        }
    }

    pub fn with_base_url(mut self, url: &str) -> Self {
        self.base_url = url.to_string();
        self
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

    async fn embed_batch(&self, batch: Vec<String>) -> Result<Vec<Vec<f32>>> {
        self.retry
            .run(|| async {
                let request = EmbedRequest {
                    input: batch.clone(),
                    model: self.model.clone(),
                };

                let response = self
                    .client
                    .post(format!("{}/v1/embeddings", self.base_url))
                    .bearer_auth(&self.api_key)
                    .json(&request)
                    .send()
                    .await
                    .map_err(|error| classify_send_error(&error, "OpenAI"))?;

                let status = response.status();
                if !status.is_success() {
                    let body = response.text().await.unwrap_or_default();
                    return Err(classify_status(status, "OpenAI", &body));
                }

                let embed_response: EmbedResponse = response.json().await.map_err(|error| {
                    RequestError::Fatal(format!("OpenAI malformed response: {error}"))
                })?;

                Ok(embed_response
                    .data
                    .into_iter()
                    .map(|data| data.embedding)
                    .collect())
            })
            .await
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

        let batches: Vec<Vec<String>> = texts
            .chunks(self.batch_size)
            .map(|batch| batch.to_vec())
            .collect();
        let batch_results: Vec<Result<Vec<Vec<f32>>>> =
            futures::stream::iter(batches.into_iter().map(|batch| self.embed_batch(batch)))
                .buffered(self.max_concurrent)
                .collect()
                .await;

        let mut all_embeddings = Vec::with_capacity(texts.len());
        for batch in batch_results {
            all_embeddings.extend(batch?);
        }
        Ok(all_embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::net::TcpListener;
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

    fn fast_retry_embedder(url: &str, max_retries: usize) -> OpenAiEmbedder {
        OpenAiEmbedder::new("text-embedding-3-small", "test-key", 10)
            .with_base_url(url)
            .with_max_retries(max_retries)
            .with_backoff(Duration::from_millis(1), Duration::from_millis(4))
    }

    async fn hanging_openai_endpoint() -> (String, Arc<AtomicUsize>, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind hanging endpoint");
        let url = format!("http://{}", listener.local_addr().expect("local addr"));
        let attempts = Arc::new(AtomicUsize::new(0));
        let attempts_for_task = Arc::clone(&attempts);
        let task = tokio::spawn(async move {
            while let Ok((socket, _addr)) = listener.accept().await {
                attempts_for_task.fetch_add(1, Ordering::SeqCst);
                tokio::spawn(async move {
                    let _socket = socket;
                    tokio::time::sleep(Duration::from_secs(60)).await;
                });
            }
        });
        (url, attempts, task)
    }

    #[tokio::test]
    async fn test_retry_on_503_then_success() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(503).set_body_string("unavailable"))
            .up_to_n_times(2)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [{"embedding": [0.1, 0.2], "index": 0}],
                "model": "text-embedding-3-small"
            })))
            .mount(&server)
            .await;

        let embedder = fast_retry_embedder(&server.uri(), 3);
        let embeddings = embedder.embed(&["hello".to_string()]).await.unwrap();

        assert_eq!(embeddings, vec![vec![0.1, 0.2]]);
        assert_eq!(server.received_requests().await.unwrap().len(), 3);
    }

    #[tokio::test]
    async fn test_retry_on_429_then_success() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [{"embedding": [0.5], "index": 0}],
                "model": "text-embedding-3-small"
            })))
            .mount(&server)
            .await;

        let embedder = fast_retry_embedder(&server.uri(), 2);
        let embeddings = embedder.embed(&["hello".to_string()]).await.unwrap();

        assert_eq!(embeddings, vec![vec![0.5]]);
        assert_eq!(server.received_requests().await.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn test_retry_exhaustion_on_503() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(503).set_body_string("unavailable"))
            .mount(&server)
            .await;

        let embedder = fast_retry_embedder(&server.uri(), 1);
        let error = embedder.embed(&["hello".to_string()]).await.unwrap_err();

        assert!(error.to_string().contains("retries exhausted"));
        assert!(error.to_string().contains("503"));
        assert_eq!(server.received_requests().await.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn test_fail_fast_on_401() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(401).set_body_string("unauthorized"))
            .mount(&server)
            .await;

        let embedder = fast_retry_embedder(&server.uri(), 3);
        let error = embedder.embed(&["hello".to_string()]).await.unwrap_err();

        assert!(error.to_string().contains("OpenAI API error 401"));
        assert_eq!(server.received_requests().await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_timeout_is_retried_then_exhausted() {
        let (url, attempts, server_task) = hanging_openai_endpoint().await;
        let embedder = fast_retry_embedder(&url, 1).with_timeout(Duration::from_millis(50));
        let error = embedder.embed(&["hello".to_string()]).await.unwrap_err();
        server_task.abort();

        assert!(error.to_string().contains("retries exhausted"));
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_malformed_response_fails_fast() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_string("not json"))
            .mount(&server)
            .await;

        let embedder = fast_retry_embedder(&server.uri(), 3);
        let error = embedder.embed(&["hello".to_string()]).await.unwrap_err();

        assert!(error.to_string().contains("malformed response"));
        assert_eq!(server.received_requests().await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_concurrent_batches_run_in_parallel_and_keep_order() {
        let server = MockServer::start().await;
        for (text, value) in [("t-one", 0.1f32), ("t-two", 0.2), ("t-three", 0.3)] {
            Mock::given(method("POST"))
                .and(path("/v1/embeddings"))
                .and(wiremock::matchers::body_string_contains(text))
                .respond_with(
                    ResponseTemplate::new(200)
                        .set_delay(Duration::from_millis(200))
                        .set_body_json(json!({
                            "data": [{"embedding": [value], "index": 0}],
                            "model": "text-embedding-3-small"
                        })),
                )
                .mount(&server)
                .await;
        }

        let embedder = OpenAiEmbedder::new("text-embedding-3-small", "test-key", 1)
            .with_base_url(&server.uri())
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
            elapsed < Duration::from_millis(550),
            "3 batches x 200ms should overlap with max_concurrent=3, took {elapsed:?}"
        );
    }
}
