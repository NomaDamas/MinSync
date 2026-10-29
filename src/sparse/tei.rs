//! Remote sparse encoder over Text Embeddings Inference's `POST /embed_sparse`
//! (also what `minsync serve --sparse` answers). The request/response shapes
//! are TEI's: `{"inputs": [..], "truncate"?}` -> `[[{"index", "value"}, ..], ..]`.

use super::{SparseEncoder, SparseVector};
use crate::embedder::context_limit::InputLimit;
use crate::embedder::openai::{build_client, DEFAULT_REQUEST_TIMEOUT};
use crate::embedder::retry::{
    classify_send_error, classify_status, embed_adaptive, RequestError, RetryPolicy,
};
use crate::error::{MinSyncError, Result};
use async_trait::async_trait;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::sync::atomic::AtomicUsize;
use std::time::Duration;

pub struct TeiSparseEncoder {
    model: String,
    base_url: String,
    batch_size: usize,
    /// Per-request input limit; only ever shrinks, when the server answers
    /// 413/504/timeouts for a larger batch.
    batch_limit: AtomicUsize,
    retry: RetryPolicy,
    query_prefix: Option<String>,
    passage_prefix: Option<String>,
    client: reqwest::Client,
    limit: InputLimit,
}

#[derive(Serialize)]
struct SparseRequest {
    inputs: Vec<String>,
    truncate: Option<bool>,
}

#[derive(Deserialize)]
struct SparseValue {
    index: u32,
    value: f32,
}

impl TeiSparseEncoder {
    pub fn new(model: &str, base_url: &str, batch_size: usize) -> Self {
        let model = model.strip_prefix("tei:").unwrap_or(model);
        Self {
            model: model.to_string(),
            base_url: base_url.trim_end_matches('/').to_string(),
            batch_size,
            batch_limit: AtomicUsize::new(batch_size.max(1)),
            retry: RetryPolicy::default(),
            query_prefix: None,
            passage_prefix: None,
            client: build_client(DEFAULT_REQUEST_TIMEOUT),
            limit: InputLimit::for_model(model, None),
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

    pub fn with_query_prefix(mut self, prefix: Option<String>) -> Self {
        self.query_prefix = prefix;
        self
    }

    pub fn with_passage_prefix(mut self, prefix: Option<String>) -> Self {
        self.passage_prefix = prefix;
        self
    }

    /// Cut inputs below `max_length` tokens (`None` keeps the model table's
    /// default, or 8192 for an unknown model).
    pub fn with_max_length(mut self, max_length: Option<usize>) -> Self {
        self.limit = InputLimit::for_model(&self.model, max_length);
        self
    }

    fn passage_input(&self, text: &str) -> String {
        let full = match &self.passage_prefix {
            Some(prefix) => format!("{prefix}{text}"),
            None => text.to_string(),
        };
        self.limit.apply(&full).into_owned()
    }

    async fn send_once(
        &self,
        inputs: Vec<String>,
    ) -> std::result::Result<Vec<SparseVector>, RequestError> {
        let expected = inputs.len();
        let response = self
            .client
            .post(format!("{}/embed_sparse", self.base_url))
            // The text is already cut below the context; ask the server to
            // truncate too so a prefix or tokenizer difference never fails a chunk.
            .json(&SparseRequest {
                inputs,
                truncate: Some(true),
            })
            .send()
            .await
            .map_err(|error| classify_send_error(&error, "TEI sparse"))?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(classify_status(status, "TEI sparse", &body));
        }
        let rows: Vec<Vec<SparseValue>> = response.json().await.map_err(|error| {
            RequestError::Fatal(format!("TEI sparse malformed response: {error}"))
        })?;
        if rows.len() != expected {
            return Err(RequestError::Fatal(format!(
                "TEI sparse API returned {} vectors for {expected} inputs",
                rows.len()
            )));
        }
        // TEI lists only non-zero terms. Normalize order and drop zeros so the
        // vector invariant holds whatever the server sent.
        Ok(rows
            .into_iter()
            .map(|row| {
                SparseVector::from_pairs(row.into_iter().map(|entry| (entry.index, entry.value)))
            })
            .collect())
    }

    async fn encode_inputs(&self, inputs: Vec<String>) -> Result<Vec<SparseVector>> {
        embed_adaptive(&self.retry, &self.batch_limit, inputs, |batch| {
            self.send_once(batch)
        })
        .await
    }
}

#[async_trait]
impl SparseEncoder for TeiSparseEncoder {
    fn id(&self) -> &str {
        &self.model
    }

    fn max_length(&self) -> Option<usize> {
        self.limit.max_length()
    }

    async fn count_truncated(&self, texts: &[String]) -> Result<usize> {
        Ok(texts
            .iter()
            .filter(|text| {
                let full = match &self.passage_prefix {
                    Some(prefix) => format!("{prefix}{text}"),
                    None => (*text).clone(),
                };
                self.limit.needs_cut(&full)
            })
            .count())
    }

    async fn encode(&self, texts: &[String]) -> Result<Vec<SparseVector>> {
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
            .map(|batch| batch.iter().map(|text| self.passage_input(text)).collect())
            .collect();
        let results: Vec<Result<Vec<SparseVector>>> =
            futures::stream::iter(batches.into_iter().map(|batch| self.encode_inputs(batch)))
                .buffered(1)
                .collect()
                .await;
        let mut all = Vec::with_capacity(texts.len());
        for batch in results {
            all.extend(batch?);
        }
        Ok(all)
    }

    async fn encode_query(&self, text: &str) -> Result<SparseVector> {
        let input = match &self.query_prefix {
            Some(prefix) => format!("{prefix}{text}"),
            None => text.to_string(),
        };
        self.encode_inputs(vec![input])
            .await?
            .pop()
            .ok_or_else(|| MinSyncError::Embedding("empty response".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

    /// Echoes a one-term vector per input: `{index: len, value: 1.5}`, plus a
    /// second, out-of-order, zero-weight entry the client must normalize away.
    struct Echo;

    impl Respond for Echo {
        fn respond(&self, request: &Request) -> ResponseTemplate {
            let body: Value = serde_json::from_slice(&request.body).unwrap();
            let rows: Vec<Value> = body["inputs"]
                .as_array()
                .unwrap()
                .iter()
                .map(|text| {
                    let len = text.as_str().unwrap().len();
                    json!([
                        {"index": len + 10, "value": 2.0},
                        {"index": len, "value": 1.5},
                        {"index": 3, "value": 0.0},
                    ])
                })
                .collect();
            ResponseTemplate::new(200).set_body_json(rows)
        }
    }

    fn fast(server: &MockServer, batch: usize) -> TeiSparseEncoder {
        TeiSparseEncoder::new("tei:acme/sparse", &server.uri(), batch)
            .with_max_retries(2)
            .with_backoff(Duration::from_millis(1), Duration::from_millis(4))
    }

    #[tokio::test]
    async fn posts_to_embed_sparse_and_normalizes_the_vectors() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embed_sparse"))
            .respond_with(Echo)
            .mount(&server)
            .await;
        let encoder = fast(&server, 10).with_passage_prefix(Some("[D] ".to_string()));

        let vectors = encoder.encode(&["ab".to_string()]).await.unwrap();

        // "[D] ab" is 6 bytes -> terms 6 and 16, ascending; the zero entry is gone.
        assert_eq!(vectors[0].indices(), &[6, 16]);
        assert_eq!(vectors[0].values(), &[1.5, 2.0]);
        let body: Value =
            serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
        assert_eq!(body, json!({"inputs": ["[D] ab"], "truncate": true}));
        assert_eq!(encoder.id(), "acme/sparse");
    }

    #[tokio::test]
    async fn queries_use_the_query_prefix_and_passages_are_batched() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embed_sparse"))
            .respond_with(Echo)
            .mount(&server)
            .await;
        let encoder = fast(&server, 2)
            .with_query_prefix(Some("[Q] ".to_string()))
            .with_passage_prefix(Some("[D] ".to_string()));

        assert_eq!(encoder.encode_query("q").await.unwrap().indices()[0], 5);
        let texts: Vec<String> = ["a", "b", "c"].map(String::from).to_vec();
        assert_eq!(encoder.encode(&texts).await.unwrap().len(), 3);

        let bodies: Vec<Value> = server
            .received_requests()
            .await
            .unwrap()
            .iter()
            .map(|r| serde_json::from_slice(&r.body).unwrap())
            .collect();
        assert_eq!(bodies[0]["inputs"], json!(["[Q] q"]));
        assert_eq!(bodies[1]["inputs"], json!(["[D] a", "[D] b"]));
        assert_eq!(bodies[2]["inputs"], json!(["[D] c"]));
    }

    #[tokio::test]
    async fn empty_input_sends_nothing() {
        let server = MockServer::start().await;
        let encoder = fast(&server, 10);
        assert!(encoder.encode(&[]).await.unwrap().is_empty());
        assert!(server.received_requests().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn a_wrong_vector_count_is_a_fatal_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embed_sparse"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!([[]])))
            .mount(&server)
            .await;
        let error = fast(&server, 10)
            .encode(&["a".to_string(), "b".to_string()])
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains("1 vectors for 2 inputs"),
            "{error}"
        );
        assert_eq!(
            server.received_requests().await.unwrap().len(),
            1,
            "no retry"
        );
    }

    #[tokio::test]
    async fn an_oversized_batch_is_split_not_resent() {
        struct Ceiling;
        impl Respond for Ceiling {
            fn respond(&self, request: &Request) -> ResponseTemplate {
                let body: Value = serde_json::from_slice(&request.body).unwrap();
                if body["inputs"].as_array().unwrap().len() > 2 {
                    return ResponseTemplate::new(413).set_body_string("batch too large");
                }
                Echo.respond(request)
            }
        }
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embed_sparse"))
            .respond_with(Ceiling)
            .mount(&server)
            .await;
        let texts: Vec<String> = (0..6).map(|i| i.to_string()).collect();
        let vectors = fast(&server, 6).encode(&texts).await.unwrap();
        assert_eq!(vectors.len(), 6);
        for (index, vector) in vectors.iter().enumerate() {
            assert_eq!(vector.indices()[0], 1, "input {index} kept its slot");
        }
    }

    #[tokio::test]
    async fn server_errors_are_retried_then_reported() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embed_sparse"))
            .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
            .mount(&server)
            .await;
        let error = fast(&server, 10).encode_query("x").await.unwrap_err();
        assert!(error.to_string().contains("retries exhausted"), "{error}");
        assert_eq!(server.received_requests().await.unwrap().len(), 3);
    }

    #[tokio::test]
    async fn long_input_is_cut_below_the_context_and_counted() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embed_sparse"))
            .respond_with(Echo)
            .mount(&server)
            .await;
        let encoder = fast(&server, 10)
            .with_passage_prefix(Some("[D] ".to_string()))
            .with_max_length(Some(100));
        let long = "a".repeat(5_000);
        assert_eq!(
            encoder
                .count_truncated(&["short".to_string(), long.clone()])
                .await
                .unwrap(),
            1
        );
        encoder.encode(std::slice::from_ref(&long)).await.unwrap();
        let body: Value =
            serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
        assert!(body["inputs"][0].as_str().unwrap().len() < long.len());
    }
}
