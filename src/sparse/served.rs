//! `native:` sparse encoder that uses a running `minsync serve --sparse` for
//! the same model instance when there is one, and the in-process engine
//! otherwise. Same rules as [`crate::embedder::served::ServedEmbedder`]: the
//! server is looked up once per process, prefixes are applied here, inputs go
//! out with `truncate: true`, and a server that disappears is dropped for the
//! rest of the run (the in-process engine takes over); any other error is
//! reported.

use super::{SparseEncoder, SparseVector};
use crate::embedder::served::server_disabled;
use crate::error::{MinSyncError, Result};
use async_trait::async_trait;
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::sync::OnceCell;

pub struct ServedSparseEncoder {
    local: Box<dyn SparseEncoder>,
    state_dir: PathBuf,
    instance: String,
    query_prefix: Option<String>,
    passage_prefix: Option<String>,
    batch_size: usize,
    timeout: Duration,
    remote: OnceCell<Option<Remote>>,
    gone: AtomicBool,
}

struct Remote {
    url: String,
    max_inputs: usize,
    client: reqwest::Client,
}

enum RemoteError {
    /// The server is not there any more; encode in-process instead.
    Gone(String),
    Failed(MinSyncError),
}

#[derive(Deserialize)]
struct Entry {
    index: u32,
    value: f32,
}

impl ServedSparseEncoder {
    pub fn new(
        local: Box<dyn SparseEncoder>,
        state_dir: PathBuf,
        instance: String,
        query_prefix: Option<String>,
        passage_prefix: Option<String>,
        batch_size: usize,
        timeout: Duration,
    ) -> Self {
        Self {
            local,
            state_dir,
            instance,
            query_prefix,
            passage_prefix,
            batch_size: batch_size.max(1),
            timeout,
            remote: OnceCell::new(),
            gone: AtomicBool::new(false),
        }
    }

    /// The in-process engine for `settings`, preferring a running server.
    pub fn from_config(settings: &crate::config::SparseConfig) -> Result<Self> {
        let local = super::native::NativeSparseEncoder::from_config(settings)?;
        let instance = crate::serve::sparse_instance_key(settings)?;
        let pass = settings
            .per_batch_timeout_seconds
            .unwrap_or(settings.timeout_seconds)
            .max(crate::embedder::native::CPU_BATCH_TIMEOUT.as_secs());
        Ok(Self::new(
            Box::new(local),
            crate::serve::default_state_dir(&instance),
            instance,
            settings.query_prefix.clone(),
            settings.passage_prefix.clone(),
            settings.batch_size,
            Duration::from_secs(pass),
        ))
    }

    async fn remote(&self) -> Option<&Remote> {
        if self.gone.load(Ordering::Acquire) {
            return None;
        }
        self.remote
            .get_or_init(|| async {
                let health = crate::serve::probe(&self.state_dir, &self.instance).await?;
                if health.kind != "sparse" {
                    return None;
                }
                let client = reqwest::Client::builder()
                    .timeout(self.timeout)
                    .no_proxy()
                    .build()
                    .ok()?;
                tracing::info!(url = %health.url, pid = health.pid, "using running minsync serve --sparse");
                Some(Remote {
                    url: health.url,
                    max_inputs: health.max_inputs.max(1),
                    client,
                })
            })
            .await
            .as_ref()
    }

    /// Encode `inputs` (already prefixed) through the server, or `None` when
    /// there is no usable server.
    async fn encode_remote(&self, inputs: Vec<String>) -> Option<Result<Vec<SparseVector>>> {
        let remote = self.remote().await?;
        let mut vectors = Vec::with_capacity(inputs.len());
        for batch in inputs.chunks(self.batch_size.min(remote.max_inputs)) {
            match send(remote, batch).await {
                Ok(batch_vectors) => vectors.extend(batch_vectors),
                Err(RemoteError::Gone(reason)) => {
                    tracing::warn!(%reason, "minsync serve --sparse went away; encoding in-process");
                    self.gone.store(true, Ordering::Release);
                    return None;
                }
                Err(RemoteError::Failed(error)) => return Some(Err(error)),
            }
        }
        Some(Ok(vectors))
    }
}

async fn send(
    remote: &Remote,
    batch: &[String],
) -> std::result::Result<Vec<SparseVector>, RemoteError> {
    let response = remote
        .client
        .post(format!("{}/embed_sparse", remote.url))
        .json(&serde_json::json!({ "inputs": batch, "truncate": true }))
        .send()
        .await
        .map_err(|error| {
            if error.is_timeout() {
                RemoteError::Failed(MinSyncError::Timeout(format!(
                    "minsync serve --sparse request timed out: {error}"
                )))
            } else if error.is_connect() || error.is_request() {
                RemoteError::Gone(error.to_string())
            } else {
                RemoteError::Failed(MinSyncError::Embedding(format!(
                    "minsync serve --sparse request failed: {error}"
                )))
            }
        })?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        let message = serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|value| value["error"].as_str().map(str::to_string))
            .unwrap_or(body);
        let message = format!("minsync serve --sparse {status}: {message}");
        return Err(if status == reqwest::StatusCode::SERVICE_UNAVAILABLE {
            RemoteError::Gone(message)
        } else {
            RemoteError::Failed(MinSyncError::Embedding(message))
        });
    }
    let rows: Vec<Vec<Entry>> = response.json().await.map_err(|error| {
        RemoteError::Failed(MinSyncError::Embedding(format!(
            "minsync serve --sparse malformed response: {error}"
        )))
    })?;
    if rows.len() != batch.len() {
        return Err(RemoteError::Failed(MinSyncError::Embedding(format!(
            "minsync serve --sparse returned {} vectors for {} inputs",
            rows.len(),
            batch.len()
        ))));
    }
    Ok(rows
        .into_iter()
        .map(|row| {
            SparseVector::from_pairs(row.into_iter().map(|entry| (entry.index, entry.value)))
        })
        .collect())
}

fn prefixed(prefix: Option<&str>, text: &str) -> String {
    match prefix {
        Some(prefix) => format!("{prefix}{text}"),
        None => text.to_string(),
    }
}

#[async_trait]
impl SparseEncoder for ServedSparseEncoder {
    fn id(&self) -> &str {
        self.local.id()
    }

    async fn count_truncated(&self, texts: &[String]) -> Result<usize> {
        self.local.count_truncated(texts).await
    }

    fn max_length(&self) -> Option<usize> {
        self.local.max_length()
    }

    fn is_deterministic(&self) -> bool {
        self.local.is_deterministic()
    }

    async fn encode(&self, texts: &[String]) -> Result<Vec<SparseVector>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let inputs = texts
            .iter()
            .map(|text| prefixed(self.passage_prefix.as_deref(), text))
            .collect();
        match self.encode_remote(inputs).await {
            Some(result) => result,
            None => self.local.encode(texts).await,
        }
    }

    async fn encode_query(&self, text: &str) -> Result<SparseVector> {
        let input = prefixed(self.query_prefix.as_deref(), text);
        match self.encode_remote(vec![input]).await {
            Some(result) => result?
                .pop()
                .ok_or_else(|| MinSyncError::Embedding("empty response".to_string())),
            None => self.local.encode_query(text).await,
        }
    }
}

/// Construct a [`SparseEncoder`] from the sparse id in `settings`:
/// `native:` (in-process, or a running `serve --sparse`) or `tei:` (remote).
pub fn create_sparse_encoder(
    settings: &crate::config::SparseConfig,
) -> Result<Box<dyn SparseEncoder>> {
    let id = settings.id.as_str();
    if id.starts_with("native:") {
        if server_disabled() {
            Ok(Box::new(super::native::NativeSparseEncoder::from_config(
                settings,
            )?))
        } else {
            Ok(Box::new(ServedSparseEncoder::from_config(settings)?))
        }
    } else if id.starts_with("tei:") {
        let base_url = settings
            .base_url
            .clone()
            .unwrap_or_else(|| "http://localhost:8080".to_string());
        Ok(Box::new(
            super::tei::TeiSparseEncoder::new(id, &base_url, settings.batch_size)
                .with_timeout(Duration::from_secs(settings.timeout_seconds))
                .with_max_retries(settings.max_retries)
                .with_query_prefix(settings.query_prefix.clone())
                .with_passage_prefix(settings.passage_prefix.clone())
                .with_max_length(settings.max_length),
        ))
    } else {
        Err(MinSyncError::Config(format!(
            "unknown sparse encoder id: {id}"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SparseConfig;
    use serde_json::{json, Value};
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

    const INSTANCE: &str = "Fake-Sparse-abc";

    fn vector(term: u32, weight: f32) -> SparseVector {
        SparseVector::new(vec![term], vec![weight]).unwrap()
    }

    /// Stands in for the in-process engine: `{999: len}` per text.
    #[derive(Clone, Default)]
    struct Local {
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl SparseEncoder for Local {
        fn id(&self) -> &str {
            "Fake/Sparse"
        }
        async fn encode(&self, texts: &[String]) -> Result<Vec<SparseVector>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(texts.iter().map(|t| vector(999, t.len() as f32)).collect())
        }
        async fn encode_query(&self, text: &str) -> Result<SparseVector> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(vector(998, text.len() as f32))
        }
        async fn count_truncated(&self, _texts: &[String]) -> Result<usize> {
            Ok(7)
        }
        fn max_length(&self) -> Option<usize> {
            Some(2048)
        }
        fn is_deterministic(&self) -> bool {
            true
        }
    }

    /// TEI `/embed_sparse`: `{1: len}` per input.
    struct Tei;

    impl Respond for Tei {
        fn respond(&self, request: &Request) -> ResponseTemplate {
            let body: Value = serde_json::from_slice(&request.body).unwrap();
            let rows: Vec<Value> = body["inputs"]
                .as_array()
                .unwrap()
                .iter()
                .map(|t| json!([{"index": 1, "value": t.as_str().unwrap().len() as f32}]))
                .collect();
            ResponseTemplate::new(200).set_body_json(rows)
        }
    }

    fn health(server: &MockServer, instance: &str, kind: &str, max_inputs: usize) -> Value {
        let addr = server.address();
        json!({
            "status": "ok", "kind": kind, "model_id": "Fake/Sparse", "dimension": 5,
            "dtype": "f32", "device": "cpu", "attention": "tiled", "max_length": 2048,
            "max_inputs": max_inputs, "instance": instance, "pid": 1,
            "host": addr.ip().to_string(), "port": addr.port(),
            "url": format!("http://{addr}"), "version": "test",
        })
    }

    async fn serving(
        instance: &str,
        kind: &str,
        max_inputs: usize,
    ) -> (MockServer, tempfile::TempDir) {
        let server = MockServer::builder().start().await;
        Mock::given(method("GET"))
            .and(path("/health"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(health(&server, instance, kind, max_inputs)),
            )
            .mount(&server)
            .await;
        let dir = tempfile::tempdir().unwrap();
        let addr = server.address();
        std::fs::write(dir.path().join("host"), format!("{}\n", addr.ip())).unwrap();
        std::fs::write(dir.path().join("port"), format!("{}\n", addr.port())).unwrap();
        (server, dir)
    }

    fn served(local: &Local, dir: &std::path::Path) -> ServedSparseEncoder {
        ServedSparseEncoder::new(
            Box::new(local.clone()),
            dir.to_path_buf(),
            INSTANCE.to_string(),
            Some("Q: ".to_string()),
            Some("P: ".to_string()),
            3,
            Duration::from_secs(30),
        )
    }

    async fn sparse_requests(server: &MockServer) -> Vec<Value> {
        server
            .received_requests()
            .await
            .unwrap()
            .into_iter()
            .filter(|r| r.url.path() == "/embed_sparse")
            .map(|r| serde_json::from_slice(&r.body).unwrap())
            .collect()
    }

    #[tokio::test]
    async fn a_running_sparse_server_encodes_with_client_prefixes_and_truncation() {
        let (server, dir) = serving(INSTANCE, "sparse", 512).await;
        Mock::given(method("POST"))
            .and(path("/embed_sparse"))
            .respond_with(Tei)
            .mount(&server)
            .await;
        let local = Local::default();
        let encoder = served(&local, dir.path());

        let texts: Vec<String> = ["a", "bb", "ccc", "dddd"].map(String::from).to_vec();
        let vectors = encoder.encode(&texts).await.unwrap();
        // "P: a" is 4 bytes.
        let weights: Vec<f32> = vectors.iter().map(|v| v.values()[0]).collect();
        assert_eq!(weights, vec![4.0, 5.0, 6.0, 7.0]);
        assert_eq!(encoder.encode_query("q").await.unwrap().values()[0], 4.0);
        assert_eq!(local.calls.load(Ordering::SeqCst), 0);

        assert_eq!(
            sparse_requests(&server).await,
            vec![
                json!({"inputs": ["P: a", "P: bb", "P: ccc"], "truncate": true}),
                json!({"inputs": ["P: dddd"], "truncate": true}),
                json!({"inputs": ["Q: q"], "truncate": true}),
            ]
        );
        assert_eq!(encoder.id(), "Fake/Sparse");
        assert_eq!(encoder.count_truncated(&texts).await.unwrap(), 7);
        assert!(encoder.is_deterministic());
    }

    #[tokio::test]
    async fn a_dense_server_or_another_instance_is_never_used() {
        let local = Local::default();
        let (dense, dir) = serving(INSTANCE, "dense", 512).await;
        let encoder = served(&local, dir.path());
        assert_eq!(encoder.encode_query("x").await.unwrap().indices(), &[998]);
        assert!(sparse_requests(&dense).await.is_empty());

        let (other, dir) = serving("Other-Model-def", "sparse", 512).await;
        let encoder = served(&local, dir.path());
        assert_eq!(
            encoder.encode(&["x".to_string()]).await.unwrap()[0].indices(),
            &[999]
        );
        assert!(sparse_requests(&other).await.is_empty());

        let empty = tempfile::tempdir().unwrap();
        let encoder = served(&local, empty.path());
        assert_eq!(encoder.encode_query("x").await.unwrap().indices(), &[998]);
        assert_eq!(local.calls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn requests_never_exceed_the_servers_max_inputs() {
        let (server, dir) = serving(INSTANCE, "sparse", 2).await;
        Mock::given(method("POST"))
            .and(path("/embed_sparse"))
            .respond_with(Tei)
            .mount(&server)
            .await;
        let encoder = served(&Local::default(), dir.path());
        let texts: Vec<String> = (0..5).map(|i| i.to_string()).collect();
        assert_eq!(encoder.encode(&texts).await.unwrap().len(), 5);
        let sizes: Vec<usize> = sparse_requests(&server)
            .await
            .iter()
            .map(|r| r["inputs"].as_array().unwrap().len())
            .collect();
        assert_eq!(sizes, vec![2, 2, 1]);
    }

    #[tokio::test]
    async fn a_server_that_goes_away_falls_back_for_the_rest_of_the_run() {
        let (server, dir) = serving(INSTANCE, "sparse", 512).await;
        Mock::given(method("POST"))
            .and(path("/embed_sparse"))
            .respond_with(ResponseTemplate::new(503).set_body_json(json!({"error": "wedged"})))
            .mount(&server)
            .await;
        let local = Local::default();
        let encoder = served(&local, dir.path());
        let texts = vec!["a".to_string()];
        assert_eq!(encoder.encode(&texts).await.unwrap()[0].indices(), &[999]);
        assert_eq!(encoder.encode(&texts).await.unwrap()[0].indices(), &[999]);
        assert_eq!(
            sparse_requests(&server).await.len(),
            1,
            "dropped after the 503"
        );
        assert_eq!(local.calls.load(Ordering::SeqCst), 2);

        // Connection refused (idle exit between probe and request).
        let (server, dir) = serving(INSTANCE, "sparse", 512).await;
        let local = Local::default();
        let encoder = served(&local, dir.path());
        assert!(encoder.remote().await.is_some());
        drop(server);
        assert_eq!(encoder.encode(&texts).await.unwrap()[0].indices(), &[999]);
        assert_eq!(local.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn errors_from_a_live_server_are_reported_not_hidden() {
        let (server, dir) = serving(INSTANCE, "sparse", 512).await;
        Mock::given(method("POST"))
            .and(path("/embed_sparse"))
            .respond_with(ResponseTemplate::new(413).set_body_json(
                json!({"error": "request body exceeds the limit", "error_type": "Validation"}),
            ))
            .mount(&server)
            .await;
        let local = Local::default();
        let error = served(&local, dir.path())
            .encode(&["a".to_string()])
            .await
            .unwrap_err();
        assert!(error.to_string().contains("exceeds the limit"), "{error}");
        assert_eq!(local.calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn a_wrong_vector_count_is_an_error() {
        let (server, dir) = serving(INSTANCE, "sparse", 512).await;
        Mock::given(method("POST"))
            .and(path("/embed_sparse"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!([[]])))
            .mount(&server)
            .await;
        let error = served(&Local::default(), dir.path())
            .encode(&["a".to_string(), "b".to_string()])
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains("1 vectors for 2 inputs"),
            "{error}"
        );
    }

    #[test]
    fn the_factory_picks_the_provider_from_the_id() {
        let native = create_sparse_encoder(&SparseConfig::default()).unwrap();
        assert_eq!(native.id(), "nlpai-lab/KURE-Sparse-nano");
        assert!(native.is_deterministic());

        let remote = create_sparse_encoder(&SparseConfig {
            id: "tei:nlpai-lab/KURE-Sparse-nano".to_string(),
            base_url: Some("http://localhost:1".to_string()),
            ..SparseConfig::default()
        })
        .unwrap();
        assert_eq!(remote.id(), "nlpai-lab/KURE-Sparse-nano");
        assert!(!remote.is_deterministic());

        let error = create_sparse_encoder(&SparseConfig {
            id: "openai:text-embedding-3-small".to_string(),
            ..SparseConfig::default()
        })
        .err()
        .expect("unknown provider");
        assert!(
            error.to_string().contains("unknown sparse encoder id"),
            "{error}"
        );
    }
}
