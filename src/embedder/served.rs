//! `native:` embedder that uses a running `minsync serve` for the same
//! model instance when there is one, and the in-process engine otherwise.
//!
//! The server is found through its discovery files (see [`crate::serve`])
//! once per process, on the first embedding call. Prefixes are applied here
//! and inputs are sent with `truncate: true`, matching the in-process
//! engine, which truncates at `max_length`. A server that is gone (refused
//! connection, 503 while shutting down) is dropped for the rest of the run
//! and the in-process engine takes over; any other error is reported.

use crate::embedder::Embedder;
use crate::error::{MinSyncError, Result};
use async_trait::async_trait;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::sync::OnceCell;

/// Environment switch: `MINSYNC_SERVER=off` (or `0`/`false`) always embeds
/// in-process.
pub const SERVER_ENV: &str = "MINSYNC_SERVER";

pub fn server_disabled() -> bool {
    std::env::var(SERVER_ENV).is_ok_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "off" | "0" | "false" | "no"
        )
    })
}

pub struct ServedEmbedder {
    local: Box<dyn Embedder>,
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

/// Why a request to the server did not produce vectors.
enum RemoteError {
    /// The server is not there any more; embed in-process instead.
    Gone(String),
    Failed(MinSyncError),
}

impl ServedEmbedder {
    pub fn new(
        local: Box<dyn Embedder>,
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

    /// In-process engine for `settings`, preferring a running server.
    pub fn from_config(settings: &crate::config::EmbedderConfig) -> Result<Self> {
        let local = super::native::NativeEmbedder::from_config(settings)?;
        let instance = crate::serve::instance_key(settings)?;
        // A request may span several forward passes, each bounded by the
        // server's own deadline; this only guards against a dead socket.
        let pass = settings
            .per_batch_timeout_seconds
            .unwrap_or(settings.timeout_seconds)
            .max(super::native::CPU_BATCH_TIMEOUT.as_secs());
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
                let client = crate::serve::loopback_client(self.timeout)?;
                tracing::info!(url = %health.url, pid = health.pid, "using running minsync serve");
                Some(Remote {
                    url: health.url,
                    max_inputs: health.max_inputs.max(1),
                    client,
                })
            })
            .await
            .as_ref()
    }

    /// Embed `inputs` (already prefixed) through the server, or `None` when
    /// there is no usable server.
    async fn embed_remote(&self, inputs: Vec<String>) -> Option<Result<Vec<Vec<f32>>>> {
        let remote = self.remote().await?;
        let mut vectors = Vec::with_capacity(inputs.len());
        for batch in inputs.chunks(self.batch_size.min(remote.max_inputs)) {
            match send(remote, batch).await {
                Ok(batch_vectors) => vectors.extend(batch_vectors),
                Err(RemoteError::Gone(reason)) => {
                    tracing::warn!(%reason, "minsync serve went away; embedding in-process");
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
) -> std::result::Result<Vec<Vec<f32>>, RemoteError> {
    let response = remote
        .client
        .post(format!("{}/embed", remote.url))
        .json(&serde_json::json!({ "inputs": batch, "truncate": true }))
        .send()
        .await
        .map_err(|error| {
            if error.is_timeout() {
                RemoteError::Failed(MinSyncError::Timeout(format!(
                    "minsync serve request timed out: {error}"
                )))
            } else if error.is_connect() || error.is_request() {
                // No response at all: refused, or reset because the server
                // exited (idle or SIGTERM) while the request was on the wire.
                RemoteError::Gone(error.to_string())
            } else {
                RemoteError::Failed(MinSyncError::Embedding(format!(
                    "minsync serve request failed: {error}"
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
        let message = format!("minsync serve {status}: {message}");
        return Err(if status == reqwest::StatusCode::SERVICE_UNAVAILABLE {
            RemoteError::Gone(message)
        } else {
            RemoteError::Failed(MinSyncError::Embedding(message))
        });
    }
    let vectors: Vec<Vec<f32>> = response.json().await.map_err(|error| {
        RemoteError::Failed(MinSyncError::Embedding(format!(
            "minsync serve malformed response: {error}"
        )))
    })?;
    if vectors.len() != batch.len() {
        return Err(RemoteError::Failed(MinSyncError::Embedding(format!(
            "minsync serve returned {} embeddings for {} inputs",
            vectors.len(),
            batch.len()
        ))));
    }
    Ok(vectors)
}

fn prefixed(prefix: Option<&str>, text: &str) -> String {
    match prefix {
        Some(prefix) => format!("{prefix}{text}"),
        None => text.to_string(),
    }
}

#[async_trait]
impl Embedder for ServedEmbedder {
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

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let inputs = texts
            .iter()
            .map(|text| prefixed(self.passage_prefix.as_deref(), text))
            .collect();
        match self.embed_remote(inputs).await {
            Some(result) => result,
            None => self.local.embed(texts).await,
        }
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let input = prefixed(self.query_prefix.as_deref(), text);
        match self.embed_remote(vec![input]).await {
            Some(result) => result?
                .pop()
                .ok_or_else(|| MinSyncError::Embedding("empty response".to_string())),
            None => self.local.embed_query(text).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedder::Embedder;
    use crate::error::Result;
    use serde_json::{json, Value};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

    const INSTANCE: &str = "Fake-Model-abc";

    /// Stands in for the in-process engine: `[-1, len]` per text.
    #[derive(Clone, Default)]
    struct Local {
        calls: Arc<AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl Embedder for Local {
        fn id(&self) -> &str {
            "Fake/Model"
        }
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(texts.iter().map(|t| vec![-1.0, t.len() as f32]).collect())
        }
        async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(vec![-2.0, text.len() as f32])
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

    /// TEI `/embed`: `[1, len]` per input.
    struct Tei;

    impl Respond for Tei {
        fn respond(&self, request: &Request) -> ResponseTemplate {
            let body: Value = serde_json::from_slice(&request.body).unwrap();
            let vectors: Vec<Vec<f32>> = body["inputs"]
                .as_array()
                .unwrap()
                .iter()
                .map(|t| vec![1.0, t.as_str().unwrap().len() as f32])
                .collect();
            ResponseTemplate::new(200).set_body_json(vectors)
        }
    }

    fn health(server: &MockServer, instance: &str, max_inputs: usize) -> Value {
        let addr = server.address();
        json!({
            "status": "ok", "model_id": "Fake/Model", "dimension": 2, "dtype": "f32",
            "device": "cpu", "attention": "eager", "max_length": 2048,
            "max_inputs": max_inputs, "instance": instance, "pid": 1,
            "host": addr.ip().to_string(), "port": addr.port(),
            "url": format!("http://{addr}"), "version": "test",
        })
    }

    async fn serving(instance: &str, max_inputs: usize) -> (MockServer, tempfile::TempDir) {
        // Not pooled: dropping it must close the port.
        let server = MockServer::builder().start().await;
        Mock::given(method("GET"))
            .and(path("/health"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(health(&server, instance, max_inputs)),
            )
            .mount(&server)
            .await;
        let dir = tempfile::tempdir().unwrap();
        let addr = server.address();
        std::fs::write(dir.path().join("host"), format!("{}\n", addr.ip())).unwrap();
        std::fs::write(dir.path().join("port"), format!("{}\n", addr.port())).unwrap();
        (server, dir)
    }

    fn served(local: &Local, dir: &std::path::Path) -> ServedEmbedder {
        ServedEmbedder::new(
            Box::new(local.clone()),
            dir.to_path_buf(),
            INSTANCE.to_string(),
            Some("Q: ".to_string()),
            Some("P: ".to_string()),
            3,
            std::time::Duration::from_secs(30),
        )
    }

    async fn embed_requests(server: &MockServer) -> Vec<Value> {
        server
            .received_requests()
            .await
            .unwrap()
            .into_iter()
            .filter(|r| r.url.path() == "/embed")
            .map(|r| serde_json::from_slice(&r.body).unwrap())
            .collect()
    }

    #[tokio::test]
    async fn a_running_server_embeds_with_client_prefixes_and_truncation() {
        let (server, dir) = serving(INSTANCE, 512).await;
        Mock::given(method("POST"))
            .and(path("/embed"))
            .respond_with(Tei)
            .mount(&server)
            .await;
        let local = Local::default();
        let embedder = served(&local, dir.path());

        let texts: Vec<String> = ["a", "bb", "ccc", "dddd"].map(String::from).to_vec();
        let vectors = embedder.embed(&texts).await.unwrap();
        assert_eq!(
            vectors,
            // "P: a" is 4 bytes.
            vec![
                vec![1.0, 4.0],
                vec![1.0, 5.0],
                vec![1.0, 6.0],
                vec![1.0, 7.0]
            ]
        );
        let query = embedder.embed_query("q").await.unwrap();
        assert_eq!(query, vec![1.0, 4.0]);
        assert_eq!(local.calls.load(Ordering::SeqCst), 0);

        // batch_size 3 splits four passages into 3 + 1 requests.
        let requests = embed_requests(&server).await;
        assert_eq!(
            requests,
            vec![
                json!({"inputs": ["P: a", "P: bb", "P: ccc"], "truncate": true}),
                json!({"inputs": ["P: dddd"], "truncate": true}),
                json!({"inputs": ["Q: q"], "truncate": true}),
            ]
        );
        // Identity and truncation counts stay the in-process engine's.
        assert_eq!(embedder.id(), "Fake/Model");
        assert_eq!(embedder.count_truncated(&texts).await.unwrap(), 7);
        assert!(embedder.is_deterministic());
    }

    #[tokio::test]
    async fn requests_never_exceed_the_servers_max_inputs() {
        let (server, dir) = serving(INSTANCE, 2).await;
        Mock::given(method("POST"))
            .and(path("/embed"))
            .respond_with(Tei)
            .mount(&server)
            .await;
        let embedder = served(&Local::default(), dir.path());
        let texts: Vec<String> = (0..5).map(|i| i.to_string()).collect();
        assert_eq!(embedder.embed(&texts).await.unwrap().len(), 5);
        let sizes: Vec<usize> = embed_requests(&server)
            .await
            .iter()
            .map(|r| r["inputs"].as_array().unwrap().len())
            .collect();
        assert_eq!(sizes, vec![2, 2, 1]);
    }

    #[tokio::test]
    async fn no_server_or_another_instance_uses_the_in_process_engine() {
        let local = Local::default();
        let empty = tempfile::tempdir().unwrap();
        let embedder = served(&local, empty.path());
        assert_eq!(
            embedder.embed(&["x".to_string()]).await.unwrap(),
            vec![vec![-1.0, 1.0]]
        );

        let (server, dir) = serving("Other-Model-def", 512).await;
        let embedder = served(&local, dir.path());
        assert_eq!(embedder.embed_query("x").await.unwrap(), vec![-2.0, 1.0]);
        assert!(embed_requests(&server).await.is_empty());
        assert_eq!(local.calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn a_server_that_goes_away_falls_back_for_the_rest_of_the_run() {
        let (server, dir) = serving(INSTANCE, 512).await;
        // Shutting down after a wedged pass.
        Mock::given(method("POST"))
            .and(path("/embed"))
            .respond_with(ResponseTemplate::new(503).set_body_json(json!({"error": "wedged"})))
            .mount(&server)
            .await;
        let local = Local::default();
        let embedder = served(&local, dir.path());
        let texts = vec!["a".to_string()];
        assert_eq!(embedder.embed(&texts).await.unwrap(), vec![vec![-1.0, 1.0]]);
        assert_eq!(embedder.embed(&texts).await.unwrap(), vec![vec![-1.0, 1.0]]);
        assert_eq!(embed_requests(&server).await.len(), 1);
        assert_eq!(local.calls.load(Ordering::SeqCst), 2);

        // Connection refused (idle exit between probe and request).
        let (server, dir) = serving(INSTANCE, 512).await;
        let local = Local::default();
        let embedder = served(&local, dir.path());
        assert!(embedder.remote().await.is_some());
        drop(server);
        assert_eq!(embedder.embed(&texts).await.unwrap(), vec![vec![-1.0, 1.0]]);
        assert_eq!(local.calls.load(Ordering::SeqCst), 1);
    }

    /// The server dies after accepting the connection: the request fails
    /// with a reset, not a refused connection, and must still fall back.
    #[tokio::test]
    async fn a_connection_reset_without_a_response_falls_back() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let health = serde_json::to_vec(&json!({
            "status": "ok", "model_id": "Fake/Model", "dimension": 2, "dtype": "f32",
            "device": "cpu", "attention": "eager", "max_length": 2048, "max_inputs": 512,
            "instance": INSTANCE, "pid": 1, "host": addr.ip().to_string(),
            "port": addr.port(), "url": format!("http://{addr}"), "version": "test",
        }))
        .unwrap();
        // Answers `/health`; reads `/embed` and closes without a response.
        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                let health = health.clone();
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 64 * 1024];
                    let n = socket.read(&mut buf).await.unwrap_or(0);
                    if buf[..n].starts_with(b"GET /health") {
                        let head = format!(
                            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                            health.len()
                        );
                        let _ = socket.write_all(head.as_bytes()).await;
                        let _ = socket.write_all(&health).await;
                    }
                });
            }
        });
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("host"), "127.0.0.1\n").unwrap();
        std::fs::write(dir.path().join("port"), format!("{}\n", addr.port())).unwrap();
        let local = Local::default();
        let embedder = served(&local, dir.path());
        assert!(embedder.remote().await.is_some());
        let texts = vec!["a".to_string()];
        assert_eq!(embedder.embed(&texts).await.unwrap(), vec![vec![-1.0, 1.0]]);
        assert_eq!(embedder.embed(&texts).await.unwrap(), vec![vec![-1.0, 1.0]]);
        assert_eq!(local.calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn request_errors_from_a_live_server_are_reported_not_hidden() {
        let (server, dir) = serving(INSTANCE, 512).await;
        Mock::given(method("POST"))
            .and(path("/embed"))
            .respond_with(ResponseTemplate::new(413).set_body_json(
                json!({"error": "request body exceeds the limit", "error_type": "Validation"}),
            ))
            .mount(&server)
            .await;
        let local = Local::default();
        let embedder = served(&local, dir.path());
        let error = embedder.embed(&["a".to_string()]).await.unwrap_err();
        assert!(matches!(error, MinSyncError::Embedding(_)), "{error}");
        assert!(error.to_string().contains("exceeds the limit"), "{error}");
        assert_eq!(local.calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn a_wrong_vector_count_is_an_error() {
        let (server, dir) = serving(INSTANCE, 512).await;
        Mock::given(method("POST"))
            .and(path("/embed"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!([[1.0, 2.0]])))
            .mount(&server)
            .await;
        let embedder = served(&Local::default(), dir.path());
        let error = embedder
            .embed(&["a".to_string(), "b".to_string()])
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains("1 embeddings for 2 inputs"),
            "{error}"
        );
    }
}
