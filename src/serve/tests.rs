//! Behavior of `minsync serve` against a fake engine on a real loopback
//! socket: protocol shapes, prefixes, limits, idle exit, adoption, files.

use super::*;
use crate::error::{MinSyncError, Result};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex as StdMutex;
use std::time::{Duration, Instant};

/// Embeds a text as `[len, first byte, 1]`; one token per whitespace word.
#[derive(Default)]
struct FakeEngine {
    seen: StdMutex<Vec<Vec<String>>>,
    delay: Option<Duration>,
    timeout: AtomicBool,
}

#[async_trait::async_trait]
impl ServeEngine for FakeEngine {
    async fn load(&self) -> Result<EngineInfo> {
        Ok(EngineInfo {
            model_id: "Fake/Model-1".to_string(),
            dimension: 3,
            dtype: "f32".to_string(),
            device: "cpu".to_string(),
            attention: "eager".to_string(),
            max_length: 4,
        })
    }

    async fn token_counts(&self, texts: &[String]) -> Result<Vec<usize>> {
        Ok(texts.iter().map(|t| t.split_whitespace().count()).collect())
    }

    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if self.timeout.load(Ordering::SeqCst) {
            return Err(MinSyncError::Timeout("native embedding batch stuck".into()));
        }
        if let Some(delay) = self.delay {
            tokio::time::sleep(delay).await;
        }
        self.seen.lock().unwrap().push(texts.clone());
        Ok(texts
            .iter()
            .map(|t| {
                vec![
                    t.len() as f32,
                    f32::from(t.bytes().next().unwrap_or(0)),
                    0.5,
                ]
            })
            .collect())
    }
}

fn options(dir: &std::path::Path) -> ServeOptions {
    ServeOptions {
        host: "127.0.0.1".parse().unwrap(),
        port: 0,
        idle_timeout: None,
        max_inputs: 8,
        max_request_bytes: 64 * 1024,
        state_dir: dir.to_path_buf(),
        instance: "test-instance".to_string(),
        query_prefix: Some("Q: ".to_string()),
        passage_prefix: Some("P: ".to_string()),
    }
}

struct Harness {
    url: String,
    engine: Arc<FakeEngine>,
    stopped: tokio::task::JoinHandle<Result<StopReason>>,
    stop: Arc<tokio::sync::Notify>,
    _dir: tempfile::TempDir,
}

async fn spawn_with(engine: FakeEngine, tweak: impl FnOnce(&mut ServeOptions)) -> Harness {
    let dir = tempfile::tempdir().unwrap();
    let mut opts = options(dir.path());
    tweak(&mut opts);
    let engine = Arc::new(engine);
    let started = start(engine.clone(), opts).await.unwrap();
    let Started::Serving(server) = started else {
        panic!("expected a fresh server");
    };
    let url = server.health().url.clone();
    let stop = Arc::new(tokio::sync::Notify::new());
    let signal = stop.clone();
    let stopped = tokio::spawn(async move {
        server
            .run_until(async move { signal.notified().await })
            .await
    });
    Harness {
        url,
        engine,
        stopped,
        stop,
        _dir: dir,
    }
}

async fn spawn() -> Harness {
    spawn_with(FakeEngine::default(), |_| {}).await
}

async fn post(url: &str, body: Value) -> (u16, Value) {
    let response = reqwest::Client::new()
        .post(url)
        .json(&body)
        .send()
        .await
        .unwrap();
    let status = response.status().as_u16();
    (status, response.json().await.unwrap())
}

#[tokio::test]
async fn tei_embed_returns_one_vector_per_input_in_order() {
    let h = spawn().await;
    let (status, body) = post(&format!("{}/embed", h.url), json!({"inputs": ["a", "bb"]})).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body, json!([[1.0, 97.0, 0.5], [2.0, 98.0, 0.5]]));
    // A bare string is one input.
    let (status, body) = post(&format!("{}/embed", h.url), json!({"inputs": "c"})).await;
    assert_eq!((status, body), (200, json!([[1.0, 99.0, 0.5]])));
}

#[tokio::test]
async fn prefixes_are_caller_supplied_unless_input_type_is_given() {
    let h = spawn().await;
    post(&format!("{}/embed", h.url), json!({"inputs": ["x"]})).await;
    post(
        &format!("{}/embed", h.url),
        json!({"inputs": ["x"], "input_type": "query"}),
    )
    .await;
    post(
        &format!("{}/v1/embeddings", h.url),
        json!({"input": ["x"], "model": "Fake/Model-1", "input_type": "passage"}),
    )
    .await;
    let seen = h.engine.seen.lock().unwrap().clone();
    assert_eq!(
        seen,
        vec![
            vec!["x".to_string()],
            vec!["Q: x".into()],
            vec!["P: x".into()]
        ]
    );
    let (status, body) = post(
        &format!("{}/embed", h.url),
        json!({"inputs": ["x"], "input_type": "document"}),
    )
    .await;
    assert_eq!(status, 422, "{body}");
}

#[tokio::test]
async fn openai_embeddings_match_the_openai_shape() {
    let h = spawn().await;
    let (status, body) = post(
        &format!("{}/v1/embeddings", h.url),
        json!({"model": "Fake/Model-1", "input": ["a b", "c"]}),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["object"], "list");
    assert_eq!(body["model"], "Fake/Model-1");
    assert_eq!(body["data"][0]["index"], 0);
    assert_eq!(body["data"][1]["index"], 1);
    assert_eq!(body["data"][1]["object"], "embedding");
    assert_eq!(body["data"][1]["embedding"], json!([1.0, 99.0, 0.5]));
    assert_eq!(body["usage"]["prompt_tokens"], 3);
    assert_eq!(body["usage"]["total_tokens"], 3);
}

#[tokio::test]
async fn openai_base64_encoding_decodes_to_the_float_vector() {
    let h = spawn().await;
    let (status, body) = post(
        &format!("{}/v1/embeddings", h.url),
        json!({"model": "Fake/Model-1", "input": "ab", "encoding_format": "base64"}),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(body["data"][0]["embedding"].as_str().unwrap())
        .unwrap();
    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(floats, vec![2.0, 97.0, 0.5]);
}

#[tokio::test]
async fn openai_model_must_name_the_served_model() {
    let h = spawn().await;
    for model in ["Fake/Model-1", "native:Fake/Model-1", "fake/model-1"] {
        let (status, body) = post(
            &format!("{}/v1/embeddings", h.url),
            json!({"model": model, "input": "a"}),
        )
        .await;
        assert_eq!(status, 200, "{model}: {body}");
    }
    let (status, body) = post(
        &format!("{}/v1/embeddings", h.url),
        json!({"model": "text-embedding-3-small", "input": "a"}),
    )
    .await;
    assert_eq!(status, 404, "{body}");
    assert_eq!(body["error"]["code"], "model_not_found");
    let (status, body) = post(
        &format!("{}/v1/embeddings", h.url),
        json!({"model": "Fake/Model-1", "input": "a", "dimensions": 2}),
    )
    .await;
    assert_eq!(status, 400, "{body}");
}

#[tokio::test]
async fn inputs_over_the_context_are_rejected_unless_truncation_is_requested() {
    let h = spawn().await;
    let long = "w w w w w"; // 5 tokens > max_length 4
    let (status, body) = post(&format!("{}/embed", h.url), json!({"inputs": ["a", long]})).await;
    assert_eq!(status, 413, "{body}");
    let message = body["error"].as_str().unwrap();
    assert!(message.contains("must have less than"), "{message}");
    assert_eq!(body["error_type"], "Validation");
    assert!(h.engine.seen.lock().unwrap().is_empty());

    let (status, _) = post(
        &format!("{}/embed", h.url),
        json!({"inputs": [long], "truncate": true}),
    )
    .await;
    assert_eq!(status, 200);

    let (status, body) = post(
        &format!("{}/v1/embeddings", h.url),
        json!({"model": "Fake/Model-1", "input": [long]}),
    )
    .await;
    assert_eq!(status, 413, "{body}");
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("maximum context length"));
    // The prefix counts against the context: "Q: w w" is 3 words <= 4, "Q: w w w w" is 5.
    let (status, _) = post(
        &format!("{}/embed", h.url),
        json!({"inputs": ["w w w w"], "input_type": "query"}),
    )
    .await;
    assert_eq!(status, 413);
}

#[tokio::test]
async fn request_size_limits_return_structured_errors() {
    let h = spawn().await;
    let many: Vec<String> = (0..9).map(|i| i.to_string()).collect();
    let (status, body) = post(&format!("{}/embed", h.url), json!({ "inputs": many })).await;
    assert_eq!(status, 413, "{body}");
    assert!(body["error"]
        .as_str()
        .unwrap()
        .contains("maximum allowed batch size 8"));

    let huge = "x".repeat(70 * 1024);
    let (status, body) = post(&format!("{}/embed", h.url), json!({ "inputs": [huge] })).await;
    assert_eq!(status, 413, "{body}");
    assert_eq!(body["error_type"], "Validation");

    let (status, body) = post(
        &format!("{}/v1/embeddings", h.url),
        json!({"model": "Fake/Model-1", "input": ["x".repeat(70 * 1024)]}),
    )
    .await;
    assert_eq!(status, 413, "{body}");
    assert_eq!(body["error"]["type"], "invalid_request_error");

    let (status, body) = post(&format!("{}/embed", h.url), json!({ "inputs": [] })).await;
    assert_eq!(status, 422, "{body}");
}

#[tokio::test]
async fn malformed_requests_get_errors_and_the_server_keeps_serving() {
    let h = spawn().await;
    let client = reqwest::Client::new();
    let response = client
        .post(format!("{}/embed", h.url))
        .header("content-type", "application/json")
        .body("{not json")
        .send()
        .await
        .unwrap();
    assert_eq!(response.status().as_u16(), 400);
    let body: Value = response.json().await.unwrap();
    assert!(body["error"].is_string(), "{body}");

    let (status, body) = post(&format!("{}/embed", h.url), json!({"inputs": [1, 2]})).await;
    assert_eq!(status, 422, "{body}");
    let (status, body) = post(
        &format!("{}/v1/embeddings", h.url),
        json!({"model": "Fake/Model-1", "input": [[1, 2, 3]]}),
    )
    .await;
    assert_eq!(status, 400, "{body}");
    assert!(body["error"]["message"].is_string());
    let (status, body) = post(
        &format!("{}/embed", h.url),
        json!({"inputs": ["a"], "normalize": false}),
    )
    .await;
    assert_eq!(status, 422, "{body}");

    let (status, _) = post(&format!("{}/embed", h.url), json!({"inputs": ["ok"]})).await;
    assert_eq!(status, 200);
}

#[tokio::test]
async fn health_reports_the_loaded_model() {
    let h = spawn().await;
    let body: Value = reqwest::get(format!("{}/health", h.url))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(body["status"], "ok");
    assert_eq!(body["model_id"], "Fake/Model-1");
    assert_eq!(body["dimension"], 3);
    assert_eq!(body["dtype"], "f32");
    assert_eq!(body["device"], "cpu");
    assert_eq!(body["instance"], "test-instance");
    assert_eq!(body["pid"], std::process::id());
}

#[tokio::test]
async fn a_stuck_forward_pass_answers_503_and_stops_the_server() {
    let engine = FakeEngine::default();
    engine.timeout.store(true, Ordering::SeqCst);
    let h = spawn_with(engine, |_| {}).await;
    let (status, body) = post(&format!("{}/embed", h.url), json!({"inputs": ["a"]})).await;
    assert_eq!(status, 503, "{body}");
    let reason = tokio::time::timeout(Duration::from_secs(5), h.stopped)
        .await
        .expect("server stops after a wedged pass")
        .unwrap()
        .unwrap();
    assert_eq!(reason, StopReason::Wedged);
    assert_eq!(reason.exit_code(), 7);
}

#[tokio::test]
async fn idle_timeout_stops_the_server_but_not_during_a_request() {
    let engine = FakeEngine {
        delay: Some(Duration::from_millis(700)),
        ..FakeEngine::default()
    };
    // Built first: constructing a client can take longer than the timeout.
    let client = reqwest::Client::new();
    let h = spawn_with(engine, |o| {
        o.idle_timeout = Some(Duration::from_millis(300))
    })
    .await;
    let started = Instant::now();
    // The request outlasts the idle timeout; it must still complete.
    let response = client
        .post(format!("{}/embed", h.url))
        .json(&json!({"inputs": ["a"]}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status().as_u16(), 200);
    assert!(started.elapsed() >= Duration::from_millis(700));
    // Health probes do not keep the server alive.
    let probe = h.url.clone();
    tokio::spawn(async move {
        loop {
            let _ = reqwest::get(format!("{probe}/health")).await;
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    });
    let reason = tokio::time::timeout(Duration::from_secs(5), h.stopped)
        .await
        .expect("idle server stops")
        .unwrap()
        .unwrap();
    assert_eq!(reason, StopReason::Idle);
    assert!(started.elapsed() >= Duration::from_millis(1000));
}

#[tokio::test]
async fn pid_and_port_files_exist_while_serving_and_are_removed_on_stop() {
    let dir = tempfile::tempdir().unwrap();
    let Started::Serving(server) = start(Arc::new(FakeEngine::default()), options(dir.path()))
        .await
        .unwrap()
    else {
        panic!("expected a fresh server");
    };
    let port = server.health().port;
    let read = |name: &str| std::fs::read_to_string(dir.path().join(name)).unwrap();
    assert_eq!(read("port").trim(), port.to_string());
    assert_eq!(read("pid").trim(), std::process::id().to_string());
    assert_eq!(read("host").trim(), "127.0.0.1");
    let reason = server.run_until(async {}).await.unwrap();
    assert_eq!(reason, StopReason::Signal);
    assert!(!dir.path().join("port").exists());
    assert!(!dir.path().join("pid").exists());
}

#[tokio::test]
async fn a_second_start_adopts_the_running_server_without_loading() {
    struct NeverLoad;
    #[async_trait::async_trait]
    impl ServeEngine for NeverLoad {
        async fn load(&self) -> Result<EngineInfo> {
            panic!("an adopting process must not load a model");
        }
        async fn token_counts(&self, _: &[String]) -> Result<Vec<usize>> {
            unreachable!()
        }
        async fn embed(&self, _: Vec<String>) -> Result<Vec<Vec<f32>>> {
            unreachable!()
        }
    }
    let h = spawn().await;
    let dir = h._dir.path().to_path_buf();
    let adopted = start(Arc::new(NeverLoad), options(&dir)).await.unwrap();
    let Started::Adopted(health) = adopted else {
        panic!("expected adoption");
    };
    assert_eq!(health.url, h.url);
    assert_eq!(health.model_id, "Fake/Model-1");
    h.stop.notify_one();
    assert_eq!(h.stopped.await.unwrap().unwrap(), StopReason::Signal);
}

#[tokio::test]
async fn stale_files_from_a_dead_server_are_replaced() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("port"), "1\n").unwrap();
    std::fs::write(dir.path().join("pid"), "999999\n").unwrap();
    let started = start(Arc::new(FakeEngine::default()), options(dir.path()))
        .await
        .unwrap();
    let Started::Serving(server) = started else {
        panic!("a dead owner leaves the lock free, so this process serves");
    };
    let port = std::fs::read_to_string(dir.path().join("port")).unwrap();
    assert_eq!(port.trim(), server.health().port.to_string());
}

#[test]
fn only_loopback_hosts_are_accepted() {
    for ok in ["127.0.0.1", "::1", "localhost", "127.0.0.2"] {
        assert!(parse_loopback_host(ok).is_ok(), "{ok}");
    }
    for bad in ["0.0.0.0", "::", "192.168.1.10", "example.com", ""] {
        let error = parse_loopback_host(bad).unwrap_err();
        assert!(matches!(error, MinSyncError::Config(_)), "{bad}: {error}");
    }
}

#[test]
fn instance_key_follows_what_changes_the_vectors() {
    let base = crate::config::Config::default_for("s").embedder;
    let key = instance_key(&base).unwrap();
    let mut prefixed = base.clone();
    prefixed.query_prefix = Some("q: ".into());
    prefixed.batch_size = 7;
    prefixed.timeout_seconds = 5;
    assert_eq!(instance_key(&prefixed).unwrap(), key);
    let mut explicit_auto = base.clone();
    explicit_auto.dtype = Some("auto".into());
    explicit_auto.device = Some("auto".into());
    assert_eq!(instance_key(&explicit_auto).unwrap(), key);
    for change in [
        |c: &mut crate::config::EmbedderConfig| c.dtype = Some("f32".into()),
        |c: &mut crate::config::EmbedderConfig| c.device = Some("cpu".into()),
        |c: &mut crate::config::EmbedderConfig| c.max_length = Some(512),
        |c: &mut crate::config::EmbedderConfig| c.attention = Some("eager".into()),
        |c: &mut crate::config::EmbedderConfig| c.id = "native:Other/Model".into(),
    ] {
        let mut other = base.clone();
        change(&mut other);
        assert_ne!(instance_key(&other).unwrap(), key);
    }
    let mut tei = base;
    tei.id = "tei:foo".into();
    assert!(instance_key(&tei).is_err());
}

// ---- sparse (SPLADE) engine: POST /embed_sparse ------------------------------

use crate::sparse::SparseVector;

/// Encodes a text as one term per distinct byte: `{byte: 1 + position-in-text-order}`.
#[derive(Default)]
struct FakeSparse {
    seen: StdMutex<Vec<Vec<String>>>,
    timeout: AtomicBool,
}

#[async_trait::async_trait]
impl SparseServeEngine for FakeSparse {
    async fn load(&self) -> Result<EngineInfo> {
        Ok(EngineInfo {
            model_id: "Fake/Sparse-1".to_string(),
            dimension: 256,
            dtype: "f32".to_string(),
            device: "cpu".to_string(),
            attention: "tiled".to_string(),
            max_length: 4,
        })
    }

    async fn token_counts(&self, texts: &[String]) -> Result<Vec<usize>> {
        Ok(texts.iter().map(|t| t.split_whitespace().count()).collect())
    }

    async fn encode(&self, texts: Vec<String>) -> Result<Vec<SparseVector>> {
        if self.timeout.load(Ordering::SeqCst) {
            return Err(MinSyncError::Timeout("native sparse batch stuck".into()));
        }
        self.seen.lock().unwrap().push(texts.clone());
        Ok(texts
            .iter()
            .map(|t| {
                SparseVector::from_pairs(
                    t.bytes()
                        .enumerate()
                        .map(|(i, b)| (u32::from(b), 1.0 + i as f32)),
                )
            })
            .collect())
    }
}

struct SparseHarness {
    url: String,
    engine: Arc<FakeSparse>,
    stop: Arc<tokio::sync::Notify>,
    stopped: tokio::task::JoinHandle<Result<StopReason>>,
    _dir: tempfile::TempDir,
}

async fn spawn_sparse(engine: FakeSparse) -> SparseHarness {
    let dir = tempfile::tempdir().unwrap();
    let opts = options(dir.path());
    let engine = Arc::new(engine);
    let Started::Serving(server) = start_sparse(engine.clone(), opts).await.unwrap() else {
        panic!("expected a fresh server");
    };
    let url = server.health().url.clone();
    let stop = Arc::new(tokio::sync::Notify::new());
    let signal = stop.clone();
    let stopped = tokio::spawn(async move {
        server
            .run_until(async move { signal.notified().await })
            .await
    });
    SparseHarness {
        url,
        engine,
        stop,
        stopped,
        _dir: dir,
    }
}

#[tokio::test]
async fn embed_sparse_returns_tei_index_value_rows_in_order() {
    let h = spawn_sparse(FakeSparse::default()).await;
    let (status, body) = post(
        &format!("{}/embed_sparse", h.url),
        json!({"inputs": ["ab", "b"]}),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    // 'a' = 97 weight 1, 'b' = 98 weight 2 ; second text: 'b' weight 1.
    assert_eq!(
        body,
        json!([
            [{"index": 97, "value": 1.0}, {"index": 98, "value": 2.0}],
            [{"index": 98, "value": 1.0}],
        ])
    );
    let (status, body) = post(&format!("{}/embed_sparse", h.url), json!({"inputs": "a"})).await;
    assert_eq!(
        (status, body),
        (200, json!([[{"index": 97, "value": 1.0}]]))
    );
}

#[tokio::test]
async fn embed_sparse_applies_prefixes_only_with_input_type_and_enforces_the_context() {
    let h = spawn_sparse(FakeSparse::default()).await;
    post(&format!("{}/embed_sparse", h.url), json!({"inputs": ["x"]})).await;
    post(
        &format!("{}/embed_sparse", h.url),
        json!({"inputs": ["x"], "input_type": "query"}),
    )
    .await;
    post(
        &format!("{}/embed_sparse", h.url),
        json!({"inputs": ["x"], "input_type": "passage"}),
    )
    .await;
    assert_eq!(
        h.engine.seen.lock().unwrap().clone(),
        vec![
            vec!["x".to_string()],
            vec!["Q: x".to_string()],
            vec!["P: x".to_string()]
        ]
    );

    let long = "w w w w w"; // 5 tokens > max_length 4
    let (status, body) = post(
        &format!("{}/embed_sparse", h.url),
        json!({"inputs": [long]}),
    )
    .await;
    assert_eq!(status, 413, "{body}");
    assert!(body["error"]
        .as_str()
        .unwrap()
        .contains("must have less than"));
    let (status, _) = post(
        &format!("{}/embed_sparse", h.url),
        json!({"inputs": [long], "truncate": true}),
    )
    .await;
    assert_eq!(status, 200);
}

#[tokio::test]
async fn embed_sparse_rejects_bad_requests_and_keeps_serving() {
    let h = spawn_sparse(FakeSparse::default()).await;
    let url = format!("{}/embed_sparse", h.url);
    let many: Vec<String> = (0..9).map(|i| i.to_string()).collect();
    assert_eq!(post(&url, json!({"inputs": many})).await.0, 413);
    assert_eq!(post(&url, json!({"inputs": []})).await.0, 422);
    assert_eq!(post(&url, json!({"inputs": [1, 2]})).await.0, 422);
    assert_eq!(
        post(&url, json!({"inputs": ["a"], "prompt_name": "query"}))
            .await
            .0,
        422,
        "prompt_name is unsupported: send prefixed text or input_type"
    );
    assert_eq!(post(&url, json!({"inputs": ["ok"]})).await.0, 200);
}

#[tokio::test]
async fn a_sparse_server_does_not_serve_dense_routes_and_reports_its_kind() {
    let h = spawn_sparse(FakeSparse::default()).await;
    let client = reqwest::Client::new();
    let health: Value = client
        .get(format!("{}/health", h.url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(health["status"], "ok");
    assert_eq!(health["kind"], "sparse");
    assert_eq!(health["model_id"], "Fake/Sparse-1");
    for route in ["/embed", "/v1/embeddings"] {
        let response = client
            .post(format!("{}{route}", h.url))
            .json(&json!({"inputs": ["a"], "input": ["a"]}))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status().as_u16(), 404, "{route}");
    }
}

#[tokio::test]
async fn a_dense_server_reports_kind_dense_and_has_no_embed_sparse() {
    let h = spawn().await;
    let health: Value = reqwest::get(format!("{}/health", h.url))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(health["kind"], "dense");
    // axum's default 404 has an empty body, so check the status directly.
    let response = reqwest::Client::new()
        .post(format!("{}/embed_sparse", h.url))
        .json(&json!({"inputs": ["a"]}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status().as_u16(), 404);
}

#[tokio::test]
async fn a_wedged_sparse_pass_shuts_the_server_down_with_the_timeout_exit_code() {
    let engine = FakeSparse::default();
    engine.timeout.store(true, Ordering::SeqCst);
    let h = spawn_sparse(engine).await;
    let (status, body) = post(&format!("{}/embed_sparse", h.url), json!({"inputs": ["a"]})).await;
    assert_eq!(status, 503, "{body}");
    let reason = tokio::time::timeout(Duration::from_secs(5), h.stopped)
        .await
        .expect("server stops")
        .unwrap()
        .unwrap();
    assert_eq!(reason, StopReason::Wedged);
    drop(h.stop);
}

#[test]
fn sparse_instance_key_differs_from_a_dense_one_and_tracks_vector_inputs() {
    let sparse = crate::config::SparseConfig::default();
    let key = sparse_instance_key(&sparse).unwrap();
    assert!(
        key.starts_with("nlpai-lab-KURE-Sparse-nano-sparse-"),
        "{key}"
    );
    let mut other = sparse.clone();
    other.max_length = Some(512);
    assert_ne!(key, sparse_instance_key(&other).unwrap());
    // Batching knobs and prefixes do not change the vectors.
    other = sparse.clone();
    other.batch_size = 3;
    other.query_prefix = None;
    assert_eq!(key, sparse_instance_key(&other).unwrap());
    other.id = "tei:foo/bar".to_string();
    assert!(sparse_instance_key(&other).is_err());
}
