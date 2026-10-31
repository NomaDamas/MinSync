//! CLI-level contract for issue #97 (2): an embedder that starts failing
//! after an earlier batch succeeded must abort the run after a few
//! consecutive failures instead of retrying every remaining file.

use assert_cmd::Command;
use minsync::config::Config;
use serde_json::Value;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

mod common;

const DIMENSION: usize = 4;
const FILES: usize = 30;

/// Healthy for the first request, then HTTP 504 forever.
struct DiesAfterFirst {
    requests: Arc<AtomicUsize>,
}

impl Respond for DiesAfterFirst {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        if self.requests.fetch_add(1, Ordering::SeqCst) >= 1 {
            return ResponseTemplate::new(504).set_body_string("gateway timeout");
        }
        let body: Value = serde_json::from_slice(&request.body).expect("TEI request json");
        let vectors: Vec<Vec<f32>> = body["inputs"]
            .as_array()
            .expect("inputs array")
            .iter()
            .map(|_| vec![0.5; DIMENSION])
            .collect();
        ResponseTemplate::new(200).set_body_json(vectors)
    }
}

fn minsync(root: &Path) -> Command {
    let mut command = Command::cargo_bin("minsync").expect("find minsync binary");
    command.current_dir(root);
    command
}

fn setup(root: &Path, base_url: &str) {
    minsync(root)
        .args([
            "init",
            "--embedder",
            "tei:test/model",
            "--chunker",
            "recursive",
        ])
        .assert()
        .success();
    let config_path = root.join(".minsync/config.toml");
    let mut config = Config::load(&config_path).expect("load config");
    config.embedder.base_url = Some(base_url.to_string());
    config.embedder.max_retries = 0;
    config.sync.batch_files = 2;
    let mut options = toml::value::Table::new();
    options.insert("dimension".into(), toml::Value::Integer(DIMENSION as i64));
    config.vectorstore.options = toml::Value::Table(options);
    common::use_tiny_sparse(&mut config);
    config.save(&config_path).expect("save config");
}

#[tokio::test(flavor = "multi_thread")]
async fn outage_after_a_successful_batch_aborts_with_exit_5_after_a_few_requests() {
    let requests = Arc::new(AtomicUsize::new(0));
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embed"))
        .respond_with(DiesAfterFirst {
            requests: Arc::clone(&requests),
        })
        .mount(&server)
        .await;
    let dir = tempfile::tempdir().unwrap();
    setup(dir.path(), &server.uri());
    for index in 0..FILES {
        std::fs::write(
            dir.path().join(format!("f{index:02}.txt")),
            format!("document number {index}"),
        )
        .unwrap();
    }

    let root = dir.path().to_path_buf();
    let output = tokio::task::spawn_blocking(move || {
        minsync(&root).arg("sync").output().expect("run minsync")
    })
    .await
    .expect("join");

    assert_eq!(output.status.code(), Some(5), "{output:?}");
    let sent = requests.load(Ordering::SeqCst);
    // 1 healthy + (batch commit + a few per-file retries) before aborting;
    // without the fix every remaining batch and file is retried (> 30).
    assert!(sent <= 10, "embedder was hit {sent} times");
    let progress = std::fs::read_to_string(dir.path().join(".minsync/progress.jsonl"))
        .expect("committed batch is kept in the resume log");
    assert!(!progress.trim().is_empty());
}
