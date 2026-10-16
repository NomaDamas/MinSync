//! CLI-level contract for issue #97 (2): when a TEI gateway answers 504 for
//! a large batch, `minsync sync` halves the batch instead of resending the
//! identical batch `max_retries` times.

use assert_cmd::Command;
use minsync::config::Config;
use serde_json::Value;
use std::path::Path;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

const DIMENSION: usize = 4;
const CEILING: usize = 4;

/// 504 for any request with more than `CEILING` inputs.
struct CeilingTei;

impl Respond for CeilingTei {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let body: Value = serde_json::from_slice(&request.body).expect("TEI request json");
        let inputs = body["inputs"].as_array().expect("inputs array");
        if inputs.len() > CEILING {
            return ResponseTemplate::new(504).set_body_string("gateway timeout");
        }
        let vectors: Vec<Vec<f32>> = inputs.iter().map(|_| vec![0.5; DIMENSION]).collect();
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
    config.embedder.batch_size = 64;
    config.embedder.max_retries = 3;
    let mut options = toml::value::Table::new();
    options.insert("dimension".into(), toml::Value::Integer(DIMENSION as i64));
    config.vectorstore.options = toml::Value::Table(options);
    config.save(&config_path).expect("save config");
}

#[tokio::test(flavor = "multi_thread")]
async fn gateway_504_on_a_large_batch_is_split_not_resent() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embed"))
        .respond_with(CeilingTei)
        .mount(&server)
        .await;
    let dir = tempfile::tempdir().unwrap();
    setup(dir.path(), &server.uri());
    for index in 0..32 {
        std::fs::write(
            dir.path().join(format!("f{index:02}.txt")),
            format!("document number {index}"),
        )
        .unwrap();
    }

    let root = dir.path().to_path_buf();
    let output = tokio::task::spawn_blocking(move || {
        minsync(&root)
            .args(["--format", "json", "sync"])
            .output()
            .expect("run minsync")
    })
    .await
    .expect("join");

    assert!(output.status.success(), "{output:?}");
    let result: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["files_failed"], serde_json::json!([]));
    assert_eq!(result["chunks_added"], serde_json::json!(32));

    let mut oversized = 0;
    for request in server.received_requests().await.unwrap() {
        let body: Value = serde_json::from_slice(&request.body).unwrap();
        if body["inputs"].as_array().unwrap().len() > CEILING {
            oversized += 1;
        }
    }
    // 32 -> 16 -> 8 are each probed once; resending would show 4 per size.
    assert!(
        oversized <= 3,
        "{oversized} oversized requests were sent; the batch was resent instead of split"
    );
}
