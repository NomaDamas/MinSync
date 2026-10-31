//! CLI-level contract for issue #97 (3): `chunker.options.max_chunk_size` is a
//! hard upper bound for markdown tables too, so no embedding request ever
//! carries an input larger than the configured cap.

use assert_cmd::Command;
use minsync::config::Config;
use serde_json::Value;
use std::path::Path;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

mod common;

const DIMENSION: usize = 4;
const MAX_CHUNK_SIZE: usize = 200;

struct FixedTei;

impl Respond for FixedTei {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let body: Value = serde_json::from_slice(&request.body).expect("TEI request json");
        let inputs = body["inputs"].as_array().expect("inputs array");
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
    config.embedder.max_retries = 0;
    config.chunker.options.max_chunk_size = MAX_CHUNK_SIZE;
    let mut options = toml::value::Table::new();
    options.insert("dimension".into(), toml::Value::Integer(DIMENSION as i64));
    config.vectorstore.options = toml::Value::Table(options);
    common::use_tiny_sparse(&mut config);
    config.save(&config_path).expect("save config");
}

#[tokio::test(flavor = "multi_thread")]
async fn wide_markdown_table_never_sends_an_input_over_max_chunk_size() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embed"))
        .respond_with(FixedTei)
        .mount(&server)
        .await;
    let dir = tempfile::tempdir().unwrap();
    setup(dir.path(), &server.uri());
    // Unique tokens so no two capped chunks share a content hash (identical
    // chunks are embedded once and would hide dropped content).
    let blob: String = (0..1_000).map(|index| format!("{index:04}")).collect();
    std::fs::write(
        dir.path().join("wide.md"),
        format!("| id | payload |\n| --- | --- |\n| 1 | {blob} |\n| 2 | small |\n"),
    )
    .unwrap();

    let root = dir.path().to_path_buf();
    let output = tokio::task::spawn_blocking(move || {
        minsync(&root).arg("sync").output().expect("run minsync")
    })
    .await
    .expect("join");

    assert!(output.status.success(), "{output:?}");
    let mut longest = 0;
    let mut digits = 0;
    for request in server.received_requests().await.unwrap() {
        let body: Value = serde_json::from_slice(&request.body).unwrap();
        for input in body["inputs"].as_array().unwrap() {
            let text = input.as_str().unwrap();
            longest = longest.max(text.chars().count());
            digits += text.chars().filter(char::is_ascii_digit).count();
        }
    }
    assert!(longest > 0, "nothing was embedded");
    assert!(
        longest <= MAX_CHUNK_SIZE,
        "longest embedded input was {longest} chars (cap {MAX_CHUNK_SIZE})"
    );
    // 4000 blob digits + the row ids `1` and `2`.
    assert_eq!(digits, 4_002, "table content must not be dropped");
}
