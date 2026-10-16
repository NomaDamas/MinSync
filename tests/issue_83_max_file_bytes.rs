//! CLI-level coverage for issue #83: cap source-file reads and report skips.

use assert_cmd::Command;
use minsync::config::Config;
use serde_json::{json, Value};
use std::path::Path;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

const DIMENSION: usize = 4;

struct FixedTei;

impl Respond for FixedTei {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let body: Value = serde_json::from_slice(&request.body).expect("TEI request json");
        let inputs = body["inputs"].as_array().expect("inputs array");
        let vectors = vec![vec![0.5; DIMENSION]; inputs.len()];
        ResponseTemplate::new(200).set_body_json(vectors)
    }
}

fn minsync(root: &Path) -> Command {
    let mut command = Command::cargo_bin("minsync").expect("find minsync binary");
    command.current_dir(root);
    command
}

fn setup(root: &Path, base_url: &str, max_file_bytes: u64) {
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
    config.embedder.max_retries = 1;
    config.sync.max_file_bytes = max_file_bytes;
    let mut options = toml::value::Table::new();
    options.insert("dimension".into(), toml::Value::Integer(DIMENSION as i64));
    config.vectorstore.options = toml::Value::Table(options);
    config.save(&config_path).expect("save config");
}

async fn tei_server() -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embed"))
        .respond_with(FixedTei)
        .mount(&server)
        .await;
    server
}

async fn run(root: &Path, args: &[&str]) -> std::process::Output {
    let root = root.to_path_buf();
    let args: Vec<String> = args.iter().map(|arg| arg.to_string()).collect();
    tokio::task::spawn_blocking(move || minsync(&root).args(&args).output().expect("run minsync"))
        .await
        .expect("join")
}

#[tokio::test(flavor = "multi_thread")]
async fn oversized_file_is_reported_while_other_files_are_indexed() {
    let server = tei_server().await;
    let dir = tempfile::tempdir().unwrap();
    setup(dir.path(), &server.uri(), 16);
    std::fs::write(dir.path().join("good.txt"), "small text").unwrap();
    std::fs::write(
        dir.path().join("large.txt"),
        "this file exceeds the configured limit",
    )
    .unwrap();

    let output = run(dir.path(), &["--format", "json", "sync"]).await;

    assert_eq!(output.status.code(), Some(6), "{output:?}");
    let result: Value = serde_json::from_slice(&output.stdout).expect("json result");
    assert_eq!(result["partial"], json!(true));
    assert_eq!(result["files_processed_paths"], json!(["good.txt"]));
    assert_eq!(result["files_failed"][0]["path"], json!("large.txt"));
    assert_eq!(result["files_failed"][0]["stage"], json!("size_guard"));
    assert!(result["files_failed"][0]["error"]
        .as_str()
        .unwrap()
        .contains("over max_file_bytes (16)"));
    assert_eq!(result["scan_errors"][0]["kind"], json!("size_guard"));

    let manifest: Value = serde_json::from_slice(
        &std::fs::read(dir.path().join(".minsync/manifest.json")).expect("read manifest"),
    )
    .expect("manifest json");
    assert!(manifest["files"].get("large.txt").is_none());
    assert!(manifest["files"].get("good.txt").is_some());

    let query = run(
        dir.path(),
        &["--format", "json", "query", "small", "--mode", "bm25"],
    )
    .await;
    assert!(query.status.success(), "{query:?}");
    let hits: Value = serde_json::from_slice(&query.stdout).expect("query json");
    let paths: Vec<&str> = hits
        .as_array()
        .unwrap()
        .iter()
        .map(|hit| hit["path"].as_str().unwrap())
        .collect();
    assert!(paths.contains(&"good.txt"), "{paths:?}");
    assert!(!paths.contains(&"large.txt"), "{paths:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn zero_disables_the_guard_and_lowering_the_limit_preserves_existing_chunks() {
    let server = tei_server().await;
    let dir = tempfile::tempdir().unwrap();
    setup(dir.path(), &server.uri(), 0);
    let contents = format!("{} searchablemarker", "large ".repeat(40));
    std::fs::write(dir.path().join("large.txt"), &contents).unwrap();

    let initial = run(dir.path(), &["--format", "json", "sync"]).await;
    assert_eq!(initial.status.code(), Some(0), "{initial:?}");
    let initial_result: Value = serde_json::from_slice(&initial.stdout).expect("json result");
    assert!(initial_result["files_processed_paths"]
        .as_array()
        .unwrap()
        .iter()
        .any(|path| path == "large.txt"));

    let config_path = dir.path().join(".minsync/config.toml");
    let mut config = Config::load(&config_path).expect("load config");
    config.sync.max_file_bytes = 8;
    config.save(&config_path).expect("save lower limit");

    let limited = run(dir.path(), &["--format", "json", "sync", "--full"]).await;
    assert_eq!(limited.status.code(), Some(6), "{limited:?}");
    let limited_result: Value = serde_json::from_slice(&limited.stdout).expect("json result");
    assert_eq!(
        limited_result["files_failed"][0]["stage"],
        json!("size_guard")
    );
    assert!(limited_result["scan_errors"]
        .as_array()
        .unwrap()
        .iter()
        .any(|error| { error["path"] == "large.txt" && error["kind"] == "size_guard" }));
    let manifest: Value = serde_json::from_slice(
        &std::fs::read(dir.path().join(".minsync/manifest.json")).expect("read manifest"),
    )
    .expect("manifest json");
    assert!(manifest["files"].get("large.txt").is_some());

    let query = run(
        dir.path(),
        &[
            "--format",
            "json",
            "query",
            "searchablemarker",
            "--mode",
            "bm25",
        ],
    )
    .await;
    assert!(query.status.success(), "{query:?}");
    let hits: Value = serde_json::from_slice(&query.stdout).expect("query json");
    assert!(hits
        .as_array()
        .unwrap()
        .iter()
        .any(|hit| hit["path"] == "large.txt"));
}
