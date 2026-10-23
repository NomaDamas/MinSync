//! CLI-level contract for issue #120 (1): `minsync check` reports an unhealthy
//! embedder through its exit code, not only through `embedder_ok` in the JSON.
//! A host that looks at the exit code alone used to read a dead embedder as healthy.

use assert_cmd::Command;
use minsync::config::Config;
use serde_json::Value;
use std::path::Path;
use std::process::Output;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const DIMENSION: usize = 4;
/// `MinSyncError::Embedding`, the code `sync` already uses for a failed embedder.
const EXIT_EMBEDDING: i32 = 5;

fn minsync(root: &Path) -> Command {
    let mut command = Command::cargo_bin("minsync").expect("find minsync binary");
    command.current_dir(root);
    command
}

fn setup(root: &Path, base_url: &str) {
    minsync(root)
        .args(["init", "--embedder", "tei:test/model"])
        .assert()
        .success();
    let config_path = root.join(".minsync/config.toml");
    let mut config = Config::load(&config_path).expect("load config");
    config.embedder.base_url = Some(base_url.to_string());
    config.embedder.max_retries = 0;
    let mut options = toml::value::Table::new();
    options.insert("dimension".into(), toml::Value::Integer(DIMENSION as i64));
    config.vectorstore.options = toml::Value::Table(options);
    config.save(&config_path).expect("save config");
}

async fn run_check(root: &Path, args: &[&str]) -> Output {
    let root = root.to_path_buf();
    let args: Vec<String> = args.iter().map(|arg| (*arg).to_string()).collect();
    tokio::task::spawn_blocking(move || {
        minsync(&root)
            .args(&args)
            .arg("check")
            .output()
            .expect("run minsync check")
    })
    .await
    .expect("join")
}

async fn gateway(status: u16, body: Value) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embed"))
        .respond_with(ResponseTemplate::new(status).set_body_json(body))
        .mount(&server)
        .await;
    server
}

#[tokio::test(flavor = "multi_thread")]
async fn a_healthy_embedder_exits_zero() {
    let server = gateway(200, serde_json::json!([[0.5, 0.5, 0.5, 0.5]])).await;
    let dir = tempfile::tempdir().unwrap();
    setup(dir.path(), &server.uri());

    let output = run_check(dir.path(), &["--format", "json"]).await;

    assert_eq!(output.status.code(), Some(0), "{output:?}");
    let result: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["all_passed"], true);
}

#[tokio::test(flavor = "multi_thread")]
async fn an_embedder_that_answers_404_exits_with_the_embedding_code_and_still_prints_the_json() {
    let server = gateway(404, serde_json::json!({"error": {"message": "Not Found"}})).await;
    let dir = tempfile::tempdir().unwrap();
    setup(dir.path(), &server.uri());

    let output = run_check(dir.path(), &["--format", "json"]).await;

    assert_eq!(output.status.code(), Some(EXIT_EMBEDDING), "{output:?}");
    // The report is still on stdout for a host that reads it.
    let result: Value = serde_json::from_slice(&output.stdout).expect("json report on stdout");
    assert_eq!(result["embedder_ok"], false);
    assert_eq!(result["all_passed"], false);
    // The reason is on stderr too, so a host that reads only stderr and the exit code is not left guessing.
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("404"), "stderr: {stderr}");
    // The stage is named once and the cause is not wrapped a second time.
    assert!(
        stderr.starts_with("Error: embedder check failed: "),
        "stderr: {stderr}"
    );
    assert_eq!(
        stderr.matches("embedding failed").count(),
        1,
        "stderr: {stderr}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn the_text_report_fails_the_same_way() {
    let server = gateway(500, serde_json::json!({"error": "boom"})).await;
    let dir = tempfile::tempdir().unwrap();
    setup(dir.path(), &server.uri());

    let output = run_check(dir.path(), &[]).await;

    assert_eq!(output.status.code(), Some(EXIT_EMBEDDING), "{output:?}");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Embedder:    FAIL"), "stdout: {stdout}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_bm25_only_workspace_has_no_embedder_to_fail() {
    let dir = tempfile::tempdir().unwrap();
    minsync(dir.path())
        .args(["init", "--embedder", "none"])
        .assert()
        .success();

    let output = run_check(dir.path(), &["--format", "json"]).await;

    assert_eq!(output.status.code(), Some(0), "{output:?}");
}
