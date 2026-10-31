//! CLI-level contract for issue #78: a file that fails to read, chunk or
//! embed is skipped and reported instead of aborting the whole sync.

use assert_cmd::Command;
use minsync::config::{Config, OnFileError};
use serde_json::{json, Value};
use std::path::Path;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

mod common;

const DIMENSION: usize = 4;

/// TEI stand-in: rejects any request containing `POISON`, else returns a
/// fixed vector per input.
struct PoisonAwareTei;

impl Respond for PoisonAwareTei {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let body: Value = serde_json::from_slice(&request.body).expect("TEI request json");
        let inputs = body["inputs"].as_array().expect("inputs array");
        if inputs
            .iter()
            .any(|input| input.as_str().is_some_and(|text| text.contains("POISON")))
        {
            return ResponseTemplate::new(400).set_body_string("input rejected");
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

fn setup(root: &Path, base_url: &str, on_file_error: OnFileError) {
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
    config.sync.on_file_error = on_file_error;
    let mut options = toml::value::Table::new();
    options.insert("dimension".into(), toml::Value::Integer(DIMENSION as i64));
    config.vectorstore.options = toml::Value::Table(options);
    common::use_tiny_sparse(&mut config);
    config.save(&config_path).expect("save config");
}

fn write_files(root: &Path) {
    // The tiny checkpoint only knows the `w<n>` vocabulary, so give the files
    // a queryable term the sparse query below shares with them.
    for name in ["a", "b", "d", "e"] {
        std::fs::write(
            root.join(format!("{name}.txt")),
            format!("w1 w2 good {name}"),
        )
        .unwrap();
    }
    std::fs::write(root.join("c-bad.txt"), "POISON w1 text").unwrap();
}

async fn tei_server() -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embed"))
        .respond_with(PoisonAwareTei)
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
async fn poison_file_is_skipped_with_exit_6_and_reported_in_json() {
    let server = tei_server().await;
    let dir = tempfile::tempdir().unwrap();
    setup(dir.path(), &server.uri(), OnFileError::Skip);
    write_files(dir.path());

    let output = run(dir.path(), &["--format", "json", "sync"]).await;

    assert_eq!(output.status.code(), Some(6), "{output:?}");
    let result: Value = serde_json::from_slice(&output.stdout).expect("json result");
    assert_eq!(result["partial"], json!(true));
    assert_eq!(result["files_processed"], json!(4));
    let failed = result["files_failed"].as_array().unwrap();
    assert_eq!(failed.len(), 1);
    assert_eq!(failed[0]["path"], json!("c-bad.txt"));
    assert_eq!(failed[0]["stage"], json!("embed"));
    assert!(failed[0]["error"]
        .as_str()
        .unwrap()
        .contains("input rejected"));

    let query = run(
        dir.path(),
        &["--format", "json", "query", "w1", "--mode", "sparse"],
    )
    .await;
    assert!(query.status.success(), "{query:?}");
    let hits: Value = serde_json::from_slice(&query.stdout).unwrap();
    let paths: Vec<&str> = hits
        .as_array()
        .unwrap()
        .iter()
        .map(|hit| hit["path"].as_str().unwrap())
        .collect();
    assert!(paths.contains(&"a.txt"), "{paths:?}");
    assert!(!paths.contains(&"c-bad.txt"));

    let status = run(dir.path(), &["--format", "json", "status"]).await;
    assert!(status.status.success(), "status works after a partial sync");
}

#[tokio::test(flavor = "multi_thread")]
async fn text_output_names_the_skipped_path_on_stderr() {
    let server = tei_server().await;
    let dir = tempfile::tempdir().unwrap();
    setup(dir.path(), &server.uri(), OnFileError::Skip);
    write_files(dir.path());

    let output = run(dir.path(), &["sync"]).await;

    assert_eq!(output.status.code(), Some(6));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("c-bad.txt"), "{stderr}");
    assert!(stderr.contains("embed"), "{stderr}");
}

#[tokio::test(flavor = "multi_thread")]
async fn fail_mode_keeps_embedding_exit_code_on_batch_embed_error() {
    let server = tei_server().await;
    let dir = tempfile::tempdir().unwrap();
    setup(dir.path(), &server.uri(), OnFileError::Fail);
    write_files(dir.path());

    let output = run(dir.path(), &["sync"]).await;

    assert_eq!(output.status.code(), Some(5), "{output:?}");
}

#[cfg(unix)]
#[tokio::test(flavor = "multi_thread")]
async fn unreadable_file_is_skipped_in_skip_mode_and_fatal_with_path_in_fail_mode() {
    use std::os::unix::fs::PermissionsExt;

    let server = tei_server().await;
    for (mode, expected_code) in [(OnFileError::Skip, 6), (OnFileError::Fail, 1)] {
        let dir = tempfile::tempdir().unwrap();
        setup(dir.path(), &server.uri(), mode);
        std::fs::write(dir.path().join("a.txt"), "readable").unwrap();
        let locked = dir.path().join("locked.txt");
        std::fs::write(&locked, "secret").unwrap();
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();
        if std::fs::read(&locked).is_ok() {
            return; // root ignores file permissions
        }

        let output = run(dir.path(), &["--format", "json", "sync"]).await;

        assert_eq!(
            output.status.code(),
            Some(expected_code),
            "{mode:?}: {output:?}"
        );
        if mode == OnFileError::Skip {
            let result: Value = serde_json::from_slice(&output.stdout).unwrap();
            assert_eq!(result["files_failed"][0]["path"], json!("locked.txt"));
            assert_eq!(result["files_failed"][0]["stage"], json!("read"));
            assert_eq!(result["files_processed_paths"], json!(["a.txt"]));
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(stderr.contains("locked.txt"), "{stderr}");
        }
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o644)).unwrap();
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn clean_sync_still_exits_zero_with_empty_files_failed() {
    let server = tei_server().await;
    let dir = tempfile::tempdir().unwrap();
    setup(dir.path(), &server.uri(), OnFileError::Skip);
    std::fs::write(dir.path().join("a.txt"), "fine").unwrap();

    let output = run(dir.path(), &["--format", "json", "sync"]).await;

    assert_eq!(output.status.code(), Some(0), "{output:?}");
    let result: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["partial"], json!(false));
    assert_eq!(result["files_failed"], json!([]));
}
