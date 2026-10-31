//! `minsync sync --progress jsonl` against a real child process and a mock
//! TEI embedding server (issue #81).

use assert_cmd::cargo::CommandCargoExt;
use minsync::config::Config;
use serde_json::Value;
use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

mod common;

const EVENTS: [&str; 4] = ["start", "batch", "heartbeat", "done"];

struct SlowOnePerInput(Duration);

impl Respond for SlowOnePerInput {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let body: Value = serde_json::from_slice(&request.body).expect("embed request JSON");
        let n = body["inputs"].as_array().expect("inputs array").len();
        let vectors: Vec<Vec<f32>> = (0..n).map(|_| vec![0.5, 0.5, 0.5, 0.5]).collect();
        ResponseTemplate::new(200)
            .set_body_json(serde_json::json!(vectors))
            .set_delay(self.0)
    }
}

async fn embed_server(delay: Duration) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embed"))
        .respond_with(SlowOnePerInput(delay))
        .mount(&server)
        .await;
    server
}

/// Initialized workspace with `files` one-chunk files, indexed two per batch.
fn workspace(server: &MockServer, files: usize) -> tempfile::TempDir {
    let root = tempfile::tempdir().expect("create workspace");
    let minsync_dir = root.path().join(".minsync");
    std::fs::create_dir_all(&minsync_dir).expect("create state directory");

    let mut config = Config::default_for("source-progress-test");
    config.embedder.id = "tei:test-model".to_string();
    config.embedder.base_url = Some(server.uri());
    config.embedder.max_retries = 0;
    config.sync.batch_files = 2;
    let mut options = toml::value::Table::new();
    options.insert("dimension".into(), toml::Value::Integer(4));
    config.vectorstore.options = toml::Value::Table(options);
    common::use_tiny_sparse(&mut config);
    config
        .save(&minsync_dir.join("config.toml"))
        .expect("save config");

    for index in 0..files {
        std::fs::write(
            root.path().join(format!("note-{index}.md")),
            format!("note number {index}"),
        )
        .expect("write note");
    }
    root
}

fn minsync(root: &std::path::Path) -> Command {
    let mut command = Command::cargo_bin("minsync").expect("find minsync binary");
    command.current_dir(root);
    command
}

fn parse_lines(stdout: &str) -> Vec<Value> {
    stdout
        .lines()
        .map(|line| serde_json::from_str(line).unwrap_or_else(|e| panic!("`{line}`: {e}")))
        .collect()
}

fn events(lines: &[Value]) -> Vec<&Value> {
    lines
        .iter()
        .filter(|line| line.get("event").is_some())
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn progress_jsonl_streams_documented_events_and_matches_final_result() {
    let server = embed_server(Duration::ZERO).await;
    let root = workspace(&server, 5);

    let output = minsync(root.path())
        .args(["--format", "json", "sync", "--progress", "jsonl"])
        .output()
        .expect("run sync");
    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Every stdout line is a standalone JSON object (host can parse by line).
    let lines = parse_lines(&String::from_utf8(output.stdout).expect("utf8 stdout"));
    let stream = events(&lines);
    for event in &stream {
        let name = event["event"].as_str().expect("event name");
        assert!(EVENTS.contains(&name), "undocumented event {name}");
    }

    let start = stream.first().expect("events");
    assert_eq!(start["event"], "start");
    assert_eq!(start["total_files"], 5);
    assert_eq!(start["resumed"], 0);
    assert_eq!(start["full"], true);
    assert_eq!(start["dry_run"], false);
    assert!(start["sync_token"].as_str().is_some_and(|t| !t.is_empty()));

    // 5 files at 2 per batch: 3 committed batches with monotone counters.
    let batches: Vec<_> = stream.iter().filter(|e| e["event"] == "batch").collect();
    assert_eq!(batches.len(), 3);
    let processed: Vec<u64> = batches
        .iter()
        .map(|b| b["processed"].as_u64().expect("processed"))
        .collect();
    assert_eq!(processed, [2, 4, 5]);
    assert!(batches.iter().all(|b| b["total"] == 5));
    assert_eq!(batches[0]["path_first"], "note-0.md");
    assert_eq!(batches[0]["path_last"], "note-1.md");
    assert_eq!(batches[2]["path_first"], "note-4.md");
    assert_eq!(batches[2]["path_last"], "note-4.md");

    // `done` is the last event and its counters equal the final result.
    let done = stream.last().expect("events");
    assert_eq!(done["event"], "done");
    let result = lines.last().expect("result line");
    assert!(result.get("event").is_none(), "result is the trailing line");
    assert_eq!(done["files_processed"], result["files_processed"]);
    assert_eq!(done["files_added"], result["files_added"]);
    assert_eq!(done["files_modified"], result["files_modified"]);
    assert_eq!(done["files_deleted"], result["files_deleted"]);
    assert_eq!(done["files_resumed"], result["files_resumed"]);
    assert_eq!(done["chunks_added"], result["chunks_added"]);
    assert_eq!(done["elapsed_seconds"], result["elapsed_seconds"]);
    assert_eq!(done["processed"], 5);
    assert_eq!(done["total"], 5);
    assert_eq!(result["files_added"], 5);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn progress_is_streamed_incrementally_with_heartbeats_during_a_slow_batch() {
    // Each embed call takes 2.5s; heartbeat every 1s.
    let server = embed_server(Duration::from_millis(2500)).await;
    let root = workspace(&server, 4);

    let mut child = minsync(root.path())
        .args([
            "sync",
            "--progress",
            "jsonl",
            "--progress-heartbeat-seconds",
            "1",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn sync");
    let stdout = child.stdout.take().expect("piped stdout");
    let (tx, rx) = mpsc::channel();
    let reader = std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            let line = line.expect("read line");
            if tx.send((Instant::now(), line)).is_err() {
                break;
            }
        }
    });

    let started = Instant::now();
    let status = child.wait().expect("wait for sync");
    let exited = Instant::now();
    reader.join().expect("reader thread");
    assert!(status.success());

    let received: Vec<(Instant, Value)> = rx
        .try_iter()
        .filter_map(|(at, line)| {
            serde_json::from_str::<Value>(&line)
                .ok()
                .filter(|v| v.get("event").is_some())
                .map(|v| (at, v))
        })
        .collect();

    let first_batch = received
        .iter()
        .find(|(_, e)| e["event"] == "batch")
        .expect("batch event");
    assert!(
        exited.duration_since(first_batch.0) >= Duration::from_secs(2),
        "first batch must reach the host while the second is still running"
    );

    // One 2.5s batch with a 1s heartbeat leaves at least two heartbeats
    // before the first commit.
    let heartbeats_before_batch = received
        .iter()
        .filter(|(at, e)| e["event"] == "heartbeat" && *at < first_batch.0)
        .count();
    assert!(
        heartbeats_before_batch >= 2,
        "got {heartbeats_before_batch} heartbeats in {:?}",
        first_batch.0.duration_since(started)
    );
    let heartbeat = received
        .iter()
        .find(|(_, e)| e["event"] == "heartbeat")
        .expect("heartbeat");
    assert_eq!(heartbeat.1["processed"], 0);
    assert_eq!(heartbeat.1["total"], 4);
    assert_eq!(received.last().expect("events").1["event"], "done");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn progress_file_keeps_stdout_reserved_for_the_final_result() {
    let server = embed_server(Duration::ZERO).await;
    let root = workspace(&server, 3);
    let progress_path = root.path().join("out").join("progress.jsonl");
    std::fs::create_dir_all(progress_path.parent().unwrap()).unwrap();
    // Keep the progress file out of the indexed tree.
    std::fs::write(root.path().join(".minsyncignore"), "out/\n").unwrap();

    let output = minsync(root.path())
        .args(["--format", "json", "sync", "--progress-file"])
        .arg(&progress_path)
        .output()
        .expect("run sync");
    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );

    // stdout is exactly the usual pretty-printed result blob.
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    let result: Value = serde_json::from_str(&stdout).expect("single JSON result");
    // 3 notes + `.minsyncignore` itself (the progress file is ignored).
    assert_eq!(result["files_added"], 4);
    assert!(stdout.contains("\n  \"files_added\""), "pretty printed");

    let lines = parse_lines(&std::fs::read_to_string(&progress_path).expect("progress file"));
    assert_eq!(lines.first().expect("lines")["event"], "start");
    assert_eq!(lines.last().expect("lines")["event"], "done");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn default_sync_output_has_no_progress_events() {
    let server = embed_server(Duration::ZERO).await;
    let root = workspace(&server, 2);

    let output = minsync(root.path())
        .args(["--format", "json", "sync"])
        .output()
        .expect("run sync");
    assert!(output.status.success());

    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    let result: Value = serde_json::from_str(&stdout).expect("single JSON result");
    assert_eq!(result["files_added"], 2);
    assert!(!stdout.contains("\"event\""));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn up_to_date_and_dry_run_still_emit_start_and_done() {
    let server = embed_server(Duration::ZERO).await;
    let root = workspace(&server, 2);

    let dry = minsync(root.path())
        .args(["sync", "--dry-run", "--progress", "jsonl"])
        .output()
        .expect("dry run");
    assert!(dry.status.success());
    let text = String::from_utf8(dry.stdout).unwrap();
    // Text summary moves to stderr so stdout stays machine-readable.
    let lines = parse_lines(&text);
    assert_eq!(lines[0]["event"], "start");
    assert_eq!(lines[0]["dry_run"], true);
    assert_eq!(lines[0]["total_files"], 2);
    assert_eq!(lines[0]["sync_token"], Value::Null);
    assert_eq!(lines.last().unwrap()["event"], "done");
    assert_eq!(lines.last().unwrap()["files_processed"], 0);

    assert!(minsync(root.path())
        .args(["sync"])
        .output()
        .unwrap()
        .status
        .success());
    let again = minsync(root.path())
        .args(["sync", "--progress", "jsonl"])
        .output()
        .expect("second sync");
    let lines = parse_lines(&String::from_utf8(again.stdout).unwrap());
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[0]["total_files"], 0);
    assert_eq!(lines[1]["event"], "done");
    assert_eq!(lines[1]["already_up_to_date"], true);
}

#[test]
fn unwritable_progress_file_fails_before_syncing() {
    let root = tempfile::tempdir().unwrap();
    let minsync_dir = root.path().join(".minsync");
    std::fs::create_dir_all(&minsync_dir).unwrap();
    Config::default_for("s")
        .save(&minsync_dir.join("config.toml"))
        .unwrap();

    let output = minsync(root.path())
        .args([
            "sync",
            "--progress-file",
            "/nonexistent-dir/x/progress.jsonl",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("progress.jsonl"));
}

#[test]
fn progress_file_inside_state_dir_is_rejected_without_truncating_it() {
    let root = tempfile::tempdir().unwrap();
    let minsync_dir = root.path().join(".minsync");
    std::fs::create_dir_all(&minsync_dir).unwrap();
    Config::default_for("s")
        .save(&minsync_dir.join("config.toml"))
        .unwrap();
    let resume_log = minsync_dir.join("progress.jsonl");
    std::fs::write(&resume_log, "resume-state").unwrap();

    let output = minsync(root.path())
        .args(["sync", "--progress-file", ".minsync/progress.jsonl"])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("must not be inside .minsync"));
    assert_eq!(
        std::fs::read_to_string(&resume_log).unwrap(),
        "resume-state"
    );
}

#[test]
fn progress_file_inside_indexed_tree_is_rejected() {
    let root = tempfile::tempdir().unwrap();
    let minsync_dir = root.path().join(".minsync");
    std::fs::create_dir_all(&minsync_dir).unwrap();
    Config::default_for("s")
        .save(&minsync_dir.join("config.toml"))
        .unwrap();

    let output = minsync(root.path())
        .args(["sync", "--progress-file", "p.jsonl"])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("add it to .minsyncignore"));
    assert!(!root.path().join("p.jsonl").exists());
}

#[cfg(unix)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn skipped_file_is_reported_as_file_failed_and_partial_done() {
    use std::os::unix::fs::PermissionsExt;

    let server = embed_server(Duration::ZERO).await;
    let root = workspace(&server, 3);
    let locked = root.path().join("locked.md");
    std::fs::write(&locked, "secret").unwrap();
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();
    if std::fs::read(&locked).is_ok() {
        return; // running as root: permissions do not apply
    }

    let output = minsync(root.path())
        .args(["sync", "--progress", "jsonl"])
        .output()
        .expect("run sync");
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o644)).unwrap();

    assert_eq!(output.status.code(), Some(6), "partial sync exit code");
    let lines = parse_lines(&String::from_utf8(output.stdout).unwrap());
    let failed: Vec<&Value> = lines
        .iter()
        .filter(|line| line["event"] == "file_failed")
        .collect();
    assert_eq!(failed.len(), 1);
    assert_eq!(failed[0]["path"], "locked.md");
    assert_eq!(failed[0]["stage"], "read");
    assert!(failed[0]["error"].as_str().unwrap().contains("Permission"));

    let done = lines.last().expect("events");
    assert_eq!(done["event"], "done");
    assert_eq!(done["files_failed"], 1);
    assert_eq!(done["partial"], true);
}
