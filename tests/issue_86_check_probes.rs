//! CLI-level contract for issue #86: `minsync check` reports an interrupted
//! run, and fails the vector store check when the store is not writable or
//! its volume is short of space, instead of always printing `VectorStore: OK`.

use assert_cmd::Command;
use minsync::config::Config;
use serde_json::Value;
use std::path::Path;
use std::process::Output;

/// `MinSyncError::VectorStore`, the code `sync` uses for a failed store.
const EXIT_VECTORSTORE: i32 = 4;

fn minsync(root: &Path) -> Command {
    let mut command = Command::cargo_bin("minsync").expect("find minsync binary");
    command.current_dir(root);
    command
}

/// A BM25-only workspace (no embedder to reach) that has synced once, so the
/// store directory exists. The disk-space check is off unless a test sets it.
fn setup(root: &Path) {
    std::fs::write(root.join("a.md"), "alpha beta gamma\n").unwrap();
    minsync(root)
        .args(["init", "--embedder", "none"])
        .assert()
        .success();
    set_min_free_bytes(root, 0);
    minsync(root).arg("sync").assert().success();
}

fn set_min_free_bytes(root: &Path, bytes: u64) {
    let config_path = root.join(".minsync/config.toml");
    let mut config = Config::load(&config_path).expect("load config");
    config.check.min_free_bytes = bytes;
    config.save(&config_path).expect("save config");
}

fn check_json(root: &Path) -> (Output, Value) {
    let output = minsync(root)
        .args(["--format", "json", "check"])
        .output()
        .expect("run minsync check");
    let result = serde_json::from_slice(&output.stdout).expect("json report on stdout");
    (output, result)
}

#[test]
fn a_healthy_workspace_reports_free_space_and_no_interruption() {
    let dir = tempfile::tempdir().unwrap();
    setup(dir.path());

    let (output, result) = check_json(dir.path());

    assert_eq!(output.status.code(), Some(0), "{output:?}");
    assert_eq!(result["all_passed"], true);
    assert_eq!(result["interrupted"], false);
    assert_eq!(result["resumable"], false);
    assert!(result["free_bytes"].as_u64().is_some(), "{result}");
}

#[test]
fn an_interrupted_sync_is_reported_without_failing_the_check() {
    let dir = tempfile::tempdir().unwrap();
    setup(dir.path());
    std::fs::write(
        dir.path().join(".minsync/txn.json"),
        r#"{"source_id":"s","sync_token":"t","manifest_hash_from":null,"manifest_hash_to":"h","status":"running","started_at":"2026-10-28T00:00:00Z"}"#,
    )
    .unwrap();

    let (output, result) = check_json(dir.path());

    assert_eq!(output.status.code(), Some(0), "{output:?}");
    assert_eq!(result["interrupted"], true);
    assert_eq!(result["all_passed"], true);
    assert_eq!(result["warnings"].as_array().unwrap().len(), 1, "{result}");

    let text = minsync(dir.path()).arg("check").output().unwrap();
    let stdout = String::from_utf8_lossy(&text.stdout);
    assert!(
        stdout.contains("Last sync:   interrupted"),
        "stdout: {stdout}"
    );
}

#[test]
fn too_little_free_space_fails_with_the_vectorstore_code() {
    let dir = tempfile::tempdir().unwrap();
    setup(dir.path());
    // TOML integers are i64, so this is the largest value a config can hold.
    set_min_free_bytes(dir.path(), i64::MAX as u64);

    let (output, result) = check_json(dir.path());

    assert_eq!(output.status.code(), Some(EXIT_VECTORSTORE), "{output:?}");
    assert_eq!(result["vectorstore_ok"], false);
    assert_eq!(result["all_passed"], false);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("min_free_bytes"), "stderr: {stderr}");
}

#[cfg(unix)]
#[test]
fn a_read_only_store_fails_with_the_vectorstore_code() {
    use std::os::unix::fs::PermissionsExt;

    let dir = tempfile::tempdir().unwrap();
    setup(dir.path());
    let store = dir.path().join(".minsync/store");
    let writable = std::fs::metadata(&store).unwrap().permissions();
    std::fs::set_permissions(&store, std::fs::Permissions::from_mode(0o555)).unwrap();
    // Root ignores the mode bits; there is nothing to test then.
    let can_still_write = std::fs::write(store.join(".probe"), b"").is_ok();

    let (output, result) = check_json(dir.path());
    std::fs::set_permissions(&store, writable).unwrap();
    if can_still_write {
        return;
    }

    assert_eq!(output.status.code(), Some(EXIT_VECTORSTORE), "{output:?}");
    assert_eq!(result["vectorstore_ok"], false);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("not writable"), "stderr: {stderr}");
}
