//! Issue #87: a sync that finds the workspace lock held reports who holds it,
//! and `--wait-timeout` bounds `--wait`.

use assert_cmd::Command;
use minsync::config::Config;
use minsync::state::FileLock;
use std::time::{Duration, Instant};

/// An initialized workspace whose lock is held by this test process, as a
/// running `minsync sync` would hold it.
fn locked_workspace() -> (tempfile::TempDir, FileLock) {
    let root = tempfile::tempdir().expect("create workspace");
    let minsync_dir = root.path().join(".minsync");
    std::fs::create_dir_all(&minsync_dir).expect("create state directory");
    Config::default_for("source-issue-87")
        .save(&minsync_dir.join("config.toml"))
        .expect("save config");
    let lock = FileLock::acquire(&minsync_dir.join("lock"), false).expect("acquire lock");
    (root, lock)
}

#[test]
fn sync_reports_the_lock_holder() {
    let (root, _lock) = locked_workspace();
    let output = Command::cargo_bin("minsync")
        .expect("minsync binary")
        .current_dir(root.path())
        .args(["sync"])
        .output()
        .expect("run minsync sync");

    assert_eq!(output.status.code(), Some(3));
    let stderr = String::from_utf8_lossy(&output.stderr);
    let pid = format!("pid {}", std::process::id());
    assert!(stderr.contains("another sync is in progress"), "{stderr}");
    assert!(stderr.contains(&pid), "{stderr}");
    assert!(stderr.contains("started "), "{stderr}");
}

#[test]
fn sync_json_error_includes_the_lock_holder() {
    let (root, _lock) = locked_workspace();
    let output = Command::cargo_bin("minsync")
        .expect("minsync binary")
        .current_dir(root.path())
        .args(["--format", "json", "sync"])
        .output()
        .expect("run minsync sync");

    assert_eq!(output.status.code(), Some(3));
    let stdout = String::from_utf8_lossy(&output.stdout);
    let value: serde_json::Value =
        serde_json::from_str(stdout.trim()).unwrap_or_else(|e| panic!("{e}: {stdout}"));
    assert_eq!(value["error"]["code"], "lock_failed");
    assert_eq!(value["error"]["holder"]["pid"], std::process::id());
    assert!(value["error"]["holder"]["started_at"].is_string());
    assert!(value["error"]["holder"]["command"].is_string());
    assert!(value["error"]["message"]
        .as_str()
        .unwrap()
        .starts_with("another sync is in progress"));
}

#[test]
fn sync_wait_timeout_gives_up_with_lock_failed() {
    let (root, _lock) = locked_workspace();
    let started = Instant::now();
    let output = Command::cargo_bin("minsync")
        .expect("minsync binary")
        .current_dir(root.path())
        .args(["sync", "--wait", "--wait-timeout", "1"])
        .timeout(Duration::from_secs(30))
        .output()
        .expect("run minsync sync");

    let elapsed = started.elapsed();
    assert_eq!(output.status.code(), Some(3));
    assert!(
        elapsed >= Duration::from_millis(900),
        "gave up after {elapsed:?}"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains(&format!("pid {}", std::process::id())),
        "{stderr}"
    );
}

#[test]
fn bare_pid_lock_file_still_reports_the_pid() {
    // A lock file written by minsync 0.5 and older holds only the pid.
    let holder = minsync::state::parse_lock_holder("12345\n").expect("parse bare pid");
    assert_eq!(holder.pid, Some(12345));
    assert_eq!(holder.started_at, None);
    assert_eq!(holder.command, None);
}
