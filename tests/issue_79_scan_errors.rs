//! Issue #79: one unreadable entry must not abort the scan, and entries the
//! scan could not inspect must never lose their existing chunks.

#![cfg(unix)]

use minsync::chunker::recursive::RecursiveChunker;
use minsync::embedder::Embedder;
use minsync::error::Result;
use minsync::manifest::ScanErrorKind;
use minsync::sync::MinSync;
use minsync::types::SyncResult;
use minsync::vectorstore::memory::InMemoryStore;
use minsync::vectorstore::VectorStore;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

struct ZeroEmbedder;

#[async_trait::async_trait]
impl Embedder for ZeroEmbedder {
    fn id(&self) -> &str {
        "mock-zero"
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![0.0; 8]).collect())
    }
}

/// Restores permissions on drop so the tempdir can always be cleaned up.
struct Locked(PathBuf);

impl Locked {
    fn new(path: PathBuf) -> Self {
        fs::set_permissions(&path, fs::Permissions::from_mode(0o000)).expect("lock path");
        Self(path)
    }

    fn unlock(&self) {
        let mode = if self.0.is_dir() { 0o755 } else { 0o644 };
        fs::set_permissions(&self.0, fs::Permissions::from_mode(mode)).expect("unlock path");
    }
}

impl Drop for Locked {
    fn drop(&mut self) {
        self.unlock();
    }
}

fn running_as_root() -> bool {
    // SAFETY: geteuid has no preconditions and cannot fail.
    unsafe { libc::geteuid() == 0 }
}

fn setup(root: &Path) -> MinSync {
    let sync = MinSync::new(root.to_path_buf());
    sync.init(false, "tei:test", "recursive")
        .expect("init workspace");
    sync
}

async fn run(sync: &MinSync, store: &mut InMemoryStore, full: bool) -> SyncResult {
    let chunker = RecursiveChunker::new(4096);
    sync.sync(&chunker, &ZeroEmbedder, store, full, false, false)
        .await
        .expect("sync must not abort on per-entry errors")
}

fn sorted_paths(store: &InMemoryStore) -> Vec<String> {
    let mut paths = store.all_paths();
    paths.sort();
    paths.dedup();
    paths
}

#[tokio::test]
async fn unreadable_directory_does_not_abort_and_keeps_its_chunks() {
    if running_as_root() {
        return;
    }
    let root = tempfile::tempdir().expect("tempdir");
    fs::create_dir(root.path().join("locked")).expect("mkdir");
    fs::write(root.path().join("locked/inner.txt"), "inner text").expect("write inner");
    fs::write(root.path().join("ok.txt"), "ok text").expect("write ok");
    let sync = setup(root.path());
    let mut store = InMemoryStore::new();

    let first = run(&sync, &mut store, false).await;
    assert!(first.scan_errors.is_empty());
    assert_eq!(sorted_paths(&store), vec!["locked/inner.txt", "ok.txt"]);

    let lock = Locked::new(root.path().join("locked"));
    fs::write(root.path().join("new.txt"), "new text").expect("write new");

    let second = run(&sync, &mut store, false).await;
    assert_eq!(second.scan_errors.len(), 1);
    assert_eq!(second.scan_errors[0].path, "locked");
    assert_eq!(second.scan_errors[0].kind, ScanErrorKind::Permanent);
    assert_eq!(second.files_added, 1, "readable new file is indexed");
    assert_eq!(second.files_deleted, 0, "unreadable subtree is not deleted");
    assert_eq!(
        sorted_paths(&store),
        vec!["locked/inner.txt", "new.txt", "ok.txt"]
    );

    // Nothing changed and the directory is still unreadable: the early
    // "up to date" return must still report the errors.
    let third = run(&sync, &mut store, false).await;
    assert_eq!(third.scan_errors.len(), 1);
    assert_eq!(third.files_deleted, 0);
    assert!(sorted_paths(&store).contains(&"locked/inner.txt".to_string()));

    // Fixing the permission converges without re-indexing the unchanged file.
    lock.unlock();
    let fourth = run(&sync, &mut store, false).await;
    assert!(fourth.scan_errors.is_empty());
    assert_eq!(fourth.files_processed, 0);
    assert!(sorted_paths(&store).contains(&"locked/inner.txt".to_string()));
}

#[tokio::test]
async fn full_sync_keeps_chunks_of_unreadable_directory() {
    if running_as_root() {
        return;
    }
    let root = tempfile::tempdir().expect("tempdir");
    fs::create_dir(root.path().join("locked")).expect("mkdir");
    fs::write(root.path().join("locked/inner.txt"), "inner text").expect("write inner");
    fs::write(root.path().join("ok.txt"), "ok text").expect("write ok");
    let sync = setup(root.path());
    let mut store = InMemoryStore::new();
    run(&sync, &mut store, false).await;

    let lock = Locked::new(root.path().join("locked"));
    let result = run(&sync, &mut store, true).await;

    assert_eq!(result.scan_errors.len(), 1);
    assert_eq!(
        sorted_paths(&store),
        vec!["locked/inner.txt", "ok.txt"],
        "full-sweep must not delete chunks of paths it could not inspect"
    );

    // The next scan sees the directory again and reconciles it.
    lock.unlock();
    let healed = run(&sync, &mut store, false).await;
    assert!(healed.scan_errors.is_empty());
    assert_eq!(sorted_paths(&store), vec!["locked/inner.txt", "ok.txt"]);
}

#[tokio::test]
async fn unreadable_file_is_recorded_and_keeps_its_chunks() {
    if running_as_root() {
        return;
    }
    let root = tempfile::tempdir().expect("tempdir");
    fs::write(root.path().join("secret.txt"), "secret text").expect("write secret");
    fs::write(root.path().join("ok.txt"), "ok text").expect("write ok");
    let sync = setup(root.path());
    let mut store = InMemoryStore::new();
    run(&sync, &mut store, false).await;

    let _lock = Locked::new(root.path().join("secret.txt"));
    fs::write(root.path().join("ok.txt"), "ok text, edited").expect("edit ok");
    let result = run(&sync, &mut store, false).await;

    assert_eq!(result.scan_errors.len(), 1);
    assert_eq!(result.scan_errors[0].path, "secret.txt");
    assert_eq!(result.scan_errors[0].kind, ScanErrorKind::Permanent);
    assert_eq!(result.files_modified, 1);
    assert_eq!(result.files_deleted, 0);
    assert_eq!(sorted_paths(&store), vec!["ok.txt", "secret.txt"]);
}

#[tokio::test]
async fn verify_fix_and_status_treat_unreadable_directory_as_unknown() {
    if running_as_root() {
        return;
    }
    let root = tempfile::tempdir().expect("tempdir");
    fs::create_dir(root.path().join("locked")).expect("mkdir");
    fs::write(root.path().join("locked/inner.txt"), "inner text").expect("write inner");
    fs::write(root.path().join("ok.txt"), "ok text").expect("write ok");
    let sync = setup(root.path());
    let mut store = InMemoryStore::new();
    run(&sync, &mut store, false).await;

    let _lock = Locked::new(root.path().join("locked"));
    let minsync_dir = root.path().join(".minsync");
    let chunker = RecursiveChunker::new(4096);

    let verified =
        minsync::verify::verify(&minsync_dir, root.path(), &chunker, &mut store, true, None)
            .await
            .expect("verify must not abort");
    assert!(
        !verified.fixed,
        "verify --fix must not delete unreadable chunks"
    );
    assert!(verified.basic_checks["no_stale_paths"]);
    assert_eq!(sorted_paths(&store), vec!["locked/inner.txt", "ok.txt"]);

    let status = minsync::verify::status(&minsync_dir).await.expect("status");
    assert_eq!(status.state, minsync::types::SyncState::UpToDate);
}

#[test]
fn sync_result_without_scan_errors_field_still_deserializes() {
    let json = serde_json::json!({
        "files_processed": 0, "chunks_added": 0, "chunks_updated": 0,
        "chunks_deleted": 0, "chunks_truncated": 0, "dry_run": false,
        "already_up_to_date": false, "initial_sync": false,
        "files_processed_paths": [], "files_added": 0, "files_modified": 0,
        "files_deleted": 0, "elapsed_seconds": 0.0, "embedding_api_calls": 0,
        "embedded_texts": 0, "estimated_tokens": 0, "files_checked": 0,
        "freshness_check_only": false, "query_ready": false
    });

    let result: SyncResult = serde_json::from_value(json).expect("old summaries deserialize");

    assert!(result.scan_errors.is_empty());
}
