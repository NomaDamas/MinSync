use minsync::chunker::chonkie::ChonkieChunker;
use minsync::embedder::Embedder;
use minsync::error::{MinSyncError, Result};
use minsync::query::query;
use minsync::state::Transaction;
use minsync::sync::MinSync;
use minsync::types::SyncState;
use minsync::vectorstore::memory::InMemoryStore;
use minsync::vectorstore::VectorStore;
use minsync::verify::{status, verify};
use tempfile::TempDir;

struct MockEmbedder;

#[async_trait::async_trait]
impl Embedder for MockEmbedder {
    fn id(&self) -> &str {
        "mock"
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|text| {
                let hash = minsync::id::content_hash(text);
                let bytes = hash.as_bytes();
                (0..8).map(|i| bytes[i] as f32 / 255.0).collect()
            })
            .collect())
    }
}

fn fixture() -> (TempDir, MinSync, ChonkieChunker, MockEmbedder, InMemoryStore) {
    let dir = tempfile::tempdir().expect("create tempdir");
    let sync = MinSync::new(dir.path().to_path_buf());
    let chunker = ChonkieChunker::new(32, "\n ");
    let embedder = MockEmbedder;
    let store = InMemoryStore::new();
    (dir, sync, chunker, embedder, store)
}

fn write_file(dir: &TempDir, path: &str, content: &str) {
    let path = dir.path().join(path);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create parent dir");
    }
    std::fs::write(path, content).expect("write file");
}

#[tokio::test]
async fn test_full_workflow() {
    let (dir, sync, chunker, embedder, mut store) = fixture();
    write_file(&dir, "README.md", "# MinSync\n\nalpha beta gamma");
    write_file(&dir, "src/lib.txt", "delta epsilon zeta");

    sync.init(false).expect("init succeeds");
    let sync_result = sync
        .sync(&chunker, &embedder, &mut store, true, false, false)
        .await
        .expect("sync succeeds");
    let query_results = query(
        &dir.path().join(".minsync"),
        "alpha beta",
        5,
        &embedder,
        &store,
        None,
    )
    .await
    .expect("query succeeds");
    let verify_result = verify(&dir.path().join(".minsync"), dir.path(), &chunker, &mut store, false, None)
        .await
        .expect("verify succeeds");

    assert_eq!(sync_result.files_processed, 2);
    assert!(sync_result.chunks_added > 0);
    assert!(!query_results.is_empty());
    assert_eq!(query_results[0].rank, 1);
    assert!(verify_result.all_passed);
}

#[test]
fn test_init_creates_structure() {
    let (dir, sync, _chunker, _embedder, _store) = fixture();

    let config = sync.init(false).expect("init succeeds");

    assert_eq!(config.version, 1);
    assert!(dir.path().join(".minsync/config.toml").exists());
    assert!(dir.path().join(".minsync/manifest.json").exists());
}

#[test]
fn test_init_already_exists() {
    let (_dir, sync, _chunker, _embedder, _store) = fixture();
    sync.init(false).expect("first init succeeds");

    let result = sync.init(false);

    assert!(matches!(result, Err(MinSyncError::AlreadyInitialized)));
}

#[test]
fn test_init_force() {
    let (_dir, sync, _chunker, _embedder, _store) = fixture();
    let first = sync.init(false).expect("first init succeeds");

    let second = sync.init(true).expect("force init succeeds");

    assert_ne!(first.source_id, second.source_id);
}

#[tokio::test]
async fn test_sync_full_indexes_all() {
    let (dir, sync, chunker, embedder, mut store) = fixture();
    write_file(&dir, "a.txt", "alpha beta gamma");
    write_file(&dir, "b.txt", "delta epsilon zeta");
    write_file(&dir, "nested/c.txt", "eta theta iota");
    sync.init(false).expect("init succeeds");

    let result = sync
        .sync(&chunker, &embedder, &mut store, true, false, false)
        .await
        .expect("full sync succeeds");

    assert_eq!(result.files_processed, 3);
    assert!(result.chunks_added > 0);
    assert_eq!(store.doc_count(), result.chunks_added);
    assert_eq!(store.all_paths(), vec!["a.txt", "b.txt", "nested/c.txt"]);
}

#[tokio::test]
async fn test_sync_incremental_detects_new_file() {
    let (dir, sync, chunker, embedder, mut store) = fixture();
    write_file(&dir, "a.txt", "alpha beta gamma");
    sync.init(false).expect("init succeeds");
    sync.sync(&chunker, &embedder, &mut store, true, false, false)
        .await
        .expect("initial sync succeeds");
    let initial_count = store.doc_count();

    write_file(&dir, "b.txt", "delta epsilon zeta");
    let result = sync
        .sync(&chunker, &embedder, &mut store, false, false, false)
        .await
        .expect("incremental sync succeeds");

    assert_eq!(result.files_processed_paths, vec!["b.txt"]);
    assert!(result.chunks_added > 0);
    assert_eq!(store.doc_count(), initial_count + result.chunks_added);
    assert_eq!(store.all_paths(), vec!["a.txt", "b.txt"]);
}

#[tokio::test]
async fn test_sync_incremental_detects_modification() {
    let (dir, sync, chunker, embedder, mut store) = fixture();
    write_file(&dir, "a.txt", "alpha beta gamma");
    sync.init(false).expect("init succeeds");
    sync.sync(&chunker, &embedder, &mut store, true, false, false)
        .await
        .expect("initial sync succeeds");

    write_file(&dir, "a.txt", "alpha beta gamma delta epsilon zeta");
    let result = sync
        .sync(&chunker, &embedder, &mut store, false, false, false)
        .await
        .expect("modify sync succeeds");

    assert_eq!(result.files_processed_paths, vec!["a.txt"]);
    assert_eq!(store.all_paths(), vec!["a.txt"]);
    assert!(result.chunks_added + result.chunks_updated > 0);
}

#[tokio::test]
async fn test_sync_incremental_detects_deletion() {
    let (dir, sync, chunker, embedder, mut store) = fixture();
    write_file(&dir, "a.txt", "alpha beta gamma");
    sync.init(false).expect("init succeeds");
    sync.sync(&chunker, &embedder, &mut store, true, false, false)
        .await
        .expect("initial sync succeeds");
    let initial_count = store.doc_count();
    assert!(initial_count > 0);

    std::fs::remove_file(dir.path().join("a.txt")).expect("delete file");
    let result = sync
        .sync(&chunker, &embedder, &mut store, false, false, false)
        .await
        .expect("delete sync succeeds");

    assert_eq!(result.files_processed_paths, vec!["a.txt"]);
    assert_eq!(result.chunks_deleted, initial_count);
    assert_eq!(store.doc_count(), 0);
}

#[tokio::test]
async fn test_sync_dry_run_no_side_effects() {
    let (dir, sync, chunker, embedder, mut store) = fixture();
    write_file(&dir, "a.txt", "alpha beta gamma");
    sync.init(false).expect("init succeeds");

    let result = sync
        .sync(&chunker, &embedder, &mut store, true, true, false)
        .await
        .expect("dry run succeeds");

    assert!(result.dry_run);
    assert_eq!(result.files_processed_paths, vec!["a.txt"]);
    assert_eq!(store.doc_count(), 0);
    assert!(!dir.path().join(".minsync/cursor.json").exists());
    assert!(!dir.path().join(".minsync/txn.json").exists());
}

#[tokio::test]
async fn test_sync_already_up_to_date() {
    let (dir, sync, chunker, embedder, mut store) = fixture();
    write_file(&dir, "a.txt", "alpha beta gamma");
    sync.init(false).expect("init succeeds");
    sync.sync(&chunker, &embedder, &mut store, true, false, false)
        .await
        .expect("initial sync succeeds");

    let result = sync
        .sync(&chunker, &embedder, &mut store, false, false, false)
        .await
        .expect("second sync succeeds");

    assert!(result.already_up_to_date);
    assert_eq!(result.files_processed, 0);
    assert_eq!(result.chunks_added, 0);
}

#[tokio::test]
async fn test_sync_crash_recovery_removes_stale_transaction() {
    let (dir, sync, chunker, embedder, mut store) = fixture();
    write_file(&dir, "a.txt", "alpha beta gamma");
    let config = sync.init(false).expect("init succeeds");
    Transaction::new(&config.source_id, "stale-token", None, "sha256:stale")
        .save(&dir.path().join(".minsync/txn.json"))
        .expect("save stale transaction");

    let result = sync
        .sync(&chunker, &embedder, &mut store, true, false, false)
        .await
        .expect("sync recovers");

    assert!(result.chunks_added > 0);
    assert!(!dir.path().join(".minsync/txn.json").exists());
    assert!(dir.path().join(".minsync/cursor.json").exists());
}

#[tokio::test]
async fn test_status_states() {
    let (dir, sync, chunker, embedder, mut store) = fixture();
    write_file(&dir, "a.txt", "alpha beta gamma");
    let config = sync.init(false).expect("init succeeds");

    let not_synced = status(&dir.path().join(".minsync"))
        .await
        .expect("status before sync succeeds");
    assert_eq!(not_synced.state, SyncState::NotSynced);

    sync.sync(&chunker, &embedder, &mut store, true, false, false)
        .await
        .expect("sync succeeds");
    let up_to_date = status(&dir.path().join(".minsync"))
        .await
        .expect("status after sync succeeds");
    assert_eq!(up_to_date.state, SyncState::UpToDate);

    write_file(&dir, "a.txt", "changed content");
    let out_of_date = status(&dir.path().join(".minsync"))
        .await
        .expect("status after modification succeeds");
    assert_eq!(out_of_date.state, SyncState::OutOfDate);

    Transaction::new(&config.source_id, "token", None, "sha256:next")
        .save(&dir.path().join(".minsync/txn.json"))
        .expect("save transaction");
    let interrupted = status(&dir.path().join(".minsync"))
        .await
        .expect("status interrupted succeeds");
    assert_eq!(interrupted.state, SyncState::Interrupted);
}

#[tokio::test]
async fn test_minsyncignore_filtering() {
    let (dir, sync, chunker, embedder, mut store) = fixture();
    write_file(&dir, ".minsyncignore", "ignored.txt\nignored_dir/\n");
    write_file(&dir, "kept.txt", "kept content");
    write_file(&dir, "ignored.txt", "ignored content");
    write_file(&dir, "ignored_dir/file.txt", "ignored nested content");
    sync.init(false).expect("init succeeds");

    let result = sync
        .sync(&chunker, &embedder, &mut store, true, false, false)
        .await
        .expect("sync succeeds");

    assert_eq!(result.files_processed_paths, vec![".minsyncignore", "kept.txt"]);
    assert_eq!(store.all_paths(), vec![".minsyncignore", "kept.txt"]);
}
