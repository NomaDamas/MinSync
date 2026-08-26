use minsync::chunker::chonkie::ChonkieChunker;
use minsync::chunker::create_chunker;
use minsync::chunker::recursive::RecursiveChunker;
use minsync::cli::QueryMode;
use minsync::config::Config;
use minsync::embedder::tei::TeiEmbedder;
use minsync::embedder::Embedder;
use minsync::error::{MinSyncError, Result};
use minsync::query::query;
use minsync::state::Transaction;
use minsync::sync::MinSync;
use minsync::types::SyncState;
use minsync::vectorstore::lancedb_store::LanceDbStore;
use minsync::vectorstore::memory::InMemoryStore;
use minsync::vectorstore::{create_vectorstore, VectorStore};
use minsync::verify::{status, verify};
use minsync::watch::{
    run_with_shutdown, should_index, WatchControl, WatchStartup, WatchStartupStatus,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use tempfile::TempDir;
use tokio::sync::{mpsc, oneshot};
use tokio::time::{timeout, Duration};

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

fn fixture() -> (
    TempDir,
    MinSync,
    ChonkieChunker,
    MockEmbedder,
    InMemoryStore,
) {
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

async fn next_watch_result(
    receiver: &mut mpsc::UnboundedReceiver<minsync::types::SyncResult>,
) -> minsync::types::SyncResult {
    next_watch_result_where(receiver, |_| true).await
}

async fn next_watch_result_where(
    receiver: &mut mpsc::UnboundedReceiver<minsync::types::SyncResult>,
    mut predicate: impl FnMut(&minsync::types::SyncResult) -> bool,
) -> minsync::types::SyncResult {
    timeout(Duration::from_secs(10), async {
        loop {
            let result = receiver
                .recv()
                .await
                .expect("watch progress channel remains open");
            if predicate(&result) {
                return result;
            }
        }
    })
    .await
    .expect("watch event within timeout")
}

#[tokio::test]
async fn test_watch_real_filesystem_add_modify_delete() {
    let (dir, sync, chunker, embedder, mut store) = fixture();
    sync.init(false, "tei:test", "recursive")
        .expect("init succeeds");
    let root = dir.path().to_path_buf();
    let (progress_tx, mut progress_rx) = mpsc::unbounded_channel();
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let watch_task = tokio::spawn(async move {
        run_with_shutdown(
            root,
            &chunker,
            &embedder,
            &mut store,
            Some(25),
            WatchControl {
                startup: WatchStartup::FailFast,
                progress: Some(progress_tx),
                startup_status: None,
                shutdown: shutdown_rx,
            },
        )
        .await
    });

    let initial = next_watch_result(&mut progress_rx).await;
    assert!(initial.initial_sync);

    write_file(&dir, "watch.md", "# first");
    let added = next_watch_result_where(&mut progress_rx, |result| result.files_added == 1).await;
    assert_eq!(added.files_added, 1);
    assert_eq!(added.files_modified, 0);
    assert_eq!(added.files_deleted, 0);

    write_file(&dir, "watch.md", "# second");
    let modified =
        next_watch_result_where(&mut progress_rx, |result| result.files_modified == 1).await;
    assert_eq!(modified.files_modified, 1);

    std::fs::remove_file(dir.path().join("watch.md")).expect("delete watched file");
    let deleted =
        next_watch_result_where(&mut progress_rx, |result| result.files_deleted == 1).await;
    assert_eq!(deleted.files_deleted, 1);
    assert_eq!(deleted.chunks_deleted, 1);

    shutdown_tx.send(()).expect("signal watcher shutdown");
    watch_task
        .await
        .expect("watch task joins")
        .expect("watch exits");
}

#[tokio::test]
async fn test_watch_resilient_startup_reports_failure_and_retries() {
    let (dir, sync, chunker, _embedder, mut store) = fixture();
    write_file(&dir, "pending.md", "requires embedding");
    sync.init(false, "tei:test", "recursive")
        .expect("init succeeds");
    let root = dir.path().to_path_buf();
    let flaky_embedder = FlakyEmbedder {
        calls: Arc::new(AtomicUsize::new(0)),
    };
    let (status_tx, mut status_rx) = mpsc::unbounded_channel();
    let (progress_tx, mut progress_rx) = mpsc::unbounded_channel();
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let watch_task = tokio::spawn(async move {
        run_with_shutdown(
            root,
            &chunker,
            &flaky_embedder,
            &mut store,
            Some(25),
            WatchControl {
                startup: WatchStartup::ContinueOnSyncError,
                progress: Some(progress_tx),
                startup_status: Some(status_tx),
                shutdown: shutdown_rx,
            },
        )
        .await
    });

    let status = timeout(Duration::from_secs(10), status_rx.recv())
        .await
        .expect("startup status within timeout")
        .expect("startup status remains open");
    assert!(matches!(status, WatchStartupStatus::InitialSyncFailed(_)));

    write_file(&dir, "pending.md", "retry succeeds");
    let recovered = next_watch_result(&mut progress_rx).await;
    assert_eq!(recovered.files_added, 1);
    assert!(dir.path().join(".minsync/cursor.json").exists());

    shutdown_tx.send(()).expect("signal watcher shutdown");
    watch_task
        .await
        .expect("watch task joins")
        .expect("watch exits");
}

struct FlakyEmbedder {
    calls: Arc<AtomicUsize>,
}

#[async_trait::async_trait]
impl Embedder for FlakyEmbedder {
    fn id(&self) -> &str {
        "flaky"
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
            Err(MinSyncError::Embedding("backend unavailable".to_string()))
        } else {
            Ok(texts.iter().map(|_| vec![0.0; 8]).collect())
        }
    }
}

#[tokio::test]
async fn test_full_workflow() {
    let (dir, sync, chunker, embedder, mut store) = fixture();
    write_file(&dir, "README.md", "# MinSync\n\nalpha beta gamma");
    write_file(&dir, "src/lib.txt", "delta epsilon zeta");

    sync.init(false, "openai:text-embedding-3-small", "recursive")
        .expect("init succeeds");
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
        QueryMode::Vector,
    )
    .await
    .expect("query succeeds");
    let verify_result = verify(
        &dir.path().join(".minsync"),
        dir.path(),
        &chunker,
        &mut store,
        false,
        None,
    )
    .await
    .expect("verify succeeds");

    assert_eq!(sync_result.files_processed, 2);
    assert!(sync_result.chunks_added > 0);
    assert!(!query_results.is_empty());
    assert_eq!(query_results[0].rank, 1);
    assert!(verify_result.all_passed);
}

#[tokio::test]
async fn test_utf8_multilingual_text_across_file_extensions() {
    let (dir, sync, chunker, embedder, mut store) = fixture();
    write_file(&dir, "docs/japanese.md", "日本語のメモを検索できます。");
    write_file(&dir, "notes/chinese.txt", "中文内容会作为 UTF-8 文本处理。");
    write_file(
        &dir,
        "src/korean.note",
        "한국어 파일도 확장자와 무관하게 처리됩니다.",
    );

    sync.init(false, "openai:text-embedding-3-small", "recursive")
        .expect("init succeeds");
    sync.sync(&chunker, &embedder, &mut store, true, false, false)
        .await
        .expect("sync succeeds");

    assert_eq!(
        store.all_paths(),
        vec!["docs/japanese.md", "notes/chinese.txt", "src/korean.note"]
    );

    let hits = store.query(&[1.0; 8], None, 10).expect("query docs");
    let indexed_text = hits
        .iter()
        .map(|hit| hit.text.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(indexed_text.contains("日本語"));
    assert!(indexed_text.contains("中文内容"));
    assert!(indexed_text.contains("한국어"));
}

#[test]
fn test_init_creates_structure() {
    let (dir, sync, _chunker, _embedder, _store) = fixture();

    let config = sync
        .init(false, "openai:text-embedding-3-small", "recursive")
        .expect("init succeeds");

    assert_eq!(config.version, 1);
    assert!(dir.path().join(".minsync/config.toml").exists());
    assert!(dir.path().join(".minsync/manifest.json").exists());
}

#[test]
fn test_init_already_exists() {
    let (_dir, sync, _chunker, _embedder, _store) = fixture();
    sync.init(false, "openai:text-embedding-3-small", "recursive")
        .expect("first init succeeds");

    let result = sync.init(false, "openai:text-embedding-3-small", "recursive");

    assert!(matches!(result, Err(MinSyncError::AlreadyInitialized)));
}

#[test]
fn test_init_force() {
    let (_dir, sync, _chunker, _embedder, _store) = fixture();
    let first = sync
        .init(false, "openai:text-embedding-3-small", "recursive")
        .expect("first init succeeds");

    let second = sync
        .init(true, "openai:text-embedding-3-small", "recursive")
        .expect("force init succeeds");

    assert_ne!(first.source_id, second.source_id);
}

#[tokio::test]
async fn test_sync_full_indexes_all() {
    let (dir, sync, chunker, embedder, mut store) = fixture();
    write_file(&dir, "a.txt", "alpha beta gamma");
    write_file(&dir, "b.txt", "delta epsilon zeta");
    write_file(&dir, "nested/c.txt", "eta theta iota");
    sync.init(false, "openai:text-embedding-3-small", "recursive")
        .expect("init succeeds");

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
    sync.init(false, "openai:text-embedding-3-small", "recursive")
        .expect("init succeeds");
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
    assert_eq!(result.files_added, 1);
    assert_eq!(result.files_modified, 0);
    assert_eq!(result.files_deleted, 0);
}

#[tokio::test]
async fn test_sync_incremental_detects_modification() {
    let (dir, sync, chunker, embedder, mut store) = fixture();
    write_file(&dir, "a.txt", "alpha beta gamma");
    sync.init(false, "openai:text-embedding-3-small", "recursive")
        .expect("init succeeds");
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
    assert_eq!(result.files_added, 0);
    assert_eq!(result.files_modified, 1);
    assert_eq!(result.files_deleted, 0);
}

#[tokio::test]
async fn test_sync_incremental_detects_deletion() {
    let (dir, sync, chunker, embedder, mut store) = fixture();
    write_file(&dir, "a.txt", "alpha beta gamma");
    sync.init(false, "openai:text-embedding-3-small", "recursive")
        .expect("init succeeds");
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
    assert_eq!(result.files_added, 0);
    assert_eq!(result.files_modified, 0);
    assert_eq!(result.files_deleted, 1);
}

#[tokio::test]
async fn test_sync_dry_run_no_side_effects() {
    let (dir, sync, chunker, embedder, mut store) = fixture();
    write_file(&dir, "a.txt", "alpha beta gamma");
    sync.init(false, "openai:text-embedding-3-small", "recursive")
        .expect("init succeeds");

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
async fn test_init_then_plain_sync_is_queryable() {
    // Regression for issue #31: init baselines the manifest, so a plain sync
    // on a cursor-less workspace must perform the initial full sync instead
    // of reporting already_up_to_date and leaving the workspace unqueryable.
    let (dir, sync, chunker, embedder, mut store) = fixture();
    write_file(&dir, "doc.txt", "alpha beta gamma");
    sync.init(false, "openai:text-embedding-3-small", "recursive")
        .expect("init succeeds");

    let result = sync
        .sync(&chunker, &embedder, &mut store, false, false, false)
        .await
        .expect("plain sync succeeds");

    assert!(
        !result.already_up_to_date,
        "sync must never report already_up_to_date when no cursor exists"
    );
    assert!(
        result.initial_sync,
        "sync on a cursor-less workspace must report initial_sync"
    );
    assert_eq!(result.files_processed, 1);
    assert!(result.chunks_added > 0);
    assert!(
        dir.path().join(".minsync/cursor.json").exists(),
        "plain sync on a cursor-less workspace must create cursor.json"
    );

    let query_results = query(
        &dir.path().join(".minsync"),
        "alpha beta",
        3,
        &embedder,
        &store,
        None,
        QueryMode::Vector,
    )
    .await
    .expect("query immediately after plain sync succeeds");
    assert!(!query_results.is_empty());
}

#[tokio::test]
async fn test_sync_after_initial_sync_is_incremental_not_initial() {
    let (dir, sync, chunker, embedder, mut store) = fixture();
    write_file(&dir, "a.txt", "alpha beta gamma");
    sync.init(false, "openai:text-embedding-3-small", "recursive")
        .expect("init succeeds");
    let first = sync
        .sync(&chunker, &embedder, &mut store, false, false, false)
        .await
        .expect("initial sync succeeds");
    assert!(first.initial_sync);

    let second = sync
        .sync(&chunker, &embedder, &mut store, false, false, false)
        .await
        .expect("second sync succeeds");

    assert!(second.already_up_to_date);
    assert!(!second.initial_sync);
}

#[tokio::test]
async fn test_sync_already_up_to_date() {
    let (dir, sync, chunker, embedder, mut store) = fixture();
    write_file(&dir, "a.txt", "alpha beta gamma");
    sync.init(false, "openai:text-embedding-3-small", "recursive")
        .expect("init succeeds");
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
async fn test_sync_reports_embedding_stats() {
    let (dir, sync, chunker, embedder, mut store) = fixture();
    write_file(&dir, "a.txt", "alpha beta gamma");
    write_file(&dir, "b.txt", "delta epsilon zeta");
    sync.init(false, "openai:text-embedding-3-small", "recursive")
        .expect("init succeeds");

    let result = sync
        .sync(&chunker, &embedder, &mut store, true, false, false)
        .await
        .expect("full sync succeeds");

    assert!(result.embedding_api_calls > 0);
    assert_eq!(result.embedded_texts, result.chunks_added);
    assert!(result.estimated_tokens > 0);
    assert!(result.elapsed_seconds >= 0.0);
    assert_eq!(result.files_added, 2);
    assert_eq!(result.files_modified, 0);
    assert_eq!(result.files_deleted, 0);
}

#[tokio::test]
async fn test_sync_stats_zeroed_when_up_to_date() {
    let (dir, sync, chunker, embedder, mut store) = fixture();
    write_file(&dir, "a.txt", "alpha beta gamma");
    sync.init(false, "openai:text-embedding-3-small", "recursive")
        .expect("init succeeds");
    sync.sync(&chunker, &embedder, &mut store, true, false, false)
        .await
        .expect("initial sync succeeds");

    let result = sync
        .sync(&chunker, &embedder, &mut store, false, false, false)
        .await
        .expect("second sync succeeds");

    assert!(result.already_up_to_date);
    assert_eq!(result.embedding_api_calls, 0);
    assert_eq!(result.embedded_texts, 0);
    assert_eq!(result.estimated_tokens, 0);
    assert_eq!(result.files_added, 0);
    assert_eq!(result.files_modified, 0);
    assert_eq!(result.files_deleted, 0);
}

#[tokio::test]
async fn test_cdc_chunker_limits_reembedding_after_top_insertion() {
    let dir = tempfile::tempdir().expect("create tempdir");
    let sync = MinSync::new(dir.path().to_path_buf());
    let chunker = minsync::chunker::cdc::CdcChunker::new(32, 64, 128);
    let embedder = MockEmbedder;
    let mut store = InMemoryStore::new();

    let original: String = (0..200)
        .map(|i| format!("{i} {}\n", minsync::id::content_hash(&i.to_string())))
        .collect();
    write_file(&dir, "big.txt", &original);
    sync.init(false, "openai:text-embedding-3-small", "cdc")
        .expect("init succeeds");

    let first = sync
        .sync(&chunker, &embedder, &mut store, true, false, false)
        .await
        .expect("first sync succeeds");
    assert!(
        first.embedded_texts > 20,
        "expected many chunks on first sync, got {}",
        first.embedded_texts
    );

    let edited = {
        let mut lines: Vec<&str> = original.lines().collect();
        lines.insert(1, "INSERTED LINE near the top");
        format!("{}\n", lines.join("\n"))
    };
    write_file(&dir, "big.txt", &edited);

    let second = sync
        .sync(&chunker, &embedder, &mut store, false, false, false)
        .await
        .expect("second sync succeeds");

    assert_eq!(second.files_processed_paths, vec!["big.txt"]);
    assert!(
        second.chunks_updated > 0,
        "downstream chunks should be skipped (metadata-only update)"
    );
    assert!(
        second.embedded_texts * 5 <= first.embedded_texts,
        "a top insertion should re-embed only a small fraction: first={} second={}",
        first.embedded_texts,
        second.embedded_texts
    );
}

#[tokio::test]
async fn test_sync_crash_recovery_removes_stale_transaction() {
    let (dir, sync, chunker, embedder, mut store) = fixture();
    write_file(&dir, "a.txt", "alpha beta gamma");
    let config = sync
        .init(false, "openai:text-embedding-3-small", "recursive")
        .expect("init succeeds");
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
    let config = sync
        .init(false, "openai:text-embedding-3-small", "recursive")
        .expect("init succeeds");

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
    sync.init(false, "openai:text-embedding-3-small", "recursive")
        .expect("init succeeds");

    let result = sync
        .sync(&chunker, &embedder, &mut store, true, false, false)
        .await
        .expect("sync succeeds");

    assert_eq!(
        result.files_processed_paths,
        vec![".minsyncignore", "kept.txt"]
    );
    assert_eq!(store.all_paths(), vec![".minsyncignore", "kept.txt"]);
}

#[tokio::test]
async fn test_recursive_chunker_end_to_end() {
    let dir = tempfile::tempdir().expect("create tempdir");
    let sync = MinSync::new(dir.path().to_path_buf());
    let chunker = RecursiveChunker::new(64);
    let embedder = MockEmbedder;
    let mut store = InMemoryStore::new();

    write_file(
        &dir,
        "README.md",
        "# Title\n\npara one about apples\n\npara two about oranges",
    );

    sync.init(false, "openai:text-embedding-3-small", "recursive")
        .expect("init succeeds");
    let result = sync
        .sync(&chunker, &embedder, &mut store, true, false, false)
        .await
        .expect("recursive sync succeeds");

    assert!(result.chunks_added > 0);
    assert!(store.doc_count() > 0);

    let query_results = query(
        &dir.path().join(".minsync"),
        "apples",
        5,
        &embedder,
        &store,
        None,
        QueryMode::Vector,
    )
    .await
    .expect("query succeeds");

    assert!(!query_results.is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_lancedb_backend_end_to_end() {
    let dir = tempfile::tempdir().expect("create tempdir");
    let store_dir = tempfile::tempdir().expect("create store tempdir");
    let sync = MinSync::new(dir.path().to_path_buf());
    let chunker = RecursiveChunker::new(64);
    let embedder = MockEmbedder;
    // MockEmbedder produces 8-dim vectors, so LanceDB dimension MUST be 8.
    let mut store =
        LanceDbStore::open_or_create(store_dir.path(), 8).expect("create lancedb store");

    write_file(
        &dir,
        "README.md",
        "# Title\n\npara one about apples\n\npara two about oranges",
    );
    write_file(&dir, "notes.txt", "delta epsilon zeta extra content here");

    sync.init(false, "openai:text-embedding-3-small", "recursive")
        .expect("init succeeds");
    let result = sync
        .sync(&chunker, &embedder, &mut store, true, false, false)
        .await
        .expect("lancedb full sync succeeds");

    assert!(result.chunks_added > 0);
    assert!(store.doc_count() > 0);

    let bm25_hits = store
        .query_text("apples", None, 5)
        .expect("lancedb BM25 query succeeds");
    assert!(
        bm25_hits.iter().any(|hit| hit.text.contains("apples")),
        "BM25 must retrieve the shared normalized chunk text"
    );

    let query_results = query(
        &dir.path().join(".minsync"),
        "apples",
        5,
        &embedder,
        &store,
        None,
        QueryMode::Vector,
    )
    .await
    .expect("lancedb query succeeds");
    assert!(!query_results.is_empty());

    // Incremental sync after modifying a file proves the worker-thread bridge
    // works through the full async pipeline without a nested-runtime panic.
    write_file(
        &dir,
        "notes.txt",
        "delta epsilon zeta extra content here plus brand new words",
    );
    let incremental = sync
        .sync(&chunker, &embedder, &mut store, false, false, false)
        .await
        .expect("lancedb incremental sync succeeds");

    assert_eq!(incremental.files_processed_paths, vec!["notes.txt"]);
    assert!(incremental.chunks_added + incremental.chunks_updated > 0);
    assert!(store.doc_count() > 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_lancedb_bm25_incremental_delete_and_korean_e2e() {
    let dir = tempfile::tempdir().expect("create tempdir");
    let store_dir = tempfile::tempdir().expect("create store tempdir");
    let sync = MinSync::new(dir.path().to_path_buf());
    let chunker = RecursiveChunker::new(128);
    let embedder = MockEmbedder;
    let mut store =
        LanceDbStore::open_or_create(store_dir.path(), 8).expect("create lancedb store");

    write_file(
        &dir,
        "korean.md",
        "# 환불 정책\n\n한국어 환불 정책과 배송 안내입니다.",
    );
    write_file(&dir, "other.md", "unrelated lexical content");
    sync.init(false, "openai:text-embedding-3-small", "recursive")
        .expect("init succeeds");
    sync.sync(&chunker, &embedder, &mut store, true, false, false)
        .await
        .expect("initial live sync succeeds");

    let first = store
        .query_text("환불 정책", None, 5)
        .expect("Korean BM25 query succeeds");
    assert_eq!(first.len(), 1);
    let stable_id = first[0].doc_id.clone();

    write_file(
        &dir,
        "korean.md",
        "# 환불 정책\n\n한국어 환불 정책과 배송 안내입니다. 교환 조건도 확인하세요.",
    );
    sync.sync(&chunker, &embedder, &mut store, false, false, false)
        .await
        .expect("incremental live sync succeeds");
    let modified = store
        .query_text("교환 조건", None, 5)
        .expect("updated BM25 query succeeds");
    assert!(
        modified.iter().any(|hit| hit.text.contains("교환 조건")),
        "modified shared chunk must be searchable"
    );

    std::fs::remove_file(dir.path().join("korean.md")).expect("delete source");
    sync.sync(&chunker, &embedder, &mut store, false, false, false)
        .await
        .expect("delete live sync succeeds");
    let deleted = store
        .query_text("환불 정책", None, 5)
        .expect("post-delete BM25 query succeeds");
    assert!(
        deleted.iter().all(|hit| hit.doc_id != stable_id),
        "stale deleted chunk must leave lexical results"
    );
}

#[test]
fn test_lancedb_language_specific_tokenizers_live_e2e() {
    let cases = [
        ("ko", "오늘저녁먹음", "저녁"),
        ("ja", "関西国際空港限定トートバッグ", "空港"),
        ("zh", "我们中出了一个叛徒", "叛徒"),
        ("ar", "والكتاب في المدرسة", "كتاب"),
    ];
    for (language, content, query_text) in cases {
        let store_dir = tempfile::tempdir().expect("create store tempdir");
        let mut store =
            LanceDbStore::open_with_language(store_dir.path(), 8, Default::default(), language)
                .expect("create language-aware LanceDB store");
        store
            .upsert(&[minsync::vectorstore::Document {
                id: format!("{language}-doc"),
                embedding: vec![1.0; 8],
                text: content.to_string(),
                source_id: "source".to_string(),
                path: format!("{language}.txt"),
                chunk_schema_id: "schema".to_string(),
                chunk_type: "text".to_string(),
                heading_path: String::new(),
                content_hash: "hash".to_string(),
                seen_token: "token".to_string(),
            }])
            .expect("upsert multilingual document");
        store.flush().expect("build language-aware BM25 index");
        let hits = store
            .query_text(query_text, None, 3)
            .expect("query language-aware BM25 index");
        assert_eq!(
            hits.first().map(|hit| hit.doc_id.as_str()),
            Some(format!("{language}-doc").as_str()),
            "{language} query did not match: {hits:?}"
        );
    }
}

#[test]
fn test_watch_should_index_integration() {
    let root = std::path::Path::new("/project");
    let minsync_dir = root.join(".minsync");

    assert!(should_index(&root.join("doc.md"), &minsync_dir));
    assert!(should_index(&root.join("notes.txt"), &minsync_dir));
    assert!(should_index(&root.join("image.png"), &minsync_dir));
    assert!(!should_index(
        &minsync_dir.join("manifest.json"),
        &minsync_dir
    ));
    assert!(!should_index(&minsync_dir.join("nested.md"), &minsync_dir));
    assert!(!should_index(&root.join(".git/config"), &minsync_dir));
}

#[test]
fn test_factory_selects_backends() {
    let dir = tempfile::tempdir().expect("create tempdir");
    let mut config = Config::default_for("12345678-1234-4234-9234-123456789abc");
    config.vectorstore.id = "lancedb".to_string();
    let mut table = toml::value::Table::new();
    table.insert("dimension".into(), toml::Value::Integer(8));
    config.vectorstore.options = toml::Value::Table(table);

    let store = create_vectorstore(&config, dir.path()).expect("create lancedb store");
    assert_eq!(store.doc_count(), 0);

    // Config::default_for sets chunker.id to "recursive".
    let chunker = create_chunker(&config).expect("create recursive chunker");
    assert_eq!(chunker.schema_id(), "recursive");
}

#[tokio::test]
async fn test_tei_embedder_passage_and_query_prefixes() {
    use wiremock::matchers::{body_string_contains, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;

    // Distinguish passage vs query requests by the prefix embedded in the body.
    // Real TEI responds with a BARE 2D array (no {data:...} wrapper).
    Mock::given(method("POST"))
        .and(path("/embed"))
        .and(body_string_contains("passage: "))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!([[1.0, 0.0, 0.0, 0.0]])),
        )
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/embed"))
        .and(body_string_contains("query: "))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!([[0.0, 1.0, 0.0, 0.0]])),
        )
        .mount(&server)
        .await;

    let embedder = TeiEmbedder::new("tei:test-model", &server.uri(), 64)
        .with_passage_prefix(Some("passage: ".to_string()))
        .with_query_prefix(Some("query: ".to_string()));

    let passage = embedder
        .embed(&["doc".to_string()])
        .await
        .expect("passage embed succeeds");
    assert_eq!(passage, vec![vec![1.0, 0.0, 0.0, 0.0]]);

    let query_vec = embedder
        .embed_query("hello")
        .await
        .expect("query embed succeeds");
    assert_eq!(query_vec, vec![0.0, 1.0, 0.0, 0.0]);
}

#[tokio::test]
async fn test_sync_against_flaky_tei_server_eventually_succeeds() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

    struct OnePerInput;
    impl Respond for OnePerInput {
        fn respond(&self, request: &Request) -> ResponseTemplate {
            let body: serde_json::Value =
                serde_json::from_slice(&request.body).expect("valid embed request JSON");
            let n = body["inputs"]
                .as_array()
                .map(|a| a.len())
                .expect("inputs is an array");
            let vectors: Vec<Vec<f32>> = (0..n).map(|_| vec![0.5, 0.5]).collect();
            ResponseTemplate::new(200).set_body_json(serde_json::json!(vectors))
        }
    }

    let server = MockServer::start().await;
    // The first two requests fail with a retryable 503, then the server heals.
    Mock::given(method("POST"))
        .and(path("/embed"))
        .respond_with(ResponseTemplate::new(503).set_body_string("flaky"))
        .up_to_n_times(2)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/embed"))
        .respond_with(OnePerInput)
        .mount(&server)
        .await;

    let dir = tempfile::tempdir().expect("create tempdir");
    let sync = MinSync::new(dir.path().to_path_buf());
    let chunker = ChonkieChunker::new(32, "\n ");
    let embedder = TeiEmbedder::new("tei:test-model", &server.uri(), 64)
        .with_max_retries(3)
        .with_backoff(
            std::time::Duration::from_millis(1),
            std::time::Duration::from_millis(4),
        );
    let mut store = InMemoryStore::new();

    write_file(&dir, "doc.md", "hello world from a flaky network");
    sync.init(false, "tei:test-model", "chonkie")
        .expect("init succeeds");

    let result = sync
        .sync(&chunker, &embedder, &mut store, true, false, false)
        .await
        .expect("sync succeeds despite transient 503s");

    assert!(result.chunks_added > 0);
    assert!(store.doc_count() > 0);
    assert!(dir.path().join(".minsync/cursor.json").exists());
    assert!(
        server.received_requests().await.unwrap().len() >= 3,
        "expected the two 503s to be retried"
    );
}

#[tokio::test]
async fn test_sync_embedding_failure_keeps_cursor_and_manifest_unchanged() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embed"))
        .respond_with(ResponseTemplate::new(503).set_body_string("down"))
        .mount(&server)
        .await;

    let dir = tempfile::tempdir().expect("create tempdir");
    let sync = MinSync::new(dir.path().to_path_buf());
    let chunker = ChonkieChunker::new(32, "\n ");
    let embedder = TeiEmbedder::new("tei:test-model", &server.uri(), 64)
        .with_max_retries(1)
        .with_backoff(
            std::time::Duration::from_millis(1),
            std::time::Duration::from_millis(4),
        );
    let mut store = InMemoryStore::new();

    sync.init(false, "tei:test-model", "chonkie")
        .expect("init succeeds");
    let manifest_before = std::fs::read_to_string(dir.path().join(".minsync/manifest.json"))
        .expect("read baseline manifest");
    write_file(&dir, "doc.md", "content that will fail to embed");

    let error = sync
        .sync(&chunker, &embedder, &mut store, false, false, false)
        .await
        .expect_err("sync fails after retry exhaustion");

    assert!(error.to_string().contains("retries exhausted"));
    assert!(
        !dir.path().join(".minsync/cursor.json").exists(),
        "cursor must not advance on embedding failure"
    );
    assert!(
        dir.path().join(".minsync/txn.json").exists(),
        "interrupted transaction must remain for recovery"
    );
    let manifest_after = std::fs::read_to_string(dir.path().join(".minsync/manifest.json"))
        .expect("read manifest after failed sync");
    assert_eq!(
        manifest_before, manifest_after,
        "manifest must not advance on embedding failure"
    );

    // The next sync against a healed server converges.
    struct OnePerInput;
    impl wiremock::Respond for OnePerInput {
        fn respond(&self, request: &wiremock::Request) -> ResponseTemplate {
            let body: serde_json::Value =
                serde_json::from_slice(&request.body).expect("valid embed request JSON");
            let n = body["inputs"]
                .as_array()
                .map(|a| a.len())
                .expect("inputs is an array");
            let vectors: Vec<Vec<f32>> = (0..n).map(|_| vec![0.5, 0.5]).collect();
            ResponseTemplate::new(200).set_body_json(serde_json::json!(vectors))
        }
    }
    let healed = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embed"))
        .respond_with(OnePerInput)
        .mount(&healed)
        .await;
    let healed_embedder = TeiEmbedder::new("tei:test-model", &healed.uri(), 64);

    let result = sync
        .sync(&chunker, &healed_embedder, &mut store, false, false, false)
        .await
        .expect("sync converges once the server heals");

    assert!(result.chunks_added > 0);
    assert!(dir.path().join(".minsync/cursor.json").exists());
    assert!(!dir.path().join(".minsync/txn.json").exists());
}

#[tokio::test]
async fn test_tei_full_sync_and_query_pipeline() {
    use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

    // Dynamic responder: parse {"inputs":[...]} and emit one 4-dim vector per
    // input, so the mock is robust regardless of how many chunks sync sends.
    struct OnePerInput;
    impl Respond for OnePerInput {
        fn respond(&self, request: &Request) -> ResponseTemplate {
            let body: serde_json::Value =
                serde_json::from_slice(&request.body).expect("valid embed request JSON");
            let n = body["inputs"]
                .as_array()
                .map(|a| a.len())
                .expect("inputs is an array");
            let vectors: Vec<Vec<f32>> = (0..n).map(|_| vec![0.5, 0.5, 0.5, 0.5]).collect();
            ResponseTemplate::new(200).set_body_json(serde_json::json!(vectors))
        }
    }

    let server = MockServer::start().await;
    Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/embed"))
        .respond_with(OnePerInput)
        .mount(&server)
        .await;

    let dir = tempfile::tempdir().expect("create tempdir");
    let sync = MinSync::new(dir.path().to_path_buf());
    let chunker = RecursiveChunker::new(64);
    let embedder = TeiEmbedder::new("tei:test-model", &server.uri(), 64);
    let mut store = InMemoryStore::new();

    write_file(&dir, "doc.md", "hello world");

    // init baselines the manifest, so the first real sync must use full=true.
    sync.init(false, "tei:test-model", "recursive")
        .expect("init succeeds");
    let result = sync
        .sync(&chunker, &embedder, &mut store, true, false, false)
        .await
        .expect("tei full sync succeeds");

    assert!(result.chunks_added > 0);
    assert!(store.doc_count() > 0);

    let query_results = query(
        &dir.path().join(".minsync"),
        "hello",
        1,
        &embedder,
        &store,
        None,
        QueryMode::Vector,
    )
    .await
    .expect("tei query succeeds");

    assert!(!query_results.is_empty());
}
