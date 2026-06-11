use minsync::chunker::chonkie::ChonkieChunker;
use minsync::chunker::create_chunker;
use minsync::chunker::recursive::RecursiveChunker;
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
use minsync::watch::should_index;
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

    let query_results = query(
        &dir.path().join(".minsync"),
        "apples",
        5,
        &embedder,
        &store,
        None,
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

#[test]
fn test_watch_should_index_integration() {
    let root = std::path::Path::new("/project");
    let minsync_dir = root.join(".minsync");

    assert!(should_index(&root.join("doc.md"), &minsync_dir));
    assert!(should_index(&root.join("notes.txt"), &minsync_dir));
    assert!(!should_index(&root.join("image.png"), &minsync_dir));
    assert!(!should_index(
        &minsync_dir.join("manifest.json"),
        &minsync_dir
    ));
    assert!(!should_index(&minsync_dir.join("nested.md"), &minsync_dir));
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
    )
    .await
    .expect("tei query succeeds");

    assert!(!query_results.is_empty());
}
