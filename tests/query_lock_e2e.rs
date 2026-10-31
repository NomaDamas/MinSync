use assert_cmd::Command;
use minsync::config::Config;
use minsync::sparse::native::NativeSparseEncoder;
use minsync::sparse::SparseEncoder;
use minsync::state::{Cursor, FileLock};
use minsync::vectorstore::lancedb_store::LanceDbStore;
use minsync::vectorstore::{Document, VectorStore};

mod common;

/// Prepare an initialized, synced workspace with one indexed document that
/// carries a sparse vector from the same tiny model the config names.
async fn synced_workspace() -> tempfile::TempDir {
    let root = tempfile::tempdir().expect("create workspace");
    let minsync_dir = root.path().join(".minsync");
    std::fs::create_dir_all(&minsync_dir).expect("create state directory");

    let source_id = "source-lock-test";
    let mut config = Config::default_for(source_id);
    let mut options = toml::value::Table::new();
    options.insert("dimension".into(), toml::Value::Integer(4));
    config.vectorstore.options = toml::Value::Table(options);
    common::use_tiny_sparse(&mut config);
    config
        .save(&minsync_dir.join("config.toml"))
        .expect("save config");

    // The native encoder records its id without the `native:` prefix, and the
    // stored sparse vectors are keyed by exactly that string.
    let encoder = NativeSparseEncoder::from_config(&config.sparse).expect("build tiny encoder");
    let model = encoder.id().to_string();
    let text = "w1 w2 concurrent query target".to_string();
    let vector = encoder
        .encode(std::slice::from_ref(&text))
        .await
        .expect("encode passage")
        .into_iter()
        .next()
        .expect("one vector");

    Cursor {
        source_id: source_id.to_string(),
        last_synced_at: "2026-09-20T00:00:00Z".to_string(),
        manifest_hash: "sha256:lock".to_string(),
        chunk_schema_id: "recursive".to_string(),
        embedder_id: config.embedder.id.clone(),
        collection_path: config.collection.path.clone(),
        sparse_id: Some(model.clone()),
    }
    .save(&minsync_dir.join("cursor.json"))
    .expect("save cursor");

    let store_path = minsync_dir.join(&config.collection.path);
    let mut store =
        LanceDbStore::open_with_indexing(&store_path, 4, Default::default()).expect("open LanceDB");
    store
        .upsert(&[Document {
            id: "chunk-1".to_string(),
            embedding: vec![1.0, 0.0, 0.0, 0.0],
            text,
            source_id: source_id.to_string(),
            path: "doc.md".to_string(),
            chunk_schema_id: "recursive".to_string(),
            chunk_type: "text".to_string(),
            heading_path: String::new(),
            content_hash: "sha256:chunk".to_string(),
            seen_token: "lock".to_string(),
        }])
        .expect("upsert chunk");
    store
        .set_sparse(&model, &[("chunk-1".to_string(), vector)])
        .expect("attach sparse vector");
    root
}

fn run_minsync(root: &std::path::Path, args: &[&str]) -> std::process::Output {
    Command::cargo_bin("minsync")
        .expect("find minsync binary")
        .current_dir(root)
        .args(args)
        .output()
        .expect("run minsync")
}

#[tokio::test(flavor = "multi_thread")]
async fn query_succeeds_while_writer_lock_is_held() {
    let root = synced_workspace().await;
    // Simulate a running sync: hold the workspace lock for the duration of
    // the query, as `minsync sync` would.
    let _lock = FileLock::acquire(&root.path().join(".minsync/lock"), false)
        .expect("acquire workspace lock");

    let root_path = root.path().to_path_buf();
    let output = tokio::task::spawn_blocking(move || {
        run_minsync(
            &root_path,
            &["--format", "json", "query", "w1", "--mode", "sparse"],
        )
    })
    .await
    .expect("join");

    assert!(
        output.status.success(),
        "query must not require the writer lock: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let results: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("parse JSON results");
    assert_eq!(results[0]["doc_id"], "chunk-1");
}

#[tokio::test(flavor = "multi_thread")]
async fn sync_still_fails_while_writer_lock_is_held() {
    let root = synced_workspace().await;
    let _lock = FileLock::acquire(&root.path().join(".minsync/lock"), false)
        .expect("acquire workspace lock");

    let root_path = root.path().to_path_buf();
    let output = tokio::task::spawn_blocking(move || run_minsync(&root_path, &["sync"]))
        .await
        .expect("join");

    assert_eq!(
        output.status.code(),
        Some(3),
        "sync must still reject a locked workspace; stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
}
