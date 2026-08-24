use assert_cmd::Command;
use minsync::config::Config;
use minsync::state::Cursor;
use minsync::vectorstore::lancedb_store::LanceDbStore;
use minsync::vectorstore::{Document, VectorStore};

#[test]
fn bm25_cli_queries_live_lancedb_without_embedder_credentials() {
    let root = tempfile::tempdir().expect("create workspace");
    let minsync_dir = root.path().join(".minsync");
    std::fs::create_dir_all(&minsync_dir).expect("create state directory");

    let source_id = "source-live-bm25";
    let mut config = Config::default_for(source_id);
    config.embedder.id = "openai:text-embedding-3-small".to_string();
    let mut options = toml::value::Table::new();
    options.insert("dimension".into(), toml::Value::Integer(4));
    config.vectorstore.options = toml::Value::Table(options);
    config
        .save(&minsync_dir.join("config.toml"))
        .expect("save config");
    Cursor {
        source_id: source_id.to_string(),
        last_synced_at: "2026-08-24T00:00:00Z".to_string(),
        manifest_hash: "sha256:live".to_string(),
        chunk_schema_id: "recursive".to_string(),
        embedder_id: config.embedder.id.clone(),
        collection_path: config.collection.path.clone(),
        lexical_language: config.lexical.language.clone(),
    }
    .save(&minsync_dir.join("cursor.json"))
    .expect("save cursor");

    let store_path = minsync_dir.join(&config.collection.path);
    let mut store = LanceDbStore::open_or_create(&store_path, 4).expect("open LanceDB");
    store
        .upsert(&[Document {
            id: "shared-chunk-id".to_string(),
            embedding: vec![1.0, 0.0, 0.0, 0.0],
            text: "한국어 환불 정책과 mixed language refund policy".to_string(),
            source_id: source_id.to_string(),
            path: "policy.md".to_string(),
            chunk_schema_id: "recursive".to_string(),
            chunk_type: "text".to_string(),
            heading_path: "환불 정책".to_string(),
            content_hash: "sha256:chunk".to_string(),
            seen_token: "live".to_string(),
        }])
        .expect("upsert shared chunk");
    store.flush().expect("build BM25 index");
    drop(store);

    let output = Command::cargo_bin("minsync")
        .expect("find minsync binary")
        .current_dir(root.path())
        .env_remove("OPENAI_API_KEY")
        .args(["--format", "json", "query", "환불 정책", "--mode", "bm25"])
        .output()
        .expect("run BM25 CLI");

    assert!(
        output.status.success(),
        "BM25 CLI failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let results: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("parse JSON results");
    assert_eq!(results[0]["doc_id"], "shared-chunk-id");
    assert_eq!(results[0]["path"], "policy.md");
    assert_eq!(results[0]["mode"], "bm25");
    assert_eq!(results[0]["bm25_rank"], 1);
    assert!(results[0].get("vector_rank").is_none());
    assert!(
        String::from_utf8_lossy(&output.stderr).is_empty(),
        "BM25 mode must not attempt an embedding request"
    );
}

#[test]
fn init_cli_persists_multilingual_language_option() {
    let root = tempfile::tempdir().expect("create workspace");
    Command::cargo_bin("minsync")
        .expect("find minsync binary")
        .current_dir(root.path())
        .args(["init", "--language", "multilingual"])
        .assert()
        .success();

    let config =
        Config::load(&root.path().join(".minsync/config.toml")).expect("load initialized config");
    assert_eq!(config.lexical.language, "multilingual");
}
