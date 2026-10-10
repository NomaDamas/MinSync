use assert_cmd::Command;
use minsync::config::Config;
use minsync::vectorstore::lancedb_store::LanceDbStore;
use minsync::vectorstore::{Document, VectorStore};

fn doc(i: usize) -> Document {
    Document {
        id: format!("chunk-{i}"),
        embedding: vec![1.0, 0.0, 0.0, 0.0],
        text: format!("fragment {i}"),
        source_id: "source-optimize".to_string(),
        path: "doc.md".to_string(),
        chunk_schema_id: "recursive".to_string(),
        chunk_type: "text".to_string(),
        heading_path: String::new(),
        content_hash: format!("sha256:{i}"),
        seen_token: "t".to_string(),
    }
}

/// A store that a killed sync left fragmented: one fragment per commit.
fn fragmented_workspace(chunks: usize) -> tempfile::TempDir {
    let root = tempfile::tempdir().expect("create workspace");
    let minsync_dir = root.path().join(".minsync");
    std::fs::create_dir_all(&minsync_dir).expect("create state directory");
    let mut config = Config::default_for("source-optimize");
    let mut options = toml::value::Table::new();
    options.insert("dimension".into(), toml::Value::Integer(4));
    config.vectorstore.options = toml::Value::Table(options);
    config
        .save(&minsync_dir.join("config.toml"))
        .expect("save config");
    let store_path = minsync_dir.join(&config.collection.path);
    let mut store = LanceDbStore::open_with_language(&store_path, 4, Default::default(), "simple")
        .expect("open LanceDB");
    for i in 0..chunks {
        store.upsert(&[doc(i)]).expect("upsert chunk");
    }
    assert_eq!(store.fragment_count().expect("fragments"), chunks);
    root
}

#[test]
fn optimize_compacts_a_fragmented_store_without_losing_rows() {
    let root = fragmented_workspace(30);

    let output = Command::cargo_bin("minsync")
        .expect("find minsync binary")
        .current_dir(root.path())
        .args(["--format", "json", "optimize"])
        .output()
        .expect("run optimize");
    assert!(
        output.status.success(),
        "optimize failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let json: serde_json::Value = serde_json::from_slice(&output.stdout).expect("json output");
    assert_eq!(json["optimized"], true);

    let store_path = root.path().join(".minsync/store");
    let store = LanceDbStore::open_with_language(&store_path, 4, Default::default(), "simple")
        .expect("reopen LanceDB");
    assert_eq!(store.doc_count(), 30);
    assert_eq!(store.fragment_count().expect("fragments"), 1);
}

#[test]
fn optimize_fails_without_init() {
    let root = tempfile::tempdir().expect("create workspace");
    let output = Command::cargo_bin("minsync")
        .expect("find minsync binary")
        .current_dir(root.path())
        .arg("optimize")
        .output()
        .expect("run optimize");
    assert!(!output.status.success());
}
