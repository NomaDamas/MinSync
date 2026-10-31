mod common;

use assert_cmd::Command;
use minsync::chunker::create_chunker;
use minsync::config::Config;
use minsync::state::Cursor;
use minsync::vectorstore::lancedb_store::LanceDbStore;
use minsync::vectorstore::{Document, VectorStore};

#[test]
fn sparse_cli_queries_live_lancedb_without_embedder_credentials() {
    let root = tempfile::tempdir().expect("create workspace");
    let minsync_dir = root.path().join(".minsync");
    std::fs::create_dir_all(&minsync_dir).expect("create state directory");

    let source_id = "source-live-sparse";
    let mut config = Config::default_for(source_id);
    // A dense embedder that would need an API key: sparse search must never touch it.
    config.embedder.id = "openai:text-embedding-3-small".to_string();
    common::use_tiny_sparse(&mut config);
    let mut options = toml::value::Table::new();
    options.insert("dimension".into(), toml::Value::Integer(4));
    config.vectorstore.options = toml::Value::Table(options);
    config
        .save(&minsync_dir.join("config.toml"))
        .expect("save config");
    let model = common::tiny_model_id();
    Cursor {
        source_id: source_id.to_string(),
        last_synced_at: "2026-08-24T00:00:00Z".to_string(),
        manifest_hash: "sha256:live".to_string(),
        chunk_schema_id: "recursive".to_string(),
        embedder_id: config.embedder.id.clone(),
        collection_path: config.collection.path.clone(),
        sparse_id: Some(model.clone()),
    }
    .save(&minsync_dir.join("cursor.json"))
    .expect("save cursor");

    let texts = ["w1 w2 w3 w4", "w20 w21 w22"];
    let vectors = common::encode_with_tiny(&texts, false);
    let store_path = minsync_dir.join(&config.collection.path);
    let mut store =
        LanceDbStore::open_with_indexing(&store_path, 4, Default::default()).expect("open LanceDB");
    let docs: Vec<Document> = texts
        .iter()
        .enumerate()
        .map(|(i, text)| Document {
            id: format!("chunk-{i}"),
            embedding: vec![1.0, 0.0, 0.0, 0.0],
            text: text.to_string(),
            source_id: source_id.to_string(),
            path: format!("doc-{i}.md"),
            chunk_schema_id: "recursive".to_string(),
            chunk_type: "text".to_string(),
            heading_path: format!("heading {i}"),
            content_hash: format!("sha256:chunk-{i}"),
            seen_token: "live".to_string(),
        })
        .collect();
    store.upsert(&docs).expect("upsert chunks");
    let rows: Vec<(String, minsync::sparse::SparseVector)> = docs
        .iter()
        .zip(vectors.iter().cloned())
        .map(|(doc, vector)| (doc.id.clone(), vector))
        .collect();
    store.set_sparse(&model, &rows).expect("set sparse vectors");
    store.flush().expect("flush");
    drop(store);

    let output = Command::cargo_bin("minsync")
        .expect("find minsync binary")
        .current_dir(root.path())
        .env_remove("OPENAI_API_KEY")
        .env("MINSYNC_SERVER", "off")
        .args(["--format", "json", "query", "w1 w2 w3", "--mode", "sparse"])
        .output()
        .expect("run sparse CLI");

    assert!(
        output.status.success(),
        "sparse CLI failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let results: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("parse JSON results");
    assert_eq!(results[0]["doc_id"], "chunk-0");
    assert_eq!(results[0]["path"], "doc-0.md");
    assert_eq!(results[0]["heading_path"], "heading 0");
    assert_eq!(results[0]["mode"], "sparse");
    assert_eq!(results[0]["sparse_rank"], 1);
    assert!(results[0].get("vector_rank").is_none());
    assert!(
        String::from_utf8_lossy(&output.stderr).is_empty(),
        "sparse mode must not attempt an embedding request; stderr={:?}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn sparse_query_on_an_index_from_another_model_fails_naming_the_fix() {
    let root = tempfile::tempdir().expect("create workspace");
    let minsync_dir = root.path().join(".minsync");
    std::fs::create_dir_all(&minsync_dir).expect("create state directory");
    let mut config = Config::default_for("source-stale");
    common::use_tiny_sparse(&mut config);
    let mut options = toml::value::Table::new();
    options.insert("dimension".into(), toml::Value::Integer(4));
    config.vectorstore.options = toml::Value::Table(options);
    config
        .save(&minsync_dir.join("config.toml"))
        .expect("save config");
    // The last sync encoded with some other model (or none: a BM25-era cursor).
    for sparse_id in [Some("someone/else".to_string()), None] {
        Cursor {
            source_id: "source-stale".to_string(),
            last_synced_at: "2026-08-24T00:00:00Z".to_string(),
            manifest_hash: "sha256:live".to_string(),
            chunk_schema_id: "recursive".to_string(),
            embedder_id: config.embedder.id.clone(),
            collection_path: config.collection.path.clone(),
            sparse_id,
        }
        .save(&minsync_dir.join("cursor.json"))
        .expect("save cursor");
        let mut store = LanceDbStore::open_with_indexing(
            &minsync_dir.join(&config.collection.path),
            4,
            Default::default(),
        )
        .expect("open LanceDB");
        store.flush().expect("flush");
        drop(store);

        let output = Command::cargo_bin("minsync")
            .expect("find minsync binary")
            .current_dir(root.path())
            .env("MINSYNC_SERVER", "off")
            .args(["query", "w1", "--mode", "sparse"])
            .output()
            .expect("run sparse CLI");

        assert!(!output.status.success());
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("not ready") && stderr.contains("minsync sync"),
            "{stderr}"
        );
    }
}

#[test]
fn uninitialized_commands_report_not_initialized() {
    let root = tempfile::tempdir().expect("create workspace");

    for args in [
        vec!["query", "alpha", "--mode", "sparse"],
        vec!["check"],
        vec!["verify"],
        vec!["watch"],
    ] {
        let output = Command::cargo_bin("minsync")
            .expect("find minsync binary")
            .current_dir(root.path())
            .args(&args)
            .output()
            .expect("run command");

        assert_eq!(
            output.status.code(),
            Some(1),
            "command {:?} should fail with a user error",
            args
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("not initialized"),
            "command {:?} should explain initialization requirement; stderr={stderr:?}",
            args
        );
        assert!(
            !root.path().join(".minsync/lock").exists(),
            "command {:?} should not create state in an uninitialized workspace",
            args
        );
    }
}

#[test]
fn verify_failure_returns_nonzero_exit_code() {
    let root = tempfile::tempdir().expect("create workspace");
    Command::cargo_bin("minsync")
        .expect("find minsync binary")
        .current_dir(root.path())
        .args(["init"])
        .assert()
        .success();

    let output = Command::cargo_bin("minsync")
        .expect("find minsync binary")
        .current_dir(root.path())
        .args(["--format", "json", "verify"])
        .output()
        .expect("run verify");

    assert_eq!(output.status.code(), Some(1));
    let result: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("parse verification result");
    assert_eq!(result["all_passed"], false);
}

#[test]
fn verify_cli_reports_duplicate_vector_corruption_pairs() {
    let root = tempfile::tempdir().expect("create workspace");
    std::fs::write(
        root.path().join("first.txt"),
        "apple orchard weather report",
    )
    .expect("write first file");
    std::fs::write(
        root.path().join("second.txt"),
        "quantum mechanics lecture notes",
    )
    .expect("write second file");

    Command::cargo_bin("minsync")
        .expect("find minsync binary")
        .current_dir(root.path())
        .args(["init"])
        .assert()
        .success();

    let minsync_dir = root.path().join(".minsync");
    let mut config = Config::load(&minsync_dir.join("config.toml")).expect("load config");
    let mut options = toml::value::Table::new();
    options.insert("dimension".into(), toml::Value::Integer(4));
    config.vectorstore.options = toml::Value::Table(options);
    config
        .save(&minsync_dir.join("config.toml"))
        .expect("save test config");
    let chunker = create_chunker(&config).expect("create chunker");
    Cursor {
        source_id: config.source_id.clone(),
        last_synced_at: "2026-09-30T00:00:00Z".to_string(),
        manifest_hash: "sha256:test".to_string(),
        chunk_schema_id: chunker.schema_id().to_string(),
        embedder_id: config.embedder.id.clone(),
        collection_path: config.collection.path.clone(),
        sparse_id: None,
    }
    .save(&minsync_dir.join("cursor.json"))
    .expect("save cursor");

    let mut store = LanceDbStore::open_with_indexing(
        &minsync_dir.join(&config.collection.path),
        4,
        Default::default(),
    )
    .expect("open LanceDB");
    store
        .upsert(&[
            Document {
                id: "chunk-a".to_string(),
                embedding: vec![1.0, 0.0, 0.0, 0.0],
                text: "apple orchard weather report".to_string(),
                source_id: config.source_id.clone(),
                path: "first.txt".to_string(),
                chunk_schema_id: chunker.schema_id().to_string(),
                chunk_type: "text".to_string(),
                heading_path: String::new(),
                content_hash: "hash-a".to_string(),
                seen_token: "token".to_string(),
            },
            Document {
                id: "chunk-b".to_string(),
                embedding: vec![1.0, 0.0, 0.0, 0.0],
                text: "quantum mechanics lecture notes".to_string(),
                source_id: config.source_id,
                path: "second.txt".to_string(),
                chunk_schema_id: chunker.schema_id().to_string(),
                chunk_type: "text".to_string(),
                heading_path: String::new(),
                content_hash: "hash-b".to_string(),
                seen_token: "token".to_string(),
            },
        ])
        .expect("upsert corrupt documents");
    store.flush().expect("flush LanceDB");
    drop(store);

    let output = Command::cargo_bin("minsync")
        .expect("find minsync binary")
        .current_dir(root.path())
        .args(["--format", "json", "verify"])
        .output()
        .expect("run verify");

    assert_eq!(output.status.code(), Some(1));
    let result: serde_json::Value =
        serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
            panic!(
                "parse verification result: {error}; stdout={:?}; stderr={:?}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            )
        });
    assert_eq!(
        result["embedding_integrity"]["flagged_pairs"]
            .as_array()
            .unwrap_or_else(|| panic!("verification JSON: {result}"))
            .len(),
        1
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("verification failed"),
        "stderr={:?}",
        String::from_utf8_lossy(&output.stderr)
    );

    let text_output = Command::cargo_bin("minsync")
        .expect("find minsync binary")
        .current_dir(root.path())
        .args(["verify"])
        .output()
        .expect("run text verify");
    assert_eq!(text_output.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&text_output.stdout)
            .contains("Recommendation: run minsync sync --full"),
        "stdout={:?}",
        String::from_utf8_lossy(&text_output.stdout)
    );
}

#[test]
fn build_corrupt_verify_fixture_from_environment() {
    let Some(path) = std::env::var_os("MINSYNC_CORRUPT_FIXTURE") else {
        return;
    };
    let root = std::path::PathBuf::from(path);
    std::fs::create_dir_all(&root).expect("create fixture root");
    std::fs::write(root.join("first.txt"), "apple orchard weather report")
        .expect("write first file");
    std::fs::write(root.join("second.txt"), "quantum mechanics lecture notes")
        .expect("write second file");
    Command::cargo_bin("minsync")
        .expect("find minsync binary")
        .current_dir(&root)
        .args(["init"])
        .assert()
        .success();

    let minsync_dir = root.join(".minsync");
    let mut config = Config::load(&minsync_dir.join("config.toml")).expect("load config");
    let mut options = toml::value::Table::new();
    options.insert("dimension".into(), toml::Value::Integer(4));
    config.vectorstore.options = toml::Value::Table(options);
    config
        .save(&minsync_dir.join("config.toml"))
        .expect("save test config");
    let chunker = create_chunker(&config).expect("create chunker");
    Cursor {
        source_id: config.source_id.clone(),
        last_synced_at: "2026-09-30T00:00:00Z".to_string(),
        manifest_hash: "sha256:test".to_string(),
        chunk_schema_id: chunker.schema_id().to_string(),
        embedder_id: config.embedder.id.clone(),
        collection_path: config.collection.path.clone(),
        sparse_id: None,
    }
    .save(&minsync_dir.join("cursor.json"))
    .expect("save cursor");
    let mut store = LanceDbStore::open_with_indexing(
        &minsync_dir.join(&config.collection.path),
        4,
        Default::default(),
    )
    .expect("open LanceDB");
    store
        .upsert(&[
            Document {
                id: "chunk-a".to_string(),
                embedding: vec![1.0, 0.0, 0.0, 0.0],
                text: "apple orchard weather report".to_string(),
                source_id: config.source_id.clone(),
                path: "first.txt".to_string(),
                chunk_schema_id: chunker.schema_id().to_string(),
                chunk_type: "text".to_string(),
                heading_path: String::new(),
                content_hash: "hash-a".to_string(),
                seen_token: "token".to_string(),
            },
            Document {
                id: "chunk-b".to_string(),
                embedding: vec![1.0, 0.0, 0.0, 0.0],
                text: "quantum mechanics lecture notes".to_string(),
                source_id: config.source_id,
                path: "second.txt".to_string(),
                chunk_schema_id: chunker.schema_id().to_string(),
                chunk_type: "text".to_string(),
                heading_path: String::new(),
                content_hash: "hash-b".to_string(),
                seen_token: "token".to_string(),
            },
        ])
        .expect("upsert corrupt documents");
    store.flush().expect("flush LanceDB");
}
