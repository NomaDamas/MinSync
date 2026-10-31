//! Upgrading a workspace that was indexed by the BM25 release.
//!
//! The on-disk shape of that release is rebuilt here byte for byte where it
//! matters: a `config.toml` with a `[lexical]` table, a `cursor.json` with
//! `lexical_language` and no `sparse_id`, and a LanceDB table that has a
//! `lexical_text` column with a full-text index and no sparse columns. Then the
//! real `minsync` binary runs the commands a user would.

mod common;

use arrow_array::types::Float32Type;
use arrow_array::{FixedSizeListArray, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use assert_cmd::Command;
use lancedb::index::scalar::FtsIndexBuilder;
use lancedb::index::Index;
use minsync::config::Config;
use minsync::manifest::Manifest;
use minsync::vectorstore::lancedb_store::LanceDbStore;
use minsync::vectorstore::VectorStore;
use std::path::Path;
use std::sync::Arc;

const DIM: usize = 4;
const SOURCE_ID: &str = "source-upgrade";

/// path -> text, one chunk each.
const FILES: [(&str, &str); 3] = [
    ("alpha.md", "w1 w2 w3 w4 w5"),
    ("beta.md", "w10 w11 w12 w13"),
    ("gamma.md", "w20 w21 w22 w23 w24"),
];

fn minsync(root: &Path) -> Command {
    let mut command = Command::cargo_bin("minsync").expect("find minsync binary");
    command
        .current_dir(root)
        .env_remove("OPENAI_API_KEY")
        .env("MINSYNC_SERVER", "off");
    command
}

fn run_json(root: &Path, args: &[&str]) -> serde_json::Value {
    let output = minsync(root)
        .args(["--format", "json"])
        .args(args)
        .output()
        .expect("run minsync");
    assert!(
        output.status.success(),
        "`minsync {}` failed: {}",
        args.join(" "),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("json output")
}

/// A workspace exactly as the BM25 release left it after a completed sync.
fn bm25_era_workspace() -> tempfile::TempDir {
    let root = tempfile::tempdir().expect("create workspace");
    for (name, text) in FILES {
        std::fs::write(root.path().join(name), format!("{text}\n")).expect("write file");
    }
    // Let the current binary lay out the state directory, then rewrite it in
    // the old shape.
    minsync(root.path())
        .args(["init", "--embedder", "tei:test/model"])
        .assert()
        .success();
    let minsync_dir = root.path().join(".minsync");
    let config_path = minsync_dir.join("config.toml");
    let mut config = Config::load(&config_path).expect("load config");
    config.source_id = SOURCE_ID.to_string();
    common::use_tiny_sparse(&mut config);
    let mut options = toml::value::Table::new();
    options.insert("dimension".into(), toml::Value::Integer(DIM as i64));
    config.vectorstore.options = toml::Value::Table(options);
    config.save(&config_path).expect("save config");
    let mut text = std::fs::read_to_string(&config_path).expect("read config");
    text.push_str("\n[lexical]\nlanguage = \"ko\"\n");
    std::fs::write(&config_path, text).expect("write legacy config");

    // The manifest of the finished sync, and a cursor that matches it.
    let (manifest, _, _) = Manifest::scan_keeping_unknown_with_max_file_bytes(
        root.path(),
        SOURCE_ID,
        None,
        None,
        config.sync.max_file_bytes,
    )
    .expect("scan");
    manifest
        .save(&minsync_dir.join("manifest.json"))
        .expect("save manifest");
    std::fs::write(
        minsync_dir.join("cursor.json"),
        serde_json::json!({
            "source_id": SOURCE_ID,
            "last_synced_at": "2026-08-24T00:00:00Z",
            "manifest_hash": manifest.manifest_hash(),
            "chunk_schema_id": "recursive",
            "embedder_id": "test/model",
            "collection_path": config.collection.path,
            "lexical_language": "ko",
        })
        .to_string(),
    )
    .expect("write legacy cursor");

    write_bm25_era_table(&minsync_dir.join(&config.collection.path));
    root
}

fn write_bm25_era_table(path: &Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                DIM as i32,
            ),
            false,
        ),
        Field::new("text", DataType::Utf8, false),
        Field::new("lexical_text", DataType::Utf8, false),
        Field::new("source_id", DataType::Utf8, false),
        Field::new("path", DataType::Utf8, false),
        Field::new("chunk_schema_id", DataType::Utf8, false),
        Field::new("chunk_type", DataType::Utf8, false),
        Field::new("heading_path", DataType::Utf8, false),
        Field::new("content_hash", DataType::Utf8, false),
        Field::new("seen_token", DataType::Utf8, false),
    ]));
    let column = |f: &dyn Fn(usize, &str, &str) -> String| -> Arc<StringArray> {
        Arc::new(StringArray::from_iter_values(
            FILES
                .iter()
                .enumerate()
                .map(|(i, (name, text))| f(i, name, text)),
        ))
    };
    let vectors = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        FILES
            .iter()
            .enumerate()
            .map(|(i, _)| Some(vec![Some(i as f32 + 1.0), Some(0.5), Some(0.0), Some(0.0)])),
        DIM as i32,
    );
    let batch = RecordBatch::try_new(
        schema,
        vec![
            column(&|i, _, _| format!("legacy-{i}")),
            Arc::new(vectors),
            column(&|_, _, text| text.to_string()),
            column(&|_, _, text| format!("tokenized {text}")),
            column(&|_, _, _| SOURCE_ID.to_string()),
            column(&|_, name, _| name.to_string()),
            column(&|_, _, _| "recursive".to_string()),
            column(&|_, _, _| "text".to_string()),
            column(&|_, _, _| String::new()),
            column(&|_, _, text| minsync::id::content_hash(text)),
            column(&|_, _, _| "old-token".to_string()),
        ],
    )
    .expect("legacy batch");
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    runtime.block_on(async {
        let conn = lancedb::connect(path.to_str().unwrap())
            .execute()
            .await
            .expect("connect");
        let table = conn
            .create_table("documents", vec![batch])
            .execute()
            .await
            .expect("create legacy table");
        table
            .create_index(&["lexical_text"], Index::FTS(FtsIndexBuilder::default()))
            .execute()
            .await
            .expect("legacy FTS index");
    });
}

#[test]
fn a_plain_sync_upgrades_a_bm25_index_in_place_without_re_embedding() {
    let root = bm25_era_workspace();

    // Before: reading is refused with the way out, and nothing was modified.
    let before = minsync(root.path())
        .args(["query", "w1", "--mode", "sparse"])
        .output()
        .expect("query before upgrade");
    assert!(!before.status.success());
    let stderr = String::from_utf8_lossy(&before.stderr);
    assert!(stderr.contains("minsync sync"), "{stderr}");

    // The dense embedder is unreachable (no TEI server exists): an upgrade
    // that tried to re-embed anything would fail here.
    let sync = run_json(root.path(), &["sync"]);

    assert_eq!(
        sync["already_up_to_date"], false,
        "the sparse backlog is work"
    );
    assert_eq!(sync["files_processed"], 0, "no file is re-read");
    assert_eq!(sync["chunks_added"], 0);
    assert_eq!(sync["embedding_api_calls"], 0);
    assert_eq!(sync["embedded_texts"], 0);
    assert_eq!(sync["sparse_encoded"], 3);
    assert_eq!(sync["partial"], false);

    let cursor: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root.path().join(".minsync/cursor.json")).expect("read cursor"),
    )
    .expect("cursor json");
    assert_eq!(cursor["sparse_id"], common::tiny_model_id());
    assert!(
        cursor.get("lexical_language").is_none(),
        "the old key is gone"
    );

    // After: sparse search answers exactly as a from-scratch sync would.
    let texts: Vec<&str> = FILES.iter().map(|(_, text)| *text).collect();
    let docs = common::encode_with_tiny(&texts, false);
    let query = common::encode_with_tiny(&["w10 w11 w12"], true).remove(0);
    let expected: Vec<String> = common::brute_force_ranking(&docs, &query)
        .into_iter()
        .map(|i| FILES[i].0.to_string())
        .collect();
    let results = run_json(root.path(), &["query", "w10 w11 w12", "--mode", "sparse"]);
    let paths: Vec<String> = results
        .as_array()
        .unwrap()
        .iter()
        .map(|hit| hit["path"].as_str().unwrap().to_string())
        .collect();
    assert!(
        expected.len() >= 2,
        "the oracle must rank several documents for the order to mean anything: {expected:?}"
    );
    assert_eq!(
        paths, expected,
        "the migrated index ranks like the model does"
    );

    // The store is a sparse-ready table now: no FTS column, dense vectors kept.
    let store_path = root.path().join(".minsync").join("store");
    let store = LanceDbStore::open_readonly(&store_path, DIM, Default::default())
        .expect("a migrated table opens read-only");
    let kept = store
        .fetch(&["legacy-1".to_string()])
        .expect("fetch legacy row");
    assert_eq!(
        kept[0].embedding,
        vec![2.0, 0.5, 0.0, 0.0],
        "dense vector untouched"
    );

    // And it stays up to date from here on.
    let again = run_json(root.path(), &["sync"]);
    assert_eq!(again["already_up_to_date"], true);
    assert_eq!(again["sparse_encoded"], 0);
}

#[test]
fn the_upgraded_workspace_keeps_syncing_incrementally() {
    let root = bm25_era_workspace();
    run_json(root.path(), &["sync"]);

    std::fs::write(root.path().join("delta.md"), "w30 w31 w32\n").expect("add file");
    std::fs::remove_file(root.path().join("alpha.md")).expect("delete file");

    // The new chunk needs a dense vector: serve it from a TEI stand-in.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let server = runtime.block_on(async {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embed"))
            .respond_with(ResponseTemplate::new(200).set_body_json(vec![vec![0.5; DIM]]))
            .mount(&server)
            .await;
        server
    });
    let config_path = root.path().join(".minsync/config.toml");
    let mut config = Config::load(&config_path).expect("load config");
    config.embedder.base_url = Some(server.uri());
    config.save(&config_path).expect("save config");

    let sync = run_json(root.path(), &["sync"]);
    assert_eq!(sync["files_added"], 1);
    assert_eq!(sync["files_deleted"], 1);
    assert_eq!(sync["sparse_encoded"], 1, "only the new chunk is encoded");

    let results = run_json(root.path(), &["query", "w30 w31", "--mode", "sparse"]);
    assert_eq!(results[0]["path"], "delta.md");
    let gone = run_json(
        root.path(),
        &["query", "w1 w2 w3 w4 w5", "--mode", "sparse"],
    );
    assert!(gone
        .as_array()
        .unwrap()
        .iter()
        .all(|hit| hit["path"] != "alpha.md"));
}
