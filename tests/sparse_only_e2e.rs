//! Live CLI end-to-end: a workspace initialised with `--embedder none` indexes
//! and searches with the sparse (SPLADE) encoder only. Every command runs with
//! the model cache pointed at an empty directory, offline, and no API keys, so
//! any attempt to load a dense model or download weights fails the test. The
//! sparse model is the tiny checked-in checkpoint (`tests/fixtures/tiny_splade`),
//! which runs through the same native code path as KURE-Sparse-nano.
//!
//! The tiny vocabulary knows `w0`..`w61`; any other word is `<unk>`. Documents
//! are therefore written as `w` words, and search results are compared with a
//! brute-force inner-product ranking computed in-process from the same model.

mod common;

use assert_cmd::Command;
use std::path::Path;

fn minsync(root: &Path, cache: &Path) -> Command {
    let mut command = Command::cargo_bin("minsync").expect("find minsync binary");
    command
        .current_dir(root)
        .env_remove("OPENAI_API_KEY")
        .env("MINSYNC_SERVER", "off")
        .env("HF_HOME", cache)
        .env("HF_HUB_OFFLINE", "1")
        .env("FASTEMBED_CACHE_DIR", cache);
    command
}

fn run_json(command: &mut Command, args: &[&str]) -> serde_json::Value {
    let output = command
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
    serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "`minsync {}` printed invalid JSON ({error}): {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stdout)
        )
    })
}

/// File name -> text. Each is one chunk.
const DOCS: [(&str, &str); 4] = [
    ("fruit.md", "w1 w2 w3 w4 w5 w6"),
    ("sea.md", "w10 w11 w12 w13 w14"),
    ("hill.md", "w20 w21 w22 w23 w24 w25 w26"),
    ("mix.md", "w1 w10 w20 w30 w40"),
];

fn workspace() -> (tempfile::TempDir, tempfile::TempDir) {
    let root = tempfile::tempdir().expect("create workspace");
    let cache = tempfile::tempdir().expect("create empty model cache");
    for (name, text) in DOCS {
        std::fs::write(root.path().join(name), format!("{text}\n")).expect("write doc");
    }
    (root, cache)
}

fn init_sparse_only(root: &Path, cache: &Path) -> serde_json::Value {
    let init = run_json(&mut minsync(root, cache), &["init", "--embedder", "none"]);
    common::use_tiny_sparse_in(root);
    init
}

/// Paths the CLI returned, in rank order.
fn paths(results: &serde_json::Value) -> Vec<String> {
    results
        .as_array()
        .expect("results array")
        .iter()
        .map(|hit| hit["path"].as_str().expect("path").to_string())
        .collect()
}

/// The ranking the model itself implies for `query` over `corpus`.
fn ranking_over(corpus: &[(&str, &str)], query: &str) -> Vec<String> {
    let docs: Vec<&str> = corpus.iter().map(|(_, text)| *text).collect();
    let doc_vectors = common::encode_with_tiny(&docs, false);
    let query_vector = common::encode_with_tiny(&[query], true).remove(0);
    common::brute_force_ranking(&doc_vectors, &query_vector)
        .into_iter()
        .map(|i| corpus[i].0.to_string())
        .collect()
}

fn expected_paths(query: &str) -> Vec<String> {
    ranking_over(&DOCS, query)
}

#[test]
fn none_embedder_indexes_and_searches_with_sparse_only() {
    let (root, cache) = workspace();

    let init = init_sparse_only(root.path(), cache.path());
    assert_eq!(init["embedder"]["id"], "none");
    assert_eq!(init["vectorstore"]["options"]["dimension"], 1);
    assert_eq!(init["sparse"]["id"], "native:nlpai-lab/KURE-Sparse-nano");
    assert!(init.get("lexical").is_none(), "no BM25 section is written");

    let sync = run_json(&mut minsync(root.path(), cache.path()), &["sync"]);
    assert_eq!(sync["files_processed"], 4);
    assert_eq!(sync["chunks_added"], 4);
    assert_eq!(sync["embedding_api_calls"], 0);
    assert_eq!(sync["embedded_texts"], 0);
    assert_eq!(sync["sparse_encoded"], 4);
    assert_eq!(sync["partial"], false);

    let cursor: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root.path().join(".minsync/cursor.json")).expect("read cursor"),
    )
    .expect("cursor json");
    assert_eq!(cursor["sparse_id"], common::tiny_model_id());

    // `--mode` defaults to sparse when the workspace has no dense embedder.
    let query = "w1 w2 w3";
    let results = run_json(&mut minsync(root.path(), cache.path()), &["query", query]);
    let expected = expected_paths(query);
    assert!(!expected.is_empty(), "the oracle must find something");
    assert_eq!(paths(&results), expected, "CLI ranking equals brute force");
    assert_eq!(results[0]["mode"], "sparse");
    assert_eq!(results[0]["sparse_rank"], 1);
    assert!(results[0].get("vector_rank").is_none());
    assert!(results[0].get("bm25_rank").is_none());
    assert!(results[0]["index_state"].get("fts_indexed").is_none());
    assert_eq!(results[0]["index_state"]["ann_indexed"], false);

    let explicit = run_json(
        &mut minsync(root.path(), cache.path()),
        &["query", "w10 w11", "--mode", "sparse", "-k", "2"],
    );
    assert_eq!(
        explicit.as_array().unwrap().len(),
        2.min(expected_paths("w10 w11").len())
    );
    assert_eq!(
        paths(&explicit),
        expected_paths("w10 w11")[..explicit.as_array().unwrap().len()]
    );
}

#[test]
fn none_embedder_stays_incremental_and_removes_deleted_files() {
    let (root, cache) = workspace();
    init_sparse_only(root.path(), cache.path());
    run_json(&mut minsync(root.path(), cache.path()), &["sync"]);

    let unchanged = run_json(&mut minsync(root.path(), cache.path()), &["sync"]);
    assert_eq!(unchanged["already_up_to_date"], true);
    assert_eq!(unchanged["sparse_encoded"], 0);

    std::fs::write(root.path().join("sea.md"), "w50 w51 w52 w53\n").expect("modify sea");
    std::fs::remove_file(root.path().join("fruit.md")).expect("delete fruit");
    let sync = run_json(&mut minsync(root.path(), cache.path()), &["sync"]);
    assert_eq!(sync["files_modified"], 1);
    assert_eq!(sync["files_deleted"], 1);
    assert_eq!(sync["embedded_texts"], 0);
    assert_eq!(
        sync["sparse_encoded"], 1,
        "only the modified chunk is encoded"
    );

    // After the edit the corpus is sea.md (new text), hill.md and mix.md.
    let corpus = [
        ("sea.md", "w50 w51 w52 w53"),
        ("hill.md", DOCS[2].1),
        ("mix.md", DOCS[3].1),
    ];
    for query in [
        "w1 w2 w3 w4 w5 w6",
        "w50 w51 w52 w53",
        "w10 w11 w12 w13 w14",
    ] {
        let results = run_json(&mut minsync(root.path(), cache.path()), &["query", query]);
        let got = paths(&results);
        assert_eq!(
            got,
            ranking_over(&corpus, query),
            "query {query:?}: {results}"
        );
        assert!(
            !got.contains(&"fruit.md".to_string()),
            "a deleted file must not be returned: {results}"
        );
        for hit in results.as_array().unwrap() {
            let text = hit["text"].as_str().unwrap();
            assert!(
                !text.contains("w10 w11"),
                "the old text of a modified file must be gone: {results}"
            );
        }
    }

    let status = run_json(&mut minsync(root.path(), cache.path()), &["status"]);
    assert_eq!(status["embedder"], "none");
    assert_eq!(status["state"], "UpToDate");
}

#[test]
fn none_embedder_rejects_vector_and_hybrid_queries_with_guidance() {
    let (root, cache) = workspace();
    init_sparse_only(root.path(), cache.path());
    run_json(&mut minsync(root.path(), cache.path()), &["sync"]);

    for mode in ["vector", "hybrid"] {
        let output = minsync(root.path(), cache.path())
            .args(["query", "w1", "--mode", mode])
            .output()
            .expect("run query");
        assert!(
            !output.status.success(),
            "{mode} must fail without dense vectors"
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("--mode sparse"),
            "{mode} error must point at sparse, got: {stderr}"
        );
    }
}

#[test]
fn none_embedder_check_probes_only_the_sparse_encoder() {
    let (root, cache) = workspace();
    init_sparse_only(root.path(), cache.path());
    let check = run_json(&mut minsync(root.path(), cache.path()), &["check"]);
    assert_eq!(check["embedder_ok"], true);
    assert_eq!(check["sparse_ok"], true);
    assert_eq!(check["all_passed"], true);
}

#[test]
fn none_embedder_survives_enough_rows_to_trigger_ann_indexing() {
    let (root, cache) = workspace();
    // Past the default ANN build threshold (256 rows): a zero-vector HNSW
    // build would fail, so sparse-only stores must never attempt it.
    for i in 0..300 {
        // A distinctive term per note: the pair (w{a} w{b}) identifies it.
        std::fs::write(
            root.path().join(format!("note-{i:03}.md")),
            format!("w{} w{} w{}\n", i % 50, (i / 50) + 50, (i % 7) + 1),
        )
        .expect("write note");
    }
    init_sparse_only(root.path(), cache.path());
    let sync = run_json(&mut minsync(root.path(), cache.path()), &["sync"]);
    assert_eq!(sync["partial"], false);
    assert!(sync["chunks_added"].as_u64().unwrap() >= 300);
    assert_eq!(
        sync["sparse_encoded"].as_u64().unwrap() + sync["sparse_reused"].as_u64().unwrap(),
        sync["chunks_added"].as_u64().unwrap()
    );

    let hit = run_json(
        &mut minsync(root.path(), cache.path()),
        &["query", "w17 w52 w4"],
    );
    assert!(!hit.as_array().unwrap().is_empty());
    assert_eq!(hit[0]["index_state"]["ann_indexed"], false);
}

#[test]
fn a_sparse_query_before_any_sync_says_to_sync_first() {
    let (root, cache) = workspace();
    init_sparse_only(root.path(), cache.path());
    let output = minsync(root.path(), cache.path())
        .args(["query", "w1"])
        .output()
        .expect("run query");
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("minsync sync"));
}
