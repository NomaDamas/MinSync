//! Live CLI end-to-end: a workspace initialised with `--embedder none` indexes
//! and searches with BM25 only. Every command runs with the model cache
//! pointed at an empty directory, offline, and no API keys, so any attempt to
//! load or call an embedder fails the test.

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

fn workspace() -> (tempfile::TempDir, tempfile::TempDir) {
    let root = tempfile::tempdir().expect("create workspace");
    let cache = tempfile::tempdir().expect("create empty model cache");
    std::fs::write(
        root.path().join("fruit.md"),
        "# Fruit\n\nThe zebrafish orchard grows quinces and pears.\n",
    )
    .expect("write fruit");
    std::fs::write(
        root.path().join("sea.md"),
        "# Sea\n\nTrawlers haul mackerel and herring at dawn.\n",
    )
    .expect("write sea");
    (root, cache)
}

#[test]
fn none_embedder_indexes_and_searches_with_bm25_only() {
    let (root, cache) = workspace();

    let init = run_json(
        &mut minsync(root.path(), cache.path()),
        &["init", "--embedder", "none"],
    );
    assert_eq!(init["embedder"]["id"], "none");
    assert_eq!(init["vectorstore"]["options"]["dimension"], 1);

    let sync = run_json(&mut minsync(root.path(), cache.path()), &["sync"]);
    assert_eq!(sync["files_processed"], 2);
    assert!(sync["chunks_added"].as_u64().unwrap() >= 2);
    assert_eq!(sync["embedding_api_calls"], 0);
    assert_eq!(sync["embedded_texts"], 0);
    assert_eq!(sync["partial"], false);

    // `--mode` defaults to bm25 when the workspace has no embedder.
    let results = run_json(
        &mut minsync(root.path(), cache.path()),
        &["query", "mackerel"],
    );
    assert_eq!(results[0]["path"], "sea.md");
    assert_eq!(results[0]["mode"], "bm25");
    assert_eq!(results[0]["bm25_rank"], 1);
    assert!(results[0].get("vector_rank").is_none());
    assert_eq!(results[0]["index_state"]["fts_indexed"], true);
    assert_eq!(results[0]["index_state"]["ann_indexed"], false);

    let explicit = run_json(
        &mut minsync(root.path(), cache.path()),
        &["query", "quinces", "--mode", "bm25"],
    );
    assert_eq!(explicit[0]["path"], "fruit.md");
    assert_eq!(explicit.as_array().unwrap().len(), 1);
}

#[test]
fn none_embedder_stays_incremental_and_removes_deleted_files() {
    let (root, cache) = workspace();
    run_json(
        &mut minsync(root.path(), cache.path()),
        &["init", "--embedder", "none"],
    );
    run_json(&mut minsync(root.path(), cache.path()), &["sync"]);

    let unchanged = run_json(&mut minsync(root.path(), cache.path()), &["sync"]);
    assert_eq!(unchanged["already_up_to_date"], true);

    std::fs::write(
        root.path().join("sea.md"),
        "# Sea\n\nTrawlers haul anchovy and herring at dawn.\n",
    )
    .expect("modify sea");
    std::fs::remove_file(root.path().join("fruit.md")).expect("delete fruit");
    let sync = run_json(&mut minsync(root.path(), cache.path()), &["sync"]);
    assert_eq!(sync["files_modified"], 1);
    assert_eq!(sync["files_deleted"], 1);
    assert_eq!(sync["embedded_texts"], 0);

    let gone = run_json(
        &mut minsync(root.path(), cache.path()),
        &["query", "quinces"],
    );
    assert_eq!(gone.as_array().unwrap().len(), 0);
    let stale = run_json(
        &mut minsync(root.path(), cache.path()),
        &["query", "mackerel"],
    );
    assert_eq!(stale.as_array().unwrap().len(), 0);
    let fresh = run_json(
        &mut minsync(root.path(), cache.path()),
        &["query", "anchovy"],
    );
    assert_eq!(fresh[0]["path"], "sea.md");

    let status = run_json(&mut minsync(root.path(), cache.path()), &["status"]);
    assert_eq!(status["embedder"], "none");
    assert_eq!(status["state"], "UpToDate");
}

#[test]
fn none_embedder_rejects_vector_and_hybrid_queries_with_guidance() {
    let (root, cache) = workspace();
    run_json(
        &mut minsync(root.path(), cache.path()),
        &["init", "--embedder", "none"],
    );
    run_json(&mut minsync(root.path(), cache.path()), &["sync"]);

    for mode in ["vector", "hybrid"] {
        let output = minsync(root.path(), cache.path())
            .args(["query", "mackerel", "--mode", mode])
            .output()
            .expect("run query");
        assert!(!output.status.success(), "{mode} must fail without vectors");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("--mode bm25"),
            "{mode} error must point at bm25, got: {stderr}"
        );
    }
}

#[test]
fn none_embedder_check_passes_without_a_model() {
    let (root, cache) = workspace();
    run_json(
        &mut minsync(root.path(), cache.path()),
        &["init", "--embedder", "none"],
    );
    let check = run_json(&mut minsync(root.path(), cache.path()), &["check"]);
    assert_eq!(check["embedder_ok"], true);
    assert_eq!(check["all_passed"], true);
}

#[test]
fn none_embedder_survives_enough_rows_to_trigger_ann_indexing() {
    let (root, cache) = workspace();
    // Past the default ANN build threshold (256 rows): a zero-vector HNSW
    // build would fail, so BM25-only stores must never attempt it.
    for i in 0..300 {
        std::fs::write(
            root.path().join(format!("note-{i}.md")),
            format!("note number {i} mentions token{i}xyz\n"),
        )
        .expect("write note");
    }
    run_json(
        &mut minsync(root.path(), cache.path()),
        &["init", "--embedder", "none"],
    );
    let sync = run_json(&mut minsync(root.path(), cache.path()), &["sync"]);
    assert_eq!(sync["partial"], false);
    assert!(sync["chunks_added"].as_u64().unwrap() >= 300);

    let hit = run_json(
        &mut minsync(root.path(), cache.path()),
        &["query", "token217xyz"],
    );
    assert_eq!(hit[0]["path"], "note-217.md");
    assert_eq!(hit[0]["index_state"]["ann_indexed"], false);
}
