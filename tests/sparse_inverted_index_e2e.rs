//! Live CLI end-to-end: `minsync sync` leaves a sparse inverted index, `query`
//! answers from it with exactly the ranking a scan gives, and any change to the
//! table makes the index step aside until the next sync. The sparse model is
//! the tiny checked-in checkpoint, run through the same native code path as
//! KURE-Sparse-nano; every command runs offline with an empty model cache.

mod common;

use assert_cmd::Command;
use minsync::config::Config;
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

fn run_json(root: &Path, cache: &Path, args: &[&str]) -> serde_json::Value {
    let output = minsync(root, cache)
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

/// `(doc id, score)` of every hit, in rank order.
fn ranked(results: &serde_json::Value) -> Vec<(String, f64)> {
    results
        .as_array()
        .expect("results array")
        .iter()
        .map(|hit| {
            (
                hit["doc_id"].as_str().expect("doc_id").to_string(),
                hit["score"].as_f64().expect("score"),
            )
        })
        .collect()
}

fn set_index_options(root: &Path, enabled: bool, min_rows: i64) {
    let path = root.join(".minsync/config.toml");
    let mut config = Config::load(&path).expect("load config");
    let mut options = match std::mem::replace(
        &mut config.vectorstore.options,
        toml::Value::Table(Default::default()),
    ) {
        toml::Value::Table(table) => table,
        _ => Default::default(),
    };
    options.insert(
        "sparse_inverted_index".into(),
        toml::Value::Boolean(enabled),
    );
    options.insert(
        "sparse_index_min_rows".into(),
        toml::Value::Integer(min_rows),
    );
    config.vectorstore.options = toml::Value::Table(options);
    config.save(&path).expect("save config");
}

/// 60 notes over the tiny vocabulary `w0`..`w61`: each is a distinct run of
/// words, so every query has a graded set of hits.
fn write_notes(root: &Path, count: usize) {
    for i in 0..count {
        let words: Vec<String> = (0..6)
            .map(|j| format!("w{}", (i * 3 + j * 5) % 62))
            .collect();
        std::fs::write(root.join(format!("note-{i:03}.md")), words.join(" ") + "\n")
            .expect("write note");
    }
}

fn workspace(count: usize) -> (tempfile::TempDir, tempfile::TempDir) {
    let root = tempfile::tempdir().expect("workspace");
    let cache = tempfile::tempdir().expect("empty model cache");
    write_notes(root.path(), count);
    run_json(root.path(), cache.path(), &["init", "--embedder", "none"]);
    common::use_tiny_sparse_in(root.path());
    // Index from the first row on, so a small corpus exercises it.
    set_index_options(root.path(), true, 1);
    (root, cache)
}

const QUERIES: [&str; 4] = [
    "w0 w5 w10",
    "w3 w8 w13 w18",
    "w61 w1",
    "w20 w21 w22 w23 w24",
];

#[test]
fn query_answers_from_the_index_with_exactly_the_scans_ranking() {
    let (root, cache) = workspace(60);
    run_json(root.path(), cache.path(), &["sync"]);

    let first = run_json(
        root.path(),
        cache.path(),
        &["query", QUERIES[0], "-k", "10"],
    );
    assert_eq!(
        first[0]["index_state"]["sparse_indexed"], true,
        "a finished sync leaves an index that describes the table: {first}"
    );
    assert!(root.path().join(".minsync/store/sparse-index.inv").exists());

    let mut via_index = Vec::new();
    for query in QUERIES {
        via_index.push(ranked(&run_json(
            root.path(),
            cache.path(),
            &["query", query, "-k", "10"],
        )));
    }

    set_index_options(root.path(), false, 1);
    let scan = run_json(
        root.path(),
        cache.path(),
        &["query", QUERIES[0], "-k", "10"],
    );
    assert_eq!(scan[0]["index_state"]["sparse_indexed"], false);
    for (query, from_index) in QUERIES.iter().zip(&via_index) {
        let from_scan = ranked(&run_json(
            root.path(),
            cache.path(),
            &["query", query, "-k", "10"],
        ));
        assert!(
            !from_scan.is_empty(),
            "{query}: the oracle must find something"
        );
        assert_eq!(*from_index, from_scan, "{query}: index and scan differ");
    }
}

#[test]
fn an_edit_is_found_at_once_and_the_index_is_current_again_after_the_next_sync() {
    let (root, cache) = workspace(60);
    run_json(root.path(), cache.path(), &["sync"]);

    // Rewrite one note and add another: neither is in the index yet.
    std::fs::write(root.path().join("note-007.md"), "w40 w41 w42 w43 w44 w45\n").unwrap();
    std::fs::write(root.path().join("note-new.md"), "w50 w51 w52 w53 w54 w55\n").unwrap();
    run_json(root.path(), cache.path(), &["sync"]);

    let probes = [
        "w50 w51 w52 w53 w54 w55",
        "w40 w41 w42 w43 w44 w45",
        QUERIES[0],
    ];
    let mut via_index = Vec::new();
    for probe in probes {
        let hits = run_json(root.path(), cache.path(), &["query", probe, "-k", "60"]);
        assert_eq!(
            hits[0]["index_state"]["sparse_indexed"], true,
            "a finished sync leaves a current index: {hits}"
        );
        via_index.push((ranked(&hits), hits));
    }

    // The edited and the added note are in the answers with their new text.
    let paths_of = |hits: &serde_json::Value| -> Vec<String> {
        hits.as_array()
            .unwrap()
            .iter()
            .map(|hit| hit["path"].as_str().unwrap().to_string())
            .collect()
    };
    assert!(
        paths_of(&via_index[0].1).contains(&"note-new.md".to_string()),
        "the added note must be searchable: {}",
        via_index[0].1
    );
    assert!(
        paths_of(&via_index[1].1).contains(&"note-007.md".to_string()),
        "the edited note must be searchable: {}",
        via_index[1].1
    );
    // The old text of the edited note (w21 w26 w31 w36 w41 w46) is gone from
    // every answer: no stale row survives in the index.
    for (_, hits) in &via_index {
        for hit in hits.as_array().unwrap() {
            assert!(
                !hit["text"]
                    .as_str()
                    .unwrap()
                    .contains("w21 w26 w31 w36 w41 w46"),
                "stale text of note-007.md came back: {hit}"
            );
        }
    }

    // And the index is exactly what a scan gives, for the edited corpus.
    set_index_options(root.path(), false, 1);
    for (probe, (from_index, _)) in probes.iter().zip(&via_index) {
        let scan = run_json(root.path(), cache.path(), &["query", probe, "-k", "60"]);
        assert_eq!(scan[0]["index_state"]["sparse_indexed"], false);
        assert_eq!(*from_index, ranked(&scan), "{probe}: index and scan differ");
    }
}

#[test]
fn deleting_a_file_removes_it_from_the_index_answers() {
    let (root, cache) = workspace(60);
    run_json(root.path(), cache.path(), &["sync"]);
    let before = run_json(root.path(), cache.path(), &["query", QUERIES[0], "-k", "5"]);
    let top = before[0]["path"].as_str().unwrap().to_string();

    std::fs::remove_file(root.path().join(&top)).unwrap();
    run_json(root.path(), cache.path(), &["sync"]);

    let after = run_json(root.path(), cache.path(), &["query", QUERIES[0], "-k", "5"]);
    assert_eq!(after[0]["index_state"]["sparse_indexed"], true);
    assert!(
        after
            .as_array()
            .unwrap()
            .iter()
            .all(|hit| hit["path"] != top.as_str()),
        "{top} was deleted: {after}"
    );
}

#[test]
fn a_corrupt_index_file_costs_nothing_but_the_scan() {
    let (root, cache) = workspace(60);
    run_json(root.path(), cache.path(), &["sync"]);
    let want = ranked(&run_json(
        root.path(),
        cache.path(),
        &["query", QUERIES[1], "-k", "10"],
    ));

    let file = root.path().join(".minsync/store/sparse-index.inv");
    let bytes = std::fs::read(&file).expect("index file");
    std::fs::write(&file, &bytes[..bytes.len() / 3]).expect("truncate");

    let got = run_json(
        root.path(),
        cache.path(),
        &["query", QUERIES[1], "-k", "10"],
    );
    assert_eq!(got[0]["index_state"]["sparse_indexed"], false);
    assert_eq!(ranked(&got), want);

    // `optimize` rebuilds it.
    run_json(root.path(), cache.path(), &["optimize"]);
    let again = run_json(
        root.path(),
        cache.path(),
        &["query", QUERIES[1], "-k", "10"],
    );
    assert_eq!(again[0]["index_state"]["sparse_indexed"], true);
    assert_eq!(ranked(&again), want);
}

#[test]
fn a_workspace_below_the_minimum_never_gets_an_index() {
    let (root, cache) = workspace(30);
    set_index_options(root.path(), true, 10_000);
    run_json(root.path(), cache.path(), &["sync"]);

    let hits = run_json(root.path(), cache.path(), &["query", QUERIES[0], "-k", "5"]);

    assert_eq!(hits[0]["index_state"]["sparse_indexed"], false);
    assert!(!root.path().join(".minsync/store/sparse-index.inv").exists());
    assert!(!hits.as_array().unwrap().is_empty());
}
