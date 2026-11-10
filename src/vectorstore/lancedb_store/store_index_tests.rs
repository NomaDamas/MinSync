//! The sparse inverted index through the store: every answer must equal the
//! scan's, and the index must never answer for a table it does not describe.

use super::*;
use crate::sparse::SparseVector;
use tempfile::TempDir;

fn doc(id: &str, path: &str) -> Document {
    Document {
        id: id.to_string(),
        embedding: vec![1.0, 0.0, 0.0, 0.0],
        text: format!("text {id}"),
        source_id: "source-1".to_string(),
        path: path.to_string(),
        chunk_schema_id: "schema-1".to_string(),
        chunk_type: "text".to_string(),
        heading_path: String::new(),
        content_hash: format!("hash-{id}"),
        seen_token: "t".to_string(),
    }
}

fn sv(pairs: &[(u32, f32)]) -> SparseVector {
    SparseVector::from_pairs(pairs.iter().copied())
}

/// A store that indexes from the first row on, with a tiny memory budget so
/// the writer runs many passes.
fn indexed_store() -> (TempDir, LanceDbStore) {
    let dir = tempfile::tempdir().expect("create tempdir");
    let store = LanceDbStore::open_with_indexing(
        dir.path(),
        4,
        IndexingConfig {
            sparse_index_min_rows: 1,
            sparse_index_memory_postings: 16,
            ..IndexingConfig::default()
        },
    )
    .expect("create store");
    (dir, store)
}

fn plain_scan_store(dir: &std::path::Path) -> LanceDbStore {
    LanceDbStore::open_with_indexing(
        dir,
        4,
        IndexingConfig {
            sparse_inverted_index: false,
            ..IndexingConfig::default()
        },
    )
    .expect("open store without the index")
}

fn ranked(hits: &[crate::vectorstore::QueryHit]) -> Vec<(String, f32)> {
    hits.iter()
        .map(|hit| (hit.doc_id.clone(), hit.score))
        .collect()
}

/// `n` rows with pseudo-random vectors over a small vocabulary, encoded by `m`.
fn seed(store: &mut LanceDbStore, n: usize) -> Vec<(String, SparseVector)> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state >> 33
    };
    let docs: Vec<_> = (0..n)
        .map(|i| doc(&format!("d{i:05}"), &format!("f{}.txt", i % 7)))
        .collect();
    store.upsert(&docs).expect("seed rows");
    let rows: Vec<(String, SparseVector)> = docs
        .iter()
        .map(|doc| {
            let terms = next() % 9;
            let vector = sv(&(0..terms)
                .map(|_| ((next() % 40) as u32, (next() % 500 + 1) as f32 / 50.0))
                .collect::<Vec<_>>());
            (doc.id.clone(), vector)
        })
        .collect();
    store.set_sparse("m", &rows).expect("set sparse");
    rows
}

#[test]
fn an_indexed_query_returns_exactly_what_the_scan_returns() {
    let (dir, mut store) = indexed_store();
    seed(&mut store, 400);
    store.flush().expect("flush builds the index");
    assert!(
        store.sparse_index_ready().expect("state"),
        "flush must leave an index for a table above the minimum"
    );

    let scan = plain_scan_store(dir.path());
    for query in [
        sv(&[(1, 1.0)]),
        sv(&[(3, 0.5), (9, 2.0), (30, 1.0)]),
        sv(&[(39, 1.0), (2, 1.0), (7, 1.0), (12, 1.0)]),
    ] {
        for k in [1, 5, 50, 1000] {
            let via_index = store.query_sparse("m", &query, None, k).expect("indexed");
            let via_scan = scan.query_sparse("m", &query, None, k).expect("scan");
            assert_eq!(ranked(&via_index), ranked(&via_scan), "k {k} {query:?}");
            let paths: Vec<_> = via_index.iter().map(|h| h.path.clone()).collect();
            let want: Vec<_> = via_scan.iter().map(|h| h.path.clone()).collect();
            assert_eq!(paths, want, "metadata comes from the table either way");
        }
    }
}

#[test]
fn a_retry_removes_interrupted_index_builds_without_touching_other_files() {
    let (dir, mut store) = indexed_store();
    seed(&mut store, 40);
    let orphan = dir.path().join(".minsync-sparse-index-interrupted");
    let unrelated = dir.path().join("other.tmp");
    std::fs::write(&orphan, b"partial index").unwrap();
    std::fs::write(&unrelated, b"unrelated").unwrap();
    store.flush().unwrap();
    assert!(!orphan.exists());
    assert_eq!(std::fs::read(&unrelated).unwrap(), b"unrelated");
    assert!(store.sparse_index_ready().unwrap());
}

/// Sixty of 240 rows have the very same vector, so they tie at the k-th score
/// for every `k` up to 60: which of them are in the answer is decided by id,
/// and only an exact rescore can do that (the index stores brackets, not
/// weights). Whether a given `k` is answered through the index or, because
/// the candidates exceed a quarter of the table, by the scan, the ranking
/// must be the scan's.
#[test]
fn rows_tied_at_the_kth_score_are_ordered_by_id() {
    let (dir, mut store) = indexed_store();
    let docs: Vec<_> = (0..240)
        .map(|i| doc(&format!("d{i:05}"), &format!("f{}.txt", i % 5)))
        .collect();
    store.upsert(&docs).expect("rows");
    let rows: Vec<(String, SparseVector)> = docs
        .iter()
        .enumerate()
        .map(|(i, doc)| {
            let vector = if i % 4 == 0 {
                sv(&[(1, 2.0), (2, 1.0)])
            } else {
                sv(&[(1, 0.5), (3, 1.0 + (i % 11) as f32)])
            };
            (doc.id.clone(), vector)
        })
        .collect();
    store.set_sparse("m", &rows).expect("sparse");
    store.flush().expect("flush builds the index");
    assert!(store.sparse_index_ready().expect("state"));

    let scan = plain_scan_store(dir.path());
    let query = sv(&[(1, 1.0), (2, 1.0)]);
    for k in [1usize, 7, 30, 59, 60, 61, 100, 500] {
        let via_index = store.query_sparse("m", &query, None, k).expect("indexed");
        let via_scan = scan.query_sparse("m", &query, None, k).expect("scan");
        assert_eq!(ranked(&via_index), ranked(&via_scan), "k {k}");
    }
    let top = store.query_sparse("m", &query, None, 3).expect("top");
    let ids: Vec<_> = top.iter().map(|hit| hit.doc_id.as_str()).collect();
    assert_eq!(
        ids,
        ["d00000", "d00004", "d00008"],
        "ties go to the smaller id"
    );
}

/// A `k` as large as the table makes nearly every row a candidate; reading
/// them back by id would cost more than the scan it is meant to replace, and
/// the answer is the same either way.
#[test]
fn a_k_as_large_as_the_table_is_answered_by_the_scan_with_the_same_ranking() {
    let (dir, mut store) = indexed_store();
    seed(&mut store, 300);
    store.flush().expect("flush");
    let scan = plain_scan_store(dir.path());
    let query = sv(&[(3, 0.5), (9, 2.0), (30, 1.0), (12, 1.0)]);

    for k in [150usize, 300, 5000] {
        let via_index = store.query_sparse("m", &query, None, k).expect("indexed");
        let via_scan = scan.query_sparse("m", &query, None, k).expect("scan");
        assert_eq!(ranked(&via_index), ranked(&via_scan), "k {k}");
    }
}

#[test]
fn a_filter_is_applied_exactly_whether_or_not_an_index_exists() {
    let (dir, mut store) = indexed_store();
    seed(&mut store, 300);
    store.flush().expect("flush");
    let scan = plain_scan_store(dir.path());

    for filter in [
        Filter::Eq("path".into(), "f3.txt".into()),
        Filter::And(vec![
            Filter::Eq("source_id".into(), "source-1".into()),
            Filter::NotIn("path".into(), vec!["f1.txt".into(), "f2.txt".into()]),
        ]),
        Filter::In("path".into(), vec!["nothing.txt".into()]),
    ] {
        let query = sv(&[(3, 1.0), (9, 1.0), (21, 1.0)]);
        let a = store
            .query_sparse("m", &query, Some(&filter), 25)
            .expect("indexed");
        let b = scan
            .query_sparse("m", &query, Some(&filter), 25)
            .expect("scan");
        assert_eq!(ranked(&a), ranked(&b), "{filter:?}");
    }
}

#[test]
fn a_write_after_the_build_is_never_missed() {
    let (_dir, mut store) = indexed_store();
    seed(&mut store, 300);
    store.flush().expect("flush builds the index");
    assert!(store.sparse_index_ready().expect("state"));

    // A row the index has never seen, with a score no other row can reach.
    store.upsert(&[doc("late", "late.txt")]).expect("upsert");
    store
        .set_sparse("m", &[("late".to_string(), sv(&[(5, 1000.0)]))])
        .expect("set sparse");

    let hits = store
        .query_sparse("m", &sv(&[(5, 1.0)]), None, 3)
        .expect("query after write");
    assert_eq!(hits[0].doc_id, "late", "the stale index must not answer");
    assert_eq!(hits[0].score, 1000.0);
    assert!(
        !store.sparse_index_ready().expect("state"),
        "an index of an older version is not ready"
    );

    // The next flush describes the new version again.
    store.flush().expect("flush rebuilds");
    assert!(store.sparse_index_ready().expect("state"));
    let again = store
        .query_sparse("m", &sv(&[(5, 1.0)]), None, 3)
        .expect("query");
    assert_eq!(ranked(&again), ranked(&hits));
}

#[test]
fn deleting_and_replacing_rows_changes_the_answer_at_once() {
    let (_dir, mut store) = indexed_store();
    seed(&mut store, 300);
    store.flush().expect("flush");
    let query = sv(&[(2, 1.0), (8, 1.0)]);
    let before = store.query_sparse("m", &query, None, 5).expect("before");
    let top = before[0].doc_id.clone();

    store
        .delete_by_filter(&Filter::Eq("id".into(), top.clone()))
        .expect("delete");

    let after = store.query_sparse("m", &query, None, 5).expect("after");
    assert!(after.iter().all(|hit| hit.doc_id != top), "{after:?}");
}

#[test]
fn a_damaged_index_file_falls_back_to_the_scan() {
    let (dir, mut store) = indexed_store();
    seed(&mut store, 300);
    store.flush().expect("flush");
    let query = sv(&[(3, 1.0), (9, 1.0)]);
    let want = ranked(&store.query_sparse("m", &query, None, 10).expect("good"));

    let file = sparse_index::index_path(dir.path());
    let bytes = std::fs::read(&file).expect("the index file exists");
    std::fs::write(&file, &bytes[..bytes.len() / 2]).expect("truncate it");

    let got = ranked(&store.query_sparse("m", &query, None, 10).expect("fallback"));
    assert_eq!(got, want);
}

#[test]
fn another_model_never_uses_the_index_of_this_one() {
    let (_dir, mut store) = indexed_store();
    seed(&mut store, 300);
    store.flush().expect("flush");

    let hits = store
        .query_sparse("other-model", &sv(&[(3, 1.0)]), None, 5)
        .expect("query");

    assert!(hits.is_empty());
}

#[test]
fn a_table_below_the_minimum_keeps_no_index() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut store = LanceDbStore::open_with_indexing(
        dir.path(),
        4,
        IndexingConfig {
            sparse_index_min_rows: 1000,
            ..IndexingConfig::default()
        },
    )
    .expect("store");
    seed(&mut store, 50);
    store.flush().expect("flush");

    assert!(!store.sparse_index_ready().expect("state"));
    assert!(!store
        .query_sparse("m", &sv(&[(3, 1.0)]), None, 5)
        .expect("scan")
        .is_empty());
}

#[test]
fn turning_the_option_off_removes_the_index_and_scans() {
    let (dir, mut store) = indexed_store();
    seed(&mut store, 300);
    store.flush().expect("flush");
    assert!(store.sparse_index_ready().expect("state"));
    drop(store);

    let mut store = plain_scan_store(dir.path());
    store.flush().expect("flush with the option off");

    assert!(!sparse_index::index_path(dir.path()).exists());
    assert!(!store
        .query_sparse("m", &sv(&[(3, 1.0)]), None, 5)
        .expect("scan")
        .is_empty());
}

#[test]
fn no_index_is_built_while_rows_still_wait_for_their_sparse_vector() {
    let (_dir, mut store) = indexed_store();
    let rows = seed(&mut store, 300);
    // A sync flushes after the dense writes and again after the sparse phase.
    // Rows added in between are not encoded yet: indexing now would build a
    // file the second flush throws away.
    store
        .upsert(&[doc("fresh", "fresh.txt")])
        .expect("upsert a row with no sparse vector");
    store.flush().expect("flush before the sparse phase");
    assert!(
        !store.sparse_index_ready().expect("state"),
        "an index over a half-encoded table is wasted work"
    );

    store
        .set_sparse("m", &[("fresh".to_string(), sv(&[(1, 1.0)]))])
        .expect("encode it");
    store.flush().expect("flush after the sparse phase");
    assert!(store.sparse_index_ready().expect("state"));
    assert_eq!(store.count_missing_sparse("m").expect("missing"), 0);
    assert_eq!(rows.len(), 300);
}

#[test]
fn a_reader_opened_beside_a_writer_never_returns_a_newer_or_older_answer() {
    let (dir, mut writer) = indexed_store();
    seed(&mut writer, 300);
    writer.flush().expect("flush");
    let reader =
        LanceDbStore::open_readonly(dir.path(), 4, IndexingConfig::default()).expect("reader");
    let query = sv(&[(3, 1.0), (9, 1.0)]);
    let pinned = ranked(&reader.query_sparse("m", &query, None, 10).expect("reader"));

    writer
        .set_sparse("m", &[("d00001".to_string(), sv(&[(3, 900.0)]))])
        .expect("write after the reader opened");
    writer.flush().expect("flush");

    // The reader pinned its version at open: the new sidecar does not match,
    // so it scans the old table rather than mixing versions.
    let again = ranked(&reader.query_sparse("m", &query, None, 10).expect("reader"));
    assert_eq!(again, pinned);
    let fresh =
        LanceDbStore::open_readonly(dir.path(), 4, IndexingConfig::default()).expect("new reader");
    let newest = fresh.query_sparse("m", &query, None, 10).expect("fresh");
    assert_eq!(newest[0].doc_id, "d00001");
}
