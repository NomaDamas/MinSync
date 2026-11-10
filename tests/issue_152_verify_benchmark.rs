//! `minsync verify` wall time on large LanceDB stores (issue #152).
//!
//! Run with `cargo test --release --test issue_152_verify_benchmark -- --ignored --nocapture`.
//! `MINSYNC_BENCH_ROWS` (comma-separated row counts) overrides the sizes.

use minsync::chunker::recursive::RecursiveChunker;
use minsync::chunker::Chunker;
use minsync::config::Config;
use minsync::sync::MinSync;
use minsync::vectorstore::{create_vectorstore, Document, VectorStore};
use minsync::verify::verify;
use std::time::Instant;

const DEFAULT_ROWS: [usize; 3] = [10_000, 20_000, 40_000];
const UPSERT_BATCH: usize = 20_000;

fn row_counts() -> Vec<usize> {
    std::env::var("MINSYNC_BENCH_ROWS")
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(|count| count.trim().parse().expect("row count"))
                .collect()
        })
        .unwrap_or_else(|| DEFAULT_ROWS.to_vec())
}

/// A workspace initialized for `embedder` whose store holds `rows` chunks
/// with the vectors `vector(i)` returns. The files themselves are not
/// written: the integrity check only reads the store.
fn workspace(
    embedder: &str,
    rows: usize,
    vector: impl Fn(usize) -> Vec<f32>,
) -> (tempfile::TempDir, RecursiveChunker) {
    let root = tempfile::tempdir().expect("create workspace");
    MinSync::new(root.path().to_path_buf())
        .init(false, embedder, "recursive")
        .expect("init succeeds");
    let chunker = RecursiveChunker::new(1024);
    let (config, mut store) = open_store(&root);
    for start in (0..rows).step_by(UPSERT_BATCH) {
        let docs: Vec<Document> = (start..rows.min(start + UPSERT_BATCH))
            .map(|i| Document {
                id: format!("chunk-{i:08}"),
                embedding: vector(i),
                text: format!("row {i} talks about topic {} and item {}", i % 97, i / 97),
                source_id: config.source_id.clone(),
                path: format!("file-{}.txt", i / 16),
                chunk_schema_id: chunker.schema_id().to_string(),
                chunk_type: "text".to_string(),
                heading_path: String::new(),
                content_hash: format!("hash-{i}"),
                seen_token: "token".to_string(),
            })
            .collect();
        store.insert_new(&docs).expect("insert rows");
    }
    store.flush().expect("flush");
    (root, chunker)
}

fn open_store(root: &tempfile::TempDir) -> (Config, Box<dyn VectorStore>) {
    let minsync_dir = root.path().join(".minsync");
    let config = Config::load(&minsync_dir.join("config.toml")).expect("load config");
    let store = create_vectorstore(&config, &minsync_dir.join(&config.collection.path))
        .expect("open store");
    (config, store)
}

async fn time_verify(root: &tempfile::TempDir, chunker: &RecursiveChunker) {
    let (_config, mut store) = open_store(root);
    let started = Instant::now();
    let result = verify(
        &root.path().join(".minsync"),
        root.path(),
        chunker,
        store.as_mut(),
        false,
        Some(0),
    )
    .await
    .expect("verify succeeds");
    println!(
        "  verify_ms={} flagged_pairs={}",
        started.elapsed().as_millis(),
        result.embedding_integrity.flagged_pairs.len()
    );
}

#[tokio::test]
#[ignore = "performance measurement; run with --release --ignored --nocapture"]
async fn benchmark_issue_152_verify_sparse_only() {
    for rows in row_counts() {
        let (root, chunker) = workspace("none", rows, |_| vec![0.0]);
        println!("sparse_only rows={rows}");
        time_verify(&root, &chunker).await;
    }
}
