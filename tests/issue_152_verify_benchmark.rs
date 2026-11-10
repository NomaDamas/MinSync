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
                text: format!("alpha{i} beta{i} gamma{i} delta{i}"),
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

async fn time_verify(root: &tempfile::TempDir, chunker: &RecursiveChunker) -> usize {
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
    let flagged = result.embedding_integrity.flagged_pairs.len();
    println!(
        "  verify_ms={} flagged_pairs={flagged}",
        started.elapsed().as_millis()
    );
    flagged
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

/// Every this many rows, one row's vector is a near copy (cosine ~0.9998)
/// of the row before it, over unrelated text: a pair verify must flag.
const TWIN_EVERY: usize = 1_000;
const DENSE_DIMENSION: usize = 768;

/// Uniform noise in [-1, 1) from a per-row xorshift stream, around a shared
/// component like the one sentence embeddings have.
fn dense_vector(i: usize) -> Vec<f32> {
    let base = i - usize::from(i % TWIN_EVERY == 1);
    let mut state = (base as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    };
    let mut vector: Vec<f32> = (0..DENSE_DIMENSION).map(|_| 0.3 + next()).collect();
    if base != i {
        let mut state = (i as u64).wrapping_mul(0xD1B5_4A32_D192_ED03) | 1;
        for value in &mut vector {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *value += 0.02 * ((state >> 40) as f32 / (1u64 << 23) as f32 - 1.0);
        }
    }
    vector
}

#[tokio::test]
#[ignore = "performance measurement; run with --release --ignored --nocapture"]
async fn benchmark_issue_152_verify_dense() {
    for rows in row_counts() {
        let (root, chunker) = workspace("tei:google/embeddinggemma-300m", rows, dense_vector);
        println!("dense dim={DENSE_DIMENSION} rows={rows}");
        let flagged = time_verify(&root, &chunker).await;
        assert_eq!(
            flagged,
            rows.div_ceil(TWIN_EVERY) - usize::from(rows % TWIN_EVERY == 1)
        );
    }
}
