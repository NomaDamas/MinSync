//! Sparse search through the real LanceDB store: scan vs. inverted index.
//!
//! Uses the vectors written by `sparse_inverted_bench encode`, so no model is
//! needed here. Everything goes through `LanceDbStore`, i.e. the code path of
//! `minsync sync` (writes, `flush`) and `minsync query` (read-only open).
//!
//! ```text
//! sparse_store_bench load  <encoded> <docs.jsonl> <store dir> [--scale N] [--no-index]
//! sparse_store_bench query <store dir> <encoded> [--k K] [--passes P] [--scan]
//! sparse_store_bench cold  <store dir> <encoded> [--k K] [--runs R] [--scan]
//! ```
//!
//! `query` times `query_sparse` on one open store (warm); `cold` opens the
//! store from scratch before every query, the way each `minsync query`
//! process does (model loading is not part of it: the query vector is given).
//! Every `query` run first checks that the index and the scan agree.

use minsync::config::SPARSE_ONLY_DIMENSION;
use minsync::sparse::SparseVector;
use minsync::vectorstore::lancedb_store::{IndexingConfig, LanceDbStore};
use minsync::vectorstore::{Document, VectorStore};
use serde::Deserialize;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

const MODEL: &str = "nlpai-lab/KURE-Sparse-nano";

#[derive(Deserialize)]
struct Encoded {
    docs: Vec<SparseVector>,
    queries: Vec<SparseVector>,
}

#[derive(Deserialize)]
struct Line {
    text: String,
}

fn indexing(use_index: bool) -> IndexingConfig {
    IndexingConfig {
        build_ann_index: false,
        sparse_inverted_index: use_index,
        ..IndexingConfig::default()
    }
}

fn load(encoded: &str, docs_path: &str, dir: &Path, scale: usize, use_index: bool, source: &str) {
    let payload: Encoded =
        serde_json::from_reader(BufReader::new(std::fs::File::open(encoded).expect("open")))
            .expect("parse encoded");
    let texts: Vec<String> = BufReader::new(std::fs::File::open(docs_path).expect("open docs"))
        .lines()
        .map(|line| {
            serde_json::from_str::<Line>(&line.expect("read"))
                .expect("parse")
                .text
        })
        .collect();
    assert_eq!(texts.len(), payload.docs.len(), "docs and vectors differ");

    std::fs::create_dir_all(dir).expect("store dir");
    let mut store =
        LanceDbStore::open_with_indexing(dir, SPARSE_ONLY_DIMENSION, indexing(use_index))
            .expect("open store");
    let started = Instant::now();
    let total = scale * texts.len();
    let mut written = 0usize;
    for copy in 0..scale {
        for chunk in (0..texts.len()).collect::<Vec<_>>().chunks(2000) {
            let rows: Vec<Document> = chunk
                .iter()
                .map(|&i| Document {
                    id: format!("{copy:02}-{i:07}"),
                    embedding: vec![0.0; SPARSE_ONLY_DIMENSION],
                    text: texts[i].clone(),
                    source_id: source.to_string(),
                    path: format!("doc-{}.md", i % 4000),
                    chunk_schema_id: "line-cdc+heading".to_string(),
                    chunk_type: "text".to_string(),
                    heading_path: String::new(),
                    content_hash: format!("hash-{copy:02}-{i:07}"),
                    seen_token: "t".to_string(),
                })
                .collect();
            store.upsert(&rows).expect("upsert");
            let vectors: Vec<(String, SparseVector)> = chunk
                .iter()
                .map(|&i| (format!("{copy:02}-{i:07}"), payload.docs[i].clone()))
                .collect();
            store.set_sparse(MODEL, &vectors).expect("set sparse");
            written += chunk.len();
        }
    }
    let loaded = started.elapsed().as_secs_f64();
    let started = Instant::now();
    store.flush().expect("flush");
    let flushed = started.elapsed().as_secs_f64();
    println!(
        "loaded {written}/{total} rows in {loaded:.1}s; flush ({}) {flushed:.2}s",
        if use_index {
            "with index build"
        } else {
            "no index"
        }
    );
}

fn dir_bytes(dir: &Path) -> u64 {
    let mut total = 0;
    for entry in std::fs::read_dir(dir).expect("read dir").flatten() {
        let meta = entry.metadata().expect("meta");
        total += if meta.is_dir() {
            dir_bytes(&entry.path())
        } else {
            meta.len()
        };
    }
    total
}

fn open_reader(dir: &Path, use_index: bool) -> LanceDbStore {
    LanceDbStore::open_readonly(dir, SPARSE_ONLY_DIMENSION, indexing(use_index))
        .expect("open reader")
}

fn pct(sorted: &[f64], p: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

fn report(label: &str, mut ms: Vec<f64>) {
    ms.sort_by(|a, b| a.total_cmp(b));
    let mean = ms.iter().sum::<f64>() / ms.len() as f64;
    println!(
        "  {label:<26} mean {mean:>8.2} ms | p50 {:>8.2} | p95 {:>8.2} | max {:>8.2}",
        pct(&ms, 0.5),
        pct(&ms, 0.95),
        ms.last().copied().unwrap_or(0.0)
    );
}

fn query(dir: &Path, encoded: &str, k: usize, passes: usize, scan_only: bool, cap: usize) {
    let payload: Encoded =
        serde_json::from_reader(BufReader::new(std::fs::File::open(encoded).expect("open")))
            .expect("parse encoded");
    let mut payload = payload;
    payload.queries.truncate(cap);
    let state = open_reader(dir, true).index_state().expect("state");
    let index_on_disk = minsync::vectorstore::lancedb_store::LanceDbStore::open_readonly(
        dir,
        SPARSE_ONLY_DIMENSION,
        indexing(true),
    )
    .expect("reader")
    .sparse_index_ready()
    .expect("ready");
    println!(
        "{} queries, top-{k} | rows {} | inverted index describes the table: {index_on_disk} (state: {})",
        payload.queries.len(),
        open_reader(dir, true).doc_count(),
        state.sparse_indexed
    );

    let scan = open_reader(dir, false);
    let indexed = open_reader(dir, true);
    if !scan_only && index_on_disk {
        for (n, q) in payload.queries.iter().enumerate() {
            let a = indexed.query_sparse(MODEL, q, None, k).expect("indexed");
            let b = scan.query_sparse(MODEL, q, None, k).expect("scan");
            let a: Vec<_> = a.iter().map(|h| (h.doc_id.clone(), h.score)).collect();
            let b: Vec<_> = b.iter().map(|h| (h.doc_id.clone(), h.score)).collect();
            assert_eq!(a, b, "query {n}: index and scan disagree");
        }
        println!(
            "  parity: index and scan agree on all {} queries",
            payload.queries.len()
        );
    }

    let time = |store: &LanceDbStore| -> Vec<f64> {
        let mut ms = Vec::new();
        for pass in 0..=passes {
            for q in &payload.queries {
                let started = Instant::now();
                std::hint::black_box(store.query_sparse(MODEL, q, None, k).expect("query"));
                if pass > 0 {
                    ms.push(started.elapsed().as_secs_f64() * 1e3);
                }
            }
        }
        ms
    };
    report("scan (LanceDB stream)", time(&scan));
    if !scan_only && index_on_disk {
        report("inverted index", time(&indexed));
    }
}

fn cold(dir: &Path, encoded: &str, k: usize, runs: usize, scan_only: bool, cap: usize) {
    let payload: Encoded =
        serde_json::from_reader(BufReader::new(std::fs::File::open(encoded).expect("open")))
            .expect("parse encoded");
    let mut payload = payload;
    payload.queries.truncate(cap);
    let mut ms = Vec::new();
    for run in 0..runs {
        let q = &payload.queries[run % payload.queries.len()];
        let started = Instant::now();
        let store = open_reader(dir, !scan_only);
        std::hint::black_box(store.query_sparse(MODEL, q, None, k).expect("query"));
        ms.push(started.elapsed().as_secs_f64() * 1e3);
    }
    report(
        if scan_only {
            "open + scan"
        } else {
            "open + inverted index"
        },
        ms,
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str, default: usize| -> usize {
        args.windows(2)
            .find(|w| w[0] == name)
            .map(|w| w[1].parse().expect("number"))
            .unwrap_or(default)
    };
    let has = |name: &str| args.iter().any(|a| a == name);
    match args.get(1).map(String::as_str) {
        Some("load") if args.len() >= 5 => {
            let dir = Path::new(&args[4]);
            let source = args
                .windows(2)
                .find(|w| w[0] == "--source")
                .map(|w| w[1].clone())
                .unwrap_or_else(|| "bench-source".to_string());
            load(
                &args[2],
                &args[3],
                dir,
                flag("--scale", 1),
                !has("--no-index"),
                &source,
            );
            println!("  store directory: {:.0} MB", dir_bytes(dir) as f64 / 1e6);
            let index = dir.join("sparse-index.inv");
            if let Ok(meta) = std::fs::metadata(&index) {
                println!(
                    "  of which sparse-index.inv: {:.0} MB",
                    meta.len() as f64 / 1e6
                );
            }
        }
        Some("query") if args.len() >= 4 => query(
            Path::new(&args[2]),
            &args[3],
            flag("--k", 10),
            flag("--passes", 3),
            has("--scan"),
            flag("--queries", usize::MAX),
        ),
        Some("cold") if args.len() >= 4 => cold(
            Path::new(&args[2]),
            &args[3],
            flag("--k", 10),
            flag("--runs", 30),
            has("--scan"),
            flag("--queries", usize::MAX),
        ),
        _ => {
            eprintln!("usage: sparse_store_bench load <encoded> <docs.jsonl> <dir> [--scale N] [--no-index]");
            eprintln!(
                "       sparse_store_bench query <dir> <encoded> [--k K] [--passes P] [--queries N] [--scan]"
            );
            eprintln!("       sparse_store_bench cold <dir> <encoded> [--k K] [--runs R] [--scan]");
            std::process::exit(2);
        }
    }
}
