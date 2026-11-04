//! Full scan vs. the on-disk inverted index for sparse (SPLADE) search, on
//! real vectors.
//!
//! Two steps, because encoding is slow and measuring is not:
//!
//! ```text
//! # 1. Encode a corpus and its queries once (real KURE-Sparse-nano, Metal).
//! MINSYNC_TEST_DEVICE=metal cargo run --release --example sparse_inverted_bench -- \
//!     encode docs.jsonl queries.jsonl out.sparse
//!
//! # 2. Measure. `--scale N` repeats the corpus N times with distinct ids so
//! #    the same real term statistics can be timed at 2x, 5x ... the size.
//! cargo run --release --example sparse_inverted_bench -- bench out.sparse --scale 5
//! ```
//!
//! `docs.jsonl` / `queries.jsonl`: one `{"text": "..."}` per line.
//! The `bench` step checks, for every query, that the index returns exactly the
//! full scan's ranking (same ids, same scores) before it reports any time.

use minsync::config::SparseConfig;
use minsync::sparse::inverted::{IndexBuilder, InvertedFile};
use minsync::sparse::native::NativeSparseEncoder;
use minsync::sparse::{SparseEncoder, SparseVector, TopK};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::time::Instant;

#[derive(Deserialize)]
struct Line {
    text: String,
}

#[derive(Serialize, Deserialize)]
struct Encoded {
    docs: Vec<SparseVector>,
    queries: Vec<SparseVector>,
}

fn read_lines(path: &str) -> Vec<Line> {
    BufReader::new(std::fs::File::open(path).expect("open input"))
        .lines()
        .map(|line| serde_json::from_str(&line.expect("read line")).expect("parse line"))
        .collect()
}

async fn encode(docs: &str, queries: &str, out: &str) {
    let settings = SparseConfig {
        device: std::env::var("MINSYNC_TEST_DEVICE").ok(),
        dtype: Some("f32".to_string()),
        ..SparseConfig::default()
    };
    let encoder = NativeSparseEncoder::from_config(&settings).expect("build encoder");
    let docs = read_lines(docs);
    let queries = read_lines(queries);

    // Longest first, as sync does, so batches pad little.
    let mut order: Vec<usize> = (0..docs.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(docs[i].text.len()));
    let started = Instant::now();
    let mut encoded: Vec<Option<SparseVector>> = vec![None; docs.len()];
    for (done, chunk) in order.chunks(256).enumerate() {
        let texts: Vec<String> = chunk.iter().map(|&i| docs[i].text.clone()).collect();
        let vectors = encoder.encode(&texts).await.expect("encode docs");
        for (&i, vector) in chunk.iter().zip(vectors) {
            encoded[i] = Some(vector);
        }
        if done % 10 == 0 {
            eprintln!(
                "encoded {} / {} docs ({:.0}s)",
                ((done + 1) * 256).min(docs.len()),
                docs.len(),
                started.elapsed().as_secs_f64()
            );
        }
    }
    let mut query_vectors = Vec::new();
    for query in &queries {
        query_vectors.push(
            encoder
                .encode_query(&query.text)
                .await
                .expect("encode query"),
        );
    }
    let payload = Encoded {
        docs: encoded.into_iter().map(|v| v.expect("encoded")).collect(),
        queries: query_vectors,
    };
    let mut writer = BufWriter::new(std::fs::File::create(out).expect("create output"));
    serde_json::to_writer(&mut writer, &payload).expect("write output");
    writer.flush().expect("flush");
    eprintln!("wrote {out} in {:.0}s", started.elapsed().as_secs_f64());
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

fn summary(label: &str, mut ms: Vec<f64>) -> f64 {
    ms.sort_by(|a, b| a.total_cmp(b));
    let mean = ms.iter().sum::<f64>() / ms.len() as f64;
    println!(
        "  {label:<22} mean {mean:>8.3} ms | p50 {:>8.3} | p95 {:>8.3} | max {:>8.3}",
        percentile(&ms, 0.5),
        percentile(&ms, 0.95),
        ms.last().copied().unwrap_or(0.0)
    );
    mean
}

fn bench(path: &str, scale: usize, k: usize, memory_postings: usize) {
    let payload: Encoded =
        serde_json::from_reader(BufReader::new(std::fs::File::open(path).expect("open")))
            .expect("parse encoded");
    let base = payload.docs.len();
    let nnz_doc = payload.docs.iter().map(SparseVector::len).sum::<usize>() as f64 / base as f64;
    let nnz_query = payload.queries.iter().map(SparseVector::len).sum::<usize>() as f64
        / payload.queries.len() as f64;

    // `scale` copies with distinct ids. Ties between copies then break by id
    // identically in both engines, so the parity check stays meaningful.
    let rows: Vec<(String, &SparseVector)> = (0..scale)
        .flat_map(|copy| {
            payload
                .docs
                .iter()
                .enumerate()
                .map(move |(i, doc)| (format!("{copy:02}-{i:07}"), doc))
        })
        .collect();
    println!(
        "{} docs ({scale} x {base}), {} queries | mean nnz: doc {nnz_doc:.0}, query {nnz_query:.0} | top-{k}",
        rows.len(),
        payload.queries.len()
    );

    let dir = tempfile::tempdir().expect("tempdir");
    let file_path = dir.path().join("bench.inv");
    let built = Instant::now();
    let mut builder =
        IndexBuilder::new(&file_path, "bench", 1, 1, memory_postings).expect("builder");
    for (id, doc) in &rows {
        builder.count_row(id, "s", doc.indices()).expect("count");
    }
    let passes = builder.plan().expect("plan");
    for _ in 0..passes {
        for (id, doc) in &rows {
            builder
                .add_row(id, doc.indices(), doc.values())
                .expect("add");
        }
        builder.end_pass().expect("end pass");
    }
    let stats = builder.finish().expect("finish").expect("indexed");
    let index = InvertedFile::open(&file_path).expect("open index");
    println!(
        "  index file: {} postings, {:.0} MB, built in {:.2}s ({passes} pass{})",
        stats.postings,
        stats.bytes as f64 / 1e6,
        built.elapsed().as_secs_f64(),
        if passes == 1 { "" } else { "es" }
    );

    // Parity first: same ids, same order, same scores, for every query.
    let mut read_total = 0usize;
    for (n, query) in payload.queries.iter().enumerate() {
        let mut top = TopK::new(k);
        for (id, doc) in &rows {
            top.offer(query.dot(doc), id);
        }
        assert_eq!(
            index.search(query, k).expect("search"),
            top.into_sorted(),
            "query {n}: the index differs from the full scan"
        );
        read_total += query
            .indices()
            .iter()
            .map(|&t| index.posting_len(t).expect("posting_len"))
            .sum::<usize>();
    }
    println!(
        "  parity: all {} queries return the full scan's exact ranking and scores",
        payload.queries.len()
    );
    println!(
        "  postings read per query (mean): {:.0} of {} ({:.1}%)",
        read_total as f64 / payload.queries.len() as f64,
        stats.postings,
        100.0 * read_total as f64 / payload.queries.len() as f64 / stats.postings as f64
    );

    // One untimed pass to warm caches, then three timed passes.
    let mut scan_ms = Vec::new();
    let mut index_ms = Vec::new();
    for pass in 0..4 {
        for query in &payload.queries {
            let started = Instant::now();
            let mut top = TopK::new(k);
            for (id, doc) in &rows {
                top.offer(query.dot(doc), id);
            }
            std::hint::black_box(top.into_sorted());
            let scan = started.elapsed().as_secs_f64() * 1e3;

            let started = Instant::now();
            std::hint::black_box(index.search(query, k).expect("search"));
            let inverted = started.elapsed().as_secs_f64() * 1e3;
            if pass > 0 {
                scan_ms.push(scan);
                index_ms.push(inverted);
            }
        }
    }
    let scan_mean = summary("full scan (in memory)", scan_ms);
    let index_mean = summary("inverted index (file)", index_ms);
    println!("  speedup (mean): {:.1}x", scan_mean / index_mean);
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str, default: usize| -> usize {
        args.windows(2)
            .find(|w| w[0] == name)
            .map(|w| w[1].parse().expect("number"))
            .unwrap_or(default)
    };
    match args.get(1).map(String::as_str) {
        Some("encode") if args.len() == 5 => encode(&args[2], &args[3], &args[4]).await,
        Some("bench") if args.len() >= 3 => bench(
            &args[2],
            flag("--scale", 1),
            flag("--k", 10),
            flag("--memory-postings", 32_000_000),
        ),
        _ => {
            eprintln!("usage: sparse_inverted_bench encode <docs.jsonl> <queries.jsonl> <out>");
            eprintln!(
                "       sparse_inverted_bench bench <encoded> [--scale N] [--k K] [--memory-postings P]"
            );
            std::process::exit(2);
        }
    }
}
