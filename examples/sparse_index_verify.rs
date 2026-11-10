//! Does the inverted index change what a search returns?
//!
//! Compares three engines over the same encoded vectors, query by query:
//!
//! * the **oracle**: an in-memory full scan with `dot_terms` and `TopK`,
//! * the **index**: on-disk candidates rescored from their exact vectors,
//! * with `--store DIR`, the LanceDB table's `query_sparse` with the index and
//!   with `sparse_inverted_index = false` (the scan the product used before).
//!
//! For every query and every `k` it checks that the lists have the same length,
//! the same ids in the same order, and scores that are equal as **f32 bit
//! patterns**, not within a tolerance. It also reports, with relevance
//! judgements, nDCG@10 / Recall@100 / MRR@10 per engine, so "the same ranking"
//! is also shown in the metric a user cares about.
//!
//! ```text
//! sparse_index_verify <encoded> [--qrels FILE] [--scale N] [--ks 1,10,100,1000]
//!                     [--docs-as-queries N] [--store DIR] [--model ID]
//! ```
//!
//! `--qrels` is a TSV of `query_index<TAB>doc_index<TAB>grade`.
//! `--docs-as-queries N` additionally uses N stored documents as queries: long
//! (hundreds of terms), dense in common terms, the worst case for pruning.

use minsync::sparse::inverted::{IndexBuilder, InvertedFile};
use minsync::sparse::{SparseVector, TopK};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Deserialize)]
struct Encoded {
    docs: Vec<SparseVector>,
    queries: Vec<SparseVector>,
}

/// What the store does with the index: score only the rows it names, from
/// their exact vectors, with the scan's own loop. Returns the ranking and how
/// many candidates were scored.
fn through_index(
    index: &InvertedFile,
    by_id: &HashMap<&str, &SparseVector>,
    query: &SparseVector,
    k: usize,
) -> (Ranking, usize) {
    let candidates = index.candidates(query, k).expect("candidates");
    let mut top = TopK::new(k);
    for id in &candidates {
        top.offer(query.dot(by_id[id.as_str()]), id);
    }
    let ranking = top
        .into_sorted()
        .into_iter()
        .map(|(score, id)| (id, score.to_bits()))
        .collect();
    (ranking, candidates.len())
}

type Ranking = Vec<(String, u32)>; // (id, f32 bits)

fn oracle(rows: &[(String, &SparseVector)], query: &SparseVector, k: usize) -> Ranking {
    let mut top = TopK::new(k);
    for (id, doc) in rows {
        top.offer(query.dot(doc), id);
    }
    top.into_sorted()
        .into_iter()
        .map(|(score, id)| (id, score.to_bits()))
        .collect()
}

fn first_difference(a: &Ranking, b: &Ranking) -> Option<String> {
    if a.len() != b.len() {
        return Some(format!("length {} vs {}", a.len(), b.len()));
    }
    for (rank, (x, y)) in a.iter().zip(b).enumerate() {
        if x.0 != y.0 {
            return Some(format!("rank {rank}: id {} vs {}", x.0, y.0));
        }
        if x.1 != y.1 {
            return Some(format!(
                "rank {rank} ({}): score bits {:#010x} vs {:#010x} ({} vs {})",
                x.0,
                x.1,
                y.1,
                f32::from_bits(x.1),
                f32::from_bits(y.1)
            ));
        }
    }
    None
}

#[derive(Default)]
struct Tally {
    compared: usize,
    differing: usize,
    first: Option<String>,
    max_len: usize,
}

impl Tally {
    fn add(&mut self, label: &str, a: &Ranking, b: &Ranking) {
        self.compared += 1;
        self.max_len = self.max_len.max(a.len());
        if let Some(why) = first_difference(a, b) {
            self.differing += 1;
            self.first.get_or_insert_with(|| format!("{label}: {why}"));
        }
    }
}

fn dcg(gains: &[f64]) -> f64 {
    gains
        .iter()
        .enumerate()
        .map(|(i, g)| (2f64.powf(*g) - 1.0) / ((i + 2) as f64).log2())
        .sum()
}

#[derive(Default)]
struct Metrics {
    ndcg10: f64,
    recall100: f64,
    mrr10: f64,
    n: usize,
}

impl Metrics {
    fn add(&mut self, ranked: &[usize], judged: &HashMap<usize, f64>) {
        let relevant: HashSet<usize> = judged
            .iter()
            .filter(|(_, g)| **g > 0.0)
            .map(|(d, _)| *d)
            .collect();
        if relevant.is_empty() {
            return;
        }
        self.n += 1;
        let gains: Vec<f64> = ranked
            .iter()
            .take(10)
            .map(|d| judged.get(d).copied().unwrap_or(0.0))
            .collect();
        let mut ideal: Vec<f64> = judged.values().copied().filter(|g| *g > 0.0).collect();
        ideal.sort_by(|a, b| b.total_cmp(a));
        ideal.truncate(10);
        self.ndcg10 += dcg(&gains) / dcg(&ideal).max(f64::MIN_POSITIVE);
        self.recall100 += ranked
            .iter()
            .take(100)
            .filter(|d| relevant.contains(d))
            .count() as f64
            / relevant.len() as f64;
        self.mrr10 += ranked
            .iter()
            .take(10)
            .position(|d| relevant.contains(d))
            .map_or(0.0, |p| 1.0 / (p + 1) as f64);
    }
    fn line(&self, label: &str) -> String {
        let n = self.n.max(1) as f64;
        format!(
            "  {label:<22} nDCG@10 {:.6} | Recall@100 {:.6} | MRR@10 {:.6} | {} judged queries",
            self.ndcg10 / n,
            self.recall100 / n,
            self.mrr10 / n,
            self.n
        )
    }
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == name).map(|w| w[1].clone())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: sparse_index_verify <encoded> [--qrels FILE] [--scale N] [--ks 1,10,100,1000] [--docs-as-queries N] [--store DIR] [--model ID]");
        std::process::exit(2);
    }
    let payload: Encoded = serde_json::from_reader(BufReader::new(
        std::fs::File::open(&args[1]).expect("open encoded"),
    ))
    .expect("parse encoded");
    let scale: usize = flag(&args, "--scale").map_or(1, |s| s.parse().expect("scale"));
    let ks: Vec<usize> = flag(&args, "--ks")
        .unwrap_or_else(|| "1,10,100,1000".into())
        .split(',')
        .map(|k| k.parse().expect("k"))
        .collect();
    let mut ks = ks;
    if flag(&args, "--qrels").is_some() && !ks.contains(&100) {
        ks.push(100); // Recall@100 needs a list that deep
    }
    let base = payload.docs.len();

    // The loader of `sparse_store_bench` names rows the same way, so the id tie-break
    // (smaller id first) is the same in every engine.
    let rows: Vec<(String, &SparseVector)> = (0..scale)
        .flat_map(|copy| {
            payload
                .docs
                .iter()
                .enumerate()
                .map(move |(i, d)| (format!("{copy:02}-{i:07}"), d))
        })
        .collect();

    let mut queries: Vec<(String, &SparseVector)> = payload
        .queries
        .iter()
        .enumerate()
        .map(|(i, q)| (format!("query {i}"), q))
        .collect();
    let extra: usize = flag(&args, "--docs-as-queries").map_or(0, |s| s.parse().expect("n"));
    let stride = (base / extra.max(1)).max(1);
    for n in 0..extra {
        let at = (n * stride) % base;
        queries.push((format!("doc-as-query {at}"), &payload.docs[at]));
    }

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("verify.inv");
    let mut builder = IndexBuilder::new(&path, "verify", 1, 1, 8_000_000).expect("builder");
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
    let index = InvertedFile::open(&path).expect("open");
    println!(
        "{} docs x {} queries ({} real + {} docs-as-queries), k in {:?}; index {:.0} MB, {passes} build passes",
        rows.len(),
        queries.len(),
        payload.queries.len(),
        extra,
        ks,
        stats.bytes as f64 / 1e6
    );

    let by_id: HashMap<&str, &SparseVector> =
        rows.iter().map(|(id, doc)| (id.as_str(), *doc)).collect();
    let mut vs_index = Tally::default();
    let mut oracle_lists: HashMap<(usize, usize), Ranking> = HashMap::new();
    let mut candidate_counts: HashMap<usize, Vec<usize>> = HashMap::new();
    for (qi, (label, q)) in queries.iter().enumerate() {
        for &k in &ks {
            let want = oracle(&rows, q, k);
            let (got, scored) = through_index(&index, &by_id, q, k);
            candidate_counts.entry(k).or_default().push(scored);
            vs_index.add(&format!("{label}, k={k}"), &want, &got);
            oracle_lists.insert((qi, k), want);
        }
    }
    for &k in &ks {
        let counts = &candidate_counts[&k];
        println!(
            "candidates scored for k={k}: mean {:.1}, max {} of {} rows",
            counts.iter().sum::<usize>() as f64 / counts.len() as f64,
            counts.iter().max().copied().unwrap_or(0),
            rows.len()
        );
    }
    println!(
        "inverted index vs in-memory oracle: {} comparisons, {} differ, longest list {}{}",
        vs_index.compared,
        vs_index.differing,
        vs_index.max_len,
        vs_index
            .first
            .as_ref()
            .map_or(String::new(), |f| format!("  FIRST: {f}"))
    );

    // Retrieval quality from the judgements, per engine.
    if let Some(qrels) = flag(&args, "--qrels") {
        let mut judged: HashMap<usize, HashMap<usize, f64>> = HashMap::new();
        for line in BufReader::new(std::fs::File::open(qrels).expect("qrels")).lines() {
            let line = line.expect("line");
            let mut f = line.split('\t');
            let (q, d, g): (usize, usize, f64) = (
                f.next().expect("q").parse().expect("q"),
                f.next().expect("d").parse().expect("d"),
                f.next().expect("g").parse().expect("g"),
            );
            judged.entry(q).or_default().insert(d, g);
        }
        let (mut by_oracle, mut by_index) = (Metrics::default(), Metrics::default());
        for qi in 0..payload.queries.len() {
            let Some(j) = judged.get(&qi) else { continue };
            let doc_of = |r: &Ranking| -> Vec<usize> {
                r.iter()
                    .map(|(id, _)| id[3..].parse::<usize>().expect("id"))
                    .collect()
            };
            by_oracle.add(&doc_of(&oracle_lists[&(qi, 100)]), j);
            let (got, _) = through_index(&index, &by_id, &payload.queries[qi], 100);
            by_index.add(&doc_of(&got), j);
        }
        println!("{}", by_oracle.line("scan (oracle)"));
        println!("{}", by_index.line("inverted index"));
        assert_eq!(
            format!("{:.12}", by_oracle.ndcg10),
            format!("{:.12}", by_index.ndcg10),
            "nDCG differs"
        );
    }

    // The same through LanceDB, i.e. through the product's code path.
    if let Some(store_dir) = flag(&args, "--store") {
        store_check(&args, Path::new(&store_dir), &queries, &ks, &oracle_lists);
    }

    assert_eq!(
        vs_index.differing, 0,
        "the index returned a different answer"
    );
    println!("RESULT: identical (ids, order and f32 bit patterns) in every comparison");
}

fn store_check(
    args: &[String],
    dir: &Path,
    queries: &[(String, &SparseVector)],
    ks: &[usize],
    oracle_lists: &HashMap<(usize, usize), Ranking>,
) {
    use minsync::vectorstore::lancedb_store::{IndexingConfig, LanceDbStore};
    use minsync::vectorstore::VectorStore;
    let model = flag(args, "--model").unwrap_or_else(|| "nlpai-lab/KURE-Sparse-nano".into());
    let open = |index: bool| {
        LanceDbStore::open_readonly(
            dir,
            1,
            IndexingConfig {
                build_ann_index: false,
                sparse_inverted_index: index,
                ..IndexingConfig::default()
            },
        )
        .expect("open store")
    };
    let (with_index, scan) = (open(true), open(false));
    assert!(
        with_index.sparse_index_ready().expect("ready"),
        "the store has no index that describes its table"
    );
    let (mut idx_vs_scan, mut idx_vs_oracle, mut scan_vs_oracle) =
        (Tally::default(), Tally::default(), Tally::default());
    for (qi, (label, q)) in queries.iter().enumerate() {
        for &k in ks {
            let run = |store: &LanceDbStore| -> Ranking {
                store
                    .query_sparse(&model, q, None, k)
                    .expect("query_sparse")
                    .into_iter()
                    .map(|h| (h.doc_id, h.score.to_bits()))
                    .collect()
            };
            let (a, b) = (run(&with_index), run(&scan));
            let key = format!("{label}, k={k}");
            idx_vs_scan.add(&key, &a, &b);
            if let Some(want) = oracle_lists.get(&(qi, k)) {
                idx_vs_oracle.add(&key, want, &a);
                scan_vs_oracle.add(&key, want, &b);
            }
        }
    }
    for (name, t) in [
        ("LanceDB: index vs LanceDB scan", &idx_vs_scan),
        ("LanceDB: index vs oracle", &idx_vs_oracle),
        ("LanceDB: scan vs oracle", &scan_vs_oracle),
    ] {
        println!(
            "{name}: {} comparisons, {} differ{}",
            t.compared,
            t.differing,
            t.first
                .as_ref()
                .map_or(String::new(), |f| format!("  FIRST: {f}"))
        );
        assert_eq!(t.differing, 0, "{name}");
    }
}
