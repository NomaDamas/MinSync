//! Multilingual BM25 tokenization, Okapi BM25 scoring, and a compressed
//! on-disk inverted index.
//!
//! # Architecture
//!
//! BM25 complements dense vectors and learned-sparse (SPLADE) retrieval:
//! - On low-spec PCs that cannot run even a 0.1B SPLADE model, users can
//!   disable both dense (`--embedder none`) and sparse (`--sparse none`) and
//!   still get fast, zero-ML, multilingual BM25 retrieval.
//! - On machines with dense and/or SPLADE enabled, BM25 runs alongside them
//!   so queries can pick any combination (`bm25`, `vector`, `sparse`, `hybrid`,
//!   `all`, or explicit `--mode`) — including feeding multi-feature LTR
//!   rerankers with independent dense + SPLADE + BM25 signals.
//!
//! # Modules
//! - [`tokenizer`]: Pure-Rust multilingual tokenization (English/Latin,
//!   Korean via `lindera` ko-dic, Japanese via `lindera` IPADIC, Chinese via
//!   `jieba-rs`, Arabic light stemmer, and `multilingual` script-aware routing).
//! - [`inverted`]: Compressed on-disk BM25 inverted index (`bm25-index.inv`),
//!   reusing the block bit-packing and positioned-read design of
//!   [`crate::sparse::inverted`] with an embedded dictionary, doc-length table,
//!   and exact Okapi BM25 scoring (k1 = 1.2, b = 0.75).

pub mod inverted;
pub mod tokenizer;

use crate::sparse::TopK;
use std::collections::HashMap;

/// Default Okapi BM25 term-frequency saturation parameter.
pub const DEFAULT_K1: f32 = 1.2;

/// Default Okapi BM25 document-length normalization parameter.
pub const DEFAULT_B: f32 = 0.75;

/// Lucene-style smooth IDF: `ln(1 + (N - df + 0.5) / (df + 0.5))`.
/// Always strictly positive for any `df >= 1` and `N >= df`, so even terms
/// appearing in more than half the corpus still contribute a positive weight
/// (never negative scores that break top-k heaps or fusion).
#[inline]
pub fn idf(doc_count: u64, doc_freq: u32) -> f32 {
    let n = doc_count as f64;
    let df = doc_freq as f64;
    (1.0 + (n - df + 0.5) / (df + 0.5)).ln() as f32
}

/// Okapi BM25 term score contribution for a single term occurrence in a document.
///
/// `tf`: term frequency in the document (`>= 1`).
/// `doc_len`: document length in tokens (`>= 1`).
/// `avg_doc_len`: corpus average document length (`> 0`).
/// `term_idf`: [`idf(doc_count, doc_freq)`].
#[inline]
pub fn bm25_tf_component(tf: u32, doc_len: u32, avg_doc_len: f32, k1: f32, b: f32) -> f32 {
    let tf = tf as f32;
    let dl = doc_len as f32;
    let denom = tf + k1 * (1.0 - b + b * (dl / avg_doc_len));
    (tf * (k1 + 1.0)) / denom
}

/// Compute term frequencies and total token length for a tokenized document.
pub fn term_frequencies(tokens: &[String]) -> (Vec<(String, u32)>, u32) {
    let mut counts: HashMap<&str, u32> = HashMap::new();
    for token in tokens {
        *counts.entry(token.as_str()).or_insert(0) += 1;
    }
    let mut pairs: Vec<(String, u32)> = counts
        .into_iter()
        .map(|(term, tf)| (term.to_string(), tf))
        .collect();
    pairs.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    let doc_len = u32::try_from(tokens.len()).unwrap_or(u32::MAX);
    (pairs, doc_len)
}

/// Exact Okapi BM25 scoring over an in-memory collection of `(id, tokens)` rows.
/// Used both by [`crate::vectorstore::memory::InMemoryStore`] and as the
/// fallback scan when the on-disk BM25 inverted index cannot answer a query
/// (e.g. unsupported filter, below `min_rows`, or mid-sync before `flush`).
pub fn score_bm25_scan(
    query_tokens: &[String],
    docs: &[(String, Vec<String>)],
    topk: usize,
) -> Vec<(f32, String)> {
    if topk == 0 || query_tokens.is_empty() || docs.is_empty() {
        return Vec::new();
    }
    // Count query term frequencies (a repeated query term scales its contribution).
    let mut query_tf: HashMap<&str, u32> = HashMap::new();
    for token in query_tokens {
        *query_tf.entry(token.as_str()).or_insert(0) += 1;
    }

    let n = docs.len() as u64;
    let total_len: u64 = docs.iter().map(|(_, tokens)| tokens.len() as u64).sum();
    if total_len == 0 {
        return Vec::new();
    }
    let avg_dl = (total_len as f64 / n as f64) as f32;

    // Document frequency for each distinct query term across all matching docs.
    let mut df: HashMap<&str, u32> = HashMap::with_capacity(query_tf.len());
    // Per-document query-term frequencies; only store non-zero hits.
    let mut doc_matches: Vec<(usize, Vec<(&str, u32)>)> = Vec::new();

    for (idx, (_, tokens)) in docs.iter().enumerate() {
        if tokens.is_empty() {
            continue;
        }
        let mut counts: HashMap<&str, u32> = HashMap::new();
        for token in tokens {
            if let Some((&qterm, _)) = query_tf.get_key_value(token.as_str()) {
                *counts.entry(qterm).or_insert(0) += 1;
            }
        }
        if !counts.is_empty() {
            for &term in counts.keys() {
                *df.entry(term).or_insert(0) += 1;
            }
            let mut entries: Vec<(&str, u32)> = counts.into_iter().collect();
            entries.sort_unstable_by(|a, b| a.0.cmp(b.0));
            doc_matches.push((idx, entries));
        }
    }

    let mut top = TopK::new(topk);
    for (idx, entries) in doc_matches {
        let doc_len = docs[idx].1.len() as u32;
        let mut score = 0.0f32;
        for (term, tf) in entries {
            let term_df = df[term];
            let qtf = query_tf[term] as f32;
            let term_idf = idf(n, term_df);
            score += qtf * term_idf * bm25_tf_component(tf, doc_len, avg_dl, DEFAULT_K1, DEFAULT_B);
        }
        top.offer(score, &docs[idx].0);
    }
    top.into_sorted()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idf_is_positive_and_monotonically_decreasing_with_df() {
        let rare = idf(100, 1);
        let medium = idf(100, 10);
        let common = idf(100, 90);
        assert!(rare > medium);
        assert!(medium > common);
        assert!(common > 0.0);
    }

    #[test]
    fn bm25_scan_ranks_exact_and_higher_tf_documents_first() {
        let docs = vec![
            (
                "d1".to_string(),
                vec!["rust".into(), "search".into(), "engine".into()],
            ),
            (
                "d2".to_string(),
                vec!["rust".into(), "rust".into(), "search".into(), "bm25".into()],
            ),
            (
                "d3".to_string(),
                vec!["python".into(), "web".into(), "framework".into()],
            ),
        ];
        let query = vec!["rust".to_string(), "bm25".to_string()];
        let ranked = score_bm25_scan(&query, &docs, 10);
        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].1, "d2");
        assert_eq!(ranked[1].1, "d1");
        assert!(ranked[0].0 > ranked[1].0);
    }

    #[test]
    fn bm25_scan_respects_document_length_normalization() {
        let short = vec!["zebra".into(), "apple".into()];
        let mut long = vec!["zebra".into()];
        for i in 0..50 {
            long.push(format!("filler{i}"));
        }
        let docs = vec![("long".to_string(), long), ("short".to_string(), short)];
        let ranked = score_bm25_scan(&["zebra".to_string()], &docs, 10);
        assert_eq!(ranked[0].1, "short");
        assert_eq!(ranked[1].1, "long");
        assert!(ranked[0].0 > ranked[1].0);
    }
}
