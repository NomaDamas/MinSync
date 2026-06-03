//! Shared vector-similarity and metadata-filter helpers.
//!
//! These were originally defined in the now-removed JSON vector store. The
//! in-memory store (used for tests) still relies on a brute-force cosine scan,
//! so the primitives live here as a backend-agnostic utility module.

use crate::vectorstore::{Document, Filter};

/// Cosine similarity between two equal-length vectors.
///
/// Returns `0.0` when the vectors differ in length, are empty, or either has a
/// zero magnitude. Higher is more similar (matches the `VectorStore` query
/// contract where larger scores rank first).
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for (left, right) in a.iter().zip(b) {
        dot += left * right;
        norm_a += left * left;
        norm_b += right * right;
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a.sqrt() * norm_b.sqrt())
}

/// Evaluate a [`Filter`] against a document's string metadata fields.
pub fn matches_filter(doc: &Document, filter: &Filter) -> bool {
    match filter {
        Filter::Eq(field, value) => {
            get_field(doc, field).is_some_and(|doc_value| doc_value == value)
        }
        Filter::Neq(field, value) => {
            get_field(doc, field).is_some_and(|doc_value| doc_value != value)
        }
        Filter::And(filters) => filters.iter().all(|nested| matches_filter(doc, nested)),
    }
}

fn get_field<'a>(doc: &'a Document, field_name: &str) -> Option<&'a str> {
    match field_name {
        "id" => Some(&doc.id),
        "text" => Some(&doc.text),
        "source_id" => Some(&doc.source_id),
        "path" => Some(&doc.path),
        "chunk_schema_id" => Some(&doc.chunk_schema_id),
        "chunk_type" => Some(&doc.chunk_type),
        "heading_path" => Some(&doc.heading_path),
        "content_hash" => Some(&doc.content_hash),
        "seen_token" => Some(&doc.seen_token),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc() -> Document {
        Document {
            id: "a".to_string(),
            embedding: vec![1.0, 0.0],
            text: "text a".to_string(),
            source_id: "source-1".to_string(),
            path: "a.txt".to_string(),
            chunk_schema_id: "schema-1".to_string(),
            chunk_type: "text".to_string(),
            heading_path: "heading a".to_string(),
            content_hash: "hash-a".to_string(),
            seen_token: "token".to_string(),
        }
    }

    #[test]
    fn test_cosine_similarity_known() {
        let similarity = cosine_similarity(&[1.0, 1.0], &[1.0, 0.0]);

        assert!((similarity - 0.707).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_mismatched_or_empty() {
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[1.0]), 0.0);
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
    }

    #[test]
    fn test_matches_filter_eq_and_neq() {
        let doc = doc();
        assert!(matches_filter(
            &doc,
            &Filter::Eq("path".to_string(), "a.txt".to_string())
        ));
        assert!(!matches_filter(
            &doc,
            &Filter::Eq("path".to_string(), "b.txt".to_string())
        ));
        assert!(matches_filter(
            &doc,
            &Filter::Neq("seen_token".to_string(), "other".to_string())
        ));
    }

    #[test]
    fn test_matches_filter_and() {
        let doc = doc();
        assert!(matches_filter(
            &doc,
            &Filter::And(vec![
                Filter::Eq("path".to_string(), "a.txt".to_string()),
                Filter::Neq("seen_token".to_string(), "stale".to_string()),
            ])
        ));
    }

    #[test]
    fn test_matches_filter_unknown_field() {
        let doc = doc();
        assert!(!matches_filter(
            &doc,
            &Filter::Eq("nonexistent".to_string(), "x".to_string())
        ));
    }
}
