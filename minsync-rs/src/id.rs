use sha2::{Digest, Sha256};

/// Compute deterministic document ID.
/// Formula: sha256(source_id \0 path \0 chunk_schema_id \0 chunk_type \0 content_hash \0 dup_index)
pub fn compute_doc_id(
    source_id: &str,
    path: &str,
    chunk_schema_id: &str,
    chunk_type: &str,
    content_hash: &str,
    dup_index: usize,
) -> String {
    let dup_index = dup_index.to_string();
    let payload = [
        source_id,
        path,
        chunk_schema_id,
        chunk_type,
        content_hash,
        &dup_index,
    ]
    .join("\0");

    let mut hasher = Sha256::new();
    hasher.update(payload.as_bytes());
    hex::encode(hasher.finalize())
}

/// Compute content hash of text.
pub fn content_hash(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_doc_id_deterministic() {
        let first = compute_doc_id("source", "src/lib.rs", "schema", "text", "hash", 0);
        let second = compute_doc_id("source", "src/lib.rs", "schema", "text", "hash", 0);

        assert_eq!(first, second);
        assert_eq!(first.len(), 64);
    }

    #[test]
    fn test_doc_id_different_inputs() {
        let first = compute_doc_id("source", "src/lib.rs", "schema", "text", "hash", 0);
        let second = compute_doc_id("source", "src/main.rs", "schema", "text", "hash", 0);

        assert_ne!(first, second);
    }

    #[test]
    fn test_doc_id_different_dup_index() {
        let first = compute_doc_id("source", "src/lib.rs", "schema", "text", "hash", 0);
        let second = compute_doc_id("source", "src/lib.rs", "schema", "text", "hash", 1);

        assert_ne!(first, second);
    }

    #[test]
    fn test_content_hash_deterministic() {
        let first = content_hash("same text");
        let second = content_hash("same text");

        assert_eq!(first, second);
        assert_eq!(first.len(), 64);
    }

    #[test]
    fn test_content_hash_different() {
        assert_ne!(content_hash("text a"), content_hash("text b"));
    }
}
