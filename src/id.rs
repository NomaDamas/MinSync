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

/// Derive `(doc_id, content_hash)` for each chunk of a file, assigning
/// duplicate indices to chunks with identical content hash + heading path.
/// Sync (doc building) and verify (expected-id sampling) must agree on this
/// derivation, so both call here.
pub fn doc_ids_for_chunks(
    source_id: &str,
    path: &str,
    schema_id: &str,
    chunks: &[crate::types::Chunk],
) -> Vec<(String, String)> {
    let mut dup_counter: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    chunks
        .iter()
        .map(|chunk| {
            let chunk_content_hash = content_hash(&chunk.text);
            let dup_key = format!("{}\0{}", chunk_content_hash, chunk.heading_path);
            let dup_index = dup_counter.entry(dup_key).or_insert(0);
            let doc_id = compute_doc_id(
                source_id,
                path,
                schema_id,
                &chunk.chunk_type,
                &chunk_content_hash,
                *dup_index,
            );
            *dup_index += 1;
            (doc_id, chunk_content_hash)
        })
        .collect()
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

    fn chunk(text: &str) -> crate::types::Chunk {
        crate::types::Chunk {
            text: text.to_string(),
            chunk_type: "chunk".to_string(),
            heading_path: String::new(),
        }
    }

    #[test]
    fn test_doc_ids_for_chunks_duplicates_get_distinct_ids() {
        let chunks = vec![chunk("same"), chunk("same"), chunk("other")];

        let ids = doc_ids_for_chunks("source", "a.txt", "schema", &chunks);

        assert_eq!(ids.len(), 3);
        assert_ne!(ids[0].0, ids[1].0, "duplicate chunks get distinct doc ids");
        assert_eq!(ids[0].1, ids[1].1, "duplicate chunks share a content hash");
        assert_ne!(ids[0].0, ids[2].0);
    }

    #[test]
    fn test_doc_ids_for_chunks_deterministic() {
        let chunks = vec![chunk("alpha"), chunk("beta")];

        let first = doc_ids_for_chunks("source", "a.txt", "schema", &chunks);
        let second = doc_ids_for_chunks("source", "a.txt", "schema", &chunks);

        assert_eq!(first, second);
    }
}
