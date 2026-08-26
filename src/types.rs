use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Chunk {
    pub text: String,
    pub chunk_type: String,
    pub heading_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SyncResult {
    pub files_processed: usize,
    pub chunks_added: usize,
    pub chunks_updated: usize,
    pub chunks_deleted: usize,
    pub dry_run: bool,
    pub already_up_to_date: bool,
    /// True when this run performed the initial full sync because no
    /// cursor.json existed (machine-readable readiness state).
    pub initial_sync: bool,
    pub files_processed_paths: Vec<String>,
    pub files_added: usize,
    pub files_modified: usize,
    pub files_deleted: usize,
    pub elapsed_seconds: f64,
    pub embedding_api_calls: usize,
    pub embedded_texts: usize,
    pub estimated_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QueryResult {
    pub doc_id: String,
    pub path: String,
    pub heading_path: String,
    pub chunk_type: String,
    pub text: String,
    pub score: f32,
    pub content_commit: String,
    pub rank: usize,
    pub mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector_rank: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bm25_rank: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StatusResult {
    pub source_id: String,
    pub collection: String,
    pub chunker: String,
    pub embedder: String,
    pub vectorstore: String,
    pub last_synced_at: Option<String>,
    pub state: SyncState,
    pub manifest_hash: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SyncState {
    NotSynced,
    UpToDate,
    OutOfDate,
    Interrupted,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CheckResult {
    pub embedder_ok: bool,
    pub vectorstore_ok: bool,
    pub all_passed: bool,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VerifyResult {
    pub all_passed: bool,
    pub basic_checks: HashMap<String, bool>,
    pub fixed: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_types_serde_roundtrip() {
        let result = StatusResult {
            source_id: "source".to_string(),
            collection: "collection".to_string(),
            chunker: "chunker".to_string(),
            embedder: "embedder".to_string(),
            vectorstore: "vectorstore".to_string(),
            last_synced_at: Some("2026-05-28T00:00:00Z".to_string()),
            state: SyncState::UpToDate,
            manifest_hash: Some("hash".to_string()),
        };

        let json = serde_json::to_string(&result).expect("serialize status result");
        let parsed: StatusResult = serde_json::from_str(&json).expect("deserialize status result");

        assert_eq!(result, parsed);
    }
}
