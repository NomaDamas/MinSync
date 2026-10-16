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
    pub chunks_truncated: usize,
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
    /// New chunks whose vector was reused from an identical chunk (same
    /// content hash) instead of being embedded again.
    #[serde(default)]
    pub embeddings_reused: usize,
    pub estimated_tokens: usize,
    pub files_checked: usize,
    pub freshness_check_only: bool,
    pub query_ready: bool,
    /// Files skipped because an interrupted earlier run had already
    /// committed them (crash resume). NOT included in `files_processed`.
    #[serde(default)]
    pub files_resumed: usize,
    /// Entries the manifest scan could not inspect (unreadable directory,
    /// I/O error, ...). Their previously indexed chunks are kept untouched.
    #[serde(default)]
    pub scan_errors: Vec<crate::manifest::ScanError>,
    /// Files that could not be indexed (unreadable, unparseable, embed or
    /// store failure) and were skipped when `[sync] on_file_error = "skip"`.
    /// They are retried by the next sync.
    #[serde(default)]
    pub files_failed: Vec<FileFailure>,
    /// True iff `files_failed` is non-empty: the run completed but skipped
    /// files. The CLI exits with a distinct code for it.
    #[serde(default)]
    pub partial: bool,
}

/// Pipeline stage that failed for one file.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum FileStage {
    Read,
    Chunk,
    Truncate,
    Embed,
    Store,
}

impl std::fmt::Display for FileStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Read => "read",
            Self::Chunk => "chunk",
            Self::Truncate => "truncate",
            Self::Embed => "embed",
            Self::Store => "store",
        })
    }
}

/// One skipped file: where it failed and why. `error` is the bare error
/// text; the path is its own field.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileFailure {
    pub path: String,
    pub stage: FileStage,
    pub error: String,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index_state: Option<IndexState>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IndexState {
    pub fts_indexed: bool,
    pub ann_indexed: bool,
    pub unindexed_rows: Option<usize>,
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
    pub index_state: Option<IndexState>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VerifyResult {
    pub all_passed: bool,
    pub basic_checks: HashMap<String, bool>,
    pub embedding_integrity: EmbeddingIntegrityResult,
    pub fixed: bool,
    pub index_state: Option<IndexState>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EmbeddingIntegrityResult {
    pub flagged_pairs: Vec<EmbeddingIntegrityPair>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EmbeddingIntegrityPair {
    pub chunk_ids: Vec<String>,
    pub paths: Vec<String>,
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
