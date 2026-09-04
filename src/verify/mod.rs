//! Index consistency verification.
//!
//! Module layout: [`status`](mod@status) reports sync state, [`health`]
//! checks the environment, [`sampling`] recomputes expected doc IDs, and
//! this file orchestrates `minsync verify` (basic checks + stale repair).

mod health;
mod sampling;
mod status;

pub use health::check;
pub use status::status;

use crate::chunker::Chunker;
use crate::config::Config;
use crate::error::{MinSyncError, Result};
use crate::manifest::Manifest;
use crate::state::Cursor;
use crate::types::VerifyResult;
use crate::vectorstore::{Filter, VectorStore};
use sampling::{expected_doc_ids, sample_paths};
use std::collections::{HashMap, HashSet};
use std::io::ErrorKind;
use std::path::Path;

pub async fn verify(
    minsync_dir: &Path,
    root: &Path,
    chunker: &dyn Chunker,
    store: &mut dyn VectorStore,
    fix: bool,
    sample: Option<usize>,
) -> Result<VerifyResult> {
    let config_path = minsync_dir.join("config.toml");
    if !config_path.exists() {
        return Err(MinSyncError::NotInitialized);
    }

    let config = Config::load(&config_path)?;
    let cursor_path = minsync_dir.join("cursor.json");
    let txn_path = minsync_dir.join("txn.json");
    let cursor = match Cursor::load(&cursor_path) {
        Ok(cursor) => Some(cursor),
        Err(MinSyncError::Io(error)) if error.kind() == ErrorKind::NotFound => None,
        Err(error) => return Err(error),
    };

    let mut basic_checks = HashMap::new();
    basic_checks.insert("cursor_exists".to_string(), cursor.is_some());
    basic_checks.insert("no_pending_txn".to_string(), !txn_path.exists());
    basic_checks.insert(
        "schema_matches_config".to_string(),
        cursor
            .as_ref()
            .is_some_and(|active_cursor| active_cursor.chunk_schema_id == chunker.schema_id()),
    );

    let manifest = Manifest::scan(root, &config.source_id)?;
    let index_state = store.index_state()?;
    let stale_paths = find_stale_paths(&manifest, store);
    basic_checks.insert("no_stale_paths".to_string(), stale_paths.is_empty());

    let fixed = if fix {
        let repaired = repair_stale_paths(store, &config, &stale_paths)?;
        if repaired {
            basic_checks.insert("no_stale_paths".to_string(), true);
        }
        repaired
    } else {
        false
    };

    let mut sample_ok = true;
    for path in sample_paths(&manifest, sample) {
        let expected_ids = expected_doc_ids(root, &path, &config, chunker)?;
        let fetched = store.fetch(&expected_ids)?;
        if fetched.len() != expected_ids.len() {
            sample_ok = false;
            break;
        }
        let fetched_ids: HashSet<_> = fetched.into_iter().map(|doc| doc.id).collect();
        if !expected_ids.iter().all(|id| fetched_ids.contains(id)) {
            sample_ok = false;
            break;
        }
    }
    basic_checks.insert("sample_chunks_match".to_string(), sample_ok);

    Ok(VerifyResult {
        all_passed: basic_checks.values().all(|passed| *passed),
        basic_checks,
        fixed,
        index_state,
    })
}

/// Paths present in the store but absent from the manifest (deleted or
/// newly ignored files whose chunks were never swept).
fn find_stale_paths(manifest: &Manifest, store: &dyn VectorStore) -> Vec<String> {
    let manifest_paths: HashSet<_> = manifest.files.keys().cloned().collect();
    store
        .all_paths()
        .into_iter()
        .filter(|path| !manifest_paths.contains(path))
        .collect()
}

/// Delete all chunks for the given stale paths. Returns true when anything
/// was actually deleted (and flushed).
fn repair_stale_paths(
    store: &mut dyn VectorStore,
    config: &Config,
    stale_paths: &[String],
) -> Result<bool> {
    let mut fixed = false;
    for path in stale_paths {
        let deleted = store.delete_by_filter(&Filter::And(vec![
            Filter::Eq("source_id".to_string(), config.source_id.clone()),
            Filter::Eq("path".to_string(), path.clone()),
        ]))?;
        fixed |= deleted > 0;
    }
    if fixed {
        store.flush()?;
    }
    Ok(fixed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunker::chonkie::ChonkieChunker;
    use crate::embedder::Embedder;
    use crate::id::content_hash;
    use crate::state::Transaction;
    use crate::sync::MinSync;
    use crate::types::SyncState;
    use crate::vectorstore::memory::InMemoryStore;
    use crate::vectorstore::Document;
    use async_trait::async_trait;
    use tempfile::TempDir;

    struct MockEmbedder;

    #[async_trait]
    impl Embedder for MockEmbedder {
        fn id(&self) -> &str {
            "mock"
        }

        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|text| {
                    let hash = content_hash(text);
                    let bytes = hash.as_bytes();
                    (0..8).map(|i| bytes[i] as f32 / 255.0).collect()
                })
                .collect())
        }
    }

    fn fixture() -> (
        TempDir,
        MinSync,
        ChonkieChunker,
        MockEmbedder,
        InMemoryStore,
    ) {
        let dir = tempfile::tempdir().expect("create tempdir");
        let sync = MinSync::new(dir.path().to_path_buf());
        let chunker = ChonkieChunker::new(32, "\n ");
        let embedder = MockEmbedder;
        let store = InMemoryStore::new();
        (dir, sync, chunker, embedder, store)
    }

    #[tokio::test]
    async fn test_status_not_initialized() {
        let dir = tempfile::tempdir().expect("create tempdir");

        let result = status(&dir.path().join(".minsync")).await;

        assert!(matches!(result, Err(MinSyncError::NotInitialized)));
    }

    #[tokio::test]
    async fn test_status_not_synced() {
        let (dir, sync, _chunker, _embedder, _store) = fixture();
        sync.init(false, "openai:text-embedding-3-small", "recursive")
            .expect("init succeeds");

        let result = status(&dir.path().join(".minsync"))
            .await
            .expect("status succeeds");

        assert_eq!(result.state, SyncState::NotSynced);
        assert_eq!(result.last_synced_at, None);
    }

    #[tokio::test]
    async fn test_status_up_to_date() {
        let (dir, sync, chunker, embedder, mut store) = fixture();
        std::fs::write(dir.path().join("a.txt"), "alpha beta gamma").expect("write file");
        sync.init(false, "openai:text-embedding-3-small", "recursive")
            .expect("init succeeds");
        sync.sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("sync succeeds");

        let result = status(&dir.path().join(".minsync"))
            .await
            .expect("status succeeds");

        assert_eq!(result.state, SyncState::UpToDate);
    }

    #[tokio::test]
    async fn test_status_out_of_date() {
        let (dir, sync, chunker, embedder, mut store) = fixture();
        std::fs::write(dir.path().join("a.txt"), "alpha beta gamma").expect("write file");
        sync.init(false, "openai:text-embedding-3-small", "recursive")
            .expect("init succeeds");
        sync.sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("sync succeeds");
        std::fs::write(dir.path().join("a.txt"), "changed").expect("modify file");
        let result = status(&dir.path().join(".minsync"))
            .await
            .expect("status succeeds");

        assert_eq!(result.state, SyncState::OutOfDate);
    }

    #[tokio::test]
    async fn test_status_interrupted() {
        let (dir, sync, chunker, embedder, mut store) = fixture();
        std::fs::write(dir.path().join("a.txt"), "alpha beta gamma").expect("write file");
        let config = sync
            .init(false, "openai:text-embedding-3-small", "recursive")
            .expect("init succeeds");
        sync.sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("sync succeeds");
        Transaction::new(&config.source_id, "token", None, "sha256:next")
            .save(&dir.path().join(".minsync").join("txn.json"))
            .expect("save transaction");

        let result = status(&dir.path().join(".minsync"))
            .await
            .expect("status succeeds");

        assert_eq!(result.state, SyncState::Interrupted);
    }

    #[tokio::test]
    async fn test_check_all_pass() {
        let (dir, sync, _chunker, embedder, store) = fixture();
        sync.init(false, "openai:text-embedding-3-small", "recursive")
            .expect("init succeeds");

        let result = check(&dir.path().join(".minsync"), &embedder, &store)
            .await
            .expect("check succeeds");

        assert!(result.embedder_ok);
        assert!(result.vectorstore_ok);
        assert!(result.all_passed);
        assert!(result.errors.is_empty());
    }

    #[tokio::test]
    async fn test_verify_clean() {
        let (dir, sync, chunker, embedder, mut store) = fixture();
        std::fs::write(dir.path().join("a.txt"), "alpha beta gamma").expect("write file");
        sync.init(false, "openai:text-embedding-3-small", "recursive")
            .expect("init succeeds");
        sync.sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("sync succeeds");

        let result = verify(
            &dir.path().join(".minsync"),
            dir.path(),
            &chunker,
            &mut store,
            false,
            None,
        )
        .await
        .expect("verify succeeds");

        assert!(result.all_passed);
        assert!(!result.fixed);
    }

    #[tokio::test]
    async fn test_verify_stale_fix() {
        let (dir, sync, chunker, embedder, mut store) = fixture();
        std::fs::write(dir.path().join("a.txt"), "alpha beta gamma").expect("write file");
        let config = sync
            .init(false, "openai:text-embedding-3-small", "recursive")
            .expect("init succeeds");
        sync.sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("sync succeeds");
        store
            .upsert(&[Document {
                id: "stale".to_string(),
                embedding: vec![0.0; 8],
                text: "stale".to_string(),
                source_id: config.source_id,
                path: "deleted.txt".to_string(),
                chunk_schema_id: chunker.schema_id().to_string(),
                chunk_type: "text".to_string(),
                heading_path: String::new(),
                content_hash: "stale".to_string(),
                seen_token: "old".to_string(),
            }])
            .expect("upsert stale doc");

        let result = verify(
            &dir.path().join(".minsync"),
            dir.path(),
            &chunker,
            &mut store,
            true,
            None,
        )
        .await
        .expect("verify succeeds");

        assert!(result.all_passed);
        assert!(result.fixed);
        assert_eq!(store.all_paths(), vec!["a.txt"]);
    }
}
