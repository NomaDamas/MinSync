use crate::chunker::Chunker;
use crate::config::Config;
use crate::embedder::Embedder;
use crate::error::{MinSyncError, Result};
use crate::id::{compute_doc_id, content_hash};
use crate::manifest::Manifest;
use crate::normalize::normalize_text;
use crate::state::Cursor;
use crate::types::{CheckResult, StatusResult, SyncState, VerifyResult};
use crate::vectorstore::{Filter, VectorStore};
use std::collections::{HashMap, HashSet};
use std::io::ErrorKind;
use std::path::Path;

pub async fn status(minsync_dir: &Path) -> Result<StatusResult> {
    let config_path = minsync_dir.join("config.toml");
    if !config_path.exists() {
        return Err(MinSyncError::NotInitialized);
    }

    let config = Config::load(&config_path)?;
    let cursor_path = minsync_dir.join("cursor.json");
    let txn_path = minsync_dir.join("txn.json");

    if !cursor_path.exists() {
        return Ok(status_result(&config, None, SyncState::NotSynced, None));
    }

    let cursor = Cursor::load(&cursor_path)?;
    if txn_path.exists() {
        return Ok(status_result(
            &config,
            Some(&cursor),
            SyncState::Interrupted,
            Some(cursor.manifest_hash.clone()),
        ));
    }

    let manifest_hash = load_or_scan_manifest(minsync_dir, &config)?.manifest_hash();
    let state = if manifest_hash == cursor.manifest_hash {
        SyncState::UpToDate
    } else {
        SyncState::OutOfDate
    };

    Ok(status_result(&config, Some(&cursor), state, Some(manifest_hash)))
}

pub async fn check(minsync_dir: &Path, embedder: &dyn Embedder, store: &dyn VectorStore) -> Result<CheckResult> {
    if !minsync_dir.join("config.toml").exists() {
        return Err(MinSyncError::NotInitialized);
    }

    let mut errors = Vec::new();
    let embedder_ok = match embedder.embed_single("hello").await {
        Ok(vector) if vector.is_empty() => {
            errors.push("embedder returned empty embedding".to_string());
            false
        }
        Ok(_) => true,
        Err(error) => {
            errors.push(format!("embedder check failed: {error}"));
            false
        }
    };

    let vectorstore_ok = {
        let _ = store.doc_count();
        true
    };

    Ok(CheckResult {
        embedder_ok,
        vectorstore_ok,
        all_passed: embedder_ok && vectorstore_ok && errors.is_empty(),
        errors,
    })
}

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
    let manifest_paths: HashSet<_> = manifest.files.keys().cloned().collect();
    let store_paths = store.all_paths();
    let stale_paths: Vec<_> = store_paths
        .iter()
        .filter(|path| !manifest_paths.contains(*path))
        .cloned()
        .collect();
    let stale_clean = stale_paths.is_empty();
    basic_checks.insert("no_stale_paths".to_string(), stale_clean);

    let mut fixed = false;
    if fix {
        for path in &stale_paths {
            let deleted = store.delete_by_filter(&Filter::And(vec![
                Filter::Eq("source_id".to_string(), config.source_id.clone()),
                Filter::Eq("path".to_string(), path.clone()),
            ]))?;
            fixed |= deleted > 0;
        }
        if fixed {
            store.flush()?;
            basic_checks.insert("no_stale_paths".to_string(), true);
        }
    }

    let sample_paths = sample_paths(&manifest, sample);
    let mut sample_ok = true;
    for path in sample_paths {
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
    })
}

fn status_result(
    config: &Config,
    cursor: Option<&Cursor>,
    state: SyncState,
    manifest_hash: Option<String>,
) -> StatusResult {
    StatusResult {
        source_id: config.source_id.clone(),
        collection: config.collection.name.clone(),
        chunker: config.chunker.id.clone(),
        embedder: config.embedder.id.clone(),
        vectorstore: config.vectorstore.id.clone(),
        last_synced_at: cursor.map(|active_cursor| active_cursor.last_synced_at.clone()),
        state,
        manifest_hash,
    }
}

fn load_or_scan_manifest(minsync_dir: &Path, config: &Config) -> Result<Manifest> {
    if let Some(root) = minsync_dir.parent() {
        return Manifest::scan(root, &config.source_id);
    }

    Manifest::load(&minsync_dir.join("manifest.json"))
}

fn sample_paths(manifest: &Manifest, sample: Option<usize>) -> Vec<String> {
    let mut paths: Vec<_> = manifest.files.keys().cloned().collect();
    paths.sort();
    if let Some(limit) = sample {
        paths.truncate(limit);
    }
    paths
}

fn expected_doc_ids(root: &Path, path: &str, config: &Config, chunker: &dyn Chunker) -> Result<Vec<String>> {
    let raw_text = match std::fs::read_to_string(root.join(path)) {
        Ok(text) => text,
        Err(error) if error.kind() == ErrorKind::InvalidData => String::new(),
        Err(error) => return Err(MinSyncError::Io(error)),
    };
    let text = normalize_text(&raw_text, &config.normalize);
    let chunks = chunker.chunk(&text, path)?;
    let mut dup_counter: HashMap<String, usize> = HashMap::new();

    Ok(chunks
        .into_iter()
        .map(|chunk| {
            let chunk_content_hash = content_hash(&chunk.text);
            let dup_key = format!("{}\0{}", chunk_content_hash, chunk.heading_path);
            let dup_index = dup_counter.entry(dup_key).or_insert(0);
            let doc_id = compute_doc_id(
                &config.source_id,
                path,
                chunker.schema_id(),
                &chunk.chunk_type,
                &chunk_content_hash,
                *dup_index,
            );
            *dup_index += 1;
            doc_id
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunker::chonkie::ChonkieChunker;
    use crate::state::Transaction;
    use crate::sync::MinSync;
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

    fn fixture() -> (TempDir, MinSync, ChonkieChunker, MockEmbedder, InMemoryStore) {
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
        sync.init(false).expect("init succeeds");

        let result = status(&dir.path().join(".minsync")).await.expect("status succeeds");

        assert_eq!(result.state, SyncState::NotSynced);
        assert_eq!(result.last_synced_at, None);
    }

    #[tokio::test]
    async fn test_status_up_to_date() {
        let (dir, sync, chunker, embedder, mut store) = fixture();
        std::fs::write(dir.path().join("a.txt"), "alpha beta gamma").expect("write file");
        sync.init(false).expect("init succeeds");
        sync.sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("sync succeeds");

        let result = status(&dir.path().join(".minsync")).await.expect("status succeeds");

        assert_eq!(result.state, SyncState::UpToDate);
    }

    #[tokio::test]
    async fn test_status_out_of_date() {
        let (dir, sync, chunker, embedder, mut store) = fixture();
        std::fs::write(dir.path().join("a.txt"), "alpha beta gamma").expect("write file");
        sync.init(false).expect("init succeeds");
        sync.sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("sync succeeds");
        std::fs::write(dir.path().join("a.txt"), "changed").expect("modify file");
        let result = status(&dir.path().join(".minsync")).await.expect("status succeeds");

        assert_eq!(result.state, SyncState::OutOfDate);
    }

    #[tokio::test]
    async fn test_status_interrupted() {
        let (dir, sync, chunker, embedder, mut store) = fixture();
        std::fs::write(dir.path().join("a.txt"), "alpha beta gamma").expect("write file");
        let config = sync.init(false).expect("init succeeds");
        sync.sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("sync succeeds");
        Transaction::new(&config.source_id, "token", None, "sha256:next")
            .save(&dir.path().join(".minsync").join("txn.json"))
            .expect("save transaction");

        let result = status(&dir.path().join(".minsync")).await.expect("status succeeds");

        assert_eq!(result.state, SyncState::Interrupted);
    }

    #[tokio::test]
    async fn test_check_all_pass() {
        let (dir, sync, _chunker, embedder, store) = fixture();
        sync.init(false).expect("init succeeds");

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
        sync.init(false).expect("init succeeds");
        sync.sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("sync succeeds");

        let result = verify(&dir.path().join(".minsync"), dir.path(), &chunker, &mut store, false, None)
            .await
            .expect("verify succeeds");

        assert!(result.all_passed);
        assert!(!result.fixed);
    }

    #[tokio::test]
    async fn test_verify_stale_fix() {
        let (dir, sync, chunker, embedder, mut store) = fixture();
        std::fs::write(dir.path().join("a.txt"), "alpha beta gamma").expect("write file");
        let config = sync.init(false).expect("init succeeds");
        sync.sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("sync succeeds");
        store.upsert(&[Document {
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

        let result = verify(&dir.path().join(".minsync"), dir.path(), &chunker, &mut store, true, None)
            .await
            .expect("verify succeeds");

        assert!(result.all_passed);
        assert!(result.fixed);
        assert_eq!(store.all_paths(), vec!["a.txt"]);
    }
}
