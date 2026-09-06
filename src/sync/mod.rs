//! Incremental sync orchestration: locking, transaction lifecycle, manifest
//! diffing, and cursor advancement. Per-file work lives in [`indexer`];
//! result accounting helpers live in [`result`].

mod indexer;
mod result;

use crate::chunker::Chunker;
use crate::config::Config;
use crate::embedder::Embedder;
use crate::error::{MinSyncError, Result};
use crate::manifest::{FileChange, Manifest};
use crate::state::{Cursor, FileLock, Transaction};
use crate::types::SyncResult;
use crate::vectorstore::{Filter, VectorStore};
use indexer::{index_file, SyncFileContext};
use result::{change_path, empty_sync_result};
use std::path::{Component, Path, PathBuf};

pub struct MinSync {
    root: PathBuf,
    minsync_dir: PathBuf,
}

impl MinSync {
    pub fn new(root: PathBuf) -> Self {
        let minsync_dir = root.join(".minsync");
        Self { root, minsync_dir }
    }

    pub fn minsync_dir(&self) -> &std::path::Path {
        &self.minsync_dir
    }

    /// Initialize `.minsync/` with a default config and baseline manifest.
    pub fn init(&self, force: bool, embedder_id: &str, chunker_id: &str) -> Result<Config> {
        if self.minsync_dir.exists() && !force {
            return Err(MinSyncError::AlreadyInitialized);
        }

        let _lock = force
            .then(|| FileLock::acquire(&self.minsync_dir.join("lock"), false))
            .transpose()?;
        if force {
            self.reset_forced_state()?;
        }
        std::fs::create_dir_all(&self.minsync_dir)?;
        let source_id = uuid::Uuid::new_v4().to_string();
        let mut config = Config::default_for(&source_id);
        config.embedder.id = embedder_id.to_string();
        config.chunker.id = chunker_id.to_string();

        config.save(&self.minsync_dir.join("config.toml"))?;
        Manifest::scan(&self.root, &source_id)?.save(&self.minsync_dir.join("manifest.json"))?;

        Ok(config)
    }

    fn reset_forced_state(&self) -> Result<()> {
        let config_path = self.minsync_dir.join("config.toml");
        if config_path.exists() {
            let previous = Config::load(&config_path)?;
            let previous_store =
                collection_store_path(&self.minsync_dir, &previous.collection.path)?;
            if previous_store.exists() {
                std::fs::remove_dir_all(previous_store)?;
            }
        }

        for state_file in ["cursor.json", "txn.json"] {
            let path = self.minsync_dir.join(state_file);
            if path.exists() {
                std::fs::remove_file(path)?;
            }
        }
        Ok(())
    }
}

pub(crate) fn collection_store_path(minsync_dir: &Path, configured_path: &str) -> Result<PathBuf> {
    let base = minsync_dir.canonicalize()?;
    let path = Path::new(configured_path);
    let mut candidate = base.clone();
    let mut has_normal_component = false;
    for component in path.components() {
        match component {
            Component::Normal(name) => {
                has_normal_component = true;
                candidate.push(name);
            }
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(MinSyncError::Config(format!(
                    "collection.path must stay inside .minsync/: {configured_path}"
                )));
            }
        }

        match std::fs::symlink_metadata(&candidate) {
            Ok(metadata) => {
                if metadata.file_type().is_symlink() {
                    return Err(MinSyncError::Config(format!(
                        "collection.path must stay inside .minsync/: {configured_path}"
                    )));
                }
                let resolved = candidate.canonicalize()?;
                if !resolved.starts_with(&base) {
                    return Err(MinSyncError::Config(format!(
                        "collection.path must stay inside .minsync/: {configured_path}"
                    )));
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => break,
            Err(error) => return Err(error.into()),
        }
    }

    if !has_normal_component {
        return Err(MinSyncError::Config(
            "collection.path must name a directory inside .minsync/".to_string(),
        ));
    }
    Ok(base.join(path))
}

impl MinSync {
    /// Incrementally synchronize changed files into the supplied vector store.
    pub async fn sync(
        &self,
        chunker: &dyn Chunker,
        embedder: &dyn Embedder,
        store: &mut dyn VectorStore,
        full: bool,
        dry_run: bool,
        wait_lock: bool,
    ) -> Result<SyncResult> {
        let _lock = FileLock::acquire(&self.minsync_dir.join("lock"), wait_lock)?;
        self.sync_locked(chunker, embedder, store, full, dry_run)
            .await
    }

    pub(crate) async fn sync_locked(
        &self,
        chunker: &dyn Chunker,
        embedder: &dyn Embedder,
        store: &mut dyn VectorStore,
        full: bool,
        dry_run: bool,
    ) -> Result<SyncResult> {
        let config_path = self.minsync_dir.join("config.toml");
        if !config_path.exists() {
            return Err(MinSyncError::NotInitialized);
        }

        let config = Config::load(&config_path)?;

        let txn_path = self.minsync_dir.join("txn.json");
        let cursor_path = self.minsync_dir.join("cursor.json");
        let manifest_path = self.minsync_dir.join("manifest.json");

        // A cursor-less workspace has never completed an initial sync: its
        // baseline manifest (written by `init`) cannot drive a meaningful
        // incremental diff, so upgrade to a full sync. This also guarantees
        // the `already_up_to_date` early return below is unreachable while
        // no cursor exists.
        let initial_sync = !cursor_path.exists();
        let lexical_rebuild = if cursor_path.exists() {
            Cursor::load(&cursor_path)?.lexical_language != config.lexical.language
        } else {
            false
        };
        let full = full || initial_sync || lexical_rebuild;

        if txn_path.exists() {
            Transaction::remove(&txn_path)?;
        }

        let stored_manifest = if manifest_path.exists() {
            Some(Manifest::load(&manifest_path)?)
        } else {
            None
        };
        let old_manifest = if full {
            Manifest::new(&config.source_id)
        } else {
            stored_manifest
                .clone()
                .unwrap_or_else(|| Manifest::new(&config.source_id))
        };
        let scan_baseline = if full { None } else { stored_manifest.as_ref() };
        let start = std::time::Instant::now();
        let new_manifest =
            Manifest::scan_with_baseline(&self.root, &config.source_id, scan_baseline)?;
        let changes = Manifest::diff(&old_manifest, &new_manifest);

        if changes.is_empty() && !full {
            let mut result = empty_sync_result(dry_run, true);
            result.files_checked = new_manifest.files.len();
            result.query_ready = cursor_path.exists();
            result.elapsed_seconds = start.elapsed().as_secs_f64();
            return Ok(result);
        }

        if dry_run {
            let files_processed_paths = changes.iter().map(change_path).collect();
            let files_added = changes
                .iter()
                .filter(|change| matches!(change, FileChange::Added(_)))
                .count();
            let files_modified = changes
                .iter()
                .filter(|change| matches!(change, FileChange::Modified(_)))
                .count();
            let files_deleted = changes
                .iter()
                .filter(|change| matches!(change, FileChange::Deleted(_)))
                .count();
            let mut result = SyncResult {
                files_processed_paths,
                files_added,
                files_modified,
                files_deleted,
                dry_run: true,
                already_up_to_date: false,
                initial_sync,
                ..empty_sync_result(true, false)
            };
            result.files_checked = new_manifest.files.len();
            result.elapsed_seconds = start.elapsed().as_secs_f64();
            return Ok(result);
        }

        let sync_token = uuid::Uuid::new_v4().to_string().replace('-', "");
        Transaction::new(
            &config.source_id,
            &sync_token,
            Some(old_manifest.manifest_hash()),
            &new_manifest.manifest_hash(),
        )
        .save(&txn_path)?;

        let mut result = empty_sync_result(false, false);
        result.initial_sync = initial_sync;
        result.files_checked = new_manifest.files.len();
        result.query_ready = true;
        if lexical_rebuild {
            result.chunks_deleted += store.delete_by_filter(&Filter::Eq(
                "source_id".to_string(),
                config.source_id.clone(),
            ))?;
        }

        for change in &changes {
            match change {
                FileChange::Added(path) => {
                    result.files_added += 1;
                    let context = SyncFileContext {
                        config: &config,
                        chunker,
                        embedder,
                        store,
                        sync_token: &sync_token,
                    };
                    index_file(&self.root, path, context, &mut result).await?;
                }
                FileChange::Modified(path) => {
                    result.files_modified += 1;
                    let context = SyncFileContext {
                        config: &config,
                        chunker,
                        embedder,
                        store,
                        sync_token: &sync_token,
                    };
                    index_file(&self.root, path, context, &mut result).await?;
                }
                FileChange::Deleted(path) => {
                    result.files_deleted += 1;
                    let deleted = store.delete_by_filter(&Filter::And(vec![
                        Filter::Eq("source_id".to_string(), config.source_id.clone()),
                        Filter::Eq("path".to_string(), path.clone()),
                    ]))?;
                    result.chunks_deleted += deleted;
                    result.files_processed += 1;
                    result.files_processed_paths.push(path.clone());
                }
            }
        }

        if full {
            result.chunks_deleted += store.delete_by_filter(&Filter::And(vec![
                Filter::Eq("source_id".to_string(), config.source_id.clone()),
                Filter::Neq("seen_token".to_string(), sync_token.clone()),
            ]))?;
        }

        store.flush()?;
        Cursor {
            source_id: config.source_id.clone(),
            last_synced_at: chrono::Utc::now().to_rfc3339(),
            manifest_hash: new_manifest.manifest_hash(),
            chunk_schema_id: chunker.schema_id().to_string(),
            embedder_id: embedder.id().to_string(),
            collection_path: config.collection.path.clone(),
            lexical_language: config.lexical.language.clone(),
        }
        .save(&cursor_path)?;
        new_manifest.save(&manifest_path)?;
        Transaction::remove(&txn_path)?;

        result.elapsed_seconds = start.elapsed().as_secs_f64();
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunker::chonkie::ChonkieChunker;
    use crate::embedder::Embedder;
    use crate::sync::indexer::{index_file, SyncFileContext};
    use crate::vectorstore::memory::InMemoryStore;
    use tempfile::TempDir;

    struct MockEmbedder;

    #[async_trait::async_trait]
    impl Embedder for MockEmbedder {
        fn id(&self) -> &str {
            "mock"
        }

        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|text| {
                    let hash = crate::id::content_hash(text);
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

    #[test]
    fn test_init_creates_minsync_dir() {
        let (_dir, sync, _chunker, _embedder, _store) = fixture();

        let config = sync
            .init(false, "openai:text-embedding-3-small", "recursive")
            .expect("init succeeds");

        assert!(sync.minsync_dir.exists());
        assert!(sync.minsync_dir.join("config.toml").exists());
        assert!(sync.minsync_dir.join("manifest.json").exists());
        assert_eq!(config.version, 1);
    }

    #[test]
    fn test_init_already_initialized() {
        let (_dir, sync, _chunker, _embedder, _store) = fixture();
        sync.init(false, "openai:text-embedding-3-small", "recursive")
            .expect("first init succeeds");

        let result = sync.init(false, "openai:text-embedding-3-small", "recursive");

        assert!(matches!(result, Err(MinSyncError::AlreadyInitialized)));
    }

    #[test]
    fn test_init_force_reinit() {
        let (_dir, sync, _chunker, _embedder, _store) = fixture();
        let first = sync
            .init(false, "openai:text-embedding-3-small", "recursive")
            .expect("first init succeeds");

        let second = sync
            .init(true, "openai:text-embedding-3-small", "recursive")
            .expect("force init succeeds");

        assert_ne!(first.source_id, second.source_id);
    }

    #[test]
    fn test_init_force_removes_previous_store_and_cursor() {
        let (dir, sync, _chunker, _embedder, _store) = fixture();
        sync.init(false, "openai:text-embedding-3-small", "recursive")
            .expect("first init succeeds");
        std::fs::create_dir_all(sync.minsync_dir.join("store"))
            .expect("create previous vector store");
        std::fs::write(sync.minsync_dir.join("store/data"), "stale")
            .expect("write previous vector store data");
        std::fs::write(sync.minsync_dir.join("cursor.json"), "{}").expect("write stale cursor");
        std::fs::write(sync.minsync_dir.join("txn.json"), "{}").expect("write stale transaction");

        sync.init(true, "openai:text-embedding-3-small", "recursive")
            .expect("force init succeeds");

        assert!(!dir.path().join(".minsync/store").exists());
        assert!(!dir.path().join(".minsync/cursor.json").exists());
        assert!(!dir.path().join(".minsync/txn.json").exists());
    }

    #[test]
    fn test_init_force_allows_lancedb_dimension_change() {
        let (_dir, sync, _chunker, _embedder, _store) = fixture();
        let mut first = sync
            .init(false, "openai:text-embedding-3-small", "recursive")
            .expect("first init succeeds");
        first.vectorstore.options["dimension"] = toml::Value::Integer(4);
        first
            .save(&sync.minsync_dir.join("config.toml"))
            .expect("save first dimension");
        let store_path = sync.minsync_dir.join(&first.collection.path);
        let store = crate::vectorstore::lancedb_store::LanceDbStore::open_or_create(&store_path, 4)
            .expect("create first-dimension store");
        drop(store);

        let mut second = sync
            .init(true, "openai:text-embedding-3-small", "recursive")
            .expect("force init succeeds");
        second.vectorstore.options["dimension"] = toml::Value::Integer(8);
        second
            .save(&sync.minsync_dir.join("config.toml"))
            .expect("save second dimension");
        let reopened = crate::vectorstore::lancedb_store::LanceDbStore::open_or_create(
            &sync.minsync_dir.join(&second.collection.path),
            8,
        )
        .expect("open second-dimension store");

        assert_eq!(reopened.dimension(), 8);
        assert_eq!(reopened.doc_count(), 0);
    }

    #[test]
    fn test_init_force_rejects_collection_path_outside_minsync() {
        let (dir, sync, _chunker, _embedder, _store) = fixture();
        let mut first = sync
            .init(false, "openai:text-embedding-3-small", "recursive")
            .expect("first init succeeds");
        first.collection.path = "../outside".to_string();
        first
            .save(&sync.minsync_dir.join("config.toml"))
            .expect("save unsafe collection path");
        let outside = dir.path().join("outside");
        std::fs::create_dir_all(&outside).expect("create outside directory");
        std::fs::write(outside.join("data"), "keep").expect("write outside data");

        let error = sync
            .init(true, "openai:text-embedding-3-small", "recursive")
            .expect_err("unsafe collection path must fail");

        assert!(error.to_string().contains("must stay inside .minsync"));
        assert!(outside.join("data").exists());
    }

    #[cfg(unix)]
    #[test]
    fn test_init_force_rejects_intermediate_symlink_escape() {
        use std::os::unix::fs::symlink;

        let (dir, sync, _chunker, _embedder, _store) = fixture();
        let mut first = sync
            .init(false, "openai:text-embedding-3-small", "recursive")
            .expect("first init succeeds");
        first.collection.path = "escape/store".to_string();
        first
            .save(&sync.minsync_dir.join("config.toml"))
            .expect("save symlink path");

        let outside = dir.path().join("outside");
        std::fs::create_dir_all(outside.join("store")).expect("create outside store");
        std::fs::write(outside.join("store/data"), "keep").expect("write outside data");
        symlink(&outside, sync.minsync_dir.join("escape")).expect("create escape symlink");

        let error = sync
            .init(true, "openai:text-embedding-3-small", "recursive")
            .expect_err("symlink collection path must fail");

        assert!(error.to_string().contains("must stay inside .minsync"));
        assert!(outside.join("store/data").exists());
    }

    #[test]
    fn test_init_force_respects_workspace_lock() {
        let (_dir, sync, _chunker, _embedder, _store) = fixture();
        sync.init(false, "openai:text-embedding-3-small", "recursive")
            .expect("first init succeeds");
        let _lock =
            FileLock::acquire(&sync.minsync_dir.join("lock"), false).expect("hold workspace lock");

        let error = sync
            .init(true, "openai:text-embedding-3-small", "recursive")
            .expect_err("force init must reject a locked workspace");

        assert!(matches!(error, MinSyncError::LockFailed));
    }

    #[test]
    fn test_init_honors_overrides() {
        let (_dir, sync, _chunker, _embedder, _store) = fixture();

        let config = sync
            .init(false, "tei:intfloat/multilingual-e5-small", "chonkie")
            .expect("init succeeds");

        let saved = Config::load(&sync.minsync_dir.join("config.toml"))
            .expect("load saved config succeeds");

        assert_eq!(config.embedder.id, "tei:intfloat/multilingual-e5-small");
        assert_eq!(config.chunker.id, "chonkie");
        assert_eq!(saved.embedder.id, "tei:intfloat/multilingual-e5-small");
        assert_eq!(saved.chunker.id, "chonkie");
    }

    #[tokio::test]
    async fn test_sync_full_first_time() {
        let (dir, sync, chunker, embedder, mut store) = fixture();
        std::fs::write(dir.path().join("a.txt"), "alpha beta gamma").expect("write file");
        sync.init(false, "openai:text-embedding-3-small", "recursive")
            .expect("init succeeds");

        let result = sync
            .sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("sync succeeds");

        assert_eq!(result.files_processed, 1);
        assert!(result.chunks_added > 0);
        assert_eq!(store.doc_count(), result.chunks_added);
    }

    #[tokio::test]
    async fn test_sync_incremental_add() {
        let (dir, sync, chunker, embedder, mut store) = fixture();
        std::fs::write(dir.path().join("a.txt"), "alpha beta gamma").expect("write file");
        sync.init(false, "openai:text-embedding-3-small", "recursive")
            .expect("init succeeds");
        sync.sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("first sync succeeds");

        std::fs::write(dir.path().join("b.txt"), "delta epsilon zeta").expect("write new file");
        let result = sync
            .sync(&chunker, &embedder, &mut store, false, false, false)
            .await
            .expect("incremental sync succeeds");

        assert_eq!(result.files_processed_paths, vec!["b.txt"]);
        assert!(result.chunks_added > 0);
        assert_eq!(store.all_paths(), vec!["a.txt", "b.txt"]);
    }

    #[tokio::test]
    async fn test_sync_incremental_modify() {
        let (dir, sync, chunker, embedder, mut store) = fixture();
        std::fs::write(dir.path().join("a.txt"), "alpha beta gamma").expect("write file");
        sync.init(false, "openai:text-embedding-3-small", "recursive")
            .expect("init succeeds");
        sync.sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("first sync succeeds");

        std::fs::write(dir.path().join("a.txt"), "alpha beta gamma delta").expect("modify file");
        let result = sync
            .sync(&chunker, &embedder, &mut store, false, false, false)
            .await
            .expect("modify sync succeeds");

        assert_eq!(result.files_processed_paths, vec!["a.txt"]);
        assert!(result.chunks_added + result.chunks_updated > 0);
    }

    #[tokio::test]
    async fn test_sync_incremental_delete() {
        let (dir, sync, chunker, embedder, mut store) = fixture();
        let path = dir.path().join("a.txt");
        std::fs::write(&path, "alpha beta gamma").expect("write file");
        sync.init(false, "openai:text-embedding-3-small", "recursive")
            .expect("init succeeds");
        sync.sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("first sync succeeds");
        assert!(store.doc_count() > 0);

        std::fs::remove_file(path).expect("delete file");
        let result = sync
            .sync(&chunker, &embedder, &mut store, false, false, false)
            .await
            .expect("delete sync succeeds");

        assert_eq!(result.files_processed_paths, vec!["a.txt"]);
        assert!(result.chunks_deleted > 0);
        assert_eq!(store.doc_count(), 0);
    }

    #[tokio::test]
    async fn test_full_sync_sweeps_stale_deleted_paths() {
        let (dir, sync, chunker, embedder, mut store) = fixture();
        let path = dir.path().join("a.txt");
        std::fs::write(&path, "alpha beta gamma").expect("write file");
        sync.init(false, "openai:text-embedding-3-small", "recursive")
            .expect("init succeeds");
        sync.sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("first sync succeeds");

        std::fs::remove_file(path).expect("delete file");
        std::fs::write(dir.path().join("b.txt"), "delta epsilon zeta").expect("write b");
        let result = sync
            .sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("full sync succeeds");

        assert!(result.chunks_deleted > 0);
        assert_eq!(store.all_paths(), vec!["b.txt"]);
    }

    #[tokio::test]
    async fn test_index_file_missing_path_deletes_existing_chunks() {
        let (dir, _sync, chunker, embedder, mut store) = fixture();
        let config = Config::default_for("source-1");
        store
            .upsert(&[crate::vectorstore::Document {
                id: "doc-1".to_string(),
                embedding: vec![1.0],
                text: "stale".to_string(),
                source_id: config.source_id.clone(),
                path: "missing.txt".to_string(),
                chunk_schema_id: chunker.schema_id().to_string(),
                chunk_type: "text".to_string(),
                heading_path: String::new(),
                content_hash: "sha256:stale".to_string(),
                seen_token: "old".to_string(),
            }])
            .expect("upsert stale doc");
        let mut result = empty_sync_result(false, false);
        let context = SyncFileContext {
            config: &config,
            chunker: &chunker,
            embedder: &embedder,
            store: &mut store,
            sync_token: "new-token",
        };

        index_file(dir.path(), "missing.txt", context, &mut result)
            .await
            .expect("missing file is tolerated");

        assert_eq!(result.files_processed, 1);
        assert_eq!(result.files_processed_paths, vec!["missing.txt"]);
        assert_eq!(result.chunks_deleted, 1);
        assert_eq!(store.doc_count(), 0);
    }

    #[tokio::test]
    async fn test_sync_already_up_to_date() {
        let (dir, sync, chunker, embedder, mut store) = fixture();
        std::fs::write(dir.path().join("a.txt"), "alpha beta gamma").expect("write file");
        sync.init(false, "openai:text-embedding-3-small", "recursive")
            .expect("init succeeds");
        sync.sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("first sync succeeds");

        let result = sync
            .sync(&chunker, &embedder, &mut store, false, false, false)
            .await
            .expect("second sync succeeds");

        assert!(result.already_up_to_date);
        assert_eq!(result.files_processed, 0);
        assert_eq!(result.files_checked, 1);
        assert!(result.freshness_check_only);
        assert!(result.query_ready);
        assert!(result.elapsed_seconds > 0.0);
    }

    #[tokio::test]
    async fn test_sync_noop_reports_freshness_check_elapsed_time() {
        let (dir, sync, chunker, embedder, mut store) = fixture();
        std::fs::write(dir.path().join("a.txt"), "alpha beta gamma").expect("write file");
        sync.init(false, "openai:text-embedding-3-small", "recursive")
            .expect("init succeeds");
        sync.sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("first sync succeeds");

        let result = sync
            .sync(&chunker, &embedder, &mut store, false, false, false)
            .await
            .expect("no-op sync succeeds");

        assert!(result.already_up_to_date);
        assert!(
            result.elapsed_seconds > 0.0,
            "no-op sync must report freshness-check time"
        );
    }

    #[tokio::test]
    async fn test_sync_dry_run() {
        let (dir, sync, chunker, embedder, mut store) = fixture();
        std::fs::write(dir.path().join("a.txt"), "alpha beta gamma").expect("write file");
        sync.init(false, "openai:text-embedding-3-small", "recursive")
            .expect("init succeeds");

        let result = sync
            .sync(&chunker, &embedder, &mut store, true, true, false)
            .await
            .expect("dry run succeeds");

        assert!(result.dry_run);
        assert_eq!(result.files_processed_paths, vec!["a.txt"]);
        assert_eq!(store.doc_count(), 0);
        assert!(!sync.minsync_dir.join("cursor.json").exists());
    }

    #[tokio::test]
    async fn test_sync_crash_recovery() {
        let (dir, sync, chunker, embedder, mut store) = fixture();
        std::fs::write(dir.path().join("a.txt"), "alpha beta gamma").expect("write file");
        let config = sync
            .init(false, "openai:text-embedding-3-small", "recursive")
            .expect("init succeeds");
        Transaction::new(&config.source_id, "stale-token", None, "sha256:stale")
            .save(&sync.minsync_dir.join("txn.json"))
            .expect("write stale txn");

        let result = sync
            .sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("sync recovers");

        assert!(result.chunks_added > 0);
        assert!(!sync.minsync_dir.join("txn.json").exists());
        assert!(sync.minsync_dir.join("cursor.json").exists());
    }
}
