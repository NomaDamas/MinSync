//! Incremental sync orchestration: locking, transaction lifecycle, manifest
//! diffing, and cursor advancement. Per-file work lives in [`indexer`];
//! result accounting helpers live in [`result`].

mod indexer;
mod progress;
mod reporter;
mod result;

use crate::chunker::Chunker;
use crate::config::{Config, OnFileError};
use crate::embedder::Embedder;
use crate::error::{MinSyncError, Result};
use crate::manifest::{FileChange, Manifest, ScanErrorKind};
use crate::state::{Cursor, FileLock, Transaction};
use crate::types::{FileFailure, FileStage, SyncResult};
use crate::vectorstore::{Filter, VectorStore};
use indexer::{index_batch, BatchItem, BatchKind, SyncFileContext};
use progress::{ProgressHeader, ProgressLog, PROGRESS_FILE};
use result::{change_path, empty_sync_result};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use reporter::RunStart;
pub use reporter::{ProgressReporter, DEFAULT_HEARTBEAT_SECONDS};

pub struct MinSync {
    root: PathBuf,
    minsync_dir: PathBuf,
    /// Overrides `[sync] checkpoint_every_files` (0 disables checkpoints).
    checkpoint_every: Option<usize>,
    /// Machine-readable progress stream (`--progress jsonl`); `None` = off.
    progress: Option<Arc<ProgressReporter>>,
}

impl MinSync {
    pub fn new(root: PathBuf) -> Self {
        let minsync_dir = root.join(".minsync");
        Self {
            root,
            minsync_dir,
            checkpoint_every: None,
            progress: None,
        }
    }

    /// Checkpoint every `files` processed files instead of the configured
    /// `[sync] checkpoint_every_files`. `0` disables checkpoints.
    pub fn with_checkpoint_every(mut self, files: usize) -> Self {
        self.checkpoint_every = Some(files);
        self
    }

    /// Stream machine-readable progress events (`start`, `batch`,
    /// `heartbeat`, `done`) to `reporter` while syncing.
    pub fn with_progress(mut self, reporter: Arc<ProgressReporter>) -> Self {
        self.progress = Some(reporter);
        self
    }

    pub fn minsync_dir(&self) -> &std::path::Path {
        &self.minsync_dir
    }

    /// Initialize `.minsync/` with a default config and baseline manifest.
    pub fn init(&self, force: bool, embedder_id: &str, chunker_id: &str) -> Result<Config> {
        if self.minsync_dir.exists() && !force {
            return Err(MinSyncError::AlreadyInitialized);
        }

        if force {
            std::fs::create_dir_all(&self.minsync_dir)?;
        }
        let _lock = force
            .then(|| FileLock::acquire(&self.minsync_dir.join("lock"), false))
            .transpose()?;
        let reset_staging = if force {
            Some(self.reset_forced_state()?)
        } else {
            None
        };
        std::fs::create_dir_all(&self.minsync_dir)?;
        let source_id = uuid::Uuid::new_v4().to_string();
        let mut config = Config::default_for(&source_id);
        config.embedder.id = embedder_id.to_string();
        config.chunker.id = chunker_id.to_string();
        if embedder_id == crate::config::EMBEDDINGGEMMA_ID {
            config.embedder.query_prefix =
                Some(crate::config::EMBEDDINGGEMMA_QUERY_PREFIX.to_string());
            config.embedder.passage_prefix =
                Some(crate::config::EMBEDDINGGEMMA_PASSAGE_PREFIX.to_string());
        } else {
            config.embedder.query_prefix = None;
            config.embedder.passage_prefix = None;
        }
        if let Some(dimension) = crate::config::known_embedding_dimension(embedder_id) {
            config.vectorstore.options["dimension"] = toml::Value::Integer(dimension as i64);
        }

        config.save(&self.minsync_dir.join("config.toml"))?;
        // Baseline only (the first sync is always full): a file that cannot
        // be read here is skipped and reported by that sync, not fatal.
        Manifest::scan_with_baseline_stats_and_max_file_bytes(
            &self.root,
            &source_id,
            None,
            config.sync.max_file_bytes,
        )?
        .0
        .save(&self.minsync_dir.join("manifest.json"))?;
        if let Some(staging) = reset_staging {
            std::fs::remove_dir_all(staging)?;
        }

        Ok(config)
    }

    fn reset_forced_state(&self) -> Result<PathBuf> {
        let config_path = self.minsync_dir.join("config.toml");
        let previous = if config_path.exists() {
            Some(Config::load(&config_path)?)
        } else {
            None
        };
        let staging = self
            .minsync_dir
            .join(format!(".reset-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir(&staging)?;

        for state_file in ["cursor.json", "txn.json", progress::PROGRESS_FILE] {
            let path = self.minsync_dir.join(state_file);
            if path.exists() {
                std::fs::rename(&path, staging.join(state_file))?;
            }
        }
        let previous_store_path = previous
            .map(|config| config.collection.path)
            .unwrap_or_else(|| "store".to_string());
        let previous_store = collection_store_path(&self.minsync_dir, &previous_store_path)?;
        if previous_store.exists() {
            std::fs::rename(previous_store, staging.join("store"))?;
        }
        Ok(staging)
    }
}

/// Fingerprint of everything that changes what a processed file writes to
/// the store. A resume log is only trusted when this matches, so editing the
/// chunker/normalizer/embedder/language between a crash and the retry
/// re-indexes instead of silently keeping rows built the old way.
fn config_fingerprint(config: &Config, chunker: &dyn Chunker, embedder: &dyn Embedder) -> String {
    let material = serde_json::json!({
        "chunker": &config.chunker,
        "chunker_schema": chunker.schema_id(),
        "normalize": &config.normalize,
        "embedder": &config.embedder,
        "embedder_id": embedder.id(),
        "lexical": &config.lexical,
        "vectorstore": &config.vectorstore,
        "collection": &config.collection,
    });
    format!(
        "sha256:{}",
        hex::encode(Sha256::digest(material.to_string()))
    )
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
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
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
        let outcome = self.sync_run(chunker, embedder, store, full, dry_run).await;
        if outcome.is_err() {
            // No `done` for a failed run; just stop the heartbeat.
            if let Some(progress) = &self.progress {
                progress.abort();
            }
        }
        outcome
    }

    async fn sync_run(
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
        let progress_path = self.minsync_dir.join(PROGRESS_FILE);

        let stored_manifest = if manifest_path.exists() {
            Some(Manifest::load(&manifest_path)?)
        } else {
            None
        };
        let stored_cursor = if cursor_path.exists() {
            Some(Cursor::load(&cursor_path)?)
        } else {
            None
        };
        // A baseline manifest from `init` is not evidence that the index was
        // populated. Trust the pair only when the cursor describes the exact
        // manifest on disk; a crash between writing either file forces a safe
        // full sync on the next run.
        let has_completed_sync = matches!(
            (&stored_cursor, &stored_manifest),
            (Some(cursor), Some(manifest))
                if cursor.source_id == config.source_id
                    && manifest.source_id == config.source_id
                    && cursor.manifest_hash == manifest.manifest_hash()
        );
        let initial_sync = !has_completed_sync;
        let stored_lexical_language = stored_cursor
            .as_ref()
            .filter(|_| has_completed_sync)
            .map(|cursor| cursor.lexical_language.clone())
            .unwrap_or_else(|| config.lexical.language.clone());
        let lexical_rebuild = stored_lexical_language != config.lexical.language;
        let full = full || initial_sync || lexical_rebuild;

        // Read the interrupted run's log BEFORE clearing its txn marker.
        let resume_state = ProgressLog::load(&progress_path);
        if txn_path.exists() {
            Transaction::remove(&txn_path)?;
        }

        let old_manifest = if full {
            Manifest::new(&config.source_id)
        } else {
            stored_manifest
                .clone()
                .unwrap_or_else(|| Manifest::new(&config.source_id))
        };
        let scan_baseline = if full { None } else { stored_manifest.as_ref() };
        let start = std::time::Instant::now();
        let skip_failures = config.sync.on_file_error == OnFileError::Skip;
        // Entries the scan could not inspect are carried over from the stored
        // manifest: unknown, never deleted (issue #79).
        let (new_manifest, scan_stats, unknown_paths) =
            Manifest::scan_keeping_unknown_with_max_file_bytes(
                &self.root,
                &config.source_id,
                scan_baseline,
                stored_manifest.as_ref(),
                config.sync.max_file_bytes,
            )?;
        let scan_errors = scan_stats.errors;
        if !skip_failures {
            // `on_file_error = "fail"`: abort on the first unreadable entry,
            // naming it.
            if let Some(first) = scan_errors.first() {
                return Err(MinSyncError::Manifest(format!(
                    "{}: {}",
                    first.path, first.error
                )));
            }
        }
        // `on_file_error = "skip"`: an unreadable entry is also a skipped
        // file, so the run is `partial` (exit 6) and the next sync retries it.
        let scan_failures: Vec<FileFailure> = if skip_failures {
            scan_errors
                .iter()
                .map(|error| FileFailure {
                    path: error.path.clone(),
                    stage: if error.kind == ScanErrorKind::SizeGuard {
                        FileStage::SizeGuard
                    } else {
                        FileStage::Read
                    },
                    error: error.error.clone(),
                })
                .collect()
        } else {
            Vec::new()
        };
        let unknown_set: HashSet<&str> = unknown_paths.iter().map(String::as_str).collect();
        let mut changes = Manifest::diff(&old_manifest, &new_manifest);
        changes.retain(|change| !unknown_set.contains(change_path(change).as_str()));
        // A sweep must spare rows of paths this run could not inspect and of
        // files it skipped (their old rows stay searchable until they succeed).
        let sweep_filter = |sync_token: &str, failed: &[FileFailure]| {
            let mut filters = vec![
                Filter::Eq("source_id".to_string(), config.source_id.clone()),
                Filter::Neq("seen_token".to_string(), sync_token.to_string()),
            ];
            if !unknown_paths.is_empty() {
                filters.push(Filter::NotIn("path".to_string(), unknown_paths.clone()));
            }
            filters.extend(
                failed
                    .iter()
                    .map(|failure| Filter::Neq("path".to_string(), failure.path.clone())),
            );
            Filter::And(filters)
        };

        if changes.is_empty() && !full {
            if let Some(progress) = &self.progress {
                progress.start(RunStart {
                    started: start,
                    total: 0,
                    resumed: 0,
                    full,
                    dry_run,
                    sync_token: None,
                });
                for failure in &scan_failures {
                    progress.file_failed(failure);
                }
            }
            let mut result = empty_sync_result(dry_run, true);
            result.partial = !scan_failures.is_empty();
            result.files_failed = scan_failures;
            result.files_checked = new_manifest.files.len();
            result.query_ready = cursor_path.exists();
            result.scan_errors = scan_errors;
            result.elapsed_seconds = start.elapsed().as_secs_f64();
            if let Some(progress) = &self.progress {
                progress.done(&result, 0);
            }
            return Ok(result);
        }

        if dry_run {
            if let Some(progress) = &self.progress {
                progress.start(RunStart {
                    started: start,
                    total: changes.len(),
                    resumed: 0,
                    full,
                    dry_run: true,
                    sync_token: None,
                });
                for failure in &scan_failures {
                    progress.file_failed(failure);
                }
            }
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
            result.partial = !scan_failures.is_empty();
            result.files_failed = scan_failures;
            result.files_checked = new_manifest.files.len();
            result.scan_errors = scan_errors;
            result.elapsed_seconds = start.elapsed().as_secs_f64();
            if let Some(progress) = &self.progress {
                progress.done(&result, changes.len());
            }
            return Ok(result);
        }

        // Resuming an interrupted run: reuse its sync token (the end-of-run
        // full sweep deletes rows whose `seen_token` differs, so files the
        // earlier run already committed must keep carrying it) and skip the
        // files it logged as done.
        let fingerprint = config_fingerprint(&config, chunker, embedder);
        let resume = resume_state.filter(|state| {
            state.header.source_id == config.source_id
                && state.header.full == full
                && state.header.manifest_hash_from == old_manifest.manifest_hash()
                && state.header.config_fingerprint == fingerprint
        });
        let sync_token = match &resume {
            Some(state) => state.header.sync_token.clone(),
            None => uuid::Uuid::new_v4().to_string().replace('-', ""),
        };
        Transaction::new(
            &config.source_id,
            &sync_token,
            Some(old_manifest.manifest_hash()),
            &new_manifest.manifest_hash(),
        )
        .save(&txn_path)?;
        let header = ProgressHeader {
            source_id: config.source_id.clone(),
            sync_token: sync_token.clone(),
            full,
            manifest_hash_from: old_manifest.manifest_hash(),
            config_fingerprint: fingerprint,
        };
        let carried = resume
            .as_ref()
            .map(|state| state.processed.clone())
            .unwrap_or_default();
        let mut progress = ProgressLog::create(&progress_path, &header, &carried)?;

        let mut result = empty_sync_result(false, false);
        result.files_failed = scan_failures;
        result.initial_sync = initial_sync;
        result.files_checked = new_manifest.files.len();
        result.scan_errors = scan_errors;
        result.query_ready = true;
        if lexical_rebuild {
            // Rows indexed under the previous lexical language must go. Keyed
            // on the token so a resumed run keeps what it already rewrote.
            result.chunks_deleted +=
                store.delete_by_filter(&sweep_filter(&sync_token, &result.files_failed))?;
        }

        let checkpoint_every = self
            .checkpoint_every
            .unwrap_or(config.sync.checkpoint_every_files);
        let batch_files = config.sync.batch_files.max(1);

        // Classify every change once: count it, and queue what still has to
        // be written (resumed files are already committed).
        let mut pending: Vec<(BatchItem, Option<String>)> = Vec::new();
        for change in &changes {
            let (path, kind, content_hash) = match change {
                FileChange::Added(path) => {
                    result.files_added += 1;
                    (
                        path,
                        BatchKind::Index,
                        new_manifest.files.get(path).map(|e| e.content_hash.clone()),
                    )
                }
                FileChange::Modified(path) => {
                    result.files_modified += 1;
                    (
                        path,
                        BatchKind::Index,
                        new_manifest.files.get(path).map(|e| e.content_hash.clone()),
                    )
                }
                FileChange::Deleted(path) => {
                    result.files_deleted += 1;
                    (path, BatchKind::Delete, None)
                }
            };
            if resume
                .as_ref()
                .is_some_and(|state| state.is_done(path, content_hash.as_deref()))
            {
                result.files_resumed += 1;
                continue;
            }
            pending.push((
                BatchItem {
                    path: path.clone(),
                    kind,
                    // Logged by the interrupted run but changed since: its old
                    // rows carry the reused token, so the stale sweep cannot
                    // tell them from fresh ones. Clear the path first.
                    purge_first: resume
                        .as_ref()
                        .is_some_and(|state| state.processed.contains_key(path)),
                },
                content_hash,
            ));
        }

        if let Some(progress) = &self.progress {
            progress.start(RunStart {
                started: start,
                total: changes.len(),
                resumed: result.files_resumed,
                full,
                dry_run: false,
                sync_token: Some(&sync_token),
            });
            for failure in &result.files_failed {
                progress.file_failed(failure);
            }
        }

        let mut processed_this_run = 0usize;
        let mut since_checkpoint = 0usize;
        let mut cursor_at = 0usize;
        while cursor_at < pending.len() {
            // Never let a batch straddle a checkpoint boundary, so the
            // checkpoint cadence stays exactly `checkpoint_every` files.
            let room = if checkpoint_every > 0 {
                checkpoint_every - since_checkpoint
            } else {
                usize::MAX
            };
            let take = batch_files.min(room).min(pending.len() - cursor_at);
            let group = &pending[cursor_at..cursor_at + take];
            cursor_at += take;

            let items: Vec<BatchItem> = group
                .iter()
                .map(|(item, _)| BatchItem {
                    path: item.path.clone(),
                    kind: item.kind,
                    purge_first: item.purge_first,
                })
                .collect();
            let context = SyncFileContext {
                config: &config,
                chunker,
                embedder,
                store,
                sync_token: &sync_token,
            };
            let failed_before = result.files_failed.len();
            index_batch(&self.root, &items, context, &mut result).await?;
            if let Some(progress) = &self.progress {
                for failure in &result.files_failed[failed_before..] {
                    progress.file_failed(failure);
                }
            }

            // Logged only after the store commits returned: a logged path is
            // always safe to skip on resume. One fsync for the whole batch.
            // Failed files stay unlogged so a resumed run retries them.
            let failed_now: HashSet<&str> = result.files_failed[failed_before..]
                .iter()
                .map(|failure| failure.path.as_str())
                .collect();
            let logged: Vec<(String, Option<String>)> = group
                .iter()
                .filter(|(item, _)| !failed_now.contains(item.path.as_str()))
                .map(|(item, hash)| (item.path.clone(), hash.clone()))
                .collect();
            progress.append(&logged)?;

            processed_this_run += take;
            since_checkpoint += take;
            if let Some(progress) = &self.progress {
                if let (Some(first), Some(last)) = (group.first(), group.last()) {
                    progress.batch(
                        processed_this_run + result.files_resumed,
                        &first.0.path,
                        &last.0.path,
                    );
                }
            }
            if checkpoint_every > 0 && since_checkpoint >= checkpoint_every {
                since_checkpoint = 0;
                store.checkpoint()?;
                tracing::info!(
                    processed = processed_this_run + result.files_resumed,
                    total = changes.len(),
                    elapsed_seconds = start.elapsed().as_secs(),
                    "sync progress"
                );
            }
        }

        if full {
            result.chunks_deleted +=
                store.delete_by_filter(&sweep_filter(&sync_token, &result.files_failed))?;
        }

        store.flush()?;

        // A skipped file must be retried by the next sync, so the saved
        // manifest keeps its previous entry (or none, if it was never
        // indexed) instead of recording it as done.
        let mut saved_manifest = new_manifest.clone();
        // Only a matching cursor/manifest pair proves that stored entries
        // correspond to indexed chunks. In particular, `init`'s baseline and
        // a manifest left behind by an interrupted commit are not trusted.
        let prior_indexed_manifest = if has_completed_sync {
            stored_manifest.as_ref()
        } else {
            None
        };
        for failure in &result.files_failed {
            match prior_indexed_manifest.and_then(|manifest| manifest.files.get(&failure.path)) {
                Some(entry) => {
                    saved_manifest
                        .files
                        .insert(failure.path.clone(), entry.clone());
                }
                None => {
                    saved_manifest.files.remove(&failure.path);
                }
            }
        }
        result.partial = !result.files_failed.is_empty();
        saved_manifest.save(&manifest_path)?;
        Cursor {
            source_id: config.source_id.clone(),
            last_synced_at: chrono::Utc::now().to_rfc3339(),
            manifest_hash: saved_manifest.manifest_hash(),
            chunk_schema_id: chunker.schema_id().to_string(),
            embedder_id: embedder.id().to_string(),
            collection_path: config.collection.path.clone(),
            // Repeat a partial lexical rebuild: a failed file may have lost
            // its old rows during the pre-sweep before indexing failed.
            lexical_language: if lexical_rebuild && !result.files_failed.is_empty() {
                stored_lexical_language.clone()
            } else {
                config.lexical.language.clone()
            },
        }
        .save(&cursor_path)?;
        ProgressLog::remove(&progress_path)?;
        Transaction::remove(&txn_path)?;

        result.elapsed_seconds = start.elapsed().as_secs_f64();
        if let Some(progress) = &self.progress {
            progress.done(&result, changes.len());
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunker::chonkie::ChonkieChunker;
    use crate::embedder::Embedder;
    use crate::sync::indexer::{index_batch, BatchItem, BatchKind, SyncFileContext};
    use crate::vectorstore::memory::InMemoryStore;
    use tempfile::TempDir;

    /// Single-file convenience over [`index_batch`] for per-file tests.
    async fn index_file(
        root: &Path,
        path: &str,
        context: SyncFileContext<'_>,
        result: &mut SyncResult,
    ) -> Result<()> {
        let item = BatchItem {
            path: path.to_string(),
            kind: BatchKind::Index,
            purge_first: false,
        };
        index_batch(root, &[item], context, result).await
    }

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
    fn test_init_applies_native_qwen3_defaults() {
        let (_dir, sync, _chunker, _embedder, _store) = fixture();

        let config = sync
            .init(false, crate::config::DEFAULT_EMBEDDER_ID, "recursive")
            .expect("init succeeds");

        assert_eq!(config.embedder.id, "native:Qwen/Qwen3-Embedding-0.6B");
        assert_eq!(config.embedder.query_prefix, None);
        assert_eq!(config.embedder.passage_prefix, None);
        assert_eq!(
            config.vectorstore.options["dimension"].as_integer(),
            Some(1024)
        );
    }

    #[test]
    fn test_init_embeddinggemma_keeps_prefixes_and_dimension() {
        let (_dir, sync, _chunker, _embedder, _store) = fixture();

        let config = sync
            .init(false, "tei:google/embeddinggemma-300m", "recursive")
            .expect("init succeeds");

        assert_eq!(
            config.embedder.query_prefix.as_deref(),
            Some(crate::config::EMBEDDINGGEMMA_QUERY_PREFIX)
        );
        assert_eq!(
            config.embedder.passage_prefix.as_deref(),
            Some(crate::config::EMBEDDINGGEMMA_PASSAGE_PREFIX)
        );
        assert_eq!(
            config.vectorstore.options["dimension"].as_integer(),
            Some(768)
        );
    }

    #[test]
    fn test_init_maps_known_embedder_dimension_and_clears_prefixes() {
        let (_dir, sync, _chunker, _embedder, _store) = fixture();

        let config = sync
            .init(false, "openai:text-embedding-3-small", "recursive")
            .expect("init succeeds");

        assert_eq!(
            config.vectorstore.options["dimension"].as_integer(),
            Some(1536)
        );
        assert_eq!(config.embedder.query_prefix, None);
        assert_eq!(config.embedder.passage_prefix, None);
    }

    #[test]
    fn test_init_unknown_embedder_keeps_default_dimension() {
        let (_dir, sync, _chunker, _embedder, _store) = fixture();

        let config = sync
            .init(false, "tei:custom/model", "recursive")
            .expect("init succeeds");

        assert_eq!(
            config.vectorstore.options["dimension"].as_integer(),
            Some(1024)
        );
        assert_eq!(config.embedder.query_prefix, None);
        assert_eq!(config.embedder.passage_prefix, None);
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
    fn test_init_force_creates_missing_workspace_before_locking() {
        let (_dir, sync, _chunker, _embedder, _store) = fixture();

        sync.init(true, "openai:text-embedding-3-small", "recursive")
            .expect("force init succeeds in a new workspace");
    }

    #[test]
    fn test_init_force_rejects_parent_escape_after_missing_component() {
        let (dir, sync, _chunker, _embedder, _store) = fixture();
        sync.init(false, "openai:text-embedding-3-small", "recursive")
            .expect("first init succeeds");
        let mut config =
            Config::load(&sync.minsync_dir.join("config.toml")).expect("load initial config");
        config.collection.path = "missing/../../outside".to_string();
        config
            .save(&sync.minsync_dir.join("config.toml"))
            .expect("save unsafe collection path");

        let error = sync
            .init(true, "openai:text-embedding-3-small", "recursive")
            .expect_err("parent escape after missing component must fail");

        assert!(error.to_string().contains("must stay inside .minsync"));
        assert!(!dir.path().join("outside").exists());
    }

    #[test]
    fn test_init_force_resets_partial_workspace_without_config() {
        let (_dir, sync, _chunker, _embedder, _store) = fixture();
        std::fs::create_dir_all(sync.minsync_dir.join("store")).expect("create stale vector store");
        std::fs::write(sync.minsync_dir.join("store/data"), "stale")
            .expect("write stale vector store data");
        std::fs::write(sync.minsync_dir.join("cursor.json"), "{}").expect("write stale cursor");
        std::fs::write(sync.minsync_dir.join("txn.json"), "{}").expect("write stale transaction");

        sync.init(true, "openai:text-embedding-3-small", "recursive")
            .expect("force init succeeds");

        assert!(!sync.minsync_dir.join("store").exists());
        assert!(!sync.minsync_dir.join("cursor.json").exists());
        assert!(!sync.minsync_dir.join("txn.json").exists());
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

    /// Emits NaN for the first chunk of every multi-text batch, mimicking a
    /// GPU backend that corrupts one row of a large batch.
    struct NanInBatchEmbedder {
        calls: std::sync::atomic::AtomicUsize,
    }

    #[async_trait::async_trait]
    impl Embedder for NanInBatchEmbedder {
        fn id(&self) -> &str {
            "nan-in-batch"
        }

        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let mut vectors = MockEmbedder.embed(texts).await?;
            if texts.len() > 1 {
                vectors[0][3] = f32::NAN;
            }
            Ok(vectors)
        }
    }

    /// Always emits a non-finite vector for texts containing `poison`.
    struct PoisonEmbedder;

    #[async_trait::async_trait]
    impl Embedder for PoisonEmbedder {
        fn id(&self) -> &str {
            "poison"
        }

        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            let mut vectors = MockEmbedder.embed(texts).await?;
            for (text, vector) in texts.iter().zip(vectors.iter_mut()) {
                if text.contains("poison") {
                    vector[0] = f32::INFINITY;
                }
            }
            Ok(vectors)
        }
    }

    #[tokio::test]
    async fn test_index_file_re_embeds_non_finite_chunk_alone() {
        let (dir, _sync, chunker, _embedder, mut store) = fixture();
        let config = Config::default_for("source-1");
        std::fs::write(
            dir.path().join("a.txt"),
            "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron",
        )
        .expect("write file");
        let embedder = NanInBatchEmbedder {
            calls: std::sync::atomic::AtomicUsize::new(0),
        };
        let mut result = empty_sync_result(false, false);
        let context = SyncFileContext {
            config: &config,
            chunker: &chunker,
            embedder: &embedder,
            store: &mut store,
            sync_token: "token",
        };

        index_file(dir.path(), "a.txt", context, &mut result)
            .await
            .expect("non-finite chunk is repaired by a single-chunk retry");

        assert!(
            result.chunks_added > 1,
            "fixture must produce a multi-chunk batch"
        );
        assert_eq!(embedder.calls.load(std::sync::atomic::Ordering::SeqCst), 2);
        assert_eq!(store.doc_count(), result.chunks_added);
        assert!(store
            .all_embeddings()
            .iter()
            .all(|embedding| embedding.iter().all(|value| value.is_finite())));
    }

    #[tokio::test]
    async fn test_index_file_reports_path_for_persistent_non_finite_chunk() {
        let (dir, _sync, chunker, _embedder, mut store) = fixture();
        let mut config = Config::default_for("source-1");
        config.sync.on_file_error = crate::config::OnFileError::Fail;
        std::fs::write(dir.path().join("bad.txt"), "poison").expect("write file");
        let mut result = empty_sync_result(false, false);
        let context = SyncFileContext {
            config: &config,
            chunker: &chunker,
            embedder: &PoisonEmbedder,
            store: &mut store,
            sync_token: "token",
        };

        let error = index_file(dir.path(), "bad.txt", context, &mut result)
            .await
            .expect_err("persistent non-finite vector must fail");

        let message = error.to_string();
        assert!(message.contains("bad.txt"), "{message}");
        assert!(message.contains("chunk 0"), "{message}");
        assert_eq!(store.doc_count(), 0);
    }

    /// Records every `embed` call so tests can assert batching and dedup.
    #[derive(Default)]
    struct RecordingEmbedder {
        calls: std::sync::Mutex<Vec<Vec<String>>>,
    }

    impl RecordingEmbedder {
        fn calls(&self) -> Vec<Vec<String>> {
            self.calls.lock().unwrap().clone()
        }
    }

    #[async_trait::async_trait]
    impl Embedder for RecordingEmbedder {
        fn id(&self) -> &str {
            "recording"
        }

        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            self.calls.lock().unwrap().push(texts.to_vec());
            MockEmbedder.embed(texts).await
        }
    }

    fn init_openai(sync: &MinSync) {
        sync.init(false, "openai:text-embedding-3-small", "recursive")
            .expect("init succeeds");
    }

    #[tokio::test]
    async fn test_sync_embeds_identical_files_once() {
        let (dir, sync, chunker, _embedder, mut store) = fixture();
        let body = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu";
        std::fs::write(dir.path().join("a.txt"), body).expect("write a");
        std::fs::write(dir.path().join("copy-of-a.txt"), body).expect("write copy");
        init_openai(&sync);
        let embedder = RecordingEmbedder::default();

        let result = sync
            .sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("sync succeeds");

        let per_file = result.chunks_added / 2;
        assert!(
            per_file > 1,
            "fixture must chunk each file into several pieces"
        );
        assert_eq!(result.embedded_texts, per_file);
        assert_eq!(result.embeddings_reused, per_file);
        assert_eq!(
            result.embedded_texts + result.embeddings_reused,
            result.chunks_added
        );
        assert_eq!(embedder.calls().len(), 1, "one window, one embed call");
        assert_eq!(store.doc_count(), result.chunks_added);
        assert!(store
            .all_embeddings()
            .iter()
            .all(|embedding| embedding.iter().all(|value| value.is_finite())));
    }

    #[tokio::test]
    async fn test_sync_reuses_stored_embeddings_for_later_copies() {
        let (dir, sync, chunker, _embedder, mut store) = fixture();
        let body = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu";
        std::fs::write(dir.path().join("a.txt"), body).expect("write a");
        init_openai(&sync);
        let embedder = RecordingEmbedder::default();
        let first = sync
            .sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("first sync succeeds");

        std::fs::write(dir.path().join("later-copy.txt"), body).expect("write copy");
        let second = sync
            .sync(&chunker, &embedder, &mut store, false, false, false)
            .await
            .expect("second sync succeeds");

        assert_eq!(second.files_added, 1);
        assert_eq!(second.chunks_added, first.chunks_added);
        assert_eq!(second.embedded_texts, 0);
        assert_eq!(second.embeddings_reused, first.chunks_added);
        assert_eq!(second.embedding_api_calls, 0);
        assert_eq!(
            embedder.calls().len(),
            1,
            "the copy must not call the embedder"
        );
        assert_eq!(store.doc_count(), first.chunks_added * 2);
    }

    #[tokio::test]
    async fn test_sync_batches_across_files_longest_first() {
        let (dir, sync, chunker, _embedder, mut store) = fixture();
        std::fs::write(dir.path().join("a.txt"), "short one").expect("write a");
        std::fs::write(
            dir.path().join("b.txt"),
            "a considerably longer paragraph of words that spans far more characters",
        )
        .expect("write b");
        std::fs::write(dir.path().join("c.txt"), "medium sized text here").expect("write c");
        init_openai(&sync);
        let embedder = RecordingEmbedder::default();

        sync.sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("sync succeeds");

        let calls = embedder.calls();
        assert_eq!(calls.len(), 1, "small files share one embed call");
        let lengths: Vec<usize> = calls[0].iter().map(String::len).collect();
        let mut sorted = lengths.clone();
        sorted.sort_by(|a, b| b.cmp(a));
        assert_eq!(
            lengths, sorted,
            "texts must reach the embedder longest first"
        );
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

    // ---- issue #71: batched writes, checkpoints, crash resume -------------

    use crate::vectorstore::{Document, DocumentUpdate, QueryHit};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// Wraps [`InMemoryStore`] to count write calls and inject a crash.
    #[derive(Default)]
    struct SpyStore {
        inner: InMemoryStore,
        update_batch_sizes: Vec<usize>,
        upsert_calls: usize,
        insert_calls: usize,
        delete_calls: usize,
        existing_ids_calls: AtomicUsize,
        checkpoints: usize,
        /// Fail every write of new rows once this many have succeeded.
        fail_upserts_after: Option<usize>,
        /// Fail any insert that carries a row for this path.
        poison_path: Option<String>,
    }

    impl SpyStore {
        fn checks(&self) -> usize {
            self.existing_ids_calls.load(Ordering::SeqCst)
        }
    }

    impl VectorStore for SpyStore {
        fn upsert(&mut self, docs: &[Document]) -> Result<()> {
            if self
                .fail_upserts_after
                .is_some_and(|n| self.upsert_calls >= n)
            {
                return Err(MinSyncError::VectorStore("injected crash".into()));
            }
            self.upsert_calls += 1;
            self.inner.upsert(docs)
        }
        fn insert_new(&mut self, docs: &[Document]) -> Result<()> {
            if self
                .fail_upserts_after
                .is_some_and(|n| self.upsert_calls >= n)
            {
                return Err(MinSyncError::VectorStore("injected crash".into()));
            }
            if let Some(poison) = &self.poison_path {
                if docs.iter().any(|doc| &doc.path == poison) {
                    return Err(MinSyncError::VectorStore("disk full".into()));
                }
            }
            self.upsert_calls += 1;
            self.insert_calls += 1;
            self.inner.upsert(docs)
        }
        fn existing_ids(&self, ids: &[String]) -> Result<std::collections::HashSet<String>> {
            self.existing_ids_calls.fetch_add(1, Ordering::SeqCst);
            Ok(self.inner.fetch(ids)?.into_iter().map(|d| d.id).collect())
        }
        fn update(&mut self, updates: &[DocumentUpdate]) -> Result<()> {
            self.update_batch_sizes.push(updates.len());
            self.inner.update(updates)
        }
        fn fetch(&self, ids: &[String]) -> Result<Vec<Document>> {
            self.inner.fetch(ids)
        }
        fn delete_by_filter(&mut self, filter: &Filter) -> Result<usize> {
            self.delete_calls += 1;
            self.inner.delete_by_filter(filter)
        }
        fn query(
            &self,
            vector: &[f32],
            filter: Option<&Filter>,
            topk: usize,
        ) -> Result<Vec<QueryHit>> {
            self.inner.query(vector, filter, topk)
        }
        fn query_text(
            &self,
            text: &str,
            filter: Option<&Filter>,
            topk: usize,
        ) -> Result<Vec<QueryHit>> {
            self.inner.query_text(text, filter, topk)
        }
        fn flush(&mut self) -> Result<()> {
            self.inner.flush()
        }
        fn checkpoint(&mut self) -> Result<()> {
            self.checkpoints += 1;
            Ok(())
        }
        fn doc_count(&self) -> usize {
            self.inner.doc_count()
        }
        fn all_paths(&self) -> Vec<String> {
            self.inner.all_paths()
        }
    }

    /// [`MockEmbedder`] that also counts how many texts it was asked to embed.
    struct CountingEmbedder {
        texts: Arc<AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl Embedder for CountingEmbedder {
        fn id(&self) -> &str {
            "mock"
        }
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            self.texts.fetch_add(texts.len(), Ordering::SeqCst);
            MockEmbedder.embed(texts).await
        }
    }

    fn counting_embedder() -> (CountingEmbedder, Arc<AtomicUsize>) {
        let texts = Arc::new(AtomicUsize::new(0));
        (
            CountingEmbedder {
                texts: texts.clone(),
            },
            texts,
        )
    }

    const MULTI_CHUNK: &str = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda";

    fn write_numbered_files(dir: &TempDir, count: usize) {
        for i in 0..count {
            std::fs::write(
                dir.path().join(format!("f{i}.txt")),
                format!("{MULTI_CHUNK} file{i} {MULTI_CHUNK}"),
            )
            .expect("write file");
        }
    }

    fn init_sync(sync: &MinSync) {
        sync.init(false, "openai:text-embedding-3-small", "recursive")
            .expect("init succeeds");
    }

    #[tokio::test]
    async fn test_index_file_writes_all_reused_chunks_in_one_update() {
        let (dir, _sync, chunker, embedder, _store) = fixture();
        let config = Config::default_for("source-1");
        std::fs::write(dir.path().join("a.txt"), MULTI_CHUNK).expect("write file");
        let mut store = SpyStore::default();

        for token in ["t1", "t2"] {
            let mut result = empty_sync_result(false, false);
            let context = SyncFileContext {
                config: &config,
                chunker: &chunker,
                embedder: &embedder,
                store: &mut store,
                sync_token: token,
            };
            index_file(dir.path(), "a.txt", context, &mut result)
                .await
                .expect("index file");
        }

        let chunks = store.doc_count();
        assert!(chunks > 1, "fixture must be multi-chunk");
        assert_eq!(
            store.update_batch_sizes,
            vec![chunks],
            "second pass reuses every chunk: ONE update call carrying all of them"
        );
    }

    #[tokio::test]
    async fn test_sync_checkpoints_every_n_files() {
        let (dir, sync, chunker, embedder, _store) = fixture();
        let sync = sync.with_checkpoint_every(2);
        write_numbered_files(&dir, 5);
        init_sync(&sync);
        let mut store = SpyStore::default();

        sync.sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("sync");

        assert_eq!(store.checkpoints, 2, "after file 2 and file 4 of 5");
    }

    fn set_batch_files(sync: &MinSync, files: usize) {
        let path = sync.minsync_dir.join("config.toml");
        let mut config = Config::load(&path).expect("load config");
        config.sync.batch_files = files;
        config.save(&path).expect("save config");
    }

    /// Store write calls of an initial sync over `files` files (all new).
    async fn initial_sync_calls(files: usize) -> (SpyStore, SyncResult) {
        let (dir, sync, chunker, embedder, _store) = fixture();
        write_numbered_files(&dir, files);
        init_sync(&sync);
        let mut store = SpyStore::default();
        let result = sync
            .sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("sync");
        (store, result)
    }

    #[tokio::test]
    async fn test_manifest_cursor_mismatch_forces_rebuild_after_interrupted_state_commit() {
        let (dir, sync, chunker, _embedder, _store) = fixture();
        write_numbered_files(&dir, 6);
        init_sync(&sync);

        // Model a crash after the old code saved a partial cursor but before
        // it replaced init's full baseline manifest.
        let manifest_path = sync.minsync_dir.join("manifest.json");
        let manifest = Manifest::load(&manifest_path).expect("load init manifest");
        let mut partial_manifest = manifest.clone();
        assert!(partial_manifest.files.remove("f0.txt").is_some());
        let config = Config::load(&sync.minsync_dir.join("config.toml")).expect("load config");
        Cursor {
            source_id: config.source_id.clone(),
            last_synced_at: chrono::Utc::now().to_rfc3339(),
            manifest_hash: partial_manifest.manifest_hash(),
            chunk_schema_id: chunker.schema_id().to_string(),
            embedder_id: "mock".to_string(),
            collection_path: config.collection.path.clone(),
            lexical_language: config.lexical.language.clone(),
        }
        .save(&sync.minsync_dir.join("cursor.json"))
        .expect("write cursor from interrupted state commit");

        let mut store = SpyStore::default();
        let result = sync
            .sync(&chunker, &MockEmbedder, &mut store, false, false, false)
            .await
            .expect("mismatched state triggers a complete rebuild");

        assert!(result.initial_sync);
        assert_eq!(result.files_processed, 6);
        assert_eq!(store.all_paths().len(), 6);
        assert_eq!(
            Cursor::load(&sync.minsync_dir.join("cursor.json"))
                .expect("load committed cursor")
                .manifest_hash,
            Manifest::load(&manifest_path)
                .expect("load committed manifest")
                .manifest_hash(),
            "cursor and manifest must describe the same completed sync"
        );
    }

    #[tokio::test]
    async fn test_store_write_calls_do_not_grow_with_files_in_a_batch() {
        let (small, small_result) = initial_sync_calls(4).await;
        let (large, large_result) = initial_sync_calls(20).await;

        assert_eq!(small_result.files_processed, 4);
        assert_eq!(large_result.files_processed, 20);
        assert_eq!(
            (small.checks(), small.insert_calls, small.delete_calls),
            (large.checks(), large.insert_calls, large.delete_calls),
            "one existence check, one insert and the same sweeps for 4 or 20 files"
        );
        assert_eq!(large.checks(), 1);
        assert_eq!(large.insert_calls, 1);
        assert!(large.update_batch_sizes.is_empty());
        assert_eq!(
            large_result.embedding_api_calls, 1,
            "one embed call per batch"
        );
        assert_eq!(large.inner.all_paths().len(), 20);
    }

    #[tokio::test]
    async fn test_modified_files_reuse_chunks_in_one_update_per_batch() {
        let (dir, sync, chunker, embedder, _store) = fixture();
        write_numbered_files(&dir, 10);
        init_sync(&sync);
        let mut store = SpyStore::default();
        sync.sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("initial sync");
        let (updates_before, inserts_before, checks_before) = (
            store.update_batch_sizes.len(),
            store.insert_calls,
            store.checks(),
        );
        for i in 0..10 {
            let path = dir.path().join(format!("f{i}.txt"));
            let mut text = std::fs::read_to_string(&path).expect("read");
            text.push_str(" extra tail words appended after the crash point");
            std::fs::write(path, text).expect("modify");
        }

        let result = sync
            .sync(&chunker, &embedder, &mut store, false, false, false)
            .await
            .expect("incremental sync");

        assert_eq!(result.files_modified, 10);
        assert!(result.chunks_updated > 0, "unchanged chunks are reused");
        assert_eq!(store.checks() - checks_before, 1);
        assert_eq!(store.update_batch_sizes.len() - updates_before, 1);
        assert_eq!(store.insert_calls - inserts_before, 1);
        assert_eq!(
            *store.update_batch_sizes.last().expect("update batch"),
            result.chunks_updated,
            "all reused chunks of all 10 files in one update"
        );
    }

    #[tokio::test]
    async fn test_batch_size_one_writes_each_file_separately() {
        let (dir, sync, chunker, embedder, _store) = fixture();
        write_numbered_files(&dir, 5);
        init_sync(&sync);
        set_batch_files(&sync, 1);
        let mut store = SpyStore::default();

        sync.sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("sync");

        assert_eq!(store.insert_calls, 5);
        assert_eq!(store.checks(), 5);
    }

    #[tokio::test]
    async fn test_batches_never_straddle_a_checkpoint() {
        let (dir, sync, chunker, embedder, _store) = fixture();
        let sync = sync.with_checkpoint_every(3);
        write_numbered_files(&dir, 7);
        init_sync(&sync);
        set_batch_files(&sync, 4);
        let mut store = SpyStore::default();

        sync.sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("sync");

        // groups of 3, 3, 1: a batch of 4 is cut at each checkpoint boundary
        assert_eq!(store.insert_calls, 3);
        assert_eq!(store.checkpoints, 2);
    }

    #[tokio::test]
    async fn test_crash_mid_batch_redoes_only_the_uncommitted_batch() {
        let (dir, sync, chunker, _embedder, _store) = fixture();
        write_numbered_files(&dir, 6);
        init_sync(&sync);
        set_batch_files(&sync, 2);
        set_on_file_error(&sync, crate::config::OnFileError::Fail);
        let mut store = SpyStore {
            fail_upserts_after: Some(2),
            ..SpyStore::default()
        };
        sync.sync(&chunker, &MockEmbedder, &mut store, true, false, false)
            .await
            .expect_err("third batch crashes");
        store.fail_upserts_after = None;
        assert_eq!(store.inner.all_paths().len(), 4, "two batches committed");

        let result = sync
            .sync(&chunker, &MockEmbedder, &mut store, false, false, false)
            .await
            .expect("resume");

        assert_eq!(result.files_resumed, 4);
        assert_eq!(result.files_processed, 2, "only the failed batch is redone");
        assert_eq!(store.inner.all_paths().len(), 6);
        assert_eq!(
            store.doc_count(),
            uninterrupted_doc_count(&chunker, 6).await
        );
    }

    /// Run an initial sync that crashes on the 4th file's upsert.
    async fn crash_initial_sync(
        dir: &TempDir,
        sync: &MinSync,
        chunker: &ChonkieChunker,
    ) -> SpyStore {
        write_numbered_files(dir, 6);
        init_sync(sync);
        set_batch_files(sync, 1);
        set_on_file_error(sync, crate::config::OnFileError::Fail);
        let mut store = SpyStore {
            fail_upserts_after: Some(3),
            ..SpyStore::default()
        };
        let error = sync
            .sync(chunker, &MockEmbedder, &mut store, true, false, false)
            .await
            .expect_err("injected crash fails the sync");
        assert!(error.to_string().contains("injected crash"), "{error}");
        assert!(sync.minsync_dir.join("txn.json").exists());
        assert!(sync.minsync_dir.join("progress.jsonl").exists());
        assert!(!sync.minsync_dir.join("cursor.json").exists());
        store
    }

    async fn uninterrupted_doc_count(chunker: &ChonkieChunker, files: usize) -> usize {
        let (dir, sync, _chunker, embedder, mut store) = fixture();
        write_numbered_files(&dir, files);
        init_sync(&sync);
        sync.sync(chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("reference sync");
        store.doc_count()
    }

    #[tokio::test]
    async fn test_resume_after_crash_skips_committed_files() {
        let (dir, sync, chunker, _embedder, _store) = fixture();
        let mut store = crash_initial_sync(&dir, &sync, &chunker).await;
        store.fail_upserts_after = None;
        let (embedder, embedded) = counting_embedder();

        let result = sync
            .sync(&chunker, &embedder, &mut store, false, false, false)
            .await
            .expect("resumed sync");

        assert_eq!(result.files_resumed, 3);
        assert_eq!(result.files_processed, 3);
        assert_eq!(
            embedded.load(Ordering::SeqCst),
            result.embedded_texts,
            "only unfinished files are embedded"
        );
        assert_eq!(
            store.doc_count(),
            uninterrupted_doc_count(&chunker, 6).await,
            "full-sync sweep must not delete the already-committed files' chunks"
        );
        assert_eq!(store.all_paths().len(), 6);
        assert!(sync.minsync_dir.join("cursor.json").exists());
        assert!(!sync.minsync_dir.join("txn.json").exists());
        assert!(!sync.minsync_dir.join("progress.jsonl").exists());
    }

    #[tokio::test]
    async fn test_resume_reprocesses_files_changed_since_interruption() {
        let (dir, sync, chunker, _embedder, _store) = fixture();
        let mut store = crash_initial_sync(&dir, &sync, &chunker).await;
        store.fail_upserts_after = None;
        std::fs::write(
            dir.path().join("f0.txt"),
            "CHANGED after the crash CHANGED after the crash",
        )
        .expect("rewrite f0");

        let result = sync
            .sync(&chunker, &MockEmbedder, &mut store, false, false, false)
            .await
            .expect("resumed sync");

        assert_eq!(result.files_resumed, 2, "f1, f2 reused; f0 changed");
        let hits = store
            .query(
                &[0.5; 8],
                Some(&Filter::Eq("path".to_string(), "f0.txt".to_string())),
                50,
            )
            .expect("query f0");
        assert!(!hits.is_empty());
        assert!(
            hits.iter()
                .all(|hit| !hit.text.contains("alpha") && !hit.text.contains("file0")),
            "no pre-crash chunk of f0 may survive: {hits:?}"
        );
        let joined: String = hits.iter().map(|hit| hit.text.as_str()).collect();
        assert!(joined.contains("CHANGED"), "{hits:?}");
    }

    /// The `fail_on_call`-th `embed` call (1-based) reports a native deadline
    /// expiry instead of embedding.
    struct TimeoutOnCall {
        calls: AtomicUsize,
        fail_on_call: usize,
    }

    #[async_trait::async_trait]
    impl Embedder for TimeoutOnCall {
        fn id(&self) -> &str {
            "mock"
        }
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
            if call == self.fail_on_call {
                return Err(MinSyncError::Timeout("stuck native batch".to_string()));
            }
            MockEmbedder.embed(texts).await
        }
    }

    #[tokio::test]
    async fn test_embedder_timeout_aborts_with_code_7_and_rerun_resumes_committed_files() {
        let (dir, sync, chunker, _embedder, mut store) = fixture();
        write_numbered_files(&dir, 6);
        init_sync(&sync);
        set_batch_files(&sync, 1);
        let stuck = TimeoutOnCall {
            calls: AtomicUsize::new(0),
            fail_on_call: 4,
        };

        let error = sync
            .sync(&chunker, &stuck, &mut store, true, false, false)
            .await
            .expect_err("timeout aborts the run");

        assert!(matches!(error, MinSyncError::Timeout(_)), "{error}");
        assert_eq!(error.exit_code(), 7);
        assert!(sync.minsync_dir.join("progress.jsonl").exists());
        assert!(!sync.minsync_dir.join("cursor.json").exists());
        assert_eq!(store.all_paths().len(), 3, "three files committed first");

        let (healthy, healthy_count) = counting_embedder();
        let result = sync
            .sync(&chunker, &healthy, &mut store, false, false, false)
            .await
            .expect("rerun resumes");

        assert_eq!(result.files_resumed, 3);
        assert_eq!(result.files_processed, 3);
        assert_eq!(
            healthy_count.load(Ordering::SeqCst),
            result.embedded_texts,
            "committed files are not re-embedded"
        );
        assert_eq!(store.all_paths().len(), 6);
    }

    #[tokio::test]
    async fn test_resume_discarded_when_chunking_config_changes() {
        let (dir, sync, chunker, _embedder, _store) = fixture();
        let mut store = crash_initial_sync(&dir, &sync, &chunker).await;
        store.fail_upserts_after = None;
        let config_path = sync.minsync_dir.join("config.toml");
        let mut config = Config::load(&config_path).expect("load config");
        config.normalize.collapse_whitespace = true;
        config.save(&config_path).expect("save config");

        let result = sync
            .sync(&chunker, &MockEmbedder, &mut store, false, false, false)
            .await
            .expect("sync after config change");

        assert_eq!(result.files_resumed, 0);
        assert_eq!(result.files_processed, 6);
    }

    #[tokio::test]
    async fn test_resume_incremental_sync_honors_recorded_deletions() {
        let (dir, sync, chunker, embedder, _store) = fixture();
        let mut store = SpyStore::default();
        for name in ["a-del", "b-del", "c-mod", "d-mod", "e-keep"] {
            std::fs::write(dir.path().join(format!("{name}.txt")), MULTI_CHUNK)
                .expect("write file");
        }
        init_sync(&sync);
        set_batch_files(&sync, 1);
        sync.sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("initial sync");

        std::fs::remove_file(dir.path().join("a-del.txt")).expect("remove a");
        std::fs::remove_file(dir.path().join("b-del.txt")).expect("remove b");
        for name in ["c-mod", "d-mod"] {
            std::fs::write(
                dir.path().join(format!("{name}.txt")),
                format!("{MULTI_CHUNK} brand new tail for {name}"),
            )
            .expect("modify file");
        }
        set_on_file_error(&sync, crate::config::OnFileError::Fail);
        store.upsert_calls = 0;
        store.fail_upserts_after = Some(1);
        sync.sync(&chunker, &embedder, &mut store, false, false, false)
            .await
            .expect_err("crash while indexing d-mod");
        store.fail_upserts_after = None;

        let result = sync
            .sync(&chunker, &embedder, &mut store, false, false, false)
            .await
            .expect("resumed sync");

        assert_eq!(result.files_resumed, 3, "two deletions and c-mod");
        assert_eq!(result.files_processed, 1, "only d-mod remains");
        assert_eq!(
            store.all_paths(),
            vec!["c-mod.txt", "d-mod.txt", "e-keep.txt"]
        );
    }

    // ---- issue #78: skip-and-record failing files -------------------------

    use crate::chunker::Chunker;
    use crate::types::{Chunk, FileStage};

    /// Fails on any text containing `POISON`, like a parser rejecting a file.
    struct PoisonChunker(ChonkieChunker);

    impl Chunker for PoisonChunker {
        fn schema_id(&self) -> &str {
            self.0.schema_id()
        }
        fn chunk(&self, text: &str, path: &str) -> Result<Vec<Chunk>> {
            if text.contains("POISON") {
                return Err(MinSyncError::Other("parser rejected file".into()));
            }
            self.0.chunk(text, path)
        }
    }

    /// Fails the whole `embed` call when any text contains `POISON`.
    struct RejectingEmbedder;

    #[async_trait::async_trait]
    impl Embedder for RejectingEmbedder {
        fn id(&self) -> &str {
            "mock"
        }
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            if texts.iter().any(|text| text.contains("POISON")) {
                return Err(MinSyncError::Embedding("backend rejected input".into()));
            }
            MockEmbedder.embed(texts).await
        }
    }

    fn set_on_file_error(sync: &MinSync, mode: crate::config::OnFileError) {
        let path = sync.minsync_dir.join("config.toml");
        let mut config = Config::load(&path).expect("load config");
        config.sync.on_file_error = mode;
        config.save(&path).expect("save config");
    }

    fn write_good_and_poison(dir: &TempDir) {
        for name in ["a", "b", "d", "e"] {
            std::fs::write(
                dir.path().join(format!("{name}.txt")),
                format!("{MULTI_CHUNK} {name}"),
            )
            .expect("write good file");
        }
        std::fs::write(
            dir.path().join("c-bad.txt"),
            format!("{MULTI_CHUNK} POISON"),
        )
        .expect("write poison file");
    }

    const GOOD: [&str; 4] = ["a.txt", "b.txt", "d.txt", "e.txt"];

    #[tokio::test]
    async fn test_chunk_failure_skips_file_and_indexes_the_rest() {
        let (dir, sync, chunker, embedder, mut store) = fixture();
        write_good_and_poison(&dir);
        init_sync(&sync);
        let chunker = PoisonChunker(chunker);

        let result = sync
            .sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("sync completes despite the bad file");

        assert_eq!(store.all_paths(), GOOD);
        assert!(result.partial);
        assert_eq!(result.files_failed.len(), 1);
        let failure = &result.files_failed[0];
        assert_eq!(failure.path, "c-bad.txt");
        assert_eq!(failure.stage, FileStage::Chunk);
        assert!(failure.error.contains("parser rejected file"));
        assert_eq!(result.files_processed, 4);
        assert!(!result
            .files_processed_paths
            .contains(&"c-bad.txt".to_string()));
    }

    #[tokio::test]
    async fn test_embed_failure_poison_file_does_not_take_batch_mates_down() {
        let (dir, sync, chunker, _embedder, mut store) = fixture();
        write_good_and_poison(&dir);
        init_sync(&sync);

        let result = sync
            .sync(&chunker, &RejectingEmbedder, &mut store, true, false, false)
            .await
            .expect("sync completes despite the bad file");

        assert_eq!(store.all_paths(), GOOD, "all 4 batch-mates committed");
        assert!(result.partial);
        assert_eq!(result.files_failed.len(), 1);
        assert_eq!(result.files_failed[0].path, "c-bad.txt");
        assert_eq!(result.files_failed[0].stage, FileStage::Embed);
        assert!(result.files_failed[0]
            .error
            .contains("backend rejected input"));
        assert_eq!(result.files_processed, 4);
    }

    #[tokio::test]
    async fn test_store_failure_poison_file_is_isolated_by_per_file_retry() {
        let (dir, sync, chunker, embedder, _store) = fixture();
        write_good_and_poison(&dir);
        init_sync(&sync);
        let mut store = SpyStore {
            poison_path: Some("c-bad.txt".into()),
            ..SpyStore::default()
        };

        let result = sync
            .sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("sync completes despite the bad file");

        assert_eq!(store.inner.all_paths(), GOOD);
        assert_eq!(result.files_failed.len(), 1);
        assert_eq!(result.files_failed[0].path, "c-bad.txt");
        assert_eq!(result.files_failed[0].stage, FileStage::Store);
        assert!(result.partial);
    }

    #[tokio::test]
    async fn test_read_failure_in_index_batch_names_path_and_stage() {
        let (dir, _sync, chunker, embedder, _store) = fixture();
        // A directory where a file is expected: read_to_string fails with
        // EISDIR, which is neither InvalidData nor NotFound.
        std::fs::create_dir(dir.path().join("dir.txt")).expect("mkdir");
        std::fs::write(dir.path().join("ok.txt"), MULTI_CHUNK).expect("write");
        let config = Config::default_for("source-1");
        let mut store = InMemoryStore::new();
        let mut result = empty_sync_result(false, false);
        let items = ["dir.txt", "ok.txt"].map(|path| BatchItem {
            path: path.to_string(),
            kind: BatchKind::Index,
            purge_first: false,
        });
        let context = SyncFileContext {
            config: &config,
            chunker: &chunker,
            embedder: &embedder,
            store: &mut store,
            sync_token: "t1",
        };

        index_batch(dir.path(), &items, context, &mut result)
            .await
            .expect("skip mode swallows the read error");

        assert_eq!(store.all_paths(), vec!["ok.txt"]);
        assert_eq!(result.files_failed.len(), 1);
        assert_eq!(result.files_failed[0].path, "dir.txt");
        assert_eq!(result.files_failed[0].stage, FileStage::Read);
        assert_eq!(result.files_processed_paths, vec!["ok.txt"]);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_unreadable_file_is_skipped_and_retried_once_fixed() {
        use std::os::unix::fs::PermissionsExt;

        let (dir, sync, chunker, embedder, mut store) = fixture();
        write_numbered_files(&dir, 3);
        let locked = dir.path().join("locked.txt");
        std::fs::write(&locked, MULTI_CHUNK).expect("write locked");
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000))
            .expect("chmod 000");
        if std::fs::read(&locked).is_ok() {
            return; // running as root: permissions do not apply
        }
        init_sync(&sync);

        let first = sync
            .sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("sync completes despite the unreadable file");

        assert_eq!(store.all_paths(), vec!["f0.txt", "f1.txt", "f2.txt"]);
        assert!(first.partial);
        assert_eq!(first.files_failed.len(), 1);
        assert_eq!(first.files_failed[0].path, "locked.txt");
        assert_eq!(first.files_failed[0].stage, FileStage::Read);
        assert!(!first.already_up_to_date);

        // Still unreadable: nothing changed for the indexed files, yet the
        // failure is reported again rather than silently forgotten.
        let again = sync
            .sync(&chunker, &embedder, &mut store, false, false, false)
            .await
            .expect("second sync");
        assert!(again.partial);
        assert_eq!(again.files_failed[0].path, "locked.txt");
        assert_eq!(again.files_processed, 0);

        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o644))
            .expect("chmod 644");
        let fixed = sync
            .sync(&chunker, &embedder, &mut store, false, false, false)
            .await
            .expect("sync after fix");
        assert!(!fixed.partial);
        assert!(fixed.files_failed.is_empty());
        assert_eq!(fixed.files_processed_paths, vec!["locked.txt"]);
        assert!(store.all_paths().contains(&"locked.txt".to_string()));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_full_sync_keeps_old_chunks_of_unreadable_indexed_file() {
        use std::os::unix::fs::PermissionsExt;

        let (dir, sync, chunker, embedder, mut store) = fixture();
        write_numbered_files(&dir, 2);
        let locked = dir.path().join("locked.txt");
        std::fs::write(&locked, MULTI_CHUNK).expect("write locked");
        init_sync(&sync);
        sync.sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("initial sync");
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000))
            .expect("chmod 000");
        if std::fs::read(&locked).is_ok() {
            return; // running as root: permissions do not apply
        }

        let full = sync
            .sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("full sync completes");

        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o644))
            .expect("chmod 644");
        assert!(full.partial);
        assert!(
            store.all_paths().contains(&"locked.txt".to_string()),
            "full sweep must not delete the skipped file's old chunks"
        );
    }

    #[tokio::test]
    async fn test_skipped_file_is_retried_by_next_sync() {
        let (dir, sync, chunker, embedder, mut store) = fixture();
        write_good_and_poison(&dir);
        init_sync(&sync);
        let poison_chunker = PoisonChunker(ChonkieChunker::new(32, "\n "));
        sync.sync(&poison_chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("first sync");

        std::fs::write(dir.path().join("c-bad.txt"), MULTI_CHUNK).expect("fix file");
        let result = sync
            .sync(&chunker, &embedder, &mut store, false, false, false)
            .await
            .expect("retry sync");

        assert!(!result.partial);
        assert!(result.files_failed.is_empty());
        assert_eq!(result.files_processed_paths, vec!["c-bad.txt"]);
        assert!(store.all_paths().contains(&"c-bad.txt".to_string()));
    }

    #[tokio::test]
    async fn test_skipped_modified_file_keeps_old_chunks_until_it_succeeds() {
        let (dir, sync, chunker, embedder, mut store) = fixture();
        std::fs::write(dir.path().join("a.txt"), MULTI_CHUNK).expect("write");
        init_sync(&sync);
        sync.sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("initial sync");
        std::fs::write(dir.path().join("a.txt"), format!("{MULTI_CHUNK} POISON")).expect("modify");

        let result = sync
            .sync(
                &PoisonChunker(chunker),
                &embedder,
                &mut store,
                false,
                false,
                false,
            )
            .await
            .expect("sync completes");

        assert!(result.partial);
        assert_eq!(
            store.all_paths(),
            vec!["a.txt"],
            "old chunks stay searchable"
        );
        let status = sync
            .sync(
                &PoisonChunker(ChonkieChunker::new(32, "\n ")),
                &embedder,
                &mut store,
                false,
                false,
                false,
            )
            .await
            .expect("retry still fails the same way");
        assert!(status.partial, "modified file is retried, not marked done");
    }

    #[tokio::test]
    async fn test_on_file_error_fail_aborts_with_path_in_error() {
        let (dir, sync, chunker, embedder, mut store) = fixture();
        write_good_and_poison(&dir);
        init_sync(&sync);
        set_on_file_error(&sync, crate::config::OnFileError::Fail);

        let error = sync
            .sync(
                &PoisonChunker(chunker),
                &embedder,
                &mut store,
                true,
                false,
                false,
            )
            .await
            .expect_err("fail mode aborts on the first bad file");

        assert!(
            error.to_string().contains("c-bad.txt"),
            "error must name the file: {error}"
        );
        assert!(!sync.minsync_dir.join("cursor.json").exists());
    }

    #[tokio::test]
    async fn test_on_file_error_fail_embed_error_keeps_exit_code_and_names_path() {
        let (dir, sync, chunker, _embedder, mut store) = fixture();
        write_good_and_poison(&dir);
        init_sync(&sync);
        set_on_file_error(&sync, crate::config::OnFileError::Fail);

        let error = sync
            .sync(&chunker, &RejectingEmbedder, &mut store, true, false, false)
            .await
            .expect_err("fail mode aborts");

        assert!(matches!(error, MinSyncError::Embedding(_)));
        assert_eq!(error.exit_code(), 5);
    }

    #[test]
    fn test_on_file_error_defaults_to_skip_and_parses_from_toml() {
        let config: crate::config::SyncConfig = toml::from_str("").expect("empty section");
        assert_eq!(config.on_file_error, crate::config::OnFileError::Skip);
        let config: crate::config::SyncConfig =
            toml::from_str("on_file_error = \"fail\"").expect("fail section");
        assert_eq!(config.on_file_error, crate::config::OnFileError::Fail);
        assert!(toml::from_str::<crate::config::SyncConfig>("on_file_error = \"x\"").is_err());
    }

    #[tokio::test]
    async fn test_embedder_outage_is_a_hard_failure_not_a_pile_of_skipped_files() {
        struct DownEmbedder;
        #[async_trait::async_trait]
        impl Embedder for DownEmbedder {
            fn id(&self) -> &str {
                "mock"
            }
            async fn embed(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>> {
                Err(MinSyncError::Embedding("service unavailable".into()))
            }
        }
        let (dir, sync, chunker, _embedder, mut store) = fixture();
        write_numbered_files(&dir, 6);
        init_sync(&sync);

        let error = sync
            .sync(&chunker, &DownEmbedder, &mut store, true, false, false)
            .await
            .expect_err("outage must abort, not skip every file");

        assert_eq!(error.exit_code(), 5);
        assert!(error.to_string().contains("service unavailable"), "{error}");
        assert!(!sync.minsync_dir.join("cursor.json").exists());
    }
}
