//! Incremental sync orchestration: locking, transaction lifecycle, manifest
//! diffing, and cursor advancement. Per-file work lives in [`indexer`];
//! result accounting helpers live in [`result`].

mod indexer;
mod reporter;
mod result;
mod sparse_phase;

use crate::chunker::Chunker;
use crate::config::{Config, OnFileError};
use crate::embedder::Embedder;
use crate::error::{MinSyncError, Result};
use crate::incremental::journal::{Journal, Record, RunHeader, Status, JOURNAL_FILE};
use crate::manifest::{FileChange, Manifest, ScanErrorKind};
use crate::sparse::SparseEncoder;
use crate::state::{Cursor, FileLock, Transaction};
use crate::types::{FileFailure, FileStage, SyncResult};
use crate::vectorstore::{Filter, VectorStore};
use indexer::{index_batch, BatchItem, BatchKind, SyncFileContext};
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
    /// Encodes the sparse (SPLADE) vector of every stored row. Without one a
    /// sync is dense-only and leaves the sparse columns alone.
    sparse: Option<Arc<dyn SparseEncoder>>,
}

impl MinSync {
    pub fn new(root: PathBuf) -> Self {
        let minsync_dir = root.join(".minsync");
        Self {
            root,
            minsync_dir,
            checkpoint_every: None,
            progress: None,
            sparse: None,
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

    /// Encode a sparse vector for every stored row at the end of each sync
    /// (the rows this run wrote, and any older row that has none from this
    /// model), and record the model in the cursor.
    pub fn with_sparse_encoder(mut self, encoder: Arc<dyn SparseEncoder>) -> Self {
        self.sparse = Some(encoder);
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

        for state_file in ["cursor.json", "txn.json", JOURNAL_FILE] {
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
/// chunker/normalizer/embedder between a crash and the retry
/// re-indexes instead of silently keeping rows built the old way. The sparse
/// encoder is not part of it: the sparse phase is driven by the store, so a
/// changed model is found by what the rows hold, not by the resume log.
fn config_fingerprint(config: &Config, chunker: &dyn Chunker, embedder: &dyn Embedder) -> String {
    let material = serde_json::json!({
        "chunker": &config.chunker,
        "chunker_schema": chunker.schema_id(),
        "normalize": &config.normalize,
        "embedder": &config.embedder,
        "embedder_id": embedder.id(),
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
        let progress_path = self.minsync_dir.join(JOURNAL_FILE);

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
        let full = full || initial_sync;

        // Read the interrupted run's log BEFORE clearing its txn marker.
        let resume_state = Journal::load(&progress_path).into_ledger(&progress_path);
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

        let sparse_pending = match &self.sparse {
            Some(encoder) if !dry_run => store.count_missing_sparse(encoder.id())? > 0,
            _ => false,
        };
        if changes.is_empty() && !full && !sparse_pending {
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
        let expected = RunHeader::new(
            VECTOR_STAGE,
            &config.source_id,
            full,
            &old_manifest.manifest_hash(),
            &fingerprint,
        );
        let resume = resume_state.filter(|ledger| ledger.header.resumes(&expected));
        let header = match &resume {
            Some(ledger) => RunHeader {
                run_token: ledger.header.run_token.clone(),
                created_at: ledger.header.created_at.clone(),
                ..expected
            },
            None => expected,
        };
        let sync_token = header.run_token.clone();
        Transaction::new(
            &config.source_id,
            &sync_token,
            Some(old_manifest.manifest_hash()),
            &new_manifest.manifest_hash(),
        )
        .save(&txn_path)?;
        let mut journal = Journal::create(&progress_path, &header, resume.as_ref())?;

        let mut result = empty_sync_result(false, false);
        result.files_failed = scan_failures;
        result.initial_sync = initial_sync;
        result.files_checked = new_manifest.files.len();
        result.scan_errors = scan_errors;
        result.query_ready = true;

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
            let status = resume.as_ref().map_or(Status::Unseen, |ledger| {
                ledger.status(path, content_hash.as_deref())
            });
            if status == Status::Committed {
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
                    purge_first: status == Status::Superseded,
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
        let mut systemic_streak = 0usize;
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
            index_batch(
                &self.root,
                &items,
                context,
                &mut result,
                &mut systemic_streak,
            )
            .await?;
            if let Some(progress) = &self.progress {
                for failure in &result.files_failed[failed_before..] {
                    progress.file_failed(failure);
                }
            }

            // Phase two of the stage commit contract. The store commits of
            // this batch have returned (`index_batch` above), and only now
            // are completions made durable, so a recorded path is always safe
            // to skip on resume. A crash right here repeats the batch, which
            // is harmless because the commits are idempotent. One fsync for
            // the whole batch. Failures are recorded as failures and never
            // count as done, so a resumed run retries them.
            journal.append(&completion_records(
                group,
                &result.files_failed[failed_before..],
            ))?;

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

        // The sparse phase: encode every row that lacks a vector from the
        // current model. A failed file keeps its dense rows (their text is
        // fine) and is retried by the next sync like any skipped file.
        let dense_failures = result.files_failed.len();
        let mut sparse_complete = true;
        if let Some(encoder) = &self.sparse {
            let progress = self.progress.clone();
            let skip = skip_failures;
            let sparse = sparse_phase::catch_up(
                store,
                encoder.as_ref(),
                sparse_phase::ROWS_PER_BATCH,
                |failure| {
                    if !skip {
                        return Err(MinSyncError::Embedding(format!(
                            "{}: {}",
                            failure.path, failure.error
                        )));
                    }
                    if let Some(progress) = &progress {
                        progress.file_failed(failure);
                    }
                    Ok(())
                },
            )
            .await?;
            result.sparse_encoded = sparse.encoded;
            result.sparse_reused = sparse.reused;
            sparse_complete = sparse.failures.is_empty();
            for failure in sparse.failures {
                if !result.files_failed.iter().any(|f| f.path == failure.path) {
                    result.files_failed.push(failure);
                }
            }
            store.flush()?;
        }

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
        for failure in &result.files_failed[..dense_failures] {
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
            // Only a complete sparse phase proves every row is searchable by
            // this model; a partial one leaves the cursor as it was, so a
            // sparse query refuses the index until a sync finishes the job.
            sparse_id: match &self.sparse {
                Some(encoder) if sparse_complete => Some(encoder.id().to_string()),
                // No encoder this run: any chunk it wrote has no sparse
                // vector, so the index no longer covers every row.
                None if result.chunks_added == 0 => stored_cursor
                    .as_ref()
                    .and_then(|cursor| cursor.sparse_id.clone()),
                None => None,
                Some(_) => stored_cursor
                    .as_ref()
                    .and_then(|cursor| cursor.sparse_id.clone()),
            },
        }
        .save(&cursor_path)?;
        Journal::remove(&progress_path)?;
        Transaction::remove(&txn_path)?;

        result.elapsed_seconds = start.elapsed().as_secs_f64();
        if let Some(progress) = &self.progress {
            progress.done(&result, changes.len());
        }
        Ok(result)
    }
}

/// Stage identity in the run ledger header.
const VECTOR_STAGE: &str = "vector";

/// Completion records for one batch whose store commits have returned: a
/// committed (or deleted) record per file that made it, and a failure record
/// — never a completion — per file listed in `failures`.
fn completion_records(
    group: &[(BatchItem, Option<String>)],
    failures: &[FileFailure],
) -> Vec<Record> {
    let failed: std::collections::HashMap<&str, &FileFailure> = failures
        .iter()
        .map(|failure| (failure.path.as_str(), failure))
        .collect();
    group
        .iter()
        .map(|(item, version)| {
            if let Some(failure) = failed.get(item.path.as_str()) {
                return Record::failed(
                    &item.path,
                    version.as_deref(),
                    format!("{}: {}", failure.stage, failure.error),
                );
            }
            match version {
                Some(version) if item.kind == BatchKind::Index => {
                    Record::committed(&item.path, version, None)
                }
                _ => Record::deleted(&item.path),
            }
        })
        .collect()
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
        index_batch(root, &[item], context, result, &mut 0).await
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

    struct ForbiddenSparseOnlyEmbedder;

    #[async_trait::async_trait]
    impl Embedder for ForbiddenSparseOnlyEmbedder {
        fn id(&self) -> &str {
            "none"
        }

        fn is_sparse_only(&self) -> bool {
            true
        }

        async fn embed(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>> {
            panic!("a sparse-only sync must never embed");
        }
    }

    #[tokio::test]
    async fn test_sync_sparse_only_stores_chunks_without_embedding() {
        let (dir, sync, chunker, _embedder, mut store) = fixture();
        std::fs::write(dir.path().join("a.txt"), "alpha beta gamma delta").expect("write file");
        sync.init(false, "none", "recursive")
            .expect("init succeeds");

        let result = sync
            .sync(
                &chunker,
                &ForbiddenSparseOnlyEmbedder,
                &mut store,
                true,
                false,
                false,
            )
            .await
            .expect("sync succeeds without embedding");

        assert!(result.chunks_added > 0);
        assert_eq!(result.embedded_texts, 0);
        assert_eq!(result.embedding_api_calls, 0);
        assert_eq!(store.doc_count(), result.chunks_added);
        let cursor = Cursor::load(&sync.minsync_dir.join("cursor.json")).expect("cursor");
        assert_eq!(cursor.embedder_id, "none");
        let placeholder = vec![0.0; crate::config::SPARSE_ONLY_DIMENSION];
        assert!(store
            .all_embeddings()
            .iter()
            .all(|embedding| embedding == &placeholder));
    }

    #[tokio::test]
    async fn test_init_none_sets_placeholder_dimension() {
        let (_dir, sync, _chunker, _embedder, _store) = fixture();
        let config = sync.init(false, "none", "recursive").expect("init");
        assert_eq!(config.embedder.id, "none");
        assert!(config.is_sparse_only());
        assert_eq!(
            config.vectorstore.options["dimension"].as_integer(),
            Some(crate::config::SPARSE_ONLY_DIMENSION as i64)
        );
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

    /// Always emits a non-finite vector for texts containing `poison` and
    /// counts single-text calls; `deterministic` mirrors in-process backends.
    struct CountingPoisonEmbedder {
        deterministic: bool,
        single_calls: std::sync::atomic::AtomicUsize,
    }

    #[async_trait::async_trait]
    impl Embedder for CountingPoisonEmbedder {
        fn id(&self) -> &str {
            "counting-poison"
        }

        fn is_deterministic(&self) -> bool {
            self.deterministic
        }

        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            if texts.len() == 1 {
                self.single_calls
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
            PoisonEmbedder.embed(texts).await
        }
    }

    async fn single_calls_until_non_finite_failure(deterministic: bool) -> usize {
        let (dir, _sync, chunker, _embedder, mut store) = fixture();
        let mut config = Config::default_for("source-1");
        config.sync.on_file_error = crate::config::OnFileError::Fail;
        config.embedder.max_retries = 3;
        std::fs::write(dir.path().join("bad.txt"), "poison").expect("write file");
        let embedder = CountingPoisonEmbedder {
            deterministic,
            single_calls: std::sync::atomic::AtomicUsize::new(0),
        };
        let mut result = empty_sync_result(false, false);
        let context = SyncFileContext {
            config: &config,
            chunker: &chunker,
            embedder: &embedder,
            store: &mut store,
            sync_token: "token",
        };

        index_file(dir.path(), "bad.txt", context, &mut result)
            .await
            .expect_err("persistent non-finite vector must fail");

        embedder
            .single_calls
            .load(std::sync::atomic::Ordering::SeqCst)
    }

    #[tokio::test]
    async fn test_deterministic_embedder_is_not_re_run_on_identical_non_finite_input() {
        // The initial embed of a lone chunk is itself a single-text call, so
        // one extra single-chunk attempt means 2 calls in total.
        assert_eq!(single_calls_until_non_finite_failure(true).await, 2);
    }

    #[tokio::test]
    async fn test_non_deterministic_embedder_keeps_max_retries_for_non_finite_chunk() {
        assert_eq!(single_calls_until_non_finite_failure(false).await, 4);
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
        fn rows_missing_sparse(
            &self,
            model: &str,
            limit: usize,
        ) -> Result<Vec<crate::vectorstore::SparseTarget>> {
            self.inner.rows_missing_sparse(model, limit)
        }
        fn count_missing_sparse(&self, model: &str) -> Result<usize> {
            self.inner.count_missing_sparse(model)
        }
        fn set_sparse(
            &mut self,
            model: &str,
            rows: &[(String, crate::sparse::SparseVector)],
        ) -> Result<()> {
            self.inner.set_sparse(model, rows)
        }
        fn fetch_sparse_by_content_hash(
            &self,
            model: &str,
            hashes: &[String],
        ) -> Result<std::collections::HashMap<String, crate::sparse::SparseVector>> {
            self.inner.fetch_sparse_by_content_hash(model, hashes)
        }
        fn query_sparse(
            &self,
            model: &str,
            vector: &crate::sparse::SparseVector,
            filter: Option<&Filter>,
            topk: usize,
        ) -> Result<Vec<QueryHit>> {
            self.inner.query_sparse(model, vector, filter, topk)
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
            sparse_id: None,
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

    /// The store commit of batch 3 succeeds, then the process dies before
    /// the journal append (the at-least-once window): the batch is redone,
    /// and because the commit is idempotent nothing is duplicated.
    #[tokio::test]
    async fn test_crash_between_store_commit_and_journal_append_is_repaired_by_the_retry() {
        let (dir, sync, chunker, _embedder, _store) = fixture();
        write_numbered_files(&dir, 6);
        init_sync(&sync);
        set_batch_files(&sync, 2);
        set_on_file_error(&sync, crate::config::OnFileError::Fail);
        let mut store = SpyStore::default();

        crate::incremental::journal::fail_append_after(2);
        let error = sync
            .sync(&chunker, &MockEmbedder, &mut store, true, false, false)
            .await
            .expect_err("the process dies before the third record");
        assert!(error.to_string().contains("injected crash"), "{error}");

        assert_eq!(
            store.inner.all_paths().len(),
            6,
            "all three batches reached the store"
        );
        let progress_path = sync.minsync_dir.join(JOURNAL_FILE);
        let ledger = match Journal::load(&progress_path) {
            crate::incremental::journal::Load::Loaded(ledger) => ledger,
            other => panic!("ledger must survive the crash: {other:?}"),
        };
        assert_eq!(
            ledger.records().count(),
            4,
            "only two batches were recorded"
        );
        assert!(sync.minsync_dir.join("txn.json").exists());
        assert!(!sync.minsync_dir.join("cursor.json").exists());

        let (embedder, embedded) = counting_embedder();
        let result = sync
            .sync(&chunker, &embedder, &mut store, false, false, false)
            .await
            .expect("retry repairs the missing record");

        assert_eq!(result.files_resumed, 4);
        assert_eq!(result.files_processed, 2, "the unrecorded batch is redone");
        assert_eq!(embedded.load(Ordering::SeqCst), result.embedded_texts);
        assert_eq!(
            store.doc_count(),
            uninterrupted_doc_count(&chunker, 6).await,
            "an idempotent re-commit must not duplicate rows"
        );
        assert!(sync.minsync_dir.join("cursor.json").exists());
        assert!(!progress_path.exists());
    }

    #[test]
    fn test_completion_records_keep_successes_failures_and_deletions_apart() {
        use crate::types::{FileFailure, FileStage};
        let item = |path: &str, kind| BatchItem {
            path: path.to_string(),
            kind,
            purge_first: false,
        };
        let group = vec![
            (item("a.txt", BatchKind::Index), Some("h1".to_string())),
            (item("b.txt", BatchKind::Index), Some("h2".to_string())),
            (item("gone.txt", BatchKind::Delete), None),
        ];
        let failures = vec![FileFailure {
            path: "b.txt".to_string(),
            stage: FileStage::Embed,
            error: "request too large".to_string(),
        }];

        assert_eq!(
            completion_records(&group, &failures),
            vec![
                Record::committed("a.txt", "h1", None),
                Record::failed("b.txt", Some("h2"), "embed: request too large"),
                Record::deleted("gone.txt"),
            ],
            "a failed file is never recorded as committed"
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

        index_batch(dir.path(), &items, context, &mut result, &mut 0)
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

    /// Embeds normally for its first `healthy_calls` calls, then is down.
    struct DiesAfter {
        calls: AtomicUsize,
        healthy_calls: usize,
    }

    #[async_trait::async_trait]
    impl Embedder for DiesAfter {
        fn id(&self) -> &str {
            "mock"
        }
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
            if call > self.healthy_calls {
                return Err(MinSyncError::Embedding("service unavailable".into()));
            }
            MockEmbedder.embed(texts).await
        }
    }

    #[tokio::test]
    async fn test_embedder_outage_after_a_successful_batch_still_aborts_quickly() {
        let (dir, sync, chunker, _embedder, mut store) = fixture();
        write_numbered_files(&dir, 20);
        init_sync(&sync);
        set_batch_files(&sync, 2);
        let dying = DiesAfter {
            calls: AtomicUsize::new(0),
            healthy_calls: 1,
        };

        let error = sync
            .sync(&chunker, &dying, &mut store, true, false, false)
            .await
            .expect_err("outage after the first batch must abort");

        assert_eq!(error.exit_code(), 5, "{error}");
        assert!(error.to_string().contains("service unavailable"), "{error}");
        let calls = dying.calls.load(Ordering::SeqCst);
        // 1 healthy + whole batch + per-file/bisect calls + 3 failing chunks x
        // `max_retries` (3) attempts each; without the streak every remaining
        // batch and chunk is retried (well over 30).
        assert!(
            calls <= 14,
            "embedder was hit {calls} times; the run kept going after the outage"
        );
        assert_eq!(
            store.all_paths().len(),
            2,
            "the healthy batch stays committed"
        );
        assert!(sync.minsync_dir.join("progress.jsonl").exists());
        assert!(!sync.minsync_dir.join("cursor.json").exists());
    }

    #[tokio::test]
    async fn test_streak_spans_single_file_batches() {
        let (dir, sync, chunker, _embedder, mut store) = fixture();
        write_numbered_files(&dir, 20);
        init_sync(&sync);
        set_batch_files(&sync, 1);
        let dying = DiesAfter {
            calls: AtomicUsize::new(0),
            healthy_calls: 2,
        };

        let error = sync
            .sync(&chunker, &dying, &mut store, true, false, false)
            .await
            .expect_err("outage must abort across one-file batches");

        assert_eq!(error.exit_code(), 5, "{error}");
        let calls = dying.calls.load(Ordering::SeqCst);
        assert!(calls <= 13, "embedder was hit {calls} times");
        assert_eq!(store.all_paths().len(), 2);
    }

    #[tokio::test]
    async fn test_isolated_poison_file_after_success_is_still_skipped() {
        let (dir, sync, chunker, _embedder, mut store) = fixture();
        write_good_and_poison(&dir);
        init_sync(&sync);
        set_batch_files(&sync, 1);

        let result = sync
            .sync(&chunker, &RejectingEmbedder, &mut store, true, false, false)
            .await
            .expect("one bad file among good ones is skipped");

        assert_eq!(result.files_failed.len(), 1);
        assert_eq!(result.files_failed[0].path, "c-bad.txt");
        assert_eq!(result.files_processed, 4);
    }

    // ---- issue #114: a failed chunk counts once toward the systemic streak -

    /// Rejects any `embed` call that contains a chunk with `BADCHUNK`, even
    /// when the call also carries good chunks (a request-level failure such as
    /// an over-limit token total). Records every call's size.
    struct RejectsBadChunks {
        calls: std::sync::Mutex<Vec<usize>>,
    }

    impl RejectsBadChunks {
        fn new() -> Self {
            Self {
                calls: std::sync::Mutex::new(Vec::new()),
            }
        }
        fn call_sizes(&self) -> Vec<usize> {
            self.calls.lock().unwrap().clone()
        }
    }

    #[async_trait::async_trait]
    impl Embedder for RejectsBadChunks {
        fn id(&self) -> &str {
            "mock"
        }
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            self.calls.lock().unwrap().push(texts.len());
            if texts.iter().any(|text| text.contains("BADCHUNK")) {
                return Err(MinSyncError::Embedding("request too large".into()));
            }
            MockEmbedder.embed(texts).await
        }
    }

    /// A file whose first chunk is bad and whose other chunks are fine.
    fn write_file_with_one_bad_chunk(dir: &TempDir, name: &str) {
        std::fs::write(
            dir.path().join(name),
            format!("BADCHUNK {name} {MULTI_CHUNK} {MULTI_CHUNK} {MULTI_CHUNK}"),
        )
        .expect("write file with a bad chunk");
    }

    fn set_max_retries(sync: &MinSync, retries: usize) {
        let path = sync.minsync_dir.join("config.toml");
        let mut config = Config::load(&path).expect("load config");
        config.embedder.max_retries = retries;
        config.save(&path).expect("save config");
    }

    #[tokio::test]
    async fn test_bad_chunks_in_adjacent_files_do_not_abort_a_healthy_run() {
        let (dir, sync, chunker, _embedder, mut store) = fixture();
        // Four adjacent bad files, then good ones: with the per-file streak
        // (limit 3) the run aborts before reaching the good files.
        for name in ["a1.txt", "a2.txt", "a3.txt", "a4.txt"] {
            write_file_with_one_bad_chunk(&dir, name);
        }
        for name in ["b1.txt", "b2.txt", "b3.txt"] {
            std::fs::write(dir.path().join(name), format!("{MULTI_CHUNK} {name}"))
                .expect("write good file");
        }
        init_sync(&sync);
        set_batch_files(&sync, 25);
        set_max_retries(&sync, 3);
        let embedder = RejectsBadChunks::new();

        let result = sync
            .sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("bad chunks surrounded by good ones must not abort the run");

        assert_eq!(
            store.all_paths(),
            vec!["b1.txt", "b2.txt", "b3.txt"],
            "the good files are committed"
        );
        let mut failed: Vec<&str> = result
            .files_failed
            .iter()
            .map(|failure| failure.path.as_str())
            .collect();
        failed.sort_unstable();
        assert_eq!(failed, vec!["a1.txt", "a2.txt", "a3.txt", "a4.txt"]);
        assert!(result.partial);
    }

    #[tokio::test]
    async fn test_a_failed_file_names_the_bad_chunk_and_keeps_its_good_chunks_out_of_the_store() {
        let (dir, sync, chunker, _embedder, mut store) = fixture();
        write_file_with_one_bad_chunk(&dir, "a.txt");
        std::fs::write(dir.path().join("z.txt"), format!("{MULTI_CHUNK} z")).expect("write");
        init_sync(&sync);
        set_max_retries(&sync, 2);
        let embedder = RejectsBadChunks::new();

        let result = sync
            .sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("sync completes");

        assert_eq!(result.files_failed.len(), 1);
        assert_eq!(result.files_failed[0].path, "a.txt");
        assert_eq!(result.files_failed[0].stage, FileStage::Embed);
        assert!(
            result.files_failed[0].error.contains("chunk 0"),
            "the report points at the bad chunk: {}",
            result.files_failed[0].error
        );
        assert_eq!(
            store.all_paths(),
            vec!["z.txt"],
            "a partly embedded file is not written"
        );
    }

    #[tokio::test]
    async fn test_one_bad_chunk_is_attempted_max_retries_times_and_counts_once() {
        let (dir, sync, chunker, _embedder, mut store) = fixture();
        write_file_with_one_bad_chunk(&dir, "a.txt");
        init_sync(&sync);
        set_max_retries(&sync, 3);
        let embedder = RejectsBadChunks::new();

        sync.sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("a single bad chunk is a skipped file, not an abort");

        let single_chunk_calls = embedder.call_sizes().iter().filter(|&&n| n == 1).count();
        assert_eq!(
            single_chunk_calls,
            3,
            "the bad chunk is sent alone exactly max_retries times: {:?}",
            embedder.call_sizes()
        );
    }

    #[tokio::test]
    async fn test_every_chunk_failing_is_an_outage_and_aborts_after_three_chunks() {
        let (dir, sync, chunker, _embedder, mut store) = fixture();
        for index in 0..10 {
            std::fs::write(
                dir.path().join(format!("f{index}.txt")),
                format!("BADCHUNK {index}"),
            )
            .expect("write");
        }
        init_sync(&sync);
        set_max_retries(&sync, 1);
        let embedder = RejectsBadChunks::new();

        let error = sync
            .sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect_err("three failed chunks in a row abort the run");

        assert_eq!(error.exit_code(), 5, "{error}");
        let calls = embedder.call_sizes().len();
        assert!(
            calls <= 8,
            "the embedder was hit {calls} times after the outage began"
        );
    }

    /// Fails every call that carries more than one chunk or a `BADCHUNK`
    /// chunk, so a lone good chunk succeeds between failures.
    #[tokio::test]
    async fn test_a_good_chunk_between_bad_chunks_resets_the_streak() {
        let (dir, sync, chunker, _embedder, mut store) = fixture();
        // Each file: one bad chunk then good text. Isolation embeds the good
        // chunks alone (success) between the bad ones, so the streak never
        // reaches three even with five bad files.
        for index in 0..5 {
            write_file_with_one_bad_chunk(&dir, &format!("f{index}.txt"));
        }
        init_sync(&sync);
        set_batch_files(&sync, 1);
        set_max_retries(&sync, 1);
        let embedder = RejectsBadChunks::new();

        let result = sync
            .sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("interleaved successes keep the run alive");

        assert_eq!(result.files_failed.len(), 5);
        assert!(store.all_paths().is_empty());
    }

    /// Encodes a chunk as the term count of its words; rejects chunks that
    /// contain a configured word. Counts every `encode` call it answers.
    struct CountingSparse {
        model: &'static str,
        reject: Option<&'static str>,
        encoded_texts: std::sync::Mutex<Vec<String>>,
    }

    impl CountingSparse {
        fn new(model: &'static str) -> Arc<Self> {
            Arc::new(Self {
                model,
                reject: None,
                encoded_texts: std::sync::Mutex::new(Vec::new()),
            })
        }

        fn rejecting(model: &'static str, word: &'static str) -> Arc<Self> {
            Arc::new(Self {
                model,
                reject: Some(word),
                encoded_texts: std::sync::Mutex::new(Vec::new()),
            })
        }

        fn encoded(&self) -> usize {
            self.encoded_texts.lock().unwrap().len()
        }
    }

    #[async_trait::async_trait]
    impl SparseEncoder for CountingSparse {
        fn id(&self) -> &str {
            self.model
        }

        async fn encode(&self, texts: &[String]) -> Result<Vec<crate::sparse::SparseVector>> {
            if let Some(word) = self.reject {
                if texts.iter().any(|text| text.contains(word)) {
                    return Err(MinSyncError::Embedding(format!("cannot encode `{word}`")));
                }
            }
            self.encoded_texts
                .lock()
                .unwrap()
                .extend(texts.iter().cloned());
            Ok(texts
                .iter()
                .map(|text| {
                    crate::sparse::SparseVector::from_pairs(
                        text.split_whitespace().map(|word| (word.len() as u32, 1.0)),
                    )
                })
                .collect())
        }

        async fn encode_query(&self, text: &str) -> Result<crate::sparse::SparseVector> {
            Ok(self.encode(&[text.to_string()]).await?.remove(0))
        }
    }

    fn cursor_of(sync: &MinSync) -> Cursor {
        Cursor::load(&sync.minsync_dir.join("cursor.json")).expect("cursor")
    }

    #[tokio::test]
    async fn test_sync_with_an_encoder_encodes_every_chunk_and_records_the_model() {
        let (dir, sync, chunker, embedder, mut store) = fixture();
        std::fs::write(dir.path().join("a.txt"), "alpha beta gamma").expect("write a");
        std::fs::write(dir.path().join("b.txt"), "delta epsilon zeta eta").expect("write b");
        sync.init(false, "openai:text-embedding-3-small", "recursive")
            .expect("init");
        let encoder = CountingSparse::new("m1");
        let sync = sync.with_sparse_encoder(encoder.clone());

        let result = sync
            .sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("sync");

        assert!(result.chunks_added > 0);
        assert_eq!(result.sparse_encoded, result.chunks_added);
        assert_eq!(store.count_missing_sparse("m1").unwrap(), 0);
        assert_eq!(cursor_of(&sync).sparse_id.as_deref(), Some("m1"));
        assert!(!result.partial);
    }

    #[tokio::test]
    async fn test_a_sync_with_nothing_changed_encodes_nothing() {
        let (dir, sync, chunker, embedder, mut store) = fixture();
        std::fs::write(dir.path().join("a.txt"), "alpha beta gamma").expect("write a");
        sync.init(false, "openai:text-embedding-3-small", "recursive")
            .expect("init");
        let encoder = CountingSparse::new("m1");
        let sync = sync.with_sparse_encoder(encoder.clone());
        sync.sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("first");
        let before = encoder.encoded();

        let result = sync
            .sync(&chunker, &embedder, &mut store, false, false, false)
            .await
            .expect("second");

        assert!(result.already_up_to_date);
        assert_eq!(encoder.encoded(), before);
    }

    #[tokio::test]
    async fn test_only_the_changed_files_chunks_are_encoded_on_an_incremental_sync() {
        let (dir, sync, chunker, embedder, mut store) = fixture();
        std::fs::write(dir.path().join("a.txt"), "alpha beta gamma").expect("write a");
        std::fs::write(dir.path().join("b.txt"), "delta epsilon zeta").expect("write b");
        sync.init(false, "openai:text-embedding-3-small", "recursive")
            .expect("init");
        let encoder = CountingSparse::new("m1");
        let sync = sync.with_sparse_encoder(encoder.clone());
        sync.sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("first");
        let before = encoder.encoded();

        std::fs::write(dir.path().join("b.txt"), "completely different words").expect("edit b");
        let result = sync
            .sync(&chunker, &embedder, &mut store, false, false, false)
            .await
            .expect("incremental");

        assert_eq!(result.sparse_encoded, result.chunks_added);
        assert_eq!(encoder.encoded() - before, result.chunks_added);
        assert_eq!(store.count_missing_sparse("m1").unwrap(), 0);
    }

    #[tokio::test]
    async fn test_an_index_built_without_sparse_is_upgraded_by_a_plain_sync() {
        let (dir, sync, chunker, embedder, mut store) = fixture();
        std::fs::write(dir.path().join("a.txt"), "alpha beta gamma").expect("write a");
        std::fs::write(dir.path().join("b.txt"), "delta epsilon zeta").expect("write b");
        sync.init(false, "openai:text-embedding-3-small", "recursive")
            .expect("init");
        // The BM25 era: a completed sync with no sparse encoder.
        sync.sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("legacy");
        assert_eq!(cursor_of(&sync).sparse_id, None);

        // Upgrade: same files, now with an encoder, no --full. The dense
        // embedder must not be called at all.
        let encoder = CountingSparse::new("m1");
        let upgraded = MinSync::new(dir.path().to_path_buf()).with_sparse_encoder(encoder.clone());
        let result = upgraded
            .sync(&chunker, &NoDenseCalls, &mut store, false, false, false)
            .await
            .expect("upgrade sync");

        assert!(!result.already_up_to_date, "the sparse backlog is work");
        assert_eq!(result.files_processed, 0, "no file is re-read");
        assert_eq!(result.chunks_added, 0, "no chunk is re-embedded");
        assert_eq!(result.embedded_texts, 0);
        assert!(result.sparse_encoded > 0);
        assert_eq!(store.count_missing_sparse("m1").unwrap(), 0);
        assert_eq!(cursor_of(&upgraded).sparse_id.as_deref(), Some("m1"));

        let again = upgraded
            .sync(&chunker, &NoDenseCalls, &mut store, false, false, false)
            .await
            .expect("settled");
        assert!(again.already_up_to_date);
    }

    /// Same identity as `MockEmbedder`, but any call to embed is a bug.
    struct NoDenseCalls;

    #[async_trait::async_trait]
    impl Embedder for NoDenseCalls {
        fn id(&self) -> &str {
            "mock"
        }

        async fn embed(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>> {
            panic!("the sparse upgrade must not call the dense embedder");
        }
    }

    #[tokio::test]
    async fn test_a_dry_run_never_encodes() {
        let (dir, sync, chunker, embedder, mut store) = fixture();
        std::fs::write(dir.path().join("a.txt"), "alpha beta gamma").expect("write a");
        sync.init(false, "openai:text-embedding-3-small", "recursive")
            .expect("init");
        let encoder = CountingSparse::new("m1");
        let sync = sync.with_sparse_encoder(encoder.clone());

        let result = sync
            .sync(&chunker, &embedder, &mut store, true, true, false)
            .await
            .expect("dry run");

        assert!(result.dry_run);
        assert_eq!(encoder.encoded(), 0);
        assert_eq!(store.doc_count(), 0);
    }

    #[tokio::test]
    async fn test_changing_the_model_re_encodes_every_row_without_touching_dense_vectors() {
        let (dir, sync, chunker, embedder, mut store) = fixture();
        std::fs::write(dir.path().join("a.txt"), "alpha beta gamma").expect("write a");
        sync.init(false, "openai:text-embedding-3-small", "recursive")
            .expect("init");
        let first = sync.with_sparse_encoder(CountingSparse::new("old"));
        first
            .sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("first");
        let dense_before = store.all_embeddings();

        let second_encoder = CountingSparse::new("new");
        let second =
            MinSync::new(dir.path().to_path_buf()).with_sparse_encoder(second_encoder.clone());
        let result = second
            .sync(&chunker, &embedder, &mut store, false, false, false)
            .await
            .expect("model change");

        assert!(result.sparse_encoded > 0);
        assert_eq!(result.embedded_texts, 0, "dense side is untouched");
        assert_eq!(store.count_missing_sparse("new").unwrap(), 0);
        assert_eq!(cursor_of(&second).sparse_id.as_deref(), Some("new"));
        let mut after = store.all_embeddings();
        let mut before = dense_before;
        after.sort_by(|a, b| a.partial_cmp(b).unwrap());
        before.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(after, before);
    }

    #[tokio::test]
    async fn test_a_chunk_the_encoder_rejects_fails_its_file_and_leaves_the_cursor_unsparse() {
        let (dir, sync, chunker, embedder, mut store) = fixture();
        std::fs::write(dir.path().join("good.txt"), "alpha beta gamma").expect("write good");
        std::fs::write(dir.path().join("bad.txt"), "poison words here").expect("write bad");
        sync.init(false, "openai:text-embedding-3-small", "recursive")
            .expect("init");
        let sync = sync.with_sparse_encoder(CountingSparse::rejecting("m1", "poison"));

        let result = sync
            .sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("skip policy completes");

        assert!(result.partial);
        assert_eq!(result.files_failed.len(), 1);
        assert_eq!(result.files_failed[0].path, "bad.txt");
        assert_eq!(result.files_failed[0].stage, FileStage::Sparse);
        assert!(result.files_failed[0].error.contains("cannot encode"));
        // The dense index is complete: a sparse failure must not cost it.
        assert_eq!(store.all_paths(), vec!["bad.txt", "good.txt"]);
        // Sparse search is refused until a sync finishes the job.
        assert_eq!(cursor_of(&sync).sparse_id, None);
        assert_eq!(store.count_missing_sparse("m1").unwrap(), 1);

        // The encoder recovers: the next sync encodes the leftover row and
        // only then records the model.
        let healed = CountingSparse::new("m1");
        let healed_sync =
            MinSync::new(dir.path().to_path_buf()).with_sparse_encoder(healed.clone());
        let retry = healed_sync
            .sync(&chunker, &embedder, &mut store, false, false, false)
            .await
            .expect("retry");
        assert!(!retry.partial, "{:?}", retry.files_failed);
        assert_eq!(healed.encoded(), 1, "only the leftover row");
        assert_eq!(cursor_of(&healed_sync).sparse_id.as_deref(), Some("m1"));
    }

    #[tokio::test]
    async fn test_a_sparse_failure_under_the_fail_policy_aborts_the_sync() {
        let (dir, sync, chunker, embedder, mut store) = fixture();
        std::fs::write(dir.path().join("bad.txt"), "poison words here").expect("write bad");
        sync.init(false, "openai:text-embedding-3-small", "recursive")
            .expect("init");
        let mut config = Config::load(&sync.minsync_dir.join("config.toml")).expect("config");
        config.sync.on_file_error = OnFileError::Fail;
        config
            .save(&sync.minsync_dir.join("config.toml"))
            .expect("save");
        let sync = sync.with_sparse_encoder(CountingSparse::rejecting("m1", "poison"));

        let error = sync
            .sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect_err("fail policy aborts on the first sparse failure");

        assert!(error.to_string().contains("bad.txt"), "{error}");
        assert_eq!(
            cursor_of_or_none(&sync),
            None,
            "a failed run records nothing"
        );
    }

    fn cursor_of_or_none(sync: &MinSync) -> Option<Cursor> {
        Cursor::load(&sync.minsync_dir.join("cursor.json")).ok()
    }

    #[tokio::test]
    async fn test_a_dense_only_sync_after_a_sparse_one_forgets_that_the_index_is_covered() {
        let (dir, sync, chunker, embedder, mut store) = fixture();
        std::fs::write(dir.path().join("a.txt"), "alpha beta gamma").expect("write a");
        sync.init(false, "openai:text-embedding-3-small", "recursive")
            .expect("init");
        let with_sparse = sync.with_sparse_encoder(CountingSparse::new("m1"));
        with_sparse
            .sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("first");
        assert_eq!(cursor_of(&with_sparse).sparse_id.as_deref(), Some("m1"));

        // A library user (or an old binary) syncs without an encoder and adds a file.
        std::fs::write(dir.path().join("b.txt"), "delta epsilon zeta").expect("write b");
        let dense_only = MinSync::new(dir.path().to_path_buf());
        dense_only
            .sync(&chunker, &embedder, &mut store, false, false, false)
            .await
            .expect("dense-only");

        // b.txt's chunks have no sparse vector, so coverage must not be claimed.
        assert_eq!(cursor_of(&dense_only).sparse_id, None);
    }

    #[tokio::test]
    async fn test_a_dense_only_sync_that_adds_nothing_keeps_the_recorded_model() {
        let (dir, sync, chunker, embedder, mut store) = fixture();
        std::fs::write(dir.path().join("a.txt"), "alpha beta gamma").expect("write a");
        sync.init(false, "openai:text-embedding-3-small", "recursive")
            .expect("init");
        let with_sparse = sync.with_sparse_encoder(CountingSparse::new("m1"));
        with_sparse
            .sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("first");

        let dense_only = MinSync::new(dir.path().to_path_buf());
        dense_only
            .sync(&chunker, &embedder, &mut store, false, false, false)
            .await
            .expect("nothing to do");

        assert_eq!(cursor_of(&dense_only).sparse_id.as_deref(), Some("m1"));
    }

    #[tokio::test]
    async fn test_duplicate_chunk_text_across_files_is_encoded_once() {
        let (dir, sync, chunker, embedder, mut store) = fixture();
        for name in ["a.txt", "b.txt", "c.txt"] {
            std::fs::write(dir.path().join(name), "the very same sentence").expect("write");
        }
        sync.init(false, "openai:text-embedding-3-small", "recursive")
            .expect("init");
        let encoder = CountingSparse::new("m1");
        let sync = sync.with_sparse_encoder(encoder.clone());

        let result = sync
            .sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("sync");

        assert_eq!(result.chunks_added, 3);
        assert_eq!(encoder.encoded(), 1);
        assert_eq!(result.sparse_encoded, 1);
        assert_eq!(result.sparse_reused, 2);
    }

    #[tokio::test]
    async fn test_heading_context_is_the_text_both_encoders_and_the_store_read() {
        let (dir, sync, _chunker, _embedder, mut store) = fixture();
        std::fs::write(
            dir.path().join("guide.md"),
            "# Guide\n\n## Install\n\nRun the installer.\n",
        )
        .expect("write guide");
        std::fs::write(dir.path().join("notes.txt"), "# not a heading\nplain\n").expect("write");
        sync.init(false, "openai:text-embedding-3-small", "recursive")
            .expect("init");
        let config_path = sync.minsync_dir.join("config.toml");
        let mut config = Config::load(&config_path).expect("load config");
        config.chunker.options.heading_context = true;
        config.chunker.options.max_chunk_size = 160;
        config.save(&config_path).expect("save config");
        let chunker = crate::chunker::create_chunker(&config).expect("chunker");
        let embedder = RecordingEmbedder::default();
        let encoder = CountingSparse::new("m1");
        let sync = sync.with_sparse_encoder(encoder.clone());

        sync.sync(chunker.as_ref(), &embedder, &mut store, true, false, false)
            .await
            .expect("sync");

        let mut stored: Vec<String> = store
            .documents()
            .expect("documents")
            .into_iter()
            .map(|doc| doc.text)
            .collect();
        stored.sort();
        let mut dense = embedder.calls().concat();
        dense.sort();
        let mut sparse = encoder.encoded_texts.lock().unwrap().clone();
        sparse.sort();
        // Normalization trims each file's trailing newline before chunking.
        assert_eq!(
            stored,
            [
                "guide.md\n# Guide\n\n## Install\n\nRun the installer.",
                "notes.txt\n# not a heading\nplain",
            ]
        );
        assert_eq!(dense, stored);
        assert_eq!(sparse, stored);
    }
}
