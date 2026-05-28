use crate::chunker::Chunker;
use crate::config::Config;
use crate::embedder::Embedder;
use crate::error::{MinSyncError, Result};
use crate::id::{compute_doc_id, content_hash};
use crate::manifest::{FileChange, Manifest};
use crate::normalize::normalize_text;
use crate::state::{Cursor, FileLock, Transaction};
use crate::types::SyncResult;
use crate::vectorstore::{Document, DocumentUpdate, Filter, VectorStore};
use std::collections::HashMap;
use std::path::PathBuf;

pub struct MinSync {
    root: PathBuf,
    minsync_dir: PathBuf,
}

impl MinSync {
    pub fn new(root: PathBuf) -> Self {
        let minsync_dir = root.join(".minsync");
        Self { root, minsync_dir }
    }

    /// Initialize `.minsync/` with a default config and baseline manifest.
    pub fn init(&self, force: bool) -> Result<Config> {
        if self.minsync_dir.exists() && !force {
            return Err(MinSyncError::AlreadyInitialized);
        }

        std::fs::create_dir_all(&self.minsync_dir)?;
        let source_id = uuid::Uuid::new_v4().to_string();
        let config = Config::default_for(&source_id);

        config.save(&self.minsync_dir.join("config.toml"))?;
        Manifest::scan(&self.root, &source_id)?.save(&self.minsync_dir.join("manifest.json"))?;

        Ok(config)
    }

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
        let config_path = self.minsync_dir.join("config.toml");
        if !config_path.exists() {
            return Err(MinSyncError::NotInitialized);
        }

        let config = Config::load(&config_path)?;
        let _lock = FileLock::acquire(&self.minsync_dir.join("lock"), wait_lock)?;

        let txn_path = self.minsync_dir.join("txn.json");
        let cursor_path = self.minsync_dir.join("cursor.json");
        let manifest_path = self.minsync_dir.join("manifest.json");

        if txn_path.exists() {
            Transaction::remove(&txn_path)?;
        }

        let old_manifest = if full || !manifest_path.exists() {
            Manifest::new(&config.source_id)
        } else {
            Manifest::load(&manifest_path)?
        };
        let new_manifest = Manifest::scan(&self.root, &config.source_id)?;
        let changes = Manifest::diff(&old_manifest, &new_manifest);

        if changes.is_empty() && !full {
            return Ok(empty_sync_result(dry_run, true));
        }

        if dry_run {
            let files_processed_paths = changes.iter().map(change_path).collect();
            return Ok(SyncResult {
                files_processed_paths,
                dry_run: true,
                already_up_to_date: false,
                ..empty_sync_result(true, false)
            });
        }

        let sync_token = uuid::Uuid::new_v4().to_string().replace('-', "");
        Transaction::new(
            &config.source_id,
            &sync_token,
            Some(old_manifest.manifest_hash()),
            &new_manifest.manifest_hash(),
        )
        .save(&txn_path)?;

        let start = std::time::Instant::now();
        let mut result = SyncResult {
            files_processed: 0,
            chunks_added: 0,
            chunks_updated: 0,
            chunks_deleted: 0,
            dry_run: false,
            already_up_to_date: false,
            files_processed_paths: Vec::new(),
            elapsed_seconds: 0.0,
            embedding_api_calls: 0,
            embedded_texts: 0,
            estimated_tokens: 0,
        };

        for change in &changes {
            match change {
                FileChange::Added(path) | FileChange::Modified(path) => {
                    self.sync_file(
                        path,
                        &config,
                        chunker,
                        embedder,
                        store,
                        &sync_token,
                        &mut result,
                    )
                    .await?;
                }
                FileChange::Deleted(path) => {
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

        store.flush()?;
        Cursor {
            source_id: config.source_id.clone(),
            last_synced_at: chrono::Utc::now().to_rfc3339(),
            manifest_hash: new_manifest.manifest_hash(),
            chunk_schema_id: chunker.schema_id().to_string(),
            embedder_id: embedder.id().to_string(),
            collection_path: config.collection.path.clone(),
        }
        .save(&cursor_path)?;
        new_manifest.save(&manifest_path)?;
        Transaction::remove(&txn_path)?;

        result.elapsed_seconds = start.elapsed().as_secs_f64();
        Ok(result)
    }

    async fn sync_file(
        &self,
        path: &str,
        config: &Config,
        chunker: &dyn Chunker,
        embedder: &dyn Embedder,
        store: &mut dyn VectorStore,
        sync_token: &str,
        result: &mut SyncResult,
    ) -> Result<()> {
        let raw_text = match std::fs::read_to_string(self.root.join(path)) {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::InvalidData => String::new(),
            Err(error) => return Err(MinSyncError::Io(error)),
        };
        let text = normalize_text(&raw_text, &config.normalize);
        let chunks = chunker.chunk(&text, path)?;
        let schema_id = chunker.schema_id();
        let mut docs_to_embed = Vec::new();
        let mut dup_counter: HashMap<String, usize> = HashMap::new();

        for chunk in chunks {
            let chunk_content_hash = content_hash(&chunk.text);
            let dup_key = format!("{}\0{}", chunk_content_hash, chunk.heading_path);
            let dup_index = dup_counter.entry(dup_key).or_insert(0);
            let doc_id = compute_doc_id(
                &config.source_id,
                path,
                schema_id,
                &chunk.chunk_type,
                &chunk_content_hash,
                *dup_index,
            );
            *dup_index += 1;

            if store.fetch(std::slice::from_ref(&doc_id))?.is_empty() {
                docs_to_embed.push(Document {
                    id: doc_id,
                    embedding: Vec::new(),
                    text: chunk.text,
                    source_id: config.source_id.clone(),
                    path: path.to_string(),
                    chunk_schema_id: schema_id.to_string(),
                    chunk_type: chunk.chunk_type,
                    heading_path: chunk.heading_path,
                    content_hash: chunk_content_hash,
                    seen_token: sync_token.to_string(),
                });
            } else {
                store.update(&[DocumentUpdate {
                    id: doc_id,
                    seen_token: sync_token.to_string(),
                    path: path.to_string(),
                    heading_path: chunk.heading_path,
                }])?;
                result.chunks_updated += 1;
            }
        }

        if !docs_to_embed.is_empty() {
            let texts: Vec<String> = docs_to_embed.iter().map(|doc| doc.text.clone()).collect();
            let embeddings = embedder.embed(&texts).await?;
            if embeddings.len() != docs_to_embed.len() {
                return Err(MinSyncError::Embedding(format!(
                    "expected {} embeddings, got {}",
                    docs_to_embed.len(),
                    embeddings.len()
                )));
            }

            result.embedding_api_calls += 1;
            result.embedded_texts += texts.len();
            result.estimated_tokens += texts.iter().map(|text| text.len() / 4).sum::<usize>();

            for (doc, embedding) in docs_to_embed.iter_mut().zip(embeddings) {
                doc.embedding = embedding;
            }
            result.chunks_added += docs_to_embed.len();
            store.upsert(&docs_to_embed)?;
        }

        result.chunks_deleted += store.delete_by_filter(&Filter::And(vec![
            Filter::Eq("source_id".to_string(), config.source_id.clone()),
            Filter::Eq("path".to_string(), path.to_string()),
            Filter::Neq("seen_token".to_string(), sync_token.to_string()),
        ]))?;
        result.files_processed += 1;
        result.files_processed_paths.push(path.to_string());

        Ok(())
    }
}

fn change_path(change: &FileChange) -> String {
    match change {
        FileChange::Added(path) | FileChange::Modified(path) | FileChange::Deleted(path) => path.clone(),
    }
}

fn empty_sync_result(dry_run: bool, already_up_to_date: bool) -> SyncResult {
    SyncResult {
        files_processed: 0,
        chunks_added: 0,
        chunks_updated: 0,
        chunks_deleted: 0,
        dry_run,
        already_up_to_date,
        files_processed_paths: Vec::new(),
        elapsed_seconds: 0.0,
        embedding_api_calls: 0,
        embedded_texts: 0,
        estimated_tokens: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunker::chonkie::ChonkieChunker;
    use crate::embedder::Embedder;
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

    fn fixture() -> (TempDir, MinSync, ChonkieChunker, MockEmbedder, InMemoryStore) {
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

        let config = sync.init(false).expect("init succeeds");

        assert!(sync.minsync_dir.exists());
        assert!(sync.minsync_dir.join("config.toml").exists());
        assert!(sync.minsync_dir.join("manifest.json").exists());
        assert_eq!(config.version, 1);
    }

    #[test]
    fn test_init_already_initialized() {
        let (_dir, sync, _chunker, _embedder, _store) = fixture();
        sync.init(false).expect("first init succeeds");

        let result = sync.init(false);

        assert!(matches!(result, Err(MinSyncError::AlreadyInitialized)));
    }

    #[test]
    fn test_init_force_reinit() {
        let (_dir, sync, _chunker, _embedder, _store) = fixture();
        let first = sync.init(false).expect("first init succeeds");

        let second = sync.init(true).expect("force init succeeds");

        assert_ne!(first.source_id, second.source_id);
    }

    #[tokio::test]
    async fn test_sync_full_first_time() {
        let (dir, sync, chunker, embedder, mut store) = fixture();
        std::fs::write(dir.path().join("a.txt"), "alpha beta gamma").expect("write file");
        sync.init(false).expect("init succeeds");

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
        sync.init(false).expect("init succeeds");
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
        sync.init(false).expect("init succeeds");
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
        sync.init(false).expect("init succeeds");
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
    async fn test_sync_already_up_to_date() {
        let (dir, sync, chunker, embedder, mut store) = fixture();
        std::fs::write(dir.path().join("a.txt"), "alpha beta gamma").expect("write file");
        sync.init(false).expect("init succeeds");
        sync.sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("first sync succeeds");

        let result = sync
            .sync(&chunker, &embedder, &mut store, false, false, false)
            .await
            .expect("second sync succeeds");

        assert!(result.already_up_to_date);
        assert_eq!(result.files_processed, 0);
    }

    #[tokio::test]
    async fn test_sync_dry_run() {
        let (dir, sync, chunker, embedder, mut store) = fixture();
        std::fs::write(dir.path().join("a.txt"), "alpha beta gamma").expect("write file");
        sync.init(false).expect("init succeeds");

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
        let config = sync.init(false).expect("init succeeds");
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
