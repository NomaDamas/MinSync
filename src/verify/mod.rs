//! Index consistency verification.
//!
//! Module layout: [`status`](mod@status) reports sync state, [`health`]
//! checks the environment, [`sampling`] recomputes expected doc IDs, and
//! this file orchestrates `minsync verify` (basic checks + stale repair);
//! [`integrity`] finds near-identical vectors over differing texts.

mod health;
mod integrity;
mod sampling;
mod status;

pub use health::check;
pub use status::status;

use crate::chunker::Chunker;
use crate::config::Config;
use crate::error::{MinSyncError, Result};
use crate::manifest::Manifest;
use crate::state::Cursor;
use crate::types::{EmbeddingIntegrityResult, VerifyResult};
use crate::vectorstore::{Filter, VectorStore};
use integrity::check_embedding_integrity;
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

    // Paths the scan could not inspect are unknown, never stale: carrying
    // them keeps `--fix` from deleting their chunks (issue #79).
    let stored_manifest = Manifest::load(&minsync_dir.join("manifest.json")).ok();
    let (manifest, _, unknown_paths) = Manifest::scan_keeping_unknown_with_max_file_bytes(
        root,
        &config.source_id,
        None,
        stored_manifest.as_ref(),
        config.sync.max_file_bytes,
    )?;
    let unknown_set: HashSet<&str> = unknown_paths.iter().map(String::as_str).collect();
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
    let index_state = store.index_state()?;

    let mut sample_ok = true;
    let readable_manifest = {
        let mut readable = manifest.clone();
        readable
            .files
            .retain(|path, _| !unknown_set.contains(path.as_str()));
        readable
    };
    for path in sample_paths(&readable_manifest, sample) {
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
    // A sparse-only index stores a zero placeholder for every dense vector:
    // no pair has a cosine similarity, so reading every row finds nothing.
    let embedding_integrity = if config.is_sparse_only() {
        EmbeddingIntegrityResult::default()
    } else {
        check_embedding_integrity(store)?
    };
    basic_checks.insert(
        "embedding_integrity".to_string(),
        embedding_integrity.flagged_pairs.is_empty(),
    );

    Ok(VerifyResult {
        all_passed: basic_checks.values().all(|passed| *passed),
        basic_checks,
        embedding_integrity,
        fixed,
        index_state,
    })
}

fn texts_differ_substantially(left: &str, right: &str) -> bool {
    if left == right {
        return false;
    }
    let left_terms: HashSet<_> = left.split_whitespace().map(str::to_lowercase).collect();
    let right_terms: HashSet<_> = right.split_whitespace().map(str::to_lowercase).collect();
    let union = left_terms.union(&right_terms).count();
    union == 0 || left_terms.intersection(&right_terms).count() * 2 < union
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
    use crate::types::{EmbeddingIntegrityPair, SyncState};
    use crate::vectorstore::memory::InMemoryStore;
    use crate::vectorstore::similarity::cosine_similarity;
    use crate::vectorstore::{Document, DocumentUpdate, QueryHit};
    use async_trait::async_trait;
    use integrity::MAX_FLAGGED_PAIRS;
    use tempfile::TempDir;

    struct MockEmbedder;

    struct StateTrackingStore {
        inner: InMemoryStore,
        before_repair: crate::types::IndexState,
        after_repair: crate::types::IndexState,
        repaired: bool,
        full_scans: std::sync::atomic::AtomicUsize,
    }

    impl VectorStore for StateTrackingStore {
        fn scan_chunks(
            &self,
            after: Option<&str>,
            limit: usize,
        ) -> Result<Vec<crate::vectorstore::StoredChunk>> {
            self.inner.scan_chunks(after, limit)
        }
        fn upsert(&mut self, docs: &[Document]) -> Result<()> {
            self.inner.upsert(docs)
        }

        fn update(&mut self, updates: &[DocumentUpdate]) -> Result<()> {
            self.inner.update(updates)
        }

        fn fetch(&self, ids: &[String]) -> Result<Vec<Document>> {
            self.inner.fetch(ids)
        }

        fn documents(&self) -> Result<Vec<Document>> {
            self.full_scans
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.inner.documents()
        }

        fn delete_by_filter(&mut self, filter: &Filter) -> Result<usize> {
            let deleted = self.inner.delete_by_filter(filter)?;
            if deleted > 0 {
                self.repaired = true;
            }
            Ok(deleted)
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
            scope: Option<&crate::vectorstore::Filter>,
            limit: usize,
        ) -> Result<Vec<crate::vectorstore::SparseTarget>> {
            self.inner.rows_missing_sparse(model, scope, limit)
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
        ) -> Result<HashMap<String, crate::sparse::SparseVector>> {
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

        fn doc_count(&self) -> usize {
            self.inner.doc_count()
        }

        fn all_paths(&self) -> Vec<String> {
            self.inner.all_paths()
        }

        fn index_state(&self) -> Result<Option<crate::types::IndexState>> {
            Ok(Some(if self.repaired {
                self.after_repair.clone()
            } else {
                self.before_repair.clone()
            }))
        }
    }

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
    async fn test_status_first_sync_interrupted() {
        let (dir, sync, _chunker, _embedder, _store) = fixture();
        let config = sync
            .init(false, "openai:text-embedding-3-small", "recursive")
            .expect("init succeeds");
        Transaction::new(&config.source_id, "token", None, "sha256:next")
            .save(&dir.path().join(".minsync").join("txn.json"))
            .expect("save transaction");

        let result = status(&dir.path().join(".minsync"))
            .await
            .expect("status succeeds");

        assert_eq!(result.state, SyncState::Interrupted);
        assert_eq!(result.last_synced_at, None);
    }

    #[tokio::test]
    async fn test_check_all_pass() {
        let (dir, sync, _chunker, embedder, store) = fixture();
        sync.init(false, "openai:text-embedding-3-small", "recursive")
            .expect("init succeeds");

        let result = check(&dir.path().join(".minsync"), &embedder, None, &store)
            .await
            .expect("check succeeds");

        assert!(result.embedder_ok);
        assert!(result.vectorstore_ok);
        assert!(result.all_passed);
        assert!(result.errors.is_empty());
    }

    struct ProbeSparse(Result<()>);

    #[async_trait::async_trait]
    impl crate::sparse::SparseEncoder for ProbeSparse {
        fn id(&self) -> &str {
            "probe/sparse"
        }
        async fn encode(&self, texts: &[String]) -> Result<Vec<crate::sparse::SparseVector>> {
            Ok(texts.iter().map(|_| Default::default()).collect())
        }
        async fn encode_query(&self, _text: &str) -> Result<crate::sparse::SparseVector> {
            match &self.0 {
                Ok(()) => Ok(Default::default()),
                Err(error) => Err(MinSyncError::Embedding(error.to_string())),
            }
        }
    }

    #[tokio::test]
    async fn test_check_probes_the_sparse_encoder() {
        let (dir, sync, _chunker, embedder, store) = fixture();
        sync.init(false, "openai:text-embedding-3-small", "recursive")
            .expect("init succeeds");
        let minsync_dir = dir.path().join(".minsync");

        let healthy = ProbeSparse(Ok(()));
        let ok = check(&minsync_dir, &embedder, Some(&healthy), &store)
            .await
            .expect("check succeeds");
        assert!(ok.sparse_ok && ok.all_passed, "{:?}", ok.errors);

        let broken = ProbeSparse(Err(MinSyncError::Embedding("model gone".into())));
        let bad = check(&minsync_dir, &embedder, Some(&broken), &store)
            .await
            .expect("check reports instead of failing");
        assert!(!bad.sparse_ok);
        assert!(bad.embedder_ok, "the dense probe is independent");
        assert!(!bad.all_passed);
        assert!(
            bad.errors
                .iter()
                .any(|e| e.contains("sparse encoder") && e.contains("model gone")),
            "{:?}",
            bad.errors
        );
    }

    struct SparseOnly;

    #[async_trait::async_trait]
    impl Embedder for SparseOnly {
        fn id(&self) -> &str {
            "none"
        }

        fn is_sparse_only(&self) -> bool {
            true
        }

        async fn embed(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>> {
            panic!("check must not probe a sparse-only embedder");
        }
    }

    #[tokio::test]
    async fn test_check_sparse_only_skips_embedder_probe() {
        let (dir, sync, _chunker, _embedder, store) = fixture();
        sync.init(false, "none", "recursive")
            .expect("init succeeds");

        let result = check(&dir.path().join(".minsync"), &SparseOnly, None, &store)
            .await
            .expect("check succeeds");

        assert!(result.embedder_ok);
        assert!(result.all_passed);
    }

    #[tokio::test]
    async fn test_verify_sparse_only_placeholder_vectors_are_not_flagged() {
        let (dir, sync, chunker, _embedder, mut store) = fixture();
        std::fs::write(
            dir.path().join("a.txt"),
            "alpha beta gamma delta epsilon zeta",
        )
        .expect("write a");
        std::fs::write(
            dir.path().join("b.txt"),
            "completely different words about oceans",
        )
        .expect("write b");
        sync.init(false, "none", "recursive")
            .expect("init succeeds");
        sync.sync(&chunker, &SparseOnly, &mut store, true, false, false)
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

        assert!(result.embedding_integrity.flagged_pairs.is_empty());
        assert!(result.all_passed, "{:?}", result.basic_checks);
    }

    #[tokio::test]
    async fn test_verify_sparse_only_skips_the_embedding_integrity_scan() {
        let (dir, sync, chunker, _embedder, mut store) = fixture();
        std::fs::write(dir.path().join("a.txt"), "alpha beta gamma").expect("write a");
        sync.init(false, "none", "recursive")
            .expect("init succeeds");
        sync.sync(&chunker, &SparseOnly, &mut store, true, false, false)
            .await
            .expect("sync succeeds");
        let state = crate::types::IndexState {
            ann_indexed: false,
            unindexed_rows: None,
        };
        let mut tracking_store = StateTrackingStore {
            inner: store,
            before_repair: state.clone(),
            after_repair: state,
            repaired: false,
            full_scans: Default::default(),
        };

        let result = verify(
            &dir.path().join(".minsync"),
            dir.path(),
            &chunker,
            &mut tracking_store,
            false,
            None,
        )
        .await
        .expect("verify succeeds");

        assert!(result.all_passed, "{:?}", result.basic_checks);
        assert_eq!(
            tracking_store
                .full_scans
                .load(std::sync::atomic::Ordering::SeqCst),
            0,
            "placeholder vectors can not be compared, so verify must not read every row"
        );
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

    #[tokio::test]
    async fn test_verify_fix_reports_post_repair_index_state() {
        let (dir, sync, chunker, embedder, mut store) = fixture();
        std::fs::write(dir.path().join("a.txt"), "alpha beta gamma").expect("write file");
        sync.init(false, "openai:text-embedding-3-small", "recursive")
            .expect("init succeeds");
        sync.sync(&chunker, &embedder, &mut store, true, false, false)
            .await
            .expect("sync succeeds");
        let config = Config::load(&dir.path().join(".minsync/config.toml")).expect("load config");
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

        let before_repair = crate::types::IndexState {
            ann_indexed: true,
            unindexed_rows: Some(1),
        };
        let after_repair = crate::types::IndexState {
            ann_indexed: true,
            unindexed_rows: Some(0),
        };
        let mut tracking_store = StateTrackingStore {
            inner: store,
            before_repair: before_repair.clone(),
            after_repair: after_repair.clone(),
            repaired: false,
            full_scans: Default::default(),
        };

        let result = verify(
            &dir.path().join(".minsync"),
            dir.path(),
            &chunker,
            &mut tracking_store,
            true,
            None,
        )
        .await
        .expect("verify succeeds");

        assert!(result.fixed);
        assert_eq!(result.index_state, Some(after_repair));
    }

    /// Deterministic xorshift values in [-1, 1).
    struct Noise(u64);

    impl Noise {
        fn next(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        }

        fn vector(&mut self, dim: usize) -> Vec<f32> {
            (0..dim).map(|_| self.next()).collect()
        }
    }

    fn unit(mut v: Vec<f32>) -> Vec<f32> {
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter_mut().for_each(|x| *x /= norm);
        v
    }

    /// A unit vector at cosine `cos` from the unit vector `base`.
    fn twin(base: &[f32], cos: f32, noise: &mut Noise) -> Vec<f32> {
        let raw = noise.vector(base.len());
        let along: f32 = raw.iter().zip(base).map(|(r, b)| r * b).sum();
        let ortho = unit(raw.iter().zip(base).map(|(r, b)| r - along * b).collect());
        let sin = (1.0 - cos * cos).sqrt();
        base.iter()
            .zip(&ortho)
            .map(|(b, o)| cos * b + sin * o)
            .collect()
    }

    fn chunk(id: &str, path: &str, text: &str, embedding: Vec<f32>) -> Document {
        Document {
            id: id.to_string(),
            embedding,
            text: text.to_string(),
            source_id: "source".to_string(),
            path: path.to_string(),
            chunk_schema_id: "schema".to_string(),
            chunk_type: "text".to_string(),
            heading_path: String::new(),
            content_hash: content_hash(text),
            seen_token: "token".to_string(),
        }
    }

    /// Every pair, compared the way verify did before #152.
    fn all_pairs_oracle(store: &dyn VectorStore) -> Vec<EmbeddingIntegrityPair> {
        let documents = store.documents().expect("documents");
        let mut pairs = Vec::new();
        for (index, left) in documents.iter().enumerate() {
            for right in &documents[index + 1..] {
                if cosine_similarity(&left.embedding, &right.embedding) >= 0.999
                    && texts_differ_substantially(&left.text, &right.text)
                {
                    pairs.push(EmbeddingIntegrityPair {
                        chunk_ids: vec![left.id.clone(), right.id.clone()],
                        paths: vec![left.path.clone(), right.path.clone()],
                    });
                }
            }
        }
        pairs
    }

    #[test]
    fn test_embedding_integrity_matches_all_pairs_oracle() {
        let dim = 48;
        let mut noise = Noise(0x152);
        let mut store = InMemoryStore::new();
        let mut docs = Vec::new();
        for i in 0..600 {
            let base = unit(noise.vector(dim));
            let text = format!("chunk {i} alpha{i} beta{i} gamma{i}");
            docs.push(chunk(
                &format!("c{i:04}"),
                &format!("f{i}.txt"),
                &text,
                base.clone(),
            ));
            match i % 6 {
                // Above the threshold with unrelated text: flagged.
                0 => docs.push(chunk(
                    &format!("c{i:04}-hi"),
                    &format!("g{i}.txt"),
                    &format!("unrelated words for {i} delta{i}"),
                    twin(&base, 0.9993, &mut noise),
                )),
                // Below the threshold: not flagged.
                1 => docs.push(chunk(
                    &format!("c{i:04}-lo"),
                    &format!("g{i}.txt"),
                    &format!("unrelated words for {i} delta{i}"),
                    twin(&base, 0.998, &mut noise),
                )),
                // Same vector, same words in another case: not flagged.
                2 => docs.push(chunk(
                    &format!("c{i:04}-case"),
                    &format!("g{i}.txt"),
                    &text.to_uppercase(),
                    base.clone(),
                )),
                // The same file twice plus a flagged twin: every copy pairs.
                3 => {
                    docs.push(chunk(
                        &format!("c{i:04}-copy"),
                        &format!("copy{i}.txt"),
                        &text,
                        base.clone(),
                    ));
                    docs.push(chunk(
                        &format!("c{i:04}-hi"),
                        &format!("g{i}.txt"),
                        &format!("unrelated words for {i} delta{i}"),
                        base.iter().map(|x| x * 3.0).collect(),
                    ));
                }
                _ => {}
            }
        }
        store.upsert(&docs).expect("upsert");

        let expected = all_pairs_oracle(&store);
        let result = check_embedding_integrity(&store).expect("integrity check");

        assert_eq!(expected.len(), 100 + 2 * 100);
        assert_eq!(result.flagged_pairs, expected);
        assert!(!result.truncated);
    }

    #[test]
    fn test_embedding_integrity_stops_at_the_report_cap() {
        // An embedder that returns one vector for every text puts all
        // n(n-1)/2 pairs over the threshold.
        let rows = 200;
        let mut store = InMemoryStore::new();
        let docs: Vec<_> = (0..rows)
            .map(|i| {
                chunk(
                    &format!("c{i:04}"),
                    &format!("f{i}.txt"),
                    &format!("text number {i} word{i}"),
                    vec![0.5; 16],
                )
            })
            .collect();
        store.upsert(&docs).expect("upsert");

        let result = check_embedding_integrity(&store).expect("integrity check");

        assert!(rows * (rows - 1) / 2 > MAX_FLAGGED_PAIRS);
        assert!(result.truncated);
        assert_eq!(result.flagged_pairs.len(), MAX_FLAGGED_PAIRS);
        let oracle = all_pairs_oracle(&store);
        assert!(result
            .flagged_pairs
            .iter()
            .all(|pair| oracle.contains(pair)));
    }
}
