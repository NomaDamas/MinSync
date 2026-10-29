//! LanceDB-backed [`VectorStore`].
//!
//! Module layout:
//! - [`schema`]: Arrow schema and Document/QueryHit <-> record-batch conversion
//! - [`filters`]: [`Filter`] -> LanceDB SQL translation and escaping
//! - [`index`]: ANN indexing config and maintenance policy
//! - [`worker`]: command channel and the worker thread's event loop
//! - [`inner`]: async LanceDB table operations run on the worker
//! - this file: the synchronous public store facade

mod filters;
mod index;
mod inner;
mod schema;
mod worker;

pub use filters::filter_to_sql;
pub use index::IndexingConfig;

use crate::error::{MinSyncError, Result};
use crate::sparse::SparseVector;
use crate::types::IndexState;
use crate::vectorstore::{Document, DocumentUpdate, Filter, QueryHit, SparseTarget, VectorStore};
use lancedb::DistanceType;
use std::path::Path;
use std::sync::mpsc;
use std::thread::{self, JoinHandle};
use worker::{run_worker, Command, Resp};

const DEFAULT_DIMENSION: usize = 1536;
const TABLE_NAME: &str = "documents";
const VECTOR_COLUMN: &str = "vector";
const DISTANCE_COLUMN: &str = "_distance";
/// Which model produced the row's sparse vector (null: not encoded yet).
const SPARSE_MODEL_COLUMN: &str = "sparse_model";
const SPARSE_INDICES_COLUMN: &str = "sparse_indices";
const SPARSE_VALUES_COLUMN: &str = "sparse_values";

/// The distance metric used for BOTH index construction and query time. These
/// must match: an index trained with one metric returns invalid results when
/// searched with another. Cosine matches the in-memory store's scoring.
const DISTANCE_TYPE: DistanceType = DistanceType::Cosine;

pub struct LanceDbStore {
    tx: mpsc::Sender<Command>,
    worker: Option<JoinHandle<()>>,
    dim: usize,
}

impl LanceDbStore {
    pub fn open_or_create(path: &Path, dimension: usize) -> Result<Self> {
        Self::open_with_indexing(path, dimension, IndexingConfig::default())
    }

    pub fn open_with_indexing(
        path: &Path,
        dimension: usize,
        indexing: IndexingConfig,
    ) -> Result<Self> {
        Self::open(path, dimension, indexing, false)
    }

    /// Open the store for read-only queries. Never creates the table or
    /// applies schema migrations, so it is safe to run concurrently with a
    /// sync holding the writer lock: Lance versions are immutable snapshots,
    /// and a query pins the latest version at open time.
    pub fn open_readonly(path: &Path, dimension: usize, indexing: IndexingConfig) -> Result<Self> {
        Self::open(path, dimension, indexing, true)
    }

    fn open(
        path: &Path,
        dimension: usize,
        indexing: IndexingConfig,
        read_only: bool,
    ) -> Result<Self> {
        let dim = if dimension == 0 {
            DEFAULT_DIMENSION
        } else {
            dimension
        };
        schema::validate_dimension(dim)?;
        let uri = path
            .to_str()
            .ok_or_else(|| MinSyncError::VectorStore("LanceDB path is not valid UTF-8".into()))?
            .to_string();

        let (cmd_tx, cmd_rx) = mpsc::channel();
        let (init_tx, init_rx) = mpsc::channel();
        let worker =
            thread::spawn(move || run_worker(uri, dim, indexing, read_only, cmd_rx, init_tx));

        match init_rx.recv().map_err(to_store_error)? {
            Ok(()) => Ok(Self {
                tx: cmd_tx,
                worker: Some(worker),
                dim,
            }),
            Err(error) => {
                let _ = worker.join();
                Err(error)
            }
        }
    }

    pub fn open_with_default_dimension(path: &Path) -> Result<Self> {
        Self::open_or_create(path, DEFAULT_DIMENSION)
    }

    pub fn dimension_from_options(options: Option<&toml::Value>) -> Result<usize> {
        let Some(options) = options else {
            return Ok(DEFAULT_DIMENSION);
        };
        let Some(value) = options.get("dimension") else {
            return Ok(DEFAULT_DIMENSION);
        };
        let Some(dimension) = value.as_integer() else {
            return Err(MinSyncError::Config(
                "vectorstore.options.dimension must be an integer".into(),
            ));
        };
        let dimension = usize::try_from(dimension).map_err(|_| {
            MinSyncError::Config("vectorstore.options.dimension must be positive".into())
        })?;
        if dimension == 0 {
            return Err(MinSyncError::Config(
                "vectorstore.options.dimension must be positive".into(),
            ));
        }
        schema::validate_dimension(dimension)
            .map_err(|error| MinSyncError::Config(error.to_string()))?;
        Ok(dimension)
    }

    pub fn dimension(&self) -> usize {
        self.dim
    }

    pub fn index_names(&self) -> Result<Vec<String>> {
        self.request(Command::IndexNames)
    }

    pub fn index_state(&self) -> Result<IndexState> {
        self.request(Command::IndexState)
    }

    /// Number of data fragments in the table.
    pub fn fragment_count(&self) -> Result<usize> {
        self.request(Command::FragmentCount)
    }

    fn request<T>(&self, make: impl FnOnce(Resp<T>) -> Command) -> Result<T> {
        let (resp_tx, resp_rx) = mpsc::channel();
        self.tx.send(make(resp_tx)).map_err(to_store_error)?;
        resp_rx.recv().map_err(to_store_error)?
    }
}

impl Drop for LanceDbStore {
    fn drop(&mut self) {
        let _ = self.tx.send(Command::Shutdown);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl VectorStore for LanceDbStore {
    fn upsert(&mut self, docs: &[Document]) -> Result<()> {
        self.request(|resp| Command::Upsert(docs.to_vec(), resp))
    }

    fn update(&mut self, updates: &[DocumentUpdate]) -> Result<()> {
        self.request(|resp| Command::Update(updates.to_vec(), resp))
    }

    fn insert_new(&mut self, docs: &[Document]) -> Result<()> {
        self.request(|resp| Command::InsertNew(docs.to_vec(), resp))
    }

    fn existing_ids(&self, ids: &[String]) -> Result<std::collections::HashSet<String>> {
        self.request(|resp| Command::ExistingIds(ids.to_vec(), resp))
    }

    fn fetch(&self, ids: &[String]) -> Result<Vec<Document>> {
        self.request(|resp| Command::Fetch(ids.to_vec(), resp))
    }

    fn fetch_embeddings_by_content_hash(
        &self,
        content_hashes: &[String],
    ) -> Result<std::collections::HashMap<String, Vec<f32>>> {
        self.request(|resp| Command::FetchEmbeddingsByContentHash(content_hashes.to_vec(), resp))
    }

    fn documents(&self) -> Result<Vec<Document>> {
        self.request(Command::Documents)
    }

    fn delete_by_filter(&mut self, filter: &Filter) -> Result<usize> {
        self.request(|resp| Command::Delete(filter.clone(), resp))
    }

    fn query(&self, vector: &[f32], filter: Option<&Filter>, topk: usize) -> Result<Vec<QueryHit>> {
        self.request(|resp| Command::Query {
            vector: vector.to_vec(),
            filter: filter.cloned(),
            topk,
            resp,
        })
    }

    fn rows_missing_sparse(&self, model: &str, limit: usize) -> Result<Vec<SparseTarget>> {
        self.request(|resp| Command::RowsMissingSparse {
            model: model.to_string(),
            limit,
            resp,
        })
    }

    fn count_missing_sparse(&self, model: &str) -> Result<usize> {
        self.request(|resp| Command::CountMissingSparse(model.to_string(), resp))
    }

    fn set_sparse(&mut self, model: &str, rows: &[(String, SparseVector)]) -> Result<()> {
        self.request(|resp| Command::SetSparse {
            model: model.to_string(),
            rows: rows.to_vec(),
            resp,
        })
    }

    fn fetch_sparse_by_content_hash(
        &self,
        model: &str,
        content_hashes: &[String],
    ) -> Result<std::collections::HashMap<String, SparseVector>> {
        self.request(|resp| Command::FetchSparseByContentHash {
            model: model.to_string(),
            hashes: content_hashes.to_vec(),
            resp,
        })
    }

    fn query_sparse(
        &self,
        model: &str,
        vector: &SparseVector,
        filter: Option<&Filter>,
        topk: usize,
    ) -> Result<Vec<QueryHit>> {
        self.request(|resp| Command::QuerySparse {
            model: model.to_string(),
            vector: vector.clone(),
            filter: filter.cloned(),
            topk,
            resp,
        })
    }

    fn flush(&mut self) -> Result<()> {
        self.request(Command::Flush)
    }

    fn checkpoint(&mut self) -> Result<()> {
        self.request(Command::Checkpoint)
    }

    fn doc_count(&self) -> usize {
        match self.request(Command::DocCount) {
            Ok(count) => count,
            Err(error) => {
                tracing::warn!(%error, "failed to count LanceDB documents");
                0
            }
        }
    }

    fn all_paths(&self) -> Vec<String> {
        match self.request(Command::AllPaths) {
            Ok(paths) => paths,
            Err(error) => {
                tracing::warn!(%error, "failed to list LanceDB paths");
                Vec::new()
            }
        }
    }

    fn index_state(&self) -> Result<Option<IndexState>> {
        LanceDbStore::index_state(self).map(Some)
    }
}

fn to_store_error(error: impl std::fmt::Display) -> MinSyncError {
    MinSyncError::VectorStore(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn doc(id: &str, path: &str, seen_token: &str, embedding: Vec<f32>) -> Document {
        Document {
            id: id.to_string(),
            embedding,
            text: format!("text {id}"),
            source_id: "source-1".to_string(),
            path: path.to_string(),
            chunk_schema_id: "schema-1".to_string(),
            chunk_type: "text".to_string(),
            heading_path: format!("heading {id}"),
            content_hash: format!("hash-{id}"),
            seen_token: seen_token.to_string(),
        }
    }

    fn store() -> (TempDir, LanceDbStore) {
        let dir = tempfile::tempdir().expect("create tempdir");
        let store = LanceDbStore::open_or_create(dir.path(), 4).expect("create lancedb store");
        (dir, store)
    }

    #[test]
    fn test_upsert_and_fetch() {
        let (_dir, mut store) = store();
        let docs = vec![
            doc("a", "a.txt", "token-1", vec![1.0, 0.0, 0.0, 0.0]),
            doc("b", "b.txt", "token-1", vec![0.0, 1.0, 0.0, 0.0]),
            doc("c", "c.txt", "token-1", vec![0.0, 0.0, 1.0, 0.0]),
        ];
        store.upsert(&docs).expect("upsert docs");
        let fetched = store
            .fetch(&["a".to_string(), "b".to_string(), "c".to_string()])
            .expect("fetch docs");
        assert_eq!(store.doc_count(), 3);
        assert_eq!(fetched.len(), 3);
        assert_eq!(fetched[0].id, "a");
        assert_eq!(fetched[1].embedding, vec![0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_upsert_replaces() {
        let (_dir, mut store) = store();
        store
            .upsert(&[doc("a", "old.txt", "old", vec![1.0, 0.0, 0.0, 0.0])])
            .expect("upsert old");
        store
            .upsert(&[doc("a", "new.txt", "new", vec![0.0, 1.0, 0.0, 0.0])])
            .expect("upsert new");
        let fetched = store.fetch(&["a".to_string()]).expect("fetch doc");
        assert_eq!(store.doc_count(), 1);
        assert_eq!(fetched[0].path, "new.txt");
        assert_eq!(fetched[0].seen_token, "new");
        assert_eq!(fetched[0].embedding, vec![0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_flush_prunes_after_repeated_upsert_replacements() {
        let (_dir, mut store) = store();
        store
            .upsert(&[
                doc("a", "old-a.txt", "old", vec![1.0, 0.0, 0.0, 0.0]),
                doc("b", "old-b.txt", "old", vec![0.0, 1.0, 0.0, 0.0]),
            ])
            .expect("upsert initial docs");
        store.flush().expect("flush initial docs");

        for generation in 0..3 {
            store
                .upsert(&[
                    doc(
                        "a",
                        &format!("new-a-{generation}.txt"),
                        &format!("token-{generation}"),
                        vec![0.0, 0.0, 1.0, 0.0],
                    ),
                    doc(
                        "b",
                        &format!("new-b-{generation}.txt"),
                        &format!("token-{generation}"),
                        vec![0.0, 0.0, 0.0, 1.0],
                    ),
                ])
                .expect("upsert replacement docs");
            store.flush().expect("flush replacement docs");
        }

        let fetched = store
            .fetch(&["a".to_string(), "b".to_string()])
            .expect("fetch replacement docs");
        assert_eq!(store.doc_count(), 2);
        assert_eq!(fetched.len(), 2);
        assert_eq!(fetched[0].path, "new-a-2.txt");
        assert_eq!(fetched[0].seen_token, "token-2");
        assert_eq!(fetched[1].path, "new-b-2.txt");
        assert_eq!(fetched[1].seen_token, "token-2");
    }

    #[test]
    fn test_upsert_dedupes_input_batch_last_wins() {
        let (_dir, mut store) = store();
        store
            .upsert(&[
                doc("a", "old.txt", "old", vec![1.0, 0.0, 0.0, 0.0]),
                doc("a", "new.txt", "new", vec![0.0, 1.0, 0.0, 0.0]),
            ])
            .expect("upsert duplicate docs");
        let fetched = store.fetch(&["a".to_string()]).expect("fetch doc");
        assert_eq!(store.doc_count(), 1);
        assert_eq!(fetched[0].path, "new.txt");
    }

    #[test]
    fn test_update_metadata() {
        let (_dir, mut store) = store();
        store
            .upsert(&[doc("a", "old.txt", "old", vec![1.0, 0.0, 0.0, 0.0])])
            .expect("upsert doc");
        store
            .update(&[DocumentUpdate {
                id: "a".to_string(),
                seen_token: "new".to_string(),
                path: "new.txt".to_string(),
                heading_path: "new heading".to_string(),
            }])
            .expect("update doc");
        let fetched = store.fetch(&["a".to_string()]).expect("fetch doc");
        assert_eq!(fetched[0].seen_token, "new");
        assert_eq!(fetched[0].path, "new.txt");
        assert_eq!(fetched[0].heading_path, "new heading");
    }

    #[test]
    fn test_delete_by_eq_filter() {
        let (_dir, mut store) = store();
        store
            .upsert(&[
                doc("a", "x.txt", "token", vec![1.0, 0.0, 0.0, 0.0]),
                doc("b", "x.txt", "token", vec![0.0, 1.0, 0.0, 0.0]),
                doc("c", "y.txt", "token", vec![0.0, 0.0, 1.0, 0.0]),
            ])
            .expect("upsert docs");
        let deleted = store
            .delete_by_filter(&Filter::Eq("path".to_string(), "x.txt".to_string()))
            .expect("delete docs");
        assert_eq!(deleted, 2);
        assert_eq!(store.doc_count(), 1);
    }

    #[test]
    fn test_delete_by_and_filter() {
        let (_dir, mut store) = store();
        store
            .upsert(&[
                doc("a", "same.txt", "keep", vec![1.0, 0.0, 0.0, 0.0]),
                doc("b", "same.txt", "drop", vec![0.0, 1.0, 0.0, 0.0]),
                doc("c", "other.txt", "drop", vec![0.0, 0.0, 1.0, 0.0]),
            ])
            .expect("upsert docs");
        let deleted = store
            .delete_by_filter(&Filter::And(vec![
                Filter::Eq("path".to_string(), "same.txt".to_string()),
                Filter::Neq("seen_token".to_string(), "keep".to_string()),
            ]))
            .expect("delete docs");
        assert_eq!(deleted, 1);
        assert_eq!(store.doc_count(), 2);
        assert!(store.fetch(&["b".to_string()]).expect("fetch b").is_empty());
    }

    #[test]
    fn test_query_topk() {
        let (_dir, mut store) = store();
        store
            .upsert(&[
                doc("near", "path.txt", "token", vec![1.0, 0.0, 0.0, 0.0]),
                doc("mid", "path.txt", "token", vec![0.7, 0.7, 0.0, 0.0]),
                doc("far", "path.txt", "token", vec![0.0, 1.0, 0.0, 0.0]),
            ])
            .expect("upsert docs");
        let hits = store
            .query(&[1.0, 0.0, 0.0, 0.0], None, 2)
            .expect("query docs");
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].doc_id, "near");
        assert!(hits[0].score >= hits[1].score);
        assert!(hits[0].score > 0.99);
    }

    #[test]
    fn test_query_with_filter() {
        let (_dir, mut store) = store();
        store
            .upsert(&[
                doc("a", "x.txt", "token", vec![1.0, 0.0, 0.0, 0.0]),
                doc("b", "y.txt", "token", vec![1.0, 0.0, 0.0, 0.0]),
                doc("c", "x.txt", "token", vec![0.0, 1.0, 0.0, 0.0]),
            ])
            .expect("upsert docs");
        let hits = store
            .query(
                &[1.0, 0.0, 0.0, 0.0],
                Some(&Filter::Eq("path".to_string(), "x.txt".to_string())),
                10,
            )
            .expect("query docs");
        assert_eq!(hits.len(), 2);
        assert!(hits.iter().all(|hit| hit.path == "x.txt"));
    }

    #[test]
    fn test_query_topk_zero_and_non_finite_rejected() {
        let (_dir, mut store) = store();
        store
            .upsert(&[doc("a", "x.txt", "token", vec![1.0, 0.0, 0.0, 0.0])])
            .expect("upsert docs");
        assert!(store
            .query(&[f32::NAN, 0.0, 0.0, 0.0], None, 0)
            .expect("zero topk")
            .is_empty());
        assert!(store.query(&[f32::NAN, 0.0, 0.0, 0.0], None, 1).is_err());
    }

    #[test]
    fn test_upsert_rejects_non_finite_embedding() {
        let (_dir, mut store) = store();
        assert!(store
            .upsert(&[doc(
                "a",
                "x.txt",
                "token",
                vec![f32::INFINITY, 0.0, 0.0, 0.0]
            )])
            .is_err());
    }

    #[test]
    fn test_fetch_embeddings_by_content_hash_returns_only_known_hashes() {
        let (_dir, mut store) = store();
        let mut shared = doc("b", "copy.txt", "token", vec![0.0, 1.0, 0.0, 0.0]);
        shared.content_hash = "hash-a".to_string();
        store
            .upsert(&[
                doc("a", "a.txt", "token", vec![1.0, 0.0, 0.0, 0.0]),
                shared,
                doc("c", "c.txt", "token", vec![0.0, 0.0, 1.0, 0.0]),
            ])
            .expect("upsert docs");

        let found = store
            .fetch_embeddings_by_content_hash(&[
                "hash-a".to_string(),
                "hash-c".to_string(),
                "hash-missing".to_string(),
                "it's quoted".to_string(),
            ])
            .expect("lookup succeeds");

        assert_eq!(found.len(), 2);
        assert_eq!(found["hash-c"], vec![0.0, 0.0, 1.0, 0.0]);
        assert!(
            found["hash-a"] == vec![1.0, 0.0, 0.0, 0.0]
                || found["hash-a"] == vec![0.0, 1.0, 0.0, 0.0]
        );
        assert!(store
            .fetch_embeddings_by_content_hash(&[])
            .expect("empty lookup succeeds")
            .is_empty());
    }

    #[test]
    fn test_all_paths() {
        let (_dir, mut store) = store();
        store
            .upsert(&[
                doc("a", "b.txt", "token", vec![1.0, 0.0, 0.0, 0.0]),
                doc("b", "a.txt", "token", vec![0.0, 1.0, 0.0, 0.0]),
                doc("c", "c.txt", "token", vec![0.0, 0.0, 1.0, 0.0]),
                doc("d", "a.txt", "token", vec![0.0, 0.0, 0.0, 1.0]),
            ])
            .expect("upsert docs");
        assert_eq!(store.all_paths(), vec!["a.txt", "b.txt", "c.txt"]);
    }

    #[test]
    fn test_readonly_open_missing_table_errors() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let result = LanceDbStore::open_readonly(dir.path(), 4, IndexingConfig::default());
        assert!(
            matches!(result, Err(crate::error::MinSyncError::NeverSynced)),
            "read-only open on a missing table must report NeverSynced"
        );
    }

    #[test]
    fn test_readonly_query_runs_alongside_writer() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let mut writer = LanceDbStore::open_or_create(dir.path(), 4).expect("create writer store");
        writer
            .upsert(&[doc("a", "a.txt", "token", vec![1.0, 0.0, 0.0, 0.0])])
            .expect("upsert doc");
        writer.flush().expect("flush writer");

        // A second, read-only connection must open and query while the writer
        // connection stays open — this is the `minsync query` during
        // `minsync sync` scenario.
        let reader = LanceDbStore::open_readonly(dir.path(), 4, IndexingConfig::default())
            .expect("open read-only store alongside writer");
        let hits = reader
            .query(&[1.0, 0.0, 0.0, 0.0], None, 1)
            .expect("query through read-only store");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "a");

        // Writes committed after the reader opened do not break the reader's
        // pinned snapshot, even across a compact+prune flush.
        writer
            .upsert(&[doc("b", "b.txt", "token", vec![0.0, 1.0, 0.0, 0.0])])
            .expect("upsert second doc");
        writer.flush().expect("flush second write");
        let hits = reader
            .query(&[1.0, 0.0, 0.0, 0.0], None, 1)
            .expect("reader still queries after writer flush");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "a");

        // A freshly opened reader sees the new state.
        drop(reader);
        let reader = LanceDbStore::open_readonly(dir.path(), 4, IndexingConfig::default())
            .expect("reopen read-only store");
        assert_eq!(reader.doc_count(), 2);
    }

    #[test]
    fn test_persistence_roundtrip() {
        let dir = tempfile::tempdir().expect("create tempdir");
        {
            let mut store = LanceDbStore::open_or_create(dir.path(), 4).expect("create store");
            store
                .upsert(&[doc("a", "a.txt", "token", vec![1.0, 0.0, 0.0, 0.0])])
                .expect("upsert doc");
        }
        let loaded = LanceDbStore::open_or_create(dir.path(), 4).expect("reopen store");
        let fetched = loaded.fetch(&["a".to_string()]).expect("fetch loaded doc");
        assert_eq!(fetched.len(), 1);
        assert_eq!(fetched[0].path, "a.txt");
        assert_eq!(fetched[0].embedding, vec![1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_open_existing_dimension_mismatch_errors() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let store = LanceDbStore::open_or_create(dir.path(), 4).expect("create store");
        drop(store);
        assert!(LanceDbStore::open_or_create(dir.path(), 3).is_err());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_store_works_inside_multithread_tokio_runtime() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let mut store = LanceDbStore::open_or_create(dir.path(), 4).expect("create store");
        store
            .upsert(&[doc("a", "a.txt", "token", vec![1.0, 0.0, 0.0, 0.0])])
            .expect("upsert doc");
        let hits = store
            .query(&[1.0, 0.0, 0.0, 0.0], None, 1)
            .expect("query doc");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "a");
    }

    fn indexed_store(dir: &TempDir, index_build_threshold: usize) -> LanceDbStore {
        LanceDbStore::open_with_indexing(
            dir.path(),
            4,
            IndexingConfig {
                index_build_threshold,
                index_optimize_delta_threshold: 1,
                ..IndexingConfig::default()
            },
        )
        .expect("create lancedb store")
    }

    fn spread_doc(i: usize) -> Document {
        let angle = i as f32 * 0.013;
        doc(
            &format!("doc-{i}"),
            "corpus.txt",
            "token",
            vec![angle.cos(), angle.sin(), 0.0, 0.0],
        )
    }

    #[test]
    fn test_ann_index_skipped_when_disabled_even_above_threshold() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let mut store = LanceDbStore::open_with_indexing(
            dir.path(),
            4,
            IndexingConfig {
                index_build_threshold: 10,
                build_ann_index: false,
                ..IndexingConfig::default()
            },
        )
        .expect("create store");
        let docs: Vec<_> = (0..50).map(spread_doc).collect();
        store.upsert(&docs).expect("upsert docs");
        store.flush().expect("flush");
        let indices = store.index_names().expect("list indices");
        assert!(
            !indices.iter().any(|name| name.contains("vector")),
            "disabled ANN must not be built, got {indices:?}"
        );
    }

    #[test]
    fn test_no_index_built_below_threshold() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let mut store = indexed_store(&dir, 1000);
        let docs: Vec<_> = (0..50).map(spread_doc).collect();
        store.upsert(&docs).expect("upsert docs");
        store.flush().expect("flush");
        let indices = store.index_names().expect("list indices");
        assert!(
            !indices.iter().any(|name| name.contains("text")),
            "no full-text index exists any more, got {indices:?}"
        );
        assert!(
            !indices.iter().any(|name| name.contains("vector")),
            "no ANN vector index should exist below the row threshold, got {indices:?}"
        );
    }

    #[test]
    fn test_index_built_above_threshold_and_query_still_accurate() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let mut store = indexed_store(&dir, 256);
        let docs: Vec<_> = (0..300).map(spread_doc).collect();
        store.upsert(&docs).expect("upsert docs");
        store.flush().expect("flush builds index");

        assert!(
            !store.index_names().expect("list indices").is_empty(),
            "ANN index should be built once the threshold is crossed"
        );

        // The nearest vector to doc-0's embedding is doc-0 itself. A Cosine-built
        // index queried with Cosine must return it first; an L2/Cosine mismatch
        // would scramble this ranking.
        let target = spread_doc(0).embedding;
        let hits = store.query(&target, None, 1).expect("query");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "doc-0");
        assert!(hits[0].score > 0.99);
    }

    #[test]
    fn test_rows_added_after_index_remain_searchable() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let mut store = indexed_store(&dir, 256);
        let docs: Vec<_> = (0..300).map(spread_doc).collect();
        store.upsert(&docs).expect("upsert initial docs");
        store.flush().expect("flush builds index");

        store
            .upsert(&[doc("late", "corpus.txt", "token", vec![0.0, 0.0, 1.0, 0.0])])
            .expect("upsert late doc");
        store.flush().expect("flush folds delta");

        let hits = store
            .query(&[0.0, 0.0, 1.0, 0.0], None, 1)
            .expect("query late doc");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "late");
    }

    fn version_count(dir: &TempDir) -> usize {
        std::fs::read_dir(dir.path().join("documents.lance").join("_versions"))
            .expect("read _versions")
            .count()
    }

    #[test]
    fn test_update_batch_commits_once_and_applies_per_id_values() {
        let (dir, mut store) = store();
        let docs: Vec<_> = (0..40)
            .map(|i| doc(&format!("id-{i}"), "a.txt", "old", vec![1.0, 0.0, 0.0, 0.0]))
            .collect();
        store.upsert(&docs).expect("upsert docs");

        let before = version_count(&dir);
        let updates: Vec<_> = (0..40)
            .map(|i| DocumentUpdate {
                id: format!("id-{i}"),
                seen_token: "new".to_string(),
                path: "a.txt".to_string(),
                heading_path: format!("heading {i}"),
            })
            .collect();
        store.update(&updates).expect("batch update");

        assert!(
            version_count(&dir) - before <= 1,
            "a batch update must be a single commit"
        );
        let ids: Vec<_> = (0..40).map(|i| format!("id-{i}")).collect();
        let fetched = store.fetch(&ids).expect("fetch docs");
        assert_eq!(fetched.len(), 40);
        for (i, doc) in fetched.iter().enumerate() {
            assert_eq!(doc.seen_token, "new");
            assert_eq!(doc.heading_path, format!("heading {i}"));
            assert_eq!(doc.text, format!("text id-{i}"), "other columns untouched");
            assert_eq!(doc.embedding, vec![1.0, 0.0, 0.0, 0.0]);
        }
    }

    #[test]
    fn test_update_never_inserts_unknown_ids() {
        let (_dir, mut store) = store();
        store
            .upsert(&[doc("a", "a.txt", "old", vec![1.0, 0.0, 0.0, 0.0])])
            .expect("upsert doc");
        store
            .update(&[
                DocumentUpdate {
                    id: "a".to_string(),
                    seen_token: "new".to_string(),
                    path: "a.txt".to_string(),
                    heading_path: "h".to_string(),
                },
                DocumentUpdate {
                    id: "ghost".to_string(),
                    seen_token: "new".to_string(),
                    path: "ghost.txt".to_string(),
                    heading_path: "h".to_string(),
                },
            ])
            .expect("update with unknown id");
        assert_eq!(store.doc_count(), 1);
        assert!(store
            .fetch(&["ghost".to_string()])
            .expect("fetch ghost")
            .is_empty());
    }

    #[test]
    fn test_upsert_replacement_commits_once() {
        let (dir, mut store) = store();
        store
            .upsert(&[
                doc("a", "a.txt", "old", vec![1.0, 0.0, 0.0, 0.0]),
                doc("b", "a.txt", "old", vec![0.0, 1.0, 0.0, 0.0]),
            ])
            .expect("seed docs");
        let before = version_count(&dir);
        store
            .upsert(&[
                doc("b", "a.txt", "new", vec![0.0, 0.0, 1.0, 0.0]),
                doc("c", "a.txt", "new", vec![0.0, 0.0, 0.0, 1.0]),
            ])
            .expect("upsert replace + insert");
        assert!(
            version_count(&dir) - before <= 1,
            "replace+insert must be a single commit"
        );
        assert_eq!(store.doc_count(), 3);
        let fetched = store
            .fetch(&["b".to_string(), "c".to_string()])
            .expect("fetch");
        assert_eq!(fetched[0].seen_token, "new");
        assert_eq!(fetched[0].embedding, vec![0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_delete_matching_nothing_commits_nothing() {
        let (dir, mut store) = store();
        store
            .upsert(&[doc("a", "a.txt", "t", vec![1.0, 0.0, 0.0, 0.0])])
            .expect("seed");
        let before = version_count(&dir);
        let deleted = store
            .delete_by_filter(&Filter::Eq("path".to_string(), "nope.txt".to_string()))
            .expect("delete nothing");
        assert_eq!(deleted, 0);
        assert_eq!(version_count(&dir), before);
    }

    #[test]
    fn test_flush_builds_scalar_indexes_for_id_and_path() {
        let (_dir, mut store) = store();
        store
            .upsert(&[doc("a", "a.txt", "t", vec![1.0, 0.0, 0.0, 0.0])])
            .expect("seed");
        store.flush().expect("flush");
        let names = store.index_names().expect("index names");
        assert!(names.iter().any(|n| n.starts_with("id_")), "{names:?}");
        assert!(names.iter().any(|n| n.starts_with("path_")), "{names:?}");
        // Idempotent: a second flush must not fail or duplicate them.
        store.flush().expect("second flush");
        assert_eq!(store.index_names().expect("index names").len(), names.len());
    }

    #[test]
    fn test_checkpoint_compacts_fragments_above_threshold_and_keeps_data() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let mut store = LanceDbStore::open_with_indexing(
            dir.path(),
            4,
            IndexingConfig {
                compact_fragment_threshold: 8,
                ..IndexingConfig::default()
            },
        )
        .expect("create store");
        for i in 0..20 {
            store
                .upsert(&[doc(
                    &format!("id-{i}"),
                    "a.txt",
                    "t",
                    vec![1.0, 0.0, 0.0, 0.0],
                )])
                .expect("upsert");
        }
        assert!(store.fragment_count().expect("fragments") >= 20);
        store.checkpoint().expect("checkpoint");
        assert!(
            store.fragment_count().expect("fragments") < 8,
            "checkpoint above threshold must compact"
        );
        assert_eq!(store.doc_count(), 20);
        let fetched = store.fetch(&["id-7".to_string()]).expect("fetch after");
        assert_eq!(fetched.len(), 1, "scalar-indexed lookup still finds rows");
    }

    #[test]
    fn test_checkpoint_below_threshold_does_not_rewrite_data() {
        let (_dir, mut store) = store();
        for i in 0..3 {
            store
                .upsert(&[doc(
                    &format!("id-{i}"),
                    "a.txt",
                    "t",
                    vec![1.0, 0.0, 0.0, 0.0],
                )])
                .expect("upsert");
        }
        let before = store.fragment_count().expect("fragments");
        store.checkpoint().expect("checkpoint");
        assert_eq!(store.fragment_count().expect("fragments"), before);
        assert_eq!(store.doc_count(), 3);
    }

    #[test]
    fn test_existing_ids_returns_only_present_ids() {
        let (_dir, mut store) = store();
        store
            .upsert(&[
                doc("a", "a.txt", "t", vec![1.0, 0.0, 0.0, 0.0]),
                doc("b", "a.txt", "t", vec![0.0, 1.0, 0.0, 0.0]),
            ])
            .expect("seed");
        let found = store
            .existing_ids(&["a".to_string(), "ghost".to_string(), "b".to_string()])
            .expect("existing ids");
        assert_eq!(found.len(), 2);
        assert!(found.contains("a") && found.contains("b"));
        assert!(store.existing_ids(&[]).expect("empty").is_empty());
    }

    #[test]
    fn test_existing_ids_is_not_truncated_for_large_id_lists() {
        // A silently truncated existence check would make `insert_new`
        // append duplicates of chunks that are already stored.
        let (_dir, mut store) = store();
        let docs: Vec<_> = (0..1500)
            .map(|i| doc(&format!("id-{i}"), "a.txt", "t", vec![1.0, 0.0, 0.0, 0.0]))
            .collect();
        store.upsert(&docs).expect("seed");
        let mut ids: Vec<String> = (0..1500).map(|i| format!("id-{i}")).collect();
        ids.push("ghost".to_string());
        let found = store.existing_ids(&ids).expect("existing ids");
        assert_eq!(found.len(), 1500);
        assert!(!found.contains("ghost"));
    }

    #[test]
    fn test_insert_new_appends_in_one_commit_without_scanning() {
        let (dir, mut store) = store();
        store
            .upsert(&[doc("seed", "a.txt", "t", vec![1.0, 0.0, 0.0, 0.0])])
            .expect("seed");
        let before = version_count(&dir);
        let docs: Vec<_> = (0..25)
            .map(|i| doc(&format!("n-{i}"), "b.txt", "t", vec![0.0, 1.0, 0.0, 0.0]))
            .collect();
        store.insert_new(&docs).expect("insert new");
        assert_eq!(version_count(&dir) - before, 1);
        assert_eq!(store.doc_count(), 26);
        let found = store.fetch(&["n-24".to_string()]).expect("fetch inserted");
        assert_eq!(found[0].path, "b.txt");
    }

    #[test]
    fn test_insert_new_rejects_non_finite_embedding() {
        let (_dir, mut store) = store();
        let bad = doc("a", "a.txt", "t", vec![f32::NAN, 0.0, 0.0, 0.0]);
        assert!(store.insert_new(&[bad]).is_err());
        assert_eq!(store.doc_count(), 0);
    }

    #[test]
    fn test_delete_by_in_filter() {
        let (_dir, mut store) = store();
        store
            .upsert(&[
                doc("a", "x.txt", "t", vec![1.0, 0.0, 0.0, 0.0]),
                doc("b", "y.txt", "t", vec![0.0, 1.0, 0.0, 0.0]),
                doc("c", "z.txt", "t", vec![0.0, 0.0, 1.0, 0.0]),
            ])
            .expect("seed");
        let deleted = store
            .delete_by_filter(&Filter::In(
                "path".to_string(),
                vec!["x.txt".to_string(), "z.txt".to_string()],
            ))
            .expect("delete in");
        assert_eq!(deleted, 2);
        assert_eq!(store.all_paths(), vec!["y.txt"]);
    }

    fn sv(pairs: &[(u32, f32)]) -> SparseVector {
        SparseVector::from_pairs(pairs.iter().copied())
    }

    #[test]
    fn test_fresh_rows_are_missing_sparse_until_set() {
        let (_dir, mut store) = store();
        store
            .upsert(&[
                doc("a", "a.txt", "t", vec![1.0, 0.0, 0.0, 0.0]),
                doc("b", "b.txt", "t", vec![0.0, 1.0, 0.0, 0.0]),
            ])
            .expect("seed");
        // Null columns, not just a different model: the SQL predicate must
        // keep NULL rows in the "missing" set.
        assert_eq!(store.count_missing_sparse("m").expect("count"), 2);
        let rows = store.rows_missing_sparse("m", 10).expect("rows");
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|row| row.text.starts_with("text ")));

        store
            .set_sparse("m", &[("a".to_string(), sv(&[(3, 1.5), (9, 0.25)]))])
            .expect("set sparse");

        assert_eq!(store.count_missing_sparse("m").expect("count"), 1);
        let missing = store.rows_missing_sparse("m", 10).expect("rows");
        assert_eq!(missing.len(), 1);
        assert_eq!(missing[0].id, "b");
        // Another model has encoded nothing.
        assert_eq!(store.count_missing_sparse("other").expect("count"), 2);
    }

    #[test]
    fn test_rows_missing_sparse_honours_the_limit() {
        let (_dir, mut store) = store();
        let docs: Vec<_> = (0..7)
            .map(|i| doc(&format!("d{i}"), "x.txt", "t", vec![1.0, 0.0, 0.0, 0.0]))
            .collect();
        store.upsert(&docs).expect("seed");
        assert_eq!(store.rows_missing_sparse("m", 3).expect("rows").len(), 3);
        assert!(store.rows_missing_sparse("m", 0).expect("rows").is_empty());
        assert_eq!(store.count_missing_sparse("m").expect("count"), 7);
    }

    #[test]
    fn test_query_sparse_is_exact_inner_product_with_filter() {
        let (_dir, mut store) = store();
        store
            .upsert(&[
                doc("a", "a.txt", "t", vec![1.0, 0.0, 0.0, 0.0]),
                doc("b", "b.txt", "t", vec![1.0, 0.0, 0.0, 0.0]),
                doc("c", "c.txt", "t", vec![1.0, 0.0, 0.0, 0.0]),
                doc("d", "d.txt", "t", vec![1.0, 0.0, 0.0, 0.0]),
            ])
            .expect("seed");
        store
            .set_sparse(
                "m",
                &[
                    ("a".to_string(), sv(&[(1, 1.0), (2, 1.0)])),
                    ("b".to_string(), sv(&[(1, 3.0)])),
                    ("c".to_string(), sv(&[(9, 5.0)])),
                    ("d".to_string(), SparseVector::default()),
                ],
            )
            .expect("set sparse");

        let hits = store
            .query_sparse("m", &sv(&[(1, 2.0), (2, 0.5)]), None, 10)
            .expect("query");
        let ranked: Vec<(&str, f32)> = hits.iter().map(|h| (h.doc_id.as_str(), h.score)).collect();
        assert_eq!(ranked, vec![("b", 6.0), ("a", 2.5)]);
        assert_eq!(hits[0].path, "b.txt");
        assert_eq!(hits[0].text, "text b");
        assert_eq!(hits[0].content_hash, "hash-b");

        let only_a = store
            .query_sparse(
                "m",
                &sv(&[(1, 2.0)]),
                Some(&Filter::Eq("path".to_string(), "a.txt".to_string())),
                10,
            )
            .expect("filtered");
        assert_eq!(only_a.len(), 1);
        assert_eq!(only_a[0].doc_id, "a");

        assert_eq!(
            store
                .query_sparse("m", &sv(&[(1, 2.0)]), None, 1)
                .expect("topk")
                .len(),
            1
        );
        assert!(store
            .query_sparse("m", &sv(&[(1, 2.0)]), None, 0)
            .expect("k0")
            .is_empty());
        assert!(store
            .query_sparse("m", &SparseVector::default(), None, 5)
            .expect("empty")
            .is_empty());
        assert!(store
            .query_sparse("other-model", &sv(&[(1, 2.0)]), None, 5)
            .expect("other model")
            .is_empty());
    }

    #[test]
    fn test_query_sparse_scores_rows_across_many_batches() {
        // Enough rows that the scan streams several record batches.
        let (_dir, mut store) = store();
        let docs: Vec<_> = (0..3000)
            .map(|i| doc(&format!("d{i:05}"), "x.txt", "t", vec![1.0, 0.0, 0.0, 0.0]))
            .collect();
        store.insert_new(&docs).expect("seed");
        let rows: Vec<(String, SparseVector)> = (0..3000)
            .map(|i| (format!("d{i:05}"), sv(&[(1, 1.0 + (i % 100) as f32)])))
            .collect();
        for chunk in rows.chunks(1000) {
            store.set_sparse("m", chunk).expect("set sparse");
        }
        assert_eq!(store.count_missing_sparse("m").expect("count"), 0);

        let hits = store
            .query_sparse("m", &sv(&[(1, 1.0)]), None, 5)
            .expect("query");
        let ids: Vec<&str> = hits.iter().map(|h| h.doc_id.as_str()).collect();
        // Score 100 is shared by d00099, d00199, ...; ties go to the smaller id.
        assert_eq!(ids, vec!["d00099", "d00199", "d00299", "d00399", "d00499"]);
        assert!(hits.iter().all(|h| h.score == 100.0));
    }

    #[test]
    fn test_set_sparse_ignores_unknown_ids_and_last_write_wins() {
        let (_dir, mut store) = store();
        store
            .upsert(&[doc("a", "a.txt", "t", vec![1.0, 0.0, 0.0, 0.0])])
            .expect("seed");
        store
            .set_sparse(
                "m",
                &[
                    ("ghost".to_string(), sv(&[(1, 1.0)])),
                    ("a".to_string(), sv(&[(1, 1.0)])),
                    ("a".to_string(), sv(&[(2, 4.0)])),
                ],
            )
            .expect("set sparse");
        assert_eq!(store.doc_count(), 1, "an unknown id is never inserted");
        let hits = store
            .query_sparse("m", &sv(&[(2, 1.0)]), None, 5)
            .expect("query");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].score, 4.0);
        store
            .set_sparse("m", &[])
            .expect("an empty batch is a no-op");
    }

    #[test]
    fn test_replacing_a_row_clears_its_sparse_vector() {
        let (_dir, mut store) = store();
        store
            .upsert(&[doc("a", "a.txt", "t", vec![1.0, 0.0, 0.0, 0.0])])
            .expect("seed");
        store
            .set_sparse("m", &[("a".to_string(), sv(&[(1, 1.0)]))])
            .expect("set sparse");
        assert_eq!(store.count_missing_sparse("m").expect("count"), 0);

        // The same id written again is new content as far as sparse goes.
        store
            .upsert(&[doc("a", "a.txt", "t2", vec![1.0, 0.0, 0.0, 0.0])])
            .expect("replace");

        assert_eq!(store.count_missing_sparse("m").expect("count"), 1);
        assert!(store
            .query_sparse("m", &sv(&[(1, 1.0)]), None, 5)
            .expect("query")
            .is_empty());
    }

    #[test]
    fn test_update_keeps_the_sparse_vector() {
        // A reused chunk (same id) only moves path/heading/token: its text,
        // and so its sparse vector, are unchanged.
        let (_dir, mut store) = store();
        store
            .upsert(&[doc("a", "old.txt", "t", vec![1.0, 0.0, 0.0, 0.0])])
            .expect("seed");
        store
            .set_sparse("m", &[("a".to_string(), sv(&[(1, 2.0)]))])
            .expect("set sparse");

        store
            .update(&[DocumentUpdate {
                id: "a".to_string(),
                seen_token: "t2".to_string(),
                path: "new.txt".to_string(),
                heading_path: "h".to_string(),
            }])
            .expect("update");

        let hits = store
            .query_sparse("m", &sv(&[(1, 1.0)]), None, 5)
            .expect("query");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, "new.txt");
        assert_eq!(store.count_missing_sparse("m").expect("count"), 0);
    }

    #[test]
    fn test_fetch_sparse_by_content_hash_is_model_scoped() {
        let (_dir, mut store) = store();
        store
            .upsert(&[
                doc("a", "a.txt", "t", vec![1.0, 0.0, 0.0, 0.0]),
                doc("b", "b.txt", "t", vec![1.0, 0.0, 0.0, 0.0]),
                doc("c", "c.txt", "t", vec![1.0, 0.0, 0.0, 0.0]),
            ])
            .expect("seed");
        store
            .set_sparse("m", &[("a".to_string(), sv(&[(4, 2.0)]))])
            .expect("m");
        store
            .set_sparse("old", &[("b".to_string(), sv(&[(5, 1.0)]))])
            .expect("old");

        let found = store
            .fetch_sparse_by_content_hash(
                "m",
                &[
                    "hash-a".to_string(),
                    "hash-b".to_string(),
                    "hash-c".to_string(),
                ],
            )
            .expect("fetch");
        assert_eq!(found.len(), 1);
        assert_eq!(found["hash-a"], sv(&[(4, 2.0)]));
        assert!(store
            .fetch_sparse_by_content_hash("m", &[])
            .expect("empty")
            .is_empty());
    }

    #[test]
    fn test_sparse_survives_reopen_and_flush() {
        let dir = tempfile::tempdir().expect("tempdir");
        {
            let mut store = LanceDbStore::open_or_create(dir.path(), 4).expect("create");
            store
                .upsert(&[doc("a", "a.txt", "t", vec![1.0, 0.0, 0.0, 0.0])])
                .expect("seed");
            store
                .set_sparse("m", &[("a".to_string(), sv(&[(7, 3.0)]))])
                .expect("set");
            store.flush().expect("flush compacts and prunes");
        }
        let store = LanceDbStore::open_or_create(dir.path(), 4).expect("reopen");
        let hits = store
            .query_sparse("m", &sv(&[(7, 1.0)]), None, 5)
            .expect("query");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].score, 3.0);
        let state = store.index_state().expect("state");
        assert!(!state.ann_indexed);
    }

    /// A table exactly as the BM25 release wrote it: `lexical_text` column
    /// and an FTS index on it, no sparse columns.
    fn write_bm25_era_table(path: &std::path::Path, rows: usize) {
        use arrow_array::types::Float32Type;
        use arrow_array::{FixedSizeListArray, RecordBatch, StringArray};
        use arrow_schema::{DataType, Field, Schema};
        use lancedb::index::scalar::FtsIndexBuilder;
        use lancedb::index::Index;
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 4),
                false,
            ),
            Field::new("text", DataType::Utf8, false),
            Field::new("lexical_text", DataType::Utf8, false),
            Field::new("source_id", DataType::Utf8, false),
            Field::new("path", DataType::Utf8, false),
            Field::new("chunk_schema_id", DataType::Utf8, false),
            Field::new("chunk_type", DataType::Utf8, false),
            Field::new("heading_path", DataType::Utf8, false),
            Field::new("content_hash", DataType::Utf8, false),
            Field::new("seen_token", DataType::Utf8, false),
        ]));
        let strings = |f: &dyn Fn(usize) -> String| -> Arc<StringArray> {
            Arc::new(StringArray::from_iter_values((0..rows).map(f)))
        };
        let vectors = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            (0..rows).map(|i| Some(vec![Some(i as f32), Some(1.0), Some(0.0), Some(0.0)])),
            4,
        );
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                strings(&|i| format!("old{i}")),
                Arc::new(vectors),
                strings(&|i| format!("legacy text {i}")),
                strings(&|i| format!("legacy tokenized {i}")),
                strings(&|_| "source-1".to_string()),
                strings(&|i| format!("f{i}.txt")),
                strings(&|_| "schema-1".to_string()),
                strings(&|_| "text".to_string()),
                strings(&|_| String::new()),
                strings(&|i| format!("hash-old{i}")),
                strings(&|_| "token".to_string()),
            ],
        )
        .expect("legacy batch");
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            let conn = lancedb::connect(path.to_str().unwrap())
                .execute()
                .await
                .unwrap();
            let table = conn
                .create_table("documents", vec![batch])
                .execute()
                .await
                .unwrap();
            table
                .create_index(&["lexical_text"], Index::FTS(FtsIndexBuilder::default()))
                .execute()
                .await
                .unwrap();
        });
    }

    #[test]
    fn test_bm25_era_table_is_migrated_in_place_and_keeps_dense_vectors() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_bm25_era_table(dir.path(), 5);

        let mut store = LanceDbStore::open_or_create(dir.path(), 4).expect("open migrates");

        assert_eq!(store.doc_count(), 5, "no row is lost");
        let fetched = store.fetch(&["old3".to_string()]).expect("fetch");
        assert_eq!(fetched[0].text, "legacy text 3");
        assert_eq!(
            fetched[0].embedding,
            vec![3.0, 1.0, 0.0, 0.0],
            "dense vector untouched"
        );
        // Every row reads as not-yet-encoded, so the next sync encodes it.
        assert_eq!(store.count_missing_sparse("m").expect("count"), 5);
        // The FTS index is gone with its column.
        let indices = store.index_names().expect("indices");
        assert!(
            !indices.iter().any(|name| name.contains("lexical")),
            "FTS index must be dropped, got {indices:?}"
        );
        // And the table now behaves like any sparse-ready one.
        store
            .set_sparse("m", &[("old3".to_string(), sv(&[(1, 1.0)]))])
            .expect("set sparse on a migrated table");
        let hits = store
            .query_sparse("m", &sv(&[(1, 1.0)]), None, 5)
            .expect("query");
        assert_eq!(hits[0].doc_id, "old3");
        store.flush().expect("flush on a migrated table");
        // Dense search still answers.
        assert_eq!(
            store.query(&[3.0, 1.0, 0.0, 0.0], None, 1).expect("dense")[0].doc_id,
            "old3"
        );
    }

    #[test]
    fn test_migration_is_idempotent_across_reopens() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_bm25_era_table(dir.path(), 2);
        {
            let mut store = LanceDbStore::open_or_create(dir.path(), 4).expect("migrate");
            store
                .set_sparse("m", &[("old0".to_string(), sv(&[(2, 2.0)]))])
                .expect("set");
        }
        let store = LanceDbStore::open_or_create(dir.path(), 4).expect("reopen");
        assert_eq!(store.doc_count(), 2);
        assert_eq!(
            store.count_missing_sparse("m").expect("count"),
            1,
            "sparse data kept"
        );
    }

    #[test]
    fn test_readonly_open_refuses_a_bm25_era_table_without_touching_it() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_bm25_era_table(dir.path(), 2);

        let error = LanceDbStore::open_readonly(dir.path(), 4, IndexingConfig::default())
            .err()
            .expect("read-only must not migrate");
        let message = error.to_string();
        assert!(message.contains("minsync sync"), "{message}");

        // Still a legacy table afterwards: a writer migrates it, not a reader.
        let writer = LanceDbStore::open_or_create(dir.path(), 4).expect("writer migrates");
        assert_eq!(writer.doc_count(), 2);
    }
}
