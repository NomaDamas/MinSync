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
use crate::types::IndexState;
use crate::vectorstore::{Document, DocumentUpdate, Filter, QueryHit, VectorStore};
use lancedb::DistanceType;
use std::path::Path;
use std::sync::mpsc;
use std::thread::{self, JoinHandle};
use worker::{run_worker, Command, Resp};

const DEFAULT_DIMENSION: usize = 1536;
const TABLE_NAME: &str = "documents";
const VECTOR_COLUMN: &str = "vector";
const DISTANCE_COLUMN: &str = "_distance";

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
        Self::open_with_language(path, dimension, indexing, "simple")
    }

    pub fn open_with_language(
        path: &Path,
        dimension: usize,
        indexing: IndexingConfig,
        language: &str,
    ) -> Result<Self> {
        crate::tokenizer::validate_language(language)?;
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
        let language = language.to_string();
        let worker =
            thread::spawn(move || run_worker(uri, dim, indexing, language, cmd_rx, init_tx));

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

    fn fetch(&self, ids: &[String]) -> Result<Vec<Document>> {
        self.request(|resp| Command::Fetch(ids.to_vec(), resp))
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

    fn query_text(
        &self,
        text: &str,
        filter: Option<&Filter>,
        topk: usize,
    ) -> Result<Vec<QueryHit>> {
        self.request(|resp| Command::QueryText {
            text: text.to_string(),
            filter: filter.cloned(),
            topk,
            resp,
        })
    }

    fn flush(&mut self) -> Result<()> {
        self.request(Command::Flush)
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
    fn test_no_index_built_below_threshold() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let mut store = indexed_store(&dir, 1000);
        let docs: Vec<_> = (0..50).map(spread_doc).collect();
        store.upsert(&docs).expect("upsert docs");
        store.flush().expect("flush");
        let indices = store.index_names().expect("list indices");
        assert!(
            indices.len() == 1 && indices[0].contains("text"),
            "only the BM25 text index should exist below the ANN row threshold, got {indices:?}"
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
}
