pub mod lancedb_store;
pub mod memory;
pub mod similarity;

use crate::error::Result;
use crate::types::IndexState;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub embedding: Vec<f32>,
    pub text: String,
    pub source_id: String,
    pub path: String,
    pub chunk_schema_id: String,
    pub chunk_type: String,
    pub heading_path: String,
    pub content_hash: String,
    pub seen_token: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentUpdate {
    pub id: String,
    pub seen_token: String,
    pub path: String,
    pub heading_path: String,
}

#[derive(Debug, Clone)]
pub struct QueryHit {
    pub doc_id: String,
    pub path: String,
    pub heading_path: String,
    pub chunk_type: String,
    pub text: String,
    pub score: f32,
    pub content_hash: String,
}

/// A stored chunk that still needs a sparse vector from the current model.
#[derive(Debug, Clone)]
pub struct SparseTarget {
    pub id: String,
    pub path: String,
    pub content_hash: String,
    pub text: String,
}

#[derive(Debug, Clone)]
pub enum Filter {
    Eq(String, String),
    Neq(String, String),
    /// Field equals any of the values; an empty list matches nothing.
    In(String, Vec<String>),
    /// Field equals none of the values; an empty list matches everything.
    NotIn(String, Vec<String>),
    And(Vec<Filter>),
}

pub trait VectorStore: Send + Sync {
    fn upsert(&mut self, docs: &[Document]) -> Result<()>;
    /// Insert documents whose ids the caller knows are NOT in the store
    /// (e.g. just checked with [`VectorStore::existing_ids`]). Stores can
    /// make this a plain append instead of a replace-or-insert. Inserting an
    /// id that already exists is a caller bug; the default delegates to
    /// [`VectorStore::upsert`], which stays correct either way.
    fn insert_new(&mut self, docs: &[Document]) -> Result<()> {
        self.upsert(docs)
    }
    fn update(&mut self, updates: &[DocumentUpdate]) -> Result<()>;
    fn fetch(&self, ids: &[String]) -> Result<Vec<Document>>;
    /// Return a stored embedding for each of `content_hashes` that already
    /// exists in the store, keyed by content hash. Embeddings depend only on
    /// chunk text (the embedder is fixed per collection), so sync reuses them
    /// instead of re-embedding identical chunks found under other paths.
    /// Stores that cannot look this up efficiently return an empty map.
    fn fetch_embeddings_by_content_hash(
        &self,
        _content_hashes: &[String],
    ) -> Result<std::collections::HashMap<String, Vec<f32>>> {
        Ok(std::collections::HashMap::new())
    }
    fn documents(&self) -> Result<Vec<Document>> {
        Ok(Vec::new())
    }
    /// Which of `ids` are present. Stores should avoid materializing full
    /// rows (vectors, text) for what is only an existence check.
    fn existing_ids(&self, ids: &[String]) -> Result<std::collections::HashSet<String>> {
        Ok(self.fetch(ids)?.into_iter().map(|doc| doc.id).collect())
    }
    fn delete_by_filter(&mut self, filter: &Filter) -> Result<usize>;
    fn query(&self, vector: &[f32], filter: Option<&Filter>, topk: usize) -> Result<Vec<QueryHit>>;

    /// At most `limit` rows that hold no sparse vector from `model`: never
    /// encoded, or encoded by another model. Sync encodes these, so a fresh
    /// row, an old row after a model change and a row left over from an
    /// interrupted run are all the same case.
    fn rows_missing_sparse(&self, model: &str, limit: usize) -> Result<Vec<SparseTarget>>;
    /// How many rows [`VectorStore::rows_missing_sparse`] would return without a limit.
    fn count_missing_sparse(&self, model: &str) -> Result<usize>;
    /// Attach sparse vectors from `model` to existing rows by id. An id that
    /// is no longer stored is ignored, never inserted. An empty vector is a
    /// valid answer (the text has no terms) and marks the row as done.
    fn set_sparse(&mut self, model: &str, rows: &[(String, SparseVector)]) -> Result<()>;
    /// A stored sparse vector from `model` for each of `content_hashes` that
    /// has one, keyed by content hash. Identical text encodes identically,
    /// so sync copies these instead of running the model again.
    fn fetch_sparse_by_content_hash(
        &self,
        model: &str,
        content_hashes: &[String],
    ) -> Result<std::collections::HashMap<String, SparseVector>>;
    /// Exact top-`topk` by inner product against rows encoded by `model`;
    /// rows that share no term with `vector` are not hits.
    fn query_sparse(
        &self,
        model: &str,
        vector: &SparseVector,
        filter: Option<&Filter>,
        topk: usize,
    ) -> Result<Vec<QueryHit>>;
    fn flush(&mut self) -> Result<()>;
    /// Cheap mid-sync maintenance so a long run that is killed leaves a
    /// store that is still fast to reopen: compact once fragments pile up
    /// and prune old versions. Unlike [`VectorStore::flush`] it never
    /// retrains or rebuilds search indexes. Stores with nothing to maintain
    /// keep the default no-op.
    fn checkpoint(&mut self) -> Result<()> {
        Ok(())
    }
    fn doc_count(&self) -> usize;
    fn all_paths(&self) -> Vec<String>;
    fn index_state(&self) -> Result<Option<IndexState>> {
        Ok(None)
    }
}

use crate::config::Config;
use crate::error::MinSyncError;
use crate::sparse::SparseVector;
use std::path::Path;

/// Construct a [`VectorStore`] implementation from the vectorstore id in `config`.
///
/// Returns [`MinSyncError::Config`] when the id is unknown.
pub fn create_vectorstore(config: &Config, store_path: &Path) -> Result<Box<dyn VectorStore>> {
    create_vectorstore_with_mode(config, store_path, false)
}

/// Construct a read-only [`VectorStore`] for querying. Unlike
/// [`create_vectorstore`], this never creates the table or migrates its
/// schema, so it is safe to use without the writer lock while a sync runs.
pub fn create_vectorstore_readonly(
    config: &Config,
    store_path: &Path,
) -> Result<Box<dyn VectorStore>> {
    create_vectorstore_with_mode(config, store_path, true)
}

fn create_vectorstore_with_mode(
    config: &Config,
    store_path: &Path,
    read_only: bool,
) -> Result<Box<dyn VectorStore>> {
    match config.vectorstore.id.as_str() {
        "lancedb" => {
            let options = Some(&config.vectorstore.options);
            let mut indexing = lancedb_store::IndexingConfig::from_options(options)?;
            let dimension = if config.is_sparse_only() {
                // Vectors are placeholders: pin the width so a stale
                // `dimension` option cannot reject them, and skip ANN.
                indexing.build_ann_index = false;
                crate::config::SPARSE_ONLY_DIMENSION
            } else {
                lancedb_store::LanceDbStore::dimension_from_options(options)?
            };
            let store = if read_only {
                lancedb_store::LanceDbStore::open_readonly(store_path, dimension, indexing)?
            } else {
                lancedb_store::LanceDbStore::open_with_indexing(store_path, dimension, indexing)?
            };
            Ok(Box::new(store))
        }
        other => Err(MinSyncError::Config(format!(
            "unknown vectorstore id: {other}"
        ))),
    }
}

#[cfg(test)]
mod factory_tests {
    use super::*;
    use crate::config::Config;

    #[test]
    fn test_create_vectorstore_lancedb() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let mut config = Config::default_for("12345678-1234-4234-9234-123456789abc");
        config.vectorstore.id = "lancedb".to_string();
        let mut table = toml::value::Table::new();
        table.insert("dimension".into(), toml::Value::Integer(4));
        config.vectorstore.options = toml::Value::Table(table);
        let store = create_vectorstore(&config, dir.path()).expect("create lancedb store");
        assert_eq!(store.doc_count(), 0);
    }

    #[test]
    fn test_create_vectorstore_sparse_only_ignores_dimension_and_skips_ann() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let mut config = Config::default_for("12345678-1234-4234-9234-123456789abc");
        config.embedder.id = "none".to_string();
        // A stale dimension (e.g. left over from `init --force`) must not matter.
        let mut table = toml::value::Table::new();
        table.insert("dimension".into(), toml::Value::Integer(1024));
        table.insert("index_build_threshold".into(), toml::Value::Integer(1));
        config.vectorstore.options = toml::Value::Table(table);

        let mut store = create_vectorstore(&config, dir.path()).expect("create store");
        let docs: Vec<Document> = (0..4)
            .map(|i| Document {
                id: format!("id-{i}"),
                embedding: vec![0.0; crate::config::SPARSE_ONLY_DIMENSION],
                text: format!("lexical row {i}"),
                source_id: "s".to_string(),
                path: format!("{i}.txt"),
                chunk_schema_id: "recursive".to_string(),
                chunk_type: "text".to_string(),
                heading_path: String::new(),
                content_hash: format!("h{i}"),
                seen_token: "t".to_string(),
            })
            .collect();
        store.upsert(&docs).expect("upsert placeholder vectors");
        store.flush().expect("flush must not build an ANN index");

        let state = store.index_state().expect("state").expect("lancedb state");
        assert!(!state.ann_indexed);
        assert_eq!(store.doc_count(), 4);
    }

    #[test]
    fn test_create_vectorstore_unknown() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let mut config = Config::default_for("12345678-1234-4234-9234-123456789abc");
        config.vectorstore.id = "bogus".to_string();
        let result = create_vectorstore(&config, dir.path());
        assert!(matches!(result, Err(MinSyncError::Config(_))));
    }
}
