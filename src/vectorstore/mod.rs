pub mod json_store;
pub mod lancedb_store;
pub mod memory;

use crate::error::Result;
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

#[derive(Debug, Clone)]
pub enum Filter {
    Eq(String, String),
    Neq(String, String),
    And(Vec<Filter>),
}

pub trait VectorStore: Send + Sync {
    fn upsert(&mut self, docs: &[Document]) -> Result<()>;
    fn update(&mut self, updates: &[DocumentUpdate]) -> Result<()>;
    fn fetch(&self, ids: &[String]) -> Result<Vec<Document>>;
    fn delete_by_filter(&mut self, filter: &Filter) -> Result<usize>;
    fn query(&self, vector: &[f32], filter: Option<&Filter>, topk: usize) -> Result<Vec<QueryHit>>;
    fn flush(&mut self) -> Result<()>;
    fn doc_count(&self) -> usize;
    fn all_paths(&self) -> Vec<String>;
}

use crate::config::Config;
use crate::error::MinSyncError;
use std::path::Path;

/// Construct a [`VectorStore`] implementation from the vectorstore id in `config`.
///
/// Returns [`MinSyncError::Config`] when the id is unknown.
pub fn create_vectorstore(config: &Config, store_path: &Path) -> Result<Box<dyn VectorStore>> {
    match config.vectorstore.id.as_str() {
        "json" => Ok(Box::new(json_store::JsonStore::new(store_path)?)),
        "lancedb" => {
            let dimension =
                lancedb_store::LanceDbStore::dimension_from_options(Some(&config.vectorstore.options))?;
            Ok(Box::new(lancedb_store::LanceDbStore::open_or_create(
                store_path, dimension,
            )?))
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
    fn test_create_vectorstore_json() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let config = Config::default_for("12345678-1234-4234-9234-123456789abc");
        let store = create_vectorstore(&config, dir.path()).expect("create json store");
        assert_eq!(store.doc_count(), 0);
    }

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
    fn test_create_vectorstore_unknown() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let mut config = Config::default_for("12345678-1234-4234-9234-123456789abc");
        config.vectorstore.id = "bogus".to_string();
        let result = create_vectorstore(&config, dir.path());
        assert!(matches!(result, Err(MinSyncError::Config(_))));
    }
}
