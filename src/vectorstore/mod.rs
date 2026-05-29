pub mod json_store;
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
