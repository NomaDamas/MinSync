use crate::error::Result;
use crate::vectorstore::similarity::{cosine_similarity, matches_filter};
use crate::vectorstore::{Document, DocumentUpdate, Filter, QueryHit, VectorStore};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

#[derive(Default)]
pub struct InMemoryStore {
    docs: HashMap<String, Document>,
}

impl InMemoryStore {
    pub fn new() -> Self {
        Self::default()
    }
}

impl VectorStore for InMemoryStore {
    fn upsert(&mut self, docs: &[Document]) -> Result<()> {
        for doc in docs {
            self.docs.insert(doc.id.clone(), doc.clone());
        }

        Ok(())
    }

    fn update(&mut self, updates: &[DocumentUpdate]) -> Result<()> {
        for update in updates {
            if let Some(doc) = self.docs.get_mut(&update.id) {
                doc.seen_token = update.seen_token.clone();
                doc.path = update.path.clone();
                doc.heading_path = update.heading_path.clone();
            }
        }

        Ok(())
    }

    fn fetch(&self, ids: &[String]) -> Result<Vec<Document>> {
        Ok(ids
            .iter()
            .filter_map(|id| self.docs.get(id).cloned())
            .collect())
    }

    fn delete_by_filter(&mut self, filter: &Filter) -> Result<usize> {
        let original_len = self.docs.len();
        self.docs.retain(|_, doc| !matches_filter(doc, filter));

        Ok(original_len.saturating_sub(self.docs.len()))
    }

    fn query(&self, vector: &[f32], filter: Option<&Filter>, topk: usize) -> Result<Vec<QueryHit>> {
        let mut hits: Vec<_> = self
            .docs
            .values()
            .filter(|doc| filter.is_none_or(|active_filter| matches_filter(doc, active_filter)))
            .map(|doc| QueryHit {
                doc_id: doc.id.clone(),
                path: doc.path.clone(),
                heading_path: doc.heading_path.clone(),
                chunk_type: doc.chunk_type.clone(),
                text: doc.text.clone(),
                score: cosine_similarity(vector, &doc.embedding),
                content_hash: doc.content_hash.clone(),
            })
            .collect();

        hits.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(Ordering::Equal)
        });
        hits.truncate(topk);
        Ok(hits)
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }

    fn doc_count(&self) -> usize {
        self.docs.len()
    }

    fn all_paths(&self) -> Vec<String> {
        let mut paths: Vec<_> = self
            .docs
            .values()
            .map(|doc| doc.path.clone())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        paths.sort();
        paths
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_memory_store_basic() {
        let mut store = InMemoryStore::new();
        store
            .upsert(&[
                doc("a", "keep.txt", "old", vec![1.0, 0.0]),
                doc("b", "drop.txt", "old", vec![0.0, 1.0]),
            ])
            .expect("upsert docs");
        store
            .update(&[DocumentUpdate {
                id: "a".to_string(),
                seen_token: "new".to_string(),
                path: "keep.txt".to_string(),
                heading_path: "updated heading".to_string(),
            }])
            .expect("update doc");

        let fetched = store.fetch(&["a".to_string()]).expect("fetch doc");
        let hits = store.query(&[1.0, 0.0], None, 1).expect("query docs");
        let deleted = store
            .delete_by_filter(&Filter::Eq("path".to_string(), "drop.txt".to_string()))
            .expect("delete docs");

        assert_eq!(fetched[0].seen_token, "new");
        assert_eq!(hits[0].doc_id, "a");
        assert_eq!(deleted, 1);
        assert_eq!(store.doc_count(), 1);
    }
}
