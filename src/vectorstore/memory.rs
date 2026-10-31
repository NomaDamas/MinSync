use crate::error::Result;
use crate::sparse::{dot_terms, SparseVector, TopK};
use crate::vectorstore::similarity::{cosine_similarity, matches_filter};
use crate::vectorstore::{Document, DocumentUpdate, Filter, QueryHit, SparseTarget, VectorStore};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

#[derive(Default)]
pub struct InMemoryStore {
    docs: HashMap<String, Document>,
    /// Sparse vector per row id, tagged with the model that produced it.
    sparse: HashMap<String, (String, SparseVector)>,
}

impl InMemoryStore {
    pub fn new() -> Self {
        Self::default()
    }

    #[cfg(test)]
    pub fn all_embeddings(&self) -> Vec<Vec<f32>> {
        self.docs
            .values()
            .map(|doc| doc.embedding.clone())
            .collect()
    }
}

impl VectorStore for InMemoryStore {
    fn upsert(&mut self, docs: &[Document]) -> Result<()> {
        for doc in docs {
            self.sparse.remove(&doc.id);
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

    fn fetch_embeddings_by_content_hash(
        &self,
        content_hashes: &[String],
    ) -> Result<HashMap<String, Vec<f32>>> {
        let wanted: HashSet<&str> = content_hashes.iter().map(String::as_str).collect();
        let mut found = HashMap::new();
        for doc in self.docs.values() {
            if wanted.contains(doc.content_hash.as_str())
                && doc.embedding.iter().all(|value| value.is_finite())
            {
                found
                    .entry(doc.content_hash.clone())
                    .or_insert_with(|| doc.embedding.clone());
            }
        }
        Ok(found)
    }

    fn fetch(&self, ids: &[String]) -> Result<Vec<Document>> {
        Ok(ids
            .iter()
            .filter_map(|id| self.docs.get(id).cloned())
            .collect())
    }

    fn documents(&self) -> Result<Vec<Document>> {
        let mut docs: Vec<_> = self.docs.values().cloned().collect();
        docs.sort_by(|left, right| left.id.cmp(&right.id));
        Ok(docs)
    }

    fn delete_by_filter(&mut self, filter: &Filter) -> Result<usize> {
        let original_len = self.docs.len();
        self.docs.retain(|_, doc| !matches_filter(doc, filter));
        let docs = &self.docs;
        self.sparse.retain(|id, _| docs.contains_key(id));

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
    fn rows_missing_sparse(&self, model: &str, limit: usize) -> Result<Vec<SparseTarget>> {
        let mut missing: Vec<&Document> = self
            .docs
            .values()
            .filter(|doc| {
                self.sparse
                    .get(&doc.id)
                    .is_none_or(|(stored, _)| stored != model)
            })
            .collect();
        missing.sort_by(|left, right| left.id.cmp(&right.id));
        Ok(missing
            .into_iter()
            .take(limit)
            .map(|doc| SparseTarget {
                id: doc.id.clone(),
                path: doc.path.clone(),
                content_hash: doc.content_hash.clone(),
                text: doc.text.clone(),
            })
            .collect())
    }

    fn count_missing_sparse(&self, model: &str) -> Result<usize> {
        Ok(self
            .docs
            .keys()
            .filter(|id| {
                self.sparse
                    .get(*id)
                    .is_none_or(|(stored, _)| stored != model)
            })
            .count())
    }

    fn set_sparse(&mut self, model: &str, rows: &[(String, SparseVector)]) -> Result<()> {
        for (id, vector) in rows {
            if self.docs.contains_key(id) {
                self.sparse
                    .insert(id.clone(), (model.to_string(), vector.clone()));
            }
        }
        Ok(())
    }

    fn fetch_sparse_by_content_hash(
        &self,
        model: &str,
        content_hashes: &[String],
    ) -> Result<HashMap<String, SparseVector>> {
        let wanted: HashSet<&str> = content_hashes.iter().map(String::as_str).collect();
        let mut found = HashMap::new();
        for doc in self.docs.values() {
            if !wanted.contains(doc.content_hash.as_str()) {
                continue;
            }
            if let Some((stored, vector)) = self.sparse.get(&doc.id) {
                if stored == model {
                    found
                        .entry(doc.content_hash.clone())
                        .or_insert_with(|| vector.clone());
                }
            }
        }
        Ok(found)
    }

    fn query_sparse(
        &self,
        model: &str,
        vector: &SparseVector,
        filter: Option<&Filter>,
        topk: usize,
    ) -> Result<Vec<QueryHit>> {
        let mut top = TopK::new(topk);
        for doc in self.docs.values() {
            if !filter.is_none_or(|active| matches_filter(doc, active)) {
                continue;
            }
            if let Some((stored, row)) = self.sparse.get(&doc.id) {
                if stored == model {
                    top.offer(
                        dot_terms(
                            vector.indices(),
                            vector.values(),
                            row.indices(),
                            row.values(),
                        ),
                        &doc.id,
                    );
                }
            }
        }
        Ok(top
            .into_sorted()
            .into_iter()
            .filter_map(|(score, id)| {
                let doc = self.docs.get(&id)?;
                Some(QueryHit {
                    doc_id: doc.id.clone(),
                    path: doc.path.clone(),
                    heading_path: doc.heading_path.clone(),
                    chunk_type: doc.chunk_type.clone(),
                    text: doc.text.clone(),
                    score,
                    content_hash: doc.content_hash.clone(),
                })
            })
            .collect())
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

#[cfg(test)]
mod sparse_tests {
    use super::*;

    fn doc(id: &str, path: &str, hash: &str) -> Document {
        Document {
            id: id.to_string(),
            embedding: vec![0.0],
            text: format!("text {id}"),
            source_id: "source-1".to_string(),
            path: path.to_string(),
            chunk_schema_id: "schema-1".to_string(),
            chunk_type: "text".to_string(),
            heading_path: String::new(),
            content_hash: hash.to_string(),
            seen_token: "t".to_string(),
        }
    }

    fn sv(pairs: &[(u32, f32)]) -> SparseVector {
        SparseVector::from_pairs(pairs.iter().copied())
    }

    #[test]
    fn a_row_is_missing_until_the_current_model_has_encoded_it() {
        let mut store = InMemoryStore::new();
        store
            .upsert(&[doc("a", "a.txt", "ha"), doc("b", "b.txt", "hb")])
            .unwrap();
        assert_eq!(store.count_missing_sparse("m1").unwrap(), 2);

        store
            .set_sparse("m1", &[("a".to_string(), sv(&[(1, 1.0)]))])
            .unwrap();
        let missing: Vec<String> = store
            .rows_missing_sparse("m1", 10)
            .unwrap()
            .into_iter()
            .map(|row| row.id)
            .collect();
        assert_eq!(missing, vec!["b"]);

        // Another model sees every row as missing again: switching models re-encodes.
        assert_eq!(store.count_missing_sparse("m2").unwrap(), 2);
    }

    #[test]
    fn an_empty_vector_still_counts_as_encoded() {
        let mut store = InMemoryStore::new();
        store.upsert(&[doc("a", "a.txt", "ha")]).unwrap();
        store
            .set_sparse("m", &[("a".to_string(), SparseVector::default())])
            .unwrap();
        assert_eq!(store.count_missing_sparse("m").unwrap(), 0);
        assert!(store
            .query_sparse("m", &sv(&[(1, 1.0)]), None, 5)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn rows_missing_sparse_honours_the_limit_in_a_stable_order() {
        let mut store = InMemoryStore::new();
        let docs: Vec<Document> = ["d", "b", "c", "a"]
            .iter()
            .map(|id| doc(id, "x.txt", &format!("h{id}")))
            .collect();
        store.upsert(&docs).unwrap();
        let ids: Vec<String> = store
            .rows_missing_sparse("m", 2)
            .unwrap()
            .into_iter()
            .map(|row| row.id)
            .collect();
        assert_eq!(ids, vec!["a", "b"]);
        assert_eq!(store.count_missing_sparse("m").unwrap(), 4);
    }

    #[test]
    fn query_sparse_ranks_by_inner_product_and_ignores_rows_without_a_shared_term() {
        let mut store = InMemoryStore::new();
        store
            .upsert(&[
                doc("a", "a.txt", "ha"),
                doc("b", "b.txt", "hb"),
                doc("c", "c.txt", "hc"),
            ])
            .unwrap();
        store
            .set_sparse(
                "m",
                &[
                    ("a".to_string(), sv(&[(1, 1.0), (2, 1.0)])),
                    ("b".to_string(), sv(&[(1, 3.0)])),
                    ("c".to_string(), sv(&[(9, 5.0)])),
                ],
            )
            .unwrap();
        let hits = store
            .query_sparse("m", &sv(&[(1, 2.0), (2, 0.5)]), None, 10)
            .unwrap();
        let ranked: Vec<(&str, f32)> = hits.iter().map(|h| (h.doc_id.as_str(), h.score)).collect();
        assert_eq!(ranked, vec![("b", 6.0), ("a", 2.5)]);
        // Rows from a different model are not comparable and are never scored.
        assert!(store
            .query_sparse("other", &sv(&[(1, 2.0)]), None, 10)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn query_sparse_applies_the_filter_and_topk() {
        let mut store = InMemoryStore::new();
        store
            .upsert(&[
                doc("a", "a.txt", "ha"),
                doc("b", "b.txt", "hb"),
                doc("c", "b.txt", "hc"),
            ])
            .unwrap();
        let rows: Vec<(String, SparseVector)> = ["a", "b", "c"]
            .iter()
            .enumerate()
            .map(|(i, id)| (id.to_string(), sv(&[(1, 1.0 + i as f32)])))
            .collect();
        store.set_sparse("m", &rows).unwrap();
        let filter = Filter::Eq("path".to_string(), "b.txt".to_string());
        let hits = store
            .query_sparse("m", &sv(&[(1, 1.0)]), Some(&filter), 1)
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "c");
        assert!(store
            .query_sparse("m", &sv(&[(1, 1.0)]), None, 0)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn set_sparse_never_inserts_an_unknown_id_and_delete_drops_the_sparse_entry() {
        let mut store = InMemoryStore::new();
        store.upsert(&[doc("a", "a.txt", "ha")]).unwrap();
        store
            .set_sparse(
                "m",
                &[
                    ("a".to_string(), sv(&[(1, 1.0)])),
                    ("ghost".to_string(), sv(&[(1, 1.0)])),
                ],
            )
            .unwrap();
        assert_eq!(store.doc_count(), 1);
        store
            .delete_by_filter(&Filter::Eq("path".to_string(), "a.txt".to_string()))
            .unwrap();
        store.upsert(&[doc("a", "a.txt", "ha")]).unwrap();
        // The reborn row has no sparse vector: it must be encoded again.
        assert_eq!(store.count_missing_sparse("m").unwrap(), 1);
    }

    #[test]
    fn fetch_sparse_by_content_hash_returns_only_the_requested_model_and_hashes() {
        let mut store = InMemoryStore::new();
        store
            .upsert(&[doc("a", "a.txt", "ha"), doc("b", "b.txt", "hb")])
            .unwrap();
        store
            .set_sparse("m", &[("a".to_string(), sv(&[(4, 2.0)]))])
            .unwrap();
        store
            .set_sparse("old", &[("b".to_string(), sv(&[(5, 1.0)]))])
            .unwrap();
        let found = store
            .fetch_sparse_by_content_hash("m", &["ha".to_string(), "hb".to_string()])
            .unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found["ha"], sv(&[(4, 2.0)]));
    }

    #[test]
    fn replacing_a_row_through_upsert_keeps_or_drops_its_sparse_vector_with_the_row() {
        // An upsert replaces the row's content, so the old vector no longer
        // describes it: it must be encoded again.
        let mut store = InMemoryStore::new();
        store.upsert(&[doc("a", "a.txt", "ha")]).unwrap();
        store
            .set_sparse("m", &[("a".to_string(), sv(&[(1, 1.0)]))])
            .unwrap();
        store.upsert(&[doc("a", "a.txt", "ha-new")]).unwrap();
        assert_eq!(store.count_missing_sparse("m").unwrap(), 1);
    }
}
