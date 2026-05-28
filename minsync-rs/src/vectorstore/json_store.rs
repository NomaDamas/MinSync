use crate::error::{MinSyncError, Result};
use crate::vectorstore::{Document, DocumentUpdate, Filter, QueryHit, VectorStore};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use tempfile::NamedTempFile;

pub struct JsonStore {
    docs: Vec<Document>,
    index: HashMap<String, usize>,
    path: PathBuf,
    dirty: bool,
}

impl JsonStore {
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let docs_path = path.join("docs.json");
        let docs = if docs_path.exists() {
            let content = fs::read_to_string(&docs_path)?;
            serde_json::from_str(&content)?
        } else {
            Vec::new()
        };
        let index = build_index(&docs);

        Ok(Self {
            docs,
            index,
            path,
            dirty: false,
        })
    }

    fn docs_path(&self) -> PathBuf {
        self.path.join("docs.json")
    }

    fn rebuild_index(&mut self) {
        self.index = build_index(&self.docs);
    }
}

impl VectorStore for JsonStore {
    fn upsert(&mut self, docs: &[Document]) -> Result<()> {
        for doc in docs {
            if let Some(position) = self.index.get(&doc.id).copied() {
                self.docs[position] = doc.clone();
            } else {
                self.index.insert(doc.id.clone(), self.docs.len());
                self.docs.push(doc.clone());
            }
        }

        if !docs.is_empty() {
            self.dirty = true;
        }

        Ok(())
    }

    fn update(&mut self, updates: &[DocumentUpdate]) -> Result<()> {
        let mut changed = false;
        for update in updates {
            if let Some(position) = self.index.get(&update.id).copied() {
                let doc = &mut self.docs[position];
                doc.seen_token = update.seen_token.clone();
                doc.path = update.path.clone();
                doc.heading_path = update.heading_path.clone();
                changed = true;
            }
        }

        if changed {
            self.dirty = true;
        }

        Ok(())
    }

    fn fetch(&self, ids: &[String]) -> Result<Vec<Document>> {
        Ok(ids
            .iter()
            .filter_map(|id| {
                self.index
                    .get(id)
                    .map(|position| self.docs[*position].clone())
            })
            .collect())
    }

    fn delete_by_filter(&mut self, filter: &Filter) -> Result<usize> {
        let original_len = self.docs.len();
        self.docs.retain(|doc| !matches_filter(doc, filter));
        let deleted = original_len.saturating_sub(self.docs.len());

        if deleted > 0 {
            self.rebuild_index();
            self.dirty = true;
        }

        Ok(deleted)
    }

    fn query(&self, vector: &[f32], filter: Option<&Filter>, topk: usize) -> Result<Vec<QueryHit>> {
        let mut hits: Vec<_> = self
            .docs
            .iter()
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
        if !self.dirty {
            return Ok(());
        }

        fs::create_dir_all(&self.path)?;
        let docs_path = self.docs_path();
        let mut tmp = NamedTempFile::new_in(&self.path)?;
        let content = serde_json::to_vec_pretty(&self.docs)?;
        tmp.write_all(&content)?;
        tmp.flush()?;
        tmp.as_file().sync_all()?;
        tmp.persist(&docs_path)
            .map_err(|error| MinSyncError::Io(error.error))?;
        self.dirty = false;

        Ok(())
    }

    fn doc_count(&self) -> usize {
        self.docs.len()
    }

    fn all_paths(&self) -> Vec<String> {
        let mut paths: Vec<_> = self
            .docs
            .iter()
            .map(|doc| doc.path.clone())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        paths.sort();
        paths
    }
}

fn build_index(docs: &[Document]) -> HashMap<String, usize> {
    docs.iter()
        .enumerate()
        .map(|(position, doc)| (doc.id.clone(), position))
        .collect()
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for (left, right) in a.iter().zip(b) {
        dot += left * right;
        norm_a += left * left;
        norm_b += right * right;
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a.sqrt() * norm_b.sqrt())
}

pub fn matches_filter(doc: &Document, filter: &Filter) -> bool {
    match filter {
        Filter::Eq(field, value) => {
            get_field(doc, field).is_some_and(|doc_value| doc_value == value)
        }
        Filter::Neq(field, value) => {
            get_field(doc, field).is_some_and(|doc_value| doc_value != value)
        }
        Filter::And(filters) => filters.iter().all(|nested| matches_filter(doc, nested)),
    }
}

fn get_field<'a>(doc: &'a Document, field_name: &str) -> Option<&'a str> {
    match field_name {
        "id" => Some(&doc.id),
        "text" => Some(&doc.text),
        "source_id" => Some(&doc.source_id),
        "path" => Some(&doc.path),
        "chunk_schema_id" => Some(&doc.chunk_schema_id),
        "chunk_type" => Some(&doc.chunk_type),
        "heading_path" => Some(&doc.heading_path),
        "content_hash" => Some(&doc.content_hash),
        "seen_token" => Some(&doc.seen_token),
        _ => None,
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

    fn store() -> JsonStore {
        let dir = tempfile::tempdir().expect("create tempdir");
        JsonStore::new(dir.path()).expect("create json store")
    }

    #[test]
    fn test_upsert_and_fetch() {
        let mut store = store();
        let docs = vec![
            doc("a", "a.txt", "token-1", vec![1.0, 0.0]),
            doc("b", "b.txt", "token-1", vec![0.0, 1.0]),
            doc("c", "c.txt", "token-1", vec![1.0, 1.0]),
        ];

        store.upsert(&docs).expect("upsert docs");
        let fetched = store
            .fetch(&["a".to_string(), "b".to_string(), "c".to_string()])
            .expect("fetch docs");

        assert_eq!(fetched.len(), 3);
        assert_eq!(fetched[0].id, "a");
        assert_eq!(fetched[1].id, "b");
        assert_eq!(fetched[2].id, "c");
    }

    #[test]
    fn test_upsert_replaces() {
        let mut store = store();
        store
            .upsert(&[doc("a", "old.txt", "old", vec![1.0, 0.0])])
            .expect("upsert old");
        store
            .upsert(&[doc("a", "new.txt", "new", vec![0.0, 1.0])])
            .expect("upsert new");

        let fetched = store.fetch(&["a".to_string()]).expect("fetch doc");

        assert_eq!(store.doc_count(), 1);
        assert_eq!(fetched[0].path, "new.txt");
        assert_eq!(fetched[0].seen_token, "new");
    }

    #[test]
    fn test_update_metadata() {
        let mut store = store();
        store
            .upsert(&[doc("a", "old.txt", "old", vec![1.0])])
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
        let mut store = store();
        store
            .upsert(&[
                doc("a", "x.txt", "token", vec![1.0]),
                doc("b", "x.txt", "token", vec![1.0]),
                doc("c", "y.txt", "token", vec![1.0]),
                doc("d", "y.txt", "token", vec![1.0]),
                doc("e", "y.txt", "token", vec![1.0]),
            ])
            .expect("upsert docs");

        let deleted = store
            .delete_by_filter(&Filter::Eq("path".to_string(), "x.txt".to_string()))
            .expect("delete docs");

        assert_eq!(deleted, 2);
        assert_eq!(store.doc_count(), 3);
    }

    #[test]
    fn test_delete_by_neq_filter() {
        let mut store = store();
        store
            .upsert(&[
                doc("a", "a.txt", "keep", vec![1.0]),
                doc("b", "b.txt", "drop", vec![1.0]),
                doc("c", "c.txt", "drop", vec![1.0]),
            ])
            .expect("upsert docs");

        let deleted = store
            .delete_by_filter(&Filter::Neq("seen_token".to_string(), "keep".to_string()))
            .expect("delete docs");

        assert_eq!(deleted, 2);
        assert_eq!(
            store.fetch(&["a".to_string()]).expect("fetch kept").len(),
            1
        );
    }

    #[test]
    fn test_delete_by_and_filter() {
        let mut store = store();
        store
            .upsert(&[
                doc("a", "same.txt", "keep", vec![1.0]),
                doc("b", "same.txt", "drop", vec![1.0]),
                doc("c", "other.txt", "drop", vec![1.0]),
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
        assert!(store
            .fetch(&["b".to_string()])
            .expect("fetch deleted")
            .is_empty());
    }

    #[test]
    fn test_query_topk() {
        let mut store = store();
        let docs: Vec<_> = (0..10)
            .map(|i| {
                doc(
                    &format!("doc-{i}"),
                    "path.txt",
                    "token",
                    vec![i as f32, 0.0],
                )
            })
            .collect();
        store.upsert(&docs).expect("upsert docs");

        let hits = store.query(&[10.0, 0.0], None, 3).expect("query docs");

        assert_eq!(hits.len(), 3);
        assert!(hits[0].score >= hits[1].score);
        assert!(hits[1].score >= hits[2].score);
        assert_eq!(hits[0].doc_id, "doc-1");
    }

    #[test]
    fn test_query_with_filter() {
        let mut store = store();
        store
            .upsert(&[
                doc("a", "x.txt", "token", vec![1.0, 0.0]),
                doc("b", "y.txt", "token", vec![1.0, 0.0]),
                doc("c", "x.txt", "token", vec![0.0, 1.0]),
            ])
            .expect("upsert docs");

        let hits = store
            .query(
                &[1.0, 0.0],
                Some(&Filter::Eq("path".to_string(), "x.txt".to_string())),
                10,
            )
            .expect("query docs");

        assert_eq!(hits.len(), 2);
        assert!(hits.iter().all(|hit| hit.path == "x.txt"));
    }

    #[test]
    fn test_query_empty_store() {
        let store = store();

        let hits = store
            .query(&[1.0, 0.0], None, 10)
            .expect("query empty store");

        assert!(hits.is_empty());
    }

    #[test]
    fn test_persistence_roundtrip() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let mut store = JsonStore::new(dir.path()).expect("create store");
        store
            .upsert(&[doc("a", "a.txt", "token", vec![1.0, 0.0])])
            .expect("upsert doc");
        store.flush().expect("flush store");

        let loaded = JsonStore::new(dir.path()).expect("load store");
        let fetched = loaded.fetch(&["a".to_string()]).expect("fetch loaded doc");

        assert_eq!(fetched.len(), 1);
        assert_eq!(fetched[0].path, "a.txt");
    }

    #[test]
    fn test_doc_count() {
        let mut store = store();
        store
            .upsert(&[
                doc("a", "a.txt", "token", vec![1.0]),
                doc("b", "b.txt", "token", vec![1.0]),
            ])
            .expect("upsert docs");

        assert_eq!(store.doc_count(), 2);
    }

    #[test]
    fn test_all_paths() {
        let mut store = store();
        store
            .upsert(&[
                doc("a", "a.txt", "token", vec![1.0]),
                doc("b", "b.txt", "token", vec![1.0]),
                doc("c", "c.txt", "token", vec![1.0]),
                doc("d", "a.txt", "token", vec![1.0]),
            ])
            .expect("upsert docs");

        assert_eq!(store.all_paths(), vec!["a.txt", "b.txt", "c.txt"]);
    }

    #[test]
    fn test_cosine_similarity_known() {
        let similarity = cosine_similarity(&[1.0, 1.0], &[1.0, 0.0]);

        assert!((similarity - 0.707).abs() < 0.001);
    }
}
