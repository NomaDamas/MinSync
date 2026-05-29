use crate::config::Config;
use crate::embedder::Embedder;
use crate::error::{MinSyncError, Result};
use crate::state::Cursor;
use crate::types::QueryResult;
use crate::vectorstore::{Filter, VectorStore};
use std::path::Path;

pub async fn query(
    minsync_dir: &Path,
    text: &str,
    k: usize,
    embedder: &dyn Embedder,
    store: &dyn VectorStore,
    filter_expr: Option<&str>,
) -> Result<Vec<QueryResult>> {
    if text.trim().is_empty() {
        return Err(MinSyncError::EmptyQuery);
    }

    let _config = Config::load(&minsync_dir.join("config.toml"))?;
    let cursor_path = minsync_dir.join("cursor.json");
    if !cursor_path.exists() {
        return Err(MinSyncError::NeverSynced);
    }
    let cursor = Cursor::load(&cursor_path)?;

    let query_vec = embedder.embed_query(text).await?;

    let mut filters = vec![Filter::Eq(
        "source_id".to_string(),
        cursor.source_id.clone(),
    )];
    if let Some(expr) = filter_expr {
        if !expr.trim().is_empty() {
            return Err(MinSyncError::Other(
                "filter expressions are not supported yet".to_string(),
            ));
        }
    }

    let filter = Filter::And(filters.drain(..).collect());
    let hits = store.query(&query_vec, Some(&filter), k)?;

    Ok(hits
        .into_iter()
        .enumerate()
        .map(|(i, hit)| QueryResult {
            doc_id: hit.doc_id,
            path: hit.path,
            heading_path: hit.heading_path,
            chunk_type: hit.chunk_type,
            text: hit.text,
            score: hit.score,
            content_commit: hit.content_hash,
            rank: i + 1,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::state::Cursor;
    use crate::vectorstore::memory::InMemoryStore;
    use crate::vectorstore::Document;
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
                .map(|text| match text.as_str() {
                    "search text" => vec![1.0, 0.0, 0.0, 0.0],
                    "mid text" => vec![0.7, 0.7, 0.0, 0.0],
                    _ => vec![0.0, 1.0, 0.0, 0.0],
                })
                .collect())
        }
    }

    struct SentinelEmbedder;

    const SENTINEL_EMBED_VEC: [f32; 4] = [1.0, 0.0, 0.0, 0.0];
    const SENTINEL_QUERY_VEC: [f32; 4] = [0.0, 1.0, 0.0, 0.0];

    #[async_trait::async_trait]
    impl Embedder for SentinelEmbedder {
        fn id(&self) -> &str {
            "sentinel"
        }

        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| SENTINEL_EMBED_VEC.to_vec()).collect())
        }

        async fn embed_query(&self, _text: &str) -> Result<Vec<f32>> {
            Ok(SENTINEL_QUERY_VEC.to_vec())
        }
    }

    fn write_query_state(dir: &TempDir, source_id: &str) {
        let minsync_dir = dir.path().join(".minsync");
        std::fs::create_dir_all(&minsync_dir).expect("create .minsync");

        Config::default_for(source_id)
            .save(&minsync_dir.join("config.toml"))
            .expect("save config");
        Cursor {
            source_id: source_id.to_string(),
            last_synced_at: "2026-05-28T00:00:00Z".to_string(),
            manifest_hash: "sha256:abc".to_string(),
            chunk_schema_id: "schema-1".to_string(),
            embedder_id: "mock".to_string(),
            collection_path: "store".to_string(),
        }
        .save(&minsync_dir.join("cursor.json"))
        .expect("save cursor");
    }

    fn doc(id: &str, source_id: &str, path: &str, embedding: Vec<f32>) -> Document {
        Document {
            id: id.to_string(),
            embedding,
            text: format!("text {id}"),
            source_id: source_id.to_string(),
            path: path.to_string(),
            chunk_schema_id: "schema-1".to_string(),
            chunk_type: "text".to_string(),
            heading_path: format!("heading {id}"),
            content_hash: format!("hash-{id}"),
            seen_token: "token".to_string(),
        }
    }

    #[tokio::test]
    async fn test_query_returns_results() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let minsync_dir = dir.path().join(".minsync");
        let mut store = InMemoryStore::new();
        let embedder = MockEmbedder;
        let source_id = "source-1";
        write_query_state(&dir, source_id);

        let query_vec = embedder
            .embed_single("search text")
            .await
            .expect("embed query");
        store
            .upsert(&[
                doc("a", source_id, "a.txt", vec![0.0, 1.0, 0.0, 0.0]),
                doc("b", source_id, "b.txt", query_vec.clone()),
                doc("c", source_id, "c.txt", vec![0.7, 0.7, 0.0, 0.0]),
            ])
            .expect("upsert docs");

        let results = query(&minsync_dir, "search text", 3, &embedder, &store, None)
            .await
            .expect("query succeeds");

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].doc_id, "b");
        assert_eq!(results[0].rank, 1);
        assert!(results[0].score >= results[1].score);
        assert!(results[1].score >= results[2].score);
    }

    #[tokio::test]
    async fn test_query_empty_text() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let minsync_dir = dir.path().join(".minsync");
        std::fs::create_dir_all(&minsync_dir).expect("create .minsync");

        let embedder = MockEmbedder;
        let store = InMemoryStore::new();

        let result = query(&minsync_dir, "", 5, &embedder, &store, None).await;

        assert!(matches!(result, Err(MinSyncError::EmptyQuery)));
    }

    #[tokio::test]
    async fn test_query_never_synced() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let minsync_dir = dir.path().join(".minsync");
        std::fs::create_dir_all(&minsync_dir).expect("create .minsync");
        Config::default_for("source-1")
            .save(&minsync_dir.join("config.toml"))
            .expect("save config");

        let embedder = MockEmbedder;
        let store = InMemoryStore::new();

        let result = query(&minsync_dir, "search text", 5, &embedder, &store, None).await;

        assert!(matches!(result, Err(MinSyncError::NeverSynced)));
    }

    #[tokio::test]
    async fn test_query_empty_store() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let minsync_dir = dir.path().join(".minsync");
        std::fs::create_dir_all(&minsync_dir).expect("create .minsync");
        write_query_state(&dir, "source-1");

        let embedder = MockEmbedder;
        let store = InMemoryStore::new();

        let results = query(&minsync_dir, "search text", 5, &embedder, &store, None)
            .await
            .expect("query succeeds");

        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_query_uses_embed_query() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let minsync_dir = dir.path().join(".minsync");
        let mut store = InMemoryStore::new();
        let embedder = SentinelEmbedder;
        let source_id = "source-1";
        write_query_state(&dir, source_id);

        store
            .upsert(&[
                doc("embed", source_id, "embed.txt", SENTINEL_EMBED_VEC.to_vec()),
                doc("query", source_id, "query.txt", SENTINEL_QUERY_VEC.to_vec()),
            ])
            .expect("upsert docs");

        let results = query(&minsync_dir, "search text", 2, &embedder, &store, None)
            .await
            .expect("query succeeds");

        assert_eq!(results[0].doc_id, "query");
    }
}
