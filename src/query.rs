use crate::cli::QueryMode;
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
    mode: QueryMode,
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

    let filter = Filter::And(std::mem::take(&mut filters));
    let hits = match mode {
        QueryMode::Vector => {
            let query_vec = embedder.embed_query(text).await?;
            store
                .query(&query_vec, Some(&filter), k)?
                .into_iter()
                .enumerate()
                .map(|(rank, hit)| (hit, Some(rank + 1), None))
                .collect()
        }
        QueryMode::Bm25 => store
            .query_text(text, Some(&filter), k)?
            .into_iter()
            .enumerate()
            .map(|(rank, hit)| (hit, None, Some(rank + 1)))
            .collect(),
        QueryMode::Hybrid => {
            let query_vec = embedder.embed_query(text).await?;
            let candidate_k = k.saturating_mul(4).max(k);
            let vector_hits = store.query(&query_vec, Some(&filter), candidate_k)?;
            let text_hits = store.query_text(text, Some(&filter), candidate_k)?;
            reciprocal_rank_fusion(vector_hits, text_hits, k)
        }
    };

    Ok(hits
        .into_iter()
        .enumerate()
        .map(|(i, (hit, vector_rank, bm25_rank))| QueryResult {
            doc_id: hit.doc_id,
            path: hit.path,
            heading_path: hit.heading_path,
            chunk_type: hit.chunk_type,
            text: hit.text,
            score: hit.score,
            content_commit: hit.content_hash,
            rank: i + 1,
            mode: mode.as_str().to_string(),
            vector_rank,
            bm25_rank,
        })
        .collect())
}

pub fn query_text(
    minsync_dir: &Path,
    text: &str,
    k: usize,
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
    if filter_expr.is_some_and(|expr| !expr.trim().is_empty()) {
        return Err(MinSyncError::Other(
            "filter expressions are not supported yet".to_string(),
        ));
    }
    let filter = Filter::And(vec![Filter::Eq("source_id".to_string(), cursor.source_id)]);
    Ok(store
        .query_text(text, Some(&filter), k)?
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
            mode: QueryMode::Bm25.as_str().to_string(),
            vector_rank: None,
            bm25_rank: Some(i + 1),
        })
        .collect())
}

fn reciprocal_rank_fusion(
    vector_hits: Vec<crate::vectorstore::QueryHit>,
    text_hits: Vec<crate::vectorstore::QueryHit>,
    topk: usize,
) -> Vec<(crate::vectorstore::QueryHit, Option<usize>, Option<usize>)> {
    use std::collections::HashMap;

    const RRF_K: f32 = 60.0;
    let mut fused = HashMap::new();
    for (source, hits) in [("vector", vector_hits), ("bm25", text_hits)] {
        for (rank, mut hit) in hits.into_iter().enumerate() {
            let contribution = 1.0 / (RRF_K + rank as f32 + 1.0);
            fused
                .entry(hit.doc_id.clone())
                .and_modify(
                    |existing: &mut (
                        crate::vectorstore::QueryHit,
                        Option<usize>,
                        Option<usize>,
                    )| {
                        existing.0.score += contribution;
                        if source == "vector" {
                            existing.1 = Some(rank + 1);
                        } else {
                            existing.2 = Some(rank + 1);
                        }
                    },
                )
                .or_insert_with(|| {
                    hit.score = contribution;
                    (
                        hit,
                        (source == "vector").then_some(rank + 1),
                        (source == "bm25").then_some(rank + 1),
                    )
                });
        }
    }
    let mut hits: Vec<_> = fused.into_values().collect();
    hits.sort_by(|left, right| {
        right
            .0
            .score
            .partial_cmp(&left.0.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.0.doc_id.cmp(&right.0.doc_id))
    });
    hits.truncate(topk);
    hits
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
            lexical_language: "simple".to_string(),
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

        let results = query(
            &minsync_dir,
            "search text",
            3,
            &embedder,
            &store,
            None,
            QueryMode::Vector,
        )
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

        let result = query(
            &minsync_dir,
            "",
            5,
            &embedder,
            &store,
            None,
            QueryMode::Vector,
        )
        .await;

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

        let result = query(
            &minsync_dir,
            "search text",
            5,
            &embedder,
            &store,
            None,
            QueryMode::Vector,
        )
        .await;

        let err = result.expect_err("query without cursor fails");
        assert!(matches!(err, MinSyncError::NeverSynced));
        assert!(
            err.to_string().contains("minsync sync --full"),
            "error must name the command that resolves the state, got: {err}"
        );
    }

    #[tokio::test]
    async fn test_query_empty_store() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let minsync_dir = dir.path().join(".minsync");
        std::fs::create_dir_all(&minsync_dir).expect("create .minsync");
        write_query_state(&dir, "source-1");

        let embedder = MockEmbedder;
        let store = InMemoryStore::new();

        let results = query(
            &minsync_dir,
            "search text",
            5,
            &embedder,
            &store,
            None,
            QueryMode::Vector,
        )
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

        let results = query(
            &minsync_dir,
            "search text",
            2,
            &embedder,
            &store,
            None,
            QueryMode::Vector,
        )
        .await
        .expect("query succeeds");

        assert_eq!(results[0].doc_id, "query");
    }
}
