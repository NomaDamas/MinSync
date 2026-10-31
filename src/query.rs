use crate::cli::QueryMode;
use crate::config::Config;
use crate::embedder::Embedder;
use crate::error::{MinSyncError, Result};
use crate::sparse::SparseEncoder;
use crate::state::Cursor;
use crate::types::QueryResult;
use crate::vectorstore::{Filter, QueryHit, VectorStore};
use std::path::Path;

/// Search the index.
///
/// `vector` needs the dense `embedder`; `sparse` needs the sparse `encoder`;
/// `hybrid` needs both and fuses the two rankings. A mode whose half is
/// missing is an error, never a silent fallback to the other.
#[allow(clippy::too_many_arguments)]
pub async fn query(
    minsync_dir: &Path,
    text: &str,
    k: usize,
    embedder: &dyn Embedder,
    encoder: Option<&dyn SparseEncoder>,
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
    let index_state = store.index_state()?;
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
        QueryMode::Sparse => sparse_hits(&cursor, encoder, store, text, &filter, k)
            .await?
            .into_iter()
            .enumerate()
            .map(|(rank, hit)| (hit, None, Some(rank + 1)))
            .collect(),
        QueryMode::Hybrid => {
            let query_vec = embedder.embed_query(text).await?;
            let candidate_k = k.saturating_mul(4).max(k);
            let vector_hits = store.query(&query_vec, Some(&filter), candidate_k)?;
            let sparse_hits =
                sparse_hits(&cursor, encoder, store, text, &filter, candidate_k).await?;
            reciprocal_rank_fusion(vector_hits, sparse_hits, k)
        }
    };

    Ok(hits
        .into_iter()
        .enumerate()
        .map(|(i, (hit, vector_rank, sparse_rank))| QueryResult {
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
            sparse_rank,
            index_state: index_state.clone(),
        })
        .collect())
}

/// Top sparse hits for `text`, or why the index cannot answer a sparse query.
///
/// Scores from different models are not comparable, and a row without a
/// vector from the configured model can never be found, so a cursor whose
/// `sparse_id` is not the encoder's model means the last sync did not leave
/// the whole index encoded by it.
async fn sparse_hits(
    cursor: &Cursor,
    encoder: Option<&dyn SparseEncoder>,
    store: &dyn VectorStore,
    text: &str,
    filter: &Filter,
    k: usize,
) -> Result<Vec<QueryHit>> {
    let encoder = encoder.ok_or_else(|| {
        MinSyncError::Config("sparse search needs a sparse encoder, but none is configured".into())
    })?;
    match cursor.sparse_id.as_deref() {
        Some(indexed) if indexed == encoder.id() => {}
        Some(indexed) => {
            return Err(MinSyncError::SparseNotReady(format!(
                "the index was encoded with `{indexed}`, the configured model is `{}`",
                encoder.id()
            )))
        }
        None => {
            return Err(MinSyncError::SparseNotReady(
                "no sparse vectors have been built for this index".to_string(),
            ))
        }
    }
    let vector = encoder.encode_query(text).await?;
    store.query_sparse(encoder.id(), &vector, Some(filter), k)
}

fn reciprocal_rank_fusion(
    vector_hits: Vec<QueryHit>,
    sparse_hits: Vec<QueryHit>,
    topk: usize,
) -> Vec<(QueryHit, Option<usize>, Option<usize>)> {
    use std::collections::HashMap;

    const RRF_K: f32 = 60.0;
    let mut fused: HashMap<String, (QueryHit, Option<usize>, Option<usize>)> = HashMap::new();
    for (is_vector, hits) in [(true, vector_hits), (false, sparse_hits)] {
        for (rank, mut hit) in hits.into_iter().enumerate() {
            let contribution = 1.0 / (RRF_K + rank as f32 + 1.0);
            let rank = rank + 1;
            match fused.get_mut(&hit.doc_id) {
                Some(existing) => {
                    existing.0.score += contribution;
                    if is_vector {
                        existing.1 = Some(rank);
                    } else {
                        existing.2 = Some(rank);
                    }
                }
                None => {
                    hit.score = contribution;
                    let entry = (hit, is_vector.then_some(rank), (!is_vector).then_some(rank));
                    fused.insert(entry.0.doc_id.clone(), entry);
                }
            }
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
    use crate::sparse::SparseVector;
    use crate::state::Cursor;
    use crate::vectorstore::memory::InMemoryStore;
    use crate::vectorstore::Document;
    use async_trait::async_trait;
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
            sparse_id: Some("fake/sparse".to_string()),
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
            None,
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
            None,
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
            None,
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
            None,
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
            None,
            &store,
            None,
            QueryMode::Vector,
        )
        .await
        .expect("query succeeds");

        assert_eq!(results[0].doc_id, "query");
    }

    /// Encodes a query as the terms named by its words: `t3` is term 3.
    struct FakeSparse(&'static str);

    #[async_trait]
    impl SparseEncoder for FakeSparse {
        fn id(&self) -> &str {
            self.0
        }
        async fn encode(&self, texts: &[String]) -> Result<Vec<SparseVector>> {
            texts.iter().map(|t| term_vector(t)).collect()
        }
        async fn encode_query(&self, text: &str) -> Result<SparseVector> {
            term_vector(text)
        }
    }

    fn term_vector(text: &str) -> Result<SparseVector> {
        Ok(SparseVector::from_pairs(
            text.split_whitespace().filter_map(|w| {
                w.strip_prefix('t')?
                    .parse::<u32>()
                    .ok()
                    .map(|term| (term, 1.0))
            }),
        ))
    }

    /// Rows a, b, c with sparse vectors a:{1}, b:{1,2}, c:{9} and dense
    /// vectors that rank c first, then b, then a for the query "search text".
    fn sparse_store(source_id: &str) -> InMemoryStore {
        let mut store = InMemoryStore::new();
        store
            .upsert(&[
                doc("a", source_id, "a.txt", vec![0.0, 1.0, 0.0, 0.0]),
                doc("b", source_id, "b.txt", vec![0.7, 0.7, 0.0, 0.0]),
                doc("c", source_id, "c.txt", vec![1.0, 0.0, 0.0, 0.0]),
            ])
            .unwrap();
        store
            .set_sparse(
                "fake/sparse",
                &[
                    ("a".to_string(), term_vector("t1").unwrap()),
                    ("b".to_string(), term_vector("t1 t2").unwrap()),
                    ("c".to_string(), term_vector("t9").unwrap()),
                ],
            )
            .unwrap();
        store
    }

    #[tokio::test]
    async fn sparse_mode_ranks_by_inner_product_and_reports_sparse_rank() {
        let dir = tempfile::tempdir().unwrap();
        write_query_state(&dir, "source-1");
        let store = sparse_store("source-1");
        let encoder = FakeSparse("fake/sparse");

        let results = query(
            &dir.path().join(".minsync"),
            "t1 t2",
            5,
            &MockEmbedder,
            Some(&encoder),
            &store,
            None,
            QueryMode::Sparse,
        )
        .await
        .unwrap();

        let ids: Vec<&str> = results.iter().map(|r| r.doc_id.as_str()).collect();
        assert_eq!(ids, vec!["b", "a"], "c shares no term and must not appear");
        assert_eq!(results[0].mode, "sparse");
        assert_eq!(results[0].sparse_rank, Some(1));
        assert_eq!(results[0].vector_rank, None);
        assert_eq!(results[0].score, 2.0);
        let json = serde_json::to_value(&results[0]).unwrap();
        assert_eq!(json["sparse_rank"], 1);
        assert!(json.get("bm25_rank").is_none());
        assert!(json.get("vector_rank").is_none());
    }

    #[tokio::test]
    async fn sparse_mode_never_calls_the_dense_embedder() {
        struct Forbidden;
        #[async_trait]
        impl Embedder for Forbidden {
            fn id(&self) -> &str {
                "forbidden"
            }
            async fn embed(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>> {
                panic!("sparse search must not embed");
            }
        }
        let dir = tempfile::tempdir().unwrap();
        write_query_state(&dir, "source-1");
        let store = sparse_store("source-1");

        let results = query(
            &dir.path().join(".minsync"),
            "t1",
            5,
            &Forbidden,
            Some(&FakeSparse("fake/sparse")),
            &store,
            None,
            QueryMode::Sparse,
        )
        .await
        .unwrap();

        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn hybrid_fuses_both_rankings_and_credits_each_source() {
        let dir = tempfile::tempdir().unwrap();
        write_query_state(&dir, "source-1");
        let store = sparse_store("source-1");

        // Dense query "search text" = [1,0,0,0]: ranks c, b, a.
        // Sparse query "t1 t2": ranks b, a.  b is in both near the top.
        let results = query(
            &dir.path().join(".minsync"),
            "search text",
            3,
            &MockEmbedder,
            Some(&TextToTerms),
            &store,
            None,
            QueryMode::Hybrid,
        )
        .await
        .unwrap();

        let by_id: std::collections::HashMap<&str, &QueryResult> =
            results.iter().map(|r| (r.doc_id.as_str(), r)).collect();
        assert_eq!(by_id["c"].vector_rank, Some(1));
        assert_eq!(by_id["c"].sparse_rank, None);
        assert_eq!(by_id["b"].vector_rank, Some(2));
        assert_eq!(by_id["b"].sparse_rank, Some(1));
        assert_eq!(by_id["a"].sparse_rank, Some(2));
        assert_eq!(results[0].doc_id, "b", "present in both lists wins");
        assert!(results.iter().all(|r| r.mode == "hybrid"));
        assert!(results.windows(2).all(|w| w[0].score >= w[1].score));
    }

    /// Maps the dense query text onto fixed sparse terms.
    struct TextToTerms;

    #[async_trait]
    impl SparseEncoder for TextToTerms {
        fn id(&self) -> &str {
            "fake/sparse"
        }
        async fn encode(&self, _texts: &[String]) -> Result<Vec<SparseVector>> {
            Ok(Vec::new())
        }
        async fn encode_query(&self, _text: &str) -> Result<SparseVector> {
            term_vector("t1 t2")
        }
    }

    #[tokio::test]
    async fn sparse_search_without_an_encoder_is_an_error_not_a_fallback() {
        let dir = tempfile::tempdir().unwrap();
        write_query_state(&dir, "source-1");
        let store = sparse_store("source-1");

        for mode in [QueryMode::Sparse, QueryMode::Hybrid] {
            let error = query(
                &dir.path().join(".minsync"),
                "t1",
                5,
                &MockEmbedder,
                None,
                &store,
                None,
                mode,
            )
            .await
            .unwrap_err();
            assert!(matches!(error, MinSyncError::Config(_)), "{error}");
        }
    }

    #[tokio::test]
    async fn an_index_built_by_another_model_or_none_is_refused_with_the_fix() {
        let dir = tempfile::tempdir().unwrap();
        write_query_state(&dir, "source-1");
        let store = sparse_store("source-1");

        let other = query(
            &dir.path().join(".minsync"),
            "t1",
            5,
            &MockEmbedder,
            Some(&FakeSparse("new/model")),
            &store,
            None,
            QueryMode::Sparse,
        )
        .await
        .unwrap_err();
        assert!(matches!(other, MinSyncError::SparseNotReady(_)));
        let message = other.to_string();
        assert!(
            message.contains("fake/sparse") && message.contains("new/model"),
            "{message}"
        );
        assert!(message.contains("minsync sync"), "{message}");

        // A cursor with no sparse model at all (a BM25-era index).
        let minsync_dir = dir.path().join(".minsync");
        let mut cursor = Cursor::load(&minsync_dir.join("cursor.json")).unwrap();
        cursor.sparse_id = None;
        cursor.save(&minsync_dir.join("cursor.json")).unwrap();
        let none = query(
            &minsync_dir,
            "t1",
            5,
            &MockEmbedder,
            Some(&FakeSparse("fake/sparse")),
            &store,
            None,
            QueryMode::Hybrid,
        )
        .await
        .unwrap_err();
        assert!(matches!(none, MinSyncError::SparseNotReady(_)), "{none}");
        assert!(none.to_string().contains("minsync sync"));
    }

    #[tokio::test]
    async fn vector_mode_does_not_need_the_sparse_index() {
        let dir = tempfile::tempdir().unwrap();
        write_query_state(&dir, "source-1");
        let minsync_dir = dir.path().join(".minsync");
        let mut cursor = Cursor::load(&minsync_dir.join("cursor.json")).unwrap();
        cursor.sparse_id = None;
        cursor.save(&minsync_dir.join("cursor.json")).unwrap();
        let store = sparse_store("source-1");

        let results = query(
            &minsync_dir,
            "search text",
            3,
            &MockEmbedder,
            None,
            &store,
            None,
            QueryMode::Vector,
        )
        .await
        .unwrap();

        assert_eq!(results.len(), 3);
    }

    #[tokio::test]
    async fn queries_only_see_rows_of_the_cursor_source() {
        let dir = tempfile::tempdir().unwrap();
        write_query_state(&dir, "source-1");
        let mut store = sparse_store("other-source");
        store
            .upsert(&[doc(
                "mine",
                "source-1",
                "mine.txt",
                vec![1.0, 0.0, 0.0, 0.0],
            )])
            .unwrap();
        store
            .set_sparse(
                "fake/sparse",
                &[("mine".to_string(), term_vector("t1").unwrap())],
            )
            .unwrap();

        let results = query(
            &dir.path().join(".minsync"),
            "t1",
            5,
            &MockEmbedder,
            Some(&FakeSparse("fake/sparse")),
            &store,
            None,
            QueryMode::Sparse,
        )
        .await
        .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, "mine");
    }
}
