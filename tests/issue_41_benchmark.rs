use async_trait::async_trait;
use minsync::chunker::chonkie::ChonkieChunker;
use minsync::cli::QueryMode;
use minsync::embedder::Embedder;
use minsync::error::Result;
use minsync::query::query;
use minsync::sync::MinSync;
use minsync::vectorstore::memory::InMemoryStore;
use minsync::vectorstore::{Document, VectorStore};
use std::time::Instant;

struct LocalEmbedder;

#[async_trait]
impl Embedder for LocalEmbedder {
    fn id(&self) -> &str {
        "local:issue-41"
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|text| vector_for(text)).collect())
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        Ok(vector_for(text))
    }
}

fn vector_for(text: &str) -> Vec<f32> {
    let digest = minsync::id::content_hash(text);
    digest
        .as_bytes()
        .iter()
        .take(8)
        .map(|byte| *byte as f32 / 255.0)
        .collect()
}

fn percentile(samples: &mut [f64], percentile: f64) -> f64 {
    samples.sort_by(f64::total_cmp);
    let index = ((samples.len() - 1) as f64 * percentile).round() as usize;
    samples[index]
}

async fn measure_sync(
    sync: &MinSync,
    chunker: &ChonkieChunker,
    embedder: &LocalEmbedder,
    store: &mut InMemoryStore,
    full: bool,
) -> f64 {
    let started = Instant::now();
    sync.sync(chunker, embedder, store, full, false, false)
        .await
        .expect("local benchmark sync succeeds");
    started.elapsed().as_secs_f64() * 1_000.0
}

fn seed_documents(store: &mut InMemoryStore, scale: usize, source_id: &str) {
    let documents: Vec<_> = (0..scale * 8)
        .map(|index| Document {
            id: format!("doc-{index}"),
            embedding: vector_for(&format!("document {index} selective")),
            text: format!("document {index} selective benchmark text"),
            source_id: source_id.to_string(),
            path: format!("doc-{index}.txt"),
            chunk_schema_id: "benchmark".to_string(),
            chunk_type: "text".to_string(),
            heading_path: String::new(),
            content_hash: format!("hash-{index}"),
            seen_token: "benchmark".to_string(),
        })
        .collect();
    store.upsert(&documents).expect("seed benchmark documents");
}

#[tokio::test]
async fn issue_41_local_benchmark_reports_sync_and_query_scaling() {
    let embedder = LocalEmbedder;
    let chunker = ChonkieChunker::new(128, "\n ");
    let mut rows = Vec::new();

    for scale in [1_usize, 2, 4, 8] {
        let root = tempfile::tempdir().expect("create benchmark root");
        for index in 0..scale * 8 {
            std::fs::write(
                root.path().join(format!("doc-{index}.txt")),
                format!("document {index} selective benchmark text"),
            )
            .expect("write benchmark document");
        }
        let sync = MinSync::new(root.path().to_path_buf());
        sync.init(false, "local:issue-41", "chonkie")
            .expect("initialize benchmark workspace");
        let mut store = InMemoryStore::new();

        let full_ms = measure_sync(&sync, &chunker, &embedder, &mut store, true).await;
        let noop_ms = measure_sync(&sync, &chunker, &embedder, &mut store, false).await;
        std::fs::write(
            root.path().join("doc-0.txt"),
            "changed selective benchmark text",
        )
        .expect("modify benchmark document");
        let changed_ms = measure_sync(&sync, &chunker, &embedder, &mut store, false).await;

        let source_id = minsync::state::Cursor::load(&root.path().join(".minsync/cursor.json"))
            .expect("load benchmark cursor")
            .source_id;
        seed_documents(&mut store, scale, &source_id);
        let mut query_samples = [Vec::new(), Vec::new(), Vec::new()];
        for _ in 0..5 {
            let started = Instant::now();
            query(
                &root.path().join(".minsync"),
                "selective",
                5,
                &embedder,
                &store,
                None,
                QueryMode::Bm25,
            )
            .await
            .expect("BM25 benchmark query succeeds");
            query_samples[0].push(started.elapsed().as_secs_f64() * 1_000.0);

            let started = Instant::now();
            query(
                &root.path().join(".minsync"),
                "selective",
                5,
                &embedder,
                &store,
                None,
                QueryMode::Vector,
            )
            .await
            .expect("vector benchmark query succeeds");
            query_samples[1].push(started.elapsed().as_secs_f64() * 1_000.0);

            let started = Instant::now();
            query(
                &root.path().join(".minsync"),
                "selective",
                5,
                &embedder,
                &store,
                None,
                QueryMode::Hybrid,
            )
            .await
            .expect("hybrid benchmark query succeeds");
            query_samples[2].push(started.elapsed().as_secs_f64() * 1_000.0);
        }

        rows.push(serde_json::json!({
            "scale": scale,
            "documents": scale * 8,
            "sync_ms": {
                "full": full_ms,
                "unchanged": noop_ms,
                "changed_file": changed_ms
            },
            "query_ms_p50": {
                "bm25": percentile(&mut query_samples[0], 0.50),
                "vector": percentile(&mut query_samples[1], 0.50),
                "hybrid": percentile(&mut query_samples[2], 0.50)
            },
            "query_ms_p95": {
                "bm25": percentile(&mut query_samples[0], 0.95),
                "vector": percentile(&mut query_samples[1], 0.95),
                "hybrid": percentile(&mut query_samples[2], 0.95)
            },
            "backend": "in_memory",
            "index_state": "not_applicable"
        }));
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&rows).expect("serialize benchmark report")
    );
}
