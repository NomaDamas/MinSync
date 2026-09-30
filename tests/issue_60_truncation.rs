use minsync::chunker::recursive::RecursiveChunker;
use minsync::embedder::Embedder;
use minsync::error::Result;
use minsync::sync::MinSync;
use minsync::types::SyncResult;
use minsync::vectorstore::memory::InMemoryStore;
use std::io::Write;
use std::sync::{Arc, Mutex};
use tracing_subscriber::fmt::MakeWriter;

struct TruncatingMockEmbedder;

#[async_trait::async_trait]
impl Embedder for TruncatingMockEmbedder {
    fn id(&self) -> &str {
        "mock-truncating"
    }

    async fn count_truncated(&self, texts: &[String]) -> Result<usize> {
        Ok(usize::from(!texts.is_empty()))
    }

    fn max_length(&self) -> Option<usize> {
        Some(2048)
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![0.0; 8]).collect())
    }
}

#[derive(Clone)]
struct SharedWriter(Arc<Mutex<Vec<u8>>>);

impl Write for SharedWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0
            .lock()
            .expect("lock test writer")
            .extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> MakeWriter<'a> for SharedWriter {
    type Writer = Self;

    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

#[test]
fn default_sync_result_serializes_zero_chunks_truncated() {
    let result = SyncResult {
        files_processed: 0,
        chunks_added: 0,
        chunks_updated: 0,
        chunks_deleted: 0,
        chunks_truncated: 0,
        dry_run: false,
        already_up_to_date: false,
        initial_sync: false,
        files_processed_paths: Vec::new(),
        files_added: 0,
        files_modified: 0,
        files_deleted: 0,
        elapsed_seconds: 0.0,
        embedding_api_calls: 0,
        embedded_texts: 0,
        estimated_tokens: 0,
        files_checked: 0,
        freshness_check_only: false,
        query_ready: false,
    };

    let json = serde_json::to_string(&result).expect("serialize default sync result");
    assert!(json.contains(r#""chunks_truncated":0"#));
}

#[tokio::test]
async fn sync_counts_truncated_chunks_and_warns_once_per_file() {
    let root = tempfile::tempdir().expect("create sync fixture");
    let sync = MinSync::new(root.path().to_path_buf());
    sync.init(false, "tei:test", "recursive")
        .expect("initialize sync fixture");
    std::fs::write(root.path().join("truncated.md"), "content").expect("write sync fixture");
    let chunker = RecursiveChunker::new(4096);
    let embedder = TruncatingMockEmbedder;
    let mut store = InMemoryStore::new();
    let log = Arc::new(Mutex::new(Vec::new()));
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .with_ansi(false)
        .with_writer(SharedWriter(log.clone()))
        .finish();

    let _guard = tracing::subscriber::set_default(subscriber);
    let result = sync
        .sync(&chunker, &embedder, &mut store, true, false, false)
        .await
        .expect("sync succeeds");

    let logs =
        String::from_utf8(log.lock().expect("lock test log").clone()).expect("logs are utf8");
    assert_eq!(result.chunks_truncated, 1);
    assert_eq!(logs.matches("hit the 2048-token limit").count(), 1);
    assert!(logs.contains("truncated.md"));
}
