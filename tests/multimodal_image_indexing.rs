//! Integration tests for direct multimodal image indexing, incremental updates,
//! query retrieval, and verification.

use async_trait::async_trait;
use minsync::chunker::create_chunker;
use minsync::cli::QueryMode;
use minsync::config::EMBEDDINGGEMMA2_MULTIMODAL_ID;
use minsync::embedder::{Embedder, ImageInput};
use minsync::error::Result;
use minsync::query::query;
use minsync::sync::MinSync;
use minsync::vectorstore::memory::InMemoryStore;
use minsync::vectorstore::VectorStore;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Deterministic fake multimodal embedder mapping images and text into one
/// 4-dimensional space:
/// - PNG starting with b"\x89PNGcat" -> [1.0, 0.0, 0.0, 0.0]
/// - PNG starting with b"\x89PNGdog" -> [0.0, 1.0, 0.0, 0.0]
/// - query/text containing "cat"     -> [1.0, 0.0, 0.0, 0.0]
/// - query/text containing "dog"     -> [0.0, 1.0, 0.0, 0.0]
/// - other text                      -> [0.0, 0.0, 1.0, 0.0]
struct FakeMultimodalEmbedder {
    image_calls: AtomicUsize,
    text_calls: AtomicUsize,
}

impl FakeMultimodalEmbedder {
    fn new() -> Self {
        Self {
            image_calls: AtomicUsize::new(0),
            text_calls: AtomicUsize::new(0),
        }
    }
}

#[async_trait]
impl Embedder for FakeMultimodalEmbedder {
    fn id(&self) -> &str {
        "google/embeddinggemma-2"
    }

    fn supports_images(&self) -> bool {
        true
    }

    fn is_deterministic(&self) -> bool {
        true
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.text_calls.fetch_add(1, Ordering::SeqCst);
        Ok(texts
            .iter()
            .map(|text| {
                let lower = text.to_ascii_lowercase();
                if lower.contains("cat") {
                    vec![1.0, 0.0, 0.0, 0.0]
                } else if lower.contains("dog") {
                    vec![0.0, 1.0, 0.0, 0.0]
                } else {
                    vec![0.0, 0.0, 1.0, 0.0]
                }
            })
            .collect())
    }

    async fn embed_images(&self, images: &[ImageInput]) -> Result<Vec<Vec<f32>>> {
        self.image_calls.fetch_add(1, Ordering::SeqCst);
        Ok(images
            .iter()
            .map(|img| {
                if img.bytes.windows(3).any(|w| w == b"cat") {
                    vec![1.0, 0.0, 0.0, 0.0]
                } else if img.bytes.windows(3).any(|w| w == b"dog") {
                    vec![0.0, 1.0, 0.0, 0.0]
                } else {
                    vec![0.0, 0.0, 0.0, 1.0]
                }
            })
            .collect())
    }
}

/// Text-only embedder: image files stay ignored (empty) just like before.
struct TextOnlyEmbedder;

#[async_trait]
impl Embedder for TextOnlyEmbedder {
    fn id(&self) -> &str {
        "text-only"
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![0.0, 0.0, 1.0, 0.0]).collect())
    }
}

#[tokio::test]
async fn multimodal_embedder_indexes_images_alongside_text_and_retrieves_by_text_query() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();

    std::fs::write(root.join("note.md"), "quarterly finance budget summary").unwrap();
    std::fs::write(root.join("kitty.png"), b"\x89PNG\r\n\x1a\ncat-pixels-v1").unwrap();
    std::fs::write(root.join("puppy.jpg"), b"\xff\xd8\xffdog-pixels-v1").unwrap();

    let ms = MinSync::new(root.to_path_buf());
    let mut config = ms
        .init(false, EMBEDDINGGEMMA2_MULTIMODAL_ID, "line-cdc")
        .unwrap();
    config.vectorstore.options["dimension"] = toml::Value::Integer(4);
    config.save(&ms.minsync_dir().join("config.toml")).unwrap();

    let chunker = create_chunker(&config).unwrap();
    let embedder = FakeMultimodalEmbedder::new();
    let mut store = InMemoryStore::new();

    let first = ms
        .sync(chunker.as_ref(), &embedder, &mut store, false, false, false)
        .await
        .unwrap();
    assert_eq!(first.files_processed, 3);
    assert_eq!(first.chunks_added, 3);
    assert_eq!(embedder.image_calls.load(Ordering::SeqCst), 1);
    assert_eq!(embedder.text_calls.load(Ordering::SeqCst), 1);

    // Semantic text query "find the cat photo" retrieves kitty.png at rank 1.
    let cat_hits = query(
        ms.minsync_dir(),
        "find the cat photo",
        3,
        &embedder,
        None,
        &store,
        None,
        QueryMode::Vector,
    )
    .await
    .unwrap();
    assert_eq!(cat_hits[0].path, "kitty.png");
    assert_eq!(cat_hits[0].chunk_type, "image");
    assert!(cat_hits[0].score > 0.99);

    // Query for dog retrieves puppy.jpg at rank 1.
    let dog_hits = query(
        ms.minsync_dir(),
        "show me the dog",
        3,
        &embedder,
        None,
        &store,
        None,
        QueryMode::Vector,
    )
    .await
    .unwrap();
    assert_eq!(dog_hits[0].path, "puppy.jpg");
    assert_eq!(dog_hits[0].chunk_type, "image");

    // Incremental sync with no changes does zero work.
    let noop = ms
        .sync(chunker.as_ref(), &embedder, &mut store, false, false, false)
        .await
        .unwrap();
    assert!(noop.already_up_to_date);
    assert_eq!(embedder.image_calls.load(Ordering::SeqCst), 1);

    // Modifying kitty.png re-embeds only the changed image; deleting puppy.jpg removes its row.
    std::fs::write(root.join("kitty.png"), b"\x89PNG\r\n\x1a\ndog-now-v2").unwrap();
    std::fs::remove_file(root.join("puppy.jpg")).unwrap();

    let updated = ms
        .sync(chunker.as_ref(), &embedder, &mut store, false, false, false)
        .await
        .unwrap();
    assert_eq!(updated.files_modified, 1);
    assert_eq!(updated.files_deleted, 1);
    assert_eq!(store.doc_count(), 2);

    let after_hits = query(
        ms.minsync_dir(),
        "dog",
        2,
        &embedder,
        None,
        &store,
        None,
        QueryMode::Vector,
    )
    .await
    .unwrap();
    assert_eq!(after_hits[0].path, "kitty.png");

    // Verify confirms expected image doc_ids match the store.
    let report = minsync::verify::verify(
        ms.minsync_dir(),
        root,
        chunker.as_ref(),
        &mut store,
        false,
        None,
    )
    .await
    .unwrap();
    assert!(
        report.basic_checks.get("sample_chunks_match") == Some(&true),
        "verify report: {report:?}"
    );
}

#[tokio::test]
async fn text_only_embedder_keeps_ignoring_binary_images() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();

    std::fs::write(root.join("note.md"), "hello world").unwrap();
    std::fs::write(root.join("photo.png"), b"\x89PNG\r\n\x1a\n\xff\xfe\xfd").unwrap();

    let ms = MinSync::new(root.to_path_buf());
    let mut config = ms.init(false, "native:Qwen/Qwen3-Embedding-0.6B", "line-cdc").unwrap();
    config.vectorstore.options["dimension"] = toml::Value::Integer(4);
    config.save(&ms.minsync_dir().join("config.toml")).unwrap();

    let chunker = create_chunker(&config).unwrap();
    let embedder = TextOnlyEmbedder;
    let mut store = InMemoryStore::new();

    let synced = ms
        .sync(chunker.as_ref(), &embedder, &mut store, false, false, false)
        .await
        .unwrap();
    assert_eq!(synced.files_processed, 2);
    // Only note.md produced a chunk; binary photo.png stayed empty under text-only embedder.
    assert_eq!(store.doc_count(), 1);
    assert_eq!(store.documents().unwrap()[0].path, "note.md");
}
