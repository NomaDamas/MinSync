//! Live check of the native EmbeddingGemma 2 embedder against the REAL
//! `google/embeddinggemma-2` checkpoint, compared with the reference
//! implementation (Sentence-Transformers, fp32, CPU; `golden_real.json`).
//!
//! Needs the model in the Hugging Face cache (or network): run with
//! `cargo test --release --test gemma2_live -- --ignored --nocapture`.
//! `MINSYNC_TEST_DEVICE=cpu|metal` picks the device (default cpu).

use minsync::config::Config;
use minsync::embedder::native::NativeEmbedder;
use minsync::embedder::Embedder;
use serde::Deserialize;

#[derive(Deserialize)]
struct Golden {
    texts: Vec<String>,
    emb: Vec<Vec<f32>>,
}

fn golden() -> Golden {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/embeddinggemma2/golden_real.json");
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn embedder(dtype: &str) -> NativeEmbedder {
    let mut config = Config::default_for("12345678-1234-4234-9234-123456789abc");
    config.embedder.id = "native:google/embeddinggemma-2".to_string();
    config.embedder.device =
        Some(std::env::var("MINSYNC_TEST_DEVICE").unwrap_or_else(|_| "cpu".into()));
    config.embedder.dtype = Some(dtype.to_string());
    NativeEmbedder::from_config(&config.embedder).unwrap()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm(a) * norm(b))
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: loads native:google/embeddinggemma-2; run with --ignored"]
async fn matches_the_sentence_transformers_reference_in_f32() {
    let golden = golden();
    let model = embedder("f32");
    let identity = model.identity().await.unwrap();
    assert_eq!(identity.dimension, 768);
    assert_eq!(identity.dtype, "f32");

    // Every text carries its own task prefix, so no prefix is configured. Two
    // of them exceed the 512-token sliding radius, and one the 2048 context.
    let batched = model.embed(&golden.texts).await.unwrap();
    for (row, expected) in golden.emb.iter().enumerate() {
        let similarity = cosine(&batched[row], expected);
        assert!(
            similarity > 0.9999,
            "batched row {row}: cosine {similarity}"
        );
        let alone = model.embed_single(&golden.texts[row]).await.unwrap();
        let similarity = cosine(&alone, expected);
        assert!(similarity > 0.9999, "alone row {row}: cosine {similarity}");
    }
}
