//! Live checks of the native sparse encoder against the REAL
//! `nlpai-lab/KURE-Sparse-nano` checkpoint, compared with the reference
//! implementation (Sentence-Transformers, fp32, CPU; `st_f32_golden.json`).
//!
//! Needs the model in the Hugging Face cache (or network): run with
//! `cargo test --release --test sparse_live -- --ignored --nocapture`.
//! `MINSYNC_TEST_DEVICE=cpu|metal` picks the device (default cpu).

use minsync::config::SparseConfig;
use minsync::sparse::native::NativeSparseEncoder;
use minsync::sparse::served::create_sparse_encoder;
use minsync::sparse::{SparseEncoder, SparseVector};
use serde::Deserialize;

#[derive(Deserialize)]
struct Golden {
    query: String,
    docs: Vec<String>,
    q: Raw,
    d: Vec<Raw>,
}

#[derive(Deserialize)]
struct Raw {
    indices: Vec<u32>,
    values: Vec<f32>,
}

fn golden() -> Golden {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/kure_sparse_nano/st_f32_golden.json");
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn settings() -> SparseConfig {
    SparseConfig {
        device: Some(std::env::var("MINSYNC_TEST_DEVICE").unwrap_or_else(|_| "cpu".into())),
        dtype: Some("f32".to_string()),
        ..SparseConfig::default()
    }
}

/// Largest absolute weight difference over the union of terms, and the number
/// of terms present in only one of the two vectors.
fn compare(ours: &SparseVector, theirs: &Raw) -> (f32, usize) {
    let mut max_diff = 0f32;
    let mut mismatched = 0;
    let (mut i, mut j) = (0, 0);
    while i < ours.len() || j < theirs.indices.len() {
        let (a, b) = (ours.indices().get(i), theirs.indices.get(j));
        match (a, b) {
            (Some(&x), Some(&y)) if x == y => {
                max_diff = max_diff.max((ours.values()[i] - theirs.values[j]).abs());
                i += 1;
                j += 1;
            }
            (Some(&x), Some(&y)) if x < y => {
                max_diff = max_diff.max(ours.values()[i]);
                mismatched += 1;
                i += 1;
            }
            (Some(_), None) => {
                max_diff = max_diff.max(ours.values()[i]);
                mismatched += 1;
                i += 1;
            }
            _ => {
                max_diff = max_diff.max(theirs.values[j]);
                mismatched += 1;
                j += 1;
            }
        }
    }
    (max_diff, mismatched)
}

fn to_vector(raw: &Raw) -> SparseVector {
    SparseVector::new(raw.indices.clone(), raw.values.clone()).unwrap()
}

#[tokio::test]
#[ignore = "live: loads the real nlpai-lab/KURE-Sparse-nano checkpoint"]
async fn native_encoder_matches_sentence_transformers_on_the_real_model() {
    let golden = golden();
    let encoder = NativeSparseEncoder::from_config(&settings()).unwrap();
    let identity = encoder.identity().await.unwrap();
    println!("identity: {identity:?}");
    assert_eq!(identity.dimension, 36282);

    let query = encoder.encode_query(&golden.query).await.unwrap();
    let docs = encoder.encode(&golden.docs).await.unwrap();

    // The model-card scores (dot product), and the reference's own.
    let reference_scores: Vec<f32> = golden
        .d
        .iter()
        .map(|d| to_vector(&golden.q).dot(&to_vector(d)))
        .collect();
    let our_scores: Vec<f32> = docs.iter().map(|d| query.dot(d)).collect();
    println!("reference {reference_scores:?}\nnative    {our_scores:?}");

    let (diff, mismatched) = compare(&query, &golden.q);
    println!("query: max weight diff {diff:.2e}, {mismatched} terms present in one side only");
    assert!(diff < 2e-3, "query weights diverge by {diff}");
    for (index, (ours, theirs)) in docs.iter().zip(&golden.d).enumerate() {
        let (diff, mismatched) = compare(ours, theirs);
        println!(
            "doc {index}: nnz {} (ref {}), max diff {diff:.2e}, {mismatched} one-sided terms",
            ours.len(),
            theirs.indices.len()
        );
        assert!(diff < 2e-3, "doc {index} weights diverge by {diff}");
    }
    for (index, (a, b)) in our_scores.iter().zip(&reference_scores).enumerate() {
        assert!(
            (a - b).abs() / b.max(1.0) < 5e-3,
            "score {index}: {a} vs {b}"
        );
    }
    // Ranking is what retrieval consumes.
    assert!(
        our_scores[0] > our_scores[1] && our_scores[1] > our_scores[2],
        "{our_scores:?}"
    );
}

#[tokio::test]
#[ignore = "live: loads the real nlpai-lab/KURE-Sparse-nano checkpoint"]
async fn batching_and_long_inputs_do_not_change_real_model_vectors() {
    let golden = golden();
    let encoder = NativeSparseEncoder::from_config(&settings()).unwrap();
    let alone: Vec<SparseVector> = {
        let mut out = Vec::new();
        for doc in &golden.docs {
            out.push(
                encoder
                    .encode(std::slice::from_ref(doc))
                    .await
                    .unwrap()
                    .remove(0),
            );
        }
        out
    };
    // Same docs batched together with a long filler that forces padding.
    let filler = "한국어 검색 모델은 긴 문서도 처리해야 한다. ".repeat(120);
    let mut inputs = vec![filler.clone()];
    inputs.extend(golden.docs.clone());
    let together = encoder.encode(&inputs).await.unwrap();
    for (index, doc) in alone.iter().enumerate() {
        let other = &together[index + 1];
        assert_eq!(
            doc.indices(),
            other.indices(),
            "doc {index} terms changed with padding"
        );
        for (a, b) in doc.values().iter().zip(other.values()) {
            assert!((a - b).abs() < 2e-3, "doc {index}: {a} vs {b}");
        }
    }
    assert!(!together[0].is_empty());
}

#[tokio::test]
#[ignore = "live: loads the real nlpai-lab/KURE-Sparse-nano checkpoint"]
async fn a_running_serve_sparse_gives_the_same_vectors_as_in_process() {
    use std::process::{Command, Stdio};
    let golden = golden();
    let state = tempfile::tempdir().unwrap();
    let workspace = tempfile::tempdir().unwrap();
    let bin = env!("CARGO_BIN_EXE_minsync");
    let mut child = Command::new(bin)
        .current_dir(workspace.path())
        .args([
            "--format",
            "json",
            "serve",
            "--sparse",
            "--port",
            "0",
            "--idle-timeout",
            "0",
            "--state-dir",
        ])
        .arg(state.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn minsync serve --sparse");
    let stdout = child.stdout.take().unwrap();
    let mut line = String::new();
    std::io::BufRead::read_line(&mut std::io::BufReader::new(stdout), &mut line).unwrap();
    let health: serde_json::Value = serde_json::from_str(&line).expect("serving event");
    println!("server: {health}");
    assert_eq!(health["event"], "serving");
    assert_eq!(health["kind"], "sparse");
    let url = health["url"].as_str().unwrap().to_string();

    // Speak TEI to it, exactly as a `tei:` client would.
    let remote = create_sparse_encoder(&SparseConfig {
        id: "tei:nlpai-lab/KURE-Sparse-nano".to_string(),
        base_url: Some(url),
        max_length: Some(2048),
        ..SparseConfig::default()
    })
    .unwrap();
    let query = remote.encode_query(&golden.query).await.unwrap();
    let docs = remote.encode(&golden.docs).await.unwrap();
    let (diff, _) = compare(&query, &golden.q);
    assert!(diff < 2e-3, "served query diverges by {diff}");
    for (index, (served, reference)) in docs.iter().zip(&golden.d).enumerate() {
        let (diff, _) = compare(served, reference);
        assert!(diff < 2e-3, "served doc {index} diverges by {diff}");
    }
    let _ = child.kill();
    let _ = child.wait();
}

#[tokio::test]
#[ignore = "live: loads the real checkpoint on the default device (Metal on macOS)"]
async fn half_precision_on_the_default_device_keeps_scores_and_ranking() {
    let golden = golden();
    let on_default = |dtype: &str| SparseConfig {
        device: Some("auto".to_string()),
        dtype: Some(dtype.to_string()),
        ..SparseConfig::default()
    };
    let reference: Vec<f32> = golden
        .d
        .iter()
        .map(|d| to_vector(&golden.q).dot(&to_vector(d)))
        .collect();
    for dtype in ["f16", "f32"] {
        let encoder = NativeSparseEncoder::from_config(&on_default(dtype)).unwrap();
        let identity = encoder.identity().await.unwrap();
        let query = encoder.encode_query(&golden.query).await.unwrap();
        let docs = encoder.encode(&golden.docs).await.unwrap();
        let scores: Vec<f32> = docs.iter().map(|d| query.dot(d)).collect();
        let (qdiff, qmismatch) = compare(&query, &golden.q);
        let relative: Vec<f32> = scores
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs() / b)
            .collect();
        println!(
            "{}/{}: query max weight diff {qdiff:.4} ({qmismatch} one-sided terms) | scores {scores:?} | rel err {relative:?}",
            identity.device, identity.dtype
        );
        assert!(scores[0] > scores[1] && scores[1] > scores[2], "{scores:?}");
        for r in relative {
            assert!(r < 0.02, "{dtype}: relative score error {r}");
        }
    }
}

#[tokio::test]
#[ignore = "live benchmark: prints throughput for cpu/f32, metal/f32, metal/f16"]
async fn throughput_by_device_and_dtype() {
    let unit = "세종대왕은 1443년에 훈민정음을 창제하고 1446년에 이를 반포하였다. 김치는 배추나 무를 소금에 절인 뒤 고춧가루와 젓갈을 넣어 발효시킨 음식이다. ";
    // 128 chunks of ~900 chars, the size MinSync's recursive chunker emits.
    let chunks: Vec<String> = (0..128)
        .map(|i| format!("{i} {}", unit.repeat(5 + i % 3)))
        .collect();
    for (device, dtype) in [("cpu", "f32"), ("metal", "f32"), ("metal", "f16")] {
        let settings = SparseConfig {
            device: Some(device.to_string()),
            dtype: Some(dtype.to_string()),
            ..SparseConfig::default()
        };
        let Ok(encoder) = NativeSparseEncoder::from_config(&settings) else {
            continue;
        };
        if encoder.identity().await.is_err() {
            println!("{device}/{dtype}: unavailable");
            continue;
        }
        encoder.encode(&chunks[..4]).await.unwrap(); // warm up
        let started = std::time::Instant::now();
        let vectors = encoder.encode(&chunks).await.unwrap();
        let seconds = started.elapsed().as_secs_f64();
        let nnz = vectors.iter().map(SparseVector::len).sum::<usize>() / vectors.len();
        println!(
            "{device}/{dtype}: {:>6.1} chunks/s ({seconds:.2}s for {}), mean nnz {nnz}",
            chunks.len() as f64 / seconds,
            chunks.len()
        );
    }
}
