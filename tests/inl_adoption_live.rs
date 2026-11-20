//! Live check that a `native:` workspace adopts a real `inl serve` daemon and
//! gets the same vectors as the in-process engine — the shared im-not-llm
//! cache root and `(kind, model)` discovery, and the sparse prefix contract.
//!
//! Needs the real `inl` binary (env `INL_BIN`, default
//! `/tmp/inl-0.3.3/bin/inl`) and the weights. Run with an isolated HOME so the
//! daemon publishes under a throwaway root:
//!
//! ```sh
//! unset HF_HOME
//! INL_BIN=/tmp/inl-0.3.3/bin/inl INL_CACHE_DIR=/tmp/qa-models \
//!   cargo test --release --test inl_adoption_live -- --ignored --nocapture
//! ```
//!
//! The daemon's `HOME` is isolated inside the test; the test process keeps the
//! real one and discovers the daemon through the serve root it prints.

use minsync::config::{EmbedderConfig, SparseConfig, DEFAULT_SPARSE_MAX_LENGTH};
use minsync::embedder::Embedder;
use minsync::serve::{find_adoptable, EngineKind};
use minsync::sparse::{SparseEncoder, SparseVector};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

fn inl_bin() -> String {
    std::env::var("INL_BIN").unwrap_or_else(|_| "/tmp/inl-0.3.3/bin/inl".to_string())
}

/// One term of a `/embed_sparse` row, as the server serializes it.
#[derive(serde::Deserialize)]
struct Entry {
    index: u32,
    value: f32,
}

/// A spawned `inl serve` and the throwaway root it publishes under.
struct Daemon {
    child: Child,
    _home: tempfile::TempDir,
    serve_root: PathBuf,
}

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Spawn `inl serve` with an isolated `HOME`; return the process and the
/// serve root to look under.
fn spawn_inl(sparse: bool) -> Daemon {
    let home = tempfile::tempdir().unwrap();
    let serve_root = home.path().join(".cache/im-not-llm/serve");

    let mut command = Command::new(inl_bin());
    command
        .args(["serve", "--device", "metal", "--idle-timeout", "0"])
        .env("HOME", home.path())
        .env_remove("HF_HOME")
        .env_remove("FASTEMBED_CACHE_DIR")
        .stdout(Stdio::null())
        .stderr(Stdio::inherit());
    if sparse {
        command.arg("--sparse");
    }
    let child = command.spawn().expect("spawn inl serve");
    Daemon {
        child,
        _home: home,
        serve_root,
    }
}

/// The first instance directory under `serve_root` that answers `/health` OK.
async fn wait_ready(serve_root: &Path) -> String {
    let client = reqwest::Client::new();
    let deadline = Instant::now() + Duration::from_secs(300);
    loop {
        if let Ok(entries) = std::fs::read_dir(serve_root) {
            for entry in entries.flatten() {
                let dir = entry.path();
                let Ok(port) = std::fs::read_to_string(dir.join("port")) else {
                    continue;
                };
                let Ok(port) = port.trim().parse::<u16>() else {
                    continue;
                };
                if let Ok(response) = client
                    .get(format!("http://127.0.0.1:{port}/health"))
                    .send()
                    .await
                {
                    if response.status().is_success() {
                        if let Ok(health) = response.json::<serde_json::Value>().await {
                            if health["status"] == "ok" {
                                return format!("http://127.0.0.1:{port}");
                            }
                        }
                    }
                }
            }
        }
        assert!(
            Instant::now() < deadline,
            "inl serve did not become ready under {}",
            serve_root.display()
        );
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm(a) * norm(b))
}

/// Max absolute weight difference and one-sided term count, as `sparse_live`.
fn compare(ours: &SparseVector, theirs: &SparseVector) -> (f32, usize) {
    let (mut i, mut j) = (0, 0);
    let (mut max_diff, mut mismatched) = (0f32, 0usize);
    while i < ours.len() || j < theirs.len() {
        let (a, b) = (ours.indices().get(i), theirs.indices().get(j));
        match (a, b) {
            (Some(&x), Some(&y)) if x == y => {
                max_diff = max_diff.max((ours.values()[i] - theirs.values()[j]).abs());
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
                max_diff = max_diff.max(theirs.values()[j]);
                mismatched += 1;
                j += 1;
            }
        }
    }
    (max_diff, mismatched)
}

fn dense_settings() -> EmbedderConfig {
    let mut embedder = minsync::config::Config::default_for("live").embedder;
    embedder.id = "native:Qwen/Qwen3-Embedding-0.6B".to_string();
    embedder.device = Some("metal".to_string());
    embedder
}

fn sparse_settings() -> SparseConfig {
    SparseConfig {
        id: "native:nlpai-lab/KURE-Sparse-nano".to_string(),
        device: Some("metal".to_string()),
        dtype: Some("f32".to_string()),
        ..SparseConfig::default()
    }
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: loads Qwen3-Embedding-0.6B on Metal and spawns the real inl binary"]
async fn a_running_inl_serve_is_adopted_and_matches_in_process_dense_vectors() {
    let daemon = spawn_inl(false);
    let url = wait_ready(&daemon.serve_root).await;

    let found = find_adoptable(
        &daemon.serve_root,
        "Qwen/Qwen3-Embedding-0.6B",
        EngineKind::Dense,
        2048,
        Some("metal"),
        None,
    )
    .await
    .expect("the inl dense daemon is adopted by (kind, model)");
    assert_eq!(found.health.url, url);
    assert_eq!(found.health.model_id, "Qwen/Qwen3-Embedding-0.6B");

    // A pinned device that the daemon does not run is not adopted.
    assert!(find_adoptable(
        &daemon.serve_root,
        "Qwen/Qwen3-Embedding-0.6B",
        EngineKind::Dense,
        2048,
        Some("cpu"),
        None
    )
    .await
    .is_none());

    let texts = vec![
        "MinSync indexes text files incrementally.".to_string(),
        "벡터 검색은 의미가 비슷한 문서를 찾는다.".to_string(),
    ];
    let local = minsync::embedder::native::NativeEmbedder::from_config(&dense_settings()).unwrap();
    let expected = local.embed(&texts).await.unwrap();

    let served: Vec<Vec<f32>> = reqwest::Client::new()
        .post(format!("{url}/embed"))
        .json(&serde_json::json!({ "inputs": texts, "truncate": true }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(served.len(), expected.len());
    for (row, (a, b)) in expected.iter().zip(&served).enumerate() {
        let c = cosine(a, b);
        assert!(c > 0.999_9, "row {row}: cosine {c}");
    }
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: loads KURE-Sparse-nano on Metal and spawns the real inl binary"]
async fn a_running_inl_serve_sparse_is_adopted_and_matches_in_process_vectors() {
    let daemon = spawn_inl(true);
    let url = wait_ready(&daemon.serve_root).await;

    let found = find_adoptable(
        &daemon.serve_root,
        "nlpai-lab/KURE-Sparse-nano",
        EngineKind::Sparse,
        DEFAULT_SPARSE_MAX_LENGTH,
        Some("metal"),
        None,
    )
    .await
    .expect("the inl sparse daemon is adopted by (kind, model)");
    assert_eq!(found.health.url, url);
    assert_eq!(found.health.kind, "sparse");

    let encoder =
        minsync::sparse::native::NativeSparseEncoder::from_config(&sparse_settings()).unwrap();
    let (query_prefix, passage_prefix) = ("[Q] ", "[D] ");

    // What the served client sends: prefixed text, no `input_type`. The server
    // must add nothing (its `[Q]`/`[D]` defaults only apply with `input_type`),
    // so the adopted vectors equal the in-process ones.
    let passage = "MinSync는 텍스트 파일을 증분 인덱싱한다.";
    let query = "증분 인덱싱";

    let local_passages = encoder.encode(&[passage.to_string()]).await.unwrap();
    let remote_passages: Vec<Vec<Entry>> = reqwest::Client::new()
        .post(format!("{url}/embed_sparse"))
        .json(&serde_json::json!({
            "inputs": [format!("{passage_prefix}{passage}")],
            "truncate": true
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let remote = SparseVector::from_pairs(
        remote_passages[0]
            .iter()
            .map(|entry| (entry.index, entry.value)),
    );
    let (diff, mismatched) = compare(&local_passages[0], &remote);
    println!("passage: max diff {diff:.2e}, {mismatched} one-sided terms");
    assert!(diff < 2e-3, "passage weights diverge by {diff}");

    let local_query = encoder.encode_query(query).await.unwrap();
    let remote_query: Vec<Vec<Entry>> = reqwest::Client::new()
        .post(format!("{url}/embed_sparse"))
        .json(&serde_json::json!({
            "inputs": [format!("{query_prefix}{query}")],
            "truncate": true
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let remote_query = SparseVector::from_pairs(
        remote_query[0]
            .iter()
            .map(|entry| (entry.index, entry.value)),
    );
    let (diff, mismatched) = compare(&local_query, &remote_query);
    println!("query: max diff {diff:.2e}, {mismatched} one-sided terms");
    assert!(diff < 2e-3, "query weights diverge by {diff}");
}
