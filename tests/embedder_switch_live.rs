//! Live check of `sync --reembed` on REAL checkpoints: an index built by
//! Qwen3-Embedding-0.6B (1024 dimensions) is switched to EmbeddingGemma 2
//! (768 dimensions), through the real CLI, in-process on the local device.
//!
//! Needs both models in the Hugging Face cache (or network access) and takes
//! a minute on Metal: run with
//! `cargo test --release --test embedder_switch_live -- --ignored --nocapture`.

use assert_cmd::Command;
use minsync::config::Config;
use serde_json::Value;
use std::path::Path;

mod common;

const QWEN: &str = "native:Qwen/Qwen3-Embedding-0.6B";
const GEMMA: &str = "native:google/embeddinggemma-2";

fn minsync(root: &Path) -> Command {
    let mut command = Command::cargo_bin("minsync").expect("find minsync binary");
    command.current_dir(root).env("MINSYNC_SERVER", "off");
    command
}

fn json(root: &Path, args: &[&str]) -> Value {
    let output = minsync(root).args(args).output().expect("run minsync");
    assert!(
        output.status.success(),
        "minsync {args:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("json output")
}

fn paths(hits: &Value) -> Vec<String> {
    hits.as_array()
        .expect("array of hits")
        .iter()
        .map(|hit| hit["path"].as_str().unwrap().to_string())
        .collect()
}

#[test]
#[ignore = "live: loads Qwen3-Embedding-0.6B and EmbeddingGemma 2; run with --ignored"]
fn a_real_qwen3_index_is_rebuilt_by_embeddinggemma_2_and_answers_queries() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    minsync(root)
        .args(["init", "--embedder", QWEN])
        .assert()
        .success();
    let config_path = root.join(".minsync/config.toml");
    let mut config = Config::load(&config_path).unwrap();
    config.embedder.max_length = Some(512);
    common::use_tiny_sparse(&mut config);
    config.save(&config_path).unwrap();

    let docs = [
        ("refund.md", "# 환불 정책\n\n구매 증빙 인증서를 제출한 고객은 14일 이내에 전액 환불을 받을 수 있습니다.\n"),
        ("travel.md", "# 출장 규정\n\n해외 출장비는 출장 종료 후 5영업일 안에 정산해야 합니다.\n"),
        ("security.md", "# 보안 수칙\n\n모든 직원은 분기마다 비밀번호를 변경해야 합니다.\n"),
        ("revenue.md", "# Revenue\n\nThird quarter revenue grew twenty three percent over the previous quarter.\n"),
        ("aurora.md", "# Aurora\n\nThe northern lights are caused by charged particles from the sun hitting the atmosphere.\n"),
    ];
    for (name, text) in docs {
        std::fs::write(root.join(name), text).unwrap();
    }
    let first = json(root, &["sync", "--format", "json"]);
    assert!(first["chunks_added"].as_u64().unwrap() >= 5, "{first}");
    let sparse_before = paths(&json(
        root,
        &[
            "query",
            "비밀번호 변경",
            "--mode",
            "sparse",
            "--format",
            "json",
        ],
    ));

    // Switch to EmbeddingGemma 2 exactly as `init --embedder` would write it.
    let mut config = Config::load(&config_path).unwrap();
    config.embedder.id = GEMMA.to_string();
    config.embedder.dtype = Some("f32".to_string());
    config.embedder.query_prefix = Some(minsync::config::EMBEDDINGGEMMA_QUERY_PREFIX.to_string());
    config.embedder.passage_prefix =
        Some(minsync::config::EMBEDDINGGEMMA_PASSAGE_PREFIX.to_string());
    let mut options = toml::value::Table::new();
    options.insert("dimension".into(), toml::Value::Integer(768));
    config.vectorstore.options = toml::Value::Table(options);
    config.save(&config_path).unwrap();

    // Refused until rebuilt, with the cause and the remedy.
    let refused = minsync(root).args(["sync"]).output().unwrap();
    assert_eq!(refused.status.code(), Some(8));
    let message = String::from_utf8_lossy(&refused.stderr);
    assert!(
        message.contains("Qwen/Qwen3-Embedding-0.6B")
            && message.contains("google/embeddinggemma-2")
            && message.contains("--reembed"),
        "{message}"
    );

    let rebuilt = json(root, &["sync", "--reembed", "--format", "json"]);
    assert!(rebuilt["reembedded"].as_u64().unwrap() >= 5, "{rebuilt}");
    assert_eq!(rebuilt["sparse_encoded"], 0, "{rebuilt}");

    // The vectors are Gemma 2's now: queries land on the right documents.
    for (question, expected) in [
        ("환불 받으려면 어떤 인증서가 필요해?", "refund.md"),
        ("What causes the northern lights?", "aurora.md"),
        ("비밀번호는 얼마나 자주 바꿔야 해?", "security.md"),
    ] {
        let hits = paths(&json(
            root,
            &[
                "query", question, "--mode", "vector", "--k", "1", "--format", "json",
            ],
        ));
        assert_eq!(hits, vec![expected.to_string()], "{question}");
    }

    // Sparse search is exactly what it was.
    let sparse_after = paths(&json(
        root,
        &[
            "query",
            "비밀번호 변경",
            "--mode",
            "sparse",
            "--format",
            "json",
        ],
    ));
    assert_eq!(sparse_after, sparse_before);

    // And back to a normal incremental state.
    let again = json(root, &["sync", "--format", "json"]);
    assert_eq!(again["already_up_to_date"], true, "{again}");
}
