//! CLI-level contract for the `line-cdc` chunker (#131): sized to the embedder,
//! a remote embedder's context guard never has to cut a chunk, and every input
//! the embedder receives carries its heading context line.

use assert_cmd::Command;
use minsync::config::Config;
use serde_json::Value;
use std::path::Path;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

mod common;

const DIMENSION: usize = 4;
/// A small remote context so that many chunks press against the budget.
const MAX_LENGTH: usize = 256;

struct FixedTei;

impl Respond for FixedTei {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let body: Value = serde_json::from_slice(&request.body).expect("TEI request json");
        let inputs = body["inputs"].as_array().expect("inputs array");
        let vectors: Vec<Vec<f32>> = inputs.iter().map(|_| vec![0.5; DIMENSION]).collect();
        ResponseTemplate::new(200).set_body_json(vectors)
    }
}

fn minsync(root: &Path) -> Command {
    let mut command = Command::cargo_bin("minsync").expect("find minsync binary");
    command.current_dir(root).env("MINSYNC_SERVER", "off");
    command
}

fn setup(root: &Path, base_url: &str) {
    minsync(root)
        .args([
            "init",
            "--embedder",
            "tei:test/model",
            "--chunker",
            "line-cdc",
        ])
        .assert()
        .success();
    let config_path = root.join(".minsync/config.toml");
    let mut config = Config::load(&config_path).expect("load config");
    config.embedder.base_url = Some(base_url.to_string());
    config.embedder.max_retries = 0;
    config.embedder.max_length = Some(MAX_LENGTH);
    config.chunker.options.heading_context = true;
    let mut options = toml::value::Table::new();
    options.insert("dimension".into(), toml::Value::Integer(DIMENSION as i64));
    config.vectorstore.options = toml::Value::Table(options);
    common::use_tiny_sparse(&mut config);
    config.save(&config_path).expect("save config");
}

/// Hangul-heavy prose and a pipe table under nested headings. Hangul is where
/// tokenizers disagree most, so this is where a chunk sized for one count and
/// cut by another would show.
fn report() -> String {
    let mut text = String::from("# 2026년 예산 보고서\n\n");
    for section in 0..6 {
        text.push_str(&format!("## {section}. 세부 편성 내역\n\n"));
        // Short lines wrapped mid-sentence: chunks fill up to the budget.
        for line in 0..60 {
            text.push_str(&format!(
                "{section}장 {line}행 인건비 편성액 검토 의견과 내부 결재\n"
            ));
        }
        text.push_str("\n| 비목 | 편성액 | 비고 |\n| --- | --- | --- |\n");
        for row in 0..20 {
            text.push_str(&format!(
                "| 연구활동비 {section}-{row} | {} | 내부 |\n",
                13_166 + row
            ));
        }
        text.push('\n');
    }
    text
}

#[tokio::test(flavor = "multi_thread")]
async fn line_cdc_chunks_fit_a_remote_embedder_without_being_cut() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embed"))
        .respond_with(FixedTei)
        .mount(&server)
        .await;
    let dir = tempfile::tempdir().unwrap();
    setup(dir.path(), &server.uri());
    std::fs::write(dir.path().join("report.md"), report()).unwrap();

    let root = dir.path().to_path_buf();
    let output = tokio::task::spawn_blocking(move || {
        minsync(&root)
            .args(["sync", "--format", "json"])
            .output()
            .expect("run minsync")
    })
    .await
    .expect("join");

    assert!(output.status.success(), "{output:?}");
    let result: Value = serde_json::from_slice(&output.stdout).expect("sync json");
    assert_eq!(result["chunks_truncated"], 0, "{result}");
    let mut inputs = Vec::new();
    for request in server.received_requests().await.unwrap() {
        let body: Value = serde_json::from_slice(&request.body).unwrap();
        for input in body["inputs"].as_array().unwrap() {
            inputs.push(input.as_str().unwrap().to_string());
        }
    }
    assert!(inputs.len() > 10, "{} inputs", inputs.len());
    for input in &inputs {
        assert!(
            input.starts_with("report.md"),
            "no context line: {:?}",
            &input[..input.len().min(60)]
        );
    }
    assert!(
        inputs.iter().any(|input| input.starts_with("report.md > ")),
        "no input carries a heading trail"
    );
}
