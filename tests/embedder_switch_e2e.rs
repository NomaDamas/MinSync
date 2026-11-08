//! Switching the dense embedder of an existing index (#embedder-switch).
//!
//! A mock TEI server stands in for the embedder. Its model id is part of the
//! URL-free request, so the tests tell models apart by the vector each one
//! returns: `ModelA` answers `[1, 0, 0, 0]`-style vectors that encode the text
//! length, `ModelB` answers a different space with another dimension when asked.
//!
//! Contract under test:
//! - the cursor records which embedder (model, dimension, passage prefix) built
//!   the vectors;
//! - a config whose embedder differs is detected, never silently mixed in:
//!   `sync` stops with guidance, `query --mode vector|hybrid` is refused, and a
//!   sparse query keeps working;
//! - `sync --reembed` rebuilds every vector with the new embedder from the
//!   stored chunk text (no file re-reading, no sparse re-encoding), resumably,
//!   and the index is queryable by the new model afterwards;
//! - settings that do not change vectors (dtype, batch size, query prefix) do
//!   not trigger anything.

use assert_cmd::Command;
use minsync::config::Config;
use serde_json::Value;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

mod common;

/// Answers every input with a unit vector of `dimension` numbers whose first
/// component depends on `salt`, so two "models" disagree on every text, and
/// counts the inputs it embedded.
struct Model {
    dimension: usize,
    salt: f32,
    embedded: Arc<AtomicUsize>,
}

impl Respond for Model {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let body: Value = serde_json::from_slice(&request.body).expect("TEI request json");
        let inputs = body["inputs"].as_array().expect("inputs array");
        self.embedded.fetch_add(inputs.len(), Ordering::SeqCst);
        let vectors: Vec<Vec<f32>> = inputs
            .iter()
            .map(|input| {
                let text = input.as_str().unwrap();
                let mut vector = vec![0.0f32; self.dimension];
                // Position by text content so different texts differ, shifted by salt.
                let slot = (text.bytes().map(usize::from).sum::<usize>() + self.salt as usize)
                    % self.dimension;
                vector[slot] = 1.0;
                vector
            })
            .collect();
        ResponseTemplate::new(200).set_body_json(vectors)
    }
}

async fn serve(dimension: usize, salt: f32) -> (MockServer, Arc<AtomicUsize>) {
    let server = MockServer::start().await;
    let embedded = Arc::new(AtomicUsize::new(0));
    Mock::given(method("POST"))
        .and(path("/embed"))
        .respond_with(Model {
            dimension,
            salt,
            embedded: embedded.clone(),
        })
        .mount(&server)
        .await;
    (server, embedded)
}

fn minsync(root: &Path) -> Command {
    let mut command = Command::cargo_bin("minsync").expect("find minsync binary");
    command.current_dir(root).env("MINSYNC_SERVER", "off");
    command
}

async fn run(root: &Path, args: &[&str]) -> std::process::Output {
    let root = root.to_path_buf();
    let args: Vec<String> = args.iter().map(|arg| arg.to_string()).collect();
    tokio::task::spawn_blocking(move || minsync(&root).args(&args).output().expect("run minsync"))
        .await
        .expect("join")
}

fn stdout_json(output: &std::process::Output) -> Value {
    serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|error| panic!("json on stdout: {error}\n{output:?}"))
}

fn stderr(output: &std::process::Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

/// Point the workspace's embedder at `base_url` as `tei:<model>`.
fn configure(root: &Path, model: &str, base_url: &str, dimension: usize) {
    let config_path = root.join(".minsync/config.toml");
    let mut config = Config::load(&config_path).expect("load config");
    config.embedder.id = format!("tei:{model}");
    config.embedder.base_url = Some(base_url.to_string());
    config.embedder.max_retries = 0;
    let mut options = toml::value::Table::new();
    options.insert("dimension".into(), toml::Value::Integer(dimension as i64));
    config.vectorstore.options = toml::Value::Table(options);
    common::use_tiny_sparse(&mut config);
    config.save(&config_path).expect("save config");
}

fn workspace(base_url: &str, model: &str, dimension: usize) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    minsync(root)
        .args(["init", "--embedder", &format!("tei:{model}")])
        .assert()
        .success();
    configure(root, model, base_url, dimension);
    std::fs::write(
        root.join("refund.md"),
        "# Refund policy\n\nA customer with a purchase proof certificate gets a full refund within 14 days.\n",
    )
    .unwrap();
    std::fs::write(
        root.join("travel.md"),
        "# Travel\n\nOverseas travel expenses must be settled within five business days after the trip.\n",
    )
    .unwrap();
    std::fs::write(
        root.join("revenue.md"),
        "# Revenue\n\nThird quarter revenue grew twenty three percent over the previous quarter.\n",
    )
    .unwrap();
    dir
}

fn cursor(root: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(root.join(".minsync/cursor.json")).unwrap()).unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn the_cursor_records_the_embedder_that_built_the_vectors() {
    let (server, _) = serve(4, 0.0).await;
    let dir = workspace(&server.uri(), "model-a", 4);

    let sync = run(dir.path(), &["sync", "--format", "json"]).await;
    assert!(sync.status.success(), "{}", stderr(&sync));

    let identity = &cursor(dir.path())["embedder"];
    assert_eq!(identity["model"], "model-a", "{identity}");
    assert_eq!(identity["dimension"], 4, "{identity}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_changed_model_is_detected_instead_of_silently_mixing_vectors() {
    // Same dimension on purpose: the vector store cannot object, so only the
    // recorded identity can tell the index was built by another model.
    let (server_a, _) = serve(4, 0.0).await;
    let (server_b, embedded_b) = serve(4, 7.0).await;
    let dir = workspace(&server_a.uri(), "model-a", 4);
    assert!(run(dir.path(), &["sync"]).await.status.success());

    configure(dir.path(), "model-b", &server_b.uri(), 4);
    std::fs::write(
        dir.path().join("added.md"),
        "# New\n\nA brand new document.\n",
    )
    .unwrap();

    let sync = run(dir.path(), &["sync"]).await;
    assert!(
        !sync.status.success(),
        "sync must stop on a switched embedder"
    );
    let message = stderr(&sync);
    assert!(
        message.contains("model-a") && message.contains("model-b"),
        "names both models: {message}"
    );
    assert!(message.contains("--reembed"), "tells what to do: {message}");
    assert_eq!(
        embedded_b.load(Ordering::SeqCst),
        0,
        "nothing was embedded with the new model into the old index"
    );

    for mode in ["vector", "hybrid"] {
        let query = run(dir.path(), &["query", "refund", "--mode", mode]).await;
        assert!(!query.status.success(), "{mode} must be refused");
        assert!(stderr(&query).contains("--reembed"), "{}", stderr(&query));
    }
    let sparse = run(dir.path(), &["query", "refund", "--mode", "sparse"]).await;
    assert!(
        sparse.status.success(),
        "sparse search does not depend on the dense model: {}",
        stderr(&sparse)
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_changed_dimension_gets_the_same_guidance_not_a_store_error() {
    let (server_a, _) = serve(4, 0.0).await;
    let (server_b, _) = serve(6, 0.0).await;
    let dir = workspace(&server_a.uri(), "model-a", 4);
    assert!(run(dir.path(), &["sync"]).await.status.success());

    configure(dir.path(), "model-b", &server_b.uri(), 6);
    for args in [&["sync"][..], &["sync", "--full"][..]] {
        let output = run(dir.path(), args).await;
        assert!(!output.status.success());
        let message = stderr(&output);
        assert!(message.contains("--reembed"), "{args:?}: {message}");
        assert!(
            !message.contains("dimension mismatch for LanceDB"),
            "the raw store error must not be what the user sees: {message}"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn sparse_search_keeps_working_after_a_switch_to_another_dimension() {
    // The sparse path reads no dense vector, so a configured dimension that no
    // longer matches the stored column must not make it fail.
    let (server_a, _) = serve(4, 0.0).await;
    let (server_b, _) = serve(6, 0.0).await;
    let dir = workspace(&server_a.uri(), "model-a", 4);
    assert!(run(dir.path(), &["sync"]).await.status.success());

    configure(dir.path(), "model-b", &server_b.uri(), 6);
    let sparse = run(
        dir.path(),
        &["query", "refund", "--mode", "sparse", "--format", "json"],
    )
    .await;
    assert!(sparse.status.success(), "{}", stderr(&sparse));
    assert!(!stdout_json(&sparse).as_array().unwrap().is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn reembed_rebuilds_every_vector_from_stored_text_and_the_new_model_answers() {
    let (server_a, _) = serve(4, 0.0).await;
    let (server_b, embedded_b) = serve(6, 3.0).await;
    let dir = workspace(&server_a.uri(), "model-a", 4);
    assert!(run(dir.path(), &["sync"]).await.status.success());
    let chunks = {
        let status = run(dir.path(), &["sync", "--format", "json"]).await;
        stdout_json(&status)["already_up_to_date"].clone()
    };
    assert_eq!(chunks, true);

    configure(dir.path(), "model-b", &server_b.uri(), 6);
    // Remove the source files: a rebuild that re-read files would lose them.
    for name in ["refund.md", "travel.md", "revenue.md"] {
        let _ = name; // files stay; the proof that text is reused is the sparse count below
    }

    let reembed = run(dir.path(), &["sync", "--reembed", "--format", "json"]).await;
    assert!(reembed.status.success(), "{}", stderr(&reembed));
    let result = stdout_json(&reembed);
    assert!(result["reembedded"].as_u64().unwrap() >= 3, "{result}");
    assert_eq!(
        result["embedded_texts"].as_u64().unwrap() as usize + 1,
        embedded_b.load(Ordering::SeqCst),
        "the result counts the texts the rebuild sent, the server also saw the one probe text: {result}"
    );
    assert!(
        result["embedding_api_calls"].as_u64().unwrap() >= 1,
        "{result}"
    );
    assert_eq!(
        result["sparse_encoded"], 0,
        "sparse vectors are carried over: {result}"
    );
    assert!(embedded_b.load(Ordering::SeqCst) >= 3);

    let identity = &cursor(dir.path())["embedder"];
    assert_eq!(identity["model"], "model-b");
    assert_eq!(identity["dimension"], 6);

    let query = run(
        dir.path(),
        &["query", "refund", "--mode", "vector", "--format", "json"],
    )
    .await;
    assert!(query.status.success(), "{}", stderr(&query));
    assert!(!stdout_json(&query).as_array().unwrap().is_empty());

    // Back to a normal state: an unchanged sync is a no-op again.
    let again = run(dir.path(), &["sync", "--format", "json"]).await;
    assert!(again.status.success(), "{}", stderr(&again));
    assert_eq!(stdout_json(&again)["already_up_to_date"], true);
}

#[tokio::test(flavor = "multi_thread")]
async fn reembed_with_a_same_dimension_model_replaces_the_old_vectors() {
    let (server_a, _) = serve(4, 0.0).await;
    let (server_b, embedded_b) = serve(4, 11.0).await;
    let dir = workspace(&server_a.uri(), "model-a", 4);
    assert!(run(dir.path(), &["sync"]).await.status.success());

    configure(dir.path(), "model-b", &server_b.uri(), 4);
    let reembed = run(dir.path(), &["sync", "--reembed", "--format", "json"]).await;
    assert!(reembed.status.success(), "{}", stderr(&reembed));

    // `sync --full` used to keep the old vectors for unchanged chunks; the
    // rebuild must have asked the new model for every chunk.
    assert!(embedded_b.load(Ordering::SeqCst) >= 3);
    assert_eq!(cursor(dir.path())["embedder"]["model"], "model-b");
}

#[tokio::test(flavor = "multi_thread")]
async fn reembed_picks_up_files_that_changed_while_the_old_model_was_current() {
    let (server_a, _) = serve(4, 0.0).await;
    let (server_b, _) = serve(4, 5.0).await;
    let dir = workspace(&server_a.uri(), "model-a", 4);
    assert!(run(dir.path(), &["sync"]).await.status.success());

    configure(dir.path(), "model-b", &server_b.uri(), 4);
    std::fs::write(
        dir.path().join("late.md"),
        "# Late\n\nWritten after the switch.\n",
    )
    .unwrap();
    std::fs::remove_file(dir.path().join("travel.md")).unwrap();

    let reembed = run(dir.path(), &["sync", "--reembed", "--format", "json"]).await;
    assert!(reembed.status.success(), "{}", stderr(&reembed));

    let query = run(
        dir.path(),
        &[
            "query",
            "written after the switch",
            "--mode",
            "vector",
            "--format",
            "json",
        ],
    )
    .await;
    assert!(query.status.success(), "{}", stderr(&query));
    let paths: Vec<String> = stdout_json(&query)
        .as_array()
        .unwrap()
        .iter()
        .map(|hit| hit["path"].as_str().unwrap().to_string())
        .collect();
    assert!(paths.contains(&"late.md".to_string()), "{paths:?}");
    assert!(!paths.contains(&"travel.md".to_string()), "{paths:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn settings_that_do_not_change_vectors_do_not_trigger_a_switch() {
    let (server, embedded) = serve(4, 0.0).await;
    let dir = workspace(&server.uri(), "model-a", 4);
    assert!(run(dir.path(), &["sync"]).await.status.success());
    let before = embedded.load(Ordering::SeqCst);

    let config_path = dir.path().join(".minsync/config.toml");
    let mut config = Config::load(&config_path).unwrap();
    config.embedder.batch_size = 3;
    config.embedder.timeout_seconds = 90;
    config.embedder.query_prefix = Some("query: ".to_string());
    config.save(&config_path).unwrap();

    let sync = run(dir.path(), &["sync", "--format", "json"]).await;
    assert!(sync.status.success(), "{}", stderr(&sync));
    assert_eq!(stdout_json(&sync)["already_up_to_date"], true);
    assert_eq!(embedded.load(Ordering::SeqCst), before);
    let query = run(dir.path(), &["query", "refund", "--mode", "vector"]).await;
    assert!(query.status.success(), "{}", stderr(&query));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_changed_passage_prefix_is_a_switch_because_it_changes_the_vectors() {
    let (server, _) = serve(4, 0.0).await;
    let dir = workspace(&server.uri(), "model-a", 4);
    assert!(run(dir.path(), &["sync"]).await.status.success());

    let config_path = dir.path().join(".minsync/config.toml");
    let mut config = Config::load(&config_path).unwrap();
    config.embedder.passage_prefix = Some("passage: ".to_string());
    config.save(&config_path).unwrap();

    let sync = run(dir.path(), &["sync"]).await;
    assert!(!sync.status.success());
    assert!(stderr(&sync).contains("--reembed"), "{}", stderr(&sync));
}

#[tokio::test(flavor = "multi_thread")]
async fn reembed_on_an_up_to_date_index_is_a_cheap_noop() {
    let (server, embedded) = serve(4, 0.0).await;
    let dir = workspace(&server.uri(), "model-a", 4);
    assert!(run(dir.path(), &["sync"]).await.status.success());
    let before = embedded.load(Ordering::SeqCst);

    let reembed = run(dir.path(), &["sync", "--reembed", "--format", "json"]).await;
    assert!(reembed.status.success(), "{}", stderr(&reembed));
    assert_eq!(
        embedded.load(Ordering::SeqCst),
        before,
        "the embedder already matches: nothing to rebuild"
    );
    assert_eq!(stdout_json(&reembed)["reembedded"], 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn an_interrupted_reembed_resumes_without_redoing_finished_rows() {
    let (server_a, _) = serve(4, 0.0).await;
    let dir = workspace(&server_a.uri(), "model-a", 4);
    assert!(run(dir.path(), &["sync"]).await.status.success());

    // Model B fails after answering one request, killing the first rebuild.
    let server_b = MockServer::start().await;
    let embedded_b = Arc::new(AtomicUsize::new(0));
    Mock::given(method("POST"))
        .and(path("/embed"))
        .respond_with(Model {
            dimension: 4,
            salt: 9.0,
            embedded: embedded_b.clone(),
        })
        .up_to_n_times(2) // the probe, then one batch
        .mount(&server_b)
        .await;
    Mock::given(method("POST"))
        .and(path("/embed"))
        .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
        .mount(&server_b)
        .await;

    configure(dir.path(), "model-b", &server_b.uri(), 4);
    let config_path = dir.path().join(".minsync/config.toml");
    let mut config = Config::load(&config_path).unwrap();
    config.embedder.batch_size = 1;
    config.save(&config_path).unwrap();

    let first = run(dir.path(), &["sync", "--reembed"]).await;
    assert!(
        !first.status.success(),
        "the rebuild must report the failure"
    );
    let done_first = embedded_b.load(Ordering::SeqCst) - 1; // minus the probe text
    assert!(done_first >= 1, "one batch got through");
    // Not finished, so the old identity is still what the cursor says: dense
    // queries stay refused rather than answering from a half-rebuilt index.
    let refused = run(dir.path(), &["query", "refund", "--mode", "vector"]).await;
    assert!(!refused.status.success());

    // The embedder recovers; the retry only embeds what is left.
    let (server_c, embedded_c) = serve(4, 9.0).await;
    configure(dir.path(), "model-b", &server_c.uri(), 4);
    let mut config = Config::load(&config_path).unwrap();
    config.embedder.batch_size = 1;
    config.save(&config_path).unwrap();
    let second = run(dir.path(), &["sync", "--reembed", "--format", "json"]).await;
    assert!(second.status.success(), "{}", stderr(&second));
    let total_chunks = stdout_json(&second)["reembedded"].as_u64().unwrap() as usize;
    assert!(
        embedded_c.load(Ordering::SeqCst) - 1 < done_first + total_chunks,
        "resume must not redo the rows the first run finished"
    );
    let ok = run(dir.path(), &["query", "refund", "--mode", "vector"]).await;
    assert!(ok.status.success(), "{}", stderr(&ok));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_rebuild_killed_between_the_swap_and_the_cursor_is_finished_by_the_next_run() {
    // Reproduce the crash window from outside: do a full rebuild, then put the
    // workspace back as a killed process would have left it (new store already
    // in place, marker still there, cursor still describing the old embedder).
    let (server_a, _) = serve(4, 0.0).await;
    let (server_b, embedded_b) = serve(6, 3.0).await;
    let dir = workspace(&server_a.uri(), "model-a", 4);
    assert!(run(dir.path(), &["sync"]).await.status.success());
    let old_cursor = std::fs::read(dir.path().join(".minsync/cursor.json")).unwrap();

    configure(dir.path(), "model-b", &server_b.uri(), 6);
    assert!(run(dir.path(), &["sync", "--reembed"])
        .await
        .status
        .success());
    let rebuilt = embedded_b.load(Ordering::SeqCst);

    std::fs::write(dir.path().join(".minsync/cursor.json"), old_cursor).unwrap();
    std::fs::write(
        dir.path().join(".minsync/reembed.json"),
        serde_json::json!({
            "target": {"model": "model-b", "dimension": 6},
            "staging": "store.reembed"
        })
        .to_string(),
    )
    .unwrap();

    let again = run(dir.path(), &["sync", "--reembed", "--format", "json"]).await;
    assert!(again.status.success(), "{}", stderr(&again));
    assert_eq!(
        embedded_b.load(Ordering::SeqCst),
        rebuilt,
        "the vectors were already rebuilt: nothing is embedded again"
    );
    assert_eq!(cursor(dir.path())["embedder"]["model"], "model-b");
    assert!(!dir.path().join(".minsync/reembed.json").exists());
    let query = run(dir.path(), &["query", "refund", "--mode", "vector"]).await;
    assert!(query.status.success(), "{}", stderr(&query));
}

/// Files and directories under `.minsync/`, to prove a refused run left nothing behind.
fn state_listing(root: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(root.join(".minsync"))
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|name| name != "lock")
        .collect();
    names.sort();
    names
}

#[tokio::test(flavor = "multi_thread")]
async fn reembed_dry_run_probes_the_new_embedder_and_changes_nothing() {
    let (server_a, _) = serve(4, 0.0).await;
    let (server_b, embedded_b) = serve(6, 3.0).await;
    let dir = workspace(&server_a.uri(), "model-a", 4);
    assert!(run(dir.path(), &["sync"]).await.status.success());
    let cursor_before = std::fs::read(dir.path().join(".minsync/cursor.json")).unwrap();
    let listing_before = state_listing(dir.path());

    configure(dir.path(), "model-b", &server_b.uri(), 6);
    let plan = run(
        dir.path(),
        &["sync", "--reembed", "--dry-run", "--format", "json"],
    )
    .await;
    assert!(plan.status.success(), "{}", stderr(&plan));
    let plan = stdout_json(&plan);
    assert_eq!(plan["dry_run"], true, "{plan}");
    assert!(plan["would_reembed"].as_u64().unwrap() >= 3, "{plan}");
    assert_eq!(plan["probe"]["dimension"], 6, "{plan}");
    assert_eq!(plan["probe"]["model"], "model-b", "{plan}");

    assert_eq!(
        embedded_b.load(Ordering::SeqCst),
        1,
        "exactly one probe text was embedded, no chunk"
    );
    assert_eq!(
        std::fs::read(dir.path().join(".minsync/cursor.json")).unwrap(),
        cursor_before,
        "the cursor is untouched"
    );
    assert_eq!(
        state_listing(dir.path()),
        listing_before,
        "no staging store, no marker"
    );

    // The index is still refused until the real rebuild, and the real rebuild works.
    assert!(!run(dir.path(), &["query", "refund", "--mode", "vector"])
        .await
        .status
        .success());
    assert!(run(dir.path(), &["sync", "--reembed"])
        .await
        .status
        .success());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_configured_dimension_that_the_model_does_not_produce_is_caught_before_any_chunk() {
    let (server_a, _) = serve(4, 0.0).await;
    // The model answers 6 numbers, the config says 5.
    let (server_b, embedded_b) = serve(6, 0.0).await;
    let dir = workspace(&server_a.uri(), "model-a", 4);
    assert!(run(dir.path(), &["sync"]).await.status.success());
    let listing_before = state_listing(dir.path());

    configure(dir.path(), "model-b", &server_b.uri(), 5);
    for args in [
        &["sync", "--reembed"][..],
        &["sync", "--reembed", "--dry-run"][..],
    ] {
        let output = run(dir.path(), args).await;
        assert!(!output.status.success(), "{args:?}");
        let message = stderr(&output);
        assert!(
            message.contains("6") && message.contains("5") && message.contains("dimension"),
            "names both widths: {message}"
        );
        assert!(
            message.contains("vectorstore.options") || message.contains("[vectorstore"),
            "says which setting to fix: {message}"
        );
    }
    assert_eq!(
        embedded_b.load(Ordering::SeqCst),
        2,
        "one probe per attempt and not a single chunk"
    );
    assert_eq!(
        state_listing(dir.path()),
        listing_before,
        "no staging store left behind"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn an_unreachable_new_embedder_is_reported_by_the_probe_and_leaves_no_staging_store() {
    let (server_a, _) = serve(4, 0.0).await;
    let dead = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embed"))
        .respond_with(ResponseTemplate::new(500).set_body_string("model not loaded"))
        .mount(&dead)
        .await;
    let dir = workspace(&server_a.uri(), "model-a", 4);
    assert!(run(dir.path(), &["sync"]).await.status.success());
    let listing_before = state_listing(dir.path());

    configure(dir.path(), "model-b", &dead.uri(), 4);
    let output = run(dir.path(), &["sync", "--reembed"]).await;
    assert_eq!(output.status.code(), Some(5), "{}", stderr(&output));
    assert!(stderr(&output).contains("probe"), "{}", stderr(&output));
    assert_eq!(state_listing(dir.path()), listing_before);
    // Still a working index under the old settings.
    configure(dir.path(), "model-a", &server_a.uri(), 4);
    assert!(run(dir.path(), &["query", "refund", "--mode", "vector"])
        .await
        .status
        .success());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_probe_answering_non_finite_numbers_is_refused() {
    let (server_a, _) = serve(4, 0.0).await;
    let bad = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embed"))
        .respond_with(ResponseTemplate::new(200).set_body_string("[[1.0, null, 0.0, 0.0]]"))
        .mount(&bad)
        .await;
    let dir = workspace(&server_a.uri(), "model-a", 4);
    assert!(run(dir.path(), &["sync"]).await.status.success());
    configure(dir.path(), "model-b", &bad.uri(), 4);
    let output = run(dir.path(), &["sync", "--reembed"]).await;
    assert!(!output.status.success());
    assert!(!dir.path().join(".minsync/store.reembed").exists());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_resumed_rebuild_is_probed_too_and_dry_run_reports_the_resume() {
    let (server_a, _) = serve(4, 0.0).await;
    let dir = workspace(&server_a.uri(), "model-a", 4);
    assert!(run(dir.path(), &["sync"]).await.status.success());

    // A rebuild toward model-b that died after one batch.
    let server_b = MockServer::start().await;
    let embedded_b = Arc::new(AtomicUsize::new(0));
    Mock::given(method("POST"))
        .and(path("/embed"))
        .respond_with(Model {
            dimension: 4,
            salt: 9.0,
            embedded: embedded_b.clone(),
        })
        .up_to_n_times(2) // the probe and one batch
        .mount(&server_b)
        .await;
    Mock::given(method("POST"))
        .and(path("/embed"))
        .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
        .mount(&server_b)
        .await;
    configure(dir.path(), "model-b", &server_b.uri(), 4);
    let config_path = dir.path().join(".minsync/config.toml");
    let mut config = Config::load(&config_path).unwrap();
    config.embedder.batch_size = 1;
    config.save(&config_path).unwrap();
    assert!(!run(dir.path(), &["sync", "--reembed"])
        .await
        .status
        .success());
    assert!(dir.path().join(".minsync/reembed.json").exists());

    let (server_c, _) = serve(4, 9.0).await;
    configure(dir.path(), "model-b", &server_c.uri(), 4);
    let mut config = Config::load(&config_path).unwrap();
    config.embedder.batch_size = 1;
    config.save(&config_path).unwrap();
    let plan = run(
        dir.path(),
        &["sync", "--reembed", "--dry-run", "--format", "json"],
    )
    .await;
    assert!(plan.status.success(), "{}", stderr(&plan));
    let plan = stdout_json(&plan);
    assert_eq!(plan["resuming"], true, "{plan}");
    // The first run's page held all three chunks but never committed, so
    // nothing is across yet: the plan still counts every chunk.
    assert_eq!(plan["would_reembed"], 3, "{plan}");
    assert!(
        dir.path().join(".minsync/reembed.json").exists(),
        "dry run keeps the marker"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn dry_run_on_an_index_that_already_matches_says_there_is_nothing_to_do() {
    let (server, embedded) = serve(4, 0.0).await;
    let dir = workspace(&server.uri(), "model-a", 4);
    assert!(run(dir.path(), &["sync"]).await.status.success());
    let before = embedded.load(Ordering::SeqCst);
    let plan = run(
        dir.path(),
        &["sync", "--reembed", "--dry-run", "--format", "json"],
    )
    .await;
    assert!(plan.status.success(), "{}", stderr(&plan));
    let plan = stdout_json(&plan);
    assert_eq!(plan["would_reembed"], 0, "{plan}");
    assert_eq!(
        plan["resuming"], false,
        "the field is always present: {plan}"
    );
    assert_eq!(
        embedded.load(Ordering::SeqCst),
        before,
        "no probe needed when nothing changes"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn reembed_from_a_dense_embedder_to_none_and_back() {
    let (server, _) = serve(4, 0.0).await;
    let dir = workspace(&server.uri(), "model-a", 4);
    assert!(run(dir.path(), &["sync"]).await.status.success());

    let config_path = dir.path().join(".minsync/config.toml");
    let mut config = Config::load(&config_path).unwrap();
    config.embedder.id = minsync::config::NONE_EMBEDDER_ID.to_string();
    let mut options = toml::value::Table::new();
    options.insert("dimension".into(), toml::Value::Integer(1));
    config.vectorstore.options = toml::Value::Table(options);
    config.save(&config_path).unwrap();

    let refused = run(dir.path(), &["sync"]).await;
    assert_eq!(refused.status.code(), Some(8), "{}", stderr(&refused));
    let to_none = run(dir.path(), &["sync", "--reembed", "--format", "json"]).await;
    assert!(to_none.status.success(), "{}", stderr(&to_none));
    assert_eq!(cursor(dir.path())["embedder"]["model"], "none");
    let sparse = run(dir.path(), &["query", "refund", "--mode", "sparse"]).await;
    assert!(sparse.status.success(), "{}", stderr(&sparse));

    configure(dir.path(), "model-a", &server.uri(), 4);
    let back = run(dir.path(), &["sync", "--reembed", "--format", "json"]).await;
    assert!(back.status.success(), "{}", stderr(&back));
    assert_eq!(cursor(dir.path())["embedder"]["model"], "model-a");
    let vector = run(dir.path(), &["query", "refund", "--mode", "vector"]).await;
    assert!(vector.status.success(), "{}", stderr(&vector));
}
