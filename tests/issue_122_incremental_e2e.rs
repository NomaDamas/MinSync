//! Crash/resume behaviour of the incremental core against real child
//! processes (issue #122): the native sync is killed mid-run, the external
//! materialization host is killed mid-run, and a materialize -> index
//! pipeline keeps each stage's completion state separate.

use assert_cmd::cargo::CommandCargoExt;
use minsync::config::Config;
use minsync::incremental::journal::{Journal, Load, JOURNAL_FILE};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::{Duration, Instant};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

// ---------------------------------------------------------------- embed mock

struct SlowEmbed(Duration);

impl Respond for SlowEmbed {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let body: Value = serde_json::from_slice(&request.body).expect("embed request JSON");
        let inputs = body["inputs"].as_array().expect("inputs array").len();
        let vectors: Vec<Vec<f32>> = (0..inputs).map(|_| vec![0.5, 0.5, 0.5, 0.5]).collect();
        ResponseTemplate::new(200)
            .set_body_json(json!(vectors))
            .set_delay(self.0)
    }
}

async fn embed_server(delay: Duration) -> MockServer {
    let server = MockServer::start().await;
    mount_embed(&server, delay).await;
    server
}

async fn mount_embed(server: &MockServer, delay: Duration) {
    Mock::given(method("POST"))
        .and(path("/embed"))
        .respond_with(SlowEmbed(delay))
        .mount(server)
        .await;
}

async fn embedded_inputs(server: &MockServer) -> usize {
    server
        .received_requests()
        .await
        .unwrap_or_default()
        .iter()
        .filter(|request| request.url.path() == "/embed")
        .map(|request| {
            let body: Value = serde_json::from_slice(&request.body).expect("embed body");
            body["inputs"].as_array().map_or(0, Vec::len)
        })
        .sum()
}

// ----------------------------------------------------------- vector workspace

/// Initialized vector workspace at `root`, indexing one-chunk files two per
/// batch against the mock TEI server.
fn init_workspace(root: &Path, server: &MockServer) {
    let minsync_dir = root.join(".minsync");
    std::fs::create_dir_all(&minsync_dir).expect("create state directory");
    let mut config = Config::default_for("source-issue-122");
    config.embedder.id = "tei:test-model".to_string();
    config.embedder.base_url = Some(server.uri());
    config.embedder.max_retries = 0;
    config.sync.batch_files = 2;
    let mut options = toml::value::Table::new();
    options.insert("dimension".into(), toml::Value::Integer(4));
    config.vectorstore.options = toml::Value::Table(options);
    config
        .save(&minsync_dir.join("config.toml"))
        .expect("save config");
}

fn minsync(root: &Path) -> Command {
    let mut command = Command::cargo_bin("minsync").expect("find minsync binary");
    command.current_dir(root);
    command
}

fn write_notes(root: &Path, count: usize) {
    for index in 0..count {
        std::fs::write(
            root.join(format!("note-{index}.md")),
            format!("note number {index}"),
        )
        .expect("write note");
    }
}

fn sync_json(root: &Path) -> (bool, Value) {
    let output = minsync(root)
        .args(["--format", "json", "sync"])
        .output()
        .expect("run sync");
    let result = serde_json::from_slice(&output.stdout).unwrap_or(Value::Null);
    (output.status.success(), result)
}

fn ledger_records(root: &Path) -> usize {
    match Journal::load(&root.join(".minsync").join(JOURNAL_FILE)) {
        Load::Loaded(ledger) => ledger.records().count(),
        other => panic!("expected a readable resume log, got {other:?}"),
    }
}

/// Wait until the sync's resume log holds at least `records` completions.
async fn wait_for_records(root: &Path, records: usize) {
    let log = root.join(".minsync").join(JOURNAL_FILE);
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let lines = std::fs::read_to_string(&log)
            .map(|text| text.lines().count().saturating_sub(1))
            .unwrap_or(0);
        if lines >= records {
            return;
        }
        assert!(Instant::now() < deadline, "sync never committed {records}");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

// ------------------------------------------------------ materialization host

struct Stage {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Stage {
    fn spawn(state_dir: &Path) -> Self {
        let mut child = Command::cargo_bin("minsync")
            .expect("find minsync binary")
            .args(["stage", "--state-dir"])
            .arg(state_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn minsync stage");
        let stdin = child.stdin.take().expect("piped stdin");
        let stdout = BufReader::new(child.stdout.take().expect("piped stdout"));
        Self {
            child,
            stdin,
            stdout,
        }
    }

    fn call(&mut self, request: Value) -> Value {
        writeln!(self.stdin, "{request}").expect("send request");
        self.stdin.flush().expect("flush request");
        let mut line = String::new();
        self.stdout.read_line(&mut line).expect("read response");
        serde_json::from_str(line.trim()).unwrap_or_else(|e| panic!("response `{line}`: {e}"))
    }

    /// Close stdin the way a host that exited would, and return the exit code.
    fn hang_up(self) -> i32 {
        let Stage {
            mut child, stdin, ..
        } = self;
        drop(stdin);
        child.wait().expect("wait").code().expect("exit code")
    }

    /// SIGKILL: no cleanup of any kind runs.
    fn kill(mut self) {
        self.child.kill().expect("kill");
        self.child.wait().expect("reap");
    }
}

fn version_of(bytes: &[u8]) -> String {
    format!("sha256:{}", hex::encode(Sha256::digest(bytes)))
}

/// The "parser": markdown for a source document. Deterministic in the
/// content alone, so a parser fingerprint change does not by itself change
/// what the downstream indexer sees.
fn parse(key: &str, content: &str) -> String {
    format!("# {key}\n\n{}\n", content.trim())
}

/// Host-side corpus: `src/<key>` documents, outputs under `out/<key>.md`.
struct Corpus {
    src: PathBuf,
    out: PathBuf,
}

impl Corpus {
    fn new(root: &Path) -> Self {
        let src = root.join("src");
        let out = root.join("mirror");
        std::fs::create_dir_all(&src).expect("src");
        std::fs::create_dir_all(&out).expect("out");
        Self { src, out }
    }

    fn write(&self, key: &str, content: &str) {
        std::fs::write(self.src.join(key), content).expect("write source");
    }

    fn items(&self) -> Vec<Value> {
        let mut keys: Vec<String> = std::fs::read_dir(&self.src)
            .expect("read src")
            .map(|entry| {
                entry
                    .expect("entry")
                    .file_name()
                    .into_string()
                    .expect("utf8")
            })
            .collect();
        keys.sort();
        keys.into_iter()
            .map(|key| {
                let bytes = std::fs::read(self.src.join(&key)).expect("read source");
                json!({"key": key, "version": version_of(&bytes)})
            })
            .collect()
    }

    fn open_request(&self, fingerprint: &str) -> Value {
        json!({
            "op": "open",
            "stage": "parse",
            "source_id": "corpus-122",
            "fingerprint": fingerprint,
            "output_root": self.out,
            "items": self.items(),
        })
    }

    /// Parse one document the way a host does: temp file, rename, commit.
    fn materialize(&self, stage: &mut Stage, decision: &Value) -> Value {
        let key = decision["key"].as_str().expect("key");
        let version = decision["version"].as_str().expect("version");
        let content = std::fs::read_to_string(self.src.join(key)).expect("read source");
        let rel = format!("{key}.md");
        let tmp = self.out.join(format!(".{rel}.tmp"));
        std::fs::write(&tmp, parse(key, &content)).expect("write tmp");
        std::fs::rename(&tmp, self.out.join(&rel)).expect("rename into place");
        stage.call(json!({
            "op": "commit",
            "key": key,
            "version": version,
            "output": {"path": rel},
        }))
    }

    /// Open a run and materialize up to `limit` pending items; returns the
    /// open response and the keys that were committed.
    fn run(&self, stage: &mut Stage, fingerprint: &str, limit: usize) -> (Value, Vec<String>) {
        let opened = stage.call(self.open_request(fingerprint));
        assert_eq!(opened["ok"], true, "{opened}");
        let mut done = Vec::new();
        for decision in opened["decisions"].as_array().expect("decisions") {
            if done.len() >= limit {
                break;
            }
            if decision["action"] == "delete" {
                continue;
            }
            let response = self.materialize(stage, decision);
            assert_eq!(response["ok"], true, "{response}");
            done.push(decision["key"].as_str().expect("key").to_string());
        }
        (opened, done)
    }
}

fn state_dir(root: &Path) -> PathBuf {
    root.join("parse-state")
}

fn keys(open: &Value) -> Vec<String> {
    open["decisions"]
        .as_array()
        .expect("decisions")
        .iter()
        .map(|d| d["key"].as_str().expect("key").to_string())
        .collect()
}

// ------------------------------------------------------------------- tests

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_native_sync_killed_mid_run_resumes_only_the_unfinished_files() {
    let slow = embed_server(Duration::from_millis(400)).await;
    let root = tempfile::tempdir().expect("workspace");
    init_workspace(root.path(), &slow);
    write_notes(root.path(), 10);

    let mut child = minsync(root.path())
        .arg("sync")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn sync");
    wait_for_records(root.path(), 2).await;
    child.kill().expect("kill sync");
    child.wait().expect("reap sync");

    let committed = ledger_records(root.path());
    assert!(
        (2..10).contains(&committed),
        "the kill must land mid-run, {committed} committed"
    );
    assert!(root.path().join(".minsync/txn.json").exists());
    assert!(!root.path().join(".minsync/cursor.json").exists());

    // Same server (the embedder config is part of the run fingerprint),
    // now answering immediately and counting from zero.
    slow.reset().await;
    mount_embed(&slow, Duration::ZERO).await;
    let (ok, result) = sync_json(root.path());
    assert!(ok, "{result}");
    assert_eq!(result["files_resumed"], committed, "{result}");
    assert_eq!(result["files_processed"], 10 - committed, "{result}");
    assert_eq!(
        embedded_inputs(&slow).await,
        10 - committed,
        "committed files are not embedded again"
    );
    assert!(root.path().join(".minsync/cursor.json").exists());
    assert!(!root.path().join(".minsync").join(JOURNAL_FILE).exists());
}

#[tokio::test]
async fn a_materialization_host_killed_mid_run_resumes_from_pending_items() {
    let root = tempfile::tempdir().expect("root");
    let corpus = Corpus::new(root.path());
    for index in 0..6 {
        corpus.write(&format!("doc-{index}.txt"), &format!("document {index}"));
    }

    let mut stage = Stage::spawn(&state_dir(root.path()));
    let (_, done) = corpus.run(&mut stage, "parser-1", 3);
    assert_eq!(done.len(), 3);
    stage.kill(); // SIGKILL: no finish, no flush, no cleanup

    let mut stage = Stage::spawn(&state_dir(root.path()));
    let (opened, redone) = corpus.run(&mut stage, "parser-1", usize::MAX);
    assert_eq!(opened["resumed"], true, "{opened}");
    assert_eq!(opened["counts"]["resumed"], 3, "{opened}");
    assert_eq!(redone.len(), 3, "only the unfinished items are parsed");
    assert!(
        redone.iter().all(|key| !done.contains(key)),
        "{done:?} vs {redone:?}"
    );
    let finished = stage.call(json!({"op": "finish"}));
    assert_eq!(finished["committed"], 6, "{finished}");
    assert_eq!(finished["pending"], 0, "{finished}");
    assert_eq!(stage.hang_up(), 0);

    let mut stage = Stage::spawn(&state_dir(root.path()));
    let (opened, redone) = corpus.run(&mut stage, "parser-1", usize::MAX);
    assert!(redone.is_empty(), "{opened}");
    assert_eq!(opened["counts"]["skip"], 6);
}

#[tokio::test]
async fn a_host_that_hangs_up_without_finishing_leaves_a_resumable_run() {
    let root = tempfile::tempdir().expect("root");
    let corpus = Corpus::new(root.path());
    corpus.write("a.txt", "alpha");
    corpus.write("b.txt", "beta");

    let mut stage = Stage::spawn(&state_dir(root.path()));
    corpus.run(&mut stage, "parser-1", 1);
    assert_eq!(stage.hang_up(), 1, "an unfinished run is not a success");

    let mut stage = Stage::spawn(&state_dir(root.path()));
    let (opened, redone) = corpus.run(&mut stage, "parser-1", usize::MAX);
    assert_eq!(opened["resumed"], true);
    assert_eq!(redone, ["b.txt"]);
}

#[tokio::test]
async fn an_output_written_just_before_a_kill_is_not_trusted_until_committed() {
    let root = tempfile::tempdir().expect("root");
    let corpus = Corpus::new(root.path());
    corpus.write("a.txt", "alpha");

    let mut stage = Stage::spawn(&state_dir(root.path()));
    let opened = stage.call(corpus.open_request("parser-1"));
    assert_eq!(keys(&opened), ["a.txt"]);
    // The output commit happened (file in place); the process dies before
    // the completion record.
    std::fs::write(corpus.out.join("a.txt.md"), parse("a.txt", "alpha")).expect("output");
    stage.kill();

    let mut stage = Stage::spawn(&state_dir(root.path()));
    let (opened, redone) = corpus.run(&mut stage, "parser-1", usize::MAX);
    assert_eq!(
        redone,
        ["a.txt"],
        "{opened}: the missing record is repaired"
    );
    let finished = stage.call(json!({"op": "finish"}));
    assert_eq!(finished["committed"], 1);
    assert_eq!(
        std::fs::read_to_string(corpus.out.join("a.txt.md")).unwrap(),
        parse("a.txt", "alpha"),
        "the idempotent re-commit leaves the same bytes"
    );
}

#[tokio::test]
async fn a_second_run_on_the_same_stage_is_refused_with_the_lock_exit_code() {
    let root = tempfile::tempdir().expect("root");
    let corpus = Corpus::new(root.path());
    corpus.write("a.txt", "alpha");

    let mut first = Stage::spawn(&state_dir(root.path()));
    assert_eq!(first.call(corpus.open_request("parser-1"))["ok"], true);

    let mut second = Stage::spawn(&state_dir(root.path()));
    let refused = second.call(corpus.open_request("parser-1"));
    assert_eq!(refused["ok"], false, "{refused}");
    assert_eq!(refused["error"]["code"], "lock_failed", "{refused}");
    assert_eq!(second.hang_up(), 3);

    // Killing the holder releases the lock: a crash never wedges the stage.
    first.kill();
    let mut third = Stage::spawn(&state_dir(root.path()));
    assert_eq!(third.call(corpus.open_request("parser-1"))["ok"], true);
}

#[tokio::test]
async fn an_unreadable_source_keeps_its_output_and_a_confirmed_deletion_removes_it() {
    let root = tempfile::tempdir().expect("root");
    let corpus = Corpus::new(root.path());
    corpus.write("keep.txt", "keep");
    corpus.write("unreadable.txt", "unreadable");
    corpus.write("gone.txt", "gone");

    let mut stage = Stage::spawn(&state_dir(root.path()));
    corpus.run(&mut stage, "parser-1", usize::MAX);
    assert_eq!(stage.call(json!({"op": "finish"}))["committed"], 3);
    assert_eq!(stage.hang_up(), 0);

    // `unreadable.txt` cannot be observed this time; `gone.txt` is confirmed
    // absent. Only the latter may lose its output.
    std::fs::remove_file(corpus.src.join("gone.txt")).unwrap();
    std::fs::remove_file(corpus.src.join("unreadable.txt")).unwrap();
    let mut stage = Stage::spawn(&state_dir(root.path()));
    let mut open = corpus.open_request("parser-1");
    open["unknown"] = json!(["unreadable.txt"]);
    let opened = stage.call(open);
    assert_eq!(
        opened["decisions"],
        json!([{"key": "gone.txt", "action": "delete"}])
    );
    assert_eq!(opened["counts"]["preserve"], 1);
    let refused = stage.call(json!({"op": "delete", "key": "unreadable.txt"}));
    assert_eq!(refused["error"]["code"], "preserved_item", "{refused}");
    let deleted = stage.call(json!({"op": "delete", "key": "gone.txt"}));
    assert_eq!(deleted["output_removed"], true, "{deleted}");
    stage.call(json!({"op": "finish"}));
    assert_eq!(stage.hang_up(), 0);

    assert!(corpus.out.join("unreadable.txt.md").exists());
    assert!(corpus.out.join("keep.txt.md").exists());
    assert!(!corpus.out.join("gone.txt.md").exists());
}

/// materialize -> index: a parsed output is reused when only indexing fails,
/// and each stage's completion state moves independently of the other.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn parsed_outputs_are_reused_when_only_vector_indexing_fails() {
    let server = embed_server(Duration::ZERO).await;
    let root = tempfile::tempdir().expect("root");
    let corpus = Corpus::new(root.path());
    for index in 0..4 {
        corpus.write(&format!("doc-{index}.txt"), &format!("document {index}"));
    }
    init_workspace(&corpus.out, &server);

    let mut stage = Stage::spawn(&state_dir(root.path()));
    let (_, done) = corpus.run(&mut stage, "parser-1", usize::MAX);
    assert_eq!(done.len(), 4);
    assert_eq!(stage.call(json!({"op": "finish"}))["committed"], 4);
    assert_eq!(stage.hang_up(), 0);

    // The embedder is down: the indexing stage fails, the parsing stage is
    // already complete.
    server.reset().await;
    Mock::given(method("POST"))
        .and(path("/embed"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;
    let (ok, _) = sync_json(&corpus.out);
    assert!(!ok, "indexing must fail while the embedder is down");

    let mut stage = Stage::spawn(&state_dir(root.path()));
    let (opened, redone) = corpus.run(&mut stage, "parser-1", usize::MAX);
    assert!(redone.is_empty(), "nothing is parsed again: {opened}");
    assert_eq!(opened["counts"]["skip"], 4);
    assert_eq!(stage.hang_up(), 1, "(unfinished on purpose)");

    // The embedder recovers: indexing completes from the stable outputs.
    server.reset().await;
    mount_embed(&server, Duration::ZERO).await;
    let (ok, result) = sync_json(&corpus.out);
    assert!(ok, "{result}");
    assert_eq!(result["files_added"], 4, "{result}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn each_stage_is_invalidated_only_by_its_own_configuration() {
    let slow = embed_server(Duration::from_millis(300)).await;
    let root = tempfile::tempdir().expect("root");
    let corpus = Corpus::new(root.path());
    for index in 0..8 {
        corpus.write(&format!("doc-{index}.txt"), &format!("document {index}"));
    }
    init_workspace(&corpus.out, &slow);

    // Stage 1 completes. Stage 2 (indexing) is killed midway.
    let mut stage = Stage::spawn(&state_dir(root.path()));
    corpus.run(&mut stage, "parser-1", usize::MAX);
    assert_eq!(stage.call(json!({"op": "finish"}))["committed"], 8);
    assert_eq!(stage.hang_up(), 0);

    let mut child = minsync(&corpus.out)
        .arg("sync")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn sync");
    wait_for_records(&corpus.out, 2).await;
    child.kill().expect("kill sync");
    child.wait().expect("reap sync");
    assert!(ledger_records(&corpus.out) >= 2);

    // Change only the downstream (vector) configuration.
    let config_path = corpus.out.join(".minsync/config.toml");
    let mut config = Config::load(&config_path).expect("load config");
    config.chunker.options.max_chunk_size += 1;
    config.save(&config_path).expect("save config");

    let mut stage = Stage::spawn(&state_dir(root.path()));
    let (opened, redone) = corpus.run(&mut stage, "parser-1", usize::MAX);
    assert!(redone.is_empty(), "parsing is untouched: {opened}");
    assert_eq!(opened["counts"]["skip"], 8);
    stage.call(json!({"op": "finish"}));
    assert_eq!(stage.hang_up(), 0);

    slow.reset().await;
    mount_embed(&slow, Duration::ZERO).await;
    let (ok, result) = sync_json(&corpus.out);
    assert!(ok, "{result}");
    assert_eq!(
        result["files_resumed"], 0,
        "the interrupted indexing run is invalid under the new vector config: {result}"
    );
    assert_eq!(result["files_processed"], 8, "{result}");

    // Change only the parser: parsing redoes everything, indexing does not
    // (the parser produced identical text).
    let mut stage = Stage::spawn(&state_dir(root.path()));
    let (opened, redone) = corpus.run(&mut stage, "parser-2", usize::MAX);
    assert_eq!(redone.len(), 8, "{opened}");
    stage.call(json!({"op": "finish"}));
    assert_eq!(stage.hang_up(), 0);
    let (ok, result) = sync_json(&corpus.out);
    assert!(ok, "{result}");
    assert_eq!(result["already_up_to_date"], true, "{result}");
}

fn inspect_request(corpus: &Corpus) -> Value {
    json!({
        "op": "inspect",
        "stage": "parse",
        "source_id": "corpus-122",
        "output_root": corpus.out,
    })
}

fn sorted_keys(list: &Value) -> Vec<String> {
    let mut keys: Vec<String> = list
        .as_array()
        .expect("list")
        .iter()
        .map(|entry| entry["key"].as_str().expect("key").to_string())
        .collect();
    keys.sort();
    keys
}

/// A host probes the binary and reads the durable state from a second
/// process while a writer holds the stage lock, without opening a run.
#[tokio::test]
async fn capabilities_plan_and_inspect_read_beside_a_running_writer() {
    let root = tempfile::tempdir().expect("root");
    let corpus = Corpus::new(root.path());
    corpus.write("a.txt", "alpha");
    corpus.write("b.txt", "beta");

    // Probing a fresh stage creates nothing.
    let mut probe = Stage::spawn(&state_dir(root.path()));
    let caps = probe.call(json!({"op": "capabilities"}));
    assert_eq!(caps["ok"], true, "{caps}");
    assert_eq!(caps["protocol"], 2, "{caps}");
    for feature in ["deferred_failure", "deliberate_skip", "plan", "inspect"] {
        assert!(
            caps["features"]
                .as_array()
                .expect("features")
                .contains(&json!(feature)),
            "{caps}"
        );
    }
    assert_eq!(probe.hang_up(), 0, "a read-only session exits cleanly");
    assert!(!state_dir(root.path()).exists(), "probing creates no state");

    // A writer commits `a.txt` and is still running, lock held.
    let mut writer = Stage::spawn(&state_dir(root.path()));
    let opened = writer.call(corpus.open_request("parser-1"));
    assert_eq!(keys(&opened), ["a.txt", "b.txt"]);
    let decision = opened["decisions"][0].clone();
    assert_eq!(corpus.materialize(&mut writer, &decision)["ok"], true);

    let mut reader = Stage::spawn(&state_dir(root.path()));
    let inspected = reader.call(inspect_request(&corpus));
    assert_eq!(inspected["ok"], true, "{inspected}");
    assert_eq!(sorted_keys(&inspected["items"]), ["a.txt"], "{inspected}");
    let mut plan = corpus.open_request("parser-1");
    plan["op"] = json!("plan");
    let planned = reader.call(plan);
    assert_eq!(planned["ok"], true, "{planned}");
    assert_eq!(planned["counts"]["skip"], 1, "{planned}");
    assert_eq!(planned["counts"]["process"], 1, "{planned}");
    assert_eq!(reader.hang_up(), 0, "reading never needed the lock");

    let decision = opened["decisions"][1].clone();
    assert_eq!(corpus.materialize(&mut writer, &decision)["ok"], true);
    let finished = writer.call(json!({"op": "finish"}));
    assert_eq!(finished["committed"], 2, "{finished}");
    assert_eq!(writer.hang_up(), 0);
}

/// A held failure, a deliberate omission and a discarding failure are all
/// durable across SIGKILL, and `inspect` reports them after the resumed run.
#[tokio::test]
async fn held_failures_omissions_and_discards_survive_a_kill() {
    let root = tempfile::tempdir().expect("root");
    let corpus = Corpus::new(root.path());
    for key in ["a.txt", "b.txt", "c.txt", "d.txt"] {
        corpus.write(key, &format!("{key} v1"));
    }
    let mut stage = Stage::spawn(&state_dir(root.path()));
    corpus.run(&mut stage, "parser-1", usize::MAX);
    assert_eq!(stage.call(json!({"op": "finish"}))["committed"], 4);
    assert_eq!(stage.hang_up(), 0);

    for key in ["b.txt", "c.txt", "d.txt"] {
        corpus.write(key, &format!("{key} v2"));
    }
    let open = |retry_failed: bool| {
        let mut open = corpus.open_request("parser-1");
        open["retry_fingerprint"] = json!("retry-1");
        open["retry_failed"] = json!(retry_failed);
        open
    };
    let version = |opened: &Value, key: &str| {
        opened["decisions"]
            .as_array()
            .expect("decisions")
            .iter()
            .find(|decision| decision["key"] == key)
            .unwrap_or_else(|| panic!("{key} in {opened}"))["version"]
            .clone()
    };

    let mut stage = Stage::spawn(&state_dir(root.path()));
    let opened = stage.call(open(false));
    assert_eq!(keys(&opened), ["b.txt", "c.txt", "d.txt"], "{opened}");
    let held = stage.call(json!({"op": "fail", "key": "b.txt",
        "version": version(&opened, "b.txt"), "error": "bad xref", "defer": true}));
    assert_eq!(held["ok"], true, "{held}");
    let omitted = stage.call(json!({"op": "skip", "key": "c.txt",
        "version": version(&opened, "c.txt"), "reason": "unsupported type"}));
    assert_eq!(omitted["ok"], true, "{omitted}");
    let discarded = stage.call(json!({"op": "fail", "key": "d.txt",
        "version": version(&opened, "d.txt"), "error": "corrupt", "discard_output": true}));
    assert_eq!(discarded["ok"], true, "{discarded}");
    stage.kill(); // SIGKILL before finish

    assert!(
        corpus.out.join("b.txt.md").exists(),
        "last good output kept"
    );
    assert!(!corpus.out.join("c.txt.md").exists(), "omission removed it");
    assert!(!corpus.out.join("d.txt.md").exists(), "discard removed it");

    let mut stage = Stage::spawn(&state_dir(root.path()));
    let opened = stage.call(open(false));
    assert_eq!(opened["resumed"], true, "{opened}");
    assert_eq!(opened["counts"]["resumed"], 1, "the omission: {opened}");
    assert_eq!(
        opened["counts"]["skip"], 3,
        "a, held b, omitted c: {opened}"
    );
    let decisions = opened["decisions"].as_array().expect("decisions");
    assert_eq!(decisions.len(), 2, "{opened}");
    assert_eq!(decisions[0]["key"], "b.txt");
    assert_eq!(decisions[0]["action"], "skip");
    assert_eq!(decisions[0]["deferred"], true);
    assert_eq!(decisions[0]["error"], "bad xref");
    assert_eq!(decisions[1]["key"], "d.txt");
    assert_eq!(decisions[1]["action"], "retry");
    assert_eq!(corpus.materialize(&mut stage, &decisions[1])["ok"], true);
    let finished = stage.call(json!({"op": "finish"}));
    assert_eq!(
        finished["pending"], 0,
        "a held failure is not work: {finished}"
    );
    assert_eq!(stage.hang_up(), 0);

    let mut reader = Stage::spawn(&state_dir(root.path()));
    let inspected = reader.call(inspect_request(&corpus));
    assert_eq!(
        sorted_keys(&inspected["items"]),
        ["a.txt", "b.txt", "d.txt"],
        "{inspected}"
    );
    assert_eq!(
        inspected["failed"],
        json!([{"key": "b.txt", "version": version(&opened, "b.txt"),
                "error": "bad xref", "deferred": true}])
    );
    assert_eq!(sorted_keys(&inspected["skipped"]), ["c.txt"]);
    assert_eq!(reader.hang_up(), 0);

    // An explicit retry releases the held failure.
    let mut stage = Stage::spawn(&state_dir(root.path()));
    let opened = stage.call(open(true));
    assert_eq!(
        opened["decisions"],
        json!([{"key": "b.txt", "action": "retry", "version": version(&opened, "b.txt")}])
    );
    stage.kill();
}
