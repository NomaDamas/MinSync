//! The narrow host protocol for a materialization stage: one JSON object per
//! line on stdin, one JSON response per line on stdout.
//!
//! One session is one run. The process holds the stage's state lock from
//! `open` until `finish` or exit; the OS releases it on any exit, including
//! `kill -9`, so a dead host or a dead MinSync never wedges the stage. A
//! session that ends without `finish` leaves the journal behind, and the
//! next `open` resumes from it.
//!
//! ```text
//! -> {"op":"open","stage":"parse","source_id":"corpus","fingerprint":"parser-1",
//!     "output_root":"/data/mirror","items":[{"key":"a.pdf","version":"sha256:.."}],
//!     "unknown":["b.pdf"],"full":false,"verify":"fast","wait":false,
//!     "retry_fingerprint":"retry-1","retry_failed":false}
//! <- {"ok":true,"op":"open","run_token":"..","resumed":false,
//!     "counts":{"process":1,"retry":0,"skip":0,"resumed":0,"delete":0,"preserve":1},
//!     "decisions":[{"key":"a.pdf","action":"process","version":"sha256:.."}]}
//! -> {"op":"commit","key":"a.pdf","version":"sha256:..",
//!     "output":{"path":"a.pdf.md","size":123,"digest":"sha256:.."}}
//! <- {"ok":true,"op":"commit","key":"a.pdf","output":{"path":"a.pdf.md","size":123,"digest":".."}}
//! -> {"op":"fail","key":"c.pdf","version":"..","error":"parser crashed",
//!     "defer":true,"discard_output":false}
//! -> {"op":"skip","key":"d.pdf","version":"..","reason":"unsupported type"}
//! -> {"op":"delete","key":"gone.pdf"}
//! -> {"op":"finish"}
//! <- {"ok":true,"op":"finish","committed":1,"deleted":0,"failed":0,"pending":0}
//! ```
//!
//! `plan` and `inspect` are read-only and need no open run (they are also
//! accepted in the middle of one). Neither takes the stage lock, creates
//! state or writes anything.
//!
//! `capabilities` is the same kind of request and answers what this build
//! supports:
//!
//! ```text
//! -> {"op":"capabilities"}
//! <- {"ok":true,"op":"capabilities","protocol":2,
//!     "features":["retry_fingerprint","retry_failed","deferred_failure",
//!                 "discard_output","deliberate_skip","plan","inspect"],
//!     "journal_format":3,"snapshot_format":2}
//! ```
//!
//! A host must ask for it before sending the newer fields (`retry_fingerprint`,
//! `retry_failed`, `defer`, `discard_output`) or ops (`skip`, `plan`,
//! `inspect`): an older binary parses the known ops with unknown fields
//! *ignored*, so it would silently hold nothing, discard nothing and retry at
//! the wrong time. Against an older binary this request itself is refused with
//! `invalid_request` — that refusal is the marker.
//!
//! ```text
//! -> {"op":"plan","stage":"parse","source_id":"corpus","fingerprint":"parser-1",
//!     "output_root":"/data/mirror","items":[...],"unknown":["b.pdf"]}
//! <- {"ok":true,"op":"plan","counts":{...},
//!     "decisions":[{"key":"a.pdf","action":"skip","version":"sha256:.."},
//!                  {"key":"c.pdf","action":"skip","version":"sha256:..",
//!                   "deferred":true,"error":"parser crashed"}]}
//! -> {"op":"inspect","stage":"parse","source_id":"corpus",
//!     "output_root":"/data/mirror","verify":"fast"}
//! <- {"ok":true,"op":"inspect",
//!     "items":[{"key":"a.pdf","version":"sha256:..","fingerprint":"parser-1",
//!               "output":{"path":"a.pdf.md","size":123,"digest":"sha256:..",
//!                         "mtime_ns":"1761300000000000000"},
//!               "updated_at":"2026-10-24T09:20:00+00:00"}],
//!     "failed":[{"key":"c.pdf","version":"sha256:..","error":"parser crashed",
//!                "deferred":true},
//!               {"key":"e.pdf","version":null,"error":"bad xref",
//!                "deferred":false}],
//!     "skipped":[{"key":"d.pdf","version":"sha256:..","reason":"unsupported type",
//!                 "fingerprint":"parser-1"}],
//!     "known":[{"key":"a.pdf","version":"sha256:.."},
//!              {"key":"c.pdf","version":"sha256:.."},
//!              {"key":"d.pdf","version":"sha256:.."},
//!              {"key":"e.pdf","version":null}]}
//! ```
//!
//! `known` is the roster of every key the durable snapshot and journal hold —
//! completions, omissions and failures — whether or not its output still
//! verifies and whether or not its failure is deferred. A host uses it to
//! keep the state of roots it cannot currently read, never as a substitute
//! for a verified output. A `version` of `null` marks a failure recorded
//! before versions were kept; every other entry carries the source version
//! the state is at, taken from the failure, then the omission, then the
//! completion.
//!
//! `open` and `plan` return decisions for work to do (`process`, `retry`,
//! `delete`) and, so the host can tell a held item from an up-to-date one,
//! for skips that rest on a deferred failure. `plan` additionally returns
//! every other item's decision, so a host can report staleness; those skips
//! carry no `deferred`/`error`.
//!
//! A refused request is answered with
//! `{"ok":false,"op":"..","error":{"code":"..","message":".."}}` and the
//! session goes on. A lock held by another run ends the session
//! (exit code 3), a clean `finish` exits 0, and input that ends mid-run
//! exits 1 with the run left resumable.

use super::journal::FORMAT_VERSION;
use super::plan::SourceItem;
use super::snapshot::SNAPSHOT_FORMAT;
use super::stage::{FailOptions, OutputClaim, StageRun, StageSpec, VerifyMode};
use super::view;
use crate::error::{MinSyncError, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
struct ItemRequest {
    key: String,
    version: String,
}

#[derive(Deserialize)]
struct OpenRequest {
    stage: String,
    source_id: String,
    fingerprint: String,
    output_root: PathBuf,
    items: Vec<ItemRequest>,
    #[serde(default)]
    unknown: Vec<String>,
    #[serde(default)]
    full: bool,
    #[serde(default)]
    wait: bool,
    #[serde(default)]
    verify: VerifyMode,
    /// The host's retry policy identity; a deferred failure is released when
    /// it changes.
    #[serde(default)]
    retry_fingerprint: Option<String>,
    /// Retry deferred failures now.
    #[serde(default)]
    retry_failed: bool,
}

#[derive(Deserialize)]
struct OutputRequest {
    path: String,
    #[serde(default)]
    size: Option<u64>,
    #[serde(default)]
    digest: Option<String>,
}

#[derive(Deserialize)]
struct CommitRequest {
    key: String,
    version: String,
    #[serde(default)]
    fingerprint: Option<String>,
    output: OutputRequest,
}

#[derive(Deserialize)]
struct FailRequest {
    key: String,
    version: String,
    #[serde(default)]
    fingerprint: Option<String>,
    #[serde(default)]
    error: String,
    /// Hold the failure instead of retrying it on the next run.
    #[serde(default)]
    defer: bool,
    /// Throw the item's previous output away with the failure.
    #[serde(default)]
    discard_output: bool,
}

#[derive(Deserialize)]
struct SkipRequest {
    key: String,
    version: String,
    #[serde(default)]
    reason: String,
}

#[derive(Deserialize)]
struct InspectRequest {
    stage: String,
    source_id: String,
    output_root: PathBuf,
    #[serde(default)]
    verify: VerifyMode,
}

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum Request {
    /// Report what this build can do, so a host never silently relies on a
    /// field or op an older binary would ignore.
    Capabilities,
    Open(OpenRequest),
    Plan(OpenRequest),
    Inspect(InspectRequest),
    Commit(CommitRequest),
    Fail(FailRequest),
    Skip(SkipRequest),
    Delete {
        key: String,
    },
    Finish,
}

/// Protocol revision this build speaks. A host must ask `capabilities` (and
/// refuse to run when it is not answered) before sending `retry_fingerprint`,
/// `retry_failed`, `defer`, `discard_output`, `skip`, `plan` or `inspect`: an
/// older binary parses `open` and `fail` with unknown fields *ignored*, and
/// would silently hold nothing, discard nothing and retry at the wrong time.
pub const PROTOCOL_VERSION: u32 = 2;

/// Capabilities of this build, in the order they were added.
pub const FEATURES: [&str; 7] = [
    "retry_fingerprint",
    "retry_failed",
    "deferred_failure",
    "discard_output",
    "deliberate_skip",
    "plan",
    "inspect",
];

fn error_code(error: &MinSyncError) -> &'static str {
    match error {
        MinSyncError::Stage { code, .. } => code,
        MinSyncError::LockFailed => "lock_failed",
        MinSyncError::Io(_) => "io_error",
        _ => "internal",
    }
}

fn respond(output: &mut impl Write, value: &Value) -> Result<()> {
    writeln!(output, "{value}")?;
    output.flush()?;
    Ok(())
}

fn refusal(op: &str, error: &MinSyncError) -> Value {
    json!({
        "ok": false,
        "op": op,
        "error": {"code": error_code(error), "message": error.to_string()},
    })
}

fn decision_json(decision: &super::plan::Decision) -> Value {
    let mut value = json!({"key": decision.key, "action": decision.action});
    if let Some(version) = &decision.version {
        value["version"] = json!(version);
    }
    if let Some(error) = &decision.deferred_error {
        // The item is not work: it has a held failure and, if it ever had
        // one, its last good output is still in place.
        value["deferred"] = json!(true);
        value["error"] = json!(error);
    }
    value
}

impl Request {
    fn op(&self) -> &'static str {
        match self {
            Request::Capabilities => "capabilities",
            Request::Open(_) => "open",
            Request::Plan(_) => "plan",
            Request::Inspect(_) => "inspect",
            Request::Commit(_) => "commit",
            Request::Fail(_) => "fail",
            Request::Skip(_) => "skip",
            Request::Delete { .. } => "delete",
            Request::Finish => "finish",
        }
    }
}

/// The `capabilities` answer: what a host may rely on from this build. A host
/// that asks and gets `invalid_request` back is talking to an older binary and
/// must not send the newer fields or ops at all.
fn capabilities() -> Value {
    json!({
        "ok": true,
        "op": "capabilities",
        "protocol": PROTOCOL_VERSION,
        "features": FEATURES,
        "journal_format": FORMAT_VERSION,
        "snapshot_format": SNAPSHOT_FORMAT,
    })
}

fn source_items(items: Vec<ItemRequest>) -> Vec<SourceItem> {
    items
        .into_iter()
        .map(|item| SourceItem {
            key: item.key,
            version: item.version,
        })
        .collect()
}

fn stage_spec(request: &OpenRequest) -> StageSpec {
    StageSpec {
        stage: request.stage.clone(),
        source_id: request.source_id.clone(),
        fingerprint: request.fingerprint.clone(),
        retry_fingerprint: request.retry_fingerprint.clone(),
        retry_failed: request.retry_failed,
        output_root: request.output_root.clone(),
        full: request.full,
        verify: request.verify,
        wait: request.wait,
    }
}

fn open_run(state_dir: &Path, open: OpenRequest) -> Result<(StageRun, Value)> {
    let spec = stage_spec(&open);
    let unknown = open.unknown.clone();
    let (run, report) = StageRun::open(state_dir, spec, source_items(open.items), unknown)?;
    let response = json!({
        "ok": true,
        "op": "open",
        "run_token": report.run_token,
        "resumed": report.resumed,
        "counts": report.counts,
        "decisions": report.decisions.iter().map(decision_json).collect::<Vec<_>>(),
    });
    Ok((run, response))
}

/// Read-only: what a run of the same request would decide.
fn plan_run(state_dir: &Path, open: OpenRequest) -> Result<Value> {
    let spec = stage_spec(&open);
    let unknown = open.unknown;
    let preview = view::preview(state_dir, &spec, &source_items(open.items), &unknown)?;
    Ok(json!({
        "ok": true,
        "op": "plan",
        "counts": preview.counts,
        "decisions": preview.decisions.iter().map(decision_json).collect::<Vec<_>>(),
    }))
}

/// Read-only: what the stage durably holds.
fn inspect_run(state_dir: &Path, request: InspectRequest) -> Result<Value> {
    let inspection = view::inspect(
        state_dir,
        &request.stage,
        &request.source_id,
        &request.output_root,
        request.verify,
    )?;
    // `mtime_ns` goes out as a decimal string: a nanosecond timestamp does
    // not fit a JSON number without losing precision in a JavaScript host.
    let items: Vec<Value> = inspection
        .items
        .iter()
        .map(|item| {
            json!({
                "key": item.key,
                "version": item.version,
                "fingerprint": item.fingerprint,
                "output": {
                    "path": item.output.path,
                    "size": item.output.size,
                    "digest": item.output.digest,
                    "mtime_ns": item.output.mtime_ns.to_string(),
                },
                "updated_at": item.updated_at,
            })
        })
        .collect();
    let failed: Vec<Value> = inspection
        .failed
        .iter()
        .map(|failure| {
            json!({
                "key": failure.key,
                "version": failure.version,
                "error": failure.error,
                "deferred": failure.deferred,
            })
        })
        .collect();
    let skipped: Vec<Value> = inspection
        .skipped
        .iter()
        .map(|skipped| {
            json!({
                "key": skipped.key,
                "version": skipped.version,
                "reason": skipped.reason,
                "fingerprint": skipped.fingerprint,
            })
        })
        .collect();
    // The roster of every key the stage holds, independent of whether its
    // output verified or its failure is held: a host uses it to keep state it
    // cannot currently observe, never as a substitute for a verified output.
    let known: Vec<Value> = inspection
        .known
        .iter()
        .map(|known| json!({"key": known.key, "version": known.version}))
        .collect();
    Ok(json!({
        "ok": true,
        "op": "inspect",
        "items": items,
        "failed": failed,
        "skipped": skipped,
        "known": known,
    }))
}

/// Answer a request that works on an open run and does not end it.
fn handle(run: &mut StageRun, request: Request) -> Value {
    let op = request.op();
    match request {
        Request::Commit(commit) => {
            let claim = OutputClaim {
                path: commit.output.path,
                size: commit.output.size,
                digest: commit.output.digest,
            };
            match run.commit(
                &commit.key,
                &commit.version,
                commit.fingerprint.as_deref(),
                &claim,
            ) {
                Ok(recorded) => json!({
                    "ok": true,
                    "op": op,
                    "key": commit.key,
                    "output": {
                        "path": recorded.path,
                        "size": recorded.size,
                        "digest": recorded.digest,
                    },
                }),
                Err(error) => refusal(op, &error),
            }
        }
        Request::Fail(fail) => {
            let options = FailOptions {
                defer: fail.defer,
                discard_output: fail.discard_output,
            };
            match run.fail(
                &fail.key,
                &fail.version,
                fail.fingerprint.as_deref(),
                &fail.error,
                options,
            ) {
                Ok(()) => json!({"ok": true, "op": op, "key": fail.key}),
                Err(error) => refusal(op, &error),
            }
        }
        Request::Skip(skip) => match run.skip(&skip.key, &skip.version, &skip.reason) {
            Ok(()) => json!({"ok": true, "op": op, "key": skip.key}),
            Err(error) => refusal(op, &error),
        },
        Request::Delete { key } => match run.delete(&key) {
            Ok(removed) => json!({
                "ok": true,
                "op": op,
                "key": key,
                "output_removed": removed,
            }),
            Err(error) => refusal(op, &error),
        },
        Request::Capabilities
        | Request::Open(_)
        | Request::Plan(_)
        | Request::Inspect(_)
        | Request::Finish => refusal(
            op,
            &MinSyncError::Other(format!("`{op}` is not handled on an open run")),
        ),
    }
}

/// Run one session over `input`/`output`, returning the process exit code.
/// `state_dir` holds the stage's lock, run marker, journal and snapshot.
pub fn serve(state_dir: &Path, input: impl BufRead, output: &mut impl Write) -> Result<i32> {
    let mut run: Option<StageRun> = None;

    for line in input.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let request: Request = match serde_json::from_str(&line) {
            Ok(request) => request,
            Err(error) => {
                let error = MinSyncError::stage("invalid_request", error.to_string());
                respond(output, &refusal("unknown", &error))?;
                continue;
            }
        };

        // The read-only requests never touch the stage lock or its state, so
        // they are answered the same way before, during and after a run.
        if matches!(request, Request::Capabilities) {
            respond(output, &capabilities())?;
            continue;
        }
        if let Request::Open(open) = request {
            if run.is_some() {
                let error = MinSyncError::stage("already_open", "a run is already open");
                respond(output, &refusal("open", &error))?;
                continue;
            }
            match open_run(state_dir, open) {
                Ok((opened, response)) => {
                    respond(output, &response)?;
                    run = Some(opened);
                }
                Err(error) => {
                    respond(output, &refusal("open", &error))?;
                    if matches!(error, MinSyncError::LockFailed) {
                        return Ok(error.exit_code());
                    }
                }
            }
            continue;
        }
        if let Request::Plan(open) = request {
            let response = match plan_run(state_dir, open) {
                Ok(response) => response,
                Err(error) => refusal("plan", &error),
            };
            respond(output, &response)?;
            continue;
        }
        if let Request::Inspect(request) = request {
            let response = match inspect_run(state_dir, request) {
                Ok(response) => response,
                Err(error) => refusal("inspect", &error),
            };
            respond(output, &response)?;
            continue;
        }

        let op = request.op();
        let Some(active) = run.as_mut() else {
            let error = MinSyncError::stage("not_open", "send `open` first");
            respond(output, &refusal(op, &error))?;
            continue;
        };
        if !matches!(request, Request::Finish) {
            respond(output, &handle(active, request))?;
            continue;
        }

        let Some(finishing) = run.take() else {
            continue;
        };
        return match finishing.finish() {
            Ok(report) => {
                respond(
                    output,
                    &json!({
                        "ok": true,
                        "op": op,
                        "committed": report.committed,
                        "deleted": report.deleted,
                        "failed": report.failed,
                        "pending": report.pending,
                    }),
                )?;
                Ok(0)
            }
            Err(error) => {
                respond(output, &refusal(op, &error))?;
                Ok(error.exit_code())
            }
        };
    }

    if run.is_some() {
        tracing::warn!("input ended before `finish`; the run was kept and will resume");
        return Ok(1);
    }
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::incremental::journal::JOURNAL_FILE;
    use crate::incremental::snapshot::SNAPSHOT_FILE;
    use serde_json::{json, Value};
    use std::io::Cursor;

    struct Env {
        _dir: tempfile::TempDir,
        state: PathBuf,
        out: PathBuf,
    }

    fn env() -> Env {
        let dir = tempfile::tempdir().expect("tempdir");
        let out = dir.path().join("out");
        std::fs::create_dir_all(&out).expect("output root");
        Env {
            state: dir.path().join("state"),
            out,
            _dir: dir,
        }
    }

    fn open_request(env: &Env, items: &[(&str, &str)]) -> Value {
        json!({
            "op": "open",
            "stage": "parse",
            "source_id": "corpus",
            "fingerprint": "parser-1",
            "output_root": env.out,
            "items": items.iter().map(|(key, version)| json!({"key": key, "version": version})).collect::<Vec<_>>(),
        })
    }

    fn commit_request(key: &str, version: &str, path: &str) -> Value {
        json!({"op": "commit", "key": key, "version": version, "output": {"path": path}})
    }

    fn write_output(env: &Env, rel: &str, content: &str) {
        std::fs::write(env.out.join(rel), content).expect("write output");
    }

    /// Feed `requests` as one session and return the responses and exit code.
    fn session(env: &Env, requests: &[Value]) -> (Vec<Value>, i32) {
        let input: String = requests
            .iter()
            .map(|request| format!("{request}\n"))
            .collect();
        session_raw(env, &input)
    }

    fn session_raw(env: &Env, input: &str) -> (Vec<Value>, i32) {
        let mut output = Vec::new();
        let code = serve(&env.state, Cursor::new(input.to_string()), &mut output)
            .unwrap_or_else(|error| panic!("serve: {error}"));
        let responses = String::from_utf8(output)
            .expect("utf8")
            .lines()
            .map(|line| serde_json::from_str(line).unwrap_or_else(|e| panic!("`{line}`: {e}")))
            .collect();
        (responses, code)
    }

    fn error_code(response: &Value) -> &str {
        assert_eq!(response["ok"], false, "{response}");
        response["error"]["code"].as_str().expect("error code")
    }

    #[test]
    fn a_full_session_opens_commits_and_finishes() {
        let env = env();
        write_output(&env, "a.md", "alpha");
        write_output(&env, "b.md", "beta");
        let (responses, code) = session(
            &env,
            &[
                open_request(&env, &[("a", "v1"), ("b", "v1")]),
                commit_request("a", "v1", "a.md"),
                commit_request("b", "v1", "b.md"),
                json!({"op": "finish"}),
            ],
        );
        assert_eq!(code, 0);
        assert_eq!(responses.len(), 4);

        let open = &responses[0];
        assert_eq!(open["ok"], true);
        assert_eq!(open["op"], "open");
        assert_eq!(open["resumed"], false);
        assert!(open["run_token"].as_str().is_some_and(|t| !t.is_empty()));
        assert_eq!(open["counts"]["process"], 2);
        assert_eq!(
            open["decisions"],
            json!([
                {"key": "a", "action": "process", "version": "v1"},
                {"key": "b", "action": "process", "version": "v1"},
            ])
        );

        assert_eq!(responses[1]["ok"], true);
        assert_eq!(responses[1]["output"]["path"], "a.md");
        assert_eq!(responses[1]["output"]["size"], 5);
        assert!(responses[1]["output"]["digest"]
            .as_str()
            .is_some_and(|d| d.starts_with("sha256:")));

        assert_eq!(
            responses[3],
            json!({"ok": true, "op": "finish", "committed": 2, "deleted": 0, "failed": 0, "pending": 0})
        );

        // The next session finds nothing to do.
        let (responses, _) = session(
            &env,
            &[
                open_request(&env, &[("a", "v1"), ("b", "v1")]),
                json!({"op": "finish"}),
            ],
        );
        assert_eq!(responses[0]["decisions"], json!([]));
        assert_eq!(responses[0]["counts"]["skip"], 2);
    }

    #[test]
    fn a_refused_request_gets_a_coded_error_and_the_session_continues() {
        let env = env();
        write_output(&env, "a.md", "alpha");
        let (responses, code) = session(
            &env,
            &[
                open_request(&env, &[("a", "v1")]),
                commit_request("a", "v1", "../escape.md"),
                commit_request("a", "v0", "a.md"),
                commit_request("ghost", "v1", "a.md"),
                commit_request("a", "v1", "a.md"),
                json!({"op": "finish"}),
            ],
        );
        assert_eq!(code, 0);
        assert_eq!(error_code(&responses[1]), "invalid_output_path");
        assert_eq!(responses[1]["op"], "commit");
        assert_eq!(error_code(&responses[2]), "version_mismatch");
        assert_eq!(error_code(&responses[3]), "unknown_item");
        assert_eq!(responses[4]["ok"], true);
        assert_eq!(responses[5]["committed"], 1);
    }

    #[test]
    fn requests_out_of_order_or_malformed_are_refused_without_ending_the_session() {
        let env = env();
        let input = format!(
            "{}\n\nnot json\n{}\n{}\n{}\n{}\n",
            json!({"op": "commit", "key": "a", "version": "v1", "output": {"path": "a.md"}}),
            json!({"op": "teleport"}),
            open_request(&env, &[("a", "v1")]),
            open_request(&env, &[("a", "v1")]),
            json!({"op": "finish"}),
        );
        let (responses, code) = session_raw(&env, &input);
        assert_eq!(code, 0);
        assert_eq!(error_code(&responses[0]), "not_open");
        assert_eq!(error_code(&responses[1]), "invalid_request");
        assert_eq!(error_code(&responses[2]), "invalid_request");
        assert_eq!(responses[3]["ok"], true);
        assert_eq!(error_code(&responses[4]), "already_open");
        assert_eq!(responses[5]["op"], "finish");
    }

    #[test]
    fn ending_the_input_without_finish_leaves_a_resumable_run() {
        let env = env();
        write_output(&env, "a.md", "alpha");
        let (_, code) = session(
            &env,
            &[
                open_request(&env, &[("a", "v1"), ("b", "v1")]),
                commit_request("a", "v1", "a.md"),
            ],
        );
        assert_eq!(code, 1, "an unfinished run is not a success");
        assert!(env.state.join(JOURNAL_FILE).exists());
        assert!(env.state.join("txn.json").exists());
        assert!(!env.state.join("snapshot.json").exists());

        let (responses, code) = session(
            &env,
            &[
                open_request(&env, &[("a", "v1"), ("b", "v1")]),
                json!({"op": "finish"}),
            ],
        );
        assert_eq!(code, 0);
        assert_eq!(responses[0]["resumed"], true);
        assert_eq!(responses[0]["counts"]["resumed"], 1);
        assert_eq!(
            responses[0]["decisions"],
            json!([{"key": "b", "action": "process", "version": "v1"}])
        );
    }

    #[test]
    fn a_held_lock_ends_the_session_with_the_lock_exit_code() {
        let env = env();
        let first = StageRun::open(
            &env.state,
            StageSpec {
                stage: "parse".into(),
                source_id: "corpus".into(),
                fingerprint: "parser-1".into(),
                retry_fingerprint: None,
                retry_failed: false,
                output_root: env.out.clone(),
                full: false,
                verify: VerifyMode::Fast,
                wait: false,
            },
            vec![],
            vec![],
        )
        .expect("first run");

        let (responses, code) = session(&env, &[open_request(&env, &[("a", "v1")])]);
        assert_eq!(code, 3);
        assert_eq!(error_code(&responses[0]), "lock_failed");
        drop(first);
    }

    #[test]
    fn plan_and_inspect_answer_without_opening_a_run_or_touching_state() {
        let env = env();
        write_output(&env, "a.md", "alpha");
        session(
            &env,
            &[
                open_request(&env, &[("a", "v1")]),
                commit_request("a", "v1", "a.md"),
                json!({"op": "finish"}),
            ],
        );
        let snapshot = std::fs::read(env.state.join(SNAPSHOT_FILE)).expect("snapshot");

        // Read-only requests need no `open`, and leave every file alone.
        let inspect = json!({
            "op": "inspect",
            "stage": "parse",
            "source_id": "corpus",
            "output_root": env.out,
        });
        let plan = json!({
            "op": "plan",
            "stage": "parse",
            "source_id": "corpus",
            "fingerprint": "parser-1",
            "output_root": env.out,
            "items": [{"key": "a", "version": "v1"}],
        });
        let wrong_stage = json!({
            "op": "inspect",
            "stage": "chunk",
            "source_id": "corpus",
            "output_root": env.out,
        });
        let (responses, code) = session(&env, &[inspect, plan, wrong_stage]);
        assert_eq!(code, 0, "a read-only session ends cleanly");

        let inspect = &responses[0];
        assert_eq!(inspect["ok"], true);
        assert_eq!(inspect["op"], "inspect");
        assert_eq!(inspect["items"][0]["key"], "a");
        assert_eq!(inspect["items"][0]["version"], "v1");
        assert_eq!(inspect["items"][0]["fingerprint"], "parser-1");
        assert_eq!(inspect["items"][0]["output"]["path"], "a.md");
        assert_eq!(inspect["items"][0]["output"]["size"], 5);
        assert!(inspect["items"][0]["output"]["digest"]
            .as_str()
            .is_some_and(|digest| digest.starts_with("sha256:")));
        assert!(
            inspect["items"][0]["output"]["mtime_ns"].is_string(),
            "the mtime is a decimal string on the wire: {}",
            inspect["items"][0]["output"]["mtime_ns"]
        );
        assert!(inspect["items"][0]["updated_at"].is_string());
        assert_eq!(inspect["failed"], json!([]));
        assert_eq!(inspect["skipped"], json!([]));
        assert_eq!(
            inspect["known"],
            json!([{"key": "a", "version": "v1"}]),
            "the roster is always present"
        );

        assert_eq!(responses[1]["op"], "plan");
        assert_eq!(
            responses[1]["decisions"],
            json!([{"key": "a", "action": "skip", "version": "v1"}])
        );
        assert_eq!(responses[1]["counts"]["skip"], 1);
        assert_eq!(
            error_code(&responses[2]),
            "source_mismatch",
            "another stage's state is not ours to describe"
        );

        assert!(
            !env.state.join("txn.json").exists(),
            "a read-only request writes no run marker"
        );
        assert!(
            !env.state.join(JOURNAL_FILE).exists(),
            "a read-only request writes no journal"
        );
        assert_eq!(
            std::fs::read(env.state.join(SNAPSHOT_FILE)).expect("snapshot"),
            snapshot,
            "a read-only request leaves the snapshot byte for byte"
        );
    }

    #[test]
    fn capabilities_identify_this_build_without_touching_state() {
        let env = env();
        let (responses, code) = session(&env, &[json!({"op": "capabilities"})]);
        assert_eq!(code, 0);
        let response = &responses[0];
        assert_eq!(response["ok"], true);
        assert_eq!(response["op"], "capabilities");
        assert_eq!(response["protocol"], 2);
        assert_eq!(response["journal_format"], 3);
        assert_eq!(response["snapshot_format"], 2);
        assert_eq!(
            response["features"],
            json!([
                "retry_fingerprint",
                "retry_failed",
                "deferred_failure",
                "discard_output",
                "deliberate_skip",
                "plan",
                "inspect",
            ])
        );
        assert!(!env.state.exists(), "probing creates no state directory");
    }

    #[test]
    fn read_only_requests_on_a_missing_stage_report_nothing_and_create_nothing() {
        let env = env();
        let inspect = json!({
            "op": "inspect",
            "stage": "parse",
            "source_id": "corpus",
            "output_root": env.out,
        });
        let plan = json!({
            "op": "plan",
            "stage": "parse",
            "source_id": "corpus",
            "fingerprint": "parser-1",
            "output_root": env.out,
            "items": [{"key": "a", "version": "v1"}],
        });
        let (responses, code) = session(&env, &[inspect, plan]);
        assert_eq!(code, 0);
        assert_eq!(responses[0]["items"], json!([]));
        assert_eq!(responses[0]["failed"], json!([]));
        assert_eq!(responses[0]["skipped"], json!([]));
        assert_eq!(
            responses[0]["known"],
            json!([]),
            "the roster is present even when nothing is held"
        );
        assert_eq!(
            responses[1]["decisions"],
            json!([{"key": "a", "action": "process", "version": "v1"}])
        );
        assert!(
            !env.state.exists(),
            "a read-only request creates no state directory"
        );
    }

    /// `known` is the raw roster of the durable state: a completion whose
    /// output no longer verifies and a failure recorded before versions were
    /// kept are both listed, the latter with `null`.
    #[test]
    fn inspect_reports_the_known_roster_including_unverified_and_legacy_failures() {
        let env = env();
        std::fs::create_dir_all(&env.state).expect("state dir");
        // A format-1 snapshot, as an older build wrote it: no omissions, a
        // failure without a version, and a completion whose output is gone.
        let legacy = json!({
            "format": 1,
            "source_id": "corpus",
            "stage": "parse",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "items": {"a": {
                "version": "a1",
                "fingerprint": "parser-1",
                "output": {"path": "a.md", "size": 5, "digest": "sha256:missing", "mtime_ns": 0},
            }},
            "failed": {
                "c": {"version": "c1", "error": "corrupt"},
                "e": {"error": "legacy"},
            },
        });
        std::fs::write(env.state.join(SNAPSHOT_FILE), legacy.to_string()).expect("write snapshot");

        let inspect = json!({
            "op": "inspect",
            "stage": "parse",
            "source_id": "corpus",
            "output_root": env.out,
        });
        let (responses, code) = session(&env, &[inspect]);
        assert_eq!(code, 0);
        assert_eq!(
            responses[0]["items"],
            json!([]),
            "a.md does not exist, so the completion is not live"
        );
        assert_eq!(
            responses[0]["known"],
            json!([
                {"key": "a", "version": "a1"},
                {"key": "c", "version": "c1"},
                {"key": "e", "version": null},
            ])
        );
        assert!(
            !env.state.join("lock").exists(),
            "listing the roster takes no lock"
        );
    }

    #[test]
    fn a_deferred_failure_is_held_and_reported_and_a_skip_is_accepted() {
        let env = env();
        write_output(&env, "a.md", "alpha");
        let hold = json!({
            "op": "fail",
            "key": "b",
            "version": "v1",
            "error": "bad xref",
            "defer": true,
        });
        let (responses, code) = session(
            &env,
            &[
                open_request(&env, &[("a", "v1"), ("b", "v1")]),
                commit_request("a", "v1", "a.md"),
                hold,
                json!({"op": "finish"}),
            ],
        );
        assert_eq!(code, 0);
        assert_eq!(responses[2], json!({"ok": true, "op": "fail", "key": "b"}));
        assert_eq!(responses[3]["failed"], 1);
        assert_eq!(responses[3]["pending"], 0, "a held failure is not work");

        // The next run holds the failure, keeps the item out of the work
        // list, and says why.
        let (responses, _) = session(
            &env,
            &[
                open_request(&env, &[("a", "v1"), ("b", "v1")]),
                json!({"op": "finish"}),
            ],
        );
        assert_eq!(responses[0]["counts"]["skip"], 2);
        assert_eq!(
            responses[0]["decisions"],
            json!([{"key": "b", "action": "skip", "version": "v1",
                    "deferred": true, "error": "bad xref"}])
        );

        // `retry_failed` releases it.
        let mut open = open_request(&env, &[("a", "v1"), ("b", "v1")]);
        open["retry_failed"] = json!(true);
        let (responses, _) = session(&env, &[open, json!({"op": "finish"})]);
        assert_eq!(
            responses[0]["decisions"],
            json!([{"key": "b", "action": "retry", "version": "v1"}])
        );

        // A deliberate omission is accepted and removes the old output.
        let (responses, code) = session(
            &env,
            &[
                open_request(&env, &[("a", "v2"), ("b", "v1")]),
                json!({"op": "skip", "key": "a", "version": "v2", "reason": "unsupported type"}),
                json!({"op": "finish"}),
            ],
        );
        assert_eq!(code, 0);
        assert_eq!(responses[1], json!({"ok": true, "op": "skip", "key": "a"}));
        assert!(
            !env.out.join("a.md").exists(),
            "an omission removes the output a completion left behind"
        );

        // inspect reports exactly what the stage now holds: no live output,
        // the held failure, and the omission.
        let inspect = json!({
            "op": "inspect",
            "stage": "parse",
            "source_id": "corpus",
            "output_root": env.out,
        });
        let (responses, _) = session(&env, &[inspect]);
        assert_eq!(responses[0]["items"], json!([]));
        assert_eq!(
            responses[0]["failed"],
            json!([{"key": "b", "version": "v1", "error": "bad xref", "deferred": true}])
        );
        assert_eq!(
            responses[0]["skipped"],
            json!([{"key": "a", "version": "v2", "reason": "unsupported type",
                    "fingerprint": "parser-1"}])
        );
        assert_eq!(
            responses[0]["known"],
            json!([
                {"key": "a", "version": "v2"},
                {"key": "b", "version": "v1"},
            ]),
            "a held failure and an omission are both identities"
        );
    }

    #[test]
    fn a_discarding_failure_removes_the_output_and_the_completion() {
        let env = env();
        write_output(&env, "a.md", "alpha");
        let (_, code) = session(
            &env,
            &[
                open_request(&env, &[("a", "v1")]),
                commit_request("a", "v1", "a.md"),
                json!({"op": "finish"}),
            ],
        );
        assert_eq!(code, 0);
        assert!(env.out.join("a.md").exists());

        let discard = json!({
            "op": "fail",
            "key": "a",
            "version": "v2",
            "error": "corrupt",
            "discard_output": true,
        });
        let (responses, code) = session(
            &env,
            &[
                open_request(&env, &[("a", "v2")]),
                discard,
                json!({"op": "finish"}),
            ],
        );
        assert_eq!(code, 0);
        assert_eq!(responses[1], json!({"ok": true, "op": "fail", "key": "a"}));
        assert!(!env.out.join("a.md").exists(), "the output is discarded");

        let inspect = json!({
            "op": "inspect",
            "stage": "parse",
            "source_id": "corpus",
            "output_root": env.out,
        });
        let (responses, _) = session(&env, &[inspect]);
        assert_eq!(
            responses[0]["items"],
            json!([]),
            "the completion went with the output"
        );
        assert_eq!(
            responses[0]["failed"],
            json!([{"key": "a", "version": "v2", "error": "corrupt", "deferred": false}])
        );
    }

    #[test]
    fn delete_fail_unknown_full_and_verify_are_wired_through() {
        let env = env();
        write_output(&env, "a.md", "alpha");
        write_output(&env, "b.md", "beta");
        session(
            &env,
            &[
                open_request(&env, &[("a", "v1"), ("b", "v1")]),
                commit_request("a", "v1", "a.md"),
                commit_request("b", "v1", "b.md"),
                json!({"op": "finish"}),
            ],
        );

        let mut open = open_request(&env, &[("a", "v2")]);
        open["full"] = json!(true);
        open["verify"] = json!("digest");
        open["unknown"] = json!(["kept"]);
        let (responses, code) = session(
            &env,
            &[
                open,
                json!({"op": "fail", "key": "a", "version": "v2", "error": "parser crashed"}),
                json!({"op": "delete", "key": "b"}),
                json!({"op": "delete", "key": "b"}),
                json!({"op": "delete", "key": "a"}),
                json!({"op": "delete", "key": "kept"}),
                json!({"op": "finish"}),
            ],
        );
        assert_eq!(code, 0);
        assert_eq!(
            responses[0]["decisions"],
            json!([
                {"key": "a", "action": "process", "version": "v2"},
                {"key": "b", "action": "delete"},
            ])
        );
        assert_eq!(responses[0]["counts"]["preserve"], 1);
        assert_eq!(responses[1], json!({"ok": true, "op": "fail", "key": "a"}));
        assert_eq!(
            responses[2],
            json!({"ok": true, "op": "delete", "key": "b", "output_removed": true})
        );
        assert_eq!(responses[3]["output_removed"], false);
        assert_eq!(error_code(&responses[4]), "still_present");
        assert_eq!(error_code(&responses[5]), "preserved_item");
        assert_eq!(responses[6]["failed"], 1);
        assert_eq!(responses[6]["deleted"], 1);
        assert!(!env.out.join("b.md").exists());
        assert!(env.out.join("a.md").exists(), "a failure keeps old output");
    }
}
