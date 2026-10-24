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
//!     "unknown":["b.pdf"],"full":false,"verify":"fast","wait":false}
//! <- {"ok":true,"op":"open","run_token":"..","resumed":false,
//!     "counts":{"process":1,"retry":0,"skip":0,"resumed":0,"delete":0,"preserve":1},
//!     "decisions":[{"key":"a.pdf","action":"process","version":"sha256:.."}]}
//! -> {"op":"commit","key":"a.pdf","version":"sha256:..",
//!     "output":{"path":"a.pdf.md","size":123,"digest":"sha256:.."}}
//! <- {"ok":true,"op":"commit","key":"a.pdf","output":{"path":"a.pdf.md","size":123,"digest":".."}}
//! -> {"op":"fail","key":"c.pdf","version":"..","error":"parser crashed"}
//! -> {"op":"delete","key":"gone.pdf"}
//! -> {"op":"finish"}
//! <- {"ok":true,"op":"finish","committed":1,"deleted":0,"failed":0,"pending":0}
//! ```
//!
//! A refused request is answered with
//! `{"ok":false,"op":"..","error":{"code":"..","message":".."}}` and the
//! session goes on. A lock held by another run ends the session
//! (exit code 3), a clean `finish` exits 0, and input that ends mid-run
//! exits 1 with the run left resumable.

use super::plan::SourceItem;
use super::stage::{OutputClaim, StageRun, StageSpec, VerifyMode};
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
}

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum Request {
    Open(OpenRequest),
    Commit(CommitRequest),
    Fail(FailRequest),
    Delete { key: String },
    Finish,
}

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
    value
}

impl Request {
    fn op(&self) -> &'static str {
        match self {
            Request::Open(_) => "open",
            Request::Commit(_) => "commit",
            Request::Fail(_) => "fail",
            Request::Delete { .. } => "delete",
            Request::Finish => "finish",
        }
    }
}

fn open_run(state_dir: &Path, open: OpenRequest) -> Result<(StageRun, Value)> {
    let spec = StageSpec {
        stage: open.stage,
        source_id: open.source_id,
        fingerprint: open.fingerprint,
        output_root: open.output_root,
        full: open.full,
        verify: open.verify,
        wait: open.wait,
    };
    let items = open
        .items
        .into_iter()
        .map(|item| SourceItem {
            key: item.key,
            version: item.version,
        })
        .collect();
    let (run, report) = StageRun::open(state_dir, spec, items, open.unknown)?;
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
            match run.fail(
                &fail.key,
                &fail.version,
                fail.fingerprint.as_deref(),
                &fail.error,
            ) {
                Ok(()) => json!({"ok": true, "op": op, "key": fail.key}),
                Err(error) => refusal(op, &error),
            }
        }
        Request::Delete { key } => match run.delete(&key) {
            Ok(removed) => json!({
                "ok": true,
                "op": op,
                "key": key,
                "output_removed": removed,
            }),
            Err(error) => refusal(op, &error),
        },
        Request::Open(_) | Request::Finish => refusal(
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
