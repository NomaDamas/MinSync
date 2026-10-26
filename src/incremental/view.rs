//! Read-only views of a stage's durable state: `plan` (what a run would do)
//! and `inspect` (what the stage durably holds right now).
//!
//! Both reconstruct the state from the committed snapshot plus the journal
//! of a run that was interrupted or is still active, verify outputs against
//! the output root, and change nothing: no lock is taken or held, no state
//! directory is created, and no marker, journal or snapshot is written. A
//! host can ask before opening a run, or from another process while a run is
//! in progress.
//!
//! The journal is trusted only when it describes the same stage and corpus
//! and diffed against the snapshot's own baseline. A journal from an
//! unrelated baseline is the stale leftover of a run whose results the
//! snapshot already folded in; counting it would resurrect records the
//! snapshot superseded. A run that *finishes* while the two are being read
//! rewrites the snapshot and then removes the journal, so the snapshot is read
//! on both sides of the journal and a capture that straddled a finish is
//! retried (see [`capture`]).

use super::journal::{Journal, Ledger, OutputRef, RunHeader, JOURNAL_FILE};
use super::plan::{plan, Decision, PlanInput, SourceItem};
use super::snapshot::{Snapshot, SNAPSHOT_FILE};
use super::stage::{
    mtime_ns, output_matches, tally, validate_items, validate_spec, Counts, StageSpec, VerifyMode,
};
use crate::error::{MinSyncError, Result};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// What a run would do, decided but not performed.
pub struct Preview {
    pub counts: Counts,
    pub decisions: Vec<Decision>,
}

/// The durable state of a stage, as `inspect` reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Inspection {
    pub items: Vec<InspectedItem>,
    pub failed: Vec<InspectedFailure>,
    pub skipped: Vec<InspectedSkip>,
    /// Every key the durable state holds, whatever happened to its output.
    pub known: Vec<InspectedKnown>,
}

/// A committed item whose output is still intact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InspectedItem {
    pub key: String,
    pub version: String,
    pub fingerprint: String,
    pub output: OutputRef,
    /// RFC 3339, stable per output: the commit mtime the output was recorded
    /// with, or the file's own mtime for a record written before mtimes were
    /// stored.
    pub updated_at: String,
}

/// A failure the stage holds, deferred or not.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InspectedFailure {
    pub key: String,
    pub version: Option<String>,
    pub error: String,
    /// The failure is held: it will not be retried until the host changes
    /// the source, its settings or the retry policy, or asks for a retry.
    pub deferred: bool,
}

/// An item the host deliberately omitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InspectedSkip {
    pub key: String,
    pub version: String,
    pub fingerprint: String,
    pub reason: String,
}

/// A key the durable state holds, and the source version it holds it under.
///
/// This is the roster a host needs to tell an item it already knows about
/// from an item it has never seen, so an unavailable root's state is
/// preserved rather than rebuilt from an empty baseline. It is the union of
/// every completion, omission and failure, including a completion whose
/// output no longer verifies and a failure that is deferred. `version` is
/// `None` only for a failure recorded before versions were kept.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InspectedKnown {
    pub key: String,
    pub version: Option<String>,
}

/// Decide what a run of `spec` would do without opening one.
pub fn preview(
    state_dir: &Path,
    spec: &StageSpec,
    items: &[SourceItem],
    unknown: &[String],
) -> Result<Preview> {
    validate_spec(spec)?;
    let (_items, unknown_set) = validate_items(items, unknown.to_vec())?;
    let (snapshot, journal) = capture(state_dir, &spec.stage, &spec.source_id)?;
    let expected = RunHeader::new(
        &spec.stage,
        &spec.source_id,
        spec.full,
        &snapshot.hash(),
        &spec.fingerprint,
    );
    let ledger = journal
        .filter(|ledger| ledger.header.resumes(&expected))
        .unwrap_or_else(|| Ledger::new(expected));
    let root = canonical_root(&spec.output_root);
    let verify = spec.verify;
    let decisions = plan(&PlanInput {
        items,
        unknown: &unknown_set,
        fingerprint: &spec.fingerprint,
        retry_fingerprint: spec.retry_fingerprint.as_deref(),
        full: spec.full,
        retry_failed: spec.retry_failed,
        snapshot: &snapshot,
        ledger: &ledger,
        output_ok: &|output| output_matches(&root, output, verify),
    });
    let mut counts = Counts::default();
    for decision in &decisions {
        tally(&mut counts, decision);
    }
    Ok(Preview { counts, decisions })
}

/// Reconstruct what the stage durably holds, verifying every output: the
/// items whose output still verifies, every failure, every omission, and the
/// roster of every key the state holds at all (see [`InspectedKnown`]).
pub fn inspect(
    state_dir: &Path,
    stage: &str,
    source_id: &str,
    output_root: &Path,
    verify: VerifyMode,
) -> Result<Inspection> {
    let (mut durable, ledger) = capture(state_dir, stage, source_id)?;
    if let Some(ledger) = ledger {
        let fingerprint = ledger.header.fingerprint.clone();
        for record in ledger.records() {
            durable.apply_record(record, &fingerprint);
        }
    }
    let root = canonical_root(output_root);
    let items = durable
        .items
        .iter()
        .filter_map(|(key, item)| {
            let nanos = verified_mtime(&root, &item.output, verify)?;
            Some(InspectedItem {
                key: key.clone(),
                version: item.version.clone(),
                fingerprint: item.fingerprint.clone(),
                output: item.output.clone(),
                updated_at: rfc3339(nanos),
            })
        })
        .collect();
    let failed = durable
        .failed
        .iter()
        .map(|(key, failure)| InspectedFailure {
            key: key.clone(),
            version: failure.version.clone(),
            error: failure.error.clone(),
            deferred: failure.deferred,
        })
        .collect();
    let skipped = durable
        .skipped
        .iter()
        .map(|(key, skipped)| InspectedSkip {
            key: key.clone(),
            version: skipped.version.clone(),
            fingerprint: skipped.fingerprint.clone(),
            reason: skipped.reason.clone(),
        })
        .collect();
    // Every key the state holds, whatever happened to its output, so a host
    // can preserve what it cannot currently observe. A key can be both
    // completed and failed; the version is the most recent claim about it:
    // the failure, then the omission, then the completion.
    let version_of = |key: &String| {
        durable
            .failed
            .get(key)
            .and_then(|failure| failure.version.clone())
            .or_else(|| {
                durable
                    .skipped
                    .get(key)
                    .map(|skipped| skipped.version.clone())
            })
            .or_else(|| durable.items.get(key).map(|item| item.version.clone()))
    };
    let mut durable_keys: BTreeSet<&String> = BTreeSet::new();
    durable_keys.extend(durable.items.keys());
    durable_keys.extend(durable.skipped.keys());
    durable_keys.extend(durable.failed.keys());
    let known = durable_keys
        .into_iter()
        .map(|key| InspectedKnown {
            key: key.clone(),
            version: version_of(key),
        })
        .collect();
    Ok(Inspection {
        items,
        failed,
        skipped,
        known,
    })
}

/// How many times a capture is retried when a `finish` keeps landing inside
/// it.
const CAPTURE_ATTEMPTS: usize = 3;

/// The committed snapshot and the ledger of the run that is still running or
/// was interrupted, read as one consistent pair.
///
/// `finish` rewrites the snapshot first and removes the journal second, so a
/// capture reads the snapshot on both sides of the journal and retries when it
/// moved in between: without that, a host could be handed the state the run it
/// just missed replaced. A capture that keeps straddling finishes answers from
/// the snapshot alone — a state the stage really held, whose outputs the
/// caller verifies anyway — and never from a journal that does not belong to
/// it. No lock is taken at any point.
fn capture(state_dir: &Path, stage: &str, source_id: &str) -> Result<(Snapshot, Option<Ledger>)> {
    let path = state_dir.join(SNAPSHOT_FILE);
    let mut revision = std::fs::read(&path).ok();
    let mut snapshot = load_snapshot(state_dir, stage, source_id)?;
    for _ in 0..CAPTURE_ATTEMPTS {
        // The snapshot has just been read. A `finish` that lands from here on
        // — saving the new snapshot and removing the journal — is what the
        // second read below detects.
        #[cfg(test)]
        capture_hook();
        let ledger = active_ledger(state_dir, &snapshot, stage, source_id);
        let after = std::fs::read(&path).ok();
        if after == revision {
            return Ok((snapshot, ledger));
        }
        revision = after;
        snapshot = load_snapshot(state_dir, stage, source_id)?;
    }
    tracing::debug!(
        state_dir = %state_dir.display(),
        "the snapshot kept moving while it was captured; answering from it alone"
    );
    Ok((snapshot, None))
}

#[cfg(test)]
thread_local! {
    static CAPTURE_HOOK: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        std::cell::RefCell::new(None);
}

/// Test-only: run `action` once, inside the next capture, right after it has
/// read the snapshot and before it reads the journal. That is the window a
/// `finish` in another process can land in: the snapshot it replaces is the
/// one just read, and the journal it removes is the one about to be read.
#[cfg(test)]
fn interrupt_capture(action: impl FnOnce() + 'static) {
    CAPTURE_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(action)));
}

#[cfg(test)]
fn capture_hook() {
    CAPTURE_HOOK.with(|slot| {
        if let Some(action) = slot.borrow_mut().take() {
            action();
        }
    });
}

/// The committed snapshot, or an empty one when the stage has none yet. A
/// snapshot of another corpus or stage is refused, exactly as `open` refuses
/// it: its outputs are not ours to describe.
fn load_snapshot(state_dir: &Path, stage: &str, source_id: &str) -> Result<Snapshot> {
    let Some(snapshot) = Snapshot::load(&state_dir.join(SNAPSHOT_FILE))? else {
        return Ok(Snapshot::new(stage, source_id));
    };
    if snapshot.source_id != source_id || snapshot.stage != stage {
        return Err(MinSyncError::stage(
            "source_mismatch",
            format!(
                "{} holds stage {:?} of source {:?}, not stage {:?} of source {:?}",
                state_dir.display(),
                snapshot.stage,
                snapshot.source_id,
                stage,
                source_id
            ),
        ));
    }
    Ok(snapshot)
}

/// The ledger of a run whose records the snapshot does not hold yet: the
/// interrupted one, or the one this process is running. `None` when there is
/// no ledger or it belongs to an unrelated baseline.
fn active_ledger(
    state_dir: &Path,
    snapshot: &Snapshot,
    stage: &str,
    source_id: &str,
) -> Option<Ledger> {
    let path = state_dir.join(JOURNAL_FILE);
    Journal::load(&path).into_ledger(&path).filter(|ledger| {
        ledger
            .header
            .resumes_baseline(source_id, stage, &snapshot.hash())
    })
}

/// The output root as it really is, so verification cannot be fooled by a
/// symlinked root. A root that does not exist yet is used as given: nothing
/// verifies inside it, which is exactly right.
fn canonical_root(root: &Path) -> PathBuf {
    if let Ok(real) = root.canonicalize() {
        return real;
    }
    root.to_path_buf()
}

/// The stable timestamp of an output whose bytes still verify, or `None`
/// when the output is missing or changed.
fn verified_mtime(root: &Path, output: &OutputRef, mode: VerifyMode) -> Option<u128> {
    if !output_matches(root, output, mode) {
        return None;
    }
    Some(if output.mtime_ns != 0 {
        output.mtime_ns
    } else {
        std::fs::symlink_metadata(root.join(&output.path)).map_or(0, |metadata| mtime_ns(&metadata))
    })
}

/// RFC 3339 for a nanosecond timestamp; empty for a value no calendar can
/// hold (the Unix epoch is used for an unreadable mtime).
fn rfc3339(nanos: u128) -> String {
    let seconds = (nanos / 1_000_000_000).min(i64::MAX as u128) as i64;
    let subsec = (nanos % 1_000_000_000) as u32;
    chrono::DateTime::from_timestamp(seconds, subsec)
        .map_or_else(String::new, |time| time.to_rfc3339())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::incremental::journal::{Load, Record};
    use crate::incremental::stage::{OutputClaim, StageRun};

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
            _dir: dir,
            out,
        }
    }

    fn spec(env: &Env) -> StageSpec {
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
        }
    }

    fn items(list: &[(&str, &str)]) -> Vec<SourceItem> {
        list.iter()
            .map(|(key, version)| SourceItem {
                key: key.to_string(),
                version: version.to_string(),
            })
            .collect()
    }

    /// Write `key`'s output the way a host does, then commit it.
    fn produce(env: &Env, run: &mut StageRun, key: &str, version: &str, content: &str) {
        let rel = format!("{key}.md");
        std::fs::write(env.out.join(&rel), content).expect("write output");
        let claim = OutputClaim {
            path: rel,
            size: None,
            digest: None,
        };
        run.commit(key, version, None, &claim)
            .unwrap_or_else(|error| panic!("commit {key}: {error}"));
    }

    /// A stage after one finished run for `a` and one interrupted run that
    /// committed `b` and was killed before `finish`, so `b` lives only in the
    /// journal. Returns that journal, `b`'s record and the run fingerprint.
    fn mid_run(env: &Env) -> (PathBuf, Record, String) {
        let (mut run, _) =
            StageRun::open(&env.state, spec(env), items(&[("a", "v1")]), vec![]).expect("open");
        produce(env, &mut run, "a", "v1", "alpha");
        run.finish().expect("finish");

        let (mut run, _) = StageRun::open(
            &env.state,
            spec(env),
            items(&[("a", "v1"), ("b", "v1")]),
            vec![],
        )
        .expect("open");
        produce(env, &mut run, "b", "v1", "beta");
        drop(run); // killed before finish

        let journal = env.state.join(JOURNAL_FILE);
        let ledger = match Journal::load(&journal) {
            Load::Loaded(ledger) => ledger,
            other => panic!("expected a ledger, got {other:?}"),
        };
        let record = ledger.done("b").expect("b was committed").clone();
        (journal, record, ledger.header.fingerprint.clone())
    }

    /// Let a real `finish` land between a capture's two snapshot reads: the
    /// snapshot that folds the journal in is written and the journal the
    /// capture just read is removed.
    fn finish_inside_capture(env: &Env, record: Record, fingerprint: String, journal: PathBuf) {
        let snapshot = env.state.join(SNAPSHOT_FILE);
        interrupt_capture(move || {
            let mut finished = Snapshot::load(&snapshot).expect("load").expect("snapshot");
            finished.apply_record(&record, &fingerprint);
            finished.save(&snapshot).expect("save");
            Journal::remove(&journal).expect("remove the journal");
        });
    }

    fn keys(items: &[InspectedItem]) -> Vec<String> {
        items.iter().map(|item| item.key.clone()).collect()
    }

    #[test]
    fn a_capture_reads_the_journal_of_a_run_that_has_not_finished() {
        let env = env();
        mid_run(&env);
        let state =
            inspect(&env.state, "parse", "corpus", &env.out, VerifyMode::Fast).expect("inspect");
        assert_eq!(keys(&state.items), vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn a_finish_inside_a_capture_is_not_missed_by_inspect() {
        let env = env();
        let (journal, record, fingerprint) = mid_run(&env);
        finish_inside_capture(&env, record, fingerprint, journal.clone());

        let state =
            inspect(&env.state, "parse", "corpus", &env.out, VerifyMode::Fast).expect("inspect");
        assert_eq!(
            keys(&state.items),
            vec!["a".to_string(), "b".to_string()],
            "the capture retries instead of answering from the replaced snapshot"
        );
        assert!(!journal.exists(), "the emulated finish removed the journal");
    }

    #[test]
    fn a_finish_inside_a_capture_is_not_missed_by_plan() {
        let env = env();
        let (journal, record, fingerprint) = mid_run(&env);
        finish_inside_capture(&env, record, fingerprint, journal);

        let source = items(&[("a", "v1"), ("b", "v1")]);
        let planned = preview(&env.state, &spec(&env), &source, &[]).expect("preview");
        assert_eq!(
            planned.counts.process, 0,
            "both completions are already committed"
        );
        assert_eq!(planned.counts.skip, 2);
        assert_eq!(planned.decisions.len(), 2);
    }

    /// The roster of held keys is every identity the durable state carries,
    /// even when nothing about its output can be trusted: a completion whose
    /// output is gone, a released failure, a held failure and a legacy
    /// failure without a version. Reading it must take no lock and write
    /// nothing.
    #[test]
    fn inspect_known_lists_every_held_key_without_another_claim_or_a_write() {
        use crate::incremental::snapshot::{FailedItem, SkippedItem, SnapshotItem};

        let env = env();
        std::fs::create_dir_all(&env.state).expect("state dir");
        let snapshot_path = env.state.join(SNAPSHOT_FILE);

        let gone = |path: &str| OutputRef {
            path: path.into(),
            size: 5,
            digest: "sha256:missing".into(),
            mtime_ns: 0,
        };
        let failure = |version: Option<&str>, deferred: bool| FailedItem {
            version: version.map(str::to_string),
            error: "boom".into(),
            fingerprint: Some("parser-1".into()),
            deferred,
            retry_fingerprint: None,
        };
        let mut snapshot = Snapshot::new("parse", "corpus");
        // A completion whose output is gone: it is not in `items`, but the
        // key is still held.
        snapshot.items.insert(
            "a".into(),
            SnapshotItem {
                version: "a1".into(),
                fingerprint: "parser-1".into(),
                output: gone("a.md"),
            },
        );
        // An ordinary, released failure.
        snapshot
            .failed
            .insert("c".into(), failure(Some("c1"), false));
        // A deliberate omission.
        snapshot.skipped.insert(
            "d".into(),
            SkippedItem {
                version: "d1".into(),
                fingerprint: "parser-1".into(),
                reason: "unsupported type".into(),
            },
        );
        // A failure recorded before versions were kept: the only null version.
        snapshot.failed.insert(
            "e".into(),
            FailedItem {
                version: None,
                error: "legacy".into(),
                fingerprint: None,
                deferred: false,
                retry_fingerprint: None,
            },
        );
        // A held failure keeps its key visible too.
        snapshot
            .failed
            .insert("f".into(), failure(Some("f1"), true));
        // Completed and then failed: the failure's version wins.
        snapshot.items.insert(
            "g".into(),
            SnapshotItem {
                version: "g1".into(),
                fingerprint: "parser-1".into(),
                output: gone("g.md"),
            },
        );
        snapshot
            .failed
            .insert("g".into(), failure(Some("g2"), false));
        snapshot.save(&snapshot_path).expect("save");
        let before = std::fs::read(&snapshot_path).expect("snapshot");

        let state =
            inspect(&env.state, "parse", "corpus", &env.out, VerifyMode::Fast).expect("inspect");

        // No output verifies, so nothing is a live item...
        assert!(state.items.is_empty(), "{:?}", state.items);
        assert_eq!(state.skipped.len(), 1);
        assert_eq!(
            state
                .failed
                .iter()
                .map(|failure| failure.key.as_str())
                .collect::<Vec<_>>(),
            vec!["c", "e", "f", "g"]
        );
        // ...but every identity is still visible, versions taken from the
        // failure, then the omission, then the completion.
        assert_eq!(
            state.known,
            vec![
                InspectedKnown {
                    key: "a".into(),
                    version: Some("a1".into()),
                },
                InspectedKnown {
                    key: "c".into(),
                    version: Some("c1".into()),
                },
                InspectedKnown {
                    key: "d".into(),
                    version: Some("d1".into()),
                },
                InspectedKnown {
                    key: "e".into(),
                    version: None,
                },
                InspectedKnown {
                    key: "f".into(),
                    version: Some("f1".into()),
                },
                InspectedKnown {
                    key: "g".into(),
                    version: Some("g2".into()),
                },
            ]
        );

        // A read-only view takes no lock and writes nothing.
        assert!(!env.state.join("lock").exists(), "no lock is taken");
        assert!(
            !env.state.join(JOURNAL_FILE).exists(),
            "no journal is written"
        );
        assert!(!env.state.join("txn.json").exists(), "no marker is written");
        assert_eq!(
            std::fs::read(&snapshot_path).expect("snapshot"),
            before,
            "the snapshot is left byte for byte"
        );
    }
}
