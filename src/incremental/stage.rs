//! A materialization stage: an external host turns source items into
//! deterministic output files, and MinSync owns the state around them.
//!
//! The host parses, writes each output next to its final path and renames it
//! into place; MinSync never parses anything. For every item the commit
//! order is the core's one invariant:
//!
//! ```text
//! host writes the output  ->  commit() validates it  ->  journal append
//! ```
//!
//! [`StageRun::commit`] refuses to record a completion unless the output is a
//! regular file inside the managed output root, matches the claimed size and
//! digest, and belongs to the declared source version and fingerprint. A
//! crash between the host's write and `commit` leaves no record, so the item
//! is redone (the host's write is idempotent). A completion is only trusted
//! on a later run while its output still verifies.
//!
//! State lives in one directory per stage: `lock` (held for the whole run,
//! released by the OS on any exit), `txn.json` (run marker),
//! `progress.jsonl` (the append-only journal) and `snapshot.json` (the
//! committed state, rewritten once by [`StageRun::finish`]).
//!
//! Machine-readable refusals carry a stable code (`MinSyncError::Stage`):
//! `invalid_request`, `invalid_item_key`, `duplicate_item`, `unknown_item`,
//! `version_mismatch`, `fingerprint_mismatch`, `invalid_output_path`,
//! `output_missing`, `output_not_file`, `output_mismatch`, `output_conflict`,
//! `still_present`, `preserved_item`, `source_mismatch`, `corrupt_state`.

use super::fsio::{open_no_follow, sha256_file, sha256_reader, sync_for_commit};
use super::item::{validate_item_key, validate_output_path};
use super::journal::{Journal, Ledger, Outcome, OutputRef, Record, RunHeader, JOURNAL_FILE};
use super::plan::{completion, plan, Action, Decision, PlanInput, SourceItem};
use super::snapshot::{Snapshot, SNAPSHOT_FILE, SNAPSHOT_FORMAT};
use crate::error::{MinSyncError, Result};
use crate::state::{FileLock, Transaction};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

const MAX_ERROR_CHARS: usize = 2000;

/// How hard `open` checks that a recorded output is still intact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerifyMode {
    /// Present, regular, same size, and either the recorded mtime or the
    /// recorded digest. Hashing is only paid when the mtime moved.
    #[default]
    Fast,
    /// Always re-hash: catches a change that kept size and mtime.
    Digest,
}

#[derive(Debug, Clone)]
pub struct StageSpec {
    /// Stage name (`[A-Za-z0-9._-]`, at most 64 characters).
    pub stage: String,
    /// Corpus identity; a state directory belongs to exactly one.
    pub source_id: String,
    /// Everything that changes what the stage produces (parser name and
    /// version, normalization, output format, ...).
    pub fingerprint: String,
    /// The host's retry policy identity. A deferred failure is held only
    /// while this, the source version and the processing fingerprint stay
    /// the same.
    pub retry_fingerprint: Option<String>,
    /// The host asked for deferred failures to be retried after all.
    pub retry_failed: bool,
    /// Directory that holds every output. Created if missing.
    pub output_root: PathBuf,
    pub full: bool,
    pub verify: VerifyMode,
    /// Wait for another run's lock instead of failing.
    pub wait: bool,
}

/// What the host says about an output it just wrote. `size` and `digest`
/// are optional cross-checks; MinSync always measures the file itself.
#[derive(Debug, Clone)]
pub struct OutputClaim {
    /// Path relative to the output root.
    pub path: String,
    pub size: Option<u64>,
    /// `sha256:<hex>`.
    pub digest: Option<String>,
}

/// What the host wants a failure to do besides recording it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FailOptions {
    /// Hold the failure: it is not retried while the source version, the
    /// processing fingerprint and the retry fingerprint stay the same.
    pub defer: bool,
    /// Throw the item's previous completion and output away instead of
    /// leaving the last good output in place.
    pub discard_output: bool,
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize)]
pub struct Counts {
    pub process: usize,
    pub retry: usize,
    pub skip: usize,
    /// Skips that rest on the interrupted run's ledger (a subset of `skip`).
    pub resumed: usize,
    pub delete: usize,
    pub preserve: usize,
}

#[derive(Debug)]
pub struct OpenReport {
    pub run_token: String,
    /// An interrupted run's ledger was resumed.
    pub resumed: bool,
    /// Work still to do: `process`, `retry` and `delete`, in key order.
    pub decisions: Vec<Decision>,
    pub counts: Counts,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FinishReport {
    /// Items committed in this run, including ones resumed from its ledger.
    pub committed: usize,
    pub deleted: usize,
    pub failed: usize,
    /// Items that still need work (failed, never touched, or not deleted).
    pub pending: usize,
}

pub struct StageRun {
    _lock: FileLock,
    dir: PathBuf,
    spec: StageSpec,
    output_root: PathBuf,
    snapshot: Snapshot,
    ledger: Ledger,
    journal: Journal,
    items: BTreeMap<String, String>,
    unknown: BTreeSet<String>,
    /// Output path -> owning item, so one file never backs two items.
    owners: HashMap<String, String>,
}

fn invalid(message: impl Into<String>) -> MinSyncError {
    MinSyncError::stage("invalid_request", message)
}

pub(super) fn validate_spec(spec: &StageSpec) -> Result<()> {
    let name_ok = !spec.stage.is_empty()
        && spec.stage.len() <= 64
        && spec
            .stage
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
        && !spec.stage.starts_with(['.', '-']);
    if !name_ok {
        return Err(invalid(format!(
            "stage name {:?} must be 1-64 characters of [A-Za-z0-9._-] and not start with '.' or '-'",
            spec.stage
        )));
    }
    if spec.source_id.is_empty() || spec.source_id.chars().any(char::is_control) {
        return Err(invalid(
            "source_id must be non-empty text without control characters",
        ));
    }
    if spec.fingerprint.is_empty() {
        return Err(invalid("fingerprint must not be empty"));
    }
    if spec.output_root.as_os_str().is_empty() {
        return Err(invalid("output_root must not be empty"));
    }
    Ok(())
}

pub(super) fn validate_items(
    items: &[SourceItem],
    unknown: Vec<String>,
) -> Result<(BTreeMap<String, String>, BTreeSet<String>)> {
    let mut map = BTreeMap::new();
    for item in items {
        validate_item_key(&item.key)?;
        if item.version.is_empty() {
            return Err(invalid(format!("item {:?} has an empty version", item.key)));
        }
        if map.insert(item.key.clone(), item.version.clone()).is_some() {
            return Err(MinSyncError::stage(
                "duplicate_item",
                format!("item {:?} appears twice", item.key),
            ));
        }
    }
    let mut set = BTreeSet::new();
    for key in unknown {
        validate_item_key(&key)?;
        if map.contains_key(&key) {
            return Err(invalid(format!(
                "item {key:?} is listed both as present and as unknown"
            )));
        }
        set.insert(key);
    }
    Ok((map, set))
}

pub(super) fn mtime_ns(metadata: &std::fs::Metadata) -> u128 {
    metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |elapsed| elapsed.as_nanos())
}

/// Whether a recorded output is still on disk and unchanged.
pub(super) fn output_matches(root: &Path, output: &OutputRef, mode: VerifyMode) -> bool {
    if validate_output_path(&output.path).is_err() {
        return false;
    }
    let path = root.join(&output.path);
    let Ok(metadata) = std::fs::symlink_metadata(&path) else {
        return false;
    };
    if !metadata.is_file() || metadata.len() != output.size {
        return false;
    }
    if mode == VerifyMode::Fast && output.mtime_ns != 0 && mtime_ns(&metadata) == output.mtime_ns {
        return true;
    }
    sha256_file(&path).is_ok_and(|digest| digest == output.digest)
}

/// Hash of the source listing, for the run marker.
fn source_hash(items: &BTreeMap<String, String>) -> String {
    let mut hasher = Sha256::new();
    for (key, version) in items {
        hasher.update(key.as_bytes());
        hasher.update([0]);
        hasher.update(version.as_bytes());
        hasher.update([0]);
    }
    format!("sha256:{}", hex::encode(hasher.finalize()))
}

fn truncate_error(error: &str) -> String {
    error.chars().take(MAX_ERROR_CHARS).collect()
}

/// Tally one decision into a report's counts. A skip that rests on a
/// deferred failure is still a skip, and is counted like any other.
pub(super) fn tally(counts: &mut Counts, decision: &Decision) {
    match decision.action {
        Action::Process => counts.process += 1,
        Action::Retry => counts.retry += 1,
        Action::Skip => {
            counts.skip += 1;
            counts.resumed += usize::from(decision.resumed);
        }
        Action::Delete => counts.delete += 1,
        Action::Preserve => counts.preserve += 1,
    }
}

/// Remove one relative output file, refusing to leave the root through a
/// symlinked parent directory. Idempotent: a missing file is not an error.
fn unlink_output(root: &Path, path: &str) -> Result<bool> {
    validate_output_path(path)?;
    let target = root.join(path);
    // A symlinked directory must not let the unlink leave the root.
    match target.parent().unwrap_or(root).canonicalize() {
        Ok(parent) if !parent.starts_with(root) => Err(MinSyncError::stage(
            "invalid_output_path",
            format!("output {path:?} resolves outside the output root"),
        )),
        Ok(_) => match std::fs::remove_file(&target) {
            Ok(()) => Ok(true),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(error) => Err(error.into()),
        },
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error.into()),
    }
}

impl StageRun {
    /// Open a run, or resume the interrupted one.
    ///
    /// Everything is validated before any state is touched. The state lock
    /// is then taken and held until the run is finished or dropped; if
    /// another run holds it this fails with [`MinSyncError::LockFailed`]
    /// (or waits, with `spec.wait`).
    pub fn open(
        state_dir: &Path,
        spec: StageSpec,
        items: Vec<SourceItem>,
        unknown: Vec<String>,
    ) -> Result<(Self, OpenReport)> {
        validate_spec(&spec)?;
        let (item_map, unknown_set) = validate_items(&items, unknown)?;

        std::fs::create_dir_all(state_dir)?;
        let lock = FileLock::acquire(&state_dir.join("lock"), spec.wait)?;
        std::fs::create_dir_all(&spec.output_root)?;
        let output_root = spec.output_root.canonicalize()?;

        let snapshot = match Snapshot::load(&state_dir.join(SNAPSHOT_FILE))? {
            Some(snapshot) => {
                if snapshot.source_id != spec.source_id || snapshot.stage != spec.stage {
                    return Err(MinSyncError::stage(
                        "source_mismatch",
                        format!(
                            "{} holds stage {:?} of source {:?}, not stage {:?} of source {:?}",
                            state_dir.display(),
                            snapshot.stage,
                            snapshot.source_id,
                            spec.stage,
                            spec.source_id
                        ),
                    ));
                }
                snapshot
            }
            None => Snapshot::new(&spec.stage, &spec.source_id),
        };
        let baseline = snapshot.hash();

        let journal_path = state_dir.join(JOURNAL_FILE);
        let txn_path = state_dir.join("txn.json");
        let expected = RunHeader::new(
            &spec.stage,
            &spec.source_id,
            spec.full,
            &baseline,
            &spec.fingerprint,
        );
        // Read the interrupted run's ledger BEFORE clearing its marker.
        let resumed = Journal::load(&journal_path)
            .into_ledger(&journal_path)
            .filter(|ledger| ledger.header.resumes(&expected));
        Transaction::remove(&txn_path)?;

        let was_resumed = resumed.is_some();
        let (header, mut ledger) = match resumed {
            Some(ledger) => (
                RunHeader {
                    run_token: ledger.header.run_token.clone(),
                    created_at: ledger.header.created_at.clone(),
                    ..expected
                },
                ledger,
            ),
            None => (expected.clone(), Ledger::new(expected)),
        };
        ledger.header = header.clone();
        ledger.torn_tail = false; // the rewrite below drops a torn tail

        Transaction::new(
            &spec.source_id,
            &header.run_token,
            Some(baseline),
            &source_hash(&item_map),
        )
        .save(&txn_path)?;
        let journal = Journal::create(&journal_path, &header, was_resumed.then_some(&ledger))?;

        let verify = spec.verify;
        let all = plan(&PlanInput {
            items: &items,
            unknown: &unknown_set,
            fingerprint: &spec.fingerprint,
            retry_fingerprint: spec.retry_fingerprint.as_deref(),
            full: spec.full,
            retry_failed: spec.retry_failed,
            snapshot: &snapshot,
            ledger: &ledger,
            output_ok: &|output| output_matches(&output_root, output, verify),
        });
        let mut counts = Counts::default();
        let mut decisions = Vec::new();
        for decision in all {
            match decision.action {
                Action::Skip => {
                    counts.skip += 1;
                    counts.resumed += usize::from(decision.resumed);
                    // A skip that rests on a deferred failure has to be
                    // visible to the host, which otherwise cannot tell it
                    // from an item that is simply up to date.
                    if decision.deferred_error.is_some() {
                        decisions.push(decision);
                    }
                }
                Action::Preserve => counts.preserve += 1,
                Action::Process => {
                    counts.process += 1;
                    decisions.push(decision);
                }
                Action::Retry => {
                    counts.retry += 1;
                    decisions.push(decision);
                }
                Action::Delete => {
                    counts.delete += 1;
                    decisions.push(decision);
                }
            }
        }

        let mut owners = HashMap::new();
        let known_keys = snapshot.items.keys().cloned().chain(
            ledger
                .records()
                .filter(|record| record.outcome == Outcome::Committed)
                .map(|record| record.item.clone()),
        );
        for key in known_keys {
            if let Some(path) = completion(&snapshot, &ledger, &key)
                .and_then(|done| done.output)
                .map(|output| output.path.clone())
            {
                owners.insert(path, key);
            }
        }

        let report = OpenReport {
            run_token: header.run_token,
            resumed: was_resumed,
            decisions,
            counts,
        };
        Ok((
            Self {
                _lock: lock,
                dir: state_dir.to_path_buf(),
                spec: StageSpec {
                    output_root: output_root.clone(),
                    ..spec
                },
                output_root,
                snapshot,
                ledger,
                journal,
                items: item_map,
                unknown: unknown_set,
                owners,
            },
            report,
        ))
    }

    fn check_item(&self, key: &str, version: &str, fingerprint: Option<&str>) -> Result<()> {
        let Some(current) = self.items.get(key) else {
            return Err(MinSyncError::stage(
                "unknown_item",
                format!("item {key:?} is not part of this run"),
            ));
        };
        if current != version {
            return Err(MinSyncError::stage(
                "version_mismatch",
                format!("item {key:?} is at version {current:?} in this run, not {version:?}"),
            ));
        }
        if let Some(claimed) = fingerprint {
            if claimed != self.spec.fingerprint {
                return Err(MinSyncError::stage(
                    "fingerprint_mismatch",
                    format!(
                        "this run has fingerprint {:?}, not {claimed:?}",
                        self.spec.fingerprint
                    ),
                ));
            }
        }
        Ok(())
    }

    /// The output file at `path`: a regular file whose real location is
    /// inside the output root (no symlink may lead out of it).
    fn resolve_output(&self, path: &str) -> Result<PathBuf> {
        validate_output_path(path)?;
        let candidate = self.output_root.join(path);
        let metadata = match std::fs::symlink_metadata(&candidate) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(MinSyncError::stage(
                    "output_missing",
                    format!("output {path:?} does not exist"),
                ));
            }
            Err(error) => return Err(error.into()),
        };
        if metadata.file_type().is_symlink() {
            return Err(MinSyncError::stage(
                "invalid_output_path",
                format!("output {path:?} is a symlink"),
            ));
        }
        if !metadata.is_file() {
            return Err(MinSyncError::stage(
                "output_not_file",
                format!("output {path:?} is not a regular file"),
            ));
        }
        if !candidate.canonicalize()?.starts_with(&self.output_root) {
            return Err(MinSyncError::stage(
                "invalid_output_path",
                format!("output {path:?} resolves outside the output root"),
            ));
        }
        Ok(candidate)
    }

    /// Record `key` as committed for `version` once its output verifies.
    ///
    /// The output file is made durable first, then validated, and only then
    /// is the completion appended: a refused commit leaves no record.
    pub fn commit(
        &mut self,
        key: &str,
        version: &str,
        fingerprint: Option<&str>,
        claim: &OutputClaim,
    ) -> Result<OutputRef> {
        self.check_item(key, version, fingerprint)?;
        validate_output_path(&claim.path)?;
        if self
            .owners
            .get(&claim.path)
            .is_some_and(|owner| owner != key)
        {
            return Err(MinSyncError::stage(
                "output_conflict",
                format!("output {:?} already belongs to another item", claim.path),
            ));
        }
        let path = self.resolve_output(&claim.path)?;
        // One descriptor for everything below, opened without following a
        // symlink: the size, mtime and digest recorded all describe the same
        // bytes, and a link swapped in after the checks above is refused.
        let mut file = open_no_follow(&path).map_err(|error| match error.raw_os_error() {
            #[cfg(unix)]
            Some(code) if code == libc::ELOOP => MinSyncError::stage(
                "invalid_output_path",
                format!("output {:?} is a symlink", claim.path),
            ),
            _ if error.kind() == std::io::ErrorKind::NotFound => MinSyncError::stage(
                "output_missing",
                format!("output {:?} does not exist", claim.path),
            ),
            _ => MinSyncError::Io(error),
        })?;
        let metadata = file.metadata()?;
        if !metadata.is_file() {
            return Err(MinSyncError::stage(
                "output_not_file",
                format!("output {:?} is not a regular file", claim.path),
            ));
        }
        // The host renamed it into place; make sure the bytes behind that
        // name survive a power loss before anything points at them.
        sync_for_commit(&file, &path)?;
        let output = OutputRef {
            path: claim.path.clone(),
            size: metadata.len(),
            digest: sha256_reader(&mut file)?,
            mtime_ns: mtime_ns(&metadata),
        };
        if claim.size.is_some_and(|size| size != output.size) {
            return Err(MinSyncError::stage(
                "output_mismatch",
                format!(
                    "output {:?} has {} bytes, the host claimed {}",
                    claim.path,
                    output.size,
                    claim.size.unwrap_or_default()
                ),
            ));
        }
        if claim
            .digest
            .as_deref()
            .is_some_and(|digest| digest != output.digest)
        {
            return Err(MinSyncError::stage(
                "output_mismatch",
                format!(
                    "output {:?} has digest {}, the host claimed {}",
                    claim.path,
                    output.digest,
                    claim.digest.as_deref().unwrap_or_default()
                ),
            ));
        }

        let record = Record::committed(key, version, Some(output.clone()));
        self.journal.append(std::slice::from_ref(&record))?;
        if let Some(previous) = completion(&self.snapshot, &self.ledger, key)
            .and_then(|done| done.output)
            .map(|previous| previous.path.clone())
        {
            self.owners.remove(&previous);
        }
        self.ledger.apply(record);
        self.owners.insert(output.path.clone(), key.to_string());
        Ok(output)
    }

    /// Record that processing `key` failed. Nothing is committed and, unless
    /// `options.discard_output` says otherwise, the item's previous output is
    /// left alone: a failure never erases a last good result by accident.
    ///
    /// With `options.defer` the failure is held: it is not retried while the
    /// source version, the processing fingerprint and the run's retry
    /// fingerprint stay the same.
    pub fn fail(
        &mut self,
        key: &str,
        version: &str,
        fingerprint: Option<&str>,
        error: &str,
        options: FailOptions,
    ) -> Result<()> {
        self.check_item(key, version, fingerprint)?;
        if options.discard_output {
            // Remove the output before recording anything: a crash in
            // between leaves no record, so the item is simply redone.
            self.discard_output(key)?;
        }
        let retry_fingerprint = if options.defer {
            self.spec.retry_fingerprint.clone()
        } else {
            None
        };
        let record = Record::failed_with(
            key,
            Some(version),
            truncate_error(error),
            options.defer,
            retry_fingerprint,
            options.discard_output,
        );
        self.journal.append(std::slice::from_ref(&record))?;
        self.ledger.apply(record);
        Ok(())
    }

    /// Record a deliberate omission of `key`: the item is part of the
    /// source, but it produces no output and none is expected while its
    /// source version and this run's fingerprint stay the same.
    ///
    /// Any output a previous completion recorded is removed first, so an
    /// omission never leaves an output behind that nothing points at.
    pub fn skip(&mut self, key: &str, version: &str, reason: &str) -> Result<()> {
        self.check_item(key, version, None)?;
        self.discard_output(key)?;
        let record = Record::skipped(key, version, &truncate_error(reason));
        self.journal.append(std::slice::from_ref(&record))?;
        self.ledger.apply(record);
        Ok(())
    }

    /// Remove the output of an item whose source is confirmed gone, then
    /// record the deletion. Idempotent: returns whether a file was removed.
    pub fn delete(&mut self, key: &str) -> Result<bool> {
        validate_item_key(key)?;
        if self.items.contains_key(key) {
            return Err(MinSyncError::stage(
                "still_present",
                format!("item {key:?} is still part of this run's source"),
            ));
        }
        if self.unknown.contains(key) {
            return Err(MinSyncError::stage(
                "preserved_item",
                format!("item {key:?} is unobservable, so its output is preserved"),
            ));
        }
        let removed = self.discard_output(key)?;
        let record = Record::deleted(key);
        self.journal.append(std::slice::from_ref(&record))?;
        self.ledger.apply(record);
        Ok(removed.is_some())
    }

    /// Remove the output `key` is currently recorded as having, if any, and
    /// stop treating it as that file's owner. The caller appends the record
    /// that supersedes the completion.
    fn discard_output(&mut self, key: &str) -> Result<Option<String>> {
        let Some(path) = completion(&self.snapshot, &self.ledger, key)
            .and_then(|done| done.output)
            .map(|output| output.path.clone())
        else {
            return Ok(None); // nothing committed, nothing to remove
        };
        let removed = unlink_output(&self.output_root, &path)?;
        self.owners.remove(&path);
        Ok(removed.then_some(path))
    }

    /// Compact the run into the snapshot (one atomic write), drop the
    /// journal and the run marker, and release the lock. Items that were
    /// not committed keep their previous completion and are reported as
    /// pending; they are simply retried by the next run. A deferred failure
    /// is not pending: nothing will retry it until the host changes
    /// something or asks for a retry.
    pub fn finish(mut self) -> Result<FinishReport> {
        // Every record is already durable and held in `self.ledger`; close
        // the journal handle so removing the file below works on Windows too.
        drop(self.journal);
        let items: Vec<SourceItem> = self
            .items
            .iter()
            .map(|(key, version)| SourceItem {
                key: key.clone(),
                version: version.clone(),
            })
            .collect();
        let pending = plan(&PlanInput {
            items: &items,
            unknown: &self.unknown,
            fingerprint: &self.spec.fingerprint,
            retry_fingerprint: self.spec.retry_fingerprint.as_deref(),
            full: self.spec.full,
            retry_failed: self.spec.retry_failed,
            snapshot: &self.snapshot,
            ledger: &self.ledger,
            output_ok: &|_| true,
        })
        .iter()
        .filter(|decision| {
            matches!(
                decision.action,
                Action::Process | Action::Retry | Action::Delete
            )
        })
        .count();

        let mut report = FinishReport {
            committed: 0,
            deleted: 0,
            failed: 0,
            pending,
        };
        let fingerprint = self.ledger.header.fingerprint.clone();
        let mut next = std::mem::replace(&mut self.snapshot, Snapshot::new("", ""));
        for record in self.ledger.records() {
            match next.apply_record(record, &fingerprint) {
                Outcome::Committed => report.committed += 1,
                Outcome::Deleted => report.deleted += 1,
                Outcome::Failed => report.failed += 1,
                Outcome::Skipped => {}
            }
        }
        // A failure or an omission only matters while its item is still in
        // the source.
        next.failed
            .retain(|key, _| self.items.contains_key(key) || self.unknown.contains(key));
        next.skipped
            .retain(|key, _| self.items.contains_key(key) || self.unknown.contains(key));
        next.updated_at = chrono::Utc::now().to_rfc3339();
        // A snapshot loaded from an older build keeps its own format, but
        // this build writes omissions and the deferred-failure fields; the
        // saved file must be the format that describes them, or the next
        // reader would trust a format-1 file to have neither.
        next.format = SNAPSHOT_FORMAT;

        next.save(&self.dir.join(SNAPSHOT_FILE))?;
        Journal::remove(&self.dir.join(JOURNAL_FILE))?;
        Transaction::remove(&self.dir.join("txn.json"))?;
        Ok(report)
    }

    pub fn run_token(&self) -> &str {
        &self.ledger.header.run_token
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::incremental::journal::{Journal, Load, JOURNAL_FILE};
    use crate::incremental::snapshot::{Snapshot, SNAPSHOT_FILE};
    use sha2::{Digest, Sha256};

    struct Env {
        _dir: tempfile::TempDir,
        state: PathBuf,
        out: PathBuf,
    }

    fn env() -> Env {
        let dir = tempfile::tempdir().expect("tempdir");
        let state = dir.path().join("state");
        let out = dir.path().join("out");
        std::fs::create_dir_all(&out).expect("output root");
        Env {
            _dir: dir,
            state,
            out,
        }
    }

    fn spec(env: &Env, fingerprint: &str) -> StageSpec {
        StageSpec {
            stage: "parse".into(),
            source_id: "corpus".into(),
            fingerprint: fingerprint.into(),
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

    fn open(env: &Env, fingerprint: &str, list: &[(&str, &str)]) -> (StageRun, OpenReport) {
        StageRun::open(&env.state, spec(env, fingerprint), items(list), vec![])
            .unwrap_or_else(|error| panic!("open: {error}"))
    }

    fn pending(report: &OpenReport) -> Vec<(String, Action)> {
        report
            .decisions
            .iter()
            .map(|decision| (decision.key.clone(), decision.action))
            .collect()
    }

    fn actions(list: &[(&str, Action)]) -> Vec<(String, Action)> {
        list.iter()
            .map(|(key, action)| (key.to_string(), *action))
            .collect()
    }

    /// What a host does: write beside the target, then rename into place.
    fn write_output(env: &Env, rel: &str, content: &str) {
        let target = env.out.join(rel);
        std::fs::create_dir_all(target.parent().unwrap()).expect("parent");
        let tmp = env.out.join(format!(".{}.tmp", rel.replace('/', "_")));
        std::fs::write(&tmp, content).expect("write tmp");
        std::fs::rename(&tmp, &target).expect("rename");
    }

    fn claim(path: &str) -> OutputClaim {
        OutputClaim {
            path: path.into(),
            size: None,
            digest: None,
        }
    }

    fn output(path: &str) -> OutputRef {
        OutputRef {
            path: path.into(),
            size: 1,
            digest: "sha256:x".into(),
            mtime_ns: 1,
        }
    }

    /// Process `key` the way a host would: write its output, then commit it.
    fn produce(env: &Env, run: &mut StageRun, key: &str, version: &str, content: &str) {
        let rel = format!("{key}.md");
        write_output(env, &rel, content);
        run.commit(key, version, None, &claim(&rel))
            .unwrap_or_else(|error| panic!("commit {key}: {error}"));
    }

    fn code(error: MinSyncError) -> &'static str {
        match error {
            MinSyncError::Stage { code, .. } => code,
            other => panic!("expected a stage error, got {other:?}"),
        }
    }

    fn journal_records(env: &Env) -> usize {
        match Journal::load(&env.state.join(JOURNAL_FILE)) {
            Load::Loaded(ledger) => ledger.records().count(),
            other => panic!("expected a ledger, got {other:?}"),
        }
    }

    #[test]
    fn commit_finish_then_reopen_skips_everything() {
        let env = env();
        let (mut run, report) = open(&env, "fp", &[("a", "v1"), ("b", "v1")]);
        assert!(!report.resumed);
        assert_eq!(
            pending(&report),
            actions(&[("a", Action::Process), ("b", Action::Process)])
        );
        produce(&env, &mut run, "a", "v1", "alpha");
        produce(&env, &mut run, "b", "v1", "beta");
        let finished = run.finish().expect("finish");
        assert_eq!(finished.committed, 2);
        assert_eq!(finished.pending, 0);

        assert!(env.state.join(SNAPSHOT_FILE).exists());
        assert!(!env.state.join(JOURNAL_FILE).exists(), "journal compacted");
        assert!(!env.state.join("txn.json").exists(), "marker cleared");

        let (_run, report) = open(&env, "fp", &[("a", "v1"), ("b", "v1")]);
        assert!(report.decisions.is_empty());
        assert_eq!(report.counts.skip, 2);
        assert_eq!(report.counts.resumed, 0);
    }

    #[test]
    fn a_killed_run_resumes_from_its_pending_items() {
        let env = env();
        let all = [("a", "v1"), ("b", "v1"), ("c", "v1")];
        let (mut run, _) = open(&env, "fp", &all);
        produce(&env, &mut run, "a", "v1", "alpha");
        produce(&env, &mut run, "b", "v1", "beta");
        drop(run); // killed: no finish

        let (mut run, report) = open(&env, "fp", &all);
        assert!(report.resumed);
        assert_eq!(pending(&report), actions(&[("c", Action::Process)]));
        assert_eq!(report.counts.resumed, 2);
        assert_eq!(report.counts.skip, 2);
        produce(&env, &mut run, "c", "v1", "gamma");
        run.finish().expect("finish");

        let (_run, report) = open(&env, "fp", &all);
        assert!(report.decisions.is_empty());
        assert_eq!(report.counts.skip, 3);
    }

    #[test]
    fn an_output_written_before_a_crash_but_never_recorded_is_redone_idempotently() {
        let env = env();
        let (run, _) = open(&env, "fp", &[("a", "v1")]);
        write_output(&env, "a.md", "alpha");
        // crash after the output commit, before the completion record
        drop(run);
        assert!(env.out.join("a.md").exists());

        let (mut run, report) = open(&env, "fp", &[("a", "v1")]);
        assert_eq!(
            pending(&report),
            actions(&[("a", Action::Process)]),
            "an unrecorded output is not a completion"
        );
        produce(&env, &mut run, "a", "v1", "alpha");
        run.finish().expect("finish");
        assert_eq!(
            std::fs::read_to_string(env.out.join("a.md")).unwrap(),
            "alpha"
        );
    }

    #[test]
    fn commit_rejects_unsafe_output_paths_and_records_nothing() {
        let env = env();
        std::fs::write(env.out.parent().unwrap().join("outside.md"), "secret").unwrap();
        let (mut run, _) = open(&env, "fp", &[("a", "v1")]);
        for path in [
            "../outside.md",
            "/etc/hosts",
            "a/../../outside.md",
            "",
            "x\\y.md",
        ] {
            let error = run
                .commit("a", "v1", None, &claim(path))
                .expect_err(&format!("{path:?}"));
            assert_eq!(code(error), "invalid_output_path", "{path:?}");
        }
        assert_eq!(journal_records(&env), 0);
    }

    #[cfg(unix)]
    #[test]
    fn commit_rejects_symlinks_that_leave_the_output_root() {
        let env = env();
        let outside = env.out.parent().unwrap().join("outside");
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("secret.md"), "secret").unwrap();
        std::os::unix::fs::symlink(outside.join("secret.md"), env.out.join("link.md")).unwrap();
        std::os::unix::fs::symlink(&outside, env.out.join("linkdir")).unwrap();

        let (mut run, _) = open(&env, "fp", &[("a", "v1")]);
        for path in ["link.md", "linkdir/secret.md"] {
            let error = run
                .commit("a", "v1", None, &claim(path))
                .expect_err(&format!("{path:?}"));
            assert_eq!(code(error), "invalid_output_path", "{path:?}");
        }
        assert_eq!(journal_records(&env), 0);
    }

    #[test]
    fn commit_requires_an_existing_regular_file() {
        let env = env();
        std::fs::create_dir_all(env.out.join("dir.md")).unwrap();
        let (mut run, _) = open(&env, "fp", &[("a", "v1")]);
        assert_eq!(
            code(
                run.commit("a", "v1", None, &claim("missing.md"))
                    .unwrap_err()
            ),
            "output_missing"
        );
        assert_eq!(
            code(run.commit("a", "v1", None, &claim("dir.md")).unwrap_err()),
            "output_not_file"
        );
        assert_eq!(journal_records(&env), 0);
    }

    #[test]
    fn commit_checks_the_claimed_size_and_digest() {
        let env = env();
        let (mut run, _) = open(&env, "fp", &[("a", "v1")]);
        write_output(&env, "a.md", "alpha");
        let digest = format!("sha256:{}", hex::encode(Sha256::digest(b"alpha")));

        let wrong_size = OutputClaim {
            size: Some(99),
            ..claim("a.md")
        };
        assert_eq!(
            code(run.commit("a", "v1", None, &wrong_size).unwrap_err()),
            "output_mismatch"
        );
        let wrong_digest = OutputClaim {
            digest: Some("sha256:00".into()),
            ..claim("a.md")
        };
        assert_eq!(
            code(run.commit("a", "v1", None, &wrong_digest).unwrap_err()),
            "output_mismatch"
        );
        assert_eq!(journal_records(&env), 0);

        let good = OutputClaim {
            size: Some(5),
            digest: Some(digest.clone()),
            ..claim("a.md")
        };
        let output = run.commit("a", "v1", None, &good).expect("matching claim");
        assert_eq!(output.size, 5);
        assert_eq!(output.digest, digest);
    }

    #[test]
    fn commit_rejects_a_version_item_or_fingerprint_that_does_not_match_the_run() {
        let env = env();
        let (mut run, _) = open(&env, "fp", &[("a", "v1")]);
        write_output(&env, "a.md", "alpha");
        assert_eq!(
            code(run.commit("a", "v0", None, &claim("a.md")).unwrap_err()),
            "version_mismatch"
        );
        assert_eq!(
            code(run.commit("ghost", "v1", None, &claim("a.md")).unwrap_err()),
            "unknown_item"
        );
        assert_eq!(
            code(
                run.commit("a", "v1", Some("other-fp"), &claim("a.md"))
                    .unwrap_err()
            ),
            "fingerprint_mismatch"
        );
        assert_eq!(
            code(
                run.fail("a", "v0", None, "boom", FailOptions::default())
                    .unwrap_err()
            ),
            "version_mismatch"
        );
        assert_eq!(journal_records(&env), 0);
        run.commit("a", "v1", Some("fp"), &claim("a.md"))
            .expect("the matching claim is accepted");
    }

    #[test]
    fn two_items_cannot_claim_one_output_path() {
        let env = env();
        let (mut run, _) = open(&env, "fp", &[("a", "v1"), ("b", "v1")]);
        write_output(&env, "shared.md", "x");
        run.commit("a", "v1", None, &claim("shared.md")).unwrap();
        run.commit("a", "v1", None, &claim("shared.md"))
            .expect("an item may recommit its own path");
        assert_eq!(
            code(
                run.commit("b", "v1", None, &claim("shared.md"))
                    .unwrap_err()
            ),
            "output_conflict"
        );
    }

    #[test]
    fn a_failure_keeps_the_previous_output_and_the_item_is_retried() {
        let env = env();
        let (mut run, _) = open(&env, "fp", &[("a", "v1")]);
        produce(&env, &mut run, "a", "v1", "old alpha");
        run.finish().unwrap();

        let (mut run, report) = open(&env, "fp", &[("a", "v2")]);
        assert_eq!(pending(&report), actions(&[("a", Action::Process)]));
        run.fail("a", "v2", None, "parser crashed", FailOptions::default())
            .unwrap();
        let finished = run.finish().unwrap();
        assert_eq!(finished.failed, 1);
        assert_eq!(finished.pending, 1);
        assert_eq!(
            std::fs::read_to_string(env.out.join("a.md")).unwrap(),
            "old alpha",
            "the previous valid output is intact"
        );

        let snapshot = Snapshot::load(&env.state.join(SNAPSHOT_FILE))
            .unwrap()
            .unwrap();
        assert_eq!(snapshot.items["a"].version, "v1", "old completion kept");
        assert_eq!(snapshot.failed["a"].error, "parser crashed");

        let (mut run, report) = open(&env, "fp", &[("a", "v2")]);
        assert_eq!(pending(&report), actions(&[("a", Action::Retry)]));
        produce(&env, &mut run, "a", "v2", "new alpha");
        run.finish().unwrap();
        let snapshot = Snapshot::load(&env.state.join(SNAPSHOT_FILE))
            .unwrap()
            .unwrap();
        assert!(snapshot.failed.is_empty(), "a commit clears the failure");
        assert_eq!(snapshot.items["a"].version, "v2");
    }

    #[test]
    fn a_failure_alone_never_completes_the_item_even_after_a_crash() {
        let env = env();
        let (mut run, _) = open(&env, "fp", &[("a", "v1")]);
        run.fail("a", "v1", None, "boom", FailOptions::default())
            .unwrap();
        drop(run);

        let (_run, report) = open(&env, "fp", &[("a", "v1")]);
        assert_eq!(pending(&report), actions(&[("a", Action::Retry)]));
    }

    #[test]
    fn deleting_a_confirmed_absent_source_removes_its_output_idempotently() {
        let env = env();
        let (mut run, _) = open(&env, "fp", &[("a", "v1"), ("b", "v1")]);
        produce(&env, &mut run, "a", "v1", "alpha");
        produce(&env, &mut run, "b", "v1", "beta");
        run.finish().unwrap();

        let (mut run, report) = open(&env, "fp", &[("a", "v1")]);
        assert_eq!(pending(&report), actions(&[("b", Action::Delete)]));
        assert!(run.delete("b").unwrap(), "output removed");
        assert!(!env.out.join("b.md").exists());
        assert!(!run.delete("b").unwrap(), "second delete is a no-op");
        drop(run); // crash before finish

        let (run, report) = open(&env, "fp", &[("a", "v1")]);
        assert!(report.decisions.is_empty(), "{:?}", report.decisions);
        run.finish().unwrap();
        let snapshot = Snapshot::load(&env.state.join(SNAPSHOT_FILE))
            .unwrap()
            .unwrap();
        assert!(!snapshot.items.contains_key("b"));
        assert!(snapshot.items.contains_key("a"));
    }

    #[test]
    fn delete_tolerates_an_output_that_is_already_gone_and_refuses_present_items() {
        let env = env();
        let (mut run, _) = open(&env, "fp", &[("a", "v1"), ("b", "v1")]);
        produce(&env, &mut run, "a", "v1", "alpha");
        produce(&env, &mut run, "b", "v1", "beta");
        run.finish().unwrap();
        std::fs::remove_file(env.out.join("b.md")).unwrap();

        let (mut run, _) = open(&env, "fp", &[("a", "v1")]);
        assert!(!run.delete("b").expect("missing output is fine"));
        assert_eq!(code(run.delete("a").unwrap_err()), "still_present");
        assert!(env.out.join("a.md").exists());
    }

    #[test]
    fn an_unreadable_source_is_preserved_and_never_deleted() {
        let env = env();
        let (mut run, _) = open(&env, "fp", &[("a", "v1"), ("b", "v1")]);
        produce(&env, &mut run, "a", "v1", "alpha");
        produce(&env, &mut run, "b", "v1", "beta");
        run.finish().unwrap();

        let (mut run, report) = StageRun::open(
            &env.state,
            spec(&env, "fp"),
            items(&[("a", "v1")]),
            vec!["b".to_string()],
        )
        .unwrap();
        assert!(report.decisions.is_empty(), "{:?}", report.decisions);
        assert_eq!(report.counts.preserve, 1);
        assert_eq!(code(run.delete("b").unwrap_err()), "preserved_item");
        run.finish().unwrap();

        assert_eq!(
            std::fs::read_to_string(env.out.join("b.md")).unwrap(),
            "beta"
        );
        let snapshot = Snapshot::load(&env.state.join(SNAPSHOT_FILE))
            .unwrap()
            .unwrap();
        assert!(snapshot.items.contains_key("b"), "completion preserved");
    }

    #[test]
    fn changing_the_fingerprint_reprocesses_only_what_it_invalidates() {
        let env = env();
        let all = [("a", "v1"), ("b", "v1")];
        let (mut run, _) = open(&env, "parser-1", &all);
        produce(&env, &mut run, "a", "v1", "alpha-1");
        produce(&env, &mut run, "b", "v1", "beta-1");
        run.finish().unwrap();

        let (mut run, report) = open(&env, "parser-2", &all);
        assert_eq!(
            pending(&report),
            actions(&[("a", Action::Process), ("b", Action::Process)])
        );
        produce(&env, &mut run, "a", "v1", "alpha-2");
        run.finish().unwrap(); // b was left undone

        let (_run, report) = open(&env, "parser-2", &all);
        assert_eq!(
            pending(&report),
            actions(&[("b", Action::Process)]),
            "a keeps its new-fingerprint completion; b still carries the old one"
        );
    }

    #[test]
    fn a_full_run_reprocesses_unchanged_items_but_resumes_its_own_work() {
        let env = env();
        let (mut run, _) = open(&env, "fp", &[("a", "v1"), ("b", "v1")]);
        produce(&env, &mut run, "a", "v1", "alpha");
        produce(&env, &mut run, "b", "v1", "beta");
        run.finish().unwrap();

        let full = |env: &Env| StageSpec {
            full: true,
            ..spec(env, "fp")
        };
        let (mut run, report) = StageRun::open(
            &env.state,
            full(&env),
            items(&[("a", "v1"), ("b", "v1")]),
            vec![],
        )
        .unwrap();
        assert_eq!(
            pending(&report),
            actions(&[("a", Action::Process), ("b", Action::Process)])
        );
        produce(&env, &mut run, "a", "v1", "alpha");
        drop(run);

        let (_run, report) = StageRun::open(
            &env.state,
            full(&env),
            items(&[("a", "v1"), ("b", "v1")]),
            vec![],
        )
        .unwrap();
        assert!(report.resumed);
        assert_eq!(pending(&report), actions(&[("b", Action::Process)]));
    }

    #[test]
    fn a_ledger_from_another_baseline_or_fingerprint_is_not_resumed() {
        let env = env();
        let (mut run, _) = open(&env, "fp-1", &[("a", "v1")]);
        produce(&env, &mut run, "a", "v1", "alpha");
        drop(run);

        let (_run, report) = open(&env, "fp-2", &[("a", "v1")]);
        assert!(!report.resumed);
        assert_eq!(pending(&report), actions(&[("a", Action::Process)]));
    }

    #[test]
    fn a_torn_final_record_is_harmless_and_a_corrupt_middle_means_redo() {
        let env = env();
        let all = [("a", "v1"), ("b", "v1")];
        let (mut run, _) = open(&env, "fp", &all);
        produce(&env, &mut run, "a", "v1", "alpha");
        produce(&env, &mut run, "b", "v1", "beta");
        drop(run);

        let journal = env.state.join(JOURNAL_FILE);
        let intact = std::fs::read_to_string(&journal).unwrap();
        let mut torn = intact.clone();
        torn.push_str(r#"{"item":"c","version":"v1","outcome":"commi"#);
        std::fs::write(&journal, &torn).unwrap();
        let (run, report) = open(&env, "fp", &all);
        assert!(report.resumed);
        assert!(report.decisions.is_empty(), "{:?}", report.decisions);
        drop(run);

        let mut lines: Vec<&str> = intact.lines().collect();
        lines.insert(2, "garbage in the middle");
        std::fs::write(&journal, lines.join("\n") + "\n").unwrap();
        let (_run, report) = open(&env, "fp", &all);
        assert!(!report.resumed, "an ambiguous ledger is never trusted");
        assert_eq!(
            pending(&report),
            actions(&[("a", Action::Process), ("b", Action::Process)])
        );
    }

    #[test]
    fn a_completion_whose_output_vanished_is_rebuilt() {
        let env = env();
        let (mut run, _) = open(&env, "fp", &[("a", "v1"), ("b", "v1")]);
        produce(&env, &mut run, "a", "v1", "alpha");
        produce(&env, &mut run, "b", "v1", "beta");
        run.finish().unwrap();
        std::fs::remove_file(env.out.join("a.md")).unwrap();

        let (_run, report) = open(&env, "fp", &[("a", "v1"), ("b", "v1")]);
        assert_eq!(pending(&report), actions(&[("a", Action::Process)]));
    }

    #[test]
    fn a_modified_output_is_rebuilt_but_a_rewrite_of_identical_bytes_is_not() {
        let env = env();
        let (mut run, _) = open(&env, "fp", &[("a", "v1"), ("b", "v1")]);
        produce(&env, &mut run, "a", "v1", "alpha");
        produce(&env, &mut run, "b", "v1", "beta");
        run.finish().unwrap();

        std::thread::sleep(std::time::Duration::from_millis(50));
        write_output(&env, "a.md", "ALPHA"); // same size, different bytes
        write_output(&env, "b.md", "beta"); // new mtime, identical bytes

        let (_run, report) = open(&env, "fp", &[("a", "v1"), ("b", "v1")]);
        assert_eq!(pending(&report), actions(&[("a", Action::Process)]));
    }

    #[test]
    fn digest_verification_catches_a_change_that_preserved_size_and_mtime() {
        let env = env();
        let (mut run, _) = open(&env, "fp", &[("a", "v1")]);
        produce(&env, &mut run, "a", "v1", "alpha");
        run.finish().unwrap();

        let target = env.out.join("a.md");
        let before =
            filetime::FileTime::from_last_modification_time(&std::fs::metadata(&target).unwrap());
        std::fs::write(&target, "ALPHA").unwrap();
        filetime::set_file_mtime(&target, before).unwrap();

        let (_run, fast) = open(&env, "fp", &[("a", "v1")]);
        assert!(fast.decisions.is_empty(), "fast mode trusts size and mtime");
        drop(_run);

        let strict = StageSpec {
            verify: VerifyMode::Digest,
            ..spec(&env, "fp")
        };
        let (_run, report) =
            StageRun::open(&env.state, strict, items(&[("a", "v1")]), vec![]).unwrap();
        assert_eq!(pending(&report), actions(&[("a", Action::Process)]));
    }

    #[test]
    fn a_second_run_on_the_same_state_is_refused_while_the_first_is_open() {
        let env = env();
        let (run, _) = open(&env, "fp", &[("a", "v1")]);
        let error = StageRun::open(&env.state, spec(&env, "fp"), items(&[("a", "v1")]), vec![])
            .err()
            .expect("locked");
        assert!(matches!(error, MinSyncError::LockFailed), "{error:?}");
        drop(run);
        open(&env, "fp", &[("a", "v1")]);
    }

    #[test]
    fn state_that_belongs_to_another_corpus_or_stage_is_refused() {
        let env = env();
        let (mut run, _) = open(&env, "fp", &[("a", "v1")]);
        produce(&env, &mut run, "a", "v1", "alpha");
        run.finish().unwrap();

        let other_corpus = StageSpec {
            source_id: "another".into(),
            ..spec(&env, "fp")
        };
        let error = StageRun::open(&env.state, other_corpus, items(&[("a", "v1")]), vec![])
            .err()
            .expect("different corpus");
        assert_eq!(code(error), "source_mismatch");

        let other_stage = StageSpec {
            stage: "chunk".into(),
            ..spec(&env, "fp")
        };
        let error = StageRun::open(&env.state, other_stage, items(&[("a", "v1")]), vec![])
            .err()
            .expect("different stage");
        assert_eq!(code(error), "source_mismatch");
    }

    #[test]
    fn a_corrupt_snapshot_is_an_error_not_an_empty_baseline() {
        let env = env();
        std::fs::create_dir_all(&env.state).unwrap();
        std::fs::write(env.state.join(SNAPSHOT_FILE), "{ truncated").unwrap();
        let error = StageRun::open(&env.state, spec(&env, "fp"), items(&[("a", "v1")]), vec![])
            .err()
            .expect("corrupt snapshot");
        assert_eq!(code(error), "corrupt_state");
    }

    #[test]
    fn invalid_requests_are_rejected_before_any_state_is_written() {
        let env = env();
        let try_open = |spec: StageSpec, list: Vec<SourceItem>, unknown: Vec<String>| {
            code(
                StageRun::open(&env.state, spec, list, unknown)
                    .err()
                    .expect("must be rejected"),
            )
        };
        assert_eq!(
            try_open(spec(&env, "fp"), items(&[("a", "v1"), ("a", "v2")]), vec![]),
            "duplicate_item"
        );
        assert_eq!(
            try_open(spec(&env, "fp"), items(&[("../a", "v1")]), vec![]),
            "invalid_item_key"
        );
        assert_eq!(
            try_open(
                spec(&env, "fp"),
                items(&[("a", "v1")]),
                vec!["a".to_string()]
            ),
            "invalid_request"
        );
        assert_eq!(
            try_open(spec(&env, ""), items(&[("a", "v1")]), vec![]),
            "invalid_request"
        );
        assert_eq!(
            try_open(spec(&env, "fp"), items(&[("a", "")]), vec![]),
            "invalid_request"
        );
        let bad_stage = StageSpec {
            stage: "a b/../c".into(),
            ..spec(&env, "fp")
        };
        assert_eq!(
            try_open(bad_stage, items(&[("a", "v1")]), vec![]),
            "invalid_request"
        );
        assert!(!env.state.join("txn.json").exists());
        assert!(!env.state.join(JOURNAL_FILE).exists());
    }

    #[test]
    fn finish_reports_what_was_left_undone() {
        let env = env();
        let (mut run, _) = open(&env, "fp", &[("a", "v1"), ("b", "v1"), ("c", "v1")]);
        produce(&env, &mut run, "a", "v1", "alpha");
        run.fail("b", "v1", None, "boom", FailOptions::default())
            .unwrap();
        let finished = run.finish().unwrap();
        assert_eq!(finished.committed, 1);
        assert_eq!(finished.failed, 1);
        assert_eq!(finished.pending, 2, "b failed and c was never touched");
    }

    #[test]
    fn journal_and_snapshot_cost_follows_the_work_not_the_corpus() {
        let env = env();
        let corpus: Vec<(String, String)> = (0..400)
            .map(|index| (format!("doc-{index:04}"), "v1".to_string()))
            .collect();
        let refs: Vec<(&str, &str)> = corpus
            .iter()
            .map(|(key, version)| (key.as_str(), version.as_str()))
            .collect();
        let (mut run, _) = open(&env, "fp", &refs);
        for (key, version) in &refs {
            produce(&env, &mut run, key, version, "body");
        }
        run.finish().unwrap();
        let snapshot_path = env.state.join(SNAPSHOT_FILE);
        let snapshot_before = std::fs::read(&snapshot_path).unwrap();

        // One edited document in a 400-item corpus.
        let mut edited = refs.clone();
        edited[200].1 = "v2";
        let (mut run, report) = open(&env, "fp", &edited);
        assert_eq!(report.decisions.len(), 1);
        let journal = env.state.join(JOURNAL_FILE);
        let at_open = std::fs::metadata(&journal).unwrap().len();
        assert!(
            at_open < 1024,
            "a fresh run's journal must not carry the corpus ({at_open} bytes)"
        );
        produce(&env, &mut run, "doc-0200", "v2", "new body");
        let after_commit = std::fs::metadata(&journal).unwrap().len();
        assert!(
            after_commit - at_open < 512,
            "a checkpoint appends one record, not the corpus"
        );
        assert_eq!(
            std::fs::read(&snapshot_path).unwrap(),
            snapshot_before,
            "the snapshot is rewritten at finish, never per checkpoint"
        );
        run.finish().unwrap();
        assert_ne!(std::fs::read(&snapshot_path).unwrap(), snapshot_before);
    }

    fn snapshot_of(env: &Env) -> Snapshot {
        Snapshot::load(&env.state.join(SNAPSHOT_FILE))
            .unwrap()
            .unwrap()
    }

    #[test]
    fn skipping_an_item_removes_any_output_and_the_omission_is_reused() {
        let env = env();
        let (mut run, _) = open(&env, "fp", &[("a", "v1")]);
        produce(&env, &mut run, "a", "v1", "alpha");
        run.finish().unwrap();
        assert!(env.out.join("a.md").exists());

        // v2 is deliberately omitted: the item stays in the source, but it
        // produces nothing, so the v1 output must not be left behind.
        let (mut run, report) = open(&env, "fp", &[("a", "v2")]);
        assert_eq!(pending(&report), actions(&[("a", Action::Process)]));
        run.skip("a", "v2", "unsupported type").unwrap();
        let finished = run.finish().unwrap();
        assert_eq!(finished.committed, 0);
        assert_eq!(finished.failed, 0);
        assert_eq!(finished.pending, 0, "an omission is not work");
        assert!(!env.out.join("a.md").exists(), "the stale output is gone");

        let snapshot = snapshot_of(&env);
        assert!(!snapshot.items.contains_key("a"));
        assert_eq!(snapshot.skipped["a"].version, "v2");
        assert_eq!(snapshot.skipped["a"].fingerprint, "fp");
        assert_eq!(snapshot.skipped["a"].reason, "unsupported type");

        // The same source version is reusable with no output at all.
        let (_run, report) = open(&env, "fp", &[("a", "v2")]);
        assert!(report.decisions.is_empty(), "{:?}", report.decisions);
        assert_eq!(report.counts.skip, 1);
        assert_eq!(report.counts.process, 0);
    }

    /// A snapshot an older build wrote loads at format 1 and stays that way
    /// while it is merely read, but a `finish` that folds an omission into it
    /// must save the format that describes the omission. Recording it under a
    /// format that claims omissions cannot exist would hand the next reader a
    /// lie about the state it just read.
    #[test]
    fn finishing_a_legacy_snapshot_writes_the_current_format() {
        let env = env();
        let (mut run, _) = open(&env, "fp", &[("a", "v1")]);
        produce(&env, &mut run, "a", "v1", "alpha");
        run.finish().unwrap();

        // Stand in for what an older build left behind: the same content,
        // format 1, as if written before omissions existed.
        let mut legacy = snapshot_of(&env);
        legacy.format = 1;
        legacy.save(&env.state.join(SNAPSHOT_FILE)).unwrap();

        let (mut run, report) = open(&env, "fp", &[("a", "v1"), ("b", "v1")]);
        assert_eq!(pending(&report), actions(&[("b", Action::Process)]));
        run.skip("a", "v1", "unsupported type").unwrap();
        run.finish().unwrap();

        let saved = snapshot_of(&env);
        assert_eq!(
            saved.format, SNAPSHOT_FORMAT,
            "the finish records the format its omission needs"
        );
        assert!(saved.items.is_empty());
        assert_eq!(saved.skipped["a"].version, "v1");
    }

    #[test]
    fn a_changed_source_or_fingerprint_reprocesses_an_omitted_item() {
        let env = env();
        let all = [("a", "v1")];
        let (mut run, _) = open(&env, "fp", &all);
        run.skip("a", "v1", "unsupported type").unwrap();
        run.finish().unwrap();

        let (run, report) = open(&env, "fp", &[("a", "v2")]);
        assert_eq!(pending(&report), actions(&[("a", Action::Process)]));
        drop(run); // the state lock lives as long as the run binding

        let (_run, report) = open(&env, "parser-2", &all);
        assert_eq!(
            pending(&report),
            actions(&[("a", Action::Process)]),
            "the host bakes its policy into the source version or fingerprint"
        );
    }

    #[test]
    fn an_omission_survives_a_crash_and_never_needs_an_output_to_verify() {
        let env = env();
        let all = [("a", "v1")];
        let (mut run, _) = open(&env, "fp", &all);
        run.skip("a", "v1", "unsupported type").unwrap();
        drop(run); // killed before finish

        let (run, report) = open(&env, "fp", &all);
        assert!(report.resumed);
        assert!(report.decisions.is_empty(), "{:?}", report.decisions);
        assert_eq!(report.counts.resumed, 1);
        run.finish().unwrap();
        assert_eq!(snapshot_of(&env).skipped["a"].version, "v1");
    }

    #[test]
    fn a_deferred_failure_keeps_the_last_good_output_and_holds_the_item() {
        fn held_spec(
            env: &Env,
            retry_fingerprint: Option<&str>,
            retry_failed: bool,
            fingerprint: &str,
        ) -> StageSpec {
            StageSpec {
                retry_fingerprint: retry_fingerprint.map(str::to_string),
                retry_failed,
                ..spec(env, fingerprint)
            }
        }

        let env = env();
        let (mut run, _) = open(&env, "fp", &[("a", "v1")]);
        produce(&env, &mut run, "a", "v1", "alpha");
        run.finish().unwrap();

        let held = held_spec(&env, Some("retry-1"), false, "fp");
        let (mut run, report) =
            StageRun::open(&env.state, held, items(&[("a", "v2")]), vec![]).unwrap();
        assert_eq!(pending(&report), actions(&[("a", Action::Process)]));
        run.fail(
            "a",
            "v2",
            None,
            "bad xref",
            FailOptions {
                defer: true,
                discard_output: false,
            },
        )
        .unwrap();
        let finished = run.finish().unwrap();
        assert_eq!(finished.failed, 1);
        assert_eq!(finished.pending, 0, "a held failure is not work");
        assert!(
            env.out.join("a.md").exists(),
            "the last good output is preserved"
        );

        // Same source version, settings and retry policy: held, reported as
        // a skip decision with its diagnostic, counted as a skip.
        let (_run, report) = StageRun::open(
            &env.state,
            held_spec(&env, Some("retry-1"), false, "fp"),
            items(&[("a", "v2")]),
            vec![],
        )
        .unwrap();
        assert_eq!(report.counts.skip, 1);
        assert_eq!(report.counts.process, 0);
        assert_eq!(report.counts.retry, 0);
        assert_eq!(report.decisions.len(), 1);
        assert_eq!(report.decisions[0].action, Action::Skip);
        assert_eq!(report.decisions[0].version.as_deref(), Some("v2"));
        assert_eq!(
            report.decisions[0].deferred_error.as_deref(),
            Some("bad xref")
        );
        let snapshot = snapshot_of(&env);
        assert!(
            snapshot.failed["a"].deferred,
            "the held failure is recorded"
        );
        assert_eq!(
            snapshot.items["a"].version, "v1",
            "the failure and the v1 completion coexist"
        );
        drop(_run); // the state lock lives as long as the run binding

        // A changed retry policy, a changed fingerprint, and an explicit
        // retry all release it.
        for released in [
            held_spec(&env, Some("retry-1"), true, "fp"),
            held_spec(&env, Some("retry-2"), false, "fp"),
            held_spec(&env, Some("retry-1"), false, "parser-2"),
        ] {
            let (run, report) =
                StageRun::open(&env.state, released, items(&[("a", "v2")]), vec![]).unwrap();
            assert!(
                matches!(report.decisions[0].action, Action::Retry | Action::Process),
                "{:?}",
                report.decisions
            );
            drop(run);
        }
    }

    #[test]
    fn a_failure_that_discards_the_output_drops_the_completion_and_the_file() {
        let env = env();
        let (mut run, _) = open(&env, "fp", &[("a", "v1")]);
        produce(&env, &mut run, "a", "v1", "alpha");
        run.finish().unwrap();

        let (mut run, _) = open(&env, "fp", &[("a", "v2")]);
        run.fail(
            "a",
            "v2",
            None,
            "corrupt",
            FailOptions {
                defer: false,
                discard_output: true,
            },
        )
        .unwrap();
        run.finish().unwrap();
        assert!(
            !env.out.join("a.md").exists(),
            "a failure only erases the last good output when it is asked to"
        );
        let snapshot = snapshot_of(&env);
        assert!(
            !snapshot.items.contains_key("a"),
            "no completion survives it"
        );
        assert_eq!(snapshot.failed["a"].error, "corrupt");

        // The failure is retried, not held.
        let (_run, report) = open(&env, "fp", &[("a", "v2")]);
        assert_eq!(pending(&report), actions(&[("a", Action::Retry)]));
    }

    #[test]
    fn a_confirmed_gone_item_is_deleted_whatever_state_it_held() {
        let env = env();
        let (mut run, _) = open(
            &env,
            "fp",
            &[("kept", "v1"), ("failed", "v1"), ("skipped", "v1")],
        );
        produce(&env, &mut run, "kept", "v1", "alpha");
        run.fail("failed", "v1", None, "boom", FailOptions::default())
            .unwrap();
        run.skip("skipped", "v1", "unsupported type").unwrap();
        run.finish().unwrap();

        let (mut run, report) = open(&env, "fp", &[("kept", "v1")]);
        assert_eq!(
            pending(&report),
            actions(&[("failed", Action::Delete), ("skipped", Action::Delete)]),
            "a held failure and an omission must be cleared when the source goes"
        );
        assert!(
            !run.delete("failed").unwrap(),
            "the failure left no output to remove"
        );
        assert!(
            !run.delete("skipped").unwrap(),
            "the omission left no output to remove"
        );
        run.finish().unwrap();

        let snapshot = snapshot_of(&env);
        assert!(snapshot.failed.is_empty(), "the failure is cleared");
        assert!(snapshot.skipped.is_empty(), "the omission is cleared");
        assert!(snapshot.items.contains_key("kept"));

        // Nothing is left to delete on the next run.
        let (_run, report) = open(&env, "fp", &[("kept", "v1")]);
        assert!(report.decisions.is_empty(), "{:?}", report.decisions);
    }

    #[test]
    fn deleting_an_item_whose_last_completion_was_discarded_still_clears_the_failure() {
        let env = env();
        let (mut run, _) = open(&env, "fp", &[("a", "v1")]);
        produce(&env, &mut run, "a", "v1", "alpha");
        run.finish().unwrap();

        // The item changes, fails badly enough to discard its output, and
        // then disappears from the source entirely.
        let (mut run, _) = open(&env, "fp", &[("a", "v2")]);
        run.fail(
            "a",
            "v2",
            None,
            "corrupt",
            FailOptions {
                defer: false,
                discard_output: true,
            },
        )
        .unwrap();
        run.finish().unwrap();
        assert!(!env.out.join("a.md").exists());

        let (mut run, report) = open(&env, "fp", &[]);
        assert_eq!(pending(&report), actions(&[("a", Action::Delete)]));
        assert!(!run.delete("a").unwrap());
        run.finish().unwrap();
        let snapshot = snapshot_of(&env);
        assert!(snapshot.failed.is_empty());
        assert!(snapshot.items.is_empty());
    }

    #[test]
    fn the_running_ledger_and_the_compacted_snapshot_agree_record_for_record() {
        let committed = Record::committed("a", "v1", Some(output("a.md")));
        let records = [
            committed.clone(),
            Record::skipped("b", "v1", "unsupported type"),
            Record::failed("c", Some("v1"), "boom"),
            Record::failed_with("d", Some("v1"), "held", true, Some("retry-1".into()), false),
            Record::committed("e", "v1", Some(output("e.md"))),
            Record::failed_with("e", Some("v2"), "corrupt", false, None, true),
            Record::skipped("f", "v1", "unsupported type"),
            Record::committed("f", "v2", Some(output("f.md"))),
            Record::deleted("g"),
            Record::failed_with("h", Some("v1"), "held", true, None, false),
            Record::skipped("h", "v2", "unsupported type"),
            Record::committed("i", "v1", Some(output("i.md"))),
            Record::failed_with("i", Some("v2"), "later", false, None, false),
        ];
        let mut ledger = Ledger::new(RunHeader::new("parse", "corpus", false, "base", "fp"));
        let mut snapshot = Snapshot::new("parse", "corpus");
        for record in &records {
            ledger.apply(record.clone());
            snapshot.apply_record(record, "fp");
        }

        for key in ["a", "b", "c", "d", "e", "f", "g", "h", "i"] {
            let done = ledger.done(key).map(|record| record.outcome);
            assert_eq!(
                snapshot.items.contains_key(key),
                done == Some(Outcome::Committed),
                "{key}: completions"
            );
            assert_eq!(
                snapshot.skipped.contains_key(key),
                ledger.skipped(key).is_some(),
                "{key}: omissions"
            );
            assert_eq!(
                snapshot.failed.contains_key(key),
                ledger.failed(key).is_some(),
                "{key}: failures"
            );
        }
        // A completion stands alone.
        assert!(snapshot.items.contains_key("a"));
        assert!(!snapshot.failed.contains_key("a"));
        // A failure keeps the last good completion of the same item...
        assert!(snapshot.items.contains_key("i"));
        assert!(snapshot.failed.contains_key("i"));
        assert_eq!(snapshot.items["i"].version, "v1");
        assert_eq!(snapshot.failed["i"].version.as_deref(), Some("v2"));
        // ...unless the failure discarded that output.
        assert!(!snapshot.items.contains_key("e"));
        assert!(snapshot.failed.contains_key("e"));
        // A completion replaces an omission; an omission clears a failure.
        assert!(snapshot.items.contains_key("f"));
        assert!(!snapshot.skipped.contains_key("f"));
        assert!(snapshot.skipped.contains_key("h"));
        assert!(!snapshot.failed.contains_key("h"));
        // The fingerprint state travels with the record it belongs to.
        assert_eq!(snapshot.items["a"].fingerprint, "fp");
        assert_eq!(snapshot.skipped["b"].fingerprint, "fp");
        assert!(snapshot.failed["d"].deferred);
        assert_eq!(snapshot.failed["d"].fingerprint.as_deref(), Some("fp"));
        assert_eq!(
            snapshot.failed["d"].retry_fingerprint.as_deref(),
            Some("retry-1")
        );
    }
}
