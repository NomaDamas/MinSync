//! Resume planning: what a stage still has to do, given the current source
//! items, the last committed snapshot and the ledger of an interrupted run.
//!
//! The planner is pure: it never touches the filesystem (output verification
//! is a callback) and never performs work. It produces one decision per
//! item and the stage adapter acts on it.
//!
//! Besides committed outputs, the planner understands two states that are
//! deliberately *not* work: a deferred failure (a failure the host asked not
//! to retry until its source, the processing fingerprint or the retry
//! fingerprint changes) and a deliberate omission (`skip`: the host decided
//! the item produces no output). Both are reusable while their source
//! version and processing fingerprint match, and both are released by the
//! host asking for a retry.

use super::journal::{Ledger, Outcome, OutputRef};
use super::snapshot::Snapshot;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

/// One current source item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceItem {
    pub key: String,
    /// Source version: a content hash, or any host-chosen string that
    /// changes whenever the source content does.
    pub version: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Action {
    /// New or changed since its recorded completion (or its completion no
    /// longer verifies): do the work.
    Process,
    /// Failed before for this exact source version: try again.
    Retry,
    /// Already committed for this version and fingerprint: nothing to do.
    /// Also used for an item that is deliberately omitted or whose failure
    /// is deferred: it is not work now.
    Skip,
    /// Source confirmed gone: remove its output.
    Delete,
    /// Source unreadable or unobservable: keep the previous output.
    Preserve,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decision {
    pub key: String,
    pub action: Action,
    /// Source version the decision is about (`None` for deletions and
    /// preserved items).
    pub version: Option<String>,
    /// A skip that rests on the interrupted run's ledger rather than on the
    /// committed snapshot.
    pub resumed: bool,
    /// A `skip` that rests on a deferred failure: the failure's diagnostic.
    /// The item keeps its last good output and is not retried.
    pub deferred_error: Option<String>,
}

pub struct PlanInput<'a> {
    pub items: &'a [SourceItem],
    /// Keys the source could not be observed for: never deleted.
    pub unknown: &'a BTreeSet<String>,
    /// Processing fingerprint of this run.
    pub fingerprint: &'a str,
    /// Host-supplied retry fingerprint of this run (`None` when absent).
    pub retry_fingerprint: Option<&'a str>,
    /// A full run reuses only what it committed itself (via its ledger) and
    /// releases every deferred failure.
    pub full: bool,
    /// The host asked for deferred failures to be retried.
    pub retry_failed: bool,
    pub snapshot: &'a Snapshot,
    /// Ledger of the interrupted run being resumed (empty for a fresh run).
    pub ledger: &'a Ledger,
    /// Whether a recorded output is still present and unchanged.
    pub output_ok: &'a dyn Fn(&OutputRef) -> bool,
}

impl PlanInput<'_> {
    /// Whether deferred failures are released rather than held.
    fn releases_deferred(&self) -> bool {
        self.full || self.retry_failed
    }
}

/// The completion an item currently has: the interrupted run's ledger wins
/// over the committed snapshot.
pub(super) struct Completion<'a> {
    pub version: &'a str,
    pub fingerprint: &'a str,
    pub output: Option<&'a OutputRef>,
    pub from_ledger: bool,
}

/// `None` = no output exists for `key` (never committed, or deleted).
pub(super) fn completion<'a>(
    snapshot: &'a Snapshot,
    ledger: &'a Ledger,
    key: &str,
) -> Option<Completion<'a>> {
    match ledger.done(key) {
        Some(record) if record.outcome == Outcome::Committed => Some(Completion {
            version: record.version.as_deref().unwrap_or_default(),
            fingerprint: &ledger.header.fingerprint,
            output: record.output.as_ref(),
            from_ledger: true,
        }),
        Some(_) => None,
        None => snapshot.items.get(key).map(|item| Completion {
            version: &item.version,
            fingerprint: &item.fingerprint,
            output: Some(&item.output),
            from_ledger: false,
        }),
    }
}

/// A deliberate omission of `key`: no output is expected for it.
struct Omission<'a> {
    version: &'a str,
    fingerprint: &'a str,
    from_ledger: bool,
}

/// `None` = the item was not deliberately omitted under this baseline.
fn omission<'a>(snapshot: &'a Snapshot, ledger: &'a Ledger, key: &str) -> Option<Omission<'a>> {
    match ledger.skipped(key) {
        Some(record) => Some(Omission {
            version: record.version.as_deref().unwrap_or_default(),
            fingerprint: &ledger.header.fingerprint,
            from_ledger: true,
        }),
        // A completion or deletion of this run supersedes an older omission.
        None if ledger.done(key).is_none() => snapshot.skipped.get(key).map(|skipped| Omission {
            version: &skipped.version,
            fingerprint: &skipped.fingerprint,
            from_ledger: false,
        }),
        None => None,
    }
}

/// A deferred failure of `key`, with what it was deferred under.
struct Deferral<'a> {
    version: Option<&'a str>,
    fingerprint: Option<&'a str>,
    retry_fingerprint: Option<&'a str>,
    error: &'a str,
}

/// `None` = the item's latest failure, if any, is not deferred.
fn deferral<'a>(snapshot: &'a Snapshot, ledger: &'a Ledger, key: &str) -> Option<Deferral<'a>> {
    if let Some(record) = ledger.failed(key) {
        return record.deferred.then_some(Deferral {
            version: record.version.as_deref(),
            fingerprint: Some(&ledger.header.fingerprint),
            retry_fingerprint: record.retry_fingerprint.as_deref(),
            error: record.error.as_deref().unwrap_or_default(),
        });
    }
    snapshot
        .failed
        .get(key)
        .filter(|failure| failure.deferred)
        .map(|failure| Deferral {
            version: failure.version.as_deref(),
            fingerprint: failure.fingerprint.as_deref(),
            retry_fingerprint: failure.retry_fingerprint.as_deref(),
            error: &failure.error,
        })
}

/// Whether `key` holds any failure, deferred or not.
fn failed_somewhere(snapshot: &Snapshot, ledger: &Ledger, key: &str) -> bool {
    ledger.failed(key).is_some() || snapshot.failed.contains_key(key)
}

/// Whether `key` already failed for exactly `version`.
fn failed_before(snapshot: &Snapshot, ledger: &Ledger, key: &str, version: &str) -> bool {
    let in_ledger = ledger
        .failed(key)
        .is_some_and(|record| record.version.as_deref() == Some(version));
    in_ledger
        || snapshot
            .failed
            .get(key)
            .is_some_and(|failure| failure.version.as_deref() == Some(version))
}

/// One decision per item, in key order.
pub fn plan(input: &PlanInput<'_>) -> Vec<Decision> {
    let mut decisions: BTreeMap<&str, Decision> = BTreeMap::new();
    let mut present: BTreeSet<&str> = BTreeSet::new();

    for item in input.items {
        present.insert(&item.key);
        let reusable = completion(input.snapshot, input.ledger, &item.key).filter(|done| {
            done.version == item.version
                && done.fingerprint == input.fingerprint
                && (!input.full || done.from_ledger)
                && done.output.is_some_and(|output| (input.output_ok)(output))
        });
        if let Some(done) = reusable {
            decisions.insert(
                &item.key,
                skip(&item.key, Some(item.version.as_str()), done.from_ledger),
            );
            continue;
        }
        // An item the host deliberately omitted has no output to verify: its
        // version and the fingerprint are the whole record.
        let omitted = omission(input.snapshot, input.ledger, &item.key).filter(|omitted| {
            omitted.version == item.version && omitted.fingerprint == input.fingerprint
        });
        if let Some(omitted) = omitted {
            decisions.insert(
                &item.key,
                skip(&item.key, Some(item.version.as_str()), omitted.from_ledger),
            );
            continue;
        }
        // A deferred failure holds until its source, its settings or the
        // retry policy change, or the host asks for a retry.
        if let Some(deferral) = deferral(input.snapshot, input.ledger, &item.key) {
            let held = !input.releases_deferred()
                && deferral.version == Some(item.version.as_str())
                && deferral.fingerprint == Some(input.fingerprint)
                && deferral.retry_fingerprint == input.retry_fingerprint;
            if held {
                decisions.insert(
                    &item.key,
                    Decision {
                        key: item.key.clone(),
                        action: Action::Skip,
                        version: Some(item.version.clone()),
                        resumed: false,
                        deferred_error: Some(deferral.error.to_string()),
                    },
                );
                continue;
            }
        }
        let action = if failed_before(input.snapshot, input.ledger, &item.key, &item.version) {
            Action::Retry
        } else {
            Action::Process
        };
        decisions.insert(
            &item.key,
            Decision {
                key: item.key.clone(),
                action,
                version: Some(item.version.clone()),
                resumed: false,
                deferred_error: None,
            },
        );
    }

    for key in input.unknown {
        decisions.entry(key).or_insert_with(|| Decision {
            key: key.clone(),
            action: Action::Preserve,
            version: None,
            resumed: false,
            deferred_error: None,
        });
    }

    // Anything whose source is confirmed gone and that still holds durable
    // state gets a deletion: an output to remove, an omission to forget, or
    // a failure to clear. An item the ledger itself cannot place is left to
    // the next run.
    let known = input
        .snapshot
        .items
        .keys()
        .chain(input.snapshot.skipped.keys())
        .chain(input.snapshot.failed.keys())
        .map(String::as_str)
        .chain(input.ledger.records().map(|record| record.item.as_str()));
    for key in known {
        if present.contains(key) || input.unknown.contains(key) {
            continue;
        }
        let holds_state = completion(input.snapshot, input.ledger, key).is_some()
            || omission(input.snapshot, input.ledger, key).is_some()
            || failed_somewhere(input.snapshot, input.ledger, key);
        if !holds_state {
            continue;
        }
        decisions.entry(key).or_insert_with(|| Decision {
            key: key.to_string(),
            action: Action::Delete,
            version: None,
            resumed: false,
            deferred_error: None,
        });
    }

    decisions.into_values().collect()
}

/// A `skip` decision for a present item.
fn skip(key: &str, version: Option<&str>, resumed: bool) -> Decision {
    Decision {
        key: key.to_string(),
        action: Action::Skip,
        version: version.map(str::to_string),
        resumed,
        deferred_error: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::incremental::journal::{OutputRef, Record, RunHeader};
    use crate::incremental::snapshot::{FailedItem, SkippedItem, Snapshot, SnapshotItem};
    use std::collections::BTreeSet;

    const FP: &str = "fp";

    fn source(key: &str, version: &str) -> SourceItem {
        SourceItem {
            key: key.to_string(),
            version: version.to_string(),
        }
    }

    fn output(path: &str) -> OutputRef {
        OutputRef {
            path: path.to_string(),
            size: 1,
            digest: "sha256:x".to_string(),
            mtime_ns: 1,
        }
    }

    fn entry(key: &str, version: &str, fingerprint: &str) -> SnapshotItem {
        SnapshotItem {
            version: version.to_string(),
            fingerprint: fingerprint.to_string(),
            output: output(&format!("{key}.md")),
        }
    }

    fn snapshot(entries: &[(&str, &str)]) -> Snapshot {
        let mut snapshot = Snapshot::new("parse", "corpus");
        for (key, version) in entries {
            snapshot
                .items
                .insert(key.to_string(), entry(key, version, FP));
        }
        snapshot
    }

    fn ledger(records: Vec<Record>) -> Ledger {
        let mut ledger = Ledger::new(RunHeader::new("parse", "corpus", false, "base", FP));
        for record in records {
            ledger.apply(record);
        }
        ledger
    }

    fn committed(key: &str, version: &str) -> Record {
        Record::committed(key, version, Some(output(&format!("{key}.md"))))
    }

    /// A committed (not deferred) failure of `version`.
    fn failure(version: &str, error: &str) -> FailedItem {
        FailedItem {
            version: Some(version.into()),
            error: error.into(),
            fingerprint: Some(FP.into()),
            deferred: false,
            retry_fingerprint: None,
        }
    }

    /// A failure the host asked not to retry.
    fn deferred(version: &str, error: &str, fingerprint: &str, retry: Option<&str>) -> FailedItem {
        FailedItem {
            version: Some(version.into()),
            error: error.into(),
            fingerprint: Some(fingerprint.into()),
            deferred: true,
            retry_fingerprint: retry.map(str::to_string),
        }
    }

    struct Case {
        items: Vec<SourceItem>,
        unknown: BTreeSet<String>,
        snapshot: Snapshot,
        ledger: Ledger,
        full: bool,
        retry_failed: bool,
        fingerprint: &'static str,
        retry_fingerprint: Option<&'static str>,
    }

    impl Case {
        fn new(items: Vec<SourceItem>) -> Self {
            Self {
                items,
                unknown: BTreeSet::new(),
                snapshot: snapshot(&[]),
                ledger: ledger(vec![]),
                full: false,
                retry_failed: false,
                fingerprint: FP,
                retry_fingerprint: None,
            }
        }

        fn plan_with(&self, output_ok: &dyn Fn(&OutputRef) -> bool) -> Vec<(String, Action, bool)> {
            plan(&PlanInput {
                items: &self.items,
                unknown: &self.unknown,
                fingerprint: self.fingerprint,
                retry_fingerprint: self.retry_fingerprint,
                full: self.full,
                retry_failed: self.retry_failed,
                snapshot: &self.snapshot,
                ledger: &self.ledger,
                output_ok,
            })
            .into_iter()
            .map(|decision| (decision.key, decision.action, decision.resumed))
            .collect()
        }

        /// The decisions with their deferred-failure diagnostic, if any.
        fn detailed(&self) -> Vec<(String, Action, Option<String>)> {
            plan(&PlanInput {
                items: &self.items,
                unknown: &self.unknown,
                fingerprint: self.fingerprint,
                retry_fingerprint: self.retry_fingerprint,
                full: self.full,
                retry_failed: self.retry_failed,
                snapshot: &self.snapshot,
                ledger: &self.ledger,
                output_ok: &|_| true,
            })
            .into_iter()
            .map(|decision| (decision.key, decision.action, decision.deferred_error))
            .collect()
        }

        fn actions(&self) -> Vec<(String, Action)> {
            self.plan_with(&|_| true)
                .into_iter()
                .map(|(key, action, _)| (key, action))
                .collect()
        }
    }

    fn pairs(list: &[(&str, Action)]) -> Vec<(String, Action)> {
        list.iter()
            .map(|(key, action)| (key.to_string(), *action))
            .collect()
    }

    #[test]
    fn new_items_are_processed_in_key_order() {
        let case = Case::new(vec![source("b", "v"), source("a", "v")]);
        assert_eq!(
            case.actions(),
            pairs(&[("a", Action::Process), ("b", Action::Process)])
        );
    }

    #[test]
    fn unchanged_items_are_skipped_and_changed_ones_processed() {
        let mut case = Case::new(vec![source("same", "v1"), source("edited", "v2")]);
        case.snapshot = snapshot(&[("same", "v1"), ("edited", "v1")]);
        assert_eq!(
            case.actions(),
            pairs(&[("edited", Action::Process), ("same", Action::Skip)])
        );
    }

    #[test]
    fn a_processing_fingerprint_change_invalidates_every_old_completion() {
        let mut case = Case::new(vec![source("a", "v1"), source("b", "v1")]);
        case.snapshot = snapshot(&[("a", "v1"), ("b", "v1")]);
        case.fingerprint = "parser-v2";
        assert_eq!(
            case.actions(),
            pairs(&[("a", Action::Process), ("b", Action::Process)])
        );
    }

    #[test]
    fn an_output_that_no_longer_verifies_is_rebuilt_not_skipped() {
        let mut case = Case::new(vec![source("a", "v1"), source("b", "v1")]);
        case.snapshot = snapshot(&[("a", "v1"), ("b", "v1")]);
        let decisions = case.plan_with(&|output| output.path != "a.md");
        assert_eq!(
            decisions
                .into_iter()
                .map(|(key, action, _)| (key, action))
                .collect::<Vec<_>>(),
            pairs(&[("a", Action::Process), ("b", Action::Skip)])
        );
    }

    #[test]
    fn a_full_run_only_skips_what_this_run_already_committed() {
        let mut case = Case::new(vec![source("old", "v1"), source("done", "v1")]);
        case.full = true;
        case.snapshot = snapshot(&[("old", "v1"), ("done", "v1")]);
        case.ledger = ledger(vec![committed("done", "v1")]);
        assert_eq!(
            case.actions(),
            pairs(&[("done", Action::Skip), ("old", Action::Process)])
        );
    }

    #[test]
    fn interrupted_run_completions_are_skipped_and_flagged_resumed() {
        let mut case = Case::new(vec![source("a", "v1"), source("b", "v1")]);
        case.ledger = ledger(vec![committed("a", "v1")]);
        assert_eq!(
            case.plan_with(&|_| true),
            vec![
                ("a".to_string(), Action::Skip, true),
                ("b".to_string(), Action::Process, false)
            ]
        );
    }

    #[test]
    fn a_completion_for_an_older_version_is_reprocessed() {
        let mut case = Case::new(vec![source("a", "v2")]);
        case.ledger = ledger(vec![committed("a", "v1")]);
        assert_eq!(case.actions(), pairs(&[("a", Action::Process)]));
    }

    #[test]
    fn a_ledger_completion_with_a_missing_output_is_rebuilt() {
        let mut case = Case::new(vec![source("a", "v1")]);
        case.ledger = ledger(vec![committed("a", "v1")]);
        let decisions = case.plan_with(&|_| false);
        assert_eq!(decisions, vec![("a".to_string(), Action::Process, false)]);
    }

    #[test]
    fn failed_items_are_retried_until_their_source_changes() {
        let mut case = Case::new(vec![
            source("snap-fail", "v2"),
            source("ledger-fail", "v2"),
            source("changed-since", "v3"),
        ]);
        case.snapshot
            .failed
            .insert("snap-fail".into(), failure("v2", "boom"));
        case.snapshot
            .failed
            .insert("changed-since".into(), failure("v2", "boom"));
        case.ledger = ledger(vec![Record::failed("ledger-fail", Some("v2"), "boom")]);
        assert_eq!(
            case.actions(),
            pairs(&[
                ("changed-since", Action::Process),
                ("ledger-fail", Action::Retry),
                ("snap-fail", Action::Retry),
            ])
        );
    }

    #[test]
    fn a_failure_never_hides_an_earlier_valid_completion() {
        // v1 is committed; v2 failed in this run. The item is at v2 now, so
        // it is retried, not skipped on the strength of v1.
        let mut case = Case::new(vec![source("a", "v2")]);
        case.snapshot = snapshot(&[("a", "v1")]);
        case.ledger = ledger(vec![Record::failed("a", Some("v2"), "boom")]);
        assert_eq!(case.actions(), pairs(&[("a", Action::Retry)]));
    }

    #[test]
    fn confirmed_absent_items_are_deleted_once() {
        let mut case = Case::new(vec![source("kept", "v1")]);
        case.snapshot = snapshot(&[("kept", "v1"), ("gone", "v1"), ("gone-done", "v1")]);
        case.ledger = ledger(vec![Record::deleted("gone-done")]);
        assert_eq!(
            case.actions(),
            pairs(&[("gone", Action::Delete), ("kept", Action::Skip)])
        );
    }

    #[test]
    fn unknown_items_are_preserved_and_never_deleted() {
        let mut case = Case::new(vec![source("a", "v1")]);
        case.snapshot = snapshot(&[("a", "v1"), ("unreadable", "v1")]);
        case.unknown.insert("unreadable".into());
        case.unknown.insert("never-seen".into());
        assert_eq!(
            case.actions(),
            pairs(&[
                ("a", Action::Skip),
                ("never-seen", Action::Preserve),
                ("unreadable", Action::Preserve),
            ])
        );
    }

    #[test]
    fn a_recorded_deletion_followed_by_a_return_is_processed_again() {
        let mut case = Case::new(vec![source("a", "v1")]);
        case.snapshot = snapshot(&[("a", "v1")]);
        case.ledger = ledger(vec![Record::deleted("a")]);
        assert_eq!(case.actions(), pairs(&[("a", Action::Process)]));
    }

    fn omitted(snapshot: &mut Snapshot, key: &str, version: &str, fingerprint: &str) {
        snapshot.skipped.insert(
            key.to_string(),
            SkippedItem {
                version: version.into(),
                fingerprint: fingerprint.into(),
                reason: "unsupported type".into(),
            },
        );
    }

    #[test]
    fn a_deliberate_omission_is_reused_without_an_output() {
        let mut case = Case::new(vec![source("a", "v1")]);
        omitted(&mut case.snapshot, "a", "v1", FP);
        // The omitted item has no output at all; verification cannot matter.
        assert_eq!(
            case.plan_with(&|_| false),
            vec![("a".to_string(), Action::Skip, false)]
        );
    }

    #[test]
    fn a_changed_source_or_fingerprint_reprocesses_an_omitted_item() {
        let mut case = Case::new(vec![source("a", "v1")]);
        omitted(&mut case.snapshot, "a", "v1", FP);
        case.fingerprint = "parser-v2";
        assert_eq!(case.actions(), pairs(&[("a", Action::Process)]));

        let mut case = Case::new(vec![source("a", "v2")]);
        omitted(&mut case.snapshot, "a", "v1", FP);
        assert_eq!(case.actions(), pairs(&[("a", Action::Process)]));
    }

    #[test]
    fn this_runs_omission_is_skipped_and_flagged_resumed() {
        let mut case = Case::new(vec![source("a", "v1")]);
        case.ledger = ledger(vec![Record::skipped("a", "v1", "unsupported type")]);
        assert_eq!(
            case.plan_with(&|_| false),
            vec![("a".to_string(), Action::Skip, true)]
        );
    }

    #[test]
    fn a_completion_of_this_run_supersedes_an_older_omission() {
        let mut case = Case::new(vec![source("a", "v1")]);
        omitted(&mut case.snapshot, "a", "v1", FP);
        case.ledger = ledger(vec![committed("a", "v1")]);
        // The ledger committed a real output for the same version, so the
        // stale omission must not hide it.
        assert_eq!(
            case.plan_with(&|_| false),
            vec![("a".to_string(), Action::Process, false)]
        );
    }

    #[test]
    fn a_deferred_failure_is_held_as_a_skip_with_its_diagnostic() {
        let mut case = Case::new(vec![source("a", "v2")]);
        case.snapshot
            .failed
            .insert("a".into(), deferred("v2", "bad xref", FP, None));
        assert_eq!(
            case.detailed(),
            vec![("a".into(), Action::Skip, Some("bad xref".into()))]
        );
    }

    #[test]
    fn a_deferred_failure_still_reports_the_last_good_completion() {
        // v1 is committed; v2 failed and was deferred. The item is not work
        // now, and the v1 output is still on disk.
        let mut case = Case::new(vec![source("a", "v2")]);
        case.snapshot = snapshot(&[("a", "v1")]);
        case.snapshot
            .failed
            .insert("a".into(), deferred("v2", "bad xref", FP, None));
        assert_eq!(
            case.detailed(),
            vec![("a".into(), Action::Skip, Some("bad xref".into()))]
        );
    }

    #[test]
    fn a_changed_source_fingerprint_or_retry_fingerprint_releases_a_deferred_failure() {
        fn base(
            items: Vec<SourceItem>,
            fingerprint: &'static str,
            retry: Option<&'static str>,
        ) -> Case {
            let mut case = Case::new(items);
            case.snapshot
                .failed
                .insert("a".into(), deferred("v2", "bad xref", FP, Some("retry-1")));
            case.fingerprint = fingerprint;
            case.retry_fingerprint = retry;
            case
        }

        // Source changed: the deferred failure is for another version.
        assert_eq!(
            base(vec![source("a", "v3")], FP, Some("retry-1")).actions(),
            pairs(&[("a", Action::Process)])
        );
        // Settings changed.
        assert_eq!(
            base(vec![source("a", "v2")], "parser-v2", Some("retry-1")).actions(),
            pairs(&[("a", Action::Retry)])
        );
        // Retry policy changed.
        assert_eq!(
            base(vec![source("a", "v2")], FP, Some("retry-2")).actions(),
            pairs(&[("a", Action::Retry)])
        );
        // Nothing changed: held.
        assert_eq!(
            base(vec![source("a", "v2")], FP, Some("retry-1")).actions(),
            pairs(&[("a", Action::Skip)])
        );
    }

    #[test]
    fn retry_failed_and_full_release_deferred_failures() {
        let mut case = Case::new(vec![source("a", "v2")]);
        case.snapshot
            .failed
            .insert("a".into(), deferred("v2", "bad xref", FP, None));
        case.retry_failed = true;
        assert_eq!(case.actions(), pairs(&[("a", Action::Retry)]));

        let mut case = Case::new(vec![source("a", "v2")]);
        case.snapshot
            .failed
            .insert("a".into(), deferred("v2", "bad xref", FP, None));
        case.full = true;
        assert_eq!(case.actions(), pairs(&[("a", Action::Retry)]));
    }

    #[test]
    fn a_legacy_failure_without_a_recorded_fingerprint_is_released_not_held() {
        let mut case = Case::new(vec![source("a", "v2")]);
        case.snapshot.failed.insert(
            "a".into(),
            FailedItem {
                version: Some("v2".into()),
                error: "bad xref".into(),
                fingerprint: None,
                deferred: true,
                retry_fingerprint: None,
            },
        );
        assert_eq!(case.actions(), pairs(&[("a", Action::Retry)]));
    }

    #[test]
    fn this_runs_ordinary_failure_supersedes_a_stored_deferral() {
        let mut case = Case::new(vec![source("a", "v2")]);
        case.snapshot
            .failed
            .insert("a".into(), deferred("v2", "old", FP, None));
        case.ledger = ledger(vec![Record::failed("a", Some("v2"), "new")]);
        assert_eq!(case.actions(), pairs(&[("a", Action::Retry)]));
    }

    #[test]
    fn a_deferred_failure_of_this_run_is_held_without_dropping_a_completion() {
        let mut case = Case::new(vec![source("a", "v2")]);
        case.ledger = ledger(vec![
            committed("a", "v1"),
            Record::failed_with(
                "a",
                Some("v2"),
                "bad xref",
                true,
                Some("retry-1".into()),
                false,
            ),
        ]);
        case.retry_fingerprint = Some("retry-1");
        assert_eq!(
            case.detailed(),
            vec![("a".to_string(), Action::Skip, Some("bad xref".to_string()))]
        );
    }
}
