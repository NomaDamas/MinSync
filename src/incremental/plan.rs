//! Resume planning: what a stage still has to do, given the current source
//! items, the last committed snapshot and the ledger of an interrupted run.
//!
//! The planner is pure: it never touches the filesystem (output verification
//! is a callback) and never performs work. It produces one decision per
//! item and the stage adapter acts on it.

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
    /// Source version to process (`None` for deletions and preserved items).
    pub version: Option<String>,
    /// A skip that rests on the interrupted run's ledger rather than on the
    /// committed snapshot.
    pub resumed: bool,
}

pub struct PlanInput<'a> {
    pub items: &'a [SourceItem],
    /// Keys the source could not be observed for: never deleted.
    pub unknown: &'a BTreeSet<String>,
    /// Processing fingerprint of this run.
    pub fingerprint: &'a str,
    /// A full run reuses only what it committed itself (via its ledger).
    pub full: bool,
    pub snapshot: &'a Snapshot,
    /// Ledger of the interrupted run being resumed (empty for a fresh run).
    pub ledger: &'a Ledger,
    /// Whether a recorded output is still present and unchanged.
    pub output_ok: &'a dyn Fn(&OutputRef) -> bool,
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
        let (action, resumed) = match reusable {
            Some(done) => (Action::Skip, done.from_ledger),
            None if failed_before(input.snapshot, input.ledger, &item.key, &item.version) => {
                (Action::Retry, false)
            }
            None => (Action::Process, false),
        };
        decisions.insert(
            &item.key,
            Decision {
                key: item.key.clone(),
                action,
                version: (action != Action::Skip).then(|| item.version.clone()),
                resumed,
            },
        );
    }

    for key in input.unknown {
        decisions.entry(key).or_insert_with(|| Decision {
            key: key.clone(),
            action: Action::Preserve,
            version: None,
            resumed: false,
        });
    }

    // Anything with an output that is neither present nor merely unobserved
    // has had its source confirmed gone.
    let known = input.snapshot.items.keys().map(String::as_str).chain(
        input
            .ledger
            .records()
            .filter(|record| record.outcome == Outcome::Committed)
            .map(|record| record.item.as_str()),
    );
    for key in known {
        if present.contains(key)
            || input.unknown.contains(key)
            || completion(input.snapshot, input.ledger, key).is_none()
        {
            continue;
        }
        decisions.entry(key).or_insert_with(|| Decision {
            key: key.to_string(),
            action: Action::Delete,
            version: None,
            resumed: false,
        });
    }

    decisions.into_values().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::incremental::journal::{OutputRef, Record, RunHeader};
    use crate::incremental::snapshot::{FailedItem, Snapshot, SnapshotItem};
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

    struct Case {
        items: Vec<SourceItem>,
        unknown: BTreeSet<String>,
        snapshot: Snapshot,
        ledger: Ledger,
        full: bool,
        fingerprint: &'static str,
    }

    impl Case {
        fn new(items: Vec<SourceItem>) -> Self {
            Self {
                items,
                unknown: BTreeSet::new(),
                snapshot: snapshot(&[]),
                ledger: ledger(vec![]),
                full: false,
                fingerprint: FP,
            }
        }

        fn plan_with(&self, output_ok: &dyn Fn(&OutputRef) -> bool) -> Vec<(String, Action, bool)> {
            plan(&PlanInput {
                items: &self.items,
                unknown: &self.unknown,
                fingerprint: self.fingerprint,
                full: self.full,
                snapshot: &self.snapshot,
                ledger: &self.ledger,
                output_ok,
            })
            .into_iter()
            .map(|decision| (decision.key, decision.action, decision.resumed))
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
        case.snapshot.failed.insert(
            "snap-fail".into(),
            FailedItem {
                version: Some("v2".into()),
                error: "boom".into(),
            },
        );
        case.snapshot.failed.insert(
            "changed-since".into(),
            FailedItem {
                version: Some("v2".into()),
                error: "boom".into(),
            },
        );
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
}
