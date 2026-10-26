//! The committed state of a stage between runs.
//!
//! The journal records what a run committed *while it runs*; at the end of a
//! successful run it is compacted into this snapshot (one atomic write) and
//! deleted. The snapshot is therefore never rewritten per item, only once per
//! run, and a crash at any point leaves either the old snapshot plus a
//! resumable journal, or the new snapshot.

use super::fsio::atomic_write;
use super::journal::{Outcome, OutputRef, Record};
use crate::error::{MinSyncError, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;

pub const SNAPSHOT_FILE: &str = "snapshot.json";
/// Snapshot format written by this build. Format 1 stored only committed
/// items and failures; format 2 adds deliberate omissions (`skipped`) and the
/// deferred-failure fields. A snapshot from a newer build is refused.
pub const SNAPSHOT_FORMAT: u32 = 2;

/// A committed item: the source version and processing fingerprint it was
/// built under, and the output that holds the result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotItem {
    pub version: String,
    pub fingerprint: String,
    pub output: OutputRef,
}

/// The latest failure of an item that has no newer commit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FailedItem {
    pub version: Option<String>,
    pub error: String,
    /// The processing fingerprint the item failed under; `None` in snapshots
    /// written before it was recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fingerprint: Option<String>,
    /// The failure is deferred: it is not retried while the source version,
    /// processing fingerprint and retry fingerprint stay the same.
    #[serde(default, skip_serializing_if = "is_false")]
    pub deferred: bool,
    /// The retry fingerprint in effect when the failure was deferred.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_fingerprint: Option<String>,
}

/// A deliberate, successful omission: the item has no output and none is
/// expected while its source version and processing fingerprint stay the
/// same.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkippedItem {
    pub version: String,
    pub fingerprint: String,
    pub reason: String,
}

fn is_false(value: &bool) -> bool {
    !*value
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Snapshot {
    pub format: u32,
    pub source_id: String,
    pub stage: String,
    pub updated_at: String,
    pub items: BTreeMap<String, SnapshotItem>,
    #[serde(default)]
    pub skipped: BTreeMap<String, SkippedItem>,
    pub failed: BTreeMap<String, FailedItem>,
}

impl Snapshot {
    pub fn new(stage: &str, source_id: &str) -> Self {
        Self {
            format: SNAPSHOT_FORMAT,
            source_id: source_id.to_string(),
            stage: stage.to_string(),
            updated_at: chrono::Utc::now().to_rfc3339(),
            items: BTreeMap::new(),
            skipped: BTreeMap::new(),
            failed: BTreeMap::new(),
        }
    }

    /// `Ok(None)` when no snapshot exists yet. A snapshot that exists but
    /// cannot be trusted is an error, not an empty baseline: treating it as
    /// empty would orphan every output it described.
    pub fn load(path: &Path) -> Result<Option<Self>> {
        let bytes = match std::fs::read(path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(MinSyncError::Io(error)),
        };
        let corrupt = |reason: String| {
            MinSyncError::stage(
                "corrupt_state",
                format!(
                    "{} is unusable ({reason}); move it aside to rebuild this stage from scratch",
                    path.display()
                ),
            )
        };
        let snapshot: Self =
            serde_json::from_slice(&bytes).map_err(|error| corrupt(error.to_string()))?;
        if snapshot.format > SNAPSHOT_FORMAT {
            return Err(corrupt(format!(
                "format {} is newer than supported format {SNAPSHOT_FORMAT}",
                snapshot.format
            )));
        }
        Ok(Some(snapshot))
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        atomic_write(path, &serde_json::to_vec(self)?)
    }

    /// One ledger record's effect on the durable state: the single rule the
    /// running [`Ledger`](super::journal::Ledger) and the committed snapshot
    /// agree on.
    ///
    /// * a completion replaces an omission and clears the failure;
    /// * an omission replaces a completion and clears the failure;
    /// * a failure is kept without erasing an earlier completion, unless it
    ///   discarded that completion's output.
    ///
    /// `fingerprint` is the processing fingerprint of the run that wrote
    /// `record`, which is what a completion or omission is reused under.
    /// Returns the record's outcome so a caller can count what it folded.
    pub(super) fn apply_record(&mut self, record: &Record, fingerprint: &str) -> Outcome {
        match record.outcome {
            Outcome::Committed => {
                self.failed.remove(&record.item);
                self.skipped.remove(&record.item);
                if let (Some(version), Some(output)) = (&record.version, &record.output) {
                    self.items.insert(
                        record.item.clone(),
                        SnapshotItem {
                            version: version.clone(),
                            fingerprint: fingerprint.to_string(),
                            output: output.clone(),
                        },
                    );
                }
            }
            Outcome::Deleted => {
                self.items.remove(&record.item);
                self.skipped.remove(&record.item);
                self.failed.remove(&record.item);
            }
            Outcome::Skipped => {
                self.failed.remove(&record.item);
                self.items.remove(&record.item);
                self.skipped.insert(
                    record.item.clone(),
                    SkippedItem {
                        version: record.version.clone().unwrap_or_default(),
                        fingerprint: fingerprint.to_string(),
                        reason: record.reason.clone().unwrap_or_default(),
                    },
                );
            }
            Outcome::Failed => {
                if record.discard_output {
                    self.items.remove(&record.item);
                    self.skipped.remove(&record.item);
                }
                self.failed.insert(
                    record.item.clone(),
                    FailedItem {
                        version: record.version.clone(),
                        error: record.error.clone().unwrap_or_default(),
                        fingerprint: Some(fingerprint.to_string()),
                        deferred: record.deferred,
                        retry_fingerprint: record.retry_fingerprint.clone(),
                    },
                );
            }
        }
        record.outcome
    }

    /// Identity of the committed state, for a run header's `baseline`. It
    /// covers what was committed, omitted and failed, not when the file was
    /// written.
    ///
    /// A snapshot written before deliberate omissions and deferred failures
    /// existed hashes exactly as it did then (their maps and fields are
    /// absent), so a run interrupted by an older build still resumes.
    pub fn hash(&self) -> String {
        let mut hasher = Sha256::new();
        for (key, item) in &self.items {
            for part in [
                "item",
                key,
                &item.version,
                &item.fingerprint,
                &item.output.path,
                &item.output.size.to_string(),
                &item.output.digest,
                &item.output.mtime_ns.to_string(),
            ] {
                hasher.update(part.as_bytes());
                hasher.update([0]);
            }
            hasher.update(b"\n");
        }
        for (key, skipped) in &self.skipped {
            for part in [
                "skipped",
                key,
                &skipped.version,
                &skipped.fingerprint,
                &skipped.reason,
            ] {
                hasher.update(part.as_bytes());
                hasher.update([0]);
            }
            hasher.update(b"\n");
        }
        for (key, failure) in &self.failed {
            let version = failure
                .version
                .as_deref()
                .map_or_else(|| "-".to_string(), |version| format!("+{version}"));
            for part in ["failed", key, &version, &failure.error] {
                hasher.update(part.as_bytes());
                hasher.update([0]);
            }
            if failure.deferred || failure.retry_fingerprint.is_some() {
                for part in [
                    "deferred",
                    if failure.deferred { "1" } else { "0" },
                    failure.retry_fingerprint.as_deref().unwrap_or("-"),
                    failure.fingerprint.as_deref().unwrap_or("-"),
                ] {
                    hasher.update(part.as_bytes());
                    hasher.update([0]);
                }
            }
            hasher.update(b"\n");
        }
        format!("sha256:{}", hex::encode(hasher.finalize()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::incremental::journal::OutputRef;

    fn output(path: &str) -> OutputRef {
        OutputRef {
            path: path.into(),
            size: 4,
            digest: "sha256:d".into(),
            mtime_ns: 9,
        }
    }

    fn item(version: &str) -> SnapshotItem {
        SnapshotItem {
            version: version.into(),
            fingerprint: "fp".into(),
            output: output("a.md"),
        }
    }

    #[test]
    fn save_and_load_roundtrip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join(SNAPSHOT_FILE);
        let mut snapshot = Snapshot::new("parse", "corpus");
        snapshot.items.insert("a.pdf".into(), item("v1"));
        snapshot.failed.insert(
            "b.pdf".into(),
            FailedItem {
                version: Some("v2".into()),
                error: "bad xref".into(),
                fingerprint: Some("fp".into()),
                deferred: false,
                retry_fingerprint: None,
            },
        );
        snapshot.skipped.insert(
            "c.pdf".into(),
            SkippedItem {
                version: "v1".into(),
                fingerprint: "fp".into(),
                reason: "unsupported type".into(),
            },
        );
        snapshot.save(&path).expect("save");

        let loaded = Snapshot::load(&path).expect("load").expect("present");
        assert_eq!(loaded, snapshot);
    }

    #[test]
    fn a_missing_snapshot_is_none_and_a_corrupt_one_is_an_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join(SNAPSHOT_FILE);
        assert!(Snapshot::load(&path).expect("absent is fine").is_none());

        std::fs::write(&path, "{ not json").expect("write");
        let error = Snapshot::load(&path).expect_err("corrupt");
        assert!(
            matches!(
                error,
                MinSyncError::Stage {
                    code: "corrupt_state",
                    ..
                }
            ),
            "{error:?}"
        );
    }

    #[test]
    fn a_newer_snapshot_format_is_refused() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join(SNAPSHOT_FILE);
        let mut snapshot = Snapshot::new("parse", "corpus");
        snapshot.format = SNAPSHOT_FORMAT + 1;
        snapshot.save(&path).expect("save");
        let error = Snapshot::load(&path).expect_err("newer");
        assert!(
            matches!(
                error,
                MinSyncError::Stage {
                    code: "corrupt_state",
                    ..
                }
            ),
            "{error:?}"
        );
    }

    #[test]
    fn hash_tracks_content_but_not_the_write_time() {
        let mut a = Snapshot::new("parse", "corpus");
        a.items.insert("a.pdf".into(), item("v1"));
        let mut b = a.clone();
        b.updated_at = "a different time".into();
        assert_eq!(a.hash(), b.hash());

        b.items.insert("a.pdf".into(), item("v2"));
        assert_ne!(a.hash(), b.hash());

        let mut c = a.clone();
        c.failed.insert(
            "x".into(),
            FailedItem {
                version: None,
                error: "e".into(),
                fingerprint: None,
                deferred: false,
                retry_fingerprint: None,
            },
        );
        assert_ne!(a.hash(), c.hash());
        assert!(a.hash().starts_with("sha256:"));
    }

    /// A snapshot written before omissions and deferral existed hashed only
    /// its items and plain failures; the extended hash must not move.
    #[test]
    fn hash_of_a_legacy_shaped_snapshot_is_unchanged() {
        let mut snapshot = Snapshot::new("parse", "corpus");
        snapshot.items.insert("a.pdf".into(), item("v1"));
        snapshot.format = 1;
        snapshot.failed.insert(
            "b.pdf".into(),
            FailedItem {
                version: Some("v2".into()),
                error: "bad xref".into(),
                fingerprint: None,
                deferred: false,
                retry_fingerprint: None,
            },
        );
        assert_eq!(snapshot.hash(), legacy_hash(&snapshot));
    }

    /// The pre-extension hash, kept here as the oracle for the test above.
    fn legacy_hash(snapshot: &Snapshot) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        for (key, item) in &snapshot.items {
            for part in [
                "item",
                key,
                &item.version,
                &item.fingerprint,
                &item.output.path,
                &item.output.size.to_string(),
                &item.output.digest,
                &item.output.mtime_ns.to_string(),
            ] {
                hasher.update(part.as_bytes());
                hasher.update([0]);
            }
            hasher.update(b"\n");
        }
        for (key, failure) in &snapshot.failed {
            let version = failure
                .version
                .as_deref()
                .map_or_else(|| "-".to_string(), |version| format!("+{version}"));
            for part in ["failed", key, &version, &failure.error] {
                hasher.update(part.as_bytes());
                hasher.update([0]);
            }
            hasher.update(b"\n");
        }
        format!("sha256:{}", hex::encode(hasher.finalize()))
    }
}
