//! The committed state of a stage between runs.
//!
//! The journal records what a run committed *while it runs*; at the end of a
//! successful run it is compacted into this snapshot (one atomic write) and
//! deleted. The snapshot is therefore never rewritten per item, only once per
//! run, and a crash at any point leaves either the old snapshot plus a
//! resumable journal, or the new snapshot.

use super::fsio::atomic_write;
use super::journal::OutputRef;
use crate::error::{MinSyncError, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;

pub const SNAPSHOT_FILE: &str = "snapshot.json";
pub const SNAPSHOT_FORMAT: u32 = 1;

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
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Snapshot {
    pub format: u32,
    pub source_id: String,
    pub stage: String,
    pub updated_at: String,
    pub items: BTreeMap<String, SnapshotItem>,
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

    /// Identity of the committed state, for a run header's `baseline`. It
    /// covers what was committed and failed, not when the file was written.
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
        for (key, failure) in &self.failed {
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
            },
        );
        assert_ne!(a.hash(), c.hash());
        assert!(a.hash().starts_with("sha256:"));
    }
}
