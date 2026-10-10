//! Crash-resume bookkeeping for a running sync.
//!
//! `cursor.json`/`manifest.json` only move once a whole sync finishes, so a
//! sync killed after hours of work used to restart from zero. This module
//! keeps an append-only log, `progress.jsonl`, next to them:
//!
//! - line 1: a [`ProgressHeader`] identifying the run (token, baseline,
//!   config fingerprint);
//! - every following line: one [`ProgressEntry`] — a path whose store writes
//!   are already committed, with the content hash that was indexed (`null`
//!   when the file was deleted).
//!
//! Entries are appended only after the store commit returned, so a logged
//! path is always safe to skip. A torn final line (kill mid-append) is
//! ignored on load; any other malformed line discards the log, which just
//! means the next sync redoes the work (never a correctness problem).

use crate::error::{MinSyncError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;

pub(super) const PROGRESS_FILE: &str = "progress.jsonl";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct ProgressHeader {
    pub source_id: String,
    pub sync_token: String,
    pub full: bool,
    /// Hash of the baseline manifest the interrupted run diffed against.
    pub manifest_hash_from: String,
    /// Hash of every setting that changes what gets written for a file.
    pub config_fingerprint: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct ProgressEntry {
    path: String,
    content_hash: Option<String>,
}

/// A parsed log: which paths were already committed, and with what content.
#[derive(Debug)]
pub(super) struct ResumeState {
    pub header: ProgressHeader,
    pub processed: HashMap<String, Option<String>>,
}

impl ResumeState {
    /// Whether `path` was already written with exactly this content
    /// (`None` = deletion).
    pub fn is_done(&self, path: &str, content_hash: Option<&str>) -> bool {
        self.processed
            .get(path)
            .is_some_and(|done| done.as_deref() == content_hash)
    }
}

pub(super) struct ProgressLog {
    file: File,
}

impl ProgressLog {
    /// Atomically (re)write the log as header + `entries`, then keep it open
    /// for appending.
    pub fn create(
        path: &Path,
        header: &ProgressHeader,
        entries: &HashMap<String, Option<String>>,
    ) -> Result<Self> {
        let mut content = serde_json::to_string(header)?;
        content.push('\n');
        for (entry_path, content_hash) in entries {
            content.push_str(&serde_json::to_string(&ProgressEntry {
                path: entry_path.clone(),
                content_hash: content_hash.clone(),
            })?);
            content.push('\n');
        }
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        let mut tmp = tempfile::NamedTempFile::new_in(parent)?;
        tmp.write_all(content.as_bytes())?;
        tmp.as_file().sync_all()?;
        tmp.persist(path)
            .map_err(|error| MinSyncError::Io(error.error))?;
        let file = OpenOptions::new().append(true).open(path)?;
        Ok(Self { file })
    }

    /// Durably record `entries` as committed.
    pub fn append(&mut self, entries: &[(String, Option<String>)]) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }
        let mut buffer = String::new();
        for (path, content_hash) in entries {
            buffer.push_str(&serde_json::to_string(&ProgressEntry {
                path: path.clone(),
                content_hash: content_hash.clone(),
            })?);
            buffer.push('\n');
        }
        self.file.write_all(buffer.as_bytes())?;
        self.file.sync_data()?;
        Ok(())
    }

    pub fn load(path: &Path) -> Option<ResumeState> {
        let content = std::fs::read_to_string(path).ok()?;
        let mut lines = content.split_inclusive('\n');
        let header: ProgressHeader = serde_json::from_str(lines.next()?.trim_end()).ok()?;
        let mut processed = HashMap::new();
        for line in lines {
            if !line.ends_with('\n') {
                break; // torn final append
            }
            let entry: ProgressEntry = serde_json::from_str(line.trim_end()).ok()?;
            processed.insert(entry.path, entry.content_hash);
        }
        Some(ResumeState { header, processed })
    }

    pub fn remove(path: &Path) -> Result<()> {
        match std::fs::remove_file(path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(MinSyncError::Io(error)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header() -> ProgressHeader {
        ProgressHeader {
            source_id: "s".into(),
            sync_token: "tok".into(),
            full: false,
            manifest_hash_from: "sha256:from".into(),
            config_fingerprint: "fp".into(),
        }
    }

    #[test]
    fn test_roundtrip_and_is_done_requires_matching_hash() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join(PROGRESS_FILE);
        let mut log = ProgressLog::create(&path, &header(), &HashMap::new()).expect("create");
        log.append(&[
            ("a.txt".into(), Some("h1".into())),
            ("gone.txt".into(), None),
        ])
        .expect("append");

        let state = ProgressLog::load(&path).expect("load");
        assert_eq!(state.header, header());
        assert!(state.is_done("a.txt", Some("h1")));
        assert!(!state.is_done("a.txt", Some("h2")), "changed since");
        assert!(state.is_done("gone.txt", None));
        assert!(!state.is_done("gone.txt", Some("h1")));
        assert!(!state.is_done("other.txt", Some("h1")));
    }

    #[test]
    fn test_torn_final_line_is_ignored_but_earlier_entries_survive() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join(PROGRESS_FILE);
        let mut log = ProgressLog::create(&path, &header(), &HashMap::new()).expect("create");
        log.append(&[("a.txt".into(), Some("h1".into()))])
            .expect("append");
        let mut raw = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .expect("open");
        raw.write_all(b"{\"path\":\"b.txt\",\"conte").expect("torn");

        let state = ProgressLog::load(&path).expect("load tolerates torn tail");
        assert!(state.is_done("a.txt", Some("h1")));
        assert!(!state.processed.contains_key("b.txt"));
    }

    #[test]
    fn test_corrupt_middle_line_discards_log() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join(PROGRESS_FILE);
        let header_line = serde_json::to_string(&header()).expect("header");
        std::fs::write(&path, format!("{header_line}\nnot json\n{{}}\n")).expect("write");
        assert!(ProgressLog::load(&path).is_none());
    }

    #[test]
    fn test_create_with_entries_rewrites_log_atomically() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join(PROGRESS_FILE);
        let mut entries = HashMap::new();
        entries.insert("a.txt".to_string(), Some("h1".to_string()));
        let mut log = ProgressLog::create(&path, &header(), &entries).expect("create");
        log.append(&[("b.txt".into(), Some("h2".into()))])
            .expect("append");
        let state = ProgressLog::load(&path).expect("load");
        assert!(state.is_done("a.txt", Some("h1")));
        assert!(state.is_done("b.txt", Some("h2")));
    }
}
