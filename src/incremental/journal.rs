//! The durable run ledger: an append-only log of per-item outcomes.
//!
//! `cursor.json`/`manifest.json` (or a stage snapshot) only move once a whole
//! run finishes, so a run killed after hours of work used to restart from
//! zero. The journal sits next to them and records progress as it happens:
//!
//! - line 1: a [`RunHeader`] identifying the run (corpus, stage, mode,
//!   baseline, processing fingerprint, token, creation time, format);
//! - every following line: one [`Record`] — an item whose output commit
//!   already returned, with the source version that was processed.
//!
//! A record is appended only **after** the stage's output commit succeeded,
//! so a recorded item is always safe to skip. The loader is deliberately
//! suspicious: a torn final line (kill mid-append) is ignored, any other
//! malformed line discards the whole ledger, and an absent or invalid ledger
//! just means the next run redoes the work. It never means "work completed".
//!
//! Logs written before this module existed (`path` / `content_hash` records,
//! `sync_token` / `manifest_hash_from` / `config_fingerprint` header) are read
//! through serde aliases, so an upgrade resumes an interrupted run.

use super::fsio::atomic_write;
use crate::error::{MinSyncError, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;

pub const JOURNAL_FILE: &str = "progress.jsonl";

/// Ledger format written by this build. Logs from before the field existed
/// are format 1; a log from a newer build is not trusted.
pub const FORMAT_VERSION: u32 = 2;

fn legacy_format() -> u32 {
    1
}

/// Only the vector indexer wrote ledgers before stages had names.
fn legacy_stage() -> String {
    "vector".to_string()
}

/// First line of a ledger: which run it belongs to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunHeader {
    #[serde(default = "legacy_format")]
    pub format: u32,
    /// Corpus identity.
    pub source_id: String,
    /// Stage identity (`vector`, a materialization stage's name, ...).
    #[serde(default = "legacy_stage")]
    pub stage: String,
    #[serde(alias = "sync_token")]
    pub run_token: String,
    /// Full run (every item reprocessed) rather than incremental.
    pub full: bool,
    /// Hash of the committed snapshot/manifest the run diffed against.
    #[serde(alias = "manifest_hash_from")]
    pub baseline: String,
    /// Hash of every setting that changes what the stage writes for an item.
    #[serde(alias = "config_fingerprint")]
    pub fingerprint: String,
    #[serde(default)]
    pub created_at: String,
}

impl RunHeader {
    /// Header of a fresh run: new token, current time, current format.
    pub fn new(
        stage: &str,
        source_id: &str,
        full: bool,
        baseline: &str,
        fingerprint: &str,
    ) -> Self {
        Self {
            format: FORMAT_VERSION,
            source_id: source_id.to_string(),
            stage: stage.to_string(),
            run_token: new_run_token(),
            full,
            baseline: baseline.to_string(),
            fingerprint: fingerprint.to_string(),
            created_at: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Whether a ledger with this header may be resumed by a run described
    /// by `current`. The token, creation time and format say *which* run
    /// wrote it, not *what* it computed, so they are not compared.
    pub fn resumes(&self, current: &RunHeader) -> bool {
        self.source_id == current.source_id
            && self.stage == current.stage
            && self.full == current.full
            && self.baseline == current.baseline
            && self.fingerprint == current.fingerprint
    }
}

pub fn new_run_token() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    /// The stage's output for this source version is durable.
    Committed,
    /// The item's output was removed because its source is confirmed gone.
    Deleted,
    /// Processing failed; nothing was committed. Never counts as done.
    Failed,
}

/// Identity of a committed output file, so a completion can later be
/// checked against what is actually on disk.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputRef {
    /// Path relative to the stage's output root.
    pub path: String,
    pub size: u64,
    /// `sha256:<hex>` of the file content.
    pub digest: String,
    /// Modification time at commit; `0` = unknown (always re-hash).
    #[serde(default)]
    pub mtime_ns: u128,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Record {
    /// Item key: a normalized relative path or a host-provided document id.
    pub item: String,
    /// Source version the work was done for (`None` for a deletion).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    pub outcome: Outcome,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<OutputRef>,
    /// Failure diagnostic.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl Record {
    pub fn committed(
        item: impl Into<String>,
        version: impl Into<String>,
        output: Option<OutputRef>,
    ) -> Self {
        Self {
            item: item.into(),
            version: Some(version.into()),
            outcome: Outcome::Committed,
            output,
            error: None,
        }
    }

    pub fn deleted(item: impl Into<String>) -> Self {
        Self {
            item: item.into(),
            version: None,
            outcome: Outcome::Deleted,
            output: None,
            error: None,
        }
    }

    pub fn failed(
        item: impl Into<String>,
        version: Option<&str>,
        error: impl Into<String>,
    ) -> Self {
        Self {
            item: item.into(),
            version: version.map(str::to_string),
            outcome: Outcome::Failed,
            output: None,
            error: Some(error.into()),
        }
    }
}

/// On-disk record, including the pre-core shape (`path`, `content_hash`, no
/// `outcome`: a hash means committed, `null` means deleted).
#[derive(Deserialize)]
struct RawRecord {
    #[serde(alias = "path")]
    item: String,
    #[serde(default, alias = "content_hash")]
    version: Option<String>,
    #[serde(default)]
    outcome: Option<Outcome>,
    #[serde(default)]
    output: Option<OutputRef>,
    #[serde(default)]
    error: Option<String>,
}

impl Record {
    fn parse(line: &str) -> std::result::Result<Self, String> {
        let raw: RawRecord = serde_json::from_str(line).map_err(|error| error.to_string())?;
        let outcome = raw.outcome.unwrap_or(if raw.version.is_some() {
            Outcome::Committed
        } else {
            Outcome::Deleted
        });
        if outcome == Outcome::Committed && raw.version.is_none() {
            return Err(format!(
                "committed record for {:?} has no version",
                raw.item
            ));
        }
        Ok(Self {
            item: raw.item,
            version: raw.version,
            outcome,
            output: raw.output,
            error: raw.error,
        })
    }
}

/// Whether the ledger holds a completion for an item at a given version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// Recorded as committed (or deleted) with exactly this version.
    Committed,
    /// Recorded for a different version: the item changed since, and the
    /// earlier output may still be in the store/output root.
    Superseded,
    Unseen,
}

/// A parsed ledger: the latest completion and the latest failure per item.
///
/// Completions and failures are tracked separately so a failure never erases
/// the record of an earlier commit (that output is still there and still
/// has to be reconciled), and a later commit clears the failure.
#[derive(Debug, Clone)]
pub struct Ledger {
    pub header: RunHeader,
    /// The file ended in a record that never got its newline.
    pub torn_tail: bool,
    done: BTreeMap<String, Record>,
    failed: BTreeMap<String, Record>,
}

impl Ledger {
    pub fn new(header: RunHeader) -> Self {
        Self {
            header,
            torn_tail: false,
            done: BTreeMap::new(),
            failed: BTreeMap::new(),
        }
    }

    pub fn apply(&mut self, record: Record) {
        match record.outcome {
            Outcome::Committed | Outcome::Deleted => {
                self.failed.remove(&record.item);
                self.done.insert(record.item.clone(), record);
            }
            Outcome::Failed => {
                self.failed.insert(record.item.clone(), record);
            }
        }
    }

    /// Latest committed/deleted record of `item`.
    pub fn done(&self, item: &str) -> Option<&Record> {
        self.done.get(item)
    }

    /// Latest failure of `item` that no later commit/delete cleared.
    pub fn failed(&self, item: &str) -> Option<&Record> {
        self.failed.get(item)
    }

    /// Whether `item` was already written with exactly `version`
    /// (`None` = deletion).
    pub fn status(&self, item: &str, version: Option<&str>) -> Status {
        match self.done.get(item) {
            None => Status::Unseen,
            Some(record) => {
                let same = match version {
                    Some(version) => {
                        record.outcome == Outcome::Committed
                            && record.version.as_deref() == Some(version)
                    }
                    None => record.outcome == Outcome::Deleted,
                };
                if same {
                    Status::Committed
                } else {
                    Status::Superseded
                }
            }
        }
    }

    /// Every held record, completions first, each group in item order.
    pub fn records(&self) -> impl Iterator<Item = &Record> {
        self.done.values().chain(self.failed.values())
    }

    pub fn is_empty(&self) -> bool {
        self.done.is_empty() && self.failed.is_empty()
    }
}

/// Result of reading a ledger file.
#[derive(Debug)]
pub enum Load {
    Absent,
    /// Present but untrustworthy; the reason is for diagnostics. Callers
    /// redo the work.
    Invalid(String),
    Loaded(Ledger),
}

impl Load {
    /// The ledger, logging why an unusable one is being discarded.
    pub fn into_ledger(self, path: &Path) -> Option<Ledger> {
        match self {
            Load::Loaded(ledger) => Some(ledger),
            Load::Absent => None,
            Load::Invalid(reason) => {
                tracing::warn!(
                    path = %path.display(),
                    %reason,
                    "discarding an unusable progress log; its work will be redone"
                );
                None
            }
        }
    }
}

/// Append handle on a ledger file.
pub struct Journal {
    file: File,
}

impl Journal {
    /// Atomically (re)write the ledger as header + `carried`, then keep it
    /// open for appending. Rewriting (instead of appending to what is on
    /// disk) also drops a torn tail, so a new record can never be glued onto
    /// half a line.
    pub fn create(path: &Path, header: &RunHeader, carried: Option<&Ledger>) -> Result<Self> {
        let mut content = serde_json::to_string(header)?;
        content.push('\n');
        for record in carried.into_iter().flat_map(Ledger::records) {
            content.push_str(&serde_json::to_string(record)?);
            content.push('\n');
        }
        atomic_write(path, content.as_bytes())?;
        let file = OpenOptions::new().append(true).open(path)?;
        Ok(Self { file })
    }

    /// Durably record `records`, one fsync for the whole slice. Call this
    /// only after the corresponding output commits have returned.
    pub fn append(&mut self, records: &[Record]) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        #[cfg(test)]
        if append_fault_fires() {
            return Err(MinSyncError::Other(
                "injected crash before journal append".to_string(),
            ));
        }
        let mut buffer = String::new();
        for record in records {
            buffer.push_str(&serde_json::to_string(record)?);
            buffer.push('\n');
        }
        self.file.write_all(buffer.as_bytes())?;
        self.file.sync_data()?;
        Ok(())
    }

    pub fn load(path: &Path) -> Load {
        let bytes = match std::fs::read(path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Load::Absent,
            Err(error) => return Load::Invalid(format!("unreadable: {error}")),
        };
        let Ok(content) = String::from_utf8(bytes) else {
            return Load::Invalid("not valid UTF-8".to_string());
        };
        let mut lines = content.split_inclusive('\n');
        let Some(first) = lines.next() else {
            return Load::Invalid("empty".to_string());
        };
        if !first.ends_with('\n') {
            return Load::Invalid("torn header".to_string());
        }
        let header: RunHeader = match serde_json::from_str(first.trim_end()) {
            Ok(header) => header,
            Err(error) => return Load::Invalid(format!("bad header: {error}")),
        };
        if header.format < legacy_format() || header.format > FORMAT_VERSION {
            return Load::Invalid(format!(
                "format {} is outside the supported range {}..={FORMAT_VERSION}",
                header.format,
                legacy_format()
            ));
        }
        let mut ledger = Ledger::new(header);
        for (index, line) in lines.enumerate() {
            if !line.ends_with('\n') {
                ledger.torn_tail = true; // torn final append
                break;
            }
            match Record::parse(line.trim_end()) {
                Ok(record) => ledger.apply(record),
                Err(error) => {
                    return Load::Invalid(format!("corrupt record on line {}: {error}", index + 2))
                }
            }
        }
        Load::Loaded(ledger)
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
thread_local! {
    static APPEND_FAULT: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// Test-only crash injection: after `successful` more non-empty appends on
/// this thread, the next append fails before writing anything, as if the
/// process died after the output commit and before its record.
#[cfg(test)]
pub(crate) fn fail_append_after(successful: usize) {
    APPEND_FAULT.with(|fault| fault.set(Some(successful)));
}

#[cfg(test)]
fn append_fault_fires() -> bool {
    APPEND_FAULT.with(|fault| match fault.get() {
        Some(0) => {
            fault.set(None);
            true
        }
        Some(remaining) => {
            fault.set(Some(remaining - 1));
            false
        }
        None => false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn header() -> RunHeader {
        RunHeader {
            format: FORMAT_VERSION,
            source_id: "s".into(),
            stage: "vector".into(),
            run_token: "tok".into(),
            full: false,
            baseline: "sha256:from".into(),
            fingerprint: "fp".into(),
            created_at: "2026-10-24T00:00:00Z".into(),
        }
    }

    fn output(path: &str) -> OutputRef {
        OutputRef {
            path: path.into(),
            size: 3,
            digest: "sha256:abc".into(),
            mtime_ns: 7,
        }
    }

    fn load_ledger(path: &std::path::Path) -> Ledger {
        match Journal::load(path) {
            Load::Loaded(ledger) => ledger,
            other => panic!("expected a loaded ledger, got {other:?}"),
        }
    }

    fn log_path(dir: &tempfile::TempDir) -> std::path::PathBuf {
        dir.path().join(JOURNAL_FILE)
    }

    #[test]
    fn roundtrip_and_status_requires_matching_version() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = log_path(&dir);
        let mut journal = Journal::create(&path, &header(), None).expect("create");
        journal
            .append(&[
                Record::committed("a.txt", "h1", Some(output("out/a.md"))),
                Record::deleted("gone.txt"),
            ])
            .expect("append");

        let ledger = load_ledger(&path);
        assert_eq!(ledger.header, header());
        assert!(!ledger.torn_tail);
        assert_eq!(ledger.status("a.txt", Some("h1")), Status::Committed);
        assert_eq!(
            ledger.status("a.txt", Some("h2")),
            Status::Superseded,
            "changed since"
        );
        assert_eq!(ledger.status("gone.txt", None), Status::Committed);
        assert_eq!(ledger.status("gone.txt", Some("h1")), Status::Superseded);
        assert_eq!(ledger.status("other.txt", Some("h1")), Status::Unseen);
        assert_eq!(
            ledger.done("a.txt").and_then(|r| r.output.clone()),
            Some(output("out/a.md"))
        );
    }

    #[test]
    fn logs_written_before_the_core_existed_still_resume() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = log_path(&dir);
        std::fs::write(
            &path,
            concat!(
                r#"{"source_id":"s","sync_token":"tok","full":false,"manifest_hash_from":"sha256:from","config_fingerprint":"fp"}"#,
                "\n",
                r#"{"path":"a.txt","content_hash":"h1"}"#,
                "\n",
                r#"{"path":"gone.txt","content_hash":null}"#,
                "\n",
            ),
        )
        .expect("write legacy log");

        let ledger = load_ledger(&path);
        assert_eq!(ledger.header.run_token, "tok");
        assert_eq!(ledger.header.baseline, "sha256:from");
        assert_eq!(ledger.header.fingerprint, "fp");
        assert_eq!(ledger.header.stage, "vector");
        assert_eq!(ledger.status("a.txt", Some("h1")), Status::Committed);
        assert_eq!(ledger.status("gone.txt", None), Status::Committed);
        let expected = RunHeader {
            format: ledger.header.format,
            created_at: ledger.header.created_at.clone(),
            ..header()
        };
        assert!(ledger.header.resumes(&expected));
    }

    #[test]
    fn torn_final_record_is_ignored_but_earlier_records_survive() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = log_path(&dir);
        let mut journal = Journal::create(&path, &header(), None).expect("create");
        journal
            .append(&[Record::committed("a.txt", "h1", None)])
            .expect("append");
        let mut raw = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .expect("open");
        raw.write_all(br#"{"item":"b.txt","version":"h2","outcome":"comm"#)
            .expect("torn");

        let ledger = load_ledger(&path);
        assert!(ledger.torn_tail);
        assert_eq!(ledger.status("a.txt", Some("h1")), Status::Committed);
        assert_eq!(ledger.status("b.txt", Some("h2")), Status::Unseen);
    }

    #[test]
    fn a_complete_looking_torn_record_without_newline_is_not_trusted() {
        // The last line parses but never got its newline: the append that
        // wrote it may not have finished, so it must not count as done.
        let dir = tempfile::tempdir().expect("tempdir");
        let path = log_path(&dir);
        let header_line = serde_json::to_string(&header()).expect("header");
        std::fs::write(
            &path,
            format!(
                "{header_line}\n{{\"item\":\"a.txt\",\"version\":\"h1\",\"outcome\":\"committed\"}}"
            ),
        )
        .expect("write");

        let ledger = load_ledger(&path);
        assert!(ledger.torn_tail);
        assert_eq!(ledger.status("a.txt", Some("h1")), Status::Unseen);
    }

    #[test]
    fn corrupt_middle_record_discards_the_ledger() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = log_path(&dir);
        let header_line = serde_json::to_string(&header()).expect("header");
        std::fs::write(
            &path,
            format!(
                "{header_line}\n{{\"item\":\"a.txt\",\"version\":\"h1\",\"outcome\":\"committed\"}}\nnot json\n{{\"item\":\"b.txt\",\"version\":\"h2\",\"outcome\":\"committed\"}}\n"
            ),
        )
        .expect("write");
        assert!(matches!(Journal::load(&path), Load::Invalid(_)));
    }

    #[test]
    fn absent_empty_and_header_less_logs_mean_redo() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = log_path(&dir);
        assert!(matches!(Journal::load(&path), Load::Absent));

        std::fs::write(&path, "").expect("write empty");
        assert!(matches!(Journal::load(&path), Load::Invalid(_)));

        std::fs::write(&path, "{\"item\":\"a\"}\n").expect("write header-less");
        assert!(matches!(Journal::load(&path), Load::Invalid(_)));

        std::fs::write(&path, r#"{"source_id":"s","sta"#).expect("write torn header");
        assert!(matches!(Journal::load(&path), Load::Invalid(_)));
    }

    #[test]
    fn an_unsupported_older_ledger_format_is_not_trusted() {
        // Logs from before the field existed are format 1 (the serde
        // default); an explicit 0 never existed and must not be trusted.
        let dir = tempfile::tempdir().expect("tempdir");
        let path = log_path(&dir);
        let mut zero = header();
        zero.format = 0;
        std::fs::write(
            &path,
            format!("{}\n", serde_json::to_string(&zero).expect("header")),
        )
        .expect("write");
        assert!(matches!(Journal::load(&path), Load::Invalid(_)));
    }

    #[test]
    fn a_newer_ledger_format_is_not_trusted() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = log_path(&dir);
        let mut newer = header();
        newer.format = FORMAT_VERSION + 1;
        std::fs::write(
            &path,
            format!("{}\n", serde_json::to_string(&newer).expect("header")),
        )
        .expect("write");
        assert!(matches!(Journal::load(&path), Load::Invalid(_)));
    }

    #[test]
    fn a_commit_without_a_source_version_is_corrupt() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = log_path(&dir);
        let header_line = serde_json::to_string(&header()).expect("header");
        std::fs::write(
            &path,
            format!("{header_line}\n{{\"item\":\"a.txt\",\"outcome\":\"committed\"}}\n"),
        )
        .expect("write");
        assert!(matches!(Journal::load(&path), Load::Invalid(_)));
    }

    #[test]
    fn a_failure_never_marks_an_item_done_and_keeps_the_earlier_commit() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = log_path(&dir);
        let mut journal = Journal::create(&path, &header(), None).expect("create");
        journal
            .append(&[
                Record::committed("a.txt", "h1", Some(output("out/a.md"))),
                Record::failed("a.txt", Some("h2"), "parser crashed"),
                Record::failed("b.txt", Some("h9"), "unreadable"),
            ])
            .expect("append");

        let ledger = load_ledger(&path);
        assert_eq!(
            ledger.status("a.txt", Some("h1")),
            Status::Committed,
            "the previous valid output stays recorded"
        );
        assert_eq!(ledger.status("a.txt", Some("h2")), Status::Superseded);
        assert_eq!(ledger.status("b.txt", Some("h9")), Status::Unseen);
        assert_eq!(
            ledger.failed("a.txt").and_then(|r| r.error.as_deref()),
            Some("parser crashed")
        );
        assert!(ledger.failed("b.txt").is_some());
    }

    #[test]
    fn a_later_commit_or_delete_clears_the_recorded_failure() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = log_path(&dir);
        let mut journal = Journal::create(&path, &header(), None).expect("create");
        journal
            .append(&[
                Record::failed("a.txt", Some("h1"), "boom"),
                Record::failed("b.txt", Some("h1"), "boom"),
                Record::committed("a.txt", "h1", None),
                Record::deleted("b.txt"),
            ])
            .expect("append");

        let ledger = load_ledger(&path);
        assert!(ledger.failed("a.txt").is_none());
        assert!(ledger.failed("b.txt").is_none());
        assert_eq!(ledger.status("a.txt", Some("h1")), Status::Committed);
    }

    #[test]
    fn create_rewrites_carried_records_atomically_in_item_order() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = log_path(&dir);
        let mut journal = Journal::create(&path, &header(), None).expect("create");
        journal
            .append(&[
                Record::committed("z.txt", "h1", None),
                Record::committed("a.txt", "h2", None),
                Record::failed("m.txt", Some("h3"), "boom"),
            ])
            .expect("append");
        drop(journal);
        let carried = load_ledger(&path);

        let mut journal = Journal::create(&path, &header(), Some(&carried)).expect("recreate");
        journal
            .append(&[Record::committed("b.txt", "h4", None)])
            .expect("append");

        let text = std::fs::read_to_string(&path).expect("read");
        let items: Vec<String> = text
            .lines()
            .skip(1)
            .map(|line| {
                serde_json::from_str::<serde_json::Value>(line).expect("json")["item"]
                    .as_str()
                    .expect("item")
                    .to_string()
            })
            .collect();
        assert_eq!(items, ["a.txt", "z.txt", "m.txt", "b.txt"]);
        let ledger = load_ledger(&path);
        assert_eq!(ledger.status("a.txt", Some("h2")), Status::Committed);
        assert_eq!(ledger.status("b.txt", Some("h4")), Status::Committed);
        assert!(ledger.failed("m.txt").is_some());
    }

    #[test]
    fn create_drops_a_torn_tail_instead_of_appending_to_it() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = log_path(&dir);
        let mut journal = Journal::create(&path, &header(), None).expect("create");
        journal
            .append(&[Record::committed("a.txt", "h1", None)])
            .expect("append");
        drop(journal);
        let mut raw = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .expect("open");
        raw.write_all(b"{\"item\":\"b.txt\",\"vers").expect("torn");
        let carried = load_ledger(&path);

        let mut journal = Journal::create(&path, &header(), Some(&carried)).expect("recreate");
        journal
            .append(&[Record::committed("c.txt", "h3", None)])
            .expect("append");

        let ledger = load_ledger(&path);
        assert!(
            !ledger.torn_tail,
            "{}",
            std::fs::read_to_string(&path).unwrap()
        );
        assert_eq!(ledger.status("a.txt", Some("h1")), Status::Committed);
        assert_eq!(ledger.status("c.txt", Some("h3")), Status::Committed);
    }

    #[test]
    fn appending_costs_only_the_new_records() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = log_path(&dir);
        let mut journal = Journal::create(&path, &header(), None).expect("create");
        for index in 0..500 {
            journal
                .append(&[Record::committed(format!("f{index}"), "h", None)])
                .expect("append");
        }
        let before = std::fs::metadata(&path).expect("stat").len();
        let record = Record::committed("one-more", "h", None);
        let expected = serde_json::to_string(&record).expect("json").len() as u64 + 1;
        journal.append(&[record]).expect("append");
        let after = std::fs::metadata(&path).expect("stat").len();
        assert_eq!(after - before, expected, "append must not rewrite history");
    }

    #[test]
    fn resumes_only_for_the_same_run_identity() {
        let base = header();
        let mut other_token = base.clone();
        other_token.run_token = "different".into();
        other_token.created_at = "later".into();
        assert!(
            base.resumes(&other_token),
            "token and time are not identity"
        );

        for mutate in [
            |h: &mut RunHeader| h.source_id = "other".into(),
            |h: &mut RunHeader| h.stage = "parse".into(),
            |h: &mut RunHeader| h.full = true,
            |h: &mut RunHeader| h.baseline = "sha256:other".into(),
            |h: &mut RunHeader| h.fingerprint = "other".into(),
        ] {
            let mut changed = base.clone();
            mutate(&mut changed);
            assert!(!base.resumes(&changed), "{changed:?}");
        }
    }

    #[test]
    fn remove_is_idempotent() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = log_path(&dir);
        Journal::create(&path, &header(), None).expect("create");
        Journal::remove(&path).expect("remove");
        Journal::remove(&path).expect("remove again");
        assert!(!path.exists());
    }
}
