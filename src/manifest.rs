use crate::error::{MinSyncError, Result};
use crate::file_size::size_guard_error;
use chrono::Utc;
use ignore::{DirEntry, WalkBuilder};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::UNIX_EPOCH;
use tempfile::NamedTempFile;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Manifest {
    pub version: u32,
    pub source_id: String,
    pub updated_at: String,
    pub files: HashMap<String, ManifestFileEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ManifestFileEntry {
    pub size: u64,
    pub mtime_ns: u128,
    pub content_hash: String,
    #[serde(default)]
    pub fingerprint: Option<String>,
}

/// Why a path could not be inspected during a scan.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ScanErrorKind {
    /// May succeed on a later scan (`EIO`, `EBUSY`, `ETIMEDOUT`, `ESTALE`).
    Transient,
    /// Needs user action (`EACCES`, `EPERM`, `EISDIR`, symlink loops, ...).
    Permanent,
    /// The file exceeds the configured `[sync] max_file_bytes` limit.
    SizeGuard,
}

/// A per-entry scan failure. The scan continues past it; the entry's state
/// is *unknown*, never "deleted".
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ScanError {
    /// Manifest-relative path of the file or directory that failed.
    pub path: String,
    pub kind: ScanErrorKind,
    pub error: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ScanStats {
    pub files_examined: usize,
    pub files_rehashed: usize,
    pub bytes_hashed: u64,
    pub errors: Vec<ScanError>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FileChange {
    Added(String),
    Modified(String),
    Deleted(String),
}

impl Manifest {
    pub fn new(source_id: &str) -> Self {
        Self {
            version: 1,
            source_id: source_id.to_string(),
            updated_at: Utc::now().to_rfc3339(),
            files: HashMap::new(),
        }
    }

    pub fn scan(root: &Path, source_id: &str) -> Result<Self> {
        Self::scan_with_baseline(root, source_id, None)
    }

    pub fn scan_with_baseline(
        root: &Path,
        source_id: &str,
        baseline: Option<&Manifest>,
    ) -> Result<Self> {
        Self::scan_with_baseline_stats(root, source_id, baseline).map(|(manifest, _)| manifest)
    }

    /// Scan `root`, tolerating per-entry failures.
    ///
    /// A walk, `stat`, or read error on one entry never aborts the scan: it is
    /// recorded in [`ScanStats::errors`] and the entry is left out of the
    /// returned manifest. Entries that vanished (`ENOENT`/`ENOTDIR`) are
    /// simply absent, like any deleted file. Callers that keep previously
    /// indexed state MUST pass the result through [`Manifest::carry_unknown`]
    /// so that errored entries are treated as unknown rather than deleted.
    ///
    /// An error on `root` itself is still fatal: nothing can be said about
    /// any entry, and treating it as "all files deleted" would be destructive.
    pub fn scan_with_baseline_stats(
        root: &Path,
        source_id: &str,
        baseline: Option<&Manifest>,
    ) -> Result<(Self, ScanStats)> {
        Self::scan_with_baseline_stats_and_max_file_bytes(root, source_id, baseline, 0)
    }

    /// Scan while skipping files larger than `max_file_bytes`. A value of
    /// zero keeps the unbounded behavior used by callers without sync config.
    pub fn scan_with_baseline_stats_and_max_file_bytes(
        root: &Path,
        source_id: &str,
        baseline: Option<&Manifest>,
        max_file_bytes: u64,
    ) -> Result<(Self, ScanStats)> {
        Self::scan_with_fs(root, source_id, baseline, max_file_bytes, &RealFs)
    }

    fn scan_with_fs(
        root: &Path,
        source_id: &str,
        baseline: Option<&Manifest>,
        max_file_bytes: u64,
        io: &dyn ScanFs,
    ) -> Result<(Self, ScanStats)> {
        let mut manifest = Self::new(source_id);
        let mut stats = ScanStats::default();
        let walker = WalkBuilder::new(root)
            .add_custom_ignore_filename(".minsyncignore")
            .hidden(false)
            .filter_entry(|entry| !is_internal_dir(entry))
            .build();

        for entry in walker {
            let entry = match entry {
                Ok(entry) => entry,
                Err(error) => {
                    record_walk_error(root, &error, &mut stats)?;
                    continue;
                }
            };
            let file_type = entry.file_type().ok_or_else(|| {
                MinSyncError::Manifest(format!("missing file type: {}", entry.path().display()))
            })?;

            if !file_type.is_file() {
                continue;
            }

            let path = entry.path();
            let relative_path = path
                .strip_prefix(root)
                .map_err(|error| MinSyncError::Manifest(error.to_string()))?;
            let manifest_path = relative_path.to_string_lossy().replace('\\', "/");
            stats.files_examined += 1;
            let metadata = match io.metadata(path) {
                Ok(metadata) => metadata,
                Err(error) => {
                    record_io_error(&mut stats, manifest_path, &error);
                    continue;
                }
            };
            if let Some(error) = size_guard_error(metadata.len(), max_file_bytes) {
                record_error(&mut stats, manifest_path, ScanErrorKind::SizeGuard, error);
                continue;
            }
            let mtime_ns = match metadata
                .modified()
                .map_err(|error| error.to_string())
                .and_then(|modified| {
                    modified
                        .duration_since(UNIX_EPOCH)
                        .map(|elapsed| elapsed.as_nanos())
                        .map_err(|error| error.to_string())
                }) {
                Ok(mtime_ns) => mtime_ns,
                Err(message) => {
                    record_error(&mut stats, manifest_path, ScanErrorKind::Permanent, message);
                    continue;
                }
            };
            let fingerprint = metadata_fingerprint(&metadata, mtime_ns);
            let content_hash = match baseline
                .and_then(|manifest| manifest.files.get(&manifest_path))
                .filter(|entry| fingerprint.is_some() && entry.fingerprint == fingerprint)
            {
                Some(entry) => entry.content_hash.clone(),
                None => match io.hash_file(path) {
                    Ok(hash) => {
                        stats.files_rehashed += 1;
                        stats.bytes_hashed += metadata.len();
                        hash
                    }
                    Err(error) => {
                        record_io_error(&mut stats, manifest_path, &error);
                        continue;
                    }
                },
            };

            manifest.files.insert(
                manifest_path,
                ManifestFileEntry {
                    size: metadata.len(),
                    mtime_ns,
                    content_hash,
                    fingerprint,
                },
            );
        }

        Ok((manifest, stats))
    }

    /// Scan like [`Manifest::scan_with_baseline_stats`], then carry every
    /// `carry_from` entry the scan could not inspect (see
    /// [`Manifest::carry_unknown`]). Returns the manifest, the stats (with
    /// `errors`), and the carried paths.
    ///
    /// `baseline` only speeds up hashing; `carry_from` decides what stays
    /// "known". A forced full sync passes `baseline: None` yet still carries
    /// from the stored manifest.
    pub fn scan_keeping_unknown(
        root: &Path,
        source_id: &str,
        baseline: Option<&Manifest>,
        carry_from: Option<&Manifest>,
    ) -> Result<(Self, ScanStats, Vec<String>)> {
        Self::scan_keeping_unknown_with_max_file_bytes(root, source_id, baseline, carry_from, 0)
    }

    /// Scan with a size cap, preserving previous entries that were too large
    /// to inspect so their existing index rows cannot be swept as deletions.
    pub fn scan_keeping_unknown_with_max_file_bytes(
        root: &Path,
        source_id: &str,
        baseline: Option<&Manifest>,
        carry_from: Option<&Manifest>,
        max_file_bytes: u64,
    ) -> Result<(Self, ScanStats, Vec<String>)> {
        let (mut manifest, stats) = Self::scan_with_baseline_stats_and_max_file_bytes(
            root,
            source_id,
            baseline,
            max_file_bytes,
        )?;
        let carried = match carry_from {
            Some(previous) => manifest.carry_unknown(previous, &stats.errors),
            None => Vec::new(),
        };
        Ok((manifest, stats, carried))
    }

    /// Treat every entry of `previous` at or under a path in `errors` as
    /// *unknown*: copy it into `self` so [`Manifest::diff`] reports no change
    /// and nothing downstream deletes its indexed chunks. Returns the copied
    /// paths. This is the guarantee behind "scan errors never delete existing
    /// chunks".
    pub fn carry_unknown(&mut self, previous: &Manifest, errors: &[ScanError]) -> Vec<String> {
        if errors.is_empty() {
            return Vec::new();
        }
        let unknown: HashSet<&str> = errors.iter().map(|error| error.path.as_str()).collect();
        let mut carried = Vec::new();
        for (path, entry) in &previous.files {
            if !self.files.contains_key(path) && is_under_unknown(path, &unknown) {
                self.files.insert(path.clone(), entry.clone());
                carried.push(path.clone());
            }
        }
        carried.sort();
        carried
    }

    pub fn diff(old: &Manifest, new: &Manifest) -> Vec<FileChange> {
        let mut changes = Vec::new();

        for (path, new_entry) in &new.files {
            match old.files.get(path) {
                Some(old_entry) if old_entry.content_hash != new_entry.content_hash => {
                    changes.push(FileChange::Modified(path.clone()));
                }
                None => changes.push(FileChange::Added(path.clone())),
                _ => {}
            }
        }

        for path in old.files.keys() {
            if !new.files.contains_key(path) {
                changes.push(FileChange::Deleted(path.clone()));
            }
        }

        changes.sort_by(|left, right| change_path(left).cmp(change_path(right)));
        changes
    }

    pub fn manifest_hash(&self) -> String {
        let mut paths: Vec<_> = self.files.keys().collect();
        paths.sort();

        let mut hasher = Sha256::new();
        for path in paths {
            if let Some(entry) = self.files.get(path) {
                hasher.update(path.as_bytes());
                hasher.update(b"\0");
                hasher.update(entry.content_hash.as_bytes());
            }
        }

        format!("sha256:{}", hex::encode(hasher.finalize()))
    }

    pub fn load(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let manifest = serde_json::from_str(&content)?;
        Ok(manifest)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let tmp_parent = path.parent().unwrap_or_else(|| Path::new("."));
        let mut tmp = NamedTempFile::new_in(tmp_parent)?;
        let content = serde_json::to_vec_pretty(self)?;
        tmp.write_all(&content)?;
        tmp.flush()?;
        tmp.as_file().sync_all()?;
        tmp.persist(path)
            .map_err(|error| MinSyncError::Io(error.error))?;

        Ok(())
    }
}

fn is_internal_dir(entry: &DirEntry) -> bool {
    entry
        .file_type()
        .is_some_and(|file_type| file_type.is_dir())
        && matches!(entry.file_name().to_str(), Some(".minsync" | ".git"))
}

fn change_path(change: &FileChange) -> &str {
    match change {
        FileChange::Added(path) | FileChange::Modified(path) | FileChange::Deleted(path) => path,
    }
}

#[cfg(test)]
fn prefixed_sha256(content: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content);
    format!("sha256:{}", hex::encode(hasher.finalize()))
}

trait ScanFs {
    fn metadata(&self, path: &Path) -> std::io::Result<fs::Metadata>;
    fn hash_file(&self, path: &Path) -> std::io::Result<String>;
}

struct RealFs;

impl ScanFs for RealFs {
    fn metadata(&self, path: &Path) -> std::io::Result<fs::Metadata> {
        fs::metadata(path)
    }

    fn hash_file(&self, path: &Path) -> std::io::Result<String> {
        let mut file = fs::File::open(path)?;
        let mut hasher = Sha256::new();
        std::io::copy(&mut file, &mut hasher)?;
        Ok(format!("sha256:{}", hex::encode(hasher.finalize())))
    }
}

enum IoClass {
    /// Definitively not there (`ENOENT`, `ENOTDIR`): same as a deletion.
    Gone,
    Transient,
    Permanent,
}

fn classify_io(error: &std::io::Error) -> IoClass {
    use std::io::ErrorKind;

    #[cfg(unix)]
    match error.raw_os_error() {
        Some(libc::ENOENT | libc::ENOTDIR) => return IoClass::Gone,
        Some(libc::EIO | libc::EBUSY | libc::ETIMEDOUT | libc::ESTALE) => {
            return IoClass::Transient
        }
        _ => {}
    }

    match error.kind() {
        ErrorKind::NotFound => IoClass::Gone,
        ErrorKind::TimedOut
        | ErrorKind::ResourceBusy
        | ErrorKind::StaleNetworkFileHandle
        | ErrorKind::Interrupted => IoClass::Transient,
        _ => IoClass::Permanent,
    }
}

fn record_error(stats: &mut ScanStats, path: String, kind: ScanErrorKind, error: String) {
    tracing::warn!(path = %path, ?kind, %error, "scan error: entry skipped, existing chunks kept");
    stats.errors.push(ScanError { path, kind, error });
}

fn record_io_error(stats: &mut ScanStats, path: String, error: &std::io::Error) {
    let kind = match classify_io(error) {
        IoClass::Gone => {
            tracing::debug!(path = %path, %error, "entry vanished during scan");
            return;
        }
        IoClass::Transient => ScanErrorKind::Transient,
        IoClass::Permanent => ScanErrorKind::Permanent,
    };
    record_error(stats, path, kind, error.to_string());
}

/// Record a walker error against the entry it names. Errors with no path, or
/// on `root` itself, stay fatal (see [`Manifest::scan_with_baseline_stats`]).
fn record_walk_error(root: &Path, error: &ignore::Error, stats: &mut ScanStats) -> Result<()> {
    let relative = walk_error_path(error)
        .and_then(|path| path.strip_prefix(root).ok())
        .map(|relative| relative.to_string_lossy().replace('\\', "/"))
        .filter(|relative| !relative.is_empty());
    let Some(path) = relative else {
        return Err(MinSyncError::Manifest(error.to_string()));
    };
    match error.io_error() {
        Some(io_error) => record_io_error(stats, path, io_error),
        None => record_error(stats, path, ScanErrorKind::Permanent, error.to_string()),
    }
    Ok(())
}

fn walk_error_path(error: &ignore::Error) -> Option<&Path> {
    match error {
        ignore::Error::WithPath { path, .. } => Some(path),
        ignore::Error::Loop { child, .. } => Some(child),
        ignore::Error::WithDepth { err, .. } | ignore::Error::WithLineNumber { err, .. } => {
            walk_error_path(err)
        }
        ignore::Error::Partial(errors) => errors.iter().find_map(walk_error_path),
        _ => None,
    }
}

/// True when `path` or any of its ancestor directories is in `unknown`.
fn is_under_unknown(path: &str, unknown: &HashSet<&str>) -> bool {
    let mut candidate = path;
    loop {
        if unknown.contains(candidate) {
            return true;
        }
        match candidate.rfind('/') {
            Some(index) => candidate = &candidate[..index],
            None => return false,
        }
    }
}

#[cfg(unix)]
fn metadata_fingerprint(metadata: &fs::Metadata, mtime_ns: u128) -> Option<String> {
    use std::os::unix::fs::MetadataExt;

    Some(format!(
        "{}:{}:{}:{}:{}:{}",
        metadata.len(),
        mtime_ns,
        metadata.ctime(),
        metadata.ctime_nsec(),
        metadata.dev(),
        metadata.ino(),
    ))
}

#[cfg(not(unix))]
fn metadata_fingerprint(_metadata: &fs::Metadata, _mtime_ns: u128) -> Option<String> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::file_size::size_guard_message;
    use std::fs;

    struct NoHashFs;

    impl ScanFs for NoHashFs {
        fn metadata(&self, path: &Path) -> std::io::Result<fs::Metadata> {
            fs::metadata(path)
        }

        fn hash_file(&self, _path: &Path) -> std::io::Result<String> {
            panic!("oversized file must not be hashed")
        }
    }

    fn entry(hash: &str) -> ManifestFileEntry {
        ManifestFileEntry {
            size: 1,
            mtime_ns: 1,
            content_hash: hash.to_string(),
            fingerprint: None,
        }
    }

    fn manifest_with(files: &[(&str, &str)]) -> Manifest {
        let mut manifest = Manifest::new("source-1");
        for (path, hash) in files {
            manifest.files.insert((*path).to_string(), entry(hash));
        }
        manifest
    }

    #[test]
    fn test_manifest_new() {
        let manifest = Manifest::new("source-1");

        assert_eq!(manifest.version, 1);
        assert_eq!(manifest.source_id, "source-1");
        assert!(manifest.files.is_empty());
    }

    #[test]
    fn test_manifest_scan() {
        let dir = tempfile::tempdir().expect("create tempdir");
        fs::write(dir.path().join("a.txt"), "alpha").expect("write a");
        fs::create_dir(dir.path().join("nested")).expect("create nested");
        fs::write(dir.path().join("nested/b.txt"), "beta").expect("write b");
        fs::write(dir.path().join("c.txt"), "gamma").expect("write c");

        let manifest = Manifest::scan(dir.path(), "source-1").expect("scan manifest");

        assert_eq!(manifest.files.len(), 3);
        assert_eq!(manifest.files["a.txt"].size, 5);
        assert_eq!(manifest.files["nested/b.txt"].size, 4);
        assert_eq!(manifest.files["c.txt"].size, 5);
        assert_eq!(
            manifest.files["a.txt"].content_hash,
            prefixed_sha256(b"alpha")
        );
        assert_eq!(
            manifest.files["nested/b.txt"].content_hash,
            prefixed_sha256(b"beta")
        );
        assert_eq!(
            manifest.files["c.txt"].content_hash,
            prefixed_sha256(b"gamma")
        );
    }

    #[test]
    fn test_scan_with_baseline_always_uses_current_content_hash() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let path = dir.path().join("a.txt");
        fs::write(&path, "alpha").expect("write file");
        let mut baseline = Manifest::scan(dir.path(), "source-1").expect("scan baseline");
        baseline
            .files
            .get_mut("a.txt")
            .expect("baseline entry")
            .content_hash = "sha256:baseline".to_string();
        baseline
            .files
            .get_mut("a.txt")
            .expect("baseline entry")
            .fingerprint = None;

        let manifest = Manifest::scan_with_baseline(dir.path(), "source-1", Some(&baseline))
            .expect("scan with baseline");

        assert_eq!(
            manifest.files["a.txt"].content_hash,
            prefixed_sha256(b"alpha")
        );
    }

    #[test]
    fn test_scan_with_baseline_rehashes_modified_size() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let path = dir.path().join("a.txt");
        fs::write(&path, "alpha").expect("write file");
        let mut baseline = Manifest::scan(dir.path(), "source-1").expect("scan baseline");
        baseline
            .files
            .get_mut("a.txt")
            .expect("baseline entry")
            .content_hash = "sha256:baseline".to_string();
        fs::write(&path, "alpha beta").expect("modify file");

        let manifest = Manifest::scan_with_baseline(dir.path(), "source-1", Some(&baseline))
            .expect("scan with baseline");

        assert_eq!(
            manifest.files["a.txt"].content_hash,
            prefixed_sha256(b"alpha beta")
        );
    }

    #[test]
    fn test_scan_with_baseline_rehashes_when_content_hash_is_stale() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let path = dir.path().join("a.txt");
        fs::write(&path, "alpha").expect("write original file");
        let mut baseline = Manifest::scan(dir.path(), "source-1").expect("scan baseline");
        fs::write(&path, "bravo").expect("write same-size replacement");
        let replacement_mtime_ns = fs::metadata(&path)
            .expect("read replacement metadata")
            .modified()
            .expect("read replacement mtime")
            .duration_since(UNIX_EPOCH)
            .expect("replacement mtime after epoch")
            .as_nanos();
        baseline
            .files
            .get_mut("a.txt")
            .expect("baseline entry")
            .mtime_ns = replacement_mtime_ns;

        let manifest = Manifest::scan_with_baseline(dir.path(), "source-1", Some(&baseline))
            .expect("scan with baseline");

        assert_eq!(
            manifest.files["a.txt"].content_hash,
            prefixed_sha256(b"bravo")
        );
    }

    #[cfg(unix)]
    #[test]
    fn test_scan_with_baseline_reuses_matching_fingerprint_without_hashing() {
        let dir = tempfile::tempdir().expect("create tempdir");
        fs::write(dir.path().join("a.txt"), "alpha").expect("write file");
        let baseline = Manifest::scan(dir.path(), "source-1").expect("scan baseline");

        let (manifest, stats) =
            Manifest::scan_with_baseline_stats(dir.path(), "source-1", Some(&baseline))
                .expect("scan with baseline");

        assert_eq!(
            manifest.files["a.txt"].content_hash,
            baseline.files["a.txt"].content_hash
        );
        assert_eq!(stats.files_rehashed, 0);
        assert_eq!(stats.bytes_hashed, 0);
        assert_eq!(stats.files_examined, 1);
    }

    #[cfg(not(unix))]
    #[test]
    fn test_scan_with_baseline_rehashes_when_fingerprint_unavailable() {
        let dir = tempfile::tempdir().expect("create tempdir");
        fs::write(dir.path().join("a.txt"), "alpha").expect("write file");
        let baseline = Manifest::scan(dir.path(), "source-1").expect("scan baseline");

        let (manifest, stats) =
            Manifest::scan_with_baseline_stats(dir.path(), "source-1", Some(&baseline))
                .expect("scan with baseline");

        assert_eq!(
            manifest.files["a.txt"].content_hash,
            baseline.files["a.txt"].content_hash
        );
        assert!(baseline.files["a.txt"].fingerprint.is_none());
        assert!(manifest.files["a.txt"].fingerprint.is_none());
        assert_eq!(stats.files_rehashed, 1);
        assert_eq!(stats.bytes_hashed, 5);
        assert_eq!(stats.files_examined, 1);
    }

    #[test]
    fn test_scan_with_baseline_rehashes_same_size_replacement() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let path = dir.path().join("a.txt");
        fs::write(&path, "alpha").expect("write original file");
        let baseline = Manifest::scan(dir.path(), "source-1").expect("scan baseline");
        let original_mtime = fs::metadata(&path)
            .expect("read original metadata")
            .modified()
            .expect("read original mtime");
        fs::write(&path, "bravo").expect("write replacement");
        filetime::set_file_mtime(&path, filetime::FileTime::from_system_time(original_mtime))
            .expect("restore original mtime");

        let (manifest, stats) =
            Manifest::scan_with_baseline_stats(dir.path(), "source-1", Some(&baseline))
                .expect("scan replacement");

        assert_eq!(
            manifest.files["a.txt"].content_hash,
            prefixed_sha256(b"bravo")
        );
        assert_eq!(stats.files_rehashed, 1);
        assert_eq!(stats.bytes_hashed, 5);
    }

    #[test]
    fn test_scan_without_baseline_rehashes_every_file() {
        let dir = tempfile::tempdir().expect("create tempdir");
        fs::write(dir.path().join("a.txt"), "alpha").expect("write a");
        fs::write(dir.path().join("b.txt"), "bravo").expect("write b");

        let (_manifest, stats) =
            Manifest::scan_with_baseline_stats(dir.path(), "source-1", None).expect("full scan");

        assert_eq!(stats.files_examined, 2);
        assert_eq!(stats.files_rehashed, 2);
        assert_eq!(stats.bytes_hashed, 10);
    }

    #[test]
    fn scan_skips_oversized_files_before_hashing() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let contents = "larger than the cap";
        fs::write(dir.path().join("large.txt"), contents).expect("write file");

        let (manifest, stats) = Manifest::scan_with_fs(dir.path(), "source-1", None, 4, &NoHashFs)
            .expect("scan records the size guard");

        assert!(!manifest.files.contains_key("large.txt"));
        assert_eq!(stats.files_rehashed, 0);
        assert_eq!(stats.bytes_hashed, 0);
        assert_eq!(stats.errors.len(), 1);
        assert_eq!(stats.errors[0].path, "large.txt");
        assert_eq!(stats.errors[0].kind, ScanErrorKind::SizeGuard);
        assert_eq!(
            stats.errors[0].error,
            size_guard_message(contents.len() as u64, 4)
        );
    }

    #[test]
    fn capped_scan_carries_an_old_manifest_entry_as_unknown() {
        let dir = tempfile::tempdir().expect("create tempdir");
        fs::write(dir.path().join("large.txt"), "large content").expect("write file");
        let previous = Manifest::scan(dir.path(), "source-1").expect("create previous manifest");

        let (manifest, stats, carried) = Manifest::scan_keeping_unknown_with_max_file_bytes(
            dir.path(),
            "source-1",
            None,
            Some(&previous),
            4,
        )
        .expect("scan preserves the old entry");

        assert_eq!(manifest.files["large.txt"], previous.files["large.txt"]);
        assert_eq!(carried, vec!["large.txt"]);
        assert_eq!(stats.errors[0].kind, ScanErrorKind::SizeGuard);
    }

    #[test]
    fn test_manifest_scan_with_minsyncignore() {
        let dir = tempfile::tempdir().expect("create tempdir");
        fs::write(
            dir.path().join(".minsyncignore"),
            "ignored.txt\nignored_dir/\n",
        )
        .expect("write ignore");
        fs::write(dir.path().join("kept.txt"), "kept").expect("write kept");
        fs::write(dir.path().join("ignored.txt"), "ignored").expect("write ignored");
        fs::create_dir(dir.path().join("ignored_dir")).expect("create ignored dir");
        fs::write(dir.path().join("ignored_dir/file.txt"), "ignored")
            .expect("write ignored nested");

        let manifest = Manifest::scan(dir.path(), "source-1").expect("scan manifest");

        assert!(manifest.files.contains_key("kept.txt"));
        assert!(manifest.files.contains_key(".minsyncignore"));
        assert!(!manifest.files.contains_key("ignored.txt"));
        assert!(!manifest.files.contains_key("ignored_dir/file.txt"));
    }

    #[test]
    fn test_manifest_scan_skips_dotminsync() {
        let dir = tempfile::tempdir().expect("create tempdir");
        fs::write(dir.path().join("kept.txt"), "kept").expect("write kept");
        fs::create_dir(dir.path().join(".minsync")).expect("create .minsync");
        fs::write(dir.path().join(".minsync/manifest.json"), "internal").expect("write internal");

        let manifest = Manifest::scan(dir.path(), "source-1").expect("scan manifest");

        assert!(manifest.files.contains_key("kept.txt"));
        assert!(!manifest.files.contains_key(".minsync/manifest.json"));
    }

    #[test]
    fn test_manifest_diff_added() {
        let old = manifest_with(&[("a.txt", "sha256:a")]);
        let new = manifest_with(&[("a.txt", "sha256:a"), ("b.txt", "sha256:b")]);

        assert_eq!(
            Manifest::diff(&old, &new),
            vec![FileChange::Added("b.txt".to_string())]
        );
    }

    #[test]
    fn test_manifest_diff_modified() {
        let old = manifest_with(&[("a.txt", "sha256:a")]);
        let new = manifest_with(&[("a.txt", "sha256:b")]);

        assert_eq!(
            Manifest::diff(&old, &new),
            vec![FileChange::Modified("a.txt".to_string())]
        );
    }

    #[test]
    fn test_manifest_diff_deleted() {
        let old = manifest_with(&[("a.txt", "sha256:a"), ("b.txt", "sha256:b")]);
        let new = manifest_with(&[("a.txt", "sha256:a")]);

        assert_eq!(
            Manifest::diff(&old, &new),
            vec![FileChange::Deleted("b.txt".to_string())]
        );
    }

    #[test]
    fn test_manifest_diff_mixed() {
        let old = manifest_with(&[
            ("a.txt", "sha256:a"),
            ("b.txt", "sha256:b"),
            ("c.txt", "sha256:c"),
        ]);
        let new = manifest_with(&[
            ("a.txt", "sha256:changed"),
            ("b.txt", "sha256:b"),
            ("d.txt", "sha256:d"),
        ]);

        assert_eq!(
            Manifest::diff(&old, &new),
            vec![
                FileChange::Modified("a.txt".to_string()),
                FileChange::Deleted("c.txt".to_string()),
                FileChange::Added("d.txt".to_string()),
            ]
        );
    }

    #[test]
    fn test_manifest_diff_no_changes() {
        let old = manifest_with(&[("a.txt", "sha256:a"), ("b.txt", "sha256:b")]);
        let new = manifest_with(&[("a.txt", "sha256:a"), ("b.txt", "sha256:b")]);

        assert!(Manifest::diff(&old, &new).is_empty());
    }

    #[test]
    fn test_manifest_persistence_roundtrip() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let path = dir.path().join("manifest.json");
        let manifest = manifest_with(&[("a.txt", "sha256:a"), ("b.txt", "sha256:b")]);

        manifest.save(&path).expect("save manifest");
        let loaded = Manifest::load(&path).expect("load manifest");

        assert_eq!(manifest, loaded);
    }

    #[test]
    fn test_manifest_hash_deterministic() {
        let first = manifest_with(&[("b.txt", "sha256:b"), ("a.txt", "sha256:a")]);
        let second = manifest_with(&[("a.txt", "sha256:a"), ("b.txt", "sha256:b")]);

        assert_eq!(first.manifest_hash(), second.manifest_hash());
    }

    #[test]
    fn test_manifest_hash_different() {
        let first = manifest_with(&[("a.txt", "sha256:a")]);
        let second = manifest_with(&[("a.txt", "sha256:b")]);

        assert_ne!(first.manifest_hash(), second.manifest_hash());
    }

    #[test]
    fn test_manifest_hash_includes_path() {
        let first = manifest_with(&[("a.txt", "sha256:same")]);
        let second = manifest_with(&[("b.txt", "sha256:same")]);

        assert_ne!(first.manifest_hash(), second.manifest_hash());
    }

    /// Injects per-path I/O failures into the scan.
    #[cfg(unix)]
    struct FaultyFs {
        metadata_errors: Vec<(&'static str, i32)>,
        hash_errors: Vec<(&'static str, i32)>,
    }

    #[cfg(unix)]
    impl FaultyFs {
        fn lookup(table: &[(&'static str, i32)], path: &Path) -> Option<std::io::Error> {
            let name = path.file_name()?.to_str()?;
            table
                .iter()
                .find(|(file, _)| *file == name)
                .map(|(_, errno)| std::io::Error::from_raw_os_error(*errno))
        }
    }

    #[cfg(unix)]
    impl ScanFs for FaultyFs {
        fn metadata(&self, path: &Path) -> std::io::Result<fs::Metadata> {
            match Self::lookup(&self.metadata_errors, path) {
                Some(error) => Err(error),
                None => RealFs.metadata(path),
            }
        }

        fn hash_file(&self, path: &Path) -> std::io::Result<String> {
            match Self::lookup(&self.hash_errors, path) {
                Some(error) => Err(error),
                None => RealFs.hash_file(path),
            }
        }
    }

    #[cfg(unix)]
    fn scan_faulty(dir: &Path, faulty: &FaultyFs) -> (Manifest, ScanStats) {
        Manifest::scan_with_fs(dir, "source-1", None, 0, faulty)
            .expect("scan survives entry errors")
    }

    #[cfg(unix)]
    fn three_files() -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("create tempdir");
        for name in ["a.txt", "b.txt", "c.txt"] {
            fs::write(dir.path().join(name), name).expect("write file");
        }
        dir
    }

    #[cfg(unix)]
    #[test]
    fn test_scan_transient_metadata_error_is_recorded_not_fatal() {
        let dir = three_files();
        let faulty = FaultyFs {
            metadata_errors: vec![("b.txt", libc::EIO)],
            hash_errors: vec![],
        };

        let (manifest, stats) = scan_faulty(dir.path(), &faulty);

        assert!(manifest.files.contains_key("a.txt"));
        assert!(manifest.files.contains_key("c.txt"));
        assert!(!manifest.files.contains_key("b.txt"));
        assert_eq!(stats.errors.len(), 1);
        assert_eq!(stats.errors[0].path, "b.txt");
        assert_eq!(stats.errors[0].kind, ScanErrorKind::Transient);
    }

    #[cfg(unix)]
    #[test]
    fn test_scan_classifies_errno_kinds() {
        let dir = three_files();
        let faulty = FaultyFs {
            metadata_errors: vec![("a.txt", libc::ESTALE)],
            hash_errors: vec![("b.txt", libc::EACCES), ("c.txt", libc::EBUSY)],
        };

        let (manifest, stats) = scan_faulty(dir.path(), &faulty);

        assert!(manifest.files.is_empty());
        let kinds: HashMap<_, _> = stats
            .errors
            .iter()
            .map(|error| (error.path.as_str(), error.kind))
            .collect();
        assert_eq!(kinds["a.txt"], ScanErrorKind::Transient);
        assert_eq!(kinds["b.txt"], ScanErrorKind::Permanent);
        assert_eq!(kinds["c.txt"], ScanErrorKind::Transient);
    }

    #[cfg(unix)]
    #[test]
    fn test_scan_vanished_file_is_gone_not_an_error() {
        let dir = three_files();
        let faulty = FaultyFs {
            metadata_errors: vec![("a.txt", libc::ENOENT)],
            hash_errors: vec![("b.txt", libc::ENOTDIR)],
        };

        let (manifest, stats) = scan_faulty(dir.path(), &faulty);

        assert!(stats.errors.is_empty());
        assert_eq!(manifest.files.keys().collect::<Vec<_>>(), vec!["c.txt"]);
    }

    #[cfg(unix)]
    #[test]
    fn test_transient_error_never_yields_deleted_change() {
        let dir = three_files();
        let previous = Manifest::scan(dir.path(), "source-1").expect("baseline scan");
        let faulty = FaultyFs {
            metadata_errors: vec![("b.txt", libc::EIO)],
            hash_errors: vec![],
        };

        let (mut current, stats) = scan_faulty(dir.path(), &faulty);
        assert_eq!(
            Manifest::diff(&previous, &current),
            vec![FileChange::Deleted("b.txt".to_string())],
            "precondition: raw diff alone would delete b.txt"
        );
        let carried = current.carry_unknown(&previous, &stats.errors);

        assert_eq!(carried, vec!["b.txt"]);
        assert!(Manifest::diff(&previous, &current).is_empty());
    }

    #[test]
    fn test_carry_unknown_covers_descendants_of_an_errored_directory() {
        let previous = manifest_with(&[
            ("dir/a.txt", "sha256:a"),
            ("dir/sub/b.txt", "sha256:b"),
            ("dir2/c.txt", "sha256:c"),
            ("other.txt", "sha256:o"),
        ]);
        let mut current = manifest_with(&[("other.txt", "sha256:o")]);
        let errors = vec![ScanError {
            path: "dir".to_string(),
            kind: ScanErrorKind::Permanent,
            error: "permission denied".to_string(),
        }];

        let carried = current.carry_unknown(&previous, &errors);

        assert_eq!(carried, vec!["dir/a.txt", "dir/sub/b.txt"]);
        assert_eq!(
            Manifest::diff(&previous, &current),
            vec![FileChange::Deleted("dir2/c.txt".to_string())],
            "sibling with a shared name prefix is not unknown"
        );
    }

    #[test]
    fn test_carry_unknown_keeps_fresh_entry_over_stale_one() {
        let previous = manifest_with(&[("a.txt", "sha256:old")]);
        let mut current = manifest_with(&[("a.txt", "sha256:new")]);
        let errors = vec![ScanError {
            path: "a.txt".to_string(),
            kind: ScanErrorKind::Transient,
            error: "x".to_string(),
        }];

        assert!(current.carry_unknown(&previous, &errors).is_empty());
        assert_eq!(current.files["a.txt"].content_hash, "sha256:new");
    }

    #[test]
    fn test_scan_root_error_stays_fatal() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let missing = dir.path().join("does-not-exist");

        assert!(Manifest::scan(&missing, "source-1").is_err());
    }
}
