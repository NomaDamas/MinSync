use crate::error::{MinSyncError, Result};
use chrono::Utc;
use ignore::{DirEntry, WalkBuilder};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ScanStats {
    pub files_examined: usize,
    pub files_rehashed: usize,
    pub bytes_hashed: u64,
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

    pub fn scan_with_baseline_stats(
        root: &Path,
        source_id: &str,
        baseline: Option<&Manifest>,
    ) -> Result<(Self, ScanStats)> {
        let mut manifest = Self::new(source_id);
        let mut stats = ScanStats::default();
        let walker = WalkBuilder::new(root)
            .add_custom_ignore_filename(".minsyncignore")
            .hidden(false)
            .filter_entry(|entry| !is_internal_dir(entry))
            .build();

        for entry in walker {
            let entry = entry.map_err(|error| MinSyncError::Manifest(error.to_string()))?;
            let file_type = entry.file_type().ok_or_else(|| {
                MinSyncError::Manifest(format!("missing file type: {}", entry.path().display()))
            })?;

            if !file_type.is_file() {
                continue;
            }

            stats.files_examined += 1;
            let path = entry.path();
            let metadata = fs::metadata(path)?;
            let modified = metadata.modified()?;
            let mtime_ns = modified
                .duration_since(UNIX_EPOCH)
                .map_err(|error| MinSyncError::Manifest(error.to_string()))?
                .as_nanos();
            let relative_path = path
                .strip_prefix(root)
                .map_err(|error| MinSyncError::Manifest(error.to_string()))?;
            let manifest_path = relative_path.to_string_lossy().replace('\\', "/");
            let fingerprint = metadata_fingerprint(&metadata, mtime_ns);
            let content_hash = match baseline
                .and_then(|manifest| manifest.files.get(&manifest_path))
                .filter(|entry| fingerprint.is_some() && entry.fingerprint == fingerprint)
            {
                Some(entry) => entry.content_hash.clone(),
                None => {
                    stats.files_rehashed += 1;
                    stats.bytes_hashed += metadata.len();
                    hash_file(path)?
                }
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

fn hash_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)?;
    Ok(format!("sha256:{}", hex::encode(hasher.finalize())))
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
    use std::fs;

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
}
