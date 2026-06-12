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
        let mut manifest = Self::new(source_id);
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
            let content_hash = match baseline
                .and_then(|manifest| manifest.files.get(&manifest_path))
                .filter(|entry| entry.size == metadata.len() && entry.mtime_ns == mtime_ns)
            {
                Some(entry) => entry.content_hash.clone(),
                None => hash_file(path)?,
            };

            manifest.files.insert(
                manifest_path,
                ManifestFileEntry {
                    size: metadata.len(),
                    mtime_ns,
                    content_hash,
                },
            );
        }

        Ok(manifest)
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn entry(hash: &str) -> ManifestFileEntry {
        ManifestFileEntry {
            size: 1,
            mtime_ns: 1,
            content_hash: hash.to_string(),
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
    fn test_scan_with_baseline_reuses_matching_metadata_hash() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let path = dir.path().join("a.txt");
        fs::write(&path, "alpha").expect("write file");
        let mut baseline = Manifest::scan(dir.path(), "source-1").expect("scan baseline");
        baseline
            .files
            .get_mut("a.txt")
            .expect("baseline entry")
            .content_hash = "sha256:baseline".to_string();

        let manifest = Manifest::scan_with_baseline(dir.path(), "source-1", Some(&baseline))
            .expect("scan with baseline");

        assert_eq!(manifest.files["a.txt"].content_hash, "sha256:baseline");
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
