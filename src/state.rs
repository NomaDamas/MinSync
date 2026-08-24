use crate::error::{MinSyncError, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;
use tempfile::NamedTempFile;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Cursor {
    pub source_id: String,
    pub last_synced_at: String,
    pub manifest_hash: String,
    pub chunk_schema_id: String,
    pub embedder_id: String,
    pub collection_path: String,
    #[serde(default = "default_lexical_language")]
    pub lexical_language: String,
}

fn default_lexical_language() -> String {
    "simple".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Transaction {
    pub source_id: String,
    pub sync_token: String,
    pub manifest_hash_from: Option<String>,
    pub manifest_hash_to: String,
    pub status: String,
    pub started_at: String,
}

pub struct FileLock {
    file: std::fs::File,
}

impl Cursor {
    pub fn load(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let cursor = serde_json::from_str(&content)?;
        Ok(cursor)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        atomic_write_json(self, path)
    }
}

impl Transaction {
    pub fn new(
        source_id: &str,
        sync_token: &str,
        from_hash: Option<String>,
        to_hash: &str,
    ) -> Self {
        Self {
            source_id: source_id.to_string(),
            sync_token: sync_token.to_string(),
            manifest_hash_from: from_hash,
            manifest_hash_to: to_hash.to_string(),
            status: "running".to_string(),
            started_at: Utc::now().to_rfc3339(),
        }
    }

    pub fn load(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let transaction = serde_json::from_str(&content)?;
        Ok(transaction)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        atomic_write_json(self, path)
    }

    pub fn remove(path: &Path) -> Result<()> {
        match fs::remove_file(path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(MinSyncError::Io(error)),
        }
    }
}

impl FileLock {
    /// Acquire exclusive lock. If wait=false and lock held, return LockFailed error.
    pub fn acquire(path: &Path, wait: bool) -> Result<Self> {
        use fs2::FileExt;

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(path)?;

        if wait {
            file.lock_exclusive()
                .map_err(|_| MinSyncError::LockFailed)?;
        } else {
            file.try_lock_exclusive()
                .map_err(|_| MinSyncError::LockFailed)?;
        }

        file.set_len(0)?;
        let mut f = &file;
        f.seek(SeekFrom::Start(0))?;
        write!(f, "{}", std::process::id())?;
        f.flush()?;

        Ok(Self { file })
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}

fn atomic_write_json<T: Serialize>(value: &T, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let tmp_parent = path.parent().unwrap_or_else(|| Path::new("."));
    let mut tmp = NamedTempFile::new_in(tmp_parent)?;
    let content = serde_json::to_vec_pretty(value)?;
    tmp.write_all(&content)?;
    tmp.flush()?;
    tmp.as_file().sync_all()?;
    tmp.persist(path)
        .map_err(|error| MinSyncError::Io(error.error))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Seek, SeekFrom};

    fn cursor() -> Cursor {
        Cursor {
            source_id: "source-1".to_string(),
            last_synced_at: "2026-05-28T00:00:00Z".to_string(),
            manifest_hash: "sha256:abc".to_string(),
            chunk_schema_id: "schema-1".to_string(),
            embedder_id: "embedder-1".to_string(),
            collection_path: ".minsync/collection".to_string(),
            lexical_language: "simple".to_string(),
        }
    }

    #[test]
    fn test_cursor_roundtrip() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let path = dir.path().join("cursor.json");
        let cursor = cursor();

        cursor.save(&path).expect("save cursor");
        let loaded = Cursor::load(&path).expect("load cursor");

        assert_eq!(cursor, loaded);
    }

    #[test]
    fn test_cursor_load_missing() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let path = dir.path().join("missing.json");

        assert!(Cursor::load(&path).is_err());
    }

    #[test]
    fn test_transaction_new() {
        let transaction = Transaction::new(
            "source-1",
            "0123456789abcdef",
            Some("sha256:from".to_string()),
            "sha256:to",
        );

        assert_eq!(transaction.source_id, "source-1");
        assert_eq!(transaction.sync_token, "0123456789abcdef");
        assert_eq!(
            transaction.manifest_hash_from,
            Some("sha256:from".to_string())
        );
        assert_eq!(transaction.manifest_hash_to, "sha256:to");
        assert_eq!(transaction.status, "running");
        assert!(!transaction.started_at.is_empty());
    }

    #[test]
    fn test_transaction_roundtrip() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let path = dir.path().join("txn.json");
        let transaction = Transaction::new("source-1", "token-1", None, "sha256:to");

        transaction.save(&path).expect("save transaction");
        let loaded = Transaction::load(&path).expect("load transaction");

        assert_eq!(transaction, loaded);
    }

    #[test]
    fn test_transaction_remove() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let path = dir.path().join("txn.json");
        let transaction = Transaction::new("source-1", "token-1", None, "sha256:to");

        transaction.save(&path).expect("save transaction");
        Transaction::remove(&path).expect("remove transaction");

        assert!(!path.exists());
    }

    #[test]
    fn test_transaction_remove_nonexistent() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let path = dir.path().join("missing.json");

        Transaction::remove(&path).expect("remove missing transaction");
    }

    #[test]
    fn test_filelock_acquire_release() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let path = dir.path().join("lock");

        {
            let lock = FileLock::acquire(&path, false).expect("acquire lock");
            assert!(path.exists());
            drop(lock);
        }

        let _lock = FileLock::acquire(&path, false).expect("re-acquire lock");
    }

    #[test]
    fn test_filelock_contention() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let path = dir.path().join("lock");
        let _lock = FileLock::acquire(&path, false).expect("acquire lock");

        let result = FileLock::acquire(&path, false);

        assert!(matches!(result, Err(MinSyncError::LockFailed)));
    }

    #[test]
    fn test_filelock_writes_pid() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let path = dir.path().join("lock");
        let lock = FileLock::acquire(&path, false).expect("acquire lock");

        let mut content = String::new();
        let mut file = &lock.file;
        file.seek(SeekFrom::Start(0)).expect("seek lock content");
        file.read_to_string(&mut content)
            .expect("read lock content");

        assert_eq!(content, std::process::id().to_string());
    }
}
