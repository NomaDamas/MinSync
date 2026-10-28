use crate::error::{LockHolder, MinSyncError, Result};
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

/// How long a bounded wait sleeps between lock attempts.
const LOCK_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(100);

impl FileLock {
    /// Acquire exclusive lock. If wait=false and lock held, return LockFailed error.
    pub fn acquire(path: &Path, wait: bool) -> Result<Self> {
        Self::acquire_with_timeout(path, wait, None)
    }

    /// Acquire the exclusive lock.
    ///
    /// - `wait == false`: fail at once with [`MinSyncError::LockFailed`] when
    ///   the lock is held.
    /// - `wait == true`, `timeout == None`: block until the lock is free.
    /// - `wait == true`, `timeout == Some(d)`: retry for up to `d`, then fail
    ///   with the same `LockFailed`.
    ///
    /// `LockFailed` carries the holder (pid, start time, command) read back
    /// from the lock file. The lock file holds this process's [`LockHolder`]
    /// as JSON while the lock is held.
    pub fn acquire_with_timeout(
        path: &Path,
        wait: bool,
        timeout: Option<std::time::Duration>,
    ) -> Result<Self> {
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

        let lock_failed = || MinSyncError::LockFailed {
            holder: read_lock_holder(path),
        };
        match (wait, timeout) {
            (true, None) => file.lock_exclusive().map_err(|_| lock_failed())?,
            (true, Some(timeout)) => {
                let deadline = std::time::Instant::now() + timeout;
                while file.try_lock_exclusive().is_err() {
                    let now = std::time::Instant::now();
                    if now >= deadline {
                        return Err(lock_failed());
                    }
                    std::thread::sleep(LOCK_POLL_INTERVAL.min(deadline - now));
                }
            }
            (false, _) => file.try_lock_exclusive().map_err(|_| lock_failed())?,
        }

        let holder = LockHolder {
            pid: Some(std::process::id()),
            started_at: Some(Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)),
            command: current_command(),
        };
        file.set_len(0)?;
        let mut f = &file;
        f.seek(SeekFrom::Start(0))?;
        serde_json::to_writer(&mut f, &holder)?;
        f.flush()?;

        Ok(Self { file })
    }
}

/// Parse lock file content: the JSON [`LockHolder`] written by this version,
/// or a bare pid written by MinSync <= 0.5.1. Anything else yields `None`.
pub fn parse_lock_holder(content: &str) -> Option<LockHolder> {
    let content = content.trim();
    if content.is_empty() {
        return None;
    }
    if content.starts_with('{') {
        return serde_json::from_str(content).ok();
    }
    content.parse::<u32>().ok().map(|pid| LockHolder {
        pid: Some(pid),
        ..LockHolder::default()
    })
}

/// Best-effort read of who holds the lock at `path`; `None` when the file is
/// missing, unreadable (e.g. mandatory locking) or not yet written.
pub fn read_lock_holder(path: &Path) -> Option<LockHolder> {
    fs::read_to_string(path)
        .ok()
        .and_then(|content| parse_lock_holder(&content))
}

/// The running command line with the program reduced to its file name,
/// e.g. `minsync sync --full`.
fn current_command() -> Option<String> {
    let mut args = std::env::args();
    let program = args.next()?;
    let program = Path::new(&program)
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or(program);
    Some(
        std::iter::once(program)
            .chain(args)
            .collect::<Vec<_>>()
            .join(" "),
    )
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

        assert!(matches!(result, Err(MinSyncError::LockFailed { .. })));
    }

    #[test]
    fn test_filelock_writes_holder() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let path = dir.path().join("lock");
        let lock = FileLock::acquire(&path, false).expect("acquire lock");

        let mut content = String::new();
        let mut file = &lock.file;
        file.seek(SeekFrom::Start(0)).expect("seek lock content");
        file.read_to_string(&mut content)
            .expect("read lock content");

        let holder = parse_lock_holder(&content).expect("parse lock holder");
        assert_eq!(holder.pid, Some(std::process::id()));
        let started_at = holder.started_at.expect("started_at recorded");
        chrono::DateTime::parse_from_rfc3339(&started_at).expect("RFC 3339 start time");
        assert!(holder.command.is_some_and(|command| !command.is_empty()));
    }

    #[test]
    fn test_filelock_contention_reports_holder() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let path = dir.path().join("lock");
        let _lock = FileLock::acquire(&path, false).expect("acquire lock");

        let Err(error) = FileLock::acquire(&path, false) else {
            panic!("second acquire must fail");
        };
        let MinSyncError::LockFailed { holder } = &error else {
            panic!("expected LockFailed, got {error:?}");
        };
        let holder = holder.as_ref().expect("holder read from the lock file");
        assert_eq!(holder.pid, Some(std::process::id()));
        let message = error.to_string();
        assert!(
            message.starts_with("another sync is in progress (pid "),
            "{message}"
        );
        assert!(message.contains("started "), "{message}");
    }

    #[test]
    fn test_filelock_wait_timeout_gives_up() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let path = dir.path().join("lock");
        let _lock = FileLock::acquire(&path, false).expect("acquire lock");

        let started = std::time::Instant::now();
        let result = FileLock::acquire_with_timeout(
            &path,
            true,
            Some(std::time::Duration::from_millis(300)),
        );

        assert!(matches!(
            result,
            Err(MinSyncError::LockFailed { holder: Some(_) })
        ));
        let elapsed = started.elapsed();
        assert!(
            elapsed >= std::time::Duration::from_millis(300),
            "{elapsed:?}"
        );
        assert!(elapsed < std::time::Duration::from_secs(5), "{elapsed:?}");
    }

    #[test]
    fn test_filelock_wait_timeout_acquires_once_released() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let path = dir.path().join("lock");
        let lock = FileLock::acquire(&path, false).expect("acquire lock");
        let releaser = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(200));
            drop(lock);
        });

        let acquired =
            FileLock::acquire_with_timeout(&path, true, Some(std::time::Duration::from_secs(10)));
        releaser.join().expect("join releaser");

        assert!(acquired.is_ok(), "{:?}", acquired.err());
    }

    #[test]
    fn test_parse_lock_holder_formats() {
        assert_eq!(
            parse_lock_holder("12345"),
            Some(LockHolder {
                pid: Some(12345),
                ..LockHolder::default()
            }),
            "bare pid from MinSync <= 0.5.1"
        );
        assert_eq!(
            parse_lock_holder(
                r#"{"pid":7,"started_at":"2026-10-15T10:00:00Z","command":"minsync sync --full"}"#
            ),
            Some(LockHolder {
                pid: Some(7),
                started_at: Some("2026-10-15T10:00:00Z".to_string()),
                command: Some("minsync sync --full".to_string()),
            })
        );
        assert_eq!(parse_lock_holder(""), None);
        assert_eq!(parse_lock_holder("   \n"), None);
        assert_eq!(parse_lock_holder("not a pid"), None);
        assert_eq!(parse_lock_holder("{truncated"), None);
    }

    #[test]
    fn test_lock_failed_message_without_holder() {
        assert_eq!(
            MinSyncError::LockFailed { holder: None }.to_string(),
            "another sync is in progress"
        );
        let old_format = MinSyncError::LockFailed {
            holder: parse_lock_holder("42"),
        };
        assert_eq!(
            old_format.to_string(),
            "another sync is in progress (pid 42)"
        );
    }
}
