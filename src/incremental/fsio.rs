//! Durable file primitives shared by the journal and the stage snapshot.

use crate::error::{MinSyncError, Result};
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::Path;

/// Atomically replace `path` with `bytes`: write a sibling temporary file,
/// fsync it, rename it over `path`, then fsync the directory so the rename
/// itself survives a power loss. Readers see the old or the new content,
/// never a mix.
pub(crate) fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut tmp = tempfile::NamedTempFile::new_in(parent)?;
    tmp.write_all(bytes)?;
    tmp.as_file().sync_all()?;
    tmp.persist(path)
        .map_err(|error| MinSyncError::Io(error.error))?;
    sync_dir(parent);
    Ok(())
}

/// Best-effort fsync of a directory (a no-op where the platform cannot open
/// one). A failure only narrows the crash-durability window of the rename;
/// it never changes what a reader sees.
pub(crate) fn sync_dir(dir: &Path) {
    if let Ok(handle) = std::fs::File::open(dir) {
        let _ = handle.sync_all();
    }
}

/// `sha256:<hex>` of the file's bytes, streamed.
pub(crate) fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)?;
    Ok(format!("sha256:{}", hex::encode(hasher.finalize())))
}
