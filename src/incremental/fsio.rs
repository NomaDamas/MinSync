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

/// `sha256:<hex>` of everything `reader` yields, streamed.
pub(crate) fn sha256_reader(reader: &mut impl std::io::Read) -> std::io::Result<String> {
    let mut hasher = Sha256::new();
    std::io::copy(reader, &mut hasher)?;
    Ok(format!("sha256:{}", hex::encode(hasher.finalize())))
}

/// `sha256:<hex>` of the file's bytes, streamed.
pub(crate) fn sha256_file(path: &Path) -> std::io::Result<String> {
    sha256_reader(&mut std::fs::File::open(path)?)
}

/// Open `path` for reading, refusing a symlink as its last component. The
/// refusal is part of the `open` itself (`O_NOFOLLOW`), so a link swapped in
/// after an earlier check cannot be followed. Parent directories are still
/// resolved normally: the output root is host-managed, not adversarial.
pub(crate) fn open_no_follow(path: &Path) -> std::io::Result<std::fs::File> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW)
            .open(path)
    }
    #[cfg(not(unix))]
    {
        std::fs::File::open(path)
    }
}
