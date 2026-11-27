//! The on-disk BM25 inverted index beside a LanceDB table, and the rules for
//! when it may answer a query.
//!
//! Mirrors [`super::sparse_index`]: the index is a derived copy of the table's
//! `text` column tokenized with a specific BM25 language preset. It describes
//! exactly one version of the table, so it is used only when the table is still
//! at that version and the query's language matches. Anything wrong with it
//! falls back to scanning the table directly.

use crate::bm25::inverted::{Bm25IndexMeta, Bm25InvertedFile};
use crate::error::Result;
use crate::vectorstore::Filter;
use std::path::{Path, PathBuf};

/// File name of the BM25 inverted index inside the store directory.
pub(super) const FILE_NAME: &str = "bm25-index.inv";

/// Smallest table worth indexing on disk: below it a streaming scan is already
/// sub-millisecond and keeps small workspaces file-free.
pub(super) const DEFAULT_MIN_ROWS: usize = 64;

pub(super) fn index_path(store_dir: &Path) -> PathBuf {
    store_dir.join(FILE_NAME)
}

/// Remove any leftover temp files from interrupted BM25 index builds.
pub(super) fn cleanup_interrupted_builds(store_dir: &Path) -> Result<()> {
    for entry in std::fs::read_dir(store_dir)? {
        let entry = entry?;
        if entry
            .file_name()
            .to_string_lossy()
            .starts_with(".minsync-bm25-index-")
            && entry.file_type()?.is_file()
        {
            std::fs::remove_file(entry.path())?;
        }
    }
    Ok(())
}

pub(super) fn matches_table(file: &Bm25IndexMeta, version: u64, timestamp_nanos: u128) -> bool {
    file.table_version == version && file.table_timestamp_nanos == timestamp_nanos
}

pub(super) fn describes(
    file: &Bm25IndexMeta,
    language: &str,
    version: u64,
    timestamp_nanos: u128,
) -> bool {
    file.language == language && matches_table(file, version, timestamp_nanos)
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Scope {
    Everything,
    Nothing,
    Unknown,
}

pub(super) fn scope(filter: Option<&Filter>, file: &Bm25IndexMeta) -> Scope {
    match filter {
        None => Scope::Everything,
        Some(Filter::Eq(field, value)) if field == "source_id" => {
            if *value == file.source_id {
                Scope::Everything
            } else {
                Scope::Nothing
            }
        }
        Some(Filter::And(filters)) => {
            let mut all = Scope::Everything;
            for nested in filters {
                match scope(Some(nested), file) {
                    Scope::Nothing => return Scope::Nothing,
                    Scope::Unknown => all = Scope::Unknown,
                    Scope::Everything => {}
                }
            }
            all
        }
        Some(_) => Scope::Unknown,
    }
}

pub(super) fn open(store_dir: &Path) -> Option<Bm25InvertedFile> {
    let path = index_path(store_dir);
    if !path.exists() {
        return None;
    }
    match Bm25InvertedFile::open(&path) {
        Ok(file) => Some(file),
        Err(error) => {
            tracing::warn!("ignoring the bm25 inverted index: {error}");
            None
        }
    }
}

pub(super) fn remove(store_dir: &Path) -> Result<()> {
    match std::fs::remove_file(index_path(store_dir)) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(version: u64) -> Bm25IndexMeta {
        Bm25IndexMeta {
            language: "multilingual".into(),
            source_id: "s".into(),
            table_version: version,
            table_timestamp_nanos: 5,
        }
    }

    #[test]
    fn describes_requires_exact_language_version_and_timestamp() {
        assert!(describes(&meta(3), "multilingual", 3, 5));
        assert!(!describes(&meta(3), "ko", 3, 5));
        assert!(!describes(&meta(3), "multilingual", 4, 5));
        assert!(!describes(&meta(3), "multilingual", 3, 6));
    }

    #[test]
    fn scope_matches_source_id_filters() {
        let file = meta(1);
        let source = |s: &str| Filter::Eq("source_id".into(), s.into());
        let path = Filter::Eq("path".into(), "a.txt".into());

        assert_eq!(scope(None, &file), Scope::Everything);
        assert_eq!(scope(Some(&source("s")), &file), Scope::Everything);
        assert_eq!(scope(Some(&source("other")), &file), Scope::Nothing);
        assert_eq!(
            scope(Some(&Filter::And(vec![source("s")])), &file),
            Scope::Everything
        );
        assert_eq!(scope(Some(&path), &file), Scope::Unknown);
        assert_eq!(
            scope(Some(&Filter::And(vec![source("other"), path])), &file),
            Scope::Nothing
        );
    }
}
