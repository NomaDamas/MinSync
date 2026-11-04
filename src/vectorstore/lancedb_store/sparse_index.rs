//! The on-disk inverted index beside a LanceDB table, and the rules for when
//! it may answer a query.
//!
//! The index is a derived copy of the table's sparse columns: it is never the
//! source of truth, and anything wrong with it means "scan the table instead".
//! It describes exactly one version of the table, so it is used only when the
//! table is still at that version. It lives next to the table directory, not
//! inside it (Lance owns everything under `documents.lance/`); LanceDB only
//! lists `*.lance` entries, so it ignores the file.

use crate::error::Result;
use crate::sparse::inverted::{IndexMeta, InvertedFile};
use crate::vectorstore::Filter;
use std::path::{Path, PathBuf};

/// File name of the index inside the store directory.
pub(super) const FILE_NAME: &str = "sparse-index.inv";

/// Smallest table worth indexing: below it the scan is as fast as an index
/// lookup and the file is only more to keep consistent.
pub(super) const DEFAULT_MIN_ROWS: usize = 2_000;

/// Postings held in memory while an index is written (8 bytes each: 256 MiB).
/// Each pass re-reads the sparse columns, so a larger budget means fewer
/// scans; a corpus of 100k chunks (about 75M postings) takes three.
pub(super) const DEFAULT_MEMORY_POSTINGS: usize = 32_000_000;

/// Most candidates worth reading back by id, for an index of `rows` rows:
/// beyond it a streaming scan of the table is cheaper than the id lookups
/// (for example `k` close to the table size), and gives the same answer.
pub(super) fn max_candidates(rows: usize) -> usize {
    rows / 4
}

pub(super) fn index_path(store_dir: &Path) -> PathBuf {
    store_dir.join(FILE_NAME)
}

/// Whether `file` was built from this table version. The creation time is
/// compared too: a table that was dropped and rebuilt, or restored, can reuse
/// a version number.
pub(super) fn matches_table(file: &IndexMeta, version: u64, timestamp_nanos: u128) -> bool {
    file.table_version == version && file.table_timestamp_nanos == timestamp_nanos
}

/// Whether `file` can answer a query for `model` against this table version.
pub(super) fn describes(
    file: &IndexMeta,
    model: &str,
    version: u64,
    timestamp_nanos: u128,
) -> bool {
    file.model == model && matches_table(file, version, timestamp_nanos)
}

/// Which rows of the index a filter selects.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Scope {
    /// Every indexed row: the index answers as the scan would.
    Everything,
    /// No row: the answer is empty.
    Nothing,
    /// Something the index cannot evaluate (a path, a chunk type): scan.
    Unknown,
}

/// An index covers every row of one model and one `source_id` (the builder
/// refuses anything else), so the only filters it can answer are the ones
/// that name that source, or no filter at all.
pub(super) fn scope(filter: Option<&Filter>, file: &IndexMeta) -> Scope {
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

/// Open the index if there is one. A missing file is `None`; a file that does
/// not open (torn, truncated, from another format) is `None` too, because the
/// caller can always fall back to the scan.
pub(super) fn open(store_dir: &Path) -> Option<InvertedFile> {
    let path = index_path(store_dir);
    if !path.exists() {
        return None;
    }
    match InvertedFile::open(&path) {
        Ok(file) => Some(file),
        Err(error) => {
            tracing::warn!("ignoring the sparse inverted index: {error}");
            None
        }
    }
}

/// Remove a stale or damaged index. A leftover file only costs the next build
/// an overwrite, so a missing one is not an error.
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

    fn meta(version: u64) -> IndexMeta {
        IndexMeta {
            model: "m".into(),
            source_id: "s".into(),
            table_version: version,
            table_timestamp_nanos: 5,
        }
    }

    #[test]
    fn an_index_describes_only_the_exact_version_and_model_it_was_built_from() {
        assert!(describes(&meta(3), "m", 3, 5));
        assert!(!describes(&meta(3), "m", 4, 5), "another version");
        assert!(!describes(&meta(3), "x", 3, 5), "another model");
        assert!(
            !describes(&meta(3), "m", 3, 6),
            "a restored table can reuse a version number"
        );
        assert!(matches_table(&meta(3), 3, 5));
    }

    #[test]
    fn only_filters_that_select_every_indexed_row_or_none_are_answered() {
        let file = meta(1);
        let source = |s: &str| Filter::Eq("source_id".into(), s.into());
        let path = Filter::Eq("path".into(), "a.txt".into());

        assert_eq!(scope(None, &file), Scope::Everything);
        assert_eq!(scope(Some(&source("s")), &file), Scope::Everything);
        assert_eq!(scope(Some(&source("other")), &file), Scope::Nothing);
        assert_eq!(scope(Some(&Filter::And(vec![])), &file), Scope::Everything);
        // What `minsync query` sends.
        assert_eq!(
            scope(Some(&Filter::And(vec![source("s")])), &file),
            Scope::Everything
        );
        assert_eq!(scope(Some(&path), &file), Scope::Unknown);
        assert_eq!(
            scope(Some(&Filter::And(vec![source("s"), path.clone()])), &file),
            Scope::Unknown
        );
        assert_eq!(
            scope(Some(&Filter::And(vec![source("other"), path])), &file),
            Scope::Nothing,
            "a conjunct that selects nothing decides the whole filter"
        );
        assert_eq!(
            scope(Some(&Filter::Neq("source_id".into(), "s".into())), &file),
            Scope::Unknown
        );
    }

    #[test]
    fn candidates_are_read_back_by_id_only_while_that_beats_a_scan() {
        assert_eq!(max_candidates(100_000), 25_000);
        assert_eq!(max_candidates(3), 0);
    }

    #[test]
    fn a_missing_or_garbage_file_is_no_index() {
        let dir = tempfile::tempdir().unwrap();
        assert!(open(dir.path()).is_none());

        std::fs::write(index_path(dir.path()), b"not an index at all").unwrap();
        assert!(open(dir.path()).is_none());

        remove(dir.path()).unwrap();
        remove(dir.path()).unwrap();
        assert!(!index_path(dir.path()).exists());
    }
}
