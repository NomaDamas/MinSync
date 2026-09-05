//! Result accounting helpers for sync runs.

use crate::manifest::FileChange;
use crate::types::SyncResult;

pub(super) fn change_path(change: &FileChange) -> String {
    match change {
        FileChange::Added(path) | FileChange::Modified(path) | FileChange::Deleted(path) => {
            path.clone()
        }
    }
}

pub(super) fn empty_sync_result(dry_run: bool, already_up_to_date: bool) -> SyncResult {
    SyncResult {
        files_processed: 0,
        chunks_added: 0,
        chunks_updated: 0,
        chunks_deleted: 0,
        dry_run,
        already_up_to_date,
        initial_sync: false,
        files_processed_paths: Vec::new(),
        files_added: 0,
        files_modified: 0,
        files_deleted: 0,
        elapsed_seconds: 0.0,
        embedding_api_calls: 0,
        embedded_texts: 0,
        estimated_tokens: 0,
        files_checked: 0,
        freshness_check_only: already_up_to_date,
        query_ready: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_change_path_covers_all_variants() {
        assert_eq!(
            change_path(&FileChange::Added("a.txt".to_string())),
            "a.txt"
        );
        assert_eq!(
            change_path(&FileChange::Modified("b.txt".to_string())),
            "b.txt"
        );
        assert_eq!(
            change_path(&FileChange::Deleted("c.txt".to_string())),
            "c.txt"
        );
    }

    #[test]
    fn test_empty_sync_result_flags() {
        let result = empty_sync_result(true, false);
        assert!(result.dry_run);
        assert!(!result.already_up_to_date);
        assert!(!result.initial_sync);
        assert_eq!(result.files_processed, 0);
        assert_eq!(result.chunks_added, 0);
    }
}
