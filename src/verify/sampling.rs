//! Expected-doc-id sampling: recompute the doc IDs a file should produce and
//! compare them against the store.

use crate::chunker::Chunker;
use crate::config::Config;
use crate::error::{MinSyncError, Result};
use crate::id::doc_ids_for_chunks;
use crate::manifest::Manifest;
use crate::normalize::normalize_text;
use std::io::ErrorKind;
use std::path::Path;

pub(super) fn sample_paths(manifest: &Manifest, sample: Option<usize>) -> Vec<String> {
    let mut paths: Vec<_> = manifest.files.keys().cloned().collect();
    paths.sort();
    if let Some(limit) = sample {
        paths.truncate(limit);
    }
    paths
}

pub(super) fn expected_doc_ids(
    root: &Path,
    path: &str,
    config: &Config,
    chunker: &dyn Chunker,
) -> Result<Vec<String>> {
    let raw_text = match std::fs::read_to_string(root.join(path)) {
        Ok(text) => text,
        Err(error) if error.kind() == ErrorKind::InvalidData => String::new(),
        Err(error) => return Err(MinSyncError::Io(error)),
    };
    let text = normalize_text(&raw_text, &config.normalize);
    let chunks = chunker.chunk(&text, path)?;

    Ok(
        doc_ids_for_chunks(&config.source_id, path, chunker.schema_id(), &chunks)
            .into_iter()
            .map(|(doc_id, _content_hash)| doc_id)
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::Manifest;

    fn manifest_with_paths(paths: &[&str]) -> Manifest {
        let mut manifest = Manifest::new("source-1");
        for path in paths {
            manifest.files.insert(
                path.to_string(),
                crate::manifest::ManifestFileEntry {
                    size: 0,
                    mtime_ns: 0,
                    content_hash: "hash".to_string(),
                },
            );
        }
        manifest
    }

    #[test]
    fn test_sample_paths_sorted_and_limited() {
        let manifest = manifest_with_paths(&["c.txt", "a.txt", "b.txt"]);

        assert_eq!(
            sample_paths(&manifest, None),
            vec!["a.txt", "b.txt", "c.txt"]
        );
        assert_eq!(sample_paths(&manifest, Some(2)), vec!["a.txt", "b.txt"]);
        assert!(sample_paths(&manifest, Some(0)).is_empty());
    }
}
