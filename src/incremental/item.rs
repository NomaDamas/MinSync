//! Item identity and path hygiene at the host boundary.
//!
//! An item key is a normalized relative path or a host-provided document id.
//! The core never turns a key into a file path, but a host might, so the
//! rules are strict enough that a key can never traverse out of a directory.

use crate::error::{MinSyncError, Result};
use std::path::{Component, Path};

const MAX_BYTES: usize = 4096;

/// Why `value` is not a normalized relative identifier, if it is not.
fn relative_problem(value: &str) -> Option<&'static str> {
    if value.is_empty() {
        return Some("is empty");
    }
    if value.len() > MAX_BYTES {
        return Some("is longer than 4096 bytes");
    }
    if value.chars().any(char::is_control) {
        return Some("contains a control character");
    }
    if value.contains('\\') {
        return Some("contains a backslash");
    }
    if value.starts_with('/') {
        return Some("is absolute");
    }
    if value.split('/').any(|part| matches!(part, "" | "." | "..")) {
        return Some("has an empty, `.` or `..` component");
    }
    None
}

/// Check a host-supplied item key.
pub fn validate_item_key(key: &str) -> Result<()> {
    match relative_problem(key) {
        None => Ok(()),
        Some(problem) => Err(MinSyncError::stage(
            "invalid_item_key",
            format!("item key {key:?} {problem}"),
        )),
    }
}

/// Check an output path (relative to the output root) before it is joined
/// to anything.
pub fn validate_output_path(path: &str) -> Result<()> {
    let problem = relative_problem(path).or_else(|| {
        // A string that looks relative can still be a drive prefix on
        // Windows; only plain names may be joined to the output root.
        (!Path::new(path)
            .components()
            .all(|component| matches!(component, Component::Normal(_))))
        .then_some("is not a plain relative path")
    });
    match problem {
        None => Ok(()),
        Some(problem) => Err(MinSyncError::stage(
            "invalid_output_path",
            format!("output path {path:?} {problem}"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_normalized_relative_keys_and_document_ids() {
        for key in [
            "a.txt",
            "dir/a b.md",
            "문서/한글.md",
            "doc:123",
            "..foo",
            "a..b",
            "deep/er/path/file.tar.gz",
        ] {
            validate_item_key(key).unwrap_or_else(|error| panic!("{key:?}: {error}"));
        }
    }

    #[test]
    fn rejects_keys_that_are_not_normalized_relative_identifiers() {
        let too_long = "x".repeat(5000);
        for key in [
            "",
            "/abs",
            "a//b",
            "a/",
            "./a",
            "a/./b",
            "../a",
            "a/../b",
            "a\\b",
            "a\0b",
            "a\nb",
            too_long.as_str(),
        ] {
            let error = validate_item_key(key).expect_err(&format!("{key:?} must be rejected"));
            assert!(
                matches!(
                    error,
                    MinSyncError::Stage {
                        code: "invalid_item_key",
                        ..
                    }
                ),
                "{key:?}: {error:?}"
            );
        }
    }

    #[test]
    fn output_paths_must_stay_relative_and_inside_the_root() {
        for path in ["a.md", "sub/dir/a.md", "문서.md", "a b.md"] {
            validate_output_path(path).unwrap_or_else(|error| panic!("{path:?}: {error}"));
        }
        for path in [
            "",
            "/etc/passwd",
            "../escape.md",
            "a/../../escape.md",
            "a/./b.md",
            "a//b.md",
            "a\\b.md",
            "dir/",
            "a\0b",
        ] {
            let error = validate_output_path(path).expect_err(&format!("{path:?} must fail"));
            assert!(
                matches!(
                    error,
                    MinSyncError::Stage {
                        code: "invalid_output_path",
                        ..
                    }
                ),
                "{path:?}: {error:?}"
            );
        }
    }
}
