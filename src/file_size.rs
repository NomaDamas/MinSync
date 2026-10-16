//! Bounded reads for source files that are about to be indexed.

use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

#[derive(Debug)]
pub(crate) enum ReadTextError {
    Io(io::Error),
    TooLarge { size: u64, limit: u64 },
}

pub(crate) fn size_guard_message(size: u64, limit: u64) -> String {
    format!("file is {size} bytes, over max_file_bytes ({limit})")
}

pub(crate) fn size_guard_error(size: u64, limit: u64) -> Option<String> {
    (limit > 0 && size > limit).then(|| size_guard_message(size, limit))
}

/// Read UTF-8 text without allocating more than `limit + 1` bytes when a
/// limit is configured. The extra byte distinguishes an exact-limit file
/// from a file that grew after its metadata was read.
pub(crate) fn read_text_file(path: &Path, limit: u64) -> Result<String, ReadTextError> {
    let file = File::open(path).map_err(ReadTextError::Io)?;
    let size = file.metadata().map_err(ReadTextError::Io)?.len();
    if size_guard_error(size, limit).is_some() {
        return Err(ReadTextError::TooLarge { size, limit });
    }

    if limit == 0 {
        return std::fs::read_to_string(path).map_err(ReadTextError::Io);
    }

    let mut reader = file.take(limit.saturating_add(1));
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes).map_err(ReadTextError::Io)?;
    let observed = bytes.len() as u64;
    if observed > limit {
        let size = reader
            .into_inner()
            .metadata()
            .map(|metadata| metadata.len())
            .unwrap_or(observed)
            .max(observed);
        return Err(ReadTextError::TooLarge { size, limit });
    }

    String::from_utf8(bytes)
        .map_err(|error| ReadTextError::Io(io::Error::new(io::ErrorKind::InvalidData, error)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configured_limit_rejects_a_large_file_from_metadata() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let path = dir.path().join("large.txt");
        let content = "larger than the cap";
        std::fs::write(&path, content).expect("write file");

        let error = read_text_file(&path, 4).expect_err("oversized file is rejected");

        assert!(matches!(
            error,
            ReadTextError::TooLarge { size, limit: 4 } if size == content.len() as u64
        ));
    }

    #[test]
    fn zero_limit_preserves_unbounded_reading() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let path = dir.path().join("large.txt");
        let content = "x".repeat(1024);
        std::fs::write(&path, &content).expect("write file");

        assert_eq!(read_text_file(&path, 0).expect("read file"), content);
    }

    #[test]
    fn file_at_the_limit_is_allowed() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let path = dir.path().join("exact.txt");
        std::fs::write(&path, "exact").expect("write file");

        assert_eq!(
            read_text_file(&path, 5).expect("read exact-limit file"),
            "exact"
        );
    }
}
