pub mod chonkie;
pub mod recursive;

use crate::error::Result;
use crate::types::Chunk;

pub trait Chunker: Send + Sync {
    fn schema_id(&self) -> &str;
    fn chunk(&self, text: &str, path: &str) -> Result<Vec<Chunk>>;
}

use crate::config::Config;
use crate::error::MinSyncError;

/// Construct a [`Chunker`] implementation from the chunker id in `config`.
///
/// Returns [`MinSyncError::Config`] when the id is unknown.
pub fn create_chunker(config: &Config) -> Result<Box<dyn Chunker>> {
    match config.chunker.id.as_str() {
        "recursive" => Ok(Box::new(recursive::RecursiveChunker::from_config(
            &config.chunker.options,
        ))),
        "chonkie" => Ok(Box::new(chonkie::ChonkieChunker::from_config(
            &config.chunker.options,
        ))),
        other => Err(MinSyncError::Config(format!("unknown chunker id: {other}"))),
    }
}

#[cfg(test)]
mod factory_tests {
    use super::*;
    use crate::config::Config;

    #[test]
    fn test_create_chunker_recursive() {
        let config = Config::default_for("12345678-1234-4234-9234-123456789abc");
        let chunker = create_chunker(&config).expect("create recursive chunker");
        assert_eq!(chunker.schema_id(), "recursive");
    }

    #[test]
    fn test_create_chunker_chonkie() {
        let mut config = Config::default_for("12345678-1234-4234-9234-123456789abc");
        config.chunker.id = "chonkie".to_string();
        let chunker = create_chunker(&config).expect("create chonkie chunker");
        assert_eq!(chunker.schema_id(), "chonkie");
    }

    #[test]
    fn test_create_chunker_unknown() {
        let mut config = Config::default_for("12345678-1234-4234-9234-123456789abc");
        config.chunker.id = "bogus".to_string();
        let result = create_chunker(&config);
        assert!(matches!(result, Err(MinSyncError::Config(_))));
    }
}
