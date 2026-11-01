pub mod cdc;
pub mod chonkie;
pub(crate) mod heading;
pub mod line_cdc;
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
        "cdc" => Ok(Box::new(cdc::CdcChunker::from_config(
            &config.chunker.options,
        ))),
        "line-cdc" => Ok(Box::new(line_cdc::LineCdcChunker::from_config(config))),
        other => Err(MinSyncError::Config(format!("unknown chunker id: {other}"))),
    }
}

#[cfg(test)]
mod factory_tests {
    use super::*;
    use crate::config::Config;

    /// A config for chunker `id` without heading context, so `schema_id` is
    /// the bare id. The context has its own tests in `heading`.
    fn config_for(id: &str) -> Config {
        let mut config = Config::default_for("12345678-1234-4234-9234-123456789abc");
        config.chunker.id = id.to_string();
        config.chunker.options.heading_context = false;
        config
    }

    #[test]
    fn test_create_chunker_recursive() {
        let chunker = create_chunker(&config_for("recursive")).expect("create recursive chunker");
        assert_eq!(chunker.schema_id(), "recursive");
    }

    #[test]
    fn test_create_chunker_chonkie() {
        let chunker = create_chunker(&config_for("chonkie")).expect("create chonkie chunker");
        assert_eq!(chunker.schema_id(), "chonkie");
    }

    #[test]
    fn test_create_chunker_cdc() {
        let chunker = create_chunker(&config_for("cdc")).expect("create cdc chunker");
        assert_eq!(chunker.schema_id(), "cdc");
    }

    #[test]
    fn test_create_chunker_unknown() {
        let result = create_chunker(&config_for("bogus"));
        assert!(matches!(result, Err(MinSyncError::Config(_))));
    }
}
