//! Config loading and saving. The data model lives in [`model`].

mod model;

pub use model::*;

use crate::error::{MinSyncError, Result};
use std::path::Path;

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config = toml::from_str(&content).map_err(MinSyncError::TomlParse)?;
        Ok(config)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        use std::io::Write;
        use tempfile::NamedTempFile;

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let toml_str = toml::to_string_pretty(self).map_err(MinSyncError::TomlSerialize)?;
        let tmp_parent = path.parent().unwrap_or_else(|| Path::new("."));
        let mut tmp = NamedTempFile::new_in(tmp_parent)?;
        tmp.write_all(toml_str.as_bytes())?;
        tmp.flush()?;
        tmp.persist(path)
            .map_err(|error| MinSyncError::Io(error.error))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default_for() {
        let config = Config::default_for("12345678-1234-4234-9234-123456789abc");

        assert_eq!(config.version, 1);
        assert_eq!(config.source_id, "12345678-1234-4234-9234-123456789abc");
        assert_eq!(config.collection.name, "minsync_12345678");
        assert_eq!(config.collection.path, "store");
        assert_eq!(config.chunker.id, "recursive");
        assert_eq!(config.chunker.options.max_chunk_size, 4096);
        assert_eq!(config.chunker.options.delimiters, "\n.?!");
        assert_eq!(config.embedder.id, "openai:text-embedding-3-small");
        assert_eq!(config.embedder.batch_size, 64);
        assert_eq!(config.embedder.max_concurrent, 1);
        assert_eq!(config.embedder.max_retries, 3);
        assert_eq!(config.embedder.timeout_seconds, 60);
        assert_eq!(config.embedder.base_url, None);
        assert_eq!(config.embedder.query_prefix, None);
        assert_eq!(config.embedder.passage_prefix, None);
        assert_eq!(config.vectorstore.id, "lancedb");
        assert_eq!(
            config.vectorstore.options["dimension"].as_integer(),
            Some(1536)
        );
        assert!(config.normalize.strip_trailing_whitespace);
        assert!(config.normalize.normalize_newlines);
        assert!(!config.normalize.collapse_whitespace);
        assert!(!config.normalize.strip_frontmatter);
    }

    #[test]
    fn test_config_toml_roundtrip() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let path = dir.path().join("config.toml");
        let config = Config::default_for("abcdef12-1234-4234-9234-123456789abc");

        config.save(&path).expect("save config");
        let loaded = Config::load(&path).expect("load config");

        assert_eq!(config, loaded);
    }

    #[test]
    fn test_config_load_invalid() {
        let path = std::path::Path::new("/definitely/not/a/minsync/config.toml");

        assert!(Config::load(path).is_err());
    }

    #[test]
    fn test_config_custom_values() {
        let toml = r#"
version = 1
source_id = "source-1"

[collection]
name = "custom_collection"
path = "custom_store"

[chunker]
id = "custom_chunker"

[chunker.options]
max_chunk_size = 1024
delimiters = "\n;"

[embedder]
id = "custom_embedder"
batch_size = 8
max_concurrent = 4
max_retries = 9
timeout_seconds = 5
base_url = "http://localhost:8080"
query_prefix = "query: "
passage_prefix = "passage: "

[vectorstore]
id = "custom_store"

[vectorstore.options]
url = "http://localhost:1234"
dimension = 1536

[normalize]
strip_trailing_whitespace = false
normalize_newlines = false
collapse_whitespace = true
strip_frontmatter = true
"#;

        let config: Config = toml::from_str(toml).expect("parse custom config");

        assert_eq!(config.version, 1);
        assert_eq!(config.source_id, "source-1");
        assert_eq!(config.collection.name, "custom_collection");
        assert_eq!(config.collection.path, "custom_store");
        assert_eq!(config.chunker.id, "custom_chunker");
        assert_eq!(config.chunker.options.max_chunk_size, 1024);
        assert_eq!(config.chunker.options.delimiters, "\n;");
        assert_eq!(config.embedder.id, "custom_embedder");
        assert_eq!(config.embedder.batch_size, 8);
        assert_eq!(config.embedder.max_concurrent, 4);
        assert_eq!(config.embedder.max_retries, 9);
        assert_eq!(config.embedder.timeout_seconds, 5);
        assert_eq!(
            config.embedder.base_url.as_deref(),
            Some("http://localhost:8080")
        );
        assert_eq!(config.embedder.query_prefix.as_deref(), Some("query: "));
        assert_eq!(config.embedder.passage_prefix.as_deref(), Some("passage: "));
        assert_eq!(config.vectorstore.id, "custom_store");
        assert_eq!(
            config.vectorstore.options["url"].as_str(),
            Some("http://localhost:1234")
        );
        assert_eq!(
            config.vectorstore.options["dimension"].as_integer(),
            Some(1536)
        );
        assert!(!config.normalize.strip_trailing_whitespace);
        assert!(!config.normalize.normalize_newlines);
        assert!(config.normalize.collapse_whitespace);
        assert!(config.normalize.strip_frontmatter);
    }

    #[test]
    fn test_embedder_config_optional_fields() {
        let toml_with_fields = r#"
version = 1
source_id = "source-1"

[collection]
name = "custom_collection"
path = "custom_store"

[chunker]
id = "custom_chunker"

[vectorstore]
id = "custom_store"

[embedder]
id = "custom_embedder"
batch_size = 8
max_concurrent = 4
max_retries = 9
base_url = "http://localhost:8080"
query_prefix = "query: "
passage_prefix = "passage: "

[normalize]
strip_trailing_whitespace = true
normalize_newlines = true
collapse_whitespace = false
strip_frontmatter = false
"#;

        let config_with_fields: Config =
            toml::from_str(toml_with_fields).expect("parse config with embedder optional fields");
        assert_eq!(
            config_with_fields.embedder.base_url.as_deref(),
            Some("http://localhost:8080")
        );
        assert_eq!(
            config_with_fields.embedder.query_prefix.as_deref(),
            Some("query: ")
        );
        assert_eq!(
            config_with_fields.embedder.passage_prefix.as_deref(),
            Some("passage: ")
        );

        let toml_without_fields = r#"
version = 1
source_id = "source-1"

[collection]
name = "custom_collection"
path = "custom_store"

[chunker]
id = "custom_chunker"

[vectorstore]
id = "custom_store"

[embedder]
id = "custom_embedder"
batch_size = 8
max_concurrent = 4
max_retries = 9

[normalize]
strip_trailing_whitespace = true
normalize_newlines = true
collapse_whitespace = false
strip_frontmatter = false
"#;

        let config_without_fields: Config = toml::from_str(toml_without_fields)
            .expect("parse config without embedder optional fields");
        assert_eq!(config_without_fields.embedder.base_url, None);
        assert_eq!(config_without_fields.embedder.query_prefix, None);
        assert_eq!(config_without_fields.embedder.passage_prefix, None);
    }
}
