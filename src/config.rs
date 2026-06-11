use crate::error::{MinSyncError, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Config {
    pub version: u32,
    pub source_id: String,
    pub collection: CollectionConfig,
    pub chunker: ChunkerConfig,
    pub embedder: EmbedderConfig,
    pub vectorstore: VectorStoreConfig,
    #[serde(default)]
    pub normalize: NormalizeConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CollectionConfig {
    pub name: String,
    pub path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChunkerConfig {
    pub id: String,
    #[serde(default)]
    pub options: ChunkerOptions,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChunkerOptions {
    #[serde(default = "default_max_chunk_size")]
    pub max_chunk_size: usize,
    #[serde(default = "default_delimiters")]
    pub delimiters: String,
}

fn default_max_chunk_size() -> usize {
    4096
}

fn default_delimiters() -> String {
    "\n.?!".to_string()
}

impl Default for ChunkerOptions {
    fn default() -> Self {
        Self {
            max_chunk_size: default_max_chunk_size(),
            delimiters: default_delimiters(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EmbedderConfig {
    pub id: String,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent: usize,
    #[serde(default = "default_max_retries")]
    pub max_retries: usize,
    #[serde(default = "default_timeout_seconds")]
    pub timeout_seconds: u64,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub query_prefix: Option<String>,
    #[serde(default)]
    pub passage_prefix: Option<String>,
}

fn default_batch_size() -> usize {
    64
}

fn default_max_concurrent() -> usize {
    1
}

fn default_max_retries() -> usize {
    3
}

fn default_timeout_seconds() -> u64 {
    60
}

/// For `vectorstore.id = "lancedb"`, `options.dimension` sets the embedding
/// dimension (default 1536 for `openai:text-embedding-3-small`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VectorStoreConfig {
    pub id: String,
    #[serde(default = "default_vectorstore_options")]
    pub options: toml::Value,
}

fn default_vectorstore_options() -> toml::Value {
    toml::Value::Table(toml::map::Map::new())
}

fn default_lancedb_options() -> toml::Value {
    let mut table = toml::map::Map::new();
    table.insert("dimension".to_string(), toml::Value::Integer(1536));
    toml::Value::Table(table)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NormalizeConfig {
    #[serde(default = "bool_true")]
    pub strip_trailing_whitespace: bool,
    #[serde(default = "bool_true")]
    pub normalize_newlines: bool,
    #[serde(default)]
    pub collapse_whitespace: bool,
    #[serde(default)]
    pub strip_frontmatter: bool,
}

fn bool_true() -> bool {
    true
}

impl Default for NormalizeConfig {
    fn default() -> Self {
        Self {
            strip_trailing_whitespace: true,
            normalize_newlines: true,
            collapse_whitespace: false,
            strip_frontmatter: false,
        }
    }
}

impl Config {
    pub fn default_for(source_id: &str) -> Self {
        let prefix: String = source_id.chars().take(8).collect();

        Self {
            version: 1,
            source_id: source_id.to_string(),
            collection: CollectionConfig {
                name: format!("minsync_{prefix}"),
                path: "store".to_string(),
            },
            chunker: ChunkerConfig {
                id: "recursive".to_string(),
                options: ChunkerOptions::default(),
            },
            embedder: EmbedderConfig {
                id: "openai:text-embedding-3-small".to_string(),
                batch_size: default_batch_size(),
                max_concurrent: default_max_concurrent(),
                max_retries: default_max_retries(),
                timeout_seconds: default_timeout_seconds(),
                base_url: None,
                query_prefix: None,
                passage_prefix: None,
            },
            vectorstore: VectorStoreConfig {
                id: "lancedb".to_string(),
                options: default_lancedb_options(),
            },
            normalize: NormalizeConfig::default(),
        }
    }

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
