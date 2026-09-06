//! Config data model and defaults.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Config {
    pub version: u32,
    pub source_id: String,
    pub collection: CollectionConfig,
    pub chunker: ChunkerConfig,
    pub embedder: EmbedderConfig,
    pub vectorstore: VectorStoreConfig,
    #[serde(default)]
    pub lexical: LexicalConfig,
    #[serde(default)]
    pub normalize: NormalizeConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LexicalConfig {
    #[serde(default = "default_lexical_language")]
    pub language: String,
}

fn default_lexical_language() -> String {
    "simple".to_string()
}

impl Default for LexicalConfig {
    fn default() -> Self {
        Self {
            language: default_lexical_language(),
        }
    }
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
    /// Ask TEI-compatible servers to truncate inputs at the model context.
    /// Keep disabled by default so oversized content is surfaced explicitly.
    #[serde(default)]
    pub truncate: bool,
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
                truncate: false,
            },
            vectorstore: VectorStoreConfig {
                id: "lancedb".to_string(),
                options: default_lancedb_options(),
            },
            lexical: LexicalConfig::default(),
            normalize: NormalizeConfig::default(),
        }
    }
}
