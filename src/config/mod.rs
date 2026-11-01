//! Config loading and saving. The data model lives in [`model`].

mod model;

pub use model::*;

use crate::error::{MinSyncError, Result};
use std::path::Path;

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        toml::from_str(&content).map_err(MinSyncError::TomlParse)
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
        assert_eq!(config.embedder.id, DEFAULT_EMBEDDER_ID);
        assert_eq!(config.embedder.batch_size, 64);
        assert_eq!(config.embedder.max_concurrent, 1);
        assert_eq!(config.embedder.max_retries, 3);
        assert_eq!(config.embedder.timeout_seconds, 60);
        assert_eq!(config.embedder.per_batch_timeout_seconds, None);
        assert_eq!(config.embedder.base_url, None);
        assert_eq!(config.embedder.query_prefix, None);
        assert_eq!(config.embedder.passage_prefix, None);
        assert_eq!(config.embedder.device, None);
        assert_eq!(config.embedder.dtype, None);
        assert_eq!(config.embedder.max_length, None);
        assert_eq!(config.embedder.max_batch_tokens, None);
        assert_eq!(config.embedder.model_cache_dir, None);
        assert_eq!(config.vectorstore.id, "lancedb");
        assert_eq!(config.sync.max_file_bytes, DEFAULT_MAX_FILE_BYTES);
        assert_eq!(
            config.vectorstore.options["dimension"].as_integer(),
            Some(1024)
        );
        assert!(config.normalize.strip_trailing_whitespace);
        assert!(config.normalize.normalize_newlines);
        assert!(!config.normalize.collapse_whitespace);
        assert!(!config.normalize.strip_frontmatter);
    }

    #[test]
    fn test_known_embedding_dimension() {
        assert_eq!(known_embedding_dimension(DEFAULT_EMBEDDER_ID), Some(1024));
        assert_eq!(
            known_embedding_dimension("native:Qwen/Qwen3-Embedding-0.6B"),
            Some(1024)
        );
        assert_eq!(
            known_embedding_dimension("tei:google/embeddinggemma-300m"),
            Some(768)
        );
        assert_eq!(
            known_embedding_dimension("openai:text-embedding-3-small"),
            Some(1536)
        );
        assert_eq!(
            known_embedding_dimension("openai:text-embedding-3-large"),
            Some(3072)
        );
        assert_eq!(
            known_embedding_dimension("tei:intfloat/multilingual-e5-small"),
            Some(384)
        );
        assert_eq!(known_embedding_dimension("tei:BAAI/bge-m3"), Some(1024));
        assert_eq!(known_embedding_dimension("tei:custom/model"), None);
        assert_eq!(
            known_embedding_dimension(NONE_EMBEDDER_ID),
            Some(SPARSE_ONLY_DIMENSION)
        );
    }

    #[test]
    fn test_sparse_section_defaults_to_the_kure_model_and_is_optional_on_disk() {
        let config = Config::default_for("12345678-1234-4234-9234-123456789abc");
        assert_eq!(config.sparse.id, DEFAULT_SPARSE_ID);
        assert_eq!(DEFAULT_SPARSE_ID, "native:nlpai-lab/KURE-Sparse-nano");
        assert_eq!(config.sparse.query_prefix.as_deref(), Some("[Q] "));
        assert_eq!(config.sparse.passage_prefix.as_deref(), Some("[D] "));

        // A config written before `[sparse]` existed still loads, with defaults.
        let full = toml::to_string_pretty(&config).unwrap();
        let start = full.find("[sparse]").expect("serialized [sparse] table");
        let end = full[start + 1..]
            .find("\n[")
            .map_or(full.len(), |offset| start + 1 + offset + 1);
        let legacy = format!("{}{}", &full[..start], &full[end..]);
        assert!(!legacy.contains("[sparse]"), "{legacy}");
        let loaded: Config = toml::from_str(&legacy).unwrap();
        assert_eq!(loaded.sparse, config.sparse);
    }

    #[test]
    fn test_sparse_section_roundtrips_remote_settings() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let path = dir.path().join("config.toml");
        let mut config = Config::default_for("abcdef12-1234-4234-9234-123456789abc");
        config.sparse.id = "tei:nlpai-lab/KURE-Sparse-nano".to_string();
        config.sparse.base_url = Some("http://localhost:8081".to_string());
        config.save(&path).expect("save config");
        assert_eq!(Config::load(&path).expect("load config"), config);
    }

    #[test]
    fn test_is_sparse_only_only_for_none_embedder() {
        let mut config = Config::default_for("12345678-1234-4234-9234-123456789abc");
        assert!(!config.is_sparse_only());
        config.embedder.id = "none".to_string();
        assert!(config.is_sparse_only());
        config.embedder.id = "none:foo".to_string();
        assert!(!config.is_sparse_only());
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
    fn test_heading_context_roundtrips_and_is_absent_from_files_that_leave_it_off() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let path = dir.path().join("config.toml");
        let mut config = Config::default_for("abcdef12-1234-4234-9234-123456789abc");

        // Off: the file (and the run fingerprint hashed from it) is what it
        // was before the option existed.
        config.save(&path).expect("save config");
        assert!(!std::fs::read_to_string(&path)
            .unwrap()
            .contains("heading_context"));

        config.chunker.options.heading_context = true;
        config.save(&path).expect("save config");
        assert!(std::fs::read_to_string(&path)
            .unwrap()
            .contains("heading_context = true"));
        assert_eq!(Config::load(&path).expect("load config"), config);
    }

    #[test]
    fn test_config_from_the_bm25_era_still_loads_and_drops_its_lexical_table() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let path = dir.path().join("config.toml");
        let config = Config::default_for("abcdef12-1234-4234-9234-123456789abc");
        config.save(&path).expect("save config");
        // What `minsync init --language ko` wrote before BM25 was replaced.
        let legacy = format!(
            "{}\n[lexical]\nlanguage = \"ko\"\n",
            std::fs::read_to_string(&path).expect("read config")
        );
        std::fs::write(&path, legacy).expect("write legacy config");

        let loaded = Config::load(&path).expect("a legacy [lexical] table must not break load");

        assert_eq!(loaded, config);
        loaded.save(&path).expect("save again");
        assert!(
            !std::fs::read_to_string(&path)
                .expect("read config")
                .contains("lexical"),
            "saving writes the current shape only"
        );
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
        // Configs written before max_file_bytes was added keep the safe default.
        assert_eq!(config.sync.max_file_bytes, DEFAULT_MAX_FILE_BYTES);
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
