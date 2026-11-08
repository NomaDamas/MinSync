//! What a stored dense vector depends on, recorded so a changed embedder is
//! noticed instead of silently mixed into an index built by another one.
//!
//! Two vectors are comparable only when the same model produced them from
//! the same text: the model id, the output width, and the prefix prepended to
//! passages all change the vector. Everything else in `[embedder]` (device,
//! dtype, batch size, timeouts, the *query* prefix) either leaves the vector
//! alone to within noise or only matters at query time, so it is not part of
//! the identity.

use crate::config::Config;
use crate::embedder::Embedder;
use crate::error::Result;
use serde::{Deserialize, Serialize};

/// The embedder a set of stored vectors was built with.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EmbedderIdentity {
    /// The id the embedder reports (provider prefix stripped), as in
    /// `Cursor::embedder_id`.
    pub model: String,
    /// Width of the stored vector column.
    pub dimension: usize,
    /// Prefix prepended to every passage before embedding.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub passage_prefix: Option<String>,
}

impl EmbedderIdentity {
    /// The identity the configuration asks for, without building the
    /// embedder (so a query can be checked before any model is loaded).
    pub fn from_config(config: &Config) -> Result<Self> {
        let id = config.embedder.id.as_str();
        let model = ["native:", "tei:", "openai:"]
            .iter()
            .find_map(|provider| id.strip_prefix(provider))
            .unwrap_or(id);
        Self::with_model(config, model.to_string())
    }

    /// Like [`Self::from_config`], with the model id the live `embedder`
    /// reports (what a sync records in the cursor).
    pub fn of(config: &Config, embedder: &dyn Embedder) -> Result<Self> {
        Self::with_model(config, embedder.id().to_string())
    }

    fn with_model(config: &Config, model: String) -> Result<Self> {
        Ok(Self {
            model,
            dimension: configured_dimension(config)?,
            // An empty prefix prepends nothing, exactly like no prefix.
            passage_prefix: config
                .embedder
                .passage_prefix
                .clone()
                .filter(|prefix| !prefix.is_empty()),
        })
    }

    /// Why `self` (stored) cannot serve `wanted`, or `None` when it can.
    pub fn difference(&self, wanted: &Self) -> Option<String> {
        let mut changes = Vec::new();
        if self.model != wanted.model {
            changes.push(format!("model {} -> {}", self.model, wanted.model));
        }
        if self.dimension != wanted.dimension {
            changes.push(format!(
                "dimension {} -> {}",
                self.dimension, wanted.dimension
            ));
        }
        if self.passage_prefix != wanted.passage_prefix {
            changes.push(format!(
                "passage_prefix {:?} -> {:?}",
                self.passage_prefix.as_deref().unwrap_or(""),
                wanted.passage_prefix.as_deref().unwrap_or("")
            ));
        }
        (!changes.is_empty()).then(|| changes.join(", "))
    }
}

/// Width of the vector column the configuration asks for.
pub fn configured_dimension(config: &Config) -> Result<usize> {
    if config.is_sparse_only() {
        return Ok(crate::config::SPARSE_ONLY_DIMENSION);
    }
    crate::vectorstore::lancedb_store::LanceDbStore::dimension_from_options(Some(
        &config.vectorstore.options,
    ))
}

/// Width of the dense vector column the index actually holds: what its
/// cursor recorded, else what the configuration says (an index written
/// before identities were recorded).
pub fn stored_dimension(minsync_dir: &std::path::Path, config: &Config) -> Result<usize> {
    let recorded = crate::state::Cursor::load(&minsync_dir.join("cursor.json"))
        .ok()
        .and_then(|cursor| cursor.embedder)
        .map(|identity| identity.dimension);
    match recorded {
        Some(dimension) => Ok(dimension),
        None => configured_dimension(config),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(model: &str, dimension: usize, prefix: Option<&str>) -> EmbedderIdentity {
        EmbedderIdentity {
            model: model.to_string(),
            dimension,
            passage_prefix: prefix.map(str::to_string),
        }
    }

    #[test]
    fn identical_identities_have_no_difference() {
        let a = identity("m", 8, Some("p: "));
        assert_eq!(a.difference(&a.clone()), None);
    }

    #[test]
    fn model_dimension_and_passage_prefix_each_count_and_are_all_named() {
        let stored = identity("old", 4, None);
        let wanted = identity("new", 8, Some("p: "));
        let why = stored.difference(&wanted).expect("differs");
        assert!(why.contains("old") && why.contains("new"), "{why}");
        assert!(why.contains("dimension 4 -> 8"), "{why}");
        assert!(why.contains("passage_prefix"), "{why}");
    }

    #[test]
    fn an_empty_passage_prefix_is_the_same_as_none() {
        let mut config = Config::default_for("s");
        config.embedder.passage_prefix = Some(String::new());
        assert_eq!(
            EmbedderIdentity::from_config(&config)
                .unwrap()
                .passage_prefix,
            None
        );
    }

    #[test]
    fn the_provider_prefix_is_stripped_like_the_embedders_report_it() {
        for (id, model) in [
            (
                "native:Qwen/Qwen3-Embedding-0.6B",
                "Qwen/Qwen3-Embedding-0.6B",
            ),
            ("tei:BAAI/bge-m3", "BAAI/bge-m3"),
            ("openai:text-embedding-3-small", "text-embedding-3-small"),
            ("none", "none"),
        ] {
            let mut config = Config::default_for("s");
            config.embedder.id = id.to_string();
            assert_eq!(EmbedderIdentity::from_config(&config).unwrap().model, model);
        }
    }

    #[test]
    fn the_sparse_only_workspace_has_the_placeholder_dimension() {
        let mut config = Config::default_for("s");
        config.embedder.id = crate::config::NONE_EMBEDDER_ID.to_string();
        assert_eq!(
            configured_dimension(&config).unwrap(),
            crate::config::SPARSE_ONLY_DIMENSION
        );
    }

    #[test]
    fn a_cursor_written_before_identities_existed_still_parses() {
        let json = r#"{"model":"m","dimension":4}"#;
        let parsed: EmbedderIdentity = serde_json::from_str(json).unwrap();
        assert_eq!(parsed, identity("m", 4, None));
    }
}
