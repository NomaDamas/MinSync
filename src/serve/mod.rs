//! `minsync serve`: MinSync's view of the im-not-llm loopback daemon.
//!
//! The daemon itself is [`inl_server`]: one resident engine per model instance,
//! served over loopback HTTP (TEI `POST /embed`, OpenAI `POST /v1/embeddings`,
//! sparse `POST /embed_sparse`, `GET /health`), with a state directory holding
//! `lock` / `pid` / `host` / `port`; a second `serve` that finds the lock taken
//! adopts the running server instead of loading another copy of the model.
//!
//! This module is the adapter between the two: it turns MinSync's `[embedder]`
//! / `[sparse]` settings into the daemon's model instance identity, and
//! re-exports what `minsync serve` and the `served` providers need.
//!
//! State directory (changed from MinSync's own `~/.cache/minsync/serve`): the
//! daemon keeps its files under `~/.cache/im-not-llm/serve/<instance>`
//! (`%LOCALAPPDATA%\im-not-llm\serve` on Windows), so a `minsync serve` and an
//! `inl serve` of the same model share one instance.

use crate::config::{EmbedderConfig, SparseConfig, DEFAULT_SPARSE_MAX_LENGTH};
use crate::error::{MinSyncError, Result};

pub use inl_server::{
    default_state_dir, find_instances, parse_loopback_host, probe, start, start_sparse, Daemon,
    DaemonClient, Engine, EngineKind, FoundInstance, Health, InstanceKey, ServeOptions, Server,
    ServerHandle, Started, StopReason, DEFAULT_IDLE_TIMEOUT_SECONDS, DEFAULT_MAX_INPUTS,
    DEFAULT_MAX_REQUEST_BYTES,
};

/// HTTP client for talking to a daemon on this machine.
///
/// The server is loopback plain HTTP, so TLS never runs; building the client
/// without the platform root-certificate store skips what would otherwise be
/// the dominant cost of every CLI start (reading the system keychain took
/// ~90 ms on macOS). Proxies are bypassed because the peer is local.
pub(crate) fn loopback_client(timeout: std::time::Duration) -> Option<reqwest::Client> {
    reqwest::Client::builder()
        .timeout(timeout)
        .no_proxy()
        .tls_built_in_root_certs(false)
        .build()
        .ok()
}

/// Identity of a served dense model: everything that changes the vectors the
/// native engine produces. Prefixes and batching knobs are excluded: callers
/// send prefixed text, and batching does not change results beyond
/// left-padding noise.
pub fn instance_key(settings: &EmbedderConfig) -> Result<String> {
    let Some(model) = settings.id.strip_prefix("native:") else {
        return Err(MinSyncError::Config(format!(
            "minsync serve only serves native: embedders, got '{}'",
            settings.id
        )));
    };
    Ok(inl_server::instance_key(&InstanceKey {
        model: model.to_string(),
        device: settings.device.clone(),
        dtype: settings.dtype.clone(),
        attention: settings.attention.clone(),
        max_length: Some(
            settings
                .max_length
                .unwrap_or(crate::embedder::native::DEFAULT_MAX_LENGTH),
        ),
        kind: EngineKind::Dense,
    })?)
}

/// Identity of a served sparse model: everything that changes the vectors the
/// native sparse engine produces. The `-sparse` infix the daemon adds keeps its
/// state directory apart from a dense model of the same name.
pub fn sparse_instance_key(settings: &SparseConfig) -> Result<String> {
    let Some(model) = settings.id.strip_prefix("native:") else {
        return Err(MinSyncError::Config(format!(
            "minsync serve --sparse only serves native: encoders, got '{}'",
            settings.id
        )));
    };
    Ok(inl_server::instance_key(&InstanceKey {
        model: model.to_string(),
        device: settings.device.clone(),
        dtype: settings.dtype.clone(),
        attention: None,
        max_length: Some(settings.max_length.unwrap_or(DEFAULT_SPARSE_MAX_LENGTH)),
        kind: EngineKind::Sparse,
    })?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn dense() -> EmbedderConfig {
        let mut config = Config::default_for("12345678-1234-4234-9234-123456789abc");
        config.embedder.id = "native:Qwen/Qwen3-Embedding-0.6B".to_string();
        config.embedder
    }

    #[test]
    fn only_native_models_can_be_served() {
        let mut settings = dense();
        settings.id = "tei:foo/bar".to_string();
        let error = instance_key(&settings).unwrap_err();
        assert!(error.to_string().contains("only serves native"), "{error}");

        let sparse = SparseConfig {
            id: "tei:foo/bar".to_string(),
            ..SparseConfig::default()
        };
        let error = sparse_instance_key(&sparse).unwrap_err();
        assert!(error.to_string().contains("only serves native"), "{error}");
    }

    #[test]
    fn the_instance_key_follows_what_changes_the_vectors() {
        let base = instance_key(&dense()).unwrap();
        assert!(base.starts_with("Qwen-Qwen3-Embedding-0.6B-"), "{base}");

        for tweak in [
            (|s: &mut EmbedderConfig| s.device = Some("cpu".to_string()))
                as fn(&mut EmbedderConfig),
            |s: &mut EmbedderConfig| s.dtype = Some("f32".to_string()),
            |s: &mut EmbedderConfig| s.attention = Some("eager".to_string()),
            |s: &mut EmbedderConfig| s.max_length = Some(512),
        ] {
            let mut settings = dense();
            tweak(&mut settings);
            assert_ne!(instance_key(&settings).unwrap(), base, "{settings:?}");
        }

        // Prefixes and batching knobs do not change the vectors.
        let mut settings = dense();
        settings.query_prefix = Some("q: ".to_string());
        settings.passage_prefix = Some("p: ".to_string());
        settings.batch_size = 7;
        settings.timeout_seconds = 900;
        assert_eq!(instance_key(&settings).unwrap(), base);
    }

    #[test]
    fn the_sparse_key_differs_from_a_dense_one_for_the_same_model() {
        let sparse = SparseConfig {
            id: "native:Qwen/Qwen3-Embedding-0.6B".to_string(),
            ..SparseConfig::default()
        };
        let sparse_key = sparse_instance_key(&sparse).unwrap();
        assert!(sparse_key.contains("-sparse-"), "{sparse_key}");
        assert_ne!(sparse_key, instance_key(&dense()).unwrap());
    }
}
