//! `minsync serve`: one resident native engine shared over loopback HTTP.
//!
//! The server itself (TEI `POST /embed` and `/embed_sparse`, OpenAI
//! `POST /v1/embeddings`, `GET /health`, the state directory with its lock and
//! `pid` / `host` / `port` files, adoption of a running server, idle and
//! wedged-pass shutdown) is [`inl_server`] from `im-not-llm`. This module is
//! what stays MinSync's: which `[embedder]` / `[sparse]` settings name which
//! model instance, and where its state directory lives.

use crate::config::{EmbedderConfig, SparseConfig, DEFAULT_SPARSE_MAX_LENGTH};
use crate::embedder::native::{minsync_cache_root, resolve_backend, Backend, DEFAULT_MAX_LENGTH};
use crate::error::{MinSyncError, Result};
use inl_server::{EngineKind, InstanceKey};
use std::path::PathBuf;

pub use inl_server::{
    loopback_client, parse_loopback_host, probe, start, start_sparse, Health, ServeOptions,
    Started, DEFAULT_IDLE_TIMEOUT_SECONDS, DEFAULT_MAX_INPUTS, DEFAULT_MAX_REQUEST_BYTES,
};

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
    // The engine a model runs on is part of its identity: a Neural Engine
    // server and a Metal server of one model produce different vectors.
    let device = resolved_device(model, false, &settings.device, &settings.dtype)?;
    Ok(inl_server::instance_key(&InstanceKey {
        model: model.to_string(),
        device,
        dtype: settings.dtype.clone(),
        attention: settings.attention.clone(),
        max_length: Some(settings.max_length.unwrap_or(DEFAULT_MAX_LENGTH)),
        kind: EngineKind::Dense,
    })?)
}

/// Identity of a served sparse model. The `-sparse-` infix keeps its state
/// directory apart from a dense model of the same name.
pub fn sparse_instance_key(settings: &SparseConfig) -> Result<String> {
    let Some(model) = settings.id.strip_prefix("native:") else {
        return Err(MinSyncError::Config(format!(
            "minsync serve --sparse only serves native: encoders, got '{}'",
            settings.id
        )));
    };
    let device = resolved_device(model, true, &settings.device, &settings.dtype)?;
    Ok(inl_server::instance_key(&InstanceKey {
        model: model.to_string(),
        device,
        dtype: settings.dtype.clone(),
        attention: None,
        max_length: Some(settings.max_length.unwrap_or(DEFAULT_SPARSE_MAX_LENGTH)),
        kind: EngineKind::Sparse,
    })?)
}

/// The configured device, with a default that lands on the Neural Engine named
/// `ane` so it never shares an instance with the same model on Metal.
fn resolved_device(
    model: &str,
    sparse: bool,
    device: &Option<String>,
    dtype: &Option<String>,
) -> Result<Option<String>> {
    Ok(
        match resolve_backend(model, sparse, device.as_deref(), dtype.as_deref())? {
            Backend::Ane => Some("ane".to_string()),
            Backend::Candle => device.clone(),
        },
    )
}

/// Default state directory for an instance:
/// `~/.cache/minsync/serve/<instance>` (`%LOCALAPPDATA%\minsync\serve` on
/// Windows).
pub fn default_state_dir(instance: &str) -> PathBuf {
    minsync_cache_root().join("serve").join(instance)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense(id: &str) -> EmbedderConfig {
        EmbedderConfig {
            id: id.to_string(),
            ..crate::config::Config::default_for("t").embedder
        }
    }

    #[test]
    fn dense_instances_differ_by_everything_that_changes_vectors() {
        let base = instance_key(&dense("native:org/model")).unwrap();
        assert!(base.starts_with("org-model-"), "{base}");
        assert_eq!(base, instance_key(&dense("native:org/model")).unwrap());

        let mut variants = Vec::new();
        for edit in [
            |s: &mut EmbedderConfig| s.id = "native:org/other".to_string(),
            |s: &mut EmbedderConfig| s.device = Some("cpu".to_string()),
            |s: &mut EmbedderConfig| s.dtype = Some("bf16".to_string()),
            |s: &mut EmbedderConfig| s.attention = Some("eager".to_string()),
            |s: &mut EmbedderConfig| s.max_length = Some(512),
        ] {
            let mut settings = dense("native:org/model");
            edit(&mut settings);
            variants.push(instance_key(&settings).unwrap());
        }
        variants.push(base);
        let unique: std::collections::HashSet<_> = variants.iter().collect();
        assert_eq!(unique.len(), variants.len(), "{variants:?}");
    }

    #[test]
    fn prefixes_batching_and_explicit_auto_do_not_change_the_instance() {
        let base = instance_key(&dense("native:org/model")).unwrap();
        let mut settings = dense("native:org/model");
        settings.query_prefix = Some("Q: ".to_string());
        settings.passage_prefix = Some("P: ".to_string());
        settings.batch_size = 3;
        settings.device = Some("auto".to_string());
        settings.max_length = Some(DEFAULT_MAX_LENGTH);
        assert_eq!(instance_key(&settings).unwrap(), base);
    }

    #[test]
    fn a_sparse_instance_never_collides_with_a_dense_one_of_the_same_model() {
        let sparse = sparse_instance_key(&SparseConfig {
            id: "native:org/model".to_string(),
            ..SparseConfig::default()
        })
        .unwrap();
        assert!(sparse.starts_with("org-model-sparse-"), "{sparse}");
        assert_ne!(sparse, instance_key(&dense("native:org/model")).unwrap());
    }

    #[test]
    fn only_native_ids_can_be_served() {
        let error = instance_key(&dense("tei:foo/bar")).unwrap_err();
        assert!(error.to_string().contains("only serves native"), "{error}");
        let error = sparse_instance_key(&SparseConfig {
            id: "tei:foo/bar".to_string(),
            ..SparseConfig::default()
        })
        .unwrap_err();
        assert!(error.to_string().contains("only serves native"), "{error}");
    }

    /// A server on the Neural Engine and one on Metal produce different
    /// vectors (fp16 vs f32), so an unset device that resolves to the ANE must
    /// not collide with an explicit Metal one.
    #[test]
    fn the_neural_engine_and_metal_are_different_instances() {
        let sparse = |device: Option<&str>| SparseConfig {
            id: "native:nlpai-lab/KURE-Sparse-nano".to_string(),
            device: device.map(str::to_string),
            ..SparseConfig::default()
        };
        let metal = sparse_instance_key(&sparse(Some("metal"))).unwrap();
        let auto = sparse_instance_key(&sparse(None)).unwrap();
        let ane = sparse_instance_key(&sparse(Some("ane"))).unwrap();
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            assert_eq!(auto, ane, "the default on Apple silicon is the ANE");
            assert_ne!(auto, metal);
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        assert_eq!(auto, sparse_instance_key(&sparse(Some("auto"))).unwrap());
        assert_ne!(ane, metal);
    }

    #[test]
    fn state_lives_under_the_minsync_cache() {
        let dir = default_state_dir("inst-1");
        assert!(dir.ends_with("minsync/serve/inst-1"), "{}", dir.display());
    }
}
