//! `minsync serve`: one resident native engine shared over loopback HTTP.
//!
//! The server itself (TEI `POST /embed` and `/embed_sparse`, OpenAI
//! `POST /v1/embeddings`, `GET /health`, the state directory with its lock and
//! `pid` / `host` / `port` files, adoption of a running server, idle and
//! wedged-pass shutdown) is [`inl_server`] from `im-not-llm`. This module is
//! what stays MinSync's: which `[embedder]` / `[sparse]` settings name which
//! model instance, and where its state directory lives.

use crate::config::{EmbedderConfig, SparseConfig, DEFAULT_SPARSE_MAX_LENGTH};
use crate::embedder::native::{resolve_backend, Backend, DEFAULT_MAX_LENGTH};
use crate::error::{MinSyncError, Result};
use inl_server::{FoundInstance, InstanceKey};
use std::path::{Path, PathBuf};

pub use inl_server::{
    default_state_dir, idle_timeout_from_seconds, loopback_client, parse_loopback_host, probe,
    start, start_multimodal, start_sparse, EngineKind, Health, ServeOptions, Started,
    DEFAULT_IDLE_TIMEOUT_SECONDS, DEFAULT_MAX_INPUTS, DEFAULT_MAX_REQUEST_BYTES,
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

/// Identity of a served multimodal model. The `-multimodal-` infix keeps its
/// state directory apart from dense and sparse instances of the same model.
pub fn multimodal_instance_key(settings: &EmbedderConfig) -> Result<String> {
    let model = settings
        .id
        .strip_prefix("native-multimodal:")
        .or_else(|| settings.id.strip_prefix("native:"))
        .ok_or_else(|| {
            MinSyncError::Config(format!(
                "minsync serve --multimodal only serves native/native-multimodal embedders, got '{}'",
                settings.id
            ))
        })?;
    if model.is_empty() {
        return Err(MinSyncError::Config(
            "multimodal embedder id is missing a model name".to_string(),
        ));
    }
    Ok(inl_server::instance_key(&InstanceKey {
        model: model.to_string(),
        device: settings.device.clone(),
        dtype: settings.dtype.clone(),
        attention: settings.attention.clone(),
        max_length: Some(
            settings
                .max_length
                .unwrap_or(crate::embedder::multimodal::DEFAULT_MULTIMODAL_MAX_LENGTH),
        ),
        kind: EngineKind::Multimodal,
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

/// The directory every launcher (`inl serve`, `minsync serve`) publishes its
/// instance directories in: `<inl cache root>/serve`
/// (`~/.cache/im-not-llm/serve`).
pub fn serve_root() -> PathBuf {
    inl_core::device::cache_root().join("serve")
}

/// A running server, started by any launcher, that serves `model` as `kind`
/// and would produce the same vectors this workspace expects — whatever its
/// instance key (which encodes device, dtype, attention and `max_length`, a
/// key a consumer cannot rebuild for a server it did not start).
///
/// Adoption guard: `max_length` must equal the configured one, and a device
/// or dtype the config pinned (not `auto`/`None`) must match. When the config
/// leaves device/dtype to the engine, the server's own launcher picked them
/// and the embedding space is the same, so it is adopted.
pub async fn find_adoptable(
    serve_root: &Path,
    model: &str,
    kind: EngineKind,
    max_length: usize,
    device: Option<&str>,
    dtype: Option<&str>,
) -> Option<FoundInstance> {
    inl_server::find_instances(serve_root, model, kind)
        .await
        .into_iter()
        .find(|found| {
            let health = &found.health;
            health.is_ready()
                && health.max_length == max_length
                && pinned(device).is_none_or(|want| health.device == want)
                && pinned(dtype).is_none_or(|want| health.dtype == want)
        })
}

/// The configured value when it pins a backend, else `None` (`auto` / unset
/// leaves the choice to the engine).
fn pinned(value: Option<&str>) -> Option<&str> {
    match value {
        Some(value) if !value.eq_ignore_ascii_case("auto") => Some(value),
        _ => None,
    }
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
    fn a_multimodal_instance_never_collides_with_dense_or_sparse() {
        let multimodal =
            multimodal_instance_key(&dense("native-multimodal:google/embeddinggemma-2")).unwrap();
        assert!(
            multimodal.starts_with("google-embeddinggemma-2-multimodal-"),
            "{multimodal}"
        );
        assert_ne!(
            multimodal,
            instance_key(&dense("native:google/embeddinggemma-2")).unwrap()
        );
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

    fn kure(device: Option<&str>) -> SparseConfig {
        SparseConfig {
            id: "native:nlpai-lab/KURE-Sparse-nano".to_string(),
            device: device.map(str::to_string),
            ..SparseConfig::default()
        }
    }

    /// A server on the Neural Engine and one on Metal produce different
    /// vectors (fp16 vs f32), so an unset device that resolves to the ANE must
    /// not collide with an explicit Metal one.
    #[cfg(target_os = "macos")]
    #[test]
    fn the_neural_engine_and_metal_are_different_instances() {
        let metal = sparse_instance_key(&kure(Some("metal"))).unwrap();
        let auto = sparse_instance_key(&kure(None)).unwrap();
        let ane = sparse_instance_key(&kure(Some("ane"))).unwrap();
        assert_ne!(ane, metal);
        if cfg!(target_arch = "aarch64") {
            assert_eq!(auto, ane, "the default on Apple silicon is the ANE");
            assert_ne!(auto, metal);
        } else {
            assert_eq!(auto, sparse_instance_key(&kure(Some("auto"))).unwrap());
        }
    }

    /// Off macOS an unset device is the candle path, and `device = "ane"` is a
    /// config error that names the cause, not an instance of its own.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn off_macos_the_default_is_candle_and_ane_is_refused() {
        assert_eq!(
            sparse_instance_key(&kure(None)).unwrap(),
            sparse_instance_key(&kure(Some("auto"))).unwrap()
        );
        let error = sparse_instance_key(&kure(Some("ane"))).unwrap_err();
        assert!(matches!(error, MinSyncError::Config(_)), "{error:?}");
        assert!(error.to_string().contains("Apple silicon"), "{error}");
    }

    #[test]
    fn state_lives_under_the_im_not_llm_cache() {
        let dir = default_state_dir("inst-1");
        assert!(
            dir.ends_with("im-not-llm/serve/inst-1"),
            "{}",
            dir.display()
        );
    }

    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn health_json(
        server: &MockServer,
        instance: &str,
        kind: &str,
        model: &str,
        max_length: usize,
        device: &str,
        dtype: &str,
    ) -> serde_json::Value {
        let addr = server.address();
        serde_json::json!({
            "status": "ok", "kind": kind, "model_id": model, "dimension": 2,
            "dtype": dtype, "device": device, "attention": "auto", "max_length": max_length,
            "max_inputs": 512, "instance": instance, "pid": 1,
            "host": addr.ip().to_string(), "port": addr.port(),
            "url": format!("http://{addr}"), "uptime_seconds": 1, "requests_total": 0,
            "loaded": true, "build": {"version": "test", "features": []}, "version": "test",
        })
    }

    /// A serve root with one live server published under `instance`, as
    /// another launcher would leave it.
    async fn published(
        instance: &str,
        kind: &str,
        model: &str,
        max_length: usize,
        device: &str,
        dtype: &str,
    ) -> (MockServer, tempfile::TempDir) {
        let server = MockServer::builder().start().await;
        Mock::given(method("GET"))
            .and(path("/health"))
            .respond_with(ResponseTemplate::new(200).set_body_json(health_json(
                &server, instance, kind, model, max_length, device, dtype,
            )))
            .mount(&server)
            .await;
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join(instance);
        std::fs::create_dir_all(&dir).unwrap();
        let addr = server.address();
        std::fs::write(dir.join("host"), format!("{}\n", addr.ip())).unwrap();
        std::fs::write(dir.join("port"), format!("{}\n", addr.port())).unwrap();
        (server, root)
    }

    /// A server another launcher started (a foreign instance key) is adopted
    /// by `(kind, model)` alone.
    #[tokio::test]
    async fn a_foreign_launchers_server_is_adoptable_by_kind_and_model() {
        let (_server, root) = published(
            "Qwen3-metal-f16-abc",
            "dense",
            "org/model",
            2048,
            "metal",
            "f16",
        )
        .await;
        let found = find_adoptable(
            root.path(),
            "org/model",
            EngineKind::Dense,
            2048,
            None,
            None,
        )
        .await;
        assert_eq!(found.unwrap().health.instance, "Qwen3-metal-f16-abc");
    }

    /// A server whose vectors would differ (another `max_length`, or a device
    /// or dtype the config pinned) is not adopted; `auto` is not a pin.
    #[tokio::test]
    async fn adoption_is_refused_when_the_workspace_would_produce_other_vectors() {
        // max_length differs.
        let (_s, root) = published("inst", "dense", "org/model", 512, "metal", "f16").await;
        assert!(find_adoptable(
            root.path(),
            "org/model",
            EngineKind::Dense,
            2048,
            None,
            None
        )
        .await
        .is_none());

        // A pinned device must match; an agreeing one is adopted.
        let (_s, root) = published("inst", "dense", "org/model", 2048, "ane", "f16").await;
        assert!(find_adoptable(
            root.path(),
            "org/model",
            EngineKind::Dense,
            2048,
            Some("metal"),
            None
        )
        .await
        .is_none());
        assert!(find_adoptable(
            root.path(),
            "org/model",
            EngineKind::Dense,
            2048,
            Some("ane"),
            None
        )
        .await
        .is_some());

        // A pinned dtype must match.
        assert!(find_adoptable(
            root.path(),
            "org/model",
            EngineKind::Dense,
            2048,
            None,
            Some("f32")
        )
        .await
        .is_none());
        // `auto` pins nothing.
        assert!(find_adoptable(
            root.path(),
            "org/model",
            EngineKind::Dense,
            2048,
            Some("auto"),
            Some("auto")
        )
        .await
        .is_some());

        // Another model or kind never matches.
        assert!(find_adoptable(
            root.path(),
            "org/other",
            EngineKind::Dense,
            2048,
            None,
            None
        )
        .await
        .is_none());
        assert!(find_adoptable(
            root.path(),
            "org/model",
            EngineKind::Sparse,
            2048,
            None,
            None
        )
        .await
        .is_none());
    }

    /// A directory left by a crashed server (nothing listens) is skipped.
    #[tokio::test]
    async fn a_stale_instance_directory_with_nothing_listening_is_skipped() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("dead");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("host"), "127.0.0.1\n").unwrap();
        std::fs::write(dir.join("port"), "1\n").unwrap();
        assert!(find_adoptable(
            root.path(),
            "org/model",
            EngineKind::Dense,
            2048,
            None,
            None
        )
        .await
        .is_none());
    }
}
