//! Node-API binding for MinSync's native embedder.
//!
//! Inference is `minsync::embedder::session::NativeSession`, the engine behind
//! `minsync sync` / `minsync query`: same weights and cache, same batch planner
//! and padded-token cap, same per-batch deadline. This crate only converts
//! types and maps errors. Work runs on tokio's blocking pool, never on the JS
//! thread.

use minsync::config::{Config, DEFAULT_EMBEDDER_ID};
use minsync::embedder::session::NativeSession;
use minsync::error::MinSyncError;
use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::watch;

/// Failure classes a host can tell apart. napi-rs only lets a native error
/// carry a `Status`, so the class travels as a `<CODE>: ` message prefix that
/// the JS wrapper turns into `error.code`.
#[derive(Clone, Copy)]
enum ErrorCode {
    /// A forward pass exceeded the per-batch deadline. The stuck thread cannot
    /// be cancelled, so the process must be restarted.
    Timeout,
    Config,
    Embedding,
    Aborted,
    Other,
}

impl ErrorCode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Timeout => "ERR_MINSYNC_TIMEOUT",
            Self::Config => "ERR_MINSYNC_CONFIG",
            Self::Embedding => "ERR_MINSYNC_EMBEDDING",
            Self::Aborted => "ERR_MINSYNC_ABORTED",
            Self::Other => "ERR_MINSYNC",
        }
    }

    fn status(self) -> Status {
        match self {
            Self::Config => Status::InvalidArg,
            Self::Aborted => Status::Cancelled,
            _ => Status::GenericFailure,
        }
    }

    fn error(self, message: impl std::fmt::Display) -> Error {
        Error::new(self.status(), format!("{}: {message}", self.as_str()))
    }
}

fn to_napi(error: MinSyncError) -> Error {
    let code = match &error {
        MinSyncError::Timeout(_) => ErrorCode::Timeout,
        MinSyncError::Config(_) => ErrorCode::Config,
        MinSyncError::Embedding(_) | MinSyncError::Io(_) => ErrorCode::Embedding,
        _ => ErrorCode::Other,
    };
    code.error(error)
}

/// Embedder settings, named after the `[embedder]` keys of `config.toml`
/// (camelCase). Omitted fields take the CLI's defaults.
#[napi(object)]
pub struct EmbedderOptions {
    /// `native:<hf repo id>`; defaults to `native:Qwen/Qwen3-Embedding-0.6B`.
    pub id: Option<String>,
    /// `auto` (default), `cpu`, `metal` or `cuda`.
    pub device: Option<String>,
    /// `auto` (default: f16 on Metal, f32 elsewhere), `f32`, `f16` or `bf16`.
    pub dtype: Option<String>,
    /// `auto` (default), `eager`, `tiled` or `fused`.
    pub attention: Option<String>,
    /// Prepended to every text passed to `embedQuery`.
    pub query_prefix: Option<String>,
    /// Prepended to every text passed to `embed`.
    pub passage_prefix: Option<String>,
    /// Most inputs per forward pass (default 64).
    pub batch_size: Option<u32>,
    /// Tokens kept per input (default 2048).
    pub max_length: Option<u32>,
    /// Cap on padded tokens per forward pass (default 4096).
    pub max_batch_tokens: Option<u32>,
    /// Default per-forward-pass deadline (default 60; at least 900 on CPU).
    pub timeout_seconds: Option<u32>,
    /// Per-forward-pass deadline; overrides `timeoutSeconds` and the CPU floor.
    pub per_batch_timeout_seconds: Option<u32>,
    /// Model cache directory. Without it `HF_HOME`, then
    /// `FASTEMBED_CACHE_DIR`, then `~/.cache/minsync/models`, as the CLI.
    pub model_cache_dir: Option<String>,
    /// Release the model after this many idle seconds; it reloads on the next
    /// call. Default: stay resident until `unload()` / `dispose()`.
    pub idle_unload_seconds: Option<u32>,
}

/// The embedding space the vectors belong to.
#[napi(object)]
pub struct EmbedderIdentity {
    /// As written in `config.toml`, e.g. `native:Qwen/Qwen3-Embedding-0.6B`.
    pub id: String,
    pub dimension: u32,
    /// Resolved precision: `f32`, `f16` or `bf16`.
    pub dtype: String,
    /// Resolved device: `cpu`, `metal` or `cuda`.
    pub device: String,
    pub max_length: u32,
}

/// Cancels the calls it was passed to. Used by the JS wrapper to implement
/// `AbortSignal`; not part of the public API.
#[napi]
pub struct AbortHandle {
    tx: Arc<watch::Sender<bool>>,
}

#[napi]
impl AbortHandle {
    #[napi(constructor)]
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            tx: Arc::new(watch::channel(false).0),
        }
    }

    /// Stop waiting for the call. Remaining forward passes are not started; a
    /// pass already running finishes in the background (it cannot be
    /// interrupted).
    #[napi]
    pub fn abort(&self) {
        self.tx.send_replace(true);
    }
}

async fn abortable<T>(
    abort: Option<&AbortHandle>,
    work: impl Future<Output = minsync::error::Result<T>>,
) -> Result<T> {
    let Some(abort) = abort else {
        return work.await.map_err(to_napi);
    };
    let mut aborted = abort.tx.subscribe();
    let aborted_error = || ErrorCode::Aborted.error("embedding call was aborted");
    if *aborted.borrow() {
        return Err(aborted_error());
    }
    tokio::select! {
        biased;
        _ = aborted.wait_for(|aborted| *aborted) => Err(aborted_error()),
        result = work => result.map_err(to_napi),
    }
}

fn seconds(value: u32) -> u64 {
    u64::from(value)
}

fn build_session(options: EmbedderOptions) -> minsync::error::Result<NativeSession> {
    let mut settings = Config::default_for("node").embedder;
    settings.id = options
        .id
        .unwrap_or_else(|| DEFAULT_EMBEDDER_ID.to_string());
    settings.device = options.device;
    settings.dtype = options.dtype;
    settings.attention = options.attention;
    settings.query_prefix = options.query_prefix;
    settings.passage_prefix = options.passage_prefix;
    settings.model_cache_dir = options.model_cache_dir;
    settings.max_length = options.max_length.map(|value| value as usize);
    settings.max_batch_tokens = options.max_batch_tokens.map(|value| value as usize);
    if let Some(value) = options.batch_size {
        settings.batch_size = value as usize;
    }
    if let Some(value) = options.timeout_seconds {
        settings.timeout_seconds = seconds(value);
    }
    settings.per_batch_timeout_seconds = options.per_batch_timeout_seconds.map(seconds);
    let idle = options
        .idle_unload_seconds
        .map(|value| Duration::from_secs(seconds(value)));
    NativeSession::new(&settings, idle)
}

/// In-process native embedder. The model loads on first use and stays
/// resident until `unload()`, `dispose()` or the idle timeout.
#[napi]
pub struct NativeEmbedder {
    session: Arc<NativeSession>,
}

#[napi]
impl NativeEmbedder {
    #[napi(constructor)]
    pub fn new(options: Option<EmbedderOptions>) -> Result<Self> {
        let options = options.unwrap_or(EmbedderOptions {
            id: None,
            device: None,
            dtype: None,
            attention: None,
            query_prefix: None,
            passage_prefix: None,
            batch_size: None,
            max_length: None,
            max_batch_tokens: None,
            timeout_seconds: None,
            per_batch_timeout_seconds: None,
            model_cache_dir: None,
            idle_unload_seconds: None,
        });
        let session = build_session(options).map_err(to_napi)?;
        Ok(Self {
            session: Arc::new(session),
        })
    }

    /// Model id, dimension, dtype and device. Loads the model if it is not
    /// resident, since those are only known once it is.
    #[napi]
    pub async fn identity(&self) -> Result<EmbedderIdentity> {
        let identity = self.session.identity().await.map_err(to_napi)?;
        Ok(EmbedderIdentity {
            id: identity.id,
            dimension: identity.dimension as u32,
            dtype: identity.dtype.to_string(),
            device: identity.device.to_string(),
            max_length: identity.max_length as u32,
        })
    }

    /// Embed passages (`passagePrefix` applied). Unit-length vectors, in input
    /// order.
    #[napi]
    pub async fn embed(
        &self,
        texts: Vec<String>,
        abort: Option<&AbortHandle>,
    ) -> Result<Vec<Float32Array>> {
        let vectors = abortable(abort, self.session.embed(&texts)).await?;
        Ok(vectors.into_iter().map(Float32Array::new).collect())
    }

    /// Embed a search query (`queryPrefix` applied).
    #[napi]
    pub async fn embed_query(
        &self,
        text: String,
        abort: Option<&AbortHandle>,
    ) -> Result<Float32Array> {
        let vector = abortable(abort, self.session.embed_query(&text)).await?;
        Ok(Float32Array::new(vector))
    }

    /// True while the model weights are resident.
    #[napi]
    pub fn is_loaded(&self) -> bool {
        self.session.is_loaded()
    }

    /// Release the model weights; the next call reloads them. Resolves to
    /// whether a model was resident.
    #[napi]
    pub async fn unload(&self) -> bool {
        self.session.unload().await
    }

    /// Release the model and reject every later call.
    #[napi]
    pub async fn dispose(&self) {
        self.session.dispose().await;
    }
}
