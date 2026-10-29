//! `minsync serve`: one resident native embedder shared over loopback HTTP.
//!
//! Protocols:
//! * TEI: `POST /embed` `{ "inputs", "truncate"? }` -> `[[f32]]`
//! * OpenAI: `POST /v1/embeddings` `{ "model", "input" }` -> `{ "data", ... }`
//! * `GET /health`: model id, dimension, dtype, device, pid, port, `kind`.
//!
//! A server is `dense` (the routes above) or `sparse` (`minsync serve
//! --sparse`: `POST /embed_sparse`, TEI's learned-sparse protocol); a sparse
//! server answers 404 on the dense routes and vice versa.
//!
//! Both embedding routes accept an `input_type: "query" | "passage"`
//! extension that applies the configured prefix; without it the text is
//! embedded as sent.
//!
//! The server is unauthenticated, so it only binds loopback addresses. A
//! state directory per model instance holds a `lock` (held for the
//! server's lifetime) and `pid` / `host` / `port` files; a second `start`
//! that finds the lock taken adopts the running server instead of loading
//! another copy of the model.

use crate::error::{MinSyncError, Result};
use axum::body::Bytes;
use axum::extract::rejection::BytesRejection;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::future::{Future, IntoFuture};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::time::Instant;

#[cfg(test)]
mod tests;

/// Default for `--max-inputs`: inputs accepted in one request. The batch
/// planner splits a request into forward passes, so this only bounds how
/// long one request can occupy the model.
pub const DEFAULT_MAX_INPUTS: usize = 512;

/// Default for `--max-request-bytes` (32 MiB).
pub const DEFAULT_MAX_REQUEST_BYTES: usize = 32 * 1024 * 1024;

/// Default for `--idle-timeout` in seconds.
pub const DEFAULT_IDLE_TIMEOUT_SECONDS: u64 = 300;

/// How long a second `start` waits for a server that holds the lock (and
/// may still be loading or downloading the model) to answer `/health`.
const ADOPT_WAIT: Duration = Duration::from_secs(600);

/// How long a stopping server waits for in-flight requests.
const DRAIN_WAIT: Duration = Duration::from_secs(30);

/// What the server needs from an embedding engine.
#[async_trait::async_trait]
pub trait ServeEngine: Send + Sync {
    /// Load the model (once) and describe it.
    async fn load(&self) -> Result<EngineInfo>;

    /// Tokens per input, as the model would see them before truncation.
    async fn token_counts(&self, texts: &[String]) -> Result<Vec<usize>>;

    /// Embed `texts` exactly as given (no prefixes added).
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>>;
}

/// What a sparse (SPLADE) server needs from an encoder.
#[async_trait::async_trait]
pub trait SparseServeEngine: Send + Sync {
    /// Load the model (once) and describe it. `dimension` is the number of
    /// terms in the output space.
    async fn load(&self) -> Result<EngineInfo>;

    /// Tokens per input, as the model would see them before truncation.
    async fn token_counts(&self, texts: &[String]) -> Result<Vec<usize>>;

    /// Encode `texts` exactly as given (no prefixes added).
    async fn encode(&self, texts: Vec<String>) -> Result<Vec<crate::sparse::SparseVector>>;
}

/// The engine behind one server.
#[derive(Clone)]
enum Engine {
    Dense(Arc<dyn ServeEngine>),
    Sparse(Arc<dyn SparseServeEngine>),
}

impl Engine {
    fn kind(&self) -> &'static str {
        match self {
            Self::Dense(_) => "dense",
            Self::Sparse(_) => "sparse",
        }
    }

    async fn load(&self) -> Result<EngineInfo> {
        match self {
            Self::Dense(engine) => engine.load().await,
            Self::Sparse(engine) => engine.load().await,
        }
    }

    async fn token_counts(&self, texts: &[String]) -> Result<Vec<usize>> {
        match self {
            Self::Dense(engine) => engine.token_counts(texts).await,
            Self::Sparse(engine) => engine.token_counts(texts).await,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineInfo {
    pub model_id: String,
    pub dimension: usize,
    pub dtype: String,
    pub device: String,
    pub attention: String,
    pub max_length: usize,
}

#[derive(Debug, Clone)]
pub struct ServeOptions {
    pub host: IpAddr,
    pub port: u16,
    /// `None` serves until stopped.
    pub idle_timeout: Option<Duration>,
    pub max_inputs: usize,
    pub max_request_bytes: usize,
    /// Directory for `lock`, `pid`, `host` and `port`.
    pub state_dir: PathBuf,
    /// Identity of the model configuration; see [`instance_key`].
    pub instance: String,
    pub query_prefix: Option<String>,
    pub passage_prefix: Option<String>,
}

fn dense_kind() -> String {
    "dense".to_string()
}

/// `GET /health` body; also what an adopting process reports.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Health {
    pub status: String,
    /// `dense` or `sparse`. Absent from servers older than sparse support,
    /// which are all dense.
    #[serde(default = "dense_kind")]
    pub kind: String,
    pub model_id: String,
    pub dimension: usize,
    pub dtype: String,
    pub device: String,
    pub attention: String,
    pub max_length: usize,
    pub max_inputs: usize,
    pub instance: String,
    pub pid: u32,
    pub host: String,
    pub port: u16,
    pub url: String,
    pub version: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// SIGTERM / Ctrl-C (or the caller's stop future).
    Signal,
    /// No embedding request for the idle timeout.
    Idle,
    /// A forward pass exceeded its deadline; its thread cannot be
    /// cancelled, so the process must exit to free the model.
    Wedged,
}

impl StopReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Signal => "signal",
            Self::Idle => "idle",
            Self::Wedged => "wedged",
        }
    }

    /// Process exit code for this stop: a wedged model exits like an
    /// embedder timeout ([`MinSyncError::Timeout`]).
    pub fn exit_code(self) -> i32 {
        match self {
            Self::Signal | Self::Idle => 0,
            Self::Wedged => MinSyncError::Timeout(String::new()).exit_code(),
        }
    }
}

pub enum Started {
    /// This process loaded the model and owns the listener.
    Serving(Server),
    /// Another process already serves this instance.
    Adopted(Health),
}

pub struct Server {
    listener: tokio::net::TcpListener,
    state: Arc<AppState>,
    files: StateFiles,
    idle_timeout: Option<Duration>,
    max_request_bytes: usize,
}

struct AppState {
    engine: Engine,
    info: EngineInfo,
    health: Health,
    max_inputs: usize,
    query_prefix: Option<String>,
    passage_prefix: Option<String>,
    in_flight: AtomicUsize,
    last_activity: Mutex<Instant>,
    wedged: tokio::sync::Notify,
}

/// Lock plus the discovery files; removed when the server stops.
struct StateFiles {
    dir: PathBuf,
    _lock: LockFile,
}

struct LockFile(std::fs::File);

impl Drop for LockFile {
    fn drop(&mut self) {
        let _ = fs2::FileExt::unlock(&self.0);
    }
}

impl Drop for StateFiles {
    fn drop(&mut self) {
        self.remove();
    }
}

impl StateFiles {
    fn publish(&self, health: &Health) -> Result<()> {
        write_atomic(&self.dir.join("pid"), &health.pid.to_string())?;
        write_atomic(&self.dir.join("host"), &health.host)?;
        // Last: a reader that sees `port` sees the other files too.
        write_atomic(&self.dir.join("port"), &health.port.to_string())
    }

    fn remove(&self) {
        for name in ["port", "host", "pid"] {
            let _ = std::fs::remove_file(self.dir.join(name));
        }
    }
}

fn write_atomic(path: &Path, content: &str) -> Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let mut tmp = tempfile::NamedTempFile::new_in(parent)?;
    std::io::Write::write_all(&mut tmp, format!("{content}\n").as_bytes())?;
    tmp.persist(path)
        .map_err(|error| MinSyncError::Io(error.error))?;
    Ok(())
}

/// Accept only loopback addresses: the server has no authentication.
pub fn parse_loopback_host(value: &str) -> Result<IpAddr> {
    let ip = if value.eq_ignore_ascii_case("localhost") {
        IpAddr::V4(Ipv4Addr::LOCALHOST)
    } else {
        value.parse::<IpAddr>().map_err(|_| {
            MinSyncError::Config(format!(
                "serve host '{value}' is not an IP address; use 127.0.0.1 or ::1"
            ))
        })?
    };
    if !ip.is_loopback() {
        return Err(MinSyncError::Config(format!(
            "serve host {ip} is not a loopback address; the server has no authentication, \
             so it only binds 127.0.0.1 or ::1"
        )));
    }
    Ok(ip)
}

/// Identity of a served model: everything that changes the vectors the
/// native engine produces. Prefixes and batching knobs are excluded:
/// callers send prefixed text, and batching does not change results beyond
/// left-padding noise.
pub fn instance_key(settings: &crate::config::EmbedderConfig) -> Result<String> {
    let Some(model) = settings.id.strip_prefix("native:") else {
        return Err(MinSyncError::Config(format!(
            "minsync serve only serves native: embedders, got '{}'",
            settings.id
        )));
    };
    let auto = |value: &Option<String>| value.clone().unwrap_or_else(|| "auto".to_string());
    let material = json!({
        "model": model,
        "device": auto(&settings.device),
        "dtype": auto(&settings.dtype),
        "attention": auto(&settings.attention),
        "max_length": settings
            .max_length
            .unwrap_or(crate::embedder::native::DEFAULT_MAX_LENGTH),
    });
    let digest = hex::encode(Sha256::digest(material.to_string()));
    let name: String = model
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '-' {
                c
            } else {
                '-'
            }
        })
        .collect();
    Ok(format!("{name}-{}", &digest[..12]))
}

/// Identity of a served sparse model: everything that changes the vectors the
/// native sparse engine produces. The `-sparse-` infix keeps its state
/// directory apart from a dense model of the same name.
pub fn sparse_instance_key(settings: &crate::config::SparseConfig) -> Result<String> {
    let Some(model) = settings.id.strip_prefix("native:") else {
        return Err(MinSyncError::Config(format!(
            "minsync serve --sparse only serves native: encoders, got '{}'",
            settings.id
        )));
    };
    let auto = |value: &Option<String>| value.clone().unwrap_or_else(|| "auto".to_string());
    let material = json!({
        "kind": "sparse",
        "model": model,
        "device": auto(&settings.device),
        "dtype": auto(&settings.dtype),
        "max_length": settings
            .max_length
            .unwrap_or(crate::config::DEFAULT_SPARSE_MAX_LENGTH),
    });
    let digest = hex::encode(Sha256::digest(material.to_string()));
    let name: String = model
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '-' {
                c
            } else {
                '-'
            }
        })
        .collect();
    Ok(format!("{name}-sparse-{}", &digest[..12]))
}

/// Default state directory for an instance:
/// `~/.cache/minsync/serve/<instance>` (`%LOCALAPPDATA%\minsync\serve` on
/// Windows).
pub fn default_state_dir(instance: &str) -> PathBuf {
    crate::embedder::native::minsync_cache_root()
        .join("serve")
        .join(instance)
}

/// Read `port`/`host` in `state_dir` and return the server's health when it
/// answers as `instance`.
pub async fn probe(state_dir: &Path, instance: &str) -> Option<Health> {
    let port: u16 = std::fs::read_to_string(state_dir.join("port"))
        .ok()?
        .trim()
        .parse()
        .ok()?;
    let host = std::fs::read_to_string(state_dir.join("host"))
        .map(|h| h.trim().to_string())
        .unwrap_or_else(|_| "127.0.0.1".to_string());
    let ip = parse_loopback_host(&host).ok()?;
    let url = base_url(SocketAddr::new(ip, port));
    let health: Health = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .no_proxy()
        .build()
        .ok()?
        .get(format!("{url}/health"))
        .send()
        .await
        .ok()?
        .json()
        .await
        .ok()?;
    (health.instance == instance && health.status == "ok").then_some(health)
}

fn base_url(addr: SocketAddr) -> String {
    format!("http://{addr}")
}

/// Become the server for `options.instance`, or adopt the one that already
/// is. Loads the model only in the first case.
pub async fn start(engine: Arc<dyn ServeEngine>, options: ServeOptions) -> Result<Started> {
    start_engine(Engine::Dense(engine), options).await
}

/// Like [`start`], for a sparse (SPLADE) engine: serves `POST /embed_sparse`.
pub async fn start_sparse(
    engine: Arc<dyn SparseServeEngine>,
    options: ServeOptions,
) -> Result<Started> {
    start_engine(Engine::Sparse(engine), options).await
}

async fn start_engine(engine: Engine, options: ServeOptions) -> Result<Started> {
    std::fs::create_dir_all(&options.state_dir)?;
    let lock = acquire_or_adopt(&options).await?;
    let lock = match lock {
        Ok(lock) => lock,
        Err(health) => return Ok(Started::Adopted(health)),
    };
    let files = StateFiles {
        dir: options.state_dir.clone(),
        _lock: lock,
    };
    // Whatever a dead owner left behind is stale now.
    files.remove();

    let info = engine.load().await?;
    let listener = tokio::net::TcpListener::bind(SocketAddr::new(options.host, options.port))
        .await
        .map_err(|error| {
            MinSyncError::Other(format!(
                "cannot bind {}:{}: {error}",
                options.host, options.port
            ))
        })?;
    let addr = listener.local_addr()?;
    let health = Health {
        status: "ok".to_string(),
        kind: engine.kind().to_string(),
        model_id: info.model_id.clone(),
        dimension: info.dimension,
        dtype: info.dtype.clone(),
        device: info.device.clone(),
        attention: info.attention.clone(),
        max_length: info.max_length,
        max_inputs: options.max_inputs,
        instance: options.instance.clone(),
        pid: std::process::id(),
        host: addr.ip().to_string(),
        port: addr.port(),
        url: base_url(addr),
        version: env!("CARGO_PKG_VERSION").to_string(),
    };
    files.publish(&health)?;
    let state = Arc::new(AppState {
        engine,
        info,
        health,
        max_inputs: options.max_inputs.max(1),
        query_prefix: options.query_prefix,
        passage_prefix: options.passage_prefix,
        in_flight: AtomicUsize::new(0),
        last_activity: Mutex::new(Instant::now()),
        wedged: tokio::sync::Notify::new(),
    });
    Ok(Started::Serving(Server {
        listener,
        state,
        files,
        idle_timeout: options.idle_timeout,
        max_request_bytes: options.max_request_bytes,
    }))
}

/// `Ok(lock)` when this process should serve, `Err(health)` to adopt.
async fn acquire_or_adopt(options: &ServeOptions) -> Result<std::result::Result<LockFile, Health>> {
    use fs2::FileExt;
    let path = options.state_dir.join("lock");
    let give_up = Instant::now() + ADOPT_WAIT;
    loop {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&path)?;
        if file.try_lock_exclusive().is_ok() {
            return Ok(Ok(LockFile(file)));
        }
        if let Some(health) = probe(&options.state_dir, &options.instance).await {
            return Ok(Err(health));
        }
        if Instant::now() >= give_up {
            return Err(MinSyncError::Other(format!(
                "another minsync serve holds {} but did not answer /health within {}s",
                path.display(),
                ADOPT_WAIT.as_secs()
            )));
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

impl Server {
    pub fn health(&self) -> &Health {
        &self.state.health
    }

    /// Serve until `stop` resolves, the idle timeout passes with no
    /// embedding request, or a forward pass wedges. Removes the discovery
    /// files and releases the lock on return.
    pub async fn run_until<F>(self, stop: F) -> Result<StopReason>
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let Server {
            listener,
            state,
            files,
            idle_timeout,
            max_request_bytes,
        } = self;
        let router = router(Arc::clone(&state), max_request_bytes);
        let (reason_tx, reason_rx) = tokio::sync::oneshot::channel();
        let watched = Arc::clone(&state);
        let signal = async move {
            let idle = async {
                match idle_timeout {
                    Some(timeout) => wait_idle(&watched, timeout).await,
                    None => std::future::pending().await,
                }
            };
            let reason = tokio::select! {
                () = stop => StopReason::Signal,
                () = idle => StopReason::Idle,
                () = watched.wedged.notified() => StopReason::Wedged,
            };
            let _ = reason_tx.send(reason);
        };
        let mut serve = Box::pin(
            axum::serve(listener, router)
                .with_graceful_shutdown(signal)
                .into_future(),
        );
        let reason = tokio::select! {
            biased;
            reason = reason_rx => {
                let reason = reason.unwrap_or(StopReason::Signal);
                // Let in-flight requests finish, but never hang on one.
                let _ = tokio::time::timeout(DRAIN_WAIT, serve).await;
                reason
            }
            served = &mut serve => {
                served?;
                StopReason::Signal
            }
        };
        drop(files);
        Ok(reason)
    }
}

/// Resolve once no embedding request has run for `timeout`.
async fn wait_idle(state: &AppState, timeout: Duration) {
    loop {
        let last = *state.last_activity.lock().expect("activity lock");
        tokio::time::sleep_until(last + timeout).await;
        if state.in_flight.load(Ordering::SeqCst) > 0 {
            // A request outlasting the timeout re-arms it when it ends.
            tokio::time::sleep(timeout).await;
            continue;
        }
        let last = *state.last_activity.lock().expect("activity lock");
        if last.elapsed() >= timeout {
            return;
        }
    }
}

/// Marks an embedding request in flight; the idle clock restarts when it ends.
struct Busy<'a>(&'a AppState);

impl AppState {
    fn busy(&self) -> Busy<'_> {
        self.in_flight.fetch_add(1, Ordering::SeqCst);
        self.touch();
        Busy(self)
    }

    fn touch(&self) {
        *self.last_activity.lock().expect("activity lock") = Instant::now();
    }

    fn prefix(&self, input_type: Option<InputType>) -> Option<&str> {
        match input_type? {
            InputType::Query => self.query_prefix.as_deref(),
            InputType::Passage => self.passage_prefix.as_deref(),
        }
    }

    fn serves_model(&self, requested: &str) -> bool {
        let requested = requested.strip_prefix("native:").unwrap_or(requested);
        requested.eq_ignore_ascii_case(&self.info.model_id)
    }
}

impl Drop for Busy<'_> {
    fn drop(&mut self) {
        self.0.touch();
        self.0.in_flight.fetch_sub(1, Ordering::SeqCst);
    }
}

fn router(state: Arc<AppState>, max_request_bytes: usize) -> Router {
    let routes = Router::new().route("/health", get(health));
    let routes = match state.engine {
        Engine::Dense(_) => routes
            .route("/embed", post(tei_embed))
            .route("/v1/embeddings", post(openai_embeddings)),
        Engine::Sparse(_) => routes.route("/embed_sparse", post(tei_embed_sparse)),
    };
    routes
        .layer(DefaultBodyLimit::max(max_request_bytes))
        .with_state(BodyLimited {
            state,
            max_request_bytes,
        })
}

#[derive(Clone)]
struct BodyLimited {
    state: Arc<AppState>,
    max_request_bytes: usize,
}

async fn health(State(app): State<BodyLimited>) -> Json<Health> {
    Json(app.state.health.clone())
}

/// Query/passage extension to both protocols.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
enum InputType {
    Query,
    Passage,
}

#[derive(Deserialize)]
struct TeiRequest {
    inputs: serde_json::Value,
    #[serde(default)]
    truncate: Option<bool>,
    #[serde(default)]
    normalize: Option<bool>,
    #[serde(default)]
    truncation_direction: Option<String>,
    #[serde(default)]
    prompt_name: Option<String>,
    #[serde(default)]
    input_type: Option<InputType>,
}

#[derive(Deserialize)]
struct OpenAiRequest {
    #[serde(default)]
    model: Option<String>,
    input: serde_json::Value,
    #[serde(default)]
    encoding_format: Option<String>,
    #[serde(default)]
    dimensions: Option<usize>,
    #[serde(default)]
    input_type: Option<InputType>,
}

#[derive(Clone, Copy)]
enum Protocol {
    Tei,
    OpenAi,
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
    code: Option<&'static str>,
}

impl ApiError {
    fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
            code: None,
        }
    }

    fn with_code(mut self, code: &'static str) -> Self {
        self.code = Some(code);
        self
    }

    fn render(self, protocol: Protocol) -> Response {
        match protocol {
            Protocol::Tei => {
                let error_type = match self.status {
                    StatusCode::SERVICE_UNAVAILABLE => "Unhealthy",
                    status if status.is_server_error() => "Backend",
                    _ => "Validation",
                };
                (
                    self.status,
                    Json(json!({ "error": self.message, "error_type": error_type })),
                )
                    .into_response()
            }
            Protocol::OpenAi => {
                // OpenAI reports every malformed request as 400.
                let status = if self.status == StatusCode::UNPROCESSABLE_ENTITY {
                    StatusCode::BAD_REQUEST
                } else {
                    self.status
                };
                let kind = if status.is_server_error() {
                    "server_error"
                } else {
                    "invalid_request_error"
                };
                (
                    status,
                    Json(json!({
                        "error": {
                            "message": self.message,
                            "type": kind,
                            "param": null,
                            "code": self.code,
                        }
                    })),
                )
                    .into_response()
            }
        }
    }
}

fn parse_body<T: serde::de::DeserializeOwned>(
    body: std::result::Result<Bytes, BytesRejection>,
    max_request_bytes: usize,
) -> std::result::Result<T, ApiError> {
    let bytes = body.map_err(|rejection| {
        if rejection.status() == StatusCode::PAYLOAD_TOO_LARGE {
            ApiError::new(
                StatusCode::PAYLOAD_TOO_LARGE,
                format!(
                    "request body exceeds the {max_request_bytes}-byte limit (--max-request-bytes)"
                ),
            )
        } else {
            ApiError::new(rejection.status(), rejection.body_text())
        }
    })?;
    serde_json::from_slice(&bytes).map_err(|error| {
        let status = match error.classify() {
            serde_json::error::Category::Data => StatusCode::UNPROCESSABLE_ENTITY,
            _ => StatusCode::BAD_REQUEST,
        };
        ApiError::new(status, format!("invalid request body: {error}"))
    })
}

/// A string or an array of strings. Token-id arrays are not supported.
fn input_texts(
    value: serde_json::Value,
    field: &str,
) -> std::result::Result<Vec<String>, ApiError> {
    let invalid = || {
        ApiError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            format!("`{field}` must be a string or an array of strings (token arrays are not supported)"),
        )
    };
    match value {
        serde_json::Value::String(text) => Ok(vec![text]),
        serde_json::Value::Array(items) => items
            .into_iter()
            .map(|item| match item {
                serde_json::Value::String(text) => Ok(text),
                _ => Err(invalid()),
            })
            .collect(),
        _ => Err(invalid()),
    }
}

/// Qwen3 uses byte-level BPE: an input never has more tokens than UTF-8
/// bytes plus template tokens, so only inputs this close to the context
/// need tokenizing to enforce it.
const TOKEN_SLACK: usize = 8;

struct Embedded {
    vectors: Vec<Vec<f32>>,
    /// Exact prompt tokens, when counted.
    tokens: usize,
}

/// Inputs that passed validation, with the exact prompt tokens when counted.
struct Checked {
    texts: Vec<String>,
    tokens: usize,
}

/// Validate, prefix and check the context. `count_all` tokenizes every input
/// (OpenAI usage); otherwise only inputs that may exceed the context.
async fn checked_inputs_counting(
    state: &AppState,
    texts: Vec<String>,
    input_type: Option<InputType>,
    truncate: bool,
    count_all: bool,
    protocol: Protocol,
) -> std::result::Result<Checked, ApiError> {
    if texts.is_empty() {
        return Err(ApiError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "`inputs` cannot be empty",
        ));
    }
    if texts.len() > state.max_inputs {
        return Err(ApiError::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            format!(
                "batch size {} > maximum allowed batch size {}",
                texts.len(),
                state.max_inputs
            ),
        ));
    }
    let texts: Vec<String> = match state.prefix(input_type) {
        Some(prefix) => texts.into_iter().map(|t| format!("{prefix}{t}")).collect(),
        None => texts,
    };
    let max_length = state.info.max_length;
    let (indices, candidates): (Vec<usize>, Vec<String>) = texts
        .iter()
        .enumerate()
        .filter(|(_, text)| count_all || text.len() + TOKEN_SLACK > max_length)
        .map(|(index, text)| (index, text.clone()))
        .unzip();
    let counts = if candidates.is_empty() {
        Vec::new()
    } else {
        state
            .engine
            .token_counts(&candidates)
            .await
            .map_err(|error| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))?
    };
    if !truncate {
        if let Some((index, tokens)) = indices
            .iter()
            .zip(&counts)
            .find(|(_, &tokens)| tokens > max_length)
        {
            let message = match protocol {
                Protocol::Tei => format!(
                    "input {index}: `inputs` must have less than {max_length} tokens. Given: {tokens}"
                ),
                Protocol::OpenAi => format!(
                    "input {index}: this model's maximum context length is {max_length} tokens, \
                     however you requested {tokens} tokens"
                ),
            };
            return Err(ApiError::new(StatusCode::PAYLOAD_TOO_LARGE, message)
                .with_code("context_length_exceeded"));
        }
    }
    let tokens = counts.iter().map(|&t| t.min(max_length)).sum();
    Ok(Checked { texts, tokens })
}

async fn checked_inputs(
    state: &AppState,
    texts: Vec<String>,
    input_type: Option<InputType>,
    truncate: bool,
    protocol: Protocol,
) -> std::result::Result<Vec<String>, ApiError> {
    Ok(
        checked_inputs_counting(state, texts, input_type, truncate, false, protocol)
            .await?
            .texts,
    )
}

/// Validate, prefix, check the context, embed (dense servers only).
async fn embed_inputs(
    state: &AppState,
    texts: Vec<String>,
    input_type: Option<InputType>,
    truncate: bool,
    count_all: bool,
    protocol: Protocol,
) -> std::result::Result<Embedded, ApiError> {
    let Engine::Dense(engine) = &state.engine else {
        return Err(ApiError::new(
            StatusCode::NOT_FOUND,
            "this server serves sparse vectors; use /embed_sparse",
        ));
    };
    let Checked { texts, tokens } =
        checked_inputs_counting(state, texts, input_type, truncate, count_all, protocol).await?;
    let vectors = engine
        .embed(texts)
        .await
        .map_err(|error| engine_error(state, error))?;
    Ok(Embedded { vectors, tokens })
}

async fn tei_embed(
    State(app): State<BodyLimited>,
    body: std::result::Result<Bytes, BytesRejection>,
) -> Response {
    let state = &app.state;
    let _busy = state.busy();
    let result = async {
        let request: TeiRequest = parse_body(body, app.max_request_bytes)?;
        let unsupported =
            |message: &str| Err(ApiError::new(StatusCode::UNPROCESSABLE_ENTITY, message));
        if request.normalize == Some(false) {
            return unsupported(
                "`normalize: false` is not supported: native embeddings are always unit length",
            );
        }
        if request.prompt_name.is_some() {
            return unsupported(
                "`prompt_name` is not supported: send prefixed text or `input_type`",
            );
        }
        if request
            .truncation_direction
            .as_deref()
            .is_some_and(|d| !d.eq_ignore_ascii_case("right"))
        {
            return unsupported("only `truncation_direction: \"right\"` is supported");
        }
        let texts = input_texts(request.inputs, "inputs")?;
        let truncate = request.truncate.unwrap_or(false);
        embed_inputs(
            state,
            texts,
            request.input_type,
            truncate,
            false,
            Protocol::Tei,
        )
        .await
    }
    .await;
    match result {
        Ok(embedded) => Json(embedded.vectors).into_response(),
        Err(error) => error.render(Protocol::Tei),
    }
}

async fn openai_embeddings(
    State(app): State<BodyLimited>,
    body: std::result::Result<Bytes, BytesRejection>,
) -> Response {
    let state = &app.state;
    let _busy = state.busy();
    let result = async {
        let request: OpenAiRequest = parse_body(body, app.max_request_bytes)?;
        if let Some(model) = &request.model {
            if !state.serves_model(model) {
                return Err(ApiError::new(
                    StatusCode::NOT_FOUND,
                    format!(
                        "the model `{model}` is not served here; this server serves `{}`",
                        state.info.model_id
                    ),
                )
                .with_code("model_not_found"));
            }
        }
        if let Some(dimensions) = request.dimensions {
            if dimensions != state.info.dimension {
                return Err(ApiError::new(
                    StatusCode::BAD_REQUEST,
                    format!(
                        "`dimensions` {dimensions} is not supported; this model returns {}",
                        state.info.dimension
                    ),
                ));
            }
        }
        let base64 = match request.encoding_format.as_deref() {
            None | Some("float") => false,
            Some("base64") => true,
            Some(other) => {
                return Err(ApiError::new(
                    StatusCode::BAD_REQUEST,
                    format!("`encoding_format` must be \"float\" or \"base64\", got \"{other}\""),
                ))
            }
        };
        let texts = input_texts(request.input, "input")?;
        let embedded = embed_inputs(
            state,
            texts,
            request.input_type,
            false,
            true,
            Protocol::OpenAi,
        )
        .await?;
        Ok((embedded, base64))
    }
    .await;
    match result {
        Ok((embedded, base64)) => {
            let data: Vec<serde_json::Value> = embedded
                .vectors
                .into_iter()
                .enumerate()
                .map(|(index, vector)| {
                    let embedding = if base64 {
                        let bytes: Vec<u8> = vector.iter().flat_map(|v| v.to_le_bytes()).collect();
                        json!(base64::engine::general_purpose::STANDARD.encode(bytes))
                    } else {
                        json!(vector)
                    };
                    json!({ "object": "embedding", "index": index, "embedding": embedding })
                })
                .collect();
            Json(json!({
                "object": "list",
                "data": data,
                "model": state.info.model_id,
                "usage": { "prompt_tokens": embedded.tokens, "total_tokens": embedded.tokens },
            }))
            .into_response()
        }
        Err(error) => error.render(Protocol::OpenAi),
    }
}

/// An engine failure as an HTTP error. A deadline means the model's thread is
/// stuck for good, so the server also schedules its own shutdown.
fn engine_error(state: &AppState, error: MinSyncError) -> ApiError {
    match error {
        MinSyncError::Timeout(_) => {
            state.wedged.notify_one();
            ApiError::new(
                StatusCode::SERVICE_UNAVAILABLE,
                format!("{error}; the server is shutting down"),
            )
        }
        other => ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, other.to_string()),
    }
}

#[derive(Serialize)]
struct SparseValue {
    index: u32,
    value: f32,
}

/// `POST /embed_sparse`: TEI's learned-sparse protocol.
async fn tei_embed_sparse(
    State(app): State<BodyLimited>,
    body: std::result::Result<Bytes, BytesRejection>,
) -> Response {
    let state = &app.state;
    let _busy = state.busy();
    let result = async {
        let request: TeiRequest = parse_body(body, app.max_request_bytes)?;
        if request.prompt_name.is_some() {
            return Err(ApiError::new(
                StatusCode::UNPROCESSABLE_ENTITY,
                "`prompt_name` is not supported: send prefixed text or `input_type`",
            ));
        }
        if request
            .truncation_direction
            .as_deref()
            .is_some_and(|d| !d.eq_ignore_ascii_case("right"))
        {
            return Err(ApiError::new(
                StatusCode::UNPROCESSABLE_ENTITY,
                "only `truncation_direction: \"right\"` is supported",
            ));
        }
        let texts = input_texts(request.inputs, "inputs")?;
        let truncate = request.truncate.unwrap_or(false);
        let Engine::Sparse(engine) = &state.engine else {
            return Err(ApiError::new(
                StatusCode::NOT_FOUND,
                "this server serves dense vectors; use /embed",
            ));
        };
        let texts =
            checked_inputs(state, texts, request.input_type, truncate, Protocol::Tei).await?;
        engine
            .encode(texts)
            .await
            .map_err(|error| engine_error(state, error))
    }
    .await;
    match result {
        Ok(vectors) => Json(
            vectors
                .iter()
                .map(|vector| {
                    vector
                        .indices()
                        .iter()
                        .zip(vector.values())
                        .map(|(&index, &value)| SparseValue { index, value })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>(),
        )
        .into_response(),
        Err(error) => error.render(Protocol::Tei),
    }
}
