//! Multimodal (text + image) embedder backed by `inl-vision` (EmbeddingGemma 2)
//! and `inl serve --multimodal` / `minsync serve --multimodal`.
//!
//! Texts and images map into one shared vector space (768 dimensions for
//! `google/embeddinggemma-2`). Text inputs still get the configured query /
//! passage prefixes; standalone image inputs are sent as raw encoded bytes
//! (PNG / JPEG / WebP) with no text prefix so the vision tower encodes the
//! pixels directly.

use crate::config::EmbedderConfig;
use crate::embedder::{Embedder, ImageInput};
use crate::error::{MinSyncError, Result};
use async_trait::async_trait;
use base64::Engine as _;
use inl_core::attention::AttentionRequest;
use inl_core::device::{DeviceSpec, DtypeSpec};
use inl_core::{MediaBytes, Modality, MultimodalEngine, MultimodalInput};
use inl_vision::{MultimodalEmbedder, MultimodalOptions};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::OnceCell;

/// Default context length for `inl-vision`'s EmbeddingGemma 2.
pub(crate) const DEFAULT_MULTIMODAL_MAX_LENGTH: usize = 8192;

/// Strip `native-multimodal:` (or `native:`) from an embedder id and reject an
/// empty model name.
pub(crate) fn parse_multimodal_model_id(id: &str) -> Result<&str> {
    let model = id
        .strip_prefix("native-multimodal:")
        .or_else(|| id.strip_prefix("native:"))
        .unwrap_or(id);
    if model.is_empty() {
        return Err(MinSyncError::Config(
            "multimodal embedder id is missing a model name".to_string(),
        ));
    }
    Ok(model)
}

/// Build `inl-vision` options from `[embedder]`, loading only the text and
/// image towers so audio/video weights stay untouched.
pub fn multimodal_options(settings: &EmbedderConfig) -> Result<MultimodalOptions> {
    let defaults = MultimodalOptions::default();
    if settings.batch_size == 0 {
        return Err(MinSyncError::Config(
            "embedder batch_size must be positive".to_string(),
        ));
    }
    if settings.timeout_seconds == 0 {
        return Err(MinSyncError::Config(
            "embedder timeout_seconds must be positive".to_string(),
        ));
    }
    Ok(MultimodalOptions {
        id: parse_multimodal_model_id(&settings.id)?.to_string(),
        device: DeviceSpec::parse(settings.device.as_deref())?,
        dtype: DtypeSpec::parse(settings.dtype.as_deref())?,
        attention: AttentionRequest::parse(settings.attention.as_deref())?,
        max_length: settings
            .max_length
            .unwrap_or(DEFAULT_MULTIMODAL_MAX_LENGTH),
        modalities: Some(vec![Modality::Text, Modality::Image]),
        batch_size: settings.batch_size.min(defaults.batch_size),
        max_batch_tokens: settings
            .max_batch_tokens
            .unwrap_or(defaults.max_batch_tokens),
        timeout_seconds: settings
            .per_batch_timeout_seconds
            .unwrap_or(settings.timeout_seconds)
            .max(defaults.timeout_seconds),
        cache_dir: Some(super::native::model_cache_dir(
            settings.model_cache_dir.as_deref(),
        )),
        ..defaults
    })
}

/// In-process multimodal embedder over `inl-vision` (`MultimodalEmbedder`).
pub struct NativeMultimodalEmbedder {
    model_id: String,
    max_length: usize,
    query_prefix: Option<String>,
    passage_prefix: Option<String>,
    engine: Arc<dyn MultimodalEngine>,
}

impl NativeMultimodalEmbedder {
    pub fn from_config(settings: &EmbedderConfig) -> Result<Self> {
        let options = multimodal_options(settings)?;
        let model_id = options.id.clone();
        let max_length = options.max_length;
        let engine: Arc<dyn MultimodalEngine> = Arc::new(MultimodalEmbedder::new(options));
        Ok(Self {
            model_id,
            max_length,
            query_prefix: settings.query_prefix.clone(),
            passage_prefix: settings.passage_prefix.clone(),
            engine,
        })
    }

    pub fn from_engine(
        model_id: String,
        max_length: usize,
        query_prefix: Option<String>,
        passage_prefix: Option<String>,
        engine: Arc<dyn MultimodalEngine>,
    ) -> Self {
        Self {
            model_id,
            max_length,
            query_prefix,
            passage_prefix,
            engine,
        }
    }

    /// Underlying engine for `minsync serve --multimodal`.
    pub fn into_engine(self) -> Arc<dyn MultimodalEngine> {
        self.engine
    }

    fn prefixed(prefix: &Option<String>, text: &str) -> String {
        match prefix {
            Some(prefix) => format!("{prefix}{text}"),
            None => text.to_string(),
        }
    }
}

#[async_trait]
impl Embedder for NativeMultimodalEmbedder {
    fn id(&self) -> &str {
        &self.model_id
    }

    fn max_length(&self) -> Option<usize> {
        Some(self.max_length)
    }

    fn is_deterministic(&self) -> bool {
        true
    }

    fn supports_images(&self) -> bool {
        true
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let inputs = texts
            .iter()
            .map(|text| MultimodalInput {
                text: Some(Self::prefixed(&self.passage_prefix, text)),
                ..MultimodalInput::default()
            })
            .collect();
        Ok(self.engine.embed(inputs).await?)
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let input = MultimodalInput {
            text: Some(Self::prefixed(&self.query_prefix, text)),
            ..MultimodalInput::default()
        };
        self.engine
            .embed(vec![input])
            .await?
            .into_iter()
            .next()
            .ok_or_else(|| MinSyncError::Embedding("empty response".to_string()))
    }

    async fn embed_images(&self, images: &[ImageInput]) -> Result<Vec<Vec<f32>>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }
        let inputs = images
            .iter()
            .map(|image| MultimodalInput {
                text: None,
                images: vec![MediaBytes {
                    bytes: image.bytes.clone(),
                    mime: image.mime.clone(),
                }],
                ..MultimodalInput::default()
            })
            .collect();
        Ok(self.engine.embed(inputs).await?)
    }
}

/// Served multimodal embedder: uses a running `inl serve --multimodal` /
/// `minsync serve --multimodal` daemon when one is alive, falling back to the
/// in-process [`NativeMultimodalEmbedder`] otherwise.
pub struct ServedMultimodalEmbedder {
    local: Box<dyn Embedder>,
    state_dir: PathBuf,
    instance: String,
    query_prefix: Option<String>,
    passage_prefix: Option<String>,
    batch_size: usize,
    timeout: Duration,
    model: String,
    max_length: usize,
    device: Option<String>,
    dtype: Option<String>,
    serve_root: PathBuf,
    remote: OnceCell<Option<Remote>>,
    gone: AtomicBool,
}

struct Remote {
    url: String,
    max_inputs: usize,
    client: reqwest::Client,
}

enum RemoteError {
    Gone(String),
    Failed(MinSyncError),
}

#[derive(serde::Deserialize)]
struct MultimodalResponse {
    embeddings: Vec<Vec<f32>>,
}

impl ServedMultimodalEmbedder {
    pub fn new(
        local: Box<dyn Embedder>,
        state_dir: PathBuf,
        instance: String,
        query_prefix: Option<String>,
        passage_prefix: Option<String>,
        batch_size: usize,
        timeout: Duration,
    ) -> Self {
        Self {
            local,
            state_dir,
            instance,
            query_prefix,
            passage_prefix,
            batch_size: batch_size.max(1),
            timeout,
            model: String::new(),
            max_length: DEFAULT_MULTIMODAL_MAX_LENGTH,
            device: None,
            dtype: None,
            serve_root: crate::serve::serve_root(),
            remote: OnceCell::new(),
            gone: AtomicBool::new(false),
        }
    }

    pub fn with_target(
        mut self,
        model: String,
        max_length: usize,
        device: Option<String>,
        dtype: Option<String>,
    ) -> Self {
        self.model = model;
        self.max_length = max_length;
        self.device = device;
        self.dtype = dtype;
        self
    }

    pub fn with_serve_root(mut self, serve_root: PathBuf) -> Self {
        self.serve_root = serve_root;
        self
    }

    pub fn from_config(settings: &EmbedderConfig) -> Result<Self> {
        let local = NativeMultimodalEmbedder::from_config(settings)?;
        let instance = crate::serve::multimodal_instance_key(settings)?;
        let pass = settings
            .per_batch_timeout_seconds
            .unwrap_or(settings.timeout_seconds)
            .max(inl_core::device::CPU_BATCH_TIMEOUT.as_secs());
        let max_length = settings
            .max_length
            .unwrap_or(DEFAULT_MULTIMODAL_MAX_LENGTH);
        Ok(Self::new(
            Box::new(local),
            crate::serve::default_state_dir(&instance),
            instance,
            settings.query_prefix.clone(),
            settings.passage_prefix.clone(),
            settings.batch_size.min(8),
            Duration::from_secs(pass),
        )
        .with_target(
            parse_multimodal_model_id(&settings.id)?.to_string(),
            max_length,
            settings.device.clone(),
            settings.dtype.clone(),
        ))
    }

    async fn remote(&self) -> Option<&Remote> {
        if self.gone.load(Ordering::Acquire) {
            return None;
        }
        self.remote
            .get_or_init(|| async {
                let health = match crate::serve::probe(&self.state_dir, &self.instance).await {
                    Some(health) if health.kind == "multimodal" => health,
                    _ => self.adopted_health().await?,
                };
                let client = crate::serve::loopback_client(self.timeout)?;
                tracing::info!(
                    url = %health.url,
                    pid = health.pid,
                    "using running multimodal inl/minsync serve"
                );
                Some(Remote {
                    url: health.url,
                    max_inputs: health.max_inputs.max(1),
                    client,
                })
            })
            .await
            .as_ref()
    }

    async fn adopted_health(&self) -> Option<crate::serve::Health> {
        if self.model.is_empty() {
            return None;
        }
        crate::serve::find_adoptable(
            &self.serve_root,
            &self.model,
            crate::serve::EngineKind::Multimodal,
            self.max_length,
            self.device.as_deref(),
            self.dtype.as_deref(),
        )
        .await
        .map(|found| found.health)
    }

    async fn embed_texts_remote(&self, inputs: Vec<String>) -> Option<Result<Vec<Vec<f32>>>> {
        let remote = self.remote().await?;
        let mut vectors = Vec::with_capacity(inputs.len());
        for batch in inputs.chunks(self.batch_size.min(remote.max_inputs)) {
            let payload: Vec<serde_json::Value> = batch
                .iter()
                .map(|text| serde_json::json!({ "text": text }))
                .collect();
            match send_multimodal(remote, &payload).await {
                Ok(batch_vectors) => vectors.extend(batch_vectors),
                Err(RemoteError::Gone(reason)) => {
                    tracing::warn!(%reason, "multimodal serve went away; embedding in-process");
                    self.gone.store(true, Ordering::Release);
                    return None;
                }
                Err(RemoteError::Failed(error)) => return Some(Err(error)),
            }
        }
        Some(Ok(vectors))
    }

    async fn embed_images_remote(&self, images: &[ImageInput]) -> Option<Result<Vec<Vec<f32>>>> {
        let remote = self.remote().await?;
        let mut vectors = Vec::with_capacity(images.len());
        // Keep image batches small so base64 payloads stay comfortably inside
        // the server's 32 MiB request body limit.
        let chunk_size = self.batch_size.min(remote.max_inputs).clamp(1, 4);
        for batch in images.chunks(chunk_size) {
            let payload: Vec<serde_json::Value> = batch
                .iter()
                .map(|image| {
                    let b64 = base64::engine::general_purpose::STANDARD.encode(&image.bytes);
                    serde_json::json!({ "images": [b64] })
                })
                .collect();
            match send_multimodal(remote, &payload).await {
                Ok(batch_vectors) => vectors.extend(batch_vectors),
                Err(RemoteError::Gone(reason)) => {
                    tracing::warn!(%reason, "multimodal serve went away; embedding in-process");
                    self.gone.store(true, Ordering::Release);
                    return None;
                }
                Err(RemoteError::Failed(error)) => return Some(Err(error)),
            }
        }
        Some(Ok(vectors))
    }
}

async fn send_multimodal(
    remote: &Remote,
    inputs: &[serde_json::Value],
) -> std::result::Result<Vec<Vec<f32>>, RemoteError> {
    let response = remote
        .client
        .post(format!("{}/embed_multimodal", remote.url))
        .json(&serde_json::json!({ "inputs": inputs }))
        .send()
        .await
        .map_err(|error| {
            if error.is_timeout() {
                RemoteError::Failed(MinSyncError::Timeout(format!(
                    "multimodal serve request timed out: {error}"
                )))
            } else if error.is_connect() || error.is_request() {
                RemoteError::Gone(error.to_string())
            } else {
                RemoteError::Failed(MinSyncError::Embedding(format!(
                    "multimodal serve request failed: {error}"
                )))
            }
        })?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        let message = serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|value| value["error"].as_str().map(str::to_string))
            .unwrap_or(body);
        let message = format!("multimodal serve {status}: {message}");
        return Err(if status == reqwest::StatusCode::SERVICE_UNAVAILABLE {
            RemoteError::Gone(message)
        } else {
            RemoteError::Failed(MinSyncError::Embedding(message))
        });
    }
    let parsed: MultimodalResponse = response.json().await.map_err(|error| {
        RemoteError::Failed(MinSyncError::Embedding(format!(
            "multimodal serve malformed response: {error}"
        )))
    })?;
    if parsed.embeddings.len() != inputs.len() {
        return Err(RemoteError::Failed(MinSyncError::Embedding(format!(
            "multimodal serve returned {} embeddings for {} inputs",
            parsed.embeddings.len(),
            inputs.len()
        ))));
    }
    Ok(parsed.embeddings)
}

fn prefixed(prefix: Option<&str>, text: &str) -> String {
    match prefix {
        Some(prefix) => format!("{prefix}{text}"),
        None => text.to_string(),
    }
}

#[async_trait]
impl Embedder for ServedMultimodalEmbedder {
    fn id(&self) -> &str {
        self.local.id()
    }

    fn max_length(&self) -> Option<usize> {
        self.local.max_length()
    }

    fn is_deterministic(&self) -> bool {
        self.local.is_deterministic()
    }

    fn supports_images(&self) -> bool {
        self.local.supports_images()
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let inputs = texts
            .iter()
            .map(|text| prefixed(self.passage_prefix.as_deref(), text))
            .collect();
        match self.embed_texts_remote(inputs).await {
            Some(result) => result,
            None => self.local.embed(texts).await,
        }
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let input = prefixed(self.query_prefix.as_deref(), text);
        match self.embed_texts_remote(vec![input]).await {
            Some(result) => result?
                .pop()
                .ok_or_else(|| MinSyncError::Embedding("empty response".to_string())),
            None => self.local.embed_query(text).await,
        }
    }

    async fn embed_images(&self, images: &[ImageInput]) -> Result<Vec<Vec<f32>>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }
        match self.embed_images_remote(images).await {
            Some(result) => result,
            None => self.local.embed_images(images).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use serde_json::{json, Value};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

    #[derive(Default)]
    struct FakeMultimodalLocal {
        text_calls: AtomicUsize,
        image_calls: AtomicUsize,
    }

    #[async_trait]
    impl Embedder for FakeMultimodalLocal {
        fn id(&self) -> &str {
            "google/embeddinggemma-2"
        }

        fn supports_images(&self) -> bool {
            true
        }

        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            self.text_calls.fetch_add(1, Ordering::SeqCst);
            Ok(texts.iter().map(|t| vec![-1.0, t.len() as f32]).collect())
        }

        async fn embed_images(&self, images: &[ImageInput]) -> Result<Vec<Vec<f32>>> {
            self.image_calls.fetch_add(1, Ordering::SeqCst);
            Ok(images
                .iter()
                .map(|img| vec![-2.0, img.bytes.len() as f32])
                .collect())
        }
    }

    struct MultimodalMock;

    impl Respond for MultimodalMock {
        fn respond(&self, request: &Request) -> ResponseTemplate {
            let body: Value = serde_json::from_slice(&request.body).unwrap();
            let embeddings: Vec<Vec<f32>> = body["inputs"]
                .as_array()
                .unwrap()
                .iter()
                .map(|item| {
                    if let Some(text) = item.get("text").and_then(|v| v.as_str()) {
                        vec![1.0, text.len() as f32]
                    } else {
                        let b64 = item["images"][0].as_str().unwrap();
                        let decoded = base64::engine::general_purpose::STANDARD
                            .decode(b64)
                            .unwrap();
                        vec![2.0, decoded.len() as f32]
                    }
                })
                .collect();
            ResponseTemplate::new(200).set_body_json(json!({
                "embeddings": embeddings,
                "model": "google/embeddinggemma-2",
                "dimensions": 2
            }))
        }
    }

    #[test]
    fn multimodal_options_default_to_text_and_image_modalities_only() {
        let mut settings = Config::default_for("t").embedder;
        settings.id = "native-multimodal:google/embeddinggemma-2".to_string();
        let options = multimodal_options(&settings).unwrap();
        assert_eq!(options.id, "google/embeddinggemma-2");
        assert_eq!(options.max_length, DEFAULT_MULTIMODAL_MAX_LENGTH);
        assert_eq!(
            options.modalities,
            Some(vec![Modality::Text, Modality::Image])
        );
    }

    #[tokio::test]
    async fn served_multimodal_embedder_routes_texts_and_images_to_daemon() {
        let server = MockServer::builder().start().await;
        let addr = server.address();
        let instance = "google-embeddinggemma-2-multimodal-test";
        Mock::given(method("GET"))
            .and(path("/health"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "status": "ok",
                "kind": "multimodal",
                "model_id": "google/embeddinggemma-2",
                "dimension": 2,
                "dtype": "f32",
                "device": "cpu",
                "attention": "eager",
                "max_length": DEFAULT_MULTIMODAL_MAX_LENGTH,
                "max_inputs": 16,
                "instance": instance,
                "pid": 1,
                "host": addr.ip().to_string(),
                "port": addr.port(),
                "url": format!("http://{addr}"),
                "uptime_seconds": 1,
                "requests_total": 0,
                "loaded": true,
                "build": {"version": "test", "features": []},
                "version": "test",
            })))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/embed_multimodal"))
            .respond_with(MultimodalMock)
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("host"), format!("{}\n", addr.ip())).unwrap();
        std::fs::write(dir.path().join("port"), format!("{}\n", addr.port())).unwrap();

        let embedder = ServedMultimodalEmbedder::new(
            Box::new(FakeMultimodalLocal::default()),
            dir.path().to_path_buf(),
            instance.to_string(),
            Some("Q: ".to_string()),
            Some("P: ".to_string()),
            8,
            Duration::from_secs(10),
        );

        assert!(embedder.supports_images());
        let text_vecs = embedder.embed(&["hello".to_string()]).await.unwrap();
        assert_eq!(text_vecs, vec![vec![1.0, "P: hello".len() as f32]]);

        let query_vec = embedder.embed_query("cat").await.unwrap();
        assert_eq!(query_vec, vec![1.0, "Q: cat".len() as f32]);

        let img_vecs = embedder
            .embed_images(&[ImageInput {
                bytes: vec![1, 2, 3, 4, 5],
                mime: Some("image/png".to_string()),
            }])
            .await
            .unwrap();
        assert_eq!(img_vecs, vec![vec![2.0, 5.0]]);
    }
}
