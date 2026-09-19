//! In-process native embedder backed by fastembed-rs (Qwen3 / candle).

use crate::config::EmbedderConfig;
use crate::embedder::Embedder;
use crate::error::{MinSyncError, Result};
use async_trait::async_trait;
use candle_core::{DType, Device};
use fastembed::Qwen3TextEmbedding;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokio::sync::OnceCell;

const DEFAULT_MAX_LENGTH: usize = 2048;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NativeDevice {
    Auto,
    Cpu,
    Metal,
    Cuda,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NativeDtype {
    F32,
    F16,
    Bf16,
}

pub struct NativeEmbedder {
    model_id: String,
    device: NativeDevice,
    dtype: NativeDtype,
    max_length: usize,
    batch_size: usize,
    cache_dir: Option<PathBuf>,
    query_prefix: Option<String>,
    passage_prefix: Option<String>,
    inner: OnceCell<Arc<Mutex<Qwen3TextEmbedding>>>,
}

impl NativeEmbedder {
    pub fn from_config(settings: &EmbedderConfig) -> Result<Self> {
        let model_id = parse_native_model_id(&settings.id)?.to_string();
        Ok(Self {
            model_id,
            device: parse_device(settings.device.as_deref())?,
            dtype: parse_dtype(settings.dtype.as_deref())?,
            max_length: settings.max_length.unwrap_or(DEFAULT_MAX_LENGTH),
            batch_size: settings.batch_size,
            cache_dir: settings.model_cache_dir.as_ref().map(PathBuf::from),
            query_prefix: settings.query_prefix.clone(),
            passage_prefix: settings.passage_prefix.clone(),
            inner: OnceCell::new(),
        })
    }

    async fn model(&self) -> Result<Arc<Mutex<Qwen3TextEmbedding>>> {
        self.inner
            .get_or_try_init(|| async {
                let repo = self.model_id.clone();
                let device = self.device;
                let dtype = self.dtype;
                let max_length = self.max_length;
                let cache_dir = self.cache_dir.clone();
                let loaded = tokio::task::spawn_blocking(move || {
                    apply_cache_dir(cache_dir.as_deref());
                    let device = resolve_device(device)?;
                    Qwen3TextEmbedding::from_hf(&repo, &device, dtype.to_candle(), max_length)
                        .map_err(|error| {
                            MinSyncError::Embedding(format!("native model load failed: {error}"))
                        })
                })
                .await
                .map_err(|error| {
                    MinSyncError::Embedding(format!("native model load join failed: {error}"))
                })??;
                Ok(Arc::new(Mutex::new(loaded)))
            })
            .await
            .cloned()
    }

    async fn embed_texts(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        if self.batch_size == 0 {
            return Err(MinSyncError::Embedding(
                "batch_size must be greater than 0".to_string(),
            ));
        }
        let model = self.model().await?;
        let batch_size = self.batch_size;
        tokio::task::spawn_blocking(move || {
            let model = model.lock().map_err(|error| {
                MinSyncError::Embedding(format!("native model lock poisoned: {error}"))
            })?;
            let mut all = Vec::with_capacity(texts.len());
            for chunk in texts.chunks(batch_size) {
                let vectors = model.embed(chunk).map_err(|error| {
                    MinSyncError::Embedding(format!("native embedding failed: {error}"))
                })?;
                if vectors.len() != chunk.len() {
                    return Err(MinSyncError::Embedding(format!(
                        "native embedder returned {} embeddings for {} inputs",
                        vectors.len(),
                        chunk.len()
                    )));
                }
                all.extend(vectors);
            }
            Ok(all)
        })
        .await
        .map_err(|error| {
            MinSyncError::Embedding(format!("native embedding join failed: {error}"))
        })?
    }
}

#[async_trait]
impl Embedder for NativeEmbedder {
    fn id(&self) -> &str {
        &self.model_id
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let inputs = texts
            .iter()
            .map(|text| match &self.passage_prefix {
                Some(prefix) => format!("{prefix}{text}"),
                None => text.clone(),
            })
            .collect();
        self.embed_texts(inputs).await
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let input = match &self.query_prefix {
            Some(prefix) => format!("{prefix}{text}"),
            None => text.to_string(),
        };
        self.embed_texts(vec![input])
            .await?
            .into_iter()
            .next()
            .ok_or_else(|| MinSyncError::Embedding("empty response".to_string()))
    }
}

fn parse_native_model_id(id: &str) -> Result<&str> {
    let model = id.strip_prefix("native:").unwrap_or(id);
    if model.is_empty() {
        return Err(MinSyncError::Config(
            "native embedder id is missing a model name".to_string(),
        ));
    }
    Ok(model)
}

fn parse_device(value: Option<&str>) -> Result<NativeDevice> {
    match value {
        None | Some("auto") => Ok(NativeDevice::Auto),
        Some("cpu") => Ok(NativeDevice::Cpu),
        Some("metal") => Ok(NativeDevice::Metal),
        Some("cuda") => Ok(NativeDevice::Cuda),
        Some(other) => Err(MinSyncError::Config(format!(
            "invalid native device '{other}': expected auto, cpu, metal, or cuda"
        ))),
    }
}

fn parse_dtype(value: Option<&str>) -> Result<NativeDtype> {
    match value {
        None | Some("f32") => Ok(NativeDtype::F32),
        Some("f16") => Ok(NativeDtype::F16),
        Some("bf16") => Ok(NativeDtype::Bf16),
        Some(other) => Err(MinSyncError::Config(format!(
            "invalid native dtype '{other}': expected f32, f16, or bf16"
        ))),
    }
}

impl NativeDtype {
    fn to_candle(self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::F16 => DType::F16,
            Self::Bf16 => DType::BF16,
        }
    }
}

fn resolve_device(spec: NativeDevice) -> Result<Device> {
    match spec {
        NativeDevice::Cpu => Ok(Device::Cpu),
        NativeDevice::Auto => Ok(auto_device()),
        NativeDevice::Metal => metal_device(),
        NativeDevice::Cuda => Err(MinSyncError::Config(
            "device cuda requires a CUDA build of minsync".to_string(),
        )),
    }
}

fn auto_device() -> Device {
    #[cfg(target_os = "macos")]
    {
        Device::new_metal(0).unwrap_or(Device::Cpu)
    }
    #[cfg(not(target_os = "macos"))]
    {
        Device::Cpu
    }
}

fn metal_device() -> Result<Device> {
    #[cfg(target_os = "macos")]
    {
        Device::new_metal(0)
            .map_err(|error| MinSyncError::Config(format!("failed to open metal device: {error}")))
    }
    #[cfg(not(target_os = "macos"))]
    {
        Err(MinSyncError::Config(
            "device metal is only available on macOS".to_string(),
        ))
    }
}

fn apply_cache_dir(configured: Option<&std::path::Path>) {
    if let Some(dir) = configured {
        let _ = std::fs::create_dir_all(dir);
        std::env::set_var("FASTEMBED_CACHE_DIR", dir);
        return;
    }
    if std::env::var_os("HF_HOME").is_some() || std::env::var_os("FASTEMBED_CACHE_DIR").is_some() {
        return;
    }
    let dir = default_model_cache_dir();
    let _ = std::fs::create_dir_all(&dir);
    std::env::set_var("FASTEMBED_CACHE_DIR", &dir);
}

fn default_model_cache_dir() -> PathBuf {
    #[cfg(windows)]
    {
        let root = std::env::var("LOCALAPPDATA").unwrap_or_else(|_| ".".to_string());
        PathBuf::from(root).join("minsync").join("models")
    }
    #[cfg(not(windows))]
    {
        let root = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        PathBuf::from(root)
            .join(".cache")
            .join("minsync")
            .join("models")
    }
}
