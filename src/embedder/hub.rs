//! Locating model files: a Hugging Face repo (downloaded into the cache on
//! first use) or a local directory laid out like one. Shared by every native
//! engine so they resolve `HF_HOME`, `HF_ENDPOINT` and `HF_TOKEN` alike.

use crate::error::{MinSyncError, Result};
use hf_hub::api::sync::{ApiBuilder, ApiRepo};
use std::path::{Path, PathBuf};

pub(crate) fn hf_repo(repo_id: &str, default_cache_dir: PathBuf) -> Result<ApiRepo> {
    let cache_dir = std::env::var("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or(default_cache_dir);
    let endpoint =
        std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".to_string());
    let token = std::env::var("HF_TOKEN")
        .ok()
        .map(|token| token.trim().to_string())
        .filter(|token| !token.is_empty())
        .or_else(default_hf_token);
    let api = ApiBuilder::new()
        .with_cache_dir(cache_dir)
        .with_endpoint(endpoint)
        .with_token(token)
        .with_progress(true)
        .build()
        .map_err(|e| {
            MinSyncError::Embedding(format!("native model API initialization failed: {e}"))
        })?;
    Ok(api.model(repo_id.to_string()))
}

/// Token written by `huggingface-cli login`; `Cache::default` panics without a home directory.
fn default_hf_token() -> Option<String> {
    let has_home = std::env::var_os("HOME").is_some() || std::env::var_os("USERPROFILE").is_some();
    has_home.then(|| hf_hub::Cache::default().token()).flatten()
}

/// Safetensors files of a repo, resolving `model.safetensors.index.json` for sharded checkpoints.
pub(crate) fn safetensors_weight_files(repo: &ApiRepo) -> Result<Vec<PathBuf>> {
    const SINGLE_FILE: &str = "model.safetensors";
    const INDEX_FILE: &str = "model.safetensors.index.json";
    if let Ok(path) = repo.get(SINGLE_FILE) {
        return Ok(vec![path]);
    }
    let fetch = |file: &str| {
        repo.get(file).map_err(|e| {
            MinSyncError::Embedding(format!(
                "native model weights download failed ({file}): {e}"
            ))
        })
    };
    let index: serde_json::Value = serde_json::from_slice(&std::fs::read(fetch(INDEX_FILE)?)?)
        .map_err(|e| {
            MinSyncError::Embedding(format!("native model {INDEX_FILE} is invalid: {e}"))
        })?;
    let mut shards: Vec<String> = index["weight_map"]
        .as_object()
        .ok_or_else(|| {
            MinSyncError::Embedding(format!("native model {INDEX_FILE} has no weight_map"))
        })?
        .values()
        .filter_map(|file| file.as_str().map(str::to_string))
        .collect();
    shards.sort_unstable();
    shards.dedup();
    shards.iter().map(|file| fetch(file)).collect()
}

/// Where a native model's files come from.
pub(crate) enum ModelSource {
    /// A directory laid out like a Hugging Face snapshot (offline use, tests).
    Dir(PathBuf),
    Hub(Box<ApiRepo>),
}

impl ModelSource {
    /// `model` names a local directory when one exists at that path, else a
    /// Hugging Face repo id.
    pub(crate) fn resolve(model: &str, default_cache_dir: PathBuf) -> Result<Self> {
        let path = Path::new(model);
        if path.is_dir() {
            return Ok(Self::Dir(path.to_path_buf()));
        }
        Ok(Self::Hub(Box::new(hf_repo(model, default_cache_dir)?)))
    }

    /// A file that must exist.
    pub(crate) fn fetch(&self, name: &str) -> Result<PathBuf> {
        match self {
            Self::Dir(dir) => {
                let path = dir.join(name);
                path.is_file().then_some(path).ok_or_else(|| {
                    MinSyncError::Embedding(format!(
                        "native model file {name} is missing from {}",
                        dir.display()
                    ))
                })
            }
            Self::Hub(repo) => repo.get(name).map_err(|e| {
                MinSyncError::Embedding(format!("native model {name} download failed: {e}"))
            }),
        }
    }

    pub(crate) fn weight_files(&self) -> Result<Vec<PathBuf>> {
        match self {
            Self::Dir(dir) => {
                let single = dir.join("model.safetensors");
                if single.is_file() {
                    Ok(vec![single])
                } else {
                    Err(MinSyncError::Embedding(format!(
                        "native model weights model.safetensors are missing from {}",
                        dir.display()
                    )))
                }
            }
            Self::Hub(repo) => safetensors_weight_files(repo),
        }
    }
}
