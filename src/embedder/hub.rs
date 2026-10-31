//! Locating model files: a Hugging Face repo (downloaded into the cache on
//! first use) or a local directory laid out like one. Shared by every native
//! engine so they resolve `HF_HOME`, `HF_ENDPOINT` and `HF_TOKEN` alike.
//!
//! A download announces itself in the log (the progress bar only draws on a
//! terminal, and the weights take minutes on a slow link), and the weights it
//! brings are checked against the sha256 the Hub names the blob after, so a
//! damaged partial download is replaced instead of left in the cache for every
//! later run to trip over.

use crate::error::{MinSyncError, Result};
use hf_hub::api::sync::{ApiBuilder, ApiRepo};
use hf_hub::{Cache, CacheRepo};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Why a file of a Hub repo could not be had. Displays as one actionable line.
#[derive(Debug)]
pub(crate) struct HubError {
    message: String,
    damaged: bool,
}

impl HubError {
    /// The file arrived but does not match the hash the Hub published for it,
    /// as opposed to not arriving at all.
    pub(crate) fn is_damaged(&self) -> bool {
        self.damaged
    }
}

impl std::fmt::Display for HubError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for HubError {}

/// A Hugging Face repo behind the local cache.
pub(crate) struct HubRepo {
    api: ApiRepo,
    cache: CacheRepo,
    id: String,
    endpoint: String,
    cache_dir: PathBuf,
}

pub(crate) fn hf_repo(repo_id: &str, default_cache_dir: PathBuf) -> Result<HubRepo> {
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
    HubRepo::new(repo_id, cache_dir, endpoint, token)
}

impl HubRepo {
    pub(crate) fn new(
        repo_id: &str,
        cache_dir: PathBuf,
        endpoint: String,
        token: Option<String>,
    ) -> Result<Self> {
        let api = ApiBuilder::new()
            .with_cache_dir(cache_dir.clone())
            .with_endpoint(endpoint.clone())
            .with_token(token)
            .with_progress(true)
            .build()
            .map_err(|e| {
                MinSyncError::Embedding(format!("native model API initialization failed: {e}"))
            })?;
        Ok(Self {
            api: api.model(repo_id.to_string()),
            cache: Cache::new(cache_dir.clone()).model(repo_id.to_string()),
            id: repo_id.to_string(),
            endpoint,
            cache_dir,
        })
    }

    /// The local path of `name`, downloading it first when it is not cached.
    pub(crate) fn get(&self, name: &str) -> std::result::Result<PathBuf, HubError> {
        if let Some(path) = self.cache.get(name) {
            return Ok(path);
        }
        let weights = name.ends_with(".safetensors");
        if weights {
            tracing::info!(
                repo = %self.id,
                file = name,
                from = %self.endpoint,
                into = %self.cache_dir.display(),
                "downloading model weights (first use of this model; later runs read the local copy)"
            );
        } else {
            tracing::debug!(repo = %self.id, file = name, "downloading model file");
        }
        let started = Instant::now();
        let path = self
            .api
            .get(name)
            .map_err(|source| self.download_failed(name, &source))?;
        if !weights {
            return Ok(path);
        }
        let path = self.verified(name, path)?;
        tracing::info!(
            repo = %self.id,
            file = name,
            megabytes = std::fs::metadata(&path).map_or(0, |meta| meta.len() / 1_000_000),
            seconds = started.elapsed().as_secs(),
            "model weights downloaded and verified"
        );
        Ok(path)
    }

    /// Folder of this repo inside the cache (`models--org--name`), the unit to
    /// copy to a machine that cannot reach the Hub.
    fn folder(&self) -> String {
        format!("models--{}", self.id.replace('/', "--"))
    }

    fn download_failed(&self, name: &str, source: &dyn std::fmt::Display) -> HubError {
        HubError {
            message: format!(
                "{name} of {} could not be downloaded from {}: {source}. Check the network, \
                 set HF_ENDPOINT to a mirror, or copy the folder {} from a machine that has the \
                 model into {}",
                self.id,
                self.endpoint,
                self.folder(),
                self.cache_dir.display()
            ),
            damaged: false,
        }
    }

    /// `path` as downloaded, unless its content fails the sha256 check: then it
    /// is removed and fetched once more, and a second failure is an error that
    /// leaves nothing in the cache.
    fn verified(&self, name: &str, path: PathBuf) -> std::result::Result<PathBuf, HubError> {
        let damage = |path: &Path| match check_blob(path) {
            Ok(BlobCheck::Damaged { expected, actual }) => Ok(Some((expected, actual))),
            Ok(_) => Ok(None),
            Err(error) => Err(HubError {
                message: format!("could not read {name} of {} to check it: {error}", self.id),
                damaged: false,
            }),
        };
        let Some((expected, actual)) = damage(&path)? else {
            return Ok(path);
        };
        tracing::warn!(
            repo = %self.id,
            file = name,
            expected = %expected,
            actual = %actual,
            "downloaded weights failed their sha256 check; discarding them and downloading again"
        );
        discard(&path);
        let again = self
            .api
            .get(name)
            .map_err(|source| self.download_failed(name, &source))?;
        match damage(&again)? {
            None => Ok(again),
            Some((expected, actual)) => {
                discard(&again);
                Err(HubError {
                    message: format!(
                        "{name} of {} failed its sha256 check twice (expected {expected}, got \
                         {actual}): the download is being corrupted on the way. Retry later, or \
                         copy the folder {} from a machine that has the model into {}",
                        self.id,
                        self.folder(),
                        self.cache_dir.display()
                    ),
                    damaged: true,
                })
            }
        }
    }
}

/// What a sha256 check of a cached blob found.
enum BlobCheck {
    Intact,
    Damaged {
        expected: String,
        actual: String,
    },
    /// The blob is not named after its content hash (small git files, a copy
    /// outside the Hub layout), so there is nothing to compare against.
    Unchecked,
}

/// Hash the blob behind a snapshot entry. LFS blobs are named by the sha256 of
/// their content, which is what `hf-hub` stores under `blobs/`.
fn check_blob(pointer: &Path) -> std::io::Result<BlobCheck> {
    let blob = std::fs::canonicalize(pointer)?;
    let expected = match blob.file_name().and_then(|name| name.to_str()) {
        Some(name) if name.len() == 64 && name.bytes().all(|byte| byte.is_ascii_hexdigit()) => {
            name.to_ascii_lowercase()
        }
        _ => return Ok(BlobCheck::Unchecked),
    };
    let mut file = std::fs::File::open(&blob)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let actual = hex::encode(hasher.finalize());
    Ok(if actual == expected {
        BlobCheck::Intact
    } else {
        BlobCheck::Damaged { expected, actual }
    })
}

/// Remove a cached file: the blob its snapshot entry points at, and the entry.
fn discard(pointer: &Path) {
    if let Ok(blob) = std::fs::canonicalize(pointer) {
        let _ = std::fs::remove_file(blob);
    }
    let _ = std::fs::remove_file(pointer);
}

/// Token written by `huggingface-cli login`; `Cache::default` panics without a home directory.
fn default_hf_token() -> Option<String> {
    let has_home = std::env::var_os("HOME").is_some() || std::env::var_os("USERPROFILE").is_some();
    has_home.then(|| hf_hub::Cache::default().token()).flatten()
}

/// Safetensors files of a repo, resolving `model.safetensors.index.json` for sharded checkpoints.
pub(crate) fn safetensors_weight_files(repo: &HubRepo) -> Result<Vec<PathBuf>> {
    const SINGLE_FILE: &str = "model.safetensors";
    const INDEX_FILE: &str = "model.safetensors.index.json";
    match repo.get(SINGLE_FILE) {
        Ok(path) => return Ok(vec![path]),
        // A damaged download is not evidence that the checkpoint is sharded.
        Err(error) if error.is_damaged() => {
            return Err(MinSyncError::Embedding(format!("native model {error}")))
        }
        Err(_) => {}
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
    Hub(Box<HubRepo>),
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
            Self::Hub(repo) => repo
                .get(name)
                .map_err(|error| MinSyncError::Embedding(format!("native model {error}"))),
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use wiremock::matchers::{method, path_regex};
    use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

    const COMMIT: &str = "4d68cc696984e3c342d959c659e94b48fbe16578";

    fn sha256_hex(bytes: &[u8]) -> String {
        hex::encode(Sha256::digest(bytes))
    }

    /// The two requests `hf-hub` makes per file: a `Range: bytes=0-0` probe for
    /// the headers, then a ranged GET for the body. The first `bad_bodies`
    /// bodies are corrupted copies of the same length.
    struct FakeHub {
        good: Vec<u8>,
        bad: Vec<u8>,
        bad_bodies: usize,
        bodies: Arc<AtomicUsize>,
    }

    impl Respond for FakeHub {
        fn respond(&self, request: &Request) -> ResponseTemplate {
            let range = request
                .headers
                .get("range")
                .and_then(|value| value.to_str().ok())
                .unwrap_or("");
            let start: usize = range
                .strip_prefix("bytes=")
                .and_then(|spec| spec.split('-').next())
                .and_then(|from| from.parse().ok())
                .unwrap_or(0);
            let (source, end) = if range == "bytes=0-0" {
                (&self.good, 1)
            } else if self.bodies.fetch_add(1, Ordering::SeqCst) < self.bad_bodies {
                (&self.bad, self.bad.len())
            } else {
                (&self.good, self.good.len())
            };
            ResponseTemplate::new(206)
                .insert_header("x-repo-commit", COMMIT)
                .insert_header("x-linked-etag", format!("\"{}\"", sha256_hex(&self.good)))
                .insert_header(
                    "content-range",
                    format!("bytes {start}-{}/{}", end - 1, self.good.len()),
                )
                .set_body_bytes(source[start..end].to_vec())
        }
    }

    struct Fixture {
        server: MockServer,
        repo: Arc<HubRepo>,
        good: Vec<u8>,
        bodies: Arc<AtomicUsize>,
        _cache: tempfile::TempDir,
    }

    async fn fixture(bad_bodies: usize) -> Fixture {
        let good: Vec<u8> = (0..4096u32).map(|i| (i % 251) as u8).collect();
        let mut bad = good.clone();
        bad[1000] ^= 0xFF;
        let bodies = Arc::new(AtomicUsize::new(0));
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path_regex(r"^/org/model/resolve/main/.+$"))
            .respond_with(FakeHub {
                good: good.clone(),
                bad,
                bad_bodies,
                bodies: Arc::clone(&bodies),
            })
            .mount(&server)
            .await;
        let cache = tempfile::tempdir().unwrap();
        let repo = HubRepo::new("org/model", cache.path().to_path_buf(), server.uri(), None)
            .expect("repo handle");
        Fixture {
            server,
            repo: Arc::new(repo),
            good,
            bodies,
            _cache: cache,
        }
    }

    async fn fetch(
        repo: &Arc<HubRepo>,
        name: &'static str,
    ) -> std::result::Result<PathBuf, HubError> {
        let repo = Arc::clone(repo);
        tokio::task::spawn_blocking(move || repo.get(name))
            .await
            .expect("join")
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_intact_download_is_returned_as_it_arrived() {
        let fx = fixture(0).await;
        let path = fetch(&fx.repo, "model.safetensors").await.unwrap();
        assert_eq!(std::fs::read(path).unwrap(), fx.good);
        assert_eq!(fx.bodies.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_download_that_arrives_damaged_is_discarded_and_fetched_again() {
        let fx = fixture(1).await;
        let path = fetch(&fx.repo, "model.safetensors").await.unwrap();
        assert_eq!(std::fs::read(path).unwrap(), fx.good);
        assert_eq!(fx.bodies.load(Ordering::SeqCst), 2, "one retry, no more");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_download_that_stays_damaged_fails_and_leaves_nothing_behind() {
        let fx = fixture(usize::MAX).await;
        let error = fetch(&fx.repo, "model.safetensors").await.unwrap_err();
        assert!(error.is_damaged(), "{error}");
        assert_eq!(fx.bodies.load(Ordering::SeqCst), 2, "one retry, no more");
        assert!(
            fx.repo.cache.get("model.safetensors").is_none(),
            "a damaged file must not stay in the cache"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_cached_file_is_returned_without_any_request() {
        let fx = fixture(0).await;
        fetch(&fx.repo, "model.safetensors").await.unwrap();
        let before = fx.server.received_requests().await.unwrap().len();
        let path = fetch(&fx.repo, "model.safetensors").await.unwrap();
        assert_eq!(std::fs::read(path).unwrap(), fx.good);
        assert_eq!(fx.server.received_requests().await.unwrap().len(), before);
    }

    #[test]
    fn an_unreachable_hub_error_names_the_repo_and_the_endpoint() {
        let cache = tempfile::tempdir().unwrap();
        let repo = HubRepo::new(
            "org/model",
            cache.path().to_path_buf(),
            "http://127.0.0.1:9".to_string(),
            None,
        )
        .unwrap();
        let error = repo.get("config.json").unwrap_err();
        let text = error.to_string();
        assert!(!error.is_damaged());
        assert!(text.contains("org/model"), "{text}");
        assert!(text.contains("http://127.0.0.1:9"), "{text}");
    }

    fn blob(dir: &Path, name: &str, content: &[u8]) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, content).unwrap();
        path
    }

    #[test]
    fn a_blob_named_by_its_sha256_is_checked_against_its_content() {
        let dir = tempfile::tempdir().unwrap();
        let content = b"weights".to_vec();
        let intact = blob(dir.path(), &sha256_hex(&content), &content);
        assert!(matches!(check_blob(&intact).unwrap(), BlobCheck::Intact));

        let named_for_other = sha256_hex(b"what the hub meant to send");
        let damaged = blob(dir.path(), &named_for_other, &content);
        assert!(matches!(
            check_blob(&damaged).unwrap(),
            BlobCheck::Damaged { .. }
        ));
    }

    #[test]
    fn a_file_without_a_content_hash_name_is_not_checked() {
        let dir = tempfile::tempdir().unwrap();
        let local = blob(dir.path(), "model.safetensors", b"anything");
        assert!(matches!(check_blob(&local).unwrap(), BlobCheck::Unchecked));
        // A 40-character git blob id names small non-LFS files.
        let git = blob(dir.path(), &"a".repeat(40), b"{}");
        assert!(matches!(check_blob(&git).unwrap(), BlobCheck::Unchecked));
    }
}
