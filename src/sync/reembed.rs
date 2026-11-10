//! Rebuilding every dense vector with a different embedder.
//!
//! A vector store holds vectors of one model. Switching the configured
//! embedder therefore means writing a *new* store, and the old one keeps
//! serving until the new one is complete:
//!
//! 1. the stored chunks are read page by page (text, headings, hashes and the
//!    sparse vectors; no file is re-read, nothing is re-chunked or re-encoded);
//! 2. each page is embedded with the configured embedder and written to a
//!    staging store next to the live one, along with its sparse vectors;
//! 3. when every row is across, the staging store replaces the live one in a
//!    single rename and the cursor records the new embedder.
//!
//! Progress lives in the staging store itself: a page is committed once its
//! rows are there, so a rebuild that was killed resumes at the last id it
//! wrote. A marker file names the embedder the staging store is being built
//! for, so a rebuild toward a different embedder starts over instead of
//! continuing into vectors of another model.

use crate::config::Config;
use crate::embedder::Embedder;
use crate::embedder_identity::EmbedderIdentity;
use crate::error::{MinSyncError, Result};
use crate::vectorstore::{Document, StoredChunk, VectorStore};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Chunks read, embedded and written per step.
const PAGE: usize = 256;

/// Marker file in `.minsync/`: a rebuild is in progress.
pub(crate) const MARKER_FILE: &str = "reembed.json";

/// What an interrupted rebuild needs to continue.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct Marker {
    /// The embedder the staging store is being filled for.
    pub target: EmbedderIdentity,
    /// Directory of the staging store, relative to `.minsync/`.
    pub staging: String,
}

/// Outcome of one rebuild.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Reembedded {
    /// Chunks embedded with the new embedder in this run.
    pub embedded: usize,
    /// Chunks the run found already across from an earlier, interrupted one.
    pub resumed: usize,
    /// Texts the embedder was called with in this run (identical chunks embed once).
    pub texts: usize,
    pub api_calls: usize,
}

pub(crate) fn marker_path(minsync_dir: &Path) -> PathBuf {
    minsync_dir.join(MARKER_FILE)
}

pub(crate) fn load_marker(minsync_dir: &Path) -> Option<Marker> {
    let bytes = std::fs::read(marker_path(minsync_dir)).ok()?;
    serde_json::from_slice(&bytes).ok()
}

fn save_marker(minsync_dir: &Path, marker: &Marker) -> Result<()> {
    let path = marker_path(minsync_dir);
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(marker)?)?;
    std::fs::rename(tmp, path)?;
    Ok(())
}

/// Text the probe embeds: short, and ordinary enough for any model.
const PROBE_TEXT: &str = "minsync embedder probe";

/// Embed one short text with the configured embedder and check that what comes
/// back is what the rest of the rebuild will assume: finite numbers, and the
/// width `[vectorstore.options] dimension` promises. Run before a staging
/// store exists, so a wrong dimension, an unloadable model or an unreachable
/// endpoint costs one request instead of the first page of a rebuild.
///
/// A sparse-only target stores a placeholder and has no model to probe.
pub(crate) async fn probe(config: &Config, embedder: &dyn Embedder) -> Result<EmbedderIdentity> {
    let identity = EmbedderIdentity::of(config, embedder)?;
    if config.is_sparse_only() {
        return Ok(identity);
    }
    let mut answer =
        embedder
            .embed(&[PROBE_TEXT.to_string()])
            .await
            .map_err(|error| match error {
                MinSyncError::Timeout(message) => {
                    MinSyncError::Timeout(format!("embedder probe timed out: {message}"))
                }
                other => MinSyncError::Embedding(format!(
                    "the embedder probe failed, nothing was rebuilt: {other}"
                )),
            })?;
    let bad = |what: String| MinSyncError::Embedding(format!("the embedder probe {what}"));
    if answer.len() != 1 {
        return Err(bad(format!("returned {} vectors for 1 text", answer.len())));
    }
    let vector = answer.remove(0);
    if vector.is_empty() {
        return Err(bad("returned an empty vector".to_string()));
    }
    if vector.iter().any(|value| !value.is_finite()) {
        return Err(bad("returned NaN or infinite numbers".to_string()));
    }
    if vector.len() != identity.dimension {
        return Err(MinSyncError::Config(format!(
            "the embedder produces {}-dimensional vectors but [vectorstore.options] dimension \
             is {}: set `dimension = {}` and run `minsync sync --reembed` again",
            vector.len(),
            identity.dimension,
            vector.len()
        )));
    }
    Ok(identity)
}

/// Rebuild the live store at `live` with `embedder`.
///
/// `make_store` opens a store at a path with the *target* dimension.
/// `on_page` is told how many chunks each committed page held.
pub(crate) async fn rebuild(
    minsync_dir: &Path,
    config: &Config,
    embedder: &dyn Embedder,
    live: &Path,
    live_store: &dyn VectorStore,
    make_store: &dyn Fn(&Path) -> Result<Box<dyn VectorStore>>,
    mut on_page: impl FnMut(usize),
) -> Result<(Reembedded, EmbedderIdentity)> {
    let target = EmbedderIdentity::of(config, embedder)?;
    let staging_name = format!("{}.reembed", relative_name(minsync_dir, live)?);
    let staging = minsync_dir.join(&staging_name);
    let marker = Marker {
        target: target.clone(),
        staging: staging_name,
    };

    // A leftover staging store only counts when it was started for this same
    // target; anything else is vectors of some other model.
    let resuming = load_marker(minsync_dir).as_ref() == Some(&marker) && staging.exists();
    if !resuming {
        if staging.exists() {
            std::fs::remove_dir_all(&staging)?;
        }
        save_marker(minsync_dir, &marker)?;
    }
    let mut staging_store = make_store(&staging)?;

    let mut outcome = Reembedded::default();
    let mut after: Option<String> = None;
    if resuming {
        // Everything at or before the staging store's last id is across.
        after = last_id(staging_store.as_ref())?;
    }
    loop {
        let page = live_store.scan_chunks(after.as_deref(), PAGE)?;
        if page.is_empty() {
            break;
        }
        after = page.last().map(|chunk| chunk.doc.id.clone());
        let embedded = embed_page(embedder, &page, &mut outcome).await?;
        let docs: Vec<Document> = page
            .iter()
            .zip(embedded)
            .map(|(chunk, embedding)| Document {
                embedding,
                ..chunk.doc.clone()
            })
            .collect();
        staging_store.insert_new(&docs)?;
        copy_sparse(staging_store.as_mut(), &page)?;
        outcome.embedded += page.len();
        on_page(page.len());
    }
    if resuming {
        outcome.resumed = staging_store.doc_count().saturating_sub(outcome.embedded);
    }
    staging_store.flush()?;
    drop(staging_store);
    Ok((outcome, target))
}

/// Swap the finished staging store in for the live one.
pub(crate) fn swap_in(minsync_dir: &Path, live: &Path) -> Result<()> {
    let marker = load_marker(minsync_dir)
        .ok_or_else(|| MinSyncError::Other("no embedder rebuild is in progress".to_string()))?;
    let staging = minsync_dir.join(&marker.staging);
    let retired = minsync_dir.join(format!("{}.retired", marker.staging));
    if retired.exists() {
        std::fs::remove_dir_all(&retired)?;
    }
    if staging.exists() && live.exists() {
        std::fs::rename(live, &retired)?;
    }
    if staging.exists() {
        std::fs::rename(&staging, live)?;
    }
    let _ = std::fs::remove_dir_all(retired);
    Ok(())
}

/// The rebuild is complete and the cursor says so: forget it.
pub(crate) fn finish(minsync_dir: &Path) {
    let _ = std::fs::remove_file(marker_path(minsync_dir));
}

fn relative_name(minsync_dir: &Path, live: &Path) -> Result<String> {
    live.strip_prefix(minsync_dir.canonicalize()?)
        .or_else(|_| live.strip_prefix(minsync_dir))
        .map(|path| path.to_string_lossy().into_owned())
        .map_err(|_| MinSyncError::Other("store is outside .minsync/".to_string()))
}

/// Rows the staging store of an interrupted rebuild toward `target` already
/// holds, or `None` when there is no such rebuild to continue.
pub(crate) fn staged_rows(
    minsync_dir: &Path,
    target: &EmbedderIdentity,
    open: &dyn Fn(&Path) -> Result<Box<dyn VectorStore>>,
) -> Result<Option<usize>> {
    let Some(marker) = load_marker(minsync_dir) else {
        return Ok(None);
    };
    let staging = minsync_dir.join(&marker.staging);
    if &marker.target != target || !staging.exists() {
        return Ok(None);
    }
    Ok(Some(open(&staging)?.doc_count()))
}

fn last_id(store: &dyn VectorStore) -> Result<Option<String>> {
    // Walk the staging store with the same keyset paging the source uses.
    let mut last = None;
    loop {
        let page = store.scan_chunks(last.as_deref(), 4096)?;
        match page.last() {
            Some(chunk) => last = Some(chunk.doc.id.clone()),
            None => return Ok(last),
        }
    }
}

/// One embedding per chunk of `page`; identical texts are embedded once.
async fn embed_page(
    embedder: &dyn Embedder,
    page: &[StoredChunk],
    outcome: &mut Reembedded,
) -> Result<Vec<Vec<f32>>> {
    if embedder.is_sparse_only() {
        // The vector column is a required placeholder, as the indexer writes it.
        return Ok(page
            .iter()
            .map(|_| vec![0.0; crate::config::SPARSE_ONLY_DIMENSION])
            .collect());
    }
    let mut index_of: HashMap<&str, usize> = HashMap::new();
    let mut texts: Vec<String> = Vec::new();
    let slots: Vec<usize> = page
        .iter()
        .map(|chunk| {
            *index_of
                .entry(chunk.doc.content_hash.as_str())
                .or_insert_with(|| {
                    texts.push(chunk.doc.text.clone());
                    texts.len() - 1
                })
        })
        .collect();
    let vectors = embedder.embed(&texts).await?;
    if vectors.len() != texts.len() {
        return Err(MinSyncError::Embedding(format!(
            "expected {} embeddings, got {}",
            texts.len(),
            vectors.len()
        )));
    }
    outcome.texts += texts.len();
    outcome.api_calls += 1;
    Ok(slots
        .into_iter()
        .map(|slot| vectors[slot].clone())
        .collect())
}

fn copy_sparse(store: &mut dyn VectorStore, page: &[StoredChunk]) -> Result<()> {
    let mut by_model: HashMap<&str, Vec<(String, crate::sparse::SparseVector)>> = HashMap::new();
    for chunk in page {
        if let Some((model, vector)) = &chunk.sparse {
            by_model
                .entry(model.as_str())
                .or_default()
                .push((chunk.doc.id.clone(), vector.clone()));
        }
    }
    for (model, rows) in by_model {
        store.set_sparse(model, &rows)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::SparseVector;
    use crate::vectorstore::memory::InMemoryStore;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    /// Embeds text as `[len, salt]`; counts texts; can fail after N calls.
    struct Fake {
        salt: f32,
        texts: Mutex<Vec<String>>,
        fail_after_calls: Option<usize>,
        calls: AtomicUsize,
    }

    impl Fake {
        fn new(salt: f32) -> Self {
            Self {
                salt,
                texts: Mutex::new(Vec::new()),
                fail_after_calls: None,
                calls: AtomicUsize::new(0),
            }
        }
        fn embedded(&self) -> Vec<String> {
            self.texts.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl Embedder for Fake {
        fn id(&self) -> &str {
            "fake"
        }
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            if self.fail_after_calls.is_some_and(|limit| call >= limit) {
                return Err(MinSyncError::Embedding("injected".into()));
            }
            self.texts.lock().unwrap().extend(texts.iter().cloned());
            Ok(texts
                .iter()
                .map(|text| vec![text.len() as f32, self.salt])
                .collect())
        }
    }

    fn doc(id: &str, text: &str, hash: &str) -> Document {
        Document {
            id: id.to_string(),
            embedding: vec![9.0, 9.0, 9.0],
            text: text.to_string(),
            source_id: "s".to_string(),
            path: format!("{id}.txt"),
            chunk_schema_id: "schema".to_string(),
            chunk_type: "text".to_string(),
            heading_path: format!("h-{id}"),
            content_hash: hash.to_string(),
            seen_token: "token".to_string(),
        }
    }

    fn live(count: usize) -> InMemoryStore {
        let mut store = InMemoryStore::new();
        let docs: Vec<Document> = (0..count)
            .map(|i| {
                doc(
                    &format!("id{i:04}"),
                    &format!("text {i}"),
                    &format!("hash{i}"),
                )
            })
            .collect();
        store.upsert(&docs).unwrap();
        let sparse: Vec<(String, SparseVector)> = (0..count)
            .map(|i| {
                (
                    format!("id{i:04}"),
                    SparseVector::new(vec![i as u32 + 1], vec![1.0]).unwrap(),
                )
            })
            .collect();
        store.set_sparse("sparse-model", &sparse).unwrap();
        store
    }

    fn config(dimension: i64) -> Config {
        let mut config = Config::default_for("s");
        config.embedder.id = "tei:fake".to_string();
        let mut options = toml::value::Table::new();
        options.insert("dimension".into(), toml::Value::Integer(dimension));
        config.vectorstore.options = toml::Value::Table(options);
        config
    }

    /// Hands the engine one shared in-memory store per staging path, so a
    /// second run sees what the first one left in "staging".
    type Stores = Arc<Mutex<HashMap<PathBuf, Arc<Mutex<InMemoryStore>>>>>;

    fn stores() -> (Stores, impl Fn(&Path) -> Result<Box<dyn VectorStore>>) {
        struct Shared(Arc<Mutex<InMemoryStore>>);
        impl VectorStore for Shared {
            fn upsert(&mut self, docs: &[Document]) -> Result<()> {
                self.0.lock().unwrap().upsert(docs)
            }
            fn update(&mut self, u: &[crate::vectorstore::DocumentUpdate]) -> Result<()> {
                self.0.lock().unwrap().update(u)
            }
            fn fetch(&self, ids: &[String]) -> Result<Vec<Document>> {
                self.0.lock().unwrap().fetch(ids)
            }
            fn scan_chunks(&self, after: Option<&str>, limit: usize) -> Result<Vec<StoredChunk>> {
                self.0.lock().unwrap().scan_chunks(after, limit)
            }
            fn delete_by_filter(&mut self, f: &crate::vectorstore::Filter) -> Result<usize> {
                self.0.lock().unwrap().delete_by_filter(f)
            }
            fn query(
                &self,
                v: &[f32],
                f: Option<&crate::vectorstore::Filter>,
                k: usize,
            ) -> Result<Vec<crate::vectorstore::QueryHit>> {
                self.0.lock().unwrap().query(v, f, k)
            }
            fn rows_missing_sparse(
                &self,
                m: &str,
                s: Option<&crate::vectorstore::Filter>,
                l: usize,
            ) -> Result<Vec<crate::vectorstore::SparseTarget>> {
                self.0.lock().unwrap().rows_missing_sparse(m, s, l)
            }
            fn count_missing_sparse(&self, m: &str) -> Result<usize> {
                self.0.lock().unwrap().count_missing_sparse(m)
            }
            fn set_sparse(&mut self, m: &str, r: &[(String, SparseVector)]) -> Result<()> {
                self.0.lock().unwrap().set_sparse(m, r)
            }
            fn fetch_sparse_by_content_hash(
                &self,
                m: &str,
                h: &[String],
            ) -> Result<HashMap<String, SparseVector>> {
                self.0.lock().unwrap().fetch_sparse_by_content_hash(m, h)
            }
            fn query_sparse(
                &self,
                m: &str,
                v: &SparseVector,
                f: Option<&crate::vectorstore::Filter>,
                k: usize,
            ) -> Result<Vec<crate::vectorstore::QueryHit>> {
                self.0.lock().unwrap().query_sparse(m, v, f, k)
            }
            fn flush(&mut self) -> Result<()> {
                Ok(())
            }
            fn doc_count(&self) -> usize {
                self.0.lock().unwrap().doc_count()
            }
            fn all_paths(&self) -> Vec<String> {
                self.0.lock().unwrap().all_paths()
            }
        }
        let map: Stores = Arc::default();
        let opener = {
            let map = map.clone();
            move |path: &Path| -> Result<Box<dyn VectorStore>> {
                // A real store creates its directory on open.
                std::fs::create_dir_all(path)?;
                let store = map
                    .lock()
                    .unwrap()
                    .entry(path.to_path_buf())
                    .or_default()
                    .clone();
                Ok(Box::new(Shared(store)))
            }
        };
        (map, opener)
    }

    fn workspace() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let live_path = dir.path().join("store");
        std::fs::create_dir_all(&live_path).unwrap();
        (dir, live_path)
    }

    #[tokio::test]
    async fn every_row_is_re_embedded_and_keeps_its_text_ids_and_sparse_vector() {
        let (dir, live_path) = workspace();
        let source = live(5);
        let (map, open) = stores();
        let embedder = Fake::new(7.0);

        let (outcome, target) = rebuild(
            dir.path(),
            &config(2),
            &embedder,
            &live_path,
            &source,
            &open,
            |_| {},
        )
        .await
        .unwrap();

        assert_eq!(outcome.embedded, 5);
        assert_eq!(target.dimension, 2);
        let staging = dir.path().join("store.reembed");
        let staging_store = map.lock().unwrap()[&staging].clone();
        let staging_store = staging_store.lock().unwrap();
        let rebuilt = staging_store.documents().unwrap();
        assert_eq!(rebuilt.len(), 5);
        for doc in &rebuilt {
            let original = source
                .fetch(std::slice::from_ref(&doc.id))
                .unwrap()
                .remove(0);
            assert_eq!(doc.text, original.text);
            assert_eq!(doc.path, original.path);
            assert_eq!(doc.heading_path, original.heading_path);
            assert_eq!(doc.seen_token, original.seen_token);
            assert_eq!(
                doc.embedding,
                vec![doc.text.len() as f32, 7.0],
                "new vector"
            );
        }
        assert_eq!(
            staging_store.count_missing_sparse("sparse-model").unwrap(),
            0
        );
    }

    #[tokio::test]
    async fn identical_chunk_text_is_embedded_once_per_page() {
        let (dir, live_path) = workspace();
        let mut source = InMemoryStore::new();
        source
            .upsert(&[
                doc("a", "same words", "same-hash"),
                doc("b", "same words", "same-hash"),
                doc("c", "other words", "other-hash"),
            ])
            .unwrap();
        let (_map, open) = stores();
        let embedder = Fake::new(1.0);

        let (outcome, _) = rebuild(
            dir.path(),
            &config(2),
            &embedder,
            &live_path,
            &source,
            &open,
            |_| {},
        )
        .await
        .unwrap();

        assert_eq!(outcome.embedded, 3);
        assert_eq!(embedder.embedded().len(), 2, "two distinct texts");
    }

    #[tokio::test]
    async fn an_interrupted_rebuild_resumes_after_the_last_committed_page() {
        let (dir, live_path) = workspace();
        let source = live(PAGE * 2 + 10);
        let (_map, open) = stores();

        let mut failing = Fake::new(3.0);
        failing.fail_after_calls = Some(1);
        let error = rebuild(
            dir.path(),
            &config(2),
            &failing,
            &live_path,
            &source,
            &open,
            |_| {},
        )
        .await
        .unwrap_err();
        assert!(matches!(error, MinSyncError::Embedding(_)));
        assert_eq!(failing.embedded().len(), PAGE, "one page got through");
        assert!(load_marker(dir.path()).is_some(), "the rebuild is recorded");

        let healthy = Fake::new(3.0);
        let (outcome, _) = rebuild(
            dir.path(),
            &config(2),
            &healthy,
            &live_path,
            &source,
            &open,
            |_| {},
        )
        .await
        .unwrap();
        assert_eq!(
            healthy.embedded().len(),
            PAGE + 10,
            "only what the first run did not finish"
        );
        assert_eq!(outcome.resumed, PAGE);
    }

    #[tokio::test]
    async fn a_leftover_rebuild_toward_another_embedder_is_discarded_not_continued() {
        let (dir, live_path) = workspace();
        let source = live(5);
        let (map, open) = stores();

        let mut first = Fake::new(1.0);
        first.fail_after_calls = Some(0);
        let _ = rebuild(
            dir.path(),
            &config(2),
            &first,
            &live_path,
            &source,
            &open,
            |_| {},
        )
        .await;
        // Same engine, but now the configured embedder is another width.
        let healthy = Fake::new(2.0);
        let (outcome, target) = rebuild(
            dir.path(),
            &config(3),
            &healthy,
            &live_path,
            &source,
            &open,
            |_| {},
        )
        .await
        .unwrap();

        assert_eq!(target.dimension, 3);
        assert_eq!(outcome.embedded, 5, "started over");
        assert_eq!(outcome.resumed, 0);
        assert_eq!(healthy.embedded().len(), 5);
        drop(map);
    }

    /// Answers with a fixed vector, or fails, or returns what it is told.
    struct Probed {
        answer: std::sync::Mutex<Result<Vec<Vec<f32>>>>,
        calls: AtomicUsize,
        texts: Mutex<Vec<String>>,
    }

    impl Probed {
        fn answering(vectors: Vec<Vec<f32>>) -> Self {
            Self {
                answer: Mutex::new(Ok(vectors)),
                calls: AtomicUsize::new(0),
                texts: Mutex::new(Vec::new()),
            }
        }
        fn failing(message: &str) -> Self {
            Self {
                answer: Mutex::new(Err(MinSyncError::Embedding(message.to_string()))),
                calls: AtomicUsize::new(0),
                texts: Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait]
    impl Embedder for Probed {
        fn id(&self) -> &str {
            "probed"
        }
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.texts.lock().unwrap().extend(texts.iter().cloned());
            match &*self.answer.lock().unwrap() {
                Ok(vectors) => Ok(vectors.clone()),
                Err(error) => Err(MinSyncError::Embedding(error.to_string())),
            }
        }
    }

    /// The sparse-only embedder refuses `embed` and says so.
    struct SparseOnly;

    #[async_trait]
    impl Embedder for SparseOnly {
        fn id(&self) -> &str {
            crate::config::NONE_EMBEDDER_ID
        }
        fn is_sparse_only(&self) -> bool {
            true
        }
        async fn embed(&self, _: &[String]) -> Result<Vec<Vec<f32>>> {
            Err(MinSyncError::Embedding("no dense embedder".into()))
        }
    }

    #[tokio::test]
    async fn a_sparse_only_target_writes_placeholders_and_never_calls_the_embedder() {
        let (dir, live_path) = workspace();
        let source = live(4);
        let (map, open) = stores();
        let mut none = config(1);
        none.embedder.id = crate::config::NONE_EMBEDDER_ID.to_string();

        let (outcome, target) = rebuild(
            dir.path(),
            &none,
            &SparseOnly,
            &live_path,
            &source,
            &open,
            |_| {},
        )
        .await
        .unwrap();

        assert_eq!(outcome.embedded, 4);
        assert_eq!(outcome.api_calls, 0);
        assert_eq!(target.dimension, crate::config::SPARSE_ONLY_DIMENSION);
        let staging = map.lock().unwrap()[&dir.path().join("store.reembed")].clone();
        for doc in staging.lock().unwrap().documents().unwrap() {
            assert_eq!(
                doc.embedding,
                vec![0.0; crate::config::SPARSE_ONLY_DIMENSION]
            );
        }
    }

    #[tokio::test]
    async fn the_probe_reports_the_width_the_model_really_produces() {
        let embedder = Probed::answering(vec![vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]);
        let identity = probe(&config(6), &embedder).await.unwrap();
        assert_eq!(identity.dimension, 6);
        assert_eq!(identity.model, "probed");
        assert_eq!(embedder.calls.load(Ordering::SeqCst), 1);
        assert_eq!(embedder.texts.lock().unwrap().len(), 1, "one short text");
    }

    #[tokio::test]
    async fn a_configured_width_the_model_does_not_produce_names_both_and_the_setting() {
        let embedder = Probed::answering(vec![vec![1.0; 6]]);
        let error = probe(&config(5), &embedder).await.unwrap_err().to_string();
        assert!(error.contains('6') && error.contains('5'), "{error}");
        assert!(error.contains("vectorstore.options"), "{error}");
        assert!(error.contains("dimension"), "{error}");
    }

    #[tokio::test]
    async fn a_failing_embedder_is_reported_as_the_probe_failing() {
        let embedder = Probed::failing("model not loaded");
        let error = probe(&config(4), &embedder).await.unwrap_err();
        assert!(matches!(error, MinSyncError::Embedding(_)), "{error}");
        let message = error.to_string();
        assert!(
            message.contains("probe") && message.contains("model not loaded"),
            "{message}"
        );
    }

    #[tokio::test]
    async fn empty_wrong_count_and_non_finite_answers_are_refused() {
        for answer in [
            vec![],
            vec![vec![1.0; 4], vec![1.0; 4]],
            vec![vec![]],
            vec![vec![1.0, f32::NAN, 1.0, 1.0]],
            vec![vec![1.0, f32::INFINITY, 1.0, 1.0]],
        ] {
            let embedder = Probed::answering(answer.clone());
            assert!(
                probe(&config(4), &embedder).await.is_err(),
                "{answer:?} must be refused"
            );
        }
    }

    #[tokio::test]
    async fn a_sparse_only_target_needs_no_probe() {
        let mut none = config(1);
        none.embedder.id = crate::config::NONE_EMBEDDER_ID.to_string();
        let embedder = Probed::failing("must not be called");
        let identity = probe(&none, &embedder).await.unwrap();
        assert_eq!(identity.dimension, crate::config::SPARSE_ONLY_DIMENSION);
        assert_eq!(embedder.calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn staged_rows_counts_what_an_interrupted_rebuild_already_moved() {
        let (dir, live_path) = workspace();
        let source = live(PAGE * 2 + 10);
        let (_map, open) = stores();
        let target_config = config(2);

        // No rebuild yet.
        let target = EmbedderIdentity::from_config(&target_config).unwrap();
        assert_eq!(staged_rows(dir.path(), &target, &open).unwrap(), None);

        let mut failing = Fake::new(3.0);
        failing.fail_after_calls = Some(1);
        let _ = rebuild(
            dir.path(),
            &target_config,
            &failing,
            &live_path,
            &source,
            &open,
            |_| {},
        )
        .await;
        assert_eq!(
            staged_rows(dir.path(), &target, &open).unwrap(),
            Some(PAGE),
            "one committed page"
        );

        // A rebuild toward another embedder is not this one's progress.
        let other = EmbedderIdentity::from_config(&config(3)).unwrap();
        assert_eq!(staged_rows(dir.path(), &other, &open).unwrap(), None);
    }

    #[test]
    fn swapping_replaces_the_live_store_and_clears_the_marker() {
        let dir = tempfile::tempdir().unwrap();
        let minsync = dir.path();
        let live = minsync.join("store");
        let staging = minsync.join("store.reembed");
        std::fs::create_dir_all(&live).unwrap();
        std::fs::create_dir_all(&staging).unwrap();
        std::fs::write(live.join("old"), "old").unwrap();
        std::fs::write(staging.join("new"), "new").unwrap();
        save_marker(
            minsync,
            &Marker {
                target: EmbedderIdentity {
                    model: "m".into(),
                    dimension: 2,
                    passage_prefix: None,
                },
                staging: "store.reembed".into(),
            },
        )
        .unwrap();

        swap_in(minsync, &live).unwrap();

        assert!(live.join("new").exists());
        assert!(!live.join("old").exists());
        assert!(!staging.exists());
        assert!(
            load_marker(minsync).is_some(),
            "the marker outlives the swap until the cursor records it"
        );
        // A second call (a run killed before it could save the cursor) is a no-op.
        swap_in(minsync, &live).unwrap();
        assert!(live.join("new").exists());
        finish(minsync);
        assert!(load_marker(minsync).is_none());
    }
}
