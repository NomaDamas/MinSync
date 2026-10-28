//! Environment health check for `minsync check`.

use crate::config::CheckConfig;
use crate::embedder::Embedder;
use crate::error::{MinSyncError, Result};
use crate::incremental::journal::{Journal, JOURNAL_FILE};
use crate::types::CheckResult;
use crate::vectorstore::VectorStore;
use std::io::Write;
use std::path::Path;
use std::time::Duration;

/// Probe the embedder, the vector store and its volume, and report an
/// interrupted run.
///
/// `store_path` is the store's directory; it is used for the writability and
/// free-space probes. A failed probe is reported in the result, not returned
/// as an error, so the caller always gets the full report.
pub async fn check(
    minsync_dir: &Path,
    store_path: &Path,
    settings: &CheckConfig,
    embedder: &dyn Embedder,
    store: &dyn VectorStore,
) -> Result<CheckResult> {
    if !minsync_dir.join("config.toml").exists() {
        return Err(MinSyncError::NotInitialized);
    }

    let mut errors = Vec::new();
    let mut warnings = Vec::new();
    // A BM25-only workspace has no model to probe.
    let embedder_ok = if embedder.is_lexical_only() {
        true
    } else {
        let probe = embedder.embed_single("hello");
        let outcome = if settings.timeout_seconds == 0 {
            Some(probe.await)
        } else {
            tokio::time::timeout(Duration::from_secs(settings.timeout_seconds), probe)
                .await
                .ok()
        };
        match outcome {
            Some(Ok(vector)) if vector.is_empty() => {
                errors.push("embedder returned empty embedding".to_string());
                false
            }
            Some(Ok(_)) => true,
            Some(Err(error)) => {
                errors.push(format!("embedder check failed: {error}"));
                false
            }
            None => {
                errors.push(format!(
                    "embedder check failed: no answer within {}s ([check] timeout_seconds)",
                    settings.timeout_seconds
                ));
                false
            }
        }
    };

    let mut vectorstore_ok = true;
    let index_state = match store.index_state() {
        Ok(state) => state,
        Err(error) => {
            errors.push(format!("vector store check failed: {error}"));
            vectorstore_ok = false;
            None
        }
    };

    let probe_dir = existing_dir(store_path);
    if let Err(error) = probe_writable(probe_dir) {
        errors.push(format!(
            "vector store check failed: {} is not writable: {error}",
            probe_dir.display()
        ));
        vectorstore_ok = false;
    }

    let free_bytes = fs2::available_space(probe_dir).ok();
    if settings.min_free_bytes > 0 {
        if let Some(free) = free_bytes.filter(|free| *free < settings.min_free_bytes) {
            errors.push(format!(
                "vector store check failed: {} bytes free on the store's volume, \
                 below [check] min_free_bytes = {}; free space and re-run",
                free, settings.min_free_bytes
            ));
            vectorstore_ok = false;
        }
    }

    let interrupted = minsync_dir.join("txn.json").exists();
    let resumable = interrupted && {
        let progress_path = minsync_dir.join(JOURNAL_FILE);
        Journal::load(&progress_path)
            .into_ledger(&progress_path)
            .is_some()
    };
    if interrupted {
        warnings.push(if resumable {
            "an earlier sync was interrupted; the next `minsync sync` resumes it".to_string()
        } else {
            "an earlier sync was interrupted before it logged progress; \
             the next `minsync sync` redoes it"
                .to_string()
        });
    }

    Ok(CheckResult {
        embedder_ok,
        vectorstore_ok,
        all_passed: embedder_ok && vectorstore_ok && errors.is_empty(),
        errors,
        index_state,
        interrupted,
        resumable,
        free_bytes,
        warnings,
    })
}

/// `path` itself when it is a directory, else its nearest existing ancestor
/// (a store that has not been written yet is created there).
fn existing_dir(path: &Path) -> &Path {
    path.ancestors()
        .find(|candidate| candidate.is_dir())
        .unwrap_or(path)
}

/// Create, write, sync and remove a temporary file in `dir`.
fn probe_writable(dir: &Path) -> std::io::Result<()> {
    let mut file = tempfile::Builder::new()
        .prefix(".minsync-check-")
        .tempfile_in(dir)?;
    file.write_all(b"minsync check\n")?;
    file.as_file().sync_all()?;
    file.close()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::IndexState;
    use crate::vectorstore::{Document, DocumentUpdate, Filter, QueryHit};
    use async_trait::async_trait;

    struct StubEmbedder {
        hang: bool,
    }

    #[async_trait]
    impl Embedder for StubEmbedder {
        fn id(&self) -> &str {
            "stub"
        }

        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            if self.hang {
                std::future::pending::<()>().await;
            }
            Ok(texts.iter().map(|_| vec![0.5; 4]).collect())
        }
    }

    struct StubStore {
        broken: bool,
    }

    impl VectorStore for StubStore {
        fn upsert(&mut self, _docs: &[Document]) -> Result<()> {
            Ok(())
        }
        fn update(&mut self, _updates: &[DocumentUpdate]) -> Result<()> {
            Ok(())
        }
        fn fetch(&self, _ids: &[String]) -> Result<Vec<Document>> {
            Ok(Vec::new())
        }
        fn delete_by_filter(&mut self, _filter: &Filter) -> Result<usize> {
            Ok(0)
        }
        fn query(&self, _: &[f32], _: Option<&Filter>, _: usize) -> Result<Vec<QueryHit>> {
            Ok(Vec::new())
        }
        fn query_text(&self, _: &str, _: Option<&Filter>, _: usize) -> Result<Vec<QueryHit>> {
            Ok(Vec::new())
        }
        fn flush(&mut self) -> Result<()> {
            Ok(())
        }
        fn doc_count(&self) -> usize {
            0
        }
        fn all_paths(&self) -> Vec<String> {
            Vec::new()
        }
        fn index_state(&self) -> Result<Option<IndexState>> {
            if self.broken {
                return Err(MinSyncError::VectorStore("table is corrupt".to_string()));
            }
            Ok(Some(IndexState {
                fts_indexed: true,
                ann_indexed: false,
                unindexed_rows: Some(0),
            }))
        }
    }

    /// A `.minsync` directory with a config file and a `store` directory.
    fn workspace() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.toml"), "").unwrap();
        std::fs::create_dir(dir.path().join("store")).unwrap();
        dir
    }

    async fn run(
        dir: &Path,
        settings: &CheckConfig,
        embedder: &StubEmbedder,
        store: &StubStore,
    ) -> CheckResult {
        check(dir, &dir.join("store"), settings, embedder, store)
            .await
            .expect("check")
    }

    fn no_disk_limit() -> CheckConfig {
        CheckConfig {
            min_free_bytes: 0,
            ..CheckConfig::default()
        }
    }

    #[tokio::test]
    async fn a_healthy_workspace_passes_and_reports_free_space() {
        let dir = workspace();
        let result = run(
            dir.path(),
            &no_disk_limit(),
            &StubEmbedder { hang: false },
            &StubStore { broken: false },
        )
        .await;
        assert!(result.all_passed, "{result:?}");
        assert!(result.free_bytes.is_some());
        assert!(!result.interrupted && !result.resumable);
        assert!(result.warnings.is_empty());
        // The writability probe cleans up after itself.
        assert_eq!(
            std::fs::read_dir(dir.path().join("store")).unwrap().count(),
            0
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_hung_embedder_fails_after_the_timeout() {
        let dir = workspace();
        let settings = CheckConfig {
            timeout_seconds: 3,
            ..no_disk_limit()
        };
        let result = run(
            dir.path(),
            &settings,
            &StubEmbedder { hang: true },
            &StubStore { broken: false },
        )
        .await;
        assert!(!result.embedder_ok);
        assert!(result.vectorstore_ok);
        assert!(!result.all_passed);
        assert!(
            result.errors[0].contains("no answer within 3s"),
            "{:?}",
            result.errors
        );
    }

    #[tokio::test]
    async fn a_store_error_is_reported_instead_of_discarded() {
        let dir = workspace();
        let result = run(
            dir.path(),
            &no_disk_limit(),
            &StubEmbedder { hang: false },
            &StubStore { broken: true },
        )
        .await;
        assert!(result.embedder_ok);
        assert!(!result.vectorstore_ok);
        assert!(!result.all_passed);
        assert!(result.index_state.is_none());
        assert!(
            result.errors[0].contains("table is corrupt"),
            "{:?}",
            result.errors
        );
    }

    #[tokio::test]
    async fn too_little_free_space_fails_the_store_check() {
        let dir = workspace();
        let settings = CheckConfig {
            min_free_bytes: u64::MAX,
            ..CheckConfig::default()
        };
        let result = run(
            dir.path(),
            &settings,
            &StubEmbedder { hang: false },
            &StubStore { broken: false },
        )
        .await;
        assert!(!result.vectorstore_ok);
        assert!(result.free_bytes.is_some());
        assert!(
            result.errors[0].contains("min_free_bytes"),
            "{:?}",
            result.errors
        );
    }

    #[tokio::test]
    async fn an_interrupted_run_is_reported_as_a_warning() {
        let dir = workspace();
        std::fs::write(dir.path().join("txn.json"), r#"{"status":"running"}"#).unwrap();
        let result = run(
            dir.path(),
            &no_disk_limit(),
            &StubEmbedder { hang: false },
            &StubStore { broken: false },
        )
        .await;
        assert!(result.interrupted);
        // No progress log: the next sync redoes the run.
        assert!(!result.resumable);
        assert_eq!(result.warnings.len(), 1, "{:?}", result.warnings);
        assert!(
            result.all_passed,
            "an interrupted run alone is not unhealthy"
        );
    }

    #[test]
    fn existing_dir_walks_up_to_a_directory() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("store/not/yet");
        assert_eq!(existing_dir(&missing), dir.path());
        assert_eq!(existing_dir(dir.path()), dir.path());
    }

    #[test]
    fn probe_writable_leaves_nothing_behind() {
        let dir = tempfile::tempdir().unwrap();
        probe_writable(dir.path()).expect("writable");
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    #[test]
    fn probe_writable_fails_on_a_missing_directory() {
        let dir = tempfile::tempdir().unwrap();
        assert!(probe_writable(&dir.path().join("missing")).is_err());
    }
}
