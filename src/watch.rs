//! File-watching incremental indexing.
//!
//! Watches the project tree with `notify-debouncer-full` (0.8.0-rc.2) and,
//! whenever a non-internal path changes, re-runs the existing incremental
//! [`MinSync::sync`] pipeline. The watcher never builds its own indexing path —
//! it merely decides *when* to ask `sync()` to re-diff the manifest.
//!
//! ## notify-debouncer-full 0.8.0-rc.2 API used
//!
//! - [`new_debouncer`]`(timeout: Duration, tick_rate: Option<Duration>, handler) ->
//!   Result<Debouncer<RecommendedWatcher, RecommendedCache>, notify::Error>`
//! - [`tokio::sync::mpsc::UnboundedSender<DebounceEventResult>`] implements
//!   `DebounceEventHandler` (enabled by the crate's `tokio` feature), so the
//!   sender is passed directly as the handler.
//! - `DebounceEventResult = Result<Vec<DebouncedEvent>, Vec<notify::Error>>`.
//! - `DebouncedEvent` derefs to `notify::Event`, exposing `.paths: Vec<PathBuf>`
//!   and `.need_rescan() -> bool`.
//! - `RecursiveMode` is re-exported at `notify_debouncer_full::notify::RecursiveMode`.
//! - `Debouncer::watch(path, RecursiveMode::Recursive) -> notify::Result<()>`.
//! - The `Debouncer` stops its background thread on drop (`Drop` impl).

use std::path::{Path, PathBuf};
use std::time::Duration;

use notify_debouncer_full::new_debouncer;
use notify_debouncer_full::notify::RecursiveMode;
use notify_debouncer_full::DebounceEventResult;
use std::future::Future;
use std::pin::Pin;
use tokio::sync::mpsc;
use tokio::sync::oneshot;

use crate::chunker::Chunker;
use crate::embedder::Embedder;
use crate::error::{MinSyncError, Result};
use crate::sync::MinSync;
use crate::vectorstore::VectorStore;

/// Decide whether a changed path should trigger indexing.
///
/// Pure function (no I/O):
/// - returns `false` if `path` is inside `minsync_dir`,
/// - returns `false` if any path component equals `.git`,
/// - returns `true` otherwise.
pub fn should_index(path: &Path, minsync_dir: &Path) -> bool {
    if path.starts_with(minsync_dir) {
        return false;
    }

    !path
        .components()
        .any(|component| component.as_os_str() == ".git")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WatchStartup {
    FailFast,
    ContinueOnSyncError,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WatchStartupStatus {
    InitialSyncSucceeded,
    InitialSyncFailed(String),
}

pub struct WatchControl {
    pub startup: WatchStartup,
    pub progress: Option<mpsc::UnboundedSender<crate::types::SyncResult>>,
    pub startup_status: Option<mpsc::UnboundedSender<WatchStartupStatus>>,
    pub shutdown: oneshot::Receiver<()>,
}

struct WatchRuntime {
    startup: WatchStartup,
    progress: Option<mpsc::UnboundedSender<crate::types::SyncResult>>,
    startup_status: Option<mpsc::UnboundedSender<WatchStartupStatus>>,
    shutdown: Pin<Box<dyn Future<Output = ()> + Send>>,
}

/// Run the file-watch loop until Ctrl-C.
///
/// On any non-internal path change (or a debouncer rescan signal), this
/// re-runs [`MinSync::sync`] incrementally. The store borrow is held across the
/// loop and `sync()` is called sequentially, so the `&mut dyn VectorStore`
/// never overlaps an await of itself.
pub async fn run(
    root: PathBuf,
    chunker: &dyn Chunker,
    embedder: &dyn Embedder,
    store: &mut dyn VectorStore,
    debounce_ms: Option<u64>,
    startup: WatchStartup,
) -> Result<()> {
    let _lock = crate::state::FileLock::acquire(&root.join(".minsync/lock"), false)?;
    let runtime = WatchRuntime {
        startup,
        progress: None,
        startup_status: None,
        shutdown: Box::pin(async {
            let _ = tokio::signal::ctrl_c().await;
        }),
    };
    run_inner(root, chunker, embedder, store, debounce_ms, runtime, true).await
}

pub(crate) async fn run_locked(
    root: PathBuf,
    chunker: &dyn Chunker,
    embedder: &dyn Embedder,
    store: &mut dyn VectorStore,
    debounce_ms: Option<u64>,
    startup: WatchStartup,
) -> Result<()> {
    let runtime = WatchRuntime {
        startup,
        progress: None,
        startup_status: None,
        shutdown: Box::pin(async {
            let _ = tokio::signal::ctrl_c().await;
        }),
    };
    run_inner(root, chunker, embedder, store, debounce_ms, runtime, true).await
}

pub async fn run_with_shutdown(
    root: PathBuf,
    chunker: &dyn Chunker,
    embedder: &dyn Embedder,
    store: &mut dyn VectorStore,
    debounce_ms: Option<u64>,
    control: WatchControl,
) -> Result<()> {
    let _lock = crate::state::FileLock::acquire(&root.join(".minsync/lock"), false)?;
    let runtime = WatchRuntime {
        startup: control.startup,
        progress: control.progress,
        startup_status: control.startup_status,
        shutdown: Box::pin(async {
            let _ = control.shutdown.await;
        }),
    };
    run_inner(root, chunker, embedder, store, debounce_ms, runtime, true).await
}

async fn run_inner(
    root: PathBuf,
    chunker: &dyn Chunker,
    embedder: &dyn Embedder,
    store: &mut dyn VectorStore,
    debounce_ms: Option<u64>,
    mut runtime: WatchRuntime,
    lock_held: bool,
) -> Result<()> {
    let minsync_dir = root.join(".minsync");
    let debounce = Duration::from_millis(debounce_ms.unwrap_or(500));

    let (tx, mut rx) = mpsc::unbounded_channel::<DebounceEventResult>();

    let mut debouncer = new_debouncer(debounce, None, tx)
        .map_err(|error| MinSyncError::Other(format!("failed to create file watcher: {error}")))?;

    debouncer
        .watch(&root, RecursiveMode::Recursive)
        .map_err(|error| {
            MinSyncError::Other(format!("failed to watch {}: {error}", root.display()))
        })?;

    tracing::info!("Watching {} for indexable changes", root.display());

    let sync = MinSync::new(root.clone());

    let initial = run_sync(&sync, chunker, embedder, store, lock_held).await;
    if let Ok(result) = initial.as_ref() {
        if let Some(status) = &runtime.startup_status {
            let _ = status.send(WatchStartupStatus::InitialSyncSucceeded);
        }
        if let Some(progress) = &runtime.progress {
            let _ = progress.send(result.clone());
        }
    }
    if let Err(error) = initial {
        match runtime.startup {
            WatchStartup::FailFast => return Err(error),
            WatchStartup::ContinueOnSyncError => {
                if let Some(status) = &runtime.startup_status {
                    let _ = status.send(WatchStartupStatus::InitialSyncFailed(error.to_string()));
                }
                tracing::error!("initial watch sync failed; waiting for file changes: {error}");
            }
        }
    }

    loop {
        tokio::select! {
            _ = &mut runtime.shutdown => break,
            received = rx.recv() => {
                let Some(result) = received else {
                    break;
                };

                match result {
                    Ok(events) => {
                        let needs_rescan = events.iter().any(|event| event.need_rescan());
                        let relevant = needs_rescan
                            || events.iter().any(|event| {
                                event
                                    .paths
                                    .iter()
                                    .any(|path| should_index(path, &minsync_dir))
                            });

                        if relevant {
                            if needs_rescan {
                                tracing::info!("rescan requested, running incremental sync");
                            }
                            match run_sync(&sync, chunker, embedder, store, lock_held).await {
                                Ok(result) => {
                                    if let Some(progress) = &runtime.progress {
                                        let _ = progress.send(result);
                                    }
                                }
                                Err(error) => tracing::error!("watch sync failed: {error}"),
                            }
                        }
                    }
                    Err(errors) => {
                        for error in errors {
                            tracing::error!("watch error: {error}");
                        }
                    }
                }
            }
        }
    }

    drop(debouncer);
    Ok(())
}

/// Run one incremental sync and log the outcome.
async fn run_sync(
    sync: &MinSync,
    chunker: &dyn Chunker,
    embedder: &dyn Embedder,
    store: &mut dyn VectorStore,
    lock_held: bool,
) -> Result<crate::types::SyncResult> {
    let result = if lock_held {
        sync.sync_locked(chunker, embedder, store, false, false)
            .await?
    } else {
        sync.sync(chunker, embedder, store, false, false, false)
            .await?
    };
    {
        if result.already_up_to_date {
            tracing::info!("watch sync: already up to date");
        } else {
            tracing::info!(
                "watch sync: {} files processed; files added {}, modified {}, deleted {}; chunks added {}, updated {}, deleted {}",
                result.files_processed,
                result.files_added,
                result.files_modified,
                result.files_deleted,
                result.chunks_added,
                result.chunks_updated,
                result.chunks_deleted,
            );
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn root() -> PathBuf {
        PathBuf::from("/project")
    }

    fn minsync_dir() -> PathBuf {
        root().join(".minsync")
    }

    #[test]
    fn test_should_index_accepts_all_non_internal_paths() {
        let dir = minsync_dir();
        assert!(should_index(&root().join("a.md"), &dir));
        assert!(should_index(&root().join("b.txt"), &dir));
        assert!(should_index(&root().join("c.png"), &dir));
        assert!(should_index(&root().join("d.rs"), &dir));
        assert!(should_index(&root().join("e"), &dir));
    }

    #[test]
    fn test_should_index_rejects_minsync_dir() {
        let dir = minsync_dir();
        assert!(!should_index(&dir.join("manifest.json"), &dir));
        assert!(!should_index(&dir.join("x.md"), &dir));
    }

    #[test]
    fn test_should_index_rejects_git_component() {
        let dir = minsync_dir();
        assert!(!should_index(&root().join(".git/config"), &dir));
        assert!(!should_index(&root().join("nested/.git/index"), &dir));
    }
}
