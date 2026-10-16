//! Deadline for non-cancellable blocking work (native model forward passes).
//!
//! `spawn_blocking` tasks cannot be aborted: dropping the join handle leaves
//! the thread running. A stuck candle/Metal kernel therefore wedges its
//! thread — and the model mutex it holds — for good. [`BatchDeadline`]
//! bounds how long the *caller* waits and remembers that a thread was
//! abandoned, so later calls fail fast instead of queueing behind it. The
//! stuck thread only goes away with the process, which is why callers exit
//! with a distinct code ([`MinSyncError::Timeout`]) and rely on the sync
//! resume log to continue.

use crate::error::{MinSyncError, Result};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

#[derive(Default)]
pub(crate) struct BatchDeadline {
    wedged: AtomicBool,
}

impl BatchDeadline {
    pub(crate) const fn new() -> Self {
        Self {
            wedged: AtomicBool::new(false),
        }
    }

    /// Fail fast if an earlier call already abandoned a stuck thread.
    pub(crate) fn ensure_healthy(&self, what: &str) -> Result<()> {
        if self.wedged.load(Ordering::Acquire) {
            return Err(MinSyncError::Timeout(format!(
                "{what}: an earlier call timed out and its thread is still stuck; restart minsync"
            )));
        }
        Ok(())
    }

    /// Run `task` on the blocking pool and wait at most `budget`.
    ///
    /// `what` names the work in the error message. After one expiry every
    /// later call returns [`MinSyncError::Timeout`] without running `task`.
    pub(crate) async fn run<T, F>(&self, what: &str, budget: Duration, task: F) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce() -> Result<T> + Send + 'static,
    {
        self.ensure_healthy(what)?;
        match tokio::time::timeout(budget, tokio::task::spawn_blocking(task)).await {
            Ok(joined) => joined.map_err(|error| {
                MinSyncError::Embedding(format!("{what}: blocking task failed: {error}"))
            })?,
            Err(_) => {
                self.wedged.store(true, Ordering::Release);
                Err(MinSyncError::Timeout(format!(
                    "{what} exceeded {budget:?}; its thread cannot be cancelled — re-run `minsync sync` to resume"
                )))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;
    use std::time::Instant;

    const LONG: Duration = Duration::from_secs(5);

    #[tokio::test]
    async fn returns_timeout_without_waiting_for_the_stuck_task() {
        let deadline = BatchDeadline::new();
        let started = Instant::now();

        let outcome = deadline
            .run("native embed", Duration::from_millis(50), || {
                std::thread::sleep(Duration::from_millis(600));
                Ok(1)
            })
            .await;

        let elapsed = started.elapsed();
        assert!(
            matches!(outcome, Err(MinSyncError::Timeout(_))),
            "{outcome:?}"
        );
        assert!(elapsed < Duration::from_millis(400), "waited {elapsed:?}");
    }

    #[tokio::test]
    async fn fast_task_returns_its_value_and_does_not_wedge() {
        let deadline = BatchDeadline::new();

        assert_eq!(deadline.run("embed", LONG, || Ok(7)).await.unwrap(), 7);
        assert_eq!(deadline.run("embed", LONG, || Ok(8)).await.unwrap(), 8);
    }

    #[tokio::test]
    async fn task_error_passes_through_unchanged() {
        let deadline = BatchDeadline::new();

        let outcome: Result<()> = deadline
            .run("embed", LONG, || {
                Err(MinSyncError::Embedding("boom".into()))
            })
            .await;

        assert!(matches!(outcome, Err(MinSyncError::Embedding(message)) if message == "boom"));
    }

    #[tokio::test]
    async fn calls_after_a_timeout_fail_fast_without_running() {
        let deadline = BatchDeadline::new();
        let ran = Arc::new(AtomicUsize::new(0));
        let _ = deadline
            .run("embed", Duration::from_millis(30), || {
                std::thread::sleep(Duration::from_millis(300));
                Ok(())
            })
            .await;

        let counter = ran.clone();
        let started = Instant::now();
        let outcome = deadline
            .run("embed", LONG, move || {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })
            .await;

        assert!(matches!(outcome, Err(MinSyncError::Timeout(_))));
        assert_eq!(ran.load(Ordering::SeqCst), 0, "task must not be queued");
        assert!(started.elapsed() < Duration::from_millis(250));
    }
}
