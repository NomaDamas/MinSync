//! Long-lived handle on the native embedder for hosts that keep the model
//! resident inside their own process (the Node-API binding).
//!
//! It adds the lifecycle a one-shot CLI run never needs: explicit unload,
//! idle unload, and a disposed state. Inference itself is
//! [`NativeEmbedder`], so batching, the padded-token cap, the prefixes and the
//! per-batch deadline are exactly those `minsync sync` and `minsync query` use.

use crate::config::EmbedderConfig;
use crate::embedder::native::{NativeEmbedder, NativeIdentity};
use crate::embedder::Embedder;
use crate::error::{MinSyncError, Result};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Fires a callback once the embedder has been idle for a fixed time.
///
/// Every operation holds an [`Activity`] guard. When the last guard drops, a
/// timer starts; it fires only if nothing began in the meantime.
struct IdleTimer {
    idle: Duration,
    state: Arc<Mutex<IdleState>>,
    on_idle: Arc<dyn Fn() + Send + Sync>,
}

#[derive(Default)]
struct IdleState {
    in_flight: usize,
    /// Bumped on every begin/end so a superseded timer can tell it is stale.
    generation: u64,
}

struct Activity {
    state: Arc<Mutex<IdleState>>,
    idle: Duration,
    on_idle: Arc<dyn Fn() + Send + Sync>,
}

impl IdleTimer {
    fn new(idle: Duration, on_idle: Arc<dyn Fn() + Send + Sync>) -> Self {
        Self {
            idle,
            state: Arc::new(Mutex::new(IdleState::default())),
            on_idle,
        }
    }

    fn begin(&self) -> Activity {
        let mut state = lock(&self.state);
        state.in_flight += 1;
        state.generation += 1;
        Activity {
            state: Arc::clone(&self.state),
            idle: self.idle,
            on_idle: Arc::clone(&self.on_idle),
        }
    }
}

impl Drop for Activity {
    fn drop(&mut self) {
        let generation = {
            let mut state = lock(&self.state);
            state.in_flight -= 1;
            state.generation += 1;
            if state.in_flight > 0 {
                return;
            }
            state.generation
        };
        let state = Arc::clone(&self.state);
        let on_idle = Arc::clone(&self.on_idle);
        let idle = self.idle;
        tokio::spawn(async move {
            tokio::time::sleep(idle).await;
            let quiet = {
                let state = lock(&state);
                state.in_flight == 0 && state.generation == generation
            };
            if quiet {
                on_idle();
            }
        });
    }
}

fn lock(state: &Mutex<IdleState>) -> std::sync::MutexGuard<'_, IdleState> {
    state
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

pub struct NativeSession {
    embedder: Arc<NativeEmbedder>,
    idle: Option<IdleTimer>,
    disposed: AtomicBool,
}

impl NativeSession {
    /// Build a session from `[embedder]` settings. The model is not loaded
    /// until the first call that needs it. `idle_unload` releases the model
    /// after that long without an embed call; `None` keeps it resident until
    /// [`unload`](Self::unload) or [`dispose`](Self::dispose).
    pub fn new(settings: &EmbedderConfig, idle_unload: Option<Duration>) -> Result<Self> {
        if idle_unload == Some(Duration::ZERO) {
            return Err(MinSyncError::Config(
                "idle unload must be greater than 0".to_string(),
            ));
        }
        let embedder = Arc::new(NativeEmbedder::from_config(settings)?);
        let idle = idle_unload.map(|idle| {
            let embedder = Arc::clone(&embedder);
            IdleTimer::new(
                idle,
                Arc::new(move || {
                    let embedder = Arc::clone(&embedder);
                    tokio::spawn(async move {
                        embedder.unload().await;
                    });
                }),
            )
        });
        Ok(Self {
            embedder,
            idle,
            disposed: AtomicBool::new(false),
        })
    }

    fn activity(&self) -> Result<Option<Activity>> {
        if self.disposed.load(Ordering::Acquire) {
            return Err(MinSyncError::Other(
                "native embedder has been disposed".to_string(),
            ));
        }
        Ok(self.idle.as_ref().map(IdleTimer::begin))
    }

    pub async fn identity(&self) -> Result<NativeIdentity> {
        let _activity = self.activity()?;
        self.embedder.identity().await
    }

    /// Embed passages (the passage prefix is applied).
    pub async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let _activity = self.activity()?;
        self.embedder.embed(texts).await
    }

    /// Embed a search query (the query prefix is applied).
    pub async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let _activity = self.activity()?;
        self.embedder.embed_query(text).await
    }

    /// Release the model now; the next embed reloads it.
    pub async fn unload(&self) -> bool {
        self.embedder.unload().await
    }

    pub fn is_loaded(&self) -> bool {
        self.embedder.is_loaded()
    }

    /// Release the model and refuse every later call.
    pub async fn dispose(&self) {
        self.disposed.store(true, Ordering::Release);
        self.embedder.unload().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    const IDLE: Duration = Duration::from_secs(10);

    fn timer() -> (IdleTimer, Arc<AtomicUsize>) {
        let fired = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&fired);
        let timer = IdleTimer::new(
            IDLE,
            Arc::new(move || {
                counter.fetch_add(1, Ordering::SeqCst);
            }),
        );
        (timer, fired)
    }

    async fn settle() {
        for _ in 0..5 {
            tokio::task::yield_now().await;
        }
    }

    /// Advance the paused clock, letting timer tasks register their sleep
    /// before the jump and observe it after.
    async fn advance(by: Duration) {
        settle().await;
        tokio::time::advance(by).await;
        settle().await;
    }

    #[tokio::test(start_paused = true)]
    async fn fires_once_the_embedder_has_been_idle_long_enough() {
        let (timer, fired) = timer();
        drop(timer.begin());

        advance(IDLE - Duration::from_secs(1)).await;
        assert_eq!(fired.load(Ordering::SeqCst), 0);

        advance(Duration::from_secs(2)).await;
        assert_eq!(fired.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn new_activity_restarts_the_idle_period() {
        let (timer, fired) = timer();
        drop(timer.begin());
        advance(IDLE - Duration::from_secs(1)).await;

        drop(timer.begin());
        advance(Duration::from_secs(2)).await;
        assert_eq!(fired.load(Ordering::SeqCst), 0, "first timer is stale");

        advance(IDLE).await;
        assert_eq!(fired.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn never_fires_while_an_operation_is_in_flight() {
        let (timer, fired) = timer();
        let long_running = timer.begin();
        drop(timer.begin());

        advance(IDLE * 3).await;
        assert_eq!(fired.load(Ordering::SeqCst), 0);

        drop(long_running);
        advance(IDLE + Duration::from_secs(1)).await;
        assert_eq!(fired.load(Ordering::SeqCst), 1);
    }

    fn settings() -> EmbedderConfig {
        crate::config::Config::default_for("s").embedder
    }

    #[test]
    fn zero_idle_unload_is_rejected() {
        let result = NativeSession::new(&settings(), Some(Duration::ZERO));
        assert!(matches!(result, Err(MinSyncError::Config(_))));
    }

    #[tokio::test]
    async fn disposed_session_rejects_every_call_without_loading() {
        let session = NativeSession::new(&settings(), None).unwrap();
        assert!(!session.is_loaded());
        session.dispose().await;

        let texts = vec!["x".to_string()];
        for outcome in [
            session.embed(&texts).await.map(|_| ()),
            session.embed_query("x").await.map(|_| ()),
            session.identity().await.map(|_| ()),
        ] {
            let error = outcome.unwrap_err().to_string();
            assert!(error.contains("disposed"), "{error}");
        }
        assert!(!session.is_loaded());
    }
}
