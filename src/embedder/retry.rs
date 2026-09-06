use crate::error::{MinSyncError, Result};
use std::future::Future;
use std::time::Duration;

/// Error from a single embedding HTTP request, classified for retry.
#[derive(Debug)]
pub enum RequestError {
    /// Transient: network failure, timeout, HTTP 429, HTTP 5xx.
    Retryable(String),
    /// Permanent: auth/validation 4xx, malformed response — fail fast.
    Fatal(String),
}

/// Shared retry policy: exponential backoff with equal jitter, bounded by
/// `max_delay`. `max_retries` is the number of retries after the first
/// attempt (total attempts = max_retries + 1).
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub max_retries: usize,
    pub base_delay: Duration,
    pub max_delay: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(10),
        }
    }
}

impl RetryPolicy {
    pub async fn run<T, F, Fut>(&self, mut op: F) -> Result<T>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = std::result::Result<T, RequestError>>,
    {
        let attempts = self.max_retries + 1;
        let mut last_error = String::new();
        for attempt in 0..attempts {
            match op().await {
                Ok(value) => return Ok(value),
                Err(RequestError::Fatal(message)) => {
                    return Err(MinSyncError::Embedding(message));
                }
                Err(RequestError::Retryable(message)) => {
                    tracing::warn!(
                        attempt = attempt + 1,
                        max_attempts = attempts,
                        error = %message,
                        "retryable embedding request failure"
                    );
                    last_error = message;
                    if attempt + 1 < attempts {
                        tokio::time::sleep(self.delay_for(attempt)).await;
                    }
                }
            }
        }
        Err(MinSyncError::Embedding(format!(
            "retries exhausted after {attempts} attempts: {last_error}"
        )))
    }

    /// Backoff before retry number `attempt + 1`: exponential growth from
    /// `base_delay`, capped at `max_delay`, with equal jitter (between half
    /// and the full exponential delay).
    pub fn delay_for(&self, attempt: usize) -> Duration {
        let exponent = u32::try_from(attempt.min(16)).unwrap_or(16);
        let exp_delay = self
            .base_delay
            .saturating_mul(1u32 << exponent)
            .min(self.max_delay);
        exp_delay / 2 + exp_delay.mul_f64(jitter_fraction() / 2.0)
    }
}

/// Classify an HTTP error status: 429 and 5xx are retryable, other
/// non-success statuses (auth, validation) are fatal.
pub fn classify_status(status: reqwest::StatusCode, provider: &str, body: &str) -> RequestError {
    let message = format!("{provider} API error {status}: {body}");
    if is_context_length_error(body) {
        return RequestError::Fatal(format!(
            "chunk exceeds the embedding model's context; \
             lower `chunker.options.max_chunk_size` or set `embedder.truncate = true` \
             ({message})"
        ));
    }
    if status == reqwest::StatusCode::TOO_MANY_REQUESTS || status.is_server_error() {
        RequestError::Retryable(message)
    } else {
        RequestError::Fatal(message)
    }
}

fn is_context_length_error(body: &str) -> bool {
    let body = body.to_ascii_lowercase();
    body.contains("context length")
        || body.contains("input length exceeds")
        || body.contains("must have less than")
}

/// Classify a request transport error: connection failures and timeouts are
/// retryable.
pub fn classify_send_error(error: &reqwest::Error, provider: &str) -> RequestError {
    RequestError::Retryable(format!("{provider} request failed: {error}"))
}

/// Pseudo-random fraction in [0, 1) derived from the wall clock via
/// splitmix64 — enough spread for backoff jitter without a rand dependency.
fn jitter_fraction() -> f64 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| u64::from(d.subsec_nanos()) ^ (d.as_secs() << 32))
        .unwrap_or(0x1234_5678);
    let mut z = nanos.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    fn fast_policy(max_retries: usize) -> RetryPolicy {
        RetryPolicy {
            max_retries,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(4),
        }
    }

    #[tokio::test]
    async fn test_success_first_try() {
        let policy = fast_policy(3);
        let attempts = Arc::new(AtomicUsize::new(0));

        let counter = attempts.clone();
        let result: Result<u32> = policy
            .run(|| {
                let counter = counter.clone();
                async move {
                    counter.fetch_add(1, Ordering::SeqCst);
                    Ok(7)
                }
            })
            .await;

        assert_eq!(result.unwrap(), 7);
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_retryable_then_success() {
        let policy = fast_policy(3);
        let attempts = Arc::new(AtomicUsize::new(0));

        let counter = attempts.clone();
        let result: Result<u32> = policy
            .run(|| {
                let counter = counter.clone();
                async move {
                    let attempt = counter.fetch_add(1, Ordering::SeqCst);
                    if attempt < 2 {
                        Err(RequestError::Retryable("503".to_string()))
                    } else {
                        Ok(42)
                    }
                }
            })
            .await;

        assert_eq!(result.unwrap(), 42);
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_fatal_fails_fast() {
        let policy = fast_policy(3);
        let attempts = Arc::new(AtomicUsize::new(0));

        let counter = attempts.clone();
        let result: Result<u32> = policy
            .run(|| {
                let counter = counter.clone();
                async move {
                    counter.fetch_add(1, Ordering::SeqCst);
                    Err(RequestError::Fatal("401 unauthorized".to_string()))
                }
            })
            .await;

        let error = result.unwrap_err();
        assert!(error.to_string().contains("401 unauthorized"));
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_context_length_server_error_is_fatal() {
        let error = classify_status(
            reqwest::StatusCode::INTERNAL_SERVER_ERROR,
            "TEI",
            "the input length exceeds the context length",
        );

        assert!(matches!(error, RequestError::Fatal(message) if
            message.contains("chunk exceeds the embedding model's context") &&
            message.contains("chunker.options.max_chunk_size") &&
            message.contains("embedder.truncate")));
    }

    #[tokio::test]
    async fn test_retry_exhaustion() {
        let policy = fast_policy(2);
        let attempts = Arc::new(AtomicUsize::new(0));

        let counter = attempts.clone();
        let result: Result<u32> = policy
            .run(|| {
                let counter = counter.clone();
                async move {
                    counter.fetch_add(1, Ordering::SeqCst);
                    Err(RequestError::Retryable("always 503".to_string()))
                }
            })
            .await;

        let error = result.unwrap_err();
        assert!(error.to_string().contains("retries exhausted"));
        assert!(error.to_string().contains("always 503"));
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_zero_retries_single_attempt() {
        let policy = fast_policy(0);
        let attempts = Arc::new(AtomicUsize::new(0));

        let counter = attempts.clone();
        let result: Result<u32> = policy
            .run(|| {
                let counter = counter.clone();
                async move {
                    counter.fetch_add(1, Ordering::SeqCst);
                    Err(RequestError::Retryable("503".to_string()))
                }
            })
            .await;

        assert!(result.is_err());
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_delay_grows_and_is_capped() {
        let policy = RetryPolicy {
            max_retries: 10,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(2),
        };

        let first = policy.delay_for(0);
        assert!(first >= Duration::from_millis(50));
        assert!(first <= Duration::from_millis(100));

        for attempt in 0..20 {
            assert!(policy.delay_for(attempt) <= Duration::from_secs(2));
        }
    }

    #[test]
    fn test_classify_status() {
        let retryable_429 = classify_status(
            reqwest::StatusCode::TOO_MANY_REQUESTS,
            "OpenAI",
            "slow down",
        );
        assert!(matches!(retryable_429, RequestError::Retryable(_)));

        let retryable_500 =
            classify_status(reqwest::StatusCode::INTERNAL_SERVER_ERROR, "TEI", "boom");
        assert!(matches!(retryable_500, RequestError::Retryable(_)));

        let fatal_401 = classify_status(reqwest::StatusCode::UNAUTHORIZED, "OpenAI", "no");
        assert!(matches!(fatal_401, RequestError::Fatal(_)));

        let fatal_422 = classify_status(reqwest::StatusCode::UNPROCESSABLE_ENTITY, "TEI", "bad");
        assert!(matches!(fatal_422, RequestError::Fatal(_)));
    }

    #[test]
    fn test_jitter_fraction_in_range() {
        for _ in 0..100 {
            let fraction = jitter_fraction();
            assert!((0.0..1.0).contains(&fraction));
        }
    }
}
