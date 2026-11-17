//! Learned sparse (SPLADE) retrieval: the encoder contract MinSync's sync and
//! query paths drive. The vector type and its scoring primitives, and the
//! in-process model, belong to `im-not-llm` (`inl-sparse`); providers here are
//! [`native`] (an adapter over that engine), [`tei`] (remote `/embed_sparse`)
//! and [`served`] (a running `minsync serve --sparse`).

pub mod inverted;
pub mod native;
pub mod served;
pub mod tei;

use crate::error::Result;
use async_trait::async_trait;

pub use inl_sparse::{dot_terms, SparseVector, TopK};

/// Produces [`SparseVector`]s. Mirrors [`crate::embedder::Embedder`]: passages
/// at index time, queries at search time (the model marks the two
/// differently), no query-time shortcut.
#[async_trait]
pub trait SparseEncoder: Send + Sync {
    /// Model identity recorded in the cursor (`naver/...`, without provider).
    fn id(&self) -> &str;

    /// Encode passages, one vector per input, in order.
    async fn encode(&self, texts: &[String]) -> Result<Vec<SparseVector>>;

    /// Encode one search query.
    async fn encode_query(&self, text: &str) -> Result<SparseVector>;

    /// Passages the model had to cut to fit its context.
    async fn count_truncated(&self, _texts: &[String]) -> Result<usize> {
        Ok(0)
    }

    fn max_length(&self) -> Option<usize> {
        None
    }

    /// True when re-running identical input is guaranteed to give identical
    /// output (in-process inference on a fixed device).
    fn is_deterministic(&self) -> bool {
        false
    }
}
