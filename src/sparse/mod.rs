//! Learned sparse (SPLADE) retrieval: the vector type, the scoring loops and
//! the encoder contract. Providers live in [`native`] (in-process, or a running
//! `minsync serve --sparse`), [`tei`] (remote `/embed_sparse`) and [`served`].
//!
//! The vector type and the scoring loops are the ones im-not-llm's sparse
//! engine produces and scores with, re-exported so a stored vector, an engine
//! output and a query all share one type with no conversion on the hot path.

pub mod native;
pub mod served;
pub mod tei;

pub use inl_sparse::{dot_terms, SparseVector, TopK};

use crate::error::Result;
use async_trait::async_trait;

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
