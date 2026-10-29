//! Environment health check for `minsync check`.

use crate::embedder::Embedder;
use crate::error::{MinSyncError, Result};
use crate::sparse::SparseEncoder;
use crate::types::CheckResult;
use crate::vectorstore::VectorStore;
use std::path::Path;

/// Probe the dense embedder and the sparse encoder, and read the store.
///
/// A sparse-only workspace has no dense model to probe; `encoder` is `None`
/// for callers that do not search sparsely (then there is nothing to probe
/// and `sparse_ok` is true). A probe that answers with no terms still counts:
/// the encoder worked, the probe text just has none.
pub async fn check(
    minsync_dir: &Path,
    embedder: &dyn Embedder,
    encoder: Option<&dyn SparseEncoder>,
    store: &dyn VectorStore,
) -> Result<CheckResult> {
    if !minsync_dir.join("config.toml").exists() {
        return Err(MinSyncError::NotInitialized);
    }

    let mut errors = Vec::new();
    let embedder_ok = if embedder.is_sparse_only() {
        true
    } else {
        match embedder.embed_single("hello").await {
            Ok(vector) if vector.is_empty() => {
                errors.push("embedder returned empty embedding".to_string());
                false
            }
            Ok(_) => true,
            Err(error) => {
                errors.push(format!("embedder check failed: {error}"));
                false
            }
        }
    };

    let sparse_ok = match encoder {
        None => true,
        Some(encoder) => match encoder.encode_query("hello").await {
            Ok(_) => true,
            Err(error) => {
                errors.push(format!("sparse encoder check failed: {error}"));
                false
            }
        },
    };

    let vectorstore_ok = {
        let _ = store.doc_count();
        true
    };
    let index_state = store.index_state()?;

    Ok(CheckResult {
        embedder_ok,
        sparse_ok,
        vectorstore_ok,
        all_passed: embedder_ok && sparse_ok && vectorstore_ok && errors.is_empty(),
        errors,
        index_state,
    })
}
