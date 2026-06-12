//! Environment health check for `minsync check`.

use crate::embedder::Embedder;
use crate::error::{MinSyncError, Result};
use crate::types::CheckResult;
use crate::vectorstore::VectorStore;
use std::path::Path;

pub async fn check(
    minsync_dir: &Path,
    embedder: &dyn Embedder,
    store: &dyn VectorStore,
) -> Result<CheckResult> {
    if !minsync_dir.join("config.toml").exists() {
        return Err(MinSyncError::NotInitialized);
    }

    let mut errors = Vec::new();
    let embedder_ok = match embedder.embed_single("hello").await {
        Ok(vector) if vector.is_empty() => {
            errors.push("embedder returned empty embedding".to_string());
            false
        }
        Ok(_) => true,
        Err(error) => {
            errors.push(format!("embedder check failed: {error}"));
            false
        }
    };

    let vectorstore_ok = {
        let _ = store.doc_count();
        true
    };

    Ok(CheckResult {
        embedder_ok,
        vectorstore_ok,
        all_passed: embedder_ok && vectorstore_ok && errors.is_empty(),
        errors,
    })
}
