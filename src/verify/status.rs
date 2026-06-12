//! Sync state reporting for `minsync status`.

use crate::config::Config;
use crate::error::{MinSyncError, Result};
use crate::manifest::Manifest;
use crate::state::Cursor;
use crate::types::{StatusResult, SyncState};
use std::path::Path;

pub async fn status(minsync_dir: &Path) -> Result<StatusResult> {
    let config_path = minsync_dir.join("config.toml");
    if !config_path.exists() {
        return Err(MinSyncError::NotInitialized);
    }

    let config = Config::load(&config_path)?;
    let cursor_path = minsync_dir.join("cursor.json");
    let txn_path = minsync_dir.join("txn.json");

    if !cursor_path.exists() {
        return Ok(status_result(&config, None, SyncState::NotSynced, None));
    }

    let cursor = Cursor::load(&cursor_path)?;
    if txn_path.exists() {
        return Ok(status_result(
            &config,
            Some(&cursor),
            SyncState::Interrupted,
            Some(cursor.manifest_hash.clone()),
        ));
    }

    let manifest_hash = load_or_scan_manifest(minsync_dir, &config)?.manifest_hash();
    let state = if manifest_hash == cursor.manifest_hash {
        SyncState::UpToDate
    } else {
        SyncState::OutOfDate
    };

    Ok(status_result(
        &config,
        Some(&cursor),
        state,
        Some(manifest_hash),
    ))
}

fn status_result(
    config: &Config,
    cursor: Option<&Cursor>,
    state: SyncState,
    manifest_hash: Option<String>,
) -> StatusResult {
    StatusResult {
        source_id: config.source_id.clone(),
        collection: config.collection.name.clone(),
        chunker: config.chunker.id.clone(),
        embedder: config.embedder.id.clone(),
        vectorstore: config.vectorstore.id.clone(),
        last_synced_at: cursor.map(|active_cursor| active_cursor.last_synced_at.clone()),
        state,
        manifest_hash,
    }
}

fn load_or_scan_manifest(minsync_dir: &Path, config: &Config) -> Result<Manifest> {
    if let Some(root) = minsync_dir.parent() {
        return Manifest::scan(root, &config.source_id);
    }

    Manifest::load(&minsync_dir.join("manifest.json"))
}
