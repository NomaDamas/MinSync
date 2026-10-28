use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Who holds the workspace lock, as recorded in `.minsync/lock`.
///
/// Every field is optional: a lock file written by an older MinSync holds
/// only a bare pid, and a file that is empty or unreadable yields nothing.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LockHolder {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pid: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
}

impl LockHolder {
    fn is_empty(&self) -> bool {
        self.pid.is_none() && self.started_at.is_none() && self.command.is_none()
    }
}

fn lock_holder_suffix(holder: &Option<LockHolder>) -> String {
    let Some(holder) = holder.as_ref().filter(|holder| !holder.is_empty()) else {
        return String::new();
    };
    let mut parts = Vec::new();
    if let Some(pid) = holder.pid {
        parts.push(format!("pid {pid}"));
    }
    if let Some(started_at) = &holder.started_at {
        parts.push(format!("started {started_at}"));
    }
    if let Some(command) = &holder.command {
        parts.push(format!("`{command}`"));
    }
    format!(" ({})", parts.join(", "))
}

#[derive(Debug, Error)]
pub enum MinSyncError {
    #[error("not initialized — run `minsync init` first")]
    NotInitialized,

    #[error("already initialized — use --force to reinitialize")]
    AlreadyInitialized,

    /// The workspace lock is held by another process. `holder` is what that
    /// process recorded in the lock file, when it could be read.
    #[error("another sync is in progress{}", lock_holder_suffix(.holder))]
    LockFailed { holder: Option<LockHolder> },

    #[error("schema/embedder mismatch — run sync with --full")]
    SchemaMismatch,

    #[error("never synced — run `minsync sync --full` first")]
    NeverSynced,

    #[error("query text is required")]
    EmptyQuery,

    #[error("vector store error: {0}")]
    VectorStore(String),

    #[error("embedding failed: {0}")]
    Embedding(String),

    #[error("config error: {0}")]
    Config(String),

    #[error("manifest error: {0}")]
    Manifest(String),

    #[error("embedder timeout: {0}")]
    Timeout(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("toml parse error: {0}")]
    TomlParse(#[from] toml::de::Error),

    #[error("toml serialize error: {0}")]
    TomlSerialize(#[from] toml::ser::Error),

    #[error("{0}")]
    Other(String),

    /// A refusal by the incremental core or the materialization protocol,
    /// with a stable machine-readable `code` (see `incremental::stage`).
    #[error("{message}")]
    Stage { code: &'static str, message: String },
}

impl MinSyncError {
    pub fn stage(code: &'static str, message: impl Into<String>) -> Self {
        Self::Stage {
            code,
            message: message.into(),
        }
    }

    pub fn exit_code(&self) -> i32 {
        match self {
            Self::LockFailed { .. } => 3,
            Self::VectorStore(_) => 4,
            Self::Embedding(_) => 5,
            Self::Timeout(_) => 7,
            _ => 1,
        }
    }
}

pub type Result<T> = std::result::Result<T, MinSyncError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timeout_has_distinct_exit_code() {
        let error = MinSyncError::Timeout("native batch stuck".to_string());
        assert_eq!(error.exit_code(), 7);
        assert!(error.to_string().contains("native batch stuck"));
        for other in [
            MinSyncError::LockFailed { holder: None }.exit_code(),
            MinSyncError::VectorStore(String::new()).exit_code(),
            MinSyncError::Embedding(String::new()).exit_code(),
        ] {
            assert_ne!(other, 7);
        }
    }
}
