use thiserror::Error;

#[derive(Debug, Error)]
pub enum MinSyncError {
    #[error("not initialized — run `minsync init` first")]
    NotInitialized,

    #[error("already initialized — use --force to reinitialize")]
    AlreadyInitialized,

    #[error("another sync is in progress")]
    LockFailed,

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
            Self::LockFailed => 3,
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
            MinSyncError::LockFailed.exit_code(),
            MinSyncError::VectorStore(String::new()).exit_code(),
            MinSyncError::Embedding(String::new()).exit_code(),
        ] {
            assert_ne!(other, 7);
        }
    }
}
