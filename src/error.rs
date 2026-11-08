use thiserror::Error;

#[derive(Debug, Error)]
pub enum MinSyncError {
    #[error("not initialized — run `minsync init` first")]
    NotInitialized,

    #[error("already initialized — use --force to reinitialize")]
    AlreadyInitialized,

    #[error("another sync is in progress")]
    LockFailed,

    /// The configured dense embedder is not the one that built the stored
    /// vectors (or a rebuild toward another one was interrupted). Mixing the
    /// two would make every dense score meaningless, so the index is refused
    /// until `minsync sync --reembed` rebuilds it.
    #[error(
        "the embedder changed since this index was built ({why}) — run `minsync sync --reembed` \
         to rebuild every vector with the configured embedder (stored chunk text and sparse \
         vectors are reused, no file is re-read), or restore the previous [embedder] settings"
    )]
    EmbedderChanged { why: String },

    #[error("never synced — run `minsync sync --full` first")]
    NeverSynced,

    /// A sparse or hybrid query against an index whose rows were not all
    /// encoded by the configured sparse model.
    #[error("sparse index is not ready: {0} — run `minsync sync`")]
    SparseNotReady(String),

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

/// An im-not-llm engine failure, classified into MinSync's error vocabulary so
/// exit codes and messages stay stable across the crate boundary: a timeout
/// keeps [`MinSyncError::exit_code`] 7, a bad setting stays a config error, and
/// a failed load or forward pass is an embedding error.
impl From<inl_core::InlError> for MinSyncError {
    fn from(error: inl_core::InlError) -> Self {
        match error {
            inl_core::InlError::Config(message) => Self::Config(message),
            inl_core::InlError::Inference(message) => Self::Embedding(message),
            inl_core::InlError::Timeout(message) => Self::Timeout(message),
            inl_core::InlError::InvalidInput(message) => Self::Config(message),
            inl_core::InlError::Io(error) => Self::Io(error),
            inl_core::InlError::Json(error) => Self::Json(error),
            inl_core::InlError::Other(message) => Self::Other(message),
        }
    }
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
            Self::EmbedderChanged { .. } => 8,
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

    #[test]
    fn engine_errors_map_to_their_min_sync_class_and_exit_code() {
        let cases = [
            (inl_core::InlError::Timeout("stuck".into()), 7, "stuck"),
            (inl_core::InlError::Config("bad".into()), 1, "bad"),
            (inl_core::InlError::Inference("failed".into()), 5, "failed"),
            (inl_core::InlError::InvalidInput("bad".into()), 1, "bad"),
            (inl_core::InlError::Other("other".into()), 1, "other"),
        ];
        for (engine, code, payload) in cases {
            let mapped: MinSyncError = engine.into();
            assert_eq!(mapped.exit_code(), code, "{mapped}");
            assert!(mapped.to_string().contains(payload), "{mapped}");
        }
        assert!(matches!(
            MinSyncError::from(inl_core::InlError::Timeout("stuck".into())),
            MinSyncError::Timeout(_)
        ));
        assert!(matches!(
            MinSyncError::from(inl_core::InlError::Inference("failed".into())),
            MinSyncError::Embedding(_)
        ));
        assert!(matches!(
            MinSyncError::from(inl_core::InlError::Config("bad".into())),
            MinSyncError::Config(_)
        ));
    }
}
