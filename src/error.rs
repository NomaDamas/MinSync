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

    #[error("never synced — run `minsync sync` first")]
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
}

impl MinSyncError {
    pub fn exit_code(&self) -> i32 {
        match self {
            Self::LockFailed => 3,
            Self::VectorStore(_) => 4,
            Self::Embedding(_) => 5,
            _ => 1,
        }
    }
}

pub type Result<T> = std::result::Result<T, MinSyncError>;
