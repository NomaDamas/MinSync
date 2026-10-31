use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(name = "minsync", version, about = "Git-free incremental vector index")]
pub struct Cli {
    #[arg(long, short, global = true)]
    pub verbose: bool,

    #[arg(long, short, global = true)]
    pub quiet: bool,

    #[arg(long, global = true, default_value = "text")]
    pub format: OutputFormat,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Clone, ValueEnum)]
pub enum OutputFormat {
    Text,
    Json,
}

/// Machine-readable progress stream format for `sync --progress`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum ProgressFormat {
    /// One JSON object per line.
    Jsonl,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum QueryMode {
    Vector,
    Sparse,
    Hybrid,
}

impl QueryMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Vector => "vector",
            Self::Sparse => "sparse",
            Self::Hybrid => "hybrid",
        }
    }
}

#[derive(Subcommand)]
pub enum Commands {
    /// Initialize .minsync/ in current directory
    Init {
        #[arg(long)]
        force: bool,
        /// Dense embedder id (`native:`, `tei:`, `openai:`), or `none` for a
        /// sparse-only index that needs no dense model or API key.
        #[arg(long, default_value = crate::config::DEFAULT_EMBEDDER_ID)]
        embedder: String,
        #[arg(long, default_value = "recursive")]
        chunker: String,
    },
    /// Sync files to vector index
    Sync {
        #[arg(long)]
        full: bool,
        #[arg(long)]
        dry_run: bool,
        #[arg(long)]
        wait: bool,
        #[arg(long)]
        batch_size: Option<usize>,
        /// Stream progress events to stdout while syncing (human logs stay
        /// on stderr). With `--format json` the final result becomes the
        /// last stdout line, compact.
        #[arg(long, value_enum)]
        progress: Option<ProgressFormat>,
        /// Also write the progress stream (JSONL) to this file, leaving
        /// stdout for the final result.
        #[arg(long, value_name = "PATH")]
        progress_file: Option<std::path::PathBuf>,
        /// Longest gap between progress events before a heartbeat is sent.
        #[arg(
            long,
            default_value_t = crate::sync::DEFAULT_HEARTBEAT_SECONDS,
            value_parser = clap::value_parser!(u64).range(1..)
        )]
        progress_heartbeat_seconds: u64,
    },
    /// Search indexed content
    Query {
        text: String,
        #[arg(long, short, default_value = "10")]
        k: usize,
        /// Retrieval mode. Defaults to `vector`, or to `sparse` in a workspace
        /// initialised with `--embedder none`.
        #[arg(long, value_enum)]
        mode: Option<QueryMode>,
    },
    /// Show sync status
    Status,
    /// Compact, re-index and prune the vector store without syncing.
    /// Recovers stores left fragmented by interrupted syncs.
    Optimize,
    /// Check environment health
    Check,
    /// Verify index consistency
    Verify {
        #[arg(long)]
        fix: bool,
        #[arg(long)]
        all: bool,
        #[arg(long, default_value = "10")]
        sample: usize,
    },
    /// Watch the directory and incrementally re-index changed UTF-8 text files
    Watch {
        #[arg(long)]
        debounce_ms: Option<u64>,
        /// Keep watching when the initial sync fails; later events retry sync.
        #[arg(long)]
        watch_on_sync_error: bool,
    },
    /// Serve the configured native embedder over loopback HTTP (TEI
    /// `POST /embed`, OpenAI `POST /v1/embeddings`, `GET /health`) so other
    /// processes share one loaded model. Adopts a server already running
    /// for the same model instead of loading a second copy.
    Serve {
        /// Loopback address to bind (127.0.0.1, ::1 or localhost); other
        /// addresses are rejected because the server has no authentication.
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// TCP port; 0 picks a free one (written to the `port` file).
        #[arg(long, default_value_t = 0)]
        port: u16,
        /// Seconds without an embedding request before the server exits and
        /// frees the model; 0 serves until stopped.
        #[arg(long, default_value_t = crate::serve::DEFAULT_IDLE_TIMEOUT_SECONDS)]
        idle_timeout: u64,
        /// Most inputs accepted in one request (larger requests get 413).
        #[arg(long, default_value_t = crate::serve::DEFAULT_MAX_INPUTS)]
        max_inputs: usize,
        /// Largest request body in bytes (larger requests get 413).
        #[arg(long, default_value_t = crate::serve::DEFAULT_MAX_REQUEST_BYTES)]
        max_request_bytes: usize,
        /// Directory for the `pid`, `host`, `port` and `lock` files.
        /// Defaults to ~/.cache/minsync/serve/<model instance>.
        #[arg(long, value_name = "DIR")]
        state_dir: Option<std::path::PathBuf>,
        /// Native embedder id to serve. Defaults to the workspace's
        /// `.minsync/config.toml` embedder (`[sparse]` with `--sparse`), else
        /// the default model.
        #[arg(long)]
        embedder: Option<String>,
        /// Serve the learned-sparse (SPLADE) encoder instead of the dense
        /// embedder: TEI's `POST /embed_sparse` and `GET /health`. A separate
        /// server (and state directory) from the dense one.
        #[arg(long)]
        sparse: bool,
    },
    /// Run one materialization-stage session: JSONL requests on stdin,
    /// responses on stdout. Tracks which source items an external host
    /// (a parser) has already turned into output files, so a killed or
    /// failed run resumes with only the pending items.
    ///
    /// Ops: open, commit, fail (defer / discard_output), skip, delete,
    /// finish; read-only and lock-free: capabilities, plan, inspect.
    /// Send `capabilities` before relying on any newer op or field.
    /// Protocol: docs/INCREMENTAL_CORE.md.
    Stage {
        /// Directory for the stage's lock, run marker, journal and snapshot.
        /// One stage and one corpus per directory; keep it outside the
        /// output root.
        #[arg(long, value_name = "DIR")]
        state_dir: std::path::PathBuf,
    },
}

#[cfg(test)]
mod tests {
    use super::{Cli, Commands};
    use clap::Parser;

    #[test]
    fn watch_accepts_sync_error_recovery_flag() {
        let cli = Cli::try_parse_from(["minsync", "watch", "--watch-on-sync-error"])
            .expect("watch recovery flag is accepted");

        assert!(matches!(
            cli.command,
            Commands::Watch {
                watch_on_sync_error: true,
                ..
            }
        ));
    }

    #[test]
    fn serve_sparse_is_a_flag_and_defaults_off() {
        let dense = Cli::try_parse_from(["minsync", "serve"]).expect("serve parses");
        assert!(matches!(
            dense.command,
            Commands::Serve { sparse: false, .. }
        ));
        let sparse =
            Cli::try_parse_from(["minsync", "serve", "--sparse"]).expect("serve --sparse parses");
        assert!(matches!(
            sparse.command,
            Commands::Serve { sparse: true, .. }
        ));
    }

    #[test]
    fn query_modes_are_vector_sparse_and_hybrid_only() {
        for mode in ["vector", "sparse", "hybrid"] {
            assert!(
                Cli::try_parse_from(["minsync", "query", "q", "--mode", mode]).is_ok(),
                "{mode} must parse"
            );
        }
        let bm25 = Cli::try_parse_from(["minsync", "query", "q", "--mode", "bm25"]);
        assert!(bm25.is_err(), "bm25 search no longer exists");
    }

    #[test]
    fn init_has_no_language_flag() {
        assert!(Cli::try_parse_from(["minsync", "init", "--language", "ko"]).is_err());
        assert!(Cli::try_parse_from(["minsync", "init"]).is_ok());
    }
}
