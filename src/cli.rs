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

/// Legacy shorthand for `--mode`. Prefer the independent `--vector` (alias
/// `--dense`), `--sparse`, and `--bm25` on/off flags to select any combination
/// of retrievers without a combinatorial mode list.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum QueryMode {
    Vector,
    Sparse,
    Bm25,
    Hybrid,
}

impl QueryMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Vector => "vector",
            Self::Sparse => "sparse",
            Self::Bm25 => "bm25",
            Self::Hybrid => "hybrid",
        }
    }

    pub fn to_retrievers(self) -> Retrievers {
        match self {
            Self::Vector => Retrievers {
                vector: true,
                sparse: false,
                bm25: false,
            },
            Self::Sparse => Retrievers {
                vector: false,
                sparse: true,
                bm25: false,
            },
            Self::Bm25 => Retrievers {
                vector: false,
                sparse: false,
                bm25: true,
            },
            Self::Hybrid => Retrievers {
                vector: true,
                sparse: true,
                bm25: false,
            },
        }
    }
}

/// Independent on/off switches for MinSync's three retrievers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Retrievers {
    pub vector: bool,
    pub sparse: bool,
    pub bm25: bool,
}

impl Retrievers {
    pub fn count(self) -> usize {
        usize::from(self.vector) + usize::from(self.sparse) + usize::from(self.bm25)
    }

    /// Stable label written to `QueryResult.mode`. Keeps `"vector"`, `"sparse"`,
    /// `"bm25"`, and `"hybrid"` (vector+sparse) backwards-compatible.
    pub fn label(self) -> &'static str {
        match (self.vector, self.sparse, self.bm25) {
            (true, false, false) => "vector",
            (false, true, false) => "sparse",
            (false, false, true) => "bm25",
            (true, true, false) => "hybrid",
            (true, false, true) => "vector+bm25",
            (false, true, true) => "sparse+bm25",
            (true, true, true) => "vector+sparse+bm25",
            (false, false, false) => "none",
        }
    }

    /// Resolve the active retriever set from CLI flags and workspace config:
    /// - Explicit `--mode` maps directly to its shorthand set.
    /// - If any of `--vector`, `--sparse`, `--bm25` is turned ON (`Some(true)`),
    ///   run ONLY the flags turned ON (omitted flags default to off).
    /// - If only negative toggles are passed (e.g. `--sparse=false`), start from
    ///   every retriever configured in `config` and turn off the requested ones.
    /// - If nothing was passed, keep the single-retriever workspace default
    ///   (`bm25` for BM25-only, `sparse` for sparse-only, `vector` otherwise).
    pub fn resolve(
        mode: Option<QueryMode>,
        vector: Option<bool>,
        sparse: Option<bool>,
        bm25: Option<bool>,
        config: &crate::config::Config,
    ) -> crate::error::Result<Self> {
        let selected = if let Some(mode) = mode {
            mode.to_retrievers()
        } else if vector == Some(true) || sparse == Some(true) || bm25 == Some(true) {
            Self {
                vector: vector.unwrap_or(false),
                sparse: sparse.unwrap_or(false),
                bm25: bm25.unwrap_or(false),
            }
        } else if vector.is_some() || sparse.is_some() || bm25.is_some() {
            Self {
                vector: vector.unwrap_or(!config.is_sparse_only()),
                sparse: sparse.unwrap_or(!config.is_sparse_disabled()),
                bm25: bm25.unwrap_or(config.bm25.enabled),
            }
        } else if config.is_bm25_only() {
            QueryMode::Bm25.to_retrievers()
        } else if config.is_sparse_only() {
            QueryMode::Sparse.to_retrievers()
        } else {
            QueryMode::Vector.to_retrievers()
        };

        if selected.count() == 0 {
            return Err(crate::error::MinSyncError::Config(
                "at least one retriever (--vector, --sparse, or --bm25) must be enabled".into(),
            ));
        }
        Ok(selected)
    }
}

#[derive(Subcommand)]
pub enum Commands {
    /// Initialize .minsync/ in current directory
    Init {
        #[arg(long)]
        force: bool,
        /// Dense embedder id (`native:`, `tei:`, `openai:`), `none` to disable,
        /// or `auto` (default) to decide from detected RAM & GPU headroom.
        #[arg(long, default_value = "auto")]
        embedder: String,
        /// Sparse (SPLADE) encoder id (`native:`, `tei:`), `none` to disable,
        /// or `auto` (default) to decide from detected RAM & CPU cores.
        #[arg(long, default_value = "auto")]
        sparse: String,
        /// Disable BM25 lexical indexing (enabled by default).
        #[arg(long)]
        no_bm25: bool,
        /// BM25 tokenizer language preset (`multilingual`, `simple`, `ko`,
        /// `ja`, `zh`, `ar`).
        #[arg(long, default_value = crate::bm25::tokenizer::DEFAULT_LANGUAGE)]
        language: String,
        #[arg(long, default_value = crate::config::DEFAULT_CHUNKER_ID)]
        chunker: String,
    },
    /// Sync files to vector index
    Sync {
        #[arg(long)]
        full: bool,
        /// Rebuild every dense vector with the configured embedder, from the
        /// chunk text already stored (no file is re-read, sparse vectors are
        /// kept), then sync as usual. Use it after changing `[embedder]` to
        /// another model, dimension or passage prefix. Safe to interrupt:
        /// the next run continues where it stopped. With `--dry-run` it only
        /// probes the embedder (one short text) and reports what it would
        /// rebuild, changing nothing.
        #[arg(long, conflicts_with = "full")]
        reembed: bool,
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
        /// Toggle dense vector search (`--vector` / `--dense`, or `--vector=false`).
        #[arg(
            long,
            visible_alias = "dense",
            num_args = 0..=1,
            require_equals = true,
            default_missing_value = "true",
            conflicts_with = "mode"
        )]
        vector: Option<bool>,
        /// Toggle SPLADE sparse search (`--sparse`, or `--sparse=false`).
        #[arg(
            long,
            num_args = 0..=1,
            require_equals = true,
            default_missing_value = "true",
            conflicts_with = "mode"
        )]
        sparse: Option<bool>,
        /// Toggle BM25 lexical search (`--bm25`, or `--bm25=false`).
        #[arg(
            long,
            num_args = 0..=1,
            require_equals = true,
            default_missing_value = "true",
            conflicts_with = "mode"
        )]
        bm25: Option<bool>,
        /// Shorthand retrieval mode (`vector`, `sparse`, `bm25`, `hybrid`).
        /// For arbitrary combinations (e.g. dense+bm25 or all three), use
        /// `--vector`, `--sparse`, `--bm25` on/off flags directly.
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
    /// processes share one loaded model. Adopts a server already running for
    /// the same model — another `minsync serve` or an `inl serve` from
    /// im-not-llm — instead of loading a second copy.
    Serve {
        /// Loopback address to bind (127.0.0.1, ::1 or localhost); other
        /// addresses are rejected because the server has no authentication.
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// TCP port. 0 (the default) reuses the port this instance served on last
        /// time and otherwise picks a free one (written to the `port` file).
        #[arg(long, default_value_t = 0)]
        port: u16,
        /// Seconds without an embedding request before the server exits and
        /// frees the model. The default, 0, serves until stopped.
        #[arg(long, default_value_t = crate::serve::DEFAULT_IDLE_TIMEOUT_SECONDS)]
        idle_timeout: u64,
        /// Most inputs accepted in one request (larger requests get 413).
        #[arg(long, default_value_t = crate::serve::DEFAULT_MAX_INPUTS)]
        max_inputs: usize,
        /// Largest request body in bytes (larger requests get 413).
        #[arg(long, default_value_t = crate::serve::DEFAULT_MAX_REQUEST_BYTES)]
        max_request_bytes: usize,
        /// Directory for the `pid`, `host`, `port` and `lock` files.
        /// Defaults to ~/.cache/im-not-llm/serve/<model instance>.
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
        #[arg(long, conflicts_with = "multimodal")]
        sparse: bool,
        /// Serve a multimodal (text + image) embedder such as
        /// `google/embeddinggemma-2`: `POST /embed_multimodal`, `POST /embed`,
        /// `POST /v1/embeddings`, and `GET /health`.
        #[arg(long, conflicts_with = "sparse")]
        multimodal: bool,
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

    /// `minsync serve` stays up until stopped unless `--idle-timeout` is given, like `inl serve`.
    #[test]
    fn serve_has_no_idle_timeout_by_default() {
        let cli = Cli::try_parse_from(["minsync", "serve"]).expect("serve parses");
        let Commands::Serve { idle_timeout, .. } = cli.command else {
            panic!("expected serve");
        };
        assert_eq!(idle_timeout, 0);
        assert_eq!(crate::serve::idle_timeout_from_seconds(idle_timeout), None);
        let cli = Cli::try_parse_from(["minsync", "serve", "--idle-timeout", "90"]).unwrap();
        let Commands::Serve { idle_timeout, .. } = cli.command else {
            panic!("expected serve");
        };
        assert_eq!(
            crate::serve::idle_timeout_from_seconds(idle_timeout),
            Some(std::time::Duration::from_secs(90))
        );
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
    fn query_accepts_independent_retriever_toggles_and_legacy_modes() {
        for mode in ["vector", "sparse", "bm25", "hybrid"] {
            assert!(
                Cli::try_parse_from(["minsync", "query", "q", "--mode", mode]).is_ok(),
                "{mode} must parse"
            );
        }
        assert!(Cli::try_parse_from(["minsync", "query", "q", "--mode", "nope"]).is_err());

        // Independent retriever toggles (both before and after the query text).
        let cli = Cli::try_parse_from(["minsync", "query", "--dense", "--bm25", "hello"])
            .expect("toggles before query parse cleanly");
        assert!(matches!(
            cli.command,
            Commands::Query {
                ref text,
                vector: Some(true),
                sparse: None,
                bm25: Some(true),
                mode: None,
                ..
            } if text == "hello"
        ));

        let off = Cli::try_parse_from(["minsync", "query", "hello", "--sparse=false"])
            .expect("explicit false toggle parses");
        assert!(matches!(
            off.command,
            Commands::Query {
                vector: None,
                sparse: Some(false),
                bm25: None,
                ..
            }
        ));

        // Mixing --mode with retriever toggles is rejected.
        assert!(
            Cli::try_parse_from(["minsync", "query", "q", "--mode", "bm25", "--vector"]).is_err()
        );
    }

    #[test]
    fn init_defaults_to_auto_and_accepts_retriever_flags() {
        let auto = Cli::try_parse_from(["minsync", "init"]).expect("init parses");
        assert!(matches!(
            auto.command,
            Commands::Init {
                ref embedder,
                ref sparse,
                no_bm25: false,
                ..
            } if embedder == "auto" && sparse == "auto"
        ));

        let cli = Cli::try_parse_from([
            "minsync",
            "init",
            "--embedder",
            "none",
            "--sparse",
            "none",
            "--no-bm25",
            "--language",
            "ko",
        ])
        .expect("init flags parse");
        assert!(matches!(
            cli.command,
            Commands::Init {
                ref embedder,
                ref sparse,
                no_bm25: true,
                ref language,
                ..
            } if embedder == "none" && sparse == "none" && language == "ko"
        ));
    }
}
