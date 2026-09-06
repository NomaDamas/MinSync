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

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum QueryMode {
    Vector,
    Bm25,
    Hybrid,
}

impl QueryMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Vector => "vector",
            Self::Bm25 => "bm25",
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
        #[arg(long, default_value = "openai:text-embedding-3-small")]
        embedder: String,
        #[arg(long, default_value = "recursive")]
        chunker: String,
        #[arg(long, default_value = "simple")]
        language: String,
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
    },
    /// Search indexed content
    Query {
        text: String,
        #[arg(long, short, default_value = "10")]
        k: usize,
        #[arg(long, value_enum, default_value = "vector")]
        mode: QueryMode,
    },
    /// Show sync status
    Status,
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
}
