use clap::Parser;
use minsync::cli::{Cli, Commands, OutputFormat};
use minsync::sync::MinSync;
use std::process;

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    let level = if cli.verbose {
        "debug"
    } else if cli.quiet {
        "error"
    } else {
        "info"
    };
    tracing_subscriber::fmt()
        .with_env_filter(format!("minsync={level}"))
        .with_writer(std::io::stderr)
        .init();

    let root = std::env::current_dir().unwrap_or_else(|error| {
        eprintln!("Error: cannot determine current directory: {error}");
        process::exit(1);
    });
    let minsync_dir = root.join(".minsync");

    match run(cli, root, minsync_dir).await {
        Ok(()) => process::exit(0),
        Err(error) => {
            eprintln!("Error: {error}");
            process::exit(error.exit_code());
        }
    }
}

async fn run(
    cli: Cli,
    root: std::path::PathBuf,
    minsync_dir: std::path::PathBuf,
) -> minsync::error::Result<()> {
    match cli.command {
        Commands::Init { force, embedder, chunker } => {
            let ms = MinSync::new(root);
            let config = ms.init(force, &embedder, &chunker)?;
            match cli.format {
                OutputFormat::Text => {
                    println!("Initialized MinSync in .minsync/");
                    println!("  source_id:   {}", config.source_id);
                    println!("  collection:  {}", config.collection.name);
                    println!("  chunker:     {}", config.chunker.id);
                    println!("  embedder:    {}", config.embedder.id);
                    println!("  vectorstore: {}", config.vectorstore.id);
                }
                OutputFormat::Json => {
                    println!("{}", serde_json::to_string_pretty(&config)?);
                }
            }
            Ok(())
        }
        Commands::Sync {
            full,
            dry_run,
            wait,
            batch_size,
        } => {
            let mut config = minsync::config::Config::load(&minsync_dir.join("config.toml"))?;
            if let Some(bs) = batch_size {
                config.embedder.batch_size = bs;
            }
            let chunker = minsync::chunker::create_chunker(&config)?;
            let embedder = minsync::embedder::create_embedder(&config)?;
            let store_path = minsync_dir.join(&config.collection.path);
            let mut store = minsync::vectorstore::create_vectorstore(&config, &store_path)?;

            let ms = MinSync::new(root);
            let result = ms
                .sync(
                    chunker.as_ref(),
                    embedder.as_ref(),
                    store.as_mut(),
                    full,
                    dry_run,
                    wait,
                )
                .await?;

            match cli.format {
                OutputFormat::Text => {
                    if result.already_up_to_date {
                        println!("Already up to date.");
                    } else if result.dry_run {
                        println!(
                            "Dry run: {} files would be processed",
                            result.files_processed_paths.len()
                        );
                    } else {
                        println!("Sync complete:");
                        println!("  files processed: {}", result.files_processed);
                        println!("  chunks added:    {}", result.chunks_added);
                        println!("  chunks updated:  {}", result.chunks_updated);
                        println!("  chunks deleted:  {}", result.chunks_deleted);
                        println!("  elapsed:         {:.2}s", result.elapsed_seconds);
                    }
                }
                OutputFormat::Json => {
                    println!("{}", serde_json::to_string_pretty(&result)?);
                }
            }
            Ok(())
        }
        Commands::Query { text, k } => {
            let config = minsync::config::Config::load(&minsync_dir.join("config.toml"))?;
            let embedder = minsync::embedder::create_embedder(&config)?;
            let store_path = minsync_dir.join(&config.collection.path);
            let store = minsync::vectorstore::create_vectorstore(&config, &store_path)?;

            let results = minsync::query::query(
                &minsync_dir,
                &text,
                k,
                embedder.as_ref(),
                store.as_ref(),
                None,
            )
            .await?;

            match cli.format {
                OutputFormat::Text => {
                    if results.is_empty() {
                        println!("No results found.");
                    } else {
                        println!("Found {} results:\n", results.len());
                        for result in &results {
                            println!(
                                "[{}] {} (score: {:.4})",
                                result.rank, result.path, result.score
                            );
                            if !result.heading_path.is_empty() {
                                println!("    heading: {}", result.heading_path);
                            }
                            println!("    ---");
                            let preview: String = result.text.chars().take(200).collect();
                            println!("    {preview}");
                            println!("    ---\n");
                        }
                    }
                }
                OutputFormat::Json => {
                    println!("{}", serde_json::to_string_pretty(&results)?);
                }
            }
            Ok(())
        }
        Commands::Status => {
            let result = minsync::verify::status(&minsync_dir).await?;
            match cli.format {
                OutputFormat::Text => {
                    println!("MinSync Status");
                    println!("  source_id:   {}", result.source_id);
                    println!("  collection:  {}", result.collection);
                    println!("  chunker:     {}", result.chunker);
                    println!("  embedder:    {}", result.embedder);
                    println!("  vectorstore: {}", result.vectorstore);
                    println!(
                        "  last synced: {}",
                        result.last_synced_at.as_deref().unwrap_or("(never)")
                    );
                    println!("  state:       {:?}", result.state);
                }
                OutputFormat::Json => {
                    println!("{}", serde_json::to_string_pretty(&result)?);
                }
            }
            Ok(())
        }
        Commands::Check => {
            let config = minsync::config::Config::load(&minsync_dir.join("config.toml"))?;
            let embedder = minsync::embedder::create_embedder(&config)?;
            let store_path = minsync_dir.join(&config.collection.path);
            let store = minsync::vectorstore::create_vectorstore(&config, &store_path)?;

            let result =
                minsync::verify::check(&minsync_dir, embedder.as_ref(), store.as_ref()).await?;
            match cli.format {
                OutputFormat::Text => {
                    println!("MinSync Health Check");
                    println!(
                        "  Embedder:    {}",
                        if result.embedder_ok { "OK" } else { "FAIL" }
                    );
                    println!(
                        "  VectorStore: {}",
                        if result.vectorstore_ok { "OK" } else { "FAIL" }
                    );
                    if result.all_passed {
                        println!("\nAll checks passed.");
                    } else {
                        for error in &result.errors {
                            println!("  Error: {error}");
                        }
                    }
                }
                OutputFormat::Json => {
                    println!("{}", serde_json::to_string_pretty(&result)?);
                }
            }
            Ok(())
        }
        Commands::Verify { fix, all, sample } => {
            let config = minsync::config::Config::load(&minsync_dir.join("config.toml"))?;
            let chunker = minsync::chunker::create_chunker(&config)?;
            let store_path = minsync_dir.join(&config.collection.path);
            let mut store = minsync::vectorstore::create_vectorstore(&config, &store_path)?;
            let sample_count = if all { None } else { Some(sample) };

            let result = minsync::verify::verify(
                &minsync_dir,
                &root,
                chunker.as_ref(),
                store.as_mut(),
                fix,
                sample_count,
            )
            .await?;
            match cli.format {
                OutputFormat::Text => {
                    if result.all_passed {
                        println!(
                            "ALL CHECKS PASSED{}",
                            if result.fixed { " (after fix)" } else { "" }
                        );
                    } else {
                        println!("VERIFICATION FAILED");
                    }
                }
                OutputFormat::Json => {
                    println!("{}", serde_json::to_string_pretty(&result)?);
                }
            }
            Ok(())
        }
        Commands::Watch { debounce_ms } => {
            let config = minsync::config::Config::load(&minsync_dir.join("config.toml"))?;
            let chunker = minsync::chunker::create_chunker(&config)?;
            let embedder = minsync::embedder::create_embedder(&config)?;
            let store_path = minsync_dir.join(&config.collection.path);
            let mut store = minsync::vectorstore::create_vectorstore(&config, &store_path)?;

            if let OutputFormat::Text = cli.format {
                println!("Watching {} for changes...", root.display());
            }

            minsync::watch::run(
                root,
                chunker.as_ref(),
                embedder.as_ref(),
                store.as_mut(),
                debounce_ms,
            )
            .await?;
            Ok(())
        }
    }
}
