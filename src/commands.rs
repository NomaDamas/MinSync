//! CLI command handlers. `main.rs` stays a thin binary: parse args, set up
//! logging, dispatch here, and map errors to exit codes.

use crate::cli::{Cli, Commands, OutputFormat, QueryMode};
use crate::error::Result;
use crate::sync::MinSync;
use std::path::PathBuf;

pub async fn run(cli: Cli, root: PathBuf, minsync_dir: PathBuf) -> Result<()> {
    match cli.command {
        Commands::Init {
            force,
            embedder,
            chunker,
            language,
        } => init(&cli.format, root, force, &embedder, &chunker, &language),
        Commands::Sync {
            full,
            dry_run,
            wait,
            batch_size,
        } => {
            sync(
                &cli.format,
                root,
                &minsync_dir,
                full,
                dry_run,
                wait,
                batch_size,
            )
            .await
        }
        Commands::Query { text, k, mode } => query(&cli.format, &minsync_dir, &text, k, mode).await,
        Commands::Status => status(&cli.format, &minsync_dir).await,
        Commands::Check => check(&cli.format, &minsync_dir).await,
        Commands::Verify { fix, all, sample } => {
            verify(&cli.format, root, &minsync_dir, fix, all, sample).await
        }
        Commands::Watch {
            debounce_ms,
            watch_on_sync_error,
        } => {
            watch(
                &cli.format,
                root,
                &minsync_dir,
                debounce_ms,
                watch_on_sync_error,
            )
            .await
        }
    }
}

fn init(
    format: &OutputFormat,
    root: PathBuf,
    force: bool,
    embedder: &str,
    chunker: &str,
    language: &str,
) -> Result<()> {
    let ms = MinSync::new(root);
    crate::tokenizer::validate_language(language)?;
    let mut config = ms.init(force, embedder, chunker)?;
    config.lexical.language = language.to_string();
    config.save(&ms.minsync_dir().join("config.toml"))?;
    match format {
        OutputFormat::Text => {
            println!("Initialized MinSync in .minsync/");
            println!("  source_id:   {}", config.source_id);
            println!("  collection:  {}", config.collection.name);
            println!("  chunker:     {}", config.chunker.id);
            println!("  embedder:    {}", config.embedder.id);
            println!("  vectorstore: {}", config.vectorstore.id);
            println!("  language:    {}", config.lexical.language);
        }
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&config)?);
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn sync(
    format: &OutputFormat,
    root: PathBuf,
    minsync_dir: &std::path::Path,
    full: bool,
    dry_run: bool,
    wait: bool,
    batch_size: Option<usize>,
) -> Result<()> {
    let mut config = crate::config::Config::load(&minsync_dir.join("config.toml"))?;
    if let Some(bs) = batch_size {
        config.embedder.batch_size = bs;
    }
    let chunker = crate::chunker::create_chunker(&config)?;
    let embedder = crate::embedder::create_embedder(&config)?;
    let store_path = minsync_dir.join(&config.collection.path);
    let mut store = crate::vectorstore::create_vectorstore(&config, &store_path)?;

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

    match format {
        OutputFormat::Text => {
            if result.already_up_to_date {
                println!("Already up to date.");
            } else if result.dry_run {
                println!(
                    "Dry run: {} files would be processed",
                    result.files_processed_paths.len()
                );
            } else {
                if result.initial_sync {
                    println!("Initial sync: no cursor found — performed full sync.");
                }
                println!("Sync complete:");
                println!("  files processed: {}", result.files_processed);
                println!("  files added:     {}", result.files_added);
                println!("  files modified:  {}", result.files_modified);
                println!("  files deleted:   {}", result.files_deleted);
                println!("  chunks inserted: {}", result.chunks_added);
                println!("  chunks reused:   {}", result.chunks_updated);
                println!("  chunks removed:  {}", result.chunks_deleted);
                println!();
                println!("Sync Stats");
                println!("  elapsed time:        {:.2}s", result.elapsed_seconds);
                println!("  embedding API calls: {}", result.embedding_api_calls);
                println!("  embedded texts:      {}", result.embedded_texts);
                println!("  estimated tokens:    {}", result.estimated_tokens);
            }
        }
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
    }
    Ok(())
}

async fn query(
    format: &OutputFormat,
    minsync_dir: &std::path::Path,
    text: &str,
    k: usize,
    mode: QueryMode,
) -> Result<()> {
    let config = crate::config::Config::load(&minsync_dir.join("config.toml"))?;
    let store_path = minsync_dir.join(&config.collection.path);
    let store = crate::vectorstore::create_vectorstore(&config, &store_path)?;
    let results = match mode {
        QueryMode::Bm25 => crate::query::query_text(minsync_dir, text, k, store.as_ref(), None)?,
        QueryMode::Vector | QueryMode::Hybrid => {
            let embedder = crate::embedder::create_embedder(&config)?;
            crate::query::query(
                minsync_dir,
                text,
                k,
                embedder.as_ref(),
                store.as_ref(),
                None,
                mode,
            )
            .await?
        }
    };

    match format {
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

async fn status(format: &OutputFormat, minsync_dir: &std::path::Path) -> Result<()> {
    let result = crate::verify::status(minsync_dir).await?;
    match format {
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

async fn check(format: &OutputFormat, minsync_dir: &std::path::Path) -> Result<()> {
    let config = crate::config::Config::load(&minsync_dir.join("config.toml"))?;
    let embedder = crate::embedder::create_embedder(&config)?;
    let store_path = minsync_dir.join(&config.collection.path);
    let store = crate::vectorstore::create_vectorstore(&config, &store_path)?;

    let result = crate::verify::check(minsync_dir, embedder.as_ref(), store.as_ref()).await?;
    match format {
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

async fn verify(
    format: &OutputFormat,
    root: PathBuf,
    minsync_dir: &std::path::Path,
    fix: bool,
    all: bool,
    sample: usize,
) -> Result<()> {
    let config = crate::config::Config::load(&minsync_dir.join("config.toml"))?;
    let chunker = crate::chunker::create_chunker(&config)?;
    let store_path = minsync_dir.join(&config.collection.path);
    let mut store = crate::vectorstore::create_vectorstore(&config, &store_path)?;
    let sample_count = if all { None } else { Some(sample) };

    let result = crate::verify::verify(
        minsync_dir,
        &root,
        chunker.as_ref(),
        store.as_mut(),
        fix,
        sample_count,
    )
    .await?;
    match format {
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

async fn watch(
    format: &OutputFormat,
    root: PathBuf,
    minsync_dir: &std::path::Path,
    debounce_ms: Option<u64>,
    watch_on_sync_error: bool,
) -> Result<()> {
    let config = crate::config::Config::load(&minsync_dir.join("config.toml"))?;
    let chunker = crate::chunker::create_chunker(&config)?;
    let embedder: Box<dyn crate::embedder::Embedder> = if watch_on_sync_error {
        Box::new(crate::embedder::DeferredEmbedder::new(config.clone()))
    } else {
        crate::embedder::create_embedder(&config)?
    };
    let store_path = minsync_dir.join(&config.collection.path);
    let mut store = crate::vectorstore::create_vectorstore(&config, &store_path)?;

    if let OutputFormat::Text = format {
        println!("Watching {} for changes...", root.display());
    }

    crate::watch::run(
        root,
        chunker.as_ref(),
        embedder.as_ref(),
        store.as_mut(),
        debounce_ms,
        if watch_on_sync_error {
            crate::watch::WatchStartup::ContinueOnSyncError
        } else {
            crate::watch::WatchStartup::FailFast
        },
    )
    .await?;
    Ok(())
}
