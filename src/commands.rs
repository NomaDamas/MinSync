//! CLI command handlers. `main.rs` stays a thin binary: parse args, set up
//! logging, dispatch here, and map errors to exit codes.

use crate::cli::{Cli, Commands, OutputFormat, ProgressFormat, QueryMode};
use crate::error::MinSyncError;
use crate::error::Result;
use crate::sync::MinSync;
use std::io::Write;
use std::path::PathBuf;

/// Exit code of a sync that completed but skipped files that failed to
/// index (see `SyncResult.files_failed`). Hard failures keep their own codes
/// (`MinSyncError::exit_code`).
pub const EXIT_PARTIAL_SYNC: i32 = 6;

/// Run a command and return the process exit code on success.
pub async fn run(cli: Cli, root: PathBuf, minsync_dir: PathBuf) -> Result<i32> {
    if let Commands::Sync {
        full,
        dry_run,
        wait,
        batch_size,
        progress,
        progress_file,
        progress_heartbeat_seconds,
    } = cli.command
    {
        let progress = ProgressOptions {
            stdout: progress == Some(ProgressFormat::Jsonl),
            file: progress_file,
            heartbeat: std::time::Duration::from_secs(progress_heartbeat_seconds),
        };
        return sync(
            &cli.format,
            root,
            &minsync_dir,
            full,
            dry_run,
            wait,
            batch_size,
            progress,
        )
        .await;
    }
    if let Commands::Serve {
        host,
        port,
        idle_timeout,
        max_inputs,
        max_request_bytes,
        state_dir,
        embedder,
    } = cli.command
    {
        let options = ServeArgs {
            host,
            port,
            idle_timeout,
            max_inputs,
            max_request_bytes,
            state_dir,
            embedder,
        };
        return serve(&cli.format, &minsync_dir, options).await;
    }
    if matches!(cli.command, Commands::Check) {
        return check(&cli.format, &minsync_dir).await;
    }
    run_other(cli, root, minsync_dir).await.map(|()| 0)
}

async fn run_other(cli: Cli, root: PathBuf, minsync_dir: PathBuf) -> Result<()> {
    match cli.command {
        Commands::Init {
            force,
            embedder,
            chunker,
            language,
        } => init(&cli.format, root, force, &embedder, &chunker, &language),
        Commands::Sync { .. } | Commands::Serve { .. } => unreachable!("handled by run"),
        Commands::Query { text, k, mode } => query(&cli.format, &minsync_dir, &text, k, mode).await,
        Commands::Status => status(&cli.format, &minsync_dir).await,
        Commands::Optimize => optimize(&cli.format, &minsync_dir),
        Commands::Check => unreachable!("handled by run"),
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

/// Where `minsync sync` sends its machine-readable progress stream.
struct ProgressOptions {
    /// `--progress jsonl`: events go to stdout.
    stdout: bool,
    /// `--progress-file`: events go to this file.
    file: Option<PathBuf>,
    heartbeat: std::time::Duration,
}

impl ProgressOptions {
    fn reporter(
        &self,
        root: &std::path::Path,
    ) -> Result<Option<std::sync::Arc<crate::sync::ProgressReporter>>> {
        let mut writers: Vec<Box<dyn std::io::Write + Send>> = Vec::new();
        if let Some(path) = &self.file {
            check_progress_file_location(root, path)?;
            let file = std::fs::File::create(path).map_err(|error| {
                MinSyncError::Other(format!(
                    "cannot open progress file {}: {error}",
                    path.display()
                ))
            })?;
            writers.push(Box::new(file));
        }
        if self.stdout {
            writers.push(Box::new(std::io::stdout()));
        }
        Ok((!writers.is_empty()).then(|| {
            std::sync::Arc::new(crate::sync::ProgressReporter::new(writers, self.heartbeat))
        }))
    }
}

/// Refuse a `--progress-file` that would clobber MinSync state (`.minsync/`,
/// e.g. the resume log or cursor) or that would be indexed as a source file.
/// Paths outside the workspace are always fine; paths inside it must be
/// excluded by the root `.minsyncignore`. Runs before the file is truncated.
fn check_progress_file_location(root: &std::path::Path, path: &std::path::Path) -> Result<()> {
    let Some(file_name) = path.file_name() else {
        return Ok(()); // let File::create report the unusable path
    };
    let absolute = root.join(path);
    let parent = absolute.parent().unwrap_or(root);
    // Unresolvable parent: File::create reports the real error.
    let (Ok(parent), Ok(root)) = (parent.canonicalize(), root.canonicalize()) else {
        return Ok(());
    };
    let target = parent.join(file_name);

    if target.starts_with(root.join(".minsync")) {
        return Err(MinSyncError::Other(format!(
            "progress file {} must not be inside .minsync/ (it would overwrite MinSync state)",
            path.display()
        )));
    }
    let Ok(relative) = target.strip_prefix(&root) else {
        return Ok(());
    };
    let (ignore, _) = ignore::gitignore::Gitignore::new(root.join(".minsyncignore"));
    if !ignore
        .matched_path_or_any_parents(relative, false)
        .is_ignore()
    {
        return Err(MinSyncError::Other(format!(
            "progress file {} is inside the indexed workspace; add it to .minsyncignore or write it outside the workspace",
            path.display()
        )));
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
    progress: ProgressOptions,
) -> Result<i32> {
    require_initialized(minsync_dir)?;
    // Fail on an unusable progress sink before taking the lock or syncing.
    let reporter = progress.reporter(&root)?;
    let mut ms = MinSync::new(root.clone());
    if let Some(reporter) = &reporter {
        ms = ms.with_progress(reporter.clone());
    }
    let _lock = crate::state::FileLock::acquire(&minsync_dir.join("lock"), wait)?;
    let mut config = crate::config::Config::load(&minsync_dir.join("config.toml"))?;
    if let Some(bs) = batch_size {
        config.embedder.batch_size = bs;
    }
    let chunker = crate::chunker::create_chunker(&config)?;
    let embedder = crate::embedder::create_embedder(&config)?;
    let store_path = crate::sync::collection_store_path(minsync_dir, &config.collection.path)?;
    let mut store = crate::vectorstore::create_vectorstore(&config, &store_path)?;

    let result = ms
        .sync_locked(
            chunker.as_ref(),
            embedder.as_ref(),
            store.as_mut(),
            full,
            dry_run,
        )
        .await?;

    // stdout carries the event stream when `--progress jsonl` is on, so the
    // human summary moves to stderr and the final JSON result becomes the
    // last, compact line (one JSON value per line). Otherwise unchanged.
    let events_on_stdout = progress.stdout;
    let mut out: Box<dyn std::io::Write> = if events_on_stdout {
        Box::new(std::io::stderr())
    } else {
        Box::new(std::io::stdout())
    };
    match format {
        OutputFormat::Text => {
            if result.already_up_to_date {
                writeln!(out, "Already up to date.")?;
            } else if result.dry_run {
                writeln!(
                    out,
                    "Dry run: {} files would be processed",
                    result.files_processed_paths.len()
                )?;
            } else {
                if result.initial_sync {
                    writeln!(out, "Initial sync: no cursor found — performed full sync.")?;
                }
                writeln!(out, "Sync complete:")?;
                writeln!(out, "  files processed: {}", result.files_processed)?;
                writeln!(out, "  files added:     {}", result.files_added)?;
                writeln!(out, "  files modified:  {}", result.files_modified)?;
                writeln!(out, "  files deleted:   {}", result.files_deleted)?;
                writeln!(out, "  chunks inserted: {}", result.chunks_added)?;
                writeln!(out, "  chunks reused:   {}", result.chunks_updated)?;
                writeln!(out, "  chunks removed:  {}", result.chunks_deleted)?;
                writeln!(out)?;
                writeln!(out, "Sync Stats")?;
                writeln!(out, "  elapsed time:        {:.2}s", result.elapsed_seconds)?;
                writeln!(out, "  embedding API calls: {}", result.embedding_api_calls)?;
                writeln!(out, "  embedded texts:      {}", result.embedded_texts)?;
                writeln!(out, "  embeddings reused:   {}", result.embeddings_reused)?;
                writeln!(out, "  chunks truncated:    {}", result.chunks_truncated)?;
                writeln!(out, "  scan errors:         {}", result.scan_errors.len())?;
                writeln!(out, "  estimated tokens:    {}", result.estimated_tokens)?;
            }
            if result.partial {
                eprintln!(
                    "Warning: {} file(s) skipped; they will be retried by the next sync:",
                    result.files_failed.len()
                );
                for failure in &result.files_failed {
                    eprintln!("  {} [{}]: {}", failure.path, failure.stage, failure.error);
                }
            }
        }
        OutputFormat::Json => {
            if events_on_stdout {
                writeln!(std::io::stdout(), "{}", serde_json::to_string(&result)?)?;
            } else {
                writeln!(out, "{}", serde_json::to_string_pretty(&result)?)?;
            }
        }
    }
    if matches!(format, OutputFormat::Text) {
        // Entries already listed as skipped files above are not repeated.
        for error in result
            .scan_errors
            .iter()
            .filter(|error| !result.files_failed.iter().any(|f| f.path == error.path))
        {
            eprintln!(
                "warning: scan error ({:?}) {}: {} — existing chunks kept",
                error.kind, error.path, error.error
            );
        }
    }
    Ok(if result.partial { EXIT_PARTIAL_SYNC } else { 0 })
}

async fn query(
    format: &OutputFormat,
    minsync_dir: &std::path::Path,
    text: &str,
    k: usize,
    mode: Option<QueryMode>,
) -> Result<()> {
    require_initialized(minsync_dir)?;
    // No writer lock here: querying only reads. LanceDB gives snapshot
    // isolation (the table pins the latest version at open time) and the
    // read-only store never creates or migrates anything, so search stays
    // available while a sync holds the lock. Results may lag the in-flight
    // sync; they are always consistent.
    let config = crate::config::Config::load(&minsync_dir.join("config.toml"))?;
    let mode = mode.unwrap_or(if config.is_lexical_only() {
        QueryMode::Bm25
    } else {
        QueryMode::Vector
    });
    let store_path = crate::sync::collection_store_path(minsync_dir, &config.collection.path)?;
    let store = crate::vectorstore::create_vectorstore_readonly(&config, &store_path)?;
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

/// `minsync optimize`: compact + re-index + prune only. Takes the writer
/// lock (never races a sync) but embeds nothing and reads no source files,
/// so it can repair a store too fragmented for `sync` to make progress.
fn optimize(format: &OutputFormat, minsync_dir: &std::path::Path) -> Result<()> {
    require_initialized(minsync_dir)?;
    let _lock = crate::state::FileLock::acquire(&minsync_dir.join("lock"), false)?;
    let config = crate::config::Config::load(&minsync_dir.join("config.toml"))?;
    let store_path = crate::sync::collection_store_path(minsync_dir, &config.collection.path)?;
    let mut store = crate::vectorstore::create_vectorstore(&config, &store_path)?;
    let started = std::time::Instant::now();
    store.flush()?;
    let elapsed = started.elapsed().as_secs_f64();
    match format {
        OutputFormat::Text => println!("Optimized store in {elapsed:.2}s."),
        OutputFormat::Json => println!(
            "{}",
            serde_json::json!({ "optimized": true, "elapsed_seconds": elapsed })
        ),
    }
    Ok(())
}

async fn check(format: &OutputFormat, minsync_dir: &std::path::Path) -> Result<i32> {
    require_initialized(minsync_dir)?;
    let _lock = crate::state::FileLock::acquire(&minsync_dir.join("lock"), false)?;
    let config = crate::config::Config::load(&minsync_dir.join("config.toml"))?;
    let embedder = crate::embedder::create_embedder(&config)?;
    let store_path = crate::sync::collection_store_path(minsync_dir, &config.collection.path)?;
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
    // The report above is for a person or a host that reads it; the exit code is for the host that does not.
    // `sync` already exits with these codes when the same stage fails, so a host maps them the same way.
    // The reason goes to stderr as it is: `result.errors` already says which stage failed.
    let failed = if !result.embedder_ok {
        Some(MinSyncError::Embedding(String::new()).exit_code())
    } else if !result.vectorstore_ok {
        Some(MinSyncError::VectorStore(String::new()).exit_code())
    } else {
        None
    };
    if let Some(code) = failed {
        for error in &result.errors {
            eprintln!("Error: {error}");
        }
        return Ok(code);
    }
    Ok(0)
}

async fn verify(
    format: &OutputFormat,
    root: PathBuf,
    minsync_dir: &std::path::Path,
    fix: bool,
    all: bool,
    sample: usize,
) -> Result<()> {
    require_initialized(minsync_dir)?;
    let _lock = crate::state::FileLock::acquire(&minsync_dir.join("lock"), false)?;
    let config = crate::config::Config::load(&minsync_dir.join("config.toml"))?;
    let chunker = crate::chunker::create_chunker(&config)?;
    let store_path = crate::sync::collection_store_path(minsync_dir, &config.collection.path)?;
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
                if !result.embedding_integrity.flagged_pairs.is_empty() {
                    println!("Embedding integrity flagged pairs:");
                    for pair in &result.embedding_integrity.flagged_pairs {
                        println!(
                            "  chunks {} ({})",
                            pair.chunk_ids.join(", "),
                            pair.paths.join(", ")
                        );
                    }
                    println!("Recommendation: run minsync sync --full");
                }
            }
        }
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
    }
    if result.all_passed {
        Ok(())
    } else {
        Err(MinSyncError::Other(
            "verification failed — run `minsync verify --fix` to repair".to_string(),
        ))
    }
}

async fn watch(
    format: &OutputFormat,
    root: PathBuf,
    minsync_dir: &std::path::Path,
    debounce_ms: Option<u64>,
    watch_on_sync_error: bool,
) -> Result<()> {
    require_initialized(minsync_dir)?;
    let _lock = crate::state::FileLock::acquire(&minsync_dir.join("lock"), false)?;
    let config = crate::config::Config::load(&minsync_dir.join("config.toml"))?;
    let chunker = crate::chunker::create_chunker(&config)?;
    let embedder: Box<dyn crate::embedder::Embedder> = if watch_on_sync_error {
        Box::new(crate::embedder::DeferredEmbedder::new(config.clone()))
    } else {
        crate::embedder::create_embedder(&config)?
    };
    let store_path = crate::sync::collection_store_path(minsync_dir, &config.collection.path)?;
    let mut store = crate::vectorstore::create_vectorstore(&config, &store_path)?;

    if let OutputFormat::Text = format {
        println!("Watching {} for changes...", root.display());
    }

    crate::watch::run_locked(
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

struct ServeArgs {
    host: String,
    port: u16,
    idle_timeout: u64,
    max_inputs: usize,
    max_request_bytes: usize,
    state_dir: Option<PathBuf>,
    embedder: Option<String>,
}

/// Embedder settings `serve` runs with: the workspace config when present
/// (its device/dtype/limits/prefixes), else defaults; `--embedder`
/// overrides the id.
fn serve_settings(
    minsync_dir: &std::path::Path,
    embedder: Option<String>,
) -> Result<crate::config::EmbedderConfig> {
    let config_path = minsync_dir.join("config.toml");
    let mut settings = if config_path.exists() {
        crate::config::Config::load(&config_path)?.embedder
    } else {
        crate::config::Config::default_for("serve").embedder
    };
    if let Some(id) = embedder {
        settings.id = if id.contains(':') {
            id
        } else {
            format!("native:{id}")
        };
    }
    Ok(settings)
}

async fn serve(
    format: &OutputFormat,
    minsync_dir: &std::path::Path,
    args: ServeArgs,
) -> Result<i32> {
    let host = crate::serve::parse_loopback_host(&args.host)?;
    let settings = serve_settings(minsync_dir, args.embedder)?;
    let instance = crate::serve::instance_key(&settings)?;
    let engine = crate::embedder::native::NativeEmbedder::from_config(&settings)?;
    let options = crate::serve::ServeOptions {
        host,
        port: args.port,
        idle_timeout: (args.idle_timeout > 0)
            .then(|| std::time::Duration::from_secs(args.idle_timeout)),
        max_inputs: args.max_inputs.max(1),
        max_request_bytes: args.max_request_bytes.max(1),
        state_dir: args
            .state_dir
            .unwrap_or_else(|| crate::serve::default_state_dir(&instance)),
        instance,
        query_prefix: settings.query_prefix.clone(),
        passage_prefix: settings.passage_prefix.clone(),
    };
    let state_dir = options.state_dir.clone();
    let print = |event: &str, health: &crate::serve::Health| -> Result<()> {
        match format {
            OutputFormat::Json => {
                let mut value = serde_json::to_value(health)?;
                value["event"] = serde_json::Value::from(event);
                value["state_dir"] = serde_json::Value::from(state_dir.display().to_string());
                println!("{}", serde_json::to_string(&value)?);
            }
            OutputFormat::Text => println!(
                "{event}: {} on {} ({} dims, {}, {}, pid {}); state {}",
                health.model_id,
                health.url,
                health.dimension,
                health.device,
                health.dtype,
                health.pid,
                state_dir.display()
            ),
        }
        std::io::stdout().flush()?;
        Ok(())
    };
    let server = match crate::serve::start(std::sync::Arc::new(engine), options).await? {
        crate::serve::Started::Adopted(health) => {
            print("adopted", &health)?;
            return Ok(0);
        }
        crate::serve::Started::Serving(server) => server,
    };
    print("serving", server.health())?;
    let reason = server.run_until(shutdown_signal()).await?;
    tracing::info!(reason = reason.as_str(), "minsync serve stopped");
    Ok(reason.exit_code())
}

/// SIGTERM or Ctrl-C.
async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        match signal(SignalKind::terminate()) {
            Ok(mut term) => {
                tokio::select! {
                    _ = term.recv() => {}
                    _ = tokio::signal::ctrl_c() => {}
                }
            }
            Err(_) => {
                let _ = tokio::signal::ctrl_c().await;
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

fn require_initialized(minsync_dir: &std::path::Path) -> Result<()> {
    if !minsync_dir.join("config.toml").exists() {
        return Err(MinSyncError::NotInitialized);
    }
    Ok(())
}
