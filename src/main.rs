use clap::Parser;
use minsync::cli::{Cli, OutputFormat};
use std::process;

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    let json_output = matches!(cli.format, OutputFormat::Json);

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

    match minsync::commands::run(cli, root, minsync_dir).await {
        Ok(code) => process::exit(code),
        Err(error) => {
            if json_output {
                if let Some(value) = minsync::commands::json_error(&error) {
                    println!("{value}");
                }
            }
            eprintln!("Error: {error}");
            process::exit(error.exit_code());
        }
    }
}
