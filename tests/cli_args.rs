use clap::Parser;
use minsync::cli::{Cli, Commands, QueryMode};

#[test]
fn init_accepts_language() {
    let cli = Cli::try_parse_from(["minsync", "init", "--language", "simple"])
        .expect("init --language should parse");

    match cli.command {
        Commands::Init { language, .. } => assert_eq!(language, "simple"),
        _ => panic!("expected init command"),
    }
}

#[test]
fn query_accepts_all_modes() {
    for (mode, expected) in [
        ("vector", QueryMode::Vector),
        ("bm25", QueryMode::Bm25),
        ("hybrid", QueryMode::Hybrid),
    ] {
        let cli = Cli::try_parse_from(["minsync", "query", "alpha", "--mode", mode])
            .expect("query mode should parse");

        match cli.command {
            Commands::Query { mode: actual, .. } => assert_eq!(actual.as_str(), expected.as_str()),
            _ => panic!("expected query command"),
        }
    }
}

#[test]
fn malformed_flags_are_rejected() {
    let result = Cli::try_parse_from(["minsync", "query", "alpha", "--mode", "nope"]);
    let error = match result {
        Ok(_) => panic!("unknown query mode should be rejected"),
        Err(error) => error,
    };

    assert_eq!(error.exit_code(), 2);
    assert!(error.to_string().contains("possible values"));
}
