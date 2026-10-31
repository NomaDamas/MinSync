use clap::Parser;
use minsync::cli::{Cli, Commands, QueryMode};

#[test]
fn init_parses_with_the_default_embedder() {
    let cli = Cli::try_parse_from(["minsync", "init"]).expect("init should parse");

    match cli.command {
        Commands::Init { embedder, .. } => assert!(!embedder.is_empty()),
        _ => panic!("expected init command"),
    }
}

#[test]
fn init_rejects_the_removed_language_flag() {
    let error = match Cli::try_parse_from(["minsync", "init", "--language", "simple"]) {
        Ok(_) => panic!("`init --language` was removed with the BM25 index"),
        Err(error) => error,
    };
    assert_eq!(error.exit_code(), 2);
}

#[test]
fn query_accepts_all_modes() {
    for (mode, expected) in [
        ("vector", QueryMode::Vector),
        ("sparse", QueryMode::Sparse),
        ("hybrid", QueryMode::Hybrid),
    ] {
        let cli = Cli::try_parse_from(["minsync", "query", "alpha", "--mode", mode])
            .expect("query mode should parse");

        match cli.command {
            Commands::Query { mode: actual, .. } => {
                assert_eq!(actual.expect("mode given").as_str(), expected.as_str())
            }
            _ => panic!("expected query command"),
        }
    }
}

#[test]
fn query_rejects_the_removed_bm25_mode() {
    let error = match Cli::try_parse_from(["minsync", "query", "alpha", "--mode", "bm25"]) {
        Ok(_) => panic!("`bm25` was replaced by `sparse`"),
        Err(error) => error,
    };
    assert_eq!(error.exit_code(), 2);
}

#[test]
fn query_mode_is_unset_unless_given() {
    let cli = Cli::try_parse_from(["minsync", "query", "alpha"]).expect("query parses");
    match cli.command {
        Commands::Query { mode, .. } => assert!(mode.is_none()),
        _ => panic!("expected query command"),
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
