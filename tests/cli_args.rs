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
fn init_defaults_to_the_line_cdc_chunker() {
    let cli = Cli::try_parse_from(["minsync", "init"]).expect("init should parse");

    match cli.command {
        Commands::Init { chunker, .. } => assert_eq!(chunker, "line-cdc"),
        _ => panic!("expected init command"),
    }
}

#[test]
fn init_keeps_a_chunker_chosen_on_the_command_line() {
    let cli = Cli::try_parse_from(["minsync", "init", "--chunker", "recursive"])
        .expect("init should parse");

    match cli.command {
        Commands::Init { chunker, .. } => assert_eq!(chunker, "recursive"),
        _ => panic!("expected init command"),
    }
}

#[test]
fn init_accepts_sparse_and_language_flags() {
    let cli = Cli::try_parse_from([
        "minsync",
        "init",
        "--embedder",
        "none",
        "--sparse",
        "none",
        "--language",
        "ko",
    ])
    .expect("init flags should parse");
    match cli.command {
        Commands::Init {
            embedder,
            sparse,
            language,
            ..
        } => {
            assert_eq!(embedder, "none");
            assert_eq!(sparse, "none");
            assert_eq!(language, "ko");
        }
        _ => panic!("expected init command"),
    }
}

#[test]
fn query_accepts_all_modes_and_independent_retriever_toggles() {
    for (mode, expected) in [
        ("vector", QueryMode::Vector),
        ("sparse", QueryMode::Sparse),
        ("bm25", QueryMode::Bm25),
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

    // Independent on/off toggles for all 3 retrievers.
    let cli = Cli::try_parse_from([
        "minsync", "query", "alpha", "--vector", "--sparse", "--bm25",
    ])
    .expect("all three toggles should parse");
    match cli.command {
        Commands::Query {
            vector,
            sparse,
            bm25,
            mode,
            ..
        } => {
            assert_eq!(vector, Some(true));
            assert_eq!(sparse, Some(true));
            assert_eq!(bm25, Some(true));
            assert!(mode.is_none());
        }
        _ => panic!("expected query command"),
    }

    let neg = Cli::try_parse_from(["minsync", "query", "alpha", "--sparse=false"])
        .expect("negative toggle should parse");
    match neg.command {
        Commands::Query {
            vector,
            sparse,
            bm25,
            ..
        } => {
            assert_eq!(vector, None);
            assert_eq!(sparse, Some(false));
            assert_eq!(bm25, None);
        }
        _ => panic!("expected query command"),
    }
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
