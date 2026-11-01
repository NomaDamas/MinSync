//! What `minsync init` writes for a new workspace: the `line-cdc` chunker with
//! heading context on. Workspaces that already exist are covered by the config
//! tests: a `config.toml` that does not mention `heading_context` still means
//! "off", so nothing about them is chunked again.

use assert_cmd::Command;
use minsync::chunker::create_chunker;
use minsync::config::Config;
use std::path::Path;

fn init(root: &Path, extra_args: &[&str]) -> Config {
    Command::cargo_bin("minsync")
        .expect("find minsync binary")
        .current_dir(root)
        .args(["init", "--embedder", "tei:test/model"])
        .args(extra_args)
        .assert()
        .success();
    Config::load(&root.join(".minsync/config.toml")).expect("load config")
}

#[test]
fn a_new_workspace_starts_with_line_cdc_and_heading_context() {
    let dir = tempfile::tempdir().unwrap();

    let config = init(dir.path(), &[]);

    assert_eq!(config.chunker.id, "line-cdc");
    assert!(config.chunker.options.heading_context);
    let written = std::fs::read_to_string(dir.path().join(".minsync/config.toml")).unwrap();
    assert!(written.contains("heading_context = true"), "{written}");
}

#[test]
fn choosing_another_chunker_keeps_heading_context_on() {
    let dir = tempfile::tempdir().unwrap();

    let config = init(dir.path(), &["--chunker", "recursive"]);

    assert_eq!(config.chunker.id, "recursive");
    assert!(config.chunker.options.heading_context);
    assert_eq!(
        create_chunker(&config).unwrap().schema_id(),
        "recursive+heading"
    );
}

#[test]
fn the_default_chunker_starts_every_chunk_with_its_path() {
    let dir = tempfile::tempdir().unwrap();
    let config = init(dir.path(), &[]);
    let chunker = create_chunker(&config).unwrap();

    let chunks = chunker
        .chunk(
            "# Guide\n\n## Install\n\nRun brew install minsync.\n",
            "docs/guide.md",
        )
        .unwrap();

    assert!(!chunks.is_empty());
    for chunk in &chunks {
        assert!(
            chunk.text.starts_with("docs/guide.md\n"),
            "chunk text: {:?}",
            chunk.text
        );
        assert!(chunk.heading_path.starts_with("docs/guide.md"));
    }
}
