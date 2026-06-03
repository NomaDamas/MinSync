use std::fs;

fn read_file(path: &str) -> String {
    fs::read_to_string(path).expect(path)
}

#[test]
fn ci_workflow_defines_required_multi_os_validation() {
    let workflow = read_file(".github/workflows/ci.yml");

    for needle in [
        "pull_request",
        "push",
        "main",
        "ubuntu-latest",
        "macos-latest",
        "windows-latest",
        "1.91",
        "cargo fmt --all -- --check",
        "cargo clippy --all-targets --all-features -- -D warnings",
        "cargo test",
        "cargo build --release",
        "Swatinem/rust-cache@v2",
    ] {
        assert!(
            workflow.contains(needle),
            "expected workflow to contain {needle:?}"
        );
    }

    assert!(
        !workflow.contains("secrets."),
        "workflow should not reference secrets"
    );
}

#[test]
fn readme_documents_supported_ci_assumptions() {
    let readme = read_file("README.md");

    for needle in [
        "Ubuntu",
        "macOS",
        "Windows",
        "Rust 1.91",
        "rustup",
        "C compiler",
        "vendored protoc",
        "LanceDB",
        "no secrets",
    ] {
        assert!(
            readme.contains(needle),
            "expected README to contain {needle:?}"
        );
    }
}
