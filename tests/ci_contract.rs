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
        "arduino/setup-protoc@v3",
        "repo-token: ${{ github.token }}",
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

    let protoc_step = workflow
        .find("arduino/setup-protoc@v3")
        .expect("expected workflow to define setup-protoc");
    let clippy_step = workflow
        .find("cargo clippy --all-targets --all-features -- -D warnings")
        .expect("expected workflow to define clippy");
    assert!(
        protoc_step < clippy_step,
        "expected setup-protoc to run before clippy"
    );

    assert!(
        !workflow.contains("secrets."),
        "workflow should not reference secrets"
    );
}

#[test]
fn readme_documents_supported_ci_assumptions() {
    let readme = read_file("README.md");

    for needle in [
        "assets/minsync-flow.svg",
        "curl -fsSL https://raw.githubusercontent.com/NomaDamas/MinSync/main/scripts/install.sh | sh",
        "gh repo star NomaDamas/MinSync",
        "npx skills add github:NomaDamas/MinSync/skills/minsync",
        "docs/RELEASE.md",
        "Ubuntu",
        "macOS",
        "Windows",
        "Rust 1.91",
        "rustup",
        "C compiler",
        "setup-protoc",
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

#[test]
fn release_checklist_documents_launch_gates() {
    let release = read_file("docs/RELEASE.md");

    for needle in [
        "Version and Metadata",
        "Documentation",
        "CI",
        "Install QA",
        "Agent Skill QA",
        "Release Automation",
        "Post-Release Smoke Test",
        "Rollback",
        "cargo package --list",
        "CARGO_REGISTRY_TOKEN",
        "cargo yank",
    ] {
        assert!(
            release.contains(needle),
            "expected release checklist to contain {needle:?}"
        );
    }
}

#[test]
fn installer_prompts_for_optional_repo_star() {
    let installer = read_file("scripts/install.sh");

    for needle in [
        "gh repo star",
        "NomaDamas/MinSync",
        "--yes-star",
        "--no-star",
        "--dry-run",
        "cargo install minsync",
        "GitHub CLI not found; skipping optional repo star.",
    ] {
        assert!(
            installer.contains(needle),
            "expected installer to contain {needle:?}"
        );
    }
}

#[test]
fn agent_skill_packages_minsync_operating_instructions() {
    let skill = read_file("skills/minsync/SKILL.md");

    for needle in [
        "name: minsync",
        "description:",
        "gh repo star NomaDamas/MinSync",
        "cargo install minsync",
        "minsync init",
        "minsync sync --full",
        "minsync query",
        ".minsyncignore",
        "UTF-8 text only",
        "tei:intfloat/multilingual-e5-small",
    ] {
        assert!(
            skill.contains(needle),
            "expected agent skill to contain {needle:?}"
        );
    }
}

#[test]
fn windows_ci_uses_setup_protoc_instead_of_vendored_protobuf_src() {
    let manifest = read_file("Cargo.toml");

    assert!(
        manifest.contains("[target.'cfg(not(windows))'.build-dependencies]"),
        "expected protobuf-src build dependency to be disabled on Windows"
    );
    assert!(
        !manifest.contains("\n[build-dependencies]\nprotobuf-src"),
        "expected protobuf-src not to be an unconditional build dependency"
    );
    assert!(
        manifest.contains("protobuf-src = \"2.1\""),
        "expected non-Windows builds to keep vendored protobuf-src"
    );
}
