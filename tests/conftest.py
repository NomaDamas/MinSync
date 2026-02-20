"""Shared fixtures and helpers for MinSync E2E tests."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from tests.mock_components import MockChunker, MockEmbedder, MockVectorStore

# ---------------------------------------------------------------------------
# Sample file contents (from E2E_TEST_PLAN.md)
# ---------------------------------------------------------------------------

SAMPLE_FILES: dict[str, str] = {
    "docs/guide.md": """\
# User Guide

## Getting Started

MinSync is a git-native vector index sync engine.
It detects changes via git diff and incrementally updates the vector database.

## Installation

Install MinSync using pip:

pip install minsync

## Configuration

After installation, run `minsync init` in your git repository.
""",
    "docs/api.md": """\
# API Reference

## Sync API

The sync command processes file changes and updates the vector index.

### Parameters

- `--ref`: Target branch (default: main)
- `--full`: Force full rebuild

## Query API

Search the vector index with natural language queries.
""",
    "docs/auth/login.md": """\
# Authentication

## Login Process

The login process begins with the user entering their credentials.
The system validates the credentials against the authentication provider.

## Session Management

After successful login, a session token is issued.
""",
    "docs/auth/oauth.md": """\
# OAuth Integration

## OAuth2 Flow

For third-party authentication, we use OAuth2.
The flow follows the authorization code grant type.

## Provider Configuration

Supported providers: Google, GitHub, Microsoft.
""",
    "docs/faq.md": """\
# FAQ

## How does sync work?

MinSync uses git diff to detect file changes between commits.

## Is it safe to run during CI?

Yes, MinSync uses file locking to prevent concurrent access.
""",
    "src/main.py": """\
def main():
    print("Hello, world!")

if __name__ == "__main__":
    main()
""",
    "src/utils.py": """\
def helper():
    return 42
""",
    "notes/meeting.txt": """\
Meeting notes 2026-02-20
- Discussed MinSync roadmap
- Decided on git-native approach
""",
    "README.md": """\
# MinSync

A git-native vector index sync engine.
""",
}


# ---------------------------------------------------------------------------
# Git helper functions
# ---------------------------------------------------------------------------


def _run_git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=check,
    )


def create_test_repo(tmp_path: Path, files: dict[str, str] | None = None) -> Path:
    """Create a git repository with an initial commit containing *files*."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _run_git(repo, "init", "-b", "main")
    _run_git(repo, "config", "user.email", "test@minsync.dev")
    _run_git(repo, "config", "user.name", "Test")

    if files is None:
        files = SAMPLE_FILES

    for rel_path, content in files.items():
        fpath = repo / rel_path
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_text(content, encoding="utf-8")

    _run_git(repo, "add", "-A")
    _run_git(repo, "commit", "-m", "initial commit")
    return repo


def add_commit(repo: Path, files: dict[str, str], message: str = "update") -> str:
    """Add/modify files and commit. Returns the new commit hash."""
    for rel_path, content in files.items():
        fpath = repo / rel_path
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_text(content, encoding="utf-8")
    _run_git(repo, "add", "-A")
    _run_git(repo, "commit", "-m", message)
    return get_head(repo)


def delete_commit(repo: Path, paths: list[str], message: str = "delete files") -> str:
    """Delete files and commit. Returns the new commit hash."""
    for p in paths:
        fpath = repo / p
        if fpath.exists():
            fpath.unlink()
    _run_git(repo, "add", "-A")
    _run_git(repo, "commit", "-m", message)
    return get_head(repo)


def rename_commit(repo: Path, old: str, new: str, message: str = "rename file") -> str:
    """Rename a file via git mv and commit. Returns the new commit hash."""
    _run_git(repo, "mv", old, new)
    _run_git(repo, "commit", "-m", message)
    return get_head(repo)


def get_head(repo: Path) -> str:
    """Return full HEAD commit hash."""
    result = _run_git(repo, "rev-parse", "HEAD")
    return result.stdout.strip()


def get_root_commit(repo: Path) -> str:
    """Return the root (first) commit hash — used as repo_id."""
    result = _run_git(repo, "rev-list", "--max-parents=0", "HEAD")
    # may have multiple lines for octopus merges; take last
    lines = result.stdout.strip().splitlines()
    return lines[-1].strip()


def write_minsyncignore(repo: Path, content: str) -> None:
    """Write a .minsyncignore file at the repo root."""
    (repo / ".minsyncignore").write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# State inspection helpers
# ---------------------------------------------------------------------------


def get_cursor(repo: Path) -> dict:
    """Read and return .minsync/cursor.json."""
    cursor_path = repo / ".minsync" / "cursor.json"
    return json.loads(cursor_path.read_text(encoding="utf-8"))


def get_config(repo: Path) -> dict:
    """Read and return .minsync/config.yaml as a dict."""
    config_path = repo / ".minsync" / "config.yaml"
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def test_repo(tmp_path: Path) -> Path:
    """Create a git repo with all SAMPLE_FILES committed."""
    return create_test_repo(tmp_path)


@pytest.fixture()
def mock_chunker() -> MockChunker:
    return MockChunker()


@pytest.fixture()
def mock_embedder() -> MockEmbedder:
    return MockEmbedder()


@pytest.fixture()
def mock_vector_store() -> MockVectorStore:
    return MockVectorStore()


@pytest.fixture()
def minsync_instance(test_repo, mock_chunker, mock_embedder, mock_vector_store):
    """Create a MinSync instance with mock components.

    Returns ``(ms, repo)`` tuple.
    """
    from minsync import MinSync

    ms = MinSync(
        repo_path=test_repo,
        chunker=mock_chunker,
        embedder=mock_embedder,
        vector_store=mock_vector_store,
    )
    return ms, test_repo


@pytest.fixture()
def initialized_repo(minsync_instance):
    """MinSync instance after ``init()`` has been called.

    Returns ``(repo, ms)`` tuple.
    """
    ms, repo = minsync_instance
    ms.init()
    return repo, ms


@pytest.fixture()
def synced_repo(initialized_repo):
    """MinSync instance after ``init()`` + ``sync()`` have been called.

    Returns ``(repo, ms)`` tuple.
    """
    repo, ms = initialized_repo
    ms.sync()
    return repo, ms
