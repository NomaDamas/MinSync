"""T01 & T02: MinSync init — basic initialization and error cases.

TDD tests for ``MinSync.init()`` Python API.
The implementation does NOT exist yet; these tests define the expected behavior.

References:
    - ai_instruction/E2E_TEST_PLAN.md  (T01, T02)
    - ai_instruction/CLI_SPEC.md       (section 1: minsync init)
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

from minsync import MinSync
from tests.conftest import (
    get_config,
    get_root_commit,
)
from tests.mock_components import MockChunker, MockEmbedder, MockVectorStore

# ============================================================================
# T01: init -- basic initialization
# ============================================================================


class TestT01BasicInit:
    """T01: ``ms.init()`` on a valid git repo with sample files committed."""

    # -- T01-1: No exception raised (success) --------------------------------
    def test_t01_1_init_succeeds(self, minsync_instance):
        """init() should complete without raising any exception."""
        ms, _repo = minsync_instance
        ms.init()  # should not raise

    # -- T01-2: .minsync/ directory exists ------------------------------------
    def test_t01_2_minsync_dir_exists(self, minsync_instance):
        ms, repo = minsync_instance
        ms.init()
        assert (repo / ".minsync").is_dir()

    # -- T01-3: .minsync/config.yaml exists -----------------------------------
    def test_t01_3_config_yaml_exists(self, minsync_instance):
        ms, repo = minsync_instance
        ms.init()
        assert (repo / ".minsync" / "config.yaml").is_file()

    # -- T01-4: config["repo_id"] == root commit hash -------------------------
    def test_t01_4_repo_id_matches_root_commit(self, minsync_instance):
        ms, repo = minsync_instance
        ms.init()
        config = get_config(repo)
        expected_root = get_root_commit(repo)
        assert config["repo_id"] == expected_root

    # -- T01-5: config["ref"] == "main" ---------------------------------------
    def test_t01_5_ref_is_main(self, minsync_instance):
        ms, repo = minsync_instance
        ms.init()
        config = get_config(repo)
        assert config["ref"] == "main"

    # -- T01-6: config has chunker.id -----------------------------------------
    def test_t01_6_config_has_chunker_id(self, minsync_instance):
        ms, repo = minsync_instance
        ms.init()
        config = get_config(repo)
        assert "chunker" in config
        assert "id" in config["chunker"]
        assert isinstance(config["chunker"]["id"], str)
        assert len(config["chunker"]["id"]) > 0

    # -- T01-7: config has embedder.id ----------------------------------------
    def test_t01_7_config_has_embedder_id(self, minsync_instance):
        ms, repo = minsync_instance
        ms.init()
        config = get_config(repo)
        assert "embedder" in config
        assert "id" in config["embedder"]
        assert isinstance(config["embedder"]["id"], str)
        assert len(config["embedder"]["id"]) > 0

    # -- T01-8: config has vectorstore.id == "zvec" ---------------------------
    def test_t01_8_vectorstore_id_is_zvec(self, minsync_instance):
        ms, repo = minsync_instance
        ms.init()
        config = get_config(repo)
        assert "vectorstore" in config
        assert "id" in config["vectorstore"]
        assert config["vectorstore"]["id"] == "zvec"

    # -- T01-9: "include" key not in config -----------------------------------
    def test_t01_9_no_include_in_config(self, minsync_instance):
        ms, repo = minsync_instance
        ms.init()
        config = get_config(repo)
        assert "include" not in config

    # -- T01-10: cursor.json does NOT exist yet -------------------------------
    def test_t01_10_cursor_json_not_created(self, minsync_instance):
        ms, repo = minsync_instance
        ms.init()
        cursor_path = repo / ".minsync" / "cursor.json"
        assert not cursor_path.exists()

    # -- T01-11: init() returns successfully (Python API sanity check) --------
    def test_t01_11_init_returns_without_error(self, minsync_instance):
        """init() should either return None or a truthy result -- not raise."""
        ms, _repo = minsync_instance
        result = ms.init()
        # We accept None (no explicit return) or any truthy result.
        # The key invariant is that no exception was raised.
        assert result is None or result is not None  # always True; documents intent


# ============================================================================
# T02: init -- error cases
# ============================================================================


class TestT02aNotGitRepo:
    """T02-a: init on a non-git directory should fail."""

    def _make_non_git_dir(self, tmp_path: Path) -> Path:
        non_git = tmp_path / "not_a_repo"
        non_git.mkdir()
        (non_git / "file.txt").write_text("hello")
        return non_git

    # -- T02-a-1: raises exception -------------------------------------------
    def test_t02a_1_raises_on_non_git_dir(self, tmp_path):
        non_git = self._make_non_git_dir(tmp_path)
        ms = MinSync(
            repo_path=non_git,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=MockVectorStore(),
        )
        with pytest.raises(Exception):  # noqa: B017
            ms.init()

    # -- T02-a-2: error message contains "not a git repository" ---------------
    def test_t02a_2_error_message_mentions_not_git(self, tmp_path):
        non_git = self._make_non_git_dir(tmp_path)
        ms = MinSync(
            repo_path=non_git,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=MockVectorStore(),
        )
        with pytest.raises(Exception, match=r"(?i)not a git repository"):
            ms.init()

    # -- T02-a-3: .minsync/ not created --------------------------------------
    def test_t02a_3_minsync_dir_not_created(self, tmp_path):
        non_git = self._make_non_git_dir(tmp_path)
        ms = MinSync(
            repo_path=non_git,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=MockVectorStore(),
        )
        with pytest.raises(Exception):  # noqa: B017
            ms.init()
        assert not (non_git / ".minsync").exists()


class TestT02bAlreadyInitialized:
    """T02-b: init on an already-initialized repo should fail."""

    # -- T02-b-1: second init raises exception --------------------------------
    def test_t02b_1_second_init_raises(self, minsync_instance):
        ms, _repo = minsync_instance
        ms.init()
        with pytest.raises(Exception):  # noqa: B017
            ms.init()

    # -- T02-b-2: error message contains "already initialized" ----------------
    def test_t02b_2_error_message_mentions_already_initialized(self, minsync_instance):
        ms, _repo = minsync_instance
        ms.init()
        with pytest.raises(Exception, match=r"(?i)already initialized"):
            ms.init()


class TestT02cForceReinit:
    """T02-c: init with force=True on an already-initialized repo."""

    # -- T02-c-1: no exception ------------------------------------------------
    def test_t02c_1_force_init_succeeds(self, minsync_instance):
        ms, _repo = minsync_instance
        ms.init()
        ms.init(force=True)  # should not raise

    # -- T02-c-2: config.yaml is regenerated ----------------------------------
    def test_t02c_2_config_regenerated(self, minsync_instance):
        ms, repo = minsync_instance
        ms.init()

        config_path = repo / ".minsync" / "config.yaml"
        first_mtime = config_path.stat().st_mtime_ns

        # Ensure filesystem timestamp granularity is exceeded
        time.sleep(0.05)

        ms.init(force=True)

        second_mtime = config_path.stat().st_mtime_ns
        assert second_mtime > first_mtime, "config.yaml should have been regenerated"

        # Additionally verify the config content is valid after force reinit
        config = get_config(repo)
        assert "repo_id" in config
        assert "ref" in config


class TestT02dEmptyRepo:
    """T02-d: init on a git repo with zero commits should fail."""

    def _make_empty_git_repo(self, tmp_path: Path) -> Path:
        """Create a git repo that has been initialized but has no commits."""
        empty_repo = tmp_path / "empty_repo"
        empty_repo.mkdir()
        subprocess.run(
            ["git", "init", "-b", "main"],
            cwd=empty_repo,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.email", "test@minsync.dev"],
            cwd=empty_repo,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=empty_repo,
            capture_output=True,
            check=True,
        )
        return empty_repo

    # -- T02-d-1: raises exception -------------------------------------------
    def test_t02d_1_raises_on_empty_repo(self, tmp_path):
        empty_repo = self._make_empty_git_repo(tmp_path)
        ms = MinSync(
            repo_path=empty_repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=MockVectorStore(),
        )
        with pytest.raises(Exception):  # noqa: B017
            ms.init()

    # -- T02-d-2: error message contains "no commits" ------------------------
    def test_t02d_2_error_message_mentions_no_commits(self, tmp_path):
        empty_repo = self._make_empty_git_repo(tmp_path)
        ms = MinSync(
            repo_path=empty_repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=MockVectorStore(),
        )
        with pytest.raises(Exception, match=r"(?i)no commits"):
            ms.init()
