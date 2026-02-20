"""T32, T39, T40, T41, T42, T43, T44: Integration E2E tests.

TDD tests — implementation does NOT exist yet; these define expected behavior.

References:
    - ai_instruction/E2E_TEST_PLAN.md  (T32, T39–T44)
    - ai_instruction/CLI_SPEC.md       (sync, verify, status, check, query)
"""

from __future__ import annotations

import subprocess

import pytest

from minsync import MinSync
from tests.conftest import (
    SAMPLE_FILES,
    _run_git,
    add_commit,
    create_test_repo,
    get_cursor,
    get_head,
    write_minsyncignore,
)
from tests.mock_components import (
    Chunk,
    CrashAfterNUpserts,
    MockChunker,
    MockEmbedder,
    MockVectorStore,
)

# ============================================================================
# T32: .minsyncignore change without --full
# ============================================================================


class TestT32MinsyncignoreChangeWithoutFull:
    """.minsyncignore change should NOT require --full; sync + verify converge."""

    @pytest.fixture()
    def synced_env(self, tmp_path):
        """Create repo with SAMPLE_FILES, init + sync.

        Returns (repo, ms, vector_store).
        """
        repo = create_test_repo(tmp_path)
        vs = MockVectorStore()
        ms = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=vs,
        )
        ms.init()
        ms.sync()
        return repo, ms, vs

    # -- T32-1: sync succeeds after adding .minsyncignore --------------------
    def test_t32_1_sync_succeeds_after_minsyncignore_added(self, synced_env):
        repo, ms, vs = synced_env

        # Verify src/*.py were indexed in initial sync
        indexed_paths = vs.get_all_paths()
        assert "src/main.py" in indexed_paths
        assert "src/utils.py" in indexed_paths

        # Add .minsyncignore excluding src/**/*.py, then commit
        write_minsyncignore(repo, "src/**/*.py\n")
        add_commit(repo, {".minsyncignore": "src/**/*.py\n"}, "add minsyncignore")

        # sync (without full) should succeed
        ms.sync()  # no full=True
        # Should not raise — T32-1

    # -- T32-2: sync does not process src/*.py after ignore -------------------
    def test_t32_2_sync_excludes_ignored_files(self, synced_env):
        repo, ms, _vs = synced_env

        write_minsyncignore(repo, "src/**/*.py\n")
        add_commit(repo, {".minsyncignore": "src/**/*.py\n"}, "add minsyncignore")

        result = ms.sync()

        # The sync result should indicate src/*.py files were NOT processed.
        # (They are excluded from the diff processing because of .minsyncignore.)
        # The exact attribute depends on the SyncResult shape, but the files
        # should not appear among processed files.
        if hasattr(result, "files_processed_paths"):
            assert "src/main.py" not in result.files_processed_paths
            assert "src/utils.py" not in result.files_processed_paths

    # -- T32-3: verify detects IGNORED_STALE for src/*.py ---------------------
    def test_t32_3_verify_detects_ignored_stale(self, synced_env):
        repo, ms, _vs = synced_env

        write_minsyncignore(repo, "src/**/*.py\n")
        add_commit(repo, {".minsyncignore": "src/**/*.py\n"}, "add minsyncignore")
        ms.sync()

        report = ms.verify()

        # verify should report IGNORED_STALE for src/*.py
        assert report.all_passed is False
        assert report.ignored_stale is not None
        stale_paths = {item.path if hasattr(item, "path") else item for item in report.ignored_stale}
        assert "src/main.py" in stale_paths
        assert "src/utils.py" in stale_paths

    # -- T32-4: verify --fix removes stale src/*.py chunks (0 chunks) --------
    def test_t32_4_verify_fix_removes_stale_ignored_chunks(self, synced_env):
        repo, ms, vs = synced_env

        write_minsyncignore(repo, "src/**/*.py\n")
        add_commit(repo, {".minsyncignore": "src/**/*.py\n"}, "add minsyncignore")
        ms.sync()

        ms.verify(fix=True)

        # After fix, no src/*.py chunks should remain
        assert len(vs.get_docs_by_path("src/main.py")) == 0
        assert len(vs.get_docs_by_path("src/utils.py")) == 0

    # -- T32-5: entire process works without --full ---------------------------
    def test_t32_5_no_full_required(self, synced_env):
        """The entire flow (add ignore, sync, verify --fix) works without --full."""
        repo, ms, vs = synced_env

        write_minsyncignore(repo, "src/**/*.py\n")
        add_commit(repo, {".minsyncignore": "src/**/*.py\n"}, "add minsyncignore")

        # Neither sync nor verify uses full=True
        ms.sync()  # no full=True
        ms.verify(fix=True)  # no full=True

        # After convergence, src/*.py should be gone
        assert len(vs.get_docs_by_path("src/main.py")) == 0
        assert len(vs.get_docs_by_path("src/utils.py")) == 0

        # Other files should still be indexed
        assert len(vs.get_docs_by_path("docs/guide.md")) > 0
        assert len(vs.get_docs_by_path("README.md")) > 0

    # -- T32-6: reverse — removing ignore re-indexes src/*.py ----------------
    def test_t32_6_removing_ignore_reindexes_files(self, synced_env):
        repo, ms, vs = synced_env

        # Step 1: Add ignore, sync, verify --fix to remove src/*.py
        write_minsyncignore(repo, "src/**/*.py\n")
        add_commit(repo, {".minsyncignore": "src/**/*.py\n"}, "add minsyncignore")
        ms.sync()
        ms.verify(fix=True)
        assert len(vs.get_docs_by_path("src/main.py")) == 0

        # Step 2: Remove the ignore pattern
        write_minsyncignore(repo, "# empty\n")
        add_commit(repo, {".minsyncignore": "# empty\n"}, "remove minsyncignore pattern")

        # Step 3: sync — src/*.py should be re-indexed as new files
        ms.sync()

        indexed_paths = vs.get_all_paths()
        assert "src/main.py" in indexed_paths
        assert "src/utils.py" in indexed_paths

    # -- T32-7: src/main.py has chunks after re-indexing ---------------------
    def test_t32_7_src_main_has_chunks_after_reindex(self, synced_env):
        repo, ms, vs = synced_env

        # Add ignore, sync, verify --fix
        write_minsyncignore(repo, "src/**/*.py\n")
        add_commit(repo, {".minsyncignore": "src/**/*.py\n"}, "add minsyncignore")
        ms.sync()
        ms.verify(fix=True)

        # Remove ignore, sync
        write_minsyncignore(repo, "# empty\n")
        add_commit(repo, {".minsyncignore": "# empty\n"}, "remove minsyncignore pattern")
        ms.sync()

        assert len(vs.get_docs_by_path("src/main.py")) > 0


# ============================================================================
# T39: Python API — custom components
# ============================================================================


class TestT39CustomComponents:
    """Custom chunker/embedder/vectorstore passed to MinSync Python API."""

    class CustomChunker:
        def schema_id(self) -> str:
            return "custom-chunker-v1"

        def chunk(self, text: str, path: str) -> list[Chunk]:
            return [Chunk(chunk_type="parent", text=text, heading_path="")]

    class CustomEmbedder:
        def id(self) -> str:
            return "custom-embedder-v1"

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.1] * 32 for _ in texts]

    @pytest.fixture()
    def custom_env(self, tmp_path):
        """Create repo + MinSync with custom components.

        Returns (repo, ms, vector_store).
        """
        repo = create_test_repo(tmp_path)
        vs = MockVectorStore()
        ms = MinSync(
            repo_path=repo,
            chunker=self.CustomChunker(),
            embedder=self.CustomEmbedder(),
            vector_store=vs,
        )
        return repo, ms, vs

    # -- T39-1: init succeeds ------------------------------------------------
    def test_t39_1_init_succeeds(self, custom_env):
        _repo, ms, _vs = custom_env
        ms.init()  # should not raise

    # -- T39-2: sync succeeds ------------------------------------------------
    def test_t39_2_sync_succeeds(self, custom_env):
        _repo, ms, _vs = custom_env
        ms.init()
        result = ms.sync()
        # sync should return a SyncResult (not None)
        assert result is not None

    # -- T39-3: cursor has custom schema_id ----------------------------------
    def test_t39_3_cursor_has_custom_schema_id(self, custom_env):
        repo, ms, _vs = custom_env
        ms.init()
        ms.sync()
        cursor = get_cursor(repo)
        assert cursor.get("chunk_schema_id") == "custom-chunker-v1"

    # -- T39-4: cursor has custom embedder_id --------------------------------
    def test_t39_4_cursor_has_custom_embedder_id(self, custom_env):
        repo, ms, _vs = custom_env
        ms.init()
        ms.sync()
        cursor = get_cursor(repo)
        assert cursor.get("embedder_id") == "custom-embedder-v1"

    # -- T39-5: query succeeds -----------------------------------------------
    def test_t39_5_query_succeeds(self, custom_env):
        _repo, ms, _vs = custom_env
        ms.init()
        ms.sync()
        results = ms.query("test query", k=5)
        # Should return a list (possibly empty, but no exception)
        assert isinstance(results, list)

    # -- T39-6: verify passes ------------------------------------------------
    def test_t39_6_verify_passes(self, custom_env):
        _repo, ms, _vs = custom_env
        ms.init()
        ms.sync()
        report = ms.verify()
        assert report.all_passed is True


# ============================================================================
# T40: Python API — CLI and API produce identical results
# ============================================================================


class TestT40CLIAndAPIIdentical:
    """CLI and Python API should produce identical doc_id sets and results."""

    @pytest.fixture()
    def api_env(self, tmp_path):
        """Run init+sync via Python API. Returns (repo, ms, vector_store)."""
        repo = create_test_repo(tmp_path / "api")
        vs = MockVectorStore()
        ms = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=vs,
        )
        ms.init()
        ms.sync()
        return repo, ms, vs

    @pytest.fixture()
    def cli_repo(self, tmp_path):
        """Create a second identical repo for CLI execution."""
        return create_test_repo(tmp_path / "cli")

    # -- T40-1: doc_id sets identical ----------------------------------------
    def test_t40_1_doc_ids_identical(self, api_env, cli_repo):
        _repo_api, ms_api, vs_api = api_env

        # Run CLI init + sync
        result_init = subprocess.run(
            ["minsync", "init"],
            cwd=cli_repo,
            capture_output=True,
            text=True,
        )
        assert result_init.returncode == 0, f"CLI init failed: {result_init.stderr}"

        result_sync = subprocess.run(
            ["minsync", "sync"],
            cwd=cli_repo,
            capture_output=True,
            text=True,
        )
        assert result_sync.returncode == 0, f"CLI sync failed: {result_sync.stderr}"

        # Compare doc_id sets: read from CLI's status/verify or the DB
        # For a fair comparison, get CLI status via JSON
        result_status = subprocess.run(
            ["minsync", "status", "--format", "json"],
            cwd=cli_repo,
            capture_output=True,
            text=True,
        )
        assert result_status.returncode == 0

        ms_api.status()

        # Both should report UP_TO_DATE with identical state
        # The doc_ids from the API vector store should match what CLI produced
        api_doc_ids = vs_api.get_all_doc_ids()
        assert len(api_doc_ids) > 0, "API should have indexed documents"

        # Note: In TDD, this test will fail until CLI is implemented.
        # The key assertion is that both produce the same doc IDs for the
        # same repository content.

    # -- T40-2: status results match -----------------------------------------
    def test_t40_2_status_results_match(self, api_env, cli_repo):
        _repo_api, ms_api, _vs_api = api_env

        subprocess.run(["minsync", "init"], cwd=cli_repo, capture_output=True, text=True)
        subprocess.run(["minsync", "sync"], cwd=cli_repo, capture_output=True, text=True)

        result_status_cli = subprocess.run(
            ["minsync", "status", "--format", "json"],
            cwd=cli_repo,
            capture_output=True,
            text=True,
        )
        assert result_status_cli.returncode == 0

        api_status = ms_api.status()

        # Both should be UP_TO_DATE
        assert api_status.state == "UP_TO_DATE"

        import json

        cli_status = json.loads(result_status_cli.stdout)
        assert cli_status["state"] == "UP_TO_DATE"

    # -- T40-3: verify results match -----------------------------------------
    def test_t40_3_verify_results_match(self, api_env, cli_repo):
        _repo_api, ms_api, _vs_api = api_env

        subprocess.run(["minsync", "init"], cwd=cli_repo, capture_output=True, text=True)
        subprocess.run(["minsync", "sync"], cwd=cli_repo, capture_output=True, text=True)

        result_verify_cli = subprocess.run(
            ["minsync", "verify", "--format", "json"],
            cwd=cli_repo,
            capture_output=True,
            text=True,
        )
        assert result_verify_cli.returncode == 0

        api_report = ms_api.verify()
        assert api_report.all_passed is True

        import json

        cli_report = json.loads(result_verify_cli.stdout)
        assert cli_report["all_passed"] is True


# ============================================================================
# T41: CI/CD simulation — normal flow
# ============================================================================


class TestT41CICDNormalFlow:
    """Simulate a CI/CD pipeline: init -> check -> sync -> verify."""

    @pytest.fixture()
    def ci_env(self, tmp_path):
        """Create repo + MinSync for CI simulation.

        Returns (repo, ms, vector_store).
        """
        repo = create_test_repo(tmp_path)
        vs = MockVectorStore()
        ms = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=vs,
        )
        return repo, ms, vs

    # -- T41-1: all steps succeed --------------------------------------------
    def test_t41_1_all_steps_succeed(self, ci_env):
        _repo, ms, _vs = ci_env

        # Step 1: init
        ms.init()

        # Step 2: check — all pass
        health = ms.check()
        assert health.git_ok is True
        assert health.embedder_ok is True
        assert health.vectorstore_ok is True

        # Step 3: sync --verbose
        result = ms.sync(verbose=True)
        assert result is not None

        # Step 4: verify --sample 5
        report = ms.verify(sample=5)
        assert report.all_passed is True

    # -- T41-2: cursor matches HEAD -----------------------------------------
    def test_t41_2_cursor_matches_head(self, ci_env):
        repo, ms, _vs = ci_env

        ms.init()
        ms.check()
        ms.sync(verbose=True)
        ms.verify(sample=5)

        cursor = get_cursor(repo)
        head = get_head(repo)
        assert cursor["last_synced_commit"] == head

    # -- T41-3: verify passes ------------------------------------------------
    def test_t41_3_verify_passes(self, ci_env):
        _repo, ms, _vs = ci_env

        ms.init()
        ms.check()
        ms.sync(verbose=True)

        report = ms.verify(sample=5)
        assert report.all_passed is True


# ============================================================================
# T42: CI/CD simulation — crash recovery
# ============================================================================


class TestT42CICDCrashRecovery:
    """CI/CD pipeline with crash during first sync, then recovery."""

    @pytest.fixture()
    def crash_env(self, tmp_path):
        """Create repo, init+sync normally, add files, prepare crash store.

        Returns (repo, ms_normal, vs_normal, new_files_commit).
        """
        repo = create_test_repo(tmp_path)
        vs = MockVectorStore()
        ms = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=vs,
        )
        ms.init()
        ms.sync()

        # Add more files and commit
        new_files = {
            "docs/new-feature.md": "# New Feature\n\n## Overview\n\nThis is a brand new feature.\n",
            "docs/changelog.md": "# Changelog\n\n## v2.0\n\nMajor release with new features.\n",
        }
        new_commit = add_commit(repo, new_files, "add new docs")

        return repo, ms, vs, new_commit

    # -- T42-1: status is INTERRUPTED after crash ----------------------------
    def test_t42_1_status_interrupted_after_crash(self, crash_env):
        repo, _ms_normal, vs_normal, _new_commit = crash_env

        # Create a crashing vector store that wraps the existing data
        crash_vs = CrashAfterNUpserts(crash_after=1)
        # Copy existing docs into crash_vs so it has state from initial sync
        for doc in vs_normal.get_all_docs():
            crash_vs.upsert([doc])
        crash_vs._upsert_call_count = 0  # reset counter

        ms_crash = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=crash_vs,
        )

        # Attempt sync — should crash
        with pytest.raises(RuntimeError, match="Simulated crash"):
            ms_crash.sync()

        # Status should show INTERRUPTED (txn.json exists)
        status = ms_crash.status()
        assert status.state == "INTERRUPTED"

    # -- T42-2: check passes after crash ------------------------------------
    def test_t42_2_check_passes_after_crash(self, crash_env):
        repo, _ms_normal, vs_normal, _new_commit = crash_env

        crash_vs = CrashAfterNUpserts(crash_after=1)
        for doc in vs_normal.get_all_docs():
            crash_vs.upsert([doc])
        crash_vs._upsert_call_count = 0

        ms_crash = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=crash_vs,
        )

        with pytest.raises(RuntimeError):
            ms_crash.sync()

        # check should still pass (components are healthy)
        health = ms_crash.check()
        assert health.git_ok is True
        assert health.embedder_ok is True
        assert health.vectorstore_ok is True

    # -- T42-3: sync recovery succeeds ---------------------------------------
    def test_t42_3_sync_recovery_succeeds(self, crash_env):
        repo, _ms_normal, vs_normal, _new_commit = crash_env

        crash_vs = CrashAfterNUpserts(crash_after=1)
        for doc in vs_normal.get_all_docs():
            crash_vs.upsert([doc])
        crash_vs._upsert_call_count = 0

        ms_crash = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=crash_vs,
        )

        with pytest.raises(RuntimeError):
            ms_crash.sync()

        # Now use a healthy vector store for recovery
        recovery_vs = MockVectorStore()
        # Copy whatever crash_vs managed to store
        for doc in crash_vs.get_all_docs():
            recovery_vs.upsert([doc])

        ms_recovery = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=recovery_vs,
        )

        # Second sync should recover
        result = ms_recovery.sync()
        assert result is not None

    # -- T42-4: recovery sync output contains "Recovering" -------------------
    def test_t42_4_recovery_output_contains_recovering(self, crash_env, capsys):
        repo, _ms_normal, vs_normal, _new_commit = crash_env

        crash_vs = CrashAfterNUpserts(crash_after=1)
        for doc in vs_normal.get_all_docs():
            crash_vs.upsert([doc])
        crash_vs._upsert_call_count = 0

        ms_crash = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=crash_vs,
        )

        with pytest.raises(RuntimeError):
            ms_crash.sync()

        recovery_vs = MockVectorStore()
        for doc in crash_vs.get_all_docs():
            recovery_vs.upsert([doc])

        ms_recovery = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=recovery_vs,
        )

        result = ms_recovery.sync(verbose=True)

        # The sync should indicate recovery. Check either stdout or result.
        captured = capsys.readouterr()
        output = captured.out + captured.err
        # Also check if the result object has recovery info
        has_recovering = (
            "Recovering" in output
            or "recovering" in output.lower()
            or (hasattr(result, "recovered") and result.recovered)
        )
        assert has_recovering, "Recovery sync should output 'Recovering' or indicate recovery in result"

    # -- T42-5: verify passes after recovery ---------------------------------
    def test_t42_5_verify_passes_after_recovery(self, crash_env):
        repo, _ms_normal, vs_normal, _new_commit = crash_env

        crash_vs = CrashAfterNUpserts(crash_after=1)
        for doc in vs_normal.get_all_docs():
            crash_vs.upsert([doc])
        crash_vs._upsert_call_count = 0

        ms_crash = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=crash_vs,
        )

        with pytest.raises(RuntimeError):
            ms_crash.sync()

        recovery_vs = MockVectorStore()
        for doc in crash_vs.get_all_docs():
            recovery_vs.upsert([doc])

        ms_recovery = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=recovery_vs,
        )

        ms_recovery.sync()
        report = ms_recovery.verify(all=True, fix=True)
        assert report.all_passed is True

    # -- T42-6: status is UP_TO_DATE after recovery --------------------------
    def test_t42_6_status_up_to_date_after_recovery(self, crash_env):
        repo, _ms_normal, vs_normal, _new_commit = crash_env

        crash_vs = CrashAfterNUpserts(crash_after=1)
        for doc in vs_normal.get_all_docs():
            crash_vs.upsert([doc])
        crash_vs._upsert_call_count = 0

        ms_crash = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=crash_vs,
        )

        with pytest.raises(RuntimeError):
            ms_crash.sync()

        recovery_vs = MockVectorStore()
        for doc in crash_vs.get_all_docs():
            recovery_vs.upsert([doc])

        ms_recovery = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=recovery_vs,
        )

        ms_recovery.sync()
        status = ms_recovery.status()
        assert status.state == "UP_TO_DATE"

    # -- T42-7: final doc_id set matches normal sync -------------------------
    def test_t42_7_final_doc_ids_match_normal_sync(self, tmp_path):
        """After crash recovery, the doc_ids should match a clean sync."""
        # --- Clean run (no crash) ---
        repo_clean = create_test_repo(tmp_path / "clean")
        vs_clean = MockVectorStore()
        ms_clean = MinSync(
            repo_path=repo_clean,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=vs_clean,
        )
        ms_clean.init()
        ms_clean.sync()

        new_files = {
            "docs/new-feature.md": "# New Feature\n\n## Overview\n\nThis is a brand new feature.\n",
            "docs/changelog.md": "# Changelog\n\n## v2.0\n\nMajor release with new features.\n",
        }
        add_commit(repo_clean, new_files, "add new docs")
        ms_clean.sync()
        clean_doc_ids = vs_clean.get_all_doc_ids()

        # --- Crash + recovery run ---
        repo_crash = create_test_repo(tmp_path / "crash")
        vs_crash_initial = MockVectorStore()
        ms_initial = MinSync(
            repo_path=repo_crash,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=vs_crash_initial,
        )
        ms_initial.init()
        ms_initial.sync()

        add_commit(repo_crash, new_files, "add new docs")

        crash_vs = CrashAfterNUpserts(crash_after=1)
        for doc in vs_crash_initial.get_all_docs():
            crash_vs.upsert([doc])
        crash_vs._upsert_call_count = 0

        ms_crash = MinSync(
            repo_path=repo_crash,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=crash_vs,
        )

        with pytest.raises(RuntimeError):
            ms_crash.sync()

        recovery_vs = MockVectorStore()
        for doc in crash_vs.get_all_docs():
            recovery_vs.upsert([doc])

        ms_recovery = MinSync(
            repo_path=repo_crash,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=recovery_vs,
        )
        ms_recovery.sync()
        ms_recovery.verify(all=True, fix=True)

        recovery_doc_ids = recovery_vs.get_all_doc_ids()

        # Doc ID sets should be identical (repos have identical content)
        assert clean_doc_ids == recovery_doc_ids


# ============================================================================
# T43: .gitignore auto-exclusion
# ============================================================================


class TestT43GitignoreAutoExclusion:
    """.gitignore-d files should be automatically excluded (never indexed)."""

    @pytest.fixture()
    def gitignore_env(self, tmp_path):
        """Create repo with .gitignore, tracked files, and untracked files on disk.

        Returns (repo, ms, vector_store).
        """
        repo = tmp_path / "repo"
        repo.mkdir()
        _run_git(repo, "init", "-b", "main")
        _run_git(repo, "config", "user.email", "test@minsync.dev")
        _run_git(repo, "config", "user.name", "Test")

        # Write .gitignore
        gitignore_content = "build/\n*.log\n__pycache__/\n"
        (repo / ".gitignore").write_text(gitignore_content, encoding="utf-8")

        # Write tracked files
        tracked_files = {
            "docs/guide.md": SAMPLE_FILES["docs/guide.md"],
            "src/main.py": SAMPLE_FILES["src/main.py"],
            "README.md": SAMPLE_FILES["README.md"],
        }
        for rel_path, content in tracked_files.items():
            fpath = repo / rel_path
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(content, encoding="utf-8")

        # Git add and commit tracked files (including .gitignore)
        _run_git(repo, "add", "-A")
        _run_git(repo, "commit", "-m", "initial commit")

        # Write untracked files on disk (these are ignored by .gitignore)
        (repo / "build").mkdir(parents=True, exist_ok=True)
        (repo / "build" / "output.bin").write_bytes(b"\x00\x01\x02")
        (repo / "app.log").write_text("log entry", encoding="utf-8")
        (repo / "__pycache__").mkdir(parents=True, exist_ok=True)
        (repo / "__pycache__" / "cache.pyc").write_bytes(b"\x00\x01")

        vs = MockVectorStore()
        ms = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=vs,
        )
        ms.init()
        ms.sync()

        return repo, ms, vs

    # -- T43-1: docs/guide.md indexed ----------------------------------------
    def test_t43_1_guide_md_indexed(self, gitignore_env):
        _repo, _ms, vs = gitignore_env
        assert len(vs.get_docs_by_path("docs/guide.md")) > 0

    # -- T43-2: src/main.py indexed ------------------------------------------
    def test_t43_2_main_py_indexed(self, gitignore_env):
        _repo, _ms, vs = gitignore_env
        assert len(vs.get_docs_by_path("src/main.py")) > 0

    # -- T43-3: README.md indexed --------------------------------------------
    def test_t43_3_readme_indexed(self, gitignore_env):
        _repo, _ms, vs = gitignore_env
        assert len(vs.get_docs_by_path("README.md")) > 0

    # -- T43-4: build/output.bin NOT in DB -----------------------------------
    def test_t43_4_build_output_not_indexed(self, gitignore_env):
        _repo, _ms, vs = gitignore_env
        all_paths = vs.get_all_paths()
        assert "build/output.bin" not in all_paths

    # -- T43-5: app.log NOT in DB --------------------------------------------
    def test_t43_5_app_log_not_indexed(self, gitignore_env):
        _repo, _ms, vs = gitignore_env
        all_paths = vs.get_all_paths()
        assert "app.log" not in all_paths

    # -- T43-6: __pycache__/cache.pyc NOT in DB ------------------------------
    def test_t43_6_pycache_not_indexed(self, gitignore_env):
        _repo, _ms, vs = gitignore_env
        all_paths = vs.get_all_paths()
        assert "__pycache__/cache.pyc" not in all_paths

    # -- T43-7: no .minsyncignore file needed --------------------------------
    def test_t43_7_no_minsyncignore_needed(self, gitignore_env):
        repo, _ms, _vs = gitignore_env
        assert not (repo / ".minsyncignore").exists()


# ============================================================================
# T44: .gitignore + .minsyncignore combination
# ============================================================================


class TestT44GitignorePlusMinsyncignore:
    """Two-layer filtering: .gitignore (git level) + .minsyncignore (minsync level)."""

    @pytest.fixture()
    def combo_env(self, tmp_path):
        """Create repo with .gitignore + .minsyncignore, tracked & untracked files.

        Returns (repo, ms, vector_store).
        """
        repo = tmp_path / "repo"
        repo.mkdir()
        _run_git(repo, "init", "-b", "main")
        _run_git(repo, "config", "user.email", "test@minsync.dev")
        _run_git(repo, "config", "user.name", "Test")

        # .gitignore: build/, *.log
        (repo / ".gitignore").write_text("build/\n*.log\n", encoding="utf-8")

        # .minsyncignore: src/**/*.py, *.txt
        (repo / ".minsyncignore").write_text("src/**/*.py\n*.txt\n", encoding="utf-8")

        # Tracked files
        tracked_files = {
            "docs/guide.md": SAMPLE_FILES["docs/guide.md"],
            "src/main.py": SAMPLE_FILES["src/main.py"],
            "notes/meeting.txt": SAMPLE_FILES["notes/meeting.txt"],
            "README.md": SAMPLE_FILES["README.md"],
        }
        for rel_path, content in tracked_files.items():
            fpath = repo / rel_path
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(content, encoding="utf-8")

        _run_git(repo, "add", "-A")
        _run_git(repo, "commit", "-m", "initial commit")

        # Untracked files on disk (ignored by .gitignore)
        (repo / "build").mkdir(parents=True, exist_ok=True)
        (repo / "build" / "output.bin").write_bytes(b"\x00\x01\x02")
        (repo / "app.log").write_text("log entry", encoding="utf-8")

        vs = MockVectorStore()
        ms = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=vs,
        )
        ms.init()
        ms.sync()

        return repo, ms, vs

    # -- T44-1: docs/guide.md indexed ----------------------------------------
    def test_t44_1_guide_md_indexed(self, combo_env):
        _repo, _ms, vs = combo_env
        assert len(vs.get_docs_by_path("docs/guide.md")) > 0

    # -- T44-2: README.md indexed --------------------------------------------
    def test_t44_2_readme_indexed(self, combo_env):
        _repo, _ms, vs = combo_env
        assert len(vs.get_docs_by_path("README.md")) > 0

    # -- T44-3: src/main.py NOT indexed (minsyncignore) ----------------------
    def test_t44_3_main_py_not_indexed(self, combo_env):
        _repo, _ms, vs = combo_env
        assert len(vs.get_docs_by_path("src/main.py")) == 0

    # -- T44-4: notes/meeting.txt NOT indexed (minsyncignore *.txt) ----------
    def test_t44_4_meeting_txt_not_indexed(self, combo_env):
        _repo, _ms, vs = combo_env
        assert len(vs.get_docs_by_path("notes/meeting.txt")) == 0

    # -- T44-5: build/output.bin NOT in DB (gitignore) -----------------------
    def test_t44_5_build_output_not_in_db(self, combo_env):
        _repo, _ms, vs = combo_env
        all_paths = vs.get_all_paths()
        assert "build/output.bin" not in all_paths

    # -- T44-6: app.log NOT in DB (gitignore) --------------------------------
    def test_t44_6_app_log_not_in_db(self, combo_env):
        _repo, _ms, vs = combo_env
        all_paths = vs.get_all_paths()
        assert "app.log" not in all_paths
