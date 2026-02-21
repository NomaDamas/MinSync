"""T24-T27, T33-T38: MinSync status, check, verify tests.

TDD tests for ``MinSync.status()``, ``MinSync.check()``, and ``MinSync.verify()``
Python API. The implementation does NOT exist yet; these tests define the expected
behavior.

References:
    - ai_instruction/E2E_TEST_PLAN.md  (T24-T27, T33-T38)
    - ai_instruction/CLI_SPEC.md       (sections 4, 5, 6)
"""

from __future__ import annotations

import yaml

from minsync import MinSync
from tests.conftest import (
    add_commit,
    create_test_repo,
    write_minsyncignore,
)
from tests.mock_components import (
    FailingMockEmbedder,
    FailingMockVectorStore,
    MockChunker,
    MockEmbedder,
    MockVectorStore,
)

# ============================================================================
# T24: status -- each state
# ============================================================================


class TestT24aStatusNotSynced:
    """T24-a: status returns NOT_SYNCED when init has been called but sync has not."""

    def test_state_is_not_synced(self, tmp_path):
        repo = create_test_repo(tmp_path)
        store = MockVectorStore()
        ms = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=store,
        )
        ms.init()

        status = ms.status()

        assert status.state == "NOT_SYNCED"


class TestT24bStatusUpToDate:
    """T24-b: status returns UP_TO_DATE immediately after sync."""

    def test_state_is_up_to_date(self, tmp_path):
        repo = create_test_repo(tmp_path)
        store = MockVectorStore()
        ms = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=store,
        )
        ms.init()
        ms.sync()

        status = ms.status()

        assert status.state == "UP_TO_DATE"

    def test_last_synced_commit_equals_current_head(self, tmp_path):
        repo = create_test_repo(tmp_path)
        store = MockVectorStore()
        ms = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=store,
        )
        ms.init()
        ms.sync()

        status = ms.status()

        assert status.last_synced_commit == status.current_head


class TestT24cStatusOutOfDate:
    """T24-c: status returns OUT_OF_DATE after a new commit is made post-sync."""

    def test_state_is_out_of_date(self, tmp_path):
        repo = create_test_repo(tmp_path)
        store = MockVectorStore()
        ms = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=store,
        )
        ms.init()
        ms.sync()

        # Create a new commit after sync
        add_commit(repo, {"docs/new.md": "# New Document\n\nSome content.\n"}, "add new doc")

        status = ms.status()

        assert status.state == "OUT_OF_DATE"


class TestT24dStatusStructuredData:
    """T24-d: status result has structured data with a 'state' field."""

    def test_status_has_state_field(self, tmp_path):
        repo = create_test_repo(tmp_path)
        store = MockVectorStore()
        ms = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=store,
        )
        ms.init()

        status = ms.status()

        # The status object must have a 'state' attribute
        assert hasattr(status, "state")
        assert isinstance(status.state, str)
        assert status.state in ("UP_TO_DATE", "OUT_OF_DATE", "NOT_SYNCED", "INTERRUPTED")


# ============================================================================
# T25: verify -- normal (all passed)
# ============================================================================


class TestT25VerifyNormal:
    """T25: verify reports all_passed after a clean init+sync."""

    def test_verify_all_passed(self, tmp_path):
        repo = create_test_repo(tmp_path)
        store = MockVectorStore()
        ms = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=store,
        )
        ms.init()
        ms.sync()

        report = ms.verify(all=True)

        assert report.all_passed is True


# ============================================================================
# T26: verify -- inconsistency detected
# ============================================================================


class TestT26VerifyInconsistency:
    """T26: verify detects MISSING chunks after intentional DB corruption."""

    def test_verify_detects_missing_after_direct_delete(self, tmp_path):
        repo = create_test_repo(tmp_path)
        store = MockVectorStore()
        ms = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=store,
        )
        ms.init()
        ms.sync()

        # Intentionally corrupt: delete some docs from guide.md
        docs = store.get_docs_by_path("docs/guide.md")
        assert len(docs) > 0, "guide.md should have chunks after sync"
        # Delete first 2 chunks (or all if fewer than 2)
        ids_to_delete = [d["id"] for d in docs[:2]]
        store.direct_delete(ids_to_delete)

        report = ms.verify(all=True)

        assert report.all_passed is False

    def test_verify_reports_missing_in_file_checks(self, tmp_path):
        repo = create_test_repo(tmp_path)
        store = MockVectorStore()
        ms = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=store,
        )
        ms.init()
        ms.sync()

        # Intentionally corrupt: delete some docs from guide.md
        docs = store.get_docs_by_path("docs/guide.md")
        ids_to_delete = [d["id"] for d in docs[:2]]
        store.direct_delete(ids_to_delete)

        report = ms.verify(all=True)

        # file_checks should contain information about the inconsistency
        assert report.file_checks is not None
        # There should be at least one file check that is not passing
        # (the exact structure may vary, but all_passed must be False)
        assert report.all_passed is False


# ============================================================================
# T27: verify --fix
# ============================================================================


class TestT27VerifyFix:
    """T27: verify(fix=True) repairs inconsistencies and subsequent verify passes."""

    def test_verify_fix_repairs_missing_chunks(self, tmp_path):
        repo = create_test_repo(tmp_path)
        store = MockVectorStore()
        ms = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=store,
        )
        ms.init()
        ms.sync()

        # Record chunk count before corruption
        guide_docs_before = store.get_docs_by_path("docs/guide.md")
        store.doc_count()

        # Intentionally corrupt: delete some docs from guide.md
        ids_to_delete = [d["id"] for d in guide_docs_before[:2]]
        store.direct_delete(ids_to_delete)

        # Verify with fix=True should repair the damage
        ms.verify(all=True, fix=True)

        # After fix, verify again -- should now pass
        report_after = ms.verify(all=True)
        assert report_after.all_passed is True

    def test_verify_fix_restores_deleted_chunks(self, tmp_path):
        repo = create_test_repo(tmp_path)
        store = MockVectorStore()
        ms = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=store,
        )
        ms.init()
        ms.sync()

        guide_docs_before = store.get_docs_by_path("docs/guide.md")
        count_before = len(guide_docs_before)
        assert count_before > 0

        # Delete some chunks
        ids_to_delete = [d["id"] for d in guide_docs_before[:2]]
        store.direct_delete(ids_to_delete)

        # Fix
        ms.verify(all=True, fix=True)

        # guide.md should have the same number of chunks as before corruption
        guide_docs_after = store.get_docs_by_path("docs/guide.md")
        assert len(guide_docs_after) == count_before


# ============================================================================
# T33: check -- normal health check
# ============================================================================


class TestT33CheckNormal:
    """T33: check returns healthy status with working mock components."""

    def test_git_ok(self, tmp_path):
        repo = create_test_repo(tmp_path)
        store = MockVectorStore()
        ms = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=store,
        )
        ms.init()

        health = ms.check()

        assert health.git_ok is True

    def test_embedder_ok(self, tmp_path):
        repo = create_test_repo(tmp_path)
        store = MockVectorStore()
        ms = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=store,
        )
        ms.init()

        health = ms.check()

        assert health.embedder_ok is True

    def test_vectorstore_ok(self, tmp_path):
        repo = create_test_repo(tmp_path)
        store = MockVectorStore()
        ms = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=store,
        )
        ms.init()

        health = ms.check()

        assert health.vectorstore_ok is True

    def test_no_errors(self, tmp_path):
        repo = create_test_repo(tmp_path)
        store = MockVectorStore()
        ms = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=store,
        )
        ms.init()

        health = ms.check()

        assert len(health.errors) == 0


# ============================================================================
# T34: check -- embedder failure
# ============================================================================


class TestT34CheckEmbedderFailure:
    """T34: check detects embedder failure (FailingMockEmbedder)."""

    def test_embedder_not_ok(self, tmp_path):
        repo = create_test_repo(tmp_path)
        ms = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=FailingMockEmbedder(),
            vector_store=MockVectorStore(),
        )
        ms.init()

        health = ms.check()

        assert health.embedder_ok is False

    def test_has_error_message(self, tmp_path):
        repo = create_test_repo(tmp_path)
        ms = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=FailingMockEmbedder(),
            vector_store=MockVectorStore(),
        )
        ms.init()

        health = ms.check()

        assert len(health.errors) > 0
        # At least one error should mention the embedder issue
        error_text = " ".join(health.errors).lower()
        assert "embed" in error_text or "api" in error_text or "unavailable" in error_text

    def test_git_still_ok(self, tmp_path):
        """Even when embedder fails, git check should pass."""
        repo = create_test_repo(tmp_path)
        ms = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=FailingMockEmbedder(),
            vector_store=MockVectorStore(),
        )
        ms.init()

        health = ms.check()

        assert health.git_ok is True


# ============================================================================
# T35: check -- vectorstore failure
# ============================================================================


class TestT35CheckVectorStoreFailure:
    """T35: check detects vectorstore failure (FailingMockVectorStore)."""

    def test_vectorstore_not_ok(self, tmp_path):
        repo = create_test_repo(tmp_path)
        ms = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=FailingMockVectorStore(),
        )
        ms.init()

        health = ms.check()

        assert health.vectorstore_ok is False

    def test_has_error_message(self, tmp_path):
        repo = create_test_repo(tmp_path)
        ms = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=FailingMockVectorStore(),
        )
        ms.init()

        health = ms.check()

        assert len(health.errors) > 0
        error_text = " ".join(health.errors).lower()
        assert "vector" in error_text or "connection" in error_text or "refused" in error_text

    def test_git_still_ok(self, tmp_path):
        """Even when vectorstore fails, git check should pass."""
        repo = create_test_repo(tmp_path)
        ms = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=FailingMockVectorStore(),
        )
        ms.init()

        health = ms.check()

        assert health.git_ok is True


# ============================================================================
# T36: check -- package not installed
# ============================================================================


class TestT36CheckMissingPackage:
    """T36: check detects when a configured vectorstore package is not installed.

    Approach: Create MinSync without providing a custom vector_store, then
    manipulate the config.yaml so that vectorstore.id = "weaviate". When check()
    tries to load the weaviate adapter from config, it should detect the missing
    package and report it.
    """

    def test_missing_package_detected(self, tmp_path):
        repo = create_test_repo(tmp_path)
        # First, init with mock components to create .minsync/ structure
        ms_setup = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=MockVectorStore(),
        )
        ms_setup.init()

        # Now modify config.yaml to reference "weaviate" vectorstore
        config_path = repo / ".minsync" / "config.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["vectorstore"]["id"] = "weaviate"
        config_path.write_text(yaml.dump(config), encoding="utf-8")

        # Create a new MinSync instance WITHOUT providing vector_store
        # so it attempts to load from config
        ms = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            # vector_store intentionally omitted -- should load from config
        )

        health = ms.check()

        assert health.vectorstore_ok is False
        # Error message should hint at installing the required package
        error_text = " ".join(health.errors).lower()
        assert (
            "weaviate" in error_text
            or "pip install" in error_text
            or "not found" in error_text
            or "not installed" in error_text
        )


# ============================================================================
# T37: verify -- .minsyncignore stale detection
# ============================================================================


class TestT37VerifyIgnoredStale:
    """T37: verify detects IGNORED_STALE for files matching .minsyncignore
    that still have chunks in the vector store.

    Setup:
    1. init + sync (all files indexed, including src/main.py, src/utils.py)
    2. Write .minsyncignore with "src/**" (do NOT commit, just write file)
    3. verify(all=True) should detect stale ignored files
    """

    def test_verify_detects_ignored_stale(self, tmp_path):
        repo = create_test_repo(tmp_path)
        store = MockVectorStore()
        ms = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=store,
        )
        ms.init()
        ms.sync()

        # Confirm src files were indexed
        assert len(store.get_docs_by_path("src/main.py")) > 0
        assert len(store.get_docs_by_path("src/utils.py")) > 0

        # Write .minsyncignore to exclude src/**
        write_minsyncignore(repo, "src/**\n")

        report = ms.verify(all=True)

        assert report.all_passed is False

    def test_ignored_stale_lists_src_files(self, tmp_path):
        repo = create_test_repo(tmp_path)
        store = MockVectorStore()
        ms = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=store,
        )
        ms.init()
        ms.sync()

        write_minsyncignore(repo, "src/**\n")

        report = ms.verify(all=True)

        # ignored_stale should report the stale paths
        assert report.ignored_stale is not None
        # Convert to a flat set of paths for easy checking
        # (ignored_stale may be a list of dicts, list of strings, etc.)
        if isinstance(report.ignored_stale, list):
            stale_paths = set()
            for item in report.ignored_stale:
                if isinstance(item, dict):
                    stale_paths.add(item.get("path", ""))
                elif isinstance(item, str):
                    stale_paths.add(item)
                else:
                    # Might have a .path attribute
                    stale_paths.add(getattr(item, "path", str(item)))
            assert "src/main.py" in stale_paths
            assert "src/utils.py" in stale_paths
        else:
            # If ignored_stale is something else, at minimum it should be truthy
            assert report.ignored_stale


# ============================================================================
# T38: verify --fix -- .minsyncignore stale removal
# ============================================================================


class TestT38VerifyFixIgnoredStale:
    """T38: verify(fix=True) removes stale chunks from .minsyncignore-matched files.

    Same setup as T37 but with fix=True. After fix:
    - src/main.py chunks == 0
    - src/utils.py chunks == 0
    - Other files unchanged
    - Subsequent verify passes
    """

    def test_fix_removes_src_main_chunks(self, tmp_path):
        repo = create_test_repo(tmp_path)
        store = MockVectorStore()
        ms = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=store,
        )
        ms.init()
        ms.sync()

        write_minsyncignore(repo, "src/**\n")

        ms.verify(all=True, fix=True)

        assert len(store.get_docs_by_path("src/main.py")) == 0

    def test_fix_removes_src_utils_chunks(self, tmp_path):
        repo = create_test_repo(tmp_path)
        store = MockVectorStore()
        ms = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=store,
        )
        ms.init()
        ms.sync()

        write_minsyncignore(repo, "src/**\n")

        ms.verify(all=True, fix=True)

        assert len(store.get_docs_by_path("src/utils.py")) == 0

    def test_fix_preserves_other_files(self, tmp_path):
        repo = create_test_repo(tmp_path)
        store = MockVectorStore()
        ms = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=store,
        )
        ms.init()
        ms.sync()

        # Record doc counts for non-src files before fix
        guide_count = len(store.get_docs_by_path("docs/guide.md"))
        api_count = len(store.get_docs_by_path("docs/api.md"))
        readme_count = len(store.get_docs_by_path("README.md"))
        assert guide_count > 0, "guide.md should have chunks"

        write_minsyncignore(repo, "src/**\n")

        ms.verify(all=True, fix=True)

        # Non-src files should be unchanged
        assert len(store.get_docs_by_path("docs/guide.md")) == guide_count
        assert len(store.get_docs_by_path("docs/api.md")) == api_count
        assert len(store.get_docs_by_path("README.md")) == readme_count

    def test_verify_passes_after_fix(self, tmp_path):
        repo = create_test_repo(tmp_path)
        store = MockVectorStore()
        ms = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=store,
        )
        ms.init()
        ms.sync()

        write_minsyncignore(repo, "src/**\n")

        # Fix the stale ignored files
        ms.verify(all=True, fix=True)

        # Subsequent verify should pass
        report = ms.verify(all=True)
        assert report.all_passed is True
