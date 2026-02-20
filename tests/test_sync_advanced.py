"""E2E tests for MinSync advanced sync features.

Tests covered:
    T10: sync --dry-run
    T11: sync --full (full rebuild)
    T12: deterministic ID -- different locations
    T13: deterministic ID -- rebuild identical
    T28: mark+sweep convergence
    T29: large incremental performance
    T30: .minsyncignore basic filtering
    T31: schema/embedder mismatch detection
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from tests.conftest import (
    SAMPLE_FILES,
    add_commit,
    create_test_repo,
    get_cursor,
    get_head,
    get_root_commit,
)
from tests.mock_components import MockChunker, MockEmbedder, MockVectorStore

# ============================================================================
# T10: sync --dry-run
# ============================================================================


class TestT10DryRun:
    """T10: After init+sync, add/modify files, commit, then sync(dry_run=True)."""

    def _setup(self, tmp_path: Path):
        from minsync import MinSync

        store = MockVectorStore()
        embedder = MockEmbedder()
        chunker = MockChunker()
        repo = create_test_repo(tmp_path, SAMPLE_FILES)
        ms = MinSync(repo_path=repo, chunker=chunker, embedder=embedder, vector_store=store)
        ms.init()
        ms.sync()

        # Record state before dry-run
        cursor_before = get_cursor(repo)
        doc_count_before = store.doc_count()

        # Add a new file and modify an existing one
        add_commit(
            repo,
            {
                "docs/new.md": "# New Document\n\nThis is a new document.\n",
                "docs/guide.md": SAMPLE_FILES["docs/guide.md"] + "\n## Troubleshooting\n\nCheck the logs.\n",
            },
            "add new.md and modify guide.md",
        )

        return ms, repo, store, cursor_before, doc_count_before

    def test_t10_1_no_exception(self, tmp_path: Path):
        """T10-1: dry_run sync completes without exception."""
        ms, _repo, _store, _cursor_before, _doc_count_before = self._setup(tmp_path)
        ms.sync(dry_run=True)  # Should not raise

    def test_t10_2_cursor_unchanged(self, tmp_path: Path):
        """T10-2: cursor.json is unchanged after dry-run."""
        ms, repo, _store, cursor_before, _doc_count_before = self._setup(tmp_path)
        ms.sync(dry_run=True)
        cursor_after = get_cursor(repo)
        assert cursor_after == cursor_before

    def test_t10_3_no_txn_json(self, tmp_path: Path):
        """T10-3: txn.json is not created after dry-run."""
        ms, repo, _store, _cursor_before, _doc_count_before = self._setup(tmp_path)
        ms.sync(dry_run=True)
        txn_path = repo / ".minsync" / "txn.json"
        assert not txn_path.exists()

    def test_t10_4_doc_count_unchanged(self, tmp_path: Path):
        """T10-4: document count in vector store is unchanged after dry-run."""
        ms, _repo, store, _cursor_before, doc_count_before = self._setup(tmp_path)
        ms.sync(dry_run=True)
        assert store.doc_count() == doc_count_before

    def test_t10_5_result_indicates_dry_run(self, tmp_path: Path):
        """T10-5: SyncResult indicates dry_run mode."""
        ms, _repo, _store, _cursor_before, _doc_count_before = self._setup(tmp_path)
        result = ms.sync(dry_run=True)
        assert result.dry_run is True

    def test_t10_6_result_includes_planned_files(self, tmp_path: Path):
        """T10-6: SyncResult includes the planned files (new.md, guide.md)."""
        ms, _repo, _store, _cursor_before, _doc_count_before = self._setup(tmp_path)
        result = ms.sync(dry_run=True)
        # The result should have a list of planned files or files_planned count
        # Check that both new.md and guide.md are mentioned
        planned_paths = set()
        if hasattr(result, "planned_files"):
            planned_paths = {str(p) for p in result.planned_files}
        elif hasattr(result, "files_planned"):
            # files_planned is a count; at least 2 files should be planned
            assert result.files_planned >= 2
            planned_paths = None  # skip path check
        else:
            # Fall back: check any iterable attribute that lists files
            for attr in ("files", "changed_files", "plan"):
                if hasattr(result, attr):
                    val = getattr(result, attr)
                    if hasattr(val, "__iter__"):
                        planned_paths = {str(p) for p in val}
                    break

        if planned_paths is not None:
            assert any("new.md" in p for p in planned_paths), f"docs/new.md not found in planned files: {planned_paths}"
            assert any("guide.md" in p for p in planned_paths), (
                f"docs/guide.md not found in planned files: {planned_paths}"
            )

    def test_t10_7_real_sync_after_dry_run(self, tmp_path: Path):
        """T10-7: Real sync() still works correctly after dry-run."""
        ms, repo, store, _cursor_before, _doc_count_before = self._setup(tmp_path)
        ms.sync(dry_run=True)

        # Real sync should succeed
        ms.sync()
        cursor_after = get_cursor(repo)
        assert cursor_after["last_synced_commit"] == get_head(repo)
        # new.md should now be indexed
        new_docs = store.get_docs_by_path("docs/new.md")
        assert len(new_docs) > 0
        # guide.md should have updated chunks
        guide_docs = store.get_docs_by_path("docs/guide.md")
        assert len(guide_docs) > 0
        guide_texts = " ".join(d["text"] for d in guide_docs)
        assert "Troubleshooting" in guide_texts


# ============================================================================
# T11: sync --full (full rebuild)
# ============================================================================


class TestT11FullRebuild:
    """T11: After init+sync, run sync(full=True) and verify results."""

    def _setup(self, tmp_path: Path):
        from minsync import MinSync

        store = MockVectorStore()
        embedder = MockEmbedder()
        chunker = MockChunker()
        repo = create_test_repo(tmp_path, SAMPLE_FILES)
        ms = MinSync(repo_path=repo, chunker=chunker, embedder=embedder, vector_store=store)
        ms.init()
        ms.sync()
        return ms, repo, store

    def test_t11_1_no_exception(self, tmp_path: Path):
        """T11-1: full sync completes without exception."""
        ms, _repo, _store = self._setup(tmp_path)
        ms.sync(full=True)  # Should not raise

    def test_t11_2_cursor_updated(self, tmp_path: Path):
        """T11-2: cursor is updated to HEAD after full sync."""
        ms, repo, _store = self._setup(tmp_path)
        ms.sync(full=True)
        cursor = get_cursor(repo)
        assert cursor["last_synced_commit"] == get_head(repo)

    def test_t11_3_doc_id_sets_identical(self, tmp_path: Path):
        """T11-3: doc_id sets are identical between initial and full sync."""
        ms, _repo, store = self._setup(tmp_path)
        doc_ids_a = store.get_all_doc_ids()
        ms.sync(full=True)
        doc_ids_b = store.get_all_doc_ids()
        assert doc_ids_a == doc_ids_b

    def test_t11_4_all_docs_same_seen_token(self, tmp_path: Path):
        """T11-4: After full sync, all docs have the same seen_token."""
        ms, _repo, store = self._setup(tmp_path)
        ms.sync(full=True)
        all_docs = store.get_all_docs()
        assert len(all_docs) > 0
        seen_tokens = {d.get("seen_token") for d in all_docs}
        assert len(seen_tokens) == 1, f"Expected 1 unique seen_token, got {len(seen_tokens)}: {seen_tokens}"


# ============================================================================
# T12: deterministic ID -- different locations
# ============================================================================


class TestT12DeterministicIDDifferentLocations:
    """T12: Clone same repo to two paths, init+sync each, verify identical IDs."""

    def test_t12_1_same_repo_id(self, tmp_path: Path):
        """T12-1: Both locations have the same repo_id (root commit hash)."""

        path_a = tmp_path / "a"
        path_a.mkdir()
        repo_a = create_test_repo(path_a, SAMPLE_FILES)

        # Clone repo_a to path_b
        path_b = tmp_path / "b"
        subprocess.run(
            ["git", "clone", str(repo_a), str(path_b / "repo")],
            capture_output=True,
            text=True,
            check=True,
        )
        repo_b = path_b / "repo"

        root_a = get_root_commit(repo_a)
        root_b = get_root_commit(repo_b)
        assert root_a == root_b

    def test_t12_2_doc_id_sets_identical(self, tmp_path: Path):
        """T12-2: doc_id sets are identical across both locations."""
        from minsync import MinSync

        path_a = tmp_path / "a"
        path_a.mkdir()
        repo_a = create_test_repo(path_a, SAMPLE_FILES)

        path_b = tmp_path / "b"
        subprocess.run(
            ["git", "clone", str(repo_a), str(path_b / "repo")],
            capture_output=True,
            text=True,
            check=True,
        )
        repo_b = path_b / "repo"

        store_a = MockVectorStore()
        store_b = MockVectorStore()
        chunker_a = MockChunker()
        chunker_b = MockChunker()
        embedder_a = MockEmbedder()
        embedder_b = MockEmbedder()

        ms_a = MinSync(repo_path=repo_a, chunker=chunker_a, embedder=embedder_a, vector_store=store_a)
        ms_a.init()
        ms_a.sync()

        ms_b = MinSync(repo_path=repo_b, chunker=chunker_b, embedder=embedder_b, vector_store=store_b)
        ms_b.init()
        ms_b.sync()

        ids_a = store_a.get_all_doc_ids()
        ids_b = store_b.get_all_doc_ids()
        assert ids_a == ids_b, f"Doc IDs differ: only in A={ids_a - ids_b}, only in B={ids_b - ids_a}"

    def test_t12_3_doc_content_matches(self, tmp_path: Path):
        """T12-3: Each doc_id has the same text content across both locations."""
        from minsync import MinSync

        path_a = tmp_path / "a"
        path_a.mkdir()
        repo_a = create_test_repo(path_a, SAMPLE_FILES)

        path_b = tmp_path / "b"
        subprocess.run(
            ["git", "clone", str(repo_a), str(path_b / "repo")],
            capture_output=True,
            text=True,
            check=True,
        )
        repo_b = path_b / "repo"

        store_a = MockVectorStore()
        store_b = MockVectorStore()

        ms_a = MinSync(repo_path=repo_a, chunker=MockChunker(), embedder=MockEmbedder(), vector_store=store_a)
        ms_a.init()
        ms_a.sync()

        ms_b = MinSync(repo_path=repo_b, chunker=MockChunker(), embedder=MockEmbedder(), vector_store=store_b)
        ms_b.init()
        ms_b.sync()

        docs_a = {d["id"]: d for d in store_a.get_all_docs()}
        docs_b = {d["id"]: d for d in store_b.get_all_docs()}

        assert set(docs_a.keys()) == set(docs_b.keys())
        for doc_id in docs_a:
            assert docs_a[doc_id]["text"] == docs_b[doc_id]["text"], f"Text mismatch for doc_id={doc_id}"


# ============================================================================
# T13: deterministic ID -- rebuild identical
# ============================================================================


class TestT13DeterministicIDRebuild:
    """T13: sync -> record IDs -> sync(full=True) -> record IDs -> verify identical."""

    def test_t13_1_sets_identical(self, tmp_path: Path):
        """T13-1: set_a == set_b after full rebuild."""
        from minsync import MinSync

        store = MockVectorStore()
        embedder = MockEmbedder()
        chunker = MockChunker()
        repo = create_test_repo(tmp_path, SAMPLE_FILES)
        ms = MinSync(repo_path=repo, chunker=chunker, embedder=embedder, vector_store=store)
        ms.init()
        ms.sync()

        set_a = store.get_all_doc_ids()
        assert len(set_a) > 0, "Initial sync should produce documents"

        ms.sync(full=True)

        set_b = store.get_all_doc_ids()
        assert set_a == set_b, f"Doc ID sets differ after rebuild: only in A={set_a - set_b}, only in B={set_b - set_a}"


# ============================================================================
# T28: mark+sweep convergence
# ============================================================================


class TestT28MarkSweepConvergence:
    """T28: Repeated modifications to a file should leave only current snapshot chunks."""

    def _get_expected_chunks_for_file(self, chunker: MockChunker, text: str, path: str) -> list:
        """Get the chunks that the chunker would produce for given text."""
        return chunker.chunk(text, path)

    def test_t28_convergence(self, tmp_path: Path):
        """T28-1 & T28-2: After each sync, only current snapshot chunks exist for the file."""
        from minsync import MinSync

        store = MockVectorStore()
        embedder = MockEmbedder()
        chunker = MockChunker()
        repo = create_test_repo(tmp_path, SAMPLE_FILES)
        ms = MinSync(repo_path=repo, chunker=chunker, embedder=embedder, vector_store=store)
        ms.init()
        ms.sync()

        # Step 1: Add file_a.md with initial content
        file_a_v1 = "# File A\n\n## Section One\n\nInitial content for section one.\n"
        add_commit(repo, {"file_a.md": file_a_v1}, "add file_a.md v1")
        ms.sync()

        # Verify: only current chunks for file_a.md
        docs_v1 = store.get_docs_by_path("file_a.md")
        assert len(docs_v1) > 0, "file_a.md should have chunks after first sync"
        expected_v1 = chunker.chunk(file_a_v1, "file_a.md")
        assert len(docs_v1) == len(expected_v1), f"Expected {len(expected_v1)} chunks, got {len(docs_v1)}"

        # Step 2: Heavily modify file_a.md (different chunk structure)
        file_a_v2 = (
            "# File A Rewritten\n\n"
            "## New Section Alpha\n\nCompletely new content for alpha.\n\n"
            "## New Section Beta\n\nCompletely new content for beta.\n\n"
            "## New Section Gamma\n\nCompletely new content for gamma.\n"
        )
        add_commit(repo, {"file_a.md": file_a_v2}, "rewrite file_a.md v2")
        ms.sync()

        docs_v2 = store.get_docs_by_path("file_a.md")
        expected_v2 = chunker.chunk(file_a_v2, "file_a.md")
        assert len(docs_v2) == len(expected_v2), f"Expected {len(expected_v2)} chunks after v2, got {len(docs_v2)}"
        # Ensure no stale v1 text remains
        v2_texts = {d["text"] for d in docs_v2}
        assert "Initial content for section one." not in v2_texts, "Stale v1 chunk text should not remain after v2 sync"

        # Step 3: Modify file_a.md again
        file_a_v3 = "# File A Final\n\n## Summary\n\nFinal summary of the document.\n"
        add_commit(repo, {"file_a.md": file_a_v3}, "modify file_a.md v3")
        ms.sync()

        docs_v3 = store.get_docs_by_path("file_a.md")
        expected_v3 = chunker.chunk(file_a_v3, "file_a.md")
        assert len(docs_v3) == len(expected_v3), f"Expected {len(expected_v3)} chunks after v3, got {len(docs_v3)}"
        # Ensure no stale v2 text remains
        v3_texts = {d["text"] for d in docs_v3}
        assert "Completely new content for alpha." not in v3_texts, (
            "Stale v2 chunk text should not remain after v3 sync"
        )
        assert "Completely new content for beta." not in v3_texts, "Stale v2 chunk text should not remain after v3 sync"


# ============================================================================
# T29: large incremental performance
# ============================================================================


class TestT29LargeIncrementalPerformance:
    """T29: Generate 100 files -> sync -> modify 3 -> sync -> verify efficiency."""

    def _setup(self, tmp_path: Path):
        from minsync import MinSync

        store = MockVectorStore()
        embedder = MockEmbedder()
        chunker = MockChunker()

        # Generate 100 files
        files = {}
        for i in range(100):
            files[f"docs/file_{i:03d}.md"] = (
                f"# Document {i}\n\n"
                f"## Section A\n\nContent for document {i} section A.\n\n"
                f"## Section B\n\nContent for document {i} section B.\n"
            )
        repo = create_test_repo(tmp_path, files)
        ms = MinSync(repo_path=repo, chunker=chunker, embedder=embedder, vector_store=store)
        ms.init()
        ms.sync()

        return ms, repo, store, embedder, chunker, files

    def test_t29_1_only_3_files_processed(self, tmp_path: Path):
        """T29-1: Only 3 files are processed in the incremental sync."""
        ms, repo, _store, _embedder, _chunker, _files = self._setup(tmp_path)

        # Modify only 3 files
        add_commit(
            repo,
            {
                "docs/file_010.md": "# Document 10 Updated\n\n## New Content\n\nModified content.\n",
                "docs/file_050.md": "# Document 50 Updated\n\n## New Content\n\nModified content.\n",
                "docs/file_090.md": "# Document 90 Updated\n\n## New Content\n\nModified content.\n",
            },
            "modify 3 files",
        )

        result = ms.sync()
        assert result.files_processed == 3, f"Expected 3 files processed, got {result.files_processed}"

    def test_t29_2_embedder_calls_proportional(self, tmp_path: Path):
        """T29-2: Embedder call count is proportional to 3 files' chunks only."""
        ms, repo, _store, embedder, chunker, _files = self._setup(tmp_path)

        # Record doc IDs before modification
        modified_files = {
            "docs/file_010.md": "# Document 10 Updated\n\n## New Content\n\nModified content.\n",
            "docs/file_050.md": "# Document 50 Updated\n\n## New Content\n\nModified content.\n",
            "docs/file_090.md": "# Document 90 Updated\n\n## New Content\n\nModified content.\n",
        }
        add_commit(repo, modified_files, "modify 3 files")

        # Reset embedder counters before incremental sync
        embedder.call_count = 0
        embedder.total_texts_embedded = 0

        ms.sync()

        # Calculate expected new chunks for the 3 modified files
        total_expected_new_chunks = 0
        for path, content in modified_files.items():
            chunks = chunker.chunk(content, path)
            total_expected_new_chunks += len(chunks)

        # The embedder should only be called for new chunks from the 3 files.
        # Some chunks might already exist (same content_hash), so
        # total_texts_embedded <= total_expected_new_chunks
        assert embedder.total_texts_embedded <= total_expected_new_chunks, (
            f"Embedder processed {embedder.total_texts_embedded} texts, "
            f"but only {total_expected_new_chunks} new chunks expected from 3 files"
        )
        # Should have embedded at least some texts (the modified content is different)
        assert embedder.total_texts_embedded > 0, (
            "Embedder should have processed at least some texts for modified files"
        )

    def test_t29_3_other_97_files_unchanged(self, tmp_path: Path):
        """T29-3: The other 97 files' doc_ids remain unchanged."""
        ms, repo, store, _embedder, _chunker, _files = self._setup(tmp_path)

        # Collect doc_ids for non-modified files before
        modified_paths = {"docs/file_010.md", "docs/file_050.md", "docs/file_090.md"}
        unmodified_doc_ids_before = set()
        for doc in store.get_all_docs():
            if doc.get("path") not in modified_paths:
                unmodified_doc_ids_before.add(doc["id"])

        # Modify only 3 files
        add_commit(
            repo,
            {
                "docs/file_010.md": "# Document 10 Updated\n\n## New Content\n\nModified content.\n",
                "docs/file_050.md": "# Document 50 Updated\n\n## New Content\n\nModified content.\n",
                "docs/file_090.md": "# Document 90 Updated\n\n## New Content\n\nModified content.\n",
            },
            "modify 3 files",
        )
        ms.sync()

        # Collect doc_ids for non-modified files after
        unmodified_doc_ids_after = set()
        for doc in store.get_all_docs():
            if doc.get("path") not in modified_paths:
                unmodified_doc_ids_after.add(doc["id"])

        assert unmodified_doc_ids_before == unmodified_doc_ids_after, "Doc IDs for unmodified files should not change"


# ============================================================================
# T30: .minsyncignore basic filtering
# ============================================================================


class TestT30MinsyncignoreBasicFiltering:
    """T30: .minsyncignore filters files from indexing."""

    def _setup(self, tmp_path: Path):
        from minsync import MinSync

        store = MockVectorStore()
        embedder = MockEmbedder()
        chunker = MockChunker()

        # Start with SAMPLE_FILES and add .minsyncignore
        files = dict(SAMPLE_FILES)
        files[".minsyncignore"] = "src/**/*.py\nnotes/\n"

        repo = create_test_repo(tmp_path, files)
        ms = MinSync(repo_path=repo, chunker=chunker, embedder=embedder, vector_store=store)
        ms.init()
        ms.sync()
        return ms, repo, store

    def test_t30_1_guide_md_indexed(self, tmp_path: Path):
        """T30-1: docs/guide.md is indexed."""
        _ms, _repo, store = self._setup(tmp_path)
        docs = store.get_docs_by_path("docs/guide.md")
        assert len(docs) > 0, "docs/guide.md should be indexed"

    def test_t30_2_api_md_indexed(self, tmp_path: Path):
        """T30-2: docs/api.md is indexed."""
        _ms, _repo, store = self._setup(tmp_path)
        docs = store.get_docs_by_path("docs/api.md")
        assert len(docs) > 0, "docs/api.md should be indexed"

    def test_t30_3_main_py_not_indexed(self, tmp_path: Path):
        """T30-3: src/main.py is NOT indexed (0 chunks)."""
        _ms, _repo, store = self._setup(tmp_path)
        docs = store.get_docs_by_path("src/main.py")
        assert len(docs) == 0, f"src/main.py should not be indexed, but found {len(docs)} chunks"

    def test_t30_4_utils_py_not_indexed(self, tmp_path: Path):
        """T30-4: src/utils.py is NOT indexed (0 chunks)."""
        _ms, _repo, store = self._setup(tmp_path)
        docs = store.get_docs_by_path("src/utils.py")
        assert len(docs) == 0, f"src/utils.py should not be indexed, but found {len(docs)} chunks"

    def test_t30_5_meeting_txt_not_indexed(self, tmp_path: Path):
        """T30-5: notes/meeting.txt is NOT indexed (0 chunks)."""
        _ms, _repo, store = self._setup(tmp_path)
        docs = store.get_docs_by_path("notes/meeting.txt")
        assert len(docs) == 0, f"notes/meeting.txt should not be indexed, but found {len(docs)} chunks"

    def test_t30_6_readme_indexed(self, tmp_path: Path):
        """T30-6: README.md is indexed."""
        _ms, _repo, store = self._setup(tmp_path)
        docs = store.get_docs_by_path("README.md")
        assert len(docs) > 0, "README.md should be indexed"


# ============================================================================
# T31: schema/embedder mismatch detection
# ============================================================================


class TestT31SchemaMismatchDetection:
    """T31: After init+sync, change config.yaml's chunker.id, then sync()."""

    def _setup(self, tmp_path: Path):
        from minsync import MinSync

        store = MockVectorStore()
        embedder = MockEmbedder()
        chunker = MockChunker()
        repo = create_test_repo(tmp_path, SAMPLE_FILES)
        ms = MinSync(repo_path=repo, chunker=chunker, embedder=embedder, vector_store=store)
        ms.init()
        ms.sync()
        return ms, repo, store, chunker, embedder

    def _tamper_config_chunker_id(self, repo: Path):
        """Modify config.yaml to change the chunker.id, simulating a mismatch."""
        config_path = repo / ".minsync" / "config.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["chunker"]["id"] = "different-chunker-v99"
        config_path.write_text(yaml.dump(config, default_flow_style=False), encoding="utf-8")

    def test_t31_1_sync_raises_exception(self, tmp_path: Path):
        """T31-1: sync() raises an exception on schema/embedder mismatch."""
        ms, repo, _store, _chunker, _embedder = self._setup(tmp_path)
        self._tamper_config_chunker_id(repo)
        with pytest.raises(Exception):  # noqa: B017
            ms.sync()

    def test_t31_2_error_contains_mismatch_message(self, tmp_path: Path):
        """T31-2: The error message contains 'schema/embedder mismatch'."""
        ms, repo, _store, _chunker, _embedder = self._setup(tmp_path)
        self._tamper_config_chunker_id(repo)
        with pytest.raises(Exception) as exc_info:
            ms.sync()
        error_msg = str(exc_info.value).lower()
        assert "schema" in error_msg or "mismatch" in error_msg, (
            f"Error message should contain 'schema' or 'mismatch', got: {exc_info.value}"
        )

    def test_t31_3_cursor_unchanged(self, tmp_path: Path):
        """T31-3: cursor.json is unchanged after mismatch error."""
        ms, repo, _store, _chunker, _embedder = self._setup(tmp_path)
        cursor_before = get_cursor(repo)
        self._tamper_config_chunker_id(repo)
        with pytest.raises(Exception):  # noqa: B017
            ms.sync()
        cursor_after = get_cursor(repo)
        assert cursor_after == cursor_before

    def test_t31_4_full_sync_succeeds_after_mismatch(self, tmp_path: Path):
        """T31-4: sync(full=True) succeeds after a mismatch."""
        ms, repo, _store, _chunker, _embedder = self._setup(tmp_path)
        self._tamper_config_chunker_id(repo)

        # Normal sync should fail
        with pytest.raises(Exception):  # noqa: B017
            ms.sync()

        # Full sync should succeed (rebuilds everything)
        ms.sync(full=True)
        cursor = get_cursor(repo)
        assert cursor["last_synced_commit"] == get_head(repo)
