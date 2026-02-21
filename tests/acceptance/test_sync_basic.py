"""E2E tests for MinSync sync — T03 through T09, T18, T19.

Tests cover: initial full indexing, incremental add/modify/delete/rename,
compound changes, multi-commit sync, already-up-to-date, and sync
without initialization.

These tests are written TDD-style: the implementation does NOT exist yet.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import (
    SAMPLE_FILES,
    add_commit,
    delete_commit,
    get_cursor,
    get_head,
    get_root_commit,
    rename_commit,
)
from tests.mock_components import MockChunker, MockEmbedder, MockVectorStore

# ---------------------------------------------------------------------------
# Helper: create a fully-wired MinSync instance and return (ms, store, repo)
# ---------------------------------------------------------------------------


def _make_minsync(repo: Path):
    """Create a MinSync with mock components.

    Returns ``(ms, store)`` where *store* is the MockVectorStore so tests
    can inspect its state directly.
    """
    from minsync import MinSync

    store = MockVectorStore()
    ms = MinSync(
        repo_path=repo,
        chunker=MockChunker(),
        embedder=MockEmbedder(),
        vector_store=store,
    )
    return ms, store


# ===========================================================================
# T03: sync -- first full indexing
# ===========================================================================


class TestT03InitialFullSync:
    """After init() + sync() the vector store should contain chunks for every
    git-tracked sample file, cursor.json should point to HEAD, and transient
    files (txn.json, lock) should be cleaned up.
    """

    # -- T03-1 ---------------------------------------------------------------
    def test_t03_1_sync_succeeds(self, test_repo: Path):
        ms, _store = _make_minsync(test_repo)
        ms.init()
        ms.sync()  # should not raise

    # -- T03-2 ---------------------------------------------------------------
    def test_t03_2_cursor_json_exists(self, test_repo: Path):
        ms, _store = _make_minsync(test_repo)
        ms.init()
        ms.sync()
        assert (test_repo / ".minsync" / "cursor.json").is_file()

    # -- T03-3 ---------------------------------------------------------------
    def test_t03_3_cursor_matches_head(self, test_repo: Path):
        ms, _store = _make_minsync(test_repo)
        ms.init()
        ms.sync()

        cursor = get_cursor(test_repo)
        head = get_head(test_repo)
        assert cursor["last_synced_commit"] == head

    # -- T03-4 ---------------------------------------------------------------
    def test_t03_4_txn_json_does_not_exist(self, test_repo: Path):
        ms, _store = _make_minsync(test_repo)
        ms.init()
        ms.sync()
        assert not (test_repo / ".minsync" / "txn.json").exists()

    # -- T03-5 ---------------------------------------------------------------
    def test_t03_5_lock_file_does_not_exist(self, test_repo: Path):
        ms, _store = _make_minsync(test_repo)
        ms.init()
        ms.sync()
        assert not (test_repo / ".minsync" / "lock").exists()

    # -- T03-6 ---------------------------------------------------------------
    def test_t03_6_doc_count_positive(self, test_repo: Path):
        ms, store = _make_minsync(test_repo)
        ms.init()
        ms.sync()
        assert store.doc_count() > 0

    # -- T03-7 ---------------------------------------------------------------
    def test_t03_7_all_sample_files_have_chunks(self, test_repo: Path):
        ms, store = _make_minsync(test_repo)
        ms.init()
        ms.sync()

        for path in SAMPLE_FILES:
            docs = store.get_docs_by_path(path)
            assert len(docs) >= 1, f"Expected at least 1 chunk for {path}, got 0"

    # -- T03-8 ---------------------------------------------------------------
    def test_t03_8_all_docs_have_matching_repo_id(self, test_repo: Path):
        ms, store = _make_minsync(test_repo)
        ms.init()
        ms.sync()

        repo_id = get_root_commit(test_repo)
        for doc in store.get_all_docs():
            assert doc["repo_id"] == repo_id, f"Doc {doc['id']} has repo_id={doc['repo_id']!r}, expected {repo_id!r}"

    # -- T03-9 ---------------------------------------------------------------
    def test_t03_9_all_docs_have_ref_main(self, test_repo: Path):
        ms, store = _make_minsync(test_repo)
        ms.init()
        ms.sync()

        for doc in store.get_all_docs():
            assert doc["ref"] == "main", f"Doc {doc['id']} has ref={doc['ref']!r}, expected 'main'"

    # -- T03-10 --------------------------------------------------------------
    def test_t03_10_sync_result_has_files_processed(self, test_repo: Path):
        ms, _store = _make_minsync(test_repo)
        ms.init()
        result = ms.sync()
        assert result.files_processed > 0


# ===========================================================================
# T04: sync -- incremental add
# ===========================================================================


class TestT04IncrementalAdd:
    """After initial sync, adding a new file and syncing again should
    index the new file while keeping existing chunks intact.
    """

    def _setup(self, test_repo: Path):
        ms, store = _make_minsync(test_repo)
        ms.init()
        ms.sync()
        return ms, store

    # -- T04-1 ---------------------------------------------------------------
    def test_t04_1_sync_succeeds(self, test_repo: Path):
        ms, store = self._setup(test_repo)
        store.doc_count()

        add_commit(
            test_repo,
            {
                "docs/tutorial.md": ("# Tutorial\n\n## Step 1\n\nFollow this tutorial to get started.\n"),
            },
            "add tutorial",
        )

        ms.sync()  # should not raise

    # -- T04-2 ---------------------------------------------------------------
    def test_t04_2_cursor_updated(self, test_repo: Path):
        ms, _store = self._setup(test_repo)

        new_head = add_commit(
            test_repo,
            {
                "docs/tutorial.md": "# Tutorial\n\nLearn the basics.\n",
            },
            "add tutorial",
        )

        ms.sync()
        cursor = get_cursor(test_repo)
        assert cursor["last_synced_commit"] == new_head

    # -- T04-3 ---------------------------------------------------------------
    def test_t04_3_tutorial_has_chunks(self, test_repo: Path):
        ms, store = self._setup(test_repo)

        add_commit(
            test_repo,
            {
                "docs/tutorial.md": "# Tutorial\n\nLearn the basics.\n",
            },
            "add tutorial",
        )

        ms.sync()
        docs = store.get_docs_by_path("docs/tutorial.md")
        assert len(docs) >= 1

    # -- T04-4 ---------------------------------------------------------------
    def test_t04_4_existing_file_chunks_unchanged(self, test_repo: Path):
        ms, store = self._setup(test_repo)

        # Record per-path doc counts before the incremental sync.
        counts_before: dict[str, int] = {}
        for path in SAMPLE_FILES:
            counts_before[path] = len(store.get_docs_by_path(path))

        add_commit(
            test_repo,
            {
                "docs/tutorial.md": "# Tutorial\n\nLearn the basics.\n",
            },
            "add tutorial",
        )
        ms.sync()

        for path, expected_count in counts_before.items():
            actual = len(store.get_docs_by_path(path))
            assert actual == expected_count, f"{path}: expected {expected_count} chunks, got {actual}"

    # -- T04-5 ---------------------------------------------------------------
    def test_t04_5_total_doc_count_increased(self, test_repo: Path):
        ms, store = self._setup(test_repo)
        count_before = store.doc_count()

        add_commit(
            test_repo,
            {
                "docs/tutorial.md": "# Tutorial\n\nLearn the basics.\n",
            },
            "add tutorial",
        )
        ms.sync()

        assert store.doc_count() > count_before


# ===========================================================================
# T05: sync -- incremental modify
# ===========================================================================


class TestT05IncrementalModify:
    """After initial sync, modifying a file and syncing should update its
    chunks without leaving stale data.
    """

    def _setup(self, test_repo: Path):
        ms, store = _make_minsync(test_repo)
        ms.init()
        ms.sync()
        return ms, store

    # -- T05-1 ---------------------------------------------------------------
    def test_t05_1_sync_succeeds(self, test_repo: Path):
        ms, _store = self._setup(test_repo)

        modified_guide = SAMPLE_FILES["docs/guide.md"] + ("\n## Troubleshooting\n\nCheck the logs for errors.\n")
        add_commit(test_repo, {"docs/guide.md": modified_guide}, "update guide")
        ms.sync()  # should not raise

    # -- T05-2 ---------------------------------------------------------------
    def test_t05_2_cursor_updated(self, test_repo: Path):
        ms, _store = self._setup(test_repo)

        modified_guide = SAMPLE_FILES["docs/guide.md"] + ("\n## Troubleshooting\n\nCheck the logs for errors.\n")
        new_head = add_commit(test_repo, {"docs/guide.md": modified_guide}, "update guide")
        ms.sync()

        cursor = get_cursor(test_repo)
        assert cursor["last_synced_commit"] == new_head

    # -- T05-3 ---------------------------------------------------------------
    def test_t05_3_guide_chunks_contain_troubleshooting(self, test_repo: Path):
        ms, store = self._setup(test_repo)

        modified_guide = SAMPLE_FILES["docs/guide.md"] + ("\n## Troubleshooting\n\nCheck the logs for errors.\n")
        add_commit(test_repo, {"docs/guide.md": modified_guide}, "update guide")
        ms.sync()

        guide_docs = store.get_docs_by_path("docs/guide.md")
        all_text = " ".join(d["text"] for d in guide_docs)
        assert "Troubleshooting" in all_text

    # -- T05-4 ---------------------------------------------------------------
    def test_t05_4_no_stale_chunks_for_guide(self, test_repo: Path):
        """After modification all guide.md chunks should reflect the current
        content -- no leftover chunks from the previous version that are
        not part of the new chunking result.
        """
        ms, store = self._setup(test_repo)

        modified_guide = SAMPLE_FILES["docs/guide.md"] + ("\n## Troubleshooting\n\nCheck the logs for errors.\n")
        add_commit(test_repo, {"docs/guide.md": modified_guide}, "update guide")
        ms.sync()

        # Re-chunk the modified content to get the expected set of chunk texts.
        chunker = MockChunker()
        expected_chunks = chunker.chunk(modified_guide, "docs/guide.md")
        expected_texts = {c.text for c in expected_chunks}

        actual_docs = store.get_docs_by_path("docs/guide.md")
        actual_texts = {d["text"] for d in actual_docs}

        assert actual_texts == expected_texts, (
            f"Stale chunks detected.\n  Expected: {expected_texts}\n  Actual:   {actual_texts}"
        )

    # -- T05-5 ---------------------------------------------------------------
    def test_t05_5_other_file_chunks_unchanged(self, test_repo: Path):
        ms, store = self._setup(test_repo)

        other_paths = [p for p in SAMPLE_FILES if p != "docs/guide.md"]
        counts_before = {p: len(store.get_docs_by_path(p)) for p in other_paths}

        modified_guide = SAMPLE_FILES["docs/guide.md"] + ("\n## Troubleshooting\n\nCheck the logs for errors.\n")
        add_commit(test_repo, {"docs/guide.md": modified_guide}, "update guide")
        ms.sync()

        for path, expected in counts_before.items():
            actual = len(store.get_docs_by_path(path))
            assert actual == expected, f"{path}: expected {expected} chunks, got {actual}"


# ===========================================================================
# T06: sync -- incremental delete
# ===========================================================================


class TestT06IncrementalDelete:
    """After initial sync, deleting a file and syncing should remove all
    its chunks from the vector store.
    """

    def _setup(self, test_repo: Path):
        ms, store = _make_minsync(test_repo)
        ms.init()
        ms.sync()
        return ms, store

    # -- T06-1 ---------------------------------------------------------------
    def test_t06_1_sync_succeeds(self, test_repo: Path):
        ms, _store = self._setup(test_repo)
        delete_commit(test_repo, ["docs/faq.md"], "delete faq")
        ms.sync()  # should not raise

    # -- T06-2 ---------------------------------------------------------------
    def test_t06_2_faq_has_zero_chunks(self, test_repo: Path):
        ms, store = self._setup(test_repo)
        delete_commit(test_repo, ["docs/faq.md"], "delete faq")
        ms.sync()

        docs = store.get_docs_by_path("docs/faq.md")
        assert len(docs) == 0

    # -- T06-3 ---------------------------------------------------------------
    def test_t06_3_other_files_unchanged(self, test_repo: Path):
        ms, store = self._setup(test_repo)

        other_paths = [p for p in SAMPLE_FILES if p != "docs/faq.md"]
        counts_before = {p: len(store.get_docs_by_path(p)) for p in other_paths}

        delete_commit(test_repo, ["docs/faq.md"], "delete faq")
        ms.sync()

        for path, expected in counts_before.items():
            actual = len(store.get_docs_by_path(path))
            assert actual == expected, f"{path}: expected {expected} chunks, got {actual}"

    # -- T06-4 ---------------------------------------------------------------
    def test_t06_4_total_doc_count_decreased(self, test_repo: Path):
        ms, store = self._setup(test_repo)
        count_before = store.doc_count()

        delete_commit(test_repo, ["docs/faq.md"], "delete faq")
        ms.sync()

        assert store.doc_count() < count_before


# ===========================================================================
# T07: sync -- rename
# ===========================================================================


class TestT07Rename:
    """Renaming a file should remove chunks under the old path and create
    chunks under the new path with the same textual content.
    """

    def _setup(self, test_repo: Path):
        ms, store = _make_minsync(test_repo)
        ms.init()
        ms.sync()
        return ms, store

    # -- T07-1 ---------------------------------------------------------------
    def test_t07_1_sync_succeeds(self, test_repo: Path):
        ms, _store = self._setup(test_repo)
        rename_commit(test_repo, "docs/api.md", "docs/reference.md", "rename api->reference")
        ms.sync()  # should not raise

    # -- T07-2 ---------------------------------------------------------------
    def test_t07_2_api_has_zero_chunks(self, test_repo: Path):
        ms, store = self._setup(test_repo)
        rename_commit(test_repo, "docs/api.md", "docs/reference.md", "rename api->reference")
        ms.sync()

        docs = store.get_docs_by_path("docs/api.md")
        assert len(docs) == 0

    # -- T07-3 ---------------------------------------------------------------
    def test_t07_3_reference_has_chunks(self, test_repo: Path):
        ms, store = self._setup(test_repo)
        rename_commit(test_repo, "docs/api.md", "docs/reference.md", "rename api->reference")
        ms.sync()

        docs = store.get_docs_by_path("docs/reference.md")
        assert len(docs) > 0

    # -- T07-4 ---------------------------------------------------------------
    def test_t07_4_reference_texts_match_old_api(self, test_repo: Path):
        ms, store = self._setup(test_repo)

        # Capture the chunk texts for api.md before the rename.
        old_api_texts = sorted(d["text"] for d in store.get_docs_by_path("docs/api.md"))

        rename_commit(test_repo, "docs/api.md", "docs/reference.md", "rename api->reference")
        ms.sync()

        new_ref_texts = sorted(d["text"] for d in store.get_docs_by_path("docs/reference.md"))
        assert new_ref_texts == old_api_texts


# ===========================================================================
# T08: sync -- compound change (add + modify + delete in single commit)
# ===========================================================================


class TestT08CompoundChange:
    """A single commit containing an add, a modify, and a delete should all
    be handled correctly in one sync() call.
    """

    def _setup(self, test_repo: Path):
        ms, store = _make_minsync(test_repo)
        ms.init()
        ms.sync()
        return ms, store

    def _make_compound_commit(self, test_repo: Path) -> str:
        """In a single commit: add new-feature.md, modify guide.md, delete
        docs/auth/oauth.md.  Returns the new HEAD hash.
        """
        import subprocess

        modified_guide = SAMPLE_FILES["docs/guide.md"] + (
            "\n## New Section\n\nThis section was added in the compound commit.\n"
        )

        # Write new file
        new_feature_path = test_repo / "docs" / "new-feature.md"
        new_feature_path.write_text(
            "# New Feature\n\n## Overview\n\nThis is a brand new feature.\n",
            encoding="utf-8",
        )

        # Modify existing file
        (test_repo / "docs" / "guide.md").write_text(modified_guide, encoding="utf-8")

        # Delete a file
        (test_repo / "docs" / "auth" / "oauth.md").unlink()

        # Stage all and commit
        subprocess.run(["git", "add", "-A"], cwd=test_repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "compound: add+modify+delete"],
            cwd=test_repo,
            check=True,
            capture_output=True,
        )
        return get_head(test_repo)

    # -- T08-1 ---------------------------------------------------------------
    def test_t08_1_sync_succeeds(self, test_repo: Path):
        ms, _store = self._setup(test_repo)
        self._make_compound_commit(test_repo)
        ms.sync()  # should not raise

    # -- T08-2 ---------------------------------------------------------------
    def test_t08_2_new_feature_has_chunks(self, test_repo: Path):
        ms, store = self._setup(test_repo)
        self._make_compound_commit(test_repo)
        ms.sync()

        docs = store.get_docs_by_path("docs/new-feature.md")
        assert len(docs) > 0

    # -- T08-3 ---------------------------------------------------------------
    def test_t08_3_guide_updated(self, test_repo: Path):
        ms, store = self._setup(test_repo)
        self._make_compound_commit(test_repo)
        ms.sync()

        guide_docs = store.get_docs_by_path("docs/guide.md")
        all_text = " ".join(d["text"] for d in guide_docs)
        assert "New Section" in all_text

    # -- T08-4 ---------------------------------------------------------------
    def test_t08_4_oauth_has_zero_chunks(self, test_repo: Path):
        ms, store = self._setup(test_repo)
        self._make_compound_commit(test_repo)
        ms.sync()

        docs = store.get_docs_by_path("docs/auth/oauth.md")
        assert len(docs) == 0

    # -- T08-5 ---------------------------------------------------------------
    def test_t08_5_login_and_faq_unchanged(self, test_repo: Path):
        ms, store = self._setup(test_repo)

        login_count = len(store.get_docs_by_path("docs/auth/login.md"))
        faq_count = len(store.get_docs_by_path("docs/faq.md"))

        self._make_compound_commit(test_repo)
        ms.sync()

        assert len(store.get_docs_by_path("docs/auth/login.md")) == login_count
        assert len(store.get_docs_by_path("docs/faq.md")) == faq_count


# ===========================================================================
# T09: sync -- multiple commits
# ===========================================================================


class TestT09MultipleCommits:
    """Three sequential commits (A: add a.md, B: add b.md + modify guide.md,
    C: delete a.md) followed by a single sync() call.  The final state should
    reflect the net effect of all three commits.
    """

    def _setup(self, test_repo: Path):
        ms, store = _make_minsync(test_repo)
        ms.init()
        ms.sync()
        return ms, store

    def _make_three_commits(self, test_repo: Path) -> str:
        """Create commits A, B, C and return the hash of commit C."""
        # Commit A: add docs/a.md
        add_commit(
            test_repo,
            {
                "docs/a.md": "# Document A\n\nTemporary document.\n",
            },
            "commit A: add a.md",
        )

        # Commit B: add docs/b.md + modify docs/guide.md
        modified_guide = SAMPLE_FILES["docs/guide.md"] + (
            "\n## Advanced Usage\n\nFor advanced users, configure custom pipelines.\n"
        )
        add_commit(
            test_repo,
            {
                "docs/b.md": "# Document B\n\nPermanent document.\n",
                "docs/guide.md": modified_guide,
            },
            "commit B: add b.md, modify guide",
        )

        # Commit C: delete docs/a.md
        commit_c = delete_commit(test_repo, ["docs/a.md"], "commit C: delete a.md")
        return commit_c

    # -- T09-1 ---------------------------------------------------------------
    def test_t09_1_sync_succeeds(self, test_repo: Path):
        ms, _store = self._setup(test_repo)
        self._make_three_commits(test_repo)
        ms.sync()  # should not raise

    # -- T09-2 ---------------------------------------------------------------
    def test_t09_2_cursor_points_to_commit_c(self, test_repo: Path):
        ms, _store = self._setup(test_repo)
        commit_c = self._make_three_commits(test_repo)
        ms.sync()

        cursor = get_cursor(test_repo)
        assert cursor["last_synced_commit"] == commit_c

    # -- T09-3 ---------------------------------------------------------------
    def test_t09_3_a_has_zero_chunks(self, test_repo: Path):
        ms, store = self._setup(test_repo)
        self._make_three_commits(test_repo)
        ms.sync()

        docs = store.get_docs_by_path("docs/a.md")
        assert len(docs) == 0

    # -- T09-4 ---------------------------------------------------------------
    def test_t09_4_b_has_chunks(self, test_repo: Path):
        ms, store = self._setup(test_repo)
        self._make_three_commits(test_repo)
        ms.sync()

        docs = store.get_docs_by_path("docs/b.md")
        assert len(docs) > 0

    # -- T09-5 ---------------------------------------------------------------
    def test_t09_5_guide_reflects_commit_b_changes(self, test_repo: Path):
        ms, store = self._setup(test_repo)
        self._make_three_commits(test_repo)
        ms.sync()

        guide_docs = store.get_docs_by_path("docs/guide.md")
        all_text = " ".join(d["text"] for d in guide_docs)
        assert "Advanced Usage" in all_text


# ===========================================================================
# T18: sync -- already up to date
# ===========================================================================


class TestT18AlreadyUpToDate:
    """Calling sync() a second time without any new commits should be a
    no-op: no error, cursor unchanged, doc count unchanged.
    """

    def _setup(self, test_repo: Path):
        ms, store = _make_minsync(test_repo)
        ms.init()
        ms.sync()
        return ms, store

    # -- T18-1 ---------------------------------------------------------------
    def test_t18_1_sync_succeeds(self, test_repo: Path):
        ms, _store = self._setup(test_repo)
        ms.sync()  # second sync -- should not raise

    # -- T18-2 ---------------------------------------------------------------
    def test_t18_2_result_indicates_no_work(self, test_repo: Path):
        ms, _store = self._setup(test_repo)
        result = ms.sync()

        # The result should indicate that no files were processed, or
        # provide some "already up to date" indication.  We check both
        # a possible boolean flag and a numeric field.
        is_noop = getattr(result, "already_up_to_date", False) or getattr(result, "files_processed", -1) == 0
        assert is_noop, f"Expected sync result to indicate no work done, got: {result}"

    # -- T18-3 ---------------------------------------------------------------
    def test_t18_3_cursor_unchanged(self, test_repo: Path):
        ms, _store = self._setup(test_repo)

        cursor_before = get_cursor(test_repo)
        ms.sync()
        cursor_after = get_cursor(test_repo)

        assert cursor_before["last_synced_commit"] == cursor_after["last_synced_commit"]

    # -- T18-4 ---------------------------------------------------------------
    def test_t18_4_doc_count_unchanged(self, test_repo: Path):
        ms, store = self._setup(test_repo)

        count_before = store.doc_count()
        ms.sync()
        count_after = store.doc_count()

        assert count_before == count_after


# ===========================================================================
# T19: sync -- not initialized
# ===========================================================================


class TestT19SyncNotInitialized:
    """Calling sync() on a repo where init() has NOT been called should
    raise an exception whose message contains "not initialized".
    """

    # -- T19-1 ---------------------------------------------------------------
    def test_t19_1_raises_exception(self, test_repo: Path):
        ms, _store = _make_minsync(test_repo)
        with pytest.raises(Exception):  # noqa: B017
            ms.sync()

    # -- T19-2 ---------------------------------------------------------------
    def test_t19_2_error_contains_not_initialized(self, test_repo: Path):
        ms, _store = _make_minsync(test_repo)
        with pytest.raises(Exception, match=r"(?i)not initialized"):
            ms.sync()
