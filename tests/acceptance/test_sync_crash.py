"""T14, T15, T16, T17: Crash recovery and lock tests for MinSync sync.

TDD tests for crash-safe behavior and concurrent lock enforcement.
The implementation does NOT exist yet; these tests define the expected behavior.

References:
    - ai_instruction/E2E_TEST_PLAN.md  (T14, T15, T16, T17)
    - ai_instruction/CLI_SPEC.md       (section 2: minsync sync)
    - PRD.md                           (sections 7, 10: crash-safe, lock)
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from minsync import MinSync
from minsync.core import MinSyncVectorStoreError
from tests.conftest import (
    add_commit,
    create_test_repo,
    get_cursor,
    get_head,
)
from tests.mock_components import (
    MockChunker,
    MockEmbedder,
    MockVectorStore,
    SlowMockEmbedder,
)

# ============================================================================
# T14: Crash recovery -- sync interrupted mid-process
# ============================================================================


class TestT14CrashMidSync:
    """T14: Sync crashes partway through upserting chunks.

    Scenario:
        1. init + sync with normal store (all SAMPLE_FILES indexed).
        2. Add 5 new files, commit.
        3. Replace store's upsert with crashing variant (crash after 3 upserts).
        4. Call sync() -- expect RuntimeError.
        5. Verify cursor unchanged, txn.json present.
        6. Restore normal upsert, run recovery sync.
        7. Verify cursor updated, txn.json gone, docs match clean sync, verify passes.
    """

    @staticmethod
    def _setup(tmp_path: Path):
        """Set up initial repo, init, sync, then add 5 new files."""
        store = MockVectorStore()
        chunker = MockChunker()
        embedder = MockEmbedder()
        repo = create_test_repo(tmp_path)

        ms = MinSync(
            repo_path=repo,
            chunker=chunker,
            embedder=embedder,
            vector_store=store,
        )
        ms.init()
        ms.sync()

        old_cursor = get_cursor(repo)

        # Add 5 new files and commit
        new_files = {
            f"docs/new{i}.md": f"# New Document {i}\n\n## Section\n\nContent for new file {i}." for i in range(5)
        }
        add_commit(repo, new_files, "add 5 new files")

        return ms, repo, store, old_cursor

    # -- T14-1: cursor.json unchanged after crash -----------------------------
    def test_t14_1_cursor_unchanged_after_crash(self, tmp_path):
        """After a crash mid-sync, cursor.json must remain at the old HEAD."""
        ms, repo, store, old_cursor = self._setup(tmp_path)

        # Monkeypatch upsert to crash after 3 calls
        call_count = 0
        original_upsert = store.upsert

        def crashing_upsert(docs):
            nonlocal call_count
            call_count += 1
            if call_count > 3:
                raise RuntimeError("Simulated crash during upsert")
            return original_upsert(docs)

        store.upsert = crashing_upsert

        with pytest.raises(MinSyncVectorStoreError):
            ms.sync()

        current_cursor = get_cursor(repo)
        assert current_cursor["last_synced_commit"] == old_cursor["last_synced_commit"]

    # -- T14-2: txn.json exists after crash -----------------------------------
    def test_t14_2_txn_json_exists_after_crash(self, tmp_path):
        """After a crash mid-sync, txn.json must exist (incomplete transaction)."""
        ms, repo, store, _old_cursor = self._setup(tmp_path)

        call_count = 0
        original_upsert = store.upsert

        def crashing_upsert(docs):
            nonlocal call_count
            call_count += 1
            if call_count > 3:
                raise RuntimeError("Simulated crash during upsert")
            return original_upsert(docs)

        store.upsert = crashing_upsert

        with pytest.raises(MinSyncVectorStoreError):
            ms.sync()

        txn_path = repo / ".minsync" / "txn.json"
        assert txn_path.exists(), "txn.json must exist after crash"

    # -- T14-3: recovery sync succeeds ----------------------------------------
    def test_t14_3_recovery_sync_succeeds(self, tmp_path):
        """After restoring a healthy store, recovery sync must succeed."""
        ms, _repo, store, _old_cursor = self._setup(tmp_path)

        call_count = 0
        original_upsert = store.upsert

        def crashing_upsert(docs):
            nonlocal call_count
            call_count += 1
            if call_count > 3:
                raise RuntimeError("Simulated crash during upsert")
            return original_upsert(docs)

        store.upsert = crashing_upsert

        with pytest.raises(MinSyncVectorStoreError):
            ms.sync()

        # Restore normal upsert
        store.upsert = original_upsert

        # Recovery sync should not raise
        ms.sync()

    # -- T14-4: cursor updated to latest HEAD after recovery ------------------
    def test_t14_4_cursor_updated_after_recovery(self, tmp_path):
        """After recovery sync, cursor must point to the latest HEAD."""
        ms, repo, store, _old_cursor = self._setup(tmp_path)

        call_count = 0
        original_upsert = store.upsert

        def crashing_upsert(docs):
            nonlocal call_count
            call_count += 1
            if call_count > 3:
                raise RuntimeError("Simulated crash during upsert")
            return original_upsert(docs)

        store.upsert = crashing_upsert

        with pytest.raises(MinSyncVectorStoreError):
            ms.sync()

        store.upsert = original_upsert
        ms.sync()

        latest_head = get_head(repo)
        cursor = get_cursor(repo)
        assert cursor["last_synced_commit"] == latest_head

    # -- T14-5: txn.json deleted after recovery --------------------------------
    def test_t14_5_txn_json_deleted_after_recovery(self, tmp_path):
        """After successful recovery sync, txn.json must be removed."""
        ms, repo, store, _old_cursor = self._setup(tmp_path)

        call_count = 0
        original_upsert = store.upsert

        def crashing_upsert(docs):
            nonlocal call_count
            call_count += 1
            if call_count > 3:
                raise RuntimeError("Simulated crash during upsert")
            return original_upsert(docs)

        store.upsert = crashing_upsert

        with pytest.raises(MinSyncVectorStoreError):
            ms.sync()

        store.upsert = original_upsert
        ms.sync()

        txn_path = repo / ".minsync" / "txn.json"
        assert not txn_path.exists(), "txn.json must be deleted after recovery"

    # -- T14-6: doc_ids match clean sync --------------------------------------
    def test_t14_6_doc_ids_match_clean_sync(self, tmp_path):
        """After crash + recovery, the doc_id set must match a clean sync."""
        ms, repo, store, _old_cursor = self._setup(tmp_path)

        call_count = 0
        original_upsert = store.upsert

        def crashing_upsert(docs):
            nonlocal call_count
            call_count += 1
            if call_count > 3:
                raise RuntimeError("Simulated crash during upsert")
            return original_upsert(docs)

        store.upsert = crashing_upsert

        with pytest.raises(MinSyncVectorStoreError):
            ms.sync()

        store.upsert = original_upsert
        ms.sync()

        recovered_ids = store.get_all_doc_ids()

        # Now build a clean reference: fresh repo with same state
        clean_store = MockVectorStore()
        clean_ms = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=clean_store,
        )
        clean_ms.init(force=True)
        clean_ms.sync()
        clean_ids = clean_store.get_all_doc_ids()

        assert recovered_ids == clean_ids, "Recovered doc_ids should match a clean full sync"

    # -- T14-7: verify passes after recovery -----------------------------------
    def test_t14_7_verify_passes_after_recovery(self, tmp_path):
        """After crash + recovery, ms.verify() must pass with all_passed."""
        ms, _repo, store, _old_cursor = self._setup(tmp_path)

        call_count = 0
        original_upsert = store.upsert

        def crashing_upsert(docs):
            nonlocal call_count
            call_count += 1
            if call_count > 3:
                raise RuntimeError("Simulated crash during upsert")
            return original_upsert(docs)

        store.upsert = crashing_upsert

        with pytest.raises(MinSyncVectorStoreError):
            ms.sync()

        store.upsert = original_upsert
        ms.sync()

        report = ms.verify(all=True)
        assert report.all_passed, "verify should pass after crash recovery"


# ============================================================================
# T15: Crash recovery -- flush interrupted
# ============================================================================


class TestT15CrashOnFlush:
    """T15: Sync crashes when flush() is called.

    Scenario:
        1. init + sync with normal store.
        2. Add files, commit.
        3. Monkeypatch flush to raise on first call.
        4. First sync attempt crashes on flush.
        5. Restore normal flush, run recovery sync.
        6. Verify all chunks present and no stale chunks.
    """

    @staticmethod
    def _setup(tmp_path: Path):
        """Set up initial repo, init, sync, then add new files."""
        store = MockVectorStore()
        chunker = MockChunker()
        embedder = MockEmbedder()
        repo = create_test_repo(tmp_path)

        ms = MinSync(
            repo_path=repo,
            chunker=chunker,
            embedder=embedder,
            vector_store=store,
        )
        ms.init()
        ms.sync()

        # Add new files
        new_files = {f"docs/extra{i}.md": f"# Extra {i}\n\n## Info\n\nExtra content {i}." for i in range(3)}
        add_commit(repo, new_files, "add extra files")

        return ms, repo, store

    # -- T15-1: All chunks present after recovery ------------------------------
    def test_t15_1_all_chunks_present_after_recovery(self, tmp_path):
        """After crash on flush + recovery, all expected chunks must exist."""
        ms, repo, store = self._setup(tmp_path)

        # Monkeypatch flush to crash on first call
        flush_count = 0
        original_flush = store.flush

        def crashing_flush():
            nonlocal flush_count
            flush_count += 1
            if flush_count == 1:
                raise RuntimeError("Simulated crash during flush")
            return original_flush()

        store.flush = crashing_flush

        with pytest.raises(MinSyncVectorStoreError):
            ms.sync()

        # Restore normal flush
        store.flush = original_flush

        # Recovery sync
        ms.sync()

        # Build a clean reference
        clean_store = MockVectorStore()
        clean_ms = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=clean_store,
        )
        clean_ms.init(force=True)
        clean_ms.sync()

        recovered_ids = store.get_all_doc_ids()
        clean_ids = clean_store.get_all_doc_ids()

        assert recovered_ids == clean_ids, "All expected chunks must be present"

    # -- T15-2: No stale chunks after recovery ---------------------------------
    def test_t15_2_no_stale_chunks_after_recovery(self, tmp_path):
        """After crash on flush + recovery, no stale chunks should remain."""
        ms, _repo, store = self._setup(tmp_path)

        flush_count = 0
        original_flush = store.flush

        def crashing_flush():
            nonlocal flush_count
            flush_count += 1
            if flush_count == 1:
                raise RuntimeError("Simulated crash during flush")
            return original_flush()

        store.flush = crashing_flush

        with pytest.raises(MinSyncVectorStoreError):
            ms.sync()

        store.flush = original_flush
        ms.sync()

        # Verify should pass -- no stale chunks
        report = ms.verify(all=True)
        assert report.all_passed, "No stale chunks should remain after recovery"


# ============================================================================
# T16: Crash recovery -- cursor update interrupted
# ============================================================================


class TestT16CrashOnCursorUpdate:
    """T16: Sync completes all processing and flush, but crashes during cursor write.

    Scenario:
        1. init + sync with normal store.
        2. Add files, commit.
        3. Monkeypatch to crash during cursor.json write (atomic rename).
        4. First sync attempt crashes during cursor update.
        5. Recovery sync succeeds.
        6. Final doc_ids match clean sync.

    We monkeypatch ``os.rename`` to fail when the destination is cursor.json,
    simulating a crash after flush but before the cursor is updated.
    """

    @staticmethod
    def _setup(tmp_path: Path):
        """Set up initial repo, init, sync, then add new files."""
        store = MockVectorStore()
        chunker = MockChunker()
        embedder = MockEmbedder()
        repo = create_test_repo(tmp_path)

        ms = MinSync(
            repo_path=repo,
            chunker=chunker,
            embedder=embedder,
            vector_store=store,
        )
        ms.init()
        ms.sync()

        old_cursor = get_cursor(repo)

        # Add new files
        new_files = {f"docs/cursor_test{i}.md": f"# Cursor Test {i}\n\n## Part\n\nContent {i}." for i in range(3)}
        add_commit(repo, new_files, "add cursor test files")

        return ms, repo, store, old_cursor

    # -- T16-1: cursor remains at old state after crash -----------------------
    def test_t16_1_cursor_unchanged_after_cursor_write_crash(self, tmp_path):
        """If cursor.json write fails, cursor must remain at old state."""
        ms, repo, _store, old_cursor = self._setup(tmp_path)

        original_rename = os.rename
        rename_crash_count = 0

        def failing_rename(src, dst):
            nonlocal rename_crash_count
            # Intercept rename targeting cursor.json
            if "cursor" in str(dst):
                rename_crash_count += 1
                if rename_crash_count == 1:
                    raise OSError("Simulated crash during cursor update")
            return original_rename(src, dst)

        with (
            patch("os.rename", side_effect=failing_rename),
            pytest.raises(OSError, match="Simulated crash during cursor update"),
        ):
            ms.sync()

        cursor_after_crash = get_cursor(repo)
        assert cursor_after_crash["last_synced_commit"] == old_cursor["last_synced_commit"], (
            "cursor.json must remain unchanged when cursor write is interrupted"
        )

    # -- T16-2: recovery sync succeeds ----------------------------------------
    def test_t16_2_recovery_sync_succeeds(self, tmp_path):
        """Recovery sync after cursor write crash must succeed."""
        ms, repo, _store, _old_cursor = self._setup(tmp_path)

        original_rename = os.rename
        rename_crash_count = 0

        def failing_rename(src, dst):
            nonlocal rename_crash_count
            if "cursor" in str(dst):
                rename_crash_count += 1
                if rename_crash_count == 1:
                    raise OSError("Simulated crash during cursor update")
            return original_rename(src, dst)

        with (
            patch("os.rename", side_effect=failing_rename),
            pytest.raises(OSError, match="Simulated crash during cursor update"),
        ):
            ms.sync()

        # Recovery sync (no monkey-patch active) should succeed
        ms.sync()

        latest_head = get_head(repo)
        cursor = get_cursor(repo)
        assert cursor["last_synced_commit"] == latest_head

    # -- T16-3: final doc_ids match clean sync --------------------------------
    def test_t16_3_doc_ids_match_clean_sync(self, tmp_path):
        """After crash on cursor write + recovery, doc_ids must match clean sync."""
        ms, repo, store, _old_cursor = self._setup(tmp_path)

        original_rename = os.rename
        rename_crash_count = 0

        def failing_rename(src, dst):
            nonlocal rename_crash_count
            if "cursor" in str(dst):
                rename_crash_count += 1
                if rename_crash_count == 1:
                    raise OSError("Simulated crash during cursor update")
            return original_rename(src, dst)

        with (
            patch("os.rename", side_effect=failing_rename),
            pytest.raises(OSError, match="Simulated crash during cursor update"),
        ):
            ms.sync()

        # Recovery
        ms.sync()

        recovered_ids = store.get_all_doc_ids()

        # Clean reference
        clean_store = MockVectorStore()
        clean_ms = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=clean_store,
        )
        clean_ms.init(force=True)
        clean_ms.sync()
        clean_ids = clean_store.get_all_doc_ids()

        assert recovered_ids == clean_ids, "Doc IDs after cursor-crash recovery should match a clean full sync"


# ============================================================================
# T17: Lock -- concurrent sync prevention
# ============================================================================


class TestT17ConcurrentLock:
    """T17: Two concurrent sync attempts; one must succeed, the other must fail with lock error.

    Scenario:
        1. Set up repo with SlowMockEmbedder (delay=1.0s) so sync takes time.
        2. Launch two threads both calling ms.sync().
        3. Exactly one thread succeeds; the other gets a lock error.
        4. After both finish, the lock file does not exist.
    """

    # -- T17-1 through T17-4: full concurrent test ----------------------------
    def test_t17_concurrent_lock(self, tmp_path):
        """Two concurrent syncs: one succeeds, one fails with lock error."""
        store = MockVectorStore()
        embedder = SlowMockEmbedder(delay=0.5)
        chunker = MockChunker()
        repo = create_test_repo(tmp_path)

        ms = MinSync(
            repo_path=repo,
            chunker=chunker,
            embedder=embedder,
            vector_store=store,
        )
        ms.init()

        results: list[object | None] = [None, None]
        exceptions: list[Exception | None] = [None, None]

        def run_sync(idx: int) -> None:
            try:
                results[idx] = ms.sync()
            except Exception as e:
                exceptions[idx] = e

        t1 = threading.Thread(target=run_sync, args=(0,))
        t2 = threading.Thread(target=run_sync, args=(1,))

        t1.start()
        # Small delay to create a race condition where both contend for the lock
        time.sleep(0.05)
        t2.start()

        t1.join(timeout=30)
        t2.join(timeout=30)

        # Determine which succeeded and which failed
        successes = [i for i in range(2) if exceptions[i] is None]
        failures = [i for i in range(2) if exceptions[i] is not None]

        # T17-1: Exactly one succeeds
        assert len(successes) == 1, (
            f"Exactly one sync should succeed, got {len(successes)} successes. Exceptions: {exceptions}"
        )

        # T17-2: The other gets a lock error
        assert len(failures) == 1, f"Exactly one sync should fail with lock error, got {len(failures)} failures"

        # T17-3: Lock error message contains "another sync is in progress"
        lock_error = exceptions[failures[0]]
        assert lock_error is not None
        error_msg = str(lock_error).lower()
        assert "another sync is in progress" in error_msg or "lock" in error_msg, (
            f"Lock error message should mention concurrent sync, got: {lock_error}"
        )

        # T17-4: After both finish, lock file does not exist
        lock_path = repo / ".minsync" / "lock"
        assert not lock_path.exists(), "Lock file must not exist after both sync attempts have finished"

    # -- T17-1 (explicit): one sync succeeds ----------------------------------
    def test_t17_1_one_sync_succeeds(self, tmp_path):
        """At least one of two concurrent syncs must succeed."""
        store = MockVectorStore()
        embedder = SlowMockEmbedder(delay=0.5)
        chunker = MockChunker()
        repo = create_test_repo(tmp_path)

        ms = MinSync(
            repo_path=repo,
            chunker=chunker,
            embedder=embedder,
            vector_store=store,
        )
        ms.init()

        results: list[object | None] = [None, None]
        exceptions: list[Exception | None] = [None, None]

        def run_sync(idx: int) -> None:
            try:
                results[idx] = ms.sync()
            except Exception as e:
                exceptions[idx] = e

        t1 = threading.Thread(target=run_sync, args=(0,))
        t2 = threading.Thread(target=run_sync, args=(1,))
        t1.start()
        time.sleep(0.05)
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)

        successes = sum(1 for e in exceptions if e is None)
        assert successes >= 1, "At least one sync must succeed"

    # -- T17-2 (explicit): other gets lock error ------------------------------
    def test_t17_2_other_gets_lock_error(self, tmp_path):
        """One of two concurrent syncs must fail with a lock-related error."""
        store = MockVectorStore()
        embedder = SlowMockEmbedder(delay=0.5)
        chunker = MockChunker()
        repo = create_test_repo(tmp_path)

        ms = MinSync(
            repo_path=repo,
            chunker=chunker,
            embedder=embedder,
            vector_store=store,
        )
        ms.init()

        exceptions: list[Exception | None] = [None, None]

        def run_sync(idx: int) -> None:
            try:
                ms.sync()
            except Exception as e:
                exceptions[idx] = e

        t1 = threading.Thread(target=run_sync, args=(0,))
        t2 = threading.Thread(target=run_sync, args=(1,))
        t1.start()
        time.sleep(0.05)
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)

        failures = [e for e in exceptions if e is not None]
        assert len(failures) >= 1, "One sync must fail with a lock error"

    # -- T17-3 (explicit): lock error message ---------------------------------
    def test_t17_3_lock_error_message(self, tmp_path):
        """The lock error message must mention concurrent sync."""
        store = MockVectorStore()
        embedder = SlowMockEmbedder(delay=0.5)
        chunker = MockChunker()
        repo = create_test_repo(tmp_path)

        ms = MinSync(
            repo_path=repo,
            chunker=chunker,
            embedder=embedder,
            vector_store=store,
        )
        ms.init()

        exceptions: list[Exception | None] = [None, None]

        def run_sync(idx: int) -> None:
            try:
                ms.sync()
            except Exception as e:
                exceptions[idx] = e

        t1 = threading.Thread(target=run_sync, args=(0,))
        t2 = threading.Thread(target=run_sync, args=(1,))
        t1.start()
        time.sleep(0.05)
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)

        failures = [e for e in exceptions if e is not None]
        assert len(failures) >= 1, "Expected at least one lock failure"

        lock_error = failures[0]
        error_msg = str(lock_error).lower()
        assert "another sync is in progress" in error_msg or "lock" in error_msg, (
            f"Lock error message should indicate a concurrent sync conflict, got: {lock_error}"
        )

    # -- T17-4 (explicit): lock file cleaned up after completion ---------------
    def test_t17_4_lock_file_cleaned_up(self, tmp_path):
        """After all concurrent syncs complete, the lock file must not exist."""
        store = MockVectorStore()
        embedder = SlowMockEmbedder(delay=0.5)
        chunker = MockChunker()
        repo = create_test_repo(tmp_path)

        ms = MinSync(
            repo_path=repo,
            chunker=chunker,
            embedder=embedder,
            vector_store=store,
        )
        ms.init()

        exceptions: list[Exception | None] = [None, None]

        def run_sync(idx: int) -> None:
            try:
                ms.sync()
            except Exception as e:
                exceptions[idx] = e

        t1 = threading.Thread(target=run_sync, args=(0,))
        t2 = threading.Thread(target=run_sync, args=(1,))
        t1.start()
        time.sleep(0.05)
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)

        lock_path = repo / ".minsync" / "lock"
        assert not lock_path.exists(), "Lock file must be cleaned up after all sync operations finish"

    # -- Additional lock-safety regression: stale lock reclamation ------------
    def test_t17_stale_lock_is_reclaimed(self, tmp_path):
        """A stale lock artifact should be reclaimed so recovery sync can proceed."""
        store = MockVectorStore()
        repo = create_test_repo(tmp_path)
        ms = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=store,
        )
        ms.init()

        lock_path = repo / ".minsync" / "lock"
        lock_path.write_text("invalid-pid\n", encoding="utf-8")

        result = ms.sync()
        assert result.files_processed > 0
        assert not lock_path.exists(), "Stale lock should be removed during acquisition"
