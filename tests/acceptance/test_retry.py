"""Tests for auto-retry on transient embedding API errors (Issue #6).

Covers: _is_transient_error(), _embed_with_retry(), _async_embed_with_retry(),
sync retry, query retry, verify --fix retry, and --max-retries CLI flag.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from minsync import MinSync
from minsync.core import (
    MinSyncEmbeddingError,
    _async_embed_with_retry,
    _embed_with_retry,
    _is_transient_error,
    _SyncStatsTracker,
)
from tests.conftest import create_test_repo
from tests.mock_components import (
    MockChunker,
    MockEmbedder,
    MockVectorStore,
    PermanentFailEmbedder,
    TransientFailAsyncEmbedder,
    TransientFailEmbedder,
    _HttpLikeError,
)

# ---------------------------------------------------------------------------
# Module-level autouse fixture: monkeypatch sleep for instant retries
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _instant_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda _: None)

    async def _fast_sleep(delay: float, **kwargs):
        pass

    monkeypatch.setattr(asyncio, "sleep", _fast_sleep)


# ===========================================================================
# TestIsTransientError
# ===========================================================================


class TestIsTransientError:
    def test_connection_error_is_transient(self) -> None:
        assert _is_transient_error(ConnectionError("connection refused")) is True

    def test_timeout_error_is_transient(self) -> None:
        assert _is_transient_error(TimeoutError("timed out")) is True

    def test_os_error_is_transient(self) -> None:
        assert _is_transient_error(OSError("network unreachable")) is True

    def test_429_message_is_transient(self) -> None:
        assert _is_transient_error(Exception("rate limit exceeded (429)")) is True

    def test_503_message_is_transient(self) -> None:
        assert _is_transient_error(Exception("service unavailable 503")) is True

    def test_status_code_429_is_transient(self) -> None:
        exc = _HttpLikeError("too many requests", status_code=429)
        assert _is_transient_error(exc) is True

    def test_status_code_500_is_transient(self) -> None:
        exc = _HttpLikeError("internal server error", status_code=500)
        assert _is_transient_error(exc) is True

    def test_status_code_401_is_permanent(self) -> None:
        exc = _HttpLikeError("unauthorized", status_code=401)
        assert _is_transient_error(exc) is False

    def test_status_code_400_is_permanent(self) -> None:
        exc = _HttpLikeError("bad request", status_code=400)
        assert _is_transient_error(exc) is False

    def test_invalid_api_key_is_permanent(self) -> None:
        assert _is_transient_error(Exception("invalid api key")) is False

    def test_unknown_error_treated_as_transient(self) -> None:
        assert _is_transient_error(Exception("something unexpected")) is True


# ===========================================================================
# TestEmbedWithRetry
# ===========================================================================


class TestEmbedWithRetry:
    def test_success_on_first_try(self) -> None:
        embedder = MockEmbedder()
        result = _embed_with_retry(embedder.embed, ["hello"], max_retries=3)
        assert len(result) == 1
        assert embedder.call_count == 1

    def test_transient_failure_then_success(self) -> None:
        embedder = TransientFailEmbedder(fail_count=2)
        result = _embed_with_retry(embedder.embed, ["hello"], max_retries=3)
        assert len(result) == 1
        assert embedder.call_count == 3  # 2 failures + 1 success

    def test_retries_exhausted_raises(self) -> None:
        embedder = TransientFailEmbedder(fail_count=10)
        with pytest.raises(_HttpLikeError, match="rate limit"):
            _embed_with_retry(embedder.embed, ["hello"], max_retries=2)
        assert embedder.call_count == 3  # 1 initial + 2 retries

    def test_permanent_error_immediate_raise(self) -> None:
        embedder = PermanentFailEmbedder()
        with pytest.raises(_HttpLikeError, match="invalid api key"):
            _embed_with_retry(embedder.embed, ["hello"], max_retries=3)
        assert embedder.call_count == 1

    def test_max_retries_zero_no_retry(self) -> None:
        embedder = TransientFailEmbedder(fail_count=1)
        with pytest.raises(_HttpLikeError, match="rate limit"):
            _embed_with_retry(embedder.embed, ["hello"], max_retries=0)
        assert embedder.call_count == 1

    def test_stderr_progress_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        embedder = TransientFailEmbedder(fail_count=1)
        _embed_with_retry(embedder.embed, ["hello"], max_retries=3, quiet=False)
        captured = capsys.readouterr()
        assert "attempt 1/4" in captured.err
        assert "retrying in" in captured.err

    def test_quiet_suppresses_stderr(self, capsys: pytest.CaptureFixture[str]) -> None:
        embedder = TransientFailEmbedder(fail_count=1)
        _embed_with_retry(embedder.embed, ["hello"], max_retries=3, quiet=True)
        captured = capsys.readouterr()
        assert "retrying" not in captured.err

    def test_stats_tracker_counts_only_successful_retry(self) -> None:
        embedder = TransientFailEmbedder(fail_count=2)
        tracker = _SyncStatsTracker(embedder_id=embedder.id())

        result = _embed_with_retry(
            embedder.embed,
            ["hello"],
            max_retries=3,
            quiet=True,
            stats_tracker=tracker,
        )

        assert len(result) == 1
        assert embedder.call_count == 3
        assert tracker.embedding_api_calls == 1
        assert tracker.embedded_texts == embedder.total_texts_embedded
        assert tracker.embedded_texts == 1
        assert tracker.estimated_tokens > 0


# ===========================================================================
# TestAsyncRetry
# ===========================================================================


class TestAsyncRetry:
    def test_async_transient_retry_success(self) -> None:
        embedder = TransientFailAsyncEmbedder(fail_count=2)
        result = asyncio.run(_async_embed_with_retry(embedder.async_embed, ["hello"], max_retries=3))
        assert len(result) == 1

    def test_async_retries_exhausted(self) -> None:
        embedder = TransientFailAsyncEmbedder(fail_count=10)
        with pytest.raises(_HttpLikeError, match="service unavailable"):
            asyncio.run(_async_embed_with_retry(embedder.async_embed, ["hello"], max_retries=2))

    def test_async_stats_tracker_counts_only_successful_retry(self) -> None:
        embedder = TransientFailAsyncEmbedder(fail_count=2)
        tracker = _SyncStatsTracker(embedder_id=embedder.id())

        result = asyncio.run(
            _async_embed_with_retry(
                embedder.async_embed,
                ["hello"],
                max_retries=3,
                quiet=True,
                stats_tracker=tracker,
            )
        )

        assert len(result) == 1
        assert tracker.embedding_api_calls == 1
        assert tracker.embedded_texts == embedder.total_texts_embedded
        assert tracker.embedded_texts == 1
        assert tracker.estimated_tokens > 0


# ===========================================================================
# TestSyncRetry
# ===========================================================================


def _make_minsync(repo: Path, embedder=None):
    store = MockVectorStore()
    ms = MinSync(
        repo_path=repo,
        chunker=MockChunker(),
        embedder=embedder or MockEmbedder(),
        vector_store=store,
    )
    return ms, store


class TestSyncRetry:
    def test_sync_transient_then_success(self, tmp_path: Path) -> None:
        repo = create_test_repo(tmp_path)
        embedder = TransientFailEmbedder(fail_count=2)
        ms, _store = _make_minsync(repo, embedder)
        ms.init()
        result = ms.sync()
        assert result.chunks_added > 0
        assert embedder.call_count >= 3  # at least 2 failures + 1 success

    def test_sync_max_retries_zero_fails(self, tmp_path: Path) -> None:
        repo = create_test_repo(tmp_path)
        embedder = TransientFailEmbedder(fail_count=1)
        ms, _store = _make_minsync(repo, embedder)
        ms.init()
        with pytest.raises(MinSyncEmbeddingError):
            ms.sync(max_retries=0)

    def test_sync_permanent_error_no_retry(self, tmp_path: Path) -> None:
        repo = create_test_repo(tmp_path)
        embedder = PermanentFailEmbedder()
        ms, _store = _make_minsync(repo, embedder)
        ms.init()
        with pytest.raises(MinSyncEmbeddingError):
            ms.sync()
        assert embedder.call_count == 1

    def test_sync_uses_config_max_retries(self, tmp_path: Path) -> None:
        """sync() reads max_retries from config when not passed explicitly."""
        repo = create_test_repo(tmp_path)
        embedder = TransientFailEmbedder(fail_count=2)
        ms, _store = _make_minsync(repo, embedder)
        ms.init()
        # Default config max_retries=3, so 2 failures should be OK
        result = ms.sync()
        assert result.chunks_added > 0

    def test_sync_stats_ignore_failed_retry_attempts(self, tmp_path: Path) -> None:
        repo = create_test_repo(tmp_path)
        embedder = TransientFailEmbedder(fail_count=2)
        ms, _store = _make_minsync(repo, embedder)
        ms.init()

        result = ms.sync(batch_size=2, quiet=True)

        assert result.chunks_added > 0
        assert embedder.call_count == result.stats.embedding_api_calls + 2
        assert result.stats.embedded_texts == embedder.total_texts_embedded
        assert result.stats.estimated_tokens > 0


# ===========================================================================
# TestQueryRetry
# ===========================================================================


class TestQueryRetry:
    def test_query_transient_retry_success(self, tmp_path: Path) -> None:
        repo = create_test_repo(tmp_path)
        good_embedder = MockEmbedder()
        ms, _store = _make_minsync(repo, good_embedder)
        ms.init()
        ms.sync()

        # Now swap to a transient-fail embedder for the query
        transient = TransientFailEmbedder(fail_count=2)
        ms.embedder = transient
        results = ms.query("hello")
        # Should succeed after retries
        assert isinstance(results, list)
        assert transient.call_count >= 3


# ===========================================================================
# TestVerifyFixRetry
# ===========================================================================


class TestVerifyFixRetry:
    def test_verify_fix_transient_retry_success(self, tmp_path: Path) -> None:
        repo = create_test_repo(tmp_path)
        good_embedder = MockEmbedder()
        ms, store = _make_minsync(repo, good_embedder)
        ms.init()
        ms.sync()

        # Corrupt: delete some docs so verify --fix needs to re-embed
        guide_docs = store.get_docs_by_path("docs/guide.md")
        assert len(guide_docs) > 0
        store.direct_delete([d["id"] for d in guide_docs[:2]])

        # Swap to transient fail embedder
        transient = TransientFailEmbedder(fail_count=2)
        ms.embedder = transient
        result = ms.verify(all=True, fix=True)
        assert result.fixed is True
        assert transient.call_count >= 3


# ===========================================================================
# TestCLIMaxRetries
# ===========================================================================


class TestCLIMaxRetries:
    def test_max_retries_flag_parsed(self) -> None:
        from minsync.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["sync", "--max-retries", "5"])
        assert args.max_retries == 5

    def test_max_retries_default_is_none(self) -> None:
        from minsync.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["sync"])
        assert args.max_retries is None
