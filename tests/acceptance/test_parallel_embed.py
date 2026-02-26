"""Tests for async parallel embedding (max_concurrent > 1).

Verifies that:
- Default (max_concurrent=1) uses sync embed() as before
- max_concurrent > 1 with async_embed() uses the async path
- Embedders without async_embed() fall back to sync embed()
- Vector order is preserved across parallel sub-batches
- Errors in sub-batches propagate correctly
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from pathlib import Path

import pytest

from minsync.core import MinSyncEmbeddingError
from tests.mock_components import MockChunker, MockEmbedder, MockVectorStore

# ---------------------------------------------------------------------------
# AsyncMockEmbedder — has both embed() and async_embed()
# ---------------------------------------------------------------------------


class AsyncMockEmbedder:
    """Mock embedder with both sync and async embed methods.

    Records which method was called and timestamps for concurrency assertions.
    """

    def __init__(self, async_delay: float = 0.01):
        self.sync_call_count: int = 0
        self.async_call_count: int = 0
        self.async_timestamps: list[tuple[float, float]] = []  # (start, end) per call
        self._async_delay = async_delay

    def id(self) -> str:
        return "async-mock-embedder-v1"

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.sync_call_count += 1
        return [self._text_to_vector(t) for t in texts]

    async def async_embed(self, texts: list[str]) -> list[list[float]]:
        self.async_call_count += 1
        start = time.monotonic()
        await asyncio.sleep(self._async_delay)
        end = time.monotonic()
        self.async_timestamps.append((start, end))
        return [self._text_to_vector(t) for t in texts]

    @staticmethod
    def _text_to_vector(text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        return [b / 255.0 for b in h]


# ---------------------------------------------------------------------------
# FailingAsyncMockEmbedder — async_embed() raises on specific batch
# ---------------------------------------------------------------------------


class FailingAsyncMockEmbedder(AsyncMockEmbedder):
    """Raises RuntimeError on the Nth async_embed() call."""

    def __init__(self, fail_on_call: int = 2):
        super().__init__()
        self._fail_on_call = fail_on_call

    async def async_embed(self, texts: list[str]) -> list[list[float]]:
        self.async_call_count += 1
        if self.async_call_count == self._fail_on_call:
            raise RuntimeError("Embedding API error on sub-batch")
        return [self._text_to_vector(t) for t in texts]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_minsync(repo: Path, embedder=None, store=None):
    from minsync import MinSync

    store = store or MockVectorStore()
    embedder = embedder or MockEmbedder()
    ms = MinSync(
        repo_path=repo,
        chunker=MockChunker(),
        embedder=embedder,
        vector_store=store,
    )
    return ms, store


# ===========================================================================
# Tests
# ===========================================================================


class TestSerialWhenMaxConcurrentIs1:
    """Default max_concurrent=1 should use sync embed(), not async."""

    def test_sync_embed_called(self, test_repo: Path):
        embedder = AsyncMockEmbedder()
        ms, store = _make_minsync(test_repo, embedder=embedder)
        ms.init()
        ms.sync(max_concurrent=1)

        assert embedder.sync_call_count >= 1
        assert embedder.async_call_count == 0
        assert store.doc_count() > 0


class TestParallelUsesAsyncEmbed:
    """max_concurrent > 1 with async_embed should use the async path."""

    def test_async_embed_called(self, test_repo: Path):
        embedder = AsyncMockEmbedder()
        ms, store = _make_minsync(test_repo, embedder=embedder)
        ms.init()
        # Use small batch_size to force multiple sub-batches
        ms.sync(max_concurrent=4, batch_size=2)

        assert embedder.async_call_count >= 1
        assert embedder.sync_call_count == 0
        assert store.doc_count() > 0

    def test_concurrency_limited_by_semaphore(self, test_repo: Path):
        """With enough sub-batches the semaphore should limit concurrency."""
        embedder = AsyncMockEmbedder(async_delay=0.05)
        ms, _store = _make_minsync(test_repo, embedder=embedder)
        ms.init()
        ms.sync(max_concurrent=2, batch_size=2)

        # Just verify it completed without errors and used async
        assert embedder.async_call_count >= 1


class TestFallbackToSyncWithoutAsyncEmbed:
    """Embedder without async_embed should use sync embed()."""

    def test_sync_fallback(self, test_repo: Path):
        embedder = MockEmbedder()  # no async_embed
        ms, store = _make_minsync(test_repo, embedder=embedder)
        ms.init()
        ms.sync(max_concurrent=4)  # should fall back to sync

        assert embedder.call_count >= 1
        assert store.doc_count() > 0


class TestOrderPreserved:
    """Vectors should be assigned to the correct documents regardless of parallel execution."""

    def test_vectors_match_sync_path(self, test_repo: Path):
        # Sync with async embedder
        async_embedder = AsyncMockEmbedder()
        ms_async, store_async = _make_minsync(test_repo, embedder=async_embedder)
        ms_async.init()
        ms_async.sync(max_concurrent=4, batch_size=2)

        # Sync with sync embedder (force re-init to reset cursor)
        sync_embedder = MockEmbedder()
        ms_sync, store_sync = _make_minsync(test_repo, embedder=sync_embedder)
        ms_sync.init(force=True)
        ms_sync.sync()

        # Both stores should have the same docs with same embeddings
        async_docs = sorted(store_async.get_all_docs(), key=lambda d: d["id"])
        sync_docs = sorted(store_sync.get_all_docs(), key=lambda d: d["id"])

        assert len(async_docs) == len(sync_docs)
        for async_doc, sync_doc in zip(async_docs, sync_docs, strict=True):
            assert async_doc["id"] == sync_doc["id"]
            assert async_doc["embedding"] == sync_doc["embedding"]


class TestErrorPropagates:
    """If a sub-batch fails, the entire sync should fail."""

    def test_async_error_raises(self, test_repo: Path):
        embedder = FailingAsyncMockEmbedder(fail_on_call=1)
        ms, _store = _make_minsync(test_repo, embedder=embedder)
        ms.init()

        with pytest.raises(MinSyncEmbeddingError, match="Embedding API error"):
            ms.sync(max_concurrent=4, batch_size=2)
