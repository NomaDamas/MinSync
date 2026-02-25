"""Tests for improved error messages with file/chunk context (Issue #5).

Verifies that sync and verify operations wrap lower-level exceptions
(embedding, vectorstore, chunker) into MinSyncError subclasses with
contextual information (file paths, chunk counts) and correct exit codes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from minsync import MinSync
from minsync.core import MinSyncEmbeddingError, MinSyncError, MinSyncVectorStoreError
from tests.conftest import create_test_repo
from tests.mock_components import (
    CrashAfterNUpserts,
    FailingMockChunker,
    FailingMockEmbedder,
    MockChunker,
    MockEmbedder,
    MockVectorStore,
)

# ============================================================================
# Embedding error tests
# ============================================================================


class TestEmbeddingErrors:
    """Embedding failures during sync should raise MinSyncEmbeddingError."""

    def test_embedding_error_includes_file_context(self, tmp_path: Path) -> None:
        """Error message must include text count."""
        repo = create_test_repo(tmp_path)
        ms = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=FailingMockEmbedder(),
            vector_store=MockVectorStore(),
        )
        ms.init()

        with pytest.raises(MinSyncEmbeddingError, match=r"embedding failed.*\d+ texts"):
            ms.sync()

    def test_embedding_error_exit_code_is_5(self, tmp_path: Path) -> None:
        """MinSyncEmbeddingError must have exit_code=5."""
        repo = create_test_repo(tmp_path)
        ms = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=FailingMockEmbedder(),
            vector_store=MockVectorStore(),
        )
        ms.init()

        with pytest.raises(MinSyncEmbeddingError) as exc_info:
            ms.sync()

        assert exc_info.value.exit_code == 5


# ============================================================================
# VectorStore error tests
# ============================================================================


class TestVectorStoreErrors:
    """VectorStore failures during sync should raise MinSyncVectorStoreError."""

    def test_vectorstore_upsert_error_includes_file_and_chunk_count(self, tmp_path: Path) -> None:
        """Error message must include file path and chunk count."""
        repo = create_test_repo(tmp_path)
        store = CrashAfterNUpserts(crash_after=0)
        ms = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=store,
        )
        ms.init()

        with pytest.raises(MinSyncVectorStoreError, match=r"upsert failed for .+\(\d+ chunks\)"):
            ms.sync()

    def test_vectorstore_error_exit_code_is_4(self, tmp_path: Path) -> None:
        """MinSyncVectorStoreError must have exit_code=4."""
        repo = create_test_repo(tmp_path)
        store = CrashAfterNUpserts(crash_after=0)
        ms = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=store,
        )
        ms.init()

        with pytest.raises(MinSyncVectorStoreError) as exc_info:
            ms.sync()

        assert exc_info.value.exit_code == 4


# ============================================================================
# Chunker error tests
# ============================================================================


class TestChunkerErrors:
    """Chunker failures during sync should raise MinSyncError with file path."""

    def test_chunker_error_includes_file_path(self, tmp_path: Path) -> None:
        """Error message must include the file path that failed."""
        repo = create_test_repo(tmp_path)
        ms = MinSync(
            repo_path=repo,
            chunker=FailingMockChunker(),
            embedder=MockEmbedder(),
            vector_store=MockVectorStore(),
        )
        ms.init()

        with pytest.raises(MinSyncError, match=r"chunking failed for .+"):
            ms.sync()


# ============================================================================
# Verify --fix error tests
# ============================================================================


class TestVerifyFixErrors:
    """Errors during verify --fix should include file context."""

    def test_verify_fix_embedding_error_context(self, tmp_path: Path) -> None:
        """Embedding error during verify --fix must include file path and chunk count."""
        repo = create_test_repo(tmp_path)
        store = MockVectorStore()
        good_embedder = MockEmbedder()
        ms = MinSync(
            repo_path=repo,
            chunker=MockChunker(),
            embedder=good_embedder,
            vector_store=store,
        )
        ms.init()
        ms.sync()

        # Corrupt: delete some docs so verify --fix needs to re-embed
        guide_docs = store.get_docs_by_path("docs/guide.md")
        assert len(guide_docs) > 0
        store.direct_delete([d["id"] for d in guide_docs[:2]])

        # Now swap embedder to a failing one
        ms.embedder = FailingMockEmbedder()

        with pytest.raises(MinSyncEmbeddingError, match=r"embedding failed during verify --fix"):
            ms.verify(all=True, fix=True)

    def test_verify_fix_vectorstore_error_context(self, tmp_path: Path) -> None:
        """VectorStore error during verify --fix must include file context."""
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

        # Corrupt: delete some docs so verify --fix needs to re-upsert
        guide_docs = store.get_docs_by_path("docs/guide.md")
        assert len(guide_docs) > 0
        store.direct_delete([d["id"] for d in guide_docs[:2]])

        # Monkeypatch upsert to fail
        def failing_upsert(docs: list[dict]) -> None:
            raise RuntimeError("Simulated vectorstore failure")

        store.upsert = failing_upsert  # type: ignore[assignment]

        with pytest.raises(MinSyncVectorStoreError, match=r"upsert failed during verify --fix"):
            ms.verify(all=True, fix=True)
