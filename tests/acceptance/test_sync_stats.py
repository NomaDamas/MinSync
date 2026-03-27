"""Acceptance tests for sync statistics reporting (Issue #8)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from minsync.cli import build_parser
from minsync.core import SyncResult, SyncStats
from tests.conftest import create_test_repo
from tests.mock_components import MockChunker, MockEmbedder, MockVectorStore

if TYPE_CHECKING:
    from minsync import MinSync


class _FakeMinSync:
    def __init__(self, result: SyncResult) -> None:
        self._result = result

    def sync(self, **_: object) -> SyncResult:
        return self._result


class _AsyncCountingEmbedder:
    def __init__(self) -> None:
        self.async_call_count = 0

    def id(self) -> str:
        return "async-counting-embedder-v1"

    async def async_embed(self, texts: list[str]) -> list[list[float]]:
        self.async_call_count += 1
        return [[float(len(text))] for text in texts]


def _make_minsync(repo: Path, *, embedder: object | None = None) -> tuple[MinSync, MockVectorStore]:
    from minsync import MinSync

    store = MockVectorStore()
    ms = MinSync(
        repo_path=repo,
        chunker=MockChunker(),
        embedder=embedder or MockEmbedder(),
        vector_store=store,
    )
    return ms, store


def test_sync_result_includes_stats_for_successful_sync(tmp_path: Path) -> None:
    repo = create_test_repo(tmp_path)
    embedder = MockEmbedder()
    ms, _store = _make_minsync(repo, embedder=embedder)
    ms.init()

    result = ms.sync(batch_size=2)

    assert result.stats.elapsed_seconds >= 0
    assert result.stats.embedding_api_calls == embedder.call_count
    assert result.stats.embedded_texts == embedder.total_texts_embedded
    assert result.stats.estimated_tokens > 0


def test_sync_result_zeroes_stats_when_already_up_to_date(tmp_path: Path) -> None:
    repo = create_test_repo(tmp_path)
    embedder = MockEmbedder()
    ms, _store = _make_minsync(repo, embedder=embedder)
    ms.init()
    ms.sync()

    embedder.call_count = 0
    embedder.total_texts_embedded = 0

    result = ms.sync()

    assert result.already_up_to_date is True
    assert result.stats.elapsed_seconds >= 0
    assert result.stats.embedding_api_calls == 0
    assert result.stats.embedded_texts == 0
    assert result.stats.estimated_tokens == 0


def test_sync_text_output_includes_stats_block(capsys) -> None:
    parser = build_parser()
    args = parser.parse_args(["sync"])
    fake = _FakeMinSync(
        SyncResult(
            from_commit="abc12345",
            to_commit="def67890",
            files_processed=3,
            files_processed_paths=["docs/guide.md"],
            chunks_added=12,
            chunks_updated=3,
            chunks_deleted=1,
            stats=SyncStats(
                elapsed_seconds=1.25,
                embedding_api_calls=2,
                embedded_texts=15,
                estimated_tokens=120,
            ),
        )
    )

    exit_code = args.handler(fake, args)
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Sync Stats" in captured.out
    assert "Embedding API calls: 2" in captured.out
    assert "Estimated tokens:    120" in captured.out


def test_sync_json_output_includes_stats_field(capsys) -> None:
    parser = build_parser()
    args = parser.parse_args(["--format", "json", "sync"])
    fake = _FakeMinSync(
        SyncResult(
            from_commit="abc12345",
            to_commit="def67890",
            files_processed=3,
            stats=SyncStats(
                elapsed_seconds=0.5,
                embedding_api_calls=4,
                embedded_texts=9,
                estimated_tokens=42,
            ),
        )
    )

    exit_code = args.handler(fake, args)
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["stats"]["embedding_api_calls"] == 4
    assert payload["stats"]["embedded_texts"] == 9
    assert payload["stats"]["estimated_tokens"] == 42


def test_parallel_async_sync_tracks_api_call_count(tmp_path: Path) -> None:
    repo = create_test_repo(tmp_path)
    embedder = _AsyncCountingEmbedder()
    ms, _store = _make_minsync(repo, embedder=embedder)
    ms.init()

    result = ms.sync(max_concurrent=4, batch_size=2)

    assert embedder.async_call_count > 0
    assert result.stats.embedding_api_calls == embedder.async_call_count
    assert result.stats.embedded_texts >= result.stats.embedding_api_calls
