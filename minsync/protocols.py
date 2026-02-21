"""Protocol classes and shared data types for MinSync components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class Chunk:
    """A single chunk produced by a chunker."""

    chunk_type: str  # "parent" or "child"
    text: str
    heading_path: str = ""


@runtime_checkable
class Chunker(Protocol):
    """Protocol for chunker implementations."""

    def schema_id(self) -> str: ...

    def chunk(self, text: str, path: str) -> list[Chunk]: ...


@runtime_checkable
class Embedder(Protocol):
    """Protocol for embedder implementations."""

    def id(self) -> str: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class VectorStore(Protocol):
    """Protocol for vector store implementations."""

    def upsert(self, docs: list[dict[str, Any]]) -> None: ...

    def update(self, docs: list[dict[str, Any]]) -> None: ...

    def fetch(self, ids: list[str]) -> list[dict[str, Any]]: ...

    def delete_by_filter(self, filter_expr: str) -> int: ...

    def query(
        self,
        vector: list[float],
        filter_expr: str | None = None,
        topk: int = 10,
    ) -> list[dict[str, Any]]: ...

    def flush(self) -> None: ...
