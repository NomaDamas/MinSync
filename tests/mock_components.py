"""Mock components for MinSync E2E tests.

Provides MockChunker, MockEmbedder, MockVectorStore and failure variants
that enable testing without any external dependencies.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Data classes matching expected MinSync chunk/query result shapes
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    chunk_type: str  # "parent" or "child"
    text: str
    heading_path: str = ""


@dataclass
class QueryResult:
    doc_id: str
    path: str
    heading_path: str
    chunk_type: str
    text: str
    score: float
    content_commit: str = ""


# ---------------------------------------------------------------------------
# MockChunker
# ---------------------------------------------------------------------------


class MockChunker:
    """Markdown heading-based chunker for tests.

    Splits on ``#``, ``##``, ``###`` headings.
    Each heading line becomes a *parent* chunk; the body text beneath it
    becomes a *child* chunk.  Files without any heading are returned as a
    single parent chunk.
    """

    def schema_id(self) -> str:
        return "mock-chunker-v1"

    def chunk(self, text: str, path: str) -> list[Chunk]:
        lines = text.split("\n")
        chunks: list[Chunk] = []
        heading_stack: list[str] = []
        current_body_lines: list[str] = []
        current_heading_line: str | None = None

        def _flush_section():
            if current_heading_line is not None:
                # parent chunk = heading line itself
                hp = " > ".join(heading_stack)
                chunks.append(Chunk(chunk_type="parent", text=current_heading_line, heading_path=hp))
                body = "\n".join(current_body_lines).strip()
                if body:
                    chunks.append(Chunk(chunk_type="child", text=body, heading_path=hp))

        heading_re = re.compile(r"^(#{1,3})\s+(.*)")

        for line in lines:
            m = heading_re.match(line)
            if m:
                _flush_section()
                level = len(m.group(1))
                title = m.group(2).strip()
                # adjust heading stack
                while len(heading_stack) >= level:
                    heading_stack.pop()
                heading_stack.append(title)
                current_heading_line = line.strip()
                current_body_lines = []
            else:
                current_body_lines.append(line)

        # flush last section
        _flush_section()

        # no headings at all → single parent chunk with entire text
        if not chunks:
            stripped = text.strip()
            if stripped:
                chunks.append(Chunk(chunk_type="parent", text=stripped, heading_path=""))

        return chunks


# ---------------------------------------------------------------------------
# MockEmbedder
# ---------------------------------------------------------------------------


class MockEmbedder:
    """Deterministic embedder: SHA-256 of text → 32-dim float vector."""

    def __init__(self):
        self.call_count: int = 0
        self.total_texts_embedded: int = 0

    def id(self) -> str:
        return "mock-embedder-v1"

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        self.total_texts_embedded += len(texts)
        return [self._text_to_vector(t) for t in texts]

    @staticmethod
    def _text_to_vector(text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        return [b / 255.0 for b in h]  # 32-dim


# ---------------------------------------------------------------------------
# SlowMockEmbedder (for T17 lock tests)
# ---------------------------------------------------------------------------


class SlowMockEmbedder(MockEmbedder):
    """Embedder that sleeps to simulate slow processing (for lock tests)."""

    def __init__(self, delay: float = 0.5):
        super().__init__()
        self.delay = delay

    def embed(self, texts: list[str]) -> list[list[float]]:
        time.sleep(self.delay)
        return super().embed(texts)


# ---------------------------------------------------------------------------
# MockVectorStore
# ---------------------------------------------------------------------------


class MockVectorStore:
    """In-memory vector store with filter support.

    Internal storage: ``dict[str, dict]`` mapping doc *id* → document dict.
    Each document dict has at minimum: ``id``, ``embedding``, ``text``, plus
    arbitrary metadata fields (``repo_id``, ``ref``, ``path``, ``seen_token``, …).
    """

    def __init__(self):
        self._docs: dict[str, dict] = {}

    # ---- core interface ---------------------------------------------------

    def upsert(self, docs: list[dict]) -> None:
        for doc in docs:
            self._docs[doc["id"]] = dict(doc)

    def update(self, docs: list[dict]) -> None:
        """Partial update – only overwrites provided keys, keeps the rest."""
        for doc in docs:
            did = doc["id"]
            if did in self._docs:
                self._docs[did].update(doc)

    def fetch(self, ids: list[str]) -> list[dict]:
        return [dict(self._docs[i]) for i in ids if i in self._docs]

    def delete_by_filter(self, filter_expr: str) -> int:
        to_delete = [did for did, doc in self._docs.items() if self._match(doc, filter_expr)]
        for did in to_delete:
            del self._docs[did]
        return len(to_delete)

    def query(self, vector: list[float], filter_expr: str | None = None, topk: int = 10) -> list[dict]:
        candidates = []
        for doc in self._docs.values():
            if filter_expr and not self._match(doc, filter_expr):
                continue
            emb = doc.get("embedding")
            if emb is None:
                continue
            score = self._cosine_sim(vector, emb)
            candidates.append({**doc, "score": score})
        candidates.sort(key=lambda d: d["score"], reverse=True)
        return candidates[:topk]

    def flush(self) -> None:
        """No-op for in-memory store; exists for interface compatibility."""

    # ---- test helpers -----------------------------------------------------

    def get_all_doc_ids(self) -> set[str]:
        return set(self._docs.keys())

    def get_docs_by_path(self, path: str) -> list[dict]:
        return [dict(d) for d in self._docs.values() if d.get("path") == path]

    def get_all_paths(self) -> set[str]:
        return {d["path"] for d in self._docs.values() if "path" in d}

    def doc_count(self) -> int:
        return len(self._docs)

    def direct_delete(self, ids: list[str]) -> None:
        """Delete docs by id without going through filter (for intentional corruption in T26)."""
        for did in ids:
            self._docs.pop(did, None)

    def get_all_docs(self) -> list[dict]:
        return [dict(d) for d in self._docs.values()]

    # ---- internals --------------------------------------------------------

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    @staticmethod
    def _match(doc: dict, filter_expr: str) -> bool:
        """Simple filter parser supporting ``key == 'val'`` and ``key != 'val'``
        joined by ``AND``.
        """
        clauses = [c.strip() for c in filter_expr.split(" AND ")]
        for clause in clauses:
            if "!=" in clause:
                key, val = clause.split("!=", 1)
                key = key.strip()
                val = val.strip().strip("'\"")
                if str(doc.get(key, "")) == val:
                    return False
            elif "==" in clause:
                key, val = clause.split("==", 1)
                key = key.strip()
                val = val.strip().strip("'\"")
                if str(doc.get(key, "")) != val:
                    return False
        return True


# ---------------------------------------------------------------------------
# Failing variants (for T34 / T35 health-check failure tests)
# ---------------------------------------------------------------------------


class FailingMockChunker:
    """Chunker that always raises on chunk()."""

    def schema_id(self) -> str:
        return "failing-mock-chunker"

    def chunk(self, text: str, path: str) -> list[Chunk]:
        raise RuntimeError("Chunker processing failed")


class FailingMockEmbedder:
    """Embedder that always raises on every method call."""

    def id(self) -> str:
        return "failing-mock-embedder"

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("Embedding service unavailable: API key not set")


class FailingMockVectorStore:
    """VectorStore that always raises on every method call."""

    def upsert(self, docs: list[dict]) -> None:
        raise ConnectionError("VectorStore connection refused")

    def update(self, docs: list[dict]) -> None:
        raise ConnectionError("VectorStore connection refused")

    def fetch(self, ids: list[str]) -> list[dict]:
        raise ConnectionError("VectorStore connection refused")

    def delete_by_filter(self, filter_expr: str) -> int:
        raise ConnectionError("VectorStore connection refused")

    def query(self, vector: list[float], filter_expr: str | None = None, topk: int = 10) -> list[dict]:
        raise ConnectionError("VectorStore connection refused")

    def flush(self) -> None:
        raise ConnectionError("VectorStore connection refused")

    def doc_count(self) -> int:
        raise ConnectionError("VectorStore connection refused")

    def get_all_doc_ids(self) -> set[str]:
        raise ConnectionError("VectorStore connection refused")

    def get_docs_by_path(self, path: str) -> list[dict]:
        raise ConnectionError("VectorStore connection refused")

    def get_all_paths(self) -> set[str]:
        raise ConnectionError("VectorStore connection refused")


# ---------------------------------------------------------------------------
# CrashAfterN helper (for T14 crash tests)
# ---------------------------------------------------------------------------


class CrashAfterNUpserts(MockVectorStore):
    """MockVectorStore that raises after *n* upsert calls (for crash tests)."""

    def __init__(self, crash_after: int = 3):
        super().__init__()
        self._upsert_call_count = 0
        self._crash_after = crash_after

    def seed_docs(self, docs: list[dict]) -> None:
        """Preload documents without consuming crash budget."""
        super().upsert(docs)

    def upsert(self, docs: list[dict]) -> None:
        self._upsert_call_count += 1
        if self._upsert_call_count > self._crash_after:
            raise RuntimeError("Simulated crash during upsert")
        super().upsert(docs)


# ---------------------------------------------------------------------------
# Transient / Permanent fail embedders (for retry tests)
# ---------------------------------------------------------------------------


class _HttpLikeError(Exception):
    """Exception with a status_code attribute for duck-typing."""

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


class TransientFailEmbedder(MockEmbedder):
    """Embedder that fails with a transient 429 error for the first *fail_count* calls."""

    def __init__(self, fail_count: int = 2):
        super().__init__()
        self._fail_count = fail_count

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        if self.call_count <= self._fail_count:
            raise _HttpLikeError("rate limit exceeded", status_code=429)
        self.total_texts_embedded += len(texts)
        return [self._text_to_vector(t) for t in texts]


class PermanentFailEmbedder(MockEmbedder):
    """Embedder that always fails with a permanent 401 error."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        raise _HttpLikeError("invalid api key", status_code=401)


class TransientFailAsyncEmbedder(MockEmbedder):
    """Embedder whose async_embed fails with transient 503 for the first *fail_count* calls."""

    def __init__(self, fail_count: int = 2):
        super().__init__()
        self._fail_count = fail_count
        self._async_call_count = 0

    async def async_embed(self, texts: list[str]) -> list[list[float]]:
        self._async_call_count += 1
        if self._async_call_count <= self._fail_count:
            raise _HttpLikeError("service unavailable", status_code=503)
        self.total_texts_embedded += len(texts)
        return [self._text_to_vector(t) for t in texts]


class CrashOnFlush(MockVectorStore):
    """MockVectorStore that raises on the first flush() call (for T15)."""

    def __init__(self):
        super().__init__()
        self._flush_count = 0

    def flush(self) -> None:
        self._flush_count += 1
        if self._flush_count == 1:
            raise RuntimeError("Simulated crash during flush")
        super().flush()
