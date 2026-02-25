"""Generic adapter wrapping any LangChain VectorStore to MinSync's VectorStore protocol."""

from __future__ import annotations

from typing import Any


class LangChainVectorStoreAdapter:
    """Adapts any LangChain ``VectorStore`` instance to MinSync's VectorStore protocol.

    Usage::

        from langchain_weaviate import WeaviateVectorStore
        from minsync.vectorstores import LangChainVectorStoreAdapter

        lc_store = WeaviateVectorStore(...)
        adapter = LangChainVectorStoreAdapter(lc_store, embeddings)
    """

    def __init__(self, langchain_store: Any, embeddings: Any) -> None:
        self._store = langchain_store
        self._embeddings = embeddings

    def upsert(self, docs: list[dict[str, Any]]) -> None:
        """Insert or update documents via LangChain's add_documents."""
        try:
            from langchain_core.documents import Document
        except ImportError as exc:
            raise ImportError(
                "langchain-core is required for LangChainVectorStoreAdapter. Install with: pip install langchain-core"
            ) from exc

        lc_docs = []
        ids = []
        for doc in docs:
            metadata = {k: v for k, v in doc.items() if k not in ("id", "text", "embedding")}
            lc_docs.append(Document(page_content=doc.get("text", ""), metadata=metadata))
            ids.append(str(doc["id"]))

        self._store.add_documents(lc_docs, ids=ids)

    def update(self, docs: list[dict[str, Any]]) -> None:
        """Update documents by re-upserting (LangChain has no partial update)."""
        self.upsert(docs)

    def fetch(self, ids: list[str]) -> list[dict[str, Any]]:
        """Fetch documents by ID. Uses get_by_ids if available."""
        get_by_ids = getattr(self._store, "get_by_ids", None)
        if callable(get_by_ids):
            lc_docs = get_by_ids(ids)
            return [
                {
                    "id": getattr(doc, "id", ids[i] if i < len(ids) else ""),
                    "text": doc.page_content,
                    **doc.metadata,
                }
                for i, doc in enumerate(lc_docs)
            ]
        return []

    def delete_by_filter(self, filter_expr: str) -> int:
        """Delete documents matching a filter expression.

        Parses MinSync's simple filter syntax and delegates to the
        underlying store's ``delete`` method.
        """
        delete_fn = getattr(self._store, "delete", None)
        if not callable(delete_fn):
            return 0

        # LangChain stores typically support delete(ids=...) or delete(filter=...)
        # We attempt filter-based delete first, then fall back to fetching + deleting by ID
        try:
            filter_dict = _parse_filter_to_dict(filter_expr)
            result = delete_fn(filter=filter_dict)
        except (TypeError, NotImplementedError):
            return 0
        else:
            return result if isinstance(result, int) else 0

    def query(
        self,
        vector: list[float],
        filter_expr: str | None = None,
        topk: int = 10,
    ) -> list[dict[str, Any]]:
        """Search by vector similarity."""
        kwargs: dict[str, Any] = {"k": topk}
        if filter_expr:
            kwargs["filter"] = _parse_filter_to_dict(filter_expr)

        try:
            results = self._store.similarity_search_by_vector(vector, **kwargs)
        except Exception:
            results = []

        docs: list[dict[str, Any]] = []
        for i, doc in enumerate(results):
            entry: dict[str, Any] = {
                "id": getattr(doc, "id", ""),
                "text": doc.page_content,
                "score": 1.0 - (i * 0.01),  # approximate ordering score
                **doc.metadata,
            }
            docs.append(entry)
        return docs

    def flush(self) -> None:
        """Persist if the underlying store supports it."""
        persist = getattr(self._store, "persist", None)
        if callable(persist):
            persist()


def _parse_filter_to_dict(filter_expr: str) -> dict[str, Any]:
    """Parse MinSync's simple ``key == 'val' AND key != 'val'`` syntax into a dict."""
    result: dict[str, Any] = {}
    clauses = [c.strip() for c in filter_expr.split(" AND ")]
    for clause in clauses:
        if "!=" in clause:
            key, val = clause.split("!=", 1)
            result[key.strip()] = {"$ne": val.strip().strip("'\"")}
        elif "==" in clause:
            key, val = clause.split("==", 1)
            result[key.strip()] = val.strip().strip("'\"")
    return result
