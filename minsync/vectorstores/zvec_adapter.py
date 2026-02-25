"""Zvec embedded vector database adapter for MinSync."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _translate_filter(filter_expr: str) -> str:
    """Translate MinSync filter syntax to zvec filter syntax.

    MinSync uses ``==`` for equality; zvec uses ``=``.
    ``!=`` is the same in both systems and must be preserved.
    """
    clauses = filter_expr.split(" AND ")
    translated: list[str] = []
    for clause in clauses:
        stripped = clause.strip()
        if "!=" in stripped:
            translated.append(stripped)
        elif "==" in stripped:
            translated.append(stripped.replace("==", "=", 1))
        else:
            translated.append(stripped)
    return " AND ".join(translated)


_WRITE_BATCH_SIZE = 512
_FETCH_BATCH_SIZE = 512


class ZvecVectorStore:
    """Zvec-backed vector store with file-based persistence and HNSW indexing."""

    _SCALAR_FIELDS = (
        "repo_id",
        "ref",
        "path",
        "heading_path",
        "chunk_type",
        "text",
        "content_commit",
        "seen_token",
        "chunk_schema_id",
        "content_hash",
    )
    _INDEXED_FIELDS = ("repo_id", "ref", "path", "seen_token")

    def __init__(self, db_path: str, collection_name: str = "minsync") -> None:
        self._db_path = db_path
        self._collection_name = collection_name
        self._collection: Any = None

    def _ensure_collection(self) -> Any:
        """Open an existing zvec collection, or return None if it doesn't exist."""
        if self._collection is not None:
            return self._collection

        db = Path(self._db_path)
        if not db.exists():
            return None

        import zvec

        self._collection = zvec.open(str(db))
        return self._collection

    def _create_collection(self, dimension: int) -> Any:
        """Create a new zvec collection with the given vector dimension."""
        import zvec

        db = Path(self._db_path)
        db.parent.mkdir(parents=True, exist_ok=True)

        fields = []
        for name in self._SCALAR_FIELDS:
            idx = zvec.InvertIndexParam() if name in self._INDEXED_FIELDS else None
            fields.append(zvec.FieldSchema(name, zvec.DataType.STRING, nullable=True, index_param=idx))

        vectors = [
            zvec.VectorSchema("embedding", zvec.DataType.VECTOR_FP32, dimension=dimension),
        ]

        schema = zvec.CollectionSchema(
            name=self._collection_name,
            fields=fields,
            vectors=vectors,
        )
        self._collection = zvec.create_and_open(str(db), schema)
        return self._collection

    def _to_zvec_doc(self, doc: dict[str, Any]) -> Any:
        """Convert a MinSync doc dict to a zvec Doc."""
        import zvec

        doc_id = str(doc["id"])
        embedding = doc.get("embedding")
        vectors = {"embedding": embedding} if embedding is not None else None

        fields: dict[str, Any] = {}
        for name in self._SCALAR_FIELDS:
            val = doc.get(name)
            if val is not None:
                fields[name] = str(val)

        return zvec.Doc(id=doc_id, vectors=vectors, fields=fields)

    def _from_zvec_doc(self, doc: Any, *, include_vector: bool = False) -> dict[str, Any]:
        """Convert a zvec Doc to a MinSync doc dict."""
        result: dict[str, Any] = {"id": doc.id}
        if doc.fields:
            for key, val in doc.fields.items():
                result[key] = val
        if doc.score is not None:
            result["score"] = doc.score
        if include_vector and doc.vectors:
            emb = doc.vectors.get("embedding")
            if emb is not None:
                result["embedding"] = list(emb) if not isinstance(emb, list) else emb
        return result

    def upsert(self, docs: list[dict[str, Any]]) -> None:
        if not docs:
            return
        coll = self._ensure_collection()
        if coll is None:
            first_embedding = None
            for d in docs:
                emb = d.get("embedding")
                if emb is not None:
                    first_embedding = emb
                    break
            if first_embedding is None:
                return
            coll = self._create_collection(len(first_embedding))

        zvec_docs = [self._to_zvec_doc(d) for d in docs]
        for i in range(0, len(zvec_docs), _WRITE_BATCH_SIZE):
            coll.upsert(zvec_docs[i : i + _WRITE_BATCH_SIZE])

    def update(self, docs: list[dict[str, Any]]) -> None:
        if not docs:
            return
        coll = self._ensure_collection()
        if coll is None:
            return

        zvec_docs = [self._to_zvec_doc(d) for d in docs]
        for i in range(0, len(zvec_docs), _WRITE_BATCH_SIZE):
            coll.update(zvec_docs[i : i + _WRITE_BATCH_SIZE])

    def fetch(self, ids: list[str]) -> list[dict[str, Any]]:
        coll = self._ensure_collection()
        if coll is None:
            return []
        if not ids:
            return []

        results: list[dict[str, Any]] = []
        for i in range(0, len(ids), _FETCH_BATCH_SIZE):
            batch_ids = ids[i : i + _FETCH_BATCH_SIZE]
            result_map = coll.fetch(batch_ids)
            for doc_id in batch_ids:
                if doc_id in result_map:
                    results.append(self._from_zvec_doc(result_map[doc_id], include_vector=True))
        return results

    def delete_by_filter(self, filter_expr: str) -> int:
        coll = self._ensure_collection()
        if coll is None:
            return 0

        before = coll.stats.doc_count
        coll.delete_by_filter(_translate_filter(filter_expr))
        coll.flush()
        after = coll.stats.doc_count
        return max(before - after, 0)

    def query(
        self,
        vector: list[float],
        filter_expr: str | None = None,
        topk: int = 10,
    ) -> list[dict[str, Any]]:
        import zvec

        coll = self._ensure_collection()
        if coll is None:
            return []

        vq = zvec.VectorQuery("embedding", vector=vector)
        translated = _translate_filter(filter_expr) if filter_expr else None
        raw = coll.query(vectors=vq, topk=topk, filter=translated, include_vector=True)
        return [self._from_zvec_doc(doc, include_vector=True) for doc in raw]

    def flush(self) -> None:
        coll = self._ensure_collection()
        if coll is not None:
            coll.flush()

    def doc_count(self) -> int:
        coll = self._ensure_collection()
        if coll is None:
            return 0
        return coll.stats.doc_count

    def fetch_by_filter(self, filter_expr: str) -> list[dict[str, Any]]:
        """Retrieve all documents matching a filter (used by verify)."""
        coll = self._ensure_collection()
        if coll is None:
            return []

        translated = _translate_filter(filter_expr)
        # Retrieve all matching docs via repeated query batches (zvec max topk=1024)
        all_docs: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        batch_size = 1024
        while True:
            raw = coll.query(topk=batch_size, filter=translated, include_vector=True)
            new_in_batch = 0
            for doc in raw:
                if doc.id not in seen_ids:
                    seen_ids.add(doc.id)
                    all_docs.append(self._from_zvec_doc(doc, include_vector=True))
                    new_in_batch += 1
            if new_in_batch == 0 or len(raw) < batch_size:
                break
        return all_docs
