"""Tests for the zvec vector store adapter.

Covers:
- _translate_filter pure function
- ZvecVectorStore unit tests (with real zvec)
- Factory integration (zvec installed vs fallback)
- Full integration round-trip
"""

from __future__ import annotations

import builtins
import contextlib
from unittest.mock import patch

import pytest

# Always available — pure function tests
from minsync.vectorstores.zvec_adapter import ZvecVectorStore, _translate_filter

zvec = pytest.importorskip("zvec", reason="zvec not installed")

original_import = builtins.__import__


# ===========================================================================
# TestTranslateFilter — pure function, no zvec dependency
# ===========================================================================


class TestTranslateFilter:
    def test_eq_converted(self):
        assert _translate_filter("repo_id == 'abc'") == "repo_id = 'abc'"

    def test_neq_preserved(self):
        assert _translate_filter("seen_token != 'tok1'") == "seen_token != 'tok1'"

    def test_compound_and(self):
        expr = "repo_id == 'abc' AND ref == 'main' AND seen_token != 'tok1'"
        expected = "repo_id = 'abc' AND ref = 'main' AND seen_token != 'tok1'"
        assert _translate_filter(expr) == expected

    def test_no_operator_passthrough(self):
        assert _translate_filter("field > 10") == "field > 10"


# ===========================================================================
# TestZvecVectorStore — real zvec, temp directory
# ===========================================================================


@pytest.fixture()
def zvec_store(tmp_path):
    """Create a ZvecVectorStore backed by a temp directory."""
    db_path = str(tmp_path / "test_zvec_db")
    store = ZvecVectorStore(db_path=db_path, collection_name="test")
    yield store
    if store._collection is not None:
        with contextlib.suppress(Exception):
            store._collection.destroy()


def _make_doc(doc_id: str, text: str = "hello", embedding: list[float] | None = None, **kwargs) -> dict:
    doc = {
        "id": doc_id,
        "text": text,
        "repo_id": kwargs.get("repo_id", "r1"),
        "ref": kwargs.get("ref", "main"),
        "path": kwargs.get("path", "file.md"),
        "heading_path": kwargs.get("heading_path", ""),
        "chunk_type": kwargs.get("chunk_type", "parent"),
        "content_commit": kwargs.get("content_commit", "abc123"),
        "seen_token": kwargs.get("seen_token", "tok1"),
        "chunk_schema_id": kwargs.get("chunk_schema_id", ""),
        "content_hash": kwargs.get("content_hash", "hash1"),
    }
    if embedding is not None:
        doc["embedding"] = embedding
    else:
        doc["embedding"] = [0.1, 0.2, 0.3]
    return doc


class TestZvecVectorStoreUnit:
    def test_upsert_creates_collection(self, zvec_store):
        assert zvec_store._collection is None
        docs = [_make_doc("d1")]
        zvec_store.upsert(docs)
        assert zvec_store._collection is not None

    def test_upsert_and_fetch(self, zvec_store):
        docs = [_make_doc("d1"), _make_doc("d2", text="world")]
        zvec_store.upsert(docs)
        zvec_store.flush()

        fetched = zvec_store.fetch(["d1", "d2"])
        assert len(fetched) == 2
        ids = {d["id"] for d in fetched}
        assert ids == {"d1", "d2"}

    def test_fetch_missing_id(self, zvec_store):
        zvec_store.upsert([_make_doc("d1")])
        zvec_store.flush()
        fetched = zvec_store.fetch(["d1", "nonexistent"])
        assert len(fetched) == 1
        assert fetched[0]["id"] == "d1"

    def test_fetch_empty_collection(self, zvec_store):
        assert zvec_store.fetch(["d1"]) == []

    def test_update_preserves_embedding(self, zvec_store):
        zvec_store.upsert([_make_doc("d1", embedding=[1.0, 2.0, 3.0])])
        zvec_store.flush()

        zvec_store.update([{"id": "d1", "text": "updated"}])
        zvec_store.flush()

        fetched = zvec_store.fetch(["d1"])
        assert len(fetched) == 1
        assert fetched[0]["text"] == "updated"
        assert fetched[0].get("embedding") is not None

    def test_delete_by_filter(self, zvec_store):
        zvec_store.upsert([
            _make_doc("d1", repo_id="r1"),
            _make_doc("d2", repo_id="r2"),
        ])
        zvec_store.flush()

        deleted = zvec_store.delete_by_filter("repo_id == 'r1'")
        assert deleted == 1
        assert zvec_store.doc_count() == 1

    def test_query(self, zvec_store):
        zvec_store.upsert([
            _make_doc("d1", embedding=[1.0, 0.0, 0.0], repo_id="r1"),
            _make_doc("d2", embedding=[0.0, 1.0, 0.0], repo_id="r1"),
            _make_doc("d3", embedding=[0.0, 0.0, 1.0], repo_id="r2"),
        ])
        zvec_store.flush()

        results = zvec_store.query(
            vector=[1.0, 0.0, 0.0],
            filter_expr="repo_id == 'r1'",
            topk=2,
        )
        assert len(results) <= 2
        assert all(r.get("repo_id") == "r1" for r in results)
        # d1 should rank highest
        assert results[0]["id"] == "d1"

    def test_doc_count(self, zvec_store):
        assert zvec_store.doc_count() == 0
        zvec_store.upsert([_make_doc("d1"), _make_doc("d2")])
        zvec_store.flush()
        assert zvec_store.doc_count() == 2

    def test_flush_noop_on_empty(self, zvec_store):
        zvec_store.flush()  # should not raise

    def test_fetch_by_filter(self, zvec_store):
        zvec_store.upsert([
            _make_doc("d1", repo_id="r1", ref="main"),
            _make_doc("d2", repo_id="r1", ref="dev"),
            _make_doc("d3", repo_id="r2", ref="main"),
        ])
        zvec_store.flush()

        results = zvec_store.fetch_by_filter("repo_id == 'r1' AND ref == 'main'")
        assert len(results) == 1
        assert results[0]["id"] == "d1"

    def test_upsert_empty_list(self, zvec_store):
        zvec_store.upsert([])  # should not raise or create collection
        assert zvec_store._collection is None

    def test_reopen_existing_db(self, tmp_path):
        db_path = str(tmp_path / "reopen_db")
        store1 = ZvecVectorStore(db_path=db_path)
        store1.upsert([_make_doc("d1")])
        store1.flush()
        # Release the lock so a second instance can open
        store1._collection = None

        # New instance pointing at same path
        store2 = ZvecVectorStore(db_path=db_path)
        fetched = store2.fetch(["d1"])
        assert len(fetched) == 1
        assert fetched[0]["id"] == "d1"

        store2._collection.destroy()


# ===========================================================================
# TestFactoryZvec — factory integration
# ===========================================================================


def _import_block_zvec(name, *args, **kwargs):
    if name == "zvec" or name.startswith("zvec."):
        raise ImportError(f"Mocked: {name} not available")
    return original_import(name, *args, **kwargs)


class TestFactoryZvec:
    def test_factory_returns_zvec_store(self, tmp_path):
        from minsync.factory import create_vectorstore

        config = {
            "vectorstore": {
                "id": "zvec",
                "collection": {
                    "path": str(tmp_path / "factory_db"),
                    "name": "test_coll",
                },
            }
        }
        store = create_vectorstore(config)
        assert isinstance(store, ZvecVectorStore)

    def test_factory_fallback_when_zvec_missing(self, tmp_path):
        from minsync.core import _InMemoryVectorStore
        from minsync.factory import create_vectorstore

        config = {"vectorstore": {"id": "zvec"}}
        with patch.dict("sys.modules", {"zvec": None}), patch("builtins.__import__", side_effect=_import_block_zvec):
            store = create_vectorstore(config)
            assert isinstance(store, _InMemoryVectorStore)
