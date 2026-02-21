"""Generic adapter wrapping any LangChain Embeddings to MinSync's Embedder protocol."""

from __future__ import annotations

from typing import Any


class LangChainEmbeddingsAdapter:
    """Adapts any LangChain ``Embeddings`` instance to MinSync's Embedder protocol.

    Usage::

        from langchain_openai import OpenAIEmbeddings
        from minsync.embedders import LangChainEmbeddingsAdapter

        lc = OpenAIEmbeddings(model="text-embedding-3-small")
        embedder = LangChainEmbeddingsAdapter(lc, "openai:text-embedding-3-small")
    """

    def __init__(self, langchain_embeddings: Any, embedder_id: str) -> None:
        self._embeddings = langchain_embeddings
        self._id = embedder_id

    def id(self) -> str:
        return self._id

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._embeddings.embed_documents(texts)
