"""Component factory: config ID → component instance."""

from __future__ import annotations

from typing import Any

from minsync.core import MinSyncError, _InMemoryVectorStore


def create_chunker(config: dict[str, Any]) -> Any:
    """Create a chunker from config.

    Supported chunker IDs:
    - ``markdown-heading`` → :class:`MarkdownHeadingChunker`
    - ``sliding-window`` → :class:`SlidingWindowChunker`
    """
    chunker_config = config.get("chunker") or {}
    chunker_id = str(chunker_config.get("id", "markdown-heading"))
    options = chunker_config.get("options") or {}

    max_chunk_size = int(options.get("max_chunk_size", 1000))
    overlap = int(options.get("overlap", 100))

    if chunker_id == "markdown-heading":
        from minsync.chunkers.markdown import MarkdownHeadingChunker

        return MarkdownHeadingChunker(max_chunk_size=max_chunk_size, overlap=overlap)

    if chunker_id == "sliding-window":
        from minsync.chunkers.sliding_window import SlidingWindowChunker

        return SlidingWindowChunker(max_chunk_size=max_chunk_size, overlap=overlap)

    raise MinSyncError(
        f"unknown chunker '{chunker_id}'. Supported: markdown-heading, sliding-window",
        exit_code=1,
    )


def create_embedder(config: dict[str, Any]) -> Any:
    """Create an embedder from config.

    Supported embedder ID prefixes:
    - ``openai:*`` → LangChain ``OpenAIEmbeddings``
    - ``huggingface:*`` → LangChain ``HuggingFaceEmbeddings``
    """
    from minsync.embedders.langchain_adapter import LangChainEmbeddingsAdapter

    embedder_config = config.get("embedder") or {}
    embedder_id = str(embedder_config.get("id", "openai:text-embedding-3-small"))

    if embedder_id.startswith("openai:"):
        model_name = embedder_id.split(":", 1)[1]
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:
            raise MinSyncError(
                f"langchain-openai package not found for embedder '{embedder_id}'. "
                f"Install with: pip install langchain-openai",
                exit_code=1,
            ) from exc
        lc = OpenAIEmbeddings(model=model_name)
        return LangChainEmbeddingsAdapter(lc, embedder_id)

    if embedder_id.startswith("huggingface:"):
        model_name = embedder_id.split(":", 1)[1]
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise MinSyncError(
                f"langchain-huggingface package not found for embedder '{embedder_id}'. "
                f"Install with: pip install langchain-huggingface",
                exit_code=1,
            ) from exc
        lc = HuggingFaceEmbeddings(model_name=model_name)
        return LangChainEmbeddingsAdapter(lc, embedder_id)

    raise MinSyncError(
        f"unknown embedder '{embedder_id}'. Supported prefixes: openai:, huggingface:",
        exit_code=1,
    )


def create_vectorstore(config: dict[str, Any], embedder: Any = None) -> Any:
    """Create a vector store from config.

    Supported vectorstore IDs:
    - ``zvec`` → built-in :class:`_InMemoryVectorStore`
    - ``weaviate`` → LangChain ``WeaviateVectorStore``
    - ``chroma`` → LangChain ``Chroma``
    - ``qdrant`` → LangChain ``QdrantVectorStore``
    """
    from minsync.vectorstores.langchain_adapter import LangChainVectorStoreAdapter

    vs_config = config.get("vectorstore") or {}
    vs_id = str(vs_config.get("id", "zvec"))
    options = vs_config.get("options") or {}

    if vs_id == "zvec":
        return _InMemoryVectorStore()

    if vs_id == "weaviate":
        try:
            from langchain_weaviate import WeaviateVectorStore
        except ImportError as exc:
            raise MinSyncError(
                "'weaviate-client' package not found for configured vectorstore 'weaviate'. "
                "Install with: pip install weaviate-client langchain-weaviate",
                exit_code=1,
            ) from exc
        lc_store = WeaviateVectorStore(**options)
        return LangChainVectorStoreAdapter(lc_store, embedder)

    if vs_id == "chroma":
        try:
            from langchain_chroma import Chroma
        except ImportError as exc:
            raise MinSyncError(
                "'chromadb' package not found for configured vectorstore 'chroma'. "
                "Install with: pip install chromadb langchain-chroma",
                exit_code=1,
            ) from exc
        lc_store = Chroma(**options)
        return LangChainVectorStoreAdapter(lc_store, embedder)

    if vs_id == "qdrant":
        try:
            from langchain_qdrant import QdrantVectorStore
        except ImportError as exc:
            raise MinSyncError(
                "'qdrant-client' package not found for configured vectorstore 'qdrant'. "
                "Install with: pip install qdrant-client langchain-qdrant",
                exit_code=1,
            ) from exc
        lc_store = QdrantVectorStore(**options)
        return LangChainVectorStoreAdapter(lc_store, embedder)

    raise MinSyncError(
        f"unknown vectorstore '{vs_id}'. Supported: zvec, weaviate, chroma, qdrant",
        exit_code=1,
    )
