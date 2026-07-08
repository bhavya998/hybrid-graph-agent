"""Retrieval package — hybrid vector store + reranker factories."""

from __future__ import annotations

from hga.config import Settings, get_settings
from hga.retrieval.local_reranker import LocalCrossEncoderReranker
from hga.retrieval.qdrant_store import QdrantStore
from hga.retrieval.reranker import CohereReranker, Reranker, RerankResult
from hga.retrieval.store import SearchResult, VectorRecord, VectorStore

__all__ = [
    "VectorStore",
    "VectorRecord",
    "SearchResult",
    "QdrantStore",
    "Reranker",
    "RerankResult",
    "CohereReranker",
    "LocalCrossEncoderReranker",
    "get_vector_store",
    "get_reranker",
]


def get_vector_store(settings: Settings | None = None) -> VectorStore:
    """Return the vector store (Qdrant path-mode for all profiles)."""
    s = settings or get_settings()
    if s.profile in ("local_gpu", "local"):
        return QdrantStore(path=str(s.qdrant_path), collection="chunks")
    raise NotImplementedError(f"VectorStore profile {s.profile!r} not yet implemented")


def get_reranker(settings: Settings | None = None) -> Reranker:
    """Return the reranker: local cross-encoder for GPU, Cohere for API."""
    s = settings or get_settings()
    if s.profile == "local_gpu":
        return LocalCrossEncoderReranker(model=s.local_rerank_model)
    if s.profile == "local":
        return CohereReranker(model=s.rerank_model)
    raise NotImplementedError(f"Reranker profile {s.profile!r} not yet implemented")
