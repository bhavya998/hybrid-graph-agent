"""VectorStore interface + record/result shapes.

The store owns the sparse embedder (BM25, local) and pairs it with externally
supplied dense vectors (Voyage, API). All methods are async; the Qdrant local
implementation wraps the synchronous path-mode client via a thread executor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(slots=True)
class VectorRecord:
    """A point to index: dense vector + text + payload. Sparse is generated inside."""

    chunk_id: str
    doc_id: str
    content: str
    dense_vector: list[float]
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A retrieved point with its fused/dense/sparse score and full payload."""

    chunk_id: str
    score: float
    payload: dict[str, Any]


@runtime_checkable
class VectorStore(Protocol):
    """Hybrid vector store: dense + sparse(BM25) with RRF fusion."""

    @property
    def collection(self) -> str: ...

    async def ensure_collection(self, dense_dim: int) -> None: ...

    async def upsert(self, records: list[VectorRecord], batch_size: int = 100) -> None: ...

    async def search_dense(
        self, query: list[float], k: int, filters: dict[str, Any] | None = None
    ) -> list[SearchResult]: ...

    async def search_sparse(
        self, query_text: str, k: int, filters: dict[str, Any] | None = None
    ) -> list[SearchResult]: ...

    async def search_hybrid(
        self,
        query: list[float],
        query_text: str,
        k: int,
        rrf_k: int = 60,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]: ...

    async def delete_by_doc(self, doc_id: str) -> None: ...

    async def count(self) -> int: ...
