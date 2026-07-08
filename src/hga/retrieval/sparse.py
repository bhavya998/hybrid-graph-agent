"""Sparse embedder interface + fastembed BM25 implementation.

The dense side (Voyage) is API-driven; the sparse side (BM25) runs locally via
fastembed, which keeps the all-API constraint while giving genuine lexical
hybrid retrieval. Qdrant local mode needs explicit sparse vectors, which this
module produces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class SparseVec:
    """A sparse vector — hashed-token indices and their BM25 weights."""

    indices: list[int]
    values: list[float]


@runtime_checkable
class SparseEmbedder(Protocol):
    """Lexical sparse embedder for the BM25 side of hybrid retrieval."""

    @property
    def name(self) -> str: ...

    def embed_documents(self, texts: list[str]) -> list[SparseVec]: ...

    def embed_query(self, text: str) -> SparseVec: ...


class FastembedBM25(SparseEmbedder):
    """BM25 sparse embeddings via fastembed's `Qdrant/bm25` model.

    Runs fully locally (no API, no GPU). Token hashing + stemming happens inside
    the model; we only convert its numpy outputs to plain Python scalars.
    """

    def __init__(self, model: str = "Qdrant/bm25") -> None:
        from fastembed import SparseTextEmbedding

        self._model_name = model
        self._model = SparseTextEmbedding(model_name=model)

    @property
    def name(self) -> str:
        return self._model_name

    def embed_documents(self, texts: list[str]) -> list[SparseVec]:
        return [
            SparseVec(
                indices=[int(i) for i in emb.indices],
                values=[float(v) for v in emb.values],
            )
            for emb in self._model.embed(texts)
        ]

    def embed_query(self, text: str) -> SparseVec:
        emb = next(iter(self._model.query_embed(text)))
        return SparseVec(
            indices=[int(i) for i in emb.indices],
            values=[float(v) for v in emb.values],
        )
