"""Reranker interface + Cohere implementation.

The reranker is a cross-encoder that runs AFTER hybrid fusion, narrowing the
fused candidate set to the final context window. Cohere's rerank-v3.5 is the
hosted SOTA choice; the interface allows swapping to a self-hosted model later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class RerankResult:
    """A reranked document: its original index, score, and text."""

    index: int
    relevance_score: float
    document: str


@runtime_checkable
class Reranker(Protocol):
    """Cross-encoder reranker applied to the fused candidate set."""

    @property
    def name(self) -> str: ...

    async def rerank(self, query: str, documents: list[str], top_k: int) -> list[RerankResult]: ...


class CohereReranker(Reranker):
    """Cohere rerank-v3.5 via the async v2 client. Reads COHERE_API_KEY from env."""

    def __init__(self, model: str = "rerank-v3.5") -> None:
        import cohere

        self._model = model
        self._client = cohere.AsyncClientV2()

    @property
    def name(self) -> str:
        return self._model

    async def rerank(self, query: str, documents: list[str], top_k: int) -> list[RerankResult]:
        if not documents:
            return []
        top_n = min(top_k, len(documents))
        resp = await self._client.rerank(
            model=self._model, query=query, documents=documents, top_n=top_n
        )
        out: list[RerankResult] = []
        for r in resp.results:
            idx = int(r.index)
            score = float(r.relevance_score)
            out.append(RerankResult(index=idx, relevance_score=score, document=documents[idx]))
        return out
