"""Embedder interface — provider-agnostic, modality-aware."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Dense embedder.

    `input_type` distinguishes the document side (indexing) from the query side
    (search). Asymmetric embedding notably improves Voyage models.
    """

    @property
    def name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...
