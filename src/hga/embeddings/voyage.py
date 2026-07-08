"""Voyage AI embedders — voyage-3-large (text) + voyage-multimodal-3.

Voyage models are asymmetric: documents embed with input_type="document",
queries with input_type="query". The text embedder targets 1024-dim by default.
"""

from __future__ import annotations

from typing import Any

from hga.embeddings.base import Embedder

_VOYAGE_DIMS: dict[str, int] = {
    "voyage-3-large": 1024,
    "voyage-3-lite": 512,
    "voyage-3": 1024,
    "voyage-multimodal-3": 1024,
}


class VoyageTextEmbedder(Embedder):
    """Dense text embeddings via voyage-3-large."""

    def __init__(self, model: str = "voyage-3-large", batch_size: int = 128) -> None:
        import voyageai

        self._model = model
        self._batch_size = batch_size
        self._client = voyageai.AsyncClient()

    @property
    def name(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        return _VOYAGE_DIMS.get(self._model, 1024)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            obj = await self._client.embed(batch, model=self._model, input_type="document")
            out.extend(list(v) for v in obj.embeddings)
        return out

    async def embed_query(self, text: str) -> list[float]:
        obj = await self._client.embed([text], model=self._model, input_type="query")
        return list(obj.embeddings[0])


class VoyageMultimodalEmbedder(Embedder):
    """Multimodal embeddings via voyage-multimodal-3 (text + image in one space).

    The Voyage multimodal API takes a list-of-lists where each inner list is a
    sequence of parts (text strings or image dicts). See Voyage docs for the
    exact image-part schema; here we expose a thin pass-through.
    """

    def __init__(self, model: str = "voyage-multimodal-3", batch_size: int = 32) -> None:
        import voyageai

        self._model = model
        self._batch_size = batch_size
        self._client = voyageai.AsyncClient()

    @property
    def name(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        return _VOYAGE_DIMS.get(self._model, 1024)

    async def embed_documents_multimodal(
        self, parts_batches: list[list[object]]
    ) -> list[list[float]]:
        """Embed multimodal document parts. Each inner list is one document's parts."""
        parts: Any = parts_batches
        obj = await self._client.embed(parts, model=self._model, input_type="document")
        return [list(v) for v in obj.embeddings]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self.embed_documents_multimodal([[t] for t in texts])

    async def embed_query(self, text: str) -> list[float]:
        parts: Any = [[text]]
        obj = await self._client.embed(parts, model=self._model, input_type="query")
        return list(obj.embeddings[0])
