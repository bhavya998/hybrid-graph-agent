"""Local GPU embedder — sentence-transformers / HuggingFace, no API.

Uses a lightweight bilingual embedding model (bge-small-en-v1.5, 384-dim) that
runs fast on CPU or GPU. Batched encoding with automatic device placement.
"""

from __future__ import annotations

from typing import Any

from hga.embeddings.base import Embedder

_MODEL_DIMS: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "intfloat/e5-small-v2": 384,
}


class LocalTextEmbedder(Embedder):
    """Dense text embeddings via a sentence-transformers model running locally."""

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", batch_size: int = 64) -> None:
        self._model_id = model
        self._batch_size = batch_size
        self._model: Any = None

    @property
    def name(self) -> str:
        return self._model_id

    @property
    def dim(self) -> int:
        return _MODEL_DIMS.get(self._model_id, 384)

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = SentenceTransformer(self._model_id, device=device)
        self._model.eval()

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        import asyncio

        def _do() -> list[list[float]]:
            self._load()
            raw = self._model.encode(
                texts,
                batch_size=self._batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return [list(map(float, vec)) for vec in raw]

        return await asyncio.to_thread(_do)

    async def embed_query(self, text: str) -> list[float]:
        import asyncio

        def _do() -> list[float]:
            self._load()
            vec = self._model.encode(
                text,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return list(map(float, vec))

        return await asyncio.to_thread(_do)
