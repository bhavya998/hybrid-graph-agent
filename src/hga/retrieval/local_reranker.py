"""Local GPU reranker — HuggingFace cross-encoder, no API.

Uses BAAI/bge-reranker-base (280M params), a strong lightweight cross-encoder
that runs efficiently on CPU or GPU. Batched scoring of (query, document) pairs.
"""

from __future__ import annotations

from typing import Any

from hga.retrieval.reranker import Reranker, RerankResult


class LocalCrossEncoderReranker(Reranker):
    """HuggingFace cross-encoder reranker running locally."""

    def __init__(self, model: str = "BAAI/bge-reranker-base") -> None:
        self._model_id = model
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: str = "cpu"

    @property
    def name(self) -> str:
        return self._model_id

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        self._model = AutoModelForSequenceClassification.from_pretrained(self._model_id)
        self._model.to(self._device)
        self._model.eval()

    async def rerank(self, query: str, documents: list[str], top_k: int) -> list[RerankResult]:
        import asyncio

        def _do() -> list[RerankResult]:
            if not documents:
                return []
            self._load()
            pairs = [[query, doc] for doc in documents]
            inputs = self._tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")
            import torch

            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                scores = self._model(**inputs, return_dict=True).logits.view(-1).float()
            cpu_scores = scores.cpu()
            indexed = [
                (i, float(score), documents[i])
                for i, score in enumerate(cpu_scores.tolist())
            ]
            indexed.sort(key=lambda x: x[1], reverse=True)
            top_n = min(top_k, len(indexed))
            return [
                RerankResult(index=i, relevance_score=score, document=doc)
                for i, score, doc in indexed[:top_n]
            ]

        return await asyncio.to_thread(_do)
