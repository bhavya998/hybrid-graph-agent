"""Contextual Retrieval — prepend a doc-grounding summary to each chunk.

Anthropic's technique: for each chunk, a cheap LLM generates a short context
situating the chunk within the source document. That context is prepended to the
chunk text before embedding, which materially improves retrieval. We use Haiku
and cap the document length per chunk to control cost.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hga.chunking.schema import Chunk
    from hga.llm.base import LLMClient

_PROMPT = """<document>
{doc}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Please give a short succinct context to situate this chunk within the overall
document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else."""


class Contextualizer:
    """Generates per-chunk context summaries via the configured LLM."""

    def __init__(
        self,
        llm: LLMClient,
        model: str,
        max_concurrency: int = 8,
        doc_char_limit: int = 12000,
        chunk_char_limit: int = 4000,
    ) -> None:
        self._llm = llm
        self._model = model
        self._sem = asyncio.Semaphore(max_concurrency)
        self._doc_char_limit = doc_char_limit
        self._chunk_char_limit = chunk_char_limit

    async def contextualize(self, doc_text: str, chunks: list[Chunk]) -> list[str]:
        """Return one context-summary string per chunk (empty on failure)."""
        doc = doc_text[: self._doc_char_limit]

        async def one(chunk: Chunk) -> str:
            async with self._sem:
                prompt = _PROMPT.format(doc=doc, chunk=chunk.content[: self._chunk_char_limit])
                try:
                    resp = await self._llm.complete(
                        [{"role": "user", "content": prompt}],
                        model=self._model,
                        max_tokens=200,
                        temperature=0.0,
                    )
                    return resp.strip()
                except Exception:  # noqa: BLE001
                    return ""

        return await asyncio.gather(*[one(c) for c in chunks])

    def apply(self, chunks: list[Chunk], contexts: list[str]) -> list[Chunk]:
        """Return new chunks with context_summary + embedding_text populated."""
        out: list[Chunk] = []
        for chunk, ctx in zip(chunks, contexts, strict=True):
            embedding_text = f"{ctx}\n\n{chunk.content}" if ctx else chunk.content
            out.append(
                chunk.model_copy(
                    update={"context_summary": ctx, "embedding_text": embedding_text}
                )
            )
        return out
