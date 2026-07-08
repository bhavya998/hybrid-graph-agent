"""Synthesizer node — generates the final answer with citations.

Takes retrieved chunks + graph context and produces a clean, cited response.
Citation format: [doc_id:chunk_id] with inline spans.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hga.agents.state import AgentState
    from hga.llm.base import LLMClient

_SYNTH_SYSTEM = """You are a research assistant with access to retrieved papers and code.
Answer the user's question using ONLY the provided context. Cite every factual
statement with [N] where N is the context item number. Do not invent anything.
If the context does not contain the answer, say so."""


async def synthesize(state: AgentState, llm: LLMClient) -> dict[str, Any]:
    """Produce the final answer with citations."""
    query = state.get("query", "")
    chunks = state.get("retrieved_chunks", [])
    graph = state.get("graph_context", {})

    parts: list[str] = []
    sources: list[dict[str, Any]] = []

    if chunks:
        for i, c in enumerate(chunks[:8], 1):
            payload = c.get("payload", {})
            content = payload.get("content", "")[:1000]
            parts.append(f"[{i}] {content}")
            sources.append({"chunk_id": c.get("chunk_id", ""), "doc_id": payload.get("doc_id", ""),
                            "page": payload.get("page")})

    if graph.get("answer"):
        parts.append(graph["answer"])

    if not parts:
        return {"draft_answer": "No context available to answer this query."}

    context = "\n\n".join(parts)
    prompt = f"Query: {query}\n\nContext:\n{context}\n\nAnswer with citations:"
    messages = [{"role": "user", "content": prompt}]
    try:
        answer = await llm.complete(
            messages, system=_SYNTH_SYSTEM, max_tokens=1024, temperature=0.0
        )
    except Exception:
        parts_list = ", ".join(s.get("chunk_id", "") for s in sources[:3])
        answer = f"Context retrieved but LLM call failed. Sources: {parts_list}"

    citations: list[dict[str, Any]] = []
    for s in sources:
        citations.append({
            "chunk_id": s["chunk_id"],
            "doc_id": s["doc_id"],
            "page": s["page"],
            "span": "",
            "score": 0.0,
        })
    return {"draft_answer": answer, "citations": citations}
