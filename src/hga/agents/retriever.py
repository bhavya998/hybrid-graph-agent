"""Retriever node — executes the chosen route against stores + reranker.

Routes:
  vector     → dense ANN
  hybrid     → dense + BM25 sparse, RRF fuse, cross-encoder rerank
  graph_*    → Kùzu local / global / drift search
  tool       → delegates to MCP tool layer (tool calls are accumulated)
  agentic    → decomposes via Planner node first (re-enters after planning)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hga.agents.state import AgentState, Route

if TYPE_CHECKING:
    from hga.config import Settings
    from hga.embeddings.base import Embedder
    from hga.graph.store import GraphStore
    from hga.retrieval.reranker import Reranker
    from hga.retrieval.store import VectorStore


async def retrieve(
    state: AgentState,
    store: VectorStore,
    graph_store: GraphStore,
    embedder: Embedder,
    reranker: Reranker,
    settings: Settings,
) -> dict[str, Any]:
    """Run the appropriate retrieval based on `route` in state."""
    route = state.get("route", Route.HYBRID)
    query = state.get("query", "")
    sub_queries = state.get("sub_queries", [query])
    search_query = sub_queries[-1] if sub_queries else query

    if route in (Route.VECTOR, Route.HYBRID):
        dense = await embedder.embed_query(search_query)
        if route == Route.VECTOR:
            results = await store.search_dense(dense, k=settings.dense_top_k)
        else:
            results = await store.search_hybrid(
                dense, search_query, k=settings.rerank_top_k, rrf_k=settings.rrf_k
            )
        docs = [r.payload.get("content", "") for r in results]
        if docs and results:
            reranked = await reranker.rerank(search_query, docs, top_k=settings.rerank_top_k)
            # map rerank order back to results preserving original payload
            seen: set[str] = set()
            merged: list[dict[str, Any]] = []
            for rr in reranked:
                orig = results[rr.index] if rr.index < len(results) else None
                if orig and orig.chunk_id not in seen:
                    seen.add(orig.chunk_id)
                    merged.append({"chunk_id": orig.chunk_id, "score": rr.relevance_score,
                                   "payload": orig.payload})
            results_to_store = merged
        else:
            results_to_store = [
                {"chunk_id": r.chunk_id, "score": r.score, "payload": r.payload} for r in results
            ]
        return {"retrieved_chunks": results_to_store}

    if route in (Route.GRAPH_LOCAL, Route.GRAPH_GLOBAL):
        if route == Route.GRAPH_LOCAL:
            gs = await graph_store.local_search(search_query)
        else:
            gs = await graph_store.global_search(search_query)
        return {"graph_context": {"answer": gs.answer, "entities": [e.name for e in gs.entities],
                                  "communities": [c.title for c in gs.communities]}}

    if route == Route.TOOL:
        return {}
    return {}
