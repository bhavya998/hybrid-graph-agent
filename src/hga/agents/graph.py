"""LangGraph agent graph — Router → Retriever → Critic → Synthesizer.

Retry loop: if the Critic signals low quality, the graph re-enters the Retriever
(up to `max_critique_retries` re-entries). Otherwise it routes to the Synthesizer.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

from langgraph.graph import END, StateGraph

from hga.agents.critic import critique
from hga.agents.retriever import retrieve
from hga.agents.router import route_query
from hga.agents.state import AgentState, Route
from hga.agents.synthesizer import synthesize
from hga.config import Settings, get_settings
from hga.embeddings import get_text_embedder
from hga.graph import get_graph_store
from hga.llm import get_llm_client
from hga.observability import traced
from hga.retrieval import get_reranker, get_vector_store
from hga.tools.mcp_client import MCPToolManager


def _post_router(state: AgentState) -> str:
    route = state.get("route", Route.HYBRID)
    if route == Route.TOOL:
        return "tool_executor"
    return "retrieve"


def _post_critic(state: AgentState) -> Literal["retrieve", "synthesize"]:
    feedback = state.get("critic_feedback")
    if feedback and feedback.needs_retry:
        return "retrieve"
    return "synthesize"


def _make_tool_executor(
    tool_mgr: MCPToolManager,
) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    """Build a tool_executor node that calls real MCP tools."""

    async def _exec(state: AgentState) -> dict[str, Any]:
        query = state.get("query", "")
        sub_queries = state.get("sub_queries", [query])
        search_query = sub_queries[-1] if sub_queries else query

        # Simple intent routing: keywords → tool choice
        ql = search_query.lower()
        if "list" in ql and ("paper" in ql or "corpus" in ql or "all" in ql):
            result = await tool_mgr.call_tool("list_papers")
        elif "search corpus" in ql or "find paper" in ql:
            result = await tool_mgr.call_tool("search_corpus", {"query": search_query})
        elif "search chunk" in ql or "find in text" in ql:
            result = await tool_mgr.call_tool("search_chunks", {"query": search_query})
        elif "read repo" in ql or "repo file" in ql:
            result = await tool_mgr.call_tool(
                "read_repo_file",
                {"repo_slug": "unknown", "file_path": "README.md"},
            )
        else:
            # Default: search the corpus for whatever was asked
            result = await tool_mgr.call_tool("search_corpus", {"query": search_query})

        return {"tool_results": [result]}

    return _exec


def build_graph(settings: Settings | None = None) -> StateGraph[AgentState]:
    """Build a compile-ready LangGraph agent graph."""
    s = settings or get_settings()
    llm = get_llm_client(s)
    store = get_vector_store(s)
    graph_store = get_graph_store(s)
    embedder = get_text_embedder(s)
    reranker = get_reranker(s)
    tool_mgr = MCPToolManager()

    builder: StateGraph[AgentState] = StateGraph(AgentState)
    tool_executor = _make_tool_executor(tool_mgr)

    async def _route(state: AgentState) -> dict[str, Any]:
        return await route_query(state, llm)

    async def _retrieve(state: AgentState) -> dict[str, Any]:
        return await retrieve(state, store, graph_store, embedder, reranker, s)

    async def _critique(state: AgentState) -> dict[str, Any]:
        return await critique(state, llm, s)

    async def _synthesize(state: AgentState) -> dict[str, Any]:
        return await synthesize(state, llm)

    builder.add_node("router", traced("router")(_route))
    builder.add_node("retrieve", traced("retrieve")(_retrieve))
    builder.add_node("critic", traced("critic")(_critique))
    builder.add_node("synthesize", traced("synthesize")(_synthesize))
    builder.add_node("tool_executor", traced("tool_executor")(tool_executor))

    builder.set_entry_point("router")
    builder.add_conditional_edges("router", _post_router, {
        "retrieve": "retrieve",
        "tool_executor": "tool_executor",
    })
    builder.add_edge("retrieve", "critic")
    builder.add_conditional_edges("critic", _post_critic, {
        "retrieve": "retrieve",
        "synthesize": "synthesize",
    })
    builder.add_edge("synthesize", END)
    builder.add_edge("tool_executor", "synthesize")
    return builder


def compile_graph(settings: Settings | None = None) -> Any:
    """Compile the agent graph (no persistence — add checkpointer in Day 4)."""
    return build_graph(settings).compile()


class AgentRunner:
    """High-level wrapper: run a query through the compiled graph."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._graph = compile_graph(settings=self._settings)

    async def ask(self, query: str, session_id: str | None = None) -> AgentState:
        sid = session_id or uuid.uuid4().hex[:12]
        trace_id = uuid.uuid4().hex
        initial: AgentState = {
            "query": query,
            "session_id": sid,
            "trace_id": trace_id,
        }
        result: Any = await self._graph.ainvoke(
            initial, {"configurable": {"thread_id": sid}}
        )
        return cast("AgentState", result)
