"""Agent state schema — the single TypedDict every LangGraph node reads/writes."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel


class Route(StrEnum):
    """Adaptive router output — selects the retrieval path."""

    VECTOR = "vector"
    HYBRID = "hybrid"
    GRAPH_LOCAL = "graph_local"
    GRAPH_GLOBAL = "graph_global"
    TOOL = "tool"
    AGENTIC = "agentic"


class CriticFeedback(BaseModel):
    """CRAG-style critic output — drives self-correction."""

    faithfulness: float
    sufficiency: float
    coverage: float
    needs_retry: bool
    feedback: str


def _merge_lists(left: list[Any], right: list[Any]) -> list[Any]:
    """Reducer: concatenate lists across node returns (deduping later)."""
    return (left or []) + (right or [])


class AgentState(TypedDict, total=False):
    """LangGraph state. Nodes return partial dicts; reducers merge.

    Use `Annotated[list[X], add]` reducers so parallel/iterated nodes accumulate
    rather than overwrite. Mirrors the spec in AGENTS.md.
    """

    query: str
    session_id: str
    rewritten_queries: Annotated[list[str], _merge_lists]
    route: Route
    sub_queries: list[str]
    retrieved_chunks: Annotated[list[dict[str, Any]], _merge_lists]
    reranked_chunks: list[dict[str, Any]]
    graph_context: dict[str, Any]
    tool_calls: Annotated[list[dict[str, Any]], _merge_lists]
    tool_results: Annotated[list[dict[str, Any]], _merge_lists]
    draft_answer: str
    citations: list[Any]
    critic_feedback: CriticFeedback
    retry_count: int
    trace_id: str
    messages: Annotated[list[Any], _merge_lists]


__all__ = ["Route", "CriticFeedback", "AgentState"]
