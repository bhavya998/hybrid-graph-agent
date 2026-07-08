"""Adaptive router — classifies a query into a retrieval route.

Uses a cheap LLM (Haiku) with structured JSON output to decide:
  vector | hybrid | graph_local | graph_global | tool | agentic

The "agentic" route signals a multi-step task that needs the LangGraph loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hga.agents.state import AgentState, Route

if TYPE_CHECKING:
    from hga.llm.base import LLMClient

_ROUTER_SCHEMA = {
    "type": "object",
    "properties": {
        "route": {
            "type": "string",
            "enum": [r.value for r in Route],
        },
        "reason": {"type": "string"},
    },
    "required": ["route", "reason"],
    "additionalProperties": False,
}

_ROUTER_SYSTEM = """You are a query classifier. Decide the best retrieval path:

- `vector`: simple factual question, single hop, definitions
- `hybrid`: specific facts that benefit from keyword + semantic search
- `graph_local`: entity-centric, relationships, "who wrote X?", "what cites Y?"
- `graph_global`: thematic / summarisation across the corpus
- `tool`: needs live data (web search, SQL, filesystem lookup)
- `agentic`: multi-step, complex reasoning, requires planning

Consider: arXiv AI papers (CAI/CL) + their GitHub repos."""


async def route_query(state: AgentState, llm: LLMClient) -> dict[str, Any]:
    """Classify the query and set `route` + `sub_queries` in state."""
    query = state.get("query", "")
    messages = [{"role": "user", "content": query}]
    result = await llm.complete_json(
        messages,
        schema=_ROUTER_SCHEMA,
        system=_ROUTER_SYSTEM,
        max_tokens=256,
        temperature=0.0,
    )
    chosen = result.get("route", "hybrid")
    return {"route": Route(chosen), "sub_queries": [query]}
