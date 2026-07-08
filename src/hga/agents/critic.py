"""CRAG critic — checks retrieval quality and decides whether to retry.

Uses a fast LLM (Haiku) with structured JSON to score faithfulness and
sufficiency. If the score is below threshold OR retry count is under the limit,
it signals a re-try (which the Supervisor routes back to Retriever with a
rewritten query).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hga.agents.state import AgentState, CriticFeedback

if TYPE_CHECKING:
    from hga.config import Settings
    from hga.llm.base import LLMClient

_CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "faithfulness": {"type": "number", "minimum": 0, "maximum": 1},
        "sufficiency": {"type": "number", "minimum": 0, "maximum": 1},
        "coverage": {"type": "number", "minimum": 0, "maximum": 1},
        "needs_retry": {"type": "boolean"},
        "feedback": {"type": "string"},
    },
    "required": ["faithfulness", "sufficiency", "coverage", "needs_retry", "feedback"],
    "additionalProperties": False,
}

_SUFFICIENCY_THRESHOLD = 0.6


async def critique(
    state: AgentState,
    llm: LLMClient,
    settings: Settings,
) -> dict[str, Any]:
    """Evaluate retrieval quality and set feedback."""
    query = state.get("query", "")
    chunks = state.get("retrieved_chunks", [])
    retry_count = state.get("retry_count", 0)
    graph = state.get("graph_context", {})

    context = ""
    if chunks:
        context = "\n\n---\n\n".join(
            c.get("payload", {}).get("content", "")[:800] for c in chunks[:5]
        )
    elif graph:
        context = graph.get("answer", "")[:2000]

    if not context:
        return {
            "critic_feedback": CriticFeedback(
                faithfulness=0.0,
                sufficiency=0.0,
                coverage=0.0,
                needs_retry=retry_count < settings.max_critique_retries,
                feedback="no context retrieved",
            ),
            "retry_count": retry_count + 1,
        }

    prompt = f"Query: {query}\n\nRetrieved context:\n{context}"
    messages = [{"role": "user", "content": prompt}]
    try:
        result = await llm.complete_json(
            messages,
            schema=_CRITIC_SCHEMA,
            max_tokens=256,
            temperature=0.0,
        )
    except Exception:
        return {
            "critic_feedback": CriticFeedback(
                faithfulness=0.5,
                sufficiency=0.5,
                coverage=0.5,
                needs_retry=retry_count < settings.max_critique_retries,
                feedback="critic eval failed",
            ),
            "retry_count": retry_count + 1,
        }

    needs = bool(result.get("needs_retry", False))
    hard_retry = retry_count < settings.max_critique_retries
    return {
        "critic_feedback": CriticFeedback(
            faithfulness=float(result.get("faithfulness", 0.5)),
            sufficiency=float(result.get("sufficiency", 0.5)),
            coverage=float(result.get("coverage", 0.5)),
            needs_retry=needs and hard_retry,
            feedback=str(result.get("feedback", "")),
        ),
        "retry_count": retry_count + 1,
    }
