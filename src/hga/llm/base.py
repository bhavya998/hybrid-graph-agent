"""LLM client interface — provider-agnostic, selected by config profile.

Agent-graph nodes use langchain-anthropic directly (LangGraph-native). This
interface covers bespoke async LLM calls outside the graph: contextual
retrieval summaries, the eval LLM-as-judge, and the adaptive router.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Async LLM interface for non-agent graph tasks."""

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """Return the assistant's text response."""
        ...

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        *,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Return a structured object matching the JSON schema (tool-forced)."""
        ...
