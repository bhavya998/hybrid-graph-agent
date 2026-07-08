"""Anthropic LLM client — Claude via AsyncAnthropic.

Structured output uses forced tool-use (`tool_choice`), which is stable across
SDK versions. The newer `messages.parse(output_format=...)` is available but its
return attribute names vary; tool-forcing is the most portable choice.

Calls are built via a kwargs dict so the SDK's strict overload unions are
satisfied without per-arg casts.
"""

from __future__ import annotations

import os
from typing import Any

import anthropic

from hga.llm.base import LLMClient


def _to_params(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Normalize our Message TypedDict to Anthropic's MessageParam shape."""
    return [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages]


class AnthropicLLMClient(LLMClient):
    """Async Anthropic client. Reads ANTHROPIC_API_KEY from the environment."""

    def __init__(
        self,
        default_model: str = "claude-sonnet-4-5-20250929",
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._default_model = default_model
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            timeout=timeout,
        )

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": model or self._default_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": _to_params(messages),
        }
        if system is not None:
            kwargs["system"] = system
        resp = await self._client.messages.create(**kwargs)
        return "".join(
            getattr(b, "text", "")
            for b in resp.content
            if getattr(b, "type", None) == "text"
        )

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
        kwargs: dict[str, Any] = {
            "model": model or self._default_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": _to_params(messages),
            "tools": [
                {
                    "name": "respond",
                    "description": "Respond with the structured data requested.",
                    "input_schema": schema,
                }
            ],
            "tool_choice": {"type": "tool", "name": "respond"},
        }
        if system is not None:
            kwargs["system"] = system
        resp = await self._client.messages.create(**kwargs)
        for block in resp.content:
            is_tool = getattr(block, "type", None) == "tool_use"
            named = getattr(block, "name", "") == "respond"
            if is_tool and named:
                inp = getattr(block, "input", None)
                if isinstance(inp, dict):
                    return inp
        raise ValueError("Anthropic returned no tool_use block for structured output")
