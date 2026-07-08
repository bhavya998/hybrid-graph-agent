"""LLM package — interface + factory."""

from __future__ import annotations

from hga.config import Settings, get_settings
from hga.llm.anthropic_client import AnthropicLLMClient
from hga.llm.base import LLMClient
from hga.llm.local_llm import LocalTransformersLLM

__all__ = ["LLMClient", "AnthropicLLMClient", "LocalTransformersLLM", "get_llm_client"]


def get_llm_client(settings: Settings | None = None) -> LLMClient:
    """Return the LLM client: local Qwen 1.5B 4-bit for GPU, Anthropic for API."""
    s = settings or get_settings()
    if s.profile == "local_gpu":
        return LocalTransformersLLM(
            model_id=s.local_llm_model,
            quantize_4bit=s.local_llm_4bit,
        )
    if s.profile == "local":
        return AnthropicLLMClient(default_model=s.planner_model)
    raise NotImplementedError(f"LLM profile {s.profile!r} not yet implemented")
