"""Langfuse observability — span tracing for every agent node.

Every LangGraph node is wrapped to emit a Langfuse span carrying the
``trace_id`` from ``AgentState``. A PII redaction filter runs before any payload
leaves the process. If Langfuse keys are absent, tracing degrades to a no-op
so the system runs without an external dependency.
"""

from __future__ import annotations

import functools
import hashlib
import re
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

_T = TypeVar("_T")

_PII_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[EMAIL]"),
    (re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[PHONE]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    (re.compile(r"(?i)(api[_-]?key|token|password|secret)\s*[=:]\s*\S+"), "[CREDENTIAL]"),
]


def redact(obj: Any) -> Any:
    """Recursively redact PII patterns in strings within dicts/lists."""
    if isinstance(obj, str):
        redacted = obj
        for pattern, replacement in _PII_PATTERNS:
            redacted = pattern.sub(replacement, redacted)
        return redacted
    if isinstance(obj, dict):
        return {k: redact(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact(item) for item in obj]
    return obj


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


class _NoopSpan:
    """Fallback when Langfuse keys are not configured."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self) -> _NoopSpan:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def end(self) -> None:
        pass


class Tracer:
    """Langfuse-backed tracer with a no-op fallback when keys are missing."""

    def __init__(self) -> None:
        import os

        self._public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
        self._secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")
        self._host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
        self._client: Any = None
        self._enabled = bool(self._public_key and self._secret_key)

    def _ensure_client(self) -> Any:
        if not self._enabled:
            return None
        if self._client is None:
            from langfuse import Langfuse

            self._client = Langfuse(
                public_key=self._public_key,
                secret_key=self._secret_key,
                host=self._host,
            )
        return self._client

    def trace(self, trace_id: str, name: str, user_id: str | None = None) -> Any:
        """Start a top-level trace. Returns a trace handle or noop."""
        client = self._ensure_client()
        if client is None:
            return _NoopSpan(name)
        return client.trace(id=trace_id, name=name, user_id=user_id)

    def span(self, trace_id: str, name: str, metadata: dict[str, Any] | None = None) -> Any:
        """Create a named span within a trace."""
        client = self._ensure_client()
        if client is None:
            return _NoopSpan(name)
        return client.span(
            trace_id=trace_id,
            name=name,
            metadata=redact(metadata or {}),
        )


_tracer: Tracer | None = None


def get_tracer() -> Tracer:
    global _tracer  # noqa: PLW0603
    if _tracer is None:
        _tracer = Tracer()
    return _tracer


def traced(
    name: str,
) -> Callable[[Callable[..., Awaitable[_T]]], Callable[..., Awaitable[_T]]]:
    """Decorator: wrap an async node fn to emit a Langfuse span from state['trace_id']."""

    def decorator(fn: Callable[..., Awaitable[_T]]) -> Callable[..., Awaitable[_T]]:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> _T:
            state = args[0] if args else kwargs.get("state", {})
            trace_id = (
                str(state.get("trace_id", "no-trace"))
                if isinstance(state, dict)
                else "no-trace"
            )
            tracer = get_tracer()
            span = tracer.span(trace_id, name, metadata={"query": str(state.get("query", ""))})
            with span:
                return await fn(*args, **kwargs)

        return wrapper

    return decorator
