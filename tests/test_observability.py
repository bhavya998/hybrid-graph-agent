"""Tests for the observability module — PII redaction + tracer no-op fallback."""

from __future__ import annotations

from hga.observability import redact
from hga.observability.tracing import _NoopSpan, get_tracer


def test_redact_email() -> None:
    out = redact("contact me at john.doe@example.com please")
    assert "[EMAIL]" in out
    assert "john.doe@example.com" not in out


def test_redact_phone() -> None:
    out = redact("call +1 (555) 123-4567 now")
    assert "[PHONE]" in out


def test_redact_credential() -> None:
    out = redact("api_key=sk-ant-deadbeef12345")
    assert "[CREDENTIAL]" in out


def test_redact_nested_dict() -> None:
    payload = {"user": "x", "contact": "a@b.com", "nested": {"phone": "555-010-0000"}}
    out = redact(payload)
    assert "[EMAIL]" in out["contact"]
    assert "[PHONE]" in out["nested"]["phone"]


def test_redact_passthrough_nonstring() -> None:
    assert redact(42) == 42
    assert redact(None) is None
    redacted_list = redact([1, "a@b.com"])
    assert redacted_list[0] == 1
    assert "[EMAIL]" in redacted_list[1]


def test_tracer_noop_without_keys() -> None:
    tracer = get_tracer()
    span = tracer.span("trace-1", "test-node", metadata={"query": "hello"})
    assert isinstance(span, _NoopSpan)
    with span:
        pass
