"""
Tests for the bridge from :class:`LLMCall` to a span:

- ``started_at`` propagates through to the span's recorded start time so the
  dashboard sees the real HTTP duration, not the microseconds we spent emitting.
- When the parser couldn't extract a ``model_id``, we emit under
  ``span_type="custom"`` instead of ``"llm"`` so the backend (which rejects
  ``llm`` without a model) still accepts the span.
"""

from __future__ import annotations

import time
from datetime import timedelta
from typing import Any

import pytest

from tracium.core.client import TraciumClient
from tracium.integrations.http_capture.emit import emit_llm_span
from tracium.integrations.http_capture.providers import LLMCall
from tracium.utils.datetime_utils import _utcnow


def _patched_recorder(
    client: TraciumClient, monkeypatch: pytest.MonkeyPatch
) -> list[dict[str, Any]]:
    recorded: list[dict[str, Any]] = []
    original = client.record_agent_spans

    def capture(trace_id: str, payloads: list[dict[str, Any]]) -> Any:
        recorded.extend(payloads)
        return original(trace_id, payloads)

    monkeypatch.setattr(client, "record_agent_spans", capture)
    return recorded


def test_started_at_propagates_to_span(
    tracium_client: TraciumClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """latency_ms on the wire should reflect started_at -> now, not ~0."""
    recorded = _patched_recorder(tracium_client, monkeypatch)
    started_at = _utcnow() - timedelta(milliseconds=750)

    with tracium_client.agent_trace(agent_name="latency-test"):
        emit_llm_span(
            LLMCall(
                provider="openai",
                model="gpt-4",
                operation="chat",
                input=[{"role": "user", "content": "hi"}],
                output="hello",
                input_tokens=5,
                output_tokens=2,
            ),
            started_at=started_at,
        )

    completed = [p for p in recorded if p.get("status") == "completed"]
    assert completed, "expected a completed span payload"
    latency = completed[-1].get("latency_ms")
    assert isinstance(latency, int)
    assert latency >= 700, f"expected latency ~750 ms; got {latency}"


def test_model_missing_falls_back_to_custom_span_type(
    tracium_client: TraciumClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Backend rejects llm spans without model_id, so we use a different type."""
    recorded = _patched_recorder(tracium_client, monkeypatch)

    with tracium_client.agent_trace(agent_name="no-model-test"):
        emit_llm_span(
            LLMCall(provider="unknown", model=None, operation="unknown", output="x"),
            started_at=_utcnow(),
        )

    assert recorded, "expected at least one payload"
    assert all(p.get("span_type") == "custom" for p in recorded)


def test_emit_with_model_uses_llm_span_type(
    tracium_client: TraciumClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded = _patched_recorder(tracium_client, monkeypatch)

    with tracium_client.agent_trace(agent_name="with-model"):
        emit_llm_span(
            LLMCall(provider="anthropic", model="claude-3", operation="chat", output="ok"),
            started_at=_utcnow(),
        )

    assert recorded
    assert all(p.get("span_type") == "llm" for p in recorded)


def test_emit_no_active_trace_is_no_op(tracium_client: TraciumClient) -> None:
    """``emit_llm_span`` must silently no-op when no trace is active."""
    # No agent_trace context -> no current_trace.
    emit_llm_span(
        LLMCall(provider="openai", model="gpt-4", output="x"),
        started_at=_utcnow(),
    )  # must not raise


def test_emit_error_marks_span_failed(
    tracium_client: TraciumClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded = _patched_recorder(tracium_client, monkeypatch)
    with tracium_client.agent_trace(agent_name="emit-error"):
        emit_llm_span(
            LLMCall(
                provider="openai",
                model="gpt-4",
                operation="chat",
                error="429: rate limited",
            ),
            started_at=_utcnow(),
        )

    failed = [p for p in recorded if p.get("status") == "failed"]
    assert failed, f"expected failed payload; got statuses {[p.get('status') for p in recorded]}"
    assert "rate limited" in (failed[-1].get("error") or "")


def test_started_at_field_on_payload(
    tracium_client: TraciumClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The span's ``started_at`` field on the wire should match what we passed."""
    recorded = _patched_recorder(tracium_client, monkeypatch)
    started_at = _utcnow() - timedelta(seconds=2)

    with tracium_client.agent_trace(agent_name="started-at-test"):
        emit_llm_span(
            LLMCall(provider="openai", model="gpt-4", operation="chat", output="ok"),
            started_at=started_at,
        )

    completed = [p for p in recorded if p.get("status") == "completed"]
    assert completed
    payload = completed[-1]
    # started_at is serialized as ISO 8601 string by _build_base_payload.
    assert isinstance(payload.get("started_at"), str)
    assert payload["started_at"].startswith(started_at.strftime("%Y-%m-%dT%H:%M"))


def test_emit_does_not_block_on_backend_errors(
    tracium_client: TraciumClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even if record_agent_spans raises, emit must return cleanly."""

    def boom(_trace_id: str, _payloads: list[Any]) -> Any:
        raise RuntimeError("backend down")

    monkeypatch.setattr(tracium_client, "record_agent_spans", boom)

    with tracium_client.agent_trace(agent_name="resilience"):
        # Must not raise even though the backend call would fail.
        emit_llm_span(
            LLMCall(provider="openai", model="gpt-4", output="hi"),
            started_at=_utcnow(),
        )
    # If we got here, the SDK didn't crash the user's app.


# Real-time sanity: started_at and completed_at order is respected
def test_emit_chronological_order(
    tracium_client: TraciumClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded = _patched_recorder(tracium_client, monkeypatch)
    started_at = _utcnow()
    time.sleep(0.005)

    with tracium_client.agent_trace(agent_name="order-test"):
        emit_llm_span(
            LLMCall(provider="openai", model="gpt-4", output="ok", input_tokens=1, output_tokens=1),
            started_at=started_at,
        )

    completed = [p for p in recorded if p.get("status") == "completed"]
    assert completed
    payload = completed[-1]
    assert payload["started_at"] <= payload["completed_at"]
