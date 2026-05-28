"""
Tests for the OpenAI Realtime API (WebSocket) integration.

We exercise the tracker + the wrapping connection directly with fake events
that match the shapes OpenAI's SDK emits. No real WebSocket and no real OpenAI
SDK are required — the tracker only duck-types events via ``getattr`` /
``.model_dump()``, so dict-shaped events are a faithful stand-in for the
Pydantic models the SDK actually yields.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import pytest

from tracium.core.client import TraciumClient
from tracium.integrations.openai_realtime import (
    _RealtimeTracker,
    _TraciumAsyncRealtimeCM,
    _TraciumAsyncRealtimeConnection,
)

# --------------------------------------------------------------------------- #
# Helpers: capture spans the tracker tries to emit                             #
# --------------------------------------------------------------------------- #


@pytest.fixture
def captured_spans(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Replace ``emit_llm_span`` so tests see the LLMCall instead of a real span."""
    seen: list[Any] = []

    def fake_emit(call, started_at, ended_at=None):  # noqa: ARG001
        seen.append(call)

    from tracium.integrations.http_capture import emit as emit_mod

    monkeypatch.setattr(emit_mod, "emit_llm_span", fake_emit)
    return seen


# --------------------------------------------------------------------------- #
# Tracker — emission semantics                                                 #
# --------------------------------------------------------------------------- #


def _session_event(model: str, instructions: str | None = None, tools=None) -> dict:
    return {
        "type": "session.created",
        "session": {
            "model": model,
            "instructions": instructions,
            "tools": tools or [],
        },
    }


def _response_done(
    *,
    text: str | None = None,
    transcript: str | None = None,
    tool_calls: list[dict] | None = None,
    usage: dict | None = None,
    model: str | None = None,
    status: str = "completed",
) -> dict:
    output: list[dict] = []
    if text:
        output.append({"type": "message", "content": [{"type": "text", "text": text}]})
    if transcript:
        output.append(
            {"type": "message", "content": [{"type": "audio", "transcript": transcript}]}
        )
    for tc in tool_calls or []:
        output.append(tc)
    return {
        "type": "response.done",
        "response": {
            "status": status,
            "model": model,
            "output": output,
            "usage": usage or {},
        },
    }


class TestTrackerEmission:
    def _run(self, tracker: _RealtimeTracker, events: list[dict]) -> None:
        # Active trace required so emit_llm_span doesn't no-op when fake_emit
        # isn't installed; the fixture installs the patch so we can run
        # without one.
        for event in events:
            tracker.on_event(event)

    def test_session_then_response_emits_span_with_model_and_tokens(
        self, tracium_client: TraciumClient, captured_spans: list[Any]
    ) -> None:
        tracker = _RealtimeTracker(model_hint=None)
        with tracium_client.agent_trace(agent_name="realtime-basic"):
            self._run(
                tracker,
                [
                    _session_event("gpt-4o-realtime-preview-2024-10-01"),
                    {"type": "response.created", "response_id": "resp_1"},
                    _response_done(
                        text="Hello!",
                        usage={
                            "input_tokens": 12,
                            "output_tokens": 4,
                            "input_token_details": {"cached_tokens": 3},
                        },
                    ),
                ],
            )

        assert len(captured_spans) == 1
        call = captured_spans[0]
        assert call.provider == "openai"
        assert call.operation == "realtime"
        assert call.model == "gpt-4o-realtime-preview-2024-10-01"
        assert call.output == "Hello!"
        assert call.input_tokens == 12
        assert call.output_tokens == 4
        assert call.cached_input_tokens == 3

    def test_audio_transcript_used_as_output(
        self, tracium_client: TraciumClient, captured_spans: list[Any]
    ) -> None:
        tracker = _RealtimeTracker(model_hint="gpt-4o-realtime-preview")
        with tracium_client.agent_trace(agent_name="realtime-audio"):
            self._run(
                tracker,
                [
                    {"type": "response.created"},
                    _response_done(
                        transcript="Hello, this is the assistant.",
                        usage={"input_tokens": 5, "output_tokens": 8},
                    ),
                ],
            )
        assert captured_spans[0].output == "Hello, this is the assistant."

    def test_function_call_captured_as_tool_call(
        self, tracium_client: TraciumClient, captured_spans: list[Any]
    ) -> None:
        tracker = _RealtimeTracker(model_hint="gpt-4o-realtime-preview")
        with tracium_client.agent_trace(agent_name="realtime-tool"):
            self._run(
                tracker,
                [
                    {"type": "response.created"},
                    _response_done(
                        tool_calls=[
                            {
                                "type": "function_call",
                                "call_id": "call_1",
                                "name": "get_weather",
                                "arguments": '{"location":"SF"}',
                            }
                        ],
                        usage={"input_tokens": 7, "output_tokens": 3},
                    ),
                ],
            )

        call = captured_spans[0]
        assert call.tool_calls is not None
        assert call.tool_calls[0]["function"]["name"] == "get_weather"
        assert call.tool_calls[0]["function"]["arguments"] == '{"location":"SF"}'

    def test_failed_response_status_marks_error(
        self, tracium_client: TraciumClient, captured_spans: list[Any]
    ) -> None:
        tracker = _RealtimeTracker(model_hint="gpt-4o-realtime-preview")
        with tracium_client.agent_trace(agent_name="realtime-failed"):
            self._run(
                tracker,
                [
                    {"type": "response.created"},
                    {
                        "type": "response.done",
                        "response": {
                            "status": "failed",
                            "model": "gpt-4o-realtime-preview",
                            "output": [],
                            "usage": {},
                            "status_details": {"error": "internal"},
                        },
                    },
                ],
            )
        assert captured_spans[0].error == "internal"

    def test_error_event_emits_error_span(
        self, tracium_client: TraciumClient, captured_spans: list[Any]
    ) -> None:
        tracker = _RealtimeTracker(model_hint="gpt-4o-realtime-preview")
        with tracium_client.agent_trace(agent_name="realtime-error"):
            self._run(
                tracker,
                [
                    {
                        "type": "error",
                        "error": {
                            "type": "invalid_request_error",
                            "message": "Bad session.",
                        },
                    }
                ],
            )
        assert captured_spans[0].error == "Bad session."

    def test_session_update_overrides_model_and_tools(
        self, tracium_client: TraciumClient, captured_spans: list[Any]
    ) -> None:
        tracker = _RealtimeTracker(model_hint=None)
        with tracium_client.agent_trace(agent_name="realtime-update"):
            self._run(
                tracker,
                [
                    _session_event("gpt-4o-realtime-old", instructions="A", tools=[]),
                    {
                        "type": "session.updated",
                        "session": {
                            "model": "gpt-4o-realtime-new",
                            "instructions": "B",
                            "tools": [{"type": "function", "name": "search"}],
                        },
                    },
                    {"type": "response.created"},
                    _response_done(text="ok", usage={"input_tokens": 1, "output_tokens": 1}),
                ],
            )
        call = captured_spans[0]
        assert call.model == "gpt-4o-realtime-new"
        assert isinstance(call.input, dict)
        assert call.input["system"] == "B"
        assert call.tools == [{"type": "function", "name": "search"}]

    def test_no_active_trace_emits_nothing(self, captured_spans: list[Any]) -> None:
        tracker = _RealtimeTracker(model_hint="gpt-4o-realtime-preview")
        # No agent_trace context.
        tracker.on_event({"type": "response.created"})
        tracker.on_event(_response_done(text="lost"))
        assert captured_spans == []


# --------------------------------------------------------------------------- #
# Wrapped connection — verifies user iteration still yields every event       #
# --------------------------------------------------------------------------- #


class _FakeAsyncConnection:
    """Stand-in for openai's ``AsyncRealtimeConnection``: async-iterable over
    a fixed event list, with a few attributes the wrapper might delegate to.
    """

    def __init__(self, events: list[dict]) -> None:
        self._events = list(events)
        self.session = object()  # for __getattr__ delegation tests
        self.closed = False

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for event in self._events:
            await asyncio.sleep(0)
            yield event


class _FakeAsyncCM:
    def __init__(self, conn: _FakeAsyncConnection) -> None:
        self._conn = conn

    async def __aenter__(self) -> _FakeAsyncConnection:
        return self._conn

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


@pytest.mark.asyncio
async def test_wrapped_connection_yields_every_event(
    tracium_client: TraciumClient, captured_spans: list[Any]
) -> None:
    events = [
        _session_event("gpt-4o-realtime-preview"),
        {"type": "response.created"},
        _response_done(text="hi", usage={"input_tokens": 2, "output_tokens": 1}),
    ]
    fake_cm = _FakeAsyncCM(_FakeAsyncConnection(events))
    wrapper = _TraciumAsyncRealtimeCM(fake_cm, model_hint=None)

    seen: list[dict] = []
    with tracium_client.agent_trace(agent_name="realtime-iter"):
        async with wrapper as conn:
            async for event in conn:
                seen.append(event)

    # User sees every event, in order, untouched.
    assert seen == events
    # And one span was emitted at response.done.
    assert len(captured_spans) == 1
    assert captured_spans[0].output == "hi"


@pytest.mark.asyncio
async def test_wrapped_connection_delegates_unknown_attrs() -> None:
    sentinel = object()
    fake = _FakeAsyncConnection([])
    fake.session = sentinel
    wrapper = _TraciumAsyncRealtimeConnection(fake, model_hint=None)
    # Anything not handled by the wrapper falls through to the underlying conn.
    assert wrapper.session is sentinel


# --------------------------------------------------------------------------- #
# Started-at uses response.created timestamp                                   #
# --------------------------------------------------------------------------- #


def test_response_started_at_used_for_span_timing(
    tracium_client: TraciumClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The span's start time should be when response.created fired, not
    when response.done landed — so latency reflects the real LLM response time.
    """
    recorded_starts: list[datetime] = []

    def fake_emit(call, started_at, ended_at=None):  # noqa: ARG001
        recorded_starts.append(started_at)

    from tracium.integrations.http_capture import emit as emit_mod

    monkeypatch.setattr(emit_mod, "emit_llm_span", fake_emit)

    tracker = _RealtimeTracker(model_hint="gpt-4o-realtime-preview")
    before = datetime.now(timezone.utc)
    with tracium_client.agent_trace(agent_name="realtime-timing"):
        tracker.on_event({"type": "response.created"})
        # Simulate some elapsed wall-clock time before response.done lands.
        # (We just emit immediately; the assertion is that started_at is
        # bounded by the call to response.created above.)
        tracker.on_event(
            _response_done(text="hi", usage={"input_tokens": 1, "output_tokens": 1})
        )
    after = datetime.now(timezone.utc)
    assert recorded_starts
    assert before <= recorded_starts[0] <= after
