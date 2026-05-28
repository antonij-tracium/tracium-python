"""
End-to-end tests for the httpx transport: verify that any HTTP call to a known
LLM endpoint produces an LLM span on the active trace, that streaming responses
are reconstructed correctly, and that ``owned_capture`` suppresses capture.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from tracium.core.client import TraciumClient
from tracium.integrations.http_capture.dedup import owned_capture
from tracium.integrations.http_capture.transport_httpx import (
    TraciumAsyncHTTPXTransport,
    TraciumHTTPXTransport,
)


def _build_openai_response(body: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200,
        json=body,
        headers={"content-type": "application/json"},
    )


def _build_openai_stream(events: list[str]) -> httpx.Response:
    payload = ("\n".join(events) + "\n").encode("utf-8")
    return httpx.Response(
        200,
        content=payload,
        headers={"content-type": "text/event-stream"},
    )


@pytest.fixture
def captured_spans(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Capture LLM spans the transport tries to emit instead of going through the trace."""
    seen: list[Any] = []

    def _fake_emit(call, started_at, ended_at=None):  # noqa: ARG001
        seen.append(call)

    from tracium.integrations.http_capture import emit as emit_mod
    from tracium.integrations.http_capture import transport_httpx

    monkeypatch.setattr(emit_mod, "emit_llm_span", _fake_emit)
    monkeypatch.setattr(transport_httpx, "emit_llm_span", _fake_emit)
    return seen


class TestSyncTransport:
    def test_captures_openai_chat(self, captured_spans: list[Any]) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return _build_openai_response(
                {
                    "choices": [{"message": {"role": "assistant", "content": "hi"}}],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 1},
                }
            )

        transport = TraciumHTTPXTransport(httpx.MockTransport(handler))
        with httpx.Client(transport=transport) as client:
            response = client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
            )

        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "hi"
        assert len(captured_spans) == 1
        call = captured_spans[0]
        assert call.provider == "openai"
        assert call.model == "gpt-4"
        assert call.output == "hi"
        assert call.input_tokens == 4

    def test_passes_through_non_llm(self, captured_spans: list[Any]) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"ok": True})

        transport = TraciumHTTPXTransport(httpx.MockTransport(handler))
        with httpx.Client(transport=transport) as client:
            client.get("https://example.com/api/users")
        assert captured_spans == []

    def test_owned_capture_suppresses_emit(self, captured_spans: list[Any]) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return _build_openai_response(
                {"choices": [{"message": {"content": "x"}}], "usage": {}}
            )

        transport = TraciumHTTPXTransport(httpx.MockTransport(handler))
        with httpx.Client(transport=transport) as client:
            with owned_capture():
                client.post(
                    "https://api.openai.com/v1/chat/completions",
                    json={"model": "gpt-4", "messages": []},
                )
        assert captured_spans == []

    def test_streaming_response_assembled(self, captured_spans: list[Any]) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return _build_openai_stream(
                [
                    'data: {"choices":[{"delta":{"content":"Hi"}}]}',
                    "",
                    'data: {"choices":[{"delta":{"content":" there"}}]}',
                    "",
                    'data: {"usage":{"prompt_tokens":5,"completion_tokens":2}}',
                    "",
                    "data: [DONE]",
                    "",
                ]
            )

        transport = TraciumHTTPXTransport(httpx.MockTransport(handler))
        with httpx.Client(transport=transport) as client:
            with client.stream(
                "POST",
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-4", "messages": [], "stream": True},
            ) as response:
                consumed = b"".join(response.iter_bytes())

        assert b"Hi" in consumed
        assert len(captured_spans) == 1
        call = captured_spans[0]
        assert call.output == "Hi there"
        assert call.input_tokens == 5
        assert call.output_tokens == 2

    def test_4xx_recorded_as_error(self, captured_spans: list[Any]) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                401,
                json={"error": {"message": "Invalid API key"}},
                headers={"content-type": "application/json"},
            )

        transport = TraciumHTTPXTransport(httpx.MockTransport(handler))
        with httpx.Client(transport=transport) as client:
            client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-4", "messages": []},
            )

        assert len(captured_spans) == 1
        assert captured_spans[0].error is not None
        assert "401" in captured_spans[0].error
        assert "Invalid API key" in captured_spans[0].error

    def test_transport_exception_emits_error_span(self, captured_spans: list[Any]) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("network down")

        transport = TraciumHTTPXTransport(httpx.MockTransport(handler))
        with httpx.Client(transport=transport) as client, pytest.raises(httpx.ConnectError):
            client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-4", "messages": []},
            )

        assert len(captured_spans) == 1
        assert "ConnectError" in captured_spans[0].error


class TestAsyncTransport:
    @pytest.mark.asyncio
    async def test_captures_anthropic_call(self, captured_spans: list[Any]) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "content": [{"type": "text", "text": "Hello"}],
                    "usage": {"input_tokens": 3, "output_tokens": 1},
                },
                headers={"content-type": "application/json"},
            )

        transport = TraciumAsyncHTTPXTransport(httpx.MockTransport(handler))
        async with httpx.AsyncClient(transport=transport) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                json={"model": "claude-3", "messages": [{"role": "user", "content": "hi"}]},
            )

        assert response.status_code == 200
        assert len(captured_spans) == 1
        assert captured_spans[0].provider == "anthropic"
        assert captured_spans[0].output == "Hello"


class TestInstall:
    def test_install_idempotent(self) -> None:
        from tracium.integrations.http_capture.transport_httpx import install

        install()
        install()  # second call must be a no-op

    def test_install_wraps_new_clients(self, captured_spans: list[Any]) -> None:
        """After install(), a freshly-constructed Client uses the wrapping transport."""
        from tracium.integrations.http_capture.transport_httpx import install

        install()

        def handler(request: httpx.Request) -> httpx.Response:
            return _build_openai_response(
                {"choices": [{"message": {"content": "wrapped"}}], "usage": {}}
            )

        # Using transport=MockTransport means our wrap should still apply on top.
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-4", "messages": []},
            )
        assert len(captured_spans) == 1
        assert captured_spans[0].output == "wrapped"


class TestEndToEndWithRealTrace:
    """When the active trace is real, the span should land on it."""

    def test_span_appears_on_active_trace(
        self, tracium_client: TraciumClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorded: list[dict[str, Any]] = []
        original = tracium_client.record_agent_spans

        def capture(trace_id: str, payloads: list[dict[str, Any]]) -> Any:
            recorded.extend(payloads)
            return original(trace_id, payloads)

        monkeypatch.setattr(tracium_client, "record_agent_spans", capture)

        def handler(request: httpx.Request) -> httpx.Response:
            return _build_openai_response(
                {
                    "choices": [{"message": {"content": "hello"}}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1},
                }
            )

        transport = TraciumHTTPXTransport(httpx.MockTransport(handler))

        with tracium_client.agent_trace(agent_name="end-to-end-llm"):
            with httpx.Client(transport=transport) as client:
                client.post(
                    "https://api.openai.com/v1/chat/completions",
                    json={"model": "gpt-4", "messages": []},
                )

        llm_spans = [p for p in recorded if p.get("span_type") == "llm"]
        assert llm_spans, f"no llm span recorded; saw types: {[p.get('span_type') for p in recorded]}"
        completed = [p for p in llm_spans if p.get("status") == "completed"]
        assert completed
        final = completed[-1]
        assert final["model_id"] == "gpt-4"
        assert final["input_tokens"] == 2
        assert final["output_tokens"] == 1
