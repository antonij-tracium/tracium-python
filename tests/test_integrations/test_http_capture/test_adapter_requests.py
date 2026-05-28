"""
Tests for the ``requests`` adapter.

We patch :meth:`requests.Session.send` and route fake responses through it. No
network is actually contacted.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
import requests

from tracium.integrations.http_capture import adapter_requests
from tracium.integrations.http_capture.dedup import owned_capture


@pytest.fixture
def captured_spans(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    seen: list[Any] = []

    def _fake_emit(call, started_at, ended_at=None):  # noqa: ARG001
        seen.append(call)

    monkeypatch.setattr(adapter_requests, "emit_llm_span", _fake_emit)
    return seen


@pytest.fixture
def install_once() -> None:
    """Ensure the patch is installed; idempotent across tests."""
    adapter_requests.install()


def _make_response(
    *,
    status: int = 200,
    json_body: dict[str, Any] | None = None,
    content_type: str = "application/json",
) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response.headers["content-type"] = content_type
    if json_body is not None:
        import json as _json

        response._content = _json.dumps(json_body).encode("utf-8")
    else:
        response._content = b""
    return response


def test_captures_openai_request(
    captured_spans: list[Any], install_once: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_response = _make_response(
        json_body={
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 1},
        }
    )

    def fake_send(self, request, **kwargs):  # noqa: ARG001
        return fake_response

    # The adapter wrapped Session.send; we need to swap the underlying impl
    # that the wrapper calls. The wrapper captured the original at install
    # time, so we patch via the module's bound reference.
    # Simplest: monkey-patch the underlying HTTPAdapter to return our response.
    session = requests.Session()

    captured_request = MagicMock()
    captured_request.url = "https://api.openai.com/v1/chat/completions"
    captured_request.body = b'{"model":"gpt-4","messages":[]}'

    # Bypass the network entirely by stubbing the inner function the wrapper
    # calls. The wrapper uses `original_send`; we don't have a clean handle
    # on it from outside, so we do a partial integration via session adapter.
    class _StubAdapter(requests.adapters.HTTPAdapter):
        def send(self, request, **kwargs):  # type: ignore[override]
            captured_request.url = request.url
            captured_request.body = request.body
            return fake_response

    session.mount("https://", _StubAdapter())
    session.post(
        "https://api.openai.com/v1/chat/completions",
        json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert len(captured_spans) == 1
    assert captured_spans[0].provider == "openai"
    assert captured_spans[0].input_tokens == 7


def test_owned_capture_skips_emission(
    captured_spans: list[Any], install_once: None
) -> None:
    fake_response = _make_response(
        json_body={"choices": [{"message": {"content": "x"}}], "usage": {}}
    )

    class _StubAdapter(requests.adapters.HTTPAdapter):
        def send(self, request, **kwargs):  # type: ignore[override]
            return fake_response

    session = requests.Session()
    session.mount("https://", _StubAdapter())

    with owned_capture():
        session.post(
            "https://api.openai.com/v1/chat/completions",
            json={"model": "gpt-4", "messages": []},
        )

    assert captured_spans == []


def test_non_llm_url_passes_through(
    captured_spans: list[Any], install_once: None
) -> None:
    fake_response = _make_response(json_body={"ok": True})

    class _StubAdapter(requests.adapters.HTTPAdapter):
        def send(self, request, **kwargs):  # type: ignore[override]
            return fake_response

    session = requests.Session()
    session.mount("https://", _StubAdapter())
    session.get("https://example.com/api/users")
    assert captured_spans == []


def test_install_is_idempotent() -> None:
    adapter_requests.install()
    adapter_requests.install()  # second call must be a no-op
