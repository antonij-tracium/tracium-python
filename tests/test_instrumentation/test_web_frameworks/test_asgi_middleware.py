"""Tests for the Tracium ASGI middleware that wraps FastAPI/Starlette/etc.
to open a per-request auto-trace."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tracium.instrumentation.web_frameworks.asgi import (
    TraciumASGIMiddleware,
    get_asgi_route_info,
)


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


class _Recorder:
    """Records the route info that was visible to the inner app."""

    def __init__(self) -> None:
        self.seen_route: tuple[str, str] | None = None
        self.calls: list[dict] = []

    async def app(self, scope: dict, receive: Any, send: Any) -> None:
        self.seen_route = get_asgi_route_info()
        self.calls.append(dict(scope))
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok", "more_body": False})


class TestASGIMiddleware:
    def test_http_request_sets_route_info(self):
        rec = _Recorder()
        mw = TraciumASGIMiddleware(rec.app)

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        sent: list[dict] = []

        async def send(msg):
            sent.append(msg)

        _run(mw({"type": "http", "path": "/foo/bar"}, receive, send))
        assert rec.seen_route == ("/foo/bar", "foo-bar")
        # The middleware must let the response through.
        assert any(m["type"] == "http.response.body" for m in sent)

    def test_root_path_normalized(self):
        rec = _Recorder()
        mw = TraciumASGIMiddleware(rec.app)

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        sent: list[dict] = []

        async def send(msg):
            sent.append(msg)

        _run(mw({"type": "http", "path": "/"}, receive, send))
        assert rec.seen_route == ("/", "index")

    def test_route_info_cleared_after_request(self):
        rec = _Recorder()
        mw = TraciumASGIMiddleware(rec.app)

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(msg):
            pass

        _run(mw({"type": "http", "path": "/foo"}, receive, send))
        # Outside the request, the contextvar is back to its default.
        assert get_asgi_route_info() is None

    def test_non_http_scope_passes_through(self):
        rec = _Recorder()
        mw = TraciumASGIMiddleware(rec.app)

        async def receive():
            return {"type": "lifespan.startup"}

        async def send(msg):
            pass

        # WebSocket / lifespan scopes must reach the inner app untouched and
        # must not set the route contextvar.
        async def inner(scope, receive, send):
            rec.calls.append(dict(scope))

        mw = TraciumASGIMiddleware(inner)
        _run(mw({"type": "lifespan"}, receive, send))
        assert rec.calls[0]["type"] == "lifespan"
        assert get_asgi_route_info() is None

    def test_exception_in_inner_app_finalizes(self):
        async def boom(scope, receive, send):
            raise RuntimeError("inner failure")

        mw = TraciumASGIMiddleware(boom)

        async def receive():
            return {"type": "http.request"}

        async def send(msg):
            pass

        with pytest.raises(RuntimeError, match="inner failure"):
            _run(mw({"type": "http", "path": "/x"}, receive, send))
        # Route info must be cleared even on exception.
        assert get_asgi_route_info() is None
