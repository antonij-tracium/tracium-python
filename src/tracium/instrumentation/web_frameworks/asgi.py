"""Generic ASGI middleware for Tracium.

Wraps any ASGI app (FastAPI, Starlette, Litestar, Sanic, BlackSheep, etc.) so
that each HTTP request opens an auto-trace and closes it on response. Nested
LLM spans triggered inside the request handler are attached to this trace.

WebSocket / lifespan scopes are passed through untouched.
"""

from __future__ import annotations

import contextvars
from typing import Any

_ASGI_ROUTE_INFO: contextvars.ContextVar[tuple[str, str] | None] = contextvars.ContextVar(
    "tracium_asgi_route_info",
    default=None,
)


def get_asgi_route_info() -> tuple[str, str] | None:
    """Return the current ASGI route info if set."""
    try:
        return _ASGI_ROUTE_INFO.get()
    except Exception:
        return None


class TraciumASGIMiddleware:
    """ASGI middleware that opens a Tracium auto-trace per HTTP request.

    Usage (manual):

        app = FastAPI()
        app = TraciumASGIMiddleware(app)

    `tracium.trace()` installs this automatically when it detects an ASGI app
    in caller globals.
    """

    def __init__(self, app: Any) -> None:
        self._app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self._app(scope, receive, send)
            return

        import uuid

        from ..auto_trace_tracker import (
            WEB_REQUEST_TOKEN,
            close_web_trace_on_request_completion,
        )

        path = scope.get("path") or "/"
        route_name = path.strip("/").replace("/", "-") if path != "/" else "index"
        token = _ASGI_ROUTE_INFO.set((path, route_name))
        # Stable per-request token so an auto-trace created in a threadpool
        # worker (sync FastAPI handler) can be matched back to this request
        # when we close it down here in the event loop.
        request_token = WEB_REQUEST_TOKEN.set(str(uuid.uuid4()))

        finished = False

        def finish(error: BaseException | None = None) -> None:
            nonlocal finished
            if finished:
                return
            finished = True
            try:
                close_web_trace_on_request_completion(
                    error=error if isinstance(error, Exception) else None
                )
            finally:
                try:
                    _ASGI_ROUTE_INFO.reset(token)
                except Exception:
                    pass
                try:
                    WEB_REQUEST_TOKEN.reset(request_token)
                except Exception:
                    pass

        response_started = False

        async def wrapped_send(message: dict) -> None:
            nonlocal response_started
            msg_type = message.get("type")
            if msg_type == "http.response.start":
                response_started = True
            await send(message)
            if msg_type == "http.response.body" and not message.get("more_body", False):
                # Finalize after the last body chunk so spans created during
                # streamed response generation still attach to this trace.
                finish()

        try:
            await self._app(scope, receive, wrapped_send)
        except Exception as e:
            finish(e)
            raise
        finally:
            # Safety net: if the app never emitted a final body chunk (e.g.
            # streaming + connection close), still finalize.
            if not finished:
                finish()
