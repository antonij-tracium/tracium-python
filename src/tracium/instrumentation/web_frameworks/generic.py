"""Generic WSGI instrumentation for Tracium."""

from __future__ import annotations

import contextvars
from collections.abc import Callable, Iterable, Iterator
from functools import wraps
from typing import Any, cast

_GENERIC_ROUTE_INFO: contextvars.ContextVar[tuple[str, str] | None] = contextvars.ContextVar(
    "tracium_generic_route_info",
    default=None,
)


def get_generic_route_info() -> tuple[str, str] | None:
    """Return the current WSGI route info if set."""
    try:
        return _GENERIC_ROUTE_INFO.get()
    except Exception:
        return None


def wrap_wsgi_app(app: Callable) -> Callable:
    if getattr(app, "_tracium_wrapped", False):
        return app

    @wraps(app)
    def wrapped(environ: dict[str, Any], start_response: Callable) -> Iterable[bytes]:
        from ..auto_trace_tracker import close_web_trace_on_request_completion

        path = environ.get("PATH_INFO") or "/"
        route_name = path.strip("/").replace("/", "-") if path != "/" else "index"

        token = _GENERIC_ROUTE_INFO.set((path, route_name))
        finished = False

        def finish(error: Exception | None = None) -> None:
            nonlocal finished
            if finished:
                return
            finished = True
            try:
                close_web_trace_on_request_completion(error=error)
            finally:
                try:
                    _GENERIC_ROUTE_INFO.reset(token)
                except Exception:
                    pass

        def wrapped_start_response(status: str, headers: list, exc_info=None):
            return start_response(status, headers, exc_info)

        try:
            result = app(environ, wrapped_start_response)
        except Exception as e:
            finish(e)
            raise

        def iterate() -> Iterator[bytes]:
            try:
                for chunk in result:
                    if isinstance(chunk, bytes):
                        yield chunk
                    else:
                        yield from cast(Iterable[bytes], chunk)
            except Exception as e:
                finish(e)
                raise
            finally:
                finish()

        return iterate()

    wrapped._tracium_wrapped = True  # type: ignore
    return wrapped
