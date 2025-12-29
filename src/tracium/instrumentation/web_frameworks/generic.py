"""Generic WSGI instrumentation for Tracium."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Iterator
from functools import wraps
from typing import Any, cast


def wrap_wsgi_app(app: Callable) -> Callable:
    if getattr(app, "_tracium_wrapped", False):
        return app

    _traces: dict[str, Any] = {}
    _lock = threading.Lock()

    @wraps(app)
    def wrapped(environ: dict[str, Any], start_response: Callable) -> Iterable[bytes]:
        from ...helpers.global_state import get_client, get_default_tags, get_options

        path = environ.get("PATH_INFO") or "/"
        route_name = path.strip("/").replace("/", "-") if path != "/" else "index"

        client = get_client()
        options = get_options()
        manager = client.agent_trace(
            agent_name=route_name,
            model_id=options.default_model_id,
            version=options.default_version,
            tags=get_default_tags(["@wsgi", f"@route:{route_name}"]),
        )
        handle = manager.__enter__()
        trace_id = handle.id

        with _lock:
            _traces[trace_id] = (manager, handle)

        status_code = 200
        finished = False

        def finish(error: Exception | None = None) -> None:
            nonlocal finished
            if finished:
                return
            finished = True

            with _lock:
                data = _traces.pop(trace_id, None)
            if not data:
                return

            mgr, hdl = data
            try:
                if error:
                    hdl.mark_failed(f"{type(error).__name__}: {error}")
                    mgr.__exit__(type(error), error, error.__traceback__)
                else:
                    mgr.__exit__(None, None, None)
            except Exception:
                pass

        def wrapped_start_response(status: str, headers: list, exc_info=None):
            nonlocal status_code
            try:
                status_code = int(status.split(" ", 1)[0])
            except Exception:
                status_code = 500
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
