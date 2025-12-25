"""
FastAPI/Starlette integration for route detection.

Minimal integration that only extracts route information from ASGI request context.
"""

from __future__ import annotations

import inspect
import types


def get_fastapi_route_info() -> tuple[str, str] | None:
    """
    Extract route information from FastAPI/Starlette request context.

    Returns:
        tuple: (route_path, display_name) or None if not in FastAPI context
    """
    try:
        from starlette.requests import Request as StarletteRequest
    except ImportError:
        return None

    # Find Request object in call stack
    frame = inspect.currentframe()
    request_obj = None

    if frame:
        current: types.FrameType | None = frame
        for _ in range(15):
            if current is None:
                break
            locals_dict = current.f_locals
            for var_name in ["request", "req", "http_request"]:
                if var_name in locals_dict:
                    obj = locals_dict[var_name]
                    if isinstance(obj, StarletteRequest):
                        request_obj = obj
                        break
            if request_obj:
                break
            current = current.f_back

    if not request_obj:
        try:
            from starlette_context import context

            obj = context.get("request", None)
            if obj and isinstance(obj, StarletteRequest):
                request_obj = obj
        except ImportError:
            pass

    if not request_obj:
        return None

    route_path = str(request_obj.url.path)
    if not route_path:
        return None

    route_name = None
    if hasattr(request_obj, "scope"):
        route = request_obj.scope.get("route")
        if route:
            route_name = getattr(route, "name", None) or getattr(route, "path", None)

    display_name = route_name or (
        route_path.strip("/").replace("/", "-") if route_path != "/" else "index"
    )
    return route_path, display_name


def register_fastapi_response_hook() -> None:
    """Register hooks to detect when FastAPI request handlers return responses."""
    try:
        from starlette.responses import Response

        from ..auto_trace_tracker import close_web_trace_on_request_completion

        if hasattr(Response, "_tracium_response_patched"):
            return

        original_init = Response.__init__

        def patched_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if not hasattr(self, "_tracium_trace_closed"):
                self._tracium_trace_closed = True
                close_web_trace_on_request_completion()

        Response.__init__ = patched_init

        try:
            from starlette.applications import Starlette

            if not hasattr(Starlette, "_tracium_exception_patched"):
                original_exception_handler = Starlette.exception_handler

                def patched_exception_handler(self, exc_class_or_status_code, handler=None):
                    if handler is None:
                        return original_exception_handler(self, exc_class_or_status_code)

                    async def wrapped_handler(request, exc):
                        close_web_trace_on_request_completion(error=exc)
                        return await handler(request, exc)

                    return original_exception_handler(
                        self, exc_class_or_status_code, wrapped_handler
                    )

                Starlette.exception_handler = patched_exception_handler
                setattr(Starlette, "_tracium_exception_patched", True)
        except Exception:
            pass

        setattr(Response, "_tracium_response_patched", True)
    except ImportError:
        pass
    except Exception:
        pass
