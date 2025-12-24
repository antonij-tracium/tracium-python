"""
Django integration for route detection.

Minimal integration that only extracts route information from Django request context.
"""

from __future__ import annotations

import inspect
import types
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def get_django_route_info() -> tuple[str, str] | None:
    """
    Extract route information from Django request context.
    Returns:
        tuple: (route_path, display_name) or None if not in Django context
    """
    try:
        from django.http import HttpRequest
    except ImportError:
        return None

    frame = inspect.currentframe()
    if frame is None:
        return None

    request_obj = None
    current: types.FrameType | None = frame

    for _ in range(15):
        if current is None:
            break
        for var_value in current.f_locals.values():
            if isinstance(var_value, HttpRequest):
                request_obj = var_value
                break
        if request_obj is not None:
            break
        current = current.f_back

    if request_obj is None:
        return None

    route_path = getattr(request_obj, "path_info", None) or getattr(request_obj, "path", None)
    if not route_path:
        return None

    view_name = None
    resolver_match = getattr(request_obj, "resolver_match", None)
    if resolver_match:
        view_name = getattr(resolver_match, "view_name", None) or getattr(resolver_match, "url_name", None)
        if not view_name:
            func = getattr(resolver_match, "func", None)
            if func:
                view_name = getattr(func, "__name__", None)

    route_name = route_path.strip("/").replace("/", "-") if route_path != "/" else "index"
    display_name = view_name or route_name

    return route_path, display_name


def register_django_response_hook() -> None:
    """Register hooks to detect when Django request handlers return responses."""
    try:
        from django.http import HttpResponse

        from ..auto_trace_tracker import close_web_trace_on_request_completion
    except ImportError:
        return

    if not hasattr(HttpResponse, "_tracium_response_patched"):
        original_http_response_init = HttpResponse.__init__

        def patched_http_response_init(self, *args, **kwargs):
            original_http_response_init(self, *args, **kwargs)
            if not hasattr(self, "_tracium_trace_closed"):
                self._tracium_trace_closed = True
                close_web_trace_on_request_completion()

        HttpResponse.__init__ = patched_http_response_init
        HttpResponse._tracium_response_patched = True

    try:
        import django.core.handlers.exception
        from django.core.handlers.exception import convert_exception_to_response
    except ImportError:
        return

    if not hasattr(convert_exception_to_response, "_tracium_patched"):
        original_convert = convert_exception_to_response

        def patched_convert(*args, **kwargs):
            try:
                return original_convert(*args, **kwargs)
            except Exception as e:
                close_web_trace_on_request_completion(error=e)
                raise

        django.core.handlers.exception.convert_exception_to_response = patched_convert
        convert_exception_to_response._tracium_patched = True
