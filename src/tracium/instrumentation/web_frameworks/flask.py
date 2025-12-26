"""
Flask integration for route detection and response tracking.

Minimal integration that:
- Extracts route information from Flask request context
- Tracks when responses are returned to close traces
"""

from __future__ import annotations

import threading

_FLASK_RESPONSE_PATCHED = False
_FLASK_RESPONSE_LOCK = threading.Lock()


def get_flask_route_info() -> tuple[str, str] | None:
    """
    Extract route information from Flask request context.

    Returns:
        tuple: (route_path, display_name) or None if not in Flask context
    """
    try:
        from flask import has_request_context, request

        if not has_request_context():
            return None

        route_path = getattr(request, "path", None)
        if not route_path:
            return None

        endpoint_name = getattr(request, "endpoint", None)
        if not endpoint_name:
            url_rule = getattr(request, "url_rule", None)
            if url_rule:
                endpoint_name = getattr(url_rule, "endpoint", None)

        route_name = route_path.strip("/").replace("/", "-") if route_path != "/" else "index"
        display_name = endpoint_name or route_name

        return route_path, display_name
    except ImportError:
        return None
    except Exception:
        return None


def register_flask_response_hook() -> None:
    """Register hooks to detect when Flask request handlers return responses."""
    global _FLASK_RESPONSE_PATCHED

    with _FLASK_RESPONSE_LOCK:
        if _FLASK_RESPONSE_PATCHED:
            return

        try:
            import flask
            from flask import g, has_request_context

            from ..auto_trace_tracker import close_web_trace_on_request_completion

            def _close_trace_once():
                """Close trace once per request using Flask's g object."""
                if has_request_context() and not hasattr(g, "_tracium_trace_closed"):
                    g._tracium_trace_closed = True
                    close_web_trace_on_request_completion()

            if not hasattr(flask.Flask, "_tracium_response_patched"):
                original_make_response = flask.Flask.make_response

                def patched_make_response(self, rv):
                    try:
                        response = original_make_response(self, rv)
                        _close_trace_once()
                        return response
                    except Exception as e:
                        close_web_trace_on_request_completion(error=e)
                        raise

                setattr(flask.Flask, "make_response", patched_make_response)
                setattr(flask.Flask, "_tracium_response_patched", True)

            if not hasattr(flask, "_tracium_jsonify_patched"):
                original_jsonify = flask.jsonify

                def patched_jsonify(*args, **kwargs):
                    response = original_jsonify(*args, **kwargs)
                    _close_trace_once()
                    return response

                flask.jsonify = patched_jsonify
                setattr(flask, "_tracium_jsonify_patched", True)

            if not hasattr(flask.Flask, "_tracium_exception_patched"):
                original_handle_exception = flask.Flask.handle_exception

                def patched_handle_exception(self, e):
                    close_web_trace_on_request_completion(error=e)
                    return original_handle_exception(self, e)

                setattr(flask.Flask, "handle_exception", patched_handle_exception)
                setattr(flask.Flask, "_tracium_exception_patched", True)

            _FLASK_RESPONSE_PATCHED = True
        except ImportError:
            pass
        except Exception:
            pass
