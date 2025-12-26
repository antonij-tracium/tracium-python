"""
Generic ASGI/WSGI integration for route detection.

Fallback integration that works with any framework by inspecting call stack.
"""

from __future__ import annotations

import inspect
from types import FrameType
from urllib.parse import urlparse


def get_generic_route_info() -> tuple[str, str] | None:
    """
    Extract route information by inspecting call stack for request objects.

    Works with any ASGI/WSGI framework as a fallback.

    Returns:
        tuple: (route_path, display_name) or None if no request found
    """
    try:
        frame = inspect.currentframe()
        if not frame:
            return None

        current: FrameType | None = frame
        for _ in range(10):
            if current is None:
                break

            locals_dict = current.f_locals
            for var_name in ["request", "req", "http_request", "wsgi_request"]:
                if var_name not in locals_dict:
                    continue

                request_obj = locals_dict[var_name]
                for attr in ["path", "path_info", "url", "PATH_INFO"]:
                    if not hasattr(request_obj, attr):
                        continue

                    path_value = getattr(request_obj, attr)

                    if isinstance(path_value, str):
                        route_path = path_value
                        if route_path.startswith("http"):
                            route_path = urlparse(route_path).path

                        if route_path:
                            route_name = (
                                route_path.strip("/").replace("/", "-")
                                if route_path != "/"
                                else "index"
                            )
                            return route_path, route_name

                    elif hasattr(path_value, "path"):
                        route_path = str(path_value.path)
                        if route_path:
                            route_name = (
                                route_path.strip("/").replace("/", "-")
                                if route_path != "/"
                                else "index"
                            )
                            return route_path, route_name

            current = current.f_back

        return None
    except Exception as e:
        print(f"Error getting generic route info: {e}")
        return None
