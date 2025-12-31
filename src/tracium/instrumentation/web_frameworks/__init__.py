"""
Web framework integrations for automatic route detection.

Each integration is minimal and only:
- Detects if the framework is active
- Extracts route/context information
- Returns (route_path, display_name) or None

All integrations are tried in order until one succeeds.
"""

from __future__ import annotations

from .celery import get_celery_task_info, register_celery_response_hook
from .django import get_django_route_info, register_django_response_hook
from .fastapi import get_fastapi_route_info, register_fastapi_response_hook
from .flask import get_flask_route_info, register_flask_response_hook
from .generic import get_generic_route_info
from .generic import wrap_wsgi_app as wrap_wsgi_app


def get_web_route_info() -> tuple[str, str] | None:
    """
    Try all web framework integrations to extract route information.
    Returns:
        tuple: (route_path, display_name) or None if no framework detected
    """
    integrations = [
        get_flask_route_info,
        get_fastapi_route_info,
        get_django_route_info,
        get_generic_route_info,
        get_celery_task_info,
    ]

    for integration in integrations:
        try:
            result = integration()
            if result is not None:
                return result
        except Exception:
            continue

    return None


def register_response_hooks() -> None:
    """Register hooks to detect when request handlers return responses."""
    try:
        register_flask_response_hook()
    except Exception:
        pass

    try:
        register_fastapi_response_hook()
    except Exception:
        pass

    try:
        register_django_response_hook()
    except Exception:
        pass

    try:
        register_celery_response_hook()
    except Exception:
        pass
