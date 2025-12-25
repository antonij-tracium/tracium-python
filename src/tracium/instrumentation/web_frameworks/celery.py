"""
Celery integration for task detection.

Minimal integration that only extracts task information from Celery context.
"""

from __future__ import annotations


def get_celery_task_info() -> tuple[str, str] | None:
    """
    Extract task information from Celery execution context.
    Returns:
        tuple: (task_path, display_name) or None if not in Celery context
    """
    try:
        from celery import current_task

        task = current_task
        if not task or not task.request:
            return None

        task_name = getattr(task, "name", None) or getattr(task.request, "task", None)
        if not task_name:
            return None

        display_name = task_name.split(".")[-1].replace("_", "-")
        task_path = f"/celery/{display_name}"

        return task_path, display_name
    except ImportError:
        return None
    except Exception:
        return None
