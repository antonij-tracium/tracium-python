"""
Celery integration for task detection and trace completion.

Minimal integration that:
- Extracts task information from Celery execution context
- Tracks when tasks complete to close traces
"""

from __future__ import annotations

import threading

_CELERY_SIGNALS_PATCHED = False
_CELERY_SIGNALS_LOCK = threading.Lock()


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


def register_celery_response_hook() -> None:
    """Register Celery signal handlers to detect when tasks complete and close traces."""
    global _CELERY_SIGNALS_PATCHED

    with _CELERY_SIGNALS_LOCK:
        if _CELERY_SIGNALS_PATCHED:
            return

        try:
            from celery import current_task
            from celery.signals import task_failure, task_success

            from ..auto_trace_tracker import close_web_trace_on_request_completion

            def _on_task_success(sender=None, **kwargs):
                """Close trace when Celery task completes successfully."""
                try:
                    task = current_task
                    if task and task.request:
                        close_web_trace_on_request_completion()
                except Exception:
                    close_web_trace_on_request_completion()

            def _on_task_failure(sender=None, exception=None, **kwargs):
                """Close trace when Celery task fails."""
                try:
                    task = current_task
                    if task and task.request:
                        close_web_trace_on_request_completion(error=exception)
                except Exception:
                    close_web_trace_on_request_completion(error=exception)

            task_success.connect(_on_task_success, weak=False)
            task_failure.connect(_on_task_failure, weak=False)

            _CELERY_SIGNALS_PATCHED = True
        except ImportError:
            pass
        except Exception:
            pass
