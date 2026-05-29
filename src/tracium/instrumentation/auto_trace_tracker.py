"""
Automatic trace and span tracking for tracium.trace() functionality.

This module provides intelligent detection of:
1. When to create a new trace (detecting workflow entry points)
2. When to create spans (detecting function calls)
3. Proper parent-child relationships between spans
"""

from __future__ import annotations

import atexit
import contextvars
import inspect
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core import TraciumClient
from ..models.trace_handle import AgentTraceManager

_AUTO_TRACE_CONTEXT: contextvars.ContextVar[AutoTraceContext | None] = contextvars.ContextVar(
    "tracium_auto_trace_context",
    default=None,
)

_WEB_ROUTE_WHEN_TRACE_CREATED: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "tracium_web_route_when_trace_created",
    default=None,
)

# Per-request token set by ASGI/WSGI middleware. Threadpool workers spawned by
# FastAPI for sync handlers inherit this token via our contextvar-propagation
# patch, so an auto-trace created in the worker can be registered against it.
# When the middleware finalizes the request — back in the event loop — it
# looks up the registry by this token to find the trace the worker created
# (the worker's `_AUTO_TRACE_CONTEXT` set never propagated back here).
WEB_REQUEST_TOKEN: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "tracium_web_request_token",
    default=None,
)

_CLEANUP_REGISTERED = False
_CLEANUP_LOCK = threading.Lock()
_ORIGINAL_EXCEPTHOOK = None
_ORIGINAL_ASYNCIO_HANDLER = None

# Process-wide registry of active non-web auto-traces, keyed by entry_frame_id.
# Lets sibling asyncio tasks (or threadpool workers) that share an entry frame
# adopt the same auto-trace instead of each creating their own. Web traces are
# not registered here — concurrent requests must stay isolated per ContextVar.
_AUTO_TRACE_REGISTRY: dict[str, AutoTraceContext] = {}
_AUTO_TRACE_REGISTRY_LOCK = threading.Lock()

# Per-request web traces, keyed by `WEB_REQUEST_TOKEN`. Used so the middleware
# can finalize a trace whose `_AUTO_TRACE_CONTEXT` was set in a worker thread's
# context copy and is therefore invisible to the event loop.
_WEB_AUTO_TRACE_REGISTRY: dict[str, AutoTraceContext] = {}
_WEB_AUTO_TRACE_REGISTRY_LOCK = threading.Lock()

_SKIP_PATTERNS = [
    # Library package paths. Bracketed with path separators so user files
    # like "openai_chat.py" or "my_anthropic_helper.py" aren't mistakenly
    # skipped as library frames.
    "/tracium/",
    "/openai/",
    "/anthropic/",
    "/google/",
    "/langchain/",
    "/langgraph/",
    "/threading.py",
    "/concurrent/",
    "/asyncio/",
    "/site-packages/",
    "/werkzeug/",
    "/starlette/",
    "/django/core/",
    "/uvicorn/",
    "/gunicorn/",
    # Long-lived runtimes whose bootstrap frame would otherwise be the
    # "outermost user frame" forever, defeating per-invocation trace
    # boundaries. Skipping them makes the handler/task function the entry.
    "/var/runtime/",  # AWS Lambda
    "/var/lang/",  # AWS Lambda (alt path)
    "/awslambdaric/",  # Lambda runtime interface client
    "/prefect/",  # Prefect workers
    "/celery/",  # Celery workers
    "/rq/worker",  # RQ
    "/dramatiq/",  # Dramatiq
]

_WEB_FRAMEWORK_PATTERNS = {
    "flask": ["route", "view_function", "dispatch_request"],
    "fastapi": ["endpoint", "dependant", "run_endpoint_function"],
    "django": ["view", "dispatch", "get_response"],
    "aiohttp": ["handler", "_handle"],
    "sanic": ["handle_request", "router"],
}


def _close_trace_safely(
    context: AutoTraceContext | None, error: BaseException | None = None
) -> None:
    """
    Safely close a trace context, ignoring any errors.
    Args:
        context: The auto trace context to close
        error: Optional exception that occurred, to mark trace as failed
    """
    if context is None:
        return

    try:
        with _AUTO_TRACE_REGISTRY_LOCK:
            entry_id = context.entry_frame_id
            if _AUTO_TRACE_REGISTRY.get(entry_id) is context:
                del _AUTO_TRACE_REGISTRY[entry_id]
    except Exception:
        pass

    should_mark_failed = context.has_failed_span or error is not None

    try:
        if error is not None:
            context.trace_manager.__exit__(type(error), error, error.__traceback__)
        elif should_mark_failed and not context.trace_handle._state.finished:
            try:
                error_msg = (
                    context.trace_handle._state.error or "One or more spans in this trace failed"
                )
                context.trace_handle.mark_failed(error_msg)
            except Exception:
                pass
            context.trace_manager.__exit__(None, None, None)
        else:
            context.trace_manager.__exit__(None, None, None)
    except Exception:
        try:
            trace_handle = context.trace_handle
            if hasattr(trace_handle, "_state") and trace_handle._state.status == "in_progress":
                trace_handle._state.status = "incomplete"
        except Exception:
            pass


def _cleanup_handler() -> None:
    """
    Cleanup handler to close any open auto-traces at program exit.
    Note: In web servers, this typically won't have traces to close because
    each request should clean up its own context. This is mainly for
    CLI scripts and batch jobs.
    """
    auto_context = _AUTO_TRACE_CONTEXT.get()
    if auto_context is not None:
        _close_trace_safely(auto_context)
        _AUTO_TRACE_CONTEXT.set(None)

    try:
        from ..helpers.global_state import STATE

        if STATE.client:
            STATE.client.flush()
    except Exception:
        pass


def _close_trace_on_exception(exc_type, exc_value, exc_traceback) -> None:
    """Close any open auto-traces and mark them as failed."""
    context = _AUTO_TRACE_CONTEXT.get()

    if context is not None:
        try:
            trace_handle = context.trace_handle
            if hasattr(trace_handle, "mark_failed") and exc_value is not None:
                error_msg = f"{exc_type.__name__}: {str(exc_value)}"
                try:
                    trace_handle.mark_failed(error_msg)
                except Exception:
                    pass
            context.trace_manager.__exit__(exc_type, exc_value, exc_traceback)
        except Exception:
            pass
        finally:
            _AUTO_TRACE_CONTEXT.set(None)


def _exception_handler(exc_type, exc_value, exc_traceback) -> None:
    """Global exception handler to ensure traces are closed on unhandled exceptions."""
    _close_trace_on_exception(exc_type, exc_value, exc_traceback)

    if _ORIGINAL_EXCEPTHOOK is not None:
        try:
            _ORIGINAL_EXCEPTHOOK(exc_type, exc_value, exc_traceback)
        except Exception:
            import sys

            sys.__excepthook__(exc_type, exc_value, exc_traceback)


def _asyncio_exception_handler(loop, context: dict) -> None:
    """Asyncio exception handler to ensure traces are closed on unhandled async exceptions."""
    exception = context.get("exception")
    if exception is None:
        error_msg = context.get("message", "Unknown asyncio error")
        exc_type, exc_value, exc_traceback = Exception, Exception(error_msg), None
    else:
        exc_type, exc_value, exc_traceback = type(exception), exception, exception.__traceback__

    _close_trace_on_exception(exc_type, exc_value, exc_traceback)

    if _ORIGINAL_ASYNCIO_HANDLER is not None:
        try:
            _ORIGINAL_ASYNCIO_HANDLER(loop, context)
        except Exception:
            loop.default_exception_handler(context)


def _register_asyncio_handler() -> None:
    """Register asyncio exception handler for unhandled async exceptions."""
    global _ORIGINAL_ASYNCIO_HANDLER

    try:
        import asyncio
    except ImportError:
        return

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            if not hasattr(asyncio.new_event_loop, "_tracium_patched"):
                original = asyncio.new_event_loop

                def patched():
                    loop = original()
                    handler = loop.get_exception_handler()
                    if handler is not _asyncio_exception_handler:
                        global _ORIGINAL_ASYNCIO_HANDLER
                        _ORIGINAL_ASYNCIO_HANDLER = handler or loop.default_exception_handler
                        loop.set_exception_handler(_asyncio_exception_handler)
                    return loop

                setattr(patched, "_tracium_patched", True)
                asyncio.new_event_loop = patched
            return

    handler = loop.get_exception_handler()
    if handler is not _asyncio_exception_handler:
        _ORIGINAL_ASYNCIO_HANDLER = handler or loop.default_exception_handler
        loop.set_exception_handler(_asyncio_exception_handler)


def register_cleanup() -> None:
    """Register the cleanup handler if not already registered."""
    global _CLEANUP_REGISTERED, _ORIGINAL_EXCEPTHOOK

    with _CLEANUP_LOCK:
        if _CLEANUP_REGISTERED:
            return

        import sys

        atexit.register(_cleanup_handler)

        current_hook = sys.excepthook
        if current_hook is not _exception_handler:
            _ORIGINAL_EXCEPTHOOK = (
                current_hook if current_hook is not sys.__excepthook__ else sys.__excepthook__
            )
            sys.excepthook = _exception_handler

        _register_asyncio_handler()

        _CLEANUP_REGISTERED = True


@dataclass
class AutoTraceContext:
    """Context for an automatically created trace."""

    trace_manager: AgentTraceManager
    trace_handle: Any
    entry_frame_id: str
    entry_function_name: str
    has_failed_span: bool = False

    def mark_span_failed(self) -> None:
        """Mark that a span in this trace has failed."""
        self.has_failed_span = True


def _get_user_frames() -> list[tuple[Any, ...]]:
    """Extract user frames from the call stack, skipping internal/library frames."""
    frame = inspect.currentframe()
    user_frames: list[tuple[Any, ...]] = []
    try:
        if frame is not None:
            frame = frame.f_back
        if frame is not None:
            frame = frame.f_back

        while frame is not None:
            filename = frame.f_code.co_filename
            func_name = frame.f_code.co_name

            if not any(pattern in filename for pattern in _SKIP_PATTERNS):
                user_frames.append((func_name, filename, frame.f_lineno, id(frame)))
            frame = frame.f_back
    finally:
        del frame
    return user_frames


def _get_caller_info() -> tuple[str, str, int]:
    """
    Get information about the caller function.

    Returns:
        tuple: (function_name, file_path, line_number)
    """
    user_frames = _get_user_frames()
    if not user_frames:
        return "<unknown>", "<unknown>", 0

    helper_patterns = ["call_", "_call", "wrapper", "helper", "util", "invoke"]
    skip_names = ("__init__", "__call__", "__enter__", "__exit__")

    for func_name, file_path, line_no, _ in user_frames:
        is_helper = any(pattern in func_name.lower() for pattern in helper_patterns)
        if not is_helper and func_name not in skip_names:
            return func_name, file_path, line_no

    first_frame = user_frames[0]
    return (str(first_frame[0]), str(first_frame[1]), int(first_frame[2]))


def _is_web_framework_internal(func_name: str, file_path: str) -> bool:
    """Check if a frame is internal web framework code."""
    func_lower = func_name.lower()

    for framework, patterns in _WEB_FRAMEWORK_PATTERNS.items():
        if framework in file_path.lower():
            return any(pattern in func_lower for pattern in patterns)

    return False


def _get_web_route_info() -> tuple[str, str] | None:
    """
    Extract route information from web framework request context if available.
    Delegates to framework-specific integrations in web_frameworks module.
    Returns:
        tuple: (route_path, display_name) or None if not in web context
    """
    from .web_frameworks import get_web_route_info as _get_route_info

    return _get_route_info()


def _find_endpoint_handler(user_frames: list[tuple]) -> tuple[str, str, str] | None:
    """
    Find the actual endpoint handler in web frameworks.
    Returns:
        tuple: (function_name, file_path, frame_key) or None
    """
    found_framework = False

    for i, (func_name, file_path, line_no, _) in enumerate(user_frames):
        is_framework = _is_web_framework_internal(func_name, file_path)

        if is_framework:
            found_framework = True
            continue

        if found_framework:
            frame_key = f"{file_path}:{func_name}:{line_no}"
            return func_name, file_path, frame_key

    return None


def _find_workflow_entry_point() -> tuple[str, str]:
    """
    Find the entry point of the current workflow by examining the call stack.

    For web frameworks, this finds the endpoint handler function.
    For CLI/batch scripts, this finds the outermost user function.

    Returns a unique identifier for the entry frame and a human-readable name.
    Never raises; returns ("atexit:workflow", "workflow") on any error.
    """
    try:
        web_route_info = _get_web_route_info()
        if web_route_info is not None:
            route_path, display_name = web_route_info
            frame_key = f"web:{route_path}"
            return frame_key, display_name

        user_frames = _get_user_frames()
        if not user_frames:
            return "atexit:workflow", "workflow"

        endpoint_info = _find_endpoint_handler(user_frames)
        if endpoint_info is not None:
            func_name, file_path, frame_key = endpoint_info
            return frame_key, func_name

        entry_function, entry_file, entry_line, entry_frame_id = user_frames[-1]
        # Include the frame's runtime identity so each invocation of the same
        # function (Lambda handler, Prefect task, batch-loop iteration, …) gets
        # its own trace. Without id(frame), every invocation collapses into the
        # first trace because (file, function, line) is invocation-invariant.
        frame_key = f"{entry_file}:{entry_function}:{entry_line}:{entry_frame_id}"

        if entry_function == "<module>":
            filename_stem = Path(entry_file).stem
            if filename_stem == "__main__":
                filename_stem = Path(entry_file).name
                if filename_stem.endswith(".py"):
                    filename_stem = filename_stem[:-3]
            return (
                frame_key,
                filename_stem if filename_stem and filename_stem != "__main__" else entry_function,
            )

        return frame_key, entry_function
    except Exception:
        return "atexit:workflow", "workflow"


def get_or_create_auto_trace(
    client: TraciumClient,
    agent_name: str,
    model_id: str | None = None,
    tags: list[str] | None = None,
    version: str | None = None,
) -> tuple[Any, bool]:
    """
    Get the current auto-created trace, or create one if needed.

    Uses contextvars for proper request isolation in web servers.
    Each request/task gets its own trace context.

    Returns:
        tuple: (trace_handle, created_new_trace)
    """
    from ..context.trace_context import current_trace
    from ..helpers.global_state import get_options
    from ..instrumentation.auto_detection import detect_agent_name

    manual_trace = current_trace()
    if manual_trace is not None:
        return manual_trace, False

    auto_context = _AUTO_TRACE_CONTEXT.get()
    if auto_context is not None:
        if auto_context.entry_frame_id.startswith("web:"):
            stored_route = _WEB_ROUTE_WHEN_TRACE_CREATED.get()
            current_web_route = _get_web_route_info()

            if stored_route is not None:
                if current_web_route is None:
                    _AUTO_TRACE_CONTEXT.set(None)
                    _WEB_ROUTE_WHEN_TRACE_CREATED.set(None)
                    _close_trace_safely(auto_context)
                else:
                    current_route_path, _ = current_web_route
                    if current_route_path != stored_route:
                        _AUTO_TRACE_CONTEXT.set(None)
                        _WEB_ROUTE_WHEN_TRACE_CREATED.set(None)
                        _close_trace_safely(auto_context)
                    else:
                        return auto_context.trace_handle, False
            else:
                return auto_context.trace_handle, False
        else:
            current_frame_id, _ = _find_workflow_entry_point()
            if current_frame_id == auto_context.entry_frame_id:
                return auto_context.trace_handle, False
            else:
                _AUTO_TRACE_CONTEXT.set(None)
                _close_trace_safely(auto_context)

    entry_frame_id, entry_function_name = _find_workflow_entry_point()

    if not entry_frame_id.startswith("web:"):
        with _AUTO_TRACE_REGISTRY_LOCK:
            # With frame-id-based keys, stale entries from finished invocations
            # would otherwise accumulate. Sweep them here.
            stale_keys: list[str] = []
            for key, ctx in _AUTO_TRACE_REGISTRY.items():
                try:
                    if bool(getattr(ctx.trace_handle._state, "finished", True)):
                        stale_keys.append(key)
                except Exception:
                    stale_keys.append(key)
            for key in stale_keys:
                _AUTO_TRACE_REGISTRY.pop(key, None)

            existing = _AUTO_TRACE_REGISTRY.get(entry_frame_id)
            if existing is not None:
                stale = True
                try:
                    state = existing.trace_handle._state
                    stale = bool(getattr(state, "finished", True))
                except Exception:
                    stale = True
                if not stale:
                    _AUTO_TRACE_CONTEXT.set(existing)
                    _WEB_ROUTE_WHEN_TRACE_CREATED.set(None)
                    return existing.trace_handle, False
                _AUTO_TRACE_REGISTRY.pop(entry_frame_id, None)

    if entry_frame_id.startswith("web:"):
        from .web_frameworks import register_response_hooks

        try:
            register_response_hooks()
        except Exception:
            pass

    try:
        default_agent_name = get_options().default_agent_name
    except RuntimeError:
        default_agent_name = "app"

    if agent_name == "app" or agent_name == default_agent_name or not agent_name:
        if entry_function_name and entry_function_name not in ("workflow", "<module>"):
            try:
                normalized = entry_function_name.replace("_", "-")
                if normalized and normalized not in ("main", "<module>"):
                    agent_name = normalized
            except Exception:
                pass

        if not agent_name or agent_name in ("app", default_agent_name):
            if entry_function_name in ("workflow", "<module>") and ":" in entry_frame_id:
                try:
                    file_path = entry_frame_id.split(":")[0]
                    if file_path and Path(file_path).exists():
                        script_name = Path(file_path).stem
                        if script_name == "__main__":
                            script_name = Path(file_path).name
                            if script_name.endswith(".py"):
                                script_name = script_name[:-3]
                        if script_name and script_name != "__main__":
                            agent_name = script_name.replace("_", "-")
                except Exception:
                    pass

        if not agent_name or agent_name in ("app", default_agent_name, "main"):
            agent_name = detect_agent_name(default_agent_name)

    if version is None:
        try:
            version = get_options().default_version
        except RuntimeError:
            version = None

    try:
        workspace_id = get_options().default_workspace_id
    except RuntimeError:
        workspace_id = None

    register_cleanup()

    trace_manager = client.agent_trace(
        agent_name=agent_name,
        model_id=model_id,
        tags=tags or [],
        version=version,
        workspace_id=workspace_id,
        lazy_start=True,
    )
    trace_handle = trace_manager.__enter__()

    auto_context = AutoTraceContext(
        trace_manager=trace_manager,
        trace_handle=trace_handle,
        entry_frame_id=entry_frame_id,
        entry_function_name=entry_function_name,
    )

    _AUTO_TRACE_CONTEXT.set(auto_context)

    if entry_frame_id.startswith("web:"):
        current_web_route = _get_web_route_info()
        if current_web_route:
            route_path, _ = current_web_route
            _WEB_ROUTE_WHEN_TRACE_CREATED.set(route_path)
        else:
            _WEB_ROUTE_WHEN_TRACE_CREATED.set(None)
        # Also pin this trace to the per-request token so the middleware can
        # find it even if it was created in a threadpool worker (whose context
        # set doesn't propagate back to the event loop).
        try:
            token = WEB_REQUEST_TOKEN.get()
            if token:
                with _WEB_AUTO_TRACE_REGISTRY_LOCK:
                    _WEB_AUTO_TRACE_REGISTRY[token] = auto_context
        except Exception:
            pass
    else:
        _WEB_ROUTE_WHEN_TRACE_CREATED.set(None)
        with _AUTO_TRACE_REGISTRY_LOCK:
            _AUTO_TRACE_REGISTRY[entry_frame_id] = auto_context

    return trace_handle, True


def should_close_auto_trace(force_close: bool = False) -> bool:
    """
    Determine if we should close the current auto-created trace.
    This happens when we're exiting the entry point function.
    Args:
        force_close: If True, close the trace unless we're in a web context
                     where we want to keep it open for the entire request
    """
    auto_context = _AUTO_TRACE_CONTEXT.get()
    if auto_context is None:
        return False

    if hasattr(auto_context.trace_handle, "_state") and auto_context.trace_handle._state.finished:
        return True

    current_web_route = _get_web_route_info()
    is_in_web_context = current_web_route is not None

    trace_is_web_context = auto_context.entry_frame_id.startswith("web:")

    if force_close:
        if trace_is_web_context:
            if is_in_web_context:
                return False
            return True
        return True

    current_frame_id, _ = _find_workflow_entry_point()

    return current_frame_id != auto_context.entry_frame_id


def close_auto_trace_if_needed(
    force_close: bool = False, error: BaseException | None = None
) -> None:
    """
    Close the auto-created trace if we've exited the workflow.
    Args:
        force_close: If True, close the trace unless we're in a web context
                     where we want to keep it open for the entire request
        error: Optional exception that occurred, to mark trace as failed
    """
    auto_context = _AUTO_TRACE_CONTEXT.get()
    if auto_context is None:
        return

    if auto_context.entry_frame_id.startswith("web:") and force_close:
        stored_route = _WEB_ROUTE_WHEN_TRACE_CREATED.get()
        current_web_route = _get_web_route_info()

        if stored_route is not None:
            if current_web_route is None:
                _AUTO_TRACE_CONTEXT.set(None)
                _WEB_ROUTE_WHEN_TRACE_CREATED.set(None)
                _close_trace_safely(auto_context, error=error)
                return
            else:
                current_route_path, _ = current_web_route
                if current_route_path != stored_route:
                    _AUTO_TRACE_CONTEXT.set(None)
                    _WEB_ROUTE_WHEN_TRACE_CREATED.set(None)
                    _close_trace_safely(auto_context, error=error)
                    return
                else:
                    return

    if not should_close_auto_trace(force_close=force_close):
        return

    context = _AUTO_TRACE_CONTEXT.get()
    _AUTO_TRACE_CONTEXT.set(None)
    _close_trace_safely(context, error=error)


def get_current_function_for_span(use_route_info: bool = False) -> str:
    """
    Get the current function name to use as a span name.

    This looks up the call stack to find the user's function.
    Args:
        use_route_info: If True, use route information for web frameworks.
                       If False (default), use actual function name.
                       Route info should only be used for trace/agent naming,
                       not for individual span naming.
    """
    if use_route_info:
        web_route_info = _get_web_route_info()
        if web_route_info is not None:
            _, display_name = web_route_info
            return display_name

    function_name, _, _ = _get_caller_info()
    return function_name


def cleanup_auto_trace() -> None:
    """Force cleanup of any auto-created trace.

    Also drains the process-wide registries so traces that were created in
    background threads / threadpool workers (whose ``_AUTO_TRACE_CONTEXT``
    set never propagated back to the caller of this function) still receive
    their terminal POST.
    """
    auto_context = _AUTO_TRACE_CONTEXT.get()
    if auto_context is not None:
        _close_trace_safely(auto_context)
        _AUTO_TRACE_CONTEXT.set(None)

    # Drain non-web traces created in other contexts.
    with _AUTO_TRACE_REGISTRY_LOCK:
        contexts_to_close = list(_AUTO_TRACE_REGISTRY.values())
        _AUTO_TRACE_REGISTRY.clear()
    for ctx in contexts_to_close:
        try:
            _close_trace_safely(ctx)
        except Exception:
            pass

    # Drain any leftover web traces too. Normally the middleware closes these
    # per-request, but a cleanup call (e.g. at scenario end or process exit)
    # is the right point to flush any that the middleware missed.
    with _WEB_AUTO_TRACE_REGISTRY_LOCK:
        web_to_close = list(_WEB_AUTO_TRACE_REGISTRY.values())
        _WEB_AUTO_TRACE_REGISTRY.clear()
    for ctx in web_to_close:
        try:
            _close_trace_safely(ctx)
        except Exception:
            pass


def get_current_auto_trace_context() -> AutoTraceContext | None:
    """Get the current auto-trace context."""
    return _AUTO_TRACE_CONTEXT.get()


def close_web_trace_on_request_completion(error: BaseException | None = None) -> None:
    """
    Close any open auto-trace when a web request handler returns a response.
    This is called by framework-specific response hooks (Flask, FastAPI, etc.)
    when they detect a response is being returned.
    Args:
        error: Optional exception that occurred, to mark trace as failed
    """
    auto_context = _AUTO_TRACE_CONTEXT.get()
    if auto_context is not None and auto_context.entry_frame_id.startswith("web:"):
        _AUTO_TRACE_CONTEXT.set(None)
        _WEB_ROUTE_WHEN_TRACE_CREATED.set(None)
        _close_trace_safely(auto_context, error=error)
        # Also drop the per-token registry entry if any.
        try:
            token = WEB_REQUEST_TOKEN.get()
            if token:
                with _WEB_AUTO_TRACE_REGISTRY_LOCK:
                    _WEB_AUTO_TRACE_REGISTRY.pop(token, None)
        except Exception:
            pass
        return

    # ContextVar is empty — most likely the trace was created in a sync handler
    # dispatched to a threadpool (FastAPI's default), so its `_AUTO_TRACE_CONTEXT`
    # set never propagated back to the event loop. Fall back to the per-request
    # registry, which is keyed by the token the middleware set up-front.
    try:
        token = WEB_REQUEST_TOKEN.get()
    except Exception:
        token = None
    if not token:
        return
    with _WEB_AUTO_TRACE_REGISTRY_LOCK:
        ctx = _WEB_AUTO_TRACE_REGISTRY.pop(token, None)
    if ctx is not None:
        _close_trace_safely(ctx, error=error)
