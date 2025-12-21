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
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core import TraciumClient
from ..models.trace_handle import AgentTraceManager

_AUTO_TRACE_CONTEXT: contextvars.ContextVar[AutoTraceContext | None] = contextvars.ContextVar(
    "tracium_auto_trace_context",
    default=None,
)

_GLOBAL_AUTO_TRACE: AutoTraceContext | None = None
_GLOBAL_AUTO_TRACE_LOCK = threading.Lock()

_CLEANUP_REGISTERED = False
_ORIGINAL_EXCEPTHOOK = None
_ORIGINAL_ASYNCIO_HANDLER = None

_SKIP_PATTERNS = [
    "src/tracium/",
    "openai",
    "anthropic",
    "google",
    "langchain",
    "langgraph",
    "threading",
    "concurrent",
    "asyncio",
    "site-packages",
]


def _close_trace_safely(context: AutoTraceContext | None) -> None:
    """Safely close a trace context, ignoring any errors."""
    if context is None:
        return
    try:
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

    This is called on normal program exit (via atexit). We complete traces
    normally rather than marking them as failed, since the program completed
    successfully (just without explicitly closing the trace).
    """
    global _GLOBAL_AUTO_TRACE

    with _GLOBAL_AUTO_TRACE_LOCK:
        context = _GLOBAL_AUTO_TRACE
        _GLOBAL_AUTO_TRACE = None
        _close_trace_safely(context)

    auto_context = _AUTO_TRACE_CONTEXT.get()
    if auto_context is not None and auto_context is not context:
        _close_trace_safely(auto_context)
    _AUTO_TRACE_CONTEXT.set(None)


def _close_trace_on_exception(exc_type, exc_value, exc_traceback) -> None:
    """Close any open auto-traces and mark them as failed."""
    global _GLOBAL_AUTO_TRACE

    with _GLOBAL_AUTO_TRACE_LOCK:
        context = _GLOBAL_AUTO_TRACE
        _GLOBAL_AUTO_TRACE = None

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


def _exception_handler(exc_type, exc_value, exc_traceback) -> None:
    """
    Global exception handler to ensure traces are closed on unhandled exceptions.

    This handler chains with any existing exception handler to avoid conflicts
    with other libraries that might also set sys.excepthook.
    """
    _close_trace_on_exception(exc_type, exc_value, exc_traceback)

    if _ORIGINAL_EXCEPTHOOK is not None:
        try:
            _ORIGINAL_EXCEPTHOOK(exc_type, exc_value, exc_traceback)
        except Exception:
            import sys

            sys.__excepthook__(exc_type, exc_value, exc_traceback)


def _asyncio_exception_handler(loop, context: dict) -> None:
    """
    Asyncio exception handler to ensure traces are closed on unhandled async exceptions.

    This catches exceptions in asyncio tasks that don't trigger sys.excepthook.
    Chains with any existing asyncio exception handler.
    """
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
    """
    Register asyncio exception handler for unhandled async exceptions.

    This is called both during initial setup and when event loops are created.
    """
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
                        _ORIGINAL_ASYNCIO_HANDLER = handler or loop.default_exception_handler  # noqa: N806
                        loop.set_exception_handler(_asyncio_exception_handler)
                    return loop

                patched._tracium_patched = True
                asyncio.new_event_loop = patched
            return

    handler = loop.get_exception_handler()
    if handler is not _asyncio_exception_handler:
        _ORIGINAL_ASYNCIO_HANDLER = handler or loop.default_exception_handler
        loop.set_exception_handler(_asyncio_exception_handler)


def _register_cleanup() -> None:
    """
    Register the cleanup handler if not already registered.

    Safely sets up exception handling by chaining with any existing excepthook
    and asyncio exception handler, ensuring we don't break other libraries that
    might also override these handlers.
    """
    global _CLEANUP_REGISTERED, _ORIGINAL_EXCEPTHOOK
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
    llm_call_count: int = 0

    def increment_call(self) -> None:
        """Increment the LLM call counter."""
        self.llm_call_count += 1


def _get_user_frames() -> list[tuple]:
    """Extract user frames from the call stack, skipping internal/library frames."""
    frame = inspect.currentframe()
    user_frames = []
    try:
        while frame is not None:
            filename = frame.f_code.co_filename
            if not any(pattern in filename for pattern in _SKIP_PATTERNS):
                user_frames.append((frame.f_code.co_name, filename, frame.f_lineno, id(frame)))
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

    return user_frames[0][:3]


def _find_workflow_entry_point() -> tuple[str, str]:
    """
    Find the entry point of the current workflow by examining the call stack.

    Returns a unique identifier for the entry frame and a human-readable name.
    """
    user_frames = _get_user_frames()
    if not user_frames:
        return str(uuid.uuid4()), "workflow"

    entry_patterns = ["main", "run_", "pipeline", "workflow", "process", "execute"]

    for func_name, file_path, line_no, _ in reversed(user_frames):
        if any(pattern in func_name.lower() for pattern in entry_patterns):
            return f"{file_path}:{func_name}:{line_no}", func_name

    entry_function, entry_file, entry_line, _ = user_frames[-1]
    frame_key = f"{entry_file}:{entry_function}:{entry_line}"

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


def get_or_create_auto_trace(
    client: TraciumClient,
    agent_name: str,
    model_id: str | None = None,
    tags: list[str] | None = None,
    version: str | None = None,
) -> tuple[Any, bool]:
    """
    Get the current auto-created trace, or create one if needed.

    Uses a global shared state to ensure all threads/tasks in the same
    workflow share the same trace.

    Returns:
        tuple: (trace_handle, created_new_trace)
    """
    global _GLOBAL_AUTO_TRACE

    from ..context.trace_context import current_trace
    from ..helpers.global_state import get_options
    from ..instrumentation.auto_detection import detect_agent_name

    manual_trace = current_trace()
    if manual_trace is not None:
        return manual_trace, False

    auto_context = _AUTO_TRACE_CONTEXT.get()
    if auto_context is not None:
        auto_context.increment_call()
        return auto_context.trace_handle, False

    with _GLOBAL_AUTO_TRACE_LOCK:
        if _GLOBAL_AUTO_TRACE is not None:
            _GLOBAL_AUTO_TRACE.increment_call()
            _AUTO_TRACE_CONTEXT.set(_GLOBAL_AUTO_TRACE)
            return _GLOBAL_AUTO_TRACE.trace_handle, False

        entry_frame_id, entry_function_name = _find_workflow_entry_point()

        try:
            default_agent_name = get_options().default_agent_name
        except RuntimeError:
            default_agent_name = "app"

        if agent_name == "app" or agent_name == default_agent_name or not agent_name:
            if entry_function_name in ("workflow", "<module>") and ":" in entry_frame_id:
                file_path = entry_frame_id.split(":")[0]
                if file_path and Path(file_path).exists():
                    script_name = Path(file_path).stem
                    if script_name == "__main__":
                        script_name = Path(file_path).name
                        if script_name.endswith(".py"):
                            script_name = script_name[:-3]
                    if script_name and script_name != "__main__":
                        agent_name = script_name.replace("_", "-")

            if not agent_name or agent_name in ("app", default_agent_name):
                if entry_function_name and entry_function_name not in ("workflow", "<module>"):
                    entry_patterns = ["main", "run_", "pipeline", "workflow", "process", "execute"]
                    if any(pattern in entry_function_name.lower() for pattern in entry_patterns):
                        normalized = entry_function_name.replace("_", "-")
                        if normalized.startswith("test-"):
                            normalized = normalized[5:]
                        if normalized.endswith("-main"):
                            normalized = normalized[:-5]
                        if normalized and normalized not in ("main", "<module>"):
                            agent_name = normalized
                    else:
                        agent_name = entry_function_name.replace("_", "-")

            if not agent_name or agent_name in ("app", default_agent_name, "main"):
                agent_name = detect_agent_name(default_agent_name)

        if version is None:
            try:
                version = get_options().default_version
            except RuntimeError:
                version = None

        _register_cleanup()

        trace_manager = client.agent_trace(
            agent_name=agent_name,
            model_id=model_id,
            tags=tags or [],
            version=version,
        )
        trace_handle = trace_manager.__enter__()

        auto_context = AutoTraceContext(
            trace_manager=trace_manager,
            trace_handle=trace_handle,
            entry_frame_id=entry_frame_id,
            llm_call_count=1,
        )
        _GLOBAL_AUTO_TRACE = auto_context
        _AUTO_TRACE_CONTEXT.set(auto_context)

        return trace_handle, True


def should_close_auto_trace() -> bool:
    """
    Determine if we should close the current auto-created trace.

    This happens when we're exiting the entry point function.
    """
    global _GLOBAL_AUTO_TRACE

    with _GLOBAL_AUTO_TRACE_LOCK:
        if _GLOBAL_AUTO_TRACE is None:
            return False

        current_frame_id, _ = _find_workflow_entry_point()

        return current_frame_id != _GLOBAL_AUTO_TRACE.entry_frame_id


def close_auto_trace_if_needed() -> None:
    """Close the auto-created trace if we've exited the workflow."""
    global _GLOBAL_AUTO_TRACE

    if not should_close_auto_trace():
        return

    with _GLOBAL_AUTO_TRACE_LOCK:
        context = _GLOBAL_AUTO_TRACE
        _GLOBAL_AUTO_TRACE = None
        _AUTO_TRACE_CONTEXT.set(None)
        _close_trace_safely(context)


def get_current_function_for_span() -> str:
    """
    Get the current function name to use as a span name.

    This looks up the call stack to find the user's function.
    """
    function_name, _, _ = _get_caller_info()
    return function_name


def cleanup_auto_trace() -> None:
    """Force cleanup of any auto-created trace. Used for testing."""
    global _GLOBAL_AUTO_TRACE

    with _GLOBAL_AUTO_TRACE_LOCK:
        context = _GLOBAL_AUTO_TRACE
        _GLOBAL_AUTO_TRACE = None
        _close_trace_safely(context)

    auto_context = _AUTO_TRACE_CONTEXT.get()
    if auto_context is not None and auto_context is not context:
        _close_trace_safely(auto_context)
    _AUTO_TRACE_CONTEXT.set(None)


def get_current_auto_trace_context():
    """Get the current auto-trace context."""
    return _AUTO_TRACE_CONTEXT.get()
