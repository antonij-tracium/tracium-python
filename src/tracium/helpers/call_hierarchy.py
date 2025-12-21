"""
Smart call hierarchy detection for automatic span nesting.

This module analyzes the call stack to create proper parent-child relationships
between spans without requiring explicit user code changes.
"""

from __future__ import annotations

import contextvars
import inspect
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

_FUNCTION_CONTEXT_SPANS: contextvars.ContextVar[dict[str, str]] = contextvars.ContextVar(
    "tracium_function_context_spans",
    default=None,
)

_FUNCTION_SPAN_CACHE: dict[str, str] = {}
_CACHE_LOCK = threading.Lock()

_FUNCTION_CONTEXT_TRACE_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "tracium_function_context_trace",
    default=None,
)

_INVOCATION_COUNTER: contextvars.ContextVar[dict[str, int]] = contextvars.ContextVar(
    "tracium_invocation_count",
    default=None,
)


@dataclass(frozen=True)
class FrameInfo:
    """Represents a user frame on the call stack."""

    function_name: str
    file_path: str
    line_number: int
    frame_id: int
    invocation_id: str

    def context_key(self, thread_id: int) -> str:
        """
        Generate a key that uniquely identifies this frame invocation, even across
        parallel executions of the same function.

        Uses frame_id, thread_id, and invocation_id to ensure uniqueness.
        """
        return f"{self.file_path}:{self.function_name}:{self.frame_id}:thread_{thread_id}:inv_{self.invocation_id}"


def _get_function_context_map() -> dict[str, str]:
    """Get or create the function context map for this execution context."""
    ctx_map = _FUNCTION_CONTEXT_SPANS.get()
    if ctx_map is None:
        ctx_map = {}
        _FUNCTION_CONTEXT_SPANS.set(ctx_map)
    return ctx_map


def _reset_function_context(trace_id: str | None = None) -> dict[str, str]:
    """
    Clear the cached function context map and associate it with the current trace.
    Also resets the invocation counter for a fresh start.
    """
    ctx_map: dict[str, str] = {}
    _FUNCTION_CONTEXT_SPANS.set(ctx_map)
    _FUNCTION_CONTEXT_TRACE_ID.set(trace_id)
    _INVOCATION_COUNTER.set({})
    return ctx_map


def _get_invocation_counter() -> dict[str, int]:
    """Get or create the invocation counter for this execution context."""
    counter = _INVOCATION_COUNTER.get()
    if counter is None:
        counter = {}
        _INVOCATION_COUNTER.set(counter)
    return counter


def _get_invocation_id(function_name: str, file_path: str, line_number: int) -> str:
    """
    Get a unique invocation ID for this specific function call.

    This ensures that parallel executions of the same function get different IDs.
    """
    counter = _get_invocation_counter()
    func_key = f"{file_path}:{function_name}:{line_number}"

    invocation_num = counter.get(func_key, 0) + 1
    counter[func_key] = invocation_num

    timestamp = int(time.time() * 1_000_000)
    return f"{invocation_num}_{timestamp}"


def _get_user_call_hierarchy() -> list[FrameInfo]:
    """
    Get the call hierarchy of user functions (excluding internal libraries).

    Returns:
        list of FrameInfo objects from deepest to shallowest
    """
    frame = inspect.currentframe()
    user_frames = []

    try:
        while frame is not None:
            code = frame.f_code
            filename = code.co_filename
            function_name = code.co_name

            skip_patterns = [
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

            if any(pattern in filename for pattern in skip_patterns):
                frame = frame.f_back
                continue

            if function_name in {"<module>", "__enter__", "__exit__"}:
                frame = frame.f_back
                continue

            invocation_id = _get_invocation_id(function_name, filename, frame.f_lineno)

            user_frames.append(
                FrameInfo(
                    function_name=function_name,
                    file_path=filename,
                    line_number=frame.f_lineno,
                    frame_id=id(frame),
                    invocation_id=invocation_id,
                )
            )
            frame = frame.f_back

        return user_frames
    finally:
        del frame


def get_or_create_function_span(
    trace_handle: Any,
    llm_call_name: str,
) -> tuple[str | None, str]:
    """
    Get or create a function span for the current call context.

    This analyzes the call stack to find user functions that should be parent spans.
    For example, if we have:
        research_topic() -> call_openai() -> [OpenAI API call]

    We want to create:
        - research_topic (function span, parent_id=None)
          └─ call_openai (llm span, parent_id=research_topic_span_id)

    Args:
        trace_handle: The agent trace handle
        llm_call_name: The name of the immediate calling function

    Returns:
        tuple of (parent_span_id, span_name)
            - parent_span_id: The span ID to use as parent for the LLM call
            - span_name: The name to use for the LLM span
    """
    trace_id = getattr(trace_handle, "id", None)
    ctx_map = _get_function_context_map()
    cached_trace_id = _FUNCTION_CONTEXT_TRACE_ID.get()

    if trace_id is not None and trace_id != cached_trace_id:
        ctx_map = _reset_function_context(trace_id)
    elif trace_id is None and cached_trace_id is not None:
        ctx_map = _reset_function_context(None)

    call_hierarchy = _get_user_call_hierarchy()

    if not call_hierarchy:
        return None, llm_call_name

    caller_frame = call_hierarchy[0]

    if len(call_hierarchy) == 1:
        return None, caller_frame.function_name

    parent_frames = call_hierarchy[1:]
    thread_id = threading.get_ident()
    active_keys: set[str] = set()
    parent_span_id: str | None = None

    for depth_level, frame_info in enumerate(reversed(parent_frames)):
        context_key = frame_info.context_key(thread_id)
        active_keys.add(context_key)
        span_id = ctx_map.get(context_key)

        if span_id is None:
            span_id = str(uuid.uuid4())
            trace_handle.record_span(
                span_type="function",
                name=frame_info.function_name,
                parent_span_id=parent_span_id,
                span_id=span_id,
                depth_level=depth_level,
                input={
                    "function": frame_info.function_name,
                    "file": frame_info.file_path,
                    "line": frame_info.line_number,
                },
                output={"status": "in_progress"},
            )
            ctx_map[context_key] = span_id

        parent_span_id = span_id

    stale_keys = [key for key in list(ctx_map.keys()) if key not in active_keys]
    for key in stale_keys:
        ctx_map.pop(key, None)

    span_name = caller_frame.function_name or llm_call_name
    return parent_span_id, span_name


def clear_function_context():
    """Clear the function context map. Used for testing or at the end of a workflow."""
    _FUNCTION_CONTEXT_SPANS.set({})
    _FUNCTION_CONTEXT_TRACE_ID.set(None)
    _INVOCATION_COUNTER.set({})
    with _CACHE_LOCK:
        _FUNCTION_SPAN_CACHE.clear()
