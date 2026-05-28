"""
Trace context management using contextvars.
"""

from __future__ import annotations

import contextvars
import logging
import threading

from ..models.trace_handle import AgentTraceHandle
from ..models.trace_state import TraceState

logger = logging.getLogger(__name__)

CURRENT_TRACE_STATE: contextvars.ContextVar[TraceState | None] = contextvars.ContextVar(
    "tracium_current_trace_state",
    default=None,
)

CURRENT_SPAN: contextvars.ContextVar[tuple[str, int] | None] = contextvars.ContextVar(
    "tracium_current_span",
    default=None,
)

_RAW_THREAD_WARNING_EMITTED = False


def current_parent_span_id() -> str | None:
    v = CURRENT_SPAN.get()
    return v[0] if v else None


def current_depth_level() -> int:
    """Depth a *new child* would have under the current span (0 = root)."""
    v = CURRENT_SPAN.get()
    return (v[1] + 1) if v else 0


def current_trace() -> AgentTraceHandle | None:
    state = CURRENT_TRACE_STATE.get()
    if state is None:
        _maybe_warn_raw_thread()
        return None
    return AgentTraceHandle(state)


def _maybe_warn_raw_thread() -> None:
    global _RAW_THREAD_WARNING_EMITTED
    if _RAW_THREAD_WARNING_EMITTED:
        return
    try:
        if threading.current_thread() is threading.main_thread():
            return
        from ..helpers.global_state import STATE

        if STATE.client is None:
            return
    except Exception:
        return
    _RAW_THREAD_WARNING_EMITTED = True
    logger.info(
        "tracium: current_trace() returned None inside a non-main thread. "
        "Tracium auto-patches threading.Thread and ThreadPoolExecutor.submit "
        "to propagate context, but this can be disabled via "
        "TRACIUM_DISABLE_THREAD_PROPAGATION=1, and threads constructed before "
        "tracium.init() / tracium.trace() are not patched. If you've disabled "
        "propagation or are using a custom thread primitive, spawn threads "
        "with tracium.run_in_thread / tracium.ContextThread, or wrap targets "
        "with tracium.with_context."
    )
