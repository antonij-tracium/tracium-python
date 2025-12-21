"""
Trace context management using contextvars.
"""

from __future__ import annotations

import contextvars

from ..models.trace_handle import AgentTraceHandle
from ..models.trace_state import TraceState

CURRENT_TRACE_STATE: contextvars.ContextVar[TraceState | None] = contextvars.ContextVar(
    "tracium_current_trace_state",
    default=None,
)


def current_trace() -> AgentTraceHandle | None:
    state = CURRENT_TRACE_STATE.get()
    if state is None:
        return None
    return AgentTraceHandle(state)
