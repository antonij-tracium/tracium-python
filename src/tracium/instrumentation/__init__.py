"""
Instrumentation modules for Tracium SDK.
"""

from .auto_detection import detect_agent_name
from .auto_instrumentation import configure_auto_instrumentation
from .auto_trace_tracker import AutoTraceContext, get_current_auto_trace_context
from .decorators import agent_span, agent_trace, span

__all__ = [
    "detect_agent_name",
    "configure_auto_instrumentation",
    "AutoTraceContext",
    "get_current_auto_trace_context",
    "agent_trace",
    "agent_span",
    "span",
]
