"""
Bridge from an :class:`LLMCall` to a Tracium span on the currently active trace.

Single concern: translate the normalized call into a span with correct timing
and a span_type the backend will accept.

The backend rejects ``span_type="llm"`` without a ``model_id`` (it can't compute
cost without one), so when the parser couldn't extract a model we emit the span
under ``span_type="custom"`` instead. We never silently drop a captured call.
"""

from __future__ import annotations

import logging
from datetime import datetime

from ...context.trace_context import current_trace
from ...utils.datetime_utils import _duration_ms, _utcnow
from .providers import LLMCall

logger = logging.getLogger(__name__)


def emit_llm_span(call: LLMCall, started_at: datetime, ended_at: datetime | None = None) -> None:
    """Record an LLM span on the active trace, if any.

    ``started_at`` should be the moment the underlying HTTP call began; the
    span's reported timing matches the real API duration, not the microseconds
    we spent emitting.

    Silently no-ops if there's no active trace, or if anything goes wrong: the
    SDK must never break the user's HTTP call.
    """
    trace = current_trace()
    if trace is None:
        return

    ended_at = ended_at or _utcnow()
    name = _name_for(call)
    span_type = "llm" if call.model else "custom"
    try:
        latency_ms = _duration_ms(started_at, ended_at)
    except Exception:
        latency_ms = None

    try:
        with trace.span(
            span_type=span_type,
            name=name,
            input=call.input,
            model_id=call.model,
            input_tokens=call.input_tokens,
            output_tokens=call.output_tokens,
            cached_input_tokens=call.cached_input_tokens,
            cache_creation_input_tokens=call.cache_creation_input_tokens,
            started_at=started_at,
            latency_ms=latency_ms,
        ) as span:
            if call.tools:
                span.set_tools(call.tools)
            if call.tool_calls:
                span.set_tool_calls(call.tool_calls)
            if call.output is not None:
                span.record_output(call.output)
            if call.error:
                span.mark_failed(call.error)
    except Exception as e:
        logger.debug("tracium http_capture emit failed (ignored): %s: %s", type(e).__name__, e)


def _name_for(call: LLMCall) -> str:
    """Human-readable span name like ``openai.chat`` or ``groq.embedding``."""
    return f"{call.provider}.{call.operation}"
