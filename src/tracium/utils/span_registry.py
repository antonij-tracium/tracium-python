"""
Span registry for tracking sent spans across threads.
"""

import threading
import time
from typing import TYPE_CHECKING

from ..helpers.logging_config import get_logger

if TYPE_CHECKING:
    from ..core.client import TraciumClient

logger = get_logger()

_sent_spans: dict[str, set[str]] = {}
_sent_spans_lock = threading.Lock()


def mark_span_sent(trace_id: str, span_id: str) -> None:
    """Mark a span as having been sent to the API."""
    with _sent_spans_lock:
        if trace_id not in _sent_spans:
            _sent_spans[trace_id] = set()
        _sent_spans[trace_id].add(span_id)


def is_span_sent(trace_id: str, span_id: str) -> bool:
    """Check if a span has been sent to the API."""
    with _sent_spans_lock:
        return trace_id in _sent_spans and span_id in _sent_spans[trace_id]


def ensure_parent_span_sent(
    trace_id: str, parent_span_id: str, client: "TraciumClient", max_wait_seconds: float = 2.0
) -> None:
    """
    Ensure a parent span has been sent to the API before proceeding.

    This function waits (with exponential backoff) for the parent span to be sent,
    which is critical in parallel execution scenarios where parent and child spans
    might be created in different threads.

    Args:
        trace_id: The trace ID
        parent_span_id: The parent span ID to wait for
        client: The Tracium client
        max_wait_seconds: Maximum time to wait for parent span (default 2 seconds)
    """
    if is_span_sent(trace_id, parent_span_id):
        return

    wait_time = 0.01
    total_wait = 0.0
    max_wait = max_wait_seconds

    while total_wait < max_wait:
        if is_span_sent(trace_id, parent_span_id):
            return

        time.sleep(wait_time)
        total_wait += wait_time
        wait_time = min(wait_time * 1.5, 0.1)

    if not is_span_sent(trace_id, parent_span_id):
        logger.warning(
            "Parent span %s not found in sent spans registry after waiting %.2f seconds. "
            "Proceeding anyway - backend will validate parent existence.",
            parent_span_id,
            total_wait,
        )
