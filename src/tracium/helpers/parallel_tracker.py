"""
Automatic parallel execution tracking for Tracium SDK.

This module automatically detects when spans are being executed in parallel
(e.g., in ThreadPoolExecutor, asyncio.gather, etc.) and assigns appropriate
parallel_group_id and sequence_number without requiring user intervention.
"""

from __future__ import annotations

import contextvars
import threading
import time
import uuid
from dataclasses import dataclass, field

_parallel_context: contextvars.ContextVar[ParallelContext | None] = contextvars.ContextVar(
    "tracium_parallel_context",
    default=None,
)

_span_creation_registry: dict[str, SpanCreationRecord] = {}
_registry_lock = threading.Lock()

PARALLEL_DETECTION_WINDOW = 0.5


@dataclass
class SpanCreationRecord:
    """Record of when a span was created."""

    span_id: str
    parent_span_id: str | None
    created_at: float
    thread_id: int
    parallel_group_id: str | None = None
    sequence_number: int | None = None


@dataclass
class ParallelContext:
    """Context for tracking parallel execution."""

    group_id: str
    sequence_number: int = 0
    span_ids: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def get_next_sequence(self) -> int:
        """Get next sequence number in thread-safe way."""
        with self.lock:
            seq = self.sequence_number
            self.sequence_number += 1
            return seq


def create_parallel_context() -> ParallelContext:
    """
    Create a new parallel context for tracking parallel execution.

    This is typically called automatically when parallel execution is detected,
    but can also be called manually for explicit parallel blocks.
    """
    return ParallelContext(group_id=str(uuid.uuid4()))


def enter_parallel_context(context: ParallelContext | None = None) -> ParallelContext:
    """
    Enter a parallel execution context.

    Args:
        context: Existing context to use, or None to create a new one

    Returns:
        The parallel context that was entered
    """
    if context is None:
        context = create_parallel_context()

    _parallel_context.set(context)
    return context


def exit_parallel_context() -> None:
    """Exit the current parallel execution context."""
    _parallel_context.set(None)


def get_parallel_context() -> ParallelContext | None:
    """Get the current parallel execution context, if any."""
    return _parallel_context.get()


def recheck_parallelism_for_span(
    span_id: str,
    parent_span_id: str | None,
) -> tuple[str | None, int | None]:
    """
    Re-check for parallelism for a span that's about to be sent to the API.

    This is called right before sending a span to catch cases where parallelism
    was detected after the span was created but before it was sent.

    Args:
        span_id: The span ID to check
        parent_span_id: The parent span ID

    Returns:
        Tuple of (parallel_group_id, sequence_number) or (None, None) if not parallel
    """
    with _registry_lock:
        current_record = _span_creation_registry.get(span_id)
        if current_record is not None:
            if current_record.parallel_group_id is not None:
                return current_record.parallel_group_id, current_record.sequence_number

        current_time = time.time()
        current_thread = threading.get_ident()
        recent_siblings: list[SpanCreationRecord] = []

        for record in _span_creation_registry.values():
            if record.parent_span_id == parent_span_id and record.span_id != span_id:
                time_diff = abs(current_time - record.created_at)
                if time_diff <= PARALLEL_DETECTION_WINDOW:
                    recent_siblings.append(record)

        if recent_siblings:
            different_thread_siblings = [
                r for r in recent_siblings if r.thread_id != current_thread
            ]

            if different_thread_siblings:
                group_id = None
                for sibling in recent_siblings:
                    if sibling.parallel_group_id is not None:
                        group_id = sibling.parallel_group_id
                        break

                if group_id is None:
                    group_id = str(uuid.uuid4())

                if current_record is None:
                    current_record = SpanCreationRecord(
                        span_id=span_id,
                        parent_span_id=parent_span_id,
                        created_at=current_time,
                        thread_id=current_thread,
                    )
                    _span_creation_registry[span_id] = current_record

                all_siblings = recent_siblings + [current_record]
                sorted_all = sorted(all_siblings, key=lambda r: (r.created_at, r.thread_id))

                for i, sibling in enumerate(sorted_all):
                    if sibling.parallel_group_id is None:
                        sibling.parallel_group_id = group_id
                    if sibling.sequence_number is None:
                        sibling.sequence_number = i

                return group_id, current_record.sequence_number

    return None, None


def register_span_creation(
    span_id: str,
    parent_span_id: str | None,
    provided_parallel_group_id: str | None = None,
    provided_sequence_number: int | None = None,
) -> tuple[str | None, int | None]:
    """
    Register a span creation and automatically detect if it's part of parallel execution.

    This function:
    1. Checks if span is in an explicit parallel context
    2. Detects if span is being created in parallel with other spans
    3. Assigns parallel_group_id and sequence_number automatically

    Args:
        span_id: The span ID being created
        parent_span_id: The parent span ID
        provided_parallel_group_id: User-provided parallel group ID (takes precedence)
        provided_sequence_number: User-provided sequence number (takes precedence)

    Returns:
        Tuple of (parallel_group_id, sequence_number) or (None, None) if not parallel
    """
    if provided_parallel_group_id is not None and provided_sequence_number is not None:
        return provided_parallel_group_id, provided_sequence_number

    parallel_ctx = get_parallel_context()
    if parallel_ctx is not None:
        seq = parallel_ctx.get_next_sequence()
        parallel_ctx.span_ids.append(span_id)

        record = SpanCreationRecord(
            span_id=span_id,
            parent_span_id=parent_span_id,
            created_at=time.time(),
            thread_id=threading.get_ident(),
            parallel_group_id=parallel_ctx.group_id,
            sequence_number=seq,
        )

        with _registry_lock:
            _span_creation_registry[span_id] = record

        return parallel_ctx.group_id, seq

    current_time = time.time()
    current_thread = threading.get_ident()

    with _registry_lock:
        recent_siblings: list[SpanCreationRecord] = []

        for record in _span_creation_registry.values():
            if record.parent_span_id == parent_span_id:
                time_diff = abs(current_time - record.created_at)
                if time_diff <= PARALLEL_DETECTION_WINDOW:
                    recent_siblings.append(record)

        if recent_siblings:
            different_thread_siblings = [
                r for r in recent_siblings if r.thread_id != current_thread
            ]

            if different_thread_siblings:
                group_id = None

                for sibling in recent_siblings:
                    if sibling.parallel_group_id is not None:
                        group_id = sibling.parallel_group_id
                        break

                if group_id is None:
                    group_id = str(uuid.uuid4())

                record = SpanCreationRecord(
                    span_id=span_id,
                    parent_span_id=parent_span_id,
                    created_at=current_time,
                    thread_id=current_thread,
                )
                all_siblings = recent_siblings + [record]
                sorted_all = sorted(all_siblings, key=lambda r: (r.created_at, r.thread_id))

                for i, sibling in enumerate(sorted_all):
                    if sibling.parallel_group_id is None:
                        sibling.parallel_group_id = group_id
                    if sibling.sequence_number is None:
                        sibling.sequence_number = i

                seq = record.sequence_number

                _span_creation_registry[span_id] = record

                cutoff_time = current_time - 10.0
                old_span_ids = [
                    sid
                    for sid, rec in _span_creation_registry.items()
                    if rec.created_at < cutoff_time
                ]
                for sid in old_span_ids:
                    del _span_creation_registry[sid]

                return group_id, seq

        record = SpanCreationRecord(
            span_id=span_id,
            parent_span_id=parent_span_id,
            created_at=current_time,
            thread_id=current_thread,
        )
        _span_creation_registry[span_id] = record

        cutoff_time = current_time - 10.0
        old_span_ids = [
            sid for sid, rec in _span_creation_registry.items() if rec.created_at < cutoff_time
        ]
        for sid in old_span_ids:
            del _span_creation_registry[sid]

    return None, None


def cleanup_old_records(max_age_seconds: float = 10.0) -> None:
    """
    Clean up old span creation records to prevent memory leaks.

    Args:
        max_age_seconds: Maximum age of records to keep
    """
    cutoff_time = time.time() - max_age_seconds

    with _registry_lock:
        old_span_ids = [
            sid for sid, rec in _span_creation_registry.items() if rec.created_at < cutoff_time
        ]
        for sid in old_span_ids:
            del _span_creation_registry[sid]


class ParallelExecutionContext:
    """
    Context manager for explicit parallel execution tracking.

    Use this when you want to explicitly mark a block of code as parallel:

    Example:
        >>> from tracium.parallel_tracker import ParallelExecutionContext
        >>> with ParallelExecutionContext():
        ...     with trace.span(...):
        ...         pass
    """

    def __init__(self):
        self.context: ParallelContext | None = None
        self.token: contextvars.Token | None = None

    def __enter__(self) -> ParallelContext:
        self.context = create_parallel_context()
        self.token = _parallel_context.set(self.context)
        return self.context

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.token is not None:
            _parallel_context.reset(self.token)
        return False


def copy_context_to_thread(func):
    """
    Decorator to copy contextvars to a thread function.

    This solves the issue where contextvars don't propagate to threads.

    Example:
        >>> @copy_context_to_thread
        ... def my_thread_func():
        ...     trace = current_trace()
        ...     with trace.span(...):
        ...         pass
    """
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        ctx = contextvars.copy_context()
        return ctx.run(func, *args, **kwargs)

    return wrapper
