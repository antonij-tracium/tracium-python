"""
Thread helpers for explicit Tracium context propagation.

Tracium relies on Python's :mod:`contextvars` for trace/span propagation. ContextVars
flow naturally across ``await`` boundaries and ``asyncio`` tasks but **do not**
propagate to threads spawned via :class:`threading.Thread` — that machinery starts
each thread with an empty context.

These helpers use :func:`contextvars.copy_context` to capture the parent's context
and run the target inside it, so the new thread sees the same active trace and
current span as its parent. They never monkey-patch the standard library, so
debuggers (pdb, debugpy/VSCode, PyCharm) keep working.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import threading
from collections.abc import Callable, Iterator
from typing import TypeVar

T = TypeVar("T")

__all__ = [
    "run_in_thread",
    "with_context",
    "ContextThread",
    "get_current_trace_context",
    "with_trace_context",
]


def run_in_thread(func: Callable[..., T], *args, **kwargs) -> threading.Thread:
    """
    Start a new thread that runs ``func`` with the caller's contextvars.

    Drop-in alternative to ``threading.Thread(target=...)`` + ``.start()`` when
    the target needs access to the active Tracium trace.
    """
    ctx = contextvars.copy_context()
    thread = threading.Thread(target=lambda: ctx.run(func, *args, **kwargs))
    thread.start()
    return thread


def with_context(func: Callable[..., T]) -> Callable[..., T]:
    """
    Wrap ``func`` so it runs inside a snapshot of the context that was active
    when :func:`with_context` was *called*. Intended for handing a callable to a
    raw thread without losing the parent trace::

        thread = threading.Thread(target=with_context(worker))
        thread.start()

    Wrapping happens here, inside the parent context, so the snapshot captures
    the live trace/span. The thread later runs the wrapper, which restores the
    snapshot before invoking ``func``.

    Note: applying ``@with_context`` at module-import time captures the empty
    import-time context — useful for very few cases. Prefer the call-style usage
    above, or :func:`run_in_thread` / :class:`ContextThread`.
    """
    ctx = contextvars.copy_context()

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return ctx.run(func, *args, **kwargs)

    return wrapper


def get_current_trace_context() -> dict | None:
    """
    Snapshot the active trace context into a plain dict suitable for sending
    over a process / network boundary (Celery task args, multiprocessing,
    subprocess env, HTTP headers, message queues).

    Returns ``None`` when no trace is active. The dict contains ``trace_id``
    and ``parent_span_id`` (the latter may be ``None`` if no span is open).

    Pair with :func:`with_trace_context` on the receiving side to attach
    child spans to the parent trace across process boundaries.
    """
    from ..context.trace_context import CURRENT_SPAN, CURRENT_TRACE_STATE

    state = CURRENT_TRACE_STATE.get()
    if state is None:
        return None
    span = CURRENT_SPAN.get()
    return {
        "trace_id": state.trace_id,
        "parent_span_id": span[0] if span else None,
        "agent_name": state.agent_name,
    }


@contextlib.contextmanager
def with_trace_context(
    trace_id: str,
    *,
    parent_span_id: str | None = None,
    agent_name: str | None = None,
) -> Iterator[object]:
    """
    Bind the current block (and anything it calls) to an existing trace from
    another process or system. Use this on the receiving side of a process /
    network boundary; the parent should publish its context with
    :func:`get_current_trace_context`.

    Example — Celery worker in a separate process::

        @celery.task
        def my_task(ctx, payload):
            with tracium.with_trace_context(**ctx):
                run_agent(payload)  # auto-instrumented calls join parent trace

    Requires :func:`tracium.init` to have been called so a client is available.
    No-op (still yields) if no client is configured.
    """
    from ..context.trace_context import CURRENT_SPAN, CURRENT_TRACE_STATE
    from ..helpers.global_state import get_client
    from ..models.trace_handle import AgentTraceHandle
    from ..models.trace_state import TraceState

    try:
        client = get_client()
    except Exception:
        yield None
        return
    if client is None:
        yield None
        return

    state = TraceState(
        client=client,
        trace_id=trace_id,
        agent_name=agent_name or "external",
        remote_started=True,
    )
    state_token = CURRENT_TRACE_STATE.set(state)
    span_token = None
    if parent_span_id:
        span_token = CURRENT_SPAN.set((parent_span_id, 0))
    try:
        yield AgentTraceHandle(state)
    finally:
        if span_token is not None:
            CURRENT_SPAN.reset(span_token)
        CURRENT_TRACE_STATE.reset(state_token)


class ContextThread(threading.Thread):
    """
    :class:`threading.Thread` subclass that captures the parent context at
    construction time and runs the target inside it.

    Use this when you want to subclass-style threads but still inherit the
    caller's active trace. ``threading.Thread`` itself is never monkey-patched.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tracium_ctx = contextvars.copy_context()

    def run(self):
        self._tracium_ctx.run(super().run)
