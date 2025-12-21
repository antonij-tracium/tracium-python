"""
Helper utilities for working with threads in Tracium.

This module provides utilities to make threading work seamlessly with Tracium,
regardless of which threading approach you use.
"""

from __future__ import annotations

import functools
import threading
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

__all__ = [
    "run_in_thread",
    "with_context",
    "ContextThread",
    "patch_threading_module",
    "unpatch_threading_module",
]


def run_in_thread(func: Callable[..., T], *args, **kwargs) -> threading.Thread:
    """
    Run a function in a new thread with automatic context propagation.

    This is a drop-in replacement for threading.Thread that automatically
    propagates Tracium's context (including current_trace) to the new thread,
    with each thread getting its own copy of the span_stack to prevent race conditions.

    Args:
        func: The function to run in a thread
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The started thread

    Example:
        >>> def my_task(data):
        ...     trace = current_trace()
        ...     with trace.span(span_type="task", name="my_task"):
        ...         process(data)
        ...
        >>> thread = run_in_thread(my_task, "some data")
        >>> thread.join()
    """
    from ..context.trace_context import CURRENT_TRACE_STATE as _CURRENT_TRACE_STATE
    from ..instrumentation.auto_trace_tracker import _AUTO_TRACE_CONTEXT

    current_state = _CURRENT_TRACE_STATE.get()
    auto_trace_context = _AUTO_TRACE_CONTEXT.get()

    def wrapper():
        if current_state is not None:
            thread_safe_state = current_state.copy_for_thread()
            _CURRENT_TRACE_STATE.set(thread_safe_state)

        if auto_trace_context is not None:
            _AUTO_TRACE_CONTEXT.set(auto_trace_context)

        return func(*args, **kwargs)

    thread = threading.Thread(target=wrapper)
    thread.start()
    return thread


def with_context(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator that ensures a function preserves Tracium context when called.

    Use this decorator on any function that will be called from a thread
    to ensure it has access to the current trace, with each thread getting its
    own copy of the span_stack to prevent race conditions.

    Args:
        func: The function to wrap

    Returns:
        Wrapped function that preserves context

    Example:
        >>> @with_context
        ... def worker_function(data):
        ...     trace = current_trace()
        ...     with trace.span(span_type="worker", name="process"):
        ...         return process(data)
        ...
        >>> thread = threading.Thread(target=worker_function, args=("data",))
        >>> thread.start()
    """
    from ..context.trace_context import CURRENT_TRACE_STATE as _CURRENT_TRACE_STATE
    from ..instrumentation.auto_trace_tracker import _AUTO_TRACE_CONTEXT

    captured_state = _CURRENT_TRACE_STATE.get()
    captured_auto_trace = _AUTO_TRACE_CONTEXT.get()

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if captured_state is not None:
            thread_safe_state = captured_state.copy_for_thread()
            _CURRENT_TRACE_STATE.set(thread_safe_state)

        if captured_auto_trace is not None:
            _AUTO_TRACE_CONTEXT.set(captured_auto_trace)

        return func(*args, **kwargs)

    return wrapper


class ContextThread(threading.Thread):
    """
    A Thread subclass that automatically propagates contextvars.

    This is a drop-in replacement for threading.Thread that ensures
    Tracium's context is available in the thread, with each thread getting
    its own copy of the span_stack to prevent race conditions.

    Example:
        >>> def my_task():
        ...     trace = current_trace()
        ...     with trace.span(span_type="task"):
        ...         pass
        ...
        >>> thread = ContextThread(target=my_task)
        >>> thread.start()
        >>> thread.join()
    """

    def __init__(self, *args, **kwargs):
        from ..context.trace_context import CURRENT_TRACE_STATE
        from ..instrumentation.auto_trace_tracker import _AUTO_TRACE_CONTEXT

        self._captured_state = CURRENT_TRACE_STATE.get()
        self._captured_auto_trace = _AUTO_TRACE_CONTEXT.get()
        super().__init__(*args, **kwargs)

    def run(self):
        """Run the thread's target with a thread-safe copy of the trace state."""
        from ..context.trace_context import CURRENT_TRACE_STATE
        from ..instrumentation.auto_trace_tracker import _AUTO_TRACE_CONTEXT

        if self._captured_state is not None:
            thread_safe_state = self._captured_state.copy_for_thread()
            CURRENT_TRACE_STATE.set(thread_safe_state)

        if self._captured_auto_trace is not None:
            _AUTO_TRACE_CONTEXT.set(self._captured_auto_trace)

        super().run()


def patch_threading_module():
    """
    Monkey-patch the threading module to use ContextThread by default.

    After calling this, all uses of threading.Thread will automatically
    propagate context. This is called automatically by tracium.init().
    """
    import threading as threading_module

    if threading_module.Thread is ContextThread:
        return

    if not hasattr(threading_module, "_original_Thread"):
        threading_module._original_Thread = threading_module.Thread  # type: ignore[attr-defined]

    threading_module.Thread = ContextThread


def unpatch_threading_module():
    """
    Restore the original threading.Thread class.

    This is mainly useful for testing.
    """
    import threading as threading_module

    if hasattr(threading_module, "_original_Thread"):
        threading_module.Thread = threading_module._original_Thread  # type: ignore[attr-defined]
        delattr(threading_module, "_original_Thread")
