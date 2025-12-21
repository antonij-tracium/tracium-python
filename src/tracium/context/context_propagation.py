"""
Automatic context propagation for threads and async tasks.

This module ensures that Tracium's contextvars (like current_trace) are
automatically propagated to threads and async tasks, eliminating the need
for users to manually pass trace handles.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor as _OriginalThreadPoolExecutor
from typing import TypeVar

T = TypeVar("T")

__all__ = [
    "ContextPropagatingThreadPoolExecutor",
    "context_propagating_wrapper",
    "enable_automatic_context_propagation",
    "disable_automatic_context_propagation",
    "run_in_executor_with_context",
    "patch_thread_pool_executor",
    "unpatch_thread_pool_executor",
]

_original_thread_pool_executor = _OriginalThreadPoolExecutor


def _wrap_func_with_context(func: Callable[..., T]) -> Callable[..., T]:
    """
    Wrap a function to run in the current context with a thread-safe copy of the span stack.

    This ensures that contextvars are propagated to the function
    when it runs in a different thread or async task, with each thread
    getting its own copy of the span_stack to prevent race conditions.
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


class ContextPropagatingThreadPoolExecutor(_OriginalThreadPoolExecutor):
    """
    ThreadPoolExecutor that automatically propagates contextvars to worker threads.

    This is a drop-in replacement for concurrent.futures.ThreadPoolExecutor that
    ensures Tracium's contextvars (like current_trace) are available in worker threads.

    Importantly, this ensures each thread gets its own copy of the span_stack to
    prevent race conditions when multiple threads create nested spans.
    """

    def submit(self, fn, /, *args, **kwargs):
        """
        Submit a callable to be executed with the current context.

        The callable will have access to the same contextvars as the caller,
        including the current Tracium trace, but with a thread-safe copy of the
        span stack to prevent race conditions.
        """
        from ..context.trace_context import CURRENT_TRACE_STATE as _CURRENT_TRACE_STATE
        from ..instrumentation.auto_trace_tracker import _AUTO_TRACE_CONTEXT

        ctx = contextvars.copy_context()

        current_state = ctx.get(_CURRENT_TRACE_STATE)

        auto_trace_context = ctx.get(_AUTO_TRACE_CONTEXT)

        def context_wrapper():
            if current_state is not None:
                thread_safe_state = current_state.copy_for_thread()
                _CURRENT_TRACE_STATE.set(thread_safe_state)

            if auto_trace_context is not None:
                _AUTO_TRACE_CONTEXT.set(auto_trace_context)

            return fn(*args, **kwargs)

        return super().submit(context_wrapper)


def patch_thread_pool_executor() -> None:
    """
    Patch concurrent.futures.ThreadPoolExecutor to automatically propagate context.

    After calling this, all uses of ThreadPoolExecutor will automatically
    propagate Tracium's contextvars to worker threads.
    """
    import concurrent.futures

    if concurrent.futures.ThreadPoolExecutor is ContextPropagatingThreadPoolExecutor:
        return

    concurrent.futures.ThreadPoolExecutor = ContextPropagatingThreadPoolExecutor

    import sys

    if "concurrent.futures" in sys.modules:
        sys.modules["concurrent.futures"].ThreadPoolExecutor = ContextPropagatingThreadPoolExecutor


def unpatch_thread_pool_executor() -> None:
    """
    Restore the original ThreadPoolExecutor.

    This is mainly useful for testing.
    """
    import concurrent.futures

    concurrent.futures.ThreadPoolExecutor = _original_thread_pool_executor

    import sys

    if "concurrent.futures" in sys.modules:
        sys.modules["concurrent.futures"].ThreadPoolExecutor = _original_thread_pool_executor


async def run_in_executor_with_context(executor, func, *args):
    """
    Run a function in an executor while preserving the current context with a thread-safe copy.

    This is a context-aware version of loop.run_in_executor() that ensures
    contextvars are propagated to the executor, with each thread getting its own
    copy of the span_stack to prevent race conditions.

    Args:
        executor: The executor to run the function in
        func: The function to run
        *args: Arguments to pass to the function

    Returns:
        The result of the function

    Example:
        >>> from concurrent.futures import ThreadPoolExecutor
        >>> executor = ThreadPoolExecutor()
        >>> result = await run_in_executor_with_context(executor, my_func, arg1, arg2)
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

        return func(*args)

    loop = asyncio.get_event_loop()

    return await loop.run_in_executor(executor, wrapper)


def context_propagating_wrapper(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator that wraps a function to preserve context when called from threads.

    Use this decorator on functions that will be called from threads to ensure
    they have access to the current Tracium trace.

    Example:
        >>> @context_propagating_wrapper
        ... def my_worker_function(data):
        ...     trace = current_trace()
        ...     with trace.span(...):
        ...         pass
    """
    return _wrap_func_with_context(func)


def enable_automatic_context_propagation() -> None:
    """
    Enable automatic context propagation for all threading and async operations.

    This is called automatically by tracium.trace() and tracium.init() to ensure
    a seamless experience. After calling this, users don't need to worry about
    manually passing trace handles to threads.

    This patches:
    - concurrent.futures.ThreadPoolExecutor
    - threading.Thread (via thread_helpers module)

    So users can use ANY threading approach and context will propagate automatically.
    """
    from ..helpers.thread_helpers import patch_threading_module

    patch_thread_pool_executor()
    patch_threading_module()


def disable_automatic_context_propagation() -> None:
    """
    Disable automatic context propagation.

    This restores the original behavior. Mainly useful for testing.
    """
    from ..helpers.thread_helpers import unpatch_threading_module

    unpatch_thread_pool_executor()
    unpatch_threading_module()
