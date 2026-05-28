"""Auto-propagate Tracium's contextvars across user-spawned threads.

By default Python starts each :class:`threading.Thread` with an empty context,
which means a Tracium span/trace active in the parent is not visible inside the
thread. The same applies to :meth:`concurrent.futures.ThreadPoolExecutor.submit`.

This module installs lightweight monkey-patches that capture the parent
context at construction / submit time and run the target inside it. No new
threads are spawned, and the patches are idempotent.

Disabled when the env var ``TRACIUM_DISABLE_THREAD_PROPAGATION=1`` is set, for
users on debuggers that don't tolerate the patched ``__init__`` signature.
"""

from __future__ import annotations

import contextvars
import logging
import os
import sys
import threading
from typing import Any

from ..helpers.global_state import PATCH_LOCK

logger = logging.getLogger(__name__)

_INSTALLED = False


def install() -> None:
    """Patch threading.Thread + ThreadPoolExecutor.submit to propagate contextvars.

    Idempotent. No background threads are created — these patches only modify
    construction/submit of threads the user is already creating.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    if os.environ.get("TRACIUM_DISABLE_THREAD_PROPAGATION"):
        return

    with PATCH_LOCK:
        if _INSTALLED:
            return
        for step in (
            _check_green_thread_compat,
            _patch_threading_thread,
            _patch_thread_pool_executor,
            install_asyncio_task_factory,
        ):
            try:
                step()
            except Exception as e:
                logger.debug(
                    "tracium thread-propagation step %s failed: %s: %s",
                    step.__name__,
                    type(e).__name__,
                    e,
                )
        _INSTALLED = True


def install_asyncio_task_factory() -> None:
    """Eagerly ensure every asyncio loop copies parent context onto new tasks.

    Python 3.7+ ``asyncio.Task`` already copies context when none is supplied,
    so this is mostly a safety net for codebases or custom event-loop policies
    that have installed a non-copying task factory. We:

    1. Patch ``asyncio.new_event_loop`` so future loops get our factory.
    2. If a loop is already running, install on it too.

    Both steps are idempotent.
    """
    try:
        import asyncio
    except Exception:
        return

    _patch_asyncio_new_event_loop(asyncio)

    try:
        loop = asyncio.get_running_loop()
    except Exception:
        loop = None
    if loop is not None:
        _install_factory_on_loop(loop)


def _install_factory_on_loop(loop: Any) -> None:
    if getattr(loop, "_tracium_task_factory_installed", False):
        return
    import asyncio

    previous = loop.get_task_factory()

    def factory(loop_: Any, coro: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("context", contextvars.copy_context())
        if previous is None:
            return asyncio.Task(coro, loop=loop_, **kwargs)
        return previous(loop_, coro, **kwargs)

    try:
        loop.set_task_factory(factory)
        loop._tracium_task_factory_installed = True
    except Exception:
        pass


def _patch_asyncio_new_event_loop(asyncio_mod: Any) -> None:
    original = asyncio_mod.new_event_loop
    # Distinct sentinel from `_tracium_patched` (used by the exception-handler
    # patch in auto_trace_tracker) so both wrappers can coexist on the same
    # `asyncio.new_event_loop` symbol, in either install order.
    if getattr(original, "_tracium_factory_patched", False):
        return

    def patched_new_event_loop() -> Any:
        loop = original()
        try:
            _install_factory_on_loop(loop)
        except Exception:
            pass
        return loop

    patched_new_event_loop._tracium_factory_patched = True  # type: ignore[attr-defined]
    try:
        import functools

        functools.update_wrapper(patched_new_event_loop, original)
    except Exception:
        pass
    asyncio_mod.new_event_loop = patched_new_event_loop


def _check_green_thread_compat() -> None:
    """Warn if gevent / eventlet monkey-patching has interactions we can't
    detect after the fact.

    gevent and eventlet replace ``threading.Thread`` (and related primitives)
    with green-thread shims. The interaction with Tracium depends on order:

    - ``monkey_patch_all()`` BEFORE ``tracium.init()`` → fine. Our patch wraps
      the already-replaced ``threading.Thread.__init__``.
    - ``monkey_patch_all()`` AFTER ``tracium.init()`` → gevent overwrites our
      patch and we lose thread-context propagation.

    We can only see the state at install time, so we emit a single hint
    pointing users to the correct ordering.
    """
    gevent_loaded = "gevent.monkey" in sys.modules
    eventlet_loaded = "eventlet.patcher" in sys.modules

    if not (gevent_loaded or eventlet_loaded):
        return

    already_patched = False
    if gevent_loaded:
        try:
            from gevent import monkey

            already_patched = bool(monkey.is_module_patched("threading"))
        except Exception:
            pass
    if not already_patched and eventlet_loaded:
        try:
            from eventlet import patcher

            already_patched = bool(patcher.is_monkey_patched("thread"))
        except Exception:
            pass

    runtime = "gevent" if gevent_loaded else "eventlet"
    if already_patched:
        logger.info(
            "tracium: detected %s monkey-patching applied before init() — "
            "Tracium will layer on top, context propagation works.",
            runtime,
        )
    else:
        logger.warning(
            "tracium: %s is imported but threading is not yet monkey-patched. "
            "Call %s monkey_patch_all() BEFORE tracium.init() / tracium.trace(), "
            "otherwise the later monkey-patch will overwrite Tracium's thread "
            "context-propagation patch and traces spawned in green threads will "
            "be lost.",
            runtime,
            runtime,
        )


def _patch_threading_thread() -> None:
    original_init = threading.Thread.__init__

    if getattr(original_init, "_tracium_patched", False):
        return

    def patched_init(self: threading.Thread, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        try:
            ctx = contextvars.copy_context()
        except Exception:
            return

        # Shadow `self.run` on the instance so the wrapper works even when a
        # subclass overrides `run` (class-level patches don't help there —
        # subclass `run` would shadow our base patch). The bound method we
        # capture here resolves to the subclass override if present.
        original_bound_run = self.run

        def _run_in_context() -> None:
            ctx.run(original_bound_run)

        self.run = _run_in_context  # type: ignore[method-assign]

    # Preserve signature for IDEs / debugpy. ``update_wrapper`` copies
    # attributes from the original, so set our marker AFTER it runs to
    # ensure the marker isn't clobbered.
    try:
        import functools
        import inspect

        functools.update_wrapper(patched_init, original_init)
        patched_init.__signature__ = inspect.signature(original_init)  # type: ignore[attr-defined]
    except Exception:
        pass

    patched_init._tracium_patched = True  # type: ignore[attr-defined]
    threading.Thread.__init__ = patched_init  # type: ignore[method-assign]


def _patch_thread_pool_executor() -> None:
    import concurrent.futures

    cls = concurrent.futures.ThreadPoolExecutor
    original_submit = cls.submit

    if getattr(original_submit, "_tracium_patched", False):
        return

    def patched_submit(self: Any, fn: Any, /, *args: Any, **kwargs: Any) -> Any:
        try:
            ctx = contextvars.copy_context()
        except Exception:
            return original_submit(self, fn, *args, **kwargs)

        def _runner(*a: Any, **kw: Any) -> Any:
            return ctx.run(fn, *a, **kw)

        return original_submit(self, _runner, *args, **kwargs)

    try:
        import functools

        functools.update_wrapper(patched_submit, original_submit)  # type: ignore[arg-type]
    except Exception:
        pass

    patched_submit._tracium_patched = True  # type: ignore[attr-defined]
    cls.submit = patched_submit  # type: ignore[method-assign]
