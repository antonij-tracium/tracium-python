"""
Tests for the explicit-opt-in thread helpers.

Tracium no longer monkey-patches :class:`threading.Thread`. These helpers wrap
:func:`contextvars.copy_context` so users can spawn threads that still see the
parent's active trace and current span.
"""

from __future__ import annotations

import logging
import threading

import pytest

import tracium.context.trace_context as tc
from tracium.context.trace_context import current_parent_span_id, current_trace
from tracium.core.client import TraciumClient
from tracium.helpers.thread_helpers import ContextThread, run_in_thread, with_context


def test_run_in_thread_inherits_context(tracium_client: TraciumClient) -> None:
    captured: dict[str, object] = {}

    def worker() -> None:
        captured["trace"] = current_trace()
        captured["parent"] = current_parent_span_id()

    with tracium_client.agent_trace(agent_name="thread-run") as trace:
        with trace.span(span_type="op", name="parent") as parent:
            t = run_in_thread(worker)
            t.join(timeout=5)

            inherited = captured["trace"]
            assert inherited is not None
            assert inherited.id == trace.id  # type: ignore[union-attr]
            assert captured["parent"] == parent.id


def test_with_context_wraps_at_invocation_site(tracium_client: TraciumClient) -> None:
    """``with_context(func)`` snapshots the context where the wrap happens."""
    seen_parents: list[str | None] = []

    def worker() -> None:
        seen_parents.append(current_parent_span_id())

    with tracium_client.agent_trace(agent_name="thread-wrap") as trace:
        with trace.span(span_type="op", name="span-a") as a:
            t1 = threading.Thread(target=with_context(worker))
            t1.start()
            t1.join(timeout=5)

        with trace.span(span_type="op", name="span-b") as b:
            t2 = threading.Thread(target=with_context(worker))
            t2.start()
            t2.join(timeout=5)

    assert seen_parents == [a.id, b.id]


def test_context_thread_inherits_at_construction(tracium_client: TraciumClient) -> None:
    captured: dict[str, object] = {}

    def worker() -> None:
        captured["parent"] = current_parent_span_id()

    with tracium_client.agent_trace(agent_name="thread-subclass") as trace:
        with trace.span(span_type="op", name="parent") as parent:
            t = ContextThread(target=worker)
            t.start()
            t.join(timeout=5)

    assert captured["parent"] == parent.id


def test_raw_threading_thread_does_not_propagate(
    tracium_client: TraciumClient,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When propagation is disabled, vanilla ``threading.Thread`` sees no trace
    and the one-time INFO is emitted."""
    monkeypatch.setattr(tc, "_RAW_THREAD_WARNING_EMITTED", False)
    from tracium.helpers.global_state import STATE

    monkeypatch.setattr(STATE, "client", tracium_client)

    # The thread_propagation auto-patch may already be installed by other tests
    # in the session (it's process-global and has no uninstall). Restore the
    # unpatched __init__ for the duration of this test so we actually exercise
    # the "raw thread" code path.
    init = threading.Thread.__init__
    original_init = getattr(init, "__wrapped__", init)
    if original_init is not init:
        monkeypatch.setattr(threading.Thread, "__init__", original_init)

    captured: dict[str, object] = {}

    def worker() -> None:
        captured["trace"] = current_trace()

    with tracium_client.agent_trace(agent_name="thread-raw"):
        with caplog.at_level(logging.INFO, logger=tc.__name__):
            t = threading.Thread(target=worker)
            t.start()
            t.join(timeout=5)

    assert captured["trace"] is None
    assert any("non-main thread" in rec.message for rec in caplog.records)
    assert tc._RAW_THREAD_WARNING_EMITTED is True


def test_raw_thread_warning_fires_at_most_once(
    tracium_client: TraciumClient,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tc, "_RAW_THREAD_WARNING_EMITTED", False)
    from tracium.helpers.global_state import STATE

    monkeypatch.setattr(STATE, "client", tracium_client)

    init = threading.Thread.__init__
    original_init = getattr(init, "__wrapped__", init)
    if original_init is not init:
        monkeypatch.setattr(threading.Thread, "__init__", original_init)

    def worker() -> None:
        current_trace()

    with caplog.at_level(logging.INFO, logger=tc.__name__):
        for _ in range(3):
            t = threading.Thread(target=worker)
            t.start()
            t.join(timeout=5)

    info_lines = [rec for rec in caplog.records if "non-main thread" in rec.message]
    assert len(info_lines) == 1


def test_concurrent_threads_isolated(tracium_client: TraciumClient) -> None:
    """Two ContextThreads sharing a parent span must not see each other's children."""
    barrier = threading.Barrier(2)
    seen: dict[str, list[str | None]] = {"a": [], "b": []}

    def branch(label: str, trace_handle) -> None:
        with trace_handle.span(span_type="op", name=f"{label}-1") as outer:
            barrier.wait(timeout=5)
            seen[label].append(current_parent_span_id())
            with trace_handle.span(span_type="op", name=f"{label}-2") as inner:
                seen[label].append(inner.parent_span_id)
            seen[label].append(outer.id)

    with tracium_client.agent_trace(agent_name="thread-isolation") as trace:
        ta = ContextThread(target=branch, args=("a", trace))
        tb = ContextThread(target=branch, args=("b", trace))
        ta.start()
        tb.start()
        ta.join(timeout=5)
        tb.join(timeout=5)

    assert seen["a"][0] == seen["a"][2]  # outer.id is current parent inside outer
    assert seen["a"][1] == seen["a"][2]  # inner's parent is outer
    assert seen["b"][0] == seen["b"][2]
    assert seen["b"][1] == seen["b"][2]
    assert seen["a"][2] != seen["b"][2]
