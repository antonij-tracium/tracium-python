"""
Tests for the process-wide auto-trace registry.

When ``tracium.trace()`` is in use and the user spawns concurrent asyncio tasks
that share an entry frame (e.g. ``asyncio.gather(*[same_coro() for _ in range(N)])``),
each task's ``_AUTO_TRACE_CONTEXT`` ContextVar starts as ``None`` (since the parent
never set it). Without the registry, each task creates its own auto-trace, leaving
the workflow fragmented across N disjoint traces.

The registry adoption path collapses those into a single trace as long as the
entry_frame_id matches — which is the case for sibling tasks running the same
coroutine. Web traces are deliberately *not* registered, since concurrent HTTP
requests must stay isolated per request context.
"""

from __future__ import annotations

import asyncio

import pytest

from tracium.core.client import TraciumClient
from tracium.instrumentation import auto_trace_tracker as att


@pytest.fixture(autouse=True)
def _clean_registry():
    with att._AUTO_TRACE_REGISTRY_LOCK:
        att._AUTO_TRACE_REGISTRY.clear()
    att._AUTO_TRACE_CONTEXT.set(None)
    yield
    leftover_ctx = att._AUTO_TRACE_CONTEXT.get()
    if leftover_ctx is not None:
        att._close_trace_safely(leftover_ctx)
        att._AUTO_TRACE_CONTEXT.set(None)
    # Drain registry without holding the lock across _close_trace_safely
    # (which itself acquires the lock — non-reentrant).
    with att._AUTO_TRACE_REGISTRY_LOCK:
        leftovers = list(att._AUTO_TRACE_REGISTRY.values())
        att._AUTO_TRACE_REGISTRY.clear()
    for ctx in leftovers:
        att._close_trace_safely(ctx)


@pytest.mark.asyncio
async def test_async_fanout_shares_single_auto_trace(
    tracium_client: TraciumClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(att, "_find_workflow_entry_point", lambda: ("test:fanout-entry", "fanout"))

    captured: list[tuple[str, bool]] = []

    async def worker() -> None:
        handle, created = att.get_or_create_auto_trace(tracium_client, agent_name="fanout-agent")
        captured.append((handle.id, created))

    await asyncio.gather(worker(), worker(), worker())

    trace_ids = {h for h, _ in captured}
    assert len(trace_ids) == 1
    assert sum(1 for _, created in captured if created) == 1


@pytest.mark.asyncio
async def test_finished_trace_in_registry_is_replaced(
    tracium_client: TraciumClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        att, "_find_workflow_entry_point", lambda: ("test:replace-entry", "replace")
    )

    handle1, _ = att.get_or_create_auto_trace(tracium_client, agent_name="x")

    ctx = att._AUTO_TRACE_CONTEXT.get()
    assert ctx is not None
    att._close_trace_safely(ctx)
    att._AUTO_TRACE_CONTEXT.set(None)

    handle2, created = att.get_or_create_auto_trace(tracium_client, agent_name="x")
    assert created is True
    assert handle2.id != handle1.id


def test_web_traces_not_registered(
    tracium_client: TraciumClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concurrent HTTP requests on the same route must not adopt each other's trace."""
    monkeypatch.setattr(att, "_find_workflow_entry_point", lambda: ("web:/api/foo", "/api/foo"))
    monkeypatch.setattr(att, "_get_web_route_info", lambda: ("/api/foo", "/api/foo"))
    monkeypatch.setattr(att, "register_cleanup", lambda: None)
    from tracium.instrumentation import web_frameworks as wf

    monkeypatch.setattr(wf, "register_response_hooks", lambda: None)

    handle1, _ = att.get_or_create_auto_trace(tracium_client, agent_name="web")

    with att._AUTO_TRACE_REGISTRY_LOCK:
        assert "web:/api/foo" not in att._AUTO_TRACE_REGISTRY


def test_close_unregisters_from_process_registry(
    tracium_client: TraciumClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(att, "_find_workflow_entry_point", lambda: ("test:close-entry", "close"))

    handle, _ = att.get_or_create_auto_trace(tracium_client, agent_name="close-test")
    with att._AUTO_TRACE_REGISTRY_LOCK:
        assert "test:close-entry" in att._AUTO_TRACE_REGISTRY

    ctx = att._AUTO_TRACE_CONTEXT.get()
    att._close_trace_safely(ctx)

    with att._AUTO_TRACE_REGISTRY_LOCK:
        assert "test:close-entry" not in att._AUTO_TRACE_REGISTRY
