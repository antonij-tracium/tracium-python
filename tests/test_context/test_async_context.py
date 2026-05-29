"""
Tests for trace context propagation across asyncio boundaries.

ContextVars propagate naturally to coroutines and to ``asyncio.Task`` objects
created via ``asyncio.create_task`` / ``asyncio.gather`` (each task gets its own
copy of the parent's context). These tests pin that behavior down for the
nesting + parallel-sibling scenarios that matter for tracing.

``trace.span(...)`` is a synchronous context manager — it works inside an
``async def`` because the body never suspends across the ``with`` boundary.
"""

from __future__ import annotations

import asyncio

import pytest

from tracium.context.trace_context import (
    CURRENT_SPAN,
    current_depth_level,
    current_parent_span_id,
)
from tracium.core.client import TraciumClient


@pytest.mark.asyncio
async def test_nested_async_spans_inherit_parent(tracium_client: TraciumClient) -> None:
    with tracium_client.agent_trace(agent_name="async-nested") as trace:
        with trace.span(span_type="op", name="parent") as parent:
            assert current_parent_span_id() == parent.id
            assert current_depth_level() == 1

            with trace.span(span_type="op", name="child") as child:
                assert child.parent_span_id == parent.id
                assert child._context.depth_level == 1
                assert current_parent_span_id() == child.id
                assert current_depth_level() == 2

            assert current_parent_span_id() == parent.id
            assert current_depth_level() == 1

        assert current_parent_span_id() is None
        assert current_depth_level() == 0


@pytest.mark.asyncio
async def test_asyncio_gather_siblings_share_parent(tracium_client: TraciumClient) -> None:
    with tracium_client.agent_trace(agent_name="async-gather") as trace:
        with trace.span(span_type="op", name="parent") as parent:

            async def make_child(name: str) -> tuple[str, str | None, int]:
                with trace.span(span_type="op", name=name) as h:
                    await asyncio.sleep(0)
                    return h.id, h.parent_span_id, h._context.depth_level

            results = await asyncio.gather(make_child("a"), make_child("b"), make_child("c"))

            assert all(parent_id == parent.id for _, parent_id, _ in results)
            assert all(depth == 1 for _, _, depth in results)
            assert len({sid for sid, _, _ in results}) == 3

            assert current_parent_span_id() == parent.id


@pytest.mark.asyncio
async def test_token_reset_on_exception(tracium_client: TraciumClient) -> None:
    with tracium_client.agent_trace(agent_name="async-exc") as trace:
        with trace.span(span_type="op", name="outer") as outer:
            with pytest.raises(RuntimeError):
                with trace.span(span_type="op", name="inner"):
                    raise RuntimeError("boom")

            assert current_parent_span_id() == outer.id
            assert current_depth_level() == 1


@pytest.mark.asyncio
async def test_concurrent_async_tasks_isolated(tracium_client: TraciumClient) -> None:
    """Two parallel tasks must build independent grandchild chains."""

    barrier = asyncio.Event()

    async def branch(label: str) -> tuple[str, str | None]:
        with tracium_client.agent_trace(agent_name=f"branch-{label}") as t:
            with t.span(span_type="op", name=f"a-{label}") as a:
                await barrier.wait()
                with t.span(span_type="op", name=f"b-{label}") as b:
                    return a.id, b.parent_span_id

    async def runner() -> tuple[tuple[str, str | None], tuple[str, str | None]]:
        task1 = asyncio.create_task(branch("1"))
        task2 = asyncio.create_task(branch("2"))
        await asyncio.sleep(0)
        barrier.set()
        return await task1, await task2

    (a1_id, b1_parent), (a2_id, b2_parent) = await runner()

    assert b1_parent == a1_id
    assert b2_parent == a2_id
    assert a1_id != a2_id


def test_current_span_default_state() -> None:
    """Outside any span, CURRENT_SPAN holds None."""
    assert CURRENT_SPAN.get() is None
    assert current_parent_span_id() is None
    assert current_depth_level() == 0
