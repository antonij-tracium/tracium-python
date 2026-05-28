"""
Tests for ``call_hierarchy`` ContextVar isolation.

Pre-fix, ``_INVOCATION_COUNTER`` stored a mutable ``dict`` in a ContextVar.
``contextvars.copy_context()`` (used by ``asyncio.create_task``) copies the
*reference*, so sibling tasks all mutated the same dict — counters interleaved
across tasks. The fix replaces the dict on every increment, so each task's
context evolves its own copy-on-write view.
"""

from __future__ import annotations

import asyncio

import pytest

from tracium.helpers.call_hierarchy import _INVOCATION_COUNTER, _get_invocation_id


@pytest.fixture(autouse=True)
def _reset_counter():
    _INVOCATION_COUNTER.set({})
    yield
    _INVOCATION_COUNTER.set({})


@pytest.mark.asyncio
async def test_invocation_counter_isolated_across_tasks() -> None:
    barrier = asyncio.Event()

    async def worker() -> list[int]:
        await barrier.wait()
        ids: list[int] = []
        for _ in range(5):
            invocation_id = _get_invocation_id("fn", "/path/file.py", 10)
            ids.append(int(invocation_id.split("_", 1)[0]))
            await asyncio.sleep(0)
        return ids

    task_a = asyncio.create_task(worker())
    task_b = asyncio.create_task(worker())
    await asyncio.sleep(0)
    barrier.set()

    a, b = await asyncio.gather(task_a, task_b)

    assert a == [1, 2, 3, 4, 5]
    assert b == [1, 2, 3, 4, 5]


@pytest.mark.asyncio
async def test_invocation_counter_increments_within_task() -> None:
    """Within a single task the counter still increments monotonically."""
    n1 = _get_invocation_id("f", "/x.py", 1)
    n2 = _get_invocation_id("f", "/x.py", 1)
    n3 = _get_invocation_id("f", "/x.py", 1)
    assert int(n1.split("_", 1)[0]) == 1
    assert int(n2.split("_", 1)[0]) == 2
    assert int(n3.split("_", 1)[0]) == 3


@pytest.mark.asyncio
async def test_invocation_counter_keys_are_independent() -> None:
    """Different (file, function, line) keys count separately."""
    a = _get_invocation_id("f1", "/x.py", 1)
    b = _get_invocation_id("f2", "/x.py", 2)
    c = _get_invocation_id("f1", "/x.py", 1)
    assert int(a.split("_", 1)[0]) == 1
    assert int(b.split("_", 1)[0]) == 1
    assert int(c.split("_", 1)[0]) == 2
