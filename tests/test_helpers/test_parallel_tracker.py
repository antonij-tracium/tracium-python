"""
Tests for asyncio-aware parallel detection.

Pre-fix, ``parallel_tracker`` keyed parallelism on OS thread id alone, so
``asyncio.gather(c1(), c2(), c3())`` siblings (all on one event-loop thread)
were never marked as parallel. The lane is now ``(thread_id, task_id)``.
"""

from __future__ import annotations

import asyncio

import pytest

from tracium.helpers.parallel_tracker import (
    _registry_lock,
    _span_creation_registry,
    register_span_creation,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    with _registry_lock:
        _span_creation_registry.clear()
    yield
    with _registry_lock:
        _span_creation_registry.clear()


@pytest.mark.asyncio
async def test_asyncio_gather_siblings_assigned_parallel_group() -> None:
    parent_id = "parent-1"
    barrier = asyncio.Event()

    async def make_one(span_id: str) -> tuple[str | None, int | None]:
        await barrier.wait()
        return register_span_creation(span_id, parent_span_id=parent_id)

    task_a = asyncio.create_task(make_one("span-a"))
    task_b = asyncio.create_task(make_one("span-b"))
    task_c = asyncio.create_task(make_one("span-c"))
    await asyncio.sleep(0)
    barrier.set()
    await asyncio.gather(task_a, task_b, task_c)

    with _registry_lock:
        rec_a = _span_creation_registry["span-a"]
        rec_b = _span_creation_registry["span-b"]
        rec_c = _span_creation_registry["span-c"]

    assert rec_a.parallel_group_id is not None
    assert rec_a.parallel_group_id == rec_b.parallel_group_id == rec_c.parallel_group_id

    seqs = sorted([rec_a.sequence_number, rec_b.sequence_number, rec_c.sequence_number])
    assert seqs == [0, 1, 2]

    task_ids = {rec_a.task_id, rec_b.task_id, rec_c.task_id}
    assert len(task_ids) == 3
    assert None not in task_ids


@pytest.mark.asyncio
async def test_sequential_async_calls_not_parallel() -> None:
    """Two sequential awaits in the same task share a lane → not parallel."""
    parent_id = "parent-seq"

    g1, _ = register_span_creation("span-1", parent_span_id=parent_id)
    await asyncio.sleep(0)
    g2, _ = register_span_creation("span-2", parent_span_id=parent_id)

    assert g1 is None
    assert g2 is None


def test_synchronous_same_thread_not_parallel() -> None:
    """Two sync calls on the same thread+task share a lane → not parallel."""
    parent_id = "parent-sync"
    g1, _ = register_span_creation("sync-1", parent_span_id=parent_id)
    g2, _ = register_span_creation("sync-2", parent_span_id=parent_id)
    assert g1 is None
    assert g2 is None


def test_thread_pool_still_parallel() -> None:
    """Threadpool fan-out (different thread_ids) still detected as parallel."""
    import threading

    parent_id = "parent-thread"
    barrier = threading.Barrier(3)
    results: list[tuple[str | None, int | None]] = []
    lock = threading.Lock()

    def worker(span_id: str) -> None:
        barrier.wait(timeout=5)
        out = register_span_creation(span_id, parent_span_id=parent_id)
        with lock:
            results.append(out)

    threads = [threading.Thread(target=worker, args=(f"t-{i}",)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    with _registry_lock:
        records = [_span_creation_registry[f"t-{i}"] for i in range(3)]

    group_ids = {r.parallel_group_id for r in records}
    group_ids.discard(None)
    assert len(group_ids) == 1
