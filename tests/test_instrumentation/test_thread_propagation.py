"""Tests for the threading.Thread + ThreadPoolExecutor.submit auto-patches that
propagate contextvars without requiring the user to wrap callables."""

from __future__ import annotations

import contextvars
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from tracium.instrumentation.thread_propagation import install


@pytest.fixture(scope="module", autouse=True)
def _install_propagation():
    install()
    # Idempotent — install twice should be a no-op.
    install()


_V: contextvars.ContextVar[str] = contextvars.ContextVar("test_v", default="default")


class TestThreadingThread:
    def test_target_inherits_context(self):
        _V.set("parent")
        result: dict[str, str] = {}

        def worker() -> None:
            result["v"] = _V.get()

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        assert result["v"] == "parent"

    def test_subclass_inherits_context(self):
        _V.set("subclass-parent")
        captured: dict[str, str] = {}

        class MyThread(threading.Thread):
            def run(self) -> None:
                captured["v"] = _V.get()

        t = MyThread()
        t.start()
        t.join()
        assert captured["v"] == "subclass-parent"

    def test_context_set_in_child_does_not_leak_to_parent(self):
        _V.set("parent-value")

        def worker() -> None:
            _V.set("child-value")

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        assert _V.get() == "parent-value"

    def test_signature_preserved(self):
        """Ensure the patched __init__ still accepts the documented kwargs."""
        # Should not raise.
        t = threading.Thread(target=lambda: None, name="x", daemon=True, args=())
        t.start()
        t.join()


class TestThreadPoolExecutor:
    def test_submit_propagates_context(self):
        _V.set("executor-parent")
        with ThreadPoolExecutor(max_workers=2) as ex:
            futures = [ex.submit(_V.get) for _ in range(4)]
            results = [f.result() for f in futures]
        assert all(r == "executor-parent" for r in results)

    def test_submit_args_kwargs_passthrough(self):
        _V.set("v")
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(lambda a, b, c=None: (a, b, c, _V.get()), 1, 2, c=3)
            assert fut.result() == (1, 2, 3, "v")

    def test_each_submission_captures_at_submit_time(self):
        """Different submissions should see the contextvar value as of submit()."""
        results = {}
        with ThreadPoolExecutor(max_workers=1) as ex:
            _V.set("a")
            fut_a = ex.submit(_V.get)
            _V.set("b")
            fut_b = ex.submit(_V.get)
            results["a"] = fut_a.result()
            results["b"] = fut_b.result()
        assert results["a"] == "a"
        assert results["b"] == "b"
