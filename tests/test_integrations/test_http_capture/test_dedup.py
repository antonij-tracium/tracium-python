"""
Tests for the dedup ContextVar that prevents native + HTTP layers from
double-capturing the same LLM call.
"""

from __future__ import annotations

import pytest

from tracium.integrations.http_capture.dedup import (
    LLM_CAPTURE_OWNED,
    is_owned,
    owned_capture,
)


@pytest.fixture(autouse=True)
def _reset_owned():
    LLM_CAPTURE_OWNED.set(False)
    yield
    LLM_CAPTURE_OWNED.set(False)


def test_default_state_not_owned() -> None:
    assert is_owned() is False


def test_owned_capture_sets_and_resets() -> None:
    assert is_owned() is False
    with owned_capture():
        assert is_owned() is True
    assert is_owned() is False


def test_owned_capture_nested() -> None:
    with owned_capture():
        with owned_capture():
            assert is_owned() is True
        # Inner reset shouldn't unset because outer is still active.
        assert is_owned() is True
    assert is_owned() is False


def test_exception_inside_resets() -> None:
    with pytest.raises(RuntimeError):
        with owned_capture():
            assert is_owned() is True
            raise RuntimeError("boom")
    assert is_owned() is False


@pytest.mark.asyncio
async def test_async_isolation() -> None:
    """Two parallel asyncio tasks must each have their own ownership view."""
    import asyncio

    seen: dict[str, bool | None] = {}

    async def worker(name: str, set_owned: bool) -> None:
        if set_owned:
            with owned_capture():
                await asyncio.sleep(0)
                seen[name] = is_owned()
        else:
            await asyncio.sleep(0)
            seen[name] = is_owned()

    await asyncio.gather(
        worker("a", set_owned=True),
        worker("b", set_owned=False),
    )

    assert seen["a"] is True
    assert seen["b"] is False
