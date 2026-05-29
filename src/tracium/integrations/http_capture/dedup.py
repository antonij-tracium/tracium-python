"""
Dedup primitive shared between native LLM integrations and the HTTP capture layer.

When a native integration (OpenAI, Anthropic, Google, …) takes ownership of a call,
it sets ``LLM_CAPTURE_OWNED`` for the duration of the underlying HTTP request. The
HTTP capture transport checks the flag before creating its own span; if owned, it
short-circuits and lets the native integration's span be the single source of truth.

This is purely ContextVar-based — no thread-local, no monkey-patching of state. The
flag flows naturally across ``await`` and any context Python copies.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterator

LLM_CAPTURE_OWNED: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "tracium_llm_capture_owned",
    default=False,
)


@contextlib.contextmanager
def owned_capture() -> Iterator[None]:
    """Mark the current context as already capturing an LLM call.

    Used by native integrations around the underlying HTTP call so the HTTP
    transport doesn't double-capture::

        with owned_capture():
            response = self._raw_openai_call(...)
    """
    token = LLM_CAPTURE_OWNED.set(True)
    try:
        yield
    finally:
        LLM_CAPTURE_OWNED.reset(token)


def is_owned() -> bool:
    """True if a native integration is already capturing the current call."""
    return LLM_CAPTURE_OWNED.get()
