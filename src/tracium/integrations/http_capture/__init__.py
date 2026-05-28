"""
Universal HTTP-level LLM capture.

This package is the fallback layer that lets ``tracium.trace()`` produce
high-fidelity LLM spans for *any* code that hits a known LLM HTTP endpoint —
even if we don't have a native client integration for the SDK in question.

The native integrations (OpenAI, Anthropic, Google, …) still take priority and
mark the call as :func:`dedup.owned_capture` for the duration of the request,
so we never double-count.

Public API:

- :func:`install` — patch ``httpx`` + ``requests``. Idempotent.
- :func:`is_installed` — quick check.
- :data:`LLM_CAPTURE_OWNED` — re-exported from :mod:`.dedup`.
- :func:`owned_capture` — re-exported from :mod:`.dedup`.
"""

from __future__ import annotations

import logging

from . import adapter_requests, transport_httpx
from .dedup import LLM_CAPTURE_OWNED, is_owned, owned_capture
from .providers import LLMCall, detect_provider, parse

logger = logging.getLogger(__name__)

_INSTALLED = False

__all__ = [
    "install",
    "uninstall",
    "is_installed",
    "owned_capture",
    "is_owned",
    "LLM_CAPTURE_OWNED",
    "LLMCall",
    "detect_provider",
    "parse",
]


def install() -> None:
    """Install httpx + requests hooks. Safe to call multiple times."""
    global _INSTALLED
    if _INSTALLED:
        return
    try:
        transport_httpx.install()
    except Exception as e:
        logger.debug("tracium: httpx hook failed (continuing): %s: %s", type(e).__name__, e)
    try:
        adapter_requests.install()
    except Exception as e:
        logger.debug("tracium: requests hook failed (continuing): %s: %s", type(e).__name__, e)
    _INSTALLED = True


def uninstall() -> None:
    """Reverse :func:`install` — restore httpx and requests to their original
    state. Useful for test isolation."""
    global _INSTALLED
    try:
        transport_httpx.uninstall()
    except Exception as e:
        logger.debug("tracium: httpx unhook failed (continuing): %s: %s", type(e).__name__, e)
    try:
        adapter_requests.uninstall()
    except Exception as e:
        logger.debug("tracium: requests unhook failed (continuing): %s: %s", type(e).__name__, e)
    _INSTALLED = False


def is_installed() -> bool:
    return _INSTALLED
