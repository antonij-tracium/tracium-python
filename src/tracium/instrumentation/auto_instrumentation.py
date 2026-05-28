"""
High level auto-instrumentation dispatcher for Tracium.

Three layers of LLM data capture, in priority order:

1. **LangChain callback handler** — invoked when the user is using LangChain.
2. **Native client integrations** — patches for OpenAI, Anthropic, Google. These
   take ownership of calls (via :func:`owned_capture`) so the HTTP fallback
   doesn't double-capture.
3. **HTTP fallback** — wraps ``httpx`` and ``requests``. Catches every call to
   a known LLM endpoint that the native integrations didn't handle. This is the
   "works for any code" guarantee — new providers, custom clients, exotic SDKs.
"""

from __future__ import annotations

import logging

from ..core import TraciumClient
from ..helpers.global_state import get_options
from ..integrations import http_capture
from ..integrations.anthropic import patch_anthropic
from ..integrations.google import patch_google_genai
from ..integrations.langchain import register_langchain_handler
from ..integrations.openai import patch_openai
from ..integrations.openai_realtime import patch_openai_realtime

logger = logging.getLogger(__name__)


def configure_auto_instrumentation(client: TraciumClient) -> None:
    """Configure integrations based on :func:`tracium.init` options.

    Each integration logs an INFO line on activation so operators can verify
    instrumentation is wired up. Failures are logged at DEBUG and never crash
    the host application.
    """
    options = get_options()

    from .auto_trace_tracker import register_cleanup

    register_cleanup()

    activated: list[str] = []

    # Always propagate contextvars across user-spawned threads + executors so
    # nested/parallel spans work regardless of the user's concurrency model.
    # No new threads are created — only construction of the user's threads is
    # modified to copy contextvars.
    try:
        from .thread_propagation import install as _install_thread_propagation

        _install_thread_propagation()
        activated.append("thread-propagation")
    except Exception as e:
        logger.debug(
            "tracium: thread-propagation install failed: %s: %s", type(e).__name__, e
        )

    if options.auto_instrument_langchain:
        if _safe(register_langchain_handler, client, "langchain"):
            activated.append("langchain")

    if options.auto_instrument_llm_clients:
        if _safe(patch_openai, client, "openai"):
            activated.append("openai")
        if _safe(patch_openai_realtime, client, "openai-realtime"):
            activated.append("openai-realtime")
        if _safe(patch_anthropic, client, "anthropic"):
            activated.append("anthropic")
        if _safe(patch_google_genai, client, "google"):
            activated.append("google")

        # The HTTP fallback layer catches any LLM call that the native
        # integrations didn't take ownership of — including providers we
        # don't have a native integration for (Cohere, Bedrock, Groq,
        # OpenAI-compatible self-hosted, …).
        try:
            http_capture.install()
            activated.append("http-capture")
        except Exception as e:
            logger.debug("tracium: http_capture install failed: %s: %s", type(e).__name__, e)

    if activated:
        logger.info("tracium: auto-instrumentation active: %s", ", ".join(activated))
    else:
        logger.info("tracium: auto-instrumentation disabled (no integrations selected)")


def _safe(fn, client: TraciumClient, name: str) -> bool:
    """Run an integration-installer; never let it raise out."""
    try:
        fn(client)
        return True
    except Exception as e:
        logger.debug(
            "tracium: %s integration failed to install (continuing): %s: %s",
            name,
            type(e).__name__,
            e,
        )
        return False
