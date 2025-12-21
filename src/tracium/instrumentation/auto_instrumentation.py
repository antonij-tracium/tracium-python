"""
High level auto-instrumentation dispatcher for Tracium.
"""

from __future__ import annotations

from ..core import TraciumClient
from ..helpers.global_state import get_options
from ..integrations.anthropic import patch_anthropic
from ..integrations.google import patch_google_genai
from ..integrations.langchain import register_langchain_handler
from ..integrations.openai import patch_openai


def configure_auto_instrumentation(client: TraciumClient) -> None:
    """
    Configure optional integrations based on `tracium.init` options.

    This automatically patches supported libraries when enabled:
    - OpenAI (GPT-4, GPT-3.5, etc.)
    - Anthropic (Claude)
    - Google Generative AI (Gemini)
    - LangChain
    """

    options = get_options()

    if options.auto_instrument_langchain:
        register_langchain_handler(client)
    if options.auto_instrument_llm_clients:
        patch_openai(client)
        patch_anthropic(client)
        patch_google_genai(client)
