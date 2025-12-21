"""
Public entrypoint for the Tracium SDK auto instrumentation layer.
"""

from __future__ import annotations

# ruff: noqa: E402
import os
from collections.abc import Mapping, Sequence
from typing import Any

from .context.context_propagation import patch_thread_pool_executor
from .helpers.thread_helpers import patch_threading_module

patch_thread_pool_executor()
patch_threading_module()

from .context.context_propagation import enable_automatic_context_propagation  # noqa: E402
from .context.tenant_context import get_current_tenant, set_tenant  # noqa: E402
from .context.trace_context import current_trace  # noqa: E402
from .core import TraciumClient, TraciumClientConfig, __version__
from .helpers.global_state import (
    TraciumInitOptions,
    get_default_tags,
    get_options,
    set_client,
)
from .helpers.global_state import (
    get_client as _get_client,
)
from .helpers.logging_config import configure_logging, get_logger, redact_sensitive_data
from .helpers.retry import RetryConfig, retry_with_backoff
from .helpers.validation import (
    validate_agent_name,
    validate_api_key,
    validate_error_message,
    validate_metadata,
    validate_name,
    validate_span_id,
    validate_span_type,
    validate_tags,
    validate_trace_id,
)
from .instrumentation.auto_instrumentation import configure_auto_instrumentation
from .instrumentation.decorators import agent_span, agent_trace, span
from .models.trace_handle import AgentTraceHandle, AgentTraceManager

__all__ = [
    "init",
    "trace",
    "get_client",
    "start_trace",
    "agent_trace",
    "current_trace",
    "set_tenant",
    "get_current_tenant",
    "TraciumClient",
    "AgentTraceHandle",
    "AgentTraceManager",
    "TraciumClientConfig",
    "agent_span",
    "span",
    "__version__",
    "configure_logging",
    "get_logger",
    "redact_sensitive_data",
    "RetryConfig",
    "retry_with_backoff",
    "validate_agent_name",
    "validate_api_key",
    "validate_error_message",
    "validate_metadata",
    "validate_name",
    "validate_trace_id",
    "validate_span_id",
    "validate_span_type",
    "validate_tags",
]


def init(
    api_key: str | None = None,
    *,
    base_url: str | None = None,
    config: TraciumClientConfig | None = None,
    default_agent_name: str = "app",
    default_model_id: str | None = None,
    default_version: str | None = None,
    default_tags: Sequence[str] | None = None,
    default_metadata: Mapping[str, Any] | None = None,
    auto_instrument_langchain: bool = True,
    auto_instrument_langgraph: bool = True,
    auto_instrument_llm_clients: bool = True,
    transport: Any | None = None,
) -> TraciumClient:
    """
    Initialize the Tracium SDK.

    Args:
        api_key: Tracium API key (or set TRACIUM_API_KEY env var)
        base_url: Tracium API base URL (or set TRACIUM_BASE_URL env var)
        config: Optional TraciumClientConfig for advanced configuration
        default_agent_name: Default agent name for automatic traces (default: "app")
        default_model_id: Default model ID for traces
        default_version: Optional version string for your application. If provided,
            all automatic traces will use this version. If not provided, version will
            be None (not the SDK version). You should provide your application's
            version, not the SDK version.
        default_tags: Default tags to apply to all traces
        default_metadata: Default metadata to apply to all traces
        auto_instrument_langchain: Enable automatic LangChain instrumentation
        auto_instrument_langgraph: Enable automatic LangGraph instrumentation
        auto_instrument_llm_clients: Enable automatic LLM client instrumentation
        transport: Optional custom HTTP transport

    Returns:
        TraciumClient: The initialized client
    """

    api_key = api_key or os.getenv("TRACIUM_API_KEY")
    if not api_key:
        raise ValueError("Tracium API key is required. Pass api_key or set TRACIUM_API_KEY.")

    if config is not None and base_url is not None:
        raise ValueError("Provide either config or base_url, not both.")

    client_config = config or TraciumClientConfig()
    if base_url is not None:
        client_config = TraciumClientConfig(
            base_url=base_url,
            timeout=client_config.timeout,
            user_agent=client_config.user_agent,
        )

    client = TraciumClient(api_key=api_key, config=client_config, transport=transport)
    options = TraciumInitOptions(
        default_agent_name=default_agent_name,
        default_model_id=default_model_id,
        default_version=default_version,
        default_tags=list(default_tags or []),
        default_metadata=dict(default_metadata or {}),
        auto_instrument_langchain=auto_instrument_langchain,
        auto_instrument_langgraph=auto_instrument_langgraph,
        auto_instrument_llm_clients=auto_instrument_llm_clients,
    )
    set_client(client, options=options)
    enable_automatic_context_propagation()
    configure_auto_instrumentation(client)
    return client


def get_client() -> TraciumClient:
    """Return the globally initialized Tracium client."""
    return _get_client()


def start_trace(
    *,
    agent_name: str | None = None,
    model_id: str | None = None,
    version: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    tags: Sequence[str] | None = None,
    trace_id: str | None = None,
) -> AgentTraceManager:
    """Start a trace using global defaults configured via :func:`init`."""

    client = _get_client()
    options = get_options()

    merged_metadata = {**options.default_metadata, **(metadata or {})}
    merged_tags = get_default_tags(tags)

    return client.agent_trace(
        agent_name=agent_name or options.default_agent_name,
        model_id=model_id or options.default_model_id,
        version=version or options.default_version,
        metadata=merged_metadata,
        tags=merged_tags,
        trace_id=trace_id,
    )





def trace(api_key: str | None = None, **kwargs: Any) -> TraciumClient:
    """
    ONE-LINE SETUP: Initialize Tracium and enable automatic tracing for all supported libraries.

    This is the simplest way to get started with Tracium. Just call this once at the
    start of your application and all LLM calls will be automatically tracked.

    Supported libraries (automatically instrumented):
    - OpenAI (GPT-4, GPT-3.5, etc.)
    - Anthropic (Claude)
    - Google Generative AI (Gemini)
    - LangChain
    - LangGraph

    Args:
        api_key: Tracium API key (or set TRACIUM_API_KEY env var)
        default_version: Optional version string for your application. If provided,
            all automatic traces will use this version. If not provided, version will
            be None (not the SDK version).
        **kwargs: Additional options passed to init() (e.g., default_agent_name,
            default_tags, default_metadata)

    Returns:
        TraciumClient: The initialized client

    Example:
        >>> import tracium
        >>> tracium.trace(api_key="sk_live_...")
        >>>
        >>> tracium.trace(api_key="sk_live_...", default_version="1.2.3")
        >>>
        >>> import openai
        >>> response = openai.ChatCompletion.create(
        ...     model="gpt-4",
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )
        >>>
        >>> import anthropic
        >>> client = anthropic.Anthropic()
        >>> response = client.messages.create(
        ...     model="claude-3-opus-20240229",
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )
        >>>
        >>> import google.generativeai as genai
        >>> model = genai.GenerativeModel('gemini-pro')
        >>> response = model.generate_content("Hello!")

    For more control, use :func:`init` instead and manually configure options.
    """
    kwargs.setdefault("auto_instrument_langchain", True)
    kwargs.setdefault("auto_instrument_langgraph", True)
    kwargs.setdefault("auto_instrument_llm_clients", True)
    return init(api_key=api_key, **kwargs)
