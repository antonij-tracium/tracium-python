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
from .instrumentation.web_frameworks.generic import wrap_wsgi_app
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
    "wrap_wsgi_app",
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
    ONE-LINE SETUP: Initialize Tracium with automatic tracing.

    Call this once at the start of your application. All LLM calls will be tracked.
    For WSGI apps, define your app first, then call trace().

    Supported: OpenAI, Anthropic, Google AI, LangChain, LangGraph, WSGI apps.

    Example:
        >>> def application(environ, start_response):
        ...     pass
        >>>
        >>> tracium.trace()
    """
    kwargs.setdefault("auto_instrument_langchain", True)
    kwargs.setdefault("auto_instrument_langgraph", True)
    kwargs.setdefault("auto_instrument_llm_clients", True)
    client = init(api_key=api_key, **kwargs)

    import inspect
    import sys

    frame = inspect.currentframe()
    if frame and frame.f_back:
        caller_globals = frame.f_back.f_globals
        module_name = caller_globals.get("__name__")
        caller_module = sys.modules.get(module_name) if module_name else None

        for name in ("application", "app", "wsgi_app"):
            if name not in caller_globals:
                continue

            app = caller_globals[name]
            if not callable(app) or getattr(app, "_tracium_wrapped", False):
                continue

            try:
                import flask
                if isinstance(app, flask.Flask):
                    continue
            except ImportError:
                pass

            if "django" in getattr(type(app), "__module__", "").lower():
                continue

            try:
                if len(list(inspect.signature(app).parameters)) < 2:
                    continue
            except (ValueError, TypeError):
                pass

            wrapped = wrap_wsgi_app(app)
            caller_globals[name] = wrapped
            if caller_module:
                setattr(caller_module, name, wrapped)
            break

    return client
