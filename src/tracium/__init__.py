"""
Public entrypoint for the Tracium SDK auto instrumentation layer.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

from .context.tenant_context import get_current_tenant, set_tenant
from .context.trace_context import current_trace
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
from .helpers.thread_helpers import (
    ContextThread,
    get_current_trace_context,
    run_in_thread,
    with_context,
    with_trace_context,
)
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
    "get_queue_stats",
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
    "run_in_thread",
    "with_context",
    "ContextThread",
    "get_current_trace_context",
    "with_trace_context",
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
    workspace_id: str | None = None,
    default_tags: Sequence[str] | None = None,
    default_metadata: Mapping[str, Any] | None = None,
    auto_instrument_langchain: bool = True,
    auto_instrument_langgraph: bool = True,
    auto_instrument_llm_clients: bool = True,
    capture_media: bool = False,
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
        workspace_id: Workspace ID for all traces and spans (or set TRACIUM_WORKSPACE_ID env var)
        default_tags: Default tags to apply to all traces
        default_metadata: Default metadata to apply to all traces
        auto_instrument_langchain: Enable automatic LangChain instrumentation
        auto_instrument_langgraph: Enable automatic LangGraph instrumentation
        auto_instrument_llm_clients: Enable automatic LLM client instrumentation
        capture_media: Capture audio/image data as base64 in spans (default: False).
            When enabled, OpenAI audio (speech synthesis) and image outputs are stored
            as base64-encoded data for playback/display in the Tracium UI.
        transport: Optional custom HTTP transport

    Returns:
        TraciumClient: The initialized client
    """

    api_key = api_key or os.getenv("TRACIUM_API_KEY")
    if not api_key:
        raise ValueError("Tracium API key is required. Pass api_key or set TRACIUM_API_KEY.")

    # Check for base_url from environment if not provided
    if base_url is None and config is None:
        base_url = os.getenv("TRACIUM_BASE_URL")

    if config is not None and base_url is not None:
        raise ValueError("Provide either config or base_url, not both.")

    default_workspace_id = workspace_id or os.getenv("TRACIUM_WORKSPACE_ID")

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
        default_workspace_id=default_workspace_id,
        default_tags=list(default_tags or []),
        default_metadata=dict(default_metadata or {}),
        auto_instrument_langchain=auto_instrument_langchain,
        auto_instrument_langgraph=auto_instrument_langgraph,
        auto_instrument_llm_clients=auto_instrument_llm_clients,
        capture_media=capture_media,
    )
    set_client(client, options=options)

    from .instrumentation.auto_trace_tracker import register_cleanup

    register_cleanup()

    configure_auto_instrumentation(client)
    return client


def get_client() -> TraciumClient:
    """Return the globally initialized Tracium client."""
    return _get_client()


def get_queue_stats() -> dict[str, Any]:
    """
    Get statistics about the background sender queue from the global client.

    This is a convenience function that calls get_queue_stats() on the
    globally initialized client.

    Returns:
        Dictionary with queue health metrics and event counts

    Example:
        >>> import tracium
        >>> tracium.init(api_key="...")
        >>> stats = tracium.get_queue_stats()
        >>> print(f"Queue is {stats['capacity_percent']:.1f}% full")
        >>> if stats['total_dropped'] > 0:
        ...     print(f"Warning: {stats['total_dropped']} events were dropped!")
    """
    client = _get_client()
    return client.get_queue_stats()


def start_trace(
    *,
    agent_name: str | None = None,
    model_id: str | None = None,
    version: str | None = None,
    workspace_id: str | None = None,
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
        workspace_id=workspace_id or options.default_workspace_id,
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

    _auto_wrap_wsgi_in_caller()
    _auto_wrap_lambda_handler_in_caller()
    _auto_wrap_asgi_in_caller()

    return client


def _is_serverless() -> bool:
    """True when running under AWS Lambda / Google Cloud Functions / Vercel /
    Azure Functions / Cloud Run. Used to enable per-invocation flush and the
    Lambda handler auto-wrap."""
    return any(
        os.environ.get(k)
        for k in (
            "AWS_LAMBDA_FUNCTION_NAME",
            "AWS_EXECUTION_ENV",
            "K_SERVICE",
            "FUNCTION_TARGET",
            "FUNCTION_NAME",
            "VERCEL",
            "FUNCTIONS_WORKER_RUNTIME",
            "TRACIUM_FORCE_SYNC",
        )
    )


_AUTO_WRAP_FRAME_BLOCKLIST = (
    "_pytest",
    "pytest",
    "pluggy",
    "gunicorn",
    "uvicorn",
    "hypercorn",
    "daphne",
    "click",
    "typer",
    "celery",
    "anyio",
    "asyncio",
    "importlib",
    "runpy",
)


def _candidate_caller_frames(max_depth: int = 20) -> list[Any]:
    """Walk up the stack collecting frames likely to belong to user code.

    Frames whose ``__name__`` starts with a known third-party launcher
    (pytest, gunicorn, uvicorn, click, …) or any ``tracium.*`` module are
    skipped — we don't want to wrap an ``app``/``main`` defined inside one
    of those frameworks.
    """
    import inspect

    frame = inspect.currentframe()
    if not frame:
        return []

    out: list[Any] = []
    f = frame.f_back
    depth = 0
    while f is not None and depth < max_depth:
        module_name = (f.f_globals.get("__name__") or "")
        head = module_name.split(".", 1)[0]
        is_tracium = module_name == "tracium" or module_name.startswith("tracium.")
        if head not in _AUTO_WRAP_FRAME_BLOCKLIST and not is_tracium:
            out.append(f)
        f = f.f_back
        depth += 1
    return out


def _auto_wrap_lambda_handler_in_caller() -> None:
    """If running in AWS Lambda (or similar FaaS where the runtime imports the
    user's handler by name), wrap it so that each invocation finalizes outstanding
    traces and forces a flush before the container is frozen."""
    if not _is_serverless():
        return

    import sys

    for f in _candidate_caller_frames():
        caller_globals = f.f_globals
        module_name = caller_globals.get("__name__")
        caller_module = sys.modules.get(module_name) if module_name else None

        # ``main`` is too generic to wrap from an arbitrary frame — restrict
        # it to the program entry module so we don't wrap a helper called
        # ``main`` that happens to live in caller scope.
        names: tuple[str, ...] = ("lambda_handler", "handler")
        if module_name == "__main__":
            names = names + ("main",)

        for name in names:
            if name not in caller_globals:
                continue
            fn = caller_globals[name]
            if not callable(fn) or getattr(fn, "_tracium_wrapped", False):
                continue
            wrapped = _wrap_serverless_handler(fn)
            caller_globals[name] = wrapped
            if caller_module:
                setattr(caller_module, name, wrapped)
            return


def _wrap_serverless_handler(handler: Any) -> Any:
    """Wrap a Lambda/Cloud Function handler so each invocation produces its own
    auto-trace and flushes telemetry inline before returning."""
    import functools
    import inspect

    is_async = inspect.iscoroutinefunction(handler)

    if is_async:

        @functools.wraps(handler)
        async def _async_wrapped(*args: Any, **kwargs: Any) -> Any:
            try:
                return await handler(*args, **kwargs)
            finally:
                _serverless_finalize()

        _async_wrapped._tracium_wrapped = True  # type: ignore[attr-defined]
        return _async_wrapped

    @functools.wraps(handler)
    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            return handler(*args, **kwargs)
        finally:
            _serverless_finalize()

    _wrapped._tracium_wrapped = True  # type: ignore[attr-defined]
    return _wrapped


def _serverless_finalize() -> None:
    """Per-invocation cleanup: close auto-traces, drain the process-wide
    registries, and force a flush.

    Draining the registries (not just the active contextvar) is what makes
    warm-container reuse leak-proof: if a prior invocation crashed without
    unwinding cleanly, its entries stay in ``_AUTO_TRACE_REGISTRY`` /
    ``_WEB_AUTO_TRACE_REGISTRY`` until somebody sweeps them. The next
    invocation runs ``_serverless_finalize`` on its way out, which clears
    both, so leaks can survive at most one invocation."""
    try:
        from .instrumentation.auto_trace_tracker import cleanup_auto_trace

        cleanup_auto_trace()
    except Exception:
        pass
    try:
        client = _get_client()
        # Cap the flush so a hung backend can't delay the Lambda response.
        # The cold/warm-path difference here matters: in serverless we'd
        # rather skip emitting than block the user's request.
        client.flush(timeout=5.0)
    except Exception:
        pass


def _auto_wrap_asgi_in_caller() -> None:
    """If a FastAPI / Starlette / Litestar app is in caller globals, wrap it
    with the Tracium ASGI middleware so request-scoped traces capture nested
    LLM spans."""
    import sys

    try:
        from .instrumentation.web_frameworks.asgi import TraciumASGIMiddleware
    except Exception:
        return

    for f in _candidate_caller_frames():
        caller_globals = f.f_globals
        module_name = caller_globals.get("__name__")
        caller_module = sys.modules.get(module_name) if module_name else None

        for name in ("application", "app", "asgi_app"):
            if name not in caller_globals:
                continue
            app = caller_globals[name]
            if not callable(app) or getattr(app, "_tracium_wrapped", False):
                continue
            if not _looks_like_asgi(app):
                continue
            wrapped = TraciumASGIMiddleware(app)
            wrapped._tracium_wrapped = True  # type: ignore[attr-defined]
            caller_globals[name] = wrapped
            if caller_module:
                setattr(caller_module, name, wrapped)
            return


def _looks_like_asgi(app: Any) -> bool:
    """Heuristic to detect ASGI apps so we don't wrap them as WSGI."""
    import inspect

    cls = type(app)
    module = (getattr(cls, "__module__", "") or "").lower()
    asgi_modules = (
        "starlette",
        "fastapi",
        "litestar",
        "sanic",
        "blacksheep",
        "quart",
        "uvicorn",
        "hypercorn",
        "channels",
    )
    if any(m in module for m in asgi_modules):
        return True

    call = getattr(app, "__call__", None)
    if inspect.iscoroutinefunction(call) or inspect.iscoroutinefunction(app):
        return True

    try:
        sig = inspect.signature(app)
        param_names = list(sig.parameters)
    except (ValueError, TypeError):
        return False

    if {"scope", "receive", "send"}.issubset(set(param_names)):
        return True
    if len(param_names) == 3 and param_names[:3] == ["scope", "receive", "send"]:
        return True
    return False


def _auto_wrap_wsgi_in_caller() -> None:
    import inspect
    import sys

    # Walk up the stack so calling trace() from a helper (e.g. def setup(): tracium.trace())
    # still finds the user's WSGI app in their module globals. Third-party
    # launcher frames (pytest, gunicorn, …) are skipped by the helper.
    for f in _candidate_caller_frames():
        caller_globals = f.f_globals
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

            if _looks_like_asgi(app):
                continue

            try:
                params = list(inspect.signature(app).parameters)
                if len(params) < 2:
                    continue
            except (ValueError, TypeError):
                # Can't introspect (C extension, builtin) — be conservative and skip.
                continue

            wrapped = wrap_wsgi_app(app)
            caller_globals[name] = wrapped
            if caller_module:
                setattr(caller_module, name, wrapped)
            return
