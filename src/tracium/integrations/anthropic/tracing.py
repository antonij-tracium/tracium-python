"""
Tracing functions for Anthropic API calls.

All tracing operations are designed to be non-blocking and fail-safe.
Tracing errors will never break user applications.
"""

from __future__ import annotations

import traceback
from typing import Any

from ...core.client import TraciumClient
from ...helpers.logging_config import get_logger
from .stream_wrappers import AsyncStreamWrapper, StreamWrapper
from .utils import extract_model, extract_output_data, extract_usage, normalize_messages

logger = get_logger()


def _create_trace_and_span(
    client: TraciumClient, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[Any, Any, Any]:
    """Create trace and span for API call. Returns (trace_handle, span_handle, span_context)."""
    from ...helpers.call_hierarchy import get_or_create_function_span
    from ...helpers.global_state import get_default_tags, get_options
    from ...instrumentation.auto_trace_tracker import (
        get_current_function_for_span,
        get_or_create_auto_trace,
    )

    options = get_options()
    messages_payload = normalize_messages(args, kwargs)
    model_id = extract_model(kwargs) or options.default_model_id

    trace_handle, _ = get_or_create_auto_trace(
        client=client,
        agent_name=options.default_agent_name or "app",
        model_id=model_id,
        tags=get_default_tags(["@anthropic", "@claude"]),
    )

    basic_span_name = get_current_function_for_span()
    parent_span_id, span_name = get_or_create_function_span(trace_handle, basic_span_name)

    span_context = trace_handle.span(
        span_type="llm",
        name=span_name,
        model_id=model_id,
        parent_span_id=parent_span_id,
    )
    span_handle = span_context.__enter__()

    if messages_payload is not None:
        payload = (
            messages_payload["messages"]
            if isinstance(messages_payload, dict) and "messages" in messages_payload
            else messages_payload
        )
        span_handle.record_input(payload)

    return trace_handle, span_handle, span_context


def _close_trace_on_error(error: Exception, span_handle: Any, span_context: Any) -> None:
    """Safely close trace on error."""
    try:
        span_handle.record_output(
            {
                "error": str(error),
                "error_type": type(error).__name__,
                "traceback": traceback.format_exc(),
            }
        )
        span_handle.mark_failed(str(error))
    except Exception:
        pass

    try:
        from ...instrumentation.auto_trace_tracker import get_current_auto_trace_context
        auto_context = get_current_auto_trace_context()
        if auto_context:
            auto_context.mark_span_failed()
    except Exception:
        pass

    try:
        span_context.__exit__(type(error), error, error.__traceback__)
    except Exception:
        pass

    try:
        from ...instrumentation.auto_trace_tracker import (
            _get_web_route_info,
            close_auto_trace_if_needed,
        )
        close_auto_trace_if_needed(force_close=_get_web_route_info() is not None, error=error)
    except Exception:
        pass


def trace_anthropic_call(
    client: TraciumClient,
    original_fn: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Trace a synchronous Anthropic API call."""
    # Check if we should skip tracing
    try:
        from ...helpers.global_state import is_in_langchain_callback
        if is_in_langchain_callback():
            return original_fn()
    except Exception:
        pass

    # Initialize tracing - if this fails, we still run the original function
    trace_handle = None
    span_handle = None
    span_context = None

    try:
        trace_handle, span_handle, span_context = _create_trace_and_span(client, args, kwargs)
    except Exception as e:
        logger.debug(f"Anthropic trace setup failed (continuing without tracing): {e}")

    is_streaming = kwargs.get("stream", False)

    # Execute the original function
    try:
        response = original_fn()
    except Exception as e:
        if span_handle and span_context:
            _close_trace_on_error(e, span_handle, span_context)
        raise

    # Process response - if this fails, still return the response
    try:
        if span_handle and span_context:
            if is_streaming or (hasattr(response, "__iter__") and not hasattr(response, "usage")):
                return StreamWrapper(response, span_handle, span_context)

            input_tokens, output_tokens = extract_usage(getattr(response, "usage", None))
            if input_tokens is not None or output_tokens is not None:
                span_handle.set_token_usage(input_tokens=input_tokens, output_tokens=output_tokens)

            span_handle.record_output(extract_output_data(response))
            span_context.__exit__(None, None, None)

            from ...instrumentation.auto_trace_tracker import (
                _get_web_route_info,
                close_auto_trace_if_needed,
            )
            close_auto_trace_if_needed(force_close=_get_web_route_info() is not None)
    except Exception as e:
        logger.debug(f"Anthropic response tracing failed (ignored): {e}")

    return response


async def trace_anthropic_call_async(
    client: TraciumClient,
    original_fn: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Trace an asynchronous Anthropic API call."""
    # Check if we should skip tracing
    try:
        from ...helpers.global_state import is_in_langchain_callback
        if is_in_langchain_callback():
            return await original_fn()
    except Exception:
        pass

    # Initialize tracing - if this fails, we still run the original function
    trace_handle = None
    span_handle = None
    span_context = None

    try:
        trace_handle, span_handle, span_context = _create_trace_and_span(client, args, kwargs)
    except Exception as e:
        logger.debug(f"Anthropic async trace setup failed (continuing without tracing): {e}")

    is_streaming = kwargs.get("stream", False)

    # Execute the original function
    try:
        response = await original_fn()
    except Exception as exc:
        if span_handle and span_context:
            _close_trace_on_error(exc, span_handle, span_context)
        raise

    # Process response - if this fails, still return the response
    try:
        if span_handle and span_context:
            if is_streaming or (hasattr(response, "__aiter__") and not hasattr(response, "usage")):
                return AsyncStreamWrapper(response, span_handle, span_context)

            input_tokens, output_tokens = extract_usage(getattr(response, "usage", None))
            if input_tokens is not None or output_tokens is not None:
                span_handle.set_token_usage(input_tokens=input_tokens, output_tokens=output_tokens)

            span_handle.record_output(extract_output_data(response))
            span_context.__exit__(None, None, None)

            from ...instrumentation.auto_trace_tracker import (
                _get_web_route_info,
                close_auto_trace_if_needed,
            )
            close_auto_trace_if_needed(force_close=_get_web_route_info() is not None)
    except Exception as e:
        logger.debug(f"Anthropic async response tracing failed (ignored): {e}")

    return response
