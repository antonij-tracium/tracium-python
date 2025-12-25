"""Context manager wrappers for Anthropic stream() method."""

from __future__ import annotations

from typing import Any

from ...core.client import TraciumClient
from ...helpers.call_hierarchy import get_or_create_function_span
from ...helpers.global_state import get_default_tags, get_options
from ...instrumentation.auto_trace_tracker import (
    _get_web_route_info,
    close_auto_trace_if_needed,
    get_current_function_for_span,
    get_or_create_auto_trace,
)
from .stream_wrappers import (
    AsyncStreamWrapper,
    StreamWrapper,
)
from .utils import extract_model, normalize_messages


def _create_span_context(
    client: TraciumClient, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[Any, Any, Any]:
    """Create trace and span context for streaming."""
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


class StreamContextManagerWrapper:
    """Wrapper for the context manager that wraps the stream."""

    def __init__(self, original_cm: Any, span_handle: Any, span_context: Any):
        self._original_cm = original_cm
        self._span_handle = span_handle
        self._span_context = span_context
        self._wrapped_stream: StreamWrapper | None = None

    def __enter__(self):
        original_stream = self._original_cm.__enter__()
        self._wrapped_stream = StreamWrapper(original_stream, self._span_handle, self._span_context)
        return self._wrapped_stream

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._wrapped_stream:
            aggregated_text = (
                "".join(self._wrapped_stream._text_parts)
                if self._wrapped_stream._text_parts
                else "(streaming response)"
            )
            self._span_handle.record_output(aggregated_text)

            if (
                self._wrapped_stream._input_tokens is not None
                or self._wrapped_stream._output_tokens is not None
            ):
                self._span_handle.set_token_usage(
                    input_tokens=self._wrapped_stream._input_tokens,
                    output_tokens=self._wrapped_stream._output_tokens,
                )

        result = self._original_cm.__exit__(exc_type, exc_val, exc_tb)
        self._span_context.__exit__(exc_type, exc_val, exc_tb)
        close_auto_trace_if_needed(force_close=_get_web_route_info() is not None)
        return result


class AsyncStreamContextManagerWrapper:
    """Wrapper for the async context manager that wraps the stream."""

    def __init__(self, original_cm: Any, span_handle: Any, span_context: Any):
        self._original_cm = original_cm
        self._span_handle = span_handle
        self._span_context = span_context
        self._wrapped_stream: AsyncStreamWrapper | None = None

    async def __aenter__(self):
        original_stream = await self._original_cm.__aenter__()
        self._wrapped_stream = AsyncStreamWrapper(
            original_stream, self._span_handle, self._span_context
        )
        return self._wrapped_stream

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._wrapped_stream:
            aggregated_text = (
                "".join(self._wrapped_stream._text_parts)
                if self._wrapped_stream._text_parts
                else "(streaming response)"
            )
            self._span_handle.record_output(aggregated_text)

            if (
                self._wrapped_stream._input_tokens is not None
                or self._wrapped_stream._output_tokens is not None
            ):
                self._span_handle.set_token_usage(
                    input_tokens=self._wrapped_stream._input_tokens,
                    output_tokens=self._wrapped_stream._output_tokens,
                )

        result = await self._original_cm.__aexit__(exc_type, exc_val, exc_tb)
        self._span_context.__exit__(exc_type, exc_val, exc_tb)
        close_auto_trace_if_needed(force_close=_get_web_route_info() is not None)
        return result


def wrap_stream_context_manager(
    client: TraciumClient,
    stream_context_manager: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Wrap a stream context manager to trace the stream."""
    _, span_handle, span_context = _create_span_context(client, args, kwargs)
    return StreamContextManagerWrapper(stream_context_manager, span_handle, span_context)


async def wrap_stream_context_manager_async(
    client: TraciumClient,
    stream_context_manager: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Wrap an async stream context manager to trace the stream."""
    _, span_handle, span_context = _create_span_context(client, args, kwargs)
    return AsyncStreamContextManagerWrapper(stream_context_manager, span_handle, span_context)
