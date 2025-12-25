"""
Auto-instrumentation for the OpenAI Python SDK.
"""

from __future__ import annotations

from typing import Any

from ..core.client import TraciumClient
from ..helpers.global_state import STATE, get_default_tags, get_options

openai = None


def _extract_chunk_content(chunk: Any) -> str | None:
    """Extract text content from a streaming chunk."""
    if not (hasattr(chunk, "choices") and chunk.choices):
        return None
    choice = chunk.choices[0]
    if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
        return choice.delta.content
    if hasattr(choice, "message") and hasattr(choice.message, "content"):
        return choice.message.content
    return None


def _finalize_stream(
    span_handle: Any,
    span_context: Any,
    text_parts: list[str],
    tokens: tuple[int | None, int | None, int | None],
) -> None:
    """Finalize stream processing: record output, tokens, and close span."""
    span_handle.record_output("".join(text_parts) or "(streaming response)")
    input_tokens, output_tokens, cached_input_tokens = tokens
    if any(t is not None for t in tokens):
        span_handle.set_token_usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
        )
    span_context.__exit__(None, None, None)
    from ..instrumentation.auto_trace_tracker import _get_web_route_info, close_auto_trace_if_needed

    close_auto_trace_if_needed(force_close=_get_web_route_info() is not None)


class StreamWrapper:
    """Wrapper for OpenAI streaming responses that intercepts chunks."""

    def __init__(self, original_stream: Any, span_handle: Any, span_context: Any):
        self._stream = original_stream
        self._span_handle = span_handle
        self._span_context = span_context
        self._text_parts: list[str] = []
        self._tokens: tuple[int | None, int | None, int | None] = (None, None, None)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            chunk = next(self._stream)
            if content := _extract_chunk_content(chunk):
                self._text_parts.append(content)
            if hasattr(chunk, "usage") and chunk.usage:
                self._tokens = _extract_token_usage(chunk)
            return chunk
        except StopIteration:
            _finalize_stream(self._span_handle, self._span_context, self._text_parts, self._tokens)
            raise


class AsyncStreamWrapper:
    """Wrapper for OpenAI async streaming responses that intercepts chunks."""

    def __init__(self, original_stream: Any, span_handle: Any, span_context: Any):
        self._stream = original_stream
        self._span_handle = span_handle
        self._span_context = span_context
        self._text_parts: list[str] = []
        self._tokens: tuple[int | None, int | None, int | None] = (None, None, None)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            chunk = await self._stream.__anext__()
            if content := _extract_chunk_content(chunk):
                self._text_parts.append(content)
            if hasattr(chunk, "usage") and chunk.usage:
                self._tokens = _extract_token_usage(chunk)
            return chunk
        except StopAsyncIteration:
            _finalize_stream(self._span_handle, self._span_context, self._text_parts, self._tokens)
            raise


def _normalize_prompt(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str | dict[str, Any] | None:
    if "messages" in kwargs:
        return {"messages": kwargs["messages"]}
    if args and isinstance(args[0], dict) and "messages" in args[0]:
        return {"messages": args[0]["messages"]}
    if args and isinstance(args[0], str):
        return args[0]
    return kwargs.get("prompt")


def _extract_model(kwargs: dict[str, Any]) -> str | None:
    return kwargs.get("model") if isinstance(kwargs.get("model"), str) else None


def patch_openai(client: TraciumClient) -> None:
    """Patch the OpenAI SDK to automatically trace all OpenAI API calls."""
    if STATE.openai_patched:
        return

    global openai
    try:
        if openai is None:
            import openai as imported_openai

            openai = imported_openai
        openai_module = openai
    except Exception:
        return

    get_options()

    def _patch_create(namespace: Any, is_async: bool) -> None:
        try:
            original = namespace.create
            if is_async:

                async def traced(*args: Any, **kwargs: Any) -> Any:
                    return await _trace_openai_call_async(
                        client, lambda: original(*args, **kwargs), args, kwargs
                    )
            else:

                def traced(*args: Any, **kwargs: Any) -> Any:
                    return _trace_openai_call(
                        client, lambda: original(*args, **kwargs), args, kwargs
                    )

            setattr(namespace, "create", traced)
        except Exception:
            pass

    if hasattr(openai_module, "resources") and hasattr(openai_module.resources, "chat"):
        _patch_create(openai_module.resources.chat.completions.Completions, False)
        _patch_create(openai_module.resources.chat.completions.AsyncCompletions, True)
    elif hasattr(openai_module, "ChatCompletion"):
        _patch_create(openai_module.ChatCompletion, False)

    STATE.openai_patched = True


def _handle_error(e: Exception, span_handle: Any, span_context: Any) -> None:
    """Handle error in trace: record error, mark failed, and close span."""
    import traceback

    span_handle.record_output(
        {"error": str(e), "error_type": type(e).__name__, "traceback": traceback.format_exc()}
    )
    span_handle.mark_failed(str(e))
    from ..instrumentation.auto_trace_tracker import (
        _get_web_route_info,
        close_auto_trace_if_needed,
        get_current_auto_trace_context,
    )

    if ctx := get_current_auto_trace_context():
        ctx.mark_span_failed()
    close_auto_trace_if_needed(force_close=_get_web_route_info() is not None, error=e)
    span_context.__exit__(type(e), e, e.__traceback__)


def _is_streaming(response: Any, kwargs: dict[str, Any], is_async: bool = False) -> bool:
    """Check if response is a streaming response."""
    if kwargs.get("stream", False):
        return True
    iter_attr = "__aiter__" if is_async else "__iter__"
    return (
        hasattr(response, iter_attr)
        and not hasattr(response, "usage")
        and not isinstance(response, (str | bytes | dict | list))
    )


def _trace_openai_call(
    client: TraciumClient,
    original_fn: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Trace a synchronous OpenAI API call."""
    from ..context.trace_context import current_trace
    from ..helpers.call_hierarchy import get_or_create_function_span
    from ..instrumentation.auto_trace_tracker import (
        get_current_function_for_span,
        get_or_create_auto_trace,
    )

    if (trace := current_trace()) and trace.tags and "@langchain" in trace.tags:
        return original_fn()

    options = get_options()
    prompt_payload = _normalize_prompt(args, kwargs)
    model_id = _extract_model(kwargs) or options.default_model_id

    trace_handle, _ = get_or_create_auto_trace(
        client=client,
        agent_name=options.default_agent_name or "app",
        model_id=model_id,
        tags=get_default_tags(["@openai"]),
    )

    parent_span_id, span_name = get_or_create_function_span(
        trace_handle, get_current_function_for_span()
    )
    span_context = trace_handle.span(
        span_type="llm", name=span_name, model_id=model_id, parent_span_id=parent_span_id
    )
    span_handle = span_context.__enter__()

    if prompt_payload:
        span_handle.record_input(
            prompt_payload["messages"]
            if isinstance(prompt_payload, dict) and "messages" in prompt_payload
            else prompt_payload
        )

    try:
        response = original_fn()
    except Exception as e:
        _handle_error(e, span_handle, span_context)
        raise

    if _is_streaming(response, kwargs):
        return StreamWrapper(response, span_handle, span_context)

    tokens = _extract_token_usage(response)
    if any(tokens):
        span_handle.set_token_usage(
            input_tokens=tokens[0], output_tokens=tokens[1], cached_input_tokens=tokens[2]
        )
    span_handle.record_output(_extract_output_data(response))
    span_context.__exit__(None, None, None)
    from ..instrumentation.auto_trace_tracker import _get_web_route_info, close_auto_trace_if_needed

    close_auto_trace_if_needed(force_close=_get_web_route_info() is not None)
    return response


async def _trace_openai_call_async(
    client: TraciumClient,
    original_fn: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Trace an asynchronous OpenAI API call."""
    from ..context.trace_context import current_trace
    from ..helpers.call_hierarchy import get_or_create_function_span
    from ..instrumentation.auto_trace_tracker import (
        get_current_function_for_span,
        get_or_create_auto_trace,
    )

    if (trace := current_trace()) and trace.tags and "@langchain" in trace.tags:
        return await original_fn()

    options = get_options()
    prompt_payload = _normalize_prompt(args, kwargs)
    model_id = _extract_model(kwargs) or options.default_model_id

    trace_handle, _ = get_or_create_auto_trace(
        client=client,
        agent_name=options.default_agent_name or "app",
        model_id=model_id,
        tags=get_default_tags(["@openai"]),
    )

    parent_span_id, span_name = get_or_create_function_span(
        trace_handle, get_current_function_for_span()
    )
    span_context = trace_handle.span(
        span_type="llm", name=span_name, model_id=model_id, parent_span_id=parent_span_id
    )
    span_handle = span_context.__enter__()

    if prompt_payload:
        span_handle.record_input(
            prompt_payload["messages"]
            if isinstance(prompt_payload, dict) and "messages" in prompt_payload
            else prompt_payload
        )

    try:
        response = await original_fn()
    except Exception as e:
        _handle_error(e, span_handle, span_context)
        raise

    if _is_streaming(response, kwargs, is_async=True):
        return AsyncStreamWrapper(response, span_handle, span_context)

    tokens = _extract_token_usage(response)
    if any(tokens):
        span_handle.set_token_usage(
            input_tokens=tokens[0], output_tokens=tokens[1], cached_input_tokens=tokens[2]
        )
    span_handle.record_output(_extract_output_data(response))
    span_context.__exit__(None, None, None)
    from ..instrumentation.auto_trace_tracker import _get_web_route_info, close_auto_trace_if_needed

    close_auto_trace_if_needed(force_close=_get_web_route_info() is not None)
    return response


def _extract_token_usage(response: Any) -> tuple[int | None, int | None, int | None]:
    """Extract token usage information from OpenAI response."""
    usage = getattr(response, "usage", None)
    if not usage:
        return None, None, None

    if isinstance(usage, dict):
        usage_dict = usage
    elif hasattr(usage, "model_dump"):
        try:
            usage_dict = usage.model_dump()
        except Exception:
            usage_dict = {
                attr: getattr(usage, attr)
                for attr in [
                    "prompt_tokens",
                    "completion_tokens",
                    "input_tokens",
                    "output_tokens",
                    "cached_input_tokens",
                ]
                if hasattr(usage, attr) and getattr(usage, attr) is not None
            }
    else:
        usage_dict = {
            attr: getattr(usage, attr)
            for attr in [
                "prompt_tokens",
                "completion_tokens",
                "input_tokens",
                "output_tokens",
                "cached_input_tokens",
            ]
            if hasattr(usage, attr) and getattr(usage, attr) is not None
        }

    def safe_int(val: Any) -> int | None:
        try:
            return int(val) if val is not None else None
        except (ValueError, TypeError):
            return None

    prompt_tokens = usage_dict.get("prompt_tokens") or usage_dict.get("input_tokens")
    completion_tokens = usage_dict.get("completion_tokens") or usage_dict.get("output_tokens")

    cached_tokens = None
    if isinstance(usage_dict.get("prompt_token_details"), dict):
        cached_tokens = usage_dict["prompt_token_details"].get("cached_tokens")
    if cached_tokens is None and isinstance(usage_dict.get("completion_token_details"), dict):
        cached_tokens = usage_dict["completion_token_details"].get("cached_tokens")
    if cached_tokens is None:
        cached_tokens = usage_dict.get("cached_input_tokens")

    return safe_int(prompt_tokens), safe_int(completion_tokens), safe_int(cached_tokens)


def _extract_output_data(response: Any) -> Any:
    """Extract output data from OpenAI response."""
    try:
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                return choice.message.content
            if hasattr(choice, "text"):
                return choice.text
    except Exception:
        pass
    try:
        return response.model_dump() if hasattr(response, "model_dump") else str(response)
    except Exception:
        return str(response)
