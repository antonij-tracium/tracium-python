"""
Auto-instrumentation for the OpenAI Python SDK.
"""

from __future__ import annotations

import json
from typing import Any

from ..core.client import TraciumClient
from ..helpers.global_state import STATE, get_default_tags, get_options

openai = None


def _normalize_prompt(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str | dict[str, Any] | None:
    if "messages" in kwargs:
        return {"messages": kwargs["messages"]}
    if args:
        first = args[0]
        if isinstance(first, dict) and "messages" in first:
            return {"messages": first["messages"]}
        if isinstance(first, str):
            return first
    if "prompt" in kwargs:
        return kwargs["prompt"]
    return None


def _extract_model(kwargs: dict[str, Any]) -> str | None:
    model = kwargs.get("model")
    if isinstance(model, str):
        return model
    return None


def _json_ready(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): _json_ready(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_json_ready(v) for v in value]
        return repr(value)


def patch_openai(client: TraciumClient) -> None:
    """
    Patch the OpenAI SDK to automatically trace all OpenAI API calls.

    Supports both sync and async OpenAI clients.
    """
    if STATE.openai_patched:
        return

    global openai
    openai_module = openai
    if openai_module is None:
        try:
            import openai as imported_openai
        except Exception:
            return
        openai = imported_openai
        openai_module = imported_openai

    get_options()

    if hasattr(openai_module, "resources") and hasattr(openai_module.resources, "chat"):
        try:
            target = openai_module.resources.chat.completions.Completions.create
            namespace = openai_module.resources.chat.completions.Completions
            original = target

            def traced_create(*args: Any, **kwargs: Any) -> Any:
                return _trace_openai_call(
                    client=client,
                    original_fn=lambda: original(*args, **kwargs),
                    args=args,
                    kwargs=kwargs,
                )

            setattr(namespace, "create", traced_create)
        except Exception:
            pass
    elif hasattr(openai_module, "ChatCompletion"):
        try:
            target = openai_module.ChatCompletion.create
            namespace = openai_module.ChatCompletion
            original = target

            def traced_create(*args: Any, **kwargs: Any) -> Any:
                return _trace_openai_call(
                    client=client,
                    original_fn=lambda: original(*args, **kwargs),
                    args=args,
                    kwargs=kwargs,
                )

            setattr(namespace, "create", traced_create)
        except Exception:
            pass

    if hasattr(openai_module, "resources") and hasattr(openai_module.resources, "chat"):
        try:
            async_namespace = openai_module.resources.chat.completions.AsyncCompletions
            original_async = async_namespace.create

            async def traced_async_create(*args: Any, **kwargs: Any) -> Any:
                return await _trace_openai_call_async(
                    client=client,
                    original_fn=lambda: original_async(*args, **kwargs),
                    args=args,
                    kwargs=kwargs,
                )

            setattr(async_namespace, "create", traced_async_create)
        except Exception:
            pass

    STATE.openai_patched = True


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

    existing_trace = current_trace()
    if existing_trace is not None:
        trace_tags = existing_trace.tags
        if trace_tags and "@langchain" in trace_tags:
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

    basic_span_name = get_current_function_for_span()

    parent_span_id, span_name = get_or_create_function_span(trace_handle, basic_span_name)

    with trace_handle.span(
        span_type="llm",
        name=span_name,
        model_id=model_id,
        parent_span_id=parent_span_id,
    ) as span_handle:
        if prompt_payload is not None:
            if isinstance(prompt_payload, dict) and "messages" in prompt_payload:
                span_handle.record_input(prompt_payload["messages"])
            else:
                span_handle.record_input(prompt_payload)

        try:
            response = original_fn()
        except Exception as e:
            import traceback

            error_info = {
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc(),
            }
            span_handle.record_output(error_info)
            span_handle.mark_failed(str(e))
            raise

        input_tokens, output_tokens, cached_input_tokens = _extract_token_usage(response)

        output_data = _extract_output_data(response)

        if input_tokens is not None or output_tokens is not None or cached_input_tokens is not None:
            span_handle.set_token_usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_input_tokens,
            )

        span_handle.record_output(output_data)

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

    existing_trace = current_trace()
    if existing_trace is not None:
        trace_tags = existing_trace.tags
        if trace_tags and "@langchain" in trace_tags:
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

    basic_span_name = get_current_function_for_span()

    parent_span_id, span_name = get_or_create_function_span(trace_handle, basic_span_name)

    span_context = trace_handle.span(
        span_type="llm",
        name=span_name,
        model_id=model_id,
        parent_span_id=parent_span_id,
    )
    span_handle = span_context.__enter__()

    if prompt_payload is not None:
        if isinstance(prompt_payload, dict) and "messages" in prompt_payload:
            span_handle.record_input(prompt_payload["messages"])
        else:
            span_handle.record_input(prompt_payload)

    try:
        response = await original_fn()
    except Exception as exc:
        span_context.__exit__(type(exc), exc, exc.__traceback__)
        raise

    input_tokens, output_tokens, cached_input_tokens = _extract_token_usage(response)

    output_data = _extract_output_data(response)

    if input_tokens is not None or output_tokens is not None or cached_input_tokens is not None:
        span_handle.set_token_usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
        )

    span_handle.record_output(output_data)
    span_context.__exit__(None, None, None)

    return response


def _extract_token_usage(response: Any) -> tuple[int | None, int | None, int | None]:
    """
    Extract token usage information from OpenAI response.

    Returns:
        Tuple of (input_tokens, output_tokens, cached_input_tokens)
    """
    input_tokens = None
    output_tokens = None
    cached_input_tokens = None

    usage = getattr(response, "usage", None)
    usage_dict: dict[str, Any] | None = None

    if usage:
        if isinstance(usage, dict):
            usage_dict = usage
        elif hasattr(usage, "model_dump"):
            try:
                usage_dict = usage.model_dump()
            except Exception:
                usage_dict = {}
                for attr in [
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "input_tokens",
                    "output_tokens",
                    "cached_input_tokens",
                ]:
                    if hasattr(usage, attr):
                        value = getattr(usage, attr)
                        if value is not None:
                            usage_dict[attr] = value
        else:
            usage_dict = {}
            for attr in [
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "input_tokens",
                "output_tokens",
                "cached_input_tokens",
            ]:
                if hasattr(usage, attr):
                    value = getattr(usage, attr)
                    if value is not None:
                        usage_dict[attr] = value

    if usage_dict:
        prompt_tokens = usage_dict.get("prompt_tokens") or usage_dict.get("input_tokens")
        completion_tokens = usage_dict.get("completion_tokens") or usage_dict.get("output_tokens")

        cached_tokens = None
        prompt_token_details = usage_dict.get("prompt_token_details")
        if prompt_token_details and isinstance(prompt_token_details, dict):
            cached_tokens = prompt_token_details.get("cached_tokens")

        if cached_tokens is None:
            completion_token_details = usage_dict.get("completion_token_details")
            if completion_token_details and isinstance(completion_token_details, dict):
                cached_tokens = completion_token_details.get("cached_tokens")

        if cached_tokens is None:
            cached_tokens = usage_dict.get("cached_input_tokens")

        if prompt_tokens is not None:
            try:
                input_tokens = int(prompt_tokens)
            except (ValueError, TypeError):
                pass

        if completion_tokens is not None:
            try:
                output_tokens = int(completion_tokens)
            except (ValueError, TypeError):
                pass

        if cached_tokens is not None:
            try:
                cached_input_tokens = int(cached_tokens)
            except (ValueError, TypeError):
                pass

    return input_tokens, output_tokens, cached_input_tokens


def _extract_output_data(response: Any) -> Any:
    """Extract output data from OpenAI response."""
    output_data = None
    try:
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                output_data = choice.message.content
            elif hasattr(choice, "text"):
                output_data = choice.text
    except Exception:
        pass

    if output_data is None:
        if hasattr(response, "model_dump"):
            try:
                output_data = response.model_dump()
            except Exception:
                output_data = str(response)
        else:
            output_data = str(response)

    return output_data
