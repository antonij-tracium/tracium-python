"""
Auto-instrumentation for the Anthropic Python SDK (Claude).
"""

from __future__ import annotations

from typing import Any

from ..core.client import TraciumClient
from ..helpers.global_state import STATE, get_default_tags, get_options, is_in_langchain_callback

anthropic = None


def _normalize_messages(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any] | None:
    """Extract messages from Anthropic API call."""
    if "messages" in kwargs:
        return {"messages": kwargs["messages"]}
    if args and isinstance(args[0], dict) and "messages" in args[0]:
        return {"messages": args[0]["messages"]}
    return None


def _extract_model(kwargs: dict[str, Any]) -> str | None:
    """Extract model from Anthropic API call."""
    model = kwargs.get("model")
    if isinstance(model, str):
        return model
    return None


def patch_anthropic(client: TraciumClient) -> None:
    """
    Patch the Anthropic SDK to automatically trace all Claude API calls.

    Supports both sync and async Anthropic clients.
    """
    if STATE.anthropic_patched:
        return

    global anthropic
    anthropic_module = anthropic
    if anthropic_module is None:
        try:
            import anthropic as imported_anthropic
        except Exception:
            return
        anthropic = imported_anthropic
        anthropic_module = imported_anthropic

    get_options()

    if hasattr(anthropic_module, "resources") and hasattr(anthropic_module.resources, "messages"):
        try:
            target_class = anthropic_module.resources.messages.Messages
            original_create = target_class.create

            def traced_create(self, *args: Any, **kwargs: Any) -> Any:
                return _trace_anthropic_call(
                    client=client,
                    original_fn=lambda: original_create(self, *args, **kwargs),
                    args=args,
                    kwargs=kwargs,
                    method_name="anthropic.messages.create",
                )

            target_class.create = traced_create
        except Exception:
            pass

    if hasattr(anthropic_module, "resources") and hasattr(anthropic_module.resources, "messages"):
        try:
            async_target_class = anthropic_module.resources.messages.AsyncMessages
            original_async_create = async_target_class.create

            async def traced_async_create(self, *args: Any, **kwargs: Any) -> Any:
                return await _trace_anthropic_call_async(
                    client=client,
                    original_fn=lambda: original_async_create(self, *args, **kwargs),
                    args=args,
                    kwargs=kwargs,
                    method_name="anthropic.messages.create",
                )

            async_target_class.create = traced_async_create
        except Exception:
            pass

    STATE.anthropic_patched = True


def _trace_anthropic_call(
    client: TraciumClient,
    original_fn: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    method_name: str,
) -> Any:
    if is_in_langchain_callback():
        return original_fn()

    from ..helpers.call_hierarchy import get_or_create_function_span
    from ..instrumentation.auto_trace_tracker import (
        get_current_function_for_span,
        get_or_create_auto_trace,
    )

    options = get_options()
    messages_payload = _normalize_messages(args, kwargs)
    model_id = _extract_model(kwargs) or options.default_model_id

    trace_handle, _ = get_or_create_auto_trace(
        client=client,
        agent_name=options.default_agent_name or "app",
        model_id=model_id,
        tags=get_default_tags(["@anthropic", "@claude"]),
    )

    basic_span_name = get_current_function_for_span()

    parent_span_id, span_name = get_or_create_function_span(trace_handle, basic_span_name)

    with trace_handle.span(
        span_type="llm",
        name=span_name,
        model_id=model_id,
        parent_span_id=parent_span_id,
    ) as span_handle:
        if messages_payload is not None:
            if isinstance(messages_payload, dict) and "messages" in messages_payload:
                span_handle.record_input(messages_payload["messages"])
            else:
                span_handle.record_input(messages_payload)

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

        usage = getattr(response, "usage", None)
        input_tokens = None
        output_tokens = None
        if usage:
            usage_dict = (
                usage
                if isinstance(usage, dict)
                else (
                    getattr(usage, "model_dump", lambda: None)()
                    or getattr(usage, "dict", lambda: None)()
                )
            )
            if usage_dict:
                if "input_tokens" in usage_dict:
                    input_tokens = usage_dict["input_tokens"]
                if "output_tokens" in usage_dict:
                    output_tokens = usage_dict["output_tokens"]
        if input_tokens is not None or output_tokens is not None:
            span_handle.set_token_usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        output_data = None
        try:
            if hasattr(response, "content") and response.content:
                if isinstance(response.content, list):
                    text_parts = []
                    for block in response.content:
                        if hasattr(block, "text"):
                            text_parts.append(block.text)
                        elif isinstance(block, dict) and "text" in block:
                            text_parts.append(block["text"])
                    if text_parts:
                        output_data = (
                            "\n".join(text_parts) if len(text_parts) > 1 else text_parts[0]
                        )
                elif isinstance(response.content, str):
                    output_data = response.content
        except Exception:
            pass

        if output_data is None:
            if hasattr(response, "model_dump"):
                try:
                    output_data = response.model_dump()
                except Exception:
                    output_data = str(response)
            elif hasattr(response, "dict"):
                try:
                    output_data = response.dict()
                except Exception:
                    output_data = str(response)
            else:
                output_data = str(response)

        span_handle.record_output(output_data)

        return response


async def _trace_anthropic_call_async(
    client: TraciumClient,
    original_fn: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    method_name: str,
) -> Any:
    if is_in_langchain_callback():
        return await original_fn()

    from ..helpers.call_hierarchy import get_or_create_function_span
    from ..instrumentation.auto_trace_tracker import (
        get_current_function_for_span,
        get_or_create_auto_trace,
    )

    options = get_options()
    messages_payload = _normalize_messages(args, kwargs)
    model_id = _extract_model(kwargs) or options.default_model_id

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
        if isinstance(messages_payload, dict) and "messages" in messages_payload:
            span_handle.record_input(messages_payload["messages"])
        else:
            span_handle.record_input(messages_payload)

    try:
        response = await original_fn()
    except Exception as exc:
        span_context.__exit__(type(exc), exc, exc.__traceback__)
        raise

    usage = getattr(response, "usage", None)
    input_tokens = None
    output_tokens = None
    if usage:
        usage_dict = (
            usage
            if isinstance(usage, dict)
            else (
                getattr(usage, "model_dump", lambda: None)()
                or getattr(usage, "dict", lambda: None)()
            )
        )
        if usage_dict:
            if "input_tokens" in usage_dict:
                input_tokens = usage_dict["input_tokens"]
            if "output_tokens" in usage_dict:
                output_tokens = usage_dict["output_tokens"]

    if input_tokens is not None or output_tokens is not None:
        span_handle.set_token_usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    output_data = None
    try:
        if hasattr(response, "content") and response.content:
            if isinstance(response.content, list):
                text_parts = []
                for block in response.content:
                    if hasattr(block, "text"):
                        text_parts.append(block.text)
                    elif isinstance(block, dict) and "text" in block:
                        text_parts.append(block["text"])
                if text_parts:
                    output_data = "\n".join(text_parts) if len(text_parts) > 1 else text_parts[0]
            elif isinstance(response.content, str):
                output_data = response.content
    except Exception:
        pass

    if output_data is None:
        if hasattr(response, "model_dump"):
            try:
                output_data = response.model_dump()
            except Exception:
                output_data = str(response)
        elif hasattr(response, "dict"):
            try:
                output_data = response.dict()
            except Exception:
                output_data = str(response)
            else:
                output_data = str(response)

    span_handle.record_output(output_data)
    span_context.__exit__(None, None, None)

    return response
