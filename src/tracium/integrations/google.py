"""
Auto-instrumentation for the Google Generative AI Python SDK (Gemini).
"""

from __future__ import annotations

from typing import Any

from ..core.client import TraciumClient
from ..helpers.global_state import STATE, get_default_tags, get_options

genai = None
GenerativeModel = None


def _normalize_prompt(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str | dict[str, Any] | None:
    """Extract prompt/contents from Google Generative AI API call."""
    if "contents" in kwargs:
        return {"contents": kwargs["contents"]}

    if "prompt" in kwargs:
        return kwargs["prompt"]

    if args:
        first = args[0]
        if isinstance(first, str):
            return first
        if isinstance(first, list):
            return {"contents": first}
        if isinstance(first, dict):
            return first
        if GenerativeModel is not None and isinstance(first, GenerativeModel):
            return "google"

    return None


def _extract_model_name(model_obj: Any) -> str | None:
    """Extract model name from Google GenerativeModel instance."""
    if hasattr(model_obj, "model_name"):
        return str(model_obj.model_name)
    if hasattr(model_obj, "_model_name"):
        return str(model_obj._model_name)
    return None


def patch_google_genai(client: TraciumClient) -> None:
    """
    Patch the Google Generative AI SDK to automatically trace all Gemini API calls.

    Supports both sync and async generation methods.
    """
    if STATE.google_patched:
        return

    global genai, GenerativeModel
    genai_module = genai
    generative_model_cls = GenerativeModel

    if genai_module is None or generative_model_cls is None:
        try:
            import google.generativeai as imported_genai  # type: ignore[import]
            from google.generativeai import (
                GenerativeModel as ImportedGenerativeModel,  # type: ignore[import]
            )
        except Exception:
            return
        genai = imported_genai
        GenerativeModel = ImportedGenerativeModel
        genai_module = imported_genai
        generative_model_cls = ImportedGenerativeModel

    options = get_options()

    try:
        original_generate_content = generative_model_cls.generate_content

        def traced_generate_content(self, *args: Any, **kwargs: Any) -> Any:
            model_name = _extract_model_name(self) or options.default_model_id or "gemini-pro"
            return _trace_google_call(
                client=client,
                original_fn=lambda: original_generate_content(self, *args, **kwargs),
                args=args,
                kwargs=kwargs,
                method_name="google.generativeai.generate_content",
                model_name=model_name,
            )

        generative_model_cls.generate_content = traced_generate_content
    except Exception:
        pass

    try:
        original_generate_content_async = generative_model_cls.generate_content_async

        async def traced_generate_content_async(self, *args: Any, **kwargs: Any) -> Any:
            model_name = _extract_model_name(self) or options.default_model_id or "gemini-pro"
            return await _trace_google_call_async(
                client=client,
                original_fn=lambda: original_generate_content_async(self, *args, **kwargs),
                args=args,
                kwargs=kwargs,
                method_name="google.generativeai.generate_content_async",
                model_name=model_name,
            )

        generative_model_cls.generate_content_async = traced_generate_content_async
    except Exception:
        pass

    STATE.google_patched = True


def _trace_google_call(
    client: TraciumClient,
    original_fn: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    method_name: str,
    model_name: str,
) -> Any:
    """Trace a synchronous Google Generative AI API call."""
    from ..helpers.call_hierarchy import get_or_create_function_span
    from ..instrumentation.auto_trace_tracker import (
        get_current_function_for_span,
        get_or_create_auto_trace,
    )

    options = get_options()
    prompt_payload = _normalize_prompt(args, kwargs)

    trace_handle, _ = get_or_create_auto_trace(
        client=client,
        agent_name=options.default_agent_name or "app",
        model_id=model_name,
        tags=get_default_tags(["@google", "@gemini"]),
    )

    basic_span_name = get_current_function_for_span()

    parent_span_id, span_name = get_or_create_function_span(trace_handle, basic_span_name)

    with trace_handle.span(
        span_type="llm",
        name=span_name,
        model_id=model_name,
        parent_span_id=parent_span_id,
    ) as span_handle:
        if prompt_payload is not None:
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

        input_tokens = None
        output_tokens = None
        if hasattr(response, "usage_metadata"):
            usage = response.usage_metadata
            if hasattr(usage, "prompt_token_count"):
                input_tokens = usage.prompt_token_count
            if hasattr(usage, "candidates_token_count"):
                output_tokens = usage.candidates_token_count

        if input_tokens is not None or output_tokens is not None:
            span_handle.set_token_usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        output_data = None
        try:
            if hasattr(response, "text"):
                output_data = response.text
            elif hasattr(response, "to_dict"):
                output_data = response.to_dict()
            else:
                output_data = str(response)
        except Exception:
            output_data = str(response)

        span_handle.record_output(output_data)

        return response


async def _trace_google_call_async(
    client: TraciumClient,
    original_fn: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    method_name: str,
    model_name: str,
) -> Any:
    """Trace an asynchronous Google Generative AI API call."""
    from ..helpers.call_hierarchy import get_or_create_function_span
    from ..instrumentation.auto_trace_tracker import (
        get_current_function_for_span,
        get_or_create_auto_trace,
    )

    options = get_options()
    prompt_payload = _normalize_prompt(args, kwargs)

    trace_handle, _ = get_or_create_auto_trace(
        client=client,
        agent_name=options.default_agent_name or "app",
        model_id=model_name,
        tags=get_default_tags(["@google", "@gemini"]),
    )

    basic_span_name = get_current_function_for_span()

    parent_span_id, span_name = get_or_create_function_span(trace_handle, basic_span_name)

    span_context = trace_handle.span(
        span_type="llm",
        name=span_name,
        model_id=model_name,
        parent_span_id=parent_span_id,
    )
    span_handle = span_context.__enter__()
    if prompt_payload is not None:
        span_handle.record_input(prompt_payload)

    try:
        response = await original_fn()
    except Exception as exc:
        span_context.__exit__(type(exc), exc, exc.__traceback__)
        raise

    input_tokens = None
    output_tokens = None
    if hasattr(response, "usage_metadata"):
        usage = response.usage_metadata
        if hasattr(usage, "prompt_token_count"):
            input_tokens = usage.prompt_token_count
        if hasattr(usage, "candidates_token_count"):
            output_tokens = usage.candidates_token_count

    if input_tokens is not None or output_tokens is not None:
        span_handle.set_token_usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    output_data = None
    try:
        if hasattr(response, "text"):
            output_data = response.text
        elif hasattr(response, "to_dict"):
            output_data = response.to_dict()
        else:
            output_data = str(response)
    except Exception:
        output_data = str(response)

    span_handle.record_output(output_data)
    span_context.__exit__(None, None, None)

    return response
