"""
Auto-instrumentation for the Google Generative AI Python SDKs (Gemini).
Supports both `google.generativeai` (legacy) and `google.genai` (new).
"""

from __future__ import annotations

from typing import Any

from ..core.client import TraciumClient
from ..helpers.global_state import STATE, get_default_tags, get_options


def _normalize_prompt(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str | dict[str, Any] | None:
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
        if hasattr(first, "model_name") or hasattr(first, "generate_content"):
             return "google"
    return None


def _extract_model_name(model_obj: Any) -> str | None:
    if hasattr(model_obj, "model_name"):
        return str(model_obj.model_name)
    if hasattr(model_obj, "_model_name"):
        return str(model_obj._model_name)
    return None

def patch_google_genai(client: TraciumClient) -> None:
    if STATE.google_patched:
        return

    # 1. Patch Legacy SDK (google.generativeai)
    _patch_legacy_sdk(client)

    # 2. Patch New SDK (google.genai)
    _patch_new_sdk(client)

    STATE.google_patched = True


def _patch_legacy_sdk(client: TraciumClient) -> None:
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            import google.generativeai as genai  # noqa: F401
            from google.generativeai import GenerativeModel  # noqa: F401
    except ImportError:
        return

    options = get_options()

    # Patch generate_content
    if hasattr(GenerativeModel, "generate_content"):
        try:
            original_generate_content = GenerativeModel.generate_content

            def traced_generate_content(self, *args: Any, **kwargs: Any) -> Any:
                model_name = _extract_model_name(self) or options.default_model_id or "gemini-pro"
                return _trace_google_call(
                    client=client,
                    original_fn=lambda: original_generate_content(self, *args, **kwargs),
                    args=args,
                    kwargs=kwargs,
                    method_name="google.generativeai.GenerativeModel.generate_content",
                    model_name=model_name,
                )

            GenerativeModel.generate_content = traced_generate_content
        except Exception:
            pass

    # Patch generate_content_async
    if hasattr(GenerativeModel, "generate_content_async"):
        try:
            original_generate_content_async = GenerativeModel.generate_content_async

            async def traced_generate_content_async(self, *args: Any, **kwargs: Any) -> Any:
                model_name = _extract_model_name(self) or options.default_model_id or "gemini-pro"
                return await _trace_google_call_async(
                    client=client,
                    original_fn=lambda: original_generate_content_async(self, *args, **kwargs),
                    args=args,
                    kwargs=kwargs,
                    method_name="google.generativeai.GenerativeModel.generate_content_async",
                    model_name=model_name,
                )

            GenerativeModel.generate_content_async = traced_generate_content_async
        except Exception:
            pass


def _patch_new_sdk(client: TraciumClient) -> None:
    try:
        import google.genai as genai  # noqa: F401
        from google.genai.models import AsyncModels, Models
    except ImportError:
        return

    options = get_options()

    # Patch Models.generate_content (Sync)
    if hasattr(Models, "generate_content"):
        try:
            original_generate_content_new = Models.generate_content

            def traced_generate_content_new(self, *args: Any, **kwargs: Any) -> Any:
                model_name = kwargs.get("model")
                if not model_name and args:
                     pass

                if not model_name:
                    model_name = options.default_model_id or "gemini-2.0-flash-exp"

                return _trace_google_call(
                    client=client,
                    original_fn=lambda: original_generate_content_new(self, *args, **kwargs),
                    args=args,
                    kwargs=kwargs,
                    method_name="google.genai.models.Models.generate_content",
                    model_name=str(model_name),
                )

            Models.generate_content = traced_generate_content_new
        except Exception:
             pass

    # Patch AsyncModels.generate_content (Async)
    if hasattr(AsyncModels, "generate_content"):
        try:
            original_generate_content_async_new = AsyncModels.generate_content

            async def traced_generate_content_async_new(self, *args: Any, **kwargs: Any) -> Any:
                model_name = kwargs.get("model") or options.default_model_id or "gemini-2.0-flash-exp"

                return await _trace_google_call_async(
                    client=client,
                    original_fn=lambda: original_generate_content_async_new(self, *args, **kwargs),
                    args=args,
                    kwargs=kwargs,
                    method_name="google.genai.models.AsyncModels.generate_content",
                    model_name=str(model_name),
                )

            AsyncModels.generate_content = traced_generate_content_async_new
        except Exception:
            pass


def _trace_google_call(
    client: TraciumClient,
    original_fn: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    method_name: str,
    model_name: str,
) -> Any:
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

    with trace_handle.span(span_type="llm", name=span_name, model_id=model_name, parent_span_id=parent_span_id) as span_handle:
        if prompt_payload is not None:
            span_handle.record_input(prompt_payload)
        try:
            response = original_fn()
        except Exception as e:
            import traceback
            span_handle.record_output({"error": str(e), "traceback": traceback.format_exc()})
            span_handle.mark_failed(str(e))
            raise

        usage = getattr(response, "usage_metadata", None)
        if usage:
            span_handle.set_token_usage(
                input_tokens=getattr(usage, "prompt_token_count", None),
                output_tokens=getattr(usage, "candidates_token_count", None),
            )

        output_data = getattr(response, "text", None)
        if output_data is None:
             to_dict = getattr(response, "to_dict", None)
             if to_dict:
                 output_data = to_dict()
             else:
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

    span_context = trace_handle.span(span_type="llm", name=span_name, model_id=model_name, parent_span_id=parent_span_id)
    span_handle = span_context.__enter__()
    if prompt_payload is not None:
        span_handle.record_input(prompt_payload)

    try:
        response = await original_fn()
    except Exception as exc:
        span_context.__exit__(type(exc), exc, exc.__traceback__)
        raise

    usage = getattr(response, "usage_metadata", None)
    if usage:
        span_handle.set_token_usage(
            input_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
        )

    output_data = getattr(response, "text", None)
    if output_data is None:
            to_dict = getattr(response, "to_dict", None)
            if to_dict:
                output_data = to_dict()
            else:
                output_data = str(response)

    span_handle.record_output(output_data)
    span_context.__exit__(None, None, None)

    return response
