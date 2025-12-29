"""
Auto-instrumentation for the Google Generative AI Python SDKs (Gemini).
Supports both `google.generativeai` (legacy) and `google.genai` (new).

All tracing operations are designed to be non-blocking and fail-safe.
Tracing errors will never break user applications.
"""

from __future__ import annotations

from typing import Any

from ..core.client import TraciumClient
from ..helpers.global_state import STATE, get_default_tags, get_options
from ..helpers.logging_config import get_logger

logger = get_logger()


def _normalize_prompt(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str | dict[str, Any] | None:
    try:
        if "contents" in kwargs:
            return {"contents": kwargs["contents"]}
        if "prompt" in kwargs:
            prompt = kwargs["prompt"]
            if isinstance(prompt, (str | dict)):
                return prompt
            return None
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
    except Exception:
        pass
    return None


def _extract_model_name(model_obj: Any) -> str | None:
    try:
        if hasattr(model_obj, "model_name"):
            return str(model_obj.model_name)
        if hasattr(model_obj, "_model_name"):
            return str(model_obj._model_name)
    except Exception:
        pass
    return None


def patch_google_genai(client: TraciumClient) -> None:
    if STATE.google_patched:
        return

    _patch_legacy_sdk(client)
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
    except Exception:
        return

    try:
        options = get_options()
    except Exception:
        options = None

    if hasattr(GenerativeModel, "generate_content"):
        try:
            original_generate_content = GenerativeModel.generate_content

            def traced_generate_content(self, *args: Any, **kwargs: Any) -> Any:
                model_name = (
                    _extract_model_name(self)
                    or (options.default_model_id if options else None)
                    or "gemini-pro"
                )
                return _trace_google_call(
                    client=client,
                    original_fn=lambda: original_generate_content(self, *args, **kwargs),
                    args=args,
                    kwargs=kwargs,
                    method_name="google.generativeai.GenerativeModel.generate_content",
                    model_name=model_name,
                )

            GenerativeModel.generate_content = traced_generate_content  # type: ignore[method-assign]
        except Exception:
            pass

    if hasattr(GenerativeModel, "generate_content_async"):
        try:
            original_generate_content_async = GenerativeModel.generate_content_async

            async def traced_generate_content_async(self, *args: Any, **kwargs: Any) -> Any:
                model_name = (
                    _extract_model_name(self)
                    or (options.default_model_id if options else None)
                    or "gemini-pro"
                )
                return await _trace_google_call_async(
                    client=client,
                    original_fn=lambda: original_generate_content_async(self, *args, **kwargs),
                    args=args,
                    kwargs=kwargs,
                    method_name="google.generativeai.GenerativeModel.generate_content_async",
                    model_name=model_name,
                )

            GenerativeModel.generate_content_async = traced_generate_content_async  # type: ignore[method-assign]
        except Exception:
            pass


def _patch_new_sdk(client: TraciumClient) -> None:
    try:
        import google.genai as genai  # noqa: F401
        from google.genai.models import AsyncModels, Models
    except ImportError:
        return
    except Exception:
        return

    try:
        options = get_options()
    except Exception:
        options = None

    if hasattr(Models, "generate_content"):
        try:
            original_generate_content_new = Models.generate_content

            def traced_generate_content_new(self, *args: Any, **kwargs: Any) -> Any:
                model_name = kwargs.get("model")
                if not model_name:
                    model_name = (
                        options.default_model_id if options else None
                    ) or "gemini-2.0-flash-exp"

                return _trace_google_call(
                    client=client,
                    original_fn=lambda: original_generate_content_new(self, *args, **kwargs),
                    args=args,
                    kwargs=kwargs,
                    method_name="google.genai.models.Models.generate_content",
                    model_name=str(model_name),
                )

            Models.generate_content = traced_generate_content_new  # type: ignore[method-assign]
        except Exception:
            pass

    if hasattr(AsyncModels, "generate_content"):
        try:
            original_generate_content_async_new = AsyncModels.generate_content

            async def traced_generate_content_async_new(self, *args: Any, **kwargs: Any) -> Any:
                model_name = (
                    kwargs.get("model")
                    or (options.default_model_id if options else None)
                    or "gemini-2.0-flash-exp"
                )

                return await _trace_google_call_async(
                    client=client,
                    original_fn=lambda: original_generate_content_async_new(self, *args, **kwargs),
                    args=args,
                    kwargs=kwargs,
                    method_name="google.genai.models.AsyncModels.generate_content",
                    model_name=str(model_name),
                )

            AsyncModels.generate_content = traced_generate_content_async_new  # type: ignore[method-assign]
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
    """Trace a synchronous Google API call."""
    trace_handle = None
    span_handle = None
    span_context = None

    try:
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
            span_type="llm", name=span_name, model_id=model_name, parent_span_id=parent_span_id
        )
        span_handle = span_context.__enter__()

        if prompt_payload is not None:
            span_handle.record_input(prompt_payload)
    except Exception as e:
        logger.debug(f"Google trace setup failed (continuing without tracing): {e}")

    try:
        response = original_fn()
    except Exception as e:
        if span_handle and span_context:
            try:
                import traceback

                span_handle.record_output({"error": str(e), "traceback": traceback.format_exc()})
                span_handle.mark_failed(str(e))
            except Exception:
                pass

            try:
                from ..instrumentation.auto_trace_tracker import get_current_auto_trace_context

                auto_context = get_current_auto_trace_context()
                if auto_context:
                    auto_context.mark_span_failed()
            except Exception:
                pass

            try:
                span_context.__exit__(type(e), e, e.__traceback__)
            except Exception:
                pass

            try:
                from ..instrumentation.auto_trace_tracker import (
                    _get_web_route_info,
                    close_auto_trace_if_needed,
                )

                is_web_context = _get_web_route_info() is not None
                close_auto_trace_if_needed(force_close=is_web_context, error=e)
            except Exception:
                pass
        raise

    try:
        if span_handle and span_context:
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

            from ..instrumentation.auto_trace_tracker import (
                _get_web_route_info,
                close_auto_trace_if_needed,
            )

            is_web_context = _get_web_route_info() is not None
            close_auto_trace_if_needed(force_close=is_web_context)
    except Exception as e:
        logger.debug(f"Google response tracing failed (ignored): {e}")

    return response


async def _trace_google_call_async(
    client: TraciumClient,
    original_fn: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    method_name: str,
    model_name: str,
) -> Any:
    """Trace an asynchronous Google API call."""
    trace_handle = None
    span_handle = None
    span_context = None

    try:
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
            span_type="llm", name=span_name, model_id=model_name, parent_span_id=parent_span_id
        )
        span_handle = span_context.__enter__()

        if prompt_payload is not None:
            span_handle.record_input(prompt_payload)
    except Exception as e:
        logger.debug(f"Google async trace setup failed (continuing without tracing): {e}")

    try:
        response = await original_fn()
    except Exception as exc:
        if span_handle and span_context:
            try:
                span_context.__exit__(type(exc), exc, exc.__traceback__)
            except Exception:
                pass

            try:
                from ..instrumentation.auto_trace_tracker import get_current_auto_trace_context

                auto_context = get_current_auto_trace_context()
                if auto_context:
                    auto_context.mark_span_failed()
            except Exception:
                pass

            try:
                from ..instrumentation.auto_trace_tracker import (
                    _get_web_route_info,
                    close_auto_trace_if_needed,
                )

                is_web_context = _get_web_route_info() is not None
                close_auto_trace_if_needed(force_close=is_web_context, error=exc)
            except Exception:
                pass
        raise

    try:
        if span_handle and span_context:
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

            from ..instrumentation.auto_trace_tracker import (
                _get_web_route_info,
                close_auto_trace_if_needed,
            )

            is_web_context = _get_web_route_info() is not None
            close_auto_trace_if_needed(force_close=is_web_context)
    except Exception as e:
        logger.debug(f"Google async response tracing failed (ignored): {e}")

    return response
