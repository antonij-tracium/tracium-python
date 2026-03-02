"""
Auto-instrumentation for the OpenAI Python SDK.

Patches all OpenAI resource endpoints (chat completions, completions, embeddings,
images, audio, moderations, responses, assistants) so every API call is captured
as a span.

All tracing operations are designed to be non-blocking and fail-safe.
Tracing errors will never break user applications.
"""

from __future__ import annotations

from typing import Any, cast

from ..core.client import TraciumClient
from ..helpers.global_state import STATE, get_default_tags, get_options
from ..helpers.logging_config import get_logger

openai = None
logger = get_logger()

_ENDPOINTS: list[tuple[str, str, str, str]] = [
    ("resources.chat.completions", "Completions", "AsyncCompletions", "create"),
    ("resources.completions", "Completions", "AsyncCompletions", "create"),
    ("resources.embeddings", "Embeddings", "AsyncEmbeddings", "create"),
    ("resources.images", "Images", "AsyncImages", "generate"),
    ("resources.images", "Images", "AsyncImages", "edit"),
    ("resources.images", "Images", "AsyncImages", "create_variation"),
    ("resources.audio.transcriptions", "Transcriptions", "AsyncTranscriptions", "create"),
    ("resources.audio.translations", "Translations", "AsyncTranslations", "create"),
    ("resources.audio.speech", "Speech", "AsyncSpeech", "create"),
    ("resources.moderations", "Moderations", "AsyncModerations", "create"),
    ("resources.responses", "Responses", "AsyncResponses", "create"),
    ("resources.beta.threads", "Threads", "AsyncThreads", "create_and_run"),
    ("resources.beta.threads", "Threads", "AsyncThreads", "create_and_run_poll"),
    ("resources.beta.threads", "Threads", "AsyncThreads", "create_and_run_stream"),
    ("resources.beta.threads.runs", "Runs", "AsyncRuns", "create"),
    ("resources.beta.threads.runs", "Runs", "AsyncRuns", "create_and_poll"),
    ("resources.beta.threads.runs", "Runs", "AsyncRuns", "stream"),
]

_USAGE_ATTRS = (
    "prompt_tokens",
    "completion_tokens",
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "total_tokens",
)


def _get_str_attr(obj: Any, attr: str) -> str | None:
    try:
        val = obj.get(attr) if isinstance(obj, dict) else getattr(obj, attr, None)
        return val if isinstance(val, str) else None
    except Exception:
        return None


def _extract_chunk_content(chunk: Any) -> str | None:
    try:
        if hasattr(chunk, "choices") and chunk.choices:
            choice = chunk.choices[0]
            if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
                content = choice.delta.content
                return str(content) if content is not None else None
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                content = choice.message.content
                return str(content) if content is not None else None
            if hasattr(choice, "text") and choice.text is not None:
                return str(choice.text)

        if hasattr(chunk, "type") and hasattr(chunk, "delta"):
            event_type = getattr(chunk, "type", "")
            if isinstance(event_type, str) and "delta" in event_type:
                delta = chunk.delta
                if isinstance(delta, str):
                    return delta
                if hasattr(delta, "content") and delta.content:
                    texts = [
                        content_block.text.value
                        for content_block in delta.content
                        if hasattr(content_block, "text")
                        and hasattr(content_block.text, "value")
                        and content_block.text.value
                    ]
                    if texts:
                        return "".join(texts)

        if hasattr(chunk, "output_text") and chunk.output_text:
            return cast(str, chunk.output_text)
    except Exception:
        pass
    return None


def _extract_chunk_tokens(chunk: Any) -> tuple[int | None, int | None, int | None] | None:
    try:
        if hasattr(chunk, "usage") and chunk.usage:
            return _extract_token_usage(chunk)
        if hasattr(chunk, "data") and hasattr(chunk.data, "usage") and chunk.data.usage:
            return _extract_token_usage(chunk.data)
    except Exception:
        pass
    return None


def _finalize_stream(
    span_handle: Any,
    span_context: Any,
    text_parts: list[str],
    tokens: tuple[int | None, int | None, int | None],
) -> None:
    try:
        span_handle.record_output("".join(text_parts) or "(streaming response)")
        if any(t is not None for t in tokens):
            span_handle.set_token_usage(
                input_tokens=tokens[0],
                output_tokens=tokens[1],
                cached_input_tokens=tokens[2],
            )
        span_context.__exit__(None, None, None)
    except Exception:
        pass

    try:
        from ..instrumentation.auto_trace_tracker import (
            _get_web_route_info,
            close_auto_trace_if_needed,
        )

        close_auto_trace_if_needed(force_close=_get_web_route_info() is not None)
    except Exception:
        pass


class _BaseStreamWrapper:
    def __init__(self, original_stream: Any, span_handle: Any, span_context: Any):
        self._stream = original_stream
        self._span_handle = span_handle
        self._span_context = span_context
        self._text_parts: list[str] = []
        self._tokens: tuple[int | None, int | None, int | None] = (None, None, None)
        self._finalized = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def _process_chunk(self, chunk: Any) -> None:
        try:
            if content := _extract_chunk_content(chunk):
                self._text_parts.append(content)
            if tokens := _extract_chunk_tokens(chunk):
                self._tokens = tokens
        except Exception:
            pass

    def _finalize_if_needed(self) -> None:
        if not self._finalized:
            self._finalized = True
            _finalize_stream(self._span_handle, self._span_context, self._text_parts, self._tokens)


class StreamWrapper(_BaseStreamWrapper):
    def __enter__(self) -> StreamWrapper:
        if hasattr(self._stream, "__enter__"):
            self._stream.__enter__()
        return self

    def __exit__(self, *args: Any) -> bool:
        self._finalize_if_needed()
        if hasattr(self._stream, "__exit__"):
            return cast(bool, self._stream.__exit__(*args))
        return False

    def __iter__(self) -> StreamWrapper:
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._stream)
            self._process_chunk(chunk)
            return chunk
        except StopIteration:
            self._finalize_if_needed()
            raise


class AsyncStreamWrapper(_BaseStreamWrapper):
    async def __aenter__(self) -> AsyncStreamWrapper:
        if hasattr(self._stream, "__aenter__"):
            await self._stream.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> bool:
        self._finalize_if_needed()
        if hasattr(self._stream, "__aexit__"):
            return cast(bool, await self._stream.__aexit__(*args))
        return False

    def __aiter__(self) -> AsyncStreamWrapper:
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self._stream.__anext__()
            self._process_chunk(chunk)
            return chunk
        except StopAsyncIteration:
            self._finalize_if_needed()
            raise


def _normalize_input(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    try:
        for key in ("messages", "input", "prompt", "content", "instruction", "instructions"):
            if key in kwargs:
                return kwargs[key]

        if "file" in kwargs:
            f = kwargs["file"]
            name = getattr(f, "name", None) or (f if isinstance(f, str) else "(file)")
            return {"file": name}

        if "assistant_id" in kwargs or "thread_id" in kwargs:
            payload: dict[str, Any] = {
                k: kwargs[k]
                for k in (
                    "assistant_id",
                    "thread_id",
                    "model",
                    "instructions",
                    "additional_messages",
                )
                if k in kwargs
            }
            if "thread" in kwargs and isinstance(kwargs["thread"], dict):
                thread_msgs = kwargs["thread"].get("messages")
                if thread_msgs:
                    payload["messages"] = thread_msgs
            return payload if payload else None

        if args and isinstance(args[0], dict):
            for key in ("messages", "input"):
                if key in args[0]:
                    return args[0][key]

        if args and isinstance(args[0], str):
            return args[0]

        return None
    except Exception:
        return None


def patch_openai(client: TraciumClient) -> None:
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

    try:
        get_options()
    except Exception:
        pass

    def _patch_method(namespace: Any, method_name: str, is_async: bool) -> None:
        try:
            original = getattr(namespace, method_name)
            if is_async:

                async def traced(*args: Any, **kwargs: Any) -> Any:
                    return await _trace_openai_call_async(
                        client, lambda: original(*args, **kwargs), args, kwargs
                    )
            else:

                def traced(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
                    return _trace_openai_call(
                        client, lambda: original(*args, **kwargs), args, kwargs
                    )

            setattr(namespace, method_name, traced)
        except Exception:
            pass

    if hasattr(openai_module, "resources"):
        for module_path, sync_cls, async_cls, method in _ENDPOINTS:
            try:
                ns = openai_module.resources
                for part in module_path.removeprefix("resources.").split("."):
                    ns = getattr(ns, part)
                _patch_method(getattr(ns, sync_cls), method, False)
                _patch_method(getattr(ns, async_cls), method, True)
            except (AttributeError, Exception):
                pass
    elif hasattr(openai_module, "ChatCompletion"):
        _patch_method(openai_module.ChatCompletion, "create", False)

    STATE.openai_patched = True


def _handle_error(e: Exception, span_handle: Any, span_context: Any) -> None:
    try:
        import traceback

        span_handle.record_output(
            {"error": str(e), "error_type": type(e).__name__, "traceback": traceback.format_exc()}
        )
        span_handle.mark_failed(str(e))
    except Exception:
        pass

    try:
        from ..instrumentation.auto_trace_tracker import (
            _get_web_route_info,
            close_auto_trace_if_needed,
            get_current_auto_trace_context,
        )

        if ctx := get_current_auto_trace_context():
            ctx.mark_span_failed()
        close_auto_trace_if_needed(force_close=_get_web_route_info() is not None, error=e)
    except Exception:
        pass

    try:
        span_context.__exit__(type(e), e, e.__traceback__)
    except Exception:
        pass


def _is_streaming(response: Any, kwargs: dict[str, Any], is_async: bool = False) -> bool:
    try:
        if kwargs.get("stream", False):
            return True
        if not hasattr(response, "__aiter__" if is_async else "__iter__"):
            return False
        if hasattr(response, "usage") or isinstance(response, str | bytes | dict | list):
            return False
        if hasattr(response, "read") or hasattr(response, "stream_to_file"):
            return False
        if hasattr(response, "thread_id") and hasattr(response, "status"):
            return False
        return True
    except Exception:
        return False


def _setup_trace(
    client: TraciumClient,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, Any, Any, Any, str | None]:
    from ..helpers.call_hierarchy import get_or_create_function_span
    from ..instrumentation.auto_trace_tracker import (
        get_current_function_for_span,
        get_or_create_auto_trace,
    )

    options = get_options()
    input_payload = _normalize_input(args, kwargs)
    model_id = _get_str_attr(kwargs, "model") or options.default_model_id

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

    return trace_handle, span_context, span_handle, input_payload, model_id


def _is_embedding_response(response: Any) -> bool:
    """True if response is from embeddings API (has data[].embedding)."""
    try:
        if hasattr(response, "data") and isinstance(response.data, list) and response.data:
            first = response.data[0]
            return hasattr(first, "embedding")
    except Exception:
        pass
    return False


def _finalize_response(
    response: Any,
    span_handle: Any,
    span_context: Any,
    model_id: str | None,
    output: Any,
) -> None:
    tokens = _extract_token_usage(response)
    if any(tokens):
        span_handle.set_token_usage(
            input_tokens=tokens[0], output_tokens=tokens[1], cached_input_tokens=tokens[2]
        )

    # Ensure model_id is set for cost calculation (required for embedding pricing on the backend)
    response_model = _get_str_attr(response, "model")
    if response_model:
        if not model_id or _is_embedding_response(response):
            span_handle.set_model_id(response_model)

    span_handle.record_output(output)
    span_context.__exit__(None, None, None)

    from ..instrumentation.auto_trace_tracker import (
        _get_web_route_info,
        close_auto_trace_if_needed,
    )

    close_auto_trace_if_needed(force_close=_get_web_route_info() is not None)


def _trace_openai_call(
    client: TraciumClient,
    original_fn: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    try:
        from ..context.trace_context import current_trace

        if (trace := current_trace()) and trace.tags and "@langchain" in trace.tags:
            return original_fn()
    except Exception:
        pass

    span_context = None
    span_handle = None
    model_id = None

    try:
        _, span_context, span_handle, input_payload, model_id = _setup_trace(client, args, kwargs)

        if _is_assistant_run_input(input_payload, kwargs):
            oai = _get_openai_client(args)
            tid = input_payload.get("thread_id") if isinstance(input_payload, dict) else None
            if oai and tid:
                msgs = _fetch_thread_messages_sync(oai, tid)
                if msgs:
                    input_payload = dict(input_payload)
                    input_payload["messages"] = msgs

        if input_payload is not None:
            span_handle.record_input(input_payload)
    except Exception as e:
        logger.debug(f"OpenAI trace setup failed (continuing without tracing): {e}")

    try:
        response = original_fn()
    except Exception as e:
        if span_handle and span_context:
            _handle_error(e, span_handle, span_context)
        raise

    try:
        if span_handle and span_context:
            if _is_streaming(response, kwargs):
                return StreamWrapper(response, span_handle, span_context)

            output = _extract_output_data(response)
            if _is_completed_assistant_run(output, response):
                oai = _get_openai_client(args)
                if oai:
                    reply = _fetch_assistant_reply_sync(oai, response.thread_id)
                    if reply:
                        output = reply

            _finalize_response(response, span_handle, span_context, model_id, output)
    except Exception as e:
        logger.debug(f"OpenAI response tracing failed (ignored): {e}")

    return response


async def _trace_openai_call_async(
    client: TraciumClient,
    original_fn: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    try:
        from ..context.trace_context import current_trace

        if (trace := current_trace()) and trace.tags and "@langchain" in trace.tags:
            return await original_fn()
    except Exception:
        pass

    span_context = None
    span_handle = None
    model_id = None

    try:
        _, span_context, span_handle, input_payload, model_id = _setup_trace(client, args, kwargs)

        if _is_assistant_run_input(input_payload, kwargs):
            oai = _get_openai_client(args)
            tid = input_payload.get("thread_id") if isinstance(input_payload, dict) else None
            if oai and tid:
                msgs = await _fetch_thread_messages_async(oai, tid)
                if msgs:
                    input_payload = dict(input_payload)
                    input_payload["messages"] = msgs

        if input_payload is not None:
            span_handle.record_input(input_payload)
    except Exception as e:
        logger.debug(f"OpenAI async trace setup failed (continuing without tracing): {e}")

    try:
        response = await original_fn()
    except Exception as e:
        if span_handle and span_context:
            _handle_error(e, span_handle, span_context)
        raise

    try:
        if span_handle and span_context:
            if _is_streaming(response, kwargs, is_async=True):
                return AsyncStreamWrapper(response, span_handle, span_context)

            output = _extract_output_data(response)
            if _is_completed_assistant_run(output, response):
                oai = _get_openai_client(args)
                if oai:
                    reply = await _fetch_assistant_reply_async(oai, response.thread_id)
                    if reply:
                        output = reply

            _finalize_response(response, span_handle, span_context, model_id, output)
    except Exception as e:
        logger.debug(f"OpenAI async response tracing failed (ignored): {e}")

    return response


def _usage_to_dict(usage: Any) -> dict[str, Any]:
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "model_dump"):
        try:
            return cast(dict[str, Any], usage.model_dump())
        except Exception:
            pass
    return cast(
        dict[str, Any],
        {
            attr: getattr(usage, attr)
            for attr in _USAGE_ATTRS
            if hasattr(usage, attr) and getattr(usage, attr) is not None
        },
    )


def _safe_int(val: Any) -> int | None:
    try:
        return int(val) if val is not None else None
    except (ValueError, TypeError):
        return None


def _extract_token_usage(response: Any) -> tuple[int | None, int | None, int | None]:
    try:
        usage = getattr(response, "usage", None)
        if not usage:
            return None, None, None

        usage_dict = _usage_to_dict(usage)

        prompt_tokens = usage_dict.get("prompt_tokens") or usage_dict.get("input_tokens")
        completion_tokens = usage_dict.get("completion_tokens") or usage_dict.get("output_tokens")

        if prompt_tokens is None and completion_tokens is None:
            total = usage_dict.get("total_tokens")
            if total is not None:
                prompt_tokens = total

        cached_tokens = None
        for key in ("prompt_token_details", "completion_token_details"):
            if cached_tokens is None and isinstance(usage_dict.get(key), dict):
                cached_tokens = usage_dict[key].get("cached_tokens")
        if cached_tokens is None:
            cached_tokens = usage_dict.get("cached_input_tokens")

        return _safe_int(prompt_tokens), _safe_int(completion_tokens), _safe_int(cached_tokens)
    except Exception:
        return None, None, None


def _extract_message_text(content_blocks: Any) -> str | None:
    if not isinstance(content_blocks, list):
        return None
    texts = [
        block.text.value
        for block in content_blocks
        if hasattr(block, "text") and hasattr(block.text, "value")
    ]
    return "\n".join(texts) if texts else None


def _get_openai_client(args: tuple[Any, ...]) -> Any:
    try:
        return getattr(args[0], "_client", None) if args else None
    except Exception:
        return None


def _is_assistant_run_input(payload: Any, kwargs: dict[str, Any]) -> bool:
    return (
        isinstance(payload, dict)
        and "assistant_id" in payload
        and "messages" not in payload
        and "thread" not in kwargs
    )


def _is_completed_assistant_run(output: Any, response: Any) -> bool:
    return (
        isinstance(output, dict)
        and output.get("status") == "completed"
        and hasattr(response, "thread_id")
    )


def _parse_thread_messages(page: Any) -> list[dict[str, str]] | None:
    out = [
        {"role": msg.role, "content": text}
        for msg in page.data
        if (text := _extract_message_text(msg.content))
    ]
    return out or None


def _extract_assistant_reply(page: Any) -> str | None:
    if page.data and page.data[0].role == "assistant":
        return _extract_message_text(page.data[0].content)
    return None


def _fetch_thread_messages_sync(openai_client: Any, thread_id: str) -> list[dict[str, str]] | None:
    try:
        page = openai_client.beta.threads.messages.list(thread_id=thread_id, order="asc")
        return _parse_thread_messages(page)
    except Exception:
        return None


async def _fetch_thread_messages_async(
    openai_client: Any, thread_id: str
) -> list[dict[str, str]] | None:
    try:
        page = await openai_client.beta.threads.messages.list(thread_id=thread_id, order="asc")
        return _parse_thread_messages(page)
    except Exception:
        return None


def _fetch_assistant_reply_sync(openai_client: Any, thread_id: str) -> str | None:
    try:
        page = openai_client.beta.threads.messages.list(thread_id=thread_id, limit=1, order="desc")
        return _extract_assistant_reply(page)
    except Exception:
        return None


async def _fetch_assistant_reply_async(openai_client: Any, thread_id: str) -> str | None:
    try:
        page = await openai_client.beta.threads.messages.list(
            thread_id=thread_id, limit=1, order="desc"
        )
        return _extract_assistant_reply(page)
    except Exception:
        return None


def _extract_output_data(response: Any) -> Any:
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
        if hasattr(response, "output_text") and response.output_text:
            return response.output_text
        if hasattr(response, "output") and response.output:
            output = response.output
            if isinstance(output, list):
                return [
                    item.model_dump() if hasattr(item, "model_dump") else item for item in output
                ]
            return output.model_dump() if hasattr(output, "model_dump") else output
    except Exception:
        pass

    try:
        if hasattr(response, "thread_id") and hasattr(response, "status"):
            status = response.status
            if status != "completed":
                if hasattr(response, "required_action") and response.required_action:
                    return {"status": status, "required_action": "tool_calls_pending"}
                return {"status": status}
            return {"status": "completed"}
    except Exception:
        pass

    try:
        if hasattr(response, "role") and hasattr(response, "content"):
            text = _extract_message_text(response.content)
            if text is not None:
                return text
    except Exception:
        pass

    try:
        if hasattr(response, "data") and isinstance(response.data, list) and response.data:
            first = response.data[0]
            if hasattr(first, "embedding"):
                dims = len(first.embedding) if first.embedding else None
                return {"embeddings_count": len(response.data), "dimensions": dims}
            if hasattr(first, "url") or hasattr(first, "b64_json"):
                return [
                    {
                        k: ("[base64 image data omitted]" if k == "b64_json" else getattr(item, k))
                        for k in ("url", "revised_prompt", "b64_json")
                        if hasattr(item, k) and getattr(item, k)
                    }
                    for item in response.data
                ]
            text = _extract_message_text(getattr(first, "content", None))
            if text is not None:
                return text
    except Exception:
        pass

    try:
        if hasattr(response, "text") and isinstance(response.text, str):
            return response.text
    except Exception:
        pass

    try:
        if hasattr(response, "results") and isinstance(response.results, list):
            return [r.model_dump() if hasattr(r, "model_dump") else r for r in response.results]
    except Exception:
        pass

    try:
        return response.model_dump() if hasattr(response, "model_dump") else str(response)
    except Exception:
        return str(response)
