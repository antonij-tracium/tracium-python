"""
Stream wrapper classes for intercepting Anthropic streaming responses.

All wrappers are designed to be fail-safe - any tracing errors will
never break the streaming response for the user.
"""

from __future__ import annotations

from typing import Any

from .utils import extract_usage


class TextStreamWrapper:
    """Wrapper for text_stream that intercepts text chunks."""

    def __init__(self, original_text_stream: Any, text_parts: list[str]):
        self._text_stream = original_text_stream
        self._text_parts = text_parts

    def __iter__(self):
        return self

    def __next__(self):
        text = next(self._text_stream)
        try:
            if text:
                self._text_parts.append(text)
        except Exception:
            pass
        return text


class AsyncTextStreamWrapper:
    """Wrapper for async text_stream that intercepts text chunks."""

    def __init__(self, original_text_stream: Any, text_parts: list[str]):
        self._text_stream = original_text_stream
        self._text_parts = text_parts

    def __aiter__(self):
        return self

    async def __anext__(self):
        text = await self._text_stream.__anext__()
        try:
            if text:
                self._text_parts.append(text)
        except Exception:
            pass
        return text


def _finalize_stream(
    span_handle: Any,
    span_context: Any,
    text_parts: list[str],
    input_tokens: int | None,
    output_tokens: int | None,
) -> None:
    """Finalize stream: record output, tokens, and close span. Never raises."""
    try:
        span_handle.record_output("".join(text_parts) if text_parts else "(streaming response)")
    except Exception:
        pass

    try:
        if input_tokens is not None or output_tokens is not None:
            span_handle.set_token_usage(input_tokens=input_tokens, output_tokens=output_tokens)
    except Exception:
        pass

    try:
        span_context.__exit__(None, None, None)
    except Exception:
        pass

    try:
        from ...instrumentation.auto_trace_tracker import (
            _get_web_route_info,
            close_auto_trace_if_needed,
        )

        close_auto_trace_if_needed(force_close=_get_web_route_info() is not None)
    except Exception:
        pass


class StreamWrapper:
    """Wrapper for the actual stream object that intercepts chunks."""

    def __init__(self, original_stream: Any, span_handle: Any, span_context: Any):
        self._stream = original_stream
        self._span_handle = span_handle
        self._span_context = span_context
        self._text_parts: list[str] = []
        self._input_tokens: int | None = None
        self._output_tokens: int | None = None
        self._text_stream_wrapped = False
        self._finalized = False

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._stream, name)
        if name == "text_stream" and not self._text_stream_wrapped:
            self._text_stream_wrapped = True
            return TextStreamWrapper(attr, self._text_parts)
        return attr

    def __iter__(self):
        return self

    def __next__(self):
        try:
            chunk = next(self._stream)

            try:
                chunk_type = getattr(chunk, "type", None)

                if chunk_type == "content_block_delta":
                    delta = getattr(chunk, "delta", None)
                    if delta and (text := getattr(delta, "text", None)):
                        self._text_parts.append(text)
                elif chunk_type == "content_block_start":
                    content_block = getattr(chunk, "content_block", None)
                    if content_block and (text := getattr(content_block, "text", None)):
                        self._text_parts.append(text)

                if chunk_type in ("message_stop", "message_delta"):
                    usage = getattr(chunk, "usage", None)
                    if usage:
                        self._input_tokens, self._output_tokens = extract_usage(usage)
            except Exception:
                pass

            return chunk
        except StopIteration:
            if not self._finalized:
                self._finalized = True
                _finalize_stream(
                    self._span_handle,
                    self._span_context,
                    self._text_parts,
                    self._input_tokens,
                    self._output_tokens,
                )
            raise


class AsyncStreamWrapper:
    """Wrapper for the actual async stream object that intercepts chunks."""

    def __init__(self, original_stream: Any, span_handle: Any, span_context: Any):
        self._stream = original_stream
        self._span_handle = span_handle
        self._span_context = span_context
        self._text_parts: list[str] = []
        self._input_tokens: int | None = None
        self._output_tokens: int | None = None
        self._text_stream_wrapped = False
        self._finalized = False

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._stream, name)
        if name == "text_stream" and not self._text_stream_wrapped:
            self._text_stream_wrapped = True
            return AsyncTextStreamWrapper(attr, self._text_parts)
        return attr

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            chunk = await self._stream.__anext__()

            try:
                chunk_type = getattr(chunk, "type", None)

                if chunk_type == "content_block_delta":
                    delta = getattr(chunk, "delta", None)
                    if delta and (text := getattr(delta, "text", None)):
                        self._text_parts.append(text)
                elif chunk_type == "content_block_start":
                    content_block = getattr(chunk, "content_block", None)
                    if content_block and (text := getattr(content_block, "text", None)):
                        self._text_parts.append(text)

                if chunk_type in ("message_stop", "message_delta"):
                    usage = getattr(chunk, "usage", None)
                    if usage:
                        self._input_tokens, self._output_tokens = extract_usage(usage)
            except Exception:
                pass

            return chunk
        except StopAsyncIteration:
            if not self._finalized:
                self._finalized = True
                _finalize_stream(
                    self._span_handle,
                    self._span_context,
                    self._text_parts,
                    self._input_tokens,
                    self._output_tokens,
                )
            raise
