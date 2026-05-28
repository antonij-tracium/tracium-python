"""
``requests`` library hook for universal LLM HTTP capture.

``requests`` doesn't have a transport abstraction we can wrap; the cleanest
intercept is patching :meth:`requests.Session.send`. ``requests.get/post/...``
all funnel through ``Session.send`` so we get every call from one wrapper.

Streaming responses (``stream=True``) are handled by replacing the response's
``raw.read``/``iter_content`` machinery with a tee'd version that feeds chunks
to an :class:`SSEAccumulator` while the user iterates as normal.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import datetime
from typing import Any

from ...utils.datetime_utils import _utcnow
from .dedup import is_owned
from .emit import emit_llm_span
from .providers import LLMCall, detect_provider, parse
from .sse import SSEAccumulator

logger = logging.getLogger(__name__)

_INSTALLED = False
_STREAM_CONTENT_TYPES = ("text/event-stream", "application/x-ndjson")
_AWS_EVENTSTREAM_CT = "application/vnd.amazon.eventstream"


def install() -> None:
    """Monkey-patch :meth:`requests.Session.send`. Idempotent."""
    global _INSTALLED
    if _INSTALLED:
        return
    try:
        import requests  # type: ignore[import-untyped, unused-ignore]
    except ImportError:
        return

    original_send = requests.Session.send

    def patched_send(self: Any, request: Any, **kwargs: Any) -> Any:
        url = getattr(request, "url", "") or ""
        if is_owned() or detect_provider(url) is None:
            return original_send(self, request, **kwargs)

        started_at = _utcnow()
        request_body = _request_body(request)

        try:
            response = original_send(self, request, **kwargs)
        except BaseException as e:
            _emit_for_error(url, request_body, e, started_at)
            raise

        # `stream=True` may be set per-call OR on the Session (`session.stream = True`).
        # We must respect both — reading response.content on a streamed response
        # consumes the body and breaks the caller's `iter_content`/`iter_lines`.
        is_streaming_call = bool(kwargs.get("stream")) or bool(getattr(self, "stream", False))
        if is_streaming_call and _looks_like_aws_eventstream(response):
            _wrap_bedrock_eventstream_response(response, url, request_body, started_at)
            return response
        if is_streaming_call and _looks_like_stream(response):
            _wrap_streaming_response(response, url, request_body, started_at)
            return response

        # If the caller intends to stream but the body doesn't look like a
        # known stream content-type, skip the body read rather than draining
        # the user's response.
        if is_streaming_call:
            _emit_from_buffered(url, request_body, None, response.status_code, started_at)
            return response

        try:
            body = response.content  # forces read
        except Exception:
            body = None
        _emit_from_buffered(url, request_body, body, response.status_code, started_at)
        return response

    requests.Session.send = patched_send
    _INSTALLED = True
    logger.info("tracium: http_capture installed for requests")


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _request_body(request: Any) -> bytes | None:
    body = getattr(request, "body", None)
    if body is None:
        return None
    if isinstance(body, bytes | bytearray):
        return bytes(body)
    if isinstance(body, str):
        return body.encode("utf-8", errors="replace")
    return None


def _looks_like_stream(response: Any) -> bool:
    ct = (response.headers.get("content-type") or "").lower() if response is not None else ""
    return any(ct.startswith(t) for t in _STREAM_CONTENT_TYPES)


def _looks_like_aws_eventstream(response: Any) -> bool:
    ct = (response.headers.get("content-type") or "").lower() if response is not None else ""
    return ct.startswith(_AWS_EVENTSTREAM_CT)


def _wrap_bedrock_eventstream_response(
    response: Any, url: str, request_body: bytes | None, started_at: datetime
) -> None:
    """Tee a Bedrock streaming response into a byte buffer and reconstruct on close."""
    from urllib.parse import urlparse

    from .bedrock_stream import reconstruct_response as reconstruct_bedrock_stream
    from .providers import _model_from_bedrock_url

    buffer = bytearray()
    original_iter_content = response.iter_content
    original_iter_lines = response.iter_lines
    original_close = response.close
    finalized = {"done": False}

    def finalize() -> None:
        if finalized["done"]:
            return
        finalized["done"] = True
        try:
            path = urlparse(url).path or ""
            model = _model_from_bedrock_url(url)
            reconstructed = reconstruct_bedrock_stream(path, model, bytes(buffer))
            call = parse(url, request_body, reconstructed, response.status_code)
        except Exception as e:
            logger.debug(
                "tracium Bedrock stream reconstruct failed: %s: %s",
                type(e).__name__,
                e,
            )
            return
        if call is not None:
            emit_llm_span(call, started_at)

    def teed_iter_content(*args: Any, **kwargs: Any) -> Iterator[bytes]:
        try:
            for chunk in original_iter_content(*args, **kwargs):
                if isinstance(chunk, bytes | bytearray):
                    buffer.extend(chunk)
                yield chunk
        finally:
            finalize()

    def teed_iter_lines(*args: Any, **kwargs: Any) -> Iterator[bytes | str]:
        try:
            for line in original_iter_lines(*args, **kwargs):
                if isinstance(line, bytes | bytearray):
                    buffer.extend(bytes(line) + b"\n")
                yield line
        finally:
            finalize()

    def closing() -> None:
        try:
            original_close()
        finally:
            finalize()

    response.iter_content = teed_iter_content
    response.iter_lines = teed_iter_lines
    response.close = closing


def _wrap_streaming_response(
    response: Any, url: str, request_body: bytes | None, started_at: datetime
) -> None:
    accumulator = SSEAccumulator()
    original_iter_content = response.iter_content
    original_iter_lines = response.iter_lines
    original_close = response.close
    finalized = {"done": False}

    def finalize() -> None:
        if finalized["done"]:
            return
        finalized["done"] = True
        try:
            reconstructed = accumulator.finalize()
            call = parse(url, request_body, reconstructed, response.status_code)
        except Exception as e:
            logger.debug("tracium SSE finalize failed: %s: %s", type(e).__name__, e)
            return
        if call is not None:
            emit_llm_span(call, started_at)

    def teed_iter_content(*args: Any, **kwargs: Any) -> Iterator[bytes]:
        try:
            for chunk in original_iter_content(*args, **kwargs):
                if isinstance(chunk, bytes | bytearray):
                    try:
                        accumulator.feed(bytes(chunk))
                    except Exception:
                        pass
                yield chunk
        finally:
            finalize()

    def teed_iter_lines(*args: Any, **kwargs: Any) -> Iterator[bytes | str]:
        try:
            for line in original_iter_lines(*args, **kwargs):
                if isinstance(line, bytes | bytearray):
                    try:
                        accumulator.feed(bytes(line) + b"\n")
                    except Exception:
                        pass
                elif isinstance(line, str):
                    try:
                        accumulator.feed(line.encode("utf-8") + b"\n")
                    except Exception:
                        pass
                yield line
        finally:
            finalize()

    def closing() -> None:
        try:
            original_close()
        finally:
            finalize()

    response.iter_content = teed_iter_content
    response.iter_lines = teed_iter_lines
    response.close = closing


def _emit_from_buffered(
    url: str,
    request_body: bytes | None,
    response_body: bytes | None,
    status_code: int,
    started_at: datetime,
) -> None:
    try:
        call = parse(url, request_body, response_body, status_code)
    except Exception as e:
        logger.debug("tracium parse failed: %s: %s", type(e).__name__, e)
        return
    if call is not None:
        emit_llm_span(call, started_at)


def _emit_for_error(
    url: str,
    request_body: bytes | None,
    error: BaseException,
    started_at: datetime,
) -> None:
    found = detect_provider(url)
    call = LLMCall(
        provider=found[0] if found else "unknown",
        error=f"{type(error).__name__}: {error}",
    )
    try:
        emit_llm_span(call, started_at)
    except Exception:
        pass
