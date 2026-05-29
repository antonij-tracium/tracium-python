"""
httpx transport wrappers for universal LLM HTTP capture.

We wrap whichever ``BaseTransport`` / ``AsyncBaseTransport`` an :class:`httpx.Client`
or :class:`httpx.AsyncClient` is using. Every outbound request to a known LLM
endpoint generates an LLM span on the active trace. Other requests pass through
unchanged.

Streaming responses (Server-Sent Events) are captured by wrapping the response's
byte stream with an :class:`SSEAccumulator`. The user's iteration is unaffected:
they get the same bytes; we observe a copy as it flows.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator
from datetime import datetime
from typing import Any, cast

import httpx

from ...utils.datetime_utils import _utcnow
from .bedrock_stream import reconstruct_response as reconstruct_bedrock_stream
from .dedup import is_owned
from .emit import emit_llm_span
from .providers import LLMCall, detect_provider, parse
from .sse import SSEAccumulator

logger = logging.getLogger(__name__)

_STREAM_CONTENT_TYPES = ("text/event-stream", "application/x-ndjson")
_AWS_EVENTSTREAM_CT = "application/vnd.amazon.eventstream"

# Marker attribute set on every TraciumHTTPXTransport / TraciumAsyncHTTPXTransport
# instance. The global ``install()`` patch reads it to avoid double-wrapping a
# transport that's already wrapped (manually or otherwise).
_TRACIUM_WRAPPED_FLAG = "_tracium_http_capture_wrapped"
# Tests / users can mark a transport (e.g. ``httpx.MockTransport``) to opt out
# of wrapping by setting ``transport._tracium_skip = True`` before constructing
# the Client. Wrapping a MockTransport breaks tests that rely on transport
# identity checks or that emit bodies the tracium parser can't handle.
_TRACIUM_SKIP_FLAG = "_tracium_skip"


def _should_skip_transport(transport: Any) -> bool:
    if transport is None:
        return True
    if getattr(transport, _TRACIUM_WRAPPED_FLAG, False):
        return True
    if getattr(transport, _TRACIUM_SKIP_FLAG, False):
        return True
    return False


def _looks_like_stream(response: httpx.Response) -> bool:
    ct = (response.headers.get("content-type") or "").lower()
    return any(ct.startswith(t) for t in _STREAM_CONTENT_TYPES)


def _looks_like_aws_eventstream(response: httpx.Response) -> bool:
    ct = (response.headers.get("content-type") or "").lower()
    return ct.startswith(_AWS_EVENTSTREAM_CT)


class _BytesAccumulator:
    """Trivial byte-buffer with the same ``.feed(bytes)`` interface as
    :class:`SSEAccumulator`. Used for AWS event-stream framing (frames span
    chunk boundaries, so we collect the whole body before decoding) and as a
    generic tee target for non-streaming responses so we don't have to call
    ``response.read()`` and defeat the caller's ``stream=True``.

    Capped at ``TRACIUM_HTTP_CAPTURE_MAX_BYTES`` (default 1 MiB) so a
    multi-MB response can't blow up the worker. Excess bytes are dropped;
    parsers downstream see a truncated body.
    """

    __slots__ = ("_buf", "_max_bytes", "_truncated")

    def __init__(self) -> None:
        from .sse import _max_capture_bytes

        self._buf = bytearray()
        self._max_bytes = _max_capture_bytes()
        self._truncated = False

    def feed(self, chunk: bytes) -> None:
        if not chunk:
            return
        if self._max_bytes <= 0:
            self._buf.extend(chunk)
            return
        remaining = self._max_bytes - len(self._buf)
        if remaining <= 0:
            self._truncated = True
            return
        if len(chunk) <= remaining:
            self._buf.extend(chunk)
        else:
            self._buf.extend(chunk[:remaining])
            self._truncated = True

    def get_bytes(self) -> bytes:
        return bytes(self._buf)

    @property
    def truncated(self) -> bool:
        return self._truncated


# Back-compat alias for the (now-generic) collector — used by emit helpers
# below and by tests that import the symbol directly.
_BedrockByteCollector = _BytesAccumulator


def _read_request_body(request: httpx.Request) -> bytes | None:
    """Return the buffered request body without consuming a streaming body.

    httpx populates ``request._content`` when the body was passed as bytes,
    str, json, or data (the common case for LLM SDK calls). For requests built
    from a lazy iterator / async iterator (``content=<iter>``), calling
    ``request.read()`` would consume the iterator and break the actual HTTP
    send. We only return the body when it's already materialized; otherwise we
    capture the span without an input body rather than risk corrupting the
    user's request.
    """
    try:
        content = getattr(request, "_content", None)
        if isinstance(content, bytes | bytearray) and content:
            return bytes(content)
    except Exception:
        pass
    return None


def _peek_buffered_content(response: httpx.Response) -> bytes | None:
    """If httpx already has the full body buffered (e.g. ``MockTransport``
    constructed with ``content=`` / ``json=``), return it; else ``None``.

    Lets us parse SSE bodies that arrived as a single buffered chunk without
    breaking iteration semantics for the caller.
    """
    content = getattr(response, "_content", None)
    if isinstance(content, bytes | bytearray) and content:
        return bytes(content)
    return None


class _CapturingByteStream(httpx.SyncByteStream):
    """Sync stream wrapper that tees chunks into an SSE accumulator.

    The emit callback (`on_close`) is fired exactly once — whichever happens
    first between iteration completing or the stream being closed. Either path
    produces the same captured span.
    """

    def __init__(
        self,
        wrapped: httpx.SyncByteStream,
        accumulator: Any,  # SSEAccumulator | _BedrockByteCollector
        on_close: Any,
    ) -> None:
        self._wrapped = wrapped
        self._accum = accumulator
        self._on_close = on_close
        self._fired = False

    def __iter__(self) -> Iterator[bytes]:
        try:
            for chunk in self._wrapped:
                try:
                    self._accum.feed(chunk)
                except Exception:
                    pass
                yield chunk
        finally:
            self._fire()

    def close(self) -> None:
        try:
            self._wrapped.close()
        finally:
            self._fire()

    def _fire(self) -> None:
        if self._fired:
            return
        self._fired = True
        try:
            self._on_close()
        except Exception:
            pass


class _CapturingAsyncByteStream(httpx.AsyncByteStream):
    """Async stream wrapper that tees chunks into an SSE accumulator.

    Fires its emit callback exactly once on either iteration end or close.
    """

    def __init__(
        self,
        wrapped: httpx.AsyncByteStream,
        accumulator: Any,  # SSEAccumulator | _BedrockByteCollector
        on_close: Any,
    ) -> None:
        self._wrapped = wrapped
        self._accum = accumulator
        self._on_close = on_close
        self._fired = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self._wrapped:
                try:
                    self._accum.feed(chunk)
                except Exception:
                    pass
                yield chunk
        finally:
            self._fire()

    async def aclose(self) -> None:
        try:
            await self._wrapped.aclose()
        finally:
            self._fire()

    def _fire(self) -> None:
        if self._fired:
            return
        self._fired = True
        try:
            self._on_close()
        except Exception:
            pass


def _emit_from_stream(
    url: str,
    request_body: bytes | None,
    response: httpx.Response,
    accumulator: SSEAccumulator,
    started_at: datetime,
) -> None:
    """Build an LLMCall from the assembled SSE chunks and emit the span."""
    try:
        reconstructed = accumulator.finalize()
        call = parse(url, request_body, reconstructed, response.status_code)
    except Exception as e:
        logger.debug("tracium SSE finalize failed: %s: %s", type(e).__name__, e)
        return
    if call is not None:
        emit_llm_span(call, started_at)


def _emit_from_buffered(
    url: str,
    request_body: bytes | None,
    response_body: bytes | None,
    status_code: int,
    started_at: datetime,
) -> None:
    """Build an LLMCall from a fully-buffered response and emit the span."""
    try:
        call = parse(url, request_body, response_body, status_code)
    except Exception as e:
        logger.debug("tracium parse failed: %s: %s", type(e).__name__, e)
        return
    if call is not None:
        emit_llm_span(call, started_at)


def _make_buffered_emit_cb(
    url: str,
    request_body: bytes | None,
    response: httpx.Response,
    collector: _BytesAccumulator,
    started_at: datetime,
) -> Any:
    """Build an `on_close` callback that logs truncation (if any) and emits."""
    def _cb() -> None:
        if collector.truncated:
            logger.debug(
                "tracium: captured response body truncated at %d bytes for %s",
                len(collector.get_bytes()),
                url,
            )
        _emit_from_buffered(
            url, request_body, collector.get_bytes(), response.status_code, started_at
        )
    return _cb


def _emit_from_bedrock_eventstream(
    url: str,
    request_body: bytes | None,
    response: httpx.Response,
    collector: _BedrockByteCollector,
    started_at: datetime,
) -> None:
    """Reconstruct a Bedrock streaming response and emit an LLM span.

    Bedrock streams use AWS's ``application/vnd.amazon.eventstream`` framing
    instead of SSE. We collect the raw bytes, run them through the
    eventstream parser + per-model assembler, then hand the rebuilt
    non-streaming response dict to the existing Bedrock parser.
    """
    try:
        from urllib.parse import urlparse

        path = urlparse(url).path or ""
        from .providers import _model_from_bedrock_url

        model = _model_from_bedrock_url(url)
        if collector.truncated:
            logger.debug(
                "tracium: Bedrock eventstream buffer truncated at %d bytes for %s",
                len(collector.get_bytes()), url,
            )
        reconstructed = reconstruct_bedrock_stream(path, model, collector.get_bytes())
        call = parse(url, request_body, reconstructed, response.status_code)
    except Exception as e:
        logger.debug("tracium Bedrock stream reconstruct failed: %s: %s", type(e).__name__, e)
        return
    if call is not None:
        emit_llm_span(call, started_at)


def _emit_for_error(
    url: str,
    request_body: bytes | None,
    error: BaseException,
    started_at: datetime,
) -> None:
    """Emit a span for a request that failed at the transport level."""
    call = LLMCall(
        provider=(detect_provider(url) or ("unknown", None))[0],
        error=f"{type(error).__name__}: {error}",
    )
    try:
        emit_llm_span(call, started_at)
    except Exception:
        pass


class TraciumHTTPXTransport(httpx.BaseTransport):
    """Sync transport wrapper. Captures any request to a known LLM endpoint."""

    def __init__(self, wrapped: httpx.BaseTransport) -> None:
        self._wrapped = wrapped
        # Mark this instance so the global Client.__init__ patch doesn't
        # re-wrap it (which would cause double-emit).
        setattr(self, _TRACIUM_WRAPPED_FLAG, True)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if is_owned() or detect_provider(url) is None:
            return self._wrapped.handle_request(request)

        started_at = _utcnow()
        request_body = _read_request_body(request)

        try:
            response = self._wrapped.handle_request(request)
        except BaseException as e:
            _emit_for_error(url, request_body, e, started_at)
            raise

        if _looks_like_aws_eventstream(response):
            buffered = _peek_buffered_content(response)
            if buffered is not None:
                collector = _BedrockByteCollector()
                collector.feed(buffered)
                _emit_from_bedrock_eventstream(url, request_body, response, collector, started_at)
                return response

            collector = _BedrockByteCollector()
            response.stream = _CapturingByteStream(
                cast(httpx.SyncByteStream, response.stream),
                collector,
                on_close=lambda: _emit_from_bedrock_eventstream(
                    url, request_body, response, collector, started_at
                ),
            )
            return response

        if _looks_like_stream(response):
            buffered = _peek_buffered_content(response)
            if buffered is not None:
                # Some transports (notably ``httpx.MockTransport`` constructed
                # with ``content=`` or ``json=``) deliver SSE bodies fully
                # buffered. Run the bytes through the accumulator directly —
                # the user still iterates the response normally.
                accumulator = SSEAccumulator()
                try:
                    accumulator.feed(buffered)
                except Exception:
                    pass
                _emit_from_stream(url, request_body, response, accumulator, started_at)
                return response

            accumulator = SSEAccumulator()
            response.stream = _CapturingByteStream(
                cast(httpx.SyncByteStream, response.stream),
                accumulator,
                on_close=lambda: _emit_from_stream(
                    url, request_body, response, accumulator, started_at
                ),
            )
            return response

        # Non-streaming branch. If httpx has already buffered the body (the
        # common case — ``client.get()`` / ``client.post()`` read it before
        # returning), parse directly. Otherwise tee the stream so callers
        # using ``client.stream(...)`` on a non-SSE content-type still get
        # real streaming semantics.
        buffered = _peek_buffered_content(response)
        if buffered is not None:
            _emit_from_buffered(
                url, request_body, buffered, response.status_code, started_at
            )
            return response

        collector = _BytesAccumulator()
        response.stream = _CapturingByteStream(
            cast(httpx.SyncByteStream, response.stream),
            collector,
            on_close=_make_buffered_emit_cb(
                url, request_body, response, collector, started_at
            ),
        )
        return response

    def close(self) -> None:
        self._wrapped.close()


class TraciumAsyncHTTPXTransport(httpx.AsyncBaseTransport):
    """Async equivalent of :class:`TraciumHTTPXTransport`."""

    def __init__(self, wrapped: httpx.AsyncBaseTransport) -> None:
        self._wrapped = wrapped
        setattr(self, _TRACIUM_WRAPPED_FLAG, True)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if is_owned() or detect_provider(url) is None:
            return await self._wrapped.handle_async_request(request)

        started_at = _utcnow()
        request_body = _read_request_body(request)

        try:
            response = await self._wrapped.handle_async_request(request)
        except BaseException as e:
            _emit_for_error(url, request_body, e, started_at)
            raise

        if _looks_like_aws_eventstream(response):
            buffered = _peek_buffered_content(response)
            if buffered is not None:
                collector = _BedrockByteCollector()
                collector.feed(buffered)
                _emit_from_bedrock_eventstream(url, request_body, response, collector, started_at)
                return response

            collector = _BedrockByteCollector()
            response.stream = _CapturingAsyncByteStream(
                cast(httpx.AsyncByteStream, response.stream),
                collector,
                on_close=lambda: _emit_from_bedrock_eventstream(
                    url, request_body, response, collector, started_at
                ),
            )
            return response

        if _looks_like_stream(response):
            buffered = _peek_buffered_content(response)
            if buffered is not None:
                accumulator = SSEAccumulator()
                try:
                    accumulator.feed(buffered)
                except Exception:
                    pass
                _emit_from_stream(url, request_body, response, accumulator, started_at)
                return response

            accumulator = SSEAccumulator()
            response.stream = _CapturingAsyncByteStream(
                cast(httpx.AsyncByteStream, response.stream),
                accumulator,
                on_close=lambda: _emit_from_stream(
                    url, request_body, response, accumulator, started_at
                ),
            )
            return response

        # Non-streaming branch (async). Same logic as the sync path: prefer
        # the buffered body if httpx already has it, otherwise tee the stream
        # so ``AsyncClient.stream(...)`` callers retain streaming semantics.
        buffered = _peek_buffered_content(response)
        if buffered is not None:
            _emit_from_buffered(
                url, request_body, buffered, response.status_code, started_at
            )
            return response

        collector = _BytesAccumulator()
        response.stream = _CapturingAsyncByteStream(
            cast(httpx.AsyncByteStream, response.stream),
            collector,
            on_close=_make_buffered_emit_cb(
                url, request_body, response, collector, started_at
            ),
        )
        return response

    async def aclose(self) -> None:
        await self._wrapped.aclose()


# --------------------------------------------------------------------------- #
# Installation                                                                 #
# --------------------------------------------------------------------------- #


_INSTALLED = False
_ORIGINAL_SYNC_INIT: Any | None = None
_ORIGINAL_ASYNC_INIT: Any | None = None


def install() -> None:
    """Patch :class:`httpx.Client` and :class:`httpx.AsyncClient` so every new
    instance has its transport wrapped with our LLM-capturing transport.

    Idempotent and thread-safe. Existing client instances created before
    ``install()`` are not affected — that's a small price for not retroactively
    touching live state.
    """
    global _INSTALLED, _ORIGINAL_SYNC_INIT, _ORIGINAL_ASYNC_INIT
    # Double-checked locking: avoid taking the global PATCH_LOCK on the hot
    # path once we're already installed.
    if _INSTALLED:
        return

    from ...helpers.global_state import PATCH_LOCK

    with PATCH_LOCK:
        if _INSTALLED:
            return

        import functools

        orig_sync_init = httpx.Client.__init__
        orig_async_init = httpx.AsyncClient.__init__

        @functools.wraps(orig_sync_init)
        def _patched_sync_init(self: httpx.Client, *args: Any, **kwargs: Any) -> None:
            orig_sync_init(self, *args, **kwargs)
            _wrap_transports_sync(self)

        @functools.wraps(orig_async_init)
        def _patched_async_init(self: httpx.AsyncClient, *args: Any, **kwargs: Any) -> None:
            orig_async_init(self, *args, **kwargs)
            _wrap_transports_async(self)

        # Preserve the original signature so introspection (FastAPI TestClient,
        # pydantic-settings, openapi tooling) keeps seeing the real parameters.
        try:
            import inspect

            _patched_sync_init.__signature__ = inspect.signature(orig_sync_init)  # type: ignore[attr-defined]
            _patched_async_init.__signature__ = inspect.signature(orig_async_init)  # type: ignore[attr-defined]
        except (ValueError, TypeError):
            pass

        _ORIGINAL_SYNC_INIT = orig_sync_init
        _ORIGINAL_ASYNC_INIT = orig_async_init
        httpx.Client.__init__ = _patched_sync_init  # type: ignore[method-assign]
        httpx.AsyncClient.__init__ = _patched_async_init  # type: ignore[method-assign]

        _INSTALLED = True
    logger.info("tracium: http_capture installed for httpx (sync + async)")


def uninstall() -> None:
    """Reverse :func:`install`.

    Restores the original :class:`httpx.Client` / :class:`httpx.AsyncClient`
    constructors. Useful for test isolation and for users who want to
    re-enable capture later. Already-constructed clients keep their wrapped
    transports — we never reach back into live state.
    """
    global _INSTALLED, _ORIGINAL_SYNC_INIT, _ORIGINAL_ASYNC_INIT
    if not _INSTALLED:
        return

    from ...helpers.global_state import PATCH_LOCK

    with PATCH_LOCK:
        if not _INSTALLED:
            return
        if _ORIGINAL_SYNC_INIT is not None:
            httpx.Client.__init__ = _ORIGINAL_SYNC_INIT  # type: ignore[method-assign]
        if _ORIGINAL_ASYNC_INIT is not None:
            httpx.AsyncClient.__init__ = _ORIGINAL_ASYNC_INIT  # type: ignore[method-assign]
        _ORIGINAL_SYNC_INIT = None
        _ORIGINAL_ASYNC_INIT = None
        _INSTALLED = False


def _wrap_transports_sync(client: httpx.Client) -> None:
    transport = getattr(client, "_transport", None)
    if transport is not None and not _should_skip_transport(transport):
        wrapped = TraciumHTTPXTransport(transport)
        setattr(wrapped, _TRACIUM_WRAPPED_FLAG, True)
        client._transport = wrapped

    mounts = getattr(client, "_mounts", None)
    if isinstance(mounts, dict):
        for pattern, mount in list(mounts.items()):
            if _should_skip_transport(mount):
                continue
            wrapped = TraciumHTTPXTransport(mount)
            setattr(wrapped, _TRACIUM_WRAPPED_FLAG, True)
            mounts[pattern] = wrapped


def _wrap_transports_async(client: httpx.AsyncClient) -> None:
    transport = getattr(client, "_transport", None)
    if transport is not None and not _should_skip_transport(transport):
        wrapped = TraciumAsyncHTTPXTransport(transport)
        setattr(wrapped, _TRACIUM_WRAPPED_FLAG, True)
        client._transport = wrapped

    mounts = getattr(client, "_mounts", None)
    if isinstance(mounts, dict):
        for pattern, mount in list(mounts.items()):
            if _should_skip_transport(mount):
                continue
            wrapped = TraciumAsyncHTTPXTransport(mount)
            setattr(wrapped, _TRACIUM_WRAPPED_FLAG, True)
            mounts[pattern] = wrapped
