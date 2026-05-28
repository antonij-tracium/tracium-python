"""
OpenAI Realtime API (WebSocket) instrumentation.

The Realtime API isn't HTTP — it's a bidirectional WebSocket carrying JSON
events. The HTTP transport layer can't see it, so we hook the OpenAI SDK
directly: patch ``beta.realtime.connect()`` on both ``OpenAI`` and
``AsyncOpenAI`` clients so the returned context manager yields a wrapped
connection that tees server-sent events into a stateful tracker.

Each completed ``response.done`` event becomes one ``llm`` span on the active
trace, with the same shape the rest of the SDK emits:

* ``model``    — taken from session config or the response event itself
* ``input``    — system prompt + tools captured from ``session.created`` /
                 ``session.updated`` events (no audio bytes; transcripts only)
* ``output``   — concatenated text + audio transcripts + tool calls from the
                 response's output items
* ``input_tokens`` / ``output_tokens`` / ``cached_input_tokens`` — from
                 ``response.usage``

Outgoing events (client → server) are not intercepted in this version. We rely
on the server echoing state back (``session.updated`` for instructions/tools
updates, ``conversation.item.created`` for added items), which covers cost and
output attribution. Input *audio bytes* are intentionally not captured —
they're huge and the transcript already tells the story.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from ..context.trace_context import current_trace
from ..core import TraciumClient
from ..utils.datetime_utils import _utcnow
from .http_capture.dedup import owned_capture
from .http_capture.providers import LLMCall

logger = logging.getLogger(__name__)

_INSTALLED = False


def patch_openai_realtime(_client: TraciumClient) -> None:
    """Patch OpenAI's Realtime ``connect()`` on both sync and async clients.

    Idempotent. Best-effort: any failure is logged at debug and the original
    behavior is left untouched so the user's app keeps working.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    try:
        import openai  # noqa: F401
    except ImportError:
        return

    patched_any = False
    for module_path in (
        "openai.resources.beta.realtime.realtime",
        "openai.resources.beta.realtime",
    ):
        if _patch_module(module_path):
            patched_any = True

    if patched_any:
        _INSTALLED = True
        logger.info("tracium: instrumented openai.beta.realtime (WebSocket)")


def _patch_module(module_path: str) -> bool:
    try:
        module = __import__(module_path, fromlist=["*"])
    except Exception:
        return False

    patched = False
    for class_name in ("AsyncRealtime", "Realtime"):
        cls = getattr(module, class_name, None)
        if cls is None or getattr(cls, "_tracium_realtime_patched", False):
            continue
        original_connect = getattr(cls, "connect", None)
        if original_connect is None:
            continue

        is_async = class_name.startswith("Async")

        def make_patched(orig, async_):
            def patched_connect(self: Any, *args: Any, **kwargs: Any) -> Any:
                model_hint = kwargs.get("model") or (args[0] if args else None)
                cm = orig(self, *args, **kwargs)
                if async_:
                    return _TraciumAsyncRealtimeCM(cm, model_hint)
                return _TraciumSyncRealtimeCM(cm, model_hint)

            patched_connect.__wrapped__ = orig  # type: ignore[attr-defined]
            return patched_connect

        cls.connect = make_patched(original_connect, is_async)
        cls._tracium_realtime_patched = True
        patched = True

    return patched


# --------------------------------------------------------------------------- #
# Wrapped context managers + connections                                       #
# --------------------------------------------------------------------------- #


class _TraciumAsyncRealtimeCM:
    """Wraps the async CM returned by ``AsyncRealtime.connect()``."""

    def __init__(self, wrapped: Any, model_hint: str | None) -> None:
        self._wrapped = wrapped
        self._model_hint = model_hint

    async def __aenter__(self) -> Any:
        # Take ownership of the underlying WebSocket handshake so the HTTP
        # capture layer doesn't double-emit if it ever sees the GET upgrade.
        with owned_capture():
            conn = await self._wrapped.__aenter__()
        return _TraciumAsyncRealtimeConnection(conn, self._model_hint)

    async def __aexit__(self, exc_type, exc, tb) -> Any:
        return await self._wrapped.__aexit__(exc_type, exc, tb)


class _TraciumSyncRealtimeCM:
    """Wraps the sync CM returned by ``Realtime.connect()`` (if it exists)."""

    def __init__(self, wrapped: Any, model_hint: str | None) -> None:
        self._wrapped = wrapped
        self._model_hint = model_hint

    def __enter__(self) -> Any:
        with owned_capture():
            conn = self._wrapped.__enter__()
        return _TraciumSyncRealtimeConnection(conn, self._model_hint)

    def __exit__(self, exc_type, exc, tb) -> Any:
        return self._wrapped.__exit__(exc_type, exc, tb)


class _TraciumAsyncRealtimeConnection:
    """Delegates everything to the wrapped connection except ``__aiter__``,
    which tees server events into a :class:`_RealtimeTracker` before yielding
    them to the user.
    """

    def __init__(self, wrapped: Any, model_hint: str | None) -> None:
        # Bypass __setattr__ on the wrapper by using object.__setattr__ to set
        # the two attrs we manage; everything else delegates via __getattr__.
        object.__setattr__(self, "_wrapped", wrapped)
        object.__setattr__(self, "_tracker", _RealtimeTracker(model_hint))

    def __aiter__(self) -> Any:
        return self._iter()

    async def _iter(self):
        async for event in self._wrapped:
            try:
                self._tracker.on_event(event)
            except Exception as e:
                logger.debug(
                    "tracium realtime tracker on_event failed: %s: %s",
                    type(e).__name__,
                    e,
                )
            yield event

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


class _TraciumSyncRealtimeConnection:
    def __init__(self, wrapped: Any, model_hint: str | None) -> None:
        object.__setattr__(self, "_wrapped", wrapped)
        object.__setattr__(self, "_tracker", _RealtimeTracker(model_hint))

    def __iter__(self) -> Any:
        for event in self._wrapped:
            try:
                self._tracker.on_event(event)
            except Exception as e:
                logger.debug(
                    "tracium realtime tracker on_event failed: %s: %s",
                    type(e).__name__,
                    e,
                )
            yield event

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


# --------------------------------------------------------------------------- #
# Stateful per-session tracker                                                 #
# --------------------------------------------------------------------------- #


class _RealtimeTracker:
    """Accumulates per-session context and emits a span on every response.done.

    The Realtime API is *stateful*: ``session.created`` and ``session.updated``
    events carry the model, instructions, and tools that apply to all
    subsequent responses. ``response.created`` marks the start of one response
    cycle (which may consist of many delta events); ``response.done`` carries
    the complete response payload including final usage.

    We keep just enough state to attribute each response correctly without
    interfering with the user's iteration.
    """

    __slots__ = (
        "_model",
        "_instructions",
        "_tools",
        "_response_started_at",
    )

    def __init__(self, model_hint: str | None) -> None:
        self._model = model_hint
        self._instructions: str | None = None
        self._tools: list[dict[str, Any]] | None = None
        self._response_started_at: datetime | None = None

    def on_event(self, event: Any) -> None:
        et = _event_type(event)
        if et is None:
            return

        if et in ("session.created", "session.updated"):
            self._absorb_session(_event_attr(event, "session"))
        elif et == "response.created":
            self._response_started_at = _utcnow()
        elif et == "response.done":
            self._emit_response(_event_attr(event, "response"))
        elif et == "error":
            self._emit_error(_event_attr(event, "error"))

    # -- state absorption -------------------------------------------------- #

    def _absorb_session(self, session: Any) -> None:
        if session is None:
            return
        model = _event_attr(session, "model")
        if isinstance(model, str) and model:
            self._model = model
        instructions = _event_attr(session, "instructions")
        if isinstance(instructions, str):
            self._instructions = instructions
        tools = _event_attr(session, "tools")
        if isinstance(tools, list):
            self._tools = [d for d in (_as_dict(t) for t in tools) if d is not None]

    # -- span emission ----------------------------------------------------- #

    def _emit_response(self, response: Any) -> None:
        if current_trace() is None:
            return
        started_at = self._response_started_at or _utcnow()
        self._response_started_at = None

        usage = _as_dict(_event_attr(response, "usage")) or {}
        model = _event_attr(response, "model") or self._model
        output_items = _event_attr(response, "output") or []

        text_parts, tool_calls = _extract_response_output(output_items)

        call = LLMCall(
            provider="openai",
            operation="realtime",
            model=model,
            input=self._build_input(),
            output="".join(text_parts) if text_parts else None,
            tools=self._tools,
            tool_calls=tool_calls or None,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            cached_input_tokens=(
                (usage.get("input_token_details") or {}).get("cached_tokens")
                if isinstance(usage.get("input_token_details"), dict)
                else None
            ),
        )

        status = _event_attr(response, "status")
        if isinstance(status, str) and status not in ("completed", "in_progress"):
            details = _as_dict(_event_attr(response, "status_details")) or {}
            err_msg = details.get("error") or details.get("reason") or status
            call.error = str(err_msg)[:1000]

        from .http_capture.emit import emit_llm_span

        emit_llm_span(call, started_at)

    def _emit_error(self, error: Any) -> None:
        if current_trace() is None:
            return
        error_dict = _as_dict(error) or {}
        msg = error_dict.get("message") or error_dict.get("type") or "realtime error"
        call = LLMCall(
            provider="openai",
            operation="realtime",
            model=self._model,
            input=self._build_input(),
            error=str(msg)[:1000],
        )
        from .http_capture.emit import emit_llm_span

        emit_llm_span(call, self._response_started_at or _utcnow())

    def _build_input(self) -> Any:
        if self._instructions or self._tools:
            return {"system": self._instructions, "tools": self._tools}
        return None


# --------------------------------------------------------------------------- #
# Helpers — duck-type SDK event objects (Pydantic models or dicts)            #
# --------------------------------------------------------------------------- #


def _event_type(event: Any) -> str | None:
    return _event_attr(event, "type") if event is not None else None


def _event_attr(event: Any, name: str) -> Any:
    if event is None:
        return None
    # Pydantic v2 attr access; falls back to dict get.
    if isinstance(event, dict):
        return event.get(name)
    return getattr(event, name, None)


def _as_dict(obj: Any) -> dict[str, Any] | None:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    # Pydantic v2: ``.model_dump()``; older: ``.dict()``; otherwise vars().
    for method in ("model_dump", "dict"):
        fn = getattr(obj, method, None)
        if callable(fn):
            try:
                result = fn()
                if isinstance(result, dict):
                    return result
            except Exception:
                pass
    try:
        return dict(vars(obj))
    except Exception:
        return None


def _extract_response_output(
    output_items: Any,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Walk the response's ``output[]`` array, pulling out text + tool calls.

    The Realtime ``response.output`` is heterogeneous:
      * ``message`` items with ``content[]`` of ``text``/``audio`` parts (the
        audio part carries a ``transcript`` string we can use)
      * ``function_call`` items with ``name`` and ``arguments``
    """
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    if not isinstance(output_items, list | tuple):
        return text_parts, tool_calls

    for item in output_items:
        d = _as_dict(item) or {}
        kind = d.get("type")
        if kind == "function_call":
            tool_calls.append(
                {
                    "id": d.get("call_id") or d.get("id"),
                    "type": "function",
                    "function": {
                        "name": d.get("name"),
                        "arguments": d.get("arguments", ""),
                    },
                }
            )
            continue
        # ``message`` items — concatenate text / audio transcripts.
        for part in d.get("content") or []:
            part_dict = _as_dict(part) or {}
            pkind = part_dict.get("type")
            if pkind in ("text", "input_text", "output_text") and isinstance(
                part_dict.get("text"), str
            ):
                text_parts.append(part_dict["text"])
            elif pkind in ("audio", "input_audio") and isinstance(
                part_dict.get("transcript"), str
            ):
                text_parts.append(part_dict["transcript"])

    return text_parts, tool_calls
