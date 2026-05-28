"""
Server-Sent Events accumulator for LLM streaming responses.

Modern LLM HTTP APIs stream responses as ``text/event-stream``: a sequence of
``data: <json>`` lines terminated by a blank line. Each provider has its own chunk
shape but they all converge on "concatenate text deltas, take usage from the
final chunk."

We accumulate enough state to reconstruct what the non-streaming response would
have looked like, so the same provider parsers can run over the result. Format
detection is best-effort; unknown shapes still produce a partial reconstruction.
"""

from __future__ import annotations

import json
from typing import Any

_DONE_MARKERS = {"[DONE]", "DONE"}


def _safe_json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj)
    except Exception:
        return ""


class SSEAccumulator:
    """Stateful accumulator for an LLM SSE stream.

    Feed every chunk of bytes via :meth:`feed`; call :meth:`finalize` when the
    stream ends to get a dict shaped like the equivalent non-streaming response.
    """

    __slots__ = ("_buffer", "_text", "_tool_calls", "_usage", "_model", "_role", "_finish_reason")

    def __init__(self) -> None:
        self._buffer: bytes = b""
        self._text: list[str] = []
        self._tool_calls: dict[int, dict[str, Any]] = {}
        self._usage: dict[str, Any] = {}
        self._model: str | None = None
        self._role: str | None = None
        self._finish_reason: str | None = None

    def feed(self, chunk: bytes) -> None:
        """Append bytes to the internal buffer and process complete events."""
        if not chunk:
            return
        self._buffer += chunk
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            self._handle_line(line.rstrip(b"\r"))

    def finalize(self) -> dict[str, Any]:
        """Return a non-streaming-shaped dict reconstructed from the chunks."""
        if self._buffer:
            self._handle_line(self._buffer.rstrip(b"\r"))
            self._buffer = b""

        message: dict[str, Any] = {"role": self._role or "assistant", "content": "".join(self._text)}
        if self._tool_calls:
            message["tool_calls"] = [
                self._tool_calls[k] for k in sorted(self._tool_calls.keys())
            ]
        result: dict[str, Any] = {
            "choices": [{"message": message, "finish_reason": self._finish_reason}],
        }
        if self._model:
            result["model"] = self._model
        if self._usage:
            result["usage"] = self._usage
        return result


    def _handle_line(self, line: bytes) -> None:
        if not line or line.startswith(b":"):  # heartbeat / comment
            return
        if not line.startswith(b"data:"):
            return
        payload = line[5:].strip()
        if not payload:
            return
        if payload.decode("utf-8", errors="replace").strip() in _DONE_MARKERS:
            return
        try:
            event = json.loads(payload)
        except Exception:
            return
        if isinstance(event, dict):
            self._merge_event(event)

    def _merge_event(self, event: dict[str, Any]) -> None:
        if isinstance(event.get("choices"), list) and event["choices"]:
            choice = event["choices"][0] or {}
            delta = choice.get("delta") or choice.get("message") or {}
            if isinstance(delta, dict):
                if "role" in delta and not self._role:
                    self._role = delta.get("role")
                content = delta.get("content")
                if isinstance(content, str):
                    self._text.append(content)
                elif isinstance(content, list):
                    # Anthropic-on-OpenAI compat: list of content blocks
                    for block in content:
                        if isinstance(block, dict) and isinstance(block.get("text"), str):
                            self._text.append(block["text"])
                if isinstance(delta.get("tool_calls"), list):
                    for tc in delta["tool_calls"]:
                        self._merge_tool_call(tc)
            if choice.get("finish_reason"):
                self._finish_reason = choice["finish_reason"]

        # Google Gemini streamGenerateContent (alt=sse): top-level ``candidates``.
        if isinstance(event.get("candidates"), list) and event["candidates"]:
            for cand in event["candidates"]:
                if not isinstance(cand, dict):
                    continue
                parts = (cand.get("content") or {}).get("parts")
                if isinstance(parts, list):
                    for p in parts:
                        if not isinstance(p, dict):
                            continue
                        if isinstance(p.get("text"), str):
                            self._text.append(p["text"])
                        if isinstance(p.get("functionCall"), dict):
                            idx = len(self._tool_calls)
                            fc = p["functionCall"]
                            self._tool_calls[idx] = {
                                "id": fc.get("id"),
                                "type": "function",
                                "function": {
                                    "name": fc.get("name"),
                                    "arguments": _safe_json_dumps(fc.get("args") or {}),
                                },
                            }
                if cand.get("finishReason"):
                    self._finish_reason = cand["finishReason"]
            usage_md = event.get("usageMetadata")
            if isinstance(usage_md, dict):
                # Normalize Gemini names so the OpenAI-compatible parser picks them up.
                if usage_md.get("promptTokenCount") is not None:
                    self._usage["prompt_tokens"] = usage_md["promptTokenCount"]
                if usage_md.get("candidatesTokenCount") is not None:
                    self._usage["completion_tokens"] = usage_md["candidatesTokenCount"]
                if usage_md.get("cachedContentTokenCount") is not None:
                    self._usage["cached_tokens"] = usage_md["cachedContentTokenCount"]

        # Anthropic-style: {"type": "content_block_delta", "delta": {"text": ...}}
        ev_type = event.get("type")
        if ev_type == "content_block_delta":
            delta = event.get("delta") or {}
            if isinstance(delta.get("text"), str):
                self._text.append(delta["text"])
            elif isinstance(delta.get("partial_json"), str):
                idx = event.get("index", 0)
                tc = self._tool_calls.setdefault(idx, {"function": {"arguments": ""}})
                tc["function"]["arguments"] += delta["partial_json"]
        elif ev_type == "content_block_start":
            block = event.get("content_block") or {}
            if block.get("type") == "tool_use":
                idx = event.get("index", len(self._tool_calls))
                self._tool_calls[idx] = {
                    "id": block.get("id"),
                    "type": "function",
                    "function": {"name": block.get("name"), "arguments": ""},
                }
        elif ev_type == "message_start":
            msg = event.get("message") or {}
            if isinstance(msg.get("usage"), dict):
                self._usage.update(msg["usage"])
            if msg.get("model") and not self._model:
                self._model = msg["model"]
        elif ev_type == "message_delta":
            usage = event.get("usage")
            if isinstance(usage, dict):
                self._usage.update(usage)

        if "model" in event and not self._model:
            self._model = event.get("model")
        if isinstance(event.get("usage"), dict):
            self._usage.update(event["usage"])

    def _merge_tool_call(self, tc: dict[str, Any]) -> None:
        idx = tc.get("index", 0)
        existing = self._tool_calls.setdefault(idx, {"function": {"arguments": ""}})
        if "id" in tc:
            existing["id"] = tc["id"]
        if "type" in tc:
            existing["type"] = tc["type"]
        fn = tc.get("function") or {}
        if isinstance(fn, dict):
            fn_target = existing.setdefault("function", {})
            if "name" in fn:
                fn_target["name"] = fn["name"]
            if isinstance(fn.get("arguments"), str):
                fn_target["arguments"] = fn_target.get("arguments", "") + fn["arguments"]
