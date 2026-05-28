"""
Bedrock event-stream reconstruction.

Bedrock has two streaming variants and each emits a different inner-event shape:

* ``/model/{id}/invoke-with-response-stream`` — events are the *per-model native*
  format (e.g. Anthropic Claude native events under ``"type": "content_block_delta"``)
  base64-wrapped inside the eventstream framing. Reconstructs to the non-streaming
  ``invoke`` response shape — the same dict ``_parse_bedrock_invoke`` already handles.

* ``/model/{id}/converse-stream`` — events are Bedrock's *unified* camelCase
  shape (``messageStart``, ``contentBlockDelta``, ``metadata``, …). Reconstructs to
  the non-streaming ``converse`` response shape — handled by ``_parse_bedrock_converse``.

We pick the right reconstructor by URL path: anything matching ``/converse-stream``
goes through :class:`_ConverseAssembler`; everything else (the legacy invoke
path) goes through :class:`_InvokeAssembler` which dispatches further based on
the model ID inside the URL.
"""

from __future__ import annotations

import json
from typing import Any

from .aws_eventstream import iter_bedrock_events


def reconstruct_response(url_path: str, model: str | None, buffer: bytes) -> dict[str, Any]:
    """Reconstruct a non-streaming Bedrock response dict from buffered bytes.

    Returns the response shape that :func:`parse_bedrock` expects for the
    matching non-streaming endpoint. Always returns a dict — empty if nothing
    decoded — so the downstream parser can safely call ``.get`` on it.
    """
    if "/converse-stream" in url_path:
        assembler: _Assembler = _ConverseAssembler()
    else:
        assembler = _InvokeAssembler(model or "")
    for event in iter_bedrock_events(buffer):
        try:
            assembler.feed(event)
        except Exception:
            # One bad event must not throw out the rest of the stream.
            continue
    return assembler.finalize()


# --------------------------------------------------------------------------- #
# Converse-stream                                                              #
# --------------------------------------------------------------------------- #


class _Assembler:
    def feed(self, event: dict[str, Any]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def finalize(self) -> dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError


class _ConverseAssembler(_Assembler):
    """Assemble Bedrock Converse stream events into a non-streaming response.

    Handles ``messageStart``, ``contentBlockStart``, ``contentBlockDelta``,
    ``contentBlockStop``, ``messageStop``, ``metadata``. Other event types are
    ignored.
    """

    def __init__(self) -> None:
        self._role: str | None = None
        self._content_blocks: dict[int, dict[str, Any]] = {}
        self._text_buffers: dict[int, list[str]] = {}
        self._tool_inputs: dict[int, list[str]] = {}
        self._stop_reason: str | None = None
        self._usage: dict[str, Any] = {}
        self._metrics: dict[str, Any] = {}

    def feed(self, event: dict[str, Any]) -> None:
        # Each Converse event has exactly one key naming the variant.
        for key in (
            "messageStart",
            "contentBlockStart",
            "contentBlockDelta",
            "contentBlockStop",
            "messageStop",
            "metadata",
        ):
            inner = event.get(key)
            if inner is None:
                continue
            getattr(self, f"_on_{key}")(inner)

    # -- per-event handlers --------------------------------------------- #
    # Method names intentionally mirror Bedrock's camelCase event keys so the
    # ``feed`` dispatcher loop stays a one-liner. Suppress N802 here only.

    def _on_messageStart(self, payload: dict[str, Any]) -> None:  # noqa: N802
        self._role = payload.get("role") or self._role

    def _on_contentBlockStart(self, payload: dict[str, Any]) -> None:  # noqa: N802
        idx = int(payload.get("contentBlockIndex") or 0)
        start = payload.get("start") or {}
        tool_use = start.get("toolUse") if isinstance(start, dict) else None
        if isinstance(tool_use, dict):
            self._content_blocks[idx] = {
                "toolUse": {
                    "toolUseId": tool_use.get("toolUseId"),
                    "name": tool_use.get("name"),
                    "input": {},
                }
            }
            self._tool_inputs[idx] = []

    def _on_contentBlockDelta(self, payload: dict[str, Any]) -> None:  # noqa: N802
        idx = int(payload.get("contentBlockIndex") or 0)
        delta = payload.get("delta") or {}
        if not isinstance(delta, dict):
            return
        if isinstance(delta.get("text"), str):
            self._text_buffers.setdefault(idx, []).append(delta["text"])
        elif isinstance(delta.get("toolUse"), dict):
            tu = delta["toolUse"]
            if isinstance(tu.get("input"), str):
                self._tool_inputs.setdefault(idx, []).append(tu["input"])

    def _on_contentBlockStop(self, payload: dict[str, Any]) -> None:  # noqa: N802
        idx = int(payload.get("contentBlockIndex") or 0)
        # Finalize tool-use input — Bedrock streams it as concatenated JSON.
        if idx in self._tool_inputs:
            joined = "".join(self._tool_inputs[idx])
            if joined:
                try:
                    parsed = json.loads(joined)
                except Exception:
                    parsed = joined
                block = self._content_blocks.setdefault(idx, {"toolUse": {}})
                block.setdefault("toolUse", {})["input"] = parsed

    def _on_messageStop(self, payload: dict[str, Any]) -> None:  # noqa: N802
        if isinstance(payload.get("stopReason"), str):
            self._stop_reason = payload["stopReason"]

    def _on_metadata(self, payload: dict[str, Any]) -> None:
        usage = payload.get("usage")
        if isinstance(usage, dict):
            self._usage.update(usage)
        metrics = payload.get("metrics")
        if isinstance(metrics, dict):
            self._metrics.update(metrics)

    # -- final --------------------------------------------------------- #

    def finalize(self) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        # Preserve content-block order by index.
        indices = sorted(set(self._text_buffers.keys()) | set(self._content_blocks.keys()))
        for idx in indices:
            text_parts = self._text_buffers.get(idx)
            if text_parts:
                content.append({"text": "".join(text_parts)})
            tool_block = self._content_blocks.get(idx)
            if tool_block:
                content.append(tool_block)
        result: dict[str, Any] = {
            "output": {"message": {"role": self._role or "assistant", "content": content}},
        }
        if self._stop_reason:
            result["stopReason"] = self._stop_reason
        if self._usage:
            result["usage"] = self._usage
        if self._metrics:
            result["metrics"] = self._metrics
        return result


# --------------------------------------------------------------------------- #
# InvokeWithResponseStream — dispatches to per-model assembly                 #
# --------------------------------------------------------------------------- #


class _InvokeAssembler(_Assembler):
    """Assemble per-model invoke-with-response-stream events.

    The event shape varies by model: Anthropic Claude emits SSE-style events
    (``type: content_block_delta`` etc.), Llama emits a ``generation`` chunk
    per token, Titan emits ``outputText`` chunks, and so on. We dispatch by
    inspecting the model ID and route events to the right specialized
    assembler.
    """

    def __init__(self, model: str) -> None:
        model_lower = model.lower()
        self._inner: _Assembler
        if "claude" in model_lower or "anthropic" in model_lower:
            self._inner = _ClaudeInvokeAssembler()
        elif "llama" in model_lower or "meta" in model_lower:
            self._inner = _LlamaInvokeAssembler()
        elif "titan" in model_lower or "amazon." in model_lower:
            self._inner = _TitanInvokeAssembler()
        elif "mistral" in model_lower:
            self._inner = _MistralInvokeAssembler()
        elif "cohere" in model_lower:
            self._inner = _CohereInvokeAssembler()
        else:
            self._inner = _ClaudeInvokeAssembler()  # most likely guess

    def feed(self, event: dict[str, Any]) -> None:
        self._inner.feed(event)

    def finalize(self) -> dict[str, Any]:
        return self._inner.finalize()


class _ClaudeInvokeAssembler(_Assembler):
    """Reconstruct an Anthropic Messages response from the streamed events.

    Events match the Anthropic native SSE shape (``type: content_block_delta``,
    ``type: message_start``, etc.), so this mirrors the Anthropic parser's
    expectations.
    """

    def __init__(self) -> None:
        self._text: list[str] = []
        self._tool_uses: dict[int, dict[str, Any]] = {}
        self._tool_inputs: dict[int, list[str]] = {}
        self._usage: dict[str, Any] = {}
        self._stop_reason: str | None = None
        self._model: str | None = None
        self._role: str | None = None

    def feed(self, event: dict[str, Any]) -> None:
        ev_type = event.get("type")
        if ev_type == "message_start":
            msg = event.get("message") or {}
            if isinstance(msg.get("usage"), dict):
                self._usage.update(msg["usage"])
            if isinstance(msg.get("model"), str):
                self._model = msg["model"]
            if isinstance(msg.get("role"), str):
                self._role = msg["role"]
        elif ev_type == "content_block_start":
            idx = int(event.get("index") or 0)
            block = event.get("content_block") or {}
            if block.get("type") == "tool_use":
                self._tool_uses[idx] = {
                    "type": "tool_use",
                    "id": block.get("id"),
                    "name": block.get("name"),
                    "input": {},
                }
                self._tool_inputs[idx] = []
        elif ev_type == "content_block_delta":
            idx = int(event.get("index") or 0)
            delta = event.get("delta") or {}
            if isinstance(delta.get("text"), str):
                self._text.append(delta["text"])
            elif isinstance(delta.get("partial_json"), str):
                self._tool_inputs.setdefault(idx, []).append(delta["partial_json"])
        elif ev_type == "content_block_stop":
            idx = int(event.get("index") or 0)
            if idx in self._tool_uses and idx in self._tool_inputs:
                joined = "".join(self._tool_inputs[idx])
                try:
                    self._tool_uses[idx]["input"] = json.loads(joined) if joined else {}
                except Exception:
                    self._tool_uses[idx]["input"] = joined
        elif ev_type == "message_delta":
            usage = event.get("usage")
            if isinstance(usage, dict):
                self._usage.update(usage)
            delta = event.get("delta") or {}
            if isinstance(delta.get("stop_reason"), str):
                self._stop_reason = delta["stop_reason"]
        elif ev_type == "message_stop":
            # Bedrock sometimes attaches usage to message_stop.
            metrics = event.get("amazon-bedrock-invocationMetrics")
            if isinstance(metrics, dict):
                if "inputTokenCount" in metrics:
                    self._usage["input_tokens"] = metrics["inputTokenCount"]
                if "outputTokenCount" in metrics:
                    self._usage["output_tokens"] = metrics["outputTokenCount"]

    def finalize(self) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        if self._text:
            content.append({"type": "text", "text": "".join(self._text)})
        for idx in sorted(self._tool_uses.keys()):
            content.append(self._tool_uses[idx])
        result: dict[str, Any] = {"content": content}
        if self._model:
            result["model"] = self._model
        if self._role:
            result["role"] = self._role
        if self._usage:
            result["usage"] = self._usage
        if self._stop_reason:
            result["stop_reason"] = self._stop_reason
        return result


class _LlamaInvokeAssembler(_Assembler):
    def __init__(self) -> None:
        self._text: list[str] = []
        self._prompt_tokens: int | None = None
        self._generation_tokens: int | None = None
        self._stop_reason: str | None = None

    def feed(self, event: dict[str, Any]) -> None:
        if isinstance(event.get("generation"), str):
            self._text.append(event["generation"])
        if isinstance(event.get("prompt_token_count"), int):
            self._prompt_tokens = event["prompt_token_count"]
        if isinstance(event.get("generation_token_count"), int):
            self._generation_tokens = (self._generation_tokens or 0) + event[
                "generation_token_count"
            ]
        if isinstance(event.get("stop_reason"), str):
            self._stop_reason = event["stop_reason"]

    def finalize(self) -> dict[str, Any]:
        out: dict[str, Any] = {"generation": "".join(self._text)}
        if self._prompt_tokens is not None:
            out["prompt_token_count"] = self._prompt_tokens
        if self._generation_tokens is not None:
            out["generation_token_count"] = self._generation_tokens
        if self._stop_reason:
            out["stop_reason"] = self._stop_reason
        return out


class _TitanInvokeAssembler(_Assembler):
    def __init__(self) -> None:
        self._text: list[str] = []
        self._input_tokens: int | None = None
        self._output_tokens: int | None = None
        self._completion_reason: str | None = None

    def feed(self, event: dict[str, Any]) -> None:
        if isinstance(event.get("outputText"), str):
            self._text.append(event["outputText"])
        if isinstance(event.get("inputTextTokenCount"), int):
            self._input_tokens = event["inputTextTokenCount"]
        if isinstance(event.get("totalOutputTextTokenCount"), int):
            self._output_tokens = event["totalOutputTextTokenCount"]
        elif isinstance(event.get("tokenCount"), int):
            self._output_tokens = (self._output_tokens or 0) + event["tokenCount"]
        if isinstance(event.get("completionReason"), str):
            self._completion_reason = event["completionReason"]

    def finalize(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "results": [
                {
                    "outputText": "".join(self._text),
                    "tokenCount": self._output_tokens,
                    "completionReason": self._completion_reason,
                }
            ],
        }
        if self._input_tokens is not None:
            result["inputTextTokenCount"] = self._input_tokens
        return result


class _MistralInvokeAssembler(_Assembler):
    def __init__(self) -> None:
        self._text: list[str] = []
        self._prompt_tokens: int | None = None
        self._completion_tokens: int | None = None

    def feed(self, event: dict[str, Any]) -> None:
        outputs = event.get("outputs")
        if isinstance(outputs, list):
            for o in outputs:
                if isinstance(o, dict) and isinstance(o.get("text"), str):
                    self._text.append(o["text"])
        usage = event.get("usage")
        if isinstance(usage, dict):
            if isinstance(usage.get("prompt_tokens"), int):
                self._prompt_tokens = usage["prompt_tokens"]
            if isinstance(usage.get("completion_tokens"), int):
                self._completion_tokens = usage["completion_tokens"]

    def finalize(self) -> dict[str, Any]:
        result: dict[str, Any] = {"outputs": [{"text": "".join(self._text)}]}
        usage: dict[str, Any] = {}
        if self._prompt_tokens is not None:
            usage["prompt_tokens"] = self._prompt_tokens
        if self._completion_tokens is not None:
            usage["completion_tokens"] = self._completion_tokens
        if usage:
            result["usage"] = usage
        return result


class _CohereInvokeAssembler(_Assembler):
    def __init__(self) -> None:
        self._text: list[str] = []
        self._prompt_tokens: int | None = None
        self._generation_tokens: int | None = None

    def feed(self, event: dict[str, Any]) -> None:
        gens = event.get("generations")
        if isinstance(gens, list):
            for g in gens:
                if isinstance(g, dict) and isinstance(g.get("text"), str):
                    self._text.append(g["text"])
        if isinstance(event.get("prompt_tokens"), int):
            self._prompt_tokens = event["prompt_tokens"]
        if isinstance(event.get("generation_tokens"), int):
            self._generation_tokens = (self._generation_tokens or 0) + event["generation_tokens"]

    def finalize(self) -> dict[str, Any]:
        out: dict[str, Any] = {"generations": [{"text": "".join(self._text)}]}
        if self._prompt_tokens is not None:
            out["prompt_tokens"] = self._prompt_tokens
        if self._generation_tokens is not None:
            out["generation_tokens"] = self._generation_tokens
        return out
