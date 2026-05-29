"""
Tests for AWS event-stream parsing and Bedrock streaming reconstruction.

We never call the real AWS API — instead we encode realistic event-stream
binary frames in the test helper :func:`encode_eventstream_message` and feed
them through the public parser. The encoded shape matches what Bedrock
actually returns: each message is a JSON payload (with ``{"bytes": "<base64
inner JSON>"}``) wrapped in AWS's framing.
"""

from __future__ import annotations

import base64
import json
import struct
from typing import Any

import httpx
import pytest

from tracium.integrations.http_capture.aws_eventstream import iter_bedrock_events
from tracium.integrations.http_capture.bedrock_stream import reconstruct_response
from tracium.integrations.http_capture.providers import parse
from tracium.integrations.http_capture.transport_httpx import TraciumHTTPXTransport

# --------------------------------------------------------------------------- #
# Test helpers — encode AWS event-stream binary frames                        #
# --------------------------------------------------------------------------- #


def encode_eventstream_message(headers: dict[str, str], payload: bytes) -> bytes:
    """Encode a single AWS event-stream message.

    Only string-typed headers (value-type 7) are emitted, which matches what
    Bedrock actually sends. CRCs are zeroed — we don't validate them.
    """
    header_blob = bytearray()
    for name, value in headers.items():
        name_bytes = name.encode("utf-8")
        value_bytes = value.encode("utf-8")
        header_blob.append(len(name_bytes))
        header_blob.extend(name_bytes)
        header_blob.append(7)  # string type
        header_blob.extend(struct.pack(">H", len(value_bytes)))
        header_blob.extend(value_bytes)

    headers_length = len(header_blob)
    total_length = 12 + headers_length + len(payload) + 4
    prelude = struct.pack(">II", total_length, headers_length) + b"\x00\x00\x00\x00"
    trailer = b"\x00\x00\x00\x00"
    return prelude + bytes(header_blob) + payload + trailer


def chunk_event(inner_json: dict[str, Any]) -> bytes:
    """Build a Bedrock-style chunk event: outer wraps base64(inner JSON)."""
    inner_b64 = base64.b64encode(json.dumps(inner_json).encode("utf-8")).decode("ascii")
    outer = json.dumps({"bytes": inner_b64}).encode("utf-8")
    return encode_eventstream_message({":message-type": "event", ":event-type": "chunk"}, outer)


def converse_event(inner_json: dict[str, Any]) -> bytes:
    """Bedrock Converse-stream events are also wrapped chunks — same shape."""
    return chunk_event(inner_json)


# --------------------------------------------------------------------------- #
# Low-level framing parser                                                     #
# --------------------------------------------------------------------------- #


class TestEventStreamFraming:
    def test_single_message_decoded(self) -> None:
        body = chunk_event({"type": "text", "text": "hi"})
        events = list(iter_bedrock_events(body))
        assert events == [{"type": "text", "text": "hi"}]

    def test_multiple_messages_decoded_in_order(self) -> None:
        body = chunk_event({"n": 1}) + chunk_event({"n": 2}) + chunk_event({"n": 3})
        events = list(iter_bedrock_events(body))
        assert [e["n"] for e in events] == [1, 2, 3]

    def test_malformed_message_is_skipped(self) -> None:
        # Junk bytes followed by a good message — parser should bail safely.
        body = b"\x00\x00\x00\x05junk" + chunk_event({"ok": True})
        events = list(iter_bedrock_events(body))
        # Either skipped junk and got the good one, or aborted early — both are
        # acceptable; we just want no exception.
        assert events == [] or events == [{"ok": True}]

    def test_empty_buffer_yields_nothing(self) -> None:
        assert list(iter_bedrock_events(b"")) == []


# --------------------------------------------------------------------------- #
# Bedrock Converse-stream reconstruction                                       #
# --------------------------------------------------------------------------- #


class TestConverseStreamReconstruction:
    PATH = "/model/anthropic.claude-3-5-sonnet-20240620-v1:0/converse-stream"

    def test_text_message(self) -> None:
        body = (
            converse_event({"messageStart": {"role": "assistant"}})
            + converse_event(
                {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Hi "}}}
            )
            + converse_event(
                {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "there"}}}
            )
            + converse_event({"contentBlockStop": {"contentBlockIndex": 0}})
            + converse_event({"messageStop": {"stopReason": "end_turn"}})
            + converse_event({"metadata": {"usage": {"inputTokens": 5, "outputTokens": 2}}})
        )
        recon = reconstruct_response(self.PATH, "anthropic.claude-3-5-sonnet", body)
        assert recon["output"]["message"]["role"] == "assistant"
        assert recon["output"]["message"]["content"][0] == {"text": "Hi there"}
        assert recon["stopReason"] == "end_turn"
        assert recon["usage"] == {"inputTokens": 5, "outputTokens": 2}

    def test_tool_use_streaming(self) -> None:
        body = (
            converse_event({"messageStart": {"role": "assistant"}})
            + converse_event(
                {
                    "contentBlockStart": {
                        "contentBlockIndex": 0,
                        "start": {"toolUse": {"toolUseId": "tu_1", "name": "search"}},
                    }
                }
            )
            + converse_event(
                {
                    "contentBlockDelta": {
                        "contentBlockIndex": 0,
                        "delta": {"toolUse": {"input": '{"q":"x"}'}},
                    }
                }
            )
            + converse_event({"contentBlockStop": {"contentBlockIndex": 0}})
            + converse_event({"messageStop": {"stopReason": "tool_use"}})
        )
        recon = reconstruct_response(self.PATH, "anthropic.claude-3-5-sonnet", body)
        content = recon["output"]["message"]["content"]
        tool_block = next(c for c in content if "toolUse" in c)
        assert tool_block["toolUse"]["name"] == "search"
        assert tool_block["toolUse"]["input"] == {"q": "x"}


# --------------------------------------------------------------------------- #
# Bedrock invoke-with-response-stream (per-model)                              #
# --------------------------------------------------------------------------- #


class TestInvokeStreamClaude:
    PATH = "/model/anthropic.claude-3-5-sonnet-20240620-v1:0/invoke-with-response-stream"
    MODEL = "anthropic.claude-3-5-sonnet-20240620-v1:0"

    def test_text_and_usage(self) -> None:
        body = (
            chunk_event(
                {
                    "type": "message_start",
                    "message": {"model": "claude-3-5-sonnet", "usage": {"input_tokens": 8}},
                }
            )
            + chunk_event(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                }
            )
            + chunk_event(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "Hello "},
                }
            )
            + chunk_event(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "world"},
                }
            )
            + chunk_event({"type": "message_delta", "usage": {"output_tokens": 3}})
            + chunk_event({"type": "message_stop"})
        )
        recon = reconstruct_response(self.PATH, self.MODEL, body)
        # Now feed through the high-level parser as the transport would.
        call = parse(
            "https://bedrock-runtime.us-east-1.amazonaws.com" + self.PATH,
            b"",
            recon,
            200,
        )
        assert call is not None
        assert call.provider == "bedrock"
        assert call.model == self.MODEL
        assert call.output == "Hello world"
        assert call.input_tokens == 8
        assert call.output_tokens == 3


class TestInvokeStreamLlama:
    PATH = "/model/meta.llama3-70b-instruct-v1:0/invoke-with-response-stream"
    MODEL = "meta.llama3-70b-instruct-v1:0"

    def test_generation_concat(self) -> None:
        body = (
            chunk_event({"generation": "Hi ", "prompt_token_count": 4, "generation_token_count": 1})
            + chunk_event({"generation": "there", "generation_token_count": 1})
            + chunk_event({"stop_reason": "stop", "generation_token_count": 1})
        )
        recon = reconstruct_response(self.PATH, self.MODEL, body)
        call = parse(
            "https://bedrock-runtime.us-east-1.amazonaws.com" + self.PATH,
            b"",
            recon,
            200,
        )
        assert call.output == "Hi there"
        assert call.input_tokens == 4
        assert call.output_tokens == 3
        assert call.model == self.MODEL


class TestInvokeStreamTitan:
    PATH = "/model/amazon.titan-text-express-v1/invoke-with-response-stream"
    MODEL = "amazon.titan-text-express-v1"

    def test_output_text_concat(self) -> None:
        body = chunk_event(
            {"outputText": "Hi ", "inputTextTokenCount": 3, "tokenCount": 1}
        ) + chunk_event({"outputText": "there", "tokenCount": 1, "completionReason": "FINISH"})
        recon = reconstruct_response(self.PATH, self.MODEL, body)
        call = parse(
            "https://bedrock-runtime.us-east-1.amazonaws.com" + self.PATH,
            b"",
            recon,
            200,
        )
        assert call.output == "Hi there"
        assert call.input_tokens == 3
        assert call.output_tokens == 2


# --------------------------------------------------------------------------- #
# End-to-end through the httpx transport                                       #
# --------------------------------------------------------------------------- #


class TestTransportEndToEnd:
    """Send a fake Bedrock streaming response through the transport and assert
    that a single LLM span is emitted with the reconstructed content.
    """

    def test_converse_stream_emits_span(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[Any] = []

        def fake_emit(call, started_at, ended_at=None):  # noqa: ARG001
            captured.append(call)

        from tracium.integrations.http_capture import (
            emit as emit_mod,
        )
        from tracium.integrations.http_capture import (
            transport_httpx,
        )

        monkeypatch.setattr(emit_mod, "emit_llm_span", fake_emit)
        monkeypatch.setattr(transport_httpx, "emit_llm_span", fake_emit)

        url = (
            "https://bedrock-runtime.us-east-1.amazonaws.com/"
            "model/anthropic.claude-3-5-sonnet-20240620-v1:0/converse-stream"
        )
        body = (
            converse_event({"messageStart": {"role": "assistant"}})
            + converse_event(
                {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "OK"}}}
            )
            + converse_event({"contentBlockStop": {"contentBlockIndex": 0}})
            + converse_event({"messageStop": {"stopReason": "end_turn"}})
            + converse_event({"metadata": {"usage": {"inputTokens": 6, "outputTokens": 1}}})
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=body,
                headers={"content-type": "application/vnd.amazon.eventstream"},
            )

        transport = TraciumHTTPXTransport(httpx.MockTransport(handler))
        with httpx.Client(transport=transport) as client:
            client.post(url, json={"messages": []})

        assert len(captured) == 1
        call = captured[0]
        assert call.provider == "bedrock"
        assert call.output == "OK"
        assert call.input_tokens == 6
        assert call.output_tokens == 1
