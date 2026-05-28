"""
Tests for Anthropic's two cache token buckets (``cache_read`` and
``cache_creation``) and for the batch-results JSONL endpoint.
"""

from __future__ import annotations

import json

from tracium.integrations.http_capture.providers import LLMCall, parse


def _b(payload: dict) -> bytes:
    return json.dumps(payload).encode("utf-8")


class TestCacheTokenBuckets:
    URL = "https://api.anthropic.com/v1/messages"

    def test_both_buckets_captured_separately(self) -> None:
        req = {"model": "claude-3-5-sonnet-20241022", "messages": []}
        resp = {
            "content": [{"type": "text", "text": "ok"}],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "cache_read_input_tokens": 1000,
                "cache_creation_input_tokens": 500,
            },
        }
        call = parse(self.URL, _b(req), _b(resp), 200)
        assert call.input_tokens == 100
        assert call.output_tokens == 20
        assert call.cached_input_tokens == 1000  # read
        assert call.cache_creation_input_tokens == 500  # write

    def test_no_caching_leaves_fields_none(self) -> None:
        req = {"model": "claude-3-haiku-20240307", "messages": []}
        resp = {
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 10, "output_tokens": 2},
        }
        call = parse(self.URL, _b(req), _b(resp), 200)
        assert call.cached_input_tokens is None
        assert call.cache_creation_input_tokens is None


class TestBedrockClaudeCacheBuckets:
    URL = (
        "https://bedrock-runtime.us-east-1.amazonaws.com/"
        "model/anthropic.claude-3-5-sonnet-20240620-v1:0/invoke"
    )

    def test_cache_buckets_via_bedrock(self) -> None:
        req = {"messages": [{"role": "user", "content": "hi"}]}
        resp = {
            "content": [{"type": "text", "text": "ok"}],
            "usage": {
                "input_tokens": 50,
                "output_tokens": 10,
                "cache_read_input_tokens": 200,
                "cache_creation_input_tokens": 100,
            },
        }
        call = parse(self.URL, _b(req), _b(resp), 200)
        assert call.cached_input_tokens == 200
        assert call.cache_creation_input_tokens == 100


class TestBatchResults:
    URL = "https://api.anthropic.com/v1/messages/batches/msgbatch_1/results"

    def test_jsonl_aggregation(self) -> None:
        # The endpoint returns newline-delimited JSON: one result per request.
        results = [
            {
                "custom_id": "a",
                "result": {
                    "type": "succeeded",
                    "message": {
                        "model": "claude-3-5-sonnet-20241022",
                        "usage": {
                            "input_tokens": 10,
                            "output_tokens": 3,
                            "cache_creation_input_tokens": 100,
                            "cache_read_input_tokens": 50,
                        },
                    },
                },
            },
            {
                "custom_id": "b",
                "result": {
                    "type": "succeeded",
                    "message": {
                        "model": "claude-3-5-sonnet-20241022",
                        "usage": {
                            "input_tokens": 20,
                            "output_tokens": 5,
                            "cache_creation_input_tokens": 0,
                            "cache_read_input_tokens": 80,
                        },
                    },
                },
            },
            {"custom_id": "c", "result": {"type": "errored", "error": {"message": "x"}}},
            {"custom_id": "d", "result": {"type": "canceled"}},
        ]
        body = ("\n".join(json.dumps(r) for r in results)).encode("utf-8")

        call = parse(self.URL, b"", body, 200)
        assert call.operation == "batch_results"
        assert call.model == "claude-3-5-sonnet-20241022"
        assert call.input_tokens == 30
        assert call.output_tokens == 8
        assert call.cache_creation_input_tokens == 100
        assert call.cached_input_tokens == 130
        assert call.output == {
            "succeeded": 2,
            "errored": 1,
            "canceled": 1,
            "expired": 0,
            "total": 4,
        }

    def test_empty_results_returns_zero_counts(self) -> None:
        call = parse(self.URL, b"", b"", 200)
        assert call.operation == "batch_results"
        assert call.output["total"] == 0


class TestEmitWritesCacheCreationToWire:
    """End-to-end: an LLMCall with cache_creation_input_tokens lands on the wire."""

    def test_payload_includes_cache_creation_field(
        self, tracium_client, monkeypatch
    ) -> None:
        from datetime import datetime, timezone

        from tracium.integrations.http_capture.emit import emit_llm_span

        recorded: list[dict] = []
        original = tracium_client.record_agent_spans

        def capture(trace_id, payloads):
            recorded.extend(payloads)
            return original(trace_id, payloads)

        monkeypatch.setattr(tracium_client, "record_agent_spans", capture)

        with tracium_client.agent_trace(agent_name="cache-test"):
            emit_llm_span(
                LLMCall(
                    provider="anthropic",
                    model="claude-3-5-sonnet-20241022",
                    operation="chat",
                    input_tokens=100,
                    output_tokens=20,
                    cached_input_tokens=1000,
                    cache_creation_input_tokens=500,
                    output="ok",
                ),
                started_at=datetime.now(timezone.utc),
            )

        completed = [p for p in recorded if p.get("status") == "completed"]
        assert completed
        payload = completed[-1]
        assert payload["input_tokens"] == 100
        assert payload["output_tokens"] == 20
        assert payload["cached_input_tokens"] == 1000
        assert payload["cache_creation_input_tokens"] == 500
