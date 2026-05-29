"""
Tests for the extended provider coverage:

- OpenAI Responses API
- OpenAI Assistants v2 (threads, runs, steps, tool submission)
- OpenAI Batch API
- Anthropic Batches + count_tokens
- Google Gemini streamGenerateContent / embedContent / countTokens
- Vertex AI host detection
- AWS Bedrock Converse + per-model Invoke unwrapping (Claude, Llama, Titan,
  Mistral, Cohere)

Each provider is exercised through ``parse(url, request_bytes, response_bytes,
status)`` so the tests run the same path the transports use in production.
"""

from __future__ import annotations

import json

from tracium.integrations.http_capture.providers import detect_provider, parse


def _b(payload: dict) -> bytes:
    return json.dumps(payload).encode("utf-8")


# --------------------------------------------------------------------------- #
# OpenAI Responses API                                                         #
# --------------------------------------------------------------------------- #


class TestOpenAIResponses:
    URL = "https://api.openai.com/v1/responses"

    def test_basic_text_response(self) -> None:
        req = {"model": "gpt-4o", "input": "Say hi"}
        resp = {
            "model": "gpt-4o",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hi there"}],
                }
            ],
            "usage": {
                "input_tokens": 5,
                "output_tokens": 2,
                "input_tokens_details": {"cached_tokens": 0},
            },
        }
        call = parse(self.URL, _b(req), _b(resp), 200)
        assert call is not None
        assert call.provider == "openai"
        assert call.operation == "responses"
        assert call.model == "gpt-4o"
        assert call.output == "Hi there"
        assert call.input_tokens == 5
        assert call.output_tokens == 2

    def test_function_call_output(self) -> None:
        req = {"model": "gpt-4o", "input": "weather?", "tools": [{"type": "function"}]}
        resp = {
            "output": [{"type": "function_call", "name": "get_weather", "arguments": "{}"}],
            "usage": {"input_tokens": 8, "output_tokens": 4},
        }
        call = parse(self.URL, _b(req), _b(resp), 200)
        assert call.tool_calls is not None
        assert call.tool_calls[0]["name"] == "get_weather"

    def test_error_response(self) -> None:
        resp = {"error": {"message": "Rate limited"}}
        call = parse(self.URL, _b({"model": "gpt-4o"}), _b(resp), 429)
        assert call.error is not None
        assert "429" in call.error
        assert "Rate limited" in call.error


# --------------------------------------------------------------------------- #
# OpenAI Assistants v2                                                         #
# --------------------------------------------------------------------------- #


class TestOpenAIAssistants:
    def test_create_run_returns_assistant_run_operation(self) -> None:
        url = "https://api.openai.com/v1/threads/thread_abc/runs"
        req = {"assistant_id": "asst_xyz", "tools": [{"type": "code_interpreter"}]}
        resp = {
            "id": "run_1",
            "status": "queued",
            "model": "gpt-4o",
            "usage": None,
        }
        call = parse(url, _b(req), _b(resp), 200)
        assert call.operation == "assistant.run"
        assert call.model == "gpt-4o"
        assert call.extra["assistant_status"] == "queued"
        assert call.tools and call.tools[0]["type"] == "code_interpreter"

    def test_poll_run_picks_up_completed_usage(self) -> None:
        url = "https://api.openai.com/v1/threads/thread_abc/runs/run_1"
        resp = {
            "id": "run_1",
            "status": "completed",
            "model": "gpt-4o",
            "usage": {"prompt_tokens": 50, "completion_tokens": 12},
        }
        call = parse(url, b"", _b(resp), 200)
        assert call.operation == "assistant.run"
        assert call.extra["assistant_status"] == "completed"
        assert call.input_tokens == 50
        assert call.output_tokens == 12

    def test_submit_tool_outputs(self) -> None:
        url = "https://api.openai.com/v1/threads/thread_abc/runs/run_1/submit_tool_outputs"
        req = {"tool_outputs": [{"tool_call_id": "tc_1", "output": "42"}]}
        resp = {"id": "run_1", "status": "in_progress"}
        call = parse(url, _b(req), _b(resp), 200)
        assert call.operation == "assistant.tool_submit"
        assert call.extra["assistant_status"] == "in_progress"

    def test_run_steps_with_tool_calls(self) -> None:
        url = "https://api.openai.com/v1/threads/thread_abc/runs/run_1/steps/step_1"
        resp = {
            "id": "step_1",
            "status": "completed",
            "model": "gpt-4o",
            "step_details": {
                "type": "tool_calls",
                "tool_calls": [{"id": "tc_1", "type": "function", "function": {"name": "f"}}],
            },
        }
        call = parse(url, b"", _b(resp), 200)
        assert call.operation == "assistant.step"
        assert call.tool_calls is not None
        assert call.tool_calls[0]["function"]["name"] == "f"

    def test_message_endpoint(self) -> None:
        url = "https://api.openai.com/v1/threads/thread_abc/messages"
        resp = {"object": "list", "data": []}
        call = parse(url, b"", _b(resp), 200)
        assert call.operation == "thread.message"


# --------------------------------------------------------------------------- #
# OpenAI Batch API                                                             #
# --------------------------------------------------------------------------- #


class TestOpenAIBatch:
    URL = "https://api.openai.com/v1/batches"

    def test_create_batch(self) -> None:
        req = {
            "input_file_id": "file_abc",
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        }
        resp = {
            "id": "batch_1",
            "status": "validating",
            "request_counts": {"total": 100},
        }
        call = parse(self.URL, _b(req), _b(resp), 200)
        assert call.operation == "batch"
        assert call.extra["batch_status"] == "validating"
        assert call.output == {
            "batch_id": "batch_1",
            "status": "validating",
            "request_counts": {"total": 100},
        }
        # No model_id at this layer — emit will use span_type="custom".
        assert call.model is None


# --------------------------------------------------------------------------- #
# Anthropic count_tokens + batches                                             #
# --------------------------------------------------------------------------- #


class TestAnthropicExtensions:
    def test_count_tokens(self) -> None:
        url = "https://api.anthropic.com/v1/messages/count_tokens"
        req = {
            "model": "claude-3-5-sonnet-20240620",
            "messages": [{"role": "user", "content": "hi"}],
        }
        resp = {"input_tokens": 8}
        call = parse(url, _b(req), _b(resp), 200)
        assert call.operation == "count_tokens"
        assert call.model == "claude-3-5-sonnet-20240620"
        assert call.input_tokens == 8

    def test_batches(self) -> None:
        url = "https://api.anthropic.com/v1/messages/batches"
        req = {
            "requests": [
                {"custom_id": "a", "params": {"model": "claude-3", "messages": []}},
                {"custom_id": "b", "params": {"model": "claude-3", "messages": []}},
            ]
        }
        resp = {
            "id": "msgbatch_1",
            "type": "message_batch",
            "processing_status": "in_progress",
            "request_counts": {"processing": 2},
        }
        call = parse(url, _b(req), _b(resp), 200)
        assert call.operation == "batch"
        assert call.input == {"request_count": 2}
        assert call.extra["batch_status"] == "in_progress"


# --------------------------------------------------------------------------- #
# Google Gemini variations                                                     #
# --------------------------------------------------------------------------- #


class TestGeminiExtensions:
    def test_stream_generate_content(self) -> None:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            "models/gemini-1.5-pro:streamGenerateContent"
        )
        # Even non-streamed parse of the full response should set the streaming flag.
        resp = {
            "candidates": [{"content": {"parts": [{"text": "Hi"}]}}],
            "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 1},
        }
        call = parse(url, _b({"contents": []}), _b(resp), 200)
        assert call.operation == "chat"
        assert call.extra.get("streaming") is True
        assert call.model == "gemini-1.5-pro"

    def test_embed_content(self) -> None:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            "models/text-embedding-004:embedContent"
        )
        resp = {"embedding": {"values": [0.1, 0.2]}}
        call = parse(
            url,
            _b({"content": {"parts": [{"text": "hi"}]}}),
            _b(resp),
            200,
        )
        assert call.operation == "embedding"
        assert call.model == "text-embedding-004"
        assert call.output == {"vectors": 1}

    def test_count_tokens(self) -> None:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:countTokens"
        resp = {"totalTokens": 42}
        call = parse(url, _b({"contents": []}), _b(resp), 200)
        assert call.operation == "count_tokens"
        assert call.input_tokens == 42

    def test_vertex_ai_host(self) -> None:
        url = (
            "https://us-central1-aiplatform.googleapis.com/v1/projects/p/"
            "locations/us-central1/publishers/google/models/gemini-1.5-pro:generateContent"
        )
        found = detect_provider(url)
        assert found is not None
        assert found[0] == "google-vertex"
        resp = {
            "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
        }
        call = parse(url, _b({"contents": []}), _b(resp), 200)
        assert call.model == "gemini-1.5-pro"
        assert call.output == "ok"


# --------------------------------------------------------------------------- #
# Bedrock Converse + per-model Invoke                                          #
# --------------------------------------------------------------------------- #


class TestBedrockConverse:
    def test_converse(self) -> None:
        url = (
            "https://bedrock-runtime.us-east-1.amazonaws.com/"
            "model/anthropic.claude-3-5-sonnet-20240620-v1:0/converse"
        )
        req = {
            "messages": [{"role": "user", "content": [{"text": "hi"}]}],
            "system": [{"text": "be helpful"}],
            "toolConfig": {"tools": [{"toolSpec": {"name": "search"}}]},
        }
        resp = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "hello"}],
                }
            },
            "usage": {"inputTokens": 7, "outputTokens": 2},
        }
        call = parse(url, _b(req), _b(resp), 200)
        assert call.provider == "bedrock"
        assert call.model == "anthropic.claude-3-5-sonnet-20240620-v1:0"
        assert call.output == "hello"
        assert call.input_tokens == 7
        assert call.tools and call.tools[0]["toolSpec"]["name"] == "search"


class TestBedrockInvokePerModel:
    def test_claude_via_bedrock(self) -> None:
        url = (
            "https://bedrock-runtime.us-east-1.amazonaws.com/"
            "model/anthropic.claude-3-5-sonnet-20240620-v1:0/invoke"
        )
        req = {"messages": [{"role": "user", "content": "hi"}]}
        resp = {
            "content": [{"type": "text", "text": "hello"}],
            "usage": {"input_tokens": 4, "output_tokens": 1},
        }
        call = parse(url, _b(req), _b(resp), 200)
        assert call.output == "hello"
        assert call.input_tokens == 4

    def test_llama_via_bedrock(self) -> None:
        url = (
            "https://bedrock-runtime.us-east-1.amazonaws.com/"
            "model/meta.llama3-70b-instruct-v1:0/invoke"
        )
        req = {"prompt": "Hi"}
        resp = {
            "generation": "hello",
            "prompt_token_count": 2,
            "generation_token_count": 1,
            "stop_reason": "stop",
        }
        call = parse(url, _b(req), _b(resp), 200)
        assert call.output == "hello"
        assert call.input_tokens == 2
        assert call.output_tokens == 1

    def test_titan_via_bedrock(self) -> None:
        url = (
            "https://bedrock-runtime.us-east-1.amazonaws.com/"
            "model/amazon.titan-text-express-v1/invoke"
        )
        req = {"inputText": "Hi"}
        resp = {
            "inputTextTokenCount": 2,
            "results": [{"outputText": "hello", "tokenCount": 1}],
        }
        call = parse(url, _b(req), _b(resp), 200)
        assert call.output == "hello"
        assert call.input_tokens == 2
        assert call.output_tokens == 1

    def test_mistral_via_bedrock(self) -> None:
        url = (
            "https://bedrock-runtime.us-east-1.amazonaws.com/"
            "model/mistral.mixtral-8x7b-instruct-v0:1/invoke"
        )
        req = {"prompt": "Hi"}
        resp = {
            "outputs": [{"text": "hello"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        }
        call = parse(url, _b(req), _b(resp), 200)
        assert call.output == "hello"
        assert call.input_tokens == 2
        assert call.output_tokens == 1

    def test_cohere_via_bedrock(self) -> None:
        url = (
            "https://bedrock-runtime.us-east-1.amazonaws.com/"
            "model/cohere.command-r-plus-v1:0/invoke"
        )
        req = {"prompt": "Hi"}
        resp = {
            "generations": [{"text": "hello"}],
            "prompt_tokens": 2,
            "generation_tokens": 1,
        }
        call = parse(url, _b(req), _b(resp), 200)
        assert call.output == "hello"
        assert call.input_tokens == 2
        assert call.output_tokens == 1
