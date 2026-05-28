"""
Tests for provider detection and request/response parsing.

The provider layer is pure data-in / data-out — no I/O, no globals — so the
tests are straightforward dict-shape assertions.
"""

from __future__ import annotations

import json

from tracium.integrations.http_capture.providers import (
    detect_provider,
    parse,
    parse_anthropic,
    parse_bedrock,
    parse_cohere,
    parse_google_gemini,
    parse_openai_compatible,
    parse_unknown,
)


class TestDetectProvider:
    def test_openai(self) -> None:
        assert detect_provider("https://api.openai.com/v1/chat/completions")[0] == "openai"

    def test_anthropic(self) -> None:
        assert detect_provider("https://api.anthropic.com/v1/messages")[0] == "anthropic"

    def test_groq_openai_compatible(self) -> None:
        assert detect_provider("https://api.groq.com/openai/v1/chat/completions")[0] == "groq"

    def test_together(self) -> None:
        assert detect_provider("https://api.together.xyz/v1/chat/completions")[0] == "together"

    def test_mistral(self) -> None:
        assert detect_provider("https://api.mistral.ai/v1/chat/completions")[0] == "mistral"

    def test_google_gemini(self) -> None:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent"
        assert detect_provider(url)[0] == "google"

    def test_bedrock(self) -> None:
        url = "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude/invoke"
        assert detect_provider(url)[0] == "bedrock"

    def test_cohere(self) -> None:
        assert detect_provider("https://api.cohere.ai/v1/chat")[0] == "cohere"
        assert detect_provider("https://api.cohere.com/v1/chat")[0] == "cohere"

    def test_azure_openai(self) -> None:
        url = "https://my-resource.openai.azure.com/openai/deployments/gpt-4/chat/completions"
        assert detect_provider(url)[0] == "azure-openai"

    def test_self_hosted_via_path(self) -> None:
        """Ollama / vLLM / LocalAI on `/v1/chat/completions`."""
        assert detect_provider("http://localhost:11434/v1/chat/completions")[0] == "openai-compatible"

    def test_non_llm_url(self) -> None:
        assert detect_provider("https://example.com/api/users") is None

    def test_invalid_url(self) -> None:
        assert detect_provider("") is None
        assert detect_provider("not a url") is None


class TestOpenAIParser:
    def test_chat_completion(self) -> None:
        request = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}],
        }
        response = {
            "choices": [{"message": {"role": "assistant", "content": "hello"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }
        call = parse_openai_compatible(
            "https://api.openai.com/v1/chat/completions",
            request,
            response,
            200,
        )
        assert call.provider == "openai"
        assert call.operation == "chat"
        assert call.model == "gpt-4"
        assert call.input == request["messages"]
        assert call.output == "hello"
        assert call.input_tokens == 5
        assert call.output_tokens == 2

    def test_cached_tokens_extracted(self) -> None:
        response = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "prompt_tokens_details": {"cached_tokens": 80},
            },
        }
        call = parse_openai_compatible(
            "https://api.openai.com/v1/chat/completions", {"model": "gpt-4"}, response, 200
        )
        assert call.cached_input_tokens == 80

    def test_tool_calls(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": "{}"},
                            }
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        call = parse_openai_compatible(
            "https://api.openai.com/v1/chat/completions", {"model": "gpt-4"}, response, 200
        )
        assert call.tool_calls is not None
        assert call.tool_calls[0]["function"]["name"] == "get_weather"

    def test_embedding_operation_detected(self) -> None:
        response = {"data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}]}
        call = parse_openai_compatible(
            "https://api.openai.com/v1/embeddings",
            {"model": "text-embedding-3-small", "input": "hi"},
            response,
            200,
        )
        assert call.operation == "embedding"
        # The per-operation parser now reports both count and dimensions.
        assert call.output == {"vectors": 2, "dimensions": 2}

    def test_error_response(self) -> None:
        response = {"error": {"message": "Bad request", "type": "invalid_request_error"}}
        call = parse_openai_compatible(
            "https://api.openai.com/v1/chat/completions", {"model": "gpt-4"}, response, 400
        )
        assert call.error is not None
        assert "Bad request" in call.error
        assert "400" in call.error


class TestAnthropicParser:
    def test_messages_call(self) -> None:
        request = {
            "model": "claude-3-5-sonnet-20240620",
            "system": "You are helpful",
            "messages": [{"role": "user", "content": "hi"}],
        }
        response = {
            "content": [{"type": "text", "text": "Hi there"}],
            "usage": {"input_tokens": 8, "output_tokens": 3},
        }
        call = parse_anthropic("https://api.anthropic.com/v1/messages", request, response, 200)
        assert call.provider == "anthropic"
        assert call.model == "claude-3-5-sonnet-20240620"
        assert call.output == "Hi there"
        assert call.input_tokens == 8
        assert call.output_tokens == 3
        assert isinstance(call.input, dict)
        assert call.input["system"] == "You are helpful"

    def test_tool_use_extracted(self) -> None:
        response = {
            "content": [
                {"type": "text", "text": "Let me check"},
                {"type": "tool_use", "id": "tu_1", "name": "search", "input": {"q": "x"}},
            ],
            "usage": {"input_tokens": 5, "output_tokens": 7},
        }
        call = parse_anthropic(
            "https://api.anthropic.com/v1/messages",
            {"model": "claude-3", "messages": []},
            response,
            200,
        )
        assert call.output == "Let me check"
        assert call.tool_calls is not None and len(call.tool_calls) == 1
        assert call.tool_calls[0]["name"] == "search"


class TestGoogleParser:
    def test_gemini_response(self) -> None:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-1.5-pro:generateContent"
        )
        response = {
            "candidates": [{"content": {"parts": [{"text": "Hello!"}]}}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 2},
        }
        call = parse_google_gemini(
            url, {"contents": [{"parts": [{"text": "hi"}]}]}, response, 200
        )
        assert call.provider == "google"
        assert call.model == "gemini-1.5-pro"
        assert call.output == "Hello!"
        assert call.input_tokens == 10
        assert call.output_tokens == 2


class TestBedrockParser:
    def test_invoke_extracts_model_from_url(self) -> None:
        url = (
            "https://bedrock-runtime.us-east-1.amazonaws.com/"
            "model/anthropic.claude-3-5-sonnet-20240620-v1:0/invoke"
        )
        call = parse_bedrock(url, {"messages": []}, {"output": "x", "usage": {}}, 200)
        assert call.provider == "bedrock"
        assert call.model == "anthropic.claude-3-5-sonnet-20240620-v1:0"


class TestCohereParser:
    def test_chat_call(self) -> None:
        request = {"model": "command-r", "message": "hi"}
        response = {
            "text": "hello",
            "meta": {"billed_units": {"input_tokens": 3, "output_tokens": 2}},
        }
        call = parse_cohere("https://api.cohere.ai/v1/chat", request, response, 200)
        assert call.provider == "cohere"
        assert call.output == "hello"
        assert call.input_tokens == 3


class TestUnknownParser:
    def test_falls_back_gracefully(self) -> None:
        call = parse_unknown(
            "https://novel-llm.example.com/v1/predict",
            {"model": "x", "prompt": "hi"},
            {"output": "ok"},
            200,
        )
        assert call.provider == "unknown"
        assert call.model == "x"


class TestHighLevelParse:
    def test_parses_bytes_input(self) -> None:
        body = json.dumps({"model": "gpt-4", "messages": []}).encode()
        resp = json.dumps({"choices": [{"message": {"content": "ok"}}], "usage": {}}).encode()
        call = parse("https://api.openai.com/v1/chat/completions", body, resp, 200)
        assert call is not None
        assert call.provider == "openai"
        assert call.output == "ok"

    def test_returns_none_for_non_llm(self) -> None:
        assert parse("https://example.com/api/users", b"{}", b"{}", 200) is None
