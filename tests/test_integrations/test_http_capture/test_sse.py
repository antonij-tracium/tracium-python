"""
Tests for SSE chunk accumulation across the major LLM streaming formats.
"""

from __future__ import annotations

from tracium.integrations.http_capture.sse import SSEAccumulator


def _feed_lines(acc: SSEAccumulator, lines: list[str]) -> None:
    for line in lines:
        acc.feed(line.encode("utf-8") + b"\n")


class TestOpenAIStream:
    def test_simple_text_completion(self) -> None:
        acc = SSEAccumulator()
        _feed_lines(
            acc,
            [
                'data: {"choices":[{"delta":{"role":"assistant"}}]}',
                "",
                'data: {"choices":[{"delta":{"content":"Hello"}}]}',
                "",
                'data: {"choices":[{"delta":{"content":" world"}}]}',
                "",
                'data: {"choices":[{"finish_reason":"stop"}]}',
                "",
                'data: {"usage":{"prompt_tokens":3,"completion_tokens":2}}',
                "",
                "data: [DONE]",
                "",
            ],
        )
        result = acc.finalize()
        assert result["choices"][0]["message"]["content"] == "Hello world"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["usage"] == {"prompt_tokens": 3, "completion_tokens": 2}

    def test_tool_call_assembly(self) -> None:
        acc = SSEAccumulator()
        _feed_lines(
            acc,
            [
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
                '"type":"function","function":{"name":"get_weather","arguments":""}}]}}]}',
                "",
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
                '"function":{"arguments":"{\\"loc"}}]}}]}',
                "",
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
                '"function":{"arguments":"\\":\\"sf\\"}"}}]}}]}',
                "",
                "data: [DONE]",
                "",
            ],
        )
        result = acc.finalize()
        tool_calls = result["choices"][0]["message"]["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "get_weather"
        assert tool_calls[0]["function"]["arguments"] == '{"loc":"sf"}'

    def test_partial_chunk_safety(self) -> None:
        """Splitting a single SSE event across multiple feed() calls still works."""
        acc = SSEAccumulator()
        acc.feed(b'data: {"choices":[{"delta":')
        acc.feed(b'{"content":"Hi"}}]}\n\n')
        acc.feed(b"data: [DONE]\n\n")
        result = acc.finalize()
        assert result["choices"][0]["message"]["content"] == "Hi"


class TestAnthropicStream:
    def test_text_message(self) -> None:
        acc = SSEAccumulator()
        _feed_lines(
            acc,
            [
                'data: {"type":"message_start","message":{"id":"m_1",'
                '"model":"claude-3","usage":{"input_tokens":5}}}',
                "",
                'data: {"type":"content_block_start","index":0,'
                '"content_block":{"type":"text","text":""}}',
                "",
                'data: {"type":"content_block_delta","index":0,'
                '"delta":{"type":"text_delta","text":"Hi "}}',
                "",
                'data: {"type":"content_block_delta","index":0,'
                '"delta":{"type":"text_delta","text":"there"}}',
                "",
                'data: {"type":"message_delta","usage":{"output_tokens":2}}',
                "",
                'data: {"type":"message_stop"}',
                "",
            ],
        )
        result = acc.finalize()
        assert result["choices"][0]["message"]["content"] == "Hi there"
        assert result["usage"]["input_tokens"] == 5
        assert result["usage"]["output_tokens"] == 2
        assert result["model"] == "claude-3"

    def test_tool_use_streaming(self) -> None:
        acc = SSEAccumulator()
        _feed_lines(
            acc,
            [
                'data: {"type":"content_block_start","index":0,'
                '"content_block":{"type":"tool_use","id":"tu_1","name":"search","input":{}}}',
                "",
                'data: {"type":"content_block_delta","index":0,'
                '"delta":{"type":"input_json_delta","partial_json":"{\\"q"}}',
                "",
                'data: {"type":"content_block_delta","index":0,'
                '"delta":{"type":"input_json_delta","partial_json":"\\":\\"x\\"}"}}',
                "",
            ],
        )
        result = acc.finalize()
        tool_calls = result["choices"][0]["message"]["tool_calls"]
        assert tool_calls[0]["function"]["name"] == "search"
        assert tool_calls[0]["function"]["arguments"] == '{"q":"x"}'


class TestGeminiStream:
    def test_stream_generate_content_text(self) -> None:
        acc = SSEAccumulator()
        _feed_lines(
            acc,
            [
                'data: {"candidates":[{"content":{"parts":[{"text":"Hi "}]}}]}',
                "",
                'data: {"candidates":[{"content":{"parts":[{"text":"there"}]}}],'
                '"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":2}}',
                "",
            ],
        )
        result = acc.finalize()
        assert result["choices"][0]["message"]["content"] == "Hi there"
        # Gemini usage normalized into OpenAI-compatible names so the parser
        # downstream picks them up identically.
        assert result["usage"]["prompt_tokens"] == 3
        assert result["usage"]["completion_tokens"] == 2

    def test_stream_function_call(self) -> None:
        acc = SSEAccumulator()
        _feed_lines(
            acc,
            [
                'data: {"candidates":[{"content":{"parts":['
                '{"functionCall":{"name":"search","args":{"q":"x"}}}'
                "]},"
                '"finishReason":"STOP"}]}',
                "",
            ],
        )
        result = acc.finalize()
        tool_calls = result["choices"][0]["message"]["tool_calls"]
        assert tool_calls[0]["function"]["name"] == "search"
        assert '"q"' in tool_calls[0]["function"]["arguments"]
        assert result["choices"][0]["finish_reason"] == "STOP"


class TestEdgeCases:
    def test_heartbeats_and_blank_lines_ignored(self) -> None:
        acc = SSEAccumulator()
        acc.feed(b": keepalive\n\n")
        acc.feed(b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n')
        result = acc.finalize()
        assert result["choices"][0]["message"]["content"] == "x"

    def test_invalid_json_skipped(self) -> None:
        acc = SSEAccumulator()
        acc.feed(b"data: not-json\n\n")
        acc.feed(b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n')
        result = acc.finalize()
        assert result["choices"][0]["message"]["content"] == "ok"

    def test_empty_stream(self) -> None:
        result = SSEAccumulator().finalize()
        assert result["choices"][0]["message"]["content"] == ""
