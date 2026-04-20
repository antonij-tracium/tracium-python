"""Tests for the tool call registry used to auto-link continuation LLM spans."""

from tracium.utils.tool_call_registry import (
    _extract_tool_result_ids,
    find_parent_span_id,
    register_tool_calls,
)


class TestRegisterAndLookup:
    def test_anthropic_tool_use_round_trip(self):
        """Registering an Anthropic tool_use call and looking it up via tool_result."""
        register_tool_calls(
            "trace-1",
            "span-A",
            [{"type": "tool_use", "id": "toolu_abc", "name": "get_weather", "input": {}}],
        )
        messages = [
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_abc", "content": "sunny"}],
            }
        ]
        assert find_parent_span_id("trace-1", messages) == "span-A"

    def test_openai_function_call_round_trip(self):
        """Registering an OpenAI function_call and looking it up via role=tool message."""
        register_tool_calls(
            "trace-2",
            "span-B",
            [{"id": "call_xyz", "type": "function", "function": {"name": "search", "arguments": "{}"}}],
        )
        messages = [{"role": "tool", "tool_call_id": "call_xyz", "content": "results"}]
        assert find_parent_span_id("trace-2", messages) == "span-B"

    def test_openai_responses_api_round_trip(self):
        """Registering an OpenAI Responses API function_call and looking it up."""
        register_tool_calls(
            "trace-3",
            "span-C",
            [{"type": "function_call", "call_id": "call_foo", "name": "calc", "arguments": "{}"}],
        )
        messages = [{"type": "function_call_output", "call_id": "call_foo", "content": "42"}]
        assert find_parent_span_id("trace-3", messages) == "span-C"

    def test_unknown_id_returns_none(self):
        messages = [
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "unknown_id"}]}
        ]
        assert find_parent_span_id("trace-any", messages) is None

    def test_trace_isolation(self):
        """A tool_use_id registered for trace-A should not match trace-B."""
        register_tool_calls("trace-A", "span-A", [{"type": "tool_use", "id": "shared_id"}])
        messages = [
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "shared_id"}]}
        ]
        assert find_parent_span_id("trace-B", messages) is None

    def test_empty_messages_returns_none(self):
        assert find_parent_span_id("trace-any", []) is None

    def test_none_messages_returns_none(self):
        assert find_parent_span_id("trace-any", None) is None  # type: ignore[arg-type]

    def test_no_tool_result_content_returns_none(self):
        messages = [{"role": "user", "content": "hello"}]
        assert find_parent_span_id("trace-any", messages) is None


class TestExtractToolResultIds:
    def test_anthropic_format(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "id1"},
                    {"type": "tool_result", "tool_use_id": "id2"},
                ],
            }
        ]
        ids = _extract_tool_result_ids(messages)
        assert ids == ["id1", "id2"]

    def test_openai_role_tool_format(self):
        messages = [
            {"role": "tool", "tool_call_id": "call_1", "content": "output"},
            {"role": "tool", "tool_call_id": "call_2", "content": "output"},
        ]
        ids = _extract_tool_result_ids(messages)
        assert ids == ["call_1", "call_2"]

    def test_mixed_messages_skips_non_tool(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "tool", "tool_call_id": "call_3", "content": "result"},
        ]
        ids = _extract_tool_result_ids(messages)
        assert ids == ["call_3"]

    def test_empty_input(self):
        assert _extract_tool_result_ids([]) == []
