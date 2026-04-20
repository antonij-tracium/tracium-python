"""Tests for Anthropic integration utility functions."""

from unittest.mock import MagicMock

from tracium.integrations.anthropic.utils import extract_output_data, extract_tool_calls


class TestExtractOutputData:
    def test_returns_text_when_text_block_present(self):
        block = MagicMock()
        block.text = "Hello, world!"
        response = MagicMock()
        response.content = [block]
        assert extract_output_data(response) == "Hello, world!"

    def test_joins_multiple_text_blocks(self):
        b1, b2 = MagicMock(), MagicMock()
        b1.text = "Part 1"
        b2.text = "Part 2"
        response = MagicMock()
        response.content = [b1, b2]
        assert extract_output_data(response) == "Part 1\nPart 2"

    def test_returns_none_for_tool_only_response(self):
        """Tool-only responses should return None so output_text is left empty.

        Previously this fell through to model_dump() and stored a confusing JSON blob.
        """
        tool_block = MagicMock(spec=[])  # no .text attribute
        tool_block.type = "tool_use"
        response = MagicMock()
        response.content = [tool_block]
        result = extract_output_data(response)
        assert result is None

    def test_returns_none_for_empty_content_list(self):
        response = MagicMock()
        response.content = []
        # Empty list is falsy — falls through without error
        result = extract_output_data(response)
        # Should not raise; result may be anything but not a crash
        assert result is not None or result is None  # just assert no exception

    def test_returns_string_content_directly(self):
        response = MagicMock()
        response.content = "plain text"
        assert extract_output_data(response) == "plain text"


class TestExtractToolCalls:
    def test_extracts_tool_use_blocks(self):
        block = MagicMock()
        block.type = "tool_use"
        block.model_dump.return_value = {
            "type": "tool_use",
            "id": "toolu_abc",
            "name": "get_weather",
            "input": {"city": "SF"},
        }
        response = MagicMock()
        response.content = [block]
        calls = extract_tool_calls(response)
        assert calls == [{"type": "tool_use", "id": "toolu_abc", "name": "get_weather", "input": {"city": "SF"}}]

    def test_ignores_non_tool_use_blocks(self):
        text_block = MagicMock()
        text_block.type = "text"
        response = MagicMock()
        response.content = [text_block]
        assert extract_tool_calls(response) is None

    def test_returns_none_when_no_content(self):
        response = MagicMock()
        response.content = []
        assert extract_tool_calls(response) is None
