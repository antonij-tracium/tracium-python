"""
Tests for _extract_system_prompt.
"""

import pytest

from tracium.models.span_handle import _extract_system_prompt


class TestExtractSystemPromptNone:
    """Tests for inputs that should return None."""

    @pytest.mark.parametrize("input_data", [None, 42, 3.14, True, object()])
    def test_non_extractable_types_return_none(self, input_data):
        assert _extract_system_prompt(input_data) is None

    def test_plain_string_returns_none(self):
        """Plain strings are message content, not system prompts."""
        assert _extract_system_prompt("Hello world") is None

    def test_empty_dict_returns_none(self):
        assert _extract_system_prompt({}) is None

    def test_empty_list_returns_none(self):
        assert _extract_system_prompt([]) is None

    def test_list_without_system_role(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        assert _extract_system_prompt(messages) is None

    def test_dict_with_non_string_system(self):
        """Non-string system values are not extracted."""
        assert _extract_system_prompt({"system": 123}) is None
        assert _extract_system_prompt({"system": ["block"]}) is None

    def test_dict_with_non_string_system_instruction(self):
        assert _extract_system_prompt({"system_instruction": 42}) is None


class TestExtractSystemPromptAnthropic:
    """Tests for Anthropic-style inputs."""

    def test_top_level_system_key(self):
        data = {"system": "You are a helpful assistant.", "messages": []}
        assert _extract_system_prompt(data) == "You are a helpful assistant."

    def test_system_key_with_messages(self):
        data = {
            "system": "Be concise.",
            "messages": [
                {"role": "user", "content": "Hi"},
            ],
        }
        assert _extract_system_prompt(data) == "Be concise."

    def test_system_key_takes_priority_over_messages(self):
        """Top-level system key is checked before scanning messages."""
        data = {
            "system": "Top-level prompt",
            "messages": [
                {"role": "system", "content": "Message-level prompt"},
            ],
        }
        assert _extract_system_prompt(data) == "Top-level prompt"


class TestExtractSystemPromptOpenAI:
    """Tests for OpenAI-style inputs (list of message dicts)."""

    def test_system_role_in_messages(self):
        messages = [
            {"role": "system", "content": "You are a poet."},
            {"role": "user", "content": "Write a haiku"},
        ]
        assert _extract_system_prompt(messages) == "You are a poet."

    def test_system_role_not_first(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "Late system message"},
        ]
        assert _extract_system_prompt(messages) == "Late system message"

    def test_multipart_content(self):
        """OpenAI-style multi-part content blocks."""
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a helpful assistant."},
                ],
            },
        ]
        assert _extract_system_prompt(messages) == "You are a helpful assistant."

    def test_dict_wrapped_messages(self):
        """Dict with a messages key delegates to list extraction."""
        data = {
            "messages": [
                {"role": "system", "content": "Wrapped prompt"},
                {"role": "user", "content": "Hi"},
            ],
        }
        assert _extract_system_prompt(data) == "Wrapped prompt"


class TestExtractSystemPromptGemini:
    """Tests for Google Gemini-style inputs."""

    def test_system_instruction_key(self):
        data = {"system_instruction": "You are a translator."}
        assert _extract_system_prompt(data) == "You are a translator."

    def test_system_instruction_fallback(self):
        """system_instruction is checked when system key is absent."""
        data = {"system_instruction": "Fallback prompt"}
        assert _extract_system_prompt(data) == "Fallback prompt"


class TestExtractSystemPromptLangChain:
    """Tests for LangChain/LangGraph-style inputs."""

    def test_type_field_system(self):
        messages = [
            {"type": "system", "content": "LangChain system prompt"},
            {"type": "human", "content": "Hello"},
        ]
        assert _extract_system_prompt(messages) == "LangChain system prompt"

    def test_type_field_case_insensitive(self):
        messages = [{"type": "System", "content": "Case test"}]
        assert _extract_system_prompt(messages) == "Case test"

    def test_text_field_instead_of_content(self):
        messages = [{"role": "system", "text": "Text field prompt"}]
        assert _extract_system_prompt(messages) == "Text field prompt"


class TestExtractSystemPromptEdgeCases:
    """Edge cases and mixed formats."""

    def test_non_dict_items_in_list_are_skipped(self):
        messages = [
            "not a dict",
            42,
            {"role": "system", "content": "Found it"},
        ]
        assert _extract_system_prompt(messages) == "Found it"

    def test_empty_system_content(self):
        messages = [{"role": "system", "content": ""}]
        # Empty string is falsy, falls through to text field which is also empty
        assert _extract_system_prompt(messages) == ""

    def test_dict_with_messages_key_not_list(self):
        data = {"messages": "not a list"}
        assert _extract_system_prompt(data) is None
