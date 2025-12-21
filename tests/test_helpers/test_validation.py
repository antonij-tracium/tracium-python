"""
Tests for validation functions.
"""

import pytest

from tracium.helpers.validation import (
    MAX_AGENT_NAME_LENGTH,
    MAX_ERROR_MESSAGE_LENGTH,
    MAX_METADATA_SIZE,
    MAX_NAME_LENGTH,
    MAX_SPAN_ID_LENGTH,
    MAX_SPAN_TYPE_LENGTH,
    MAX_TAG_LENGTH,
    MAX_TAGS_COUNT,
    MAX_TRACE_ID_LENGTH,
    validate_agent_name,
    validate_api_key,
    validate_error_message,
    validate_metadata,
    validate_name,
    validate_span_id,
    validate_span_type,
    validate_tags,
    validate_trace_id,
)


class TestValidateAgentName:
    """Tests for validate_agent_name."""

    def test_valid_agent_name(self):
        """Test validation of valid agent names."""
        assert validate_agent_name("my-agent") == "my-agent"
        assert validate_agent_name("my_agent") == "my_agent"
        assert validate_agent_name("my.agent") == "my.agent"
        assert validate_agent_name("My Agent 123") == "My Agent 123"

    @pytest.mark.parametrize("name", ["", "   "])
    def test_empty_agent_name_raises(self, name: str):
        """Test that empty agent name raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_agent_name(name)

    def test_too_long_agent_name_raises(self):
        """Test that too long agent name raises ValueError."""
        long_name = "a" * (MAX_AGENT_NAME_LENGTH + 1)
        with pytest.raises(ValueError, match="exceeds maximum length"):
            validate_agent_name(long_name)

    @pytest.mark.parametrize("name", ["agent@name", "agent#name"])
    def test_invalid_characters_raises(self, name: str):
        """Test that invalid characters raise ValueError."""
        with pytest.raises(ValueError, match="can only contain"):
            validate_agent_name(name)

    @pytest.mark.parametrize("value", [123, None])
    def test_non_string_raises(self, value):
        """Test that non-string input raises TypeError."""
        with pytest.raises(TypeError):
            validate_agent_name(value)


class TestValidateTraceId:
    """Tests for validate_trace_id."""

    def test_valid_trace_id(self):
        """Test validation of valid trace IDs."""
        assert validate_trace_id("trace-123") == "trace-123"
        assert validate_trace_id("trace_123") == "trace_123"
        assert validate_trace_id("trace123") == "trace123"

    def test_empty_trace_id_raises(self):
        """Test that empty trace ID raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_trace_id("")

    def test_too_long_trace_id_raises(self):
        """Test that too long trace ID raises ValueError."""
        long_id = "a" * (MAX_TRACE_ID_LENGTH + 1)
        with pytest.raises(ValueError, match="exceeds maximum length"):
            validate_trace_id(long_id)

    @pytest.mark.parametrize("trace_id", ["trace@id", "trace id"])
    def test_invalid_characters_raises(self, trace_id: str):
        """Test that invalid characters raise ValueError."""
        with pytest.raises(ValueError, match="can only contain"):
            validate_trace_id(trace_id)


class TestValidateSpanId:
    """Tests for validate_span_id."""

    def test_valid_span_id(self):
        """Test validation of valid span IDs."""
        assert validate_span_id("span-123") == "span-123"
        assert validate_span_id("span_123") == "span_123"

    def test_empty_span_id_raises(self):
        """Test that empty span ID raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_span_id("")

    def test_too_long_span_id_raises(self):
        """Test that too long span ID raises ValueError."""
        long_id = "a" * (MAX_SPAN_ID_LENGTH + 1)
        with pytest.raises(ValueError, match="exceeds maximum length"):
            validate_span_id(long_id)


class TestValidateSpanType:
    """Tests for validate_span_type."""

    def test_valid_span_type(self):
        """Test validation of valid span types."""
        assert validate_span_type("llm") == "llm"
        assert validate_span_type("tool") == "tool"

    def test_empty_span_type_raises(self):
        """Test that empty span type raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_span_type("")

    def test_too_long_span_type_raises(self):
        """Test that too long span type raises ValueError."""
        long_type = "a" * (MAX_SPAN_TYPE_LENGTH + 1)
        with pytest.raises(ValueError, match="exceeds maximum length"):
            validate_span_type(long_type)


class TestValidateTags:
    """Tests for validate_tags."""

    def test_valid_tags(self):
        """Test validation of valid tags."""
        assert validate_tags(["tag1", "tag2"]) == ["tag1", "tag2"]
        assert validate_tags(("tag1", "tag2")) == ["tag1", "tag2"]
        assert validate_tags(None) == []

    def test_empty_tags(self):
        """Test that empty tags list returns empty list."""
        assert validate_tags([]) == []

    def test_too_many_tags_raises(self):
        """Test that too many tags raises ValueError."""
        many_tags = [f"tag{i}" for i in range(MAX_TAGS_COUNT + 1)]
        with pytest.raises(ValueError, match="cannot exceed"):
            validate_tags(many_tags)

    def test_too_long_tag_raises(self):
        """Test that too long tag raises ValueError."""
        long_tag = "a" * (MAX_TAG_LENGTH + 1)
        with pytest.raises(ValueError, match="exceeds maximum length"):
            validate_tags([long_tag])

    def test_duplicate_tags_removed(self):
        """Test that duplicate tags are removed."""
        tags = ["tag1", "tag2", "tag1", "TAG1"]
        result = validate_tags(tags)
        assert len(result) == 2
        assert "tag1" in result
        assert "tag2" in result

    def test_none_tags_filtered(self):
        """Test that None tags are filtered out."""
        assert validate_tags(["tag1", None, "tag2", None]) == ["tag1", "tag2"]

    @pytest.mark.parametrize("value", ["not a list", 123])
    def test_non_sequence_raises(self, value):
        """Test that non-sequence input raises TypeError."""
        with pytest.raises(TypeError):
            validate_tags(value)


class TestValidateMetadata:
    """Tests for validate_metadata."""

    def test_valid_metadata(self):
        """Test validation of valid metadata."""
        metadata = {"key1": "value1", "key2": 123}
        assert validate_metadata(metadata) == metadata

    def test_none_metadata(self):
        """Test that None metadata returns empty dict."""
        assert validate_metadata(None) == {}

    def test_too_large_metadata_raises(self):
        """Test that too large metadata raises ValueError."""
        large_value = "x" * (MAX_METADATA_SIZE + 1)
        metadata = {"key": large_value}
        with pytest.raises(ValueError, match="exceeds maximum size"):
            validate_metadata(metadata)

    @pytest.mark.parametrize("value", ["not a dict", 123])
    def test_non_dict_raises(self, value):
        """Test that non-dict input raises TypeError."""
        with pytest.raises(TypeError):
            validate_metadata(value)

    def test_non_string_key_raises(self):
        """Test that non-string key raises TypeError."""
        with pytest.raises(TypeError, match="keys must be strings"):
            validate_metadata({123: "value"})


class TestValidateApiKey:
    """Tests for validate_api_key."""

    def test_valid_api_key(self):
        key = "sk_test_1234567890abcdefghijklmnopqrstuvwxyz"
        assert validate_api_key(key) == key

    def test_empty_api_key_raises(self):
        """Test that empty API key raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_api_key("")

    def test_too_short_api_key_raises(self):
        """Test that too short API key raises ValueError."""
        with pytest.raises(ValueError, match="too short"):
            validate_api_key("sk")

    def test_test_key_allowed(self):
        """Test that test keys are allowed when flag is set."""
        short_key = "sk_test_123"
        assert validate_api_key(short_key, allow_test_keys=True) == short_key

    def test_non_string_raises(self):
        """Test that non-string input raises TypeError."""
        with pytest.raises(TypeError):
            validate_api_key(123)


class TestValidateErrorMessage:
    """Tests for validate_error_message."""

    def test_valid_error_message(self):
        """Test validation of valid error message."""
        error = "Something went wrong"
        assert validate_error_message(error) == error

    def test_long_error_message_truncated(self):
        """Test that long error message is truncated."""
        long_error = "x" * (MAX_ERROR_MESSAGE_LENGTH + 100)
        result = validate_error_message(long_error)
        assert len(result) <= MAX_ERROR_MESSAGE_LENGTH + len("... (truncated)")
        assert result.endswith("... (truncated)")

    def test_non_string_converted(self):
        """Test that non-string is converted to string."""
        assert validate_error_message(123) == "123"
        assert validate_error_message(Exception("test")) == "test"


class TestValidateName:
    """Tests for validate_name."""

    def test_valid_name(self):
        """Test validation of valid name."""
        assert validate_name("my-name") == "my-name"

    def test_none_name(self):
        """Test that None name returns None."""
        assert validate_name(None) is None

    @pytest.mark.parametrize("name", ["", "   "])
    def test_empty_name_returns_none(self, name: str):
        """Test that empty name returns None."""
        assert validate_name(name) is None

    def test_too_long_name_raises(self):
        """Test that too long name raises ValueError."""
        long_name = "a" * (MAX_NAME_LENGTH + 1)
        with pytest.raises(ValueError, match="exceeds maximum length"):
            validate_name(long_name)

    def test_non_string_raises(self):
        """Test that non-string input raises TypeError."""
        with pytest.raises(TypeError):
            validate_name(123)
