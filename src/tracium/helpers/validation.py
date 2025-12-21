"""
Input validation and sanitization for Tracium SDK.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

MAX_AGENT_NAME_LENGTH = 256
MAX_TRACE_ID_LENGTH = 256
MAX_SPAN_ID_LENGTH = 256
MAX_SPAN_TYPE_LENGTH = 128
MAX_TAG_LENGTH = 128
MAX_TAGS_COUNT = 100
MAX_METADATA_SIZE = 1_000_000
MAX_ERROR_MESSAGE_LENGTH = 10_000
MAX_NAME_LENGTH = 512


def validate_agent_name(agent_name: str) -> str:
    """
    Validate and sanitize agent name.

    Args:
        agent_name: Agent name to validate

    Returns:
        Sanitized agent name

    Raises:
        ValueError: If agent name is invalid
    """
    if not isinstance(agent_name, str):
        raise TypeError(f"agent_name must be a string, got {type(agent_name).__name__}")

    agent_name = agent_name.strip()

    if not agent_name:
        raise ValueError("agent_name cannot be empty")

    if len(agent_name) > MAX_AGENT_NAME_LENGTH:
        raise ValueError(f"agent_name exceeds maximum length of {MAX_AGENT_NAME_LENGTH} characters")

    if not re.match(r"^[a-zA-Z0-9\s\-_.]+$", agent_name):
        raise ValueError(
            "agent_name can only contain alphanumeric characters, spaces, "
            "hyphens, underscores, and dots"
        )

    return agent_name


def validate_trace_id(trace_id: str) -> str:
    """
    Validate and sanitize trace ID.

    Args:
        trace_id: Trace ID to validate

    Returns:
        Sanitized trace ID

    Raises:
        ValueError: If trace ID is invalid
    """
    if not isinstance(trace_id, str):
        raise TypeError(f"trace_id must be a string, got {type(trace_id).__name__}")

    trace_id = trace_id.strip()

    if not trace_id:
        raise ValueError("trace_id cannot be empty")

    if len(trace_id) > MAX_TRACE_ID_LENGTH:
        raise ValueError(f"trace_id exceeds maximum length of {MAX_TRACE_ID_LENGTH} characters")

    if not re.match(r"^[a-zA-Z0-9\-_]+$", trace_id):
        raise ValueError(
            "trace_id can only contain alphanumeric characters, hyphens, and underscores"
        )

    return trace_id


def validate_span_id(span_id: str) -> str:
    """
    Validate and sanitize span ID.

    Args:
        span_id: Span ID to validate

    Returns:
        Sanitized span ID

    Raises:
        ValueError: If span ID is invalid
    """
    if not isinstance(span_id, str):
        raise TypeError(f"span_id must be a string, got {type(span_id).__name__}")

    span_id = span_id.strip()

    if not span_id:
        raise ValueError("span_id cannot be empty")

    if len(span_id) > MAX_SPAN_ID_LENGTH:
        raise ValueError(f"span_id exceeds maximum length of {MAX_SPAN_ID_LENGTH} characters")

    return span_id


def validate_span_type(span_type: str) -> str:
    """
    Validate and sanitize span type.

    Args:
        span_type: Span type to validate

    Returns:
        Sanitized span type

    Raises:
        ValueError: If span type is invalid
    """
    if not isinstance(span_type, str):
        raise TypeError(f"span_type must be a string, got {type(span_type).__name__}")

    span_type = span_type.strip()

    if not span_type:
        raise ValueError("span_type cannot be empty")

    if len(span_type) > MAX_SPAN_TYPE_LENGTH:
        raise ValueError(f"span_type exceeds maximum length of {MAX_SPAN_TYPE_LENGTH} characters")

    return span_type


def validate_tags(tags: Sequence[str] | None) -> list[str]:
    """
    Validate and sanitize tags.

    Args:
        tags: Tags to validate

    Returns:
        List of sanitized tags

    Raises:
        ValueError: If tags are invalid
    """
    if tags is None:
        return []

    if not isinstance(tags, (list, tuple)):
        raise TypeError(f"tags must be a list or tuple, got {type(tags).__name__}")

    if len(tags) > MAX_TAGS_COUNT:
        raise ValueError(f"tags cannot exceed {MAX_TAGS_COUNT} items")

    validated_tags: list[str] = []
    seen_tags: set[str] = set()

    for tag in tags:
        if tag is None:
            continue

        tag_str = str(tag).strip()
        if not tag_str:
            continue

        if len(tag_str) > MAX_TAG_LENGTH:
            raise ValueError(
                f"Tag exceeds maximum length of {MAX_TAG_LENGTH} characters: {tag_str[:50]}..."
            )

        tag_lower = tag_str.lower()
        if tag_lower not in seen_tags:
            validated_tags.append(tag_str)
            seen_tags.add(tag_lower)

    return validated_tags


def validate_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """
    Validate and sanitize metadata.

    Args:
        metadata: Metadata dictionary to validate

    Returns:
        Sanitized metadata dictionary

    Raises:
        ValueError: If metadata is invalid
    """
    if metadata is None:
        return {}

    if not isinstance(metadata, dict):
        raise TypeError(f"metadata must be a dict, got {type(metadata).__name__}")

    try:
        import json

        size = len(json.dumps(metadata).encode("utf-8"))
        if size > MAX_METADATA_SIZE:
            raise ValueError(
                f"metadata exceeds maximum size of {MAX_METADATA_SIZE} bytes "
                f"(approximately {MAX_METADATA_SIZE / 1024:.1f} KB)"
            )
    except TypeError:
        pass

    sanitized: dict[str, Any] = {}
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise TypeError(f"metadata keys must be strings, got {type(key).__name__}")

        if len(key) > 256:
            raise ValueError(
                f"metadata key exceeds maximum length of 256 characters: {key[:50]}..."
            )

        sanitized[key] = value

    return sanitized


def validate_error_message(error: str) -> str:
    """
    Validate and sanitize error message.

    Args:
        error: Error message to validate

    Returns:
        Sanitized error message

    Raises:
        ValueError: If error message is invalid
    """
    if not isinstance(error, str):
        error = str(error)

    error = error.strip()

    if len(error) > MAX_ERROR_MESSAGE_LENGTH:
        error = error[:MAX_ERROR_MESSAGE_LENGTH] + "... (truncated)"

    return error


def validate_name(name: str | None) -> str | None:
    """
    Validate and sanitize a name field (for spans, etc.).

    Args:
        name: Name to validate

    Returns:
        Sanitized name or None

    Raises:
        ValueError: If name is invalid
    """
    if name is None:
        return None

    if not isinstance(name, str):
        raise TypeError(f"name must be a string, got {type(name).__name__}")

    name = name.strip()

    if len(name) > MAX_NAME_LENGTH:
        raise ValueError(f"name exceeds maximum length of {MAX_NAME_LENGTH} characters")

    return name if name else None


def validate_api_key(api_key: str, allow_test_keys: bool = False) -> str:
    """
    Validate API key format.

    Args:
        api_key: API key to validate
        allow_test_keys: If True, allow shorter keys for testing (default: False)

    Returns:
        Validated API key

    Raises:
        ValueError: If API key is invalid
    """
    if not isinstance(api_key, str):
        raise TypeError(f"api_key must be a string, got {type(api_key).__name__}")

    api_key = api_key.strip()

    if not api_key:
        raise ValueError("api_key cannot be empty")

    min_length = 3 if allow_test_keys else 10
    if len(api_key) < min_length:
        raise ValueError(
            f"api_key appears to be invalid (too short, minimum {min_length} characters)"
        )

    return api_key
