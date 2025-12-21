"""
Structured logging configuration for Tracium SDK.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

logger = logging.getLogger("tracium")


def configure_logging(
    level: int | str = logging.WARNING,
    format_string: str | None = None,
    stream: Any = sys.stderr,
) -> None:
    """
    Configure logging for the Tracium SDK.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages
        stream: Stream to write logs to (default: stderr)
    """
    if format_string is None:
        format_string = (
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s [%(filename)s:%(lineno)d]"
        )

    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(format_string))

    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Get a logger instance for the Tracium SDK.

    Args:
        name: Optional logger name (default: "tracium")

    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(f"tracium.{name}")
    return logger


def redact_sensitive_data(
    data: dict[str, Any], sensitive_keys: set[str] | None = None
) -> dict[str, Any]:
    """
    Redact sensitive data from dictionaries for logging.

    Args:
        data: Dictionary to redact
        sensitive_keys: Set of keys to redact (default: common sensitive keys)

    Returns:
        Dictionary with sensitive values redacted
    """
    if sensitive_keys is None:
        sensitive_keys = {
            "api_key",
            "apiKey",
            "apikey",
            "password",
            "secret",
            "token",
            "authorization",
            "x-api-key",
            "X-API-Key",
        }

    sensitive_lower = {k.lower() for k in sensitive_keys}
    redacted = {}
    for key, value in data.items():
        if key.lower() in sensitive_lower:
            redacted[key] = "***REDACTED***"
        elif isinstance(value, dict):
            redacted[key] = redact_sensitive_data(value, sensitive_keys)
        elif isinstance(value, list):
            redacted[key] = [
                redact_sensitive_data(item, sensitive_keys) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            redacted[key] = value

    return redacted
