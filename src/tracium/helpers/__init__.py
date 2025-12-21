"""
Helper modules for Tracium SDK.
"""

from .logging_config import get_logger, redact_sensitive_data
from .retry import RetryConfig, retry_with_backoff
from .security import SecurityConfig
from .validation import (
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

__all__ = [
    "get_logger",
    "redact_sensitive_data",
    "RetryConfig",
    "retry_with_backoff",
    "SecurityConfig",
    "validate_agent_name",
    "validate_api_key",
    "validate_error_message",
    "validate_metadata",
    "validate_name",
    "validate_span_id",
    "validate_span_type",
    "validate_tags",
    "validate_trace_id",
]
