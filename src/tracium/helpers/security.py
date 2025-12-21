"""
Security features for Tracium SDK including rate limiting and data redaction.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from .logging_config import redact_sensitive_data


@dataclass
class RateLimiter:
    """
    Token bucket rate limiter for API requests.

    Args:
        max_requests: Maximum number of requests allowed
        time_window: Time window in seconds (default: 60)
    """

    max_requests: int = 100
    time_window: float = 60.0
    _requests: deque[float] = field(default_factory=deque, init=False)

    def acquire(self) -> bool:
        """
        Try to acquire a token for a request.

        Returns:
            True if request is allowed, False if rate limited
        """
        now = time.time()

        while self._requests and self._requests[0] < now - self.time_window:
            self._requests.popleft()

        if len(self._requests) >= self.max_requests:
            return False

        self._requests.append(now)
        return True

    def wait_time(self) -> float:
        """
        Calculate how long to wait before next request is allowed.

        Returns:
            Wait time in seconds, or 0 if request is allowed immediately
        """
        if not self._requests:
            return 0.0

        now = time.time()
        while self._requests and self._requests[0] < now - self.time_window:
            self._requests.popleft()

        if len(self._requests) < self.max_requests:
            return 0.0

        oldest = self._requests[0]
        wait = (oldest + self.time_window) - now
        return max(0.0, wait)


@dataclass
class SecurityConfig:
    """
    Security configuration for the Tracium SDK.

    Args:
        redact_sensitive_fields: Fields to redact from telemetry payloads
        rate_limit_enabled: Enable client-side rate limiting
        rate_limit_max_requests: Maximum requests per time window
        rate_limit_time_window: Time window in seconds for rate limiting
        allow_telemetry_redaction: Allow redacting sensitive data from telemetry (not just logs)
    """

    redact_sensitive_fields: set[str] = field(
        default_factory=lambda: {
            "api_key",
            "apikey",
            "apiKey",
            "password",
            "secret",
            "token",
            "authorization",
            "x-api-key",
            "X-API-Key",
            "access_token",
            "refresh_token",
            "session_id",
            "cookie",
            "credit_card",
            "ssn",
            "social_security",
            "pin",
            "pii",
            "personal_data",
        }
    )
    rate_limit_enabled: bool = True
    rate_limit_max_requests: int = 5000
    rate_limit_time_window: float = 60.0
    allow_telemetry_redaction: bool = False

    def __post_init__(self) -> None:
        """Create rate limiter if enabled."""
        if self.rate_limit_enabled:
            self._rate_limiter = RateLimiter(
                max_requests=self.rate_limit_max_requests,
                time_window=self.rate_limit_time_window,
            )
        else:
            self._rate_limiter = None

    @property
    def rate_limiter(self) -> RateLimiter | None:
        """Get the rate limiter instance."""
        return self._rate_limiter


def redact_telemetry_payload(
    payload: dict[str, Any],
    config: SecurityConfig,
) -> dict[str, Any]:
    """
    Redact sensitive fields from telemetry payloads before sending to API.

    Args:
        payload: The payload dictionary to redact
        config: Security configuration

    Returns:
        Redacted payload dictionary
    """
    if not config.allow_telemetry_redaction:
        return payload

    return redact_sensitive_data(payload, config.redact_sensitive_fields)


def check_rate_limit(config: SecurityConfig) -> tuple[bool, float]:
    """
    Check if a request is allowed under rate limiting.

    Args:
        config: Security configuration

    Returns:
        Tuple of (is_allowed, wait_time_seconds)
    """
    if not config.rate_limit_enabled or config._rate_limiter is None:
        return True, 0.0

    limiter = config._rate_limiter
    if limiter.acquire():
        return True, 0.0

    wait_time = limiter.wait_time()
    return False, wait_time
