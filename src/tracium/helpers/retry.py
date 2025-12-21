"""
Retry logic with exponential backoff for Tracium SDK API requests.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import httpx

T = TypeVar("T")


@dataclass
class RetryConfig:
    """
    Configuration for retry behavior.

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        backoff_factor: Multiplier for exponential backoff (default: 1.0)
        initial_delay: Initial delay in seconds before first retry (default: 0.1)
        max_delay: Maximum delay in seconds between retries (default: 60.0)
        retryable_status_codes: HTTP status codes that should trigger retries
        retryable_exceptions: Exception types that should trigger retries
    """

    max_retries: int = 3
    backoff_factor: float = 1.0
    initial_delay: float = 0.1
    max_delay: float = 60.0
    retryable_status_codes: set[int] = None
    retryable_exceptions: tuple[type[Exception], ...] = None

    def __post_init__(self) -> None:
        if self.retryable_status_codes is None:
            self.retryable_status_codes = {429, 500, 502, 503, 504}
        if self.retryable_exceptions is None:
            self.retryable_exceptions = (
                httpx.ConnectTimeout,
                httpx.ReadTimeout,
                httpx.NetworkError,
                httpx.ConnectError,
                httpx.PoolTimeout,
            )


def should_retry(
    exception: Exception | None,
    status_code: int | None,
    config: RetryConfig,
) -> bool:
    """
    Determine if a request should be retried based on exception or status code.

    Args:
        exception: The exception that occurred, if any
        status_code: HTTP status code from response, if any
        config: Retry configuration

    Returns:
        True if the request should be retried, False otherwise
    """
    if exception is not None:
        return isinstance(exception, config.retryable_exceptions)

    if status_code is not None:
        return status_code in config.retryable_status_codes

    return False


def calculate_backoff_delay(attempt: int, config: RetryConfig) -> float:
    """
    Calculate the delay before the next retry attempt using exponential backoff.

    Args:
        attempt: The current attempt number (0-indexed)
        config: Retry configuration

    Returns:
        Delay in seconds
    """
    delay = config.initial_delay * (config.backoff_factor**attempt)
    return min(delay, config.max_delay)


def retry_with_backoff(
    func: Callable[[], T],
    config: RetryConfig,
    on_retry: Callable[[Exception | None, int, float], None] | None = None,
) -> T:
    """
    Execute a function with retry logic and exponential backoff.

    Args:
        func: The function to execute
        config: Retry configuration
        on_retry: Optional callback called before each retry
                  (exception, attempt_number, delay_seconds)

    Returns:
        The result of the function call

    Raises:
        The last exception if all retries are exhausted
    """
    last_exception: Exception | None = None

    for attempt in range(config.max_retries + 1):
        try:
            result = func()
            if hasattr(result, "status_code") and should_retry(None, result.status_code, config):
                result.raise_for_status()
            return result
        except httpx.HTTPStatusError as e:
            last_exception = e
            status_code = e.response.status_code if e.response else None
            if not should_retry(None, status_code, config):
                raise
        except Exception as e:
            last_exception = e
            if not should_retry(e, None, config):
                raise

        if attempt < config.max_retries:
            delay = calculate_backoff_delay(attempt, config)
            if on_retry:
                on_retry(last_exception, attempt + 1, delay)
            time.sleep(delay)

    if last_exception:
        raise last_exception
    raise RuntimeError("Retry logic failed without exception")
