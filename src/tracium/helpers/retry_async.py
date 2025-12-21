"""
Async retry logic with exponential backoff for Tracium SDK API requests.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx

from .retry import RetryConfig, calculate_backoff_delay, should_retry

T = TypeVar("T")


async def retry_with_backoff_async(
    func: Callable[[], Awaitable[T]],
    config: RetryConfig,
    on_retry: Callable[[Exception | None, int, float], None] | None = None,
) -> T:
    """
    Execute an async function with retry logic and exponential backoff.

    Args:
        func: The async function to execute
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
            result = await func()
            if hasattr(result, "status_code") and should_retry(None, result.status_code, config):
                result.raise_for_status()
            return result
        except httpx.HTTPStatusError as e:
            last_exception = e
            status_code = e.response.status_code if e.response else None
            if not should_retry(e, status_code, config):
                raise
        except Exception as e:
            last_exception = e
            if not should_retry(e, None, config):
                raise

        if attempt < config.max_retries:
            delay = calculate_backoff_delay(attempt, config)
            if on_retry:
                on_retry(last_exception, attempt + 1, delay)
            await asyncio.sleep(delay)

    if last_exception:
        raise last_exception
    raise RuntimeError("Retry logic failed without exception")
