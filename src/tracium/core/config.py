"""
Configuration for Tracium client.
"""

from dataclasses import dataclass

from ..helpers.retry import RetryConfig
from ..helpers.security import SecurityConfig
from .version import __version__


@dataclass(slots=True)
class TraciumClientConfig:
    """
    Configuration for the Tracium client.

    Attributes:
        base_url: Base URL for the Tracium API
        timeout: HTTP request timeout in seconds
        user_agent: User agent string for HTTP requests
        retry_config: Configuration for retry logic with exponential backoff
        fail_open: If True, SDK errors won't break user applications (recommended)
        security_config: Security configuration (rate limiting, data redaction)
        max_queue_size: Maximum number of queued telemetry events.
            Increase if you're generating high event volumes.
        block_on_full_queue: If True, block when queue is full instead of dropping events.
            This prevents event loss but may slow down your application.
        queue_warning_threshold: Warn when queue reaches this % of capacity (0.0-1.0)
        queue_timeout: Maximum time to wait when blocking on full queue (seconds)
    """
    base_url: str = "https://api.tracium.ai"
    timeout: float = 10.0
    user_agent: str = f"TraciumSDK/{__version__}"
    retry_config: RetryConfig | None = None
    fail_open: bool = True
    security_config: SecurityConfig | None = None

    max_queue_size: int = 10000
    block_on_full_queue: bool = False
    queue_warning_threshold: float = 0.8
    queue_timeout: float = 5.0
