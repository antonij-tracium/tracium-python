"""
Configuration for Tracium client.
"""

from dataclasses import dataclass

from ..helpers.retry import RetryConfig
from ..helpers.security import SecurityConfig
from .version import __version__


@dataclass(slots=True)
class TraciumClientConfig:
    base_url: str = "https://api.tracium.ai"
    timeout: float = 10.0
    user_agent: str = f"TraciumSDK/{__version__}"
    retry_config: RetryConfig | None = None
    fail_open: bool = True
    security_config: SecurityConfig | None = None
