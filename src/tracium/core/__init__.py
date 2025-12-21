"""
Core Tracium client and configuration.
"""

from .client import TraciumClient, current_trace
from .config import TraciumClientConfig
from .version import __version__

__all__ = ["TraciumClient", "TraciumClientConfig", "__version__", "current_trace"]
