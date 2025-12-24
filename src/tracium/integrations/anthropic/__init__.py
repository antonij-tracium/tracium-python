"""
Auto-instrumentation for the Anthropic Python SDK (Claude).
"""

from .patch import patch_anthropic

__all__ = ["patch_anthropic"]

