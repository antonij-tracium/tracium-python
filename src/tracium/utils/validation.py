"""
Validation utility functions.
"""

from typing import Any

from ..helpers.logging_config import get_logger

logger = get_logger()


def _validate_and_log(func_name: str, validator, value: Any) -> Any:
    """Helper to validate and log errors consistently."""
    try:
        return validator(value)
    except (ValueError, TypeError) as e:
        logger.error("Validation error in %s: %s", func_name, str(e))
        raise
