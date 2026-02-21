"""
Check for SDK updates at init and log a one-time warning if outdated.
"""
from __future__ import annotations

import httpx
from packaging.version import Version

from ..core.version import __version__ as _current_version
from ..helpers.logging_config import get_logger

logger = get_logger()

_PYPI_URL = "https://pypi.org/pypi/tracium/json"
_warned = False


def check_for_update(*, is_test: bool = False) -> None:
    """
    Fetch latest version from PyPI and log a one-time warning if outdated.

    Skips in test scenarios. Runs synchronously at init to ensure the message
    appears when the user starts their service.
    """
    global _warned
    if _warned or is_test:
        return

    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(_PYPI_URL)
            resp.raise_for_status()
            data = resp.json()
            latest = data.get("info", {}).get("version")
            if not latest or not isinstance(latest, str):
                return

            cur = Version(_current_version)
            lat = Version(latest)
            if cur >= lat:
                return

            _warned = True
            logger.warning(
                "A new version of tracium (%s) is available. "
                "Upgrade with: pip install -U tracium",
                latest,
            )
    except Exception:
        pass  # Silent on failure - version check must not break init
