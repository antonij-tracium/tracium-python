"""
Date and time utility functions.
"""

import traceback
from collections.abc import Mapping
from datetime import datetime, timezone
from types import TracebackType
from typing import Any


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat(dt: datetime) -> str:
    return dt.isoformat(timespec="milliseconds")


def _duration_ms(start: datetime, end: datetime) -> int:
    return max(int((end - start).total_seconds() * 1000), 0)


def _format_exception(
    exc_type: type[BaseException] | None,
    exc_value: BaseException | None,
    exc_tb: TracebackType | None,
) -> str:
    if exc_type is None:
        return ""
    return "".join(traceback.format_exception(exc_type, exc_value, exc_tb))


def _copy_mapping(mapping: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(mapping) if mapping else {}
