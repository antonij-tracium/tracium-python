"""
Sanitize span input/output payloads to avoid sending large documents and base64 to the backend.

Replaces large or binary content with short summaries so users still see useful context
(e.g. that a document was used, page count, prompt text, LLM response) without payload errors.
"""

from __future__ import annotations

import base64
import json
import re
from typing import Any

# Max total size (UTF-8 bytes) for span input or output when sent to the backend.
MAX_SPAN_IO_BYTES = 100_000
# Max size for a single string before truncation.
MAX_STRING_BYTES = 100_000
BASE64_MIN_BYTES = 2_000
MAX_DEPTH = 50

PLACEHOLDER_IMAGE_OR_DATA = "[image/data omitted for size]"
PLACEHOLDER_TRUNCATED = " (truncated for size)"
PLACEHOLDER_OVER_SIZE = "[content omitted: exceeded 100KB]"


def _is_likely_base64_or_data_url(s: str) -> bool:
    if not isinstance(s, str) or len(s) < BASE64_MIN_BYTES:
        return False
    s_strip = s.strip()
    if s_strip.startswith("data:"):
        return True
    try:
        if re.match(r"^[A-Za-z0-9+/=]+$", s_strip) and len(s_strip) >= BASE64_MIN_BYTES:
            base64.b64decode(s_strip, validate=True)
            return True
    except Exception:
        pass
    return False


def _truncate_string(s: str, max_bytes: int) -> str:
    encoded = s.encode("utf-8")
    if len(encoded) <= max_bytes:
        return s
    suffix = PLACEHOLDER_TRUNCATED
    suffix_bytes = suffix.encode("utf-8")
    keep = max_bytes - len(suffix_bytes)
    if keep <= 0:
        return suffix
    truncated = encoded[:keep]
    # Back up to last valid UTF-8 boundary
    while truncated and (truncated[-1] & 0xC0) == 0x80:
        truncated = truncated[:-1]
    try:
        return truncated.decode("utf-8") + suffix
    except Exception:
        return s[: max_bytes // 4] + suffix


def _sanitize_value(value: Any, depth: int) -> Any:
    if depth > MAX_DEPTH:
        return value

    if isinstance(value, str):
        if _is_likely_base64_or_data_url(value):
            return PLACEHOLDER_IMAGE_OR_DATA
        return _truncate_string(value, MAX_STRING_BYTES)

    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if k == "image_url" and isinstance(v, dict) and isinstance(v.get("url"), str):
                url = v["url"]
                if _is_likely_base64_or_data_url(url):
                    out[k] = {"url": PLACEHOLDER_IMAGE_OR_DATA}
                else:
                    out[k] = _sanitize_value(v, depth + 1)
            elif k == "url" and isinstance(v, str) and _is_likely_base64_or_data_url(v):
                out[k] = PLACEHOLDER_IMAGE_OR_DATA
            else:
                out[k] = _sanitize_value(v, depth + 1)
        return out

    if isinstance(value, list):
        sanitized_list: list[Any] = []
        for item in value:
            sanitized_list.append(_sanitize_value(item, depth + 1))
        return sanitized_list

    return value


def sanitize_span_io(value: Any) -> Any:
    """
    Return a copy of value safe to send as span input or output.

    - Replaces base64 and data-URL content with a short placeholder.
    - Truncates long strings. Total result is capped at MAX_SPAN_IO_BYTES (100KB).
    - Recursively processes dicts and lists.
    - Never mutates the original; never raises.
    """
    if value is None:
        return None
    try:
        out = _sanitize_value(value, 0)
        payload = json.dumps(out, default=str)
        if len(payload.encode("utf-8")) > MAX_SPAN_IO_BYTES:
            return PLACEHOLDER_OVER_SIZE
        return out
    except Exception:
        return value
