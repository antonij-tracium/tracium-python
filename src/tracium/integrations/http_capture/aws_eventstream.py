"""
Minimal parser for AWS's ``application/vnd.amazon.eventstream`` framing.

Bedrock returns streaming bodies in this format (used by both
``invoke-with-response-stream`` and ``converse-stream``). We need to decode the
framing to get at the JSON payloads inside; once we have those, the same
provider parsers handle them.

The framing is documented at
https://docs.aws.amazon.com/AmazonS3/latest/API/RESTSelectObjectAppendix.html
— each message looks like::

    +--------------------------------+
    | Total length (4 bytes, BE)     |
    +--------------------------------+
    | Headers length (4 bytes, BE)   |
    +--------------------------------+
    | Prelude CRC (4 bytes)          |
    +--------------------------------+
    | Headers (variable)             |
    +--------------------------------+
    | Payload (variable)             |
    +--------------------------------+
    | Message CRC (4 bytes)          |
    +--------------------------------+

We deliberately skip CRC validation — we're observing, not reading authoritatively,
and any bytes that don't parse cleanly just get dropped. The dashboard's view of
a stream is at worst slightly incomplete, never corrupt.
"""

from __future__ import annotations

import base64
import json
import struct
from collections.abc import Iterator
from typing import Any

_PRELUDE_SIZE = 12  # total_length (4) + headers_length (4) + prelude_crc (4)
_TRAILER_SIZE = 4  # message CRC


def iter_messages(buffer: bytes) -> Iterator[tuple[dict[str, Any], bytes]]:
    """Yield ``(headers, payload_bytes)`` for each complete message in *buffer*.

    Stops at the first incomplete frame (caller can re-feed accumulated bytes
    later). Errors in a single frame are swallowed so a malformed message
    doesn't kill the rest of the stream.
    """
    pos = 0
    while pos + _PRELUDE_SIZE <= len(buffer):
        try:
            total_length, headers_length = struct.unpack(">II", buffer[pos : pos + 8])
        except struct.error:
            return
        if total_length <= 0 or total_length > len(buffer) - pos:
            return  # incomplete frame
        if total_length < _PRELUDE_SIZE + _TRAILER_SIZE + headers_length:
            pos += total_length
            continue

        headers_start = pos + _PRELUDE_SIZE
        headers_end = headers_start + headers_length
        payload_end = pos + total_length - _TRAILER_SIZE
        if payload_end < headers_end or payload_end > len(buffer):
            pos += total_length
            continue

        try:
            headers = _parse_headers(buffer[headers_start:headers_end])
        except Exception:
            headers = {}
        payload = buffer[headers_end:payload_end]

        yield headers, payload
        pos += total_length


def decode_bedrock_payload(headers: dict[str, Any], payload: bytes) -> dict[str, Any] | None:
    """Return the inner JSON event from a Bedrock streaming chunk.

    Bedrock wraps each inner event as ``{"bytes": "<base64-encoded JSON>"}``
    inside the eventstream payload. We unwrap, decode, and parse. Non-``chunk``
    events (e.g. ``exception``) are returned as their decoded JSON when
    possible, otherwise ``None``.
    """
    event_type = headers.get(":event-type")
    if isinstance(payload, bytes | bytearray) and payload:
        try:
            outer = json.loads(payload)
        except Exception:
            return None
        if isinstance(outer, dict):
            inner_b64 = outer.get("bytes")
            if isinstance(inner_b64, str):
                try:
                    inner = json.loads(base64.b64decode(inner_b64))
                    if isinstance(inner, dict):
                        if event_type and "_type" not in inner and "type" not in inner:
                            inner["_eventstream_event"] = event_type
                        return inner
                except Exception:
                    return None
            return outer
    return None


def iter_bedrock_events(buffer: bytes) -> Iterator[dict[str, Any]]:
    """Convenience: yield decoded inner JSON events for a Bedrock byte stream."""
    for headers, payload in iter_messages(buffer):
        event = decode_bedrock_payload(headers, payload)
        if event is not None:
            yield event


# --------------------------------------------------------------------------- #
# Header parsing                                                               #
# --------------------------------------------------------------------------- #


def _parse_headers(blob: bytes) -> dict[str, Any]:
    """Decode the headers section of an eventstream message.

    We only handle the value types Bedrock actually uses (string, byte array);
    others are skipped so an unknown type doesn't blow up the whole parse.
    """
    headers: dict[str, Any] = {}
    pos = 0
    while pos < len(blob):
        if pos + 1 > len(blob):
            break
        name_len = blob[pos]
        pos += 1
        if pos + name_len > len(blob):
            break
        name = blob[pos : pos + name_len].decode("utf-8", errors="replace")
        pos += name_len
        if pos + 1 > len(blob):
            break
        value_type = blob[pos]
        pos += 1
        # Type 7 = string (most common for Bedrock); type 6 = byte array.
        if value_type in (6, 7):
            if pos + 2 > len(blob):
                break
            (value_len,) = struct.unpack(">H", blob[pos : pos + 2])
            pos += 2
            if pos + value_len > len(blob):
                break
            value_bytes = blob[pos : pos + value_len]
            pos += value_len
            if value_type == 7:
                headers[name] = value_bytes.decode("utf-8", errors="replace")
            else:
                headers[name] = value_bytes
        else:
            # Unsupported type — give up on the rest of this header block.
            break
    return headers
