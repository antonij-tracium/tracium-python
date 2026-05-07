"""
Registry for linking tool calls across LLM spans.

When an LLM span returns tool_calls, their IDs are registered here so that
the next LLM call (which feeds in tool results) can automatically be linked
as a child span of the original call.

Thread-safe and works for both sync and async code.
"""

from __future__ import annotations

import threading
from typing import Any

_registry_lock = threading.Lock()
# Maps tool_use_id -> (trace_id, span_id)
_registry: dict[str, tuple[str, str]] = {}
_MAX_REGISTRY_SIZE = 2000


def _evict_if_needed() -> None:
    """Evict oldest half of entries when registry exceeds max size. Must hold lock."""
    if len(_registry) >= _MAX_REGISTRY_SIZE:
        keys = list(_registry.keys())
        for k in keys[: _MAX_REGISTRY_SIZE // 2]:
            del _registry[k]


def register_tool_calls(trace_id: str, span_id: str, tool_calls: list[dict[str, Any]]) -> None:
    """Register tool call IDs from an LLM span so continuations can be auto-linked."""
    try:
        with _registry_lock:
            _evict_if_needed()
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                # Anthropic: {"type": "tool_use", "id": "toolu_xxx", ...}
                tc_id = tc.get("id")
                # OpenAI function_call: {"type": "function_call", "call_id": "call_xxx", ...}
                if not tc_id:
                    tc_id = tc.get("call_id")
                # OpenAI chat completion: {"id": "call_xxx", "function": {...}}
                if not tc_id:
                    fn = tc.get("function")
                    if isinstance(fn, dict):
                        tc_id = fn.get("id")
                if tc_id and isinstance(tc_id, str):
                    _registry[tc_id] = (trace_id, span_id)
    except Exception:
        pass


def find_parent_span_id(trace_id: str, messages: list[Any]) -> str | None:
    """
    Look for tool_result blocks in messages and return the span_id that made those calls.

    Returns the span_id of the LLM span that originally requested the tool calls,
    or None if no match is found.
    """
    try:
        if not messages or not isinstance(messages, list):
            return None
        ids = _extract_tool_result_ids(messages)
        if not ids:
            return None
        with _registry_lock:
            for tc_id in ids:
                entry = _registry.get(tc_id)
                if entry and entry[0] == trace_id:
                    return entry[1]
    except Exception:
        pass
    return None


def _extract_tool_result_ids(messages: list[Any]) -> list[str]:
    """Extract tool_use_ids / tool_call_ids from tool_result content in messages."""
    ids: list[str] = []
    try:
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")

            # Anthropic: role="user", content=[{"type":"tool_result","tool_use_id":"..."}]
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        tid = block.get("tool_use_id")
                        if tid and isinstance(tid, str):
                            ids.append(tid)

            # OpenAI: role="tool", tool_call_id="call_xxx"
            role = msg.get("role")
            if role == "tool":
                tid = msg.get("tool_call_id")
                if tid and isinstance(tid, str):
                    ids.append(tid)

            # OpenAI Responses API: input items with type="function_call_output"
            if isinstance(content, str) and msg.get("type") == "function_call_output":
                tid = msg.get("call_id")
                if tid and isinstance(tid, str):
                    ids.append(tid)
    except Exception:
        pass
    return ids
