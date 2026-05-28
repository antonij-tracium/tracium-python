"""Helpers shared across LLM-provider integrations."""

from __future__ import annotations

from typing import Any


def extract_tools(kwargs: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Extract tool definitions from API call kwargs as a list of plain dicts.

    Accepts the tool list shape used by OpenAI, Anthropic, and Google: each
    element may be a dict, a Pydantic model (``model_dump``), an object with
    ``to_dict``, an object with ``dict``, or anything else — falling back to
    ``{"raw": str(tool)}`` so unknown shapes are still captured.

    Returns ``None`` when no tools are present or extraction fails.
    """
    try:
        tools = kwargs.get("tools")
        if not tools or not isinstance(tools, list):
            return None
        result: list[dict[str, Any]] = []
        for tool in tools:
            if isinstance(tool, dict):
                result.append(tool)
            elif hasattr(tool, "model_dump"):
                result.append(tool.model_dump())
            elif hasattr(tool, "to_dict"):
                result.append(tool.to_dict())
            elif hasattr(tool, "dict"):
                result.append(tool.dict())
            else:
                result.append({"raw": str(tool)})
        return result or None
    except Exception:
        return None
