"""Utility functions for Anthropic integration."""

from __future__ import annotations

from typing import Any


def normalize_messages(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any] | None:
    """Extract messages from Anthropic API call."""
    if "messages" in kwargs:
        return {"messages": kwargs["messages"]}
    if args and isinstance(args[0], dict) and "messages" in args[0]:
        return {"messages": args[0]["messages"]}
    return None


def extract_model(kwargs: dict[str, Any]) -> str | None:
    """Extract model from Anthropic API call."""
    model = kwargs.get("model")
    return model if isinstance(model, str) else None


def extract_usage(usage: Any) -> tuple[int | None, int | None]:
    """Extract token usage from usage object."""
    usage_dict = (
        usage
        if isinstance(usage, dict)
        else (
            getattr(usage, "model_dump", lambda: None)()
            or getattr(usage, "dict", lambda: None)()
        )
    )
    if usage_dict:
        return usage_dict.get("input_tokens"), usage_dict.get("output_tokens")
    return None, None


def extract_output_data(response: Any) -> Any:
    """Extract output data from Anthropic response."""
    try:
        if hasattr(response, "content") and response.content:
            if isinstance(response.content, list):
                text_parts = [
                    block.text if hasattr(block, "text") else block.get("text", "")
                    for block in response.content
                    if hasattr(block, "text") or (isinstance(block, dict) and "text" in block)
                ]
                if text_parts:
                    return "\n".join(text_parts) if len(text_parts) > 1 else text_parts[0]
            elif isinstance(response.content, str):
                return response.content
    except Exception:
        pass

    if hasattr(response, "model_dump"):
        try:
            return response.model_dump()
        except Exception:
            return str(response)
    elif hasattr(response, "dict"):
        try:
            return response.dict()
        except Exception:
            return str(response)
    return str(response)


