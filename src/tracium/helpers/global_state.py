"""
Global shared state for the Tracium SDK auto-instrumentation layer.
"""

from __future__ import annotations

import contextvars
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..core import TraciumClient

_LANGCHAIN_ACTIVE_RUNS: contextvars.ContextVar[set[str]] = contextvars.ContextVar(
    "_langchain_active_runs", default=set()
)


@dataclass
class TraciumInitOptions:
    """
    Captures configuration derived from ``tracium.init``.
    """

    default_agent_name: str = "app"
    default_model_id: str | None = None
    default_version: str | None = None
    default_tags: list[str] = field(default_factory=list)
    default_metadata: dict[str, Any] = field(default_factory=dict)
    auto_instrument_langchain: bool = True
    auto_instrument_langgraph: bool = True
    auto_instrument_llm_clients: bool = True


@dataclass
class _GlobalState:
    client: TraciumClient | None = None
    options: TraciumInitOptions | None = None
    langchain_registered: bool = False
    langgraph_registered: bool = False
    openai_patched: bool = False
    anthropic_patched: bool = False
    google_patched: bool = False

    def reset(self) -> None:
        self.client = None
        self.options = None
        self.langchain_registered = False
        self.langgraph_registered = False
        self.openai_patched = False
        self.anthropic_patched = False
        self.google_patched = False


STATE = _GlobalState()


def set_client(client: TraciumClient, *, options: TraciumInitOptions) -> None:
    STATE.client = client
    STATE.options = options


def get_client() -> TraciumClient:
    if STATE.client is None:
        raise RuntimeError(
            "Tracium has not been initialized. Call tracium.init(api_key=...) first.",
        )
    return STATE.client


def get_options() -> TraciumInitOptions:
    if STATE.options is None:
        raise RuntimeError(
            "Tracium has not been initialized. Call tracium.init(api_key=...) first.",
        )
    return STATE.options


def get_default_tags(extra: Sequence[str] | None = None) -> list[str]:
    options = get_options()
    combined = list(options.default_tags)
    if extra:
        combined.extend(tag for tag in extra if tag and tag not in combined)
    return combined


def is_in_langchain_callback() -> bool:
    """Check if we're currently inside a LangChain callback handler."""
    active_runs = _LANGCHAIN_ACTIVE_RUNS.get()
    return len(active_runs) > 0


def add_langchain_active_run(run_id: str) -> None:
    """Add a LangChain run ID to the active set."""
    active_runs = _LANGCHAIN_ACTIVE_RUNS.get()
    new_runs = active_runs.copy() if active_runs else set()
    new_runs.add(run_id)
    _LANGCHAIN_ACTIVE_RUNS.set(new_runs)


def remove_langchain_active_run(run_id: str) -> None:
    """Remove a LangChain run ID from the active set."""
    active_runs = _LANGCHAIN_ACTIVE_RUNS.get()
    if active_runs and run_id in active_runs:
        new_runs = active_runs.copy()
        new_runs.remove(run_id)
        _LANGCHAIN_ACTIVE_RUNS.set(new_runs)
