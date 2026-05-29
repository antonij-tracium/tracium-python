"""
Trace state model for tracking agent traces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from ..utils.datetime_utils import _utcnow

if TYPE_CHECKING:
    from ..core.client import TraciumClient


@dataclass(frozen=True, slots=True)
class LLMTraceSummary:
    """Aggregated LLM metadata for a completed trace."""

    model: str | None = None
    system_prompt: str | None = None
    llm_steps: tuple[dict[str, Any], ...] | None = None


@dataclass
class TraceState:
    client: TraciumClient
    trace_id: str
    agent_name: str
    tags: list[str] = field(default_factory=list)
    summary: dict[str, Any] | None = None
    status: str = "in_progress"
    error: str | None = None
    started_at: datetime = field(default_factory=_utcnow)
    ended_at: datetime | None = None
    duration_ms: int | None = None
    finished: bool = False
    start_payload: dict[str, Any] = field(default_factory=dict)
    model_id: str | None = None
    workspace_id: str | None = None
    version: str | None = None
    remote_started: bool = False
    has_spans: bool = False
    _llm_info: list = field(default_factory=list, repr=False)

    def ensure_remote_started(self) -> None:
        """
        Ensure the trace has been started in the backend.

        For lazy-start traces, we defer the `/agents/traces` call until the first span.

        Only marks ``remote_started=True`` when the backend POST actually
        succeeded (response is a non-empty dict carrying an id). If the create
        call fail-opened to ``{}`` (backend unreachable / 5xx), we leave the
        flag False so the next span emission retries the create — otherwise
        we'd be sending spans for a trace the backend never created, and
        every POST would 404.
        """
        if self.remote_started:
            return
        try:
            payload = self.client.start_agent_trace(
                self.agent_name,
                model_id=self.model_id,
                tags=self.tags or None,
                trace_id=self.trace_id,
                workspace_id=self.workspace_id,
                version=self.version,
            )
        except Exception:
            return

        if not isinstance(payload, dict) or not payload:
            # Fail-open returned {} — create did not reach the backend.
            return
        # Confirm at least one identifying field is present. The real backend
        # response carries `id`/`trace_id` plus agent metadata.
        if not (payload.get("id") or payload.get("trace_id")):
            return
        try:
            self.start_payload = dict(payload)
        except Exception:
            pass
        self.remote_started = True
