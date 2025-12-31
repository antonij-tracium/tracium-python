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
    span_stack: list[str] = field(default_factory=list)
    start_payload: dict[str, Any] = field(default_factory=dict)
    model_id: str | None = None
    workspace_id: str | None = None
    version: str | None = None
    remote_started: bool = False
    has_spans: bool = False

    def ensure_remote_started(self) -> None:
        """
        Ensure the trace has been started in the backend.

        For lazy-start traces, we defer the `/agents/traces` call until the first span.
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
            if isinstance(payload, dict):
                try:
                    self.start_payload = dict(payload)
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            # Avoid retry loops; worst case spans still attempt to send and fail safely.
            self.remote_started = True

    def push_span(self, span_id: str) -> None:
        self.span_stack.append(span_id)

    def pop_span(self) -> None:
        if self.span_stack:
            self.span_stack.pop()

    @property
    def current_parent_span_id(self) -> str | None:
        if not self.span_stack:
            return None
        return self.span_stack[-1]

    @property
    def current_depth_level(self) -> int:
        """Return current nesting depth (0 = root level, 1 = first level, etc.)."""
        return len(self.span_stack)

    def copy_for_thread(self) -> TraceState:
        """
        Create a thread-safe copy of this trace state.

        This is used when propagating context to threads to ensure each thread
        has its own span_stack, preventing race conditions.

        Returns:
            A new TraceState with a fresh span_stack but sharing other attributes.
        """
        return TraceState(
            client=self.client,
            trace_id=self.trace_id,
            agent_name=self.agent_name,
            tags=self.tags.copy(),
            summary=self.summary,
            status=self.status,
            error=self.error,
            started_at=self.started_at,
            ended_at=self.ended_at,
            duration_ms=self.duration_ms,
            finished=self.finished,
            span_stack=self.span_stack.copy(),
            start_payload=self.start_payload.copy(),
            model_id=self.model_id,
            workspace_id=self.workspace_id,
            version=self.version,
            remote_started=self.remote_started,
            has_spans=self.has_spans,
        )
