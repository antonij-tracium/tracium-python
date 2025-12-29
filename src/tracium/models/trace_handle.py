"""
Trace handle and manager for managing agent traces.

All operations are designed to be non-blocking and fail-safe.
SDK errors will never break user applications.
"""

from __future__ import annotations

import contextlib
import contextvars
import uuid
from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core.client import TraciumClient

from ..helpers.logging_config import get_logger
from ..helpers.validation import (
    validate_agent_name,
    validate_error_message,
    validate_metadata,
    validate_name,
    validate_span_id,
    validate_span_type,
    validate_tags,
    validate_trace_id,
)
from ..models.span_handle import AgentSpanContext
from ..models.trace_state import TraceState
from ..utils.datetime_utils import _copy_mapping, _duration_ms, _format_exception, _utcnow
from ..utils.tags import _merge_tags, _normalize_tags
from ..utils.validation import _validate_and_log

logger = get_logger()


def _safe_execute(operation_name: str, func: Any, default: Any = None) -> Any:
    """
    Safely execute a function, catching all exceptions.

    SDK operations should never break user code.
    """
    try:
        return func()
    except Exception as e:
        logger.debug(f"SDK operation '{operation_name}' failed (ignored): {type(e).__name__}: {e}")
        return default


class AgentTraceHandle:
    """
    Represents an in-flight Tracium agent trace. Provides helpers for mutating
    metadata and recording spans.

    All operations are designed to fail gracefully and never raise exceptions
    that could break user applications.
    """

    def __init__(self, state: TraceState) -> None:
        self._state = state

    @property
    def id(self) -> str:
        return self._state.trace_id

    @property
    def agent_name(self) -> str:
        return self._state.agent_name

    @property
    def tags(self) -> list[str]:
        return self._state.tags

    @property
    def summary(self) -> dict[str, Any] | None:
        return self._state.summary

    @property
    def start_payload(self) -> dict[str, Any]:
        return self._state.start_payload

    def set_summary(self, summary: Mapping[str, Any]) -> None:
        try:
            self._state.summary = _copy_mapping(summary)
        except Exception:
            pass

    def update_metadata(self, metadata: Mapping[str, Any]) -> None:
        pass

    def add_metadata_kv(self, key: str, value: Any) -> None:
        pass

    def add_tags(self, tags: Sequence[str]) -> None:
        try:
            if not tags:
                return
            validated = _validate_and_log("AgentTraceHandle.add_tags", validate_tags, tags)
            self._state.tags = _merge_tags(self._state.tags, validated)
        except Exception:
            pass

    def mark_failed(
        self,
        error: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            validated_error = _validate_and_log(
                "AgentTraceHandle.mark_failed", validate_error_message, error
            )
            validated_metadata = (
                _validate_and_log(
                    "AgentTraceHandle.mark_failed",
                    validate_metadata,
                    dict(metadata) if metadata else None,
                )
                if metadata
                else None
            )
            self._state.status = "failed"
            self._state.error = validated_error
            if validated_metadata:
                self.update_metadata(validated_metadata)
        except Exception:
            self._state.status = "failed"
            self._state.error = str(error)[:10000] if error else "Unknown error"

    def finish(
        self,
        *,
        summary: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        tags: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """
        Finish the trace. Non-blocking - sends completion asynchronously.

        Returns a status dict (may not reflect actual API response).
        """
        try:
            if self._state.finished:
                return {"trace_id": self._state.trace_id, "status": "already_finished"}

            if summary is not None:
                self.set_summary(summary)
            if tags:
                self.add_tags(tags)
            self._state.status = "completed"
            self._state.ended_at = _utcnow()
            self._state.duration_ms = _duration_ms(self._state.started_at, self._state.ended_at)

            payload = self._state.client.complete_agent_trace(
                self._state.trace_id,
                summary=self._state.summary,
                tags=self._state.tags or None,
            )
            self._state.finished = True
            return (
                payload
                if isinstance(payload, dict)
                else {"trace_id": self._state.trace_id, "status": "completed"}
            )
        except Exception as e:
            logger.debug(f"finish() failed (ignored): {type(e).__name__}: {e}")
            self._state.finished = True
            return {"trace_id": self._state.trace_id, "status": "completed"}

    def fail(
        self,
        *,
        error: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Mark the trace as failed. Non-blocking - sends failure asynchronously.

        Returns a status dict (may not reflect actual API response).
        """
        try:
            if self._state.finished:
                return {"trace_id": self._state.trace_id, "status": "already_finished"}

            self._state.status = "failed"
            self._state.error = error
            self._state.ended_at = _utcnow()
            self._state.duration_ms = _duration_ms(self._state.started_at, self._state.ended_at)

            payload = self._state.client.fail_agent_trace(
                self._state.trace_id,
                error=error,
            )
            self._state.finished = True
            return (
                payload
                if isinstance(payload, dict)
                else {"trace_id": self._state.trace_id, "status": "failed"}
            )
        except Exception as e:
            logger.debug(f"fail() failed (ignored): {type(e).__name__}: {e}")
            self._state.finished = True
            return {"trace_id": self._state.trace_id, "status": "failed"}

    def span(
        self,
        *,
        span_type: str,
        name: str | None = None,
        input: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
        tags: Sequence[str] | None = None,
        parent_span_id: str | None = None,
        span_id: str | None = None,
        depth_level: int | None = None,
        parallel_group_id: str | None = None,
        sequence_number: int | None = None,
        latency_ms: int | None = None,
        model_id: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cached_input_tokens: int | None = None,
    ) -> AgentSpanContext:
        try:
            validated_span_type = _validate_and_log("span", validate_span_type, span_type)
            validated_name = _validate_and_log("span", validate_name, name)
            validated_tags = _validate_and_log("span", validate_tags, tags)
            validated_parent_span_id = (
                _validate_and_log("span", validate_span_id, parent_span_id)
                if parent_span_id
                else None
            )
            validated_span_id = (
                _validate_and_log("span", validate_span_id, span_id) if span_id else None
            )

            return AgentSpanContext(
                state=self._state,
                span_type=validated_span_type,
                name=validated_name,
                input_payload=input,
                tags=validated_tags,
                parent_span_id=validated_parent_span_id,
                span_id=validated_span_id,
                depth_level=depth_level,
                parallel_group_id=parallel_group_id,
                sequence_number=sequence_number,
                latency_ms=latency_ms,
                model_id=model_id,
            )
        except Exception as e:
            logger.debug(f"span() creation failed (using fallback): {type(e).__name__}: {e}")
            return AgentSpanContext(
                state=self._state,
                span_type=str(span_type)[:256] if span_type else "unknown",
                name=str(name)[:256] if name else None,
                input_payload=input,
                tags=[],
                parent_span_id=parent_span_id,
                span_id=span_id,
                depth_level=depth_level,
                parallel_group_id=parallel_group_id,
                sequence_number=sequence_number,
                latency_ms=latency_ms,
                model_id=model_id,
            )

    def record_span(
        self,
        *,
        span_type: str,
        name: str | None = None,
        input: Any | None = None,
        output: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
        tags: Sequence[str] | None = None,
        parent_span_id: str | None = None,
        span_id: str | None = None,
        status: str | None = None,
        error: str | None = None,
        depth_level: int | None = None,
        parallel_group_id: str | None = None,
        sequence_number: int | None = None,
        latency_ms: int | None = None,
        model_id: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cached_input_tokens: int | None = None,
    ) -> dict[str, Any]:
        try:
            validated_error = (
                _validate_and_log("record_span", validate_error_message, error) if error else None
            )

            with self.span(
                span_type=span_type,
                name=name,
                input=input,
                tags=tags,
                parent_span_id=parent_span_id,
                span_id=span_id,
                depth_level=depth_level,
                parallel_group_id=parallel_group_id,
                sequence_number=sequence_number,
                latency_ms=latency_ms,
                model_id=model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_input_tokens,
            ) as handle:
                if output is not None:
                    handle.record_output(output)
                if status is not None and status != "completed":
                    handle.set_status(status)
                if validated_error is not None:
                    handle.mark_failed(validated_error)
            return handle.payload
        except Exception as e:
            logger.debug(f"record_span failed (returning empty): {type(e).__name__}: {e}")
            return {}


class AgentTraceManager(
    contextlib.AbstractContextManager[AgentTraceHandle],
    contextlib.AbstractAsyncContextManager[AgentTraceHandle],
):
    """
    Context manager for agent traces.

    Designed to never raise exceptions that could break user applications.
    All SDK errors are caught and logged.
    """

    def __init__(
        self,
        client: TraciumClient,
        *,
        agent_name: str,
        model_id: str | None,
        metadata: Mapping[str, Any] | None,
        tags: Sequence[str] | None,
        trace_id: str | None,
        workspace_id: str | None = None,
        version: str | None = None,
    ) -> None:
        self._client = client
        self._agent_name = agent_name
        self._model_id = model_id
        self._tags = _normalize_tags(tags)
        self._explicit_trace_id = trace_id
        self._workspace_id = workspace_id
        self._version = version
        self._state: TraceState | None = None
        self._token: contextvars.Token[TraceState | None] | None = None

    def _start(self) -> AgentTraceHandle:
        """Start the trace. Designed to never raise exceptions."""
        try:
            validated_agent_name = _validate_and_log(
                "AgentTraceManager._start", validate_agent_name, self._agent_name
            )
            validated_trace_id = (
                _validate_and_log(
                    "AgentTraceManager._start", validate_trace_id, self._explicit_trace_id
                )
                if self._explicit_trace_id
                else None
            )
            validated_tags = _validate_and_log(
                "AgentTraceManager._start", validate_tags, self._tags
            )

            self._agent_name = validated_agent_name
            self._explicit_trace_id = validated_trace_id
            self._tags = validated_tags

            response: dict[str, Any] | Any = self._client.start_agent_trace(
                validated_agent_name,
                model_id=self._model_id,
                tags=validated_tags or None,
                trace_id=validated_trace_id,
                workspace_id=self._workspace_id,
                version=self._version,
            )

            trace_id: str | None = None
            if isinstance(response, dict):
                trace_id = response.get("id") or response.get("trace_id")
            if not trace_id:
                trace_id = self._explicit_trace_id
            if not trace_id:
                trace_id = str(uuid.uuid4())
                logger.debug(
                    "Generated local trace ID (API response unavailable)",
                    extra={"trace_id": trace_id, "agent_name": validated_agent_name},
                )

            response_dict = dict(response) if isinstance(response, dict) else {}
            trace_model_id = response_dict.get("model_id") or self._model_id

            state = TraceState(
                client=self._client,
                trace_id=str(trace_id),
                agent_name=validated_agent_name,
                tags=list(validated_tags),
                start_payload=response_dict,
                model_id=trace_model_id,
            )
            self._state = state

            try:
                from ..context.trace_context import CURRENT_TRACE_STATE

                self._token = CURRENT_TRACE_STATE.set(state)
            except Exception:
                pass

            return AgentTraceHandle(state)
        except Exception as e:
            logger.debug(f"Trace start failed (using fallback): {type(e).__name__}: {e}")
            fallback_trace_id = self._explicit_trace_id or str(uuid.uuid4())
            fallback_agent_name = str(self._agent_name)[:256] if self._agent_name else "unknown"

            state = TraceState(
                client=self._client,
                trace_id=fallback_trace_id,
                agent_name=fallback_agent_name,
                tags=list(self._tags) if self._tags else [],
                start_payload={},
                model_id=self._model_id,
            )
            self._state = state

            try:
                from ..context.trace_context import CURRENT_TRACE_STATE

                self._token = CURRENT_TRACE_STATE.set(state)
            except Exception:
                pass

            return AgentTraceHandle(state)

    def __enter__(self) -> AgentTraceHandle:
        return self._start()

    async def __aenter__(self) -> AgentTraceHandle:
        return self._start()

    def _finish(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Finish the trace. Never raises exceptions."""
        try:
            state = self._state
            if state is None or state.finished:
                return

            if exc_type is not None:
                try:
                    formatted = _format_exception(exc_type, exc_value, exc_tb)
                    if formatted:
                        state.error = formatted
                except Exception:
                    state.error = str(exc_value) if exc_value else "Unknown error"
                state.status = "failed"

            state.ended_at = _utcnow()
            try:
                state.duration_ms = _duration_ms(state.started_at, state.ended_at)
            except Exception:
                pass

            if state.status == "failed":
                self._client.fail_agent_trace(
                    state.trace_id,
                    error=state.error,
                )
            else:
                self._client.complete_agent_trace(
                    state.trace_id,
                    summary=state.summary,
                    tags=state.tags or None,
                )
                state.status = "completed"

            state.finished = True
        except Exception as e:
            logger.debug(f"Trace finish failed (ignored): {type(e).__name__}: {e}")
            if self._state:
                self._state.finished = True

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        try:
            self._finish(exc_type, exc_value, exc_tb)
        finally:
            try:
                if self._token is not None:
                    from ..context.trace_context import CURRENT_TRACE_STATE

                    CURRENT_TRACE_STATE.reset(self._token)
            except Exception:
                pass

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.__exit__(exc_type, exc_value, exc_tb)
