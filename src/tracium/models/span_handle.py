"""
Span handle and context for managing agent spans.

All operations are designed to be non-blocking and fail-safe.
SDK errors will never break user applications.
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime
from types import TracebackType
from typing import TYPE_CHECKING, Any

from ..helpers.logging_config import get_logger
from ..helpers.validation import (
    validate_error_message,
    validate_tags,
)
from ..models.trace_state import TraceState
from ..utils.datetime_utils import _duration_ms, _format_exception, _isoformat, _utcnow
from ..utils.tags import _merge_tags, _normalize_tags
from ..utils.validation import _validate_and_log

if TYPE_CHECKING:
    pass

logger = get_logger()


class AgentSpanHandle:
    """
    Handle returned when recording a span inside an agent trace.

    All operations are designed to fail gracefully and never raise exceptions.
    """

    def __init__(self, context: AgentSpanContext) -> None:
        self._context = context

    @property
    def id(self) -> str:
        return self._context.span_id

    @property
    def trace_id(self) -> str:
        return self._context.state.trace_id

    @property
    def parent_span_id(self) -> str | None:
        return self._context.parent_span_id

    @property
    def payload(self) -> dict[str, Any]:
        return dict(self._context.last_payload or {})

    def record_input(self, input_data: Any) -> None:
        try:
            self._context.set_input(input_data)
        except Exception:
            pass

    def record_output(self, output_data: Any) -> None:
        try:
            self._context.set_output(output_data)
        except Exception:
            pass

    def add_metadata(self, metadata: Mapping[str, Any]) -> None:
        pass

    def add_tags(self, tags: Sequence[str]) -> None:
        try:
            validated = _validate_and_log("add_tags", validate_tags, tags)
            self._context.add_tags(validated)
        except Exception:
            pass

    def set_token_usage(
        self,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cached_input_tokens: int | None = None,
    ) -> None:
        try:
            self._context.set_token_usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_input_tokens,
            )
        except Exception:
            pass

    def mark_failed(
        self,
        error: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            validated_error = _validate_and_log("mark_failed", validate_error_message, error)
            self._context.set_failure(validated_error)
        except Exception:
            try:
                self._context.set_failure(str(error)[:10000] if error else "Unknown error")
            except Exception:
                pass

    def set_status(self, status: str) -> None:
        try:
            self._context.set_status(status)
        except Exception:
            pass


class AgentSpanContext(contextlib.AbstractContextManager["AgentSpanHandle"]):
    """
    Context manager for agent spans.

    All operations are designed to be non-blocking and never raise exceptions.
    """

    def __init__(
        self,
        *,
        state: TraceState,
        span_type: str,
        name: str | None,
        input_payload: Any | None,
        tags: Sequence[str] | None,
        parent_span_id: str | None,
        span_id: str | None,
        depth_level: int | None = None,
        parallel_group_id: str | None = None,
        sequence_number: int | None = None,
        latency_ms: int | None = None,
        model_id: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cached_input_tokens: int | None = None,
    ) -> None:
        self.state = state
        self.span_type = span_type
        self.name = name
        self._input = input_payload
        self._output: Any | None = None
        self._tags: list[str] = _normalize_tags(tags)
        self.parent_span_id = parent_span_id or state.current_parent_span_id
        self.span_id = span_id or str(uuid.uuid4())
        self.depth_level = depth_level if depth_level is not None else state.current_depth_level

        try:
            from ..helpers.parallel_tracker import register_span_creation
            detected_group_id, detected_seq_num = register_span_creation(
                span_id=self.span_id,
                parent_span_id=self.parent_span_id,
                provided_parallel_group_id=parallel_group_id,
                provided_sequence_number=sequence_number,
            )
            self.parallel_group_id = parallel_group_id or detected_group_id
            self.sequence_number = sequence_number if sequence_number is not None else detected_seq_num
        except Exception:
            self.parallel_group_id = parallel_group_id
            self.sequence_number = sequence_number

        self._latency_ms = latency_ms
        self._model_id = model_id
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._cached_input_tokens = cached_input_tokens
        self._status = "in_progress"
        self._error: str | None = None
        self._start_time = _utcnow()
        self._handle = AgentSpanHandle(self)
        self.last_payload: dict[str, Any] | None = None
        self._initially_sent = False

    def _get_tenant_id(self) -> str | None:
        try:
            from ..context.tenant_context import get_current_tenant
            return get_current_tenant()
        except Exception:
            return None

    def _build_base_payload(
        self, status: str, started_at: datetime, completed_at: datetime | None = None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.span_id,
            "span_type": self.span_type,
            "status": status,
            "started_at": started_at.isoformat(),
        }

        if completed_at:
            payload["completed_at"] = completed_at.isoformat()

        if self.name:
            payload["name"] = self.name

        if self.parent_span_id:
            payload["parent_span_id"] = self.parent_span_id
        if self.depth_level is not None:
            payload["depth_level"] = self.depth_level
        if self.parallel_group_id:
            payload["parallel_group_id"] = self.parallel_group_id
        if self.sequence_number is not None:
            payload["sequence_number"] = self.sequence_number

        model_id = self._model_id or self.state.model_id
        if model_id:
            payload["model_id"] = model_id

        tenant_id = self._get_tenant_id()
        if tenant_id:
            payload["tenant_id"] = str(tenant_id)[:255]

        return payload

    def __enter__(self) -> AgentSpanHandle:
        try:
            self.state.push_span(self.span_id)

            try:
                if self.parallel_group_id is None:
                    from ..helpers.parallel_tracker import recheck_parallelism_for_span
                    detected_group_id, detected_seq_num = recheck_parallelism_for_span(
                        span_id=self.span_id,
                        parent_span_id=self.parent_span_id,
                    )
                    if detected_group_id is not None:
                        self.parallel_group_id = detected_group_id
                    if detected_seq_num is not None:
                        self.sequence_number = detected_seq_num
            except Exception:
                pass

            try:
                if self.parent_span_id:
                    from ..utils.span_registry import ensure_parent_span_sent
                    ensure_parent_span_sent(self.state.trace_id, self.parent_span_id, self.state.client)
            except Exception:
                pass

            initial_payload = self._build_base_payload("in_progress", self._start_time)

            if self._input is not None:
                initial_payload["input"] = self._input

            try:
                self.state.client.record_agent_spans(self.state.trace_id, [initial_payload])
                self._initially_sent = True

                from ..utils.span_registry import mark_span_sent
                mark_span_sent(self.state.trace_id, self.span_id)
            except Exception:
                pass

        except Exception as e:
            logger.debug(f"Span __enter__ failed (continuing): {type(e).__name__}: {e}")

        return self._handle

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        try:
            if exc_type is not None and self._error is None:
                try:
                    self._error = _format_exception(exc_type, exc_value, exc_tb)
                except Exception:
                    self._error = str(exc_value) if exc_value else "Unknown error"
            if exc_type is not None and self._status != "failed":
                self._status = "failed"

            end_time = _utcnow()
            if self._latency_ms is not None:
                latency_ms = self._latency_ms
            else:
                try:
                    latency_ms = _duration_ms(self._start_time, end_time)
                except Exception:
                    latency_ms = None

            status = (
                "failed"
                if self._error
                else (self._status if self._status != "in_progress" else "completed")
            )

            api_payload = self._build_base_payload(status, self._start_time, end_time)

            if self._input is not None:
                api_payload["input"] = self._input
            if self._output is not None:
                api_payload["output"] = self._output
            if self._error:
                api_payload["error"] = self._error
            if latency_ms is not None:
                api_payload["latency_ms"] = latency_ms

            if self._input_tokens is not None:
                api_payload["input_tokens"] = self._input_tokens
            if self._output_tokens is not None:
                api_payload["output_tokens"] = self._output_tokens
            if self._cached_input_tokens is not None:
                api_payload["cached_input_tokens"] = self._cached_input_tokens

            payload: dict[str, Any] = {
                "span_id": self.span_id,
                "type": self.span_type,
                "status": status,
                "started_at": _isoformat(self._start_time),
                "ended_at": _isoformat(end_time),
                "latency_ms": latency_ms,
            }
            if self.name:
                payload["name"] = self.name
            if self.parent_span_id:
                payload["parent_span_id"] = self.parent_span_id
            if self._input is not None:
                payload["input"] = self._input
            if self._output is not None:
                payload["output"] = self._output
            if self._tags:
                payload["tags"] = self._tags
            if self._error:
                payload["error"] = self._error
            model_id = self._model_id or self.state.model_id
            if model_id:
                payload["model_id"] = model_id
            tenant_id = self._get_tenant_id()
            if tenant_id:
                payload["tenant_id"] = str(tenant_id)[:255]

            try:
                self.state.client.record_agent_spans(self.state.trace_id, [api_payload])
                self.last_payload = payload
            except Exception:
                self.last_payload = payload

        except Exception as e:
            logger.debug(f"Span __exit__ failed (ignored): {type(e).__name__}: {e}")
        finally:
            try:
                self.state.pop_span()
            except Exception:
                pass

    def set_input(self, input_data: Any) -> None:
        self._input = input_data

    def set_output(self, output_data: Any) -> None:
        self._output = output_data

    def update_metadata(self, metadata: Mapping[str, Any]) -> None:
        pass

    def add_tags(self, tags: Sequence[str]) -> None:
        try:
            if not tags:
                return
            validated = _validate_and_log("AgentSpanContext.add_tags", validate_tags, tags)
            self._tags = _merge_tags(self._tags, validated)
        except Exception:
            pass

    def set_token_usage(
        self,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cached_input_tokens: int | None = None,
    ) -> None:
        if input_tokens is not None:
            try:
                self._input_tokens = int(input_tokens)
            except (TypeError, ValueError):
                pass
        if output_tokens is not None:
            try:
                self._output_tokens = int(output_tokens)
            except (TypeError, ValueError):
                pass
        if cached_input_tokens is not None:
            try:
                self._cached_input_tokens = int(cached_input_tokens)
            except (TypeError, ValueError):
                pass

    def set_failure(
        self,
        error: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            validated_error = _validate_and_log(
                "AgentSpanContext.set_failure", validate_error_message, error
            )
            self._status = "failed"
            self._error = validated_error
        except Exception:
            self._status = "failed"
            self._error = str(error)[:10000] if error else "Unknown error"

    def set_status(self, status: str) -> None:
        self._status = status
