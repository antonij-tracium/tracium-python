"""
API endpoint methods for Tracium SDK.

All methods are designed to be non-blocking and fail-safe.
Telemetry data is sent asynchronously in the background.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

from ..context.tenant_context import get_current_tenant
from ..helpers.logging_config import get_logger
from ..helpers.validation import (
    validate_error_message,
    validate_span_id,
    validate_span_type,
    validate_tags,
    validate_trace_id,
)
from ..utils.validation import _validate_and_log

if TYPE_CHECKING:
    from .http_client import HTTPClient

logger = get_logger()


def _safe_execute(operation_name: str, func: Any, default: Any = None) -> Any:
    """
    Safely execute a function, catching all exceptions.

    Args:
        operation_name: Name of the operation for logging
        func: Function to execute
        default: Default value to return on error

    Returns:
        Function result or default value on error
    """
    try:
        return func()
    except Exception as e:
        logger.debug(f"SDK operation '{operation_name}' failed (ignored): {type(e).__name__}: {e}")
        return default


class TraciumAPIEndpoints:
    """
    API endpoint methods for Tracium.

    All telemetry operations are designed to:
    1. Never block user code for extended periods
    2. Never raise exceptions that could break user applications
    3. Send data asynchronously when possible
    """

    def __init__(self, http_client: HTTPClient) -> None:
        self._http = http_client

    def start_agent_trace(
        self,
        agent_name: str,
        *,
        model_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: Sequence[str] | None = None,
        trace_id: str | None = None,
        workspace_id: str | None = None,
        version: str | None = None,
    ) -> dict[str, Any]:
        """
        Start a new agent trace.

        This operation is synchronous as we need the trace ID from the response.
        However, it is designed to fail gracefully and return a usable response
        even on errors.
        """
        try:
            from ..helpers.validation import validate_agent_name

            validated_agent_name = _validate_and_log(
                "start_agent_trace", validate_agent_name, agent_name
            )
            validated_trace_id = (
                _validate_and_log("start_agent_trace", validate_trace_id, trace_id)
                if trace_id
                else None
            )
            validated_tags = _validate_and_log("start_agent_trace", validate_tags, tags)

            payload: dict[str, Any] = {"agent_name": validated_agent_name}
            if model_id:
                payload["model_id"] = str(model_id)[:256]
            if validated_tags:
                payload["tags"] = list(validated_tags)
            if validated_trace_id:
                payload["trace_id"] = validated_trace_id
            if workspace_id:
                payload["workspace_id"] = str(workspace_id)[:256]
            if version:
                payload["version"] = str(version)[:128]

            tenant_id = get_current_tenant()
            if tenant_id:
                payload["tenant_id"] = str(tenant_id)[:255]

            logger.debug(
                "Starting agent trace",
                extra={"agent_name": validated_agent_name, "trace_id": validated_trace_id},
            )
            result = self._http.post("/agents/traces", json=payload)
            if isinstance(result, dict):
                return result
            return {"id": validated_trace_id}
        except Exception as e:
            logger.debug(f"start_agent_trace failed (returning fallback): {type(e).__name__}: {e}")
            return {"id": trace_id} if trace_id else {}

    def record_agent_spans(
        self,
        trace_id: str,
        spans: Iterable[dict[str, Any]],
    ) -> Sequence[dict[str, Any]]:
        """
        Record agent spans asynchronously.

        This operation is non-blocking - spans are queued for background processing.
        """
        try:
            validated_trace_id = _validate_and_log(
                "record_agent_spans", validate_trace_id, trace_id
            )
            spans_list = list(spans)
            if not spans_list:
                return []
            if len(spans_list) > 1000:
                spans_list = spans_list[:1000]

            validated_spans: list[dict[str, Any]] = []
            for span in spans_list:
                if not isinstance(span, dict):
                    continue
                validated_span = dict(span)
                try:
                    if "span_type" in validated_span:
                        validated_span["span_type"] = validate_span_type(
                            str(validated_span["span_type"])
                        )
                    if "span_id" in validated_span:
                        validated_span["span_id"] = validate_span_id(str(validated_span["span_id"]))
                        del validated_span["span_id"]
                    if "parent_span_id" in validated_span:
                        validated_span["parent_span_id"] = validate_span_id(
                            str(validated_span["parent_span_id"])
                        )
                        del validated_span["parent_span_id"]
                except Exception:
                    pass
                validated_spans.append(validated_span)

            if not validated_spans:
                return []

            payload = {"spans": validated_spans}
            logger.debug(
                "Recording agent spans",
                extra={"trace_id": validated_trace_id, "span_count": len(validated_spans)},
            )

            self._http.post_async(f"/agents/traces/{validated_trace_id}/spans", json=payload)

            return validated_spans
        except Exception as e:
            logger.debug(f"record_agent_spans failed (ignored): {type(e).__name__}: {e}")
            return []

    def update_agent_span(
        self,
        trace_id: str,
        span_id: str,
        span_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Update an existing agent span asynchronously."""
        try:
            validated_trace_id = _validate_and_log("update_agent_span", validate_trace_id, trace_id)
            validated_span_id = _validate_and_log("update_agent_span", validate_span_id, span_id)

            validated_span = dict(span_data)
            validated_span["id"] = validated_span_id
            if "span_type" in validated_span:
                validated_span["span_type"] = validate_span_type(str(validated_span["span_type"]))

            payload = {"spans": [validated_span]}

            self._http.post_async(f"/agents/traces/{validated_trace_id}/spans", json=payload)

            return validated_span
        except Exception as e:
            logger.debug(f"update_agent_span failed (ignored): {type(e).__name__}: {e}")
            return {}

    def complete_agent_trace(
        self,
        trace_id: str,
        *,
        summary: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        tags: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Complete an agent trace asynchronously."""
        try:
            validated_trace_id = _validate_and_log(
                "complete_agent_trace", validate_trace_id, trace_id
            )
            validated_tags = _validate_and_log("complete_agent_trace", validate_tags, tags)

            payload: dict[str, Any] = {}
            if summary:
                payload["summary"] = summary
            if validated_tags:
                payload["tags"] = list(validated_tags)

            logger.debug("Completing agent trace", extra={"trace_id": validated_trace_id})

            self._http.post_async(f"/agents/traces/{validated_trace_id}/complete", json=payload)

            return {"trace_id": validated_trace_id, "status": "completed"}
        except Exception as e:
            logger.debug(f"complete_agent_trace failed (ignored): {type(e).__name__}: {e}")
            return {}

    def fail_agent_trace(
        self,
        trace_id: str,
        *,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Mark an agent trace as failed asynchronously."""
        try:
            validated_trace_id = _validate_and_log("fail_agent_trace", validate_trace_id, trace_id)
            validated_error = (
                _validate_and_log("fail_agent_trace", validate_error_message, error)
                if error
                else None
            )

            payload: dict[str, Any] = {}
            if validated_error:
                payload["error"] = validated_error

            logger.debug("Failing agent trace", extra={"trace_id": validated_trace_id})

            self._http.post_async(f"/agents/traces/{validated_trace_id}/fail", json=payload)

            return {"trace_id": validated_trace_id, "status": "failed"}
        except Exception as e:
            logger.debug(f"fail_agent_trace failed (ignored): {type(e).__name__}: {e}")
            return {}

    def trigger_drift_check(
        self,
        *,
        metrics: Sequence[str] | None = None,
        alert_channels: Sequence[str] | None = None,
        workspace_id: str | None = None,
    ) -> Sequence[dict[str, Any]]:
        """
        Manually trigger drift detection checks.

        This is a synchronous operation as the user may want the results.
        """
        try:
            params: dict[str, Any] = {}
            if metrics is not None:
                params["metrics"] = list(metrics)
            if alert_channels is not None:
                params["alert_channels"] = list(alert_channels)
            if workspace_id is not None:
                params["workspace_id"] = workspace_id
            result = self._http.post("/drift/check", params=params)
            if isinstance(result, list):
                return result
            if (
                isinstance(result, dict)
                and "results" in result
                and isinstance(result["results"], list)
            ):
                return result["results"]
            return [result] if result else []
        except Exception as e:
            logger.debug(f"trigger_drift_check failed (returning empty): {type(e).__name__}: {e}")
            return []

    def trigger_prompt_embeddings_drift_check(
        self,
        *,
        workspace_ids: Sequence[str] | None = None,
        alert_channels: Sequence[str] | None = None,
        similarity_threshold: float = 0.05,
        baseline_days: int = 60,
        current_days: int = 1,
    ) -> Sequence[dict[str, Any]]:
        """
        Manually trigger prompt embeddings drift detection.

        This is a synchronous operation as the user may want the results.
        """
        try:
            params: dict[str, Any] = {
                "similarity_threshold": similarity_threshold,
                "baseline_days": baseline_days,
                "current_days": current_days,
            }
            if workspace_ids is not None:
                params["workspace_ids"] = list(workspace_ids)
            if alert_channels is not None:
                params["alert_channels"] = list(alert_channels)
            result = self._http.post("/drift/prompt-embeddings/check", params=params)
            if isinstance(result, list):
                return result
            if (
                isinstance(result, dict)
                and "results" in result
                and isinstance(result["results"], list)
            ):
                return result["results"]
            return [result] if result else []
        except Exception as e:
            logger.debug(f"trigger_prompt_embeddings_drift_check failed: {type(e).__name__}: {e}")
            return []

    def get_current_user(self) -> dict[str, Any]:
        """
        Get current user information.

        This is a synchronous operation as we need the response.
        """
        try:
            user_info = self._http.get("/auth/me")
            if isinstance(user_info, dict):
                return {"plan": user_info.get("plan", "free")}
            return {"plan": "free"}
        except Exception as e:
            logger.debug(f"get_current_user failed (returning default): {type(e).__name__}: {e}")
            return {"plan": "free"}

    def get_gantt_data(self, trace_id: str) -> dict[str, Any]:
        """
        Get Gantt chart visualization data for an agent trace.

        This is a synchronous operation as the user needs the response.
        """
        try:
            validated_trace_id = _validate_and_log("get_gantt_data", validate_trace_id, trace_id)
            logger.debug("Fetching Gantt chart data", extra={"trace_id": validated_trace_id})
            result = self._http.get(f"/agents/traces/{validated_trace_id}/gantt")
            if isinstance(result, dict):
                return result
            return {}
        except Exception as e:
            logger.debug(f"get_gantt_data failed (returning empty): {type(e).__name__}: {e}")
            return {}
