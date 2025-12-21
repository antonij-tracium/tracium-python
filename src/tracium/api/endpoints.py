"""
API endpoint methods for Tracium SDK.
"""

from collections.abc import Iterable, Sequence
from typing import Any

from tracium.api.http_client import HTTPClient

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

logger = get_logger()


class TraciumAPIEndpoints:
    """
    API endpoint methods for Tracium.
    This class contains all the API endpoint methods that were previously in TraciumClient.
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
        return self._http.post("/agents/traces", json=payload)

    def record_agent_spans(
        self,
        trace_id: str,
        spans: Iterable[dict[str, Any]],
    ) -> Sequence[dict[str, Any]]:
        validated_trace_id = _validate_and_log("record_agent_spans", validate_trace_id, trace_id)
        spans_list = list(spans)
        if not spans_list:
            raise ValueError("spans cannot be empty")
        if len(spans_list) > 1000:
            raise ValueError("spans cannot exceed 1000 items per request")

        validated_spans: list[dict[str, Any]] = []
        for span in spans_list:
            if not isinstance(span, dict):
                raise TypeError("Each span must be a dictionary")
            validated_span = dict(span)
            if "span_type" in validated_span:
                validated_span["span_type"] = validate_span_type(str(validated_span["span_type"]))
            if "span_id" in validated_span:
                validated_span["span_id"] = validate_span_id(str(validated_span["span_id"]))
                del validated_span["span_id"]
            if "parent_span_id" in validated_span:
                validated_span["parent_span_id"] = validate_span_id(
                    str(validated_span["parent_span_id"])
                )
                del validated_span["parent_span_id"]
            validated_spans.append(validated_span)

        payload = {"spans": validated_spans}
        logger.debug(
            "Recording agent spans",
            extra={"trace_id": validated_trace_id, "span_count": len(validated_spans)},
        )
        return self._http.post(f"/agents/traces/{validated_trace_id}/spans", json=payload)

    def update_agent_span(
        self,
        trace_id: str,
        span_id: str,
        span_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Update an existing agent span by POSTing with the existing span ID."""
        validated_trace_id = _validate_and_log("update_agent_span", validate_trace_id, trace_id)
        validated_span_id = _validate_and_log("update_agent_span", validate_span_id, span_id)

        validated_span = dict(span_data)
        validated_span["id"] = validated_span_id
        if "span_type" in validated_span:
            validated_span["span_type"] = validate_span_type(str(validated_span["span_type"]))

        payload = {"spans": [validated_span]}
        logger.debug(
            "Updating agent span",
            extra={"trace_id": validated_trace_id, "span_id": validated_span_id},
        )
        result = self._http.post(f"/agents/traces/{validated_trace_id}/spans", json=payload)
        if isinstance(result, list) and len(result) > 0:
            return result[0]
        return result

    def complete_agent_trace(
        self,
        trace_id: str,
        *,
        summary: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        tags: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        validated_trace_id = _validate_and_log("complete_agent_trace", validate_trace_id, trace_id)
        validated_tags = _validate_and_log("complete_agent_trace", validate_tags, tags)

        payload: dict[str, Any] = {}
        if summary:
            payload["summary"] = summary
        if validated_tags:
            payload["tags"] = list(validated_tags)

        logger.debug("Completing agent trace", extra={"trace_id": validated_trace_id})
        return self._http.post(f"/agents/traces/{validated_trace_id}/complete", json=payload)

    def fail_agent_trace(
        self,
        trace_id: str,
        *,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        validated_trace_id = _validate_and_log("fail_agent_trace", validate_trace_id, trace_id)
        validated_error = (
            _validate_and_log("fail_agent_trace", validate_error_message, error) if error else None
        )

        payload: dict[str, Any] = {}
        if validated_error:
            payload["error"] = validated_error

        logger.debug("Failing agent trace", extra={"trace_id": validated_trace_id})
        return self._http.post(f"/agents/traces/{validated_trace_id}/fail", json=payload)

    def trigger_drift_check(
        self,
        *,
        metrics: Sequence[str] | None = None,
        alert_channels: Sequence[str] | None = None,
        workspace_id: str | None = None,
    ) -> Sequence[dict[str, Any]]:
        """
        Manually trigger drift detection checks for the current user's evaluations.

        Args:
            metrics: List of metric names to check (None = all)
            alert_channels: Channels to send alerts to ["slack", "email", "webhook"]
            workspace_id: Optional workspace ID to filter evaluations by
        """
        params: dict[str, Any] = {}
        if metrics is not None:
            params["metrics"] = list(metrics)
        if alert_channels is not None:
            params["alert_channels"] = list(alert_channels)
        if workspace_id is not None:
            params["workspace_id"] = workspace_id
        return self._http.post("/drift/check", params=params)

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
        Manually trigger prompt embeddings drift detection checks.

        Args:
            workspace_ids: List of workspace IDs to check (None = all workspaces for user)
            alert_channels: Channels to send alerts to ["slack", "email", "webhook"]
            similarity_threshold: Threshold for drift detection (0.05-0.1 recommended)
            baseline_days: Number of days for baseline (default: 60)
            current_days: Number of days for current period (default: 1)
        """
        params: dict[str, Any] = {
            "similarity_threshold": similarity_threshold,
            "baseline_days": baseline_days,
            "current_days": current_days,
        }
        if workspace_ids is not None:
            params["workspace_ids"] = list(workspace_ids)
        if alert_channels is not None:
            params["alert_channels"] = list(alert_channels)
        return self._http.post("/drift/prompt-embeddings/check", params=params)

    def create_prompt_embeddings_baseline(
        self,
        workspace_id: str,
        *,
        baseline_days: int = 60,
    ) -> dict[str, Any]:
        """
        Create or update baseline embedding centroid for a workspace.

        Args:
            workspace_id: Workspace ID
            baseline_days: Number of days to use for baseline (default: 60)
        """
        params: dict[str, Any] = {
            "workspace_id": workspace_id,
            "baseline_days": baseline_days,
        }
        return self._http.post("/drift/prompt-embeddings/baseline", params=params)

    def create_evaluation(
        self,
        input_text: str,
        output_text: str,
        *,
        context: str | None = None,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Submit an evaluation for processing.

        Embeddings are computed automatically on the backend.

        Args:
            input_text: The input/query text for evaluation
            output_text: The generated output text for evaluation
            context: Optional context for evaluation
            workspace_id: Optional workspace ID (defaults to user's default workspace)

        Returns:
            Dict containing the created evaluation with id, status, etc.

        Raises:
            ValueError: If input_text or output_text is empty
        """
        if not input_text or not input_text.strip():
            raise ValueError("input_text cannot be empty")
        if not output_text or not output_text.strip():
            raise ValueError("output_text cannot be empty")

        payload: dict[str, Any] = {
            "input_text": input_text.strip(),
            "output_text": output_text.strip(),
        }

        if context:
            payload["context"] = context.strip()
        if workspace_id:
            payload["workspace_id"] = workspace_id

        logger.debug("Creating evaluation")
        return self._http.post("/evaluations/", json=payload)

    def get_current_user(self) -> dict[str, Any]:
        """
        Get current user information including plan.

        The result is cached to avoid repeated API calls.

        Returns:
            Dict containing user info with 'plan' field (e.g., 'free', 'developer', 'startup', 'scale')
        """
        user_info = self._http.get("/auth/me")
        return {"plan": user_info.get("plan", "free")}

    def get_gantt_data(self, trace_id: str) -> dict[str, Any]:
        """
        Get Gantt chart visualization data for an agent trace.

        Returns spans organized for Gantt chart visualization with:
        - Flat span list with all relationships
        - Hierarchical tree structure (parent-child)
        - Parallel groups (spans with same parallel_group_id)
        - Timeline (chronologically ordered spans)
        - Summary statistics (total spans, max depth, parallel group count)

        Args:
            trace_id: The agent trace ID

        Returns:
            Dict containing gantt visualization data
        """
        validated_trace_id = _validate_and_log("get_gantt_data", validate_trace_id, trace_id)
        logger.debug("Fetching Gantt chart data", extra={"trace_id": validated_trace_id})
        return self._http.get(f"/agents/traces/{validated_trace_id}/gantt")
