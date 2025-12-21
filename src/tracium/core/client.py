"""
Main Tracium client class.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from ..api.endpoints import TraciumAPIEndpoints
from ..api.http_client import HTTPClient
from ..context.trace_context import current_trace
from ..helpers.logging_config import get_logger
from ..helpers.retry import RetryConfig
from ..helpers.security import SecurityConfig
from ..helpers.validation import validate_api_key
from ..models.trace_handle import AgentTraceManager
from ..utils.validation import _validate_and_log
from .config import TraciumClientConfig

logger = get_logger()


def _normalize_base_url(base_url: str) -> str:
    """
    Normalize base URL to ensure it has a valid protocol.

    If the URL doesn't start with http:// or https://, adds http:// as default.
    This handles cases where users set TRACIUM_BASE_URL to just 'localhost' or 'localhost:8000'.

    Args:
        base_url: The base URL to normalize

    Returns:
        Normalized base URL with protocol
    """
    base_url = base_url.strip()
    if not base_url.startswith(("http://", "https://")):
        base_url = f"http://{base_url}"
    return base_url


class TraciumClient:
    """
    Minimal wrapper for the Tracium telemetry + evaluation API.

    All methods raise `httpx.HTTPStatusError` if the API returns a non-2xx status.

    Example:
        >>> client = TraciumClient.init(api_key="sk_live_...")
        >>> import tracium
        >>> tracium.init(api_key="sk_live_...")
    """

    @classmethod
    def init(
        cls,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        config: TraciumClientConfig | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> TraciumClient:
        """
        Initialize a TraciumClient with a simple one-line call.

        Example:
            >>> client = TraciumClient.init(api_key="sk_live_...")
        """
        if api_key is None:
            api_key = os.getenv("TRACIUM_API_KEY")
        if not api_key:
            raise ValueError("Tracium API key is required. Pass api_key or set TRACIUM_API_KEY.")

        if base_url is None and (config is None or config.base_url is None):
            base_url = os.getenv("TRACIUM_BASE_URL", "https://api.tracium.ai")
            if base_url:
                base_url = _normalize_base_url(base_url)
        elif base_url is not None:
            base_url = _normalize_base_url(base_url)
        else:
            base_url = None

        client = cls(api_key=api_key, base_url=base_url, config=config, transport=transport)

        return client

    def __init__(
        self,
        api_key: str,
        *,
        config: TraciumClientConfig | None = None,
        base_url: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if config is not None and base_url is not None:
            raise ValueError("Provide either config or base_url, not both.")

        if config is None:
            config = TraciumClientConfig()
        if base_url is not None:
            normalized_base_url = _normalize_base_url(base_url)
            config = TraciumClientConfig(
                base_url=normalized_base_url,
                timeout=config.timeout,
                user_agent=config.user_agent,
                retry_config=config.retry_config,
                fail_open=config.fail_open,
            )
        else:
            normalized_base_url = _normalize_base_url(config.base_url)
            if normalized_base_url != config.base_url:
                config = TraciumClientConfig(
                    base_url=normalized_base_url,
                    timeout=config.timeout,
                    user_agent=config.user_agent,
                    retry_config=config.retry_config,
                    fail_open=config.fail_open,
                )

        is_test_scenario = (
            "test" in config.base_url.lower()
            or "mock" in config.base_url.lower()
            or "localhost" in config.base_url.lower()
        ) or transport is not None
        if is_test_scenario:
            validated_api_key = _validate_and_log(
                "TraciumClient.__init__",
                lambda k: validate_api_key(k, allow_test_keys=True),
                api_key,
            )
        else:
            validated_api_key = _validate_and_log(
                "TraciumClient.__init__", validate_api_key, api_key
            )
        if config.retry_config is None:
            config.retry_config = RetryConfig()
        if config.security_config is None:
            config.security_config = SecurityConfig()
        self._config = config

        headers = {
            "X-API-Key": validated_api_key,
            "User-Agent": self._config.user_agent,
        }
        httpx_client = httpx.Client(
            base_url=self._config.base_url,
            headers=headers,
            timeout=self._config.timeout,
            transport=transport,
            follow_redirects=True,
        )

        self._http = HTTPClient(httpx_client, self._config)
        self._api = TraciumAPIEndpoints(self._http)

        self._user_plan: str | None = None
        self._plan_fetched: bool = False

        logger.debug(
            "TraciumClient initialized",
            extra={
                "base_url": self._config.base_url,
                "timeout": self._config.timeout,
                "retry_config": {
                    "max_retries": self._config.retry_config.max_retries,
                    "backoff_factor": self._config.retry_config.backoff_factor,
                },
            },
        )

    def close(self) -> None:
        """Close the client"""
        self._http._client.close()

    def __enter__(self) -> TraciumClient:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def trace(self) -> TraciumClient:
        """
        Enable automatic tracing for all supported libraries with one line.

        This is a convenience method that automatically instruments:
        - OpenAI (GPT-4, GPT-3.5, etc.)
        - Anthropic (Claude)
        - Google Generative AI (Gemini)
        - LangChain
        - LangGraph

        Example:
            >>> client = TraciumClient.init(api_key="sk_live_...")
            >>> client.trace()
            >>>
            >>> import openai
            >>> response = openai.ChatCompletion.create(...)

        Returns:
            self: Returns the client instance for method chaining
        """
        from ..helpers.global_state import TraciumInitOptions, get_options, set_client
        from ..instrumentation.auto_instrumentation import configure_auto_instrumentation

        try:
            existing_options = get_options()
        except RuntimeError:
            existing_options = TraciumInitOptions(
                default_agent_name="app",
                auto_instrument_langchain=True,
                auto_instrument_langgraph=True,
                auto_instrument_llm_clients=True,
            )
            set_client(self, options=existing_options)

        configure_auto_instrumentation(self)
        return self

    def start_agent_trace(self, *args, **kwargs):
        return self._api.start_agent_trace(*args, **kwargs)

    def record_agent_spans(self, *args, **kwargs):
        return self._api.record_agent_spans(*args, **kwargs)

    def update_agent_span(self, *args, **kwargs):
        return self._api.update_agent_span(*args, **kwargs)

    def complete_agent_trace(self, *args, **kwargs):
        return self._api.complete_agent_trace(*args, **kwargs)

    def fail_agent_trace(self, *args, **kwargs):
        return self._api.fail_agent_trace(*args, **kwargs)

    def trigger_drift_check(self, *args, **kwargs):
        return self._api.trigger_drift_check(*args, **kwargs)

    def trigger_prompt_embeddings_drift_check(self, *args, **kwargs):
        return self._api.trigger_prompt_embeddings_drift_check(*args, **kwargs)

    def create_prompt_embeddings_baseline(self, *args, **kwargs):
        return self._api.create_prompt_embeddings_baseline(*args, **kwargs)

    def create_evaluation(self, *args, **kwargs):
        return self._api.create_evaluation(*args, **kwargs)

    def get_gantt_data(self, *args, **kwargs):
        return self._api.get_gantt_data(*args, **kwargs)

    def get_current_user(self) -> dict[str, Any]:
        """
        Get current user information including plan.

        The result is cached to avoid repeated API calls.
        """
        if not self._plan_fetched:
            try:
                user_info = self._api.get_current_user()
                self._user_plan = user_info.get("plan", "free")
                self._plan_fetched = True
                logger.debug(f"Fetched user plan: {self._user_plan}")
            except Exception as e:
                logger.warning(f"Failed to fetch user plan: {e}. Defaulting to 'free'")
                self._user_plan = "free"
                self._plan_fetched = True

        return {"plan": self._user_plan or "free"}

    def agent_trace(
        self,
        *,
        agent_name: str,
        model_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        tags: Sequence[str] | None = None,
        trace_id: str | None = None,
        workspace_id: str | None = None,
        version: str | None = None,
    ) -> AgentTraceManager:
        """
        Context manager that automatically manages the lifecycle of an agent trace.

        Example
        -------
        >>> client = TraciumClient(api_key="secret")
        >>> with client.agent_trace(agent_name="support-bot") as trace:
        ...     with trace.span(span_type="plan") as span:
        ...         span.record_output({"thought": "Let's call CRM"})
        """

        return AgentTraceManager(
            self,
            agent_name=agent_name,
            model_id=model_id,
            metadata=metadata,
            tags=tags,
            trace_id=trace_id,
            workspace_id=workspace_id,
            version=version,
        )


__all__ = ["TraciumClient", "current_trace"]
