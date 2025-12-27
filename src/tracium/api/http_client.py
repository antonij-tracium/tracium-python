"""
HTTP client for making API requests with retry and error handling.

All requests are processed asynchronously in the background to avoid
blocking user code. Errors are caught and logged - never propagated
to break user applications.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import httpx

from ..context.tenant_context import get_current_tenant
from ..helpers.background_sender import BackgroundSender, RequestMethod
from ..helpers.logging_config import get_logger, redact_sensitive_data
from ..helpers.retry import retry_with_backoff
from ..helpers.security import check_rate_limit, redact_telemetry_payload

if TYPE_CHECKING:
    from ..core.config import TraciumClientConfig

logger = get_logger()


class HTTPClient:
    """
    HTTP client wrapper with async background processing and error handling.

    All telemetry requests are sent asynchronously in a background thread
    to avoid blocking user code. Errors are caught and logged.
    """

    def __init__(
        self,
        httpx_client: httpx.Client,
        config: TraciumClientConfig,
    ) -> None:
        self._client = httpx_client
        self._config = config
        self._background_sender: BackgroundSender | None = None

    def _get_background_sender(self) -> BackgroundSender:
        """Get or create the background sender."""
        if self._background_sender is None:
            self._background_sender = BackgroundSender(self._client, self._config)
        return self._background_sender

    def _get_tenant_headers(self) -> dict[str, str]:
        """Get headers including tenant ID if available."""
        headers = {}
        try:
            tenant_id = get_current_tenant()
            if tenant_id:
                headers["X-Tenant-ID"] = tenant_id
        except Exception:
            pass
        return headers

    def request_async(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        """
        Queue a request for async background processing.

        This method returns immediately without blocking. The request
        is processed in a background thread. All errors are caught
        and logged - never propagated to user code.
        """
        try:
            request_method = RequestMethod[method.upper()]
            headers = self._get_tenant_headers()

            self._get_background_sender().enqueue(
                request_method,
                path,
                json=json,
                params=params,
                headers=headers if headers else None,
            )
        except Exception as e:
            # Never propagate errors to user code
            logger.debug(f"Failed to queue async request: {type(e).__name__}: {e}")

    def request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        extract_error_detail: bool = False,
    ) -> dict[str, Any] | list[Any]:
        """
        Make HTTP request with retry logic and error handling.

        Args:
            method: HTTP method ('GET', 'POST', 'PATCH')
            path: API endpoint path
            json: Optional JSON payload (for POST/PATCH)
            params: Optional query parameters
            extract_error_detail: Whether to extract detailed error info from response

        Returns:
            Response data as dict or list. Returns empty dict on error if fail_open is True.
        """
        try:
            return self._execute_request(method, path, json=json, params=params, extract_error_detail=extract_error_detail)
        except Exception as e:
            # Final catch-all - ensure we never break user code
            if self._config.fail_open:
                logger.debug(f"Request failed (fail-open mode): {type(e).__name__}: {e}")
                return {}
            raise

    def _execute_request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        extract_error_detail: bool = False,
    ) -> dict[str, Any] | list[Any]:
        """Internal method to execute HTTP requests with retry logic."""
        try:
            if self._config.security_config:
                is_allowed, wait_time = check_rate_limit(self._config.security_config)
                if not is_allowed:
                    logger.warning(
                        "Rate limit exceeded, waiting before request",
                        extra={"path": path, "wait_time_seconds": wait_time},
                    )
                    time.sleep(wait_time)
        except Exception:
            pass

        payload = None
        if json is not None:
            try:
                if self._config.security_config:
                    payload = redact_telemetry_payload(json, self._config.security_config)
                else:
                    payload = json
                import json as json_module

                logger.debug(
                    f"{method} {path} - Full payload: {json_module.dumps(payload, indent=2, default=str)}"
                )
            except Exception:
                payload = json if payload is None else payload
                logger.debug(f"{method} {path}")
        else:
            logger.debug(f"{method} {path}")

        headers = self._get_tenant_headers()

        def _make_request() -> httpx.Response:
            kwargs: dict[str, Any] = {}
            if headers:
                kwargs["headers"] = headers
            if payload is not None:
                kwargs["json"] = payload
            if params is not None:
                kwargs["params"] = params
            if method == "PATCH":
                kwargs["timeout"] = self._config.timeout

            method_func = getattr(self._client, method.lower())
            response: httpx.Response = method_func(path, **kwargs)
            response.raise_for_status()
            return response

        def _on_retry(exc: Exception | None, attempt: int, delay: float) -> None:
            logger.warning(
                "Retrying API request",
                extra={
                    "path": path,
                    "attempt": attempt,
                    "delay_seconds": delay,
                    "error": str(exc) if exc else None,
                },
            )

        try:
            if method == "PATCH" and not self._config.retry_config:
                response = _make_request()
            else:
                from ..helpers.retry import RetryConfig
                retry_config = self._config.retry_config or RetryConfig()
                response = retry_with_backoff(
                    _make_request,
                    retry_config,
                    on_retry=_on_retry,
                )

            logger.debug(
                "API request successful",
                extra={
                    "path": path,
                    "status_code": response.status_code,
                },
            )

            json_data = response.json()
            # Allow both dict and list responses - some endpoints return lists
            if not isinstance(json_data, (dict | list)):
                raise TypeError(
                    f"Expected dict or list from API response, got {type(json_data).__name__}"
                )
            return json_data
        except httpx.HTTPStatusError as e:
            return self._handle_http_error(e, path, json, extract_error_detail)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.NetworkError) as e:
            return self._handle_connection_error(e, path)
        except Exception as e:
            return self._handle_unexpected_error(e, path)

    def _handle_http_error(
        self,
        e: httpx.HTTPStatusError,
        path: str,
        json: dict[str, Any] | None,
        extract_error_detail: bool,
    ) -> dict[str, Any] | list[Any]:
        """Handle HTTP status errors."""
        status_code = e.response.status_code if e.response else None
        error_msg = f"API request failed with HTTP {status_code} at {path}"

        response_body = None
        error_detail = None
        if extract_error_detail:
            try:
                if e.response:
                    response_body = e.response.json()
                    if isinstance(response_body, dict):
                        error_detail = response_body.get("detail")
                        if isinstance(error_detail, dict):
                            error_detail = str(error_detail)
                        elif isinstance(error_detail, list) and len(error_detail) > 0:
                            error_detail = "; ".join([str(err) for err in error_detail[:3]])
            except Exception:
                try:
                    if e.response:
                        response_body = e.response.text
                        error_detail = (
                            response_body[:500] if len(response_body) > 500 else response_body
                        )
                except Exception:
                    pass

            if error_detail:
                error_msg += f"\nBackend error: {error_detail}"

        if status_code == 400:
            error_msg += "\nValidation error. Check that all required fields are provided and correctly formatted."
            error_msg += f"\nRequest was sent to: {self._config.base_url}{path}"
        elif status_code == 404:
            error_msg += f". Endpoint not found. Check that the backend is running and accessible at {self._config.base_url}"
        elif status_code == 401 or status_code == 403:
            error_msg += ". Authentication failed. Check your API key and ensure it's valid."
            error_msg += f"\nRequest was sent to: {self._config.base_url}{path}"
            error_msg += "\nVerify that:"
            error_msg += "\n  1. Your API key is set correctly (via api_key parameter or TRACIUM_API_KEY env var)"
            error_msg += "\n  2. Your API key is valid and not expired"
            error_msg += "\n  3. The API key has the necessary permissions for this endpoint"
        elif status_code is not None and status_code >= 500:
            error_msg += f". Backend server error. The backend at {self._config.base_url} may be experiencing issues."

        log_extra: dict[str, Any] = {
            "path": path,
            "status_code": status_code,
            "base_url": self._config.base_url,
        }
        if extract_error_detail:
            try:
                log_extra.update(
                    {
                        "request_data": redact_sensitive_data(json) if json else None,
                        "response_body": response_body,
                        "error_detail": error_detail,
                    }
                )
            except Exception:
                pass

        logger.error(error_msg, extra=log_extra, exc_info=True)

        if not self._config.fail_open:
            if extract_error_detail:
                enhanced_error = httpx.HTTPStatusError(
                    error_msg,
                    request=e.request,
                    response=e.response,
                )
                raise enhanced_error
            raise
        logger.warning("SDK configured to fail-open, returning empty response")
        return {}

    def _handle_connection_error(
        self,
        e: Exception,
        path: str,
    ) -> dict[str, Any] | list[Any]:
        """Handle connection errors."""
        error_msg = (
            f"Failed to connect to Tracium backend at {self._config.base_url}. "
            f"Please check:\n"
            f"  1. Is the backend accessible? (default: https://api.tracium.ai)\n"
            f"  2. Is TRACIUM_BASE_URL set correctly? (current: {self._config.base_url})\n"
            f"  3. Is there a firewall or network issue blocking the connection?"
        )
        logger.error(
            error_msg,
            extra={
                "path": path,
                "base_url": self._config.base_url,
                "error_type": type(e).__name__,
                "error": str(e),
            },
            exc_info=True,
        )
        if not self._config.fail_open:
            raise ConnectionError(error_msg) from e
        logger.warning("SDK configured to fail-open, returning empty response")
        return {}

    def _handle_unexpected_error(
        self,
        e: Exception,
        path: str,
    ) -> dict[str, Any] | list[Any]:
        """Handle unexpected errors."""
        error_msg = (
            f"API request failed with unexpected error: {type(e).__name__}: {str(e)}. "
            f"Backend URL: {self._config.base_url}, Path: {path}"
        )
        logger.error(
            error_msg,
            extra={
                "path": path,
                "base_url": self._config.base_url,
                "error_type": type(e).__name__,
            },
            exc_info=True,
        )
        if not self._config.fail_open:
            raise
        logger.warning("SDK configured to fail-open, returning empty response")
        return {}

    def get(self, path: str) -> dict[str, Any] | list[Any]:
        """Make GET request with retry logic and error handling."""
        return self.request("GET", path)

    def patch(self, path: str, *, json: dict[str, Any]) -> dict[str, Any] | list[Any]:
        """Make PATCH request with retry logic and error handling."""
        return self.request("PATCH", path, json=json)

    def post(
        self, path: str, *, json: dict[str, Any] | None = None, params: dict[str, Any] | None = None
    ) -> dict[str, Any] | list[Any]:
        """Make POST request with retry logic and error handling."""
        return self.request("POST", path, json=json, params=params, extract_error_detail=True)

    def post_async(
        self, path: str, *, json: dict[str, Any] | None = None, params: dict[str, Any] | None = None
    ) -> None:
        """Queue POST request for async background processing."""
        self.request_async("POST", path, json=json, params=params)

    def flush(self) -> None:
        """Wait for all queued async requests to complete."""
        if self._background_sender:
            self._background_sender.flush()

    def close(self) -> None:
        """Close the HTTP client and background sender."""
        try:
            if self._background_sender:
                self._background_sender.shutdown()
                self._background_sender = None
        except Exception:
            pass
        try:
            self._client.close()
        except Exception:
            pass
