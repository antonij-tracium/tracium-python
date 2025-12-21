"""
HTTP client for making API requests with retry and error handling.
"""

import time
from typing import Any

import httpx

from ..context.tenant_context import get_current_tenant
from ..helpers.logging_config import get_logger, redact_sensitive_data
from ..helpers.retry import retry_with_backoff
from ..helpers.security import check_rate_limit, redact_telemetry_payload

logger = get_logger()


class HTTPClient:
    """
    HTTP client wrapper with retry logic and error handling.
    """

    def __init__(
        self,
        httpx_client: httpx.Client,
        config: Any,
    ) -> None:
        self._client = httpx_client
        self._config = config

    def request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        extract_error_detail: bool = False,
    ) -> dict[str, Any]:
        """
        Internal method to make HTTP requests with retry logic and error handling.

        Args:
            method: HTTP method ('GET', 'POST', 'PATCH')
            path: API endpoint path
            json: Optional JSON payload (for POST/PATCH)
            params: Optional query parameters
            extract_error_detail: Whether to extract detailed error info from response (for POST)
        """
        is_allowed, wait_time = check_rate_limit(self._config.security_config)
        if not is_allowed:
            logger.warning(
                "Rate limit exceeded, waiting before request",
                extra={"path": path, "wait_time_seconds": wait_time},
            )
            time.sleep(wait_time)

        payload = None
        if json is not None:
            payload = redact_telemetry_payload(json, self._config.security_config)
            import json as json_module

            logger.debug(
                f"{method} {path} - Full payload: {json_module.dumps(payload, indent=2, default=str)}"
            )
        else:
            logger.debug(f"{method} {path}")

        headers = {}
        tenant_id = get_current_tenant()
        if tenant_id:
            headers["X-Tenant-ID"] = tenant_id

        def _make_request() -> httpx.Response:
            kwargs = {}
            if headers:
                kwargs["headers"] = headers
            if payload is not None:
                kwargs["json"] = payload
            if params is not None:
                kwargs["params"] = params
            if method == "PATCH":
                kwargs["timeout"] = self._config.timeout

            method_func = getattr(self._client, method.lower())
            response = method_func(path, **kwargs)
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
                response = retry_with_backoff(
                    _make_request,
                    self._config.retry_config,
                    on_retry=_on_retry,
                )

            logger.debug(
                "API request successful",
                extra={
                    "path": path,
                    "status_code": response.status_code,
                },
            )

            return response.json()
        except httpx.HTTPStatusError as e:
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
            elif status_code >= 500:
                error_msg += f". Backend server error. The backend at {self._config.base_url} may be experiencing issues."

            log_extra = {
                "path": path,
                "status_code": status_code,
                "base_url": self._config.base_url,
            }
            if extract_error_detail:
                log_extra.update(
                    {
                        "request_data": redact_sensitive_data(json) if json else None,
                        "response_body": response_body,
                        "error_detail": error_detail,
                    }
                )

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
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.NetworkError) as e:
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
        except Exception as e:
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

    def get(self, path: str) -> dict[str, Any]:
        """Internal method to make GET requests with retry logic and error handling."""
        return self.request("GET", path)

    def patch(self, path: str, *, json: dict[str, Any]) -> dict[str, Any]:
        """Internal method to make PATCH requests with retry logic and error handling."""
        return self.request("PATCH", path, json=json)

    def post(
        self, path: str, *, json: dict[str, Any] | None = None, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Internal method to make POST requests with retry logic and error handling."""
        return self.request("POST", path, json=json, params=params, extract_error_detail=True)
