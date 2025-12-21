"""
Tests for HTTPClient.
"""

import httpx
import pytest

from tracium.api.http_client import HTTPClient
from tracium.core.config import TraciumClientConfig
from tracium.helpers.retry import RetryConfig
from tracium.helpers.security import SecurityConfig


class TestHTTPClient:
    """Tests for HTTPClient."""

    @pytest.fixture
    def httpx_client(self) -> httpx.Client:
        """Create a mock httpx client."""
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"status": "ok"}))
        return httpx.Client(transport=transport, base_url="http://localhost:8000")

    @pytest.fixture
    def http_client(self, httpx_client: httpx.Client) -> HTTPClient:
        """Create an HTTPClient instance."""
        config = TraciumClientConfig(
            base_url="http://localhost:8000",
            retry_config=RetryConfig(max_retries=1, initial_delay=0.01),
            security_config=SecurityConfig(rate_limit_enabled=False),
        )
        return HTTPClient(httpx_client, config)

    @pytest.mark.parametrize(
        "method,json_data",
        [
            ("get", None),
            ("post", {"key": "value"}),
            ("patch", {"key": "updated"}),
        ],
    )
    def test_http_methods(self, http_client: HTTPClient, method: str, json_data: dict | None):
        """Test GET, POST, and PATCH requests."""
        method_func = getattr(http_client, method)
        kwargs = {"json": json_data} if json_data else {}
        assert method_func("/test", **kwargs) == {"status": "ok"}

    def test_request_with_tenant_header(self, httpx_client: httpx.Client):
        """Test that tenant ID is added to headers when present."""
        from tracium.context.tenant_context import set_tenant

        config = TraciumClientConfig(
            base_url="http://localhost:8000",
            retry_config=RetryConfig(max_retries=0),
            security_config=SecurityConfig(rate_limit_enabled=False),
        )

        set_tenant("test-tenant-123")
        http_client = HTTPClient(httpx_client, config)

        captured_headers = {}

        def mock_transport(request: httpx.Request) -> httpx.Response:
            captured_headers.update(dict(request.headers))
            return httpx.Response(200, json={"status": "ok"})

        httpx_client._transport = httpx.MockTransport(mock_transport)
        http_client.get("/test")

        tenant_header = None
        for key, value in captured_headers.items():
            if key.lower() == "x-tenant-id":
                tenant_header = value
                break
        assert tenant_header is not None, f"X-Tenant-ID header not found in {captured_headers}"
        assert tenant_header == "test-tenant-123"

    def test_request_retries_on_retryable_error(self, httpx_client: httpx.Client):
        """Test that requests are retried on retryable errors."""
        call_count = 0

        def mock_transport(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise httpx.ConnectTimeout("Connection timeout")
            return httpx.Response(200, json={"status": "ok"})

        httpx_client._transport = httpx.MockTransport(mock_transport)

        config = TraciumClientConfig(
            base_url="http://localhost:8000",
            retry_config=RetryConfig(max_retries=3, initial_delay=0.01),
            security_config=SecurityConfig(rate_limit_enabled=False),
        )
        http_client = HTTPClient(httpx_client, config)

        response = http_client.get("/test")
        assert response == {"status": "ok"}
        assert call_count == 2

    @pytest.mark.parametrize("fail_open", [True, False])
    def test_fail_open_behavior(self, httpx_client: httpx.Client, fail_open: bool):
        """Test fail-open behavior configuration."""
        httpx_client._transport = httpx.MockTransport(
            lambda r: httpx.Response(500, json={"error": "Internal server error"})
        )
        config = TraciumClientConfig(
            base_url="http://localhost:8000",
            retry_config=RetryConfig(max_retries=0),
            fail_open=fail_open,
            security_config=SecurityConfig(rate_limit_enabled=False),
        )
        http_client = HTTPClient(httpx_client, config)

        if fail_open:
            assert http_client.get("/test") == {}
        else:
            with pytest.raises(httpx.HTTPStatusError):
                http_client.get("/test")
