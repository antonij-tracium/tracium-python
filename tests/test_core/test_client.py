"""
Tests for TraciumClient.
"""

import httpx
import pytest

from tracium.core.client import TraciumClient
from tracium.core.config import TraciumClientConfig
from tracium.helpers.retry import RetryConfig
from tracium.helpers.security import SecurityConfig


class TestTraciumClientInit:
    """Tests for TraciumClient initialization."""

    def test_init_with_api_key(self, test_api_key: str, mock_transport: httpx.MockTransport):
        """Test initializing client with API key."""
        client = TraciumClient.init(api_key=test_api_key, transport=mock_transport)
        assert isinstance(client, TraciumClient)
        client.close()

    def test_init_with_env_var(
        self, monkeypatch: pytest.MonkeyPatch, mock_transport: httpx.MockTransport
    ):
        """Test initializing client with environment variable."""
        monkeypatch.setenv("TRACIUM_API_KEY", "sk_test_env_key")
        client = TraciumClient.init(transport=mock_transport)
        assert isinstance(client, TraciumClient)
        client.close()

    def test_init_without_api_key_raises(self, mock_transport: httpx.MockTransport):
        """Test that init without API key raises ValueError."""
        with pytest.raises(ValueError, match="Tracium API key is required"):
            TraciumClient.init(transport=mock_transport)

    def test_init_with_base_url(self, test_api_key: str, mock_transport: httpx.MockTransport):
        """Test initializing client with custom base URL."""
        base_url = "https://custom.example.com"
        client = TraciumClient.init(
            api_key=test_api_key, base_url=base_url, transport=mock_transport
        )
        assert client._config.base_url == base_url
        client.close()

    def test_init_with_config(self, test_api_key: str, mock_transport: httpx.MockTransport):
        """Test initializing client with config object."""
        config = TraciumClientConfig(
            base_url="https://config.example.com",
            timeout=30.0,
            retry_config=RetryConfig(max_retries=5),
            security_config=SecurityConfig(rate_limit_enabled=False),
        )
        client = TraciumClient.init(api_key=test_api_key, config=config, transport=mock_transport)
        assert client._config.base_url == config.base_url
        assert client._config.timeout == config.timeout
        assert client._config.retry_config.max_retries == 5
        client.close()

    def test_init_with_both_config_and_base_url_raises(
        self, test_api_key: str, mock_transport: httpx.MockTransport
    ):
        """Test that providing both config and base_url raises ValueError."""
        config = TraciumClientConfig()
        with pytest.raises(ValueError, match="Provide either config or base_url"):
            TraciumClient.init(
                api_key=test_api_key,
                base_url="https://example.com",
                config=config,
                transport=mock_transport,
            )

    def test_context_manager(self, test_api_key: str, mock_transport: httpx.MockTransport):
        """Test client as context manager."""
        with TraciumClient.init(api_key=test_api_key, transport=mock_transport) as client:
            assert isinstance(client, TraciumClient)


class TestTraciumClientMethods:
    """Tests for TraciumClient methods."""

    def test_trace_method(self, tracium_client: TraciumClient):
        """Test that trace() method returns self."""
        result = tracium_client.trace()
        assert result is tracium_client

    def test_close_method(
        self,
        test_api_key: str,
        client_config: TraciumClientConfig,
        mock_transport: httpx.MockTransport,
    ):
        """Test that close() method closes the HTTP client."""
        client = TraciumClient(api_key=test_api_key, config=client_config, transport=mock_transport)
        assert not client._http._client.is_closed
        client.close()
        assert client._http._client.is_closed

    def test_get_current_user_caches_result(self, tracium_client: TraciumClient):
        """Test that get_current_user caches the result."""
        call_count = 0

        def mock_get_current_user():
            nonlocal call_count
            call_count += 1
            return {"plan": "pro"}

        tracium_client._api.get_current_user = mock_get_current_user

        assert tracium_client.get_current_user()["plan"] == "pro"
        assert call_count == 1
        assert tracium_client.get_current_user()["plan"] == "pro"
        assert call_count == 1

    def test_get_current_user_handles_errors(self, tracium_client: TraciumClient):
        """Test that get_current_user handles errors gracefully."""

        def mock_get_current_user():
            raise Exception("API error")

        tracium_client._api.get_current_user = mock_get_current_user
        assert tracium_client.get_current_user()["plan"] == "free"
