"""
Pytest configuration and shared fixtures.
"""

from collections.abc import Generator

import httpx
import pytest

from tracium.core.client import TraciumClient
from tracium.core.config import TraciumClientConfig
from tracium.helpers.retry import RetryConfig
from tracium.helpers.security import SecurityConfig


@pytest.fixture
def mock_transport() -> httpx.MockTransport:
    """Create a mock HTTP transport for testing."""
    return httpx.MockTransport(lambda request: httpx.Response(200, json={"status": "ok"}))


@pytest.fixture
def test_api_key() -> str:
    """Return a test API key."""
    return "sk_test_1234567890abcdefghijklmnopqrstuvwxyz"


@pytest.fixture
def test_base_url() -> str:
    """Return a test base URL."""
    return "http://localhost:8000"


@pytest.fixture
def client_config(test_base_url: str) -> TraciumClientConfig:
    """Create a test client configuration."""
    return TraciumClientConfig(
        base_url=test_base_url,
        timeout=5.0,
        retry_config=RetryConfig(max_retries=1, initial_delay=0.01),
        fail_open=True,
        security_config=SecurityConfig(rate_limit_enabled=False),
    )


@pytest.fixture
def tracium_client(
    test_api_key: str, client_config: TraciumClientConfig, mock_transport: httpx.MockTransport
) -> Generator[TraciumClient, None, None]:
    """Create a TraciumClient instance for testing."""
    client = TraciumClient(
        api_key=test_api_key,
        config=client_config,
        transport=mock_transport,
    )
    yield client
    client.close()


@pytest.fixture(autouse=True)
def reset_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset environment variables before each test."""
    monkeypatch.delenv("TRACIUM_API_KEY", raising=False)
    monkeypatch.delenv("TRACIUM_BASE_URL", raising=False)
