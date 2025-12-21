"""
Tests for retry logic.
"""

from unittest.mock import Mock

import httpx
import pytest

from tracium.helpers.retry import (
    RetryConfig,
    calculate_backoff_delay,
    retry_with_backoff,
    should_retry,
)


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_default_config(self):
        """Test default retry configuration."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.backoff_factor == 1.0
        assert config.initial_delay == 0.1
        assert config.max_delay == 60.0
        assert 429 in config.retryable_status_codes
        assert 500 in config.retryable_status_codes
        assert httpx.ConnectTimeout in config.retryable_exceptions

    def test_custom_config(self):
        """Test custom retry configuration."""
        config = RetryConfig(
            max_retries=5,
            backoff_factor=2.0,
            initial_delay=0.5,
            max_delay=120.0,
        )
        assert config.max_retries == 5
        assert config.backoff_factor == 2.0
        assert config.initial_delay == 0.5
        assert config.max_delay == 120.0


class TestShouldRetry:
    """Tests for should_retry function."""

    @pytest.mark.parametrize(
        "exception,should",
        [
            (httpx.ConnectTimeout("timeout"), True),
            (httpx.NetworkError("network error"), True),
            (ValueError("invalid"), False),
        ],
    )
    def test_retry_on_exception(self, exception: Exception, should: bool):
        """Test retry decision based on exception type."""
        assert should_retry(exception, None, RetryConfig()) is should

    @pytest.mark.parametrize(
        "status_code,should",
        [
            (429, True),
            (500, True),
            (503, True),
            (200, False),
            (400, False),
            (404, False),
        ],
    )
    def test_retry_on_status_code(self, status_code: int, should: bool):
        """Test retry decision based on status code."""
        assert should_retry(None, status_code, RetryConfig()) is should


class TestCalculateBackoffDelay:
    """Tests for calculate_backoff_delay function."""

    def test_exponential_backoff(self):
        """Test exponential backoff calculation."""
        config = RetryConfig(initial_delay=1.0, backoff_factor=2.0)
        assert calculate_backoff_delay(0, config) == 1.0
        assert calculate_backoff_delay(1, config) == 2.0
        assert calculate_backoff_delay(2, config) == 4.0
        assert calculate_backoff_delay(3, config) == 8.0

    def test_max_delay_limit(self):
        """Test that delay is capped at max_delay."""
        config = RetryConfig(initial_delay=10.0, backoff_factor=2.0, max_delay=30.0)
        delay = calculate_backoff_delay(5, config)
        assert delay == 30.0


class TestRetryWithBackoff:
    """Tests for retry_with_backoff function."""

    def test_successful_call_no_retry(self):
        """Test that successful call doesn't retry."""
        func = Mock(return_value="success")
        config = RetryConfig(max_retries=3)
        result = retry_with_backoff(func, config)
        assert result == "success"
        assert func.call_count == 1

    def test_retry_on_retryable_exception(self):
        """Test retry on retryable exception."""
        func = Mock(side_effect=[httpx.ConnectTimeout("timeout"), "success"])
        config = RetryConfig(max_retries=3, initial_delay=0.01)
        result = retry_with_backoff(func, config)
        assert result == "success"
        assert func.call_count == 2

    def test_retry_exhausted_raises(self):
        """Test that exhausted retries raise the last exception."""
        func = Mock(side_effect=httpx.ConnectTimeout("timeout"))
        config = RetryConfig(max_retries=2, initial_delay=0.01)
        with pytest.raises(httpx.ConnectTimeout):
            retry_with_backoff(func, config)
        assert func.call_count == 3

    def test_no_retry_on_non_retryable_exception(self):
        """Test that non-retryable exceptions are raised immediately."""
        func = Mock(side_effect=ValueError("invalid"))
        config = RetryConfig(max_retries=3)
        with pytest.raises(ValueError):
            retry_with_backoff(func, config)
        assert func.call_count == 1

    def test_on_retry_callback(self):
        """Test that on_retry callback is called."""
        func = Mock(side_effect=[httpx.ConnectTimeout("timeout"), "success"])
        on_retry = Mock()
        config = RetryConfig(max_retries=3, initial_delay=0.01)
        result = retry_with_backoff(func, config, on_retry=on_retry)
        assert result == "success"
        assert on_retry.call_count == 1
        assert isinstance(on_retry.call_args[0][0], httpx.ConnectTimeout)
        assert on_retry.call_args[0][1] == 1
        assert on_retry.call_args[0][2] > 0

    def test_http_status_error_retry(self):
        """Test retry on HTTP status error with retryable status code."""
        response = Mock(spec=httpx.Response)
        response.status_code = 500
        error = httpx.HTTPStatusError("Server error", request=Mock(), response=response)
        func = Mock(side_effect=[error, "success"])
        config = RetryConfig(max_retries=3, initial_delay=0.01)
        result = retry_with_backoff(func, config)
        assert result == "success"
        assert func.call_count == 2

    def test_http_status_error_no_retry(self):
        """Test no retry on HTTP status error with non-retryable status code."""
        response = Mock(spec=httpx.Response)
        response.status_code = 400
        error = httpx.HTTPStatusError("Bad request", request=Mock(), response=response)
        func = Mock(side_effect=error)
        config = RetryConfig(max_retries=3)
        with pytest.raises(httpx.HTTPStatusError):
            retry_with_backoff(func, config)
        assert func.call_count == 1
