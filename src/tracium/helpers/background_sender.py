"""
Background async sender for non-blocking telemetry data transmission.

This module provides a thread-safe queue-based system for sending telemetry data
to the Tracium API without blocking user code. All API calls are processed
asynchronously in a background thread.
"""

from __future__ import annotations

import atexit
import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import httpx

from .logging_config import get_logger
from .retry import RetryConfig, calculate_backoff_delay, should_retry
from .security import check_rate_limit, redact_telemetry_payload

if TYPE_CHECKING:
    from ..core.config import TraciumClientConfig

logger = get_logger()


class RequestMethod(Enum):
    GET = "GET"
    POST = "POST"
    PATCH = "PATCH"


@dataclass
class QueuedRequest:
    """Represents a queued API request."""

    method: RequestMethod
    path: str
    json: dict[str, Any] | None = None
    params: dict[str, Any] | None = None
    headers: dict[str, str] | None = None
    callback: Any | None = None  # Optional callback for result handling


class BackgroundSender:
    """
    Thread-safe background sender for async telemetry transmission.

    Uses a daemon thread to process queued requests without blocking user code.
    All errors are caught and logged - never propagated to break user apps.
    """

    def __init__(
        self,
        httpx_client: httpx.Client,
        config: TraciumClientConfig,
        max_queue_size: int = 10000,
        flush_timeout: float = 5.0,
    ) -> None:
        self._client = httpx_client
        self._config = config
        self._queue: queue.Queue[QueuedRequest | None] = queue.Queue(maxsize=max_queue_size)
        self._shutdown = threading.Event()
        self._flush_timeout = flush_timeout
        self._worker_thread: threading.Thread | None = None
        self._started = False
        self._lock = threading.Lock()

        # Start worker thread
        self._start_worker()

        # Register cleanup on exit
        atexit.register(self._cleanup)

    def _start_worker(self) -> None:
        """Start the background worker thread."""
        with self._lock:
            if self._started:
                return
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name="tracium-background-sender",
                daemon=True,
            )
            self._worker_thread.start()
            self._started = True

    def _worker_loop(self) -> None:
        """Main worker loop that processes queued requests."""
        while not self._shutdown.is_set():
            try:
                # Wait for a request with timeout to allow checking shutdown
                try:
                    request = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if request is None:
                    # Shutdown signal
                    break

                self._process_request(request)
                self._queue.task_done()

            except Exception as e:
                # Never let worker thread die from an exception
                logger.debug(f"Background sender error (ignored): {type(e).__name__}: {e}")

    def _process_request(self, request: QueuedRequest) -> None:
        """Process a single queued request with retry logic."""
        try:
            # Check rate limit
            if self._config.security_config:
                is_allowed, wait_time = check_rate_limit(self._config.security_config)
                if not is_allowed:
                    logger.debug(f"Rate limited, waiting {wait_time}s before request to {request.path}")
                    time.sleep(wait_time)

            # Prepare payload
            payload = None
            if request.json is not None and self._config.security_config:
                payload = redact_telemetry_payload(request.json, self._config.security_config)
            elif request.json is not None:
                payload = request.json

            # Execute request with retry
            retry_config = self._config.retry_config or RetryConfig()
            last_exception: Exception | None = None

            for attempt in range(retry_config.max_retries + 1):
                try:
                    response = self._make_request(request.method, request.path, payload, request.params, request.headers)
                    response.raise_for_status()

                    # Call success callback if provided
                    if request.callback:
                        try:
                            request.callback(response.json())
                        except Exception:
                            pass

                    logger.debug(f"Background request successful: {request.method.value} {request.path}")
                    return

                except httpx.HTTPStatusError as e:
                    last_exception = e
                    status_code = e.response.status_code if e.response else None
                    if not should_retry(None, status_code, retry_config):
                        break
                except (httpx.ConnectError, httpx.ConnectTimeout, httpx.NetworkError) as e:
                    last_exception = e
                    if not should_retry(e, None, retry_config):
                        break
                except Exception as e:
                    last_exception = e
                    break

                if attempt < retry_config.max_retries:
                    delay = calculate_backoff_delay(attempt, retry_config)
                    time.sleep(delay)

            # All retries exhausted - log and continue
            if last_exception:
                logger.debug(
                    f"Background request failed after retries: {request.method.value} {request.path} - "
                    f"{type(last_exception).__name__}: {last_exception}"
                )

        except Exception as e:
            # Catch-all for any unexpected errors
            logger.debug(f"Background request error (ignored): {type(e).__name__}: {e}")

    def _make_request(
        self,
        method: RequestMethod,
        path: str,
        payload: dict[str, Any] | None,
        params: dict[str, Any] | None,
        headers: dict[str, str] | None,
    ) -> httpx.Response:
        """Make an HTTP request."""
        kwargs: dict[str, Any] = {}
        if headers:
            kwargs["headers"] = headers
        if payload is not None:
            kwargs["json"] = payload
        if params is not None:
            kwargs["params"] = params
        if method == RequestMethod.PATCH:
            kwargs["timeout"] = self._config.timeout

        method_func = getattr(self._client, method.value.lower())
        response: httpx.Response = method_func(path, **kwargs)
        return response

    def enqueue(
        self,
        method: RequestMethod,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        callback: Any | None = None,
    ) -> bool:
        """
        Enqueue a request for background processing.

        Returns True if successfully queued, False if queue is full.
        Never raises exceptions.
        """
        try:
            request = QueuedRequest(
                method=method,
                path=path,
                json=json,
                params=params,
                headers=headers,
                callback=callback,
            )
            self._queue.put_nowait(request)
            return True
        except queue.Full:
            logger.debug(f"Background sender queue full, dropping request to {path}")
            return False
        except Exception as e:
            logger.debug(f"Failed to enqueue request: {type(e).__name__}: {e}")
            return False

    def flush(self, timeout: float | None = None) -> None:
        """
        Wait for all queued requests to be processed.

        Args:
            timeout: Maximum time to wait in seconds. Uses default if None.
        """
        try:
            self._queue.join()
        except Exception:
            pass

    def _cleanup(self) -> None:
        """Clean up resources on shutdown."""
        try:
            self._shutdown.set()

            # Signal worker to stop
            try:
                self._queue.put_nowait(None)
            except Exception:
                pass

            # Wait briefly for worker to finish
            if self._worker_thread and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=self._flush_timeout)

        except Exception:
            pass

    def shutdown(self) -> None:
        """Explicitly shutdown the background sender."""
        self._cleanup()


# Global instance management
_background_sender: BackgroundSender | None = None
_sender_lock = threading.Lock()


def get_background_sender(
    httpx_client: httpx.Client,
    config: TraciumClientConfig,
) -> BackgroundSender:
    """Get or create the global background sender instance."""
    global _background_sender
    with _sender_lock:
        if _background_sender is None:
            _background_sender = BackgroundSender(httpx_client, config)
        return _background_sender


def shutdown_background_sender() -> None:
    """Shutdown the global background sender."""
    global _background_sender
    with _sender_lock:
        if _background_sender is not None:
            _background_sender.shutdown()
            _background_sender = None

