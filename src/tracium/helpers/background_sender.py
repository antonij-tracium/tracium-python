"""
Background async sender for non-blocking telemetry data transmission.

This module provides a thread-safe queue-based system for sending telemetry data
to the Tracium API without blocking user code. All API calls are processed
asynchronously in a background thread.
"""

from __future__ import annotations

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
        queue_size = getattr(config, "max_queue_size", max_queue_size)
        self._max_queue_size = queue_size
        self._queue: queue.Queue[QueuedRequest | None] = queue.Queue(maxsize=queue_size)
        self._shutdown = threading.Event()
        self._flush_timeout = flush_timeout
        self._worker_thread: threading.Thread | None = None
        self._started = False
        self._lock = threading.Lock()

        self._total_enqueued = 0
        self._total_dropped = 0
        self._total_sent = 0
        self._total_failed = 0
        self._last_warning_time = 0.0

        self._start_worker()

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
        while True:
            try:
                try:
                    request = self._queue.get(timeout=0.1)
                except queue.Empty:
                    if self._shutdown.is_set():
                        break
                    continue

                if request is None:
                    break

                self._process_request(request)
                self._queue.task_done()

            except Exception as e:
                logger.debug(f"Background sender error (ignored): {type(e).__name__}: {e}")

    def _process_request(self, request: QueuedRequest) -> None:
        """Process a single queued request with retry logic."""
        try:
            if self._config.security_config:
                is_allowed, wait_time = check_rate_limit(self._config.security_config)
                if not is_allowed:
                    logger.debug(
                        f"Rate limited, waiting {wait_time}s before request to {request.path}"
                    )
                    time.sleep(wait_time)

            payload = None
            if request.json is not None and self._config.security_config:
                payload = redact_telemetry_payload(request.json, self._config.security_config)
            elif request.json is not None:
                payload = request.json

            retry_config = self._config.retry_config or RetryConfig()
            last_exception: Exception | None = None

            for attempt in range(retry_config.max_retries + 1):
                try:
                    response = self._make_request(
                        request.method, request.path, payload, request.params, request.headers
                    )
                    response.raise_for_status()

                    if request.callback:
                        try:
                            request.callback(response.json())
                        except Exception:
                            pass

                    self._total_sent += 1
                    logger.debug(
                        f"Background request successful: {request.method.value} {request.path}"
                    )
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

            if last_exception:
                self._total_failed += 1
                logger.warning(
                    f"Background request failed after retries: {request.method.value} {request.path} - "
                    f"{type(last_exception).__name__}: {last_exception}. "
                    f"Total failures: {self._total_failed}"
                )

        except Exception as e:
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

        Returns True if successfully queued, False if queue is full and blocking is disabled.
        Never raises exceptions.

        If block_on_full_queue is enabled, this will wait up to queue_timeout seconds
        for space in the queue, preventing event loss.
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

            # Check queue capacity and warn if needed
            current_size = self._queue.qsize()
            threshold = getattr(self._config, "queue_warning_threshold", 0.8)
            capacity_ratio = current_size / self._max_queue_size

            if capacity_ratio >= threshold:
                current_time = time.time()
                # Warn at most once per minute to avoid log spam
                if current_time - self._last_warning_time > 60:
                    logger.warning(
                        f"Tracium queue at {capacity_ratio:.0%} capacity "
                        f"({current_size}/{self._max_queue_size}). "
                        f"Stats: {self._total_enqueued} enqueued, {self._total_dropped} dropped, "
                        f"{self._total_sent} sent, {self._total_failed} failed. "
                        f"Consider increasing max_queue_size in TraciumClientConfig."
                    )
                    self._last_warning_time = current_time

            # Try to enqueue based on blocking configuration
            block_on_full = getattr(self._config, "block_on_full_queue", False)
            timeout = getattr(self._config, "queue_timeout", 5.0) if block_on_full else None

            if block_on_full and timeout:
                # Blocking mode - wait for space in queue
                try:
                    self._queue.put(request, timeout=timeout)
                    self._total_enqueued += 1
                    return True
                except queue.Full:
                    # Even with blocking, we timed out
                    self._total_dropped += 1
                    logger.error(
                        f"Tracium queue full after waiting {timeout}s. "
                        f"Dropping event to {path}. Total dropped: {self._total_dropped}. "
                        "Your application is generating events faster than they can be sent. "
                        "Consider: (1) Increasing max_queue_size, (2) Reducing event volume, "
                        "or (3) Increasing queue_timeout."
                    )
                    return False
            else:
                self._queue.put_nowait(request)
                self._total_enqueued += 1
                return True

        except queue.Full:
            self._total_dropped += 1
            logger.error(
                f"Tracium queue full ({self._max_queue_size}). Dropping event to {path}. "
                f"Total dropped: {self._total_dropped}. "
                "To prevent event loss, enable block_on_full_queue=True in TraciumClientConfig "
                "or increase max_queue_size."
            )
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

            try:
                self._queue.put(None, timeout=0.5)
            except Exception:
                pass

            if self._worker_thread and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=self._flush_timeout)

        except Exception:
            pass

    def shutdown(self) -> None:
        """Explicitly shutdown the background sender."""
        self._cleanup()

    def get_stats(self) -> dict[str, Any]:
        """
        Get comprehensive statistics about the background sender.
        Returns:
            Dictionary with queue metrics, event counts, and health indicators
        Example:
            >>> stats = sender.get_stats()
            >>> print(f"Queue at {stats['capacity_percent']:.1f}% capacity")
            >>> print(f"Dropped {stats['total_dropped']} events")
        """
        try:
            current_size = self._queue.qsize()
            capacity = current_size / self._max_queue_size if self._max_queue_size > 0 else 0

            return {
                "queue_size": current_size,
                "max_queue_size": self._max_queue_size,
                "capacity_percent": capacity * 100,
                "is_healthy": capacity < 0.9 and self._total_dropped == 0,
                "total_enqueued": self._total_enqueued,
                "total_sent": self._total_sent,
                "total_failed": self._total_failed,
                "total_dropped": self._total_dropped,
                "success_rate": (
                    self._total_sent / self._total_enqueued if self._total_enqueued > 0 else 1.0
                ),
                "drop_rate": (
                    self._total_dropped / self._total_enqueued if self._total_enqueued > 0 else 0.0
                ),
                # Configuration
                "blocking_enabled": getattr(self._config, "block_on_full_queue", False),
                "queue_timeout": getattr(self._config, "queue_timeout", 5.0),
            }
        except Exception as e:
            logger.debug(f"Failed to get stats: {e}")
            return {
                "error": str(e),
                "max_queue_size": self._max_queue_size,
                "total_dropped": self._total_dropped,
            }


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
