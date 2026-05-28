"""Tests for the lazy-start + idle-shutdown background sender, and the
serverless sync-send mode. These guarantee that Tracium's daemon thread is
absent during interactive debug sessions and that FaaS environments never
spawn a worker thread."""

from __future__ import annotations

import os
import time
from unittest.mock import MagicMock

import httpx
import pytest

from tracium.core.config import TraciumClientConfig
from tracium.helpers.background_sender import (
    BackgroundSender,
    RequestMethod,
    _is_serverless_env,
)


@pytest.fixture
def mock_httpx_client():
    client = MagicMock(spec=httpx.Client)
    response = MagicMock()
    response.status_code = 200
    response.raise_for_status = MagicMock()
    response.json = MagicMock(return_value={"ok": True})
    client.post = MagicMock(return_value=response)
    client.get = MagicMock(return_value=response)
    client.patch = MagicMock(return_value=response)
    return client


@pytest.fixture
def config():
    return TraciumClientConfig()


class TestLazyStart:
    def test_no_thread_at_construction(self, mock_httpx_client, config):
        """Worker thread must not exist until something is enqueued."""
        sender = BackgroundSender(mock_httpx_client, config, idle_timeout=1.0)
        assert sender._worker_thread is None

    def test_thread_starts_on_first_enqueue(self, mock_httpx_client, config):
        sender = BackgroundSender(mock_httpx_client, config, idle_timeout=1.0)
        sender.enqueue(RequestMethod.POST, "/spans", json={"x": 1})
        assert sender._worker_thread is not None
        assert sender._worker_thread.is_alive()
        sender.shutdown()

    def test_thread_self_terminates_after_idle(self, mock_httpx_client, config):
        sender = BackgroundSender(mock_httpx_client, config, idle_timeout=0.3)
        sender.enqueue(RequestMethod.POST, "/spans", json={"x": 1})
        sender.flush()
        # Wait past idle_timeout so the worker exits on its own.
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            t = sender._worker_thread
            if t is None or not t.is_alive():
                break
            time.sleep(0.1)
        t = sender._worker_thread
        assert t is None or not t.is_alive(), "worker did not self-terminate after idle"

    def test_thread_restarts_after_idle_shutdown(self, mock_httpx_client, config):
        sender = BackgroundSender(mock_httpx_client, config, idle_timeout=0.3)
        sender.enqueue(RequestMethod.POST, "/spans", json={"x": 1})
        sender.flush()
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            t = sender._worker_thread
            if t is None or not t.is_alive():
                break
            time.sleep(0.1)
        # Second enqueue should bring it back.
        sender.enqueue(RequestMethod.POST, "/spans", json={"y": 2})
        assert sender._worker_thread is not None
        assert sender._worker_thread.is_alive()
        sender.shutdown()


class TestServerlessSyncMode:
    def test_sync_mode_never_starts_thread(self, mock_httpx_client, config):
        sender = BackgroundSender(mock_httpx_client, config, sync_mode=True)
        sender.enqueue(RequestMethod.POST, "/spans", json={"x": 1})
        # Even after enqueue, no thread exists.
        assert sender._worker_thread is None

    def test_sync_mode_posts_inline(self, mock_httpx_client, config):
        sender = BackgroundSender(mock_httpx_client, config, sync_mode=True)
        sender.enqueue(RequestMethod.POST, "/spans", json={"x": 1})
        # POST must have been called synchronously, not deferred to a worker.
        assert mock_httpx_client.post.called

    def test_flush_is_noop_in_sync_mode(self, mock_httpx_client, config):
        sender = BackgroundSender(mock_httpx_client, config, sync_mode=True)
        sender.flush()  # must return immediately, not block on a non-existent queue


class TestServerlessDetection:
    @pytest.fixture(autouse=True)
    def _clear_env(self, monkeypatch):
        for k in (
            "AWS_LAMBDA_FUNCTION_NAME",
            "AWS_EXECUTION_ENV",
            "K_SERVICE",
            "FUNCTION_TARGET",
            "FUNCTION_NAME",
            "VERCEL",
            "FUNCTIONS_WORKER_RUNTIME",
            "TRACIUM_FORCE_SYNC",
        ):
            monkeypatch.delenv(k, raising=False)

    def test_no_env_means_not_serverless(self):
        assert not _is_serverless_env()

    @pytest.mark.parametrize(
        "env_var,value",
        [
            ("AWS_LAMBDA_FUNCTION_NAME", "my-fn"),
            ("K_SERVICE", "my-service"),
            ("FUNCTION_TARGET", "main"),
            ("FUNCTION_NAME", "main"),
            ("VERCEL", "1"),
            ("FUNCTIONS_WORKER_RUNTIME", "python"),
            ("TRACIUM_FORCE_SYNC", "1"),
        ],
    )
    def test_serverless_envs_trigger_sync_mode(self, env_var, value, monkeypatch, mock_httpx_client, config):
        monkeypatch.setenv(env_var, value)
        assert _is_serverless_env()
        sender = BackgroundSender(mock_httpx_client, config)
        assert sender._sync_mode is True
        sender.enqueue(RequestMethod.POST, "/spans", json={"x": 1})
        assert sender._worker_thread is None
        assert mock_httpx_client.post.called
