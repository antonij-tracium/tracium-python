"""
Minimal LangGraph instrumentation for Tracium.

All tracing operations are designed to be non-blocking and fail-safe.
Tracing errors will never break user applications.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any

from ..context.trace_context import current_trace
from ..core.client import TraciumClient
from ..helpers.global_state import STATE
from ..helpers.logging_config import get_logger

logger = get_logger()


def _wrap_executor_method(
    client: TraciumClient, method: Callable[..., Any], method_name: str
) -> Callable[..., Any]:
    @functools.wraps(method)
    def wrapper(self, *args: Any, **kwargs: Any) -> Any:
        trace_handle = None
        span_context = None
        span_handle = None

        try:
            trace_handle = current_trace()
            if trace_handle is None:
                return method(self, *args, **kwargs)

            span_context = trace_handle.span(
                span_type="graph",
                name=getattr(self, "graph_id", method_name),
                metadata={"provider": "langgraph", "method": method_name},
            )
            span_handle = span_context.__enter__()
            span_handle.record_input({"args": args, "kwargs": kwargs})
        except Exception as e:
            logger.debug(f"LangGraph trace setup failed (continuing without tracing): {e}")

        start = time.time()

        try:
            result = method(self, *args, **kwargs)
        except Exception as exc:
            if span_handle and span_context:
                try:
                    latency_ms = int((time.time() - start) * 1000)
                    span_handle.add_metadata({"latency_ms": latency_ms})
                    span_handle.mark_failed(str(exc))
                    span_context.__exit__(type(exc), exc, exc.__traceback__)
                except Exception:
                    pass
            raise

        try:
            if span_handle and span_context:
                latency_ms = int((time.time() - start) * 1000)
                span_handle.add_metadata({"latency_ms": latency_ms})
                span_handle.record_output(result)
                span_context.__exit__(None, None, None)
        except Exception as e:
            logger.debug(f"LangGraph response tracing failed (ignored): {e}")

        return result

    return wrapper


def _wrap_executor_method_async(
    client: TraciumClient, method: Callable[..., Any], method_name: str
) -> Callable[..., Any]:
    @functools.wraps(method)
    async def wrapper(self, *args: Any, **kwargs: Any) -> Any:
        trace_handle = None
        span_context = None
        span_handle = None

        try:
            trace_handle = current_trace()
            if trace_handle is None:
                return await method(self, *args, **kwargs)

            span_context = trace_handle.span(
                span_type="graph",
                name=getattr(self, "graph_id", method_name),
                metadata={"provider": "langgraph", "method": method_name},
            )
            span_handle = span_context.__enter__()
            span_handle.record_input({"args": args, "kwargs": kwargs})
        except Exception as e:
            logger.debug(f"LangGraph async trace setup failed (continuing without tracing): {e}")

        start = time.time()

        try:
            result = await method(self, *args, **kwargs)
        except Exception as exc:
            if span_handle and span_context:
                try:
                    latency_ms = int((time.time() - start) * 1000)
                    span_handle.add_metadata({"latency_ms": latency_ms})
                    span_handle.mark_failed(str(exc))
                    span_context.__exit__(type(exc), exc, exc.__traceback__)
                except Exception:
                    pass
            raise

        try:
            if span_handle and span_context:
                latency_ms = int((time.time() - start) * 1000)
                span_handle.add_metadata({"latency_ms": latency_ms})
                span_handle.record_output(result)
                span_context.__exit__(None, None, None)
        except Exception as e:
            logger.debug(f"LangGraph async response tracing failed (ignored): {e}")

        return result

    return wrapper


def register_langgraph_hooks(client: TraciumClient) -> None:
    if STATE.langgraph_registered:
        return
    try:
        from langgraph.graph.executor import Executor
    except Exception:
        return

    try:
        if hasattr(Executor, "invoke"):
            original = getattr(Executor, "invoke")
            wrapped = _wrap_executor_method(client, original, "invoke")
            setattr(Executor, "invoke", wrapped)

        if hasattr(Executor, "ainvoke"):
            original = getattr(Executor, "ainvoke")
            wrapped = _wrap_executor_method_async(client, original, "ainvoke")
            setattr(Executor, "ainvoke", wrapped)
    except Exception:
        pass

    STATE.langgraph_registered = True
