"""
Minimal LangGraph instrumentation for Tracium.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any

from ..core.client import TraciumClient
from ..helpers.global_state import STATE, get_default_tags, get_options
from ..instrumentation.auto_trace_tracker import get_or_create_auto_trace


def _wrap_executor_method(
    client: TraciumClient, method: Callable[..., Any], method_name: str
) -> Callable[..., Any]:
    @functools.wraps(method)
    def wrapper(self, *args: Any, **kwargs: Any) -> Any:
        options = get_options()

        trace_handle, created_trace = get_or_create_auto_trace(
            client=client,
            agent_name=options.default_agent_name or "langgraph",
            model_id=options.default_model_id,
            tags=get_default_tags(["@langgraph"]),
        )

        if created_trace:
            trace_handle.set_summary(
                {
                    **options.default_metadata,
                    "provider": "langgraph",
                    "entrypoint": method_name,
                }
            )

        span_context = trace_handle.span(
            span_type="graph",
            name=getattr(self, "graph_id", method_name),
            metadata={"provider": "langgraph", "method": method_name},
        )
        span_handle = span_context.__enter__()
        span_handle.record_input({"args": args, "kwargs": kwargs})

        start = time.time()
        try:
            result = method(self, *args, **kwargs)
        except Exception as exc:
            latency_ms = int((time.time() - start) * 1000)
            span_handle.add_metadata({"latency_ms": latency_ms})
            span_handle.mark_failed(str(exc))
            span_context.__exit__(type(exc), exc, exc.__traceback__)
            raise

        latency_ms = int((time.time() - start) * 1000)
        span_handle.add_metadata({"latency_ms": latency_ms})
        span_handle.record_output(result)
        span_context.__exit__(None, None, None)

        return result

    return wrapper


def register_langgraph_hooks(client: TraciumClient) -> None:
    if STATE.langgraph_registered:
        return
    try:
        from langgraph.graph.executor import Executor  # type: ignore[import]
    except Exception:
        return

    for method_name in ("invoke", "ainvoke"):
        if hasattr(Executor, method_name):
            original = getattr(Executor, method_name)
            wrapped = _wrap_executor_method(client, original, method_name)
            setattr(Executor, method_name, wrapped)

    STATE.langgraph_registered = True
