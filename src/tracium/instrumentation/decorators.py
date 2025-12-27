"""
High-level instrumentation helpers for agentic workloads.

All tracing operations are designed to be non-blocking and fail-safe.
Decorator failures will never break user applications.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

from typing_extensions import ParamSpec

from ..context.trace_context import current_trace
from ..core import TraciumClient
from ..helpers.logging_config import get_logger
from ..models.span_handle import AgentSpanHandle

P = ParamSpec("P")
R = TypeVar("R")

logger = get_logger()


def _merge_tags(
    base: Sequence[str] | None,
    extra: Sequence[str] | None,
) -> list[str]:
    try:
        merged: list[str] = []
        for source in (base, extra):
            if not source:
                continue
            for tag in source:
                if tag is None:
                    continue
                tag_str = str(tag).strip()
                if tag_str and tag_str not in merged:
                    merged.append(tag_str)
        return merged
    except Exception:
        return []


def _copy_mapping(mapping: Mapping[str, Any] | None) -> dict[str, Any] | None:
    try:
        return dict(mapping) if mapping else None
    except Exception:
        return None


def _inject_kwarg(
    kwargs: dict[str, Any],
    *,
    key: str,
    value: Any,
) -> dict[str, Any]:
    try:
        if key in kwargs:
            raise TypeError(
                f"Argument '{key}' already present when injecting Tracium instrumentation handle.",
            )
        new_kwargs = dict(kwargs)
        new_kwargs[key] = value
        return new_kwargs
    except TypeError:
        raise
    except Exception:
        return kwargs


def agent_trace(
    *,
    client: TraciumClient,
    agent_name: str,
    model_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    tags: Sequence[str] | None = None,
    trace_id: str | None = None,
    inject_trace_arg: str | None = None,
    auto_tag: bool = True,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator that wraps a function in a Tracium agent trace.

    If tracing fails, the original function will still execute normally.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        is_coroutine = inspect.iscoroutinefunction(func)
        try:
            extra_tags = [f"@trace:{func.__name__}"] if auto_tag else None
            merged_tags = _merge_tags(tags, extra_tags) or None
        except Exception:
            merged_tags = None

        def _wrap(*args: P.args, **kwargs: P.kwargs) -> R:
            trace_handle = None
            try:
                with client.agent_trace(
                    agent_name=agent_name,
                    model_id=model_id,
                    metadata=_copy_mapping(metadata),
                    tags=merged_tags,
                    trace_id=trace_id,
                ) as trace_handle:
                    if inject_trace_arg:
                        kwargs = _inject_kwarg(kwargs, key=inject_trace_arg, value=trace_handle)  # type: ignore[assignment]
                    return func(*args, **kwargs)
            except Exception as e:
                if trace_handle is None:
                    logger.debug(f"agent_trace decorator failed (running without trace): {e}")
                    return func(*args, **kwargs)
                raise

        async def _wrap_async(*args: P.args, **kwargs: P.kwargs) -> R:
            trace_handle = None
            try:
                with client.agent_trace(
                    agent_name=agent_name,
                    model_id=model_id,
                    metadata=_copy_mapping(metadata),
                    tags=merged_tags,
                    trace_id=trace_id,
                ) as trace_handle:
                    if inject_trace_arg:
                        kwargs = _inject_kwarg(kwargs, key=inject_trace_arg, value=trace_handle)  # type: ignore[assignment]
                    result = await func(*args, **kwargs)  # type: ignore[misc]
                    return result  # type: ignore[no-any-return]
            except Exception as e:
                if trace_handle is None:
                    logger.debug(f"agent_trace async decorator failed (running without trace): {e}")
                    result = await func(*args, **kwargs)  # type: ignore[misc]
                    return result  # type: ignore[no-any-return]
                raise

        wrapper = functools.wraps(func)(_wrap_async if is_coroutine else _wrap)
        return wrapper  # type: ignore[return-value]

    return decorator


def agent_span(
    *,
    span_type: str,
    name: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    tags: Sequence[str] | None = None,
    inject_span_arg: str | None = None,
    capture_return: bool = False,
    require_trace: bool = False,
    auto_tag: bool = True,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator that records the wrapped function as a span inside the active trace.

    If tracing fails, the original function will still execute normally.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        is_coroutine = inspect.iscoroutinefunction(func)
        span_name = name or func.__name__
        try:
            extra_tags = [f"@span:{span_name}"] if auto_tag else None
            merged_tags = _merge_tags(tags, extra_tags) or None
        except Exception:
            merged_tags = None

        def _call(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                trace_handle = current_trace()
                if trace_handle is None:
                    if require_trace or inject_span_arg:
                        raise RuntimeError(
                            "agent_span decorated function called outside an active Tracium agent trace.",
                        )
                    return func(*args, **kwargs)

                with trace_handle.span(
                    span_type=span_type,
                    name=span_name,
                    metadata=_copy_mapping(metadata),
                    tags=merged_tags,
                ) as span_handle:
                    if inject_span_arg:
                        kwargs = _inject_kwarg(kwargs, key=inject_span_arg, value=span_handle)  # type: ignore[assignment]
                    result = func(*args, **kwargs)
                    if capture_return:
                        try:
                            span_handle.record_output({"return": result})
                        except Exception:
                            pass
                    return result
            except RuntimeError:
                raise
            except Exception as e:
                logger.debug(f"agent_span decorator failed (running without span): {e}")
                return func(*args, **kwargs)

        async def _call_async(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                trace_handle = current_trace()
                if trace_handle is None:
                    if require_trace or inject_span_arg:
                        raise RuntimeError(
                            "agent_span decorated coroutine called outside an active Tracium agent trace.",
                        )
                    result = await func(*args, **kwargs)  # type: ignore[misc]
                    return result  # type: ignore[no-any-return]

                with trace_handle.span(
                    span_type=span_type,
                    name=span_name,
                    metadata=_copy_mapping(metadata),
                    tags=merged_tags,
                ) as span_handle:
                    if inject_span_arg:
                        kwargs = _inject_kwarg(kwargs, key=inject_span_arg, value=span_handle)  # type: ignore[assignment]
                    result = await func(*args, **kwargs)  # type: ignore[misc]
                    if capture_return:
                        try:
                            span_handle.record_output({"return": result})
                        except Exception:
                            pass
                    return result  # type: ignore[no-any-return]
            except RuntimeError:
                raise
            except Exception as e:
                logger.debug(f"agent_span async decorator failed (running without span): {e}")
                result = await func(*args, **kwargs)  # type: ignore[misc]
                return result  # type: ignore[no-any-return]

        wrapper = functools.wraps(func)(_call_async if is_coroutine else _call)
        return wrapper  # type: ignore[return-value]

    return decorator


def span(
    *,
    span_type: str,
    name: str | None = None,
    input: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
    tags: Sequence[str] | None = None,
    parent_span_id: str | None = None,
    span_id: str | None = None,
) -> contextlib.AbstractContextManager[AgentSpanHandle]:
    """
    Convenience helper that delegates to the current trace's ``span`` context manager.
    """
    trace_handle = current_trace()
    if trace_handle is None:
        raise RuntimeError("span() requires an active Tracium agent trace.")
    return trace_handle.span(
        span_type=span_type,
        name=name,
        input=input,
        metadata=metadata,
        tags=tags,
        parent_span_id=parent_span_id,
        span_id=span_id,
    )
