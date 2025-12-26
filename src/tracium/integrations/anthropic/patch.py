"""Patching logic for Anthropic SDK."""

from __future__ import annotations

from typing import Any

from ...core.client import TraciumClient
from ...helpers.global_state import STATE, get_options
from .context_manager import wrap_stream_context_manager, wrap_stream_context_manager_async
from .tracing import trace_anthropic_call, trace_anthropic_call_async

anthropic = None


def patch_anthropic(client: TraciumClient) -> None:
    """
    Patch the Anthropic SDK to automatically trace all Claude API calls.

    Supports both sync and async Anthropic clients, including streaming.
    """
    if STATE.anthropic_patched:
        return

    global anthropic
    anthropic_module = anthropic
    if anthropic_module is None:
        try:
            import anthropic as imported_anthropic
        except Exception:
            return
        anthropic = imported_anthropic
        anthropic_module = imported_anthropic

    get_options()

    if hasattr(anthropic_module, "resources") and hasattr(anthropic_module.resources, "messages"):
        try:
            target_class = anthropic_module.resources.messages.Messages
            original_create = target_class.create

            def traced_create(self, *args: Any, **kwargs: Any) -> Any:
                return trace_anthropic_call(
                    client=client,
                    original_fn=lambda: original_create(self, *args, **kwargs),
                    args=args,
                    kwargs=kwargs,
                )

            target_class.create = traced_create

            if hasattr(target_class, "stream"):
                original_stream = target_class.stream

                def traced_stream(self, *args: Any, **kwargs: Any) -> Any:
                    return wrap_stream_context_manager(
                        client=client,
                        stream_context_manager=original_stream(self, *args, **kwargs),
                        args=args,
                        kwargs=kwargs,
                    )

                target_class.stream = traced_stream
        except Exception:
            pass

    if hasattr(anthropic_module, "resources") and hasattr(anthropic_module.resources, "messages"):
        try:
            async_target_class = anthropic_module.resources.messages.AsyncMessages
            original_async_create = async_target_class.create

            async def traced_async_create(self, *args: Any, **kwargs: Any) -> Any:
                return await trace_anthropic_call_async(
                    client=client,
                    original_fn=lambda: original_async_create(self, *args, **kwargs),
                    args=args,
                    kwargs=kwargs,
                )

            async_target_class.create = traced_async_create

            if hasattr(async_target_class, "stream"):
                original_async_stream = async_target_class.stream

                async def traced_async_stream(self, *args: Any, **kwargs: Any) -> Any:
                    return await wrap_stream_context_manager_async(
                        client=client,
                        stream_context_manager=original_async_stream(self, *args, **kwargs),
                        args=args,
                        kwargs=kwargs,
                    )

                async_target_class.stream = traced_async_stream
        except Exception:
            pass

    STATE.anthropic_patched = True
