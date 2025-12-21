"""
LangChain auto-instrumentation hooks for Tracium.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from ..context.trace_context import current_trace
from ..core import TraciumClient
from ..helpers.global_state import (
    STATE,
    add_langchain_active_run,
    get_default_tags,
    get_options,
    remove_langchain_active_run,
)
from ..instrumentation.auto_trace_tracker import get_or_create_auto_trace
from ..models.trace_handle import AgentTraceHandle, AgentTraceManager

try:
    from langchain_core.callbacks import BaseCallbackHandler
except Exception:
    BaseCallbackHandler = None


@dataclass
class _TrackedTrace:
    manager: AgentTraceManager | None
    handle: AgentTraceHandle
    owned: bool = True


class TraciumLangChainHandler(BaseCallbackHandler):
    """
    A LangChain callback handler that mirrors chains, LLM calls, and tool usage into
    Tracium agent traces / spans.
    """

    def __init__(self, client: TraciumClient) -> None:
        self._client = client
        self._root_traces: dict[str, _TrackedTrace] = {}
        self._trace_mapping: dict[str, str] = {}
        self._active_spans: dict[str, tuple[Any, Any]] = {}
        self._lock = threading.RLock()

    def _create_trace(
        self,
        run_id: str,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
    ) -> None:
        options = get_options()
        existing = current_trace()
        if existing is not None:
            tracked = _TrackedTrace(manager=None, handle=existing, owned=False)
        else:
            agent_name = (serialized.get("name") or "").strip() or ""

            tags = get_default_tags(["@langchain"])
            metadata = {**options.default_metadata}
            metadata["langchain_serialized"] = serialized
            metadata["langchain_inputs"] = inputs

            handle, created_new = get_or_create_auto_trace(
                client=self._client,
                agent_name=agent_name,
                model_id=options.default_model_id,
                tags=tags,
            )

            if created_new:
                handle.set_summary(metadata)

            tracked = _TrackedTrace(
                manager=None,
                handle=handle,
                owned=False,
            )

        with self._lock:
            self._root_traces[run_id] = tracked
            self._trace_mapping[run_id] = run_id

    def _close_run(
        self,
        run_id: str,
        *,
        outputs: dict[str, Any] | None = None,
        error: BaseException | None = None,
    ) -> None:
        tracked = None
        with self._lock:
            tracked = self._root_traces.pop(run_id, None)
            self._trace_mapping.pop(run_id, None)
        if not tracked:
            return

        handle = tracked.handle
        if outputs:
            handle.set_summary({"langchain_outputs": self._serialize_input(outputs)})
        if error:
            handle.mark_failed(str(error))
        if tracked.owned and tracked.manager is not None:
            tracked.manager.__exit__(
                type(error) if error else None,
                error,
                error.__traceback__ if error else None,
            )

    def _start_span(
        self,
        *,
        lc_run_id: str,
        owner_run_id: str,
        kind: str,
        name: str | None,
        input_payload: Any,
        model_id: str | None = None,
    ) -> None:
        with self._lock:
            if lc_run_id in self._active_spans:
                return

            tracked = self._root_traces.get(owner_run_id)

            if tracked is None:
                return

            handle = tracked.handle

            span_kwargs: dict[str, Any] = {
                "span_type": kind,
                "name": name,
                "metadata": {"source": "langchain"},
            }
            if model_id:
                span_kwargs["model_id"] = model_id

            context = handle.span(**span_kwargs)
            span_handle = context.__enter__()
            span_handle.record_input(self._serialize_input(input_payload))

            with self._lock:
                self._active_spans[lc_run_id] = (context, span_handle)
                self._trace_mapping[lc_run_id] = owner_run_id

    def _finish_span(
        self,
        *,
        lc_run_id: str,
        owner_run_id: str | None = None,
        output_payload: Any = None,
        error: BaseException | None = None,
    ) -> None:
        with self._lock:
            entry = self._active_spans.pop(lc_run_id, None)
            if owner_run_id:
                self._trace_mapping.pop(lc_run_id, None)
        if not entry:
            return

        context, span_handle = entry
        if output_payload is not None:
            span_handle.record_output(self._serialize_input(output_payload))
        if error:
            span_handle.mark_failed(str(error))
        context.__exit__(
            type(error) if error else None, error, error.__traceback__ if error else None
        )

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any],
        run_id: str,
        parent_run_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        if parent_run_id is None:
            self._create_trace(run_id, serialized or {}, inputs)
            return

        if not serialized:
            return

        node_id = serialized.get("id", "")
        if isinstance(node_id, str) and node_id.startswith("langchain.chat_models"):
            return

        owner = self._trace_mapping.get(parent_run_id, parent_run_id)
        self._trace_mapping[run_id] = owner

    async def on_chain_start_async(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        run_id: str,
        parent_run_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.on_chain_start(serialized, inputs, run_id, parent_run_id, **kwargs)

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        run_id: str,
        parent_run_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        if parent_run_id is None:
            self._close_run(run_id, outputs=outputs)
        else:
            owner = self._trace_mapping.get(parent_run_id, parent_run_id)
            self._finish_span(
                lc_run_id=run_id, owner_run_id=owner, output_payload=self._serialize_input(outputs)
            )
            with self._lock:
                self._trace_mapping.pop(run_id, None)

    async def on_chain_end_async(
        self,
        outputs: dict[str, Any],
        run_id: str,
        parent_run_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.on_chain_end(outputs, run_id, parent_run_id, **kwargs)

    def on_chain_error(
        self, error: BaseException, run_id: str, parent_run_id: str | None = None, **kwargs: Any
    ) -> None:
        if parent_run_id is None:
            self._close_run(run_id, error=error)
        else:
            owner = self._trace_mapping.get(parent_run_id, parent_run_id)
            self._finish_span(lc_run_id=run_id, owner_run_id=owner, error=error)
            with self._lock:
                self._trace_mapping.pop(run_id, None)

    async def on_chain_error_async(
        self, error: BaseException, run_id: str, parent_run_id: str | None = None, **kwargs: Any
    ) -> None:
        self.on_chain_error(error, run_id, parent_run_id, **kwargs)

    def _serialize_input(self, obj: Any) -> Any:
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, dict):
            return {k: self._serialize_input(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [self._serialize_input(item) for item in obj]

        if hasattr(obj, "choices") and hasattr(obj, "model"):
            try:
                result = {"model": obj.model, "choices": []}
                for choice in obj.choices:
                    choice_data = {
                        "index": getattr(choice, "index", 0),
                        "finish_reason": getattr(choice, "finish_reason", None),
                    }
                    if hasattr(choice, "message"):
                        message = choice.message
                        choice_data["message"] = {
                            "role": getattr(message, "role", "assistant"),
                            "content": getattr(message, "content", ""),
                        }
                        if hasattr(message, "tool_calls") and message.tool_calls:
                            choice_data["message"]["tool_calls"] = [
                                {
                                    "id": tc.id,
                                    "type": tc.type,
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                for tc in message.tool_calls
                            ]
                    elif hasattr(choice, "text"):
                        choice_data["text"] = choice.text
                    result["choices"].append(choice_data)

                if hasattr(obj, "usage") and obj.usage:
                    result["usage"] = {
                        "prompt_tokens": getattr(obj.usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(obj.usage, "completion_tokens", 0),
                        "total_tokens": getattr(obj.usage, "total_tokens", 0),
                    }

                return result
            except Exception:
                pass

        if hasattr(obj, "content"):
            return str(obj.content)

        if hasattr(obj, "to_string"):
            return obj.to_string()

        if hasattr(obj, "generations") and obj.generations:
            texts: list[str] = []
            for gen_list in obj.generations:
                for gen in gen_list:
                    if hasattr(gen, "text"):
                        texts.append(gen.text)
                    elif hasattr(gen, "message") and hasattr(gen.message, "content"):
                        texts.append(str(gen.message.content))
                    elif hasattr(gen, "content"):
                        texts.append(str(gen.content))
            return "\n\n".join(texts) if texts else str(obj)

        return str(obj)

    def on_llm_start(self, serialized, prompts, run_id, parent_run_id=None, **kwargs):
        return

    async def on_llm_start_async(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        run_id: str,
        parent_run_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.on_llm_start(serialized, prompts, run_id, parent_run_id, **kwargs)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        run_id: str,
        parent_run_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        add_langchain_active_run(run_id)

        owner = self._trace_mapping.get(parent_run_id or run_id, parent_run_id or run_id)
        if not parent_run_id:
            with self._lock:
                if owner not in self._root_traces:
                    self._create_trace(
                        run_id=owner, serialized=serialized or {}, inputs={"messages": messages}
                    )

        if parent_run_id:
            model_id = (
                kwargs.get("invocation_params", {}).get("model_name")
                or kwargs.get("invocation_params", {}).get("model")
                or serialized.get("kwargs", {}).get("model_name")
                or serialized.get("kwargs", {}).get("model")
                or serialized.get("id", [None])[-1]
                if isinstance(serialized.get("id"), list)
                else None
            )
            serialized_messages = self._serialize_input(messages)

            self._start_span(
                lc_run_id=run_id,
                owner_run_id=owner,
                kind="llm",
                name=serialized.get("name"),
                input_payload={"messages": serialized_messages},
                model_id=model_id,
            )

    async def on_chat_model_start_async(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        run_id: str,
        parent_run_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.on_chat_model_start(serialized, messages, run_id, parent_run_id, **kwargs)

    def on_chat_model_end(
        self, response: Any, run_id: str, parent_run_id: str | None = None, **kwargs: Any
    ) -> None:
        try:
            self.on_llm_end(response, run_id, parent_run_id, **kwargs)
        finally:
            remove_langchain_active_run(run_id)

    async def on_chat_model_end_async(
        self, response: Any, run_id: str, parent_run_id: str | None = None, **kwargs: Any
    ) -> None:
        self.on_chat_model_end(response, run_id, parent_run_id, **kwargs)

    def _extract_token_usage(self, response: Any, **kwargs: Any) -> dict[str, Any]:
        metadata: dict[str, Any] = {}

        try:
            if hasattr(response, "usage") and response.usage:
                usage = response.usage

                usage_dict = None
                if isinstance(usage, dict):
                    usage_dict = usage
                elif hasattr(usage, "model_dump"):
                    try:
                        usage_dict = usage.model_dump()
                    except Exception:
                        pass
                elif hasattr(usage, "dict"):
                    try:
                        usage_dict = usage.dict()
                    except Exception:
                        pass

                if usage_dict:
                    if "input_tokens" in usage_dict:
                        metadata["input_tokens"] = usage_dict["input_tokens"]
                    elif "prompt_tokens" in usage_dict:
                        metadata["input_tokens"] = usage_dict["prompt_tokens"]

                    if "output_tokens" in usage_dict:
                        metadata["output_tokens"] = usage_dict["output_tokens"]
                    elif "completion_tokens" in usage_dict:
                        metadata["output_tokens"] = usage_dict["completion_tokens"]

                    if "total_tokens" in usage_dict:
                        metadata["total_tokens"] = usage_dict["total_tokens"]
                else:
                    if hasattr(usage, "input_tokens"):
                        metadata["input_tokens"] = usage.input_tokens
                    elif hasattr(usage, "prompt_tokens"):
                        metadata["input_tokens"] = usage.prompt_tokens

                    if hasattr(usage, "output_tokens"):
                        metadata["output_tokens"] = usage.output_tokens
                    elif hasattr(usage, "completion_tokens"):
                        metadata["output_tokens"] = usage.completion_tokens

                    if hasattr(usage, "total_tokens"):
                        metadata["total_tokens"] = usage.total_tokens

            elif hasattr(response, "llm_output") and response.llm_output:
                llm_output = response.llm_output
                if isinstance(llm_output, dict):
                    usage = llm_output.get("usage", {})
                    if usage:
                        if isinstance(usage, dict):
                            if "input_tokens" in usage:
                                metadata["input_tokens"] = usage["input_tokens"]
                            if "output_tokens" in usage:
                                metadata["output_tokens"] = usage["output_tokens"]

                    token_usage = llm_output.get("token_usage", {})
                    if token_usage and not metadata:
                        if "prompt_tokens" in token_usage:
                            metadata["input_tokens"] = token_usage["prompt_tokens"]
                        if "completion_tokens" in token_usage:
                            metadata["output_tokens"] = token_usage["completion_tokens"]
                        if "total_tokens" in token_usage:
                            metadata["total_tokens"] = token_usage["total_tokens"]
        except Exception:
            pass

        return metadata

    def on_llm_end(
        self, response: Any, run_id: str, parent_run_id: str | None = None, **kwargs: Any
    ) -> None:  # type: ignore[override]
        metadata = self._extract_token_usage(response, **kwargs)
        payload = self._serialize_input(response)

        if parent_run_id:
            owner = self._trace_mapping.get(parent_run_id, parent_run_id)

            if metadata:
                token_kwargs: dict[str, Any] = {}
                if "input_tokens" in metadata:
                    token_kwargs["input_tokens"] = metadata["input_tokens"]
                if "output_tokens" in metadata:
                    token_kwargs["output_tokens"] = metadata["output_tokens"]

                if token_kwargs:
                    with self._lock:
                        entry = self._active_spans.get(run_id)
                        if entry:
                            _, span_handle = entry
                            span_handle.set_token_usage(**token_kwargs)

            self._finish_span(lc_run_id=run_id, owner_run_id=owner, output_payload=payload)
            with self._lock:
                self._trace_mapping.pop(run_id, None)
        else:
            with self._lock:
                tracked = self._root_traces.get(run_id)
                if tracked:
                    handle = tracked.handle
                    summary = {"output": payload}
                    if metadata:
                        summary["token_usage"] = metadata
                    handle.set_summary(summary)

            self._close_run(run_id, outputs={"output": payload})
            with self._lock:
                self._trace_mapping.pop(run_id, None)
            remove_langchain_active_run(run_id)

    async def on_llm_end_async(
        self, response: Any, run_id: str, parent_run_id: str | None = None, **kwargs: Any
    ) -> None:
        self.on_llm_end(response, run_id, parent_run_id, **kwargs)

    def on_llm_error(
        self, error: BaseException, run_id: str, parent_run_id: str | None = None, **kwargs: Any
    ) -> None:  # type: ignore[override]
        owner = self._trace_mapping.get(parent_run_id or run_id, parent_run_id or run_id)
        self._finish_span(lc_run_id=run_id, owner_run_id=owner, error=error)
        with self._lock:
            self._trace_mapping.pop(run_id, None)
        remove_langchain_active_run(run_id)

    async def on_llm_error_async(
        self, error: BaseException, run_id: str, parent_run_id: str | None = None, **kwargs: Any
    ) -> None:
        self.on_llm_error(error, run_id, parent_run_id, **kwargs)

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        run_id: str,
        parent_run_id: str | None = None,
        **kwargs: Any,
    ) -> None:  # type: ignore[override]
        owner = self._trace_mapping.get(parent_run_id or run_id, parent_run_id or run_id)
        self._start_span(
            lc_run_id=run_id,
            owner_run_id=owner,
            kind="tool",
            name=serialized.get("name"),
            input_payload={"input": input_str},
        )

    async def on_tool_start_async(
        self,
        serialized: dict[str, Any],
        input_str: str,
        run_id: str,
        parent_run_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.on_tool_start(serialized, input_str, run_id, parent_run_id, **kwargs)

    def on_tool_end(
        self, output: Any, run_id: str, parent_run_id: str | None = None, **kwargs: Any
    ) -> None:  # type: ignore[override]
        owner = self._trace_mapping.get(parent_run_id or run_id, parent_run_id or run_id)
        self._finish_span(
            lc_run_id=run_id, owner_run_id=owner, output_payload=self._serialize_input(output)
        )
        with self._lock:
            self._trace_mapping.pop(run_id, None)

    async def on_tool_end_async(
        self, output: Any, run_id: str, parent_run_id: str | None = None, **kwargs: Any
    ) -> None:
        self.on_tool_end(output, run_id, parent_run_id, **kwargs)

    def on_tool_error(
        self, error: BaseException, run_id: str, parent_run_id: str | None = None, **kwargs: Any
    ) -> None:  # type: ignore[override]
        owner = self._trace_mapping.get(parent_run_id or run_id, parent_run_id or run_id)
        self._finish_span(lc_run_id=run_id, owner_run_id=owner, error=error)
        with self._lock:
            self._trace_mapping.pop(run_id, None)

    async def on_tool_error_async(
        self, error: BaseException, run_id: str, parent_run_id: str | None = None, **kwargs: Any
    ) -> None:
        self.on_tool_error(error, run_id, parent_run_id, **kwargs)


def register_langchain_handler(client: TraciumClient) -> None:
    if BaseCallbackHandler is None or STATE.langchain_registered:
        return
    try:
        from langchain_core.callbacks.manager import AsyncCallbackManager, CallbackManager
    except Exception:
        return

    handler = TraciumLangChainHandler(client)

    def _augment_manager(manager_cls: Any) -> None:
        original_init = manager_cls.__init__

        def patched_init(self, *args: Any, **kwargs: Any) -> None:
            original_init(self, *args, **kwargs)
            try:
                self.add_handler(handler, inherit=True)
            except Exception:
                pass

        manager_cls.__init__ = patched_init

    _augment_manager(CallbackManager)
    _augment_manager(AsyncCallbackManager)
    STATE.langchain_registered = True
