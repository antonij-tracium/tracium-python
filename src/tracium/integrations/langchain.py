"""
LangChain auto-instrumentation hooks for Tracium.
Enhanced version with full LangChain support including:
- Traditional completion LLMs
- Streaming responses
- Agent-specific patterns
- Retrievers and RAG workflows
- Custom chain types
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import UUID

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

if TYPE_CHECKING:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.documents import Document
else:
    try:
        from langchain_core.callbacks import BaseCallbackHandler
    except Exception:
        BaseCallbackHandler = None  # type: ignore[assignment,misc]


@dataclass
class _TrackedTrace:
    manager: AgentTraceManager | None
    handle: AgentTraceHandle
    owned: bool = True


@dataclass
class _StreamBuffer:
    """Buffer for accumulating streaming text."""

    chunks: list[str]

    def add_chunk(self, chunk: str) -> None:
        self.chunks.append(chunk)

    def get_accumulated(self) -> str:
        return "".join(self.chunks)


if BaseCallbackHandler is not None:

    class TraciumLangChainHandler(BaseCallbackHandler):
        """
        A comprehensive LangChain callback handler that mirrors all LangChain operations
        into Tracium agent traces / spans.
        Supports:
        - Chat models and traditional completion LLMs
        - Streaming responses
        - Agent actions and decisions
        - Tool usage
        - Retrievers (RAG workflows)
        - Custom chain types
        """

        def __init__(self, client: TraciumClient) -> None:
            self._client = client
            self._root_traces: dict[str, _TrackedTrace] = {}
            self._trace_mapping: dict[str, str] = {}
            self._active_spans: dict[str, tuple[Any, Any]] = {}
            self._stream_buffers: dict[str, _StreamBuffer] = {}
            self._lock = threading.RLock()

        def _create_trace(
            self,
            run_id: str,
            serialized: dict[str, Any],
            inputs: dict[str, Any],
        ) -> None:
            options = get_options()

            from ..instrumentation.auto_trace_tracker import (
                _find_workflow_entry_point,
                cleanup_auto_trace,
                get_current_auto_trace_context,
            )

            existing_context = get_current_auto_trace_context()
            should_reuse = False
            if existing_context is not None:
                current_entry_frame_id, _ = _find_workflow_entry_point()
                if existing_context.entry_frame_id == current_entry_frame_id:
                    should_reuse = True
                else:
                    cleanup_auto_trace()

            _, entry_function_name = _find_workflow_entry_point()

            if entry_function_name and entry_function_name not in ("workflow", "<module>"):
                agent_name = entry_function_name.replace("_", "-")
                if agent_name.startswith("test-"):
                    agent_name = agent_name[5:]
                if agent_name.endswith("-main"):
                    agent_name = agent_name[:-5]
            else:
                agent_name = options.default_agent_name

            tags = get_default_tags(["@langchain"])
            metadata = {**options.default_metadata}
            metadata["langchain_serialized"] = self._serialize_input(serialized)
            metadata["langchain_inputs"] = self._serialize_input(inputs)

            handle, created_new = get_or_create_auto_trace(
                client=self._client,
                agent_name=agent_name,
                model_id=options.default_model_id,
                tags=tags,
            )

            if created_new:
                handle.set_summary(metadata)
            elif should_reuse:
                existing_summary = getattr(handle, "_summary", {}) or {}
                existing_summary.update(metadata)
                handle.set_summary(existing_summary)

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
            with self._lock:
                tracked = self._root_traces.pop(run_id, None)
                self._trace_mapping.pop(run_id, None)
                self._stream_buffers.pop(run_id, None)

            if not tracked:
                return

            handle = tracked.handle
            if outputs:
                handle.set_summary({"langchain_outputs": self._serialize_input(outputs)})
            if error:
                handle.mark_failed(str(error))
                from ..instrumentation.auto_trace_tracker import get_current_auto_trace_context

                auto_context = get_current_auto_trace_context()
                if auto_context:
                    auto_context.mark_span_failed()
            if tracked.owned and tracked.manager is not None:
                tracked.manager.__exit__(
                    type(error) if error else None,
                    error,
                    error.__traceback__ if error else None,
                )
            else:
                from ..instrumentation.auto_trace_tracker import (
                    _get_web_route_info,
                    close_auto_trace_if_needed,
                )

                is_web_context = _get_web_route_info() is not None
                close_auto_trace_if_needed(force_close=is_web_context, error=error)

        def _extract_model_id(
            self, serialized: dict[str, Any], kwargs: dict[str, Any]
        ) -> str | None:
            """Extract model ID from various possible locations."""
            invocation_params = kwargs.get("invocation_params", {})
            serialized_kwargs = serialized.get("kwargs", {})
            serialized_id = serialized.get("id", [])

            model_name: Any = (
                invocation_params.get("model_name")
                or invocation_params.get("model")
                or serialized_kwargs.get("model_name")
                or serialized_kwargs.get("model")
                or (
                    serialized_id[-1] if isinstance(serialized_id, list) and serialized_id else None
                )
            )
            if model_name is None:
                return None
            return str(model_name)

        def _start_span(
            self,
            *,
            lc_run_id: str,
            owner_run_id: str,
            kind: str,
            name: str | None,
            input_payload: Any,
            model_id: str | None = None,
            metadata: dict[str, Any] | None = None,
        ) -> None:
            with self._lock:
                if lc_run_id in self._active_spans:
                    return

                tracked = self._root_traces.get(owner_run_id)

                if tracked is None:
                    return

                handle = tracked.handle

                span_metadata = {"source": "langchain"}
                if metadata:
                    span_metadata.update(metadata)

                span_kwargs: dict[str, Any] = {
                    "span_type": kind,
                    "name": name,
                    "metadata": span_metadata,
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
                self._stream_buffers.pop(lc_run_id, None)

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
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            **kwargs: Any,
        ) -> None:
            run_id_str = str(run_id)
            parent_run_id_str = str(parent_run_id) if parent_run_id is not None else None
            if parent_run_id_str is None:
                self._create_trace(run_id_str, serialized or {}, inputs)
                return

            if not serialized:
                return

            owner = self._trace_mapping.get(parent_run_id_str, parent_run_id_str)
            self._trace_mapping[run_id_str] = owner

            node_id = serialized.get("id", "")
            if isinstance(node_id, str) and node_id.startswith("langchain.chat_models"):
                return

            chain_name = serialized.get("name") or serialized.get("id", ["unknown"])
            if isinstance(chain_name, list):
                chain_name = chain_name[-1] if chain_name else "unknown"

            self._start_span(
                lc_run_id=run_id_str,
                owner_run_id=owner,
                kind="chain",
                name=chain_name,
                input_payload=inputs,
                metadata={"chain_type": str(node_id)},
            )

        async def on_chain_start_async(
            self,
            serialized: dict[str, Any],
            inputs: dict[str, Any],
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            **kwargs: Any,
        ) -> None:
            self.on_chain_start(
                serialized, inputs, run_id=run_id, parent_run_id=parent_run_id, **kwargs
            )

        def on_chain_end(
            self,
            outputs: dict[str, Any],
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            **kwargs: Any,
        ) -> None:
            run_id_str = str(run_id)
            parent_run_id_str = str(parent_run_id) if parent_run_id is not None else None
            if parent_run_id_str is None:
                self._close_run(run_id_str, outputs=outputs)
            else:
                owner = self._trace_mapping.get(parent_run_id_str, parent_run_id_str)
                with self._lock:
                    if run_id_str in self._active_spans:
                        self._finish_span(
                            lc_run_id=run_id_str,
                            owner_run_id=owner,
                            output_payload=self._serialize_input(outputs),
                        )
                with self._lock:
                    self._trace_mapping.pop(run_id_str, None)

        async def on_chain_end_async(
            self,
            outputs: dict[str, Any],
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            **kwargs: Any,
        ) -> None:
            self.on_chain_end(outputs, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

        def on_chain_error(
            self,
            error: BaseException,
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            **kwargs: Any,
        ) -> None:
            run_id_str = str(run_id)
            parent_run_id_str = str(parent_run_id) if parent_run_id is not None else None
            if parent_run_id_str is None:
                self._close_run(run_id_str, error=error)
            else:
                owner = self._trace_mapping.get(parent_run_id_str, parent_run_id_str)
                with self._lock:
                    if run_id_str in self._active_spans:
                        self._finish_span(lc_run_id=run_id_str, owner_run_id=owner, error=error)
                with self._lock:
                    self._trace_mapping.pop(run_id_str, None)

        async def on_chain_error_async(
            self,
            error: BaseException,
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            **kwargs: Any,
        ) -> None:
            self.on_chain_error(error, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

        def _serialize_input(self, obj: Any) -> Any:
            """Enhanced serialization with better handling of complex types."""
            if obj is None:
                return None
            if isinstance(obj, str | int | float | bool):
                return obj
            if isinstance(obj, dict):
                return {k: self._serialize_input(v) for k, v in obj.items()}
            if isinstance(obj, list | tuple | set):
                return [self._serialize_input(item) for item in obj]

            try:
                from langchain_core.messages import BaseMessage

                if isinstance(obj, BaseMessage):
                    message_dict: dict[str, Any] = {
                        "type": obj.__class__.__name__,
                    }

                    if hasattr(obj, "content"):
                        content = obj.content
                        if isinstance(content, (list | dict)):
                            message_dict["content"] = self._serialize_input(content)
                        elif content is not None:
                            message_dict["content"] = str(content)
                        else:
                            message_dict["content"] = None
                    else:
                        message_dict["content"] = None

                    additional_fields: dict[str, Any] = {}
                    if hasattr(obj, "id") and obj.id:
                        additional_fields["id"] = str(obj.id)
                    if hasattr(obj, "name") and obj.name:
                        additional_fields["name"] = str(obj.name)
                    if hasattr(obj, "tool_calls") and obj.tool_calls:
                        additional_fields["tool_calls"] = self._serialize_input(obj.tool_calls)
                    if hasattr(obj, "tool_call_id") and obj.tool_call_id:
                        additional_fields["tool_call_id"] = str(obj.tool_call_id)
                    if hasattr(obj, "response_metadata") and obj.response_metadata:
                        try:
                            additional_fields["response_metadata"] = self._serialize_input(
                                obj.response_metadata
                            )
                        except Exception:
                            pass
                    if hasattr(obj, "additional_kwargs") and obj.additional_kwargs:
                        try:
                            additional_fields["additional_kwargs"] = self._serialize_input(
                                obj.additional_kwargs
                            )
                        except Exception:
                            pass

                    message_dict.update(additional_fields)
                    return message_dict
            except (ImportError, AttributeError, TypeError):
                pass

            try:
                from langchain_core.documents import Document

                if isinstance(obj, Document):
                    doc_dict: dict[str, Any] = {
                        "type": "Document",
                        "page_content": obj.page_content,
                    }
                    if obj.metadata:
                        doc_dict["metadata"] = self._serialize_input(obj.metadata)
                    return doc_dict
            except (ImportError, AttributeError, TypeError):
                pass

            try:
                from langchain_core.agents import AgentAction

                if isinstance(obj, AgentAction):
                    return {
                        "type": "AgentAction",
                        "tool": obj.tool,
                        "tool_input": self._serialize_input(obj.tool_input),
                        "log": obj.log,
                    }
            except (ImportError, AttributeError, TypeError):
                pass

            try:
                from langchain_core.agents import AgentFinish

                if isinstance(obj, AgentFinish):
                    return {
                        "type": "AgentFinish",
                        "return_values": self._serialize_input(obj.return_values),
                        "log": obj.log,
                    }
            except (ImportError, AttributeError, TypeError):
                pass

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
                            choice_message_dict: dict[str, Any] = {
                                "role": getattr(message, "role", "assistant"),
                                "content": getattr(message, "content", ""),
                            }
                            choice_data["message"] = choice_message_dict
                            if hasattr(message, "tool_calls") and message.tool_calls:
                                choice_message_dict["tool_calls"] = [
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

        def on_llm_start(
            self,
            serialized: dict[str, Any],
            prompts: list[str],
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            **kwargs: Any,
        ) -> None:
            """Handle traditional completion LLM calls."""
            run_id_str = str(run_id)
            parent_run_id_str = str(parent_run_id) if parent_run_id is not None else None
            add_langchain_active_run(run_id_str)

            owner = self._trace_mapping.get(
                parent_run_id_str or run_id_str, parent_run_id_str or run_id_str
            )

            if not parent_run_id_str:
                with self._lock:
                    if owner not in self._root_traces:
                        self._create_trace(
                            run_id=owner, serialized=serialized or {}, inputs={"prompts": prompts}
                        )

            model_id = self._extract_model_id(serialized, kwargs)

            self._start_span(
                lc_run_id=run_id_str,
                owner_run_id=owner,
                kind="llm",
                name=serialized.get("name", "llm"),
                input_payload={"prompts": prompts},
                model_id=model_id,
            )

        async def on_llm_start_async(
            self,
            serialized: dict[str, Any],
            prompts: list[str],
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            **kwargs: Any,
        ) -> None:
            self.on_llm_start(
                serialized, prompts, run_id=run_id, parent_run_id=parent_run_id, **kwargs
            )

        def on_chat_model_start(
            self,
            serialized: dict[str, Any],
            messages: list[list[Any]],
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            **kwargs: Any,
        ) -> None:
            run_id_str = str(run_id)
            parent_run_id_str = str(parent_run_id) if parent_run_id is not None else None
            add_langchain_active_run(run_id_str)

            owner = self._trace_mapping.get(
                parent_run_id_str or run_id_str, parent_run_id_str or run_id_str
            )
            if not parent_run_id_str:
                with self._lock:
                    if owner not in self._root_traces:
                        self._create_trace(
                            run_id=owner, serialized=serialized or {}, inputs={"messages": messages}
                        )

            model_id = self._extract_model_id(serialized, kwargs)
            serialized_messages = self._serialize_input(messages)

            self._start_span(
                lc_run_id=run_id_str,
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
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            **kwargs: Any,
        ) -> None:
            self.on_chat_model_start(
                serialized, messages, run_id=run_id, parent_run_id=parent_run_id, **kwargs
            )

        def on_llm_new_token(
            self,
            token: str,
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            **kwargs: Any,
        ) -> None:
            """Handle streaming tokens from LLM."""
            run_id_str = str(run_id)
            with self._lock:
                if run_id_str not in self._stream_buffers:
                    self._stream_buffers[run_id_str] = _StreamBuffer(chunks=[])
                self._stream_buffers[run_id_str].add_chunk(token)

        async def on_llm_new_token_async(
            self,
            token: str,
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            **kwargs: Any,
        ) -> None:
            self.on_llm_new_token(token, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

        def on_chat_model_end(
            self, response: Any, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
        ) -> None:
            run_id_str = str(run_id)
            try:
                self.on_llm_end(response, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
            finally:
                remove_langchain_active_run(run_id_str)

        async def on_chat_model_end_async(
            self, response: Any, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
        ) -> None:
            self.on_chat_model_end(response, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

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
            self, response: Any, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
        ) -> None:
            run_id_str = str(run_id)
            parent_run_id_str = str(parent_run_id) if parent_run_id is not None else None
            metadata = self._extract_token_usage(response, **kwargs)

            with self._lock:
                stream_buffer = self._stream_buffers.get(run_id_str)

            if stream_buffer:
                payload = stream_buffer.get_accumulated()
            else:
                payload = self._serialize_input(response)

            def _set_token_usage_if_present(span_handle: Any) -> None:
                if metadata:
                    token_kwargs = {
                        k: metadata[k] for k in ("input_tokens", "output_tokens") if k in metadata
                    }
                    if token_kwargs:
                        span_handle.set_token_usage(**token_kwargs)

            if parent_run_id_str:
                owner = self._trace_mapping.get(parent_run_id_str, parent_run_id_str)
                with self._lock:
                    entry = self._active_spans.get(run_id_str)
                    if entry:
                        _, span_handle = entry
                        _set_token_usage_if_present(span_handle)
                self._finish_span(lc_run_id=run_id_str, owner_run_id=owner, output_payload=payload)
                with self._lock:
                    self._trace_mapping.pop(run_id_str, None)
            else:
                with self._lock:
                    span_entry = self._active_spans.get(run_id_str)
                    tracked = self._root_traces.get(run_id_str)
                    if span_entry:
                        _, span_handle = span_entry
                        _set_token_usage_if_present(span_handle)
                        self._finish_span(
                            lc_run_id=run_id_str, owner_run_id=run_id_str, output_payload=payload
                        )
                    if tracked:
                        summary: dict[str, Any] = {"output": payload}
                        if metadata:
                            summary["token_usage"] = metadata
                        tracked.handle.set_summary(summary)
                self._close_run(run_id_str, outputs={"output": payload})
                with self._lock:
                    self._trace_mapping.pop(run_id_str, None)
                remove_langchain_active_run(run_id_str)

        async def on_llm_end_async(
            self, response: Any, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
        ) -> None:
            self.on_llm_end(response, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

        def on_llm_error(
            self,
            error: BaseException,
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            **kwargs: Any,
        ) -> None:
            run_id_str = str(run_id)
            parent_run_id_str = str(parent_run_id) if parent_run_id is not None else None
            owner = self._trace_mapping.get(
                parent_run_id_str or run_id_str, parent_run_id_str or run_id_str
            )
            self._finish_span(lc_run_id=run_id_str, owner_run_id=owner, error=error)
            with self._lock:
                self._trace_mapping.pop(run_id_str, None)
            remove_langchain_active_run(run_id_str)

        async def on_llm_error_async(
            self,
            error: BaseException,
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            **kwargs: Any,
        ) -> None:
            self.on_llm_error(error, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

        def on_tool_start(
            self,
            serialized: dict[str, Any],
            input_str: str,
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            **kwargs: Any,
        ) -> None:
            run_id_str = str(run_id)
            parent_run_id_str = str(parent_run_id) if parent_run_id is not None else None
            owner = self._trace_mapping.get(
                parent_run_id_str or run_id_str, parent_run_id_str or run_id_str
            )
            self._start_span(
                lc_run_id=run_id_str,
                owner_run_id=owner,
                kind="tool",
                name=serialized.get("name"),
                input_payload={"input": input_str},
            )

        async def on_tool_start_async(
            self,
            serialized: dict[str, Any],
            input_str: str,
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            **kwargs: Any,
        ) -> None:
            self.on_tool_start(
                serialized, input_str, run_id=run_id, parent_run_id=parent_run_id, **kwargs
            )

        def on_tool_end(
            self, output: Any, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
        ) -> None:
            run_id_str = str(run_id)
            parent_run_id_str = str(parent_run_id) if parent_run_id is not None else None
            owner = self._trace_mapping.get(
                parent_run_id_str or run_id_str, parent_run_id_str or run_id_str
            )
            self._finish_span(
                lc_run_id=run_id_str,
                owner_run_id=owner,
                output_payload=self._serialize_input(output),
            )
            with self._lock:
                self._trace_mapping.pop(run_id_str, None)

        async def on_tool_end_async(
            self, output: Any, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
        ) -> None:
            self.on_tool_end(output, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

        def on_tool_error(
            self,
            error: BaseException,
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            **kwargs: Any,
        ) -> None:
            run_id_str = str(run_id)
            parent_run_id_str = str(parent_run_id) if parent_run_id is not None else None
            owner = self._trace_mapping.get(
                parent_run_id_str or run_id_str, parent_run_id_str or run_id_str
            )
            self._finish_span(lc_run_id=run_id_str, owner_run_id=owner, error=error)
            with self._lock:
                self._trace_mapping.pop(run_id_str, None)

        async def on_tool_error_async(
            self,
            error: BaseException,
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            **kwargs: Any,
        ) -> None:
            self.on_tool_error(error, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

        def on_agent_action(
            self, action: Any, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
        ) -> None:
            """Handle agent action decisions."""
            run_id_str = str(run_id)
            parent_run_id_str = str(parent_run_id) if parent_run_id is not None else None
            owner = self._trace_mapping.get(
                parent_run_id_str or run_id_str, parent_run_id_str or run_id_str
            )

            tool_name = getattr(action, "tool", "unknown_tool")

            self._start_span(
                lc_run_id=run_id_str,
                owner_run_id=owner,
                kind="agent_action",
                name=f"agent_action_{tool_name}",
                input_payload=self._serialize_input(action),
                metadata={"tool": tool_name},
            )

        async def on_agent_action_async(
            self, action: Any, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
        ) -> None:
            self.on_agent_action(action, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

        def on_agent_finish(
            self, finish: Any, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
        ) -> None:
            """Handle agent finishing."""
            run_id_str = str(run_id)
            parent_run_id_str = str(parent_run_id) if parent_run_id is not None else None
            with self._lock:
                if run_id_str in self._active_spans:
                    owner = self._trace_mapping.get(
                        parent_run_id_str or run_id_str, parent_run_id_str or run_id_str
                    )
                    self._finish_span(
                        lc_run_id=run_id_str,
                        owner_run_id=owner,
                        output_payload=self._serialize_input(finish),
                    )

        async def on_agent_finish_async(
            self, finish: Any, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
        ) -> None:
            self.on_agent_finish(finish, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

        def on_retriever_start(
            self,
            serialized: dict[str, Any],
            query: str,
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            **kwargs: Any,
        ) -> None:
            """Handle retriever start for RAG workflows."""
            run_id_str = str(run_id)
            parent_run_id_str = str(parent_run_id) if parent_run_id is not None else None
            owner = self._trace_mapping.get(
                parent_run_id_str or run_id_str, parent_run_id_str or run_id_str
            )
            self._start_span(
                lc_run_id=run_id_str,
                owner_run_id=owner,
                kind="retriever",
                name=serialized.get("name", "retriever"),
                input_payload={"query": query},
                metadata={"retriever_type": serialized.get("id", ["unknown"])},
            )

        async def on_retriever_start_async(
            self,
            serialized: dict[str, Any],
            query: str,
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            **kwargs: Any,
        ) -> None:
            self.on_retriever_start(
                serialized, query, run_id=run_id, parent_run_id=parent_run_id, **kwargs
            )

        def on_retriever_end(
            self,
            documents: Sequence[Document],
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            **kwargs: Any,
        ) -> None:
            """Handle retriever end."""
            run_id_str = str(run_id)
            parent_run_id_str = str(parent_run_id) if parent_run_id is not None else None
            owner = self._trace_mapping.get(
                parent_run_id_str or run_id_str, parent_run_id_str or run_id_str
            )

            output_payload = {
                "documents": self._serialize_input(documents),
                "num_documents": len(documents),
            }

            self._finish_span(
                lc_run_id=run_id_str, owner_run_id=owner, output_payload=output_payload
            )
            with self._lock:
                self._trace_mapping.pop(run_id_str, None)

        async def on_retriever_end_async(
            self,
            documents: list[Any],
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            **kwargs: Any,
        ) -> None:
            self.on_retriever_end(documents, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

        def on_retriever_error(
            self,
            error: BaseException,
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            **kwargs: Any,
        ) -> None:
            """Handle retriever error."""
            run_id_str = str(run_id)
            parent_run_id_str = str(parent_run_id) if parent_run_id is not None else None
            owner = self._trace_mapping.get(
                parent_run_id_str or run_id_str, parent_run_id_str or run_id_str
            )
            self._finish_span(lc_run_id=run_id_str, owner_run_id=owner, error=error)
            with self._lock:
                self._trace_mapping.pop(run_id_str, None)

        async def on_retriever_error_async(
            self,
            error: BaseException,
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            **kwargs: Any,
        ) -> None:
            self.on_retriever_error(error, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

        def on_retry(
            self,
            retry_state: Any,
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            **kwargs: Any,
        ) -> None:
            """Handle retry events."""
            run_id_str = str(run_id)
            with self._lock:
                entry = self._active_spans.get(run_id_str)
                if entry:
                    _, span_handle = entry
                    if hasattr(span_handle, "set_metadata"):
                        retry_info = {
                            "retry_count": getattr(retry_state, "attempt_number", 0),
                            "retry_reason": str(getattr(retry_state, "outcome", "unknown")),
                        }
                        span_handle.set_metadata({"retry": retry_info})

        async def on_retry_async(
            self,
            retry_state: Any,
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            **kwargs: Any,
        ) -> None:
            self.on_retry(retry_state, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

        def on_text(
            self,
            text: str,
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            **kwargs: Any,
        ) -> None:
            """Handle intermediate text output (often used in verbose chains)."""
            run_id_str = str(run_id)
            with self._lock:
                if run_id_str in self._stream_buffers:
                    self._stream_buffers[run_id_str].add_chunk(text)

        async def on_text_async(
            self,
            text: str,
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            **kwargs: Any,
        ) -> None:
            self.on_text(text, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

else:
    TraciumLangChainHandler = None  # type: ignore[misc]


def register_langchain_handler(client: TraciumClient) -> None:
    """
    Register the comprehensive LangChain handler with automatic callback injection.
    Supports all LangChain patterns including streaming, agents, retrievers, and custom chains.
    """
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
