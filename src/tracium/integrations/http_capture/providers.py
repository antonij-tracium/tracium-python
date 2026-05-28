"""
Provider registry and body parsers for HTTP-level LLM capture.

A provider is identified by hostname + optional path prefix. Each provider
contributes a parse function that turns raw request/response payloads into a
normalized :class:`LLMCall` — the shape every span ultimately records (model,
messages, tokens, tools, error).

Adding a new provider is one entry in :data:`_RULES` plus one parser. The
generic OpenAI-compatible parser already covers most "OpenAI-API-compatible"
endpoints (Groq, Together, Fireworks, Perplexity, Anyscale, vLLM, Ollama,
self-hosted), so most additions are zero-code.

Coverage today:

OpenAI: chat/completions, completions, embeddings, images, audio, moderations,
**Responses API** (``/v1/responses``), **Assistants v2** (threads, runs, steps,
submit_tool_outputs), **Batch API**.

Anthropic: messages, **batches**, **count_tokens**.

Google: ``generateContent``, ``streamGenerateContent``, ``embedContent``,
``batchEmbedContents``, ``countTokens``. Vertex AI variants (same shapes,
different host).

AWS Bedrock: ``invoke``, ``invoke-with-response-stream``, **``converse``**,
**``converse-stream``**, with per-model native unwrapping for Anthropic, Llama,
Titan, Mistral, Cohere on the legacy Invoke path.

Cohere, plus OpenAI-compatible families (Groq, Together, Mistral, Perplexity,
Fireworks, DeepSeek, Anyscale, OpenRouter, Azure OpenAI, self-hosted).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

# --------------------------------------------------------------------------- #
# Normalized call shape                                                        #
# --------------------------------------------------------------------------- #


@dataclass
class LLMCall:
    """Normalized representation of a single LLM HTTP exchange.

    ``cached_input_tokens`` is the cache-**read** bucket (discount rate).
    ``cache_creation_input_tokens`` is the cache-**write** bucket — only
    populated by providers that distinguish them (Anthropic today).
    """

    provider: str
    model: str | None = None
    operation: str = "chat"
    # operation: chat | completion | embedding | image | audio | moderation |
    #            responses | assistant.run | assistant.step | assistant.tool_submit
    #            thread | thread.message | batch | batch_results |
    #            count_tokens | unknown
    input: Any = None
    output: Any = None
    tools: list[dict[str, Any]] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _safe_json_loads(blob: bytes | str | None) -> Any:
    if blob is None:
        return None
    if isinstance(blob, bytes):
        try:
            blob = blob.decode("utf-8", errors="replace")
        except Exception:
            return None
    try:
        return json.loads(blob)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# OpenAI parsers                                                               #
# --------------------------------------------------------------------------- #


def parse_openai(
    url: str,
    request_body: Any,
    response_body: Any,
    status_code: int,
) -> LLMCall:
    """Dispatch by URL path to the right OpenAI parser."""
    path = (urlparse(url).path or "").lower()

    if "/v1/responses" in path:
        return _parse_openai_responses(url, request_body, response_body, status_code)
    if "/v1/threads" in path:
        return _parse_openai_assistants(url, request_body, response_body, status_code)
    if "/v1/batches" in path:
        return _parse_openai_batches(url, request_body, response_body, status_code)
    return parse_openai_compatible(url, request_body, response_body, status_code)


def parse_openai_compatible(
    url: str,
    request_body: Any,
    response_body: Any,
    status_code: int,
    *,
    provider: str = "openai",
) -> LLMCall:
    """Parse OpenAI's REST API surface — dispatched by operation.

    The OpenAI API family uses different response shapes for chat, embeddings,
    images, audio (transcription / TTS), and moderation. Each is extracted by a
    dedicated branch so the captured ``output`` reflects what the user actually
    got back, not an embedding-shaped guess.
    """
    operation = _detect_openai_operation(url)
    call = LLMCall(provider=provider, operation=operation)

    req = request_body if isinstance(request_body, dict) else {}
    call.model = req.get("model")
    call.input = req.get("messages") or req.get("input") or req.get("prompt")
    if isinstance(req.get("tools"), list):
        call.tools = req["tools"]

    if status_code >= 400:
        call.error = _format_openai_error(response_body, status_code)
        return call

    resp = response_body if isinstance(response_body, dict) else {}
    _populate_usage(call, resp.get("usage"))

    if operation == "embedding":
        _extract_embedding_output(call, resp)
    elif operation == "image":
        _extract_image_output(call, resp)
    elif operation == "audio":
        _extract_audio_output(call, resp, req)
    elif operation == "moderation":
        _extract_moderation_output(call, resp)
    else:
        _extract_chat_output(call, resp)

    return call


def _extract_chat_output(call: LLMCall, resp: dict[str, Any]) -> None:
    choices = resp.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0] or {}
        msg = first.get("message") or first.get("delta") or {}
        call.output = msg.get("content") or first.get("text")
        if isinstance(msg.get("tool_calls"), list):
            call.tool_calls = msg["tool_calls"]
    elif resp.get("output_text") is not None:
        call.output = resp["output_text"]


def _extract_embedding_output(call: LLMCall, resp: dict[str, Any]) -> None:
    data = resp.get("data")
    if isinstance(data, list):
        # Report vector count + dimensions of the first vector for the dashboard.
        first = data[0] if data and isinstance(data[0], dict) else {}
        embedding = first.get("embedding")
        dims = len(embedding) if isinstance(embedding, list) else None
        call.output = {"vectors": len(data), "dimensions": dims}


def _extract_image_output(call: LLMCall, resp: dict[str, Any]) -> None:
    data = resp.get("data")
    if isinstance(data, list):
        urls: list[str] = []
        b64_count = 0
        for item in data:
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("url"), str):
                urls.append(item["url"])
            elif isinstance(item.get("b64_json"), str):
                b64_count += 1
        out: dict[str, Any] = {"images": len(data)}
        if urls:
            out["urls"] = urls
        if b64_count:
            out["b64_json_count"] = b64_count
        call.output = out


def _extract_audio_output(
    call: LLMCall, resp: dict[str, Any], req: dict[str, Any]
) -> None:
    # Whisper transcription / translation returns top-level ``text`` (and
    # optionally segments, language, duration).
    if isinstance(resp.get("text"), str):
        call.output = resp["text"]
        if isinstance(resp.get("language"), str):
            call.extra["language"] = resp["language"]
        if resp.get("duration") is not None:
            call.extra["duration_seconds"] = resp["duration"]
        return
    # TTS (/v1/audio/speech) returns binary audio bytes — JSON body is empty.
    # Surface the input text so the dashboard can show what was synthesized.
    if not resp and isinstance(req.get("input"), str):
        call.output = {"audio": "<binary>", "input_text": req["input"]}


def _extract_moderation_output(call: LLMCall, resp: dict[str, Any]) -> None:
    results = resp.get("results")
    if isinstance(results, list) and results:
        first = results[0] or {}
        flagged_categories = [
            name
            for name, flag in (first.get("categories") or {}).items()
            if flag
        ]
        call.output = {
            "flagged": bool(first.get("flagged")),
            "categories": flagged_categories,
            "result_count": len(results),
        }


def _parse_openai_responses(
    url: str, req_body: Any, resp_body: Any, status_code: int
) -> LLMCall:
    """Parse OpenAI's Responses API (``/v1/responses``)."""
    call = LLMCall(provider="openai", operation="responses")
    req = req_body if isinstance(req_body, dict) else {}
    call.model = req.get("model")
    call.input = req.get("input") or req.get("messages")
    if isinstance(req.get("tools"), list):
        call.tools = req["tools"]

    if status_code >= 400:
        call.error = _format_openai_error(resp_body, status_code)
        return call

    resp = resp_body if isinstance(resp_body, dict) else {}
    _populate_usage(call, resp.get("usage"))
    if call.model is None:
        call.model = resp.get("model")

    output = resp.get("output")
    if isinstance(output, list):
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            if kind == "message":
                for content in item.get("content", []) or []:
                    if isinstance(content, dict) and isinstance(content.get("text"), str):
                        text_parts.append(content["text"])
            elif kind in ("function_call", "tool_call"):
                tool_calls.append(item)
        if text_parts:
            call.output = "".join(text_parts)
        if tool_calls:
            call.tool_calls = tool_calls
    elif resp.get("output_text"):
        call.output = resp["output_text"]
    return call


def _parse_openai_assistants(
    url: str, req_body: Any, resp_body: Any, status_code: int
) -> LLMCall:
    """Parse OpenAI Assistants v2 (threads, runs, steps, tool submission)."""
    path = (urlparse(url).path or "").lower()
    operation = _detect_assistants_operation(path)
    call = LLMCall(provider="openai", operation=operation)

    req = req_body if isinstance(req_body, dict) else {}
    if isinstance(req.get("messages"), list):
        call.input = req["messages"]
    elif "additional_messages" in req or "tool_outputs" in req:
        call.input = req
    if isinstance(req.get("tools"), list):
        call.tools = req["tools"]

    if status_code >= 400:
        call.error = _format_openai_error(resp_body, status_code)
        return call

    resp = resp_body if isinstance(resp_body, dict) else {}
    call.model = resp.get("model") or req.get("model")
    _populate_usage(call, resp.get("usage"))

    # Run / run-step status — bubble up so dashboards can show progress.
    if isinstance(resp.get("status"), str):
        call.extra["assistant_status"] = resp["status"]

    # For run-step responses, surface tool calls + output text where present.
    step_details = resp.get("step_details") or {}
    if isinstance(step_details, dict):
        if step_details.get("type") == "tool_calls" and isinstance(
            step_details.get("tool_calls"), list
        ):
            call.tool_calls = step_details["tool_calls"]
        elif step_details.get("type") == "message_creation":
            mid = (step_details.get("message_creation") or {}).get("message_id")
            if mid:
                call.extra["message_id"] = mid

    return call


def _parse_openai_batches(
    url: str, req_body: Any, resp_body: Any, status_code: int
) -> LLMCall:
    """Parse OpenAI Batch API (``/v1/batches``).

    Batch jobs don't have a model on the request — the model lives inside each
    line of the input file. We capture the job-level metadata (status, counts,
    completion window) and emit under ``operation="batch"``.
    """
    call = LLMCall(provider="openai", operation="batch")
    req = req_body if isinstance(req_body, dict) else {}
    if isinstance(req, dict) and req:
        call.input = {
            "endpoint": req.get("endpoint"),
            "completion_window": req.get("completion_window"),
            "input_file_id": req.get("input_file_id"),
        }

    if status_code >= 400:
        call.error = _format_openai_error(resp_body, status_code)
        return call

    resp = resp_body if isinstance(resp_body, dict) else {}
    call.output = {
        "batch_id": resp.get("id"),
        "status": resp.get("status"),
        "request_counts": resp.get("request_counts"),
    }
    if resp.get("status"):
        call.extra["batch_status"] = resp["status"]
    return call


# --------------------------------------------------------------------------- #
# Anthropic parsers                                                            #
# --------------------------------------------------------------------------- #


def parse_anthropic(
    url: str, req_body: Any, resp_body: Any, status_code: int
) -> LLMCall:
    """Dispatch by URL path to the right Anthropic parser."""
    path = (urlparse(url).path or "").lower()
    if path.endswith("/count_tokens") or "/count_tokens" in path:
        return _parse_anthropic_count_tokens(url, req_body, resp_body, status_code)
    if "/messages/batches" in path:
        # GET /v1/messages/batches/{id}/results streams JSONL with per-request
        # results; treat it as a distinct operation since the shape differs.
        if path.endswith("/results") or "/results" in path:
            return _parse_anthropic_batch_results(url, req_body, resp_body, status_code)
        return _parse_anthropic_batches(url, req_body, resp_body, status_code)
    if "/files" in path:
        return _parse_anthropic_files(url, req_body, resp_body, status_code)
    return _parse_anthropic_messages(url, req_body, resp_body, status_code)


def _parse_anthropic_messages(
    url: str, req_body: Any, resp_body: Any, status_code: int
) -> LLMCall:
    call = LLMCall(provider="anthropic", operation="chat")
    req = req_body if isinstance(req_body, dict) else {}
    call.model = req.get("model")
    msgs = req.get("messages")
    system = req.get("system")
    call.input = {"system": system, "messages": msgs} if system else msgs
    if isinstance(req.get("tools"), list):
        call.tools = req["tools"]

    if status_code >= 400:
        call.error = _format_generic_error(resp_body, status_code)
        return call

    resp = resp_body if isinstance(resp_body, dict) else {}
    _populate_anthropic_usage(call, resp.get("usage"))

    content = resp.get("content")
    if isinstance(content, list):
        text_parts = [
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        tool_uses = [
            b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"
        ]
        if text_parts:
            call.output = "".join(text_parts)
        if tool_uses:
            call.tool_calls = tool_uses
    return call


def _parse_anthropic_count_tokens(
    url: str, req_body: Any, resp_body: Any, status_code: int
) -> LLMCall:
    call = LLMCall(provider="anthropic", operation="count_tokens")
    req = req_body if isinstance(req_body, dict) else {}
    call.model = req.get("model")
    call.input = req.get("messages")

    if status_code >= 400:
        call.error = _format_generic_error(resp_body, status_code)
        return call

    resp = resp_body if isinstance(resp_body, dict) else {}
    if isinstance(resp.get("input_tokens"), int):
        call.input_tokens = resp["input_tokens"]
        call.output = {"input_tokens": resp["input_tokens"]}
    return call


def _parse_anthropic_batches(
    url: str, req_body: Any, resp_body: Any, status_code: int
) -> LLMCall:
    call = LLMCall(provider="anthropic", operation="batch")
    req = req_body if isinstance(req_body, dict) else {}
    if isinstance(req.get("requests"), list):
        call.input = {"request_count": len(req["requests"])}

    if status_code >= 400:
        call.error = _format_generic_error(resp_body, status_code)
        return call

    resp = resp_body if isinstance(resp_body, dict) else {}
    call.output = {
        "batch_id": resp.get("id"),
        "processing_status": resp.get("processing_status"),
        "request_counts": resp.get("request_counts"),
    }
    if resp.get("processing_status"):
        call.extra["batch_status"] = resp["processing_status"]
    return call


def _parse_anthropic_files(
    url: str, req_body: Any, resp_body: Any, status_code: int
) -> LLMCall:
    """Parse Anthropic Files API (beta).

    Files endpoints are *storage* operations, not LLM calls — so we emit a
    span with ``span_type="custom"`` (no ``model_id``, no token costs) and
    record the action + metadata. This lets dashboards show file activity in
    the same timeline as LLM calls without polluting cost-per-model metrics.

    Endpoints covered (under ``/v1/files`` on ``api.anthropic.com``):

    * ``POST /v1/files``                — upload
    * ``GET  /v1/files``                — list
    * ``GET  /v1/files/{id}``           — get metadata
    * ``DELETE /v1/files/{id}``         — delete
    * ``GET  /v1/files/{id}/content``   — download
    """
    path = (urlparse(url).path or "").lower()
    operation, action_hint = _classify_anthropic_files_call(path, resp_body)
    call = LLMCall(provider="anthropic", operation=operation)

    if isinstance(req_body, dict):
        # Upload requests are multipart — the body is rarely a JSON dict on the
        # wire, but we capture it if the user happens to send one.
        call.input = req_body

    if status_code >= 400:
        call.error = _format_generic_error(resp_body, status_code)
        return call

    if action_hint == "list":
        resp = resp_body if isinstance(resp_body, dict) else {}
        data = resp.get("data") or []
        call.output = {"file_count": len(data) if isinstance(data, list) else 0}
        if isinstance(resp.get("has_more"), bool):
            call.extra["has_more"] = resp["has_more"]
        return call

    if action_hint == "delete":
        resp = resp_body if isinstance(resp_body, dict) else {}
        call.output = {
            "deleted": bool(resp.get("deleted", True)),
            "file_id": resp.get("id") or _id_from_files_path(path),
        }
        return call

    if action_hint == "download":
        # Body is raw file bytes — capture only the size.
        if isinstance(resp_body, bytes | bytearray):
            call.output = {
                "file_id": _id_from_files_path(path),
                "bytes": len(resp_body),
            }
        else:
            call.output = {"file_id": _id_from_files_path(path)}
        return call

    # Upload / get-metadata: response is the file's metadata object.
    resp = resp_body if isinstance(resp_body, dict) else {}
    call.output = {
        "file_id": resp.get("id"),
        "filename": resp.get("filename"),
        "mime_type": resp.get("mime_type"),
        "size_bytes": resp.get("size_bytes"),
        "type": resp.get("type"),
    }
    return call


def _classify_anthropic_files_call(
    path: str, resp_body: Any
) -> tuple[str, str]:
    """Infer the Files-API operation from URL path + response shape.

    Returns ``(operation_name, action_hint)``. The HTTP method isn't visible
    here, so we use shape heuristics: a ``data`` array means list, a
    ``deleted`` field means delete, a raw bytes body means download.
    """
    if path.endswith("/content"):
        return "file.download", "download"

    parts = [p for p in path.split("/") if p]
    is_collection = len(parts) == 2  # /v1/files

    if isinstance(resp_body, dict):
        if isinstance(resp_body.get("data"), list):
            return "file.list", "list"
        if "deleted" in resp_body:
            return "file.delete", "delete"
        if "id" in resp_body and ("filename" in resp_body or "size_bytes" in resp_body):
            return ("file.upload" if is_collection else "file.metadata"), "metadata"

    if isinstance(resp_body, bytes | bytearray):
        return "file.download", "download"

    # Unknown: fall back to a generic operation tag.
    return ("file.upload" if is_collection else "file.metadata"), "metadata"


def _id_from_files_path(path: str) -> str | None:
    """Extract ``file_xxx`` from ``/v1/files/file_xxx`` or .../content."""
    import re as _re

    match = _re.search(r"/files/([^/]+)", path)
    return match.group(1) if match else None


def _parse_anthropic_batch_results(
    url: str, req_body: Any, resp_body: Any, status_code: int
) -> LLMCall:
    """Parse the JSONL stream returned by ``GET /v1/messages/batches/{id}/results``.

    Each line is a per-request result: ``{"custom_id": ..., "result": {...}}``.
    We aggregate counts, sum tokens across successful results, and surface a
    summary so the user can see batch completion progress at a glance.
    """
    call = LLMCall(provider="anthropic", operation="batch_results")

    if status_code >= 400:
        call.error = _format_generic_error(resp_body, status_code)
        return call

    lines = _split_jsonl(resp_body)
    succeeded = 0
    errored = 0
    canceled = 0
    expired = 0
    total_input = 0
    total_output = 0
    total_cache_creation = 0
    total_cache_read = 0
    model_seen: str | None = None

    for entry in lines:
        if not isinstance(entry, dict):
            continue
        result = entry.get("result") or {}
        kind = result.get("type")
        if kind == "succeeded":
            succeeded += 1
            message = result.get("message") or {}
            usage = message.get("usage") or {}
            total_input += int(usage.get("input_tokens") or 0)
            total_output += int(usage.get("output_tokens") or 0)
            total_cache_creation += int(usage.get("cache_creation_input_tokens") or 0)
            total_cache_read += int(usage.get("cache_read_input_tokens") or 0)
            if model_seen is None and isinstance(message.get("model"), str):
                model_seen = message["model"]
        elif kind == "errored":
            errored += 1
        elif kind == "canceled":
            canceled += 1
        elif kind == "expired":
            expired += 1

    call.model = model_seen
    if total_input:
        call.input_tokens = total_input
    if total_output:
        call.output_tokens = total_output
    if total_cache_read:
        call.cached_input_tokens = total_cache_read
    if total_cache_creation:
        call.cache_creation_input_tokens = total_cache_creation

    call.output = {
        "succeeded": succeeded,
        "errored": errored,
        "canceled": canceled,
        "expired": expired,
        "total": succeeded + errored + canceled + expired,
    }
    return call


def _split_jsonl(body: Any) -> list[Any]:
    """Parse a JSONL body (newline-delimited JSON) into a list of objects."""
    if isinstance(body, list):
        return body
    if isinstance(body, dict):
        return [body]
    if isinstance(body, bytes | bytearray):
        try:
            body = body.decode("utf-8", errors="replace")
        except Exception:
            return []
    if not isinstance(body, str):
        return []
    out: list[Any] = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


# --------------------------------------------------------------------------- #
# Google Gemini / Vertex AI parsers                                            #
# --------------------------------------------------------------------------- #


def parse_google_gemini(
    url: str, req_body: Any, resp_body: Any, status_code: int
) -> LLMCall:
    """Dispatch by URL action (``:action`` suffix) to the right Gemini parser.

    Covers ``generateContent``, ``streamGenerateContent``, ``embedContent``,
    ``batchEmbedContents``, ``countTokens``. Same shapes work for Vertex AI.
    """
    path = (urlparse(url).path or "")
    action = _gemini_action(path) or "generateContent"

    if action in ("embedContent", "batchEmbedContents"):
        return _parse_gemini_embed(url, req_body, resp_body, status_code)
    if action == "countTokens":
        return _parse_gemini_count_tokens(url, req_body, resp_body, status_code)
    return _parse_gemini_generate(url, req_body, resp_body, status_code, action=action)


def _parse_gemini_generate(
    url: str, req_body: Any, resp_body: Any, status_code: int, *, action: str
) -> LLMCall:
    operation = "chat"
    call = LLMCall(provider="google", operation=operation)
    call.model = _model_from_gemini_url(url)

    req = req_body if isinstance(req_body, dict) else {}
    call.input = req.get("contents") or req.get("prompt")
    if isinstance(req.get("tools"), list):
        call.tools = req["tools"]

    if status_code >= 400:
        call.error = _format_generic_error(resp_body, status_code)
        return call

    resp = resp_body if isinstance(resp_body, dict) else {}
    _populate_gemini_usage(call, resp.get("usageMetadata"))

    candidates = resp.get("candidates")
    if isinstance(candidates, list) and candidates:
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            parts = (cand.get("content") or {}).get("parts")
            if isinstance(parts, list):
                for p in parts:
                    if not isinstance(p, dict):
                        continue
                    if isinstance(p.get("text"), str):
                        text_parts.append(p["text"])
                    if isinstance(p.get("functionCall"), dict):
                        tool_calls.append(p["functionCall"])
        if text_parts:
            call.output = "".join(text_parts)
        if tool_calls:
            call.tool_calls = tool_calls

    if action == "streamGenerateContent":
        call.extra["streaming"] = True
    return call


def _parse_gemini_embed(
    url: str, req_body: Any, resp_body: Any, status_code: int
) -> LLMCall:
    call = LLMCall(provider="google", operation="embedding")
    call.model = _model_from_gemini_url(url)
    req = req_body if isinstance(req_body, dict) else {}
    call.input = req.get("content") or req.get("requests")

    if status_code >= 400:
        call.error = _format_generic_error(resp_body, status_code)
        return call

    resp = resp_body if isinstance(resp_body, dict) else {}
    if isinstance(resp.get("embedding"), dict):
        call.output = {"vectors": 1}
    elif isinstance(resp.get("embeddings"), list):
        call.output = {"vectors": len(resp["embeddings"])}
    _populate_gemini_usage(call, resp.get("usageMetadata"))
    return call


def _parse_gemini_count_tokens(
    url: str, req_body: Any, resp_body: Any, status_code: int
) -> LLMCall:
    call = LLMCall(provider="google", operation="count_tokens")
    call.model = _model_from_gemini_url(url)
    req = req_body if isinstance(req_body, dict) else {}
    call.input = req.get("contents")

    if status_code >= 400:
        call.error = _format_generic_error(resp_body, status_code)
        return call

    resp = resp_body if isinstance(resp_body, dict) else {}
    total = resp.get("totalTokens")
    if isinstance(total, int):
        call.input_tokens = total
        call.output = {"totalTokens": total}
    return call


# --------------------------------------------------------------------------- #
# AWS Bedrock parsers                                                          #
# --------------------------------------------------------------------------- #


def parse_bedrock(
    url: str, req_body: Any, resp_body: Any, status_code: int
) -> LLMCall:
    """Dispatch by URL action to the right Bedrock parser.

    - ``/model/{id}/invoke`` and ``/invoke-with-response-stream`` use per-model
      native body shapes — we unwrap based on the model ID.
    - ``/model/{id}/converse`` and ``/converse-stream`` use Bedrock's unified
      Converse API which has a stable shape across models.
    """
    path = (urlparse(url).path or "")
    if "/converse" in path:
        return _parse_bedrock_converse(url, req_body, resp_body, status_code)
    return _parse_bedrock_invoke(url, req_body, resp_body, status_code)


def _parse_bedrock_converse(
    url: str, req_body: Any, resp_body: Any, status_code: int
) -> LLMCall:
    call = LLMCall(provider="bedrock", operation="chat")
    call.model = _model_from_bedrock_url(url)

    req = req_body if isinstance(req_body, dict) else {}
    msgs = req.get("messages")
    system = req.get("system")
    call.input = {"system": system, "messages": msgs} if system else msgs
    tool_cfg = req.get("toolConfig")
    if isinstance(tool_cfg, dict) and isinstance(tool_cfg.get("tools"), list):
        call.tools = tool_cfg["tools"]

    if status_code >= 400:
        call.error = _format_generic_error(resp_body, status_code)
        return call

    resp = resp_body if isinstance(resp_body, dict) else {}
    usage = resp.get("usage") or {}
    call.input_tokens = usage.get("inputTokens")
    call.output_tokens = usage.get("outputTokens")
    call.cached_input_tokens = usage.get("cacheReadInputTokens")

    output = (resp.get("output") or {}).get("message") or {}
    content = output.get("content")
    if isinstance(content, list):
        text_parts = [
            b.get("text", "") for b in content if isinstance(b, dict) and "text" in b
        ]
        tool_uses = [b.get("toolUse") for b in content if isinstance(b, dict) and "toolUse" in b]
        if text_parts:
            call.output = "".join(text_parts)
        if tool_uses:
            call.tool_calls = [t for t in tool_uses if t]
    return call


def _parse_bedrock_invoke(
    url: str, req_body: Any, resp_body: Any, status_code: int
) -> LLMCall:
    """Per-model unwrapping of the legacy InvokeModel API."""
    call = LLMCall(provider="bedrock", operation="chat")
    call.model = _model_from_bedrock_url(url)
    model = (call.model or "").lower()

    req = req_body if isinstance(req_body, dict) else {}

    if status_code >= 400:
        call.error = _format_generic_error(resp_body, status_code)
        # Still try to record input.
        call.input = req
        return call

    resp = resp_body if isinstance(resp_body, dict) else {}

    # Anthropic Claude over Bedrock
    if "anthropic" in model or "claude" in model:
        call.input = (
            {"system": req.get("system"), "messages": req.get("messages")}
            if req.get("system")
            else req.get("messages") or req.get("prompt")
        )
        if isinstance(req.get("tools"), list):
            call.tools = req["tools"]
        _populate_anthropic_usage(call, resp.get("usage"))
        content = resp.get("content")
        if isinstance(content, list):
            text_parts = [
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            if text_parts:
                call.output = "".join(text_parts)
        return call

    # Meta Llama
    if "llama" in model or "meta" in model:
        call.input = req.get("prompt") or req.get("messages")
        call.input_tokens = resp.get("prompt_token_count")
        call.output_tokens = resp.get("generation_token_count")
        call.output = resp.get("generation") or resp.get("output")
        return call

    # Amazon Titan
    if "titan" in model or "amazon." in model:
        call.input = req.get("inputText") or req.get("messages")
        call.input_tokens = resp.get("inputTextTokenCount")
        results = resp.get("results")
        if isinstance(results, list) and results:
            first = results[0] or {}
            call.output = first.get("outputText")
            call.output_tokens = first.get("tokenCount")
        return call

    # Mistral
    if "mistral" in model:
        call.input = req.get("prompt") or req.get("messages")
        outputs = resp.get("outputs")
        if isinstance(outputs, list) and outputs:
            first = outputs[0] or {}
            call.output = first.get("text")
        usage = resp.get("usage") or {}
        call.input_tokens = usage.get("prompt_tokens")
        call.output_tokens = usage.get("completion_tokens")
        return call

    # Cohere over Bedrock
    if "cohere" in model:
        call.input = req.get("prompt") or req.get("message")
        gens = resp.get("generations")
        if isinstance(gens, list) and gens:
            call.output = (gens[0] or {}).get("text")
        call.input_tokens = resp.get("prompt_tokens")
        call.output_tokens = resp.get("generation_tokens")
        return call

    # Unknown model — capture what we can.
    call.input = req
    call.output = resp.get("output") or resp.get("completion") or resp.get("results")
    return call


# --------------------------------------------------------------------------- #
# Cohere (direct) parser                                                       #
# --------------------------------------------------------------------------- #


def parse_cohere(url: str, req_body: Any, resp_body: Any, status_code: int) -> LLMCall:
    call = LLMCall(provider="cohere", operation="chat")
    req = req_body if isinstance(req_body, dict) else {}
    call.model = req.get("model")
    call.input = req.get("message") or req.get("messages") or req.get("texts")
    if isinstance(req.get("tools"), list):
        call.tools = req["tools"]

    if status_code >= 400:
        call.error = _format_generic_error(resp_body, status_code)
        return call

    resp = resp_body if isinstance(resp_body, dict) else {}
    meta = (resp.get("meta") or {}).get("billed_units") or {}
    call.input_tokens = meta.get("input_tokens")
    call.output_tokens = meta.get("output_tokens")
    call.output = resp.get("text") or resp.get("response")
    if isinstance(resp.get("tool_calls"), list):
        call.tool_calls = resp["tool_calls"]
    return call


# --------------------------------------------------------------------------- #
# Unknown / fallback parser                                                    #
# --------------------------------------------------------------------------- #


def parse_unknown(url: str, req_body: Any, resp_body: Any, status_code: int) -> LLMCall:
    """Last-resort parser for endpoints we recognize as LLM but don't know shape of."""
    call = LLMCall(provider="unknown", operation="unknown")
    if isinstance(req_body, dict):
        call.model = req_body.get("model") or req_body.get("model_name")
        call.input = (
            req_body.get("messages")
            or req_body.get("prompt")
            or req_body.get("input")
        )
    if status_code >= 400:
        call.error = _format_generic_error(resp_body, status_code)
    elif isinstance(resp_body, dict):
        call.output = resp_body.get("output") or resp_body.get("text")
    call.extra["url"] = url
    return call


# --------------------------------------------------------------------------- #
# Provider lookup                                                              #
# --------------------------------------------------------------------------- #


@dataclass
class _ProviderRule:
    name: str
    host_pattern: re.Pattern
    parser: Any


def _hp(pattern: str) -> re.Pattern:
    if pattern.startswith("*."):
        return re.compile(rf".*\.{re.escape(pattern[2:])}$")
    return re.compile(rf"^{re.escape(pattern)}$")


_RULES: list[_ProviderRule] = [
    _ProviderRule("openai", _hp("api.openai.com"), parse_openai),
    _ProviderRule("anthropic", _hp("api.anthropic.com"), parse_anthropic),
    _ProviderRule(
        "google",
        _hp("generativelanguage.googleapis.com"),
        parse_google_gemini,
    ),
    _ProviderRule(
        "google-vertex",
        re.compile(r".*-aiplatform\.googleapis\.com$|^aiplatform\.googleapis\.com$"),
        parse_google_gemini,
    ),
    _ProviderRule("cohere", _hp("api.cohere.ai"), parse_cohere),
    _ProviderRule("cohere", _hp("api.cohere.com"), parse_cohere),
    _ProviderRule(
        "bedrock",
        re.compile(r".*bedrock-runtime\..*\.amazonaws\.com$"),
        parse_bedrock,
    ),
    _ProviderRule(
        "groq",
        _hp("api.groq.com"),
        lambda u, r, s, c: parse_openai_compatible(u, r, s, c, provider="groq"),
    ),
    _ProviderRule(
        "together",
        _hp("api.together.xyz"),
        lambda u, r, s, c: parse_openai_compatible(u, r, s, c, provider="together"),
    ),
    _ProviderRule(
        "mistral",
        _hp("api.mistral.ai"),
        lambda u, r, s, c: parse_openai_compatible(u, r, s, c, provider="mistral"),
    ),
    _ProviderRule(
        "perplexity",
        _hp("api.perplexity.ai"),
        lambda u, r, s, c: parse_openai_compatible(u, r, s, c, provider="perplexity"),
    ),
    _ProviderRule(
        "fireworks",
        _hp("api.fireworks.ai"),
        lambda u, r, s, c: parse_openai_compatible(u, r, s, c, provider="fireworks"),
    ),
    _ProviderRule(
        "deepseek",
        _hp("api.deepseek.com"),
        lambda u, r, s, c: parse_openai_compatible(u, r, s, c, provider="deepseek"),
    ),
    _ProviderRule(
        "anyscale",
        _hp("api.endpoints.anyscale.com"),
        lambda u, r, s, c: parse_openai_compatible(u, r, s, c, provider="anyscale"),
    ),
    _ProviderRule(
        "openrouter",
        _hp("openrouter.ai"),
        lambda u, r, s, c: parse_openai_compatible(u, r, s, c, provider="openrouter"),
    ),
    _ProviderRule(
        "azure-openai",
        re.compile(r".*\.openai\.azure\.com$"),
        lambda u, r, s, c: parse_openai_compatible(u, r, s, c, provider="azure-openai"),
    ),
]

# Self-hosted OpenAI-compatible servers (Ollama, vLLM, LocalAI, …) — detected
# by URL path because the hostname is unknown.
_OPENAI_COMPAT_PATHS = re.compile(
    r"/v\d+/(chat/completions|completions|embeddings|images/generations|responses)$"
)


def detect_provider(url: str) -> tuple[str, Any] | None:
    """Return ``(provider_name, parser)`` for an LLM URL, or ``None``."""
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return None
    if not host:
        return None

    for rule in _RULES:
        if rule.host_pattern.match(host):
            return rule.name, rule.parser

    try:
        path = urlparse(url).path or ""
    except Exception:
        path = ""
    if _OPENAI_COMPAT_PATHS.search(path):
        return "openai-compatible", lambda u, r, s, c: parse_openai_compatible(
            u, r, s, c, provider=f"openai-compatible:{host}"
        )
    return None


def parse(
    url: str,
    request_body: bytes | str | dict[str, Any] | None,
    response_body: bytes | str | dict[str, Any] | None,
    status_code: int,
) -> LLMCall | None:
    """High-level entry point: detect provider and parse the call.

    Bodies are JSON-decoded best-effort. If decoding fails (e.g. the response
    is JSONL — Anthropic's batch-results endpoint streams newline-delimited
    JSON), the raw bytes/string are passed through so parsers that understand
    other framings can still handle them.
    """
    found = detect_provider(url)
    if found is None:
        return None
    _, parser = found
    req = _safe_json_loads(request_body) if isinstance(request_body, bytes | str) else request_body
    if req is None and request_body:
        req = request_body
    resp = (
        _safe_json_loads(response_body) if isinstance(response_body, bytes | str) else response_body
    )
    if resp is None and response_body:
        resp = response_body
    try:
        result: LLMCall | None = parser(url, req, resp, status_code)
        return result
    except Exception:
        return parse_unknown(url, req, resp, status_code)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _detect_openai_operation(url: str) -> str:
    path = (urlparse(url).path or "").lower()
    if "embedding" in path:
        return "embedding"
    if "image" in path:
        return "image"
    if "audio" in path or "speech" in path or "transcription" in path:
        return "audio"
    if "moderation" in path:
        return "moderation"
    return "chat"


def _detect_assistants_operation(path: str) -> str:
    if "/submit_tool_outputs" in path:
        return "assistant.tool_submit"
    if "/runs/" in path and "/steps" in path:
        return "assistant.step"
    if path.endswith("/steps") or "/steps/" in path:
        return "assistant.step"
    if "/runs" in path:
        return "assistant.run"
    if "/messages" in path:
        return "thread.message"
    return "thread"


def _gemini_action(path: str) -> str | None:
    """Extract the ``:action`` suffix from a Gemini URL path."""
    match = re.search(r":(\w+)$", path)
    return match.group(1) if match else None


def _model_from_gemini_url(url: str) -> str | None:
    match = re.search(r"/models/([^:/]+)", urlparse(url).path or "")
    return match.group(1) if match else None


def _model_from_bedrock_url(url: str) -> str | None:
    match = re.search(r"/model/([^/]+)/", urlparse(url).path or "")
    return match.group(1) if match else None


def _populate_usage(call: LLMCall, usage: Any) -> None:
    """OpenAI-style usage block."""
    if not isinstance(usage, dict):
        return
    call.input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
    call.output_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if cached is None:
        cached = (usage.get("input_tokens_details") or {}).get("cached_tokens")
    if cached is None:
        cached = usage.get("cached_tokens") or usage.get("cache_read_input_tokens")
    call.cached_input_tokens = cached


def _populate_anthropic_usage(call: LLMCall, usage: Any) -> None:
    """Populate Anthropic's three-bucket token model.

    Anthropic prompt caching emits two extra fields:
      * ``cache_read_input_tokens``  — billed at ~10% of base input rate
      * ``cache_creation_input_tokens`` — billed at 1.25× base input rate

    Both must be captured separately so the backend can apply the correct rate
    per bucket; conflating them undercounts the cache-create premium.
    """
    if not isinstance(usage, dict):
        return
    call.input_tokens = usage.get("input_tokens")
    call.output_tokens = usage.get("output_tokens")
    call.cached_input_tokens = usage.get("cache_read_input_tokens")
    call.cache_creation_input_tokens = usage.get("cache_creation_input_tokens")


def _populate_gemini_usage(call: LLMCall, usage: Any) -> None:
    if not isinstance(usage, dict):
        return
    call.input_tokens = usage.get("promptTokenCount")
    call.output_tokens = usage.get("candidatesTokenCount")
    call.cached_input_tokens = usage.get("cachedContentTokenCount")


def _format_openai_error(body: Any, status_code: int) -> str:
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        msg = body["error"].get("message") or str(body["error"])
        return f"{status_code}: {msg}"
    return _format_generic_error(body, status_code)


def _format_generic_error(body: Any, status_code: int) -> str:
    if isinstance(body, dict):
        msg = body.get("error") or body.get("message") or body.get("detail")
        if msg:
            return f"{status_code}: {msg}"
    if isinstance(body, str) and body:
        return f"{status_code}: {body[:500]}"
    return f"HTTP {status_code}"
