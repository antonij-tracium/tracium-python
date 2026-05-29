"""
Tests for Anthropic Files API parsing.

Files endpoints are *storage*, not LLM calls, so:
  * ``provider`` is still ``"anthropic"`` (so per-tenant tagging stays consistent)
  * ``operation`` is a ``file.*`` subtype, so the dashboard can segment them out
  * No ``model``/tokens are populated (the backend will treat them as
    ``span_type="custom"`` downstream via :func:`emit_llm_span`'s fallback)
"""

from __future__ import annotations

import json

from tracium.integrations.http_capture.providers import parse


def _b(payload: dict) -> bytes:
    return json.dumps(payload).encode("utf-8")


URL_COLLECTION = "https://api.anthropic.com/v1/files"
URL_ITEM = "https://api.anthropic.com/v1/files/file_abc123"
URL_CONTENT = "https://api.anthropic.com/v1/files/file_abc123/content"


class TestUpload:
    def test_upload_metadata(self) -> None:
        # Real upload bodies are multipart — we only see the JSON response.
        resp = {
            "id": "file_abc123",
            "type": "file",
            "filename": "report.pdf",
            "mime_type": "application/pdf",
            "size_bytes": 12345,
            "created_at": "2026-05-11T00:00:00Z",
        }
        call = parse(URL_COLLECTION, b"", _b(resp), 200)
        assert call is not None
        assert call.provider == "anthropic"
        assert call.operation == "file.upload"
        assert call.model is None
        assert call.output["file_id"] == "file_abc123"
        assert call.output["filename"] == "report.pdf"
        assert call.output["size_bytes"] == 12345


class TestList:
    def test_list_with_files(self) -> None:
        resp = {
            "data": [
                {"id": "file_1", "filename": "a.pdf"},
                {"id": "file_2", "filename": "b.txt"},
            ],
            "has_more": False,
        }
        call = parse(URL_COLLECTION, b"", _b(resp), 200)
        assert call.operation == "file.list"
        assert call.output == {"file_count": 2}
        assert call.extra["has_more"] is False

    def test_list_empty(self) -> None:
        resp = {"data": [], "has_more": False}
        call = parse(URL_COLLECTION, b"", _b(resp), 200)
        assert call.operation == "file.list"
        assert call.output == {"file_count": 0}


class TestGetMetadata:
    def test_get_returns_file_metadata(self) -> None:
        resp = {
            "id": "file_abc123",
            "type": "file",
            "filename": "report.pdf",
            "mime_type": "application/pdf",
            "size_bytes": 12345,
        }
        call = parse(URL_ITEM, b"", _b(resp), 200)
        assert call.operation == "file.metadata"
        assert call.output["file_id"] == "file_abc123"


class TestDelete:
    def test_delete_response(self) -> None:
        resp = {"id": "file_abc123", "type": "file_deleted", "deleted": True}
        call = parse(URL_ITEM, b"", _b(resp), 200)
        assert call.operation == "file.delete"
        assert call.output == {"deleted": True, "file_id": "file_abc123"}

    def test_delete_infers_id_from_path_if_missing(self) -> None:
        resp = {"deleted": True}
        call = parse(URL_ITEM, b"", _b(resp), 200)
        assert call.operation == "file.delete"
        assert call.output == {"deleted": True, "file_id": "file_abc123"}


class TestDownload:
    def test_binary_body_captures_size(self) -> None:
        raw = b"%PDF-1.4\n... binary pdf bytes ..."
        call = parse(URL_CONTENT, b"", raw, 200)
        assert call.operation == "file.download"
        assert call.output == {"file_id": "file_abc123", "bytes": len(raw)}

    def test_empty_body_still_records_id(self) -> None:
        call = parse(URL_CONTENT, b"", b"", 200)
        assert call.operation == "file.download"
        assert call.output["file_id"] == "file_abc123"


class TestErrors:
    def test_4xx_recorded_as_error(self) -> None:
        resp = {"error": {"type": "not_found_error", "message": "file not found"}}
        call = parse(URL_ITEM, b"", _b(resp), 404)
        assert call.operation in {"file.metadata", "file.delete", "file.upload"}
        assert call.error is not None
        assert "404" in call.error


class TestSpanTypeFallback:
    """File operations have no model_id → emit_llm_span should fall back to
    ``span_type="custom"`` instead of ``"llm"`` (the backend rejects the
    latter without a model)."""

    def test_emit_uses_custom_span_type(self, tracium_client, monkeypatch) -> None:
        from datetime import datetime, timezone

        from tracium.integrations.http_capture.emit import emit_llm_span
        from tracium.integrations.http_capture.providers import LLMCall

        recorded: list[dict] = []
        original = tracium_client.record_agent_spans

        def capture(trace_id, payloads):
            recorded.extend(payloads)
            return original(trace_id, payloads)

        monkeypatch.setattr(tracium_client, "record_agent_spans", capture)

        with tracium_client.agent_trace(agent_name="files-test"):
            emit_llm_span(
                LLMCall(
                    provider="anthropic",
                    operation="file.upload",
                    model=None,
                    output={"file_id": "file_xxx", "size_bytes": 100},
                ),
                started_at=datetime.now(timezone.utc),
            )

        assert recorded
        assert all(p.get("span_type") == "custom" for p in recorded)
