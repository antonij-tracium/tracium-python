"""
Tests for per-operation output extraction in the OpenAI parser.

Previously every non-chat response was force-classified as an embedding
``{"vectors": N}`` because of the ``data[]`` branch. Image, audio, and
moderation responses now have dedicated extraction so the dashboard shows the
actual output (URLs, transcript text, flagged categories).
"""

from __future__ import annotations

import json

from tracium.integrations.http_capture.providers import parse


def _b(payload: dict) -> bytes:
    return json.dumps(payload).encode("utf-8")


class TestEmbeddings:
    URL = "https://api.openai.com/v1/embeddings"

    def test_dimensions_captured(self) -> None:
        req = {"model": "text-embedding-3-small", "input": "hi"}
        resp = {
            "data": [{"embedding": [0.1, 0.2, 0.3, 0.4]}, {"embedding": [0.5, 0.6, 0.7, 0.8]}],
            "usage": {"prompt_tokens": 4, "total_tokens": 4},
        }
        call = parse(self.URL, _b(req), _b(resp), 200)
        assert call.operation == "embedding"
        assert call.output == {"vectors": 2, "dimensions": 4}
        assert call.input_tokens == 4


class TestImages:
    URL = "https://api.openai.com/v1/images/generations"

    def test_url_response(self) -> None:
        req = {"model": "dall-e-3", "prompt": "a cat", "n": 2}
        resp = {
            "data": [
                {"url": "https://oaidalleapiprodscus.blob.core.windows.net/x.png"},
                {"url": "https://oaidalleapiprodscus.blob.core.windows.net/y.png"},
            ]
        }
        call = parse(self.URL, _b(req), _b(resp), 200)
        assert call.operation == "image"
        assert isinstance(call.output, dict)
        assert call.output["images"] == 2
        assert len(call.output["urls"]) == 2

    def test_b64_response(self) -> None:
        req = {"model": "gpt-image-1", "prompt": "a dog", "response_format": "b64_json"}
        resp = {"data": [{"b64_json": "iVBORw0KGgoAAAANS..."}]}
        call = parse(self.URL, _b(req), _b(resp), 200)
        assert call.output == {"images": 1, "b64_json_count": 1}


class TestAudio:
    def test_transcription_text_extracted(self) -> None:
        url = "https://api.openai.com/v1/audio/transcriptions"
        # multipart upload — we only see the response body
        resp = {
            "text": "Hello world",
            "language": "english",
            "duration": 1.23,
        }
        call = parse(url, b"", _b(resp), 200)
        assert call.operation == "audio"
        assert call.output == "Hello world"
        assert call.extra["language"] == "english"
        assert call.extra["duration_seconds"] == 1.23

    def test_translation_text_extracted(self) -> None:
        url = "https://api.openai.com/v1/audio/translations"
        resp = {"text": "Bonjour"}
        call = parse(url, b"", _b(resp), 200)
        assert call.output == "Bonjour"

    def test_tts_input_text_surfaced(self) -> None:
        url = "https://api.openai.com/v1/audio/speech"
        req = {"model": "tts-1", "voice": "alloy", "input": "Speak this please"}
        # TTS returns binary audio — the response body is not JSON.
        call = parse(url, _b(req), b"", 200)
        assert call.operation == "audio"
        assert isinstance(call.output, dict)
        assert call.output["audio"] == "<binary>"
        assert call.output["input_text"] == "Speak this please"


class TestModeration:
    URL = "https://api.openai.com/v1/moderations"

    def test_flagged_categories_extracted(self) -> None:
        req = {"model": "omni-moderation-latest", "input": "some text"}
        resp = {
            "results": [
                {
                    "flagged": True,
                    "categories": {
                        "harassment": True,
                        "hate": False,
                        "violence": True,
                    },
                }
            ]
        }
        call = parse(self.URL, _b(req), _b(resp), 200)
        assert call.operation == "moderation"
        assert call.output is not None
        assert call.output["flagged"] is True
        assert set(call.output["categories"]) == {"harassment", "violence"}
        assert call.output["result_count"] == 1
