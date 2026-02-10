"""
Tests for span payload sanitizer (large content / base64 replacement).
"""

import base64

from tracium.utils.payload_sanitizer import PLACEHOLDER_OVER_SIZE, sanitize_span_io


class TestSanitizeSpanIO:
    def test_none_unchanged(self):
        assert sanitize_span_io(None) is None

    def test_small_string_unchanged(self):
        s = "Hello, world"
        assert sanitize_span_io(s) == s

    def test_data_url_replaced(self):
        large_data = "data:image/png;base64," + "x" * 3000
        out = sanitize_span_io(large_data)
        assert out == "[image/data omitted for size]"

    def test_large_base64_replaced(self):
        large_b64 = base64.b64encode(b"x" * 3000).decode("ascii")
        out = sanitize_span_io(large_b64)
        assert out == "[image/data omitted for size]"

    def test_short_data_url_unchanged(self):
        short = "data:image/png;base64,iVBORw0KGgo="
        assert len(short) < 2000
        out = sanitize_span_io(short)
        assert out == short

    def test_large_plain_string_truncated(self):
        from tracium.utils.payload_sanitizer import (
            MAX_STRING_BYTES,
            PLACEHOLDER_OVER_SIZE,
            PLACEHOLDER_TRUNCATED,
        )

        large = "plain text " * (MAX_STRING_BYTES // 10 + 100)
        out = sanitize_span_io(large)
        assert PLACEHOLDER_TRUNCATED in out or out == PLACEHOLDER_OVER_SIZE
        if isinstance(out, str) and out != PLACEHOLDER_OVER_SIZE:
            assert len(out.encode("utf-8")) <= MAX_STRING_BYTES + 100

    def test_dict_image_url_replaced(self):
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + "y" * 3000}},
            ],
        }
        out = sanitize_span_io(msg)
        assert out["content"][0]["text"] == "What's in this image?"
        assert out["content"][1]["image_url"]["url"] == "[image/data omitted for size]"

    def test_nested_structure_preserved(self):
        payload = {"messages": [{"role": "user", "content": "Short"}]}
        assert sanitize_span_io(payload) == payload

    def test_does_not_mutate_original(self):
        original = {"a": "data:image/png;base64," + "z" * 3000}
        out = sanitize_span_io(original)
        assert original["a"].startswith("data:image")
        assert out["a"] == "[image/data omitted for size]"

    def test_total_over_100kb_replaced(self):
        big_list = ["plain chunk " * 400 for _ in range(30)]
        out = sanitize_span_io(big_list)
        assert out == PLACEHOLDER_OVER_SIZE
