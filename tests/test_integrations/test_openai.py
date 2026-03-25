"""Tests for OpenAI integration helpers."""

from tracium.integrations.openai import _extract_token_usage


class _FakeUsage:
    """Minimal object that mirrors an OpenAI CompletionUsage with model_dump."""

    def __init__(self, data: dict):
        self._data = data
        for k, v in data.items():
            setattr(self, k, v)

    def model_dump(self):
        return dict(self._data)


class _FakeResponse:
    def __init__(self, usage=None):
        self.usage = usage


# ---------------------------------------------------------------------------
# _extract_token_usage — cached token subtraction
# ---------------------------------------------------------------------------
class TestExtractTokenUsage:
    def test_subtracts_cached_from_prompt(self):
        """OpenAI prompt_tokens includes cached; output should be non-cached only."""
        usage = _FakeUsage(
            {
                "prompt_tokens": 10_000,
                "completion_tokens": 5_000,
                "prompt_token_details": {"cached_tokens": 3_000},
            }
        )
        inp, out, cached = _extract_token_usage(_FakeResponse(usage))
        assert inp == 7_000  # 10000 - 3000
        assert out == 5_000
        assert cached == 3_000

    def test_zero_cached_no_change(self):
        usage = _FakeUsage(
            {
                "prompt_tokens": 10_000,
                "completion_tokens": 5_000,
                "prompt_token_details": {"cached_tokens": 0},
            }
        )
        inp, out, cached = _extract_token_usage(_FakeResponse(usage))
        assert inp == 10_000
        assert out == 5_000
        assert cached == 0

    def test_none_cached_no_subtraction(self):
        """When no cached info is available, input stays as-is."""
        usage = _FakeUsage(
            {
                "prompt_tokens": 10_000,
                "completion_tokens": 5_000,
            }
        )
        inp, out, cached = _extract_token_usage(_FakeResponse(usage))
        assert inp == 10_000
        assert out == 5_000
        assert cached is None

    def test_cached_exceeds_prompt_floors_at_zero(self):
        """Safety: if cached > prompt (shouldn't happen), floor at 0."""
        usage = _FakeUsage(
            {
                "prompt_tokens": 1_000,
                "completion_tokens": 500,
                "prompt_token_details": {"cached_tokens": 2_000},
            }
        )
        inp, out, cached = _extract_token_usage(_FakeResponse(usage))
        assert inp == 0
        assert out == 500
        assert cached == 2_000

    def test_no_usage_returns_nones(self):
        inp, out, cached = _extract_token_usage(_FakeResponse(usage=None))
        assert inp is None
        assert out is None
        assert cached is None

    def test_cached_input_tokens_field(self):
        """Fallback: cached_input_tokens at top level of usage."""
        usage = _FakeUsage(
            {
                "prompt_tokens": 8_000,
                "completion_tokens": 2_000,
                "cached_input_tokens": 3_000,
            }
        )
        inp, out, cached = _extract_token_usage(_FakeResponse(usage))
        assert inp == 5_000  # 8000 - 3000
        assert out == 2_000
        assert cached == 3_000

    def test_total_tokens_fallback_no_cached(self):
        """When only total_tokens exists, used as input with no cached subtraction."""
        usage = _FakeUsage({"total_tokens": 15_000})
        inp, out, cached = _extract_token_usage(_FakeResponse(usage))
        assert inp == 15_000
        assert out is None
        assert cached is None
