"""
Tests for _combine_llm_info and LLMTraceSummary.
"""

import pytest

from tracium.models.trace_handle import _combine_llm_info
from tracium.models.trace_state import LLMTraceSummary


class TestLLMTraceSummary:
    """Tests for the LLMTraceSummary dataclass."""

    def test_defaults(self):
        summary = LLMTraceSummary()
        assert summary.model is None
        assert summary.system_prompt is None
        assert summary.llm_steps is None

    def test_frozen(self):
        summary = LLMTraceSummary(model="gpt-4")
        with pytest.raises(AttributeError):
            summary.model = "gpt-3.5"  # type: ignore[misc]

    def test_equality(self):
        a = LLMTraceSummary(model="gpt-4", system_prompt="Be helpful.")
        b = LLMTraceSummary(model="gpt-4", system_prompt="Be helpful.")
        assert a == b

    def test_with_llm_steps(self):
        steps = ({"name": "step1", "model": "gpt-4", "system_prompt": "prompt"},)
        summary = LLMTraceSummary(model="gpt-4", system_prompt="prompt", llm_steps=steps)
        assert summary.llm_steps == steps


class TestCombineLlmInfoEmpty:
    """Tests for empty/None inputs."""

    def test_empty_list(self):
        assert _combine_llm_info([]) is None

    def test_all_none_values(self):
        result = _combine_llm_info([(None, None, None)])
        assert result is not None
        assert result.model is None
        assert result.system_prompt is None
        assert result.llm_steps == ({"name": None, "model": None, "system_prompt": None},)


class TestCombineLlmInfoSingleSpan:
    """Tests for single LLM span traces."""

    def test_single_span(self):
        result = _combine_llm_info([("chat", "gpt-4", "You are helpful.")])
        assert result is not None
        assert result.model == "gpt-4"
        assert result.system_prompt == "You are helpful."
        assert result.llm_steps == (
            {"name": "chat", "model": "gpt-4", "system_prompt": "You are helpful."},
        )

    def test_single_span_no_prompt(self):
        result = _combine_llm_info([("chat", "gpt-4", None)])
        assert result is not None
        assert result.model == "gpt-4"
        assert result.system_prompt is None

    def test_single_span_no_model(self):
        result = _combine_llm_info([("chat", None, "Be concise.")])
        assert result is not None
        assert result.model is None
        assert result.system_prompt == "Be concise."


class TestCombineLlmInfoMultiSpan:
    """Tests for multi-LLM span traces."""

    def test_same_model_same_prompt(self):
        info = [
            ("step1", "gpt-4", "Be helpful."),
            ("step2", "gpt-4", "Be helpful."),
        ]
        result = _combine_llm_info(info)
        assert result is not None
        # Ordered per-step: duplicates preserved for accurate fingerprinting
        assert result.model == "gpt-4 | gpt-4"
        assert result.system_prompt == "Be helpful. | Be helpful."
        assert len(result.llm_steps) == 2

    def test_different_models(self):
        info = [
            ("step1", "gpt-4", "prompt"),
            ("step2", "claude-3", "prompt"),
        ]
        result = _combine_llm_info(info)
        assert result is not None
        # Ordered per-step, not sorted
        assert result.model == "gpt-4 | claude-3"
        assert result.system_prompt == "prompt | prompt"

    def test_different_prompts(self):
        info = [
            ("step1", "gpt-4", "Alpha prompt"),
            ("step2", "gpt-4", "Beta prompt"),
        ]
        result = _combine_llm_info(info)
        assert result is not None
        assert result.model == "gpt-4 | gpt-4"
        assert result.system_prompt == "Alpha prompt | Beta prompt"

    def test_different_models_and_prompts(self):
        info = [
            ("planner", "gpt-4", "Plan tasks."),
            ("executor", "claude-3", "Execute tasks."),
        ]
        result = _combine_llm_info(info)
        assert result is not None
        assert result.model == "gpt-4 | claude-3"
        assert result.system_prompt == "Plan tasks. | Execute tasks."
        assert len(result.llm_steps) == 2

    def test_mixed_none_values(self):
        info = [
            ("step1", "gpt-4", None),
            ("step2", None, "Be concise."),
        ]
        result = _combine_llm_info(info)
        assert result is not None
        assert result.model == "gpt-4"
        assert result.system_prompt == "Be concise."
        assert len(result.llm_steps) == 2

    def test_preserves_step_order(self):
        info = [
            ("first", "model-a", "prompt-a"),
            ("second", "model-b", "prompt-b"),
            ("third", "model-c", "prompt-c"),
        ]
        result = _combine_llm_info(info)
        assert result is not None
        names = [step["name"] for step in result.llm_steps]
        assert names == ["first", "second", "third"]

    def test_preserves_duplicates_for_fingerprinting(self):
        """Duplicate models/prompts are preserved so per-step changes are detected."""
        info = [
            ("s1", "gpt-4", "prompt"),
            ("s2", "gpt-4", "prompt"),
            ("s3", "gpt-4", "prompt"),
        ]
        result = _combine_llm_info(info)
        assert result is not None
        assert result.model == "gpt-4 | gpt-4 | gpt-4"
        assert result.system_prompt == "prompt | prompt | prompt"

    def test_single_step_change_produces_different_summary(self):
        """Changing one step in a multi-step agent produces different model/prompt strings."""
        before = [
            ("planner", "gpt-4", "Plan"),
            ("coder", "gpt-4", "Code"),
            ("reviewer", "claude-3", "Review"),
        ]
        after = [
            ("planner", "gpt-4", "Plan"),
            ("coder", "claude-3", "Code"),  # changed model
            ("reviewer", "claude-3", "Review"),
        ]
        result_before = _combine_llm_info(before)
        result_after = _combine_llm_info(after)
        assert result_before.model != result_after.model

    def test_swapped_prompts_produces_different_summary(self):
        """Swapping prompts between steps produces different prompt strings."""
        before = [("a", "gpt-4", "Alpha"), ("b", "gpt-4", "Beta")]
        after = [("a", "gpt-4", "Beta"), ("b", "gpt-4", "Alpha")]
        result_before = _combine_llm_info(before)
        result_after = _combine_llm_info(after)
        assert result_before.system_prompt != result_after.system_prompt
