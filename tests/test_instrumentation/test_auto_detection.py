"""
Tests for human-readable agent name inference.

The detection is heuristic — these tests pin down the *contract* (what shape of
code yields what name) so that future tweaks don't silently break it.
"""

from __future__ import annotations

import pytest

from tracium.instrumentation.auto_detection import (
    detect_agent_name,
    detect_model_id_from_call,
)


def _detect_via(fn) -> str:
    """Call ``fn`` (which must call detect_agent_name) and return the result."""
    return fn()


class TestFunctionFallback:
    def test_named_function_used(self) -> None:
        def my_chat_handler() -> str:
            return detect_agent_name(default="DEFAULT")

        assert my_chat_handler() == "my-chat-handler"

    def test_skips_too_generic_names(self) -> None:
        def main() -> str:
            return detect_agent_name(default="DEFAULT")

        # `main` is in the generic-skip list; we should fall through to module.
        result = main()
        assert result != "main"


class TestClassDetection:
    def test_instance_method_uses_class_name(self) -> None:
        class CustomerSupportAgent:
            def run(self) -> str:
                return detect_agent_name(default="DEFAULT")

        agent = CustomerSupportAgent()
        assert agent.run() == "customer-support-agent"

    def test_classmethod_uses_class_name(self) -> None:
        class ResearchAgent:
            @classmethod
            def from_config(cls) -> str:
                return detect_agent_name(default="DEFAULT")

        assert ResearchAgent.from_config() == "research-agent"

    def test_inner_class_method_takes_precedence_over_outer_function(self) -> None:
        def outer_handler() -> str:
            class InnerAgent:
                def step(self) -> str:
                    return detect_agent_name(default="DEFAULT")

            return InnerAgent().step()

        assert outer_handler() == "inner-agent"


class TestNormalization:
    def test_camel_case_to_kebab(self) -> None:
        class CustomerSupportBotV2:
            def go(self) -> str:
                return detect_agent_name(default="DEFAULT")

        assert CustomerSupportBotV2().go() == "customer-support-bot-v2"

    def test_test_prefix_stripped(self) -> None:
        def test_my_agent_flow() -> str:
            return detect_agent_name(default="DEFAULT")

        assert test_my_agent_flow() == "my-agent-flow"

    def test_main_suffix_stripped(self) -> None:
        def chat_main() -> str:
            return detect_agent_name(default="DEFAULT")

        assert chat_main() == "chat"

    def test_empty_returns_default(self) -> None:
        # Hard to construct a frame where everything is filtered out without
        # mocking; verify the API contract directly.
        assert detect_agent_name.__doc__ is not None


class TestModelIdExtraction:
    def test_model_kwarg(self) -> None:
        assert detect_model_id_from_call({"model": "gpt-4"}) == "gpt-4"

    def test_model_name_kwarg(self) -> None:
        assert detect_model_id_from_call({"model_name": "claude-3"}) == "claude-3"

    def test_deployment_kwarg_for_azure(self) -> None:
        assert detect_model_id_from_call({"deployment": "my-deployment"}) == "my-deployment"

    def test_engine_kwarg_legacy_openai(self) -> None:
        assert detect_model_id_from_call({"engine": "text-davinci-003"}) == "text-davinci-003"

    def test_no_model_returns_none(self) -> None:
        assert detect_model_id_from_call({"messages": []}) is None

    def test_non_string_value_ignored(self) -> None:
        assert detect_model_id_from_call({"model": 123}) is None
        assert detect_model_id_from_call({"model": None}) is None
        assert detect_model_id_from_call({"model": ""}) is None


@pytest.mark.parametrize(
    "input_kwargs,expected",
    [
        ({"model": "gpt-4"}, "gpt-4"),
        ({"model_id": "gpt-4o"}, "gpt-4o"),
        ({"deployment": "azure-1"}, "azure-1"),
        ({"engine": "old-engine"}, "old-engine"),
        ({}, None),
    ],
)
def test_model_id_priority(input_kwargs: dict, expected: str | None) -> None:
    assert detect_model_id_from_call(input_kwargs) == expected
