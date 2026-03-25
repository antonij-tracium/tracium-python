"""Tests for trace handle fail() and _finish() sending llm_summary."""

from unittest.mock import MagicMock

from tracium.models.trace_handle import AgentTraceHandle, AgentTraceManager
from tracium.models.trace_state import TraceState


def _make_state(client, *, llm_info=None):
    state = TraceState(
        client=client,
        trace_id="test-trace-id",
        agent_name="test-agent",
        tags=[],
        start_payload={},
        model_id=None,
        workspace_id=None,
        version=None,
        remote_started=True,
        has_spans=True,
    )
    if llm_info:
        state._llm_info = llm_info
    return state


class TestFailSendsLLMSummary:
    def test_fail_sends_llm_summary(self):
        mock_client = MagicMock()
        mock_client.fail_agent_trace.return_value = {
            "trace_id": "test-trace-id",
            "status": "failed",
        }
        llm_info = [("step1", "gpt-4", "Be helpful")]
        state = _make_state(mock_client, llm_info=llm_info)
        handle = AgentTraceHandle(state)

        handle.fail(error="something broke")

        mock_client.fail_agent_trace.assert_called_once()
        call_kwargs = mock_client.fail_agent_trace.call_args
        assert call_kwargs.kwargs["error"] == "something broke"
        llm_summary = call_kwargs.kwargs["llm_summary"]
        assert llm_summary is not None
        assert llm_summary.model == "gpt-4"
        assert llm_summary.system_prompt == "Be helpful"

    def test_fail_sends_none_llm_summary_when_no_llm_info(self):
        mock_client = MagicMock()
        mock_client.fail_agent_trace.return_value = {
            "trace_id": "test-trace-id",
            "status": "failed",
        }
        state = _make_state(mock_client)
        handle = AgentTraceHandle(state)

        handle.fail(error="something broke")

        call_kwargs = mock_client.fail_agent_trace.call_args
        assert call_kwargs.kwargs["llm_summary"] is None

    def test_context_manager_exit_with_exception_sends_llm_summary(self):
        mock_client = MagicMock()
        mock_client.start_agent_trace.return_value = {"id": "test-trace-id"}
        mock_client.fail_agent_trace.return_value = {
            "trace_id": "test-trace-id",
            "status": "failed",
        }

        manager = AgentTraceManager(
            mock_client,
            agent_name="test-agent",
            model_id=None,
            metadata=None,
            tags=None,
            trace_id="test-trace-id",
        )

        try:
            with manager as handle:
                handle._state._llm_info = [("step1", "claude-3", "You are an agent")]
                raise ValueError("test error")
        except ValueError:
            pass

        mock_client.fail_agent_trace.assert_called_once()
        call_kwargs = mock_client.fail_agent_trace.call_args
        llm_summary = call_kwargs.kwargs["llm_summary"]
        assert llm_summary is not None
        assert llm_summary.model == "claude-3"
        assert llm_summary.system_prompt == "You are an agent"

    def test_fail_with_multi_step_llm_info(self):
        mock_client = MagicMock()
        mock_client.fail_agent_trace.return_value = {
            "trace_id": "test-trace-id",
            "status": "failed",
        }
        llm_info = [
            ("planner", "gpt-4", "Plan tasks"),
            ("executor", "claude-3", "Execute tasks"),
        ]
        state = _make_state(mock_client, llm_info=llm_info)
        handle = AgentTraceHandle(state)

        handle.fail(error="multi-step failure")

        call_kwargs = mock_client.fail_agent_trace.call_args
        llm_summary = call_kwargs.kwargs["llm_summary"]
        assert llm_summary is not None
        assert llm_summary.model == "gpt-4 | claude-3"
        assert llm_summary.system_prompt == "Plan tasks | Execute tasks"
        assert len(llm_summary.llm_steps) == 2
        assert llm_summary.llm_steps[0]["name"] == "planner"
        assert llm_summary.llm_steps[1]["name"] == "executor"

    def test_lazy_start_fail_before_remote_start_skips_api_call(self):
        mock_client = MagicMock()

        manager = AgentTraceManager(
            mock_client,
            agent_name="lazy-agent",
            model_id=None,
            metadata=None,
            tags=None,
            trace_id=None,
            lazy_start=True,
        )

        try:
            with manager:
                # No spans recorded, remote_started=False
                raise RuntimeError("early failure")
        except RuntimeError:
            pass

        # Should NOT call fail_agent_trace since trace was never started remotely
        mock_client.fail_agent_trace.assert_not_called()
        mock_client.start_agent_trace.assert_not_called()
