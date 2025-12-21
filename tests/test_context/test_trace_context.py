"""
Tests for trace context management.
"""


from tracium.context.trace_context import current_trace


class TestTraceContext:
    """Tests for trace context management."""

    def test_current_trace_none_initially(self):
        """Test that current_trace returns None initially."""
        assert current_trace() is None

    def test_current_trace_returns_handle_when_trace_exists(self, tracium_client):
        """Test that current_trace returns a handle when a trace is active."""
        with tracium_client.agent_trace(agent_name="test-agent") as trace_handle:
            current = current_trace()
            assert current is not None
            assert current.id == trace_handle.id
            assert current.agent_name == "test-agent"

        assert current_trace() is None
