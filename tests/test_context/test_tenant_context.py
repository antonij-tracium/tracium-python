"""
Tests for tenant context management.
"""

import threading

from tracium.context.tenant_context import get_current_tenant, set_tenant


class TestTenantContext:
    """Tests for tenant context management."""

    def test_get_current_tenant_none_initially(self):
        """Test that get_current_tenant returns None initially."""
        set_tenant(None)
        assert get_current_tenant() is None

    def test_set_and_get_tenant(self):
        """Test setting and getting tenant context."""
        tenant_id = "tenant-123"
        set_tenant(tenant_id)
        assert get_current_tenant() == tenant_id

    def test_tenant_context_thread_local(self):
        """Test that tenant context is thread-local."""
        tenant1 = "tenant-1"
        tenant2 = "tenant-2"

        def set_tenant_in_thread(tenant):
            set_tenant(tenant)
            assert get_current_tenant() == tenant

        set_tenant(tenant1)
        assert get_current_tenant() == tenant1

        thread = threading.Thread(target=set_tenant_in_thread, args=(tenant2,))
        thread.start()
        thread.join()

        assert get_current_tenant() == tenant1

    def test_clear_tenant(self):
        """Test clearing tenant context."""
        set_tenant("tenant-123")
        assert get_current_tenant() == "tenant-123"

        set_tenant(None)
        assert get_current_tenant() is None
