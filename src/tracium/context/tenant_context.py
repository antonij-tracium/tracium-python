"""
Tenant context management using contextvars.

This module provides thread-safe tenant context management that works with
async/await and thread pools. The tenant ID is stored in a context variable
that automatically propagates through async contexts and can be explicitly
propagated to threads.
"""

from __future__ import annotations

import contextvars

_current_tenant: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_tenant", default=None
)


def set_tenant(tenant_id: str | None) -> None:
    """
    Set the current tenant ID in the context.

    This tenant ID will be automatically included in all API requests made
    by the Tracium SDK. The tenant ID is stored in a context variable, which
    means it automatically propagates through async/await boundaries but may
    need explicit propagation to threads (handled automatically by the SDK).

    Args:
        tenant_id: The tenant ID to set, or None to clear the tenant context.

    Example:
        >>> import tracium
        >>> tracium.trace(api_key="sk_live_...")
        >>>
        >>> tracium.set_tenant("tenant-123")
        >>>
        >>> with tracium.agent_trace(agent_name="my-agent"):
        ...     pass

    For Flask/FastAPI applications:
        >>> @app.before_request
        >>> def set_tenant_context():
        ...     tenant_id = get_tenant_from_request(request)
        ...     tracium.set_tenant(tenant_id)
    """
    _current_tenant.set(tenant_id)


def get_current_tenant() -> str | None:
    """
    Get the current tenant ID from the context.

    Returns:
        The current tenant ID if set, None otherwise.

    Example:
        >>> import tracium
        >>> tracium.set_tenant("tenant-123")
        >>> tenant = tracium.get_current_tenant()
        >>> print(tenant)
    """
    return _current_tenant.get()
