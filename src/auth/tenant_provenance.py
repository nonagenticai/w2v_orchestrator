"""Carry a run's tenant PROVENANCE across the request → background-task boundary.

w2v-common 0.67.0 added a fifth piece to the identity context:
``TenantSource`` — HOW the tenant was established. It is not decoration and it
is not a type error when you forget it, which is exactly what makes it
dangerous:

``_apply_outbound_tenant`` (w2v_common/auth/identity_context.py) takes the
DELEGATED-HOP branch whenever the request already carries a service-account
``X-Tenant-Id: Bearer <token>`` **and** the destination is metered. Every
orchestrator LLM call is that shape — ``WMClient`` bakes the SA token and
``worker_management`` is a metered destination. On that branch an unrecorded
provenance is refused::

    raise UnvouchedTenantError(... "provenance is 'unrecorded', which is not
    vouched ...")

``RunDispatcher`` rebinds identity for a background run task from the run's
``config_snapshot`` (there is no inbound request on that path, so
``AuthMiddleware`` never populated the ContextVars). If it rebinds tenant +
license + subject but NOT the source, every metered call the run makes raises
— and nothing in the type checker, the linter or a mocked-client test says so.

So the source is snapshotted at run creation, in REQUEST scope where it is
still bound, and rebound alongside the other three.

**Legacy runs deliberately keep failing.** A run row created before this field
existed has no recorded provenance, and there is no honest way to recover it:
inventing ``LICENSE_TOKEN`` here would re-create the exact fail-open the ladder
exists to close — a bare header that any peer can set becoming an identity to
spend a tenant's credentials as. ``None`` is returned, the outbound hop refuses,
and the operator restarts the run.
"""

from __future__ import annotations

import logging

from w2v_common.auth.identity_context import get_tenant_source_from_context
from w2v_common.tenant import TenantSource

__all__ = ["snapshot_tenant_source", "tenant_source_from_snapshot"]

logger = logging.getLogger(__name__)

#: The ``config_snapshot`` key. Named to match ``license_id`` / ``subject_id``.
TENANT_SOURCE_KEY = "tenant_source"


def snapshot_tenant_source() -> str:
    """Render the request's bound ``TenantSource`` for storage in a run snapshot.

    Returns ``""`` when nothing is bound — the same "absent" spelling
    ``license_id`` already uses in ``config_snapshot``, so the JSON column keeps
    one convention.
    """
    source = get_tenant_source_from_context()
    return source.value if source is not None else ""


def tenant_source_from_snapshot(value: object) -> TenantSource | None:
    """Parse a snapshotted provenance back into a :class:`TenantSource`.

    ``None`` for anything unusable — absent (a pre-existing run), empty, or a
    member name this build of w2v-common no longer knows. ``None`` is the
    UNVOUCHED end of the ladder, so an unreadable value degrades to refusal at
    the first metered hop rather than to a delegation nobody can stand behind.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        return TenantSource(value)
    except ValueError:
        # A source this build does not know. Loud, because it means the run was
        # created by a NEWER orchestrator than the one now dispatching it.
        logger.error(
            "run config_snapshot carries an unknown tenant_source %r — treating it as "
            "unvouched, so metered downstream calls for this run will refuse. This means "
            "a newer w2v-common created the run than the one dispatching it.",
            value,
        )
        return None
