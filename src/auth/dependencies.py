"""FastAPI auth dependencies — canonical per-user-within-tenant scoping.

Two scoping axes:
- tenant_id: the organization the caller represents (X-Tenant-Id introspected
    from a license token, or the X-Tenant-Id header on trusted networks).
- subject_id: the user within the tenant (X-Subject-Id header injected by
    license-authz from the licenses DB Subject UUID; empty string when the
    caller has not provided one — treated as "tenant-only caller" downstream).

Mirrors procedure_management/src/auth/dependencies.py (commit 7f372e0,
canonical standard 2026-04-27).
"""

from typing import Annotated

from fastapi import Header, Request

VISIBILITY_VALUES = frozenset({"tenant", "user"})


# NOTE: the tenant dependency lives in ``src/api/routes/_common.py`` and
# *fails closed* (401 on missing tenant). A fail-OPEN ``get_tenant_id`` that
# defaulted to ``"anonymous"`` used to live here; it was dead (re-exported,
# imported nowhere) and a footgun, so it was removed. Use ``TenantIdDep`` from
# ``_common.py`` for tenant scoping — never reintroduce an "anonymous" default.


async def get_subject_id(request: Request) -> str:
    """Get subject (user) ID from request state.

    Returns empty string when the caller has not provided X-Subject-Id (Plan 2
    not yet rolled out, or trusted-network call without a header). Empty
    string is treated as "tenant-only caller" downstream:

    - list queries return only visibility='tenant' rows in scope().
    - mutations land with subject_id := tenant_id via the before_insert
      listener, preserving the legacy backfilled semantics for any tenant
      member.

    Returns a string (never raises) so this dependency is optional from the
    caller's perspective and matches the plan's transition strategy.
    """
    return str(getattr(request.state, "subject_id", "") or "")


def get_visibility(
    x_visibility: Annotated[str, Header(alias="X-Visibility")] = "tenant",
) -> str:
    """Caller-supplied visibility for new-row creation. Read paths ignore this.

    Valid values: 'tenant' (default — every tenant member sees + may mutate)
    or 'user' (only the creating subject sees + may mutate).

    Invalid values silently coerce to 'tenant' (the safer default — exposes
    rather than over-restricts; rejecting would break legacy clients).
    Surfaces in OpenAPI so Orval/n8n consumers can discover it.
    """
    raw = x_visibility.strip().lower()
    return raw if raw in VISIBILITY_VALUES else "tenant"
