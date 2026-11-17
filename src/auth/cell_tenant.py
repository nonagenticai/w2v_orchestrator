"""Bind the tenant an outbound in-cell call acts for, for that ONE send (P7).

w2v_common's outbound hooks (``identity_event_hook`` / ``cell_token_event_hook``)
mint the ``X-W2V-Cell-Token`` from the AMBIENT identity context, which holds
the tenant the inbound request authenticated as. Some calls explicitly act for
a tenant resolved later:

* the analysis / html_render proxies act for ``TenantIdDep`` (a claim-mode
  ``super_admin`` scope switch makes it differ from the claim);
* the youtube_upload client acts for the explicit owner tenant the route
  resolved from membership.

A cell token naming the ambient tenant there would make the callee read the
WRONG tenant once it trusts the token's ``tenant`` claim. :func:`send_as_tenant`
runs the send in a COPIED context with the explicit tenant bound, so the hook
signs that tenant and nothing leaks back into the request's own context.
"""

from __future__ import annotations

import asyncio
import contextvars
from collections.abc import Awaitable
from uuid import UUID

from w2v_common.auth.identity_context import (
    get_actor_id_from_context,
    get_caller_license_token,
    get_license_id_from_context,
    get_subject_id_from_context,
    get_tenant_id_from_context,
    get_tenant_source_from_context,
    set_identity_context,
)

__all__ = ["send_as_tenant"]


def _rebind_tenant(tenant: UUID) -> None:
    """Bind ``tenant`` as the identity tenant, keeping the rest of the identity.

    Only ever called inside a copied context. Provenance is carried over
    EXPLICITLY (``set_identity_context`` would otherwise drop it on a tenant
    change): the explicit tenant was resolved server-side for the SAME
    authenticated principal — ``get_tenant_id`` switches only for a verified
    ``super_admin``, and the yt owner comes from that same dependency — so it is
    exactly as vouched as the tenant it replaces. An unvouched ambient lane
    stays unvouched, and the hook still refuses to sign it.

    The actor (the human ``sub``) is kept; the w2v project/job are dropped by
    ``set_identity_context`` because they belong to the previous tenant.
    """
    set_identity_context(
        tenant,
        get_license_id_from_context(),
        get_subject_id_from_context(),
        get_caller_license_token(),
        tenant_source=get_tenant_source_from_context(),
        actor_id=get_actor_id_from_context(),
    )


async def _await[T](send: Awaitable[T]) -> T:
    return await send


async def send_as_tenant[T](tenant: UUID | None, send: Awaitable[T]) -> T:
    """Await ``send`` with ``tenant`` as the identity tenant of its outbound hooks.

    ``None`` or the already-bound tenant: awaited as-is (the common case — no
    task, no copy). Otherwise the coroutine runs as a task in a copy of the
    current context with ``tenant`` bound; cancellation and exceptions propagate
    unchanged.
    """
    if tenant is None or tenant == get_tenant_id_from_context():
        return await send
    ctx = contextvars.copy_context()
    ctx.run(_rebind_tenant, tenant)
    return await asyncio.create_task(_await(send), context=ctx)
