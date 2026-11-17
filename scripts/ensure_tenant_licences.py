"""Operator CLI: ensure an EXISTING w2v tenant's licences and record its ``licence_ref``.

The same operation ``POST /api/tenants`` now runs on create. Both call
:func:`src.services.license_provisioning.ensure_tenant_licence_ref`: the tenant's
``w2v`` bundle is ensured in licences (held by ``svc:<tenant>:<app>``,
idempotent ``exists`` / ``adopt`` / ``insert``) and the ``w2v`` licence becomes
the tenant's ``licence_ref``. It exists for a tenant that was created before
that, or whose create was refused half-way (live 2026-10-15: a partner-created
tenant had ``licence_ref = null`` and was repaired with ad-hoc Python in the
pod; this script replaces that).

* **Dry run is the DEFAULT**: it reads the tenant's registry row and reports
  what it would do. It makes NO licences call (the ensure is a write there).
* ``--apply`` ensures and records, then commits. Re-running is idempotent:
  licences answers ``exists`` and the same ``licence_ref`` is kept.
* A STORED licence token owns ``licence_ref`` (only the token path may change
  it): the bundle is still ensured, and the token's ref is kept and reported.
* No ``admin_audit_log`` row: the ensure writes no new w2v fact beyond
  ``licence_ref`` (exactly what ``POST /api/tenants`` records without an audit
  row), and that table's action set is closed by a CHECK. The printed
  before/after is the operator's trail; licences records the ensure itself.
* The tenant must already be registered in licences as w2v's (``POST
  /api/tenants`` / ``/api/admin/provision`` do that); otherwise licences refuses
  and the refusal is printed.

Run inside the orchestrator pod (it needs the pod's DB and licences env)::

    kubectl -n w2v exec deploy/w2v-orchestrator -c w2v-orchestrator -- \\
        python -m scripts.ensure_tenant_licences --tenant-id T            # dry run
    kubectl -n w2v exec deploy/w2v-orchestrator -c w2v-orchestrator -- \\
        python -m scripts.ensure_tenant_licences --tenant-id T --apply

Exit codes: 0 ok (dry run or applied) · 1 refused (no such tenant, or licences
refused the licence — the reason is printed), nothing recorded · 2 aborted
(bad input / licences unreachable / unexpected error), nothing recorded.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.database.repositories.tenant_licence_repository import TenantLicenceRepository
from src.services.license_provisioning import (
    TenantLicenceRefusedError,
    TenantNotRegisteredError,
    ensure_tenant_licence_ref,
)

if TYPE_CHECKING:
    from src.clients.licenses_client import LicensesClient


async def run(
    session_maker: async_sessionmaker[AsyncSession],
    licences: LicensesClient,
    *,
    tenant_id: UUID,
    apply: bool,
    out: Callable[[str], None] = print,
) -> int:
    mode = "APPLY" if apply else "DRY RUN"
    out(f"{mode}: ensure w2v licences for tenant {tenant_id}")
    async with session_maker() as session:
        current = await TenantLicenceRepository(session).stored(tenant_id)
        if current is None or current.status == "deleted":
            out(f"REFUSED: tenant {tenant_id} has no live w2v registry row. Nothing recorded.")
            return 1
        before = current.licence_ref
        token_owned = current.ciphertext is not None
        out(
            f"CURRENT  {tenant_id}: status={current.status} licence_ref={before} "
            f"stored_token={'yes' if token_owned else 'no'}"
        )
        if not apply:
            out("DRY RUN — would ensure the w2v bundle in licences and record licence_ref.")
            out("DRY RUN — nothing written.")
            return 0
        try:
            ref = await ensure_tenant_licence_ref(session, licences, tenant_id=tenant_id)
        except TenantLicenceRefusedError as exc:
            await session.rollback()
            out(f"REFUSED: licences refused the {exc.app_key!r} licence: {exc.reason}.")
            out("Nothing recorded in w2v. Resolve the refusal in licences, then re-run.")
            return 1
        except TenantNotRegisteredError:
            await session.rollback()
            out(f"REFUSED: tenant {tenant_id} has no w2v registry row. Nothing recorded.")
            return 1
        await session.commit()
    verb = "UNCHANGED" if ref == before else "RECORDED"
    note = " (a stored token owns licence_ref; kept)" if token_owned and ref == before else ""
    out(f"{verb} {tenant_id}: licence_ref={ref}{note}")
    out("APPLIED (committed).")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--tenant-id", type=UUID, required=True)
    parser.add_argument("--apply", action="store_true", help="ensure + record (default: dry run)")
    return parser


async def _main(args: argparse.Namespace) -> int:
    try:
        from src.api.routes._common import get_licenses_client
        from src.database.session import get_session_maker

        return await run(
            await get_session_maker(),
            get_licenses_client(),
            tenant_id=args.tenant_id,
            apply=args.apply,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary: report, exit 2
        print(f"ABORTED, nothing recorded: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        # Dispose the pooled engine inside the loop (see scripts/bind_partner.py).
        from src.database.connection import close_engine

        await close_engine()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main(_parser().parse_args())))
