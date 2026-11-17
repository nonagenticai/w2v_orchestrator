"""One-time backfill: reconcile project.status / finished_at for existing rows.

Projects created before the auto-status lifecycle (migration 0019) are frozen at
whatever status they were created/patched with — usually 'drafting'. This script
walks every non-archived project and runs the same
``ProjectRepository.recompute_status`` the dispatcher now calls on every run
transition, so historical projects reflect their real run state.

⛔ IT USED TO SCAN NOTHING AND CALL THAT SUCCESS. Measured against prod
2026-08-28: ``scanned=0 would_change=0``, exit 0, while the database held **183
projects (164 non-archived)**. ``projects`` is RLS-protected
(``tenant_id = current_setting('app.current_tenant')``) and ``w2v_orch_app`` does
NOT carry BYPASSRLS, so with no tenant bound the policy compares against NULL and
admits ZERO rows. The old single-session walk therefore produced a clean green
run that had reconciled nothing — and its log line was indistinguishable from a
genuinely reconciled estate. A zero-row read under RLS is not absence.

The fix is two things, and the second matters more than the first:
  1. Bind ``app.current_tenant`` per tenant and walk each tenant's projects.
  2. A VACUITY FLOOR: scanning zero projects while tenants exist is now a
     FAILURE (exit 1), not a quiet success. A maintenance tool that cannot tell
     "nothing to do" from "I could not see anything" is worse than no tool.

Idempotent: re-running is a no-op once everything is reconciled (recompute only
writes when a value changes). Run once post-deploy against the prod DB:

    DATABASE_URL=postgresql+psycopg://... uv run python -m scripts.backfill_project_status

Add ``--dry-run`` to log intended changes without writing.

⚠️ TENANT ENUMERATION IS BEST-EFFORT AND THE SCRIPT SAYS SO. Without --tenant it
reads ``tenants`` (the only non-RLS table carrying tenant ids). Measured on prod
2026-08-28, **2 of the tenant ids that own projects are absent from that table**
(0195f48b-…-0004 and 019de842-…96f9), so a bare run silently skips their rows.
There is no way for a non-BYPASSRLS role to enumerate ``DISTINCT tenant_id FROM
projects`` — the policy hides exactly the rows you would enumerate from. Pass
--tenant explicitly for those, or check with a DBA query:

    SELECT DISTINCT tenant_id FROM projects;   -- superuser / BYPASSRLS only
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.database.models import Project
from src.database.repositories.project_repository import ProjectRepository
from src.database.session import bind_session_tenant, get_session_maker

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("backfill_project_status")


async def _known_tenants(session_maker: async_sessionmaker[AsyncSession]) -> list[UUID]:
    """Tenant ids from the non-RLS ``tenants`` table. See the module docstring:
    this is NOT guaranteed to cover every tenant that owns a project."""
    async with session_maker() as session:
        rows = (await session.execute(text("SELECT id FROM tenants"))).scalars().all()
    return [UUID(str(r)) for r in rows]


async def _backfill_tenant(
    session_maker: async_sessionmaker[AsyncSession], tenant_id: UUID, dry_run: bool
) -> tuple[int, int]:
    scanned = 0
    changed = 0
    async with session_maker() as session:
        # MUST precede the first query, or the RLS policy admits nothing.
        await bind_session_tenant(session, tenant_id)
        # Children (non-null parent) first, then top-level parents — so a course
        # is recomputed after its lessons and its rollup sees fresh child status.
        ids = (
            (
                await session.execute(
                    select(Project.id)
                    .where(Project.status != "archived")
                    .order_by(Project.parent_project_id.nulls_last())
                )
            )
            .scalars()
            .all()
        )
        repo = ProjectRepository(session)
        for project_id in ids:
            scanned += 1
            before = (
                await session.execute(select(Project.status).where(Project.id == project_id))
            ).scalar_one()
            new = await repo.recompute_status(project_id)
            if new != before:
                changed += 1
                log.info("tenant %s project %s: %s -> %s", tenant_id, project_id, before, new)
        if dry_run:
            await session.rollback()
        else:
            await session.commit()
    return scanned, changed


async def main(dry_run: bool, tenants: list[UUID] | None) -> int:
    session_maker = await get_session_maker()
    tenant_ids = tenants or await _known_tenants(session_maker)
    if not tenants:
        log.warning(
            "enumerated %d tenant(s) from `tenants`; any tenant that owns projects but is "
            "absent from that table is INVISIBLE to this run — pass --tenant for those",
            len(tenant_ids),
        )

    total_scanned = 0
    total_changed = 0
    for tenant_id in tenant_ids:
        scanned, changed = await _backfill_tenant(session_maker, tenant_id, dry_run)
        log.info("tenant %s: scanned=%d changed=%d", tenant_id, scanned, changed)
        total_scanned += scanned
        total_changed += changed

    verb = "DRY RUN — rolled back." if dry_run else "committed."
    log.info(
        "%s tenants=%d scanned=%d changed=%d", verb, len(tenant_ids), total_scanned, total_changed
    )

    # ── THE VACUITY FLOOR ────────────────────────────────────────────────────
    # Scanning nothing while tenants exist is the RLS failure this script shipped
    # with for its whole life, and it looked exactly like success. Never again:
    # a run that saw no projects reports FAILURE and says why.
    if total_scanned == 0 and tenant_ids:
        log.error(
            "SCANNED ZERO PROJECTS across %d tenant(s). This is a FAILURE, not an empty "
            "estate: `projects` is RLS-protected and this role has no BYPASSRLS, so an "
            "unbound or wrong `app.current_tenant` admits zero rows. Check TENANT_RLS_MODE "
            "and that the --tenant ids actually own projects.",
            len(tenant_ids),
        )
        return 1
    if not tenant_ids:
        log.error("NO TENANTS to walk — `tenants` is empty and no --tenant was given.")
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="log changes without writing")
    parser.add_argument(
        "--tenant",
        action="append",
        type=UUID,
        dest="tenants",
        help="tenant uuid to walk (repeatable). Omit to read the `tenants` table, "
        "which the module docstring explains is not exhaustive.",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.dry_run, args.tenants)))
