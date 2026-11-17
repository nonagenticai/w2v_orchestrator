"""Admin-tier aggregates + tenant registry writes (Phase B).

Cross-cutting queries that back the admin surface. The canonical ``tenants``
table (migration 0021) is a **name + status annotation layer keyed on the
claim-derived ``tenant_id``** — w2v never mints tenant ids (it is downstream of
licenses.test / Keycloak), so a tenant is "known" if it has a registry row
*or* at least one project. The platform list is the union of both; per-tenant
aggregates walk the Run → Deliverable → Project chain plus ``run_usage``.
"""

from __future__ import annotations

import datetime
from collections.abc import Sequence
from typing import Any, Literal, TypedDict
from uuid import UUID

from sqlalchemy import case, func, literal, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Row
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.selectable import Subquery

from src.database.models import (
    OPERATOR_HELD_TENANT_STATUSES,
    Deliverable,
    Project,
    Run,
    RunUsage,
    Tenant,
    TenantMember,
)
from src.database.repositories.run_usage_repository import (
    FUNDING_LOCAL_GPU,
    IN_PROCESS_USAGE_SOURCE,
)
from src.database.session import bind_session_tenant

# Per-bucket generate_series step. Keyed by the validated ``bucket`` Literal so
# the truncation granularity (date_trunc) and the gap-fill step share Postgres'
# own calendar arithmetic — no Python week/month re-derivation to drift.
TenantCreatedVia = Literal["super_admin", "partner", "backfill"]

_BUCKET_STEP = {
    "day": func.make_interval(0, 0, 0, 1),  # days=1
    "week": func.make_interval(0, 0, 1),  # weeks=1
    "month": func.make_interval(0, 1),  # months=1
}

#: Usage-attribution labels (P6 ORCH-3). A row with no attribution is reported under
#: ``UNATTRIBUTED``, never dropped — a NULL funding (not known) is the ``UNATTRIBUTED``
#: funding bucket; a billing-mirrored row's slot is ``SLOT_PLATFORM`` (the platform
#: ROUTE, gateway/WM — says nothing about who paid).
UNATTRIBUTED = "unattributed"
SLOT_PLATFORM = "platform"


class SlotUsage(TypedDict):
    slot: str
    funding: str
    call_count: int
    cost_usd: float
    input_tokens: int
    output_tokens: int
    units: dict[str, float]


def _fold_slot_rows(rows: Sequence[Row[Any]]) -> list[SlotUsage]:
    """``(slot, funding, unit_kind, calls, cost, in, out, units)`` groups → one entry
    per (slot, funding), with ``units`` keyed by unit kind. Cost-descending."""
    out: dict[tuple[str, str], SlotUsage] = {}
    for slot, raw_funding, unit_kind, calls, cost, tin, tout, units in rows:
        funding = raw_funding if raw_funding is not None else UNATTRIBUTED
        entry = out.setdefault(
            (slot, funding),
            {
                "slot": slot,
                "funding": funding,
                "call_count": 0,
                "cost_usd": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "units": {},
            },
        )
        entry["call_count"] += calls
        entry["cost_usd"] += float(cost)
        entry["input_tokens"] += int(tin)
        entry["output_tokens"] += int(tout)
        if unit_kind is not None:
            entry["units"][unit_kind] = entry["units"].get(unit_kind, 0.0) + float(units)
    return sorted(out.values(), key=lambda e: (-e["cost_usd"], e["slot"], e["funding"]))


class AdminRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    def _project_aggregate(self) -> Subquery:
        """Per-tenant project rollup: project_count + last_activity.

        Counts real project rows (archived + child projects included) so a tenant
        never disappears from the platform list just because its projects were
        archived. Hidden storyboard BRANCH rows (``is_branch = true``) are
        EXCLUDED — they are internal fork workspaces, not user projects, and
        counting them inflates the admin dashboard (every user-facing listing
        already filters them, e.g. ``project_repository.py``). ``last_activity``
        = max(updated_at | created_at).
        """
        last_activity = func.max(func.coalesce(Project.updated_at, Project.created_at))
        return (
            select(
                Project.tenant_id.label("tenant_id"),
                func.count(Project.id).label("project_count"),
                last_activity.label("last_activity"),
            )
            .where(Project.is_branch.is_(False))
            .group_by(Project.tenant_id)
            .subquery()
        )

    async def list_tenants(self) -> list[Row[Any]]:
        """Every known tenant = registry rows FULL-JOIN project-bearing tenants,
        each with a per-tenant ``run_count`` + ``total_cost_usd`` rollup.

        A tenant with a registry row but zero projects (named ahead of activity)
        still appears; a tenant with projects but no registry row (created after
        the 0021 backfill, not yet named) still appears with ``name=NULL`` and a
        default ``status='active'``. One round-trip.

        This is the ONE genuinely cross-tenant admin read, so the per-tenant
        project/run/cost rollup comes from the ``admin_tenant_rollup()`` SECURITY
        DEFINER resolver (``src/database/rls.py``) rather than a direct aggregate:
        under RLS enforce the app connects as a non-owner role and a plain
        ``GROUP BY projects.tenant_id`` would be RLS-scoped to 0 rows. The
        resolver bypasses RLS for this trusted super-admin read while the
        data-plane policies stay unchanged. Under the table owner (today / CI) it
        returns the identical rollup, so behaviour is unchanged until the cutover.
        """
        rollup = func.admin_tenant_rollup().table_valued(
            "tenant_id", "project_count", "last_activity", "run_count", "cost_usd"
        )
        tenant_id = func.coalesce(Tenant.id, rollup.c.tenant_id)
        stmt = (
            select(
                tenant_id.label("tenant_id"),
                Tenant.name.label("name"),
                func.coalesce(Tenant.status, "active").label("status"),
                Tenant.partner_licence_id.label("partner_licence_id"),
                func.coalesce(rollup.c.project_count, 0).label("project_count"),
                rollup.c.last_activity.label("last_activity"),
                func.coalesce(rollup.c.run_count, 0).label("run_count"),
                func.coalesce(rollup.c.cost_usd, 0).label("total_cost_usd"),
            )
            .select_from(Tenant)
            .outerjoin(rollup, Tenant.id == rollup.c.tenant_id, full=True)
            .order_by(rollup.c.last_activity.desc().nulls_last())
        )
        return list((await self.session.execute(stmt)).all())

    _RUN_VIEW_COLS = (
        "run_id",
        "tenant_id",
        "project_id",
        "project_title",
        "deliverable_id",
        "blueprint_id",
        "status",
        "current_stage",
        "started_at",
        "created_at",
        "updated_at",
        "paused_at",
        "pause_reason",
    )

    async def list_active_runs(self) -> list[Row[Any]]:
        """Every pending/running run across ALL tenants (operator run pane).

        Cross-tenant super-admin read via the ``admin_active_runs()`` SECURITY
        DEFINER resolver (``src/database/rls.py``) — a plain ``SELECT FROM runs``
        would be RLS-scoped to a single tenant under enforce. Ordered oldest-first
        by start so the longest-running work surfaces at the top.
        """
        runs = func.admin_active_runs().table_valued(*self._RUN_VIEW_COLS)
        stmt = select(runs).order_by(runs.c.started_at.asc().nulls_last(), runs.c.created_at.asc())
        return list((await self.session.execute(stmt)).all())

    async def list_recent_runs(self, limit: int) -> list[Row[Any]]:
        """The most-recently-updated terminal runs across ALL tenants.

        Cross-tenant super-admin read via ``admin_recent_runs(limit)`` (SECURITY
        DEFINER). The function already orders by ``updated_at DESC LIMIT``.
        """
        runs = func.admin_recent_runs(limit).table_valued(*self._RUN_VIEW_COLS)
        return list((await self.session.execute(select(runs))).all())

    async def get_tenant(self, tenant_id: UUID) -> Tenant | None:
        """The registry row for one tenant, or None if only known via projects.

        Used to snapshot the *before* state of a tenant update for the audit log
        (the project-aggregate-joined ``get_tenant_summary`` assumes a row
        exists, so it cannot serve the project-only / first-PATCH case)."""
        return (
            await self.session.execute(select(Tenant).where(Tenant.id == tenant_id))
        ).scalar_one_or_none()

    async def tenant_licence_ref(self, tenant_id: UUID) -> UUID | None:
        """The tenant's entitling w2v licence (``licence_ref``), None when it has
        no registry row or no ref yet. Bills the tenant-API-key lane (P9-6)."""
        return (
            await self.session.execute(select(Tenant.licence_ref).where(Tenant.id == tenant_id))
        ).scalar_one_or_none()

    async def get_tenant_summary(self, tenant_id: UUID) -> Row[Any]:
        """One tenant's list-shaped summary. Assumes a registry row exists
        (callers upsert first), LEFT-joining the project aggregate for counts."""
        # Per-tenant read: bind the GUC so RLS admits exactly this tenant's
        # projects under enforce (no-op until then). See list_tenants for the
        # cross-tenant case.
        await bind_session_tenant(self.session, tenant_id)
        agg = self._project_aggregate()
        stmt = (
            select(
                Tenant.id.label("tenant_id"),
                Tenant.name.label("name"),
                Tenant.status.label("status"),
                Tenant.partner_licence_id.label("partner_licence_id"),
                func.coalesce(agg.c.project_count, 0).label("project_count"),
                agg.c.last_activity.label("last_activity"),
            )
            .select_from(Tenant)
            .outerjoin(agg, Tenant.id == agg.c.tenant_id)
            .where(Tenant.id == tenant_id)
        )
        return (await self.session.execute(stmt)).one()

    async def tenant_switchable(self, tenant_id: UUID) -> bool:
        """Can a super-admin scope-switch INTO this tenant?

        Fail-closed: a tenant must be *known* (registry row OR ≥1 project) and
        not registry-``suspended``. A registry row is authoritative — a suspended
        tenant is blocked even if it has projects.
        """
        status = (
            await self.session.execute(select(Tenant.status).where(Tenant.id == tenant_id))
        ).scalar_one_or_none()
        if status is not None:
            return status not in OPERATOR_HELD_TENANT_STATUSES
        # No registry row → known only via existing project data. Bind the GUC so
        # the project-existence check admits this tenant's rows under RLS enforce
        # (no-op until then); the registry read above is on the un-RLS'd table.
        await bind_session_tenant(self.session, tenant_id)
        has_project = (
            await self.session.execute(
                select(Project.id).where(Project.tenant_id == tenant_id).limit(1)
            )
        ).first()
        return has_project is not None

    async def upsert_tenant(
        self,
        tenant_id: UUID,
        *,
        name: str | None = None,
        status: str | None = None,
        created_via: TenantCreatedVia | None = None,
    ) -> Row[Any]:
        """Create or update a tenant's name/status annotation (super-admin).

        ``None`` means "leave unchanged" (there is no use case for un-naming a
        tenant). Always bumps ``updated_at``. Returns the merged list-shaped
        summary for the API response. ``created_via`` applies only when this
        call INSERTS the row (column default ``super_admin``); an existing
        row's provenance is never rewritten.
        """
        values: dict[str, Any] = {"id": tenant_id}
        set_: dict[str, Any] = {"updated_at": func.now()}
        if created_via is not None:
            # Provenance is fixed at creation: insert-only, never in ``set_``.
            values["created_via"] = created_via
        if name is not None:
            values["name"] = name
            set_["name"] = name
        if status is not None:
            values["status"] = status
            set_["status"] = status
        stmt = (
            pg_insert(Tenant)
            .values(**values)
            .on_conflict_do_update(index_elements=[Tenant.id], set_=set_)
        )
        await self.session.execute(stmt)
        return await self.get_tenant_summary(tenant_id)

    async def ensure_tenant(
        self,
        tenant_id: UUID,
        *,
        created_via: TenantCreatedVia,
        name: str | None = None,
        licence_ref: UUID | None = None,
        partner_licence_id: UUID | None = None,
    ) -> bool:
        """Insert the registry row if absent; True iff THIS call created it.

        Idempotent create for the P2 tenant-creation paths (super-admin
        provision, partner ``POST /api/tenants``, backfill). An existing row is
        left untouched — its name/status/provenance belong to whoever created
        or last edited it. No commit (caller's session).
        """
        result = await self.session.execute(
            pg_insert(Tenant)
            .values(
                id=tenant_id,
                created_via=created_via,
                name=name,
                licence_ref=licence_ref,
                licence_checked_at=func.now() if licence_ref is not None else None,
                partner_licence_id=partner_licence_id,
            )
            .on_conflict_do_nothing(index_elements=[Tenant.id])
            .returning(Tenant.id)
        )
        return result.first() is not None

    async def record_licence_check(
        self,
        tenant_id: UUID,
        *,
        entitled: bool,
        licence_ref: UUID | None = None,
        checked_at: datetime.datetime | None = None,
    ) -> bool:
        """Record a licence check outcome on an existing tenant row.

        ``entitled`` moves ``active`` ↔ ``licence_lapsed``; a ``suspended``
        tenant stays suspended (an operator decision a licence recovery must
        not undo). ``licence_ref`` is updated only when given (an unknown
        licence keeps the last known one). Always stamps
        ``licence_checked_at``. False when the tenant has no registry row.
        A "licences service unavailable" outcome must NOT call this — the
        caller leaves the row unchanged.
        """
        new_status = "active" if entitled else "licence_lapsed"
        result = await self.session.execute(
            update(Tenant)
            .where(Tenant.id == tenant_id)
            .values(
                status=case(
                    (Tenant.status.in_(OPERATOR_HELD_TENANT_STATUSES), Tenant.status),
                    else_=new_status,
                ),
                licence_ref=func.coalesce(
                    literal(licence_ref, type_=Tenant.licence_ref.type), Tenant.licence_ref
                ),
                licence_checked_at=checked_at if checked_at is not None else func.now(),
                updated_at=func.now(),
            )
            .returning(Tenant.id)
        )
        return result.first() is not None

    async def tenant_overview(self, tenant_id: UUID) -> dict[str, Any]:
        """Aggregate for ONE tenant: project_count, run_count, total cost_usd.

        Two statements, not one: joining projects → deliverables → runs →
        run_usage in a single query fans out project rows by run/usage
        cardinality and would inflate ``project_count``. So count projects
        standalone, then COUNT(DISTINCT run.id) + SUM(cost) over the usage
        outer-join (a run with no usage rows still counts; cost coalesces to 0).
        """
        # Per-tenant read: bind the GUC so RLS scopes the aggregates to this
        # tenant under enforce (no-op until then). The WHERE tenant_id == agrees.
        await bind_session_tenant(self.session, tenant_id)
        project_count = (
            await self.session.execute(
                select(func.count(Project.id)).where(
                    Project.tenant_id == tenant_id, Project.is_branch.is_(False)
                )
            )
        ).scalar_one()

        run_row = (
            await self.session.execute(
                select(
                    func.count(func.distinct(Run.id)),
                    func.coalesce(func.sum(RunUsage.cost_usd), 0),
                )
                .select_from(Run)
                .join(Deliverable, Run.deliverable_id == Deliverable.id)
                .join(Project, Deliverable.project_id == Project.id)
                .outerjoin(RunUsage, RunUsage.run_id == Run.id)
                .where(Project.tenant_id == tenant_id)
            )
        ).one()

        return {
            "project_count": project_count,
            "run_count": run_row[0],
            # cost_usd is Numeric → Decimal; surface a float for the schema
            # field, mirroring RunUsageRepository.total().
            "total_cost_usd": float(run_row[1]),
        }

    async def usage_detail(
        self,
        tenant_id: UUID,
        *,
        bucket: str = "day",
        window_days: int | None = None,
    ) -> dict[str, Any]:
        """Multi-dimensional cost/token breakdown for ONE tenant.

        The drill-down behind ``tenant_overview``: totals plus the same figures
        grouped by project type, by usage stage, and by model/provider. Reuses
        ``tenant_overview``'s discipline — run counts use ``COUNT(DISTINCT
        run.id)`` over a Run→Deliverable→Project chain with an OUTER join to
        ``run_usage`` so a run with no usage rows still counts (its cost
        coalesces to 0). The ``by_stage`` / ``by_model`` breakdowns INNER-join
        ``run_usage`` (a usage-less run has no stage/model to attribute), so they
        sum to the totals' cost but exclude those zero-cost runs.

        ``window_days`` scopes the WHOLE response (totals + every by-* breakdown
        + the ``time_series``) to ``run_usage`` rows created in the last N days —
        a metered-usage lens. When windowed, usage-less runs (e.g. ``web_design``
        records no ``run_usage`` rows) drop out, exactly as they are already
        absent from the chart; all-time (``window_days`` omitted) counts them.
        ``bucket`` (day/week/month) affects ONLY the ``time_series`` granularity.

        ``by_user`` / ``by_slot`` (P6 ORCH-3) group the same windowed usage rows on
        the 0083 attribution columns; every row lands in exactly one group of each
        (unattributed rows included), so each list sums to the totals' cost/tokens.
        """
        # Per-tenant read: bind the GUC so RLS scopes every aggregate below to
        # this tenant under enforce (no-op until then). The WHERE tenant_id ==
        # predicates already agree, so binding changes nothing under the owner.
        await bind_session_tenant(self.session, tenant_id)
        # A single window predicate on run_usage.created_at, threaded through
        # every usage aggregate. Empty when all-time → SQLAlchemy .where() no-op,
        # so the all-time path is byte-identical to before.
        window_clause = (
            [RunUsage.created_at >= func.now() - func.make_interval(0, 0, 0, window_days)]
            if window_days is not None
            else []
        )

        if window_days is None:
            project_count = (
                await self.session.execute(
                    select(func.count(Project.id)).where(
                        Project.tenant_id == tenant_id, Project.is_branch.is_(False)
                    )
                )
            ).scalar_one()
        else:
            # Windowed: only projects with in-window usage. COUNT(DISTINCT) over
            # the usage chain, so run/usage fan-out can't inflate it.
            project_count = (
                await self.session.execute(
                    select(func.count(func.distinct(Project.id)))
                    .select_from(Project)
                    .join(Deliverable, Deliverable.project_id == Project.id)
                    .join(Run, Run.deliverable_id == Deliverable.id)
                    .join(RunUsage, RunUsage.run_id == Run.id)
                    .where(Project.tenant_id == tenant_id, *window_clause)
                )
            ).scalar_one()

        totals_row = (
            await self.session.execute(
                select(
                    func.count(func.distinct(Run.id)),
                    func.coalesce(func.sum(RunUsage.cost_usd), 0),
                    func.coalesce(func.sum(RunUsage.input_tokens), 0),
                    func.coalesce(func.sum(RunUsage.output_tokens), 0),
                )
                .select_from(Run)
                .join(Deliverable, Run.deliverable_id == Deliverable.id)
                .join(Project, Deliverable.project_id == Project.id)
                .outerjoin(RunUsage, RunUsage.run_id == Run.id)
                .where(Project.tenant_id == tenant_id, *window_clause)
            )
        ).one()

        # By project type: start from Project (OUTER join down) so a type with
        # projects but zero runs still appears (run_count 0, cost 0). Grouping
        # by type + COUNT(DISTINCT run) means project fan-out can't inflate it.
        type_rows = (
            await self.session.execute(
                select(
                    Project.type,
                    func.count(func.distinct(Run.id)),
                    func.coalesce(func.sum(RunUsage.cost_usd), 0),
                    func.coalesce(func.sum(RunUsage.input_tokens), 0),
                    func.coalesce(func.sum(RunUsage.output_tokens), 0),
                )
                .select_from(Project)
                .outerjoin(Deliverable, Deliverable.project_id == Project.id)
                .outerjoin(Run, Run.deliverable_id == Deliverable.id)
                .outerjoin(RunUsage, RunUsage.run_id == Run.id)
                .where(
                    Project.tenant_id == tenant_id,
                    Project.is_branch.is_(False),
                    *window_clause,
                )
                .group_by(Project.type)
                .order_by(func.coalesce(func.sum(RunUsage.cost_usd), 0).desc())
            )
        ).all()

        # By stage / by model / time-series: INNER-join run_usage (a usage row IS
        # one call), so call_count = number of usage rows, and usage-less runs
        # drop out. The window predicate lives HERE so stage/model/series all
        # inherit it from one place.
        usage_base = (
            select(RunUsage)
            .join(Run, RunUsage.run_id == Run.id)
            .join(Deliverable, Run.deliverable_id == Deliverable.id)
            .join(Project, Deliverable.project_id == Project.id)
            .where(Project.tenant_id == tenant_id, *window_clause)
        ).subquery()

        stage_rows = (
            await self.session.execute(
                select(
                    usage_base.c.stage_name,
                    func.count(usage_base.c.id),
                    func.coalesce(func.sum(usage_base.c.cost_usd), 0),
                    func.coalesce(func.sum(usage_base.c.input_tokens), 0),
                    func.coalesce(func.sum(usage_base.c.output_tokens), 0),
                )
                .group_by(usage_base.c.stage_name)
                .order_by(func.coalesce(func.sum(usage_base.c.cost_usd), 0).desc())
            )
        ).all()

        model_rows = (
            await self.session.execute(
                select(
                    usage_base.c.model,
                    usage_base.c.provider,
                    func.count(usage_base.c.id),
                    func.coalesce(func.sum(usage_base.c.cost_usd), 0),
                    func.coalesce(func.sum(usage_base.c.input_tokens), 0),
                    func.coalesce(func.sum(usage_base.c.output_tokens), 0),
                )
                .group_by(usage_base.c.model, usage_base.c.provider)
                .order_by(func.coalesce(func.sum(usage_base.c.cost_usd), 0).desc())
            )
        ).all()

        # By user (P6 ORCH-3): grouped on run_usage.subject_id, joined to the tenant's
        # membership rows for an email WITHOUT a status filter, so a removed member's
        # usage still shows (as removed). A NULL subject (a pre-0083 row the backfill
        # could not attribute) is its own "unattributed" group — never dropped, so
        # the groups sum to the totals.
        user_rows = (
            await self.session.execute(
                select(
                    usage_base.c.subject_id,
                    TenantMember.email,
                    TenantMember.status,
                    func.count(usage_base.c.id),
                    func.coalesce(func.sum(usage_base.c.cost_usd), 0),
                    func.coalesce(func.sum(usage_base.c.input_tokens), 0),
                    func.coalesce(func.sum(usage_base.c.output_tokens), 0),
                )
                .select_from(usage_base)
                .outerjoin(
                    TenantMember,
                    (TenantMember.subject_id == usage_base.c.subject_id)
                    & (TenantMember.tenant_id == tenant_id),
                )
                .group_by(usage_base.c.subject_id, TenantMember.email, TenantMember.status)
                .order_by(func.coalesce(func.sum(usage_base.c.cost_usd), 0).desc())
            )
        ).all()

        # By slot: an in-process row carries its tenant slot; a billing-mirrored row
        # (not written in-process) ran on the platform route; a row with neither is
        # unattributed. Funding is grouped separately (NULL → "unattributed" in the
        # fold): the route does not imply who paid. Units are summed per unit_kind.
        # A slot-less row funded ``local_gpu`` is the platform vLLM (the orchestrator's
        # own vLLM-direct call, or a service-reported one) → the platform route.
        slot_label = case(
            (usage_base.c.slot.is_not(None), usage_base.c.slot),
            (usage_base.c.funding == FUNDING_LOCAL_GPU, literal(SLOT_PLATFORM)),
            (
                usage_base.c.metadata["source"].astext.is_distinct_from(IN_PROCESS_USAGE_SOURCE),
                literal(SLOT_PLATFORM),
            ),
            else_=literal(UNATTRIBUTED),
        )
        slot_rows = (
            await self.session.execute(
                select(
                    slot_label,
                    usage_base.c.funding,
                    usage_base.c.unit_kind,
                    func.count(usage_base.c.id),
                    func.coalesce(func.sum(usage_base.c.cost_usd), 0),
                    func.coalesce(func.sum(usage_base.c.input_tokens), 0),
                    func.coalesce(func.sum(usage_base.c.output_tokens), 0),
                    func.coalesce(func.sum(usage_base.c.units), 0),
                ).group_by(slot_label, usage_base.c.funding, usage_base.c.unit_kind)
            )
        ).all()

        # Cost/token over time: day/week/month buckets, gap-filled (zero buckets
        # appear between range ends), optionally windowed to the last N days. The
        # aggregation and the gap-fill axis both use Postgres date_trunc(...,'UTC')
        # + make_interval so their bucket boundaries are identical by construction
        # (no Python calendar re-derivation). UTC-pinned (3-arg date_trunc) because
        # the engine sets no session TimeZone — keep it deterministic.
        # usage_base is already windowed (window_clause above), so the agg needs
        # no separate window predicate — its buckets are in-window by construction.
        trunc = func.date_trunc(bucket, usage_base.c.created_at, "UTC")
        agg = (
            select(
                trunc.label("bucket"),
                func.coalesce(func.sum(usage_base.c.cost_usd), 0).label("cost_usd"),
                func.coalesce(func.sum(usage_base.c.input_tokens), 0).label("input_tokens"),
                func.coalesce(func.sum(usage_base.c.output_tokens), 0).label("output_tokens"),
            ).group_by(trunc)
        ).cte("usage_agg")

        if window_days is not None:
            # Windowed: the axis runs to today even if the last usage was earlier.
            range_start = func.date_trunc(
                bucket, func.now() - func.make_interval(0, 0, 0, window_days), "UTC"
            )
            range_end = func.date_trunc(bucket, func.now(), "UTC")
        else:
            # All-time: data-bounded. Clamp the floor to now-730d so all-time+day
            # can't explode the row count for a very long-lived tenant.
            floor = func.date_trunc(bucket, func.now() - func.make_interval(0, 0, 0, 730), "UTC")
            range_start = select(func.greatest(func.min(agg.c.bucket), floor)).scalar_subquery()
            range_end = select(func.max(agg.c.bucket)).scalar_subquery()

        # render_derived() emits the column list — `AS gen(bucket)` — so the
        # generated column is addressable; without it SQLAlchemy aliases the
        # set-returning function but names no column (anon_1.bucket → undefined).
        gen = (
            func.generate_series(range_start, range_end, _BUCKET_STEP[bucket])
            .table_valued("bucket")
            .render_derived()
        )
        series_rows = (
            await self.session.execute(
                select(
                    gen.c.bucket,
                    func.coalesce(agg.c.cost_usd, 0),
                    func.coalesce(agg.c.input_tokens, 0),
                    func.coalesce(agg.c.output_tokens, 0),
                )
                .select_from(gen.outerjoin(agg, agg.c.bucket == gen.c.bucket))
                .order_by(gen.c.bucket)
            )
        ).all()

        return {
            "totals": {
                "run_count": totals_row[0],
                "project_count": project_count,
                "input_tokens": int(totals_row[2]),
                "output_tokens": int(totals_row[3]),
                "cost_usd": float(totals_row[1]),
            },
            "by_type": [
                {
                    "type": row[0],
                    "run_count": row[1],
                    "cost_usd": float(row[2]),
                    "input_tokens": int(row[3]),
                    "output_tokens": int(row[4]),
                }
                for row in type_rows
            ],
            "by_stage": [
                {
                    "stage_name": row[0],
                    "call_count": row[1],
                    "cost_usd": float(row[2]),
                    "input_tokens": int(row[3]),
                    "output_tokens": int(row[4]),
                }
                for row in stage_rows
            ],
            "by_model": [
                {
                    "model": row[0],
                    "provider": row[1],
                    "call_count": row[2],
                    "cost_usd": float(row[3]),
                    "input_tokens": int(row[4]),
                    "output_tokens": int(row[5]),
                }
                for row in model_rows
            ],
            "by_user": [
                {
                    "subject_id": row[0],
                    "email": row[1],
                    "member_status": row[2],
                    "label": (UNATTRIBUTED if row[0] is None else (row[1] or str(row[0]))),
                    "call_count": row[3],
                    "cost_usd": float(row[4]),
                    "input_tokens": int(row[5]),
                    "output_tokens": int(row[6]),
                }
                for row in user_rows
            ],
            "by_slot": _fold_slot_rows(slot_rows),
            "time_series": [
                {
                    "bucket": row[0].date().isoformat(),
                    "cost_usd": float(row[1]),
                    "input_tokens": int(row[2]),
                    "output_tokens": int(row[3]),
                }
                for row in series_rows
            ],
        }
