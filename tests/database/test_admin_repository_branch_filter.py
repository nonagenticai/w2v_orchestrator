"""Admin aggregates must EXCLUDE hidden storyboard branch projects.

A branch is a real but hidden ``projects`` row (``is_branch = true``, migration
0052) used as an internal fork workspace — never a user project. Every
user-facing listing filters it (``project_repository``); the admin dashboard's
per-tenant counts must too, or a tenant's project total inflates by its branch
count. This test seeds N real projects + M branches for one tenant and asserts
every admin project-count/aggregate path reports N, not N+M:

  * ``_project_aggregate`` (backs ``get_tenant_summary``)
  * ``list_tenants`` (via the ``admin_tenant_rollup()`` SECURITY DEFINER resolver)
  * ``tenant_overview``
  * ``usage_detail`` all-time totals + the by-type breakdown

Real postgres (``db_session``); no mocks — the branch filter lives in SQL.
"""

from __future__ import annotations

from decimal import Decimal
from uuid import UUID, uuid4, uuid7

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import Deliverable, Project, Run, RunUsage
from src.database.repositories.admin_repository import AdminRepository

pytestmark = pytest.mark.asyncio

REAL_TYPE = "promo"
# Branch children copy a real, CHECK-valid ``type`` from their root (``is_branch`` is
# the only distinguisher). We give branches a DISTINCT valid type so the by-type
# breakdown assertion is clean: ``course`` must be absent because its only rows are
# branches. (A real ``storyboard_branch_wip`` type does not exist — it violates
# ``ck_projects_type``.)
BRANCH_TYPE = "course"
N_REAL = 2
M_BRANCH = 3


async def _mk_project(
    session: AsyncSession, tenant_id: UUID, *, ptype: str, is_branch: bool
) -> Project:
    proj = Project(
        id=uuid7(),
        tenant_id=tenant_id,
        owner_subject_id=uuid7(),
        visibility="tenant",
        type=ptype,
        title="branch" if is_branch else "real",
        is_branch=is_branch,
    )
    session.add(proj)
    await session.flush()
    return proj


async def _add_run_with_usage(session: AsyncSession, project: Project) -> None:
    deliv = Deliverable(id=uuid7(), project_id=project.id, name="d", format="16x9")
    session.add(deliv)
    await session.flush()
    run = Run(id=uuid7(), deliverable_id=deliv.id, blueprint_id="promo.v1", status="completed")
    session.add(run)
    await session.flush()
    session.add(
        RunUsage(
            id=uuid7(),
            run_id=run.id,
            stage_name="script",
            model="deepseek-v4-flash",
            provider="deepseek",
            input_tokens=100,
            output_tokens=50,
            cost_usd=Decimal("0.01"),
        )
    )
    await session.flush()


async def _seed(session: AsyncSession) -> UUID:
    tenant_id = uuid4()
    for _ in range(N_REAL):
        proj = await _mk_project(session, tenant_id, ptype=REAL_TYPE, is_branch=False)
        await _add_run_with_usage(session, proj)  # gives by_type a real REAL_TYPE row
    for _ in range(M_BRANCH):
        # Branches carry a distinct type and NO deliverables/runs — exactly the
        # zero-deliverable row that survives usage_detail's by-type OUTER joins.
        await _mk_project(session, tenant_id, ptype=BRANCH_TYPE, is_branch=True)
    await session.commit()
    return tenant_id


async def test_project_aggregate_excludes_branches(db_session: AsyncSession) -> None:
    tenant_id = await _seed(db_session)
    repo = AdminRepository(db_session)

    agg = repo._project_aggregate()
    count = (
        await db_session.execute(select(agg.c.project_count).where(agg.c.tenant_id == tenant_id))
    ).scalar_one()
    assert count == N_REAL  # not N_REAL + M_BRANCH


async def test_list_tenants_excludes_branches(db_session: AsyncSession) -> None:
    tenant_id = await _seed(db_session)
    repo = AdminRepository(db_session)

    rows = await repo.list_tenants()
    row = next(r for r in rows if r.tenant_id == tenant_id)
    assert row.project_count == N_REAL  # admin_tenant_rollup() pa-subquery filter


async def test_tenant_overview_excludes_branches(db_session: AsyncSession) -> None:
    tenant_id = await _seed(db_session)
    repo = AdminRepository(db_session)

    overview = await repo.tenant_overview(tenant_id)
    assert overview["project_count"] == N_REAL
    assert overview["run_count"] == N_REAL  # only real projects have runs


async def test_usage_detail_excludes_branches(db_session: AsyncSession) -> None:
    tenant_id = await _seed(db_session)
    repo = AdminRepository(db_session)

    detail = await repo.usage_detail(tenant_id)
    assert detail["totals"]["project_count"] == N_REAL

    by_type = {row["type"]: row for row in detail["by_type"]}
    assert BRANCH_TYPE not in by_type  # branch type must not appear in the chart
    assert REAL_TYPE in by_type
    assert by_type[REAL_TYPE]["run_count"] == N_REAL
