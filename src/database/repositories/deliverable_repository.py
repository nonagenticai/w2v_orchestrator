"""Deliverable repository — tenant-scoped CRUD via project join."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas import DeliverableStatus
from src.database.models import Deliverable, Project, Run


def derive_deliverable_status(
    run_status: str | None, pause_reason: str | None = None
) -> DeliverableStatus:
    """Map a deliverable's LATEST run status to its derived deliverable status.

    ``deliverables.status`` is never persisted (the column stays at its 'idle'
    default); like ``project.status`` it is derived on read from the latest run.
    A deliverable with no run is ``idle``. A run paused for manual review surfaces
    as ``awaiting_review`` (the one place that enum value is used); pending/running
    → ``running``; completed → ``completed``; failed/cancelled → ``failed``.

    Only a CHECKPOINT pause (``pause_reason='checkpoint'``, 0102) is waiting for a
    review. A pause someone pressed (``manual``), one 0102 could not classify
    (``unknown``) or one with no reason at all surfaces as ``paused`` — never
    "awaiting review" on a guess.
    """
    if run_status == "paused" and pause_reason != "checkpoint":
        return "paused"
    if run_status in ("pending", "running"):
        return "running"
    if run_status == "paused":
        return "awaiting_review"
    if run_status == "completed":
        return "completed"
    if run_status in ("failed", "cancelled"):
        return "failed"
    return "idle"


class DeliverableRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def derived_statuses(
        self, deliverable_ids: Sequence[UUID]
    ) -> dict[UUID, DeliverableStatus]:
        """Derived status per deliverable id, from each one's latest run.

        One windowed query (no N+1), mirroring the latest-run lookup
        :class:`~src.database.repositories.project_repository.ProjectRepository`
        already uses for ``project.status``. Run-less / missing ids default to
        ``idle``.
        """
        result: dict[UUID, DeliverableStatus] = {did: "idle" for did in deliverable_ids}
        if not deliverable_ids:
            return result
        latest = (
            select(
                Run.deliverable_id.label("did"),
                Run.status.label("status"),
                Run.pause_reason.label("pause_reason"),
                func.row_number()
                .over(
                    partition_by=Run.deliverable_id,
                    order_by=(Run.created_at.desc(), Run.id.desc()),
                )
                .label("rn"),
            )
            .where(Run.deliverable_id.in_(deliverable_ids))
            .subquery()
        )
        rows = await self.session.execute(
            select(latest.c.did, latest.c.status, latest.c.pause_reason).where(latest.c.rn == 1)
        )
        for did, run_status, pause_reason in rows.all():
            result[did] = derive_deliverable_status(run_status, pause_reason)
        return result

    async def _scoped(self, deliverable_id: UUID, tenant_id: UUID) -> Deliverable | None:
        stmt = (
            select(Deliverable)
            .join(Project, Deliverable.project_id == Project.id)
            .where(Deliverable.id == deliverable_id, Project.tenant_id == tenant_id)
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def get(self, deliverable_id: UUID, *, tenant_id: UUID) -> Deliverable | None:
        return await self._scoped(deliverable_id, tenant_id)

    async def update(
        self,
        deliverable_id: UUID,
        *,
        tenant_id: UUID,
        fields: dict[str, Any],
    ) -> Deliverable | None:
        d = await self._scoped(deliverable_id, tenant_id)
        if d is None:
            return None
        for k, v in fields.items():
            setattr(d, k, v)
        await self.session.flush()
        await self.session.refresh(d)
        return d

    async def delete(self, deliverable_id: UUID, *, tenant_id: UUID) -> bool:
        d = await self._scoped(deliverable_id, tenant_id)
        if d is None:
            return False
        await self.session.delete(d)
        await self.session.flush()
        return True
