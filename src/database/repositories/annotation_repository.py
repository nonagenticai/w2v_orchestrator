"""AnnotationRepository — CRUD + bulk-apply for the annotations table."""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid7

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import Annotation


class AnnotationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self, *, stage_output_id: UUID, text: str, element_id: str | None
    ) -> Annotation:
        a = Annotation(
            id=uuid7(),
            stage_output_id=stage_output_id,
            element_id=element_id,
            text=text,
            status="open",
        )
        self.session.add(a)
        await self.session.flush()
        return a

    async def get(self, annotation_id: UUID) -> Annotation | None:
        return (
            await self.session.execute(select(Annotation).where(Annotation.id == annotation_id))
        ).scalar_one_or_none()

    async def list_for_stage_output(
        self, stage_output_id: UUID, *, status: str | None
    ) -> list[Annotation]:
        stmt = select(Annotation).where(Annotation.stage_output_id == stage_output_id)
        if status:
            stmt = stmt.where(Annotation.status == status)
        stmt = stmt.order_by(Annotation.created_at)
        return list((await self.session.execute(stmt)).scalars().all())

    async def update(self, annotation_id: UUID, *, fields: dict[str, Any]) -> Annotation | None:
        a = await self.get(annotation_id)
        if a is None:
            return None
        for k, v in fields.items():
            setattr(a, k, v)
        await self.session.flush()
        # onupdate=func.now() expires updated_at; refresh repopulates so the
        # response serializer doesn't trigger a sync lazy-load.
        await self.session.refresh(a)
        return a

    async def delete(self, annotation_id: UUID) -> bool:
        a = await self.get(annotation_id)
        if a is None:
            return False
        await self.session.delete(a)
        await self.session.flush()
        return True

    async def bulk_apply(
        self,
        *,
        stage_output_ids: list[UUID],
        applied_in_run_id: UUID,
        element_id: str | None = None,
    ) -> int:
        """Mark every open annotation under these stage_output_ids as applied,
        with applied_in_run_id set to the new run. Returns the number flipped.

        When ``element_id`` is given, only annotations pinned to that element are
        flipped — a surgical single-clip regenerate must not touch sibling clips'
        still-open feedback on the same stage_output. When ``None`` (the default),
        every open annotation on the stage_outputs flips (reorder + whole-run
        regenerate, which re-run the whole spine).

        Caller MUST be inside the same transaction that creates the new run row,
        per spec §4 invariant 4 + §8.6.
        """
        if not stage_output_ids:
            return 0
        stmt = (
            update(Annotation)
            .where(
                Annotation.stage_output_id.in_(stage_output_ids),
                Annotation.status == "open",
            )
            .values(
                status="applied",
                applied_in_run_id=applied_in_run_id,
                updated_at=func.now(),
            )
        )
        if element_id is not None:
            stmt = stmt.where(Annotation.element_id == element_id)
        result = await self.session.execute(stmt)
        return int(getattr(result, "rowcount", 0) or 0)

    async def list_open_by_stage_for_run(self, run_id: UUID) -> dict[str, list[Annotation]]:
        """For a run, group its open annotations by stage_name."""
        from src.database.models import StageOutput

        stmt = (
            select(Annotation, StageOutput.stage_name)
            .join(StageOutput, Annotation.stage_output_id == StageOutput.id)
            .where(StageOutput.run_id == run_id, Annotation.status == "open")
        )
        rows = (await self.session.execute(stmt)).all()
        out: dict[str, list[Annotation]] = {}
        for ann, stage_name in rows:
            out.setdefault(str(stage_name), []).append(ann)
        return out
