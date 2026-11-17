"""VerbatimItemRepository — tenant-scoped CRUD with auto-increment position."""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid7

from sqlalchemy import asc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import Project, VerbatimItem


class VerbatimItemRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def _scoped_project(self, project_id: UUID, tenant_id: UUID) -> Project | None:
        return (
            await self.session.execute(
                select(Project).where(Project.id == project_id, Project.tenant_id == tenant_id)
            )
        ).scalar_one_or_none()

    async def create(
        self,
        *,
        project_id: UUID,
        text: str,
        position: int | None,
        kind: str = "tts",
        hint: str | None = None,
        scope: str = "pipeline",
        scope_target: str | None = None,
    ) -> VerbatimItem:
        if position is None:
            max_pos = (
                await self.session.execute(
                    select(func.coalesce(func.max(VerbatimItem.position), -1)).where(
                        VerbatimItem.project_id == project_id
                    )
                )
            ).scalar_one()
            position = int(max_pos) + 1
        item = VerbatimItem(
            id=uuid7(),
            project_id=project_id,
            kind=kind,
            text=text,
            hint=hint,
            scope=scope,
            scope_target=scope_target,
            position=position,
        )
        self.session.add(item)
        await self.session.flush()
        return item

    async def list_for_project(self, project_id: UUID, *, tenant_id: UUID) -> list[VerbatimItem]:
        if (await self._scoped_project(project_id, tenant_id)) is None:
            return []
        stmt = (
            select(VerbatimItem)
            .where(VerbatimItem.project_id == project_id)
            .order_by(asc(VerbatimItem.position), asc(VerbatimItem.created_at))
        )
        return list((await self.session.execute(stmt)).scalars().all())

    async def get(self, item_id: UUID, *, tenant_id: UUID) -> VerbatimItem | None:
        stmt = (
            select(VerbatimItem)
            .join(Project, VerbatimItem.project_id == Project.id)
            .where(VerbatimItem.id == item_id, Project.tenant_id == tenant_id)
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def update(
        self,
        item_id: UUID,
        *,
        tenant_id: UUID,
        fields: dict[str, Any],
    ) -> VerbatimItem | None:
        item = await self.get(item_id, tenant_id=tenant_id)
        if item is None:
            return None
        for k, v in fields.items():
            setattr(item, k, v)
        await self.session.flush()
        # onupdate=func.now() expires updated_at; refresh repopulates so the
        # response serializer doesn't trigger a sync lazy-load.
        await self.session.refresh(item)
        return item

    async def delete(self, item_id: UUID, *, tenant_id: UUID) -> bool:
        item = await self.get(item_id, tenant_id=tenant_id)
        if item is None:
            return False
        await self.session.delete(item)
        await self.session.flush()
        return True
