"""Repository for ``stage_step_results`` — the step ledger's rows.

Thin on purpose: three statements, no policy. The reuse rule (same ``input_hash`` ⇒
reuse), the size cap and the never-fail-the-stage guarantee all live one layer up
in ``src/services/stages/shared/step_ledger.py`` — stage code should use
``ctx.step_ledger(...)`` and never reach for this class directly.

Conventions match ``StageOutputRepository``: constructed around a caller-owned
``AsyncSession``, never commits, never binds the tenant. The caller binds the
tenant (``bind_session_tenant`` / ``bind_run_tenant``) BEFORE the first statement
and commits afterwards; under ``TENANT_RLS_MODE=enforce`` an unbound session reads
zero rows here and its INSERT is rejected by the policy's ``WITH CHECK``.
"""

from __future__ import annotations

from uuid import UUID, uuid7

from sqlalchemy import CursorResult, delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import StageStepResult


class StageStepRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, run_id: UUID, stage_name: str, step_key: str) -> StageStepResult | None:
        """The stored row for this sub-step, or ``None`` if it never finished."""
        result = await self.session.execute(
            select(StageStepResult).where(
                StageStepResult.run_id == run_id,
                StageStepResult.stage_name == stage_name,
                StageStepResult.step_key == step_key,
            )
        )
        return result.scalar_one_or_none()

    async def put(
        self,
        run_id: UUID,
        stage_name: str,
        step_key: str,
        input_hash: str,
        result: object,
    ) -> None:
        """Record a finished sub-step; an existing row for the key is REPLACED.

        One ``INSERT … ON CONFLICT (run_id, stage_name, step_key) DO UPDATE``, so a
        re-run with a changed ``input_hash`` overwrites hash and result together and
        two racing writers converge on one row instead of raising. ``created_at``
        keeps the first write; ``updated_at`` is bumped explicitly because a Core
        upsert does not fire the ORM ``onupdate``.

        ``result`` must be JSON-serializable and must not be ``None`` at the top
        level (the column is NOT NULL) — the ledger helper always passes an
        envelope dict.
        """
        stmt = pg_insert(StageStepResult).values(
            id=uuid7(),
            run_id=run_id,
            stage_name=stage_name,
            step_key=step_key,
            input_hash=input_hash,
            result=result,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["run_id", "stage_name", "step_key"],
            set_={
                "input_hash": stmt.excluded.input_hash,
                "result": stmt.excluded.result,
                "updated_at": func.now(),
            },
        )
        await self.session.execute(stmt)

    async def delete_for_stage(self, run_id: UUID, stage_name: str) -> int:
        """Forget every finished sub-step of ONE stage of a run; returns rows deleted.

        For exactly one caller shape: a USER action that sends an already-finished
        stage back to ``pending`` inside the same run (today only the element
        restamp, ``src/api/routes/runs/stages.py``). The user asked for that stage to
        be done again, so a stored paid result must not be replayed into it. Call it
        in the SAME transaction as the row reset so the two cannot disagree.

        NOT for the reconciler, a graceful drain or a resumable timeout: those
        re-enter the same execution and reusing the rows is the entire point.

        RLS applies as to any statement here: under ``enforce`` an unbound or
        foreign-tenant session matches zero rows and deletes nothing.
        """
        result = await self.session.execute(
            delete(StageStepResult).where(
                StageStepResult.run_id == run_id,
                StageStepResult.stage_name == stage_name,
            )
        )
        return result.rowcount if isinstance(result, CursorResult) else 0
