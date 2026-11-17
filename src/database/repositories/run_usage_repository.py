from __future__ import annotations

import datetime
from collections.abc import Sequence
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid7

from sqlalchemy import case, delete, func, or_, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import Deliverable, Project, Run, RunUsage

#: ``run_usage.metadata.source`` of rows the orchestrator writes ITSELF for in-process
#: (tenant BYO) LLM calls — ``src/services/llm/metering.py``. Billing never sees those
#: calls, so the billing mirror below must neither delete these rows nor count them
#: as "this run already has usage".
IN_PROCESS_USAGE_SOURCE = "orchestrator_in_process"

#: ``run_usage.metadata.source`` of rows a DOWNSTREAM service reported for a call it
#: made itself on the platform vLLM (``POST /api/runs/{run_id}/usage``). Those calls
#: never pass WM or billing either, so the mirror must leave them alone too.
SERVICE_REPORTED_USAGE_SOURCE = "service_reported"

#: ``run_usage.metadata.source`` stamped on every row the billing MIRROR writes
#: (``replace_for_runs``). It is what the unique index ``uq_run_usage_mirrored_group``
#: (migration 0105) keys on, so a concurrent duplicate mirror row is impossible.
#: Rows mirrored before 0105 carry no source; they still count as mirrored (the
#: predicate below) and are replaced by the next mirror of their run.
BILLING_MIRROR_USAGE_SOURCE = "billing_mirror"

#: Every source written HERE rather than mirrored from billing. The billing mirror
#: owns exactly the rows whose source is NOT one of these (NULL included).
LOCAL_USAGE_SOURCES: tuple[str, ...] = (IN_PROCESS_USAGE_SOURCE, SERVICE_REPORTED_USAGE_SOURCE)

#: The agreed funding taxonomy (memory: usage-funding-taxonomy-and-byo). ``run_usage``
#: holds one of these or NULL — NULL means "not known" and is reported as unattributed.
FUNDING_LOCAL_GPU = "local_gpu"
FUNDING_PLATFORM_EXTERNAL = "platform_external"
FUNDING_TENANT_BYO = "tenant_byo"
FUNDING_VALUES: frozenset[str] = frozenset(
    {FUNDING_LOCAL_GPU, FUNDING_PLATFORM_EXTERNAL, FUNDING_TENANT_BYO}
)


def known_funding(raw: object) -> str | None:
    """A taxonomy value, else ``None``. Never guesses: billing's per-run cost rows carry
    no funding dimension (a gateway/WM call may be platform-paid OR tenant BYO), so a
    mirrored row gets a funding only if the row itself states a taxonomy value."""
    return raw if isinstance(raw, str) and raw in FUNDING_VALUES else None


def attribution_subject(raw: object) -> UUID | None:
    """``run_usage.subject_id`` from a context/snapshot value; malformed → ``None``
    (an unattributed row is reported as such — it must not fail the write)."""
    if isinstance(raw, UUID):
        return raw
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return UUID(raw)
    except ValueError:
        return None


def mirrored_rows_predicate(source):  # noqa: ANN001, ANN201 - SQLAlchemy column expressions
    """``source`` (a ``metadata->>'source'`` expression) marks a billing-mirrored row:
    NULL or any value outside :data:`LOCAL_USAGE_SOURCES`."""
    return or_(source.is_(None), source.not_in(LOCAL_USAGE_SOURCES))


def _mirrored_rows():  # noqa: ANN202 - SQLAlchemy column expression
    """Predicate: a row the billing mirror owns (anything not written locally)."""
    return mirrored_rows_predicate(RunUsage.metadata_["source"].astext)


# Runs whose cost is final enough to mirror from billing. A still-running run's
# cost is incomplete; only reconcile terminal runs. ``cancelled`` is terminal too:
# every stage that ran before the cancel was billed, and leaving it out meant a
# cancelled run's real spend was never mirrored — it read as $0 forever, in the
# run viewer and in every tenant usage/admin aggregate.
TERMINAL_RUN_STATUSES = ("completed", "failed", "cancelled")

#: Seconds after a run's last state change before its mirrored cost may be marked
#: SETTLED (final). Billing's cost events are async; this is the flush allowance
#: for a run whose work has actually stopped.
USAGE_SETTLE_SECONDS = 30

#: The same for a CANCELLED run. Cancelling propagates to the video compiler only
#: best-effort (``services/compile_cancel.py``): a render the cancel did not reach
#: keeps running — and billing — until the compiler's hard per-job ceiling,
#: ``video_compiler_service`` ``job_work_timeout_seconds`` (3600s), kills it.
#: Settling at 30s froze that later spend out. The orchestrator cannot see the
#: compiler's job list, so the bound is the ceiling itself plus flush margin.
CANCELLED_USAGE_SETTLE_SECONDS = 3600 + 300


def usage_settle_seconds(status: str) -> int:
    """How long after its last change a run in ``status`` must wait to settle."""
    return CANCELLED_USAGE_SETTLE_SECONDS if status == "cancelled" else USAGE_SETTLE_SECONDS


def mirror_lock_key(run_id: UUID) -> str:
    """Advisory-lock name serialising every billing-mirror write for one run."""
    return f"run_usage_mirror:{run_id}"


class RunUsageRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def replace_for_run(
        self,
        *,
        run_id: UUID,
        rows: list[dict[str, Any]],
        created_at: datetime.datetime | None = None,
        mark_reconciled: bool = False,
    ) -> int:
        """Idempotently replace a run's usage rows (delete-then-insert).

        Used by the billing reconciler to mirror billing-priced cost into the
        local ``run_usage`` table the admin/usage reads aggregate over. Each row
        is ``{stage_name, model, provider, input_tokens, output_tokens,
        cost_usd}``. ``created_at`` (the run's timestamp) is stamped on every row
        so the usage time-series buckets the cost when it actually occurred, not
        at reconcile time. Returns the number of rows written.

        ``mark_reconciled`` stamps ``runs.usage_reconciled_at``: the run's cost is
        SETTLED (final) and no read re-queries billing for it again. Both the
        tenant-wide sweep and the run-viewer read set it, and only once the run is
        past its settle window (:func:`usage_settle_seconds`) — a genuinely
        zero-cost run writes no rows, so without this marker it would be re-billed
        on every load. Callers must hold the run's mirror lock
        (:meth:`try_lock_mirror`); the reconciler does.
        """
        return await self.replace_for_runs(
            [(run_id, rows, created_at)], mark_reconciled=mark_reconciled
        )

    async def try_lock_mirror(self, run_ids: Sequence[UUID]) -> set[UUID]:
        """Take each run's mirror lock WITHOUT waiting; return the ones we hold.

        ``replace_for_runs`` is delete-then-insert. Without a lock, two concurrent
        reads of the same finished run interleave their DELETEs and INSERTs and
        each leaves its own copy of the rows — measured: three simultaneous GETs
        of a $0.03 run settled it at $0.09 / $0.09 / $0.06 and froze that. A
        transaction-scoped advisory lock per run (released on commit/rollback).

        ⚠️ It is ``try``, not the blocking ``pg_advisory_xact_lock``, and that is
        the whole point. The blocking form made a CONTENDED caller WAIT — and the
        run page holds no budget for a lock a tenant-wide sweep of 200 runs is
        sitting on: measured 503 after 8s versus <0.2s on main. A run someone else
        is already reconciling is not an error; it is the normal serialisation, and
        the right answer is "skip it this load, serve the stored cost, retry later".

        Callers MUST NOT write a run that is absent from the returned set — and
        MUST commit (or roll back) to release the locks they did get.
        The unique index ``uq_run_usage_mirrored_group`` (migration 0105) is the
        backstop that makes a duplicate impossible even if a caller skips this.
        """
        if not run_ids:
            return set()
        key_to_run = {mirror_lock_key(r): r for r in run_ids}
        # ONE statement for any number of runs (the sweep reconciles up to 200 in
        # constant statements).
        rows = await self.session.execute(
            text(
                "SELECT k, pg_try_advisory_xact_lock(hashtextextended(k, 0)) AS acquired "
                "FROM unnest(CAST(:keys AS text[])) AS k"
            ),
            {"keys": list(key_to_run)},
        )
        return {key_to_run[k] for k, acquired in rows.all() if acquired}

    async def unsettled_among(self, run_ids: Sequence[UUID]) -> set[UUID]:
        """The subset of ``run_ids`` not yet marked settled (``usage_reconciled_at``)."""
        if not run_ids:
            return set()
        rows = await self.session.execute(
            select(Run.id).where(Run.id.in_(list(run_ids)), Run.usage_reconciled_at.is_(None))
        )
        return {r[0] for r in rows.all()}

    async def replace_for_runs(
        self,
        batch: list[tuple[UUID, list[dict[str, Any]], datetime.datetime | None]],
        *,
        mark_reconciled: bool = False,
    ) -> int:
        """``replace_for_run`` for many runs at once — same per-run semantics, in a
        CONSTANT number of statements (one DELETE, one subject SELECT, one UPDATE)
        instead of 3-4 per run. Each item is ``(run_id, rows, created_at)``.
        Returns the total number of rows written."""
        if not batch:
            return 0
        run_ids = [run_id for run_id, _, _ in batch]
        # Only the rows the mirror owns: in-process and service-reported rows are not
        # in billing, so replacing them with billing's view would erase them.
        await self.session.execute(
            delete(RunUsage).where(RunUsage.run_id.in_(run_ids), _mirrored_rows())
        )
        # Attribution (P6 ORCH-3): a mirrored row belongs to the run's subject, pinned
        # in its snapshot at creation; it has no tenant slot. Its funding is unknown
        # (billing cannot tell platform-paid from tenant BYO) → NULL, never guessed.
        with_rows = [run_id for run_id, rows, _ in batch if rows]
        subjects: dict[UUID, UUID | None] = {}
        if with_rows:
            subjects = {
                rid: attribution_subject(raw)
                for rid, raw in (
                    await self.session.execute(
                        select(Run.id, Run.config_snapshot["subject_id"].astext).where(
                            Run.id.in_(with_rows)
                        )
                    )
                ).all()
            }
        written = 0
        for run_id, rows, created_at in batch:
            subject = subjects.get(run_id) if rows else None
            for r in rows:
                input_tokens = int(r.get("input_tokens") or 0)
                output_tokens = int(r.get("output_tokens") or 0)
                tokens = input_tokens + output_tokens
                self.session.add(
                    RunUsage(
                        id=uuid7(),
                        run_id=run_id,
                        stage_name=r["stage_name"],
                        model=r.get("model"),
                        provider=r.get("provider"),
                        # ``or`` (not a bare default) so an explicit None in the row
                        # — a non-LLM cost event with null token/cost fields — coerces
                        # to the numeric default instead of blowing up int()/Decimal().
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cost_usd=Decimal(str(r.get("cost_usd") or "0")),
                        metadata_={
                            **(r.get("metadata") or {}),
                            "source": BILLING_MIRROR_USAGE_SOURCE,
                        },
                        subject_id=subject,
                        funding=known_funding(r.get("funding")),
                        slot=None,
                        units=Decimal(tokens) if tokens else None,
                        unit_kind="tokens" if tokens else None,
                        **({"created_at": created_at} if created_at is not None else {}),
                    )
                )
            written += len(rows)
        if mark_reconciled:
            await self.session.execute(
                update(Run).where(Run.id.in_(run_ids)).values(usage_reconciled_at=func.now())
            )
        await self.session.flush()
        return written

    async def run_ids_unsettled_usage(
        self,
        tenant_id: UUID,
        *,
        limit: int = 200,
        within_days: int = 30,
        min_terminal_age_seconds: int = USAGE_SETTLE_SECONDS,
    ) -> list[tuple[UUID, datetime.datetime]]:
        """Terminal runs for this tenant whose mirrored cost is not settled yet and
        that are now old enough to settle.

        The reconcile work-list for the tenant-wide reads: a run qualifies if it
        is terminal, recent (created within ``within_days`` — billing retains raw
        cost events ~90d and old runs rarely need refilling), past its settle
        window, and **not** marked reconciled. The marker — not row presence — is
        the settled signal: it lets a genuinely zero-cost run (no rows) drop out
        instead of being re-billed on every load, and it is why a run whose FIRST
        mirror was partial (billing still flushing, typically a cancel mid-stage)
        is still picked up here and re-read in full. This list used to require
        "zero mirrored rows", which made a partial mirror permanent.

        The settle window is ``min_terminal_age_seconds`` (the same
        :data:`USAGE_SETTLE_SECONDS` the run-viewer read uses) and
        :data:`CANCELLED_USAGE_SETTLE_SECONDS` for a cancelled run, whose render
        may still be billing. Inside it the sweep leaves a run alone; the viewer
        read re-mirrors it without marking. Bounded by ``limit``. Newest first.
        """
        settle_after = case(
            (Run.status == "cancelled", CANCELLED_USAGE_SETTLE_SECONDS),
            else_=min_terminal_age_seconds,
        )
        stmt = (
            select(Run.id, Run.created_at)
            .select_from(Run)
            .join(Deliverable, Run.deliverable_id == Deliverable.id)
            .join(Project, Deliverable.project_id == Project.id)
            .where(
                Project.tenant_id == tenant_id,
                Run.status.in_(TERMINAL_RUN_STATUSES),
                Run.created_at >= func.now() - func.make_interval(0, 0, 0, within_days),
                Run.usage_reconciled_at.is_(None),
                Run.updated_at <= func.now() - func.make_interval(0, 0, 0, 0, 0, 0, settle_after),
            )
            .order_by(Run.created_at.desc())
            .limit(limit)
        )
        return [(row[0], row[1]) for row in (await self.session.execute(stmt)).all()]

    async def insert_reported(self, row: RunUsage) -> bool:
        """Insert one service-reported row; ``False`` when its ``(run_id, call_id)``
        is already recorded (a retried report). Idempotent by
        ``uq_run_usage_run_call_id`` (migration 0093) — ``ON CONFLICT DO NOTHING``,
        so a concurrent duplicate cannot raise either."""
        values = {
            attr.columns[0]: getattr(row, attr.key)
            for attr in RunUsage.__mapper__.column_attrs
            if getattr(row, attr.key) is not None
        }
        stmt = (
            pg_insert(RunUsage)
            .values(values)
            .on_conflict_do_nothing(
                index_elements=[RunUsage.run_id, text("(metadata ->> 'call_id')")],
                index_where=text("(metadata ->> 'call_id') IS NOT NULL"),
            )
            .returning(RunUsage.id)
        )
        return (await self.session.execute(stmt)).scalar_one_or_none() is not None

    async def list_for_run(self, run_id: UUID) -> list[RunUsage]:
        return list(
            (await self.session.execute(select(RunUsage).where(RunUsage.run_id == run_id)))
            .scalars()
            .all()
        )

    async def totals_for_runs(self, run_ids: Sequence[UUID]) -> dict[UUID, tuple[int, int]]:
        """``{run_id: (input_tokens, output_tokens)}`` for a page of runs in ONE grouped
        query — the batch form of :meth:`total` for the run-list routes. A run with no
        usage rows is absent from the result (callers default it to ``(0, 0)``); an
        empty ``run_ids`` returns ``{}`` without touching the database."""
        ids = list(run_ids)
        if not ids:
            return {}
        stmt = (
            select(
                RunUsage.run_id,
                func.coalesce(func.sum(RunUsage.input_tokens), 0),
                func.coalesce(func.sum(RunUsage.output_tokens), 0),
            )
            .where(RunUsage.run_id.in_(ids))
            .group_by(RunUsage.run_id)
        )
        return {
            row[0]: (int(row[1]), int(row[2])) for row in (await self.session.execute(stmt)).all()
        }

    async def total(self, run_id: UUID) -> dict[str, Any]:
        stmt = select(
            func.coalesce(func.sum(RunUsage.input_tokens), 0),
            func.coalesce(func.sum(RunUsage.output_tokens), 0),
            func.coalesce(func.sum(RunUsage.cost_usd), 0),
        ).where(RunUsage.run_id == run_id)
        row = (await self.session.execute(stmt)).one()
        return {
            "input_tokens": row[0],
            "output_tokens": row[1],
            "cost_usd": float(row[2]),
        }
