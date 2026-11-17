"""Re-attach a resumed design stage to the html_render job it already submitted.

Every design stage (``website_redesign``, ``web_design``, ``web_design_v2.design``)
submits ONE non-idempotent job to ``html_render_service`` and then polls it. A poll
budget that expires is a *resumable* timeout: ``dag_executor`` resets the stage to
``pending`` and the reconciler re-enters ``handle()`` FROM THE TOP. Without a stash,
that re-entry submits a brand-new job while the first one keeps rendering — and
``POST /api/designs`` has no idempotency key, so the de-duplication has to live here.

**What that cost, measured.** Full-matrix run 33590213252 (2026-09-02) failed
``website-review-variants`` with ``StageTimeoutError: stage 'website_redesign.wait':
all 8 attempts timed out`` — while the job it gave up on,
``01a060f2-eb9d-724d-a49b-f76294cf447d``, went on to COMPLETE SUCCESSFULLY at
10:52:09, ~25 minutes after the orchestrator quit. Four other 2-variant jobs the same
day executed 5580 / 5885 / 7102 / 7370s, all clean. Every one of those exceeds the
8 x 590 = 4720s single-attempt wait ceiling, so attempt 2 was discarding ~100 minutes
of finished work and paying for it a second time. Raising the ceiling does not fix
that class — ``stage_call`` hard-caps ``retries`` at 10 and its backoff is
exponential and unbounded — so the attempts are made to COMPOSE instead.

**Why the write-once contract permits the pre-wait write.** ``stage_outputs.output``
is frozen by ``trg_stage_outputs_write_once`` (migration 0063) only ``WHEN OLD.status
IN ('completed','failed','skipped','cancelled')``. This stash goes through
``StageOutputRepository.record_progress``, whose UPDATE is itself guarded to
``status IN ('pending','running')``, so it can only ever touch a NON-terminal row —
the trigger is not even invoked. ``mark_pending`` (the resumable-timeout reset)
clears status/completed_at/error but deliberately leaves ``output`` intact, which is
what carries the id into the next attempt; ``mark_completed`` then replaces the whole
dict with the real stage output. No exemption GUC, no post-terminal edit.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Any

from src.database.repositories.stage_output_repository import StageOutputRepository
from src.database.session import bind_session_tenant

if TYPE_CHECKING:
    from src.clients.design_client import DesignClient
    from src.clients.schemas.design import DesignJobStatus
    from src.services.runs.run_context import RunContext

log = logging.getLogger(__name__)

SessionFactory = Callable[[], AbstractAsyncContextManager[Any]]

#: Key under which the in-flight html_render job id is stashed in the stage's own
#: ``stage_outputs.output`` while the job is still running. Shared across the three
#: design stages on purpose: the stash is per (run_id, stage_name), so the key does
#: not need to be per-stage, and one name keeps a live row readable by hand.
JOB_STASH_KEY = "in_flight_design_job_id"

#: A stashed job in one of these states is dead; a resume must submit a fresh one.
#: Anything else (queued / processing / completed / a status string this repo has
#: never seen) is re-attachable — including ``completed``, which is the case that
#: recovers the work the old code threw away: the re-attached poll returns at once
#: and the stage goes straight to ``get_design_result``. Worst case for an unknown
#: status is that ``wait_for_design`` re-raises the service's own terminal error on
#: poll 1, which is exactly today's behaviour.
DEAD_STATUSES = frozenset({"failed", "error", "cancelled", "canceled", "expired", "not_found"})


async def peek_status(client: DesignClient, job_id: str, stage_name: str) -> DesignJobStatus | None:
    """Best-effort single status read. ``None`` = the probe itself failed."""
    try:
        return await client.get_design_status(job_id)
    except Exception:  # diagnostic probe; never fail the stage on it
        log.warning("%s: status probe for job %s failed", stage_name, job_id, exc_info=True)
        return None


async def reattachable_job_id(
    client: DesignClient,
    sessions: SessionFactory | None,
    ctx: RunContext,
    stage_name: str,
) -> str | None:
    """The still-alive design job this stage submitted on a previous attempt.

    ``None`` means "submit a fresh one": no session factory wired (the stage
    degrades to the old resubmit behaviour rather than failing), nothing stashed,
    the bookkeeping read failed, or the stashed job is in a :data:`DEAD_STATUSES`
    state.
    """
    if sessions is None:
        return None
    try:
        async with sessions() as session:
            await bind_session_tenant(session, ctx.tenant_id)
            row = await StageOutputRepository(session).get(ctx.run_id, stage_name)
    except Exception:  # a bookkeeping read must never fail the stage
        log.warning("%s: could not read the stashed design job id", stage_name, exc_info=True)
        return None
    stashed = (row.output or {}).get(JOB_STASH_KEY) if row is not None else None
    job_id = str(stashed or "").strip()
    if not job_id:
        return None
    status = await peek_status(client, job_id, stage_name)
    if status is None or status.status.lower() in DEAD_STATUSES:
        log.info(
            "%s: stashed design job %s is not re-attachable (status=%s) — submitting a fresh one",
            stage_name,
            job_id,
            status.status if status else "probe-failed",
        )
        return None
    log.info(
        "%s: re-attaching to in-flight design job %s (status=%s) instead of submitting a new one",
        stage_name,
        job_id,
        status.status,
    )
    return job_id


async def stash_job_id(
    sessions: SessionFactory | None,
    ctx: RunContext,
    stage_name: str,
    job_id: str,
) -> None:
    """Persist ``job_id`` into the stage's own non-terminal ``output`` JSONB.

    Must be awaited BEFORE the wait: a poll-deadline expiry resets this stage to
    ``pending`` and re-enters ``handle()`` from the top, and without the stash that
    re-entry submits a second job while the first keeps rendering.
    """
    if sessions is None:
        return
    try:
        async with sessions() as session:
            await bind_session_tenant(session, ctx.tenant_id)
            await StageOutputRepository(session).record_progress(
                ctx.run_id, stage_name, {JOB_STASH_KEY: str(job_id)}
            )
            await session.commit()
    except Exception:  # a bookkeeping write must never fail the stage
        log.warning(
            "%s: could not stash design job %s; a resume would resubmit it",
            stage_name,
            job_id,
            exc_info=True,
        )
