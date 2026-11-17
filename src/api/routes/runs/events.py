"""Run SSE stream + the catch-up snapshot builder (ungated ``sse_router``)."""

from __future__ import annotations

import logging
from uuid import UUID

from fastapi import HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from src.api.routes._common import TenantIdDep
from src.api.sse import stream_channel
from src.database.models import Run
from src.database.repositories.stage_output_repository import StageOutputRepository
from src.database.session import bind_session_tenant

from ._shared import _tenant_owns_run, sse_router

log = logging.getLogger(__name__)


@sse_router.get("/{run_id}/events", operation_id="runs.events")
async def run_events(
    run_id: UUID,
    request: Request,
    # why: an EventSource cannot set the X-Active-Tenant-Id header — the
    # scope-switch rides ?active_tenant=, which get_tenant_id also reads.
    tenant_id: TenantIdDep,
) -> EventSourceResponse:
    # Build the catch-up snapshot on a SHORT-LIVED session, then stream. This
    # route is mounted off the tenant-gate (see asgi.py / sse_router) precisely
    # so it does NOT hold a request-scoped session for the stream's lifetime; the
    # ownership check here (stricter than require_known_tenant) plus the RLS bind
    # run on that short-lived session, which is released before streaming starts.
    factory = request.app.state.session_factory
    async with factory() as session:
        await bind_session_tenant(session, tenant_id)
        run = await _tenant_owns_run(session, run_id, tenant_id)
        if run is None:
            raise HTTPException(404, "run not found")
        try:
            catchup = await _build_run_event_snapshot(session, run)
        except Exception:
            # Snapshot is best-effort; a failure must never break the live stream.
            log.exception("run %s SSE snapshot failed; streaming live only", run_id)
            catchup = []

    return await stream_channel(request, channel=f"runs:{run_id}", catchup=catchup)


def _stage_terminal_event(rid: str, stage_name: str, status: str, error: str | None) -> dict | None:
    """The terminal SSE event for one non-'pending' stage row, or None if the
    stage is still running. Vocabulary mirrors the live publish sites so replay
    and stream are interchangeable on the consumer.
    """
    if status == "completed":
        return {"event": "stage_completed", "run_id": rid, "stage_name": stage_name}
    if status == "failed":
        return {
            "event": "stage_failed",
            "run_id": rid,
            "stage_name": stage_name,
            "error": error or "unknown",
        }
    if status == "skipped":
        # `reason` (not `error`) mirrors the live publish site: `dag_executor`
        # emits `stage_skipped(stage_name=…, reason=…)`, and
        # `StageOutputRepository.mark_skipped` persists that reason in `error`.
        return {
            "event": "stage_skipped",
            "run_id": rid,
            "stage_name": stage_name,
            "reason": error or "skipped",
        }
    if status == "cancelled":
        ev: dict = {"event": "stage_cancelled", "run_id": rid, "stage_name": stage_name}
        if error:
            ev["error"] = error
        return ev
    return None


async def _build_run_event_snapshot(session: AsyncSession, run: Run) -> list[dict]:
    """Replay current run state as the same SSE event shapes the dispatcher
    publishes live (``run_started`` / ``stage_*`` / terminal ``run_*``), so a
    client connecting mid-run — or after a fast run already finished — is brought
    up to date instead of waiting on events Redis pub/sub has already dropped.

    Order is ``run_started`` first, stage events, terminal ``run_*`` last; the
    stage events are idempotent badge updates on the consumer, so their relative
    order does not matter. A still-``pending`` run yields an empty snapshot — it
    has emitted nothing yet, so the client just waits for the live stream.
    """
    rid = str(run.id)
    events: list[dict] = []
    if run.status == "pending":
        return events
    events.append({"event": "run_started", "run_id": rid})
    for so in await StageOutputRepository(session).list_for_run(run.id):
        # Every non-'pending' row must replay a `stage_started` AND a terminal
        # event. Omitting either leaves the badge stuck on 'pending' after a
        # reconnect — the exact ghost-pending stage the hard-fail sweep to
        # 'cancelled' (`bulk_cancel_non_terminal`) was built to eliminate.
        if so.status == "pending":
            continue
        events.append({"event": "stage_started", "run_id": rid, "stage_name": so.stage_name})
        terminal_stage = _stage_terminal_event(rid, so.stage_name, so.status, so.error)
        if terminal_stage is not None:
            events.append(terminal_stage)
    terminal = {
        "completed": "run_completed",
        "failed": "run_failed",
        "paused": "run_paused",
        "cancelled": "run_cancelled",
    }.get(run.status)
    if terminal:
        ev: dict = {"event": terminal, "run_id": rid}
        if run.status == "failed" and run.error:
            ev["error"] = run.error
        if run.status == "paused" and run.current_stage:
            # Name the checkpoint under review. The replay above emits a
            # `stage_started` for EVERY non-pending stage, so without this the
            # client's "current stage" ends up as whichever row came last —
            # frequently a concurrently-started sibling that then SKIPPED — and
            # the pause banner points at a stage that never ran. On pause the
            # dispatcher re-points `current_stage` at the triggering checkpoint
            # (run_finalizer.pause_trigger_stage), so this is the reviewable one.
            ev["stage_name"] = run.current_stage
        events.append(ev)
    # Every snapshot event says it is a REPLAY. The snapshot is re-sent on EVERY
    # (re)connect — including the ``SSE_MAX_STREAM_SECONDS`` forced reconnect —
    # and carries no timestamp, so a consumer that logs events (the Studio
    # Activity Log) printed "Run started" plus every stage line a second time,
    # all at the reconnect instant, which reads exactly like a double-dispatched
    # run. It is not one: nothing was re-published. The flag lets the consumer
    # apply the state update and skip the log line. It cannot be dropped instead
    # (``Last-Event-ID``): pub/sub keeps no backlog, so the snapshot is the only
    # way a reconnecting client learns what happened during the gap.
    return [{**event, "replay": True} for event in events]
