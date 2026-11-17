"""Run read routes: detail, lightweight status, captions, PDF report, ancestors."""

from __future__ import annotations

import asyncio
import datetime
import logging
import time
from typing import Any, NamedTuple, cast
from uuid import UUID

from fastapi import HTTPException, Response
from sqlalchemy import select
from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.artifact_url_rewriter import proxy_path
from src.api.download_names import attachment_disposition
from src.api.routes._common import (
    RunUsageReconcilerDep,
    SessionDep,
    TenantIdDep,
)
from src.api.schemas import (
    RunDetail,
    RunFailureDetail,
    RunStatusLite,
    RunSummary,
    RunUsageResponse,
    RunUsageTotal,
    StageOutputResponse,
    StageStatusLite,
)
from src.clients.design_client import DesignClient
from src.clients.downstream_credential import DownstreamCredentialMissingError
from src.config import get_settings
from src.database.models import Project, Run, VideoSource
from src.database.repositories.run_repository import RunRepository
from src.database.repositories.run_usage_repository import (
    TERMINAL_RUN_STATUSES,
    RunUsageRepository,
    usage_settle_seconds,
)
from src.database.repositories.stage_output_repository import StageOutputRepository
from src.observability.metrics import run_detail_phase_seconds, run_detail_stages
from src.services.blueprints.registry import get_registry
from src.services.run_captions import srt_for_run
from src.services.run_usage_reconciler import (
    COST_UNAVAILABLE_BILLING_CREDENTIAL_MISSING,
    RunUsageReconciler,
)
from src.services.runs.elements import resolve_clip_spine_stage
from src.services.stages.shared.audit_report import build_report_html
from src.services.stages.shared.audit_report.primitives import host_of, scan_url

from ._shared import (
    _AUDIT_SCAN_STAGES,
    _MAX_ANCESTORS,
    _project_id_for_deliverable,
    _resolve_delivered_artifact_url,
    _stage_response,
    _summary,
    _tenant_owns_run,
    router,
)

log = logging.getLogger(__name__)

# The stage whose output carries per-clip ``sources[].preview_url`` — the footage
# review gate. The ONLY stage this backfill touches.
_FOOTAGE_PROBE_STAGE = "footage_probe"
# ``video_sources.preview_status`` value that means the proxy object really exists.
_PREVIEW_READY = "ready"
# The EXACT prefix of the note ``footage_probe._describe`` used to emit for a
# not-yet-rendered derivative, up to and including the trailing space before the
# parenthetical. The emitter was deleted in #809 ("a pending preview derivative is
# not a degradation"), so every surviving occurrence is frozen historical residue
# in ``stage_outputs.output`` from a run probed before that commit. Formatted with
# a source id we already hold and matched with an anchored ``startswith`` — NOT a
# regex over prose. ``notes`` is a general-purpose channel, so swallowing a note
# that is unrelated or still TRUE would be a worse bug than the stale amber text.
_STALE_PENDING_NOTE_PREFIX = "source {source_id} preview_status=pending "

# How long GET /api/runs/{id}'s DB work may run before this handler gives up and
# returns a fast, clearly-labelled 503 instead of hanging (issue #949).
#
# Measured cost of NOT having this: when worker4 (the Postgres write primary)
# rebooted 2026-09-04, this endpoint's DB calls blocked on psycopg's
# server_login_retry loop against a dying/dead backend for the full 120s until
# the CLIENT gave up with a bare ReadTimeout — an AUDIT_UNHANDLED 500 with
# duration_ms=120262.6 and no indication anywhere that the cause was "database
# down", only "something took two minutes and then failed".
#
# 8s is chosen because: (a) it is far above this endpoint's real cost — the
# PHASE TIMING metrics on the block below (ownership + stage_query +
# other_queries + response_build) are all indexed single-run lookups that run
# in well under 1s in steady state; (b) it is far below both Cloudflare's 100s
# origin-response timeout (already the documented cause of this endpoint's
# 524s) and the 120s ReadTimeout measured in #949, so operators get the 503
# BEFORE either of those fires and see "database unavailable" instead of an
# opaque client-side timeout or gateway error.
_RUN_DETAIL_DB_TIMEOUT_SECONDS = 8.0
# The read-through billing call inside that 8s budget. The billing client's own
# timeout is 15s, so unbounded a slow billing turned a healthy run read into a
# 503 "database did not respond". Bounded well inside the budget; on expiry the
# load skips the cost read-through (the next load / sweep fills it).
_RUN_DETAIL_BILLING_TIMEOUT_SECONDS = 3.0


class _NeedyPreview(NamedTuple):
    """An empty-``preview_url`` source entry plus the stage response that owns it.

    The stage is carried because the stale note lives on the OWNING stage's flat
    ``output["notes"]`` list (``services/stages/base.py`` — one list per stage, not
    per source), so repairing a source has to be able to find its way back there.
    """

    stage: StageOutputResponse
    source: dict[str, object]


def _needy_preview_sources(
    stage_responses: list[StageOutputResponse],
) -> dict[str, list[_NeedyPreview]]:
    """``footage_probe`` source entries with an EMPTY ``preview_url``, by source id.

    Split out of :func:`_backfill_ready_previews` only to keep that function under
    the complexity gate; the two are one operation.
    """
    needy: dict[str, list[_NeedyPreview]] = {}
    for stage in stage_responses:
        if stage.stage_name != _FOOTAGE_PROBE_STAGE or not stage.output:
            continue
        sources = stage.output.get("sources")
        if not isinstance(sources, list):
            continue
        for entry in cast("list[object]", sources):
            if not isinstance(entry, dict):
                continue
            source = cast("dict[str, object]", entry)
            if source.get("preview_url"):
                continue
            source_id = source.get("id")
            if isinstance(source_id, str) and source_id:
                needy.setdefault(source_id, []).append(_NeedyPreview(stage, source))
    return needy


def _drop_stale_pending_notes(
    stage_responses: list[StageOutputResponse],
    repaired: dict[UUID, set[str]],
) -> None:
    """Drop the pre-#809 ``preview_status=pending`` note for each REPAIRED source.

    The deployed footage gate rendered ten amber warnings saying the derivatives
    were not rendered yet, stacked directly above ten tiles that said "preview
    ready" and that played. The page contradicted itself and the alarming half was
    the wrong half — the note is residue, the tile is the truth.

    Two properties do the safety work, both by construction rather than by luck:

    * The match is ``startswith`` on a prefix built from an id in ``repaired`` —
      i.e. a source THIS request just proved ``ready`` against the live
      ``video_sources`` row. A source that is genuinely still ``pending`` was
      never flipped, so its id never reaches this function and its note stands.
    * The prefix pins ``preview_status=pending`` literally, so the ``failed``
      note (which #809 kept, and which carries ``row.error``) and every unrelated
      note (``... has no probed duration; treating as 0s``, music/solver notes,
      anything a future stage adds) cannot match it.

    ``degraded`` is recomputed here, and that is the whole user-visible point, not
    an oversight to be tidied away: ``_stage_response`` sets
    ``degraded = completed and bool(notes)`` BEFORE this projection runs, so
    leaving it alone would ship the fix INERT — the warnings would vanish while the
    run stayed badged "completed with warnings". A run whose only notes were stale
    ``pending`` ones becoming NOT degraded is the correct outcome and is precisely
    what #809 ruled; this just extends that ruling to rows written before it.

    Read-time projection only: ``stage.output`` is a fresh dict that
    ``model_validate`` parsed out of the JSONB column, so the persisted row is
    never touched.
    """
    for stage in stage_responses:
        source_ids = repaired.get(stage.id)
        if not source_ids or not stage.output:
            continue
        notes = stage.output.get("notes")
        if not isinstance(notes, list):
            continue
        prefixes = tuple(
            _STALE_PENDING_NOTE_PREFIX.format(source_id=source_id) for source_id in source_ids
        )
        kept = [
            note
            for note in cast("list[object]", notes)
            if not (isinstance(note, str) and note.startswith(prefixes))
        ]
        if len(kept) == len(notes):
            continue
        stage.output["notes"] = kept
        stage.degraded = stage.status == "completed" and bool(kept)


async def _backfill_ready_previews(
    session: AsyncSession,
    run_id: UUID,
    project_id: UUID,
    stage_responses: list[StageOutputResponse],
) -> None:
    """Fill in ``footage_probe``'s empty ``preview_url``s from the live source rows.

    ``footage_probe._describe`` presigns ``preview_url`` at STAGE-EXECUTION time,
    so a run whose probe ran before the preview-proxy worker existed stored
    ``preview_url: ""`` — permanently. The browser-playable object exists in MinIO
    today and the ``video_sources`` row says ``preview_status='ready'``, but the
    reviewer's footage gate still renders "no preview generated" for every clip.

    This is a READ-TIME PROJECTION and nothing else: the persisted
    ``stage_outputs.output`` row is never mutated (``_stage_response`` already
    handed us a fresh dict parsed out of the JSONB column). The URL is built with
    the SAME ``proxy_path`` the rewriter uses, so it is non-expiring and matches
    what ``artifact_proxy`` will serve — the allowlist was widened in
    ``artifact_ref_guard._ready_preview_refs`` FIRST, so this never emits a path
    the proxy would 404. A ``pending`` or ``failed`` source is left exactly as it
    was: an honest empty state beats a dead player.

    ``preview_status`` is projected alongside the URL, deliberately. The stored
    value is stale for precisely the runs this exists to repair (the worker had
    not run yet, so the probe recorded ``pending``), and the SPA gates the player
    on ``preview_status === 'ready' && preview_url``
    (``FootageProbeGrid.svelte``) — writing only the URL would ship an INERT fix.
    Both fields come from the same authoritative row, so they cannot disagree.

    Finally, a source flipped to ``ready`` here also sheds the pre-#809
    ``preview_status=pending`` note that the same staleness left behind — see
    :func:`_drop_stale_pending_notes`, which is where the safety argument lives.

    ONE extra query, only when there is something to fill.
    """
    needy = _needy_preview_sources(stage_responses)
    if not needy:
        return

    # Scoped to the run's OWN project — the same project id the ownership check
    # already resolved for this tenant — and RLS on ``video_sources`` climbs
    # project_id -> projects.tenant_id, so this cannot read another tenant's rows.
    rows = await session.execute(
        select(VideoSource.id, VideoSource.preview_key).where(
            VideoSource.project_id == project_id,
            VideoSource.preview_status == _PREVIEW_READY,
            VideoSource.preview_key.isnot(None),
        )
    )
    bucket = get_settings().storage.minio_bucket
    # Which source ids this request actually flipped to ``ready``, per owning stage.
    # Only these may lose their historical ``pending`` note.
    repaired: dict[UUID, set[str]] = {}
    for source_id, preview_key in rows.all():
        if not preview_key:
            continue
        path = proxy_path(run_id, f"{bucket}/{preview_key}", "")
        for needy_preview in needy.get(str(source_id), ()):
            needy_preview.source["preview_url"] = path
            needy_preview.source["preview_status"] = _PREVIEW_READY
            repaired.setdefault(needy_preview.stage.id, set()).add(str(source_id))
    _drop_stale_pending_notes(stage_responses, repaired)


@router.get("/{run_id}", response_model=RunDetail, operation_id="runs.get")
async def get_run(
    run_id: UUID,
    session: SessionDep,
    tenant_id: TenantIdDep,
    reconciler: RunUsageReconcilerDep,
) -> RunDetail:
    """Bounded wrapper: see ``_RUN_DETAIL_DB_TIMEOUT_SECONDS`` above (#949) — a
    dead/unreachable Postgres must surface as a fast 503, not a 120s hang."""
    try:
        return await asyncio.wait_for(
            _get_run_detail(run_id, session, tenant_id, reconciler),
            timeout=_RUN_DETAIL_DB_TIMEOUT_SECONDS,
        )
    except TimeoutError as exc:
        log.error(
            "run %s detail query exceeded %.0fs — treating as database unavailable",
            run_id,
            _RUN_DETAIL_DB_TIMEOUT_SECONDS,
        )
        raise HTTPException(
            503,
            "service_unavailable: database did not respond in time; try again shortly",
        ) from exc
    except (OperationalError, DBAPIError) as exc:
        log.error("run %s detail query hit a DB error: %s", run_id, exc)
        raise HTTPException(
            503,
            f"service_unavailable: database error ({type(exc).__name__})",
        ) from exc


def _failed_stage(run: Run, stage_responses: list[StageOutputResponse]) -> str | None:
    """The stage a FAILED run failed in; ``None`` for any other status.

    ``mark_failed`` re-points ``current_stage`` at the failing stage when the
    dispatcher knows it, so that is authoritative. A run failed without one (a
    run-level failure) falls back to the first stage row that itself failed.
    """
    if run.status != "failed":
        return None
    if run.current_stage:
        return run.current_stage
    return next((s.stage_name for s in stage_responses if s.status == "failed"), None)


async def _get_run_detail(
    run_id: UUID,
    session: AsyncSession,
    tenant_id: UUID,
    reconciler: RunUsageReconciler,
) -> RunDetail:
    # PHASE TIMING — measurement only, no behaviour change. This endpoint
    # intermittently 524s, i.e. Cloudflare's 100s origin-response timeout fires
    # (the tunnel sets only connectTimeout: 30s and the w2v VirtualService route
    # allows 3600s, so Istio is not the cutter). The marks below partition the
    # whole handler body into four labelled phases on
    # ``w2v_orchestrator_run_detail_phase_seconds``.
    #
    # Non-obvious payoff: src/middleware/metrics.py already records
    # ``http_request_duration_seconds{endpoint="/api/runs/{run_id}"}`` for the
    # ENTIRE request, so (middleware total - sum of these four phases) is the
    # FastAPI ``response_model`` re-validation + JSON encode cost, for free.
    # That work happens after this function returns and therefore cannot be
    # marked from inside it — subtraction is the only way to see it.
    _t_start = time.perf_counter()

    run = await _tenant_owns_run(session, run_id, tenant_id)
    _t_owned = time.perf_counter()
    run_detail_phase_seconds.labels(phase="ownership").observe(_t_owned - _t_start)
    if run is None:
        raise HTTPException(404, "run not found")

    so_repo = StageOutputRepository(session)
    usage_repo = RunUsageRepository(session)
    # Selects whole ORM rows, including each stage's full JSONB ``output`` —
    # there is no projection here, so this phase carries the payload's weight.
    stages = await so_repo.list_for_run(run_id)
    _t_stages = time.perf_counter()
    run_detail_phase_seconds.labels(phase="stage_query").observe(_t_stages - _t_owned)
    run_detail_stages.observe(len(stages))

    by_stage = await usage_repo.list_for_run(run_id)
    # Read-through: a terminal run's priced cost is mirrored from billing into
    # local ``run_usage`` rows so the Cost tile shows real $ instead of $0.
    # Best-effort (billing outage → nothing written, nothing marked).
    #
    # It CONVERGES, gated on ``usage_reconciled_at`` alone: every load of an
    # unsettled terminal run re-reads billing and REPLACES the mirrored rows
    # (``replace_for_runs`` deletes only billing-owned rows), so a first read that
    # caught only PART of the async cost events is corrected by the next one. It
    # used to also require "no mirrored rows yet", and a partial first mirror —
    # likeliest on a run cancelled mid-stage — then blocked every later re-read
    # while the sweep skipped the run too: the cost stayed partial forever.
    #
    # Settling is bounded: once the run is past its settle window
    # (``usage_settle_seconds``: 30s, or the compiler's render ceiling for a
    # cancelled run whose render may still be billing) the read marks it
    # reconciled and it becomes a pure local read, so a genuinely zero-cost run
    # is not re-billed on every load. That mark is what
    # ``usage_total.cost_settled`` reports to the frontend. A contended read is
    # skipped, not serialised: a read that loses the run's mirror lock writes
    # nothing and reports the stored cost as not yet settled.
    cost_unavailable_reason: str | None = None
    cost_settled = run.usage_reconciled_at is not None
    if run.usage_reconciled_at is None and run.status in TERMINAL_RUN_STATUSES:
        now = datetime.datetime.now(datetime.UTC)
        terminal_age = (now - run.updated_at).total_seconds() if run.updated_at else 0.0
        mark = terminal_age >= usage_settle_seconds(run.status)
        try:
            await reconciler.reconcile_one(
                tenant_id,
                run_id,
                run.created_at,
                mark_reconciled=mark,
                billing_timeout_s=_RUN_DETAIL_BILLING_TIMEOUT_SECONDS,
            )
        except DownstreamCredentialMissingError:
            # Billing configured, no credential to call it (enforce, no token).
            # The run read must NOT fail and must NOT show $0: report the cost as
            # unavailable. Logged + counted inside the reconciler.
            cost_unavailable_reason = COST_UNAVAILABLE_BILLING_CREDENTIAL_MISSING
        by_stage = await usage_repo.list_for_run(run_id)
        if mark and cost_unavailable_reason is None:
            # Settled only if the reconcile actually stamped the run: a billing
            # outage / timeout writes and marks nothing, so it stays unsettled.
            cost_settled = (
                await session.execute(select(Run.usage_reconciled_at).where(Run.id == run_id))
            ).scalar_one_or_none() is not None
    total = await usage_repo.total(run_id)

    project_id = await _project_id_for_deliverable(session, run.deliverable_id)
    _t_queries = time.perf_counter()
    run_detail_phase_seconds.labels(phase="other_queries").observe(_t_queries - _t_stages)

    # model_validate(from_attributes=True) per stage + rewrite_payload's full
    # recursive walk of each stage's output — the CPU half of the handler.
    stage_responses = [_stage_response(run.id, s) for s in stages]
    # Repair the historical empty ``preview_url``s BEFORE the summary/detail is
    # built — a read-time projection, never a write. See the helper's docstring.
    await _backfill_ready_previews(session, run.id, project_id, stage_responses)
    # ``degraded`` now lives on RunSummary, so the detail view feeds it through
    # the summary projection instead of setting it on RunDetail — same value,
    # same rule (any stage completed-with-warnings), one place.
    summary = _summary(
        run,
        project_id,
        degraded=any(sr.degraded for sr in stage_responses),
        tokens=(int(total["input_tokens"]), int(total["output_tokens"])),
    ).model_dump()
    # Publish the reorder spine instead of letting the Studio FE re-derive it
    # from output SHAPE: four stages of a demo_video_v1 run carry an identical
    # ``lines[]`` array and only ``demo_video_gen`` is a legal reorder target, so
    # shape-sniffing picks the wrong stage and POST /runs/{id}/reorder-clips 422s.
    # Uses the ``stages`` rows already loaded above — no extra round-trip.
    try:
        blueprint = get_registry().resolve_by_id(run.blueprint_id)
    except KeyError:
        # An unknown/retired blueprint id must not 500 a read.
        blueprint = None
    spine = resolve_clip_spine_stage(blueprint, stages) if blueprint is not None else None
    # Publish the DELIVERABLE the same way, and for the same reason: four stages
    # of a demo_video_v1 run carry an ``artifact_url`` and only the blueprint says
    # which one is the file that shipped. The stage is declared; the URL is
    # resolved off the already-rewritten stage_responses via the blueprint's own
    # skip-clean chain, so it is byte-identical to what ``stage_outputs`` serves
    # for that stage. No extra round-trip.
    delivered_stage = blueprint.delivered_stage() if blueprint is not None else None
    delivered_url = (
        _resolve_delivered_artifact_url(blueprint, stage_responses)
        if blueprint is not None
        else None
    )

    detail = RunDetail(
        **summary,
        clip_spine_stage=spine,
        delivered_artifact_stage=delivered_stage,
        delivered_artifact_url=delivered_url,
        config_snapshot=dict(run.config_snapshot),
        pause_after_stages=list(run.pause_after_stages or []),
        error=run.error,
        failure_class=run.failure_class,  # ty: ignore[invalid-argument-type]  # why: SA mapped column → str at runtime; the ck_runs_failure_class CHECK + the Literal-parity guard keep the DB inside the vocabulary, and pydantic narrows at construction (same shape as `status` in runs/_shared.py::_summary)
        failure_detail=(
            RunFailureDetail.model_validate(run.failure_detail) if run.failure_detail else None
        ),
        failed_stage=_failed_stage(run, stage_responses),
        stage_outputs=stage_responses,
        usage_total=RunUsageTotal(
            input_tokens=total["input_tokens"],
            output_tokens=total["output_tokens"],
            cost_usd=None if cost_unavailable_reason else total["cost_usd"],
            cost_status="unavailable" if cost_unavailable_reason else "ok",
            cost_unavailable_reason=cost_unavailable_reason,
            cost_settled=cost_settled,
            by_stage=[
                RunUsageResponse(
                    stage_name=u.stage_name,
                    model=u.model,
                    provider=u.provider,
                    input_tokens=u.input_tokens,
                    output_tokens=u.output_tokens,
                    cost_usd=float(u.cost_usd),
                    connection_id=u.connection_id,
                    connection_version=u.connection_version,
                )
                for u in by_stage
            ],
        ),
    )
    run_detail_phase_seconds.labels(phase="response_build").observe(
        time.perf_counter() - _t_queries
    )
    return detail


@router.get("/{run_id}/status", response_model=RunStatusLite, operation_id="runs.status")
async def get_run_status(
    run_id: UUID, session: SessionDep, tenant_id: TenantIdDep, response: Response
) -> RunStatusLite:
    """Lightweight live-status projection for the run-monitor's safety-net poll.

    Returns run-level status plus each stage's lifecycle state — and nothing
    else. Deliberately omits every stage ``output`` blob, ``config_snapshot``
    and ``usage_total`` that ``GET /{run_id}`` (RunDetail) ships: polling that
    heavy payload every few seconds grew with each completed stage and froze
    the browser on long multi-stage runs. The full detail is fetched once on
    terminal for the final render.

    ``Cache-Control: no-store`` is REQUIRED: this is a high-frequency poll whose
    whole purpose is to observe a state transition (running → completed). Without
    it, an intermediary HTTP cache (a client's request context, a browser fetch)
    can serve a cached pre-completion ``running`` response for the remainder of
    the poll even after the run is terminal — the run never *looks* done to the
    poller though the DB (and every replica) says ``completed``. This was the
    root cause of the e2e website-review shard's ~20-min stale-``running`` hang
    (2026-07-03): verified live that all orchestrator pods returned ``completed``
    while the poll still saw ``running``.
    """
    response.headers["Cache-Control"] = "no-store"
    run = await _tenant_owns_run(session, run_id, tenant_id)
    if run is None:
        raise HTTPException(404, "run not found")

    # A PROJECTION (stage_name/status/timestamps + a SQL-computed degraded bit),
    # not StageOutputRepository.list_for_run — that would read every stage's
    # ``output`` JSONB off the wire just to discard it, which is exactly the cost
    # this endpoint exists to avoid.
    stages = await RunRepository(session).stage_statuses(run_id)
    return RunStatusLite(
        id=run.id,
        status=run.status,  # ty: ignore[invalid-argument-type]  # why: SA mapped column → str at runtime; pydantic Literal narrows at construction
        current_stage=run.current_stage,
        error=run.error,
        paused_at=run.paused_at if run.status == "paused" else None,
        pause_reason=run.pause_reason if run.status == "paused" else None,  # ty: ignore[invalid-argument-type]  # why: SA mapped column → str at runtime; ck_runs_pause_reason keeps it in the Literal
        # Same rule as the detail view: a completed stage carrying notes
        # soft-failed downstream work, so the run finished green but degraded.
        degraded=any(s.degraded for s in stages),
        stage_outputs=[
            StageStatusLite(
                stage_name=s.stage_name,
                # why: the projection types this `str`; pydantic narrows to the
                # StageStatus Literal at construction (the DB CHECK is the guard).
                status=s.status,
                started_at=s.started_at,
                completed_at=s.completed_at,
                attempts=s.attempts,
            )
            for s in stages
        ],
    )


@router.get(
    "/{run_id}/captions.srt",
    operation_id="runs.captions",
    summary="Download the run's soft-CC caption track as an SRT file",
)
async def get_run_captions(
    run_id: UUID,
    session: SessionDep,
    # why: reached as an <a href download>, which cannot set the
    # X-Active-Tenant-Id header — the scope-switch rides ?active_tenant= instead.
    # get_tenant_id reads both channels, so the router-level gates see it too.
    tenant_id: TenantIdDep,
) -> Response:
    """Mint and return the run's soft-CC captions as a downloadable ``.srt``.

    Built on demand from the run's narration timing (the same accurate track the
    YouTube publish path attaches), so it's available regardless of whether the
    run was ever published — the published-modal link only existed transiently.
    Returns 404 for a run with no captionable narration (audiobook / unrecognised)
    or when the caller's tenant does not own it.
    """
    run = await _tenant_owns_run(session, run_id, tenant_id)
    if run is None:
        raise HTTPException(404, "run not found")
    srt = await srt_for_run(session, run_id)
    if not srt:
        raise HTTPException(404, "no captions for this run")
    project_id = await _project_id_for_deliverable(session, run.deliverable_id)
    title = (
        await session.execute(select(Project.title).where(Project.id == project_id))
    ).scalar_one_or_none()
    return Response(
        content=srt,
        media_type="application/x-subrip",
        # Named after the project (``acme-launch-captions.srt``), matching the
        # frontend's MP4 name. This header overrides the link's `download` hint.
        headers={"Content-Disposition": attachment_disposition(title, "-captions.srt")},
    )


@router.get(
    "/{run_id}/report/pdf",
    operation_id="runs.report.pdf",
    summary="Download an audit run's scorecard as a PDF report",
)
async def get_run_report_pdf(
    run_id: UUID,
    session: SessionDep,
    tenant_id: TenantIdDep,
) -> Response:
    """Render an audit run's scorecard to a downloadable PDF, on demand.

    Built from the run's ``<type>_scan`` structured output (seo_audit / cro_audit /
    marketing_scorecard — the same data the frontend scorecard renders) → a
    self-contained HTML page →
    html_render_service's headless-Chromium print. On demand (no stored artifact) so
    it's always current and adds no render cost to a run that is never downloaded.
    404 for a run the caller's tenant doesn't own or that has no audit scorecard.
    """
    if (await _tenant_owns_run(session, run_id, tenant_id)) is None:
        raise HTTPException(404, "run not found")
    repo = StageOutputRepository(session)
    scan: dict[str, Any] | None = None
    project_type = ""
    for name in _AUDIT_SCAN_STAGES:
        so = await repo.get(run_id, name)
        if so is not None and so.status == "completed" and so.output:
            scan = so.output
            # The matched stage name IS the audit type (``<project_type>_scan``);
            # the template is dispatched on it, never guessed from the payload.
            project_type = name.removesuffix("_scan")
            break
    if not scan:
        raise HTTPException(404, "no audit scorecard for this run")
    html = build_report_html(project_type, scan)
    # A per-request client owns an httpx connection pool; close it or every
    # download leaks one (sockets + a GC-time "unclosed client" warning).
    design = DesignClient()
    try:
        pdf = await design.render_pdf(html)
    finally:
        await design.close()
    filename = f"{project_type}-{host_of(scan_url(scan))}.pdf"
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get(
    "/{run_id}/ancestors",
    response_model=list[RunSummary],
    operation_id="runs.ancestors",
)
async def list_run_ancestors(
    run_id: UUID,
    session: SessionDep,
    tenant_id: TenantIdDep,
) -> list[RunSummary]:
    # v3 "drafts" = previous runs of the same deliverable, chained via
    # Run.parent_run_id. Walk oldest-first so the UI can render a timeline.
    run = await _tenant_owns_run(session, run_id, tenant_id)
    if run is None:
        raise HTTPException(404, "run not found")
    # Guard the walk: parent_run_id is not constrained acyclic at write time, so
    # a cycle (or a pathologically deep chain) would otherwise spin one DB query
    # per hop forever. A visited-set breaks cycles; the depth cap bounds work.
    chain: list[Run] = []
    seen: set[UUID] = {run.id}
    cursor_id: UUID | None = run.parent_run_id
    while cursor_id is not None and len(chain) < _MAX_ANCESTORS:
        if cursor_id in seen:
            log.warning("ancestor walk for run %s hit a cycle at %s; truncating", run_id, cursor_id)
            break
        seen.add(cursor_id)
        parent = (
            await session.execute(select(Run).where(Run.id == cursor_id))
        ).scalar_one_or_none()
        if parent is None:
            break
        chain.append(parent)
        cursor_id = parent.parent_run_id
    chain.reverse()
    project_id = await _project_id_for_deliverable(session, run.deliverable_id)
    # ONE aggregate for the whole chain — the ancestors list must show the same
    # completed-with-warnings badge the detail view does, without loading any
    # ancestor's stage output blobs.
    chain_ids = [r.id for r in chain]
    degraded_ids = await RunRepository(session).degraded_run_ids(chain_ids)
    tokens = await RunUsageRepository(session).totals_for_runs(chain_ids)
    return [
        _summary(r, project_id, degraded=r.id in degraded_ids, tokens=tokens.get(r.id, (0, 0)))
        for r in chain
    ]
