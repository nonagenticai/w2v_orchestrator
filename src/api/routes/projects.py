"""Projects routes — list, create (atomic + N deliverables), get, patch, delete + children + events."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Annotated, Literal, cast
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request, Response, status
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from redis.exceptions import RedisError
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from src.api.artifact_url_rewriter import rewrite_minio_url
from src.api.routes._common import (
    DeliverableRepoDep,
    ProjectRepoDep,
    RunRepoDep,
    SessionDep,
    SubjectIdDep,
    TenantIdDep,
)
from src.api.routes.runs import _summary
from src.api.routes.runs.lifecycle import cancel_run_tree
from src.api.schemas import (
    AuditBrief,
    BranchAncestor,
    BranchContext,
    CreateChildProject,
    DeliverableCreate,
    DeliverableResponse,
    DeliverableStatus,
    FromIntentRequest,
    FromIntentResponse,
    ProjectCreate,
    ProjectDetail,
    ProjectDraftCreate,
    ProjectList,
    ProjectResponse,
    ProjectUpdate,
    ReorderChildren,
    RunStatus,
    RunSummary,
    ShowcaseBrief,
)
from src.api.schemas.projects import _BRIEF_DOOR_MODELS, Voice
from src.api.sse import stream_channel
from src.database.repositories.project_repository import ProjectRepository
from src.database.repositories.run_usage_repository import RunUsageRepository
from src.database.session import bind_session_tenant
from src.services.auth_secrets import (
    AuthSecretError,
    login_needs_reentry,
    redact_auth_secrets,
    restore_redacted,
    seal_auth_secrets,
)
from src.services.blueprints.audit_family import AUDIT_PROJECT_TYPES
from src.services.project_clone import (
    ClonedSettings,
    CloneSourceNotFoundError,
    load_clone_source,
    merge_under,
)
from src.services.stages.website_review.order import OrderBelowFloorError
from src.services.tenant_defaults import captions_mode as resolve_captions_mode
from src.services.tenant_defaults import load_tenant_defaults

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/projects", tags=["projects"])
# SSE stream router — mounted in asgi.py WITHOUT the tenant-gate dependencies
# (require_known_tenant / apply_tenant_rls), both of which resolve a request-scoped
# SessionDep held open for the whole streaming response (the prod pool-exhaustion
# bug). The stream route does its own tenant-ownership check on a short-lived,
# RLS-bound session instead.
sse_router = APIRouter(prefix="/api/projects", tags=["projects"])


async def _publish_project_event(
    request: Request, project_id: UUID, event: str, **payload: object
) -> None:
    """Best-effort publish of a child-lifecycle event to projects:<id>.

    Notifications are not source of truth — the mutation has already
    committed by the time we publish. If Redis is unavailable we log and
    swallow rather than 5xx the caller; subscribers reconnect
    (EventSource auto-reconnects) and resync via the next list/get fetch.
    """
    redis = request.app.state.redis
    if redis is None:
        return
    msg = {"event": event, "project_id": str(project_id), **payload}
    try:
        await redis.publish(f"projects:{project_id}", json.dumps(msg))
    except (RedisError, OSError) as exc:
        logger.warning("Failed to publish project event %s for %s: %s", event, project_id, exc)


def _sealed_brief(brief: dict) -> dict:
    """Seal the brief's login secrets before persisting; no key configured ⇒ 503.

    The ORM listener seals again as a backstop (idempotent); this call exists so a
    missing ``AUTH_CONFIG_SECRET_KEYS`` is a clean, named 503 at the door instead of a
    500 at flush. The message never carries the secret.
    """
    try:
        return seal_auth_secrets(brief)
    except AuthSecretError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc


def _to_response(p: object) -> ProjectResponse:
    resp = ProjectResponse.model_validate(p, from_attributes=True)
    # `source_url` is a typed projection of the project's primary URL. Promo
    # stores it under `brief.source_url`; website_review/app_review store the
    # review target under `brief.target_url`; `brief.url` is a legacy key still
    # seen in older payloads. We surface whichever is set so the gallery URL
    # chip works for every url-bearing type — mirroring the FE run page's
    # `brief.target_url ?? brief.source_url ?? brief.url` cascade. The frontend
    # was casting `brief` to `Record<string, unknown>` and probing it at
    # runtime — this stable field replaces that.
    if resp.source_url is None:
        brief = getattr(p, "brief", None) or {}
        if isinstance(brief, dict):
            candidate = brief.get("source_url") or brief.get("target_url") or brief.get("url")
            if isinstance(candidate, str) and candidate:
                resp.source_url = candidate
    # Typed showcase-brief projection (same pattern as source_url). Best-effort: a
    # legacy/partial brief that no longer validates must never 500 a read.
    if getattr(p, "type", None) == "showcase":
        brief = getattr(p, "brief", None) or {}
        if isinstance(brief, dict):
            try:
                # Redacted: this projection is on every LIST row, and the brief's
                # login password must never leave the service (auth_secrets).
                resp.showcase_brief = ShowcaseBrief.model_validate(redact_auth_secrets(brief))
            except ValidationError:
                resp.showcase_brief = None
    # Typed audit-brief projection (same best-effort pattern) for the audit types
    # (derived from the audit-family source of truth, not a hand-listed tuple).
    if getattr(p, "type", None) in AUDIT_PROJECT_TYPES:
        brief = getattr(p, "brief", None) or {}
        if isinstance(brief, dict):
            try:
                resp.audit_brief = AuditBrief.model_validate(redact_auth_secrets(brief))
            except ValidationError:
                resp.audit_brief = None
    return resp


def _to_deliv(d: object, status: DeliverableStatus) -> DeliverableResponse:
    # `deliverable.status` is derived on read from the latest run (the column is
    # dead); the caller passes the already-derived value.
    resp = DeliverableResponse.model_validate(d, from_attributes=True)
    resp.status = status
    return resp


@router.get("", response_model=ProjectList, operation_id="projects.list")
async def list_projects(
    tenant_id: TenantIdDep,
    project_repo: ProjectRepoDep,
    type_: Annotated[str | None, Query(alias="type")] = None,
    status_: Annotated[str | None, Query(alias="status")] = None,
    parent_id: Annotated[UUID | None, Query()] = None,
    include_archived: Annotated[bool, Query()] = False,
    sort_by: Annotated[
        Literal["created_at", "finished_at", "updated_at", "title", "status", "type"],
        Query(),
    ] = "created_at",
    order: Annotated[Literal["asc", "desc"], Query()] = "desc",
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ProjectList:
    res = await project_repo.list(
        tenant_id=tenant_id,
        type_=type_,
        status=status_,
        parent_id=parent_id,
        include_archived=include_archived,
        sort_by=sort_by,
        order=order,
        limit=limit,
        offset=offset,
    )
    items = []
    for p in res.items:
        r = _to_response(p)
        r.deliverable_count = res.deliverable_counts.get(p.id, 0)
        # DB CHECK constrains run.status to the RunStatus literal set.
        r.last_run_status = cast("RunStatus | None", res.last_run_statuses.get(p.id))
        # ck_runs_pause_reason constrains it to the PauseReason literal set.
        r.last_run_pause_reason = cast(
            "Literal['checkpoint', 'manual', 'unknown'] | None",
            res.last_run_pause_reasons.get(p.id),
        )
        thumb = res.thumbnails.get(p.id)
        if thumb is not None:
            run_id, raw_url = thumb
            # Rewrite minio.test → /api/runs/{id}/artifact/... so the browser
            # can load it (mirrors the run-detail output rewriting).
            r.thumbnail_url = rewrite_minio_url(run_id, raw_url)
        items.append(r)
    return ProjectList(items=items, total=res.total, limit=limit, offset=offset)


#: Response header carrying the TOTAL row count behind a paged list (the body stays
#: a bare JSON array, so the shape every existing client parses is unchanged).
TOTAL_COUNT_HEADER = "X-Total-Count"


async def _clone_source(
    project_repo: ProjectRepository, source_id: UUID, *, tenant_id: UUID, target_type: str
) -> ClonedSettings:
    """``services.project_clone.load_clone_source`` with its 404 mapped for HTTP."""
    try:
        return await load_clone_source(
            project_repo, source_id, tenant_id=tenant_id, target_type=target_type
        )
    except CloneSourceNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=exc.detail) from exc


def _cloned_voice(cloned: dict, explicit: dict | None) -> dict:
    """Merged voice, or the explicit one alone if the merge is not a valid voice.

    The source's voice passed the door when IT was created; the catalog can have
    shrunk since. A stale cloned id must not make the new project uncreatable.
    """
    merged = merge_under(cloned, explicit)
    try:
        Voice.model_validate(merged)
    except ValidationError, ValueError:
        logger.warning("clone: merged voice failed validation; keeping the explicit voice")
        return dict(explicit or {})
    return merged


async def _resolved_deliverables(
    session: AsyncSession, tenant_id: UUID, deliverables: Sequence[DeliverableCreate]
) -> list[dict[str, object]]:
    """The request's deliverables with ``captions_mode`` resolved: an explicit value
    wins, else the tenant's default captions mode, else ``"none"`` (P6)."""
    defaults = await load_tenant_defaults(session, tenant_id)
    return [
        {**d.model_dump(), "captions_mode": resolve_captions_mode(d.captions_mode, defaults)}
        for d in deliverables
    ]


@router.post(
    "",
    response_model=ProjectDetail,
    status_code=status.HTTP_201_CREATED,
    operation_id="projects.create",
)
async def create_project(
    body: ProjectCreate,
    tenant_id: TenantIdDep,
    subject_id: SubjectIdDep,
    project_repo: ProjectRepoDep,
    session: SessionDep,
) -> ProjectDetail:
    if body.type == "lesson" and body.parent_project_id is None:
        raise HTTPException(400, detail="lesson projects must have a parent_project_id")
    try:
        body.check_website_review_order()
    except OrderBelowFloorError as exc:
        # A TYPED 400, not pydantic's 422: the body carries the floor and the ordered
        # value so the wizard can say which number it has to change.
        raise HTTPException(400, detail=exc.refusal.model_dump()) from exc
    brief_in, brand, voice, theme = body.brief, body.brand, body.voice, body.theme
    if body.clone_from_project_id is not None:
        # X3 (UI review 2026-10-17): the wizard's "Clone settings from" was inert.
        # Cloned values are the BASE; every key the request carries wins.
        cloned = await _clone_source(
            project_repo, body.clone_from_project_id, tenant_id=tenant_id, target_type=body.type
        )
        brand = merge_under(cloned.brand, body.brand)
        theme = merge_under(cloned.theme, body.theme)
        voice = _cloned_voice(cloned.voice, body.voice)
        if cloned.brief:
            candidate = merge_under(cloned.brief, body.brief)
            try:
                # The merged brief must pass the SAME typed door the request did
                # (this also canonicalises its duration key in place).
                ProjectCreate.model_validate({**body.model_dump(), "brief": candidate})
                brief_in = candidate
            except ValidationError:
                logger.warning("clone: merged brief failed the door; keeping the request brief")
    brief = _sealed_brief(brief_in)
    deliverables = await _resolved_deliverables(session, tenant_id, body.deliverables)
    try:
        proj = await project_repo.create_with_deliverables(
            tenant_id=tenant_id,
            owner_subject_id=subject_id,
            type_=body.type,
            title=body.title,
            brief=brief,
            brand=brand,
            voice=voice,
            theme=theme,
            visibility=body.visibility,
            parent_project_id=body.parent_project_id,
            deliverables=deliverables,
        )
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    except (IntegrityError, DBAPIError) as exc:
        msg = str(exc.orig) if exc.__cause__ else str(exc)
        if "lesson" in msg.lower() or "parent" in msg.lower():
            raise HTTPException(400, detail="lesson must have a parent of type course") from exc
        raise
    # COMMIT BEFORE RESPONDING. `create_with_deliverables` only FLUSHes, and the
    # session dependency commits in its ``yield`` teardown — which FastAPI (0.136)
    # runs AFTER the response is sent. Without this the caller is handed a 201 and
    # an id for a row that is not durable yet, so its very next request reads a DB
    # where the project does not exist and gets `404 project not found`.
    #
    # At idle the teardown commit wins that race in microseconds, which is why this
    # never showed up locally; under the live 33-scenario e2e matrix it loses, and
    # `storyboard-merge` / `panel-image` both failed exactly this way (the row
    # present afterwards, tenant correct, the 404 returned in 6 ms). One of them
    # passed on a re-run with no code change — the signature of a race.
    #
    # This mirrors what `routes/runs/lifecycle.py` and `runs/stages.py` already do
    # at every mutation; the teardown commit stays as the safety net.
    await session.commit()
    base = _to_response(proj).model_dump()
    return ProjectDetail(
        **base,
        brief=proj.brief,
        brand=proj.brand,
        voice=proj.voice,
        theme=proj.theme,
        # Freshly created — no runs yet, so every deliverable is idle.
        deliverables=[_to_deliv(d, "idle") for d in proj.deliverables],
        children=[],
        open_notes_count=0,
    )


@router.post(
    "/from-intent",
    response_model=FromIntentResponse,
    status_code=status.HTTP_201_CREATED,
    operation_id="projects.fromIntent.create",
)
async def create_project_from_intent(
    body: FromIntentRequest,
    request: Request,
    tenant_id: TenantIdDep,
    subject_id: SubjectIdDep,
    project_repo: ProjectRepoDep,
    session: SessionDep,
) -> FromIntentResponse:
    """A0 authoring front door (great-video PR 2.2): a ``VideoIntent`` → a project +
    deliverable + started run. The intent is materialized into a computed
    ``PlanDocument`` at run-start (``build_config_snapshot`` → ``resolve_plan``); the
    existing DAG renders it, so an author supplies ~15 lines of intent instead of a
    hand-tuned brief.

    Fail-fast: an authored-only ``showcase`` needs footage — without a resolvable
    ``package`` the resolver leaves ``lines`` absent and the run has nothing to
    assemble, so reject at the door with 422 rather than letting it die stage-deep at
    ``demo_video_record``. Automatic recording from ``target_url`` is deferred (PR 2.3).
    """
    intent = body.intent
    if intent.type == "showcase" and not intent.package:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                "a showcase intent requires a footage `package` (uploads://<ref> or a "
                "URL); automatic recording from target_url is not yet available"
            ),
        )
    brief: dict[str, object] = {
        "target_url": intent.target_url,
        "intent": intent.model_dump(exclude_none=True),
    }
    if intent.package:
        brief["package_url"] = intent.package
    try:
        proj = await project_repo.create_with_deliverables(
            tenant_id=tenant_id,
            owner_subject_id=subject_id,
            type_=intent.type,
            title=intent.title,
            brief=brief,
            brand={},
            voice={},
            theme={},
            visibility="tenant",
            parent_project_id=None,
            deliverables=[
                {
                    "name": intent.title,
                    "format": body.format,
                    "captions_mode": resolve_captions_mode(
                        body.captions_mode, await load_tenant_defaults(session, tenant_id)
                    ),
                    "target_duration_seconds": (
                        body.target_duration_seconds or intent.duration_seconds
                    ),
                }
            ],
        )
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    # Commit BEFORE starting the run — see `create_project`. Here it is not only the
    # caller's next request that would race the teardown commit: `RunEngine.start_run`
    # opens its OWN session (`self._sessions()`), so a merely-flushed deliverable is
    # invisible to it and it raises "deliverable not found or not visible to tenant".
    await session.commit()
    deliverable = proj.deliverables[0]
    engine = request.app.state.run_engine
    try:
        run = await engine.start_run(
            deliverable_id=deliverable.id,
            tenant_id=tenant_id,
            subject_id=subject_id,
            parent_run_id=None,
            pause_after_stages=[] if body.autopilot else None,
        )
    except OrderBelowFloorError as exc:
        # BEFORE the generic arm — it is a ValueError. The from-intent door is the
        # THIRD door the floor can refuse at, and it was the only one still answering
        # with `str(exc)`: an unbranchable string that `api/error_details.py` cannot
        # rescue, because the stringification happens HERE, at the raise site, not in
        # the envelope wrapper. Same typed body as `create_project` / `start_run`.
        raise HTTPException(400, detail=exc.refusal.model_dump()) from exc
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    return FromIntentResponse(
        project_id=proj.id,
        deliverable_id=deliverable.id,
        run_id=run.id,
        plan_preview_url=f"/api/runs/{run.id}",
        events_url=f"/api/runs/{run.id}/events",
        pause_after_stages=list(run.pause_after_stages or []),
    )


@router.post(
    "/draft",
    response_model=ProjectDetail,
    status_code=status.HTTP_201_CREATED,
    operation_id="projects.draft.create",
)
async def create_draft_project(
    body: ProjectDraftCreate,
    tenant_id: TenantIdDep,
    subject_id: SubjectIdDep,
    project_repo: ProjectRepoDep,
    session: SessionDep,
) -> ProjectDetail:
    """Create a real project EARLY (wizard's storyboard step already chose the type)
    with NO deliverables yet — status server-defaults to 'drafting'. The wizard later
    adds deliverables + PATCHes + starts runs via the existing endpoints. Additive: the
    strict ``ProjectCreate`` (normal final-submit / n8n) is untouched.

    An unknown ``type`` 422s at the ``ProjectDraftCreate`` Literal door (the DB CHECK is
    the backstop); a lesson without a parent surfaces as 400 from the repository guard.
    """
    seed: ClonedSettings | None = None
    if body.clone_from_project_id is not None:
        seed = await _clone_source(
            project_repo, body.clone_from_project_id, tenant_id=tenant_id, target_type=body.type
        )
    try:
        proj = await project_repo.create_draft(
            tenant_id=tenant_id,
            owner_subject_id=subject_id,
            type_=body.type,
            title=body.title,
            visibility=body.visibility,
            brief=_sealed_brief(seed.brief) if seed else None,
            brand=seed.brand if seed else None,
            voice=seed.voice if seed else None,
            theme=seed.theme if seed else None,
        )
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    await session.commit()  # durable before the id is handed out — see `create_project`
    base = _to_response(proj).model_dump()
    return ProjectDetail(
        **base,
        brief=proj.brief,
        brand=proj.brand,
        voice=proj.voice,
        theme=proj.theme,
        # Freshly drafted — no deliverables, no children, no runs yet.
        deliverables=[],
        children=[],
        open_notes_count=0,
    )


@router.post(
    "/{project_id}/deliverables",
    response_model=DeliverableResponse,
    status_code=status.HTTP_201_CREATED,
    operation_id="projects.deliverables.create",
)
async def add_project_deliverable(
    project_id: UUID,
    body: DeliverableCreate,
    tenant_id: TenantIdDep,
    project_repo: ProjectRepoDep,
    session: SessionDep,
) -> DeliverableResponse:
    """Add ONE deliverable to an existing project — the create-project wizard's
    finalize step (``POST /api/projects/draft`` → PATCH → *this* → start runs).

    The draft door (``projects.draft.create``) promises the wizard adds deliverables
    "via the existing endpoints"; this is that endpoint. Additive: the atomic
    ``projects.create`` path and its strict ``ProjectCreate`` (>=1 deliverable, typed
    brief door) are untouched, and the request body is the SAME ``DeliverableCreate``
    both that path and ``projects.children.create`` already accept, so the wizard
    sends one shape everywhere.

    404 — never 403 — for an unknown OR another tenant's project (the repository
    resolves ownership tenant-scoped and returns None for both), matching every other
    project sub-route so a foreign tenant cannot probe project existence.
    """
    project = await project_repo.get(project_id, tenant_id=tenant_id)
    if project is None:
        raise HTTPException(404, detail="project not found")
    if not project.deliverables:
        # The draft -> project transition: the brief the wizard PATCHed in pieces
        # must now pass the same typed door ``projects.create`` enforces.
        _enforce_brief_door(project.type, project.brief or {})
    deliverable = await project_repo.add_deliverable(
        project_id,
        tenant_id=tenant_id,
        name=body.name,
        format_=body.format,
        captions_mode=resolve_captions_mode(
            body.captions_mode, await load_tenant_defaults(session, tenant_id)
        ),
        target_duration_seconds=body.target_duration_seconds,
    )
    if deliverable is None:
        raise HTTPException(404, detail="project not found")
    # COMMIT BEFORE RESPONDING — same race as projects.create (see the long comment
    # there): the repository only FLUSHes and the session dependency commits in its
    # teardown, which FastAPI runs AFTER the response is sent. The wizard's very next
    # call reads this deliverable back (and starts a run on it), so handing out an id
    # for a not-yet-durable row is exactly the 404 that fix is about.
    await session.commit()
    # Freshly created — no runs yet, so the derived status is idle.
    return _to_deliv(deliverable, "idle")


@router.get("/{project_id}", response_model=ProjectDetail, operation_id="projects.get")
async def get_project(
    project_id: UUID,
    tenant_id: TenantIdDep,
    project_repo: ProjectRepoDep,
    deliverable_repo: DeliverableRepoDep,
) -> ProjectDetail:
    proj = await project_repo.get_with_children(project_id, tenant_id=tenant_id)
    if proj is None:
        raise HTTPException(404, detail="project not found")
    open_notes = await project_repo.open_notes_count(project_id, tenant_id=tenant_id)
    # One windowed query for all deliverables' derived statuses (no N+1).
    statuses = await deliverable_repo.derived_statuses([d.id for d in proj.deliverables])
    # A branch is a real but HIDDEN project row: it is excluded from the LIST yet
    # readable by id, so a user can land in one with nothing telling them where they
    # are and no way back to the root. Carry the linkage on the detail read. The extra
    # join is paid ONLY by branch projects (`is_branch` is false for every normal one),
    # and it is tenant-scoped via the root project — a foreign tenant's branch never
    # surfaces here (its child project is already a 404 above).
    branch: BranchContext | None = None
    if proj.is_branch:
        ctx = await project_repo.branch_context(project_id, tenant_id=tenant_id)
        if ctx is not None:
            branch = BranchContext(
                branch_id=ctx.branch_id,
                name=ctx.name,
                status=ctx.status,
                root_project_id=ctx.root_project_id,
                root_title=ctx.root_title,
                ancestors=[
                    BranchAncestor(project_id=a.project_id, title=a.title, is_branch=a.is_branch)
                    for a in ctx.ancestors
                ],
            )
    base = _to_response(proj).model_dump()
    return ProjectDetail(
        **base,
        brief=proj.brief,
        brand=proj.brand,
        voice=proj.voice,
        theme=proj.theme,
        deliverables=[_to_deliv(d, statuses[d.id]) for d in proj.deliverables],
        children=[_to_response(c) for c in proj.children],
        open_notes_count=open_notes,
        branch=branch,
    )


@router.get(
    "/{project_id}/runs",
    response_model=list[RunSummary],
    operation_id="projects.runs.list",
)
async def list_project_runs(
    project_id: UUID,
    response: Response,
    tenant_id: TenantIdDep,
    project_repo: ProjectRepoDep,
    run_repo: RunRepoDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[RunSummary]:
    # Run history for one project, across all its deliverables, newest-first.
    # The workspace History section reads this; the project list only carries
    # last_run_status. 404 on unknown/cross-tenant so a foreign tenant can't
    # probe run existence.
    proj = await project_repo.get(project_id, tenant_id=tenant_id)
    if proj is None:
        raise HTTPException(404, detail="project not found")
    runs = await run_repo.list_for_project(
        project_id, tenant_id=tenant_id, limit=limit, offset=offset
    )
    # The page is capped (default 50) and the body is a bare array, so a busy
    # project's "Total runs" read len(page) and silently stopped at 50 (UI review
    # 2026-10-17, M6). The total rides a header — additive, body shape unchanged.
    response.headers[TOTAL_COUNT_HEADER] = str(
        await run_repo.count_for_project(project_id, tenant_id=tenant_id)
    )
    # A run that finished green but soft-failed downstream work must read as
    # "completed with warnings" HERE too, not only in the detail view — the
    # History list was showing a plain "Completed" for a broken deliverable.
    # One extra aggregate query for the whole page; no stage output blobs.
    run_ids = [run.id for run in runs]
    degraded_ids = await run_repo.degraded_run_ids(run_ids)
    # Token totals for the whole page in ONE grouped query (no N+1) — the
    # deliverable page showed no tokens because the summary carried none. Same
    # tenant-bound session as run_repo, so RLS applies identically.
    tokens = await RunUsageRepository(run_repo.session).totals_for_runs(run_ids)
    return [
        _summary(
            run,
            project_id,
            degraded=run.id in degraded_ids,
            tokens=tokens.get(run.id, (0, 0)),
        )
        for run in runs
    ]


def _enforce_brief_door(project_type: str, brief: dict) -> None:
    """Run ``ProjectCreate``'s typed-brief door against a brief that reached the DB
    by another path, 422-ing with the SAME body shape the create door produces.

    WHEN it runs (the rule): a wizard draft is born with no deliverables and is
    PATCHed incrementally while the operator types, so a partial brief there is
    legal. The draft becomes a real project when its FIRST deliverable is added
    (``POST /{id}/deliverables``) — that is the transition, and it validates the
    stored brief. That transition is the ONLY place it runs: later brief PATCHes
    stay unvalidated, because live projects carry legacy partial briefs (e.g. a
    showcase with no ``target_url``) that must remain editable.
    """
    model = _BRIEF_DOOR_MODELS.get(project_type)
    if model is None:
        return
    try:
        model.model_validate(brief)
    except ValidationError as exc:
        raise RequestValidationError(
            [{**err, "loc": ("body", *err["loc"])} for err in exc.errors(include_url=False)]
        ) from exc


@router.patch(
    "/{project_id}",
    response_model=ProjectResponse,
    operation_id="projects.update",
)
async def patch_project(
    project_id: UUID,
    body: ProjectUpdate,
    tenant_id: TenantIdDep,
    project_repo: ProjectRepoDep,
    session: SessionDep,
) -> ProjectResponse:
    fields = body.model_dump(exclude_unset=True)
    # Never a column: popped unconditionally so the repository never setattr's it.
    clone_from = fields.pop("clone_from_project_id", None)
    current = None
    if isinstance(fields.get("brief"), dict) or clone_from is not None:
        current = await project_repo.get(project_id, tenant_id=tenant_id)
        if current is None:
            raise HTTPException(404, detail="project not found")
    if isinstance(fields.get("brief"), dict) and current is not None:
        # A client that read the brief got REDACTED in place of each login secret;
        # echoing it back must keep the stored secret, not overwrite it with a marker.
        fields["brief"] = _sealed_brief(restore_redacted(fields["brief"], current.brief))
        # A login whose address / username changed loses its saved secret (it was
        # approved for the old one). Logged — never the secret — so a "my password
        # vanished" report is answerable; the Brief page tells the user to re-enter.
        if dropped := login_needs_reentry(fields["brief"]):
            logger.info(
                "project %s: saved sign-in secret cleared for re-entry at %s (login changed)",
                project_id,
                dropped,
            )
    if clone_from is not None and current is not None:
        # The draft-finalize door for X3: the wizard's draft already exists, so its
        # "Clone settings from" lands on the finalize PATCH. Cloned values are the
        # base; this PATCH's fields win, and for a field it omits the project's
        # CURRENT value wins (a clone never overwrites something already set).
        cloned = await _clone_source(
            project_repo, clone_from, tenant_id=tenant_id, target_type=current.type
        )

        def _own(key: str) -> dict | None:
            value = fields.get(key)
            return value if isinstance(value, dict) else getattr(current, key)

        fields["brand"] = merge_under(cloned.brand, _own("brand"))
        fields["theme"] = merge_under(cloned.theme, _own("theme"))
        fields["voice"] = _cloned_voice(cloned.voice, _own("voice"))
        if cloned.brief:
            fields["brief"] = merge_under(cloned.brief, _own("brief"))
    proj = await project_repo.update(project_id, tenant_id=tenant_id, fields=fields)
    if proj is None:
        raise HTTPException(404, detail="project not found")
    await session.commit()  # durable before responding — see `create_project`
    return _to_response(proj)


@router.delete(
    "/{project_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    operation_id="projects.delete",
)
async def delete_project(
    project_id: UUID,
    request: Request,
    tenant_id: TenantIdDep,
    project_repo: ProjectRepoDep,
    run_repo: RunRepoDep,
    session: SessionDep,
) -> None:
    if await project_repo.get(project_id, tenant_id=tenant_id) is None:
        raise HTTPException(404, detail="project not found")
    # STOP SPEND FIRST (UI review 2026-10-17): delete only archived the row, so a
    # "producing" project vanished from the list while its run kept rendering and
    # billing, invisible. Every pending/running/PAUSED run goes through the exact
    # path ``POST /runs/{id}/cancel`` uses (descendants + course lesson children +
    # compile cancel included).
    dispatcher = request.app.state.dispatcher
    degraded = False
    for run in await run_repo.list_active_for_project(project_id, tenant_id=tenant_id):
        if not await cancel_run_tree(session, dispatcher, run):
            degraded = True
    if degraded:
        # Same contract as the cancel route: the cancellations are RECORDED
        # (committed) but no live interrupt could be delivered. Do not archive —
        # the client retries the delete, which then finds nothing active.
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "runs cancelled but live interrupt unavailable — dispatcher not ready; retry the delete",
        )
    ok = await project_repo.soft_delete(project_id, tenant_id=tenant_id)
    if not ok:
        raise HTTPException(404, detail="project not found")
    await session.commit()  # durable before the 204 — see `create_project`


@router.get(
    "/{project_id}/children",
    response_model=list[ProjectResponse],
    operation_id="projects.children.list",
)
async def list_children(
    project_id: UUID,
    tenant_id: TenantIdDep,
    project_repo: ProjectRepoDep,
) -> list[ProjectResponse]:
    proj = await project_repo.get_with_children(project_id, tenant_id=tenant_id)
    if proj is None:
        raise HTTPException(404, detail="project not found")
    return [_to_response(c) for c in proj.children]


@router.post(
    "/{project_id}/children",
    response_model=ProjectResponse,
    status_code=status.HTTP_201_CREATED,
    operation_id="projects.children.create",
)
async def create_child(
    project_id: UUID,
    body: CreateChildProject,
    request: Request,
    tenant_id: TenantIdDep,
    subject_id: SubjectIdDep,
    project_repo: ProjectRepoDep,
    session: SessionDep,
) -> ProjectResponse:
    parent = await project_repo.get(project_id, tenant_id=tenant_id)
    if parent is None:
        raise HTTPException(404, detail="parent project not found")
    brief = _sealed_brief(body.brief)
    deliverables = await _resolved_deliverables(session, tenant_id, body.deliverables)
    try:
        proj = await project_repo.create_with_deliverables(
            tenant_id=tenant_id,
            owner_subject_id=subject_id,
            type_=body.type,
            title=body.title,
            brief=brief,
            # Inherit the parent's branding so a course's lessons render on-brand;
            # an explicit per-child override (even {}) wins over inheritance.
            brand=body.brand if body.brand is not None else dict(parent.brand),
            voice=body.voice if body.voice is not None else dict(parent.voice),
            theme=body.theme if body.theme is not None else dict(parent.theme),
            visibility="tenant",
            parent_project_id=project_id,
            deliverables=deliverables,
        )
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    # Commit BEFORE publishing (see `create_project` for why the commit exists at all):
    # ``_publish_project_event``'s contract is that the mutation is already durable when
    # the event goes out, and a subscriber refetches the child by id the moment it lands.
    await session.commit()
    # Notify subscribers on the parent's channel — payload is intentionally
    # minimal (id only); the FE refetches the full child via projects.get().
    await _publish_project_event(request, project_id, "child_added", child_id=str(proj.id))
    return _to_response(proj)


@router.patch(
    "/{project_id}/children/reorder",
    status_code=status.HTTP_204_NO_CONTENT,
    operation_id="projects.children.reorder",
)
async def reorder_children(
    project_id: UUID,
    body: ReorderChildren,
    request: Request,
    tenant_id: TenantIdDep,
    project_repo: ProjectRepoDep,
    session: SessionDep,
) -> None:
    ok = await project_repo.reorder_children(
        project_id, body.ordered_child_ids, tenant_id=tenant_id
    )
    if not ok:
        raise HTTPException(400, detail="invalid child id in ordering")
    await session.commit()  # durable before the event goes out — see `create_child`
    await _publish_project_event(
        request,
        project_id,
        "child_reordered",
        order=[str(cid) for cid in body.ordered_child_ids],
    )


@sse_router.get(
    "/{project_id}/events",
    operation_id="projects.events",
    # Exclude from generated OpenAPI clients — sse-starlette doesn't pin a
    # response schema and the FE consumes this via EventSource, not a
    # typed client method.
    include_in_schema=False,
)
async def project_events(
    project_id: UUID,
    request: Request,
    # why: an EventSource cannot set the X-Active-Tenant-Id header — the
    # scope-switch rides ?active_tenant=, which get_tenant_id also reads.
    tenant_id: TenantIdDep,
) -> EventSourceResponse:
    """Subscribe to child-lifecycle events for *project_id*.

    Tenant-checked: caller's tenant must own the project. Other-tenant +
    nonexistent both yield 404 (same status code — no info leak across
    the tenant boundary). Stream events: ``child_added``,
    ``child_reordered``, ``child_status_changed``. The status event is
    emitted by ``RunDispatcher`` whenever a run on a child project
    transitions (running/paused/completed/failed/cancelled).

    The ownership check runs on a SHORT-LIVED, RLS-bound session (this route is
    mounted off the tenant-gate — see asgi.py) so the SSE stream does not pin a
    request-scoped DB connection for its lifetime.
    """
    factory = request.app.state.session_factory
    async with factory() as session:
        await bind_session_tenant(session, tenant_id)
        project = await ProjectRepository(session).get(project_id, tenant_id=tenant_id)
    if project is None:
        raise HTTPException(404, detail="project not found")
    return await stream_channel(request, channel=f"projects:{project_id}")
