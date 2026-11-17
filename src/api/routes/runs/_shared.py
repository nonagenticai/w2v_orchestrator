"""Shared routers, ownership/summary helpers, and constants for the runs routes.

The two ``APIRouter`` instances live here; the sub-modules (read / events /
lifecycle / stages) import and decorate them, and the package ``__init__``
imports the sub-modules for their registration side effects.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.artifact_url_rewriter import internal_minio_ref, rewrite_payload
from src.api.schemas import RunSummary, StageOutputResponse
from src.config import get_settings
from src.database.models import Deliverable, Project, Run, StageOutput
from src.database.repositories.project_repository import ProjectRepository
from src.services.blueprints.audit_family import AUDIT_FAMILIES
from src.services.blueprints.base import DAGBlueprint
from src.services.runs.elements import populated_element_paths
from src.services.stages.shared.degradation import is_degraded

if TYPE_CHECKING:
    from src.services.run_dispatcher import RunDispatcher

router = APIRouter(prefix="/api/runs", tags=["runs"])

# The audit scan stages whose structured output backs the downloadable PDF report.
# Derived from the audit family list (seo_audit / cro_audit / marketing_scorecard, …)
# so a new scorecard type gets its PDF for free — no drift between the blueprint
# registry and the download whitelist.
_AUDIT_SCAN_STAGES = tuple(f"{f.project_type}_scan" for f in AUDIT_FAMILIES)
# SSE stream router — mounted in asgi.py WITHOUT the tenant-gate dependencies
# (require_known_tenant / apply_tenant_rls), both of which resolve a request-scoped
# SessionDep that FastAPI holds open for the *whole streaming response*, pinning a
# pool connection for the stream's lifetime (the prod pool-exhaustion bug). The
# stream route instead does its own tenant-ownership check on a short-lived session
# (stricter than require_known_tenant) and binds RLS on that same session.
sse_router = APIRouter(prefix="/api/runs", tags=["runs"])

# Hard ceiling on the parent_run_id ancestor walk (paired with a visited-set):
# parent links are not constrained acyclic at write time.
_MAX_ANCESTORS = 1000

_COURSE_STITCH = "course_stitch"


def _summary(
    run: Run,
    project_id: UUID,
    *,
    degraded: bool = False,
    tokens: tuple[int, int] = (0, 0),
) -> RunSummary:
    """Project a Run row onto the wire summary.

    ``degraded`` is a caller-supplied aggregate rather than something derived
    here, because deriving it needs the run's stage_outputs and the list paths
    must not load those blobs — see
    ``RunRepository.degraded_run_ids`` for the one-query batch form. It defaults
    to False so a just-created run (no stage rows yet) is reported clean.

    ``tokens`` is ``(input_tokens, output_tokens)``, likewise caller-supplied: list
    paths batch it via ``RunUsageRepository.totals_for_runs`` (one grouped query
    per page, never per run).
    """
    return RunSummary(
        id=run.id,
        project_id=project_id,
        deliverable_id=run.deliverable_id,
        parent_run_id=run.parent_run_id,
        blueprint_id=run.blueprint_id,
        status=run.status,  # ty: ignore[invalid-argument-type]  # why: SA mapped column → str at runtime; pydantic Literal narrowing happens at construction
        current_stage=run.current_stage,
        started_at=run.started_at,
        completed_at=run.completed_at,
        created_at=run.created_at,
        # A cancel/fail leaves ``runs.paused_at`` stamped; only a live pause has one.
        paused_at=run.paused_at if run.status == "paused" else None,
        pause_reason=run.pause_reason if run.status == "paused" else None,  # ty: ignore[invalid-argument-type]  # why: SA mapped column → str at runtime; ck_runs_pause_reason keeps it in the Literal
        degraded=degraded,
        input_tokens=tokens[0],
        output_tokens=tokens[1],
    )


def _degradation_codes(output: object) -> list[str]:
    """The stable degradation codes carried by a PERSISTED stage output blob.

    The read-side companion to ``services/stages/shared/degradation.py``: that
    module defines nine codes and every one of them is emitted, but the only
    thing any reader could ask was the boolean ``is_degraded``. This returns
    WHICH ones, so "a failed mastering gate" and "a video that shipped silent"
    stop being the same wire value.

    Deliberately tolerant of shape, for the same reason ``is_degraded`` is: the
    blob may be ``None``, may lack the key (every row written before the channel
    existed), may carry a JSON ``null``, and — during a rolling deploy — may
    carry a code this build has never heard of. None of those may raise; a
    quality signal must not be able to 500 the endpoint that reports it. Entries
    that are not a mapping, or whose ``code`` is not a non-empty string, are
    skipped rather than reported as garbage.

    De-duplicates while PRESERVING first-seen order. ``dict.fromkeys`` and not a
    ``set``: one code raised by two steps of a stage is one fact, and the report
    order must be deterministic (a set's iteration order is not).

    Returns ``[]``, never ``None`` — so a consumer can iterate unconditionally.
    """
    if not isinstance(output, dict):
        return []
    entries = output.get("degradations")
    if not isinstance(entries, list):
        return []
    codes = [
        entry["code"]
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("code"), str) and entry["code"] != ""
    ]
    return list(dict.fromkeys(codes))


def _stage_response(run_id: UUID, so: StageOutput) -> StageOutputResponse:
    """Build a StageOutputResponse with proxied artifact URLs.

    Stage outputs persist presigned ``minio.test:8389`` URLs because the
    in-cluster video-compile service fetches the bytes via cluster DNS. The
    browser can't reach that host, so on the way out we rewrite any
    ``http://minio.test/...`` value to ``/api/runs/{run_id}/artifact/...``
    — the proxy endpoint streams the bytes back through the orchestrator
    while reusing AuthMiddleware for tenant isolation. The DB row is
    unchanged so downstream stage handlers keep using the original URLs.
    """
    response = StageOutputResponse.model_validate(so, from_attributes=True)
    # A `completed` stage soft-failed a downstream step when EITHER channel is
    # populated — surface it as `degraded` so a green-but-partial run is not
    # mistaken for a clean one:
    #   * `degradations` — the explicit, machine-readable quality channel
    #     (`shared/degradation.py`): a stable code plus a detail. This is where
    #     a failed mastering gate, an out-of-tolerance delivered duration,
    #     dropped captions, a missing sound bed, a stubbed deck and an
    #     unnormalized clip now land.
    #   * `notes` — the legacy PROSE channel, kept in the OR so nothing that
    #     reads degraded today silently becomes clean.
    # The rule itself lives in `degradation.is_degraded` (SQL twin:
    # `run_repository.degraded_stage_clause`) so the three cannot drift.
    response.degraded = so.status == "completed" and is_degraded(so.output)
    # ADDITIVE companion to the boolean above: WHICH degradations, by stable
    # code. Read straight off the persisted blob — the same bytes `output`
    # already ships — so this is a promotion of data that was always on the
    # wire, not a new fact and not a migration. It is NOT gated on
    # `status == "completed"` the way `degraded` is: gating would DROP a code
    # that is plainly visible in this very response's `output`, making the named
    # field less truthful than the blob it mirrors. `degraded` keeps its exact
    # existing semantics; nothing here can change it.
    response.degradation_codes = _degradation_codes(so.output)
    response.element_paths = (
        populated_element_paths(so.output, so.stage_name) if isinstance(so.output, dict) else []
    )
    if response.output:
        # model_validate parsed the JSONB column into a fresh dict, so the
        # in-place rewrite below does not touch the SQLAlchemy attribute.
        rewrite_payload(run_id, response.output)
    return response


#: A handle string that is PRESENT but means "absent". Stage outputs really do
#: carry these (a JSON ``null`` stringified, an empty field a stage filled in
#: defensively), and treating one as a URL would make a run look like it shipped
#: a file called "none". Byte-parity with ``final_critique._ABSENT_URL_SENTINELS``
#: is asserted by ``tests/api/routes/test_delivered_artifact.py`` — the API and
#: the pipeline must not disagree about what counts as an artifact.
_ABSENT_ARTIFACT_URL_SENTINELS: frozenset[str] = frozenset({"none", "null", "nil", "undefined", ""})


def _usable_artifact_url(raw: object) -> str:
    """A non-empty, non-sentinel artifact URL string — else ``""`` (⇒ treated as absent).

    Same predicate as ``final_critique._usable_artifact_url``; see the sentinel
    set above for why it is restated here rather than imported (the API layer
    must not import a stage module) and how the two are kept identical.
    """
    if not isinstance(raw, str):
        return ""
    url = raw.strip()
    return "" if url.lower() in _ABSENT_ARTIFACT_URL_SENTINELS else url


def _resolve_delivered_artifact_url(
    blueprint: DAGBlueprint, stage_responses: Sequence[StageOutputResponse | StageOutput]
) -> str | None:
    """The URL of the artifact this run actually DELIVERS, or ``None``.

    Walks ``blueprint.delivered_artifact_chain()`` — most downstream first — and
    returns the first completed stage that published a usable
    ``terminal_artifact_field``. That chain is derived from the DAG, so this is
    the SAME resolution the pipeline performs: on ``demo_video_v1`` it reads
    ``final_critique`` → ``demo_video_encode_master`` →
    ``demo_video_footage_critique`` → ``demo_video_compile``, and each
    skipped-clean stage falls through byte-identically because a skip publishes a
    falsy ``{}``.

    Reads the ALREADY-REWRITTEN ``stage_responses``, not the raw rows, so the URL
    served here is the same proxy URL ``stage_outputs`` serves for that stage. A
    consumer that compares the two must not be told two different strings for one
    file (an E2E spec once asserted exactly that mismatch and failed on it).

    ``None`` when no stage in the chain has completed with an artifact yet — an
    honest "not delivered", never a guess from a look-alike stage.

    Also accepts the RAW ``StageOutput`` rows (same ``stage_name``/``status``/
    ``output`` shape): the YouTube publish route needs the un-rewritten
    ``minio.test`` URL to re-sign, and must pick the SAME stage this resolves.
    """
    field = blueprint.terminal_artifact_field
    by_stage = {sr.stage_name: sr for sr in stage_responses}
    for name in blueprint.delivered_artifact_chain():
        sr = by_stage.get(name)
        if sr is None or sr.status != "completed" or not isinstance(sr.output, dict):
            continue
        url = _usable_artifact_url(sr.output.get(field))
        if url:
            return url
    return None


async def _project_id_for_deliverable(session: AsyncSession, deliverable_id: UUID) -> UUID:
    stmt = select(Deliverable.project_id).where(Deliverable.id == deliverable_id)
    project_id = (await session.execute(stmt)).scalar_one_or_none()
    if project_id is None:
        # The run's deliverable→project chain is gone (deleted mid-flight, or a
        # genuinely orphaned run). ``scalar_one`` raised an uncaught
        # NoResultFound → 500 here; surface a clean 404 instead. Callers all
        # reach this only AFTER _tenant_owns_run confirmed ownership, so a miss
        # is a vanished-chain edge, not an auth gap.
        raise HTTPException(404, "run not found")
    return project_id


async def _tenant_owns_run(session: AsyncSession, run_id: UUID, tenant_id: UUID) -> Run | None:
    """Returns the Run if the caller's tenant owns it, else None."""
    stmt = (
        select(Run, Project.tenant_id)
        .join(Deliverable, Deliverable.id == Run.deliverable_id)
        .join(Project, Project.id == Deliverable.project_id)
        .where(Run.id == run_id)
    )
    row = (await session.execute(stmt)).first()
    if row is None:
        return None
    run, owner_tenant = row
    if owner_tenant != tenant_id:
        return None
    return run


async def _recompute_project_status_for_run(session: AsyncSession, deliverable_id: UUID) -> None:
    """Re-derive the owning project's (and parent course's) stored status from a
    run's deliverable — mirrors RunDispatcher._recompute_project. Used by the
    cancel route when no pod owns the task, so the dispatcher's transition
    handler can't do it. No-op if the project can't be resolved (defensive)."""
    proj_row = (
        await session.execute(
            select(Project.id, Project.parent_project_id)
            .join(Deliverable, Deliverable.project_id == Project.id)
            .where(Deliverable.id == deliverable_id)
        )
    ).first()
    if proj_row is None:
        return
    repo = ProjectRepository(session)
    await repo.recompute_status(proj_row.id)
    if proj_row.parent_project_id is not None:
        await repo.recompute_status(proj_row.parent_project_id)


def _require_dispatcher(request: Request) -> RunDispatcher:
    """Return the run dispatcher, or 503 if it never started (Redis down).

    When Redis is unavailable at startup ``asgi.py`` sets
    ``app.state.dispatcher = None`` (degraded mode; ``/readyz`` is already red).
    Reading it unguarded would raise an opaque 500 ``AttributeError`` on the
    control routes. A clear 503 tells the caller run control is temporarily
    unavailable rather than masking it as a server error.
    """
    dispatcher = request.app.state.dispatcher
    if dispatcher is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "run control unavailable — dispatcher not ready (Redis down)",
        )
    return dispatcher


def _chapter_key(c: dict) -> str | None:
    """The stable override key for a stored chapter row.

    Prefer the explicit ``chapter_key`` emitted by ``course_stitch`` (bare
    ``lesson_project_id`` at level 0, ``f"{lesson_project_id}#{ordinal}"`` at
    level 1). Fall back to ``lesson_project_id`` for level-0 rows persisted
    before ``chapter_key`` existed, so older runs stay editable at the lesson
    level (their slides simply aren't individually addressable). ``None`` = the
    row has no stable key and is not editable."""
    key = c.get("chapter_key")
    if key:
        return str(key)
    if int(c.get("level") or 0) == 0 and c.get("lesson_project_id"):
        return str(c.get("lesson_project_id"))
    return None


def _is_presignable_artifact_url(url: str) -> bool:
    """True iff *url* is an artifact object in OUR OWN storage.

    Accepts a ``{bucket}/{key}`` object on the cluster-internal MinIO host — the
    shape ``StorageService.resign_presigned_url`` re-signs — or, on the local
    backend only, a ``file://`` artifact inside the configured ``local_path``.

    The host check is the load-bearing part and it used to be missing: the scheme
    test alone accepted ``https://attacker.example/b/k``, so an override could
    point the YouTube thumbnail passthrough (and anything else that later
    "re-signs" it) at an arbitrary host. The matcher is now ``internal_minio_ref``
    — the SAME one ``rewrite_payload`` uses to decide what becomes a proxy URL —
    so what may be stored and what may be served cannot drift apart. Bucket and
    key must both be non-empty; ownership of the *object* is a separate, stricter
    question answered by ``artifact_ref_guard.assert_refs_owned``."""
    if internal_minio_ref(url) is not None:
        bucket, _, key = urlsplit(url).path.lstrip("/").partition("/")
        return bool(bucket and key)
    parsed = urlsplit(url)
    if parsed.scheme != "file" or not parsed.path:
        return False
    # ``file://`` is the local backend's artifact form and exists nowhere else;
    # under MinIO it is only ever an attempt to name a path on the pod's disk.
    settings = get_settings()
    if settings.storage.backend != "local":
        return False
    root = Path(settings.storage.local_path).resolve()
    candidate = Path(parsed.path)
    return candidate.is_absolute() and candidate.resolve().is_relative_to(root)
