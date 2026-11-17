"""YouTube publishing routes — turn a finished run into an upload job.

OAuth grant routes (P5-3): ``/api/youtube/auth/{connect,status,disconnect}`` are
served HERE and forwarded to youtube_upload's ``/api/auth/*`` on the service-account
lane with ``X-Owner-Tenant-Id`` = the membership-resolved tenant. They used to be
routed by the VirtualService straight to youtube_upload with the user's JWT, whose
tenant claim + realm roles stop being the truth after the P2 membership cutover
(an invited user has no ``tenant_id`` attribute at all). ``/api/youtube/auth/callback``
is NOT proxied: Google redirects the browser there with no JWT and the tenant is
recovered from the one-time OAuth ``state``, so it stays a direct route.

Per-tenant YouTube config (DB-backed in youtube_upload) is proxied the same way:
``/api/youtube/settings`` (any member) and ``/api/youtube/settings/oauth-client``
(tenant_admin) go out with the owner tenant; the operator surface
``/api/admin/youtube/platform-channel``, ``/api/admin/youtube/tenants`` and
``/api/admin/tenants/{id}/youtube`` (super_admin, on ``admin_router``) goes out
with NO owner tenant. Secrets are write-only — never logged, and scrubbed from any
upstream body that echoes them. Accepted writes are recorded in admin_audit_log.

The **upload** surface: the frontend's ``youtube.startUpload``
posts a ``pipeline_job_id``, but ``youtube_upload_service`` needs a fetchable
``file_source`` URL. This router is the translation layer — it resolves the run's
final video artifact and mints a fresh presigned MinIO URL for it (a single
self-authenticating capability the upload service fetches directly) — plus thin
proxies for the job-status reads.

"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, Literal
from uuid import UUID

import httpx
from fastapi import APIRouter, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.routes._common import (
    LICENSE_GATE_ONLY,
    AuditRepoDep,
    SessionDep,
    SubjectIdDep,
    SuperAdminDep,
    TenantAdminDep,
    TenantIdDep,
)
from src.api.routes.runs._shared import _resolve_delivered_artifact_url
from src.clients.youtube_upload_client import YouTubeUploadClient
from src.database.models import Deliverable, Project, Run
from src.database.repositories.stage_output_repository import StageOutputRepository
from src.services.blueprints.base import DAGBlueprint
from src.services.blueprints.registry import get_registry
from src.services.captions import build_srt
from src.services.run_captions import caption_cues_for_completed, completed_outputs
from src.services.stages.upstream import PermanentUpstreamError, classify
from src.services.tts_storage import _get_storage, store_captions_track

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/youtube", tags=["youtube"])

#: Operator (super_admin) YouTube config — the platform channel and a tenant's
#: permission to publish on it. A separate router because it lives under
#: ``/api/admin`` and is mounted EXEMPT of the router-wide tenant gate like the
#: admin tier (per-route gates below; see ``asgi.py``).
admin_router = APIRouter(prefix="/api/admin", tags=["youtube", "admin"])


async def _start_upload(request: Request, payload: dict, tenant_id: UUID) -> dict:
    """POST the upload to youtube_upload_service, mapping an upstream refusal.

    Extracted from ``publish_to_youtube`` so the handler stays under ruff's C901
    ceiling — and so the POST path shares the ONE upstream boundary with the two
    pollers instead of being the only surface with no handler at all (every status
    used to leave it as a raw ``httpx.HTTPStatusError`` -> 500).

    ``tenant_id`` is passed EXPLICITLY as the row's owner rather than left to the
    client's ambient-ContextVar hook — see ``_owner_tenant_event_hook``.
    """
    try:
        return await _client(request).start_upload(payload, owner_tenant_id=str(tenant_id))
    except httpx.HTTPStatusError as exc:
        raise _upstream_http_error(exc, "youtube upload could not be started") from exc


def _upstream_http_error(exc: httpx.HTTPStatusError, context: str) -> HTTPException:
    """Map an upstream youtube_upload_service failure onto a status the caller can act on.

    Uses the ONE upstream policy (``stages/upstream.py``) for the only judgement that
    matters — is this worth trying again? — instead of adding a second, route-local
    status taxonomy. A route still has to answer with a status, so the two policy
    outcomes get the two standard gateway codes:

    - ``PermanentUpstreamError`` -> **502**. The upstream stated a fault no retry fixes
      (revoked OAuth grant, rejected payload, unknown category). It is deliberately NOT
      mirrored back verbatim: passing a bare 401/403 to the SPA reads as *the caller's*
      session being invalid and logs the user out of the studio over someone else's
      broken YouTube connection.
    - ``TransientUpstreamError`` -> **503 + Retry-After**. A 429 quota block, a 408, or a
      pod mid-rollout. The poller is meant to back off and come back, which it cannot
      infer from a 500.

    Before this, every status except one fell out of the handler as a raw
    ``httpx.HTTPStatusError`` and FastAPI turned it into an opaque **500** — the exact
    "reads as an outage" failure the 404 branch below was written to stop, just for all
    the other codes. ``str()`` of the classified error already carries the upstream
    status, URL and innermost sentence, so the detail stays diagnosable.
    """
    classified = classify(exc)
    if isinstance(classified, PermanentUpstreamError):
        return HTTPException(status.HTTP_502_BAD_GATEWAY, f"{context}: {classified}")
    return HTTPException(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        f"{context}: {classified}",
        headers={"Retry-After": "5"},
    )


# YouTube renders native chapters only when the description has ≥3 timestamps in
# ascending order, the first is 0:00, and every chapter is ≥10s long.
_MIN_CHAPTERS = 3
_MIN_CHAPTER_SECONDS = 10.0
_MAX_DESCRIPTION_CHARS = 5000
# Mid-lesson (level-1) sub-chapter titles are prefixed for a visible hierarchy in
# YouTube's flat chapter list; the lesson (level-0) title heads each segment.
_SUBCHAPTER_PREFIX = "— "
# Two markers within this many seconds collapse to one (a lesson marker and its
# first sub-chapter share a timestamp); the level-0 row is kept.
_COLLAPSE_EPS = 0.5

# Stage-output keys that hold a final, publishable video URL, in priority order.
_VIDEO_OUTPUT_KEYS = ("course_artifact_url", "artifact_url", "video_url")
# Terminal video stages, highest priority first: a course run's stitched output
# wins over any per-lesson compile output present on the run.
_TERMINAL_STAGES = ("course_stitch", "lesson_compile", "demo_video_compile")
# Substrings marking an audio-only stage whose ``artifact_url`` is a sound clip,
# not a video (mirrors the frontend's ``!isAudioStage`` guard).
_AUDIO_STAGE_HINTS = ("narrat", "audio", "tts")


class SetPrivacyRequest(BaseModel):
    """Body of the governed release: the visibility to raise a private upload to."""

    privacy_status: Literal["unlisted", "public"]


class PublishRequest(BaseModel):
    """Inbound body from ``w2v_frontend`` ``youtube.startUpload``."""

    pipeline_job_id: UUID
    title: str = Field(min_length=1, max_length=100)
    description: str | None = Field(default=None, max_length=5000)
    tags: list[str] | None = None
    category_id: str | None = None
    privacy_status: str = Field(default="unlisted", pattern=r"^(public|unlisted|private)$")


async def _run_for_tenant(session: AsyncSession, run_id: UUID, tenant_id: UUID) -> Run | None:
    """Return the Run iff the caller's tenant owns it, else None (tenant isolation)."""
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
    return run if owner_tenant == tenant_id else None


def _is_audio_stage(stage_name: str) -> bool:
    return any(hint in stage_name for hint in _AUDIO_STAGE_HINTS)


def _blueprint_for(run: Run) -> DAGBlueprint | None:
    try:
        return get_registry().resolve_by_id(run.blueprint_id)
    except KeyError:
        # An unknown/retired blueprint id must not 500 a publish — fall back to
        # the shape heuristic below.
        return None


def _find_video_url(stage_outputs: list[Any], blueprint: DAGBlueprint | None = None) -> str | None:
    """Return the raw (``minio.test``) URL of the run's final video.

    Reads the DB-stored stage output (the presigned ``minio.test`` URL, not
    the browser-rewritten proxy path — those only exist on the API response).

    The blueprint's ``delivered_artifact_stage`` is the authority (W11): on a
    ``demo_video_v1`` / ``showcase_v1`` run FOUR stages carry an ``artifact_url``
    and the file the customer gets is ``final_critique``'s re-mastered output,
    not ``demo_video_compile``'s intermediate cut. Resolution walks the same
    skip-clean chain ``GET /api/runs/{id}`` publishes as
    ``delivered_artifact_url`` (``_resolve_delivered_artifact_url``), so the
    upload and the run view can never disagree about which file shipped.

    Only when the blueprint is unknown, declares a non-URL deliverable field
    (``redesign_job_id``), or its chain resolves nothing does this fall back to
    the shape heuristic: the known terminal stages first, then the last
    completed, non-audio stage carrying a video URL (the frontend's
    ``isVideoStage``/``findResultStage`` selection, server-side).
    """
    if blueprint is not None and blueprint.terminal_artifact_field in _VIDEO_OUTPUT_KEYS:
        delivered = _resolve_delivered_artifact_url(blueprint, stage_outputs)
        if delivered:
            return delivered
    return _find_video_url_by_shape(stage_outputs)


def _find_video_url_by_shape(stage_outputs: list[Any]) -> str | None:
    """Pre-W11 shape heuristic — the fallback when the blueprint cannot answer."""
    completed = {
        so.stage_name: so.output for so in stage_outputs if so.status == "completed" and so.output
    }
    for stage in _TERMINAL_STAGES:
        out = completed.get(stage)
        if not out:
            continue
        for key in _VIDEO_OUTPUT_KEYS:
            url = out.get(key)
            if isinstance(url, str) and url:
                return url

    last: str | None = None
    for so in stage_outputs:
        if so.status != "completed" or not so.output or _is_audio_stage(so.stage_name):
            continue
        for key in _VIDEO_OUTPUT_KEYS:
            url = so.output.get(key)
            if isinstance(url, str) and url:
                last = url
    return last


async def _presigned_video_url(video_url: str) -> str:
    """Re-mint a *fresh* presigned MinIO GET URL for the run's video artifact.

    The stage output holds a presigned ``http://minio.test:8389/{bucket}/{key}?...``
    URL, but it was signed by the video-compile service with a short TTL
    (1h) and is stale for any publish more than an hour after render.
    ``StorageService.resign_presigned_url`` re-derives ``{bucket}/{object_key}``
    from the path and mints a fresh presigned URL with the orchestrator's own
    MinIO credentials (it can read ``w2v-compiled-videos``) — the same
    consumption-side re-sign used before video compile. The result is a single
    self-authenticating capability the upload service fetches directly.
    """
    storage = await _get_storage()
    # resign_presigned_url is resolved via getattr (not declared on _StorageLike)
    # for the same reason as artifact_exists: that Protocol describes the
    # clip-storage surface the TTS/slide test fakes implement, and we must not
    # force every partial fake to grow this method. The real StorageService
    # always provides it.
    resign = getattr(storage, "resign_presigned_url", None)
    presigned = await resign(video_url) if resign is not None else None
    if not presigned:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            "could not presign the run's video artifact for upload",
        )
    return presigned


def _format_timestamp(seconds: float) -> str:
    """Format an offset as YouTube wants it: ``M:SS`` (or ``H:MM:SS`` past 1h)."""
    total = max(int(seconds), 0)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _chapter_seconds(value: float | int | str | None) -> float:
    """Coerce a chapter offset/duration (from JSONB) to float. None-aware — a
    legitimate 0.0 must NOT be treated as missing (``x or 0.0`` would mis-handle
    the first chapter, whose start is always 0.0)."""
    return float(value) if value is not None else 0.0


def _youtube_chapter_rows(
    chapters: list[dict], overrides: dict, hidden: frozenset[str] | set[str] = frozenset()
) -> list[dict]:
    """Flatten the 2-level course chapter list into rows for the YouTube block.

    Course chapters carry ``level`` 0 (per-lesson markers) and 1 (mid-lesson
    sub-chapters, one per slide). YouTube has no nested chapters, so we render a
    single flat list where each lesson title heads its first segment and the
    lesson's sub-chapters mark the transitions *within* it.

    Two subtleties:

    * Each lesson's first sub-chapter is emitted by ``course_stitch`` at the
      lesson's own start (its lesson-relative offset is pinned to 0.0), so the
      level-0 marker and that first level-1 row share a timestamp. Sorting by
      ``(start, level)`` puts the lesson marker first, then the collapse pass
      drops the colliding sub-chapter — the lesson title wins that slot.
    * Operator title edits (``chapter_title_overrides``) are keyed by each
      chapter's stable ``chapter_key`` — the bare ``lesson_project_id`` at
      level 0, ``f"{lesson_project_id}#{ordinal}"`` at level 1. Both levels can
      be edited; the per-slide ordinal suffix is what keeps a sub-chapter from
      inheriting its parent lesson's override (a sub carries the parent's
      ``lesson_project_id``). Edited slide titles flow through here too, so they
      appear in the published description, ``— ``-prefixed like the auto ones.
      (Old rows without ``chapter_key`` fall back to ``lesson_project_id`` at
      level 0 only — exactly the pre-edit behavior.)
    * Operator-hidden sub-chapters (``hidden`` = the set of level-1
      ``chapter_key``\\ s) are dropped here, before the sort/collapse below and
      thus before any coalescing — so a hidden slide never even competes for the
      ≥10s gap budget, and lesson markers (never hideable) are untouched.
    """
    rows: list[dict] = []
    for c in chapters:
        level = int(c.get("level") or 0)
        key = c.get("chapter_key") or (str(c.get("lesson_project_id")) if level == 0 else None)
        if level == 1 and key in hidden:
            continue
        base = overrides.get(key, c.get("title")) if key else c.get("title")
        base = str(base or "").strip()
        title = base if level == 0 else f"{_SUBCHAPTER_PREFIX}{base}"
        rows.append(
            {
                "start_seconds": _chapter_seconds(c.get("start_seconds")),
                "duration_seconds": _chapter_seconds(c.get("duration_seconds")),
                "title": title,
                "level": level,
            }
        )

    rows.sort(key=lambda r: (r["start_seconds"], r["level"]))
    collapsed: list[dict] = []
    for r in rows:
        if collapsed and r["start_seconds"] - collapsed[-1]["start_seconds"] < _COLLAPSE_EPS:
            continue  # same-second collision (level-0 sorts first and is kept)
        collapsed.append(r)
    return collapsed


def _coalesce_chapter_rows(rows: list[dict], total_seconds: float = 0.0) -> list[dict]:
    """Thin a flattened (sorted, collision-collapsed) chapter list so it satisfies
    YouTube's ≥10s-gap rule by dropping only the offending *sub-chapters*, never a
    lesson marker — preserving as much mid-lesson granularity as possible instead
    of degrading the whole list to level-0 the moment one slide lands too close.

    The lesson (level-0) markers are the non-negotiable backbone (the known-good
    list). A level-1 sub-chapter is kept only when it preserves a ≥10s gap on
    *both* sides: ``≥10s`` after the previously kept marker AND ``≥10s`` before the
    NEXT lesson marker. The forward lookahead matters — without it a kept sub could
    crowd the next (mandatory) lesson marker under 10s and re-break the whole list.

    The final chapter always plays to video end, so when ``total_seconds`` is known
    we trim a trailing sub whose true to-end duration is <10s and rewrite the last
    kept row's ``duration_seconds`` to the real remainder — otherwise
    :func:`_qualifying_chapter_lines`' last-chapter check would judge a short
    trailing slide's stored per-slide duration instead of real playback length.
    """
    n = len(rows)
    if n == 0:
        return []

    # Start of the next level-0 (lesson) marker at/after each index — None past the
    # last lesson (no upper gap constraint). Reverse scan, no math.inf needed.
    next_level0: list[float | None] = [None] * n
    seen: float | None = None
    for i in range(n - 1, -1, -1):
        next_level0[i] = seen
        if rows[i]["level"] == 0:
            seen = rows[i]["start_seconds"]

    kept: list[dict] = []
    last_start: float | None = None
    for i, r in enumerate(rows):
        if r["level"] == 0:
            kept.append(r)
            last_start = r["start_seconds"]
            continue
        gap_prev = r["start_seconds"] - (last_start if last_start is not None else 0.0)
        nxt = next_level0[i]
        gap_next = (nxt - r["start_seconds"]) if nxt is not None else _MIN_CHAPTER_SECONDS
        if gap_prev >= _MIN_CHAPTER_SECONDS and gap_next >= _MIN_CHAPTER_SECONDS:
            kept.append(r)
            last_start = r["start_seconds"]

    if total_seconds > 0.0:
        while (
            len(kept) > _MIN_CHAPTERS
            and kept[-1]["level"] == 1
            and total_seconds - kept[-1]["start_seconds"] < _MIN_CHAPTER_SECONDS
        ):
            kept.pop()
        if kept:  # the last chapter runs to video end — make its duration honest
            tail = dict(kept[-1])
            tail["duration_seconds"] = max(total_seconds - tail["start_seconds"], 0.0)
            kept[-1] = tail
    return kept


def _qualifying_chapter_lines(chapters: list[dict]) -> list[str] | None:
    """Return ``M:SS Title`` lines IFF *chapters* satisfy YouTube's native-chapter
    rules (≥3 markers, first at 0:00, ascending, every gap ≥10s, last ≥10s long),
    else ``None``. Callers compose the best qualifying candidate, so this only
    judges eligibility — it never litters a description with non-functional
    timestamps."""
    ordered = sorted(chapters, key=lambda c: _chapter_seconds(c.get("start_seconds")))
    if len(ordered) < _MIN_CHAPTERS:
        return None
    starts = [_chapter_seconds(c.get("start_seconds")) for c in ordered]
    if starts[0] != 0.0:
        return None
    gaps_ok = all(starts[i + 1] - starts[i] >= _MIN_CHAPTER_SECONDS for i in range(len(starts) - 1))
    last_ok = _chapter_seconds(ordered[-1].get("duration_seconds")) >= _MIN_CHAPTER_SECONDS
    if not (gaps_ok and last_ok):
        return None
    return [
        f"{_format_timestamp(s)} {str(c.get('title') or '').strip()}"
        for s, c in zip(starts, ordered, strict=True)
    ]


def _append_chapter_block(base: str | None, lines: list[str]) -> str | None:
    """Append a ``Chapters:`` block built from *lines* to *base*, clamped to
    YouTube's 5000-char description cap (the base body is truncated to make room
    so the block stays intact)."""
    base = base or ""
    block = "Chapters:\n" + "\n".join(lines)
    sep = "\n\n"
    if not base:
        return block[:_MAX_DESCRIPTION_CHARS]
    budget = _MAX_DESCRIPTION_CHARS - len(sep) - len(block)
    if budget < 0:
        return block[:_MAX_DESCRIPTION_CHARS]
    return (base[:budget].rstrip() + sep + block)[:_MAX_DESCRIPTION_CHARS]


def _compose_description_with_chapters(base: str | None, chapters: list[dict]) -> str | None:
    """Append a qualifying YouTube chapter block to *base*, else return *base*.

    Thin wrapper over :func:`_qualifying_chapter_lines` +
    :func:`_append_chapter_block` for the single-level call shape; the publish
    flow composes directly so it can degrade a 2-level list to level-0."""
    lines = _qualifying_chapter_lines(chapters)
    if lines is None:
        return base or None
    return _append_chapter_block(base, lines)


async def _presigned_thumbnail_url(thumbnail_url: str) -> str | None:
    """Best-effort fresh presigned URL for the course cover (None on any failure).

    The cover stored by ``course_thumbnail`` is a MinIO URL of the same shape as
    the video, so the video presigner applies. A custom thumbnail is non-essential
    (the upload service also treats it best-effort), so any failure here just
    drops the thumbnail rather than failing the publish.
    """
    try:
        return await _presigned_video_url(thumbnail_url)
    except Exception:  # noqa: BLE001 -- best-effort cover; publish proceeds without it
        log.warning("could not presign course thumbnail %r; publishing without it", thumbnail_url)
        return None


async def _caption_source_url(
    completed: dict[str, dict], run_id: str, session: AsyncSession
) -> str | None:
    """Mint, store, and presign a soft-CC SRT for the run (None when N/A).

    Builds an accurate caption track from the run's narration timing
    (:func:`caption_cues_for_completed` — the direct script+narrate pair for
    promo / lesson / website_review / generic, else the stitched-course track),
    stores it as a run artifact, and re-mints a fresh presigned MinIO URL the
    upload service can fetch directly. Best-effort: types with no recognised
    narration (audiobook) yield no cues, and any failure drops captions rather
    than failing the publish — the upload service also attaches them best-effort
    (needs a youtube.force-ssl token).
    """
    try:
        cues = await caption_cues_for_completed(completed, session)
        if not cues:
            return None
        srt = build_srt(cues)
        if not srt.strip():
            return None
        # Stored + presigned by the orchestrator's own MinIO creds at publish
        # time, so the URL is fresh (1h TTL is ample — the upload starts now) and
        # self-authenticating, exactly like the slide/clip URLs the compile
        # service fetches. No re-sign needed (unlike the video, which was signed
        # by another service hours earlier).
        return await store_captions_track(run_id, "soft_cc", run_id, srt.encode("utf-8"))
    except Exception:  # noqa: BLE001 -- best-effort soft-CC; publish proceeds without it
        log.warning("could not mint soft-CC captions for run %s; publishing without them", run_id)
        return None


async def _soft_cc_source(
    run: Run, completed: dict[str, dict], run_id: str, session: AsyncSession
) -> str | None:
    """Soft-CC URL for the run, UNLESS captions are burned into the pixels.

    When the deliverable's ``captions_mode == "burned_in"`` the video already
    carries the captions in its pixels (see the compile stages), so attaching a
    soft-CC track on top would double-render captions for a viewer with CC on.
    The two modes are therefore mutually exclusive: ``none`` (the default) → a
    soft-CC track is the only caption layer; ``burned_in`` → pixels only.
    """
    deliverable_cfg = run.config_snapshot.get("deliverable") or {}
    if str(deliverable_cfg.get("captions_mode", "none")) == "burned_in":
        return None
    return await _caption_source_url(completed, run_id, session)


def _with_caption_source(job: dict, caption_source: str | None) -> dict:
    """Echo the orchestrator-minted soft-CC URL onto the upload job response.

    The upload service's response doesn't carry ``caption_source`` (the
    orchestrator is what mints + presigns the ``.srt``), so without this the run
    view has no way to offer the caption file as a download. A no-op when no
    soft-CC track was produced.
    """
    if caption_source and isinstance(job, dict):
        job["caption_source"] = caption_source
    return job


def _client(request: Request) -> YouTubeUploadClient:
    """Pod-lifetime upload client (created in the asgi lifespan, closed at shutdown)."""
    client = getattr(request.app.state, "youtube_client", None)
    if client is None:  # defensive — lifespan always sets it
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "youtube client unavailable")
    return client


# ---------------------------------------------------------------------------
# OAuth grant proxy (P5-3) — connect / status / disconnect
# ---------------------------------------------------------------------------

# Upstream statuses that describe the ORCHESTRATOR's service-account credential,
# never the caller's session. Mirroring them would make the SPA read its own
# session as invalid and log the user out over a platform misconfiguration.
_SA_LANE_REFUSALS = frozenset({status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN})


async def _proxy_auth(
    request: Request,
    method: Literal["GET", "POST", "DELETE"],
    action: str,
    tenant_id: UUID,
) -> Response:
    """Forward one OAuth grant call; mirror the upstream status + body.

    - transport failure / upstream 5xx -> **502** (no fallback: there is no other
      place the grant can be read or written);
    - upstream 401/403 -> **502** (an SA-lane refusal, see ``_SA_LANE_REFUSALS``);
    - anything else (200, 404 "no connection", 402 licence gate, 422, a 3xx) is
      passed through verbatim — status, body and ``Location`` — never followed.
    """
    try:
        upstream = await _client(request).auth_call(method, action, str(tenant_id))
    except httpx.HTTPError as exc:
        log.warning(
            "youtube auth %s: upstream unreachable for tenant %s: %s", action, tenant_id, exc
        )
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, f"youtube_upload unreachable for auth/{action}: {exc}"
        ) from exc
    code = upstream.status_code
    if code >= status.HTTP_500_INTERNAL_SERVER_ERROR or code in _SA_LANE_REFUSALS:
        log.warning(
            "youtube auth %s: upstream %s for tenant %s: %s",
            action,
            code,
            tenant_id,
            upstream.text[:500],
        )
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            f"youtube_upload auth/{action} failed upstream with HTTP {code}",
        )
    headers = {"location": upstream.headers["location"]} if "location" in upstream.headers else None
    return Response(
        content=upstream.content,
        status_code=code,
        media_type=upstream.headers.get("content-type"),
        headers=headers,
    )


@router.post("/auth/connect", operation_id="youtube.auth.connect")
async def youtube_auth_connect(request: Request, tenant_id: TenantAdminDep) -> Response:
    """Start the OAuth flow for the caller's tenant (tenant_admin).

    Upstream ``POST /api/auth/connect`` answers JSON ``{authorization_url, state}``;
    the state is bound upstream to the owner tenant sent here, so the public
    callback stores the grant for THIS tenant.
    """
    return await _proxy_auth(request, "POST", "connect", tenant_id)


@router.get("/auth/status", operation_id="youtube.auth.status")
async def youtube_auth_status(request: Request, tenant_id: TenantIdDep) -> Response:
    """Report the caller tenant's YouTube connection (any member)."""
    return await _proxy_auth(request, "GET", "status", tenant_id)


@router.delete("/auth/disconnect", operation_id="youtube.auth.disconnect")
async def youtube_auth_disconnect(request: Request, tenant_id: TenantAdminDep) -> Response:
    """Remove the caller tenant's stored grant (tenant_admin); upstream 404 passes through."""
    return await _proxy_auth(request, "DELETE", "disconnect", tenant_id)


# ---------------------------------------------------------------------------
# Per-tenant YouTube settings + operator platform channel (DB-backed config)
# ---------------------------------------------------------------------------
# youtube_upload owns the rows (``tenant_youtube_settings``, the singleton
# ``youtube_platform_channel``); the orchestrator owns WHO may touch them.
# Secrets (a tenant's OAuth ``client_secret``, the platform ``refresh_token``) are
# write-only: never logged here, and scrubbed out of any upstream body that would
# echo them (a FastAPI 422 carries the offending payload under ``input``).

#: JSON keys whose VALUE is a credential. ``input`` is FastAPI/pydantic's echo of
#: the rejected payload — it holds the secret under an innocuous name.
_SECRET_KEYS = frozenset(
    {"client_secret", "refresh_token", "oauth_client_secret", "access_token", "input"}
)
_REDACTED = "[REDACTED]"


def _scrub(value: object) -> object:
    """Recursively replace every credential-keyed value with ``[REDACTED]``."""
    if isinstance(value, dict):
        return {k: (_REDACTED if k in _SECRET_KEYS else _scrub(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    return value


def _scrubbed_body(upstream: httpx.Response) -> bytes | None:
    """The upstream body with credentials scrubbed; ``None`` when it is not JSON.

    A non-JSON body is never mirrored (it cannot be scrubbed structurally); the
    caller gets the status with an empty body instead.
    """
    if not upstream.content:
        return b""
    try:
        parsed = upstream.json()
    except ValueError:
        return None
    return json.dumps(_scrub(parsed)).encode()


async def _proxy_settings(
    context: str,
    call: Callable[[], Awaitable[httpx.Response]],
    tenant_id: UUID | None,
) -> Response:
    """Mirror one settings/operator call with the ``_proxy_auth`` status policy.

    - transport failure / upstream 5xx / upstream 401|403 (SA-lane refusal) -> **502**;
    - anything else (200, 204, 404, 409, 422 ...) -> status + scrubbed JSON body.

    Logs carry only the context, tenant and status — never a body.
    """
    try:
        upstream = await call()
    except httpx.HTTPError as exc:
        log.warning(
            "youtube %s: upstream unreachable (tenant %s): %s",
            context,
            tenant_id,
            type(exc).__name__,
        )
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, f"youtube_upload unreachable for {context}"
        ) from exc
    code = upstream.status_code
    if code >= status.HTTP_500_INTERNAL_SERVER_ERROR or code in _SA_LANE_REFUSALS:
        log.warning("youtube %s: upstream HTTP %s (tenant %s)", context, code, tenant_id)
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            f"youtube_upload {context} failed upstream with HTTP {code}",
        )
    body = _scrubbed_body(upstream)
    if code == status.HTTP_204_NO_CONTENT or not body:
        return Response(status_code=code)
    return Response(content=body, status_code=code, media_type="application/json")


async def _audit_accepted(
    response: Response,
    *,
    audit: AuditRepoDep,
    session: AsyncSession,
    actor: UUID,
    action: str,
    target_tenant_id: UUID | None,
    detail: dict[str, object],
) -> Response:
    """Write the ``admin_audit_log`` row for a config write youtube_upload ACCEPTED.

    Same durable log as every other admin / tenant-admin mutation. Only a 2xx is
    recorded (a refused write changed nothing). The change itself lives in
    youtube_upload, so this cannot be atomic with it; the row is committed before
    the response goes out (``get_db_session`` commits only after the response).
    ``detail`` must never carry a secret — callers pass names, not values.
    """
    if status.HTTP_200_OK <= response.status_code < status.HTTP_300_MULTIPLE_CHOICES:
        await audit.record(
            actor_subject_id=actor,
            action=action,
            target_tenant_id=target_tenant_id,
            detail=detail,
        )
        await session.commit()
    return response


class TenantOAuthClientIn(BaseModel):
    """A tenant's own Google OAuth client. Both fields required (upstream contract)."""

    model_config = ConfigDict(extra="forbid")
    client_id: str = Field(min_length=1, max_length=512)
    client_secret: SecretStr = Field(min_length=1, max_length=2048)


class PlatformChannelIn(BaseModel):
    """The platform fallback channel. Absent/``null`` secrets = keep the stored ones."""

    model_config = ConfigDict(extra="forbid")
    client_id: str = Field(min_length=1, max_length=512)
    client_secret: SecretStr | None = Field(default=None, min_length=1, max_length=2048)
    refresh_token: SecretStr | None = Field(default=None, min_length=1, max_length=4096)
    channel_id: str = Field(min_length=1, max_length=128)
    channel_title: str = Field(min_length=1, max_length=256)
    default_for_tenants: bool


class TenantYouTubePolicyIn(BaseModel):
    """``platform_channel_allowed``: ``null`` = follow the platform default. Required."""

    model_config = ConfigDict(extra="forbid")
    platform_channel_allowed: bool | None


@router.get("/settings", operation_id="youtube.settings.get")
async def youtube_settings_get(request: Request, tenant_id: TenantIdDep) -> Response:
    """The caller tenant's YouTube settings (any member). Secrets are never returned."""
    yt = _client(request)
    return await _proxy_settings(
        "settings", lambda: yt.settings_call("GET", "", str(tenant_id)), tenant_id
    )


@router.put("/settings/oauth-client", operation_id="youtube.settings.oauth_client.put")
async def youtube_settings_oauth_client_put(
    body: TenantOAuthClientIn,
    request: Request,
    tenant_id: TenantAdminDep,
    actor: SubjectIdDep,
    audit: AuditRepoDep,
    session: SessionDep,
) -> Response:
    """Store the tenant's own Google OAuth client (tenant_admin).

    Upstream marks an existing grant from a different client ``needs_reconnect``.
    """
    yt = _client(request)
    payload: dict[str, object] = {
        "client_id": body.client_id,
        "client_secret": body.client_secret.get_secret_value(),
    }
    response = await _proxy_settings(
        "settings/oauth-client",
        lambda: yt.settings_call(
            "PUT", "/oauth-client", str(tenant_id), json=payload, actor_id=str(actor)
        ),
        tenant_id,
    )
    return await _audit_accepted(
        response,
        audit=audit,
        session=session,
        actor=actor,
        action="youtube.oauth_client.set",
        target_tenant_id=tenant_id,
        detail={"client_id": body.client_id},
    )


@router.delete("/settings/oauth-client", operation_id="youtube.settings.oauth_client.delete")
async def youtube_settings_oauth_client_delete(
    request: Request,
    tenant_id: TenantAdminDep,
    actor: SubjectIdDep,
    audit: AuditRepoDep,
    session: SessionDep,
) -> Response:
    """Remove the tenant's own OAuth client (tenant_admin); back to the platform client."""
    yt = _client(request)
    response = await _proxy_settings(
        "settings/oauth-client",
        lambda: yt.settings_call("DELETE", "/oauth-client", str(tenant_id), actor_id=str(actor)),
        tenant_id,
    )
    return await _audit_accepted(
        response,
        audit=audit,
        session=session,
        actor=actor,
        action="youtube.oauth_client.clear",
        target_tenant_id=tenant_id,
        detail={},
    )


@admin_router.get(
    "/youtube/platform-channel",
    operation_id="admin.youtube.platform_channel.get",
    dependencies=LICENSE_GATE_ONLY,
)
async def admin_platform_channel_get(request: Request, _: SuperAdminDep) -> Response:
    """The platform fallback channel (super_admin). Secrets are never returned."""
    yt = _client(request)
    return await _proxy_settings(
        "admin/platform-channel", lambda: yt.operator_call("GET", "/platform-channel"), None
    )


@admin_router.put(
    "/youtube/platform-channel",
    operation_id="admin.youtube.platform_channel.put",
    dependencies=LICENSE_GATE_ONLY,
)
async def admin_platform_channel_put(
    body: PlatformChannelIn,
    request: Request,
    _: SuperAdminDep,
    actor: SubjectIdDep,
    audit: AuditRepoDep,
    session: SessionDep,
) -> Response:
    """Replace the platform fallback channel (super_admin); omitted secrets are kept."""
    yt = _client(request)
    payload: dict[str, object] = body.model_dump(exclude={"client_secret", "refresh_token"})
    if body.client_secret is not None:
        payload["client_secret"] = body.client_secret.get_secret_value()
    if body.refresh_token is not None:
        payload["refresh_token"] = body.refresh_token.get_secret_value()
    response = await _proxy_settings(
        "admin/platform-channel",
        lambda: yt.operator_call("PUT", "/platform-channel", json=payload, actor_id=str(actor)),
        None,
    )
    rotated = [
        name
        for name, value in (
            ("client_secret", body.client_secret),
            ("refresh_token", body.refresh_token),
        )
        if value is not None
    ]
    return await _audit_accepted(
        response,
        audit=audit,
        session=session,
        actor=actor,
        action="youtube.platform_channel.set",
        target_tenant_id=None,
        detail={
            "client_id": body.client_id,
            "channel_id": body.channel_id,
            "channel_title": body.channel_title,
            "default_for_tenants": body.default_for_tenants,
            "rotated": rotated,
        },
    )


@admin_router.get(
    "/youtube/tenants",
    operation_id="admin.youtube.tenants.list",
    dependencies=LICENSE_GATE_ONLY,
)
async def admin_youtube_tenants_list(request: Request, _: SuperAdminDep) -> Response:
    """Every tenant's YouTube config summary (super_admin), passed through as-is.

    Upstream ``GET /api/admin/tenant-settings`` on the operator lane (no owner
    tenant) returns ``[{tenant_id, platform_channel_allowed|null, has_grant,
    has_oauth_client}]`` — flags only, no secrets.
    """
    yt = _client(request)
    return await _proxy_settings(
        "admin/tenant-settings", lambda: yt.operator_call("GET", "/tenant-settings"), None
    )


@admin_router.put(
    "/tenants/{tenant_id}/youtube",
    operation_id="admin.tenants.youtube.put",
)
async def admin_tenant_youtube_put(
    tenant_id: UUID,
    body: TenantYouTubePolicyIn,
    request: Request,
    _: SuperAdminDep,
    actor: SubjectIdDep,
    audit: AuditRepoDep,
    session: SessionDep,
) -> Response:
    """Allow/deny/default a tenant's use of the platform channel (super_admin).

    Ungated by tenant/licence like ``PATCH /api/admin/tenants/{id}``: the operator
    sets policy on ANY tenant, including one that is not (yet) licensed.
    """
    yt = _client(request)
    payload: dict[str, object] = {"allowed": body.platform_channel_allowed}
    response = await _proxy_settings(
        "admin/tenants/platform-channel",
        lambda: yt.operator_call(
            "PUT",
            f"/tenants/{tenant_id}/platform-channel",
            json=payload,
            actor_id=str(actor),
        ),
        tenant_id,
    )
    return await _audit_accepted(
        response,
        audit=audit,
        session=session,
        actor=actor,
        action="youtube.tenant_policy.set",
        target_tenant_id=tenant_id,
        detail={"platform_channel_allowed": body.platform_channel_allowed},
    )


@router.post("/upload", operation_id="youtube.upload")
async def publish_to_youtube(
    body: PublishRequest,
    request: Request,
    session: SessionDep,
    tenant_id: TenantIdDep,
) -> dict:
    """Publish a completed run's final video to YouTube.

    Resolves the run (tenant-scoped), finds its final video artifact, mints a
    fresh presigned MinIO URL for it, and starts an upload job on
    ``youtube_upload_service``. Returns the downstream ``UploadJobResponse``,
    echoing the minted ``caption_source`` URL when a soft-CC track was produced
    (the upload service's own response doesn't carry it) so the run view can
    offer the ``.srt`` as a download.
    """
    run = await _run_for_tenant(session, body.pipeline_job_id, tenant_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "run not found")
    if run.status != "completed":
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"run is not completed (status={run.status}); nothing to publish yet",
        )

    stage_outputs = await StageOutputRepository(session).list_for_run(body.pipeline_job_id)
    video_url = _find_video_url(stage_outputs, _blueprint_for(run))
    if not video_url:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "run has no publishable video artifact",
        )

    completed = completed_outputs(stage_outputs)

    payload: dict[str, Any] = {
        "file_source": await _presigned_video_url(video_url),
        "file_source_type": "url",
        "title": body.title,
        "privacy_status": body.privacy_status,
    }

    # Auto-compose YouTube chapters from a course's chapter markers (server-side,
    # so any client benefits and the upload modal needs no new fields). YouTube has
    # no nested chapters, so we flatten the 2-level list — lesson titles head each
    # segment, mid-lesson (slide) sub-chapters mark the transitions within it. We
    # coalesce that flat list to YouTube's ≥10s-gap rule by dropping only the
    # offending too-close sub-chapters (never a lesson marker), so one tightly-spaced
    # slide no longer discards every sub-chapter. Only if the lesson backbone itself
    # can't qualify (two lessons <10s apart) do we fall back to the level-0-only list.
    # Operator title edits apply to both levels; operator-hidden sub-chapters
    # (hidden_chapter_keys) are dropped from the flattened list before coalescing.
    stitch_out = completed.get("course_stitch") or {}
    overrides = stitch_out.get("chapter_title_overrides") or {}
    hidden = set(stitch_out.get("hidden_chapter_keys") or [])
    chapters = stitch_out.get("chapters") or []
    total_seconds = _chapter_seconds(stitch_out.get("duration_seconds"))
    rows = _youtube_chapter_rows(chapters, overrides, hidden)
    coalesced = _coalesce_chapter_rows(rows, total_seconds)
    top_level_rows = _youtube_chapter_rows(
        [c for c in chapters if int(c.get("level") or 0) == 0], overrides, hidden
    )
    lines = _qualifying_chapter_lines(coalesced) or _qualifying_chapter_lines(top_level_rows)
    description = (
        _append_chapter_block(body.description, lines) if lines else (body.description or None)
    )
    if description:
        payload["description"] = description
    if body.tags is not None:
        payload["tags"] = body.tags
    if body.category_id is not None:
        payload["category_id"] = body.category_id

    # Course cover → YouTube custom thumbnail (best-effort). An operator override
    # wins, then the AI cover (course_thumbnail), then the copy folded onto
    # course_stitch.
    thumbnail_url = (
        stitch_out.get("thumbnail_override_url")
        or (completed.get("course_thumbnail") or {}).get("thumbnail_url")
        or stitch_out.get("thumbnail_url")
    )
    if thumbnail_url:
        presigned_thumb = await _presigned_thumbnail_url(str(thumbnail_url))
        if presigned_thumb:
            payload["thumbnail_source"] = presigned_thumb
            payload["thumbnail_source_type"] = "url"

    # Soft closed-captions (best-effort). Minted from the run's narration timing
    # for promo / lesson / website_review; the upload service attaches them via
    # captions().insert (needs a youtube.force-ssl token, else silently skipped).
    caption_source = await _soft_cc_source(run, completed, str(body.pipeline_job_id), session)
    if caption_source:
        payload["caption_source"] = caption_source
        payload["caption_source_type"] = "url"
        payload["caption_language"] = "en"

    job = await _start_upload(request, payload, tenant_id)
    return _with_caption_source(job, caption_source)


@router.get("/upload/jobs/{job_id}", operation_id="youtube.upload.get")
async def get_upload_job(job_id: str, request: Request, tenant_id: TenantIdDep) -> dict:
    """Poll a YouTube upload job (proxied to youtube_upload_service).

    Pass the upstream 404 through as a 404 rather than letting the client's
    ``raise_for_status`` bubble an ``httpx.HTTPStatusError`` into a 500 — a
    poller hammering an unreadable job otherwise spams 500s (observed live,
    ~4/min) and reads as an outage to the caller.

    A 404 does NOT mean the job was pruned or expired: ``upload_jobs`` has no
    TTL and no prune anywhere in youtube_upload_service. A tenant mismatch is
    ONE cause — measured 2026-08-20 on 35 of 48 prod rows (NULL for every row
    predating migration ``005_add_owner_tenant_id``, else a sentinel/service
    tenant) — but it is NOT the only one, and reading it as the only one costs
    a session. Measured 2026-08-23 on the id that produces every one of these
    warnings today: upstream's own log reports the owner tenant and the channel
    tenant as IDENTICAL. The row simply does not exist. So a 404 here means
    "no such job for you", and the caller's next question is whether the id was
    ever issued, not only whether the tenants line up.

    Hence the ``log.warning`` below: this branch was previously silent at every
    level, which is why 484 of these went undiagnosed. Keep it at warning.
    Volume is not evidence of breadth — the entire live signal (~90 req/hr,
    the service's #2 signature) is ONE external poller re-asking for a single
    uuid7 id stamped 2026-06-21 that upstream has never had. No product path
    reaches this route: the frontend's ``getUploadJob`` wrapper has no callers.

    Every OTHER status now goes through ``_upstream_http_error`` (the ONE upstream
    policy) instead of falling out as a raw ``httpx.HTTPStatusError`` -> 500. The 404
    branch stays FIRST and unchanged: it is a domain answer this route owns, and
    ``classify`` would call it permanent and hand back a 502.
    """
    try:
        return await _client(request).get_upload_job(job_id, owner_tenant_id=str(tenant_id))
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == status.HTTP_404_NOT_FOUND:
            log.warning("youtube upload job %s not found upstream for tenant %s", job_id, tenant_id)
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, f"upload job {job_id} not found"
            ) from exc
        raise _upstream_http_error(exc, f"upload job {job_id}") from exc


@router.post("/upload/jobs/{job_id}/retract", operation_id="youtube.upload.retract")
async def retract_upload_job(job_id: str, request: Request, tenant_id: TenantIdDep) -> dict:
    """Retract a published upload: set its YouTube video to private (proxied).

    Reduce-only and reversible — youtube_upload_service never raises visibility on
    this path, and the video stays on the channel. Scoped to the calling tenant
    exactly like the job poll. 404 (no such job for you) and 409 (the job never
    published a video) are domain answers passed through; every other upstream
    status goes through the ONE upstream policy (``_upstream_http_error``).
    """
    try:
        return await _client(request).retract_upload_job(job_id, owner_tenant_id=str(tenant_id))
    except httpx.HTTPStatusError as exc:
        code = exc.response.status_code
        if code == status.HTTP_404_NOT_FOUND:
            log.warning(
                "youtube retract: job %s not found upstream for tenant %s", job_id, tenant_id
            )
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, f"upload job {job_id} not found"
            ) from exc
        if code == status.HTTP_409_CONFLICT:
            raise HTTPException(
                status.HTTP_409_CONFLICT, f"upload job {job_id} has no published video"
            ) from exc
        raise _upstream_http_error(exc, f"retract of upload job {job_id}") from exc


@router.post("/upload/jobs/{job_id}/privacy", operation_id="youtube.upload.set_privacy")
async def set_upload_job_privacy(
    job_id: str, body: SetPrivacyRequest, request: Request, tenant_id: TenantIdDep
) -> dict:
    """Release a private upload: raise its YouTube video to unlisted/public (proxied).

    The governed counterpart of :func:`retract_upload_job`, called by marketing's
    ``marketing.publish_video`` release after a human approved the video. Only
    ``unlisted``/``public`` are accepted (lowering is /retract). Scoped to the calling
    tenant exactly like the job poll; 404/409 are domain answers passed through and
    every other upstream status goes through ``_upstream_http_error``.
    """
    try:
        return await _client(request).set_upload_job_privacy(
            job_id, body.privacy_status, owner_tenant_id=str(tenant_id)
        )
    except httpx.HTTPStatusError as exc:
        code = exc.response.status_code
        if code == status.HTTP_404_NOT_FOUND:
            log.warning(
                "youtube privacy: job %s not found upstream for tenant %s", job_id, tenant_id
            )
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, f"upload job {job_id} not found"
            ) from exc
        if code == status.HTTP_409_CONFLICT:
            raise HTTPException(
                status.HTTP_409_CONFLICT, f"upload job {job_id} has no published video"
            ) from exc
        raise _upstream_http_error(exc, f"privacy change of upload job {job_id}") from exc


@router.get("/upload/jobs", operation_id="youtube.upload.list")
async def list_upload_jobs(
    request: Request,
    tenant_id: TenantIdDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict:
    """List YouTube upload jobs for the calling tenant (proxied).

    ``tenant_id`` is forwarded EXPLICITLY as the owner scope; it used to be
    unaccessed here, so the declared dependency was decorative and the scoping
    silently rode the client's ambient ContextVar.
    """
    try:
        return await _client(request).list_upload_jobs(
            limit=limit, offset=offset, owner_tenant_id=str(tenant_id)
        )
    except httpx.HTTPStatusError as exc:
        raise _upstream_http_error(exc, "youtube upload jobs unavailable") from exc
