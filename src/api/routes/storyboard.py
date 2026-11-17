"""Storyboard routes — project-scoped CRUD for panels, scenes, assets (Phase 0).

Mirrors ``routes/verbatim.py``: ``SessionDep`` + ``TenantIdDep`` deps, a
``ProjectRepository`` ownership 404, ``body.model_dump(exclude_unset=True)`` on
create/patch, and a hand-built ``_resp(row) -> dict`` mapper per resource (NOT
``from_attributes``). The AI-authoring endpoints ``draft`` + ``rewrite`` ARE now
mounted here (Phase 1, LLM-only). The remaining AI endpoints
(render/variations/animatic) and asset reference-image upload are LATER phases —
not mounted here.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Annotated, Literal, cast
from urllib.parse import urlsplit
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, Response, status
from sqlalchemy.exc import IntegrityError
from starlette.responses import JSONResponse

from src.api.minio_stream import stream_object
from src.api.panel_image_url import panel_object_ref, proxied_job, public_media_ref
from src.api.routes._common import SessionDep, SubjectIdDep, TenantIdDep
from src.api.schemas import (
    BranchCreateRequest,
    BranchDiffResponse,
    BranchListResponse,
    BranchResponse,
    DiffEntry,
    MergeRequest,
    MergeResolution,
    MergeResponse,
    PanelApproveRequest,
    PanelEventListResponse,
    PanelEventResponse,
    PanelImageJobResponse,
    PanelImageRequest,
    PanelRewriteRequest,
    ReorderPanels,
    ReorderScenes,
    RestoreResponse,
    RestoreSnapshotRequest,
    SiblingMergeRequest,
    SiblingMergeResponse,
    SnapshotCreateRequest,
    SnapshotListResponse,
    SnapshotResponse,
    StoryboardAssetCreate,
    StoryboardAssetListResponse,
    StoryboardAssetResponse,
    StoryboardAssetUpdate,
    StoryboardDraftRequest,
    StoryboardDraftResponse,
    StoryboardPanelCreate,
    StoryboardPanelListResponse,
    StoryboardPanelResponse,
    StoryboardPanelUpdate,
    StoryboardSceneCreate,
    StoryboardSceneListResponse,
    StoryboardSceneResponse,
    StoryboardSceneUpdate,
)
from src.cache.raw import raw_redis
from src.config import get_settings
from src.database.models import StoryboardPanel
from src.database.repositories.project_repository import ProjectRepository
from src.database.repositories.storyboard_repository import (
    AssetInUseError,
    PanelVersionConflictError,
    ProjectArchivedError,
    StoryboardAssetRepository,
    StoryboardBranchRepository,
    StoryboardPanelEventRepository,
    StoryboardPanelRepository,
    StoryboardRepository,
    StoryboardSceneRepository,
    StoryboardSnapshotRepository,
    StoryboardVersionConflictError,
    UnknownAssetError,
    UnknownSceneError,
)
from src.services.storage_service import guess_artifact_content_type
from src.services.storyboard.assist import (
    DraftConflictError,
    DraftInputError,
    StoryboardAssistService,
)
from src.services.storyboard.branch_service import (
    BranchAlreadyMergedError,
    BranchNotFoundError,
    OpenChildBranchesError,
    ProjectNotFoundError,
    StoryboardBranchService,
)
from src.services.storyboard.csv_export import build_csv
from src.services.storyboard.edl_export import build_edl, build_fcpxml
from src.services.storyboard.merge_service import (
    FIELD_MERGEABLE,
    BaseSnapshotMissingError,
    BranchNotOpenError,
    FieldResolutionTargetError,
    KeepBothTargetError,
    NotSiblingsError,
    SharedBaseMissingError,
    StaleRootError,
    StoryboardMergeService,
    UnresolvedConflictError,
)
from src.services.storyboard.panel_image_jobs import PanelImageJobRunner
from src.services.storyboard.panel_presign import MediaRefJson, refresh_panel_media_url
from src.services.storyboard.pdf_export import build_pdf
from src.services.storyboard.render import PanelImageOptions
from src.services.storyboard.storyboard_snapshot_service import (
    SnapshotNotFoundError,
    StoryboardSnapshotService,
)
from src.services.visual_media import fetch_image_bytes

log = logging.getLogger(__name__)

router = APIRouter(tags=["storyboard"])

# The asset-reference failure modes, worded once (repo raises, routes translate).
_ASSET_IN_USE = "asset is referenced by storyboard panels; remove the references first"
_UNKNOWN_ASSET = "unknown asset id for this project"
_UNKNOWN_SCENE = "unknown scene id for this project"


def _unknown_asset_http(exc: UnknownAssetError) -> HTTPException:
    """A character/prop/location id that isn't an asset of THIS project → 422."""
    return HTTPException(
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail={"message": _UNKNOWN_ASSET, "ids": exc.ids},
    )


def _unknown_scene_http(exc: UnknownSceneError) -> HTTPException:
    """A panel ``scene_id`` that isn't a scene of THIS project → 422 (mirrors the
    asset mapping)."""
    return HTTPException(
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail={"message": _UNKNOWN_SCENE, "ids": exc.ids},
    )


def _shot_key_conflict_http(exc: IntegrityError, shot_key: str | None) -> HTTPException | None:
    """Map a ``uq_storyboard_panels_project_shot`` unique violation → 409, else None.

    Shared by ``create_panel`` and ``patch_panel``. The caller MUST have already rolled
    the session back (a failed flush poisons it); this only inspects ``exc.orig`` and
    builds the response. Returns ``None`` for any OTHER IntegrityError so the caller
    re-raises it unchanged.
    """
    orig = str(exc.orig) if exc.orig is not None else ""
    if "uq_storyboard_panels_project_shot" not in orig:
        return None
    # Only name a specific key when the client supplied one. When shot_key is omitted the
    # repo auto-assigns clip-<next>, which we cannot faithfully reconstruct after rollback
    # without re-querying — so avoid reporting a guessed (and possibly wrong) key.
    if shot_key:
        detail = f"A storyboard panel with shot_key '{shot_key}' already exists for this project."
    else:
        detail = (
            "A storyboard panel with the auto-assigned shot_key already exists for this "
            "project; pass an explicit shot_key."
        )
    return HTTPException(status.HTTP_409_CONFLICT, detail=detail)


def _split_resolutions(
    resolutions: list[MergeResolution],
) -> tuple[dict[UUID, str], dict[UUID, Mapping[str, str]]]:
    """Split the merge request's resolutions into (whole-row, per-field) maps.

    A duplicate key is ambiguous — two verdicts for one panel — and silently last-wins in a
    dict comprehension, so reject it as a 422. Each ``MergeResolution`` carries EITHER a
    whole-row ``resolution`` OR ``field_resolutions`` (schema-enforced XOR); the ``None`` side
    of each is dropped so a null whole-row verdict never lands in ``resolutions`` and gets
    read as "skip this field-resolved conflict", silently dropping the per-field merge.
    """
    keys = [r.key for r in resolutions]
    if len(keys) != len(set(keys)):
        dupes = sorted({str(k) for k in keys if keys.count(k) > 1})
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"message": "duplicate resolution keys", "keys": dupes},
        )
    whole_row = {r.key: r.resolution for r in resolutions if r.resolution is not None}
    per_field: dict[UUID, Mapping[str, str]] = {
        r.key: r.field_resolutions for r in resolutions if r.field_resolutions is not None
    }
    return whole_row, per_field


# ─── PDF board: panel frame resolution ───────────────────────────────────────
# The CSV/EDL/FCPXML exports only read DB rows; the PDF board is the first that
# needs the panel's PICTURE. A panel's visual lives on ``media_ref["url"]`` (the
# JSONB blob, see ``src/models/showcase_line.py`` ``MediaRef``) or on the plain
# ``image_ref`` column when that happens to hold an absolute URL.
#
# That URL is author-/pipeline-supplied, i.e. attacker-influenced, so it is NEVER
# fetched directly: ``visual_media.fetch_image_bytes`` runs it through
# ``w2v_common.net.safe_get`` (the estate's SSRF validator — resolves the host
# once, rejects private/loopback/link-local/**CGNAT**, pins the
# connection to the validated IP against DNS-rebinding, and re-validates every
# redirect hop) and enforces the content-type + byte cap. Every failure degrades
# to the builder's placeholder: a board missing a thumbnail is useful, a 500 is
# not.
_PDF_IMAGE_MAX_BYTES = 4 * 1024 * 1024  # 4 MiB per frame
_PDF_IMAGE_TIMEOUT_S = 5.0
# A board can hold hundreds of panels; cap how many frames one export will fetch
# so a single request cannot turn into an unbounded egress fan-out.
_PDF_IMAGE_MAX_FETCHES = 60


def _panel_image_url(panel: StoryboardPanel, media_ref: MediaRefJson | None = None) -> str | None:
    """The panel's absolute http(s) frame URL, or ``None`` when it has no visual.

    ``media_ref["url"]`` wins over ``image_ref``; a non-absolute ``image_ref``
    (a bare storage key, a relative proxy path) is NOT a URL we can fetch here
    and yields ``None`` → placeholder.

    ``media_ref`` may be passed EXPLICITLY to supply a re-signed copy. The stored
    value is presigned for one hour, and a board is read for days, so the row's
    own string is expired for almost every real export — see
    :func:`_resolve_panel_images`.
    """
    if media_ref is None:
        media_ref = cast("MediaRefJson", panel.media_ref or {})
    candidate = media_ref.get("url") if isinstance(media_ref, dict) else None
    if not isinstance(candidate, str) or not candidate:
        candidate = panel.image_ref
    if not isinstance(candidate, str):
        return None
    candidate = candidate.strip()
    if not candidate.startswith(("http://", "https://")):
        return None
    return candidate


def _artifact_store_allow_hosts() -> tuple[str, ...]:
    """The MinIO host(s) the PDF export may fetch a CONFINED frame URL from.

    Derived from ``StorageSettings`` (``STORAGE_MINIO_ENDPOINT``, plus the
    external endpoint when one is configured, since ``presign_get`` rewrites the
    host to it) — never hardcoded, so a redeployment that moves the artifact store
    does not silently re-break the export. Both the bare hostname and the
    ``host:port`` form are emitted because ``safe_get`` matches either and the
    stored URL may or may not carry the port.

    Empty when nothing is configured (the local backend), which leaves the guard
    at its full strength.
    """
    storage = get_settings().storage
    hosts: list[str] = []
    for endpoint in (storage.minio_endpoint, storage.minio_external_endpoint):
        if not endpoint:
            continue
        # ``minio_external_endpoint`` may carry a scheme; ``minio_endpoint`` is
        # bare ``host[:port]``. urlsplit needs a scheme to populate .hostname.
        parts = urlsplit(endpoint if "//" in endpoint else f"//{endpoint}")
        if not parts.hostname:
            continue
        hosts.append(parts.hostname)
        if parts.port is not None:
            hosts.append(f"{parts.hostname}:{parts.port}")
    return tuple(dict.fromkeys(hosts))


async def _resolve_panel_images(
    panels: list[StoryboardPanel], *, tenant_id: UUID
) -> tuple[dict[UUID, bytes], set[UUID]]:
    """Best-effort ``({panel_id: image_bytes}, {panel_ids that HAD a url and failed})``.

    Never raises: any per-panel failure (SSRF rejection, timeout, HTTP error,
    oversize body, non-image content-type) is skipped so the panel falls back to
    the placeholder and the export still succeeds. The second element is what
    lets the board distinguish "never generated" from "we had a frame and could
    not fetch it" — captioning both the same way tells a director a rendered
    shot was never made.

    **The URL is RE-SIGNED first.** ``media_ref.url`` is presigned for one hour
    and a board is read for days, so the stored string is expired for virtually
    every real export; fetching it verbatim degraded EVERY frame to a
    placeholder while the board's own UI (which re-signs in ``_panel_resp``)
    showed the images fine.

    **And the artifact host is allow-listed, per URL.** Our own frames live on
    ``minio.test``, which is an estate-internal name the SSRF guard blocks by
    design — so before this, *every* generated frame raised ``SSRFError`` and the
    board rendered the placeholder however good the generation was. The allowlist
    is applied to a single fetch at a time and only when
    :func:`panel_image_url.panel_object_ref` confirms the URL addresses an object
    under this panel's OWN project/tenant prefix. ``media_ref`` is client-writable
    (panel create and PATCH both accept it), so a blanket allowlist would let a
    tenant aim a panel at any internal host and have the export fetch it; a URL
    that fails the predicate keeps the full guard and degrades to the placeholder.
    """
    allow_hosts = _artifact_store_allow_hosts()
    targets: list[tuple[UUID, str, bool]] = []
    for panel in panels:
        fresh = await refresh_panel_media_url(cast("MediaRefJson", panel.media_ref))
        url = _panel_image_url(panel, fresh)
        if url is not None:
            # Confinement is checked against the URL we are about to fetch, not
            # the stored one: the re-sign above mints a new signature and, when an
            # external endpoint is configured, a different host.
            confined = (
                panel_object_ref(url, project_id=panel.project_id, tenant_id=tenant_id) is not None
            )
            targets.append((panel.id, url, confined))
        if len(targets) >= _PDF_IMAGE_MAX_FETCHES:
            break
    if not targets:
        return {}, set()

    resolved: dict[UUID, bytes] = {}
    unavailable: set[UUID] = set()
    async with httpx.AsyncClient(timeout=_PDF_IMAGE_TIMEOUT_S) as client:
        for panel_id, url, confined in targets:
            try:
                data, _fmt = await fetch_image_bytes(
                    url,
                    timeout=_PDF_IMAGE_TIMEOUT_S,
                    max_bytes=_PDF_IMAGE_MAX_BYTES,
                    client=client,
                    allow_hosts=allow_hosts if confined else None,
                )
            except Exception:  # degrade to placeholder, never fail the export
                log.warning("storyboard pdf: panel %s frame unavailable", panel_id, exc_info=True)
                unavailable.add(panel_id)
                continue
            resolved[panel_id] = data
    return resolved, unavailable


async def _require_project(session: SessionDep, project_id: UUID, tenant_id: UUID) -> None:
    if await ProjectRepository(session).get(project_id, tenant_id=tenant_id) is None:
        raise HTTPException(404, "project not found")


async def _require_editable_project(session: SessionDep, project_id: UUID, tenant_id: UUID) -> None:
    """404 if missing/cross-tenant, 409 (``ProjectArchivedError``) if archived (read-only).

    The write-path sibling of :func:`_require_project`, for the routes that DO carry a
    ``project_id`` and touch a project the repository chokepoint never loads directly
    (asset create, panel reorder, snapshot create/delete). One ``ProjectRepository.get``
    — the same query ``_require_project`` already runs — so it is a drop-in with zero
    extra cost. The archived case is raised as ``ProjectArchivedError`` (not a bare
    ``HTTPException``) so it maps through the ONE registered handler to a uniform 409.
    """
    project = await ProjectRepository(session).get(project_id, tenant_id=tenant_id)
    if project is None:
        raise HTTPException(404, "project not found")
    if project.status == "archived":
        raise ProjectArchivedError(project_id)


async def _project_archived_handler(request: Request, exc: Exception) -> JSONResponse:
    """Map :class:`ProjectArchivedError` → 409. The repository raises it from the by-id
    mutators (patch/delete/approve panel, scene, asset — no project_id in their path), so
    the translation cannot live in each route; it is registered once at the app level."""
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content={"detail": "project is archived and read-only"},
    )


async def _version_conflict_handler(request: Request, exc: Exception) -> JSONResponse:
    """Map :class:`StoryboardVersionConflictError` (panel/scene/asset) → 409. Raised by the
    by-id mutators when an ``expected_lock_version`` lost the compare-and-set; like the
    archived handler it lives at app level because the by-id routes carry no context to
    translate it per-route. ``exc.entity`` names the row kind, so a panel conflict keeps its
    long-standing "panel was modified concurrently" wording."""
    entity = exc.entity if isinstance(exc, StoryboardVersionConflictError) else "record"
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content={"detail": f"{entity} was modified concurrently"},
    )


def register_storyboard_exception_handlers(app: FastAPI) -> None:
    """Register the storyboard-repository → HTTP ASGI handlers on ``app``
    (``ProjectArchivedError`` → 409, ``StoryboardVersionConflictError`` → 409).

    The version-conflict handler is registered on the BASE
    ``StoryboardVersionConflictError``, so it catches the panel/scene/asset subclasses via
    the MRO with one registration.

    Called from ``src/asgi.py`` for the production app AND from the test app builder in
    ``tests/conftest.py`` — the test harness hand-builds its FastAPI app with plain
    ``include_router`` and does NOT go through ``w2v_common.asgi.create_app``, so a handler
    registered only in ``asgi.py`` would be absent under test and the mutators would surface
    a raw 500 instead of the 409 the repository intends.
    """
    app.add_exception_handler(ProjectArchivedError, _project_archived_handler)
    app.add_exception_handler(StoryboardVersionConflictError, _version_conflict_handler)


# ─── Mandatory optimistic-concurrency messages ───────────────────────────
# The destructive + bulk-ordering routes REQUIRE a lock token (unlike the opt-in
# field PATCHes). A missing token is a 400 that names the fix, not a silent
# last-write-wins; a STALE token is the existing 409 conflict shape raised by the
# repository (``StoryboardVersionConflictError`` → "… was modified concurrently").
_LOCK_REQUIRED = (
    "expected_lock_version is required when deleting a {entity}: re-read the {entity} and "
    "pass the lock_version it returned as ?expected_lock_version=<n>. A delete cannot be "
    "merged, so an unguarded one would silently discard a concurrent edit."
)
_BOARD_LOCK_REQUIRED = (
    "expected_lock_versions is required on this route: send the {{<{entity} id>: "
    "<lock_version>}} map exactly as you last read it. This write rewrites {scope}, so an "
    "unguarded one would silently discard concurrent edits."
)


async def _panel_resp(row, tenant_id: UUID) -> StoryboardPanelResponse:  # noqa: ANN001
    """Map a panel row to its wire shape, fixing its media URL on the way out.

    Two rewrites, both in the ONE mapper every panel-returning route already goes
    through rather than sprinkled per-route, so a route added later cannot forget
    them and quietly serve an image the SPA cannot render:

    1. **Re-sign.** ``media_ref.url`` is a 1h-TTL presigned URL and a storyboard
       is read for days, so the stored signature is expired on almost every read
       of a generated panel (see ``storyboard/panel_presign``). Load-bearing for
       any owned URL the proxy does not cover.
    2. **Proxy-path rewrite.** A ``minio.test`` host does not resolve in the
       user's browser, so an in-cluster URL is replaced by the panel's own
       ``/api/storyboard/panels/{id}/image/content`` path — but ONLY when the key
       is under this project's / tenant's own prefix (see ``api/panel_image_url``;
       ``media_ref`` is client-writable, so it is not evidence of ownership).
       Anything else — an operator-pasted external image, or a poisoned ref
       naming a stranger's object — passes through with just the re-sign, exactly
       as before this proxy existed.

    ``tenant_id`` is the authenticated caller's, threaded from the route rather
    than read off the row: ``StoryboardPanel`` has no ``tenant_id`` column (RLS
    climbs ``project_id -> projects.tenant_id``), and the guard must key on the
    identity that was actually authenticated.

    The stored row is untouched by both: the presigned URL stays the durable
    object identity, and the PDF export still reads it (``_resolve_panel_images``).
    """
    return StoryboardPanelResponse.model_validate(
        {
            "id": row.id,
            "project_id": row.project_id,
            "origin_id": row.origin_id,
            "shot_key": row.shot_key,
            "scene_id": row.scene_id,
            "shot_number": row.shot_number,
            "panel_index": row.panel_index,
            "status": row.status,
            "revision": row.revision,
            "lock_version": row.lock_version,
            "approved_at": row.approved_at,
            "approved_by": row.approved_by,
            "action_description": row.action_description,
            "dialogue": row.dialogue,
            "vo_text": row.vo_text,
            "sfx_cue": row.sfx_cue,
            "music_cue": row.music_cue,
            "image_ref": row.image_ref,
            # Ownership is decided BEFORE any signature is minted. Re-signing first
            # and refusing to proxy second handed the caller a live presigned URL
            # for an object the guard was refusing — see public_media_ref.
            "media_ref": public_media_ref(
                row.id,
                row.media_ref,
                project_id=row.project_id,
                tenant_id=tenant_id,
            ),
            "visual_type": row.visual_type,
            "shot_size": row.shot_size,
            "composition_notes": row.composition_notes,
            "characters": row.characters,
            "location_asset_id": row.location_asset_id,
            "props": row.props,
            "lighting_mood": row.lighting_mood,
            "art_style": row.art_style,
            "aspect_ratio": row.aspect_ratio,
            "angle": row.angle,
            "movement": row.movement,
            "movement_path": row.movement_path,
            "lens": row.lens,
            "duration_seconds": row.duration_seconds,
            "transition_out": row.transition_out,
            "gen": row.gen,
            "source": row.source,
            "position": row.position,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
        }
    )


def _scene_resp(row) -> StoryboardSceneResponse:  # noqa: ANN001
    return StoryboardSceneResponse.model_validate(
        {
            "id": row.id,
            "project_id": row.project_id,
            "name": row.name,
            "int_ext": row.int_ext,
            "location": row.location,
            "time_of_day": row.time_of_day,
            "description": row.description,
            "position": row.position,
            "lock_version": row.lock_version,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
        }
    )


def _asset_resp(row) -> StoryboardAssetResponse:  # noqa: ANN001
    return StoryboardAssetResponse.model_validate(
        {
            "id": row.id,
            "project_id": row.project_id,
            "kind": row.kind,
            "name": row.name,
            "canonical_descriptor": row.canonical_descriptor,
            "reference_images": row.reference_images,
            "seed": row.seed,
            "style_token": row.style_token,
            "lora_path": row.lora_path,
            "attributes": row.attributes,
            "position": row.position,
            "lock_version": row.lock_version,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
        }
    )


def _snapshot_resp(row, *, payload: dict | None = None) -> SnapshotResponse:  # noqa: ANN001
    """Map a ``StoryboardSnapshot`` row → response. ``payload`` is passed ONLY when
    already in hand (create/restore) so the graph-size counts are cheap; the list
    scan defers ``payload`` and must NEVER touch ``row.payload`` (a sync lazy-load
    on the async session), so it leaves the counts as None."""
    counts: dict[str, int] = {}
    if payload is not None:
        counts = {
            "scene_count": len(payload.get("scenes", [])),
            "panel_count": len(payload.get("panels", [])),
            "asset_count": len(payload.get("assets", [])),
        }
    return SnapshotResponse.model_validate(
        {
            "id": row.id,
            "kind": row.kind,
            "note": row.note,
            "actor_subject": row.actor_subject,
            "created_at": row.created_at,
            **counts,
        }
    )


def _branch_resp(row) -> BranchResponse:  # noqa: ANN001
    """Map a ``StoryboardBranch`` row → response. Hand-built (not ``from_attributes``)
    like every other mapper here, and deliberately does NOT expose the child project's
    graph: a branch listing is a lifecycle view, and walking each child's panels would
    turn one page into N+1 scans."""
    return BranchResponse.model_validate(
        {
            "id": row.id,
            "root_project_id": row.root_project_id,
            "child_project_id": row.child_project_id,
            "name": row.name,
            "base_snapshot_id": row.base_snapshot_id,
            "status": row.status,
            "created_at": row.created_at,
            "merged_at": row.merged_at,
        }
    )


def _event_resp(row) -> PanelEventResponse:  # noqa: ANN001
    return PanelEventResponse.model_validate(
        {
            "id": row.id,
            "project_id": row.project_id,
            "panel_id": row.panel_id,
            "event_type": row.event_type,
            "revision": row.revision,
            "actor_subject_id": row.actor_subject_id,
            "note": row.note,
            "created_at": row.created_at,
        }
    )


# ─── Why every mutating route below ends in `await session.commit()` ─────
# The repositories and the storyboard services (assist / snapshot / branch / merge)
# only ``flush``; they document that the ROUTE commits. The only commit used to be
# the one in ``get_db_session``'s ``yield`` teardown — which FastAPI (0.136) runs
# AFTER the response has been sent. So a caller was handed its 200/201 and an id for
# work that was not durable yet, and its very next request read a database where that
# work did not exist. At idle the teardown commit wins that race in microseconds;
# under the live 33-scenario e2e matrix it loses (``storyboard-merge`` and
# ``panel-image`` both failed exactly that way, and one passed on a re-run with no
# code change — the signature of a race).
#
# So each mutation commits after its last write and before its response is built.
# The teardown commit stays as the safety net; the error paths (``session.rollback``
# on an IntegrityError, every HTTPException) deliberately do NOT commit.
# ``routes/runs/lifecycle.py`` and ``runs/stages.py`` have always done this.
# The per-handler additions below are therefore bare — this is the explanation.


# ─── Panels ──────────────────────────────────────────────────────────────
@router.get(
    "/api/projects/{project_id}/storyboard/panels",
    response_model=StoryboardPanelListResponse,
    operation_id="storyboard.panels.list",
)
async def list_panels(
    project_id: UUID, session: SessionDep, tenant_id: TenantIdDep
) -> StoryboardPanelListResponse:
    await _require_project(session, project_id, tenant_id)
    rows = await StoryboardPanelRepository(session).list_for_project(
        project_id, tenant_id=tenant_id
    )
    items = [await _panel_resp(r, tenant_id) for r in rows]
    return StoryboardPanelListResponse(items=items, total=len(items))


@router.get(
    "/api/projects/{project_id}/storyboard/export.csv",
    operation_id="storyboard.export.csv",
)
async def export_storyboard_csv(
    project_id: UUID, session: SessionDep, tenant_id: TenantIdDep
) -> Response:
    await _require_project(session, project_id, tenant_id)
    panels = await StoryboardPanelRepository(session).list_for_project(
        project_id, tenant_id=tenant_id
    )
    scenes = await StoryboardSceneRepository(session).list_for_project(
        project_id, tenant_id=tenant_id
    )
    return Response(
        content=build_csv(panels, scenes),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="storyboard-{project_id}.csv"'},
    )


@router.get(
    "/api/projects/{project_id}/storyboard/export.edl",
    operation_id="storyboard.export.edl",
)
async def export_storyboard_edl(
    project_id: UUID, session: SessionDep, tenant_id: TenantIdDep
) -> Response:
    await _require_project(session, project_id, tenant_id)
    panels = await StoryboardPanelRepository(session).list_for_project(
        project_id, tenant_id=tenant_id
    )
    scenes = await StoryboardSceneRepository(session).list_for_project(
        project_id, tenant_id=tenant_id
    )
    return Response(
        content=build_edl(panels, scenes),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="storyboard-{project_id}.edl"'},
    )


@router.get(
    "/api/projects/{project_id}/storyboard/export.fcpxml",
    operation_id="storyboard.export.fcpxml",
)
async def export_storyboard_fcpxml(
    project_id: UUID, session: SessionDep, tenant_id: TenantIdDep
) -> Response:
    await _require_project(session, project_id, tenant_id)
    panels = await StoryboardPanelRepository(session).list_for_project(
        project_id, tenant_id=tenant_id
    )
    scenes = await StoryboardSceneRepository(session).list_for_project(
        project_id, tenant_id=tenant_id
    )
    return Response(
        content=build_fcpxml(panels, scenes),
        media_type="application/xml; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="storyboard-{project_id}.fcpxml"'},
    )


@router.get(
    "/api/projects/{project_id}/storyboard/export.pdf",
    operation_id="storyboard.export.pdf",
)
async def export_storyboard_pdf(
    project_id: UUID, session: SessionDep, tenant_id: TenantIdDep
) -> Response:
    """The PDF board — the same read shape as the CSV/EDL/FCPXML exports, plus frames.

    Unlike its three siblings this one wants BYTES, not just rows, so the panel
    images are resolved HERE (SSRF-guarded, size-capped, best-effort) and handed
    to the pure ``build_pdf`` builder as ``{panel_id: bytes}``.
    """
    project = await ProjectRepository(session).get(project_id, tenant_id=tenant_id)
    if project is None:
        raise HTTPException(404, "project not found")
    panels = await StoryboardPanelRepository(session).list_for_project(
        project_id, tenant_id=tenant_id
    )
    scenes = await StoryboardSceneRepository(session).list_for_project(
        project_id, tenant_id=tenant_id
    )
    images, unavailable = await _resolve_panel_images(panels, tenant_id=tenant_id)
    content = build_pdf(
        panels,
        scenes,
        project_name=project.title or f"Project {project_id}",
        images=images,
        unavailable=unavailable,
    )
    return Response(
        content=content,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="storyboard-{project_id}.pdf"'},
    )


@router.post(
    "/api/projects/{project_id}/storyboard/panels",
    response_model=StoryboardPanelResponse,
    status_code=status.HTTP_201_CREATED,
    operation_id="storyboard.panels.create",
)
async def create_panel(
    project_id: UUID, body: StoryboardPanelCreate, session: SessionDep, tenant_id: TenantIdDep
) -> StoryboardPanelResponse:
    storyboard = await StoryboardRepository(session).ensure_for_project(
        project_id, tenant_id=tenant_id
    )
    if storyboard is None:
        raise HTTPException(404, "project not found")
    data = body.model_dump(exclude_unset=True)
    position = data.pop("position", None)
    shot_key = data.pop("shot_key", None)
    try:
        row = await StoryboardPanelRepository(session).create(
            project_id=project_id,
            storyboard_id=storyboard.id,
            position=position,
            shot_key=shot_key,
            **data,
        )
        # Inside the ``try`` on purpose, same as patch_panel's first commit:
        # ``uq_storyboard_panels_project_shot`` is DEFERRABLE, so a shot_key
        # collision can surface at COMMIT rather than at flush, and the handler
        # below maps it to the same 409 either way. Outside the try it escaped
        # the ``except IntegrityError`` and returned 500.
        await session.commit()
    except UnknownAssetError as exc:
        raise _unknown_asset_http(exc) from exc
    except UnknownSceneError as exc:
        raise _unknown_scene_http(exc) from exc
    except IntegrityError as exc:
        # Keep the session usable after the failed flush before re-raising / 409.
        await session.rollback()
        conflict = _shot_key_conflict_http(exc, shot_key)
        if conflict is not None:
            raise conflict from exc
        raise
    return await _panel_resp(row, tenant_id)


@router.patch(
    "/api/projects/{project_id}/storyboard/panels/reorder",
    operation_id="storyboard.panels.reorder",
)
async def reorder_panels(
    project_id: UUID, body: ReorderPanels, session: SessionDep, tenant_id: TenantIdDep
) -> list[StoryboardPanelResponse]:
    # A missing/cross-tenant project is a 404 (the resource does not exist at this
    # address), NOT a 400 — 400 is reserved for a malformed ORDERING of an existing
    # project's panels. An archived project is 409 (read-only). Both are decided before
    # the repo's permutation check, which owns only the 400.
    #
    # Returns the panels in their new order, each carrying a FRESH ``lock_version`` (moved
    # rows had their token advanced). The client refreshes its cached tokens from this so a
    # post-reorder edit doesn't 409 against a token the reorder silently invalidated.
    await _require_editable_project(session, project_id, tenant_id)
    if body.expected_lock_versions is None:
        raise HTTPException(
            400, _BOARD_LOCK_REQUIRED.format(entity="panel", scope="the board order")
        )
    rows = await StoryboardPanelRepository(session).reorder(
        project_id,
        body.ordered_panel_ids,
        tenant_id=tenant_id,
        expected_lock_versions=body.expected_lock_versions,
    )
    if rows is None:  # None = non-permutation (400); [] = valid empty-board reorder.
        raise HTTPException(400, "invalid panel id in ordering")
    await session.commit()
    return [await _panel_resp(r, tenant_id) for r in rows]


@router.patch(
    "/api/projects/{project_id}/storyboard/scenes/reorder",
    operation_id="storyboard.scenes.reorder",
)
async def reorder_scenes(
    project_id: UUID, body: ReorderScenes, session: SessionDep, tenant_id: TenantIdDep
) -> list[StoryboardSceneResponse]:
    # Mirror of ``reorder_panels``: an atomic bulk reorder so the FE no longer fires N
    # per-scene PATCHes with a partial-failure window. Missing/cross-tenant project → 404,
    # archived → 409 (both via ``_require_editable_project``, decided before the repo's
    # permutation check, which owns only the 400). Returns the scenes in new order with
    # fresh ``lock_version`` tokens (same rationale as ``reorder_panels``).
    await _require_editable_project(session, project_id, tenant_id)
    if body.expected_lock_versions is None:
        raise HTTPException(
            400, _BOARD_LOCK_REQUIRED.format(entity="scene", scope="the board order")
        )
    rows = await StoryboardSceneRepository(session).reorder(
        project_id,
        body.ordered_scene_ids,
        tenant_id=tenant_id,
        expected_lock_versions=body.expected_lock_versions,
    )
    if rows is None:  # None = non-permutation (400); [] = valid empty-board reorder.
        raise HTTPException(400, "invalid scene id in ordering")
    await session.commit()
    return [_scene_resp(r) for r in rows]


async def _apply_panel_status_flip(  # noqa: ANN202
    panels: StoryboardPanelRepository,
    session: SessionDep,
    row,  # noqa: ANN001
    *,
    panel_id: UUID,
    new_status: str,
    tenant_id: UUID,
    subject_id: UUID,
    expected_lock_version: int | None,
):
    """Route a ``status`` PATCH through approve/unapprove so the audit trail +
    approved_at/by stay honest. No-op (returns ``row`` unchanged) when the status is
    already what was asked for; ``None`` if the panel vanished mid-flip (→ 404)."""
    if new_status == row.status:
        return row
    events = StoryboardPanelEventRepository(session)
    if new_status == "approved":
        return await panels.approve(
            panel_id,
            tenant_id=tenant_id,
            actor_subject_id=subject_id,
            events=events,
            expected_lock_version=expected_lock_version,
        )
    # 'draft' — the only other PanelStatus value.
    return await panels.unapprove(
        panel_id,
        tenant_id=tenant_id,
        actor_subject_id=subject_id,
        events=events,
        expected_lock_version=expected_lock_version,
    )


async def _flip_status_and_commit(  # noqa: ANN202
    panels: StoryboardPanelRepository,
    session: SessionDep,
    row,  # noqa: ANN001 — the repository's ORM row; mirrors _apply_panel_status_flip
    *,
    panel_id: UUID,
    new_status: str,
    tenant_id: UUID,
    subject_id: UUID,
    expected_lock_version: int | None,
    shot_key: str | None,
):
    """Apply the status flip and COMMIT it, mapping a collision to 409.

    Its own ``try``, rather than widening ``patch_panel``'s first one, for the
    same reason that one holds its own commit: the flip WRITES, and
    ``uq_storyboard_panels_project_shot`` is DEFERRABLE, so a collision from the
    non-status fields applied earlier in the same transaction can surface at THIS
    commit rather than at flush. With the commit outside any ``try`` it escaped
    the ``except IntegrityError`` and the caller got a 500 from a handler written
    to return 409.

    Narrow rather than wide: widening the caller's first ``try`` would also put
    the flip under its ``except UnknownAssetError`` / ``UnknownSceneError``
    clauses, which it has no business being under. Extracted to a function rather
    than left inline only because the extra branch pushed ``patch_panel`` past the
    complexity ceiling.

    The 404 below is an ``HTTPException`` and passes straight through
    ``except IntegrityError`` — it must stay AHEAD of the commit: a panel that
    vanished mid-flip must not commit.
    """
    try:
        row = await _apply_panel_status_flip(
            panels,
            session,
            row,
            panel_id=panel_id,
            new_status=new_status,
            tenant_id=tenant_id,
            subject_id=subject_id,
            expected_lock_version=expected_lock_version,
        )
        if row is None:
            raise HTTPException(404, "panel not found")
        await session.commit()
    except IntegrityError as exc:
        # The failed flush/commit poisons the session; roll back before mapping
        # (the create_panel idiom).
        await session.rollback()
        conflict = _shot_key_conflict_http(exc, shot_key)
        if conflict is not None:
            raise conflict from exc
        raise
    return row


@router.patch(
    "/api/storyboard/panels/{panel_id}",
    response_model=StoryboardPanelResponse,
    operation_id="storyboard.panels.update",
)
async def patch_panel(
    panel_id: UUID,
    body: StoryboardPanelUpdate,
    session: SessionDep,
    tenant_id: TenantIdDep,
    subject_id: SubjectIdDep,
) -> StoryboardPanelResponse:
    # A direct ``status`` PATCH (the FE dropdown still hits this route) is routed
    # through approve/unapprove so the audit trail + approved_at/by stay honest —
    # a raw ``setattr(status)`` would silently drift them. Everything runs in the
    # single request session, so it commits atomically.
    panels = StoryboardPanelRepository(session)
    updates = body.model_dump(exclude_unset=True)
    # Opt-in optimistic concurrency: not a column, so it never reaches the repo's field
    # setattr loop. A ``PanelVersionConflictError`` from the repo escapes to the app-level
    # handler → 409 (it is NOT caught by the UnknownAssetError block below).
    expected = updates.pop("expected_lock_version", None)
    # Captured before the field writes so the 409 message can name the offending key even
    # after the failed flush is rolled back (mirrors create_panel).
    shot_key = updates.get("shot_key")
    try:
        if "status" not in updates:
            row = await panels.update(
                panel_id, tenant_id=tenant_id, fields=updates, expected_lock_version=expected
            )
            if row is None:
                raise HTTPException(404, "panel not found")
            # Inside the ``try`` on purpose: ``uq_storyboard_panels_project_shot`` is
            # DEFERRABLE, so a shot_key collision can surface at COMMIT rather than at
            # flush, and the handler below maps it to the same 409 either way.
            await session.commit()
            return await _panel_resp(row, tenant_id)

        new_status = updates.pop("status")
        # Apply any remaining non-status fields first; else just load the panel so we
        # can compare current status + return it on a no-op.
        if updates:
            row = await panels.update(
                panel_id, tenant_id=tenant_id, fields=updates, expected_lock_version=expected
            )
            # The field write already consumed the CAS + advanced the token; the status
            # flip below rides the same transaction and must not re-assert the now-stale
            # expected value.
            expected = None
        else:
            row = await panels.get(panel_id, tenant_id=tenant_id)
    except UnknownAssetError as exc:
        raise _unknown_asset_http(exc) from exc
    except UnknownSceneError as exc:
        raise _unknown_scene_http(exc) from exc
    except IntegrityError as exc:
        # A shot_key PATCH that collides with a sibling → 409, not a raw 500. The failed
        # flush poisons the session, so roll back before mapping (create_panel idiom).
        await session.rollback()
        conflict = _shot_key_conflict_http(exc, shot_key)
        if conflict is not None:
            raise conflict from exc
        raise
    if row is None:
        raise HTTPException(404, "panel not found")

    row = await _flip_status_and_commit(
        panels,
        session,
        row,
        panel_id=panel_id,
        new_status=new_status,
        tenant_id=tenant_id,
        subject_id=subject_id,
        expected_lock_version=expected,
        shot_key=shot_key,
    )
    return await _panel_resp(row, tenant_id)


@router.delete(
    "/api/storyboard/panels/{panel_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    operation_id="storyboard.panels.delete",
)
async def delete_panel(
    panel_id: UUID,
    session: SessionDep,
    tenant_id: TenantIdDep,
    expected_lock_version: Annotated[int | None, Query(ge=0)] = None,
) -> None:
    """204. ``expected_lock_version`` is REQUIRED (400 without it, 409 when stale)."""
    if expected_lock_version is None:
        raise HTTPException(400, _LOCK_REQUIRED.format(entity="panel"))
    if not await StoryboardPanelRepository(session).delete(
        panel_id, tenant_id=tenant_id, expected_lock_version=expected_lock_version
    ):
        raise HTTPException(404, "panel not found")
    await session.commit()


# ─── Approval trail (Boords pattern) ─────────────────────────────────────
@router.post(
    "/api/storyboard/panels/{panel_id}/approve",
    response_model=StoryboardPanelResponse,
    operation_id="storyboard.panels.approve",
)
async def approve_panel(
    panel_id: UUID,
    body: PanelApproveRequest,
    session: SessionDep,
    tenant_id: TenantIdDep,
    subject_id: SubjectIdDep,
) -> StoryboardPanelResponse:
    row = await StoryboardPanelRepository(session).approve(
        panel_id,
        tenant_id=tenant_id,
        actor_subject_id=subject_id,
        note=body.note,
        events=StoryboardPanelEventRepository(session),
        expected_lock_version=body.expected_lock_version,
    )
    if row is None:
        raise HTTPException(404, "panel not found")
    await session.commit()
    return await _panel_resp(row, tenant_id)


@router.post(
    "/api/storyboard/panels/{panel_id}/unapprove",
    response_model=StoryboardPanelResponse,
    operation_id="storyboard.panels.unapprove",
)
async def unapprove_panel(
    panel_id: UUID,
    body: PanelApproveRequest,
    session: SessionDep,
    tenant_id: TenantIdDep,
    subject_id: SubjectIdDep,
) -> StoryboardPanelResponse:
    row = await StoryboardPanelRepository(session).unapprove(
        panel_id,
        tenant_id=tenant_id,
        actor_subject_id=subject_id,
        note=body.note,
        events=StoryboardPanelEventRepository(session),
        expected_lock_version=body.expected_lock_version,
    )
    if row is None:
        raise HTTPException(404, "panel not found")
    await session.commit()
    return await _panel_resp(row, tenant_id)


@router.get(
    "/api/projects/{project_id}/storyboard/events",
    response_model=PanelEventListResponse,
    operation_id="storyboard.events.list",
)
async def list_project_events(
    project_id: UUID, session: SessionDep, tenant_id: TenantIdDep
) -> PanelEventListResponse:
    await _require_project(session, project_id, tenant_id)
    rows = await StoryboardPanelEventRepository(session).list_for_project(
        project_id, tenant_id=tenant_id
    )
    items = [_event_resp(r) for r in rows]
    return PanelEventListResponse(items=items, total=len(items))


@router.get(
    "/api/storyboard/panels/{panel_id}/events",
    response_model=PanelEventListResponse,
    operation_id="storyboard.panels.events",
)
async def list_panel_events(
    panel_id: UUID, session: SessionDep, tenant_id: TenantIdDep
) -> PanelEventListResponse:
    rows = await StoryboardPanelEventRepository(session).list_for_panel(
        panel_id, tenant_id=tenant_id
    )
    if rows is None:
        raise HTTPException(404, "panel not found")
    items = [_event_resp(r) for r in rows]
    return PanelEventListResponse(items=items, total=len(items))


# ─── Scenes ──────────────────────────────────────────────────────────────
@router.get(
    "/api/projects/{project_id}/storyboard/scenes",
    response_model=StoryboardSceneListResponse,
    operation_id="storyboard.scenes.list",
)
async def list_scenes(
    project_id: UUID, session: SessionDep, tenant_id: TenantIdDep
) -> StoryboardSceneListResponse:
    await _require_project(session, project_id, tenant_id)
    rows = await StoryboardSceneRepository(session).list_for_project(
        project_id, tenant_id=tenant_id
    )
    items = [_scene_resp(r) for r in rows]
    return StoryboardSceneListResponse(items=items, total=len(items))


@router.post(
    "/api/projects/{project_id}/storyboard/scenes",
    response_model=StoryboardSceneResponse,
    status_code=status.HTTP_201_CREATED,
    operation_id="storyboard.scenes.create",
)
async def create_scene(
    project_id: UUID, body: StoryboardSceneCreate, session: SessionDep, tenant_id: TenantIdDep
) -> StoryboardSceneResponse:
    storyboard = await StoryboardRepository(session).ensure_for_project(
        project_id, tenant_id=tenant_id
    )
    if storyboard is None:
        raise HTTPException(404, "project not found")
    data = body.model_dump(exclude_unset=True)
    position = data.pop("position", None)
    row = await StoryboardSceneRepository(session).create(
        project_id=project_id, storyboard_id=storyboard.id, position=position, **data
    )
    await session.commit()
    return _scene_resp(row)


@router.patch(
    "/api/storyboard/scenes/{scene_id}",
    response_model=StoryboardSceneResponse,
    operation_id="storyboard.scenes.update",
)
async def patch_scene(
    scene_id: UUID, body: StoryboardSceneUpdate, session: SessionDep, tenant_id: TenantIdDep
) -> StoryboardSceneResponse:
    updates = body.model_dump(exclude_unset=True)
    # Opt-in optimistic concurrency: not a column, so pop it before the repo's field
    # setattr loop. A ``SceneVersionConflictError`` from the repo escapes to the app-level
    # handler → 409.
    expected = updates.pop("expected_lock_version", None)
    row = await StoryboardSceneRepository(session).update(
        scene_id, tenant_id=tenant_id, fields=updates, expected_lock_version=expected
    )
    if row is None:
        raise HTTPException(404, "scene not found")
    await session.commit()
    return _scene_resp(row)


@router.delete(
    "/api/storyboard/scenes/{scene_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    operation_id="storyboard.scenes.delete",
)
async def delete_scene(
    scene_id: UUID,
    session: SessionDep,
    tenant_id: TenantIdDep,
    expected_lock_version: Annotated[int | None, Query(ge=0)] = None,
) -> None:
    """204. ``expected_lock_version`` is REQUIRED (400 without it, 409 when stale)."""
    if expected_lock_version is None:
        raise HTTPException(400, _LOCK_REQUIRED.format(entity="scene"))
    if not await StoryboardSceneRepository(session).delete(
        scene_id, tenant_id=tenant_id, expected_lock_version=expected_lock_version
    ):
        raise HTTPException(404, "scene not found")
    await session.commit()


# ─── Assets ──────────────────────────────────────────────────────────────
@router.get(
    "/api/projects/{project_id}/storyboard/assets",
    response_model=StoryboardAssetListResponse,
    operation_id="storyboard.assets.list",
)
async def list_assets(
    project_id: UUID, session: SessionDep, tenant_id: TenantIdDep
) -> StoryboardAssetListResponse:
    await _require_project(session, project_id, tenant_id)
    rows = await StoryboardAssetRepository(session).list_for_project(
        project_id, tenant_id=tenant_id
    )
    items = [_asset_resp(r) for r in rows]
    return StoryboardAssetListResponse(items=items, total=len(items))


@router.post(
    "/api/projects/{project_id}/storyboard/assets",
    response_model=StoryboardAssetResponse,
    status_code=status.HTTP_201_CREATED,
    operation_id="storyboard.assets.create",
)
async def create_asset(
    project_id: UUID, body: StoryboardAssetCreate, session: SessionDep, tenant_id: TenantIdDep
) -> StoryboardAssetResponse:
    # Asset create does not pass through ``ensure_for_project`` (the panel/scene create
    # chokepoint), so the archived guard is applied here at the route.
    await _require_editable_project(session, project_id, tenant_id)
    data = body.model_dump(exclude_unset=True)
    position = data.pop("position", None)
    row = await StoryboardAssetRepository(session).create(
        project_id=project_id, position=position, **data
    )
    await session.commit()
    return _asset_resp(row)


@router.patch(
    "/api/storyboard/assets/{asset_id}",
    response_model=StoryboardAssetResponse,
    operation_id="storyboard.assets.update",
)
async def patch_asset(
    asset_id: UUID, body: StoryboardAssetUpdate, session: SessionDep, tenant_id: TenantIdDep
) -> StoryboardAssetResponse:
    updates = body.model_dump(exclude_unset=True)
    # Opt-in optimistic concurrency: not a column, so pop it before the repo's field
    # setattr loop. An ``AssetVersionConflictError`` from the repo escapes to the app-level
    # handler → 409.
    expected = updates.pop("expected_lock_version", None)
    row = await StoryboardAssetRepository(session).update(
        asset_id, tenant_id=tenant_id, fields=updates, expected_lock_version=expected
    )
    if row is None:
        raise HTTPException(404, "asset not found")
    await session.commit()
    return _asset_resp(row)


@router.delete(
    "/api/storyboard/assets/{asset_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    operation_id="storyboard.assets.delete",
    # This route has ALWAYS returned 409 at runtime (asset still referenced) but
    # declared neither it nor the 404 — so the generated client had no typed branch for
    # the one failure the UI must actually handle. Declared description-only, no
    # ``model``: the body is FastAPI's ``{"detail": …}`` (here a dict carrying the
    # referencing panels), which ``ErrorResponse`` ({error, detail: str}) does not
    # describe — a model here would swap one lie for another.
    responses={
        404: {"description": "asset not found"},
        409: {"description": "asset is still referenced by storyboard panels"},
    },
)
async def delete_asset(
    asset_id: UUID,
    session: SessionDep,
    tenant_id: TenantIdDep,
    expected_lock_version: Annotated[int | None, Query(ge=0)] = None,
) -> None:
    """204, or 409 when panels still reference the asset (the referencing panels are
    named in ``detail.panels`` so the UI can say "used by clip-1, clip-3").

    ``expected_lock_version`` is REQUIRED: 400 without it, 409 when stale."""
    if expected_lock_version is None:
        raise HTTPException(400, _LOCK_REQUIRED.format(entity="asset"))
    try:
        deleted = await StoryboardAssetRepository(session).delete(
            asset_id, tenant_id=tenant_id, expected_lock_version=expected_lock_version
        )
        # The 404 moves INSIDE with the commit and stays ahead of it: a CAS miss
        # or a missing asset must not commit. HTTPException passes straight
        # through the ``except IntegrityError`` below.
        #
        # Honest scope: unlike the merge routes, this handler is CONSISTENCY, not
        # a live 500. It never writes ``storyboard_panels`` (the repo only SELECTs
        # the referencing panels), so ``uq_storyboard_panels_project_shot`` cannot
        # fire here at all; and its own FK
        # (``storyboard_panel_assets_asset_id_fkey``, ON DELETE NO ACTION
        # DEFERRABLE INITIALLY IMMEDIATE) is never deferred on this path — nothing
        # here issues ``SET CONSTRAINTS`` — so it raises at the repo's flush,
        # already inside this try. The commit belongs in here anyway: the
        # constraint IS deferrable, and the next caller to defer it in the same
        # transaction should not have to rediscover this.
        if not deleted:
            raise HTTPException(404, "asset not found")
        await session.commit()
    except AssetInUseError as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail={"message": _ASSET_IN_USE, "panels": exc.panels},
        ) from exc
    except IntegrityError as exc:
        # Race: a panel gained a reference between the pre-check and the DELETE, so
        # the FK fires instead. Same 409, minus the panel list (we can't re-query
        # after the rollback and a guessed list would be a lie). Keep the session
        # usable before re-raising / 409 — the create_panel idiom.
        await session.rollback()
        orig = str(exc.orig) if exc.orig is not None else ""
        if "storyboard_panel_assets_asset_id_fkey" in orig:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                detail={"message": _ASSET_IN_USE, "panels": []},
            ) from exc
        raise


# ─── AI authoring (Phase 1) ──────────────────────────────────────────────
def get_assist_service(session: SessionDep) -> StoryboardAssistService:
    return StoryboardAssistService(session)


AssistServiceDep = Annotated[StoryboardAssistService, Depends(get_assist_service)]


@router.post(
    "/api/projects/{project_id}/storyboard/draft",
    response_model=StoryboardDraftResponse,
    status_code=status.HTTP_201_CREATED,
    operation_id="storyboard.draft",
)
async def draft_storyboard(
    project_id: UUID,
    body: StoryboardDraftRequest,
    session: SessionDep,
    tenant_id: TenantIdDep,
    service: AssistServiceDep,
) -> StoryboardDraftResponse:
    await _require_project(session, project_id, tenant_id)
    try:
        scenes, panels, assets = await service.draft(
            project_id=project_id, tenant_id=tenant_id, req=body
        )
    except DraftConflictError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except DraftInputError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    except UnknownAssetError as exc:
        # The resolver mints/looks up assets in THIS project, so this should be
        # unreachable — but a drifting resolver must surface as 422, never a 500.
        raise _unknown_asset_http(exc) from exc
    await session.commit()
    return StoryboardDraftResponse(
        scenes=[_scene_resp(s) for s in scenes],
        panels=[await _panel_resp(p, tenant_id) for p in panels],
        assets=[_asset_resp(a) for a in assets],
    )


@router.post(
    "/api/storyboard/panels/{panel_id}/rewrite",
    response_model=StoryboardPanelResponse,
    operation_id="storyboard.panels.rewrite",
)
async def rewrite_panel(
    panel_id: UUID,
    body: PanelRewriteRequest,
    session: SessionDep,
    tenant_id: TenantIdDep,
    service: AssistServiceDep,
) -> StoryboardPanelResponse:
    row = await service.rewrite(
        panel_id, tenant_id=tenant_id, instruction=body.instruction, fields=body.fields
    )
    if row is None:
        raise HTTPException(404, "panel not found")
    await session.commit()
    return await _panel_resp(row, tenant_id)


# ─── AI panel image (Phase 2) ────────────────────────────────────────────
def _panel_image_runner(request: Request) -> PanelImageJobRunner:
    """The process's panel-image job runner, created on first use.

    Lazy rather than lifespan-constructed so the hand-built test app (which does
    not run ``src/asgi.py``'s lifespan) gets one too; ``src/asgi.py`` still drains
    it on shutdown. Creation is race-free: there is no await between the read and
    the write, so the event loop cannot interleave two requests here.

    503 when Redis is down: without it the job record cannot be written, so the
    caller could never learn the outcome of a generation it just paid for.
    Refusing up front is the honest answer.

    ``app.state.redis`` is the w2v_common wrapper in production and a raw
    ``FakeRedis`` under the test app, so it is unwrapped HERE — once, at the
    boundary — and only the raw client is handed inward. Everything below is
    annotated ``Redis`` and uses redis-py semantics (``ex=``, verbatim values,
    ``RedisError`` propagating); passing the wrapper made that annotation a lie
    and 500'd on the first ``set``. See :mod:`src.cache.raw`.
    """
    redis = request.app.state.redis
    if redis is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "panel image generation is unavailable (no redis connection)",
        )
    try:
        raw = raw_redis(redis)
    except RuntimeError as exc:
        # The wrapper exists but never connected — same user-visible situation
        # as no Redis at all, so answer the same way instead of a 500.
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "panel image generation is unavailable (no redis connection)",
        ) from exc
    runner = getattr(request.app.state, "panel_image_jobs", None)
    if runner is None:
        runner = PanelImageJobRunner(session_factory=request.app.state.session_factory, redis=raw)
        request.app.state.panel_image_jobs = runner
    return runner


@router.post(
    "/api/storyboard/panels/{panel_id}/image",
    response_model=PanelImageJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    operation_id="storyboard.panels.generateImage",
)
async def generate_panel_image(
    panel_id: UUID,
    body: PanelImageRequest,
    request: Request,
    session: SessionDep,
    tenant_id: TenantIdDep,
) -> PanelImageJobResponse:
    """Queue an AI image for this panel; poll or watch the project SSE channel.

    202, not 200: an image is minutes of GPU work, far past any request budget.
    The panel's ``media_ref`` + ``gen`` are the durable result; the returned job
    is TTL-bounded progress. Progress also lands on ``projects:{project_id}``,
    the channel ``GET /api/projects/{id}/events`` already streams — storyboard
    rows carry no ``run_id``, so there is no run channel to use.
    """
    panel = await StoryboardPanelRepository(session).get(panel_id, tenant_id=tenant_id)
    if panel is None:
        raise HTTPException(404, "panel not found")
    # Archived projects are read-only; and this would spend money on a board
    # nobody can edit.
    await _require_editable_project(session, panel.project_id, tenant_id)
    # CAS at SUBMIT time (a compare, not a bump): a client whose panel is stale
    # is told so BEFORE a paid generation starts. The completion write later
    # touches only media_ref/gen/visual_type/source and so cannot clobber an
    # author's concurrent text edit — see StoryboardPanelImageService._persist.
    if body.expected_lock_version is not None and panel.lock_version != body.expected_lock_version:
        raise PanelVersionConflictError(panel_id)
    # This handler only READS in the request session — the durable panel write happens
    # later in ``PanelImageJobRunner._run``, which owns its own session and commits
    # there. Closing this transaction before the task is spawned means the background
    # session (a different connection) is never reading around an open snapshot of ours,
    # and a client that polls the job id immediately sees consistent panel state.
    await session.commit()
    return await _panel_image_runner(request).submit(
        panel_id=panel_id,
        project_id=panel.project_id,
        tenant_id=tenant_id,
        options=PanelImageOptions(
            size=body.size,
            provider=body.provider,
            seed=body.seed,
            negative_prompt=body.negative_prompt,
        ),
    )


# Registered BEFORE ``/image/{job_id}``: Starlette matches routes in declaration
# order and the job route's path param would otherwise swallow ``content``.
@router.get(
    "/api/storyboard/panels/{panel_id}/image/content",
    operation_id="storyboard.panels.imageContent",
    summary="Stream this panel's generated image from in-cluster MinIO",
)
async def get_panel_image_content(
    panel_id: UUID,
    request: Request,
    session: SessionDep,
    # why: loaded straight by <img> markup, which cannot set the X-Active-Tenant-Id
    # header — the scope-switch rides ?active_tenant= instead. get_tenant_id reads
    # both channels, so this dependency sees it either way.
    tenant_id: TenantIdDep,
) -> Response:
    """Serve the panel's own image to the browser.

    ``media_ref.url`` points at ``minio.test``, which does not resolve outside
    the cluster, so the SPA is handed this path instead (see
    ``api/panel_image_url``). The bytes are read with the orchestrator's own MinIO
    credentials, which carry no expiry — unlike the stored 1h presigned signature.

    There is deliberately **no** caller-supplied bucket or key — but resolving the
    panel is NOT on its own enough to authorise the object. ``media_ref`` is
    client-writable, so a caller can point its own panel at a stranger's key;
    authorising only the panel would make this a confused deputy, since MinIO is
    read with the orchestrator's credentials. ``panel_object_ref`` therefore also
    confines the key to a prefix derived from the panel's ``project_id`` and the
    authenticated tenant. Both gates are required; neither is sufficient.

    404 (not 403) for a panel of another tenant — the repository is tenant-scoped
    and a cross-tenant panel simply does not resolve, matching every sibling panel
    route. A panel with no usable image, and a panel whose ref points somewhere it
    does not own, are the same 404: the caller learns nothing from the difference.

    The ``Content-Type`` is derived from the object KEY rather than forwarded from
    MinIO's stored metadata. Every stage artifact written before
    ``storage_service.guess_artifact_content_type`` existed carries
    ``application/json`` in its metadata whatever it actually is, and those objects
    are not being re-written — so a browser gets the type from the extension the
    writer chose (``…/{media_id}.png``), which is the panel's own record of the
    format. Forwarding the stored value would keep serving PNGs as JSON.
    """
    panel = await StoryboardPanelRepository(session).get(panel_id, tenant_id=tenant_id)
    if panel is None:
        raise HTTPException(404, "panel not found")
    media_ref = cast("MediaRefJson", panel.media_ref or {})
    ref = panel_object_ref(media_ref.get("url"), project_id=panel.project_id, tenant_id=tenant_id)
    if ref is None:
        raise HTTPException(404, "panel image not found")
    bucket, object_key = ref
    return await stream_object(
        bucket, object_key, request, media_type=guess_artifact_content_type(object_key)
    )


@router.get(
    "/api/storyboard/panels/{panel_id}/image/{job_id}",
    response_model=PanelImageJobResponse,
    operation_id="storyboard.panels.imageJob",
)
async def get_panel_image_job(
    panel_id: UUID,
    job_id: str,
    request: Request,
    session: SessionDep,
    tenant_id: TenantIdDep,
) -> PanelImageJobResponse:
    """Poll one panel-image job.

    Tenancy is re-checked against the PANEL (the database), never against the
    job record — the record lives in Redis under a guessable-shaped id and is
    not an authorization artifact. A job whose ``panel_id`` does not match the
    path is a 404 for the same reason.

    ``media_ref.url`` is rewritten to the panel's image-proxy path on the way out
    (``api/panel_image_url``); the stored ``minio.test`` URL would not load in
    a browser.
    """
    if await StoryboardPanelRepository(session).get(panel_id, tenant_id=tenant_id) is None:
        raise HTTPException(404, "panel not found")
    record = await _panel_image_runner(request).store.get(job_id)
    if record is None or record.panel_id != panel_id:
        raise HTTPException(404, "image job not found")
    return proxied_job(record, tenant_id=tenant_id)


# ─── Version snapshots + restore (Phase 5) ───────────────────────────────
@router.post(
    "/api/projects/{project_id}/storyboard/snapshots",
    response_model=SnapshotResponse,
    status_code=status.HTTP_201_CREATED,
    operation_id="storyboard.snapshots.create",
)
async def create_snapshot(
    project_id: UUID,
    body: SnapshotCreateRequest,
    session: SessionDep,
    tenant_id: TenantIdDep,
    subject_id: SubjectIdDep,
) -> SnapshotResponse:
    await _require_editable_project(session, project_id, tenant_id)
    row = await StoryboardSnapshotService(session).create_snapshot(
        project_id,
        tenant_id=tenant_id,
        kind="manual",
        note=body.note,
        actor_subject=subject_id,
    )
    await session.commit()
    return _snapshot_resp(row, payload=row.payload)


@router.get(
    "/api/projects/{project_id}/storyboard/snapshots",
    response_model=SnapshotListResponse,
    operation_id="storyboard.snapshots.list",
)
async def list_snapshots(
    project_id: UUID,
    session: SessionDep,
    tenant_id: TenantIdDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> SnapshotListResponse:
    await _require_project(session, project_id, tenant_id)
    repo = StoryboardSnapshotRepository(session)
    rows = await repo.list_for_project(project_id, tenant_id=tenant_id, limit=limit, offset=offset)
    items = [_snapshot_resp(r) for r in rows]  # newest-first (repo order); counts omitted
    # ``total`` is the project-wide count, NOT len(items) — the page is bounded.
    total = await repo.count_for_project(project_id, tenant_id=tenant_id)
    return SnapshotListResponse(items=items, total=total)


@router.delete(
    "/api/projects/{project_id}/storyboard/snapshots/{snapshot_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    operation_id="storyboard.snapshots.delete",
)
async def delete_snapshot(
    project_id: UUID, snapshot_id: UUID, session: SessionDep, tenant_id: TenantIdDep
) -> None:
    """Delete a MANUAL snapshot. ``auto`` snapshots are machine-managed (minted by
    restore, capped by retention) — deleting one by hand would punch a hole in the
    undo trail, so they are 409, not 204. ``branch_base`` is machine-managed the same
    way (minted by fork, it IS the base side of the branch's three-way merge diff):
    hand-deleting one would CASCADE the branch row away and silently strand the fork,
    so it is 409 too — the only sanctioned way to remove it is to delete the branch."""
    await _require_editable_project(session, project_id, tenant_id)
    repo = StoryboardSnapshotRepository(session)
    row = await repo.get_meta(snapshot_id, project_id=project_id, tenant_id=tenant_id)
    if row is None:
        raise HTTPException(404, "snapshot not found")
    if row.kind in ("auto", "branch_base"):
        raise HTTPException(status.HTTP_409_CONFLICT, f"{row.kind} snapshots cannot be deleted")
    await repo.delete(snapshot_id, project_id=project_id, tenant_id=tenant_id)
    await session.commit()


@router.post(
    "/api/storyboard/snapshots/{snapshot_id}/restore",
    response_model=RestoreResponse,
    operation_id="storyboard.snapshots.restore",
)
async def restore_snapshot(
    snapshot_id: UUID,
    session: SessionDep,
    tenant_id: TenantIdDep,
    subject_id: SubjectIdDep,
    body: RestoreSnapshotRequest | None = None,
) -> RestoreResponse:
    """Roll the board back to a snapshot. ``expected_lock_versions`` is REQUIRED — this is
    the sharpest write in the storyboard (it rewrites scenes + panels + assets wholesale),
    so an unguarded restore discards a sibling's work with no trace. 400 without the map,
    409 when the live board no longer matches it."""
    if body is None or body.expected_lock_versions is None:
        raise HTTPException(
            400, _BOARD_LOCK_REQUIRED.format(entity="scene/panel/asset", scope="the whole board")
        )
    try:
        result = await StoryboardSnapshotService(session).restore(
            snapshot_id,
            tenant_id=tenant_id,
            actor_subject=subject_id,
            expected_lock_versions=body.expected_lock_versions,
        )
        # Inside the ``try`` on purpose, the create_panel / merge_branch idiom:
        # ``storyboard_snapshot_service.restore`` runs ``SET CONSTRAINTS ALL DEFERRED``
        # so the wholesale rewrite can hold a TRANSIENT duplicate on the DEFERRABLE
        # ``uq_storyboard_panels_project_shot`` mid-write. On that path the violation
        # can ONLY surface at COMMIT — outside the try it sailed past the
        # ``except IntegrityError`` below and the caller got a 500.
        await session.commit()
    except SnapshotNotFoundError as exc:
        raise HTTPException(404, "snapshot not found") from exc
    except IntegrityError as exc:
        # Restore rewrites scenes + panels + assets wholesale, so a writer that landed
        # a colliding shot_key between the caller's read and this write collides here.
        # Not ``_shot_key_conflict_http``: the caller supplied no shot_key, so that
        # mapper could only emit "pass an explicit shot_key", which is not advice a
        # restore caller can act on. Retryable 409 — re-read the board and restore again.
        # Roll back first so the session stays usable after the failed commit/flush.
        await session.rollback()
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "restore raced a concurrent write; re-read the board and retry",
        ) from exc
    return RestoreResponse(
        restored_snapshot_id=result["restored_snapshot_id"],
        auto_snapshot_id=result["auto_snapshot_id"],
    )


# ─── Branches: fork + lifecycle + three-way diff/merge (Phase 5) ─────────
def get_branch_service(session: SessionDep) -> StoryboardBranchService:
    return StoryboardBranchService(session)


def get_merge_service(session: SessionDep) -> StoryboardMergeService:
    return StoryboardMergeService(session)


BranchServiceDep = Annotated[StoryboardBranchService, Depends(get_branch_service)]
MergeServiceDep = Annotated[StoryboardMergeService, Depends(get_merge_service)]


async def _require_branch_at(
    session: SessionDep, project_id: UUID, branch_id: UUID, tenant_id: UUID
) -> None:
    """404 unless the branch exists AND is addressed through its ROOT project.

    Two checks, not one. The repo's ``get`` scopes by TENANT (``root_project_id ->
    projects.tenant_id``), which is the SECURITY boundary; it is not the ADDRESSING one.
    Without the ``root_project_id`` equality, a branch of project A could be diffed or
    merged through project B's URL by anyone in the same tenant and the path would be
    decorative. Same-tenant/wrong-project → 404, exactly like an unknown id: the branch
    does not exist *at this address*. (Same guard ``abandon_branch`` applies.)
    """
    branch = await StoryboardBranchRepository(session).get(branch_id, tenant_id=tenant_id)
    if branch is None or branch.root_project_id != project_id:
        raise HTTPException(404, "branch not found")


@router.post(
    "/api/projects/{project_id}/storyboard/branches",
    response_model=BranchResponse,
    status_code=status.HTTP_201_CREATED,
    operation_id="storyboard.branches.create",
    responses={
        # Forking a branch used to be a blanket 409 ("cannot branch a branch"); it is now
        # supported (#483 — a chain is closed bottom-up, guarded on the way UP by
        # OpenChildBranchesError, not by refusing the fork). The only failure here is a
        # missing/cross-tenant root → 404.
        404: {"description": "project not found"},
    },
)
async def create_branch(
    project_id: UUID,
    body: BranchCreateRequest,
    session: SessionDep,
    tenant_id: TenantIdDep,
    subject_id: SubjectIdDep,
    service: BranchServiceDep,
) -> BranchResponse:
    """Fork the project's storyboard into a hidden child project (201).

    ``project_id`` may itself be a BRANCH — that used to be a blanket 409 and is now
    supported. The fork is 1-hop whatever it forks: the link row points at ``project_id``,
    so the new branch's "root" is its IMMEDIATE parent and diff/merge/lock/RLS need no
    special case. The rule that keeps a chain sound is on the way back UP, not here — a
    branch with open children of its own refuses to merge or abandon (409), which forces
    the chain to close bottom-up instead of stranding the children in a parent that has
    already gone terminal. See ``OpenChildBranchesError``.
    """
    try:
        branch, _child = await service.create_branch(
            project_id,
            tenant_id=tenant_id,
            name=body.name,
            actor_subject=subject_id,
        )
    except ProjectNotFoundError as exc:
        # Missing and cross-tenant collapse to the same 404 by design (see the service):
        # a distinguishable 403 is an existence oracle across tenants.
        raise HTTPException(404, "project not found") from exc
    # ``session`` is the SAME object the injected ``BranchServiceDep`` was built from
    # (FastAPI caches ``SessionDep`` per request), so this commits the service's work.
    await session.commit()
    return _branch_resp(branch)


@router.get(
    "/api/projects/{project_id}/storyboard/branches",
    response_model=BranchListResponse,
    operation_id="storyboard.branches.list",
)
async def list_branches(
    project_id: UUID,
    session: SessionDep,
    tenant_id: TenantIdDep,
    # Named ``status_filter`` in Python, exposed as ``?status=`` — the bare name is taken
    # by the imported ``fastapi.status`` module. Typed as a Literal, not ``str``: an
    # unconstrained filter would turn a typo into a silently EMPTY page (the repo's WHERE
    # simply matches nothing) instead of the 422 it is.
    status_filter: Annotated[
        Literal["open", "merged", "abandoned"] | None, Query(alias="status")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> BranchListResponse:
    """One bounded, newest-first page of the root's branches. ``status`` is optional and
    means NO filter when omitted — which is NOT the same as "the open ones"."""
    await _require_project(session, project_id, tenant_id)
    repo = StoryboardBranchRepository(session)
    rows = await repo.list_for_root(
        project_id, tenant_id=tenant_id, status=status_filter, limit=limit, offset=offset
    )
    items = [_branch_resp(r) for r in rows]
    # ``total`` is the count under the SAME filter, NOT len(items) — the page is bounded,
    # and a len() "total" silently caps the UI's pager at one page.
    total = await repo.count_for_root(project_id, tenant_id=tenant_id, status=status_filter)
    return BranchListResponse(items=items, total=total)


@router.delete(
    "/api/projects/{project_id}/storyboard/branches/{branch_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    operation_id="storyboard.branches.abandon",
    responses={
        404: {"description": "branch not found"},
        409: {"description": "branch already merged, or it has open child branches"},
    },
)
async def abandon_branch(
    project_id: UUID,
    branch_id: UUID,
    session: SessionDep,
    tenant_id: TenantIdDep,
    service: BranchServiceDep,
) -> None:
    """Close a branch WITHOUT merging it and archive its child project (204).

    DELETE is the verb, but this is an ABANDON, not a destroy: the child project is
    archived, because its panels are the only record of the discarded work.

    A ``merged`` branch is 409, not a silent no-op: ``merged`` is terminal, the root
    already holds its content, and the merge already archived the child — so abandoning it
    would be a pointless re-archive of the very project the merge's provenance still points
    at. Re-abandoning an already-``abandoned`` branch is idempotent (204) — only ``merged``
    conflicts.
    """
    await _require_branch_at(session, project_id, branch_id, tenant_id)
    try:
        await service.abandon(branch_id, tenant_id=tenant_id)
    except BranchNotFoundError as exc:  # pragma: no cover — pre-checked above
        raise HTTPException(404, "branch not found") from exc
    except BranchAlreadyMergedError as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT, "branch already merged; it cannot be abandoned"
        ) from exc
    except OpenChildBranchesError as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"branch has {exc.open_children} open child branch(es); merge or abandon them first",
        ) from exc
    await session.commit()


@router.get(
    "/api/projects/{project_id}/storyboard/branches/{branch_id}/diff",
    response_model=BranchDiffResponse,
    operation_id="storyboard.branches.diff",
    responses={404: {"description": "branch not found"}, 409: {"description": "base is missing"}},
)
async def diff_branch(
    project_id: UUID,
    branch_id: UUID,
    session: SessionDep,
    tenant_id: TenantIdDep,
    service: MergeServiceDep,
) -> BranchDiffResponse:
    """Three-way diff: ``base`` (the fork's pinned ``branch_base`` snapshot) vs ``main``
    (the root's board NOW) vs ``branch`` (the child's board NOW).

    Entries are keyed by LINEAGE (``origin_id or id``), never by panel id — ids are
    re-minted per project, so an id-keyed diff would be pure noise.

    Readable for a CLOSED branch too (that is how you audit what a merge did); only
    APPLYING a closed branch is refused.

    ``root_fingerprint`` is the optimistic-concurrency token: hand it back to the merge,
    which 409s if the root moved in between. It is a content hash of the root's whole
    board (panels + scenes + assets), not a timestamp — a clock etag skews across the
    orchestrator's replicas and would fire on writes that changed nothing.
    """
    await _require_branch_at(session, project_id, branch_id, tenant_id)
    try:
        result = await service.diff(branch_id, tenant_id=tenant_id)
    except BranchNotFoundError as exc:  # pragma: no cover — pre-checked above
        raise HTTPException(404, "branch not found") from exc
    except BaseSnapshotMissingError as exc:
        # Unreachable by design (``branch_base`` is prune-immune and 409s on hand-DELETE),
        # but a base-less branch cannot be diffed three ways at all — and degrading to
        # "take everything from the branch" would be a data-loss bug wearing a diff's
        # clothes. Refuse loudly instead.
        raise HTTPException(
            status.HTTP_409_CONFLICT, "branch base snapshot is missing; the branch cannot be merged"
        ) from exc
    return BranchDiffResponse(
        branch_id=result.branch_id,
        root_project_id=result.root_project_id,
        child_project_id=result.child_project_id,
        status=result.status,
        base_snapshot_id=result.base_snapshot_id,
        root_fingerprint=result.root_fingerprint,
        conflict_count=result.conflict_count,
        mergeable_fields=sorted(FIELD_MERGEABLE),
        entries=[
            DiffEntry(
                key=e.key,
                kind=e.kind,
                shot_key=e.shot_key,
                base=e.base,
                main=e.main,
                branch=e.branch,
                changed_fields=e.changed_fields,
                main_changed_fields=e.main_changed_fields,
            )
            for e in result.entries
        ],
    )


@router.post(
    "/api/projects/{project_id}/storyboard/branches/{branch_id}/merge",
    response_model=MergeResponse,
    operation_id="storyboard.branches.merge",
    responses={
        404: {"description": "branch not found"},
        409: {
            "description": "branch not open, has open child branches, "
            "base missing, or the root moved (stale diff)"
        },
        422: {"description": "a conflicting panel has no explicit resolution"},
    },
)
async def merge_branch(
    project_id: UUID,
    branch_id: UUID,
    body: MergeRequest,
    session: SessionDep,
    tenant_id: TenantIdDep,
    subject_id: SubjectIdDep,
    service: MergeServiceDep,
) -> MergeResponse:
    """Apply the branch onto the root, then close it as ``merged``.

    ``root_fingerprint`` (from the diff) is REQUIRED and is the concurrency contract: if
    the root moved since the caller read the diff, this is a 409 rather than a silent
    clobber of whatever landed on main while the human was deciding.

    ``resolutions`` need only cover what the caller overrides — a non-conflict entry
    defaults to ``branch`` (merging the branch is what was asked for). A ``conflict`` has
    NO default: merging one unresolved is a 422, not a coin-flip, because picking a side
    is the decision the human opened the diff to make.

    409 also covers a ``merged``/``abandoned`` branch: both are terminal. Re-merging would
    replay stale content onto a root that has moved on; merging an abandoned branch would
    resurrect work its author explicitly discarded.

    The root is auto-snapshotted (``kind='auto'``) before anything is written, so a merge
    is reversible — restore ``auto_snapshot_id``. Panels are UPSERTed onto the root's
    existing ids, so the approval/event trail of every touched panel survives.
    """
    await _require_branch_at(session, project_id, branch_id, tenant_id)
    resolutions, field_resolutions = _split_resolutions(body.resolutions)
    try:
        result = await service.merge(
            branch_id,
            tenant_id=tenant_id,
            root_fingerprint=body.root_fingerprint,
            resolutions=resolutions,
            field_resolutions=field_resolutions,
            actor_subject=subject_id,
        )
        # Inside the ``try``, and here it is not defensive: ``merge_service.merge``
        # runs ``SET CONSTRAINTS ALL DEFERRED`` (merge_service.py, in the
        # ``to_write or to_delete`` branch) precisely so a reorder or shot_key
        # rename can hold a TRANSIENT duplicate on the DEFERRABLE
        # ``uq_storyboard_panels_project_shot`` mid-write. That defers the check to
        # COMMIT — so a real collision surfaces HERE and nowhere else. With this
        # line outside the try it sailed past the ``except IntegrityError`` below
        # and the caller got a raw 500 instead of the retryable 409 that block was
        # written to return.
        await session.commit()
    except BranchNotFoundError as exc:  # pragma: no cover — pre-checked above
        raise HTTPException(404, "branch not found") from exc
    except BranchNotOpenError as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT, f"branch is {exc.status}, not open; it cannot be merged"
        ) from exc
    except OpenChildBranchesError as exc:
        # Bottom-up or not at all: merging this branch is terminal, so its open children
        # would be left merging into a parent that can never carry them upward.
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"branch has {exc.open_children} open child branch(es); merge or abandon them first",
        ) from exc
    except BaseSnapshotMissingError as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT, "branch base snapshot is missing; the branch cannot be merged"
        ) from exc
    except StaleRootError as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail={
                "message": "project changed since the diff was computed; re-read the diff",
                "expected_fingerprint": exc.expected,
                "actual_fingerprint": exc.actual,
            },
        ) from exc
    except UnresolvedConflictError as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "message": "conflicting panels need an explicit resolution",
                "keys": [str(k) for k in exc.keys],
                # For a partially field-resolved conflict: key → the conflicting fields still
                # missing a branch/main pick, so the client knows exactly what to complete.
                "missing_fields": {str(k): v for k, v in exc.missing.items()},
            },
        ) from exc
    except FieldResolutionTargetError as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "message": "field_resolutions is only valid on a field-resolvable panel conflict",
                "keys": [str(k) for k in exc.keys],
            },
        ) from exc
    except KeepBothTargetError as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "message": "keep_both is only valid on a panel conflict",
                "keys": [str(k) for k in exc.keys],
            },
        ) from exc
    except IntegrityError as exc:
        # Belt-and-braces for the merge-vs-concurrent-write race: ``ensure_for_project``
        # now serializes create-vs-merge on the root projects row, but any residual
        # constraint collision (e.g. a shot_key minted by a writer that slipped a
        # different path) should surface as a retryable 409 — re-read the diff and retry —
        # never an uncaught 500. The session is rolled back so it stays usable.
        await session.rollback()
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "merge raced a concurrent write; re-read the diff and retry",
        ) from exc
    return MergeResponse.model_validate(result)


@router.get(
    "/api/projects/{project_id}/storyboard/branches/{branch_id}/diff-against/{target_branch_id}",
    response_model=BranchDiffResponse,
    operation_id="storyboard.branches.diffSibling",
    responses={
        404: {"description": "a branch was not found"},
        409: {"description": "the siblings have no shared parent base"},
        422: {"description": "the two branches are not siblings"},
    },
)
async def diff_sibling(
    project_id: UUID,
    branch_id: UUID,
    target_branch_id: UUID,
    session: SessionDep,
    tenant_id: TenantIdDep,
    service: MergeServiceDep,
) -> BranchDiffResponse:
    """Three-way diff for a SIBLING merge (branch ``branch_id`` into ``target_branch_id``,
    both forked off the same parent): ``base`` = the parent's shared ``branch_base`` (the
    common ancestor of both), ``main`` = the TARGET sibling's board now, ``branch`` = the
    SOURCE sibling's board now. Symmetric, unlike the branch->root diff.

    ``root_fingerprint`` here is the TARGET board's — hand it back to ``merge-into`` as the
    concurrency token."""
    await _require_branch_at(session, project_id, branch_id, tenant_id)
    try:
        result = await service.diff_sibling(branch_id, target_branch_id, tenant_id=tenant_id)
    except BranchNotFoundError as exc:
        raise HTTPException(404, "branch not found") from exc
    except NotSiblingsError as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "the target is not a sibling of this branch (they must share a parent)",
        ) from exc
    except SharedBaseMissingError as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "the branches have no shared parent base; a sibling merge is unavailable",
        ) from exc
    return BranchDiffResponse(
        branch_id=result.branch_id,
        root_project_id=result.root_project_id,
        child_project_id=result.child_project_id,
        status=result.status,
        base_snapshot_id=result.base_snapshot_id,
        root_fingerprint=result.root_fingerprint,
        conflict_count=result.conflict_count,
        mergeable_fields=sorted(FIELD_MERGEABLE),
        entries=[
            DiffEntry(
                key=e.key,
                kind=e.kind,
                shot_key=e.shot_key,
                base=e.base,
                main=e.main,
                branch=e.branch,
                changed_fields=e.changed_fields,
                main_changed_fields=e.main_changed_fields,
            )
            for e in result.entries
        ],
    )


@router.post(
    "/api/projects/{project_id}/storyboard/branches/{branch_id}/merge-into",
    response_model=SiblingMergeResponse,
    operation_id="storyboard.branches.mergeInto",
    responses={
        404: {"description": "a branch was not found"},
        409: {
            "description": "a branch is not open, the shared base is missing, "
            "or the target moved (stale diff)"
        },
        422: {"description": "not siblings, or a conflicting panel has no explicit resolution"},
    },
)
async def merge_into_sibling(
    project_id: UUID,
    branch_id: UUID,
    body: SiblingMergeRequest,
    session: SessionDep,
    tenant_id: TenantIdDep,
    subject_id: SubjectIdDep,
    service: MergeServiceDep,
) -> SiblingMergeResponse:
    """Apply branch ``branch_id``'s changes onto its SIBLING ``target_branch_id``.

    A cross-pollinate, not a lifecycle close: BOTH branches stay ``open``, neither child is
    archived, and the source is NOT marked ``merged``. Only the target board is written (and
    auto-snapshotted, so restore ``auto_snapshot_id`` to undo). ``target_fingerprint`` (from
    the sibling diff) is the concurrency contract on the target — a 409 if it moved. The diff
    is branch-centric on the SOURCE, so the target's own edits are preserved; a panel both
    touched differently is a conflict that needs an explicit resolution (422 otherwise)."""
    await _require_branch_at(session, project_id, branch_id, tenant_id)
    resolutions, field_resolutions = _split_resolutions(body.resolutions)
    try:
        result = await service.merge_into_sibling(
            branch_id,
            body.target_branch_id,
            tenant_id=tenant_id,
            target_fingerprint=body.target_fingerprint,
            resolutions=resolutions,
            field_resolutions=field_resolutions,
            actor_subject=subject_id,
        )
        # Same exposure as ``merge_branch``: ``merge_service.merge_into_sibling``
        # also runs ``SET CONSTRAINTS ALL DEFERRED``, so the DEFERRABLE
        # ``uq_storyboard_panels_project_shot`` is checked at COMMIT and a real
        # collision can only surface on this line. It must be inside the try or
        # the 409 below is unreachable for the one case that produces it.
        await session.commit()
    except BranchNotFoundError as exc:
        raise HTTPException(404, "branch not found") from exc
    except NotSiblingsError as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "the target is not a sibling of this branch (they must share a parent)",
        ) from exc
    except BranchNotOpenError as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT, f"a branch is {exc.status}, not open; it cannot be merged"
        ) from exc
    except SharedBaseMissingError as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "the branches have no shared parent base; a sibling merge is unavailable",
        ) from exc
    except StaleRootError as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail={
                "message": "the target moved since the diff was computed; re-read the diff",
                "expected_fingerprint": exc.expected,
                "actual_fingerprint": exc.actual,
            },
        ) from exc
    except UnresolvedConflictError as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "message": "conflicting panels need an explicit resolution",
                "keys": [str(k) for k in exc.keys],
                "missing_fields": {str(k): v for k, v in exc.missing.items()},
            },
        ) from exc
    except FieldResolutionTargetError as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "message": "field_resolutions is only valid on a field-resolvable panel conflict",
                "keys": [str(k) for k in exc.keys],
            },
        ) from exc
    except KeepBothTargetError as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "message": "keep_both is only valid on a panel conflict",
                "keys": [str(k) for k in exc.keys],
            },
        ) from exc
    except IntegrityError as exc:
        await session.rollback()
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "sibling merge raced a concurrent write; re-read the diff and retry",
        ) from exc
    return SiblingMergeResponse.model_validate(result)
