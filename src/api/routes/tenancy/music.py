"""Tenant music library (P6 ORCH-4) — ``/api/tenant/music``.

- ``GET``    any MEMBER (full ``TENANT_GATE``): the tenant's live tracks plus the upload
  ``limits`` (size cap, accepted formats, description length) so the UI never
  hardcodes them. Each track's ``url`` is the browser-reachable
  ``/api/tenant/music/{id}/content`` proxy path — NOT a presigned MinIO URL: the
  presign host is ``minio.test`` (strato sets no
  ``STORAGE_MINIO_EXTERNAL_ENDPOINT``), which a browser cannot resolve.
- ``GET /{id}/content`` any MEMBER: streams the track's bytes from MinIO with the
  orchestrator's own credentials (``api.minio_stream``, the storyboard-panel image
  pattern), range-aware so ``<audio>`` can seek. Only a live track of the caller's
  tenant whose key sits under that tenant's library prefix is served.
- ``POST``   multipart, tenant_admin (``TenantAdminDep`` + ``LICENSE_GATE_ONLY`` + RLS
  bound to the authorized effective tenant, the tenant-admin posture of
  ``/api/tenant/connections``). The rights attestation is required. The format is
  SNIFFED from the bytes (mp3/wav/ogg/flac); the client's content-type and extension
  are ignored. Size is capped by ``STORAGE_TENANT_MUSIC_MAX_BYTES``, and the duration
  is measured with ffprobe (``audio_premaster.probe_seconds``): a file ffprobe cannot
  decode is refused rather than stored as a bed that would render as silence. Any of
  those refusals is a 422.
- ``PATCH /{id}`` tenant_admin: description / category. Re-categorising while a run
  of the tenant is active keeps the old category resolvable for pinned picks
  (``pinned_categories``) until no run is active.
- ``DELETE /{id}`` tenant_admin: soft delete. The track leaves the pickable library
  at once; its object is deleted when no run of the tenant is active (a run that
  already pinned it keeps rendering it) — see ``services.tenant_sound_library``.

Every write regenerates the tenant manifest inside the same transaction, under the
tenant's advisory lock, and commits before responding. Uploads, edits and deletes
audit ``music.upload`` / ``music.update`` / ``music.delete`` (0082; ``music.update``
added by 0084). Tenant tracks are listed by
``GET /api/music/tracks`` alongside the bundled library and are picked by the pipeline
through the same tenant-scoped ``load_sound_library``.
"""

from __future__ import annotations

import datetime
import logging
import re
from typing import Annotated, Any, Protocol, cast, get_args
from uuid import UUID, uuid7

from fastapi import APIRouter, Depends, File, Form, Request, Response, UploadFile, status
from fastapi import params as fastapi_params
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.minio_stream import stream_object
from src.api.routes._common import (
    LICENSE_GATE_ONLY,
    TENANT_GATE,
    AuditRepoDep,
    SessionDep,
    SubjectIdDep,
    TenantAdminDep,
    TenantIdDep,
)
from src.api.routes.tenancy.connections import bind_admin_tenant_rls
from src.api.routes.tenancy.errors import TenancyError
from src.api.schemas.tenant_music import (
    DESCRIPTION_MAX_LENGTH,
    TenantMusicFormat,
    TenantMusicLimits,
    TenantMusicList,
    TenantMusicPatch,
    TenantMusicTrack,
    TenantSoundCategory,
)
from src.config import get_settings
from src.database.models import TenantSoundAsset
from src.services.audio_premaster import probe_seconds
from src.services.stages.sound_catalog import TENANT_SOUND_ID_PREFIX
from src.services.tenant_sound_library import (
    AUDIO_FORMATS,
    key_belongs_to_tenant,
    lock_tenant_library,
    regenerate_tenant_manifest,
    sniff_audio_format,
    tenant_has_active_runs,
    tenant_object_key,
)
from src.services.tts_storage import _get_storage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tenant/music", tags=["tenancy"])

_ADMIN_GATE: list[fastapi_params.Depends] = [
    *LICENSE_GATE_ONLY,
    Depends(bind_admin_tenant_rls),
]

_CATEGORIES: tuple[TenantSoundCategory, ...] = get_args(TenantSoundCategory)
_SLUG_RE = re.compile(r"^[a-z0-9_]{1,48}$")
#: SQLSTATE ``unique_violation`` (a concurrent upload took the same id).
_UNIQUE_VIOLATION = "23505"
_FK_VIOLATION = "23503"


class _MusicStorage(Protocol):
    async def put_object(
        self, key: str, data: bytes, content_type: str = "application/octet-stream"
    ) -> str: ...

    async def delete_object(self, key: str) -> bool: ...


def _err(http: int, code: str, message: str, **details: object) -> TenancyError:
    return TenancyError(http, code, message, details or None)


async def _storage() -> _MusicStorage:
    """The storage singleton, with the off-protocol methods this module needs
    (resolved by ``getattr`` — the ``_StorageLike`` house pattern)."""
    storage = await _get_storage()
    for name in ("put_object", "delete_object"):
        if getattr(storage, name, None) is None:
            raise _err(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "STORAGE_UNAVAILABLE",
                f"artifact storage does not support {name}",
            )
    return cast(_MusicStorage, storage)


def content_path(track_id: UUID) -> str:
    """The relative, same-origin path a browser plays this track from."""
    return f"/api/tenant/music/{track_id}/content"


def _url(row: TenantSoundAsset) -> str | None:
    if not key_belongs_to_tenant(row.object_key, row.tenant_id):  # never proxy a foreign key
        return None
    return content_path(row.id)


def _limits() -> TenantMusicLimits:
    return TenantMusicLimits(
        max_bytes=get_settings().storage.tenant_music_max_bytes,
        max_duration_s=None,
        max_description_length=DESCRIPTION_MAX_LENGTH,
        categories=sorted(_CATEGORIES),
        formats=[
            TenantMusicFormat(format=fmt, extension=ext, content_type=ctype)
            for fmt, (ext, ctype) in AUDIO_FORMATS.items()
        ],
    )


def _track(row: TenantSoundAsset) -> TenantMusicTrack:
    return TenantMusicTrack(
        id=row.id,
        sound_id=row.sound_id,
        category=cast(Any, row.category),
        description=row.description,
        duration_s=row.duration_s,
        bytes=row.bytes,
        content_type=row.content_type,
        rights_attested_by=row.rights_attested_by,
        created_at=row.created_at,
        url=_url(row),
    )


def _sound_id(slug: str | None, filename: str | None) -> str:
    if slug is not None:
        slug = slug.strip().lower()
        if not _SLUG_RE.fullmatch(slug):
            raise _err(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                "INVALID_SLUG",
                "slug must be 1-48 characters of a-z, 0-9 and _",
            )
        return f"{TENANT_SOUND_ID_PREFIX}{slug}"
    stem = (filename or "").rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
    derived = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")[:48] or uuid7().hex[-12:]
    return f"{TENANT_SOUND_ID_PREFIX}{derived}"


async def _read_capped(file: UploadFile, request: Request, cap: int) -> bytes:
    too_large = _err(
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        "FILE_TOO_LARGE",
        f"audio exceeds the {cap // (1024 * 1024)} MiB limit",
        max_bytes=cap,
    )
    declared = request.headers.get("content-length")
    # The header covers the whole multipart body (>= the file), so only an
    # over-cap value is conclusive; the read below is the real enforcement.
    if declared and declared.isdigit() and int(declared) > cap + 64 * 1024:
        raise too_large
    data = await file.read(cap + 1)
    if len(data) > cap:
        raise too_large
    if not data:
        raise _err(status.HTTP_422_UNPROCESSABLE_CONTENT, "EMPTY_FILE", "empty upload")
    return data


async def _sniff_and_probe(data: bytes) -> tuple[str, float]:
    """The sniffed container and the ffprobe'd duration, or a 422."""
    fmt = sniff_audio_format(data)
    if fmt is None:
        raise _err(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "UNSUPPORTED_AUDIO",
            "the file is not MP3, WAV, OGG or FLAC audio",
        )
    # Force the SNIFFED demuxer: ffprobe's own detection skips an ID3 tag and would
    # probe ``ID3`` + an HLS playlist as a playlist (audio_premaster.probe_seconds).
    seconds = await probe_seconds(data, fmt, demuxer=fmt)
    if seconds is None or seconds <= 0:
        raise _err(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "UNDECODABLE_AUDIO",
            "the audio could not be decoded or has no duration",
        )
    return fmt, seconds


async def _flush_new_row(session: AsyncSession, row: TenantSoundAsset) -> None:
    """INSERT the row; map a lost unique race to 409 and an unknown tenant to 404."""
    session.add(row)
    try:
        await session.flush()
    except IntegrityError as exc:
        await session.rollback()
        state = getattr(exc.orig, "sqlstate", None)
        if state == _UNIQUE_VIOLATION:
            raise _err(
                status.HTTP_409_CONFLICT,
                "SOUND_ID_TAKEN",
                f"{row.sound_id!r} is already in use; choose another slug",
                sound_id=row.sound_id,
            ) from None
        if state == _FK_VIOLATION:
            raise _err(
                status.HTTP_404_NOT_FOUND, "TENANT_NOT_FOUND", "tenant is not provisioned"
            ) from None
        raise


async def _live_row(session: AsyncSession, tenant_id: UUID, track_id: UUID) -> TenantSoundAsset:
    """The live row, read UNDER the tenant library lock (taken here, held to commit).

    Locking before the read is what makes two concurrent writes of one track
    serialise on a fresh read: the second DELETE waits for the first to commit,
    then finds the row retired or purged and answers 404 — instead of flushing an
    UPDATE against a row the first already deleted (``StaleDataError`` → 500)."""
    await lock_tenant_library(session, tenant_id)
    row = (
        await session.execute(
            select(TenantSoundAsset).where(
                TenantSoundAsset.id == track_id,
                TenantSoundAsset.tenant_id == tenant_id,
                TenantSoundAsset.deleted_at.is_(None),
            )
        )
    ).scalar_one_or_none()
    if row is None:
        raise _err(status.HTTP_404_NOT_FOUND, "TRACK_NOT_FOUND", "track not found")
    return row


async def _regenerate_or_503(
    session: AsyncSession, storage: _MusicStorage, tenant_id: UUID
) -> None:
    try:
        await regenerate_tenant_manifest(session, storage, tenant_id)
    except Exception as exc:  # noqa: BLE001 -- surfaced as 503, the transaction is rolled back
        await session.rollback()
        logger.warning("tenant %s: manifest regeneration failed (%s)", tenant_id, exc)
        raise _err(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "STORAGE_UNAVAILABLE",
            "could not update the tenant sound library; retry",
        ) from None


@router.get(
    "",
    response_model=TenantMusicList,
    operation_id="tenant.music.list",
    dependencies=TENANT_GATE,
)
async def list_tenant_music(tenant_id: TenantIdDep, session: SessionDep) -> TenantMusicList:
    rows = (
        (
            await session.execute(
                select(TenantSoundAsset)
                .where(
                    TenantSoundAsset.tenant_id == tenant_id,
                    TenantSoundAsset.deleted_at.is_(None),
                )
                .order_by(TenantSoundAsset.created_at, TenantSoundAsset.id)
            )
        )
        .scalars()
        .all()
    )
    return TenantMusicList(tracks=[_track(r) for r in rows], limits=_limits())


@router.get(
    "/{track_id}/content",
    operation_id="tenant.music.content",
    dependencies=TENANT_GATE,
    summary="Stream one tenant track from in-cluster MinIO",
)
async def get_tenant_music_content(
    track_id: UUID, request: Request, tenant_id: TenantIdDep, session: SessionDep
) -> Response:
    """The bytes behind a track's ``url``. 404 for another tenant's track, a deleted
    one, or one whose key is outside the tenant's prefix — indistinguishable."""
    row = (
        await session.execute(
            select(TenantSoundAsset).where(
                TenantSoundAsset.id == track_id,
                TenantSoundAsset.tenant_id == tenant_id,
                TenantSoundAsset.deleted_at.is_(None),
            )
        )
    ).scalar_one_or_none()
    if row is None or not key_belongs_to_tenant(row.object_key, tenant_id):
        raise _err(status.HTTP_404_NOT_FOUND, "TRACK_NOT_FOUND", "track not found")
    return await stream_object(
        get_settings().storage.minio_bucket,
        row.object_key,
        request,
        media_type=row.content_type,
    )


@router.post(
    "",
    response_model=TenantMusicTrack,
    status_code=status.HTTP_201_CREATED,
    operation_id="tenant.music.upload",
    dependencies=_ADMIN_GATE,
)
async def upload_tenant_music(
    request: Request,
    tenant_id: TenantAdminDep,
    actor: SubjectIdDep,
    audit: AuditRepoDep,
    session: SessionDep,
    file: Annotated[UploadFile, File()],
    rights_attested: Annotated[bool, Form()] = False,
    category: Annotated[str, Form()] = "music",
    description: Annotated[str, Form(max_length=DESCRIPTION_MAX_LENGTH)] = "",
    slug: Annotated[str | None, Form()] = None,
) -> TenantMusicTrack:
    if not rights_attested:
        raise _err(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "RIGHTS_NOT_ATTESTED",
            "confirm that you hold the rights to use this audio commercially "
            "(rights_attested=true)",
        )
    if category not in _CATEGORIES:
        raise _err(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "INVALID_CATEGORY",
            f"category must be one of {sorted(_CATEGORIES)}",
        )
    sound_id = _sound_id(slug, file.filename)
    data = await _read_capped(file, request, get_settings().storage.tenant_music_max_bytes)
    fmt, seconds = await _sniff_and_probe(data)

    taken = (
        await session.execute(
            select(TenantSoundAsset.id).where(
                TenantSoundAsset.tenant_id == tenant_id, TenantSoundAsset.sound_id == sound_id
            )
        )
    ).first()
    if taken is not None:
        # Includes a RETIRED row: reusing its id while a run may still resolve it
        # would silently swap that run's bed for this upload.
        raise _err(
            status.HTTP_409_CONFLICT,
            "SOUND_ID_TAKEN",
            f"{sound_id!r} is already in use (or still retiring); choose another slug",
            sound_id=sound_id,
        )

    ext, content_type = AUDIO_FORMATS[fmt]
    asset_id = uuid7()
    key = tenant_object_key(tenant_id, category, asset_id, ext)
    storage = await _storage()
    await storage.put_object(key, data, content_type)
    try:
        row = TenantSoundAsset(
            id=asset_id,
            tenant_id=tenant_id,
            sound_id=sound_id,
            category=category,
            description=description.strip(),
            object_key=key,
            duration_s=seconds,
            bytes=len(data),
            content_type=content_type,
            rights_attested_by=actor,
        )
        await _flush_new_row(session, row)
        await _regenerate_or_503(session, storage, tenant_id)
    except Exception:
        # Nothing references the object now; do not strand it.
        try:
            await storage.delete_object(key)
        except Exception as exc:  # noqa: BLE001 -- best-effort cleanup
            logger.warning("tenant sound %s: orphan cleanup failed (%s)", asset_id, exc)
        raise
    await audit.record(
        actor_subject_id=actor,
        action="music.upload",
        target_tenant_id=tenant_id,
        detail={
            "id": str(asset_id),
            "sound_id": sound_id,
            "category": category,
            "bytes": len(data),
            "duration_s": seconds,
            "content_type": content_type,
            "rights": "tenant-declared",
        },
    )
    await session.commit()
    await session.refresh(row)
    return _track(row)


@router.patch(
    "/{track_id}",
    response_model=TenantMusicTrack,
    operation_id="tenant.music.update",
    dependencies=_ADMIN_GATE,
)
async def update_tenant_music(
    track_id: UUID,
    body: TenantMusicPatch,
    tenant_id: TenantAdminDep,
    actor: SubjectIdDep,
    audit: AuditRepoDep,
    session: SessionDep,
) -> TenantMusicTrack:
    row = await _live_row(session, tenant_id, track_id)
    before = {"description": row.description, "category": row.category}
    if body.description is not None:
        row.description = body.description
    if body.category is not None and body.category != row.category:
        # A run that already picked this track under its old category must keep
        # resolving it (a run pins its config for its whole lifetime).
        if await tenant_has_active_runs(session, tenant_id):
            row.pinned_categories = sorted({*row.pinned_categories, row.category})
        row.category = body.category
    changed = sorted(k for k, v in before.items() if getattr(row, k) != v)
    await session.flush()
    storage = await _storage()
    await _regenerate_or_503(session, storage, tenant_id)
    await audit.record(
        actor_subject_id=actor,
        action="music.update",
        target_tenant_id=tenant_id,
        detail={
            "id": str(track_id),
            "sound_id": row.sound_id,
            "changed": changed,
            "before": {k: before[k] for k in changed},
            "after": {k: getattr(row, k) for k in changed},
        },
    )
    await session.commit()
    await session.refresh(row)
    return _track(row)


@router.delete(
    "/{track_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    operation_id="tenant.music.delete",
    dependencies=_ADMIN_GATE,
)
async def delete_tenant_music(
    track_id: UUID,
    tenant_id: TenantAdminDep,
    actor: SubjectIdDep,
    audit: AuditRepoDep,
    session: SessionDep,
) -> Response:
    row = await _live_row(session, tenant_id, track_id)
    sound_id = row.sound_id
    row.deleted_at = datetime.datetime.now(datetime.UTC)
    await session.flush()
    storage = await _storage()
    await _regenerate_or_503(session, storage, tenant_id)
    # Purged = no active run needed it, object and row are gone; otherwise retired.
    purged = (
        await session.execute(select(TenantSoundAsset.id).where(TenantSoundAsset.id == track_id))
    ).first() is None
    await audit.record(
        actor_subject_id=actor,
        action="music.delete",
        target_tenant_id=tenant_id,
        detail={"id": str(track_id), "sound_id": sound_id, "object_deleted": purged},
    )
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
