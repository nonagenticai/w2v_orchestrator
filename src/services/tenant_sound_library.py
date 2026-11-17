"""Tenant music library (P6 ORCH-4): the ``tenant_sound_assets`` table → manifest.

The table is the source of truth; ``sound_library/tenants/{t}/manifest.json`` (what
``sound_catalog.load_sound_library`` reads) is DERIVED from it here, after every
change, so the two can only disagree for the instant between a commit and the next
regeneration.

Concurrency: :func:`regenerate_tenant_manifest` takes
``pg_advisory_xact_lock`` keyed on the tenant inside the caller's transaction, reads
every row AFTER acquiring it, writes the manifest, and the lock is released by the
caller's commit. Two concurrent uploads therefore serialise: the second waits for the
first to COMMIT, and under READ COMMITTED its post-lock read sees the first's row, so
the last manifest written is always a superset of both. A regeneration whose
transaction then fails leaves a manifest that may name a row that never committed;
the next regeneration corrects it, and a dangling id only resolves to a missing
object (the bed is dropped and recorded as a degradation, never a crash).

Pinning (a running run must not lose a track deleted under it): a delete is a SOFT
delete. While any run of the tenant is ``pending``/``running``/``paused``, the deleted
row stays in the manifest marked ``retired`` — it still resolves for an id the run
already pinned, but is never offered for a new pick (``SoundLibrary.is_valid`` /
``ids_in``). The first regeneration that finds NO active run purges retired rows:
object deleted, row removed. A retired row whose object delete fails is kept and
retried on the next regeneration. Known limit: purging is triggered by the next
library change, not by a timer, so a retired object can outlive its run until the
tenant changes its library again (offboarding deletes the whole prefix regardless).

Re-categorising is pinned the same way: a run pins its config for its whole lifetime,
so a PATCH that moves a track from ``music`` to ``ambient`` while a run is active
records ``music`` in the row's ``pinned_categories``. The manifest carries them, and
``SoundLibrary.resolves`` (the path for an id a run already picked) accepts a pinned
category, while ``is_valid``/``ids_in`` (new picks) see only the live one. The same
no-active-run regeneration that purges retired rows clears them.

Other pods pick a new manifest up within ``sound_catalog._LIB_TTL_SECONDS``.
"""

from __future__ import annotations

import json
import logging
from typing import Protocol
from uuid import UUID

from sqlalchemy import exists, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import Deliverable, Project, Run, TenantSoundAsset
from src.services.stages.sound_catalog import (
    clear_sound_library_cache,
    tenant_manifest_key,
    tenant_prefix,
)

logger = logging.getLogger(__name__)

#: Run statuses that may still resolve a sound (``ck_runs_status`` minus terminal).
ACTIVE_RUN_STATUSES: tuple[str, ...] = ("pending", "running", "paused")

#: Sniffed container → (file extension, stored content type). The client's
#: multipart content-type header is never trusted; the bytes decide.
AUDIO_FORMATS: dict[str, tuple[str, str]] = {
    "mp3": ("mp3", "audio/mpeg"),
    "wav": ("wav", "audio/wav"),
    "ogg": ("ogg", "audio/ogg"),
    "flac": ("flac", "audio/flac"),
}


class _LibraryStorage(Protocol):
    async def put_object(
        self, key: str, data: bytes, content_type: str = "application/octet-stream"
    ) -> str: ...

    async def delete_object(self, key: str) -> bool: ...


def sniff_audio_format(data: bytes) -> str | None:
    """The container ``data`` really is (``mp3``/``wav``/``ogg``/``flac``), or ``None``.

    Magic bytes only — the extension and the multipart content-type are client
    claims. The MPEG frame-sync test is loose (any ``0xFFEx``), so a sniffed
    ``mp3`` is only provisional: the ffprobe duration probe that follows is the
    decodability gate.
    """
    if data.startswith(b"fLaC"):
        return "flac"
    if data.startswith(b"OggS"):
        return "ogg"
    if data.startswith(b"RIFF") and data[8:12] == b"WAVE":
        return "wav"
    if data.startswith(b"ID3"):
        return "mp3"
    if len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
        return "mp3"
    return None


def tenant_object_key(tenant_id: UUID, category: str, asset_id: UUID, ext: str) -> str:
    """``sound_library/tenants/{t}/{category}/{asset_id}.{ext}`` — upload-addressed,
    never derived from a client filename."""
    return f"{tenant_prefix(str(tenant_id))}/{category}/{asset_id}.{ext}"


def key_belongs_to_tenant(key: str, tenant_id: UUID) -> bool:
    """True when ``key`` sits under ``tenant_id``'s own library prefix."""
    return key.startswith(f"{tenant_prefix(str(tenant_id))}/")


async def tenant_has_active_runs(session: AsyncSession, tenant_id: UUID) -> bool:
    """Is any run of ``tenant_id`` still able to resolve a sound?"""
    stmt = select(
        exists()
        .where(Run.deliverable_id == Deliverable.id)
        .where(Deliverable.project_id == Project.id)
        .where(Project.tenant_id == tenant_id)
        .where(Run.status.in_(ACTIVE_RUN_STATUSES))
    )
    return bool((await session.execute(stmt)).scalar())


async def lock_tenant_library(session: AsyncSession, tenant_id: UUID) -> None:
    """Serialise library writers of one tenant until the caller's transaction ends."""
    await session.execute(
        text("SELECT pg_advisory_xact_lock(hashtextextended(:k, 0))"),
        {"k": f"tenant_sound_library:{tenant_id}"},
    )


def _entry(row: TenantSoundAsset) -> dict[str, object]:
    entry: dict[str, object] = {
        "key": row.object_key,
        "description": row.description,
        "category": row.category,
        "duration_s": row.duration_s,
        "retired": row.deleted_at is not None,
    }
    pinned = sorted(set(row.pinned_categories or ()) - {row.category})
    if pinned:
        entry["pinned_categories"] = pinned
    return entry


async def _purge_retired(
    session: AsyncSession, storage: _LibraryStorage, rows: list[TenantSoundAsset]
) -> set[UUID]:
    """Delete the objects of retired rows and the rows themselves; return purged ids.

    A row whose object could not be deleted is kept (still retired) so the next
    regeneration retries it — dropping the row would strand the object forever."""
    purged: set[UUID] = set()
    for row in rows:
        try:
            deleted = await storage.delete_object(row.object_key)
        except Exception as exc:  # noqa: BLE001 -- retried on the next regeneration
            logger.warning("tenant sound %s: object delete failed (%s)", row.id, exc)
            continue
        if not deleted:
            logger.warning("tenant sound %s: object delete reported failure", row.id)
            continue
        await session.delete(row)
        purged.add(row.id)
    if purged:
        await session.flush()
    return purged


async def regenerate_tenant_manifest(
    session: AsyncSession, storage: _LibraryStorage, tenant_id: UUID
) -> dict[str, dict[str, object]]:
    """Rebuild and store ``tenant_id``'s manifest from the table; return it.

    Must run inside the transaction that made the change, BEFORE its commit (the
    advisory lock is transaction-scoped). Purges retired rows and clears pinned
    categories when no run of the tenant is active (module doc)."""
    await lock_tenant_library(session, tenant_id)
    rows = list(
        (
            await session.execute(
                select(TenantSoundAsset)
                .where(TenantSoundAsset.tenant_id == tenant_id)
                .order_by(TenantSoundAsset.created_at, TenantSoundAsset.id)
            )
        )
        .scalars()
        .all()
    )
    retired = [r for r in rows if r.deleted_at is not None]
    pinned = [r for r in rows if r.pinned_categories]
    if (retired or pinned) and not await tenant_has_active_runs(session, tenant_id):
        purged = await _purge_retired(session, storage, retired)
        rows = [r for r in rows if r.id not in purged]
        for row in pinned:
            if row.id not in purged:
                row.pinned_categories = []
        await session.flush()
    manifest: dict[str, dict[str, object]] = {}
    # Retired first, live last: a live row always wins its id. (Upload refuses to
    # reuse an id that is still retiring, so this is belt-and-braces.)
    for row in sorted(rows, key=lambda r: r.deleted_at is None):
        if not key_belongs_to_tenant(row.object_key, tenant_id):  # DB CHECK backstop
            logger.error("tenant sound %s: key %r outside tenant prefix", row.id, row.object_key)
            continue
        manifest[row.sound_id] = _entry(row)
    await storage.put_object(
        tenant_manifest_key(str(tenant_id)),
        json.dumps(manifest, sort_keys=True).encode("utf-8"),
        "application/json",
    )
    clear_sound_library_cache()
    return manifest
