"""Shared sound-library sync core — upload a sound dir + (re)write its manifest.

Moved out of ``scripts/sync_sound_library.py`` so the app can call it: the
orchestrator seeds the bundled baseline at startup when the shared manifest is
missing (``src.lifecycle.sound_library_seed``), and the CLI script is now a thin
wrapper over :func:`sync`. Uploads only — never deletes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Protocol

from src.services.stages.sound_catalog import (
    BUILTIN_CATALOG,
    MANIFEST_NAME,
    SHARED_PREFIX,
    SoundCategory,
    is_tenant_sound_id,
)
from src.services.storage_service import StorageService

logger = logging.getLogger(__name__)

#: Bundled royalty-free library — ``COPY assets/ ./assets/`` in the Dockerfile.
DEFAULT_SOUND_SRC = Path(__file__).resolve().parents[3] / "assets" / "sound_effects"

#: Object key of the shared-library manifest (the seeded-or-not marker).
SHARED_MANIFEST_KEY = f"{SHARED_PREFIX}/{MANIFEST_NAME}"


class _UploadStorage(Protocol):
    """The slice of ``StorageService`` the sync needs — write one object by key."""

    async def put_object(self, key: str, data: bytes, content_type: str = ...) -> str: ...


_DEFAULT_SRC = Path(__file__).resolve().parent.parent / "assets" / "sound_effects"

# Source sub-directory name -> manifest category (the Literal the catalog uses).
_DIR_TO_CATEGORY: dict[str, SoundCategory] = {
    "transitions": "transition",
    "accents": "accent",
    "music": "music",
    "ambient": "ambient",
}

_CONTENT_TYPES = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".flac": "audio/flac",
}


def _probe_duration(path: Path) -> float:
    """Best-effort media duration in seconds via ffprobe (0.0 if unavailable)."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0.0
    try:
        out = subprocess.run(  # noqa: S603 -- fixed ffprobe args over a local file path
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return round(float(out.stdout.strip()), 3)
    except OSError, ValueError, subprocess.SubprocessError:
        return 0.0


def _humanise(stem: str) -> str:
    return stem.replace("_", " ").replace("-", " ").strip().capitalize()


def _describe(sfx_id: str, descriptions: dict[str, str]) -> str:
    """Manifest description, best first: explicit ``descriptions.json`` override,
    then the built-in catalog's curated text, then a humanised filename."""
    if sfx_id in descriptions:
        return descriptions[sfx_id]
    builtin = BUILTIN_CATALOG.get(sfx_id)
    return builtin.description if builtin else _humanise(sfx_id)


def _scope_prefix(scope: str, tenant: str | None) -> str:
    if scope == "shared":
        return SHARED_PREFIX
    if scope == "tenant":
        raise SystemExit(
            "--scope tenant is retired: tenant libraries are uploaded through "
            "POST /api/tenant/music (the tenant manifest is derived from the database)"
        )
    raise SystemExit(f"unknown --scope {scope!r}")


def _load_descriptions(src: Path) -> dict[str, str]:
    desc_file = src / "descriptions.json"
    if not desc_file.is_file():
        return {}
    try:
        return {str(k): str(v) for k, v in json.loads(desc_file.read_text()).items()}
    except (ValueError, TypeError) as exc:
        logger.warning("ignoring malformed %s: %s", desc_file, exc)
        return {}


async def _upload_library(
    store: _UploadStorage, src: Path, prefix: str, descriptions: dict[str, str]
) -> int:
    """Upload every audio file under ``src`` + (re)write the manifest. Returns the
    process exit code (0 ok, 1 nothing found)."""
    manifest: dict[str, dict[str, object]] = {}
    uploaded = 0

    for category_dir, category in _DIR_TO_CATEGORY.items():
        cat_path = src / category_dir
        if not cat_path.is_dir():
            continue
        for asset in sorted(cat_path.iterdir()):
            if not asset.is_file() or asset.suffix.lower() not in _CONTENT_TYPES:
                continue
            sfx_id = asset.stem
            if is_tenant_sound_id(sfx_id):
                # Reserved for tenant uploads; the loader drops it from the shared layer.
                logger.error("skipping %s: the 'tenant_' id prefix is reserved", asset)
                continue
            key = f"{prefix}/{category_dir}/{asset.name}"
            data = await asyncio.to_thread(asset.read_bytes)
            await store.put_object(key, data, content_type=_CONTENT_TYPES[asset.suffix.lower()])
            manifest[sfx_id] = {
                "key": key,
                "description": _describe(sfx_id, descriptions),
                "category": category,
                "duration_s": await asyncio.to_thread(_probe_duration, asset),
            }
            uploaded += 1
            logger.info("uploaded %s → %s (%d bytes)", sfx_id, key, len(data))

    if not manifest:
        logger.error("no audio files found under %s — nothing to sync", src)
        return 1

    manifest_key = f"{prefix}/{MANIFEST_NAME}"
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    await store.put_object(manifest_key, manifest_bytes, content_type="application/json")
    logger.info("wrote manifest %s (%d sounds)", manifest_key, len(manifest))
    logger.info("sync complete: %d uploaded into %s", uploaded, prefix)
    return 0


async def sync(
    src: Path,
    scope: str,
    tenant: str | None,
    storage: _UploadStorage | None = None,
) -> int:
    if not src.is_dir():
        logger.error("source dir not found: %s", src)
        return 2

    descriptions = _load_descriptions(src)
    prefix = _scope_prefix(scope, tenant)

    # Own the storage lifecycle only when we created it; an injected storage
    # (tests) is the caller's to close.
    owns_storage = storage is None
    svc: StorageService | None = None
    if storage is not None:
        store: _UploadStorage = storage
    else:
        svc = StorageService()
        await svc.initialize()
        store = svc

    try:
        return await _upload_library(store, src, prefix, descriptions)
    finally:
        if owns_storage and svc is not None:
            await svc.close()


class _SeedStorage(_UploadStorage, Protocol):
    """``_UploadStorage`` plus the definitive absent-probe the seed gates on."""

    async def object_size(self, key: str) -> int | None: ...


async def seed_shared_library_if_missing(
    storage: _SeedStorage, src: Path = DEFAULT_SOUND_SRC
) -> bool:
    """Upload the bundled shared library iff ``shared/manifest.json`` is ABSENT.

    Returns ``True`` when it seeded, ``False`` when the manifest was already
    there. Gated on :meth:`StorageService.object_size`, whose only ``None`` is a
    genuine ``NoSuchKey`` — a transport error RAISES, so an unreachable MinIO
    never reads as "absent" and never overwrites a richer, hand-synced manifest
    with the bundled baseline. Idempotent; uploads only, never deletes.
    """
    if await storage.object_size(SHARED_MANIFEST_KEY) is not None:
        return False
    logger.warning(
        "sound_library: %s is missing — seeding the bundled library from %s",
        SHARED_MANIFEST_KEY,
        src,
    )
    rc = await sync(src, "shared", None, storage=storage)
    if rc != 0:
        raise RuntimeError(f"sound_library seed from {src} failed (rc={rc})")
    return True
