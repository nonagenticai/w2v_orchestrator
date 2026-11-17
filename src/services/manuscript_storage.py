"""Manuscript upload persistence + read-back.

The audiobook wizard uploads a manuscript file (.txt/.docx/.epub) *before* the
project exists, so the file is staged under a tenant-scoped namespace and the
returned storage ``ref`` (MinIO key / local path) is carried in the project
brief (``brief.manuscript_ref``). The ``audiobook_ingest`` stage later reads it
back by that ref — by key, never by an arbitrary URL, so there is no SSRF
surface.

Mirrors ``tts_storage``'s process-wide lazy-singleton pattern (StageHandlers
and routes get a RunContext / Request, not ``app.state``). Tests inject a fake
via ``set_manuscript_storage_for_tests``.
"""

from __future__ import annotations

from typing import Protocol

from src.services._lazy_singleton import AsyncLazySingleton
from src.services.storage_service import StorageService

_STAGE = "upload"


class _ManuscriptStorageLike(Protocol):
    async def initialize(self) -> None: ...
    async def store_stage_artifact(
        self, job_id: str, stage_name: str, data: bytes, filename: str
    ) -> str: ...
    async def retrieve(self, ref: str) -> bytes | None: ...


async def _make_storage() -> _ManuscriptStorageLike:
    svc = StorageService()
    await svc.initialize()
    return svc


_storage_singleton: AsyncLazySingleton[_ManuscriptStorageLike] = AsyncLazySingleton(_make_storage)


async def _get_storage() -> _ManuscriptStorageLike:
    return await _storage_singleton.get()


def set_manuscript_storage_for_tests(storage: _ManuscriptStorageLike | None) -> None:
    """Override (or reset) the process-wide manuscript storage singleton."""
    _storage_singleton.set_for_tests(storage)


async def store_manuscript(tenant_id: str, filename: str, data: bytes) -> str:
    """Persist an uploaded manuscript and return its storage ref.

    Namespaced by tenant (not project — the project doesn't exist yet at upload
    time). The ref is opaque to the caller; pass it to ``fetch_manuscript`` to
    read the bytes back.
    """
    storage = await _get_storage()
    return await storage.store_stage_artifact(f"manuscripts/{tenant_id}", _STAGE, data, filename)


async def fetch_manuscript(ref: str) -> bytes:
    """Read an uploaded manuscript back by the ref ``store_manuscript`` returned."""
    storage = await _get_storage()
    data = await storage.retrieve(ref)
    if data is None:
        raise RuntimeError(f"manuscript not found at ref {ref!r}")
    return data
