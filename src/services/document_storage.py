"""Source-document upload persistence + read-back.

The creation wizard can attach arbitrary source documents (PDF/book/docx/txt/md)
to a project *before* it exists — a promo of "anything" is grounded in a
combination of sources. Each file is staged under a tenant-scoped namespace and
the returned opaque storage ``ref`` is carried in the project brief
(``brief.sources[*].ref`` for ``kind == "document"``). The ``source_ingest``
stage later reads it back by that ref — by key, never by an arbitrary URL, so
there is no SSRF surface.

The ref round-trips through a brief the caller writes, so on read-back it is
**untrusted input**: the underlying facade's ``retrieve(key)`` has no tenant
concept and would happily serve another tenant's upload. ``fetch_document``
therefore re-derives the key against the reading tenant's own
``artifacts/documents/{tenant_id}/`` prefix (:mod:`src.services.object_key_guard`)
— that, not the ref's opacity, is what makes the read tenant-isolated.

Mirrors ``manuscript_storage``'s process-wide lazy-singleton pattern (handlers
and routes get a RunContext / Request, not ``app.state``). Tests inject a fake
via ``set_document_storage_for_tests``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from src.config import get_settings
from src.services._lazy_singleton import AsyncLazySingleton
from src.services.object_key_guard import KeyConfinementError, confine_key
from src.services.storage_service import StorageService

_STAGE = "upload"


class _DocumentStorageLike(Protocol):
    async def initialize(self) -> None: ...
    async def store_stage_artifact(
        self, job_id: str, stage_name: str, data: bytes, filename: str
    ) -> str: ...
    async def retrieve(self, ref: str) -> bytes | None: ...


async def _make_storage() -> _DocumentStorageLike:
    svc = StorageService()
    await svc.initialize()
    return svc


_storage_singleton: AsyncLazySingleton[_DocumentStorageLike] = AsyncLazySingleton(_make_storage)


async def _get_storage() -> _DocumentStorageLike:
    return await _storage_singleton.get()


def set_document_storage_for_tests(storage: _DocumentStorageLike | None) -> None:
    """Override (or reset) the process-wide document storage singleton."""
    _storage_singleton.set_for_tests(storage)


async def store_document(tenant_id: str, filename: str, data: bytes) -> str:
    """Persist an uploaded source document and return its opaque storage ref.

    Namespaced by tenant (not project — the project doesn't exist yet at upload
    time). Pass the ref to ``fetch_document`` to read the bytes back.

    The ref is always the **object key**, never the backend's own return value.
    Under ``STORAGE_BACKEND=local`` the facade's ``put_bytes`` returns
    ``str(local_path / key)`` — an ABSOLUTE PATH — and a ref starting with ``/``
    is refused by ``normalize_key``, so on that backend every legitimate
    ``fetch_document`` raised "document not found". Returning the key makes the
    ref backend-independent (the facade's local ``get_bytes`` resolves a relative
    ref against ``local_path`` itself), and it is the only form the tenant-prefix
    confinement can be stated against at all.
    """
    storage = await _get_storage()
    written = await storage.store_stage_artifact(_job_id(tenant_id), _STAGE, data, filename)
    return _as_object_key(written)


def _as_object_key(ref: str) -> str:
    """The object key behind a storage ref, whichever backend produced it.

    MinIO refs are already keys and pass through. A local-backend ref is an
    absolute path under the configured ``local_path``; strip that root back off.
    A path that is NOT under the root is left alone so it fails the confinement
    check loudly rather than being coerced into looking like a key.
    """
    if not ref.startswith("/"):
        return ref
    root = Path(get_settings().storage.local_path)
    candidate = Path(ref)
    if not candidate.is_relative_to(root):
        return ref
    return candidate.relative_to(root).as_posix()


def _job_id(tenant_id: str) -> str:
    """The pseudo-``job_id`` an upload is filed under — uploads predate the project."""
    return f"documents/{tenant_id}"


def document_prefix(tenant_id: str) -> str:
    """The object-key prefix every one of ``tenant_id``'s uploads lives under.

    Mirrors the ``artifacts/{job_id}/…`` **object key** ``store_stage_artifact``
    writes at, for the pseudo-job ``_job_id`` above — the two must move together,
    hence the shared helper rather than two hand-written f-strings.

    Note this is the key, which is NOT what ``store_stage_artifact`` *returns* on
    every backend: MinIO returns the key, the local backend returns the absolute
    file path it wrote. ``_as_object_key`` reconciles the two so exactly one form
    — this one — is ever confined, stored in a brief, or read back.
    """
    return f"artifacts/{_job_id(tenant_id)}"


async def fetch_document(ref: str, tenant_id: str) -> bytes:
    """Read an uploaded document back by the ref ``store_document`` returned.

    ``ref`` comes off a caller-written brief, so it is re-derived against
    ``tenant_id``'s own upload prefix before it reaches storage: the ref may only
    name a suffix under that prefix, never a sibling tenant's key. No tenant, no
    read — an unbound run cannot be confined, and reading unconfined is the bug.
    """
    if not (tenant_id or "").strip():
        raise RuntimeError("refusing to read an uploaded document without a tenant")
    try:
        # ``_as_object_key`` also rescues refs written into a brief before
        # ``store_document`` returned keys: on the local backend those are
        # absolute paths, which the confinement check refuses outright.
        key = confine_key(document_prefix(tenant_id), _as_object_key(ref))
    except KeyConfinementError as exc:
        # Same shape as the not-found path below: the caller (source_ingest) turns
        # either into a soft-fail note, and a probe must not learn which it hit.
        raise RuntimeError(f"document not found at ref {ref!r}") from exc
    storage = await _get_storage()
    data = await storage.retrieve(key)
    if data is None:
        raise RuntimeError(f"document not found at ref {ref!r}")
    return data
