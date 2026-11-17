"""Uploaded-document read-back is confined to the reading tenant's prefix.

The brief carries ``sources[*].ref`` verbatim from whoever wrote it, and the
shared storage facade's ``retrieve(key)`` has no tenant concept — so
``fetch_document`` must re-derive the ref against the caller's own
``artifacts/documents/{tenant}/`` prefix. The happy path here is the cheap half;
the escapes below are the point. No DB; in-memory storage fake.

Every test runs against BOTH storage backends, because the two do not return the
same thing from a write and the old fake hid that: MinIO's ``put_bytes`` returns
the object key, the local backend's returns ``str(local_path / key)`` — an
ABSOLUTE PATH, which ``normalize_key`` refuses outright. Under
``STORAGE_BACKEND=local``, therefore, every legitimate document read raised
"document not found"; the fake returned the minio-shaped key on both, so the
break was invisible. The fake below reproduces each backend's real return shape,
which is the only reason these tests can see the bug at all.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from unittest.mock import patch

import pytest

from src.services.document_storage import (
    document_prefix,
    fetch_document,
    set_document_storage_for_tests,
    store_document,
)

pytestmark = pytest.mark.asyncio

_TENANT = "11111111-1111-1111-1111-111111111111"
_OTHER_TENANT = "22222222-2222-2222-2222-222222222222"

# Stands in for ``settings.storage.local_path``; never touched on disk (the fake
# keeps bytes in a dict), it only has to be the root the local backend would
# prepend to a key.
_LOCAL_ROOT = "/var/lib/w2v/artifacts-root"


class _FakeStorage:
    """In-memory ``_DocumentStorageLike`` reproducing the real
    ``artifacts/{job_id}/{stage}/{filename}`` key shape — the confinement rule is
    only meaningful against the key shape production actually writes — AND each
    backend's real *return* shape, which is a different thing entirely.

    Objects are always stored under the key. ``retrieve`` accepts the key, and on
    the local backend also the absolute path, mirroring the facade's local
    ``get_bytes`` (which resolves a relative ref against ``local_path`` and takes
    an absolute one as-is).
    """

    def __init__(self, backend: str) -> None:
        self.backend = backend
        self.objects: dict[str, bytes] = {}
        self.reads: list[str] = []

    async def initialize(self) -> None:
        pass

    async def store_stage_artifact(
        self, job_id: str, stage_name: str, data: bytes, filename: str
    ) -> str:
        key = f"artifacts/{job_id}/{stage_name}/{filename}"
        self.objects[key] = data
        # THE DIVERGENCE: minio hands back the key, local hands back a disk path.
        return key if self.backend == "minio" else f"{_LOCAL_ROOT}/{key}"

    async def retrieve(self, ref: str) -> bytes | None:
        self.reads.append(ref)
        if self.backend == "local" and ref.startswith(f"{_LOCAL_ROOT}/"):
            ref = str(PurePosixPath(ref).relative_to(_LOCAL_ROOT))
        return self.objects.get(ref)


@pytest.fixture(params=["minio", "local"])
def fake_storage(request: pytest.FixtureRequest):
    fake = _FakeStorage(request.param)
    set_document_storage_for_tests(fake)
    with patch("src.services.document_storage.get_settings") as mock_settings:
        mock_settings.return_value.storage.backend = request.param
        mock_settings.return_value.storage.local_path = _LOCAL_ROOT
        yield fake
    set_document_storage_for_tests(None)


async def test_own_ref_round_trips(fake_storage) -> None:
    ref = await store_document(_TENANT, "book.pdf", b"%PDF own")

    assert ref.startswith(f"{document_prefix(_TENANT)}/")
    assert await fetch_document(ref, _TENANT) == b"%PDF own"


async def test_foreign_tenant_ref_is_rejected(fake_storage) -> None:
    """The exploit: own a run, name the *other* tenant's key in the brief."""
    victim_ref = await store_document(_OTHER_TENANT, "secret.pdf", b"%PDF secret")
    assert victim_ref in fake_storage.objects  # the object really is readable by key

    with pytest.raises(RuntimeError, match="document not found"):
        await fetch_document(victim_ref, _TENANT)
    assert fake_storage.reads == []  # never reached storage at all


@pytest.mark.parametrize(
    "escaping_ref",
    [f"artifacts/documents/{_TENANT}/../{_OTHER_TENANT}/upload/secret.pdf"],
)
async def test_escaping_refs_are_rejected(fake_storage, escaping_ref: str) -> None:
    with pytest.raises(RuntimeError, match="document not found"):
        await fetch_document(escaping_ref, _TENANT)
    assert fake_storage.reads == []


async def test_untenanted_read_is_refused(fake_storage) -> None:
    """No tenant means no prefix to confine against — refuse, never read raw."""
    ref = await store_document(_TENANT, "book.pdf", b"%PDF own")

    with pytest.raises(RuntimeError, match="without a tenant"):
        await fetch_document(ref, "")
    assert fake_storage.reads == []


# ---------------------------------------------------------------------------
# Backend-shape reconciliation — the bug the old fake hid
# ---------------------------------------------------------------------------


async def test_store_document_returns_an_object_key_on_every_backend(fake_storage) -> None:
    """A ref is always a KEY, never the local backend's absolute path.

    This is the whole fix for ``STORAGE_BACKEND=local``: the ref goes into a
    brief and comes back through ``confine_key``, which refuses anything starting
    with ``/`` — so a path-shaped ref meant "document not found" for every
    legitimate read on that backend.

    Mutation killed: dropping ``_as_object_key`` from ``store_document`` (the
    local parametrisation returns ``/var/lib/w2v/artifacts-root/artifacts/…`` and
    fails on the very first assertion).
    """
    ref = await store_document(_TENANT, "book.pdf", b"%PDF own")

    assert not ref.startswith("/")
    assert ref == f"{document_prefix(_TENANT)}/upload/book.pdf"
    # And it is the key the object was actually written under, on both backends.
    assert fake_storage.objects[ref] == b"%PDF own"


async def test_round_trip_reads_the_bytes_on_every_backend(fake_storage) -> None:
    """The end-to-end claim: store → brief → read back, on local as well as minio.

    Mutation killed: the same one as above, seen from the read side — before the
    fix this raised ``RuntimeError("document not found")`` under ``local`` while
    passing under ``minio``, which is exactly the shape of a bug a single-backend
    fake cannot show you.
    """
    ref = await store_document(_TENANT, "book.pdf", b"%PDF own")

    assert await fetch_document(ref, _TENANT) == b"%PDF own"


async def test_legacy_absolute_local_ref_is_still_readable(fake_storage) -> None:
    """Briefs written BEFORE this fix carry an absolute local path; still read.

    ``fetch_document`` reconciles the ref to a key before confining, so an already
    persisted brief does not become permanently unreadable the day the return
    shape changed.

    Mutation killed: dropping ``_as_object_key`` from ``fetch_document`` — the
    absolute ref is refused by ``normalize_key`` and this raises.
    """
    await store_document(_TENANT, "book.pdf", b"%PDF own")
    legacy_ref = f"{_LOCAL_ROOT}/{document_prefix(_TENANT)}/upload/book.pdf"

    assert await fetch_document(legacy_ref, _TENANT) == b"%PDF own"


@pytest.mark.parametrize(
    "absolute_ref",
    ["/etc/passwd"],
)
async def test_absolute_refs_outside_the_tenant_prefix_are_refused(
    fake_storage, absolute_ref: str
) -> None:
    """Reconciling a path to a key must never become a way to READ any path.

    Mutation killed: making ``_as_object_key`` strip a leading ``/`` blindly (or
    return ``candidate.name``) instead of requiring containment in the configured
    root and then re-confining to the tenant prefix — each of these would start
    resolving.
    """
    with pytest.raises(RuntimeError, match="document not found"):
        await fetch_document(absolute_ref, _TENANT)
    assert fake_storage.reads == []
