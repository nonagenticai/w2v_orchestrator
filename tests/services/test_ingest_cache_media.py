"""Durable media helpers for the source-ingest cache (tts_storage).

A cached image is stored under the durable ``media-cache/`` prefix and the cache
row keeps its OBJECT KEY, never a presigned URL — the consume path re-mints a
fresh URL so cross-run reuse survives presign expiry. No DB; uses the in-memory
storage fake via ``set_storage_for_tests``.
"""

from __future__ import annotations

import pytest

from src.services.tts_storage import (
    presign_ingest_cache_image,
    set_storage_for_tests,
    store_ingest_cache_image,
)

pytestmark = pytest.mark.asyncio


class _FakeStorage:
    """In-memory backend exposing the generic keyed-object surface the ingest-cache
    media helpers use (put_object/object_url/object_exists), plus the ``_StorageLike``
    clip-storage stubs so it satisfies ``set_storage_for_tests``' type."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    async def initialize(self) -> None:
        pass

    async def store_stage_artifact(
        self, job_id: str, stage_name: str, data: bytes, filename: str
    ) -> str:
        return f"{job_id}/{stage_name}/{filename}"

    async def get_presigned_url(
        self, path: str, expires: int | None = None, bucket: str | None = None
    ) -> str | None:
        return None

    async def put_object(
        self, key: str, data: bytes, content_type: str = "application/octet-stream"
    ) -> str:
        self.objects[key] = data
        return key

    async def object_url(self, key: str) -> str:
        # Real backend mints a URL WITHOUT verifying existence.
        return f"minio://bucket/{key}"

    async def object_exists(self, url: str) -> bool:
        key = url.removeprefix("minio://bucket/")
        return key in self.objects


@pytest.fixture
def fake_storage():
    fake = _FakeStorage()
    set_storage_for_tests(fake)
    yield fake
    set_storage_for_tests(None)


async def test_store_returns_durable_key_under_media_cache(fake_storage) -> None:
    tenant = "11111111-1111-1111-1111-111111111111"
    fp = "abcdef0123456789" * 4
    key = await store_ingest_cache_image(tenant, fp, 0, b"\x89PNGdata", "png")

    assert key is not None
    # Durable, tenant-scoped, sharded, under the GC'd media-cache root — NOT a run prefix.
    assert key == f"media-cache/tenants/{tenant}/ingest/{fp[:2]}/{fp}/0.png"
    assert "artifacts/" not in key
    assert fake_storage.objects[key] == b"\x89PNGdata"


async def test_presign_remints_from_key(fake_storage) -> None:
    tenant = "22222222-2222-2222-2222-222222222222"
    fp = "0011223344556677" * 4
    key = await store_ingest_cache_image(tenant, fp, 2, b"jpgbytes", "jpg")
    assert key is not None

    url = await presign_ingest_cache_image(key)
    assert url is not None
    assert url == f"minio://bucket/{key}"  # freshly minted from the stored key
    assert not url.startswith("http")  # not a stored/expired presigned URL — re-minted


async def test_degrades_when_backend_lacks_keyed_put(fake_storage) -> None:
    """A partial backend (no put_object) yields None rather than raising — the
    caller falls back to per-run publish."""

    class _Partial:
        """A backend that satisfies the clip-storage surface but LACKS the generic
        put_object — the ingest-cache helper must degrade to None, not raise."""

        async def initialize(self) -> None: ...

        async def store_stage_artifact(
            self, job_id: str, stage_name: str, data: bytes, filename: str
        ) -> str:
            return f"{job_id}/{filename}"

        async def get_presigned_url(
            self, path: str, expires: int | None = None, bucket: str | None = None
        ) -> str | None:
            return None

    set_storage_for_tests(_Partial())
    key = await store_ingest_cache_image("t", "f" * 64, 0, b"x", "png")
    assert key is None
    set_storage_for_tests(None)
