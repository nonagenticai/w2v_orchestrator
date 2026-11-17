"""P3b — cross-run, tenant-scoped content-addressed media cache.

Exercises ``tts_storage._store_media_asset`` + ``find_existing_media`` directly
against an in-memory fake storage, covering the three contracts:

* flag ON + tenant bound → an item stored in one run is reused from a DIFFERENT
  run (cross-run), and only stored once;
* flag ON but a DIFFERENT tenant → miss (isolation; a content hash is not an
  auth boundary);
* flag OFF (or no tenant) → exact P2 per-run behavior (reuse only within a run).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from uuid import UUID

import pytest
from w2v_common.auth.identity_context import _tenant_id_var, set_identity_context
from w2v_common.tenant import InvalidTenantIdError

from src.config import clear_settings_cache
from src.services import tts_storage
from src.services.tts_storage import (
    find_existing_media,
    set_storage_for_tests,
    store_generated_image,
)

#: w2v-common 0.67.0 parses the ambient tenant as a `uuid.UUID`; the former
#: "tenant-1"/"tenant-2" slugs are now REJECTED at `set_identity_context`.
_T1 = "11111111-1111-1111-1111-111111111111"
_T2 = "22222222-2222-2222-2222-222222222222"
#: The nil uuid -- the pre-0.67 `auth.default_tenant_id` fallback.
_NIL = "00000000-0000-0000-0000-000000000000"

_MID = "deadbeefcafe0000deadbeefcafe0000deadbeef"  # a content-addressed media id
_EXT = "png"


class _MemStorage:
    """Minimal in-memory stand-in for ``StorageService``.

    Implements both surfaces the cache touches: the generic keyed API
    (``put_object`` / ``object_url``) used by the cross-run path, and the
    per-run API (``store_stage_artifact`` / ``stage_artifact_url``) used by the
    P2 fallback. ``object_exists`` decodes the shared ``mem://own/<key>`` URL
    shape both produce. ``writes`` counts every object write so a test can prove
    a second run did NOT re-store.
    """

    _URL_PREFIX = "mem://own/"

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.writes = 0

    async def initialize(self) -> None: ...

    # --- generic keyed API (cross-run path) ---
    async def put_object(self, key: str, data: bytes, content_type: str = "x") -> str:
        self.writes += 1
        self.objects[key] = data
        return key

    async def object_url(self, key: str) -> str:
        return f"{self._URL_PREFIX}{key}"

    # --- per-run API (P2 fallback) ---
    async def store_stage_artifact(
        self, job_id: str, stage_name: str, data: bytes, filename: str
    ) -> str:
        key = f"artifacts/{job_id}/{stage_name}/{filename}"
        self.writes += 1
        self.objects[key] = data
        return key

    async def stage_artifact_url(self, job_id: str, stage_name: str, filename: str) -> str:
        return f"{self._URL_PREFIX}artifacts/{job_id}/{stage_name}/{filename}"

    async def get_presigned_url(
        self, path: str, expires: int | None = None, bucket: str | None = None
    ) -> str:
        return f"{self._URL_PREFIX}{path}"

    # --- shared existence probe ---
    async def object_exists(self, url: str) -> bool | None:
        if not url.startswith(self._URL_PREFIX):
            return None
        return url[len(self._URL_PREFIX) :] in self.objects


@pytest.fixture
def storage() -> Iterator[_MemStorage]:
    fake = _MemStorage()
    set_storage_for_tests(fake)
    try:
        yield fake
    finally:
        set_storage_for_tests(None)


def _set_cross_run(enabled: bool) -> None:
    if enabled:
        os.environ["STORAGE_MEDIA_CACHE_CROSS_RUN"] = "true"
    else:
        os.environ.pop("STORAGE_MEDIA_CACHE_CROSS_RUN", None)
    clear_settings_cache()


@pytest.fixture(autouse=True)
def _reset_env_and_identity() -> Iterator[None]:
    try:
        yield
    finally:
        os.environ.pop("STORAGE_MEDIA_CACHE_CROSS_RUN", None)
        clear_settings_cache()
        set_identity_context(tenant_id=None, license_id=None, subject_id=None)


@pytest.mark.asyncio
async def test_tenant_isolation(storage: _MemStorage) -> None:
    _set_cross_run(True)
    set_identity_context(tenant_id=_T1, license_id=None, subject_id=None)
    await store_generated_image("run-A", "ai_image_gen", _MID, b"img", _EXT)

    # A different tenant with an identical content id must NOT see the blob.
    set_identity_context(tenant_id=_T2, license_id=None, subject_id=None)
    miss = await find_existing_media("run-A", "ai_image_gen", _MID, _EXT)
    assert miss is None

    # ...and the original tenant still hits.
    set_identity_context(tenant_id=_T1, license_id=None, subject_id=None)
    assert await find_existing_media("run-Z", "ai_image_gen", _MID, _EXT) is not None


@pytest.mark.asyncio
async def test_flag_off_is_per_run(storage: _MemStorage) -> None:
    _set_cross_run(False)
    set_identity_context(tenant_id=_T1, license_id=None, subject_id=None)

    url = await store_generated_image("run-A", "ai_image_gen", _MID, b"img", _EXT)
    assert url == f"mem://own/artifacts/run-A/ai_image_gen/{_MID}.{_EXT}"

    # A different run does NOT reuse (per-run scoping, exact P2 behavior)...
    assert await find_existing_media("run-B", "ai_image_gen", _MID, _EXT) is None
    # ...but the same run does.
    assert await find_existing_media("run-A", "ai_image_gen", _MID, _EXT) == url


@pytest.mark.asyncio
async def test_no_tenant_falls_back_to_per_run(storage: _MemStorage) -> None:
    _set_cross_run(True)  # flag on, but no tenant bound
    set_identity_context(tenant_id=None, license_id=None, subject_id=None)

    url = await store_generated_image("run-A", "ai_image_gen", _MID, b"img", _EXT)
    assert url == f"mem://own/artifacts/run-A/ai_image_gen/{_MID}.{_EXT}"
    assert await find_existing_media("run-B", "ai_image_gen", _MID, _EXT) is None


@pytest.mark.asyncio
async def test_nil_tenant_does_not_become_a_shared_cache_bucket(storage: _MemStorage) -> None:
    """The nil uuid must fall back to per-run, NOT key a shared tenant bucket.

    This is the hole a ``None``-only guard leaves open. The nil uuid was the
    pre-0.67 ``auth.default_tenant_id`` fallback and it is perfectly truthy, so
    ``if tenant:`` happily accepts it — and every unattributed caller across
    every tenant would then read and write the SAME
    ``media-cache/tenants/00000000-0000-0000-0000-000000000000/`` prefix.
    Generated media would be served across tenant boundaries, which is exactly
    what this module's docstring promises never happens ("a content hash is not
    an auth boundary").
    """
    _set_cross_run(True)
    # The nil uuid can no longer be BOUND (w2v-common 0.68.0 refuses it at
    # ``set_identity_context``), so it is planted directly into the ContextVar
    # the cache reads. That is deliberately reaching under the door: the point
    # of this test is the CACHE layer's behaviour if a nil tenant ever reached
    # it, which ``_cross_run_tenant`` still handles via ``tenant_scope_key``.
    # The door itself is pinned by
    # ``test_the_nil_tenant_is_refused_before_it_can_reach_the_cache`` below.
    token = _tenant_id_var.set(UUID(int=0))
    try:
        url = await store_generated_image("run-A", "ai_image_gen", _MID, b"img", _EXT)
        # Per-run path, NOT a media-cache/tenants/<nil>/ path.
        assert url == f"mem://own/artifacts/run-A/ai_image_gen/{_MID}.{_EXT}"
        assert tts_storage.MEDIA_CACHE_ROOT not in url
        # And a different run must not reuse it, exactly as with no tenant at all.
        assert await find_existing_media("run-B", "ai_image_gen", _MID, _EXT) is None
    finally:
        _tenant_id_var.reset(token)

    # The decisive isolation claim: a REAL tenant must not be able to read what
    # the unattributed caller just wrote.
    set_identity_context(tenant_id=_T1, license_id=None, subject_id=None)
    assert await find_existing_media("run-A", "ai_image_gen", _MID, _EXT) is None


@pytest.mark.parametrize(
    "spelling",
    [_NIL],
    ids=["hyphenated"],
)
def test_the_nil_tenant_is_refused_before_it_can_reach_the_cache(spelling: UUID | str) -> None:
    """The shared-bucket hole is now closed one layer EARLIER, at the door.

    ``_cross_run_tenant`` reads the ambient ContextVar and nothing else, and
    w2v-common 0.68.0 made ``set_identity_context`` run every value through
    ``parse_tenant_id``, which refuses the nil uuid. So no caller can bind it
    and the cache can no longer be reached with a nil tenant at all — the
    protection above became STRUCTURAL rather than a runtime check.

    Asserted here so the structure is pinned rather than assumed: if the library
    ever stops refusing, this goes red and the cache-layer guard becomes
    load-bearing again instead of quietly rotting into decoration.
    """
    with pytest.raises(InvalidTenantIdError, match=r"(?i)nil uuid"):
        set_identity_context(tenant_id=spelling, license_id=None, subject_id=None)
