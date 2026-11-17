"""Music-track catalogue route tests (WS-1).

``GET /api/music/tracks`` exposes the ``music``-category slice of the sound
catalog so the frontend can drop its hardcoded bed list. These tests exercise
the route with a seeded tenant (the same ``request.state`` seeding pattern the
other route tests use) and a stubbed storage backend so the loader returns the
built-in baseline without a live MinIO.
"""

from typing import Any, cast
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.routes.music import router as music_router
from src.services import tts_storage
from src.services.stages import sound_catalog

pytestmark = pytest.mark.asyncio


class _FakeSoundStorage:
    """Storage stub whose manifest reads always miss, so ``load_sound_library``
    degrades to the built-in baseline (no MinIO in unit tests)."""

    async def get_object_bytes(self, key: str) -> bytes | None:
        return None


@pytest.fixture(autouse=True)
def _stub_sound_storage():
    """Inject the manifest-miss storage stub + clear the per-tenant cache so the
    endpoint resolves to the built-in catalog deterministically."""
    sound_catalog.clear_sound_library_cache()
    tts_storage.set_storage_for_tests(cast(Any, _FakeSoundStorage()))
    yield
    tts_storage.set_storage_for_tests(None)
    sound_catalog.clear_sound_library_cache()


def _seeded_music_app() -> FastAPI:
    """Minimal app mounting only the music router behind a tenant-seeding
    middleware (mirrors conftest's ``_SeedAuthState``)."""
    tenant_id = str(uuid4())
    subject_id = str(uuid4())

    class _SeedAuthState(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):  # type: ignore[no-untyped-def]
            request.state.tenant_id = tenant_id
            request.state.subject_id = subject_id
            return await call_next(request)

    app = FastAPI()
    app.include_router(music_router)
    app.add_middleware(_SeedAuthState)
    return app


async def test_list_music_tracks_returns_builtin_beds() -> None:
    app = _seeded_music_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        r = await c.get("/api/music/tracks")

    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"tracks"}
    tracks = body["tracks"]
    assert isinstance(tracks, list)

    ids = {t["id"] for t in tracks}
    # The three built-in music beds must always be present.
    assert {"corporate_uplift", "ambient_tech", "jazz_lounge"} <= ids

    for track in tracks:
        assert set(track) == {"id", "label", "category"}
        assert track["category"] == "music"
        assert track["label"]


async def test_list_music_tracks_requires_tenant_claim() -> None:
    """With no seeded tenant on request.state the tenant dep 401s."""
    app = FastAPI()
    app.include_router(music_router)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        r = await c.get("/api/music/tracks")
    assert r.status_code == 401
