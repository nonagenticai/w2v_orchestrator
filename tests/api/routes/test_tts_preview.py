"""Tests for the /api/tts/preview voice-preview endpoint.

The endpoint reuses ``TTSClient.synthesize`` (the same client every narrate
stage uses) and returns raw audio bytes for in-browser playback. We patch the
client method so the test is DB- and network-free, and build a minimal FastAPI
app with a tenant-id-seeding middleware (mirrors ``test_critique_proxy``) rather
than the heavy conftest ``app`` fixture which needs live postgres.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from starlette.middleware.base import BaseHTTPMiddleware

import src.api.routes.tts as tts_routes
from src.api.routes.tts import router as tts_router
from src.clients.schemas.tts import TTSSynthesis
from src.clients.tts_client import TTSSynthesisError


def _synth(audio_bytes: bytes, fmt: str = "wav") -> TTSSynthesis:
    """Build a minimal TTSSynthesis for preview fakes (route reads bytes + format)."""
    return TTSSynthesis(
        audio_bytes=audio_bytes,
        sample_rate=0,
        duration_seconds=0.0,
        provider="",
        voice="",
        format=fmt,
    )


@pytest.fixture
def app_with_seeded_tenant() -> FastAPI:
    tenant_id = str(uuid4())

    class _SeedAuthState(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            request.state.tenant_id = tenant_id
            request.state.subject_id = str(uuid4())
            return await call_next(request)

    app = FastAPI()
    app.add_middleware(_SeedAuthState)
    app.include_router(tts_router)
    return app


@pytest.fixture
async def client(app_with_seeded_tenant):
    transport = ASGITransport(app=app_with_seeded_tenant)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


def _patch_tts(monkeypatch, *, synthesize):
    """Patch TTSClient so construction is cheap and synthesize/close are stubbed."""
    captured: dict[str, Any] = {"calls": 0}

    class _FakeTTSClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def synthesize(self, **kwargs: Any) -> TTSSynthesis:
            captured["calls"] += 1
            captured["kwargs"] = kwargs
            return await synthesize(**kwargs)

        async def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(tts_routes, "TTSClient", _FakeTTSClient)
    return captured


class _FakeRedis:
    """Minimal stand-in for the orchestrator RedisClient used by the route.

    Mirrors the JSON-roundtrip contract: ``set`` stores the (already
    JSON-serializable) value as-is and ``get`` returns it, matching what the
    real client yields after serialize→deserialize."""

    def __init__(self) -> None:
        self.store: dict[str, Any] = {}
        self.ttls: dict[str, int | None] = {}

    async def get(self, key: str) -> Any:
        return self.store.get(key)

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        self.store[key] = value
        self.ttls[key] = ttl
        return True


@pytest.fixture
def app_with_redis(app_with_seeded_tenant) -> FastAPI:
    app_with_seeded_tenant.state.redis = _FakeRedis()
    return app_with_seeded_tenant


@pytest.fixture
async def redis_client(app_with_redis):
    transport = ASGITransport(app=app_with_redis)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c, app_with_redis.state.redis


async def test_preview_returns_audio_bytes(client, monkeypatch):
    async def _ok(**kwargs: Any) -> TTSSynthesis:
        return _synth(b"RIFFfake-wav-bytes")

    captured = _patch_tts(monkeypatch, synthesize=_ok)

    resp = await client.post(
        "/api/tts/preview",
        json={"provider": "kokoro", "voice": "am_adam", "speed": 1.25},
    )

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    assert resp.content == b"RIFFfake-wav-bytes"
    # selected voice/provider/speed are forwarded to the TTS client
    assert captured["kwargs"]["voice"] == "am_adam"
    assert captured["kwargs"]["provider"] == "kokoro"
    assert captured["kwargs"]["speed"] == 1.25
    # an English voice gets the English sample
    assert "preview" in captured["kwargs"]["text"].lower()
    assert captured["closed"] is True


async def test_preview_uses_language_sample_for_non_english_voice(client, monkeypatch):
    async def _ok(**kwargs: Any) -> TTSSynthesis:
        return _synth(b"x")

    captured = _patch_tts(monkeypatch, synthesize=_ok)

    resp = await client.post("/api/tts/preview", json={"voice": "jf_alpha"})
    assert resp.status_code == 200
    # Japanese voice → Japanese sample (not the English default)
    assert captured["kwargs"]["text"] == "これは選択したナレーション音声のプレビューです。"


async def test_preview_synthesis_error_maps_to_502(client, monkeypatch):
    async def _boom(**kwargs: Any) -> TTSSynthesis:
        raise TTSSynthesisError("tts_backend returned no audio_base64")

    captured = _patch_tts(monkeypatch, synthesize=_boom)

    resp = await client.post("/api/tts/preview", json={"voice": "am_adam"})
    assert resp.status_code == 502
    assert "preview failed" in resp.json()["detail"]
    # client is still closed on the error path
    assert captured["closed"] is True


async def test_preview_busy_lane_maps_to_503_with_retry_after(client, monkeypatch):
    """#1106: a GPU lane that stayed busy past the deadline is a retryable 503."""
    from src.clients.gpu_busy import GpuBusyError

    async def _busy(**kwargs: Any) -> TTSSynthesis:
        raise GpuBusyError("tts stayed busy", service="tts", status_code=503)

    captured = _patch_tts(monkeypatch, synthesize=_busy)

    resp = await client.post("/api/tts/preview", json={"voice": "am_adam"})
    assert resp.status_code == 503
    assert resp.headers.get("Retry-After") == "30"
    assert captured["closed"] is True


async def test_preview_byo_refusal_maps_to_402_with_failure_detail(client, monkeypatch):
    """#1106: a byo_required tenant gets the structured refusal, not a 500."""
    from src.services.providers.errors import SlotNotConfiguredError

    async def _refused(**kwargs: Any) -> TTSSynthesis:
        raise SlotNotConfiguredError("tts")

    _patch_tts(monkeypatch, synthesize=_refused)

    resp = await client.post("/api/tts/preview", json={"voice": "am_adam"})
    assert resp.status_code == 402
    assert resp.json()["detail"] == {
        "code": "byo_provider_required",
        "modality": "tts",
        "slots": "tts",
        "message": "Configure your own TTS provider to continue",
    }


@pytest.mark.parametrize("speed", [0.1])
async def test_preview_rejects_out_of_range_speed(client, monkeypatch, speed):
    _patch_tts(monkeypatch, synthesize=lambda **_: None)
    resp = await client.post("/api/tts/preview", json={"speed": speed})
    assert resp.status_code == 422


async def test_preview_uses_short_retry_budget(client, monkeypatch):
    async def _ok(**kwargs: Any) -> TTSSynthesis:
        return _synth(b"x")

    captured = _patch_tts(monkeypatch, synthesize=_ok)

    resp = await client.post("/api/tts/preview", json={"voice": "am_adam"})
    assert resp.status_code == 200
    # interactive preview path caps retries (fast-fail), unlike narration
    assert captured["kwargs"]["attempts"] == 2
    # #1111: a BUSY lane is not bounded by those 2 tries — it gets its own wait
    # budget, kept under Cloudflare's ~100s edge timeout.
    assert captured["kwargs"]["busy_deadline_seconds"] == 90.0


async def test_preview_busy_passes_the_backends_retry_after_on(client, monkeypatch):
    """#1111: the exhausted busy budget answers 503 with the backend's OWN hint."""
    from src.clients.gpu_busy import GpuBusyError

    async def _busy(**kwargs: Any) -> TTSSynthesis:
        raise GpuBusyError("tts stayed busy", service="tts", status_code=503, retry_after=44.2)

    _patch_tts(monkeypatch, synthesize=_busy)

    resp = await client.post("/api/tts/preview", json={"voice": "am_adam"})
    assert resp.status_code == 503
    assert resp.headers.get("Retry-After") == "45"
    assert "busy" in resp.json()["detail"]


async def test_preview_serves_second_request_from_cache(redis_client, monkeypatch):
    client, fake_redis = redis_client

    async def _ok(**kwargs: Any) -> TTSSynthesis:
        return _synth(b"RIFFfake-wav-bytes")

    captured = _patch_tts(monkeypatch, synthesize=_ok)

    body = {"provider": "kokoro", "voice": "am_adam", "speed": 1.25}
    first = await client.post("/api/tts/preview", json=body)
    assert first.status_code == 200
    assert first.content == b"RIFFfake-wav-bytes"
    assert captured["calls"] == 1
    # the clip is now cached under a deterministic (provider, voice, speed) key
    assert len(fake_redis.store) == 1

    second = await client.post("/api/tts/preview", json=body)
    assert second.status_code == 200
    assert second.content == b"RIFFfake-wav-bytes"
    assert second.headers["content-type"] == "audio/wav"
    # served from cache — synthesize was NOT called a second time
    assert captured["calls"] == 1


async def test_preview_distinct_params_get_distinct_cache_keys(redis_client, monkeypatch):
    client, fake_redis = redis_client

    async def _ok(**kwargs: Any) -> TTSSynthesis:
        return _synth(b"x")

    captured = _patch_tts(monkeypatch, synthesize=_ok)

    await client.post("/api/tts/preview", json={"voice": "am_adam", "speed": 1.0})
    await client.post("/api/tts/preview", json={"voice": "af_heart", "speed": 1.0})
    await client.post("/api/tts/preview", json={"voice": "am_adam", "speed": 1.5})

    # three distinct (voice, speed) tuples → three synth calls + three keys
    assert captured["calls"] == 3
    assert len(fake_redis.store) == 3


async def test_preview_corrupt_cache_entry_falls_back_to_synth(redis_client, monkeypatch):
    client, fake_redis = redis_client

    async def _ok(**kwargs: Any) -> TTSSynthesis:
        return _synth(b"fresh")

    captured = _patch_tts(monkeypatch, synthesize=_ok)

    body = {"voice": "am_adam", "speed": 1.0}
    # populate the cache key naturally, then corrupt its base64 payload
    resp_seed = await client.post("/api/tts/preview", json=body)
    assert resp_seed.status_code == 200
    (cache_key,) = list(fake_redis.store)
    fake_redis.store[cache_key] = {"audio_base64": "!!!not-base64!!!", "format": "wav"}

    resp = await client.post("/api/tts/preview", json=body)
    assert resp.status_code == 200
    assert resp.content == b"fresh"
    # corrupt entry was discarded → synth ran again (seed + this call = 2)
    assert captured["calls"] == 2
