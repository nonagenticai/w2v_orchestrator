"""Ken Burns paid work must survive a re-entry — and a restamp must re-record.

Three memories used to live outside the run's step ledger:

1. The per-shot prescale PIN lived only in Redis (24h TTL). Redis down at a
   shot's first dispatch, or the key gone, and the re-entered stage sent a
   different still under the SAME idempotency key — the recorder's (key + body
   hash) cache misses and the clip is paid for twice.
2. The recorder idempotency key carried no generation. video_production caches a
   SUBMIT for 24h, so after a restamp (same ``run_id``) a shot whose recording
   FAILED got the original failed job back instead of a new recording.
3. ``rescue_frozen_stills`` fetched + rendered + stored again on every re-entry.

Each test below fails on the tree before the fix.
"""

from __future__ import annotations

import base64
import io
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from PIL import Image
from w2v_common.auth import set_correlation_context

from src.services import tts_storage
from src.services.runs.run_context import RunContext
from src.services.stages.shared import ken_burns, ken_burns_local
from src.services.stages.shared.ken_burns import apply_ken_burns
from tests.support.reentry import InMemoryStepStore, ctx_with_ledger

_RASTER = {"capture_width": 5120, "capture_height": 2880}
#: The stage these helpers run inside, as ``DAGExecutor._run_one`` names it.
_COMPILE = "compile"


class _FakeStorage:
    async def initialize(self) -> None:
        pass

    async def store_stage_artifact(
        self, job_id: str, stage_name: str, data: bytes, filename: str
    ) -> str:
        return f"artifacts/{job_id}/{stage_name}/{filename}"

    async def get_presigned_url(
        self, path: str, expires: int | None = None, bucket: str | None = None
    ) -> str | None:
        return f"https://minio.test/{path}"


class _SharedCache:
    def __init__(self) -> None:
        self.data: dict[str, Any] = {}

    async def get(self, key: str) -> Any:
        return self.data.get(key)

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        self.data[key] = value
        return True

    async def set_nx(self, key: str, value: Any, *, ex: int) -> bool | None:
        if key in self.data:
            return False
        self.data[key] = value
        return True


class _DownCache:
    async def get(self, key: str) -> Any:
        raise ConnectionError("redis down")

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        raise ConnectionError("redis down")

    async def set_nx(self, key: str, value: Any, *, ex: int) -> bool | None:
        raise ConnectionError("redis down")


def _new_process() -> None:
    """A pod roll: every process-local dict of the module is empty."""
    for name in ("_OBSERVED_CAPTURE_RASTER", "_SHARED_RASTER_NEXT_PUBLISH", "_SHOT_PRESCALE_PIN"):
        getattr(ken_burns, name).clear()
    ken_burns._shared_raster_warned = False


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    tts_storage.set_storage_for_tests(_FakeStorage())
    _new_process()
    png = io.BytesIO()
    Image.new("RGB", (384, 216), (200, 40, 40)).save(png, format="PNG")

    async def _fetch(url: str, client: Any) -> tuple[bytes, str]:
        return (png.getvalue(), "image/png")

    async def _not_stored(*_a: Any, **_k: Any) -> None:
        # The pod rolled MID-recording: the clip is not stored, only a replay saves it.
        return None

    monkeypatch.setattr(ken_burns, "_fetch_own_asset", _fetch)
    monkeypatch.setattr(ken_burns, "find_existing_media", _not_stored)
    # As the executor does before every handler: the helpers' ledger rows must
    # land under the COMPILE stage's name, the one a restamp deletes by.
    set_correlation_context("test-run", _COMPILE)
    yield
    set_correlation_context(None, None)
    _new_process()
    tts_storage.set_storage_for_tests(None)


def _use_cache(monkeypatch, cache: Any) -> None:
    monkeypatch.setattr(ken_burns, "_shared_raster_cache", lambda: cache)


def _video(capture: dict[str, Any] | None = None) -> AsyncMock:
    v = AsyncMock()
    v.record_animation.return_value = "kb-1"
    v.wait_for_content_recording.return_value = {"duration_seconds": 3.0, **(capture or {})}
    v.get_recording_video.return_value = (b"\x00mp4", "video/mp4", None)
    return v


def _ctx(store: InMemoryStepStore) -> RunContext:
    return ctx_with_ledger(
        RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={}), store
    )


def _still(line: str) -> dict[str, Any]:
    return {
        "id": f"s-{line}",
        "line_id": line,
        "image_url": f"https://minio.test/{line}.png?X-Amz-Signature={uuid4().hex}",
        "seconds": 3.0,
    }


def _submits(video: AsyncMock) -> list[tuple[str, str]]:
    return [
        (c.kwargs["idempotency_key"], c.kwargs["params"]["image"])
        for c in video.record_animation.await_args_list
    ]


def _size(image_data_uri: str) -> tuple[int, int]:
    with Image.open(io.BytesIO(base64.b64decode(image_data_uri.split(",", 1)[1]))) as im:
        return im.size


def _restamp(store: InMemoryStepStore, run_id: Any, stage_name: str) -> None:
    """What ``_reset_downstream_stages`` does to the ledger: ``delete_for_stage``."""
    for key in [k for k in store.rows if k[0] == run_id and k[1] == stage_name]:
        del store.rows[key]


# ── 1. the prescale pin is durable ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_pin_survives_redis_down_at_reentry(monkeypatch) -> None:
    """Pod A (warm, Redis up) pins shot "a" to the sharp raster and dies before
    the clip is stored. Pod B (cold, Redis DOWN) re-enters: without a durable pin
    it would offer the ordered dims — different bytes, same key, a second bill."""
    store = InMemoryStepStore()
    ctx = _ctx(store)
    _use_cache(monkeypatch, _SharedCache())

    await apply_ken_burns(ctx, [_still("warmup")], client=_video(_RASTER))
    pod_a = _video(_RASTER)
    await apply_ken_burns(ctx, [_still("a")], client=pod_a)
    [(a_key, a_image)] = _submits(pod_a)
    assert _size(a_image) == (5120, 2880)  # premise: pod A really pinned the sharp raster

    _new_process()
    _use_cache(monkeypatch, _DownCache())

    pod_b = _video(_RASTER)
    await apply_ken_burns(ctx, [_still("a")], client=pod_b)

    assert _submits(pod_b) == [(a_key, a_image)]
    assert any(k.startswith("kb_pin:") for k in store.step_keys(_COMPILE))


@pytest.mark.asyncio
async def test_pin_survives_the_redis_key_expiring(monkeypatch) -> None:
    """Pod A (cold) pins the ordered dims; the Redis pin later expires (24h TTL)
    while the published raster stays. Pod B must still send the ordered-dims still."""
    store = InMemoryStepStore()
    ctx = _ctx(store)
    cache = _SharedCache()
    _use_cache(monkeypatch, cache)

    pod_a = _video(_RASTER)
    await apply_ken_burns(ctx, [_still("a")], client=pod_a)
    [(a_key, a_image)] = _submits(pod_a)
    order = pod_a.record_animation.await_args.kwargs
    assert _size(a_image) == (order["width"], order["height"])  # premise: cold

    for key in [k for k in cache.data if ":shot_prescale:" in k]:
        del cache.data[key]  # the pin TTL ran out; the raster observation did not
    _new_process()

    pod_b = _video(_RASTER)
    await apply_ken_burns(ctx, [_still("a")], client=pod_b)

    assert _submits(pod_b) == [(a_key, a_image)]


# ── 2. a restamp moves the recorder idempotency key ──────────────────────────


@pytest.mark.asyncio
async def test_reentry_keeps_the_key_and_a_restamp_changes_it(monkeypatch) -> None:
    store = InMemoryStepStore()
    ctx = _ctx(store)
    _use_cache(monkeypatch, _SharedCache())

    first = _video()
    await apply_ken_burns(ctx, [_still("a")], client=first)
    [(key_1, _)] = _submits(first)
    assert key_1.startswith(f"{ctx.run_id}:ken_burns:")

    # Pod death → re-entry: the SAME key, so the in-flight job is replayed.
    _new_process()
    again = _video()
    await apply_ken_burns(ctx, [_still("a")], client=again)
    assert [k for k, _ in _submits(again)] == [key_1]

    # Restamp: the compile stage's ledger rows are deleted with the stage reset.
    _restamp(store, ctx.run_id, _COMPILE)
    _new_process()
    rerun = _video()
    await apply_ken_burns(ctx, [_still("a")], client=rerun)
    [(key_2, _)] = _submits(rerun)

    assert key_2 != key_1, "a restamped re-run must not get the original (failed) job back"
    assert key_2.startswith(f"{ctx.run_id}:ken_burns:")


@pytest.mark.asyncio
async def test_saturation_retry_keys_follow_the_generation(monkeypatch) -> None:
    store = InMemoryStepStore()
    ctx = _ctx(store)
    _use_cache(monkeypatch, _SharedCache())

    async def _no_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(ken_burns, "_saturation_sleep", _no_sleep)
    saturated = RuntimeError(
        "video_production job kb-1 failed: QUEUE_SATURATED: no per-pod recording slot"
    )

    def _sat_once() -> AsyncMock:
        v = _video()
        v.wait_for_content_recording.side_effect = [saturated, {"duration_seconds": 3.0}]
        return v

    before = _sat_once()
    await apply_ken_burns(ctx, [_still("a")], client=before)
    base_1, sat_1 = [k for k, _ in _submits(before)]
    assert sat_1 == f"{base_1}:sat1"

    _restamp(store, ctx.run_id, _COMPILE)
    _new_process()
    after = _sat_once()
    await apply_ken_burns(ctx, [_still("a")], client=after)
    base_2, sat_2 = [k for k, _ in _submits(after)]

    assert sat_2 == f"{base_2}:sat1"
    assert {base_2, sat_2}.isdisjoint({base_1, sat_1})


@pytest.mark.asyncio
async def test_without_a_durable_ledger_the_key_is_unchanged(monkeypatch) -> None:
    """No DB-backed run (NullStepLedger): a generation that cannot be stored would
    change on every call, so the key stays the pre-generation one."""
    _use_cache(monkeypatch, _SharedCache())
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})
    video = _video()
    await apply_ken_burns(ctx, [_still("a")], client=video)
    [(key, _)] = _submits(video)
    prefix = f"{ctx.run_id}:ken_burns:"
    assert key.startswith(prefix) and ":" not in key[len(prefix) :]


# ── 3. the local rescue is not re-rendered on re-entry ───────────────────────


@pytest.mark.asyncio
async def test_rescue_reentry_does_not_render_again(monkeypatch) -> None:
    store = InMemoryStepStore()
    ctx = _ctx(store)
    render = AsyncMock(return_value=b"\x00mp4")
    fetch = AsyncMock(return_value=(b"\x89PNG", "image/png"))
    monkeypatch.setattr(ken_burns_local, "render_local_ken_burns", render)
    monkeypatch.setattr(ken_burns, "_fetch_own_asset", fetch)

    first = await ken_burns_local.rescue_frozen_stills(ctx, [_still("a")], budget_seconds=600.0)
    assert render.await_count == 1
    url = first[0]["video_url"]

    # Re-entry on a fresh pod: the still's presigned query has rotated.
    _new_process()
    again = await ken_burns_local.rescue_frozen_stills(ctx, [_still("a")], budget_seconds=600.0)

    assert render.await_count == 1, "re-entry re-rendered a still it already rescued"
    assert fetch.await_count == 1
    assert again[0]["video_url"] == url
    stored = store.rows[(ctx.run_id, _COMPILE, "kb_local:s-a")].result
    # A reference to the stored object — never the rendered bytes.
    assert isinstance(stored, dict)
    assert set(stored["value"]) == {"media_id", "url"}
    assert stored["value"]["url"] == url


@pytest.mark.asyncio
async def test_rescue_with_a_different_frame_renders_again(monkeypatch) -> None:
    store = InMemoryStepStore()
    ctx = _ctx(store)
    render = AsyncMock(return_value=b"\x00mp4")
    monkeypatch.setattr(ken_burns_local, "render_local_ken_burns", render)

    first = await ken_burns_local.rescue_frozen_stills(ctx, [_still("a")], budget_seconds=600.0)
    longer = {**_still("a"), "seconds": 5.0}
    second = await ken_burns_local.rescue_frozen_stills(ctx, [longer], budget_seconds=600.0)

    # A changed input is a NEW render at the new length, not the stored clip.
    assert [c.kwargs["duration_s"] for c in render.await_args_list] == [3.0, 5.0]
    assert second[0]["video_url"] != first[0]["video_url"]


# ── 4. every kb_pin lookup says where its target came from ───────────────────


def _pin_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    # ken_burns' OWN pin lines only. ``step_ledger`` also logs ``step=kb_pin:<media>``
    # (hit/miss) and, when its records are captured (as in CI), a bare ``"kb_pin" in``
    # filter interleaved them and broke the ordering assertions.
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == ken_burns.__name__ and r.getMessage().startswith("ken_burns: kb_pin")
    ]


@pytest.mark.asyncio
async def test_kb_pin_logs_redis_won_then_ledger(monkeypatch, caplog) -> None:
    """First dispatch claims the pin in Redis; the re-entry is answered by the
    ledger row. Both name their source and the target."""
    caplog.set_level("INFO", logger=ken_burns.__name__)
    store = InMemoryStepStore()
    ctx = _ctx(store)
    _use_cache(monkeypatch, _SharedCache())
    ledger = ken_burns._ken_burns_ledger(ctx)

    args = (str(ctx.run_id), "m1", 1920, 1080, None)
    await ken_burns._pin_shot_prescale_target(*args, ledger=ledger)
    await ken_burns._pin_shot_prescale_target(*args, ledger=ledger)

    lines = _pin_lines(caplog)
    assert len(lines) == 2, lines
    assert "media=m1 source=redis-won target=1920x1080" in lines[0]
    assert "media=m1 source=ledger target=1920x1080" in lines[1]


@pytest.mark.asyncio
async def test_kb_pin_logs_redis_lost_with_the_winners_target(monkeypatch, caplog) -> None:
    caplog.set_level("INFO", logger=ken_burns.__name__)
    cache = _SharedCache()
    cache.data[ken_burns._shot_pin_cache_key("r1", "m1")] = [5120, 2880]
    _use_cache(monkeypatch, cache)

    target = await ken_burns._pin_shot_prescale_target("r1", "m1", 1920, 1080, None)

    assert target == (5120, 2880)
    assert any("source=redis-lost target=5120x2880" in line for line in _pin_lines(caplog))


@pytest.mark.asyncio
async def test_kb_pin_logs_local_when_the_cache_is_down(monkeypatch, caplog) -> None:
    caplog.set_level("INFO", logger=ken_burns.__name__)
    _use_cache(monkeypatch, _DownCache())

    await ken_burns._pin_shot_prescale_target("r1", "m1", 1920, 1080, None)

    assert any("source=local target=1920x1080" in line for line in _pin_lines(caplog))
