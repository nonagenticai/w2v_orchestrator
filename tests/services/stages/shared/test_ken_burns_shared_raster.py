"""The observed capture raster must survive a pod roll.

``ken_burns`` prescales each still to the raster the recorder REPORTED for the
order. That observation used to live only in a process-local dict, so the
prescale target — and therefore the bytes of the still in the request body —
depended on pod warmth. video_production_service replays a submit only when the
``idempotency_key`` AND the body hash match, so a warm pod's submit and the cold
successor's resubmit of the SAME key carried different bytes and bought a second
recording. These tests pin the shared cache that closes that, and that it can
never fail a compile.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from PIL import Image

from src.services import tts_storage
from src.services.runs.run_context import RunContext
from src.services.stages.shared import ken_burns
from src.services.stages.shared.ken_burns import apply_ken_burns

_RASTER = {"capture_width": 5120, "capture_height": 2880}


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
    """Stands in for the orchestrator's Redis wrapper: JSON-ish ``get``/``set``."""

    def __init__(self) -> None:
        self.data: dict[str, Any] = {}
        self.ttls: dict[str, int | None] = {}
        self.gets = 0

    async def get(self, key: str) -> Any:
        self.gets += 1
        return self.data.get(key)

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        self.data[key] = value
        self.ttls[key] = ttl
        return True

    async def set_nx(self, key: str, value: Any, *, ex: int) -> bool | None:
        # w2v_common ``RedisClient.set_nx``: True = won, False = already held.
        if key in self.data:
            return False
        self.data[key] = value
        self.ttls[key] = ex
        return True


class _DownCache:
    """Redis is unreachable: every call raises, the way a raw socket error would."""

    def __init__(self) -> None:
        self.calls = 0

    async def get(self, key: str) -> Any:
        self.calls += 1
        raise ConnectionError("redis down")

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        self.calls += 1
        raise ConnectionError("redis down")

    async def set_nx(self, key: str, value: Any, *, ex: int) -> bool | None:
        self.calls += 1
        raise ConnectionError("redis down")


class _UnconnectedCache:
    """The REAL wrapper's outage shape: it swallows the error and answers
    ``None``/``False`` (w2v_common ``RedisClient`` before ``connect`` or on a
    ``RedisError``) instead of raising."""

    async def get(self, key: str) -> Any:
        return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        return False

    async def set_nx(self, key: str, value: Any, *, ex: int) -> bool | None:
        return None


def _new_process() -> None:
    """What a pod roll does to this module: every process-local dict is empty."""
    for name in ("_OBSERVED_CAPTURE_RASTER", "_SHARED_RASTER_NEXT_PUBLISH", "_SHOT_PRESCALE_PIN"):
        getattr(ken_burns, name, {}).clear()
    if hasattr(ken_burns, "_shared_raster_warned"):
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
        # The pod rolled MID-recording: the clip is not in the store yet, so the
        # content-addressed lookup cannot save the resubmit — only a replay can.
        return None

    monkeypatch.setattr(ken_burns, "_fetch_own_asset", _fetch)
    monkeypatch.setattr(ken_burns, "find_existing_media", _not_stored)
    yield
    _new_process()
    tts_storage.set_storage_for_tests(None)


def _use_cache(monkeypatch, cache: Any) -> None:
    # ``raising=False`` so that on a tree WITHOUT the shared cache this file
    # fails on its assertions (the defect), not on a missing attribute.
    monkeypatch.setattr(ken_burns, "_shared_raster_cache", lambda: cache, raising=False)


def _video(capture: dict[str, Any] | None = None) -> AsyncMock:
    v = AsyncMock()
    v.record_animation.return_value = "kb-1"
    v.wait_for_content_recording.return_value = {"duration_seconds": 3.0, **(capture or {})}
    v.get_recording_video.return_value = (b"\x00mp4", "video/mp4", None)
    return v


def _ctx() -> RunContext:
    return RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})


def _still(line: str = "a") -> dict[str, Any]:
    return {
        "id": f"s-{line}",
        "line_id": line,
        "image_url": f"https://minio.test/{line}.png",
        "seconds": 3.0,
    }


def _is_pin(key: str) -> bool:
    return ":shot_prescale:" in key


def _submit(video: AsyncMock, call: int = -1) -> tuple[str, str]:
    kwargs = video.record_animation.await_args_list[call].kwargs
    return kwargs["idempotency_key"], kwargs["params"]["image"]


def _sent_size(video: AsyncMock, call: int = -1) -> tuple[int, int]:
    blob = base64.b64decode(_submit(video, call)[1].split(",", 1)[1])
    with Image.open(io.BytesIO(blob)) as im:
        return im.size


@pytest.mark.asyncio
async def test_a_cold_pod_resubmits_the_same_bytes_the_warm_pod_sent(monkeypatch) -> None:
    """THE regression. Same run, same shot, same idempotency key — across a pod
    roll the still handed to the recorder must be byte-identical, or the
    recorder's (key + body hash) cache misses and records the clip a second time."""
    cache = _SharedCache()
    _use_cache(monkeypatch, cache)
    ctx = _ctx()

    # Pod A warms up: an earlier shot's recording reports the real raster.
    await apply_ken_burns(ctx, [_still("warmup")], client=_video(_RASTER))

    # Pod A, warm, submits shot "a" — then rolls before the clip is stored.
    warm = _video(_RASTER)
    await apply_ken_burns(ctx, [_still("a")], client=warm)
    warm_key, warm_image = _submit(warm)
    assert _sent_size(warm) == (5120, 2880)  # the premise: pod A really was warm

    _new_process()

    # Pod B, cold, resumes the run and resubmits shot "a".
    cold = _video(_RASTER)
    await apply_ken_burns(ctx, [_still("a")], client=cold)
    cold_key, cold_image = _submit(cold)

    assert cold_key == warm_key
    assert cold_image == warm_image


@pytest.mark.asyncio
async def test_the_observation_is_published_with_a_24h_ttl_and_read_once_per_call(
    monkeypatch,
) -> None:
    cache = _SharedCache()
    _use_cache(monkeypatch, cache)

    await apply_ken_burns(_ctx(), [_still("a"), _still("b"), _still("c")], client=_video(_RASTER))

    shared = {k: v for k, v in cache.data.items() if not _is_pin(k)}
    assert list(shared.values()) == [[5120, 2880]]
    assert set(cache.ttls.values()) == {24 * 60 * 60}  # the raster AND the shot pins
    # One read for the whole fan-out, not one per shot.
    assert cache.gets == 1

    # A warm process does no shared read at all.
    await apply_ken_burns(_ctx(), [_still("d")], client=_video(_RASTER))
    assert cache.gets == 1


@pytest.mark.asyncio
async def test_cache_down_on_read_and_write_is_todays_behaviour(monkeypatch, caplog) -> None:
    """Redis unreachable in BOTH directions: no exception, no soft-failed shot,
    and exactly the process-local behaviour — cold = ordered dims, then warm."""
    cache = _DownCache()
    _use_cache(monkeypatch, cache)
    notes: list[str] = []

    with caplog.at_level(logging.WARNING, logger=ken_burns.__name__):
        first = _video(_RASTER)
        out = await apply_ken_burns(_ctx(), [_still("a")], client=first, notes=notes)
        ordered = first.record_animation.await_args.kwargs
        assert _sent_size(first) == (ordered["width"], ordered["height"])  # cold
        assert out[0]["video_url"]

        second = _video(_RASTER)
        out = await apply_ken_burns(_ctx(), [_still("b")], client=second, notes=notes)
        assert _sent_size(second) == (5120, 2880)  # process-local observation
        assert out[0]["video_url"]

    assert notes == []  # the outage did not degrade a single shot
    assert cache.calls >= 2  # both the read and the write were really attempted
    warnings = [r for r in caplog.records if "shared capture-raster cache" in r.getMessage()]
    assert len(warnings) == 1  # announced once, not per shot


@pytest.mark.asyncio
async def test_a_hung_cache_is_bounded(monkeypatch) -> None:
    class _Hung:
        async def get(self, key: str) -> Any:
            await asyncio.sleep(3600)

        async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
            await asyncio.sleep(3600)
            return True

        async def set_nx(self, key: str, value: Any, *, ex: int) -> bool | None:
            await asyncio.sleep(3600)
            return True

    _use_cache(monkeypatch, _Hung())
    monkeypatch.setattr(ken_burns, "_SHARED_RASTER_IO_TIMEOUT_S", 0.01, raising=False)
    video = _video(_RASTER)
    out = await asyncio.wait_for(apply_ken_burns(_ctx(), [_still("a")], client=video), timeout=5)
    assert out[0]["video_url"]


@pytest.mark.parametrize(
    "garbage",
    ["5120x2880"],
)
@pytest.mark.asyncio
async def test_a_garbage_cached_value_is_ignored(monkeypatch, garbage: object) -> None:
    cache = _SharedCache()
    _use_cache(monkeypatch, cache)
    probe = _video()
    await apply_ken_burns(_ctx(), [_still("probe")], client=probe)
    ordered = probe.record_animation.await_args.kwargs
    assert cache.gets == 1
    cache.data[
        ken_burns._shared_raster_cache_key(
            ordered["width"], ordered["height"], ordered["quality_preset"]
        )
    ] = garbage

    _new_process()
    video = _video()  # reports no raster, so nothing can overwrite the garbage
    out = await apply_ken_burns(_ctx(), [_still("a")], client=video)

    assert cache.gets == 2  # the garbage really was read
    assert _sent_size(video) == (ordered["width"], ordered["height"])
    assert ken_burns._OBSERVED_CAPTURE_RASTER == {}
    assert out[0]["video_url"]


@pytest.mark.asyncio
async def test_a_valid_cached_value_is_adopted(monkeypatch) -> None:
    """The positive control for the garbage test: same seam, a GOOD value, and
    the prescale follows it — so "ignored" above is not "the read is dead"."""
    cache = _SharedCache()
    _use_cache(monkeypatch, cache)
    probe = _video()
    await apply_ken_burns(_ctx(), [_still("probe")], client=probe)
    ordered = probe.record_animation.await_args.kwargs
    cache.data[
        ken_burns._shared_raster_cache_key(
            ordered["width"], ordered["height"], ordered["quality_preset"]
        )
    ] = [5120, 2880]

    _new_process()
    video = _video()
    await apply_ken_burns(_ctx(), [_still("a")], client=video)
    assert _sent_size(video) == (5120, 2880)


# ---------------------------------------------------------------------------
# The per-SHOT prescale pin.
#
# The shared raster above only helps a pod that starts AFTER an observation was
# published. The first pod to place an order sends its first batch at the
# ordered dims, and the raster those very recordings publish then changes the
# prescale target for (a) the successor pod after a roll and (b) an in-process
# ``stage_gather`` retry — same idempotency key, different still bytes, so the
# recorder's (key + body hash) cache misses and the clip is paid for twice.
# ---------------------------------------------------------------------------


class _Recorder:
    """A video_production stand-in that keeps every submit it was sent.

    ``hang_first`` names a shot whose FIRST wait never returns, which is the only
    way ``stage_gather`` retries ``_gen`` (it soft-fails every ``Exception``
    itself, so only the per-item timeout re-runs it).
    """

    def __init__(self, hang_first: str | None = None) -> None:
        self.submits: list[dict[str, Any]] = []
        self._hang_first = hang_first
        self._hung = False

    async def record_animation(self, **kwargs: Any) -> str:
        self.submits.append(kwargs)
        return f"job-{len(self.submits)}-{kwargs['part_id']}"

    async def wait_for_content_recording(self, job_id: str, timeout: float) -> dict[str, Any]:
        if self._hang_first and job_id.endswith(f"kb-s-{self._hang_first}") and not self._hung:
            self._hung = True
            await asyncio.sleep(3600)
        return {"duration_seconds": 3.0, **_RASTER}

    async def get_recording_video(self, job_id: str) -> tuple[bytes, str, None]:
        return (b"\x00mp4", "video/mp4", None)

    def for_shot(self, line: str) -> list[tuple[str, str]]:
        """``(idempotency_key, image data URI)`` of every submit for one shot."""
        return [
            (s["idempotency_key"], s["params"]["image"])
            for s in self.submits
            if s["part_id"] == f"kb-s-{line}"
        ]


def _client(recorder: _Recorder) -> Any:
    """``apply_ken_burns`` types ``client`` as the real VideoProductionClient."""
    return recorder


def _size(image_data_uri: str) -> tuple[int, int]:
    with Image.open(io.BytesIO(base64.b64decode(image_data_uri.split(",", 1)[1]))) as im:
        return im.size


@pytest.mark.asyncio
async def test_a_cold_start_first_batch_is_replayed_by_the_successor_pod(monkeypatch) -> None:
    """Nothing in Redis. Pod A dispatches its first batch at the ordered dims;
    those recordings publish the real raster; pod A rolls. Pod B must send, for
    the SAME shot, the SAME key and the SAME bytes — not the raster-sized still."""
    cache = _SharedCache()
    _use_cache(monkeypatch, cache)
    ctx = _ctx()

    pod_a = _Recorder()
    await apply_ken_burns(ctx, [_still("a"), _still("b")], client=_client(pod_a))
    (a_key, a_image) = pod_a.for_shot("a")[0]
    order = pod_a.submits[0]
    assert _size(a_image) == (order["width"], order["height"])  # premise: cold
    # Premise: the raster really was published, so pod B CAN be tempted by it.
    assert [5120, 2880] in [v for k, v in cache.data.items() if not _is_pin(k)]

    _new_process()

    pod_b = _Recorder()
    await apply_ken_burns(ctx, [_still("a"), _still("b")], client=_client(pod_b))

    assert pod_b.for_shot("a") == [(a_key, a_image)]
    assert pod_b.for_shot("b") == pod_a.for_shot("b")


@pytest.mark.asyncio
async def test_a_different_run_is_not_held_to_another_runs_pin(monkeypatch) -> None:
    """The pin is per RUN: it must not freeze the estate at the ordered dims.
    The NEXT run starts from the published raster, as before."""
    cache = _SharedCache()
    _use_cache(monkeypatch, cache)

    await apply_ken_burns(_ctx(), [_still("a")], client=_client(_Recorder()))
    _new_process()
    later = _Recorder()
    await apply_ken_burns(_ctx(), [_still("a")], client=_client(later))

    assert _size(later.for_shot("a")[0][1]) == (5120, 2880)


@pytest.mark.asyncio
async def test_two_pods_racing_the_pin_both_use_the_winner(monkeypatch) -> None:
    """Pod B is WARM (it would choose 5120x2880) but pod A's pin for this run got
    there first: pod B reads the winner back and sends pod A's bytes."""
    cache = _SharedCache()
    _use_cache(monkeypatch, cache)
    ctx = _ctx()

    pod_a = _Recorder()
    await apply_ken_burns(ctx, [_still("a")], client=_client(pod_a))  # cold: pins ordered dims

    # Pod B's process-local observation says 5120x2880 for this order.
    order = pod_a.submits[0]
    getattr(ken_burns, "_SHOT_PRESCALE_PIN", {}).clear()
    assert ken_burns._observed_capture_raster(
        order["width"], order["height"], order["quality_preset"]
    ) == (5120, 2880)

    pod_b = _Recorder()
    await apply_ken_burns(ctx, [_still("a")], client=_client(pod_b))
    assert pod_b.for_shot("a") == pod_a.for_shot("a")


@pytest.mark.asyncio
async def test_the_pin_is_written_before_the_first_paid_call(monkeypatch) -> None:
    cache = _SharedCache()
    _use_cache(monkeypatch, cache)
    pins_at_submit: list[list[Any]] = []

    class _Probe(_Recorder):
        async def record_animation(self, **kwargs: Any) -> str:
            pins_at_submit.append([v for k, v in cache.data.items() if _is_pin(k)])
            return await super().record_animation(**kwargs)

    ctx = _ctx()
    probe = _Probe()
    await apply_ken_burns(ctx, [_still("a")], client=_client(probe))

    order = probe.submits[0]
    assert pins_at_submit == [[[order["width"], order["height"]]]]
    # Scoped by run AND by the media id the idempotency key is built from.
    media_id = order["idempotency_key"].rsplit(":", 1)[1]
    assert [k for k in cache.data if _is_pin(k)] == [
        ken_burns._shot_pin_cache_key(str(ctx.run_id), media_id)
    ]


@pytest.mark.parametrize("cache_factory", [_SharedCache])
@pytest.mark.asyncio
async def test_an_in_process_retry_resends_the_same_bytes(monkeypatch, cache_factory) -> None:
    """Shot "a" is submitted cold and its wait hangs; sibling "b" completes and
    reports the raster; ``stage_gather`` then retries "a". The retry carries the
    SAME key, so it must carry the SAME still — with or without Redis."""
    _use_cache(monkeypatch, cache_factory())
    # Shrink the per-item ceiling (stage_call's floor is 1.0s) so the hung wait
    # times out and ``stage_gather`` re-runs ``_gen`` for that shot.
    monkeypatch.setattr(ken_burns, "_MIN_PHASE_BUDGET_S", 0.0)
    monkeypatch.setattr(ken_burns, "_PER_ITEM_OVERHEAD_S", 1.8)

    recorder = _Recorder(hang_first="a")
    await apply_ken_burns(
        _ctx(), [_still("a"), _still("b")], client=_client(recorder), budget_seconds=0.2
    )

    sent = recorder.for_shot("a")
    assert len(sent) == 2  # premise: "a" really was submitted twice
    order = recorder.submits[0]
    assert ken_burns._observed_capture_raster(  # premise: "b" really did warm the process
        order["width"], order["height"], order["quality_preset"]
    ) == (5120, 2880)
    assert sent[0] == sent[1]


@pytest.mark.parametrize("cache_factory", [_DownCache])
@pytest.mark.asyncio
async def test_redis_down_the_stage_still_records_every_shot(monkeypatch, cache_factory) -> None:
    _use_cache(monkeypatch, cache_factory())
    notes: list[str] = []
    recorder = _Recorder()
    ctx = _ctx()

    out = await apply_ken_burns(
        ctx, [_still("a"), _still("b")], client=_client(recorder), notes=notes
    )

    assert [bool(shot.get("video_url")) for shot in out] == [True, True]
    assert notes == []
    # Without Redis the pin still holds for this process: a second pass over the
    # same run (a compile-stage retry) resends the first pass's bytes.
    again = _Recorder()
    await apply_ken_burns(ctx, [_still("a")], client=_client(again), notes=notes)
    assert again.for_shot("a") == recorder.for_shot("a")
    assert notes == []


def _five() -> list[dict[str, Any]]:
    """More stills than ``_CONCURRENCY``: "e" is first dispatched only after
    a first-batch recording has completed and reported the raster."""
    assert ken_burns._CONCURRENCY == 3
    return [_still(line) for line in "abcde"]


@pytest.mark.asyncio
async def test_a_cold_start_softens_only_the_first_batch(monkeypatch) -> None:
    """The pin is per SHOT. Only the stills dispatched before any raster is
    known go out at the ordered dims; a later still of the SAME run is sent at
    the sharp raster — a run-wide pin would have held it soft too."""
    _use_cache(monkeypatch, _SharedCache())
    recorder = _Recorder()
    await apply_ken_burns(_ctx(), _five(), client=_client(recorder))

    order = recorder.submits[0]
    ordered = (order["width"], order["height"])
    sizes = {line: _size(recorder.for_shot(line)[0][1]) for line in "abcde"}
    assert [sizes[line] for line in "abc"] == [ordered] * 3  # premise: cold batch
    assert sizes["e"] == (5120, 2880)


@pytest.mark.asyncio
async def test_the_successor_pod_resends_a_later_sharp_shot_byte_identically(
    monkeypatch,
) -> None:
    """Shot "e" was first dispatched sharp. The successor pod must resend it
    sharp even if it can no longer see the raster itself (shared key expired)."""
    cache = _SharedCache()
    _use_cache(monkeypatch, cache)
    ctx = _ctx()
    pod_a = _Recorder()
    await apply_ken_burns(ctx, _five(), client=_client(pod_a))
    assert _size(pod_a.for_shot("e")[0][1]) == (5120, 2880)  # premise

    _new_process()
    for key in [k for k in cache.data if not _is_pin(k)]:
        del cache.data[key]  # pod B is cold AND has nothing shared to adopt

    pod_b = _Recorder()
    await apply_ken_burns(ctx, [_still("e"), _still("a")], client=_client(pod_b))

    assert pod_b.for_shot("e") == pod_a.for_shot("e")
    assert pod_b.for_shot("a") == pod_a.for_shot("a")  # and the soft one stays soft


class _LostPinCache(_SharedCache):
    """The claim is lost and the winner's value cannot be read ``fail_reads`` times."""

    def __init__(self, fail_reads: int) -> None:
        super().__init__()
        self.fail_reads = fail_reads

    async def get(self, key: str) -> Any:
        if _is_pin(key) and self.fail_reads > 0:
            self.fail_reads -= 1
            raise ConnectionError("redis read failed")
        return await super().get(key)


async def _lost_pin(monkeypatch, fail_reads: int) -> tuple[_Recorder, list[str], Any]:
    cache = _LostPinCache(fail_reads)
    _use_cache(monkeypatch, cache)
    ctx = _ctx()
    pod_a = _Recorder()
    await apply_ken_burns(ctx, [_still("a")], client=_client(pod_a))  # cold: holds the pin
    ken_burns._SHOT_PRESCALE_PIN.clear()  # pod B: warm on the raster, no local pin
    cache.fail_reads = fail_reads

    notes: list[str] = []
    pod_b = _Recorder()
    out = await apply_ken_burns(ctx, [_still("a")], client=_client(pod_b), notes=notes)
    return pod_b, notes, (pod_a, out)


@pytest.mark.asyncio
async def test_a_lost_pin_whose_read_fails_once_is_retried_and_replayed(monkeypatch) -> None:
    pod_b, notes, (pod_a, out) = await _lost_pin(monkeypatch, fail_reads=1)
    assert pod_b.for_shot("a") == pod_a.for_shot("a")
    assert notes == []
    assert out[0]["video_url"]


@pytest.mark.asyncio
async def test_a_lost_pin_that_stays_unreadable_is_not_submitted_blind(monkeypatch) -> None:
    """Another dispatch holds the pin and we cannot learn its bytes: no paid
    call. The shot keeps its still and says why; the stage does not fail."""
    pod_b, notes, (_pod_a, out) = await _lost_pin(monkeypatch, fail_reads=99)
    assert pod_b.submits == []
    assert "video_url" not in out[0]
    assert len(notes) == 2 and "_ShotPinUnreadableError" in notes[0]
