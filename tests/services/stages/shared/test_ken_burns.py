"""Tests for the on-by-default Ken Burns still→clip shot post-pass."""

from __future__ import annotations

import socket
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from w2v_common.net import safe_fetch

from src.services import tts_storage
from src.services.runs.run_context import RunContext
from src.services.stages.shared import ken_burns
from src.services.stages.shared.degradation import (
    KEN_BURNS_ALL_STILLS_FROZEN,
    KEN_BURNS_STILL_QUEUE_SATURATED,
    KEN_BURNS_STILL_SOFT_FAILED,
)
from src.services.stages.shared.ken_burns import apply_ken_burns
from src.services.visual_media import MEDIA_FETCH_USER_AGENT

#: Captured at import time, BEFORE the autouse ``_stub_fetch`` fixture replaces the
#: attribute — the UA tests below must exercise the REAL fetch, not the stub.
_REAL_FETCH_OWN_ASSET = ken_burns._fetch_own_asset


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


@pytest.fixture
def fake_storage():
    tts_storage.set_storage_for_tests(_FakeStorage())
    yield
    tts_storage.set_storage_for_tests(None)


@pytest.fixture(autouse=True)
def _stub_fetch(monkeypatch):
    """Avoid real HTTP — every fetch returns a tiny PNG."""

    async def _fake_fetch(url: str, client: Any) -> tuple[bytes, str]:
        return (b"\x89PNG\r\n", "image/png")

    monkeypatch.setattr(ken_burns, "_fetch_own_asset", _fake_fetch)


def _video_fake() -> AsyncMock:
    v = AsyncMock()
    v.record_animation.return_value = "kb-1"
    v.wait_for_content_recording.return_value = {"duration_seconds": 5.0}
    v.get_recording_video.return_value = (b"\x00mp4", "video/mp4", None)
    return v


def _ctx(config_snapshot: dict[str, Any]) -> RunContext:
    return RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot=config_snapshot)


def _still(line: str, seconds: float | None = 3.0) -> dict[str, Any]:
    """A bare still shot. ``seconds=None`` omits the duration field entirely, which
    is how a shot that carries no own duration reaches the per-shot resolver."""
    shot: dict[str, Any] = {
        "id": f"s-{line}",
        "line_id": line,
        "image_url": f"https://minio.test/{line}.png",
    }
    if seconds is not None:
        shot["seconds"] = seconds
    return shot


@pytest.mark.asyncio
async def test_absent_block_upgrades_still_with_defaults(fake_storage) -> None:
    """THE DEFECT: the block was never set by anything (planner/gitops/brief), so
    the old presence-opt-in gate was unreachable and every still froze. Absent
    block now means ENABLED, on the ``_DEFAULT_*`` sub-knobs."""
    video = _video_fake()
    out = await apply_ken_burns(_ctx({}), [_still("a", seconds=None)], client=video)
    assert out[0]["video_url"].startswith("https://")
    assert video.record_animation.await_count == 1
    params = video.record_animation.await_args.kwargs["params"]
    assert params["preset"] == "zoom_in"
    # `contain` is the default now that motion is on for EVERY bare still: the
    # cover path zooms to 1.14 with no safe inset and would crop ~6% off each
    # side of a slide / chart / screenshot we did not choose.
    assert params["fit"] == "contain"
    # No own ``seconds`` on the shot ⇒ the _DEFAULT_DURATION_S fallback.
    assert params["duration_ms"] == 5000


@pytest.mark.asyncio
async def test_empty_block_upgrades_still(fake_storage) -> None:
    """``ken_burns: {}`` — what ``PlanDocument.ken_burns`` defaults to — is not
    an opt-out; it carries no ``enabled`` key, so motion stays on."""
    video = _video_fake()
    out = await apply_ken_burns(_ctx({"ken_burns": {}}), [_still("a")], client=video)
    assert out[0]["video_url"].startswith("https://")
    assert video.record_animation.await_count == 1


@pytest.mark.asyncio
async def test_block_without_enabled_key_upgrades_still(fake_storage) -> None:
    """A block that only tunes a knob is not an opt-out either."""
    video = _video_fake()
    out = await apply_ken_burns(
        _ctx({"ken_burns": {"preset": "pan_right"}}), [_still("a")], client=video
    )
    assert out[0]["video_url"].startswith("https://")
    assert video.record_animation.await_args.kwargs["params"]["preset"] == "pan_right"


@pytest.mark.asyncio
async def test_disabled_block_is_noop() -> None:
    video = _video_fake()
    out = await apply_ken_burns(
        _ctx({"ken_burns": {"enabled": False}}), [_still("a")], client=video
    )
    video.record_animation.assert_not_awaited()
    assert "video_url" not in out[0]


@pytest.mark.parametrize("falsy", [False])
@pytest.mark.asyncio
async def test_bare_falsy_scalar_in_brief_is_an_opt_out(falsy: object) -> None:
    """THE INVERSION: the project brief is ``extra='allow'`` and accepts
    ``ken_burns`` as a BARE SCALAR, and ``_lift_brief_config_keys`` copies a brief
    block to the snapshot top level only when it is TRUTHY. So an authored
    ``ken_burns: false`` never reaches the snapshot, and reading the snapshot
    alone would call that "absent" and turn motion ON — the exact opposite of
    what the author asked for. The brief is consulted for precisely this case."""
    video = _video_fake()
    shots = [_still("a")]
    out = await apply_ken_burns(_ctx({"brief": {"ken_burns": falsy}}), shots, client=video)
    assert out is shots
    video.record_animation.assert_not_awaited()


@pytest.mark.asyncio
async def test_bare_falsy_scalar_at_snapshot_top_level_is_an_opt_out() -> None:
    """Same opt-out, but for a per-run ``config_overrides`` that set the bare
    scalar directly on the snapshot — it is not a Mapping, so it must not fall
    through the block-shaped branch into ENABLED."""
    video = _video_fake()
    shots = [_still("a")]
    out = await apply_ken_burns(_ctx({"ken_burns": False}), shots, client=video)
    assert out is shots
    video.record_animation.assert_not_awaited()


@pytest.mark.asyncio
async def test_bare_true_scalar_enables_on_defaults(fake_storage) -> None:
    """``ken_burns: true`` as a bare scalar is a plain opt-in carrying no knobs."""
    video = _video_fake()
    out = await apply_ken_burns(_ctx({"ken_burns": True}), [_still("a")], client=video)
    assert out[0]["video_url"].startswith("https://")
    assert video.record_animation.await_args.kwargs["params"]["fit"] == "contain"


@pytest.mark.asyncio
async def test_snapshot_block_wins_over_brief(fake_storage) -> None:
    """The brief is only consulted when the snapshot has no key at all — a lifted
    or overridden block still takes precedence."""
    video = _video_fake()
    ctx = _ctx({"ken_burns": {"preset": "pan_up"}, "brief": {"ken_burns": False}})
    out = await apply_ken_burns(ctx, [_still("a")], client=video)
    assert out[0]["video_url"].startswith("https://")
    assert video.record_animation.await_args.kwargs["params"]["preset"] == "pan_up"


@pytest.mark.parametrize("falsy", [False])
@pytest.mark.asyncio
async def test_explicit_falsy_enabled_is_noop(falsy: object) -> None:
    """An EXPLICIT opt-out still wins, in every falsy shape a config snapshot
    can carry it (JSON/YAML round-trips give all four)."""
    video = _video_fake()
    shots = [_still("a")]
    out = await apply_ken_burns(_ctx({"ken_burns": {"enabled": falsy}}), shots, client=video)
    assert out is shots
    video.record_animation.assert_not_awaited()


@pytest.mark.asyncio
async def test_upgrades_still_to_clip(fake_storage) -> None:
    video = _video_fake()
    ctx = _ctx(
        {
            "deliverable": {"format": "16x9"},
            "ken_burns": {"enabled": True},
            "resolution": "1080p",
        }
    )
    out = await apply_ken_burns(ctx, [_still("a"), _still("b")], client=video)
    assert all(s["video_url"].startswith("https://") for s in out)
    # Still image kept as the degrade path.
    assert all(s["image_url"].startswith("https://minio.test/") for s in out)
    assert video.record_animation.await_count == 2
    kwargs = video.record_animation.await_args_list[0].kwargs
    assert kwargs["template_id"] == "ken_burns"
    assert kwargs["width"] == 1920 and kwargs["height"] == 1080
    assert kwargs["params"]["preset"] == "zoom_in"
    assert kwargs["params"]["fit"] == "contain"  # default: never crop an unchosen still
    assert kwargs["params"]["image"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_fit_is_in_media_id_cache_key(fake_storage, monkeypatch) -> None:
    """fit is folded into the content-address media_id, so the SAME source image
    at fit=cover vs fit=contain resolves to DIFFERENT cache keys — a contain clip
    is never served a previously-recorded cover render (the point of FIX D)."""
    seen: list[tuple] = []
    real = ken_burns.deterministic_media_id

    def _spy(*args: Any) -> str:
        seen.append(args)
        return real(*args)

    monkeypatch.setattr(ken_burns, "deterministic_media_id", _spy)
    video = _video_fake()
    await apply_ken_burns(
        _ctx({"ken_burns": {"enabled": True, "fit": "cover"}}), [_still("a")], client=video
    )
    await apply_ken_burns(
        _ctx({"ken_burns": {"enabled": True, "fit": "contain"}}), [_still("a")], client=video
    )
    assert "cover" in seen[0] and "contain" in seen[1]
    assert seen[0] != seen[1]  # differing fit ⇒ differing cache key


@pytest.mark.asyncio
async def test_skips_shots_that_already_have_video_with_absent_block(fake_storage) -> None:
    """The still-only ``candidates`` filter is unchanged on the new default-on
    path: real footage is never overwritten by a pan/zoom of its poster frame."""
    video = _video_fake()
    footage = {
        "id": "f",
        "line_id": "x",
        "video_url": "https://minio.test/clip.mp4",
        "image_url": "https://minio.test/x.png",
    }
    out = await apply_ken_burns(_ctx({}), [footage, _still("a")], client=video)
    assert out[0]["video_url"] == "https://minio.test/clip.mp4"
    assert out[1]["video_url"].startswith("https://minio.test/artifacts/")
    assert video.record_animation.await_count == 1


@pytest.mark.asyncio
async def test_skips_non_http_image(fake_storage) -> None:
    video = _video_fake()
    local = {"id": "l", "line_id": "y", "image_url": "file:///tmp/y.png"}
    proxy = {"id": "p", "line_id": "z", "image_url": "/api/runs/abc/artifact/z.png"}
    out = await apply_ken_burns(
        _ctx({"ken_burns": {"enabled": True}}), [local, proxy], client=video
    )
    assert out is not None
    assert all("video_url" not in s for s in out)
    video.record_animation.assert_not_awaited()


@pytest.mark.asyncio
async def test_render_error_soft_fails(fake_storage) -> None:
    video = _video_fake()
    video.record_animation.side_effect = RuntimeError("vps down")
    shots = [_still("a")]
    out = await apply_ken_burns(_ctx({"ken_burns": {"enabled": True}}), shots, client=video)
    # Shot keeps its still; compile proceeds.
    assert "video_url" not in out[0]


@pytest.mark.asyncio
async def test_custom_preset_and_duration(fake_storage) -> None:
    video = _video_fake()
    ctx = _ctx(
        {
            "deliverable": {"format": "9x16"},
            "ken_burns": {"enabled": True, "preset": "pan_left", "duration_seconds": 8},
            "resolution": "1080p",
        }
    )
    # No own ``seconds`` on the shot, so the config duration is what applies.
    await apply_ken_burns(ctx, [_still("a", seconds=None)], client=video)
    kwargs = video.record_animation.await_args.kwargs
    assert kwargs["params"]["preset"] == "pan_left"
    assert kwargs["params"]["duration_ms"] == 8000
    assert kwargs["duration_ms"] == 8000
    assert kwargs["width"] == 1080 and kwargs["height"] == 1920


@pytest.mark.asyncio
async def test_shot_duration_wins_over_config_duration(fake_storage) -> None:
    """THE DEFECT: ``duration_ms`` was ONE run-wide value, so a 12s narration shot
    got a 5s pan and 7s of ``_freeze_pad_suffix`` cloning the last frame — the very
    frozen-still artefact this pass exists to remove. The shot's own ``seconds``
    now drives the clip, per shot, in one gather."""
    video = _video_fake()
    ctx = _ctx({"ken_burns": {"enabled": True, "duration_seconds": 5}})
    await apply_ken_burns(
        ctx, [_still("long", seconds=12.0), _still("short", seconds=2.5)], client=video
    )
    # ⛔ KEYED BY SHOT, NOT BY CALL ORDER. These shots are recorded through
    # `stage_gather` at `_CONCURRENCY = 2`, so with two shots they genuinely race
    # and `await_args_list` is in COMPLETION order, which is scheduling and not
    # behaviour. (It flipped once a `to_thread` hop entered the path: green
    # locally, red in CI.) The film is unaffected — `stage_gather` assigns
    # `results[index]` and documents "Results in the same order as items" — so
    # asserting the sequence tested the event loop. Keying on `part_id` asserts
    # the property the docstring above actually claims: each shot's OWN seconds
    # drove ITS OWN clip.
    by_shot = {
        c.kwargs["part_id"]: c.kwargs["params"]["duration_ms"]
        for c in video.record_animation.await_args_list
    }
    assert by_shot == {"kb-s-long": 12000, "kb-s-short": 2500}
    # The recorder kwarg travels with the params so the recording is actually that
    # long — asserting only params would pass on a clip cut back to the default.
    assert {
        c.kwargs["part_id"]: c.kwargs["duration_ms"] for c in video.record_animation.await_args_list
    } == {"kb-s-long": 12000, "kb-s-short": 2500}


@pytest.mark.asyncio
async def test_shot_duration_is_clamped(fake_storage) -> None:
    """Bounded both ways: a sub-second beat is not worth a browser recording round
    trip, and a 10-minute still must not be recorded frame-by-frame into
    ``_RECORD_TIMEOUT_S``."""
    video = _video_fake()
    await apply_ken_burns(
        _ctx({}), [_still("tiny", seconds=0.2), _still("huge", seconds=600.0)], client=video
    )
    # Keyed by shot, not by call order — see the note in
    # `test_shot_duration_wins_over_config_duration`: these two race at
    # `_CONCURRENCY = 2`, so `await_args_list` is completion order, and the clamp
    # is a per-shot property anyway.
    by_shot = {
        c.kwargs["part_id"]: c.kwargs["duration_ms"] for c in video.record_animation.await_args_list
    }
    assert by_shot == {
        "kb-s-tiny": int(ken_burns._MIN_DURATION_S * 1000),
        "kb-s-huge": int(ken_burns._MAX_DURATION_S * 1000),
    }


@pytest.mark.asyncio
async def test_media_id_is_stable_across_presigned_signatures(fake_storage, monkeypatch) -> None:
    """THE DEFECT: the media id hashed the FULL presigned URL, whose
    ``X-Amz-Signature``/``X-Amz-Date`` are re-minted by ``refresh_shot_asset_urls``
    on every compile. The content-address therefore changed every run,
    ``find_existing_media`` never hit, and every clip was re-recorded and re-paid.
    Two different signatures of the SAME object must give the SAME id."""
    seen: list[str] = []
    real = ken_burns.deterministic_media_id

    def _spy(*args: Any) -> str:
        seen.append(real(*args))
        return seen[-1]

    monkeypatch.setattr(ken_burns, "deterministic_media_id", _spy)

    base = "https://minio.test/w2v-orchestrator/runs/r1/a.png"
    for query in ("X-Amz-Date=20260817T090000Z&X-Amz-Signature=aaa", "X-Amz-Signature=bbb"):
        shot = _still("a")
        shot["image_url"] = f"{base}?{query}"
        await apply_ken_burns(_ctx({}), [shot], client=_video_fake())

    assert len(seen) == 2
    assert seen[0] == seen[1]


@pytest.mark.asyncio
async def test_media_id_still_separates_distinct_objects(fake_storage, monkeypatch) -> None:
    """The query-strip must not collapse two DIFFERENT images onto one cache key —
    a fix that made every still reuse the first clip would also 'pass' the test
    above."""
    seen: list[str] = []
    real = ken_burns.deterministic_media_id

    def _spy(*args: Any) -> str:
        seen.append(real(*args))
        return seen[-1]

    monkeypatch.setattr(ken_burns, "deterministic_media_id", _spy)
    for name in ("a", "b"):
        shot = _still(name)
        shot["image_url"] = f"https://minio.test/w2v-orchestrator/{name}.png?X-Amz-Signature=z"
        await apply_ken_burns(_ctx({}), [shot], client=_video_fake())
    assert seen[0] != seen[1]


@pytest.mark.asyncio
async def test_record_carries_a_stable_idempotency_key(fake_storage) -> None:
    """``stage_gather`` runs with ``per_item_retries=1``: without a dedupe key a
    retry after a wait timeout starts a SECOND paid recording. The key must be
    present and identical across a re-sign of the same object (same construction as
    motion_graphics_gen / ai_video_gen)."""
    ctx = _ctx({})
    keys: list[str] = []
    for query in ("X-Amz-Signature=aaa", "X-Amz-Signature=bbb"):
        video = _video_fake()
        shot = _still("a")
        shot["image_url"] = f"https://minio.test/w2v-orchestrator/a.png?{query}"
        await apply_ken_burns(ctx, [shot], client=video)
        key = video.record_animation.await_args.kwargs["idempotency_key"]
        assert isinstance(key, str) and key
        keys.append(key)

    assert keys[0] == keys[1]
    assert keys[0].startswith(f"{ctx.run_id}:ken_burns:")


# ── QUEUE_SATURATED: re-submit instead of silently keeping a still ───────────

_SATURATED = RuntimeError(
    "video_production job kb-1 failed: QUEUE_SATURATED: no per-pod recording slot "
    "within 300s (per-pod cap 3)"
)


@pytest.fixture
def backoff_sleeps(monkeypatch) -> list[float]:
    """Record the saturation backoffs instead of sleeping through them."""
    slept: list[float] = []

    async def _no_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(ken_burns, "_saturation_sleep", _no_sleep)
    return slept


def _keys(video: AsyncMock) -> list[str]:
    return [call.kwargs["idempotency_key"] for call in video.record_animation.await_args_list]


@pytest.mark.asyncio
async def test_saturated_then_success_yields_a_clip_under_a_new_key(
    fake_storage, backoff_sleeps
) -> None:
    """video_production answers a repeated idempotency key with the ORIGINAL
    (failed) job, so the retry must go out under a DISTINCT key — and carry the
    identical request body, because the recorder replays on key + body hash."""
    video = _video_fake()
    video.wait_for_content_recording.side_effect = [_SATURATED, {"duration_seconds": 3.0}]
    ctx = _ctx({})
    notes: list[str] = []
    degradations: list[Any] = []
    out = await apply_ken_burns(
        ctx, [_still("a")], client=video, notes=notes, degradations=degradations
    )

    assert out[0]["video_url"].startswith("https://")
    first, second = _keys(video)
    assert first.startswith(f"{ctx.run_id}:ken_burns:") and ":sat" not in first
    assert second == f"{first}:sat1"
    bodies = [
        {k: v for k, v in call.kwargs.items() if k != "idempotency_key"}
        for call in video.record_animation.await_args_list
    ]
    assert bodies[0] == bodies[1]
    assert len(backoff_sleeps) == 1 and backoff_sleeps[0] >= ken_burns._SATURATION_BACKOFF_BASE_S
    # A shot that ended up animated is not a degradation.
    assert notes == [] and degradations == []


@pytest.mark.asyncio
async def test_saturated_past_the_retry_cap_keeps_the_still_with_the_saturation_code(
    fake_storage, backoff_sleeps
) -> None:
    video = _video_fake()
    video.wait_for_content_recording.side_effect = _SATURATED
    notes: list[str] = []
    degradations: list[Any] = []
    out = await apply_ken_burns(
        _ctx({}), [_still("a")], client=video, notes=notes, degradations=degradations
    )

    assert "video_url" not in out[0]
    keys = _keys(video)
    # 1 original + _SATURATION_RETRIES re-submissions, every one a distinct key,
    # in a deterministic order a successor pod would replay.
    assert len(keys) == ken_burns._SATURATION_RETRIES + 1 == len(set(keys))
    assert keys[1:] == [f"{keys[0]}:sat{n}" for n in range(1, ken_burns._SATURATION_RETRIES + 1)]
    assert len(backoff_sleeps) == ken_burns._SATURATION_RETRIES
    codes = [entry["code"] for entry in degradations]
    assert KEN_BURNS_STILL_QUEUE_SATURATED in codes
    assert KEN_BURNS_STILL_SOFT_FAILED not in codes
    # The prose note users read is still there, and is the coded entry's detail.
    assert any("s-a soft-failed, kept as a still" in note for note in notes)
    assert degradations[0]["detail"] == notes[0]


@pytest.mark.asyncio
async def test_a_non_saturation_failure_does_not_retry(fake_storage, backoff_sleeps) -> None:
    """Today's behaviour, exactly: one submit, the still stays, generic code."""
    video = _video_fake()
    video.wait_for_content_recording.side_effect = RuntimeError(
        "video_production job kb-1 failed: chromium crashed"
    )
    degradations: list[Any] = []
    out = await apply_ken_burns(_ctx({}), [_still("a")], client=video, degradations=degradations)

    assert "video_url" not in out[0]
    assert video.record_animation.await_count == 1
    assert backoff_sleeps == []
    # The shot's own code, then the pass-level "every still stayed a still".
    assert [entry["code"] for entry in degradations] == [
        KEN_BURNS_STILL_SOFT_FAILED,
        KEN_BURNS_ALL_STILLS_FROZEN,
    ]


@pytest.mark.asyncio
async def test_no_saturation_retry_when_the_remaining_budget_cannot_hold_one(
    fake_storage, backoff_sleeps
) -> None:
    """A retry that cannot finish before the pass deadline is never started. The
    floor is video_production's 300s queue wait PLUS the record floor: a budget
    just over that sum is under it once the backoff (>= 5s) is taken off, so
    no ``:sat1`` job is submitted to be paid for and abandoned."""
    assert ken_burns._SATURATION_RETRY_MIN_BUDGET_S == 330.0
    video = _video_fake()
    video.wait_for_content_recording.side_effect = _SATURATED
    degradations: list[Any] = []
    out = await apply_ken_burns(
        _ctx({}),
        [_still("a")],
        client=video,
        degradations=degradations,
        budget_seconds=ken_burns._SATURATION_RETRY_MIN_BUDGET_S + 1.0,
    )

    assert "video_url" not in out[0]
    assert video.record_animation.await_count == 1
    assert backoff_sleeps == []
    # The shot's own code, then the pass-level "every still stayed a still".
    assert [entry["code"] for entry in degradations] == [
        KEN_BURNS_STILL_QUEUE_SATURATED,
        KEN_BURNS_ALL_STILLS_FROZEN,
    ]


@pytest.mark.asyncio
async def test_an_error_merely_quoting_the_marker_does_not_retry(
    fake_storage, backoff_sleeps
) -> None:
    """Only the exact ``poll_job`` shape counts: ``... failed: QUEUE_SATURATED: ...``."""
    for text in (
        "video_production job kb-1 failed: chromium crashed (not QUEUE_SATURATED: a render bug)",
        "QUEUE_SATURATED: no per-pod recording slot within 300s",
        "video_production job kb-1 failed: QUEUE_SATURATED_ISH: something else",
    ):
        video = _video_fake()
        video.wait_for_content_recording.side_effect = RuntimeError(text)
        degradations: list[Any] = []
        out = await apply_ken_burns(
            _ctx({}), [_still("a")], client=video, degradations=degradations
        )
        assert "video_url" not in out[0]
        assert video.record_animation.await_count == 1, text
        assert degradations[0]["code"] == KEN_BURNS_STILL_SOFT_FAILED, text
    assert backoff_sleeps == []


@pytest.mark.asyncio
async def test_a_successor_pod_replays_the_same_keys_and_the_same_request(
    fake_storage, backoff_sleeps
) -> None:
    """Re-entry after a pod roll: attempt numbering restarts at 0, so the second
    invocation of the same shot sends ``base, :sat1, ...`` again, each with the
    byte-identical request — which is what makes video_production answer with
    the predecessor's jobs instead of starting new paid ones."""
    ctx = _ctx({})
    sent: list[list[dict[str, Any]]] = []
    for _pod in range(2):
        video = _video_fake()
        video.wait_for_content_recording.side_effect = _SATURATED
        await apply_ken_burns(ctx, [_still("a")], client=video)
        sent.append([dict(call.kwargs) for call in video.record_animation.await_args_list])

    first, second = sent
    base = first[0]["idempotency_key"]
    assert [call["idempotency_key"] for call in first] == [
        base,
        *(f"{base}:sat{n}" for n in range(1, ken_burns._SATURATION_RETRIES + 1)),
    ]
    # Every kwarg of every attempt — key, still bytes (data URI), part_id, dims.
    assert second == first
    assert first[0]["params"]["image"].startswith("data:image/")


@pytest.mark.asyncio
async def test_a_pass_deadline_timeout_after_saturation_is_coded_as_saturation(
    fake_storage, backoff_sleeps, monkeypatch
) -> None:
    """The re-submitted job outlives the pass deadline: the still was lost to the
    saturated queue, not to a slow render, and the code says so."""
    outcomes: list[Exception] = [_SATURATED, TimeoutError("job kb-1 still unfinished")]

    async def _wait(_client: Any, _job_id: str, _deadline: float) -> dict[str, object]:
        raise outcomes.pop(0)

    monkeypatch.setattr(ken_burns, "_wait_for_recording", _wait)
    video = _video_fake()
    notes: list[str] = []
    degradations: list[Any] = []
    out = await apply_ken_burns(
        _ctx({}), [_still("a")], client=video, notes=notes, degradations=degradations
    )

    assert "video_url" not in out[0]
    assert video.record_animation.await_count == 2 and outcomes == []
    assert degradations[0]["code"] == KEN_BURNS_STILL_QUEUE_SATURATED
    assert "after 1 QUEUE_SATURATED attempt(s)" in notes[0]


@pytest.mark.asyncio
async def test_a_pass_deadline_timeout_without_saturation_stays_a_plain_soft_fail(
    fake_storage, backoff_sleeps, monkeypatch
) -> None:
    async def _wait(_client: Any, _job_id: str, _deadline: float) -> dict[str, object]:
        raise TimeoutError("job kb-1 still unfinished")

    monkeypatch.setattr(ken_burns, "_wait_for_recording", _wait)
    video = _video_fake()
    degradations: list[Any] = []
    await apply_ken_burns(_ctx({}), [_still("a")], client=video, degradations=degradations)
    assert video.record_animation.await_count == 1
    assert degradations[0]["code"] == KEN_BURNS_STILL_SOFT_FAILED


@pytest.mark.asyncio
async def test_a_700s_budget_buys_one_saturation_retry_not_two(backoff_sleeps, monkeypatch) -> None:
    """Each saturated attempt burns video_production's 300s queue wait. After the
    first, ~400s is left (>= 330s floor): retry. After the second, ~90s is left:
    a ``:sat2`` job could not even leave the queue, so it is never submitted."""
    clock = [1000.0]
    monkeypatch.setattr(ken_burns, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    keys: list[str] = []

    async def _submit(key: str) -> str:
        keys.append(key)
        return "kb-1"

    async def _wait(_client: Any, _job_id: str, _deadline: float) -> dict[str, object]:
        clock[0] += ken_burns._VP_QUEUE_WAIT_TIMEOUT_S
        raise _SATURATED

    monkeypatch.setattr(ken_burns, "_wait_for_recording", _wait)
    with pytest.raises(RuntimeError, match="QUEUE_SATURATED"):
        await ken_burns._record_with_saturation_retry(
            _submit, AsyncMock(), base_key="k", deadline=clock[0] + 700.0, shot_id="s-a"
        )
    assert keys == ["k", "k:sat1"]
    assert len(backoff_sleeps) == 1


def test_saturation_backoff_is_exponential_with_bounded_jitter() -> None:
    base = ken_burns._SATURATION_BACKOFF_BASE_S
    for attempt in range(1, ken_burns._SATURATION_RETRIES + 1):
        floor = base * 2 ** (attempt - 1)
        assert floor <= ken_burns._saturation_backoff_seconds(attempt) <= floor + base


@pytest.fixture
def _stub_dns(monkeypatch):
    """Offline, deterministic resolution — ``safe_get`` resolves every host."""
    hosts = {"upload.wikimedia.org": "185.15.59.240", "minio.test": "93.184.216.34"}

    def fake_getaddrinfo(host, port, *args, **kwargs):
        addr = hosts.get(host)
        if addr is None:
            raise OSError(f"stubbed DNS: unknown host {host!r}")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (addr, port or 0))]

    monkeypatch.setattr(safe_fetch.socket, "getaddrinfo", fake_getaddrinfo)


@pytest.mark.asyncio
async def test_the_shot_fetch_sends_the_identifying_user_agent(_stub_dns) -> None:
    """``_fetch_own_asset`` sent no User-Agent, and was judged benign only because
    ``republish_image`` is *expected* to have rewritten the asset to MinIO first — an
    invariant asserted NOWHERE. This module's own docstring says the opposite: the URL
    is "a value carried on a shot dict — the same dict real-media / operator-pinned URLs
    flow through", which is precisely why it goes through ``safe_get``. So the UA is
    sent, and this pins it ON THE WIRE with a handler that 403s the default exactly as
    Wikimedia's policy does.
    """
    seen: list[str] = []

    def wikimedia(request: httpx.Request) -> httpx.Response:
        ua = request.headers.get("user-agent", "")
        seen.append(ua)
        if ua.startswith("python-httpx") or not ua:
            return httpx.Response(403, headers={"content-type": "text/plain"}, content=b"no")
        return httpx.Response(200, headers={"content-type": "image/png"}, content=b"\x89PNG\r\n")

    async with httpx.AsyncClient(transport=httpx.MockTransport(wikimedia)) as client:
        data, content_type = await _REAL_FETCH_OWN_ASSET(
            "https://upload.wikimedia.org/wikipedia/commons/5/50/Teams.png", client
        )

    assert data == b"\x89PNG\r\n"
    assert content_type == "image/png"
    assert seen == [MEDIA_FETCH_USER_AGENT]


@pytest.mark.asyncio
async def test_the_ua_is_sent_on_the_minio_path_too(_stub_dns) -> None:
    """The happy path is our own presigned artifact. A presigned SigV4 URL signs ``Host``
    and its signed-headers list, never ``User-Agent``, so adding the header must not
    change the MinIO fetch — and the header must be there, not conditional on the host."""
    seen: list[str] = []

    def minio(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("user-agent", ""))
        return httpx.Response(200, headers={"content-type": "image/png"}, content=b"\x89PNG\r\n")

    async with httpx.AsyncClient(transport=httpx.MockTransport(minio)) as client:
        data, _ct = await _REAL_FETCH_OWN_ASSET(
            "https://minio.test/w2v-orchestrator/a.png?X-Amz-Signature=aaa", client
        )

    assert data == b"\x89PNG\r\n"
    assert seen == [MEDIA_FETCH_USER_AGENT]


# ── observability: budget received, per-shot wait + outcome, timeout not re-paid ─


@pytest.mark.asyncio
async def test_a_plain_timeout_is_never_resubmitted_even_with_budget_left(
    backoff_sleeps, monkeypatch, caplog
) -> None:
    """A timed-out job is still in flight on video_production and will most likely
    complete: a re-submit under a distinct key would pay for a SECOND recording
    of the same clip. Even with most of the pass budget left, it is logged and
    raised — never re-submitted."""
    caplog.set_level("INFO", logger=ken_burns.__name__)
    clock = [1000.0]
    monkeypatch.setattr(ken_burns, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    keys: list[str] = []

    async def _submit(key: str) -> str:
        keys.append(key)
        return "kb-7"

    async def _wait(_client: Any, _job_id: str, _deadline: float) -> dict[str, object]:
        raise TimeoutError("job kb-7 still unfinished")

    monkeypatch.setattr(ken_burns, "_wait_for_recording", _wait)
    with pytest.raises(TimeoutError):
        await ken_burns._record_with_saturation_retry(
            _submit, AsyncMock(), base_key="k", deadline=clock[0] + 900.0, shot_id="s-a"
        )
    assert keys == ["k"]
    assert backoff_sleeps == []
    [line] = [r.getMessage() for r in caplog.records if "NOT re-submitting" in r.getMessage()]
    assert "shot s-a job kb-7 (key k)" in line


@pytest.mark.asyncio
async def test_the_pass_logs_its_budget_and_each_shots_wait_and_outcome(
    fake_storage, monkeypatch, caplog
) -> None:
    caplog.set_level("INFO", logger=ken_burns.__name__)
    video = _video_fake()
    video.wait_for_content_recording.side_effect = [
        {"duration_seconds": 3.0},
        RuntimeError("video_production job kb-1 failed: render crashed"),
    ]
    await apply_ken_burns(_ctx({}), [_still("a"), _still("b")], client=video, budget_seconds=1000.0)

    messages = [r.getMessage() for r in caplog.records]
    [start] = [m for m in messages if "pass start" in m]
    assert "budget 1000s (offered 1000s), 2 still(s)" in start
    outcomes = sorted(m for m in messages if " outcome=" in m)
    assert len(outcomes) == 2, outcomes
    assert any("outcome=recorded waited=" in m for m in outcomes)
    assert any("outcome=soft_failed:RuntimeError waited=" in m for m in outcomes)
    assert all("started_at=" in m for m in outcomes)
