"""Ken Burns: record at the run's master cadence (O13), and hand the recorder a
still that is already at frame resolution (O12).

O13 — ``fps`` was hardcoded to 30 while the compiler conforms the master to
whatever ``_resolve_master_cadence`` picks. The clip reaches the compiler with no
``Shot.target_fps``, so ``_cadence_fragment`` emits a bare decimating ``fps=N``
and cannot interpolate: a 30fps continuous pan into a 24fps master loses 6 of
every 30 frames.

O12 — the largest upscale in the still path was performed by Chromium's default
smoothing (a 1536 still against a 5120 raster), not by any filter we chose. The
measured win from pre-upscaling with lanczos is recorded on ``_prescale_still``.
"""

from __future__ import annotations

import base64
import io
import math
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from PIL import Image

from src.services import tts_storage
from src.services.runs.run_context import RunContext
from src.services.stages.shared import ken_burns
from src.services.stages.shared.ken_burns import apply_ken_burns


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


def _png(w: int, h: int, *, alpha: bool = False) -> bytes:
    """A deterministic, genuinely detailed test still (flat colour is unresampleable)."""
    mode = "RGBA" if alpha else "RGB"
    img = Image.new(mode, (w, h))
    px = img.load()
    assert px is not None
    for y in range(h):
        for x in range(w):
            v = 255 if (x // 3 + y // 3) % 2 else 0
            px[x, y] = (v, (x * 7) % 256, (y * 11) % 256) + ((128,) if alpha else ())
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _sine_png(w: int, h: int) -> bytes:
    """A mid-frequency sine grating — detailed but band-limited.

    Deliberately NOT the hard checkerboard ``_png`` builds: on a step edge
    lanczos rings, and a JPEG pass then smooths that ringing back toward the
    bilinear result, so a filter comparison on it measures the encoder rather
    than the resampler.
    """
    img = Image.new("RGB", (w, h))
    px = img.load()
    assert px is not None
    for y in range(h):
        for x in range(w):
            v = int(127 + 120 * math.sin(x / 2.5) * math.cos(y / 3.5))
            px[x, y] = (v, (v + 40) % 256, 255 - v)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def _storage():
    tts_storage.set_storage_for_tests(_FakeStorage())
    yield
    tts_storage.set_storage_for_tests(None)


@pytest.fixture
def _fetch_returns(monkeypatch):
    """Install the bytes ``_fetch_own_asset`` should hand the recording path."""

    def _install(data: bytes, content_type: str = "image/png") -> None:
        async def _fake(url: str, client: Any) -> tuple[bytes, str]:
            return (data, content_type)

        monkeypatch.setattr(ken_burns, "_fetch_own_asset", _fake)

    return _install


def _video_fake() -> AsyncMock:
    v = AsyncMock()
    v.record_animation.return_value = "kb-1"
    v.wait_for_content_recording.return_value = {"duration_seconds": 3.0}
    v.get_recording_video.return_value = (b"\x00mp4", "video/mp4", None)
    return v


def _ctx(snapshot: dict[str, Any]) -> RunContext:
    return RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot=snapshot)


def _still() -> dict[str, Any]:
    return {"id": "s-a", "line_id": "a", "image_url": "https://minio.test/a.png", "seconds": 3.0}


# --------------------------------------------------------------------------- O13


@pytest.mark.asyncio
async def test_no_cadence_block_records_at_30(_fetch_returns) -> None:
    """Presence-is-opt-in: an un-opted run keeps the pre-existing 30fps exactly."""
    _fetch_returns(_png(64, 36))
    video = _video_fake()
    await apply_ken_burns(_ctx({}), [_still()], client=video)
    assert video.record_animation.await_args.kwargs["fps"] == 30


@pytest.mark.asyncio
@pytest.mark.parametrize("target", [24])
async def test_records_at_the_runs_master_cadence(_fetch_returns, target: int) -> None:
    """A 24p master must be fed 24fps clips, or the compiler decimates the pan."""
    _fetch_returns(_png(64, 36))
    video = _video_fake()
    await apply_ken_burns(_ctx({"cadence": {"target_fps": target}}), [_still()], client=video)
    assert video.record_animation.await_args.kwargs["fps"] == target


@pytest.mark.asyncio
async def test_master_fps_argument_wins_over_the_config(_fetch_returns) -> None:
    """``action_reel``'s master rate comes from its FOOTAGE, not its config — the
    caller must be able to override, or that product silently records at 30."""
    _fetch_returns(_png(64, 36))
    video = _video_fake()
    await apply_ken_burns(
        _ctx({"cadence": {"target_fps": 24}}), [_still()], client=video, master_fps=50
    )
    assert video.record_animation.await_args.kwargs["fps"] == 50


@pytest.mark.asyncio
async def test_media_id_folds_fps_only_when_it_differs_from_30(_fetch_returns, monkeypatch) -> None:
    """A non-default cadence changes the bytes ⇒ must change the content address.
    A 30fps run produces byte-identical clips to the pre-O13 code ⇒ must NOT, or
    every cached clip in the estate is invalidated and re-paid for nothing."""
    _fetch_returns(_png(64, 36))
    seen: list[tuple[Any, ...]] = []
    real = ken_burns.deterministic_media_id

    def _spy(*args: Any) -> str:
        seen.append(args)
        return real(*args)

    monkeypatch.setattr(ken_burns, "deterministic_media_id", _spy)
    for snapshot in ({}, {"cadence": {"target_fps": 30}}, {"cadence": {"target_fps": 24}}):
        await apply_ken_burns(_ctx(snapshot), [_still()], client=_video_fake())

    none_block, thirty, twentyfour = seen
    assert none_block == thirty, "a 30fps cadence must not invalidate cached clips"
    assert twentyfour != thirty, "a 24fps cadence must re-address the clip"
    assert "fps24" in twentyfour and "fps30" not in thirty


# --------------------------------------------------------------------------- O12


def _uri_image(video: AsyncMock) -> Image.Image:
    uri = video.record_animation.await_args.kwargs["params"]["image"]
    head, _, b64 = uri.partition(",")
    assert head.startswith("data:image/"), head
    return Image.open(io.BytesIO(base64.b64decode(b64)))


@pytest.mark.asyncio
async def test_still_smaller_than_the_frame_is_upscaled_to_frame_dims(_fetch_returns) -> None:
    """The recorder must receive frame-resolution pixels, so Chromium's default
    smoothing never performs the biggest upscale in this path."""
    _fetch_returns(_png(384, 216))
    video = _video_fake()
    await apply_ken_burns(_ctx({"resolution": "1080p"}), [_still()], client=video)
    assert _uri_image(video).size == (1920, 1080)


@pytest.mark.asyncio
async def test_still_already_at_frame_size_is_passed_through_untouched(_fetch_returns) -> None:
    """The killed ``_zoompan_prescale`` measured WORSE because it prescaled an
    already-frame-sized input. Never re-encode what does not need it."""
    data = _png(1920, 1080)
    _fetch_returns(data)
    video = _video_fake()
    await apply_ken_burns(_ctx({"resolution": "1080p"}), [_still()], client=video)
    uri = video.record_animation.await_args.kwargs["params"]["image"]
    assert base64.b64decode(uri.partition(",")[2]) == data


def test_the_ceiling_cannot_regress_the_old_absolute_one() -> None:
    """P1 stated STRUCTURALLY, not sampled.

    The two cases above are witnesses; this is the property. The predicate is a
    single comparison on ``out_px`` that does not mention the target, so it is the
    same bound at every tier — and as long as it sits at or above the 5120x2880
    ceiling it replaced, EVERY geometry that ceiling admitted is still admitted,
    whatever its aspect and whatever the target. Lowering the constant below that
    reintroduces the regression on a tier this file happens not to sample.
    """
    assert ken_burns._MAX_UPSCALE_ABS_PIXELS >= 5120 * 2880


# --------------------------------------------------------------------- O12b
# The prescale target is the raster the RECORDER REPORTS (capture_width /
# capture_height, video_production_service #182), not an assumed supersample
# factor. Unreported must behave exactly as O12 did.


@pytest.fixture(autouse=True)
def _clear_observed_raster() -> Any:
    """The observed-raster cache is process-lifetime, so it would otherwise leak
    between tests and make this module order-dependent."""
    ken_burns._OBSERVED_CAPTURE_RASTER.clear()
    yield
    ken_burns._OBSERVED_CAPTURE_RASTER.clear()


def _sent_image_size(video: AsyncMock, call: int = -1) -> tuple[int, int]:
    """Decode the still actually handed to the recorder on a given call."""
    uri = video.record_animation.await_args_list[call].kwargs["params"]["image"]
    blob = base64.b64decode(uri.split(",", 1)[1])
    with Image.open(io.BytesIO(blob)) as im:
        return im.size


def _reporting_video_fake(capture: dict[str, Any]) -> AsyncMock:
    v = _video_fake()
    v.wait_for_content_recording.return_value = {"duration_seconds": 3.0, **capture}
    return v


@pytest.mark.asyncio
async def test_unreported_raster_prescales_to_the_ordered_dims(_fetch_returns) -> None:
    """NULL capture_width is the contract's "not reported" — behaviour must be
    exactly O12's: prescale to the ordered frame, nothing else."""
    _fetch_returns(_png(384, 216))
    video = _video_fake()  # wait_for_content_recording reports no capture raster
    await apply_ken_burns(_ctx({}), [_still()], client=video)
    kwargs = video.record_animation.await_args.kwargs
    assert _sent_image_size(video) == (kwargs["width"], kwargs["height"])


@pytest.mark.asyncio
async def test_a_reported_raster_drives_the_prescale(_fetch_returns) -> None:
    """The whole point: once the recorder has declared its capture raster, the
    still is sized to THAT, so Chromium never smooths it up to the surface."""
    _fetch_returns(_png(384, 216))
    video = _reporting_video_fake({"capture_width": 5120, "capture_height": 2880})

    await apply_ken_burns(_ctx({}), [_still()], client=video)
    ordered = video.record_animation.await_args.kwargs
    # First shot of a cold process: nothing observed yet ⇒ the ordered dims.
    assert _sent_image_size(video) == (ordered["width"], ordered["height"])

    # The observation from that recording now drives the next one.
    await apply_ken_burns(_ctx({}), [_still()], client=video)
    assert _sent_image_size(video) == (5120, 2880)


@pytest.mark.asyncio
async def test_a_clamped_down_raster_is_not_an_upscale(_fetch_returns) -> None:
    """``capture_width == width`` is how the contract reports "supersampling was
    clamped down" (a 4k order). It must resolve to the ordered dims, NOT be
    mistaken for a raster to upscale to."""
    _fetch_returns(_png(384, 216))
    video = _video_fake()
    probe = await apply_ken_burns(_ctx({}), [_still()], client=video)
    assert probe  # the stage produced something
    ordered = video.record_animation.await_args.kwargs
    w, h = ordered["width"], ordered["height"]

    video = _reporting_video_fake({"capture_width": w, "capture_height": h})
    await apply_ken_burns(_ctx({}), [_still()], client=video)
    await apply_ken_burns(_ctx({}), [_still()], client=video)
    assert _sent_image_size(video) == (w, h)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "capture",
    [{"capture_width": None, "capture_height": None}],
    ids=["null"],
)
async def test_an_unusable_report_is_discarded(_fetch_returns, capture: dict[str, Any]) -> None:
    """Anything that is not a usable pair of pixel dimensions must leave the
    prescale on its pre-feature path rather than resize to a bogus target."""
    _fetch_returns(_png(384, 216))
    video = _reporting_video_fake(capture)
    notes: list[str] = []
    await apply_ken_burns(_ctx({}), [_still()], client=video, notes=notes)
    await apply_ken_burns(_ctx({}), [_still()], client=video, notes=notes)
    kwargs = video.record_animation.await_args.kwargs
    assert _sent_image_size(video) == (kwargs["width"], kwargs["height"])
    assert ken_burns._OBSERVED_CAPTURE_RASTER == {}
    # DISCARDED, not "blew up on the way past". Reading an unusable report must
    # not raise: the soft-fail in ``_gen`` would swallow it, the shot would lose
    # its motion, and the cache would be empty for that reason instead of this
    # one — which is exactly what the two assertions above would still accept.
    assert notes == [], f"an unusable capture report cost the shot its clip: {notes}"


@pytest.mark.asyncio
async def test_the_observed_raster_never_reaches_the_media_id(_fetch_returns, monkeypatch) -> None:
    """⛔ THE CONTENT ADDRESS IS THE ORDER, AND ONLY THE ORDER.

    The observed raster is the one input in this path that is NOT a property of
    the order: it depends on whether a recording at this order has completed in
    THIS process yet. Folding it — even conditionally on "it differs from the
    order" — makes the same shot of the same run address differently depending on
    pod warmth, so every warm process re-addresses every clip the estate recorded
    under O12. ``find_existing_media`` has a tenant-scoped CROSS-RUN lane, so that
    is the tenant's whole cached ken_burns estate re-recorded to re-sharpen work
    already delivered — the exact cost the O12 comment weighed and declined.

    So: cold, warm, supersampled and clamped must all produce the SAME media id
    arguments, byte-for-byte identical to the pre-O12b address.
    """
    _fetch_returns(_png(384, 216))
    seen: list[tuple[Any, ...]] = []
    real = ken_burns.deterministic_media_id

    def _spy(*args: Any) -> str:
        seen.append(args)
        return real(*args)

    monkeypatch.setattr(ken_burns, "deterministic_media_id", _spy)

    video = _video_fake()
    await apply_ken_burns(_ctx({}), [_still()], client=video)
    w, h = (
        video.record_animation.await_args.kwargs["width"],
        (video.record_animation.await_args.kwargs["height"]),
    )

    ken_burns._OBSERVED_CAPTURE_RASTER.clear()
    supersampled = _reporting_video_fake({"capture_width": 5120, "capture_height": 2880})
    await apply_ken_burns(_ctx({}), [_still()], client=supersampled)  # cold
    await apply_ken_burns(_ctx({}), [_still()], client=supersampled)  # warm

    ken_burns._OBSERVED_CAPTURE_RASTER.clear()
    clamped = _reporting_video_fake({"capture_width": w, "capture_height": h})
    await apply_ken_burns(_ctx({}), [_still()], client=clamped)
    await apply_ken_burns(_ctx({}), [_still()], client=clamped)

    unreported, ss_cold, ss_warm, cl_cold, cl_warm = seen
    # The WARM supersampled call is the one that matters: it is the only call whose
    # still was prescaled to something other than the ordered dims, so it is the only
    # one an observation could leak into.
    assert _sent_image_size(supersampled) == (5120, 2880), (
        "the warm call did not actually prescale to the observed raster, so this "
        "test would pass without exercising anything"
    )
    assert ss_warm == ss_cold == unreported, (
        "pod warmth changed the content address — every warm process now "
        "re-addresses the estate's O12 clips and re-pays for the re-record"
    )
    assert cl_cold == cl_warm == unreported
    # RELATIVE equality alone is satisfiable by folding the SAME extra component
    # into every call, so state the absolute claim too: no media id in this path
    # may carry a raster component at all.
    for tup in seen:
        assert not any("raster" in str(part) for part in tup), (
            f"an order-keyed clip must address exactly as it did pre-O12b: {tup}"
        )
