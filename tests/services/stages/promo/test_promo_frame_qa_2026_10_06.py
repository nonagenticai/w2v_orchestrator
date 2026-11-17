"""Regression tests for the defects frame QA found in promo run 01a10fcf (2026-10-06).

1. the planned holds never reached the film: 43.5s of a 60s order, 0-0.56s gaps;
2. the kinetic clips ordered for lines 2 and 3 were rendered and never used;
3. a still whose Ken Burns recording failed ``QUEUE_SATURATED`` shipped frozen;
4. cards opened empty and the CTA's text was up for ~1.4s;
5. AI stills had a pillarbox seam; the caption box was striped per word;
6. "merqi.nl" was spoken without its dot and "robots.txt" broke in two.

The fixtures are the run's real script lines and measured narration lengths
(shared with ``test_promo_frame_qa_2026_10_05``: the two runs used one script).
"""

from __future__ import annotations

import asyncio
import io
import shutil
import subprocess
import sys
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from PIL import Image, ImageChops, ImageDraw, ImageStat

from src.config import get_settings
from src.services.stages.primitives import compile_stage
from src.services.stages.promo import fill
from src.services.stages.promo.narrate import (
    PromoNarrateStage,
    brand_domains,
    narration_language,
    spoken_text,
)
from src.services.stages.promo.render import (
    END_CARD_SETTLED_LEAD_SECONDS,
    PromoRenderStage,
    _plan_visuals,
)
from src.services.stages.shared import ken_burns_local
from src.services.stages.shared.degradation import (
    CAPTION_KARAOKE_OFF,
    KEN_BURNS_LOCAL_FALLBACK,
    KEN_BURNS_LOCAL_FALLBACK_FAILED,
    KEN_BURNS_NEAR_STATIC,
    KEN_BURNS_STILL_QUEUE_SATURATED,
    MOTION_CLIP_UNUSED,
    SFX_CUE_DROPPED,
    Degradation,
    advisory,
    degradation,
)
from src.services.stages.website_review.compile import WebsiteReviewCompileStage
from tests.services.stages.promo.test_promo_frame_qa_2026_10_05 import (
    _LINES,
    _ctx,
    _promo_cfg,
    _promo_prior,
    _render,
)

_NARRATED = sum(seconds for _i, _t, seconds in _LINES)


# ------------------------------------------------------------------ 1. holds


def test_every_held_shot_declares_the_ceiling_the_compiler_needs() -> None:
    """video_compiler honours a narrated shot's ``seconds`` beyond its narration
    ONLY when the shot also carries ``max_seconds`` (renderer
    ``_narrated_picture_floor``); a hold without it is cut back to the audio."""
    shots = PromoRenderStage().build_shots(_ctx(_promo_cfg(), _promo_prior()))
    assert len(shots) == len(_LINES)
    for shot, (_line_id, _text, narrated) in zip(shots, _LINES, strict=True):
        assert shot["seconds"] == pytest.approx(narrated + 2.0, abs=0.01)
        assert shot["max_seconds"] >= shot["seconds"]


@pytest.mark.asyncio
async def test_the_compile_request_fills_the_order_inside_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _out, captured, _vp = await _render(monkeypatch, _promo_cfg(), 57.4)
    sent = captured["shots"]
    narrated = [s for s in sent if s["line_id"] != "__title_outro__"]
    assert all(s.get("max_seconds", 0) >= s["seconds"] for s in narrated)
    total = sum(float(s["seconds"]) for s in sent)
    assert 60 * 0.9 <= total <= 60 * 1.1
    assert total > _NARRATED + 8.0  # more than narration + end card: the holds are in


def test_a_hold_never_lowers_a_ceiling_the_shot_already_had() -> None:
    held = fill.apply_holds([{"seconds": 2.0, "max_seconds": 9.0}], 1.0)
    assert held == [{"seconds": 3.0, "max_seconds": 9.0}]


def test_no_hold_leaves_the_shots_untouched() -> None:
    shots = [{"seconds": 3.0}, {"seconds": 4.0}]
    assert fill.apply_holds(shots, 0.0) == shots


# ----------------------------------------------------------- 2. kinetic clips


def _story_run_01a10fcf() -> list[dict[str, Any]]:
    """The run's storyboard typing: slide, ai_image x4, slide, slide."""
    kinds = ["slide", "ai_image", "ai_image", "ai_image", "ai_image", "slide", "slide"]
    return [
        {"id": f"s-{i}", "line_id": i, "seconds": s, "visual_type": kind}
        for (i, _t, s), kind in zip(_LINES, kinds, strict=True)
    ]


def _plan(**overrides: Any) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    kwargs: dict[str, Any] = {
        "slide_by_line": {i: f"https://m/slide-{i}.png" for i, _t, _s in _LINES},
        "footage_primary": [],
        "footage_cutaways": [],
        "html_urls": [],
        "ai_urls": [],
        "ai_image_by_line": {i: f"https://m/ai-{i}.png" for i in ("l2", "l3", "l4", "l5")},
        "motion_clip_by_line": {i: f"https://m/mg-{i}.mp4" for i in ("l1", "l2", "l3")},
        "motion_clip_urls": [],
    }
    return _plan_visuals(_story_run_01a10fcf(), **{**kwargs, **overrides})


def test_a_keyed_kinetic_clip_takes_its_line_from_a_generated_still() -> None:
    image, video, _fit = _plan()
    assert video == {i: f"https://m/mg-{i}.mp4" for i in ("l1", "l2", "l3")}
    # Lines nobody ordered motion for keep their generated still.
    assert image["l4"] == "https://m/ai-l4.png" and "l4" not in video


def test_an_unkeyed_pool_clip_never_displaces_a_generated_still() -> None:
    _image, video, _fit = _plan(
        motion_clip_by_line={}, motion_clip_urls=["https://m/pool-1.mp4", "https://m/pool-2.mp4"]
    )
    assert not {"l2", "l3", "l4", "l5"} & set(video)


def test_a_rendered_clip_the_film_does_not_carry_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior = _promo_prior()
    prior["motion_graphics_gen"] = {
        "clips": [
            {"line_id": "l1", "status": "completed", "video_url": "https://m/mg-l1.mp4"},
            {"line_id": "l2", "status": "completed", "video_url": "https://m/mg-l2.mp4"},
        ]
    }
    stage = PromoRenderStage()
    ctx = _ctx(_promo_cfg(), prior)
    # Both keyed clips are laid by the real plan: nothing to record.
    assert stage._unused_motion_clip_advisories(ctx) == []
    # Something outranked line 2's clip (footage / curated media): recorded, by line.
    laid = [{"line_id": "l1", "video_url": "https://m/mg-l1.mp4"}, {"line_id": "l2"}]
    monkeypatch.setattr(stage, "build_shots", lambda _ctx: laid)
    (unused,) = stage._unused_motion_clip_advisories(ctx)
    assert unused["code"] == MOTION_CLIP_UNUSED and "line l2" in unused["detail"]


# ------------------------------------------------- 4. cards read from the cut


def _prior_with_animated_slides() -> dict[str, Any]:
    prior = _promo_prior()
    prior["promo_slides"] = {
        "slides": [
            {
                "line_id": i,
                "image_url": f"https://m/slide-{i}.png",
                "motion_video_url": f"https://m/slide-{i}.mp4",
            }
            for i, _t, _s in _LINES
        ]
    }
    return prior


def test_a_slide_line_shows_its_text_from_the_first_frame() -> None:
    """The animated slide builds its text in from an EMPTY frame under narration
    that starts at the cut; the still carries the same text from frame 0."""
    shots = PromoRenderStage().build_shots(_ctx(_promo_cfg(), _prior_with_animated_slides()))
    for shot in shots:
        assert shot.get("image_url") == f"https://m/slide-{shot['line_id']}.png"
        assert not shot.get("video_url")


def test_the_cta_line_stays_up_long_enough_to_read_even_with_no_hold() -> None:
    cfg = _promo_cfg()
    cfg["brief"]["target_duration_seconds"] = 38  # on target: hold is 0
    ctx = _ctx(cfg, _promo_prior())
    assert fill.promo_fill(ctx).hold_seconds == 0.0
    shots = PromoRenderStage().build_shots(ctx)
    assert shots[-1]["line_id"] == "l7"
    assert shots[-1]["seconds"] >= fill.CTA_MIN_SECONDS >= 2.5
    assert shots[-1]["max_seconds"] >= shots[-1]["seconds"]
    # Only the CTA is stretched; the plan counts that stretch.
    assert shots[0]["seconds"] == pytest.approx(_LINES[0][2], abs=0.01)
    assert fill.promo_fill(ctx).cta_floor_seconds == pytest.approx(2.5 - _LINES[-1][2], abs=0.01)


@pytest.mark.asyncio
async def test_the_end_card_starts_after_its_build_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """The card is recorded 2.8s longer and the shot starts 2.8s in, so the
    headline (650-1500ms), subtitle and brand mark (2000-2800ms) are up at the cut."""
    _out, captured, title_vp = await _render(monkeypatch, _promo_cfg(), 57.4)
    last = captured["shots"][-1]
    assert last["line_id"] == "__title_outro__"
    assert last["seconds"] == pytest.approx(8.0)
    assert last["clip_lead_seconds"] == pytest.approx(END_CARD_SETTLED_LEAD_SECONDS)
    assert END_CARD_SETTLED_LEAD_SECONDS >= 2.8
    recorded_ms = title_vp.record_animation.await_args.kwargs["duration_ms"]
    assert recorded_ms == 8000 + round(END_CARD_SETTLED_LEAD_SECONDS * 1000)


# ----------------------------------------------- 5. pillarbox seam, caption box


def _png(width: int, height: int) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (width, height), (10, 40, 80)).save(out, format="PNG")
    return out.getvalue()


def test_a_16_10_still_is_cropped_to_a_native_16_9_one() -> None:
    cropped = ken_burns_local.cover_crop_still(_png(1536, 960), 3840, 2160)
    assert cropped is not None
    assert Image.open(io.BytesIO(cropped)).size == (1536, 864)


def test_a_still_already_at_the_frame_aspect_is_left_alone() -> None:
    assert ken_burns_local.cover_crop_still(_png(1920, 1080), 3840, 2160) is None
    assert ken_burns_local.cover_crop_still(b"not an image", 3840, 2160) is None


@pytest.fixture
def _local_io(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Fake the fetch and the store of the local still passes; record the stores."""
    stored: dict[str, Any] = {"images": [], "videos": []}

    async def _fetch(url: str, _client: object) -> tuple[bytes, str]:
        return _png(1536, 960), "image/png"

    async def _store_image(**kw: Any) -> str:
        stored["images"].append(kw)
        return f"https://minio.test/i/{kw['image_id']}.png"

    async def _store_video(**kw: Any) -> str:
        stored["videos"].append(kw)
        return f"https://minio.test/v/{kw['video_id']}.mp4"

    monkeypatch.setattr(ken_burns_local.kb, "_fetch_own_asset", _fetch)
    monkeypatch.setattr(ken_burns_local, "store_generated_image", _store_image)
    monkeypatch.setattr(ken_burns_local, "store_generated_video", _store_video)
    return stored


@pytest.mark.asyncio
async def test_only_cover_fit_stills_are_cropped_before_ken_burns(
    _local_io: dict[str, Any],
) -> None:
    shots: list[dict[str, Any]] = [
        {"id": "a", "image_url": "https://m/ai.png", "fit": "cover", "seconds": 6.0},
        {"id": "b", "image_url": "https://m/slide.png", "seconds": 4.0},
        {"id": "c", "image_url": "https://m/x.png", "video_url": "https://m/x.mp4", "fit": "cover"},
    ]
    ctx = _ctx(_promo_cfg(resolution="4k"), _promo_prior())
    out = await ken_burns_local.crop_cover_stills_to_frame(ctx, shots)
    assert out[0]["image_url"].startswith("https://minio.test/i/")
    assert out[1:] == shots[1:]
    (stored,) = _local_io["images"]
    assert Image.open(io.BytesIO(stored["image_bytes"])).size == (1536, 864)


@pytest.mark.asyncio
async def test_every_product_crops_its_cover_stills_before_ken_burns(
    _local_io: dict[str, Any],
) -> None:
    """Ken Burns is local for every product (2026-10-20), so every product crops."""
    ctx = _ctx(_promo_cfg(), _promo_prior())
    for stage in (PromoRenderStage(), WebsiteReviewCompileStage()):
        shots: list[dict[str, Any]] = [{"id": "a", "image_url": "https://m/ai.png", "fit": "cover"}]
        assert await stage._prepare_local_stills(ctx, shots, []) == {0}
        assert shots[0]["image_url"].startswith("https://minio.test/i/")


@pytest.mark.asyncio
async def test_a_promo_caption_line_is_one_run_so_it_gets_one_box(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The compiler draws its caption box once per karaoke run (per word) and the
    overlapping semi-transparent boxes stripe; a promo sends no karaoke runs."""
    monkeypatch.setattr(get_settings().pipeline, "captions_karaoke", True)
    ctx = _ctx(_promo_cfg(), _promo_prior())
    assert WebsiteReviewCompileStage().captions_karaoke(ctx) is True
    assert PromoRenderStage().captions_karaoke(ctx) is False
    out, captured, _vp = await _render(
        monkeypatch, _promo_cfg(deliverable={"format": "16x9", "captions_mode": "burned_in"}), 57.4
    )
    assert captured["captions_karaoke"] is False
    # The trade (no word highlight) is on the record, without degrading the run.
    assert CAPTION_KARAOKE_OFF in [a["code"] for a in out.advisories]
    assert CAPTION_KARAOKE_OFF not in [d["code"] for d in out.degradations]
    monkeypatch.setattr(get_settings().pipeline, "captions_karaoke", False)
    out, _captured, _vp = await _render(
        monkeypatch, _promo_cfg(deliverable={"format": "16x9", "captions_mode": "burned_in"}), 57.4
    )
    assert CAPTION_KARAOKE_OFF not in [a["code"] for a in out.advisories]


# ------------------------------------------------------- 3. never a frozen still

_SATURATED = (
    "ken_burns: shot s-l3 soft-failed, kept as a still: RuntimeError: video_production job "
    "j failed: QUEUE_SATURATED: no per-pod recording slot within 300s (per-pod cap 3)"
)


def _after_ken_burns() -> list[dict[str, Any]]:
    """Shots as ``apply_ken_burns`` returned them on the run: line 3 still frozen."""
    return [
        {"id": "s-l2", "image_url": "https://m/l2.png", "video_url": "https://m/kb-l2.mp4"},
        {"id": "s-l3", "image_url": "https://m/l3.png", "seconds": 7.98, "fit": "cover"},
    ]


@pytest.mark.asyncio
async def test_a_still_the_recorder_failed_is_animated_locally(
    monkeypatch: pytest.MonkeyPatch, _local_io: dict[str, Any]
) -> None:
    render = AsyncMock(return_value=b"\x00mp4")
    monkeypatch.setattr(ken_burns_local, "render_local_ken_burns", render)
    notes = [_SATURATED, "something else"]
    advisories: list[Degradation] = []
    # The coded twins of those notes, as ``apply_ken_burns`` records them.
    degradations = [
        degradation(KEN_BURNS_STILL_QUEUE_SATURATED, _SATURATED),
        degradation(SFX_CUE_DROPPED, "something else"),
    ]
    out = await ken_burns_local.rescue_frozen_stills(
        _ctx(_promo_cfg(resolution="4k"), _promo_prior()),
        _after_ken_burns(),
        notes=notes,
        advisories=advisories,
        budget_seconds=600.0,
        degradations=degradations,
    )
    # Withdrawn WITH its note: a shot that now moves must not stay coded frozen.
    assert [d["code"] for d in degradations] == [SFX_CUE_DROPPED]
    assert out[0]["video_url"] == "https://m/kb-l2.mp4"  # a recorded clip is kept
    assert out[1]["video_url"].startswith("https://minio.test/v/")
    kwargs = render.await_args_list[0].kwargs
    assert (kwargs["width"], kwargs["height"]) == (3840, 2160)
    assert kwargs["duration_s"] == pytest.approx(7.98)  # the shot's own length, hold included
    assert notes == ["something else"]  # no longer frozen: the soft-fail note is withdrawn
    assert [a["code"] for a in advisories] == [KEN_BURNS_LOCAL_FALLBACK]
    assert "QUEUE_SATURATED" in advisories[0]["detail"]


@pytest.mark.asyncio
async def test_a_failed_local_render_keeps_the_still_and_says_so(
    monkeypatch: pytest.MonkeyPatch, _local_io: dict[str, Any]
) -> None:
    monkeypatch.setattr(
        ken_burns_local,
        "render_local_ken_burns",
        AsyncMock(side_effect=ken_burns_local.LocalKenBurnsError("ffmpeg exited 1")),
    )
    notes = [_SATURATED]
    advisories: list[Degradation] = []
    degradations = [degradation(KEN_BURNS_STILL_QUEUE_SATURATED, _SATURATED)]
    out = await ken_burns_local.rescue_frozen_stills(
        _ctx(_promo_cfg(), _promo_prior()),
        _after_ken_burns(),
        notes=notes,
        advisories=advisories,
        budget_seconds=600.0,
        degradations=degradations,
    )
    assert [d["code"] for d in degradations] == [
        KEN_BURNS_STILL_QUEUE_SATURATED,
        KEN_BURNS_LOCAL_FALLBACK_FAILED,
    ]
    assert degradations[1]["detail"] == notes[1]
    assert "video_url" not in out[1]
    assert notes[0] == _SATURATED and "stays frozen" in notes[1]
    assert advisories == []


@pytest.mark.asyncio
async def test_ken_burns_opted_out_animates_nothing(
    monkeypatch: pytest.MonkeyPatch, _local_io: dict[str, Any]
) -> None:
    render = AsyncMock(return_value=b"\x00mp4")
    monkeypatch.setattr(ken_burns_local, "render_local_ken_burns", render)
    shots = _after_ken_burns()
    out = await ken_burns_local.rescue_frozen_stills(
        _ctx(_promo_cfg(ken_burns=False), _promo_prior()), shots, budget_seconds=600.0
    )
    assert out == shots
    render.assert_not_awaited()


@pytest.mark.asyncio
async def test_every_product_renders_its_stills_locally_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local-only for every product: the rescue renders every still (local_first),
    and the recorder-clip near-static probe never runs — there are no recorder clips."""
    rescued = [{"id": "s-l3", "video_url": "https://minio.test/v/local.mp4"}]
    rescue = AsyncMock(return_value=rescued)
    flag = AsyncMock()
    monkeypatch.setattr(compile_stage, "rescue_frozen_stills", rescue)
    monkeypatch.setattr(compile_stage, "flag_near_static_clips", flag)
    ctx = _ctx(_promo_cfg(), _promo_prior())
    for stage in (PromoRenderStage(), WebsiteReviewCompileStage()):
        rescue.reset_mock()
        out = await stage._rescue_local_stills(
            ctx, _after_ken_burns(), {0, 1}, notes=[], advisories=[], budget_seconds=321.0
        )
        assert out == rescued
        kwargs = rescue.await_args_list[0].kwargs
        assert kwargs["budget_seconds"] == 321.0
        assert kwargs["local_first"] is True
    flag.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_near_static_clip_is_recorded_and_a_moving_one_is_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scores = {"https://m/flat.mp4": 38.0, "https://m/moving.mp4": 21.0}

    async def _psnr(url: str, *, timeout_s: float) -> float:
        return scores[url]

    monkeypatch.setattr(ken_burns_local, "clip_motion_psnr", _psnr)
    shots: list[dict[str, Any]] = [
        {"id": "flat", "video_url": "https://m/flat.mp4"},
        {"id": "moving", "video_url": "https://m/moving.mp4"},
        {"id": "footage", "video_url": "https://m/never-probed.mp4"},
        {"id": "frozen", "image_url": "https://m/x.png"},
    ]
    advisories: list[Degradation] = []
    await ken_burns_local.flag_near_static_clips(shots, {0, 1, 3}, advisories=advisories)
    assert [a["code"] for a in advisories] == [KEN_BURNS_NEAR_STATIC]
    assert "flat" in advisories[0]["detail"] and "38.0 dB" in advisories[0]["detail"]


def test_psnr_separates_identical_frames_from_different_ones() -> None:
    assert ken_burns_local.psnr(b"\x10" * 64, b"\x10" * 64) == float("inf")
    assert ken_burns_local.psnr(b"\x00" * 64, b"\xff" * 64) == pytest.approx(0.0)


_FFMPEG = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="needs ffmpeg (in the service image)"
)
_W, _H = 640, 360
_GREEN = (0, 255, 0)


def _marker_still() -> bytes:
    """A 16:9 still: a red square dead centre, a green band on all four edges
    (a designed slide whose content runs to its safe margin)."""
    still = Image.new("RGB", (1280, 720), (20, 30, 60))
    draw = ImageDraw.Draw(still)
    draw.rectangle((600, 320, 679, 399), fill=(255, 0, 0))
    for box in ((0, 0, 1279, 7), (0, 712, 1279, 719), (0, 0, 7, 719), (1272, 0, 1279, 719)):
        draw.rectangle(box, fill=_GREEN)
    out = io.BytesIO()
    still.save(out, format="PNG")
    return out.getvalue()


def _frame(video: Any, index: int) -> Image.Image:
    raw = subprocess.run(  # noqa: S603 -- fixed argv, test-local file
        [
            str(shutil.which("ffmpeg")),
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vf",
            f"select=eq(n\\,{index})",
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    ).stdout
    return Image.frombytes("RGB", (_W, _H), raw)


def _red_marker(frame: Image.Image) -> tuple[float, float, int]:
    """(centre x, centre y, width) of the red square."""
    pixels = cast(Any, frame.load())
    hits = [
        (x, y)
        for y in range(_H)
        for x in range(_W)
        if pixels[x, y][0] > 180 and pixels[x, y][1] < 90 and pixels[x, y][2] < 90
    ]
    xs, ys = [x for x, _ in hits], [y for _, y in hits]
    return sum(xs) / len(xs), sum(ys) / len(ys), max(xs) - min(xs) + 1


def _edges_showing_green(frame: Image.Image) -> list[bool]:
    """Whether the green band is inside the frame at the left, top, right, bottom."""
    pixels = cast(Any, frame.load())

    def green(x: int, y: int) -> bool:
        r, g, _b = pixels[x, y]
        return g > 150 and r < 120

    return [
        any(green(x, _H // 2) for x in range(24)),
        any(green(_W // 2, y) for y in range(24)),
        any(green(x, _H // 2) for x in range(_W - 24, _W)),
        any(green(_W // 2, y) for y in range(_H - 24, _H)),
    ]


async def _local_clip(tmp_path: Any, fit: str) -> list[Image.Image]:
    video = await ken_burns_local.render_local_ken_burns(
        _marker_still(), width=_W, height=_H, fps=30, duration_s=2.0, timeout_s=120.0, fit=fit
    )
    clip = tmp_path / f"{fit}.mp4"
    clip.write_bytes(video)
    return [_frame(clip, index) for index in (0, 30, 59)]


@_FFMPEG
@pytest.mark.asyncio
async def test_the_cover_push_in_zooms_about_the_frame_centre(tmp_path: Any) -> None:
    """The first version cropped at 0,0: the centre marker drifted from (639,360)
    to (715,403) at 1280x720 and the right/bottom edges lost ~10%."""
    first, middle, last = (_red_marker(f) for f in await _local_clip(tmp_path, "cover"))
    for cx, cy, _width in (first, middle, last):
        assert abs(cx - (_W - 1) / 2) <= 2.0 and abs(cy - (_H - 1) / 2) <= 2.0
    # ...and it really zooms: the marker grows by about LOCAL_ZOOM_TRAVEL.
    assert first[2] < middle[2] < last[2]
    assert last[2] / first[2] == pytest.approx(1 + ken_burns_local.LOCAL_ZOOM_TRAVEL, abs=0.04)


@_FFMPEG
@pytest.mark.asyncio
async def test_a_designed_slide_is_never_cropped_by_the_local_move(tmp_path: Any) -> None:
    """A slide's text runs to its safe margin, so the recorder keeps slides on
    ``contain``; the local fallback must too. All four edge bands stay inside the
    frame on every frame — while a cover push-in has cut them by the middle."""
    frames = await _local_clip(tmp_path, "contain")
    assert [_edges_showing_green(f) for f in frames] == [[True] * 4] * 3
    first, middle, last = (_red_marker(f) for f in frames)
    for cx, cy, _width in (first, middle, last):
        assert abs(cx - (_W - 1) / 2) <= 2.0 and abs(cy - (_H - 1) / 2) <= 2.0
    assert first[2] < middle[2] < last[2]  # still a move, not a frozen frame
    cover = await _local_clip(tmp_path, "cover")
    assert _edges_showing_green(cover[1]) == [False] * 4


def _edge_band_width(frame: Image.Image) -> list[int]:
    """How far (px) the green band reaches IN from the left, top, right, bottom
    edge, counted from the edge pixel itself (0 = the edge pixel is not green)."""
    pixels = cast(Any, frame.load())

    def green(x: int, y: int) -> bool:
        r, g, _b = pixels[x, y]
        return g > 150 and r < 120

    def run(points: list[tuple[int, int]]) -> int:
        count = 0
        for x, y in points:
            if not green(x, y):
                break
            count += 1
        return count

    return [
        run([(x, _H // 3) for x in range(24)]),
        run([(_W // 3, y) for y in range(24)]),
        run([(_W - 1 - x, _H // 3) for x in range(24)]),
        run([(_W // 3, _H - 1 - y) for y in range(24)]),
    ]


@_FFMPEG
@pytest.mark.asyncio
async def test_a_slide_with_the_frame_aspect_fills_the_frame_with_no_ring(tmp_path: Any) -> None:
    """Frame QA of run 01a11082: the ``contain`` move drew every 16:9 slide inset
    at 94% over a blurred copy — a darker ring around the slide. A slide that
    already has the frame's aspect is shown full-frame on EVERY frame: its four
    edge bands sit at the frame edge (the 8 px band of the 1280 px still is 4 px
    at 640 px), the picture never changes scale, and it still moves (a brightness
    breath), so it is not a frozen still."""
    frames = await _local_clip(tmp_path, "full")
    for frame in frames:
        assert all(3 <= width <= 6 for width in _edge_band_width(frame)), _edge_band_width(frame)
    markers = [_red_marker(f) for f in frames]
    assert len({width for _x, _y, width in markers}) == 1  # no zoom at all
    for cx, cy, _width in markers:
        assert abs(cx - (_W - 1) / 2) <= 2.0 and abs(cy - (_H - 1) / 2) <= 2.0
    diff = ImageChops.difference(frames[0], frames[1]).convert("L")
    assert ImageStat.Stat(diff).mean[0] > 2.0  # the breath is there
    # The inset move it replaces: the band does NOT reach the edge at the start.
    inset = await _local_clip(tmp_path, "contain")
    assert 0 in _edge_band_width(inset[0])


@_FFMPEG
@pytest.mark.asyncio
async def test_ken_burns_cover_moves_must_not_reveal_borders(tmp_path: Any) -> None:
    """The cover move only ever zooms IN about the centre (1.0 -> 1 + travel), so
    its source window always lies inside the still and no frame shows anything
    from outside it. Outside its source ``perspective`` repeats the edge pixels,
    which would WIDEN the still's 4 px edge band; it never does, and the ring is
    never black either."""
    for frame in await _local_clip(tmp_path, "cover"):
        assert max(_edge_band_width(frame)) <= 5, _edge_band_width(frame)
        pixels = cast(Any, frame.load())
        ring = [
            *((x, y) for x in range(_W) for y in (0, _H - 1)),
            *((x, y) for y in range(_H) for x in (0, _W - 1)),
        ]
        darkest = min(sum(pixels[x, y]) for x, y in ring)
        assert darkest > 40, darkest


@pytest.mark.parametrize(
    ("size", "fit", "expected"),
    [((1920, 1080), None, "full")],
)
def test_which_local_move_a_still_takes(
    size: tuple[int, int], fit: str | None, expected: str
) -> None:
    """``full`` is the per-line-motion move (``full_frame``); without the flag a
    slide takes the ``contain`` move whatever its aspect, as it always did."""
    shot = {"id": "s", "image_url": "https://m/s.png", **({"fit": fit} if fit else {})}
    old = "cover" if fit == "cover" else "contain"
    assert ken_burns_local.local_fit(shot, _png(*size), 3840, 2160, full_frame=True) == expected
    assert ken_burns_local.local_fit(shot, b"not an image", 3840, 2160, full_frame=True) == old
    assert ken_burns_local.local_fit(shot, _png(*size), 3840, 2160) == old


@pytest.mark.asyncio
async def test_slides_take_the_contain_move_and_ai_stills_the_cover_one(
    monkeypatch: pytest.MonkeyPatch, _local_io: dict[str, Any]
) -> None:
    render = AsyncMock(return_value=b"\x00mp4")
    monkeypatch.setattr(ken_burns_local, "render_local_ken_burns", render)
    shots: list[dict[str, Any]] = [
        {"id": "cta-slide", "image_url": "https://m/slide-l7.png", "seconds": 4.1},
        {"id": "ai", "image_url": "https://m/l3.png", "seconds": 8.0, "fit": "cover"},
    ]
    notes = [
        "ken_burns: shot cta-slide soft-failed, kept as a still: RuntimeError: QUEUE_SATURATED",
        "ken_burns: shot ai soft-failed, kept as a still: RuntimeError: QUEUE_SATURATED",
        "ken_burns: ALL 2 still shot(s) soft-failed — no motion was applied and the video "
        "is a sequence of frozen stills (~92% duplicate frames)",
    ]
    advisories: list[Degradation] = []
    out = await ken_burns_local.rescue_frozen_stills(
        _ctx(_promo_cfg(), _promo_prior()),
        shots,
        notes=notes,
        advisories=advisories,
        budget_seconds=600.0,
    )
    # The fake fetch returns a 16:10 still: the slide takes the inset move.
    assert [call.kwargs["fit"] for call in render.await_args_list] == ["contain", "cover"]
    assert all(shot["video_url"] for shot in out)
    assert out[0]["video_url"] != out[1]["video_url"]
    # Nothing is frozen any more: neither the per-shot notes nor the "ALL ... frozen
    # stills" verdict of the recorder pass may survive the rescue.
    assert notes == []
    assert "contain-safe" in advisories[0]["detail"] and "cover-safe" in advisories[1]["detail"]


def test_the_local_clip_is_rendered_at_1080p_at_most() -> None:
    """One 4K render peaked at 2.67 GB against the pod's 4Gi limit."""
    assert ken_burns_local.local_render_dims(3840, 2160) == (1920, 1080)
    assert ken_burns_local.local_render_dims(2160, 3840) == (1080, 1920)
    assert ken_burns_local.local_render_dims(1280, 720) == (1280, 720)


@pytest.mark.asyncio
async def test_a_4k_order_is_rendered_capped_and_on_two_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[list[str]] = []

    async def _fake_run(cmd: list[str], _timeout_s: float) -> bytes:
        seen.append(cmd)
        raise ken_burns_local.LocalKenBurnsError("stop before reading the output")

    monkeypatch.setattr(ken_burns_local, "_run", _fake_run)
    with pytest.raises(ken_burns_local.LocalKenBurnsError):
        await ken_burns_local.render_local_ken_burns(
            b"png", width=3840, height=2160, fps=30, duration_s=6.0, timeout_s=10.0
        )
    (cmd,) = seen
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "crop=1920:1080" in graph and "3840" not in graph
    assert cmd[cmd.index("-threads") + 1] == "2"
    assert cmd[cmd.index("-filter_complex_threads") + 1] == "2"


@pytest.mark.asyncio
async def test_a_cancelled_render_kills_its_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    started = asyncio.Event()
    procs: list[Any] = []
    real = asyncio.create_subprocess_exec

    async def _spawn(*_cmd: str, **kwargs: Any) -> Any:
        proc = await real(sys.executable, "-c", "import time; time.sleep(60)", **kwargs)
        procs.append(proc)
        started.set()
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _spawn)
    task = asyncio.create_task(ken_burns_local._run(["ffmpeg"], 60.0))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert procs[0].returncode is not None  # killed and reaped, not orphaned


# ------------------------------------------------------------ 6. pronunciation


_MERQI = ("merqi.nl",)


def test_the_brand_domain_is_spoken_with_its_dot() -> None:
    assert spoken_text("Request access at merqi.nl", domains=_MERQI) == (
        "Request access at merqi dot N L"
    )
    assert spoken_text("Go to www.merqi.nl.", domains=_MERQI) == "Go to W W W dot merqi dot N L."
    assert spoken_text("Open App.Merqi.nl now", domains=_MERQI) == "Open App dot Merqi dot N L now"


def test_a_multi_label_tld_is_spoken_whole() -> None:
    assert spoken_text("See merqi.co.uk today", domains=("merqi.co.uk",)) == (
        "See merqi dot C O dot U K today"
    )
    assert spoken_text("See example.com.", domains=("example.com",)) == "See example dot com."


def test_only_the_brands_own_domain_is_rewritten() -> None:
    """The generic ``word.suffix`` rule turned "daily.no setup" into a domain and
    "merqi.co.uk" into "merqi dot C O.uk"."""
    for text in (
        "It works with daily.no setup.",
        "See merqi.co.uk today",  # not this run's domain
        "Sold in the U.S. e.g. at 3.14 each.",
        "Done.Next",
        "Mail hello@merqi.nl or open https://merqi.nl/pricing",
        "Try other.nl instead",
    ):
        assert spoken_text(text, domains=_MERQI) == text
    assert spoken_text("Request access at merqi.nl") == "Request access at merqi.nl"


def test_a_known_file_name_is_not_broken_into_two_sentences() -> None:
    line = _LINES[4][1]
    assert "robots.txt" in line
    assert spoken_text(line) == line.replace("robots.txt", "robots dot T X T")


def test_the_dot_is_said_in_the_narration_language() -> None:
    assert spoken_text("Vraag toegang aan op merqi.nl", domains=_MERQI, language="nl") == (
        "Vraag toegang aan op merqi punt N L"
    )
    assert narration_language({"voice": {"language": "nl-NL"}}) == "nl"
    assert narration_language({"brief": {}}) == "en"


def test_the_brand_domain_comes_from_the_briefs_own_url() -> None:
    assert brand_domains({"brief": {"intent": {"url": "https://www.merqi.nl/x?y=1"}}}) == _MERQI
    assert brand_domains({"intent": {"url": "merqi.nl"}, "brief": {"url": "https://merqi.nl"}}) == (
        _MERQI
    )
    assert brand_domains({"brief": {"intent": {"url": "not a url"}}}) == ()


def test_the_stage_speaks_this_runs_cta_and_leaves_the_script_alone() -> None:
    cfg = _promo_cfg()
    cfg["brief"] = {
        **cfg["brief"],
        "intent": {"url": "https://merqi.nl"},
        "pronunciations": {"merqi": "MER-kee"},
    }
    item = {"text": "Request access at merqi.nl"}
    assert PromoNarrateStage().synth_text_for(item, _ctx(cfg, _promo_prior())) == (
        "Request access at MER-kee dot N L",
        False,
    )
    # The caption source is untouched: the stage rewrites what it SENDS, not the script.
    assert item["text"] == "Request access at merqi.nl"
    # A whole-name respelling wins over the domain rule.
    cfg["brief"]["pronunciations"] = {"merqi.nl": "murky dot enn ell"}
    assert PromoNarrateStage().synth_text_for(item, _ctx(cfg, _promo_prior()))[0] == (
        "Request access at murky dot enn ell"
    )


# ------------------------------------------------ the records reach the output


@pytest.mark.asyncio
async def test_the_records_of_both_passes_are_persisted_on_the_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real PromoRenderStage.run: what the pre-submit and the plan recorded is
    on the stage output's non-degrading ``advisories``."""
    unused = advisory(MOTION_CLIP_UNUSED, "motion clip keyed to line l2 was rendered but unused")

    async def _flag(_shots: object, _was_still: object, *, advisories: list[Degradation]) -> None:
        advisories.append(advisory(KEN_BURNS_NEAR_STATIC, "shot s-l4: clip is near-static"))

    monkeypatch.setattr(compile_stage, "flag_near_static_clips", _flag)
    # The near-static probe only runs on the recorder path, which a product must
    # now opt back into (the base is local-only).
    monkeypatch.setattr(PromoRenderStage, "ken_burns_local_only", lambda _self, _ctx: False)
    monkeypatch.setattr(
        PromoRenderStage, "_unused_motion_clip_advisories", lambda _self, _ctx: [unused]
    )
    out, _captured, _vp = await _render(monkeypatch, _promo_cfg(), 57.4)
    codes = [a["code"] for a in out.advisories]
    assert MOTION_CLIP_UNUSED in codes and KEN_BURNS_NEAR_STATIC in codes
    assert not [d for d in out.degradations if d["code"] in codes]
