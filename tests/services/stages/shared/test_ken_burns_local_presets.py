"""``ken_burns.preset`` in the LOCAL renderer (``shared/ken_burns_local``).

#1127 made every compile stage animate its stills locally, and the local move
was a centred zoom-in only: ``zoom_out`` / ``pan_{left,right,up,down}`` were read
into ``KenBurnsConfig.preset`` and then ignored. These pin the filter each preset
builds, that the stage pass hands the preset to the renderer, that an unknown
preset is recorded (never silent), and — where ffmpeg is installed — what each
preset really does to the picture.
"""

from __future__ import annotations

import io
import re
import shutil
import subprocess
from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from PIL import Image, ImageDraw

from src.services.runs.run_context import RunContext
from src.services.stages.shared import ken_burns_local
from src.services.stages.shared.degradation import KEN_BURNS_PRESET_UNKNOWN, Degradation
from src.services.stages.shared.motion_slides import format_dims, quality_tier
from src.services.tts_storage import deterministic_media_id

_FRAMES = 60
_PANS = ("pan_left", "pan_right", "pan_up", "pan_down")
_ALL = ("zoom_in", "zoom_out", *_PANS)


def _ctx(config: dict[str, Any]) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"deliverable": {"format": "16x9"}, **config},
        prior_run_outputs={},
    )


# ---------------------------------------------------------------------------
# The filter expression of each preset
# ---------------------------------------------------------------------------
def _corners(graph: str) -> dict[str, str]:
    match = re.search(r"perspective=(.*?):sense=source", graph)
    assert match, graph
    return dict(
        (key, value.strip("'"))
        for key, value in (part.split("=", 1) for part in match.group(1).split(":"))
    )


def _window(graph: str, on: int, w: int = 1920, h: int = 1080) -> tuple[float, ...]:
    """(left, top, right, bottom) of the source window at output frame ``on``.

    The corner expressions are plain arithmetic in ``W``/``H``/``on``, so they are
    evaluated here exactly as ffmpeg would.
    """
    c = _corners(graph)
    env = {"W": w, "H": h, "on": on, "__builtins__": {}}
    # x0/y0 = top-left, x3/y3 = bottom-right; the window is axis-aligned.
    assert c["x0"] == c["x2"] and c["x1"] == c["x3"] and c["y0"] == c["y1"] and c["y2"] == c["y3"]
    return cast(
        tuple[float, ...],
        tuple(eval(c[k], env) for k in ("x0", "y0", "x1", "y2")),  # noqa: S307 -- test-local expression
    )


def _graph(fit: str, preset: str) -> str:
    return ken_burns_local.local_ken_burns_filter(1920, 1080, 30, _FRAMES, fit, preset)


def test_the_local_presets_are_the_recorder_presets() -> None:
    assert set(_ALL) == ken_burns_local.LOCAL_PRESETS
    assert ken_burns_local.DEFAULT_PRESET == "zoom_in"


@pytest.mark.parametrize("fit", ["cover"])
def test_zoom_in_is_byte_identical_to_the_filter_before_presets(fit: str) -> None:
    """Default and explicit ``zoom_in`` are the pre-preset filter, string for string."""
    before = ken_burns_local.local_ken_burns_filter(1920, 1080, 30, _FRAMES, fit)
    assert _graph(fit, "zoom_in") == before
    assert f"(1-1/(1+{ken_burns_local.LOCAL_ZOOM_TRAVEL}*on/59))/2" in _graph("cover", "zoom_in")


@pytest.mark.parametrize("fit", ["cover"])
def test_zoom_in_and_zoom_out_are_mirror_images(fit: str) -> None:
    zin, zout = _graph(fit, "zoom_in"), _graph(fit, "zoom_out")
    assert "(1-on/59)" in zout and "(1-on/59)" not in zin
    assert _window(zout, 0) == pytest.approx(_window(zin, _FRAMES - 1))
    assert _window(zout, _FRAMES - 1) == pytest.approx(_window(zin, 0))
    # zoom_out: the window GROWS (the picture recedes), centred throughout.
    first, last = _window(zout, 0), _window(zout, _FRAMES - 1)
    assert (first[2] - first[0]) < (last[2] - last[0])
    for left, top, right, bottom in (first, last):
        assert (left + right) / 2 == pytest.approx(960) and (top + bottom) / 2 == pytest.approx(540)


@pytest.mark.parametrize(
    ("preset", "axis", "start_side"),
    [("pan_left", 0, -1)],
)
def test_a_cover_pan_holds_its_zoom_and_drifts_back_to_centre_inside_the_still(
    preset: str, axis: int, start_side: int
) -> None:
    graph = _graph("cover", preset)
    zoom = 1 + ken_burns_local.LOCAL_ZOOM_TRAVEL
    centre = (960.0, 540.0)
    size = (1920.0, 1080.0)
    for on in range(_FRAMES):
        left, top, right, bottom = _window(graph, on)
        assert right - left == pytest.approx(1920 / zoom)  # held zoom: no scale change
        assert bottom - top == pytest.approx(1080 / zoom)
        assert left >= -1e-6 and top >= -1e-6  # never past the still's edge
        assert right <= 1920 + 1e-6 and bottom <= 1080 + 1e-6
    first, last = _window(graph, 0), _window(graph, _FRAMES - 1)
    first_centre = ((first[0] + first[2]) / 2, (first[1] + first[3]) / 2)
    last_centre = ((last[0] + last[2]) / 2, (last[1] + last[3]) / 2)
    assert last_centre == pytest.approx(centre)
    # It starts flush with one edge of the still (the whole slack), on the named side.
    # (The expressions carry 6 decimals: a hundredth of a pixel at 1080p.)
    assert (first_centre[axis] - centre[axis]) * start_side == pytest.approx(
        size[axis] * (0.5 - 0.5 / zoom), abs=0.01
    )
    assert first_centre[1 - axis] == pytest.approx(centre[1 - axis])


@pytest.mark.parametrize("preset", ["pan_left"])
def test_a_contain_pan_never_cuts_the_inset_slide(preset: str) -> None:
    """The slide is inset at ``1/(1+travel)`` of the composite; on every frame the
    window must contain all of it — and the pan must actually move."""
    graph = _graph("contain", preset)
    inset_w = ken_burns_local._even(1920 / (1 + ken_burns_local.LOCAL_CONTAIN_TRAVEL))
    inset_h = ken_burns_local._even(1080 / (1 + ken_burns_local.LOCAL_CONTAIN_TRAVEL))
    travel = min(1920 / inset_w, 1080 / inset_h) - 1.0
    slide = (
        960 - 960 / (1 + travel),
        540 - 540 / (1 + travel),
        960 + 960 / (1 + travel),
        540 + 540 / (1 + travel),
    )
    for on in range(_FRAMES):
        left, top, right, bottom = _window(graph, on)
        assert left <= slide[0] + 1e-6 and top <= slide[1] + 1e-6
        assert right >= slide[2] - 1e-6 and bottom >= slide[3] - 1e-6
        assert left >= -1e-6 and right <= 1920 + 1e-6 and top >= -1e-6 and bottom <= 1080 + 1e-6
    assert _window(graph, 0) != pytest.approx(_window(graph, _FRAMES - 1))


def test_an_unknown_preset_in_the_filter_is_the_zoom_in() -> None:
    assert _graph("cover", "spin_wildly") == _graph("cover", "zoom_in")


@pytest.mark.parametrize("preset", ["zoom_in"])
def test_the_full_move_has_no_geometry_for_a_preset_to_steer(preset: str) -> None:
    assert _graph("full", preset) == _graph("full", "zoom_in")
    assert "perspective" not in _graph("full", preset)


# ---------------------------------------------------------------------------
# The stage pass threads the preset to the renderer
# ---------------------------------------------------------------------------
def _png(width: int, height: int) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (width, height), (10, 40, 80)).save(out, format="PNG")
    return out.getvalue()


@pytest.fixture
def _local_io(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    stored: dict[str, Any] = {"videos": []}

    async def _fetch(url: str, _client: object) -> tuple[bytes, str]:
        return _png(1536, 960), "image/png"

    async def _store_video(**kw: Any) -> str:
        stored["videos"].append(kw)
        return f"https://minio.test/v/{kw['video_id']}.mp4"

    monkeypatch.setattr(ken_burns_local.kb, "_fetch_own_asset", _fetch)
    monkeypatch.setattr(ken_burns_local, "store_generated_video", _store_video)
    return stored


def _stills() -> list[dict[str, Any]]:
    return [
        {"id": "ai", "image_url": "https://m/ai.png", "seconds": 4.0, "fit": "cover"},
        {"id": "slide", "image_url": "https://m/slide.png", "seconds": 3.0},
    ]


async def _rescue(
    config: dict[str, Any], notes: list[str], degradations: list[Degradation]
) -> list[Any]:
    out = await ken_burns_local.rescue_frozen_stills(
        _ctx(config),
        cast(Any, _stills()),
        notes=notes,
        advisories=[],
        budget_seconds=600.0,
        degradations=degradations,
        local_first=True,
    )
    return list(out)


@pytest.mark.asyncio
@pytest.mark.parametrize("preset", ["zoom_in"])
async def test_the_stage_pass_hands_the_authored_preset_to_the_renderer(
    monkeypatch: pytest.MonkeyPatch, _local_io: dict[str, Any], preset: str
) -> None:
    """On #1127's code ``render_local_ken_burns`` got no ``preset`` at all."""
    render = AsyncMock(return_value=b"\x00mp4")
    monkeypatch.setattr(ken_burns_local, "render_local_ken_burns", render)
    notes: list[str] = []
    degradations: list[Degradation] = []
    out = await _rescue({"ken_burns": {"preset": preset}}, notes, degradations)
    assert [call.kwargs["preset"] for call in render.await_args_list] == [preset, preset]
    assert all(shot["video_url"] for shot in out)
    assert notes == [] and degradations == []


@pytest.mark.asyncio
async def test_a_brief_authored_preset_reaches_the_renderer(
    monkeypatch: pytest.MonkeyPatch, _local_io: dict[str, Any]
) -> None:
    render = AsyncMock(return_value=b"\x00mp4")
    monkeypatch.setattr(ken_burns_local, "render_local_ken_burns", render)
    await _rescue({"brief": {"ken_burns": {"preset": "pan_up"}}}, [], [])
    assert {call.kwargs["preset"] for call in render.await_args_list} == {"pan_up"}


@pytest.mark.asyncio
async def test_a_preset_is_part_of_the_clip_address_but_zoom_in_keeps_its_old_one(
    monkeypatch: pytest.MonkeyPatch, _local_io: dict[str, Any]
) -> None:
    """A pan must never be served the cached zoom-in clip; a zoom-in resumed
    across this deploy must find the clip it already rendered."""
    monkeypatch.setattr(
        ken_burns_local, "render_local_ken_burns", AsyncMock(return_value=b"\x00mp4")
    )
    await _rescue({}, [], [])
    await _rescue({"ken_burns": {"preset": "pan_left"}}, [], [])
    ids = [video["video_id"] for video in _local_io["videos"]]
    zoom_in_ids, pan_ids = ids[:2], ids[2:]
    content_key = ken_burns_local.kb._content_key("https://m/ai.png")
    snapshot = _ctx({}).config_snapshot
    width, height = format_dims("16x9", quality_tier(snapshot))
    fps = ken_burns_local.kb._master_fps(snapshot, None)
    assert zoom_in_ids[0] == deterministic_media_id(
        "ken_burns_local", "cover", content_key, 4000, width, height, fps
    )
    assert set(zoom_in_ids).isdisjoint(pan_ids)


@pytest.mark.asyncio
async def test_an_unknown_preset_takes_the_zoom_in_and_is_recorded(
    monkeypatch: pytest.MonkeyPatch, _local_io: dict[str, Any]
) -> None:
    render = AsyncMock(return_value=b"\x00mp4")
    monkeypatch.setattr(ken_burns_local, "render_local_ken_burns", render)
    notes: list[str] = []
    degradations: list[Degradation] = []
    out = await _rescue({"ken_burns": {"preset": "spin_wildly"}}, notes, degradations)
    assert {call.kwargs["preset"] for call in render.await_args_list} == {"zoom_in"}
    assert all(shot["video_url"] for shot in out)  # still animated, not frozen
    assert [d["code"] for d in degradations] == [KEN_BURNS_PRESET_UNKNOWN]
    assert "'spin_wildly'" in degradations[0]["detail"]
    assert notes == [degradations[0]["detail"]]


@pytest.mark.asyncio
@pytest.mark.parametrize("config", [{}])
async def test_an_absent_preset_is_the_default_and_records_nothing(
    monkeypatch: pytest.MonkeyPatch, _local_io: dict[str, Any], config: dict[str, Any]
) -> None:
    render = AsyncMock(return_value=b"\x00mp4")
    monkeypatch.setattr(ken_burns_local, "render_local_ken_burns", render)
    notes: list[str] = []
    degradations: list[Degradation] = []
    await _rescue(config, notes, degradations)
    assert {call.kwargs["preset"] for call in render.await_args_list} == {"zoom_in"}
    assert notes == [] and degradations == []


# ---------------------------------------------------------------------------
# One real render per preset (ffmpeg 4.4 locally, 7.1 in prod: the expressions
# use only ``W``/``H``/``on`` and arithmetic, which both evaluate alike)
# ---------------------------------------------------------------------------
_FFMPEG = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
_W, _H = 640, 360
_GREEN = (0, 255, 0)


def _marker_still() -> bytes:
    """A 16:9 still: a red square dead centre and a green band on all four edges."""
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


async def _clip(tmp_path: Any, fit: str, preset: str) -> list[Image.Image]:
    video = await ken_burns_local.render_local_ken_burns(
        _marker_still(),
        width=_W,
        height=_H,
        fps=30,
        duration_s=2.0,
        timeout_s=120.0,
        fit=fit,
        preset=preset,
    )
    clip = tmp_path / f"{fit}-{preset}.mp4"
    clip.write_bytes(video)
    return [_frame(clip, index) for index in (0, 30, 59)]


_CX, _CY = (_W - 1) / 2, (_H - 1) / 2
#: Where the centre marker STARTS for each pan (+1 = right/below centre) — the
#: picture then drifts to centre in the named direction.
_PAN_START = {"pan_left": (1, 0), "pan_right": (-1, 0), "pan_up": (0, 1), "pan_down": (0, -1)}


@_FFMPEG
@pytest.mark.asyncio
async def test_a_real_zoom_out_shrinks_the_picture_about_the_centre(tmp_path: Any) -> None:
    first, middle, last = (_red_marker(f) for f in await _clip(tmp_path, "cover", "zoom_out"))
    # (Mid-move frames sit on sub-pixel edges, so the thresholded width is not
    # monotone frame to frame; the end points are exact.)
    assert first[2] > last[2] and middle[2] <= first[2]
    assert first[2] / last[2] == pytest.approx(1 + ken_burns_local.LOCAL_ZOOM_TRAVEL, abs=0.04)
    for cx, cy, _width in (first, middle, last):
        assert abs(cx - _CX) <= 2.0 and abs(cy - _CY) <= 2.0


@_FFMPEG
@pytest.mark.asyncio
@pytest.mark.parametrize("preset", ["pan_left"])
async def test_a_real_cover_pan_moves_the_picture_the_named_way(tmp_path: Any, preset: str) -> None:
    first, middle, last = (_red_marker(f) for f in await _clip(tmp_path, "cover", preset))
    sx, sy = _PAN_START[preset]
    # The marker starts the whole slack off centre on the named side (38 px of
    # 640 across, 21.6 px of 360 down) and ends centred.
    zoom = 1 + ken_burns_local.LOCAL_ZOOM_TRAVEL
    slack_x, slack_y = (zoom - 1) / 2 * _W, (zoom - 1) / 2 * _H
    off_x, off_y = first[0] - _CX, first[1] - _CY
    assert off_x * sx >= 0.85 * slack_x if sx else abs(off_x) <= 2.0
    assert off_y * sy >= 0.85 * slack_y if sy else abs(off_y) <= 2.0
    assert abs(last[0] - _CX) <= 2.0 and abs(last[1] - _CY) <= 2.0
    assert abs(middle[0] - _CX) < abs(off_x) + 1 and abs(middle[1] - _CY) < abs(off_y) + 1
    assert first[2] == pytest.approx(last[2], abs=2)  # a pan, not a zoom


@_FFMPEG
@pytest.mark.asyncio
@pytest.mark.parametrize("preset", ["zoom_out"])
async def test_a_real_contain_move_never_crops_the_slide(tmp_path: Any, preset: str) -> None:
    frames = await _clip(tmp_path, "contain", preset)
    assert [_edges_showing_green(f) for f in frames] == [[True] * 4] * 3
    first, last = _red_marker(frames[0]), _red_marker(frames[-1])
    assert first != last  # it moves
