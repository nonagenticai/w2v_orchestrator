from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.services.runs.run_context import RunContext
from src.services.stages.promo.render import (
    PromoRenderStage,
)
from src.services.stages.sound_catalog import SoundLibrary
from src.services.visual_media import overlay_motion_slides as _overlay_motion_slides
from tests.support.geometry_wire import callouts_wire, cursor_wire, focus_wire


def _use_builtin_library(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate beds against the built-in library, hermetic from storage."""

    async def _fake(*_args: object, **_kwargs: object) -> SoundLibrary:
        return SoundLibrary.builtin()

    monkeypatch.setattr("src.services.stages.shared.sound_design.load_sound_library", _fake)


def _shot(line_id: str, vtype: str) -> dict:
    return {"id": line_id, "line_id": line_id, "visual_type": vtype, "seconds": 1.0}


def test_overlay_motion_slides_upgrades_only_still_slide_lines() -> None:
    # l0 shows its still slide → upgraded to its animated clip; l1 shows an html
    # screenshot (dedicated image) → untouched; l2 already has a footage clip in
    # video_by_line → untouched.
    slide_by_line = {"l0": "s0", "l1": "s1", "l2": "s2"}
    image_by_line = {"l0": "s0", "l1": "html1", "l2": "s2"}
    video_by_line = {"l2": "footage2"}
    fit_by_line: dict[str, str] = {}
    _overlay_motion_slides(
        image_by_line,
        video_by_line,
        fit_by_line,
        slide_by_line=slide_by_line,
        motion_slide_by_line={"l0": "motion0", "l1": "motion1", "l2": "motion2"},
    )
    assert video_by_line == {"l2": "footage2", "l0": "motion0"}  # only l0 upgraded
    assert fit_by_line == {"l0": "contain"}  # slide text not cropped
    assert image_by_line["l0"] == "s0"  # still PNG remains the degrade path


def test_overlay_motion_slides_empty_is_noop() -> None:
    video_by_line: dict[str, str] = {}
    fit_by_line: dict[str, str] = {}
    _overlay_motion_slides(
        {"l0": "s0"},
        video_by_line,
        fit_by_line,
        slide_by_line={"l0": "s0"},
        motion_slide_by_line={},
    )
    assert video_by_line == {}
    assert fit_by_line == {}


def _ctx_with(captions_mode: str) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "deliverable": {"format": "16x9", "captions_mode": captions_mode},
        },
        prior_run_outputs={
            "promo_script": {
                "lines": [{"id": "l1", "text": "x", "seconds": 1.0}],
                "total_seconds": 1.0,
            },
            "promo_storyboard": {
                "shots": [
                    {
                        "id": "s1",
                        "line_id": "l1",
                        "description": "d",
                        "seconds": 1.0,
                        "visual_style": "x",
                    }
                ],
            },
        },
    )


@pytest.mark.asyncio
async def test_render_burns_in_captions_when_mode_is_burned_in() -> None:
    fake = AsyncMock()
    fake.compile.return_value = {"artifact_url": "http://stub/a.mp4", "duration_seconds": 1.0}
    stage = PromoRenderStage(client=fake)

    out = await stage.run({}, ctx=_ctx_with("burned_in"))

    assert out.artifact_url == "http://stub/a.mp4"
    assert out.duration_seconds == 1.0
    assert out.captions_burned is True
    _, kwargs = fake.compile.call_args
    assert kwargs["tags"]["captions_mode"] == "burned_in"
    assert kwargs["tags"]["format"] == "16x9"
    assert kwargs["tags"]["stage"] == "promo_render"


@pytest.mark.asyncio
async def test_render_does_not_burn_in_when_mode_is_none() -> None:
    fake = AsyncMock()
    fake.compile.return_value = {"artifact_url": "http://stub/b.mp4", "duration_seconds": 2.5}
    stage = PromoRenderStage(client=fake)

    out = await stage.run({}, ctx=_ctx_with("none"))

    assert out.captions_burned is False
    assert out.duration_seconds == 2.5


@pytest.mark.asyncio
async def test_render_raises_without_storyboard_or_script() -> None:
    fake = AsyncMock()
    stage = PromoRenderStage(client=fake)
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})

    with pytest.raises(RuntimeError, match="storyboard"):
        await stage.run({}, ctx=ctx)


def test_build_shots_first_line_per_screen_gets_clip_rest_get_still() -> None:
    """End-to-end keying: the FIRST footage line that shows a screen gets that
    screen's reveal clip (motion); later lines on the same screen get the still.
    A second screen gets its own reveal clip on its first line."""
    stage = PromoRenderStage(client=AsyncMock())
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"deliverable": {"format": "16x9", "captions_mode": "none"}},
        prior_run_outputs={
            "promo_script": {
                "lines": [
                    {"id": "q1", "text": "What's this?", "seconds": 2.0, "speaker": "a"},
                    {"id": "a1", "text": "The leaderboard.", "seconds": 2.0, "speaker": "b"},
                    {"id": "a2", "text": "It updates live.", "seconds": 2.0, "speaker": "b"},
                    {"id": "a3", "text": "Now the sandbox.", "seconds": 2.0, "speaker": "b"},
                ],
                "total_seconds": 8.0,
            },
            "promo_storyboard": {
                "shots": [
                    {"id": s, "line_id": s, "visual_type": "footage", "seconds": 2.0}
                    for s in ("q1", "a1", "a2", "a3")
                ],
            },
            "site_capture": {
                "screens": [
                    {
                        "id": "leaderboard",
                        "label": "Leaderboard",
                        "image_url": "https://minio/lead.png",
                        "clip_url": "https://minio/lead.mp4",
                    },
                    {
                        "id": "sandbox",
                        "label": "Sandbox",
                        "image_url": "https://minio/sb.png",
                        "clip_url": "https://minio/sb.mp4",
                    },
                ],
            },
            "promo_visual_director": {
                "assignments": [
                    {"line_id": "q1", "screen_id": "leaderboard"},
                    {"line_id": "a1", "screen_id": "leaderboard"},
                    {"line_id": "a2", "screen_id": "leaderboard"},
                    {"line_id": "a3", "screen_id": "sandbox"},
                ],
            },
        },
    )
    shots = {s["line_id"]: s for s in stage.build_shots(ctx)}
    # q1 first shows the leaderboard → reveal clip; a1/a2 hold it → still.
    assert shots["q1"].get("video_url") == "https://minio/lead.mp4"
    assert shots["q1"].get("fit") == "cover"
    assert shots["a1"].get("image_url") == "https://minio/lead.png"
    assert "video_url" not in shots["a1"]
    assert shots["a2"].get("image_url") == "https://minio/lead.png"
    # a3 first shows the sandbox → its own reveal clip.
    assert shots["a3"].get("video_url") == "https://minio/sb.mp4"


# --------------------------------------------------------------------------
# ORCH-3.3 directed shot-plan (promo_footage_direction) — callout / zoom-off / split
# --------------------------------------------------------------------------

_BBOX_A: dict[str, object] = {"x": 10, "y": 20, "w": 100, "h": 40}
_BBOX_B: dict[str, object] = {"x": 5, "y": 5, "w": 50, "h": 50}
_BBOX_C: dict[str, object] = {"x": 30, "y": 40, "w": 80, "h": 60}
# The RECORDER shapes the site_capture screens carry (dict bbox + the recorder-only
# ``action`` key); ``*_wire`` states what they become on the compile wire.
_TRACK_A: list[dict[str, object]] = [{"t": 0.5, "bbox": _BBOX_A, "action": "click"}]
_TRACK_B: list[dict[str, object]] = [{"t": 0.5, "bbox": _BBOX_B, "action": "click"}]
_TRACK_C: list[dict[str, object]] = [{"t": 0.5, "bbox": _BBOX_C, "action": "click"}]
_CURSOR_A: list[dict[str, object]] = [{"t": 0.0, "x": 1, "y": 1}, {"t": 0.5, "x": 9, "y": 9}]
_CURSOR_B: list[dict[str, object]] = [{"t": 0.0, "x": 2, "y": 2}, {"t": 0.5, "x": 8, "y": 8}]
_CURSOR_C: list[dict[str, object]] = [{"t": 0.0, "x": 3, "y": 3}, {"t": 0.5, "x": 7, "y": 7}]


def _footage_direction_ctx(*, plans: list[dict[str, Any]] | None) -> RunContext:
    """Three footage lines, each REVEALING its own screen (so each gets a reveal
    clip → ``video_url``, the footage gate ``_attach_clip_framing`` requires). Every
    screen carries a captured focus (with bbox) + cursor track. ``plans`` None ⇒ the
    ``promo_footage_direction`` stage is absent entirely (the byte-identical path)."""
    outputs: dict[str, dict[str, Any]] = {
        "promo_script": {
            "lines": [
                {"id": "l1", "text": "a", "seconds": 2.0},
                {"id": "l2", "text": "b", "seconds": 2.0},
                {"id": "l3", "text": "c", "seconds": 2.0},
            ],
            "total_seconds": 6.0,
        },
        "promo_storyboard": {
            "shots": [
                {"id": s, "line_id": s, "visual_type": "footage", "seconds": 2.0}
                for s in ("l1", "l2", "l3")
            ],
        },
        "site_capture": {
            "screens": [
                {
                    "id": "scrA",
                    "image_url": "https://minio/a.png",
                    "clip_url": "https://minio/a.mp4",
                    "focus_track": _TRACK_A,
                    "cursor_track": _CURSOR_A,
                    "viewport": [1280, 720],
                },
                {
                    "id": "scrB",
                    "image_url": "https://minio/b.png",
                    "clip_url": "https://minio/b.mp4",
                    "focus_track": _TRACK_B,
                    "cursor_track": _CURSOR_B,
                    "viewport": [1280, 720],
                },
                {
                    "id": "scrC",
                    "image_url": "https://minio/c.png",
                    "clip_url": "https://minio/c.mp4",
                    "focus_track": _TRACK_C,
                    "cursor_track": _CURSOR_C,
                    "viewport": [1280, 720],
                },
            ],
        },
        "promo_visual_director": {
            "assignments": [
                {"line_id": "l1", "screen_id": "scrA"},
                {"line_id": "l2", "screen_id": "scrB"},
                {"line_id": "l3", "screen_id": "scrC"},
            ],
        },
    }
    if plans is not None:
        outputs["promo_footage_direction"] = {"plans": plans}
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "deliverable": {"format": "16x9", "captions_mode": "none"},
            "direction": {"footage": {"auto_zoom": True}},
        },
        prior_run_outputs=outputs,
    )


def test_footage_direction_applies_callout_zoom_off_and_split() -> None:
    """The shot-plan rides the captured geometry: l1 gets a callout at its captured
    bbox, l2's zoom-off co-removes its focus+cursor, l3 gets a split resolved to
    l2's footage URL (via video_by_line)."""
    stage = PromoRenderStage(client=AsyncMock())
    ctx = _footage_direction_ctx(
        plans=[
            {"line_id": "l1", "zoom": "auto", "callout": {"type": "badge", "label": "Click"}},
            {"line_id": "l2", "zoom": "off"},
            {"line_id": "l3", "zoom": "auto", "split": {"layout": "h", "second_clip": "l2"}},
        ]
    )
    shots = {s["line_id"]: s for s in stage.build_shots(ctx)}
    # l1: callout placed ONLY at the captured focus bbox (never an LLM coordinate).
    # ...as a one-element ARRAY on the wire (``Shot.callouts`` is a list), with the
    # content carried verbatim and only the bbox converted to the declared 4-tuple.
    assert shots["l1"]["callouts"] == callouts_wire(
        {"type": "badge", "label": "Click", "at": 0.5, "bbox": _BBOX_A}
    )
    assert shots["l1"]["focus_track"] == focus_wire(_TRACK_A)
    # l2: zoom off co-removes focus + cursor (a cursor easing to a no-zoom target is
    # the bug). The reveal clip itself is untouched.
    assert "focus_track" not in shots["l2"]
    assert "cursor_track" not in shots["l2"]
    assert shots["l2"]["video_url"] == "https://minio/b.mp4"
    # l3: split resolves second_clip=l2 to l2's footage URL compile-side.
    assert shots["l3"]["split"] == {"layout": "h", "second_source_url": "https://minio/b.mp4"}
    assert "callouts" not in shots["l3"]


def test_absent_footage_direction_is_byte_identical() -> None:
    """No promo_footage_direction prior_output ⇒ the helper returns the captured maps
    untouched and emits no callouts/split ⇒ the shot list is byte-identical to a run
    with the stage present but empty (today's behaviour)."""
    stage = PromoRenderStage(client=AsyncMock())
    never_ran = stage.build_shots(_footage_direction_ctx(plans=None))
    empty_plans = stage.build_shots(_footage_direction_ctx(plans=[]))
    assert never_ran == empty_plans
    for shots in (never_ran, empty_plans):
        assert all("callouts" not in s for s in shots)
        assert all("split" not in s for s in shots)
        by = {s["line_id"]: s for s in shots}
        # Captured geometry rides through unchanged (the seam is inert).
        # Every captured keyframe survives, in order, with the same count — only the
        # recorder->wire normalisation (dict bbox -> 4-tuple, ``action`` dropped)
        # applied, which is what _attach_clip_framing does to ANY track.
        assert by["l1"]["focus_track"] == focus_wire(_TRACK_A)
        assert by["l2"]["cursor_track"] == cursor_wire(_CURSOR_B)


# --------------------------------------------------------------------------
# Sound design: config resolution + per-line plan + music/ambient bed
# --------------------------------------------------------------------------

from src.services.stages.shared.sound_design import (  # noqa: E402
    sfx_by_line,
    sound_config,
    transition_sfx_id,
)


def _sound_ctx(sound: dict | None, *, director_cues: list | None = None) -> RunContext:
    prior: dict = {}
    if director_cues is not None:
        prior["promo_sound_director"] = {"cues": director_cues}
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"sound_effects": sound} if sound is not None else {},
        prior_run_outputs=prior,
    )


def test_transition_id_resolution() -> None:
    assert transition_sfx_id({}) is None  # not enabled
    assert transition_sfx_id({"enabled": True}) == "whoosh_soft"  # default
    assert transition_sfx_id({"enabled": True, "transitions": True}) == "whoosh_soft"
    assert transition_sfx_id({"enabled": True, "transitions": False}) is None
    assert transition_sfx_id({"enabled": True, "transitions": "swoosh_fast"}) == "swoosh_fast"
    # An explicit id passes through as-is; membership is validated at presign time
    # (resolve_shot_sound_urls) against the loaded library, where an unresolvable
    # id is dropped. So this stays a pure-config decision.
    assert transition_sfx_id({"enabled": True, "transitions": "anything"}) == "anything"


def test_sfx_by_line_groups_director_cues() -> None:
    cfg = {"enabled": True}
    ctx = _sound_ctx(
        cfg,
        director_cues=[
            {"line_id": "l1", "sfx_id": "click_ui", "at_seconds": 1.0, "gain_db": -8.0},
            {"line_id": "l1", "sfx_id": "ding_positive", "at_seconds": 2.0},
        ],
    )
    by_line = sfx_by_line(ctx, cfg, "promo_sound_director")
    assert len(by_line["l1"]) == 2
    assert by_line["l1"][0]["sfx_id"] == "click_ui"
    # No brief-level sfx_gain_db → the director's per-cue gain is honored, and a
    # cue that omits gain_db falls back to the -10dB default.
    assert by_line["l1"][0]["gain_db"] == -8.0
    assert by_line["l1"][1]["gain_db"] == -10.0


def test_sfx_by_line_gain_db_override_beats_per_cue_gain() -> None:
    # Regression for #363: the director emits a uniform per-cue gain, so a brief's
    # sfx_gain_db must OVERRIDE it (the old `cue.gain_db or sfx_gain_db` never fired).
    cfg = {"enabled": True, "sfx_gain_db": -18.0}
    ctx = _sound_ctx(
        cfg,
        director_cues=[
            {"line_id": "l1", "sfx_id": "click_ui", "at_seconds": 1.0, "gain_db": -8.0},
            {"line_id": "l1", "sfx_id": "ding_positive", "at_seconds": 2.0},
        ],
    )
    by_line = sfx_by_line(ctx, cfg, "promo_sound_director")
    assert [c["gain_db"] for c in by_line["l1"]] == [-18.0, -18.0]


def test_sfx_by_line_empty_when_disabled() -> None:
    ctx = _sound_ctx({}, director_cues=[{"line_id": "l1", "sfx_id": "click_ui", "at_seconds": 1.0}])
    assert sfx_by_line(ctx, sound_config(ctx), "promo_sound_director") == {}


async def test_build_audio_bed_none_when_disabled() -> None:
    stage = PromoRenderStage(client=AsyncMock())
    bed = await stage.build_audio_bed(_sound_ctx(None), [{"seconds": 5.0}])
    assert bed is None


async def test_build_audio_bed_resolves_music_and_ambient(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_presign(sfx_id: str) -> str:
        return f"https://minio.test/w2v-orchestrator/sound_library/shared/{sfx_id}.mp3?sig=x"

    monkeypatch.setattr("src.services.stages.shared.sound_design.presign_sound_asset", fake_presign)
    _use_builtin_library(monkeypatch)
    stage = PromoRenderStage(client=AsyncMock())
    cfg = {"enabled": True, "music": "corporate_uplift", "ambient": "room_tone"}
    shots = [{"seconds": 4.0}, {"seconds": 6.0}]
    bed = await stage.build_audio_bed(_sound_ctx(cfg), shots)
    assert bed is not None
    assert bed["music"]["url"].endswith("corporate_uplift.mp3?sig=x")
    # Ambient spans the FINAL encoded duration (sum of shot seconds).
    assert bed["ambient"][0]["end_seconds"] == 10.0
    assert bed["ambient"][0]["url"].endswith("room_tone.mp3?sig=x")


async def test_build_audio_bed_invalid_music_id_omitted(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_presign(sfx_id: str) -> str:
        return f"https://minio.test/{sfx_id}.mp3"

    monkeypatch.setattr("src.services.stages.shared.sound_design.presign_sound_asset", fake_presign)
    _use_builtin_library(monkeypatch)
    stage = PromoRenderStage(client=AsyncMock())
    # "room_tone" is an ambient id, not music → music omitted; ambient via default.
    cfg = {"enabled": True, "music": "room_tone", "ambient": True}
    bed = await stage.build_audio_bed(_sound_ctx(cfg), [{"seconds": 3.0}])
    assert bed is not None
    assert "music" not in bed
    assert bed["ambient"][0]["url"].endswith("room_tone.mp3")


def test_promo_render_compile_tags_omit_crf_when_mastering_crf_authored() -> None:
    """The gate applies on the promo render path too: an authored `mastering.crf`
    suppresses the tier crf tag so the compiler honours the mastering-block crf."""
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "resolution": "4k",
            "deliverable": {"format": "16x9"},
            "mastering": {"crf": 14},
        },
    )
    tags = PromoRenderStage(client=AsyncMock()).compile_tags(ctx)
    assert "crf" not in tags
    assert tags["render_width"] == "3840"
    assert tags["render_height"] == "2160"
