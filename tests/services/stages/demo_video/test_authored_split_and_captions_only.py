"""Authored ``split`` shot + captions-only (silent) mode.

* ``line.split`` pins a two-pane layout: the LLM footage director never sees the line,
  and compile applies the author's split over any director/critique plan — even when
  ``direction.footage`` is off.
* ``brief.captions_only`` skips TTS: narrate emits no clips (and is not flagged as having
  "produced nothing"), compile ships each line's text as a SOFT overlay caption (never
  burned in unless ``captions_mode=burned_in`` is explicit) and times an unauthored
  beat by reading speed.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from pydantic import ValidationError
from w2v_common.clients import WMResponse

from src.api.schemas.projects import ShowcaseLine
from src.models.showcase_line import AuthoredLine, AuthoredSplit
from src.services.runs.dag_executor import DAGExecutor
from src.services.runs.run_context import RunContext
from src.services.stages.base import StageSkippedError
from src.services.stages.demo_video.compile import DemoVideoCompileStage
from src.services.stages.demo_video.footage_direction import DemoVideoFootageDirectionStage
from src.services.stages.demo_video.narrate import (
    DemoVideoNarrateOutput,
    DemoVideoNarrateStage,
    DemoVideoSilentNarrateOutput,
)
from src.services.stages.primitives.compile_stage import captions_mode
from src.services.stages.shared.authored_storyboard import authored_lines
from src.services.stages.shared.captions_only import (
    MAX_READING_SECONDS,
    MIN_READING_SECONDS,
    reading_seconds,
)
from src.services.stages.shared.footage_plans import (
    authored_split_plans,
    pin_authored_splits,
    resolve_footage_plans,
    split_second_clip_tags,
)

_SPLIT = {"layout": "h", "second_clip": "clip-2"}


# ── the authored split: model + door ──────────────────────────────────────────


@pytest.mark.parametrize("bad", [{"layout": "diag", "second_clip": "clip-2"}])
def test_split_model_rejects_a_bad_layout_or_missing_clip(bad: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        AuthoredSplit.model_validate(bad)


def test_split_model_rejects_an_empty_second_clip() -> None:
    with pytest.raises(ValidationError):
        AuthoredSplit(layout="h", second_clip="  @ ")


def test_the_api_door_accepts_a_split_and_rejects_a_bad_one() -> None:
    assert ShowcaseLine.model_validate({"text": "x", "split": _SPLIT}).split is not None
    with pytest.raises(ValidationError):
        ShowcaseLine.model_validate({"text": "x", "split": {"layout": "z", "second_clip": "a"}})


def test_a_brief_line_carries_its_split_into_the_authored_spine() -> None:
    # The authored spine is rebuilt FIELD-BY-FIELD from brief.lines: an unlisted field is
    # silently dropped, which is the failure this pins.
    lines = authored_lines(
        {
            "lines": [
                {"id": "clip-1", "text": "Requester files.", "split": _SPLIT},
                {"id": "clip-2", "text": "GM approves."},
            ]
        }
    )
    assert lines is not None
    assert lines[0].split == AuthoredSplit(layout="h", second_clip="clip-2")
    assert lines[1].split is None


# ── the authored split: planner bypass + compile precedence ───────────────────


def _wm(plans: list[dict[str, Any]]) -> AsyncMock:
    fake = AsyncMock()
    fake.message.return_value = WMResponse(
        output={"plans": plans},
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )
    return fake


def _direction_ctx(lines: list[dict[str, Any]]) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"direction": {"footage": {"auto_zoom": True}}},
        prior_run_outputs={
            "demo_video_gen": {"lines": lines},
            "demo_video_narrate": {"clips": []},
            "demo_video_record": {
                "clips": [{"line_id": line["id"], "video_url": "http://v/x.mp4"} for line in lines]
            },
        },
    )


async def test_the_llm_director_never_sees_an_authored_split_line() -> None:
    fake = _wm([{"line_id": "clip-2", "zoom": "auto"}])
    lines = [
        {"id": "clip-1", "text": "Requester files.", "seconds": 4.0, "split": _SPLIT},
        {"id": "clip-2", "text": "GM approves.", "seconds": 4.0},
    ]
    out = await DemoVideoFootageDirectionStage(client=fake).run({}, ctx=_direction_ctx(lines))
    sent = fake.message.await_args.kwargs["message"]
    assert '"line_id": "clip-2"' in sent
    assert '"line_id": "clip-1"' not in sent
    assert [p.line_id for p in out.plans] == ["clip-2"]


def test_the_authors_split_wins_over_the_directors_plan_for_that_line() -> None:
    authored = authored_split_plans([{"id": "clip-1", "split": _SPLIT}, {"id": "clip-2"}])
    assert set(authored) == {"clip-1"}
    director: dict[str, dict[str, object]] = {
        "clip-1": {"line_id": "clip-1", "zoom": "off", "split": {"layout": "v", "second_clip": "x"}}
    }
    merged = pin_authored_splits(director, authored)
    assert merged["clip-1"]["split"] == _SPLIT
    assert merged["clip-1"]["zoom"] == "off"  # the rest of the director's plan is kept
    assert director["clip-1"]["split"] == {"layout": "v", "second_clip": "x"}  # not mutated


def test_a_pinned_split_resolves_its_second_clip_to_a_url_and_drops_a_dangling_one() -> None:
    plans = authored_split_plans(
        [
            {"id": "clip-1", "split": _SPLIT},
            {"id": "clip-3", "split": {**_SPLIT, "second_clip": "nope"}},
        ]
    )
    *_, splits = resolve_footage_plans(
        line_ids=["clip-1", "clip-2", "clip-3"],
        plan_by_line=plans,
        focus_track_by_line=None,
        cursor_track_by_line=None,
        video_by_line={"clip-1": "http://v/1.mp4", "clip-2": "http://v/2.mp4"},
    )
    assert splits == {"clip-1": {"layout": "h", "second_source_url": "http://v/2.mp4"}}


def _compile_ctx(
    lines: list[dict[str, Any]],
    *,
    brief: dict[str, Any] | None = None,
    footage: bool = False,
    mode: str = "none",
) -> RunContext:
    config: dict[str, Any] = {"brief": brief or {}, "deliverable": {"captions_mode": mode}}
    if footage:
        config["direction"] = {"footage": {"auto_zoom": True}}
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot=config,
        prior_run_outputs={
            "demo_video_gen": {"lines": lines},
            "demo_video_narrate": {"clips": []},
            "demo_video_record": {
                "clips": [
                    {"line_id": "clip-1", "video_url": "http://v/1.mp4"},
                    {"line_id": "clip-2", "video_url": "http://v/2.mp4"},
                ]
            },
        },
    )


def test_compile_stamps_the_authored_split_on_the_shot_with_footage_direction_off() -> None:
    lines = [
        {"id": "clip-1", "text": "Requester files.", "seconds": 4.0, "split": _SPLIT},
        {"id": "clip-2", "text": "GM approves.", "seconds": 4.0},
    ]
    shots = DemoVideoCompileStage(client=AsyncMock()).build_shots(_compile_ctx(lines))
    assert shots[0]["split"] == {"layout": "h", "second_source_url": "http://v/2.mp4"}
    assert "split" not in shots[1]


def test_compile_without_an_authored_split_is_unchanged() -> None:
    lines = [{"id": "clip-1", "text": "a", "seconds": 4.0}, {"id": "clip-2", "text": "b"}]
    shots = DemoVideoCompileStage(client=AsyncMock()).build_shots(_compile_ctx(lines))
    assert all("split" not in s for s in shots)


# ── captions-only ─────────────────────────────────────────────────────────────


def test_reading_seconds_is_clamped() -> None:
    assert reading_seconds("Hi") == MIN_READING_SECONDS
    assert reading_seconds(" ".join(["word"] * 300)) == MAX_READING_SECONDS
    assert reading_seconds(" ".join(["word"] * 12)) == pytest.approx(4.0)


def _narrate_ctx(brief: dict[str, Any]) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"voice": {"voice_id": "af_heart"}, "brief": brief},
        prior_run_outputs={
            "demo_video_gen": {"lines": [{"id": "clip-1", "text": "Requester files the ticket."}]}
        },
    )


async def test_captions_only_narrate_makes_no_tts_call_and_emits_no_clips() -> None:
    client = AsyncMock()
    out = await DemoVideoNarrateStage(client=client).run(
        {}, ctx=_narrate_ctx({"captions_only": True})
    )
    assert isinstance(out, DemoVideoSilentNarrateOutput)
    assert out.clips == []
    client.synthesize.assert_not_called()


async def test_a_voiced_run_still_goes_to_tts() -> None:
    # Control: without the flag the same stage does reach the TTS client.
    client = AsyncMock()
    client.synthesize.side_effect = RuntimeError("tts reached")
    with pytest.raises(Exception):  # noqa: B017 — any failure proves TTS was tried
        await DemoVideoNarrateStage(client=client).run({}, ctx=_narrate_ctx({}))
    client.synthesize.assert_called()


async def test_captions_only_false_or_truthy_string_is_not_silent() -> None:
    for flag in (False, "true", 1, None):
        client = AsyncMock()
        client.synthesize.side_effect = RuntimeError("tts reached")
        with pytest.raises(Exception):  # noqa: B017 — any failure proves TTS was tried
            await DemoVideoNarrateStage(client=client).run(
                {}, ctx=_narrate_ctx({"captions_only": flag})
            )
        client.synthesize.assert_called()


def test_the_silent_output_is_not_flagged_as_producing_nothing() -> None:
    # The executor enforces `produces` on type(result): a deliberate empty must not be
    # recorded as "completed without producing clips" (a degraded run).
    silent: dict[str, object] = {"clips": [], "notes": []}
    DAGExecutor._enforce_produces("demo_video_narrate", DemoVideoSilentNarrateOutput, silent)
    assert silent["notes"] == []
    voiced: dict[str, object] = {"clips": [], "notes": []}
    DAGExecutor._enforce_produces("demo_video_narrate", DemoVideoNarrateOutput, voiced)
    assert voiced["notes"]  # control: the voiced model still flags an empty clip list


_SILENT_LINES: list[dict[str, Any]] = [
    {"id": "clip-1", "text": "Sophie files a ticket for a new hire.", "seconds": 0.0},
    {"id": "clip-2", "text": "The manager approves.", "seconds": 6.5},
]


def test_captions_only_defaults_to_the_soft_overlay_never_burned_in() -> None:
    """Owner rule: subtitles are never burned in by default. A captions-only run with
    no explicit mode ships the overlay track; nothing is stamped onto the picture."""
    ctx = _compile_ctx(_SILENT_LINES, brief={"captions_only": True})
    assert captions_mode(ctx) == "overlay"
    shots = DemoVideoCompileStage(client=AsyncMock()).build_shots(ctx)
    assert all("caption" not in s for s in shots)
    assert shots[0]["seconds"] == pytest.approx(reading_seconds(_SILENT_LINES[0]["text"]))
    assert shots[1]["seconds"] == pytest.approx(6.5)  # authored seconds win
    assert all("audio_url" not in s for s in shots)


def test_captions_only_overlay_builds_shot_spanning_soft_cues() -> None:
    ctx = _compile_ctx(_SILENT_LINES, brief={"captions_only": True})
    stage = DemoVideoCompileStage(client=AsyncMock())
    shots = stage.build_shots(ctx)
    cues = stage.static_caption_cues(ctx, shots)
    assert cues == [
        {
            "shot_index": i,
            "cues": [{"text": ln["text"], "start": 0.0, "end": shots[i]["seconds"], "words": []}],
        }
        for i, ln in enumerate(_SILENT_LINES)
    ]
    assert stage._emit_shot_timeline(ctx) is True  # the SRT sidecar's timing source


def test_captions_only_with_explicit_burned_in_still_burns() -> None:
    ctx = _compile_ctx(_SILENT_LINES, brief={"captions_only": True}, mode="burned_in")
    assert captions_mode(ctx) == "burned_in"
    shots = DemoVideoCompileStage(client=AsyncMock()).build_shots(ctx)
    assert [s["caption"] for s in shots] == [ln["text"] for ln in _SILENT_LINES]


def test_a_voiced_run_gets_no_static_cues() -> None:
    ctx = _compile_ctx(_SILENT_LINES, mode="overlay")
    stage = DemoVideoCompileStage(client=AsyncMock())
    assert stage.static_caption_cues(ctx, stage.build_shots(ctx)) is None


def test_voiced_overlay_publishes_clip_refs_and_stamps_no_static_caption() -> None:
    """overlay word-times the narration exactly like burned_in; only the render differs."""
    lines = [{"id": "clip-1", "text": "Sophie files.", "seconds": 0.0}]
    ctx = _compile_ctx(lines, mode="overlay")
    ctx.prior_run_outputs["demo_video_narrate"] = {
        "clips": [{"id": "n1", "line_id": "clip-1", "seconds": 2.0, "artifact_url": "http://a"}]
    }
    stage = DemoVideoCompileStage(client=AsyncMock())
    refs = stage.caption_clip_refs(ctx)
    assert list(refs) == ["clip-1"] and refs["clip-1"].text == "Sophie files."
    assert all("caption" not in s for s in stage.build_shots(ctx))


def test_a_voiced_compile_burns_no_captions_with_captions_mode_none() -> None:
    lines = [{"id": "clip-1", "text": "Sophie files.", "seconds": 0.0}]
    shots = DemoVideoCompileStage(client=AsyncMock()).build_shots(_compile_ctx(lines))
    assert "caption" not in shots[0]


def test_authored_line_split_default_is_none() -> None:
    assert AuthoredLine().split is None


def test_a_split_may_name_a_recorded_clip_tag_that_no_line_plays_over() -> None:
    # Approval beats: requester chat on the left, the GM's separately recorded clip on
    # the right. The GM clip has no narration line of its own — the split names its tag.
    plans = authored_split_plans(
        [{"id": "beat-2", "clip": "clip-1", "split": {"layout": "h", "second_clip": "@clip-2"}}]
    )
    *_, splits = resolve_footage_plans(
        line_ids=["beat-2"],
        plan_by_line=plans,
        focus_track_by_line=None,
        cursor_track_by_line=None,
        video_by_line={"beat-2": "http://v/requester.mp4"},
        video_by_clip={"clip-1": "http://v/requester.mp4", "clip-2": "http://v/gm.mp4"},
    )
    assert splits == {"beat-2": {"layout": "h", "second_source_url": "http://v/gm.mp4"}}


def test_record_keeps_a_split_second_clip_tag_but_not_a_line_id() -> None:
    lines: list[dict[str, Any]] = [
        {"id": "beat-1", "clip": "clip-1"},
        {"id": "beat-2", "clip": "clip-1", "split": {"layout": "h", "second_clip": "clip-2"}},
        {"id": "beat-3", "split": {"layout": "v", "second_clip": "beat-1"}},  # a line id
    ]
    assert split_second_clip_tags(lines) == {"clip-2"}


async def test_director_false_skips_the_llm_planner_without_a_model_call() -> None:
    fake = _wm([{"line_id": "clip-1", "zoom": "auto"}])
    lines = [{"id": "clip-1", "text": "a", "seconds": 4.0}]
    ctx = _direction_ctx(lines)
    ctx.config_snapshot["direction"] = {"footage": {"auto_zoom": True, "director": False}}
    with pytest.raises(StageSkippedError) as exc:
        await DemoVideoFootageDirectionStage(client=fake).run({}, ctx=ctx)
    assert exc.value.missing == ["direction.footage.director"]
    fake.message.assert_not_called()


async def test_director_unset_or_true_still_plans() -> None:
    for extra in ({}, {"director": True}):
        fake = _wm([{"line_id": "clip-1", "zoom": "auto"}])
        ctx = _direction_ctx([{"id": "clip-1", "text": "a", "seconds": 4.0}])
        ctx.config_snapshot["direction"] = {"footage": {"auto_zoom": True, **extra}}
        out = await DemoVideoFootageDirectionStage(client=fake).run({}, ctx=ctx)
        assert [p.line_id for p in out.plans] == ["clip-1"]
        fake.message.assert_called()
