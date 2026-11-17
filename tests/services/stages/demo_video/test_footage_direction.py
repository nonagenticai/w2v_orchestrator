"""demo_video_footage_direction — authoritative shot-plan validation + skip.

The stage is advisory-in / authoritative-out (mirrors sound_director): the model's
plan is coerced to known members, unknown line_ids are dropped, and — the load-
bearing invariant — a line whose recording captured NO click has no bbox to aim at,
so its zoom is forced off and any callout dropped.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from w2v_common.clients import WMResponse

from src.services.runs.run_context import RunContext
from src.services.stages.base import StageSkippedError
from src.services.stages.demo_video.footage_direction import DemoVideoFootageDirectionStage
from src.services.stages.shared.footage_direction import (
    _MAX_CALLOUT_SHOTS_PER_RUN,
    Callout,
    FootageDirectionStage,
)


def _resp(obj: Any) -> WMResponse:
    return WMResponse(
        output=obj,
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )


def _ctx(
    *,
    footage: bool = True,
    record_focus: bool = True,
    lines: list[dict[str, Any]] | None = None,
) -> RunContext:
    direction: dict[str, Any] = {"footage": {"auto_zoom": True}} if footage else {}
    gen_lines = lines or [
        {"id": "clip-1", "text": "Meet the dashboard.", "seconds": 0.0},
        {"id": "clip-2", "text": "Eleven rooms detected.", "seconds": 0.0},
    ]
    record_clips: list[dict[str, Any]] = []
    if record_focus:
        record_clips = [
            {
                "line_id": "clip-1",
                "video_url": "http://v/1.mp4",
                "focus_track": [{"t": 0.5, "bbox": {"x": 10, "y": 20, "w": 100, "h": 40}}],
            },
            # clip-2 recorded no click target (no focus_track) → has_click False.
            {"line_id": "clip-2", "video_url": "http://v/2.mp4"},
        ]
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"direction": direction} if direction else {},
        prior_run_outputs={
            "demo_video_gen": {"lines": gen_lines},
            "demo_video_narrate": {
                "clips": [
                    {"id": "n1", "line_id": "clip-1", "seconds": 4.0},
                    {"id": "n2", "line_id": "clip-2", "seconds": 3.0},
                ]
            },
            "demo_video_record": {"clips": record_clips},
        },
    )


async def test_skipped_when_footage_not_requested() -> None:
    with pytest.raises(StageSkippedError) as exc:
        await DemoVideoFootageDirectionStage(client=AsyncMock()).run({}, ctx=_ctx(footage=False))
    assert exc.value.missing == ["direction.footage"]


class _NoFootageLinesStage(FootageDirectionStage):
    """Stands in for any pipeline whose shotlist yielded zero footage-typed lines
    (a slides-only promo, say) — the base's empty-list branch is what's under test."""

    name = "test_no_footage_lines"
    worker_name = "footage-director"

    def planner_lines(self, ctx: RunContext) -> list[Any]:
        return []


async def test_no_lines_to_direct_skips_clean_never_fails_the_run() -> None:
    """No footage-typed lines ⇒ nothing to frame ⇒ skip clean, NOT a precondition
    failure that kills the whole video.

    Load-bearing since WS-1 seeds `direction.footage` by default: the block's presence
    no longer proves an author asked for directed footage, so an empty planner line
    list must never be able to fail an otherwise-good run.
    """
    with pytest.raises(StageSkippedError) as exc:
        await _NoFootageLinesStage(client=AsyncMock()).run({}, ctx=_ctx())
    assert exc.value.missing == ["footage_lines"]


async def test_no_redrive_feedback_message_is_inert() -> None:
    """Without redrive_feedback the built message carries no critic block — the
    normal DAG run is byte-identical to today (feature inert by default)."""
    fake = AsyncMock()
    fake.message.return_value = _resp({"plans": []})
    await DemoVideoFootageDirectionStage(client=fake).run({}, ctx=_ctx())
    message = fake.message.call_args.kwargs["message"]
    assert "CRITIC FEEDBACK" not in message

    # Explicit None threads identically to omitting the field entirely.
    fake_none = AsyncMock()
    fake_none.message.return_value = _resp({"plans": []})
    await DemoVideoFootageDirectionStage(client=fake_none).run(
        {"redrive_feedback": None}, ctx=_ctx()
    )
    assert fake_none.message.call_args.kwargs["message"] == message


async def test_redrive_feedback_appended_to_message() -> None:
    """A non-empty redrive_feedback appends the critic block verbatim so a holistic
    critic stage can re-drive the director from the seams it found."""
    fake = AsyncMock()
    fake.message.return_value = _resp({"plans": []})
    await DemoVideoFootageDirectionStage(client=fake).run(
        {"redrive_feedback": "continuity break at clip-3"}, ctx=_ctx()
    )
    message = fake.message.call_args.kwargs["message"]
    assert "CRITIC FEEDBACK" in message
    assert "continuity break at clip-3" in message


async def test_unknown_line_id_dropped_and_enums_coerced() -> None:
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {
            "plans": [
                {
                    "line_id": "clip-1",
                    "zoom": "SUPERZOOM",  # bad → auto
                    "reframe": "wobble",  # bad → follow-cursor
                    "transition_signal": "explode",  # bad → none
                },
                {"line_id": "ghost", "zoom": "auto"},  # unknown line → dropped
            ]
        }
    )
    out = await DemoVideoFootageDirectionStage(client=fake).run({}, ctx=_ctx())
    assert [p.line_id for p in out.plans] == ["clip-1"]
    p = out.plans[0]
    assert (p.zoom, p.reframe, p.transition_signal) == ("auto", "follow-cursor", "none")


async def test_label_clamped_and_bad_callout_type_dropped() -> None:
    long_label = "x" * 400
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {
            "plans": [
                {
                    "line_id": "clip-1",
                    "zoom": "auto",
                    "callout": {"type": "badge", "label": long_label},
                },
                {"line_id": "clip-2", "callout": {"type": "bogus", "label": "hi"}},  # bad type
            ]
        }
    )
    out = await DemoVideoFootageDirectionStage(client=fake).run({}, ctx=_ctx())
    by = {p.line_id: p for p in out.plans}
    assert by["clip-1"].callout is not None
    assert by["clip-1"].callout.type == "badge"
    assert len(by["clip-1"].callout.label) == 120
    # clip-2 has no captured click AND a bad callout type → callout None either way.
    assert by["clip-2"].callout is None


async def test_no_click_forces_zoom_off_and_drops_callout() -> None:
    fake = AsyncMock()
    # The model tries to zoom + callout clip-2, but its recording captured no click.
    fake.message.return_value = _resp(
        {
            "plans": [
                {
                    "line_id": "clip-2",
                    "zoom": "auto",
                    "callout": {"type": "label", "label": "here"},
                },
            ]
        }
    )
    out = await DemoVideoFootageDirectionStage(client=fake).run({}, ctx=_ctx())
    p = out.plans[0]
    assert p.line_id == "clip-2"
    assert p.zoom == "off"  # no target ⇒ forced off
    assert p.callout is None  # no target ⇒ dropped


async def test_clip_keying_fans_has_click_onto_shared_lines() -> None:
    # Three dialogue lines all play over the ONE recorded clip-1 (has focus) → each
    # inherits has_click True, so the model's zoom survives on all three.
    lines = [
        {"id": "a", "clip": "clip-1", "text": "one"},
        {"id": "b", "clip": "clip-1", "text": "two"},
        {"id": "c", "clip": "clip-1", "text": "three"},
    ]
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {"plans": [{"line_id": lid, "zoom": "auto"} for lid in ("a", "b", "c")]}
    )
    out = await DemoVideoFootageDirectionStage(client=fake).run({}, ctx=_ctx(lines=lines))
    assert {p.line_id: p.zoom for p in out.plans} == {"a": "auto", "b": "auto", "c": "auto"}


async def test_card_line_with_no_clip_is_inert() -> None:
    # A showcase card line (no clip, no recording) has no target → zoom forced off.
    lines = [{"id": "card-1", "text": "Intro card", "seconds": 2.0}]
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {
            "plans": [
                {"line_id": "card-1", "zoom": "auto", "callout": {"type": "badge", "label": "x"}}
            ]
        }
    )
    out = await DemoVideoFootageDirectionStage(client=fake).run({}, ctx=_ctx(lines=lines))
    p = out.plans[0]
    assert p.zoom == "off"
    assert p.callout is None


async def test_spotlight_and_arrow_callout_types_survive() -> None:
    # Two has-click lines both bound to clip-1 (captured focus) → spotlight/arrow
    # are now known types and ride through _validate.
    lines = [
        {"id": "a", "clip": "clip-1", "text": "one"},
        {"id": "b", "clip": "clip-1", "text": "two"},
    ]
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {
            "plans": [
                {"line_id": "a", "zoom": "auto", "callout": {"type": "spotlight", "label": "look"}},
                {
                    "line_id": "b",
                    "zoom": "auto",
                    "callout": {"type": "arrow", "label": "over here"},
                },
            ]
        }
    )
    out = await DemoVideoFootageDirectionStage(client=fake).run({}, ctx=_ctx(lines=lines))
    by = {p.line_id: p for p in out.plans}
    a = by["a"].callout
    b = by["b"].callout
    assert a is not None and a.type == "spotlight"
    assert b is not None and b.type == "arrow"


async def test_unknown_callout_type_dropped_on_has_click_line() -> None:
    # clip-1 HAS a captured click, so a dropped callout can only be the bad type.
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {
            "plans": [
                {"line_id": "clip-1", "zoom": "auto", "callout": {"type": "bogus", "label": "hi"}},
            ]
        }
    )
    out = await DemoVideoFootageDirectionStage(client=fake).run({}, ctx=_ctx())
    p = out.plans[0]
    assert p.line_id == "clip-1"
    assert p.callout is None  # unknown type → dropped by _parse_callout


async def test_run_total_callout_cap_nulls_overflow_keeps_plans() -> None:
    # Seven has-click lines (all bound to clip-1) each get a callout; the run-total
    # cap keeps the callout on the first _MAX_CALLOUT_SHOTS_PER_RUN and nulls the
    # rest — but every plan survives.
    n = _MAX_CALLOUT_SHOTS_PER_RUN
    lines = [{"id": f"L{i}", "clip": "clip-1", "text": "t"} for i in range(n + 1)]
    fake = AsyncMock()

    # n+1 = 7 lines now span multiple bounded WM calls (batched split). A real
    # worker returns plans only for the lines its batch was shown, so the mock
    # echoes only the batch's line_ids — the run-total callout cap still applies
    # once over the order-preserved union (first _MAX_CALLOUT_SHOTS_PER_RUN keep it).
    def _batch_resp(*_: Any, **kwargs: Any) -> WMResponse:
        msg = kwargs["message"]
        ids = [f"L{i}" for i in range(n + 1) if f'"line_id": "L{i}"' in msg]
        return _resp(
            {
                "plans": [
                    {"line_id": lid, "zoom": "auto", "callout": {"type": "badge", "label": "x"}}
                    for lid in ids
                ]
            }
        )

    fake.message.side_effect = _batch_resp
    out = await DemoVideoFootageDirectionStage(client=fake).run({}, ctx=_ctx(lines=lines))
    assert len(out.plans) == n + 1  # no plan dropped
    with_callout = [p.line_id for p in out.plans if p.callout is not None]
    assert with_callout == [f"L{i}" for i in range(n)]  # first N keep the callout
    assert out.plans[n].callout is None  # the (N+1)th is nulled


async def test_malformed_splits_dropped_to_none() -> None:
    # Bad layout, missing second_clip, and non-dict split all coerce to None — the
    # authoritative-stage invariant (the compiler never sees a stray split).
    lines = [
        {"id": "clip-1", "text": "one", "seconds": 0.0},
        {"id": "clip-2", "text": "two", "seconds": 0.0},
        {"id": "clip-3", "text": "three", "seconds": 0.0},
    ]
    record_clips = [
        {
            "line_id": lid,
            "video_url": f"http://v/{lid}.mp4",
            "focus_track": [{"t": 0.5, "bbox": {"x": 1, "y": 2, "w": 3, "h": 4}}],
        }
        for lid in ("clip-1", "clip-2", "clip-3")
    ]
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"direction": {"footage": {"auto_zoom": True}}},
        prior_run_outputs={
            "demo_video_gen": {"lines": lines},
            "demo_video_narrate": {
                "clips": [
                    {"id": f"n{i}", "line_id": lid, "seconds": 2.0}
                    for i, lid in enumerate(("clip-1", "clip-2", "clip-3"))
                ]
            },
            "demo_video_record": {"clips": record_clips},
        },
    )
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {
            "plans": [
                {
                    "line_id": "clip-1",
                    "zoom": "auto",
                    "split": {"layout": "x", "second_clip": "clip-2"},  # bad layout
                },
                {"line_id": "clip-2", "zoom": "auto", "split": {"layout": "h"}},  # no second_clip
                {"line_id": "clip-3", "zoom": "auto", "split": "not-a-dict"},
            ]
        }
    )
    out = await DemoVideoFootageDirectionStage(client=fake).run({}, ctx=ctx)
    by = {p.line_id: p for p in out.plans}
    assert by["clip-1"].split is None  # bad layout
    assert by["clip-2"].split is None  # missing second_clip
    assert by["clip-3"].split is None  # non-dict


def test_callout_model_has_no_bbox_field() -> None:
    # Coordinates are compile-side only; an LLM-supplied bbox must have nowhere to
    # land on the plan model.
    assert "bbox" not in Callout.model_fields


def test_a_track_of_placeholder_keyframes_is_not_a_click_target() -> None:
    """merqi spike 01a0fc55: a non-empty track whose keyframes are all zero-filled
    placeholders / non-interactions has nothing to zoom at, so the line must plan
    zoom OFF exactly like a no-click line — not punch into empty space."""
    ctx = _ctx()
    record = ctx.prior_run_outputs["demo_video_record"]
    record["clips"][1]["focus_track"] = [
        {"t": 0.5, "bbox": {"x": 0, "y": 0, "w": 0, "h": 0}, "action": "click"},
        {"t": 0.9, "bbox": {"x": 10, "y": 20, "w": 100, "h": 40}, "action": "reveal"},
    ]
    lines = DemoVideoFootageDirectionStage(client=AsyncMock()).planner_lines(ctx)
    assert {line["line_id"]: line["has_click"] for line in lines} == {
        "clip-1": True,
        "clip-2": False,
    }
