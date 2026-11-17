"""ORCH-3.3 shot-plan consumption in DemoVideoCompileStage.build_shots.

The plan is INTENT only: a zoom-OFF co-removes its focus+cursor tracks, a static
reframe drops the cursor, and a callout is placed ONLY at the bbox captured in the
recording's focus track — never at a coordinate the LLM invented. Absent (stage
skipped ⇒ ``{}``) the whole seam is inert and the shot list is byte-identical.

Pure + deterministic: a fake ``ctx`` carries a ``config_snapshot`` + ``prior_output``,
and the stage is built via ``__new__`` (``build_shots`` does not use ``self``).
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any, cast

from src.services.blueprints.demo_video import DEMO_VIDEO_BLUEPRINT
from src.services.blueprints.showcase import SHOWCASE_BLUEPRINT
from src.services.stages.demo_video.compile import DemoVideoCompileStage
from tests.support.geometry_wire import bbox_wire, cursor_wire, focus_wire

_BBOX = {"x": 10, "y": 20, "w": 100, "h": 40}
_FOCUS: list[dict[str, object]] = [{"t": 0.5, "bbox": _BBOX, "action": "click"}]
_CURSOR: list[dict[str, object]] = [{"t": 0.0, "x": 5, "y": 5}, {"t": 0.5, "x": 60, "y": 40}]
# The same geometry as it lands on the compile wire: bbox as the declared 4-tuple,
# the recorder-only ``action`` key dropped, callouts as an ARRAY. See
# tests/contracts/test_directed_footage_geometry_contract.py.
_BBOX_WIRE = bbox_wire(_BBOX)
_FOCUS_WIRE = focus_wire(_FOCUS)
_CURSOR_WIRE = cursor_wire(_CURSOR)
_FOOTAGE = {"footage": {"auto_zoom": True}}

_GEN = {
    "lines": [{"id": "clip-1", "text": "a"}, {"id": "clip-2", "text": "b"}],
    "bdd_job_id": "job-1",
}
_NARRATE = {
    "clips": [
        {"line_id": "clip-1", "artifact_url": "http://a/1.mp3", "seconds": 4.0},
        {"line_id": "clip-2", "artifact_url": "http://a/2.mp3", "seconds": 3.0},
    ]
}


def _record(*, clip2_focus: bool = True) -> dict[str, Any]:
    clip2: dict[str, Any] = {"line_id": "clip-2", "video_url": "http://v/2.mp4", "seconds": 3.0}
    if clip2_focus:
        clip2["focus_track"] = copy.deepcopy(_FOCUS)
        clip2["cursor_track"] = copy.deepcopy(_CURSOR)
    return {
        "clips": [
            {
                "line_id": "clip-1",
                "video_url": "http://v/1.mp4",
                "seconds": 4.0,
                "focus_track": copy.deepcopy(_FOCUS),
                "cursor_track": copy.deepcopy(_CURSOR),
            },
            clip2,
        ]
    }


def _stage() -> DemoVideoCompileStage:
    return DemoVideoCompileStage.__new__(DemoVideoCompileStage)


def _ctx(
    *,
    plans: list[dict[str, Any]] | None = None,
    critique_plans: list[dict[str, Any]] | None = None,
    gen: dict[str, Any] | None = None,
    record: dict[str, Any] | None = None,
    include_stage: bool = True,
) -> Any:
    outputs: dict[str, Any] = {
        "demo_video_gen": copy.deepcopy(gen or _GEN),
        "demo_video_narrate": copy.deepcopy(_NARRATE),
        "demo_video_record": copy.deepcopy(record or _record()),
    }
    if include_stage:
        outputs["demo_video_footage_direction"] = {"plans": plans or []}
    # S20: the footage-critique persists its nudged plans as `footage_plans`; compile
    # PREFERS them (truthy) over the base direction plans. Absent unless a test opts in.
    if critique_plans is not None:
        outputs["demo_video_footage_critique"] = {"footage_plans": critique_plans}
    return SimpleNamespace(
        config_snapshot={"direction": copy.deepcopy(_FOOTAGE)},
        prior_output=lambda stage: outputs.get(stage),
    )


def _by_line(shots: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {s["line_id"]: s for s in shots}


def test_zoom_off_co_removes_focus_and_cursor_and_no_callout() -> None:
    ctx = _ctx(
        plans=[
            {"line_id": "clip-1", "zoom": "off", "callout": {"type": "badge", "label": "x"}},
            {"line_id": "clip-2", "zoom": "auto"},
        ]
    )
    by = _by_line(_stage().build_shots(ctx))
    # clip-1 zoom off: focus + cursor removed, no callout.
    assert "focus_track" not in by["clip-1"]
    assert "cursor_track" not in by["clip-1"]
    assert "callouts" not in by["clip-1"]
    # clip-2 untouched: capture intact.
    assert by["clip-2"]["focus_track"] == _FOCUS_WIRE
    assert by["clip-2"]["cursor_track"] == _CURSOR_WIRE


def test_static_reframe_drops_only_the_cursor() -> None:
    ctx = _ctx(plans=[{"line_id": "clip-1", "zoom": "auto", "reframe": "static"}])
    by = _by_line(_stage().build_shots(ctx))
    assert by["clip-1"]["focus_track"] == _FOCUS_WIRE  # zoom target kept
    assert "cursor_track" not in by["clip-1"]  # cursor dropped
    # clip-2 has no plan → both tracks intact.
    assert by["clip-2"]["cursor_track"] == _CURSOR_WIRE


def test_callout_uses_captured_bbox_and_drops_when_absent() -> None:
    ctx = _ctx(
        plans=[
            {"line_id": "clip-1", "zoom": "auto", "callout": {"type": "badge", "label": "Click"}},
            {"line_id": "clip-2", "zoom": "auto", "callout": {"type": "label", "label": "Nope"}},
        ],
        record=_record(clip2_focus=False),  # clip-2 captured no focus → no bbox
    )
    by = _by_line(_stage().build_shots(ctx))
    assert by["clip-1"]["callouts"] == [
        {"type": "badge", "label": "Click", "at": 0.5, "bbox": _BBOX_WIRE}
    ]
    # No captured bbox for clip-2 ⇒ callout dropped (never placed at a guess).
    assert "callouts" not in by["clip-2"]


def test_absent_plan_is_byte_identical_and_emits_no_callouts() -> None:
    # A skipped stage stores `{}` (no plans); a stage that never ran returns None.
    # Both leave the captured tracks untouched and emit no callouts key.
    skipped = _stage().build_shots(_ctx(plans=[]))
    never_ran = _stage().build_shots(_ctx(include_stage=False))
    assert skipped == never_ran
    for shots in (skipped, never_ran):
        assert all("callouts" not in s for s in shots)
        by = _by_line(shots)
        # Capture rides through unchanged (the seam is inert).
        assert by["clip-1"]["focus_track"] == _FOCUS_WIRE
        assert by["clip-1"]["cursor_track"] == _CURSOR_WIRE


# ── S20: compile prefers the footage-critique's persisted plans on a re-compile ──
def test_compile_prefers_critique_footage_plans_over_base_direction() -> None:
    # Base direction turned clip-1's zoom OFF (would strip focus+cursor); the
    # footage-critique's persisted plans turn it back ON with a callout. On a
    # re-compile (encode_master) compile must PREFER the critique plans, so clip-1
    # keeps its capture AND gains the nudged callout (compose: footage + mastering).
    ctx = _ctx(
        plans=[{"line_id": "clip-1", "zoom": "off"}],
        critique_plans=[
            {"line_id": "clip-1", "zoom": "auto", "callout": {"type": "badge", "label": "Go"}},
        ],
    )
    by = _by_line(_stage().build_shots(ctx))
    assert by["clip-1"]["focus_track"] == _FOCUS_WIRE  # critique zoom=auto wins over base off
    assert by["clip-1"]["callouts"][0]["label"] == "Go"  # nudged callout applied
    assert by["clip-1"]["callouts"][0]["bbox"] == _BBOX_WIRE  # bbox still from the capture


def test_compile_falls_back_to_base_when_critique_plans_empty() -> None:
    # Empty footage_plans (no nudge / critique skipped-clean) is FALSY ⇒ compile falls
    # back to the base direction plans (TRUTHY check, not `is not None`).
    base = _by_line(_stage().build_shots(_ctx(plans=[{"line_id": "clip-1", "zoom": "off"}])))
    empty_crit = _by_line(
        _stage().build_shots(_ctx(plans=[{"line_id": "clip-1", "zoom": "off"}], critique_plans=[]))
    )
    assert "focus_track" not in base["clip-1"]  # base zoom=off stripped the capture
    assert base == empty_crit  # empty override changes nothing


def test_compile_absent_critique_output_is_byte_identical_to_first_compile() -> None:
    # No-op byte-identity: the FIRST compile runs BEFORE footage_critique in the DAG,
    # so its critique output is absent ⇒ base plans; an empty persisted override lands
    # on the same base plans ⇒ the resolved shots are identical.
    absent = _stage().build_shots(_ctx(plans=[{"line_id": "clip-1", "zoom": "auto"}]))
    empty_crit = _stage().build_shots(
        _ctx(plans=[{"line_id": "clip-1", "zoom": "auto"}], critique_plans=[])
    )
    assert absent == empty_crit


def test_clip_keying_fans_plan_and_bbox_onto_shared_lines() -> None:
    # Three lines all play over the ONE recorded clip-1: each inherits the plan and
    # the same captured bbox.
    gen = {
        "lines": [
            {"id": "a", "clip": "clip-1", "text": "one"},
            {"id": "b", "clip": "clip-1", "text": "two"},
            {"id": "c", "clip": "clip-1", "text": "three"},
        ],
        "bdd_job_id": "job-1",
    }
    record = {
        "clips": [
            {
                "line_id": "clip-1",
                "video_url": "http://v/1.mp4",
                "seconds": 4.0,
                "focus_track": copy.deepcopy(_FOCUS),
                "cursor_track": copy.deepcopy(_CURSOR),
            }
        ]
    }
    narrate = {
        "clips": [
            {"line_id": lid, "artifact_url": f"http://a/{lid}.mp3", "seconds": 2.0}
            for lid in ("a", "b", "c")
        ]
    }
    ctx = SimpleNamespace(
        config_snapshot={"direction": copy.deepcopy(_FOOTAGE)},
        prior_output=lambda stage: {
            "demo_video_gen": copy.deepcopy(gen),
            "demo_video_narrate": copy.deepcopy(narrate),
            "demo_video_record": copy.deepcopy(record),
            "demo_video_footage_direction": {
                "plans": [
                    {"line_id": lid, "zoom": "auto", "callout": {"type": "badge", "label": lid}}
                    for lid in ("a", "b", "c")
                ]
            },
        }.get(stage),
    )
    by = _by_line(_stage().build_shots(cast(Any, ctx)))
    for lid in ("a", "b", "c"):
        assert by[lid]["callouts"][0]["bbox"] == _BBOX_WIRE
        assert by[lid]["callouts"][0]["label"] == lid
        assert by[lid]["focus_track"] == _FOCUS_WIRE


def test_split_resolves_second_clip_to_its_video_url() -> None:
    # A split names another line_id; compile resolves it to that line's footage URL
    # (via video_by_line) and stamps {layout, second_source_url} on the shot.
    ctx = _ctx(
        plans=[
            {
                "line_id": "clip-1",
                "zoom": "auto",
                "split": {"layout": "h", "second_clip": "clip-2"},
            },
        ]
    )
    by = _by_line(_stage().build_shots(ctx))
    assert by["clip-1"]["split"] == {"layout": "h", "second_source_url": "http://v/2.mp4"}
    assert "split" not in by["clip-2"]


def test_split_dangling_ref_dropped() -> None:
    # second_clip names no footage-bearing line ⇒ no fetchable URL ⇒ no split key.
    ctx = _ctx(
        plans=[
            {
                "line_id": "clip-1",
                "zoom": "auto",
                "split": {"layout": "h", "second_clip": "ghost"},
            },
        ]
    )
    by = _by_line(_stage().build_shots(ctx))
    assert "split" not in by["clip-1"]


def test_split_self_ref_dropped() -> None:
    # A clip beside itself is meaningless ⇒ dropped.
    ctx = _ctx(
        plans=[
            {
                "line_id": "clip-1",
                "zoom": "auto",
                "split": {"layout": "h", "second_clip": "clip-1"},
            },
        ]
    )
    by = _by_line(_stage().build_shots(ctx))
    assert "split" not in by["clip-1"]


def test_split_omitted_on_non_footage_line() -> None:
    # A split authored on a card line (no video_url) never rides the shot — the
    # footage-gate in _attach_clip_framing drops it even though the ref resolves.
    gen = {
        "lines": [{"id": "clip-1", "text": "a"}, {"id": "card-1", "text": "intro card"}],
        "bdd_job_id": "job-1",
    }
    record = {
        "clips": [
            {
                "line_id": "clip-1",
                "video_url": "http://v/1.mp4",
                "seconds": 4.0,
                "focus_track": copy.deepcopy(_FOCUS),
                "cursor_track": copy.deepcopy(_CURSOR),
            }
        ]
    }
    ctx = SimpleNamespace(
        config_snapshot={"direction": copy.deepcopy(_FOOTAGE)},
        prior_output=lambda stage: {
            "demo_video_gen": copy.deepcopy(gen),
            "demo_video_narrate": copy.deepcopy(_NARRATE),
            "demo_video_record": copy.deepcopy(record),
            "demo_video_footage_direction": {
                "plans": [
                    {
                        "line_id": "card-1",
                        "zoom": "auto",
                        "split": {"layout": "v", "second_clip": "clip-1"},
                    },
                ]
            },
        }.get(stage),
    )
    by = _by_line(_stage().build_shots(cast(Any, ctx)))
    assert "split" not in by["card-1"]  # not footage → split never rides


def test_footage_direction_depends_on_record_in_both_blueprints() -> None:
    for bp in (DEMO_VIDEO_BLUEPRINT, SHOWCASE_BLUEPRINT):
        spec = bp.stage("demo_video_footage_direction")
        assert "demo_video_record" in spec.depends_on, bp.id
        # …and compile consumes it.
        assert "demo_video_footage_direction" in bp.stage("demo_video_compile").depends_on, bp.id


# --- A11 momentum consume: shot_timeline opt-in gate + lift ------------------


def _gate_ctx(config_snapshot: dict[str, Any]) -> Any:
    return SimpleNamespace(config_snapshot=config_snapshot)


def test_emit_shot_timeline_true_when_ship_gate_active() -> None:
    # Present + non-empty + mode != "off" ⇒ opt in so final_critique's gate reads it.
    assert _stage()._emit_shot_timeline(_gate_ctx({"ship_gate": {"mode": "auto"}})) is True
    assert _stage()._emit_shot_timeline(_gate_ctx({"ship_gate": {"mode": "critique_only"}})) is True


def test_emit_shot_timeline_false_when_gate_absent_empty_or_off() -> None:
    # No ship_gate key, an empty block, and an explicit "off" mode are all probe-free.
    assert _stage()._emit_shot_timeline(_gate_ctx({})) is False
    assert _stage()._emit_shot_timeline(_gate_ctx({"ship_gate": {}})) is False
    assert _stage()._emit_shot_timeline(_gate_ctx({"ship_gate": {"mode": "off"}})) is False


def test_build_output_lifts_shot_timeline() -> None:
    timeline = [{"line_id": "clip-1", "start": 0.0, "end": 5.0}]
    out = _stage().build_output(
        {
            "artifact_url": "s3://x",
            "duration_seconds": 10.0,
            "shot_timeline": timeline,
        },
        _gate_ctx({}),
    )
    assert out.shot_timeline == timeline


def test_build_output_shot_timeline_none_when_absent_or_non_list() -> None:
    # Absent ⇒ None (probe-free compile); a non-list value is defensively coerced to None.
    absent = _stage().build_output(
        {"artifact_url": "s3://x", "duration_seconds": 10.0},
        _gate_ctx({}),
    )
    non_list = _stage().build_output(
        {"artifact_url": "s3://x", "duration_seconds": 10.0, "shot_timeline": "nope"},
        _gate_ctx({}),
    )
    assert absent.shot_timeline is None
    assert non_list.shot_timeline is None
