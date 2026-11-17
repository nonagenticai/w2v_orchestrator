"""Pure unit tests for the ORCH-0.8 accessibility Tier-1 resolver.

No DB, no client — every assertion is a total function of literal inputs. The
central invariant under test is INERT-by-default: absent an `accessibility`
policy, nothing is stamped and the shot dicts are byte-identical to before.
"""

from __future__ import annotations

from typing import Any

from src.services.stages.shared.accessibility import (
    A11yPolicy,
    a11y_report,
    build_a11y_policy,
    choose_backplate,
    silent_gaps,
    text_region_geometry,
)
from src.services.stages.shot_assembly import assemble_shots


def _policy(**kw: Any) -> A11yPolicy:
    return A11yPolicy(**kw)


# ------------------------------------------------------------- choose_backplate
def test_choose_backplate_none_when_passing() -> None:
    # #000 on #fff is 21:1 — well above the AA floor, so no backplate.
    assert choose_backplate("#000000", "#ffffff", _policy()) is None


def test_choose_backplate_fill_below_floor() -> None:
    # Low-contrast pair (light grey on white) fails AA → a backplate fill is returned.
    fill = choose_backplate("#bbbbbb", "#ffffff", _policy())
    assert fill is not None
    assert fill.startswith("rgba(")


# ----------------------------------------------------------------- a11y_report


def test_a11y_report_flags_failing_region() -> None:
    shots = [{"line_id": "l1", "text_color": "#bbbbbb", "bg_color": "#ffffff"}]
    report = a11y_report(shots, [], _policy())
    assert report.failing_line_ids == ["l1"]
    assert "l1" in report.backplates
    assert report.insights and "l1" in report.insights[0]


# ----------------------------------------------------- assemble_shots INERT case
def _one_shot() -> list[dict[str, object]]:
    return [{"id": "s1", "line_id": "l1", "seconds": 2.0, "description": "hi"}]


def test_assemble_shots_inert_without_policy() -> None:
    shots = assemble_shots(
        storyboard_shots=_one_shot(),
        images=[],
        narration_clips=[],
    )
    built = shots[0]
    assert "contrast_backplate" not in built
    assert "a11y_exempt" not in built
    assert "flash_exempt" not in built


def test_assemble_shots_low_contrast_with_policy() -> None:
    story = [
        {
            "id": "s1",
            "line_id": "l1",
            "seconds": 2.0,
            "text_color": "#bbbbbb",
            "bg_color": "#ffffff",
        }
    ]
    shots = assemble_shots(
        storyboard_shots=story,
        images=[],
        narration_clips=[],
        a11y_policy=_policy(),
    )
    built = shots[0]
    assert built["contrast_backplate"].startswith("rgba(")
    assert built["a11y_exempt"] is False
    assert built["flash_exempt"] is False


def test_assemble_shots_passing_contrast_flags_only() -> None:
    # A passing region with a policy present: exempt flags stamped, NO backplate key.
    story = [
        {
            "id": "s1",
            "line_id": "l1",
            "seconds": 2.0,
            "text_color": "#000000",
            "bg_color": "#ffffff",
        }
    ]
    built = assemble_shots(
        storyboard_shots=story,
        images=[],
        narration_clips=[],
        a11y_policy=_policy(),
    )[0]
    assert "contrast_backplate" not in built
    assert built["a11y_exempt"] is False
    assert built["flash_exempt"] is False


# ---------------------------------------------------------- text_region_geometry
def test_text_region_geometry_deterministic() -> None:
    shot = {"line_id": "l1"}
    a = text_region_geometry(shot)
    b = text_region_geometry(shot)
    assert a == b
    assert a.line_id == "l1"
    assert a.text_color == "#ffffff"
    assert a.bg_color == "#000000"


# ------------------------------------------------------------------ silent_gaps
def test_silent_gaps_threshold_boundary() -> None:
    # gap between end=1000ms and start=1170ms is exactly 170ms → included at boundary.
    clips_at = [
        {"line_id": "a", "start_ms": 0, "end_ms": 1000},
        {"line_id": "b", "start_ms": 1170, "end_ms": 2000},
    ]
    gaps = silent_gaps(clips_at, min_gap_ms=170)
    assert len(gaps) == 1
    assert gaps[0]["gap_ms"] == 170
    assert gaps[0]["after_line_id"] == "a"

    # 169ms gap is below the bound → excluded.
    clips_below = [
        {"line_id": "a", "start_ms": 0, "end_ms": 1000},
        {"line_id": "b", "start_ms": 1169, "end_ms": 2000},
    ]
    assert silent_gaps(clips_below, min_gap_ms=170) == []


def test_silent_gaps_untimed_clips_empty() -> None:
    assert silent_gaps([{"line_id": "a"}, {"line_id": "b"}], min_gap_ms=170) == []


def test_silent_gaps_seconds_keys() -> None:
    clips = [
        {"line_id": "a", "start": 0.0, "end": 1.0},
        {"line_id": "b", "start": 1.5, "end": 2.0},
    ]
    gaps = silent_gaps(clips, min_gap_ms=170)
    assert len(gaps) == 1
    assert gaps[0]["gap_ms"] == 500


# --------------------------------------------------------------- build_a11y_policy
def test_build_a11y_policy_absent_returns_none() -> None:
    assert build_a11y_policy({}) is None
    assert build_a11y_policy({"accessibility": "nope"}) is None


def test_build_a11y_policy_present_reads_knobs() -> None:
    policy = build_a11y_policy(
        {
            "accessibility": {
                "wcag_level": "AAA",
                "large_text": True,
                "min_ratio": 7.0,
                "backplate_opacity": 0.8,
                "backplate_color": "#101010",
                "min_silent_gap_ms": 200,
            }
        }
    )
    assert policy is not None
    assert policy.wcag_level == "AAA"
    assert policy.large_text is True
    assert policy.min_ratio == 7.0
    assert policy.backplate_opacity == 0.8
    assert policy.backplate_color == "#101010"
    assert policy.min_silent_gap_ms == 200
