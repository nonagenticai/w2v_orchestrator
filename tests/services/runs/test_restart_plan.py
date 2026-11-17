"""The restart planner as a pure function: one rule per test, a tiny DAG.

a ──► b ──► d
└───► c
e  (independent)
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass

import pytest

from src.services.blueprints.base import DAGBlueprint, StageSpec
from src.services.runs import restart_plan
from src.services.runs.restart_plan import compute_restart_plan
from src.services.runs.resume_inputs import RESUME_INPUTS_KEY

_T0 = datetime.datetime(2026, 10, 27, 12, 0, tzinfo=datetime.UTC)


@dataclass
class _Row:
    stage_name: str
    status: str = "completed"
    output: object = None
    completed_at: datetime.datetime | None = _T0
    edited_at: datetime.datetime | None = None
    config_reads: list[str] | None = None

    def __post_init__(self) -> None:
        if self.output is None and self.status == "completed":
            self.output = {"ok": True}


def _bp() -> DAGBlueprint:
    def h(_deps):  # never called
        raise AssertionError

    return DAGBlueprint(
        id="plan_test_v1",
        project_type="generic",
        stages=[
            StageSpec(name="a", handler_factory=h),
            StageSpec(name="b", depends_on=("a",), handler_factory=h),
            StageSpec(name="c", depends_on=("a",), handler_factory=h),
            StageSpec(name="d", depends_on=("b",), handler_factory=h),
            StageSpec(name="e", handler_factory=h),
        ],
    )


@pytest.fixture(autouse=True)
def _resumable(monkeypatch) -> None:
    monkeypatch.setattr(
        restart_plan,
        "RESUMABLE_BLUEPRINTS",
        frozenset({*restart_plan.RESUMABLE_BLUEPRINTS, "plan_test_v1"}),
    )


def _snap(**extra: object) -> dict[str, object]:
    return {RESUME_INPUTS_KEY: {"version": 1, "unresumable": []}, "brief": {"x": 1}, **extra}


def _plan(rows, *, status="cancelled", parent=None, would_be=None, annotated=()):
    return compute_restart_plan(
        blueprint=_bp(),
        parent_status=status,
        rows=rows,
        parent_snapshot=parent if parent is not None else _snap(),
        would_be_snapshot=would_be if would_be is not None else _snap(),
        annotated_stages=set(annotated),
    )


def _actions(plan) -> dict[str, str]:
    return {s.stage_name: s.action for s in plan.stages}


def test_cancelled_mid_flight_stage_and_its_dependents_re_run_the_rest_is_reused() -> None:
    plan = _plan(
        [_Row("a"), _Row("b", status="cancelled", output={"half": 1}), _Row("c"), _Row("e")]
    )
    assert plan.mode == "partial"
    assert _actions(plan) == {"a": "reuse", "b": "rerun", "c": "reuse", "d": "rerun", "e": "reuse"}
    by = {s.stage_name: s for s in plan.stages}
    assert by["b"].reasons[0].code == "cancelled"
    assert by["d"].reasons[0].code in {"not_run", "upstream_rerun"}


def test_an_edit_keeps_the_stage_and_re_runs_only_its_dependents() -> None:
    later = _T0 + datetime.timedelta(minutes=5)
    rows = [_Row("a", edited_at=later), _Row("b"), _Row("c"), _Row("d"), _Row("e")]
    plan = _plan(rows)
    assert _actions(plan) == {"a": "reuse", "b": "rerun", "c": "rerun", "d": "rerun", "e": "reuse"}
    assert {s.stage_name: s for s in plan.stages}["a"].reasons[0].code == "edit_kept"
    assert plan.edited_stages == {"a"}
    assert plan.rerun_roots == frozenset()


def test_an_edit_made_before_the_stage_last_completed_is_not_an_edit() -> None:
    earlier = _T0 - datetime.timedelta(minutes=5)
    rows = [_Row(n) for n in "bcde"] + [_Row("a", edited_at=earlier)]
    plan = _plan(rows, status="completed")
    assert plan.mode == "full"
    assert plan.full_reasons[0].code == "nothing_changed"


def test_an_edited_stage_that_must_re_run_anyway_reports_the_edit_is_replaced() -> None:
    later = _T0 + datetime.timedelta(minutes=5)
    rows = [
        _Row("a", config_reads=["voice"]),
        _Row("b", edited_at=later, config_reads=[]),
        _Row("c", config_reads=[]),
        _Row("d", config_reads=[]),
        _Row("e", config_reads=[]),
    ]
    plan = _plan(rows, would_be=_snap(voice={"v": 2}))
    b = {s.stage_name: s for s in plan.stages}["b"]
    assert b.action == "rerun"
    assert [r.code for r in b.reasons] == ["upstream_rerun", "edit_replaced"]


def test_only_recorded_readers_of_a_changed_input_re_run() -> None:
    rows = [
        _Row("a", config_reads=["brief"]),
        _Row("b", config_reads=["voice"]),
        _Row("c", config_reads=["brief"]),
        _Row("d", config_reads=[]),
        _Row("e", config_reads=["brand"]),
    ]
    plan = _plan(rows, parent=_snap(voice={"v": 1}), would_be=_snap(voice={"v": 2}))
    assert _actions(plan) == {"a": "reuse", "b": "rerun", "c": "reuse", "d": "rerun", "e": "reuse"}
    assert plan.changed_inputs == ("voice",)
    b = {s.stage_name: s for s in plan.stages}["b"]
    assert b.reasons[0].code == "input_changed"
    assert b.reasons[0].inputs == ("voice",)


def test_an_unrecorded_or_wholesale_reader_re_runs_on_any_input_change() -> None:
    rows = [
        _Row("a", config_reads=["brief"]),
        _Row("b", config_reads=None),
        _Row("c", config_reads=["*"]),
        _Row("d", config_reads=[]),
        _Row("e", config_reads=["brief"]),
    ]
    plan = _plan(rows, would_be=_snap(theme={"t": 1}))
    by = {s.stage_name: s for s in plan.stages}
    assert by["b"].reasons[0].code == "inputs_unrecorded"
    assert by["c"].reasons[0].code == "reads_everything"
    assert by["a"].action == "reuse" and by["e"].action == "reuse"


def test_run_scoped_keys_are_not_input_changes() -> None:
    rows = [_Row(n, config_reads=None) for n in "abcde"]
    plan = _plan(
        rows,
        status="completed",
        parent=_snap(subject_id="u1", license_id="l1"),
        would_be=_snap(subject_id="u2", license_id="l2"),
    )
    assert plan.full_reasons[0].code == "nothing_changed"


def test_a_skipped_stage_re_evaluates_without_dragging_its_dependents() -> None:
    rows = [
        _Row("a"),
        _Row("b", status="skipped"),
        _Row("c"),
        _Row("d"),
        _Row("e", status="failed"),
    ]
    plan = _plan(rows, status="failed")
    acts = _actions(plan)
    assert acts["b"] == "rerun"  # no output to inherit — re-checks its own gate
    assert acts["d"] == "reuse"  # not dragged along by a skip
    assert acts["e"] == "rerun"


def test_a_skip_whose_input_changed_re_runs_its_dependents() -> None:
    """AI images were OFF (the stage skipped); the user switches them ON: the stage
    now produces output, so everything built without it must re-run."""
    rows = [
        _Row("a", config_reads=[]),
        _Row("b", status="skipped", config_reads=["ai_image_gen"]),
        _Row("c", config_reads=[]),
        _Row("d", config_reads=[]),
        _Row("e", config_reads=[]),
    ]
    plan = _plan(rows, status="completed", would_be=_snap(ai_image_gen={"on": True}))
    assert _actions(plan) == {"a": "reuse", "b": "rerun", "c": "reuse", "d": "rerun", "e": "reuse"}


def test_open_feedback_re_runs_the_stage_and_its_dependents() -> None:
    rows = [_Row(n) for n in "abcde"]
    plan = _plan(rows, status="completed", annotated={"b"})
    assert _actions(plan) == {"a": "reuse", "b": "rerun", "c": "reuse", "d": "rerun", "e": "reuse"}


def test_a_blueprint_off_the_allowlist_is_a_full_re_run(monkeypatch) -> None:
    monkeypatch.setattr(restart_plan, "RESUMABLE_BLUEPRINTS", frozenset())
    plan = _plan([_Row(n) for n in "abcde"])
    assert plan.mode == "full"
    assert plan.full_reasons[0].code == "not_resumable"


def test_a_plan_that_would_reuse_nothing_is_a_full_re_run() -> None:
    plan = _plan([_Row("a", status="cancelled")], status="cancelled")
    assert plan.mode == "full"
    assert plan.full_reasons[0].code == "nothing_reusable"


def test_a_failed_run_whose_every_step_finished_re_runs_everything() -> None:
    """The failure came AFTER the DAG (e.g. the final artifact was missing): copying
    every finished step into the child would just reproduce it."""
    plan = _plan([_Row(n) for n in "abcde"], status="failed")
    assert plan.mode == "full"
    assert plan.full_reasons[0].code == "all_steps_finished"


def test_a_paused_parent_is_never_planned() -> None:
    plan = _plan([_Row(n) for n in "abcde"], status="paused")
    assert plan.mode == "full"
    assert plan.full_reasons[0].code == "not_plannable"


def test_a_changed_live_input_re_runs_its_live_reader(monkeypatch) -> None:
    from src.services.runs.resume_inputs import LiveInput

    monkeypatch.setattr(
        restart_plan, "STAGE_LIVE_INPUTS", {"c": frozenset({LiveInput.VIDEO_SOURCES})}
    )
    parent = _snap()
    parent[RESUME_INPUTS_KEY] = {"version": 1, "unresumable": [], "video_sources": [["1"]]}
    would = _snap()
    would[RESUME_INPUTS_KEY] = {"version": 1, "unresumable": [], "video_sources": [["1"], ["2"]]}
    plan = _plan(
        [_Row(n, config_reads=[]) for n in "abcde"],
        status="completed",
        parent=parent,
        would_be=would,
    )
    assert _actions(plan) == {"a": "reuse", "b": "reuse", "c": "rerun", "d": "reuse", "e": "reuse"}
    assert plan.changed_inputs == ("live:video_sources",)
