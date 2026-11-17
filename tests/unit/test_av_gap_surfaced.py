"""4.4 — ``av_gap_seconds`` is surfaced on the run instead of being dropped.

The compiler MEASURES it on the real rendered file and emits it unconditionally on
every master; until now nothing in this repo read it, so the A/V-coverage verdict
rested on a hand-collected n=10 while the number rode in on every single run.

Surfaced exactly like ``static_frame_ratio``: evidence on the gestalt block, never a
gate term.
"""

from __future__ import annotations

from uuid import uuid4

from src.services.runs.run_context import RunContext
from src.services.stages.demo_video.final_critique import FinalCritiqueStage


def _ctx(compile_out: dict | None) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={},
        prior_run_outputs={"demo_video_compile": compile_out} if compile_out is not None else {},
    )


def _gap(compile_out: dict | None) -> float | None:
    return FinalCritiqueStage()._av_gap_seconds(_ctx(compile_out))


def test_the_measured_gap_is_read_off_the_mastering_report() -> None:
    assert _gap({"mastering": {"av_gap_seconds": 8.8}}) == 8.8
    assert _gap({"mastering": {"av_gap_seconds": 0.0}}) == 0.0
    assert _gap({"mastering": {"av_gap_seconds": 39.6}}) == 39.6


def test_an_int_gap_is_read_as_a_number() -> None:
    assert _gap({"mastering": {"av_gap_seconds": 3}}) == 3.0


def test_a_missing_report_or_key_is_none_not_a_fabricated_zero() -> None:
    """⛔ 0.0 would read as 'the picture fully covered the narration' — a QUALITY
    claim this repo never observed. Absence must stay absent."""
    assert _gap({"mastering": {}}) is None
    assert _gap({"mastering": None}) is None
    assert _gap({}) is None
    assert _gap(None) is None
    assert _gap({"mastering": {"av_gap_seconds": None}}) is None
    assert _gap({"mastering": {"av_gap_seconds": "8.8s"}}) is None


# ⛔ THE "IT RIDES ON THE GESTALT BLOCK" TEST DOES NOT LIVE HERE ANY MORE.
#
# It used to, and it was VACUOUS: it built the gestalt dict itself out of two reader
# calls and then asserted the two keys were in the dict it had just written. Deleting
# the surfacing code in `final_critique._reevaluate` — the only thing that actually puts
# `av_gap_seconds` on the block a run persists — left it green, so it covered nothing it
# claimed to.
#
# The real one drives the STAGE and reads `out.gestalt_summary`, so it needs that
# module's analyzer/WM harness. It lives beside its `static_frame_ratio` twin:
# `tests/services/stages/demo_video/test_final_critique.py`
# ::test_the_measured_av_gap_is_surfaced_on_the_persisted_gestalt_block.
#
# What remains in this file is the READER (`_av_gap_seconds`) — real production code,
# every case reachable, and each assertion fails when the reader is reverted.
