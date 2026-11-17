"""An INFEASIBLE reconcile must land on the QUERYABLE channel, not only in prose.

MEASURED, 2026-09-11 — two runs of ``demo_video_full_chain_todomvc``, same 44.00s
order, same ``config_snapshot`` (input_hash ``e10808d4…``), both post-deploy, neither a
relief child:

===============  =========================  =========================
                 run ee67 (19:10)           run 5bf5 (19:38)
===============  =========================  =========================
recorded clips   12.32/10.64/17.72/12.16/   11.48/12.92/12.96/12.80/
                 29.92 = **82.76s**         11.40 = **61.56s**
reconcile verdict "INFEASIBLE — exceeds the  "INFEASIBLE — … by 17.56s"
                 44.00s target by 38.76s"
``degradations`` **[]**                     **[]**
delivered        68.075s (**+54.72%**)      42.600s (-3.18%)
===============  =========================  =========================

The stage diagnosed the overrun correctly BOTH times and recorded it nowhere a query
can reach: ``reconcile_timeline.degradations`` was empty on both. Note that ee67's
clip-5 ran 29.92s because a Playwright scenario FAILED (a 5000ms ``toBeChecked``
timeout) — wildly-over-plan footage is a REAL input, not a hypothetical, and the stage
has to behave sanely on it rather than assume well-behaved clips.

These tests pin what the stage now does with that finding. They do NOT pin that the
delivered film conforms: nothing here shortens anything, and the entry itself says so.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

import pytest

from src.services.runs.run_context import RunContext
from src.services.stages.demo_video.reconcile_timeline import ReconcileTimelineStage
from src.services.stages.shared.degradation import (
    DURATION_OUT_OF_TOLERANCE,
    FOOTAGE_TRIMMED,
    FREEZE_PAD_CEILING_EXCEEDED,
    NARRATION_UNSAYABLE,
    ORDERED_DURATION_INFEASIBLE,
    TIMELINE_OFF_TARGET,
    TIMELINE_UNDERFULL,
)

#: The five clip lengths ``demo_video_record`` measured on run ee67, in spine order.
#: Clip 5 is the failed-Playwright-scenario clip.
EE67_CLIP_SECONDS = [12.32, 10.64, 17.72, 12.16, 29.92]
#: 82.76s of footage.
EE67_FOOTAGE_TOTAL = sum(EE67_CLIP_SECONDS)
#: The order both runs were placed against.
ORDERED_SECONDS = 44.0
#: ``_TEXT`` (28 words) at the default 150 wpm — every beat's narration.
_NARRATION_SECONDS = 28 / 150.0 * 60.0
#: What ``trim_footage_tails`` leaves: each tail cut to max(narration, 60% clip, 2.0s),
#: never longer than the clip (clip-2's 10.64s runs SHORTER than its 11.2s narration, so
#: it is not trimmed at all). 11.2 + 10.64 + 11.2 + 11.2 + 17.952 = 62.192s.
EE67_TRIMMED_FLOOR = sum(min(c, max(_NARRATION_SECONDS, 0.6 * c, 2.0)) for c in EE67_CLIP_SECONDS)

_TEXT = (
    "The application loads and the user is shown the empty todo list with a single "
    "input field waiting for the first task to be typed in by hand."
)


def _ctx(
    *,
    clip_seconds: list[float],
    target: float = ORDERED_SECONDS,
    mode: str = "auto",
    per_beat_text_seconds: float = 8.0,
) -> RunContext:
    """A run whose ONLY duration evidence is ``demo_video_record``'s measured clips.

    Deliberately carries no ``structural_relief`` block and no bdd
    ``expected_recording_seconds``: this is the shape both live runs had, the one in
    which ``gen``'s feasibility floor returns early with no yield figure.
    """
    lines = [
        {
            "id": f"clip-{i + 1}",
            "text": _TEXT,
            "seconds": per_beat_text_seconds,
            "clip": f"clip-{i + 1}",
        }
        for i in range(len(clip_seconds))
    ]
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "direction": {"timing": {"mode": mode}},
            "deliverable": {"target_duration_seconds": target},
            "voice": {"voice_id": "af_heart"},
        },
        prior_run_outputs={
            "demo_video_gen": {
                "lines": lines,
                "bdd_job_id": "",
                "features_generated": 0,
                "package_url": "",
            },
            "demo_video_record": {
                "clips": [
                    {"id": f"clip-{i + 1}", "line_id": f"clip-{i + 1}", "seconds": s}
                    for i, s in enumerate(clip_seconds)
                ]
            },
        },
    )


def _infeasible_entries(degradations: list[Any]) -> list[Any]:
    return [d for d in degradations if d["code"] == ORDERED_DURATION_INFEASIBLE]


@pytest.mark.asyncio
async def test_the_ee67_shape_names_ordered_projected_and_overrun() -> None:
    """82.76s of measured footage against a 44.00s order ⇒ one loud, arithmetic entry.

    Footage tails are now TRIMMED first (owner ruling): 82.76s → 62.19s at the floors, which
    still overruns — so the infeasibility entry names the TRIMMED projection, and the cut and
    the residual ride their own codes.
    """
    stage = ReconcileTimelineStage()
    out = await stage.handle(stage.input_model(), _ctx(clip_seconds=EE67_CLIP_SECONDS))

    entries = _infeasible_entries(out.degradations)
    assert len(entries) == 1, f"expected exactly one infeasibility entry, got {out.degradations}"
    detail = entries[0]["detail"]

    # ORDERED — the number the customer asked for.
    assert "ordered 44.00s" in detail
    # PROJECTED — the beats' own floors: the measured clips with their tails trimmed.
    assert f"project {EE67_TRIMMED_FLOOR:.2f}s" in detail
    assert "62.19s" in detail
    # OVERRUN — the difference still left after trimming (38.76s before the trim).
    assert f"overrun of {EE67_TRIMMED_FLOOR - ORDERED_SECONDS:.2f}s" in detail
    assert "18.19s" in detail
    # WHAT IT DID — nothing was available, and the entry says which knob was off
    # rather than implying a relief pass ran.
    assert "no structural relief pass ran" in detail
    assert "relief.mode='off'" in detail
    # PROVENANCE — that the projection rests on footage this run actually measured.
    assert "5 of 5 beat(s) are pinned to footage demo_video_record actually measured" in detail
    assert "totalling 62.19s" in detail
    # The TRIM, with the measured total it started from…
    trimmed = [d for d in out.degradations if d["code"] == FOOTAGE_TRIMMED]
    assert len(trimmed) == 1, out.degradations
    # 82.76s → 62.19s overall = 20.57s cut, all of it from the four beats whose footage ran
    # past their narration (72.12s of them → 51.55s); clip-2 is not among them.
    assert "cut by 20.57s in total, 72.12s of measured footage -> 51.55s" in trimmed[0]["detail"]
    assert "clip-2" not in trimmed[0]["detail"]  # shorter than its narration: untouched
    # …and the residual the floors could not absorb.
    assert [d for d in out.degradations if d["code"] == DURATION_OUT_OF_TOLERANCE]


@pytest.mark.asyncio
async def test_the_overrun_is_detected_with_no_bdd_yield_anywhere() -> None:
    """No ``expected_recording_seconds``, no parent yield — measured clips still bound it.

    This is root cause 1 stated as a test: ``gen._apply_feasibility_floor`` returns early
    without a per-scenario yield figure, so on the bdd path NOTHING upstream bounds the
    plan. The context here supplies neither rung; the finding must still be raised, from
    ``demo_video_record``'s clips alone.
    """
    ctx = _ctx(clip_seconds=EE67_CLIP_SECONDS)
    assert "structural_relief" not in ctx.config_snapshot
    assert "expected_recording_seconds" not in str(ctx.prior_output("demo_video_gen"))

    stage = ReconcileTimelineStage()
    out = await stage.handle(stage.input_model(), ctx)

    assert _infeasible_entries(out.degradations)


@pytest.mark.asyncio
async def test_the_entry_is_logged_at_error(caplog: pytest.LogCaptureFixture) -> None:
    """A record nobody can see is not a record: the overrun reaches the operator log."""
    stage = ReconcileTimelineStage()
    with caplog.at_level(logging.ERROR, logger="src.services.stages.demo_video.reconcile_timeline"):
        await stage.handle(stage.input_model(), _ctx(clip_seconds=EE67_CLIP_SECONDS))

    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert any(f"overrun of {EE67_TRIMMED_FLOOR - ORDERED_SECONDS:.2f}s" in m for m in errors), (
        errors
    )


@pytest.mark.asyncio
async def test_a_plan_that_fits_the_order_degrades_nothing() -> None:
    """The no-change path. A feasible timeline emits NO entry on this channel.

    Same five-beat shape, clips that comfortably fit a 44.00s order. If this ever goes
    red the channel has become a signal that fires on everything, which is worth less
    than no signal at all.
    """
    fitting = [4.0, 4.0, 4.0, 4.0, 4.0]
    stage = ReconcileTimelineStage()
    out = await stage.handle(
        stage.input_model(),
        _ctx(clip_seconds=fitting, per_beat_text_seconds=6.0),
    )

    # No INFEASIBLE entry: the plan fits under the order. (It is UNDERFULL — 20s of
    # beats against 44s — and that note now carries its own coded twin rather than
    # reaching the run record as the executor's ``unclassified_note``; it raised
    # ``degraded`` through ``notes`` before that change too, so no run became newly
    # degraded. What must hold is that every coded entry is the twin of a note.)
    assert not _infeasible_entries(out.degradations)
    # Exact, not ``len(degradations) <= len(notes)`` — that held on an EMPTY list,
    # so it could not catch the coded twins vanishing. Measured on this case: four
    # notes, each with its coded twin, in note order.
    assert [d["code"] for d in out.degradations] == [
        TIMELINE_UNDERFULL,
        NARRATION_UNSAYABLE,
        TIMELINE_OFF_TARGET,
        FREEZE_PAD_CEILING_EXCEEDED,
    ]
    assert len(out.degradations) == len(out.notes)


@pytest.mark.asyncio
async def test_reconcile_only_records_the_finding_without_touching_the_spine() -> None:
    """Mutate nothing is about the SPINE — an inspection run still reports what it saw."""
    ctx = _ctx(clip_seconds=EE67_CLIP_SECONDS, mode="reconcile_only")
    gen_out = ctx.prior_output("demo_video_gen") or {}
    before = [dict(line) for line in gen_out["lines"]]

    stage = ReconcileTimelineStage()
    out = await stage.handle(stage.input_model(), ctx)

    assert _infeasible_entries(out.degradations)
    assert [line.model_dump()["seconds"] for line in out.lines] == [
        line["seconds"] for line in before
    ]


@pytest.mark.asyncio
async def test_an_upstream_degradation_is_not_clobbered() -> None:
    """The plan's own entries survive; this stage APPENDS, exactly as ``notes`` does."""
    ctx = _ctx(clip_seconds=EE67_CLIP_SECONDS)
    gen = ctx.prior_output("demo_video_gen") or {}
    gen["degradations"] = [{"code": "slides_stubbed", "detail": "deck came back empty"}]

    stage = ReconcileTimelineStage()
    out = await stage.handle(stage.input_model(), ctx)

    codes = [d["code"] for d in out.degradations]
    # Upstream first, then this stage's own entries — the INFEASIBLE finding leads them;
    # the coded twins of its other degrading notes follow.
    assert codes[:2] == ["slides_stubbed", ORDERED_DURATION_INFEASIBLE]
    assert codes.count(ORDERED_DURATION_INFEASIBLE) == 1
