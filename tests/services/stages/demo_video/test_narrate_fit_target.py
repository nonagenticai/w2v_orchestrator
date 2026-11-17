"""demo_video_narrate.fit_target_for — which lines carry an allotment.

`reconcile_timeline` stamps `seconds` (the allotment) alongside `vo_rate` (the
words-per-minute ESTIMATE `speed_for` consumes). Returning the allotment here
arms the base's measured-duration correction. The same care `speed_for` takes
applies: a footage/card beat, or a run where reconcile_timeline was skipped,
carries no `seconds`, and None means "accept what TTS returns" — today's
behavior exactly.
"""

from __future__ import annotations

from uuid import uuid4

from src.services.runs.run_context import RunContext
from src.services.stages.demo_video.narrate import DemoVideoNarrateStage


def _ctx() -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={},
        prior_run_outputs={},
    )


def test_stamped_allotment_is_the_fit_target() -> None:
    stage = DemoVideoNarrateStage()
    assert stage.fit_target_for({"id": "clip-1", "seconds": 6.5}, _ctx()) == 6.5


def test_integer_allotment_is_coerced_to_float() -> None:
    stage = DemoVideoNarrateStage()
    target = stage.fit_target_for({"id": "clip-1", "seconds": 7}, _ctx())
    assert target == 7.0
    assert isinstance(target, float)


def test_a_beat_with_no_allotment_opts_out() -> None:
    """A footage/card beat, or a skipped reconcile_timeline: no `seconds` at all."""
    stage = DemoVideoNarrateStage()
    assert stage.fit_target_for({"id": "clip-1", "text": "Hi."}, _ctx()) is None


def test_non_positive_and_unparseable_allotments_opt_out() -> None:
    """Zero, negative and junk are meaningless as an allotment — they must not
    reach the correction arithmetic (which divides by the target)."""
    stage = DemoVideoNarrateStage()
    assert stage.fit_target_for({"id": "c", "seconds": 0}, _ctx()) is None
    assert stage.fit_target_for({"id": "c", "seconds": -3.0}, _ctx()) is None
    assert stage.fit_target_for({"id": "c", "seconds": None}, _ctx()) is None
    assert stage.fit_target_for({"id": "c", "seconds": "soon"}, _ctx()) is None


# ══ THE FREEZE PAD IS PART OF THE WINDOW THE RATE WAS CHOSEN AGAINST ══════════
#
# `solve_timeline` picks a FOOTAGE beat's `vo_rate` against
# `allotted + freeze_pad_seconds`: the clip may hold its last frame frozen for up to
# `DEFAULT_FREEZE_PAD_CEILING`, and the narration is deliberately allowed to use that
# extension rather than over-compress into the bare clip. Fitting to `seconds` alone
# re-synthesized the line SLOWER than the solver intended, by up to the whole ceiling on
# every padded beat — a second, independent slow-narration lever, firing precisely on the
# footage beats demo_video is made of.


def test_the_solver_granted_freeze_pad_is_added_to_the_fit_target() -> None:
    stage = DemoVideoNarrateStage()
    target = stage.fit_target_for(
        {"id": "clip-1", "seconds": 6.5, "freeze_pad_seconds": 1.5}, _ctx()
    )
    assert target == 8.0
    # ⛔ And it is NOT the bare allotment — the whole defect was returning that.
    assert target != 6.5


def test_the_pad_is_bounded_by_what_the_solver_actually_granted() -> None:
    """Not a blanket ceiling: each beat gets ITS pad, and most get a small one or none.
    A test that only ever used the 1.5s ceiling could not tell the two apart."""
    stage = DemoVideoNarrateStage()
    assert (
        stage.fit_target_for({"id": "c", "seconds": 10.0, "freeze_pad_seconds": 0.4}, _ctx())
        == 10.4
    )


def test_a_beat_with_no_pad_is_byte_identical_to_before() -> None:
    """Non-footage / already-fits beats are stamped 0.0, and a spine from a run where
    reconcile_timeline never stamped the field has no key at all. Both must return the
    bare allotment, exactly as they did before the pad was read."""
    stage = DemoVideoNarrateStage()
    assert (
        stage.fit_target_for({"id": "c", "seconds": 6.5, "freeze_pad_seconds": 0.0}, _ctx()) == 6.5
    )
    assert stage.fit_target_for({"id": "c", "seconds": 6.5}, _ctx()) == 6.5
    assert (
        stage.fit_target_for({"id": "c", "seconds": 6.5, "freeze_pad_seconds": None}, _ctx()) == 6.5
    )


def test_a_junk_or_negative_pad_contributes_nothing_rather_than_corrupting_the_target() -> None:
    """The pad comes off a stage output, so it is not guaranteed numeric. A bad value
    must degrade to "no pad" — never shorten the target, which would make the fit pass
    speed the line UP past what the solver chose."""
    stage = DemoVideoNarrateStage()
    for pad in ("1.5s", -2.0, True, [1.5]):
        assert (
            stage.fit_target_for({"id": "c", "seconds": 6.5, "freeze_pad_seconds": pad}, _ctx())
            == 6.5
        ), pad


def test_a_beat_with_no_allotment_still_opts_out_even_when_padded() -> None:
    """The pad is an EXTENSION of an allotment, never an allotment of its own."""
    stage = DemoVideoNarrateStage()
    assert stage.fit_target_for({"id": "c", "freeze_pad_seconds": 1.5}, _ctx()) is None
    assert (
        stage.fit_target_for({"id": "c", "seconds": 0, "freeze_pad_seconds": 1.5}, _ctx()) is None
    )


# ══ THE UNDERSPOKEN HOLD IS SUBTRACTED, NOT ADDED ═════════════════════════════
#
# Wave 7 computed `hold_seconds` precisely so an underspoken beat could cover its
# residual with a HELD FRAME instead of slowed speech. `fit_target_for` handing
# narration the whole allotment anyway re-synthesizes the line to fill seconds the
# picture already agreed to cover, dragging the delivery toward `_FIT_FLOOR`. The
# hold points the OTHER WAY from the freeze pad: the pad EXTENDS the window, the
# hold CARVES OUT of it.


def test_the_underspoken_hold_is_subtracted_from_the_fit_target() -> None:
    stage = DemoVideoNarrateStage()
    target = stage.fit_target_for({"id": "clip-1", "seconds": 6.5, "hold_seconds": 1.5}, _ctx())
    assert target == 5.0
    # ⛔ And it is NOT the bare allotment — handing narration the whole window is
    # the exact defect this fix closes.
    assert target != 6.5


def test_the_hold_and_the_pad_combine_correctly() -> None:
    """Pad ADDS to the window (footage may hold up to the ceiling), hold CARVES
    the underspoken residual back out — both apply, in that order."""
    stage = DemoVideoNarrateStage()
    target = stage.fit_target_for(
        {"id": "c", "seconds": 6.5, "freeze_pad_seconds": 1.5, "hold_seconds": 0.5}, _ctx()
    )
    assert target == 7.5


def test_a_beat_with_no_hold_is_byte_identical_to_before() -> None:
    """Non-underspoken beats are stamped 0.0, and a spine from a run where
    reconcile_timeline never stamped the field has no key at all. Both must
    return the bare allotment, exactly as before the hold was read."""
    stage = DemoVideoNarrateStage()
    assert stage.fit_target_for({"id": "c", "seconds": 6.5, "hold_seconds": 0.0}, _ctx()) == 6.5
    assert stage.fit_target_for({"id": "c", "seconds": 6.5}, _ctx()) == 6.5
    assert stage.fit_target_for({"id": "c", "seconds": 6.5, "hold_seconds": None}, _ctx()) == 6.5


def test_a_junk_or_negative_hold_contributes_nothing_rather_than_corrupting_the_target() -> None:
    """The hold comes off a stage output, so it is not guaranteed numeric. A bad
    value must degrade to "no hold" — never lengthen the target, which would slow
    the line down for a hold that does not exist."""
    stage = DemoVideoNarrateStage()
    for hold in ("1.5s", -2.0, True, [1.5]):
        assert (
            stage.fit_target_for({"id": "c", "seconds": 6.5, "hold_seconds": hold}, _ctx()) == 6.5
        ), hold


def test_a_hold_that_would_consume_the_whole_window_falls_back_to_the_window() -> None:
    """A corrupt/oversized hold (>= the window) must not invert the arithmetic
    into a non-positive fit target — fall back to the un-subtracted window rather
    than handing the base a target that would divide-by-near-zero or go negative."""
    stage = DemoVideoNarrateStage()
    assert stage.fit_target_for({"id": "c", "seconds": 6.5, "hold_seconds": 6.5}, _ctx()) == 6.5
    assert stage.fit_target_for({"id": "c", "seconds": 6.5, "hold_seconds": 9.0}, _ctx()) == 6.5
