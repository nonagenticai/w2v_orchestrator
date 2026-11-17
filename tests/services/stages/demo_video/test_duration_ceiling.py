"""demo_video_gen — the DURATION CEILING, the missing mirror of the underfull floor.

Measured on live run ``01a089c3`` (2026-09-10): a 30s order produced 5 beats, bdd
guaranteed 22.00s of recording per scenario, and the orchestrator logged
``duration plan OK — 5 beat(s) x 22.00s measured/beat`` — because 5 clears the ONE beat
the 50% floor requires. The film shipped at **218.47s, 728% of the order**, and every
gate passed it: ``reconcile_timeline`` diagnosed ``cant_converge`` and completed anyway.

``plan_beats`` had a floor term and no ceiling term. These tests pin the ceiling.
No DB, no LLM, no bdd call.
"""

from __future__ import annotations

from typing import Any

from src.services.stages.demo_video.gen import (
    DemoVideoLine,
)
from src.services.stages.shared.duration_conformance import DURATION_TOLERANCE_FRACTION
from src.services.stages.shared.duration_plan import plan_beats

# The production numbers, verbatim.
_ORDERED = 30.0
_PER_BEAT = 22.0
_FLOOR_RATIO = 0.5


def _line(idx: int, **overrides: object) -> DemoVideoLine:
    base: dict[str, object] = {"id": f"clip-{idx}", "text": f"Beat {idx}."}
    base.update(overrides)
    return DemoVideoLine.model_validate(base)


def _deliverable(seconds: float | None = _ORDERED) -> dict[str, Any]:
    return {"target_duration_seconds": seconds}


class TestPlanBeatsCeiling:
    def test_the_production_overshoot_is_resized_down(self) -> None:
        """5 x 22.00s against a 30s order must NOT be OK. This is run 01a089c3.

        ⭐ WAVE 14, N4. ``tolerance_fraction`` now DEFAULTS to the conformance band, so
        the answer here is the ENUMERATED one: nothing conforms at 22.00s/beat (1 beat
        is -26.7%, 2 is +46.7%), and the chosen direction is short. The rounded-quotient
        answer (2 beats / 44.00s / +46.7%) is what the band exists to reject, and it is
        pinned below as the ``tolerance_fraction=None`` legacy arithmetic.
        """
        plan = plan_beats(
            5,
            target_seconds=_ORDERED,
            floor_ratio=_FLOOR_RATIO,
            per_beat_seconds=_PER_BEAT,
            max_beats=5,
        )
        assert plan.verdict == "RESIZED_DOWN"
        assert plan.beats == 1
        assert plan.max_beats_for_target == 1
        assert plan.achievable_seconds == 22.0
        # Still a legal sizing: a resize delivers, a refusal delivers nothing.
        assert plan.satisfiable is True

        legacy = plan_beats(
            5,
            target_seconds=_ORDERED,
            floor_ratio=_FLOOR_RATIO,
            per_beat_seconds=_PER_BEAT,
            max_beats=5,
            tolerance_fraction=None,
        )
        assert (legacy.beats, legacy.achievable_seconds) == (2, 44.0)

    def test_explain_names_the_overshoot(self) -> None:
        plan = plan_beats(
            5,
            target_seconds=_ORDERED,
            floor_ratio=_FLOOR_RATIO,
            per_beat_seconds=_PER_BEAT,
            max_beats=5,
        )
        assert "RESIZED_DOWN" in plan.explain()
        assert "OVERSHOOTS" in plan.explain()

    def test_the_floor_still_wins_over_the_ceiling(self) -> None:
        """A tiny per-beat yield: the ceiling must never cut BELOW the underfull floor.

        target 30s, floor 15s, 1.0s/beat -> floor needs 15, ceiling wants 30. Resizing
        down to 30 is right; resizing to anything under 15 would re-create the ORIGINAL
        underfull defect from the other direction.
        """
        plan = plan_beats(
            40,
            target_seconds=_ORDERED,
            floor_ratio=_FLOOR_RATIO,
            per_beat_seconds=1.0,
            max_beats=40,
        )
        assert plan.verdict == "RESIZED_DOWN"
        assert plan.beats == 30
        assert plan.beats >= plan.min_beats_for_floor


# ceil(15 / 2)


class TestTheCeilingIsCheckedAgainstTheProducts:
    """⭐ O4 (A/V quality wave 13) — ``RESIZED_DOWN`` could not bind on its OWN CITED CASE.

    ``DurationPlan``'s docstring introduces ``RESIZED_DOWN`` as the verdict for a plan
    that OVERSHOOTS the ordered duration, and the overshoot it names is this module's
    own headline: ``30.00s ordered, 22.00s measured/beat, 2 beats = 44.00s`` (+46.7%).
    Trace the old arithmetic:

        needed  = ceil(15.00 / 22.00) = 1
        ceiling = ceil(30.00 / 22.00) = 2
        trigger = requested_beats > max(ceiling, needed) = 2 > 2 = FALSE

    ⇒ verdict ``OK``, and a +46.7% overshoot certified as a plan that fits.

    The cause is named in ``beat_counts_within_band``'s own docstring — *"a bound
    computed by rounding cannot be checked by the same rounding"*. ``ceiling`` is
    ``beats_for_seconds``, which rounds UP, so it can only ever catch a plan that
    overshoots the ROUNDED-UP bound, never one that overshoots the ORDER. That helper
    was written precisely to fix this, and ``plan_beats`` never called it.
    """

    def test_the_verdicts_own_cited_case_now_binds(self) -> None:
        """+46.7% was certified OK. It must not be."""
        before = plan_beats(
            2,
            target_seconds=_ORDERED,
            floor_ratio=_FLOOR_RATIO,
            per_beat_seconds=_PER_BEAT,
            max_beats=2,
            # The legacy arithmetic, now reachable ONLY by asking for it explicitly.
            tolerance_fraction=None,
        )
        assert before.verdict == "OK", "the defect, pinned: the old bound cannot bind here"
        assert before.beats * _PER_BEAT == 44.0

        after = plan_beats(
            2,
            target_seconds=_ORDERED,
            floor_ratio=_FLOOR_RATIO,
            per_beat_seconds=_PER_BEAT,
            max_beats=2,
            tolerance_fraction=0.10,
        )
        assert after.verdict == "RESIZED_DOWN"
        assert after.beats == 1
        assert after.achievable_seconds == 22.0

    def test_an_overshoot_the_rounded_bound_misses_is_resized_into_the_band(self) -> None:
        """A case where a conforming count DOES exist and the old bound sailed past it:
        3 x 14.00s = 42.00s (+40.0%) against a 30.00s order, where 2 x 14.00s = 28.00s
        lands inside [27.00, 33.00]. ``ceil(30/14) = 3``, so the rounded ceiling was the
        overshooting count itself.
        """
        old = plan_beats(
            3,
            target_seconds=_ORDERED,
            floor_ratio=_FLOOR_RATIO,
            per_beat_seconds=14.0,
            max_beats=3,
            tolerance_fraction=None,
        )
        assert (old.verdict, old.beats * 14.0) == ("OK", 42.0)

        new = plan_beats(
            3,
            target_seconds=_ORDERED,
            floor_ratio=_FLOOR_RATIO,
            per_beat_seconds=14.0,
            max_beats=3,
            tolerance_fraction=0.10,
        )
        assert new.verdict == "RESIZED_DOWN"
        assert new.beats == 2
        assert new.achievable_seconds == 28.0

    def test_it_prefers_missing_short_when_nothing_conforms(self) -> None:
        """At 22.00s/beat NOTHING conforms to a 30.00s order: 1 beat is -26.7%, 2 is
        +46.7%. The choice is which way to MISS, and wave 13 settles it — the ordered
        duration carries a real under-delivery allowance, and the compiler cannot cut
        audio, so short beats long.
        """
        plan = plan_beats(
            2,
            target_seconds=_ORDERED,
            floor_ratio=_FLOOR_RATIO,
            per_beat_seconds=_PER_BEAT,
            max_beats=2,
            tolerance_fraction=0.10,
        )
        assert plan.beats * _PER_BEAT < _ORDERED, "missing short is the chosen direction"

    def test_a_conforming_plan_is_left_completely_alone(self) -> None:
        """The bound may only TIGHTEN. Without this, the three tests above would also
        pass an implementation that resized every plan downwards.
        """
        plan = plan_beats(
            2,
            target_seconds=_ORDERED,
            floor_ratio=_FLOOR_RATIO,
            per_beat_seconds=14.0,
            max_beats=2,
            tolerance_fraction=0.10,
        )
        assert (plan.verdict, plan.beats) == ("OK", 2)
        assert plan.achievable_seconds == 28.0

    def test_no_shorter_count_exists_so_the_old_answer_stands(self) -> None:
        """40.00s/beat against a 30.00s order: even ONE beat is +33.3%, so there is no
        count that misses short. The pre-existing answer is already the only one, and
        the new check must not invent a zero-beat plan.
        """
        plan = plan_beats(
            1,
            target_seconds=_ORDERED,
            floor_ratio=_FLOOR_RATIO,
            per_beat_seconds=40.0,
            max_beats=1,
            tolerance_fraction=0.10,
        )
        assert (plan.verdict, plan.beats) == ("OK", 1)

    def test_omitting_the_tolerance_arms_the_band(self) -> None:
        """⭐ WAVE 14, N4 — THE INVERSION. Omitting ``tolerance_fraction`` must now be
        identical to PASSING ``DURATION_TOLERANCE_FRACTION``, and must NOT be identical
        to ``None``, on at least one case.

        This test asserted the opposite until wave 14 ("the default must change
        NOTHING"), and that default is precisely how the +46.7% overshoot class stayed
        live on three of the four call sites — including the PAID relief path in
        ``structural_relief_policy.decide_relief_shape``, which never passed it.
        """
        differed = 0
        for per_beat, requested in ((22.0, 2), (14.0, 3), (14.0, 2), (10.0, 5), (40.0, 1)):
            kwargs: dict[str, Any] = {
                "target_seconds": _ORDERED,
                "floor_ratio": _FLOOR_RATIO,
                "per_beat_seconds": per_beat,
                "max_beats": requested,
            }
            bare = plan_beats(requested, **kwargs)
            armed = plan_beats(requested, **kwargs, tolerance_fraction=DURATION_TOLERANCE_FRACTION)
            legacy = plan_beats(requested, **kwargs, tolerance_fraction=None)
            assert bare == armed, (per_beat, requested)
            differed += bare != legacy
        assert differed >= 1, "the default is indistinguishable from None — it is not armed"
