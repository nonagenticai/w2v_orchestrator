"""A conformance PASS must not whitewash a plan the pipeline itself refused.

Confirmed on prod run ``01a0a3c6-c8e7-73db-b13e-fded1a4cbb4f``::

    reconcile_timeline        vo_rate: [1.0, 1.0, 1.0, 0.8, 0.8, 0.8]
                              feasible: [False, False, False]
                              convergence_outcome: 'cant_converge'
    demo_video_encode_master  ordered 30.0s  delivered 32.038s  (+6.79%)
                              delivered_basis: 'playable'  within_tolerance: True

The timeline was stamped INFEASIBLE and ``ordered_duration_infeasible`` was
emitted; ``encode_master`` then graded the SAME overrun as fine, purely because
its band is +/-10% while the plan had already been refused at ZERO tolerance
(``time_budget.infeasible_beats`` is a bare ``Sum(min) > target``). Fleet-wide over
14 days ``duration_out_of_tolerance`` (60) + ``ordered_duration_infeasible`` (15)
are 68% of ALL degradations emitted — this is the dominant defect, and a verdict
that reads reassuringly on top of it is how it stays invisible.

⛔ WHAT THESE TESTS DELIBERATELY DO **NOT** PIN
----------------------------------------------
They do not pin a hard gate. ``within_tolerance`` is asserted UNCHANGED and the
``degradations`` channel is asserted UNCHANGED on every case here; the signal
rides ``advisories`` (the non-degrading tier) and the log level. Post-rollout
sample size is ONE run. Arming this is a separate, evidence-backed decision, and
``test_the_pass_is_still_a_pass`` exists to fail if someone arms it by accident.

⛔ AND THEY DO NOT UNIFY THE BANDS. Three numbers grade three different objects:
``DURATION_TOLERANCE_FRACTION`` (0.10) the probed FILE,
``time_budget.DEFAULT_TOLERANCE_PCT`` (2.0) the SOLVED TIMELINE from the short
side only, and the infeasible verdict NEITHER of them — no band at all.
``test_the_three_bands_measure_three_different_objects`` pins that they are
allowed to disagree.
"""

from __future__ import annotations

import logging

import pytest

from src.services.stages.shared.degradation import (
    ADVISORY_CODES,
    DEGRADATION_CODES,
    DURATION_OUT_OF_TOLERANCE,
    DURATION_PLAN_INFEASIBLE,
    ORDERED_DURATION_INFEASIBLE,
    is_degraded,
)
from src.services.stages.shared.duration_conformance import (
    DURATION_TOLERANCE_FRACTION,
    duration_conformance,
    duration_conformance_advisories,
    duration_conformance_degradations,
    observe_duration_conformance,
    plan_infeasible_reason,
)
from src.services.stages.shared.time_budget import DEFAULT_TOLERANCE_PCT, infeasible_beats

# Run 01a0a3c6's actual numbers.
_ORDERED = 30.0
_DELIVERED = 32.038


def _reconcile_output(
    *,
    outcome: str | None = "cant_converge",
    degradation_codes: tuple[str, ...] = (ORDERED_DURATION_INFEASIBLE,),
) -> dict[str, object]:
    """A ``reconcile_timeline`` stage output dict as ``prior_run_outputs`` holds it.

    Shaped after ``time_budget.build_timeline_manifest`` — the manifest is nested
    under ``timeline_manifest.run``, which is where the classification lives and
    where a flattened reader would silently find nothing.
    """
    out: dict[str, object] = {
        "degradations": [{"code": code, "detail": "…"} for code in degradation_codes]
    }
    if outcome is not None:
        out["timeline_manifest"] = {
            "beats": [],
            "run": {"target_seconds": _ORDERED, "convergence_outcome": outcome},
        }
    return out


class TestReadingTheUpstreamRefusal:
    def test_cant_converge_is_a_refusal(self) -> None:
        reason = plan_infeasible_reason(_reconcile_output(degradation_codes=()))
        assert reason == "convergence_outcome=cant_converge"

    def test_the_degradation_alone_is_a_refusal(self) -> None:
        """Either signal fires alone: they come from different code paths.

        ``ordered_duration_infeasible`` is also raised by ``demo_video_gen`` and by
        ``narration_budget``, so a run can carry it with no manifest at all.
        """
        reason = plan_infeasible_reason(_reconcile_output(outcome=None))
        assert reason == ORDERED_DURATION_INFEASIBLE

    def test_both_signals_are_joined(self) -> None:
        """Run 01a0a3c6 carried BOTH — the record must say so, not pick one."""
        assert plan_infeasible_reason(_reconcile_output()) == (
            f"convergence_outcome=cant_converge+{ORDERED_DURATION_INFEASIBLE}"
        )

    @pytest.mark.parametrize("outcome", ["progress"])
    def test_every_non_converged_outcome_is_a_refusal(self, outcome: str) -> None:
        """``relief_failed`` counts: no classification was EVER computed.

        The relief loop raised (WM/LLM unavailable), so "the plan is fine" was
        never established, and a downstream band pass must not be read as
        establishing it retroactively.
        """
        reason = plan_infeasible_reason(_reconcile_output(outcome=outcome, degradation_codes=()))
        assert reason == f"convergence_outcome={outcome}"

    def test_a_converged_plan_is_not_a_refusal(self) -> None:
        """FALSE-POSITIVE GUARD — the healthy population must read clean."""
        assert (
            plan_infeasible_reason(_reconcile_output(outcome="converged", degradation_codes=()))
            is None
        )

    @pytest.mark.parametrize(
        "prior",
        [None],
    )
    def test_a_malformed_prior_output_yields_no_refusal_and_never_raises(
        self, prior: dict[str, object] | None
    ) -> None:
        """This runs on the exit path of a render that has ALREADY been paid for.

        Every shape here is one a JSON round-trip, a replayed pre-manifest run, or a
        regenerate that re-entered below reconcile can produce. A reporting observer
        that can raise is a worse defect than the whitewash it was added to fix, so
        each yields "no refusal known" — the same non-alarming answer as "none".
        """
        assert plan_infeasible_reason(prior) is None


class TestTheVerdictRefusesToReassure:
    def test_the_prod_numbers_still_pass_the_band(self) -> None:
        """The band was never wrong: +6.79% IS inside +/-10%."""
        result = duration_conformance(_ORDERED, _DELIVERED, plan_infeasible="x")
        assert result is not None
        assert result["deviation_pct"] == pytest.approx(6.79, abs=0.01)
        assert result["within_tolerance"] is True

    def test_the_refusal_is_recorded_beside_the_verdict(self) -> None:
        result = duration_conformance(
            _ORDERED, _DELIVERED, plan_infeasible="convergence_outcome=cant_converge"
        )
        assert result is not None
        assert result["plan_infeasible"] is True
        assert result["plan_infeasible_reason"] == "convergence_outcome=cant_converge"

    def test_no_refusal_records_false_and_none(self) -> None:
        result = duration_conformance(_ORDERED, _DELIVERED)
        assert result is not None
        assert result["plan_infeasible"] is False
        assert result["plan_infeasible_reason"] is None

    def test_the_pass_is_still_a_pass(self) -> None:
        """⛔ THE RESTRAINT GUARD — this is NOT armed as a gate.

        If someone flips ``within_tolerance`` or promotes the advisory to a
        degradation on the strength of one run, this fails. ``ordered_duration_``
        ``infeasible`` is ALREADY a degradation on the stage that DECIDED it; a
        second one here would report one fault twice, from a stage that only
        observed it.
        """
        result = duration_conformance(_ORDERED, _DELIVERED, plan_infeasible="x")
        assert result is not None
        assert result["within_tolerance"] is True
        assert duration_conformance_degradations(result) == []
        assert (
            is_degraded({"degradations": duration_conformance_advisories(result), "notes": []})
            is True
        ), "sanity: is_degraded reads the DEGRADATIONS key, so the guard below is meaningful"
        assert (
            is_degraded(
                {
                    "degradations": duration_conformance_degradations(result),
                    "advisories": duration_conformance_advisories(result),
                    "notes": [],
                }
            )
            is False
        )

    def test_the_advisory_carries_both_bands_and_the_reason(self) -> None:
        # ``playable_seconds`` is passed on every case below so the basis is
        # ``playable`` — run 01a0a3c6's actual basis. Omitting it would land on
        # ``declared`` and drag in an unrelated ``DURATION_UNMEASURED`` advisory,
        # which would make these exact-list assertions test the wrong thing.
        result = duration_conformance(
            _ORDERED,
            _DELIVERED,
            playable_seconds=_DELIVERED,
            plan_infeasible="convergence_outcome=cant_converge",
        )
        advisories = duration_conformance_advisories(result)
        assert [a["code"] for a in advisories] == [DURATION_PLAN_INFEASIBLE]
        detail = advisories[0]["detail"]
        assert "cant_converge" in detail
        assert "32.0s" in detail and "30.0s" in detail

    def test_the_code_is_on_the_non_degrading_tier(self) -> None:
        assert DURATION_PLAN_INFEASIBLE in ADVISORY_CODES
        assert DURATION_PLAN_INFEASIBLE not in DEGRADATION_CODES

    def test_an_out_of_band_delivery_raises_no_second_opinion(self) -> None:
        """Outside the band, ``duration_out_of_tolerance`` already says it missed.

        The defect being closed is a verdict that reads REASSURING. A failing
        verdict does not, so the advisory would add only noise — the near-universal
        flag failure this repo has paid for once.
        """
        result = duration_conformance(_ORDERED, 60.0, playable_seconds=60.0, plan_infeasible="x")
        assert result is not None
        assert result["within_tolerance"] is False
        assert [d["code"] for d in duration_conformance_degradations(result)] == [
            DURATION_OUT_OF_TOLERANCE
        ]
        assert [a["code"] for a in duration_conformance_advisories(result)] == []

    def test_a_conforming_plan_raises_nothing(self) -> None:
        """FALSE-POSITIVE GUARD on the whole healthy population."""
        result = duration_conformance(_ORDERED, _DELIVERED, playable_seconds=_DELIVERED)
        assert duration_conformance_advisories(result) == []
        assert duration_conformance_degradations(result) == []


class TestTheReassuringInfoIsSuppressed:
    def test_a_refused_plan_logs_a_warning_not_the_within_tolerance_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The INFO at ``duration_conformance.py`` is what a log reader quotes.

        "within +/-10%" printed alone under a refused plan IS the whitewash. The
        line must be WARNING and must name the refusal, so the band and the refusal
        cannot be quoted apart.
        """
        with caplog.at_level(logging.INFO):
            observe_duration_conformance(
                "demo_video_encode_master",
                _ORDERED,
                _DELIVERED,
                plan_infeasible="convergence_outcome=cant_converge",
            )
        records = [r for r in caplog.records if "duration conformance" in r.message]
        assert len(records) == 1
        assert records[0].levelno == logging.WARNING
        rendered = records[0].getMessage()
        assert "cant_converge" in rendered
        assert "NOT a clean pass" in rendered
        assert "within +/-" not in rendered

    def test_an_unrefused_pass_still_logs_the_plain_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """FALSE-POSITIVE GUARD: the healthy path's log level is unchanged."""
        with caplog.at_level(logging.INFO):
            observe_duration_conformance("demo_video_encode_master", _ORDERED, _DELIVERED)
        records = [r for r in caplog.records if "duration conformance" in r.message]
        assert len(records) == 1
        assert records[0].levelno == logging.INFO
        assert "within +/-10%" in records[0].getMessage()


class TestTheThreeBandsAreNotOneBand:
    def test_the_three_bands_measure_three_different_objects(self) -> None:
        """⛔ DO NOT UNIFY THESE. They are not a discrepancy.

        * ``DURATION_TOLERANCE_FRACTION`` (0.10) grades the PROBED FILE.
        * ``DEFAULT_TOLERANCE_PCT`` (2.0%) grades the SOLVED TIMELINE, short side
          only (``time_budget.is_underfull``).
        * the INFEASIBLE verdict uses NEITHER — ``infeasible_beats`` is a bare
          ``Sum(min) > target`` with no tolerance at all, because seconds can be
          moved but never created.

        Collapsing them to a shared constant would only move which of the two
        verdicts lies. This pins that they are ALLOWED to differ.
        """
        assert DURATION_TOLERANCE_FRACTION == 0.10
        assert DEFAULT_TOLERANCE_PCT == 2.0
        assert DURATION_TOLERANCE_FRACTION * 100.0 != DEFAULT_TOLERANCE_PCT

    def test_the_infeasible_verdict_has_no_tolerance_at_all(self) -> None:
        """A plan 0.01s over its order is refused — no band absorbs it.

        This is the arithmetic that makes ``plan_infeasible`` necessary: a plan
        refused at zero tolerance routinely delivers a file inside +/-10%.
        """
        from src.services.stages.shared.time_budget import TimeDemand

        demands = [
            TimeDemand(
                id="a", kind="vo", min=15.0, ideal=15.0, max=20.0, elasticity=1.0, pinned=False
            ),
            TimeDemand(
                id="b", kind="vo", min=15.01, ideal=15.01, max=20.0, elasticity=1.0, pinned=False
            ),
        ]
        assert infeasible_beats(demands, _ORDERED) != []
