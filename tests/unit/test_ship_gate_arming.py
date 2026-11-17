"""The A11 ship gate's ARMING axis — the gate that could not fail, and its three states.

THE DEFECT THIS LOCKS. ``ShipGateEnforcementError`` required ``ship_gate.enforce``
truthy AND ``mode == "auto"`` (the only site that could raise it, ``_enforced_accept``,
was reachable only from ``_auto_redrive``). Production seeds
``{"mode": "critique_only"}`` with no ``enforce`` key — ``ship_gate_default_enforce``
defaults False — so both conditions were false SIMULTANEOUSLY and ``ship_ready`` was a
verdict that explained and never decided. ``_failed_ceilings`` named terms for a
decision nobody ever took.

Two axes now, deliberately not one field:
  * ``ship_gate.mode`` — REPAIR: does the gate try to fix a failure (``auto``) or only
    describe it (``critique_only``). It no longer controls whether the gate blocks.
  * ``PIPELINE_SHIP_GATE_<TERM>_ARMED`` — BLOCKING: per term, backed out from a GitOps
    configmap with no code change and no rollout.

And THREE states, not two, because the judge can measure nothing. ``insufficient_
evidence`` is never a pass and never a fail: ``gestalt_summary`` is measurably ``{}`` on
~55% of live rows, and a footage critic that starved on 48.7% of runs was read as "no
flagged shots ⇒ good footage" for months. A gate armed on ``ship_ready`` (a single bool
that is False for "measured and bad" AND for "measured nothing") would have convicted
every one of those runs of a defect nobody observed.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from src.config import PipelineSettings
from src.services.stages.demo_video.final_critique import (
    _TERM_ARMED_DEFAULTS,
    _TERM_KILL_SWITCHES,
    GATE_FAIL,
    GATE_INSUFFICIENT_EVIDENCE,
    GATE_PASS,
    SHIP_GATE_TERMS,
    TERM_COHESION,
    TERM_CONTINUITY,
    TERM_DURATION,
    TERM_FAIL,
    TERM_INSUFFICIENT_EVIDENCE,
    TERM_LOUDNESS,
    TERM_MOMENTUM,
    ShipGateTermVerdict,
    gate_armed,
    gate_tier,
    ship_gate_decision,
    ship_gate_record,
    ship_gate_term_verdicts,
)

# A gestalt whose loudness is MEASURED and out of the (-14, -11) window, and whose
# visual terms are observable (so continuity/cohesion are real verdicts, not nulls).
_MEASURED_BAD_LOUDNESS: dict[str, object] = {
    "insufficient_evidence": False,
    "continuity_breaks": [],
    "cohesion": 0.8,
    "loudness": {"integrated_lufs": -6.0},
    "loudness_verdict": "fail",
    "momentum_gap": 0.2,
    "momentum_verdict": "unconfigured",
    "ship_ready": False,
}


def _verdicts(
    gestalt: dict[str, object],
    *,
    ceilings: dict[str, object] | None = None,
    conformance: dict[str, object] | None = None,
    settings: object | None = None,
) -> list[ShipGateTermVerdict]:
    return ship_gate_term_verdicts(
        gestalt,
        ceilings or {},
        conformance,
        settings=settings if settings is not None else PipelineSettings(),
    )


def _state(verdicts: list[ShipGateTermVerdict], term: str) -> str:
    return next(v.state for v in verdicts if v.term == term)


# ── the kill-switch table is REAL: every entry names a live Settings field ────
def test_every_kill_switch_names_a_field_that_actually_exists_on_settings() -> None:
    # A kill-switch table whose field names are typos reads as "every term disarmed" —
    # `gate_armed` falls through to `_TERM_ARMED_DEFAULTS` and the configmap flip does
    # nothing. Pin the names against the real model, not against a copy of themselves.
    fields = set(PipelineSettings.model_fields)
    assert set(_TERM_KILL_SWITCHES) == set(SHIP_GATE_TERMS)
    missing = {t: f for t, f in _TERM_KILL_SWITCHES.items() if f not in fields}
    assert missing == {}, f"kill switches naming nonexistent settings fields: {missing}"


def test_armed_defaults_agree_with_the_real_settings_defaults() -> None:
    # `_TERM_ARMED_DEFAULTS` exists only so a settings object PREDATING the fields reads
    # the real default instead of fail-open False. If it drifts from config.py it lies to
    # exactly the callers that depend on it.
    settings = PipelineSettings()
    for term, field in _TERM_KILL_SWITCHES.items():
        assert _TERM_ARMED_DEFAULTS[term] is getattr(settings, field), term


def test_loudness_is_disarmed_and_a_real_failing_master_no_longer_blocks_the_run() -> None:
    # Regression pin, 2026-09-16. Loudness was armed on an explicitly UNMEASURED fail
    # rate ("n=0 ... accepted false-failure risk"). Measured live on RS
    # w2v-orchestrator-7b5bbcf5cf: 3 of 3 demo_video runs failed final_critique at
    # -14.2 / -14.1 / -14.1 LUFS. Because final_critique is the delivered_artifact_stage
    # for demo_video AND showcase, and relief only spawns on a COMPLETED run, an
    # enforcing loudness term took both blueprints down and silently disabled
    # demo_video auto-convergence.
    #
    # This test uses a REAL measured value from that outage (-14.1), not the synthetic
    # -6.0 the other tests use, so it fails if anyone re-arms without fixing the bus.
    settings = PipelineSettings()
    assert settings.ship_gate_loudness_armed is False
    assert gate_armed(settings, TERM_LOUDNESS) is False

    live = {**_MEASURED_BAD_LOUDNESS, "loudness": {"integrated_lufs": -14.1}}
    verdicts = _verdicts(live, settings=settings)
    decision = ship_gate_decision(verdicts)
    record = ship_gate_record(verdicts, decision)
    assert TERM_LOUDNESS not in cast("list[str]", record["failed_terms"])

    # ... but the DIAGNOSIS survives the disarm, or the bus can never be calibrated.
    loudness = next(v for v in verdicts if v.term == TERM_LOUDNESS)
    assert loudness.state == TERM_FAIL
    assert loudness.armed is False
    assert loudness.as_row()["value"] == -14.1


def test_enforce_all_still_arms_loudness_so_the_escalation_survives_the_back_out() -> None:
    # Disarming by default must not remove the per-run `ship_gate.enforce: true` path —
    # otherwise backing a term out quietly deletes the escalation as well.
    assert gate_armed(PipelineSettings(), TERM_LOUDNESS, enforce_all=True) is True


def test_a_settings_object_missing_the_field_reads_the_real_default_not_fail_open() -> None:
    # The fail-OPEN direction (unknown ⇒ disarmed) is how this gate came to decide
    # nothing. A double with no ship-gate fields at all must still arm an ARMED term.
    # The exemplar is MOMENTUM, not loudness: loudness was disarmed 2026-09-16 on a
    # measured 3-of-3 fail rate, so using it here would assert the default rather than
    # the fall-through this test exists to pin.
    double = SimpleNamespace()
    assert gate_armed(double, TERM_MOMENTUM) is True
    assert gate_armed(double, TERM_COHESION) is False
    assert gate_armed(double, TERM_LOUDNESS) is False


def test_an_unknown_term_can_only_be_armed_by_the_whole_gate_escalation() -> None:
    assert gate_armed(PipelineSettings(), "invented_term") is False
    assert gate_armed(PipelineSettings(), "invented_term", enforce_all=True) is True


def test_enforce_all_arms_even_a_deliberately_disarmed_term() -> None:
    # `ship_gate.enforce: true` survives as the per-run ESCALATION — it arms every term,
    # including the ones this wave refused to arm by default.
    settings = PipelineSettings()
    assert gate_armed(settings, TERM_COHESION) is False
    assert gate_armed(settings, TERM_COHESION, enforce_all=True) is True


def test_the_two_axes_are_separate_fields() -> None:
    # Tier answers "what kind of threshold"; arming answers "does it block now". Duration
    # is STRUCTURAL and NOT armed at the same time — which is only expressible because
    # they are two fields. Collapsing them into one would make this state unreachable.
    assert gate_tier(TERM_DURATION) == "structural"
    assert gate_armed(PipelineSettings(), TERM_DURATION) is False
    assert gate_tier("invented_term") == "tunable"  # drift UNDER-reports severity


# ── three states, kept apart ──────────────────────────────────────────────────
def test_a_measured_out_of_window_loudness_is_an_affirmative_fail() -> None:
    # Armed explicitly: the three-state design is what is under test here, not the
    # default. Loudness is disarmed by default since 2026-09-16 (measured 3-of-3 fail
    # rate); with it disarmed the decision would be insufficient_evidence and this test
    # would assert nothing about FAIL vs the other two states.
    armed = PipelineSettings(ship_gate_loudness_armed=True)
    verdicts = _verdicts(dict(_MEASURED_BAD_LOUDNESS), settings=armed)
    assert _state(verdicts, TERM_LOUDNESS) == TERM_FAIL
    assert ship_gate_decision(verdicts) == GATE_FAIL


def test_an_unmeasurable_loudness_is_insufficient_evidence_never_a_fail() -> None:
    # The distinction the whole design turns on: the compiler could not measure the
    # master. That is not evidence the film is loud, and convicting it would destroy a
    # paid deliverable over a probe failure.
    gestalt = {**_MEASURED_BAD_LOUDNESS, "loudness_verdict": "insufficient_evidence"}
    verdicts = _verdicts(gestalt)
    assert _state(verdicts, TERM_LOUDNESS) == TERM_INSUFFICIENT_EVIDENCE
    assert ship_gate_decision(verdicts) != GATE_FAIL


def test_an_empty_gestalt_fails_nothing_and_passes_nothing() -> None:
    # ~55% of live rows. Under a `ship_ready`-based gate every one of them would have
    # hard-failed (`ship_ready` is False on insufficient evidence). Here: no claim.
    verdicts = _verdicts({})
    assert {v.state for v in verdicts} == {TERM_INSUFFICIENT_EVIDENCE}
    assert ship_gate_decision(verdicts) == GATE_INSUFFICIENT_EVIDENCE


def test_a_pass_requires_an_armed_term_to_have_actually_measured_something() -> None:
    # "Nothing objected" is not a pass when nothing looked. Loudness measured and in
    # window ⇒ a real pass; the same gestalt with loudness unmeasured ⇒ no claim, even
    # though nothing failed in either case.
    # Armed explicitly — see the companion test below for what the DEFAULT now means.
    armed = PipelineSettings(ship_gate_loudness_armed=True)
    good = {
        **_MEASURED_BAD_LOUDNESS,
        "loudness": {"integrated_lufs": -12.5},
        "loudness_verdict": "pass",
    }
    assert ship_gate_decision(_verdicts(good, settings=armed)) == GATE_PASS
    starved = {**good, "loudness_verdict": "insufficient_evidence"}
    assert ship_gate_decision(_verdicts(starved, settings=armed)) == GATE_INSUFFICIENT_EVIDENCE


def test_with_loudness_disarmed_the_gate_can_no_longer_return_a_pass() -> None:
    # ⚠️ The COST of the 2026-09-16 back-out, pinned so it is not discovered by surprise.
    # Continuity, cohesion and duration are all disarmed, and momentum reads
    # `unconfigured` in the live config (nothing sets `ceilings["momentum_gap_max"]`),
    # so loudness was the ONLY term that could ever produce an affirmative GATE_PASS.
    # Disarming it means the gate now decides NOTHING rather than deciding wrongly on
    # 100% of masters. That is the better of two bad states while the speech bus is
    # broken, but it is a real loss of signal and re-arming should restore it.
    good = {
        **_MEASURED_BAD_LOUDNESS,
        "loudness": {"integrated_lufs": -12.5},
        "loudness_verdict": "pass",
    }
    assert ship_gate_decision(_verdicts(good)) == GATE_INSUFFICIENT_EVIDENCE


def test_an_unconfigured_momentum_ceiling_observed_nothing_so_it_cannot_pass() -> None:
    # `momentum_ok` is True when `momentum_gap_max` is unset, which made momentum a
    # VACUOUSLY TRUE AND-term of `ship_ready` on every master ever produced. Read as a
    # three-state verdict it is insufficient evidence, so it cannot manufacture a pass.
    verdicts = _verdicts({**_MEASURED_BAD_LOUDNESS, "loudness_verdict": "insufficient_evidence"})
    assert _state(verdicts, TERM_MOMENTUM) == TERM_INSUFFICIENT_EVIDENCE
    assert ship_gate_decision(verdicts) == GATE_INSUFFICIENT_EVIDENCE


def test_a_configured_momentum_ceiling_turns_the_seam_live() -> None:
    gestalt = {
        **_MEASURED_BAD_LOUDNESS,
        "loudness_verdict": "insufficient_evidence",
        "momentum_verdict": "fail",
    }
    verdicts = _verdicts(gestalt, ceilings={"momentum_gap_max": 0.1})
    assert _state(verdicts, TERM_MOMENTUM) == TERM_FAIL
    assert ship_gate_decision(verdicts) == GATE_FAIL


def test_an_unmeasured_cohesion_is_not_convicted_as_a_bad_cohesion() -> None:
    # `_failed_ceilings` folded `cohesion is None` in with `cohesion < floor` under one
    # reason string, so at the only place a decision is taken an ABSENT measurement was
    # indistinguishable from a bad one.
    verdicts = _verdicts({**_MEASURED_BAD_LOUDNESS, "cohesion": None})
    assert _state(verdicts, TERM_COHESION) == TERM_INSUFFICIENT_EVIDENCE


def test_insufficient_evidence_in_the_gestalt_unmeasures_the_visual_terms() -> None:
    verdicts = _verdicts({**_MEASURED_BAD_LOUDNESS, "insufficient_evidence": True})
    assert _state(verdicts, TERM_CONTINUITY) == TERM_INSUFFICIENT_EVIDENCE
    assert _state(verdicts, TERM_COHESION) == TERM_INSUFFICIENT_EVIDENCE
    # …but the loudness measurement is independent of the visual gestalt and survives.
    assert _state(verdicts, TERM_LOUDNESS) == TERM_FAIL


# ── the duration term READS an existing verdict; it invents no ceiling ────────
def test_the_duration_term_reads_the_existing_conformance_verdict() -> None:
    # The 218.47s-against-a-30.0s-order shape. The gate had NO duration term at all, so
    # a 728% film was badged ship-ready. This wires the verdict
    # `stages/shared/duration_conformance` already produces — no new threshold here.
    conformance: dict[str, object] = {
        "ordered_seconds": 30.0,
        "delivered_seconds": 218.47,
        "ratio": 7.28,
        "deviation_pct": 628.2,
        "within_tolerance": False,
        "delivered_basis": "declared",
    }
    verdicts = _verdicts({}, conformance=conformance)
    duration = next(v for v in verdicts if v.term == TERM_DURATION)
    assert duration.state == TERM_FAIL
    assert "218.5s vs 30.0s ordered" in duration.detail
    assert "728% of order" in duration.detail
    # NOT armed by default — a measured refusal (129%-273% of order across four live
    # types). So a 728% film is now DIAGNOSED on the record without yet blocking.
    assert duration.armed is False
    assert ship_gate_decision(verdicts) != GATE_FAIL


def test_arming_duration_makes_the_728_percent_film_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    # …and flipping the switch is all it takes. No code change.
    monkeypatch.setenv("PIPELINE_SHIP_GATE_DURATION_ARMED", "true")
    conformance: dict[str, object] = {
        "ordered_seconds": 30.0,
        "delivered_seconds": 218.47,
        "ratio": 7.28,
        "deviation_pct": 628.2,
        "within_tolerance": False,
        "delivered_basis": "declared",
    }
    verdicts = _verdicts({}, conformance=conformance, settings=PipelineSettings())
    assert next(v for v in verdicts if v.term == TERM_DURATION).armed is True
    assert ship_gate_decision(verdicts) == GATE_FAIL


def test_no_ordered_duration_is_insufficient_evidence_not_a_conforming_pass() -> None:
    # "Nobody ordered a length" and "the length ordered was met" are different facts.
    verdicts = _verdicts({}, conformance=None)
    duration = next(v for v in verdicts if v.term == TERM_DURATION)
    assert duration.state == TERM_INSUFFICIENT_EVIDENCE
    assert "no usable ordered duration" in duration.detail


# ── the record is what makes a failure calibratable ──────────────────────────
def test_the_record_names_the_failed_term_its_value_and_its_limit() -> None:
    # Loudness is DISARMED by default since 2026-09-16, so this arms it explicitly:
    # the test exists to pin the RECORD's shape for a failing armed term, not to
    # assert the default. Reading it as a default assertion would make it vacuous.
    armed = PipelineSettings(ship_gate_loudness_armed=True)
    verdicts = _verdicts(dict(_MEASURED_BAD_LOUDNESS), settings=armed)
    record = ship_gate_record(verdicts, ship_gate_decision(verdicts))
    assert record["decision"] == GATE_FAIL
    assert record["failed_terms"] == [TERM_LOUDNESS]
    row = next(r for r in verdicts if r.term == TERM_LOUDNESS).as_row()
    assert row["value"] == -6.0  # the MEASURED number …
    # CORRECTED 2026-09-16 (was "[-14.0, -11.0] LUFS"). Deliberately still a LITERAL and
    # still exact — the band is exactly what this row must name, so deriving it from
    # `ship_loudness_window()` would make the assertion tautological. Only the number
    # moved: compiler PR #217 (2026-09-12) made -14.0 the shipped master target, and the
    # old window convicted the very target the product ships, so it was recentred on it.
    # WIDTH IS UNCHANGED at 3.0 LU — this is a recentring, not a loosening.
    assert row["limit"] == "[-15.5, -12.5] LUFS"  # … and what it was compared against
    assert row["tier"] == "tunable"
    assert row["armed"] is True
    # EVERY term appears, armed or not — so "was this even blocking" is answerable.
    rows = cast("list[dict[str, object]]", record["terms"])
    assert [r["term"] for r in rows] == list(SHIP_GATE_TERMS)


def test_the_record_separates_unmeasured_terms_from_passing_ones() -> None:
    # Loudness armed explicitly so this still exercises TWO armed terms; by default it
    # is disarmed (2026-09-16, measured 3-of-3 fail rate) and would drop out of the set,
    # shrinking what this test covers rather than testing the mechanism it names.
    armed = PipelineSettings(ship_gate_loudness_armed=True)
    verdicts = _verdicts({}, settings=armed)
    record = ship_gate_record(verdicts, ship_gate_decision(verdicts))
    assert record["decision"] == GATE_INSUFFICIENT_EVIDENCE
    assert record["failed_terms"] == []
    # The armed terms are named as UNMEASURED, not silently absent from the failures.
    unmeasured = cast("list[str]", record["unmeasured_terms"])
    assert set(unmeasured) == {TERM_MOMENTUM, TERM_LOUDNESS}


def test_a_disarmed_term_is_still_measured_and_still_recorded() -> None:
    # Disarming removes the BLOCK, not the diagnosis — otherwise backing a term out
    # destroys the evidence needed to calibrate it and it can never be armed again.
    verdicts = _verdicts({**_MEASURED_BAD_LOUDNESS, "cohesion": 0.01})
    cohesion = next(v for v in verdicts if v.term == TERM_COHESION)
    assert cohesion.state == TERM_FAIL
    assert cohesion.armed is False
    assert cohesion.value == 0.01
    failed = cast("list[str]", ship_gate_record(verdicts, GATE_FAIL)["failed_terms"])
    assert TERM_COHESION not in failed
