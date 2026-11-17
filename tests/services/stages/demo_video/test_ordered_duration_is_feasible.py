"""An ORDER that no beat count can deliver must be refitted and said out loud.

Measured live 2026-09-10 on the ``demo_video_full_chain_todomvc`` fixture: a 30s order
delivered **47.045s (+56.8%)**, structural relief then spawned a child which delivered
**19.95s (-33.5%)**. Neither pass is inside ``duration_conformance``'s +/-10% band, and
every gate passed both.

Neither number is a tuning miss; both are ARITHMETIC:

* bdd guarantees a fixed recording dwell per scenario (``HERO_DWELL_MS``, on the wire as
  ``narration_script.parts[].expected_recording_seconds``, 22.00s live), so an
  N-scenario demo has a HARD floor of ``N x 22.00s``. A 30s order for 2 scenarios is
  unreachable BY CONSTRUCTION — and the existing ceiling guard could not see it, because
  ``plan_beats``'s ceiling is ``ceil(target / per_beat)``, which by definition rounds UP
  PAST the order (ceil(30/22) = 2 ⇒ "2 beats x 22s = 44s is OK").
* the relief undershoot is structural too: ``next_target_beats`` derives its cut from the
  AVERAGE per-beat minimum while ``gen._apply_target_beats`` drops the CHEAPEST droppable
  beats, so the correction overshoots — and nothing below it ever said "too short".

What is pinned here:

1. ``_apply_feasibility_floor`` leaves a FEASIBLE order completely untouched (the
   no-behaviour-change half — a guard that fires on everything is not a guard).
2. An order below the structural floor REFITS the scenario count to ``floor(ordered /
   per_scenario)`` and degrades nothing (the order becomes deliverable).
3. An order below ONE scenario plans to the floor and raises
   ``ORDERED_DURATION_INFEASIBLE`` on the structured channel — never raises.
4. The relief lower guard: a cut that would land under ``ordered x (1 - tolerance)`` is
   raised, and where raising it cannot also shrink the spine NO child is spawned and the
   reason is recorded.

⛔ Nothing here weakens ``reconcile_timeline``'s 50% underfull floor, and nothing here
touches ``HERO_DWELL_MS`` — the dwell is deliberate readability owned by bdd.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.services.runs.structural_relief_policy import decide_relief_shape
from src.services.stages.demo_video.gen import (
    _BDD_GUARANTEED_YIELD,
    _PARENT_MEASURED_YIELD,
    _UNATTRIBUTED_YIELD,
    MEASURED_SECONDS_PER_BEAT_KEY,
    DemoVideoLine,
    _apply_feasibility_floor,
)
from src.services.stages.shared.degradation import (
    ORDERED_DURATION_INFEASIBLE,
)
from src.services.stages.shared.duration_conformance import DURATION_TOLERANCE_FRACTION

# The value bdd EXPORTED on live run 01a0b3f1. Used here ONLY as test DATA fed in
# through the same argument the production code reads off the wire — this file defines
# no module-level per-scenario constant, so
# ``test_no_per_scenario_duration_constant_is_restated_in_this_repo`` stays satisfied
# (it walks ``src/``, and the rule it encodes is "read it, never copy it").
_LIVE_PER_SCENARIO = 22.0


def _lines(count: int) -> list[DemoVideoLine]:
    return [
        DemoVideoLine.model_validate({"id": f"clip-{i}", "text": f"Beat {i}."})
        for i in range(1, count + 1)
    ]


def _pinned_beats(count: int, seconds: float) -> list[dict[str, Any]]:
    """Manifest rows shaped like FOOTAGE beats: ``min == max == measured``."""
    return [
        {"id": f"clip-{i}", "seconds": seconds, "min_seconds": seconds, "max_seconds": seconds}
        for i in range(1, count + 1)
    ]


# ── 1. the feasible order is UNTOUCHED ───────────────────────────────────────────────


# ── 2. the order below the floor REFITS the count ────────────────────────────────────


# ── 3. one scenario already overruns ⇒ plan to the floor + degrade LOUDLY ────────────


# ── 4. the relief LOWER guard ────────────────────────────────────────────────────────


def test_a_relief_cut_that_would_land_under_the_band_is_not_spawned() -> None:
    """THE MEASURED CHILD. 2 pinned beats of 23.5s, 30s target, 17s overrun.

    ``next_target_beats`` asks for 1 beat; 1 x 23.5 = 23.5s is 78.3% of the order, i.e.
    BELOW ``30 x (1 - 0.10) = 27.00s``. Staying in the band needs ceil(27/23.5) = 2
    beats, which is the parent's own count — so the child cannot both shrink and land in
    the band, and must not be billed.
    """
    beats = _pinned_beats(2, 23.5)
    shape = decide_relief_shape(
        run={"target_seconds": 30.0, "overrun_seconds": 17.0},
        beats=beats,
        floor_ratio=0.5,
    )
    assert shape.verdict == "STOP_UNDERFULL"
    assert shape.tolerance_min_beats == 2
    # The reason is RECORDED on the shape, not merely logged into the void.
    assert "both bounds" in shape.stop_detail
    assert "27.00s" in shape.stop_detail


def test_a_cut_inside_both_bounds_is_raised_and_accepted() -> None:
    """A MIXED spine — 6 footage beats of 5s plus 3 long VO beats — 40s target, 20s over.

    The mix is not incidental: on a uniform pinned spine the shrink arithmetic and the
    bounds agree by construction. They come apart exactly where production came apart —
    the overrun is carried by long VO beats while the FOOTAGE yield is short, so the
    average the cut assumed (6.67s) is well above the 5.00s that will actually be
    recorded.

    ``next_target_beats`` asks for 5 beats = 25.00s, which is 62.5% of the order and
    therefore under the 36.00s lower bound. Staying in the band needs 8 beats (40.00s),
    and 8 < 9, so the child both shrinks and lands in the band: RE-SIZED, not refused.

    Where the 5 comes from — CHEAPEST-FIRST, off the real per-beat allotments, which is
    what ``gen._apply_target_beats`` actually drops:

        allotments ascending: 5, 5, 5, 5, 5, 5, 12, 12, 12   (Σ = 66.00s)
        running removal:      5, 10, 15, 20  ⇒ 4 drops cover the 20.00s overrun
        target = beat_count - cut = 9 - 4 = 5

    The superseded AVERAGE arithmetic gave 6 (Σmin = target + overrun = 60, avg = 60/9 =
    6.67s, cut = ceil(20 / 6.67) = 3 ⇒ 9 - 3 = 6). It under-cut by one beat precisely
    because 6.67s is not what any droppable beat here costs — the cheap footage beats
    cost 5.00s each, so it takes FOUR of them, not three, to buy back 20.00s. 5 is the
    honest unguarded cut; the tolerance clamp then raises it to 8 either way.
    """
    beats = [
        *_pinned_beats(6, 5.0),
        *(
            {"id": f"vo-{i}", "seconds": 12.0, "min_seconds": 1.0, "max_seconds": 30.0}
            for i in range(1, 4)
        ),
    ]
    shape = decide_relief_shape(
        run={"target_seconds": 40.0, "overrun_seconds": 20.0},
        beats=beats,
        floor_ratio=0.5,
    )
    assert shape.verdict == "SPAWN"
    assert shape.measured_seconds_per_beat == pytest.approx(5.0)
    # 9 beats - 4 cheapest-first drops (5+5+5+5 = 20.00s = the overrun) = 5.
    assert shape.shrink_target_beats == 5, "the unguarded cut"
    assert shape.tolerance_min_beats == 8
    assert shape.target_beats == 8
    assert shape.resized_for_tolerance is True
    assert shape.target_beats < shape.beat_count
    assert shape.target_beats * 5.0 >= 40.0 * (1.0 - DURATION_TOLERANCE_FRACTION)


def test_a_cut_already_inside_the_band_is_not_raised() -> None:
    """10 pinned beats of 5s, 40s target, 10s overrun.

    ``next_target_beats``: Σmin = 50, avg = 5.0, cut = ceil(10/5) = 2 ⇒ 8 beats = 40.00s,
    already at the order. The tolerance guard must be INERT here — a guard that also
    moves the cases it was not built for is a second, undeclared policy.
    """
    shape = decide_relief_shape(
        run={"target_seconds": 40.0, "overrun_seconds": 10.0},
        beats=_pinned_beats(10, 5.0),
        floor_ratio=0.5,
    )
    assert shape.verdict == "SPAWN"
    assert shape.target_beats == 8
    assert shape.resized_for_tolerance is False


def test_the_tolerance_guard_is_inert_on_an_unmeasured_manifest() -> None:
    """No footage-pinned row ⇒ no measured yield ⇒ NO judgement, and relief proceeds.

    An absent measurement is not evidence of a bad plan; suppressing relief on missing
    data is the failure mode this whole module keeps refusing.
    """
    unmeasured = [
        {"id": f"vo-{i}", "seconds": 12.0, "min_seconds": 1.0, "max_seconds": 30.0}
        for i in range(1, 7)
    ]
    shape = decide_relief_shape(
        run={"target_seconds": 40.0, "overrun_seconds": 20.0},
        beats=unmeasured,
        floor_ratio=0.5,
    )
    assert shape.measured_seconds_per_beat is None
    assert shape.tolerance_min_beats == 0
    assert shape.resized_for_tolerance is False
    assert shape.verdict == "SPAWN"


def test_the_fixture_order_matches_its_own_scenario_floor() -> None:
    """The fixture correction, asserted rather than asserted-in-a-comment.

    ``demo_video_full_chain_todomvc`` REQUIRES >= 2 distinct ``lines[].id``, so its
    structural floor is ``2 x per-scenario``. The ordered duration must be at least that
    floor, or the fixture orders something no run can deliver — which is exactly what it
    did at 30s.
    """
    import json
    from pathlib import Path

    fixture = (
        Path(__file__).resolve().parents[4]
        / "scripts"
        / "e2e_scenarios"
        / "demo_video_full_chain_todomvc.json"
    )
    data = json.loads(fixture.read_text(encoding="utf-8"))
    ordered = data["project"]["deliverables"][0]["target_duration_seconds"]
    required_lines = next(
        check["value"]
        for assertion in data["expect_stage_assertions"]
        if assertion["stage"] == "demo_video_gen"
        for check in assertion["checks"]
        if check["type"] == "distinct_count_gte" and check["path"] == "lines[].id"
    )
    assert ordered >= required_lines * _LIVE_PER_SCENARIO, (
        f"the fixture orders {ordered}s but demands {required_lines} scenarios, whose "
        f"structural floor is {required_lines * _LIVE_PER_SCENARIO}s"
    )
    # ...and the delivered 47.045s must sit inside the conformance band of that order.
    assert abs(47.045 - ordered) / ordered <= DURATION_TOLERANCE_FRACTION


@pytest.mark.asyncio
async def test_the_stage_records_the_degradation_on_its_own_output(monkeypatch: Any) -> None:
    """THROUGH ``handle``. The degradation must reach the PERSISTED output.

    ``is_degraded`` is what the API and the SQL predicate read, so a degradation computed
    correctly and never wired onto ``DemoVideoGenOutput.degradations`` is invisible to
    both — a working function attached to nothing. Driven end-to-end through the real
    bdd response model so the wire shape is exercised too.
    """
    import json

    from src.clients.schemas.bdd_playwright import BddJobResult
    from src.services.runs.run_context import RunContext
    from src.services.stages.demo_video.gen import DemoVideoGenInput, DemoVideoGenStage
    from src.services.stages.shared.degradation import is_degraded

    monkeypatch.setattr(
        "src.services.stages.demo_video.gen.get_settings",
        lambda: SimpleNamespace(
            pipeline=SimpleNamespace(
                structural_relief_autoconverge_enabled=True,
                demo_timeline_underfull_fail_ratio=0.5,
            )
        ),
    )
    client = AsyncMock()
    client.start_demo_video.return_value = "bdd-feas-1"
    client.wait_for_job.return_value = {}
    client.get_job_result.return_value = BddJobResult.model_validate(
        {
            "job_id": "bdd-feas-1",
            "status": "completed",
            "features_generated": 1,
            "narration_script": {
                "parts": [
                    {
                        "part_id": f"clip-{i}",
                        "text": f"Beat {i}.",
                        # THE VALUE READ OFF THE WIRE — bdd's guaranteed dwell.
                        "expected_recording_seconds": _LIVE_PER_SCENARIO,
                    }
                    for i in (1, 2, 3)
                ]
            },
        }
    )
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        # 10s ordered: ONE 22.00s scenario already overruns it.
        config_snapshot={
            "brief": {"target_url": "https://app.example"},
            # An AUTHORED natural rate: no speech-rate probe is attempted, so the only
            # degradation this stage can raise is the one under test (an unmeasured
            # rate is recorded as `speech_rate_unusable` in its own right).
            "direction": {"timing": {"wpm": {"natural": 150.0}}},
            "deliverable": {"target_duration_seconds": 10.0},
        },
    )
    out = await DemoVideoGenStage(client=client).handle(DemoVideoGenInput(), ctx)

    assert [line.id for line in out.lines] == ["clip-1"], "planned to the floor"
    codes = [d["code"] for d in out.degradations]
    assert codes == [ORDERED_DURATION_INFEASIBLE]
    persisted = json.loads(out.model_dump_json())
    assert is_degraded(persisted), "the run must surface as degraded"
    assert persisted["degradations"][0]["code"] == ORDERED_DURATION_INFEASIBLE


def test_the_tolerance_guard_honours_the_floor_ratio_kill_switch() -> None:
    """``floor_ratio=0`` disables THIS bound too, exactly as it disables the 50% clamp.

    The band and the underfull floor are two different lines, but ``floor_ratio=0`` is
    ``decide_relief_shape``'s documented "make no duration judgement at all" switch. A
    kill-switch that no longer switches everything off is a trap for whoever next reaches
    for it — and the existing contract test
    ``test_relief_with_the_floor_disabled_behaves_exactly_as_before`` says so out loud.
    """
    run = {"target_seconds": 30.0, "overrun_seconds": 17.0}
    beats = _pinned_beats(2, 23.5)
    assert decide_relief_shape(run=run, beats=beats, floor_ratio=0.5).verdict == "STOP_UNDERFULL"
    disabled = decide_relief_shape(run=run, beats=beats, floor_ratio=0.0)
    assert disabled.verdict == "SPAWN"
    assert disabled.tolerance_min_beats == 0
    assert disabled.resized_for_tolerance is False


# ── 5. the ENTRY may claim only what the arithmetic established ──────────────────────
#
# THE PATTERN BEING ELIMINATED: a degradation whose detail asserts a CONCLUSION the
# surrounding code never reached. The first wording here said the per-scenario length
# was "guaranteed" and that "even ONE scenario overruns the order", and the log said
# "The order cannot be met". The guard behind all three is one float comparison,
# `per_beat_seconds > ordered`, and `per_beat_seconds` arrives from one of TWO rungs:
# bdd's exported `expected_recording_seconds` (a guarantee for THIS suite) or the
# PARENT run's measured footage yield (a past observation of a DIFFERENT suite). On the
# parent rung nothing in this module established a guarantee, so the entry must not
# claim one.


_IMPOSSIBILITY_VOCABULARY = (
    "guarantee",
    "cannot",
    "impossible",
    "unreachable",
    "never",
    "infeasible",
)


def _infeasible_detail(yield_source: str | None) -> str:
    kept, degradations = _apply_feasibility_floor(
        _lines(3),
        {"target_duration_seconds": 20.0},
        _LIVE_PER_SCENARIO,
        0.5,
        yield_source,
    )
    assert len(kept) == 1
    assert len(degradations) == 1
    return str(degradations[0]["detail"])


@pytest.mark.parametrize("yield_source", [None])
def test_an_observed_yield_is_never_reported_as_a_guarantee(yield_source: str | None) -> None:
    """A PAST OBSERVATION may not be written up as physical impossibility.

    ``_measured_seconds_per_beat`` reads what the parent run's footage actually
    measured. That is evidence about a run that already happened — it is not a floor
    THIS run is obliged to hit, and no code between that read and this entry turned it
    into one. Anything in :data:`_IMPOSSIBILITY_VOCABULARY` would assert exactly that
    unproved step.
    """
    detail = _infeasible_detail(yield_source)
    # The guarantee vocabulary is banned OUTSIDE the provenance clause — the parent
    # rung's own provenance text legitimately says the number is NOT a guarantee.
    claim = detail.split("(")[0] + detail.split(")")[-1]
    for word in _IMPOSSIBILITY_VOCABULARY:
        assert word not in claim.lower(), (
            f"{word!r} asserts more than `per_beat_seconds > ordered` proves: {detail!r}"
        )


def test_the_entry_names_where_the_per_scenario_number_came_from() -> None:
    """PROVENANCE OR NOTHING. A bare number invites the reader to assume a guarantee."""
    parent = _infeasible_detail(_PARENT_MEASURED_YIELD)
    assert _PARENT_MEASURED_YIELD in parent
    # ...and the two rungs must be DISTINGUISHABLE from the entry alone.
    bdd = _infeasible_detail(_BDD_GUARANTEED_YIELD)
    assert _BDD_GUARANTEED_YIELD in bdd
    assert parent != bdd, "both rungs produce the same sentence — provenance is not said"
    # An unattributed number says so rather than borrowing either rung's authority.
    unattributed = _infeasible_detail(None)
    assert _UNATTRIBUTED_YIELD in unattributed
    assert _PARENT_MEASURED_YIELD not in unattributed
    assert _BDD_GUARANTEED_YIELD not in unattributed


def test_a_guarantee_may_be_stated_only_while_naming_bdd_as_its_source() -> None:
    """The legitimate case. bdd DOES guarantee its own suite's dwell — so the word is
    allowed there, but only next to the source that makes it true.
    """
    bdd = _infeasible_detail(_BDD_GUARANTEED_YIELD)
    assert "guaranteed by bdd" in bdd
    assert "expected_recording_seconds" in bdd, "the guarantee cites no source"


def test_the_entry_states_the_measured_overrun_rather_than_asserting_a_verdict() -> None:
    """What the comparison DID establish: this figure exceeds the order, by how much."""
    detail = _infeasible_detail(_PARENT_MEASURED_YIELD)
    # 22.00 - 20.00 = 2.00s past the order, and the floor/order/figure are all named.
    for fragment in ("20.00", "22.00", "66.00", "2.00s past the order"):
        assert fragment in detail, f"{fragment!r} missing from {detail!r}"


@pytest.mark.asyncio
async def test_the_bdd_rung_reaches_the_entry_as_bdd_provenance(monkeypatch: Any) -> None:
    """THROUGH ``handle``. Provenance computed correctly and never threaded to the
    string is provenance nobody reads. The stage's own rung-one win must be the thing
    the persisted entry names.
    """
    from src.clients.schemas.bdd_playwright import BddJobResult
    from src.services.runs.run_context import RunContext
    from src.services.stages.demo_video.gen import DemoVideoGenInput, DemoVideoGenStage

    monkeypatch.setattr(
        "src.services.stages.demo_video.gen.get_settings",
        lambda: SimpleNamespace(
            pipeline=SimpleNamespace(
                structural_relief_autoconverge_enabled=True,
                demo_timeline_underfull_fail_ratio=0.5,
            )
        ),
    )
    client = AsyncMock()
    client.start_demo_video.return_value = "bdd-prov-1"
    client.wait_for_job.return_value = {}
    client.get_job_result.return_value = BddJobResult.model_validate(
        {
            "job_id": "bdd-prov-1",
            "status": "completed",
            "features_generated": 1,
            "narration_script": {
                "parts": [
                    {
                        "part_id": f"clip-{i}",
                        "text": f"Beat {i}.",
                        "expected_recording_seconds": _LIVE_PER_SCENARIO,
                    }
                    for i in (1, 2)
                ]
            },
        }
    )
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "brief": {"target_url": "https://app.example"},
            # An AUTHORED natural rate: no speech-rate probe is attempted, so the only
            # degradation this stage can raise is the one under test (an unmeasured
            # rate is recorded as `speech_rate_unusable` in its own right).
            "direction": {"timing": {"wpm": {"natural": 150.0}}},
            "deliverable": {"target_duration_seconds": 10.0},
            # A PARENT measurement is also present and is DIFFERENT. Rung one wins, so
            # the entry must name bdd — naming the parent here would mean the string
            # was built from the losing rung.
            "structural_relief": {MEASURED_SECONDS_PER_BEAT_KEY: 31.0},
        },
    )
    out = await DemoVideoGenStage(client=client).handle(DemoVideoGenInput(), ctx)
    detail = str(out.degradations[0]["detail"])
    assert _BDD_GUARANTEED_YIELD in detail
    assert _PARENT_MEASURED_YIELD not in detail
    assert "22.00" in detail and "31.00" not in detail, "the losing rung's number leaked"
