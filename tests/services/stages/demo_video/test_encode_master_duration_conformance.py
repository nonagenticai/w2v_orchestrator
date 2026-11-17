"""``demo_video_encode_master`` RECORDS its delivered-vs-ORDERED duration.

``duration_conformance`` was wired into ``promo``, ``lesson``, ``website_review``
and ``app_review`` on 2026-09-10 and NOT into demo_video's terminal stage — which
is the blueprint that then shipped run ``01a08afc`` at **47.662s against a 30s
order (159%)** with no field, note or log line anywhere in the run record saying
so. A conformance failure invisible in the run record can never be
regression-tested; these tests pin the record.

RECORDING, NOT ENFORCEMENT: nothing here may truncate a master or fail a stage,
so every test also asserts ``duration_seconds`` came out untouched.

⛔ THIS FILE USED TO PIN THE DEFECT AS THE CONTRACT
--------------------------------------------------
It asserted, as an intended invariant, that "EVERY verdict this stage produces is
``declared``, BY CONSTRUCTION", on the reasoning that "``EncodeMasterOutput``
carries neither a ``mastering`` report nor an ``actual_duration_seconds``".
**The first half of that is false.** ``EncodeMasterOutput.mastering``
(``encode_master.py:270``) is the compiler's own mastering report, fed from
``compile_out["mastering"]`` and replaced by the re-mastered one when the critic
loop converges — precisely the shape
:func:`~src.services.stages.shared.duration_conformance.playable_seconds_from_mastering`
reads. ``declared`` means the plan grading itself, so the file was asserting that
this stage may never grade the delivered FILE. It now passes
``playable_seconds=``, and these tests assert the measured rung.

⚠️ ``declared`` IS STILL REACHABLE, and is still the right verdict when it is
----------------------------------------------------------------------------
The mastering block is absent on a compile that never mastered, and
``mastering.degraded_report()`` nulls BOTH stream fields when ``post_mux_verify``
crashes. Those fall through to ``declared`` + :data:`DURATION_UNMEASURED`, and
the tests that exercise that path pass ``compile_mastering=None`` to say so
explicitly rather than by accident.

⛔ THE TIER IS THE POINT, AND IT IS WHY THIS FILE MATTERS
--------------------------------------------------------
:data:`DURATION_UNMEASURED` rides ``advisories``, NOT ``degradations``. "Nobody
measured this" is a statement about OUR instrumentation, not a defect in the
artifact, and on the degrading channel it would report every unmasterable
delivery ``degraded=True`` including perfectly conforming ones — the
near-universal-flag failure the advisory tier exists to prevent. It spent one
commit on ``degradations``; ``test_a_conforming_unprobed_master_is_not_degraded``
below is the regression guard.

The exact-list assertions below name the channel deliberately: loosening one to a
subset/``in`` check would let the basis silently regress back to the plan grading
itself, which is the defect the basis field exists to make impossible to miss.

``DURATION_NOT_ORDERED`` is the one verdict that stays a single-code list: it is
returned for ``result is None``, where there is no basis to stamp at all.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from src.services.runs.run_context import RunContext
from src.services.stages.demo_video.encode_master import EncodeMasterInput, EncodeMasterStage
from src.services.stages.shared.degradation import (
    DEGRADATION_CODES,
    DURATION_NOT_ORDERED,
    DURATION_OUT_OF_TOLERANCE,
    DURATION_PICTURE_TRUNCATED,
    DURATION_PLAN_INFEASIBLE,
    DURATION_UNMEASURED,
    ORDERED_DURATION_INFEASIBLE,
    STRUCTURAL_RELIEF_REFUSED,
    is_degraded,
)
from src.services.stages.shared.duration_conformance import (
    DELIVERED_BASIS_DECLARED,
    DELIVERED_BASIS_PLAYABLE,
    demo_video_ordered_seconds,
)

# The live 2026-09-10 measurement this wiring exists to make visible.
_ORDERED = 30.0
_DELIVERED = 47.662


def _streams(video: float | None, audio: float | None) -> dict[str, Any]:
    """A compiler mastering report carrying the two SEPARATELY probed stream lengths.

    This is the block ``playable_seconds_from_mastering`` reads. Handing it to
    ``_ctx(compile_mastering=...)`` is what moves a verdict off ``declared``.
    """
    return {"video_stream_seconds": video, "audio_stream_seconds": audio}


def _ctx(
    *,
    duration: float = _DELIVERED,
    deliverable: dict[str, Any] | None = None,
    direction: dict[str, Any] | None = None,
    mastering: dict[str, Any] | None = None,
    compile_mastering: dict[str, Any] | None = None,
    reconcile: dict[str, Any] | None = None,
) -> RunContext:
    """``mastering`` is the AUTHORED config (the critic opt-in); ``compile_mastering``
    is the compiler's measured REPORT on the prior stage's output. They are different
    things and only the second one can produce a measured verdict."""
    config: dict[str, Any] = {}
    if deliverable is not None:
        config["deliverable"] = deliverable
    if direction is not None:
        config["direction"] = direction
    if mastering is not None:
        config["mastering"] = mastering
    compile_out: dict[str, Any] = {
        "artifact_url": "http://v/master.mp4",
        "duration_seconds": duration,
    }
    if compile_mastering is not None:
        compile_out["mastering"] = compile_mastering
    prior: dict[str, Any] = {"demo_video_compile": compile_out}
    if reconcile is not None:
        # The upstream PLAN verdict. Both blueprints that reach encode_master put
        # ``reconcile_timeline`` above it, so this key is present on every live run;
        # the tests that omit it are pinning the "no refusal known" fallback.
        prior["reconcile_timeline"] = reconcile
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot=config,
        prior_run_outputs=prior,
    )


def _stage() -> EncodeMasterStage:
    return EncodeMasterStage(client=AsyncMock(), compile_client=AsyncMock())


async def test_the_159_percent_delivery_reaches_the_stage_output() -> None:
    """Run 01a08afc's numbers, now in the record instead of nowhere."""
    out = await _stage().handle(
        EncodeMasterInput(), _ctx(deliverable={"target_duration_seconds": _ORDERED})
    )

    conformance = out.duration_conformance
    assert conformance is not None, "the delivered-vs-ordered comparison was not recorded"
    assert conformance["ordered_seconds"] == pytest.approx(_ORDERED)
    assert conformance["delivered_seconds"] == pytest.approx(_DELIVERED)
    assert conformance["ratio"] == pytest.approx(1.5887, abs=1e-3)
    assert conformance["within_tolerance"] is False
    # Observation, never truncation.
    assert out.duration_seconds == pytest.approx(_DELIVERED)


async def test_breach_lands_on_the_degradation_channel() -> None:
    """The queryable verdict beside the numbers — and the run is not FAILED by it.

    The compile stage handed this run a real mastering report, so the verdict is
    taken from the PLAYABLE length of the delivered file — ``playable``, not
    ``declared`` — and no ``DURATION_UNMEASURED`` advisory is raised: there is
    nothing unmeasured about it. The CHANNELS are still pinned: the breach
    degrades the run and nothing else rides along.
    """
    out = await _stage().handle(
        EncodeMasterInput(),
        _ctx(
            deliverable={"target_duration_seconds": _ORDERED},
            compile_mastering=_streams(_DELIVERED, _DELIVERED),
        ),
    )

    assert [d["code"] for d in out.degradations] == [DURATION_OUT_OF_TOLERANCE]
    assert [a["code"] for a in out.advisories] == []
    assert "47.7s" in out.degradations[0]["detail"]
    assert out.duration_conformance is not None
    assert out.duration_conformance["delivered_basis"] == DELIVERED_BASIS_PLAYABLE
    assert out.artifact_url == "http://v/master.mp4"


async def test_an_order_that_is_already_met_is_not_flagged() -> None:
    """FALSE-POSITIVE GUARD: a conforming master must not be altered or degraded.

    The whole hazard of a duration mechanism is that it fires on runs that were
    already right. A 30s master against a 30s order records ``within_tolerance``
    and raises NO tolerance verdict and NO note — so nothing was altered and the
    breach channel stays silent.

    And it is a MEASURED pass, not the plan agreeing with itself: the compile
    stage's mastering report puts both streams at the ordered length, so the
    basis is ``playable`` and BOTH lists are empty. Asserted exactly — an extra
    code appearing, or the basis sliding back to ``declared`` (which would mean
    this stage stopped reading the report and went back to grading its own plan),
    must each fail.
    """
    out = await _stage().handle(
        EncodeMasterInput(),
        _ctx(
            duration=_ORDERED,
            deliverable={"target_duration_seconds": _ORDERED},
            compile_mastering=_streams(_ORDERED, _ORDERED),
        ),
    )

    conformance = out.duration_conformance
    assert conformance is not None
    assert conformance["within_tolerance"] is True
    assert conformance["ratio"] == pytest.approx(1.0)
    assert conformance["delivered_basis"] == DELIVERED_BASIS_PLAYABLE
    assert out.degradations == []
    assert out.advisories == []
    assert out.notes == []
    assert out.duration_seconds == pytest.approx(_ORDERED)


async def test_the_measured_file_outranks_the_plan_that_declared_it() -> None:
    """⭐ THE DEFECT THIS WAVE EXISTS TO CLOSE — the plan grading itself.

    The compile stage DECLARES 30.0s (the nominal sum of shot durations, derived
    from the very 30s order it is being graded against, so it agrees by
    construction), and the delivered file measures 47.662s of playable content.
    Before ``playable_seconds=`` was passed, this master recorded
    ``within_tolerance=True`` with an empty degradations list — a 159% delivery
    reported as on target, by the check written to catch exactly that.

    A ``declared`` verdict here is therefore not a weaker pass, it is the wrong
    answer, and the basis field is asserted so the two can never be confused.
    """
    out = await _stage().handle(
        EncodeMasterInput(),
        _ctx(
            duration=_ORDERED,
            deliverable={"target_duration_seconds": _ORDERED},
            compile_mastering=_streams(_DELIVERED, _DELIVERED),
        ),
    )

    conformance = out.duration_conformance
    assert conformance is not None
    assert conformance["delivered_basis"] == DELIVERED_BASIS_PLAYABLE
    assert conformance["delivered_seconds"] == pytest.approx(_DELIVERED)
    assert conformance["within_tolerance"] is False
    assert [d["code"] for d in out.degradations] == [DURATION_OUT_OF_TOLERANCE]
    assert is_degraded(out.model_dump()) is True
    # RECORDING, never truncation: the shipped length is untouched.
    assert out.duration_seconds == pytest.approx(_ORDERED)


async def test_a_truncated_picture_reaches_the_degradation_channel() -> None:
    """⭐ 2e — the picture-vs-audio gap, which this blueprint never computed.

    ``picture_truncation_degradations`` had five call sites (action_reel,
    app_review, lesson, promo, website_review) and none on the demo_video /
    showcase path, so a master whose picture stopped 40s before its narration
    shipped with nothing in the record saying so.

    The conformance verdict CANNOT express it — it grades the PLAYABLE length,
    which on a truncated master IS the audio, so it reads on target (asserted
    below, because that is the whole reason this needs its own code).
    """
    out = await _stage().handle(
        EncodeMasterInput(),
        _ctx(
            duration=_ORDERED,
            deliverable={"target_duration_seconds": _ORDERED},
            compile_mastering=_streams(20.0, _ORDERED),
        ),
    )

    conformance = out.duration_conformance
    assert conformance is not None
    assert conformance["within_tolerance"] is True, (
        "the playable-length verdict is on target — which is exactly why the "
        "truncation needs a code of its own"
    )
    assert [d["code"] for d in out.degradations] == [DURATION_PICTURE_TRUNCATED]
    assert "10.0s before the audio" in out.degradations[0]["detail"]
    assert is_degraded(out.model_dump()) is True


async def test_a_degraded_mastering_report_falls_back_to_declared() -> None:
    """``mastering.degraded_report()`` nulls BOTH stream fields when
    ``post_mux_verify`` crashes, and the compiler's ``0.0`` is an unknown
    sentinel, not a zero-length video. Neither may be graded: they fall to
    ``declared`` + the unmeasured ADVISORY, and raise no truncation claim off a
    measurement that does not exist."""
    out = await _stage().handle(
        EncodeMasterInput(),
        _ctx(
            duration=_ORDERED,
            deliverable={"target_duration_seconds": _ORDERED},
            compile_mastering=_streams(0.0, None),
        ),
    )

    conformance = out.duration_conformance
    assert conformance is not None
    assert conformance["delivered_basis"] == DELIVERED_BASIS_DECLARED
    assert out.degradations == []
    assert [a["code"] for a in out.advisories] == [DURATION_UNMEASURED]


async def test_a_conforming_unprobed_master_is_not_degraded() -> None:
    """⭐ THE REGRESSION THIS TIERING EXISTS TO PREVENT.

    A compile that never mastered (``compile_mastering=None`` here, the default)
    leaves the delivery unprobed. If ``DURATION_UNMEASURED`` were a degradation,
    a perfectly conforming delivery would read ``degraded=True``. It spent one
    commit there and did exactly that.

    So: a conforming, unprobed master reads CLEAN through ``is_degraded`` (the
    single Python definition, SQL twin ``run_repository.degraded_stage_clause``)
    while STILL carrying the unmeasured fact for anyone querying it. Both halves
    are asserted — a run that is clean because the signal was dropped entirely
    would be the other failure mode, not a fix.
    """
    out = await _stage().handle(
        EncodeMasterInput(),
        _ctx(duration=_ORDERED, deliverable={"target_duration_seconds": _ORDERED}),
    )

    assert [a["code"] for a in out.advisories] == [DURATION_UNMEASURED]
    assert is_degraded(out.model_dump()) is False


async def test_no_ordered_duration_records_nothing_rather_than_conforming() -> None:
    """ "Nobody ordered a length" is not "the length was met".

    demo_video has NO blueprint default — ``reconcile_timeline`` skips clean
    without a target rather than inventing one — so an absent order must leave
    the field ``None``, never ``within_tolerance=True``.

    The absence is REPORTED rather than silent: a ``None`` verdict is invisible
    to any ``within_tolerance = false`` query, so the missing order surfaces on
    the degradations channel as ``DURATION_NOT_ORDERED``. That is not a verdict
    on the length — it records that no length was ever ordered.
    """
    out = await _stage().handle(EncodeMasterInput(), _ctx(deliverable={"format": "16x9"}))

    assert out.duration_conformance is None
    assert [d["code"] for d in out.degradations] == [DURATION_NOT_ORDERED]


async def test_direction_timing_target_outranks_the_deliverable() -> None:
    """demo_video's OWN ladder (reconcile_timeline.py:1005-1009), not brief-only.

    ``direction.timing.target_seconds`` is the explicit override the timing plan
    honours first. Resolving from the deliverable alone would score a 45s film
    ordered at 45s as 150% of a phantom 30s order.
    """
    ctx = _ctx(
        duration=45.0,
        deliverable={"target_duration_seconds": _ORDERED},
        direction={"timing": {"mode": "auto", "target_seconds": 45}},
    )
    out = await _stage().handle(EncodeMasterInput(), ctx)

    conformance = out.duration_conformance
    assert conformance is not None
    assert conformance["ordered_seconds"] == pytest.approx(45.0)
    assert conformance["within_tolerance"] is True


async def test_the_loop_failure_exit_also_records_the_comparison() -> None:
    """A soft-degraded master is still a PAID, delivered master.

    Stamping only the converged exit would leave every critic/loop failure with
    no length record — the same invisibility, narrowed rather than fixed.

    The soft-degrade exit gets the SAME two entries as the converged one: the
    breach on ``degradations``, plus the ``DURATION_UNMEASURED`` advisory because
    encode_master cannot reach the compiler's ffprobe (see the module docstring).
    Asserting the exact lists on both exits is what stops one of them drifting
    into a different verdict shape — or a different TIER.
    """
    ctx = _ctx(
        deliverable={"target_duration_seconds": _ORDERED},
        mastering={"critic": {"iterations": 1}},
    )
    with patch(
        "src.services.stages.demo_video.encode_master.run_evaluator_optimizer",
        AsyncMock(side_effect=RuntimeError("loop exploded")),
    ):
        out = await _stage().handle(EncodeMasterInput(), ctx)

    conformance = out.duration_conformance
    assert conformance is not None
    assert conformance["within_tolerance"] is False
    assert conformance["delivered_basis"] == DELIVERED_BASIS_DECLARED
    assert [d["code"] for d in out.degradations] == [DURATION_OUT_OF_TOLERANCE]
    assert [a["code"] for a in out.advisories] == [DURATION_UNMEASURED]
    # The loop-failure note is still there — conformance EXTENDS the record.
    assert any("WITHOUT being assessed" in note for note in out.notes)


async def test_conformance_survives_model_dump() -> None:
    """Stage outputs persist via ``model_dump``; a field that serialises away is
    exactly as invisible as the deviation it exists to surface."""
    dumped = (
        await _stage().handle(
            EncodeMasterInput(), _ctx(deliverable={"target_duration_seconds": _ORDERED})
        )
    ).model_dump()

    assert dumped["duration_conformance"]["ratio"] == pytest.approx(1.5887, abs=1e-3)
    assert dumped["degradations"][0]["code"] == DURATION_OUT_OF_TOLERANCE


@pytest.mark.parametrize(
    ("snapshot", "expected"),
    [({"deliverable": {"target_duration_seconds": 30}}, 30.0)],
)
def test_ordered_seconds_ladder(snapshot: dict[str, Any], expected: float | None) -> None:
    assert demo_video_ordered_seconds(snapshot) == expected


# ── the breach must also reach the PROSE channel ─────────────────────────────
# Run 01a09172 (2026-09-11) recorded `ratio 1.5472` / `within_tolerance: false` AND a
# `duration_out_of_tolerance` degradation — and the run rendered as NOT degraded with
# `notes: []`. `degraded` is derived from the note channel (`is_degraded`, SQL twin
# `run_repository.degraded_stage_clause`), so a breach that writes only the structured
# entry is invisible to every reader that predates that channel.
async def test_the_breach_also_lands_on_the_prose_note_channel() -> None:
    out = await _stage().handle(
        EncodeMasterInput(), _ctx(deliverable={"target_duration_seconds": _ORDERED})
    )

    duration_notes = [n for n in out.notes if "ordered" in n]
    assert len(duration_notes) == 1, f"the breach raised no prose note: {out.notes}"
    note = duration_notes[0]
    assert note.startswith("encode_master: ")
    # ONE measurement, never two wordings: the note carries the degradation's own detail.
    assert note == f"encode_master: {out.degradations[0]['detail']}"
    assert "47.7s" in note and "30.0s" in note and "159%" in note
    assert is_degraded(out.model_dump()) is True


async def test_a_missing_order_stays_prose_silent() -> None:
    """BLAST-RADIUS GUARD, the one that matters most here.

    ``duration_not_ordered`` fires on every deliverable that never specified a length,
    so only the out-of-tolerance verdict is mirrored into the PROSE channel. The
    structured entry still records the absence.

    ⚠️ PROSE-SILENT IS NOT DEGRADED-SILENT, and ``encode_master._conformed`` used to
    carry a comment claiming it was. ``is_degraded`` is ``bool(degradations) or
    bool(notes)`` — an OR over BOTH channels — so the structured entry flips
    ``degraded`` on its own. That is the shared module's stated intent ("'nothing was
    ordered' is not 'on target'"), and it is asserted here so the stale reasoning
    cannot come back as a silent behaviour change.
    """
    out = await _stage().handle(EncodeMasterInput(), _ctx(deliverable={"format": "16x9"}))

    assert [d["code"] for d in out.degradations] == [DURATION_NOT_ORDERED]
    assert out.notes == []
    assert is_degraded(out.model_dump()) is True


async def test_a_conforming_master_adds_no_note() -> None:
    """FALSE-POSITIVE GUARD: an order that was MET must stay prose-clean.

    The prose channel is what ``is_degraded`` reads today, and ONLY
    ``DURATION_OUT_OF_TOLERANCE`` is mirrored into it (``encode_master._conformed``)
    — so a met order raises no note even though its verdict is ``declared``.
    ``DURATION_UNMEASURED`` is the one structured entry and it is an ADVISORY,
    asserted exactly: it exists because encode_master cannot reach the compiler's
    ffprobe (see the module docstring), so this is an unverified verdict rather
    than a pass — and an unverified verdict must not degrade the run.
    """
    out = await _stage().handle(
        EncodeMasterInput(),
        _ctx(duration=_ORDERED, deliverable={"target_duration_seconds": _ORDERED}),
    )

    assert out.notes == []
    assert out.degradations == []
    assert [a["code"] for a in out.advisories] == [DURATION_UNMEASURED]


# --------------------------------------------------------------------------- #
# A BAND PASS ON A PLAN THE PIPELINE ITSELF REFUSED — prod run
# 01a0a3c6-c8e7-73db-b13e-fded1a4cbb4f. reconcile_timeline stamped the timeline
# INFEASIBLE (feasible: [False, False, False], convergence_outcome 'cant_converge',
# half the beats already floored to vo_rate 0.8) and raised
# ordered_duration_infeasible; this stage then graded 32.038s against a 30.0s order
# as within_tolerance=True at +6.79%, purely because its band is +/-10% while the
# refusal has NO band at all. The band was right. Quoting it alone was the defect.
#
# ⛔ NOT A GATE. Every test below asserts `degradations` UNCHANGED and
# `within_tolerance` STILL TRUE — the signal rides `advisories` only. Sample size
# post-rollout is one run.
# --------------------------------------------------------------------------- #

_REFUSED_DELIVERED = 32.038


def _refused_plan(outcome: str = "cant_converge") -> dict[str, Any]:
    return {
        "timeline_manifest": {"beats": [], "run": {"convergence_outcome": outcome}},
        "degradations": [{"code": ORDERED_DURATION_INFEASIBLE, "detail": "ordered 30.00s; …"}],
    }


async def test_a_refused_plan_is_named_beside_the_band_pass() -> None:
    out = await _stage().handle(
        EncodeMasterInput(),
        _ctx(
            duration=_REFUSED_DELIVERED,
            deliverable={"target_duration_seconds": _ORDERED},
            compile_mastering=_streams(_REFUSED_DELIVERED, _REFUSED_DELIVERED),
            reconcile=_refused_plan(),
        ),
    )

    conformance = out.duration_conformance
    assert conformance is not None
    # The numbers are UNTOUCHED — this changes what is reported, not what is graded.
    assert conformance["deviation_pct"] == pytest.approx(6.79, abs=0.01)
    assert conformance["within_tolerance"] is True
    assert conformance["delivered_basis"] == DELIVERED_BASIS_PLAYABLE
    # …and the refusal now travels with them.
    assert conformance["plan_infeasible"] is True
    assert "cant_converge" in (conformance["plan_infeasible_reason"] or "")
    assert ORDERED_DURATION_INFEASIBLE in (conformance["plan_infeasible_reason"] or "")

    assert [a["code"] for a in out.advisories] == [DURATION_PLAN_INFEASIBLE]
    # ⛔ THE RESTRAINT GUARD: the degrading channel and the prose channel are
    # untouched, so nothing that shipped clean before ships degraded now.
    assert out.degradations == []
    assert out.notes == []
    assert out.duration_seconds == pytest.approx(_REFUSED_DELIVERED)


async def test_a_converged_plan_adds_nothing() -> None:
    """FALSE-POSITIVE GUARD — the healthy population is byte-identical to before."""
    out = await _stage().handle(
        EncodeMasterInput(),
        _ctx(
            duration=_ORDERED,
            deliverable={"target_duration_seconds": _ORDERED},
            compile_mastering=_streams(_ORDERED, _ORDERED),
            reconcile={
                "timeline_manifest": {"beats": [], "run": {"convergence_outcome": "converged"}}
            },
        ),
    )

    assert out.duration_conformance is not None
    assert out.duration_conformance["plan_infeasible"] is False
    assert out.advisories == []
    assert out.degradations == []
    assert out.notes == []


async def test_a_run_with_no_reconcile_output_is_not_reported_as_refused() -> None:
    """A replay/regenerate that re-entered below reconcile must read as "unknown".

    Absent is not a refusal. It must also not raise on the exit path of a paid
    render — the assertion here is that the stage completes at all.
    """
    out = await _stage().handle(
        EncodeMasterInput(),
        _ctx(
            duration=_ORDERED,
            deliverable={"target_duration_seconds": _ORDERED},
            compile_mastering=_streams(_ORDERED, _ORDERED),
        ),
    )

    assert out.duration_conformance is not None
    assert out.duration_conformance["plan_infeasible"] is False
    assert out.advisories == []


async def test_a_refused_plan_outside_the_band_raises_only_the_tolerance_breach() -> None:
    """Out of band, ``duration_out_of_tolerance`` already says the order was missed.

    The advisory exists to stop a REASSURING verdict, and a failing verdict is not
    reassuring. Adding it here too would be a near-universal second flag on exactly
    the population already flagged.
    """
    out = await _stage().handle(
        EncodeMasterInput(),
        _ctx(
            deliverable={"target_duration_seconds": _ORDERED},
            compile_mastering=_streams(_DELIVERED, _DELIVERED),
            reconcile=_refused_plan(),
        ),
    )

    assert [d["code"] for d in out.degradations] == [DURATION_OUT_OF_TOLERANCE]
    assert [a["code"] for a in out.advisories] == []


# ── the relief gate's REFUSAL reaches the run record ─────────────────────────────────
#
# THE DEFECT THIS PINS. `structural_relief_policy.decide_relief_shape` detects an order
# no legal beat count can satisfy, at plan time, before anything paid happens, refuses to
# bill the render, and writes a full sentence of arithmetic onto `ReliefShape.stop_detail`
# saying why. On `origin/main`, `git grep -n "STOP_UNDERFULL\|stop_detail" -- src/` matched
# ONLY `structural_relief.py` and `structural_relief_policy.py`: the verdict never left the
# pair of files that produce it. It reached no conformance output and no degradation code,
# so a run shipped out of band with no visible explanation and three separate sessions
# inferred three different wrong mechanisms for short/long films.

#: The live captured verdict, reproduced exactly: a 44.00s order against 5 footage-pinned
#: beats measuring 12.424s each. 5 x 12.424 = 62.12s, i.e. 18.12s over. The gate enumerates
#: every legal count below 5 and finds 3 = 37.27s (under the 39.60s lower bound) and
#: 4 = 49.70s (over the 48.40s upper bound) — no count lands inside the +/-10% band.
_RELIEF_ORDERED = 44.0
_RELIEF_PER_BEAT = 12.424
_RELIEF_BEATS = 5
_RELIEF_OVERRUN = _RELIEF_BEATS * _RELIEF_PER_BEAT - _RELIEF_ORDERED
#: What that run actually delivered.
_RELIEF_DELIVERED = 54.507


def _unsatisfiable_plan() -> dict[str, Any]:
    """A `reconcile_timeline` output carrying the live STOP_UNDERFULL manifest."""
    return {
        "timeline_manifest": {
            "beats": [
                {
                    "id": f"clip-{i}",
                    "seconds": _RELIEF_PER_BEAT,
                    "min_seconds": _RELIEF_PER_BEAT,
                    "max_seconds": _RELIEF_PER_BEAT,
                }
                for i in range(1, _RELIEF_BEATS + 1)
            ],
            "run": {
                "convergence_outcome": "cant_converge",
                "target_seconds": _RELIEF_ORDERED,
                "overrun_seconds": _RELIEF_OVERRUN,
            },
        },
    }


async def test_the_relief_gates_refusal_reaches_the_degradation_channel() -> None:
    """THE POINT OF THIS WAVE: the verdict is no longer write-only.

    Asserts the WHOLE path — the manifest `reconcile_timeline` persisted, through
    `relief_shape_from_manifest`, onto `encode_master`'s stage output — not just that the
    reader returns something. If any link stops carrying it, this goes red.
    """
    out = await _stage().handle(
        EncodeMasterInput(),
        _ctx(
            duration=_RELIEF_DELIVERED,
            deliverable={"target_duration_seconds": _RELIEF_ORDERED},
            compile_mastering=_streams(_RELIEF_DELIVERED, _RELIEF_DELIVERED),
            reconcile=_unsatisfiable_plan(),
        ),
    )

    codes = [d["code"] for d in out.degradations]
    assert STRUCTURAL_RELIEF_REFUSED in codes, (
        f"the relief gate's refusal did not reach the stage output — got {codes}"
    )
    assert STRUCTURAL_RELIEF_REFUSED in DEGRADATION_CODES
    # …and it flips the run's degraded flag, i.e. it is visible, not merely stored.
    assert is_degraded(out.model_dump())

    entry = next(d for d in out.degradations if d["code"] == STRUCTURAL_RELIEF_REFUSED)
    detail = entry["detail"]
    # The VERDICT, attributed to the gate. encode_master passes the MEASURED delivery
    # (54.507s, +23.9%, above the band), but the footage basis already refuses — no legal
    # beat count lands in band at 12.42s/beat — and a footage refusal is never re-opened
    # by the delivery, so that is the refusal the spawner will make too.
    assert "STOP_UNDERFULL" in detail
    # The sizing basis is named as a sentence a customer can read, never as an identifier.
    assert "sized from this run's footage, which ruled it out before the delivered film" in detail
    assert "footage_basis_refused" not in detail
    # The gate's own arithmetic — the sentence that previously reached nothing — so a
    # reader can see WHY it refused and not merely THAT it did.
    for fragment in ("44.00s", "no legal beat count lands inside the +/-10% band"):
        assert fragment in detail, f"{fragment!r} missing from {detail!r}"
    # The per-beat figure is attributed to THIS run's footage, never stated as a constant.
    assert "12.424s/beat as measured on THIS run's footage" in detail


async def test_an_unsatisfiable_order_is_visible_but_never_terminal() -> None:
    """⛔ VISIBILITY ONLY. A run that terminates never reaches `mark_completed`, and the
    relief hook is completion-gated — so enforcing here would silently disable relief
    spawning outright. The stage must still return its master."""
    out = await _stage().handle(
        EncodeMasterInput(),
        _ctx(
            duration=_RELIEF_DELIVERED,
            deliverable={"target_duration_seconds": _RELIEF_ORDERED},
            compile_mastering=_streams(_RELIEF_DELIVERED, _RELIEF_DELIVERED),
            reconcile=_unsatisfiable_plan(),
        ),
    )
    assert out.artifact_url is not None
    assert out.duration_seconds == pytest.approx(_RELIEF_DELIVERED)


async def test_a_spawnable_overrun_reports_no_refusal() -> None:
    """The RESTRAINT guard. A shape the gate will actually relieve is not a refusal.

    6 footage beats of 5.00s plus 3 VO beats of 12.00s against a 40.00s order, 20.00s
    over — `decide_relief_shape` returns SPAWN for exactly this (a corrective re-render
    IS coming), so nothing is reported. Without the `verdict == "SPAWN"` check this fires
    on every convergeable overrun, which is the "signal that fires on everything".
    """
    manifest_beats = [
        {"id": f"clip-{i}", "seconds": 5.0, "min_seconds": 5.0, "max_seconds": 5.0}
        for i in range(1, 7)
    ] + [
        {"id": f"vo-{i}", "seconds": 12.0, "min_seconds": 1.0, "max_seconds": 30.0}
        for i in range(1, 4)
    ]
    out = await _stage().handle(
        EncodeMasterInput(),
        _ctx(
            duration=60.0,
            deliverable={"target_duration_seconds": 40.0},
            compile_mastering=_streams(60.0, 60.0),
            reconcile={
                "timeline_manifest": {
                    "beats": manifest_beats,
                    "run": {
                        "convergence_outcome": "cant_converge",
                        "target_seconds": 40.0,
                        "overrun_seconds": 20.0,
                    },
                }
            },
        ),
    )
    assert STRUCTURAL_RELIEF_REFUSED not in [d["code"] for d in out.degradations]
