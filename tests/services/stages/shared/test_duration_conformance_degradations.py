"""``duration_conformance_degradations`` must make "nothing was ordered" QUERYABLE.

Before this module, ``duration_conformance(None, ...)`` short-circuited to
``None`` and ``duration_conformance_degradations(None)`` returned ``[]`` —  a
live ``showcase_v1`` run recorded ``duration_conformance: null`` on
``demo_video_encode_master`` and it never showed up on the ``degradations``
channel either. A null verdict is neither pass nor fail, so it was invisible
to any ``within_tolerance = false`` query: a gate that cannot fail.

These tests pin the fix: an absent/unusable order now emits a distinct
``DURATION_NOT_ORDERED`` degradation, while ``duration_conformance()`` itself
keeps returning ``None`` (there genuinely is no verdict to give), and the
existing ordered-run behaviour (conforming -> ``[]``, breaching ->
``DURATION_OUT_OF_TOLERANCE``) is unchanged for a PROBED verdict.

They also pin the second gate-that-cannot-fail, which lives in the COMPANION
function ``duration_conformance_advisories``: a verdict reached on the DECLARED
timeline instead of an ffprobe emits ``DURATION_UNMEASURED``, IN ADDITION to any
tolerance entry. "The plan matched itself" must never be byte-identical to "the
encoded file was measured and conforms".

⛔ ``DURATION_UNMEASURED`` is an ADVISORY, not a degradation, and these tests pin
that tier. It was briefly a degradation and that was wrong: "nobody probed the
file" describes OUR INSTRUMENTATION, not the artifact, and it fires on every
stage that cannot reach a probe — ``demo_video`` and ``showcase`` route through
``demo_video/encode_master.py``, which structurally never can. On
``degradations`` it would flip ``degraded=True`` on 100% of two content types
forever, conforming masters included: the near-universal-flag failure the
advisory tier exists to prevent. ``duration_conformance_degradations`` must
therefore NEVER emit it, and the degradation channel keeps its old exact-list
guarantees for unprobed verdicts too.
"""

from __future__ import annotations

from src.services.stages.shared.degradation import (
    ADVISORY_CODES,
    DEGRADATION_CODES,
    DURATION_NOT_ORDERED,
    DURATION_OUT_OF_TOLERANCE,
    DURATION_UNMEASURED,
)
from src.services.stages.shared.duration_conformance import (
    duration_conformance,
    duration_conformance_advisories,
    duration_conformance_degradations,
)


class TestAbsentOrderIsQueryable:
    def test_no_usable_order_emits_duration_not_ordered(self) -> None:
        result = duration_conformance(None, 42.0)
        assert result is None  # the verdict itself stays honest: no order, no verdict

        degradations = duration_conformance_degradations(result)
        assert [d["code"] for d in degradations] == [DURATION_NOT_ORDERED]
        # `code` is the queryable half and must stay a bare constant; the
        # run-specific detail belongs in `detail`.
        assert degradations[0]["detail"]
        assert isinstance(degradations[0]["detail"], str)

    def test_zero_ordered_also_emits_duration_not_ordered(self) -> None:
        # duration_conformance() treats a zero/negative order the same as an
        # absent one (it is the only shape that could divide by zero) — the
        # degradation channel must not silently swallow that case either.
        result = duration_conformance(0.0, 42.0)
        assert result is None
        assert [d["code"] for d in duration_conformance_degradations(result)] == [
            DURATION_NOT_ORDERED
        ]

    def test_non_numeric_order_also_emits_duration_not_ordered(self) -> None:
        result = duration_conformance("not-a-number", 42.0)  # ty: ignore[invalid-argument-type]  # why: deliberately off-contract input — config_snapshot is untyped JSON, so a string can reach this at runtime
        assert result is None
        assert [d["code"] for d in duration_conformance_degradations(result)] == [
            DURATION_NOT_ORDERED
        ]


class TestOrderedRunsUnchanged:
    """The pre-existing ordered-run behaviour must not move.

    That holds for DECLARED verdicts as well as MEASURED ones: the unmeasured
    fact rides the separate ``advisories`` channel, so nothing an unmeasured run
    reports on ``degradations`` differs from what a measured one reports.
    """

    def test_conforming_playable_delivery_emits_nothing(self) -> None:
        result = duration_conformance(120.0, 999.0, playable_seconds=120.0)
        assert result is not None
        assert result["within_tolerance"] is True

        degradations = duration_conformance_degradations(result)
        assert degradations == []
        codes = [d["code"] for d in degradations]
        assert DURATION_NOT_ORDERED not in codes
        assert DURATION_OUT_OF_TOLERANCE not in codes
        assert DURATION_UNMEASURED not in codes

    def test_conforming_declared_delivery_emits_no_degradation(self) -> None:
        # Unprobed and in band: the LENGTH is not reported wrong, so NOTHING
        # lands on the degrading channel. The verdict is still unverified and
        # says so — on `advisories`, which never flips `degraded`.
        result = duration_conformance(120.0, 120.0)
        assert result is not None
        assert result["within_tolerance"] is True

        assert duration_conformance_degradations(result) == []
        codes = [a["code"] for a in duration_conformance_advisories(result)]
        assert codes == [DURATION_UNMEASURED]

    def test_breaching_playable_delivery_emits_only_out_of_tolerance(self) -> None:
        # The 2026-09-10 measured website_review breach: 120s ordered, 327.6s delivered.
        result = duration_conformance(120.0, 120.0, playable_seconds=327.6)
        assert result is not None
        assert result["within_tolerance"] is False

        degradations = duration_conformance_degradations(result)
        assert [d["code"] for d in degradations] == [DURATION_OUT_OF_TOLERANCE]
        assert "327.6" in degradations[0]["detail"]
        assert "120.0" in degradations[0]["detail"]
        # And a breach must never be mistaken for "nothing was ordered".
        assert DURATION_NOT_ORDERED not in [d["code"] for d in degradations]


class TestUnmeasuredIsItsOwnFact:
    """An UNPROBED verdict must be queryable as unverified, not as a pass.

    The defect: ``delivered_seconds`` was fed ``result["duration_seconds"]``,
    which ``primitives/compile_stage.py:192-195`` says is the compiler's NOMINAL
    sum of shot durations — the intended timeline, not the encoded file. On 38
    live runs the recorded value was byte-identical to it and the ffprobe
    (``actual_duration_seconds``) was never persisted. So the plan was checking
    itself and structurally could not catch a truncated master.
    """

    def test_duration_unmeasured_is_an_advisory_code_not_a_degradation(self) -> None:
        # The tier IS the contract. advisory() raises on an unregistered code,
        # so a missing entry here blows up at the write site; and membership in
        # DEGRADATION_CODES would make degradation() accept it again, which is
        # how it came to flip `degraded` on every unprobed master.
        assert DURATION_UNMEASURED in ADVISORY_CODES
        assert DURATION_UNMEASURED not in DEGRADATION_CODES

    def test_unmeasured_coexists_with_out_of_tolerance(self) -> None:
        # Both facts at once: the length is wrong AND nobody measured it. The
        # contract is ADDITION, not substitution — dropping either loses a
        # distinct fact about a paid artifact. They ride DIFFERENT channels: the
        # breach degrades the run, the doubt does not.
        result = duration_conformance(120.0, 327.6)
        assert result is not None
        assert result["within_tolerance"] is False

        assert [d["code"] for d in duration_conformance_degradations(result)] == [
            DURATION_OUT_OF_TOLERANCE
        ]
        assert [a["code"] for a in duration_conformance_advisories(result)] == [DURATION_UNMEASURED]

    def test_zero_playable_falls_back_and_raises_unmeasured(self) -> None:
        # 0.0 is the compiler's "unknown" sentinel on ANY probe failure/timeout
        # (video_compiler_service renderer.py:668-673), NOT a zero-length video.
        # It must fall back to the declared value and be reported unverified;
        # treating it as a measurement would report a 0%-of-order delivery
        # manufactured out of a missing measurement.
        result = duration_conformance(120.0, 120.0, playable_seconds=0.0)
        assert result is not None
        assert result["delivered_seconds"] == 120.0
        assert result["within_tolerance"] is True  # NOT a 0% catastrophe

        assert duration_conformance_degradations(result) == []
        codes = [a["code"] for a in duration_conformance_advisories(result)]
        assert codes == [DURATION_UNMEASURED]

    def test_unmeasured_detail_names_the_declared_basis(self) -> None:
        result = duration_conformance(120.0, 120.0)
        assert result is not None
        entry = next(
            a for a in duration_conformance_advisories(result) if a["code"] == DURATION_UNMEASURED
        )
        # The detail must say WHY the verdict is untrustworthy; a generic
        # "quality problem" is a worse diagnostic than no entry at all.
        assert "DECLARED" in entry["detail"]
        assert "never probed" in entry["detail"]

    def test_absent_order_does_not_also_claim_unmeasured(self) -> None:
        # `None` result means there is no verdict at all, so there is no
        # unverified verdict to report either — DURATION_NOT_ORDERED alone, and
        # NO advisory (a `None` must not crash or invent an unmeasured entry).
        codes = [d["code"] for d in duration_conformance_degradations(None)]
        assert codes == [DURATION_NOT_ORDERED]
        assert duration_conformance_advisories(None) == []
