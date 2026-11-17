"""The structural tier is READ off the compiler's `violations[]`, not string-matched.

Before this, `_structural_reasons` could only ever match `av_coverage` — the sole
member of the compiler's `_ALWAYS_ENFORCED_STAGES` — and the compiler RAISES
`MasteringGateError` on an always-enforced stage in BOTH advisory and enforce mode.
So a DELIVERED advisory report can never carry `av_coverage` in `rerun_stages`, and
`MASTERING_GATE_FAILED` reduced to "absent report OR dead frame".

The compiler now publishes a per-violation list on the advisory report::

    {"stage": str, "metric": str, "value": float|None,
     "limit": float|None, "tier": "structural"|"tunable"}

These tests pin BEHAVIOUR on both shapes: the tier is honoured when present, and a
report that predates the key degrades to exactly today's output — never to a crash
and never to a false clean.
"""

from __future__ import annotations

import pytest

from src.services.stages.shared.degradation import MASTERING_GATE_FAILED
from src.services.stages.shared.mastering_gate import (
    mastering_gate_degradations,
    mastering_gate_notes,
)


def _codes(report: object) -> list[str]:
    return [d["code"] for d in mastering_gate_degradations(report)]


def _details(report: object) -> list[str]:
    return [d["detail"] for d in mastering_gate_degradations(report)]


# ── THE PAYLOAD TIER IS HONOURED ─────────────────────────────────────────────


def test_a_structural_violation_the_name_mirror_never_knew_is_recorded() -> None:
    """The whole point: a structural stage that is NOT `av_coverage` now records.

    `stream_integrity` is deliberately absent from `_STRUCTURAL_STAGES` and from
    `enforced_stages`, so the pre-`violations` path returns NOTHING for this report
    (asserted in the fallback test below). Reading `tier` is what arms it.
    """
    report = {
        "gate_pass": False,
        "rerun_stages": ["stream_integrity"],
        "violations": [
            {
                "stage": "stream_integrity",
                "metric": "audio_stream_count",
                "value": 0.0,
                "limit": 1.0,
                "tier": "structural",
                "detail": "stream_integrity: the delivered mux carries no audio stream",
            }
        ],
    }
    assert _codes(report) == [MASTERING_GATE_FAILED]
    detail = _details(report)[0]
    assert "stream_integrity" in detail
    assert "carries no audio stream" in detail


def test_tunable_violations_alone_still_record_nothing() -> None:
    """The anti-noise guard must survive the new source.

    207 of 214 production masters carry `gate_pass=false`, almost all of it a
    miscalibrated deband ceiling. If `tier: "tunable"` recorded, this channel would
    light on ~96.7% of renders and mean nothing.
    """
    report = {
        "gate_pass": False,
        "rerun_stages": ["deband", "loudness"],
        "violations": [
            {
                "stage": "deband",
                "metric": "banding_score",
                "value": 1.0,
                "limit": 0.15,
                "tier": "tunable",
                "detail": "deband: banding 1.0 over 0.15",
            },
            {
                "stage": "loudness",
                "metric": "integrated_lufs",
                "value": -12.0,
                "limit": -16.0,
                "tier": "tunable",
                "detail": "loudness: -12.0 vs -16.0",
            },
        ],
    }
    assert mastering_gate_degradations(report) == []
    assert mastering_gate_notes(report) == []


def test_a_structural_entry_records_even_among_tunable_ones() -> None:
    """Mixed report ⇒ ONE entry, describing the structural half only."""
    report = {
        "gate_pass": False,
        "rerun_stages": ["deband", "av_coverage"],
        "violations": [
            {
                "stage": "deband",
                "metric": "banding_score",
                "value": 1.0,
                "limit": 0.15,
                "tier": "tunable",
                "detail": "deband: banding 1.0 over 0.15",
            },
            {
                "stage": "av_coverage",
                "metric": "av_gap_seconds",
                "value": 39.6,
                "limit": 0.5,
                "tier": "structural",
                "detail": "av_coverage: picture ends 39.6s before the narration",
            },
        ],
    }
    details = _details(report)
    assert len(details) == 1, details
    assert "39.6s before the narration" in details[0]
    # The tunable half must not be quoted as a reason — only as the rerun list.
    assert "banding 1.0" not in details[0]


def test_dead_frame_is_not_double_reported_from_the_violation_list() -> None:
    """`_blank_picture` already owns this defect, with its own wording+evidence.

    A `dead_frame` entry tiered structural must NOT also produce a per-stage reason;
    that would give one master two reasons for one defect, the second of them worded
    for a different failure.
    """
    report = {
        "gate_pass": False,
        "rerun_stages": ["dead_frame"],
        "dead_frame_ratio": 0.44,
        "violations": [
            {
                "stage": "dead_frame",
                "metric": "dead_frame_ratio",
                "value": 0.44,
                "limit": 0.10,
                "tier": "structural",
                "detail": "dead_frame: 0.44 over 0.10",
            },
        ],
    }
    details = _details(report)
    assert len(details) == 1, details
    assert details[0].count("blank_picture") == 1
    # The compiler's own dead_frame sentence must not appear a second time.
    assert "0.44 over 0.10" not in details[0]


def test_a_violation_without_detail_still_names_its_metric() -> None:
    """No `detail` ⇒ an honest generic line, never a borrowed sentence.

    Inheriting `av_coverage`'s wording ("the picture does not cover the narration")
    for an unknown stage would make this channel assert a failure it never observed.
    """
    report = {
        "gate_pass": False,
        "violations": [
            {
                "stage": "mux_integrity",
                "metric": "moov_ok",
                "value": 0.0,
                "limit": 1.0,
                "tier": "structural",
            },
        ],
    }
    detail = _details(report)[0]
    assert "mux_integrity" in detail
    assert "moov_ok=0.0" in detail
    assert "does not cover the delivered narration" not in detail


# ── THE FALLBACK: A REPORT THAT PREDATES `violations` ────────────────────────


def test_absent_violations_key_reproduces_the_old_structural_behaviour() -> None:
    """A pod mid-roll emits the OLD report shape. It must grade exactly as before."""
    old_shape = {
        "gate_pass": False,
        "rerun_stages": ["av_coverage"],
        "enforced_stages": ["av_coverage"],
        "av_gap_seconds": 39.6,
    }
    details = _details(old_shape)
    assert len(details) == 1, details
    assert "av_coverage" in details[0]
    assert "av_gap_seconds=39.6" in details[0]

    # And the same report with a stage the NAME mirror does not know stays silent —
    # this is the limitation the tier source exists to lift, pinned so the two
    # paths are provably different rather than assumed so.
    unknown = {"gate_pass": False, "rerun_stages": ["stream_integrity"]}
    assert mastering_gate_degradations(unknown) == []


def test_an_empty_violations_list_falls_back_rather_than_reading_clean() -> None:
    """`gate_pass=False` with no violation named is a serialization accident.

    Trusting `[]` as "nothing structural was violated" would be a FALSE CLEAN, so
    absence of usable entries takes the mirror path.

    The structural signal here is deliberately one ONLY the mirror can see — an
    `av_coverage` name match, with no blank picture and no `dead_frame_ratio` — so
    the assertion distinguishes the two paths instead of passing on either.
    """
    report = {
        "gate_pass": False,
        "rerun_stages": ["av_coverage"],
        "enforced_stages": ["av_coverage"],
        "av_gap_seconds": 39.6,
        "violations": [],
    }
    details = _details(report)
    assert len(details) == 1, details
    assert "av_gap_seconds=39.6" in details[0]


def test_a_pre_tier_violation_list_falls_back_rather_than_reading_clean() -> None:
    """A list from the pre-`tier` violation model says nothing about the tier.

    Reading "no structural entry" off it is the same false clean, so it takes the
    mirror path too — which finds `av_coverage` by name.
    """
    report = {
        "gate_pass": False,
        "rerun_stages": ["av_coverage"],
        "enforced_stages": ["av_coverage"],
        "av_gap_seconds": 12.0,
        # No `tier` on any entry.
        "violations": [
            {
                "stage": "av_coverage",
                "metric": "av_gap_seconds",
                "value": 12.0,
                "limit": 0.5,
                "detail": "av_coverage: gap 12.0s",
            },
        ],
    }
    details = _details(report)
    assert len(details) == 1, details
    assert "av_gap_seconds=12.0" in details[0]


@pytest.mark.parametrize(
    "junk",
    ["violations"],
    ids=["string"],
)
def test_a_junk_violations_payload_grades_as_if_the_key_were_absent(junk: object) -> None:
    """A malformed payload must fall back, and the fallback must still FIND the defect.

    This test replaced an assertion-free "never raises" loop that the repo's
    ``test_vacuous_test_guard`` correctly rejected: it passed no matter what these
    functions returned, including returning nothing at all on a structurally broken
    master. "Does not crash" is not the property that matters here — "does not go
    quiet" is.

    So the assertion is an EQUALITY against the same report with the ``violations``
    key removed entirely, i.e. the pre-key behaviour, on a report whose structural
    defect is visible ONLY to the name path (`av_coverage` in ``rerun_stages`` +
    ``enforced_stages``, no blank picture, no ``dead_frame_ratio``). A junk payload
    that silently disarmed the tier would return ``[]`` here and fail both
    assertions; the old loop would have passed.
    """
    base: dict[str, object] = {
        "gate_pass": False,
        "rerun_stages": ["av_coverage"],
        "enforced_stages": ["av_coverage"],
        "av_gap_seconds": 39.6,
    }
    report = {**base, "violations": junk}

    # Not silence: the defect is still recorded, with its own metric.
    details = _details(report)
    assert len(details) == 1, details
    assert "av_gap_seconds=39.6" in details[0]
    # And byte-identical to the pre-`violations` shape, on BOTH channels.
    assert mastering_gate_degradations(report) == mastering_gate_degradations(base)
    assert mastering_gate_notes(report) == mastering_gate_notes(base)


def test_a_tiered_entry_naming_no_stage_records_without_borrowing_a_sentence() -> None:
    """``[{"tier": "structural"}]`` IS a usable list — it must record, honestly.

    Unlike the junk shapes above this one carries a tier, so it is authoritative and
    takes the tier path. Two things must hold: it records (a structural entry is not
    dropped for want of a stage name), and it does NOT inherit `av_coverage`'s
    sentence for a stage it never identified.
    """
    report = {
        "gate_pass": False,
        "rerun_stages": ["deband"],
        "violations": [{"tier": "structural"}],
    }

    details = _details(report)
    assert len(details) == 1, details
    assert "unknown" in details[0]
    assert "does not cover the delivered narration" not in details[0]


def test_swapping_the_tier_string_changes_the_outcome() -> None:
    """The discriminating test for this whole file: the tier must be READ.

    A tier test that passes whether or not the tier is consulted is exactly the
    failure mode the guard exists for. ONE report, ONE field different — the tier
    string — and the two must produce genuinely different observable results.

    ``GateTier`` is strictly two-valued (``structural``/``tunable``); armed-vs-
    advisory is a separate axis compiler-side and is deliberately NOT encoded here.
    """

    def _report(tier: str) -> dict[str, object]:
        return {
            "gate_pass": False,
            "rerun_stages": ["stream_integrity"],
            "violations": [
                {
                    "stage": "stream_integrity",
                    "metric": "audio_stream_count",
                    "value": 0.0,
                    "limit": 1.0,
                    "tier": tier,
                    "detail": "stream_integrity: the delivered mux carries no audio stream",
                }
            ],
        }

    structural = mastering_gate_degradations(_report("structural"))
    tunable = mastering_gate_degradations(_report("tunable"))

    assert [d["code"] for d in structural] == [MASTERING_GATE_FAILED]
    assert "carries no audio stream" in structural[0]["detail"]
    # The stage is named ONCE. The compiler's own `detail` already opens with
    # `f"{stage}: "`, so prepending it unconditionally read "stream_integrity:
    # stream_integrity: the delivered mux…" in the entry a reviewer sees.
    assert structural[0]["detail"].count("stream_integrity:") == 1
    # The tunable half of the same defect records NOTHING — the anti-noise guard.
    assert tunable == []
    assert structural != tunable


# ── THE DEAD-FRAME CUTOFF IS THE COMPILER'S, WHEN IT STATES ONE ──────────────


def test_the_quoted_cutoff_comes_from_the_compilers_own_limit() -> None:
    """`_DEAD_FRAME_MIN_RATIO` is a MIRROR of `render_dead_frame_min_ratio`.

    When the compiler states its live value (the `limit` of a `dead_frame`
    violation) the verdict and the quoted evidence must use THAT number, so a
    retuned compiler does not leave this module grading on a stale copy.
    """
    retuned = {
        "gate_pass": False,
        "rerun_stages": ["dead_frame"],
        "dead_frame_ratio": 0.44,
        "violations": [
            {
                "stage": "dead_frame",
                "metric": "dead_frame_ratio",
                "value": 0.44,
                "limit": 0.35,
                "tier": "structural",
            },
        ],
    }
    note = mastering_gate_notes(retuned)[0]
    assert "cutoff 0.35" in note
    assert "cutoff 0.1" not in note


def test_a_ratio_under_the_compilers_retuned_cutoff_is_not_blank() -> None:
    """The cutoff must DECIDE, not merely decorate the sentence.

    0.22 is over this module's mirrored 0.10 and under the compiler's stated 0.35.
    The compiler did not call it dead (no `dead_frame` in `rerun_stages`), so
    neither may we.
    """
    report = {
        "gate_pass": False,
        "rerun_stages": ["deband"],
        "dead_frame_ratio": 0.22,
        "violations": [
            {
                "stage": "deband",
                "metric": "banding_score",
                "value": 1.0,
                "limit": 0.15,
                "tier": "tunable",
                "detail": "deband: banding 1.0",
            },
            {
                "stage": "dead_frame",
                "metric": "dead_frame_ratio",
                "value": 0.22,
                "limit": 0.35,
                "tier": "tunable",
                "detail": "dead_frame: under limit",
            },
        ],
    }
    assert mastering_gate_notes(report) == []
    assert mastering_gate_degradations(report) == []


def test_without_a_stated_limit_the_mirrored_cutoff_still_decides() -> None:
    """No `violations` ⇒ the 0.10 mirror, i.e. today's behaviour, unchanged."""
    note = mastering_gate_notes(
        {"gate_pass": False, "rerun_stages": ["deband"], "dead_frame_ratio": 0.25}
    )[0]
    assert "cutoff 0.1" in note
