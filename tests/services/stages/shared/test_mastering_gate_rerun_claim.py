"""A mastering-gate degradation may not contradict its own evidence.

THE DEFECT. ``mastering_gate_degradations`` built its detail from a fixed sentence —
"failed STRUCTURALLY and no stage was re-run" — while the same branch is reached with
``rerun_stages`` NON-EMPTY (the compiler asked for re-runs and at least one of them is
structural). The entry therefore printed the words "no stage was re-run" immediately
followed by ``rerun_stages=av_coverage``: a degradation whose prose denies the fact it
is quoting.

The degradations channel is the record of what was silently changed about a PAID
deliverable. A sentence there that the surrounding code never established is worse
than silence, because it is read as evidence.

What is pinned here is the CLAIM, not a substring: whenever the reported re-run list is
non-empty the detail must not assert that nothing was re-run, and whenever it IS empty
the detail must say that no stage was named — the two cases must be distinguishable.
"""

from __future__ import annotations

import re

import pytest

from src.services.stages.shared.mastering_gate import (
    mastering_gate_degradations,
    mastering_gate_notes,
)

# Any sentence of this shape is the banned claim. Written as a pattern rather than one
# literal so a reworded "no stages were re-run" cannot slip the guard.
_DENIES_ANY_RERUN = re.compile(r"\bno stages?\b[^—(]*\bre-?run\b", re.IGNORECASE)


def _structural_report(stages: list[str]) -> dict[str, object]:
    """A report that reaches the structural branch: blank picture, gate failed."""
    return {
        "gate_pass": False,
        "rerun_stages": stages,
        # `dead_frame_ratio` over the compiler's 0.10 cutoff ⇒ `_blank_picture` ⇒ a
        # structural reason exists even when `rerun_stages` names nothing, which is how
        # the empty case reaches this branch.
        #
        # ⚠️ This fixture used to trigger the branch with `luma_spread: 0.0` alone. That
        # disjunct was RETIRED: luma_spread is the MIN over every frame of YMAX-YMIN, so
        # one fade-to-black zeroes it on a perfectly good film (live run 01a08f66:
        # dead_frame_ratio=0.009825, no dead_frame stage, luma_spread=0.0 — stamped
        # BLANK with both numbers contradicting the verdict). Nothing about what this
        # file PINS depends on which metric opens the branch — the claim under test is
        # the PROSE — so the trigger is moved to the live signal and luma_spread is kept
        # only to prove it is still quoted as reported-but-not-used.
        "luma_spread": 0.0,
        "dead_frame_ratio": 0.25,
    }


@pytest.mark.parametrize(
    "stages",
    [["av_coverage"]],
)
def test_a_nonempty_rerun_list_is_never_described_as_no_stage_re_run(
    stages: list[str],
) -> None:
    """THE CONTRADICTION. Stages were reported ⇒ the prose may not deny them."""
    entries = mastering_gate_degradations(_structural_report(stages))
    assert len(entries) == 1, f"expected one structural entry for {stages!r}"
    detail = entries[0]["detail"]
    # The evidence really is present in this entry — so the prose is being read next
    # to it, which is what makes a denial a lie rather than an omission.
    assert f"rerun_stages={', '.join(stages)}" in detail
    assert not _DENIES_ANY_RERUN.search(detail), (
        f"detail denies a re-run while reporting {stages!r}: {detail!r}"
    )
    # ...and it must actually name them in prose, not only inside the parenthetical.
    for stage in stages:
        assert stage in detail.split("(rerun_stages=")[0], (
            f"{stage!r} is reported but not named in the sentence: {detail!r}"
        )


def test_an_empty_rerun_list_says_so_and_stays_distinguishable() -> None:
    """The other half: nothing was named, and the entry says exactly that.

    The two cases must not collapse into one sentence — a reader has to be able to tell
    "the compiler asked for nothing" from "the compiler asked for these" without
    parsing the parenthetical.
    """
    empty = mastering_gate_degradations(_structural_report([]))
    assert len(empty) == 1
    detail = empty[0]["detail"]
    assert _DENIES_ANY_RERUN.search(detail), (
        f"an empty rerun list must be stated, not glossed: {detail!r}"
    )
    assert "rerun_stages=none reported" in detail

    named = mastering_gate_degradations(_structural_report(["av_coverage"]))[0]["detail"]
    # Only the CLAIM, i.e. the part before the reasons list — the reasons differ
    # between these two reports for an unrelated reason (one carries an av_coverage
    # reason), so comparing whole details would pass on a fixed sentence.
    claim_empty = detail.split("—")[0]
    claim_named = named.split("—")[0]
    assert claim_empty != claim_named, "the two cases share one fixed claim sentence"


# ═════════════════════════════════════════════════════════════════════════════════════
# THE UNFIXED TWIN — the same defect, in the channel `is_degraded` is derived from
#
# The fix above landed on `mastering_gate_degradations` only. `mastering_gate_notes`
# kept the identical sentence, and it is the one that MATTERS: `runs/_shared.py` derives
# `degraded` from `notes`, not from the degradations list.
#
# Its final branch made TWO claims the code never established:
#
#   (a) "no stage was re-run" — printed with `rerun_stages` NON-EMPTY, the same
#       self-denying sentence fixed in the other channel; and
#   (b) "gate FAILED" — while the only guard on the branch is `_blank_picture`, which
#       reads `rerun_stages` and `dead_frame_ratio` and NEVER reads `gate_pass`. A
#       report of {"gate_pass": True, "rerun_stages": [], "dead_frame_ratio": 0.25}
#       therefore emitted the words "gate FAILED" about a gate the compiler reported as
#       PASSED. (When this was written the guard's second input was `luma_spread`; that
#       disjunct has since been retired — see `_blank_by_ratio`. The defect and the
#       property pinned here are unchanged: the branch still does not read `gate_pass`.)
#
# What is pinned below is the CLAIM in both directions: the note may not deny re-runs it
# is quoting, and it may not assert a failed gate the report does not report.
# ═════════════════════════════════════════════════════════════════════════════════════

#: Any sentence asserting the gate itself failed. Pattern, not a literal, so a reworded
#: "the gate failed" cannot slip past.
_ASSERTS_GATE_FAILED = re.compile(r"\bgate\s+(?:has\s+)?FAIL(?:ED|S)?\b", re.IGNORECASE)


def _blank_by_ratio(gate_pass: object, stages: list[str]) -> dict[str, object]:
    """A report reaching the FINAL note branch: blank by ratio, no `dead_frame` stage.

    `dead_frame` is deliberately absent — that stage returns from an earlier branch with
    its own (accurate) wording, so it would not exercise the sentence under test.

    ⚠️ Renamed from `_blank_by_luma`: `luma_spread <= 0.0` no longer opens this branch
    (see `_structural_report`). It is still carried, because the note must keep quoting
    it as reported-but-not-used, but `dead_frame_ratio` is what reaches the sentence.
    """
    report: dict[str, object] = {
        "rerun_stages": stages,
        "luma_spread": 0.0,
        "dead_frame_ratio": 0.25,
    }
    if gate_pass is not None:
        report["gate_pass"] = gate_pass
    return report


@pytest.mark.parametrize("stages", [["deband"]])
def test_a_note_never_denies_the_rerun_stages_it_is_quoting(stages: list[str]) -> None:
    """(a) Stages were reported ⇒ the note may not say nothing was re-run."""
    notes = mastering_gate_notes(_blank_by_ratio(False, stages))
    assert len(notes) == 1
    note = notes[0]
    assert f"rerun_stages={', '.join(stages)}" in note
    assert not _DENIES_ANY_RERUN.search(note), f"note denies re-runs it quotes: {note!r}"
    for stage in stages:
        assert stage in note.split("(rerun_stages=")[0], (
            f"{stage!r} is reported but not named in the sentence: {note!r}"
        )


def test_an_empty_rerun_list_is_stated_and_stays_distinguishable() -> None:
    """The other half of (a): nothing named ⇒ say so, and not in the named case's words."""
    empty = mastering_gate_notes(_blank_by_ratio(False, []))[0]
    assert _DENIES_ANY_RERUN.search(empty), f"an empty list must be stated: {empty!r}"
    named = mastering_gate_notes(_blank_by_ratio(False, ["deband"]))[0]
    assert empty.split("(rerun_stages=")[0] != named.split("(rerun_stages=")[0]


@pytest.mark.parametrize("gate_pass", [True])
def test_a_note_never_asserts_a_failed_gate_the_report_did_not_report(
    gate_pass: object,
) -> None:
    """(b) THE UNREAD FIELD. The branch's guard is `_blank_picture`, which never looks at
    `gate_pass`. So with a PASSING verdict (or none at all) the note must report the
    blank picture it DID observe and quote the verdict honestly — never claim a failure.
    """
    report = _blank_by_ratio(gate_pass, [])
    notes = mastering_gate_notes(report)
    assert len(notes) == 1, "a blank picture is still degraded — the note must exist"
    note = notes[0]
    assert not _ASSERTS_GATE_FAILED.search(note), (
        f"note asserts a failed gate while the report says gate_pass={gate_pass!r}: {note!r}"
    )
    # ...and it states what WAS observed, by the metric that observed it — which is
    # `dead_frame_ratio` against its cutoff, NOT `luma_spread`. Citing luma_spread as
    # the proof would be this file's own defect in a new place: a note naming as
    # evidence a metric the branch never consulted. luma_spread must still APPEAR (it
    # is on the report and a reader needs to see it), but marked as not having decided.
    assert "dead_frame_ratio=0.25" in note
    assert "luma_spread=0.0 reported, not used" in note
    # The verdict is quoted rather than invented: present ⇒ its value, absent ⇒ said so.
    assert ("gate_pass=True" in note) if gate_pass is True else ("no gate_pass" in note)


def test_the_reported_false_verdict_is_still_quoted() -> None:
    """Honesty runs both ways: a genuinely failed gate is reported as such."""
    note = mastering_gate_notes(_blank_by_ratio(False, []))[0]
    assert "gate_pass=False" in note
