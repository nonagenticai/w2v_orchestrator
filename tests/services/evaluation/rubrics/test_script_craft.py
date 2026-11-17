"""Unit tests for the deterministic ``script_craft`` rubric (ORCH-1.2).

Pure — no DB, no network, no LLM. Exercises the frozen ``score_script`` public
contract plus each craft-metric helper.
"""

from __future__ import annotations

from typing import ClassVar

from src.services.evaluation.models import MetricResult
from src.services.evaluation.rubrics.script_craft import (
    abstraction_ratio,
    composite,
    cta_craft,
    hook_strength,
    score_script,
    specificity_density,
    turn_presence,
    word_budget,
)

# A CTA verbatim item pinned across every scoring call.
_CTA = "Start your free trial today."
_VERBATIM = [_CTA]

# A deliberately weak script: generic openers, marketing cliches, no hook, no
# turn, a soft non-imperative close. ``w3`` is locked to prove locks are honored.
_WEAK_SCRIPT: list[dict] = [
    {"id": "w1", "text": "Our platform is innovative and seamless.", "seconds": 4, "locked": False},
    {"id": "w2", "text": "Our platform is innovative and robust.", "seconds": 4, "locked": False},
    {"id": "w3", "text": "It empowers you to leverage synergy.", "seconds": 4, "locked": True},
    {
        "id": "w4",
        "text": "You should probably think about learning more.",
        "seconds": 4,
        "locked": False,
    },
]

# A strong script: a question+number hook, concrete nouns and digits, exactly one
# turn ("Then"), and a single pinned imperative CTA.
_STRONG_SCRIPT: list[dict] = [
    {
        "id": "s1",
        "text": "How do 3 clicks replace 284 spreadsheets?",
        "seconds": 4,
        "locked": False,
    },
    {
        "id": "s2",
        "text": "Watch the dashboard sort 1,200 invoices instantly.",
        "seconds": 4,
        "locked": False,
    },
    {
        "id": "s3",
        "text": "Then Maria approves the report from her phone.",
        "seconds": 4,
        "locked": False,
    },
    {"id": "s4", "text": _CTA, "seconds": 4, "locked": False},
]


def _score(script: list[dict]) -> dict:
    return score_script(script, verbatim_items=_VERBATIM, target_duration_seconds=16.0)


class TestWeakVsStrong:
    def test_weak_hook_specificity_abstraction_low(self) -> None:
        report = _score(_WEAK_SCRIPT)
        first = report["per_line"]["w1"]
        assert first["hook_strength"].score < 70
        assert first["specificity_density"].score < 70
        assert first["abstraction_ratio"].score < 70

    def test_strong_hook_specificity_abstraction_higher(self) -> None:
        report = _score(_STRONG_SCRIPT)
        first = report["per_line"]["s1"]
        assert first["hook_strength"].score >= 70
        assert first["specificity_density"].score >= 70
        assert first["abstraction_ratio"].score >= 70

    def test_strong_composite_beats_weak(self) -> None:
        weak = _score(_WEAK_SCRIPT)
        strong = _score(_STRONG_SCRIPT)
        assert strong["composite"] > weak["composite"]


class TestWordBudget:
    def test_on_budget_scores_full(self) -> None:
        # 4s * 150/60 == 10 word budget; exactly 10 words → 0% deviation.
        text = "one two three four five six seven eight nine ten"
        result = word_budget(text, 4.0)
        assert result.score == 100.0

    def test_over_budget_flags(self) -> None:
        # 20 words against a 10-word budget → 100% deviation → flag.
        text = " ".join(["word"] * 20)
        result = word_budget(text, 4.0)
        assert result.score < 70
        assert result.threshold != "pass"

    def test_a_severe_overfill_is_flagged_independent_of_the_composite(self) -> None:
        """word_budget carries weight 0.10 of 12 metrics, so a bad word_budget score
        alone is diluted 10-to-1 in the weighted-mean composite — nowhere near enough
        to guarantee it sinks a ship gate on its own. It must still surface as a
        NAMED finding via ``flags`` — a per-line channel that does not go through
        the weighted average at all — so a severe over-fill is never invisible just
        because the composite absorbed it."""
        # 4s * 150/60 == 10-word budget; 24 words is a 140% deviation — severe overfill,
        # matching the measured 50-135% range this rubric exists to catch.
        text = " ".join(["word"] * 24)
        line = {"id": "b1", "text": text, "seconds": 4.0, "locked": False}
        report = _score([line])
        result = report["per_line"]["b1"]["word_budget"]
        assert result.score < 70
        assert result.threshold != "pass"
        # The finding is named, independent of whatever the composite lands on.
        assert "word_budget" in report["flags"]["b1"]

    def test_no_clip_seconds_is_neutral_but_not_a_pass(self) -> None:
        """Nothing was measured (no clip length to pace against), so the label must
        say so — ``"insufficient_evidence"``, never ``"pass"``, which would read
        identically to "paced correctly" to anything consulting the threshold string
        rather than the raw score. The neutral 75.0 score is unchanged and still
        sits above ``_FLAG_THRESHOLD`` so it does not flag the line either way."""
        result = word_budget("some words here", None)
        assert result.score == 75.0
        assert result.threshold == "insufficient_evidence"


class TestOpenerVariety:
    def test_duplicate_openers_flagged_on_later_line(self) -> None:
        report = _score(_WEAK_SCRIPT)
        flags = report["flags"]
        # w1 and w2 both open "our platform is" → the later beat (w2) is flagged.
        assert "opener_variety" in flags["w2"]
        assert "opener_variety" not in flags.get("w1", [])


class TestTurnPresence:
    def test_exactly_one_turn(self) -> None:
        lines = [
            {"id": "a", "text": "We start calm and steady."},
            {"id": "b", "text": "But everything changes at once."},
            {"id": "c", "text": "We finish strong."},
        ]
        result = turn_presence(lines)
        assert result.raw_value["turn_count"] == 1
        assert result.score == 100.0

    def test_no_turn_flags(self) -> None:
        lines = [{"id": "a", "text": "We win."}, {"id": "b", "text": "We win again."}]
        result = turn_presence(lines)
        assert result.raw_value["turn_count"] == 0
        assert result.score < 70


class TestCtaCraft:
    def test_pinned_imperative_passes(self) -> None:
        result = cta_craft(_CTA, _VERBATIM)
        assert result.score >= 70
        assert result.threshold == "pass"

    def test_soft_unpinned_close_flags(self) -> None:
        result = cta_craft("You should probably think about learning more.", _VERBATIM)
        assert result.score < 70


class TestLockedLines:
    def test_locked_line_never_flagged(self) -> None:
        report = _score(_WEAK_SCRIPT)
        # w3 is locked and would otherwise flag on specificity/abstraction.
        assert "w3" not in report["flags"]
        # ...but it is still scored in the per-line report.
        assert report["per_line"]["w3"]["abstraction_ratio"].score < 70


_PREAMBLE_SCRIPT: list[dict] = [
    {
        "id": "p1",
        "text": "Sure, here is your video script for the launch: it begins now.",
        "seconds": 4,
        "locked": False,
    },
    {
        "id": "p2",
        "text": "Watch the dashboard sort 1,200 invoices instantly.",
        "seconds": 4,
        "locked": False,
    },
    {
        "id": "p3",
        "text": "Then Maria approves the report from her phone.",
        "seconds": 4,
        "locked": False,
    },
    {"id": "p4", "text": _CTA, "seconds": 4, "locked": False},
]


class TestPreambleFlagging:
    """``preamble`` is a per_story metric (weight 0.0, no _COMPOSITE_WEIGHTS
    entry — deliberately so, per the module docstring's convention note) that
    used to be computed and then discarded: never in composite, never in
    flags. script_doctor's revise loop only reads ``flags``
    (`_flagged_lines`/`_critique_msg` in
    ``src/services/stages/shared/script_doctor.py``), so an undetected
    preamble shipped straight through. These pin the fix: a failing preamble
    score is attributed to the FIRST line's id in ``flags``.
    """

    def test_leading_preamble_flags_first_line(self) -> None:
        report = _score(_PREAMBLE_SCRIPT)
        assert report["per_story"]["preamble"].score < 70
        assert "preamble" in report["flags"]["p1"]

    def test_clean_script_has_no_preamble_flag_anywhere(self) -> None:
        report = _score(_STRONG_SCRIPT)
        assert report["per_story"]["preamble"].score >= 70
        for names in report["flags"].values():
            assert "preamble" not in names

    def test_locked_first_line_is_not_flagged(self) -> None:
        script = [dict(line) for line in _PREAMBLE_SCRIPT]
        script[0] = {**script[0], "locked": True}
        report = _score(script)
        assert report["per_story"]["preamble"].score < 70
        assert "p1" not in report["flags"]

    def test_duration_budget_is_never_flagged(self) -> None:
        """Pins the deliberate scope decision: duration_budget has the
        identical inert shape (weight 0.0, absent from _COMPOSITE_WEIGHTS)
        but is NOT surfaced in flags by this fix — a later edit widening
        the flagging must do so explicitly, not by accident.
        """
        # Wildly over budget vs a 16s target (see `_score`'s target_duration_seconds).
        over_budget_script = [
            {"id": "b1", "text": " ".join(["word"] * 400), "seconds": 4, "locked": False},
        ]
        report = _score(over_budget_script)
        assert report["per_story"]["duration_budget"].score < 70
        for names in report["flags"].values():
            assert "duration_budget" not in names


class TestReportShape:
    def test_score_script_returns_script_report(self) -> None:
        report = _score(_STRONG_SCRIPT)
        assert set(report.keys()) == {"composite", "per_line", "per_story", "flags"}
        assert isinstance(report["composite"], float)
        assert 0.0 <= report["composite"] <= 100.0

        assert set(report["per_line"].keys()) == {"s1", "s2", "s3", "s4"}
        for metrics in report["per_line"].values():
            assert all(isinstance(mr, MetricResult) for mr in metrics.values())

        assert "turn_presence" in report["per_story"]
        assert "readability" in report["per_story"]
        for mr in report["per_story"].values():
            assert isinstance(mr, MetricResult)

        for line_id, names in report["flags"].items():
            assert isinstance(line_id, str)
            assert isinstance(names, list)
            assert all(isinstance(name, str) for name in names)

    def test_composite_recomputes_from_report(self) -> None:
        report = _score(_STRONG_SCRIPT)
        assert composite(report) == report["composite"]

    def test_direct_metric_helpers_return_metricresult(self) -> None:
        assert isinstance(specificity_density("3 dashboards on screen"), MetricResult)
        assert isinstance(abstraction_ratio("seamless robust synergy"), MetricResult)
        assert isinstance(hook_strength("Why does this matter?"), MetricResult)


class TestDegenerateInputsDoNotCrash:
    """Edge inputs (empty script, one-word line, empty text, no seconds) must
    return the documented shape without a ZeroDivision / IndexError."""

    def test_empty_script(self) -> None:
        report = score_script([], verbatim_items=[], target_duration_seconds=60.0)
        assert set(report) == {"composite", "per_line", "per_story", "flags"}
        assert isinstance(report["composite"], float)
        assert report["per_line"] == {}

    def test_one_word_script(self) -> None:
        report = score_script(
            [{"id": "clip-1", "text": "Ship."}],
            verbatim_items=[],
            target_duration_seconds=60.0,
        )
        assert isinstance(report["composite"], float)
        assert "clip-1" in report["per_line"]

    def test_empty_text_and_no_seconds(self) -> None:
        report = score_script(
            [{"id": "clip-1", "text": "", "seconds": None}],
            verbatim_items=[],
            target_duration_seconds=0.0,
        )
        assert isinstance(report["composite"], float)

    def test_direct_metric_helpers_on_empty(self) -> None:
        assert isinstance(specificity_density(""), MetricResult)
        assert isinstance(abstraction_ratio(""), MetricResult)
        assert isinstance(word_budget("", 0.0), MetricResult)
        assert isinstance(turn_presence([]), MetricResult)
        assert isinstance(cta_craft("", []), MetricResult)


# ─────────────────────────────────────────────────────────────────────────────
# N6 — the rubric scores against the run's MEASURED speaking rate.
#
# Before this seam `word_budget` and `_duration_budget` both compared a word
# count to `seconds * 150/60` no matter how fast the voice actually talks. The
# live kokoro voice delivers ~120 wpm, so a beat that will OVERRUN its clip
# scored a clean pass and `script_doctor`'s rewrite loop — which is driven by
# exactly these flags — was told to ADD words to it.
#
# The rate comes in as `score_script(..., wpm=)`; the CALL SITE that supplies it
# is `stages/shared/script_doctor.py::_doctor` and is pinned by
# `tests/services/stages/shared/test_script_doctor.py
# ::test_the_rubric_is_scored_against_the_measured_speaking_rate`.
# ─────────────────────────────────────────────────────────────────────────────


class TestWordBudgetHonoursTheMeasuredRate:
    # 24 words in 12.0s == exactly 120 wpm: on target for the real voice, and
    # 20% under the 150 wpm target (30 words) the rubric used to assume.
    _TEXT = " ".join(f"word{i}" for i in range(24))
    _SECONDS = 12.0

    def test_a_slower_measured_rate_moves_the_target_and_the_score(self) -> None:
        at_default = word_budget(self._TEXT, self._SECONDS)
        at_measured = word_budget(self._TEXT, self._SECONDS, 120.0)

        assert at_default.raw_value["target"] == 30.0
        assert at_measured.raw_value["target"] == 24.0
        # Perfectly paced for the voice that will speak it — zero deviation.
        assert at_measured.raw_value["deviation_pct"] == 0.0
        assert at_measured.score == 100.0
        # ...and mis-scored against the assumed rate, by the full 20% deviation.
        assert at_default.raw_value["deviation_pct"] == 20.0
        assert at_default.score < at_measured.score

    def test_a_faster_measured_rate_moves_the_target_the_other_way(self) -> None:
        # Direction matters: a measured rate ABOVE the default must RAISE the
        # target, or this is a one-sided fudge rather than a real reference.
        assert word_budget(self._TEXT, self._SECONDS, 180.0).raw_value["target"] == 36.0

    def test_an_omitted_rate_is_byte_identical_to_the_old_constant(self) -> None:
        assert word_budget(self._TEXT, self._SECONDS) == word_budget(
            self._TEXT, self._SECONDS, 150.0
        )


class TestScoreScriptThreadsTheRate:
    _LINES: ClassVar[list[dict]] = [
        {"id": "clip-1", "text": " ".join(f"word{i}" for i in range(24)), "seconds": 12.0},
    ]

    def _budget(self, wpm: float | None) -> MetricResult:
        report = score_script(
            self._LINES,
            verbatim_items=[],
            target_duration_seconds=12.0,
            wpm=wpm,
        )
        return report["per_line"]["clip-1"]["word_budget"]

    def test_the_rate_reaches_the_per_line_metric(self) -> None:
        assert self._budget(120.0).raw_value["target"] == 24.0
        assert self._budget(None).raw_value["target"] == 30.0

    def test_the_rate_reaches_the_per_story_duration_budget(self) -> None:
        report = score_script(
            self._LINES,
            verbatim_items=[],
            target_duration_seconds=12.0,
            wpm=120.0,
        )
        assert report["per_story"]["duration_budget"].raw_value["target"] == 24.0

    def test_an_unusable_rate_falls_back_instead_of_scoring_against_nonsense(self) -> None:
        # A probe that degrades to 0.0/-1.0/NaN upstream must not make every beat
        # infinitely over budget — an absent measurement is not a measurement of 0.
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            assert self._budget(bad).raw_value["target"] == 30.0
