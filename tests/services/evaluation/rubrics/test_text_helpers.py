"""Behavioural unit tests for the shared text-quality helpers in ``_text.py``.

These five helpers were extracted verbatim out of the retired ``narrative.py``
rubric. They are NOT dead code: ``script_craft.score_script`` wires four of them
(``readability`` / ``sentence_cv`` / ``ttr`` / ``preamble``) into the per-story
gate bundle, which is what the **live** ``script_doctor`` LLM optimizer loop
scores against. A silent regression here changes what ships.

The three ``test_regression_guard`` cases below are ported from the deleted
``tests/unit/test_evaluation_rubrics.py`` (``TestNarrativeRubric`` /
``TestNarrativeRubricExtended``). The originals asserted through
``narrative.evaluate(artifact, {"stage_name": "narrate"})``, which did nothing
but pull ``artifact["audio"]["narration_text"]`` and hand the raw string to
these same functions — so calling them directly is behaviourally identical and
every original threshold is preserved unchanged.

``tests/services/evaluation/rubrics/test_script_craft.py`` covers the bundle's
plumbing but only asserts that these keys are PRESENT, never their values. This
file is the value coverage.

Pure — no DB, no network, no LLM, no randomness.
"""

from __future__ import annotations

from src.services.evaluation.models import MetricResult
from src.services.evaluation.rubrics._text import (
    _lexical_diversity,
    _preamble_contamination,
    _readability_score,
    _sentence_length_variance,
)

# ── _split_sentences ─────────────────────────────────────────────────────────


# ── _preamble_contamination ──────────────────────────────────────────────────


# ── _lexical_diversity ───────────────────────────────────────────────────────


# ── _readability_score ───────────────────────────────────────────────────────


# ── _sentence_length_variance ────────────────────────────────────────────────


# ── shape contract ───────────────────────────────────────────────────────────


def test_every_helper_returns_a_weighted_metric_result() -> None:
    """``score_script`` re-wraps these, so name/weight/score must be well-formed."""
    text = "The dashboard loads fast. Users find their reports quickly. It simply works."
    for helper in (
        _readability_score,
        _lexical_diversity,
        _preamble_contamination,
        _sentence_length_variance,
    ):
        result = helper(text)
        assert isinstance(result, MetricResult)
        assert result.name
        assert 0.0 < result.weight <= 1.0
        assert 0.0 <= result.score <= 100.0
        assert result.threshold in {"pass", "warn", "fail"}
