"""Threshold classification — the boundary rule every rubric metric inherits.

Moved here from the retired ``tests/unit/test_evaluation_scoring.py`` (whose
other cases covered the deleted ``score_to_grade`` / ``compute_weighted_score``
/ ``classify_pass_fail`` surface) so the surviving ``_kernel.metric`` path keeps
its boundary coverage.
"""

from __future__ import annotations

from src.services.evaluation.rubrics._kernel import metric
from src.services.evaluation.scoring import classify_metric_threshold


def test_classify_metric_threshold() -> None:
    """Metric-level pass/warn/fail boundaries (>=70 pass, >=40 warn, else fail)."""
    assert classify_metric_threshold(80.0) == "pass"
    assert classify_metric_threshold(50.0) == "warn"
    assert classify_metric_threshold(30.0) == "fail"
    assert classify_metric_threshold(70.0) == "pass"
    assert classify_metric_threshold(39.9) == "fail"


def test_metric_auto_classifies_threshold_from_score() -> None:
    assert metric("m", 90.0, 0.5).threshold == "pass"
    assert metric("m", 55.0, 0.5).threshold == "warn"
    assert metric("m", 10.0, 0.5).threshold == "fail"


def test_metric_explicit_threshold_overrides_score() -> None:
    """Neutral / early-return branches pin a status independent of the score."""
    result = metric("m", 50.0, 0.5, threshold="pass", detail="no data", raw_value={"n": 0})
    assert result.threshold == "pass"
    assert result.score == 50.0
    assert result.detail == "no data"
