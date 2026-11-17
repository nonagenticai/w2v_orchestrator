"""Metric pass/warn/fail classification."""

from __future__ import annotations


def classify_metric_threshold(score: float) -> str:
    """Classify a single metric score as pass/warn/fail."""
    if score >= 70:
        return "pass"
    if score >= 40:
        return "warn"
    return "fail"
