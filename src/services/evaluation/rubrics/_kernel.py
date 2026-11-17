"""Shared rubric-scoring kernel.

Centralises the one piece of construction every rubric metric repeats: building
a :class:`MetricResult` with its pass/warn/fail threshold auto-classified from
the score. Each rubric keeps its own domain heuristics and only declares its
dimensions (weights + names + detail strings) — ``config over kernel``.

Historical note: this module also carried ``tiered_score`` (a descending
``(minimum, score)`` ladder). Its only callers were the legacy per-stage rubrics
(``critique``/``redesign``/``review``/``script``), which were retired along with
the ``evaluate_pipeline`` surface that dispatched to them, so it went with them.
"""

from __future__ import annotations

from src.services.evaluation.models import MetricResult
from src.services.evaluation.scoring import classify_metric_threshold


def metric(
    name: str,
    score: float,
    weight: float,
    *,
    raw_value: object = None,
    detail: str = "",
    threshold: str | None = None,
) -> MetricResult:
    """Build a :class:`MetricResult`.

    The pass/warn/fail ``threshold`` is auto-classified from ``score`` via
    :func:`classify_metric_threshold` unless an explicit ``threshold`` override
    is supplied — the neutral / early-return branches pin a status ("pass",
    "warn", "fail") independent of the numeric score, and those pass it through.
    """
    return MetricResult(
        name=name,
        score=score,
        weight=weight,
        raw_value=raw_value,
        threshold=threshold if threshold is not None else classify_metric_threshold(score),
        detail=detail,
    )
