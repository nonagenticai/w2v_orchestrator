"""Deterministic craft scoring for the ``script_doctor`` loop.

Scope note: this package used to host a generic per-stage pipeline evaluator
(``evaluate_pipeline`` / ``evaluate_stage`` + a rubric module per legacy stage).
That surface had no production caller and its stage→rubric map keyed on stage
names that no longer exist, so it was retired. What remains is the one
live-reachable path: :mod:`~src.services.evaluation.rubrics.script_craft`.
"""

from src.services.evaluation.models import MetricResult

__all__ = ["MetricResult"]
