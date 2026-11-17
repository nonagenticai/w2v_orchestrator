"""Evaluation data models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MetricResult(BaseModel):
    """A single evaluation metric result."""

    name: str
    score: float = Field(ge=0, le=100)
    weight: float = Field(ge=0, le=1)
    raw_value: Any = None
    threshold: str = "pass"  # "pass" | "warn" | "fail"
    detail: str = ""
