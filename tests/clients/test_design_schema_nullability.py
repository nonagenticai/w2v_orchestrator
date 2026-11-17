"""The design client's mirror must accept the UNSCORED shape the service really emits.

WHY THIS FILE EXISTS

`html_render_service` deliberately returns NULL for an accessibility score it
could not compute (`src/api/schemas.py`: `AccessibilityScoreResponse.overall_score:
float | None`, "None = UNSCORED (assessment did not happen). Never substitute a
number."; produced in `accessibility_scorer.py` on an unparseable LLM reply,
alongside an `unscored_reason`). The orchestrator's mirror declared both
`overall_score` and the per-category `score` as REQUIRED floats, so a design job
that completed — and was billed — blew up in `DesignJobResult.model_validate`
with a pydantic ValidationError. The user saw a traceback instead of
"accessibility unscored".

The mirror also silently DROPPED the service's degradation signals
(`unscored_reason`, `concept_image_degraded`, `concept_image_error`), which made a
design that lost its concept images indistinguishable from one where none were
requested. Pydantic discards unknown keys, so nothing failed — the information
just vanished.

These tests assert on service-shaped payloads, not on hand-built models.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.clients.schemas.design import (
    DesignAccessibilityCategory,
    DesignAccessibilityScore,
    DesignJobResult,
    DesignJobStatus,
)


def _result_payload(**overrides: object) -> dict[str, object]:
    """A minimal `RedesignResultResponse`-shaped body, as the service serializes it."""
    payload: dict[str, object] = {
        "job_id": "dsn-1",
        "url": "https://acme.test",
        "scope": "redesign",
        "style": "modern",
    }
    payload.update(overrides)
    return payload


def test_unscored_accessibility_validates_and_reason_is_readable() -> None:
    """The exact shape `accessibility_scorer` emits when the LLM reply is unparseable."""
    result = DesignJobResult.model_validate(
        _result_payload(
            accessibility={
                "overall_score": None,
                "unscored_reason": "llm reply was not valid json",
                "categories": [],
                "findings": [],
                "html_was_fixed": False,
            }
        )
    )

    assert result.accessibility is not None
    # UNSCORED must survive as None — never coerced to 0.0, which would be
    # indistinguishable from a real "totally inaccessible" measurement.
    assert result.accessibility.overall_score is None
    assert result.accessibility.unscored_reason == "llm reply was not valid json"


def test_unscored_category_score_validates() -> None:
    """A category the LLM omitted arrives as `score: null`, not as a number."""
    score = DesignAccessibilityScore.model_validate(
        {
            "overall_score": 82.0,
            "categories": [
                {"category": "contrast", "score": 91.0, "findings": []},
                {"category": "aria", "score": None, "findings": []},
            ],
        }
    )

    assert [c.score for c in score.categories] == [91.0, None]


def test_category_score_defaults_to_none_when_absent() -> None:
    """An omitted score is UNSCORED, not zero — matching the server default."""
    assert DesignAccessibilityCategory.model_validate({"category": "aria"}).score is None


def test_a_scored_assessment_still_validates() -> None:
    """The happy path must keep working — nullability is not a loosening of the number."""
    score = DesignAccessibilityScore.model_validate(
        {"overall_score": 88.5, "categories": [], "findings": []}
    )
    assert score.overall_score == 88.5
    assert score.unscored_reason is None


def test_concept_image_degradation_survives_on_the_result() -> None:
    """A failed concept-image generation must be distinguishable from "none requested"."""
    result = DesignJobResult.model_validate(
        _result_payload(
            concept_image_assets=[],
            concept_image_degraded=True,
            concept_image_error="ai_content_generation_service returned 503",
        )
    )

    assert result.concept_image_degraded is True
    assert result.concept_image_error == "ai_content_generation_service returned 503"


def test_concept_image_degradation_survives_on_the_status() -> None:
    """The same signal rides on the status body the orchestrator polls."""
    status = DesignJobStatus.model_validate(
        {
            "job_id": "dsn-1",
            "status": "completed",
            "concept_image_degraded": True,
            "concept_image_error": "image provider timeout",
        }
    )

    assert status.concept_image_degraded is True
    assert status.concept_image_error == "image provider timeout"


def test_no_degradation_reads_as_clean() -> None:
    """Absent signals default to "not degraded" — the field must not be required."""
    result = DesignJobResult.model_validate(_result_payload())
    assert result.concept_image_degraded is False
    assert result.concept_image_error is None

    status = DesignJobStatus.model_validate({"job_id": "dsn-1", "status": "completed"})
    assert status.concept_image_degraded is False
    assert status.concept_image_error is None


def test_verification_and_html_validation_survive() -> None:
    """The other two degradation surfaces the service exposes on the result."""
    result = DesignJobResult.model_validate(
        _result_payload(
            html_validation={"errors": ["unclosed <div>"], "passed": False},
            verification_status="failed",
            verification_report={"reason": "screenshot diff over threshold"},
        )
    )

    assert result.html_validation == {"errors": ["unclosed <div>"], "passed": False}
    assert result.verification_status == "failed"
    assert result.verification_report == {"reason": "screenshot diff over threshold"}


def test_a_non_numeric_score_is_still_rejected() -> None:
    """Optional is not "anything goes" — a string score is still a contract break."""
    with pytest.raises(ValidationError):
        DesignAccessibilityScore.model_validate({"overall_score": "unscored"})
