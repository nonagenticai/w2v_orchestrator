"""Contract tests for the bdd_playwright typed response models.

These pin the two shape traps that a hand-written mirror gets wrong, both read
out of the downstream service rather than assumed:

1. ``narration_script`` is a JSON column that serializes as ``{}`` when the job
   produced no narration (bdd ``src/api/routes/jobs.py`` -> ``job.narration_script
   or {}``). Every field must therefore have a default.
2. ``expected_text`` is written onto a part **only** when the knowledge pack
   authored one (bdd ``src/services/job_processor.py`` -> ``if
   expected_text.get(part_id)``). Modelling it as required would reject every
   ordinary (non-authored) demo run.

They are DB-free and run in the unit tier.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.clients.schemas.bdd_playwright import BddJobResult, BddJobStatus


def test_result_accepts_empty_narration_script():
    """A job with no narration sends ``{}`` — it must validate, not explode."""
    result = BddJobResult.model_validate(
        {"job_id": "j1", "status": "completed", "narration_script": {}}
    )
    assert result.narration_script.parts == []
    assert result.narration_script.total_words == 0


def test_result_accepts_missing_narration_script_entirely():
    result = BddJobResult.model_validate({"job_id": "j1", "status": "failed", "error": "boom"})
    assert result.narration_script.parts == []
    assert result.error == "boom"


def test_part_expected_text_is_optional_and_defaults_none():
    """The ordinary (non-authored) path omits ``expected_text`` on every part."""
    result = BddJobResult.model_validate(
        {
            "job_id": "j1",
            "status": "completed",
            "narration_script": {"parts": [{"part_id": "clip-1", "text": "Hi."}]},
        }
    )
    part = result.narration_script.parts[0]
    assert part.expected_text is None
    # Unset numeric/string fields fall back to defaults rather than failing.
    assert part.estimated_duration_seconds == 0.0
    assert part.scenario_name == ""


def test_part_carries_authored_expected_text_when_present():
    """O1b: the pack-authored expected on-screen text survives onto the part."""
    result = BddJobResult.model_validate(
        {
            "job_id": "j1",
            "status": "completed",
            "narration_script": {
                "parts": [
                    {
                        "part_id": "clip-1",
                        "text": "Hi.",
                        "expected_text": "Dashboard",
                        "estimated_duration_seconds": 2.5,
                    }
                ]
            },
        }
    )
    part = result.narration_script.parts[0]
    assert part.expected_text == "Dashboard"
    assert part.estimated_duration_seconds == 2.5


def test_unknown_downstream_field_is_ignored_not_fatal():
    """Forward-compat: bdd ADDING a response field must never break a live run."""
    result = BddJobResult.model_validate(
        {"job_id": "j1", "status": "completed", "some_future_bdd_field": {"nested": 1}}
    )
    assert result.job_id == "j1"


def test_opaque_json_blobs_round_trip_without_any():
    """``artifact_paths`` / ``downstream_jobs`` accept arbitrary JSON objects."""
    result = BddJobResult.model_validate(
        {
            "job_id": "j1",
            "status": "completed",
            "artifact_paths": {"zip": "s3://bucket/pkg.zip"},
            "downstream_jobs": {"tts": {"job_id": "t1"}},
        }
    )
    assert result.artifact_paths["zip"] == "s3://bucket/pkg.zip"
    assert result.downstream_jobs["tts"] == {"job_id": "t1"}


def test_unknown_status_value_is_rejected():
    """``status`` is the one field the orchestrator branches on — a value outside
    the shared JobStatus literal is a real contract break and must be loud."""
    with pytest.raises(ValidationError):
        BddJobResult.model_validate({"job_id": "j1", "status": "cancelled"})


def test_step_coverage_survives_model_validate():
    """The grounding report must not be dropped by ``extra="ignore"`` — it was,
    before ``BddJobResult`` declared the field at all. Proves both the drop and
    the fix: fails on a ``BddJobResult`` with no ``step_coverage`` field."""
    result = BddJobResult.model_validate(
        {
            "job_id": "j1",
            "status": "completed",
            "step_coverage": {
                "steps_total": 10,
                "steps_stubbed": 2,
                "stub_ratio": 0.2,
                "degraded": False,
                "degraded_threshold": 0.5,
                "ungrounded_gherkin_concepts_flagged": 1,
                "ungrounded_gherkin_concepts": [
                    {
                        "file": "demo.feature",
                        "scenario": "Add a todo",
                        "step": "Then I should see the completed count",
                        "concept": "completed count",
                    }
                ],
            },
        }
    )
    assert result.step_coverage is not None
    assert result.step_coverage.ungrounded_gherkin_concepts_flagged == 1
    assert result.step_coverage.ungrounded_gherkin_concepts[0].concept == "completed count"


def test_step_coverage_empty_wire_dict_is_absent_not_a_clean_report():
    """bdd sends ``job.step_coverage or {}`` when the column was never
    populated — that must validate to ``None`` (ABSENT), never to a
    ``BddStepCoverage`` with all-zero defaults (which would read identically to
    a genuinely reported, zero-findings suite)."""
    result = BddJobResult.model_validate(
        {"job_id": "j1", "status": "completed", "step_coverage": {}}
    )
    assert result.step_coverage is None


def test_step_coverage_missing_entirely_is_also_absent():
    result = BddJobResult.model_validate({"job_id": "j1", "status": "completed"})
    assert result.step_coverage is None


def test_step_coverage_reported_clean_is_distinguishable_from_absent():
    """EMPTY (reported, nothing flagged) must be a real object, not ``None`` —
    a future reader needs to tell "bdd ran grounding and found nothing" apart
    from "bdd never ran grounding"."""
    result = BddJobResult.model_validate(
        {
            "job_id": "j1",
            "status": "completed",
            "step_coverage": {"steps_total": 6, "steps_stubbed": 0},
        }
    )
    assert result.step_coverage is not None
    assert result.step_coverage.ungrounded_gherkin_concepts_flagged == 0
    assert result.step_coverage.ungrounded_gherkin_concepts == []


def test_status_envelope_validates_poll_payload():
    """``wait_for_job``/``get_job`` settle on the status envelope, not the result."""
    status = BddJobStatus.model_validate(
        {"job_id": "j1", "status": "processing", "stream": "demo_video"}
    )
    assert status.status == "processing"
    assert status.download_url is None
