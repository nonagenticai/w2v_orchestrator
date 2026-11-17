"""Contract tests for the ai_content_generation typed response models.

These pin the shape trap that a hand-written mirror got wrong and shipped as a
prod bug: the VIDEO status envelope is THIN (no asset URL), the VIDEO result
envelope is FAT. ``ai_video_gen`` used to read ``video_url`` off the *status*
poll — where the field does not exist — and always tripped its own
downstream-response gate. Modelling the two shapes as distinct classes makes
that a type error, and these tests pin the field sets against the downstream
``ai_content_generation_service/src/api/schemas.py``.

DB-free, unit tier.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.clients.schemas.ai_content_generation import (
    AcgConceptImageJob,
    AcgVideoGenerationResult,
    AcgVideoJobStatus,
)


def test_video_status_is_thin_no_asset_fields() -> None:
    """The status poll (``GET /jobs/{id}``) carries NO video_url/duration — only
    a ``result_url`` pointer. Guards the exact bug that motivated this model."""
    status = AcgVideoJobStatus.model_validate(
        {
            "job_id": "v1",
            "status": "completed",
            "provider": "stub",
            "result_url": "/api/content/videos/jobs/v1/result",
        }
    )
    assert status.status == "completed"
    assert status.result_url == "/api/content/videos/jobs/v1/result"
    # The bug: these asset fields do NOT exist on the status model.
    assert not hasattr(status, "video_url")
    assert not hasattr(status, "asset_id")
    assert not hasattr(status, "actual_duration_s")


def test_video_result_carries_url_and_duration() -> None:
    """The result endpoint (``/jobs/{id}/result``) is FAT: video_url + the
    duration field is ``actual_duration_s`` (NOT ``duration_s``)."""
    result = AcgVideoGenerationResult.model_validate(
        {
            "job_id": "v1",
            "status": "completed",
            "provider": "stub",
            "video_url": "https://minio.test/v/v1.mp4",
            "actual_duration_s": 5.0,
            "cost": 0.12,
        }
    )
    assert result.video_url == "https://minio.test/v/v1.mp4"
    assert result.actual_duration_s == 5.0
    # ``duration_s`` (the name the old consumer read) is not a field here.
    assert not hasattr(result, "duration_s")


def test_video_result_all_optional_fields_default() -> None:
    """A minimal completed payload (only the two required fields) must validate;
    every asset field is optional because a provider can 200 with a dead job."""
    result = AcgVideoGenerationResult.model_validate({"job_id": "v1", "status": "completed"})
    assert result.video_url is None
    assert result.actual_duration_s is None


def test_video_status_requires_job_id_and_status() -> None:
    with pytest.raises(ValidationError):
        AcgVideoJobStatus.model_validate({"status": "completed"})  # missing job_id


def test_concept_image_status_is_fat_carries_asset_inline() -> None:
    """Unlike video, the concept-image STATUS already carries the asset inline —
    which is why the image path never needed a second result fetch."""
    job = AcgConceptImageJob.model_validate(
        {
            "job_id": "i1",
            "asset_id": "asset-1",
            "status": "completed",
            "image_url": "https://minio.test/c/asset-1.png",
            "download_url": "/api/concept-images/assets/asset-1/download",
        }
    )
    assert job.asset_id == "asset-1"
    assert job.image_url == "https://minio.test/c/asset-1.png"
    assert job.download_url == "/api/concept-images/assets/asset-1/download"


def test_models_ignore_unknown_downstream_fields() -> None:
    """extra='ignore' (pydantic default): a downstream ADDING a field can never
    break the orchestrator."""
    result = AcgVideoGenerationResult.model_validate(
        {"job_id": "v1", "status": "completed", "brand_new_downstream_field": {"x": 1}}
    )
    assert result.job_id == "v1"
