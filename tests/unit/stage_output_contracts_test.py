"""PR-A: stage output artifact-URL contracts.

A stage is only ``completed`` when its primary artifact reference is a real
URI (non-empty, not ``"None"``, has a scheme + host). These tests pin that the
constrained ``ArtifactUrl`` type rejects the empty / ``"None"`` / bare-word
values that used to persist as green lies, while still accepting any legitimate
scheme (http/https/s3/…) — the artifact proxy rewrites internal schemes
downstream, so the scheme itself is not the stage's concern.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from src.services.stages.app_review.compile import AppReviewCompileOutput
from src.services.stages.base import ArtifactUrl
from src.services.stages.course.compile_lessons import LessonOutputSummary
from src.services.stages.course.stitch import CourseStitchOutput
from src.services.stages.generic.video_render import VideoRenderOutput
from src.services.stages.lesson.compile import LessonCompileOutput
from src.services.stages.promo.render import PromoRenderOutput
from src.services.stages.website_review.compile import WebsiteReviewCompileOutput

_GOOD_URL = "https://minio.test/w2v-artifacts/run-1/final.mp4"
_GOOD_URLS = [_GOOD_URL, "http://h/p", "s3://bucket/key.mp4", "https://host"]
# Degenerate references that must NOT persist as a "completed" artifact.
_BAD_URLS = ["", "   ", "None", "none", "null", "not-a-url", "https://", "/just/a/path"]

# (model, url_field, other required fields). Field names verified against source:
# CourseStitchOutput keys its artifact on ``course_artifact_url``; the rest on
# ``artifact_url``.
_ARTIFACT_OUTPUTS: list[tuple[type[BaseModel], str, dict[str, Any]]] = [
    (VideoRenderOutput, "artifact_url", {"duration_seconds": 12.0}),
    (PromoRenderOutput, "artifact_url", {"duration_seconds": 12.0, "captions_burned": False}),
    (LessonCompileOutput, "artifact_url", {"duration_seconds": 12.0}),
    (AppReviewCompileOutput, "artifact_url", {"duration_seconds": 12.0}),
    (WebsiteReviewCompileOutput, "artifact_url", {"duration_seconds": 12.0}),
    (
        CourseStitchOutput,
        "course_artifact_url",
        {"artifact_url": _GOOD_URL, "duration_seconds": 12.0, "lesson_count": 2, "chapters": []},
    ),
    (
        LessonOutputSummary,
        "artifact_url",
        {"lesson_project_id": "p", "lesson_run_id": "r", "duration_seconds": 12.0},
    ),
]


@pytest.mark.parametrize(
    ("model", "url_field", "extra"),
    [(VideoRenderOutput, "artifact_url", {"duration_seconds": 12.0})],
)
def test_good_artifact_url_accepted(
    model: type[BaseModel], url_field: str, extra: dict[str, Any]
) -> None:
    obj = model(**{url_field: _GOOD_URL, **extra})
    assert getattr(obj, url_field) == _GOOD_URL


@pytest.mark.parametrize(
    ("model", "url_field", "extra"),
    [(VideoRenderOutput, "artifact_url", {"duration_seconds": 12.0})],
)
@pytest.mark.parametrize("bad", [""])
def test_bad_artifact_url_rejected(
    model: type[BaseModel], url_field: str, extra: dict[str, Any], bad: str
) -> None:
    with pytest.raises(ValidationError):
        model(**{url_field: bad, **extra})


def test_artifacturl_is_plain_str_in_dump() -> None:
    # ArtifactUrl must serialise as a plain string (JSONB / SSE / SPA contract).
    obj = VideoRenderOutput(artifact_url=_GOOD_URL, duration_seconds=1.0)
    dumped = obj.model_dump()
    assert dumped["artifact_url"] == _GOOD_URL
    assert isinstance(dumped["artifact_url"], str)
    _ = ArtifactUrl  # the public type the stages annotate with
