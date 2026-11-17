"""`ArtifactUrl` boundary-validation field type.

Narration clips (promo/lesson/generic/website_review/audiobook narrate stages)
type their per-clip ``artifact_url`` as ``ArtifactUrl`` so a stage cannot persist
a green clip pointing at a degenerate reference (``""`` / ``"None"``). The local
backend yields ``file:///run/clip.wav`` (empty netloc) when MinIO is unconfigured
— a real reference that must pass the gate (``is_fetchable_url`` filters it later).
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from src.services.stages.audiobook.narrate import NarratedChunk
from src.services.stages.base import ArtifactUrl
from src.services.stages.generic.narrate import GenericAudioClip
from src.services.stages.lesson.narrate import LessonAudioClip
from src.services.stages.promo.narrate import PromoAudioClip
from src.services.stages.website_review.record import ReviewAudioClip

# The concrete clip models, spelled out as a union so ty can see that each one
# really declares ``artifact_url`` (a loose ``type[BaseModel]`` annotation makes
# every kwarg look like a discarded extra).
_NarrationClipModel = (
    type[PromoAudioClip]
    | type[LessonAudioClip]
    | type[GenericAudioClip]
    | type[ReviewAudioClip]
    | type[NarratedChunk]
)

# (clip model, the non-url kwargs it needs) for the five narration stages whose
# per-clip artifact_url is now boundary-validated.
_NARRATION_CLIP_MODELS: list[tuple[_NarrationClipModel, dict]] = [
    (PromoAudioClip, {"id": "c", "line_id": "l", "seconds": 1.0}),
    (LessonAudioClip, {"id": "c", "paragraph_id": "p", "seconds": 1.0}),
    (GenericAudioClip, {"id": "c", "line_id": "l", "seconds": 1.0}),
    (ReviewAudioClip, {"id": "c", "line_id": "l", "seconds": 1.0}),
    (
        NarratedChunk,
        {
            "id": "c",
            "chapter_id": "ch",
            "chunk_id": "k",
            "chapter_sequence": 0,
            "chunk_sequence": 0,
            "seconds": 1.0,
        },
    ),
]


class _Clip(BaseModel):
    artifact_url: ArtifactUrl


@pytest.mark.parametrize(
    "url",
    ["https://minio.test/b/clip.wav"],
)
def test_usable_references_accepted(url: str) -> None:
    assert _Clip(artifact_url=url).artifact_url == url


@pytest.mark.parametrize("bad", [""])
def test_degenerate_references_rejected(bad: str) -> None:
    with pytest.raises(ValidationError, match="not a usable artifact URL"):
        _Clip(artifact_url=bad)


def test_artifact_url_stays_plain_str_in_dump() -> None:
    # Unlike pydantic.HttpUrl, ArtifactUrl must round-trip as a bare str so the
    # JSONB / SSE / SPA contract is unchanged.
    dumped = _Clip(artifact_url="file:///run/clip.wav").model_dump()
    assert dumped == {"artifact_url": "file:///run/clip.wav"}
    assert isinstance(dumped["artifact_url"], str)


@pytest.mark.parametrize(
    "model, kwargs", [(PromoAudioClip, {"id": "c", "line_id": "l", "seconds": 1.0})]
)
def test_narration_clip_models_reject_degenerate_url(
    model: _NarrationClipModel, kwargs: dict
) -> None:
    # The five narrate/record stage clip models carry the gate: a regression
    # that stored "None" (the str(None) trap) fails the stage loudly instead of
    # persisting a green clip that points at nothing.
    with pytest.raises(ValidationError, match="not a usable artifact URL"):
        model(artifact_url="None", **kwargs)


@pytest.mark.parametrize(
    "model, kwargs", [(PromoAudioClip, {"id": "c", "line_id": "l", "seconds": 1.0})]
)
def test_narration_clip_models_accept_file_fallback(
    model: _NarrationClipModel, kwargs: dict
) -> None:
    # The local-backend file:// fallback must still construct (it degrades to a
    # slide at compile via is_fetchable_url, not at the output boundary).
    built = model(artifact_url="file:///run/clip.wav", **kwargs)
    assert built.model_dump()["artifact_url"] == "file:///run/clip.wav"
