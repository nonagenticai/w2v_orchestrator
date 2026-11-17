"""A ken_burns pass that upgraded NOTHING has to say so in the persisted output.

``apply_ken_burns`` soft-fails every clip it cannot render, and that is correct:
a still that keeps its still must never abort the compile. But motion is ON by
default for stills on EVERY blueprint, so when video_production is down or the
presigns have expired the whole pass degrades at once — every still stays a
still, the compile succeeds, and the run completes ``completed`` with nothing
recorded anywhere. What ships is exactly the ~92%-duplicate-frame slideshow the
function's own docstring exists to prevent. ``final_critique._static_ratio``
reads the symptom but is advisory and demo_video-only, so promo/lesson/course/
action_reel/website_review have no signal at all.

Asserted on the RETURNED OUTPUT, never on the notes sink and never on a call
count: ``ken_burns.py`` filling a list it was handed proves nothing about whether
``_run_presubmit`` passes its list in and ``handle`` folds it into the output
that is PERSISTED. Delete the wiring and a sink-level assertion still passes.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import uuid7

import pytest

from src.services import tts_storage
from src.services.stages.base import ArtifactUrl, StageInput, StageOutputWithNotes
from src.services.stages.primitives.compile_stage import CompileStageBase

_STILL_URL = "https://example.invalid/still-a.jpg"


class _FakeStorage:
    """Enough storage for the compile tail's re-sign passes."""

    async def resign_presigned_url(self, url: str) -> str:
        return url


class _Input(StageInput):
    pass


class _NotesOutput(StageOutputWithNotes):
    """A final-video compile output — the promo/demo_video shape."""

    artifact_url: ArtifactUrl
    duration_seconds: float


class _StillsStage(CompileStageBase[_Input, _NotesOutput]):
    """A product whose shots are bare stills, i.e. the ken_burns candidate set."""

    name = "test_compile_ken_burns"
    input_model = _Input
    output_model = _NotesOutput

    def ken_burns_local_only(self, ctx: Any) -> bool:
        # Pins the video_production RECORDER path, which the base no longer takes.
        return False

    def build_shots(self, ctx: Any) -> list[dict[str, Any]]:
        return [
            {"id": "s1", "line_id": "l1", "seconds": 4.0, "image_url": _STILL_URL},
            {"id": "s2", "line_id": "l2", "seconds": 4.0, "image_url": _STILL_URL},
        ]

    def build_output(self, result: dict[str, Any], ctx: Any) -> _NotesOutput:
        return _NotesOutput(
            artifact_url=str(result["artifact_url"]),
            duration_seconds=float(result["duration_seconds"]),
        )

    def compile_tags(self, ctx: Any) -> dict[str, str]:
        return {}


def _capturing_client(captured: dict[str, Any]) -> AsyncMock:
    async def _compile(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"artifact_url": "http://m/w2v-compiled-videos/out.mp4", "duration_seconds": 8.0}

    client = AsyncMock()
    client.compile = AsyncMock(side_effect=_compile)
    return client


@pytest.mark.asyncio
async def test_a_fully_degraded_ken_burns_pass_reaches_the_persisted_stage_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE wiring. Every still soft-fails; the compile still runs (soft-fail
    intact) and the run persists both WHICH shots stayed stills and the louder
    all-degraded line."""
    boom = RuntimeError("video_production 503: ken_burns record failed")

    # Patched at the FETCH seam inside ``ken_burns`` — never at ``apply_ken_burns``
    # itself, which would skip the soft-fail/append path under test.
    async def _boom(*_a: Any, **_kw: Any) -> tuple[bytes, str]:
        raise boom

    monkeypatch.setattr("src.services.stages.shared.ken_burns._fetch_own_asset", _boom)

    async def _no_cache(*_a: Any, **_kw: Any) -> str | None:
        return None

    monkeypatch.setattr("src.services.stages.shared.ken_burns.find_existing_media", _no_cache)

    captured: dict[str, Any] = {}
    stage = _StillsStage(_capturing_client(captured))

    tts_storage.set_storage_for_tests(cast(Any, _FakeStorage()))
    try:
        output = await stage.handle(
            _Input(),
            ctx=cast(
                Any,
                SimpleNamespace(
                    run_id=uuid7(),
                    render_idempotency_discriminator="",
                    config_snapshot={"title": "Ridgeline"},
                    prior_output=lambda _stage: None,
                ),
            ),
        )
    finally:
        tts_storage.set_storage_for_tests(None)

    # Soft-fail intact: both shots reached the compiler, still as stills.
    assert [str(shot.get("id")) for shot in captured["shots"]] == ["s1", "s2"]
    assert not any(shot.get("video_url") for shot in captured["shots"])

    # ...and the reasons survived into the output the run PERSISTS.
    per_shot = [note for note in output.notes if "shot s1" in note]
    assert per_shot, f"no per-shot ken_burns reason reached the output: {output.notes}"
    assert "RuntimeError" in per_shot[0] and str(boom) in per_shot[0], per_shot[0]

    all_degraded = [
        note for note in output.notes if "ken_burns: ALL" in note and "frozen stills" in note
    ]
    assert all_degraded, f"the all-degraded reason never reached the output: {output.notes}"
