"""The music/ambient bed must outlive the compile, and a bed the compiler did not
apply must be RECORDED.

Two defects on one path (the bed a compile job fetches):

* D — the bed URL was presigned with the storage default (3600s) while the
  orchestrator waits up to ``COMPILE_WAIT_CEILING_SECONDS`` (7500s) for the
  compile, so a compile over an hour 403'd on the bed and shipped without it.
* C — the compiler reports that loss (``music_bed.applied is False``), and the
  shared compile path never read it: the run ended ``completed`` with no bed and
  no signal.

The website_review ``build_audio_bed`` override is used as the vehicle on
purpose: it is the path the brief named as an override of the shared resolver.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.services import tts_storage
from src.services.blueprints.course import COURSE_BLUEPRINT
from src.services.compile_budget import COMPILE_WAIT_CEILING_SECONDS, PRESIGN_TTL_SECONDS
from src.services.runs.run_context import RunContext
from src.services.stages import sound_catalog
from src.services.stages.course.stitch import CourseStitchStage
from src.services.stages.primitives.compile_stage import music_bed_degradations
from src.services.stages.shared import sound_design
from src.services.stages.shared.degradation import MUSIC_BED_NOT_APPLIED
from src.services.stages.sound_catalog import SoundLibrary
from src.services.stages.website_review.compile import WebsiteReviewCompileStage


class _RecordingStorage:
    """Only ``get_presigned_url`` — no ``resign_presigned_url``, so the shot
    re-sign pass leaves shots alone and every recorded call is a bed presign."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int | None]] = []

    async def initialize(self) -> None:
        pass

    async def store_stage_artifact(
        self, job_id: str, stage_name: str, data: bytes, filename: str
    ) -> str:
        return f"artifacts/{job_id}/{stage_name}/{filename}"

    async def get_presigned_url(
        self, path: str, expires: int | None = None, bucket: str | None = None
    ) -> str | None:
        self.calls.append((path, expires))
        return f"https://minio.invalid/{path}?X-Amz-Expires={expires}"


@pytest.fixture
def storage(monkeypatch: pytest.MonkeyPatch) -> Iterator[_RecordingStorage]:
    async def _library(*_a: object, **_k: object) -> SoundLibrary:
        return SoundLibrary.builtin()

    fake = _RecordingStorage()
    # The REAL presign_sound_asset runs; only its library + storage are faked.
    monkeypatch.setattr(sound_catalog, "load_sound_library", _library)
    monkeypatch.setattr(sound_design, "load_sound_library", _library)
    tts_storage.set_storage_for_tests(fake)
    yield fake
    tts_storage.set_storage_for_tests(None)


def _ctx(sound: dict[str, Any] | None) -> RunContext:
    snapshot: dict[str, Any] = {"deliverable": {"format": "16x9"}}
    if sound is not None:
        snapshot["sound_effects"] = sound
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot=snapshot,
        prior_run_outputs={
            "website_review_script": {
                "lines": [
                    {"id": "L1", "text": "Problem.", "seconds": 5.0, "phase": "before"},
                    {"id": "L2", "text": "Fix.", "seconds": 4.0, "phase": "after"},
                ]
            },
            "website_review_record": {
                "clips": [
                    {"id": "c1", "line_id": "L1", "artifact_url": "http://a/1.wav", "seconds": 5.0},
                    {"id": "c2", "line_id": "L2", "artifact_url": "http://a/2.wav", "seconds": 4.0},
                ]
            },
            "website_redesign": {
                "visuals": [
                    {"id": "v1", "role": "before", "image_url": "http://i/before.png"},
                    {"id": "v2", "role": "after", "image_url": "http://i/after.png"},
                ]
            },
        },
    )


_MUSIC = {"enabled": True, "music": "jazz_lounge", "ambient": False}


def _client(music_bed: dict[str, Any] | None) -> AsyncMock:
    fake = AsyncMock()
    result: dict[str, Any] = {"artifact_url": "http://stub/review.mp4", "duration_seconds": 9.0}
    if music_bed is not None:
        result["music_bed"] = music_bed
    fake.compile.return_value = result
    return fake


def _codes(out: Any) -> list[str]:
    return [d["code"] for d in out.degradations]


# ── D: bed presign TTL ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_bed_is_presigned_with_the_compile_ttl(storage: _RecordingStorage) -> None:
    fake = _client({"requested": True, "applied": True})
    await WebsiteReviewCompileStage(client=fake).run({}, ctx=_ctx(_MUSIC))

    music = fake.compile.call_args.kwargs.get("music")
    assert music is not None, "no music bed reached the compile request"
    bed_calls = [c for c in storage.calls if "jazz_lounge" in c[0]]
    assert bed_calls, storage.calls
    assert {expires for _, expires in bed_calls} == {PRESIGN_TTL_SECONDS}
    # The whole point: the bed must still be fetchable at the end of the longest wait.
    assert PRESIGN_TTL_SECONDS > COMPILE_WAIT_CEILING_SECONDS


# ── C: compiler-reported unapplied bed ────────────────────────────────────────


@pytest.mark.asyncio
async def test_unapplied_bed_is_a_degradation(storage: _RecordingStorage) -> None:
    fake = _client({"requested": True, "applied": False, "fetch_failed": True})
    out = await WebsiteReviewCompileStage(client=fake).run({}, ctx=_ctx(_MUSIC))

    bed = [d for d in out.degradations if d["code"] == MUSIC_BED_NOT_APPLIED]
    assert len(bed) == 1, _codes(out)
    assert "fetch_failed=True" in bed[0]["detail"]
    # A degradation, not a failure: the artifact still ships.
    assert out.artifact_url == "http://stub/review.mp4"


@pytest.mark.asyncio
async def test_applied_bed_records_nothing(storage: _RecordingStorage) -> None:
    fake = _client({"requested": True, "applied": True, "fetch_failed": False})
    out = await WebsiteReviewCompileStage(client=fake).run({}, ctx=_ctx(_MUSIC))
    assert MUSIC_BED_NOT_APPLIED not in _codes(out)


@pytest.mark.asyncio
async def test_no_bed_requested_ignores_a_stray_report(storage: _RecordingStorage) -> None:
    fake = _client({"requested": True, "applied": False, "fetch_failed": True})
    out = await WebsiteReviewCompileStage(client=fake).run({}, ctx=_ctx(None))
    assert fake.compile.call_args.kwargs.get("music") is None
    assert MUSIC_BED_NOT_APPLIED not in _codes(out)


def test_absent_report_and_ambient_loss() -> None:
    assert music_bed_degradations({}, {"music": {"url": "u"}}) == []
    lost = music_bed_degradations(
        {"music_bed": {"ambient_fetch_failed": True, "ambient_lost": 1, "ambient_requested": 1}},
        {"ambient": [{"url": "u"}]},
    )
    assert [d["code"] for d in lost] == [MUSIC_BED_NOT_APPLIED]


# ── E: course stitch waits as long as every other compile ─────────────────────


@pytest.mark.asyncio
async def test_course_stitch_waits_the_compile_ceiling(storage: _RecordingStorage) -> None:
    fake = AsyncMock()
    fake.compile.return_value = {"artifact_url": "http://stub/c.mp4", "duration_seconds": 60.0}
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={},
        prior_run_outputs={
            "course_compile_lessons": {
                "lessons": [
                    {
                        "lesson_project_id": "p1",
                        "lesson_run_id": "r1",
                        "artifact_url": "http://stub/l1.mp4",
                        "duration_seconds": 60.0,
                    }
                ]
            }
        },
    )
    await CourseStitchStage(client=fake).run({}, ctx=ctx)

    assert fake.compile.call_args.kwargs["timeout"] == COMPILE_WAIT_CEILING_SECONDS
    outer = COURSE_BLUEPRINT.stage("course_stitch").default_timeout_seconds
    assert outer is not None and outer > COMPILE_WAIT_CEILING_SECONDS
