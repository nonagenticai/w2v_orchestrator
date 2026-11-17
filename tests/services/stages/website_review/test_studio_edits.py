"""Studio edits on a website_review run must reach the compiled video.

Prod (run 01a09f82-7179-73a5-a23a-40c15d148b37): apply-edits 422'd
``stage 'website_review_script' does not carry a clip timeline``, and even past that
gate the compile stage had no ``build_audio_bed`` (its ``audio_bed`` phase ran 0.0s)
and rebuilt its shots without the per-clip edit fields — so a music bed + ducking
edit, or a caption/speed edit, would have re-stitched a green child run that changed
nothing.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.services.blueprints.website_review import WEBSITE_REVIEW_BLUEPRINT
from src.services.runs.elements import compile_cut_stage, resolve_clip_spine_stage
from src.services.runs.run_context import RunContext
from src.services.stages.shared import sound_design
from src.services.stages.sound_catalog import SoundLibrary
from src.services.stages.website_review.compile import WebsiteReviewCompileStage


class _Row:
    def __init__(self, stage_name: str, output: dict[str, Any]) -> None:
        self.stage_name = stage_name
        self.status = "completed"
        self.output = output


def test_website_review_script_is_the_clip_spine_and_compile_is_the_cut() -> None:
    lines = [{"id": "L1", "text": "a"}, {"id": "L2", "text": "b"}]
    rows = [
        _Row("website_review_script", {"lines": lines}),
        _Row("website_review_record", {"clips": [{"id": "c1", "line_id": "L1"}]}),
        _Row("website_review_visual_director", {"assignments": [{"line_id": "L2"}]}),
    ]
    spine = resolve_clip_spine_stage(WEBSITE_REVIEW_BLUEPRINT, cast("Any", rows))
    assert spine == "website_review_script"
    assert compile_cut_stage(WEBSITE_REVIEW_BLUEPRINT, spine) == "website_review_compile"


def _ctx(*, sound: dict[str, Any] | None = None, line_extra: dict[str, Any] | None = None):
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
                    {
                        "id": "L2",
                        "text": "Fix.",
                        "seconds": 4.0,
                        "phase": "after",
                        **(line_extra or {}),
                    },
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


@pytest.fixture
def stubbed_sound(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _library(*_a: object, **_k: object) -> SoundLibrary:
        return SoundLibrary.builtin()

    async def _presign(sfx_id: str) -> str:
        return f"https://example.invalid/{sfx_id}.mp3?sig=x"

    monkeypatch.setattr(sound_design, "load_sound_library", _library)
    monkeypatch.setattr(sound_design, "presign_sound_asset", _presign)


def _fake_client() -> AsyncMock:
    fake = AsyncMock()
    fake.compile.return_value = {"artifact_url": "http://stub/review.mp4", "duration_seconds": 9.0}
    return fake


@pytest.mark.asyncio
async def test_music_edit_reaches_the_compile_request_with_ducking(stubbed_sound: None) -> None:
    """The exact override apply-edits writes for ``music:{track, gain_db, fades, duck_db}``."""
    fake = _fake_client()
    sound = {
        "enabled": True,
        "music": "jazz_lounge",
        "music_gain_db": -16.0,
        "fade_in_s": 1.0,
        "fade_out_s": 2.0,
        "duck_db": -6.0,
    }
    await WebsiteReviewCompileStage(client=fake).run({}, ctx=_ctx(sound=sound))

    _, kwargs = fake.compile.call_args
    music = kwargs.get("music")
    assert music is not None, "website_review_compile sent no music bed"
    assert music["url"].startswith("https://example.invalid/jazz_lounge.mp3")
    assert music["gain_db"] == -16.0
    assert music["fade_in_s"] == 1.0
    assert music["fade_out_s"] == 2.0
    assert music["duck_db"] == -6.0


@pytest.mark.asyncio
async def test_no_sound_config_sends_no_bed(stubbed_sound: None) -> None:
    fake = _fake_client()
    await WebsiteReviewCompileStage(client=fake).run({}, ctx=_ctx())
    _, kwargs = fake.compile.call_args
    assert kwargs.get("music") is None


@pytest.mark.asyncio
async def test_per_clip_edit_fields_survive_the_spine_rebuild() -> None:
    fake = _fake_client()
    ctx = _ctx(line_extra={"caption": "burned text", "speed": 1.25, "fit": "cover"})
    await WebsiteReviewCompileStage(client=fake).run({}, ctx=ctx)

    _, kwargs = fake.compile.call_args
    shots = {s["line_id"]: s for s in kwargs["shots"]}
    assert shots["L2"]["speed"] == 1.25
    assert shots["L2"]["fit"] == "cover"
    assert shots["L2"].get("caption") == "burned text"
    # an unedited line carries none of them
    assert "speed" not in shots["L1"]
