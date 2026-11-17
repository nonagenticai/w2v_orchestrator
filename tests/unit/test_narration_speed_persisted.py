"""NarrationStageBase persists the KEPT clip's synthesis speed + word count.

Before this, ``speed``/``word_count`` reached only the MinIO sidecar; the DB stage
output carried id/line_id/artifact_url/seconds, so a line shipped at speed 0.5
was invisible without object-storage archaeology.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from src.clients.schemas.tts import TTSSynthesis
from src.services.runs.run_context import RunContext
from src.services.stages.action_reel.narrate import ActionReelAudioClip
from src.services.stages.audit_video.models import AuditAudioClip
from src.services.stages.base import StageInput, StageOutputWithNotes
from src.services.stages.demo_video.narrate import DemoVideoAudioClip
from src.services.stages.generic.line_fit import SPEED_EXPONENT
from src.services.stages.generic.narrate import GenericAudioClip
from src.services.stages.primitives.narration import NarrationStageBase

_NATURAL_SECONDS = 10.0


class _In(StageInput):
    pass


class _Out(StageOutputWithNotes):
    clips: list[DemoVideoAudioClip]


class _Stage(NarrationStageBase[_In, _Out]):
    name = "speed_narrate"
    input_model = _In
    output_model = _Out
    source_stage = "gen"
    source_list_key = "lines"
    clip_model = DemoVideoAudioClip

    def fit_target_for(self, item: dict, ctx: RunContext) -> float | None:
        allotment = item.get("seconds")
        return float(allotment) if allotment else None

    def speed_for(self, item: dict, ctx: RunContext) -> float | None:
        rate = item.get("vo_rate")
        return float(rate) if rate else None


def _client() -> AsyncMock:
    async def _synthesize(**kwargs: Any) -> TTSSynthesis:
        speed = kwargs.get("speed")
        return TTSSynthesis(
            audio_bytes=b"\x01",
            sample_rate=24000,
            # kokoro's calibrated response (line_fit.SPEED_EXPONENT), not 1/speed.
            duration_seconds=_NATURAL_SECONDS
            * float(speed if speed is not None else 1.0) ** -SPEED_EXPONENT,
            provider="kokoro",
            voice="af_heart",
        )

    client = AsyncMock()
    client.synthesize.side_effect = _synthesize
    return client


async def _run(lines: list[dict], *, existing: tuple[str, float] | None = None) -> _Out:
    stage = _Stage(client=_client())
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"voice": {"voice_id": "af_heart"}},
        prior_run_outputs={"gen": {"lines": lines}},
    )
    with (
        patch(
            "src.services.stages.primitives.narration.find_existing_clip",
            new=AsyncMock(return_value=existing),
        ),
        patch(
            "src.services.stages.primitives.narration.store_tts_clip",
            new=AsyncMock(side_effect=lambda **kw: f"https://minio.test/{kw['clip_id']}.wav"),
        ),
    ):
        return await stage.handle(_In(), ctx)


async def test_requested_speed_and_words_are_persisted() -> None:
    out = await _run([{"id": "l1", "text": "One two three four.", "vo_rate": 0.8}])
    clip = out.clips[0]
    assert clip.speed == 0.8
    assert clip.words == 4
    assert "speed" in clip.model_dump()


async def test_backend_default_speed_persists_as_none_not_one() -> None:
    out = await _run([{"id": "l1", "text": "One two three."}])
    assert out.clips[0].speed is None
    assert out.clips[0].words == 3


async def test_fit_resynth_persists_the_corrected_speed() -> None:
    # 10.0s at the backend default vs an 8.0s allotment ⇒ re-synth at 1.25 ** (1/k).
    out = await _run([{"id": "l1", "text": "One two.", "seconds": 8.0}])
    clip = out.clips[0]
    assert clip.seconds == pytest.approx(8.0)
    assert clip.speed == pytest.approx(1.25 ** (1 / SPEED_EXPONENT))
    assert clip.words == 2


async def test_reused_clip_reports_the_requested_speed() -> None:
    out = await _run(
        [{"id": "l1", "text": "One two three.", "vo_rate": 0.7}],
        existing=("https://minio.test/reused.wav", 14.2857),
    )
    clip = out.clips[0]
    assert clip.artifact_url == "https://minio.test/reused.wav"
    assert clip.speed == 0.7
    assert clip.words == 3


def test_every_owned_clip_model_accepts_legacy_and_new_shapes() -> None:
    legacy = {"id": "c", "line_id": "l", "artifact_url": "https://x/c.wav", "seconds": 1.0}
    for model in (DemoVideoAudioClip, ActionReelAudioClip, AuditAudioClip, GenericAudioClip):
        old = model.model_validate(legacy)
        assert old.speed is None and old.words is None
        new = model.model_validate({**legacy, "speed": 0.5, "words": 7})
        assert (new.speed, new.words) == (0.5, 7)
