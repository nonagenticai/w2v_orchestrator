"""NarrationStageBase.speed_for — the per-line TTS speed hook (C13 consumption).

``reconcile_timeline`` stamps a ``vo_rate`` on the lines it had to fit; narrate
turns that into the TTS ``speed``. The base hook returns ``None``, which must keep
EVERY existing narrate stage byte-identical: no ``speed`` on the wire to the TTS
backend, and the same content-addressed clip id (so cached clips still hit).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

from src.clients.schemas.tts import TTSSynthesis
from src.services.runs.run_context import RunContext
from src.services.stages.base import ArtifactUrl, StageInput, StageOutput
from src.services.stages.primitives.narration import NarrationStageBase
from src.services.tts_storage import deterministic_clip_id


class _Clip(StageOutput):
    id: str
    line_id: str
    artifact_url: ArtifactUrl
    seconds: float


class _In(StageInput):
    pass


class _Out(StageOutput):
    clips: list[_Clip]


class _PlainNarrateStage(NarrationStageBase[_In, _Out]):
    """A narrate stage that does NOT override speed_for (promo/lesson/website_review)."""

    name = "plain_narrate"
    input_model = _In
    output_model = _Out
    source_stage = "plain_gen"
    source_list_key = "lines"
    clip_model = _Clip


class _RatedNarrateStage(_PlainNarrateStage):
    """A narrate stage that DOES override speed_for (demo_video's shape)."""

    name = "rated_narrate"

    def speed_for(self, item: dict, ctx: RunContext) -> float | None:
        rate = item.get("vo_rate")
        return float(rate) if rate else None


def _tts() -> AsyncMock:
    client = AsyncMock()
    client.synthesize.return_value = TTSSynthesis(
        audio_bytes=b"\x01", sample_rate=0, duration_seconds=3.0, provider="", voice=""
    )
    return client


def _ctx(lines: list[dict]) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"voice": {"voice_id": "af_heart"}},
        prior_run_outputs={"plain_gen": {"lines": lines}},
    )


def _storage_patches():
    return (
        patch(
            "src.services.stages.primitives.narration.find_existing_clip",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "src.services.stages.primitives.narration.store_tts_clip",
            new=AsyncMock(side_effect=lambda **kw: f"https://minio.test/{kw['clip_id']}.wav"),
        ),
    )


def test_speed_for_default_is_none() -> None:
    assert (
        _PlainNarrateStage(client=_tts()).speed_for({"id": "l1", "text": "Hi."}, _ctx([])) is None
    )


async def test_base_narrate_sends_no_speed_and_keeps_the_clip_id() -> None:
    """The default hook ⇒ the TTS call carries NO `speed` kwarg and the clip id is
    the same content-address as before the hook existed (cache stays warm)."""
    client = _tts()
    stage = _PlainNarrateStage(client=client)
    finder, storer = _storage_patches()
    with finder, storer:
        out = await stage.handle(stage.input_model(), _ctx([{"id": "l1", "text": "Hello."}]))

    assert "speed" not in client.synthesize.call_args.kwargs
    assert client.synthesize.call_args.kwargs["voice"] == "af_heart"
    assert out.clips[0].id == deterministic_clip_id("Hello.", "af_heart", None)


async def test_overridden_hook_threads_speed_onto_the_tts_call() -> None:
    """A line carrying a vo_rate narrates at that speed; a line without one is
    unchanged (no `speed` kwarg at all, not `speed=1.0`)."""
    client = _tts()
    stage = _RatedNarrateStage(client=client)
    finder, storer = _storage_patches()
    with finder, storer:
        out = await stage.handle(
            stage.input_model(),
            _ctx(
                [
                    {"id": "l1", "text": "Fast line.", "vo_rate": 1.12},
                    {"id": "l2", "text": "Plain line."},
                ]
            ),
        )

    by_text = {c.kwargs["text"]: c.kwargs for c in client.synthesize.call_args_list}
    assert by_text["Fast line."]["speed"] == 1.12
    assert "speed" not in by_text["Plain line."]
    # The rated line's clip id folds the speed in; the plain one does not.
    by_line = {c.line_id: c.id for c in out.clips}
    assert by_line["l1"] == deterministic_clip_id("Fast line.", "af_heart", 1.12)
    assert by_line["l2"] == deterministic_clip_id("Plain line.", "af_heart", None)
