"""The master frame rate that reaches the compile request.

``CompileRequest.cadence`` is the master rate every source is conformed to. Get
it wrong and nothing fails: the render succeeds, the artifact exists, and every
pan carries a duplicate-frame judder (a 23.976 source under a 30 master is
resampled 5:4). So the wiring is asserted on the CALL, not on the resolver — a
hook that resolves 24 and is never invoked looks identical to a green run.

Two halves, and the second is the one that matters:

* ``ActionReelCompileStage`` folds ``footage_probe``'s MEASURED rates in, so a
  drone shoot renders on a 24 master;
* ``CompileStageBase`` — which serves every other product — resolves from the run
  config ONLY, exactly as it did before the hook existed. A base that started
  reading footage would change demo_video/promo/audiobook renders, which is a
  regression regardless of how right the action_reel half is.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import uuid7

import pytest

from src.services import tts_storage
from src.services.runs.run_context import RunContext
from src.services.stages.base import ArtifactUrl, StageInput, StageOutput
from src.services.stages.primitives.compile_stage import CompileStageBase

_URL = "https://example.invalid/drone-a.mp4"

#: 24000/1001 — what the drone corpus actually reports. Deliberately the raw
#: fractional value, not 24: the whole point of the reconciler is that IT snaps
#: to the legal broadcast set, so a test that pre-rounds proves nothing.
_NTSC_24 = 23.976023976023978


class _FakeStorage:
    """Enough storage for the compile tail's re-sign pass."""

    async def resign_presigned_url(self, url: str) -> str:
        return url


class _Input(StageInput):
    pass


class _Output(StageOutput):
    artifact_url: ArtifactUrl
    duration_seconds: float


class _Stage(CompileStageBase[_Input, _Output]):
    """A bare non-action_reel product: the base's own behaviour, nothing else."""

    name = "test_compile"
    input_model = _Input
    output_model = _Output

    def build_shots(self, ctx: Any) -> list[dict[str, Any]]:
        return [{"id": "s1", "line_id": "l1", "seconds": 1.0}]

    def build_output(self, result: dict[str, Any], ctx: Any) -> _Output:
        return _Output(
            artifact_url=str(result["artifact_url"]),
            duration_seconds=float(result["duration_seconds"]),
        )

    def compile_tags(self, ctx: Any) -> dict[str, str]:
        return {}  # keep this unit off ctx.config_snapshot's deliverable block


def _capturing_client(captured: dict[str, Any]) -> AsyncMock:
    """A video-compile client that records the submitted kwargs and returns a
    minimal valid result. ``cadence`` is read off the captured kwargs rather than
    a positional, which is how the real client receives it."""

    async def _compile(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"artifact_url": "http://m/w2v-compiled-videos/out.mp4", "duration_seconds": 2.0}

    client = AsyncMock()
    client.compile = AsyncMock(side_effect=_compile)
    return client


async def _run(stage: CompileStageBase[Any, Any], payload: Any, ctx: Any) -> None:
    """Drive the compile tail with the storage singleton stubbed out."""
    tts_storage.set_storage_for_tests(cast(Any, _FakeStorage()))
    try:
        await stage.handle(payload, ctx=ctx)
    finally:
        tts_storage.set_storage_for_tests(None)


# --- action_reel: the master rate comes from the FOOTAGE ---------------------


def _reel_ctx(
    *, source_fps: list[float], config_snapshot: dict[str, Any] | None = None
) -> RunContext:
    """A reel run whose footage_probe measured *source_fps*, one entry per clip."""
    return RunContext(
        run_id=uuid7(),
        deliverable_id=uuid7(),
        config_snapshot=config_snapshot if config_snapshot is not None else {},
        prior_run_outputs={
            "footage_probe": {
                "sources": [{"id": f"src-{i}", "fps": fps} for i, fps in enumerate(source_fps)],
            },
            "cut_plan": {
                "shots": [
                    {
                        "id": "s0",
                        "source_id": "src-0",
                        "video_url": _URL,
                        "in_seconds": 1.0,
                        "out_seconds": 3.0,
                        "seconds": 2.0,
                        "speed": 1.0,
                    }
                ],
                "total_seconds": 2.0,
                # No bed: keeps this test off the sound-catalog presign path, which
                # has nothing to do with cadence.
                "music_track_id": "",
                "bpm": 120.0,
                "beat_grid_available": True,
            },
            "graphics_design": {"cards": []},
        },
    )


# --- the base: unchanged for every other product ----------------------------


@pytest.mark.asyncio
async def test_base_sends_no_cadence_when_nothing_asked_for_one() -> None:
    """THE regression guard. Presence-is-opt-in: a run with no cadence intent must
    still send no cadence key, so the request stays byte-identical to a
    pre-cadence compile for demo_video/promo/audiobook/lesson."""
    captured: dict[str, Any] = {}
    stage = _Stage(_capturing_client(captured))

    await _run(
        stage,
        _Input(),
        SimpleNamespace(
            run_id=uuid7(),
            render_idempotency_discriminator="",
            config_snapshot={},
            prior_output=lambda _stage: None,
        ),
    )

    assert captured["cadence"] is None
