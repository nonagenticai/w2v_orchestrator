"""The compile stage must RECORD ken_burns clips at the cadence it RESOLVED.

``apply_ken_burns`` takes a ``master_fps`` override (O13) and falls back to a
config-only resolution when it is absent. That fallback is correct for every
config-driven product and WRONG for exactly one: ``resolve_cadence`` is a HOOK,
and ``action_reel`` overrides it with ``footage_probe``'s measured source rates
(a 23.976 drone corpus). So unless ``CompileStageBase`` passes its resolved
cadence through, the one product that deliberately computes its own master rate
is the one product that records at 30 while its master conforms to something
else.

The consequence is not a cosmetic mismatch. A ken_burns clip reaches the
compiler with NO ``Shot.target_fps`` (nothing passes
``build_shots(target_fps=...)``), so ``_cadence_fragment`` sees
``source_fps=None``, believes the ratio is 1.0, and emits a bare DECIMATING
``fps=<master>`` with the interpolation branch structurally unreachable — every
fifth frame of a smooth pan dropped on a 24p master.

⭐ These assert the fps the RECORDER is actually handed, through the real
``apply_ken_burns``, not that ``handle`` passes a particular keyword. A test that
asserts the call's shape survives the call site being deleted.
"""

from __future__ import annotations

import io
from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import uuid7

import pytest
from PIL import Image

from src.services import tts_storage
from src.services.runs.run_context import RunContext
from src.services.stages.base import ArtifactUrl, StageInput, StageOutputWithNotes
from src.services.stages.primitives.compile_stage import CompileStageBase
from src.services.stages.shared import ken_burns

# The config says 30 and the FOOTAGE says 24. Every assertion below depends on
# those disagreeing: if the snapshot carried the same rate the hook returns, a
# stage that dropped the override entirely would still look correct.
_CONFIG_FPS = 30
_FOOTAGE_FPS = 24


def _png(w: int, h: int) -> bytes:
    """A genuinely detailed still — flat colour is unresampleable."""
    img = Image.new("RGB", (w, h))
    px = img.load()
    assert px is not None
    for y in range(h):
        for x in range(w):
            v = 255 if (x // 3 + y // 3) % 2 else 0
            px[x, y] = (v, (x * 7) % 256, (y * 11) % 256)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class _FakeStorage:
    async def initialize(self) -> None:
        pass

    async def resign_presigned_url(self, url: str) -> str:
        return url

    async def store_stage_artifact(
        self, job_id: str, stage_name: str, data: bytes, filename: str
    ) -> str:
        return f"artifacts/{job_id}/{stage_name}/{filename}"

    async def get_presigned_url(
        self, path: str, expires: int | None = None, bucket: str | None = None
    ) -> str | None:
        return f"https://minio.test/{path}"


class _Input(StageInput):
    pass


class _Output(StageOutputWithNotes):
    artifact_url: ArtifactUrl
    duration_seconds: float


class _ConfigDrivenStage(CompileStageBase[_Input, _Output]):
    """The base's own behaviour: cadence comes from the config."""

    name = "test_cadence_config"
    input_model = _Input
    output_model = _Output

    def ken_burns_local_only(self, ctx: Any) -> bool:
        # Pins the video_production RECORDER path, which the base no longer takes.
        return False

    def build_shots(self, ctx: Any) -> list[dict[str, Any]]:
        # A bare STILL, which is what ken_burns upgrades.
        return [
            {
                "id": "s-a",
                "line_id": "a",
                "image_url": "https://minio.test/a.png",
                "seconds": 3.0,
            }
        ]

    def build_output(self, result: dict[str, Any], ctx: Any) -> _Output:
        return _Output(
            artifact_url=str(result["artifact_url"]),
            duration_seconds=float(result["duration_seconds"]),
        )

    def compile_tags(self, ctx: Any) -> dict[str, str]:
        return {}


class _FootageDrivenStage(_ConfigDrivenStage):
    """``action_reel``'s shape: the master rate is a property of the FOOTAGE.

    Deliberately does NOT read the config — that is the whole point of the hook,
    and the real override reads ``ctx.prior_output("footage_probe")``.
    """

    name = "test_cadence_footage"

    def resolve_cadence(self, ctx: RunContext) -> int | None:
        return _FOOTAGE_FPS


def _video_fake() -> AsyncMock:
    v = AsyncMock()
    v.record_animation.return_value = "kb-1"
    v.wait_for_content_recording.return_value = {"duration_seconds": 3.0}
    v.get_recording_video.return_value = (b"\x00mp4", "video/mp4", None)
    return v


def _compile_client() -> AsyncMock:
    async def _compile(**_kwargs: Any) -> dict[str, Any]:
        return {
            "artifact_url": "http://m/w2v-compiled-videos/out.mp4",
            "duration_seconds": 3.0,
            "actual_duration_seconds": 3.0,
            "mastering": {"rerun_stages": [], "gate_pass": True, "luma_spread": 210.0},
        }

    client = AsyncMock()
    client.compile = AsyncMock(side_effect=_compile)
    return client


async def _recorded_fps(stage_cls: type[Any], monkeypatch: pytest.MonkeyPatch) -> int:
    """Drive ``handle`` for real and report the fps the RECORDER was handed."""
    video = _video_fake()
    monkeypatch.setattr(ken_burns, "VideoProductionClient", lambda *a, **k: video)

    async def _fake_fetch(url: str, client: Any) -> tuple[bytes, str]:
        return (_png(64, 36), "image/png")

    monkeypatch.setattr(ken_burns, "_fetch_own_asset", _fake_fetch)

    stage = stage_cls(client=cast(Any, _compile_client()))
    ctx = RunContext(
        run_id=uuid7(),
        deliverable_id=uuid7(),
        config_snapshot={"cadence": {"target_fps": _CONFIG_FPS}},
    )
    tts_storage.set_storage_for_tests(cast(Any, _FakeStorage()))
    try:
        await stage.handle(_Input(), ctx=ctx)
    finally:
        tts_storage.set_storage_for_tests(None)

    assert video.record_animation.await_args is not None, (
        "the recorder was never called — this harness no longer exercises the "
        "ken_burns path, so every assertion on it is vacuous"
    )
    return int(video.record_animation.await_args.kwargs["fps"])


@pytest.mark.asyncio
async def test_a_footage_driven_product_records_at_its_footage_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⭐ THE regression. Before the wiring this recorded at 30 — the config rate —
    while the master conformed to the footage-derived 24, so the compiler
    decimated every pan it was handed."""
    assert await _recorded_fps(_FootageDrivenStage, monkeypatch) == _FOOTAGE_FPS


@pytest.mark.asyncio
async def test_a_config_driven_product_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No behaviour change for the products whose cadence IS their config — the
    hook returns what ``_master_fps`` would have resolved anyway, so these clips
    are byte-identical (and `fps` folds into the media id only when != 30, so no
    cached clip in the estate is invalidated)."""
    assert await _recorded_fps(_ConfigDrivenStage, monkeypatch) == _CONFIG_FPS


@pytest.mark.asyncio
async def test_the_recorded_rate_is_the_one_sent_to_the_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The conform must be a NO-OP: whatever cadence the compile request carries
    is the cadence the clip was recorded at.

    This is the property that actually matters — asserting the recorded fps alone
    would still pass if the two resolutions drifted apart, which is precisely the
    defect (a value computed correctly upstream and discarded at a handoff).
    """
    video = _video_fake()
    monkeypatch.setattr(ken_burns, "VideoProductionClient", lambda *a, **k: video)

    async def _fake_fetch(url: str, client: Any) -> tuple[bytes, str]:
        return (_png(64, 36), "image/png")

    monkeypatch.setattr(ken_burns, "_fetch_own_asset", _fake_fetch)

    compile_client = _compile_client()
    stage = _FootageDrivenStage(client=cast(Any, compile_client))
    ctx = RunContext(
        run_id=uuid7(),
        deliverable_id=uuid7(),
        config_snapshot={"cadence": {"target_fps": _CONFIG_FPS}},
    )
    tts_storage.set_storage_for_tests(cast(Any, _FakeStorage()))
    try:
        await stage.handle(_Input(), ctx=ctx)
    finally:
        tts_storage.set_storage_for_tests(None)

    recorded = int(video.record_animation.await_args.kwargs["fps"])
    sent = compile_client.compile.await_args.kwargs["cadence"]
    assert recorded == sent == _FOOTAGE_FPS, (
        f"recorded at {recorded}fps but the master conforms to {sent}fps — "
        "the compiler will resample a continuous pan"
    )
