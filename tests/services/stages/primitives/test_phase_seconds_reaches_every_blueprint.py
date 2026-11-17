"""Per-phase compile timing must reach EVERY render blueprint's persisted output.

``phase_tracker.phase(...)`` has always LOGGED elapsed time per named phase
(``title_cards``/``presign_1``/``ken_burns``/``presign_2``/``sound``/
``audio_bed``/``captions``/``stage_call``) — but never persisted it anywhere a
run record can be queried. Establishing that ``lesson_compile`` went from 1.59
to 23.6 compile-seconds per delivered output-second between June and September
required reconstructing timings from job tables a pruner deletes, instead of
reading ``stage_outputs``.

``base.py::PhaseTimingFields`` (``phase_seconds`` + ``ken_burns_shot_count``)
closes that gap through the same central-carry pattern
``MasteringReportFields`` already uses: populated once in
``CompileStageBase.handle`` behind a
``"<name>" in type(output).model_fields`` guard, so it reaches every
blueprint without a per-stage change. These tests assert the wiring ON THE
CALL, mirroring ``test_mastering_report_reaches_every_blueprint.py`` — a stage
whose output model merely HAS the field but whose handle never populates it
looks identical to a compile that never ran.
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
    async def resign_presigned_url(self, url: str) -> str:
        return url


class _Input(StageInput):
    pass


class _Output(StageOutputWithNotes):
    artifact_url: ArtifactUrl
    duration_seconds: float


def _ctx() -> Any:
    return SimpleNamespace(
        run_id=uuid7(),
        render_idempotency_discriminator="",
        config_snapshot={},
        prior_output=lambda _stage: None,
    )


class _FootageStage(CompileStageBase[_Input, _Output]):
    """No stills at all — the discriminator this instrumentation exists to
    answer is "how many stills needed a Ken Burns job", and a footage-only
    render is the population where that answer is legitimately zero."""

    name = "test_phase_seconds_footage"
    input_model = _Input
    output_model = _Output

    def build_shots(self, ctx: Any) -> list[dict[str, Any]]:
        return [{"id": "s1", "line_id": "l1", "seconds": 4.0, "video_url": "https://x/clip.mp4"}]

    def build_output(self, result: dict[str, Any], ctx: Any) -> _Output:
        return _Output(
            artifact_url=str(result["artifact_url"]),
            duration_seconds=float(result["duration_seconds"]),
        )

    def compile_tags(self, ctx: Any) -> dict[str, str]:
        return {}


class _StillsStage(_FootageStage):
    """Two bare stills — the ken_burns candidate set."""

    name = "test_phase_seconds_stills"

    def build_shots(self, ctx: Any) -> list[dict[str, Any]]:
        return [
            {"id": "s1", "line_id": "l1", "seconds": 4.0, "image_url": _STILL_URL},
            {"id": "s2", "line_id": "l2", "seconds": 4.0, "image_url": _STILL_URL},
        ]


def _client() -> AsyncMock:
    async def _compile(**_kwargs: Any) -> dict[str, Any]:
        return {"artifact_url": "http://m/w2v-compiled-videos/out.mp4", "duration_seconds": 8.0}

    client = AsyncMock()
    client.compile = AsyncMock(side_effect=_compile)
    return client


async def _run(stage_cls: type[Any], monkeypatch: pytest.MonkeyPatch) -> Any:
    # Soft-fail the ken_burns fetch seam so the still stays a still and no real
    # network call happens — same seam ``test_compile_stage_ken_burns_notes.py``
    # patches. Irrelevant to a stage with no stills, harmless to patch anyway.
    async def _boom(*_a: Any, **_kw: Any) -> tuple[bytes, str]:
        raise RuntimeError("no network in this test")

    monkeypatch.setattr("src.services.stages.shared.ken_burns._fetch_own_asset", _boom)

    async def _no_cache(*_a: Any, **_kw: Any) -> str | None:
        return None

    monkeypatch.setattr("src.services.stages.shared.ken_burns.find_existing_media", _no_cache)

    stage = stage_cls(client=cast(Any, _client()))
    tts_storage.set_storage_for_tests(cast(Any, _FakeStorage()))
    try:
        return await stage.handle(_Input(), ctx=_ctx())
    finally:
        tts_storage.set_storage_for_tests(None)


@pytest.mark.asyncio
async def test_phase_seconds_reaches_the_output_with_every_phase_that_ran(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = await _run(_FootageStage, monkeypatch)
    assert out.phase_seconds is not None, "phase_seconds never reached the persisted output"
    # Every phase this base ALWAYS opens for a footage render (title_cards is
    # gated on the hook, which this stage never overrides, so it still opens).
    for name in (
        "title_cards",
        "presign_1",
        "ken_burns",
        "presign_2",
        "sound",
        "audio_bed",
        "captions",
        "stage_call",
    ):
        assert name in out.phase_seconds, f"phase {name!r} missing from {out.phase_seconds}"
        assert out.phase_seconds[name] >= 0.0


def test_every_compile_blueprint_can_carry_phase_seconds() -> None:
    """A field the output model cannot hold is a field nobody ever sees. Same
    walk as ``test_mastering_report_reaches_every_blueprint.py`` —
    ``handle`` populates only when the model HAS the field, so this asserts
    the precondition for every blueprint rather than trusting inheritance."""
    import importlib
    import pkgutil

    import src.services.blueprints as blueprints

    for mod in pkgutil.iter_modules(blueprints.__path__):
        importlib.import_module(f"{blueprints.__name__}.{mod.name}")

    subclasses: list[type[Any]] = []
    pending: list[type[Any]] = [CompileStageBase]
    while pending:
        for sub in pending.pop().__subclasses__():
            subclasses.append(sub)
            pending.append(sub)

    checked = [
        s
        for s in subclasses
        if getattr(s, "output_model", None) is not None
        and s.__module__.startswith("src.services.stages.")
    ]
    assert len(checked) >= 8, f"expected the fleet of compile stages, found {len(checked)}"
    for field in ("phase_seconds", "ken_burns_shot_count"):
        missing = [s.__name__ for s in checked if field not in s.output_model.model_fields]
        assert not missing, f"these compile stages cannot carry {field!r}: {missing}"
