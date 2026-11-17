"""The compiler's mastering gate must degrade EVERY render blueprint — M3a.

The gate lived in ``demo_video/compile.py``, so a blank/dead delivered master
degraded exactly one of the nine products that subclass ``CompileStageBase``.
lesson, website_review, generic/video_render, action_reel, promo, app_review and
audit_video (plus its seo_audit / cro_audit re-declarations) shipped the same
blank file with a clean ``completed`` and no note at all.

The gate is a property of the COMPILER'S RESPONSE, not of demo_video, so it now
runs in the shared primitive. These tests assert the wiring on the CALL — a stage
that computes a note and never attaches it looks identical to a green run.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import uuid7

import pytest

from src.services import tts_storage
from src.services.runs.run_context import RunContext
from src.services.stages.base import ArtifactUrl, StageInput, StageOutputWithNotes
from src.services.stages.primitives.compile_stage import CompileStageBase

_BLANK_MASTER = {
    # The blank-master signature from run 01a00b81-2576-71a5-becc-ffc4a843e6db:
    # a stage the compiler wants re-run, a failed gate, and a dead_frame_ratio over
    # the compiler's 0.10 cutoff (luma_spread=0.0 alone is NOT blank: one flat frame
    # zeroes that min-over-frames metric — run 01a08f66).
    "rerun_stages": ["deband"],
    "gate_pass": False,
    "dead_frame_ratio": 0.25,
    "luma_spread": 0.0,
    "banding_scores": [1.0, 1.0],
    "duplicate_frame_ratio": 0.338,
}


class _FakeStorage:
    async def resign_presigned_url(self, url: str) -> str:
        return url


class _Input(StageInput):
    pass


class _Output(StageOutputWithNotes):
    artifact_url: ArtifactUrl
    duration_seconds: float


class _Stage(CompileStageBase[_Input, _Output]):
    """A bare NON-demo_video product — the base's own behaviour, nothing else."""

    name = "test_compile"
    input_model = _Input
    output_model = _Output

    def build_shots(self, ctx: Any) -> list[dict[str, Any]]:
        return [{"id": "s1", "line_id": "l1", "seconds": 1.0}]

    def build_output(self, result: dict[str, Any], ctx: Any) -> _Output:
        # Deliberately attaches NO notes of its own — every other blueprint's
        # build_output looks like this, which is why the gate had to move.
        return _Output(
            artifact_url=str(result["artifact_url"]),
            duration_seconds=float(result["duration_seconds"]),
        )

    def compile_tags(self, ctx: Any) -> dict[str, str]:
        return {}


def _client(mastering: dict[str, Any] | None) -> AsyncMock:
    async def _compile(**_kwargs: Any) -> dict[str, Any]:
        result: dict[str, Any] = {
            "artifact_url": "http://m/w2v-compiled-videos/out.mp4",
            "duration_seconds": 1.0,
            "actual_duration_seconds": 1.0,
        }
        if mastering is not None:
            result["mastering"] = mastering
        return result

    client = AsyncMock()
    client.compile = AsyncMock(side_effect=_compile)
    return client


async def _run(mastering: dict[str, Any] | None) -> _Output:
    stage = _Stage(client=cast(Any, _client(mastering)))
    ctx = RunContext(run_id=uuid7(), deliverable_id=uuid7(), config_snapshot={})
    tts_storage.set_storage_for_tests(cast(Any, _FakeStorage()))
    try:
        return await stage.handle(_Input(), ctx=ctx)
    finally:
        tts_storage.set_storage_for_tests(None)


@pytest.mark.asyncio
async def test_a_blank_master_degrades_a_non_demo_video_blueprint() -> None:
    """THE point of M3a: this stage is not demo_video and never was."""
    out = await _run(_BLANK_MASTER)
    assert out.artifact_url  # degraded, not failed — the artifact still exists
    assert len(out.notes) == 1
    note = out.notes[0]
    assert "deband" in note  # names the actual rerun stage
    assert "dead_frame_ratio=0.25 (cutoff 0.1)" in note  # names the blank-master metric


@pytest.mark.asyncio
async def test_an_absent_mastering_report_is_reported_as_unmeasured() -> None:
    """CORRECTED: this asserted ``out.notes == []`` and so PINNED the fail-open.

    A compiler that returns no mastering report at all (an older build, or its own
    post-mux verify pass failed) produced a run byte-identical to one whose gate ran
    and PASSED — the deliverable was reported as mastered with nothing having checked
    it. The gap now reaches the run through the same DEGRADE notes channel, worded so
    a reviewer can tell "not measured" from "measured and fine".
    """
    out = await _run(None)
    assert len(out.notes) == 1, out.notes
    assert "UNMEASURED" in out.notes[0]
    assert "gate FAILED" not in out.notes[0]


def test_every_compile_blueprint_can_carry_a_note() -> None:
    """A note the output model cannot hold is a note nobody ever sees.

    ``CompileStageBase.handle`` attaches notes only when the output model has the
    field, so this asserts the precondition for the gate reaching each blueprint
    rather than trusting inheritance.
    """
    import importlib
    import pkgutil

    import src.services.blueprints as blueprints

    # Import every blueprint module so its compile stage class is defined and
    # therefore visible to __subclasses__.
    for mod in pkgutil.iter_modules(blueprints.__path__):
        importlib.import_module(f"{blueprints.__name__}.{mod.name}")

    subclasses: list[type[Any]] = []
    pending: list[type[Any]] = [CompileStageBase]
    while pending:
        for sub in pending.pop().__subclasses__():
            subclasses.append(sub)
            pending.append(sub)

    # Only PRODUCTION stages. Sibling test modules define their own
    # CompileStageBase subclasses, so an unfiltered __subclasses__ walk gives a
    # different answer depending on which files pytest already imported — the
    # count differs between running this file alone and running the suite.
    checked = [
        s
        for s in subclasses
        if getattr(s, "output_model", None) is not None
        and s.__module__.startswith("src.services.stages.")
    ]
    assert len(checked) >= 8, f"expected the fleet of compile stages, found {len(checked)}"
    missing = [s.__name__ for s in checked if "notes" not in s.output_model.model_fields]
    assert not missing, f"these compile stages cannot carry a mastering note: {missing}"
