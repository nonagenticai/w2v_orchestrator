"""The RAW mastering report must reach EVERY render blueprint — O7.

The mastering GATE (its prose notes and its structured degradations) has been
shared since ``shared/mastering_gate.py``. The NUMBERS BEHIND the verdict were
not: ``mastering`` / ``output_sha256`` / ``loudness`` were populated only in
``demo_video/compile.py``'s own ``build_output``. So on lesson, action_reel,
app_review and website_review a downstream reader could see *that* the gate
failed but never *by how much* — measured loudness, true peak and colour tags
were simply absent from the persisted output — and ``final_critique`` could not
pass-read ``loudness`` and would have had to re-measure it.

The fields now live on the shared carrier (``base.py::MasteringReportFields``,
inherited by ``StageOutputWithNotes``) and are populated centrally in
``CompileStageBase.handle`` through the same
``"<name>" in type(output).model_fields`` guard that already carries notes and
degradations. These tests assert the wiring ON THE CALL — a stage whose model
has the field but whose handle never populates it looks identical to a compiler
that returned nothing.
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

# A healthy post-mux report: the gate PASSED, so nothing here degrades the run.
# That is deliberate — the raw report must be carried on a CLEAN master too,
# otherwise the only runs with readable numbers are the broken ones.
_HEALTHY_MASTERING = {
    "rerun_stages": [],
    "gate_pass": True,
    "luma_spread": 210.0,
    "integrated_lufs": -14.02,
    "true_peak_dbtp": -1.1,
    "colorspace": "bt709",
}
_LOUDNESS = {"integrated_lufs": -14.02, "true_peak_dbtp": -1.1, "lra": 6.4}
_SHA256 = "9f2b" + "0" * 60


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

    name = "test_compile_report"
    input_model = _Input
    output_model = _Output

    def build_shots(self, ctx: Any) -> list[dict[str, Any]]:
        return [{"id": "s1", "line_id": "l1", "seconds": 1.0}]

    def build_output(self, result: dict[str, Any], ctx: Any) -> _Output:
        # Populates NOTHING from the mastering report — exactly like every
        # non-demo_video blueprint's build_output, which is why the carry had to
        # move onto the shared base.
        return _Output(
            artifact_url=str(result["artifact_url"]),
            duration_seconds=float(result["duration_seconds"]),
        )

    def compile_tags(self, ctx: Any) -> dict[str, str]:
        return {}


def _client(extra: dict[str, Any]) -> AsyncMock:
    async def _compile(**_kwargs: Any) -> dict[str, Any]:
        return {
            "artifact_url": "http://m/w2v-compiled-videos/out.mp4",
            "duration_seconds": 1.0,
            "actual_duration_seconds": 1.0,
            **extra,
        }

    client = AsyncMock()
    client.compile = AsyncMock(side_effect=_compile)
    return client


async def _run(extra: dict[str, Any], stage_cls: type[Any] = _Stage) -> Any:
    stage = stage_cls(client=cast(Any, _client(extra)))
    ctx = RunContext(run_id=uuid7(), deliverable_id=uuid7(), config_snapshot={})
    tts_storage.set_storage_for_tests(cast(Any, _FakeStorage()))
    try:
        return await stage.handle(_Input(), ctx=ctx)
    finally:
        tts_storage.set_storage_for_tests(None)


@pytest.mark.asyncio
async def test_the_raw_report_reaches_a_non_demo_video_blueprint() -> None:
    """THE point of O7: this stage is not demo_video and never was.

    Before the fix all three of these came back ``None`` on every blueprint but
    demo_video, no matter what the compiler reported.
    """
    out = await _run(
        {"mastering": _HEALTHY_MASTERING, "output_sha256": _SHA256, "loudness": _LOUDNESS}
    )
    assert out.mastering == _HEALTHY_MASTERING
    assert out.output_sha256 == _SHA256
    assert out.loudness == _LOUDNESS


@pytest.mark.asyncio
async def test_the_report_is_carried_on_a_clean_master_too() -> None:
    """The gate passed and the run is not degraded — the numbers still ride.

    If the carry were conditional on a failed gate, the only runs with readable
    loudness would be the broken ones, and ``final_critique`` (which pass-reads
    ``loudness`` on every run) would be back to re-measuring.
    """
    out = await _run(
        {"mastering": _HEALTHY_MASTERING, "output_sha256": _SHA256, "loudness": _LOUDNESS}
    )
    assert out.notes == [], out.notes
    assert out.degradations == []
    assert out.loudness is not None


@pytest.mark.asyncio
async def test_an_older_compiler_build_yields_none_not_a_crash() -> None:
    """All three fields are optional: a build that emits none of them must not
    fail the stage, and must not fabricate an empty report either."""
    out = await _run({"mastering": _HEALTHY_MASTERING})
    assert out.mastering == _HEALTHY_MASTERING
    assert out.output_sha256 is None
    assert out.loudness is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "junk"),
    [pytest.param("mastering", {}, id="mastering_empty_dict")],
)
async def test_an_unusable_value_becomes_none_rather_than_riding(field: str, junk: object) -> None:
    """The compiler's response is untyped JSON. A non-dict, an empty dict or a
    blank hash is NOT a report — carrying it would make "the compiler measured
    nothing" indistinguishable from "the compiler measured and returned this",
    which is this codebase's #1 defect class."""
    out = await _run({field: junk})
    assert getattr(out, field) is None


def test_every_compile_blueprint_can_carry_the_raw_report() -> None:
    """A field the output model cannot hold is a field nobody ever sees.

    ``handle`` populates only when the model HAS the field, so this asserts the
    precondition for the report reaching each blueprint rather than trusting
    inheritance. Same walk as the sibling notes test — see
    ``test_mastering_gate_reaches_every_blueprint.py``.
    """
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

    # Only PRODUCTION stages — sibling test modules define their own subclasses,
    # so an unfiltered walk gives a different answer depending on import order.
    checked = [
        s
        for s in subclasses
        if getattr(s, "output_model", None) is not None
        and s.__module__.startswith("src.services.stages.")
    ]
    assert len(checked) >= 8, f"expected the fleet of compile stages, found {len(checked)}"
    for field in ("mastering", "output_sha256", "loudness"):
        missing = [s.__name__ for s in checked if field not in s.output_model.model_fields]
        assert not missing, f"these compile stages cannot carry {field!r}: {missing}"
