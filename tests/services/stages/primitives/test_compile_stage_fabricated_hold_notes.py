"""A shot whose duration was FABRICATED must not reach the user as a measured one.

``assemble_shots`` floors a reconciled ``0.0`` to a 1-second hold, and that floor
is correct: ``video_compiler`` declares ``Shot.seconds`` as ``gt=0``, so emitting
the zero would 422 the whole compile at the terminal, already-paid stage. The
defect is that the substitution was SILENT — a line whose real narration length
could not be measured rendered as a confident 1-second shot, byte-identical to a
line that genuinely asked for one second, with nothing recorded anywhere.

That is a fail-open on a measurement gap: the emitted ``seconds`` is a claim the
code cannot support. This pins the gap as EXPRESSIBLE and LOUD through the
existing soft-degradation channel (``StageOutputWithNotes.notes`` →
``api/routes/runs/_shared.py`` → a ``degraded`` run) while the compile still
succeeds.

Asserted on the RETURNED OUTPUT and on the payload the compiler actually
received, never on the notes sink and never on a call count: a helper filling a
list it was handed proves nothing about whether ``_run_presubmit`` drains it and
``handle`` folds it into the output that is PERSISTED.
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
from src.services.stages.shot_assembly import assemble_shots


class _FakeStorage:
    async def resign_presigned_url(self, url: str) -> str:
        return url


class _Input(StageInput):
    pass


class _NotesOutput(StageOutputWithNotes):
    artifact_url: ArtifactUrl
    duration_seconds: float


class _AssembledStage(CompileStageBase[_Input, _NotesOutput]):
    """A product that builds its shots through the shared ``assemble_shots``."""

    name = "test_compile_fabricated_hold"
    input_model = _Input
    output_model = _NotesOutput

    def __init__(
        self,
        client: Any,
        rows: list[dict[str, Any]],
        clips: list[dict[str, Any]],
    ) -> None:
        super().__init__(client)
        self._rows = rows
        self._clips = clips

    def build_shots(self, ctx: Any) -> list[dict[str, Any]]:
        return assemble_shots(
            storyboard_shots=self._rows,
            images=[],
            narration_clips=self._clips,
        )

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


async def _run(rows: list[dict[str, Any]], clips: list[dict[str, Any]]) -> tuple[Any, dict]:
    captured: dict[str, Any] = {}
    stage = _AssembledStage(_capturing_client(captured), rows, clips)
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
    return output, captured


@pytest.mark.asyncio
async def test_a_fabricated_hold_reaches_the_persisted_stage_output() -> None:
    """THE wiring: an unmeasurable line still compiles, and says so."""
    output, captured = await _run(
        rows=[
            {"id": "s1", "line_id": "l1", "seconds": 0.0, "description": "a"},
            {"id": "s2", "line_id": "l2", "seconds": 3.0, "description": "b"},
        ],
        clips=[],
    )

    # The floor is intact — the compile still runs, and `Shot.seconds > 0` holds.
    by_id = {str(s["id"]): s for s in captured["shots"]}
    assert by_id["s1"]["seconds"] == 1.0
    assert by_id["s2"]["seconds"] == 3.0
    # ...and no private marker leaked onto the wire payload.
    assert not [k for k in by_id["s1"] if k.startswith("_")], by_id["s1"]

    fabricated = [note for note in output.notes if "FABRICATED" in note]
    assert fabricated, f"the fabricated hold never reached the output: {output.notes}"
    # It must name WHICH line, or the note cannot be acted on.
    assert "l1" in fabricated[0], fabricated[0]
    assert "l2" not in fabricated[0], fabricated[0]


@pytest.mark.asyncio
async def test_a_line_that_genuinely_asked_for_one_second_is_not_flagged() -> None:
    """The control. A real 1.0s storyboard row is MEASURED, not fabricated."""
    output, captured = await _run(
        rows=[{"id": "s1", "line_id": "l1", "seconds": 1.0, "description": "a"}],
        clips=[],
    )
    assert captured["shots"][0]["seconds"] == 1.0
    assert not [n for n in output.notes if "FABRICATED" in n], output.notes


@pytest.mark.asyncio
async def test_a_narrated_line_is_measured_and_not_flagged() -> None:
    """The other control: a zero-second storyboard row that narration DID measure."""
    output, captured = await _run(
        rows=[{"id": "s1", "line_id": "l1", "seconds": 0.0, "description": "a"}],
        clips=[{"line_id": "l1", "artifact_url": "http://a/1.wav", "seconds": 7.5}],
    )
    assert captured["shots"][0]["seconds"] == 7.5
    assert not [n for n in output.notes if "FABRICATED" in n], output.notes
