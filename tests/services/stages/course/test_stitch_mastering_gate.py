"""course_stitch must read the compiler's mastering gate — O8.

`course` was the fleet's one genuine A/V blind spot. Every other product type gets
the gate from `CompileStageBase.handle`; `course_stitch` calls `client.compile()`
DIRECTLY, so no base-class fix could ever reach it and loudness, true peak, colour,
CFR, dead-frame and the gate verdict were all measured by the compiler and then
discarded for every course deliverable.

These tests assert the wiring on the CALL — a stage that computes a note and never
attaches it looks identical to a clean run — and they assert that course grades the
SAME WAY as the base does on the same report, since a drift there is the defect.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

from src.services.runs.run_context import RunContext
from src.services.stages.course.stitch import CourseStitchStage
from src.services.stages.shared.degradation import MASTERING_GATE_FAILED
from src.services.stages.shared.mastering_gate import (
    mastering_gate_degradations,
    mastering_gate_notes,
)

# The blank-master signature from run 01a00b81-2576-71a5-becc-ffc4a843e6db: a stage
# the compiler wants re-run, a failed gate, and a dead_frame_ratio over the 0.10
# cutoff. (luma_spread=0.0 alone is NOT blank — one fade-to-black frame zeroes it.)
_BLANK_MASTER = {
    "rerun_stages": ["deband"],
    "gate_pass": False,
    "dead_frame_ratio": 0.25,
    "luma_spread": 0.0,
    "banding_scores": [1.0, 1.0],
}
_CLEAN_MASTER = {"rerun_stages": [], "gate_pass": True, "luma_spread": 210.0}


def _ctx() -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"deliverable": {"format": "16x9"}},
        prior_run_outputs={
            "course_compile_lessons": {
                "lessons": [
                    {
                        "lesson_project_id": "p1",
                        "lesson_run_id": "r1",
                        "artifact_url": "u1",
                        "duration_seconds": 60.0,
                    }
                ]
            }
        },
    )


async def _run(mastering: dict[str, Any] | None) -> Any:
    fake = AsyncMock()
    result: dict[str, Any] = {
        "artifact_url": "http://stub/course.mp4",
        "duration_seconds": 60.0,
    }
    if mastering is not None:
        result["mastering"] = mastering
    fake.compile.return_value = result
    return await CourseStitchStage(client=fake).run({}, ctx=_ctx())


async def test_a_blank_course_master_degrades_instead_of_shipping_silently() -> None:
    """THE point of O8: the course deliverable now carries the gate verdict."""
    out = await _run(_BLANK_MASTER)

    # Degraded, NOT failed — the artifact exists and is a paid deliverable.
    assert out.course_artifact_url
    notes = [n for n in out.notes if n.startswith("mastering:")]
    assert len(notes) == 1, out.notes
    assert "deband" in notes[0]  # names the compiler's own rerun stage
    assert "dead_frame_ratio=0.25 (cutoff 0.1)" in notes[0]  # names the metric
    # And the MACHINE-READABLE half, which is what the run record queries.
    assert [d["code"] for d in out.degradations] == [MASTERING_GATE_FAILED]


async def test_an_absent_course_mastering_report_is_reported_as_unmeasured() -> None:
    """An absent measurement reported as a passing one is this repo's #1 defect.

    A compiler build that returns no `mastering` block at all must not make a course
    byte-identical to one whose gate ran and PASSED.
    """
    out = await _run(None)
    notes = [n for n in out.notes if "UNMEASURED" in n]
    assert len(notes) == 1, out.notes
    assert [d["code"] for d in out.degradations] == [MASTERING_GATE_FAILED]


async def test_course_grades_identically_to_the_shared_helpers() -> None:
    """Mirror-the-base, asserted as an EQUALITY rather than trusted.

    If course ever computes its own verdict, it will silently grade differently from
    the other nine blueprints on the same report. Comparing against the helpers on
    the same payload is what makes that impossible to do quietly.
    """
    for report in (_BLANK_MASTER, _CLEAN_MASTER, None):
        out = await _run(report)
        expected_notes = mastering_gate_notes(report)
        assert [n for n in out.notes if n.startswith("mastering:")] == expected_notes
        assert out.degradations == mastering_gate_degradations(report)


async def test_the_stage_own_notes_survive_alongside_the_gate() -> None:
    """The gate is APPENDED, never assigned: a resolution shortfall must not be lost."""
    fake = AsyncMock()
    fake.compile.return_value = {
        "artifact_url": "http://stub/course.mp4",
        "duration_seconds": 60.0,
        # Ordered 2k, delivered 1080p ⇒ this stage's own resolution note.
        "width": 1920,
        "height": 1080,
        "mastering": _BLANK_MASTER,
    }
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"resolution": "2k", "deliverable": {"format": "16x9"}},
        prior_run_outputs={
            "course_compile_lessons": {
                "lessons": [
                    {
                        "lesson_project_id": "p1",
                        "lesson_run_id": "r1",
                        "artifact_url": "u1",
                        "duration_seconds": 60.0,
                    }
                ]
            }
        },
    )

    out = await CourseStitchStage(client=fake).run({}, ctx=ctx)

    assert any("2k" in n and "1920x1080" in n for n in out.notes)
    assert any(n.startswith("mastering:") for n in out.notes)


async def test_a_course_master_carries_the_compilers_advisories() -> None:
    """course_stitch reads the report itself, so it must surface `advisories` too."""
    from src.services.stages.shared.degradation import MASTERING_ADVISORY
    from src.services.stages.shared.mastering_gate import mastering_gate_advisories

    advisory_entry = {
        "stage": "banding_residual",
        "metric": "banding_scores",
        "value": 0.262,
        "limit": 0.15,
        "tier": "tunable",
    }
    report = {**_CLEAN_MASTER, "advisories": [advisory_entry]}
    out = await _run(report)
    assert [a["code"] for a in out.advisories] == [MASTERING_ADVISORY]
    assert out.advisories == mastering_gate_advisories(report)
    assert out.degradations == []
    assert out.notes == []
    # Absent key ⇒ unchanged.
    assert (await _run(_CLEAN_MASTER)).advisories == []
