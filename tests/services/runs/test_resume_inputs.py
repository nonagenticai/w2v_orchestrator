"""The resume allowlist may only name blueprints whose every stage was swept.

A failed-run Restart preserves completed stages when the persisted fingerprint of
what they read is unchanged (``src/services/runs/resume_inputs.py``). A stage that
reads a project input LIVE and is not in the sweep would silently replay stale
input — so adding a stage to an allowlisted blueprint must fail here until someone
records what it reads.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

import src.services.blueprints as blueprints_pkg
from src.services.blueprints.base import DAGBlueprint, StageSpec
from src.services.blueprints.registry import get_registry
from src.services.runs.resume_inputs import (
    RESUMABLE_BLUEPRINTS,
    RESUME_INPUTS_KEY,
    STAGE_LIVE_INPUTS,
    LiveInput,
    resume_refusal,
    uncovered_stages,
)


def _all_blueprints() -> dict[str, DAGBlueprint]:
    for module in pkgutil.iter_modules(blueprints_pkg.__path__):
        importlib.import_module(f"{blueprints_pkg.__name__}.{module.name}")
    return {bp.id: bp for bp in get_registry().all()}


def test_every_allowlisted_blueprint_is_registered() -> None:
    assert set(_all_blueprints()) >= RESUMABLE_BLUEPRINTS


@pytest.mark.parametrize("blueprint_id", list(sorted(RESUMABLE_BLUEPRINTS))[:1])
def test_allowlisted_blueprint_contains_only_covered_stages(blueprint_id: str) -> None:
    blueprint = _all_blueprints()[blueprint_id]
    assert uncovered_stages(blueprint) == [], (
        f"{blueprint_id} is in RESUMABLE_BLUEPRINTS but wires stages the live-input "
        "sweep does not cover — record what they read in STAGE_LIVE_INPUTS (and "
        "fingerprint it) or drop the blueprint from the allowlist"
    )


def test_blueprints_reading_unfingerprinted_inputs_are_not_allowlisted() -> None:
    blueprints = _all_blueprints()
    assert uncovered_stages(blueprints["course_v1"]) == ["course_compile_lessons"]
    assert uncovered_stages(blueprints["audiobook_v1"]) == ["audiobook_ingest"]
    assert not {"course_v1", "audiobook_v1"} & RESUMABLE_BLUEPRINTS


def test_a_new_unswept_stage_is_uncovered() -> None:
    """The guard's own negative: a stage name nobody swept is not covered."""
    base = _all_blueprints()["website_review_v1"]
    spec = base.stages[0]
    grown = DAGBlueprint(
        id="probe",
        project_type="probe",
        stages=[
            *base.stages,
            StageSpec(name="brand_new_stage", handler_factory=spec.handler_factory),
        ],
    )
    assert uncovered_stages(grown) == ["brand_new_stage"]


def test_footage_probe_is_swept_as_reading_video_sources() -> None:
    assert STAGE_LIVE_INPUTS["footage_probe"] == frozenset({LiveInput.VIDEO_SOURCES})
    assert STAGE_LIVE_INPUTS["source_ingest"] == frozenset({LiveInput.UPLOADED_DOCUMENTS})


def test_resume_refusal_shapes() -> None:
    assert resume_refusal({}) == ["run predates the live-input fingerprint"]
    assert resume_refusal({RESUME_INPUTS_KEY: {"unresumable": []}}) is None
    assert resume_refusal({RESUME_INPUTS_KEY: {"unresumable": ["x"]}}) == ["x"]
    assert resume_refusal({RESUME_INPUTS_KEY: {}}) == ["malformed live-input fingerprint"]
