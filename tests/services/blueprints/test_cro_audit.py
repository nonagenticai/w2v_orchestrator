from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

import src.services.blueprints.audit_family  # noqa: F401  registers the audit families
from src.services.blueprints.registry import get_registry
from src.services.runs.run_finalizer import (
    RunDecision,
    decide_run_terminal_state,
    verify_terminal_artifact,
)

CRO_AUDIT_BLUEPRINT = get_registry().resolve("cro_audit")

_VIDEO_STAGES = [
    "cro_audit_script",
    "cro_audit_slides",
    "cro_audit_narrate",
    "cro_audit_record",
    "cro_audit_compile",
]


def test_cro_audit_registered() -> None:
    assert get_registry().resolve("cro_audit") is CRO_AUDIT_BLUEPRINT


def test_cro_audit_topology() -> None:
    order = CRO_AUDIT_BLUEPRINT.topological_order()
    # Scan is the root; the optional video chain follows in dependency order.
    assert order.index("cro_audit_scan") < order.index("cro_audit_script")
    assert order.index("cro_audit_script") < order.index("cro_audit_slides")
    assert order.index("cro_audit_slides") < order.index("cro_audit_narrate")
    assert order.index("cro_audit_slides") < order.index("cro_audit_record")
    assert order.index("cro_audit_narrate") < order.index("cro_audit_compile")
    assert order.index("cro_audit_record") < order.index("cro_audit_compile")


def test_narrate_and_record_co_scheduled() -> None:
    # Both depend only on cro_audit_slides so the scheduler makes them eligible in
    # the same round (TTS ‖ deck render), mirroring seo_audit_narrate ‖ record.
    assert CRO_AUDIT_BLUEPRINT.stage("cro_audit_narrate").depends_on == ("cro_audit_slides",)
    assert CRO_AUDIT_BLUEPRINT.stage("cro_audit_record").depends_on == ("cro_audit_slides",)


def test_cro_audit_compile_is_terminal_artifact_stage() -> None:
    assert CRO_AUDIT_BLUEPRINT.terminal_artifact_stage == "cro_audit_compile"
    assert CRO_AUDIT_BLUEPRINT.project_type == "cro_audit"
    assert CRO_AUDIT_BLUEPRINT.stage("cro_audit_compile").depends_on == (
        "cro_audit_narrate",
        "cro_audit_record",
    )


@dataclass
class _Row:
    stage_name: str
    status: str
    output: dict[str, Any] | None = None


def test_scan_only_run_with_video_skipped_reaches_terminal_state() -> None:
    """A default scan-only run (generate_video unset) skips the whole video chain,
    yet still reaches a terminal state — the scan-only deliverable is preserved.

    ``skipped`` counts as terminal, so ``terminal_stage_names`` == every stage →
    DAG_COMPLETE; the skipped ``cro_audit_compile`` terminal row holds no output,
    so ``verify_terminal_artifact`` returns None (no artifact gate).
    """
    rows = [
        _Row("source_ingest", "skipped"),  # opt-in grounding leaf, skipped with no brief.sources
        _Row("cro_audit_scan", "completed", output={"overall_score": 80.0}),
    ]
    rows += [_Row(name, "skipped") for name in _VIDEO_STAGES]

    decision = decide_run_terminal_state(CRO_AUDIT_BLUEPRINT, rows, pause_set=set())
    assert decision is RunDecision.DAG_COMPLETE


def test_source_ingest_precedes_scan() -> None:
    # Opt-in multi-source grounding: source_ingest is a no-dep leaf that the scan
    # depends on, so its scraped grounding_text is available to the scorecard call.
    order = CRO_AUDIT_BLUEPRINT.topological_order()
    assert "source_ingest" in order
    assert order.index("source_ingest") < order.index("cro_audit_scan")
    assert CRO_AUDIT_BLUEPRINT.stage("cro_audit_scan").depends_on == ("source_ingest",)
    assert CRO_AUDIT_BLUEPRINT.stage("source_ingest").depends_on == ()


@pytest.mark.asyncio
async def test_verify_terminal_artifact_none_when_compile_skipped() -> None:
    rows = [_Row("cro_audit_scan", "completed", output={"overall_score": 80.0})]
    rows += [_Row(name, "skipped") for name in _VIDEO_STAGES]
    # Skipped terminal row (no output) → no artifact to gate on → None (completes).
    assert await verify_terminal_artifact(CRO_AUDIT_BLUEPRINT, rows) is None
