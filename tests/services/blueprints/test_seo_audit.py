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

SEO_AUDIT_BLUEPRINT = get_registry().resolve("seo_audit")

_VIDEO_STAGES = [
    "seo_audit_script",
    "seo_audit_slides",
    "seo_audit_narrate",
    "seo_audit_record",
    "seo_audit_compile",
]


def test_seo_audit_registered() -> None:
    assert get_registry().resolve("seo_audit") is SEO_AUDIT_BLUEPRINT


def test_seo_audit_topology() -> None:
    order = SEO_AUDIT_BLUEPRINT.topological_order()
    # Scan is the root; the optional video chain follows in dependency order.
    assert order.index("seo_audit_scan") < order.index("seo_audit_script")
    assert order.index("seo_audit_script") < order.index("seo_audit_slides")
    assert order.index("seo_audit_slides") < order.index("seo_audit_narrate")
    assert order.index("seo_audit_slides") < order.index("seo_audit_record")
    assert order.index("seo_audit_narrate") < order.index("seo_audit_compile")
    assert order.index("seo_audit_record") < order.index("seo_audit_compile")


def test_narrate_and_record_co_scheduled() -> None:
    # Both depend only on seo_audit_slides so the scheduler makes them eligible in
    # the same round (TTS ‖ deck render), mirroring lesson_narrate ‖ ai_image_gen.
    assert SEO_AUDIT_BLUEPRINT.stage("seo_audit_narrate").depends_on == ("seo_audit_slides",)
    assert SEO_AUDIT_BLUEPRINT.stage("seo_audit_record").depends_on == ("seo_audit_slides",)


def test_seo_audit_compile_is_terminal_artifact_stage() -> None:
    assert SEO_AUDIT_BLUEPRINT.terminal_artifact_stage == "seo_audit_compile"
    assert SEO_AUDIT_BLUEPRINT.project_type == "seo_audit"
    assert SEO_AUDIT_BLUEPRINT.stage("seo_audit_compile").depends_on == (
        "seo_audit_narrate",
        "seo_audit_record",
    )


@dataclass
class _Row:
    stage_name: str
    status: str
    output: dict[str, Any] | None = None


def test_scan_only_run_with_video_skipped_reaches_terminal_state() -> None:
    """The key correctness property: a default scan-only run (generate_video
    unset) skips the whole video chain, yet still reaches a terminal state.

    ``skipped`` counts as terminal, so ``terminal_stage_names`` == every stage →
    DAG_COMPLETE; the skipped ``seo_audit_compile`` terminal row holds no output,
    so ``verify_terminal_artifact`` returns None (no artifact gate) and the run
    completes on the scan output.
    """
    rows = [
        _Row("source_ingest", "skipped"),  # opt-in grounding leaf, skipped with no brief.sources
        _Row("seo_audit_scan", "completed", output={"overall_score": 80.0}),
    ]
    rows += [_Row(name, "skipped") for name in _VIDEO_STAGES]

    decision = decide_run_terminal_state(SEO_AUDIT_BLUEPRINT, rows, pause_set=set())
    assert decision is RunDecision.DAG_COMPLETE


@pytest.mark.asyncio
async def test_verify_terminal_artifact_none_when_compile_skipped() -> None:
    rows = [_Row("seo_audit_scan", "completed", output={"overall_score": 80.0})]
    rows += [_Row(name, "skipped") for name in _VIDEO_STAGES]
    # Skipped terminal row (no output) → no artifact to gate on → None (completes).
    assert await verify_terminal_artifact(SEO_AUDIT_BLUEPRINT, rows) is None
