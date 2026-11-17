from __future__ import annotations

import src.services.blueprints.audit_family  # noqa: F401  registers the audit families
from src.services.blueprints.registry import get_registry

MARKETING_SCORECARD_BLUEPRINT = get_registry().resolve("marketing_scorecard")


def test_marketing_scorecard_registered() -> None:
    assert get_registry().resolve("marketing_scorecard") is MARKETING_SCORECARD_BLUEPRINT


def test_marketing_scorecard_is_scan_only() -> None:
    # Scan-only: the scan's structured output IS the deliverable (no video), so there
    # is no run-level artifact gate (mirrors growth_bundle). The opt-in source_ingest
    # leaf precedes the scan (skipped when the brief carries no sources).
    assert MARKETING_SCORECARD_BLUEPRINT.project_type == "marketing_scorecard"
    assert MARKETING_SCORECARD_BLUEPRINT.terminal_artifact_stage is None
    assert MARKETING_SCORECARD_BLUEPRINT.topological_order() == [
        "source_ingest",
        "marketing_scorecard_scan",
    ]
    assert MARKETING_SCORECARD_BLUEPRINT.stage("marketing_scorecard_scan").depends_on == (
        "source_ingest",
    )
