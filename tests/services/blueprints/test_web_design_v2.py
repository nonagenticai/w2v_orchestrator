"""web_design_v2 blueprint topology + registration."""

from __future__ import annotations

from src.services.blueprints.registry import get_registry
from src.services.blueprints.web_design_v2 import WEB_DESIGN_V2_BLUEPRINT


def test_registered() -> None:
    assert get_registry().resolve("web_design_v2") is WEB_DESIGN_V2_BLUEPRINT
    assert WEB_DESIGN_V2_BLUEPRINT.id == "web_design_v2"
    assert WEB_DESIGN_V2_BLUEPRINT.project_type == "web_design_v2"


def test_topology_is_linear_pipeline() -> None:
    order = WEB_DESIGN_V2_BLUEPRINT.topological_order()
    assert order == ["brand_extract", "design", "verify", "assemble"]


def test_dependencies() -> None:
    bp = WEB_DESIGN_V2_BLUEPRINT
    assert bp.stage("brand_extract").depends_on == ()
    assert bp.stage("design").depends_on == ("brand_extract",)
    assert bp.stage("verify").depends_on == ("design",)
    assert bp.stage("assemble").depends_on == ("verify",)


def test_terminal_artifact_stage_is_assemble() -> None:
    assert WEB_DESIGN_V2_BLUEPRINT.terminal_artifact_stage == "assemble"
