"""site_capture wiring in the promo blueprint.

site_capture is an opt-in, no-dependency leaf (it co-schedules with
promo_analyze at run open). Three stages consume it: promo_script (scraped
site copy → grounding), promo_visual_director (per-screen list → line→screen
map) and promo_render (recorded clips / screenshots → background footage).
Its handler factory must build a SiteCaptureStage.
"""

from __future__ import annotations

from src.services.blueprints.promo import PROMO_BLUEPRINT
from src.services.stages.generic.site_capture import SiteCaptureStage


def test_site_capture_present_as_leaf() -> None:
    spec = PROMO_BLUEPRINT.stage("site_capture")
    assert spec.depends_on == (), "site_capture should be a no-dep leaf"


def test_site_capture_consumers_are_script_and_render() -> None:
    consumers = sorted(s.name for s in PROMO_BLUEPRINT.stages if "site_capture" in s.depends_on)
    assert consumers == [
        "promo_footage_direction",
        "promo_render",
        "promo_script",
        "promo_visual_director",
        "source_ingest",
    ]


def test_site_capture_handler_factory_builds() -> None:
    spec = PROMO_BLUEPRINT.stage("site_capture")
    assert isinstance(spec.handler_factory({}), SiteCaptureStage)


def test_site_capture_does_not_break_dag_resolution() -> None:
    # With site_capture added, the DAG must still resolve to a valid topological
    # order (no cycle, no orphan).
    order = PROMO_BLUEPRINT.topological_order()
    assert "site_capture" in order
    assert order.index("site_capture") < order.index("promo_script")
    assert order.index("site_capture") < order.index("promo_render")
