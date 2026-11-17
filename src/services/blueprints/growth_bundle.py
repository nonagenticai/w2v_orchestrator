"""GrowthBundleBlueprint — standalone outbound channel-copy pack.

The leanest blueprint in the fleet: one LLM stage that produces LinkedIn /
Twitter / email / landing-page copy (with a CTA url) straight from the project
brief — no video, no prior stages. The same ``growth_bundle`` stage is embedded
as a tail companion inside promo / course / demo_video / showcase / website_review
runs; this blueprint exposes it on its own so a caller (e.g. Merqi's channel-copy
path) can render snippets without producing a video.

The stage reads only ``config_snapshot`` (title / brief / voice); its optional
``script`` prior-output degrades to ``{}`` when no prior stage ran, so it runs
standalone. NOTE: the stage's ``require_video`` gate skips it when
``brief.generate_video`` is explicitly False — a standalone growth_bundle brief
must LEAVE ``generate_video`` UNSET (it defaults True) so the stage runs.

``terminal_artifact_stage=None``: the stage emits ``snippets`` (text), not an
``artifact_url``, so there is no run-level artifact gate.
"""

from __future__ import annotations

from src.services.blueprints.base import DAGBlueprint, StageSpec, handler
from src.services.blueprints.registry import get_registry
from src.services.stages.growth_bundle.handler import GrowthBundleStage

GROWTH_BUNDLE_BLUEPRINT = DAGBlueprint(
    id="growth_bundle_v1",
    project_type="growth_bundle",
    terminal_artifact_stage=None,
    stages=[
        StageSpec(
            name="growth_bundle",
            handler_factory=handler(GrowthBundleStage),
            # LLM text stage on the shared 660s llm_budget ladder; the outer
            # ceiling must exceed the inner budget by >=30s (see
            # test_llm_timeout_ladder). Keep the base 700s default.
        ),
    ],
)


get_registry().register(GROWTH_BUNDLE_BLUEPRINT)
