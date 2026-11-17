"""WebDesignV2Blueprint — computed-assembly, interactive, verified web design.

A richer web-design pipeline than ``web_design`` (single greenfield stage):

    brand_extract → design → verify → assemble

* **brand_extract** — seed palette/fonts from ``brief.target_url`` (analysis_service)
  or the project brand block.
* **design** — html_render_service in ``mode="assembly"``: selects vetted catalog
  blocks, themes from the brand, injects ``embeds`` (three.js seam), runs its
  verification lane. A design that fails verification is failed by the service, so
  this stage fails and the run cannot complete.
* **verify** — surface the verification report as an explicit gate/UI step.
* **assemble** — the terminal deliverable (live preview URL + HTML + PDF +
  screenshots + report).

Consumed by the brand service via the standard project/run API
(``type="web_design_v2"``).
"""

from __future__ import annotations

from src.services.blueprints.base import (
    DAGBlueprint,
    DeliverableProbe,
    StageFactory,
    StageSpec,
    handler,
)
from src.services.blueprints.registry import get_registry
from src.services.stages.base import StageHandler
from src.services.stages.web_design_v2.assemble import WebDesignAssembleStage
from src.services.stages.web_design_v2.brand_extract import BrandExtractStage
from src.services.stages.web_design_v2.design import WebDesignV2Stage
from src.services.stages.web_design_v2.verify import WebDesignVerifyStage


def _web_design_v2_design(deps: dict) -> StageHandler:
    # Deps-aware purely for the session_factory: the stage stashes its in-flight
    # html_render job id so a resumable-timeout re-dispatch RE-ATTACHES instead of
    # submitting a second job. ``.get`` — absence degrades to the old resubmit
    # behaviour rather than failing the run.
    return WebDesignV2Stage(session_factory=deps.get("session_factory"))


WEB_DESIGN_V2_BLUEPRINT = DAGBlueprint(
    id="web_design_v2",
    project_type="web_design_v2",
    terminal_artifact_stage="assemble",
    # ``assemble`` publishes ``live_html_url`` / ``pdf_url``, which are
    # ORCHESTRATOR PROXY routes (``/api/redesign/jobs/{id}/html|pdf`` →
    # html_render), never storage URLs — an object-store probe of them can only
    # ever answer "don't know". The verifiable handle behind both is the
    # html_render job id, so that is what the finalizer probes.
    terminal_artifact_field="redesign_job_id",
    terminal_artifact_probe=DeliverableProbe.DESIGN_JOB,
    stages=[
        StageSpec(
            name="brand_extract",
            handler_factory=handler(BrandExtractStage),
            default_timeout_seconds=120,
        ),
        StageSpec(
            name="design",
            depends_on=("brand_extract",),
            handler_factory=StageFactory(WebDesignV2Stage, _web_design_v2_design),
            # start(60) + wait(4x590) + result(2x60); assembly runs an LLM plan +
            # Chromium render + the verification lane (+ up to 2 refine passes).
            default_timeout_seconds=3000,
        ),
        StageSpec(
            name="verify",
            depends_on=("design",),
            handler_factory=handler(WebDesignVerifyStage),
            default_timeout_seconds=60,
        ),
        StageSpec(
            name="assemble",
            depends_on=("verify",),
            handler_factory=handler(WebDesignAssembleStage),
            default_timeout_seconds=60,
        ),
    ],
)


get_registry().register(WEB_DESIGN_V2_BLUEPRINT)
