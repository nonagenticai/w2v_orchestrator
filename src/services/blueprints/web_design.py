"""WebDesignBlueprint — single-stage greenfield web design from a brief.

The simplest blueprint in the fleet: one stage that drives html_render_service's
design engine in greenfield mode (``scope="landing"``) and surfaces the generated
HTML + screenshot. No critique, no video — the deliverable is the HTML file (with
on-demand PDF). An optional ``brief.target_url`` seeds the design from an existing
site via the engine's refine path.
"""

from __future__ import annotations

from src.services.blueprints.base import (
    DAGBlueprint,
    DeliverableProbe,
    StageFactory,
    StageSpec,
)
from src.services.blueprints.registry import get_registry
from src.services.stages.base import StageHandler
from src.services.stages.web_design.design import WebDesignStage


def _web_design(deps: dict) -> StageHandler:
    # Deps-aware (not bare ``handler(...)``) purely for the session_factory: the
    # stage stashes its in-flight html_render job id in its own stage_outputs row
    # so a resumable-timeout re-dispatch RE-ATTACHES instead of submitting a
    # second job. ``.get`` — absence degrades to the old resubmit behaviour
    # rather than failing the run.
    return WebDesignStage(session_factory=deps.get("session_factory"))


WEB_DESIGN_BLUEPRINT = DAGBlueprint(
    id="web_design_v1",
    project_type="web_design",
    terminal_artifact_stage="web_design",
    # The deliverable is an html_render JOB (HTML + on-demand PDF), not an object
    # we minted in MinIO: ``WebDesignOutput`` publishes ``redesign_job_id`` and the
    # download links are orchestrator PROXY routes over html_render. So the handle
    # is the job id and the probe asks html_render, not the object store.
    terminal_artifact_field="redesign_job_id",
    terminal_artifact_probe=DeliverableProbe.DESIGN_JOB,
    stages=[
        StageSpec(
            name="web_design",
            handler_factory=StageFactory(WebDesignStage, _web_design),
            # Worst case = start(60s) + wait(4 x 590s) + result(2 x 60s); the engine runs
            # an LLM design + Chromium screenshot. Mirror website_redesign's budget.
            default_timeout_seconds=3000,
        ),
    ],
)


get_registry().register(WEB_DESIGN_BLUEPRINT)
