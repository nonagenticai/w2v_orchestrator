"""AppReviewBlueprint — inspect → critique → script → record → compile.

DORMANT / FUTURE FUNCTIONALITY (parked 2026-06-03). app_review is hidden from
the frontend (the project-type picker option + the ?mode=critique_only deep-link
are commented out in w2v_frontend), so no new app_review projects are created
through the UI. The blueprint is intentionally LEFT REGISTERED here so the
pipeline stays callable (existing projects, the post-promote E2E scenario, and
direct API use keep working) — a deliberate "dormant, not removed" state.
Re-expose it in the frontend when the feature is brought back.
"""

from __future__ import annotations

from src.services.blueprints.base import DAGBlueprint, StageSpec, handler
from src.services.blueprints.registry import get_registry
from src.services.stages.app_review.compile import AppReviewCompileStage
from src.services.stages.app_review.critique import AppCritiqueStage
from src.services.stages.app_review.inspect import AppInspectStage
from src.services.stages.app_review.record import AppReviewRecordStage
from src.services.stages.app_review.script import AppReviewScriptStage
from src.services.stages.growth_bundle.handler import GrowthBundleStage
from src.services.stages.primitives.compile_stage import COMPILE_STAGE_OUTER_TIMEOUT_SECONDS

APP_REVIEW_BLUEPRINT = DAGBlueprint(
    id="app_review_v1",
    project_type="app_review",
    terminal_artifact_stage="app_review_compile",
    stages=[
        StageSpec(
            name="app_inspect",
            handler_factory=handler(AppInspectStage),
            default_timeout_seconds=900,
            default_pause_after=True,
        ),
        StageSpec(
            name="app_critique",
            depends_on=("app_inspect",),
            handler_factory=handler(AppCritiqueStage),
            default_pause_after=True,
        ),
        StageSpec(
            name="app_review_script",
            depends_on=("app_critique",),
            handler_factory=handler(AppReviewScriptStage),
            default_pause_after=True,
        ),
        StageSpec(
            name="app_review_record",
            depends_on=("app_review_script",),
            handler_factory=handler(AppReviewRecordStage),
            # Inner stage_gather worst case: 2 attempts x 600s = 1200s per item.
            # Mirrors website_review_record's tuning.
            default_timeout_seconds=1300,
        ),
        StageSpec(
            name="app_review_compile",
            depends_on=("app_review_record",),
            handler_factory=handler(AppReviewCompileStage),
            # Outer Timeout must cover the compile-wait backstop below it (see
            # compile_stage.COMPILE_STAGE_OUTER_TIMEOUT_SECONDS) or the wait is
            # truncated one layer up from the fix that already covers this in
            # compile_stage.py.
            default_timeout_seconds=COMPILE_STAGE_OUTER_TIMEOUT_SECONDS,
        ),
        StageSpec(
            name="growth_bundle",
            depends_on=("app_review_compile",),
            handler_factory=handler(GrowthBundleStage),
        ),
    ],
)


get_registry().register(APP_REVIEW_BLUEPRINT)
