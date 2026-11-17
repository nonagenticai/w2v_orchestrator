"""app_review_compile — assemble narration + visuals into the final review video."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from src.services.stages.base import ArtifactUrl, StageInput, StageOutputWithNotes
from src.services.stages.primitives.compile_stage import (
    CompileStageBase,
    playable_duration_seconds,
)
from src.services.stages.shared.duration_conformance import (
    DurationConformance,
    duration_conformance_advisories,
    duration_conformance_degradations,
    observe_duration_conformance,
    picture_truncation_degradations,
    review_ordered_seconds,
)
from src.services.stages.shot_assembly import assemble_shots

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext


class AppReviewCompileInput(StageInput):
    pass


class AppReviewCompileOutput(StageOutputWithNotes):
    # Declared output contract — enforced once, centrally, in
    # `runs/dag_executor.py::_enforce_produces`. This is a TERMINAL artifact
    # producer: `artifact_url` IS the deliverable, not a hint about it, so an
    # empty one is never an acceptable outcome. Hence `produces_required` — the
    # stage FAILS instead of degrading. Degrade was the wrong default here: it let
    # the run finish while handing the user a reference to nothing, which is the
    # exact defect class this contract exists to close.
    produces: ClassVar[tuple[str, ...]] = ("artifact_url",)
    produces_required: ClassVar[bool] = True

    artifact_url: ArtifactUrl
    duration_seconds: float
    # Delivered-vs-ORDERED duration, recorded so a deviation is visible in the
    # run record: app_review masters measured 140% of their ordered length on
    # 2026-09-10 and nothing anywhere noticed, because nothing compared the two.
    # Observation ONLY — it never fails or degrades the stage, and it is
    # deliberately NOT a `notes` entry (a note flips the whole run to
    # `degraded`). `None` means no usable duration was ordered, which is NOT the
    # same fact as "on target".
    duration_conformance: DurationConformance | None = None


class AppReviewCompileStage(CompileStageBase[AppReviewCompileInput, AppReviewCompileOutput]):
    name = "app_review_compile"
    input_model = AppReviewCompileInput
    output_model = AppReviewCompileOutput

    def build_shots(self, ctx: RunContext) -> list[dict[str, Any]]:
        record = ctx.prior_output("app_review_record")
        if not record:
            raise RuntimeError("app_review_record required")
        visuals = record["visuals"]
        # Spine = per-line visuals; audio joins by line_id. A screenshot is the
        # shot's still image (keyed by its own line_id, 1:1 with the visual); a
        # fact_card has none and falls back to the compositor's text card. No
        # storyboard here, so no `description`.
        image_by_line = {
            v["line_id"]: v["artifact_url"]
            for v in visuals
            if v.get("kind") == "screenshot" and v.get("artifact_url")
        }
        return assemble_shots(
            storyboard_shots=[
                {"id": v["id"], "line_id": v["line_id"], "seconds": float(v["seconds"] or 0.0)}
                for v in visuals
            ],
            images=[],
            narration_clips=[
                {
                    "line_id": n["line_id"],
                    "artifact_url": n["artifact_url"],
                    "seconds": n["seconds"],
                }
                for n in record["narrations"]
            ],
            image_by_line=image_by_line,
        )

    def build_output(self, result: dict[str, Any], ctx: RunContext) -> AppReviewCompileOutput:
        delivered = float(result["duration_seconds"])
        # Ordered length resolves exactly as `app_review/script.py:64` does
        # (`brief.duration_seconds`, default 120) — the same number the script
        # was written against, so the comparison is like-for-like. A breach
        # lands on BOTH the structured field (the numbers) and the
        # `degradations` channel (the verdict that makes the run report
        # `degraded`).
        conformance = observe_duration_conformance(
            self.name,
            review_ordered_seconds(ctx.config_snapshot),
            delivered,
            playable_seconds=playable_duration_seconds(result),
        )
        return AppReviewCompileOutput(
            artifact_url=str(result["artifact_url"]),
            duration_seconds=delivered,
            duration_conformance=conformance,
            degradations=[
                *duration_conformance_degradations(conformance),
                # A SECOND, independent fact about the same master. The verdict
                # above grades the PLAYABLE length (max of the two streams)
                # against the order; it cannot express a picture that stopped
                # early, because on such a master that max IS the audio. The
                # picture-vs-audio gap therefore gets its own code — one number
                # carrying both facts is the regression this replaces.
                *picture_truncation_degradations(result.get("mastering")),
            ],
            # The unmeasured fact rides the NON-degrading tier: a length nobody
            # probed is an unverified verdict, not a bad artifact.
            advisories=duration_conformance_advisories(conformance),
        )
