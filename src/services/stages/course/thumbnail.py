"""course_thumbnail — generate one AI cover image for a course.

Runs in parallel with ``course_compile_lessons`` (both depend only on
``course_outline``), so the cover is ready by the time ``course_stitch`` runs
— with no impact on the 4h lesson-fanout critical path. The cover is used as
the in-app video poster and the YouTube custom thumbnail.

Skippable by design: cover generation calls the (GPU, occasionally flaky)
image-gen backend, and a missing cover must NEVER fail a course. Both an
explicit opt-out (``course_thumbnail: false`` / ``{enabled: false}`` in the run
config) and any generation failure raise ``StageSkippedError`` → the stage is
marked ``skipped`` and the run still reaches ``completed``. ``course_stitch``
reads this stage's output best-effort (None when skipped).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, ClassVar

from src.clients.ai_content_generation_client import AIContentGenerationClient
from src.clients.schemas.ai_content_generation import AcgConceptImageJob
from src.services.concept_images import DEFAULT_BUDGET_S, generate_concept_image
from src.services.stages.base import (
    ArtifactUrl,
    SlotUse,
    StageHandler,
    StageInput,
    StageOutputWithNotes,
    StagePreconditionError,
    StageSkippedError,
)
from src.services.tts_storage import (
    deterministic_media_id,
    find_existing_media,
    store_generated_image,
)

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

log = logging.getLogger(__name__)

# The whole cover budget (queue + render + same-key retry). Not a render budget:
# the one-at-a-time image backend queues this behind every run's images, and the
# old 180 s render-sized deadline timed covers out in the queue. The helper
# (``src.services.concept_images``) owns submit/wait/retry/cancel. Well inside
# the stage's 1300 s ``default_timeout_seconds`` (blueprints/course.py).
_DEFAULT_BUDGET_S = DEFAULT_BUDGET_S
_MIN_AUTHORED_BUDGET_S = 60.0
# YouTube thumbnails are 16:9; the image-gen backend caps resolution on-device,
# so this is an upper hint, not a guarantee.
_DEFAULT_SIZE = "1280x720"


class CourseThumbnailInput(StageInput):
    pass  # All inputs come from ctx.config_snapshot.


class CourseThumbnailOutput(StageOutputWithNotes):
    # Optional because a cover-generation FAILURE now completes-with-notes instead of
    # laundering itself into `skipped` (see `handle`). `course_stitch` already reads this
    # as `thumb.get("thumbnail_url")` and renders without a poster when absent
    # (`stitch.py:315-316`), so the consumer contract is unchanged.
    thumbnail_url: ArtifactUrl | None = None


class CourseThumbnailStage(StageHandler[CourseThumbnailInput, CourseThumbnailOutput]):
    name = "course_thumbnail"
    input_model = CourseThumbnailInput
    output_model = CourseThumbnailOutput
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (SlotUse("image"),)

    def __init__(self, client: AIContentGenerationClient | None = None) -> None:
        self._client = client or AIContentGenerationClient()

    async def handle(
        self,
        payload: CourseThumbnailInput,
        ctx: RunContext,
    ) -> CourseThumbnailOutput:
        cfg = ctx.config_snapshot.get("course_thumbnail")
        # Default-on. Opt out with `course_thumbnail: false` or
        # `course_thumbnail: {enabled: false}` (used by E2E to stay deterministic
        # and off the flaky GPU path).
        if cfg is False or (isinstance(cfg, dict) and cfg.get("enabled") is False):
            raise StageSkippedError(
                "course_thumbnail: disabled via config",
                missing=["course_thumbnail.enabled"],
            )

        cfg = cfg if isinstance(cfg, dict) else {}
        prompt = str(cfg.get("prompt") or self._build_prompt(ctx))
        job_config = self._build_job_config(cfg)
        # ``job_timeout_s`` now names the whole budget; clamped so an authored
        # value can neither outlive the default nor starve the queue wait.
        budget_s = max(
            min(float(cfg.get("job_timeout_s") or _DEFAULT_BUDGET_S), _DEFAULT_BUDGET_S),
            _MIN_AUTHORED_BUDGET_S,
        )

        run_id = str(ctx.run_id)
        # Stable key over the generative inputs (config affects the output), so the
        # same (prompt, config) always addresses the same stored cover. Mirrors
        # `generic/ai_image_gen.py` — this is the only paid-generation call site in
        # the stage tree, so it gets the same idempotency shape as its nine siblings.
        job_key = json.dumps(job_config, sort_keys=True)
        media_id = deterministic_media_id(prompt, job_key, None)

        try:
            # Resume/retry idempotency: if this exact cover was already generated and
            # republished this run, reuse the stored asset instead of paying for
            # another ACG generation. A hit means the fetchable object exists, so it
            # also skips the re-download/re-store.
            existing = await find_existing_media(run_id, self.name, media_id, "png")
            if existing is not None:
                return CourseThumbnailOutput(thumbnail_url=existing)
            # Idempotency: a re-dispatch (resumable timeout) or reaper re-claim
            # re-runs this handler. Without a dedupe key the re-submit starts a
            # SECOND billed GPU image job that ACG cannot dedupe, because nothing
            # addresses the first. The key (run + stage + content-addressed
            # media_id) makes ACG return the ORIGINAL job on the re-submit.
            result = await generate_concept_image(
                self._client,
                prompt,
                job_config,
                idempotency_key=f"{run_id}:{self.name}:{media_id}",
                budget_s=budget_s,
            )
            thumbnail_url = await self._store_cover(result, run_id=run_id, image_id=media_id)
        # Control-flow signals are NOT failures — they must reach the run engine, not
        # be swallowed into a degraded "no poster". `StagePreconditionError` is a
        # `StageError` subclass, so the bare `except Exception` below catches it
        # unless it is re-raised here alongside `StageSkippedError`, exactly as the
        # sibling soft-degrade wrappers do (`shared/script_doctor.py`,
        # `audiobook/qa.py`, `shared/loop_engine.py`, `shared/footage_critique.py`).
        except StageSkippedError, StagePreconditionError:
            raise
        except Exception as exc:
            # A cover failure must never fail the course — but it must not disguise itself
            # as "the run never asked for a cover" either. `StageSkippedError` is defined
            # (errors.py) as an OPT-IN stage the run did not request, and the DAG executor
            # maps it to `skipped`: a terminal state that counts toward completion and is
            # invisible in run status. So a paid, attempted, FAILED image generation was
            # indistinguishable from a brief that never wanted one, and the course shipped
            # with a blank poster on a clean green run.
            #
            # Completed-with-notes is the channel that already exists for exactly this:
            # non-empty `notes` renders the run `degraded` (api/routes/runs/_shared.py),
            # `thumbnail_url` stays None, and `course_stitch` renders without a poster
            # exactly as it did before. Same outcome for the video, honest status.
            log.error("course_thumbnail: cover generation failed", exc_info=True)
            return CourseThumbnailOutput(
                thumbnail_url=None,
                notes=[f"course_thumbnail: cover generation failed ({exc}); no poster frame"],
            )

        return CourseThumbnailOutput(thumbnail_url=thumbnail_url)

    async def _store_cover(self, result: AcgConceptImageJob, *, run_id: str, image_id: str) -> str:
        """Resolve the ACG result to a fetchable artifact-store URL.

        ACG returns either a license-gated, ACG-relative download path
        (``/api/concept-images/assets/...``) or an absolute URL. Relative paths
        are downloaded (the client carries the licence token) and re-published to
        the run store so the FE poster + the YouTube uploader can both fetch them;
        absolute URLs pass through. Mirrors ``ai_image_gen._republish_relative``.
        """
        url = (result.image_url or result.download_url or "").strip()
        if not url or url.lower() in {"none", "null"}:
            raise RuntimeError(f"no usable image_url (status={result.status!r})")
        if url.startswith("/"):
            data = await self._client.download_concept_image(url)
            return await store_generated_image(
                run_id=run_id,
                stage_name=self.name,
                image_id=image_id,
                image_bytes=data,
                image_format="png",
            )
        return url

    @staticmethod
    def _build_prompt(ctx: RunContext) -> str:
        """Compose a cover prompt from the course title + brief audience/outcome."""
        cfg = ctx.config_snapshot
        brief = cfg.get("brief") or {}
        title = str(cfg.get("title") or "Course")
        audience = str(brief.get("audience") or "").strip()
        outcome = str(brief.get("outcome") or "").strip()
        parts = [
            f"Professional course cover artwork for '{title}'.",
        ]
        if audience:
            parts.append(f"Audience: {audience}.")
        if outcome:
            parts.append(f"Outcome: {outcome}.")
        parts.append(
            "Clean, modern, editorial 16:9 thumbnail composition with strong focal "
            "imagery and ample negative space; no embedded text or watermarks."
        )
        return " ".join(parts)

    @staticmethod
    def _build_job_config(cfg: dict) -> dict[str, Any]:
        job_config: dict[str, Any] = {"size": str(cfg.get("size") or _DEFAULT_SIZE)}
        for key in ("provider", "style", "negative_prompt", "extra_params"):
            if cfg.get(key) is not None:
                job_config[key] = cfg[key]
        return job_config
