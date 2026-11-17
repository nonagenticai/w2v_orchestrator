"""ai_image_gen — dispatch concept-image generation to ACG.

Reads `ai_image_gen` config from ``ctx.config_snapshot``:

  ai_image_gen:
    prompts:        # list[str]; required-by-default. If absent, falls back
                    # to a single prompt derived from the brief title+angle.
    prompts_from_stage:  # optional: name of a prior stage whose output has
                    # `prompts:[{line_id,prompt}]` (e.g. "promo_image_prompts").
                    # When set + non-empty, those per-line prompts win and each
                    # image carries its line_id (keyed into the render); else
                    # the static `prompts` above are used.
    provider: ...   # optional override; ACG defaults to its configured
                    # default_provider (e.g. "local_image" in cluster).
    size: "WxH"
    style: ...
    negative_prompt: ...
    extra_params: {...}

Output: one ``GeneratedImage`` per prompt with stable ``id`` for annotation
pinning. Per-prompt jobs run via ``stage_gather`` with bounded concurrency,
so a 6-prompt brief renders in parallel groups instead of strictly serially.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, ClassVar

import httpx
from pydantic import Field
from sqlalchemy import select
from w2v_common.stages import stage_gather

from src.clients.ai_content_generation_client import AIContentGenerationClient
from src.clients.schemas.ai_content_generation import AcgConceptImageJob
from src.database.models import Run
from src.database.repositories.stage_output_repository import StageOutputRepository
from src.database.session import bind_session_tenant
from src.services.concept_images import DEFAULT_BUDGET_S, generate_concept_image
from src.services.runs.run_context import REGENERATE_FROM_STAGE_KEY
from src.services.stages.base import (
    SlotUse,
    StageDownstreamResponseError,
    StageHandler,
    StageInput,
    StageOutput,
    StageOutputWithNotes,
    StageSkippedError,
)
from src.services.stages.generic._job_config import build_job_config
from src.services.stages.upstream import TransientUpstreamError
from src.services.tts_storage import (
    artifact_exists,
    deterministic_media_id,
    find_existing_media,
    store_generated_image,
)
from src.services.visual_media import assign_generic_line_media

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

log = logging.getLogger(__name__)

# Whole budget per image: queue + render + the helper's same-key retry. The
# helper (``src.services.concept_images``) owns submit/wait/retry/cancel and
# stops waiting DEADLINE_MARGIN_S before this, so stage_gather never cuts a poll
# off mid-flight. See that module for why the deadline must cover queue time.
_PER_ITEM_TIMEOUT_SECONDS = DEFAULT_BUDGET_S
_CONCURRENCY = 3
#: An authored ``job_timeout_s`` (now: this image's budget) is clamped here.
_MIN_AUTHORED_BUDGET_S = 60.0

# Frame-shaped default for the self-hosted mflux path. Stills are placed
# ``object-fit: contain`` on a 16:9 canvas (2560x1440), so a square still left
# ~44% of the frame width as blurred backdrop. 512x320 (1.6:1) is the valid
# shape closest to 16:9 at the backend's 512 long-edge cap: every hop accepts
# it unchanged (ACG ``_clamp_dim`` /64, gateway 256..2048, backend
# ``multiple_of=64``, ``_aspect_clamp`` no-op at long edge 512), and it is
# 163840 px vs 512²=262144, so generation cannot get slower. 16:9 itself is
# unreachable: 288 is not a multiple of 64 and every 16:9 request (1024x576,
# 1280x720) is floored to 512x256 (2:1) by the backend clamp.
#
# ⛔ DO NOT RAISE THIS LINE TO GET SHARPER STILLS. IT CANNOT WORK, AND IT MAKES
# THEM WORSE. The real ceiling is NOT here — it is ``IMAGE_GEN_MAX_DIM=512``,
# live on mini1, sourced from the gitops ConfigMap ``image-gen-backend-env``:
#   the deployment manifests:76
# ``flux_service._aspect_clamp`` scales the LONG edge down to that cap, so a
# bigger ask is not honoured, it is RESHAPED. Worked example (this exact revert):
# asking 896x512 comes back 512x293 — 150016 px against today's 163840, i.e. a
# ~14k-pixel LOSS of real detail plus a worse aspect. Yes, stills are still
# LANCZOS-upscaled to a 1536 long edge, and yes that is a ~5x blow-up of invented
# detail; this line is not the lever that fixes it.
#
# The ONLY correct order is: raise the ConfigMap cap FIRST, confirm it live on the
# device, THEN raise this constant. That was costed and DECLINED (owner, 2026-09-11):
# ~50-70 s/image against a measured 28.7 s, and a ~16-19 GB peak footprint on a 16 GB
# box already 1.75 GB into swap. Until that budget changes, 512x320 is the best
# shape available and ``test_ai_image_default_aspect`` pins it.
DEFAULT_LOCAL_IMAGE_SIZE = "512x320"
# Providers that take the default. A cloud provider (openai_images /
# gpt-image-1) only accepts its own fixed size set, so it keeps its own default.
_DEFAULT_SIZE_PROVIDERS = frozenset({"local_image"})


class AiImageGenInput(StageInput):
    pass  # All inputs come from ctx.config_snapshot["ai_image_gen"].


class GeneratedImage(StageOutput):
    id: str
    # When the prompts came from a per-line source (``prompts_from_stage``),
    # carry the line_id so promo_render can key each image to its narration
    # line instead of distributing round-robin. None for generic/static prompts.
    line_id: str | None = None
    prompt: str
    asset_id: str | None
    image_url: str | None
    provider_used: str | None
    status: str


class AiImageGenOutput(StageOutputWithNotes):
    # Every "nothing to generate" path in this stage raises StageSkippedError
    # (not requested; opted in but no prompts resolvable), so a COMPLETED
    # ai_image_gen holding zero images means the provider loop ran and returned
    # nothing — `video_render`/`shot_assembly` then quietly fall back to text
    # cards for every ai_image line. Degrade, never fail: the render is still
    # shippable without them.
    produces: ClassVar[tuple[str, ...]] = ("images",)

    images: list[GeneratedImage]
    # Per-line prompts NOT rendered because the render shows real media, motion
    # graphics or AI video on that line instead (generic only — see
    # ``GenericAiImageGenStage``). Empty whenever those opt-in stages are off.
    skipped_covered_line_ids: list[str] = Field(default_factory=list)
    # Images copied from the PARENT run's ai_image_gen output on a regenerate
    # (same line_id + final prompt + job config ⇒ same content-addressed id), vs
    # images this run paid the provider for. ``reused`` (this run's own store,
    # i.e. a resume) counts as neither.
    carried_forward_count: int = 0
    rendered_count: int = 0


class AiImageGenStage(StageHandler[AiImageGenInput, AiImageGenOutput]):
    name = "ai_image_gen"
    input_model = AiImageGenInput
    output_model = AiImageGenOutput
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (SlotUse("image", when="brief.ai_image_gen"),)

    def __init__(self, client: AIContentGenerationClient | None = None) -> None:
        self._client = client or AIContentGenerationClient()

    async def handle(
        self,
        payload: AiImageGenInput,
        ctx: RunContext,
    ) -> AiImageGenOutput:
        # Stage is opt-in: skip (not fail, not complete-empty) unless the run
        # config has an `ai_image_gen` block. Raising StageSkippedError lets the DAG
        # executor mark the stage 'skipped' and the run still reach 'completed'.
        # Returning an empty result instead was the old lie: a run with no
        # provider config completed green holding zero images, and video_render
        # then produced a text-only placeholder.
        cfg = ctx.config_snapshot.get("ai_image_gen")
        if not cfg:
            raise StageSkippedError(
                "ai_image_gen: no ai_image_gen config block (opt-in stage not requested)",
                missing=["ai_image_gen"],
            )

        prompts = self._apply_clip_notes(self._resolve_prompts(cfg, ctx), ctx)
        if not prompts:
            raise StageSkippedError(
                "ai_image_gen: opted in but no prompts could be resolved",
                missing=["ai_image_gen.prompts"],
            )
        prompts, skipped_covered = self._drop_covered(prompts, ctx)

        job_config = self._build_job_config(cfg)
        # Frame-shaped default when no size is authored: an authored size always
        # wins, and only the local (mflux) provider is defaulted -- cloud providers
        # accept only their own fixed sizes. Inlined rather than a helper so this
        # adds no further `dict[str, Any]` signature (see tests/test_any_ratchet.py),
        # and NOT folded into _build_job_config, whose image/video parity is pinned
        # by tests/services/stages/generic/test_job_config.py.
        if "size" not in job_config and (
            job_config.get("provider") is None
            or job_config.get("provider") in _DEFAULT_SIZE_PROVIDERS
        ):
            job_config = {**job_config, "size": DEFAULT_LOCAL_IMAGE_SIZE}
        # An authored budget may shorten the wait, never outlive the per-item
        # budget (beyond it stage_gather would cancel the call mid-poll instead).
        budget_s = max(
            min(
                float(cfg.get("job_timeout_s") or _PER_ITEM_TIMEOUT_SECONDS),
                _PER_ITEM_TIMEOUT_SECONDS,
            ),
            _MIN_AUTHORED_BUDGET_S,
        )
        run_id = str(ctx.run_id)
        # Stable key over the generative inputs (config affects the output), so
        # the same (prompt, config, line) always addresses the same stored image.
        job_key = json.dumps(job_config, sort_keys=True)
        parent_images = await self._parent_images(ctx)

        async def _generate(item: tuple[str, str | None]) -> GeneratedImage:
            prompt, line_id = item
            media_id = deterministic_media_id(prompt, job_key, line_id)
            # Regenerate child: the parent already rendered this exact image
            # (unchanged line, prompt and config) — carry it forward, unpaid.
            carried = parent_images.get(media_id)
            if carried is not None:
                return carried.model_copy(update={"status": CARRIED_FORWARD})
            # Resume/retry idempotency: if this exact image was already generated
            # and republished (keyed by media_id below) this run, reuse the stored
            # asset instead of paying for another ACG generation. A hit means the
            # fetchable object exists, so it also skips the re-download/re-store.
            existing = await find_existing_media(run_id, self.name, media_id, "png")
            if existing is not None:
                return GeneratedImage(
                    id=media_id,
                    line_id=line_id,
                    prompt=prompt,
                    asset_id=None,
                    image_url=existing,
                    provider_used=None,
                    status="reused",
                )
            # Idempotency: the per-item key (run + stage + content-addressed
            # media_id) makes ACG return the ORIGINAL job on the helper's retry,
            # so a re-submit never starts a SECOND paid generation.
            result = await generate_concept_image(
                self._client,
                prompt,
                job_config,
                idempotency_key=f"{run_id}:{self.name}:{media_id}",
                budget_s=budget_s,
            )
            return _to_generated_image(prompt, result, image_id=media_id, line_id=line_id)

        images = await stage_gather(
            prompts,
            _generate,
            concurrency=_CONCURRENCY,
            per_item_timeout_seconds=_PER_ITEM_TIMEOUT_SECONDS,
            # The helper already retried with the same key inside the budget; a
            # second stage_gather attempt would wait the whole budget again.
            per_item_retries=0,
            # NOT the default ``(Exception,)``: a PermanentUpstreamError (e.g. a
            # byo_required tenant's ``SlotNotConfiguredError``) must propagate
            # unwrapped, not buried in StageRetriesExhaustedError.
            retry_on=(RuntimeError, OSError, ValueError, httpx.HTTPError, TransientUpstreamError),
            stage_name="ai_image_gen",
        )
        images = await self._republish_relative(images, run_id=run_id)
        # Validate the downstream's response: a provider can 200 with a failed
        # payload (no/`"None"` url) for an individual prompt. Trusting that is how
        # dead images reached video_render. Fail the stage loud instead. Checked
        # AFTER the gather (not inside `_generate`) so a deterministic bad
        # response is not pointlessly retried, and the error names this stage.
        for img in images:
            url = img.image_url
            if not url or str(url).strip().lower() in {"none", "null"}:
                raise StageDownstreamResponseError(
                    f"ai_image_gen: provider returned no usable image_url for "
                    f"prompt {img.prompt!r} (status={img.status!r})"
                )
        carried_n = sum(1 for img in images if img.status == CARRIED_FORWARD)
        rendered_n = sum(1 for img in images if img.status not in {CARRIED_FORWARD, "reused"})
        if parent_images:
            log.info(
                "ai_image_gen: %d image(s) carried forward from the parent run, %d rendered",
                carried_n,
                rendered_n,
            )
        return AiImageGenOutput(
            images=images,
            skipped_covered_line_ids=skipped_covered,
            carried_forward_count=carried_n,
            rendered_count=rendered_n,
        )

    def _drop_covered(
        self, prompts: list[tuple[str, str | None]], ctx: RunContext
    ) -> tuple[list[tuple[str, str | None]], list[str]]:
        """Never pay for a still the render will throw away.

        Drops per-line prompts whose line already shows another visual; unkeyed
        prompts (round-robin pool) are never dropped. Raises StageSkippedError
        when every prompt is covered. Returns ``(kept, skipped_line_ids)``.
        """
        covered = self._covered_line_ids(ctx)
        skipped = sorted(
            {line_id for _, line_id in prompts if line_id is not None and line_id in covered}
        )
        if not skipped:
            return prompts, []
        kept = [(p, lid) for p, lid in prompts if lid is None or lid not in covered]
        log.info(
            "ai_image_gen: skipping %d line(s) covered by real media / motion "
            "graphics / AI video: %s",
            len(skipped),
            skipped,
        )
        if not kept:
            raise StageSkippedError(
                f"ai_image_gen: every line ({len(skipped)}) is covered by real "
                f"media / motion graphics / AI video; no stills to render "
                f"(skipped line_ids: {skipped})",
            )
        return kept, skipped

    def _covered_line_ids(self, ctx: RunContext) -> frozenset[str]:
        """Lines whose render shows another visual instead of this stage's still.

        Base (promo / course): none — their renders apply their own rules.
        """
        del ctx
        return frozenset()

    async def _parent_images(self, ctx: RunContext) -> dict[str, GeneratedImage]:
        """Parent-run images reusable by content-addressed id. Base: none."""
        del ctx
        return {}

    async def _republish_relative(
        self, images: list[GeneratedImage], *, run_id: str
    ) -> list[GeneratedImage]:
        """Re-publish ACG-relative image URLs to fetchable artifact-store URLs.

        ACG returns a license-gated, ACG-relative download path
        (``/api/concept-images/assets/{id}/download``) that the anonymous
        video-compile service cannot fetch — it treats the path as a local file
        and ffmpeg fails. Download the bytes (the client carries the licence
        token) and re-publish to the run store, mirroring slides/site-capture.
        Best-effort: a failure leaves the original URL (consumers that can't
        fetch it degrade to their fallback). Absolute http(s) URLs pass through.
        """
        out: list[GeneratedImage] = []
        for img in images:
            url = (img.image_url or "").strip()
            if url.startswith("/"):
                try:
                    data = await self._client.download_concept_image(url)
                    # Key the stored object by the item's deterministic media id
                    # (img.id) so a resumed run's find_existing_media locates it
                    # and skips regeneration.
                    fetchable = await store_generated_image(
                        run_id=run_id,
                        stage_name=self.name,
                        image_id=img.id,
                        image_bytes=data,
                        image_format="png",
                    )
                    out.append(img.model_copy(update={"image_url": fetchable}))
                    continue
                except Exception as exc:  # noqa: BLE001 -- best-effort republish; keep original on failure
                    log.warning("ai_image_gen: republish of %s failed: %s", url, exc)
            out.append(img)
        return out

    @staticmethod
    def _resolve_prompts(cfg: dict[str, Any], ctx: RunContext) -> list[tuple[str, str | None]]:
        """Return ``(prompt, line_id)`` pairs. ``line_id`` is set only when the
        prompts come from a per-line source (``cfg["prompts_from_stage"]`` names a
        prior stage whose output has ``prompts: [{line_id, prompt}, ...]``), so the
        images can be keyed to their narration line; otherwise it is None and the
        static ``cfg["prompts"]`` / brief fallback is used."""
        from_stage = cfg.get("prompts_from_stage")
        if from_stage:
            prior = ctx.prior_output(str(from_stage)) or {}
            pairs = [
                (str(p["prompt"]), str(p["line_id"]) if p.get("line_id") else None)
                for p in (prior.get("prompts") or [])
                if isinstance(p, dict) and p.get("prompt")
            ]
            if pairs:
                return pairs
        raw = cfg.get("prompts")
        if isinstance(raw, list) and raw:
            return [(str(p), None) for p in raw if p]
        single = cfg.get("prompt")
        if single:
            return [(str(single), None)]
        full_cfg = ctx.config_snapshot
        brief = full_cfg.get("brief") or {}
        title = full_cfg.get("title") or "concept image"
        angle = brief.get("angle") or ""
        text = f"{title}. {angle}".strip(". ")
        return [(text, None)] if title else []

    @staticmethod
    def _apply_clip_notes(
        prompts: list[tuple[str, str | None]], ctx: RunContext
    ) -> list[tuple[str, str | None]]:
        """Fold per-clip reviewer feedback into the matching line's prompt.

        A single-clip regenerate (``build_regen_clip_state``) folds the clip's
        open feedback into this stage keyed by ``line_id``. Appending it to that
        one line's prompt changes its ``deterministic_media_id`` → a cache miss
        that regenerates only that image, while every other line's prompt is
        unchanged → its stored image is reused. Notes whose ``element_id`` does
        not match a per-line prompt (e.g. static round-robin prompts) are no-ops.
        """
        notes = ctx.annotations_for("ai_image_gen")
        if not notes:
            return prompts
        by_line: dict[str, list[str]] = {}
        for n in notes:
            eid = n.get("element_id")
            text = str(n.get("text") or "").strip()
            if eid and text:
                by_line.setdefault(str(eid), []).append(text)
        if not by_line:
            return prompts
        out: list[tuple[str, str | None]] = []
        for prompt, line_id in prompts:
            texts = by_line.get(line_id) if line_id else None
            if texts:
                revision = " ".join(texts)
                out.append((f"{prompt}\n\nApply this reviewer feedback: {revision}", line_id))
            else:
                out.append((prompt, line_id))
        return out

    @staticmethod
    def _build_job_config(cfg: dict[str, Any]) -> dict[str, Any]:
        # Shared base keys (provider/style/negative_prompt/extra_params) live in
        # _job_config so they can't drift from ai_video_gen; only image-specific
        # extras are declared here.
        return build_job_config(
            cfg,
            (
                "size",
                "quality",
                "reference_image_urls",
                "target_usage",
                "usage_context",
            ),
        )


def _to_generated_image(
    prompt: str, result: AcgConceptImageJob, *, image_id: str, line_id: str | None = None
) -> GeneratedImage:
    return GeneratedImage(
        id=image_id,
        line_id=line_id,
        prompt=prompt,
        asset_id=result.asset_id,
        image_url=result.image_url or result.download_url,
        provider_used=result.provider,
        status=result.status,
    )


#: ``GeneratedImage.status`` of an image copied from the parent run's output.
CARRIED_FORWARD = "carried_forward"


def make_generic_ai_image_gen_stage(deps: dict) -> StageHandler:
    """Blueprint factory: wire the dispatcher's ``session_factory`` in.

    ``.get`` so a deps-less build (tests, the requirements manifest) still
    constructs; without a factory the stage simply never carries forward.
    """
    return GenericAiImageGenStage(session_factory=deps.get("session_factory"))


class GenericAiImageGenStage(AiImageGenStage):
    """``ai_image_gen`` for the generic blueprint: skip lines video_render covers.

    ``video_render`` discards the AI still on any line carrying real media,
    motion graphics or AI video. Coverage comes from
    :func:`~src.services.visual_media.assign_generic_line_media` — the SAME
    function the render uses — so the two can never disagree. The generic
    blueprint makes this stage depend on those opt-in stages so their outputs
    exist when it runs; with all of them skipped nothing is covered and the
    stage behaves exactly like :class:`AiImageGenStage`.
    """

    def __init__(
        self,
        client: AIContentGenerationClient | None = None,
        *,
        session_factory=None,  # noqa: ANN001 -- house pattern (see footage_probe)
    ) -> None:
        super().__init__(client)
        self._session_factory = session_factory

    def _covered_line_ids(self, ctx: RunContext) -> frozenset[str]:
        return assign_generic_line_media(ctx).covered_line_ids

    async def _parent_images(self, ctx: RunContext) -> dict[str, GeneratedImage]:
        """The parent run's ai_image_gen images, keyed by content-addressed id.

        A regenerate child (``runs.parent_run_id``) re-runs this stage whenever
        an upstream media stage re-runs, because coverage may have moved. Only
        newly-uncovered lines or changed prompts need a render: every other
        line's ``deterministic_media_id(prompt, job_config, line_id)`` matches an
        image the parent already holds. Reads the parent's ``stage_outputs`` row
        — the same rows ``regenerate`` preserves from — so a chain of regenerates
        carries the original URL through. Only fetchable images whose object is
        not DEFINITIVELY gone (``artifact_exists`` is False only when absent) are
        offered. Any failure → ``{}`` (or, per image, that line is skipped) → render
        as before; never fails the stage. Not on a regenerate the user aimed at THIS
        stage (``config_snapshot[REGENERATE_FROM_STAGE_KEY]``): that asks for new
        variants, so every line renders.
        """
        if self._session_factory is None:
            return {}
        if ctx.config_snapshot.get(REGENERATE_FROM_STAGE_KEY) == self.name:
            # The user explicitly re-ran THIS stage: they want fresh variants of
            # unchanged prompts, so nothing is carried forward.
            return {}
        try:
            async with self._session_factory() as session:
                await bind_session_tenant(session, ctx.tenant_id)
                parent_run_id = (
                    await session.execute(select(Run.parent_run_id).where(Run.id == ctx.run_id))
                ).scalar_one_or_none()
                if parent_run_id is None:
                    return {}
                row = await StageOutputRepository(session).get(parent_run_id, self.name)
                if row is None or row.status != "completed" or not row.output:
                    return {}
                parent_output = AiImageGenOutput.model_validate(row.output)
        except Exception as exc:  # noqa: BLE001 -- best-effort reuse; render on any failure
            log.warning("ai_image_gen: could not read parent images: %s", exc)
            return {}
        out: dict[str, GeneratedImage] = {}
        for img in parent_output.images:
            url = str(img.image_url or "")
            if not url.startswith(("http://", "https://")):
                continue
            try:
                exists = await artifact_exists(url)
            except Exception as exc:  # noqa: BLE001 -- unverifiable → render that line
                log.warning("ai_image_gen: could not check parent image %s: %s", img.id, exc)
                continue
            if exists is False:
                continue
            out[img.id] = img
        return out
