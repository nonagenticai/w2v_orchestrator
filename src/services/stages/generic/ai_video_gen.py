"""ai_video_gen — dispatch text/image-to-video generation to ACG.

Generative-video sibling of ``ai_image_gen``. Calls the ACG service's
``/api/content/videos/generate`` endpoint, which fans the request out to
whichever provider is registered (Runway, Kling, Veo, Sora, Seedance,
Wan locally, or the deterministic ``stub`` provider when
``VIDEO_PROVIDER=stub`` is set on ACG).

Reads ``ai_video_gen`` config from ``ctx.config_snapshot``:

  ai_video_gen:
    prompts:          # list[str]; required-by-default. If absent, falls
                      # back to a single prompt derived from the brief.
    provider:         # optional override; ACG defaults to whatever
                      # ``generation.default_provider`` resolved to.
    duration_s: 5.0   # seconds
    width: 1280
    height: 720
    style:            # optional
    image_url:        # optional reference image for image-to-video
    fallback_providers: ["kling", "seedance"]   # optional retry chain
    extra_params: {...}

Output: one ``GeneratedVideo`` per prompt with a stable ``id`` for
annotation pinning. Per-prompt jobs run via ``stage_gather`` with bounded
concurrency so a multi-clip brief renders in parallel groups.

Stage is OPT-IN: if no ``ai_video_gen`` block is set on the run config
snapshot, the stage returns an empty result without making any HTTP
call. Blueprints that wire the stage but configure no per-run input are
zero-cost.

Stage is also CAPABILITY-GATED: opting in does not mean the deployment can
serve it. Before submitting anything NEW the stage probes ACG's live provider
registry (``GET /api/content/videos/providers``, which only lists providers
whose credentials/health check passed at ACG start-up) and skips when no
capable provider exists — the generation request never reaches the service.
The gate is deliberately downstream of the resume/reuse lookup: a clip that was
already generated and stored is already paid for, so a resumed run hands it
back whatever the registry says today.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Any, ClassVar

import httpx
from w2v_common.stages import stage_call, stage_gather

from src.clients.ai_content_generation_client import AIContentGenerationClient
from src.clients.schemas.ai_content_generation import (
    AcgVideoGenerationResult,
    AcgVideoJobStatus,
)
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
from src.services.stages.upstream import (
    PermanentUpstreamError,
    TransientUpstreamError,
    guarded,
)
from src.services.tts_storage import (
    deterministic_media_id,
    find_existing_media,
    store_generated_video,
)

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

log = logging.getLogger(__name__)

_DEFAULT_JOB_TIMEOUT_S = 600.0  # Generative video is slow; 10 min default.
_PER_ITEM_TIMEOUT_SECONDS = 1200.0
_CONCURRENCY = 2
# The capability probe is a cached (60s server-side) registry read; it must never
# be the thing that stalls a run.
_PROVIDER_PROBE_TIMEOUT_S = 15.0
#: Ceiling on the WHOLE re-publish pass (#1111). Each download may sit out a busy
#: lane for up to ``GPU_BUSY_DEADLINE_SECONDS`` (300s) and they run in sequence, so
#: without this N clips could wait N x 300s. Equal to ``_DEFAULT_JOB_TIMEOUT_S``
#: (one generation job's budget) and half of ``_PER_ITEM_TIMEOUT_SECONDS``; what
#: does not finish in time keeps its original URL (the best-effort path below).
_REPUBLISH_TIMEOUT_S = 600.0
# One retry on a TRANSIENT probe failure (see _provider_gate_reason): enough to ride
# out a single 503/connection blip without turning a 15s probe into a minute.
_PROVIDER_PROBE_RETRIES = 1
_PROVIDER_PROBE_BACKOFF_S = 0.5
# House convention (llm_budget.LLM_RETRY_ON): an ALLOW-list, so PermanentUpstreamError
# — which deliberately does not subclass RuntimeError — can never be swallowed here.
_PROVIDER_PROBE_RETRY_ON: tuple[type[BaseException], ...] = (
    TransientUpstreamError,
    httpx.TransportError,
)


class AiVideoGenInput(StageInput):
    pass  # All inputs come from ctx.config_snapshot["ai_video_gen"].


class GeneratedVideo(StageOutput):
    id: str
    prompt: str
    asset_id: str | None
    video_url: str | None
    duration_s: float | None
    provider_used: str | None
    status: str


class AiVideoGenOutput(StageOutputWithNotes):
    # All three "not applicable" paths (no config, provider gate blocked, no
    # prompts) raise StageSkippedError, so a COMPLETED stage with zero videos is
    # a provider loop that produced nothing. `visual_media.collect_ai_video_urls`
    # then returns [] and the render silently loses every ai_video shot.
    produces: ClassVar[tuple[str, ...]] = ("videos",)

    videos: list[GeneratedVideo]


class AiVideoGenStage(StageHandler[AiVideoGenInput, AiVideoGenOutput]):
    name = "ai_video_gen"
    input_model = AiVideoGenInput
    output_model = AiVideoGenOutput
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (
        SlotUse("video_gen", when="brief.ai_video_gen"),
    )

    def __init__(self, client: AIContentGenerationClient | None = None) -> None:
        self._client = client or AIContentGenerationClient()

    async def handle(
        self,
        payload: AiVideoGenInput,
        ctx: RunContext,
    ) -> AiVideoGenOutput:
        # Opt-in: skip (not fail, not complete-empty) unless the run config has an
        # `ai_video_gen` block. StageSkippedError → the DAG marks this 'skipped' and
        # the run still reaches 'completed'. See ai_image_gen for why returning an
        # empty result was the old silent-success lie.
        cfg = ctx.config_snapshot.get("ai_video_gen")
        if not cfg:
            raise StageSkippedError(
                "ai_video_gen: no ai_video_gen config block (opt-in stage not requested)",
                missing=["ai_video_gen"],
            )

        prompts = self._resolve_prompts(cfg, ctx.config_snapshot)
        job_config = self._build_job_config(cfg)
        job_timeout = float(cfg.get("job_timeout_s") or _DEFAULT_JOB_TIMEOUT_S)
        run_id = str(ctx.run_id)
        job_key = json.dumps(job_config, sort_keys=True)

        # Resume/retry idempotency FIRST. Generative video is the most expensive item
        # in the pipeline; an already-generated+stored clip is already paid for, so it
        # is handed back regardless of what the provider registry says today. Doing the
        # lookup before the capability gate is what stops a resumed run — one whose
        # clips all exist — from being skipped (and its assets discarded) just because
        # ACG lost its provider since. The gate only guards work that is still to do.
        reusable = await self._find_reusable(prompts, run_id=run_id, job_key=job_key)

        # Capability gate: opting in is not the same as being ABLE. Ask ACG which
        # providers actually registered before spending a job on it. Skipped entirely
        # when every prompt already resolves to stored media (nothing to dispatch).
        if not prompts or any(p not in reusable for p in prompts):
            blocked = await self._provider_gate_reason(cfg)
            if blocked is not None:
                raise StageSkippedError(blocked, missing=["ai_video_gen.provider"])

        if not prompts:
            raise StageSkippedError(
                "ai_video_gen: opted in but no prompts could be resolved",
                missing=["ai_video_gen.prompts"],
            )

        async def _generate(prompt: str) -> GeneratedVideo:
            media_id = deterministic_media_id(prompt, job_key)
            existing = reusable.get(prompt)
            if existing is not None:
                return GeneratedVideo(
                    id=media_id,
                    prompt=prompt,
                    asset_id=None,
                    video_url=existing,
                    duration_s=None,
                    provider_used=None,
                    status="reused",
                )
            # Idempotency: stage_gather retries the WHOLE callable on a transient
            # wait/get failure. Without a dedupe key the re-submit would start a
            # SECOND paid provider render for the same prompt (double-bill). The
            # per-item key (run + stage + content-addressed media_id) makes ACG
            # return the ORIGINAL job on the re-submit, so only the wait/get is
            # actually retried. Mirrors demo_video/record.py's start_execution.
            job_id = await self._client.start_generation(
                prompt, job_config, idempotency_key=f"{run_id}:{self.name}:{media_id}"
            )
            status = await self._client.wait_for_generation(job_id, timeout=job_timeout)
            # The status poll returns the THIN envelope (no video_url); the asset
            # URL + duration live on the /result endpoint, which 409s unless the
            # job is fully "completed". Fetch it only then; a partial/errored
            # terminal status yields no result and is caught by the post-gather
            # video_url guard below (matching "validate after gather, don't retry
            # a deterministic bad response").
            result = (
                await self._client.get_generation_result(job_id)
                if status.status == "completed"
                else None
            )
            return _to_generated_video(prompt, status, result, video_id=media_id)

        videos = await stage_gather(
            prompts,
            _generate,
            concurrency=_CONCURRENCY,
            per_item_timeout_seconds=_PER_ITEM_TIMEOUT_SECONDS,
            per_item_retries=1,
            stage_name="ai_video_gen",
        )
        videos = await self._republish_relative(videos, run_id=run_id)
        # Validate the downstream response (see ai_image_gen): a provider can 200
        # with a failed payload carrying no/`"None"` video_url. Fail loud rather
        # than pass a dead clip to the compositor.
        for vid in videos:
            url = vid.video_url
            if not url or str(url).strip().lower() in {"none", "null"}:
                raise StageDownstreamResponseError(
                    f"ai_video_gen: provider returned no usable video_url for "
                    f"prompt {vid.prompt!r} (status={vid.status!r})"
                )
        return AiVideoGenOutput(videos=videos)

    async def _find_reusable(
        self, prompts: list[str], *, run_id: str, job_key: str
    ) -> dict[str, str]:
        """Map prompt → already-stored clip URL, for the prompts that have one.

        Probed concurrently over the DISTINCT prompts (``find_existing_media`` is a
        cheap ``stat_object`` and never raises — an indeterminate probe reads as a
        miss, i.e. regenerate). Duplicate prompts share a ``media_id``, so one probe
        answers for all of them.
        """
        if not prompts:
            return {}
        distinct = list(dict.fromkeys(prompts))
        found = await asyncio.gather(
            *(
                find_existing_media(run_id, self.name, deterministic_media_id(p, job_key), "mp4")
                for p in distinct
            )
        )
        return {p: url for p, url in zip(distinct, found, strict=True) if url is not None}

    async def _provider_gate_reason(self, cfg: dict) -> str | None:
        """Return a skip reason when no capable provider can serve this request.

        ``None`` means "go ahead". The probe reads ACG's live registry
        (``GET /api/content/videos/providers``), which only contains providers
        whose credentials/health check passed at ACG start-up — so an empty list
        is a truthful "this deployment cannot generate video", never a hardcoded
        assumption. Two ways to be blocked:

        1. ACG registered no usable provider at all.
        2. The run pinned specific providers (``provider`` + ``fallback_providers``)
           and NONE of them registered — submitting would burn the whole
           job/poll/timeout budget only to fail.

        PROBE FAILURE is NOT one of them, and is classified the way every other
        upstream call in this repo classifies failure (``stages/upstream.guarded``
        + an ALLOW-list ``retry_on``, mirroring ``llm_budget.LLM_RETRY_ON``):

        * ``PermanentUpstreamError`` (404 — this ACG build has no providers route,
          401/403/422 …): a DEFINITE structural answer that no retry can change.
          Skip clean, with a reason distinct from "registry empty" so the two stay
          diagnosable apart.
        * anything transient (5xx blip, 429/408, connect/read error, timeout):
          re-raised after ``_PROVIDER_PROBE_RETRIES``. A probe timeout is NOT
          evidence that the service is unservable, and swallowing it into a skip
          would silently and TERMINALLY drop the most expensive stage of the run
          with no failure signal and no resume. Failing loud keeps the run's normal
          retry/resume path available.
        """
        try:
            listing = await stage_call(
                lambda: guarded(self._client.list_video_providers),
                timeout_seconds=_PROVIDER_PROBE_TIMEOUT_S,
                retries=_PROVIDER_PROBE_RETRIES,
                backoff_initial=_PROVIDER_PROBE_BACKOFF_S,
                stage_name="ai_video_gen.provider_probe",
                retry_on=_PROVIDER_PROBE_RETRY_ON,
            )
        except PermanentUpstreamError as exc:
            log.warning("ai_video_gen: provider registry is permanently unreadable: %s", exc)
            return (
                f"ai_video_gen: provider registry is unreadable ({exc}); "
                "not dispatching an unservable generation"
            )

        # ACG registry keys are lowercase (``kling``, ``veo``, …) but a run config can
        # pin "Kling"; compare casefolded so a cosmetic case difference is not read as
        # "that provider does not exist".
        available = {name.casefold() for name in listing.available_names}
        if not available:
            return "ai_video_gen: no capable provider configured"

        requested = self._requested_providers(cfg)
        if requested and not (requested & available):
            return (
                "ai_video_gen: no capable provider configured "
                f"(requested {sorted(requested)}, available {sorted(available)})"
            )
        return None

    @staticmethod
    def _requested_providers(cfg: dict) -> set[str]:
        """Providers the run pinned, casefolded (empty set == 'ACG's default is fine')."""
        names: set[str] = set()
        primary = cfg.get("provider")
        if primary:
            names.add(str(primary).casefold())
        fallbacks = cfg.get("fallback_providers")
        if isinstance(fallbacks, list):
            names.update(str(f).casefold() for f in fallbacks if f)
        return names

    async def _republish_relative(
        self, videos: list[GeneratedVideo], *, run_id: str
    ) -> list[GeneratedVideo]:
        """Re-publish ACG-relative video URLs to fetchable artifact-store URLs.

        ACG returns a license-gated, ACG-relative download path
        (``/api/content/videos/jobs/{id}/download``) that the anonymous
        video-compile service cannot fetch — it treats the path as a local file
        and ffmpeg fails. Download the bytes (the client carries the licence
        token) and re-publish to the run store, mirroring
        ``ai_image_gen._republish_relative``. Best-effort: a failure leaves the
        original URL (the render's ``collect_ai_video_urls`` then drops a
        still-unfetchable URL so the compile degrades to its fallback rather
        than 500-ing). Absolute http(s) URLs pass through.
        """
        out: list[GeneratedVideo] = []
        deadline = time.monotonic() + _REPUBLISH_TIMEOUT_S
        for vid in videos:
            url = (vid.video_url or "").strip()
            if url.startswith("/"):
                try:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("re-publish deadline exhausted")
                    data = await asyncio.wait_for(
                        self._client.download_generated_video(url), timeout=remaining
                    )
                    # Key the stored clip by the item's deterministic media id
                    # (vid.id) so a resumed run's find_existing_media locates it.
                    fetchable = await store_generated_video(
                        run_id=run_id,
                        stage_name=self.name,
                        video_id=vid.id,
                        video_bytes=data,
                        video_format="mp4",
                    )
                    out.append(vid.model_copy(update={"video_url": fetchable}))
                    continue
                except Exception as exc:  # noqa: BLE001 -- best-effort republish; keep original on failure
                    log.warning("ai_video_gen: republish of %s failed: %s", url, exc)
            out.append(vid)
        return out

    @staticmethod
    def _resolve_prompts(cfg: dict, full_cfg: dict) -> list[str]:
        raw = cfg.get("prompts")
        if isinstance(raw, list) and raw:
            return [str(p) for p in raw if p]
        single = cfg.get("prompt")
        if single:
            return [str(single)]
        brief = full_cfg.get("brief") or {}
        title = full_cfg.get("title") or "concept video"
        angle = brief.get("angle") or ""
        return [f"{title}. {angle}".strip(". ")] if title else []

    @staticmethod
    def _build_job_config(cfg: dict) -> dict[str, Any]:
        # Shared base keys (provider/style/negative_prompt/extra_params) live in
        # _job_config so they can't drift from ai_image_gen; only video-specific
        # extras are declared here.
        return build_job_config(
            cfg,
            (
                "duration_s",
                "width",
                "height",
                "image_url",
                "fallback_providers",
                "budget_limit",
                "seed",
            ),
        )


def _to_generated_video(
    prompt: str,
    status: AcgVideoJobStatus,
    result: AcgVideoGenerationResult | None,
    *,
    video_id: str,
) -> GeneratedVideo:
    """Build a ``GeneratedVideo`` from the status envelope + optional result.

    ``result`` is ``None`` when the job finished in a non-``completed`` terminal
    state (partial/completed_with_errors) — then there is no asset URL, and the
    post-gather ``video_url`` guard fails the stage loud. Video results carry no
    ``asset_id`` (that is an image-only concept), so it is always ``None``.
    """
    return GeneratedVideo(
        id=video_id,
        prompt=prompt,
        asset_id=None,
        video_url=result.video_url if result else None,
        duration_s=result.actual_duration_s if result else None,
        provider_used=result.provider if result else status.provider,
        status=result.status if result else status.status,
    )
