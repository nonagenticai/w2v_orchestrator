"""growth_bundle — typed snippet pack for outbound channels.

Given the rendered output's brief + script, produce LinkedIn / Twitter / email /
landing-page copy with a CTA url.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any, ClassVar, cast
from uuid import uuid7

from pydantic import Field
from w2v_common.stages import StageRetriesExhaustedError, stage_call

from src.observability.metrics import growth_bundle_advisory_errors_total
from src.services.auth_secrets import AUTH_BLOCK_KEYS, SEALED, SECRET_FIELDS
from src.services.llm import WMClient
from src.services.stages.base import (
    SlotUse,
    StageHandler,
    StageInput,
    StageOutput,
    StageOutputWithDegradations,
)
from src.services.stages.demo_video.spine import resolve_plan_output
from src.services.stages.generic.grounding import grounding_sentences
from src.services.stages.llm_budget import (
    LLM_RETRY_ON,
    downshift_reasoning,
    llm_budget_seconds,
    llm_client_timeout_seconds,
    llm_retries,
    output_token_budget,
)
from src.services.stages.shared.accessibility import (
    a11y_report,
    build_a11y_policy,
    silent_gaps,
)
from src.services.stages.shared.accessibility_llm import describe_gaps, rank_findings
from src.services.stages.shared.claim_grounding import analyze_claims
from src.services.stages.shared.degradation import GROWTH_BUNDLE_EMPTY, degradation
from src.services.stages.shared.retention import analyze_retention
from src.services.stages.shared.video_gate import require_video
from src.services.stages.upstream import guarded

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

logger = logging.getLogger(__name__)

# Patience budget for the WM growth-bundle LLM call comes from the shared LLM
# ladder (``PipelineSettings.llm_budget_seconds`` / ``llm_retries``, read via
# ``llm_budget_seconds()`` / ``llm_retries()``). pgx-local vLLM is shared (~5
# sessions + heavy ks-* reasoning workers) and inference-router QUEUES on a
# per-provider concurrency semaphore, so under load this tail stage must WAIT for
# its slot rather than cut itself off and re-queue. The default budget (660s)
# sits above the 600s downstream stack (WM ir_http_read_timeout_s +
# inference-router provider timeout) and below the 700s outer stage
# TimeoutMiddleware, so a router-successful response is captured. Was 300s — the
# lone hop below the 600s stack, which killed growth_bundle while the ks-* callers
# (patient) queued and succeeded on the same vLLM. inference-router already fails
# over to deepseek when pgx is genuinely DEGRADED (>660s x3).
# WM worker name (regex ^[a-zA-Z0-9_-]+$). WM auto-creates it on first call and
# the per-call system_prompt steers it; the provider/model lives on the WM
# worker config, NOT here — that's the whole point of routing through WM.
_WORKER_NAME = "growth-bundle"

# The channels this stage asks for and accepts. Module-level so the OUTPUT budget
# below is DERIVED from the same tuple the validator filters on and the two
# cannot drift; ``GrowthBundleStage._CHANNELS`` aliases it.
_GROWTH_CHANNELS: tuple[str, ...] = ("linkedin", "twitter", "email", "landing_page")

# Worst-case ANSWER tokens for ONE snippet: channel + cta_url + the copy itself.
# The longest channel here is landing_page (a few short paragraphs, ~350 words
# ~= 500 tokens); 600 covers that plus keys, braces and JSON escaping.
_MAX_TOKENS_PER_SNIPPET = 600
# Worst case is one snippet per requested channel — the schema has no other
# array — so 4 * 600 = 2400.
_GROWTH_ANSWER_TOKENS = len(_GROWTH_CHANNELS) * _MAX_TOKENS_PER_SNIPPET
# Explicit per-call OUTPUT budget. Omitting ``max_tokens`` leaves WM's 6000
# default, which a REASONING model can burn entirely on thinking before emitting
# any JSON -> WM 422 -> PermanentUpstreamError -> terminal stage failure (promo
# E2E run 01a0181b died exactly this way after 686 s). ``output_token_budget``
# adds the reasoning headroom on top of the answer estimate and caps at the
# router's chain deadline; see ``stages/llm_budget.py`` for the mechanism.
_GROWTH_MAX_TOKENS = output_token_budget(answer_tokens=_GROWTH_ANSWER_TOKENS)

_SYSTEM_PROMPT = (
    "You are a growth marketer. Produce channel-specific outbound copy "
    "(LinkedIn, Twitter, email, landing page) with a CTA url. "
    "Write one snippet for EVERY channel. Return only JSON matching the schema."
)

_SNIPPET_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "text": {"type": "string", "description": "The finished copy for this channel."},
        "cta_url": {"type": ["string", "null"]},
    },
    "required": ["text"],
}

# Object-rooted structured-output schema (WM convention). WM returns
# ``response.output`` already parsed to this shape.
#
# ONE REQUIRED PROPERTY PER CHANNEL, not a ``snippets`` array. The array shape let
# the model answer ``{"snippets": []}``: showcase spike 01a0fd4d-9948 got exactly
# that, 8 completion tokens on both attempts (worker-management log, 2026-10-02
# 15:55:40Z and 15:55:42Z), and the run went ``failed`` after its video was
# delivered. ``minItems`` cannot fix it: worker-management strips it before building
# the grammar (see ``tests/test_output_schema_bounds.py`` and the note in
# ``action_reel/segment_select.py``). Required properties ARE expressible in the
# grammar, so guided decoding cannot emit an empty bundle, and a channel name can
# no longer drift from ``_GROWTH_CHANNELS`` (the old free-text ``channel`` string
# was matched exactly against it).
_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": dict.fromkeys(_GROWTH_CHANNELS, _SNIPPET_SCHEMA),
    "required": list(_GROWTH_CHANNELS),
}


class _NoUsableSnippetsError(ValueError):
    """The reply parsed but carried no channel with non-blank copy."""


def _parse_snippets(data: object) -> list[GrowthSnippet]:
    """One :class:`GrowthSnippet` per channel whose block carries non-blank text.

    Raises :class:`_NoUsableSnippetsError` when none does, so the retry budget
    re-prompts and, if every draw is empty, the caller can tell this apart from a
    transport or upstream failure."""
    if not isinstance(data, Mapping):
        raise _NoUsableSnippetsError(f"reply is {type(data).__name__}, not an object")
    reply = cast("Mapping[str, object]", data)
    snippets: list[GrowthSnippet] = []
    for channel in _GROWTH_CHANNELS:
        block = reply.get(channel)
        if not isinstance(block, Mapping):
            continue
        fields = cast("Mapping[str, object]", block)
        text = fields.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        cta = fields.get("cta_url")
        snippets.append(
            GrowthSnippet(
                id=str(uuid7()),
                channel=channel,
                text=text,
                cta_url=cta if isinstance(cta, str) and cta.strip() else None,
            )
        )
    if not snippets:
        raise _NoUsableSnippetsError(
            f"no channel in {list(_GROWTH_CHANNELS)} carried copy; reply keys {sorted(reply)}"
        )
    return snippets


#: The ONLY brief fields the copywriter prompt may carry. An allowlist, not a
#: denylist: a login can sit at any depth (``sources[i].auth_info``,
#: ``site_capture``, a list of ``auth_config`` entries), and a key list that strips
#: the known places leaks the next one. Each field is copied only as plain text.
_PROMPT_BRIEF_FIELDS: tuple[str, ...] = (
    "call_to_action",
    "cta_url",
    "value_proposition",
    "audience",
    "angle",
    "description",
    "goals",
)


#: The string subfields a structured prompt value (``{"label": "Book", ...}``) may
#: be reduced to. Every other key of that mapping, a password included, is dropped.
_TEXT_SUBFIELDS: tuple[str, ...] = ("label", "text", "url")


def _one_text(value: object) -> str | None:
    """A single prompt-safe string: a non-blank ``str``, or a mapping's first
    non-blank ``str`` under :data:`_TEXT_SUBFIELDS`. Never anything else."""
    if isinstance(value, str):
        return value if value.strip() else None
    if isinstance(value, Mapping):
        fields = cast("Mapping[str, object]", value)
        for key in _TEXT_SUBFIELDS:
            sub = fields.get(key)
            if isinstance(sub, str) and sub.strip():
                return sub
    return None


def _plain_text(value: object) -> str | list[str] | None:
    """``value`` as text the prompt may show, the ONE coercion every interpolated
    value goes through: a string (see :func:`_one_text`) or a list of them.

    A mapping keeps only an allowlisted text subfield; a sealed ``{"$sealed": ...}``
    blob, a nested list or any other structure is dropped, so no secret can ride in
    under an allowlisted key."""
    if isinstance(value, list):
        items = [t for v in value if (t := _one_text(v)) is not None]
        return items or None
    return _one_text(value)


def _prompt_line(value: object, default: str) -> str:
    """A prompt line's value through :func:`_plain_text`, joined, or ``default``."""
    text = _plain_text(value)
    if isinstance(text, list):
        return "; ".join(text)
    return text if text is not None else default


_SECRET_KEYS = SECRET_FIELDS | AUTH_BLOCK_KEYS


def _scrubbed(value: object) -> object:
    """A copy of a prior stage's output with every secret-bearing key and sealed
    blob removed, at any depth (keys from ``auth_secrets``)."""
    if isinstance(value, Mapping):
        fields = cast("Mapping[str, object]", value)
        if SEALED in fields:
            return None
        return {k: _scrubbed(v) for k, v in fields.items() if str(k) not in _SECRET_KEYS}
    if isinstance(value, list):
        return [_scrubbed(v) for v in value]
    return value


def _prompt_brief(brief: Mapping[str, object]) -> dict[str, object]:
    """The brief as the copywriter sees it, built from :data:`_PROMPT_BRIEF_FIELDS`.

    Secret-free by construction: nothing outside the allowlist is read, and every
    value is plain text. The brand contributes its name only."""
    out: dict[str, object] = {}
    brand = brief.get("brand")
    brand_name = (
        cast("Mapping[str, object]", brand).get("name") if isinstance(brand, Mapping) else brand
    )
    if (name := _one_text(brand_name)) is not None:
        out["brand"] = name
    for field in _PROMPT_BRIEF_FIELDS:
        if (text := _plain_text(brief.get(field))) is not None:
            out[field] = text
    return out


def _delivered_video_url(ctx: RunContext) -> str | None:
    """An ``artifact_url`` some earlier stage of THIS run already produced, or None.

    ``growth_bundle`` is the tail companion of the video blueprints (it depends on
    the delivered-artifact stage), so a URL here means the run's video exists. The
    standalone ``growth_bundle_v1`` blueprint has no prior stages and gets None."""
    for output in ctx.prior_run_outputs.values():
        url = output.get("artifact_url") if isinstance(output, Mapping) else None
        if isinstance(url, str) and url:
            return url
    return None


def _empty_reply_cause(exc: BaseException) -> _NoUsableSnippetsError | None:
    """The :class:`_NoUsableSnippetsError` behind a retries-exhausted error, if any."""
    if not isinstance(exc, StageRetriesExhaustedError):
        return None
    seen: BaseException | None = exc.last_error
    while seen is not None:
        if isinstance(seen, _NoUsableSnippetsError):
            return seen
        seen = seen.__cause__ or seen.__context__
    return None


class GrowthBundleInput(StageInput):
    pass


class GrowthSnippet(StageOutput):
    id: str
    channel: str
    text: str
    cta_url: str | None = None


class GrowthBundleStageOutput(StageOutputWithDegradations):
    snippets: list[GrowthSnippet]
    # Advisory retention / pacing flags (ORCH-0.3). Human-readable strings only,
    # never gating. Deliberately NOT `notes`: a truthy `notes` auto-marks the run
    # `degraded` (api/routes/runs/_shared.py) — these are informational, not a
    # soft failure. Empty by default ⇒ output byte-identical when no beats exist.
    retention_insights: list[str] = Field(default_factory=list)
    # Advisory claim-grounding flags (ORCH-0.7). Same never-gating contract as
    # `retention_insights`: human-readable strings only, NEVER `notes` (a truthy
    # `notes` auto-degrades the run). Populated only when the run carried a `legal`
    # brief block; absent `legal` ⇒ no lint ⇒ empty ⇒ byte-identical output.
    claim_insights: list[str] = Field(default_factory=list)
    # Advisory accessibility flags (ORCH-0.8). Same shape/contract as
    # `retention_insights`: human-readable strings only, never gating, NOT `notes`
    # (a truthy `notes` auto-marks the run degraded). Empty by default ⇒ output
    # byte-identical when the run carries no `accessibility` policy block.
    a11y_insights: list[str] = Field(default_factory=list)

    # A19 Tier-2 advisory LLM layer. Empty by default ⇒ byte-identical unless the
    # run opts into the accessibility critic / audio-description sidecar. Never a
    # render input (the growth bundle is companion marketing/artefact data), so an
    # opted-in run's delivered video is identical to an opted-out one's.
    a11y_critique: dict[str, object] = Field(default_factory=dict)
    audio_description: list[dict[str, object]] = Field(default_factory=list)

    # A11 advisory ship-gate pass-through — the final_critique quality badge +
    # chapters. None/empty by default ⇒ byte-identical when the run did not opt into
    # `ship_gate` (the stage then skipped-clean, its output `{}`). Companion outbound
    # metadata only — never a render input.
    quality_badge: str | None = None
    chapters: list[dict[str, object]] = Field(default_factory=list)


def _attach_ship_gate(result: GrowthBundleStageOutput, ctx: RunContext) -> None:
    """Surface the A11 advisory ship-gate badge + chapters (ORCH-A11). Absent
    ``final_critique`` output (stage skipped-clean ⇒ falsy ``{}``) ⇒ no mutation ⇒
    byte-identical output. Advisory companion metadata only — never gates."""
    final = ctx.prior_output("final_critique")
    if not final:
        return
    badge = final.get("quality_badge")
    if isinstance(badge, str) and badge:
        result.quality_badge = badge
    chapters = final.get("chapters")
    if isinstance(chapters, list):
        result.chapters = [c for c in chapters if isinstance(c, dict)]


def _attach_a11y_insights(
    result: GrowthBundleStageOutput,
    ctx: RunContext,
    cfg: dict[str, Any],
    gen: dict[str, Any] | None,
) -> None:
    """Advisory accessibility insights (ORCH-0.8) — extracted from ``handle`` to
    keep it under the C901 gate. Presence-is-opt-in: only a run carrying an
    ``accessibility`` policy block resolves a policy; every other run returns
    immediately, so the output stays byte-identical. Same never-gates discipline
    as retention — a bug here must NEVER discard the stage's successful result.
    """
    if not cfg.get("accessibility"):
        return
    try:
        policy = build_a11y_policy(cfg)
        shots = (gen or {}).get("lines") or []
        narrate = ctx.prior_output("demo_video_narrate") or {}
        clips = narrate.get("clips") or []
        report = a11y_report(shots, clips, policy)
        result.a11y_insights = report.insights
    except Exception:  # advisory insight, never gates
        logger.warning("growth_bundle: accessibility advisory insight failed", exc_info=True)
        growth_bundle_advisory_errors_total.labels(site="accessibility").inc()


async def _attach_a11y_llm(
    wm: WMClient,
    result: GrowthBundleStageOutput,
    ctx: RunContext,
    cfg: Mapping[str, object],
    gen: Mapping[str, object] | None,
    *,
    timeout_seconds: float,
    retries: int,
) -> None:
    """A19 advisory Tier-2 LLM layer — a11y-critic ranking + audio-description
    sidecar. EXPLICIT opt-in: an ``accessibility`` block WITHOUT a truthy
    ``critic`` / ``audio_description`` key runs only the deterministic Tier-1
    insights, so it stays byte-identical to before A19. Advisory only — the
    growth bundle is companion data, never a render input, so an opted-in run's
    delivered video is identical. Never gates: any failure degrades to empty.
    """
    raw_a11y = cfg.get("accessibility")
    if not isinstance(raw_a11y, Mapping):
        return
    a11y = cast("Mapping[str, object]", raw_a11y)
    want_critic = bool(a11y.get("critic")) and a11y.get("critic") != "off"
    want_ad = bool(a11y.get("audio_description")) and a11y.get("audio_description") != "off"
    if not (want_critic or want_ad):
        return
    try:
        policy = build_a11y_policy(dict(cfg))
        if policy is None:
            return
        # The Tier-1 helpers (a11y_report / silent_gaps) take the pipeline's
        # dict[str, Any] shot/clip JSON; cast at this seam so the caller-facing
        # signature stays Any-free.
        shots = cast("list[dict[str, Any]]", (gen or {}).get("lines") or [])
        narrate = ctx.prior_output("demo_video_narrate") or {}
        clips = cast("list[dict[str, Any]]", narrate.get("clips") or [])
        report = a11y_report(shots, clips, policy)

        # Critic: rank the deterministic findings. Skipped when there is nothing to
        # rank (no contrast/gap issues) so it never burns a WM call on an empty report.
        if want_critic and report.insights:
            result.a11y_critique = await stage_call(
                downshift_reasoning(
                    lambda think: guarded(
                        lambda: rank_findings(wm, report, policy, enable_reasoning=think)
                    )
                ),
                timeout_seconds=timeout_seconds,
                retries=retries,
                retry_on=LLM_RETRY_ON,
                stage_name="growth_bundle_a11y_critic",
            )

        # Audio-description sidecar over the over-threshold silent gaps. Skipped
        # when there are no gaps to describe.
        if want_ad:
            gaps = silent_gaps(clips, min_gap_ms=policy.min_silent_gap_ms)
            if gaps:
                text_by_line = {
                    str(s.get("line_id") or s.get("id") or ""): str(s.get("text") or "")
                    for s in shots
                }
                result.audio_description = await stage_call(
                    downshift_reasoning(
                        lambda think: guarded(
                            lambda: describe_gaps(
                                wm, gaps, text_by_line, ctx=ctx, enable_reasoning=think
                            )
                        )
                    ),
                    timeout_seconds=timeout_seconds,
                    retries=retries,
                    retry_on=LLM_RETRY_ON,
                    stage_name="growth_bundle_audio_describer",
                )
    except Exception:  # advisory layer, never gates the deliverable
        logger.warning("growth_bundle: a11y-llm advisory insight failed", exc_info=True)
        growth_bundle_advisory_errors_total.labels(site="a11y-llm").inc()


class GrowthBundleStage(StageHandler[GrowthBundleInput, GrowthBundleStageOutput]):
    name = "growth_bundle"
    input_model = GrowthBundleInput
    output_model = GrowthBundleStageOutput
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (
        SlotUse("llm"),
        SlotUse("tts"),
    )

    _CHANNELS: ClassVar[tuple[str, ...]] = _GROWTH_CHANNELS

    def __init__(self, client: WMClient | None = None) -> None:
        self._llm_timeout = llm_budget_seconds()
        self._llm_retries = llm_retries()
        self._wm = client or WMClient(timeout=llm_client_timeout_seconds())

    async def handle(
        self,
        payload: GrowthBundleInput,
        ctx: RunContext,
    ) -> GrowthBundleStageOutput:
        # The growth bundle packages a produced video for outbound channels; a
        # run that opted out of the video (generate_video=false) doesn't want it.
        # Flag defaults to True, so blueprints that never set it are unaffected.
        require_video(ctx)
        cfg = ctx.config_snapshot
        brief = cfg.get("brief") or {}
        script = (
            ctx.prior_output("promo_script")
            or ctx.prior_output("app_review_script")
            or ctx.prior_output("course_outline")
            or {}
        )
        if not script:
            # demo_video / showcase have no script stage: their content IS the line
            # spine. Without this the prompt said "Prior content: {}".
            lines = resolve_plan_output(ctx).get("lines") or []
            spoken = [str(line.get("text") or "") for line in lines if isinstance(line, Mapping)]
            narration = [text for text in spoken if text.strip()]
            if narration:
                script = {"narration": narration}
        # EVERY interpolated value goes through the same plain-text coercion
        # (_plain_text / _prompt_brief / _scrubbed): nothing reaches the prompt raw.
        voice = cfg.get("voice")
        descriptor = voice.get("descriptor") if isinstance(voice, Mapping) else None
        prompt = (
            "Generate channel-specific copy for this content: one snippet for EACH of "
            "linkedin, twitter, email and landing_page.\n"
            f"Title: {_prompt_line(cfg.get('title'), '')}\n"
            f"Brief: {_prompt_brief(brief)}\n"
            f"Voice: {_prompt_line(descriptor, 'neutral')}\n"
            f'CTA: "{_prompt_line(brief.get("call_to_action"), "Learn more.")}"\n'
            f"Prior content: {_scrubbed(script)}\n"
        )

        async def _attempt(enable_reasoning: bool | None) -> GrowthBundleStageOutput:
            # Shape-validate inside the retry budget so a reply missing a snippet
            # field (or yielding no valid channels) re-prompts rather than failing
            # the run on the first draw.
            resp = await self._wm.message(
                enable_reasoning=enable_reasoning,
                worker_name=_WORKER_NAME,
                message=prompt,
                system_prompt=_SYSTEM_PROMPT,
                output_schema=_OUTPUT_SCHEMA,
                # See _GROWTH_MAX_TOKENS: passing nothing leaves WM's 6000
                # default, which the thinking model can spend without emitting
                # the JSON — a 422 this stage cannot retry.
                max_tokens=_GROWTH_MAX_TOKENS,
            )
            try:
                snippets = _parse_snippets(resp.output)
            except _NoUsableSnippetsError as exc:
                raise _NoUsableSnippetsError(
                    f"growth_bundle: LLM output missing expected fields: {exc}"
                ) from exc
            return GrowthBundleStageOutput(snippets=snippets)

        try:
            result = await self._call_llm(_attempt)
        except StageRetriesExhaustedError as exc:
            # Every draw parsed but none carried copy. On a run whose video was
            # already delivered, the bundle is a companion: keep the run and record
            # the gap with a code. Any OTHER failure (timeout, upstream error, a
            # standalone run whose copy IS the deliverable) still fails the stage.
            empty = _empty_reply_cause(exc)
            video_url = _delivered_video_url(ctx)
            if empty is None or video_url is None:
                raise
            logger.warning(
                "growth_bundle: no usable channel copy after %d attempts; video %s "
                "already delivered, degrading instead of failing the run",
                exc.attempts,
                video_url,
            )
            result = GrowthBundleStageOutput(
                snippets=[],
                degradations=[
                    degradation(
                        GROWTH_BUNDLE_EMPTY,
                        f"growth_bundle: {exc.attempts} attempt(s) returned no usable "
                        f"channel copy ({empty}); the delivered video is unaffected",
                    )
                ],
            )
        return await self._finish(result, ctx)

    async def _call_llm(
        self, attempt: Callable[[bool | None], Awaitable[GrowthBundleStageOutput]]
    ) -> GrowthBundleStageOutput:
        """The bundle LLM call under the shared retry/downshift ladder."""
        return await stage_call(
            # retries=1 (was 2): under vLLM contention a timed-out attempt just
            # abandons its queue slot and re-queues at the back — thrash that
            # makes the stall worse. One patient attempt (660s) plus a single
            # retry for a genuine FAST error is the right shape under the 700s
            # outer cap; sustained overload spills to deepseek at the router.
            # guarded() + LLM_RETRY_ON: a permanent WM failure (402 no LLM
            # provider, prompt over the token limit) never consumes that one
            # retry — it fails now, carrying the actionable sentence.
            downshift_reasoning(lambda think: guarded(lambda: attempt(think))),
            timeout_seconds=self._llm_timeout,
            retries=self._llm_retries,
            retry_on=LLM_RETRY_ON,
            stage_name="growth_bundle",
        )

    async def _finish(
        self, result: GrowthBundleStageOutput, ctx: RunContext
    ) -> GrowthBundleStageOutput:
        """Attach the advisory insights (retention, claims, ship gate, a11y)."""
        cfg = ctx.config_snapshot

        # Advisory retention insights (ORCH-0.3). When this run produced a
        # demo_video_gen beat list, score it for 31s-cliff / pacing risk and
        # attach the human-readable flags. Purely advisory — never gates. The
        # common case (no upstream demo_video_gen, or no beats) skips entirely, so
        # the output stays byte-identical to before this change.
        # Doctored spine when script_doctor ran, else demo_video_gen; {} when neither
        # (the common non-demo_video case), so the block below skips entirely.
        gen = resolve_plan_output(ctx)
        if gen:
            beats = gen.get("lines")
            if beats:
                # Advisory only: a bug here must NEVER discard the stage's
                # already-successful LLM result. Swallow and log, leaving
                # retention_insights empty.
                try:
                    deliverable = cfg.get("deliverable") or {}
                    report = analyze_retention(
                        beats,
                        weights=cfg.get("retention_weights"),
                        target_seconds=deliverable.get("target_duration_seconds"),
                    )
                    result.retention_insights = report.insights
                except Exception:  # advisory insight, never gates
                    logger.warning(
                        "growth_bundle: retention advisory insight failed",
                        exc_info=True,
                    )
                    growth_bundle_advisory_errors_total.labels(site="retention").inc()

        # Advisory claim-grounding lint (ORCH-0.7). Sibling of the retention block:
        # runs ONLY when the operator supplied a `legal` brief block — an absent
        # `legal` short-circuits so no lint runs and the output stays byte-identical.
        # Flags marketing superlatives ("the best", "#1") in the demo_video_gen
        # lines that no ingested source sentence supports. Purely advisory — never
        # gates; a bug here must NEVER discard the stage's successful LLM result.
        if cfg.get("legal"):
            try:
                sentences = grounding_sentences(ctx)
                gen_out = resolve_plan_output(ctx)
                report = analyze_claims(gen_out.get("lines") or [], sentences)
                result.claim_insights = report.insights
            except Exception:  # advisory insight, never gates
                logger.warning(
                    "growth_bundle: claim-grounding advisory insight failed",
                    exc_info=True,
                )
                growth_bundle_advisory_errors_total.labels(site="claim-grounding").inc()

        _attach_ship_gate(result, ctx)
        _attach_a11y_insights(result, ctx, cfg, gen)
        await _attach_a11y_llm(
            self._wm,
            result,
            ctx,
            cfg,
            gen,
            timeout_seconds=self._llm_timeout,
            retries=self._llm_retries,
        )
        return result
