"""audiobook_narrate — TTS every planned chunk, persist each clip.

Mirrors the generic/lesson narrate stages: one sync ``TTSClient.synthesize``
per chunk, bytes persisted via ``store_tts_clip``, ``stage_gather`` keeping a
few in flight. Each clip carries its chapter + chunk sequence so ``stitch`` can
reassemble the book in order regardless of completion order.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, ClassVar

import httpx
from pydantic import ValidationError
from w2v_common.stages import stage_gather

from src.clients.schemas.tts_gateway import GatewayJobStatus, SubmitBatchResult
from src.clients.tts_client import TTSClient
from src.clients.tts_gateway_client import TTSGatewayClient, is_byo_refusal
from src.config import get_settings
from src.services.providers.errors import SlotNotConfiguredError
from src.services.providers.tts_inprocess import tenant_tts_bound
from src.services.stages.audiobook.voice_catalog import clamp_speed as _clamp_speed
from src.services.stages.base import (
    ArtifactUrl,
    SlotUse,
    StageHandler,
    StageInput,
    StageOutput,
    StageOutputWithNotes,
    StagePreconditionError,
)
from src.services.stages.primitives.narration import synthesize_and_store
from src.services.stages.primitives.ssml import build_line_ssml

# Outer poll budget for the gateway path — the stage timeout (blueprint, 7500s) sits above it;
# the workers keep draining independently, so a reaped+resumed narrate re-polls.
_GATEWAY_POLL_BUDGET_SECONDS = 7200.0

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

logger = logging.getLogger(__name__)

_PER_ITEM_TIMEOUT_SECONDS = 600.0
# Audiobook chunks are long-form (~2000 chars → 20-40s synth each), and mini2's
# kokoro box serializes all synthesis behind one lock. Deep concurrency here
# just builds a queue that trips the TTS read timeout with zero throughput gain
# (the box renders one chunk at a time regardless). Keep a little pipelining
# margin, not a deep fan-out.
_CONCURRENCY = 2

# tts_backend SynthesizeRequest.text hard cap (its src/models.py Field max_length).
# SSML markup inflates the char count; if a directed line's lowered <speak> document
# would exceed the cap, fall back to the plain text — a slightly less expressive
# render always beats a backend 422 that hard-fails a paid narrate.
_SYNTH_TEXT_MAX_CHARS = 10000

# Bounded retries for the one-shot gateway batch submit. Submit is idempotent (the
# gateway dedupes on (tenant, voice, speed, text)), so a transient transport blip at
# submit — like the poll loop already tolerates — must not tear down the whole
# narrate before a single chunk was even queued.
_GATEWAY_SUBMIT_RETRIES = 3


def _lower_line(
    text: str, pause_before_ms: int, emphasis_indices: list[int], *, ssml_enabled: bool
) -> tuple[str, bool]:
    """Lower one narration line to SSML — shared by the direct and gateway paths.

    Returns ``(synth_text, use_ssml)``. ``build_line_ssml`` returns ``None`` when
    there is nothing to express (no pause, no emphasis) → plain text, byte-identical.
    A lowered document over the backend text cap falls back to plain text so a
    heavily-directed long chunk degrades gracefully instead of 422-ing the render."""
    if not ssml_enabled:
        return text, False
    lowered = build_line_ssml(text, pause_before_ms, emphasis_indices)
    if lowered is None or len(lowered) > _SYNTH_TEXT_MAX_CHARS:
        return text, False
    return lowered, True


def _is_retryable_submit_error(exc: Exception) -> bool:
    """Only a transient failure earns a retry; a permanent one must fail fast.

    Transport errors (connect/read/write/timeout) and a gateway-side 5xx/429 are
    blips a second attempt can clear. Any other 4xx — a malformed chunk payload
    (422), an auth reject (401/403) — is deterministic: retrying it burns the
    backoff sleeps and fails anyway, delaying the paid render's real error.
    A bad base URL (``httpx.InvalidURL``) and a response the client could not
    validate (``pydantic.ValidationError``) are likewise permanent."""
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status >= 500 or status == 429
    return False


async def _submit_batch_resilient(
    gateway: TTSGatewayClient, batch_id: str, chunks: list[dict], run_id: str
) -> SubmitBatchResult:
    """Submit the batch, tolerating a transient transport blip (idempotent submit).

    Mirrors the poll loop's blip-tolerance: a single HTTP hiccup at submit (pod
    churn, a gateway HPA scale-down mid-request) must not tear down the narrate
    before a single chunk was even queued. Bounded to ``_GATEWAY_SUBMIT_RETRIES``;
    a genuinely-down gateway still surfaces as a ``StagePreconditionError``.

    ``httpx.InvalidURL`` (a misconfigured gateway base URL) inherits ``Exception``
    and NOT ``HTTPError``, and ``ValidationError`` comes out of the client's
    ``SubmitBatchResult.model_validate`` — both would otherwise escape raw and
    hard-fail a paid render instead of degrading to the precondition error."""
    for attempt in range(1, _GATEWAY_SUBMIT_RETRIES + 1):
        try:
            return await gateway.submit_batch(batch_id, chunks)
        except (httpx.HTTPError, httpx.InvalidURL, ValidationError) as exc:
            if attempt == _GATEWAY_SUBMIT_RETRIES or not _is_retryable_submit_error(exc):
                raise StagePreconditionError(
                    f"audiobook_narrate: gateway submit failed after {attempt} attempt(s): {exc!r}"
                ) from exc
            logger.warning(
                "audiobook_narrate run=%s gateway submit blip (%s) attempt %d/%d, retrying",
                run_id,
                type(exc).__name__,
                attempt,
                _GATEWAY_SUBMIT_RETRIES,
            )
            await asyncio.sleep(min(2.0 * attempt, 10.0))
    raise StagePreconditionError(  # pragma: no cover — loop returns or raises above
        "audiobook_narrate: gateway submit exhausted retries unexpectedly"
    )


class AudiobookNarrateInput(StageInput):
    pass


class NarratedChunk(StageOutput):
    id: str
    chapter_id: str
    chunk_id: str
    chapter_sequence: int
    chunk_sequence: int
    artifact_url: ArtifactUrl
    #: Measured clip length, or None when NOBODY measured it (the TTS gateway
    #: could not probe the render — tts_gateway#38). None is not 0.0: a zero is a
    #: CLAIM that the clip is silent, and it is a claim every downstream consumer
    #: believes — the compiler's ``AudioClip.seconds`` is ``gt=0`` (so a 0.0 422s
    #: the paid stitch) and the m4b marker cursor would misplace every chapter
    #: after it. Optional so a run resumed from a pre-#38 narrate output validates.
    seconds: float | None = None
    #: Companion flag: the length is UNKNOWN, not zero. Redundant with
    #: ``seconds is None`` by construction, but it is what makes the gap
    #: legible in the persisted stage output and in the API response.
    duration_unmeasured: bool = False
    # Synth provenance (voice id + kokoro speed) so a downstream stage — the
    # listen-back re-roll (audiobook_qa) — can re-synthesize a flagged chunk with
    # a corrected text but the EXACT same (voice, speed). ``voice`` is None when
    # the project voice was unset (backend default). Defaulted so a run resumed
    # from a pre-provenance narrate output still validates (those clips simply
    # cannot be re-rolled).
    voice: str | None = None
    speed: float | None = None


class AudiobookNarrateOutput(StageOutputWithNotes):
    # Declared output contract — enforced once, centrally, in
    # `runs/dag_executor.py::_enforce_produces`. Empty here means the stage
    # reported success having produced nothing, which degrades the run.
    # `audiobook_stitch` (which hard-fails on it) and `audiobook_qa` read `clips`.
    produces: ClassVar[tuple[str, ...]] = ("clips",)

    clips: list[NarratedChunk]


class AudiobookNarrateStage(StageHandler[AudiobookNarrateInput, AudiobookNarrateOutput]):
    name = "audiobook_narrate"
    input_model = AudiobookNarrateInput
    output_model = AudiobookNarrateOutput
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (SlotUse("tts"),)

    def __init__(self, client: TTSClient | None = None) -> None:
        self._client = client or TTSClient()

    async def handle(
        self,
        payload: AudiobookNarrateInput,
        ctx: RunContext,
    ) -> AudiobookNarrateOutput:
        # Span mode (voice acting) emits speaker-attributed units from
        # audiobook_segment in the SAME shape as the plan; prefer it when present,
        # else the char-count plan (chunk/off modes). One clip per unit either way.
        source = ctx.prior_output("audiobook_segment")
        if not source or not source.get("chapters"):
            source = ctx.prior_output("audiobook_plan")
        if not source or not source.get("chapters"):
            raise StagePreconditionError("audiobook_narrate: no plan/segment output")

        voice = ctx.config_snapshot.get("voice") or {}
        run_id = str(ctx.run_id)

        # Per-chunk voice casting from audiobook_cast (voice acting). When absent
        # — voice_acting off, or an old run resumed before the stage existed —
        # every chunk falls back to the single narrator voice.
        cast = ctx.prior_output("audiobook_cast") or {}
        # The narrator voice is cast's CATALOG-CHECKED ``narrator_voice_id`` (it
        # replaces an unknown project voice with the default narrator); the raw
        # project voice only for a run resumed before cast emitted it. Speed is
        # clamped to the gateway/kokoro range [0.5, 2.0] — out of it, ChunkIn 422s
        # NON-retryably and the whole book fails.
        voice_id = cast.get("narrator_voice_id") or voice.get("voice_id")
        speed = _clamp_speed(voice.get("speed"), 1.0)
        perf_by_chunk = {
            str(p["chunk_id"]): p for p in (cast.get("performances") or []) if p.get("chunk_id")
        }

        # Per-chunk performance directives from audiobook_direct (spec 4.1). Absent
        # — no brief.audiobook.direction (the stage skip-cleaned), or an old run
        # resumed before the stage existed — every chunk keeps the cast/voice speed
        # and no SSML, byte-identical to today. rate_hint carries as speed on both
        # paths; pause/emphasis lower to SSML (via _lower_line) on BOTH the direct
        # and the gateway path when NARRATION_SSML_ENABLED (TTSB-3 + #22).
        direct = ctx.prior_output("audiobook_direct") or {}
        directive_by_chunk = {
            str(d["chunk_id"]): d for d in (direct.get("directives") or []) if d.get("chunk_id")
        }

        # Flatten to a single work list so all chunks narrate concurrently
        # across chapters; each item keeps the linkage stitch needs.
        work: list[dict] = []
        for chapter in source["chapters"]:
            for chunk in chapter.get("chunks") or []:
                perf = perf_by_chunk.get(str(chunk["id"]))
                base_speed = float(perf["speed"]) if perf else speed
                directive = directive_by_chunk.get(str(chunk["id"]))
                # rate_hint (kokoro speed) overrides the cast speed when the director
                # set one; None ⇒ keep cast speed (byte-identical).
                rate_hint = directive.get("rate_hint") if directive else None
                work.append(
                    {
                        "chapter_id": str(chapter["id"]),
                        "chapter_sequence": int(chapter.get("sequence") or 0),
                        "chunk_id": str(chunk["id"]),
                        "chunk_sequence": int(chunk.get("sequence") or 0),
                        "text": str(chunk["text"]),
                        "voice": perf["voice_id"] if perf else voice_id,
                        "speed": _clamp_speed(rate_hint, base_speed),
                        "pause_before_ms": int(directive.get("pause_before_ms") or 0)
                        if directive
                        else 0,
                        "emphasis_word_indices": list(directive.get("emphasis_word_indices") or [])
                        if directive
                        else [],
                    }
                )

        logger.info(
            "audiobook_narrate run=%s chunks=%d chapters=%d concurrency=%d",
            run_id,
            len(work),
            len(source["chapters"]),
            _CONCURRENCY,
        )

        # Gateway path (opt-in): submit the whole book to the pooling claim-queue
        # and poll until every chunk is done. Resumable + fans out across workers.
        # P5-1: a run that pinned a TENANT ``tts`` connection skips the claim-queue
        # (its workers synthesize on OUR provider) and takes the sync TTSClient path,
        # which dispatches to the tenant's provider in-process.
        if get_settings().downstream.audiobook_narrate_via_gateway and not tenant_tts_bound():
            return await self._narrate_via_gateway(work, run_id)

        ssml_enabled = get_settings().narration_ssml_enabled

        async def _narrate(item: dict) -> NarratedChunk:
            text = item["text"]
            # TTSB-3: lower the director's leading pause + emphasis into SSML when
            # the flag is on (shared with the gateway path via _lower_line).
            synth_text, use_ssml = _lower_line(
                text,
                item.get("pause_before_ms", 0) or 0,
                item.get("emphasis_word_indices") or [],
                ssml_enabled=ssml_enabled,
            )
            try:
                clip_id, artifact_url, seconds = await synthesize_and_store(
                    self._client,
                    # When use_ssml, this IS the SSML document; deterministic_clip_id
                    # keys on it, so a directed line self-invalidates its clip cache.
                    text=synth_text,
                    # Kokoro voice id, NOT the free-text descriptor; TTS 500s on
                    # an unknown voice. Cast-validated per chunk; None → default.
                    voice=item["voice"],
                    run_id=run_id,
                    stage_name="audiobook_narrate",
                    speed=item["speed"],
                    ssml=use_ssml,
                )
            except Exception as exc:
                # Pinpoint the exact chunk that failed — stage_gather only logs an
                # anonymous retry warning, so without this a trial-run failure is
                # untraceable to its chapter/chunk/text.
                logger.error(
                    "audiobook_narrate run=%s FAILED ch=%d chunk=%d voice=%s chars=%d text=%r: %s",
                    run_id,
                    item["chapter_sequence"],
                    item["chunk_sequence"],
                    item["voice"],
                    len(text),
                    text[:120],
                    exc,
                )
                raise
            logger.debug(
                "audiobook_narrate run=%s ch=%d chunk=%d voice=%s chars=%d -> %.2fs",
                run_id,
                item["chapter_sequence"],
                item["chunk_sequence"],
                item["voice"],
                len(text),
                seconds,
            )
            return NarratedChunk(
                id=clip_id,
                chapter_id=item["chapter_id"],
                chunk_id=item["chunk_id"],
                chapter_sequence=item["chapter_sequence"],
                chunk_sequence=item["chunk_sequence"],
                artifact_url=artifact_url,
                seconds=seconds,
                voice=item["voice"],
                speed=item["speed"],
            )

        clips = await stage_gather(
            work,
            _narrate,
            concurrency=_CONCURRENCY,
            per_item_timeout_seconds=_PER_ITEM_TIMEOUT_SECONDS,
            per_item_retries=1,
            stage_name="audiobook_narrate",
        )
        logger.info(
            "audiobook_narrate run=%s narrated %d/%d chunks, total measured audio %.1fs",
            run_id,
            len(clips),
            len(work),
            # The direct path always measures (synthesize_and_store returns a real
            # duration), but `seconds` is now Optional on the shared model — skip
            # the unknowns rather than summing a None into a lie.
            sum(c.seconds for c in clips if c.seconds is not None),
        )
        return AudiobookNarrateOutput(clips=clips)

    async def _narrate_via_gateway(self, work: list[dict], run_id: str) -> AudiobookNarrateOutput:
        """Submit the book to the TTS gateway claim-queue and poll to completion.

        Idempotent submit + stateless poll: a reaped+resumed narrate re-submits
        (the gateway dedupes, most already ``done``) and re-polls a batch the
        workers kept draining — no synthesis is repeated. Each work item maps to
        its job via the submit response's ``ref → idempotency_key`` so the clip's
        chapter/chunk linkage (which stitch needs) is preserved.
        """
        gateway = TTSGatewayClient()
        try:
            batch_id = f"{run_id}:audiobook_narrate"
            # #22 gateway-path SSML parity: lower the director's pause/emphasis into the
            # chunk text exactly as the direct path does. The gateway carries no ssml
            # flag; the kokoro worker forwards the text verbatim and the backend
            # auto-detects the leading <speak> (is_ssml), so a directed line self-
            # invalidates its content-addressed clip while a plain line — build_line_ssml
            # returns None → raw text — keeps its key (byte-identical, cache-stable).
            ssml_enabled = get_settings().narration_ssml_enabled
            chunks = [
                {
                    "ref": w["chunk_id"],
                    "voice": w["voice"],
                    "speed": w["speed"],
                    "text": _lower_line(
                        w["text"],
                        w.get("pause_before_ms", 0) or 0,
                        w.get("emphasis_word_indices") or [],
                        ssml_enabled=ssml_enabled,
                    )[0],
                }
                for w in work
            ]
            submitted = await _submit_batch_resilient(gateway, batch_id, chunks, run_id)
            key_by_ref = {str(j.ref): j.idempotency_key for j in submitted.jobs if j.ref}
            logger.info(
                "audiobook_narrate run=%s via gateway: submitted %d chunks batch=%s",
                run_id,
                len(work),
                batch_id,
            )

            deadline = time.monotonic() + _GATEWAY_POLL_BUDGET_SECONDS
            backoff = 2.0
            jobs: list[GatewayJobStatus] = []
            while time.monotonic() < deadline:
                # A single transient transport blip (pod churn under cluster load, a
                # gateway HPA scale-down mid-request) must NOT tear down a long,
                # mostly-complete narrate — the batch keeps draining on the workers
                # regardless. Swallow it and re-poll; the deadline is the real bound,
                # and a genuinely-down gateway still surfaces as the "did not
                # complete" precondition error below.
                try:
                    jobs = (await gateway.get_batch(batch_id)).jobs
                except httpx.HTTPError as exc:
                    logger.warning(
                        "audiobook_narrate run=%s gateway poll blip (%s), re-polling",
                        run_id,
                        type(exc).__name__,
                    )
                    await asyncio.sleep(min(backoff, 30.0))
                    backoff *= 1.5
                    continue
                terminal = [j for j in jobs if j.status in ("done", "failed")]
                if jobs and len(terminal) >= len(jobs):
                    break
                await asyncio.sleep(min(backoff, 30.0))
                backoff *= 1.5

            failed = [j for j in jobs if j.status == "failed"]
            refused = next((j for j in failed if is_byo_refusal(j.error)), None)
            if refused is not None:
                # The worker failed the job TERMINALLY on modality_gateway's 402: the
                # tenant is byo_required with no TTS provider. Surface the structured
                # refusal (byo_provider_required / tts), not the worker's prose.
                raise SlotNotConfiguredError("tts", upstream_detail=refused.error)
            if failed:
                raise StagePreconditionError(
                    f"audiobook_narrate: {len(failed)} chunk(s) failed on the gateway: "
                    f"{failed[0].error!r}"
                )

            job_by_key = {j.idempotency_key: j for j in jobs}
            clips: list[NarratedChunk] = []
            unmeasured: list[str] = []
            for item in work:
                key = key_by_ref.get(str(item["chunk_id"]))
                job = job_by_key.get(key) if key else None
                if not job or job.status != "done" or not job.result_url:
                    raise StagePreconditionError(
                        f"audiobook_narrate: chunk {item['chunk_id']} did not complete on the gateway"
                    )
                # `or 0.0` here USED TO re-fabricate, one layer up, exactly the
                # zero tts_gateway#38 stopped fabricating: the gateway now persists
                # SQL NULL + `duration_unmeasured` for a render it could not probe.
                # Carry the GAP instead of inventing a number. The audio is the
                # valuable thing and it is intact, so the clip still ships —
                # failing the run or dropping the clip would cost the user real
                # narration to avoid a misplaced marker. Instead the unknown stays
                # expressible (None) and gets made LOUD below (a stage note ⇒ the
                # run reports `degraded`).
                measured = (
                    None
                    if (job.duration_unmeasured or job.duration_seconds is None)
                    else float(job.duration_seconds)
                )
                if measured is None:
                    unmeasured.append(str(item["chunk_id"]))
                clips.append(
                    NarratedChunk(
                        id=str(job.job_id),
                        chapter_id=item["chapter_id"],
                        chunk_id=item["chunk_id"],
                        chapter_sequence=item["chapter_sequence"],
                        chunk_sequence=item["chunk_sequence"],
                        artifact_url=str(job.result_url),
                        seconds=measured,
                        duration_unmeasured=measured is None,
                        voice=item["voice"],
                        speed=item["speed"],
                    )
                )
            notes: list[str] = []
            if unmeasured:
                # The orchestrator pod has no ffmpeg/ffprobe (see audiobook_stitch's
                # module docstring — that is WHY the concat is delegated to
                # video_compiler_service), so it cannot measure the clip locally.
                # Making the gap loud is the most honest thing available here.
                note = (
                    f"{len(unmeasured)} of {len(clips)} narrated chunk(s) came back from the "
                    "TTS gateway with an UNMEASURED duration; the audio is intact but its "
                    "length is unknown, so m4b chapter markers after it may be misplaced: "
                    + ", ".join(sorted(unmeasured)[:20])
                )
                logger.warning("audiobook_narrate run=%s %s", run_id, note)
                notes.append(note)
            logger.info(
                "audiobook_narrate run=%s via gateway: %d clips, total measured audio %.1fs "
                "(%d unmeasured)",
                run_id,
                len(clips),
                sum(c.seconds for c in clips if c.seconds is not None),
                len(unmeasured),
            )
            return AudiobookNarrateOutput(clips=clips, notes=notes)
        finally:
            await gateway.close()  # BaseServiceClient teardown is close(), not aclose()
