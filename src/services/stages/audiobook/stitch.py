"""audiobook_stitch — concat narration clips into per-chapter + combined audio.

Terminal stage. Delegates the actual ffmpeg work to video_compiler_service
(``POST /api/concat-audio``) — the orchestrator pod has no ffmpeg. Two passes:
per chapter (chunk clips, no gap) then the whole book (per-chapter files, with
a small inter-chapter pause). The combined file is the run's terminal artifact,
exposed as ``artifact_url`` so the run finalizer can verify it.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar

import httpx
from pydantic import ValidationError
from w2v_common.stages.execution import stage_call

from src.clients.tts_gateway_client import TTSGatewayClient
from src.clients.video_compile import VideoCompileClient
from src.config import get_settings
from src.services.providers.tts_inprocess import tenant_tts_bound
from src.services.stages.audiobook.pacing import resolve_chapter_gap_ms
from src.services.stages.base import (
    ArtifactUrl,
    SlotUse,
    StageDownstreamResponseError,
    StageHandler,
    StageInput,
    StageOutput,
    StagePreconditionError,
    validate_compile_result,
)
from src.services.stages.shot_presign import refresh_shot_asset_urls

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

logger = logging.getLogger(__name__)

_DEFAULT_OUTPUT_FORMAT = "mp3"
_DEFAULT_CHAPTER_GAP_MS = 700
# Mirror the concat_audio() client default; the stage_call outer budget adds a
# small margin over it (matches the course/compile-stage idiom).
_CONCAT_TIMEOUT_SECONDS = 1800.0


class AudiobookStitchInput(StageInput):
    pass


class StitchedChapter(StageOutput):
    id: str
    title: str
    sequence: int
    artifact_url: ArtifactUrl
    seconds: float


class CombinedAudiobook(StageOutput):
    artifact_url: ArtifactUrl
    total_seconds: float


class AudiobookStitchOutput(StageOutput):
    # Declared output contract — enforced once, centrally, in
    # `runs/dag_executor.py::_enforce_produces`. `produces_required` means the
    # stage FAILS instead of degrading. Degrade was the wrong default here: it let
    # the run finish while handing the user a reference to nothing, which is the
    # exact defect class this contract exists to close.
    produces: ClassVar[tuple[str, ...]] = ("artifact_url",)
    produces_required: ClassVar[bool] = True

    chapters: list[StitchedChapter]
    combined: CombinedAudiobook
    # Top-level terminal artifact the run finalizer probes (the combined book).
    artifact_url: ArtifactUrl
    duration_seconds: float
    # ACX-compliance report from the compiler's program master; None unless the
    # brief opted into `audiobook.master` (byte-identical output when absent).
    acx_report: dict[str, float | bool | None] | None = None


def _as_float(value: object) -> float | None:
    """Coerce a brief-supplied value to float, or None if absent/malformed.

    The brief is authored outside this codebase; a bad numeric (e.g.
    ``voice.speed: "fast"`` or ``master.lufs_target: "loud"``) must be IGNORED,
    never raise ``ValueError`` out of the terminal paid stitch stage. Mirrors
    ingest's defensive coercion of ``brief.pronunciations``."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        return float(value)
    except TypeError, ValueError:
        return None


def _advisory_seconds(value: object) -> float | None:
    """The per-clip duration to advertise to the compiler, or None when unknown.

    ``video_compiler_service``'s ``AudioClip.seconds`` is ADVISORY (it always
    re-probes the concatenated output) and constrained ``gt=0``. So a clip whose
    length nobody measured must go over the wire as ``None`` — the honest
    "unknown" — not as a 0.0, which is BOTH a lie (it claims silent audio) and a
    422 that hard-fails the terminal, already-paid stitch stage. Anything absent,
    malformed, non-finite or non-positive collapses to None for the same reason."""
    coerced = _as_float(value)
    if coerced is None or not math.isfinite(coerced) or coerced <= 0:
        return None
    return coerced


def _acx_master_block(audiobook_cfg: dict) -> dict[str, float | bool] | None:
    """Resolve the opt-in `brief.audiobook.master` block into the compiler's
    ``acx_master`` payload, or None when the brief did not opt in.

    Accepts ``master: true`` or ``master: {enabled?, lufs_target?, tp?, lra?}``;
    ``{enabled: false}`` (or absent) ⇒ None ⇒ byte-identical legacy concat."""
    raw = audiobook_cfg.get("master")
    if not raw:
        return None
    if raw is True:
        return {"enabled": True}
    if isinstance(raw, dict):
        if raw.get("enabled") is False:
            return None
        block: dict[str, float | bool] = {"enabled": True}
        for key in ("lufs_target", "tp", "lra"):
            coerced = _as_float(raw.get(key))
            if coerced is not None:
                block[key] = coerced
        return block
    return None


def _build_chapter_markers(
    stitched: list[StitchedChapter], gap_ms: int
) -> list[dict[str, str | int]]:
    """Cumulative m4b chapter markers from the stitched per-chapter durations.

    Chapter i starts at the summed prior durations plus ``i`` inter-chapter gaps
    (the combined concat inserts ``gap_ms`` of silence between chapters), so the
    marker offsets line up with the delivered book. Titles come from the plan."""
    markers: list[dict[str, str | int]] = []
    cursor_ms = 0.0
    for chapter in stitched:
        # The cursor is CUMULATIVE: one fabricated length misplaces this chapter's
        # marker and every marker after it. ``validate_compile_result`` already
        # rejects a non-positive ``duration_seconds`` from the compiler, so this
        # is defence in depth — it makes the cursor structurally incapable of
        # consuming a duration nobody measured, rather than relying on a caller
        # upstream to have checked.
        if not math.isfinite(chapter.seconds) or chapter.seconds <= 0:
            raise StageDownstreamResponseError(
                f"audiobook_stitch: chapter {chapter.id!r} has no measured duration "
                f"(got {chapter.seconds!r}); refusing to place m4b chapter markers from "
                "a fabricated length — every marker from here on would be misplaced."
            )
        markers.append({"title": chapter.title, "start_ms": round(cursor_ms)})
        cursor_ms += chapter.seconds * 1000.0 + gap_ms
    return markers


class AudiobookStitchStage(StageHandler[AudiobookStitchInput, AudiobookStitchOutput]):
    name = "audiobook_stitch"
    input_model = AudiobookStitchInput
    output_model = AudiobookStitchOutput
    # Polls (never synthesizes on) the TTS gateway to re-presign narrate's clips —
    # the same tts requirement audiobook_narrate already declares.
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (SlotUse("tts"),)

    def __init__(
        self,
        client: VideoCompileClient | None = None,
        gateway_factory: Callable[[], TTSGatewayClient] | None = None,
    ) -> None:
        self._client = client or VideoCompileClient()
        self._gateway_factory = gateway_factory or TTSGatewayClient

    async def _fresh_gateway_urls(self, run_id: str) -> dict[str, str]:
        """``{job_id: freshly presigned result_url}`` for this run's gateway narrate batch.

        Gateway-rendered clips live in the TTS gateway's OWN bucket, which
        ``refresh_shot_asset_urls`` (w2v_common ``_is_own_bucket``: own + ``w2v-*``
        only) cannot re-sign — so their URLs keep the 1h TTL from narrate's last
        poll and 403 at concat once QA / a pause / a long book pushes stitch past
        that hour, unrecoverably (a retry replays the same dead URL). The gateway
        presigns on every poll, so re-polling the batch IS the refresh. Same
        predicate narrate used to pick the gateway path; best-effort — a failed
        poll keeps the current URLs (still valid inside the hour).
        """
        if not get_settings().downstream.audiobook_narrate_via_gateway or tenant_tts_bound():
            return {}
        gateway = self._gateway_factory()
        try:
            batch = await gateway.get_batch(f"{run_id}:audiobook_narrate")
        except (httpx.HTTPError, ValidationError) as exc:
            logger.warning(
                "audiobook_stitch run=%s gateway URL refresh failed (%s); keeping URLs",
                run_id,
                type(exc).__name__,
            )
            return {}
        finally:
            await gateway.close()
        return {
            str(j.job_id): str(j.result_url)
            for j in batch.jobs
            if j.status == "done" and j.result_url
        }

    async def handle(
        self,
        payload: AudiobookStitchInput,
        ctx: RunContext,
    ) -> AudiobookStitchOutput:
        # Chapter titles/order come from segment (span mode) or plan; narrate
        # clips carry chapter_id + unit sequence either way.
        plan = ctx.prior_output("audiobook_segment")
        if not plan or not plan.get("chapters"):
            plan = ctx.prior_output("audiobook_plan")
        narration = ctx.prior_output("audiobook_narrate")
        if not plan or not plan.get("chapters"):
            raise StagePreconditionError("audiobook_stitch: no plan/segment output")
        if not narration or not narration.get("clips"):
            raise StagePreconditionError("audiobook_stitch: audiobook_narrate produced no clips")

        # Audiobook output options ride in the brief (run_engine only hoists
        # specific top-level keys into config_snapshot; the full brief is always
        # present under "brief").
        brief = ctx.config_snapshot.get("brief") or {}
        audiobook_cfg = brief.get("audiobook") or {}
        output_format = str(audiobook_cfg.get("output_format") or _DEFAULT_OUTPUT_FORMAT)
        # Inter-chapter gap (L3.1): an explicit `chapter_gap_ms`/`pacing.chapter_ms`
        # override wins; else a brief that opted into pacing (a `direction` or
        # `pacing` block) gets a computed, speed-scaled value; else the legacy
        # constant — so a non-opted run is byte-identical to today. An explicit
        # 0-gap is honoured (the resolver uses `is not None`, not `or`).
        narrator_speed = _as_float((ctx.config_snapshot.get("voice") or {}).get("speed")) or 1.0
        computed_gap = resolve_chapter_gap_ms(audiobook_cfg, speed=narrator_speed)
        gap_ms = computed_gap if computed_gap is not None else _DEFAULT_CHAPTER_GAP_MS
        run_id = str(ctx.run_id)

        # Surgical re-roll (audiobook_qa): prefer a corrected clip over the narrate
        # clip of the same chunk_id — swap only its id/audio_url/seconds, keeping the
        # chapter linkage + ordering. audiobook_qa is opt-in/skip-clean and a skipped
        # stage yields no output, so ``corrected`` is empty and this is a no-op for
        # every non-re-roll run (byte-identical to the pre-re-roll stitch).
        qa_out = ctx.prior_output("audiobook_qa") or {}
        corrected = {
            str(c["chunk_id"]): c
            for c in (qa_out.get("corrected_clips") or [])
            if isinstance(c, dict) and c.get("chunk_id")
        }
        clips_by_chapter: dict[str, list[dict]] = {}
        for clip in narration["clips"]:
            fix = corrected.get(str(clip.get("chunk_id")))
            if fix is not None:
                clip = {
                    **clip,
                    "id": str(fix["id"]),
                    "artifact_url": str(fix["artifact_url"]),
                    "seconds": fix["seconds"],
                }
            clips_by_chapter.setdefault(str(clip["chapter_id"]), []).append(clip)

        chapter_specs = sorted(plan["chapters"], key=lambda c: int(c.get("sequence") or 0))
        logger.info(
            "audiobook_stitch run=%s chapters=%d clips=%d format=%s gap_ms=%d",
            run_id,
            len(chapter_specs),
            len(narration["clips"]),
            output_format,
            gap_ms,
        )
        stitched: list[StitchedChapter] = []
        for chapter in chapter_specs:
            chapter_id = str(chapter["id"])
            clips = sorted(
                clips_by_chapter.get(chapter_id, []),
                key=lambda c: int(c.get("chunk_sequence") or 0),
            )
            if not clips:
                raise StagePreconditionError(
                    f"audiobook_stitch: chapter {chapter_id} has no narrated clips"
                )
            # Narration clip artifact_urls were presigned at synth time with a 1h
            # TTL; a run paused/iterated past that hour hands video_compiler a dead
            # signature → its fetch 403s → concat 500s. Re-mint a fresh TTL right
            # before the POST (the helper's _ASSET_URL_KEYS includes "audio_url").
            fresh_urls = await self._fresh_gateway_urls(run_id)
            chapter_clips = [
                {
                    "id": str(c["id"]),
                    "audio_url": fresh_urls.get(str(c["id"])) or str(c["artifact_url"]),
                    # Advisory only, and `gt=0` on the compiler: an unmeasured
                    # clip (tts_gateway#38 ⇒ narrate `seconds=None`) — or a legacy
                    # narrate output that persisted the fabricated 0.0 — must be
                    # advertised as "unknown", never as a length nobody measured.
                    "seconds": _advisory_seconds(c.get("seconds")),
                }
                for c in clips
            ]
            await refresh_shot_asset_urls(chapter_clips)
            # Bind the loop vars as lambda defaults (the factory runs in this
            # iteration, but this silences ruff B023 and is unambiguously safe).
            result = await stage_call(
                lambda clips=chapter_clips, cid=chapter_id: self._client.concat_audio(
                    clips=clips,
                    output_format=output_format,
                    gap_ms=0,
                    tags={"run_id": run_id, "chapter_id": cid},
                    timeout=_CONCAT_TIMEOUT_SECONDS,
                    # Per-chapter key (N distinct concats per run) — a bare
                    # run_id:stage would dedupe them all to the first chapter's
                    # audio. Stable across a resumable-timeout re-dispatch.
                    idempotency_key=f"{run_id}:audiobook_stitch:chapter:{cid}",
                ),
                timeout_seconds=_CONCAT_TIMEOUT_SECONDS + 120,
                retries=0,  # non-idempotent stitch: a retry would duplicate the concat
                stage_name="audiobook_stitch:chapter",
            )
            validate_compile_result(result, stage="audiobook_stitch")
            logger.info(
                "audiobook_stitch run=%s chapter seq=%d clips=%d -> %.1fs",
                run_id,
                int(chapter.get("sequence") or 0),
                len(clips),
                float(result["duration_seconds"]),
            )
            stitched.append(
                StitchedChapter(
                    id=chapter_id,
                    title=str(chapter.get("title") or "Chapter"),
                    sequence=int(chapter.get("sequence") or 0),
                    artifact_url=str(result["artifact_url"]),
                    seconds=float(result["duration_seconds"]),
                )
            )

        # Per-chapter artifact_urls were just presigned by concat_audio above, but a
        # many-chapter book's early chapters can age while later ones stitch — refresh
        # them too so the combined pass never hands the compiler a near-expired URL.
        combined_clips = [
            {"id": ch.id, "audio_url": str(ch.artifact_url), "seconds": ch.seconds}
            for ch in stitched
        ]
        await refresh_shot_asset_urls(combined_clips)
        # Opt-in ACX program master (A12 seam) + m4b chapter markers apply to the
        # COMBINED book only — the per-chapter concats above stay unmastered. Both
        # are None for a legacy brief ⇒ the request is byte-identical to today.
        acx_master = _acx_master_block(audiobook_cfg)
        chapters = _build_chapter_markers(stitched, gap_ms) if output_format == "m4b" else None
        combined_result = await stage_call(
            lambda: self._client.concat_audio(
                clips=combined_clips,
                output_format=output_format,
                gap_ms=gap_ms,
                acx_master=acx_master,
                chapters=chapters,
                tags={"run_id": run_id, "kind": "combined"},
                timeout=_CONCAT_TIMEOUT_SECONDS,
                # Distinct from the per-chapter keys; stable across a resumable
                # -timeout re-dispatch so the combined book isn't re-concatenated.
                idempotency_key=f"{run_id}:audiobook_stitch:combined",
            ),
            timeout_seconds=_CONCAT_TIMEOUT_SECONDS + 120,
            retries=0,  # non-idempotent stitch: a retry would duplicate the concat
            stage_name="audiobook_stitch:book",
        )
        validate_compile_result(combined_result, stage="audiobook_stitch")
        combined_url = str(combined_result["artifact_url"])
        combined_seconds = float(combined_result["duration_seconds"])
        acx_report = combined_result.get("acx_report")
        logger.info(
            "audiobook_stitch run=%s combined chapters=%d total=%.1fs acx_pass=%s",
            run_id,
            len(stitched),
            combined_seconds,
            (acx_report or {}).get("pass") if acx_report else None,
        )

        return AudiobookStitchOutput(
            chapters=stitched,
            combined=CombinedAudiobook(artifact_url=combined_url, total_seconds=combined_seconds),
            artifact_url=combined_url,
            duration_seconds=combined_seconds,
            acx_report=acx_report,
        )
