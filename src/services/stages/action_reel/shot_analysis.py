"""shot_analysis — measure each uploaded source before anything selects from it.

One ``video_analyzer_service`` job per source (submit → poll → fetch result),
fanned out through the shared ``stage_gather`` so a shoot of ten clips does not
serialise into ten sequential minutes. What comes back is the MEASURED signal
the selector needs and an LLM cannot invent:

* ``scene_boundaries`` — the cut points already in the footage. A segment that
  straddles one is a segment that changes subject mid-shot, so the selector is
  shown them explicitly rather than being left to guess from a description.
* ``motion_timeline`` — the 0..1 frame-delta curve. For a watersport/drone reel
  this IS the interest curve: the jump, the carve, the fly-through are motion
  peaks, and the dead air between them is what gets cut.
* ``mean_sharpness`` / ``mean_brightness`` / ``overall_score`` — the quality
  floor. A soft or blown-out clip should lose to a sharp one even when the
  action is comparable.

**This stage pauses** (see the blueprint): it is the first point at which the
run has said anything about the operator's footage, and everything downstream —
which seconds survive, what the reel costs to render — is decided from these
numbers. Catching "the analyzer never saw clip 3" here is minutes; catching it
after the compile is a wasted render.

Soft per source, hard in aggregate. One analyzer failure degrades to a note and
that source simply carries no measurements (the selector can still use it on
duration alone); ALL of them failing is a precondition failure, because a
selection made with zero measured signal is a coin toss wearing a rationale.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, ClassVar

from w2v_common.stages import stage_gather

from src.clients.video_analyzer_client import VideoAnalyzerClient
from src.services.stages.action_reel.sampling import (
    MOTION_SAMPLES_PER_SOURCE as _MOTION_SAMPLES_PER_SOURCE,
)
from src.services.stages.base import (
    SlotUse,
    StageHandler,
    StageInput,
    StageOutput,
    StageOutputWithNotes,
    StagePreconditionError,
)

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

log = logging.getLogger(__name__)

# Per-source analyzer budget + retries. Mirrors the fleet's other stage_gather
# fan-outs (narrate / record: 600s x 1 retry).
#
# ⚠️ ``_PER_ITEM_RETRIES`` is INERT here, deliberately kept for shape. ``stage_gather``
# retries on ``retry_on=(Exception,)``, and ``_analyze_one`` catches ``Exception`` and
# returns an ``AnalyzedSource`` carrying the failure sentence — so nothing ever escapes
# to be retried. The per-source ceiling is therefore ONE attempt: the poll budget below
# plus submit/fetch, hard-bounded by ``_PER_ITEM_TIMEOUT_SECONDS``. The stage budget at
# the bottom of this block is computed from that, NOT from ``(1 + retries)``.
_PER_ITEM_TIMEOUT_SECONDS = 600.0
_PER_ITEM_RETRIES = 1
# How many analyses this stage keeps in flight. This is a MEMORY-SAFETY bound on
# the analyzer, not a throughput knob — it was 4, and 4 is measured-fatal now that
# this stage asks for a real frame budget (see ``_SAMPLING_OVERRIDES`` below).
#
# Measured inside a live ``video-analyzer-service`` pod (limits: cpu=2, memory=4Gi),
# real 4K HEVC drone clips:
#
#     one ffmpeg 4K HEVC seek-decode peaks at    444 MiB RSS
#     pod baseline                               ~400 MiB
#     => cost is ~400 + 444 x concurrent_jobs, LINEAR IN JOB COUNT, and
#        essentially independent of ``max_frames``
#
#     3 concurrent   peak memory.current 1850 MiB / 4096   oom_kill 0   (safe)
#     4 concurrent   OOMKilled, exitCode 137, pod restarted             (fatal)
#
# (The 444 MiB is itself avoidable: ``nproc`` reports 8 inside the 2-CPU limit, so
# ffmpeg spawns 8 frame-threads each holding a full 4K DPB. Pinning ``-threads``
# would cut it sharply — but that is the ANALYZER's code, a different repo and lane.
# Flagged, not fixed here.)
#
# Why 2 and not 3, when 3 measured safe: the analyzer has NO self-protection. Its
# ``PIPELINE_MAX_CONCURRENT_JOBS`` / ``settings.max_concurrent_jobs`` is declared at
# ``video_analyzer_service:src/config.py:65`` and read NOWHERE — no semaphore, no
# queue, no ``limit_concurrency``; ``_kickoff_pipeline`` calls a bare
# ``asyncio.create_task``. So the only thing bounding per-pod concurrency is THIS
# number divided by however many replicas a request happens to land on, and a
# Service does not promise an even split. At 2, worst-case skew is 2-on-one-pod —
# measured safe. At 4 it is 4-on-one-pod — measured fatal. Replica count is not a
# defence either: it drifted 2 -> 3 mid-session with no HPA in the manifest.
#
# And 4 bought nothing to begin with. Measured: 1 job 37.6s, 2 concurrent 83.0s
# EACH — 2.2x wall for 2x work, i.e. marginally worse than serial, because ffmpeg
# already saturates the 2-CPU quota. A pod is effectively serial; only replicas add
# throughput. So this costs no measurable time and removes the OOM.
_CONCURRENCY = 2
# Poll budget inside one attempt. Strictly BELOW the per-item timeout so
# poll_analysis's own clear "did not finish" error surfaces before stage_gather
# cancels the attempt with a bare TimeoutError.
_POLL_TIMEOUT_SECONDS = 540.0

# --- The stage budget, derived from THIS module's own constants ---
#
# The fan-out runs in WAVES: ``stage_gather`` keeps at most ``_CONCURRENCY`` items in
# flight, so wall clock is ``ceil(N / _CONCURRENCY) * per-source ceiling``, not one
# per-source ceiling. And the blueprint's ``default_timeout_seconds`` is enforced
# PER STAGE, not per attempt — ``runs/dag_executor.py`` wraps the whole handler in a
# single ``TimeoutMiddleware`` (one ``asyncio.wait_for``).
#
# The blueprint used to hardcode 1400s with a comment doing PER-SOURCE arithmetic
# ("(1 + 1 retries) * 600s = 1200s ... + 100s headroom"). That number was wrong twice
# over: it assumed a SINGLE wave (right for N <= 4, wrong for every shoot above it),
# and it multiplied by a retry that can never fire (see ``_PER_ITEM_RETRIES``). At the
# operator's real 10-clip corpus the true worst case is 3 waves x 600s = 1800s against
# a 1400s budget. Recorded here so nobody re-derives the old figure.
#
# ⚠️ ``_TIMEOUT_SOURCE_CEILING`` is a BUDGETING assumption, NOT a cap. Unlike
# ``narrate``'s ``MAX_NARRATION_LINES`` — where the items are AUTHORED lines and a
# too-long brief is an authoring mistake worth refusing — N here is the operator's
# UPLOADED FOOTAGE. Silently dropping their clips is strictly worse than overrunning:
# ``footage_probe`` never slices ``sources`` and neither does this stage. A corpus
# larger than the ceiling still RUNS in full; it may simply exceed the budget, and
# ``TimeoutError`` is in ``dag_executor._RESUMABLE_TIMEOUT_EXC``, so the stage is reset
# to pending and RESUMES rather than failing the run.
#
# 12 covers the known 10-clip demo corpus with margin. It used to be described as
# "three full waves at concurrency 4"; at ``_CONCURRENCY = 2`` it is six waves, so the
# derived budget below moves 1900s -> 3700s.
#
# That increase is DELIBERATE and the coupling is kept rather than decoupled. The
# tempting alternative — derive the budget from replicas x the measured 52s per source,
# holding it near 1900s — models a replica count that is not knowable at import and that
# was observed drifting 2 -> 3 with no HPA. Worse, it errs in the dangerous direction:
# this stage has NO per-source checkpoint, so a premature stage timeout resumes and
# RE-SUBMITS every clip, and ``dag_executor`` eventually exhausts the attempt cap and
# fails the run terminally. Overshooting the budget costs latency on a stage that is
# already resumable; undershooting it costs the whole run. Erring high is the safe
# direction, and 3700s is the honest worst case for the submission model actually used.
#
# The cost, stated plainly: a genuinely wedged stage now takes ~1h to fail instead of
# ~30min.
_TIMEOUT_SOURCE_CEILING = 12

#: ceil(ceiling / concurrency) waves x the one-attempt per-source ceiling, + 100s
#: headroom for the presign/store overhead outside the fan-out (the same +100 shape
#: every sibling fan-out stage uses). The blueprint IMPORTS this — a literal there
#: could drift from the constants above, which is exactly how 1400s survived.
SHOT_ANALYSIS_TIMEOUT_SECONDS = (
    int(math.ceil(_TIMEOUT_SOURCE_CEILING / _CONCURRENCY) * _PER_ITEM_TIMEOUT_SECONDS) + 100
)

# What this stage asks the analyzer NOT to do. It reads exactly three things off
# the result — ``duration_seconds``, ``scene_boundaries``, ``quality_metrics``
# (see ``_analyze_one``) — and every flag below is `True` by DEFAULT server-side
# (video_analyzer_service ``src/api/schemas.py`` ``AnalysisRequest``), so an
# override-free submit buys a whole vision pipeline whose output is thrown away:
# the vision-LLM pass, a per-frame tesseract loop, the frame-index/description
# artifact writes — and, because the analyzer's ``_needs_transcript`` is
# ``bool(request.generate_walkthrough)``, a full Whisper transcription too.
# On a ten-clip drone shoot that is minutes of GPU time and real vision spend per
# run for nothing.
#
# ⚠️ Do NOT "restore the defaults" here. If a future consumer of this stage needs
# OCR or frame descriptions, ADD the field it needs to ``AnalyzedSource`` and
# flip the one flag that feeds it — the flags and the fields must move together.
#
# ``compute_quality_metrics`` stays ON, explicitly rather than by default: it is
# the flag that gates ``_run_quality_stage``, i.e. the sharpness / brightness /
# motion-curve numbers ``segment_select`` actually selects on. Naming it here
# means a change to the server's default cannot silently blind this stage.
_ANALYZER_OVERRIDES = {
    "generate_walkthrough": False,
    "generate_ui_inventory": False,
    "generate_frame_index": False,
    "extract_frame_descriptions": False,
    "extract_ocr": False,
    "compute_quality_metrics": True,
}


# --- How finely the analyzer SAMPLES each clip. action_reel only. ---
#
# Without these two keys the submit inherits the fleet ConfigMap
# (`PIPELINE_FRAME_INTERVAL_SECONDS=3.0`, `PIPELINE_MAX_FRAMES=200`), and the
# analyzer's `MIN_FRAMES_FOR_QUALITY=5` floor then narrows a short clip to
# exactly five samples. Measured on the operator's real corpus: a 9.009 s 4K
# drone clip yielded 5 frames -> 4 motion points on a 1.802 s lattice, and
# `scene_boundaries` are a strict SUBSET of the sample timestamps (the analyzer's
# `scene_detector.detect` only ever returns `frames[i].timestamp`), so every cut
# this reel can make is pinned to that lattice. At 132 BPM the beat period is
# 0.455 s, i.e. the reel lands up to four beats off the wave hit, every time.
#
# WHY 0.25 s: it is the coarsest interval that still saturates the frame budget
# below on the SHORTEST clip in the corpus (6.006 s / 0.25 = 24 samples). It is
# a desired-interval FLOOR, not the interval that runs: `max_frames` widens it
# per clip (see below), so 0.25 only binds on clips under ~6 s.
#
# WHY the frame budget is `_MOTION_SAMPLES_PER_SOURCE + 1`, imported rather than
# written as a literal: `segment_select` DOWNSAMPLES the curve to
# `_MOTION_SAMPLES_PER_SOURCE` points before it reaches the LLM, and N frames
# produce N-1 motion points (the curve is per-PAIR). So frame N+2 onwards is
# extracted, paid for, and then thrown away by `_downsample_motion` — the
# selector's localisation is capped by the PROMPT lattice (duration / (cap - 1)),
# not by the analyzer's. Deriving the budget from that constant means raising the
# prompt cap raises the sampling with it instead of silently doing nothing.
#
# MEASURED COST (video-analyzer-service pod, cpu limit 2, 4 GiB, real 4K HEVC
# 23.976 fps clip C0011.MP4, 9.009 s, 67 MB, single job, 2026-08-21):
#
#     interval  eff.lattice  frames  motion pts  wall
#     3.0 (now)     1.802 s       5           4   12.8 s
#     1.0           1.000 s       9           8   24.6 s
#     0.5           0.500 s      18          17   41.3 s
#     0.25          0.250 s      36          35   80.9 s
#     0.25 + this frame budget   0.360 s  25  24   51.7 s   <- what ships
#
# Extraction is one `ffmpeg -ss` seek+decode per frame and is linear at ~2.0 s
# per frame on 4K HEVC; it dominates (46.9 s of the 51.7 s). Because the frame
# budget binds on every clip at or above ~6 s, the cost is duration-INDEPENDENT:
# ~52 s per source whatever the clip's length.
#
# BUDGET HEADROOM (same measurement; `SHOT_ANALYSIS_TIMEOUT_SECONDS` = 3700 s at
# `_CONCURRENCY = 2` — see the derivation above for why the coupling is kept):
#   * Concurrency buys NOTHING inside one analyzer pod — measured 1 job 37.6 s,
#     2 jobs 83.0 s each (2.2x, no throughput gain: ffmpeg already saturates the
#     2-cpu limit), and 4 concurrent 4K jobs OOMKilled the pod. Treat one pod as
#     SERIAL and count only replicas.
#     ⚠️ Do NOT reason from a replica count. There is no HPA in the w2v manifest for
#     this Deployment and the count was observed drifting 2 -> 3 mid-session, so any
#     safety argument resting on "there are N replicas" is not durable. The durable
#     bound is `_CONCURRENCY` itself.
#   * `_TIMEOUT_SOURCE_CEILING` = 12 sources x 52 s = 624 s of pod-work, i.e.
#     ~312 s wall at two replicas and 624 s if one replica is down — against the
#     3700 s budget that is ~5.9x headroom in the SINGLE-replica worst case.
#     The 10-clip corpus is 517 s of pod-work: 259 s / 517 s, i.e. 7.2x-14.3x.
#   * Per source: 52 s solo, ~114 s against one co-resident job, versus
#     `_POLL_TIMEOUT_SECONDS` = 540 s -> 4.7x headroom.
#
# NOT measured / known limits, deliberately left alone here:
#   * The MinIO fetch of the original (67-134 MB) and the audio extract are
#     interval-independent and were not timed; they are unchanged by this.
#   * `preview_key` is never written by anything in the fleet, so `_analyze_one`
#     always falls back to `original_url` and this cost is always the 4K cost.
#   * `scene_boundaries` do not improve with the finer lattice — the detector's
#     0.6 chi-square threshold flags 22 of 25 samples on this drone clip. The
#     motion curve is what gets better; the cut list stays near-degenerate.
_FRAME_INTERVAL_SECONDS = 0.25
_MAX_FRAMES = _MOTION_SAMPLES_PER_SOURCE + 1

_SAMPLING_OVERRIDES = {
    "frame_interval_seconds": _FRAME_INTERVAL_SECONDS,
    "max_frames": _MAX_FRAMES,
}


class ShotAnalysisInput(StageInput):
    """No typed inputs — sources are read from ``footage_probe``."""


class MotionPoint(StageOutput):
    """One point on the analyzer's motion curve: ``motion`` (0..1) at ``t`` seconds."""

    t: float
    motion: float


class AnalyzedSource(StageOutput):
    """Measurements for one source. ``id`` is the ``video_sources`` row id — the
    same key ``footage_probe`` emitted and ``segment_select`` selects against."""

    id: str
    duration_s: float = 0.0
    #: Cut points detected IN the footage, absolute seconds.
    scene_boundaries: list[float] = []  # noqa: RUF012 -- pydantic per-instance default
    motion_timeline: list[MotionPoint] = []  # noqa: RUF012
    mean_sharpness: float = 0.0
    mean_brightness: float = 0.0
    motion_score: float = 0.0
    overall_score: float = 0.0
    #: "" when the analyzer MEASURED this source; the failure sentence otherwise.
    #: Kept per source (not only in ``notes``) so a consumer can tell "measured,
    #: and it is a dull clip" from "never measured" — those must not weigh the
    #: same. A job that COMPLETED without ``quality_metrics`` sets this too: an
    #: all-zero score profile with an empty ``error`` is the same lie as a
    #: swallowed exception, only harder to see.
    error: str = ""


class ShotAnalysisOutput(StageOutputWithNotes):
    # Declared output contract — enforced once, centrally, in
    # `runs/dag_executor.py::_enforce_produces`. Empty here means the stage
    # reported success having produced nothing, which degrades the run.
    # `segment_select` selects against these measurements (scene cuts + motion curve);
    # an empty list means it would select against nothing.
    produces: ClassVar[tuple[str, ...]] = ("sources",)

    sources: list[AnalyzedSource]


class ShotAnalysisStage(StageHandler[ShotAnalysisInput, ShotAnalysisOutput]):
    name = "shot_analysis"
    input_model = ShotAnalysisInput
    output_model = ShotAnalysisOutput
    # Measure-only analyzer submit (every LLM-backed analyzer feature off): no LLM.
    provider_slots: ClassVar[tuple[SlotUse, ...]] = ()

    def __init__(self, client: VideoAnalyzerClient | None = None) -> None:
        self._client = client or VideoAnalyzerClient()

    async def handle(self, payload: ShotAnalysisInput, ctx: RunContext) -> ShotAnalysisOutput:
        probe = ctx.prior_output("footage_probe") or {}
        sources = probe.get("sources") or []
        if not sources:
            raise StagePreconditionError(
                "shot_analysis: footage_probe emitted no sources",
                missing=["footage_probe.sources"],
            )

        analyzed = await stage_gather(
            sources,
            self._analyze_one,
            concurrency=_CONCURRENCY,
            per_item_timeout_seconds=_PER_ITEM_TIMEOUT_SECONDS,
            per_item_retries=_PER_ITEM_RETRIES,
            stage_name=self.name,
        )
        notes = [f"source {a.id}: analysis failed ({a.error})" for a in analyzed if a.error]
        if analyzed and len(notes) == len(analyzed):
            # Every single source failed. Degrading here would hand segment_select a
            # measurement-free corpus and let it emit a confident, meaningless
            # rationale — fail loud instead.
            raise StagePreconditionError(
                "shot_analysis: video_analyzer failed for every source; "
                f"no measured signal to select from ({notes[0]})",
                missing=["video_analyzer"],
            )
        return ShotAnalysisOutput(sources=analyzed, notes=notes)

    async def _analyze_one(self, source: dict) -> AnalyzedSource:
        """Analyze one source. Soft-fails into an ``error``-carrying descriptor.

        The URL preference is preview-then-original: the preview proxy is a small,
        seekable re-encode and analyzing it is a fraction of the work of pulling a
        134 MiB 4K original through the analyzer — the measurements this stage
        needs (scene cuts, motion curve, exposure) survive the downscale.
        """
        source_id = str(source.get("id") or "")
        url = str(source.get("preview_url") or "") or str(source.get("original_url") or "")
        duration = float(source.get("duration_s") or 0.0)
        if not url:
            return AnalyzedSource(
                id=source_id,
                duration_s=duration,
                error="no playable url (original could not be presigned)",
            )
        try:
            job_id = await self._client.start_analysis(
                video_url=url,
                # Measure-only submit at action_reel's own sampling density —
                # see _ANALYZER_OVERRIDES and _SAMPLING_OVERRIDES.
                config_overrides={**_ANALYZER_OVERRIDES, **_SAMPLING_OVERRIDES},
            )
            await self._client.poll_analysis(job_id, timeout=_POLL_TIMEOUT_SECONDS)
            result = await self._client.get_result(job_id)
        except Exception as exc:  # noqa: BLE001 -- per-source soft-fail; see docstring
            log.warning("shot_analysis: source %s failed: %s", source_id, exc)
            return AnalyzedSource(id=source_id, duration_s=duration, error=str(exc))

        quality = result.quality_metrics
        if quality is None:
            # The job COMPLETED but carries no measurements. Filling the score
            # fields with 0.0 and leaving ``error`` empty reported "measured, and
            # it scored zero" for something that was never measured — exactly the
            # conflation the ``error`` field's docstring forbids, and the
            # selector would then read a genuinely-usable clip as the worst in
            # the corpus. ``duration_s`` / ``scene_boundaries`` are kept because
            # they are real: they come off the result, not off the quality stage,
            # and the selector can still cut on duration alone.
            return AnalyzedSource(
                id=source_id,
                duration_s=float(result.duration_seconds or duration),
                scene_boundaries=list(result.scene_boundaries or []),
                error="analyzer returned no quality_metrics (source is unmeasured)",
            )
        return AnalyzedSource(
            id=source_id,
            # The analyzer probed the real file, so prefer its duration over the
            # upload-time ffprobe blob (which is empty on a still-pending row).
            duration_s=float(result.duration_seconds or duration),
            scene_boundaries=list(result.scene_boundaries or []),
            motion_timeline=[MotionPoint(t=s.t, motion=s.motion) for s in quality.motion_timeline],
            mean_sharpness=quality.mean_sharpness,
            mean_brightness=quality.mean_brightness,
            motion_score=quality.motion_score,
            overall_score=quality.overall_score,
        )
