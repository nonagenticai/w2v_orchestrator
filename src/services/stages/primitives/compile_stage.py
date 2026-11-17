"""CompileStageBase — the invariant video-compile tail shared by render stages.

Every compile/render stage does the same thing once its shots are built: bail
loud if there are no shots, POST them to the video-compile service (a
non-idempotent render, so ``retries=0``), validate the response actually
produced a usable artifact, then unpack ``{artifact_url, duration_seconds}``.

Only the shot-building and (for promo) an extra output flag differ, and the
``video_render`` stage carries a smaller tag set — those are the hooks.
Shot-building itself stays per-stage (the join keys genuinely differ: lesson's
paragraph→slide→image two-hop, app_review's per-line screenshot, the other
three's ``assemble_shots`` round-robin), so this base deliberately does not try
to unify it.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
from abc import abstractmethod
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, TypedDict, cast

from src.clients.transcription_worker_client import TranscriptionWorkerClient
from src.clients.video_compile import VideoCompileClient
from src.config import get_settings
from src.services.compile_budget import COMPILE_WAIT_CEILING_SECONDS
from src.services.stages.base import (
    SLOT_USE_OPTIONAL,
    SlotUse,
    StageHandler,
    StageInput,
    StageOutput,
    StagePreconditionError,
    validate_compile_result,
)
from src.services.stages.phase_tracker import (
    PhaseDeadlineError,
    phase,
    phase_totals,
    reset_phase_totals,
)
from src.services.stages.shared.cadence import build_cadence_spec
from src.services.stages.shared.captions_only import captions_only
from src.services.stages.shared.colorist import resolve_mood
from src.services.stages.shared.degradation import (
    CAPTIONS_MISSING,
    COMPILE_DURATION_MISMATCH,
    COMPILE_DURATION_UNVERIFIED,
    MUSIC_BED_NOT_APPLIED,
    Degradation,
    degradation,
)
from src.services.stages.shared.disclaimer_card import (
    build_disclaimer_shot,
    splice_disclaimer_card,
)
from src.services.stages.shared.duration_conformance import (
    playable_seconds_from_mastering,
)
from src.services.stages.shared.finish import build_finish_spec
from src.services.stages.shared.ken_burns import (
    _PER_ITEM_OVERHEAD_S as _KB_PER_ITEM_OVERHEAD_S,
)
from src.services.stages.shared.ken_burns import (
    _PHASE_BUDGET_S as _KB_PHASE_BUDGET_S,
)
from src.services.stages.shared.ken_burns import (
    _resolve_cfg as _ken_burns_cfg,
)
from src.services.stages.shared.ken_burns import apply_ken_burns
from src.services.stages.shared.ken_burns_local import (
    crop_cover_stills_to_frame,
    flag_near_static_clips,
    rescue_frozen_stills,
)
from src.services.stages.shared.mastering_gate import (
    mastering_gate_advisories,
    mastering_gate_degradations,
    mastering_gate_notes,
)
from src.services.stages.shared.mastering_plan import build_mastering_profile
from src.services.stages.shared.motion_slides import format_dims, quality_tier
from src.services.stages.shared.title_cards import (
    TitleCardPlan,
    plan_title_cards,
    render_title_card_plan,
    splice_title_cards,
)
from src.services.stages.shot_assembly import drain_fabricated_hold_notes
from src.services.stages.shot_presign import refresh_shot_asset_urls
from src.services.stages.shot_sound import resolve_shot_sound_urls
from src.services.storage_service import StorageService
from src.services.tenant_defaults import captions_karaoke as resolve_captions_karaoke
from src.services.tts_storage import _get_storage
from src.services.word_timed_captions import ClipRef, ShotCaptionsPayload, build_shot_captions

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

logger = logging.getLogger(__name__)


def _compile_idempotency_key(ctx: RunContext, stage_name: str) -> str:
    """Resume-stable dedup key for a compile submit.

    ``run_id`` is fresh per regenerate and there is one compile per run per stage,
    so ``{run_id}:{stage}`` survives a resumable-timeout re-dispatch unchanged (the
    compiler returns the existing job for a repeat key instead of running a second
    paid render). The optional ``render_idempotency_discriminator`` distinguishes a
    footage-critique RE-render (same run_id + name, DIFFERENT shots) so it is NOT
    deduped to the original render.
    """
    base = f"{ctx.run_id}:{stage_name}"
    disc = ctx.render_idempotency_discriminator
    return f"{base}:{disc}" if disc else base


def render_payload_discriminator(prefix: str, payload: object) -> str:
    """A PAYLOAD-derived ``render_idempotency_discriminator`` for a re-compile.

    A re-compile driven off ``dataclasses.replace(ctx, ...)`` keeps ``run_id`` AND the
    stage name, so with the default empty discriminator its idempotency key collides
    with the compile already performed in this run: ``video_compile`` returns the
    EXISTING job and the caller ships the ORIGINAL artifact while reporting the new
    settings as applied.

    The discriminator is derived from the CHANGED PAYLOAD rather than from a loop
    ordinal, so both halves of idempotency stay honest:

    * two re-compiles whose payload differs never collide (the defect this fixes);
    * two re-compiles with an IDENTICAL payload still dedupe onto the render already
      paid for — a per-call ordinal / timestamp / uuid would instead multiply the
      render bill for byte-identical work.

    An empty/falsy payload means "nothing changed versus the original render", so it
    returns ``""`` and the re-compile dedupes onto that original render.

    (``footage_critique`` keeps a pass-ordinal discriminator on purpose: there the
    documented contract is first-nudge-wins across a resume, not payload equality.)
    """
    if not payload:
        return ""
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return f"{prefix}:{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


# Two payload shapes this module traffics in, named ONCE rather than respelled at
# every signature. `build_shots`, the splice/re-sign helpers and the compile client
# have always exchanged bare payload dicts here; extracting `_run_presubmit` would
# otherwise have multiplied that untyped spelling across a carrier plus new
# signatures, which the anti-Any ratchet (tests/test_any_ratchet.py) refuses --
# correctly. Naming them is the fix that gate is asking for at this layer; a real
# pydantic model for a Shot is a far larger change touching every stage.
# Naming them is the fix the ratchet is asking for at this layer -- a real pydantic
# model for a Shot is a much larger change that would touch every stage.
type ShotDict = dict[str, Any]
type PayloadDict = dict[str, Any]

# The orchestrator's own give-up deadline for one compile()/`_await_job` wait,
# bound to the compiler's OWN worst case (video_compiler_service
# Settings.reclaim_max_lifetime_seconds, 7200s default) — see
# `src.services.compile_budget` for the full derivation and the presign-TTL
# consumer that must stay in sync with it.
#
# ⛔ THIS USED TO BE 1200.0 (20 min), and that was BELOW the compiler's own
# realistic worst case (a single ffmpeg subprocess is allowed 1800s, plus
# reaper redispatch on top). Production incident 2026-09-17 (compile job 1602):
# the orchestrator abandoned a compile job the compiler still considered
# healthy (lease heartbeating every ~2min) not once but TWICE, the second
# abandonment triggering a full stage resume — re-rendering title cards —
# WHILE the original compile was still running. See compile_budget.py's
# module docstring for the full timeline.
_COMPILE_TIMEOUT_SECONDS = COMPILE_WAIT_CEILING_SECONDS  # 7500.0

# Aggregate deadline over the pre-submit asset block (title cards → captions,
# everything BEFORE the compile-wait backstop below). Before this, that ~120-line
# block had no deadline at all: on 2026-09-03/04 it burned the blueprint's outer
# ceiling three times over (126 minutes total) with one `logger.warning` and zero
# DB writes, and the failure that eventually surfaced was an anonymous,
# zero-argument TimeoutError from the OUTER budget — nothing said where it died.
# See `src/services/stages/phase_tracker.py` for the full incident writeup.
#
# 1800s (30 min) is comfortably below COMPILE_STAGE_OUTER_TIMEOUT_SECONDS below,
# so THIS deadline fires first and produces an attributable error naming the
# phase, rather than the anonymous outer one.
#
# The phases that can WAIT inside it, and how the budget is shared (re-derived
# 2026-10-05, when promo gained a default end card and Ken Burns stopped
# abandoning queued jobs at 600s):
#
# * title cards — one ``wait_for_content_recording`` of up to 600s per card,
#   intro then outro: <= 600s for promo's end card, <= 1200s with both;
# * ken_burns — polls each job to a pass deadline of its OWN budget,
#   ``_KEN_BURNS_PASS_BUDGET_SECONDS`` (= ``ken_burns._PHASE_BUDGET_S``, 1000s).
#   It used to be GIVEN only what was left here minus the tail reserve, so a
#   slow title card silently shortened it: on an UNCONTENDED run 3 of 9 shots
#   (the third recorder wave — 3 slots/pod) were dropped to stills after only
#   280-395s of waiting. Now, entering the pass, the aggregate deadline is
#   RESCHEDULED (never shortened) so it can hold the pass's own budget plus its
#   per-item overhead, the local-rescue reserve and the tail reserve — see
#   ``_ensure_ken_burns_room``. The deadline still bounds the block: it only
#   moves by an amount fixed in advance, and ``_PRESUBMIT_WORST_CASE_SECONDS``
#   is what the outer per-stage ceiling is sized from;
# * captions — the ASR fallback is 300s per clip; the caption build gets
#   whatever is left (minus ``_PRESUBMIT_FINAL_MARGIN_SECONDS``) under its own
#   timeout, and a timeout there soft-fails to CAPTIONS_MISSING like any other
#   caption failure;
# * presign / sound / audio bed — local SigV4 signing and a library read,
#   sub-second.
#
# Worst case, promo under congestion: 600 (card) + 1000 + 120 (Ken Burns: the
# recorder pass and its per-item overhead) + <= 360 (the local still rescue —
# ``_local_rescue_reserve``) + 420 for re-sign, sound and captions = 2500, i.e.
# the deadline is moved out by 700s at the Ken Burns entry. A deadline hit now
# means a phase stalled past its own bound (a leaked resource), not a busy fleet.
_PRESUBMIT_DEADLINE_SECONDS = 1800.0
# The Ken Burns recorder pass's OWN wall budget — never the remainder of the
# aggregate deadline (see the ken_burns bullet above).
_KEN_BURNS_PASS_BUDGET_SECONDS = _KB_PHASE_BUDGET_S
# ``apply_ken_burns`` bounds each item at its pass budget plus this (fetch,
# prescale, upload), so the room reserved for the pass includes it.
_KEN_BURNS_PASS_OVERHEAD_SECONDS = _KB_PER_ITEM_OVERHEAD_S
# Held back from the Ken Burns pass for everything after it: the caption build's
# 300s ASR timeout plus re-sign / sound / audio bed.
_PRESUBMIT_TAIL_RESERVE_SECONDS = 420.0
# A product that opted in to the local Ken Burns fallback (promo) also holds this
# much per still back from the RECORDER pass, capped, so the local rescue of a
# still the recorder failed always has time to run. Run 01a11082 (2026-10-06): a
# recorder job ran the pass to its deadline and the fallback then refused with
# "no pre-submit budget left" — the film shipped a frozen still. A product that
# did not opt in holds nothing back: its recorder pass keeps the full budget.
_LOCAL_RESCUE_SECONDS_PER_STILL = 90.0
_LOCAL_RESCUE_RESERVE_CAP_SECONDS = 360.0
# Held back from the caption build so its own timeout fires (soft) before the
# aggregate deadline does (hard).
_PRESUBMIT_FINAL_MARGIN_SECONDS = 30.0

# Backstop margin around the compile wait itself (see the "belt-and-braces"
# comment on `_compile_with_backstop` below).
_COMPILE_BACKSTOP_MARGIN_SECONDS = 120.0

# Outer per-stage ceiling (StageSpec.default_timeout_seconds → TimeoutMiddleware)
# for every blueprint stage built on this base (``*_compile`` / ``*_render``).
# Blueprints import this rather than hardcoding a number, so the outer wrapper
# can never again silently fall below the wait it wraps — exactly how the old
# 1200s inner wait and the old 2500s outer ceiling both undershot the compiler's
# real worst case at once. `int(...)` because StageSpec.default_timeout_seconds
# is typed `int | None`.
#
# The pre-submit term is its WORST case, not the 1800s base: the Ken Burns entry
# may reschedule the aggregate deadline to ``now + _ken_burns_room(...)``, and
# ``now`` can be as late as the base deadline itself.
_PRESUBMIT_WORST_CASE_SECONDS = (
    _PRESUBMIT_DEADLINE_SECONDS
    + _KEN_BURNS_PASS_BUDGET_SECONDS
    + _KEN_BURNS_PASS_OVERHEAD_SECONDS
    + _LOCAL_RESCUE_RESERVE_CAP_SECONDS
    + _PRESUBMIT_TAIL_RESERVE_SECONDS
)  # 3700
COMPILE_STAGE_OUTER_TIMEOUT_SECONDS = int(
    _PRESUBMIT_WORST_CASE_SECONDS
    + _COMPILE_TIMEOUT_SECONDS
    + _COMPILE_BACKSTOP_MARGIN_SECONDS
    + 180.0
)  # 11500


def _local_rescue_reserve(was_still: set[int] | None) -> float:
    """Seconds the recorder pass leaves for the local still rescue after it.

    ``None`` (the product did not opt in) → ``0.0``: the recorder keeps the whole
    budget, exactly as before. Otherwise ``min(stills x 90s, 360s)``.
    """
    if not was_still:
        return 0.0
    return min(len(was_still) * _LOCAL_RESCUE_SECONDS_PER_STILL, _LOCAL_RESCUE_RESERVE_CAP_SECONDS)


def _ken_burns_room(was_still: set[int] | None) -> float:
    """Seconds the aggregate deadline must still hold when the Ken Burns pass starts.

    The pass's own budget and per-item overhead, the local-rescue reserve, and the
    tail reserve (re-sign / sound / captions) after it.
    """
    return (
        _KEN_BURNS_PASS_BUDGET_SECONDS
        + _KEN_BURNS_PASS_OVERHEAD_SECONDS
        + _local_rescue_reserve(was_still)
        + _PRESUBMIT_TAIL_RESERVE_SECONDS
    )


def _ensure_ken_burns_room(cm: asyncio.Timeout, was_still: set[int] | None) -> float:
    """Move the aggregate deadline out (never in) so the Ken Burns pass gets its OWN budget.

    Returns the seconds the deadline was moved by (``0.0`` when it already held
    the room). Bounded: the room is a constant plus a capped reserve, so the
    block's worst case is ``_PRESUBMIT_WORST_CASE_SECONDS``.
    """
    room = _ken_burns_room(was_still)
    remaining = _presubmit_remaining(cm)
    if remaining >= room:
        return 0.0
    cm.reschedule(asyncio.get_running_loop().time() + room)
    return room - remaining


def _presubmit_remaining(cm: asyncio.Timeout) -> float:
    """Seconds left before the aggregate pre-submit deadline ``cm`` fires."""
    when = cm.when()
    if when is None:  # pragma: no cover - the block always sets a deadline
        return _PRESUBMIT_DEADLINE_SECONDS
    return when - asyncio.get_running_loop().time()


async def _compile_with_backstop(coro: Awaitable[PayloadDict], *, stage_name: str) -> PayloadDict:
    """Await one ``VideoCompileClient.compile()`` call with a thin backstop.

    Deliberately NOT routed through ``w2v_common.stages.stage_call``:
    ``stage_call`` hard-validates ``timeout_seconds <= 3600s``
    (``w2v_common.stages.execution._MAX_TIMEOUT_SECONDS``) and raises
    ``ValueError`` above it. ``_COMPILE_TIMEOUT_SECONDS`` (7500s) plus this
    backstop's own margin is well past that cap — so wrapping this call in
    ``stage_call`` would either crash on the ``ValueError`` or (if clamped to
    3600s) silently truncate the wait back down, reintroducing the exact defect
    this module exists to fix, just one layer lower than the 1200s constant
    that shipped it the first time.

    ``compile()``'s own internal deadline (``_await_job``, an unbounded plain
    float — see ``clients/video_compile.py``) is the REAL governor here; this
    ``asyncio.wait_for`` is only a belt-and-braces backstop against a bug in
    that loop failing to honour its own timeout, sized a little higher so the
    client's own, better-attributed ``TimeoutError`` fires first.

    ``retries=0`` was already the only legal value at the old ``stage_call``
    call site (a compile is non-idempotent — see ``_compile_idempotency_key``),
    so its retry machinery bought nothing here and dropping it costs nothing.
    A bare ``TimeoutError`` on expiry (from either layer) is already resumable
    — ``dag_executor._RESUMABLE_TIMEOUT_EXC`` matches the builtin type directly,
    the same contract ``stage_call``'s own ``StageTimeoutError`` gave it.
    """
    backstop_seconds = _COMPILE_TIMEOUT_SECONDS + _COMPILE_BACKSTOP_MARGIN_SECONDS
    try:
        return await asyncio.wait_for(coro, timeout=backstop_seconds)
    except TimeoutError:
        logger.warning(
            "%s: compile backstop timed out after %.1fs — compile()'s own %.1fs "
            "liveness-based deadline should have fired first; this is the "
            "belt-and-braces path, not the expected one",
            stage_name,
            backstop_seconds,
            _COMPILE_TIMEOUT_SECONDS,
        )
        raise


# Per-tier CRF for the terminal compile (LOWER = sharper/bigger).
#
# Pinned to the compiler's own delivery CRF, which is **18**, not 20 — see
# `video_compiler_service/src/config.py::render_crf` (default 18, "the
# near-transparent cell for the 2K/1440p canvas this fleet actually renders
# at"). The comment here previously said "the video_compiler env default is 20"
# and the table was `{1080p: 20, 2k: 19, 4k: 18}`. Both were stale, and the
# combination was actively harmful: `quality_tier` normalises an unset tier to
# `2k` (`shared/motion_slides.py::_DEFAULT_TIER`), so EVERY production master
# that did not pin `mastering.crf` was tagged `crf=19` and the compiler's own 18
# was tag-shadowed away — i.e. the previous wave's CRF fix never reached a single
# delivered file. It also dragged the intermediate down with it, since
# `encode_ladder` derives the polish pass as `crf - render_intermediate_crf_delta`
# (19-6=13 instead of 18-6=12).
#
# Why UNIFORM rather than a per-tier ramp: x264's CRF is a constant-QUALITY
# target, deliberately resolution-independent, so a ramp encodes an assumption
# the encoder does not share. The old ramp got its own stated intent backwards —
# it put 1080p at 20, the SOFTEST cell in the table, on the theory that small
# canvases need less bitrate. The opposite holds perceptually: with fewer pixels
# each quantisation artifact occupies a LARGER fraction of the frame, so 1080p
# needs at least the same CRF as 2K, never a looser one. 1080p is also the
# cheapest tier in absolute bitrate, so nothing was being bought by softening it.
#
# 4K stays at 18 (unchanged). It is already at the delivery CRF; going lower
# would multiply the most expensive tier's bitrate for no measured gain, and
# going higher would soften the one tier viewers actually zoom into.
#
# Cost of the change is bitrate only (~12%), no extra encode time.
_CRF_BY_TIER: dict[str, int] = {"1080p": 18, "2k": 18, "4k": 18}

# Fallback for a tier absent from the table. `quality_tier` normalises unknown
# values to `2k`, so this is unreachable today — it is pinned to the delivery CRF
# anyway so that a future tier added to `_TIER_SCALE` alone cannot silently
# reintroduce the soft cell this change removed.
_CRF_FALLBACK = 18

# Auto-QA tolerance for declared-vs-actual compile duration: a gap larger than
# max(_DURATION_QA_ABS_TOLERANCE_S, declared * _DURATION_QA_REL_TOLERANCE) flags
# a possible truncation / short-render. Kept loose so normal encode rounding and
# tpad freeze-padding never trip it.
_DURATION_QA_ABS_TOLERANCE_S = 2.0
_DURATION_QA_REL_TOLERANCE = 0.05


# An UNPROBED length is not a verified one. ``actual_duration_seconds`` is the
# compiler's ffprobe of the delivered mux, and `probe_output_duration` returns the
# **0.0 sentinel meaning "unknown"** on ANY probe failure/timeout
# (`video_compiler_service/src/services/renderer.py:369-381`) — so a failed probe and
# a healthy 1:1 render used to produce the SAME empty note list. That made "nobody
# measured this master's length" byte-identical to "its length was measured and is
# correct", which is this codebase's #1 defect class: a truncated master (234s of
# frames under a 263.5s declaration) ships CLEAN whenever the probe is the thing that
# broke. The gap therefore gets its OWN note, distinct in wording from both "checked
# and fine" (no note) and "checked and short" (the mismatch note below), exactly as
# `mastering_gate_notes` does for an absent mastering report.
#
# Blast radius, checked rather than assumed. The mux path emits
# ``actual_duration_seconds`` on every completed render, so the 0.0 sentinel — a
# FAILED probe — is the case this note actually fires on, and it fires on ~0
# healthy runs. The absent-key case is effectively dead on the compile path (the
# probe is unconditional) and the audio-concat result cannot reach this function
# at all, so absent is treated as unmeasured without a carve-out -- see the
# branch below for why no "was it probed" flag is available to consult.
_DURATION_UNMEASURED_NOTE = (
    "compile duration: the compiler reported no usable actual_duration_seconds "
    "(ffprobe of the delivered mux returned unknown), so the declared-vs-encoded "
    "duration check never ran — this master's length is UNMEASURED, not verified; "
    "degrading"
)


def audio_bed_record(bed: PayloadDict, result: PayloadDict) -> PayloadDict:
    """What bed this compile SENT and what the compiler said it did with it.

    The positive half of the bed record: ``music_bed_degradations`` and the
    ``SOUND_BED_MISSING`` sink only ever speak when a bed was lost, so a promo
    with a music bed and one without one persisted identical outputs and a frame
    QA could not tell them apart. Recorded on outputs that declare an
    ``audio_bed`` field (promo_render); ``compiler`` is the compiler's own
    ``music_bed`` report, ``None`` from a compiler that does not send one.
    """
    music = bed.get("music")
    report = result.get("music_bed")
    return {
        "music_sent": isinstance(music, dict) and bool(music.get("url")),
        "ambient_cues_sent": len(bed.get("ambient") or []),
        "compiler": dict(report) if isinstance(report, dict) else None,
    }


def music_bed_degradations(result: PayloadDict, bed: PayloadDict) -> list[Degradation]:
    """Record a bed the orchestrator SENT that the compiler says did not land.

    The compiler answers every compile that asked for a bed with a ``music_bed``
    report, and flags a lost one explicitly (``video_compiler_service``
    ``renderer.py::_flag_unapplied_music_bed``): ``applied: False`` (+
    ``fetch_failed``) for music, ``ambient_fetch_failed`` / ``ambient_lost`` for
    ambient. Nothing here read it, so a master with no bed shipped ``completed``
    with no trace. Gated on what THIS request carried (``bed``), so a stale or
    unexpected report on a bed-less compile never degrades a run. An absent or
    non-dict report is not evidence of loss (older compiler) and records nothing.
    A degradation, never a failure: the master exists and is otherwise usable.
    """
    report = result.get("music_bed")
    if not isinstance(report, dict):
        return []
    out: list[Degradation] = []
    if bed.get("music") and report.get("applied") is False:
        reason = (
            "the compiler could not fetch the bed asset"
            if report.get("fetch_failed") is True
            else "the compiler did not apply it (cause not reported)"
        )
        out.append(
            degradation(
                MUSIC_BED_NOT_APPLIED,
                f"music bed was sent to the compiler but not applied: {reason} "
                f"(fetch_failed={report.get('fetch_failed')!r}); the delivered video "
                "has no music bed",
            )
        )
    if bed.get("ambient") and report.get("ambient_fetch_failed") is True:
        out.append(
            degradation(
                MUSIC_BED_NOT_APPLIED,
                f"ambient bed: {report.get('ambient_lost')!r} of "
                f"{report.get('ambient_requested')!r} cue(s) could not be fetched by the "
                "compiler; the delivered video is missing that ambience",
            )
        )
    return out


def _compile_duration_qa_notes(result: PayloadDict) -> list[str]:
    """Flag a material gap between the DECLARED and ACTUAL compile duration.

    ``duration_seconds`` is the nominal sum of shot durations; the compiler also
    reports ``actual_duration_seconds`` (ffprobe of the encoded output). A gap
    beyond tolerance means the file is shorter than intended — a truncated or
    short-rendered output.

    Returns note strings. Empty ONLY when the check actually ran and passed — an
    absent / non-numeric / non-finite / non-positive ``actual`` yields
    :data:`_DURATION_UNMEASURED_NOTE` instead of silence (see the comment above it):
    the guard could not run, which is neither a pass nor a fail, and it must not be
    reported as a pass. Still only a note — never raises, never fails the run.

    The DECLARED side is deliberately NOT part of that trigger. ``handle`` calls
    ``validate_compile_result`` (``stages/artifacts.py:40-52``) BEFORE this, and it
    already HARD-FAILS the stage on a missing / non-numeric / non-finite /
    non-positive ``duration_seconds``. So on the live path ``declared`` is
    guaranteed usable here, and the declared-side early returns below are reachable
    only from a direct call: degrading on them would be dead code in the fleet while
    widening the trigger across all nine ``CompileStageBase`` subclasses for nothing.
    """
    declared_raw = result.get("duration_seconds")
    actual_raw = result.get("actual_duration_seconds")
    if actual_raw is None:
        # Absent means nothing measured it. Checked rather than assumed: the
        # compiler calls ``probe_output_duration`` unconditionally on every video
        # compile, so the field is always present on a real compile result, and
        # ``JobStatus`` carries no companion "was it probed" flag to consult --
        # the compiler's internal ``duration_probed`` never reaches the wire.
        # The audio-concat result, which legitimately has no encoded-video length,
        # cannot arrive here at all: ``AudiobookStitchStage`` is a plain
        # ``StageHandler`` calling ``/api/concat-audio``, not a ``CompileStageBase``.
        return [_DURATION_UNMEASURED_NOTE]
    try:
        actual = float(actual_raw)
    except TypeError, ValueError:
        return [_DURATION_UNMEASURED_NOTE]
    if not math.isfinite(actual) or actual <= 0:
        return [_DURATION_UNMEASURED_NOTE]
    # Declared side: an unusable declaration is already a hard failure upstream
    # (see the docstring), so it stays a silent skip rather than a second note.
    if declared_raw is None:
        return []
    try:
        declared = float(declared_raw)
    except TypeError, ValueError:
        return []
    if not math.isfinite(declared) or declared <= 0:
        return []
    gap = abs(declared - actual)
    tolerance = max(_DURATION_QA_ABS_TOLERANCE_S, declared * _DURATION_QA_REL_TOLERANCE)
    if gap <= tolerance:
        return []
    return [
        f"compile duration mismatch: declared {declared:.1f}s vs encoded "
        f"{actual:.1f}s (gap {gap:.1f}s > {tolerance:.1f}s tolerance) — possible "
        "truncation or short-render"
    ]


def _compile_duration_qa_degradations(qa_notes: list[str]) -> list[Degradation]:
    """The STRUCTURED twin of :func:`_compile_duration_qa_notes`' output.

    Takes the notes that function returned rather than re-deriving the verdict,
    so the two channels cannot disagree: the unmeasured note becomes
    :data:`COMPILE_DURATION_UNVERIFIED`, the only other note it can return (the
    declared-vs-encoded gap) becomes :data:`COMPILE_DURATION_MISMATCH`, and
    ``detail`` is the note itself. No new degraded run: each note already
    degrades the stage.
    """
    return [
        degradation(
            COMPILE_DURATION_UNVERIFIED
            if note == _DURATION_UNMEASURED_NOTE
            else COMPILE_DURATION_MISMATCH,
            note,
        )
        for note in qa_notes
    ]


def playable_duration_seconds(result: PayloadDict) -> float | None:
    """The PLAYABLE length of the delivered mux, or None when unmeasured.

    ⛔ THE OBVIOUS FIELD IS THE WRONG ONE, and a sibling ``probed_duration_seconds``
    reading it stood here until it was deleted. ``actual_duration_seconds`` is
    ``min(video_stream, container)``
    (``video_compiler_service/src/services/renderer.py::_resolve_probed_duration``),
    i.e. the **PICTURE** length. On a master whose picture stopped early it is the
    TRUNCATED value, so grading a delivery with it reports a broken, over-long
    film as an on-target one. It is not a "delivered" measurement at all and no
    conformance verdict may be taken against it; ``_compile_duration_qa_notes``
    above reads it directly, which is legitimate because that note grades declared
    against ENCODED PICTURE and says so.

    This reads the compiler's ``mastering`` report instead, which probes the
    video and audio streams SEPARATELY for exactly this reason
    (``video_compiler_service/src/api/schemas.py``: "the defect these exist for
    shipped for months because the report carried ONE reduced duration scalar"),
    and returns ``max`` of the two — the length a player reports.

    ``None`` — never a fallback, and never the picture-only number — whenever the
    mastering block is absent, degraded (``mastering.degraded_report()`` nulls
    BOTH stream fields when ``post_mux_verify`` crashes) or either stream is
    unmeasured. The caller then falls to
    :func:`~src.services.stages.shared.duration_conformance.duration_conformance`'s
    only other rung, the DECLARED length, which is stamped ``declared`` on the
    record and raises ``DURATION_UNMEASURED``.
    """
    return playable_seconds_from_mastering(result.get("mastering"))


def _as_dict_or_none(value: object) -> dict[str, object] | None:
    """A compiler-response field narrowed to a dict, or None.

    The compiler's response is untyped JSON, so a field can legitimately be
    absent (older build) or arrive as a non-dict. Anything that is not a
    populated dict becomes None rather than being carried onto the output — the
    same narrowing demo_video's ``build_output`` already applied inline. An
    EMPTY dict is also None: "the compiler returned no report" and "the compiler
    returned an empty report" are the same fact downstream, and keeping {} would
    make the non-clobbering guard above treat a blank as a real value.
    """
    return dict(value) if isinstance(value, dict) and value else None


def _as_str_or_none(value: object) -> str | None:
    """A compiler-response field narrowed to a non-empty string, or None."""
    return value if isinstance(value, str) and value.strip() else None


def resolution_tags(ctx: RunContext) -> dict[str, str]:
    """Resolution + CRF compile tags for the run's quality tier.

    Forwarded to video_compiler_service, which reads them per-job to set the
    output canvas + encode quality — so a 2K/4K run raises the final MP4 without
    a service redeploy. Shared by every compile/render stage's ``compile_tags``
    so the terminal canvas always matches the upstream motion/slide resolution.

    The ``crf`` tag is OMITTED when the author pinned ``mastering.crf`` — the
    compiler prefers an explicit compile-tag over the mastering-block value
    (``if _key not in tags``), so an always-present tier CRF would tag-shadow the
    authored/advisor CRF and it could never take effect. Dropping the tag lets
    the compiler fall back to the mastering-block crf. ``render_width`` /
    ``render_height`` always ride (the canvas is never authored via mastering).
    """
    deliv = ctx.config_snapshot.get("deliverable") or {}
    tier = quality_tier(ctx.config_snapshot)
    width, height = format_dims(deliv.get("format"), tier)
    tags = {
        "render_width": str(width),
        "render_height": str(height),
    }
    # Presence-is-opt-in: only an authored `mastering.crf` (profile.crf not None)
    # suppresses the tier tag; an absent mastering block or a mastering block that
    # doesn't pin crf keeps the tag byte-identical to before.
    profile = build_mastering_profile(ctx.config_snapshot)
    if profile is None or profile.crf is None:
        tags["crf"] = str(_CRF_BY_TIER.get(tier, _CRF_FALLBACK))
    return tags


class _DimensionFields(TypedDict, total=False):
    """The ``{width, height}`` subset a compile result contributes to an output.

    A precise (keys-known) return type so ``**output_dimension_fields(result)``
    into a StageOutput constructor only ever populates width/height — never the
    unrelated ``notes: list[str]`` field on the base.
    """

    width: int
    height: int


def output_dimension_fields(result: PayloadDict) -> _DimensionFields:
    """Extract the real encoded ``{width, height}`` from a compile result.

    video_compiler ffprobes the produced MP4 and returns its actual dimensions,
    letting a compile/render stage expose the true output resolution (so the e2e
    post-promote check can assert the 2K/4K tier landed). Absent/zero values
    (older compiler builds, or a probe miss) are dropped → the stage output
    fields stay ``None`` rather than reporting a bogus 0.
    """
    out: _DimensionFields = {}
    for key in ("width", "height"):
        try:
            val = int(result.get(key) or 0)
        except TypeError, ValueError:
            val = 0
        if val > 0:
            out[key] = val
    return out


def captions_mode(ctx: RunContext) -> str:
    """The deliverable's ``captions_mode`` (``none`` | ``burned_in`` | ``overlay``).

    Single source of truth for the burned-in gate, shared by every compile/render
    stage (and the ``compile_tags`` forwarded to the video-compile service) so the
    setting is read identically everywhere.
    """
    mode = str((ctx.config_snapshot.get("deliverable") or {}).get("captions_mode", "none"))
    # A captions-only (silent) render IS its captions: with no voice there is nothing
    # else to read, so `none` still ships captions -- but as the SOFT overlay track
    # (mov_text + SRT sidecar), never burned into the picture. Owner rule: subtitles
    # are never burned in by default; only an explicit `burned_in` burns.
    if mode == "none" and captions_only(ctx):
        return "overlay"
    return mode


@dataclass(slots=True)
class _PresubmitAssets:
    """Everything the pre-submit asset block resolves, handed to the compile call.

    A plain carrier so `handle`'s own complexity stays low — the branching lives
    in `CompileStageBase._run_presubmit`, not here.
    """

    shots: list[ShotDict]
    bed: PayloadDict
    mastering: PayloadDict | None
    finish: PayloadDict | None
    cadence: int | None
    captions: list[ShotCaptionsPayload] | None
    # Degrade reasons raised DURING pre-submit (currently: a dropped title card).
    # Carried out rather than logged-and-forgotten, because a soft-failed card
    # otherwise leaves no trace once the pod rolls -- which is exactly why the
    # 2026-09-03 -60.000s loss took three sessions to attribute.
    notes: list[str]
    # The same carry-out for the STRUCTURED channel (`shared/degradation.py`):
    # dropped captions and a dropped music/ambient bed are raised here, deep in
    # the asset block, and must reach `handle` to be recorded on the output.
    # Each was previously a `log.warning` (captions) or nothing at all (bed).
    degradations: list[Degradation]
    # Count of shots eligible for a Ken Burns still→clip job (image_url, no
    # video_url), measured immediately before `apply_ken_burns` runs. Carried
    # out so `handle` can persist it alongside `phase_seconds["ken_burns"]` as
    # the denominator that turns a raw total into an interpretable rate — see
    # `base.py::PhaseTimingFields`. Defaulted (not required) so a future direct
    # constructor of this carrier is not forced to supply an enrichment value
    # it may have nothing meaningful to say.
    ken_burns_shot_count: int | None = None
    # Sub-stage checkpoint evidence: how many Ken Burns clips / title cards were
    # served from the content-addressed store instead of being re-recorded (and
    # re-paid) — non-zero on a compile resumed after a pod roll. ``None`` when the
    # phase never ran (no candidates / cards disabled), ``0`` when it ran cold.
    ken_burns_cache_hits: int | None = None
    title_card_cache_hits: int | None = None
    # The ADVISORY twin of `degradations`: facts raised during pre-submit that do
    # not make the delivery bad (a still animated by the local Ken Burns fallback,
    # a near-static recorded clip). Recorded on the output's `advisories` field.
    advisories: list[Degradation] = field(default_factory=list)


def _title_card_cache_hits(plan: TitleCardPlan | None, hits: list[str]) -> int | None:
    """Reused-card count, or ``None`` when the run plans no title cards at all
    (so "no cards planned" never reads as "cards rendered cold").

    Takes the SAME plan object the render was handed (``title_card_plan`` hook,
    resolved once per compile) — it does not plan for itself, so the count's
    ``None``-ness cannot disagree with what was put on screen.
    """
    if plan is None or not plan.card_count:
        return None
    return len(hits)


class CompileStageBase[I: StageInput, O: StageOutput](StageHandler[I, O]):
    """Shared compile tail. Subclasses implement ``build_shots`` + ``build_output``."""

    #: Suffix for the StagePreconditionError raised when build_shots is empty.
    #: Word-timed captions transcribe each clip (``TranscriptionWorkerClient``).
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (SlotUse("asr", when=SLOT_USE_OPTIONAL),)
    no_shots_message: ClassVar[str] = "no shots to compile"

    def __init__(self, client: VideoCompileClient | None = None) -> None:
        # compile() is async submit→poll, so the per-request read timeout
        # (BaseServiceClient default, capped at 600s) only has to cover a quick
        # submit/poll GET — never the whole render. The render budget is the
        # poll-loop wall-clock below (_COMPILE_TIMEOUT_SECONDS), which has no
        # 600s ceiling. This is what fixed the website_review compile ReadTimeout
        # (post-promote run 27535907827): long/queued renders no longer hold a
        # connection open past the cap.
        self._client = client or VideoCompileClient()

    @abstractmethod
    def build_shots(self, ctx: RunContext) -> list[ShotDict]:
        """Return compositor shot dicts; raise RuntimeError if required prior
        outputs are missing (distinct from an empty-but-valid shot list)."""

    @abstractmethod
    def build_output(self, result: PayloadDict, ctx: RunContext) -> O:
        """Map the validated compile response to the stage's typed output."""

    async def build_audio_bed(
        self,
        ctx: RunContext,
        shots: list[ShotDict],
        *,
        degradations: list[Degradation] | None = None,
    ) -> PayloadDict | None:
        """Request-level continuous music + ambient bed for the whole video.

        Returns ``{"music": {...}, "ambient": [...]}`` (either key optional) or
        ``None`` for no bed. The base is a no-op so every non-sound compile stage
        is unchanged; ``promo_render`` overrides it. ``shots`` is the FINAL list
        (post title-card splice + ken-burns) so a subclass can size ambient
        windows against the encoded duration.

        ``degradations`` is an optional SINK: an override that resolves catalog
        ids to URLs appends a ``sound_bed_missing`` entry to it for every chosen
        bed it could not resolve, so a video that ships silent says so in the run
        record. Keyword-only with a ``None`` default, so an override that has no
        such failure mode (and every existing caller/test) is unaffected.
        """
        return None

    def compile_tags(self, ctx: RunContext) -> dict[str, str]:
        """Tags forwarded on the compile call.

        Default carries the deliverable ``format`` + ``captions_mode``; the
        ``video_render`` stage overrides this to the minimal run_id/stage pair.
        """
        deliv = ctx.config_snapshot.get("deliverable") or {}
        return {
            "run_id": str(ctx.run_id),
            "stage": self.name,
            "format": str(deliv.get("format", "16x9")),
            "captions_mode": captions_mode(ctx),
            **resolution_tags(ctx),
        }

    def caption_clip_refs(self, ctx: RunContext) -> dict[str, ClipRef]:
        """Per-line narration clips to word-time-caption, keyed by shot line_id.

        Base returns {} → no word-timed captions (byte-identical). A stage opts in
        by overriding this AND setting deliverable captions_mode=burned_in.
        """
        return {}

    def _overlay_static_cues(
        self, ctx: RunContext, shots: list[ShotDict]
    ) -> list[ShotCaptionsPayload] | None:
        """``static_caption_cues`` in overlay mode ONLY -- burned_in keeps its
        static drawtext path and never receives shot-spanning cues."""
        if captions_mode(ctx) != "overlay":
            return None
        return self.static_caption_cues(ctx, shots)

    def static_caption_cues(
        self, ctx: RunContext, shots: list[ShotDict]
    ) -> list[ShotCaptionsPayload] | None:
        """Shot-spanning caption cues for a render with NO narration to word-time.

        Consulted only in ``overlay`` mode when no word-timed cues were built, so a
        silent (captions-only) film still gets its soft mov_text track instead of
        no captions at all. Base returns ``None`` (byte-identical).
        """
        return None

    def resolve_cadence(self, ctx: RunContext) -> int | None:
        """The master frame rate for this render, or ``None`` to leave it to the compiler.

        The compositor's ``CompileRequest.cadence`` is a single int master-cadence
        override (snapped to {24,25,30,50,60}), not a struct — so this returns the
        resolved target fps, and ``None`` means "send no cadence key", which is
        what keeps a run with no cadence intent byte-identical to a pre-cadence
        compile.

        The base resolves it the way every product has since ORCH-0.6: purely from
        the run config (``shared/cadence.build_cadence_spec``). It is a HOOK rather
        than an inline call because that resolution is config-only, and a product
        whose master rate is a property of its FOOTAGE rather than of its config
        has nowhere else to say so — ``build_shots`` returns shots, and per-shot
        ``Shot.target_fps`` is a per-source conform, not the master. ``action_reel``
        overrides this to fold in ``footage_probe``'s measured source rates.

        Subclasses must keep the ``None``-means-omit contract: returning the
        compiler's own default (30) instead of ``None`` would start sending a
        cadence key on runs that never asked for one.
        """
        spec = build_cadence_spec(ctx.config_snapshot)
        return spec.target_fps if spec is not None else None

    def title_card_plan(self, ctx: RunContext) -> TitleCardPlan | None:
        """THE cards this compile renders — resolved once, read by every consumer.

        Base: the :meth:`title_card_config` block exactly as configured (no order
        cap), today's behaviour for every product whose script does not budget
        around the cards (promo's default end card is already sized by its fill
        plan, so it is planned uncapped too).
        A product whose script RESERVES the card time overrides this to return what
        that script recorded, never a fresh computation (see ``website_review``):
        the render and the cache-hit accounting both take this one object.
        """
        return plan_title_cards(self.title_card_config(ctx), ordered_seconds=None)

    def title_cards_enabled(self, ctx: RunContext) -> bool:
        """Whether the opt-in ``title_card`` block may splice intro/outro SHOTS.

        Base: ``True`` — byte-for-byte today's behaviour for every product. The
        gate is still ``title_card.enabled`` in ``plan_title_cards``; this
        hook only lets a product say the SPLICE is wrong for it.

        Why a product would say that: a spliced card is a free-standing shot
        prepended to the list, and NOTHING rebases the timeline after it. On a
        beat-cut product that is fatal and silent — the request-level music bed
        has no start/offset field (track t=0 IS video t=0), so a 3 s intro shifts
        every planned cut 3 s off the beat while the duration QA still passes
        (declared and actual totals both include the card).

        The telling detail: the request-level VOICE track IS compensated — it is
        sized against the POST-splice shot list the base hands ``build_audio_bed``
        (see ``action_reel/compile._build_voice_track``) — and the music bed is
        not. One of the two request-level audio streams being rebased and the
        other not is an oversight, not a design, which is why this is a hook and
        not a new compiler field.
        """
        return True

    def title_card_config(self, ctx: RunContext) -> object:
        """The ``title_card`` block the intro/outro splice renders from.

        Base: the run's own ``title_card`` config block, unchanged — byte-for-byte
        today's behaviour. A product with a DEFAULT card (promo's branded end
        card) overrides this; ``plan_title_cards`` stays the only gate.
        """
        return ctx.config_snapshot.get("title_card")

    def ken_burns_local_fallback(self, ctx: RunContext) -> bool:
        """Whether a still whose Ken Burns recording failed is animated locally.

        Base: ``False`` — a failed recording keeps its still, byte-identical to
        before. Promo opts in (see ``shared/ken_burns_local``): its cover-fit
        stills are cropped to the frame aspect before the recorder sees them, a
        still the recorder fails is rendered locally, and near-static clips are
        recorded.
        """
        return False

    def ken_burns_local_only(self, ctx: RunContext) -> bool:
        """Whether EVERY still is animated locally, never sent to the recorder.

        Base: ``True`` for EVERY product (owner decision 2026-10-20). The
        video_production recorder screenshots every frame in a browser, which for
        one 20s 1440p still measured **977s of execution with zero queue wait**
        (run 01a156a8-978b, 2026-10-20, ``per_frame_ms=1501``, ``capture_scale=2``)
        — 13s inside the 1000s pass budget — and a live generic run timed out on
        the same ~1000s-per-still cost. The local ffmpeg move
        (``shared/ken_burns_local``) renders the same never-crop move in seconds.
        Implies the local passes (``ken_burns_local_fallback``). A product may
        still override this to ``False`` to send stills to the recorder.
        """
        return True

    def caption_look(self, ctx: RunContext) -> dict[str, object] | None:
        """The burned-in captions' typography, sent to the compiler.

        Base: ``None`` — the compiler's default look, byte-identical. A product
        whose captions must match its cards overrides this (promo, per-line).
        """
        return None

    def captions_karaoke(self, ctx: RunContext) -> bool:
        """Whether burned-in captions carry per-word karaoke timing.

        Base: the value pinned in the run snapshot at creation (a run override or the
        tenant default, P6), else the ``PIPELINE_CAPTIONS_KARAOKE`` setting. A product
        whose caption line must sit in ONE continuous box turns it off (the
        compiler draws its box per karaoke run — see promo's override).
        """
        return resolve_captions_karaoke(
            ctx.config_snapshot, get_settings().pipeline.captions_karaoke
        )

    def title_card_settled_lead_seconds(self, ctx: RunContext) -> float:
        """Seconds of the title-card template's entrance to cut off the head.

        Base: ``0.0`` — the card plays from its first frame, byte-identical. A
        product whose card must read from the cut renders the card this much
        LONGER and starts it this far in (``Shot.clip_lead_seconds``), so the
        build-in animation is never on screen (see ``render_title_card_plan``).
        """
        return 0.0

    def _emit_shot_timeline(self, ctx: RunContext) -> bool:
        """Whether to request the compiler's authoritative shot_timeline. Base:
        never (probe-free, byte-identical). Only DemoVideoCompileStage overrides."""
        return False

    async def handle(self, payload: I, ctx: RunContext) -> O:
        # Opt in to per-phase timing accumulation for THIS call (see
        # `phase_tracker.reset_phase_totals`) so the `title_cards`/`presign_1`/
        # `ken_burns`/`presign_2`/`sound`/`audio_bed`/`captions`/`stage_call`
        # phases below can be persisted onto the output further down instead of
        # only ever reaching a log line. Instrumentation only: does not change
        # what any phase does or how long it is allowed to take.
        reset_phase_totals()
        shots = self.build_shots(ctx)
        if not shots:
            raise StagePreconditionError(f"{self.name}: {self.no_shots_message}")

        assets = await self._run_presubmit(ctx, shots)

        # compile() polls up to _COMPILE_TIMEOUT_SECONDS, LIVENESS-based: it keeps
        # polling GET /api/jobs/{id} as long as the job answers `processing` (a
        # confirmed-lost job — bounded consecutive 404s — fails fast well before
        # this deadline; see video_compile.py's `_await_job`). This is NOT routed
        # through `w2v_common.stages.stage_call` — see `_compile_with_backstop`
        # for why that would silently reintroduce the exact defect being fixed
        # here.
        async with phase("stage_call"):
            result = await _compile_with_backstop(
                self._client.compile(
                    shots=assets.shots,
                    tags=self.compile_tags(ctx),
                    music=assets.bed.get("music"),
                    ambient=assets.bed.get("ambient"),
                    mastering=assets.mastering,
                    finish=assets.finish,
                    cadence=assets.cadence,
                    captions=assets.captions,
                    captions_render=captions_mode(ctx),
                    # Per-video caption look tracks the SAME run-level mood as the
                    # colour grade (resolve_mood is the grade's own resolver);
                    # karaoke is a pipeline kill-switch. Both are folded into the
                    # payload by the client ONLY when `captions` is present, so a
                    # captions-off compile is byte-identical.
                    caption_mood=resolve_mood(ctx.config_snapshot),
                    captions_karaoke=self.captions_karaoke(ctx),
                    caption_look=self.caption_look(ctx),
                    emit_shot_timeline=self._emit_shot_timeline(ctx),
                    timeout=_COMPILE_TIMEOUT_SECONDS,
                    # Stable across a resumable-timeout re-dispatch (run_id is
                    # fresh per regenerate; one compile per run per stage) so the
                    # compiler dedupes a re-submit instead of running a second paid
                    # render. The discriminator keeps a footage-critique re-render
                    # (same run_id + name, DIFFERENT shots) from deduping to the
                    # original — empty for a normal render (fully resume-safe),
                    # non-empty for a re-render.
                    idempotency_key=_compile_idempotency_key(ctx, self.name),
                ),
                stage_name=self.name,
            )
        validate_compile_result(result, stage=self.name)
        output = self.build_output(result, ctx)
        # Auto-QA: the compiler reports the ACTUAL ffprobe'd duration alongside
        # the DECLARED sum-of-shots. A material gap means a truncated / short-
        # rendered file (e.g. 234s of frames under a 263.5s declaration). Surface
        # it through the shared notes channel → the run reports `degraded` without
        # failing (the artifact still exists and may be usable/salvageable).
        # The compiler's mastering gate is a property of ITS response, not of any one
        # blueprint, so it is read here for all nine CompileStageBase subclasses. It
        # used to live in demo_video/compile.py, where a blank master degraded exactly
        # one blueprint and shipped silently from the other eight.
        duration_qa_notes = _compile_duration_qa_notes(result)
        qa_notes = [
            *assets.notes,
            *mastering_gate_notes(result.get("mastering")),
            *duration_qa_notes,
        ]
        # Only the final-video compile outputs carry a `notes` channel; the
        # intermediate compiles (shot_sound bed, ken_burns clip) deliberately do
        # not, so a duration quirk on an intermediate never flags the run.
        if qa_notes and "notes" in type(output).model_fields:
            existing: list[str] = getattr(output, "notes", [])
            output = output.model_copy(update={"notes": [*existing, *qa_notes]})
        # The STRUCTURED channel, same shape and the same guard. Unlike the notes
        # above, the mastering entry is NOT narrowed to a blank picture: every
        # `gate_pass=false` is recorded (see `mastering_gate_degradations`), which
        # is the 207-of-214 population that used to ship with no record at all.
        # APPENDED, never assigned: `build_output` may already have recorded a
        # duration-conformance breach on this same output.
        # `assets.degradations` carries the coded twin of every pre-submit note in
        # `assets.notes` (ken burns stills, dropped SFX, fabricated holds, ...) and
        # the duration QA notes get theirs here, so none of the prose merged above
        # is left for the DAG executor to stamp `unclassified_note`.
        stage_degradations = [
            *mastering_gate_degradations(result.get("mastering")),
            *assets.degradations,
            *music_bed_degradations(result, assets.bed),
            *_compile_duration_qa_degradations(duration_qa_notes),
        ]
        if stage_degradations and "degradations" in type(output).model_fields:
            carried: list[Degradation] = getattr(output, "degradations", [])
            output = output.model_copy(update={"degradations": [*carried, *stage_degradations]})
        # The compiler's ADVISORY tier (`MasteringReport.advisories`, e.g. a residual
        # banding breach it measured and deliberately did not gate). Its own field, never
        # `degradations` / `notes`: the compiler kept these out of `violations[]` so they
        # must not flip `degraded`. Absent key (today's compiler) ⇒ `[]` ⇒ unchanged.
        mastering_advisories = [
            *mastering_gate_advisories(result.get("mastering")),
            *assets.advisories,
        ]
        if mastering_advisories and "advisories" in type(output).model_fields:
            kept: list[Degradation] = getattr(output, "advisories", [])
            output = output.model_copy(update={"advisories": [*kept, *mastering_advisories]})
        # The RAW mastering report, same shape and the same guard. The GATE's
        # verdict already reaches every blueprint above; these are the numbers
        # BEHIND the verdict (loudness, true peak, colour tags) plus the delivered
        # file's content hash. They used to be populated in demo_video's own
        # `build_output` ONLY, so on lesson / action_reel / app_review /
        # website_review a downstream reader could see that the gate failed but
        # never by how much, and `final_critique` could not pass-read `loudness`.
        # Carried here for all of them — see `base.py::MasteringReportFields`.
        #
        # NON-CLOBBERING: a blueprint whose `build_output` already set one of
        # these keeps its own value. demo_video reads them off the same `result`
        # dict, so today the two agree; the guard is so that a lane which
        # deliberately post-processes a field (e.g. redacting or normalising a
        # report) is not silently overwritten by the raw one.
        # Per-phase timing (`title_cards`/`presign_1`/`ken_burns`/`presign_2`/
        # `sound`/`audio_bed`/`captions`/`stage_call`) accumulated since
        # `reset_phase_totals()` at the top of `handle`, plus the Ken Burns
        # shot count measured in `_run_presubmit` — see
        # `base.py::PhaseTimingFields` for why these exist and the same
        # NON-CLOBBERING contract as the mastering fields above. Read here
        # (not earlier) so the dict includes `stage_call`, the LAST phase to
        # close before this line runs.
        phase_seconds = phase_totals()
        report_updates = {
            key: value
            for key, value in (
                ("mastering", _as_dict_or_none(result.get("mastering"))),
                ("output_sha256", _as_str_or_none(result.get("output_sha256"))),
                ("loudness", _as_dict_or_none(result.get("loudness"))),
                ("phase_seconds", phase_seconds or None),
                ("ken_burns_shot_count", assets.ken_burns_shot_count),
                ("audio_bed", audio_bed_record(assets.bed, result)),
                ("ken_burns_cache_hits", assets.ken_burns_cache_hits),
                ("title_card_cache_hits", assets.title_card_cache_hits),
            )
            if value is not None
            and key in type(output).model_fields
            and getattr(output, key, None) is None
        }
        if report_updates:
            output = output.model_copy(update=report_updates)
        return output

    async def _prepare_local_stills(
        self,
        ctx: RunContext,
        shots: list[ShotDict],
        notes: list[str],
        *,
        degradations: list[Degradation] | None = None,
    ) -> set[int] | None:
        """Before Ken Burns, for a product that opted in to the local still passes.

        Cover-fit stills get the frame's own aspect IN PLACE (``shots`` is
        mutated), so the recorder has no pillar to blur. Returns the indexes that
        are stills going in — ``None`` when the product did not opt in.
        """
        if not (self.ken_burns_local_fallback(ctx) or self.ken_burns_local_only(ctx)):
            return None
        shots[:] = await crop_cover_stills_to_frame(
            ctx, shots, notes=notes, degradations=degradations
        )
        return {i for i, s in enumerate(shots) if s.get("image_url") and not s.get("video_url")}

    async def _rescue_local_stills(
        self,
        ctx: RunContext,
        shots: list[ShotDict],
        was_still: set[int] | None,
        *,
        notes: list[str],
        advisories: list[Degradation],
        budget_seconds: float,
        degradations: list[Degradation] | None = None,
    ) -> list[ShotDict]:
        """After Ken Burns: a still the recorder failed never ships frozen.

        The orchestrator animates it itself, inside what is left of the
        pre-submit budget (the tail reserve stays reserved), and measures the
        clips the recorder did deliver. No-op unless the product opted in.
        """
        if was_still is None:
            return shots
        local_only = self.ken_burns_local_only(ctx)
        if not local_only:
            await flag_near_static_clips(shots, was_still, advisories=advisories)
        return await rescue_frozen_stills(
            ctx,
            shots,
            notes=notes,
            advisories=advisories,
            master_fps=self.resolve_cadence(ctx),
            budget_seconds=budget_seconds,
            degradations=degradations,
            local_first=local_only,
        )

    def _hand_ken_burns_its_budget(
        self,
        ctx: RunContext,
        cm: asyncio.Timeout,
        was_still: set[int] | None,
        stills: int,
    ) -> float:
        """Make room for the Ken Burns pass's OWN budget; return the seconds added.

        Only when there is something to animate: a run with no bare still (or
        with Ken Burns opted out) keeps the base 1800s block untouched.
        """
        remaining = _presubmit_remaining(cm)
        extended_by = 0.0
        if stills and _ken_burns_cfg(ctx.config_snapshot) is not None:
            extended_by = _ensure_ken_burns_room(cm, was_still)
        logger.info(
            "%s: ken_burns pass budget %.0fs for %d still(s) — %.0fs of the "
            "pre-submit deadline was left, extended by %.0fs",
            self.name,
            _KEN_BURNS_PASS_BUDGET_SECONDS,
            stills,
            remaining,
            extended_by,
        )
        return extended_by

    async def _recorder_ken_burns_pass(
        self,
        ctx: RunContext,
        cm: asyncio.Timeout,
        shots: list[ShotDict],
        was_still: set[int] | None,
        stills: int,
        *,
        notes: list[str],
        cache_hits: list[int],
        degradations: list[Degradation],
    ) -> tuple[list[ShotDict], float]:
        """The video_production recorder pass; ``(shots, seconds the deadline moved)``.

        The pass's OWN budget, not what title cards left over: the aggregate
        deadline is moved out to hold it (plus the rescue and tail reserves) — see
        ``_ensure_ken_burns_room``. Skipped entirely for a local-only product
        (``ken_burns_local_only``): no recorder pass means no recorder budget to
        make room for, so the local renders after it run inside the base deadline.
        """
        if self.ken_burns_local_only(ctx):
            return shots, 0.0
        extended_by = self._hand_ken_burns_its_budget(ctx, cm, was_still, stills)
        shots = await apply_ken_burns(
            ctx,
            shots,
            notes=notes,
            master_fps=self.resolve_cadence(ctx),
            budget_seconds=_KEN_BURNS_PASS_BUDGET_SECONDS,
            cache_hits=cache_hits,
            degradations=degradations,
        )
        return shots, extended_by

    async def _run_presubmit(self, ctx: RunContext, shots: list[ShotDict]) -> _PresubmitAssets:
        """Run the pre-submit asset-prep block under one aggregate deadline.

        Everything here — title cards, two re-sign passes, ken burns, sound, the
        audio bed, mastering/finish/cadence resolution, and captions — used to run
        in ``handle`` with NO aggregate deadline at all (see
        ``_PRESUBMIT_DEADLINE_SECONDS`` for the incident). It now runs under
        ``asyncio.timeout``, and ``presubmit_phase`` tracks which named phase was
        active so a deadline hit names WHERE it died. ``asyncio.timeout`` (not
        ``wait_for``) is deliberate: it cancels IN-PLACE on the current task rather
        than spawning a child task with its own copied context, so phase
        attribution does not depend on ContextVar propagation across tasks —
        ``presubmit_phase`` is a plain local, correct regardless.

        Re-entry after a pod death (audited 2026-10-06 for the step ledger): this
        block restarts from the top, and no phase other than the caption ASR
        fallback needs ``ctx.step_ledger`` — do not wrap the rest.

        * ``title_cards`` — content-addressed; a stored card is a cache hit.
        * disclaimer splice, mastering / finish / cadence — pure, no I/O.
        * ``presign_1`` / ``presign_2`` / ``sound`` / ``audio_bed`` — presigns of
          objects we already hold plus a sound-library read; unpaid, and they MUST
          repeat (a stored signature is an expiring handle, ledger rule 3). The
          ``action_reel`` bed is the same: a presign and pure cue arithmetic.
        * ``ken_burns`` — recorder jobs are pinned and carry an Idempotency-Key
          (``shared/ken_burns.py``). The local passes around it (cover crop, the
          frozen-still rescue, the near-static probe) are in-pod Pillow/ffmpeg
          writing deterministic media ids: repeated CPU, never a second bill.
        * ``captions`` — the per-clip ASR fallback is the one paid call, and it is
          ledgered inside ``build_shot_captions``.

        The submit that follows carries ``_compile_idempotency_key``.
        """
        presubmit_phase = "title_cards"
        # Held so the handler below can ask `cm.expired()` whether OUR aggregate
        # deadline is what fired. Without that check, an INNER timeout — e.g.
        # ken_burns' own per-item budget (its pass budget + overhead), or a title-card
        # render hitting its 600s cap — would be caught here and relabelled as an
        # aggregate-deadline breach, sending the next reader after the wrong budget.
        presubmit_notes: list[str] = []
        # Structured twin of `presubmit_notes` -- see `_PresubmitAssets.degradations`.
        presubmit_degradations: list[Degradation] = []
        presubmit_advisories: list[Degradation] = []
        title_card_hits: list[str] = []
        ken_burns_hits: list[int] = []
        title_card_cache_hits: int | None = None
        # A shot whose duration `assemble_shots` could not measure carries a 1s
        # FABRICATED hold. Drain the markers here — the one place EVERY
        # CompileStageBase subclass passes through — so the substitution reaches
        # the persisted output instead of shipping as a confident 1-second shot,
        # and so the orchestrator-internal marker never reaches the compiler.
        #
        # Placement, precisely: it must run BEFORE `splice_title_cards` below,
        # which prepends/appends shots this drain has no marker for and would
        # otherwise shift nothing but add noise to the scan. It sits outside the
        # deadline block only because it is pure dict work — no I/O, no await —
        # so it can neither consume the aggregate budget nor be cancelled by it.
        # It does NOT survive a deadline hit: both branches of the
        # `except TimeoutError` below RE-RAISE, so an asset phase timing out
        # fails the stage, no output is built, and these notes die with it. The
        # note is a soft-degradation channel for a compile that SUCCEEDS; a
        # timeout is a hard failure and is reported as one.
        presubmit_notes.extend(
            drain_fabricated_hold_notes(shots, degradations=presubmit_degradations)
        )
        cm = asyncio.timeout(_PRESUBMIT_DEADLINE_SECONDS)
        # Seconds the Ken Burns entry moved the deadline out (see `_ensure_ken_burns_room`).
        extended_by = 0.0
        try:
            async with cm:
                presubmit_phase = "title_cards"
                async with phase("title_cards"):
                    # Opt-in animated intro/outro title card, injected after the
                    # empty-guard (a title-card-only video is never meaningful) and
                    # before re-signing so the freshly stored card URL gets the same
                    # TTL refresh as every shot. Gated on `title_cards_enabled` so a
                    # beat-locked product can decline the SPLICE (which no pass
                    # rebases) without losing its title — see the hook.
                    if self.title_cards_enabled(ctx):
                        card_plan = self.title_card_plan(ctx)
                        intro, outro = await render_title_card_plan(
                            ctx,
                            card_plan,
                            cfg=self.title_card_config(ctx),
                            settled_lead_seconds=self.title_card_settled_lead_seconds(ctx),
                            notes=presubmit_notes,
                            degradations=presubmit_degradations,
                            cache_hits=title_card_hits,
                        )
                        title_card_cache_hits = _title_card_cache_hits(card_plan, title_card_hits)
                        shots = splice_title_cards(shots, intro, outro)
                    # A18 legal tail (AC-8): the disclaimer card ``claim_finish``
                    # mandated, as an existing solid-text ``Shot.description`` (no
                    # compiler change). Splices AFTER the outro so the legal line is
                    # genuinely last. `None` — the case for every run that did not
                    # opt into `legal.mode` — returns the list UNCHANGED (AC-1).
                    shots = splice_disclaimer_card(shots, build_disclaimer_shot(ctx))

                presubmit_phase = "presign_1"
                async with phase("presign_1"):
                    # FIRST re-sign: ``apply_ken_burns`` FETCHES each still's
                    # ``image_url`` itself. On a paused/iterated run those stored
                    # signatures are already past their 1h TTL, so the fetch 403s
                    # (SignatureDoesNotMatch), the per-shot blanket soft-fail
                    # swallows it, and ken_burns silently does nothing on exactly
                    # the runs that need it most. Re-signing HERE hands it live
                    # URLs.
                    shots = await refresh_shot_asset_urls(shots)

                presubmit_phase = "ken_burns"
                # Measured immediately before the call, on the same eligibility
                # test `apply_ken_burns` itself uses (still image, no video) —
                # see the module docstring in `shared/ken_burns.py`. This is the
                # denominator for `phase_seconds["ken_burns"]`: the discriminator
                # for the compile-time regression this instruments is "how many
                # stills need a Ken Burns job", and a raw phase total cannot
                # answer that alone.
                ken_burns_shot_count = sum(
                    1 for s in shots if s.get("image_url") and not s.get("video_url")
                )
                async with phase("ken_burns"):
                    was_still = await self._prepare_local_stills(
                        ctx, shots, presubmit_notes, degradations=presubmit_degradations
                    )
                    shots, extended_by = await self._recorder_ken_burns_pass(
                        ctx,
                        cm,
                        shots,
                        was_still,
                        ken_burns_shot_count,
                        notes=presubmit_notes,
                        cache_hits=ken_burns_hits,
                        degradations=presubmit_degradations,
                    )
                    shots = await self._rescue_local_stills(
                        ctx,
                        shots,
                        was_still,
                        notes=presubmit_notes,
                        advisories=presubmit_advisories,
                        budget_seconds=_presubmit_remaining(cm) - _PRESUBMIT_TAIL_RESERVE_SECONDS,
                        degradations=presubmit_degradations,
                    )

                presubmit_phase = "presign_2"
                async with phase("presign_2"):
                    # SECOND re-sign, and NOT a redundant one: it covers the URLs
                    # the pass above just MINTED (the freshly stored ken_burns
                    # clips) plus anything the tail added, and it moves every
                    # asset's TTL window to seconds-before-render rather than
                    # seconds-before-ken-burns — a ken_burns pass can take minutes.
                    # ``refresh_shot_asset_urls`` is idempotent by construction: it
                    # re-signs the object a URL points at and, on any decline/error,
                    # keeps the value it was given, so a second pass over
                    # already-fresh URLs is safe.
                    shots = await refresh_shot_asset_urls(shots)

                presubmit_phase = "sound"
                async with phase("sound"):
                    # Resolve per-shot sound-effect catalog ids → freshly presigned
                    # sound-library URLs (no-op when no shot carries sound).
                    # Same sink as ken_burns/title_cards above: a dropped SFX cue
                    # ships a film with a missing sound bed, and without this the
                    # only record died with the pod.
                    shots = await resolve_shot_sound_urls(
                        shots, notes=presubmit_notes, degradations=presubmit_degradations
                    )

                presubmit_phase = "audio_bed"
                async with phase("audio_bed"):
                    # Request-level continuous music/ambient bed (None for
                    # non-sound stages).
                    # `degradations` is a SINK, mirroring the `notes=` sink
                    # `resolve_shot_sound_urls` takes above: a chosen music or
                    # ambient bed that fails to presign is dropped silently
                    # inside `sound_design.resolve_audio_bed`, shipping a video
                    # with no bed and leaving no trace anywhere.
                    bed = (
                        await self.build_audio_bed(ctx, shots, degradations=presubmit_degradations)
                        or {}
                    )

                # Deterministic mastering profile (ORCH-0.5). Presence-is-opt-in:
                # absent `mastering` block ⇒ None ⇒ no `mastering` key on the
                # request, so the compile request is byte-identical to before.
                mastering_profile = build_mastering_profile(ctx.config_snapshot)
                mastering = mastering_profile.as_mastering_payload() if mastering_profile else None
                # Deterministic color finish (ORCH-0.4 letterbox + ORCH-3.3
                # whole-film grade). Same presence-is-opt-in contract as
                # `mastering` — an absent `finish` block ⇒ None ⇒ no `finish` key
                # on the request ⇒ byte-identical payload. `build_finish_spec` runs
                # the deterministic `colorist` off this snapshot's mood/style/
                # design tokens when the grade layer is opted in
                # (`finish.enabled`), so the finishing-exclusive grade knobs
                # (look/tints/vignette/grain/glow/sharpen/grade_intensity) ride
                # inside the `finish` dict — no `VideoCompileClient.compile`
                # signature change. The grade is gated behind
                # `PIPELINE_GRADE_DEFAULT_ON_SHOWCASE` (default off) until the
                # compiler's grade chain ships, and only rides when actually set
                # (see FinishSpec.as_finish_payload). `broadcast_safe` is still
                # never sent (a pre-encode luma clamp cannot survive libx264
                # ringing); the tonal dials stay identity — per-shot ColorAdjust
                # owns those.
                finish_spec = build_finish_spec(
                    ctx.config_snapshot,
                    grade_enabled=get_settings().pipeline.grade_default_on_showcase,
                )
                finish = finish_spec.as_finish_payload() if finish_spec else None
                # Opt-in cadence (ORCH-0.6): a resolved master fps rides the
                # compile request. Absent `cadence` block ⇒ None ⇒ the kwarg is
                # dropped in the client ⇒ byte-identical request payload.
                # Resolution goes through the `resolve_cadence` hook so a product
                # whose master rate comes from its FOOTAGE (action_reel) can
                # answer it without the base learning about footage. Per-shot
                # retime still rides Shot.target_fps/retime.
                cadence = self.resolve_cadence(ctx)

                presubmit_phase = "captions"
                async with phase("captions"):
                    # Word-timed captions (ORCH-4.2). Gated on the kill-switch AND
                    # a deliverable captions_mode of burned_in (baked into the
                    # picture) or overlay (muxed as a selectable soft mov_text
                    # track); a stage opts in by overriding `caption_clip_refs`.
                    # The cue-building below is mode-agnostic — burn-vs-mux is
                    # decided compiler-side off `captions_render` (sent on the
                    # compile call). Base returns {} ⇒ no build ⇒ byte-identical
                    # request. Captions must NEVER fail the render — any error
                    # soft-fails to no captions.
                    captions: list[ShotCaptionsPayload] | None = None
                    if get_settings().pipeline.captions_default_on_showcase and captions_mode(
                        ctx
                    ) in ("burned_in", "overlay"):
                        clip_refs = self.caption_clip_refs(ctx)
                        if clip_refs:
                            # Constructed per compile; MUST be closed or every
                            # captioned render leaks an httpx pool on the
                            # long-lived multi-replica pod (same leak class fixed
                            # in _narrate_via_gateway). try/finally, not just
                            # except.
                            asr = TranscriptionWorkerClient()
                            try:
                                # _get_storage() is typed to the abstract
                                # _StorageLike (the seam tests inject fakes
                                # through); the live singleton is always the
                                # concrete StorageService, which carries
                                # retrieve_stage_artifact.
                                storage = cast(StorageService, await _get_storage())
                                # Its own timeout, inside the aggregate one: a
                                # slow ASR fallback costs the subtitles
                                # (CAPTIONS_MISSING below), never the render.
                                async with asyncio.timeout(
                                    _presubmit_remaining(cm) - _PRESUBMIT_FINAL_MARGIN_SECONDS
                                ):
                                    captions = await build_shot_captions(
                                        final_shots=shots,
                                        clip_refs=clip_refs,
                                        job_id=str(ctx.run_id),
                                        storage=storage,
                                        asr=asr,
                                        ledger=ctx.step_ledger(self.name),
                                    )
                            except Exception:  # captions must never fail the render
                                logger.warning(
                                    "word-timed caption build failed; rendering without",
                                    exc_info=True,
                                )
                                captions = None
                                # A log line is not a record: captions were
                                # REQUESTED (the kill-switch is on, the mode is
                                # burned_in/overlay, and this stage published
                                # clip refs) and EVERY subtitle is now missing
                                # from a delivered video. That was invisible in
                                # the run record until this entry.
                                presubmit_degradations.append(
                                    degradation(
                                        CAPTIONS_MISSING,
                                        "word-timed caption build failed; the delivered "
                                        f"video has NO subtitles ({len(clip_refs)} clip "
                                        "refs were submitted)",
                                    )
                                )
                            finally:
                                # BaseServiceClient teardown is close(), not aclose().
                                await asr.close()
                        # Overlay with nothing to word-time (a captions-only film
                        # has no narration clips): fall back to the stage's
                        # shot-spanning static cues so the soft track still ships.
                        captions = captions or self._overlay_static_cues(ctx, shots)
                    # Double-caption suppression: a shot that produced a word-timed
                    # cue must NOT also carry its static per-shot `caption` string.
                    # In burned_in that would burn the line twice (static drawtext
                    # + timed ASS); in overlay it would burn the static line into
                    # the picture while the word-timed track is muxed soft —
                    # either way the static caption is redundant. Clear it ONLY for
                    # shots that actually got cues, so a line whose ASR/download
                    # failed keeps its reliable static caption (no regression vs
                    # the pre-captions baseline on a transcription outage).
                    # Author-driven overrides on uncued shots are untouched.
                    if captions:
                        for entry in captions:
                            idx = entry["shot_index"]
                            if 0 <= idx < len(shots):
                                shots[idx].pop("caption", None)

                return _PresubmitAssets(
                    shots=shots,
                    bed=bed,
                    mastering=mastering,
                    finish=finish,
                    cadence=cadence,
                    captions=captions,
                    notes=presubmit_notes,
                    degradations=presubmit_degradations,
                    ken_burns_shot_count=ken_burns_shot_count,
                    ken_burns_cache_hits=len(ken_burns_hits) if ken_burns_shot_count else None,
                    title_card_cache_hits=title_card_cache_hits,
                    advisories=presubmit_advisories,
                )
        except TimeoutError as exc:
            if not cm.expired():
                # An INNER budget fired (ken_burns per-item, a title-card render).
                # Its message is more precise than ours and names the right number —
                # let it through untouched rather than relabelling it.
                raise
            # PhaseDeadlineError, NOT a bare TimeoutError: an unstamped timeout
            # is attributed `unknown` by `attribute_exception`, which is the exact
            # symptom this whole change exists to remove.
            raise PhaseDeadlineError(
                f"{self.name}: pre-submit asset block exceeded "
                f"{_PRESUBMIT_DEADLINE_SECONDS + extended_by:.0f}s aggregate deadline "
                f"(died in phase={presubmit_phase!r})"
            ) from exc
