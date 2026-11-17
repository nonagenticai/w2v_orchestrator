"""final_critique — the A11 advisory ship gate (PR-1, critique_only, inert-by-default).

A SOFT, advisory-only final-render critic that runs AFTER the footage-critique
loop. It measures whole-film gestalt from the video_analyzer output the pipeline
already produces plus the compiler's already-measured final-mux loudness, scores
it (optionally via a ``film-critic`` WM worker), and hands a quality badge +
chapters to ``growth_bundle`` and the delivered artifact through to
``render_manifest``.

Invariants (do not weaken):
  * NEVER hard-blocks UNLESS the run OPTED IN via ``ship_gate.enforce`` (default off).
    The exceptions that may leave ``handle`` are ``StageSkippedError`` (opt-out), a genuine
    ``StagePreconditionError`` when NEITHER footage_critique NOR compile produced an
    artifact, and — only when ``enforce`` is true AND ``mode == auto`` AND the best draft
    the bounded re-drive could produce STILL fails the ceilings — a terminal
    ``ShipGateEnforcementError`` (the A11 hard gate; #719 mechanism → run ``failed``).
    Every OTHER failure — analyzer outage, WM error, malformed model output, an internal
    defect — degrades to *accept the current artifact* with an advisory finding. With
    ``enforce`` false (the default) the gate is byte-for-byte identical to accept-best.
  * Byte-identical to today when ``ship_gate`` is absent or ``mode == "off"``: the
    stage skips-clean, its output is ``{}``, and downstream reads compile's
    artifact unchanged.
  * ``mode == "auto"`` fires a BOUNDED (≤2-pass) in-stage re-drive: it re-RUNS the
    owning per-dimension LLM stage named by the film-critic's clamped directives —
    ``demo_video_footage_direction`` / ``demo_video_sound_director`` feed compile
    directly, ``voice_director`` re-drives narration via a ``demo_video_narrate``
    re-synth — threading the critique feedback into that stage's own LLM prompt,
    merges the regenerated items back by id (only the flagged ids), re-compiles ONCE
    and re-scores, keeps the highest-scoring draft (never regresses — a worse pass
    ships the earlier artifact) and NEVER blocks. It NEVER enqueues sub-DAG work,
    starts a child run, or re-drives ``demo_video_footage_critique`` /
    ``final_critique`` / ``growth_bundle``. Every re-drive throw site degrades to
    *accept the current best*. ``critique_only`` (and ``off`` / absent) stay
    byte-identical: no director/narrate/compile stage is constructed or called,
    ``rerun_stages == []``, and clamped directives persisted advisory-only.

The gestalt math is pure and lives in ``stages/shared/gestalt_metrics``; this
handler only gathers its inputs off ``ctx`` + the analyzer, fuses them, and (when
the gestalt is not already ship-ready) asks the WM critic. LLM calls route through
``worker_management`` (the ``WMClient`` gateway) — never a provider directly.

⚠️ KNOWN LIMIT — the visual terms are NOT currently measurable, and this stage cannot
fix that. ``video_analyzer_service`` emits per-frame records of
``{index, timestamp, path, description, ocr_text, labels}`` and no per-frame
``brightness`` / ``sharpness`` / ``dominant_color``. ``_feature_vector`` therefore
substitutes WHOLE-VIDEO means for two components and pads ``[0, 0, 0]`` for colour,
leaving ``motion`` as the only component that can differ between shots — and on a
near-static screen capture that is 0 as well. **Continuity, cohesion and tonal drift are
structurally unobservable on this wire shape**, and the lexical tone proxy is a ~24-word
keyword counter that real narration prose never trips.

What this stage guarantees is therefore HONESTY, not measurement: when the evidence
cannot support a term it persists ``None``, marks the run ``insufficient_evidence``,
emits NO Prometheus sample for that term, and degrades the badge to ``needs-review`` —
so the nightly calibration corpus stays empty-and-visibly-uncalibratable instead of
filling with fabricated ``0.0``/``1.0`` samples that would drive the derived ceilings to
a gate no real film can pass. Closing the gap needs per-frame visual features from the
analyzer; that is upstream work.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar, cast

from pydantic import Field
from w2v_common.stages import stage_call

from src.clients.schemas.video_analyzer import (
    VideoAnalyzerJobResult,
    VideoAnalyzerMotionSample,
    VideoAnalyzerVisualSample,
)
from src.clients.tts_client import TTSClient
from src.clients.video_analyzer_client import VideoAnalyzerClient
from src.clients.video_compile import VideoCompileClient
from src.config import get_settings
from src.observability import metrics
from src.services.llm import WMClient
from src.services.stages.base import (
    SLOT_USE_OPTIONAL,
    ArtifactUrl,
    SlotUse,
    StageHandler,
    StageInput,
    StageOutput,
    StageOutputWithDegradations,
    StagePreconditionError,
    StageSkippedError,
)
from src.services.stages.demo_video.compile import (
    DemoVideoCompileInput,
    DemoVideoCompileOutput,
    DemoVideoCompileStage,
)
from src.services.stages.demo_video.footage_direction import DemoVideoFootageDirectionStage
from src.services.stages.demo_video.narrate import DemoVideoNarrateInput, DemoVideoNarrateStage
from src.services.stages.demo_video.sound_director import DemoVideoSoundDirectorStage
from src.services.stages.demo_video.spine import resolve_plan_lines, resolve_plan_output
from src.services.stages.demo_video.voice_director import VoiceDirectorInput, VoiceDirectorStage
from src.services.stages.llm_budget import (
    LLM_RETRY_ON,
    MEDIUM_ANSWER_TOKENS,
    downshift_reasoning,
    llm_budget_seconds,
    llm_client_timeout_seconds,
    llm_retries,
    output_token_budget,
)
from src.services.stages.primitives.compile_stage import render_payload_discriminator
from src.services.stages.shared.degradation import FOOTAGE_BELOW_BAR, REVIEW_SKIPPED, degradation
from src.services.stages.shared.duration_conformance import (
    DURATION_TOLERANCE_FRACTION,
    demo_video_ordered_seconds,
    duration_conformance,
    observe_duration_conformance,
)
from src.services.stages.shared.footage_direction import FootageDirectionInput
from src.services.stages.shared.gestalt_metrics import (
    _DEFAULT_COHESION_FLOOR,  # single source of truth for the cohesion-floor default
    _DEFAULT_CONTINUITY_CEILING,  # single source of truth for the continuity ceiling
    LOUDNESS_FAIL,
    LOUDNESS_INSUFFICIENT_EVIDENCE,
    LOUDNESS_PASS,
    MOMENTUM_FAIL,
    MOMENTUM_INSUFFICIENT_EVIDENCE,
    MOMENTUM_PASS,
    MOMENTUM_UNCONFIGURED,
    assembled_momentum_curve,
    build_beat_intervals,
    gestalt_summary,
    loudness_window,
    momentum_gap,
    sample_frames_at_cuts,
)
from src.services.stages.shared.loop_engine import Critique, run_evaluator_optimizer
from src.services.stages.shared.retention import predicted_watch_curve
from src.services.stages.shared.sound_director import SoundDirectorInput
from src.services.stages.upstream import guarded
from src.utils import as_float, describe_exception, opt_float

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

log = logging.getLogger(__name__)

# WM worker name (regex ^[a-zA-Z0-9_-]+$). WM auto-creates it on the first call and
# the per-call system_prompt steers it; the provider/model lives on the WM worker
# config, NOT here — that's the whole point of routing through WM.
_WORKER_NAME = "film-critic"
_OFF = "off"
_MODE_DEFAULT = "critique_only"
_MODE_AUTO = "auto"

# ── A11 ship-gate ARMING ─────────────────────────────────────────────────────
# Per-TERM three-state verdict vocabulary. THREE states, not two, because the
# gate's evidence can be absent: "the judge measured and it was bad" and "the
# judge measured nothing" are different facts and only one of them may fail a
# run. `INSUFFICIENT_EVIDENCE` is never a pass either — it degrades the badge and
# is named in the record, so an unmeasurable term cannot masquerade as a clear.
TERM_PASS: str = "pass"  # noqa: S105 - a verdict token, not a secret
TERM_FAIL: str = "fail"
TERM_INSUFFICIENT_EVIDENCE: str = "insufficient_evidence"

# The whole-gate decision, same three states aggregated over the ARMED terms only.
GATE_PASS: str = "pass"  # noqa: S105 - a verdict token, not a secret
GATE_FAIL: str = "fail"
GATE_INSUFFICIENT_EVIDENCE: str = "insufficient_evidence"

TERM_CONTINUITY: str = "continuity"
TERM_COHESION: str = "cohesion"
TERM_MOMENTUM: str = "momentum"
TERM_LOUDNESS: str = "loudness"
TERM_DURATION: str = "duration"

#: Every ship-gate term, in record order.
SHIP_GATE_TERMS: tuple[str, ...] = (
    TERM_CONTINUITY,
    TERM_COHESION,
    TERM_MOMENTUM,
    TERM_LOUDNESS,
    TERM_DURATION,
)

# AXIS 1 — what KIND of threshold each term is. Published on the record so a reader
# can tell a structurally undeliverable film from one that missed a tunable ceiling,
# WITHOUT re-deriving intent by string-matching the detail text. This axis says
# nothing about whether the term blocks; see `_TERM_KILL_SWITCHES` for that.
_TIER_STRUCTURAL: str = "structural"
_TIER_TUNABLE: str = "tunable"
_TERM_TIERS: dict[str, str] = {
    # The picture either is or is not the length that was ordered — no degree.
    TERM_DURATION: _TIER_STRUCTURAL,
    TERM_CONTINUITY: _TIER_TUNABLE,
    TERM_COHESION: _TIER_TUNABLE,
    TERM_MOMENTUM: _TIER_TUNABLE,
    TERM_LOUDNESS: _TIER_TUNABLE,
}
# Drift must UNDER-report severity, never over-report it.
_DEFAULT_TERM_TIER: str = _TIER_TUNABLE

# AXIS 2 — whether a term BLOCKS right now. One `PipelineSettings` field (and
# therefore one env var) per term, so a miscalibrated term is disarmed from the
# GitOps configmap rather than by a code change and a rollout. Mirrors
# `video_compiler_service/src/services/mastering.py::_GATE_KILL_SWITCHES`.
# A term absent from this map can only ever be armed by the whole-gate escalation.
_TERM_KILL_SWITCHES: dict[str, str] = {
    TERM_CONTINUITY: "ship_gate_continuity_armed",
    TERM_COHESION: "ship_gate_cohesion_armed",
    TERM_MOMENTUM: "ship_gate_momentum_armed",
    TERM_LOUDNESS: "ship_gate_loudness_armed",
    TERM_DURATION: "ship_gate_duration_armed",
}

# The default each switch carries, duplicated here so `gate_armed` still answers
# correctly for a settings object that PREDATES the fields (a test double, a
# `SimpleNamespace`) instead of silently reading every term as disarmed — the
# fail-OPEN direction that is exactly how this gate came to decide nothing. Must
# agree with `src/config.py`; `tests/unit/test_ship_gate_arming.py` pins the
# agreement field-by-field.
_TERM_ARMED_DEFAULTS: dict[str, bool] = {
    # NOT ARMED — structurally unobservable on the current analyzer wire shape
    # (no per-frame brightness/sharpness/colour; five of six feature components
    # constant by construction). The 0.5 ceiling was never calibrated.
    TERM_CONTINUITY: False,
    # NOT ARMED — same unobservability, and settled by a recorded measurement:
    # cohesion measures 0.0082 against a 0.5 floor, i.e. ~100% would fail.
    TERM_COHESION: False,
    # ARMED, and safe by construction: with no `ceilings["momentum_gap_max"]`
    # configured (the live state) `momentum_verdict` is `unconfigured`, which is
    # insufficient evidence and never a fail. This is the seam, pre-wired.
    TERM_MOMENTUM: True,
    # NOT ARMED — a MEASURED refusal, exactly like TERM_COHESION above. Armed
    # 2026-09-15 on an explicitly unmeasured fail rate (n=0); measured
    # 2026-09-16 on RS w2v-orchestrator-7b5bbcf5cf at 3 of 3 demo_video runs
    # failing final_critique (-14.2 / -14.1 / -14.1 LUFS vs the (-14, -11)
    # window). The term itself is sound and the window is NOT the defect: the
    # ratified target is -12.5 LUFS, so delivery is 1.6-1.7 LU BELOW target and
    # the gate was convicting a real, systematically quiet speech bus that
    # nothing downstream of this gate can remediate. Disarmed because
    # `final_critique` is the `delivered_artifact_stage` for demo_video AND
    # showcase, and because relief only spawns on a COMPLETED run, so enforcing
    # here also silently disables demo_video auto-convergence. RE-ARM once the
    # speech-bus level is fixed.
    TERM_LOUDNESS: False,
    # NOT ARMED — a MEASURED refusal. `duration_conformance` records 129%-273%
    # of order across four live types, so arming it today fails the majority of
    # deliveries over an overshoot this gate cannot remediate.
    TERM_DURATION: False,
}


class ShipGateEnforcementError(RuntimeError):
    """A11 ENFORCE-mode terminal failure (opt-in hard gate).

    Raised ONLY when ``ship_gate.enforce`` is true, ``mode == auto``, and the best draft
    the bounded re-drive could produce STILL fails the ship-gate ceilings (``ship_ready``
    is False on the same gestalt the badge reads). Subclasses ``RuntimeError`` so it rides
    the exact terminal-failure mechanism ``demo_video_record`` uses to reject a blank clip
    (#719 / e4eed44): the DAG executor persists the message into ``StageOutput.error`` and
    lands the run at ``status=failed``. There is NO refund path — ``grep -rn refund`` over
    this repo finds docstrings only, and ``run_usage_reconciler`` merely mirrors usage
    read-only, so the tenant is still billed for the stages that already ran. A DISTINCT type
    (not a bare ``RuntimeError``) so ``handle``'s soft-degrade wrapper re-raises it terminally
    instead of swallowing it as an internal defect and accepting the artifact.

    With ``enforce`` false (the default) this is never raised and behaviour is byte-for-byte
    identical to the accept-best gate."""


# Bounds for the model's free-text fields and the 1-5 dimension scores.
_MAX_TEXT = 200
_SCORE_MIN = 1
_SCORE_MAX = 5
_OVERALL_MIN = 1.0
_OVERALL_MAX = 5.0
# A failed re-evaluation (analyzer/WM outage on a recompiled artifact) must never
# outrank an evaluated draft, so it scores below the [1,5] clamp floor — keep-highest
# then retains the current best rather than shipping an unevaluated recompile.
_FAILED_EVAL_SCORE = _OVERALL_MIN - 1.0

# Feature-vector normalization (gestalt). The per-frame components live on wildly
# different native scales — brightness/motion are already 0..1, dominant_rgb is 0..255,
# and sharpness (Laplacian variance) is unbounded and typically hundreds. A raw L2 over
# them is dominated by whichever axis is widest (measured cohesion 0.0082, sharpness/
# colour-driven), so continuity/cohesion carry no real signal. Each component is mapped
# into ~[0,1] before the metric so every axis contributes comparably:
#   brightness → clamp 0..1 (already luma-normalized) ; dominant_rgb → /255 per channel ;
#   motion     → clamp 0..1 ; sharpness → saturating s/(s+K), monotonic and outlier-robust
# (no fixed max to guess, no hard clamp). ``_SHARPNESS_SCALE`` is the Laplacian variance
# at which sharpness reads 0.5 — a starting midpoint, re-tunable by the nightly gestalt
# calibration (the ship-gate ceilings are calibrated live, not baked here).
_SHARPNESS_SCALE = 500.0

# Chapter cap — keep the audit chapter list minimal.
_MAX_CHAPTERS = 24

# The advisory finding text emitted when the sampled shots carry no visual signal — too
# few of them, or several that are identical in feature space.
_INSUFFICIENT_EVIDENCE_DETAIL = (
    "insufficient evidence: the sampled shots carry no discriminating visual signal, so "
    "continuity, cohesion and tonal drift could not be measured — the ship gate made no "
    "claim (the analyzer emits no per-frame brightness/sharpness/colour)"
)

# Above this measured duplicate-frame ratio the rendered film is essentially static, which
# is the input shape that makes the visual gestalt terms unmeasurable. Advisory only.
_STATIC_FRAME_RATIO_ADVISORY = 0.9

# Bounded re-drive (mode="auto"). At most 2 re-drive passes, keeping the
# highest-scoring draft (monotonic non-regression), converging at overall >= 4.0.
# Both are DEFAULTS — the ship_gate config block may override `max_passes` /
# `score_threshold` (clamped to [1, 2]) via `_redrive_bounds`.
_REDRIVE_MAX_PASSES = 2
_REDRIVE_THRESHOLD = 4.0

# The per-dimension LLM stages a re-drive re-RUNS from the film-critic feedback
# (PR-3). Each owns one gestalt dimension and feeds compile IN-STAGE (footage plans
# / sound cues directly; voice via a narrate re-synth), so re-running it needs NO
# sub-DAG enqueue and NO child run — the whole point of the soft in-stage gate.
# `demo_video_compile` is deliberately NOT here: a bare recompile with no upstream
# change is a no-op, so a compile directive stays advisory. footage_critique /
# final_critique are excluded (cycle: they drive/contain THIS stage); growth_bundle
# / render_manifest are excluded (downstream of the artifact). Keep in sync with
# `_REDRIVE_ADAPTER` and `_REDRIVE_ORDER`.
_LLM_REDRIVABLE: frozenset[str] = frozenset(
    {"voice_director", "demo_video_sound_director", "demo_video_footage_direction"}
)

# Deterministic re-drive order per pass. Footage + sound feed compile directly;
# voice must precede the narrate re-synth (compile reads the fresh clips). Fixed
# order so a multi-stage re-drive is reproducible.
_REDRIVE_ORDER: tuple[str, ...] = (
    "demo_video_footage_direction",
    "demo_video_sound_director",
    "voice_director",
)

# Per-stage merge adapter: (output-list-key, id-field). The regenerated stage
# output is a list under `output-list-key`; each item is joined back onto the
# baseline output by `id-field`, replacing ONLY the flagged ids (unflagged rows
# ride through verbatim so the re-drive touches only the weak beats).
_REDRIVE_ADAPTER: dict[str, tuple[str, str]] = {
    "voice_director": ("directives", "line_id"),
    "demo_video_sound_director": ("cues", "line_id"),
    "demo_video_footage_direction": ("plans", "line_id"),
}

# The real blueprint stage-name set a directive's `stage` is clamped to. Hardcoded
# (with this comment) as the UNION of the demo_video + showcase blueprint stage
# names — deriving it from the registry would create an import cycle (the
# blueprints import THIS module). A directive naming any other stage is a
# hallucination and is dropped. Keep in sync if a demo_video/showcase stage is
# added or renamed.
_VALID_STAGES: frozenset[str] = frozenset(
    {
        "demo_video_gen",
        "source_ingest",
        "script_doctor",
        "claim_finish",
        "retention_plan",
        "reconcile_timeline",
        "voice_director",
        "demo_video_narrate",
        "demo_video_record",
        "showcase_slides",
        "demo_video_sound_director",
        "demo_video_footage_direction",
        "demo_video_compile",
        "demo_video_footage_critique",
        "demo_video_encode_master",
        "final_critique",
        "growth_bundle",
        "render_manifest",
    }
)

# video_analyzer read config: scene boundaries (for the cut spine + chapters),
# whole-video quality metrics + motion timeline, and per-frame descriptions/OCR
# (for the tonal proxy). No walkthrough/inventory. Bounds the frame sample.
_ANALYSIS_MAX_FRAMES = 48
#: video_analyzer's OWN wall-clock ceiling on a single analysis job:
#: `PipelineSettings.job_deadline_seconds` in `video_analyzer_service/src/config.py`,
#: default 1800, `ge=0` where 0 disables. Verified 2026-09-01 to have NO override
#: anywhere in GitOps or in any live ConfigMap, and verified against the server's own
#: behaviour: job 01a057dd-278a-713f-beb3-e755a967f505 was killed by the analyzer at
#: exactly 1800s with "Analysis exceeded the 1800s wall-clock deadline and was
#: cancelled".
#:
#: This is the value the poll budget below WANTS to be keyed to and currently cannot
#: reach — see the shortfall paragraph there. It is recorded as a constant anyway so
#: the gap is a named, testable quantity rather than a thing someone has to rediscover
#: by reading two repos.
#:
#: ⚠️ Nothing in this repo can observe the real value: it is another service's config
#: default, so this is a hand-mirrored copy. If `job_deadline_seconds` is ever changed
#: in `video_analyzer_service` — or overridden in GitOps — update this line by hand.
#: No test here can notice that drift.
_ANALYZER_JOB_DEADLINE_S = 1800.0
#: This stage's own wall-clock ceiling — `default_timeout_seconds` on the
#: `final_critique` StageSpec in `src/services/blueprints/demo_video.py`. Mirrored
#: here (not imported) because the blueprint imports this module; the duplication is
#: pinned by `test_final_critique.py::
#: test_analyzer_poll_timeout_is_sized_from_measurement_and_fits_the_stage`, which
#: reads the real spec.
_STAGE_TIMEOUT_S = 2500.0
#: Worst-case wall-clock of the LLM ladder this stage runs AFTER the analyzer poll
#: returns (the critic call plus its retry). Sequential with the poll, so the two
#: share `_STAGE_TIMEOUT_S`.
_LLM_LADDER_WORST_CASE_S = 1320.0
# How long to poll video_analyzer before giving up and soft-degrading to an empty
# gestalt.
#
# ⛔ 600.0 IS THE TRANSPORT CEILING, AND EXCEEDING IT RAISES RATHER THAN CAPS.
# Every poll on this path goes `VideoAnalyzerClient.poll_analysis` ->
# `BaseServiceClient.poll_job`, whose FIRST statement is
# `timeout = _clamp_timeout(timeout)` (`w2v_common/clients/base.py`).
# `_clamp_timeout` takes its ceiling from `_env_float("CLIENT_MAX_TIMEOUT_S",
# _MAX_TIMEOUT)` with `_MAX_TIMEOUT = 600.0`, and for `value > ceiling` it RAISES
# `ValueError` — it does NOT silently cap. `CLIENT_MAX_TIMEOUT_S` is set nowhere in
# this repo and nowhere in gitops, so the effective ceiling is 600.0.
#
# ⚠️ A VALUE ABOVE 600 IS THEREFORE NOT MERELY INEFFECTIVE, IT IS FATAL: the poll
# blows up before the first GET, on EVERY analyzer poll. This constant was briefly
# `_STAGE_TIMEOUT_S - _LLM_LADDER_WORST_CASE_S - 80` = 1100.0, derived from the
# StageSpec budget. That derivation named the WRONG binding constraint — the stage
# budget is the LOOSER bound and the client clamp binds far below it — and it shipped
# green because every unit test here mocks the analyzer client, so the real
# `_clamp_timeout` never ran.
# ⇒ Do NOT re-derive this from `_STAGE_TIMEOUT_S`. The stage budget's only job is to
#   CONTAIN this poll plus the LLM ladder; it can never justify RAISING it.
#   `tests/services/stages/test_poll_timeouts_within_client_ceiling.py` pins that
#   against the library's own constant, for this stage and any future one.
#
# ⛔ AND DO NOT RE-DERIVE IT FROM A PERCENTILE OF OBSERVED RUNTIMES either — that is
# what both values before 1100 did, and both went stale within weeks:
#
#   180.0 — "2% above the observed max (176.2s)", measured 2026-08-04 over 210 jobs.
#   420.0 — "~2.4x the slowest observed success", set when 180 started truncating.
#
# The analyzer's runtime distribution moves with cluster contention and with pgx model
# swaps; it is not a property of this code. Measured 2026-09-01 over 14 days, 536
# analyzer jobs: p90 227s, max 1384s, and 27 of the 536 ran longer than 420s. 600
# recovers the 420-600s band of that tail, which is as far as this transport reaches.
#
# ⚠️ THAT LEAVES THE TRUNCATION BUG BOUNDED, NOT CLOSED — and the remaining gap is now
# a LIBRARY limit, not a budget choice. The ideal key is the analyzer's own
# `_ANALYZER_JOB_DEADLINE_S` (1800), because every job provably resolves by then, but
# 1800 is UNREACHABLE THROUGH THIS TRANSPORT: it is above the clamp, so it raises.
# Jobs between 600s and 1800s are still truncated. Closing that needs a w2v_common
# change (raise `_MAX_TIMEOUT`, or set `CLIENT_MAX_TIMEOUT_S` for this deployment)
# AND a matching stage re-budget — an owner-level decision, not a side effect here.
#
# It cannot make a run hang: this is the analyzer branch of a SOFT gate — exceeding
# it still degrades to accept-the-artifact (see `_reevaluate`'s analyzer `except`),
# it just stops discarding samples that were about to arrive.
#
# ⚠️ Do NOT justify any of this from `job_deadline_seconds`' own docstring, which
# reads "1800s is deliberately ~4x the orchestrator's 420s poll timeout". That
# sentence points at a value this line no longer holds, and the two would chase each
# other. The analyzer's deadline is an independent variable; this is a dependent one.
_ANALYSIS_POLL_TIMEOUT_S = 600.0
_ANALYSIS_CONFIG: dict[str, object] = {
    "generate_walkthrough": False,
    "generate_ui_inventory": False,
    "generate_frame_index": False,
    "detect_scenes": True,
    "compute_quality_metrics": True,
    "extract_ocr": True,
    "extract_frame_descriptions": True,
    "max_frames": _ANALYSIS_MAX_FRAMES,
}

_SYSTEM_PROMPT = (
    "You are a film critic scoring a finished short marketing/demo video as a whole. "
    "You receive MEASURED whole-film gestalt signals (visual cohesion, shot-to-shot "
    "continuity breaks, tonal drift across the narration, and the mastered loudness) "
    "plus the ordered list of shot ids. Score the film on four dimensions 1-5 "
    "(cohesion, continuity, momentum_experienced, tonal_consistency) and give an "
    "overall 1-5. Consider the measured momentum_gap in the gestalt as one advisory "
    "signal when scoring momentum_experienced (the worst under-delivery of the assembled "
    "momentum curve vs the predicted watch curve — a larger gap SUGGESTS the film lost "
    "momentum the pacing promised, but it is an approximate signal, not an authoritative "
    "threshold; weigh it with the rest). Prefer restraint — most films are fine. When a seam is genuinely "
    "weak, name it in findings (with the from_id/to_id of the cut and a one-sentence "
    "`detail` saying concretely what is wrong — the detail is quoted verbatim to the "
    "upstream director that has to fix it, so an empty or generic one is useless). You MAY suggest "
    "directives (which upstream stage to revisit, for which shot ids) but they are "
    "ADVISORY — you never block the ship. Return only JSON matching the schema."
)

# Object-rooted structured-output schema (WM convention). Scores are 1-5.
_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "cohesion": {"type": "integer"},
        "continuity": {"type": "integer"},
        "momentum_experienced": {"type": "integer"},
        "tonal_consistency": {"type": "integer"},
        "overall": {"type": "number"},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "seam": {
                        "type": "object",
                        "properties": {
                            "from_id": {"type": "string"},
                            "to_id": {"type": "string"},
                        },
                    },
                    "dimension": {"type": "string"},
                    "severity": {"type": "string"},
                    # The human-readable WHY. Was absent from the schema, so the model
                    # never emitted one and `_redrive_feedback` formatted every bullet
                    # fed to a re-driven director as a literally empty
                    # "- [continuity/minor] ". Required so a finding always explains
                    # itself.
                    "detail": {"type": "string"},
                },
                "required": ["dimension", "severity", "detail"],
            },
        },
        "directives": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "stage": {"type": "string"},
                    "spine_ids": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string"},
                },
                "required": ["stage", "spine_ids"],
            },
        },
    },
    "required": ["overall"],
}


# MEDIUM answer: ``_OUTPUT_SCHEMA`` is one verdict object — four 1-5 integers and an
# ``overall`` — plus a ``findings`` list of SEAMS the critic objected to (each with a
# prose ``detail``) and a short advisory ``directives`` list. The lists track the
# problems found, not the shot count, but each entry carries real prose, so this is
# above the single-verdict tier.
#
# ⚠️ There used to be an ``"accept": {"type": "boolean"}`` property here. Removed:
# nothing downstream ever read ``data.get("accept")`` — the ship/no-ship decision is
# derived from ``overall`` + the advisory findings (see ``_accept`` /
# ``ship_ready``), and the system prompt itself tells the model "you never block the
# ship". Asking the model for a boolean opinion that is then silently thrown away is
# worse than not asking: it reads, from the schema alone, as though the critic's
# accept/reject verdict matters, when the actual gate is the deterministic
# overall/findings computation. Not wired into that gate either — this task is
# explicitly scoped to never arm a new gate — so deletion is the honest fix, not
# a second read path for a number nothing else agrees to trust.
_CRITIC_MAX_TOKENS = output_token_budget(answer_tokens=MEDIUM_ANSWER_TOKENS)


# ── output models ────────────────────────────────────────────────────────────
class Seam(StageOutput):
    from_id: str
    to_id: str


class FinalCritiqueScores(StageOutput):
    """The film-critic's verdict. Each dimension is ``None`` when the reply did not
    carry a usable value for it — NEVER a fabricated neutral. The four dimensions used
    to default to ``_NEUTRAL`` (3), which is indistinguishable from a genuine "average"
    verdict; the live corpus shows a critic that returned no dimensions at all persisting
    ``3/3/3/3`` next to ``overall: 1.0``, and those fabricated numbers were then read as
    evidence. An absent dimension is now visibly absent."""

    cohesion: int | None = None  # 1-5
    continuity: int | None = None  # 1-5
    momentum_experienced: int | None = None  # 1-5
    tonal_consistency: int | None = None  # 1-5
    overall: float  # 1-5


class FinalCritiqueFinding(StageOutput):
    dimension: str
    severity: str
    seam: Seam | None = None
    detail: str = ""


class FinalCritiqueDirective(StageOutput):
    stage: str
    spine_ids: list[str]
    reason: str = ""


class FinalCritiqueChapter(StageOutput):
    start: float
    title: str


class FinalCritiqueInput(StageInput):
    """No typed inputs — everything is read from ``ctx`` (prior output + config)."""


class FinalCritiqueOutput(StageOutputWithDegradations):
    """Advisory ship-gate verdict + pass-through delivered artifact.

    ``artifact_url`` / ``duration_seconds`` carry the accepted-best artifact:
    footage_critique's output (else compile's) in ``critique_only`` / ``off``, or the
    highest-scoring re-driven artifact under ``mode == "auto"``. ``render_manifest`` /
    the growth bundle read whatever this stage accepted. ``rerun_stages`` is ``[]``
    unless an ``auto`` re-drive actually EXECUTED an LLM-redrivable owning stage for the
    RETAINED draft — it is a report of work done, recorded by ``_redrive`` as it ran, not
    a re-derivation from the critic's directives.
    """

    # Declared output contract — enforced centrally in
    # `runs/dag_executor.py::_enforce_produces`. `produces_required` because
    # `render_manifest` reads this stage FIRST when resolving the delivered
    # artifact: the accept-current contract guarantees a URL on every path
    # (critique_only / off / auto), so an empty one means the accept broke and the
    # run would otherwise ship a ledger pointing at nothing.
    # Nothing ELSE here is declarable and that is deliberate: `rerun_stages` is []
    # unless an auto re-drive executed, `gestalt_summary` is measurably {} on a LARGE
    # minority of live rows (see gestalt_unavailable_reason below), and findings/
    # directives/chapters/quality_badge all have legitimate empty defaults.
    #
    # ⚠️ The "~11%" this comment used to quote was measured 2026-08-04 and is stale by
    # a factor of five. Re-measured 2026-09-01 on the live DB over 14 days: of 42
    # completed final_critique rows, 23 (55%) carry gestalt_summary == {}. Adding
    # `gestalt_summary` to `produces` would therefore hard-fail the MAJORITY of runs
    # over a signal this stage is explicitly allowed to do without — do not do it. The
    # breakdown also shows why raising the poll timeout is not a complete fix: only 7
    # of those 23 (17% of all rows) were the 420s truncation, 1 was an analyzer-side
    # failure, and 15 recorded NO reason at all — a separate, still-undiagnosed path
    # that leaves the field empty without an advisory. The calibration validator is
    # the layer that catches empty gestalt, and it worked correctly throughout.
    produces: ClassVar[tuple[str, ...]] = ("artifact_url",)
    produces_required: ClassVar[bool] = True

    artifact_url: ArtifactUrl
    duration_seconds: float
    scores: FinalCritiqueScores | None = None
    findings: list[FinalCritiqueFinding] = Field(default_factory=list)
    directives: list[FinalCritiqueDirective] = Field(default_factory=list)
    gestalt_summary: dict[str, object] = Field(default_factory=dict)
    gestalt_unavailable_reason: str | None = None
    """Why ``gestalt_summary`` is empty, when it is — otherwise ``None``.

    An empty ``gestalt_summary`` is a DEAD SIGNAL: every consumer reads that field,
    but the reason a soft-degrade produced it is recorded in ``findings`` under a
    different dimension, so nothing that reads the gestalt can explain it. Measured
    on the live corpus (7 days to 2026-08-04): of 18 completed ``final_critique``
    rows, 2 carried ``gestalt_summary == {}`` and exactly those 2 carried an analyzer
    TIMEOUT finding — an 11% silent loss of calibration samples. Downstream this
    surfaces as ``gestalt_summary.continuity_stats.n resolved no numeric value
    (got [])``, which names neither the analyzer nor the timeout.

    Derived in ``_accept`` from the advisory findings, so every soft-degrade branch
    is covered by construction rather than by remembering to thread a reason.
    """
    ship_gate: dict[str, object] = Field(default_factory=dict)
    """The ARMED ship-gate record — ``{decision, terms[], armed_terms, failed_terms,
    unmeasured_terms}`` (see :func:`ship_gate_record`).

    ``decision`` is three-state: ``pass`` / ``fail`` / ``insufficient_evidence``. The
    third is the one this field exists for — it is neither a pass nor a fail, and it is
    the honest answer whenever no ARMED term could observe its subject (an absent
    analyzer result, an unmeasurable loudness, a momentum term with no configured
    ceiling). ``gestalt_summary`` alone cannot express that: its ``ship_ready`` is a
    single boolean that reads False for "measured and bad" and for "measured nothing"
    alike, which is how a starved judge comes to look like a clean verdict.

    ⛔ NOT in ``produces``. Deliberately: a soft-degrade path that never reaches the
    gate records ``insufficient_evidence`` with an empty ``terms`` list, and that row is
    correct rather than a defect — declaring it required would hard-fail exactly the
    runs whose evidence was missing.
    """
    quality_badge: str = "needs-review"
    chapters: list[FinalCritiqueChapter] = Field(default_factory=list)
    rerun_stages: list[str] = Field(default_factory=list)


def require_ship_gate(ctx: RunContext) -> None:
    """Skip-clean unless a truthy ``ship_gate`` block is present with mode != off.

    Raises ``StageSkippedError`` (→ the stage is ``skipped``, terminal-but-fine, its
    output ``{}``) when the run did not opt into the ship gate, so downstream reads
    compile's artifact unchanged (byte-identical to a run without this stage).
    """
    ship_gate = ctx.config_snapshot.get("ship_gate")
    if not isinstance(ship_gate, Mapping) or not ship_gate:
        raise StageSkippedError(
            "final_critique: ship_gate not requested (opt-in stage)",
            missing=["ship_gate"],
        )
    if str(ship_gate.get("mode") or "").lower() == _OFF:
        raise StageSkippedError(
            "final_critique: ship_gate mode is off",
            missing=["ship_gate.mode"],
        )


# ── coercion helpers (pure, never raise) ─────────────────────────────────────
# Scalar coercion here is the STRICT family (``src.utils.as_float`` / ``opt_float``):
# these are analyzer/compiler measurements that are already supposed to BE numbers, so a
# string or a bool arriving is a wire-shape bug, not a value to rescue. Do NOT swap in the
# PERMISSIVE ``coerce_float`` — it would parse ``"1.5"``, map ``True`` ⇒ ``1.0`` and pass
# NaN straight through into the gate arithmetic and the Prometheus histograms.
# ``opt_float`` is the optional twin used for gestalt terms: a missing/bool/non-numeric/NaN
# term becomes ``None`` and is SKIPPED (never observed) rather than observed as a spurious
# ``0.0``. Rejecting NaN matters most here — a NaN observed into a Prometheus Histogram
# permanently poisons its ``_sum``, corrupting the averages this telemetry exists to compute.


def _emit_gestalt_metrics(gestalt: Mapping[str, object]) -> None:
    """Push the ship-gate's numeric terms to Prometheus + one info log line.

    INERT w.r.t. the gate: this only OBSERVES / increments and reads ``gestalt`` — it
    never mutates it or influences ``ship_ready``. Existence purely so the ship-gate
    ceilings can be calibrated from real renders. Any failure is swallowed (counted on
    ``gestalt_metrics_emit_errors_total`` + logged) so an observability defect can never
    break a render.
    """
    try:
        momentum_gap = opt_float(gestalt.get("momentum_gap"))
        if momentum_gap is not None:
            metrics.gestalt_momentum_gap.observe(momentum_gap)

        cohesion = opt_float(gestalt.get("cohesion"))
        if cohesion is not None:
            metrics.gestalt_cohesion.observe(cohesion)

        continuity_stats = gestalt.get("continuity_stats")
        continuity_max = (
            opt_float(cast("Mapping[str, object]", continuity_stats).get("max"))
            if isinstance(continuity_stats, Mapping)
            else None
        )
        if continuity_max is not None:
            metrics.gestalt_continuity_max_delta.observe(continuity_max)

        loudness = gestalt.get("loudness")
        lufs = (
            opt_float(cast("Mapping[str, object]", loudness).get("integrated_lufs"))
            if isinstance(loudness, Mapping)
            else None
        )
        if lufs is not None:
            metrics.gestalt_loudness_lufs.observe(lufs)

        ship_ready = bool(gestalt.get("ship_ready"))
        momentum_ok = bool(gestalt.get("momentum_ok"))
        metrics.gestalt_ship_gate_total.labels(
            ship_ready=str(ship_ready).lower(),
            momentum_ok=str(momentum_ok).lower(),
        ).inc()

        log.info(
            "final_critique: gestalt ship-gate metrics momentum_gap=%s cohesion=%s "
            "continuity_max=%s lufs=%s ship_ready=%s momentum_ok=%s "
            "continuity_ceiling_suggested=%s",
            momentum_gap,
            cohesion,
            continuity_max,
            lufs,
            ship_ready,
            momentum_ok,
            gestalt.get("continuity_ceiling_suggested"),
        )
    except Exception:  # telemetry is advisory, never breaks the gate
        metrics.gestalt_metrics_emit_errors_total.inc()
        log.warning("final_critique: gestalt metrics emit failed", exc_info=True)
        return


def _clamp_dim(value: object) -> int | None:
    """Coerce a model dimension score into the ``[1, 5]`` int band; junk ⇒ ``None``.

    Returns ``None`` (NOT the neutral 3) for a missing / non-numeric / non-finite value.
    A fabricated 3 is a claim the model never made and cannot be told apart from a
    genuine average verdict downstream — see :class:`FinalCritiqueScores`."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        n = round(float(value))
    except TypeError, ValueError, OverflowError:
        return None  # inf ⇒ OverflowError, nan ⇒ ValueError
    return max(_SCORE_MIN, min(_SCORE_MAX, n))


def _clamp_overall(value: object, fallback: float | None) -> float | None:
    """The model's overall clamped into ``[1, 5]``; junk / non-finite ⇒ ``fallback``
    (which is itself ``None`` when there is no dimension to average)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return fallback
    f = float(value)
    if not math.isfinite(f):  # NaN / ±inf
        return fallback
    return max(_OVERALL_MIN, min(_OVERALL_MAX, f))


def _seq(value: object) -> list[object]:
    """A plain list from an untyped JSON value (never a str/bytes/mapping)."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


_SYNTHETIC_LINE_PREFIX = "__"  # title-card / disclaimer synthetic shots (not authored beats)


def _coerce_span(raw_start: object, raw_end: object) -> tuple[float, float] | None:
    """A `(start, end)` float span from a shot_timeline entry, or None if the
    values are missing/non-numeric/NaN or the span is non-positive (`end <= start`).
    Coerces without a ty-ignore via the STRICT `opt_float` (rejects bool and NaN)."""
    start = opt_float(raw_start)
    end = opt_float(raw_end)
    if start is None or end is None or end <= start:  # missing/junk or non-positive span
        return None
    return start, end


def _to_intervals(
    shot_timeline: list[dict[str, object]] | None, spine_ids: Sequence[str]
) -> list[tuple[str, float, float]] | None:
    """Reshape the compiler's authoritative shot_timeline into (id, start, end)
    beat intervals for sample_frames_at_cuts. Returns None (⇒ caller falls back to
    intended-duration build_beat_intervals) unless every AUTHORED entry is
    well-formed. Synthetic spliced shots (``__``-prefixed line_ids: title cards,
    disclaimer) are SKIPPED — they are not authored beats and their ids are not on
    the spine. The first authored interval's start is clamped to 0.0 so cuts in a
    title/intro lead attribute to beat 0 rather than clamping to the LAST beat."""
    if not shot_timeline:
        return None
    spine = set(spine_ids)
    rows: list[tuple[str, float, float]] = []
    for entry in shot_timeline:
        if not isinstance(entry, Mapping):
            return None
        lid = entry.get("line_id")
        if not isinstance(lid, str):
            return None
        if lid.startswith(_SYNTHETIC_LINE_PREFIX):
            continue  # synthetic spliced shot — not an authored beat
        if lid not in spine:  # a non-sentinel id off the spine ⇒ anomaly, fall back
            return None
        span = _coerce_span(entry.get("start"), entry.get("end"))
        if span is None:
            return None
        rows.append((lid, span[0], span[1]))
    if not rows:
        return None
    rows.sort(key=lambda r: r[1])
    rows = _clamp_overlaps(rows)
    first_id, _first_start, first_end = rows[0]
    rows[0] = (first_id, 0.0, first_end)
    return rows


def _clamp_overlaps(
    rows: list[tuple[str, float, float]],
) -> list[tuple[str, float, float]]:
    """De-overlap start-sorted spans so they partition the timeline.

    The compiler's bus offsets OVERLAP by the xfade duration when consecutive
    shots crossfade (``_bus_offsets_xfade`` example [(0,4),(3,9),(8,15)]).
    sample_frames_at_cuts is first-match-wins on ``start <= t < end``, so a raw
    overlap would attribute a cut inside the crossfade to the EARLIER beat,
    violating the half-open "boundary cut joins the later beat" rule. Clamp each
    end to the next start (the last end is untouched); a hard cut already has
    ``end == next.start`` so it is a no-op there."""
    for i in range(len(rows) - 1):
        rid, start, end = rows[i]
        next_start = rows[i + 1][1]
        if end > next_start:
            rows[i] = (rid, start, next_start)
    return rows


# ⛔ `_failed_ceilings` / `_loudness_reasons` were DELETED here, not orphaned. They
# explained a NOT-ship-ready gestalt in prose for the enforce hard-fail reason, and the
# per-term record below supersedes them strictly: `ship_gate_term_verdicts` reads the
# same producer verdicts and additionally carries each term's measured VALUE, the LIMIT
# it was compared against, its TIER and whether it was ARMED. Keeping both would leave a
# second, unreachable diagnosis path that a reader would mistake for the live one — and
# the prose version could not express `insufficient_evidence` as distinct from a fail
# (it reported an unmeasured cohesion and an unmeasured loudness as reasons the gate
# objected, which is the exact conflation that makes arming unsafe).


def gate_tier(term: str) -> str:
    """AXIS 1 — the KIND of threshold ``term`` is: ``structural`` or ``tunable``.

    Published on the ship-gate record so a reader can separate a structurally
    undeliverable film from one that missed a tunable ceiling without string-matching
    prose. Says NOTHING about whether the term blocks — see :func:`gate_armed`. An
    unknown term reads ``tunable``: drift must UNDER-report severity."""
    return _TERM_TIERS.get(term, _DEFAULT_TERM_TIER)


def gate_armed(settings: object, term: str, *, enforce_all: bool = False) -> bool:
    """AXIS 2 — whether an AFFIRMATIVE failure of ``term`` must FAIL the run.

    Three inputs, in precedence order — the shape of
    ``video_compiler_service``'s ``mastering.gate_armed``:

    1. ``enforce_all`` — the per-run ``ship_gate.enforce: true`` escalation. Arms
       EVERY term, including the ones deliberately left disarmed. It is an
       escalation, not the only way in; that it USED to be the only way in (and was
       never set) is precisely why this gate could not fail.
    2. the term's own kill switch (:data:`_TERM_KILL_SWITCHES`), read off
       ``PipelineSettings`` / env. This is the back-out path: a miscalibrated term is
       disarmed from the configmap, never by a code change and a rollout.
    3. :data:`_TERM_ARMED_DEFAULTS` — used when the settings object does not carry
       the field at all (a test double, an older Settings), so it reads the REAL
       default instead of fail-open ``False``.

    A term with no kill switch can only ever be armed by (1)."""
    if enforce_all:
        return True
    switch = _TERM_KILL_SWITCHES.get(term)
    if switch is None:
        return False
    return bool(getattr(settings, switch, _TERM_ARMED_DEFAULTS.get(term, False)))


@dataclass(frozen=True)
class ShipGateTermVerdict:
    """One ship-gate term's three-state verdict, with the numbers behind it.

    ``state`` is :data:`TERM_PASS` / :data:`TERM_FAIL` /
    :data:`TERM_INSUFFICIENT_EVIDENCE`. The third is NOT a soft fail and NOT a soft
    pass: it records that the term's subject could not be observed, which is the
    distinction a two-state gate structurally cannot make — and the one that matters
    here, because ``gestalt_summary`` is measurably ``{}`` on ~55% of live rows and an
    absent measurement must never be convicted as a bad film (nor absolved as a good
    one).

    ``value`` / ``limit`` are the measured number and the ceiling it was compared
    against, ``None`` where the check is non-numeric or nothing was measured. They are
    what makes an enforcement failure DIAGNOSABLE: the record says which term failed and
    with what value, so a false failure can be calibrated away instead of the whole gate
    being switched off."""

    term: str
    state: str
    tier: str
    armed: bool
    value: float | None = None
    limit: str | None = None
    detail: str = ""

    def as_row(self) -> dict[str, object]:
        """The persisted row. ``dict[str, object]`` — no consumer indexes it."""
        return {
            "term": self.term,
            "state": self.state,
            "tier": self.tier,
            "armed": self.armed,
            "value": self.value,
            "limit": self.limit,
            "detail": self.detail,
        }


def _continuity_verdict(gestalt: Mapping[str, object], ceilings: Mapping[str, object]) -> str:
    if not gestalt or bool(gestalt.get("insufficient_evidence")):
        return TERM_INSUFFICIENT_EVIDENCE
    breaks = gestalt.get("continuity_breaks")
    if not isinstance(breaks, Sequence) or isinstance(breaks, (str, bytes, bytearray)):
        return TERM_INSUFFICIENT_EVIDENCE
    return TERM_FAIL if breaks else TERM_PASS


def _cohesion_verdict(gestalt: Mapping[str, object]) -> str:
    # ⚠️ An UNMEASURED cohesion is insufficient evidence, NOT a failure. The old
    # explainer folded `cohesion is None` in with `cohesion < floor` under one
    # "cohesion (unmeasured < floor …)" reason, so an absent measurement was
    # indistinguishable from a bad one at the only place the decision is made.
    if not gestalt or bool(gestalt.get("insufficient_evidence")):
        return TERM_INSUFFICIENT_EVIDENCE
    return TERM_INSUFFICIENT_EVIDENCE if opt_float(gestalt.get("cohesion")) is None else ""


def ship_gate_term_verdicts(
    gestalt: Mapping[str, object],
    ceilings: Mapping[str, object],
    conformance: Mapping[str, object] | None,
    *,
    settings: object,
    enforce_all: bool = False,
) -> list[ShipGateTermVerdict]:
    """Every ship-gate term's three-state verdict, in :data:`SHIP_GATE_TERMS` order.

    PURE — no I/O, no logging, no settings mutation. Each term is read from the
    verdict its OWN producer published (``loudness_verdict`` / ``momentum_verdict`` /
    ``continuity_breaks`` / the ``duration_conformance`` record), never re-derived
    here, so this can never disagree with the producer. ``conformance`` is the
    EXISTING delivered-vs-ordered verdict handed in by the caller — this gate defines
    no duration ceiling of its own and computes no new one.

    ⛔ Does NOT read ``gestalt["ship_ready"]``. That flag is a monolithic AND over all
    five terms which is also False on ``insufficient_evidence``, so enforcing on it
    would (a) fail every run whose evidence is missing and (b) be unable to name which
    term actually failed. Per-term is what makes arming survivable."""
    out: list[ShipGateTermVerdict] = []

    def add(term: str, state: str, value: float | None, limit: str | None, detail: str) -> None:
        out.append(
            ShipGateTermVerdict(
                term=term,
                state=state,
                tier=gate_tier(term),
                armed=gate_armed(settings, term, enforce_all=enforce_all),
                value=value,
                limit=limit,
                detail=detail,
            )
        )

    # continuity — adjacent-shot L2 breaks against the (hardcoded) ceiling.
    _c = opt_float(ceilings.get("continuity_l2"))
    ceiling = _DEFAULT_CONTINUITY_CEILING if _c is None else _c
    cont = _continuity_verdict(gestalt, ceilings)
    breaks = gestalt.get("continuity_breaks")
    n_breaks = (
        float(len(breaks)) if isinstance(breaks, Sequence) and not isinstance(breaks, str) else None
    )
    add(
        TERM_CONTINUITY,
        cont,
        None if cont == TERM_INSUFFICIENT_EVIDENCE else n_breaks,
        f"0 breaks at L2 > {ceiling:.3f}",
        "visual continuity unobservable on this wire shape"
        if cont == TERM_INSUFFICIENT_EVIDENCE
        else f"{int(n_breaks or 0)} continuity break(s) over L2 {ceiling:.3f}",
    )

    # cohesion — whole-film feature coherence against the floor.
    cohesion = opt_float(gestalt.get("cohesion"))
    _f = opt_float(ceilings.get("cohesion_floor"))
    floor = _DEFAULT_COHESION_FLOOR if _f is None else _f
    coh_unmeasured = _cohesion_verdict(gestalt)
    if coh_unmeasured:
        add(TERM_COHESION, coh_unmeasured, None, f">= {floor:.3f}", "cohesion unmeasured")
    else:
        assert cohesion is not None  # noqa: S101 - narrowed by `_cohesion_verdict`
        add(
            TERM_COHESION,
            TERM_FAIL if cohesion < floor else TERM_PASS,
            cohesion,
            f">= {floor:.3f}",
            f"cohesion {cohesion:.4f} against floor {floor:.3f}",
        )

    # momentum — read off the producer's own three-state verdict. `unconfigured`
    # (the live state: nothing sets `momentum_gap_max`) is INSUFFICIENT EVIDENCE,
    # never a pass: a term with no ceiling observed nothing.
    gap = opt_float(gestalt.get("momentum_gap"))
    _m = opt_float(ceilings.get("momentum_gap_max"))
    mom_state = {
        MOMENTUM_PASS: TERM_PASS,
        MOMENTUM_FAIL: TERM_FAIL,
        MOMENTUM_UNCONFIGURED: TERM_INSUFFICIENT_EVIDENCE,
        MOMENTUM_INSUFFICIENT_EVIDENCE: TERM_INSUFFICIENT_EVIDENCE,
    }.get(str(gestalt.get("momentum_verdict") or ""), TERM_INSUFFICIENT_EVIDENCE)
    add(
        TERM_MOMENTUM,
        mom_state,
        gap if mom_state != TERM_INSUFFICIENT_EVIDENCE else None,
        "unconfigured (no momentum_gap_max)" if _m is None else f"<= {_m:.3f}",
        "no momentum ceiling configured — nothing observed"
        if _m is None
        else f"momentum_gap {gap if gap is None else round(gap, 4)} against {_m:.3f}",
    )

    # loudness — the compiler's MEASURED integrated LUFS, read off its three-state
    # verdict. The one term that is an objective upstream measurement.
    lo, hi = loudness_window(ceilings)
    loud = gestalt.get("loudness")
    lufs = (
        opt_float(cast("Mapping[str, object]", loud).get("integrated_lufs"))
        if isinstance(loud, Mapping)
        else None
    )
    loud_state = {
        LOUDNESS_PASS: TERM_PASS,
        LOUDNESS_FAIL: TERM_FAIL,
        LOUDNESS_INSUFFICIENT_EVIDENCE: TERM_INSUFFICIENT_EVIDENCE,
    }.get(str(gestalt.get("loudness_verdict") or ""), TERM_INSUFFICIENT_EVIDENCE)
    add(
        TERM_LOUDNESS,
        loud_state,
        lufs if loud_state != TERM_INSUFFICIENT_EVIDENCE else None,
        f"[{lo:.1f}, {hi:.1f}] LUFS",
        "loudness unmeasured — no evidence either way"
        if loud_state == TERM_INSUFFICIENT_EVIDENCE
        else f"{'unmeasured' if lufs is None else f'{lufs:.1f} LUFS'} against [{lo:.1f}, {hi:.1f}]",
    )

    # duration — the EXISTING delivered-vs-ordered verdict, READ not re-invented.
    # `None` means nobody ordered a usable length: insufficient evidence, never a
    # pass (a gate that reads "no order" as "order met" is how a 218.47s film
    # shipped green against a 30.0s order).
    if conformance is None:
        add(
            TERM_DURATION,
            TERM_INSUFFICIENT_EVIDENCE,
            None,
            f"+/-{DURATION_TOLERANCE_FRACTION * 100.0:.0f}% of order",
            "no usable ordered duration — conformance not checked",
        )
    else:
        ratio = opt_float(conformance.get("ratio"))
        add(
            TERM_DURATION,
            TERM_PASS if conformance.get("within_tolerance") else TERM_FAIL,
            opt_float(conformance.get("deviation_pct")),
            f"+/-{DURATION_TOLERANCE_FRACTION * 100.0:.0f}% of order",
            f"delivered {opt_float(conformance.get('delivered_seconds')) or 0.0:.1f}s vs "
            f"{opt_float(conformance.get('ordered_seconds')) or 0.0:.1f}s ordered "
            f"({(ratio or 0.0) * 100.0:.0f}% of order, basis="
            f"{conformance.get('delivered_basis')})",
        )
    return out


def ship_gate_decision(verdicts: Sequence[ShipGateTermVerdict]) -> str:
    """Aggregate the ARMED terms into one three-state whole-gate decision.

    * :data:`GATE_FAIL` — at least one ARMED term returned an AFFIRMATIVE
      :data:`TERM_FAIL`. This, and only this, blocks the run.
    * :data:`GATE_PASS` — at least one armed term actually MEASURED its subject and
      cleared, and none failed. A measurement is required: "nothing objected"
      is not a pass when nothing looked.
    * :data:`GATE_INSUFFICIENT_EVIDENCE` — no armed term produced a measurement (or
      no term is armed at all). The artifact still ships, but the record says the gate
      made no claim. This is the state that stops a starved judge scoring a pass —
      the footage critic starved on 48.7% of runs and "no flagged shots" was read as
      good footage for months.

    Disarmed terms are reported on the record and ignored here, by construction: the
    arming axis is the only thing that converts a verdict into a decision."""
    armed = [v for v in verdicts if v.armed]
    if any(v.state == TERM_FAIL for v in armed):
        return GATE_FAIL
    if any(v.state == TERM_PASS for v in armed):
        return GATE_PASS
    return GATE_INSUFFICIENT_EVIDENCE


def ship_gate_record(verdicts: Sequence[ShipGateTermVerdict], decision: str) -> dict[str, object]:
    """The persisted ship-gate record: the decision plus EVERY term's row.

    Every term appears — armed or not, passed, failed or unmeasured — so the record
    answers "which ceiling failed, with what value, and was it even armed" from the
    row itself. A gate that fails opaquely gets switched off within a day; this is
    what makes a false failure calibratable instead."""
    return {
        "decision": decision,
        "terms": [v.as_row() for v in verdicts],
        "armed_terms": [v.term for v in verdicts if v.armed],
        "failed_terms": [v.term for v in verdicts if v.armed and v.state == TERM_FAIL],
        "unmeasured_terms": [
            v.term for v in verdicts if v.armed and v.state == TERM_INSUFFICIENT_EVIDENCE
        ],
    }


def _duration_override(
    ctx: RunContext,
    gestalt: Mapping[str, object],
    duration: float,
    *,
    stage: str,
) -> tuple[Mapping[str, object], list[FinalCritiqueFinding]]:
    """Apply the ordered-vs-delivered verdict to the ship decision, deterministically.

    Returns ``(gestalt, extra_findings)``. Inside tolerance, or with no usable ordered
    duration, BOTH are returned unchanged/empty — this can only ever withdraw a
    ship-ready badge, never grant one, so a run that was not ship-ready before is
    untouched and a conforming run is byte-identical to before.

    ⚠️ ``gestalt`` is only rewritten when it is already NON-EMPTY. An empty gestalt
    already yields ``ship_ready=False`` and a ``needs-review`` badge, so the override has
    nothing to withdraw there — and seeding ``{"ship_ready": False}`` into an empty dict
    would make it truthy, which flips :func:`_unavailable_reason` from "gestalt was not
    computed" to "computed but carries no usable signal" and would count the row as a
    calibration sample it is not. The finding is still recorded in that case, so the
    breach is never silent.
    """
    conformance = observe_duration_conformance(
        stage, demo_video_ordered_seconds(ctx.config_snapshot), duration
    )
    if conformance is None or conformance["within_tolerance"]:
        return gestalt, []
    detail = (
        f"delivered {conformance['delivered_seconds']:.1f}s against "
        f"{conformance['ordered_seconds']:.1f}s ordered "
        f"({conformance['ratio'] * 100.0:.0f}% of order, "
        f"{conformance['deviation_pct']:+.1f}% deviation) — outside "
        f"+/-{DURATION_TOLERANCE_FRACTION * 100.0:.0f}%; not ship-ready regardless of "
        "the ceilings verdict"
    )
    finding = FinalCritiqueFinding(
        dimension="duration_conformance",
        severity="major",
        seam=None,
        detail=detail[:_MAX_TEXT],
    )
    if not gestalt:
        return gestalt, [finding]
    # The ceilings verdict is PRESERVED under its own key rather than destroyed: the
    # nightly calibration corpus reads the ceilings terms, and the fact that continuity
    # / loudness / momentum passed is still true and still worth having.
    return {
        **gestalt,
        "ship_ready": False,
        "ship_ready_ceilings": bool(gestalt.get("ship_ready")),
        "duration_conformance": dict(conformance),
    }, [finding]


def _footage_bar_override(
    ctx: RunContext, gestalt: Mapping[str, object]
) -> tuple[Mapping[str, object], list[FinalCritiqueFinding]]:
    """Withdraw ship-ready when the footage critic scored the shipped footage BELOW its bar.

    Reads the structured ``footage_below_bar`` degradation ``demo_video_footage_critique``
    published — never its prose note, never a re-derived score — so this cannot disagree
    with the producer. Run 01a0ab23 shipped footage scored 2.0 against a 4.0 bar; the
    ceilings verdict cannot see footage quality at all, so nothing stopped a
    ``ship-ready`` badge.

    Same contract as :func:`_duration_override`: absent code (the stage skipped, a row
    from before the code existed, or footage that met the bar) returns BOTH unchanged,
    it can only WITHDRAW the badge, an EMPTY gestalt is not seeded, and the ceilings
    verdict is preserved under ``ship_ready_ceilings`` — ``setdefault`` semantics, so a
    duration override that already moved it is not overwritten with ``False``.
    Badge only: this never fails the run and never re-drives anything.
    """
    critique = ctx.prior_output("demo_video_footage_critique") or {}
    raw = critique.get("degradations") if isinstance(critique, Mapping) else None
    entries = raw if isinstance(raw, Sequence) and not isinstance(raw, str) else []
    below = [e for e in entries if isinstance(e, Mapping) and e.get("code") == FOOTAGE_BELOW_BAR]
    if not below:
        return gestalt, []
    finding = FinalCritiqueFinding(
        dimension="footage_critique",
        severity="major",
        seam=None,
        detail=(
            f"footage below the critic's bar — {below[0].get('detail', '')}; "
            "not ship-ready regardless of the ceilings verdict"
        )[:_MAX_TEXT],
    )
    if not gestalt:
        return gestalt, [finding]
    return {
        **gestalt,
        "ship_ready": False,
        "ship_ready_ceilings": gestalt.get("ship_ready_ceilings", bool(gestalt.get("ship_ready"))),
    }, [finding]


def _badge(ship_ready: bool, overall: float | None) -> str:
    if ship_ready:
        return "ship-ready"
    if overall is None:
        return "needs-review"
    if overall >= 4.0:
        return "strong"
    if overall >= 3.0:
        return "fair"
    return "weak"


def _chapters(
    scene_boundaries: Sequence[float] | None, duration: float
) -> list[FinalCritiqueChapter]:
    """A minimal chapter list from the analyzer's scene cuts (start + generic title)."""
    starts: list[float] = []
    for raw in scene_boundaries or []:
        t = as_float(raw, -1.0)
        if t < 0.0 or (duration > 0.0 and t > duration):
            continue
        starts.append(round(t, 3))
    starts = sorted(set(starts))
    if not starts or starts[0] > 0.0:
        starts = [0.0, *starts]
    starts = starts[:_MAX_CHAPTERS]
    return [FinalCritiqueChapter(start=s, title=f"Chapter {i + 1}") for i, s in enumerate(starts)]


def _nearest_frame(
    frame_analyses: Sequence[Mapping[str, object]], t: float
) -> Mapping[str, object]:
    """The frame record sampled AT OR AFTER the cut ``t`` (``{}`` when none).

    Directional ON PURPOSE. ``t`` is a shot's START, and the analyzer samples on a fixed
    grid (``frame_interval_seconds`` defaults to 3.0), so an undirected
    nearest-neighbour lookup routinely returns the frame BEFORE the cut — which is the
    PREVIOUS shot's content. Measured: with a 3.0s grid, ``clip-2`` starting at t=15.8
    resolved to the frame at t=15.0, i.e. clip-1. Every "per-shot" feature vector and
    every per-shot description could therefore describe the wrong shot, silently
    inverting which seam a finding names.

    Preference order: the EARLIEST frame at or after ``t`` (the first frame that actually
    shows this shot), falling back to the closest earlier frame only when the cut sits
    past the last sample — better a late frame than no frame, and the fallback keeps a
    trailing shot from dropping out of the evidence entirely.
    """
    after: Mapping[str, object] | None = None
    after_gap = float("inf")
    before: Mapping[str, object] | None = None
    before_gap = float("inf")
    for fa in frame_analyses:
        if not isinstance(fa, Mapping):
            continue
        ts = fa.get("timestamp")
        if ts is None:
            ts = fa.get("t")
        if isinstance(ts, bool) or not isinstance(ts, (int, float)):
            continue
        delta = float(ts) - t
        if delta >= 0.0:
            if delta < after_gap:
                after_gap = delta
                after = fa
        elif -delta < before_gap:
            before_gap = -delta
            before = fa
    return after or before or {}


def _motion_at(motion_timeline: Sequence[VideoAnalyzerMotionSample] | None, t: float) -> float:
    """The motion sample nearest ``t`` from the analyzer's typed motion timeline."""
    if not motion_timeline:
        return 0.0
    best_motion = 0.0
    best_gap = float("inf")
    for sample in motion_timeline:
        gap = abs(sample.t - t)
        if gap < best_gap:
            best_gap = gap
            best_motion = sample.motion
    return best_motion


def _nearest_visual(
    visual_timeline: Sequence[VideoAnalyzerVisualSample] | None, t: float
) -> VideoAnalyzerVisualSample | None:
    """The per-frame visual sample nearest ``t`` from the analyzer's ``visual_timeline``.

    ``None`` when the timeline is empty — a job an older analyzer wrote carries no
    per-frame visual curve, so the caller falls back to the whole-video means (which
    gestalt then reads as zero-dispersion / insufficient_evidence, not a fake pass).
    """
    if not visual_timeline:
        return None
    best: VideoAnalyzerVisualSample | None = None
    best_gap = float("inf")
    for sample in visual_timeline:
        gap = abs(sample.t - t)
        if gap < best_gap:
            best_gap = gap
            best = sample
    return best


def _feature_vector(
    brightness: float, sharpness: float, rgb: Sequence[object], motion: float
) -> list[float]:
    """A NORMALIZED per-shot float vector: brightness, sharpness, dominant-colour RGB, motion.

    Each component is mapped into ~[0,1] (see ``_SHARPNESS_SCALE``) so the L2 metrics in
    :mod:`gestalt_metrics` weight every axis comparably instead of being dominated by the
    widest native scale (sharpness/colour). ``brightness`` and ``motion`` are already
    luma-/delta-normalized (clamped defensively); ``rgb`` is 0..255 per channel (padded to
    three, ``/255``); ``sharpness`` is a Laplacian variance squashed by the saturating
    ``s/(s+K)`` — monotonic, outlier-robust, no max to guess. All inputs are read
    defensively: junk is a zero component, never a raise.
    """
    b = _clamp01(as_float(brightness))
    raw_s = as_float(sharpness)
    s = raw_s / (raw_s + _SHARPNESS_SCALE) if raw_s > 0.0 else 0.0  # saturating → [0, 1)
    channels = [as_float(c) for c in list(rgb)[:3]]
    while len(channels) < 3:
        channels.append(0.0)
    r, g, bl = (max(0.0, min(255.0, c)) / 255.0 for c in channels)
    m = _clamp01(as_float(motion))
    return [b, s, r, g, bl, m]


def _clamp01(x: float) -> float:
    """Clamp to the closed unit interval — an already-normalized component's guard."""
    return max(0.0, min(1.0, x))


def _merge_by_id(
    old_list: Sequence[object],
    regenerated: Sequence[object],
    id_field: str,
    flagged: frozenset[str],
) -> list[dict[str, object]]:
    """Splice a regenerated stage output back onto its baseline, by id.

    Walks ``old_list`` in order, keeping every UNFLAGGED row verbatim and replacing
    each flagged id's rows with the regenerated ones for that id (a stage may emit
    0..N rows per id — sound cues — so this is per-id, not 1:1). Flagged ids the
    critic re-emitted that had no baseline row (e.g. an accent added where there was
    none) are appended in regenerated order, so a re-drive can ADD, not only replace.
    Baseline dicts are copied, never mutated (ctx's own dicts stay untouched)."""
    new_by_id: dict[str, list[dict[str, object]]] = {}
    for item in regenerated:
        if not isinstance(item, Mapping):
            continue
        row = cast("Mapping[str, object]", item)
        iid = str(row.get(id_field))
        if iid in flagged:
            new_by_id.setdefault(iid, []).append(dict(row))

    merged: list[dict[str, object]] = []
    emitted: set[str] = set()
    for entry in old_list:
        if not isinstance(entry, Mapping):
            continue
        erow = cast("Mapping[str, object]", entry)
        eid = str(erow.get(id_field))
        if eid in flagged:
            if eid not in emitted:
                merged.extend(new_by_id.get(eid, []))
                emitted.add(eid)
            continue  # the baseline flagged row is superseded by the regenerated one(s)
        merged.append(dict(erow))
    for iid, items in new_by_id.items():
        if iid not in emitted:
            merged.extend(items)
    return merged


def _redrive_feedback(
    flagged: frozenset[str],
    reasons: Sequence[str],
    findings: Sequence[FinalCritiqueFinding],
) -> str:
    """The film-critic feedback threaded into a re-driven stage's LLM prompt.

    The directive reason(s) for that stage, followed by one bullet per whole-film
    finding whose seam touches a flagged id — so the director sees exactly why its
    beats were called weak. Empty findings / reasons simply contribute nothing.

    Each bullet names the SEAM as well as the detail. Before ``detail`` was threaded
    through (it was missing from ``_OUTPUT_SCHEMA`` and never read in
    ``_interpret_findings``) every bullet was literally ``"- [continuity/minor] "`` — the
    re-driven director was told a dimension and a severity and nothing else. Carrying the
    seam ids keeps a bullet informative even when the model omits the detail."""
    parts: list[str] = []
    joined = " ".join(r for r in reasons if r).strip()
    if joined:
        parts.append(joined)
    for finding in findings:
        seam = finding.seam
        if seam is None or not (seam.from_id in flagged or seam.to_id in flagged):
            continue
        bullet = f"- [{finding.dimension}/{finding.severity}] {seam.from_id}->{seam.to_id}"
        if finding.detail:
            bullet = f"{bullet}: {finding.detail}"
        parts.append(bullet)
    return "\n".join(parts)


@dataclass(frozen=True)
class _CutSpine:
    """The ONE cut spine every gestalt term is measured against.

    ``cuts`` are ``(spine_id, cut_time)`` pairs; ``intended_intervals`` are the
    intended-duration ``(id, start, end)`` beat spans (the axis
    ``predicted_watch_curve`` runs on); ``lines`` is the resolved plan spine.
    """

    cuts: list[tuple[str, float]]
    intended_intervals: list[tuple[str, float, float]]
    lines: list[dict]
    # True when the INTENDED spine is corrupt (an unparseable beat ``seconds``). The
    # momentum term must then be skipped entirely: ``predicted_watch_curve`` is
    # deliberately TOLERANT and would happily read the bad beat as 0.0s, so computing a
    # gap against it would fabricate a number from a spine we already know is broken.
    # Distinct from "no intended durations at all" (``intended_intervals == []``), which
    # is a legitimate spine shape that momentum still handles.
    intended_broken: bool = False


@dataclass(frozen=True)
class _Draft:
    """One evaluated point in the re-drive loop: an artifact + its film-critic verdict.

    Carries everything ``_accept`` needs so the loop can keep the highest-scoring
    draft and unpack the retained one into the stage output. All fields are what a
    single ``_reevaluate`` pass produced against ``artifact_url``.
    """

    artifact_url: str
    duration: float
    gestalt: Mapping[str, object]
    scores: FinalCritiqueScores | None
    findings: list[FinalCritiqueFinding]
    directives: list[FinalCritiqueDirective]
    scene_boundaries: Sequence[float] | None
    # True ONLY on the ship-ready short-circuit (scores is None because the gestalt
    # cleared the bar with zero WM calls). A `scores is None` draft with
    # `ship_ready=False` is a FAILED evaluation (analyzer/WM outage), NOT a pass — the
    # re-drive loop must never treat it as a perfect score.
    ship_ready: bool = False
    # The owning LLM stages that were ACTUALLY EXECUTED to produce THIS draft's artifact,
    # accumulated across passes. NOT "what the critic asked for" and NOT "what should be
    # fixed next" — `rerun_stages` is a report of work done, and it is read as one. It was
    # previously derived from the RETAINED draft's directives filtered by baseline
    # presence, which reported `['demo_video_footage_direction']` for a run where the
    # director raised, `optimizer_fn` swallowed it, and the ORIGINAL artifact shipped.
    rerun_stages: tuple[str, ...] = ()
    # Set ONLY when the video analyzer failed for THIS draft (the deliberate fail-open:
    # the film still ships). Becomes the ``review_skipped`` degradation on accept, so the
    # unreviewed ship is visible on the run record, not only in an ERROR log.
    analyzer_error: str | None = None


class FinalCritiqueStage(StageHandler[FinalCritiqueInput, FinalCritiqueOutput]):
    """The A11 advisory ship gate. Standalone (not a footage_critique subclass); it
    reuses the same soft-degrade idioms — every analyzer/WM failure accepts the
    current artifact with an advisory finding rather than failing the deliverable."""

    name = "final_critique"
    input_model = FinalCritiqueInput
    output_model = FinalCritiqueOutput
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (
        SlotUse("llm"),
        SlotUse("tts"),
        SlotUse("asr", when=SLOT_USE_OPTIONAL),
    )
    worker_name = _WORKER_NAME

    def __init__(
        self,
        client: WMClient | None = None,
        analyzer: VideoAnalyzerClient | None = None,
        compile_client: VideoCompileClient | None = None,
        tts_client: TTSClient | None = None,
    ) -> None:
        self._llm_timeout = llm_budget_seconds()
        self._llm_retries = llm_retries()
        self._wm = client or WMClient(timeout=llm_client_timeout_seconds())
        self._analyzer = analyzer or VideoAnalyzerClient()
        # The compile client threaded into DemoVideoCompileStage on a re-drive
        # (mode="auto"); unused in critique_only / off. Same seam the footage
        # critique loop uses so a re-compile reuses the CANONICAL compile path.
        self._compile_client = compile_client or VideoCompileClient()
        # TTS client threaded into DemoVideoNarrateStage ONLY when a voice_director
        # re-drive fires (mode="auto") — narration must be re-synthesized so compile
        # reads the new clips. Left None (not eagerly constructed) so critique_only /
        # off never touch TTS; narrate builds its own default when it does run.
        self._tts_client = tts_client
        # Content-address guard: reuse an analyzer result for the same delivered
        # file hash instead of re-submitting (keyed by output sha256).
        self._result_cache: dict[str, VideoAnalyzerJobResult] = {}

    async def handle(self, payload: FinalCritiqueInput, ctx: RunContext) -> FinalCritiqueOutput:
        # Opt-in + skip-clean: absent `ship_gate` / `mode == "off"` ⇒ StageSkippedError.
        require_ship_gate(ctx)

        # Precondition: the delivered artifact. Read the accepted-best from
        # footage_critique, else compile. Only a genuinely missing artifact (both
        # empty) is a hard StagePreconditionError — the soft gate raises nothing else.
        artifact_url, duration = self._starting_artifact(ctx)
        loudness = self._compile_loudness(ctx)
        shot_timeline = self._shot_timeline(ctx)
        mode = self._mode(ctx)

        # Belt-and-suspenders soft gate: the analyzer/WM paths swallow their own
        # failures, but any other defect must STILL accept the current artifact.
        try:
            return await self._evaluate(ctx, artifact_url, duration, loudness, shot_timeline, mode)
        except StageSkippedError, StagePreconditionError, ShipGateEnforcementError:
            # ENFORCE is a TERMINAL hard-fail (opt-in), NOT a soft-degrade: it must escape
            # the accept-current wrapper below and land the run at status=failed (#719).
            raise
        except Exception as exc:
            # An UNEXPECTED defect here is NOT evidence the film is fine — it is evidence
            # the gate never reached a verdict. Two changes to stop that reading as a pass:
            #   1. ERROR level WITH the traceback. The old `%s`-only warning threw away every
            #      internal defect's stack, so a broken critique was indistinguishable from a
            #      clean advisory degrade in the logs.
            #   2. Under ENFORCE the run OPTED INTO a hard gate. Accepting an artifact the
            #      gate could not evaluate silently grants the pass the caller asked us to
            #      withhold — so re-raise terminally (same #719 mechanism as a ceilings
            #      failure) instead. With enforce false the accept-current contract below is
            #      byte-for-byte unchanged.
            log.error(
                "final_critique: gate raised before reaching a verdict: %s",
                exc,
                exc_info=True,
            )
            if self._enforce(ctx):
                raise ShipGateEnforcementError(
                    "final_critique: ship gate ENFORCE could not be evaluated — the gate "
                    f"raised before producing a verdict: {exc}"
                ) from exc
            return self._accept(
                ctx,
                artifact_url,
                duration,
                gestalt={},
                scores=None,
                findings=[_advisory("stage", str(exc))],
                directives=[],
                scene_boundaries=None,
            )

    # ── inputs ───────────────────────────────────────────────────────────────
    def _starting_artifact(self, ctx: RunContext) -> tuple[str, float]:
        # Prefer the most-downstream accepted-best artifact: the S19 encode/mux master
        # (`demo_video_encode_master`) when it ran + re-mastered, then the L4 footage
        # critique, then compile. Each earlier stage's output is falsy `{}` when it
        # skipped-clean, so the chain falls through byte-identically.
        encode = ctx.prior_output("demo_video_encode_master") or {}
        url = _usable_artifact_url(encode.get("artifact_url"))
        dur = encode.get("duration_seconds")
        if not url:
            critique = ctx.prior_output("demo_video_footage_critique") or {}
            url = _usable_artifact_url(critique.get("artifact_url"))
            dur = critique.get("duration_seconds")
        if not url:
            compile_out = ctx.prior_output("demo_video_compile") or {}
            url = _usable_artifact_url(compile_out.get("artifact_url"))
            dur = compile_out.get("duration_seconds")
        if not url:
            raise StagePreconditionError(
                "final_critique: no compiled artifact from encode_master, "
                "footage_critique or compile",
                missing=["demo_video_compile.artifact_url"],
            )
        return str(url), as_float(dur)

    def _compile_loudness(self, ctx: RunContext) -> Mapping[str, object] | None:
        compile_out = ctx.prior_output("demo_video_compile") or {}
        loudness = compile_out.get("loudness")
        return loudness if isinstance(loudness, Mapping) else None

    def _mastering(self, ctx: RunContext) -> Mapping[str, object] | None:
        """The compiler's mastering report, or ``None`` when it shipped none."""
        compile_out = ctx.prior_output("demo_video_compile") or {}
        mastering = compile_out.get("mastering")
        return cast("Mapping[str, object]", mastering) if isinstance(mastering, Mapping) else None

    def _av_gap_seconds(self, ctx: RunContext) -> float | None:
        """The compiler's measured ``av_gap_seconds`` off its mastering report.

        Seconds of delivered NARRATION the delivered PICTURE does not cover. The
        compiler emits it unconditionally on every master and, until now, nothing in
        this repo read it — so the A/V-coverage verdict rested on a hand-collected
        n=10 instead of on the number already riding in on every single run. Surfaced
        exactly like :meth:`_static_ratio`: evidence on the gestalt block, never a gate
        term. ``None`` when the compiler shipped no report (older builds).
        """
        mastering = self._mastering(ctx)
        if mastering is None:
            return None
        return opt_float(mastering.get("av_gap_seconds"))

    def _static_ratio(self, ctx: RunContext) -> float | None:
        """The compiler's measured ``duplicate_frame_ratio`` off its mastering report.

        Already measured on the real rendered file and sitting unread in the compile
        output. It is the one DIRECT signal we have for "this film barely moves", which is
        exactly the input shape that makes the visual gestalt terms unmeasurable — so the
        gate reports it rather than leaving the reader to guess why the evidence is thin.
        ``None`` when the compiler shipped no report (older builds)."""
        compile_out = ctx.prior_output("demo_video_compile") or {}
        mastering = compile_out.get("mastering")
        if not isinstance(mastering, Mapping):
            return None
        return opt_float(cast("Mapping[str, object]", mastering).get("duplicate_frame_ratio"))

    def _shot_timeline(self, ctx: RunContext) -> list[dict[str, object]] | None:
        compile_out = ctx.prior_output("demo_video_compile") or {}
        st = compile_out.get("shot_timeline")
        return st if isinstance(st, list) else None

    def _mode(self, ctx: RunContext) -> str:
        ship_gate = ctx.config_snapshot.get("ship_gate")
        if not isinstance(ship_gate, Mapping):
            return _MODE_DEFAULT
        return str(ship_gate.get("mode") or _MODE_DEFAULT).lower()

    def _ceilings(self, ctx: RunContext) -> Mapping[str, object]:
        ship_gate = ctx.config_snapshot.get("ship_gate")
        if not isinstance(ship_gate, Mapping):
            return {}
        ceilings = ship_gate.get("ceilings")
        return ceilings if isinstance(ceilings, Mapping) else {}

    def _enforce(self, ctx: RunContext) -> bool:
        """True when this run opted into ENFORCE (the A11 opt-in HARD gate), read off the
        per-run ``ship_gate`` block exactly as ``_mode`` / ``_ceilings`` read theirs — so a
        run-level ``ship_gate.enforce`` override (or the seeded default) is honoured.
        Absent ⇒ False, which is byte-identical to today (accept best)."""
        ship_gate = ctx.config_snapshot.get("ship_gate")
        if not isinstance(ship_gate, Mapping):
            return False
        return bool(ship_gate.get("enforce"))

    def _term_verdicts(self, ctx: RunContext, draft: _Draft) -> list[ShipGateTermVerdict]:
        """Per-term three-state verdicts for ``draft``, with arming resolved.

        Arming is read from ``PipelineSettings`` at STAGE time (not from the run's
        ``config_snapshot``), so flipping a kill switch in the GitOps configmap takes
        effect on runs that were created before the flip. The per-run
        ``ship_gate.enforce: true`` block is still honoured, as the escalation that arms
        every term."""
        # The SAME pure verdict the `_accept` duration override applies, computed from
        # the same two inputs — so the decision and the badge can never disagree about
        # length. `duration_conformance` (not `observe_…`) because the observing variant
        # logs, and this must not double-log the accept path's line.
        conformance = duration_conformance(
            demo_video_ordered_seconds(ctx.config_snapshot), draft.duration
        )
        return ship_gate_term_verdicts(
            draft.gestalt,
            self._ceilings(ctx),
            conformance,
            settings=get_settings().pipeline,
            enforce_all=self._enforce(ctx),
        )

    def _enforced_accept(
        self, ctx: RunContext, draft: _Draft, *, rerun_stages: list[str]
    ) -> FinalCritiqueOutput:
        """The single accept choke-point: accept ``draft`` unless an ARMED ship-gate term
        affirmatively FAILED — then hard-fail the run.

        ⚠️ THIS IS THE ARMING SITE, and what it replaced is why the gate could not fail.
        It used to require ``ship_gate.enforce`` AND be reachable only from
        :meth:`_auto_redrive` (i.e. ``mode == auto``); production seeds
        ``{"mode": "critique_only"}`` with no ``enforce`` key, so BOTH conditions were
        false at once and ``ship_ready`` only ever explained. It is now reached from
        :meth:`_evaluate` on every mode the stage does not skip, and the decision comes
        from per-term arming (:func:`gate_armed`) rather than one never-set flag.

        THE THREE STATES ARE KEPT APART. Only :data:`GATE_FAIL` — an armed term with an
        AFFIRMATIVE fail verdict — blocks. :data:`GATE_INSUFFICIENT_EVIDENCE` means no
        armed term could observe its subject: the artifact ships (a gate that could not
        measure has not convicted anything) but the record says the gate made no claim,
        so it is not a pass either. That distinction is load-bearing —
        ``gestalt_summary`` is measurably ``{}`` on ~55% of live rows, and enforcing on
        ``ship_ready`` (False on ``insufficient_evidence``) would have failed all of them
        over a missing measurement.

        The raised message NAMES every armed failing term with its value and the limit it
        was compared against; the full per-term record is persisted on the output."""
        verdicts = self._term_verdicts(ctx, draft)
        decision = ship_gate_decision(verdicts)
        if decision == GATE_FAIL:
            failed = [v for v in verdicts if v.armed and v.state == TERM_FAIL]
            named = "; ".join(f"{v.term} [{v.tier}]: {v.detail} (limit {v.limit})" for v in failed)
            log.error(
                "final_critique: ship gate FAILED — armed term(s) %s: %s",
                [v.term for v in failed],
                named,
            )
            raise ShipGateEnforcementError(
                "final_critique: ship gate FAILED — "
                f"{len(failed)} armed term(s) did not meet their ceiling: {named}"
            )
        if decision == GATE_INSUFFICIENT_EVIDENCE:
            # Never a pass. Logged at WARNING and recorded so a gate that is measuring
            # nothing is visible as such, instead of reading as a clean run.
            log.warning(
                "final_critique: ship gate made NO CLAIM (insufficient evidence) — armed "
                "terms %s all unmeasured; accepting the artifact without a verdict",
                [v.term for v in verdicts if v.armed],
            )
        return self._accept_draft(ctx, draft, rerun_stages=rerun_stages, verdicts=verdicts)

    def _spine_ids(self, ctx: RunContext) -> list[str]:
        lines = resolve_plan_output(ctx).get("lines") or []
        return [
            str(line["id"])
            for line in lines
            if isinstance(line, Mapping) and line.get("id") is not None
        ]

    def _override_ids(self, ctx: RunContext) -> tuple[frozenset[str], frozenset[str]]:
        """Author-set per-shot ship-gate overrides (A11 PR-4), read off the current
        best line spine: ``(pinned, exempt)``. ``pinned`` beats are NEVER a re-drive
        target; ``exempt`` beats suppress the ship-gate continuity finding touching
        them. Both default empty ⇒ absent overrides are byte-identical."""
        lines = resolve_plan_lines(ctx)
        pinned = frozenset(
            str(line["id"])
            for line in lines
            if isinstance(line, Mapping) and line.get("id") and line.get("pin")
        )
        exempt = frozenset(
            str(line["id"])
            for line in lines
            if isinstance(line, Mapping) and line.get("id") and line.get("continuity_exempt")
        )
        return pinned, exempt

    def _output_sha(self, ctx: RunContext) -> str:
        compile_out = ctx.prior_output("demo_video_compile") or {}
        sha = compile_out.get("output_sha256")
        return str(sha) if isinstance(sha, str) else ""

    # ── evaluation ─────────────────────────────────────────────────────────────
    async def _evaluate(
        self,
        ctx: RunContext,
        artifact_url: str,
        duration: float,
        loudness: Mapping[str, object] | None,
        shot_timeline: list[dict[str, object]] | None,
        mode: str,
    ) -> FinalCritiqueOutput:
        # One evaluation pass over the delivered artifact (analyzer → gestalt →
        # film-critic). `critique_only` / `off` accept it verbatim (rerun_stages=[],
        # byte-identical to PR-1); `auto` hands it to the bounded re-drive loop.
        initial = await self._reevaluate(ctx, artifact_url, duration, loudness, shot_timeline)
        if mode == _MODE_AUTO:
            return await self._auto_redrive(ctx, initial)
        # ⚠️ `_enforced_accept`, NOT `_accept_draft`. `mode` is the REPAIR axis (does the
        # gate try to fix a failure) and arming is the BLOCKING axis; routing
        # critique_only through the bare accept conflated them and made the only
        # hard-fail site unreachable on the one mode production actually runs. A
        # critique_only run now blocks on an armed failing term — it simply does not
        # attempt a re-drive first.
        return self._enforced_accept(ctx, initial, rerun_stages=[])

    async def _reevaluate(
        self,
        ctx: RunContext,
        artifact_url: str,
        duration: float,
        loudness: Mapping[str, object] | None,
        shot_timeline: list[dict[str, object]] | None,
    ) -> _Draft:
        """Score ONE artifact end-to-end and return a ``_Draft``.

        The analyzer + WM failure paths degrade EXACTLY as PR-1 did — but return a
        ``_Draft`` (directives=[]) instead of an ``_accept`` output — so the re-drive
        loop can uniformly keep the highest-scoring draft. ``sha`` is read off ``ctx``
        (the re-compile swaps a fresh compile output onto its ctx, so a re-driven
        artifact is analyzed against its OWN sha, never a stale cache hit)."""
        spine_ids = self._spine_ids(ctx)
        pinned_ids, exempt_ids = self._override_ids(ctx)
        sha = self._output_sha(ctx)

        # SOFT: any analyzer failure ⇒ advisory finding + accept the current artifact.
        try:
            result = await self._analyze(artifact_url, sha)
        except Exception as exc:
            # TOLERABLE degrade: the analyzer is the gate's EVIDENCE source, not the
            # deliverable. Losing it costs the measurement, and the empty gestalt already
            # empty gestalt now reads as `insufficient_evidence` on every term, which is
            # NEITHER a pass nor a fail: losing the analyzer must not convict the film.
            # But it is still a swallowed exception, so it is logged at ERROR with the
            # traceback — never a bare `%s` that hides which call actually blew up.
            log.error(
                "final_critique: analyzer unavailable, accepting current: %s",
                exc,
                exc_info=True,
            )
            return _Draft(
                artifact_url=artifact_url,
                duration=duration,
                gestalt={},
                scores=None,
                findings=[_advisory("analyzer", str(exc))],
                directives=[],
                scene_boundaries=None,
                analyzer_error=describe_exception(exc)[:300],
            )

        spine = self._cut_spine(ctx, result, duration, spine_ids, shot_timeline)
        frames, tone_shots = self._build_frames(result, spine.cuts)

        # A11 PR-5: measure the ACTUAL post-render momentum curve (motion-at-cuts,
        # retention-shaped) and diff it against A1's PREDICTED watch curve, surfacing
        # the worst sag as ADVISORY evidence for the critic (never a gate). SOFT: any
        # bad / empty / degenerate signal ⇒ gap None (unobservable, never a flattering 0).
        #
        # The cut spine itself is built ONCE (see `_cut_spine`) and shared with the
        # frames above, so the gate's continuity/cohesion/tonal evidence and its momentum
        # evidence can never be computed against two different sets of cuts.
        try:
            if spine.intended_broken:
                # The predicted curve is derived from the SAME beat durations that failed
                # to parse; `predicted_watch_curve` is tolerant and would read them as
                # 0.0s, so a gap computed against it would be fiction. Say "unmeasured".
                raise ValueError("intended beat spine unusable — momentum not measurable")
            t0 = spine.cuts[0][1] if spine.cuts else 0.0
            motion_timeline = (
                result.quality_metrics.motion_timeline if result.quality_metrics else None
            )
            beat_motion = [_motion_at(motion_timeline, t) for _, t in spine.cuts]
            if not any(m > 0.0 for m in beat_motion):
                # NO measured motion anywhere. The assembled curve degenerates to the same
                # pure exponential decay as the predicted one, so the diff reads a tidy
                # ~0.0 — which is not "the film delivered its momentum", it is "the
                # analyzer handed us nothing to modulate with". Observing that 0.0 into the
                # nightly corpus drives the derived `momentum_gap_max` CEILING to 0.0.
                # Uniform NON-zero motion is a real (if uninformative) measurement and is
                # deliberately still measured — only total absence is unmeasurable.
                raise ValueError("no motion measured at any cut — momentum not measurable")
            # The retention CLIFF is a step, and momentum_gap normalizes scale but not a
            # step: evaluating it on the rendered clock here while `predicted_watch_curve`
            # evaluates it on the INTENDED cumulative offset injected a spurious ~0.11 gap
            # whenever a render overrun straddled the 30s cliff. Feed the intended offset
            # of each cut's own beat so both curves take the step at the same beat.
            intended_offset = {bid: start for bid, start, _end in spine.intended_intervals}
            cliff_times = [intended_offset.get(sid, t - t0) for sid, t in spine.cuts]
            assembled = assembled_momentum_curve(
                [t - t0 for _, t in spine.cuts],
                beat_motion,
                duration,
                cliff_times=cliff_times,
            )
            assembled_pairs = list(zip([sid for sid, _ in spine.cuts], assembled, strict=True))
            predicted_pairs = [
                (str(line["id"]), value)
                for line, value in zip(spine.lines, predicted_watch_curve(spine.lines), strict=True)
                if isinstance(line, Mapping) and line.get("id") is not None
            ]
            gap = momentum_gap(assembled_pairs, predicted_pairs)
        except Exception:  # momentum is advisory, never breaks the gate
            log.warning(
                "final_critique: momentum-gap computation failed; persisting null",
                exc_info=True,
            )
            gap = None

        gestalt = gestalt_summary(
            frames=frames,
            frame_analyses=tone_shots,
            loudness=loudness,
            ceilings=self._ceilings(ctx),
            momentum_gap=gap,
            mastering=self._mastering(ctx),
        )
        # The compiler's measured duplicate-frame ratio rides in as EVIDENCE (never a gate
        # term): it is the direct explanation for why a degenerate verdict is degenerate.
        static_ratio = self._static_ratio(ctx)
        # `av_gap_seconds` rides the SAME way and for the same reason: measured by the
        # compiler on the real rendered file, emitted on every master, and strictly
        # additive here — evidence, never a gate term (`ship_ready` does not read it).
        gestalt = {
            **gestalt,
            "static_frame_ratio": static_ratio,
            "av_gap_seconds": self._av_gap_seconds(ctx),
        }
        # INERT observability: push the ship-gate's numeric terms to Prometheus + one
        # log line so the ceilings can be calibrated from real renders. Never mutates
        # gestalt / affects ship_ready; self-swallows any failure.
        _emit_gestalt_metrics(gestalt)
        scene_boundaries = result.scene_boundaries

        # Ship-ready gestalt ⇒ accept with ZERO WM calls.
        if bool(gestalt.get("ship_ready")):
            return _Draft(
                artifact_url=artifact_url,
                duration=duration,
                gestalt=gestalt,
                scores=None,
                findings=[],
                directives=[],
                scene_boundaries=scene_boundaries,
                ship_ready=True,
            )

        # A deterministic advisory finding when the measured loudness sits outside the
        # ship window — the loudness is measured, no need to ask the LLM about it. It
        # rides on every not-ship-ready accept (WM success OR failure).
        base_findings = self._loudness_findings(gestalt, ctx)

        # INSUFFICIENT EVIDENCE — either fewer than two sampled shots, or two-or-more
        # shots whose feature vectors are IDENTICAL (zero dispersion). Continuity,
        # cohesion and tonal drift are unobservable in both cases, so there is nothing for
        # a film critic to critique (the gestalt it would receive is a wall of nulls):
        # skip the WM call and say so plainly. STILL SOFT: the artifact ships unchanged,
        # growth_bundle still runs — only the badge degrades to `needs-review`
        # (scores is None ⇒ `_badge(False, None)`). Before this, the same input produced
        # `ship_ready=True` off two vacuously-true terms and a `ship-ready` badge.
        if bool(gestalt.get("insufficient_evidence")):
            log.warning(
                "final_critique: insufficient evidence (%d sampled shot(s), "
                "static_frame_ratio=%s) — the ship gate cannot be evaluated; "
                "accepting current with needs-review",
                len(frames),
                static_ratio,
            )
            findings = [*base_findings, _advisory("gestalt", _INSUFFICIENT_EVIDENCE_DETAIL)]
            findings.extend(_static_input_findings(static_ratio))
            return _Draft(
                artifact_url=artifact_url,
                duration=duration,
                gestalt=gestalt,
                scores=None,
                findings=findings,
                directives=[],
                scene_boundaries=scene_boundaries,
            )
        base_findings = [*base_findings, *_static_input_findings(static_ratio)]

        # Not ship-ready ⇒ ask the film-critic worker. SOFT: any WM error ⇒ advisory
        # finding + accept.
        try:
            data = await self._critic(artifact_url, gestalt, spine_ids)
        except Exception as exc:
            # TOLERABLE degrade: the critic is an ENRICHMENT over an already-measured
            # gestalt (which stays populated here), the badge falls to needs-review, and
            # ENFORCE still reads `ship_ready` off that real gestalt. ERROR + traceback so
            # a persistently broken worker is visible rather than a one-line advisory.
            log.error(
                "final_critique: film-critic worker error, accepting current: %s",
                exc,
                exc_info=True,
            )
            return _Draft(
                artifact_url=artifact_url,
                duration=duration,
                gestalt=gestalt,
                scores=None,
                findings=[*base_findings, _advisory("film_critic", str(exc))],
                directives=[],
                scene_boundaries=scene_boundaries,
            )

        scores, score_findings = _interpret_scores(data)
        return _Draft(
            artifact_url=artifact_url,
            duration=duration,
            gestalt=gestalt,
            scores=scores,
            findings=[*base_findings, *score_findings, *_interpret_findings(data, exempt_ids)],
            directives=_interpret_directives(data, spine_ids, pinned_ids),
            scene_boundaries=scene_boundaries,
        )

    # ── bounded re-drive (mode="auto") ─────────────────────────────────────────
    def _redrive_bounds(self, ctx: RunContext) -> tuple[int, float]:
        """``(max_passes, score_threshold)`` for the re-drive loop, read off the
        ``ship_gate`` config block (mirrors ``retention_plan``'s
        ``ret.get("iterations")`` / ``ret.get("score_threshold")``). ``max_passes`` is
        hard-clamped to ``[1, 2]``; both default to the module constants."""
        ship_gate = ctx.config_snapshot.get("ship_gate")
        cfg = ship_gate if isinstance(ship_gate, Mapping) else {}
        raw_passes = cfg.get("max_passes")
        passes = (
            int(raw_passes)
            if isinstance(raw_passes, (int, float)) and not isinstance(raw_passes, bool)
            else _REDRIVE_MAX_PASSES
        )
        passes = max(1, min(2, passes))
        threshold = as_float(cfg.get("score_threshold"), _REDRIVE_THRESHOLD)
        return passes, threshold

    async def _auto_redrive(self, ctx: RunContext, initial: _Draft) -> FinalCritiqueOutput:
        """Fire a bounded (≤2-pass) re-drive of the owning per-dimension LLM stages.

        Only directives on ``_LLM_REDRIVABLE`` stages re-run a stage; every other
        directive stays advisory. The loop is seeded with ``initial`` and keeps the
        HIGHEST-scoring draft, so a worse pass-2 ships pass-1's artifact (monotonic
        non-regression). Every throw site degrades to accept the current best — the
        ship gate never blocks."""
        redrivable = [d for d in initial.directives if d.stage in _LLM_REDRIVABLE]
        if not redrivable:
            # No owning LLM stage to re-run ⇒ no re-drive (byte-identical to
            # critique_only for this draft). Still enforce-gated: under enforce a
            # ceilings-failing draft the gate cannot improve is a terminal failure.
            return self._enforced_accept(ctx, initial, rerun_stages=[])

        max_passes, threshold = self._redrive_bounds(ctx)
        # Every swallowed re-drive failure, as an advisory finding on the ACCEPTED output.
        # Logging alone left the persisted record of a re-drive whose director raised
        # byte-identical to one that ran and simply did not improve — the same silence
        # every other soft-degrade branch here avoids by recording an `_advisory`.
        redrive_failures: list[FinalCritiqueFinding] = []

        async def critic_fn(draft: _Draft) -> Critique:
            # Re-uses the film-critic verdict already on the draft (no extra WM call).
            # `scores is None` is ambiguous: a genuinely ship-ready draft converged
            # (5.0), but an analyzer/WM FAILURE on a recompiled artifact is a failed
            # evaluation that must never outrank the known-good initial — score it
            # below the clamp floor so keep-highest keeps the current best.
            if draft.scores is not None:
                score = draft.scores.overall
            elif draft.ship_ready:
                score = _OVERALL_MAX
            else:
                score = _FAILED_EVAL_SCORE
            rerun = [d.stage for d in draft.directives if d.stage in _LLM_REDRIVABLE]
            return Critique(score=score, rerun_stages=rerun, directives=None)

        async def optimizer_fn(draft: _Draft, _critique: Critique) -> _Draft:
            # SOFT: ANY re-drive / re-compile / re-analyze failure returns the SAME
            # draft ⇒ the loop plateaus on the current best rather than raising.
            try:
                directives = [d for d in draft.directives if d.stage in _LLM_REDRIVABLE]
                if not directives:
                    return draft
                new_ctx, out, executed = await self._redrive(ctx, directives, draft.findings)
                new_loudness = out.loudness if isinstance(out.loudness, Mapping) else None
                # Read the shot_timeline FRESH off the recompiled output — NOT the
                # handle-time value — or a re-driven artifact joins against the stale
                # original timeline (the result_ctx does not republish shot_timeline).
                new_st = out.shot_timeline
                new_shot_timeline = new_st if isinstance(new_st, list) else None
                redriven = await self._reevaluate(
                    new_ctx,
                    str(out.artifact_url),
                    as_float(out.duration_seconds),
                    new_loudness,
                    new_shot_timeline,
                )
                # Carry the ACTUALLY-EXECUTED set onto the new draft, cumulative across
                # passes. If this draft loses keep-highest, its rerun_stages loses with it
                # — the shipped artifact and the reported work stay in lockstep.
                return replace(
                    redriven,
                    rerun_stages=tuple(sorted(set(draft.rerun_stages) | set(executed))),
                )
            except Exception as exc:
                # TOLERABLE degrade: keep-highest is a genuine non-regression — the CURRENT
                # best artifact still ships and is still gated. What was masking is the log:
                # a director that raised looked identical to a pass that simply did not
                # improve. ERROR + traceback names it.
                log.error(
                    "final_critique: auto re-drive step failed, keeping best: %s",
                    exc,
                    exc_info=True,
                )
                redrive_failures.append(
                    _advisory(
                        "redrive",
                        f"auto re-drive step failed, kept the current best: "
                        f"{type(exc).__name__}: {exc}",
                    )
                )
                return draft

        # No id_extractor (loop_engine's C0 guard): each re-driven director re-runs
        # over the SAME spine and every merge preserves the id-set (only flagged rows
        # are replaced), so the shot-id set is invariant by construction.
        try:
            result = await run_evaluator_optimizer(
                initial,
                critic_fn=critic_fn,
                optimizer_fn=optimizer_fn,
                threshold=threshold,
                max_passes=max_passes,
                stage_name=self.name,
                timeout_seconds=self._llm_timeout,
                retries=self._llm_retries,
            )
        except ShipGateEnforcementError:
            # A terminal enforce failure raised from INSIDE the loop (an optimizer pass that
            # reached `_enforced_accept`) must not be recaptured as "loop error" and then
            # re-decided against `initial` — it is already the gate's verdict.
            raise
        except Exception as exc:
            # TOLERABLE degrade: the fall-back still goes through `_enforced_accept`, so an
            # ENFORCE run cannot pass on a ceilings-failing draft just because the loop
            # broke. ERROR + traceback so the broken loop is not silent.
            log.error(
                "final_critique: auto re-drive loop error, accepting current: %s",
                exc,
                exc_info=True,
            )
            loop_failure = _advisory(
                "redrive",
                f"auto re-drive loop failed, accepted the initial draft: "
                f"{type(exc).__name__}: {exc}",
            )
            return self._enforced_accept(
                ctx,
                _with_findings(initial, [*redrive_failures, loop_failure]),
                rerun_stages=[],
            )

        best = _with_findings(result.draft, redrive_failures)
        # rerun_stages is what `_redrive` ACTUALLY EXECUTED for the RETAINED draft —
        # recorded by the optimizer as the work happened, not re-derived afterwards from
        # the critic's directives. The old derivation (retained draft's redrivable
        # directives ∩ stages with a baseline output) reported work that never happened in
        # three measured cases: a director that raised (optimizer swallowed it and returned
        # the unchanged draft), every targeted id pinned away (`_redrive` `continue`s), and
        # pass 2 — where the directives name what should be fixed NEXT, not what was fixed.
        return self._enforced_accept(ctx, best, rerun_stages=list(best.rerun_stages))

    async def _redrive(
        self,
        ctx: RunContext,
        directives: Sequence[FinalCritiqueDirective],
        findings: Sequence[FinalCritiqueFinding],
    ) -> tuple[RunContext, DemoVideoCompileOutput, list[str]]:
        """Re-run the owning per-dimension LLM stage(s) named by ``directives``, then
        re-compile ONCE from the swapped ctx.

        For each redrivable stage (in the fixed ``footage → sound → voice`` order),
        the film-critic feedback is threaded into that stage's own LLM prompt via
        ``redrive_feedback``; the regenerated items are merged back onto the baseline
        by id (only the flagged ids). A ``voice_director`` re-drive re-synthesizes
        narration (``demo_video_narrate``) so compile reads the new clips — footage /
        sound feed compile directly. A stage whose baseline output is missing (the
        gate never ran) is DROPPED with an advisory log, never constructed. Then the
        CANONICAL ``DemoVideoCompileStage.handle`` runs once (title cards / mastering /
        captions ride identically) — never a DAG re-run / child run.

        Returns ``(result_ctx, compile_output, executed)``: a ctx carrying the FRESH
        compile output (new artifact_url / sha / loudness) so the follow-up
        ``_reevaluate`` analyzes the new file against its own sha, the compile output,
        and the sorted list of stages that were ACTUALLY RUN. ``executed`` is the ONLY
        honest source for the stage's ``rerun_stages`` — it excludes a stage dropped for
        a missing baseline and a stage whose every flagged id was pinned away, and it is
        never produced at all when a director raises (the caller keeps the prior
        draft's set)."""
        new_prior = dict(ctx.prior_run_outputs)

        # Defense-in-depth (A11 PR-4): even if a pinned id slipped through the directive
        # clamp, a pinned beat must NEVER be re-driven. Subtract it from every stage's
        # flagged set (mirrors retention_plan.py's `(flagged & spine) - locked` idiom).
        pinned_ids, _ = self._override_ids(ctx)

        # Group flagged spine ids + reasons per redrivable stage (union across the
        # stage's directives).
        flagged_by_stage: dict[str, set[str]] = {}
        reasons_by_stage: dict[str, list[str]] = {}
        for d in directives:
            if d.stage not in _LLM_REDRIVABLE:
                continue
            flagged_by_stage.setdefault(d.stage, set()).update(
                set(map(str, d.spine_ids)) - pinned_ids
            )
            if d.reason:
                reasons_by_stage.setdefault(d.stage, []).append(d.reason)

        voice_redriven = False
        executed: list[str] = []
        for stage in _REDRIVE_ORDER:
            flagged_set = flagged_by_stage.get(stage)
            if not flagged_set:
                continue
            baseline = ctx.prior_output(stage)
            if not baseline:
                # Gate off — the owning stage never ran (absent) or skip-cleaned (a
                # StageSkippedError stores a falsy ``{}``; dag_executor.py:137). A
                # re-drive would re-construct the stage only for it to skip-clean again
                # and abort the whole pass; drop it (advisory), never re-run.
                log.warning(
                    "final_critique: %s named for re-drive but it never ran — dropping", stage
                )
                continue
            flagged = frozenset(flagged_set)
            feedback = _redrive_feedback(flagged, reasons_by_stage.get(stage, []), findings)
            key, id_field = _REDRIVE_ADAPTER[stage]
            regenerated = await self._run_director(
                stage, feedback, replace(ctx, prior_run_outputs=new_prior)
            )
            new_rows = regenerated.get(key)
            old_rows = baseline.get(key)
            merged = _merge_by_id(
                old_rows if isinstance(old_rows, list) else [],
                new_rows if isinstance(new_rows, list) else [],
                id_field,
                flagged,
            )
            new_prior[stage] = {key: merged}
            executed.append(stage)  # the director RAN and its output was merged in
            if stage == "voice_director":
                voice_redriven = True

        # Voice changes the synthesized narration ⇒ re-run narrate on the swapped ctx
        # so compile reads the fresh clips. Footage / sound need no intermediate.
        if voice_redriven:
            nout = await DemoVideoNarrateStage(self._tts_client).handle(
                DemoVideoNarrateInput(), replace(ctx, prior_run_outputs=new_prior)
            )
            new_prior["demo_video_narrate"] = nout.model_dump()

        # The re-driven director outputs (and a re-synthesized narration) are the ONLY
        # thing that changed versus the run's original compile. `replace(ctx, ...)`
        # keeps run_id AND the stage name, so with the default empty discriminator this
        # re-compile dedupes onto that original compile: video_compile returns the
        # EXISTING job, the paid director + TTS work is discarded, and `_reevaluate`
        # then reads the ORIGINAL output_sha256 and concludes the re-drive did not help
        # (the note below anticipated a stale sha but assumed the compile was fresh).
        # Keyed on the re-drive DELTA, so an identical re-drive still dedupes to the
        # render already paid for, and an empty delta (every named stage dropped)
        # dedupes onto the original render.
        redrive_delta = {
            key: value
            for key, value in new_prior.items()
            if key not in ctx.prior_run_outputs or ctx.prior_run_outputs[key] != value
        }
        compile_ctx = replace(
            ctx,
            prior_run_outputs=new_prior,
            render_idempotency_discriminator=render_payload_discriminator("redrive", redrive_delta),
        )
        out = await DemoVideoCompileStage(client=self._compile_client).handle(
            DemoVideoCompileInput(), compile_ctx
        )
        # Publish the fresh compile output so `_reevaluate(result_ctx, …)` reads the
        # NEW sha via `_output_sha` (a stale sha would return the initial analysis
        # from the content-address cache and mask any real improvement).
        result_ctx = replace(
            compile_ctx,
            prior_run_outputs={
                **compile_ctx.prior_run_outputs,
                "demo_video_compile": {
                    **(ctx.prior_output("demo_video_compile") or {}),
                    "artifact_url": str(out.artifact_url),
                    "duration_seconds": as_float(out.duration_seconds),
                    "output_sha256": out.output_sha256,
                    "loudness": out.loudness,
                },
            },
        )
        return result_ctx, out, sorted(executed)

    async def _run_director(self, stage: str, feedback: str, ctx: RunContext) -> dict[str, object]:
        """Construct + run one owning director stage with ``redrive_feedback`` threaded
        into its LLM prompt, returning its ``model_dump()``. All three route their WM
        traffic through the shared ``WMClient`` (``self._wm``)."""
        if stage == "voice_director":
            vout = await VoiceDirectorStage(self._wm).handle(
                VoiceDirectorInput(redrive_feedback=feedback), ctx
            )
            return vout.model_dump()
        if stage == "demo_video_sound_director":
            sout = await DemoVideoSoundDirectorStage(self._wm).handle(
                SoundDirectorInput(redrive_feedback=feedback), ctx
            )
            return sout.model_dump()
        fout = await DemoVideoFootageDirectionStage(self._wm).handle(
            FootageDirectionInput(redrive_feedback=feedback), ctx
        )
        return fout.model_dump()

    def _loudness_findings(
        self, gestalt: Mapping[str, object], ctx: RunContext
    ) -> list[FinalCritiqueFinding]:
        """A deterministic advisory finding when the measured integrated LUFS is
        outside the ship window (``[]`` when in-window / unmeasured)."""
        loud = gestalt.get("loudness")
        if not isinstance(loud, Mapping):
            return []
        lufs = cast("Mapping[str, object]", loud).get("integrated_lufs")
        if not isinstance(lufs, (int, float)) or isinstance(lufs, bool):
            return []
        lo, hi = loudness_window(self._ceilings(ctx))
        if lo <= float(lufs) <= hi:
            return []
        return [
            FinalCritiqueFinding(
                dimension="loudness",
                severity="advisory",
                detail=f"integrated {float(lufs):.1f} LUFS outside [{lo:.1f}, {hi:.1f}]",
            )
        ]

    async def _analyze(self, artifact_url: str, sha: str) -> VideoAnalyzerJobResult:
        """Submit the artifact to video_analyzer (content-addressed by output sha)."""
        if sha and sha in self._result_cache:
            return self._result_cache[sha]
        job_id = await self._analyzer.start_analysis(
            video_url=artifact_url, config_overrides=_ANALYSIS_CONFIG
        )
        await self._analyzer.poll_analysis(job_id, timeout=_ANALYSIS_POLL_TIMEOUT_S)
        result = await self._analyzer.get_result(job_id)
        if sha:
            self._result_cache[sha] = result
        return result

    def _cut_spine(
        self,
        ctx: RunContext,
        result: VideoAnalyzerJobResult,
        duration: float,
        spine_ids: Sequence[str],
        shot_timeline: list[dict[str, object]] | None,
    ) -> _CutSpine:
        """Build THE cut spine — the single set of ``(spine_id, cut_time)`` pairs every
        gestalt term (continuity, cohesion, tonal, momentum) is measured against.

        CUT TIMES come from the compiler's AUTHORITATIVE ``shot_timeline`` when it is
        present, falling back to the analyzer's ``scene_boundaries`` only when it is
        absent / empty / malformed. The compiler knows exactly where every shot starts
        because it placed them; the analyzer's scene detector infers cuts from pixels and
        returned ONE boundary for a 29.8s multi-shot film in prod, collapsing ``frames``
        to a single entry.

        ⚠️ **A CORRECT SPINE IS NOT A MEASURABLE GATE.** Do not read this as "the visual
        terms now work". It fixes the SAMPLING — which shots are looked at, and which beat
        each cut belongs to — and nothing else. The analyzer's per-frame records carry only
        ``{index, timestamp, path, description, ocr_text, labels}``; there is NO per-frame
        ``brightness`` / ``sharpness`` / ``dominant_color``, so ``_feature_vector`` fills
        five of six components from whole-video constants and only ``motion`` can differ
        between shots. **Continuity, cohesion and tonal drift remain structurally
        unmeasurable on the data the analyzer currently emits**, and on a near-static render
        (measured ``duplicate_frame_ratio`` 0.9754) even motion is flat. Sampling more
        shots off a correct spine therefore produces MORE identical vectors, not more
        signal — which is exactly why ``gestalt_summary`` now detects zero dispersion and
        reports ``insufficient_evidence`` instead of publishing the extra zeros. Closing
        the gap for real needs per-frame visual features from ``video_analyzer_service``;
        that is upstream work, not something this stage can fix.

        The cut→beat JOIN is time-containment against those same intervals (or, on the
        fallback path, ``build_beat_intervals``' intended durations) — never the legacy
        positional zip, which truncated to ``min(len(cuts), len(spine_ids))`` (8 beats vs
        12 cuts ⇒ the whole tail of the film never sampled) and mis-attributed every seam's
        ``from_id`` / ``to_id``.

        SOFT: any failure yields an empty spine, which surfaces honestly as
        ``insufficient_evidence`` rather than as a silently flattering verdict.

        The two derivations are guarded SEPARATELY on purpose. ``build_beat_intervals``
        now RAISES on an unparseable ``seconds`` (loud beats silent), but that is a
        MOMENTUM-only input: one malformed beat duration must cost the momentum term, not
        the continuity / cohesion / tonal evidence that the rendered timeline can still
        supply on its own."""
        lines = resolve_plan_lines(ctx)
        intended_broken = False
        try:
            intended = build_beat_intervals(lines) or []
        except Exception:  # a bad duration costs momentum, not the gestalt
            log.warning(
                "final_critique: intended beat spine unusable; momentum will not be measured",
                exc_info=True,
            )
            intended, intended_broken = [], True
        try:
            # Prefer the compiler's rendered timeline; None ⇒ intended-duration fallback.
            rendered = _to_intervals(shot_timeline, spine_ids)
            beat_intervals = rendered or (intended or None)
            cut_times: Sequence[float] | None = (
                [start for _sid, start, _end in rendered] if rendered else result.scene_boundaries
            )
            cuts = sample_frames_at_cuts(
                cut_times, duration, spine_ids, beat_intervals=beat_intervals
            )
        except Exception:  # SOFT: a spine defect degrades, never raises
            log.warning("final_critique: cut-spine derivation failed; no evidence", exc_info=True)
            return _CutSpine(
                cuts=[], intended_intervals=[], lines=lines, intended_broken=intended_broken
            )
        return _CutSpine(
            cuts=cuts, intended_intervals=intended, lines=lines, intended_broken=intended_broken
        )

    def _build_frames(
        self, result: VideoAnalyzerJobResult, cuts: Sequence[tuple[str, float]]
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        """Map whole-MP4 signals → per-shot ``frames`` (feature vectors) + ``tone_shots``
        (per-shot text) for the gestalt fuse, over the shared spine from
        :meth:`_cut_spine`."""
        metrics = result.quality_metrics
        brightness = as_float(metrics.mean_brightness if metrics else None)
        sharpness = as_float(metrics.mean_sharpness if metrics else None)
        motion_timeline = metrics.motion_timeline if metrics else None
        # Per-frame visual curve (analyzer PR #73). Present ⇒ each shot gets its OWN
        # brightness / sharpness / dominant colour, so continuity + cohesion actually
        # vary between shots. Empty (older analyzer / older job) ⇒ fall back to the
        # whole-video means, which are identical for every shot — gestalt reads that as
        # zero-dispersion / insufficient_evidence, never a flattering constant.
        visual_timeline = metrics.visual_timeline if metrics else None
        raw_frames = [fa for fa in (result.frame_analyses or []) if isinstance(fa, Mapping)]

        frames: list[dict[str, object]] = []
        tone_shots: list[dict[str, object]] = []
        for shot_id, t in cuts:
            fa = _nearest_frame(raw_frames, t)
            motion = _motion_at(motion_timeline, t)
            vs = _nearest_visual(visual_timeline, t)
            if vs is not None:
                shot_brightness, shot_sharpness = vs.brightness, vs.sharpness
                shot_rgb: Sequence[float] = [float(c) for c in vs.dominant_rgb]
            else:  # back-compat: whole-video means (constant across shots)
                shot_brightness, shot_sharpness, shot_rgb = brightness, sharpness, ()
            frames.append(
                {
                    "shot_id": shot_id,
                    "features": _feature_vector(shot_brightness, shot_sharpness, shot_rgb, motion),
                }
            )
            text = fa.get("description") or fa.get("ocr_text") or fa.get("text") or ""
            tone_shots.append({"shot_id": shot_id, "description": str(text)})
        return frames, tone_shots

    async def _critic(
        self, artifact_url: str, gestalt: Mapping[str, object], spine_ids: Sequence[str]
    ) -> Mapping[str, object]:
        """Ask the film-critic WM worker for a whole-film verdict (routed through the
        shared LLM budget + guarded upstream classifier)."""
        message = _build_message(artifact_url, gestalt, spine_ids)

        async def _attempt(enable_reasoning: bool | None) -> Mapping[str, object]:
            resp = await self._wm.message(
                enable_reasoning=enable_reasoning,
                worker_name=self.worker_name,
                message=message,
                system_prompt=_SYSTEM_PROMPT,
                output_schema=_OUTPUT_SCHEMA,
                max_tokens=_CRITIC_MAX_TOKENS,
            )
            data = resp.output
            if not isinstance(data, Mapping):
                # ValueError is in LLM_RETRY_ON, so this re-prompts for a conforming
                # object rather than failing on the first non-object draw.
                raise ValueError("final_critique: WM output is not an object")
            return data

        return await stage_call(
            downshift_reasoning(lambda think: guarded(lambda: _attempt(think))),
            timeout_seconds=self._llm_timeout,
            retries=self._llm_retries,
            retry_on=LLM_RETRY_ON,
            stage_name="final_critique",
        )

    def _accept_draft(
        self,
        ctx: RunContext,
        draft: _Draft,
        *,
        rerun_stages: list[str],
        verdicts: Sequence[ShipGateTermVerdict] | None = None,
    ) -> FinalCritiqueOutput:
        """Unpack a retained ``_Draft`` into the stage output with the given
        ``rerun_stages`` (``[]`` for critique_only / no in-stage re-drive).

        ``verdicts`` are the per-term ship-gate rows already computed by
        :meth:`_enforced_accept`, passed down rather than recomputed so the persisted
        record is byte-identical to the rows the decision was taken on. ``None`` only on
        the intra-loop bookkeeping calls, which are not an accept."""
        return self._accept(
            ctx,
            draft.artifact_url,
            draft.duration,
            gestalt=draft.gestalt,
            scores=draft.scores,
            findings=draft.findings,
            directives=draft.directives,
            scene_boundaries=draft.scene_boundaries,
            rerun_stages=rerun_stages,
            verdicts=verdicts,
            analyzer_error=draft.analyzer_error,
        )

    def _accept(
        self,
        ctx: RunContext,
        artifact_url: str,
        duration: float,
        *,
        gestalt: Mapping[str, object],
        scores: FinalCritiqueScores | None,
        findings: list[FinalCritiqueFinding],
        directives: list[FinalCritiqueDirective],
        scene_boundaries: Sequence[float] | None,
        rerun_stages: list[str] | None = None,
        verdicts: Sequence[ShipGateTermVerdict] | None = None,
        analyzer_error: str | None = None,
    ) -> FinalCritiqueOutput:
        """Build the accept output. ``rerun_stages`` records which LLM-redrivable
        owning stages were ACTUALLY re-run for the retained draft (``[]`` for
        critique_only / off / the soft-degrade paths — byte-identical to PR-1).

        THE DURATION OVERRIDE. ``ship_ready`` arrives here as the ceilings verdict
        computed by ``gestalt_summary`` — continuity, loudness, momentum. It has never
        known the ordered length, so a master delivered at 155% of its order was badged
        ``ship-ready`` with ``ship_ready: true`` and not one duration finding. A film
        nobody ordered the length of is a different product from the one that was paid
        for, so the ordered-vs-delivered verdict is applied here, AFTER the ceilings
        verdict and deterministically (arithmetic, not the film-critic LLM, which is
        never asked about duration at all): out of tolerance ⇒ never ``ship-ready``.

        Measured against the artifact THIS STAGE ACCEPTED, not encode_master's — under
        ``auto`` the re-drive can retain a different, differently-long master, and the
        badge must describe what ships.
        """
        gestalt, duration_findings = _duration_override(ctx, gestalt, duration, stage=self.name)
        gestalt, footage_findings = _footage_bar_override(ctx, gestalt)
        findings = [*findings, *duration_findings, *footage_findings]
        ship_ready = bool(gestalt.get("ship_ready")) if gestalt else False
        overall = scores.overall if scores is not None else None
        rows = list(verdicts or [])
        decision = ship_gate_decision(rows) if rows else GATE_INSUFFICIENT_EVIDENCE
        return FinalCritiqueOutput(
            artifact_url=artifact_url,
            duration_seconds=duration,
            scores=scores,
            findings=findings,
            directives=directives,
            gestalt_summary=dict(gestalt),
            gestalt_unavailable_reason=_unavailable_reason(gestalt, findings),
            # The DIAGNOSIS. Every term, its three-state verdict, its measured value,
            # the limit it was compared against, its tier, and whether it was armed —
            # so "the gate failed" is always answerable as "which ceiling, at what
            # value, and was it even blocking". A run that never reached the gate
            # (`verdicts is None`, the soft-degrade paths) records
            # `insufficient_evidence` with no terms, which is the truth about it.
            ship_gate=ship_gate_record(rows, decision),
            quality_badge=_badge(ship_ready, overall),
            chapters=_chapters(scene_boundaries, duration),
            rerun_stages=rerun_stages or [],
            # The deliberate fail-open made VISIBLE: the film ships, but the record says
            # it shipped UNREVIEWED (degraded run, warnings panel, queryable code).
            degradations=(
                [
                    degradation(
                        REVIEW_SKIPPED,
                        "the video analyzer returned no reading, so the final review did not "
                        f"run and the film shipped unreviewed — {analyzer_error}",
                    )
                ]
                if analyzer_error
                else []
            ),
        )


# ── model-output interpretation (pure, never raise) ──────────────────────────
def _static_input_findings(static_ratio: float | None) -> list[FinalCritiqueFinding]:
    """An advisory finding when the compiler measured a near-static render.

    The compiler already computes ``duplicate_frame_ratio`` on the real file; the ship
    gate simply never read it. A ~97.5%-duplicate render is degenerate INPUT — it is why
    the visual terms come back unmeasured — and naming that is far more useful to a reader
    than an unexplained wall of nulls. Advisory only: it never gates."""
    if static_ratio is None or static_ratio < _STATIC_FRAME_RATIO_ADVISORY:
        return []
    return [
        FinalCritiqueFinding(
            dimension="static_input",
            severity="advisory",
            detail=(
                f"{static_ratio * 100:.1f}% of rendered frames are duplicates — the film is "
                "essentially static, so the visual gestalt terms have nothing to measure"
            ),
        )
    ]


# Dimensions used by the soft-degrade advisories that can leave the gestalt empty:
# `_advisory("analyzer", …)` (analyzer call failed/timed out), `_advisory("stage", …)`
# (handle()'s outer catch-all) and `_advisory("film_critic", …)` (the WM critic errored).
# Anything else is a content finding about a film that WAS measured.
_SIGNAL_SOURCE_DIMENSIONS: frozenset[str] = frozenset({"analyzer", "stage", "film_critic"})


def _usable_gestalt_terms(gestalt: Mapping[str, object]) -> list[str]:
    """The gestalt's calibration-bearing numeric terms that actually resolved.

    These are exactly the three terms the nightly calibration corpus and
    ``_emit_gestalt_metrics`` consume. ``continuity_stats.n`` is deliberately NOT one of
    them: a degenerate distribution still carries ``{"n": <pairs>, "degenerate": True}``
    — the numeric shadow of a distribution with no ``min``/``max``/percentiles — so
    counting ``n`` as signal would read the very artefact the degenerate guard exists to
    make visible as evidence.
    """
    stats = gestalt.get("continuity_stats")
    continuity_max = (
        opt_float(cast("Mapping[str, object]", stats).get("max"))
        if isinstance(stats, Mapping)
        else None
    )
    candidates = {
        "cohesion": opt_float(gestalt.get("cohesion")),
        "momentum_gap": opt_float(gestalt.get("momentum_gap")),
        "continuity_stats.max": continuity_max,
    }
    return sorted(name for name, value in candidates.items() if value is not None)


def _unavailable_reason(
    gestalt: Mapping[str, object], findings: Sequence[FinalCritiqueFinding]
) -> str | None:
    """Why the gestalt carries no usable signal, taken from the advisory that explains it.

    Derived rather than threaded: every soft-degrade branch already records its cause as
    an advisory finding, so reading them back here covers all of them — including any
    added later — instead of relying on each new `except` to remember a new argument.

    ⚠️ ABSENT AND VACUOUS ARE THE SAME DEFECT. A gestalt dict can be TRUTHY and still
    contribute nothing: measured live, 5 completed rows (2026-08-28…31) carried a full
    dict whose ``cohesion`` and ``momentum_gap`` were both NULL and whose
    ``continuity_stats`` held no distribution, alongside ``insufficient_evidence: true``
    and a 0.64-0.91 ``static_frame_ratio``. A bare ``if gestalt: return None`` called
    those AVAILABLE, so no reason was recorded and every one of them was counted as a
    usable calibration sample. The discriminator is therefore the SUBSTANCE — are any of
    the calibration-bearing numeric terms resolved (see :func:`_usable_gestalt_terms`) —
    not the presence of the dict and not the ``insufficient_evidence`` flag, which is read
    only as corroboration so a future producer that forgets to set it is still caught.

    SOFT BY DESIGN: this only makes an unusable sample say so. ``gestalt_summary`` stays
    out of ``produces``, so a reason here never fails the stage.

    Returns ``None`` only when the gestalt carries real measured signal, so a populated
    summary is never annotated with a stale reason.
    """
    if _usable_gestalt_terms(gestalt):
        return None
    if gestalt:
        # Present but vacuous: name the shape so the row explains itself without a join.
        detail = (
            "gestalt was computed but carries no usable signal — cohesion, momentum_gap "
            "and continuity_stats.max all resolved no numeric value"
        )
        if bool(gestalt.get("insufficient_evidence")):
            detail += "; insufficient_evidence=true"
        static_ratio = opt_float(gestalt.get("static_frame_ratio"))
        if static_ratio is not None:
            detail += f"; static_frame_ratio={static_ratio:.2f}"
        return detail
    for finding in findings:
        if finding.dimension in _SIGNAL_SOURCE_DIMENSIONS:
            return finding.detail
    # Empty with no advisory to explain it is itself worth saying out loud: silence here
    # is exactly the dead signal this field exists to remove.
    return "gestalt was not computed and no advisory recorded why"


# The sentinel strings an upstream stage stringifies a MISSING artifact into. They are
# truthy, so a bare `if not url` accepted them and the gate happily "evaluated" a film
# that does not exist — the exact shape that lets a run reach terminal status with no
# encoded master. Treated as absent so the chain falls through to the next producer and,
# failing that, raises StagePreconditionError.
_ABSENT_URL_SENTINELS: frozenset[str] = frozenset({"none", "null", "nil", "undefined", ""})


def _usable_artifact_url(raw: object) -> str:
    """A non-empty, non-sentinel artifact URL string — else ``""`` (⇒ treated as absent)."""
    if not isinstance(raw, str):
        return ""
    url = raw.strip()
    return "" if url.lower() in _ABSENT_URL_SENTINELS else url


def _with_findings(draft: _Draft, extra: Sequence[FinalCritiqueFinding]) -> _Draft:
    """``draft`` with ``extra`` findings appended (the same object when there are none)."""
    return replace(draft, findings=[*draft.findings, *extra]) if extra else draft


def _advisory(dimension: str, detail: str) -> FinalCritiqueFinding:
    """A soft-degrade finding surfaced when a signal source was unavailable."""
    return FinalCritiqueFinding(
        dimension=dimension, severity="advisory", seam=None, detail=detail[:_MAX_TEXT]
    )


_DIMENSIONS: tuple[str, ...] = (
    "cohesion",
    "continuity",
    "momentum_experienced",
    "tonal_consistency",
)


def _interpret_scores(
    data: Mapping[str, object],
) -> tuple[FinalCritiqueScores | None, list[FinalCritiqueFinding]]:
    """Clamp the four 1-5 dimensions + overall, WITHOUT fabricating anything.

    Returns ``(scores, advisory_findings)``. A dimension the reply did not carry
    (missing / non-numeric / non-finite) persists as ``None`` rather than the old
    neutral ``3``, and every such gap raises an advisory finding naming the missing
    dimensions — so a reader can tell "the critic said average" from "the critic said
    nothing". A reply with NO usable dimension and NO usable overall yields
    ``scores=None`` (there is no verdict to persist) plus the advisory; the badge then
    degrades to ``needs-review`` instead of inheriting a made-up mean.

    A missing ``overall`` still falls back to the mean — but only of the dimensions that
    were ACTUALLY present."""
    dims: dict[str, int | None] = {name: _clamp_dim(data.get(name)) for name in _DIMENSIONS}
    present = [v for v in dims.values() if v is not None]
    missing = [name for name, v in dims.items() if v is None]

    findings: list[FinalCritiqueFinding] = []
    if missing:
        findings.append(
            _advisory(
                "film_critic",
                "critic reply carried no usable score for: " + ", ".join(missing),
            )
        )

    fallback = (sum(present) / len(present)) if present else None
    overall = _clamp_overall(data.get("overall"), fallback)
    if overall is None:
        # Nothing usable at all — do NOT invent a verdict out of four absent numbers.
        return None, findings
    return (
        FinalCritiqueScores(
            cohesion=dims["cohesion"],
            continuity=dims["continuity"],
            momentum_experienced=dims["momentum_experienced"],
            tonal_consistency=dims["tonal_consistency"],
            overall=overall,
        ),
        findings,
    )


def _interpret_findings(
    data: Mapping[str, object], exempt_ids: frozenset[str] = frozenset()
) -> list[FinalCritiqueFinding]:
    """Interpret the film-critic's findings, dropping any CONTINUITY finding whose
    seam touches an author-``continuity_exempt`` beat (A11 PR-4). The author override
    wins over the critic's computed finding; only continuity is suppressed (a cohesion
    / tonal finding on the same beat survives). ``exempt_ids`` empty ⇒ byte-identical."""
    out: list[FinalCritiqueFinding] = []
    for raw in _seq(data.get("findings")):
        if not isinstance(raw, Mapping):
            continue
        row = cast("Mapping[str, object]", raw)
        seam = None
        raw_seam = row.get("seam")
        if isinstance(raw_seam, Mapping):
            sm = cast("Mapping[str, object]", raw_seam)
            if sm.get("from_id") and sm.get("to_id"):
                seam = Seam(from_id=str(sm["from_id"]), to_id=str(sm["to_id"]))
        finding = FinalCritiqueFinding(
            dimension=str(row.get("dimension") or "")[:_MAX_TEXT],
            severity=str(row.get("severity") or "")[:_MAX_TEXT],
            seam=seam,
            # The WHY, threaded verbatim into a re-driven director's prompt by
            # `_redrive_feedback`. Never populated before this change (it was absent from
            # `_OUTPUT_SCHEMA` too), so every re-drive bullet was empty.
            detail=str(row.get("detail") or "")[:_MAX_TEXT],
        )
        if (
            finding.dimension == "continuity"
            and finding.seam is not None
            and (finding.seam.from_id in exempt_ids or finding.seam.to_id in exempt_ids)
        ):
            continue
        out.append(finding)
    return out


def _interpret_directives(
    data: Mapping[str, object],
    spine_ids: Sequence[str],
    pinned_ids: frozenset[str] = frozenset(),
) -> list[FinalCritiqueDirective]:
    """Clamp every directive: ``stage`` to the real blueprint stage-name set and
    ``spine_ids`` to the run's real ids, EXCLUDING author-``pin`` beats (A11 PR-4) —
    a pinned beat is never a re-drive target. Drop a directive whose stage is unknown
    OR whose clamped spine_ids is empty (a directive emptied entirely by pinning is
    dropped here). Persisted advisory-only — never a re-drive. ``pinned_ids`` empty ⇒
    byte-identical."""
    run_ids = set(spine_ids)
    out: list[FinalCritiqueDirective] = []
    for raw in _seq(data.get("directives")):
        if not isinstance(raw, Mapping):
            continue
        row = cast("Mapping[str, object]", raw)
        stage = str(row.get("stage") or "")
        if stage not in _VALID_STAGES:
            continue
        ids = [
            str(i)
            for i in _seq(row.get("spine_ids"))
            if str(i) in run_ids and str(i) not in pinned_ids
        ]
        if not ids:
            continue
        out.append(
            FinalCritiqueDirective(
                stage=stage, spine_ids=ids, reason=str(row.get("reason") or "")[:_MAX_TEXT]
            )
        )
    return out


def _build_message(
    artifact_url: str, gestalt: Mapping[str, object], spine_ids: Sequence[str]
) -> str:
    return (
        "Whole-film gestalt signals measured off the finished video:\n"
        f"{json.dumps(dict(gestalt), ensure_ascii=False, sort_keys=True, default=str)}\n\n"
        f"Ordered shot ids: {list(spine_ids)}\n"
        f"Delivered artifact: {artifact_url}\n"
    )
