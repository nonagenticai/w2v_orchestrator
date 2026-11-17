"""demo_video_encode_master — the encode/mux mastering evaluator-optimizer (spec 13 §5).

A bounded, OPT-IN, SOFT-gate stage that scores a rendered video's DETERMINISTIC
``MasteringReport`` (the post-mux loudness/limiter/colour/framerate report the
video compiler already produces), asks an ``encode-critic`` WM worker to grade it
and propose a mastering delta, has a ``mux-mastering-advisor`` WM worker turn that
delta into a concrete mastering block, CLAMPS every knob to its §5 range, and —
strictly GUARDED — re-compiles ONCE through the canonical compile path when a gate
genuinely failed. It is DISABLED-BY-DEFAULT: without a ``mastering.critic`` config
block it runs the shared ``loop_engine`` with ``critic_fn=None`` (zero awaits, zero
WM traffic) and emits the UNCHANGED compile artifact — byte-identical to a run that
never had this stage.

Structurally a twin of ``demo_video/footage_critique.py`` + ``final_critique.py``:
the WM calls self-catch (any worker/compile error degrades to accept-current), the
whole loop is wrapped belt-and-suspenders, and the retained best artifact is carried
downstream. The mastering report is primarily a whole-file aggregate, but the compiler
now also emits an OPTIONAL per-shot banding channel
(``per_shot: [{line_id, banding, needs_deband}]``); the critic reads it when present
and falls back to the aggregate ``banding_scores`` when it is absent (older builds).

Authoritative-out invariant: the models are advisory. Every numeric knob the advisor
proposes is clamped at the boundary (:func:`_clamp_mastering_block`) and an author who
pinned a mastering knob ALWAYS wins over the advisor (only knobs the author left unset
are filled). ``crf`` reaches the encode alongside loudness/deband/maxrate/colorspace:
``resolution_tags`` (``primitives/compile_stage.py``) OMITS the per-tier ``crf``
``compile_tag`` whenever a mastering ``crf`` is set, so the mastering-block value (the
clamped advisor delta, or an author-pinned crf) is what the compiler encodes.

LLM calls route through ``worker_management`` (the ``WMClient`` gateway) — never a
provider directly; the model behind each worker is WM config.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar, cast

import httpx
from w2v_common.stages import StageRetriesExhaustedError, stage_call

from src.clients.video_compile import VideoCompileClient
from src.config import get_settings
from src.services.llm import WMClient, WMResponse
from src.services.runs.structural_relief_policy import (
    SIZING_BASIS_PHRASES,
    relief_shape_from_manifest,
)
from src.services.stages.base import (
    SLOT_USE_OPTIONAL,
    ArtifactUrl,
    SlotUse,
    StageHandler,
    StageInput,
    StageOutputWithNotes,
    StagePreconditionError,
)
from src.services.stages.demo_video.compile import DemoVideoCompileInput, DemoVideoCompileStage
from src.services.stages.llm_budget import (
    MEDIUM_ANSWER_TOKENS,
    SMALL_ANSWER_TOKENS,
    attempt_plan,
    downshift_reasoning,
    llm_budget_seconds,
    llm_client_timeout_seconds,
    llm_retries,
    output_token_budget,
)
from src.services.stages.primitives.compile_stage import render_payload_discriminator
from src.services.stages.shared.degradation import (
    DURATION_OUT_OF_TOLERANCE,
    STRUCTURAL_RELIEF_REFUSED,
    Degradation,
    degradation,
)
from src.services.stages.shared.duration_conformance import (
    DurationConformance,
    demo_video_ordered_seconds,
    duration_conformance_advisories,
    duration_conformance_degradations,
    observe_duration_conformance,
    picture_truncation_degradations,
    plan_infeasible_reason,
    playable_seconds_from_mastering,
)
from src.services.stages.shared.loop_engine import Critique, run_evaluator_optimizer
from src.services.stages.shared.mastering_gate import remediable_by_remaster
from src.services.stages.shared.mastering_plan import ship_loudness_window
from src.services.stages.shared.provenance import clamp
from src.services.stages.upstream import (
    PermanentUpstreamError,
    TransientUpstreamError,
    guarded,
)
from src.utils import as_float, describe_exception

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

log = logging.getLogger(__name__)

# WM worker names (regex ^[a-zA-Z0-9_-]+$). WM auto-creates each on first call and
# the per-call system_prompt steers it; provider/model live on the WM worker config.
_CRITIC_WORKER = "encode-critic"
_ADVISOR_WORKER = "mux-mastering-advisor"
_COMPUTED_BY = "mux-mastering-advisor"

# Convergence band (1-5, sibling convention). A run may override the threshold /
# pass budget via `mastering.critic.score_threshold` / `.iterations`.
_SCORE_MIN = 1.0
_SCORE_MAX = 5.0
_THRESHOLD = 4.0
_MAX_PASSES = 2

# §5 clamp ranges. crf is the tighter mastering band (the compiler's own crf range is
# wider — this is the mastering-advisor band).
#
# ⭐ The advisor's legal loudness range is DERIVED FROM THE SHIP WINDOW, which is itself
# derived from the target the compiler actually applies (`shared/loudness_contract.py`).
# Nothing here restates a number. It used to be the literal `-16.0, -13.0` against a gate
# band of (-14.0, -11.0): the advisor was therefore PERMITTED to clamp `loudness_target`
# anywhere down to -16.0, i.e. 2 LU BELOW the floor the gate convicts at, without ever
# misbehaving. Two constants that had to agree, in two files, with nothing holding them
# together. Deriving makes "the encoder may aim where the gate convicts" structurally
# unreachable, and `tests/.../test_loudness_contract.py` fails if it ever stops being.
_LOUDNESS_LO, _LOUDNESS_HI = ship_loudness_window()
_CRF_LO, _CRF_HI = 16, 24
_DEBAND_VALUES = frozenset({"auto", "on", "off"})
_DEBAND_DEFAULT = "auto"

# The clampable/threadable mastering knobs the advisor may set — an authored value
# for any of these WINS over the advisor's proposal.
_KNOBS = ("loudness_target", "crf", "deband", "maxrate")

# The aggregate MasteringReport fields summarised into each worker prompt. The report
# is an untyped JSON blob from the compiler; a missing key simply summarises as null.
_REPORT_KEYS = (
    "colorspace_ok",
    "integrated_lufs",
    "true_peak_dbtp",
    "banding_scores",
    "cfr_ok",
    "duplicate_frame_ratio",
    "gate_pass",
    "rerun_stages",
    # OPTIONAL per-shot banding channel emitted by newer compiler builds; surfaces as
    # None on older reports that omit it (defensive by construction — see _report_summary).
    "per_shot",
)

_CRITIC_SYSTEM_PROMPT = (
    "You are an encode / mux mastering critic for a finished short demo video. You "
    "receive the DETERMINISTIC mastering report measured off the actual rendered file "
    "(colour-space correctness, integrated loudness in LUFS, true-peak in dBTP, banding "
    "scores, constant-frame-rate check, duplicate-frame ratio, and whether the "
    "deterministic mastering gate passed) plus the delivery intent. When the report "
    "carries a per_shot array, each entry gives one shot's banding (line_id, banding "
    "score, needs_deband) — weigh those shots, otherwise reason from the aggregate "
    "banding_scores. Grade the master "
    "1-5 overall and flag each dimension (critic_colorspace_ok, banding_ok, "
    "loudness_ok, framerate_ok — critic_colorspace_ok is YOUR judgement, distinct "
    "from the report's own measured colorspace_ok above). When a knob is off, "
    "propose a CONSERVATIVE profile_delta "
    "(loudness_target, crf, deband, maxrate) — small nudges only, never a rewrite. You "
    "are advisory: the pipeline clamps every value and an author's pinned knob always "
    "wins. Return only JSON matching the schema."
)

_ADVISOR_SYSTEM_PROMPT = (
    "You are a mux mastering advisor. Given the encode-critic's proposed mastering "
    "delta and the measured mastering report, return a CONCRETE, conservative mastering "
    "block (loudness_target in LUFS, crf, deband one of auto/on/off, maxrate as a "
    "string like '12M'). Prefer the smallest change that fixes the flagged dimension; "
    "every value is clamped downstream so stay inside sane broadcast ranges. Return "
    "only JSON matching the schema."
)

# Object-rooted structured-output schemas (WM convention). ``per_shot`` is OPTIONAL
# (only ``overall_score`` is required) so a critic that omits it still validates — it
# mirrors the compiler's per-shot banding channel when the model chose to reason per shot.
#
# ``critic_colorspace_ok`` is deliberately NOT named ``colorspace_ok``: that name is
# already taken by ``_REPORT_KEYS`` (the compiler's MEASURED value, fed into this same
# prompt via ``_report_summary``) — same name, two producers (a deterministic ffprobe
# check vs. an LLM's opinion), two meanings, one prompt. Prior to this rename nothing in
# this module actually READ the critic's boolean (see ``critic_fn`` — only
# ``overall_score`` / ``profile_delta`` / ``rerun_stages`` are extracted from ``data``),
# so this is wiring-only: the field stays unread, just no longer name-collided.
_CRITIC_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "required": ["overall_score"],
    "properties": {
        "overall_score": {"type": "number"},
        "critic_colorspace_ok": {"type": "boolean"},
        "banding_ok": {"type": "boolean"},
        "loudness_ok": {"type": "boolean"},
        "framerate_ok": {"type": "boolean"},
        "profile_delta": {
            "type": "object",
            "properties": {
                "loudness_target": {"type": "number"},
                "crf": {"type": "integer"},
                "deband": {"type": "string"},
                "maxrate": {"type": "string"},
            },
        },
        "per_shot": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "line_id": {"type": "string"},
                    "banding": {"type": "number"},
                    "needs_deband": {"type": "boolean"},
                },
            },
        },
        "rerun_stages": {"type": "array", "items": {"type": "string"}},
    },
}

_ADVISOR_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "loudness_target": {"type": "number"},
        "crf": {"type": "integer"},
        "deband": {"type": "string"},
        "maxrate": {"type": "string"},
    },
}


# MEDIUM answer: ``_CRITIC_OUTPUT_SCHEMA`` is one verdict object (a score, four
# booleans, a 4-field ``profile_delta``) plus an OPTIONAL ``per_shot`` array of
# 3-field rows, one per banded shot — a modest list of tiny objects, well short of a
# per-line rewrite.
_CRITIC_MAX_TOKENS = output_token_budget(answer_tokens=MEDIUM_ANSWER_TOKENS)

# SMALL answer: ``_ADVISOR_OUTPUT_SCHEMA`` is a single flat object of FOUR scalars
# (loudness_target / crf / deband / maxrate) with no array anywhere in it — the
# canonical "one small fixed-shape reply" case.
_ADVISOR_MAX_TOKENS = output_token_budget(answer_tokens=SMALL_ANSWER_TOKENS)


@dataclass(frozen=True)
class _EncodeDraft:
    """One point in the loop: a compiled artifact + the mastering that produced it.

    ``mastering_block`` is the effective mastering config the compiler consumed for
    this artifact; ``report`` is the deterministic MasteringReport measured off it.
    The loop keeps the highest-scoring draft; the clamped recommendation + provenance
    for the retained draft are looked up out of a closure map keyed by ``artifact_url``
    (mirrors ``footage_critique``'s ``records_by_url``) rather than threaded on the
    frozen draft, so an advisory (guard-blocked) pass on the SAME url still surfaces it.
    """

    artifact_url: str
    duration_seconds: float
    mastering_block: dict[str, object]
    report: dict[str, object]


class EncodeMasterInput(StageInput):
    """No typed inputs — everything is read from ``ctx`` (prior output + config)."""


class EncodeMasterOutput(StageOutputWithNotes):
    """Best mastered artifact seen + the fresh report + the encode-critic verdict.

    ``artifact_url`` is ALWAYS a usable artifact: the UNCHANGED compile/footage-critique
    artifact when the stage is disabled or soft-degrades, or the re-compiled artifact
    when a guarded re-master fired. ``final_critique`` + ``render_manifest`` prefer this
    stage's ``artifact_url`` (falling back to footage_critique / compile), so an accepted
    re-master reaches the deliverable/provenance ledger. ``encode_critic`` carries
    ``{score, profile_recommendation, provenance, rerun_stages}`` (``None`` when disabled).
    """

    # Declared output contract — enforced centrally in
    # `runs/dag_executor.py::_enforce_produces`. `produces_required` because this
    # artifact IS what final_critique and render_manifest ship: the docstring above
    # is the invariant ("ALWAYS a usable artifact" — disabled and soft-degrade paths
    # both pass the upstream artifact through, and a missing upstream raises
    # StagePreconditionError before here). An empty one therefore means the
    # pass-through itself broke, and shipping a manifest that points at nothing is
    # never the better outcome. `mastering` / `encode_critic` are None on the
    # disabled default path and `duration_seconds` is telemetry — not declarable.
    produces: ClassVar[tuple[str, ...]] = ("artifact_url",)
    produces_required: ClassVar[bool] = True

    artifact_url: ArtifactUrl
    duration_seconds: float
    mastering: dict[str, object] | None = None
    encode_critic: dict[str, object] | None = None
    #: Delivered-vs-ORDERED length of the master this stage ships. RECORDING, not
    #: enforcement: nothing here truncates a master or fails the stage. ``None`` when
    #: no usable duration was ordered — which is NOT the same claim as "on target".
    duration_conformance: DurationConformance | None = None


def _as_mapping(value: object) -> Mapping[str, object]:
    """Narrow an untyped JSON value to a ``Mapping[str, object]`` (``{}`` otherwise)."""
    if isinstance(value, Mapping):
        return cast("Mapping[str, object]", value)
    return {}


def _relief_refusal_degradation(
    prior_output: Mapping[str, object] | None, *, delivered_seconds: float | None = None
) -> Degradation | None:
    """The structural-relief gate's STOP verdict, on the QUERYABLE channel.

    A PURE READER of a verdict that already exists. It decides nothing: no band
    arithmetic, no beat counting, no spawn. :func:`relief_shape_from_manifest` is the
    same entry point ``reconcile_timeline._cant_converge_note`` and the post-completion
    spawner (``runs/structural_relief.py``) both call, on the same persisted manifest,
    with the same ``floor_ratio`` — so this cannot disagree with either by construction.

    WHY THIS EXISTS. ``git grep -n "STOP_UNDERFULL\\|stop_detail" origin/main -- src/``
    matched ONLY ``structural_relief.py`` and ``structural_relief_policy.py``: the gate
    detected an unsatisfiable order at plan time, before anything paid happened, refused
    to bill the render, wrote a full sentence of arithmetic explaining why — and that
    sentence reached no conformance output and no degradation code. The run shipped out
    of band with no visible explanation, and three separate sessions inferred three
    different WRONG mechanisms for short/long films while the answer was being written
    down and discarded on every run. Captured verbatim from a live 44.00s order::

        verdict: STOP_UNDERFULL   measured_seconds_per_beat: 12.424   beat_count: 5
        stop_detail: "no legal beat count lands inside the +/-10% band [39.60s, 48.40s]
        of the 44.00s order at 12.42s measured/beat: ... refusing to bill a render that
        lands outside the band"

    ⚠️ WORDING IS ATTRIBUTIVE ON PURPOSE. The detail says what the GATE said, and the
    per-beat figure is reported as the gate's measurement of THIS run's footage — not as
    a pipeline constant. ``measured_seconds_per_beat`` is per-scenario; a reader who
    mistakes it for a constant repeats the ``_SECONDS_PER_SCENARIO`` defect this repo has
    already paid for. If the lane that owns the policy changes the verdict, this reader
    reports the new one unchanged and needs no edit.

    WHY HERE and not in ``reconcile_timeline``, which produces the manifest. Two reasons,
    and the second is binding: (1) this is the demo_video/showcase CONFORMANCE EXIT, the
    same channel that already carries ``duration_conformance``'s verdict, and it already
    reads ``ctx.prior_output("reconcile_timeline")`` for exactly this class of upstream
    refusal (see :func:`plan_infeasible_reason` at the call site); (2)
    ``structural_relief_policy`` imports ``shared/duration_conformance``, so the shared
    conformance module CANNOT import the policy back without a cycle. A per-blueprint
    reader is also the correct scope rather than a shortcut: structural relief is
    demo_video-only (``structural_relief._FROM_STAGE == "demo_video_gen"``).

    TWO GATES, and the second one is load-bearing rather than belt-and-braces:

    * ``convergence_outcome == "cant_converge"`` — every other outcome either converged
      or never asked for relief.
    * ``overrun_seconds > 0`` — ``decide_relief_shape`` returns ``STOP_CANNOT_SHRINK``
      for ``overrun <= 0``, which is the ordinary "there is nothing to relieve" answer,
      NOT a refusal to remedy anything. Reporting it would fire this code on a
      ``cant_converge`` manifest that carries no overrun figure at all — e.g. a manifest
      whose ``run`` block predates those keys, which reads as ``0.0``. Verified against
      the live restraint guard in
      ``test_encode_master_duration_conformance.test_a_refused_plan_is_named_beside_the_band_pass``,
      which asserts ``out.degradations == []`` on exactly that shape: without this gate
      the entry fires there and turns a clean run degraded. That is the
      ``mastering_gate``-fires-on-96.7%-of-masters failure, caught before shipping.

    ``None`` when there is no relief manifest, when the manifest cannot support the
    judgement (``relief_shape_from_manifest`` ⇒ ``None``), on either gate above, or on
    ``SPAWN`` — a corrective render IS coming, so there is nothing to report.

    ⛔ NEVER FATAL, and never able to break a completed run. Per ``shared/degradation.py``
    a degradation is visibility, not a verdict: making an unsatisfiable order terminal
    would stop the run reaching ``mark_completed``, which is the ONLY path that spawns
    relief at all — an enforcing change here would silently disable the feature it
    reports on. Defensive for the same reason :func:`plan_infeasible_reason` is: this
    reads a JSON-round-tripped dict owned by another stage, on the exit path of a render
    that has already been PAID for, so every access is type-guarded and a malformed
    manifest yields ``None`` ("no refusal known"), never an exception.
    """
    if not isinstance(prior_output, Mapping):
        return None
    manifest = prior_output.get("timeline_manifest")
    if not isinstance(manifest, Mapping):
        return None
    run = manifest.get("run")
    if not isinstance(run, Mapping) or run.get("convergence_outcome") != "cant_converge":
        return None
    shape = relief_shape_from_manifest(
        cast("Mapping[str, object]", manifest),
        floor_ratio=get_settings().pipeline.demo_timeline_underfull_fail_ratio,
        # The master is already measured here, so this forecast is sized and guarded
        # off the same delivery the spawner (``runs/structural_relief``) will use.
        delivered_seconds=delivered_seconds,
    )
    # ``STOP_ALREADY_CONFORMING`` is a refusal to re-render a film that already MET the
    # order — reporting it as "the order is not met" would be false.
    if (
        shape is None
        or shape.verdict in ("SPAWN", "STOP_ALREADY_CONFORMING")
        or shape.overrun_seconds <= 0
    ):
        return None
    per_beat = (
        f"{shape.measured_seconds_per_beat:.3f}s/beat as measured on THIS run's footage"
        if shape.measured_seconds_per_beat is not None
        else "an UNMEASURED per-beat yield"
    )
    detail = shape.stop_detail or (
        "the gate refused without naming the arithmetic (no stop_detail on this verdict)"
    )
    return degradation(
        STRUCTURAL_RELIEF_REFUSED,
        f"the structural-relief gate returned {shape.verdict} and refused to bill a "
        f"corrective re-render, so this run ships with no second attempt and the order "
        f"is not met. Ordered {shape.target_seconds:.2f}s; overrun "
        f"{shape.overrun_seconds:.2f}s over {shape.beat_count} beat(s) at {per_beat}"
        f" ({SIZING_BASIS_PHRASES[shape.sizing_basis]}). "
        f"The gate's stated reason: {detail}. The user is the remaining lever — re-run "
        f"demo_video_gen with fewer beats, shorten the copy, or raise the ordered "
        f"duration.",
    )


def _seq(value: object) -> list[object]:
    """A plain list from an untyped JSON value (never a str/bytes/mapping)."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _is_number(value: object) -> bool:
    """A real numeric (bool is an int subclass — excluded)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_int(value: object, default: int) -> int:
    if not _is_number(value):
        return default
    return int(cast("float", value))


def _clamp_score(value: object) -> float:
    """Coerce a model score into the ``[1, 5]`` band; junk ⇒ the low end."""
    if not _is_number(value):
        return _SCORE_MIN
    score = float(cast("float", value))
    if score != score:  # NaN
        return _SCORE_MIN
    return max(_SCORE_MIN, min(_SCORE_MAX, score))


def _author_set(authored: Mapping[str, object], knob: str) -> bool:
    """Whether the author pinned ``knob`` in the mastering config (present + not None)."""
    return knob in authored and authored[knob] is not None


def _report_rerun(report: Mapping[str, object]) -> list[str]:
    """The deterministic report's ``rerun_stages`` as a plain ``list[str]``."""
    return [str(s) for s in _seq(report.get("rerun_stages"))]


def _report_gate_pass(report: Mapping[str, object]) -> bool | None:
    """The report's own ``gate_pass`` verdict — ``True``/``False`` when measured,
    ``None`` when ABSENT.

    ⚠️ This replaces a bare ``report.get("gate_pass", True)``, which read an absent
    verdict (an older compiler build, or a report whose post-mux verify itself never
    ran) as a PASS — indistinguishable from a gate that ran and actually passed. That
    is this codebase's #1 defect class (see ``shared/mastering_gate.py``'s
    ``_UNMEASURED_NOTE``). Returning ``None`` makes the absence a fact the caller can
    see and log, instead of a silent fabrication.

    ⛔ ARMS NOTHING: the caller's recompile decision only fires on an explicit
    ``gate_pass is False`` (see the call site), so an unmeasured report still does
    NOT trigger a re-master — byte-identical to the previous default-True behaviour
    for that decision. Only the visibility changed.
    """
    value = report.get("gate_pass")
    return value if isinstance(value, bool) else None


def _report_summary(report: Mapping[str, object]) -> dict[str, object]:
    """The aggregate report projection handed to each worker prompt."""
    return {key: report.get(key) for key in _REPORT_KEYS}


def _critic_message(report: Mapping[str, object], delivery_intent: Mapping[str, object]) -> str:
    return (
        "MASTERING REPORT (deterministic, measured off the rendered file):\n"
        + json.dumps(_report_summary(report), ensure_ascii=False, sort_keys=True, default=str)
        + "\n\nDELIVERY INTENT:\n"
        + json.dumps(dict(delivery_intent), ensure_ascii=False, sort_keys=True, default=str)
    )


def _advisor_message(profile_delta: Mapping[str, object], report: Mapping[str, object]) -> str:
    return (
        "The encode-critic proposes this mastering delta:\n"
        + json.dumps(dict(profile_delta), ensure_ascii=False, sort_keys=True, default=str)
        + "\n\nCurrent mastering report:\n"
        + json.dumps(_report_summary(report), ensure_ascii=False, sort_keys=True, default=str)
        + "\n\nReturn a concrete, conservative mastering block "
        "(loudness_target, crf, deband, maxrate)."
    )


def _clamp_mastering_block(
    proposed: Mapping[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Clamp the advisor's proposed knobs to their §5 ranges + record provenance.

    Numerics (``loudness_target`` / ``crf``) route through :func:`provenance.clamp`
    (which records whether the value actually moved); ``deband`` is coerced to the
    ``{auto,on,off}`` enum (default ``auto`` on junk); ``maxrate`` is a pass-through
    string. Each touched knob contributes one ``{knob, proposed, clamped, bound}``
    provenance record. A knob the advisor omitted is left out entirely.
    """
    block: dict[str, object] = {}
    provenance: list[dict[str, object]] = []

    loudness = proposed.get("loudness_target")
    if _is_number(loudness):
        proposed_lt = float(cast("float", loudness))
        resolved = clamp(proposed_lt, _LOUDNESS_LO, _LOUDNESS_HI, computed_by=_COMPUTED_BY)
        block["loudness_target"] = resolved.value
        provenance.append(
            {
                "knob": "loudness_target",
                "proposed": proposed_lt,
                "clamped": resolved.value,
                "bound": [_LOUDNESS_LO, _LOUDNESS_HI],
            }
        )

    crf = proposed.get("crf")
    if _is_number(crf):
        proposed_crf = int(cast("float", crf))
        resolved = clamp(proposed_crf, _CRF_LO, _CRF_HI, computed_by=_COMPUTED_BY)
        block["crf"] = resolved.value
        provenance.append(
            {
                "knob": "crf",
                "proposed": proposed_crf,
                "clamped": resolved.value,
                "bound": [_CRF_LO, _CRF_HI],
            }
        )

    if proposed.get("deband") is not None:
        raw_deband = proposed.get("deband")
        coerced = (
            raw_deband
            if isinstance(raw_deband, str) and raw_deband in _DEBAND_VALUES
            else _DEBAND_DEFAULT
        )
        block["deband"] = coerced
        provenance.append(
            {
                "knob": "deband",
                "proposed": raw_deband,
                "clamped": coerced,
                "bound": sorted(_DEBAND_VALUES),
            }
        )

    maxrate = proposed.get("maxrate")
    if isinstance(maxrate, str) and maxrate.strip():
        block["maxrate"] = maxrate
        provenance.append(
            {"knob": "maxrate", "proposed": maxrate, "clamped": maxrate, "bound": None}
        )

    return block, provenance


def _apply_authored(
    clamped_block: Mapping[str, object], authored: Mapping[str, object]
) -> tuple[dict[str, object], dict[str, object]]:
    """Merge the author's pinned knobs over the advisor's clamped block.

    Returns ``(effective_block, effective_fills)``: ``effective_block`` is the final
    per-knob value the author WOULD ship (authored value where pinned, else the clamped
    advisor value) — this is what is surfaced as the recommendation; ``effective_fills``
    is only the knobs the author left unset, i.e. what the re-compile actually adds on
    top of the authored mastering block. An authored knob is therefore never overwritten.
    """
    effective_block = dict(clamped_block)
    effective_fills = dict(clamped_block)
    for knob in _KNOBS:
        if _author_set(authored, knob):
            effective_block[knob] = authored[knob]
            effective_fills.pop(knob, None)
    return effective_block, effective_fills


class EncodeMasterStage(StageHandler[EncodeMasterInput, EncodeMasterOutput]):
    """The encode-critic / mux-mastering-advisor bounded evaluator-optimizer (spec 13 §5).

    Standalone (not a footage_critique subclass); it reuses the same soft-degrade idioms
    — every worker / compile failure accepts the current artifact rather than failing the
    deliverable. WM auto-creates ``encode-critic`` + ``mux-mastering-advisor`` on the
    first call."""

    name = "demo_video_encode_master"
    input_model = EncodeMasterInput
    output_model = EncodeMasterOutput
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (
        SlotUse("llm"),
        SlotUse("tts"),
        SlotUse("asr", when=SLOT_USE_OPTIONAL),
    )

    def __init__(
        self,
        client: WMClient | None = None,
        compile_client: VideoCompileClient | None = None,
    ) -> None:
        self._llm_timeout = llm_budget_seconds()
        self._llm_retries = llm_retries()
        self._wm = client or WMClient(timeout=llm_client_timeout_seconds())
        # The compile client threaded into DemoVideoCompileStage on a guarded re-master;
        # unused on the disabled / advisory-only paths. Same seam the footage-critique +
        # final-critique loops use so a re-compile reuses the CANONICAL compile path.
        self._compile_client = compile_client or VideoCompileClient()

    # ── inputs ───────────────────────────────────────────────────────────────
    def _mastering_cfg(self, ctx: RunContext) -> Mapping[str, object]:
        return _as_mapping(ctx.config_snapshot.get("mastering"))

    def _starting_artifact(self, ctx: RunContext) -> tuple[str, float]:
        """``(artifact_url, duration_seconds)`` of the video to master: the L4
        footage-critique's accepted-best artifact when it ran, else compile's. A
        genuinely missing artifact (both empty) is a hard ``StagePreconditionError``."""
        critique = ctx.prior_output("demo_video_footage_critique") or {}
        url = critique.get("artifact_url")
        dur = critique.get("duration_seconds")
        if not url:
            compile_out = ctx.prior_output("demo_video_compile") or {}
            url = compile_out.get("artifact_url")
            dur = compile_out.get("duration_seconds")
        if not url:
            raise StagePreconditionError(
                "demo_video_encode_master: no compiled artifact from footage_critique or compile",
                missing=["demo_video_compile.artifact_url"],
            )
        return str(url), as_float(dur)

    # ── upstream calls (the ONE policy, one level inside the closures) ───────
    def _wm_ladder(self) -> tuple[float, int]:
        """Per-attempt budget + retry count for ONE WM call from this stage.

        Sized to fit INSIDE the loop engine's own per-attempt budget: the engine
        wraps each closure in ``stage_call(timeout_seconds=self._llm_timeout)``, and
        a ladder that overran it would be CANCELLED mid-way — which loses the ledger
        append this whole design exists to protect."""
        return attempt_plan(self._llm_timeout, self._llm_retries)

    async def _critic_call(
        self, draft: _EncodeDraft, delivery_intent: Mapping[str, object]
    ) -> WMResponse:
        per_attempt, retries = self._wm_ladder()
        return await stage_call(
            downshift_reasoning(
                lambda think: guarded(
                    lambda: self._wm.message(
                        enable_reasoning=think,
                        worker_name=_CRITIC_WORKER,
                        message=_critic_message(draft.report, delivery_intent),
                        system_prompt=_CRITIC_SYSTEM_PROMPT,
                        output_schema=_CRITIC_OUTPUT_SCHEMA,
                        max_tokens=_CRITIC_MAX_TOKENS,
                    )
                )
            ),
            timeout_seconds=per_attempt,
            retries=retries,
            retry_on=(TransientUpstreamError, httpx.TransportError),
            stage_name=f"{self.name}:encode_critic",
        )

    async def _advisor_call(
        self, profile_delta: Mapping[str, object], draft: _EncodeDraft
    ) -> WMResponse:
        per_attempt, retries = self._wm_ladder()
        return await stage_call(
            downshift_reasoning(
                lambda think: guarded(
                    lambda: self._wm.message(
                        enable_reasoning=think,
                        worker_name=_ADVISOR_WORKER,
                        message=_advisor_message(profile_delta, draft.report),
                        system_prompt=_ADVISOR_SYSTEM_PROMPT,
                        output_schema=_ADVISOR_OUTPUT_SCHEMA,
                        max_tokens=_ADVISOR_MAX_TOKENS,
                    )
                )
            ),
            timeout_seconds=per_attempt,
            retries=retries,
            retry_on=(TransientUpstreamError, httpx.TransportError),
            stage_name=f"{self.name}:mastering_advisor",
        )

    # ── handler ──────────────────────────────────────────────────────────────
    async def handle(  # noqa: C901 — critic/optimizer closures capture the loop's shared soft-degrade state (reco_by_url, authored); splitting hurts clarity
        self, payload: EncodeMasterInput, ctx: RunContext
    ) -> EncodeMasterOutput:
        start_url, start_duration = self._starting_artifact(ctx)
        compile_out = ctx.prior_output("demo_video_compile") or {}
        report = dict(_as_mapping(compile_out.get("mastering")))
        authored = self._mastering_cfg(ctx)
        # Presence-is-opt-in: a truthy `mastering.critic` (a non-empty config dict, or a
        # bare `true`) enables the loop; absent / falsy (`{}` included) disables it. When
        # enabled but not a dict, the per-run threshold/iterations fall back to defaults.
        critic_raw = authored.get("critic")
        critic_enabled = bool(critic_raw)
        critic_cfg = _as_mapping(critic_raw)

        initial = _EncodeDraft(
            artifact_url=start_url,
            duration_seconds=start_duration,
            mastering_block=dict(authored),
            report=report,
        )

        # DISABLED-BY-DEFAULT: no truthy `mastering.critic` ⇒ run the loop with
        # `critic_fn=None` (zero awaits, zero WM calls) and emit the UNCHANGED artifact +
        # the existing report as `mastering`, empty recommendation. Byte-identical.
        if not critic_enabled:

            async def _noop_optimizer(draft: _EncodeDraft, _critique: Critique) -> _EncodeDraft:
                return draft

            await run_evaluator_optimizer(initial, critic_fn=None, optimizer_fn=_noop_optimizer)
            return self._conformed(self._accept_current(initial, report, encode_critic=None), ctx)

        threshold = as_float(critic_cfg.get("score_threshold"), _THRESHOLD)
        max_passes = _as_int(critic_cfg.get("iterations"), _MAX_PASSES)
        delivery_intent: dict[str, object] = {"platform": authored.get("platform")}
        # The retained draft's clamped recommendation + provenance, keyed by artifact_url
        # (the loop's LoopResult exposes only the best draft, so — like footage_critique's
        # records_by_url — we look the recommendation back up after the loop).
        reco_by_url: dict[str, dict[str, object]] = {}
        # Why these two ledgers exist: both closures below degrade on a worker error
        # rather than failing a paid render, and the critic's degrade returns a Critique
        # scored exactly AT the threshold so the loop converges. That is the right
        # runtime behaviour and is deliberately kept — but it means a critic that never
        # ran emits the same `score` as a critic that ran and passed at the threshold.
        # `provenance == []` / `rerun_stages == []` were doing that job implicitly, and
        # they cannot: a genuine pass with no rerun looks identical. Record the failures
        # explicitly so "approved" and "never assessed" are distinguishable downstream.
        critic_failures: list[str] = []
        advisor_failures: list[str] = []

        async def critic_fn(draft: _EncodeDraft) -> Critique:
            # SOFT: an unregistered / erroring worker degrades to accept-current (score at
            # the threshold ⇒ converge, no optimize), never raising.
            #
            # ⚠️ THE CATCH MUST STAY INSIDE THIS CLOSURE. ``run_evaluator_optimizer``
            # already wraps ``critic_fn`` in ``stage_call(guarded(...))`` with
            # ``on_loop_error="raise"``, so letting an upstream failure ESCAPE routes
            # exhaustion to the loop's raise → this stage's outer ``except Exception →
            # _accept_current(encode_critic=None)``, which discards ``critic_failures`` /
            # ``advisor_failures`` entirely. The run would go from "degraded, and here is
            # why" to "clean, no note" — the exact opposite of the point. So the retry
            # ladder is run HERE, one level in, where the ledger survives it.
            try:
                resp = await self._critic_call(draft, delivery_intent)
            except PermanentUpstreamError as exc:
                # No retry can fix it (402 no-provider, an unregistered worker, a schema
                # WM rejects). Logged at ERROR with the real upstream sentence and
                # recorded as such, so "never assessed for a fixable reason" is
                # distinguishable from "the worker was down".
                log.error(
                    "%s: encode-critic PERMANENT upstream fault, accepting current "
                    "(master NOT assessed): %s",
                    self.name,
                    exc,
                )
                critic_failures.append(f"permanent upstream fault — {exc}")
                return Critique(score=threshold, rerun_stages=[], directives={})
            except StageRetriesExhaustedError as exc:
                log.warning(
                    "%s: encode-critic transient upstream failure after %d attempt(s), "
                    "accepting current: %s",
                    self.name,
                    exc.attempts,
                    exc.last_error,
                )
                critic_failures.append(
                    f"transient upstream failure after {exc.attempts} attempt(s) — "
                    f"{describe_exception(exc.last_error)}"
                )
                return Critique(score=threshold, rerun_stages=[], directives={})
            except Exception as exc:  # noqa: BLE001 — worker error must not fail the run
                log.warning("%s: encode-critic worker error, accepting current: %s", self.name, exc)
                critic_failures.append(f"{type(exc).__name__}: {exc}")
                return Critique(score=threshold, rerun_stages=[], directives={})
            data = _as_mapping(resp.output)
            score = _clamp_score(data.get("overall_score"))
            profile_delta = dict(_as_mapping(data.get("profile_delta")))
            proposal_rerun = [str(s) for s in _seq(data.get("rerun_stages"))]
            # The deterministic report's rerun_stages win over the critic's proposal.
            rerun = _report_rerun(draft.report) or proposal_rerun
            return Critique(
                score=score, rerun_stages=rerun, directives={"profile_delta": profile_delta}
            )

        async def optimizer_fn(draft: _EncodeDraft, critique: Critique) -> _EncodeDraft:
            directives = _as_mapping(critique.directives)
            profile_delta = _as_mapping(directives.get("profile_delta"))
            # SOFT: an advisor error keeps the current draft (the loop plateaus).
            # Same ledger-preservation reasoning as ``critic_fn`` above: the classify +
            # retry happens INSIDE this closure so ``advisor_failures`` survives it.
            try:
                resp = await self._advisor_call(profile_delta, draft)
            except PermanentUpstreamError as exc:
                log.error(
                    "%s: mux-mastering-advisor PERMANENT upstream fault, keeping current: %s",
                    self.name,
                    exc,
                )
                advisor_failures.append(f"permanent upstream fault — {exc}")
                return draft
            except StageRetriesExhaustedError as exc:
                log.warning(
                    "%s: mux-mastering-advisor transient upstream failure after %d "
                    "attempt(s), keeping current: %s",
                    self.name,
                    exc.attempts,
                    exc.last_error,
                )
                advisor_failures.append(
                    f"transient upstream failure after {exc.attempts} attempt(s) — "
                    f"{describe_exception(exc.last_error)}"
                )
                return draft
            except Exception as exc:  # noqa: BLE001 — worker error must not fail the run
                log.warning("%s: mux-mastering-advisor error, keeping current: %s", self.name, exc)
                advisor_failures.append(f"{type(exc).__name__}: {exc}")
                return draft
            clamped_block, provenance = _clamp_mastering_block(_as_mapping(resp.output))
            # AUTHORED-OVERRIDE WINS: never overwrite a pinned knob; only fill the unset.
            effective_block, effective_fills = _apply_authored(clamped_block, authored)
            recommendation: dict[str, object] = {
                "profile_recommendation": effective_block,
                "provenance": provenance,
            }

            # Re-compile IFF the critic/report asked for a rerun OR the deterministic
            # gate failed. footage_critique persists its nudged plans to its DAG output,
            # so this re-compile reproduces the critique's footage deterministically
            # (compile prefers `demo_video_footage_critique.footage_plans`) AND applies
            # the mastering delta — the two loops compose instead of fighting (S20).
            # Otherwise the recommendation is advisory-only (stashed on the current draft).
            gate_pass = _report_gate_pass(draft.report)
            if gate_pass is None:
                log.info(
                    "%s: draft report carries no gate_pass verdict — the mastering "
                    "gate never ran (report UNMEASURED, not a pass); not treating "
                    "that absence as a recompile trigger.",
                    self.name,
                )
            # ⛔ SCOPED TO WHAT A RE-MASTER CAN ACTUALLY FIX. This used to be
            # `bool(critique.rerun_stages) or gate_pass is False`, and the second term
            # is near-redundant with the first: every check that clears `gate_pass`
            # compiler-side also appends its stage here. What that cost: `colorspace_ok`
            # fails on 23.3% of masters (n=136 live) on an upstream video-TAGGING
            # defect, and the compiler's own `mastering.py` states that re-running
            # mastering re-emits the identical tag and can never clear it. So roughly a
            # third of all runs bought 2 LLM calls PLUS a full real-ffmpeg recompile
            # (twice, `_MAX_PASSES = 2`) that provably could not change the verdict that
            # triggered it. `cfr` is the same shape: appended to `rerun_stages` while
            # the compiler says no stage of the pipeline can remediate cadence.
            #
            # The gate term is KEPT, not deleted — a genuinely remediable failure
            # (loudness drift, a dead signalstats probe, a blank render) still
            # recompiles exactly as before, and an unclassified/new stage still counts
            # as remediable. See `shared/mastering_gate.remediable_by_remaster`.
            remediable = remediable_by_remaster(critique.rerun_stages)
            # `gate_pass is False` with NO stage named is unreachable on today's
            # compiler, but if it ever happens the report tells us nothing about WHAT
            # failed — and an unnamed failure is not evidence of an unremediable one, so
            # it keeps the old behaviour rather than silently swallowing the re-master.
            should_recompile = bool(remediable) or (
                gate_pass is False and not critique.rerun_stages
            )
            if critique.rerun_stages and not remediable:
                log.info(
                    "%s: every stage the mastering gate named (%s) is one the compiler "
                    "declares unfixable by re-mastering (gate_pass=%s) — skipping the "
                    "re-compile it could not have satisfied; the recommendation is "
                    "stashed as advisory and the failed gate still degrades the run.",
                    self.name,
                    ", ".join(critique.rerun_stages),
                    gate_pass,
                )
            if not should_recompile:
                reco_by_url[draft.artifact_url] = recommendation
                return draft

            # Re-compile through the CANONICAL compile path with the effective mastering
            # block merged onto the authored one. loudness_target / deband / maxrate /
            # colorspace AND `crf` all take effect: setting a mastering `crf` makes
            # `resolution_tags` drop the per-tier `crf` compile_tag, so this clamped
            # value is the one the compiler encodes.
            try:
                merged_mastering: dict[str, object] = {**dict(authored), **effective_fills}
                # `replace(ctx, ...)` keeps run_id AND the stage name, so without a
                # discriminator EVERY draft in this loop collides onto the run's first
                # compile key: video_compile returns the ORIGINAL job and this stage
                # ships the original video while `_EncodeDraft.report` (built from
                # `out.mastering`) reports the new crf/loudnorm/deband as applied.
                # Keyed on the MERGED mastering block, not the pass ordinal, so two
                # drafts that land on identical settings still dedupe to the render
                # already paid for; identical to the authored block ⇒ no delta ⇒ ""
                # ⇒ dedupes onto the original render.
                new_ctx = replace(
                    ctx,
                    config_snapshot={**ctx.config_snapshot, "mastering": merged_mastering},
                    render_idempotency_discriminator=(
                        ""
                        if merged_mastering == dict(authored)
                        else render_payload_discriminator("mastering", merged_mastering)
                    ),
                )
                out = await DemoVideoCompileStage(client=self._compile_client).handle(
                    DemoVideoCompileInput(), new_ctx
                )
            except Exception as exc:  # noqa: BLE001 — re-compile failure ⇒ keep current
                log.warning("%s: mastering re-compile failed, keeping current: %s", self.name, exc)
                reco_by_url[draft.artifact_url] = recommendation
                return draft
            new_draft = _EncodeDraft(
                artifact_url=str(out.artifact_url),
                duration_seconds=as_float(out.duration_seconds),
                mastering_block=effective_block,
                report=dict(_as_mapping(out.mastering)),
            )
            reco_by_url[new_draft.artifact_url] = recommendation
            return new_draft

        # Belt-and-suspenders: the closures swallow their own failures, but a defect in
        # the loop itself must STILL accept the current artifact — the soft gate never
        # lets this stage fail the deliverable.
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
        except Exception as exc:  # noqa: BLE001 — SOFT gate: accept current on any loop error
            # The soft gate stays: a loop-level fault (a loop timeout, a defect in the
            # loop, the ``stage_call``/``guarded`` wrapper itself) must NOT fail a paid
            # render. But it fires OUTSIDE the two closures, so nothing else records
            # that the master went unassessed — and emitting
            # ``encode_critic=None, notes=[]`` here is BYTE-IDENTICAL to the
            # critic-DELIBERATELY-DISABLED path above. `api/routes/runs/_shared.py`
            # derives `degraded` from a non-empty `notes`, so that shape reports a
            # clean, assessed master that was never assessed. Carry the reason (plus
            # whatever the closures already banked before the loop died) out in the
            # SAME vocabulary the closures use — `critic_unavailable` + an explicit
            # note — so "disabled" and "crashed" are distinguishable downstream.
            log.warning("%s: loop error, accepting current: %s", self.name, exc)
            loop_failure = f"{type(exc).__name__}: {exc}"
            critic_failures.append(f"loop error — {loop_failure}")
            loop_notes = [
                f"encode_master: the encode-critic loop itself failed, so this master "
                f"was accepted WITHOUT being assessed — {loop_failure}"
            ]
            if len(critic_failures) > 1:
                loop_notes.append(
                    f"encode_master: the encode-critic worker had already failed on "
                    f"{len(critic_failures) - 1} pass(es) before the loop failed "
                    f"— first error: {critic_failures[0]}"
                )
            if advisor_failures:
                loop_notes.append(
                    f"encode_master: the mux-mastering-advisor failed on "
                    f"{len(advisor_failures)} pass(es), so no mastering delta was "
                    f"applied — first error: {advisor_failures[0]}"
                )
            return self._conformed(
                self._accept_current(
                    initial,
                    report,
                    encode_critic={
                        # No verdict exists — the loop never returned one. `score: None`
                        # says that outright rather than borrowing the threshold, which a
                        # genuine pass also emits.
                        "score": None,
                        "profile_recommendation": {},
                        "provenance": [],
                        "rerun_stages": [],
                        "critic_unavailable": True,
                        "loop_failed": True,
                        "critic_failed_passes": len(critic_failures),
                        "advisor_failed_passes": len(advisor_failures),
                    },
                    notes=loop_notes,
                ),
                ctx,
            )

        best = result.draft
        reco = reco_by_url.get(best.artifact_url, {})
        encode_critic: dict[str, object] = {
            "score": result.score,
            "profile_recommendation": reco.get("profile_recommendation", {}),
            "provenance": reco.get("provenance", []),
            "rerun_stages": list(result.rerun_stages),
            # `score` alone cannot answer "was this master actually assessed?" — the
            # critic's soft-degrade scores at the threshold to converge. These two say
            # so outright, so no consumer has to infer approval from a number that a
            # dead worker also produces.
            "critic_unavailable": bool(critic_failures),
            "critic_failed_passes": len(critic_failures),
            "advisor_failed_passes": len(advisor_failures),
        }
        notes: list[str] = []
        if critic_failures:
            # A note is the machine-readable degrade signal for a run
            # (`api/routes/runs/_shared.py` derives `degraded` from a non-empty
            # `notes`), so an unassessed master no longer reads as a clean pass.
            notes.append(
                f"encode_master: the encode-critic worker failed on "
                f"{len(critic_failures)} of {result.passes} pass(es), so this master "
                f"was accepted WITHOUT being assessed (score {result.score:.1f} is the "
                f"convergence threshold, not a verdict) — first error: "
                f"{critic_failures[0]}"
            )
        if advisor_failures:
            notes.append(
                f"encode_master: the mux-mastering-advisor failed on "
                f"{len(advisor_failures)} pass(es), so no mastering delta was applied "
                f"— first error: {advisor_failures[0]}"
            )
        if not result.converged and result.rerun_stages:
            notes.append(
                f"encode_master: accepted best master at score {result.score:.1f} "
                f"(< {threshold:.1f}) after {result.passes} pass(es) — {result.reason}"
            )
        return self._conformed(
            EncodeMasterOutput(
                artifact_url=best.artifact_url,
                duration_seconds=best.duration_seconds,
                mastering=best.report or None,
                encode_critic=encode_critic,
                notes=notes,
            ),
            ctx,
        )

    def _conformed(self, out: EncodeMasterOutput, ctx: RunContext) -> EncodeMasterOutput:
        """Stamp the delivered-vs-ORDERED comparison onto the output this stage ships.

        Applied at EVERY exit of :meth:`handle` — the disabled default path, the
        loop-failure soft-degrade, and the converged path — because all three ship a
        real, paid master and the length of a master is a fact about the delivery, not
        about whether the encode critic happened to run. Stamping only the "good" exit
        is how ``demo_video`` came to be the one video blueprint with no conformance
        record at all while ``website_review`` (273%), ``lesson`` (216%),
        ``app_review`` (140%) and ``promo`` (129%) all had one.

        RECORDING ONLY. ``observe_duration_conformance`` never raises and never
        truncates; the breach lands on the structured ``duration_conformance`` field
        (the numbers), on the ``degradations`` channel (the queryable verdict) and — for
        an out-of-tolerance verdict only — on ``notes``, which is what actually flips the
        run's ``degraded`` flag without failing it. The "nobody probed this file"
        fact lands on ``advisories`` — the NON-degrading tier — because it fires on
        every delivery whose mastering report is absent or degraded, so on
        ``degradations`` it would report those masters degraded for a gap in OUR
        instrumentation rather than a defect in the artifact. Existing ``degradations``,
        ``advisories`` and ``notes`` are all EXTENDED, never replaced — a
        mastering-gate entry or a non-convergence note banked upstream of here must
        survive.
        """
        # ⭐ THE MEASURED RUNG IS REACHABLE HERE, and the comment that stood in its
        # place said it was not. `EncodeMasterOutput.mastering` (:270) is the compiler's
        # own mastering report — fed from `compile_out["mastering"]` in `handle`, and
        # replaced by the re-mastered report when the critic loop converges — which is
        # exactly the shape `playable_seconds_from_mastering` reads
        # (`video_stream_seconds` / `audio_stream_seconds`, probed SEPARATELY). Passing
        # it is the difference between grading the delivered FILE and grading the PLAN
        # against itself: without it every verdict this stage has ever stamped was
        # `declared`, i.e. the nominal sum of shot durations agreeing with the order it
        # was derived from — structurally incapable of catching a master whose picture
        # stopped early, which is the failure this check exists for.
        #
        # `None` still falls through to `declared` + the DURATION_UNMEASURED advisory:
        # an absent mastering block (the disabled-critic path on a compile that never
        # mastered), a `degraded_report()` that nulled both stream fields when
        # `post_mux_verify` crashed, or the compiler's `0.0` unknown sentinel. That is
        # why the unmeasured fact stays an ADVISORY — on `degradations` it would flip
        # `degraded=True` on every unmasterable delivery, conforming ones included.
        #
        # ⭐ AND THE UPSTREAM REFUSAL TRAVELS WITH IT. `reconcile_timeline` can stamp a
        # timeline INFEASIBLE — `convergence_outcome="cant_converge"` plus an
        # `ordered_duration_infeasible` degradation — and that verdict has no band at
        # all: it is a bare `Σ min > target`, seconds that cannot be created. This
        # stage's band is +/-10% on the probed file. Those grade DIFFERENT OBJECTS, so
        # a plan refused at zero tolerance routinely delivers a file inside +/-10%, and
        # until now this stage reported that as a bare pass. Prod run
        # `01a0a3c6-c8e7-73db-b13e-fded1a4cbb4f`: timeline `feasible: [False, False,
        # False]`, half the beats already floored to `vo_rate 0.8`, refusal emitted —
        # and then `delivered 32.038s / ordered 30.0s, +6.79%, within_tolerance: True`.
        # The band was not wrong; it was quoted without the refusal beside it.
        #
        # ⛔ RECORDING ONLY, again: this moves the LOG LINE (INFO -> WARNING) and adds
        # an ADVISORY. `within_tolerance` is untouched, no degradation is added, and
        # nothing that passed before stops passing — the infeasibility is already a
        # degradation on the stage that decided it, and this stage only observed it.
        # Read ONCE and reused: `plan_infeasible_reason` and `_relief_refusal_degradation`
        # ask two different questions of the SAME persisted stage output, and fetching it
        # twice is how two readers end up disagreeing about one run.
        reconciled = ctx.prior_output("reconcile_timeline")
        conformance = observe_duration_conformance(
            self.name,
            demo_video_ordered_seconds(ctx.config_snapshot),
            out.duration_seconds,
            playable_seconds=playable_seconds_from_mastering(out.mastering),
            # Both blueprints that reach this stage (`demo_video`, `showcase`) place
            # `reconcile_timeline` upstream of it, so its output is always in
            # `prior_run_outputs` by the time this runs. `None` when it is not — a
            # replayed pre-manifest run, a regenerate that re-entered below it — and
            # `plan_infeasible_reason` reads that as "no refusal known", never as a
            # refusal and never as an exception.
            plan_infeasible=plan_infeasible_reason(reconciled),
        )
        degradations = [
            *duration_conformance_degradations(conformance),
            # A THIRD independent fact, and the one that had no channel at all. The two
            # entries around it grade the DELIVERED FILE (its length, its picture); this
            # one reports that the gate which could have CORRECTED the length declined to
            # spend a render doing so. `plan_infeasible` above records that the plan was
            # refused — it does not say whether a remedy was attempted, and until now
            # nothing did: the verdict and its arithmetic lived only in the two files
            # that produce them (see `_relief_refusal_degradation`).
            *(
                [entry]
                if (
                    entry := _relief_refusal_degradation(
                        reconciled,
                        delivered_seconds=(
                            conformance["delivered_seconds"] if conformance is not None else None
                        ),
                    )
                )
                is not None
                else []
            ),
            # A SECOND, INDEPENDENT FACT about the same master, and the demo_video /
            # showcase path was the one blueprint missing it — `lesson`, `app_review`,
            # `action_reel`, `promo` and `website_review` all call this at their own
            # compile exits. The verdict above grades the PLAYABLE length (the max of
            # the two streams) against the order, so on a truncated master that max IS
            # the audio and the truncation is invisible to it. One number cannot carry
            # both facts; this is the other one.
            *picture_truncation_degradations(out.mastering),
        ]
        advisories = duration_conformance_advisories(conformance)
        # BOTH CHANNELS, ONE FAULT. The structured entry above is the queryable
        # verdict, but ``degraded`` on the run record is still derived from ``notes``
        # alone (``api/routes/runs/_shared.py`` → ``is_degraded``), so a master that
        # shipped 155% of its ordered length was recorded with the breach in
        # ``degradations`` and the run flagged NOT degraded. The note carries the SAME
        # sentence as the degradation's ``detail`` — one measurement, never two
        # wordings that can drift apart.
        #
        # ⛔ ONLY ``duration_out_of_tolerance`` is mirrored into PROSE, and that is the
        # whole of what this list decides. ``duration_not_ordered`` (and, since this
        # stage gained the measured rung, ``duration_picture_truncated``) stay out of
        # ``notes`` so the prose channel reads only about a MISSED order.
        #
        # ⚠️ THAT IS NOT THE SAME AS "CANNOT FLIP ``degraded``", and this comment used
        # to claim it was. ``degradation.is_degraded`` (``shared/degradation.py:283-285``)
        # is ``bool(degradations) or bool(notes)`` — an OR over BOTH channels — so every
        # entry appended above already flips ``degraded`` regardless of the prose. The
        # "prose-silent ⇒ harmless" reasoning predates the ``degradations`` channel and
        # was stale the moment that channel landed.
        #
        # Left as-is deliberately: ``duration_conformance``'s module docstring makes
        # degrading on an absent order the INTENT — "'nothing was ordered' is not 'on
        # target'", and a ``duration_conformance: null`` field is invisible to any
        # ``within_tolerance = false`` query — and the same module owns that decision
        # for the four other blueprints that call it. Demoting it to an advisory HERE
        # only would make demo_video the one blueprint disagreeing with the shared
        # module; doing it fleet-wide is a change to
        # ``duration_conformance_degradations``, not to this list.
        notes = [
            f"encode_master: {entry['detail']}"
            for entry in degradations
            if entry["code"] == DURATION_OUT_OF_TOLERANCE
        ]
        return out.model_copy(
            update={
                "duration_conformance": conformance,
                "degradations": [*out.degradations, *degradations],
                "advisories": [*out.advisories, *advisories],
                "notes": [*out.notes, *notes],
            }
        )

    def _accept_current(
        self,
        draft: _EncodeDraft,
        report: Mapping[str, object],
        *,
        encode_critic: dict[str, object] | None,
        notes: list[str] | None = None,
    ) -> EncodeMasterOutput:
        """Emit the current artifact + the existing report (the soft/disabled fallback).

        ``notes`` defaults to empty for the DISABLED path (nothing degraded — the run
        is genuinely clean). A caller that reaches here because something FAILED must
        pass its reasons, or the run reports a clean, assessed master that was never
        assessed.
        """
        return EncodeMasterOutput(
            artifact_url=draft.artifact_url,
            duration_seconds=draft.duration_seconds,
            mastering=dict(report) or None,
            encode_critic=encode_critic,
            notes=list(notes or []),
        )
