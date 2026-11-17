"""The compiler's mastering gate, read on behalf of EVERY render blueprint.

This logic lived in ``demo_video/compile.py`` and therefore protected exactly one
blueprint of the nine that subclass :class:`CompileStageBase`. The gate is a
property of the compiler's response, not of demo_video, so it belongs here:
lesson, website_review, generic/video_render, action_reel, promo, app_review and
audit_video (plus its seo_audit / cro_audit re-declarations) all shipped a blank
master with no note at all.

Severity is DEGRADE, deliberately. A non-empty ``notes`` on a ``completed`` stage
makes ``runs/_shared.py`` report the run as ``degraded`` WITHOUT failing it. A
failed mastering gate on an otherwise-playable, correct-length artifact is a
QUALITY problem, not a corrupt one — hard-failing would destroy a paid
deliverable over a banding score.

⚠️ ``duplicate_frames`` is deliberately NOT flagged here yet. The compiler began
emitting it (W1-3), but the fleet median duplicate ratio is 0.904 against a 0.10
ceiling, so wiring it today would degrade ~100% of runs — the exact "badge that
lights on 96.7% of runs" failure the narrowing below was written to avoid. It
becomes a signal once the CAPTURE defect that produces those duplicates is
fixed, not before.
"""

from __future__ import annotations

from typing import cast

from src.services.stages.shared.degradation import (
    MASTERING_ADVISORY,
    MASTERING_GATE_FAILED,
    Degradation,
    advisory,
    degradation,
)

# An ABSENT report is not a passing one. The compiler's post-mux verify is what
# produces this block; when it is missing (an older compiler build, or the verify
# pass itself failed) NOTHING measured the delivered master, and returning `[]` made
# that byte-identical to a gate that ran and PASSED. That is this codebase's #1
# defect class — an absent measurement reported as a passing one — so the gap gets
# its OWN note, distinct in wording from both "passed" (no note) and "failed"
# (the notes below, which name the failing metric).
_UNMEASURED_NOTE = (
    "mastering: the compiler returned NO mastering report, so the mastering gate "
    "never ran — this master is UNMEASURED, not verified; degrading"
)


# ⛔ CROSS-REPO DUPLICATED THRESHOLD — READ BEFORE RETUNING EITHER END.
#
# This is a MIRROR of the compiler's own blank-picture cutoff:
#   setting: `render_dead_frame_min_ratio` (default 0.10)
#   declared: `video_compiler_service/src/config.py` (field `render_dead_frame_min_ratio`)
#   applied:  `video_compiler_service/src/services/mastering.py` `post_mux_verify`
#             (`dead_min_ratio = float(getattr(settings, "render_dead_frame_min_ratio", 0.10))`,
#             then `_dead_frame(dead_ratio, dead_min_ratio)`)
#
# CHECKED on `origin/main` of the compiler (2026-09-15): the advisory report publishes
# `dead_frame_ratio`, `enforced_stages`, `unmeasured` and every metric — but NO cutoff
# key. The only place the compiler states the number on the wire is the `limit` field
# of a `dead_frame` entry in the per-violation list (see `_dead_frame_cutoff` below),
# and that entry exists ONLY once the compiler has itself decided `dead_frame` — by
# which point `"dead_frame" in rerun_stages` already decides `_blank_picture` without
# any comparison. So the number cannot be fully sourced from the payload today and the
# mirror has to stay.
#
# ⚠️ CONSEQUENCE OF RETUNING THE COMPILER SETTING WITHOUT TOUCHING THIS LINE: the
# compiler stops calling a master dead while this module keeps calling it dead (or vice
# versa), i.e. the orchestrator's blank-picture verdict silently grades on the OLD
# cutoff and the two repos disagree about the same file. Retune both, in the same wave.
#
# THE REAL FIX IS COMPILER-SIDE and is one key: publish
# `"dead_frame_min_ratio": dead_min_ratio` on the returned advisory dict (exactly as
# it already publishes `enforced_stages` for the tier mirror one block down), and
# `_dead_frame_cutoff` below becomes a pure read of the payload with this constant
# demoted to a pre-key fallback.
_DEAD_FRAME_MIN_RATIO = 0.10


def _number(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _violations(report: dict[str, object]) -> list[dict[str, object]]:
    """The compiler's per-violation list, or ``[]`` when this report predates it.

    Contract (compiler `MasteringGateViolation`, extended with `tier` by the wave
    that publishes the list on the advisory report)::

        {"stage": str, "metric": str, "value": float|None,
         "limit": float|None, "tier": "structural"|"tunable"}

    Everything here is defensive on purpose: a pod mid-roll emits the OLD report
    shape (no `violations` key at all), and a half-serialized list must degrade to
    the pre-key behaviour rather than crash or read as clean.
    """
    raw = report.get("violations")
    if not isinstance(raw, (list, tuple)):
        return []
    return [cast("dict[str, object]", v) for v in raw if isinstance(v, dict)]


def _dead_frame_cutoff(report: dict[str, object]) -> float:
    """The blank-picture cutoff THE COMPILER USED, when it says so; else the mirror.

    A `dead_frame` violation carries the compiler's live `render_dead_frame_min_ratio`
    in its `limit`. Preferring it means the quoted evidence and the comparison both
    use the producer's number instead of this module's copy of it whenever the
    producer states one — see the block above for why the copy cannot be retired yet.
    """
    for violation in _violations(report):
        if str(violation.get("stage")) == "dead_frame":
            limit = _number(violation.get("limit"))
            if limit is not None:
                return limit
    return _DEAD_FRAME_MIN_RATIO


def _blank_picture(report: dict[str, object]) -> bool:
    """True when the compiler's mastering report says the PICTURE is blank.

    Two signals, both the compiler's own: its ``dead_frame`` verdict in
    ``rerun_stages``, and ``dead_frame_ratio`` at or above its 10% cutoff.

    ⚠️ ``luma_spread`` is deliberately NOT a signal. It is the MIN over every frame
    of YMAX-YMIN (compiler ``mastering.py``), so ONE flat frame — a fade to black, a
    held title card — makes it 0.0 on a perfectly good film. The compiler retired it
    for exactly that reason in favour of ``dead_frame_ratio``. Live run
    ``01a08f66-7e66-712d-a80e-74ad92f62991``: ``dead_frame_ratio=0.009825`` (under 1%
    of frames flat), no ``dead_frame`` stage, ``luma_spread=0.0`` — a fine master that
    the old ``luma_spread <= 0.0`` disjunct stamped as BLANK.
    Deliberately NOT a general "gate failed" test: see the note at the call site.
    """
    raw = report.get("rerun_stages") or []
    stages = [str(s) for s in raw] if isinstance(raw, (list, tuple)) else []
    if "dead_frame" in stages:
        return True
    ratio = _number(report.get("dead_frame_ratio"))
    return ratio is not None and ratio >= _dead_frame_cutoff(report)


def _blank_evidence(report: dict[str, object]) -> str:
    """The metrics that justify a blank verdict — never contradicting evidence.

    ``dead_frame_ratio`` is what decided; ``luma_spread`` is quoted only as
    "reported, not used" so a reader cannot mistake a single flat frame for proof.
    """
    parts: list[str] = []
    if "dead_frame_ratio" in report:
        parts.append(
            f"dead_frame_ratio={report['dead_frame_ratio']} (cutoff {_dead_frame_cutoff(report)})"
        )
    if "luma_spread" in report:
        parts.append(f"luma_spread={report['luma_spread']} reported, not used")
    return ", ".join(parts)


# ── REMEDIABLE-BY-REMASTER ──────────────────────────────────────────────────
# A stage named in `rerun_stages` that a SECOND COMPILE (a new mastering block put
# through the canonical compile path) provably cannot move. Consumed by
# `demo_video/encode_master.py`, whose re-master loop costs 2 LLM calls PLUS a full
# real-ffmpeg recompile per pass (`_MAX_PASSES = 2`).
#
# ⛔ DENYLIST, NOT AN ALLOWLIST, and the direction is deliberate. An unknown or new
# compiler stage falls through as REMEDIABLE — i.e. today's behaviour, a recompile —
# so this change can only ever REMOVE a re-master the compiler itself says is futile,
# never silently withhold one from a stage nobody has classified yet. The unsafe
# direction would be the other one: a new remediable stage defaulting to "do not
# bother" loses a fix without saying so.
#
# Each entry is the compiler's OWN statement, not an inference of ours
# (`video_compiler_service/src/services/mastering.py`, read on origin/main):
#
# * `colorspace` — `_probe_colorspace`: a mis-tagged master stays "`colorspace_ok=False`
#   forever (re-running re-emitted the identical tag, so it never cleared)", and the
#   gate block calls it "an upstream TAGGING regression to fix at the source — it is
#   deliberately left gating, not widened to hide it". It fails on 23.3% of masters
#   (n=136 live), so this single entry is what stops roughly a third of all renders
#   buying a loop that cannot help. Measured cause and remedy are both UPSTREAM of the
#   mastering block, so no value the advisor can emit changes the probe's answer.
#
# * `cfr` — the ADVISORY-ONLY block: cadence is "REPORTED but NEVER gating ... because
#   no stage of this pipeline can remediate them", and `"cfr"` is nonetheless appended
#   to `rerun_stages` as a plain list append ("stages that MIGHT help"). It can never
#   clear `gate_pass`, so before this predicate it reached the loop only through the
#   `bool(critique.rerun_stages)` term — and bought a full recompile on its own. The
#   compiler marks the line to change "if cadence ever becomes remediable"; until it
#   does, this is the consumer honouring what that block already says.
#
# DELIBERATELY ABSENT, so the omissions are decisions and not oversights:
#   `loudness` — the compiler calls an in-range miss "the real, remediable finding",
#     and the advisor's `loudness_target` is exactly the knob that moves it.
#   `deband` — advisory compiler-side now, but the deband TRIGGER still runs per
#     compile and the advisor's `deband` knob still changes the picture.
#   `banding_measurement` — "the remediation is to MEASURE again", which is precisely
#     what a second compile does (it is a dead signalstats probe, not a bad master).
#   `dead_frame` — a blank render re-renders; nothing declares that futile.
#   `av_coverage` — always-enforced compiler-side, so it RAISES instead of returning a
#     report and cannot reach this loop at all. Listing it would assert a
#     remediability claim about a path that does not exist.
_UNREMEDIABLE_BY_REMASTER = frozenset({"colorspace", "cfr"})


def remediable_by_remaster(stages: list[str]) -> list[str]:
    """The subset of ``stages`` a re-master can actually act on, order preserved.

    Empty ⇒ every stage the compiler/critic named is one the compiler itself declares
    unfixable by re-running mastering, so a recompile is a provably futile spend. See
    :data:`_UNREMEDIABLE_BY_REMASTER` for the per-stage citation and for why an
    unclassified stage counts as remediable.
    """
    return [s for s in stages if s not in _UNREMEDIABLE_BY_REMASTER]


def mastering_gate_notes(mastering: object) -> list[str]:
    """Flag a FAILED mastering gate reported by the compiler's post-mux pass.

    The video-compiler sets ``gate_pass=False`` and lists the stages it wants re-run
    in ``rerun_stages`` whenever the delivered file fails a measured mastering check.
    ``"dead_frame"`` (flat luma — a blank/black/frozen master that still decodes to the
    right length, so the compile duration-guard never trips) is only ONE such reason,
    and it kept its own wording below. Every OTHER failed gate — e.g. run
    ``01a00b81-2576-71a5-becc-ffc4a843e6db``: ``gate_pass=false``,
    ``rerun_stages=["deband"]``, banding 1.0, ``duplicate_frame_ratio=0.338``,
    ``luma_spread=0.0`` — used to sail through completely unflagged because the guard
    matched the literal string ``"dead_frame"`` only. Nothing re-runs those stages, so
    the failed gate must at least be visible.

    Severity is DEGRADE, deliberately unchanged: we surface it through the SAME soft
    ``notes`` channel the compile duration-QA uses — a non-empty ``notes`` on a
    ``completed`` stage makes ``runs/_shared.py`` report the run as ``degraded``
    WITHOUT failing it. A failed mastering gate on an otherwise-playable, correct-length
    artifact is a QUALITY problem, not a corrupt artifact: the file exists and may be
    salvageable / re-run, so hard-failing here would destroy a paid deliverable over a
    banding score. Returns note strings (empty only when the report is PRESENT and
    reports a passing gate with nothing to re-run → never a false positive). An
    absent/malformed report yields :data:`_UNMEASURED_NOTE` instead of silence: the
    gate could not run, which is neither a pass nor a fail, and it must not be
    reported as a pass. Still only a note — never raises, never fails the run.
    """
    if not isinstance(mastering, dict):
        return [_UNMEASURED_NOTE]
    report = cast("dict[str, object]", mastering)
    raw_stages = report.get("rerun_stages") or []
    stages = [str(s) for s in raw_stages] if isinstance(raw_stages, (list, tuple)) else []
    if "dead_frame" in stages:
        evidence = _blank_evidence(report)
        return [
            "mastering: delivered master looks blank/dead"
            + (f" ({evidence})" if evidence else "")
            + "; degrading"
        ]
    # ⚠️ NARROWED after measuring production. Firing on bare `gate_pass is False` (or any
    # non-empty `rerun_stages`) sounded right and is NOISE: over 30 days, 207 of 214
    # masters carrying a report have `gate_pass=false` — **96.7%** — because
    # `banding_score` awards `smoothness * 0.25` to ANY held frame with luma range ≥ 60
    # (`video_compiler_service/.../mastering.py:107-127`), and `post_mux_verify` keeps the
    # film's WORST frame. One title card or freeze-pad tail therefore condemns a fine
    # master. A badge that lights on 96.7% of runs tells a reviewer nothing.
    #
    # What survives is the signal that is actually about the PICTURE being blank:
    # `dead_frame` or `dead_frame_ratio >= 0.10` — see `_blank_picture`. A floor-level
    # `luma_spread` used to count too and was WRONG: it is the min over all frames, so
    # one fade-to-black zeroes it (run 01a08f66: dead_frame_ratio=0.0098, spread=0.0).
    if not _blank_picture(report):
        return []
    # THE NOTE MAY STATE ONLY WHAT THIS BRANCH ESTABLISHED, and this branch is reached
    # on ONE observation: `_blank_picture` was true and `"dead_frame"` was NOT among the
    # re-run stages (that case returned above), i.e. `dead_frame_ratio` at or above the
    # compiler's 0.10 cutoff. It did NOT read `gate_pass`, so "gate FAILED" was never
    # established here — a report of `{"gate_pass": True, "rerun_stages": [],
    # "dead_frame_ratio": 0.25}` reaches this line — and it did not establish that
    # nothing was re-run either, since `stages` may be non-empty. Both claims shipped
    # anyway, side by side with the evidence denying them. Say the observation, then
    # quote the report's own verdict and list. The wording mirrors
    # `mastering_gate_degradations` so the two channels agree.
    #
    # ⚠️ MERGE NOTE: #985 wrote this sentence as "BLANK/dead by luma_spread" and quoted
    # `luma_spread` inline, because when it was written `luma_spread <= 0.0` was still
    # the deciding disjunct. It no longer is (see `_blank_picture`), so naming it as the
    # decider would reintroduce exactly the contradiction #985 came here to remove — a
    # note citing as proof a metric the branch did not consult. The honesty requirement
    # is unchanged and satisfied by naming the metric that DID decide: `_blank_evidence`
    # quotes `dead_frame_ratio` against its cutoff and marks `luma_spread`
    # reported-but-not-used.
    metrics = _blank_evidence(report)
    rerun = ", ".join(stages)
    rerun_clause = (
        f"the compiler asked to re-run {rerun}"
        if stages
        else "the compiler named no stage to re-run"
    )
    verdict_clause = (
        f"the report's own gate verdict is gate_pass={report['gate_pass']!r}"
        if "gate_pass" in report
        else "the report carries no gate_pass verdict"
    )
    return [
        f"mastering: the delivered master's picture is BLANK/dead; {verdict_clause} "
        f"and {rerun_clause} (rerun_stages={rerun or 'none reported'}"
        + (f"; {metrics}" if metrics else "")
        + "); degrading"
    ]


# ── STRUCTURAL vs TUNABLE-THRESHOLD ─────────────────────────────────────────
# The compiler already draws this line, in `video_compiler_service/src/services/
# mastering.py` above `_ALWAYS_ENFORCED_STAGES`:
#
#   "Every other check in this module is a quality JUDGEMENT — how much banding
#    is too much, how far off target loudness may drift — and reasonable
#    operators disagree, which is exactly what the advisory default exists for.
#    A master whose picture stops before its narration does not admit that
#    reading: it is a structurally broken artifact."
#
# That distinction is the fix for this channel's noise problem. Every violation
# the compiler raises writes its `stage` into `rerun_stages` (mastering.py 1300-
# 1382: colorspace, banding_measurement, deband, loudness, dead_frame,
# duplicate_frames, av_coverage), so `rerun_stages` IS the per-violation stage
# vocabulary and is the surface this tier is derived from — not a re-derivation
# from the metric values.
#
# ⚠️ WHAT THE ADVISORY REPORT DOES NOT CARRY, stated rather than guessed: the
# typed `MasteringGateViolation` list (which is where a per-violation tier would
# live) is attached ONLY to `MasteringGateError`, i.e. enforce mode. The returned
# advisory dict publishes `rerun_stages` and the metrics and NOTHING that says
# which of those stages the compiler considers always-enforced. So the tier
# cannot be read off the payload; the stage NAMES are the only shared vocabulary,
# and this set mirrors the compiler's intent by name.
#
# The drift risk is therefore one-directional and is the safe direction: an
# unknown/new stage falls through as ADVISORY, which is the compiler's own
# default (`_ALWAYS_ENFORCED_STAGES` is an explicit opt-in list, everything else
# is a judgement). A new always-enforced stage added compiler-side would be
# MISSED here until this set is updated — it can never produce a false positive.
# The clean fix is compiler-side and belongs in that repo: publish
# `"enforced_stages": sorted(_ALWAYS_ENFORCED_STAGES)` (or a per-violation tier)
# on the returned report, and this set becomes a read of the payload instead of a
# mirror of a name.
_STRUCTURAL_STAGES = frozenset({"av_coverage"})

# ⛔ `dead_frame` IS NOT IN THE SET ABOVE, AND MUST NOT BE ADDED. It is already armed
# in this tier by a different route: `_blank_picture` returns True whenever
# `"dead_frame"` appears in `rerun_stages`, and `_structural_reasons` then appends the
# `blank_picture` reason. Verified on `{"gate_pass": False, "rerun_stages":
# ["dead_frame"], "dead_frame_ratio": 0.44}`, which yields a structural degradation
# entry with the set exactly as it stands.
#
# Adding it would therefore NOT arm anything — it would DOUBLE-report one master, and
# the second report would be false: the per-stage sentence below is `av_coverage`'s
# ("the delivered picture does not cover the delivered narration"), which is not what a
# dead frame is. Measured output of that edit:
#   "... — dead_frame: the delivered picture does not cover the delivered narration;
#    blank_picture: the delivered master's picture is BLANK/dead (...)"
# Hence the explicit skip in `_structural_reasons` — the dedupe is enforced in code, not
# left to this comment, so a stage list arriving from the payload cannot reintroduce it.
_BLANK_PICTURE_STAGES = frozenset({"dead_frame"})

# Per-stage wording. The sentence used to be hardcoded for `av_coverage` inside the
# loop, which was safe only while the set could contain nothing else. Once the set can
# come from the COMPILER'S payload (`enforced_stages`, below), any stage name may reach
# that loop, and inheriting av_coverage's sentence would make this channel assert a
# specific structural failure it never observed. Unknown stages get an honest generic
# line naming the stage and nothing more.
_STRUCTURAL_REASON_TEXT: dict[str, str] = {
    "av_coverage": "the delivered picture does not cover the delivered narration",
}
_GENERIC_STRUCTURAL_REASON = "the compiler enforces this stage and the delivered master failed it"


def _structural_stages(report: dict[str, object]) -> frozenset[str]:
    """The enforced-stage set: the compiler's own list when it publishes one.

    Prefers the report's `enforced_stages` (the compiler-side fix described above —
    `sorted(_ALWAYS_ENFORCED_STAGES)` on the returned advisory dict) over the hardcoded
    mirror, so the two repos stop drifting by name.

    ⚠️ ABSENCE FALLS BACK TO THE MIRROR, NEVER TO THE EMPTY SET. Every master produced
    before that key ships carries no `enforced_stages`, and reading absence as "the
    compiler enforces nothing" would silently disarm this tier on the entire existing
    fleet — turning a missing key into a fleet-wide loss of the one structural signal
    this channel exists for. A malformed or EMPTY list takes the same fallback for the
    same reason: `[]` is far more likely to be a serialization accident than a
    deliberate "nothing is structural".
    """
    raw = report.get("enforced_stages")
    if isinstance(raw, (list, tuple, set, frozenset)):
        named = {str(x) for x in raw if str(x)}
        if named:
            return frozenset(named)
    return _STRUCTURAL_STAGES


# Metrics quoted in a STRUCTURAL degradation's detail, per structural reason. The
# detail must name what actually earned the entry, so the tunable metrics are
# deliberately absent: `banding_scores` and `duplicate_frame_ratio` say nothing
# about why this master is structurally broken, and a high duplicate ratio is
# INTENTIONAL DWELL in this pipeline, never a defect.
_STRUCTURAL_METRIC_KEYS: dict[str, tuple[str, ...]] = {
    "av_coverage": ("av_gap_seconds", "video_stream_seconds", "audio_stream_seconds"),
}


def _quote(report: dict[str, object], keys: tuple[str, ...]) -> str:
    return ", ".join(f"{k}={report[k]}" for k in keys if k in report)


def _tiered_violations(report: dict[str, object]) -> list[dict[str, object]] | None:
    """The per-violation list when it is USABLE as a tier source, else ``None``.

    ``None`` means "fall back to the name-based mirror", and every rejection below
    takes that branch deliberately:

    * **key absent** — every master rendered before the compiler publishes
      ``violations`` (and a pod mid-roll, which emits both shapes concurrently).
    * **no usable entry carries ``tier``** — which covers three shapes at once:
      a list from the pre-tier violation model
      (`stage`/`metric`/`value`/`limit`/`detail`), a malformed list whose entries
      are not mappings, and an EMPTY list (a `gate_pass=False` report naming no
      violation is self-contradictory — far likelier a serialization accident
      than a statement that nothing was violated). Reading any of them as "no
      structural tier present" would be a FALSE CLEAN.

    ⚠️ ONE PREDICATE, NOT TWO. An earlier draft had a separate ``if not
    violations: return None`` guard above this one. It was arithmetically
    unfalsifiable — ``any(...)`` over an empty list is already ``False``, so the
    empty case could never reach a different outcome, and a mutation deleting that
    guard changed no behaviour and failed no test. A guard nothing can falsify
    reads as protection and provides none, so the empty case is stated here and
    enforced by the predicate that actually decides it.

    A list that DOES carry tiers is authoritative: the compiler appends one entry
    per cleared check, so the absence of a ``structural`` entry there is a real
    statement that nothing structural was violated.
    """
    violations = _violations(report)
    if not any("tier" in v for v in violations):
        return None
    return violations


def _violation_reason(violation: dict[str, object]) -> str:
    """One structural reason, worded from the violation's OWN numbers.

    The compiler's ``detail`` is preferred when present — it is the producer's own
    sentence about the defect and cannot drift from what it measured. Otherwise the
    stage's known wording (or the honest generic line) plus metric/value/limit.

    ⚠️ The stage prefix is added only when the ``detail`` does not ALREADY open with
    it. Every ``detail`` the compiler writes today is formatted ``f"{stage}: …"``
    (`mastering.py`, e.g. ``"dead_frame: 0.44 of frames sit below …"``), so
    prepending unconditionally produced ``"deband: deband: banding 1.0"`` in the
    degradation a reviewer actually reads. The prefix is still applied to a detail
    that omits it, so the stage is never lost.
    """
    stage = str(violation.get("stage") or "unknown")
    detail = violation.get("detail")
    if isinstance(detail, str) and detail.strip():
        text = detail.strip()
        return text if text.startswith(f"{stage}:") else f"{stage}: {text}"
    text = _STRUCTURAL_REASON_TEXT.get(stage, _GENERIC_STRUCTURAL_REASON)
    metric = violation.get("metric")
    value = _number(violation.get("value"))
    limit = _number(violation.get("limit"))
    parts = [
        f"{metric}={value}" if metric and value is not None else None,
        f"limit={limit}" if limit is not None else None,
    ]
    quoted = ", ".join(p for p in parts if p)
    return f"{stage}: {text}" + (f" ({quoted})" if quoted else "")


def _structural_reasons(report: dict[str, object], stages: list[str]) -> list[str]:
    """The structural violations on this report, each naming its own metrics.

    Two, and only two, admit no advisory reading:

    * ``av_coverage`` — the delivered picture stops before the delivered
      narration. The compiler's own always-enforced stage.
    * a BLANK/dead picture — :func:`_blank_picture`, i.e. the compiler's
      ``dead_frame`` verdict OR ``dead_frame_ratio`` >= its cutoff. There is no
      threshold at which shipping a black film is the right call.

    Everything else in ``rerun_stages`` (deband, duplicate_frames, colorspace,
    loudness) is a tunable ceiling and is left to the mastering report, which
    already carries it for anyone who wants it. That is what advisory means.

    TWO SOURCES, in order. When the report carries a usable per-violation list
    (:func:`_tiered_violations`) the tier is READ from it — ``tier ==
    "structural"`` — so this channel is no longer limited to the one name the
    mirror below knows. Otherwise the pre-``violations`` behaviour stands
    unchanged: ``enforced_stages``/``_STRUCTURAL_STAGES`` matched against
    ``rerun_stages``. The blank-picture reason is appended on BOTH paths, so an
    old-shape report keeps exactly the output it has today.
    """
    reasons: list[str] = []
    tiered = _tiered_violations(report)
    if tiered is not None:
        # ── PREFERRED PATH: the compiler's OWN per-violation tier ──────────────
        # The name-matching path below can only ever match `av_coverage` (the sole
        # member of the compiler's `_ALWAYS_ENFORCED_STAGES`) — and the compiler
        # RAISES `MasteringGateError` on an always-enforced stage in BOTH modes, so
        # a DELIVERED advisory report can never carry `av_coverage` in
        # `rerun_stages` at all. That made this whole tier reduce to
        # "`_blank_picture`", i.e. `MASTERING_GATE_FAILED` meant only "absent report
        # OR dead frame". Reading `tier == "structural"` off the payload is what
        # makes the tier able to name anything the compiler actually enforces,
        # without this module guessing from stage names.
        for violation in tiered:
            if str(violation.get("tier")) != "structural":
                continue
            # Same dedupe as the name path: `_blank_picture` reports these below
            # with the correct wording and its own evidence.
            if str(violation.get("stage")) in _BLANK_PICTURE_STAGES:
                continue
            reasons.append(_violation_reason(violation))
        if _blank_picture(report):
            metrics = _blank_evidence(report)
            reasons.append(
                "blank_picture: the delivered master's picture is BLANK/dead"
                + (f" ({metrics})" if metrics else "")
            )
        return reasons
    enforced = _structural_stages(report)
    for stage in stages:
        # `_BLANK_PICTURE_STAGES` is skipped here even when the compiler names it
        # enforced: `_blank_picture` below already reports it, with the correct
        # wording and its own evidence. Without this the same master would carry two
        # reasons for one defect.
        if stage in enforced and stage not in _BLANK_PICTURE_STAGES:
            metrics = _quote(report, _STRUCTURAL_METRIC_KEYS.get(stage, ()))
            text = _STRUCTURAL_REASON_TEXT.get(stage, _GENERIC_STRUCTURAL_REASON)
            reasons.append(f"{stage}: {text}" + (f" ({metrics})" if metrics else ""))
    if _blank_picture(report):
        metrics = _blank_evidence(report)
        reasons.append(
            "blank_picture: the delivered master's picture is BLANK/dead"
            + (f" ({metrics})" if metrics else "")
        )
    return reasons


def mastering_gate_degradations(mastering: object) -> list[Degradation]:
    """A STRUCTURALLY failed mastering gate, as a structured degradation entry.

    ⚠️ This function is NOT "every ``gate_pass=false``", and the reason is the
    same one that forced ``mastering_gate_notes`` to narrow. 207 of 214 masters
    over 30 days carry ``gate_pass=false`` — 96.7% — and the config's own comment
    says the deband ceiling of 0.15 fires on 96.7% of masters. That is a
    MISCALIBRATED THRESHOLD, not 96.7% genuinely broken films. Recording all of
    them would make ``degraded`` near-universal on render runs, and a signal that
    lights on everything is ignored — which recreates the exact defect this
    channel exists to fix.

    So the entry is gated on the compiler's OWN structural-vs-judgement line (see
    :data:`_STRUCTURAL_STAGES`), not on the bare gate verdict:

    * **Structural** violations (``av_coverage``, a blank/dead picture) ⇒ ONE
      entry, whose ``detail`` names only the structural metric(s) that earned it.
    * **Tunable-threshold** violations alone (deband, duplicate_frames,
      colorspace, loudness drift) ⇒ NO entry. They stay in the mastering report,
      which is where an advisory number belongs.
    * Both ⇒ one entry, describing the structural half only.

    An ABSENT or malformed report still records, and that is neither tier: the
    gate did not RUN, so nothing measured this master. "An absent measurement
    reported as a passing one" is this codebase's #1 defect class, and it is
    rare — unlike a tunable breach — so recording it costs no noise.

    Returns ``[]`` for a passing gate AND for a gate that failed only on tunable
    ceilings. Never raises, never fails a stage.
    """
    if not isinstance(mastering, dict):
        return [degradation(MASTERING_GATE_FAILED, _UNMEASURED_NOTE)]
    report = cast("dict[str, object]", mastering)
    raw_stages = report.get("rerun_stages") or []
    stages = [str(s) for s in raw_stages] if isinstance(raw_stages, (list, tuple)) else []
    # `is False`, not falsiness: an ABSENT `gate_pass` is not a reported failure
    # (that case is carried by `rerun_stages`), and `gate_pass: 0` from a
    # hand-written payload should still read as a failure — which `is False`
    # misses, so the `rerun_stages` arm is what covers it.
    # A non-empty `violations` list is itself a reported failure: the compiler appends
    # one entry per check that CLEARED `gate_pass`. Included here so a payload that
    # carries violations but (through a serialization quirk) neither `gate_pass=False`
    # nor `rerun_stages` cannot return early as clean.
    if report.get("gate_pass") is not False and not stages and not _violations(report):
        return []
    reasons = _structural_reasons(report, stages)
    if not reasons:
        # THE ANTI-NOISE GUARD. The gate failed, and every violation behind it is
        # a tunable ceiling reasonable operators disagree about. The verdict and
        # the numbers are both already on the persisted mastering report; this
        # channel stays quiet so the entries it DOES carry keep meaning something.
        return []
    # THE CLAIM MUST MATCH THE LIST. This branch is reached with `stages` EMPTY (a
    # `gate_pass=False` report that named no re-run) AND with `stages` NON-EMPTY (the
    # compiler asked for re-runs, and at least one of them is structural). The old
    # wording asserted "no stage was re-run" unconditionally, so it could print that
    # sentence directly alongside `rerun_stages=deband` — a degradation entry
    # contradicting its own evidence. Say only what `stages` establishes.
    rerun = ", ".join(stages)
    rerun_clause = (
        f"the compiler asked to re-run {rerun}"
        if stages
        else "the compiler named no stage to re-run"
    )
    return [
        degradation(
            MASTERING_GATE_FAILED,
            f"the compiler's mastering gate failed STRUCTURALLY and {rerun_clause} "
            f"— {'; '.join(reasons)} (rerun_stages={rerun or 'none reported'})",
        )
    ]


def mastering_gate_advisories(mastering: object) -> list[Degradation]:
    """The compiler's ``MasteringReport.advisories``, one ADVISORY entry each.

    Contract (``video_compiler_service`` ``MasteringReport.advisories``, same five keys
    as a violation)::

        {"stage": str, "metric": str, "value": float|None,
         "limit": float|None, "tier": "structural"|"tunable"}

    ⛔ Deliberately NOT routed through :func:`_violations` /
    :func:`mastering_gate_degradations`: the compiler keeps these OUT of
    ``violations[]`` because every entry there is a check that cleared ``gate_pass``.
    They land on the NON-degrading ``advisories`` field, never on ``notes`` (a note
    flips ``degraded``).

    ABSENT ⇒ ``[]``, exactly today's behaviour: the currently-deployed compiler and
    every older report carry no ``advisories`` key, and an absent/malformed REPORT is
    already recorded as UNMEASURED by :func:`mastering_gate_degradations` — recording
    it twice here would double count. Non-dict entries are dropped. Never raises.
    """
    if not isinstance(mastering, dict):
        return []
    raw = cast("dict[str, object]", mastering).get("advisories")
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[Degradation] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        item = cast("dict[str, object]", entry)
        out.append(
            advisory(
                MASTERING_ADVISORY,
                f"the compiler measured {item.get('stage')!s} breached and did not gate it: "
                f"{item.get('metric')!s}={item.get('value')!r} against limit "
                f"{item.get('limit')!r} (tier={item.get('tier')!s})",
            )
        )
    return out
