"""demo_video_compile — join narration + recorded interaction clips → MP4.

One storyboard shot per ``demo_video_gen`` line (the scenario text is the
text-card fallback). Narration audio joins by ``line_id``; the recorded scenario
clip (``demo_video_record``) is the shot's ``video_url``, keyed 1:1 by the same
``clip-<n>`` id. ``assemble_shots`` reconciles each shot's duration to its
narration clip; the video-compiler then freezes the recording's last frame when
the narration runs longer (``renderer._freeze_pad_suffix``) and trims it when the
narration is shorter — so a fixed-length interaction recording always fills its
narrated beat. Recorded clips keep the default ``contain`` fit (no crop) so the
full UI stays visible regardless of the recording's viewport aspect.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, cast

from src.services.captions import caption_text_by_line
from src.services.stages.base import (
    SLOT_USE_OPTIONAL,
    ArtifactUrl,
    Degradation,
    SlotUse,
    StageInput,
    StageOutputWithNotes,
    StageSkippedError,
)
from src.services.stages.demo_video.reconcile_timeline import timeline_bounds_were_ordered
from src.services.stages.demo_video.spine import resolve_plan_output
from src.services.stages.primitives.compile_stage import (
    CompileStageBase,
    ShotDict,
    captions_mode,
)
from src.services.stages.shared.accessibility import build_a11y_policy
from src.services.stages.shared.authored_storyboard import authored_lines
from src.services.stages.shared.captions_only import (
    captions_only,
    reading_seconds,
    static_caption_cues,
)
from src.services.stages.shared.cold_open import _peak_motion_lead, cold_open_window
from src.services.stages.shared.compute_plan import transition_gain_db
from src.services.stages.shared.duration_conformance import demo_video_ordered_seconds
from src.services.stages.shared.footage_plans import (
    authored_split_plans,
    pin_authored_splits,
    resolve_footage_plans,
)
from src.services.stages.shared.sound_design import (
    resolve_audio_bed,
    sfx_by_line,
    sound_config,
    transition_sfx_id,
)
from src.services.stages.shared.time_budget import has_authored_clip_window
from src.services.stages.shared.title_cards import TitleCardPlan, plan_title_cards
from src.services.stages.shot_assembly import assemble_shots, normalize_footage_viewport
from src.services.visual_media import (
    collect_motion_slides_by_line,
    is_fetchable_url,
    overlay_motion_slides,
)
from src.services.word_timed_captions import ClipRef, ShotCaptionsPayload

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

log = logging.getLogger(__name__)


# One reconciled plan line as `reconcile_timeline` emits it. The three extractors below
# each pull a single field out of the same shape, so they share a name for it rather than
# spelling `dict[str, Any]` three times — untyped payloads are ratcheted repo-wide.
_PlanLine = dict[str, Any]


def _shot_seconds(line: _PlanLine, *, silent: bool) -> float:
    """The storyboard duration of one shot. The authored ``seconds``, except that a
    captions-only beat with none (nothing narrated to measure) takes its reading time."""
    seconds = float(line.get("seconds") or 0.0)
    if silent and seconds <= 0 and line.get("text"):
        return reading_seconds(str(line["text"]))
    return seconds


def _seconds_by_line(lines: list[_PlanLine]) -> dict[str, float]:
    """Per-line duration allotment from the plan (reconcile_timeline's ``seconds``).

    Fills a shot's duration only when no narration clip fixed it (see
    ``shot_assembly._reconcile_seconds``). A line with no/zero ``seconds`` is left
    out entirely so the shot keeps today's storyboard-derived value.
    """
    return {str(line["id"]): float(line["seconds"]) for line in lines if line.get("seconds")}


def _hold_seconds_by_line(lines: list[_PlanLine]) -> dict[str, float]:
    """Per-line underspoken hold from the plan (reconcile_timeline's ``hold_seconds``).

    When a beat's authored words are too few for its allotment, ``vo_rate_for`` clamps
    the rate at the natural floor and the beat BY CONSTRUCTION still ends short of the
    allotment. This is that unfilled remainder, and it makes the PICTURE hold rather
    than the SPEECH stretch further — the owner's "narration weirdly slows down" and the
    short master were the same event.

    A line with no/zero ``hold_seconds`` is left out entirely, so a plan that never
    underspoke yields ``{}`` and the assembly is byte-identical to before
    (``assemble_shots`` treats ``{}`` exactly like ``None``).
    """
    return {
        str(line["id"]): float(line["hold_seconds"]) for line in lines if line.get("hold_seconds")
    }


def _min_max_by_line(lines: list[_PlanLine]) -> dict[str, tuple[float, float]]:
    """Per-line ``(min, max)`` window from the plan (reconcile_timeline's bounds).

    ONLY the bounds reconcile_timeline actually stamped — a missing bound is never
    defaulted or synthesised, because ``shot_assembly`` clamps unconditionally, even
    over a narrated duration. Footage beats floor at their measured clip length
    (``min == clip``), so the clamp can never trim them (C0). Their ``max`` is
    ``clip + freeze_pad`` (Step B): the extra headroom lets the renderer hold the
    freeze-pad at the ceiling, and since a footage shot's reconciled duration starts
    at the clip length it never stretches up to that ``max`` on its own.
    """
    return {
        str(line["id"]): (float(line["min_seconds"]), float(line["max_seconds"]))
        for line in lines
        if line.get("min_seconds") is not None and line.get("max_seconds") is not None
    }


def _unbounded_footage_beats(
    lines: list[_PlanLine],
    clips_by_line: Mapping[str, Mapping[str, Any]],
    bounds_by_line: Mapping[str, tuple[float, float]],
) -> list[tuple[str, float]]:
    """Beats that have RECORDED FOOTAGE but no ``[min, max]`` window, and what each one
    is about to lose — ``(line_id, measured clip seconds)``, in spine order.

    This is the impossible state, expressed WITHOUT asking what reconcile_timeline did.
    A beat backed by a measured clip has a length that was paid for and probed; nothing
    downstream can re-derive it, so with no window ``shot_assembly`` lets the narration
    clip declare the beat's duration in BOTH directions and the paid footage is simply
    gone. A beat with NO measured clip (a still, an image beat, a skip-cleaned capture)
    is legitimately unbounded and is not reported.

    The join is the SAME one ``video_by_line`` uses — ``line["clip"]`` when authored,
    else the line's own id — so several dialogue lines sharing one recording are each
    checked against the clip they actually play over.

    ⛔ PER-LINE, never all-or-nothing. ``_min_max_by_line`` drops a line when EITHER
    bound is missing and reconcile stamps per-line, so the common shape is PARTIAL
    loss: 4 of 5 beats losing their window still leaves ``bounds_by_line`` truthy, and
    a guard that asked "is the map empty?" reported nothing on a film 9.1s short. The
    measured 7.6s incident was the total-loss corner of the same defect, not its shape.

    ``seconds`` here is ``DemoVideoClip.seconds`` — the probed length of the FINALIZED
    clip (``record._stamp_durations``). A clip whose length is missing or zero was never
    measured, so it is not evidence of anything and is skipped: a probe that cannot tell
    "absent" from "zero" is how this defect shipped in the first place.
    """
    unbounded: list[tuple[str, float]] = []
    for line in lines:
        line_id = str(line["id"])
        clip = clips_by_line.get(str(line.get("clip") or line_id))
        if not isinstance(clip, Mapping):
            continue
        try:
            measured = float(clip.get("seconds") or 0.0)
        except TypeError, ValueError:
            continue
        if measured > 0.0 and line_id not in bounds_by_line:
            unbounded.append((line_id, measured))
    return unbounded


def _cold_open_active(open_style: object, hook_block: Mapping[str, object] | None) -> bool:
    """Whether the deterministic cold-open pass claims the opening shot: an explicit
    ``open_style == "cold_open"`` or a present (non-``None``) ``direction.hook`` block.

    An explicit ``hook.mode == "off"`` force-off ALWAYS wins and makes the gate
    inactive — regardless of ``open_style`` — so both callers agree the opener is not
    a cold open: the opener-framing pass (``_resolve_cold_open``) stays inert AND the
    build-path anchor-deferral (``_resolve_clip_anchor``) does NOT defer the opener,
    letting it keep its own motion-settled ``clip_anchor: "auto"`` lead instead of
    losing it to a cold-open pass that has bailed.

    Single source for the activation gate, shared by ``_resolve_cold_open`` (which
    frames the opener) and ``_resolve_clip_anchor`` (which defers the opener to it) so
    the two can never drift.
    """
    if hook_block is not None and hook_block.get("mode") == "off":
        return False
    return open_style == "cold_open" or hook_block is not None


def _usable_out_point(clip: object) -> float | None:
    """The OUT-point that keeps a TRIMMED clip's dead tail off screen, or ``None``.

    ``demo_video_record._trim_failed_scenario_tails`` lowers ``seconds`` below the
    file length (kept on ``recorded_seconds``) when a failed scenario ends in its
    timeout wait. Every orchestrator reader honours ``seconds``, but the compiler
    derives its seeks from the FILE (``video_compiler_service``
    ``renderer._clip_seek_args``: ``clip_len = _ffprobe_duration(clip)``; ``'end'``
    seeks to ``clip_len - shot``, a lead clamps to ``clip_len - shot``, a pan to a
    fraction of it) — so without an out-point a trimmed clip would be seeked INTO the
    frozen tail. ``Shot.clip_out_seconds`` is the compiler's source-timeline
    out-point, honoured under both anchors and bounded with an input-side ``-t``.

    ``None`` for an untrimmed clip, so every other shot stays byte-identical.
    """
    if not isinstance(clip, Mapping):
        return None
    seconds, recorded = clip.get("seconds"), clip.get("recorded_seconds")
    if not all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in (seconds, recorded)
    ):
        return None
    usable, full = float(cast("float", seconds)), float(cast("float", recorded))
    return usable if 0.0 < usable < full else None


def _bound_window_to_usable(
    clip: object,
    anchor: str | None,
    lead: float | None,
    out: float | None,
    shot_seconds: float,
) -> tuple[str | None, float | None, float | None]:
    """Close a trimmed clip's window at its usable end; untrimmed ⇒ inputs unchanged.

    Setting ``clip_out_seconds`` switches the compiler into WINDOW mode, whose lead
    rules differ (``renderer._clip_seek_args``): the lead is taken verbatim, is a FLOOR
    under ``'end'``, and ``out <= lead`` is rejected by the request schema. So the lead
    is re-expressed to mean what it meant before the out-point existed:

    * an author's own out-point is kept when it is earlier; a later one is cut to usable;
    * under ``'end'`` with no authored window the compiler IGNORED the lead ⇒ dropped;
    * otherwise it is clamped to ``out - shot_seconds`` exactly as the compiler clamps
      a lead to ``clip_len - shot`` — and never left ``>= out``.
    """
    usable = _usable_out_point(clip)
    if usable is None:
        return anchor, lead, out
    authored_window = out is not None
    bounded_out = min(out, usable) if out is not None else usable
    if lead:
        if anchor == "end" and not authored_window:
            lead = None
        elif not authored_window or lead >= bounded_out:
            lead = min(lead, max(bounded_out - shot_seconds, 0.0))
        if lead is not None and lead >= bounded_out:
            lead = None
    return anchor, lead, bounded_out


def _footage_out_points(plan_output: Mapping[str, object]) -> dict[str, float]:
    """reconcile_timeline's per-beat footage OUT-points, keyed by beat (= line) id.

    reconcile writes ``footage_out_point_seconds`` onto ``timeline_manifest.beats[]``
    (``_stamp_footage_out_points``) — NOT onto ``lines[]``, whose model does not carry
    it. So that manifest is the one place this is read from. ``timeline_manifest`` is
    ``None`` when nothing was trimmed and relief did not run ⇒ ``{}`` ⇒ no cuts.
    A beat whose value is absent / null / non-numeric / non-positive is left out
    (= untrimmed), so it compiles exactly as before.

    UNITS: the value is a FILE-TIME out-point (seconds from the recording's head — the
    axis ``clip_out_seconds`` uses). reconcile only trims beats whose window starts at
    file time 0, so it equals the trimmed duration there. A beat whose line AUTHORS its
    window (``time_budget.has_authored_clip_window``) is dropped here too, even if an
    older manifest stamped one: a cut measured from 0 would land inside the author's
    window (10-15s of a 20s recording "cut to 12" played 10-12s, then freeze-held).
    """
    manifest = plan_output.get("timeline_manifest")
    if not isinstance(manifest, Mapping):
        return {}
    beats = manifest.get("beats")
    if not isinstance(beats, list):
        return {}
    lines = plan_output.get("lines")
    windowed = {
        str(line.get("id"))
        for line in (lines if isinstance(lines, list) else [])
        if isinstance(line, Mapping) and has_authored_clip_window(line)
    }
    points: dict[str, float] = {}
    for beat in beats:
        if not isinstance(beat, Mapping) or beat.get("id") is None:
            continue
        if str(beat["id"]) in windowed:
            continue
        raw = beat.get("footage_out_point_seconds")
        if not isinstance(raw, (int, float)) or isinstance(raw, bool) or raw <= 0.0:
            continue
        points[str(beat["id"])] = float(raw)
    return points


def _floor_out_points_at_measured_narration(
    out_points: Mapping[str, float], narrate: Mapping[str, object]
) -> dict[str, float]:
    """Never cut a beat's footage before its MEASURED narration ends.

    reconcile's trim floor is ``max(narration, 60% clip, 2.0s)``, but it runs BEFORE
    narrate, so its "narration" is ``words / wpm * 60`` at ONE pooled rate. A line spoken
    slower than that pool is cut under its own voice. Run ``01a11610-6ada`` (P4): the
    script sample measured 76 words in 28.12s ⇒ 162.14 wpm, clip-4's 25 words ⇒ an
    out-point of 9.251s — while that line, synthesized alone, ran 9.454s. The compiler
    then held a frozen frame over the last 0.2s of narration that real footage existed
    for. Here the narration IS measured, so each out-point is raised to it; a raised
    point at or past the clip's usable end is a no-op in :func:`_clip_cut_at` (the clip
    plays untrimmed, never extended). Beats without an out-point are untouched.
    """
    clips = narrate.get("clips")
    measured: dict[str, float] = {}
    for clip in clips if isinstance(clips, list) else []:
        if not isinstance(clip, Mapping) or clip.get("line_id") is None:
            continue
        raw = clip.get("seconds")
        if isinstance(raw, (int, float)) and not isinstance(raw, bool) and math.isfinite(raw):
            lid = str(clip["line_id"])
            measured[lid] = max(measured.get(lid, 0.0), float(raw))
    floored = dict(out_points)
    for lid, point in out_points.items():
        spoken = measured.get(lid, 0.0)
        if spoken > point:
            log.warning(
                "demo_video_compile: beat %s footage out-point %.3fs is before its "
                "measured narration end %.3fs — raising the cut to the narration",
                lid,
                point,
                spoken,
            )
            floored[lid] = spoken
    return floored


def _clip_cut_at(
    clip: Mapping[str, Any] | None, out_point: float | None
) -> Mapping[str, Any] | None:
    """``clip`` as seen by ONE beat whose footage is cut at ``out_point``.

    Rides the EXISTING record-trim path rather than a second mechanism: a cut is
    expressed exactly like ``demo_video_record._trim_failed_scenario_tails`` — the
    usable ``seconds`` lowered below the file length (``recorded_seconds``) — so
    ``_usable_out_point`` / ``_bound_window_to_usable`` close the window at the cut
    and re-express any lead against it, and the motion-peak clamp and cold-open window
    see the shortened usable length. An earlier record trim wins (``min``); a cut at
    or past the usable end is a no-op. ``None`` / not a mapping ⇒ ``clip`` unchanged
    (the same object), so beats without a cut stay byte-identical.

    The cut is per BEAT, not per clip: several lines can share one recording, so the
    shared ``clips_by_line`` entry is never mutated.
    """
    if out_point is None or clip is None:
        return clip

    def _num(v: object) -> float | None:
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    seconds, recorded = _num(clip.get("seconds")), _num(clip.get("recorded_seconds"))
    # The FILE length the compiler seeks against: the record-trim's recorded length,
    # else the measured length, else unknown (still cut — the compiler bounds the
    # window with an input-side ``-t``, so an unmeasured file is cut all the same).
    full = recorded if recorded is not None else seconds
    usable = seconds if seconds is not None and seconds > 0.0 else full
    if usable is not None and out_point >= usable:
        return clip
    return {
        **clip,
        "seconds": out_point,
        "recorded_seconds": full if full is not None and full > 0.0 else math.inf,
    }


def _resolve_clip_anchor(
    line: Mapping[str, object],
    clips_by_line: Mapping[str, object],
    idx: int,
    cold_open_active: bool,
    *,
    generated: bool = False,
    shot_seconds: float = 0.0,
) -> tuple[str | None, float | None]:
    """Resolve a per-line clip_anchor. ``"auto"`` → compute the motion-settled
    lead and emit ``("start", lead)``; any explicit/absent value passes through
    unchanged (author wins, byte-identical). The opener is left to cold_open when
    active (it is motion-peak AND hook-aware, strictly richer).

    ``generated`` (a model-generated spine, ``authored_lines(brief) is None``): a line
    with NO anchor and NO lead had no author to choose one, and the compiler would
    otherwise start it at lead 0 — the recording's page load. It is resolved like
    ``"auto"``, falling back to that head when the curve has no distinguished peak, and
    the lead is clamped so ``lead + shot_seconds`` never passes the clip's ``seconds``
    (its USABLE end — a record-trimmed clip's ``seconds`` is already cut to it)."""
    raw = line.get("clip_anchor")
    default_auto = generated and raw is None and line.get("clip_lead_seconds") is None
    if raw != "auto" and not default_auto:
        lead = line.get("clip_lead_seconds")
        return (
            raw if isinstance(raw, str) else None,
            float(lead) if isinstance(lead, (int, float)) and not isinstance(lead, bool) else None,
        )
    if idx == 0 and cold_open_active:
        return (None, None)
    clip_raw = clips_by_line.get(str(line.get("clip") or line.get("id")))
    clip: Mapping[str, object] = (
        cast("Mapping[str, object]", clip_raw) if isinstance(clip_raw, Mapping) else {}
    )
    cs_raw = clip.get("seconds")
    cs = float(cs_raw) if isinstance(cs_raw, (int, float)) and not isinstance(cs_raw, bool) else 0.0
    mt = clip.get("motion_timeline")
    # `_peak_motion_lead` coerces each sample defensively; the cast just tells ty the
    # narrowed list carries motion-sample mappings (the value came off an `object` map).
    timeline = cast("Sequence[Mapping[str, object]]", mt) if isinstance(mt, list) else None
    lead = _peak_motion_lead(timeline, cs)
    if lead is not None and default_auto:
        lead = round(min(lead, max(cs - shot_seconds, 0.0)), 3)
        if lead <= 0.0:
            # Clamped to the head: identical to the unanchored default, so emit that.
            return (None, None)
    return ("start", lead) if lead is not None else (None, None)


class DemoVideoCompileInput(StageInput):
    pass


class DemoVideoCompileOutput(StageOutputWithNotes):
    # Declared output contract — enforced once, centrally, in
    # `runs/dag_executor.py::_enforce_produces`. This is a TERMINAL artifact
    # producer: `artifact_url` IS the deliverable, not a hint about it, so an
    # empty one is never an acceptable outcome. Hence `produces_required` — the
    # stage FAILS instead of degrading. Degrade was the wrong default here: it let
    # the run finish while handing the user a reference to nothing, which is the
    # exact defect class this contract exists to close.
    produces: ClassVar[tuple[str, ...]] = ("artifact_url",)
    produces_required: ClassVar[bool] = True

    artifact_url: ArtifactUrl
    duration_seconds: float
    # Audit-only enrichment surfaced by render_manifest (A20). The compiler
    # always returns a post-mux `mastering` report; `output_sha256` is the content
    # hash of the delivered file (absent on older compiler builds → None). Neither
    # affects the render — they only feed the manifest's audit fields.
    mastering: dict[str, object] | None = None
    output_sha256: str | None = None
    # Post-mux measured loudness (A11) threaded through for the advisory ship gate
    # (`final_critique`), which pass-reads it rather than re-measuring. Absent on
    # older compiler builds → None; it never affects the render.
    loudness: dict | None = None
    # Authoritative per-beat [start,end) rendered timeline (A11 momentum consume),
    # emitted by the compiler ONLY when `_emit_shot_timeline` opted in; absent /
    # probe-free ⇒ None. Audit/ship-gate advisory only — never affects the render.
    shot_timeline: list[dict[str, object]] | None = None


class DemoVideoCompileStage(CompileStageBase[DemoVideoCompileInput, DemoVideoCompileOutput]):
    name = "demo_video_compile"
    input_model = DemoVideoCompileInput
    output_model = DemoVideoCompileOutput
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (
        SlotUse("llm"),
        SlotUse("tts"),
        SlotUse("asr", when=SLOT_USE_OPTIONAL),
    )

    def build_shots(self, ctx: RunContext) -> list[dict[str, Any]]:
        # Video-capture opted out (`record.enabled: false`) ⇒ demo_video_record
        # skip-cleaned, so there is no recorded footage to compile. Skip clean here
        # too rather than hard-failing on the missing `record` output below — this
        # makes the plan-only opt-out safe UNCONDITIONALLY (not only when the run
        # also happens to pause before this stage). Mirrors the same gate in
        # demo_video_record.handle(); downstream footage_critique/encode_master
        # already tolerate a skipped compile (`... or {}` + `if not url`).
        if (ctx.config_snapshot.get("record") or {}).get("enabled") is False:
            raise StageSkippedError(
                "demo_video_compile: record.enabled=false — video-capture branch "
                "opted out; no footage to compile"
            )

        # Line TEXT/description spine: doctored output when script_doctor ran, else
        # demo_video_gen (skip-clean). Recorded footage still joins by the same ids.
        gen = resolve_plan_output(ctx)
        narrate = ctx.prior_output("demo_video_narrate")
        record = ctx.prior_output("demo_video_record")
        if not (gen and narrate and record):
            raise RuntimeError(
                "demo_video_compile: demo_video_gen + demo_video_narrate + "
                "demo_video_record required"
            )

        lines = gen["lines"]
        out_points = _floor_out_points_at_measured_narration(_footage_out_points(gen), narrate)
        # Inputs the C4 content-aware anchor needs: the per-clip record (its measured
        # `motion_timeline`) keyed by line, and whether the deterministic cold-open pass
        # owns the opener — computed from the SAME gate `_resolve_cold_open` activates on
        # so a `clip_anchor:"auto"` on line 0 is deferred to that richer pass, not
        # double-stamped here.
        clips_by_line = {str(c["line_id"]): c for c in record.get("clips", [])}
        open_style = ctx.config_snapshot.get("open_style")
        hook = (ctx.config_snapshot.get("direction") or {}).get("hook")
        hook_block = hook if isinstance(hook, dict) and hook else None
        cold_open_active = _cold_open_active(open_style, hook_block)
        # Same generated-vs-authored test narrate and reconcile_timeline use. Only a
        # GENERATED spine's unanchored beats get the motion-peak start by default.
        generated_spine = authored_lines(ctx.config_snapshot.get("brief") or {}) is None

        # brief.captions_only: no narration clip will fix a shot's length, so an
        # unauthored beat gets the time its caption needs to be read (see `_shot_seconds`).
        silent = captions_only(ctx)
        storyboard_shots: list[dict[str, object]] = []
        for idx, line in enumerate(lines):
            # reconcile_timeline's per-beat footage cut (`timeline_manifest.beats[]
            # .footage_out_point_seconds`), applied through the record-trim path: this beat sees its clip as if the
            # recording ended at the cut. Absent/null ⇒ the clip itself, unchanged.
            clip_key = str(line.get("clip") or line["id"])
            beat_clip = _clip_cut_at(clips_by_line.get(clip_key), out_points.get(str(line["id"])))
            beat_clips = (
                clips_by_line
                if beat_clip is clips_by_line.get(clip_key)
                else {**clips_by_line, clip_key: beat_clip}
            )
            # Per-line footage framing (compositor Shot.clip_*). Author wins: an
            # explicit/absent anchor passes through byte-identically; only
            # `clip_anchor:"auto"` is resolved to the motion-settled lead (C4).
            anchor, lead = _resolve_clip_anchor(
                line,
                beat_clips,
                idx,
                cold_open_active,
                generated=generated_spine,
                shot_seconds=float(line.get("seconds") or 0.0),
            )
            raw_out = line.get("clip_out_seconds")
            anchor, lead, clip_out = _bound_window_to_usable(
                beat_clip,
                anchor,
                lead,
                float(raw_out) if isinstance(raw_out, (int, float)) and raw_out else None,
                float(line.get("seconds") or 0.0),
            )
            storyboard_shots.append(
                {
                    "id": str(line["id"]),
                    "line_id": str(line["id"]),
                    "seconds": _shot_seconds(line, silent=silent),
                    "description": str(line.get("text") or ""),
                    "clip_anchor": anchor,
                    "clip_lead_seconds": lead,
                    # The window's OUT-point is a pure author control — there is no
                    # "auto" resolution for it (the motion curve localises a moment to
                    # start on, not one to stop on), so it passes straight through
                    # like `caption`/`fit`. None unless authored ⇒ assemble_shots omits
                    # the key ⇒ byte-identical. Rebuilding this spine dict field-by-field
                    # means an unlisted field is SILENTLY DROPPED, which is why it has to
                    # be named here and not merely on the AuthoredLine model.
                    # A record-TRIMMED clip always gets one (`_bound_window_to_usable`).
                    "clip_out_seconds": clip_out,
                    # Studio per-clip overrides (None unless authored). assemble_shots
                    # lets an authored caption/fit win over the computed maps, so a
                    # caption burns in even when captions_mode != burned_in.
                    "caption": line.get("caption"),
                    "fit": line.get("fit"),
                }
            )

        # Recorded interaction footage, keyed by its ``@clip-<n>`` tag.
        recorded_by_clip = {
            str(clip["line_id"]): str(clip["video_url"])
            for clip in record.get("clips", [])
            if clip.get("video_url")
        }
        # Bind each line to its backing clip. A line defaults to the clip with its
        # own id (the recorded-footage 1:1 join), but an authored line may name a
        # `clip` to play over — so several dialogue lines can share one footage beat
        # (e.g. a question + its answer over the same recording).
        video_by_line = {
            str(line["id"]): recorded_by_clip[str(line.get("clip") or line["id"])]
            for line in lines
            if str(line.get("clip") or line["id"]) in recorded_by_clip
        }

        # ORCH-3.2 directed footage (demo_video/showcase): FTR captured the per-clip
        # focus/cursor geometry + recorded viewport JS-side; thread it onto the same
        # footage shots via the SAME `clip → line` join used for `video_by_line`, so
        # several dialogue lines sharing one recording all inherit that clip's tracks.
        # Gated on `direction.footage` (presence-is-opt-in, same gate as the promo
        # path); absent ⇒ all three maps None ⇒ no geometry rides any shot ⇒
        # byte-identical. Clips lacking a track are simply omitted.
        footage = (ctx.config_snapshot.get("direction") or {}).get("footage")
        focus_track_by_line: dict[str, list[dict[str, object]]] | None = None
        cursor_track_by_line: dict[str, list[dict[str, object]]] | None = None
        footage_viewport_by_line: dict[str, list[int]] | None = None
        # ORCH-3.3 directed shot-plan (demo_video_footage_direction): a per-line
        # framing plan turns a zoom OFF (co-removing its focus+cursor tracks — a
        # cursor easing to a no-zoom target is the failure), drops a cursor for a
        # `static` reframe, and adds a callout whose coordinates come ONLY from the
        # captured focus bbox (never the LLM). Absent (stage skipped ⇒ `{}`) ⇒ no
        # plan ⇒ every map untouched ⇒ byte-identical.
        callouts_by_line: dict[str, dict[str, object]] | None = None
        split_by_line: dict[str, dict[str, object]] | None = None
        # A split the AUTHOR pinned on a line (`line.split`) is honoured whether or not
        # `direction.footage` is on, and it wins over the director's/critique's plan.
        authored_splits = authored_split_plans(lines)
        if not footage and authored_splits:
            _, _, _, split_by_line = resolve_footage_plans(
                line_ids=[str(line["id"]) for line in lines],
                plan_by_line=authored_splits,
                focus_track_by_line=None,
                cursor_track_by_line=None,
                video_by_line=video_by_line,
                video_by_clip=recorded_by_clip,
            )
        if footage:
            record_clips = record.get("clips", [])
            focus_by_clip = {
                str(c["line_id"]): c["focus_track"] for c in record_clips if c.get("focus_track")
            }
            cursor_by_clip = {
                str(c["line_id"]): c["cursor_track"] for c in record_clips if c.get("cursor_track")
            }
            viewport_by_clip = {
                str(c["line_id"]): vp
                for c in record_clips
                if (vp := normalize_footage_viewport(c.get("viewport"))) is not None
            }
            # Fan each clip's geometry onto every line bound to that clip (the same
            # `line.get("clip") or line["id"]` join `video_by_line` uses). Empty ⇒ None.
            focus_track_by_line = {
                str(line["id"]): focus_by_clip[k]
                for line in lines
                if (k := str(line.get("clip") or line["id"])) in focus_by_clip
            } or None
            cursor_track_by_line = {
                str(line["id"]): cursor_by_clip[k]
                for line in lines
                if (k := str(line.get("clip") or line["id"])) in cursor_by_clip
            } or None
            footage_viewport_by_line = {
                str(line["id"]): viewport_by_clip[k]
                for line in lines
                if (k := str(line.get("clip") or line["id"])) in viewport_by_clip
            } or None

            # ORCH-3.3: apply the authored shot-plan on top of the captured tracks.
            # The plan is INTENT only — coordinates come from `focus_by_clip`'s bbox,
            # never the LLM. Skipped stage ⇒ `{}` ⇒ `plan_by_line` empty ⇒ every map
            # is left exactly as the capture built it (byte-identical).
            # PREFER the footage-critique's persisted nudged plans when present, so a
            # re-compile (encode_master, S20) rebuilds the footage from the critique's
            # improvements instead of the L1 baseline. TRUTHY check (not `is not None`):
            # an empty `[]` (no nudge / critique skipped-clean) falls back to the base
            # footage-direction plans. On the FIRST compile the critique output is
            # absent (compile precedes footage_critique in the DAG) ⇒ base plans ⇒
            # byte-identical.
            critique_out = ctx.prior_output("demo_video_footage_critique") or {}
            critique_plans = critique_out.get("footage_plans")
            base_direction_plans = (ctx.prior_output("demo_video_footage_direction") or {}).get(
                "plans", []
            )
            plan_source = critique_plans if critique_plans else base_direction_plans
            plan_by_line = pin_authored_splits(
                {str(p["line_id"]): p for p in plan_source}, authored_splits
            )
            # The callout bbox is read from `focus_track_by_line` (built above from
            # `focus_by_clip` via the SAME `clip → line` join), so this is identical
            # to reading `focus_by_clip[key]` directly — shared with the promo path.
            focus_track_by_line, cursor_track_by_line, callouts_by_line, split_by_line = (
                resolve_footage_plans(
                    line_ids=[str(line["id"]) for line in lines],
                    plan_by_line=plan_by_line,
                    focus_track_by_line=focus_track_by_line,
                    cursor_track_by_line=cursor_track_by_line,
                    video_by_line=video_by_line,
                    video_by_clip=recorded_by_clip,
                )
            )

        # Branded card beats (showcase blueprint): each authored card line shows its
        # still PNG, upgraded to its branded animated clip when showcase_slides
        # rendered one. Absent for a plain demo_video run (the stage isn't in that
        # DAG) → empty maps, so the recorded-footage behavior is byte-identical.
        slides_out = ctx.prior_output("showcase_slides") or {}
        slide_by_line = {
            str(s["line_id"]): str(s["image_url"])
            for s in slides_out.get("slides", [])
            if s.get("line_id") and is_fetchable_url(s.get("image_url"))
        }
        image_by_line = dict(slide_by_line)
        fit_by_line: dict[str, str] = {}
        overlay_motion_slides(
            image_by_line,
            video_by_line,
            fit_by_line,
            slide_by_line=slide_by_line,
            motion_slide_by_line=collect_motion_slides_by_line(slides_out),
        )

        # Burned-in captions (opt-in via deliverable captions_mode=burned_in),
        # keyed by the same line id the soft-CC publish track uses. ONLY burned_in:
        # a static per-shot `caption` is always drawn into the picture by the
        # compiler, so `overlay` must not stamp one -- its captions ride as cues
        # (`caption_clip_refs` / `static_caption_cues`) muxed as soft mov_text.
        captions_by_line = (
            caption_text_by_line({"demo_video_gen": gen})
            if captions_mode(ctx) == "burned_in"
            else None
        )
        # NB: double-burn suppression (clearing a shot's static caption when it also
        # gets a word-timed cue) happens in CompileStageBase.handle AFTER the async
        # caption build — success-based, so a line whose ASR fails keeps its static
        # caption instead of losing captions entirely on a transcription outage.

        # Sound design (opt-in via brief.sound_effects): per-line accent cues from
        # demo_video_sound_director plus a transition whoosh on every cut. Catalog
        # ids here, resolved to presigned URLs in the compile tail. Empty/None when
        # sound is off → shots carry no sound, plain demo_video runs stay identical.
        sound_cfg = sound_config(ctx)

        # ORCH-1.3a cold-open: reframe the opening footage shot to a punchy
        # window. Presence-opt-in — inert unless open_style / direction.hook asks.
        cold_open_plan = self._resolve_cold_open(ctx, lines, record, out_points)
        # Compute-plan seam (C13): reconcile_timeline stamps a per-line allotment
        # (`seconds`) plus the window it must stay inside (`min_seconds`/`max_seconds`).
        # Only what the stage ACTUALLY stamped is passed through — never a synthesised
        # or defaulted bound — because `assemble_shots` clamps a bound unconditionally,
        # even over a narrated duration. A footage beat is pinned min == max == its
        # measured clip length, so the clamp is a no-op there (C0: never trimmed). When
        # reconcile_timeline did not run, both maps are empty ⇒ the reconcile is
        # byte-identical to before (`assemble_shots` treats `{}` exactly like `None`).
        bounds_by_line = _min_max_by_line(lines)
        # SILENT-SHORT GUARD (measured 2026-09-16, two runs). Nothing downstream
        # notices a footage beat that lost its `[min_seconds, max_seconds]` window:
        # with no bound, the narration clip's length declares the beat's duration in
        # BOTH directions (a beat ships SHORTER than its clip, and a beat whose
        # narration overruns its clip ships LONGER), the compiler renders exactly what
        # it was handed, and `declared` matches the encoded stream to the millisecond —
        # so every duration check downstream AGREES with a film that is short of its
        # own footage. Measured: 62.120s of footage-pinned beats (min == max == the
        # measured clip) delivered as 54.442s, 7.6s of paid footage dropped with no
        # note and no degradation.
        #
        # ⛔ WHAT THIS DELIBERATELY DOES NOT READ: `ctx.prior_output("reconcile_timeline")`.
        # The root cause of the incident is that that call was FALSY, so a guard
        # requiring it truthy cannot fire in the state it was written for — and
        # `prior_run_outputs` collapses "row absent", "skipped" and "completed with a
        # falsy output" into one value (`run_dispatcher` seeds skips as `{}`), so the
        # call cannot express the distinction at all. Driving all three states
        # reproduced the 7.670s shortfall with an EMPTY error log.
        #
        # So the two questions are asked of two sources that cannot collapse:
        #   * WAS a window ordered? — `timeline_bounds_were_ordered`, read off
        #     `config_snapshot`, which is the order itself and is present on every
        #     context regardless of what reconcile's row says. This is what keeps the
        #     guard quiet on the pre-ORCH-1.4 topology (no `direction.timing` ⇒ no
        #     bounds were ever ordered ⇒ narration lengths are the CORRECT outcome),
        #     which is the case that must not fire or the signal means nothing.
        #   * WHICH beats lost one? — `_unbounded_footage_beats`, off the MEASURED
        #     clip lengths in `demo_video_record`. A beat with paid, probed footage and
        #     no window is the impossible state; a beat with no footage is not. This is
        #     also what makes the report PER-LINE, so partial loss (the common shape) is
        #     reported rather than hidden behind a non-empty map.
        #
        # Visibility ONLY: never a synthesised bound (`_min_max_by_line` must not
        # invent one) and never a raised failure — a run that terminates never reaches
        # `mark_completed`, which silently disables structural-relief spawning for the
        # whole fleet.
        if timeline_bounds_were_ordered(ctx):
            unbounded = _unbounded_footage_beats(lines, clips_by_line, bounds_by_line)
            if unbounded:
                narrated = {
                    str(c["line_id"]): float(c.get("seconds") or 0.0)
                    for c in narrate.get("clips", [])
                }
                footage_at_risk = sum(seconds for _lid, seconds in unbounded)
                narration_instead = sum(narrated.get(lid, 0.0) for lid, _s in unbounded)
                log.error(
                    "%s: this run ORDERED a reconciled timeline, but %d of %d beat(s) "
                    "reach the compiler with recorded footage and NO "
                    "[min_seconds, max_seconds] window — each will take its NARRATION "
                    "length in both directions, so %.3fs of measured footage ships as "
                    "%.3fs (short by %.3fs) and every downstream duration check will "
                    "agree with it. Per beat (line: footage -> narration): %s",
                    self.name,
                    len(unbounded),
                    len(lines),
                    footage_at_risk,
                    narration_instead,
                    footage_at_risk - narration_instead,
                    ", ".join(
                        f"{lid}: {seconds:.3f}s -> {narrated.get(lid, 0.0):.3f}s"
                        for lid, seconds in unbounded
                    ),
                )
        return assemble_shots(
            storyboard_shots=storyboard_shots,
            images=[],
            narration_clips=narrate["clips"],
            captions_by_line=captions_by_line,
            image_by_line=image_by_line,
            video_by_line=video_by_line,
            fit_by_line=fit_by_line,
            seconds_by_line=_seconds_by_line(lines),
            min_max_by_line=bounds_by_line,
            hold_seconds_by_line=_hold_seconds_by_line(lines),
            sfx_by_line=sfx_by_line(ctx, sound_cfg, "demo_video_sound_director"),
            transition_sfx_id=transition_sfx_id(sound_cfg),
            # Same brief-level `sfx_gain_db` the accents honour (sfx_by_line), so a
            # single render can't ship -20 dB accents against -10 dB whooshes.
            transition_sfx_gain_db=transition_gain_db(sound_cfg),
            # Accessibility Tier-1 (ORCH-0.8) — None unless the run opted in via an
            # `accessibility` block, so shots stay byte-identical for every other run.
            a11y_policy=build_a11y_policy(ctx.config_snapshot),
            cold_open_plan=cold_open_plan,
            focus_track_by_line=focus_track_by_line,
            cursor_track_by_line=cursor_track_by_line,
            footage_viewport_by_line=footage_viewport_by_line,
            callouts_by_line=callouts_by_line,
            split_by_line=split_by_line,
        )

    def title_card_plan(self, ctx: RunContext) -> TitleCardPlan | None:
        """Cards count INSIDE the order, capped at 20% of it (owner ruling).

        The base plans the ``title_card`` block uncapped (``ordered_seconds=None``),
        so a demo_video/showcase run spliced e.g. a 6s outro ON TOP of a 30s order.
        Here the cap applies against the SAME ordered length the duration-conformance
        verdict grades this film against (``demo_video_ordered_seconds``:
        ``direction.timing.target_seconds`` → ``deliverable.target_duration_seconds``).
        No order on the run ⇒ ``None`` ⇒ planned as configured (the base behaviour);
        no invented default order.
        """
        return plan_title_cards(
            self.title_card_config(ctx),
            ordered_seconds=demo_video_ordered_seconds(ctx.config_snapshot),
        )

    def caption_clip_refs(self, ctx: RunContext) -> dict[str, ClipRef]:
        """Per-line narration clips to word-time-caption (ORCH-4.2).

        Reuses the EXACT text/narrate plumbing ``build_shots`` uses to build the
        static ``captions_by_line`` map — doctored-or-gen spine via
        ``resolve_plan_output`` + ``caption_text_by_line`` for the text, and the
        ``demo_video_narrate`` clips for the audio join — so the word-timed and
        static caption paths can never drift. Inert ({}) unless the deliverable
        opted into ``captions_mode`` ``burned_in`` or ``overlay`` (same cues; the
        compiler burns the former and muxes the latter as soft mov_text); the
        kill-switch gate lives one layer up in ``CompileStageBase.handle``.
        """
        if captions_mode(ctx) not in ("burned_in", "overlay"):
            return {}
        gen = resolve_plan_output(ctx) or {}
        text_by_line = caption_text_by_line({"demo_video_gen": gen}) or {}
        narrate = ctx.prior_output("demo_video_narrate") or {}
        refs: dict[str, ClipRef] = {}
        for clip in narrate.get("clips", []):
            line_id = str(clip["line_id"])
            text = text_by_line.get(line_id)
            if not text:
                continue
            refs[line_id] = ClipRef(
                line_id=line_id,
                text=text,
                narrate_stage="demo_video_narrate",
                clip_filename=f"{clip['id']}.wav",
                seconds=float(clip.get("seconds") or 0.0),
            )
        return refs

    def static_caption_cues(
        self, ctx: RunContext, shots: list[ShotDict]
    ) -> list[ShotCaptionsPayload] | None:
        """Captions-only overlay: one shot-spanning soft cue per captioned line.

        A silent run has no narration clip to word-time, so without this the
        overlay track would be empty. Same caption text source as
        ``caption_clip_refs`` (doctored-or-gen spine). Voiced runs return ``None``.
        """
        if not captions_only(ctx):
            return None
        text_by_line = caption_text_by_line({"demo_video_gen": resolve_plan_output(ctx) or {}})
        return static_caption_cues(shots, text_by_line or {})

    def _resolve_cold_open(
        self,
        ctx: RunContext,
        lines: list[dict[str, Any]],
        record: dict[str, Any],
        out_points: Mapping[str, float] | None = None,
    ) -> tuple[str, float] | None:
        """Resolve the opt-in cold-open window for the opening footage shot
        (ORCH-1.3a). Returns ``None`` — the inert, byte-identical path — unless the
        run authored an ``open_style`` or a live ``direction.hook`` block, the
        opening line binds real footage, and that clip has a real duration. Only the
        pure :func:`cold_open_window` decides the ``(anchor, lead)``; this method
        only gathers its inputs and never touches ``seconds``."""
        open_style = ctx.config_snapshot.get("open_style")
        hook = (ctx.config_snapshot.get("direction") or {}).get("hook")
        hook_block = hook if isinstance(hook, dict) and hook else None
        # Explicit force-off always wins — an author can turn a cold open off even
        # after selecting open_style.
        if hook_block is not None and hook_block.get("mode") == "off":
            return None
        # Cold-open activates ONLY on an explicit `open_style == "cold_open"` request
        # or a present (non-off) `direction.hook` block. Other styles (`logo_open`,
        # `auto`) and an absent block stay inert / byte-identical in 1.3a — auto
        # cold-open selection is ORCH-1.3b's hook-doctor, not this deterministic pass.
        if not _cold_open_active(open_style, hook_block):
            return None
        if not lines:
            return None
        first = lines[0]
        clip_id = str(first.get("clip") or first["id"])
        clips_by_line = {str(c["line_id"]): c for c in record.get("clips", [])}
        # The opener's reconcile cut (if any) shortens the footage the cold-open window
        # is framed inside, exactly as a record trim does.
        opening_clip = (
            _clip_cut_at(clips_by_line.get(clip_id), (out_points or {}).get(str(first["id"]))) or {}
        )
        clip_seconds = float(opening_clip.get("seconds") or 0.0)
        # An opening card/slide with no footage → nothing to reframe → None.
        if clip_seconds <= 0:
            return None
        # The measured motion curve of the opening footage (ORCH-1.3c), when
        # demo_video_record's QA pass captured one. `record` is the stage's persisted
        # JSON, so these are plain dicts — cold_open_window coerces defensively and
        # falls back to duration-derived framing when the curve is absent or flat.
        motion_timeline = opening_clip.get("motion_timeline")
        anchor, lead = cold_open_window(
            clip_seconds,
            hook if isinstance(hook, dict) else None,
            open_style if isinstance(open_style, str) else None,
            motion_timeline if isinstance(motion_timeline, list) else None,
        )
        # A trimmed opener renders in window mode (the spine closes it at the usable
        # end), so re-express the cold-open lead against that window.
        raw_out = first.get("clip_out_seconds")
        anchor, bounded_lead, _out = _bound_window_to_usable(
            opening_clip,
            anchor,
            lead,
            float(raw_out) if isinstance(raw_out, (int, float)) and raw_out else None,
            float(first.get("seconds") or 0.0),
        )
        return (anchor or "end", bounded_lead or 0.0)

    def _emit_shot_timeline(self, ctx: RunContext) -> bool:
        # Emit the authoritative timeline ONLY when final_critique's ship-gate will
        # consume it: block present AND non-empty AND mode != "off" (mirrors
        # require_ship_gate + final_critique._mode). Absent/empty/off ⇒ probe-free.
        # A captions-only overlay film also needs it: its SRT sidecar has no
        # narration clips to time cues from, so it reads the real shot spans.
        if captions_only(ctx) and captions_mode(ctx) == "overlay":
            return True
        ship_gate = ctx.config_snapshot.get("ship_gate")
        if not isinstance(ship_gate, Mapping) or not ship_gate:
            return False
        return str(ship_gate.get("mode") or "critique_only").lower() != "off"

    async def build_audio_bed(
        self,
        ctx: RunContext,
        shots: list[dict[str, Any]],
        *,
        degradations: list[Degradation] | None = None,
    ) -> dict[str, Any] | None:
        """Continuous music + ambient bed from brief.sound_effects (shared
        resolver). ``None`` when sound design is off / no bed resolves. The
        ``degradations`` sink is forwarded so a chosen-but-unresolvable bed is
        recorded rather than silently dropped."""
        return await resolve_audio_bed(ctx, shots, sound_config(ctx), degradations=degradations)

    def build_output(self, result: dict[str, Any], ctx: RunContext) -> DemoVideoCompileOutput:
        mastering = result.get("mastering")
        output_sha256 = result.get("output_sha256")
        loudness = result.get("loudness")
        shot_timeline = result.get("shot_timeline")
        return DemoVideoCompileOutput(
            artifact_url=str(result["artifact_url"]),
            duration_seconds=float(result["duration_seconds"]),
            mastering=mastering if isinstance(mastering, dict) else None,
            output_sha256=str(output_sha256) if isinstance(output_sha256, str) else None,
            loudness=loudness if isinstance(loudness, dict) else None,
            shot_timeline=shot_timeline if isinstance(shot_timeline, list) else None,
            # The mastering gate now runs for EVERY render blueprint, in
            # CompileStageBase — see shared/mastering_gate.py. It used to be applied
            # here, which protected demo_video and nothing else.
        )
