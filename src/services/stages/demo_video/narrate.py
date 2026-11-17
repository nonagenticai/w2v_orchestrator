"""demo_video_narrate — TTS each demo scenario's narration line.

A thin NarrationStageBase subclass: one synthesize call per ``demo_video_gen``
line, the WAV persisted via ``store_tts_clip`` keyed by ``line_id`` (= the
scenario's ``clip-<n>`` join id) so demo_video_compile can mux the narration
track 1:1 against the recorded clip.

Single narrator by default. When ``brief.dialogue.enabled`` is set, ``voice_for``
casts each line to its speaker's voice (the ``speaker`` slot demo_video_gen
stamped) so the showcase blueprint narrates a two-voice dialogue — the same
``voice_for_speaker`` mechanism promo / website_review use.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar, cast

from src.config import get_settings
from src.services.stages.base import (
    ArtifactUrl,
    SlotUse,
    StageInput,
    StageOutput,
    StageOutputWithNotes,
)
from src.services.stages.demo_video.spine import PLAN_STAGE_PRECEDENCE, resolve_plan_output
from src.services.stages.primitives.dialogue import voice_for_speaker
from src.services.stages.primitives.narration import NarrationStageBase
from src.services.stages.primitives.ssml import build_line_ssml
from src.services.stages.shared.authored_storyboard import authored_lines
from src.services.stages.shared.captions_only import captions_only
from src.services.stages.shared.degradation import URL_REWRITTEN_BEFORE_TTS, degradation
from src.services.stages.shared.speakable_text import contains_url, speakable_text
from src.services.tts_storage import pad_stored_clip_head

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

log = logging.getLogger(__name__)


def _generated_spine(ctx: RunContext) -> bool:
    """True when this run's spine is MODEL-GENERATED — the same test
    ``reconcile_timeline._speakable_generated_spine`` and ``demo_video_gen`` use
    (:func:`authored_lines` over ``brief``), never ``locked``."""
    return authored_lines(ctx.config_snapshot.get("brief") or {}) is None


def _url_in_generated_line(item: dict, ctx: RunContext) -> bool:
    """A generated line whose text still carries a URL at narrate — the skip path.

    ``reconcile_timeline`` makes a generated spine speakable, but it SKIPS on a run with
    no ``direction.timing`` block, and then script_doctor / claim_finish /
    retention_plan's text reaches narrate unrewritten. Owner-authored lines keep their
    URL verbatim.
    """
    text = item.get("text")
    return isinstance(text, str) and contains_url(text) and _generated_spine(ctx)


def _freeze_pad(item: dict) -> float:
    """This beat's solver-granted freeze pad, in seconds — ``0.0`` when it has none.

    ``reconcile_timeline`` stamps ``freeze_pad_seconds`` per line off
    ``Allotment.freeze_pad_seconds``; it is 0.0 for every non-footage / already-fits
    beat and absent entirely on a run where that stage never stamped the spine. Any
    non-numeric or negative value is read as "no pad" rather than allowed to corrupt
    the fit target.
    """
    raw = item.get("freeze_pad_seconds")
    if raw is None or isinstance(raw, bool):
        return 0.0
    try:
        pad = float(raw)
    except TypeError, ValueError:
        return 0.0
    return pad if pad > 0 else 0.0


def _window(item: dict) -> float | None:
    """The voice's FULL window for this line — ``seconds + freeze_pad_seconds`` — or
    ``None`` when the line carries no usable allotment (see ``fit_target_for``)."""
    allotment = item.get("seconds")
    if allotment is None:
        return None
    try:
        target = float(allotment)
    except TypeError, ValueError:
        return None
    if target <= 0:
        return None
    return target + _freeze_pad(item)


def _hold(item: dict) -> float:
    """This beat's UNDERSPOKEN hold, in seconds — ``0.0`` when it has none.

    ``reconcile_timeline`` stamps ``hold_seconds`` per line off
    ``Allotment.hold_seconds`` (``time_budget.underspoken_hold``): the seconds this
    beat's allotment holds that its words cannot fill even at the natural-rate
    floor, already ceiling-capped upstream. ``shot_assembly._apply_underspoken_hold``
    spends those seconds as a HELD LAST FRAME on the picture.

    Same defensive coercion as :func:`_freeze_pad` — absent / non-numeric / bool /
    negative all read as "no hold" rather than corrupting the fit target.
    """
    raw = item.get("hold_seconds")
    if raw is None or isinstance(raw, bool):
        return 0.0
    try:
        hold = float(raw)
    except TypeError, ValueError:
        return 0.0
    return hold if hold > 0 else 0.0


def _fitted(item: dict) -> bool:
    """A line reconcile_timeline's MEASURED fit synthesized (``fitted_seconds`` set).

    Narrate replays that exact plain-text clip: the fit measured it, and the hold /
    keep decision were computed from its length. SSML (pause / emphasis) would change
    the clip id and the clip's length, i.e. re-pay synthesis and ship a different take.
    """
    raw = item.get("fitted_seconds")
    return isinstance(raw, int | float) and not isinstance(raw, bool) and raw > 0


def _fitted_slack(item: dict) -> float:
    raw = item.get("fitted_slack_seconds")
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        return 0.0
    return max(0.0, float(raw))


class DemoVideoNarrateInput(StageInput):
    pass


class DemoVideoAudioClip(StageOutput):
    id: str
    line_id: str
    artifact_url: ArtifactUrl
    seconds: float
    # Speed the kept clip was synthesized at (None = backend default) and its
    # spoken word count, filled by NarrationStageBase. Optional so outputs
    # persisted before these fields existed still validate.
    speed: float | None = None
    words: int | None = None


class DemoVideoNarrateOutput(StageOutputWithNotes):
    # `demo_video_compile` hard-fails on a falsy narrate output but NOT on an empty
    # `clips` list, and then builds every shot from it — so zero clips compiles a
    # silent film that reports completed. Produced by NarrationStageBase fanning out
    # over the spine lines, so non-empty whenever the spine is; there is no gate and
    # no mode in which an empty is intended.
    produces: ClassVar[tuple[str, ...]] = ("clips",)

    clips: list[DemoVideoAudioClip]


class DemoVideoSilentNarrateOutput(DemoVideoNarrateOutput):
    """The narrate output of a deliberate captions-only run: NO clips, by design.

    A distinct class (the executor reads ``type(result)``) so the ``produces`` contract
    inherited from :class:`DemoVideoNarrateOutput` does not flag a mode that is silent on
    purpose as a stage that "completed without producing clips"."""

    produces: ClassVar[tuple[str, ...]] = ()


class DemoVideoNarrateStage(NarrationStageBase[DemoVideoNarrateInput, DemoVideoNarrateOutput]):
    name = "demo_video_narrate"
    input_model = DemoVideoNarrateInput
    output_model = DemoVideoNarrateOutput
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (SlotUse("tts"),)
    source_stage = "demo_video_gen"
    source_list_key = "lines"
    source_id_field = "line_id"
    clip_model = DemoVideoAudioClip
    # Keep in lockstep with ``resolve_script`` below — the re-narrate patcher has
    # no RunContext and resolves the read stage from this tuple.
    source_stage_chain = PLAN_STAGE_PRECEDENCE

    def resolve_script(self, ctx: RunContext) -> dict[str, Any] | None:
        # Narrate the doctored spine when script_doctor ran, else demo_video_gen
        # (skip-clean fallback). Same `lines` shape either way, so the base fan-out
        # is untouched — only which output supplies the text changes.
        return resolve_plan_output(ctx)

    async def handle(
        self, payload: DemoVideoNarrateInput, ctx: RunContext
    ) -> DemoVideoNarrateOutput:
        if captions_only(ctx):
            # brief.captions_only: no TTS call, no audio. The spine must still exist
            # (compile reads the same lines), so a missing one fails like the voiced path.
            if not self.resolve_script(ctx):
                raise RuntimeError(f"{self.source_stage} output missing")
            log.info("demo_video_narrate captions_only run=%s: TTS skipped", ctx.run_id)
            return DemoVideoSilentNarrateOutput(clips=[])
        out = await super().handle(payload, ctx)
        script = self.resolve_script(ctx) or {}
        out = await self._direct_fitted_lines(out, script, ctx)
        rewritten = sorted(
            str(item.get("id"))
            for item in script.get(self.source_list_key) or []
            if isinstance(item, dict) and _url_in_generated_line(item, ctx)
        )
        if rewritten:
            # The voice speaks different words than the line's rate/word budget was
            # computed on (reconcile_timeline skipped, or a later writer re-added a
            # URL/path). Machine-readable only — no prose note restating it.
            out.degradations = [
                *out.degradations,
                degradation(
                    URL_REWRITTEN_BEFORE_TTS,
                    f"generated line(s) {', '.join(rewritten)}: a URL or raw path was "
                    "rewritten to speakable text before TTS; the line's rate budget was "
                    "computed on the pre-rewrite text",
                ),
            ]
            log.warning(
                "demo_video_narrate url_rewritten_at_narrate run=%s lines=%s",
                ctx.run_id,
                rewritten,
            )
        return out

    async def _direct_fitted_lines(
        self, out: DemoVideoNarrateOutput, script: dict, ctx: RunContext
    ) -> DemoVideoNarrateOutput:
        """Apply voice_director's direction to FITTED lines without re-synthesizing.

        A leading pause becomes leading SILENCE padded onto the measured clip (no TTS
        call) — only when it fits the beat's slack (``fitted_slack_seconds``: the
        footage window the clip leaves unspoken, minus its hold), so the film's length
        is unchanged; otherwise it is dropped. Emphasis is never applied to a fitted
        line (it would need a different synthesis). Every decision is logged per line.
        Unfitted lines are untouched (they took the SSML path as before).
        """
        items = {
            str(item.get("id")): item
            for item in script.get(self.source_list_key) or []
            if isinstance(item, dict) and _fitted(item)
        }
        if not items:
            return out
        clips = []
        for clip in out.clips:
            item = items.get(clip.line_id)
            directive = self._matching_directive(item, ctx) if item is not None else None
            if item is None or directive is None:
                clips.append(clip)
                continue
            if directive.get("emphasis_word_indices"):
                log.info(
                    "demo_video_narrate fitted line %s: emphasis DROPPED (it would need a new "
                    "synthesis; the measured clip is replayed)",
                    clip.line_id,
                )
            pause = max(0, int(self.pause_seconds_for_directive(directive) * 1000)) / 1000.0
            if pause <= 0:
                clips.append(clip)
                continue
            slack = _fitted_slack(item)
            if pause > slack:
                log.info(
                    "demo_video_narrate fitted line %s: leading pause %.2fs DROPPED (slack "
                    "%.2fs — it would lengthen the beat)",
                    clip.line_id,
                    pause,
                    slack,
                )
                clips.append(clip)
                continue
            padded = await pad_stored_clip_head(str(ctx.run_id), self.name, clip.id, pause)
            if padded is None:
                log.warning(
                    "demo_video_narrate fitted line %s: leading pause %.2fs DROPPED (could "
                    "not pad the stored clip)",
                    clip.line_id,
                    pause,
                )
                clips.append(clip)
                continue
            clip_id, url, seconds = padded
            log.info(
                "demo_video_narrate fitted line %s: leading pause %.2fs APPLIED as silence "
                "(slack %.2fs; %.2fs -> %.2fs)",
                clip.line_id,
                pause,
                slack,
                clip.seconds,
                seconds,
            )
            clips.append(
                clip.model_copy(update={"id": clip_id, "artifact_url": url, "seconds": seconds})
            )
        return out.model_copy(update={"clips": clips})

    def voice_for(self, item: dict, voice: dict, ctx: RunContext) -> str | None:
        # Off (no brief.dialogue.enabled) → the single configured voice, unchanged.
        return voice_for_speaker(item, voice, ctx.config_snapshot.get("brief") or {})

    def _matching_directive(self, item: dict, ctx: RunContext) -> dict | None:
        # Shared by synth_text_for and pause_seconds_for so both read the SAME
        # PerformanceDirective — the pause fed to the fit solver must come from
        # here, never re-parsed out of the SSML string synth_text_for builds.
        # (Return type is bare `dict`, not `dict[str, Any]` — see
        # tests/test_any_ratchet.py; this repo counts that spelling.)
        if not get_settings().narration_ssml_enabled:
            return None
        voice_director = ctx.prior_output("voice_director") or {}
        line_id = item.get("id")
        for directive in voice_director.get("directives") or []:
            if directive.get("line_id") == line_id:
                return cast("dict", directive)
        return None

    def synth_text_for(self, item: dict, ctx: RunContext) -> tuple[str, bool]:
        # TTSB-3: lower voice_director's per-line pause + emphasis into SSML,
        # gated OFF by default. When off → the plain line text (byte-identical
        # to the pre-SSML path). Same directive lookup as speed_for (matched by
        # line id against ctx.prior_output("voice_director")).
        text = str(item["text"])
        # ⛔ FINAL SPEAKABLE GUARD — the one boundary every run crosses. See
        # `_url_in_generated_line`; `handle` records that it fired.
        rewritten = _url_in_generated_line(item, ctx)
        if rewritten:
            text = speakable_text(text)
        # A FITTED line replays its measured plain-text clip — see `_fitted`; its
        # direction is applied (or dropped) by `_direct_fitted_lines`, never as SSML.
        directive = None if _fitted(item) else self._matching_directive(item, ctx)
        if directive is None:
            return text, False
        # `speed_for` is a pure read of `item`/`ctx` (no side effects, called again
        # unconditionally in `handle`), so calling it here too is safe and is the
        # only way `build_line_ssml` learns the rate the emphasised span will
        # actually be delivered at — see ssml.py `emphasis_affordable`.
        ssml = build_line_ssml(
            text,
            directive.get("pause_before_ms", 0) or 0,
            # voice_director indexed the UNREWRITTEN `text.split()`; after the URL
            # rewrite those positions point at the wrong words, so emphasis is dropped
            # (the pause is position-free and kept).
            [] if rewritten else directive.get("emphasis_word_indices") or [],
            vo_rate=self.speed_for(item, ctx),
        )
        # None ⇒ nothing to express (no pause, no emphasis) → plain path.
        return (ssml, True) if ssml is not None else (text, False)

    def pause_seconds_for(self, item: dict, ctx: RunContext) -> float:
        # The fixed <break> the line above lowers into SSML is silence the
        # backend renders verbatim (never scaled by `speed`), so the fit solver
        # must not attribute it to the speakable portion of the line — see
        # NarrationStageBase.pause_seconds_for. Reads the SAME directive
        # synth_text_for used to build the break, not the rendered markup.
        directive = None if _fitted(item) else self._matching_directive(item, ctx)
        if directive is None:
            return 0.0
        return self.pause_seconds_for_directive(directive)

    @staticmethod
    def pause_seconds_for_directive(directive: dict) -> float:
        pause_ms = directive.get("pause_before_ms", 0) or 0
        try:
            pause_ms = int(pause_ms)
        except TypeError, ValueError:
            return 0.0
        return max(0, pause_ms) / 1000.0

    def fit_target_for(self, item: dict, ctx: RunContext) -> float | None:
        # The line's ALLOTMENT, as `reconcile_timeline` stamped it — the same field
        # whose companion `vo_rate` `speed_for` consumes below. `vo_rate` is only an
        # ESTIMATE (words-per-minute), so the audio synthesized at it routinely
        # misses this number; returning it here lets the base measure the real
        # duration and re-synthesize the line once at the exactly-correct speed
        # instead of leaving the compiler to freeze-pad the overrun.
        #
        # Same care as `speed_for` about WHICH lines carry one: a footage/card beat
        # and a skip-clean run where reconcile_timeline never ran carry no
        # `seconds`, and None here means "accept what TTS returns" — today's
        # behavior exactly. A non-positive value is meaningless as an allotment and
        # is treated the same way rather than driving the correction arithmetic.
        # ⛔ AND THE FREEZE PAD IS PART OF THE WINDOW. `solve_timeline` chooses a FOOTAGE
        # beat's `vo_rate` against `seconds + freeze_pad_seconds` — the clip may hold its
        # last frame frozen for up to `DEFAULT_FREEZE_PAD_CEILING` (1.5s), and the
        # narration is deliberately allowed to use that extension instead of
        # over-compressing into the bare clip. Fitting to `seconds` alone therefore
        # re-synthesized the line SLOWER than the solver intended, by up to the whole
        # ceiling on every padded beat: a second, independent slow-narration lever, and
        # one that fires precisely on the footage beats demo_video is made of.
        #
        # `freeze_pad_seconds` is 0.0 / absent on every non-footage and already-fits beat
        # (and on a skip-clean run where reconcile_timeline never ran), so the target is
        # byte-identical to before for those — the pad is only ever ADDED where the
        # solver actually granted one. A non-numeric or negative value contributes
        # nothing rather than corrupting the allotment.
        # ⛔ AND THE UNDERSPOKEN HOLD IS *NOT* PART OF THE WINDOW — it is subtracted.
        # `hold_seconds` points the OTHER WAY from `freeze_pad_seconds`. The pad is
        # extra window GRANTED to a voice that overruns; the hold is the residual a
        # beat's words cannot fill even at the natural-rate floor, which
        # `shot_assembly._apply_underspoken_hold` spends as a HELD LAST FRAME on the
        # picture. Wave 7 computed that hold precisely so an underspoken beat could
        # cover its residual with a held picture instead of slowed speech — and then
        # this function handed the narration the WHOLE allotment anyway, so the line
        # was still re-synthesized to fill seconds the picture had already agreed to
        # cover, clamping at `_FIT_FLOOR` (0.80) and dragging the delivery. The hold
        # was added to the shot ON TOP of that. Subtracting it here is what makes the
        # two halves agree: the voice fits the seconds it is actually meant to speak,
        # the frame holds the rest.
        #
        # 0.0 on every beat that is not underspoken (and on a skip-clean run where
        # reconcile_timeline never stamped the spine), so the target is byte-identical
        # to before everywhere the hold does not exist. The guard below keeps a
        # corrupt/oversized hold from producing a non-positive allotment — such a value
        # is ignored rather than allowed to invert the fit arithmetic.
        window = _window(item)
        if window is None:
            return None
        spoken = window - _hold(item)
        return spoken if spoken > 0 else window

    def measured_fit_target_for(
        self, item: dict, ctx: RunContext, measured_seconds: float
    ) -> float | None:
        # ⛔ THE HOLD MAY NEVER MAKE A LINE FASTER. `fit_target_for` subtracts the hold
        # because the PLAN estimated (from a words-per-minute guess, before any audio
        # existed) that the words run out early. The measurement is the truth, and it
        # can refute that estimate. Run `01a0b007-9c2c` clip-5: planned underspoken
        # (16 words ⇒ hold 1.50s against a 9.56s footage beat), MEASURED at 11.91s at
        # speed 0.97 — the URL in the line is spoken slowly, 84.7 wpm. Fitting to
        # `9.56 - 1.50 = 8.06s` re-synthesized it at 1.469, a 47% speed-up whose only
        # purpose was to open a silent hold; the full 9.56s window needed 1.224.
        #
        # Rule: the voice's target is the measurement CLAMPED into
        # `[window - hold, window]`:
        #   * measured past the window ⇒ the window (speed up only as far as the
        #     picture requires — never overrun the footage);
        #   * measured inside it ⇒ itself (within tolerance ⇒ kept, no re-synthesis:
        #     neither sped up to open the hold nor dragged to close it);
        #   * measured short of `window - hold` ⇒ `window - hold` (today's behaviour —
        #     a genuinely underspoken line is not dragged to fill the held seconds).
        # With no hold the range collapses to the window ⇒ byte-identical to before.
        if _fitted(item):
            # The fit already measured and kept this exact clip; its hold was computed
            # from it. Accept it as is — a re-fit here is the double correction.
            return None
        spoken = self.fit_target_for(item, ctx)
        window = _window(item)
        if spoken is None or window is None or measured_seconds <= 0:
            return spoken
        return min(max(measured_seconds, spoken), window)

    def speed_for(self, item: dict, ctx: RunContext) -> float | None:
        # Final-speed precedence (ORCH-1.4), first non-None wins:
        #   1. author_rate — the AuthoredLine per-line `rate` override. It rides the
        #      spine as `item["rate"]`: gen dumps the whole AuthoredLine
        #      (`_authored_output`) and reconcile_timeline preserves every field via
        #      `{**line, **patch}`, so the authored value reaches here untouched.
        #   2. vo_rate — stamped per line by reconcile_timeline to fit the VO into
        #      its allotment (footage/card beats, or a skipped stage, carry none).
        #   3. voice_director's PerformanceDirective.rate_hint — matched by line id
        #      from ctx.prior_output("voice_director"). When that stage RAN but gave
        #      this line no usable hint, the neutral 1.0 governs.
        #   4. No voice_director output AND no author rate → today's behavior EXACTLY
        #      (AC9): the vo_rate path, i.e. None when there is no vo_rate.
        author_rate = item.get("rate")
        if author_rate is not None:
            return float(author_rate)
        vo_rate = item.get("vo_rate")
        if vo_rate is not None:
            return float(vo_rate)
        # prior_output → None (stage absent) or {} (skip-clean) ⇒ falsy ⇒ fall
        # through to the behavior-preserving None below. A truthy dict means the
        # stage actually ran, so its directive (or the neutral 1.0) governs.
        voice_director = ctx.prior_output("voice_director") or {}
        if voice_director:
            line_id = item.get("id")
            for directive in voice_director.get("directives") or []:
                if directive.get("line_id") == line_id:
                    rate_hint = directive.get("rate_hint")
                    if rate_hint is not None:
                        return float(rate_hint)
                    break
            return 1.0
        return None
