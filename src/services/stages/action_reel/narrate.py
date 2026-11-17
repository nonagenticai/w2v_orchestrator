"""narrate — OPT-IN voice-over for an action reel.

Most action reels have no narration: the footage and the bed are the whole
edit. So this stage is opt-in and SKIP-CLEAN — with no ``brief.narration``
block it raises ``StageSkippedError`` and the DAG marks it ``skipped``, which is
a terminal state that counts toward completion, so ``compile``'s ``depends_on``
never blocks a run that did not ask for a voice-over.

Why it hangs off ``cut_plan`` rather than the brief alone: a voice-over is
written against the cut, and it must be able to see how long the reel is before
it is spoken over it. The dependency also keeps narration BELOW the human review
of the shot list — narrating an unapproved cut is work thrown away the moment
the reviewer reorders a shot.

Why it does NOT pause: narration here is authored by the operator (the brief
carries the lines), so there is nothing GENERATED to review — exactly the
reasoning showcase applies to its authored ``demo_video_gen``. The audio is
reviewed in the finished reel.

Mechanically this is the fleet's standard per-line TTS fan-out
(``NarrationStageBase`` → ``stage_gather`` → ``store_tts_clip``), so the clips
land keyed by ``line_id`` exactly like every other narrate stage and the compile
stage muxes them the same way. The stage NAME is the shared ``narrate`` for that
reason — the fleet's per-item timeout table and the re-narrate patcher both key
off it.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar

from src.services.stages.base import (
    ArtifactUrl,
    SlotUse,
    StageInput,
    StageOutput,
    StageOutputWithNotes,
    StagePreconditionError,
    StageSkippedError,
)
from src.services.stages.primitives.narration import (
    _CONCURRENCY as _FANOUT_CONCURRENCY,
)
from src.services.stages.primitives.narration import (
    _PER_ITEM_TIMEOUT_SECONDS as _FANOUT_PER_ITEM_TIMEOUT_SECONDS,
)
from src.services.stages.primitives.narration import NarrationStageBase

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

# --- The line-count cap and the stage budget, derived from ONE arithmetic ---
#
# The fan-out is `stage_gather(concurrency=4, per_item_timeout_seconds=600,
# per_item_retries=1)`, so the wall clock is set by WAVES, not by lines:
#
#     worst case = ceil(N / _FANOUT_CONCURRENCY) * (1 + 1 retry) * 600s
#                = ceil(N / 4) * 1200s
#
# The blueprint used to declare 1300s with a comment claiming a 1200s worst case.
# That arithmetic silently assumed ONE wave: it is right for N <= 4 and wrong for
# every brief above it — at N = 5 the true worst case is 2400s, and `narrate` is a
# PAID stage (live TTS), so the stage would be cancelled by the outer
# TimeoutMiddleware after ~22 minutes of spend with nothing stored.
#
# Fix chosen: **cap the authored line count** (option b) rather than only raise the
# ceiling (option a). `_narration_lines` reads `brief.narration.lines` verbatim —
# blanks dropped, nothing added — so N is AUTHOR-controlled and unbounded. No finite
# timeout is "provably sufficient" against an unbounded N; only a cap makes the
# budget a proof rather than a bet. And a brief that cannot be served is better
# refused at the top of the stage, before the first paid synthesize call, than
# discovered by a timeout an hour of TTS later.
#
# The ceiling then FOLLOWS from the cap instead of being a second independent guess:
# 12 lines is 3 waves, 3 * 1200s = 3600s, + 100s headroom for store/probe overhead
# (the same +100 shape `shot_analysis` uses). Twelve voice-over lines is already
# more speech than a reel-length edit can carry, so the cap refuses authoring
# mistakes, not real briefs. Both numbers move together because both are computed
# here from the primitive's own constants — the blueprint imports
# `NARRATE_TIMEOUT_SECONDS`, so it can no longer drift from the fan-out it budgets.

#: Worst case for ONE line: the initial attempt plus the single configured retry.
_PER_LINE_WORST_CASE_SECONDS = _FANOUT_PER_ITEM_TIMEOUT_SECONDS * 2

#: The most authored lines this stage will accept. Above it the run fails fast.
MAX_NARRATION_LINES = 12

#: ceil(MAX / concurrency) waves * per-line worst case, + 100s headroom.
NARRATE_TIMEOUT_SECONDS = (
    int(math.ceil(MAX_NARRATION_LINES / _FANOUT_CONCURRENCY) * _PER_LINE_WORST_CASE_SECONDS) + 100
)


class ActionReelNarrateInput(StageInput):
    pass


class ActionReelAudioClip(StageOutput):
    id: str
    line_id: str
    artifact_url: ArtifactUrl
    seconds: float
    # Speed the kept clip was synthesized at (None = backend default) and its
    # spoken word count, filled by NarrationStageBase. Optional so outputs
    # persisted before these fields existed still validate.
    speed: float | None = None
    words: int | None = None


class ActionReelNarrateOutput(StageOutputWithNotes):
    # Declared output contract — enforced once, centrally, in
    # `runs/dag_executor.py::_enforce_produces`. Empty here means the stage
    # reported success having produced nothing, which degrades the run.
    # `compile` reads `clips` to lay the voice-over. A reel WITHOUT voice-over is normal
    # and skips clean (StageSkippedError) before any TTS spend, so a COMPLETED narrate
    # holding no clips is a swallowed synth failure, never the opt-out.
    produces: ClassVar[tuple[str, ...]] = ("clips",)

    clips: list[ActionReelAudioClip]


class ActionReelNarrateStage(NarrationStageBase[ActionReelNarrateInput, ActionReelNarrateOutput]):
    name = "narrate"
    input_model = ActionReelNarrateInput
    output_model = ActionReelNarrateOutput
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (SlotUse("tts"),)
    # The stage this handler READS is the brief, not a prior stage; the value is
    # kept pointing at cut_plan (its DAG dependency) so the base class's
    # "<source_stage> output missing" error names something real if resolve_script
    # ever returns nothing.
    source_stage = "cut_plan"
    source_list_key = "lines"
    source_id_field = "line_id"
    clip_model = ActionReelAudioClip

    async def handle(
        self, payload: ActionReelNarrateInput, ctx: RunContext
    ) -> ActionReelNarrateOutput:
        # Opt-in gate BEFORE any TTS spend. Skip-clean (not a precondition
        # failure): a reel with no voice-over is the NORMAL product, not a
        # misconfigured run.
        lines = _narration_lines(ctx)
        if not lines:
            raise StageSkippedError(
                "narrate: no brief.narration.lines (action_reel voice-over is opt-in)",
                missing=["brief.narration.lines"],
            )
        # The budget guard, also BEFORE the first paid call. A brief this stage
        # cannot finish inside its own ceiling must be refused while refusing is
        # free — not cancelled mid-fan-out with the spend already made. See the
        # ceil(N / 4) * 1200s arithmetic at the head of this module.
        if len(lines) > MAX_NARRATION_LINES:
            raise StagePreconditionError(
                f"narrate: brief.narration.lines has {len(lines)} lines; this stage "
                f"synthesizes {_FANOUT_CONCURRENCY} at a time and its "
                f"{NARRATE_TIMEOUT_SECONDS}s budget covers at most "
                f"{MAX_NARRATION_LINES}. Shorten the voice-over, or split the reel.",
            )
        return await super().handle(payload, ctx)

    def resolve_script(self, ctx: RunContext) -> dict[str, object] | None:
        """The authored narration lines, normalised to the base class's contract.

        The base fan-out reads ``item["id"]`` and ``item["text"]``. An operator
        writing a brief supplies text and often no ids, so an id is derived from
        the line's position. Positional ids are stable for a given brief, which
        is what the re-narrate patcher needs to rewrite one line's text without
        re-synthesizing the rest.
        """
        return {"lines": _narration_lines(ctx)}


def _narration_lines(ctx: RunContext) -> list[dict[str, object]]:
    """``brief.narration.lines`` normalised to ``[{id, text}]``; ``[]`` when absent.

    Accepts either plain strings or objects, because both shapes show up in real
    briefs. A blank line is dropped rather than synthesized — TTS on an empty
    string is a paid call that returns silence.
    """
    brief = ctx.config_snapshot.get("brief") or {}
    raw = (brief.get("narration") or {}).get("lines") or []
    lines: list[dict[str, object]] = []
    for index, entry in enumerate(raw):
        if isinstance(entry, str):
            text, line_id = entry.strip(), f"vo-{index + 1}"
        elif isinstance(entry, dict):
            text = str(entry.get("text") or "").strip()
            line_id = str(entry.get("id") or f"vo-{index + 1}")
        else:
            continue
        if text:
            lines.append({"id": line_id, "text": text})
    return lines
