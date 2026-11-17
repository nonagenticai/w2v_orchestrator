"""cut_plan — join the selected segments to the music grid into an ordered shot list.

The stage that JOINS the two branches of this blueprint: ``segment_select``
(which seconds of footage survive) and ``music_select`` (where the beats are).
Its output is the reel's timeline — the thing ``graphics_design`` decorates,
``narrate`` may talk over, and ``compile`` renders.

**The beat-fit solver lives here** (``solve_cut_plan``). It walks the beat list
and hands each selected segment a contiguous window of beats; the segment's own
length divided by that window's length is the per-clip ``speed`` (time-warp) the
compositor applies, so the shot ENDS on the beat it was given.

The single most important rule in this module
---------------------------------------------
**Every cut time is derived ABSOLUTELY from the beat list, never cumulatively.**
A shot's ``seconds`` is always the difference of two absolute boundaries, each
derived on its own from ``beats`` (see the two sections below for where the
first boundary comes from and why each one is snapped to a frame), and never a
sum of per-shot durations. This is
not a stylistic preference: real grids carry 8-35 ms of genuine musical jitter
per beat, and a solver that adds up rounded per-shot durations (or that
synthesises a uniform ``60 / bpm`` grid instead of using the measured times)
walks off the music by tens of milliseconds within a twenty-shot reel. The
first cuts look right, the last ones are visibly late, and the bug is invisible
in any short test. Every arithmetic decision below is shaped by that rule:

* ``seconds`` is computed from two beat indices, so the error of shot *i* is
  independent of shots ``0..i-1``;
* ``seconds`` is deliberately NOT rounded — a 0.5 ms rounding per shot is
  exactly the class of error this module exists to avoid;
* the timeline position of each shot is carried on ``start_seconds`` as the
  absolute boundary it starts on, so a consumer never has to re-derive it by
  summing.

Where the timeline starts: video t=0, not ``beats[0]``
------------------------------------------------------
``compile`` lays the music bed from the head of the track against the head of
the video — there is no bed-offset field, so **track t=0 IS video t=0**. That
makes the beat times directly usable as video-time cut boundaries, but it also
means the FIRST shot cannot start at ``beats[0]``: doing so would place every
cut ``beats[0]`` (0.2-2.0 s on a real bed) earlier than the beat the viewer
hears. So the reel is anchored at video t=0: shot 0 runs from ``0.0`` up to the
first chosen beat — it covers the musical intro — and from there every boundary
is an absolute beat time exactly as heard. A grid whose first beat sits near
zero is not a special case: the origin is folded into the window BEFORE the
``MIN_SHOT_SECONDS`` filter, so a would-be sub-minimum opening shot is simply
never offered and the solver reaches for a wider span (see ``_windows``).

Why the boundaries are snapped to the frame grid
------------------------------------------------
Each shot is encoded on its own and the results are concatenated, so a renderer
can only ever produce whole frames: a planned 2.0000 s shot comes back as
2.0020 s at 23.976 fps. Those roundings are one-sided and they PREFIX-SUM down
the concat — ~300-600 ms typical over a thirty-shot reel, against a beat
tolerance of ~20 ms — which is the very drift this module exists to prevent,
re-entering through the renderer instead of the solver. The fix is to ask only
for durations the renderer can actually make: every boundary is snapped to the
output frame grid HERE, in the solver. The discipline is unchanged — each
boundary is quantised from its own absolute beat time and the durations are
differences of quantised boundaries, so the error stays bounded by half a frame
and never accumulates.

Fitting, and the repair ladder
------------------------------
A shot occupies ``k`` beats, ``k`` drawn from the musical phrase lengths
``{2, 4, 8}`` (plus ``{1}`` for a short accent). ``speed = source_len / window``
wants to sit in the "invisible" band ``(0.85, 1.20)`` where a viewer cannot see
the warp. When it does not, the repairs are tried IN THIS ORDER, cheapest and
least visible first:

a. **another phrase span** — an adjacent ``k`` often lands the same footage in
   the band for free;
b. **trim the source inward** — segments arrive 6-18 s long with real slack, so
   a too-long source is shortened to exactly its beat window and plays at 1.0x
   (no warp at all beats a warp);
c. **accept a warp inside the hard band** ``[SPEED_MIN, SPEED_MAX]``, widening
   the span choice to any beat count rather than only the phrase lengths — an
   off-phrase span is a worse musical fit than 2/4/8 but a far better one than a
   1.9x time-warp, so the two compete on cost rather than by precedence;
d. **last resort: the clip cannot honestly fill its window.** The span nearest
   its natural length is still used (so the cut stays ON the grid and the reel
   stays contiguous), the speed is clamped into the hard band *as far as the
   segment's own footage allows* — a segment shorter than the clamp needs is
   played below ``SPEED_MIN`` rather than trimmed past its approved out-point —
   and the shot is stamped ``beat_locked=False`` with a note. Never silent.

The one equation every emitted shot satisfies
---------------------------------------------
``speed == (out_seconds - in_seconds) / seconds``. It is not a coincidence of
the arithmetic, it is the CONTRACT: the renderer derives a shot's rendered
length as ``(out - in) / speed``, so a triple that does not satisfy it renders a
shot of a length nobody planned and pushes every later cut off the music by the
difference. Two consequences are load-bearing below:

* ``speed`` is derived from the EMITTED (millisecond-rounded) trim, never from
  the pre-rounding source length — a ~0.5 ms rounding lives in the only channel
  that sets shot length, and it prefix-sums down the concat;
* when repair (d) clamps the speed into the hard band, the SOURCE LENGTH is
  re-derived from the clamped speed (``speed * window``). Clamping the speed and
  keeping the natural length is the same bug wearing a repair's clothes: a 9 s
  source in a 2 s window, clamped from 4.5x to 2.0x, renders 4.5 s of reel
  inside a 2 s beat window and every later cut is 2.5 s late. That re-derivation
  is itself capped at what the segment ACTUALLY offers
  (``min(speed * window, source_len)``): in the too-short direction
  ``speed * window`` runs past the approved out-point, i.e. past footage a human
  selected and, at the end of a source, past footage that exists at all. The
  emitted speed is then allowed below ``SPEED_MIN`` — the shot is already
  ``beat_locked=False`` and noted, and a visible slow-motion beats an invented
  frame.

One segment may become more than one shot
-----------------------------------------
The walk used to be ``for segment in segments``, which quietly made SHOT COUNT
BOUNDED BY SEGMENT COUNT. A live 45 s ask came back as a 33.7 s reel off ten ~7 s
segments (69 s of footage) against a 58 s bed with 130 beats: neither the music
nor the footage ran out — the segment LIST did, at ~3.4 s of beat window each.
The honesty machinery worked perfectly and reported an 11.3 s shortfall nobody
needed to have.

So the walk runs a queue instead of a list. Repair (b) trims an over-long segment
to its beat window and leaves the rest of that approved footage unused; when a
length was requested and the reel is still short, the segment goes back on the
queue with its in-point advanced past what was already shown (``_Take``), BEHIND
every segment still waiting for its first shot. Breadth before depth: the reel
keeps the variety the selection was drawn for, and second helpings are taken only
because the ask is otherwise unreachable. Two guarantees hold it honest —

* **no footage is ever replayed.** A re-take is the segment's unshown tail, never
  a loop. When the tails run out the reel is short, and it says so;
* **without a target nothing changes.** No requested length means no gap to
  close, so the plan is the approved selection at one shot each, exactly as before.

``notes`` is the run's degradation channel — a completed stage with non-empty
notes surfaces the run as ``degraded`` — so it records only genuine degradation
(no grid, a clamped shot, an unfetchable source, an exhausted grid). Trims and
in-band warps are the solver doing its job and are not notes.

**This stage pauses**: the shot list is the reel. Reordering or dropping a shot
here is free; doing it after the compile is a re-render. It stays the
``compile_cut_stage`` for the same reason — a timeline reorder must re-run from
here, not from the LLM selection above it (which would re-draw a different edit
entirely and discard the reviewer's approval).
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, NamedTuple, cast

from pydantic import Field

from src.services.stages.action_reel.cadence import DEFAULT_MASTER_FPS, resolve_master_fps
from src.services.stages.action_reel.target_duration import target_seconds
from src.services.stages.base import (
    StageHandler,
    StageInput,
    StageOutput,
    StageOutputWithNotes,
    StagePreconditionError,
)
from src.services.stages.shot_presign import refresh_shot_asset_urls

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

# The time-warp envelope the solver is allowed to work inside. An editorial
# decision, not a solver detail: under 0.5x the footage judders, over 2.0x the
# action reads as a glitch. NARROWER than what the compiler itself accepts
# ([0.25, 4.0]) — the compiler's bounds are what the renderer survives, these
# are what a viewer tolerates.
#
# The two are NOT interchangeable and the editorial pair does not simply win.
# Repair (d) knowingly emits below `SPEED_MIN` rather than read footage past the
# out-point a reviewer approved (the reasoning is at the clamp itself): an
# over-slow warp is a visible compromise on a shot already stamped
# `beat_locked=False`, inventing footage is a silent one. So `SPEED_MIN` is what
# the ladder AIMS for, and `COMPILER_SPEED_MIN` below is the floor no emitted
# shot may cross whatever it costs.
SPEED_MIN = 0.5
SPEED_MAX = 2.0
#: The compiler's own hard floor on a shot's speed. Mirrors
#: `tests/contracts/compile_request.schema.json` -> `$defs/Shot.properties.speed`
#: (`minimum: 0.25`; its `maximum: 4.0` is already covered by `SPEED_MAX`), and
#: is named here so the two cannot drift apart in silence. Crossing it is not a
#: worse-looking shot: the compiler 422s the WHOLE shots list, so one out-of-band
#: speed loses the entire paid render — after every human review gate approved
#: it — and the user is shown "render failed" with no reel at all.
COMPILER_SPEED_MIN = 0.25
#: The "invisible" band. Inside it a time-warp reads as the footage's own pace;
#: outside it the viewer starts to feel the clip being pushed or dragged. Every
#: repair below is an attempt to land back in here.
PREFERRED_SPEED_MIN = 0.85
PREFERRED_SPEED_MAX = 1.20
#: Musical phrase lengths, in beats. A cut on 2/4/8 lands where the ear already
#: expects a change; a cut on 3 or 5 is legal but reads as a mistake unless the
#: alternative is worse (see the cost function).
BEAT_SPANS: tuple[int, ...] = (2, 4, 8)
#: A single beat is allowed only for a genuinely short accent segment — at any
#: normal tempo one beat is under a second, which is a flash, not a shot.
ACCENT_SPAN = 1
ACCENT_MAX_SOURCE_SECONDS = 1.5
#: Ceiling on the off-phrase spans repair (c)/(d) may reach for. A 12s segment
#: (segment_select's own maximum) at 120 BPM needs 24 beats; 32 leaves headroom
#: for slower tempos without letting one shot eat a whole reel.
MAX_FREE_SPAN = 32
#: Minimum RENDERED shot length. Below this the cut is a flicker regardless of
#: how well it fits the grid, so such a beat window is never offered at all.
MIN_SHOT_SECONDS = 0.6
#: The substring ``_empty_plan_error`` filters "this approved segment did not
#: reach the reel" notes by. Deliberately the VERBLESS fragment, so it matches
#: both the per-segment note ("segment X *was* not placed: …") and the
#: grid-exhausted note ("N selected segments *were* not placed"). The filter used
#: to hardcode "was not placed", which matched the first and silently missed the
#: second — so on a pure grid exhaustion the sample came back empty and an error
#: with a full list of reasons to give ended with "Reasons: not recorded". Number
#: agreement is a property of English, not of the diagnosis; the marker must not
#: depend on it. Every note that should be sampled must contain this fragment.
_UNPLACED_NOTE_MARKER = "not placed"
#: Tolerance for matching a downbeat time back to its beat index. Downbeats come
#: from the same analyzer pass as the beats, so they normally match to the float;
#: the tolerance only absorbs a serialisation round-trip.
_DOWNBEAT_TOLERANCE = 0.02
#: What a shot plays at when there is no grid to fit to: no time-warp. Kept as a
#: name because "1.0" appearing bare in the degradation path reads like a
#: guessed value rather than a deliberate "we did not warp anything".
STUB_SPEED = 1.0
#: Frame grid the solver quantises against when the caller supplies no rate.
#: Held equal to :data:`cadence.DEFAULT_MASTER_FPS` so an unparameterised solve
#: plans on exactly the grid compile would have encoded at for a shoot it could
#: not probe — the two must never be independently editable.
#:
#: PRODUCTION NEVER TAKES THIS DEFAULT. The handler resolves the rate from the
#: footage probe via ``resolve_master_fps(ctx.config_snapshot, probe)`` and
#: passes it in, and that function returns an ``int`` off the legal broadcast
#: set. So every rate the solver has ever seen in prod is a WHOLE number, and
#: half a frame is at worst 1/48 s — the drift budget this module is built
#: around is an integer-grid budget.
#:
#: This read ``24000 / 1001`` with a comment reasoning about the 20.9 ms
#: residual of a 23.976 NTSC grid "which is what the compile stage encodes at".
#: That arithmetic never ran: cadence cannot return a non-integer rate, so no
#: NTSC grid can reach the solver from any live path. The parameter survives the
#: correction because it is a real seam — the tests solve through it directly,
#: and a caller that has already resolved its own rate should pass it rather
#: than reach into the solver. Should NTSC ever be genuinely required, it is
#: cadence's legal set that has to admit it first; changing this line alone
#: would only desynchronise the plan from the encode.
DEFAULT_FRAME_RATE = float(DEFAULT_MASTER_FPS)
#: Slack allowed when checking that shot N starts where shot N-1 ended. Not a
#: tolerance on the edit — a real gap is orders of magnitude larger — only on
#: ``a + (b - a) == b`` in binary floating point, which is not exact.
_CONTIGUITY_EPSILON = 1e-9
#: Least UNUSED footage a segment must still hold before the walk will come back
#: for a second shot from it (see ``solve_cut_plan``). Set to ``MIN_SHOT_SECONDS``
#: because that is the shortest window the grid ever offers: a leftover at least
#: that long can be placed at a speed inside the hard band on any real grid (the
#: offered windows run densely from ``MIN_SHOT_SECONDS`` upward, so some window
#: lands in ``[leftover / SPEED_MAX, leftover / SPEED_MIN]``), while a shorter one
#: would only reach the grid through repair (d) — a clamped, ``beat_locked=False``
#: shot with a note, which is a degradation to buy length with, not a bargain.
MIN_REUSE_SECONDS = MIN_SHOT_SECONDS
#: How far the planned reel may fall short of a requested length before it counts
#: as a TRUNCATED edit rather than grid-snap noise. One frame at any legal rate is
#: under 42 ms; anything past a quarter second means the music (or the selection)
#: ran out before the author's target did, which the author must be told.
TRUNCATION_TOLERANCE_SECONDS = 0.25
#: How far a reel may run OVER the requested length before the over-run stops
#: being the stopping rule working and becomes a degradation the reviewer has to
#: be told about, as a fraction of the target.
#:
#: A fraction rather than "longer than the final shot": both walks emit the shot
#: that crosses the target and then stop, so ``overrun`` is by CONSTRUCTION
#: strictly less than the final shot's length — gating on that would be a note
#: that can never fire. What the stopping rule does NOT bound is how long that
#: one crossing shot is: a 45s clip against a 20s ask ships a reel more than
#: twice the requested length, entirely within the rule, and ``overrun_seconds``
#: is read by nothing (unlike ``shortfall_seconds``, which the frontend renders),
#: so the reviewer approved that gate with nothing to see. A quarter of the ask
#: leaves an ordinary reel — which overshoots by one ~3s shot on a ~30s target —
#: unbadged, which is the property the notes channel is worth anything for.
OVERRUN_NOTE_FRACTION = 0.25


class CutPlanInput(StageInput):
    """No typed inputs — segments + grid are read from the two prior branches."""


class PlannedShot(StageOutput):
    """One shot of the reel, in playback order.

    ``id`` is the ``segment_select`` segment id, NOT a fresh uuid: the plan is a
    projection of the approved selection, so keeping the id makes a shot
    traceable back to the rationale a human signed off on (and lets a
    regeneration diff the two). When one segment yields a SECOND shot (its unused
    tail — see the module docstring) that shot's id carries a ``#2`` suffix: the
    id is a join key downstream (compile's ``line_id``, a graphics card's
    ``shot_id``), so a duplicate would paste one shot's overlay and narration
    onto the other.

    ``in_seconds`` / ``out_seconds`` are the SOURCE trim: the compositor plays
    that span of the source (see the compile stage's ``clip_lead_seconds``
    mapping), time-warped by ``speed`` so it fills exactly ``seconds`` of the
    reel's timeline. The three are one equation — ``speed`` is always
    ``(out_seconds - in_seconds) / seconds`` — and ``seconds`` is the beat
    window, never the source length.
    """

    id: str
    source_id: str
    video_url: str = ""
    in_seconds: float = 0.0
    out_seconds: float = 0.0
    seconds: float = 0.0
    speed: float = STUB_SPEED
    #: Index into ``music_select.beats`` this shot STARTS on; ``-1`` when there
    #: was no grid to fit to (the even-spacing degradation).
    beat_index: int = -1
    #: Where this shot starts on the VIDEO timeline, which is also the track
    #: timeline (the bed is laid at t=0 against t=0, no offset). Every value but
    #: the first shot's is a beat time snapped to the output frame grid; the
    #: first is ``0.0``, so the opening shot covers the musical intro. Carried so
    #: no consumer has to reconstruct the timeline by summing durations — the one
    #: operation this module exists to avoid.
    start_seconds: float = 0.0
    #: False when the clip could not honestly fill its beat window and its speed
    #: had to be clamped into the hard band (repair (d)), or when there was no
    #: grid at all. A consumer that must not ship a loose cut checks THIS.
    beat_locked: bool = True
    rationale: str = ""


class CutPlanOutput(StageOutputWithNotes):
    # Declared output contract — enforced once, centrally, in
    # `runs/dag_executor.py::_enforce_produces`. Empty here means the stage
    # reported success having produced nothing, which degrades the run.
    # `compile`, `graphics_design` and `narrate` all read `shots`; the shot list IS the
    # reel.
    produces: ClassVar[tuple[str, ...]] = ("shots",)

    shots: list[PlannedShot]
    total_seconds: float = 0.0
    #: The bed this plan is cut against, carried forward so compile does not have
    #: to re-read music_select to lay the music track.
    music_track_id: str = ""
    music_url: str = ""
    bpm: float = 0.0
    #: False when music_select could not fetch a grid (analyzer down, or an
    #: un-beat-trackable bed). The solver degrades to even spacing in that case
    #: rather than failing — recorded here so the degradation is visible.
    beat_grid_available: bool = False
    #: Retained as an explicit "this plan came out of the real solver" flag: a
    #: consumer must not infer solvedness from ``speed == 1.0``, which a solved
    #: plan may legitimately choose for every shot. True means the beat-fit solver
    #: did NOT run and the shots are the ``_even_spacing`` degradation — the e2e
    #: proof harness trusts this flag, so it is computed, never asserted.
    solver_stubbed: bool = False
    #: The frame grid this plan was quantised against — the SAME rate compile
    #: conforms its master to (both ask ``cadence.resolve_master_fps``). Carried so
    #: a disagreement between plan and render is visible in the output rather than
    #: only as drift in the finished file.
    frame_rate: float = 0.0
    #: The length the author asked for, ``0.0`` when they asked for none.
    target_seconds: float = 0.0
    #: ``target_seconds - total_seconds`` when the reel came out SHORTER than
    #: requested, else ``0.0``. A structured field and not only a note because a
    #: truncated edit is the one degradation an operator must not be able to skim
    #: past: real beds are ~58s, so a 90s request quietly becomes a 58s reel.
    shortfall_seconds: float = 0.0
    #: ``total_seconds - target_seconds`` when the reel came out LONGER than
    #: requested, else ``0.0``. The mirror of ``shortfall_seconds``, and it exists
    #: because that gate is one-sided: a plan that ran long reported a shortfall
    #: of 0.0 and looked exactly like one that hit the target.
    #:
    #: A SMALL over-run carries no note: both walks stop by emitting the shot that
    #: CROSSES the target, so overshooting is normal and noting every instance
    #: would surface every length-targeted reel as ``degraded``. A RUNAWAY one
    #: does earn a note — see ``OVERRUN_NOTE_FRACTION`` for where the line is and
    #: why it is not "longer than the final shot". This number alone reached no
    #: consumer: nothing reads ``overrun_seconds`` (unlike ``shortfall_seconds``,
    #: which the frontend renders), so a 20s ask that planned a 45s reel showed
    #: the reviewer a green gate with nothing to see.
    overrun_seconds: float = 0.0
    #: How many APPROVED segments the beat grid ran out before it could place —
    #: i.e. footage a human said yes to and the reel does not contain. Structured
    #: for exactly the reason ``shortfall_seconds`` is: a truncated edit is the
    #: degradation an operator must not be able to skim past, and this is the
    #: same truncation counted in segments instead of seconds. It was a number
    #: interpolated into a note and nowhere else, so no consumer — not the e2e
    #: proof harness, not the run-status roll-up — could gate on it.
    #:
    #: Counts FIRST takes only (a queued re-take is spare material the walk
    #: offered itself, not a selection anybody approved), and it is ``0`` on the
    #: no-grid ``_even_spacing`` path: that degradation has no grid to exhaust,
    #: and the length it stops at is the requested one, reported as an over/under
    #: run rather than as dropped selections.
    unplaced_segments: int = 0
    #: How many approved segments were dropped BEFORE the solver ever saw them,
    #: because ``footage_probe`` presigned no fetchable url for their source. The
    #: same class of loss as ``unplaced_segments`` — footage a human approved that
    #: the reel does not contain — and it sat, for exactly as long, as a count
    #: interpolated into a note string and nothing else. A note is prose: no gate,
    #: no roll-up and no proof harness can read it, and pattern-matching one is
    #: worse than not reading it at all. The named ids stay in the note (the count
    #: says how much was lost, the note says which), and both are rendered from
    #: this one list so they cannot drift.
    unfetchable_segments: int = 0
    #: How many APPROVED segments the walk never reached because the reel had
    #: already reached its requested LENGTH. Deliberately a separate number from
    #: ``unplaced_segments`` and not folded into it: the two are opposite
    #: verdicts. Running out of beat grid is a failure of the bed against the
    #: selection (the reviewer should pick a longer track); stopping at the
    #: target is the stopping rule working exactly as designed, and badging it as
    #: "unplaced" would teach a reviewer to ignore a field that also carries the
    #: real one.
    #:
    #: The three readings:
    #:
    #: * ``0`` with a target set — every approved segment was walked; the reel is
    #:   the whole selection, not a prefix of it.
    #: * ``0`` with no target — vacuous. Without a length to fill, the walk never
    #:   stops early, so this can only ever be 0 and says nothing.
    #: * ``> 0`` — the plan is the TOP-SCORING prefix of a longer selection.
    #:   ``segment_select`` over-selects (``_COVERAGE_FACTOR = 2.0``) so the
    #:   solver has choices, so a healthy 45s reel routinely leaves ~45s of
    #:   approved material behind. That is fine when every SOURCE still made the
    #:   cut and it is not fine when one did not — which is what
    #:   ``sources_absent`` below answers, and why this number alone is reported
    #:   without a note.
    #:
    #: Counts FIRST takes only, for the same reason ``unplaced_segments`` does: a
    #: queued re-take is spare material the walk offered itself, not a selection
    #: a human approved and then did not see.
    segments_unused_at_target: int = 0
    #: The source ids (uploaded clips) that contributed ZERO shots to this plan.
    #:
    #: This is the fact an operator actually needs and the one nothing reported.
    #: A reviewer who uploaded ten clips and asked for 45s got a reel built from
    #: the top-scoring ~45s of a ~90s selection, with ``unplaced_segments: 0``
    #: and ``shortfall_seconds: 0.0`` — both honestly zero — while whole clips
    #: they had shot and approved were absent from their own reel, named nowhere.
    #: A per-segment count cannot express that: losing three of nine segments off
    #: one source is trivia if the source still appears, and losing the one
    #: segment a source had is the reviewer's whole afternoon.
    #:
    #: The three readings:
    #:
    #: * empty — every source in the selection reached the reel.
    #: * some ids, and the reel REACHED its target length — usual, and usually a
    #:   judgement the reviewer will accept: the brief deliberately asks for ~2x
    #:   the reel so the planner can choose, so the surplus is the system
    #:   working. Reported here and NOWHERE ELSE.
    #: * some ids, and the reel FELL SHORT (or had no target) — footage was
    #:   approved, the reel had room for it, and it still did not make the cut.
    #:   Actionable: pick a longer bed, or re-select. This is the only case that
    #:   also emits a note.
    #:
    #: That asymmetry is the whole design and it is NOT cosmetic. ``notes`` is a
    #: status channel — ``api/routes/runs/_shared.py:96`` sets ``degraded =
    #: completed and bool(output["notes"])`` — so a note here flips the run. The
    #: first cut of this fix noted every absence and turned **181 of 200**
    #: realistic plans (10 clips, 20 segments in score-DESC order, a 45s ask)
    #: from clean to degraded. Nine healthy reels in ten wearing a damage badge
    #: is not honesty; it is a badge reviewers learn to click past, which would
    #: have buried the real cases as thoroughly as the original silence did. The
    #: FIELDS are lossless and free — they do not touch ``degraded`` — so the
    #: full fact lives here on every path and only the status change is gated.
    #:
    #: When the note does fire it names these same ids, rendered from this one
    #: list so the prose and the field cannot drift (the same discipline as
    #: ``unfetchable_segments`` and its note).
    #:
    #: Reporting-only, and verified as such: over 400 randomised fixtures the
    #: change moved no shot, order, duration, speed or beat index
    #: (``runs=400 shot/unplaced diffs=0``).
    sources_absent: list[str] = Field(default_factory=list)
    #: ``len(sources_absent)``, carried separately so a gate, a roll-up or the
    #: e2e proof harness can threshold on a number without having to parse a list
    #: — the same reason ``unfetchable_segments`` is a count beside its note.
    sources_absent_count: int = 0


@dataclass(frozen=True)
class _Take:
    """One visit to a segment: which slice of it the walk is about to place.

    ``take == 1`` is the segment as ``segment_select`` approved it. A later take
    is the SAME segment's still-unused tail — ``in_seconds`` advanced to where the
    previous take's out-point landed, ``out_seconds`` unchanged — so a second shot
    from one segment shows footage the reel has not shown yet. Nothing is ever
    replayed: this is the opposite of looping, and it is why a genuinely short
    corpus still runs out and still reports a shortfall.
    """

    segment: dict[str, object]
    in_seconds: float
    out_seconds: float
    take: int

    @property
    def shot_id(self) -> str:
        """The emitted ``PlannedShot.id``. Shot ids are JOIN KEYS downstream
        (compile keys narration off ``line_id == id``, graphics_design pins a card
        to a ``shot_id``), so two shots from one segment must not share one — a
        duplicate would silently paste one shot's card and voice-over onto the
        other. The suffix keeps the segment traceable while making it unique.

        The separator is ``-t`` and NOT ``#``: the compiler constrains a shot id to
        ``^[A-Za-z0-9_.-]+$`` (``video_compiler_service`` ``_SAFE_ID_PATTERN``) because
        it interpolates the id into scratch FILE PATHS and inline into ffmpeg
        FILTERGRAPHS. A ``#`` is outside that charset, so the whole compile request
        was rejected with a 422 and the run died after every earlier stage had
        succeeded. Any future suffix must stay inside that charset."""
        base = _text(self.segment.get("id"))
        return base if self.take == 1 else f"{base}-t{self.take}"


class CutPlanSolution(NamedTuple):
    """What the solver produced: the shots, its notes, and what it could NOT place.

    A ``NamedTuple`` and not a bare tuple because ``unplaced_segments`` was the
    third thing the walk already knew and the only one it threw away — it existed
    solely as a number interpolated into a note string, which no consumer can
    gate on. Same reasoning as ``CutPlanOutput.shortfall_seconds``: a truncated
    edit must be a FIELD, not prose.
    """

    shots: list[dict[str, object]]
    notes: list[str]
    #: Approved (first-take) segments the grid ran out before placing. See
    #: ``CutPlanOutput.unplaced_segments``, which this feeds verbatim.
    unplaced_segments: int = 0
    #: First-take segments the walk never reached because the target LENGTH was
    #: already met. See ``CutPlanOutput.segments_unused_at_target``, which this
    #: feeds verbatim. Kept apart from ``unplaced_segments`` on purpose: hitting
    #: the target is the stopping rule succeeding, exhausting the grid is not.
    segments_unused_at_target: int = 0
    #: Source ids in ``segments`` that no emitted shot came from. A tuple and not
    #: a list because this is a fact about a finished walk, and the handler
    #: renders both the field and the note from it. Feeds
    #: ``CutPlanOutput.sources_absent``.
    sources_absent: tuple[str, ...] = ()
    #: True when the walk stopped because the BEAT GRID ran out — no window from
    #: the cursor on could hold another shot. This cannot be inferred from
    #: ``unplaced_segments``: the walk guard is ``cursor < len(queue)``, so the
    #: grid-exhausted break always has at least one pending take to count and
    #: ``unplaced_segments`` is therefore always >= 1 on that path. Reading the
    #: count alone made "the music bed is too short" indistinguishable from
    #: "the footage is too short for any window", and the reviewer was sent to
    #: fix the footage. The music-bed cause needs its OWN flag.
    grid_exhausted: bool = False


@dataclass(frozen=True)
class _Fit:
    """One segment's placement on the grid. Plain data — no pydantic, no ctx."""

    span: int
    beat_index: int
    start_seconds: float
    seconds: float
    in_seconds: float
    out_seconds: float
    speed: float
    beat_locked: bool


@dataclass(frozen=True)
class _Unplaceable:
    """A segment no window on this grid can hold at a speed the COMPILER accepts.

    Deliberately NOT ``None``. ``None`` means the grid ran out, which is terminal
    — nothing queued behind this segment can be placed either, so the walk stops.
    This is local: the beat the walk stands on is untouched and the next segment
    may fill it perfectly, so the walk skips this one and carries on. ``reason``
    travels with it because the caller owns the note channel and a bare "dropped"
    tells a reviewer nothing about which of their approved moments went, or why.
    """

    reason: str


def solve_cut_plan(
    segments: list[dict[str, object]],
    beats: list[float],
    *,
    downbeats: list[float] | None = None,
    target_seconds: float | None = None,
    frame_rate: float = DEFAULT_FRAME_RATE,
) -> CutPlanSolution:
    """Fit ``segments`` to ``beats``; return a :class:`CutPlanSolution` of plain data.

    Pure function on purpose: the whole value of this stage is arithmetic over
    two lists of numbers, and arithmetic that can only be exercised through a
    ``RunContext`` does not get the tests it deserves. The handler below is a
    thin adapter — read prior outputs, call this, join the source URLs on.

    ``frame_rate`` is the rate the reel will be ENCODED at; every boundary is
    snapped to it so the plan only asks for durations the renderer can produce.

    Each returned shot is a ``PlannedShot`` field mapping minus ``video_url``
    (which the handler joins from ``footage_probe``).

    Raises ``StagePreconditionError`` when the grid is not strictly increasing —
    an upstream defect, surfaced as one.
    """
    if frame_rate <= 0 or not math.isfinite(frame_rate):
        raise StagePreconditionError(
            f"cut_plan: frame_rate must be a positive finite rate, got {frame_rate!r}",
            missing=["frame_rate"],
        )
    if not beats:
        # The degradation still has to honour the LENGTH that was asked for.
        # `segment_select` deliberately over-selects (~2x coverage) so the solver
        # has material to choose from, so emitting every segment at its natural
        # length turns a 20s request into a 40s reel.
        return _even_spacing(segments, target_seconds)
    _validate_grid(beats)

    downbeat_indices = _downbeat_indices(beats, downbeats or [])
    shots: list[dict[str, object]] = []
    notes: list[str] = []
    # Approved segments that did not reach the reel — the grid ran out before
    # them, or no window on it could hold them at a speed the compiler accepts.
    # Accumulated here rather than computed at the return so the count the notes
    # quote and the count the output carries are the SAME running number — the
    # two drifting apart is how a prose-only number gets "structured" and still
    # lies.
    unplaced = 0
    # Set only where the walk breaks out because no beat window is left. See
    # `CutPlanSolution.grid_exhausted` for why the count cannot stand in for it.
    grid_exhausted = False
    # Approved segments the walk never reached because the reel was already as
    # long as the author asked for. NOT added to `unplaced` above: that number is
    # a failure report (the bed ran out before the selection did) and this one is
    # a success report (the selection outlasted the ask), and a reviewer who
    # learns that the field mixes the two stops reading it. Counted at the break
    # rather than derived at the return because only the walk knows where it
    # stopped — a segment id can appear in `shots` from an earlier take.
    unused_at_target = 0
    # The walk's ONLY carried state is a beat INDEX. Not a running clock: an
    # index cannot drift, a clock can. Every time below is read out of `beats`.
    index = 0
    elapsed = 0.0
    # The video-time position the next shot MUST begin on. Purely a guard: it is
    # compared against, never emitted, so this running sum can never reach a cut.
    boundary = 0.0

    # The work list, walked as a FIFO. It starts as one take per segment — which
    # is what it stayed until this queue existed, and that was the bug: shot count
    # was bounded by SEGMENT count, so ten ~7s segments against ~3.4s beat windows
    # could only ever build ~34s of reel however long the bed and the ask were.
    # Repair (b) trims each over-long segment to its window and leaves the REST of
    # that approved footage on the floor; a length-targeted walk that is still
    # short may come back for it (see `_Take`). Appending eagerly is safe because
    # the walk's stopping rules are unchanged — a re-take is simply never reached
    # once `elapsed` has crossed the target.
    queue: list[_Take] = [
        _Take(
            segment=segment,
            in_seconds=_as_float(segment.get("in_seconds")),
            out_seconds=_as_float(segment.get("out_seconds")),
            take=1,
        )
        for segment in segments
    ]

    cursor = 0
    while cursor < len(queue):
        take = queue[cursor]
        segment = take.segment
        remaining = None if target_seconds is None else max(0.0, target_seconds - elapsed)
        # Whether the NEXT shot changes source is what decides where THIS
        # shot ends — and the next shot begins on that same beat. Preferring a
        # downbeat there is how "a section change lands on a downbeat" is
        # implemented without ever skipping beats: skipping would open a gap
        # between two contiguous shots, and a gap in a cut list is a black frame.
        next_take = queue[cursor + 1] if cursor + 1 < len(queue) else None
        next_source_differs = next_take is not None and _text(
            next_take.segment.get("source_id")
        ) != _text(segment.get("source_id"))

        fit = _fit_segment(
            beats,
            downbeat_indices,
            index,
            in_seconds=take.in_seconds,
            out_seconds=take.out_seconds,
            score=_as_float(segment.get("score")),
            section_change=next_source_differs,
            remaining=remaining,
            frame_rate=frame_rate,
        )
        if fit is None:
            # Out of grid: the track is shorter than the selection. Emitting a
            # shot past the last beat would put footage over silence. Only FIRST
            # takes are counted as "not placed": a queued re-take is spare
            # material the walk offered itself, not a selection a human approved
            # and then did not see.
            unplaced += sum(1 for pending in queue[cursor:] if pending.take == 1)
            grid_exhausted = True
            notes.append(
                # Must contain `_UNPLACED_NOTE_MARKER` — `_empty_plan_error` samples
                # its reason lines by it. It reads "were" here and "was" in
                # `_record_unplaceable`; the marker is verbless precisely so both
                # are sampled. Before that, this note was invisible to the filter
                # and the error ended "Reasons: not recorded".
                f"beat grid exhausted after {len(shots)} shots; "
                f"{unplaced} selected segments were {_UNPLACED_NOTE_MARKER}"
            )
            break

        if isinstance(fit, _Unplaceable):
            # Not terminal, unlike the exhausted grid above: the beat the walk
            # stands on is still free and the segment behind this one may fill it
            # perfectly, so the walk skips this take and keeps its index. Dropping
            # the shot costs the reviewer one approved moment; EMITTING it costs
            # the whole reel — the compiler validates the shots list as a unit and
            # rejects all of it over one out-of-band speed, so the render dies
            # after every review gate has already approved it. A re-take is not
            # counted or noted: it is spare material the walk offered itself, and
            # the length it does not buy is already reported as a shortfall.
            unplaced += _record_unplaceable(take, fit, notes)
            cursor += 1
            continue

        # Contiguity, asserted where it is produced rather than only in the
        # tests. A gap between two shots is a black frame and an overlap is a
        # dropped one; either way every LATER cut is off the music, so this must
        # fail loudly at the shot that broke it and not silently ship a reel.
        if fit.beat_index != index:
            raise AssertionError(
                f"cut_plan: shot {len(shots)} was placed at beat {fit.beat_index} while "
                f"the walk stood at beat {index} — the reel would gap or overlap"
            )
        if abs(fit.start_seconds - boundary) > _CONTIGUITY_EPSILON:
            raise AssertionError(
                f"cut_plan: shot {len(shots)} starts at {fit.start_seconds:.6f}s but the "
                f"previous shot ended at {boundary:.6f}s"
            )

        if not fit.beat_locked:
            # Four decimals, not two. This note is the ONLY place a human sees the
            # number the compiler will validate, and `:.2f` rounds an illegal
            # 0.24984x into a legal-looking "0.25x" — a reviewer reads a shot that
            # is inside the contract and approves a render that 422s. The point of
            # printing evidence is that a violation must be VISIBLE as one, so the
            # precision has to be finer than the distance a bug can hide in
            # (a millisecond of trim over a ~2s window is ~1e-4 of speed).
            notes.append(
                f"segment {_text(segment.get('id'))}: no beat window this clip can fill "
                f"at a usable speed; placed on the grid at a clamped {fit.speed:.4f}x "
                f"(beat_locked=False)"
            )

        shots.append(
            {
                "id": take.shot_id,
                "source_id": _text(segment.get("source_id")),
                "in_seconds": fit.in_seconds,
                "out_seconds": fit.out_seconds,
                "seconds": fit.seconds,
                "speed": fit.speed,
                "beat_index": fit.beat_index,
                "start_seconds": fit.start_seconds,
                "beat_locked": fit.beat_locked,
                "rationale": _text(segment.get("rationale")),
            }
        )
        # Whatever this take did NOT use goes back on the queue for a second
        # helping, but only behind every segment still waiting for its first.
        _queue_retake(queue, take, fit, target_seconds)

        # The next shot starts exactly where this one ended — contiguity is an
        # index handoff, not an arithmetic one.
        index = fit.beat_index + fit.span
        boundary = fit.start_seconds + fit.seconds
        # `elapsed` is a BUDGET counter for the target length only. It never
        # reaches a shot: no emitted time is derived from it, so the fact that it
        # is a running sum cannot put a cut off the beat.
        elapsed += fit.seconds
        if target_seconds is not None and elapsed >= target_seconds:
            # The reel is as long as it was asked to be, so the walk stops — that
            # is correct and NOTHING here changes it. What was wrong is that
            # everything still queued behind this shot was then discarded in
            # total silence: `unplaced` does not rise on this path, so a 45s reel
            # cut from the top-scoring half of a ~90s selection reported
            # `unplaced_segments: 0`, `shortfall_seconds: 0.0`, and a clip the
            # operator uploaded could be absent from their own reel with nothing
            # anywhere naming it. Count what is left, and count it SEPARATELY.
            # Only first takes: a queued re-take is the walk's own spare
            # material, and counting it would inflate the number with footage
            # nobody selected.
            unused_at_target = sum(1 for pending in queue[cursor + 1 :] if pending.take == 1)
            break
        cursor += 1

    sources_absent = _absent_sources(segments, shots)
    # "Did the reel come out the length it was asked for" is what decides whether
    # the absence above is a degradation or the system working — see
    # `_absent_sources_notes`. Read off `elapsed` rather than off the break,
    # because the target can also be met by a walk that simply ran out of queue
    # having already filled the ask.
    reached_target = target_seconds is not None and elapsed >= target_seconds
    notes.extend(
        _absent_sources_notes(
            sources_absent, segments, target_seconds, reached_target=reached_target
        )
    )
    return CutPlanSolution(shots, notes, unplaced, unused_at_target, sources_absent, grid_exhausted)


def _absent_sources(
    segments: list[dict[str, object]], shots: list[dict[str, object]]
) -> tuple[str, ...]:
    """Source ids present in the selection that no emitted shot came from.

    Computed over the FINISHED shot list rather than over whatever the walk did
    not consume, because those are different questions and only this one is the
    reviewer's: a source whose second segment was dropped is still in the reel,
    and a source whose only segment was dropped is not — and the walk cannot tell
    them apart while it is still running. Every stopping reason feeds this, so it
    is honest on the grid-exhausted and unplaceable paths too, not only at the
    target break that motivated it.

    Selection order, de-duplicated: the ids are read by a human against the clip
    list they uploaded, and set order would shuffle that list on every run of an
    otherwise identical plan.
    """
    placed = {_text(shot.get("source_id")) for shot in shots}
    absent: list[str] = []
    for segment in segments:
        source_id = _text(segment.get("source_id"))
        if source_id and source_id not in placed and source_id not in absent:
            absent.append(source_id)
    return tuple(absent)


def _absent_sources_notes(
    absent: tuple[str, ...],
    segments: list[dict[str, object]],
    target_seconds: float | None,
    *,
    reached_target: bool,
) -> list[str]:
    """The absent-source note, or NOTHING — and NOTHING is the important half.

    ``notes`` is not a neutral channel. ``api/routes/runs/_shared.py:96`` reads

        response.degraded = so.status == "completed" and bool(output.get("notes"))

    so ANY note on a completed stage flips the whole run to ``degraded``. That
    makes the note a status change, not a remark, and it is why this function
    exists instead of an ``if absent:`` at the call site.

    The first cut of this fix noted every absence, and that was a regression as
    bad as the silence it replaced: over 200 realistic plans (10 clips, 20
    segments in the score-DESC order ``_merge_batch_draws`` actually produces, a
    45s ask) **181 of 200 flipped from clean to degraded**. Nine healthy reels in
    ten carrying a damage badge is not honesty, it is noise — and a badge that
    fires on nine reels in ten is one reviewers learn to click past, which would
    have buried the real cases too.

    So the gate is not a ratio ("more than half the sources") — a threshold is a
    magic number nobody can defend at a review gate. It is whether the absence is
    EXPLAINED by something the reviewer already asked for:

    * **The target was reached** — the reel is the length the brief asked for.
      ``segment_select`` over-selects (``_COVERAGE_FACTOR = 2.0``) precisely so
      the solver can choose, so leaving clips unused is the system doing its job.
      The fields still carry it; NO note, and the run stays clean.
    * **The target was NOT reached** (a shortfall, an exhausted grid, no target
      at all) — footage was approved, the reel had room for it, and it still did
      not make the cut. Nothing explains that away, the reviewer can act on it,
      and the run genuinely is degraded. Note it, and name the ids.

    Note the asymmetry this creates on the no-grid ``_even_spacing`` path: there,
    absence can ONLY arise from the target break (a walk that is short of the ask
    has emitted every segment it had), so this note never fires there and the
    fields are the whole report. That is correct, not an oversight.

    The ids the note names are the same tuple the fields carry, rendered once, so
    prose and field cannot drift.
    """
    if not absent or reached_target:
        return []
    sources = {_text(segment.get("source_id")) for segment in segments}
    sources.discard("")
    short_of = f" of the {target_seconds:.1f}s it was asked for" if target_seconds else ""
    return [
        f"SOURCES ABSENT: {len(absent)} of the {len(sources)} sources in the selection "
        f"contributed no shots and are missing from this reel: {', '.join(absent)} — and the "
        f"plan still stopped SHORT{short_of}, so this is approved footage the reel had room "
        "for; the notes above say what stopped it"
    ]


def _queue_retake(queue: list[_Take], take: _Take, fit: _Fit, target_seconds: float | None) -> None:
    """Offer the footage this take did NOT use back to the walk, at the BACK of
    the queue — behind every segment still waiting for its first shot, so breadth
    beats depth and the reel keeps the variety the selection was drawn for.

    Guarded on a target because without one there is no length to fill: the plan
    is then exactly the approved selection at one shot each, as it was before the
    queue existed. The leftover must clear ``MIN_REUSE_SECONDS`` because anything
    shorter could only reach the grid through the clamp of repair (d), and a
    flagged, noted degradation is something to buy length with, never a bargain.
    """
    leftover = take.out_seconds - fit.out_seconds
    if target_seconds is None or leftover < MIN_REUSE_SECONDS:
        return
    queue.append(
        _Take(
            segment=take.segment,
            in_seconds=fit.out_seconds,
            out_seconds=take.out_seconds,
            take=take.take + 1,
        )
    )


def _record_unplaceable(take: _Take, unplaceable: _Unplaceable, notes: list[str]) -> int:
    """Note a dropped take and return what it adds to the unplaced count.

    Only a FIRST take counts. A re-take is footage the walk offered itself out of
    a segment already on screen, so dropping one is not a moment the reviewer
    approved and did not see — the length it fails to buy surfaces as the
    shortfall the handler already reports, and counting it here would inflate a
    structured number the reviewer reads as "your selection was ignored".
    """
    if take.take != 1:
        return 0
    notes.append(
        f"segment {_text(take.segment.get('id'))} was {_UNPLACED_NOTE_MARKER}: {unplaceable.reason}"
    )
    return 1


def _fit_segment(
    beats: list[float],
    downbeat_indices: set[int],
    index: int,
    *,
    in_seconds: float,
    out_seconds: float,
    score: float,
    section_change: bool,
    remaining: float | None,
    frame_rate: float,
) -> _Fit | _Unplaceable | None:
    """Place one segment starting at beat ``index``, or ``None`` if the grid ran out.

    The repair ladder of the module docstring, in order. Each rung is tried only
    because the one above it could not place the shot — a shot that fits at a
    phrase span in the invisible band never reaches the trim or the clamp.
    """
    source_len = max(0.0, out_seconds - in_seconds)
    if source_len <= 0 or index >= len(beats):
        return None

    # Where this shot begins on the video timeline. For the FIRST shot that is
    # the head of the video, not ``beats[0]``: bed t=0 == video t=0, so anchoring
    # shot 0 to the first beat would pull every cut in the reel `beats[0]` early.
    # The opening shot therefore covers the musical intro and ends on the first
    # chosen beat. For every later shot the origin is the beat it starts on,
    # which is by construction the previous shot's end.
    origin = _quantise(0.0 if index == 0 else beats[index], frame_rate)

    # Every span the grid can still offer from here, computed once: the repair
    # ladder is a series of narrowing filters over this one list. Windows are
    # measured FROM `origin`, so the ladder reasons about the length the renderer
    # will actually produce — including for the intro-covering first shot.
    free = _windows(beats, index, range(1, MAX_FREE_SPAN + 1), origin=origin, frame_rate=frame_rate)
    if not free:
        return None
    phrase = _phrase_windows(free, source_len)
    ranker = _Ranker(
        index=index,
        downbeat_indices=downbeat_indices,
        score=score,
        section_change=section_change,
        remaining=remaining,
        source_len=source_len,
    )

    in_point = _round_ms(in_seconds)

    def build(span: int, window: float, *, src_len: float, locked: bool, bias: int = 0) -> _Fit:
        # THE rule, in one place: `seconds` is the window, and the window came
        # out of `quantise(beats[index + span]) - origin` — a difference of two
        # absolute, independently-snapped boundaries. Nothing here sums.
        #
        # `speed` is DERIVED from the emitted trim rather than taken from the
        # caller, which is what makes `speed == (out - in) / seconds` true of the
        # emitted triple and not merely of the numbers the ladder was reasoning
        # about. The renderer reads a shot's length as `(out - in) / speed`, so
        # deriving it anywhere else — from an unrounded length, or worse from a
        # length the caller then clamped away from — silently re-lengths the shot.
        # `bias` floors (-1) / ceils (+1) the out-point instead of rounding to
        # nearest; repair (d) uses it so the millisecond of trim rounding cannot
        # push a just-clamped speed back outside the hard band.
        out_point = _round_ms(in_point + src_len, bias)
        return _Fit(
            span=span,
            beat_index=index,
            start_seconds=origin,
            seconds=window,
            in_seconds=in_point,
            out_seconds=out_point,
            speed=(out_point - in_point) / window,
            beat_locked=locked,
        )

    # (a) A phrase span that is already invisible. The common case, no repair.
    best = ranker.best(ranker.within(phrase, PREFERRED_SPEED_MIN, PREFERRED_SPEED_MAX))
    if best is not None:
        span, window = best
        return build(span, window, src_len=source_len, locked=True)

    # (b) Too long for every phrase span → trim the source inward. Segments run
    # 6-18s with real slack, so shortening one to exactly its window buys a 1.0x
    # playback, and no warp always beats some warp. The WIDEST window is chosen
    # because it discards the least footage. The kept part is the HEAD: without
    # an intra-segment motion curve here there is no better signal, and the head
    # is what segment_select's own rationale describes (it truncates from the
    # head for the same reason). The window is >= MIN_SHOT_SECONDS by
    # construction, so this can never trim below the minimum shot length.
    if phrase and ranker.all_too_fast(phrase):
        span, window = max(phrase, key=lambda candidate: candidate[1])
        # `speed` comes back as 1.0 to within the millisecond the trim is rounded
        # to (<0.2% at MIN_SHOT_SECONDS) rather than exactly 1.0. That residue is
        # deliberate: honouring the window is what keeps the cut on the beat, and
        # a 0.2% warp is invisible where a millisecond of over-run is not.
        return build(span, window, src_len=window, locked=True)

    # (c) Accept a legal warp — now over ANY beat count, not just the phrase
    # lengths. An off-phrase span is a worse musical fit (it pays 0.30) but a
    # much better picture than a 1.9x time-warp, so the two compete on cost.
    best = ranker.best(ranker.within(free, SPEED_MIN, SPEED_MAX))
    if best is not None:
        span, window = best
        return build(span, window, src_len=source_len, locked=True)

    # (d) Last resort. The clip cannot fill any window at a legal speed. It still
    # takes the window closest to its natural length — staying ON the grid keeps
    # the reel contiguous and keeps every LATER cut on the beat, which is worth
    # more than this one shot — and its speed is clamped into the hard band. The
    # caller stamps beat_locked=False and writes a note: this is a compromise the
    # reviewer must see, never a silent one.
    #
    # The clamp costs FOOTAGE, not TIME. Clamping the speed while keeping the
    # natural source length would break `speed == (out - in) / seconds` and the
    # shot would render `source_len / speed` seconds inside a `window`-second
    # slot — a 9s source in a 2s window clamped from 4.5x to 2.0x renders 4.5s
    # and puts every later cut 2.5s late, which is the one failure this whole
    # module exists to prevent. So the source is re-trimmed to what the clamped
    # speed can actually cover in `window`: the shot still ends on its beat, and
    # what is lost is the tail of a clip that was never going to fit.
    #
    # Which is why the choice of window is narrowed FIRST. A window wider than
    # `emitted_len / COMPILER_SPEED_MIN` cannot be filled by this segment at any
    # speed the compiler will accept — all of the footage, played as slowly as
    # the contract allows, still runs out before the beat does — so it is not a
    # candidate at all. Filtering here rather than clamping afterwards is the
    # whole point: `build` DERIVES the speed from the trim it emits, so a clamp
    # applied to `natural` and then discarded by the `min(...)` below never
    # reached the emitted shot. Shot 0 is where this bites — its origin is
    # quantised to 0.0, so its narrowest window is the whole musical intro
    # (`beats[1]`), and a 0.4s segment against a 2.0s intro emitted 0.2x.
    #
    # `emitted_len`, not `source_len`: the filter has to test the SAME number the
    # shot will carry, and `build` emits a millisecond-FLOORED out-point (bias=-1
    # below). The unrounded `source_len` is up to 1ms longer than what that floor
    # leaves, and a filter that admits a window on a millisecond the shot never
    # gets is not a filter — it is the same bug one rounding away. A 0.508334s
    # source against a 2.033333s window passed on 0.508334/0.25 = 2.033336 and
    # then emitted floor_ms → 0.508/2.033333 = 0.2498x, i.e. exactly the 422 this
    # filter exists to prevent. So it is derived through `_round_ms` at the same
    # in-point `build` uses, which is what stops the two from ever drifting again.
    # It is an upper bound on the trim for every candidate here: `build` is called
    # with `min(speed * window, source_len)`, so no window gets MORE footage than
    # this, and the shorter-trim branch is the one where `speed >= SPEED_MIN`
    # anyway and the floor is nowhere near.
    emitted_len = _round_ms(in_point + source_len, -1) - in_point
    placeable = [
        candidate for candidate in free if candidate[1] <= emitted_len / COMPILER_SPEED_MIN
    ]
    if not placeable:
        narrowest = min(window for _, window in free)
        return _Unplaceable(
            f"{emitted_len:.3f}s of footage cannot fill the narrowest {narrowest:.3f}s window "
            f"this grid offers here without falling under the compiler's "
            f"{COMPILER_SPEED_MIN:g}x floor"
        )
    span, window = min(placeable, key=lambda candidate: abs(candidate[1] - source_len))
    natural = source_len / window
    speed = max(SPEED_MIN, min(SPEED_MAX, natural))
    # ...and the re-trim may never reach PAST the segment the reviewer approved.
    # In the too-SHORT direction (`natural < SPEED_MIN`) `speed * window` is
    # LONGER than the segment: a 0.45s segment at 10.0-10.45 clamped up to 0.5x
    # asked for an out-point of 10.501, i.e. 51ms of footage nobody selected —
    # and at the end of a source, footage that does not exist, which the renderer
    # answers with a frozen or short tail. Nothing downstream catches it
    # (`compile._source_window` only checks `out > in >= 0`), so it is clamped
    # HERE, at the only place that knows what the segment offered.
    #
    # The cost is that the emitted speed then falls BELOW `SPEED_MIN` — `build`
    # derives it from the trim that is actually sent, so the triple still
    # satisfies `speed == (out - in) / seconds` and the shot still fills its beat
    # window exactly. A slower-than-editorial slow-motion is a visible compromise
    # on a shot already stamped `beat_locked=False` and noted; inventing footage
    # is a silent one. It bottoms out at `COMPILER_SPEED_MIN` and not lower —
    # but ONLY because the candidate filter above tests `emitted_len`, the
    # ms-floored trim this call is about to emit, and not the unrounded
    # `source_len`. Against `source_len` the guarantee was false by up to a
    # millisecond of trim, which is a fraction of a percent of speed and is on
    # the WRONG side of a hard bound; the filter and `build` must round the same
    # way or the fence is drawn around a number nobody ships. The bias stays a
    # FLOOR in both directions for the same reason: the millisecond of trim
    # rounding must move inward, never past the bound it was just clamped to.
    return build(span, window, src_len=min(speed * window, source_len), locked=False, bias=-1)


def _phrase_windows(free: list[tuple[int, float]], source_len: float) -> list[tuple[int, float]]:
    """The subset of ``free`` whose span is a musical phrase length. A single
    beat counts as one only for a genuinely short accent segment — at any normal
    tempo one beat is under a second, which is a flash, not a shot."""
    allowed = set(BEAT_SPANS)
    if source_len <= ACCENT_MAX_SOURCE_SECONDS:
        allowed.add(ACCENT_SPAN)
    return [(span, window) for span, window in free if span in allowed]


@dataclass(frozen=True)
class _Ranker:
    """Scores candidate ``(span, window)`` placements for ONE segment.

    Split out of ``_fit_segment`` so each rung of the repair ladder reads as a
    single line there: the ladder is the interesting part, the arithmetic that
    orders equally-legal candidates is not.
    """

    index: int
    downbeat_indices: set[int]
    score: float
    section_change: bool
    remaining: float | None
    source_len: float

    def within(
        self, candidates: list[tuple[int, float]], low: float, high: float
    ) -> list[tuple[int, float]]:
        """Candidates whose implied speed sits inside ``[low, high]``."""
        return [c for c in candidates if low <= self.source_len / c[1] <= high]

    def all_too_fast(self, candidates: list[tuple[int, float]]) -> bool:
        """True when EVERY candidate would have to be sped past the invisible
        band — the signature of a source that is simply too long, and the only
        case where trimming (repair (b)) is the right answer."""
        return all(self.source_len / window > PREFERRED_SPEED_MAX for _, window in candidates)

    def best(self, candidates: list[tuple[int, float]]) -> tuple[int, float] | None:
        return min(candidates, key=self._cost) if candidates else None

    def _cost(self, candidate: tuple[int, float]) -> float:
        """Lower is better. Symmetric in speed (1.25x is as wrong as 0.8x)."""
        span, window = candidate
        speed = self.source_len / window
        value = abs(math.log(speed))
        if not (PREFERRED_SPEED_MIN <= speed <= PREFERRED_SPEED_MAX):
            # Only bites in repair (c), where in-band and out-of-band candidates
            # compete: it is what makes a clean off-phrase span beat a 1.9x warp.
            value += 0.75
        if span not in BEAT_SPANS and span != ACCENT_SPAN:
            value += 0.30
        if self.section_change and (self.index + span) not in self.downbeat_indices:
            value += 0.12
        # Let the strongest moments breathe: a high-scoring segment pays less for
        # a long span, so given two equally invisible fits it takes the wider one.
        value -= 0.06 * max(0.0, min(1.0, self.score)) * min(span, 8) / 8
        if self.remaining is not None and window > self.remaining:
            value += 0.20
        return value


def _windows(
    beats: list[float],
    index: int,
    spans: Iterable[int],
    *,
    origin: float,
    frame_rate: float,
) -> list[tuple[int, float]]:
    """``(span, window)`` for every span that exists on the grid and is long
    enough to render. ``window`` is ALWAYS ``quantise(beats[index + span]) -
    origin`` — the difference of two absolute boundaries, each read from the
    measured grid, never ``span * 60 / bpm``. Real grids jitter by tens of
    milliseconds and the emitted cuts must track the music, not an idealisation
    of it; the quantisation is applied to each BOUNDARY for the same reason, so
    that the frame rounding cannot compound down the concat either.

    ``MIN_SHOT_SECONDS`` is checked against that final window and not against
    the raw beat interval, which is what makes both edge cases safe without a
    special path: a span whose quantised length would render as a flicker is
    never offered, and neither is one that a near-zero ``beats[0]`` would
    otherwise shrink below the minimum for the intro-covering first shot."""
    out: list[tuple[int, float]] = []
    for span in spans:
        end = index + span
        if end >= len(beats):
            break
        window = _quantise(beats[end], frame_rate) - origin
        if window < MIN_SHOT_SECONDS:
            continue
        out.append((span, window))
    return out


def _quantise(seconds: float, frame_rate: float) -> float:
    """Snap ONE absolute timeline position to the output frame grid.

    Deliberately takes a position and not a duration: quantising durations and
    then laying them end to end is the accumulating rounding this replaces.
    Every caller passes an absolute time, so the error of each boundary is at
    most half a frame and is independent of every other boundary."""
    return round(seconds * frame_rate) / frame_rate


def _round_ms(seconds: float, bias: int = 0) -> float:
    """Round a SOURCE time-point to milliseconds (``bias`` -1 floors, +1 ceils).

    Source in/out points are emitted at millisecond resolution — finer is noise
    against a decoder's seek. The direction is a parameter because the rounding
    is not cosmetic: ``speed`` is derived from the rounded pair, so a caller that
    has just clamped the speed to a hard bound needs the rounding to move the
    trim INWARD from that bound rather than across it.
    """
    scaled = seconds * 1000.0
    if bias < 0:
        return math.floor(scaled) / 1000.0
    if bias > 0:
        return math.ceil(scaled) / 1000.0
    return round(scaled) / 1000.0


def _validate_grid(beats: list[float]) -> None:
    """Reject a grid that is not strictly increasing, before anything indexes it.

    ``bisect_left`` below — and every ``beats[i + k] - beats[i]`` in the solver —
    silently assumes sorted input: an unsorted or duplicated grid does not raise,
    it yields zero or negative windows and a plan whose cuts are out of order.
    The grid comes from an analyzer upstream, so a malformed one is an input
    defect and belongs in the run's precondition channel, naming the index that
    broke it, rather than in an unrelated traceback fifty lines later. One pass,
    no allocation — cheap enough to run on every solve."""
    previous = beats[0]
    if not math.isfinite(previous) or previous < 0:
        raise StagePreconditionError(
            f"cut_plan: beat grid starts at {previous!r}; beat times must be finite and >= 0",
            missing=["music_select.beats"],
        )
    for offset, moment in enumerate(beats[1:], start=1):
        if not math.isfinite(moment) or moment <= previous:
            raise StagePreconditionError(
                f"cut_plan: beat grid is not strictly increasing at index {offset} "
                f"({previous!r} -> {moment!r}); music_select must emit sorted beat times",
                missing=["music_select.beats"],
            )
        previous = moment


def _downbeat_indices(beats: list[float], downbeats: list[float]) -> set[int]:
    """Beat indices that are also downbeats, matched by TIME rather than by
    assuming a 4/4 ``index % 4`` — the analyzer reports the meter it heard, and
    a bar-one that lands on beat 3 of our list is exactly the case where an
    assumed modulus would put every section change in the wrong place."""
    found: set[int] = set()
    for moment in downbeats:
        pivot = bisect_left(beats, moment)
        for candidate in (pivot, pivot - 1):
            if (
                0 <= candidate < len(beats)
                and abs(beats[candidate] - moment) <= _DOWNBEAT_TOLERANCE
            ):
                found.add(candidate)
                break
    return found


def _even_spacing(
    segments: list[dict[str, object]], target_seconds: float | None = None
) -> CutPlanSolution:
    """Degradation for a missing/empty grid: every segment at its natural length.

    Never a failure. music_select soft-fails its analyzer call by design, and
    losing the grid must cost the reel its beat-sync, not the whole run. Speeds
    are 1.0 (no warp applied — there is nothing to warp TO), ``beat_index`` is
    ``-1``, and ``beat_locked`` is False on every shot so nothing downstream
    mistakes an evenly-spaced cut for a solved one.

    Losing the grid costs the BEAT-SYNC and nothing else: the requested LENGTH is
    still honoured, on the same stopping rule the solved walk uses (emit, then
    stop once the cursor has reached the target). Without that this path ships
    every selected segment, and ``segment_select`` over-selects by design
    (``_COVERAGE_FACTOR = 2.0``) so the solver has choices — which turned a 20s
    request into a 40s reel while ``shortfall_seconds`` reported 0.0, because a
    shortfall gate is one-sided and an over-run is not a shortfall.
    """
    shots: list[dict[str, object]] = []
    # This is the ONE path that walks a clock rather than a grid — there is no
    # grid to read times out of. It is self-consistent anyway: the clock advances
    # by the EMITTED length of each shot, so the positions are the true prefix
    # sums of what the renderer will actually produce, with nothing hiding in
    # between. There is no music to drift against here in any case.
    cursor = 0.0
    # The same count the solved walk keeps, for the same reason and told apart in
    # the same way. This path has no grid, so it can never "run out of grid" —
    # every segment it does not emit was left behind because the requested LENGTH
    # was already met, which is the stopping rule succeeding. Reporting that as
    # `unplaced_segments` would be a failure badge on a success; reporting it as
    # nothing at all is what this path did, and it is the identical silent
    # truncation the solved walk had: a reviewer who has learned to read
    # `sources_absent` would read this path's empty list as "every clip is in
    # there" precisely when the grid is missing and the plan is already degraded.
    unused_at_target = 0
    for position, segment in enumerate(segments):
        # The trim is rounded to milliseconds FIRST and the length taken from the
        # rounded pair, not the other way round: at 1.0x the renderer's
        # `(out - in) / speed` IS the shot length, so a `seconds` derived from the
        # unrounded pair would disagree with what actually renders by up to a
        # millisecond per shot — and here, uniquely, the positions are prefix sums.
        start = _round_ms(_as_float(segment.get("in_seconds")))
        end = _round_ms(_as_float(segment.get("out_seconds")))
        seconds = max(0.0, end - start)
        shots.append(
            {
                "id": _text(segment.get("id")),
                "source_id": _text(segment.get("source_id")),
                "in_seconds": start,
                "out_seconds": end,
                "seconds": seconds,
                "speed": STUB_SPEED,
                "beat_index": -1,
                "start_seconds": cursor,
                "beat_locked": False,
                "rationale": _text(segment.get("rationale")),
            }
        )
        cursor += seconds
        # Same rule as the solved walk (`elapsed >= target_seconds`): the shot
        # that crosses the target is kept, and nothing after it is emitted. A
        # reel is allowed to run a shot long; it is not allowed to run double.
        if target_seconds is not None and cursor >= target_seconds:
            # Every segment behind this one is unconsumed. There is no re-take
            # queue on this path — one shot per segment, in order — so the
            # remainder of the list IS the first-take remainder, with nothing to
            # filter out.
            unused_at_target = len(segments) - position - 1
            break
    # `unplaced_segments` stays 0 here: there is no grid to exhaust, so nothing
    # was dropped for want of one. The segments this walk stopped short of are
    # reported as `segments_unused_at_target` and, where a whole source went
    # missing with them, named in the note below.
    notes = [
        "no beat grid available; shots are spaced evenly at their natural length "
        "at 1.0x and are NOT cut to the music"
    ]
    sources_absent = _absent_sources(segments, shots)
    # Same gate as the solved walk, and on this path it can only ever come out
    # one way: absence here REQUIRES the target break (a walk that fell short has
    # emitted every segment it had), so the note never fires and the fields are
    # the whole report. Called anyway rather than short-circuited, so the two
    # paths cannot drift apart if that reasoning ever stops holding.
    reached_target = target_seconds is not None and cursor >= target_seconds
    notes.extend(
        _absent_sources_notes(
            sources_absent, segments, target_seconds, reached_target=reached_target
        )
    )
    return CutPlanSolution(shots, notes, 0, unused_at_target, sources_absent)


def _overrun_notes(total_seconds: float, target: float, overrun: float) -> list[str]:
    """The over-run note, or nothing — the mirror of the TRUNCATED note.

    Only a RUNAWAY over-run is said out loud (see ``OVERRUN_NOTE_FRACTION``): both
    walks stop by emitting the shot that CROSSES the target, so a small overshoot
    is the stopping rule working and noting it would badge every length-targeted
    reel ``degraded``. What the rule does not bound is how LONG that crossing shot
    is, and ``overrun_seconds`` is read by no consumer — so a reel at twice the
    requested length reached the reviewer as a green gate with nothing on it.
    """
    if overrun <= target * OVERRUN_NOTE_FRACTION:
        return []
    return [
        f"OVERRUN: the reel is {total_seconds:.1f}s but {target:.1f}s was requested "
        f"({overrun:.1f}s long) — more than the stopping rule accounts for; trim the "
        "selection or raise the target"
    ]


def _text(value: object) -> str:
    return str(value or "")


def _as_float(value: object) -> float:
    try:
        return float(value)  # ty: ignore[invalid-argument-type]
    except TypeError, ValueError:
        return 0.0


def _empty_plan_error(
    *,
    beats: list[float],
    playable: int,
    unplaced_segments: int,
    solver_notes: list[str],
    grid_exhausted: bool = False,
) -> StagePreconditionError:
    """The precondition for an empty plan — TWO causes, told apart.

    Returning ``shots=[]`` "successfully" is not an option either way: it only
    moves the failure four stages down into compile, where it surfaces as an
    unrelated message about an empty render. But blaming the wrong cause is
    barely better, because it sends the next person to fix something that was
    never broken:

    (1) The GRID ran out — every beat window from index 0 is shorter than
        ``MIN_SHOT_SECONDS``, or the track is shorter than one shot.
    (2) Every segment was UNPLACEABLE — the grid is perfectly fine (long
        windows), the footage is simply too short to fill one of them without
        emitting a speed under the compiler's floor. ``unplaced_segments``
        is what tells the two apart.

    ⚠️ ``unplaced_segments`` does NOT tell them apart, which is why
    ``grid_exhausted`` is tested FIRST. The solver's walk guard is
    ``cursor < len(queue)``, so on the grid-exhausted break there is always at
    least one pending take and ``unplaced_segments`` is always >= 1 — cause (1)
    could therefore never be reported, and a MUSIC BED problem was handed to the
    reviewer as a footage problem with ``missing=["segment_select.segments"]``.
    Anything gating on ``missing=["music_select.beats"]`` never saw it fire.

    And the reasons ride along. ``solver_notes`` already carries one line per
    dropped segment naming WHICH approved moment went and WHY; raising without
    them threw the whole diagnosis away and left the reviewer of a run that had
    passed five human gates with nothing but a beat count. Bounded to the first
    few — a 200-segment selection must not paste 200 notes into an error message
    — with the total kept so nobody reads the sample as the whole.
    """
    if grid_exhausted:
        return StagePreconditionError(
            f"cut_plan: the beat grid ({len(beats)} beats) could not hold a single shot "
            f"of at least {MIN_SHOT_SECONDS}s from the {playable} selected segments; "
            "the plan is empty and there is nothing to compile",
            missing=["music_select.beats"],
        )
    if unplaced_segments:
        dropped = [note for note in solver_notes if _UNPLACED_NOTE_MARKER in note]
        sample = "; ".join(dropped[:3])
        if len(dropped) > 3:
            sample += f"; (+{len(dropped) - 3} more)"
        return StagePreconditionError(
            f"cut_plan: none of the {playable} selected segments could be placed on the beat "
            f"grid ({len(beats)} beats) without emitting a speed under the compiler's "
            f"{COMPILER_SPEED_MIN:g}x floor; {unplaced_segments} segments were dropped and the "
            f"plan is empty. Reasons: {sample or 'not recorded'}",
            missing=["segment_select.segments"],
        )
    return StagePreconditionError(
        f"cut_plan: the beat grid ({len(beats)} beats) could not hold a single shot "
        f"of at least {MIN_SHOT_SECONDS}s from the {playable} selected segments; "
        "the plan is empty and there is nothing to compile",
        missing=["music_select.beats"],
    )


class CutPlanStage(StageHandler[CutPlanInput, CutPlanOutput]):
    name = "cut_plan"
    input_model = CutPlanInput
    output_model = CutPlanOutput

    async def handle(self, payload: CutPlanInput, ctx: RunContext) -> CutPlanOutput:
        selection = ctx.prior_output("segment_select") or {}
        segments = cast("list[dict[str, object]]", selection.get("segments") or [])
        if not segments:
            raise StagePreconditionError(
                "cut_plan: segment_select produced no segments",
                missing=["segment_select.segments"],
            )
        music = ctx.prior_output("music_select") or {}
        beats = [_as_float(b) for b in (music.get("beats") or [])]
        downbeats = [_as_float(b) for b in (music.get("downbeats") or [])]

        # Source URLs live on footage_probe (segment_select carries only ids, so
        # the plan has to re-join to get something the compositor can fetch).
        probe = ctx.prior_output("footage_probe") or {}
        url_by_source = await _fetchable_source_urls(probe)

        notes: list[str] = []
        # BEFORE the solver, not after: a segment whose source cannot be fetched
        # is dropped by compile, and a shot that the grid allocated a window to
        # but the render omits leaves a hole the music plays straight through —
        # every later cut lands late by exactly that shot's `seconds`. Filtering
        # first means the grid is allocated only to shots that will exist.
        playable: list[dict[str, object]] = []
        excluded: list[str] = []
        for segment in segments:
            source_id = _text(segment.get("source_id"))
            if url_by_source.get(source_id):
                playable.append(segment)
            else:
                excluded.append(f"segment {_text(segment.get('id'))} (source {source_id})")
        if excluded:
            # Named, never a silent shrink: these are shots a human approved and
            # the reel will not contain them. The note carries WHICH; the count
            # rides out as `unfetchable_segments` so a gate can read HOW MANY
            # without parsing this sentence.
            notes.append(
                "excluded from the cut before solving (no fetchable source url): "
                + ", ".join(excluded)
            )
        if not playable:
            raise StagePreconditionError(
                f"cut_plan: none of the {len(segments)} selected segments has a fetchable "
                "source url; footage_probe presigned no usable original for any of them",
                missing=["footage_probe.sources[].original_url"],
            )

        # One frame grid for the plan and the render. `compile` asks the same
        # function for the same answer; planning on 23.976 and rendering on 30 is
        # the renderer re-rounding every boundary we just snapped (see cadence).
        frame_rate = float(resolve_master_fps(ctx.config_snapshot, probe))
        target = target_seconds(ctx)

        solution = solve_cut_plan(
            playable,
            beats,
            downbeats=downbeats,
            target_seconds=target,
            frame_rate=frame_rate,
        )
        planned = solution.shots
        solver_notes = solution.notes
        unplaced_segments = solution.unplaced_segments
        notes.extend(solver_notes)
        if not planned:
            raise _empty_plan_error(
                beats=beats,
                playable=len(playable),
                unplaced_segments=unplaced_segments,
                solver_notes=solver_notes,
                grid_exhausted=solution.grid_exhausted,
            )

        shots: list[PlannedShot] = []
        for shot in planned:
            source_id = _text(shot.get("source_id"))
            video_url = url_by_source.get(source_id, "")
            shots.append(
                PlannedShot(
                    id=_text(shot.get("id")),
                    source_id=source_id,
                    video_url=video_url,
                    in_seconds=_as_float(shot.get("in_seconds")),
                    out_seconds=_as_float(shot.get("out_seconds")),
                    seconds=_as_float(shot.get("seconds")),
                    speed=_as_float(shot.get("speed")),
                    beat_index=int(_as_float(shot.get("beat_index"))),
                    start_seconds=_as_float(shot.get("start_seconds")),
                    beat_locked=bool(shot.get("beat_locked")),
                    rationale=_text(shot.get("rationale")),
                )
            )

        # Derived from the two OUTER boundaries, not summed over the shots: the
        # reel's length is "where the last shot ends minus where the first one
        # starts", one subtraction of two absolute times. Summing here would
        # re-introduce, at the very last line of the stage, exactly the
        # accumulation the solver above is built to avoid — and the shots are
        # contiguous by assertion, so the two are the same number anyway when
        # nothing is wrong, which is precisely why the summed form hid errors.
        total_seconds = (shots[-1].start_seconds + shots[-1].seconds) - shots[0].start_seconds

        # The truncation gate. Nothing above compares the requested length to
        # what the grid could actually hold, so a 90s request against a real 58s
        # bed (the catalog's 120s claim is wrong) came out as a 58s reel with a
        # note buried among the others. It is surfaced as a NUMBER as well: this
        # stage pauses for review, so the reviewer either accepts the shorter cut
        # or re-picks the bed — which is why this is a loud degradation and not a
        # hard failure that would throw away an otherwise good reel.
        shortfall = 0.0
        # The other side of the same question, as a number only (see the field):
        # "did this plan deliver the length that was asked for" cannot be
        # answered by a one-sided gate.
        overrun = 0.0
        if target is not None and total_seconds > target + TRUNCATION_TOLERANCE_SECONDS:
            overrun = round(total_seconds - target, 3)
            # `extend` a 0-or-1 list rather than an `if` around an `append`: the
            # decision lives in `_overrun_notes` (which is where its reasoning is
            # written anyway) and `handle` is already at the C901 ceiling.
            notes.extend(_overrun_notes(total_seconds, target, overrun))
        if target is not None and total_seconds < target - TRUNCATION_TOLERANCE_SECONDS:
            shortfall = round(target - total_seconds, 3)
            notes.append(
                f"TRUNCATED: the reel is {total_seconds:.1f}s but {target:.1f}s was requested "
                f"({shortfall:.1f}s short) — the music bed or the selected footage ran out "
                "first; pick a longer bed or select more segments"
            )

        return CutPlanOutput(
            shots=shots,
            total_seconds=total_seconds,
            music_track_id=_text(music.get("track_id")),
            music_url=_text(music.get("track_url")),
            bpm=_as_float(music.get("bpm")),
            beat_grid_available=bool(beats),
            # Computed, never asserted: without a grid the plan is the
            # `_even_spacing` degradation, which is precisely what "stubbed"
            # means. Hard-coding False here told every consumer — including the
            # e2e proof harness — that an unsolved plan had been solved.
            solver_stubbed=not beats,
            frame_rate=frame_rate,
            target_seconds=target or 0.0,
            shortfall_seconds=shortfall,
            overrun_seconds=overrun,
            # Straight from the solver, never re-derived from the note text: the
            # note is a rendering of this number, not its source.
            unplaced_segments=unplaced_segments,
            # Straight from the same list the note above was rendered from.
            unfetchable_segments=len(excluded),
            # Both straight from the solver, for the same reason
            # `unplaced_segments` is: the solver already emitted the note, and a
            # field re-derived from note text is a field that will one day
            # disagree with it.
            segments_unused_at_target=solution.segments_unused_at_target,
            sources_absent=list(solution.sources_absent),
            sources_absent_count=len(solution.sources_absent),
            notes=notes,
        )


async def _fetchable_source_urls(probe: dict[str, object]) -> dict[str, str]:
    """``source id -> a URL that is still alive NOW``, re-signed at plan time.

    ``footage_probe.original_url`` is presigned with a 1 h TTL, and cut_plan runs
    after TWO human-paced pause gates (``shot_analysis``, ``segment_select``). An
    expired presigned URL is still a perfectly non-empty string, so it sails past
    every ``if not video_url`` guard in this stage and in compile, and fails deep
    inside the compositor's fetch with a message about the render rather than
    about a signature. Re-minting here means the string this stage emits is one
    the reviewer can click and one compile's own re-sign pass starts from.

    Reuses ``shot_presign.refresh_shot_asset_urls`` — the fleet's storage seam for
    exactly this — rather than presigning from ``object_key``: ``ProbedSource``
    does not carry the key (only the already-signed URL), and adding it there is
    another stage's change. That seam re-derives the key from the URL and is
    best-effort: a non-owned or unparseable URL keeps its original value, so the
    worst case is today's behaviour rather than a new failure mode.
    """
    rows: list[dict[str, str]] = [
        {"id": _text(source.get("id")), "video_url": _text(source.get("original_url"))}
        for source in cast("list[dict[str, object]]", probe.get("sources") or [])
    ]
    await refresh_shot_asset_urls(rows)
    return {
        str(row["id"]): str(row["video_url"])
        for row in rows
        if row["id"] and isinstance(row["video_url"], str) and row["video_url"]
    }
