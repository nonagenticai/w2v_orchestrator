"""graphics_design — the animated graphics laid over the cut.

This is the "animated graphics over the cut" half of the product promise. It
designs the reel's overlay layer and RENDERS it: today, one branded title
treatment on the opening shot, as a real alpha-channel clip that the compiler
composites ON TOP of the footage (``Shot.overlay_clips``) rather than cutting
to full-frame.

**Why an overlay and not a card.** A full-frame animated card is an ordinary
shot with a ``video_url`` and needs nothing new — but it also stops the reel to
show it. The product sells continuous footage, so the title has to fly in OVER
the opening shot. That requires a clip with a genuine alpha channel: VP9 in
WebM, requested via ``record_animation(transparent=True)``. ``lower_third`` is
the ONLY template the recorder will render transparently (every other library
template paints its own opaque full-bleed stage and the route 400s the
request), which is why the template id below is not a config knob.

⚠️ **Alpha does not ride in the pixel format.** A correct VP9-alpha WebM
reports ``pix_fmt=yuv420p`` and carries the WebM container tag
``alpha_mode=1``. Anything that "validates alpha" by looking for ``yuva420p``
rejects every correct build. Do not add such an assertion here or in a test.

**Two card kinds today.** The deterministic title treatment on the opening shot,
and up to ``_MAX_CALLOUTS`` per-shot CALLOUTS written by a model off each shot's
``segment_select`` rationale and its cut telemetry. An end card and beat-aligned
entrance timing are still the follow-up (see the TODO below).

**The callout designer is advisory; this module is the enforcement.** Every
editorial rule the prompt states — the cap, one card per shot, never shot 0,
never a shot too short to hold a whole card — is re-applied in
``_usable_callouts`` over the model's draw, because a model asked for three
returns four. That is not tidiness: the compiler caps ``Shot.overlay_clips`` at
two with a ``max_length``, so an unenforced rule does not cost a card, it 422s
the WHOLE render and the reel is lost. Nothing is repaired, only dropped — a
shot index the model invented is not evidence about which shot it meant — and
every drop is COUNTED into the output, because one callout on the reel reads
very differently depending on whether the designer wrote one or wrote six.

**A designer outage never costs the title.** The WM call is made before the
title render and its every failure resolves to "no callouts, ``callouts_degraded
= True``"; the title treatment is deterministic and ships regardless. The four
callout counters are read as a SET for the same reason the two flags below are:
a bare ``callouts_designed == 0`` is produced by a designer that chose none, a
cut with no legal shot to place one on, a draw where everything was unusable,
and a designer that never answered.

**Failure is loud, in the output.** The renderer SILENTLY DROPS an overlay it
cannot fetch (``overlay_clip.dropped``) and renders on, so a bad URL ships a
reel with no graphics, no error, and a green run. This stage therefore never
emits a card without a stored URL: a render failure degrades to an EMPTY card
list plus a note that says in plain words that the reel will carry no graphics.
Because that degradation is indistinguishable from a legitimate "this brief has
no title to render" by looking at ``cards`` alone — both are ``[]`` plus a note —
the failure sets ``cards_render_failed=True``. That flag is the ONLY
machine-readable way to tell a render outage from an untitled brief; a consumer
that wants to retry, alert on, or count outages reads it and never tries to
pattern-match the note prose. The mirror case — an empty list that NO designer
ever looked at (no shots, or no title and no brand) — sets ``cards_stubbed``, so
the three readings of ``cards == []`` stay tellable apart by a consumer.

**This stage pauses** because graphics are the most visible layer of the reel
and the cheapest to reject before a render.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, cast
from uuid import uuid7

from src.clients.video_production_client import VideoProductionClient
from src.services.llm import WMClient
from src.services.stages.action_reel.cadence import resolve_master_fps
from src.services.stages.base import SlotUse, StageHandler, StageInput, StageOutput
from src.services.stages.llm_budget import (
    MEDIUM_ANSWER_TOKENS,
    llm_client_timeout_seconds,
    output_token_budget,
)
from src.services.stages.shared.motion_slides import (
    format_dims,
    motion_brand_params,
    quality_tier,
)
from src.services.tts_storage import store_generated_video

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

log = logging.getLogger(__name__)

#: The only overlay-safe template in video_production_service. Not configurable:
#: asking any other template for alpha is answered with a 400.
_TEMPLATE = "lower_third"
#: WebM/VP9 — the container that can carry the alpha plane. mp4 is REJECTED by
#: the recorder for a transparent render rather than silently flattened.
_CARD_FORMAT = "webm"
#: Recorder-poll budget. It MUST stay strictly under the blueprint's
#: ``graphics_design`` stage ceiling — and, now that up to four cards are
#: recorded SEQUENTIALLY, under `GRAPHICS_DESIGN_TIMEOUT_SECONDS` divided by
#: `1 + _MAX_CALLOUTS` plus the LLM leg. 240 s x 4 = 960 s is what that ceiling
#: is built from; see its own arithmetic below.
#:
#: This is not a taste choice. The stage's ``TimeoutMiddleware`` is
#: ``asyncio.wait_for``, which cancels by injecting ``CancelledError`` — a
#: BaseException, so the ``except Exception`` around the render below does NOT
#: catch it. With the inner budget ABOVE the outer ceiling (it was 600 s, copied
#: from ``shared/title_cards`` which runs under a 2500 s compile stage) the outer
#: always won, which made the whole degrade path — empty card list, plain-words
#: note, ``cards_render_failed=True`` — UNREACHABLE on the single likeliest
#: outage: a slow recorder. The stage hard-failed the run instead, flatly
#: contradicting the blueprint's own "never a StageSkippedError… degrades" note.
#:
#: The headroom is what the rest of the stage still has to do after this call
#: returns: fetch the rendered WebM and store it (both under the same ceiling),
#: plus building the degraded output itself. 20% of the ceiling is the same slack
#: convention the LLM timeout ladder uses. Raising this above the ceiling
#: re-breaks the degrade path silently; the ladder test in
#: tests/services/stages/action_reel/test_graphics_design.py pins the relation.
_RECORD_TIMEOUT_S = 240.0
#: On-screen length of the title treatment. Long enough to read a title and a
#: brand line at reel pace, short enough that it is gone before the first cut on
#: a typical 3-6s opening shot.
_CARD_SECONDS = 3.0
#: Floor for the rendered clip. The recorder rejects `duration_ms` under 100,
#: and a card shorter than this cannot complete an entrance at all.
_MIN_CARD_MS = 800
#: Entrance / exit as a FRACTION of the card, not fixed milliseconds: a card
#: clamped to a short opening shot must still fly in and out rather than have a
#: fixed 600ms entrance eat most of its life.
_ENTRANCE_FRACTION = 0.2
_EXIT_FRACTION = 0.17


def _text_digest(headline: str, strap: str) -> str:
    """Short, stable digest of the exact strings a card will render.

    Part of the recorder idempotency key (see ``_render_card``). NUL-joined
    rather than concatenated so ``("ab", "c")`` and ``("a", "bc")`` cannot
    collide onto one cached recording. blake2b at 8 bytes: this is a cache key,
    not a security boundary, and 16 hex characters keep the key readable in a
    log line.
    """
    payload = f"{headline}\x00{strap}".encode()
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


# --- the callout designer ----------------------------------------------------
#
# The second half of the overlay layer: per-shot callouts written by a model off
# each shot's own `rationale` (the sentence a human approved at segment_select)
# and its timeline telemetry. The title card above is deterministic and cannot
# fail for editorial reasons; these can, so every constraint below is enforced
# HERE, in code, and never only in the prompt. A model asked for "at most three"
# returns four often enough that the compiler's own overlay cap would 422 the
# whole render — one over-enthusiastic draw must cost a missing card, never the
# reel.

#: Hard cap on callout cards in the WHOLE reel. A reel is 20-60 s of continuous
#: footage; a fourth caption turns an edit into a slideshow, and every card is a
#: paid transparent render on top of that.
_MAX_CALLOUTS = 3
#: Cards per shot, this stage's own limit. The compiler caps `Shot.overlay_clips`
#: at TWO (`video_compiler_service/src/api/schemas.py::_MAX_OVERLAY_CLIPS`, a
#: `max_length`, so a third 422s the entire render) and `compile._overlay_clip`
#: mirrors that as a truncation. One is stricter on purpose: two captions stacked
#: on one shot is not a design, and staying a card clear of the compiler's bound
#: means a future end card cannot push a shot over it.
_MAX_CARDS_PER_SHOT = 1
#: Shot 0 belongs to the title treatment. A callout there would be the second
#: overlay on the one shot that already carries one, and it would fight the title
#: for the same corner.
_FIRST_CALLOUT_SHOT_INDEX = 1
#: How many shots are DESCRIBED to the model. The prompt must not grow with the
#: cut: a 40-shot reel would otherwise send 40 rationales and spend the whole
#: token budget on input. Over the cap the shots are RESAMPLED end to end
#: (`_even_indices`) rather than head-truncated, so the model still sees the end
#: of the reel — the same idiom, and the same reasoning, as
#: `segment_select._downsample_motion`.
_MAX_SHOTS_DESCRIBED = 16
#: Per-shot rationale characters sent to the model. The sentence is a one-line
#: justification; anything past this is a model that ran on.
_MAX_RATIONALE_CHARS = 160
#: On-screen text limits, enforced on the way OUT. `lower_third` is a strap, not
#: a paragraph: a 200-character "headline" overflows the safe area and renders as
#: clipped text over somebody's footage. Trimmed rather than dropped — a long
#: headline is still a headline.
_MAX_HEADLINE_CHARS = 32
_MAX_STRAP_CHARS = 56

#: Outer `TimeoutMiddleware` ceiling for this stage, pinned by the blueprint.
#:
#: Arithmetic, because the number is a ladder rung and not a preference:
#:   LLM      930 s — `llm_client_timeout_seconds()`, ONE attempt. The designer
#:            calls `WMClient.message` DIRECTLY: there is no `stage_call`, no
#:            `guarded`, no in-stage retry, so the "x2 attempts" an earlier
#:            revision of this comment claimed cannot happen. A second execution
#:            is `max_stage_attempts`, which is a fresh run of the whole stage
#:            OUTSIDE this ceiling and must not be paid for twice inside it.
#:   renders  (1 title + `_MAX_CALLOUTS`) x `_RECORD_TIMEOUT_S` = 960 s, SEQUENTIAL
#:            (each card is its own paid recording; they are not raced).
#:   headroom 110 s to fetch, store and build the output under the ceiling.
#: 930 + 960 + 110 = 2000.
#:
#: FLOOR, separately: `tests/services/blueprints/test_llm_timeout_ladder.py`
#: requires `outer >= llm_budget_seconds(950) x 2 + 30 = 1930` for every stage
#: holding a WM client at the shared rung. 2000 clears it with 70 s to spare, so
#: both bounds hold and neither is silently doing the other's job.
#:
#: Sized DOWN from a first pass at 3000, which took the ladder floor as the LLM
#: term and then added the renders on top — double-counting ~1000 s. Ceilings are
#: not free: the action_reel scenario carries a 3600 s end-to-end deadline
#: (`scripts/e2e_scenarios/matrix.json`), and the blueprint's ceilings already
#: sum far past it, so every second granted here is a second the run's own
#: deadline has to find.
#:
#: ⚠️ This ceiling was 300 s while the stage was title-only, which INVERTED the
#: ladder the moment a WM call joined the stage. Lowering it back
#: without removing the WM call inverts the ladder: `TimeoutMiddleware` is
#: `asyncio.wait_for`, which cancels with a BaseException that the degrade paths'
#: `except Exception` cannot catch, so the callout and render degrades become
#: unreachable and a slow model HARD-FAILS a paid, human-reviewed stage.
GRAPHICS_DESIGN_TIMEOUT_SECONDS = 2000

#: See `output_token_budget`: omitting `max_tokens` leaves WM's 6000 default,
#: which a reasoning model burns before emitting any JSON — and that 422 is
#: TERMINAL. `tests/test_wm_max_tokens_gate.py` enforces the floor.
#: MEDIUM: at most three small objects with two short prose fields each.
_CALLOUT_MAX_TOKENS = output_token_budget(answer_tokens=MEDIUM_ANSWER_TOKENS)

_CALLOUT_SYSTEM_PROMPT = (
    "You are the graphics designer on an action-sports reel. You receive the shot "
    "list of a cut that is already locked: for each shot its index, how long it is "
    "on screen, where it starts in the reel, its speed, and the one-sentence "
    "rationale the editor wrote for keeping it. Write short on-screen callouts for "
    "the few shots that genuinely earn one — a trick landed, a location revealed, a "
    "speed worth naming. Each callout is a headline of at most a handful of words "
    "and a shorter strap under it, both written for a viewer who has no sound on. "
    "Return at most three callouts for the whole reel, at most one per shot, and "
    "never one for shot 0 (the title treatment already owns that shot). Only "
    "reference a shot index you were actually given. Fewer, stronger callouts beat "
    "a caption on every shot: a reel narrated by captions is a slideshow. If no "
    "shot earns one, return an empty list."
)

# Object-rooted structured-output schema (WM convention).
#
# ⚠️ NO `minItems`/`maxItems`/`minLength`/`maximum` anywhere, and this is not
# style. worker-management STRIPS every such keyword before building the vLLM
# grammar, so the bound constrains generation NOWHERE and advertises a guarantee
# nothing applies — `tests/test_output_schema_bounds.py` bans the whole family and
# explains the outage that produced the ban. The limits live in the descriptions
# below (where the model reads them) and are ENFORCED in `_usable_callouts`.
_CALLOUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "cards": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "shot_index": {
                        "type": "integer",
                        "description": (
                            "Index of the shot this callout sits on, exactly as given in "
                            "the shot list. Never 0 - the title treatment owns that shot."
                        ),
                    },
                    "headline": {
                        "type": "string",
                        "description": (
                            f"The on-screen headline, at most {_MAX_HEADLINE_CHARS} "
                            "characters. A few words, not a sentence."
                        ),
                    },
                    "strap": {
                        "type": "string",
                        "description": (
                            f"The smaller line under the headline, at most {_MAX_STRAP_CHARS} "
                            "characters. May be empty."
                        ),
                    },
                },
                "required": ["shot_index", "headline", "strap"],
            },
        }
    },
    "required": ["cards"],
}


@dataclass(frozen=True)
class _Callout:
    """One callout that survived every editorial constraint, ready to render."""

    shot_index: int
    headline: str
    strap: str
    on_screen_seconds: float


def _even_indices(count: int, cap: int) -> list[int]:
    """``cap`` indices spanning ``range(count)`` end to end, first and LAST included.

    The same helper, and the same reasoning, as
    ``segment_select._even_indices``: a stride of ``count // cap`` is 1 for every
    ``count`` in ``[cap + 1, 2 * cap - 1]``, so striding silently head-truncates
    exactly where a long reel needs resampling — and the END of a reel (the
    landing, the reveal) is the half that would be thrown away.
    """
    if cap <= 0 or count <= 0:
        return []
    if count <= cap:
        return list(range(count))
    if cap == 1:
        return [count - 1]
    return [round(i * (count - 1) / (cap - 1)) for i in range(cap)]


def _trim(value: object, limit: int) -> str:
    """*value* as collapsed single-line text, cut to *limit* characters."""
    return " ".join(str(value or "").split())[:limit].rstrip()


def _describe_shots(shots: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """The bounded shot list handed to the model, oldest editorial fact first.

    Bounded on BOTH axes — how many shots are described (`_MAX_SHOTS_DESCRIBED`,
    resampled) and how much of each rationale (`_MAX_RATIONALE_CHARS`) — because
    ``output_token_budget`` bounds the answer only. An unbounded prompt makes the
    model think longer, which spends that budget faster against the same deadline.

    Shot 0 is described too even though it can never carry a callout: it is the
    opening of the reel and the model is choosing which MOMENTS earn a caption,
    which it cannot judge from a list the beginning has been cut out of.
    """
    described: list[dict[str, object]] = []
    for i in _even_indices(len(shots), _MAX_SHOTS_DESCRIBED):
        shot = shots[i]
        described.append(
            {
                "shot_index": i,
                "seconds": round(_positive(shot.get("seconds"), 0.0), 2),
                "starts_at_seconds": round(_positive(shot.get("start_seconds"), 0.0), 2),
                "speed": round(_positive(shot.get("speed"), 1.0), 2),
                "rationale": _trim(shot.get("rationale"), _MAX_RATIONALE_CHARS),
            }
        )
    return described


def _usable_callouts(
    proposals: object, shots: Sequence[Mapping[str, object]]
) -> tuple[list[_Callout], int]:
    """``(callouts to render, how many proposals were DROPPED)``.

    Every editorial constraint the prompt states is re-applied here, because a
    prompt is a request and this is the enforcement. The drop count is returned
    rather than logged so the stage can report it: a reviewer looking at one
    callout must be able to tell "the designer wrote one" from "the designer
    wrote six and five were unusable", which the card list alone cannot say.

    The order of the checks is the order of their cost. Nothing here repairs a
    proposal: a shot index the model invented is not evidence about which shot it
    meant, and guessing would put a caption on someone else's footage.
    """
    if not isinstance(proposals, list):
        # Not the shape the schema promised. Nothing survived, and nothing was
        # examined either — the caller reports this as a designer failure.
        return [], 0
    accepted: list[_Callout] = []
    per_shot: dict[int, int] = {}
    dropped = 0
    for raw in proposals:
        if len(accepted) >= _MAX_CALLOUTS:
            dropped += 1
            continue
        if not isinstance(raw, Mapping):
            dropped += 1
            continue
        # The narrowed mapping's key/value types are unknowable off a JSON blob;
        # every read below is guarded, so the cast buys the reads a type without
        # buying any of them an assumption.
        entry = cast(Mapping[str, object], raw)
        index = entry.get("shot_index")
        # `bool` is a subclass of `int`, so `True` passes a bare isinstance check
        # and would address shot 1.
        if not isinstance(index, int) or isinstance(index, bool):
            dropped += 1
            continue
        if index < _FIRST_CALLOUT_SHOT_INDEX or index >= len(shots):
            dropped += 1
            continue
        if per_shot.get(index, 0) >= _MAX_CARDS_PER_SHOT:
            dropped += 1
            continue
        # The SAME window rule the title card lives under: a shot that cannot hold
        # a whole card gets none, rather than one that is trimmed mid-exit-wipe.
        on_screen_seconds = _card_window(shots[index])
        if on_screen_seconds is None:
            dropped += 1
            continue
        headline = _trim(entry.get("headline"), _MAX_HEADLINE_CHARS)
        if not headline:
            # A blank lower-third is worse than no lower-third — the same call the
            # untitled-brief path already makes for the title treatment.
            dropped += 1
            continue
        per_shot[index] = per_shot.get(index, 0) + 1
        accepted.append(
            _Callout(
                shot_index=index,
                headline=headline,
                strap=_trim(entry.get("strap"), _MAX_STRAP_CHARS),
                on_screen_seconds=on_screen_seconds,
            )
        )
    return accepted, dropped


def _positive(value: object, default: float) -> float:
    """A usable POSITIVE finite number, or *default*.

    Every number this stage reads off a ``cut_plan`` shot (a length, a speed) is
    meaningless at zero or below, and a junk value must not become a plausible
    one — a 0.0 that survives as 0.0 renders a card nobody can see, and a NaN
    reaches the compiler as an unrenderable overlay window.
    """
    try:
        number = float(value)  # ty: ignore[invalid-argument-type]
    except TypeError, ValueError, OverflowError:
        return default
    return number if math.isfinite(number) and number > 0 else default


def _card_timeline(duration_ms: int) -> dict[str, object]:
    """``lower_third`` entrance/hold/exit split for a card of *duration_ms*.

    The template reads all three and derives its own keyframes; hold takes
    whatever the entrance and exit do not, so the three always sum to
    *duration_ms* — the length of the RENDERED clip.

    That sum is not by itself a promise about the delivered reel. The compiler
    trims the overlay to the card's declared ``seconds`` window, so the exit
    wipe only survives to the screen while that window covers the whole rendered
    clip. Keeping the two lengths equal is the caller's job, and
    ``_card_window`` is where it is done: it refuses a card outright rather than
    let a short shot buy a clip whose tail is cut off mid-wipe.
    """
    entrance = int(duration_ms * _ENTRANCE_FRACTION)
    exit_ms = int(duration_ms * _EXIT_FRACTION)
    return {
        "entrance_ms": entrance,
        "hold_ms": max(duration_ms - entrance - exit_ms, 0),
        "exit_ms": exit_ms,
    }


def _card_window(shot: Mapping[str, object]) -> float | None:
    """On-screen seconds for a card pinned to *shot*, or ``None`` to refuse it.

    The card can never outlive the shot it is pinned to — an overlay longer than
    its shot would be cut mid-animation at the first edit — so the length is
    clamped to the shot. But the clamp alone is not enough, because the rendered
    clip has a floor (``_MIN_CARD_MS``: below it the entrance cannot even
    complete) and the DELIVERED window has none. A 0.6 s opening shot, which
    ``cut_plan`` admits happily, used to render an 800 ms clip and then declare a
    600 ms window; the compiler trims to the declaration, so the title vanished
    mid-exit-wipe on the first frame of the delivered reel — a visible glitch on
    the most visible element of the product.

    Raising the floor is not the answer (a shorter card cannot animate) and
    lowering the declaration is a lie about what was rendered. The only honest
    answer for a shot that cannot hold a whole card is to place no card there,
    which is what ``None`` says. Every returned value is long enough that
    ``int(seconds * 1000) == duration_ms``, so the rendered clip and the
    declared window are the same length by construction.

    The shot's own length is read WITHOUT a fallback, which is the second half
    of the same lesson. It used to come through ``_positive(..., _CARD_SECONDS)``,
    so a shot carrying no ``seconds``, a ``0.0``, or a ``NaN`` was silently read
    as a full 3.0 s — and 3.0 s clears every check below, so the card was both
    rendered and DECLARED at 3 s. ``compile`` reads the very same missing field
    as ``1.0`` (``seconds if seconds > 0.0 else 1.0``), so the compiler laid a
    1 s shot under a 3 s overlay and trimmed the title off mid-exit-wipe: the
    original defect, reproduced exactly on the input this function exists to
    make safe. And these degenerate shots are reachable — ``cut_plan``'s no-grid
    ``_even_spacing`` path emits ``max(0.0, end - start)``, which is ``0.0`` for
    a segment whose in and out round to the same millisecond.

    The fix is to refuse, not to agree with ``compile`` on a shared invented
    number. Two modules inventing the SAME fallback would still be two modules
    inventing a length nobody measured, and this function's whole contract is
    that it never asserts a shot can hold a card unless it knows the shot's
    length. A shot of unknown length is a shot that cannot be shown to hold
    anything, so it gets ``None`` — the same honest refusal a too-short shot
    gets. ``compile``'s own 1.0 fallback is a separate concern (it must still
    place SOME shot on the timeline) and is deliberately left alone.

    ``None`` here also covers the non-finite case: ``NaN`` and ``inf`` both fail
    the ``math.isfinite`` test below rather than sliding through the comparison
    — ``NaN < _MIN_CARD_MS`` is False, so a bare comparison would have ACCEPTED
    a NaN card and sent an unrenderable window to the compiler.
    """
    raw = shot.get("seconds")
    try:
        shot_seconds = float(raw)  # ty: ignore[invalid-argument-type]
    except TypeError, ValueError, OverflowError:
        return None
    if not math.isfinite(shot_seconds) or shot_seconds <= 0:
        return None
    on_screen_seconds = min(_CARD_SECONDS, shot_seconds)
    if int(on_screen_seconds * 1000) < _MIN_CARD_MS:
        return None
    return on_screen_seconds


class GraphicsDesignInput(StageInput):
    """No typed inputs — the plan is read from ``cut_plan``."""


class GraphicsCard(StageOutput):
    """One animated overlay, rendered and stored.

    ``shot_id`` keys the card to a ``cut_plan`` shot (empty ⇒ a full-frame card
    that is not attached to any one shot, e.g. the end card). ``at_seconds`` is
    relative to that shot's start, not to the reel — a reordered timeline must
    not silently move every callout.

    ``at_seconds`` / ``seconds`` are on the shot's SOURCE timeline, which is the
    timeline ``OverlayClip.retime_with_speed=True`` reads (the compiler divides
    both by the shot's ``speed`` to place them on the rendered timeline). That
    conversion is done HERE, where the shot's speed is known and the card's
    on-screen length is being chosen anyway — ``compile`` stays a dumb join and
    never does timeline arithmetic. On the ordinary un-warped shot the two
    timelines are identical.
    """

    id: str
    shot_id: str = ""
    kind: str = ""
    text: str = ""
    at_seconds: float = 0.0
    seconds: float = 0.0
    #: Stored alpha-WebM the compiler composites. EMPTY IS NEVER EMITTED: a card
    #: whose render failed is dropped (see the module docstring) because the
    #: renderer would silently skip an unfetchable overlay and report success.
    url: str = ""
    #: ffmpeg overlay placement expressions over ``W/H`` (canvas) and ``w/h``
    #: (overlay), charset-restricted by the compiler. The card is rendered at the
    #: full canvas size with its own internal safe-area padding, so the default
    #: is a 1:1 corner-pin; they exist as fields so a per-shot callout (the
    #: follow-up) can place itself without a compiler contract change.
    x: str = "0"
    y: str = "0"
    #: Compositing opacity. 1.0 because the card's own artwork already carries
    #: its translucency in the alpha plane; a second global fade would double it.
    opacity: float = 1.0


class GraphicsDesignOutput(StageOutput):
    cards: list[GraphicsCard] = []  # noqa: RUF012 -- pydantic per-instance default
    #: True ONLY when a card was designed and its render/store failed. Never infer
    #: an outage by testing ``cards == []`` — three different things produce an
    #: empty card list plus a note (see ``cards_stubbed``).
    cards_render_failed: bool = False
    #: True when the empty card list is a STUB rather than a decision: ``cut_plan``
    #: produced no shots, or the brief carries neither a title nor a brand name.
    #: No designer ran on those paths — there is no designer yet (see the TODO at
    #: the top of ``handle``); the whole "design" is ``_title_text`` reading two
    #: snapshot keys — so reporting them the same way as a deliberate "no cards"
    #: makes a stub look like a judgement. This stage PAUSES for human review, and
    #: a reviewer approving "no graphics" must be able to tell which of the two
    #: they are approving; both paths do say so in a note, but note prose is
    #: skimmable and unfilterable, so the distinction is carried structurally.
    #:
    #: Read the two flags as a SET, the same way ``segment_select`` pairs
    #: ``near_duplicates_dropped`` with ``near_duplicates_kept_for_coverage``: a
    #: bare empty list is one-sided. Neither flag ⇒ the cards, or their absence,
    #: are what the design produced. ``cards_stubbed`` ⇒ there was nothing to
    #: design. ``cards_render_failed`` ⇒ a card WAS designed and its render/store
    #: failed. Both set is unreachable: a stubbed path renders nothing to fail.
    #: The one live case of "neither flag but no cards" is an opening shot too
    #: short to hold a whole card (see ``_card_window``) — a judgement, which is
    #: why it must NOT borrow the stub flag: a reviewer told "there was nothing to
    #: design" would go hunting for a missing title instead of a longer opening.
    cards_stubbed: bool = False
    #: How many callout cards were designed, rendered, stored AND emitted. Read it
    #: with the three fields below, never alone — the same SET discipline the two
    #: flags above are documented under, for the same reason: a bare ``0`` is
    #: produced by four different things (the designer judged no shot earned a
    #: callout, the model was never asked because the cut is one shot long, the
    #: model answered and every proposal was unusable, the model never answered at
    #: all) and a reviewer approving "no callouts" must know which one they are
    #: approving. ``(0, 0, False, 0)`` is the only "the designer looked and chose
    #: none" reading.
    callouts_designed: int = 0
    #: How many of the model's proposals were REJECTED by the editorial rules in
    #: ``_usable_callouts`` — over the cap, a second card on one shot, shot 0, a
    #: shot index that is not an in-range integer, a shot too short to hold a whole
    #: card, an empty headline. A judgement about the draw, not an outage: a
    #: non-zero value here beside a non-zero ``callouts_designed`` is the normal
    #: shape of a model that over-answered and was trimmed.
    callouts_dropped_unusable: int = 0
    #: How many APPROVED callouts were then lost to a render/store failure. Kept
    #: apart from the field above for the same reason ``cards_render_failed`` is
    #: kept apart from ``cards_stubbed``: one is an editorial decision and the
    #: other is an outage worth retrying, and a single "dropped" counter would make
    #: them indistinguishable.
    callouts_render_failed: int = 0
    #: True when the callout DESIGNER did not answer — the WM call raised, or came
    #: back without the ``cards`` array the schema promises — and the stage fell
    #: back to the title treatment alone. The reel still ships with its title; this
    #: flag is the only way to tell that from a designer that ran and chose nothing.
    callouts_degraded: bool = False
    notes: list[str] = []  # noqa: RUF012


class GraphicsDesignStage(StageHandler[GraphicsDesignInput, GraphicsDesignOutput]):
    name = "graphics_design"
    input_model = GraphicsDesignInput
    output_model = GraphicsDesignOutput
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (SlotUse("llm"),)
    #: worker-management auto-creates an unknown worker on first use, so this name
    #: needs no registration step anywhere.
    worker_name = "action-reel-graphics-designer"

    def __init__(
        self, client: VideoProductionClient | None = None, wm: WMClient | None = None
    ) -> None:
        self._client = client or VideoProductionClient()
        # NOT `llm_budget_seconds()`. The client and the stage clock would hold the
        # same number and the stage clock, which starts first, wins every tie — so
        # the retryable `httpx.ReadTimeout` could never be raised. Same rung, same
        # reasoning, as `segment_select`; an AST guard in
        # tests/services/blueprints/test_llm_timeout_ladder.py fails on the tie.
        self._wm = wm or WMClient(timeout=llm_client_timeout_seconds())

    async def handle(self, payload: GraphicsDesignInput, ctx: RunContext) -> GraphicsDesignOutput:
        # TODO(action_reel graphics designer, follow-up PR): an end card, and
        # entrance timing snapped to the beat grid (`cut_plan.bpm` / `beat_index`).
        # The per-shot callouts half of the original TODO is below.
        plan = ctx.prior_output("cut_plan") or {}
        shots = plan.get("shots") or []
        if not shots:
            return GraphicsDesignOutput(
                cards=[],
                cards_stubbed=True,
                notes=["graphics_design: cut_plan produced no shots; nothing to overlay"],
            )

        title, subtitle = self._title_text(ctx)
        if not title:
            return GraphicsDesignOutput(
                cards=[],
                cards_stubbed=True,
                notes=[
                    "graphics_design: the brief carries neither a title nor a brand name, "
                    "so there is no title treatment to design; the reel renders with NO graphics"
                ],
            )

        opening = shots[0]
        on_screen_seconds = _card_window(opening)
        if on_screen_seconds is None:
            # A deliberate design decision, so NEITHER flag is set: a reviewer
            # reading `cards_stubbed` would be told "no designer ever ran", and
            # reading `cards_render_failed` would be told "a card was designed
            # and the render broke". Both are false here — the designer looked at
            # the opening shot and judged it unable to hold a whole card, either
            # because it is too short or because it declares no usable length at
            # all (missing/zero/NaN `seconds`, which `_card_window` refuses
            # rather than guessing a length for). The
            # note carries that fact, because it is an editorial call a human may
            # want to answer by re-cutting the opening rather than by retrying.
            return GraphicsDesignOutput(
                cards=[],
                notes=[
                    "graphics_design: the opening shot is too short to hold a title card, "
                    "or carries no usable length at all (a card needs at least "
                    f"{_MIN_CARD_MS / 1000:.1f}s to fly in and back out, and the overlay may "
                    "not outlive the shot it is pinned to), so no card was placed and the "
                    "reel renders with NO graphics. Lengthen the opening shot to get a title."
                ],
            )
        # The designer runs BEFORE the title render and its failure is absorbed
        # here, never raised: a callout outage must not cost the reel the one card
        # the product has always shipped. The reverse order would put a paid render
        # in front of the cheap call that decides how many more renders to buy.
        callouts, dropped, degraded = await self._design_callouts(ctx, shots)

        card = await self._render_card(
            ctx,
            opening,
            kind="title",
            headline=title,
            strap=subtitle,
            on_screen_seconds=on_screen_seconds,
            slot="title",
        )
        if card is None:
            # The title render failing is an outage of the same recorder every
            # callout would use, so the callouts are not attempted: three more
            # requests to a service that just failed buy nothing but latency on a
            # stage a human is waiting at.
            return GraphicsDesignOutput(
                cards=[],
                cards_render_failed=True,
                callouts_dropped_unusable=dropped,
                callouts_degraded=degraded,
                notes=[
                    "graphics_design: the title overlay FAILED TO RENDER and was dropped; "
                    "the reel will be delivered with NO graphics on it (the footage and the "
                    "cut are unaffected). Re-run this stage to try again."
                ],
            )

        callout_cards: list[GraphicsCard] = []
        render_failed = 0
        for callout in callouts:
            rendered = await self._render_card(
                ctx,
                shots[callout.shot_index],
                kind="callout",
                headline=callout.headline,
                strap=callout.strap,
                on_screen_seconds=callout.on_screen_seconds,
                slot=f"callout-{callout.shot_index}",
            )
            if rendered is None:
                # Soft-fail PER CARD, and counted. `cards_render_failed` stays
                # False on purpose: it means "the reel has no graphics", which is
                # not true while the title is on it.
                render_failed += 1
                continue
            callout_cards.append(rendered)

        notes: list[str] = []
        if degraded:
            notes.append(
                "graphics_design: the callout designer did not answer, so the reel carries "
                "its title treatment ONLY and no per-shot callouts. The title, the footage "
                "and the cut are unaffected. Re-run this stage to try the callouts again."
            )
        if render_failed:
            notes.append(
                f"graphics_design: {render_failed} designed callout(s) FAILED TO RENDER and "
                "were dropped; the reel keeps its title treatment and the callouts that did "
                "render. Re-run this stage to try again."
            )
        return GraphicsDesignOutput(
            cards=[card, *callout_cards],
            callouts_designed=len(callout_cards),
            callouts_dropped_unusable=dropped,
            callouts_render_failed=render_failed,
            callouts_degraded=degraded,
            notes=notes,
        )

    async def _design_callouts(
        self, ctx: RunContext, shots: list[Mapping[str, object]]
    ) -> tuple[list[_Callout], int, bool]:
        """``(callouts, dropped, degraded)`` — the model's draw, after enforcement.

        Never raises. Every failure of the designer resolves to
        ``([], 0, True)``: the caller still renders the title, and the flag is
        what tells a reviewer that the empty callout list is an outage rather than
        an editorial judgement.
        """
        if len(shots) <= _FIRST_CALLOUT_SHOT_INDEX:
            # A one-shot cut has no shot a callout may legally sit on (shot 0 is
            # the title's). Not degraded — there was nothing to ask about, and
            # paying for a call whose every answer is unusable is worse than not
            # calling.
            return [], 0, False
        try:
            response = await self._wm.message(
                worker_name=self.worker_name,
                message=json.dumps({"shots": _describe_shots(shots)}, ensure_ascii=False),
                system_prompt=_CALLOUT_SYSTEM_PROMPT,
                output_schema=_CALLOUT_SCHEMA,
                # See _CALLOUT_MAX_TOKENS: omitting this leaves WM's 6000 default,
                # which a thinking model can burn before emitting any JSON — and
                # that 422 is TERMINAL, not retryable.
                max_tokens=_CALLOUT_MAX_TOKENS,
                # Chain-of-thought OFF, and this is not a preference. Measured
                # through worker-management on the production path (segment_select's
                # `_one_batch` carries the full write-up): with reasoning ON the
                # model spends the whole budget thinking and returns NOTHING, which
                # surfaces one layer up as `Exceeded maximum output retries (3)`;
                # with it OFF the same call answers in well under a minute. This
                # stage PAUSES for a human, so a call that times out costs a person
                # sitting in front of a stalled run.
                enable_reasoning=False,
            )
        except Exception as exc:  # noqa: BLE001 -- soft-fail: the title still ships
            log.warning("graphics_design: callout designer failed, title only: %s", exc)
            return [], 0, True
        data = response.output
        proposals = data.get("cards") if isinstance(data, dict) else None
        if not isinstance(proposals, list):
            # The schema promised a `cards` array and did not deliver one. That is
            # the designer failing, not the designer choosing nothing — reported as
            # a degrade so the two stay tellable apart.
            log.warning("graphics_design: callout designer returned no `cards` array")
            return [], 0, True
        callouts, dropped = _usable_callouts(proposals, shots)
        return callouts, dropped, False

    def _title_text(self, ctx: RunContext) -> tuple[str, str]:
        """``(title, subtitle)`` for the opening treatment.

        The project title is the headline and the brand name the strap; a run
        with only a brand promotes it to the headline rather than rendering a
        card with an empty name line.
        """
        snapshot = ctx.config_snapshot
        brand = snapshot.get("brand") or {}
        brand_name = str(brand.get("name") or "")
        project_title = str(snapshot.get("title") or "")
        if project_title:
            return project_title, brand_name
        return brand_name, ""

    async def _render_card(
        self,
        ctx: RunContext,
        shot: Mapping[str, object],
        *,
        kind: str,
        headline: str,
        strap: str,
        on_screen_seconds: float,
        slot: str,
    ) -> GraphicsCard | None:
        """Record → poll → fetch → store one transparent card; ``None`` on failure.

        Same sequence (and same per-card soft-fail posture) as
        ``shared/title_cards.build_title_card_shots`` — the difference is only
        ``transparent=True`` + a WebM container, because this clip is composited
        over the picture instead of being spliced beside it.

        ONE renderer for both card kinds on purpose. The title and the callouts
        differ only in their text, their shot and their `kind`; giving the callouts
        their own copy of this sequence would give them their own copy of the
        alpha-WebM contract, the source-timeline conversion and the
        never-emit-a-card-without-a-URL rule — three invariants that took an
        outage each to learn and must not exist twice.

        *slot* keys the idempotency: it distinguishes the title from each callout
        so a resubmitted stage reuses every original recording instead of paying
        for a second one, and so two cards on the same run cannot collide on one
        job. `lower_third` is the ONLY overlay-safe template, for both kinds.
        """
        shot_id = str(shot.get("id") or "")
        # The rendered clip is EXACTLY the window this card will declare below.
        # There is no floor applied here on purpose: a floor would silently make
        # the clip longer than the window the compiler trims it to, and the tail
        # of the exit wipe would be cut off in the delivered reel. `_card_window`
        # has already refused any shot that cannot hold the minimum, so this
        # multiplication needs no rescue.
        duration_ms = int(on_screen_seconds * 1000)
        # SOURCE-timeline length (see GraphicsCard). speed 0.0 means the plan is
        # unsolved; 1.0 keeps the two timelines identical, which is the only
        # honest answer when there is no warp to convert against.
        speed = _positive(shot.get("speed"), 1.0)

        deliverable = ctx.config_snapshot.get("deliverable") or {}
        width, height = format_dims(deliverable.get("format"), quality_tier(ctx.config_snapshot))
        brand = ctx.config_snapshot.get("brand") or {}
        # Accent colours only, NOT `theme_motion_params`: the surface override it
        # resolves paints a background this render deliberately throws away
        # (the recorder neutralises `body`'s background to get the alpha plane),
        # so sending it would only invite a future reader to "fix" the contrast
        # of a canvas that is not there.
        params: dict[str, object] = {
            "name": headline,
            "role": strap,
            **motion_brand_params(brand),
            **_card_timeline(duration_ms),
        }
        card_id = str(uuid7())
        try:
            job_id = await self._client.record_animation(
                template_id=_TEMPLATE,
                params=params,
                part_id=f"reel-{slot}",
                # The MASTER frame grid, not the deliverable's nominal fps. A card
                # authored at 30 and conformed into a 24fps master is resampled
                # 5:4 — judder on the most visible element in the reel, which is
                # the exact judder `cadence` exists to kill. `cut_plan` and
                # `compile` both ask this same function; so does the overlay.
                fps=resolve_master_fps(ctx.config_snapshot, ctx.prior_output("footage_probe")),
                width=width,
                height=height,
                duration_ms=duration_ms,
                output_format=_CARD_FORMAT,
                transparent=True,
                # A resubmit after a transient wait failure must return the
                # ORIGINAL job rather than start a second paid recording.
                #
                # ⚠️ THE TEXT IS PART OF THE KEY, and that is the whole point.
                # `video_production_client` dedupes on this key SERVER-SIDE and
                # hands back the original job, so a key that omits the text
                # returns the OLD WebM for new words. That was safe while the
                # only card was the deterministic title; a callout headline is
                # model-generated and changes between draws. Without the digest,
                # this stage's own note — "Re-run this stage to try the callouts
                # again" — was a lie: the re-run would report the new headline in
                # `card.text` and deliver a reel still showing the old one. A
                # short blake2b of the exact rendered strings is enough; it must
                # cover BOTH lines, because a strap-only edit is still an edit.
                idempotency_key=(
                    f"{ctx.run_id}:{self.name}:{shot_id}:{slot}:{_text_digest(headline, strap)}"
                ),
            )
            await self._client.wait_for_content_recording(job_id, timeout=_RECORD_TIMEOUT_S)
            video_bytes, _ctype, _disp = await self._client.get_recording_video(job_id)
            url = await store_generated_video(
                run_id=str(ctx.run_id),
                stage_name=self.name,
                video_id=card_id,
                video_bytes=video_bytes,
                video_format=_CARD_FORMAT,
            )
        except Exception as exc:  # noqa: BLE001 -- soft-fail: the reel still renders
            log.warning("graphics_design: %s overlay render failed, dropping card: %s", kind, exc)
            return None
        if not url:
            # Storage handed back nothing to fetch. Emitting the card anyway
            # would be the exact lie this stage exists to avoid: the compiler
            # drops an unfetchable overlay in silence.
            log.warning("graphics_design: %s overlay stored to an empty URL, dropping card", kind)
            return None
        return GraphicsCard(
            id=card_id,
            shot_id=shot_id,
            kind=kind,
            text=headline,
            at_seconds=0.0,
            seconds=round(on_screen_seconds * speed, 3),
            url=url,
        )
