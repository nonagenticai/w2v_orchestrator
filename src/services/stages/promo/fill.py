"""Promo fill plan — the branded end card and the breathing holds that bring a
narrated promo up to the length it was ordered at.

Why this exists (run 01a10d84, 2026-10-05): a promo ordered at 60s with a
verbatim ``source_script`` of seven approved lines narrated 35.4s and shipped at
59% of its order. A verbatim script cannot be lengthened — every word is
approved — so the only honest fill is picture time: a pause after each line
and a branded end card carrying the brand name and the call-to-action. The film
also ended on the CTA line's blank gradient because no end card was ever asked
for (``title_card`` is opt-in and nobody opted in).

Policy, deliberately bounded:

* the end card is ON by default for a promo that has a brand name or a CTA
  (an explicit ``title_card`` block in the run config always wins, including
  an explicit ``enabled: false``);
* the end card COUNTS toward the order. It starts at ``_END_CARD_BASE_S``,
  but never pushes an otherwise on-target film over the +10% ceiling: when the
  narration leaves less room than that, the card shrinks to the room left
  (keeping ``_CEILING_MARGIN_S`` of headroom for the encoder), and when the
  room is under ``_END_CARD_MIN_S`` — too short to read — the default card is
  dropped (the CTA is still the last spoken line). 30s ordered, 31s narrated:
  a 1.75s card, 32.75s delivered (+9%), not 34s (+13%);
* a shortfall goes to the end card first (up to ``_END_CARD_MAX_S``), then to
  an equal pause after every narrated line, at most ``_HOLD_MAX_S`` each —
  longer than that and the film reads as dead air;
* an order the bounded plan cannot meet is NOT papered over, in either
  direction: narration too short to fill even with a full card and full
  pauses, or narration (plus an operator-fixed outro) already over the
  ceiling, is recorded as ``plan_infeasible`` with the numbers, so
  ``duration_conformance`` reports WHY. Per the existing duration policy the
  render still ships (conformance is a recorded verdict, never a failure); the
  lever is the order — a ``target_duration_seconds`` the approved lines fit.

Per-line motion promos (``direction.motion == "per_line"``) narrow this, by run
01a11295 (2026-10-06, REJECTED at frame QA): there the end card is absorbed into
the closing line's own card (``promo_render``), and the fill stretched it to
``_END_CARD_MAX_S`` to reach a 60s order — narration ended at 47.1s and the
closing card then held for 10.5s on music alone, 18% of the film. So on those
films the closing shot ends at most :data:`CLOSING_TAIL_MAX_SECONDS` after the
last line's narration (its hold, the CTA floor and the end card together), the
end card is never stretched to fill the order, and a film that cannot reach the
band without that padding is recorded as ``plan_infeasible`` instead. Every
other promo keeps the policy above unchanged (its separate end card is a card
of its own, built in and read as one — see the 2026-10-05 tests).

Pure functions over ``ctx`` so the render's shot holds (``build_shots``) and the
end card's length (``title_card_config``) are computed from the same inputs and
cannot disagree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.services.stages.shared.duration_conformance import (
    DURATION_TOLERANCE_FRACTION,
    promo_ordered_seconds,
)
from src.services.stages.shared.motion_slides import motion_per_line_requested
from src.services.stages.shared.title_cards import title_card_seconds
from src.services.stages.shot_assembly import ShotDict

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

_END_CARD_BASE_S = 3.0
_END_CARD_MAX_S = 8.0
# Shorter than this a two-line card cannot be read; the default card is dropped.
_END_CARD_MIN_S = 1.5
_HOLD_MAX_S = 2.0
#: The call-to-action (last) line's card stays on screen at least this long, so a
#: short CTA ("Request access at merqi.nl", 2.1s narrated) can still be read.
CTA_MIN_SECONDS = 2.5
#: Per-line motion promos: the closing shot (the CTA line's card, which carries
#: the end card) ends at most this long after the last narration ends — as
#: DELIVERED, i.e. after the compiler puts the shot on its frame grid.
CLOSING_TAIL_MAX_SECONDS = 3.0
#: The coarsest master cadence the compiler delivers (video_compiler
#: ``cadence._LEGAL_RATES``: 24/25/30/50/60). The compiler encodes a shot of
#: ``s`` seconds as ``ceil(s * fps)`` frames, so a shot grows by up to one frame;
#: one frame at 24 fps covers that at every legal rate.
COARSEST_FPS = 24
#: Planning slack for the millisecond rounding of ``duration_ms`` / shot seconds.
_MS_SLACK_S = 0.001
# Headroom kept under the +10% ceiling when the card is shrunk to fit: the
# compiler's encoded length runs a few frames past the declared shot sum.
_CEILING_MARGIN_S = 0.25


@dataclass(frozen=True)
class PromoFill:
    """How a promo's picture time is padded up to its order."""

    ordered_seconds: float | None
    narrated_seconds: float
    line_count: int
    end_card_seconds: float
    hold_seconds: float
    #: Why the DEFAULT end card (the on-screen CTA) was planned out, with the
    #: numbers; ``None`` when it is kept or none was asked for. Recorded on the
    #: render as ``END_CARD_DROPPED`` — a promo whose CTA is never SHOWN is a
    #: conversion defect, never a silent trim.
    end_card_dropped: str | None = None
    #: Extra picture time the CTA line needs to reach :data:`CTA_MIN_SECONDS`.
    cta_floor_seconds: float = 0.0
    #: The cap on the closing tail (hold + CTA floor + end card after the last
    #: narration), when one applies: :data:`CLOSING_TAIL_MAX_SECONDS` on a
    #: per-line motion promo, ``None`` otherwise.
    closing_tail_max_seconds: float | None = None

    @property
    def projected_seconds(self) -> float:
        return (
            self.narrated_seconds
            + self.end_card_seconds
            + self.hold_seconds * self.line_count
            + self.cta_floor_seconds
        )

    @property
    def infeasible_reason(self) -> str | None:
        """Why the order cannot be met by the bounded plan; ``None`` when it can."""
        if not self.ordered_seconds or not self.line_count:
            return None  # nothing ordered, or no narration measured to plan from
        ceiling = self.ordered_seconds * (1.0 + DURATION_TOLERANCE_FRACTION)
        if self.projected_seconds > ceiling:
            return (
                f"narration is {self.narrated_seconds:.1f}s over {self.line_count} line(s)"
                + (
                    f" plus {self.end_card_seconds:.1f}s of end/title card(s)"
                    if self.end_card_seconds
                    else ""
                )
                + f" = {self.projected_seconds:.1f}s, over the {ceiling:.1f}s ceiling of the "
                f"{self.ordered_seconds:.0f}s ordered — order >= "
                f"{self.projected_seconds / (1.0 + DURATION_TOLERANCE_FRACTION):.0f}s or "
                "cut script"
            )
        floor = self.ordered_seconds * (1.0 - DURATION_TOLERANCE_FRACTION)
        if self.projected_seconds >= floor:
            return None
        capped = (
            f" (the closing card ends <= {self.closing_tail_max_seconds:.1f}s after the "
            "last line and is never padded to fill the order)"
            if self.closing_tail_max_seconds is not None
            else ""
        )
        return (
            f"narration is {self.narrated_seconds:.1f}s over {self.line_count} line(s); "
            f"with a {self.end_card_seconds:.1f}s end card{capped} and "
            f"{self.hold_seconds:.1f}s pauses the film reaches {self.projected_seconds:.1f}s of the "
            f"{self.ordered_seconds:.0f}s ordered — order <= "
            f"{self.projected_seconds / (1.0 - DURATION_TOLERANCE_FRACTION):.0f}s or "
            "supply more script"
        )


def plan_promo_fill(
    ordered_seconds: float | None,
    line_seconds: list[float],
    *,
    end_card: bool,
    fixed_card_seconds: float = 0.0,
    closing_tail_max: float | None = None,
) -> PromoFill:
    """Split the shortfall between the end card and per-line pauses (bounded).

    ``end_card`` is the stretchable default card; ``fixed_card_seconds`` is a
    card of fixed length the operator configured (only the pauses flex then).
    ``closing_tail_max`` (per-line motion promos): the default card is not
    stretched toward the order, and it is cut so the last line's hold, its CTA
    floor and the card end at most this long after the last narration.
    """
    narrated = sum(max(0.0, s) for s in line_seconds)
    lines = len(line_seconds)
    card = _END_CARD_BASE_S if end_card else fixed_card_seconds
    dropped: str | None = None
    if end_card and ordered_seconds:
        card, dropped = _fit_default_card(ordered_seconds, narrated, card)
    hold = 0.0
    # Pauses are planned only AFTER the card and only up to the ORDER (not the
    # ceiling), so they never cost the card its room: the CTA is placed first.
    shortfall = (ordered_seconds or 0.0) - narrated - card
    if shortfall > 0:
        if end_card and not dropped and closing_tail_max is None:
            extra = min(shortfall, _END_CARD_MAX_S - card)
            card += max(extra, 0.0)
            shortfall -= max(extra, 0.0)
        if lines and shortfall > 0:
            hold = min(shortfall / lines, _HOLD_MAX_S)
    cta_floor = max(0.0, CTA_MIN_SECONDS - (max(0.0, line_seconds[-1]) + hold)) if lines else 0.0
    if end_card and closing_tail_max is not None:
        card = max(
            0.0, min(card, round(planned_tail_budget(closing_tail_max), 3) - hold - cta_floor)
        )
    return PromoFill(
        ordered_seconds=ordered_seconds,
        narrated_seconds=narrated,
        line_count=lines,
        end_card_seconds=round(card, 3),
        hold_seconds=round(hold, 3),
        end_card_dropped=dropped,
        cta_floor_seconds=round(cta_floor, 3),
        closing_tail_max_seconds=closing_tail_max if end_card else None,
    )


def planned_tail_budget(delivered_max: float) -> float:
    """The closing tail to PLAN so the DELIVERED tail is at most ``delivered_max``.

    Run 01a11511 (2026-10-07) planned exactly 3.0 s (2.12 s of narration + 2.0 s
    hold + 1.0 s card = a 5.122 s shot) and delivered 3.04 s: the compiler
    rounds every shot UP to whole frames (5.122 s → 154 frames at 30 fps =
    5.133 s) and ``duration_ms`` rounds to the millisecond. So the plan keeps one
    frame of the coarsest legal cadence, plus a millisecond, in hand.
    """
    return delivered_max - 1.0 / COARSEST_FPS - _MS_SLACK_S


#: The closing tail the fill PLANS on a per-line film (2.957 s for a 3.0 s cap).
CLOSING_TAIL_PLANNED_SECONDS = round(planned_tail_budget(CLOSING_TAIL_MAX_SECONDS), 3)


def delivered_seconds(seconds: float, fps: int) -> float:
    """A shot's length after the compiler puts it on the ``fps`` frame grid (whole
    frames, rounded up; the millisecond shot length is what is sent)."""
    return math.ceil(round(seconds, 3) * fps - 1e-9) / fps


def _fit_default_card(ordered: float, narrated: float, card: float) -> tuple[float, str | None]:
    """The default card's length under the order's ceiling, and why it was dropped.

    * fits with ``_CEILING_MARGIN_S`` of headroom → unchanged;
    * the room left (with headroom) is readable → shrunk into it;
    * without the headroom a minimum card still fits inside the band → kept at
      ``_END_CARD_MIN_S`` (the CTA outranks the encoder headroom);
    * narration ALONE is already over the ceiling → kept at the minimum too:
      dropping it would not bring the film back into the band, it would only
      lose the CTA (``plan_infeasible`` names the overrun);
    * otherwise → dropped, and the reason carries the numbers.
    """
    ceiling = ordered * (1.0 + DURATION_TOLERANCE_FRACTION)
    room = ceiling - narrated
    if room - _CEILING_MARGIN_S >= card:
        return card, None
    if room - _CEILING_MARGIN_S >= _END_CARD_MIN_S:
        return room - _CEILING_MARGIN_S, None
    if room >= _END_CARD_MIN_S or narrated > ceiling:
        return _END_CARD_MIN_S, None
    return 0.0, (
        f"default end card (on-screen CTA) dropped: {narrated:.1f}s of narration leaves "
        f"{room:.2f}s under the {ceiling:.1f}s ceiling of the {ordered:.0f}s order, less "
        f"than the {_END_CARD_MIN_S:.1f}s a readable card needs; the CTA is spoken but not "
        f"shown — order >= "
        f"{math.ceil((narrated + _END_CARD_MIN_S) / (1.0 + DURATION_TOLERANCE_FRACTION))}s "
        "to keep it"
    )


def _brand_name(ctx: RunContext) -> str:
    brand = ctx.config_snapshot.get("brand") or {}
    return str(brand.get("name") or "").strip() if isinstance(brand, dict) else ""


def _call_to_action(ctx: RunContext) -> str:
    brief = ctx.config_snapshot.get("brief") or {}
    return str(brief.get("call_to_action") or "").strip() if isinstance(brief, dict) else ""


def default_end_card(ctx: RunContext) -> bool:
    """Whether the promo gets its default branded end card."""
    if ctx.config_snapshot.get("title_card") is not None:
        return False  # an explicit block (even a disabled one) is the operator's call
    return bool(_brand_name(ctx) or _call_to_action(ctx))


def _narrated_line_seconds(ctx: RunContext) -> list[float]:
    """Per storyboard line: the narration clip's length, else the planned length."""
    story = ctx.prior_output("promo_storyboard") or {}
    narrate = ctx.prior_output("promo_narrate") or {}
    clip_seconds = {
        str(clip.get("line_id")): float(clip.get("seconds") or 0.0)
        for clip in narrate.get("clips") or []
        if isinstance(clip, dict)
    }
    out: list[float] = []
    for shot in story.get("shots") or []:
        line_id = str(shot.get("line_id"))
        seconds = clip_seconds.get(line_id) or float(shot.get("seconds") or 0.0)
        out.append(seconds)
    return out


def promo_fill(ctx: RunContext) -> PromoFill:
    """The run's fill plan, from the ordered length and the measured narration.

    The default end card stretches (base to max); explicitly configured cards
    keep their own length and only the pauses absorb the rest.
    """
    ordered = promo_ordered_seconds(ctx.config_snapshot)
    lines = _narrated_line_seconds(ctx)
    if default_end_card(ctx):
        return plan_promo_fill(
            ordered,
            lines,
            end_card=True,
            closing_tail_max=(
                CLOSING_TAIL_MAX_SECONDS if motion_per_line_requested(ctx.config_snapshot) else None
            ),
        )
    return plan_promo_fill(
        ordered,
        lines,
        end_card=False,
        # Planned exactly as the compile plans it (CompileStageBase.title_card_plan:
        # uncapped), so the fill and the render agree on the card seconds.
        fixed_card_seconds=title_card_seconds(
            ctx.config_snapshot.get("title_card"), ordered_seconds=None
        ),
    )


def end_card_config(ctx: RunContext) -> ShotDict | None:
    """The default branded end card's ``title_card`` block, or ``None``.

    Brand name (from the project's brand — the tenant's Brand Kit, verbatim) as
    the title, the brief's call-to-action beneath it. With no brand name the
    CTA is the title. Sized by :func:`promo_fill`.
    """
    if not default_end_card(ctx):
        return None
    seconds = promo_fill(ctx).end_card_seconds
    if seconds <= 0:
        # The plan dropped the card to keep the film inside the order's ceiling.
        # ``None`` — never a 0s block: ``plan_title_cards`` reads a falsy
        # ``duration_seconds`` as "use the 3s default".
        return None
    name, cta = _brand_name(ctx), _call_to_action(ctx)
    return {
        "enabled": True,
        "intro": False,
        "outro": True,
        "outro_title": name or cta,
        "outro_subtitle": cta if name else "",
        "duration_seconds": seconds,
    }


def _held(shot: ShotDict, seconds: float) -> ShotDict:
    """``shot`` lasting ``seconds`` — declared so the COMPILER honours it.

    ``max_seconds`` is load-bearing, not decoration. video_compiler treats a
    narrated shot's ``seconds`` as a floor ONLY when the shot also carries
    ``max_seconds`` (``renderer._narrated_picture_floor``: "ONLY A RECONCILED
    ``seconds`` IS A FLOOR"); without it the segment is cut to the probed
    narration and the planned hold is discarded. Run 01a10fcf (2026-10-06): the
    fill planned a 2.0s hold after each of 7 lines, every shot dict carried it,
    and the film still delivered 43.5s of a 60s order with 0.0-0.56s gaps —
    each shot exactly its narration. ``max_seconds`` is also the freeze-pad
    drift ceiling (advisory unless the compile carries ``freeze_pad_cap``); at
    the shot's full length it caps nothing.
    """
    prior_max = shot.get("max_seconds")
    ceiling = max(seconds, float(prior_max)) if isinstance(prior_max, (int, float)) else seconds
    return {**shot, "seconds": seconds, "max_seconds": ceiling}


def apply_holds(
    shots: list[ShotDict], hold_seconds: float, *, last_min_seconds: float = 0.0
) -> list[ShotDict]:
    """Lengthen every narrated shot by ``hold_seconds`` — a pause after its line.

    The narration still starts at the shot's head; the extra picture time is the
    breath after the sentence (the music bed carries it). The LAST shot (the
    promo's call-to-action line) is additionally held to ``last_min_seconds`` so
    its card stays readable. Nothing to add → unchanged.
    """
    out: list[ShotDict] = []
    last = max((i for i, shot in enumerate(shots) if shot.get("seconds")), default=-1)
    for index, shot in enumerate(shots):
        base = float(shot.get("seconds") or 0.0)
        if not base:
            out.append(shot)
            continue
        seconds = base + max(hold_seconds, 0.0)
        if index == last:
            seconds = max(seconds, last_min_seconds)
        out.append(_held(shot, seconds) if seconds > base else shot)
    return out


def extend_shot(shots: list[ShotDict], line_id: str, extra_seconds: float) -> list[ShotDict]:
    """``shots`` with the shot of ``line_id`` lasting ``extra_seconds`` longer.

    Declared the way :func:`apply_holds` declares a hold (``max_seconds`` with
    it), so the compiler honours it. Promo's closing beat takes the default end
    card's seconds this way when the motion brief ran (see ``render``).
    """
    if extra_seconds <= 0:
        return shots
    return [
        _held(shot, float(shot.get("seconds") or 0.0) + extra_seconds)
        if str(shot.get("line_id")) == line_id
        else shot
        for shot in shots
    ]
