"""website_review's ordered length — ONE resolver for script, compile and the door.

Prod run 01a09f82-7179-73a5-a23a-40c15d148b37 ordered 10.0s and delivered 15.1s.
Two defects, one per half of the arithmetic:

* **The order key was split.** The wizard writes ``deliverable.target_duration_seconds``;
  the script budget and the compile's conformance check only ever read
  ``brief.duration_seconds`` (default 120). A 10s order was invisible to both.
* **The script budgeted the whole film as narration.** The compile splices
  intro/outro title cards (``title_card`` block, 3s each by default) around the
  narrated lines, so a script that fills the order exactly still overruns by the
  cards: 4 lines ~9s + 2 cards 6s = 15.1s.

So the ordered length resolves here, once, and the script writes against
:func:`narration_budget_seconds` — the order minus the fixed seconds the compile adds.
The +/-10% conformance band is untouched; it still grades the delivered film against
the FULL order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Literal

from pydantic import BaseModel, Field, model_validator

from src.services.stages.shared.duration_conformance import REVIEW_DEFAULT_SECONDS, _usable
from src.services.stages.shared.title_cards import TitleCardPlan, plan_title_cards
from src.services.stages.shot_assembly import MIN_NARRATED_BEAT_SECONDS

if TYPE_CHECKING:
    from collections.abc import Mapping

#: The acts the website_review blueprint STRUCTURALLY guarantees, and therefore the
#: narrated lines a review can never have fewer of. Derived from the script stage, not
#: chosen:
#:
#: * ``script._TWO_ACT_STRUCTURE_BLOCK`` instructs the draw to emit a "before" act (the
#:   critique, played over the ORIGINAL page) and then an "after" act (the reveal,
#:   played over the REDESIGN) — "the after act is the payoff".
#: * ``script._protected_indices`` then pins the first ``phase == "after"`` line against
#:   the trim, because ``compile._key_lines`` binds the redesign's visuals to after-act
#:   lines: with the after act gone "the redesign — the entire point of the video —
#:   never appears on screen".
#:
#: So a one-line website_review is not a SHORT review, it is a different (broken)
#: deliverable, and an order that can only pay for one line is an order the pipeline
#: cannot deliver. ``len()`` rather than a literal so the two stay tied.
REQUIRED_NARRATED_PHASES: Final[tuple[str, ...]] = ("before", "after")
MIN_NARRATED_LINES: Final = len(REQUIRED_NARRATED_PHASES)

#: The shortest order the PRODUCT accepts, whatever the physics allow.
#:
#: Owner decision 2026-09-15: **the floor is 10s everywhere**. The wizard already
#: refused anything under 30s client-side while the server happily took 10s (and, with
#: title cards off, 2.4s) — two doors disagreeing about the same number. 10s is now the
#: one number, and it is enforced regardless of the ``title_card`` block.
#:
#: This sits ABOVE the physical floor (:func:`physical_minimum_order_seconds` — 8.4s
#: with cards on, 2.4s off), so it is enforceable without fighting the compiler: no
#: order this accepts is one the blueprint cannot physically deliver. It is a PRODUCT
#: number, not a derived one — do not "recompute" it from
#: ``MIN_NARRATED_BEAT_SECONDS``, which is the compiler's 1.2s flash-cut floor and a
#: different concern entirely.
MINIMUM_ORDER_SECONDS: Final = 10.0


class OrderRefusal(BaseModel):
    """The typed 400 body when a website_review order is below the floor.

    Same shape as ``api.routes.media.ThumbRefusal``: a machine-readable ``reason``, the
    two numbers that decided it, and a human ``message`` the FE renders verbatim. The
    create door used to surface this as a pydantic ``422`` whose body was a validator
    traceback — an operator saw "unprocessable entity" and no number at all.

    Branch on ``reason`` (a closed Literal); render ``message``, do not compose prose
    from the floats.
    """

    reason: Literal["order_below_floor"] = "order_below_floor"
    ordered_seconds: float
    minimum_seconds: float
    message: str = ""

    @model_validator(mode="after")
    def _fill_message(self) -> OrderRefusal:
        if not self.message:
            self.message = (
                f"website_review ordered {self.ordered_seconds:g}s, but the shortest "
                f"possible review is {self.minimum_seconds:g}s. "
                f"Order at least {self.minimum_seconds:g}s."
            )
        return self


class OrderBelowFloorError(ValueError):
    """``ValueError`` carrying the typed body, so both doors can answer 400 with it.

    Stays a ``ValueError`` so the existing ``except ValueError -> 400`` at the run door
    keeps working if a caller is ever missed; the routes that DO know about it hand the
    client :attr:`refusal` instead of ``str(exc)``.
    """

    def __init__(self, refusal: OrderRefusal) -> None:
        super().__init__(refusal.message)
        self.refusal = refusal


def _section(snapshot: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = snapshot.get(key)
    return value if isinstance(value, dict) else {}


def website_review_ordered_seconds(snapshot: Mapping[str, object]) -> float:
    """``deliverable.target_duration_seconds`` → ``brief.duration_seconds`` → 120.

    The deliverable key wins: it is what the wizard writes and it is per-deliverable,
    so it is the more specific order. ``brief.duration_seconds`` remains honoured for
    briefs authored against the old key.
    """
    for value in (
        _section(snapshot, "deliverable").get("target_duration_seconds"),
        _section(snapshot, "brief").get("duration_seconds"),
    ):
        usable = _usable(value)
        if usable is not None and usable > 0:
            return usable
    return REVIEW_DEFAULT_SECONDS


#: Key of the reservation on the ``website_review_script`` stage output.
RESERVED_TITLE_CARDS_KEY: Final = "reserved_title_cards"


class ReservedTitleCards(BaseModel):
    """The title-card seconds a script SUBTRACTED from its narration budget.

    Persisted on ``WebsiteReviewScriptOutput.reserved_title_cards`` and rendered
    verbatim by the compile. It is NOT the ``title_card`` block: that is what the
    brief CONFIGURED (2 x 3.0s, say); this is what one particular script actually
    RESERVED after the 20% cap (one 2.0s intro on a 10s order). ``0.0`` means "this
    script reserved no such card" — a recorded fact, distinct from the whole record
    being ABSENT, which means the script predates the record.
    """

    intro_seconds: float = Field(ge=0.0)
    outro_seconds: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _both_cards_share_one_length(self) -> ReservedTitleCards:
        # A plan has ONE per-card length (``TitleCardPlan.seconds_per_card``), so a
        # record with two different positive lengths was not written by this code and
        # cannot be rendered "exactly" — refuse it rather than pick one silently.
        if 0 < self.intro_seconds != self.outro_seconds > 0:
            raise ValueError(
                f"reserved title cards disagree on length: intro {self.intro_seconds:g}s, "
                f"outro {self.outro_seconds:g}s"
            )
        return self

    @classmethod
    def from_plan(cls, plan: TitleCardPlan | None) -> ReservedTitleCards:
        if plan is None:
            return cls(intro_seconds=0.0, outro_seconds=0.0)
        return cls(intro_seconds=plan.intro_seconds, outro_seconds=plan.outro_seconds)

    @property
    def total_seconds(self) -> float:
        return self.intro_seconds + self.outro_seconds

    def as_plan(self) -> TitleCardPlan | None:
        """The render plan for exactly these seconds; ``None`` when none were reserved."""
        intro, outro = self.intro_seconds > 0, self.outro_seconds > 0
        if not intro and not outro:
            return None
        return TitleCardPlan(
            intro=intro,
            outro=outro,
            seconds_per_card=self.intro_seconds if intro else self.outro_seconds,
        )


def _title_card_block(snapshot: Mapping[str, object]) -> object:
    """The ``title_card`` block, falling back to ``brief.title_card`` for a pre-run
    brief (the run snapshot lifts that key verbatim, see ``run_setup._BRIEF_LIFT_KEYS``)."""
    cfg = snapshot.get("title_card")
    if cfg is None:
        cfg = _section(snapshot, "brief").get("title_card")
    return cfg


def reserve_title_cards(snapshot: Mapping[str, object]) -> ReservedTitleCards:
    """SCRIPT TIME ONLY: the card seconds a script written NOW takes out of its budget.

    The ``title_card`` block planned against :func:`website_review_ordered_seconds`,
    i.e. capped at 20% of the order: a 10s order reserves one 2.0s intro, not 6.0s.
    Cards are opt-in: a brief with no enabled block reserves none.

    The script stage persists this on its output and the compile renders THAT record
    (:func:`title_card_plan_to_render`). The compile never calls this function: a run
    scripted under one cap rule and compiled under another would render cards its
    narration was never budgeted around.
    """
    plan = plan_title_cards(
        _title_card_block(snapshot), ordered_seconds=website_review_ordered_seconds(snapshot)
    )
    return ReservedTitleCards.from_plan(plan)


def title_card_plan_to_render(
    snapshot: Mapping[str, object], script_output: Mapping[str, object] | None
) -> TitleCardPlan | None:
    """COMPILE TIME: the ONE resolver of the cards website_review puts on screen.

    Every compile-side reader (the render, the cache-hit accounting) comes through
    here, so they cannot disagree with each other or with the script. In order:

    1. The snapshot's ``title_card`` block is not enabled -> no cards. The reservation
       is an upper bound on what the script left room for, never a reason to render a
       card the run no longer asks for.
    2. The inherited script output RECORDS a reservation -> render exactly it. That
       includes a recorded "no cards" (both ``0.0``): the script kept those seconds
       for narration, so rendering the configured cards would overrun the order.
    3. The record is ABSENT (key missing or ``None``: a script produced before the
       record existed) -> the cards AS CONFIGURED, uncapped. Such a script budgeted
       ``order - configured cards``; capping here is how a 10s order scripted with a
       4.0s narration budget would have shipped a ~6s film.

    The cap is never recomputed at compile time. Absent and "none reserved" are told
    apart by the key's presence, never by truthiness.
    """
    configured = plan_title_cards(snapshot.get("title_card"), ordered_seconds=None)
    if configured is None:
        return None
    recorded = (script_output or {}).get(RESERVED_TITLE_CARDS_KEY)
    if recorded is None:
        return configured
    return ReservedTitleCards.model_validate(recorded).as_plan()


def fixed_non_narration_seconds(snapshot: Mapping[str, object]) -> float:
    """Seconds the compile adds that no narration line can fill (the title cards).

    :func:`reserve_title_cards`' total — the same plan the script stage persists as
    its reservation, so the budget and the record are one computation.
    """
    return reserve_title_cards(snapshot).total_seconds


def narration_budget_seconds(snapshot: Mapping[str, object]) -> float:
    """What the script may spend on spoken lines: order minus the fixed card time."""
    return max(
        website_review_ordered_seconds(snapshot) - fixed_non_narration_seconds(snapshot), 0.0
    )


def minimum_narration_seconds() -> float:
    """The narration the blueprint must produce, whatever the order says.

    :data:`MIN_NARRATED_LINES` lines at the compile's narrated-beat floor. Independent
    of the snapshot on purpose: this half of the floor does NOT move with the
    ``title_card`` block, which is exactly the hole the old floor had — with cards off
    it collapsed to a single 1.2s line, so a bare 3s order was accepted while the same
    order WITH cards was correctly refused.
    """
    return MIN_NARRATED_LINES * MIN_NARRATED_BEAT_SECONDS


def physical_minimum_order_seconds(snapshot: Mapping[str, object]) -> float:
    """The shortest order the compile can actually deliver: narration + any cards.

    2.4s with cards off; with cards on it is 2.4s plus whatever the cards are capped to
    at THIS order (at most 20% of it). Kept as its own function because
    it is the only half of the floor that MOVES with the snapshot — the product floor
    is a constant — and because it is what proves 10s is not fighting the compiler.
    """
    return fixed_non_narration_seconds(snapshot) + minimum_narration_seconds()


def minimum_order_seconds(snapshot: Mapping[str, object]) -> float:
    """The floor the door enforces: the product's 10s, or physics if physics is higher.

    ``max`` rather than a bare constant so an exotic ``title_card`` block (cards long
    enough to exceed 10s on their own) cannot be accepted into a run the compile must
    then overrun. With every shipping config the product floor governs, which is the
    point: the answer is 10s whether or not title cards are on.
    """
    return max(MINIMUM_ORDER_SECONDS, physical_minimum_order_seconds(snapshot))


def _minimum_breakdown(cards: float) -> str:
    """The parenthetical explaining where :func:`minimum_order_seconds` comes from."""
    narration = (
        f"{MIN_NARRATED_LINES} narrated lines of at least "
        f"{MIN_NARRATED_BEAT_SECONDS:g}s each — the critique and the redesign reveal"
        if MIN_NARRATED_LINES > 1
        else f"one narrated line of at least {MIN_NARRATED_BEAT_SECONDS:g}s"
    )
    if cards > 0:
        return f"{cards:g}s of title cards + {narration}"
    return narration


def validate_website_review_order(snapshot: Mapping[str, object]) -> None:
    """Raise :class:`OrderBelowFloorError` when the order is below the floor.

    Only meaningful when a video is requested (``brief.generate_video``, default on).
    The raised error carries a typed :class:`OrderRefusal`; both doors turn it into a
    **400** whose body names both the floor and the ordered value.
    """
    if not bool(_section(snapshot, "brief").get("generate_video", True)):
        return
    ordered = website_review_ordered_seconds(snapshot)
    minimum = minimum_order_seconds(snapshot)
    if ordered >= minimum:
        return
    cards = fixed_non_narration_seconds(snapshot)
    if minimum > MINIMUM_ORDER_SECONDS:
        # Physics governs (cards long enough to outrun the product floor): explain
        # where the number came from, and offer the remedy the operator has not
        # already taken — never "disable title_card" when cards are already off.
        remedy = (
            f"Order at least {minimum:g}s or disable title_card."
            if cards > 0
            else f"Order at least {minimum:g}s."
        )
        message = (
            f"website_review ordered {ordered:g}s, but the shortest possible review is "
            f"{minimum:g}s ({_minimum_breakdown(cards)}). {remedy}"
        )
    else:
        # The product floor governs — the ordinary case, and the same number whether
        # or not title cards are on, so the message must not mention them.
        message = (
            f"website_review ordered {ordered:g}s, but the shortest possible review is "
            f"{minimum:g}s — the minimum order, with or without title cards. "
            f"Order at least {minimum:g}s."
        )
    raise OrderBelowFloorError(
        OrderRefusal(ordered_seconds=ordered, minimum_seconds=minimum, message=message)
    )
