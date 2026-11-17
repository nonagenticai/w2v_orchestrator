"""The motion templates a promo motion brief may author, as the recorder draws them.

Everything here is derived from video_production_service's
``src/services/recording/templates/`` as of video_production_service#188 (the
"film kit": ``two_beat``, ``levels``, ``checklist``, ``handoff`` and the opt-in
film layouts of ``two_up`` / ``timeline`` / ``animated_title_card``; each
template's header comment is the authoritative schema, read 2026-10-06). It is
a pure description — no I/O, no model — of which ``window.PARAMS`` keys each
template READS, what its ``marks`` mean, and whether its foreground ever comes
to rest, so the brief stage can validate, time and budget a card before
anything is recorded.

Facts that drive the shape of this module:

* **One beat list per template.** Every film template reads ``marks``: absolute
  beat offsets in ms, in a fixed per-template order (:meth:`TemplateParams.beats`).
  The recorder pins beat 0 to 0 and leads it in early, so the first frame is
  never empty. The brief places each later beat where its phrase is SPOKEN
  (``motion_brief.narration_marks``) — never by spreading entrances over the
  hold.
* **Continuous foreground motion.** The film templates keep a foreground element
  moving for the whole clip (a dial arc, a marker stepping the ladder, a
  travelling progress line, packets on a route — measured on #188: 0.0-0.5 s
  longest foreground-static stretch with the ambient layer hidden, at 4 s and at
  12 s). :attr:`TemplateParams.continuous_motion` records that; the legacy
  ``section_divider`` (kept only as a last-resort fallback) settles after its
  entrance and is still budgeted from its entrance events.
* **No built-in copy, except the title card.** A film template draws nothing for
  an omitted field. ``animated_title_card`` still carries demo defaults for
  ``eyebrow`` / ``title`` / ``subtitle`` / ``brand_mark``, so those four (and
  ``chapter``) are always sent, blank when unused.
* **Icons are names.** ``icon`` is looked up in the recorder's ``_icons.js``; an
  unknown name draws nothing. A name outside :data:`ICON_NAMES` is DROPPED here,
  never sent.

Not authorable, by design: ``pipeline_story`` (a fixed 16.8 s film about a
Planka board and an n8n engine whose copy no param removes) and
``kinetic_typography`` (types the sentence out, then never moves again).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: No shot may sit with its foreground at rest for longer than this.
MAX_STATIC_SECONDS = 2.5

#: Glyph names of video_production_service
#: ``src/services/recording/templates/_icons.js`` (``window.ICONS``) at #188 —
#: vendored, because an unknown name silently draws no glyph. The last nine were
#: added by #188 for the film templates.
ICON_NAMES = frozenset(
    {
        "check", "arrow_right", "arrow_up_right", "bolt", "star", "spark", "target",
        "rocket", "layers", "grid", "chart", "trend_up", "gauge", "clock", "calendar",
        "shield", "lock", "globe", "users", "chat", "doc", "code", "play", "heart", "gear",
        "cross", "bot", "user", "headset", "chat_dots", "braces", "eye", "mail", "route",
    }
)  # fmt: skip


def known_icon(name: object) -> str:
    """``name`` when the recorder has that glyph, else ``""`` (dropped, not sent)."""
    text = str(name or "").strip()
    return text if text in ICON_NAMES else ""


#: ``(start_ms, end_ms)`` of one foreground element's entrance.
type Event = tuple[int, int]


class _Part(BaseModel):
    """A nested piece of a card (a panel, a step, a node)."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, populate_by_name=True)

    def sent(self) -> dict[str, object]:
        """The keys to send: only what is set."""
        return {key: value for key, value in self.model_dump(by_alias=True).items() if value}


class TemplateParams(BaseModel):
    """Content params of one authorable template.

    ``extra="forbid"`` is the rule that a brief may not carry a param the
    template ignores: an unknown key is a validation error, not a silent no-op.
    The ``max_length`` bounds are this module's readability limits; the list
    bounds are the templates' own (``slice(0, 6)``).
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, populate_by_name=True)

    template_id: ClassVar[str]
    #: A foreground element keeps moving for the whole clip (module docstring).
    continuous_motion: ClassVar[bool] = True
    #: The shared ambient layer drifts for the whole clip (never counted as motion).
    ambient_drift: ClassVar[bool] = True

    def template_params(self) -> dict[str, object]:
        """The ``window.PARAMS`` content keys to send."""
        raise NotImplementedError

    def screen_text(self) -> list[str]:
        """Every string this card draws, in reading order."""
        raise NotImplementedError

    def beats(self) -> list[str | None]:
        """The template's ``marks`` slots, in mark order.

        Each entry is the on-screen phrase that beat brings in, or ``None`` for
        a beat that carries no phrase of its own (it opens with the card).
        """
        raise NotImplementedError

    def settle_events(self, marks: list[int]) -> list[Event]:
        """Entrances of a template that SETTLES (``continuous_motion`` False)."""
        return []

    def phrases(self) -> list[tuple[str, Literal["headline", "label", "negated"]]]:
        """Every phrase the card shows, with its slot (see ``headline.phrase_problem``).

        A figure (``levels``' stat) is not a phrase — it is checked as a figure.
        """
        return [(text, "headline") for text in self.screen_text()]

    @property
    def graphic(self) -> str:
        """The drawn element this card carries besides type (``""`` = type only).

        Frame QA of run 01a11511: five of eight scenes were a ring or plain text.
        A card a plain line may get must draw something that SAYS something —
        a ladder, rows with ticks, nodes on a route, a track, panels, an icon.
        """
        return ""


def _shown(*values: str) -> list[str]:
    return [value for value in values if value]


def _set(**values: object) -> dict[str, object]:
    return {key: value for key, value in values.items() if value}


# ── the two-beat's text budget, measured in the recorder ─────────────────────
#
# Render 01a1179d shipped a card whose second line ended "…an audit trail you
# can": the stage cut the text to a character count, and nothing knew what the
# template can actually show. Measured in Chromium at 1920x1080 on
# video_production_service two_beat (Inter 700, ``.copy`` 58u = 1114 px wide,
# 6.8u = 130.56 px type, ``MotionLib.fit`` shrinks to 0.6 = 78.34 px and then
# gives up): at the floor the title holds 3 lines, the emphasis 2. The glyph
# widths below (em) reproduce every measured line count (77/56 chars → 3/2
# lines; 95/69 → 4/3).
_COPY_PX = 1114.0
_FLOOR_PX = 130.56 * 0.6
TWO_BEAT_TITLE_LINES = 3
TWO_BEAT_EMPHASIS_LINES = 2


def _glyph_em(char: str) -> float:
    if char == " ":
        return 0.26
    if char in "ijl.,'!:;|":
        return 0.28
    if char in "ftr-":
        return 0.38
    if char in "mwMW":
        return 0.86
    if char.isupper():
        return 0.68
    if char.isdigit():
        return 0.6
    return 0.56


#: Safety margin on the line width: the glyph table is an average, and a
#: capital- or W/M-heavy line runs wider than it predicts (audit of #1086: 3
#: of 60 boundary strings predicted to fit overflowed). With 9% off the width,
#: no string of the measured set that is predicted to fit overflows.
FIT_MARGIN = 0.09


def wrapped_lines(
    text: str, *, width_px: float = _COPY_PX * (1 - FIT_MARGIN), font_px: float = _FLOOR_PX
) -> int:
    """Lines ``text`` wraps to in the two-beat at the fit floor (greedy, by word)."""
    lines, current = 0, ""
    for word in text.split():
        candidate = f"{current} {word}" if current else word
        if current and sum(map(_glyph_em, candidate)) * font_px > width_px:
            lines, current = lines + 1, word
        else:
            current = candidate
    return lines + (1 if current else 0)


class TwoBeatParams(TemplateParams):
    """``two_beat`` — a headline, then an emphasised second beat, beside the dial.

    Marks ``[title, emphasis]``. Also the safe fallback: a key phrase alone.
    ``icon`` is a glyph drawn at the dial's centre (video_production_service
    two_beat ``icon`` param); without one the card is the dial and type only,
    which a plain line never gets (:attr:`graphic`).
    """

    template_id: ClassVar[str] = "two_beat"

    # (the real budget is in lines: :meth:`overflow`)
    title: str = Field(min_length=1, max_length=200)
    emphasis: str = Field(default="", max_length=200)
    kicker: str = Field(default="", max_length=60)
    icon: str = Field(default="", max_length=20)

    _icon = field_validator("icon", mode="before")(known_icon)

    def template_params(self) -> dict[str, object]:
        return _set(kicker=self.kicker, title=self.title, emphasis=self.emphasis, icon=self.icon)

    def overflow(self) -> str | None:
        """Why the text does not fit the template at its smallest type, or ``None``."""
        title = wrapped_lines(self.title)
        if title > TWO_BEAT_TITLE_LINES:
            return f"title wraps to {title} lines (the two-beat holds {TWO_BEAT_TITLE_LINES})"
        emphasis = wrapped_lines(self.emphasis) if self.emphasis else 0
        if emphasis > TWO_BEAT_EMPHASIS_LINES:
            return (
                f"second line wraps to {emphasis} lines (the two-beat holds "
                f"{TWO_BEAT_EMPHASIS_LINES})"
            )
        return None

    @property
    def graphic(self) -> str:
        return f"icon {self.icon}" if self.icon else ""

    def screen_text(self) -> list[str]:
        return _shown(self.kicker, self.title, self.emphasis)

    def beats(self) -> list[str | None]:
        return [self.title, self.emphasis or None]


class LevelsParams(TemplateParams):
    """``levels`` — a hero stat beside a ladder that fills from its lowest rung
    (``from_label``) to its highest (``to_label``).

    Marks ``[title + ladder outline, stat, from_label, to_label]``. ``levels`` is
    the number of rungs drawn (2..8), a drawing parameter, not a claim. The end
    labels are optional: a line that names no ends ("graduated autonomy levels
    for over thirty capabilities") gets an unlabelled ladder — the template
    draws an empty label as nothing — never ends the line does not say.
    """

    template_id: ClassVar[str] = "levels"

    value: str = Field(min_length=1, max_length=12)
    suffix: str = Field(default="", max_length=4)
    label: str = Field(default="", max_length=60)
    from_label: str = Field(default="", max_length=40)
    to_label: str = Field(default="", max_length=40)
    title: str = Field(default="", max_length=80)
    kicker: str = Field(default="", max_length=60)
    levels: int = Field(default=5, ge=2, le=8)

    @property
    def graphic(self) -> str:
        return "ladder"

    def phrases(self) -> list[tuple[str, Literal["headline", "label", "negated"]]]:
        heads = _shown(self.kicker, self.title, self.label)
        ends = _shown(self.from_label, self.to_label)
        return [*((text, "headline") for text in heads), *((text, "label") for text in ends)]

    def template_params(self) -> dict[str, object]:
        return {
            **_set(kicker=self.kicker, title=self.title),
            "value": self.value,
            **_set(suffix=self.suffix, label=self.label),
            "levels": self.levels,
            **_set(from_label=self.from_label, to_label=self.to_label),
        }

    def screen_text(self) -> list[str]:
        stat = f"{self.value}{self.suffix}"
        return _shown(self.kicker, self.title, stat, self.label, self.from_label, self.to_label)

    def beats(self) -> list[str | None]:
        return [self.title or None, self.value, self.from_label or None, self.to_label or None]


class Panel(_Part):
    title: str = Field(min_length=1, max_length=60)
    body: str = Field(default="", max_length=120)
    icon: str = Field(default="", max_length=20)
    state: Literal["", "chosen", "rejected"] = ""

    _icon = field_validator("icon", mode="before")(known_icon)


class TwoUpParams(TemplateParams):
    """``two_up`` with ``layout: "panels"`` — two panels and their verdicts.

    Marks ``[kicker, title, left verdict, right verdict]``: the header and both
    undecided panels are on screen from the first frame; a side's mark delivers
    its ``state`` (``chosen`` → accent + tick, ``rejected`` → cross + strike).
    """

    template_id: ClassVar[str] = "two_up"

    left: Panel
    right: Panel
    title: str = Field(default="", max_length=80)
    kicker: str = Field(default="", max_length=60)

    def template_params(self) -> dict[str, object]:
        return {
            "layout": "panels",
            **_set(kicker=self.kicker, title=self.title),
            "two_up": {"left": self.left.sent(), "right": self.right.sent()},
        }

    def screen_text(self) -> list[str]:
        sides = _shown(self.left.title, self.left.body, self.right.title, self.right.body)
        return [*_shown(self.kicker, self.title), *sides]

    def beats(self) -> list[str | None]:
        return [None, None, self.left.title, self.right.title]

    @property
    def graphic(self) -> str:
        return "panels"

    def phrases(self) -> list[tuple[str, Literal["headline", "label", "negated"]]]:
        out: list[tuple[str, Literal["headline", "label", "negated"]]] = [
            (text, "headline") for text in _shown(self.kicker, self.title)
        ]
        for panel in (self.left, self.right):
            slot: Literal["headline", "label", "negated"] = (
                "negated" if panel.state == "rejected" else "headline"
            )
            out.extend((text, slot) for text in _shown(panel.title, panel.body))
        return out


class TrackStep(_Part):
    title: str = Field(min_length=1, max_length=60)
    body: str = Field(default="", max_length=100)
    icon: str = Field(default="", max_length=20)

    _icon = field_validator("icon", mode="before")(known_icon)


class TimelineParams(TemplateParams):
    """``timeline`` with ``layout: "track"`` — a horizontal track of 2-6 steps.

    Marks ``[kicker, title, step 1, step 2, …]``: the rail and every dimmed node
    are on screen from the first frame; the progress arrow ARRIVES at step i on
    its mark. A step without an icon shows its number.
    """

    template_id: ClassVar[str] = "timeline"

    steps: list[TrackStep] = Field(min_length=2, max_length=6)
    title: str = Field(default="", max_length=80)
    kicker: str = Field(default="", max_length=60)

    def template_params(self) -> dict[str, object]:
        return {
            "layout": "track",
            **_set(kicker=self.kicker, title=self.title),
            "steps": [step.sent() for step in self.steps],
        }

    def screen_text(self) -> list[str]:
        steps = [text for step in self.steps for text in _shown(step.title, step.body)]
        return [*_shown(self.kicker, self.title), *steps]

    def beats(self) -> list[str | None]:
        return [None, None, *(step.title for step in self.steps)]

    @property
    def graphic(self) -> str:
        return "track"


class ChecklistItem(_Part):
    label: str = Field(min_length=1, max_length=60)
    state: Literal["yes", "no"] = "yes"

    def sent(self) -> dict[str, object]:
        return {"label": self.label, "state": self.state}


class ChecklistParams(TemplateParams):
    """``checklist`` — 2-6 rows judged one after another (a drawn tick or cross).

    Marks ``[every row, undecided; verdict 1; verdict 2; …]``.
    """

    template_id: ClassVar[str] = "checklist"

    items: list[ChecklistItem] = Field(min_length=2, max_length=6)
    title: str = Field(default="", max_length=80)
    kicker: str = Field(default="", max_length=60)

    def template_params(self) -> dict[str, object]:
        return {
            **_set(kicker=self.kicker, title=self.title),
            "items": [item.sent() for item in self.items],
        }

    def screen_text(self) -> list[str]:
        return [*_shown(self.kicker, self.title), *(item.label for item in self.items)]

    def beats(self) -> list[str | None]:
        return [None, *(item.label for item in self.items)]

    @property
    def graphic(self) -> str:
        return "ticks"

    def phrases(self) -> list[tuple[str, Literal["headline", "label", "negated"]]]:
        heads: list[tuple[str, Literal["headline", "label", "negated"]]] = [
            (text, "headline") for text in _shown(self.kicker, self.title)
        ]
        return [
            *heads,
            *((item.label, "negated" if item.state == "no" else "headline") for item in self.items),
        ]


class HandoffNode(_Part):
    label: str = Field(min_length=1, max_length=40)
    icon: str = Field(default="", max_length=20)

    _icon = field_validator("icon", mode="before")(known_icon)


class HandoffRoute(_Part):
    label: str = Field(min_length=1, max_length=40)


class HandoffParams(TemplateParams):
    """``handoff`` — a source node, a labelled route with travelling packets, a
    destination node that takes over.

    Marks ``[title + both nodes + route, route label + packets, destination]``.
    ``from`` is a Python keyword, hence the alias.
    """

    template_id: ClassVar[str] = "handoff"

    source: HandoffNode = Field(alias="from")
    route: HandoffRoute
    to: HandoffNode
    title: str = Field(default="", max_length=80)
    kicker: str = Field(default="", max_length=60)

    def template_params(self) -> dict[str, object]:
        return {
            **_set(kicker=self.kicker, title=self.title),
            "from": self.source.sent(),
            "route": self.route.sent(),
            "to": self.to.sent(),
        }

    def screen_text(self) -> list[str]:
        nodes = [self.source.label, self.route.label, self.to.label]
        return [*_shown(self.kicker, self.title), *nodes]

    def beats(self) -> list[str | None]:
        return [None, self.route.label, self.to.label]

    @property
    def graphic(self) -> str:
        return "route"

    def phrases(self) -> list[tuple[str, Literal["headline", "label", "negated"]]]:
        heads = _shown(self.kicker, self.title, self.source.label)
        return [
            *((text, "headline") for text in heads),
            (self.route.label, "label"),
            (self.to.label, "headline"),
        ]


class TitleCardParams(TemplateParams):
    """``animated_title_card`` with ``style: "film"`` — the lockup inside the dial.

    Marks ``[chapter, eyebrow, title, subtitle]`` (positional; the first row
    that has content opens with the card). ``chapter`` is always blank and
    ``brand_mark`` always equals the title, which the template then does not
    draw — its own default for it is another company's name, and a closing
    beat's brand name must be drawn once, as the headline.
    """

    template_id: ClassVar[str] = "animated_title_card"

    title: str = Field(min_length=1, max_length=80)
    subtitle: str = Field(default="", max_length=120)
    eyebrow: str = Field(default="", max_length=60)

    def template_params(self) -> dict[str, object]:
        return {
            "style": "film",
            "eyebrow": self.eyebrow,
            "chapter": "",
            "title": self.title,
            "subtitle": self.subtitle,
            "brand_mark": self.title,
        }

    def screen_text(self) -> list[str]:
        return _shown(self.eyebrow, self.title, self.subtitle)

    def beats(self) -> list[str | None]:
        rows: list[str | None] = [None, self.eyebrow or None, self.title, self.subtitle or None]
        first = next(index for index, row in enumerate(rows) if row)
        rows[first] = None  # the first row with content is pinned to the card's start
        return rows


class SectionDividerParams(TemplateParams):
    """``section_divider`` (the pre-film template) — LAST-RESORT fallback only.

    A headline and a rule that draws in; it then SETTLES (only the ambient glow
    drifts), so its static stretch is budgeted from its entrances. Marks
    ``[kicker?, title, rule]`` (base 80 ms; 720 / 720 / 640 ms).
    """

    template_id: ClassVar[str] = "section_divider"
    continuous_motion: ClassVar[bool] = False

    title: str = Field(min_length=1, max_length=80)
    kicker: str = Field(default="", max_length=60)

    def template_params(self) -> dict[str, object]:
        return {"kicker": self.kicker, "title": self.title}

    def screen_text(self) -> list[str]:
        return _shown(self.kicker, self.title)

    def beats(self) -> list[str | None]:
        return [None, *([self.title] if self.kicker else []), None]

    def settle_events(self, marks: list[int]) -> list[Event]:
        durations = [*([720] if self.kicker else []), 720, 640]
        return sorted(
            (mark + 80, mark + 80 + dur) for mark, dur in zip(marks, durations, strict=True)
        )


#: template id → its param model. The brief's whole vocabulary.
PARAM_MODELS: dict[str, type[TemplateParams]] = {
    model.template_id: model
    for model in (
        TwoBeatParams,
        LevelsParams,
        TwoUpParams,
        TimelineParams,
        ChecklistParams,
        HandoffParams,
        TitleCardParams,
        SectionDividerParams,
    )
}

#: Templates that exist only on a recorder with video_production_service#188.
FILM_KIT_TEMPLATES = frozenset(PARAM_MODELS) - {"section_divider"}

#: Registered recorder templates a brief must never author, and why.
UNAUTHORABLE: dict[str, str] = {
    "kinetic_typography": "types the narration out word by word and never moves again",
    "pipeline_story": (
        "a fixed 16.8s story whose on-screen copy (Planka board, n8n engine, module "
        "names, captions) no param removes"
    ),
}


def foreground_events(params: TemplateParams, marks: list[int]) -> list[Event]:
    """When each element of a SETTLING card enters (``[]`` for a film template)."""
    return [] if params.continuous_motion else params.settle_events(marks)


def longest_rest_ms(events: list[Event], *, clip_ms: int, shot_ms: int) -> int:
    """Longest stretch of the shot in which no foreground element is entering.

    The clip plays from the shot's first frame. Past the clip's end the compiler
    holds the last frame, so nothing after ``clip_ms`` moves; entrances the clip
    never reaches do not happen.
    """
    playing = min(clip_ms, shot_ms)
    cursor = longest = 0
    for start, end in sorted(events):
        if start >= playing:
            break
        longest = max(longest, start - cursor)
        cursor = max(cursor, min(end, playing))
    return max(longest, shot_ms - cursor)


def card_static_seconds(params: TemplateParams, marks: list[int], *, shot_ms: int) -> float:
    """Longest foreground-static stretch the plan predicts for one card.

    A film template moves for its whole clip (documented behaviour, measured on
    video_production_service#188), and the clip is recorded for the whole shot:
    ``0.0``. A settling template rests from its last entrance to the cut.
    """
    if params.continuous_motion:
        return 0.0
    events = foreground_events(params, marks)
    return longest_rest_ms(events, clip_ms=shot_ms, shot_ms=shot_ms) / 1000


@dataclass(frozen=True)
class PlannedShot:
    """One shot of the film as the PLAN knows it.

    ``kind``:

    * ``motion_card`` — a recorded template: ``continuous`` (a film template,
      moving for its whole clip) or settling after its ``events``;
    * ``clip`` — footage / generated video: its inner motion is not knowable
      from the plan, so only a freeze-pad past ``clip_seconds`` is counted;
    * ``still`` — an image: static for the whole shot unless ``ken_burns`` is
      expected, in which case the plan EXPECTS a pan for the whole shot (run
      01a11082 shipped a still frozen for 8 s because its Ken Burns render timed
      out — only the render can confirm it).
    """

    line_id: str
    seconds: float
    kind: Literal["motion_card", "clip", "still"]
    clip_seconds: float | None = None
    events: tuple[Event, ...] = ()
    continuous: bool = False
    ken_burns: bool = False
    end_card: bool = False


def static_seconds(shot: PlannedShot) -> float:
    """Longest motionless stretch the plan predicts inside ``shot``."""
    shot_ms = round(shot.seconds * 1000)
    if shot.kind == "still":
        return 0.0 if shot.ken_burns else shot.seconds
    clip_ms = shot_ms if shot.clip_seconds is None else round(shot.clip_seconds * 1000)
    if shot.kind == "clip" or shot.continuous:
        return max(0, shot_ms - clip_ms) / 1000
    return longest_rest_ms(list(shot.events), clip_ms=clip_ms, shot_ms=shot_ms) / 1000


def film_static_seconds(shots: list[PlannedShot]) -> tuple[dict[str, float], float]:
    """``({line_id: static seconds}, the film's longest)`` — the end card excepted.

    A cut is a change of picture, so a stretch never spans two shots.
    """
    per_line = {shot.line_id: round(static_seconds(shot), 3) for shot in shots}
    counted = [per_line[shot.line_id] for shot in shots if not shot.end_card]
    return per_line, max(counted, default=0.0)
