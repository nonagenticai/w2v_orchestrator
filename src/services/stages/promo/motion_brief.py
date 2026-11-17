"""promo_motion_brief — one designed motion card per narration line.

Why (frame QA of run 01a11082, 2026-10-06): a promo's motion was the narration
sentence typed out word by word on one gradient, three lines running, each card
then frozen for 3-5.6 s. The request can only name ONE template for the whole
run and cannot know the script's line ids, so the designing has to happen here,
after the script exists. The first cut of this stage authored the pre-film
templates and its preview failed review as "text cards on navy"; it now authors
the recorder's film kit (video_production_service#188 — :mod:`motion_templates`).

OPT-IN on ``direction.motion == "per_line"`` (a brief key, lifted to the config
snapshot with the rest of ``direction``); otherwise the stage skips and the run
is unchanged. It also skips when the request authored its own
``motion_graphics.items`` — those win in ``motion_graphics_gen`` anyway.

For every script line it produces one item from the recorder's template
vocabulary. A model proposes; nothing it says is trusted. Each proposal is
checked, deterministically:

1. **the line's own words only** — every word and number on the card is a token
   of that line (:func:`foreign_tokens`), a figure keeps the suffix the line
   gives it, and no symbol appears that the line does not have. One exemption:
   the CLOSING beat (the call-to-action line, last in the script) may also show
   the brand name;
2. **not the sentence typed out** — a type card (``two_beat``, a title card) or
   any card with a headline carrying 80% or more of the line's tokens is
   rejected, in whatever order it shows them (:func:`sentence_coverage`); a
   diagram with no headline may lay more of the line out across its parts, but
   no single part may carry 80% (:func:`typed_out`). A line of eight words or
   fewer (a tagline) and the call-to-action line are a phrase already, and may
   be shown whole;
2b. **every phrase reads as one** — a run of the line's words as written, cut
   at a phrase boundary, never opened by a stranded word or ended by a
   dangling one (:mod:`headline`, run 01a11511: "Merqi runs motion", "Ever
   requesting contact", "Builds multimedia assets tailored");
2c. **every card carries a graphic** — a ladder, ticks, a route, a track,
   panels, or (``two_beat``) an icon chosen from the line's words; type on a
   ring is not a card. The closing beat is the brand lockup on the dial — the
   film's end card — and is the one exemption;
3. **no template on more than two lines of the film**, and an unused template
   is preferred (:func:`choose_items`);
4. **no foreground at rest for more than 2.5 s** — the film templates move for
   their whole clip and the clip is recorded for the whole shot; a card of a
   template that settles is swapped, or reported.

Timing is the narration's, never a delay: the card's first beat is at 0 and each
later beat lands where its phrase is SPOKEN (:func:`narration_marks`). Every
card also carries ``film_t_ms`` — where its shot starts in the film — so the
shared ambient layer is one continuous clock across cuts.

The closing beat is not designed, it is fixed: the brand name over the call to
action on the film-style title card. It IS the promo's end card: when the promo
would get its default branded end card (``fill.end_card_config``, which says the
same two lines), the closing beat's shot absorbs the seconds the fill plan gave
that card and the render splices no separate card — one recording, no cut, the
film's planned total unchanged (``ClosingBeat.end_card_seconds``).

A rejected proposal is replaced by a rule-built card. When the model is
unavailable every line is authored by :func:`deterministic_drafts` — a pure
function, which is also what makes the stage testable without a model.

The output IS the record: per item who authored it, what was rejected and why,
what was repaired, and the planned static stretch per line.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Literal
from uuid import uuid7

from pydantic import Field, ValidationError
from w2v_common.stages import stage_call

from src.services.llm import WMClient
from src.services.stages.base import (
    SlotUse,
    StageHandler,
    StageInput,
    StageOutput,
    StageOutputWithNotes,
    StagePreconditionError,
    StageSkippedError,
)
from src.services.stages.llm_budget import (
    LLM_RETRY_ON,
    MEDIUM_ANSWER_TOKENS,
    downshift_reasoning,
    llm_budget_seconds,
    llm_client_timeout_seconds,
    llm_retries,
    output_token_budget,
)
from src.services.stages.promo.fill import CTA_MIN_SECONDS, end_card_config, promo_fill
from src.services.stages.promo.headline import (
    DETERMINERS,
    FUNCTION_WORDS,
    SUBJECT_PRONOUNS,
    Role,
    clause_end,
    clause_starts,
    clean_spans,
    condition_card_problem,
    element_order_problem,
    headline_start_problem,
    is_negator,
    locate,
    modifier_list,
    predicate,
    problems,
    sources_for,
    starts_with_verb,
    strip_determiner,
    strip_stranded,
    subject,
    tokens,
    trim_trailing,
    with_figures,
)
from src.services.stages.promo.motion_templates import (
    ICON_NAMES,
    MAX_STATIC_SECONDS,
    PARAM_MODELS,
    TWO_BEAT_EMPHASIS_LINES,
    TWO_BEAT_TITLE_LINES,
    UNAUTHORABLE,
    HandoffParams,
    PlannedShot,
    TemplateParams,
    TimelineParams,
    TwoBeatParams,
    TwoUpParams,
    card_static_seconds,
    film_static_seconds,
    foreground_events,
    wrapped_lines,
)
from src.services.stages.shared.motion_slides import motion_per_line_requested
from src.services.stages.upstream import guarded

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

log = logging.getLogger(__name__)

STAGE_NAME = "promo_motion_brief"
#: A card showing this share of the line's tokens IS the sentence.
SENTENCE_COVERAGE_LIMIT = 0.8
#: A line this short is a tagline; showing it whole (in beats) is a headline,
#: not "the sentence typed out". ("Go-to-market that runs itself, under your
#: control." is seven words.)
SHORT_LINE_WORDS = 8
#: No template carries more than this many lines of one film.
MAX_TEMPLATE_USES = 2
#: The closing beat's template (its one use is reserved before any line is settled).
CLOSING_TEMPLATE = "animated_title_card"

_KEY_PHRASE_WORDS = 6
_RUN_WORDS = 5
_SIDE_WORDS = 4
_HEADLINE_TAIL_WORDS = 3

#: Templates whose design is the words themselves (rule 2 counts the whole card).
TYPE_CARDS = frozenset({"two_beat", CLOSING_TEMPLATE, "section_divider"})
#: Shown when no word of the line names a glyph: a neutral accent mark.
DEFAULT_ICON = "spark"
#: Symbols that carry meaning. One of these on a card must be in the line; every
#: other non-alphanumeric character (arrows, dashes, dots, quotes) is a
#: separator glyph a card may draw.
_MEANINGFUL_SYMBOLS = frozenset("%$€£+=<>&@#")
_CLAUSE = re.compile(r"\s*[,;:]\s+|\s+[\u2014\u2013-]\s+")
_NUMBER = re.compile(r"(?<![\w.])(\d[\d.,]*\d|\d)([+%]|[a-zA-Z]{1,2}\b)?")
_CONTRAST = re.compile(r"^(.+?),\s+(?:not|rather than|instead of)\s+(.+)$", re.IGNORECASE)
_SEQUENCE = re.compile(r"^(.+?)\s+(?:before|then)\s+(.+)$", re.IGNORECASE)
#: "…, avoiding X or Y" — a trailing clause that says what is NOT done.
_NEGATIVE = re.compile(
    r"^(.+?),\s+(?:avoiding|without|excluding|rejecting|never using)\s+(.+)$", re.IGNORECASE
)
#: "<subject> is/are [adverb] routed to <destination>".
_ROUTED = re.compile(
    r"^(.+?)\s+(?:is|are|gets|get)\s+((?:\w+\s+)?(?:routed|sent|handed|escalated|passed|"
    r"transferred|forwarded|delivered|assigned))\s+(?:over\s+|on\s+)?(?:to|into)\s+(.+)$",
    re.IGNORECASE,
)
_ALTERNATIVE = re.compile(r"\s*,\s*(?:(?:or|and|nor)\s+)?|\s+(?:or|and|nor)\s+", re.IGNORECASE)
_FIGURE = re.compile(r"\d[\d.,]*[+%]?")
_DOMAIN = re.compile(r"^[\w-]+(?:\.[\w-]+)+$")
_ARTICLES = frozenset({"the", "a", "an"})
_AUXILIARIES = frozenset(
    {"is", "are", "was", "were", "be", "been", "can", "will", "must", "should", "do", "does"}
)
#: After one of these the next word is a verb, not part of a noun phrase.
_MODALS = frozenset({"can", "will", "must", "should", "may", "could", "would", "to"})
_CONNECTORS = _AUXILIARIES | frozenset(
    {
        "by", "with", "from", "of", "in", "on", "at", "for", "through", "via", "to", "into",
        "and", "or", "that", "which", "than", "as", "but", "over",
    }
)  # fmt: skip
_STOPWORDS = (
    _CONNECTORS
    | _ARTICLES
    | frozenset({"it", "its", "this", "these", "those", "your", "our", "you", "we", "they", "only"})
)

#: A word of a label → the recorder glyph that stands for it (first match in the
#: label wins). Every value is a name in :data:`ICON_NAMES`; a label with no
#: match carries no icon.
_ICON_WORDS: dict[str, str] = {
    "code": "code", "software": "code", "rules": "braces", "logic": "braces",
    "language": "chat_dots", "conversation": "chat_dots", "conversations": "chat_dots",
    "chat": "chat_dots", "messages": "chat_dots", "models": "bot", "model": "bot",
    "bot": "bot", "ai": "bot", "automation": "bot", "automated": "bot", "automatic": "bot",
    "attention": "eye", "visibility": "eye", "awareness": "eye", "review": "eye",
    "contact": "mail", "email": "mail", "mail": "mail", "inbox": "mail", "outreach": "mail",
    "human": "user", "person": "user", "reviewer": "user", "owner": "user", "manager": "user",
    "agent": "headset", "support": "headset", "team": "users", "teams": "users",
    "people": "users", "customers": "users", "secure": "shield", "security": "shield",
    "safe": "shield", "private": "lock", "locked": "lock", "public": "globe",
    "global": "globe", "time": "clock", "schedule": "calendar", "fast": "bolt",
    "speed": "bolt", "growth": "trend_up", "data": "chart", "metrics": "chart",
    "docs": "doc", "document": "doc", "documents": "doc", "settings": "gear",
    "launch": "rocket", "goal": "target", "tests": "check", "approved": "check",
    # run 01a11511's plain lines
    "you": "user", "multimedia": "layers", "assets": "layers", "context": "target",
    "target": "target", "targets": "target", "marketing": "chart", "surfaces": "grid",
    "roi": "trend_up", "fix": "target", "decisions": "check", "sources": "globe",
    "files": "doc", "capabilities": "grid", "levels": "chart", "autonomy": "gauge",
    "market": "rocket", "say": "chat",
}  # fmt: skip
assert set(_ICON_WORDS.values()) <= ICON_NAMES  # noqa: S101 -- import-time vocabulary guard

#: ``(template_id, raw content params)`` — a card nobody has validated yet.
type Draft = tuple[str, Mapping[str, object]]


@dataclass(frozen=True)
class ScriptLine:
    id: str
    text: str
    #: The whole shot: narration plus the planned hold after it.
    shot_seconds: float
    #: The CLOSING beat: the call-to-action line, last in the script.
    is_cta: bool = False
    #: How long the line is SPOKEN (beats are placed inside this).
    narration_seconds: float = 0.0
    #: The brand name — set on the closing beat only, the one line that may show it.
    brand_name: str = ""
    #: Seconds of the default end card this (closing) line's shot absorbed.
    end_card_seconds: float = 0.0

    @property
    def spoken_ms(self) -> int:
        """Narration length in ms (the whole shot when it was not measured)."""
        return round((self.narration_seconds or self.shot_seconds) * 1000)


# ── rule 1 + 2: what a card may show ────────────────────────────────────────


def foreign_tokens(screen: Sequence[str], line_text: str) -> list[str]:
    """Words, numbers and meaningful symbols on the card that the line does not have.

    A spelled-out number is its digits, and "over N" also says "N+"
    (:func:`headline.with_figures`): "for over thirty capabilities" may show "30+".
    """
    allowed = set(tokens(line_text))
    figures = with_figures(line_text)
    words = [word for text in screen for word in tokens(text) if word not in allowed]
    symbols = [
        char
        for text in screen
        for char in text
        if char in _MEANINGFUL_SYMBOLS and char not in figures
    ]
    return sorted(set(words)) + sorted(set(symbols))


def sentence_coverage(screen: Sequence[str], line_text: str) -> float:
    """Share of the line's tokens the card shows, each counted once per occurrence.

    Deliberately order-blind. It is never lower than the share shown IN ORDER, so
    every card that types the sentence out is caught — and so is one that shows
    the same words with its two halves swapped (a kicker carrying the second
    clause above a headline carrying the first), which an in-order measure
    passes.
    """
    line = tokens(line_text)
    if not line:
        return 0.0
    available = Counter(word for text in screen for word in tokens(text))
    shown = 0
    for token in line:
        if available[token] > 0:
            available[token] -= 1
            shown += 1
    return shown / len(line)


def _figure_violation(params: TemplateParams, line_text: str) -> str | None:
    """A figure must be written as the line writes it (``30+``, never ``30%``)."""
    compact = re.sub(r"\s+", "", with_figures(line_text).lower())
    for text in params.screen_text():
        for match in _FIGURE.finditer(text):
            figure = match.group(0).rstrip(".,")
            if figure.lower() not in compact:
                return f"figure {figure!r} is not written that way in the line"
    return None


def check_draft(line: ScriptLine, draft: Draft) -> tuple[TemplateParams | None, str | None]:
    """``(validated params, None)`` or ``(None, why the card is rejected)``."""
    template_id, raw = draft
    if template_id in UNAUTHORABLE:
        return None, f"template {template_id!r} may not be authored: {UNAUTHORABLE[template_id]}"
    model = PARAM_MODELS.get(template_id)
    if model is None:
        return None, f"template {template_id!r} is not in the brief's vocabulary"
    try:
        params = model.model_validate(raw)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(part) for part in first["loc"])
        return None, f"params do not fit {template_id}: {where}: {first['msg']}"
    screen = params.screen_text()
    # The closing beat — and only it — may also show the brand name.
    sayable = f"{line.text} {line.brand_name}" if line.is_cta else line.text
    foreign = foreign_tokens(screen, sayable)
    if foreign:
        return None, f"on-screen text not from the line: {', '.join(foreign)}"
    figure = _figure_violation(params, sayable)
    if figure:
        return None, figure
    why = _line_problem(line, params, screen)
    return (None, why) if why else (params, None)


def _shape_problem(line: ScriptLine, params: TemplateParams) -> str | None:
    """What render 01a1179d showed, refused before anything is emitted:

    * text the template cannot show at its smallest type (it was cut to a
      character count: "…an audit trail you can");
    * a headline that hands over to its second line on a determiner, modal or
      preposition ("…and trace every" / "decision back…": the two halves are
      revealed in two styles, seconds apart);
    * one word standing for a whole line with a verb ("Attention", 8 s).
    """
    if not isinstance(params, TwoBeatParams):
        return None
    overflow = params.overflow()
    if overflow:
        return f"two_beat does not fit: {overflow}"
    title = params.title.split()
    if params.emphasis and title and _bare(title[-1]) in FUNCTION_WORDS | _HANDOVER_WORDS:
        return f"headline hands over on {_bare(title[-1])!r} — a beat ends on a whole phrase"
    if not params.emphasis and len(title) == 1 and len(line.text.split()) > 3:
        return f"one word ({params.title!r}) cannot carry the line"
    return None


#: Words a first beat may not end on besides function words: a modal or
#: quantifier waits for what it governs ("you can", "trace every").
_HANDOVER_WORDS = frozenset(
    {"every", "each", "all", "any", "some", "can", "could", "will", "would", "may", "might",
     "must", "should", "shall", "very", "more", "most", "less", "few"}
)  # fmt: skip


def _line_problem(line: ScriptLine, params: TemplateParams, screen: Sequence[str]) -> str | None:
    """Why the card misreads the line (rules 2, 2b, 2c and the gate), or ``None``."""
    if not line.is_cta:
        shape = _shape_problem(line, params)
        if shape:
            return shape
    asked = is_question(line.text) and not line.is_cta
    if not line.is_cta and not asked and gated(line.text):
        return _gated_problem(line, params)
    problem = _text_problem(line, params, screen)
    if problem:
        return problem
    if not line.is_cta and not params.graphic:
        return f"{params.template_id} as authored is type only — every card must carry a graphic"
    if (
        not line.is_cta
        and not asked
        and not whole_clause_split(line.text, params)
        and len(line.text.split()) > SHORT_LINE_WORDS
    ):
        return typed_out(params, line.text)
    return None


def _text_problem(line: ScriptLine, params: TemplateParams, screen: Sequence[str]) -> str | None:
    """Why the card's words misread the line, or ``None``."""
    if line.is_cta:
        broken = problems(params.phrases(), sources_for(line.text, *_brand(line)))
        return f"on-screen phrase reads as broken: {broken[0]}" if broken else None
    question = _question_problem(line, params)
    if question:
        return question
    phrases = params.phrases()
    if (
        isinstance(params, TwoBeatParams)
        and params.emphasis
        and _continues(line.text, params.title, params.emphasis)
    ):
        # the second beat continues the title: one phrase of the line
        phrases = [
            *(
                (text, slot)
                for text, slot in phrases
                if text not in (params.title, params.emphasis)
            ),
            (f"{params.title} {params.emphasis}", HEADLINE),
        ]
    broken = problems(phrases, sources_for(line.text))
    if broken:
        return f"on-screen phrase reads as broken: {broken[0]}"
    for headline in _headlines(params):
        start = headline_start_problem(line.text, _with_continuation(line.text, params, headline))
        if start:
            return f"on-screen {start}"
    beat = _second_beat_problem(line.text, params)
    if beat:
        return f"on-screen {beat}"
    hostless = condition_card_problem(line.text, screen)
    if hostless:
        return f"on-screen text {hostless}"
    elements = card_elements(params)
    order = element_order_problem(line.text, elements) if len(elements) > 1 else None
    return f"on-screen element {order}" if order else None


HEADLINE: Literal["headline"] = "headline"


def _with_continuation(line_text: str, params: TemplateParams, headline: str) -> str:
    """A two-beat's title with its second beat when that beat continues it in the
    line: the clause a headline must complete may run on into the second line."""
    if (
        isinstance(params, TwoBeatParams)
        and headline == params.title
        and params.emphasis
        and _continues(line_text, params.title, params.emphasis)
    ):
        return f"{params.title} {params.emphasis}"
    return headline


def whole_clause_split(line_text: str, params: TemplateParams) -> bool:
    """A two-beat whose title + second beat are ONE clause from its start to its
    end, split in two: the sentence rule yields to the whole-clause rule there."""
    if not isinstance(params, TwoBeatParams) or not params.emphasis:
        return False
    if not _continues(line_text, params.title, params.emphasis):
        return False
    words = line_text.strip().split()
    combined = f"{params.title} {params.emphasis}"
    for first, last in locate(words, combined):
        if first in clause_starts(words) and last >= clause_end(words, first):
            return True
    return False


def _second_beat_problem(line_text: str, params: TemplateParams) -> str | None:
    """A two-beat's emphasis either continues its title in the line (the next
    words) or starts where a headline may — never a fragment cut from inside a
    clause ("targets already pay attention" out of "…where targets already…")."""
    if not isinstance(params, TwoBeatParams) or not params.emphasis:
        return None
    if _continues(line_text, params.title, params.emphasis):
        return None
    return headline_start_problem(line_text, params.emphasis)


def _continues(line_text: str, title: str, emphasis: str) -> bool:
    joined, line = tokens(f"{title} {emphasis}"), tokens(line_text)
    return any(line[i : i + len(joined)] == joined for i in range(len(line) - len(joined) + 1))


def _headlines(params: TemplateParams) -> list[str]:
    """The card's free-standing headline(s): its ``title`` and ``kicker`` (a
    two-beat's emphasis is its second beat, not a headline)."""
    return [
        text
        for text in (getattr(params, "title", ""), getattr(params, "kicker", ""))
        if isinstance(text, str) and text
    ]


def card_elements(params: TemplateParams) -> list[tuple[str, Role]]:
    """The parts of a multi-element card cut from the line, in reading order, with
    their role (``headline.element_order_problem``). A checklist (a list) and a
    ladder (a number) are list- and number-shaped and keep their own rules."""
    if isinstance(params, TimelineParams):
        return [(step.title, "node") for step in params.steps]
    if isinstance(params, HandoffParams):
        return [
            (params.source.label, "node"),
            (params.route.label, "connector"),
            (params.to.label, "node"),
        ]
    if isinstance(params, TwoUpParams):
        return [(params.left.title, "node"), (params.right.title, "node")]
    if isinstance(params, TwoBeatParams) and params.emphasis:
        return [(params.title, "node"), (params.emphasis, "node")]
    return []


#: First words that make a line a question even without "?" (audit of d6f223d9:
#: "Why would anything ship before you approve it." became "Ship → Approve it").
#: Used ONLY to detect the first word, never to cut.
_QUESTION_OPENERS = frozenset(
    {
        "why", "what", "how", "who", "when", "where", "which", "is", "are", "was", "were",
        "can", "could", "will", "would", "should", "do", "does", "did", "has", "have",
        "may", "might", "must", "shall",
    }
)  # fmt: skip


def is_question(text: str) -> bool:
    """A line ending in "?", or opened by a wh-word or an inverted auxiliary
    ("When…," with a comma is a leading condition, not a question)."""
    words = text.strip().split()
    if not words:
        return False
    if text.strip().endswith("?"):
        return True
    return _bare(words[0]) in _QUESTION_OPENERS and "," not in text


def _question_problem(line: ScriptLine, params: TemplateParams) -> str | None:
    """A question is shown as itself, one headline — never derived into steps.

    Audit of 603186d3: "Why would anything ship before you approve it?" became a
    track "Ship → Approve it", which states the opposite of the question.
    """
    if not is_question(line.text):
        return None
    if not isinstance(params, TwoBeatParams) or params.emphasis or params.kicker:
        return f"{params.template_id}: a question is shown as itself, as one headline"
    marked = line.text.strip().endswith("?")
    if tokens(params.title) != tokens(line.text) or (marked and not params.title.endswith("?")):
        return "a question's headline is the question itself"
    return None


def _question_drafts(line: ScriptLine) -> list[Draft]:
    return [("two_beat", {"title": line.text.strip(), "icon": _icon(line.text)})]


def _brand(line: ScriptLine) -> tuple[str, ...]:
    return (line.brand_name,) if line.is_cta and line.brand_name else ()


def typed_out(params: TemplateParams, line_text: str) -> str | None:
    """Rule 2 — the sentence typed out — for a line of more than eight words.

    A TYPE card (``two_beat``, the title card, the divider) — and any card
    with a free-standing headline (``title`` / ``kicker``) — may not carry 80%
    of the line's words in all: there the words ARE the design. A DIAGRAM card
    with no headline (panels, a track, rows, a route, a ladder) lays the line's
    clauses out as parts of a drawing, each a clean span of the line
    (:func:`headline.phrase_problem`); it may show more of the line, but no
    single element may carry 80% of it (that element is the sentence typed out).
    "Merqi runs your go-to-market motion" | "You keep the final say" on two
    panels, the system's beside yours, is a design; the same words as a
    headline are the sentence.
    """
    screen = params.screen_text()
    headline = bool(getattr(params, "title", "") or getattr(params, "kicker", ""))
    if params.template_id in TYPE_CARDS or headline:
        coverage = sentence_coverage(screen, line_text)
    else:
        coverage = max((sentence_coverage([text], line_text) for text in screen), default=0.0)
    if coverage >= SENTENCE_COVERAGE_LIMIT:
        return f"shows {coverage:.0%} of the line's words — the sentence typed out"
    return None


# ── the deterministic author ────────────────────────────────────────────────


def _bare(word: str) -> str:
    return re.sub(r"[^\w]", "", word.lower())


def _phrase(words: Sequence[str]) -> str:
    text = " ".join(words).strip(" ,;:.!?")
    if _DOMAIN.match(text.split(" ", 1)[0]):
        return text  # a domain keeps its own casing ("merqi.nl")
    return text[:1].upper() + text[1:]


def _as_spoken(words: Sequence[str]) -> str:
    """The words as the line writes them — a second beat continues the sentence."""
    return " ".join(words).strip(" ,;:.!?")


def _clauses(text: str) -> list[list[str]]:
    clean = text.strip().rstrip(".!?").strip()
    return [part.split() for part in _CLAUSE.split(clean) if part.strip()]


def _without_article(words: list[str]) -> list[str]:
    return words[1:] if len(words) > 1 and _bare(words[0]) in _ARTICLES else words


def _content_runs(words: Sequence[str]) -> list[list[str]]:
    runs: list[list[str]] = [[]]
    for word in words:
        if _bare(word) in _STOPWORDS:
            runs.append([])
        else:
            runs[-1].append(word)
    return [run for run in runs if run]


def _longest_run(words: Sequence[str], limit: int = _RUN_WORDS) -> list[str]:
    """The longest stretch of content words (the first, on a tie), capped."""
    runs = _content_runs(words)
    return max(runs, key=len)[:limit] if runs else []


def _auxiliary_index(words: Sequence[str]) -> int | None:
    return next((i for i, word in enumerate(words) if i and _bare(word) in _AUXILIARIES), None)


def key_phrase(text: str) -> list[str]:
    """The words a card may carry INSTEAD of the sentence.

    A short opening clause is the phrase ("Go-to-market that runs itself");
    otherwise what the sentence says of its subject (the words after its
    auxiliary: "automatically routed to a human agent"); otherwise the longest
    stretch of content words.
    """
    clauses = _clauses(text)
    if not clauses:
        return []
    first = _without_article(clauses[0])
    if len(clauses) > 1 and len(first) <= _KEY_PHRASE_WORDS:
        return first
    aux = _auxiliary_index(first)
    if aux is not None:
        predicate = first[aux + 1 :]
        if 2 <= len(predicate) <= _KEY_PHRASE_WORDS:
            return predicate
        if len(predicate) > _KEY_PHRASE_WORDS:
            return _longest_run(predicate)
    return _longest_run(first) or first[:_RUN_WORDS]


def _secondary_phrase(text: str) -> list[str]:
    """A supporting phrase from the rest of the line (``[]`` when there is none)."""
    clauses = _clauses(text)
    if len(clauses) > 1:
        other = _without_article(clauses[1])
        return other if len(other) <= _KEY_PHRASE_WORDS else _longest_run(other)
    first = _without_article(clauses[0]) if clauses else []
    aux = _auxiliary_index(first)
    return _longest_run(first[:aux]) if aux is not None else []


def icon_for(label: str) -> str:
    """The recorder glyph for ``label`` (``""`` when no word of it has one)."""
    return next((_ICON_WORDS[word] for word in tokens(label) if word in _ICON_WORDS), "")


def _with_icon(label: str, **more: object) -> dict[str, object]:
    icon = icon_for(label)
    return {"title": label, **({"icon": icon} if icon else {}), **more}


def _tail_phrase(words: Sequence[str], limit: int = _RUN_WORDS) -> list[str]:
    """The words after the last connector ("…are made by | deterministic code")."""
    tail: list[str] = []
    for word in reversed(words):
        if _bare(word) in _CONNECTORS:
            break
        tail.insert(0, word)
    return tail[-limit:]


def _headed(template_id: str, body: dict[str, object], title: list[str]) -> list[Draft]:
    """``body`` under a headline, a shorter headline, then none.

    The validator keeps the first that fits: a headline that would put the whole
    sentence on the card is dropped for the tail of its longest content run
    ("pull-based demand generation"), and that for no headline at all.
    """
    short = _longest_run(title, len(title))[-_HEADLINE_TAIL_WORDS:]
    titles = [title, short] if short and short != title else [title]
    headed: list[Draft] = [
        (template_id, {**body, "title": _phrase(words)}) for words in titles if words
    ]
    return [*headed, (template_id, body)]


def _range(text: str) -> tuple[list[str], list[str]] | None:
    """The two ends of "from A to B" (``None`` when the line has no such range)."""
    for clause in _clauses(text):
        bare = [_bare(word) for word in clause]
        if "from" not in bare:
            continue
        start = bare.index("from")
        if "to" not in bare[start + 2 :]:
            continue
        end = bare.index("to", start + 2)
        left, right = clause[start + 1 : end], clause[end + 1 :]
        if 1 <= len(left) <= _SIDE_WORDS and 1 <= len(right) <= _SIDE_WORDS:
            return left, right
    return None


def _noun_run_before(words: Sequence[str]) -> list[str]:
    """The longest content run of ``words`` as a noun phrase.

    A run that follows a modal or a subject pronoun ("can SET graduated autonomy
    levels", "you SET …") opens with a verb, which is dropped. Fewer than two words is not a heading.
    """
    best: list[str] = []
    run: list[str] = []
    after_modal = False
    for word in [*words, ""]:
        bare = _bare(word)
        if word and bare not in _STOPWORDS:
            run.append(word)
            continue
        noun = run[1:] if after_modal else run
        if len(noun) > len(best):
            best = noun
        run, after_modal = [], bare in _MODALS or bare in SUBJECT_PRONOUNS
    return best[:_RUN_WORDS] if len(best) > 1 else []


def _figure(text: str) -> tuple[re.Match[str], list[str], list[str]] | None:
    """``(the number, the noun after it, the heading before it)`` of a line with a figure."""
    text = with_figures(text)
    match = _NUMBER.search(text)
    if match is None:
        return None
    clause_end = _CLAUSE.search(text, match.end())
    after = text[match.end() : clause_end.start() if clause_end else len(text)].split()
    label = next(iter(_content_runs(after)), [])[:_SIDE_WORDS]
    return match, label, _noun_run_before(text[: match.start()].split())


_GRADED = re.compile(r"\b(?:levels?|graduated|tiers?|grades?|stages?)\b", re.IGNORECASE)


def _levels_drafts(text: str) -> list[Draft]:
    """A figure AND a range ("30+ capabilities, ranging from A to B") → the stat
    beside a ladder that fills from A to B. A figure on a line about graded
    levels with no named ends ("graduated autonomy levels for over thirty
    capabilities") → the same ladder, unlabelled: the ends are never invented."""
    figure, ends = _figure(text), _range(text)
    if figure is None or (ends is None and not _GRADED.search(text)):
        return []
    match, label, heading = figure
    body: dict[str, object] = {
        "value": match.group(1),
        "suffix": match.group(2) or "",
        "label": _as_spoken(label),
        **({"from_label": _phrase(ends[0]), "to_label": _phrase(ends[1])} if ends else {}),
    }
    titled: list[Draft] = [("levels", {**body, "title": _phrase(heading)})] if heading else []
    return [*titled, ("levels", body)]


def _alternatives(words: Sequence[str]) -> list[list[str]]:
    """ "scraped or bought lists" → [scraped lists], [bought lists].

    The parts of an or/and list. When every part but the last is one bare
    modifier, the last part's head noun is shared out to them.
    """
    parts = [part.split() for part in _ALTERNATIVE.split(" ".join(words)) if part.strip()]
    parts = [_without_article(part) for part in parts]
    if len(parts) > 1 and len(parts[-1]) > 1 and all(len(part) == 1 for part in parts[:-1]):
        head = parts[-1][1:]
        parts = [[*part, *head] for part in parts[:-1]] + [parts[-1]]
    return [part[:_SIDE_WORDS] for part in parts if part]


def _with_limiter(words: Sequence[str], run: list[str]) -> list[str]:
    """``run`` with the limiter right before it in ``words`` ("only public sources")."""
    for index in range(len(words) - len(run) + 1):
        if (
            list(words[index : index + len(run)]) == run
            and index
            and _bare(words[index - 1]) == "only"
        ):
            return [words[index - 1], *run]
    return run


def _checklist_drafts(text: str) -> list[Draft]:
    """ "does A and B, avoiding X or Y" → A ✓, B ✓, X ✗, Y ✗."""
    match = _NEGATIVE.match(text.strip().rstrip(".!?"))
    if match is None:
        return []
    done = [
        _with_limiter(part.split(), runs[-1])[:_SIDE_WORDS]
        for part in re.split(r"\s+and\s+", match.group(1))
        if (runs := _content_runs(part.split()))
    ]
    avoided = _alternatives(_clauses(match.group(2))[0])
    if not done or not avoided or len(done) + len(avoided) > 6:
        return []
    items = [
        *({"label": _phrase(words), "state": "yes"} for words in done),
        *({"label": _phrase(words), "state": "no"} for words in avoided),
    ]
    return [("checklist", {"items": items})]


def _handoff_drafts(text: str) -> list[Draft]:
    """ "<subject> are automatically routed to <destination>" → source → route → destination."""
    match = _ROUTED.match(text.strip().rstrip(".!?"))
    if match is None:
        return []
    subject = _without_article(match.group(1).split())
    destination = _without_article(_clauses(match.group(3))[0])[:_SIDE_WORDS]
    if not subject or not destination:
        return []
    source, target = _phrase(subject[-1:]), _phrase(destination)
    body: dict[str, object] = {
        "from": {"label": source, "icon": icon_for(source) or "bot"},
        "route": {"label": _phrase(match.group(2).split())},
        "to": {"label": target, "icon": icon_for(target) or "user"},
    }
    return _headed("handoff", body, subject if 2 <= len(subject) <= _KEY_PHRASE_WORDS else [])


#: "<subject> route(s) [adverb] to <destination>" — the active form of :data:`_ROUTED`.
_ROUTES = re.compile(
    r"^(.+?)\s+((?:route|routes|escalate|escalates|pass|passes|hand\s+off|hands\s+off|"
    r"go|goes|move|moves)(?:\s+\w+ly)?)\s+(?:over\s+|on\s+)?(?:to|into)\s+(.+)$",
    re.IGNORECASE,
)


def _route_drafts(text: str) -> list[Draft]:
    """ "<subject> route automatically to <destination>" → subject → route → destination."""
    match = _ROUTES.match(text.strip().rstrip(".!?"))
    if match is None:
        return []
    source = _without_article(match.group(1).split())
    destination = _without_article(_clauses(match.group(3))[0])[:_SIDE_WORDS]
    if not source or not destination:
        return []
    route = {"label": _phrase(match.group(2).split())}
    target = _phrase(destination)
    to = {"label": target, "icon": icon_for(target) or "user"}
    drafts: list[Draft] = []
    for words in (source, source[-1:]):  # the whole subject, then its head noun
        label = _phrase(words)
        drafts.append(
            ("handoff", {"from": {"label": label, "icon": icon_for(label) or "bot"},
                         "route": route, "to": to})
        )  # fmt: skip
    return drafts


#: "<subject> builds X tailored to Y" — something made FROM a context: input → output.
_TAILORED = re.compile(
    r"^(.+?)\s+(?:builds?|creates?|makes?|generates?|writes?|produces?|designs?|drafts?)\s+"
    r"(.+?)\s+(tailored|customi[sz]ed|personali[sz]ed|adapted|matched|fitted)\s+(?:to|for)\s+(.+)$",
    re.IGNORECASE,
)


def _tailored_drafts(text: str) -> list[Draft]:
    """ "builds X tailored to Y" → a hand-off from the context Y to the product X."""
    match = _TAILORED.match(text.strip().rstrip(".!?"))
    if match is None:
        return []
    product = strip_determiner(match.group(2).split())
    context = _clauses(match.group(4))[0]
    if not product or not context:
        return []
    made = _phrase(product)
    drafts: list[Draft] = []
    for words in (context, strip_determiner(context), context[-2:]):
        label = _phrase(words)
        drafts.append(
            (
                "handoff",
                {
                    # in the line's order: the assets, tailored, to the context
                    "from": {"label": made, "icon": icon_for(made) or "layers"},
                    "route": {"label": _phrase([match.group(3)])},
                    "to": {"label": label, "icon": icon_for(label) or "target"},
                },
            )
        )
    return drafts


#: "X while you Y" / "X, and you Y" — the machine's part beside the person's.
_PARALLEL = re.compile(r"^(.+?),?\s+(?:while|whilst|and|as)\s+((?:you|we)\s+.+)$", re.IGNORECASE)


def _parallel_drafts(text: str) -> list[Draft]:
    """ "Merqi runs X while you keep Y" → two panels: the system's part, then yours."""
    match = _PARALLEL.match(text.strip().rstrip(".!?"))
    if match is None:
        return []
    machine, person = match.group(1).split(), match.group(2).split()
    if len(machine) > _KEY_PHRASE_WORDS + 1 or len(person) > _KEY_PHRASE_WORDS + 1:
        return []
    right = {"title": _phrase(person), "icon": "user"}
    return [
        ("two_up", {"left": {"title": _phrase(words), "icon": "bot"}, "right": right})
        for words in (machine, predicate(machine))
    ]


def _only_drafts(text: str) -> list[Draft]:
    """ "uses only A and respects B" → a checklist of what the line commits to: A ✓, B ✓."""
    if not re.search(r"\bonly\b", text, re.IGNORECASE):
        return []
    clause = _clauses(text)[0]
    parts = re.split(r"\s+and\s+", " ".join(predicate(clause)), flags=re.IGNORECASE)
    items: list[str] = []
    for part in parts:
        words = part.split()
        bare_words = [_bare(word) for word in words]
        if "only" in bare_words:
            # the limiter stays on the card: "Only public sources", never "Public sources"
            words = words[bare_words.index("only") :]
        words = trim_trailing(words)
        if not 1 <= len(words) <= _SIDE_WORDS:
            return []
        items.append(_phrase(words))
    if not 2 <= len(items) <= 6:
        return []
    return [("checklist", {"items": [{"label": item, "state": "yes"} for item in items]})]


_SHARED_HEAD = re.compile(
    r"^(?:([\w-]+)\s+)?([\w-]+)\s+(?:and|or)\s+([\w-]+)\s+([\w-]+)$", re.IGNORECASE
)


def _coordination_drafts(text: str) -> list[Draft]:
    """ "Deterministic code handles critical A and B decisions" → the subject over a
    checklist of what it takes: A decisions ✓, B decisions ✓."""
    words = text.strip().rstrip(".!?").split()
    owner = subject(words)
    if not 1 <= len(owner) <= _SIDE_WORDS or len(owner) + 1 >= len(words):
        return []
    match = _SHARED_HEAD.match(" ".join(words[len(owner) + 1 :]))
    if match is None:
        return []
    head = match.group(4)
    items = [{"label": _phrase([part, head]), "state": "yes"} for part in match.group(2, 3)]
    return [
        ("checklist", {"title": _phrase(strip_determiner(owner)), "items": items}),
        ("checklist", {"items": items}),
    ]


def _contrast_drafts(text: str) -> list[Draft]:
    """ "X, not Y" → two panels: X chosen, Y rejected."""
    clean = text.strip().rstrip(".!?")
    match = _CONTRAST.match(clean)
    if match is None:
        return []
    left_part = match.group(1).split()
    right = _without_article(_clauses(match.group(2))[0])[:_KEY_PHRASE_WORDS]
    left = _tail_phrase(left_part)
    if not left or not right:
        return []
    aux = _auxiliary_index(left_part)
    subject = _without_article(left_part[:aux]) if aux is not None else []
    body: dict[str, object] = {
        "left": _with_icon(_phrase(left), state="chosen"),
        "right": _with_icon(_phrase(right), state="rejected"),
    }
    return _headed("two_up", body, subject if 2 <= len(subject) <= _KEY_PHRASE_WORDS else [])


def _track(parts: Sequence[Sequence[str]], title: list[str]) -> list[Draft]:
    steps = [_with_icon(_phrase(part)) for part in parts]
    return _headed("timeline", {"steps": steps}, title)


#: "<doing X> to find/identify/pick <Y>" — a step and what it leads to.
_PURPOSE = re.compile(
    r"^(.+?)\s+(?:in order\s+)?to\s+((?:find|identify|pick|choose|surface|spot|select|rank)\s+.+)$",
    re.IGNORECASE,
)


def _step_words(words: Sequence[str]) -> list[str]:
    """One step of a sequence: its words after the last connector, without a
    subject or a stranded adverb, at most :data:`_SIDE_WORDS` (``[]`` when longer)."""
    tail = _tail_phrase(words, len(words))
    step = trim_trailing(strip_stranded(tail))
    if step and not _bare(step[0]).endswith("ing"):
        step = predicate(step)
        if step and starts_with_verb(" ".join(step)):
            # a finite verb leaves its subject behind: the step is its object
            # ("earns attention" → "Attention"), a node is a noun phrase
            step = _without_article(list(step[1:]))
    return step if len(step) <= _SIDE_WORDS else []


def _step_drafts(text: str) -> list[Draft]:
    """ "A before B", "A then B", "from A to B", or a list of three or more → a track."""
    clauses = _clauses(text)
    for index, clause in enumerate(clauses):
        match = _SEQUENCE.match(" ".join(clause))
        if match is None:
            continue
        # "The system earns attention before ever requesting contact": the steps
        # are what is DONE — "Earns attention", "Requesting contact" — never the
        # subject, and never a stranded "ever".
        first = _step_words(match.group(1).split())
        second = _step_words(match.group(2).split())
        if first and second:
            return _track([first, second], key_phrase(text) if index else [])
    purpose = _PURPOSE.match(text.strip().rstrip(".!?"))
    if purpose is not None:
        # "scores your marketing surfaces to find the highest-ROI fix next":
        # what is scored → what it finds.
        scored = trim_trailing(strip_determiner(predicate(purpose.group(1).split())[1:]))
        found = trim_trailing(strip_determiner(purpose.group(2).split()[1:]))
        if 1 <= len(scored) <= _SIDE_WORDS and 1 <= len(found) <= _SIDE_WORDS:
            return _track([scored, found], [])
    ends = _range(text)
    if ends is not None:
        return _track(ends, key_phrase(text))
    listed = re.split(r",\s*(?:and\s+)?|\s+and\s+", text.strip().rstrip(".!?"))
    if text.count(",") >= 2 and len(listed) >= 3:
        parts = [_tail_phrase(listed[0].split(), _SIDE_WORDS), *(p.split() for p in listed[1:])]
        if all(1 <= len(part) <= _SIDE_WORDS for part in parts):
            return _track(parts[:6], [])
    return []


def _figure_drafts(text: str) -> list[Draft]:
    """A figure with no range: the heading, then the figure as the second beat."""
    figure = _figure(text)
    if figure is None or not figure[1]:
        return []
    match, label, heading = figure
    emphasis = f"{match.group(1)}{match.group(2) or ''} {_as_spoken(label)}"
    title = heading or key_phrase(text)
    if not title:
        return []
    return [("two_beat", {"title": _phrase(title), "emphasis": emphasis, "icon": _icon(text)})]


def _icon(text: str) -> str:
    """The glyph a type card carries: the first word of ``text`` that names one,
    else :data:`DEFAULT_ICON` — a type card never goes out without a graphic."""
    return icon_for(text) or DEFAULT_ICON


def _closing_drafts(line: ScriptLine) -> list[Draft]:
    """The closing beat: the brand name over the call to action (the line itself)."""
    action = _as_spoken(line.text.split())
    if not line.brand_name:
        return [(CLOSING_TEMPLATE, {"title": action})]
    return [
        (CLOSING_TEMPLATE, {"title": line.brand_name, "subtitle": action}),
        (CLOSING_TEMPLATE, {"title": action}),
    ]


def _phrase_drafts(line: ScriptLine) -> list[Draft]:
    """Key-phrase cards: the phrase in two beats beside an icon from the line."""
    primary = _phrase(key_phrase(line.text))
    secondary = _as_spoken(_secondary_phrase(line.text))
    if not primary:
        return []
    icon = _icon(line.text)
    drafts: list[Draft] = []
    if secondary:
        drafts.append(("two_beat", {"title": primary, "emphasis": secondary, "icon": icon}))
    drafts.append(("two_beat", {"title": primary, "icon": icon}))
    return drafts


def _valid(line: ScriptLine, drafts: Sequence[Draft]) -> list[TemplateParams]:
    return [params for params, _reason in (check_draft(line, d) for d in drafts) if params]


def _phrase_text(words: Sequence[str]) -> str:
    return _phrase(words) if words else ""


def fallback_cards(line: ScriptLine) -> list[TemplateParams]:
    """Cards for a line of no shape — every one passes :func:`check_draft`.

    The phrase is a CLEAN span of the line (:func:`headline.clean_spans`: a
    leading clause cut at a clause boundary, else the strongest noun phrase),
    never a run of words with the middle dropped. It goes into ``two_beat``
    beside an icon chosen from the line's words; a second clean span, when the
    line has one, is its emphasis. A ``timeline`` of the line's two strongest
    phrases is offered too, so the two-uses rule has somewhere to go.
    """
    icon = _icon(line.text)
    if gated(line.text):
        return _valid(line, gated_cards(line)) or [_whole_line_card(line, icon)]
    spans = clean_spans(line.text, max_words=_KEY_PHRASE_WORDS)
    whole = line.text.strip().rstrip(".!?").split()
    if len(whole) <= SHORT_LINE_WORDS and whole not in spans:
        spans = [*spans, whole]  # a tagline that cannot be shortened cleanly
    # The line's first clause, whole, comes first (audit of #1084): a headline
    # that starts at a subject runs to its clause's end; a shorter headline is
    # then only a whole clause or the line's subject / object noun phrase.
    drafts: list[Draft] = [
        ("two_beat", {**split, "icon": icon}) for split in _clause_splits(line.text)
    ]
    for title in spans[:12]:
        after = [
            span
            for span in clean_spans(line.text, max_words=_SIDE_WORDS)
            if not set(map(_bare, span)) & set(map(_bare, title))
        ]
        if after:
            emphasis = _as_spoken(after[0])
            drafts.append(
                ("two_beat", {"title": _phrase(title), "emphasis": emphasis, "icon": icon})
            )
        drafts.append(("two_beat", {"title": _phrase(title), "icon": icon}))
    nouns = [span for span in clean_spans(line.text, max_words=_SIDE_WORDS) if len(span) > 1]
    if len(nouns) >= 2 and not set(map(_bare, nouns[0])) & set(map(_bare, nouns[1])):
        drafts.extend(_track(nouns[:2], []))
    cards = _valid(line, drafts)
    return cards or [_whole_line_card(line, icon)]


#: Words that open a subordinate clause the line's claim may rest on.
_SUBORDINATE = frozenset(
    {"until", "unless", "without", "if", "when", "whenever", "before", "after", "except",
     "while", "which", "who", "whom", "whose", "that"}
)  # fmt: skip


def gated(text: str) -> bool:
    """A line with a condition, a relative clause or a negation (audit of
    fac79031, decided): it never gets a diagram derived from its clauses — only
    its main clause + that clause whole (:func:`main_and_clause`), or the whole
    line. Clause-derived steps / handoffs / rows lost "before you approve them"
    and "Merqi hands the thread…" there; diagrams stay for plain lines."""
    words = text.split()
    return any(
        _bare(word) in _SUBORDINATE or is_negator(word) or _second_clause_at(words, i)
        for i, word in enumerate(words)
    )


def _second_clause_at(words: Sequence[str], index: int) -> bool:
    """``words[index]`` is and/but/or/nor opening a second clause with its OWN
    subject ("…leads | and a human closes the deal", "…| but your team picks…").

    Its next word is a determiner or a subject pronoun. A verb list with one
    subject ("drafts, reviews, | and schedules posts") and joined modifiers
    ("financial | and consent decisions") have neither, and stay ungated.
    """
    if _bare(words[index]) not in {"and", "but", "or", "nor"} or index + 1 >= len(words):
        return False
    following = _bare(words[index + 1])
    return following in DETERMINERS or following in SUBJECT_PRONOUNS


def main_and_clause(text: str) -> tuple[str, str] | None:
    """``(main clause, subordinate clause)`` of a gated line, each WHOLE, together
    the whole line — or ``None`` when it has no subordinate clause.

    The main clause is the one that does not start with the subordinate word,
    whether the subordinate clause leads ("When a lead replies, | Merqi hands
    the thread to your sales rep") or trails ("Merqi drafts, reviews, and
    schedules posts | before you approve them").
    """
    words = text.strip().split()
    at = next(
        (
            i
            for i, word in enumerate(words)
            if _bare(word) in _SUBORDINATE or _second_clause_at(words, i)
        ),
        None,
    )
    if at is None:
        return None
    if at == 0:
        comma = next((i for i, word in enumerate(words) if re.search(r"[,;:]$", word)), None)
        if comma is None or comma + 1 >= len(words):
            return None
        return _phrase(words[comma + 1 :]), _as_spoken(words[: comma + 1]).rstrip(",;:")
    return _phrase(words[:at]), _as_spoken(words[at:])


def gated_cards(line: ScriptLine) -> list[Draft]:
    """The only cards a gated line gets: (a) its main clause as the headline and
    the subordinate clause as the sub-line, both whole; or the whole line when
    it is short and has no subordinate clause. Anything else → the whole-line
    card, reported (:func:`_whole_line_card`)."""
    icon = _icon(line.text)
    split = main_and_clause(line.text)
    if split is not None:
        return [("two_beat", {"title": split[0], "emphasis": split[1], "icon": icon})]
    return [("two_beat", {"title": _as_spoken(line.text.strip().split()), "icon": icon})]


def _gated_problem(line: ScriptLine, params: TemplateParams) -> str | None:
    """A gated line's card must be :func:`gated_cards` — the model's included."""
    allowed = [
        (tokens(str(raw.get("title", ""))), tokens(str(raw.get("emphasis", ""))))
        for _template, raw in gated_cards(line)
    ]
    shown = (
        (tokens(params.title), tokens(params.emphasis))
        if isinstance(params, TwoBeatParams)
        else None
    )
    if shown not in allowed:
        return (
            f"{params.template_id}: a line with a condition, relative clause or negation is "
            "shown as its main clause + that clause, each whole"
        )
    if not params.graphic:
        return f"{params.template_id} as authored is type only — every card must carry a graphic"
    if len(line.text.split()) > SHORT_LINE_WORDS and not params.template_params().get("emphasis"):
        return "the whole line in one headline — shown whole, reported"
    return None


def _clause_splits(text: str) -> list[dict[str, object]]:
    """The line's first clause as headline + second line, both whole together:
    a clause too long for one headline (run 01a11693: "Merqi executes your
    entire revenue motion" / "under a single governed engine")."""
    words = text.strip().split()
    end = clause_end(words, 0)
    clause = words[: end + 1]
    if len(words) <= SHORT_LINE_WORDS and end == len(words) - 1:
        return [{"title": _as_spoken(words)}]  # a short line is its own headline
    return [
        {"title": _phrase(clause[:cut]), "emphasis": _as_spoken(clause[cut:])}
        for cut in _good_cuts(clause)
    ]


def _good_cuts(words: Sequence[str]) -> list[int]:
    """Where ``words`` may be split into headline + second line, best first.

    The headline never ends on a function word, modal or quantifier; a cut at
    a comma or right before a function word ("…single use | and expires…",
    "…motion | under a…") reads best; both halves must fit the two-beat at its
    smallest type (:meth:`TwoBeatParams.overflow`)."""
    cuts = [
        cut
        for cut in range(2, len(words))
        if _bare(words[cut - 1]) not in FUNCTION_WORDS | _HANDOVER_WORDS
        and wrapped_lines(" ".join(words[:cut])) <= TWO_BEAT_TITLE_LINES
        and wrapped_lines(" ".join(words[cut:])) <= TWO_BEAT_EMPHASIS_LINES
    ]
    return sorted(
        cuts, key=lambda cut: (not _reads_as_a_break(words, cut), _break_tier(words[cut]), -cut)
    )


#: Where a second line reads best: before a clause (and/but/or, a condition, a
#: relative pronoun), then before a preposition, then anywhere else.
_CLAUSE_TURNS = frozenset(
    {"and", "but", "or", "nor", "while", "whereas", "because", "until", "unless", "if",
     "when", "before", "after", "where", "which", "who", "that", "without", "so"}
)  # fmt: skip


def _break_tier(word: str) -> int:
    bare = _bare(word)
    if bare in _CLAUSE_TURNS:
        return 0
    return 1 if bare in FUNCTION_WORDS else 2


def _reads_as_a_break(words: Sequence[str], cut: int) -> bool:
    """A cut right before a function word, at ; : or at a comma — never at the
    comma between two modifiers of one noun phrase ("graduated, | governed
    autonomy")."""
    if _bare(words[cut]) in FUNCTION_WORDS or re.search(r"[;:]$", words[cut - 1]):
        return True
    # a comma, unless it joins two modifiers of one noun phrase
    return words[cut - 1].endswith(",") and not modifier_list(words[cut - 1], words[cut])


def _whole_line_card(line: ScriptLine, icon: str) -> TemplateParams:
    """The last resort: the whole line on ``two_beat`` with an icon.

    Never fails the run over card text (audit of #1080: a negation that must
    keep its scope can leave a long line no shorter phrase). It is NOT checked
    by :func:`check_draft` — it may break the sentence rule — and
    :func:`choose_items` reports it as ``NOT repaired``.
    """
    words = line.text.strip().split()
    # Never cut to a character count (render 01a1179d: "…audit trail you can"):
    # the whole line, split where it reads and fits; when nothing fits, the
    # split that overflows least — every word still sent, the overflow
    # reported (choose_items) and flagged by the recorder.
    cuts = _good_cuts(words)
    if not cuts:
        cuts = sorted(
            range(1, len(words)),
            key=lambda cut: (
                _bare(words[cut - 1]) in FUNCTION_WORDS | _HANDOVER_WORDS,
                max(0, wrapped_lines(" ".join(words[:cut])) - TWO_BEAT_TITLE_LINES)
                + max(0, wrapped_lines(" ".join(words[cut:])) - TWO_BEAT_EMPHASIS_LINES),
                not _reads_as_a_break(words, cut),
            ),
        )
    cut = cuts[0] if cuts else len(words)
    title = " ".join(words[:cut]).strip(" ,;:")
    return TwoBeatParams(title=title, emphasis=_as_spoken(words[cut:]), icon=icon)


def fallback_card(line: ScriptLine) -> TemplateParams:
    """The safe card: ``two_beat`` carrying the line's cleanest phrase and an icon."""
    return fallback_cards(line)[0]


def designed_cards(line: ScriptLine) -> list[TemplateParams]:
    """The cards the line's SHAPE asks for, best first (``[]`` when it has none).

    By meaning, not only by numbers (run 01a11511: plain sentences got a ring
    and text). The closing beat → its fixed card. A figure and a range, or a
    figure on graded levels → ``levels``; "…, avoiding X or Y" / "only A and B"
    / "X handles A and B <noun>" → ``checklist``; "routed / route(s) to Z" →
    ``handoff``; "builds X tailored to Y" → ``handoff`` from Y to X; "X, not Y"
    → ``two_up`` chosen/rejected; "X while you Y" → ``two_up`` panels, the
    system's part beside yours; "A before B" / "from A to B" / "scores A to
    find B" / a list → a ``timeline`` track; a figure alone → ``two_beat`` with
    the figure as its second beat. Each has already passed :func:`check_draft`.
    """
    if line.is_cta:
        return _valid(line, _closing_drafts(line))
    if is_question(line.text):
        return _valid(line, _question_drafts(line))
    if gated(line.text):
        return _valid(line, gated_cards(line))
    text = line.text
    built = [
        *_levels_drafts(text),
        *_checklist_drafts(text),
        *_only_drafts(text),
        *_handoff_drafts(text),
        *_route_drafts(text),
        *_tailored_drafts(text),
        *_contrast_drafts(text),
        *_parallel_drafts(text),
        *_step_drafts(text),
        *_coordination_drafts(text),
        *_figure_drafts(text),
        # no designed shape: the line's first clause, whole (render 01a1179d:
        # never a key phrase like "Whole revenue motion" before it)
        *(("two_beat", {**split, "icon": _icon(text)}) for split in _clause_splits(text)),
    ]
    return _valid(line, built)


def deterministic_drafts(line: ScriptLine) -> list[TemplateParams]:
    """Every card the rules can build for ``line``, best first — no model.

    :func:`designed_cards`, then the key phrase in two beats, then
    :func:`fallback_cards`. Never empty; on a line that is not the closing beat,
    every card carries a graphic.
    """
    if is_question(line.text) and (asked := designed_cards(line)):
        return asked  # a question is shown as itself, nothing else
    generic = [] if line.is_cta else _valid(line, _phrase_drafts(line))
    fallback = [] if line.is_cta else fallback_cards(line)
    cards = [*designed_cards(line), *generic, *fallback]
    unique: list[TemplateParams] = []
    for card in cards:
        if card not in unique:
            unique.append(card)
    return unique


# ── timing: the narration's, never a delay ──────────────────────────────────


def _spoken_index(words: Sequence[str], phrase: str, start: int) -> int | None:
    """Index of the word at which ``phrase`` starts being spoken, at or after ``start``."""
    wanted = tokens(phrase)
    if not wanted:
        return None
    spoken = [tokens(word) for word in words]
    hits = [i for i in range(start, len(words)) if spoken[i] and spoken[i][0] == wanted[0]]
    if not hits:
        return None
    rest = wanted[len(spoken[hits[0]]) :][:1]
    for index in hits:  # prefer the occurrence whose next word continues the phrase
        following = spoken[index + 1] if index + 1 < len(words) else []
        if not rest or following[:1] == rest:
            return index
    return hits[0]


def narration_marks(params: TemplateParams, line: ScriptLine) -> list[int]:
    """One beat offset (ms) per ``marks`` slot of the card's template.

    Beat 0 is 0 — the card's first content is on screen from the cut. Every
    later beat lands where its phrase is SPOKEN: the index of the phrase's first
    word over the line's word count, times the narration's length. A beat with
    no phrase of its own, or whose phrase the line does not say (the brand name
    on the closing beat), opens with the beat before it. Beats never run
    backwards and never fall after the narration ends — nothing is held back
    into the pause.
    """
    words = line.text.split()
    marks: list[int] = []
    cursor = 0
    for position, phrase in enumerate(params.beats()):
        at = marks[-1] if marks else 0
        if position and phrase and words:
            index = _spoken_index(words, phrase, cursor)
            if index is not None:
                cursor = index
                at = max(at, min(round(index / len(words) * line.spoken_ms), line.spoken_ms))
        marks.append(at)
    return marks


# ── rules 3 + 4: two uses per template, and motion through the hold ─────────


@dataclass
class Authored:
    """One line's card and how it came to be."""

    line: ScriptLine
    params: TemplateParams
    authored_by: Literal["model", "fallback"]
    rejections: list[str] = field(default_factory=list)
    repairs: list[str] = field(default_factory=list)

    @property
    def shot_ms(self) -> int:
        return round(self.line.shot_seconds * 1000)

    @property
    def marks(self) -> list[int]:
        return narration_marks(self.params, self.line)

    @property
    def static_seconds(self) -> float:
        return card_static_seconds(self.params, self.marks, shot_ms=self.shot_ms)


def _resting(line: ScriptLine, params: TemplateParams) -> float:
    shot_ms = round(line.shot_seconds * 1000)
    return card_static_seconds(params, narration_marks(params, line), shot_ms=shot_ms)


def _options(item: Authored) -> list[tuple[int, TemplateParams]]:
    """``(tier, card)`` for the line, best first: 0 = designed for the line (the
    model's card, then the line's shape), 1 = a key-phrase card."""
    designed = designed_cards(item.line)
    generic = [card for card in deterministic_drafts(item.line) if card not in designed]
    if item.authored_by == "model":
        designed = [item.params, *(card for card in designed if card != item.params)]
        generic = [card for card in generic if card != item.params]
    return [*((0, card) for card in designed), *((1, card) for card in generic)]


def choose_items(proposed: Sequence[Authored]) -> list[Authored]:
    """Settle every line's card, in script order, under rules 3 and 4.

    Per line, among its cards whose template has carried fewer than
    :data:`MAX_TEMPLATE_USES` lines so far: a card designed for the line beats a
    key-phrase card; within that, a template the film has not used yet beats one
    it has; within that, the author's order. A card whose foreground would rest
    past :data:`MAX_STATIC_SECONDS` loses to any that keeps moving. The closing
    beat's template has one use reserved from the start.

    Nothing is traded away silently: when every card of a line is over the
    template limit (a film longer than the vocabulary can carry), or would rest
    too long, the least bad one is kept and ``Authored.repairs`` says
    ``NOT repaired``.
    """
    used: Counter[str] = Counter()
    closing = next((item for item in proposed if item.line.is_cta), None)
    if closing is not None:
        used[CLOSING_TEMPLATE] += 1
    chosen: list[Authored] = []
    for item in proposed:
        if item is closing:
            used[CLOSING_TEMPLATE] -= 1  # its own reserved use
        options = _options(item)
        ranked = sorted(
            (
                (
                    used[card.template_id] >= MAX_TEMPLATE_USES,
                    _resting(item.line, card) > MAX_STATIC_SECONDS,
                    tier,
                    used[card.template_id],
                    order,
                )
                for order, (tier, card) in enumerate(options)
            ),
        )
        over_limit, rests, _tier, uses, order = ranked[0]
        params = options[order][1]
        settled = Authored(
            line=item.line,
            params=params,
            authored_by=item.authored_by if params == item.params else "fallback",
            rejections=list(item.rejections),
            repairs=list(item.repairs),
        )
        _ok, unfit = check_draft(item.line, (params.template_id, params.model_dump(by_alias=True)))
        if unfit is not None:
            settled.repairs.append(f"NOT repaired: no card fits the line — shown whole ({unfit})")
        if params != item.params:
            was = item.params.template_id
            if used[was] >= MAX_TEMPLATE_USES:
                why = f"{was} already carries {used[was]} lines of the film"
            elif _resting(item.line, item.params) > MAX_STATIC_SECONDS:
                why = (
                    f"the proposed {was} would rest {_resting(item.line, item.params):.1f}s "
                    f"of a {item.line.shot_seconds:.1f}s shot"
                )
            else:
                why = f"{was} is already in the film and {params.template_id} is not"
            settled.repairs.append(f"swapped to a rule-built {params.template_id}: {why}")
        if over_limit:
            settled.repairs.append(
                f"NOT repaired: {params.template_id} now carries {uses + 1} lines "
                f"(limit {MAX_TEMPLATE_USES}) — no other template fits this line"
            )
        if rests:
            settled.repairs.append(
                f"NOT repaired: foreground rests {_resting(item.line, params):.1f}s of a "
                f"{item.line.shot_seconds:.1f}s shot (limit {MAX_STATIC_SECONDS}s)"
            )
        used[params.template_id] += 1
        chosen.append(settled)
    return chosen


def author_items(
    lines: Sequence[ScriptLine], proposals: Mapping[str, Draft] | None
) -> list[Authored]:
    """Every line's final card. ``proposals`` is the model's answer by line id
    (``None`` = the model was not available: every line is authored by rule).
    The closing beat is never the model's: it is the brand over the call to action."""
    proposed: list[Authored] = []
    for line in lines:
        draft = (proposals or {}).get(line.id)
        if proposals is None or line.is_cta:
            proposed.append(Authored(line, deterministic_drafts(line)[0], "fallback"))
            continue
        params, reason = (
            check_draft(line, draft) if draft else (None, "the model returned no card for the line")
        )
        if params is not None:
            proposed.append(Authored(line, params, "model"))
        else:
            template = draft[0] if draft else "none"
            proposed.append(
                Authored(
                    line,
                    deterministic_drafts(line)[0],
                    "fallback",
                    rejections=[f"{template}: {reason}"],
                )
            )
    return choose_items(proposed)


# ── the model ───────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a motion designer cutting a short explainer film. For every narration line "
    "you design ONE on-screen card from a fixed set of animated templates. A card is a "
    "DESIGN of the line's idea — a figure on a ladder, a choice, a sequence, a checklist, "
    "a hand-off — never the sentence written out. Return only JSON matching the schema."
)
_TEMPLATE_GUIDE = f"""Templates (use only the fields listed for the template you pick):
- two_beat: a headline, then a second emphasised beat, beside a dial with an icon at its \
centre. title (2-6 words), emphasis (optional, 2-4 words, the payoff), icon (REQUIRED). For a \
tagline or a line with one idea.
- levels: a big figure beside a ladder that fills from its lowest to its highest level. value \
(the figure, in digits), suffix (its sign, e.g. "+" for "over N"), label (the noun after the \
figure), from_label, to_label (the two ends of the range, 1-2 words each — ONLY when the line \
names them; leave both out otherwise), title (optional). For a number with a range ("from A to \
B") or with graded levels.
- two_up: two panels. left{{title, icon, state}}, right{{title, icon, state}} with state \
"chosen" or "rejected" (or no state: two parts side by side), title (optional headline). For \
"X, not Y", and for "X while you Y" (the system's part beside yours).
- timeline: a track of 2-6 steps an arrow travels. steps[{{title, icon}}], title (optional). \
For a sequence ("A before B") or a list.
- checklist: 2-6 rows, each ticked or crossed. items[{{label, state}}] with state "yes" (done / \
used) or "no" (avoided / refused), title (optional). For what is done and what is not.
- handoff: a source node, a labelled route, a destination node. from{{label, icon}}, \
route{{label}}, to{{label, icon}}, title (optional). For something routed / escalated / sent to \
someone, or made from an input ("builds X tailored to Y": Y → tailored → X).

Icons (optional, by name; any other name is dropped): {", ".join(sorted(ICON_NAMES))}.

Hard rules — a card that breaks one is thrown away:
1. Every word and number on the card must be a word or number of THAT line. No synonyms, no \
added claims, no brand name.
2. Never show the whole sentence: use at most about two thirds of the line's words — keywords.
3. Every phrase must read as a phrase: copy a run of the line's words AS WRITTEN — never drop \
words from the middle ("Merqi runs motion" is wrong); never start on "and", "while", "before", \
"ever"; never end on "the", "to", "tailored", "financial". A noun phrase keeps its own modifiers.
4. Every card shows a graphic: two_beat always carries an icon; animated_title_card is only for \
the call to action.
5. No template on more than two lines of the film; prefer a different template for every line.
6. Never use kinetic_typography or pipeline_story.
7. The last line (the call to action) is designed for you; whatever you return for it is ignored."""
_TEXT: dict[str, object] = {"type": "string"}
_ICON: dict[str, object] = {"type": "string", "enum": ["", *sorted(ICON_NAMES)]}
_PANEL: dict[str, object] = {
    "type": "object",
    "properties": {
        "title": _TEXT,
        "icon": _ICON,
        "state": {"type": "string", "enum": ["chosen", "rejected"]},
    },
}
_NODE: dict[str, object] = {"type": "object", "properties": {"label": _TEXT, "icon": _ICON}}
#: One flat item shape for every template: a constrained decoder gets a grammar
#: it can build, and :func:`_draft_from` keeps only what the item actually set.
OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "line_id": _TEXT,
                    "template_id": {
                        "type": "string",
                        "enum": sorted(set(PARAM_MODELS) - {"section_divider", CLOSING_TEMPLATE}),
                    },
                    "icon": _ICON,
                    **dict.fromkeys(
                        (
                            "title", "kicker", "emphasis", "subtitle", "value", "suffix",
                            "label", "from_label", "to_label",
                        ),
                        _TEXT,
                    ),
                    "left": _PANEL,
                    "right": _PANEL,
                    "steps": {
                        "type": "array",
                        "items": {"type": "object", "properties": {"title": _TEXT, "icon": _ICON}},
                    },
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": _TEXT,
                                "state": {"type": "string", "enum": ["yes", "no"]},
                            },
                        },
                    },
                    "from": _NODE,
                    "route": {"type": "object", "properties": {"label": _TEXT}},
                    "to": _NODE,
                },
                "required": ["line_id", "template_id"],
            },
        }
    },
    "required": ["items"],
}  # fmt: skip
_NOT_CONTENT = frozenset({"line_id", "template_id", "params"})
_MAX_TOKENS = output_token_budget(answer_tokens=MEDIUM_ANSWER_TOKENS)


def build_message(lines: Sequence[ScriptLine]) -> str:
    """The user message: the template guide and the lines, with their ids and length."""
    block = "\n".join(
        f'- line_id {line.id} ({line.shot_seconds:.1f}s on screen): "{line.text}"' for line in lines
    )
    return f"{_TEMPLATE_GUIDE}\n\nDesign one card per line.\n\nLines:\n{block}"


def _pruned(value: object) -> object:
    """``value`` without empty strings / lists / objects (``None`` when nothing is left)."""
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, Mapping):
        kept = {str(k): p for k, v in value.items() if (p := _pruned(v)) is not None}
        return kept or None
    if isinstance(value, list):
        kept_items = [p for v in value if (p := _pruned(v)) is not None]
        return kept_items or None
    return value


def _draft_from(item: Mapping[str, object]) -> Draft:
    """One model item → a draft. Fields may be flat or under ``params``; a figure
    may come as a number; ``two_up``'s sides may come nested (as the recorder
    takes them). Empty fields are dropped; anything left that the template does
    not read is rejected by its param model."""
    nested = item.get("params")
    flat = {key: value for key, value in item.items() if key not in _NOT_CONTENT}
    merged = _pruned({**flat, **(nested if isinstance(nested, Mapping) else {})})
    raw = dict(merged) if isinstance(merged, Mapping) else {}
    if isinstance(raw.get("value"), (int, float)) and not isinstance(raw.get("value"), bool):
        raw["value"] = f"{raw['value']:g}"
    sides = raw.pop("two_up", None)
    if isinstance(sides, Mapping):
        raw = {**raw, **{k: v for k, v in sides.items() if k in ("left", "right")}}
    raw.pop("layout", None)
    raw.pop("style", None)
    return str(item.get("template_id") or ""), raw


def proposals_from(output: object, line_ids: set[str]) -> dict[str, Draft]:
    """The model's answer as ``{line_id: draft}`` (first card per known line)."""
    items = output.get("items") if isinstance(output, Mapping) else None
    drafts: dict[str, Draft] = {}
    for item in items if isinstance(items, list) else []:
        if isinstance(item, Mapping) and str(item.get("line_id")) in line_ids:
            drafts.setdefault(str(item.get("line_id")), _draft_from(item))
    return drafts


# ── the stage ───────────────────────────────────────────────────────────────


class PromoMotionBriefInput(StageInput):
    pass


class MotionBriefItem(StageOutput):
    id: str
    line_id: str
    template_id: str
    #: Content params plus ``marks`` (beat offsets from the narration),
    #: ``duration_ms`` (the recorder needs it in the params too) and ``film_t_ms``
    #: (where the shot starts in the film) — what ``motion_graphics_gen`` sends,
    #: under the run's brand params.
    params: dict[str, object]
    #: The WHOLE shot: narration plus the planned hold.
    duration_ms: int
    authored_by: Literal["model", "fallback"]
    #: Why a proposed card for this line was thrown away (one reason each).
    rejections: list[str] = []  # noqa: RUF012 — pydantic copies field defaults per-instance
    #: What the stage changed after authoring (a swapped template), or could not.
    repairs: list[str] = []  # noqa: RUF012 — pydantic copies field defaults per-instance


class StaticStretch(StageOutput):
    line_id: str
    template_id: str
    shot_seconds: float
    #: When the card's last beat lands (its last entrance, for a settling template).
    settles_at_seconds: float
    #: Longest stretch of the shot the plan predicts with the foreground at rest.
    #: ``0.0`` for a film template (it moves for its whole clip — documented
    #: behaviour, see motion_templates); the ambient layer is never counted.
    static_seconds: float
    #: The template keeps a foreground element moving for the whole clip.
    continuous_motion: bool
    #: The shared ambient layer drifts for the whole clip (not counted above).
    ambient_drift: bool


class ClosingBeat(StageOutput):
    """The call-to-action card — the film's last shot."""

    line_id: str
    #: The card's params exactly as authored (content, marks, film_t_ms).
    params: dict[str, object]
    #: The WHOLE closing shot, ``end_card_seconds`` included.
    duration_ms: int
    #: Seconds of the default branded end card this beat absorbed: the render
    #: extends the closing line's shot by exactly this and splices no end card.
    #: ``0.0`` — no default card (an operator's own ``title_card`` block, or the
    #: fill planned the card out) — and the end card, if any, is spliced as before.
    end_card_seconds: float = 0.0


class MotionBriefUsage(StageOutput):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0


class PromoMotionBriefOutput(StageOutputWithNotes):
    produces: ClassVar[tuple[str, ...]] = ("items",)

    items: list[MotionBriefItem]
    static_stretches: list[StaticStretch]
    #: The film's longest planned static stretch among these cards.
    max_static_seconds: float
    #: Lines whose card breaks a rule the stage could not repair, with the numbers.
    unrepaired: list[str] = []  # noqa: RUF012 — pydantic copies field defaults per-instance
    #: Length of the film these cards cover (sum of the shots), ms.
    film_ms: int = 0
    #: The closing beat (``None`` when the script does not end on its call to action).
    closing_beat: ClosingBeat | None = None
    #: Why the model's answer was not used at all (``None`` when it was).
    model_error: str | None = None
    token_usage: MotionBriefUsage = Field(default_factory=MotionBriefUsage)


def _request_authored_items(config_snapshot: Mapping[str, object]) -> bool:
    block = config_snapshot.get("motion_graphics")
    if not isinstance(block, Mapping):
        return False
    return bool(block.get("items")) or bool(block.get("items_from_stage"))


def _brand_name(config_snapshot: Mapping[str, object]) -> str:
    brand = config_snapshot.get("brand")
    return str(brand.get("name") or "").strip() if isinstance(brand, Mapping) else ""


def script_lines(ctx: RunContext) -> list[ScriptLine]:
    """The script's lines with the seconds each is SPOKEN and ON SCREEN.

    Narration length is the measured clip (``promo_narrate``), else the script's
    own estimate; the hold after it and the call-to-action floor are the same
    numbers ``promo_render`` applies (:func:`promo_fill`, ``apply_holds``). The
    closing beat is the LAST line when it is the brief's call to action; its shot
    also carries the default end card's seconds (the render splices no card then).
    """
    script = ctx.prior_output("promo_script") or {}
    narrated = {
        str(clip.get("line_id")): float(clip.get("seconds") or 0.0)
        for clip in (ctx.prior_output("promo_narrate") or {}).get("clips") or []
        if isinstance(clip, dict)
    }
    brief = ctx.config_snapshot.get("brief")
    cta = tokens(str(brief.get("call_to_action") or "")) if isinstance(brief, dict) else []
    hold = promo_fill(ctx).hold_seconds
    end_card = end_card_config(ctx)
    end_card_seconds = float(end_card["duration_seconds"]) if end_card else 0.0
    rows = [row for row in script.get("lines") or [] if isinstance(row, dict) and row.get("id")]
    lines: list[ScriptLine] = []
    for index, row in enumerate(rows):
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        spoken = narrated.get(str(row["id"])) or float(row.get("seconds") or 0.0)
        seconds = spoken + hold
        last = index == len(rows) - 1
        if last:
            seconds = max(seconds, CTA_MIN_SECONDS)
        closing = last and bool(cta) and tokens(text) == cta
        absorbed = end_card_seconds if closing else 0.0
        seconds += absorbed
        lines.append(
            ScriptLine(
                id=str(row["id"]),
                text=text,
                shot_seconds=round(seconds, 3),
                is_cta=closing,
                narration_seconds=round(spoken, 3),
                brand_name=_brand_name(ctx.config_snapshot) if closing else "",
                end_card_seconds=absorbed,
            )
        )
    return lines


def build_output(
    authored: Sequence[Authored],
    *,
    model_error: str | None = None,
    usage: MotionBriefUsage | None = None,
) -> PromoMotionBriefOutput:
    """The stage's record from the settled cards (pure — the dry run uses it too)."""
    shots = [
        PlannedShot(
            line_id=item.line.id,
            seconds=item.line.shot_seconds,
            kind="motion_card",
            clip_seconds=item.shot_ms / 1000,
            events=tuple(foreground_events(item.params, item.marks)),
            continuous=item.params.continuous_motion,
        )
        for item in authored
    ]
    per_line, longest = film_static_seconds(shots)
    items: list[MotionBriefItem] = []
    film_ms = 0
    for item in authored:
        items.append(
            MotionBriefItem(
                id=str(uuid7()),
                line_id=item.line.id,
                template_id=item.params.template_id,
                params={
                    **item.params.template_params(),
                    "marks": item.marks,
                    "duration_ms": item.shot_ms,
                    # Where this shot starts in the film: the ambient layer is a
                    # function of film time, so it runs on across every cut.
                    "film_t_ms": film_ms,
                },
                duration_ms=item.shot_ms,
                authored_by=item.authored_by,
                rejections=item.rejections,
                repairs=item.repairs,
            )
        )
        film_ms += item.shot_ms
    last = authored[-1] if authored else None
    closing = (
        ClosingBeat(
            line_id=last.line.id,
            params=items[-1].params,
            duration_ms=last.shot_ms,
            end_card_seconds=last.line.end_card_seconds,
        )
        if last is not None and last.line.is_cta and last.params.template_id == CLOSING_TEMPLATE
        else None
    )
    return PromoMotionBriefOutput(
        items=items,
        static_stretches=[
            StaticStretch(
                line_id=item.line.id,
                template_id=item.params.template_id,
                shot_seconds=item.line.shot_seconds,
                settles_at_seconds=max(
                    [*item.marks, *(end for _start, end in shot.events)], default=0
                )
                / 1000,
                static_seconds=per_line[item.line.id],
                continuous_motion=item.params.continuous_motion,
                ambient_drift=item.params.ambient_drift,
            )
            for item, shot in zip(authored, shots, strict=True)
        ],
        max_static_seconds=longest,
        unrepaired=[
            f"line {item.line.id}: {repair}"
            for item in authored
            for repair in item.repairs
            if repair.startswith("NOT repaired")
        ],
        film_ms=film_ms,
        closing_beat=closing,
        model_error=model_error,
        token_usage=usage or MotionBriefUsage(),
    )


def _usage_from(metrics: object) -> MotionBriefUsage:
    """One WM call's ``usage_metrics`` (an untyped blob) as counted tokens."""
    usage = metrics if isinstance(metrics, Mapping) else {}

    def _count(*keys: str) -> int:
        for key in keys:
            raw = usage.get(key)
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                return int(raw)
        return 0

    return MotionBriefUsage(
        prompt_tokens=_count("prompt_tokens", "input_tokens"),
        completion_tokens=_count("completion_tokens", "output_tokens"),
        total_tokens=_count("total_tokens", "tokens"),
        calls=1,
    )


class PromoMotionBriefStage(StageHandler[PromoMotionBriefInput, PromoMotionBriefOutput]):
    name = STAGE_NAME
    input_model = PromoMotionBriefInput
    output_model = PromoMotionBriefOutput
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (SlotUse("llm"),)

    def __init__(self, client: WMClient | None = None) -> None:
        self._llm_timeout = llm_budget_seconds()
        self._llm_retries = llm_retries()
        self._wm = client or WMClient(timeout=llm_client_timeout_seconds())

    async def handle(
        self,
        payload: PromoMotionBriefInput,
        ctx: RunContext,
    ) -> PromoMotionBriefOutput:
        if not motion_per_line_requested(ctx.config_snapshot):
            raise StageSkippedError(
                "promo_motion_brief: direction.motion is not 'per_line' (opt-in stage not requested)",
                missing=["direction.motion"],
            )
        if _request_authored_items(ctx.config_snapshot):
            raise StageSkippedError(
                "promo_motion_brief: the request authored its own motion_graphics items; "
                "those are rendered as given",
                missing=[],
            )
        lines = script_lines(ctx)
        if not lines:
            raise StagePreconditionError(
                "promo_motion_brief: promo_script produced no lines to design cards for",
                missing=["promo_script.lines"],
            )

        usage = MotionBriefUsage()
        message = build_message(lines)
        line_ids = {line.id for line in lines}

        async def _attempt(enable_reasoning: bool | None) -> dict[str, Draft]:
            nonlocal usage
            resp = await self._wm.message(
                enable_reasoning=enable_reasoning,
                worker_name="promo-motion-brief",
                message=message,
                system_prompt=_SYSTEM_PROMPT,
                output_schema=OUTPUT_SCHEMA,
                max_tokens=_MAX_TOKENS,
            )
            call = _usage_from(resp.usage_metrics)
            usage = MotionBriefUsage(
                prompt_tokens=usage.prompt_tokens + call.prompt_tokens,
                completion_tokens=usage.completion_tokens + call.completion_tokens,
                total_tokens=usage.total_tokens + call.total_tokens,
                calls=usage.calls + 1,
            )
            drafts = proposals_from(resp.output, line_ids)
            if not drafts:
                raise ValueError("promo_motion_brief: the model returned no card for any line")
            return drafts

        proposals: dict[str, Draft] | None
        model_error: str | None = None
        try:
            # Soft-fail: guarded() makes a permanent WM failure reach the
            # deterministic author after ONE call, not the whole retry ladder.
            proposals = await stage_call(
                downshift_reasoning(lambda think: guarded(lambda: _attempt(think))),
                timeout_seconds=self._llm_timeout,
                retries=self._llm_retries,
                retry_on=LLM_RETRY_ON,
                stage_name=self.name,
            )
        except Exception as exc:  # noqa: BLE001 -- soft-fail: every line is authored by rule instead
            log.warning("promo_motion_brief: model unavailable, authoring by rule: %s", exc)
            proposals, model_error = None, f"{type(exc).__name__}: {exc}"

        return build_output(author_items(lines, proposals), model_error=model_error, usage=usage)
