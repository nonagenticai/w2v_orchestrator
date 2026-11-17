"""promo_motion_brief — one designed card per line, nothing on it the line does not say.

The fixture is the real script, narration lengths and order of run 01a11082
(2026-10-06), whose film failed frame QA: three lines typed out word by word on
one gradient, each then frozen for 3-5.6 s. The cards the rules must author for
it are the ones of the approved preview sheet (``template_params.json``,
video_production_service#188), with their timing taken from the narration.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest

from src.services.runs.run_context import RunContext
from src.services.stages.base import StageSkippedError
from src.services.stages.promo import motion_brief
from src.services.stages.promo.fill import CLOSING_TAIL_PLANNED_SECONDS
from src.services.stages.promo.headline import phrase_problem, sources_for
from src.services.stages.promo.motion_brief import (
    OUTPUT_SCHEMA,
    PromoMotionBriefStage,
    ScriptLine,
    author_items,
    build_output,
    check_draft,
    deterministic_drafts,
    fallback_card,
    foreign_tokens,
    gated,
    narration_marks,
    proposals_from,
    script_lines,
    sentence_coverage,
)
from src.services.stages.promo.motion_templates import ICON_NAMES, MAX_STATIC_SECONDS

# (line id, text, narrated seconds) — run 01a11082.
_SCRIPT = [
    ("l1", "Go-to-market that runs itself, under your control.", 2.778),
    (
        "l2",
        "Users can set graduated autonomy levels for 30+ capabilities, ranging from "
        "prohibited to automatic.",
        6.955,
    ),
    (
        "l3",
        "Critical financial and consent decisions are made by deterministic code, not "
        "language models.",
        5.982,
    ),
    (
        "l4",
        "The platform prioritizes pull-based demand generation, earning attention before "
        "requesting contact.",
        6.231,
    ),
    (
        "l5",
        "The system uses only public sources and respects robots.txt, avoiding scraped or "
        "bought lists.",
        6.580,
    ),
    (
        "l6",
        "High-ticket and sensitive conversations are automatically routed to a human agent.",
        4.876,
    ),
    ("l7", "Request access at merqi.nl", 2.039),
]
_TEXT = {line_id: text for line_id, text, _seconds in _SCRIPT}
#: The end card the closing beat absorbs: what is left of the PLANNED closing
#: tail (3.0 s cap less one 24 fps frame and 1 ms — run 01a11511) after the hold.
_END = round(CLOSING_TAIL_PLANNED_SECONDS - 2.0, 3)


def _ctx(*, direction: object = None, motion_graphics: object = None) -> RunContext:
    snapshot: dict[str, object] = {
        "brand": {"name": "Merqi"},
        "theme": {},
        "brief": {"call_to_action": "Request access at merqi.nl", "target_duration_seconds": 60},
        "deliverable": {"format": "16x9", "target_duration_seconds": 60},
        "resolution": "4k",
    }
    if direction is not None:
        snapshot["direction"] = direction
    if motion_graphics is not None:
        snapshot["motion_graphics"] = motion_graphics
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot=snapshot,
        prior_run_outputs={
            "promo_script": {
                "lines": [{"id": i, "text": t, "seconds": 3.0} for i, t, _s in _SCRIPT]
            },
            "promo_storyboard": {
                "shots": [{"id": f"s-{i}", "line_id": i, "seconds": 3.0} for i, _t, _s in _SCRIPT]
            },
            "promo_narrate": {"clips": [{"line_id": i, "seconds": s} for i, _t, s in _SCRIPT]},
        },
    )


def _lines() -> list[ScriptLine]:
    return script_lines(_ctx(direction={"motion": "per_line"}))


def _line(line_id: str) -> ScriptLine:
    return next(line for line in _lines() if line.id == line_id)


def _wm(output: object, usage: object = None) -> AsyncMock:
    client = AsyncMock()
    client.message = AsyncMock(return_value=SimpleNamespace(output=output, usage_metrics=usage))
    return client


# ── the shot each card has to fill ──────────────────────────────────────────


def test_each_line_is_on_screen_for_its_narration_plus_the_planned_hold() -> None:
    """35.4s narrated of a 60s order: the fill plans a 2.0s pause after every line,
    and the closing beat also carries the end card's 1.0s — what is left of the
    3.0s closing tail after the hold (run 01a11295: never padded to 8.0s)."""
    lines = _lines()

    assert {line.id: line.shot_seconds for line in lines} == {
        "l1": 4.778,
        "l2": 8.955,
        "l3": 7.982,
        "l4": 8.231,
        "l5": 8.58,
        "l6": 6.876,
        "l7": 4.996,
    }
    assert {line.id: line.end_card_seconds for line in lines if line.end_card_seconds} == {
        "l7": _END
    }
    assert {line.id: line.narration_seconds for line in lines} == {i: s for i, _t, s in _SCRIPT}
    # Only the closing beat knows the brand name: it is the one line that may show it.
    assert [(line.id, line.brand_name) for line in lines if line.is_cta] == [("l7", "Merqi")]
    assert {line.brand_name for line in lines if not line.is_cta} == {""}


# ── the deterministic author: THIS script → the approved sheet ──────────────

#: The content of every card of the approved preview sheet (template_params.json),
#: except line 6's source node, which the sheet labelled "Merqi" — a word line 6
#: does not say. The rules label it from the line. Line 1 also carries an icon
#: since run 01a11511: a two_beat is never type on a ring alone.
_SHEET = {
    # Lines 1, 3, 4 carry a relative clause / negation / condition (audit of
    # fac79031): main clause + that clause, each whole — or the whole line.
    "l1": (
        "two_beat",
        {
            "title": "Go-to-market",
            "emphasis": "that runs itself, under your control",
            "icon": "rocket",
        },
    ),
    "l2": (
        "levels",
        {
            "title": "Graduated autonomy levels",
            "value": "30",
            "suffix": "+",
            "label": "capabilities",
            "levels": 5,
            "from_label": "Prohibited",
            "to_label": "Automatic",
        },
    ),
    # "…, not language models": a negation with no subordinate clause, 13 words —
    # the whole line, reported NOT repaired
    "l3": (
        "two_beat",
        {
            # (split where it fits the measured budget with its 9% margin)
            "title": "Critical financial and consent decisions are made",
            "emphasis": "by deterministic code, not language models",
            "icon": "check",
        },
    ),
    # the main clause is 4 lines at the fit floor with the margin: the whole
    # line, split at its comma, reported NOT repaired
    "l4": (
        "two_beat",
        {
            "title": "The platform prioritizes pull-based demand generation",
            "emphasis": "earning attention before requesting contact",
            "icon": "eye",
        },
    ),
    "l5": (
        "checklist",
        {
            "items": [
                # "uses ONLY public sources": the limiter stays (audit of #1080)
                {"label": "Only public sources", "state": "yes"},
                {"label": "Respects robots.txt", "state": "yes"},
                {"label": "Scraped lists", "state": "no"},
                {"label": "Bought lists", "state": "no"},
            ]
        },
    ),
    "l6": (
        "handoff",
        {
            "title": "High-ticket and sensitive conversations",
            "from": {"label": "Conversations", "icon": "chat_dots"},
            "route": {"label": "Automatically routed"},
            "to": {"label": "Human agent", "icon": "user"},
        },
    ),
    "l7": (
        "animated_title_card",
        {
            "style": "film",
            "eyebrow": "",
            "chapter": "",
            "title": "Merqi",
            "subtitle": "Request access at merqi.nl",
            "brand_mark": "Merqi",
        },
    ),
}


def _content(params: dict[str, object]) -> dict[str, object]:
    return {k: v for k, v in params.items() if k not in ("marks", "duration_ms", "film_t_ms")}


def test_the_real_script_is_designed_by_rule_as_the_approved_sheet() -> None:
    out = build_output(author_items(_lines(), None))

    assert {item.line_id: (item.template_id, _content(item.params)) for item in out.items} == (
        _SHEET
    )
    assert {item.authored_by for item in out.items} == {"fallback"}
    # Reported, never hidden: line 3 shown whole; a third two_beat (lines 1, 3, 4
    # are all gated) is over the template limit.
    assert [entry.split(":", 1)[0] for entry in out.unrepaired] == ["line l3", "line l4", "line l4"]
    # The line's own words only: every string on every card is the line's (the
    # closing beat also has the brand name).
    for item in out.items:
        sayable = _TEXT[item.line_id] + (" Merqi" if item.line_id == "l7" else "")
        assert not foreign_tokens(_strings(_content(item.params)), sayable), item.line_id
    assert build_output(author_items(_lines(), None)).items[3].params == out.items[3].params


#: A different script, of the same sentence shapes in other words, and one line
#: of no shape at all (``g``).
_OTHER = [
    ("a", "Ship faster, with less risk.", 2.0),
    ("b", "You can run 12 pipelines, moving from manual to automated.", 4.0),
    ("c", "Reviews are done by senior engineers, not interns.", 3.5),
    ("d", "The pipeline validates every change, running tests before merging code.", 4.5),
    (
        "e",
        "The agent reads only public docs and respects rate limits, without stored cookies "
        "or shared passwords.",
        6.5,
    ),
    ("f", "Risky changes are always escalated to a senior reviewer.", 3.6),
    (
        "g",
        "Our platform quietly keeps thousands of small engineering teams productive through "
        "every release cycle.",
        6.0,
    ),
    ("h", "Start free at example.dev", 1.8),
]


def _other_lines() -> list[ScriptLine]:
    last = len(_OTHER) - 1
    return [
        ScriptLine(
            id=i,
            text=t,
            shot_seconds=s + 2.0,
            is_cta=index == last,
            narration_seconds=s,
            brand_name="Example" if index == last else "",
        )
        for index, (i, t, s) in enumerate(_OTHER)
    ]


def test_the_rules_are_sentence_shapes_not_this_script() -> None:
    out = build_output(author_items(_other_lines(), None))
    cards = {item.line_id: (item.template_id, _content(item.params)) for item in out.items}

    assert cards == {
        # No word of the line names a glyph: the neutral accent mark.
        # "less" is a negator: a short gated line is shown whole
        "a": ("two_beat", {"title": "Ship faster, with less risk", "icon": "spark"}),
        "b": (
            "levels",
            {
                "value": "12",
                "label": "pipelines",
                "levels": 5,
                "from_label": "Manual",
                "to_label": "Automated",
            },
        ),
        # gated (a negation / a condition): main clause + that clause, or whole
        "c": (
            "two_beat",
            {"title": "Reviews are done by senior engineers, not interns", "icon": "spark"},
        ),
        "d": (
            "two_beat",
            {
                "title": "The pipeline validates every change, running tests",
                "emphasis": "before merging code",
                "icon": "check",
            },
        ),
        "e": (
            "two_beat",
            {
                "title": "The agent reads only public docs and respects rate limits",
                "emphasis": "without stored cookies or shared passwords",
                "icon": "headset",
            },
        ),
        "f": (
            "handoff",
            {
                "title": "Risky changes",
                "from": {"label": "Changes", "icon": "bot"},
                "route": {"label": "Always escalated"},
                "to": {"label": "Senior reviewer", "icon": "user"},
            },
        ),
        # No shape: the safe fallback — the line's cleanest phrases in two_beat,
        # beside an icon from its words. It was "Platform quietly keeps thousands",
        # which stops inside "thousands OF small engineering teams" (run 01a11511).
        "g": (
            "two_beat",
            # (its earlier emphasis "Our platform" came BEFORE the title in the
            # line: a card's elements keep the line's order — audit of d6f223d9)
            # the line's first clause, whole, split where it fits the template
            # (render 01a1179d: measured two-beat budget, never a key phrase)
            {
                "title": "Our platform quietly keeps thousands of small engineering teams",
                "emphasis": "productive through every release cycle",
                "icon": "users",
            },
        ),
        "h": (
            "animated_title_card",
            {
                "style": "film",
                "eyebrow": "",
                "chapter": "",
                "title": "Example",
                "subtitle": "Start free at example.dev",
                "brand_mark": "Example",
            },
        ),
    }
    # four gated lines (a, c, d, e) plus g: two_beat over its limit, reported
    assert [entry.split(":", 1)[0] for entry in out.unrepaired] == ["line d", "line e", "line g"]


def test_a_line_of_no_shape_degrades_to_clean_phrases_in_two_beats_with_an_icon() -> None:
    line = ScriptLine(id="x", text="Calm, precise and always on your side.", shot_seconds=5.0)

    designed = deterministic_drafts(line)
    assert designed[0].template_id == "two_beat"
    card = fallback_card(line)
    assert card.template_id == "two_beat"
    assert card.template_params()["icon"] == "spark"  # no word names a glyph
    # a short line (<= 8 words) is its own headline, whole (render 01a1179d)
    assert card.screen_text() == ["Calm, precise and always on your side"]
    for text in card.screen_text():
        assert phrase_problem(text, sources_for(line.text)) is None, text


def test_every_card_the_author_can_build_passes_the_rules() -> None:
    for line in [*_lines(), *_other_lines()]:
        drafts = deterministic_drafts(line)
        assert drafts, line.text
        for params in drafts:
            assert line.is_cta or params.graphic, (line.text, params)
            if gated(line.text) and len(drafts) == 1:
                continue  # the whole-line card of a gated line: reported, not "passing"
            raw = params.model_dump(by_alias=True)
            checked, reason = check_draft(line, (params.template_id, raw))
            assert reason is None, (line.text, params.template_id, reason)
            assert checked is not None
            assert params.template_id not in ("kinetic_typography", "pipeline_story")


# ── rule 1: the line's own words only ───────────────────────────────────────


def test_a_card_with_a_word_the_line_does_not_say_is_replaced() -> None:
    proposals = {
        "l6": (
            "handoff",
            {
                "from": {"label": "Merqi", "icon": "bot"},
                "route": {"label": "Automatically routed"},
                "to": {"label": "Human agent"},
            },
        )
    }

    item = author_items([_line("l6")], proposals)[0]

    assert item.authored_by == "fallback"
    assert item.rejections == ["handoff: on-screen text not from the line: merqi"]
    assert item.params.template_id == "handoff"
    assert item.params.screen_text()[1] == "Conversations"
    assert not foreign_tokens(item.params.screen_text(), _TEXT["l6"])


@pytest.mark.parametrize(("suffix", "accepted"), [("+", True)])
def test_a_figure_keeps_the_suffix_the_line_gives_it(suffix: str, accepted: bool) -> None:
    """The line says "30+": "30+" and "30" may be shown, "30%" may not."""
    draft = (
        "levels",
        {"value": "30", "suffix": suffix, "from_label": "Prohibited", "to_label": "Automatic"},
    )

    params, reason = check_draft(_line("l2"), draft)

    assert (params is not None) is accepted
    assert (reason is None) is accepted


def test_a_number_the_line_does_not_contain_is_rejected() -> None:
    draft = ("levels", {"value": "3", "from_label": "Prohibited", "to_label": "Automatic"})

    params, reason = check_draft(_line("l2"), draft)

    assert params is None
    assert reason == "on-screen text not from the line: 3"


def test_the_brand_name_may_appear_on_the_closing_beat_only() -> None:
    rejected, reason = check_draft(_line("l1"), ("two_beat", {"title": "Merqi runs itself"}))
    closing, none = check_draft(
        _line("l7"), ("animated_title_card", {"title": "Merqi", "subtitle": "Request access"})
    )
    more, why = check_draft(
        _line("l7"), ("animated_title_card", {"title": "Merqi platform", "subtitle": "Request"})
    )

    assert rejected is None
    assert reason == "on-screen text not from the line: merqi"
    assert closing is not None and none is None
    assert more is None and why == "on-screen text not from the line: platform"


# ── rule 2: never the sentence typed out ────────────────────────────────────


def test_a_card_that_shows_the_whole_sentence_is_rejected() -> None:
    """Rule 2 on a type card that is NOT one clause split in two (a clause split
    is the whole-clause rule's two-line card, audit of #1084)."""
    from src.services.stages.promo.motion_brief import typed_out
    from src.services.stages.promo.motion_templates import TwoBeatParams

    line = _line("l5")
    card = TwoBeatParams(
        title="The system uses only public sources and respects robots.txt",
        emphasis="avoiding scraped or bought lists",
        icon="globe",
    )

    assert typed_out(card, line.text) == "shows 100% of the line's words — the sentence typed out"
    assert sentence_coverage(["Only public sources", "Respects robots.txt"], line.text) < 0.8


def test_swapping_the_two_halves_of_the_sentence_does_not_get_it_past_the_rule() -> None:
    draft = (
        "two_beat",
        {
            "title": "Avoiding scraped or bought lists",
            "emphasis": "the system uses only public sources and respects robots.txt",
            "icon": "globe",
        },
    )

    params, reason = check_draft(_line("l5"), draft)

    assert params is None
    # refused (reversed order / a second beat that is not a whole clause)
    assert reason is not None


def test_a_tagline_may_be_shown_whole_in_two_beats() -> None:
    """Six words are a headline, not a sentence being typed: two beats carry all of it."""
    line = _phrase_line("t", "Built for operators, calm by default.")
    draft = (
        "two_beat",
        {"title": "Built for operators", "emphasis": "calm by default", "icon": "users"},
    )

    assert check_draft(line, draft)[1] is None


@pytest.mark.parametrize("template_id", ["kinetic_typography"])
def test_the_templates_that_cannot_be_designed_are_never_authored(template_id: str) -> None:
    params, reason = check_draft(_line("l1"), (template_id, {"words": ["Go-to-market"]}))

    assert params is None
    assert reason is not None and reason.startswith(f"template '{template_id}' may not be authored")


def test_a_param_the_template_does_not_read_is_rejected() -> None:
    draft = ("two_beat", {"title": "Graduated autonomy levels", "subtitle": "30+"})

    params, reason = check_draft(_line("l2"), draft)

    assert params is None
    assert reason == "params do not fit two_beat: subtitle: Extra inputs are not permitted"


# ── rule 3: no template on more than two lines ──────────────────────────────


def _phrase_line(line_id: str, text: str) -> ScriptLine:
    return ScriptLine(id=line_id, text=text, shot_seconds=4.0, narration_seconds=2.0)


def test_no_template_carries_a_third_line_even_when_the_uses_are_apart() -> None:
    """The old rule only forbade three in a row; A, B, A, C, A passed it."""
    lines = [
        _phrase_line("s1", "Under your control."),
        _line("l5"),
        _phrase_line("s2", "Built for operators."),
        _line("l6"),
        _phrase_line("s3", "Calm by default."),
    ]
    proposals = {
        "s1": ("two_beat", {"title": "Under your control", "icon": "user"}),
        "s2": ("two_beat", {"title": "Built for operators", "icon": "users"}),
        "s3": ("two_beat", {"title": "Calm by default", "icon": "spark"}),
    }

    items = author_items(lines, proposals)

    templates = [item.params.template_id for item in items]
    # The title card is the closing beat's alone (it is type on a ring), so a
    # third plain phrase cannot be moved to it: it keeps two_beat and that is
    # REPORTED, never traded for a card without a graphic (run 01a11511).
    assert templates == ["two_beat", "checklist", "two_beat", "handoff", "two_beat"]
    assert [item.authored_by for item in items] == [
        "model",
        "fallback",
        "model",  # a second two_beat is allowed (the line has no other shape)
        "fallback",
        "model",  # no card of another template has a graphic for "Calm by default"
    ]
    assert items[2].repairs == []
    assert items[4].repairs == [
        "NOT repaired: two_beat now carries 3 lines (limit 2) — no other template fits this line"
    ]


def test_an_unused_template_is_preferred_to_a_second_use() -> None:
    """The model's second two_beat is valid, but the line's shape asks for levels."""
    proposals = {
        "l1": ("two_beat", {"title": "Go-to-market that runs itself", "icon": "rocket"}),
        "l2": (
            "two_beat",
            {"title": "Graduated autonomy levels", "icon": "gauge"},
        ),
    }

    items = author_items([_line("l1"), _line("l2")], proposals)

    assert [item.params.template_id for item in items] == ["two_beat", "levels"]
    assert items[1].repairs == [
        "swapped to a rule-built levels: two_beat is already in the film and levels is not"
    ]


def test_the_closing_beat_keeps_its_template_however_often_the_film_wanted_it() -> None:
    lines = [_phrase_line(f"s{i}", f"Plain phrase number {word}.") for i, word in enumerate("abcd")]
    proposals = {
        line.id: ("two_beat", {"title": f"Phrase number {'abcd'[int(line.id[1])]}"})
        for line in lines
    }

    items = author_items([*lines, _line("l7")], proposals)

    templates = [item.params.template_id for item in items]
    assert templates.count("animated_title_card") == 1
    assert templates[-1] == "animated_title_card"
    assert items[-1].params.screen_text() == ["Merqi", "Request access at merqi.nl"]
    # Four phrase lines with no shape: two_beat (with an icon) is the only card
    # with a graphic they can have. The title card and the divider are type on a
    # ring, so they are never given to a line; the third and fourth uses are
    # reported, not hidden.
    assert templates[:4] == ["two_beat"] * 4
    assert all(item.params.graphic for item in items[:4])
    assert [bool(item.repairs) for item in items[:4]] == [False, False, True, True]
    assert items[3].repairs[-1].startswith("NOT repaired: two_beat now carries 4 lines")


def test_the_real_script_uses_no_template_more_than_twice() -> None:
    """Or reports it: three of this script's lines are gated (audit of fac79031)
    and can only be two_beat cards — the third use is NOT repaired, on record."""
    for proposals in (None, {}):  # no model at all / a model that answered nothing
        items = author_items(_lines(), proposals)
        templates = [item.params.template_id for item in items]
        for template in set(templates):
            over = [item for item in items if item.params.template_id == template][2:]
            assert all(any(r.startswith("NOT repaired") for r in item.repairs) for item in over)


# ── timing: from the narration, never a delay ───────────────────────────────


def test_each_beat_lands_where_its_phrase_is_spoken() -> None:
    """Word index of the phrase / the line's word count x the line's narration."""
    out = build_output(author_items(_lines(), None))

    assert {item.line_id: item.params["marks"] for item in out.items} == {
        # "that" is word 1 of 7, spoken over 2.778s
        "l1": [0, 397],
        # "30+" 7/14, "prohibited" 11/14, "automatic." 13/14 of 6.955s
        "l2": [0, 3478, 5465, 6458],
        # the whole line: its second half at "by" 7/13 of 5.982s
        "l3": [0, 3221],
        # its second half at "earning" 6/11 of 6.231s
        "l4": [0, 3399],
        # "only" 3/14, "respects" 7/14 (its label is the line's), "scraped" 10/14 ...
        "l5": [0, 1410, 3290, 4700, 5640],
        # route at "automatically" 5/11, destination at "human" 9/11 of 4.876s
        "l6": [0, 2216, 3989],
        "l7": [0, 0, 0, 0],
    }
    for item, line in zip(out.items, _lines(), strict=True):
        marks = item.params["marks"]
        assert isinstance(marks, list)
        assert marks[0] == 0  # first content at t=0
        assert marks == sorted(marks)
        # never in the hold: no beat after the narration ends
        assert max(marks) <= round(line.narration_seconds * 1000)
        assert item.params["duration_ms"] == item.duration_ms  # the whole shot


def test_a_beat_whose_phrase_is_not_spoken_opens_with_the_beat_before_it() -> None:
    from src.services.stages.promo.motion_templates import TwoBeatParams

    line = _phrase_line("x", "Fast and calm.")

    assert narration_marks(TwoBeatParams(title="Fast", emphasis="Never late"), line) == [0, 0]
    assert narration_marks(TwoBeatParams(title="Fast", emphasis="Calm"), line) == [0, 1333]


def test_every_card_starts_where_its_shot_starts_in_the_film() -> None:
    out = build_output(author_items(_lines(), None))

    assert [item.params["film_t_ms"] for item in out.items] == [
        0,
        4778,
        13733,
        21715,
        29946,
        38526,
        45402,
    ]
    # The CTA line's shot IS the end card: 4039 ms of line and hold plus the
    # 957 ms the fill plan gave the card (a 2.957 s planned closing tail, so the
    # DELIVERED tail stays <= 3.0 s on any frame grid), one clip.
    assert out.film_ms == 50398
    assert out.closing_beat is not None
    assert out.closing_beat.line_id == "l7"
    assert out.closing_beat.duration_ms == 4996
    assert out.closing_beat.end_card_seconds == _END
    assert out.closing_beat.params == out.items[-1].params


# ── rule 4: motion through the hold ─────────────────────────────────────────


def test_every_clip_lasts_its_whole_shot_and_the_film_never_rests() -> None:
    out = build_output(author_items(_lines(), None))

    assert {item.line_id: item.duration_ms for item in out.items} == {
        "l1": 4778,
        "l2": 8955,
        "l3": 7982,
        "l4": 8231,
        "l5": 8580,
        "l6": 6876,
        "l7": 4996,
    }
    assert out.max_static_seconds == 0.0
    assert {s.static_seconds for s in out.static_stretches} == {0.0}
    assert all(s.continuous_motion for s in out.static_stretches)
    stretch = next(s for s in out.static_stretches if s.line_id == "l2")
    assert stretch.settles_at_seconds == pytest.approx(6.458)


def test_a_card_that_settles_is_swapped_for_one_that_keeps_moving() -> None:
    """The old divider stops once its rule is drawn: 7.2s at rest in an 8.0s shot."""
    line = _phrase_line("x", "Under your control.")
    line = ScriptLine(id="x", text=line.text, shot_seconds=8.0, narration_seconds=2.0)

    item = author_items([line], {"x": ("section_divider", {"title": "Under your control"})})[0]

    # The divider settles AND is type only: it is now refused at the check, and the
    # rule-built two_beat (with an icon) that replaces it keeps moving.
    assert item.params.template_id == "two_beat"
    assert item.params.graphic
    assert item.rejections == [
        "section_divider: section_divider as authored is type only — every card must carry "
        "a graphic"
    ]
    assert item.static_seconds <= MAX_STATIC_SECONDS


# ── the stage ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("direction", [None])
async def test_the_stage_skips_unless_the_brief_asks_for_per_line_motion(direction: object) -> None:
    wm = _wm({"items": []})

    with pytest.raises(StageSkippedError) as raised:
        await PromoMotionBriefStage(client=wm).run({}, ctx=_ctx(direction=direction))

    assert raised.value.missing == ["direction.motion"]
    assert wm.message.await_count == 0


async def test_the_stage_skips_when_the_request_authored_its_own_items() -> None:
    wm = _wm({"items": []})
    ctx = _ctx(
        direction={"motion": "per_line"},
        motion_graphics={"template_id": "stat", "items": [{"params": {"value": "30"}}]},
    )

    with pytest.raises(StageSkippedError, match="authored its own motion_graphics items"):
        await PromoMotionBriefStage(client=wm).run({}, ctx=ctx)

    assert wm.message.await_count == 0


async def test_a_valid_model_card_is_kept_and_an_invalid_one_replaced() -> None:
    wm = _wm(
        {
            "items": [
                {
                    "line_id": "l3",
                    "template_id": "two_up",
                    "title": "",
                    "left": {"title": "Deterministic code", "icon": "braces", "state": "chosen"},
                    "right": {"title": "Language models", "icon": "sparkles", "state": "rejected"},
                    "steps": [],
                },
                {
                    "line_id": "l4",
                    "template_id": "timeline",
                    "steps": [{"title": "Earning attention"}, {"title": "AI outreach"}],
                },
                {
                    "line_id": "l6",
                    "template_id": "handoff",
                    "from": {"label": "Conversations", "icon": "chat_dots"},
                    "route": {"label": "Automatically routed"},
                    "to": {"label": "Human agent", "icon": "sparkles"},
                },
                {"line_id": "l7", "template_id": "two_beat", "title": "Request access"},
                {"line_id": "not-a-line", "template_id": "two_beat", "title": "x"},
            ]
        },
        usage={"prompt_tokens": 900, "completion_tokens": 210, "total_tokens": 1110},
    )

    out = await PromoMotionBriefStage(client=wm).run({}, ctx=_ctx(direction={"motion": "per_line"}))

    by_line = {item.line_id: item for item in out.items}
    assert list(by_line) == [line_id for line_id, _t, _s in _SCRIPT]
    assert by_line["l6"].authored_by == "model"
    # An icon the recorder does not have is dropped, never sent.
    assert by_line["l6"].params["to"] == {"label": "Human agent"}
    # Line 3 has a negation: the model's panels are refused (audit of fac79031).
    assert by_line["l3"].authored_by == "fallback"
    assert by_line["l3"].rejections == [
        "two_up: two_up: a line with a condition, relative clause or negation is shown as "
        "its main clause + that clause, each whole"
    ]
    assert by_line["l4"].authored_by == "fallback"
    assert by_line["l4"].rejections == ["timeline: on-screen text not from the line: ai, outreach"]
    assert by_line["l1"].rejections == ["none: the model returned no card for the line"]
    # The closing beat is fixed: the model's card for it is never used.
    assert _content(by_line["l7"].params) == _SHEET["l7"][1]
    assert out.model_error is None
    assert out.token_usage.model_dump() == {
        "prompt_tokens": 900,
        "completion_tokens": 210,
        "total_tokens": 1110,
        "calls": 1,
    }
    for item in out.items:
        sayable = _TEXT[item.line_id] + (" Merqi" if item.line_id == "l7" else "")
        assert not foreign_tokens(_strings(_content(item.params)), sayable), item
    sent = wm.message.await_args.kwargs
    assert sent["output_schema"] is OUTPUT_SCHEMA
    assert all(f"line_id {line_id}" in sent["message"] for line_id, _t, _s in _SCRIPT)


def test_the_model_is_offered_exactly_the_film_kit_and_the_recorders_icons() -> None:
    schema: Any = OUTPUT_SCHEMA
    item = schema["properties"]["items"]["items"]
    # The title card is the closing beat's alone: never offered for a line.
    assert item["properties"]["template_id"]["enum"] == [
        "checklist",
        "handoff",
        "levels",
        "timeline",
        "two_beat",
        "two_up",
    ]
    icons = item["properties"]["left"]["properties"]["icon"]["enum"]
    assert set(icons) == {"", *ICON_NAMES}


def _strings(value: object) -> list[str]:
    """Every TEXT string a params blob carries, however deeply nested (not the
    enum-like ``state`` / ``icon`` / ``layout`` / ``style`` values)."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [
            text
            for key, child in value.items()
            if key not in ("state", "icon", "layout", "style", "brand_mark")
            for text in _strings(child)
        ]
    if isinstance(value, list):
        return [text for child in value for text in _strings(child)]
    return []


@pytest.mark.parametrize(
    "failure",
    [httpx.ConnectError("worker_management unreachable")],
)
async def test_a_model_failure_never_fails_the_run(
    failure: Exception, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(motion_brief, "llm_retries", lambda: 0)
    wm = AsyncMock()
    wm.message = AsyncMock(side_effect=failure)

    out = await PromoMotionBriefStage(client=wm).run({}, ctx=_ctx(direction={"motion": "per_line"}))

    assert [item.template_id for item in out.items] == [t for t, _p in _SHEET.values()]
    assert {item.authored_by for item in out.items} == {"fallback"}
    assert out.model_error is not None and type(failure).__name__ in out.model_error
    assert out.notes == []  # the output is the record; the run is not degraded


async def test_an_answer_with_no_usable_card_is_a_model_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(motion_brief, "llm_retries", lambda: 0)

    out = await PromoMotionBriefStage(client=_wm("I could not produce JSON")).run(
        {}, ctx=_ctx(direction={"motion": "per_line"})
    )

    assert len(out.items) == len(_SCRIPT)
    assert out.model_error is not None
    assert out.model_error.endswith("the model returned no card for any line")
    assert out.token_usage.calls == 1


def test_a_nested_params_answer_and_a_numeric_figure_are_read() -> None:
    """A model that was not grammar-constrained answers ``params: {...}``, ``30``,
    and ``two_up`` sides nested the way the recorder takes them."""
    output = {
        "items": [
            {
                "line_id": "l2",
                "template_id": "levels",
                "params": {
                    "value": 30,
                    "suffix": "+",
                    "from_label": "Prohibited",
                    "to_label": "Automatic",
                    "kicker": " ",
                },
            },
            {
                "line_id": "l3",
                "template_id": "two_up",
                "params": {
                    "layout": "panels",
                    "two_up": {"left": {"title": "Deterministic code"}, "right": {"title": "x"}},
                },
            },
        ]
    }

    drafts = proposals_from(output, {"l2", "l3"})

    assert drafts["l2"] == (
        "levels",
        {"value": "30", "suffix": "+", "from_label": "Prohibited", "to_label": "Automatic"},
    )
    assert drafts["l3"] == (
        "two_up",
        {"left": {"title": "Deterministic code"}, "right": {"title": "x"}},
    )
    assert check_draft(_line("l2"), drafts["l2"])[1] is None


# ── one closing beat: the CTA card IS the end card ──────────────────────────


def _vp() -> AsyncMock:
    client = AsyncMock()
    client.record_animation = AsyncMock(return_value="job-end")
    client.wait_for_content_recording = AsyncMock(return_value={"status": "completed"})
    client.get_recording_video = AsyncMock(return_value=(b"\x00mp4", "video/mp4", None))
    return client


async def _end_card(
    monkeypatch: pytest.MonkeyPatch, ctx: RunContext
) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(record_animation kwargs, outro shot)`` of the promo's end card, rendered
    exactly as the promo compile renders it (its config, plan and settled lead)."""
    from src.services.stages.promo.render import PromoRenderStage
    from src.services.stages.shared import title_cards

    monkeypatch.setattr(title_cards, "find_existing_media", AsyncMock(return_value=None))
    monkeypatch.setattr(
        title_cards, "store_generated_video", AsyncMock(return_value="https://m/end.mp4")
    )
    stage = PromoRenderStage()
    client = _vp()
    _intro, outro = await title_cards.render_title_card_plan(
        ctx,
        stage.title_card_plan(ctx),
        client=client,
        cfg=stage.title_card_config(ctx),
        settled_lead_seconds=stage.title_card_settled_lead_seconds(ctx),
    )
    assert outro is not None
    return client.record_animation.await_args.kwargs, outro


async def _with_brief(ctx: RunContext) -> RunContext:
    out = await PromoMotionBriefStage(client=_wm({"items": []})).run({}, ctx=ctx)
    ctx.prior_run_outputs["promo_motion_brief"] = out.model_dump(mode="json")
    return ctx


async def _brief_and_cards(ctx: RunContext, recorded: list[dict[str, Any]]) -> None:
    """Run the brief and motion_graphics_gen on ``ctx`` as the DAG would, before
    the render; ``recorded`` gets every ``record_animation`` call of the cards."""
    from src.services.stages.generic.motion_graphics_gen import MotionGraphicsGenStage

    await _with_brief(ctx)
    client = _vp()

    async def _record(**kw: Any) -> str:
        recorded.append(kw)
        return f"job-{kw['template_id']}-{len(recorded)}"

    client.record_animation = AsyncMock(side_effect=_record)
    out = await MotionGraphicsGenStage(client=client).run({}, ctx=ctx)
    ctx.prior_run_outputs["motion_graphics_gen"] = out.model_dump(mode="json")


@pytest.fixture
def _gen_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = "src.services.stages.generic.motion_graphics_gen"
    monkeypatch.setattr(f"{gen}.find_existing_media", AsyncMock(return_value=None))
    monkeypatch.setattr(
        f"{gen}.store_generated_video",
        AsyncMock(side_effect=lambda **kw: f"https://minio.test/v/{kw['video_id']}.mp4"),
    )


@pytest.mark.usefixtures("_gen_storage")
async def test_the_closing_beat_is_the_end_card_one_shot_one_recording(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real PromoRenderStage.run after the brief and the card recordings: the
    CTA line's card (brand over the call to action) runs on for the ~1.0 s the fill
    gave the end card — the closing shot ends <= 3.0 s after the CTA's narration
    (run 01a11295: it used to run on for 8.0 s, 10 s of music after the last
    word). ONE closing shot, no ``__title_outro__``, no end-card recording, and
    the L7 card is recorded for that whole shot."""
    from tests.services.stages.promo.test_promo_frame_qa_2026_10_05 import (
        _LINES,
        _promo_cfg,
        _render,
    )

    recorded: list[dict[str, Any]] = []

    async def _prepare(ctx: RunContext) -> None:
        await _brief_and_cards(ctx, recorded)

    cfg = _promo_cfg(direction={"motion": "per_line"})
    _out, captured, title_vp = await _render(monkeypatch, cfg, 57.4, prepare=_prepare)
    shots = captured["shots"]

    assert [shot["line_id"] for shot in shots] == [i for i, _t, _s in _LINES]
    title_vp.record_animation.assert_not_awaited()  # no separate end card is recorded
    closing = shots[-1]
    narrated = dict((i, s) for i, _t, s in _LINES)["l7"]
    # line + 2.0 s hold + the end card, declared so the compiler honours it.
    assert closing["seconds"] == pytest.approx(narrated + 2.0 + _END)
    assert closing["max_seconds"] == pytest.approx(closing["seconds"])
    l7 = [kw for kw in recorded if kw["params"].get("subtitle") == "Request access at merqi.nl"]
    assert len(l7) == 1
    assert l7[0]["template_id"] == "animated_title_card"
    assert l7[0]["duration_ms"] == round(closing["seconds"] * 1000)
    assert l7[0]["params"]["duration_ms"] == l7[0]["duration_ms"]
    assert closing["video_url"].startswith("https://minio.test/v/")
    # The film's planned total: narration + holds + the capped card, NOT padded
    # up to the 60 s order (test_promo_closing_tail records the shortfall).
    total = sum(float(shot["seconds"]) for shot in shots)
    assert total == pytest.approx(sum(s for _i, _t, s in _LINES) + 7 * 2.0 + _END)


async def test_without_the_motion_brief_the_render_keeps_its_separate_end_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No brief: the CTA line is line + hold, then the 8.0 s end card — as before."""
    from tests.services.stages.promo.test_promo_frame_qa_2026_10_05 import (
        _LINES,
        _promo_cfg,
        _render,
    )

    _out, captured, title_vp = await _render(monkeypatch, _promo_cfg(), 57.4)
    shots = captured["shots"]

    assert [shot["line_id"] for shot in shots] == [
        *(i for i, _t, _s in _LINES),
        "__title_outro__",
    ]
    assert shots[-2]["seconds"] == pytest.approx(_LINES[-1][2] + 2.0)
    assert shots[-1]["seconds"] == pytest.approx(8.0)
    title_vp.record_animation.assert_awaited_once()
    total = sum(float(shot["seconds"]) for shot in shots)
    assert total == pytest.approx(sum(s for _i, _t, s in _LINES) + 7 * 2.0 + 8.0)


@pytest.mark.usefixtures("_gen_storage")
async def test_with_the_closing_card_laid_the_promo_plans_no_separate_end_card() -> None:
    """The end card is absorbed only when the closing SHOT carries its card —
    decided from the laid plan, never from the brief's output alone."""
    from src.services.stages.promo.render import PromoRenderStage

    ctx = _ctx(direction={"motion": "per_line"})
    for clip in ctx.prior_run_outputs["promo_narrate"]["clips"]:
        clip["artifact_url"] = f"https://m/{clip['line_id']}.wav"  # the render lays the shots
    stage = PromoRenderStage()

    # The brief planned the closing beat, but no card is recorded yet: the
    # closing shot is not its card, so the separate end card stays.
    await _with_brief(ctx)
    assert ctx.prior_run_outputs["promo_motion_brief"]["closing_beat"]["end_card_seconds"] > 0
    assert stage.title_card_plan(ctx) is not None

    await _brief_and_cards(ctx, [])
    assert stage.title_card_config(ctx) is None
    assert stage.title_card_plan(ctx) is None
    # Not the brief, the absence of a brief closing beat: the card comes back.
    ctx.prior_run_outputs["promo_motion_brief"]["closing_beat"] = None
    assert stage.title_card_plan(ctx) is not None


async def test_an_end_card_that_says_something_else_is_still_drawn_in_the_film_style(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An operator's own ``title_card`` text does not repeat the closing beat: it
    is recorded settled (1.5 s lead, all marks 0) with the film clock continued
    from the end of the last card."""
    ctx = _ctx(direction={"motion": "per_line"})
    ctx.config_snapshot["title_card"] = {
        "enabled": True,
        "intro": False,
        "outro": True,
        "outro_title": "Merqi",
        "outro_subtitle": "merqi.nl",
        "duration_seconds": 3.0,
    }
    ctx = await _with_brief(ctx)
    # An operator's own card is never absorbed: the closing beat is its line only.
    assert ctx.prior_run_outputs["promo_motion_brief"]["closing_beat"]["end_card_seconds"] == 0

    sent, shot = await _end_card(monkeypatch, ctx)

    assert shot["clip_lead_seconds"] == 1.5
    assert sent["params"]["style"] == "film"
    assert sent["params"]["subtitle"] == "merqi.nl"
    assert sent["params"]["marks"] == [0, 0, 0, 0]
    assert sent["params"]["film_t_ms"] == 49441 - 1500
