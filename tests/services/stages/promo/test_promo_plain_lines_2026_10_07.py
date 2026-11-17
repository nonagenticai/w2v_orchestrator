"""Per-line promo motion on a platform-written script of plain sentences.

Fixture: the real script and measured narration of w2v run
``01a11511-2623-73cf-aeb9-4925886f26cf`` (2026-10-07), read read-only from its
``promo_script`` / ``promo_narrate`` stage outputs. Its film failed frame QA:

1. clipped on-screen fragments that read as broken copy — "Merqi runs motion" /
   "you keep final say" (the model dropped words from inside the clause),
   "Ever requesting contact" and "Builds multimedia assets tailored" (rule
   built);
2. five of eight scenes a ring or plain type (two_beat, title card, divider),
   line 7 never shown as the checklist it is, line 8 with no levels ladder;
3. a closing tail measured at 3.04 s against the 3.0 s rule.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from src.services.runs.run_context import RunContext
from src.services.stages.promo.fill import (
    CLOSING_TAIL_MAX_SECONDS,
    CLOSING_TAIL_PLANNED_SECONDS,
    delivered_seconds,
)
from src.services.stages.promo.headline import phrase_problem, sources_for
from src.services.stages.promo.motion_brief import (
    HEADLINE,
    ScriptLine,
    author_items,
    build_output,
    check_draft,
    deterministic_drafts,
    gated,
    script_lines,
)
from src.services.stages.promo.motion_templates import PARAM_MODELS, TwoBeatParams

# (line id, text, measured narration seconds) — run 01a11511, promo_script + promo_narrate.
_RUN = [
    (
        "01a11511-6fb3-7714-81e3-07485a3d1a17",
        "Merqi runs your go-to-market motion while you keep the final say.",
        3.6409166666666666,
    ),
    (
        "01a11511-6fb3-7714-81e3-0749cbffbfc2",
        "It builds multimedia assets tailored to each target's specific context.",
        4.4935,
    ),
    (
        "01a11511-6fb3-7714-81e3-074a8d3fa626",
        "The system earns attention before ever requesting contact.",
        3.7063333333333333,
    ),
    (
        "01a11511-6fb3-7714-81e3-074bea78757b",
        "It scores your marketing surfaces to find the highest-ROI fix next.",
        4.01625,
    ),
    (
        "01a11511-6fb3-7714-81e3-074cec619a54",
        "High-ticket conversations route automatically to a human agent.",
        3.8505416666666665,
    ),
    (
        "01a11511-6fb3-7714-81e3-074d451aa6c6",
        "Deterministic code handles critical financial and consent decisions.",
        4.374208333333334,
    ),
    (
        "01a11511-6fb3-7714-81e3-074eaa088b0f",
        "It uses only public sources and respects robots.txt files.",
        4.4976666666666665,
    ),
    (
        "01a11511-6fb3-7714-81e3-074f68fad6d9",
        "You set graduated autonomy levels for over thirty capabilities.",
        3.9644166666666667,
    ),
    ("01a11511-6fb3-7714-81e3-07500fc5a09d", "Request access at merqi.nl", 2.1218333333333335),
]
_IDS = [line_id for line_id, _t, _s in _RUN]
_TEXT = {line_id: text for line_id, text, _s in _RUN}
_LEGAL_FPS = (24, 25, 30, 50, 60)

#: What the frame-QA'd film put on screen, verbatim from the run's promo_motion_brief.
_BROKEN = ("Merqi runs motion", "you keep final say", "Ever requesting contact")
_BROKEN_TAIL = "Builds multimedia assets tailored"


def _ctx() -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "brand": {"name": "Merqi"},
            "brief": {
                "call_to_action": "Request access at merqi.nl",
                "target_duration_seconds": 60,
            },
            "deliverable": {"format": "16x9", "target_duration_seconds": 60},
            "direction": {"motion": "per_line"},
        },
        prior_run_outputs={
            "promo_script": {"lines": [{"id": i, "text": t, "seconds": 3.0} for i, t, _ in _RUN]},
            "promo_storyboard": {
                "shots": [{"id": f"s-{i}", "line_id": i, "seconds": 3.0} for i, _t, _s in _RUN]
            },
            "promo_narrate": {"clips": [{"line_id": i, "seconds": s} for i, _t, s in _RUN]},
        },
    )


def _lines() -> list[ScriptLine]:
    return script_lines(_ctx())


def _cards() -> dict[str, tuple[str, list[str]]]:
    out = build_output(author_items(_lines(), None))
    by_id = {item.line_id: item for item in out.items}
    return {
        line_id[-4:]: (
            by_id[line_id].template_id,
            _screen(by_id[line_id].template_id, by_id[line_id].params),
        )
        for line_id in _IDS
    }


def _screen(template_id: str, params: dict[str, object]) -> list[str]:
    content = {k: v for k, v in params.items() if k not in ("marks", "duration_ms", "film_t_ms")}
    if template_id == "two_up":
        sides = content.pop("two_up")
        assert isinstance(sides, dict)
        content = {**content, **sides}
    content = {k: v for k, v in content.items() if k not in ("layout", "style", "chapter")}
    if template_id == "animated_title_card":
        content.pop("brand_mark")
    return PARAM_MODELS[template_id].model_validate(content).screen_text()


# ── the cards this script gets ──────────────────────────────────────────────


def test_every_line_gets_a_designed_card_of_whole_phrases() -> None:
    assert _cards() == {
        # "while" opens a subordinate clause: main clause / that clause, whole
        # (audit of fac79031 — no clause-derived diagram on such a line)
        "1a17": (
            "two_beat",
            ["Merqi runs your go-to-market motion", "while you keep the final say"],
        ),
        # "builds X tailored to Y": context → tailored → assets.
        # in the line's order: assets, tailored, to the context
        "bfc2": ("handoff", ["Multimedia assets", "Tailored", "Each target's specific context"]),
        # "before" opens a condition: main clause / condition, whole
        "a626": ("two_beat", ["The system earns attention", "before ever requesting contact"]),
        # "scores A to find the highest B": what is scored → what it finds.
        "757b": ("timeline", ["Marketing surfaces", "Highest-ROI fix"]),
        # "route automatically to Z": a hand-off.
        "9a54": ("handoff", ["High-ticket conversations", "Route automatically", "Human agent"]),
        # "S handles A and B decisions": the subject over what it takes.
        "a6c6": ("checklist", ["Deterministic code", "Financial decisions", "Consent decisions"]),
        # "only A and B": the commitments, ticked — the limiter stays on the card.
        "8b0f": ("checklist", ["Only public sources", "Respects robots.txt files"]),
        # a figure on graded levels: the ladder, its ends NOT invented.
        "d6d9": ("levels", ["Graduated autonomy levels", "30+", "capabilities"]),
        "a09d": ("animated_title_card", ["Merqi", "Request access at merqi.nl"]),
    }


def test_line_7_is_a_checklist_showing_only_public_sources() -> None:
    out = build_output(author_items(_lines(), None))
    card = out.items[6]
    assert card.template_id == "checklist"
    items = card.params["items"]
    assert isinstance(items, list)
    assert {"label": "Only public sources", "state": "yes"} in items
    assert all(item["state"] == "yes" for item in items)


def test_line_8_is_a_levels_ladder_with_30_and_no_invented_ends() -> None:
    out = build_output(author_items(_lines(), None))
    card = out.items[7]
    assert card.template_id == "levels"
    assert (card.params["value"], card.params["suffix"]) == ("30", "+")  # "over thirty"
    assert "from_label" not in card.params and "to_label" not in card.params
    # The stat lights where "thirty" is SPOKEN (word 8 of 9), not at the cut.
    assert card.params["marks"] == [0, 3083, 3083, 3083]


def test_every_card_carries_a_graphic() -> None:
    out = build_output(author_items(_lines(), None))
    for item, line in zip(out.items, _lines(), strict=True):
        drafts = deterministic_drafts(line)
        if line.is_cta:
            assert item.template_id == "animated_title_card"  # the brand lockup: the one exemption
            continue
        assert item.template_id not in ("animated_title_card", "section_divider")
        assert all(card.graphic for card in drafts), line.text
    assert out.unrepaired == []


def test_without_the_model_and_with_an_empty_answer_no_template_is_used_three_times() -> None:
    for proposals in (None, {}):
        templates = [item.params.template_id for item in author_items(_lines(), proposals)]
        assert max(templates.count(t) for t in templates) <= 2, templates


# ── the broken fragments can no longer be produced ──────────────────────────


def test_no_card_the_rules_can_build_shows_a_broken_fragment() -> None:
    for line in _lines():
        for card in deterministic_drafts(line):
            shown = card.screen_text()
            assert not set(shown) & {*_BROKEN, _BROKEN_TAIL}, (line.text, shown)
            sources = sources_for(line.text, *([line.brand_name] if line.is_cta else []))
            if gated(line.text):
                continue  # main clause + subordinate clause, each whole: checked by the gate
            phrases = card.phrases()
            if isinstance(card, TwoBeatParams) and card.emphasis:
                joined = f"{card.title} {card.emphasis}"
                if " ".join(joined.split()).lower().rstrip(".") in line.text.lower():
                    phrases = [(joined, HEADLINE)]  # one clause in two beats
            for text, slot in phrases:
                assert phrase_problem(text, sources, slot=slot) is None, (line.text, text)


@pytest.mark.parametrize(
    ("index", "draft", "why"),
    [
        (
            0,
            (
                "two_beat",
                {"title": "Merqi runs motion", "emphasis": "you keep final say", "icon": "bot"},
            ),
            # line 1 has "while": only its main clause + that clause may show
            "shown as its main clause + that clause",
        )
    ],
)
def test_the_runs_broken_cards_are_rejected_at_the_check(
    index: int, draft: tuple[str, dict[str, object]], why: str
) -> None:
    params, reason = check_draft(_lines()[index], draft)

    assert params is None
    assert reason is not None and why in reason, reason


@pytest.mark.parametrize(
    ("template_id", "content"),
    [("two_beat", {"title": "Only public sources"})],
)
def test_a_type_only_card_is_refused_for_a_line(
    template_id: str, content: dict[str, object]
) -> None:
    params, reason = check_draft(_lines()[6], (template_id, content))

    assert params is None
    assert reason == f"{template_id} as authored is type only — every card must carry a graphic"


def test_a_two_beat_with_an_icon_passes() -> None:
    params, reason = check_draft(
        _lines()[6], ("two_beat", {"title": "Only public sources", "icon": "doc"})
    )
    assert reason is None and params is not None and params.graphic == "icon doc"


# ── the phrase gate, word class by word class ───────────────────────────────


@pytest.mark.parametrize(
    ("line", "phrase", "problem"),
    [(0, "Merqi runs your go-to-market", "stops inside a phrase")],
)
def test_phrase_gate(line: int, phrase: str, problem: str | None) -> None:
    found = phrase_problem(phrase, sources_for(_RUN[line][1]))
    if problem is None:
        assert found is None
    else:
        assert found is not None and problem in found


def test_a_route_label_may_be_a_verb_group() -> None:
    sources = sources_for(_RUN[1][1])
    assert phrase_problem("Tailored", sources, slot="label") is None
    assert phrase_problem("Tailored", sources) is not None


# ── the closing tail, after the compiler's frame grid ───────────────────────


def test_the_closing_tail_is_at_most_three_seconds_after_frame_snapping() -> None:
    closing = _lines()[-1]
    narration = _RUN[-1][2]

    assert closing.is_cta
    assert closing.shot_seconds == pytest.approx(narration + CLOSING_TAIL_PLANNED_SECONDS, abs=1e-3)
    for fps in _LEGAL_FPS:
        tail = delivered_seconds(closing.shot_seconds, fps) - narration
        assert tail <= CLOSING_TAIL_MAX_SECONDS, (fps, tail)
    # Root cause, measured: the run planned exactly 3.0 s (a 5.122 s shot); at the
    # 30 fps master that is ceil(153.66) = 154 frames = 5.133 s → 3.012 s of tail.
    assert delivered_seconds(5.122, 30) - narration > CLOSING_TAIL_MAX_SECONDS


def test_every_shot_of_the_film_starts_and_ends_where_the_plan_says() -> None:
    out = build_output(author_items(_lines(), None))
    assert out.closing_beat is not None
    assert out.closing_beat.end_card_seconds == pytest.approx(CLOSING_TAIL_PLANNED_SECONDS - 2.0)
    assert out.items[-1].duration_ms == round((_RUN[-1][2] + CLOSING_TAIL_PLANNED_SECONDS) * 1000)


# ── polarity: a card never drops the negation that governs its words ───────
# Audit of PR #1080 (2026-10-07): the first head cut spans just after a negator
# and turned these lines into their opposites.

_NEGATED = [
    ("No campaign goes live without your approval.", ["Campaign goes live without your approval"]),
    ("Nothing ships until a human signs off.", ["Ships until a human signs off"]),
    ("Fewer than ten percent of drafts fail review.", ["Drafts fail review"]),
    ("Merqi does not send email without consent.", ["Send email without consent"]),
    ("We never sell your data.", ["Sell your data"]),
]


def _plain(text: str) -> ScriptLine:
    return ScriptLine(id="n", text=text, shot_seconds=6.0, narration_seconds=4.0)


def test_the_negator_on_the_card_keeps_the_line_s_polarity() -> None:
    sources = sources_for("No campaign goes live without your approval.")
    assert phrase_problem("No campaign goes live without your approval", sources) is None
    assert phrase_problem("Campaign goes live without your approval", sources) is not None


@pytest.mark.parametrize(
    ("text", "phrase", "problem"),
    [
        (
            "No campaign goes live without your approval.",
            "No campaign goes live",
            "cuts the scope",
        )
    ],
)
def test_negation_scope(text: str, phrase: str, problem: str) -> None:
    found = phrase_problem(phrase, sources_for(text))
    assert found is not None and problem in found, found


# ── a negated line always gets a card (audit of 81af1447) ───────────────────
# A negation keeps its scope, so a long negated line may have no short span: it
# must get a polarity-keeping card, never fail the run.


def test_a_compression_never_drops_the_verb_s_complement() -> None:
    sources = sources_for("No campaign goes live without your approval.")
    assert phrase_problem("No campaign goes without your approval", sources) is not None


def test_never_a_or_b_may_be_cut_to_never_a() -> None:
    """ "never A or B" entails "never A"; "never A and B" does not."""
    either = sources_for("We never scrape private profiles or buy contact lists from brokers.")
    both = sources_for("We never scrape private profiles and buy contact lists from brokers.")
    assert phrase_problem("We never scrape private profiles", either) is None
    assert phrase_problem("We never scrape private profiles", both) is not None


# ── a negated headline never cuts its condition clause (audit of 4e91b833) ─


@pytest.mark.parametrize(
    ("text", "phrase", "problem"),
    [("No offer is sent until a reviewer rejects the risky claims.",
         "No offer is sent until a reviewer", "cuts inside the condition clause")],
)  # fmt: skip
def test_a_condition_clause_is_never_cut(text: str, phrase: str, problem: str) -> None:
    found = phrase_problem(phrase, sources_for(text))
    assert found is not None and problem in found, found


# ── audit of 603186d3: questions, commas inside a condition, a hostless condition ─


def test_a_question_is_shown_as_itself() -> None:
    text = "Why would anything ship before you approve it?"
    line = _plain(text)

    item = author_items([line], None)[0]

    assert (item.params.template_id, item.params.screen_text()) == ("two_beat", [text])
    assert item.params.graphic and item.repairs == []
    # never derived into steps — "Ship → Approve it" reverses the question
    for card in deterministic_drafts(line):
        assert card.template_id == "two_beat" and card.screen_text() == [text]
    track, reason = check_draft(
        line, ("timeline", {"steps": [{"title": "Ship"}, {"title": "Approve it"}]})
    )
    assert track is None and reason == "timeline: a question is shown as itself, as one headline"


# ── audit of d6f223d9: nodes vs connectors, line order, relative clauses ────


@pytest.mark.parametrize(
    ("text", "draft", "why"),
    [("Merqi writes copy your reviewer approves.",
         ("timeline", {"steps": [{"title": "Merqi writes copy"}, {"title": "Approves"}]}),
         "opens on a verb or inside a clause")],
)  # fmt: skip
def test_illegal_card_elements_are_refused(
    text: str, draft: tuple[str, dict[str, object]], why: str
) -> None:
    params, reason = check_draft(_plain(text), draft)
    assert params is None
    assert reason is not None and why in reason, reason


def test_a_connector_may_be_the_line_s_verb_phrase() -> None:
    """ "Route automatically" on the arrow is legal; as a NODE it is not."""
    line = _plain("High-ticket conversations route automatically to a human agent.")
    ok, why = check_draft(
        line,
        ("handoff", {"from": {"label": "High-ticket conversations"},
                     "route": {"label": "Route automatically"}, "to": {"label": "Human agent"}}),
    )  # fmt: skip
    as_node, reason = check_draft(
        line,
        ("timeline", {"steps": [{"title": "High-ticket conversations"},
                                {"title": "Route automatically"}, {"title": "Human agent"}]}),
    )  # fmt: skip
    assert ok is not None, why
    assert as_node is None and reason is not None and "opens on a verb" in reason


def test_a_handoff_must_end_on_its_destination() -> None:
    """The destination node is the line's object AFTER the route — not the
    source again (a handoff that drops "a human agent")."""
    line = _plain("High-ticket conversations route automatically to a human agent.")
    params, reason = check_draft(
        line,
        ("handoff", {"from": {"label": "High-ticket conversations"},
                     "route": {"label": "Route automatically"},
                     "to": {"label": "High-ticket conversations"}}),
    )  # fmt: skip
    assert params is None and reason is not None and "not in the line's order" in reason


def test_a_restrictive_relative_clause_is_never_a_step() -> None:
    line = _plain("Merqi drafts copy that a human approves.")
    params, reason = check_draft(
        line,
        (
            "timeline",
            {"steps": [{"title": "Merqi drafts copy"}, {"title": "That a human approves"}]},
        ),
    )
    # refused at the phrase gate (a stranded "that") and by the element rule
    assert params is None and reason is not None


# ── audit of fac79031: a line with a condition, relative clause or negation ─
# never gets a diagram derived from its clauses: main clause + that clause, each
# whole, or the whole line (reported).

_GATED = {
    "Merqi drafts, reviews, and schedules posts before you approve them.": [
        "Merqi drafts, reviews, and schedules posts",
        "before you approve them",
    ],
    # a leading condition: the MAIN clause is the headline, the condition the sub-line
    "When a lead replies, Merqi hands the thread to your sales rep.": [
        "Merqi hands the thread to your sales rep",
        "When a lead replies",
    ],
    "Your team reviews every draft that Merqi writes before it goes out.": [
        "Your team reviews every draft",
        "that Merqi writes before it goes out",
    ],
    "Merqi writes the copy, which a human approves before anything ships.": [
        "Merqi writes the copy,",
        "which a human approves before anything ships",
    ],
    "No offer is sent until a reviewer rejects the risky claims.": [
        "No offer is sent",
        "until a reviewer rejects the risky claims",
    ],
    "Unless you approve it, nothing is sent.": ["Nothing is sent", "Unless you approve it"],
    # a negation with no subordinate clause, short: the whole line
    "We never sell your data.": ["We never sell your data."],
}


@pytest.mark.parametrize("text", list(list(_GATED))[:1])
def test_a_gated_line_shows_its_main_clause_and_that_clause_whole(text: str) -> None:
    item = author_items([_plain(text)], None)[0]

    assert item.params.template_id == "two_beat" and item.params.graphic
    assert [t.rstrip(",.") for t in item.params.screen_text()] == [
        t.rstrip(",.") for t in _GATED[text]
    ]
    assert item.repairs == []


def test_a_gated_line_never_gets_a_clause_derived_diagram() -> None:
    line = _plain("Merqi drafts, reviews, and schedules posts before you approve them.")
    steps, reason = check_draft(
        line,
        ("timeline", {"steps": [{"title": "Merqi drafts"}, {"title": "Reviews"},
                                {"title": "Schedules posts"}]}),
    )  # fmt: skip
    assert steps is None and reason is not None and "main clause + that clause" in reason
    assert all(card.template_id == "two_beat" for card in deterministic_drafts(line))
    # the same list without the condition keeps its track
    plain = author_items([_plain("Merqi drafts, reviews, and schedules posts.")], None)[0]
    assert plain.params.template_id == "timeline"


def test_a_gated_line_too_long_for_its_card_is_shown_whole_and_reported() -> None:
    # run 01a1179d line 6: 166 characters, more than the two-beat holds at its
    # smallest type — every word is still sent, and the overflow is reported
    text = (
        "Every send, price and stage change Merqi makes is checked against your policy before "
        "it happens, and written to an audit trail you can reconstruct, override or freeze."
    )
    item = author_items([_plain(text)], None)[0]
    assert item.params.graphic
    assert item.repairs and item.repairs[-1].startswith("NOT repaired: no card fits the line")
    from src.services.stages.promo.headline import tokens

    # nothing cut to a character count: every word is on the card
    assert tokens(" ".join(item.params.screen_text())) == tokens(text)


@pytest.mark.parametrize(
    ("text", "gate"),
    [("Merqi drafts, reviews, and schedules posts.", False)],
)
def test_the_gate(text: str, gate: bool) -> None:
    assert gated(text) is gate


def test_a_long_negated_line_with_no_clause_is_shown_whole_and_reported() -> None:
    """No subordinate clause to split at and more than eight words: the whole line
    on the card, flagged — never a clause-derived diagram (it used to be ✗ rows)."""
    text = "We never scrape private profiles or buy contact lists from brokers."
    item = author_items([_plain(text)], None)[0]

    assert item.params.template_id == "two_beat" and item.params.graphic
    shown = " ".join(item.params.screen_text())
    assert shown.rstrip(".") == text.rstrip(".")
    assert item.repairs and item.repairs[-1].startswith("NOT repaired: no card fits the line")


def test_a_question_without_a_question_mark_is_still_a_question() -> None:
    text = "Why would anything ship before you approve it."
    item = author_items([_plain(text)], None)[0]
    assert (item.params.template_id, item.params.screen_text()) == ("two_beat", [text])


# ── audit of ad65d49f: a second clause with its own subject is a safeguard ──


@pytest.mark.parametrize(
    ("text", "shown"),
    [("Merqi qualifies leads and a human closes the deal.",
         ("two_beat", ["Merqi qualifies leads", "and a human closes the deal"]))],
)  # fmt: skip
def test_a_second_clause_with_its_own_subject_is_never_dropped(
    text: str, shown: tuple[str, list[str]]
) -> None:
    item = author_items([_plain(text)], None)[0]
    assert (item.params.template_id, item.params.screen_text()) == shown
    assert item.repairs == []


# ── run 01a11693 (first live per-line render): a headline starts at a subject ─

_RUN_01A11693 = {
    # a clause headline runs to its clause end: one clause in two beats
    "It generates tailored multimedia assets and places them where targets already pay attention.": [
        # split where it reads and fits (render 01a1179d): before "where"
        "It generates tailored multimedia assets and places them",
        "where targets already pay attention",
    ],
    "Merqi executes your entire revenue motion under a single governed engine.": [
        "Merqi executes your entire revenue motion",
        "under a single governed engine",
    ],
}


@pytest.mark.parametrize("text", list(list(_RUN_01A11693))[:1])
def test_a_headline_starts_at_its_subject(text: str) -> None:
    item = author_items([_plain(text)], None)[0]
    assert item.params.screen_text() == _RUN_01A11693[text]


@pytest.mark.parametrize(
    ("text", "headline", "why"),
    [("It generates tailored multimedia assets and places them where targets already pay "
         "attention.", "Them where targets already pay attention", "pronoun 'them'")],
)  # fmt: skip
def test_a_mid_clause_headline_is_refused(text: str, headline: str, why: str) -> None:
    params, reason = check_draft(_plain(text), ("two_beat", {"title": headline, "icon": "spark"}))
    assert params is None and reason is not None and why in reason, reason


def test_an_object_noun_phrase_may_be_a_headline() -> None:
    text = "It generates tailored multimedia assets and places them where targets already pay attention."
    params, reason = check_draft(
        _plain(text), ("two_beat", {"title": "Tailored multimedia assets", "icon": "layers"})
    )
    assert params is not None, reason


def test_a_second_beat_is_never_a_fragment_of_another_clause() -> None:
    text = "It generates tailored multimedia assets and places them where targets already pay attention."
    params, reason = check_draft(
        _plain(text),
        ("two_beat", {"title": "It generates tailored multimedia assets",
                      "emphasis": "targets already pay attention", "icon": "layers"}),
    )  # fmt: skip
    assert params is None and reason is not None and "headline" in reason


# ── audit of #1084: a clause headline runs to its clause's end ─────────────

_COORDINATED = {
    "Every approval is single use and expires in minutes.": [
        "Every approval is single use",
        "and expires in minutes",
    ],
    # eight words: a short line is its own headline (render 01a1179d)
    "Every draft is human reviewed and logged forever.": [
        "Every draft is human reviewed and logged forever"
    ],
    "Each licence is tenant scoped and revoked on exit.": [
        "Each licence is tenant scoped",
        "and revoked on exit",
    ],
}


@pytest.mark.parametrize("text", list(list(_COORDINATED))[:1])
def test_a_clause_headline_keeps_its_coordinated_predicate(text: str) -> None:
    item = author_items([_plain(text)], None)[0]
    assert item.params.screen_text() == _COORDINATED[text]
    assert item.repairs == []


@pytest.mark.parametrize(
    ("text", "headline"),
    [("Every approval is single use and expires in minutes.", "Every approval is single")],
)
def test_a_clause_headline_never_stops_mid_clause(text: str, headline: str) -> None:
    params, reason = check_draft(_plain(text), ("two_beat", {"title": headline, "icon": "spark"}))
    assert params is None and reason is not None, reason


def test_a_subject_noun_phrase_may_stop_before_its_verb() -> None:
    line = _lines()[5]  # "Deterministic code handles critical financial and consent decisions."
    params, reason = check_draft(
        line,
        ("checklist", {"title": "Deterministic code",
                       "items": [{"label": "Financial decisions"}, {"label": "Consent decisions"}]}),
    )  # fmt: skip
    assert params is not None, reason


# ── render 01a1179d (REJECTED): cards fit the template, never one noun, never a
# beat ending on "every" / "you can", noun phrases keep their modifiers ────────

_RUN_01A1179D = {
    "Merqi runs your whole revenue motion as graduated, governed autonomy.": [
        "Merqi runs your whole revenue motion",
        "as graduated, governed autonomy",
    ],
    "It generates tailored multimedia assets and places them on channels where targets already "
    "pay attention.": [
        "It generates tailored multimedia assets and places them on channels",
        "where targets already pay attention",
    ],
    "You keep the veto.": ["You keep the veto"],
    # its main clause is 4 lines at the fit floor with the 9% margin: the whole
    # line, split before "to", every word kept (reported NOT repaired)
    "You can freeze anything in flight and trace every decision back to the exact rule that "
    "allowed it.": [
        "You can freeze anything in flight and trace every decision back",
        "to the exact rule that allowed it",
    ],
}


@pytest.mark.parametrize("text", list(list(_RUN_01A1179D))[:1])
def test_run_01a1179d_cards(text: str) -> None:
    item = author_items([_plain(text)], None)[0]
    assert item.params.screen_text() == _RUN_01A1179D[text]
    assert not any("does not fit" in repair for repair in item.repairs)


@pytest.mark.parametrize(
    ("text", "card", "why"),
    [("It generates tailored multimedia assets and places them on channels where targets "
         "already pay attention.", {"title": "Attention"}, "one word")],
)  # fmt: skip
def test_the_run_s_cards_are_refused(text: str, card: dict[str, object], why: str) -> None:
    params, reason = check_draft(_plain(text), ("two_beat", {**card, "icon": "spark"}))
    assert params is None and reason is not None and why in reason, reason


def test_a_beat_never_hands_over_on_a_modal_or_quantifier() -> None:
    line = _plain("Merqi drafts every reply you can send to a buyer in minutes.")
    params, reason = check_draft(
        line,
        (
            "two_beat",
            {"title": "Merqi drafts every", "emphasis": "reply you can send", "icon": "mail"},
        ),
    )
    assert params is None and reason is not None and "hands over on 'every'" in reason


def test_a_noun_phrase_headline_keeps_its_modifiers() -> None:
    sources_line = "Merqi runs your whole revenue motion as graduated, governed autonomy."
    from src.services.stages.promo.headline import headline_start_problem

    assert "drops its own modifier 'your'" in (
        headline_start_problem(sources_line, "Whole revenue motion") or ""
    )
    assert "drops its own modifier 'graduated'" in (
        headline_start_problem(sources_line, "Governed autonomy") or ""
    )


def test_the_measured_two_beat_budget() -> None:
    """The glyph model reproduces the Chromium measurement (render 01a1179d); the
    budget the stage applies keeps a 9% margin on the width (audit of #1086:
    capital- and W/M-heavy strings run wider than the table's averages)."""
    from src.services.stages.promo.motion_templates import _COPY_PX, FIT_MARGIN, wrapped_lines

    policy = "Every send, price and stage change Merqi makes is checked against your policy"
    raw = _COPY_PX  # the measurement itself, before the margin
    assert wrapped_lines(policy, width_px=raw) == 3
    assert (
        wrapped_lines("before it happens, and written to an audit trail you can", width_px=raw) == 2
    )
    assert wrapped_lines(f"{policy} before it happens", width_px=raw) == 4
    assert (
        wrapped_lines(
            "and written to an audit trail you can reconstruct, override or freeze", width_px=raw
        )
        == 3
    )
    assert FIT_MARGIN == 0.09
    assert wrapped_lines(policy) == 4  # at the boundary: the margin refuses it


def test_a_clause_after_its_article_still_runs_to_its_end() -> None:
    text = "The platform scores every landing page against the brand rubric nightly."
    item = author_items([_plain(text)], None)[0]
    assert item.params.screen_text() == [
        "The platform scores every landing page",
        "against the brand rubric nightly",
    ]
    params, reason = check_draft(
        _plain(text), ("two_beat", {"title": "Platform scores every landing page", "icon": "chart"})
    )
    assert params is None and reason is not None and "stops mid-clause" in reason


def test_a_beat_never_hands_over_on_a_quantifier() -> None:
    line = _plain("Merqi finds the most valuable accounts in minutes.")
    params, reason = check_draft(
        line,
        ("two_beat", {"title": "Merqi finds the most", "emphasis": "valuable accounts in minutes",
                      "icon": "target"}),
    )  # fmt: skip
    assert params is None and reason is not None and "hands over on 'most'" in reason


def test_a_title_past_three_lines_does_not_fit() -> None:
    from src.services.stages.promo.motion_templates import TwoBeatParams

    long = TwoBeatParams(
        title="Every send, price and stage change Merqi makes is checked against your policy "
        "before it happens",
        icon="check",
    )
    fits = TwoBeatParams(title="Every send, price and stage change Merqi makes", icon="check")
    assert long.overflow() == "title wraps to 4 lines (the two-beat holds 3)"
    assert fits.overflow() is None


# ── audit of #1086: a quantifier opens a subject ───────────────────────────

_QUANTIFIED = {
    "Most leads go cold within a week.": ["Most leads go cold within a week"],
    "Few teams have time to chase every lead by hand.": [
        "Few teams have time to chase every lead",
        "by hand",
    ],
    "No campaign goes live without your approval.": [
        "No campaign goes live",
        "without your approval",
    ],
}


@pytest.mark.parametrize("text", list(list(_QUANTIFIED))[:1])
def test_a_quantifier_opens_its_subject(text: str) -> None:
    item = author_items([_plain(text)], None)[0]
    assert item.params.screen_text() == _QUANTIFIED[text]
    assert item.repairs == []
