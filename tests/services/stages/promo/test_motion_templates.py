"""motion_templates — what each film template reads, and when it moves.

The shapes pinned here are the recorder's own (video_production_service#188:
each template's header comment, ``_icons.js``, ``_motion_lib.js``'s film kit)
and the frozen preview contract (``template_params.json`` of the approved sheet).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.services.stages.promo.motion_templates import (
    FILM_KIT_TEMPLATES,
    ICON_NAMES,
    MAX_STATIC_SECONDS,
    PARAM_MODELS,
    ChecklistParams,
    HandoffParams,
    LevelsParams,
    PlannedShot,
    SectionDividerParams,
    TimelineParams,
    TitleCardParams,
    TwoBeatParams,
    TwoUpParams,
    card_static_seconds,
    film_static_seconds,
    known_icon,
    longest_rest_ms,
    static_seconds,
)

#: ``window.ICONS`` keys of video_production_service ``_icons.js`` at #188
#: (be272668), copied by hand from the file — the vendored constant must match.
_ICONS_JS = (
    "check arrow_right arrow_up_right bolt star spark target rocket layers grid chart "
    "trend_up gauge clock calendar shield lock globe users chat doc code play heart gear "
    "cross bot user headset chat_dots braces eye mail route"
)


def test_the_vocabulary_is_the_film_kit_plus_one_last_resort() -> None:
    assert sorted(PARAM_MODELS) == [
        "animated_title_card",
        "checklist",
        "handoff",
        "levels",
        "section_divider",
        "timeline",
        "two_beat",
        "two_up",
    ]
    assert "section_divider" not in FILM_KIT_TEMPLATES
    assert {"kinetic_typography", "pipeline_story"}.isdisjoint(PARAM_MODELS)


def test_the_vendored_icon_names_are_the_recorders() -> None:
    assert set(_ICONS_JS.split()) == ICON_NAMES


@pytest.mark.parametrize(("name", "kept"), [("code", "code")])
def test_an_unknown_icon_is_dropped_not_sent(name: str, kept: str) -> None:
    assert known_icon(name) == kept
    panel = TwoUpParams(
        left={"title": "Deterministic code", "icon": name, "state": "chosen"},
        right={"title": "Language models", "state": "rejected"},
    ).template_params()["two_up"]
    assert isinstance(panel, dict)
    assert panel["left"] == {
        "title": "Deterministic code",
        **({"icon": kept} if kept else {}),
        "state": "chosen",
    }


def test_each_template_sends_exactly_the_preview_contract_shape() -> None:
    """Content keys as in the approved sheet's ``template_params.json``."""
    assert TwoBeatParams(
        title="Go-to-market that runs itself", emphasis="under your control"
    ).template_params() == {
        "title": "Go-to-market that runs itself",
        "emphasis": "under your control",
    }
    assert LevelsParams(
        title="Graduated autonomy levels",
        value="30",
        suffix="+",
        label="capabilities",
        from_label="Prohibited",
        to_label="Automatic",
    ).template_params() == {
        "title": "Graduated autonomy levels",
        "value": "30",
        "suffix": "+",
        "label": "capabilities",
        "levels": 5,
        "from_label": "Prohibited",
        "to_label": "Automatic",
    }
    assert TimelineParams(
        title="Pull-based demand generation",
        steps=[{"title": "Earning attention", "icon": "eye"}, {"title": "Requesting contact"}],
    ).template_params() == {
        "layout": "track",
        "title": "Pull-based demand generation",
        "steps": [{"title": "Earning attention", "icon": "eye"}, {"title": "Requesting contact"}],
    }
    assert ChecklistParams(
        items=[
            {"label": "Public sources", "state": "yes"},
            {"label": "Bought lists", "state": "no"},
        ]
    ).template_params() == {
        "items": [
            {"label": "Public sources", "state": "yes"},
            {"label": "Bought lists", "state": "no"},
        ]
    }
    assert HandoffParams.model_validate(
        {
            "title": "High-ticket and sensitive conversations",
            "from": {"icon": "bot", "label": "Conversations"},
            "route": {"label": "Automatically routed"},
            "to": {"icon": "user", "label": "Human agent"},
        }
    ).template_params() == {
        "title": "High-ticket and sensitive conversations",
        "from": {"label": "Conversations", "icon": "bot"},
        "route": {"label": "Automatically routed"},
        "to": {"label": "Human agent", "icon": "user"},
    }


def test_the_title_card_is_film_style_and_never_draws_another_companys_name() -> None:
    """Left unsent, the title card shows demo copy and "nonagentic.ai"; ``brand_mark``
    equal to the title is not drawn, so the name shows once."""
    assert TitleCardParams(title="Merqi", subtitle="Request access").template_params() == {
        "style": "film",
        "eyebrow": "",
        "chapter": "",
        "title": "Merqi",
        "subtitle": "Request access",
        "brand_mark": "Merqi",
    }


@pytest.mark.parametrize(
    ("model", "raw"),
    [(TwoBeatParams, {"title": "x", "subtitle": "y"})],
)
def test_params_a_template_would_ignore_or_mangle_do_not_validate(
    model: type[TwoBeatParams], raw: dict[str, object]
) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(raw)


def test_the_beats_follow_each_templates_own_mark_order() -> None:
    assert TwoBeatParams(title="A", emphasis="B").beats() == ["A", "B"]
    assert LevelsParams(value="30", from_label="P", to_label="A").beats() == [None, "30", "P", "A"]
    assert TwoUpParams(left={"title": "L"}, right={"title": "R"}).beats() == [None, None, "L", "R"]
    assert TimelineParams(steps=[{"title": "S1"}, {"title": "S2"}]).beats() == [
        None,
        None,
        "S1",
        "S2",
    ]
    assert ChecklistParams(items=[{"label": "a"}, {"label": "b", "state": "no"}]).beats() == [
        None,
        "a",
        "b",
    ]
    assert HandoffParams.model_validate(
        {"from": {"label": "s"}, "route": {"label": "r"}, "to": {"label": "d"}}
    ).beats() == [None, "r", "d"]
    # The first row with content opens with the card — the recorder pins it to 0.
    assert TitleCardParams(title="Merqi", subtitle="Request access").beats() == [
        None,
        None,
        None,
        "Request access",
    ]


def test_film_templates_never_rest_and_the_old_divider_does() -> None:
    """The film kit keeps a foreground element moving for the whole clip (measured
    on #188: <= 0.5 s foreground-static at 4 s and 12 s); the pre-film divider
    stops after its rule has drawn in."""
    for params in (
        TwoBeatParams(title="Under your control"),
        TitleCardParams(title="Merqi"),
        ChecklistParams(items=[{"label": "a"}, {"label": "b"}]),
    ):
        assert card_static_seconds(params, [0] * len(params.beats()), shot_ms=12000) == 0.0
    divider = SectionDividerParams(title="Under your control")
    assert divider.continuous_motion is False
    # Entrances 80-800 (title) and 80-720 (rule): at rest from 0.8s to the cut.
    assert card_static_seconds(divider, [0, 0], shot_ms=6000) == pytest.approx(5.2)
    assert card_static_seconds(divider, [0, 0], shot_ms=6000) > MAX_STATIC_SECONDS


def test_a_clip_shorter_than_its_shot_freezes_for_the_difference() -> None:
    assert longest_rest_ms([(0, 600)], clip_ms=3000, shot_ms=5000) == 4400
    assert longest_rest_ms([(0, 600), (4000, 4600)], clip_ms=3000, shot_ms=5000) == 4400


def test_the_plan_predicts_the_static_stretch_of_every_kind_of_shot() -> None:
    shots = [
        PlannedShot("card", 8.0, "motion_card", clip_seconds=8.0, continuous=True),
        PlannedShot("short-card", 8.0, "motion_card", clip_seconds=5.0, continuous=True),
        PlannedShot("divider", 6.0, "motion_card", events=((80, 800),)),
        PlannedShot("still", 8.0, "still"),
        PlannedShot("kb", 8.0, "still", ken_burns=True),
        PlannedShot("clip", 8.0, "clip", clip_seconds=6.0),
        PlannedShot("end", 9.0, "still", end_card=True),
    ]

    per_line, longest = film_static_seconds(shots)

    assert per_line == {
        "card": 0.0,
        "short-card": 3.0,
        "divider": 5.2,
        "still": 8.0,
        "kb": 0.0,
        "clip": 2.0,
        "end": 9.0,
    }
    assert longest == 8.0  # the end card is not counted
    assert static_seconds(shots[0]) == 0.0
