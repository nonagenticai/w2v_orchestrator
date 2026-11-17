"""Pure (no-DB) tests for the storyboard → AuthoredLine normalizer.

Covers ``src/services/stages/shared/authored_storyboard.py``:

* ``storyboard_panels(brief)`` — None when absent/empty; panel→AuthoredLine
  field mapping (shot_key→id via normalize_clip_key, vo_text/dialogue/
  action_description text precedence, duration_seconds→seconds,
  shot_number→scenario_name).
* ``authored_lines(brief)`` — storyboard-wins-over-``lines`` precedence, the
  legacy ``brief["lines"]`` fallback, and byte-identical output to the inline
  legacy mapping when no storyboard is present.
"""

from __future__ import annotations

from src.models.showcase_line import AuthoredLine
from src.services.stages.shared.authored_storyboard import (
    authored_lines,
    storyboard_panels,
)


# ─── storyboard_panels ───────────────────────────────────────────────────
def test_storyboard_panels_none_when_absent() -> None:
    assert storyboard_panels({}) is None
    assert storyboard_panels({"lines": [{"text": "x"}]}) is None


def test_storyboard_panels_none_when_empty_list() -> None:
    # An explicitly empty board is not an authored spine → None (not []).
    assert storyboard_panels({"storyboard": []}) is None


def test_storyboard_panels_maps_fields() -> None:
    brief = {
        "storyboard": [
            {
                "shot_key": "@clip-3",
                "vo_text": "  narration  ",
                "duration_seconds": 4.0,
                "shot_number": "12A",
                "speaker": " a ",
                "expected_text": " Dashboard ",
            }
        ]
    }
    lines = storyboard_panels(brief)
    assert lines is not None
    (line,) = lines
    assert isinstance(line, AuthoredLine)
    # shot_key normalized via normalize_clip_key (leading @ stripped) → id.
    assert line.id == "clip-3"
    assert line.text == "narration"
    assert line.seconds == 4.0
    assert line.scenario_name == "12A"
    assert line.speaker == "a"
    assert line.expected_text == "Dashboard"


def test_storyboard_panels_text_precedence() -> None:
    # vo_text wins over dialogue wins over action_description.
    vo = storyboard_panels(
        {"storyboard": [{"vo_text": "vo", "dialogue": "dlg", "action_description": "act"}]}
    )
    dlg = storyboard_panels({"storyboard": [{"dialogue": "dlg", "action_description": "act"}]})
    act = storyboard_panels({"storyboard": [{"action_description": "act"}]})
    none_txt = storyboard_panels({"storyboard": [{"shot_key": "clip-1"}]})
    assert vo is not None and dlg is not None and act is not None and none_txt is not None
    assert vo[0].text == "vo"
    assert dlg[0].text == "dlg"
    assert act[0].text == "act"
    # No narration field at all → empty text.
    assert none_txt[0].text == ""


def test_storyboard_panels_shot_key_fallback_and_seconds_alias() -> None:
    # Missing shot_key → clip-<idx+1>; seconds alias honored.
    lines = storyboard_panels({"storyboard": [{"vo_text": "a"}, {"vo_text": "b", "seconds": 2.5}]})
    assert lines is not None
    assert [line.id for line in lines] == ["clip-1", "clip-2"]
    assert lines[1].seconds == 2.5


# ─── authored_lines ──────────────────────────────────────────────────────
def test_authored_lines_none_when_neither_present() -> None:
    assert authored_lines({}) is None
    assert authored_lines({"storyboard": [], "lines": []}) is None


def test_authored_lines_storyboard_wins_over_lines() -> None:
    brief = {
        "storyboard": [{"shot_key": "clip-1", "vo_text": "from board"}],
        "lines": [{"id": "clip-1", "text": "from lines"}],
    }
    lines = authored_lines(brief)
    assert lines is not None
    assert [line.text for line in lines] == ["from board"]
    # Identical to the storyboard-only normalization.
    assert lines == storyboard_panels(brief)


def test_authored_lines_falls_back_to_legacy_lines() -> None:
    brief = {
        "lines": [
            {
                "id": "@clip-1",
                "text": " hi ",
                "seconds": 1.5,
                "scenario_name": "Intro",
                "gherkin": "Given x",
                "speaker": "guide",
                "clip": "@clip-1",
                "clip_anchor": "end",
                "clip_lead_seconds": 0.5,
                "expected_text": " Home ",
            }
        ]
    }
    lines = authored_lines(brief)
    assert lines is not None
    (line,) = lines
    # Byte-identical to the legacy inline mapping (field-for-field coercion).
    assert line == AuthoredLine(
        id="clip-1",
        text="hi",
        seconds=1.5,
        scenario_name="Intro",
        gherkin="Given x",
        speaker="guide",
        clip="clip-1",
        clip_anchor="end",
        clip_lead_seconds=0.5,
        expected_text="Home",
    )


def test_authored_lines_legacy_role_aliases_speaker_and_id_fallback() -> None:
    brief = {
        "lines": [
            {"text": "one", "role": "narrator"},
            {"text": "two"},
        ]
    }
    lines = authored_lines(brief)
    assert lines is not None
    # `role` is accepted as a `speaker` alias; missing id → clip-<idx+1>.
    assert lines[0].speaker == "narrator"
    assert [line.id for line in lines] == ["clip-1", "clip-2"]


# ─── locked (merqi showcase spike 01a0fc55) ──────────────────────────────
def test_authored_lines_carry_locked_from_brief_lines() -> None:
    # The field-by-field rebuild dropped `locked`, so every downstream rewriter
    # (script_doctor / claim_finish / retention_plan / reconcile_timeline) saw an
    # authored, author-pinned line as unlocked and rewrote / padded it.
    brief = {
        "lines": [
            {"id": "l1", "text": "Verbatim one.", "clip": "clip-1", "locked": True},
            {"id": "l2", "text": "Free two.", "clip": "clip-2"},
        ]
    }
    lines = authored_lines(brief)
    assert lines is not None
    assert [line.locked for line in lines] == [True, False]
    assert lines[0].model_dump()["locked"] is True


def test_storyboard_panels_carry_locked() -> None:
    lines = storyboard_panels(
        {"storyboard": [{"shot_key": "clip-1", "vo_text": "x", "locked": True}]}
    )
    assert lines is not None
    assert lines[0].locked is True
