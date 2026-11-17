"""Tests for the shared line→visual sync primitive (promo + website_review)."""

from __future__ import annotations

from src.services.stages.shared.visual_sync import is_question, resolve_assignments


def _ans(lid: str) -> dict[str, str]:
    return {"line_id": lid, "text": "the thing.", "speaker": "b"}


def _q(lid: str) -> dict[str, str]:
    return {"line_id": lid, "text": "what about it?", "speaker": "a"}


def test_is_question_by_speaker_or_trailing_qmark() -> None:
    assert is_question({"speaker": "a", "text": "anything"})
    assert is_question({"speaker": "b", "text": "really?"})
    assert not is_question({"speaker": "b", "text": "here it is."})


def test_default_does_not_spread_promo_behaviour_unchanged() -> None:
    # Without the flag (how promo calls it), a collapsed LLM map carries one visual
    # forward across every answer — the spread net must NOT fire.
    lines = [_ans("a1"), _ans("a2"), _ans("a3")]
    out = resolve_assignments(lines, ["s1", "s2"], {"a1": "s1", "a2": "s1", "a3": "s1"})
    assert out == {"a1": "s1", "a2": "s1", "a3": "s1"}


def test_spread_on_collapse_walks_sections_top_to_bottom() -> None:
    lines = [_ans("a1"), _ans("a2"), _ans("a3"), _ans("a4")]
    out = resolve_assignments(
        lines,
        ["s1", "s2", "s3", "s4"],
        dict.fromkeys(("a1", "a2", "a3", "a4"), "s2"),
        spread_on_collapse=True,
    )
    assert out == {"a1": "s1", "a2": "s2", "a3": "s3", "a4": "s4"}


def test_spread_on_collapse_with_empty_map_still_distributes() -> None:
    lines = [_ans("a1"), _ans("a2"), _ans("a3")]
    out = resolve_assignments(lines, ["s1", "s2"], {}, spread_on_collapse=True)
    # 3 answers across 2 sections: 0,0,1 -> s1,s1,s2
    assert out == {"a1": "s1", "a2": "s1", "a3": "s2"}


def test_spread_holds_questions_on_previous_section() -> None:
    lines = [_ans("a1"), _q("q1"), _ans("a2")]
    out = resolve_assignments(
        lines, ["s1", "s2"], {"a1": "s1", "a2": "s1"}, spread_on_collapse=True
    )
    # collapse on s1 -> two answers spread to s1,s2; the question between holds s1.
    assert out == {"a1": "s1", "q1": "s1", "a2": "s2"}


def test_no_spread_when_llm_already_used_multiple_sections() -> None:
    lines = [_ans("a1"), _ans("a2")]
    out = resolve_assignments(
        lines, ["s1", "s2", "s3"], {"a1": "s2", "a2": "s3"}, spread_on_collapse=True
    )
    # 2 distinct sections used -> the LLM's semantic map is respected as-is.
    assert out == {"a1": "s2", "a2": "s3"}


def test_no_spread_with_single_section() -> None:
    lines = [_ans("a1"), _ans("a2")]
    out = resolve_assignments(lines, ["only"], {}, spread_on_collapse=True)
    assert out == {"a1": "only", "a2": "only"}
