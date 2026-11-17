"""Tests for the normalize_source_script helper."""

from __future__ import annotations

from src.services.stages.base import normalize_source_script


def test_line_split_one_segment_per_nonblank_line() -> None:
    segs = normalize_source_script("Line one.\n\nLine two.\n", split="line")
    assert [t for t, _ in segs] == ["Line one.", "Line two."]


def test_paragraph_split_on_blank_lines() -> None:
    text = "Para one line a.\nPara one line b.\n\nPara two."
    segs = normalize_source_script(text, split="paragraph")
    assert [t for t, _ in segs] == [
        "Para one line a.\nPara one line b.",
        "Para two.",
    ]


def test_seconds_estimate_at_least_one() -> None:
    segs = normalize_source_script("Hi", split="line")
    assert segs[0][1] >= 1.0


def test_empty_input_returns_empty_list() -> None:
    assert normalize_source_script("", split="line") == []


def test_whitespace_only_returns_empty_list() -> None:
    assert normalize_source_script("   \n\n  \t ", split="paragraph") == []
