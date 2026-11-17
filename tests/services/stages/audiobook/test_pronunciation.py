"""Unit tests for the per-book pronunciation lexicon (pure functions, no I/O)."""

from __future__ import annotations

from typing import Any, cast

from src.services.stages.audiobook.pronunciation import (
    apply_lexicon,
    build_lexicon,
)


def test_empty_lexicon_is_noop() -> None:
    text = "The quick brown fox."
    assert apply_lexicon(text, {}) == text
    assert apply_lexicon(text, build_lexicon(None)) == text


def test_whole_word_only_no_partial_match() -> None:
    # "Bos" must not fire inside "Boston" / "Bossy".
    lex = {"Bos": "Boss"}
    assert apply_lexicon("Bos went to Boston, feeling bossy.", lex) == (
        "Boss went to Boston, feeling bossy."
    )


def test_case_insensitive_canonical_replacement() -> None:
    lex = {"café": "cafay"}
    # Any casing matches; the canonical spoken value is emitted.
    assert apply_lexicon("The Café and the café.", lex) == "The cafay and the cafay."


def test_multi_word_key_wins_over_fragment() -> None:
    # Longest-first ordering: "de Jong" beats a bare "Jong" entry.
    lex = {"Jong": "yong", "de Jong": "duh YONG"}
    assert apply_lexicon("Mr de Jong met young Jong.", lex) == "Mr duh YONG met young yong."


def test_quotes_and_boundaries_preserved() -> None:
    # Dialogue quote characters are untouched so span-splitting still works.
    lex = {"Koster": "KOSS-ter"}
    src = '"Koster," she said, "is late."'
    assert apply_lexicon(src, lex) == '"KOSS-ter," she said, "is late."'


def test_idempotent() -> None:
    lex = {"Merqi": "MER-kee"}
    once = apply_lexicon("Merqi arrived.", lex)
    assert apply_lexicon(once, lex) == once


def test_regex_special_chars_in_key_are_escaped() -> None:
    lex = {"C.A.B.": "C-A-B"}
    # The dots are literal, not "any char" — "CxAxBx" must not match.
    assert apply_lexicon("The C.A.B. met.", lex) == "The C-A-B met."
    assert apply_lexicon("CxAxBx", lex) == "CxAxBx"


# default retained


def test_build_lexicon_drops_bad_entries() -> None:
    # The brief is free-form JSON, so it may carry non-string / empty keys and
    # values — drop them defensively rather than crash.
    bad = cast(Any, {"": "x", "Good": "good", "Bad": 123, 5: "y"})
    merged = build_lexicon(bad)
    assert merged == {"Good": "good"}


def test_build_lexicon_none_is_empty() -> None:
    assert build_lexicon(None) == dict()
