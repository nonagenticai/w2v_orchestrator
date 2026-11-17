"""Unit tests for grouping word-timed tokens into on-screen ShotCues.

The grouper turns a flat stream of TimedTokens into a small number of cues,
each with at most `max_lines` lines and each line at most `max_chars_per_line`
characters, flushing on sentence-final punctuation.
"""

from __future__ import annotations

from src.services.caption_cues import ShotCue, group_tokens_into_cues
from src.services.caption_reconcile import TimedToken


def _tok(text: str, start: float, end: float, source: str = "asr") -> TimedToken:
    return TimedToken(text=text, start=start, end=end, source=source)


def _even_tokens(words: list[str], step: float = 0.5) -> list[TimedToken]:
    return [_tok(w, i * step, (i + 1) * step) for i, w in enumerate(words)]


def test_wraps_at_max_line_chars() -> None:
    # "one two three four" won't fit on a single 10-char line.
    tokens = _even_tokens(["one", "two", "three", "four"])

    cues = group_tokens_into_cues(tokens, max_chars_per_line=10, max_lines=2)

    for cue in cues:
        for line in cue.lines:
            assert len(line) <= 10
    # It did wrap — more than one line got produced overall.
    total_lines = sum(len(c.lines) for c in cues)
    assert total_lines >= 2


def test_sentence_final_forces_flush() -> None:
    tokens = _even_tokens(["Hi", "there.", "more", "text"])

    cues = group_tokens_into_cues(tokens, max_chars_per_line=42, max_lines=2)

    assert len(cues) >= 2
    # The sentence-final token closes its cue; "more" begins a later cue.
    first_text = " ".join(cues[0].lines)
    assert "there." in first_text
    assert "more" not in first_text
    assert any("more" in " ".join(c.lines) for c in cues[1:])


def test_two_line_flush() -> None:
    # Small line budget forces a new line per word; a 3rd line overflows the
    # 2-line cue and flushes a new cue.
    tokens = _even_tokens(["aaa", "bbb", "ccc", "ddd"])

    cues = group_tokens_into_cues(tokens, max_chars_per_line=5, max_lines=2)

    assert len(cues) >= 2
    for cue in cues:
        assert len(cue.lines) <= 2


def test_relative_time_monotonic() -> None:
    tokens = _even_tokens(["Alpha", "beta", "gamma.", "Delta", "epsilon"])

    cues = group_tokens_into_cues(tokens, max_chars_per_line=42, max_lines=2)

    assert cues
    assert cues[0].start == tokens[0].start
    for i in range(len(cues) - 1):
        assert cues[i].start <= cues[i].end
        assert cues[i].end <= cues[i + 1].start + 1e-9


def test_single_line_cue() -> None:
    tokens = _even_tokens(["Hi", "there"])

    cues = group_tokens_into_cues(tokens, max_chars_per_line=42, max_lines=2)

    assert len(cues) == 1
    cue = cues[0]
    assert isinstance(cue, ShotCue)
    assert len(cue.lines) == 1
    assert cue.lines[0] == "Hi there"
    assert cue.start == tokens[0].start
    assert cue.end == tokens[-1].end


# --------------------------------------------------------------------------
# per-word timings retained for karaoke (words[])
# --------------------------------------------------------------------------


def test_cue_words_carry_every_token_in_order() -> None:
    """A cue's ``words`` is the flat, in-order list of its tokens' (text, start, end)
    — the per-word timings the compiler needs for karaoke."""
    tokens = _even_tokens(["Hi", "there"])
    cue = group_tokens_into_cues(tokens, max_chars_per_line=42, max_lines=2)[0]
    assert cue.words == [("Hi", 0.0, 0.5), ("there", 0.5, 1.0)]
    # words span exactly the cue window.
    assert cue.words[0][1] == cue.start
    assert cue.words[-1][2] == cue.end


def test_cue_words_partition_across_cues_without_loss() -> None:
    """Every token lands in exactly one cue's ``words`` (no drop, no duplication),
    and the words match that cue's rendered text — so karaoke can't desync."""
    words = ["Alpha", "beta", "gamma.", "Delta", "epsilon"]
    tokens = _even_tokens(words)
    cues = group_tokens_into_cues(tokens, max_chars_per_line=42, max_lines=2)
    # union of all cue.words == the original token stream, in order
    flat = [w for cue in cues for w in cue.words]
    assert [w[0] for w in flat] == words
    # each cue's words reconstruct its own text
    for cue in cues:
        assert " ".join(w[0] for w in cue.words) == " ".join(cue.lines)


def test_cue_words_unaffected_by_monotonic_clamp() -> None:
    """The monotonic clamp adjusts cue start/end but carries the words through
    unchanged (the compiler re-clamps word times against the cue window)."""
    # overlapping token times force the clamp to move a cue's start.
    tokens = [_tok("one.", 0.0, 2.0), _tok("two.", 1.0, 1.5)]
    cues = group_tokens_into_cues(tokens, max_chars_per_line=42, max_lines=2)
    assert [w[0] for cue in cues for w in cue.words] == ["one.", "two."]
