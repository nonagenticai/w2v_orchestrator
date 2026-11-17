"""build_line_ssml — the pure line→SSML lowering for narrate (TTSB-3).

The load-bearing property is the None path: no pause + no emphasis ⇒ ``None``,
so the caller sends raw text byte-identically (cache stays warm). Everything
else asserts the SSML shape, that emphasis indices are ``text.split()``
positions, and that tokens are XML-escaped BEFORE the ``<emphasis>`` wrap.
"""

from __future__ import annotations

from typing import cast

import pytest

from src.services.stages.primitives.ssml import (
    EMPHASIS_MIN_VO_RATE,
    build_line_ssml,
    emphasis_affordable,
)


def test_no_pause_no_emphasis_returns_none() -> None:
    """The plain-text path: nothing to express ⇒ None (caller uses raw text)."""
    assert build_line_ssml("Hello world", 0, []) is None


def test_negative_pause_no_emphasis_returns_none() -> None:
    assert build_line_ssml("Hello world", -10, []) is None


def test_pause_only() -> None:
    assert (
        build_line_ssml("Hello world", 500, []) == '<speak><break time="500ms"/>Hello world</speak>'
    )


def test_emphasis_only_wraps_the_right_word() -> None:
    """Index 1 stresses the second split() token; the others stay plain."""
    assert (
        build_line_ssml("Hello world today", 0, [1])
        == '<speak>Hello <emphasis level="strong">world</emphasis> today</speak>'
    )


def test_pause_and_emphasis_together() -> None:
    assert (
        build_line_ssml("Hello world today", 500, [0]) == '<speak><break time="500ms"/>'
        '<emphasis level="strong">Hello</emphasis> world today</speak>'
    )


def test_out_of_range_indices_are_ignored() -> None:
    """Indices >= len(tokens) (and negative) don't wrap anything, but the
    presence of an index still takes us off the None path."""
    result = build_line_ssml("Hello world", 0, [5, -1])
    assert result == "<speak>Hello world</speak>"
    assert "<emphasis" not in result


def test_token_is_xml_escaped_before_emphasis_wrap() -> None:
    """ "R&D" at index 0: the & is escaped to &amp; AND the emphasis wraps the
    already-escaped token — no raw & survives, no broken tag."""
    result = build_line_ssml("R&D wins", 0, [0])
    assert result == ('<speak><emphasis level="strong">R&amp;D</emphasis> wins</speak>')
    assert "&amp;" in result
    # No raw ampersand that is not part of an entity.
    assert "R&D" not in result


def test_escapes_unemphasized_tokens_too() -> None:
    """Escaping happens for every token, emphasized or not. A pause forces the
    SSML path (pause=0 + no emphasis would short-circuit to None by contract)."""
    assert (
        build_line_ssml("a < b & c", 300, [])
        == '<speak><break time="300ms"/>a &lt; b &amp; c</speak>'
    )


def test_first_and_last_indices() -> None:
    """A token at index 0 and at the final index both wrap correctly."""
    assert (
        build_line_ssml("one two three", 0, [0, 2])
        == '<speak><emphasis level="strong">one</emphasis> two '
        '<emphasis level="strong">three</emphasis></speak>'
    )


# ══ EMPHASIS IS BOUNDED AT THE SOURCE (root cause B, 1b) ══════════════════════
#
# `tts_backend` renders a `strong` emphasis span as its OWN kokoro utterance at
# `request_speed * 0.88`. A line already fit to (or near) `VO_RATE_NATURAL_FLOOR`
# (0.80) cannot afford that further drop without a mid-sentence rate cliff below
# the floor the rest of the line is held to — so a line at or under
# `EMPHASIS_MIN_VO_RATE` gets NO emphasis at all, belt-and-braces with the
# backend-side clamp (item 2a).


def test_emphasis_min_vo_rate_is_the_floor_over_the_backend_multiplier() -> None:
    assert pytest.approx(0.80 / 0.88) == EMPHASIS_MIN_VO_RATE


def test_emphasis_affordable_true_above_the_threshold() -> None:
    assert emphasis_affordable(1.0) is True
    assert emphasis_affordable(EMPHASIS_MIN_VO_RATE + 0.01) is True


def test_emphasis_affordable_false_below_the_threshold() -> None:
    assert emphasis_affordable(EMPHASIS_MIN_VO_RATE) is True  # boundary: >=, not >
    assert emphasis_affordable(EMPHASIS_MIN_VO_RATE - 0.01) is False
    assert emphasis_affordable(0.80) is False  # the fit floor itself is unaffordable


def test_emphasis_affordable_none_and_junk_default_to_true() -> None:
    """None = no resolved rate = backend default 1.0 = affordable; junk degrades
    the same way rather than silently suppressing direction on a coercion bug."""
    assert emphasis_affordable(None) is True
    # Directive-derived values are not guaranteed numeric at runtime (an
    # LLM-shaped stage output) even though the type hint says `float | None` —
    # the `cast` documents that this call is deliberately exercising the
    # defensive coercion, not a real caller shape.
    assert emphasis_affordable(cast("float", "soon")) is True
    assert emphasis_affordable(float("nan")) is True


def test_a_line_below_the_floor_gets_no_emphasis_at_all() -> None:
    """The whole point: emphasis is dropped for the LINE, not silently degraded
    to a lesser level — a line this slow cannot afford ANY further rate step.
    A pause keeps the line on the SSML path so the dropped emphasis is visible
    in the rendered markup rather than short-circuiting to None."""
    result = build_line_ssml("Hello world today", 200, [1], vo_rate=0.70)
    assert result == '<speak><break time="200ms"/>Hello world today</speak>'
    assert "<emphasis" not in result


def test_a_line_at_a_safe_rate_keeps_its_emphasis() -> None:
    result = build_line_ssml("Hello world today", 0, [1], vo_rate=1.0)
    assert result == '<speak>Hello <emphasis level="strong">world</emphasis> today</speak>'


def test_unaffordable_emphasis_with_no_pause_falls_all_the_way_to_none() -> None:
    """Dropping the only thing this line had to express (emphasis, no pause)
    must fall through to the byte-identical plain-text path, not an inert
    <speak> wrapper around unmarked text."""
    assert build_line_ssml("Hello world", 0, [1], vo_rate=0.70) is None


def test_omitting_vo_rate_is_byte_identical_to_the_pre_1b_behaviour() -> None:
    """The default parameter must not change any existing caller's output."""
    assert build_line_ssml("Hello world today", 0, [1]) == build_line_ssml(
        "Hello world today", 0, [1], vo_rate=None
    )
