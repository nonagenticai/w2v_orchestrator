"""Pins for the two deliberately-separate float-coercer families in ``src.utils``.

``as_float`` / ``opt_float`` are STRICT (isinstance-guarded, bool-rejecting, NaN-safe);
``coerce_float`` is PERMISSIVE (parses numeric strings, accepts bool / ``Decimal``, lets
NaN through). They look interchangeable and are not: they disagree on four live input
shapes. Every assertion below exists so a future "simplification" that folds the two
together FAILS LOUDLY here instead of silently changing behaviour at the call sites in
audio_premaster / redesign_motion / motion_graphics_gen / storyboard.assist /
gestalt_metrics.

``clamp_score`` (bottom) is the third, separate concern: it is the ENFORCEMENT half of the
LLM rubric-score contract whose declarative half (a JSON-Schema ``minimum``/``maximum``)
is unenforceable by guided decoding and fatal at validation. See
``tests/test_output_schema_bounds.py`` for the ban that pairs with it.
"""

from __future__ import annotations

import math
from decimal import Decimal

from src.utils import as_float, clamp_score, coerce_float, format_exception_message, opt_float


# ── divergence 1: numeric strings ────────────────────────────────────────────
def test_numeric_string_strict_rejects_permissive_parses() -> None:
    assert as_float("1.5") == 0.0
    assert opt_float("1.5") is None
    assert coerce_float("1.5") == 1.5


# ── divergence 2: bool is not a measurement ──────────────────────────────────
def test_bool_strict_rejects_permissive_coerces_to_one() -> None:
    assert as_float(True) == 0.0
    assert as_float(False) == 0.0
    assert opt_float(True) is None
    assert coerce_float(True) == 1.0
    assert coerce_float(False) == 0.0


# ── divergence 3: NaN poisons downstream arithmetic ──────────────────────────
def test_nan_strict_maps_to_default_or_none_permissive_passes_through() -> None:
    nan = float("nan")
    assert as_float(nan) == 0.0
    assert as_float(nan, default=-1.0) == -1.0
    assert opt_float(nan) is None
    permissive = coerce_float(nan)
    assert permissive is not None
    assert math.isnan(permissive)  # NOT rescued — the whole point of the split


# ── divergence 4: Decimal (a SupportsFloat that is not int/float) ────────────
def test_decimal_strict_rejects_permissive_accepts() -> None:
    assert as_float(Decimal("2")) == 0.0
    assert opt_float(Decimal("2")) is None
    assert coerce_float(Decimal("2")) == 2.0


# ── agreement: real numbers, None, and junk ──────────────────────────────────
def test_all_three_agree_on_plain_numbers() -> None:
    assert as_float(1.5) == 1.5
    assert as_float(3) == 3.0
    assert opt_float(1.5) == 1.5
    assert coerce_float(1.5) == 1.5


def test_none_and_junk() -> None:
    assert as_float(None) == 0.0
    assert as_float(None, default=7.0) == 7.0
    assert as_float(object()) == 0.0
    assert opt_float(None) is None
    assert opt_float(object()) is None
    assert coerce_float(None) is None
    assert coerce_float(object()) is None
    assert coerce_float("not a number") is None
    assert coerce_float([1.0]) is None


# ── format_exception_message: never returns an empty string ──────────────────
def test_format_exception_message_falls_back_to_repr() -> None:
    assert format_exception_message(ValueError("boom")) == "boom"
    assert format_exception_message(RuntimeError()) != ""


def test_non_finite_strict_maps_to_default_or_none_permissive_passes_through() -> None:
    # REGRESSION (item 4): ±inf used to slip through the STRICT family (only NaN was
    # rejected). It is not inert downstream — an infinite feature makes
    # `statistics.pstdev` raise ValueError inside gestalt_metrics, which unwinds to
    # final_critique's soft-degrade and SILENTLY DISCARDS the entire gestalt because of
    # ONE bad frame; it also poisons a Prometheus histogram `_sum` and is not valid JSON.
    for inf in (float("inf"), float("-inf")):
        assert as_float(inf) == 0.0
        assert as_float(inf, default=-1.0) == -1.0
        assert opt_float(inf) is None
        permissive = coerce_float(inf)
        assert permissive is not None
        assert math.isinf(permissive)  # NOT rescued — the strict/permissive split holds


# ── clamp_score: the bound the schema cannot carry ───────────────────────────
def test_clamp_score_rescues_the_value_that_killed_prod() -> None:
    """``overall_score: 12`` is the literal value that failed 100% of demo_video runs.

    Out-of-band reads as NO ANSWER (the ``default``), never as the nearest bound.
    Saturating ``12`` to ``5.0`` would be the worst reading available: a top score
    clears every convergence gate, so a magnitude the critic leaked out of its own
    prompt would silently certify the draft as perfect and skip the optimizer pass.
    """
    assert clamp_score(12) == 0.0
    assert clamp_score(0) == 0.0
    assert clamp_score(-40) == 0.0
    assert clamp_score(5.0001) == 0.0
    # In band ⇒ returned untouched, never rounded or re-scaled.
    assert clamp_score(3) == 3.0
    assert clamp_score(5) == 5.0
    assert clamp_score(1) == 1.0
    assert clamp_score(4.2) == 4.2


def test_clamp_score_missing_or_junk_keeps_the_historical_safe_default() -> None:
    """Absent / unparseable ⇒ ``0.0``, the value every call site read before the clamp.

    Deliberately OUTSIDE the [1, 5] band: a critic that answered nothing must not
    outrank one that scored the draft honestly badly.
    """
    assert clamp_score(None) == 0.0
    assert clamp_score({}.get("overall_score")) == 0.0
    assert clamp_score("not a number") == 0.0
    assert clamp_score(object()) == 0.0
    assert clamp_score([4]) == 0.0
    assert clamp_score(float("nan")) == 0.0  # NaN would poison the loop's max()
    assert clamp_score(None, default=3.0) == 3.0


def test_clamp_score_parses_strings_but_rejects_bool() -> None:
    """PERMISSIVE like ``coerce_float`` at the wire, except ``bool`` — JSON ``true`` is
    not a score, and Python would otherwise silently read it as ``1.0``."""
    assert clamp_score("4") == 4.0
    assert clamp_score("12") == 0.0  # parsed, then rejected as out-of-band
    assert clamp_score(True) == 0.0
    assert clamp_score(False) == 0.0


def test_clamp_score_band_is_overridable() -> None:
    assert clamp_score(0.5, low=0.0, high=1.0) == 0.5
    assert clamp_score(0.0, low=0.0, high=1.0) == 0.0  # inclusive at both edges
    assert clamp_score(1.0, low=0.0, high=1.0) == 1.0
    assert clamp_score(99, low=0.0, high=1.0) == 0.0  # out-of-band ⇒ default
    assert clamp_score(-1, low=0.0, high=1.0) == 0.0


def test_clamp_score_holds_the_reconcile_timeline_feasible_band() -> None:
    """``reconcile_timeline`` affine-maps the score into 3.0-5.0, which must stay above
    the 2.0 infeasible cap. That invariant now rests on the clamp, not on the schema."""
    for raw in (12, -3, 0, 5, 3, "junk", None):
        assert 3.0 <= 3.0 + (clamp_score(raw) / 5.0) * 2.0 <= 5.0
