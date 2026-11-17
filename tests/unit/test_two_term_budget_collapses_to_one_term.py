"""⭐ WAVE 14, N1 — WHY THE TWO-TERM WORD BUDGET CANNOT BE BUILT AT THE BUDGET SITE.

The keystone ask of wave 14 was to replace the word budget's single padded
words-per-minute with the explicit two-term model the sample already measures::

    seconds  ~=  words / speech_rate  +  per_piece_pause x piece_count

The reasoning is sound and the diagnosis is right: dividing by the PADDED rate bakes the
sample's AVERAGE pause-per-word into every beat, so a beat with more sentence boundaries
than the sample is under-budgeted. ``speech_rate._log_two_term_decomposition`` says the
same thing in prose and has said it since 2026-09-15.

⛔ BUT AT THE BUDGET SITE THE SUBSTITUTION IS AN ARITHMETIC NO-OP, AND THIS IS THE FACT
THAT SETTLES IT. ``narration_words_target`` / ``compute_plan.beat_word_budget`` run
SECONDS -> WORDS, and they run BEFORE any text exists: the LLM chooses the sentence
structure after the budget is written, so ``piece_count`` is not known, it must be
PREDICTED. The only predictor the pipeline has is the sample's own pieces-per-word
ratio — and substituting it collapses the two-term expression back into the one-term
one, exactly, for every word count. That is what this test proves.

So a "two-term budget" built from the measurement this repo has would emit byte-identical
numbers while looking like a fix. It can only differ by using a piece count that is NOT
derived from the sample — i.e. a guess — and ``_log_two_term_decomposition`` already
costed that: ``s*`` spans 0.93-1.11 with PERFECT piece knowledge and 0.83-1.24 once the
count is mis-predicted by +/-2, against a tolerance band (``NARRATION_BUDGET_RATE`` 1.0
down to ``VO_RATE_DRAG_FLOOR`` 0.97) that is 3% wide.

⭐ WHAT WOULD UNBLOCK IT — and it is cheap, but it is a CROSS-REPO CONTRACT, not an
arithmetic change in this file. The orchestrator already sends bdd a per-beat WORD count
(``narration_words_per_scenario``). Send a per-beat SENTENCE count alongside it and the
piece count stops being a prediction and becomes a commitment: the two-term budget is
then implementable, falsifiable, and — because it models the padding explicitly — safe
under the ``tts_backend`` edge-trim that is about to remove ~0.74s of padding per
sentence. Until bdd_playwright_service honours such a key, a producer-side key here
would be a payload nothing reads.

⚠️ AND THE TRIM DOES NOT FORCE THE ISSUE. The brief's justification was that the trim
would push the delivered rate through the floor (``tts_backend``'s own
``test_edge_trim.py``: a clip 0.894 -> 0.660). That is the arithmetic of a rate measured
BEFORE the trim being applied to audio rendered AFTER it. The rate this pipeline budgets
with is not a constant: ``resolve_script_sample_wpm`` SYNTHESIZES a sample of the actual
drafted lines through the live backend, per run, so a trimmed backend is measured as
trimmed and the one-term budget tracks it on its own. (The cache that could have carried
a stale one across is per-``run_id``; see wave 14 N2.)

No I/O. Pure arithmetic, so this cannot rot.
"""

from __future__ import annotations

import pytest

#: One real ``af_heart`` sample, the shape ``_SampleTotals`` pools: 35 words over four
#: measured lines, 24.00s of speech and 9.48s of padding across 6 approximate pieces.
_WORDS = 35
_SPEECH_SECONDS = 24.00
_PAUSE_SECONDS = 9.48
_PIECES = 6


def _padded_wpm() -> float:
    """What the budget divides by today."""
    return _WORDS / ((_SPEECH_SECONDS + _PAUSE_SECONDS) / 60.0)


def _speech_wpm() -> float:
    """The un-padded rate ``_log_two_term_decomposition`` derives and throws away."""
    return _WORDS / (_SPEECH_SECONDS / 60.0)


@pytest.mark.parametrize("words", [1])
def test_the_two_term_budget_is_identical_when_pieces_come_from_the_sample(
    words: int,
) -> None:
    """``words / speech_wpm + per_piece x (pieces_per_word x words)`` == ``words / padded_wpm``.

    Both sides are linear in ``words`` with the same coefficient, by construction: the
    padded rate IS the sum of the two terms over the same sample. So the "fix" reduces to
    multiplying and dividing by the identical constant.
    """
    one_term = words * 60.0 / _padded_wpm()

    per_piece_pause = _PAUSE_SECONDS / _PIECES
    pieces_per_word = _PIECES / _WORDS
    two_term = words * 60.0 / _speech_wpm() + per_piece_pause * (pieces_per_word * words)

    assert two_term == pytest.approx(one_term, rel=1e-12)


def test_the_two_term_budget_only_differs_on_a_piece_count_nobody_can_know_yet() -> None:
    """It DOES differ once the piece count is independent of the sample — which is the
    problem, not the solution: at the budget site that count is a guess.

    A 75-word budget (30s at 150 wpm) whose real sentence density is +/-2 pieces off the
    sample's spans more than the entire 3% tolerance the detector leaves, in BOTH
    directions. The wrong answer here is not a rounding error; it is the owner's
    complaint or an overrun, depending on the sign.
    """
    words = 75
    per_piece_pause = _PAUSE_SECONDS / _PIECES
    predicted_pieces = _PIECES / _WORDS * words

    baseline = words * 60.0 / _padded_wpm()
    spread = [
        words * 60.0 / _speech_wpm() + per_piece_pause * (predicted_pieces + delta)
        for delta in (-2, 0, 2)
    ]

    assert spread[1] == pytest.approx(baseline, rel=1e-12)
    # The mis-prediction is worth more than the whole tolerance band in each direction.
    tolerance = 1.0 - 0.97
    assert abs(spread[0] / baseline - 1.0) > tolerance
    assert abs(spread[2] / baseline - 1.0) > tolerance
