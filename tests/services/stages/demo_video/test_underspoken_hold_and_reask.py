"""A/V quality wave 7 — the UNDERSPOKEN beat: quantify the residual, then re-ask for words.

THE DEFECT THESE TESTS PIN. ``vo_rate_for`` clamps a beat whose words are too few for
its seconds to ``VO_RATE_DRAG_FLOOR`` (0.97, the OPERATIVE floor — it used to clamp at
``VO_RATE_NATURAL_FLOOR``, 0.80, which is now only the pathological backstop). Delivered
duration is then ``natural_seconds / 0.97``, which BY CONSTRUCTION is still shorter than
the allotment —
so the listener hears dragged speech AND the beat ships short. Both live complaints
("narration weirdly slows down", a 24.8s master against a 30s order) are that one clamp.
Before this change the residual had no number (``_underspoken_note`` named the beats and
fixed nothing) and no corrective path (the relief loop only ever makes text SHORTER).

Each test below is written to FAIL on the pre-change code:

* ``test_underspoken_hold_quantifies_the_unfilled_residual`` — ``Allotment.hold_seconds``
  did not exist, and the arithmetic it carries was computed nowhere.
* ``test_hold_is_not_folded_into_the_freeze_pad`` — the trap. ``freeze_pad_seconds`` is
  EXTRA WINDOW GRANTED TO THE VOICE (``narrate.fit_target_for`` returns
  ``seconds + freeze_pad_seconds`` and re-synthesizes to fill it), so routing the residual
  through it would drag delivery FURTHER, not less. This asserts the two stay separate.
* ``test_reask_lengthens_an_underspoken_beat_and_clears_the_flag`` — the fix: one bounded
  ``beat-expander`` call per beat, then a deterministic RE-SOLVE over the real text.
* ``test_reask_refuses_a_rewrite_that_overshoots`` / ``..._that_comes_back_short`` — the
  bounds. A re-ask that would convert an underspoken beat into an unsayable one, or that
  does not help, must keep the ORIGINAL text rather than fabricate progress.
"""

from __future__ import annotations

import dataclasses
import math
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal, cast

import pytest

if TYPE_CHECKING:
    from src.services.llm import WMClient

from src.services.stages.demo_video.reconcile_timeline import (
    _EXPAND_ASK_RATE,
    _EXPAND_MAX_PASSES,
    ReconcileTimelineStage,
    _TimingConfig,
)
from src.services.stages.shared.time_budget import (
    DEFAULT_FREEZE_PAD_CEILING,
    DEFAULT_WPM_NATURAL,
    NARRATION_BUDGET_RATE,
    VO_RATE_DRAG_FLOOR,
    TimeDemand,
    collect_demands,
    underspoken_hold,
    vo_rate_for,
)
from src.services.stages.shared.time_budget import (
    reconcile_timeline as solve_timeline,
)


class _FakeWM:
    """A WMClient stand-in that returns one canned ``{"text": ...}`` per call.

    Records every call so a test can assert the re-ask is BOUNDED — one per beat, no
    re-ask of a re-ask.
    """

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    async def message(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        reply = self._replies.pop(0) if self._replies else ""

        return SimpleNamespace(output={"text": reply})


def _cfg(target: float = 30.0) -> _TimingConfig:
    return _TimingConfig(
        mode="auto",
        target=target,
        tolerance_pct=10.0,
        wpm_band=(0.5, 2.0),
        wpm_natural=DEFAULT_WPM_NATURAL,
        freeze_pad_ceiling_seconds=DEFAULT_FREEZE_PAD_CEILING,
        vo_rate_drag_floor=VO_RATE_DRAG_FLOOR,
        relief_mode="auto",
        relief_iterations=2,
        relief_threshold=4.0,
    )


def _stage(wm: _FakeWM) -> ReconcileTimelineStage:
    stage = ReconcileTimelineStage()
    stage._wm = cast("WMClient", wm)
    return stage


# ---------------------------------------------------------------------------
# 1. The residual gets a number.
# ---------------------------------------------------------------------------


def test_underspoken_hold_quantifies_the_unfilled_residual() -> None:
    """The seconds the clamped rate leaves unfilled — the number the note was missing.

    6 words at 150 wpm is 2.40s of natural speech. Spoken at natural speed (never
    slowed) it is delivered in 2.40s, against a 10.0s allotment ⇒ 7.60s unfilled, capped
    at the 1.50s per-beat ceiling. (At the old 0.97 / 0.80 clamps the same beat was
    dragged to 2.4742s / 3.00s — the residual only ever grows as the drag is removed.)

    FAILS BEFORE: ``underspoken_hold`` did not exist.
    """
    allotment_seconds = 10.0
    words = 6

    _, _, underspoken = vo_rate_for(allotment_seconds, words, DEFAULT_WPM_NATURAL)
    assert underspoken, "precondition: this beat must be the clamped case"

    natural = words / DEFAULT_WPM_NATURAL * 60.0
    spoken = natural / 1.0
    assert spoken == pytest.approx(2.40, abs=1e-4)
    # The defect in one assertion: at natural speed the beat cannot fill its allotment.
    assert spoken < allotment_seconds

    hold = underspoken_hold(allotment_seconds, words, DEFAULT_WPM_NATURAL)
    assert hold == pytest.approx(DEFAULT_FREEZE_PAD_CEILING)

    # Uncapped, the residual is the whole gap; the cap is what bounds a dead frame.
    generous = underspoken_hold(allotment_seconds, words, DEFAULT_WPM_NATURAL, ceiling=99.0)
    assert generous == pytest.approx(allotment_seconds - spoken)


def test_a_beat_that_is_not_underspoken_holds_nothing() -> None:
    """No flag ⇒ no hold, so the caller may apply it unconditionally.

    FAILS BEFORE: same — the function did not exist.
    """
    # 40 words at 150 wpm = 16.0s natural, which OVERRUNS a 10s allotment.
    _, _, underspoken = vo_rate_for(10.0, 40, DEFAULT_WPM_NATURAL)
    assert not underspoken
    assert underspoken_hold(10.0, 40, DEFAULT_WPM_NATURAL) == 0.0
    # And the divide-by-zero-safe edges stay neutral rather than returning a corrupt float.
    assert underspoken_hold(10.0, 0, DEFAULT_WPM_NATURAL) == 0.0
    assert underspoken_hold(0.0, 6, DEFAULT_WPM_NATURAL) == 0.0
    assert underspoken_hold(float("nan"), 6, DEFAULT_WPM_NATURAL) == 0.0


def test_solve_timeline_carries_the_hold_on_the_flagged_beat() -> None:
    """The solver stamps the residual next to the flag it belongs to.

    FAILS BEFORE: ``Allotment`` had no ``hold_seconds``.
    """
    demands = [
        TimeDemand(
            id="clip-1",
            kind="vo",
            min=1.0,
            ideal=10.0,
            max=15.0,
            elasticity=1.0,
            pinned=False,
            word_count=6,
        ),
    ]
    allotments = solve_timeline(demands, target=10.0, tolerance_pct=10.0)
    (beat,) = allotments
    assert beat.underspoken is True
    assert beat.hold_seconds > 0.0
    assert beat.vo_rate == 1.0


def test_hold_is_not_folded_into_the_freeze_pad() -> None:
    """⛔ THE TRAP. The two pads point in OPPOSITE directions and must stay separate.

    ``freeze_pad_seconds`` is window granted TO THE VOICE — ``narrate.fit_target_for``
    returns ``seconds + freeze_pad_seconds`` and re-synthesizes the line to fill it. If
    the underspoken residual were added there, narrate would drag delivery down to its
    ``_FIT_FLOOR`` trying to fill a window the words already cannot fill: the "narration
    weirdly slows down" complaint, made WORSE by its own fix.

    FAILS BEFORE: ``hold_seconds`` did not exist, so there was nothing to keep separate —
    and it fails after any change that folds one field into the other.
    """
    demands = [
        TimeDemand(
            id="clip-1",
            kind="vo",
            min=1.0,
            ideal=10.0,
            max=15.0,
            elasticity=1.0,
            pinned=False,
            word_count=6,
        ),
    ]
    (beat,) = solve_timeline(demands, target=10.0, tolerance_pct=10.0)
    assert beat.hold_seconds > 0.0
    # A non-footage beat's narration does not overrun any clip, so it is granted NO
    # extra speaking window. The residual must not have leaked into this field.
    assert beat.freeze_pad_seconds == 0.0


# ---------------------------------------------------------------------------
# 2. The re-ask.
# ---------------------------------------------------------------------------


#: A FOOTAGE-PINNED beat, because that is what a demo_video spine actually is: its
#: seconds are the ffprobe'd clip length (min == ideal == max), so the word count is the
#: only lever left. 6 words against 10.00s ⇒ 2.40s of natural speech, clamped to the 0.80
#: floor ⇒ 3.00s delivered, 7.00s unfilled. The measured word floor is 20.
_CLIPS = {"clip-1": 10.0}


def _natural_ask(window_seconds: float, wpm: float = DEFAULT_WPM_NATURAL) -> int:
    """The word count that fills ``window_seconds`` at NATURAL pace (rate 1.0).

    Derived from the same inequality ``vo_rate_for`` uses, at ``_EXPAND_ASK_RATE``
    instead of the 0.80 floor — so this helper cannot silently agree with a
    floor-rate ask.
    """
    return math.ceil(window_seconds * wpm * _EXPAND_ASK_RATE / 60.0)


def _underspoken_spine() -> list[dict[str, object]]:
    return [{"id": "clip-1", "text": "One two three four five six", "seconds": 10.0}]


@pytest.mark.asyncio
async def test_reask_lengthens_an_underspoken_beat_and_clears_the_flag() -> None:
    """THE FIX. One bounded call, then a RE-SOLVE over the text that now exists.

    The re-solve is the point: the stage never asserts it improved something, it asks
    the deterministic solver to re-read the real text. A beat that is genuinely fixed
    stops being flagged; one that is not, stays flagged.

    FAILS BEFORE: ``_expand_underspoken`` did not exist — there was NO re-ask anywhere in
    the pipeline, which is the headline finding.
    """
    cfg = _cfg(target=10.0)
    lines = _underspoken_spine()
    demands = collect_demands(
        lines, clip_seconds=_CLIPS, wpm_band=cfg.wpm_band, wpm_natural=cfg.wpm_natural
    )
    allotments = solve_timeline(demands, target=cfg.target, tolerance_pct=cfg.tolerance_pct)
    assert any(a.underspoken for a in allotments), "precondition: the beat starts flagged"

    # 25 words: the NATURAL-RATE count for 10s (10 * 150 * 1.0 / 60 = 25), which is what
    # the re-ask now asks for. 20 — the 0.80-floor count — is deliberately NOT the ask:
    # see `_EXPAND_ASK_RATE`.
    long_text = " ".join(f"w{i}" for i in range(26))
    wm = _FakeWM([long_text])

    new_lines, _, new_allotments, notes = await _stage(wm)._expand_underspoken(
        lines, demands, allotments, cfg, clip_seconds=_CLIPS
    )

    assert len(wm.calls) == 1, "a pass that REACHES the ask must not buy another one"
    assert f"AT LEAST {_natural_ask(10.0)} words" in wm.calls[0]["message"], (
        "the ASK must be the natural-rate count, not the 0.80-floor count"
    )
    assert wm.calls[0]["worker_name"] == "beat-expander"
    assert wm.calls[0]["output_schema"] == {
        "type": "object",
        "required": ["text"],
        "properties": {"text": {"type": "string"}},
    }
    assert new_lines[0]["text"] == long_text
    assert not any(a.underspoken for a in new_allotments), "the re-solve must clear the flag"
    assert notes and "UNDERSPOKEN RE-ASK" in notes[0]


@pytest.mark.asyncio
async def test_reask_that_comes_back_short_keeps_the_original() -> None:
    """A rewrite that does not help is DROPPED, never applied, and the beat stays flagged.

    FAILS BEFORE: no re-ask existed to bound.
    """
    cfg = _cfg(target=10.0)
    lines = _underspoken_spine()
    demands = collect_demands(
        lines, clip_seconds=_CLIPS, wpm_band=cfg.wpm_band, wpm_natural=cfg.wpm_natural
    )
    allotments = solve_timeline(demands, target=cfg.target, tolerance_pct=cfg.tolerance_pct)

    wm = _FakeWM(["Two words"])  # shorter than the original
    new_lines, _, new_allotments, notes = await _stage(wm)._expand_underspoken(
        lines, demands, allotments, cfg, clip_seconds=_CLIPS
    )

    assert new_lines[0]["text"] == lines[0]["text"], "the ORIGINAL text must survive"
    assert any(a.underspoken for a in new_allotments), "and the beat must stay flagged"
    assert notes and "nothing usable" in notes[0]


@pytest.mark.asyncio
async def test_reask_refuses_a_rewrite_that_overshoots() -> None:
    """Overshooting trades an underspoken beat for an UNSAYABLE one — strictly worse.

    The compiler can hold a frame but it cannot cut audio, so a beat whose words now
    overrun its clip is a worse defect than the one being fixed. The rewrite is refused
    and the original kept.

    FAILS BEFORE: no re-ask, and therefore no ceiling on one.
    """
    cfg = _cfg(target=10.0)
    lines = _underspoken_spine()
    demands = collect_demands(
        lines, clip_seconds=_CLIPS, wpm_band=cfg.wpm_band, wpm_natural=cfg.wpm_natural
    )
    allotments = solve_timeline(demands, target=cfg.target, tolerance_pct=cfg.tolerance_pct)

    wm = _FakeWM([" ".join(f"w{i}" for i in range(400))])
    new_lines, _, _, notes = await _stage(wm)._expand_underspoken(
        lines, demands, allotments, cfg, clip_seconds=_CLIPS
    )

    assert new_lines[0]["text"] == lines[0]["text"]
    assert notes and "nothing usable" in notes[0]


@pytest.mark.asyncio
async def test_an_overshoot_on_pass_one_still_buys_the_next_pass() -> None:
    """Showcase single-beat: an overshoot spends its pass, it does not END the beat.

    The URL and ungrounded refusals already ``continue``; the overshoot branch used to
    ``break``, so a one-beat film got exactly one ask out of ``_EXPAND_MAX_PASSES``.
    Pass 1 overshoots, pass 2 lands in band ⇒ pass 2's text ships.

    FAILS BEFORE: one call, the original text kept, the beat still flagged.
    """
    cfg = _cfg(target=10.0)
    lines = _underspoken_spine()
    demands = collect_demands(
        lines, clip_seconds=_CLIPS, wpm_band=cfg.wpm_band, wpm_natural=cfg.wpm_natural
    )
    allotments = solve_timeline(demands, target=cfg.target, tolerance_pct=cfg.tolerance_pct)

    in_band = " ".join(f"w{i}" for i in range(_natural_ask(10.0)))
    wm = _FakeWM([" ".join(f"w{i}" for i in range(400)), in_band, "never reached"])
    new_lines, _, new_allotments, _notes = await _stage(wm)._expand_underspoken(
        lines, demands, allotments, cfg, clip_seconds=_CLIPS
    )

    assert len(wm.calls) == 2, "the overshoot must buy pass 2, and pass 2 reaching the ask ends it"
    assert "Beat (6 words)" in wm.calls[1]["message"], "pass 2 re-asks against the kept text"
    assert new_lines[0]["text"] == in_band
    assert not any(a.underspoken for a in new_allotments)


@pytest.mark.asyncio
async def test_reask_never_rewrites_a_claim_bearing_beat() -> None:
    """Writing INTO a beat that carries a grounded claim is how an unsupported one is born.

    FAILS BEFORE: no re-ask existed, so nothing enforced the exclusion.
    """
    cfg = _cfg(target=10.0)
    lines = [{**_underspoken_spine()[0], "claim_status": "grounded"}]
    demands = collect_demands(
        lines, clip_seconds=_CLIPS, wpm_band=cfg.wpm_band, wpm_natural=cfg.wpm_natural
    )
    allotments = solve_timeline(demands, target=cfg.target, tolerance_pct=cfg.tolerance_pct)

    wm = _FakeWM([" ".join(f"w{i}" for i in range(21))])
    new_lines, _, _, notes = await _stage(wm)._expand_underspoken(
        lines, demands, allotments, cfg, clip_seconds=_CLIPS
    )

    assert wm.calls == [], "a claim-bearing beat must never reach the model"
    assert new_lines[0]["text"] == lines[0]["text"]
    assert notes == []


@pytest.mark.asyncio
async def test_a_footage_pinned_beat_is_eligible_to_lengthen() -> None:
    """The asymmetry that makes this pass non-inert on demo_video.

    Trimming EXCLUDES footage-pinned beats (their seconds cannot move, so shortening
    text cannot help). Lengthening must INCLUDE them for exactly the same reason: a beat
    pinned to its measured clip has no seconds lever left, so words are the only lever
    there is. Copying the trim predicate here would have made the whole pass inert on
    demo_video, whose spine is footage beats almost end to end.

    FAILS BEFORE: no re-ask existed.
    """
    cfg = _cfg(target=10.0)
    lines = _underspoken_spine()
    clip_seconds = {"clip-1": 10.0}
    demands = collect_demands(
        lines, clip_seconds=clip_seconds, wpm_band=cfg.wpm_band, wpm_natural=cfg.wpm_natural
    )
    assert any(d.kind == "footage" for d in demands), "precondition: the beat is footage-pinned"
    allotments = solve_timeline(demands, target=cfg.target, tolerance_pct=cfg.tolerance_pct)
    assert any(a.underspoken for a in allotments)

    wm = _FakeWM([" ".join(f"w{i}" for i in range(26))])
    await _stage(wm)._expand_underspoken(lines, demands, allotments, cfg, clip_seconds=clip_seconds)
    assert len(wm.calls) == 1, "a footage-pinned underspoken beat MUST be re-asked"


@pytest.mark.asyncio
async def test_a_wm_outage_degrades_the_run_but_never_fails_it() -> None:
    """The deliverable is the deterministic allotment vector; relief is enrichment on top.

    FAILS BEFORE: the wrapper did not exist.
    """

    class _Boom:
        async def message(self, **kwargs: Any) -> Any:
            raise RuntimeError("worker_management unreachable")

    cfg = _cfg(target=10.0)
    lines = _underspoken_spine()
    demands = collect_demands(
        lines, clip_seconds=_CLIPS, wpm_band=cfg.wpm_band, wpm_natural=cfg.wpm_natural
    )
    allotments = solve_timeline(demands, target=cfg.target, tolerance_pct=cfg.tolerance_pct)

    stage = ReconcileTimelineStage()
    stage._wm = cast("WMClient", _Boom())
    out_lines, _out_demands, out_allotments, notes = await stage._maybe_expand_underspoken(
        lines, demands, allotments, cfg, clip_seconds=_CLIPS
    )

    assert out_lines == lines
    assert out_allotments == allotments
    assert notes and "UNAVAILABLE" in notes[0]


@pytest.mark.asyncio
async def test_the_ask_is_the_natural_rate_count_not_the_floor_count() -> None:
    """2a. Asking for the FLOOR count guarantees a clamp AT the floor.

    Measured live on run 01a0a136: the generator returned 14 words against a 24-word
    budget on every beat, the re-ask topped each one to the count that fills it at
    the delivery floor (21/26/21 at the then-current 0.80), and narrate still recorded "clip-4 needs speed
    0.746 … outside the 0.8-2.0 band; clamped to 0.80". A beat topped up to the floor is
    BY CONSTRUCTION a beat that gets clamped to the floor — which is the owner's "the
    narration weirdly slows down".

    So the prompt must carry the NATURAL-RATE count, not the floor count. Both numbers
    are computed here from the flag's own constants, so this cannot pass by coincidence.

    ⚠️ WHAT THIS TEST CAN AND CANNOT PROVE AFTER WAVE 13. It used to prove the point by
    showing the prompt carried one integer (25, the natural count) and not another (20,
    the floor count). That worked while the floor was 0.80: the ask rate (1.0) and the
    floor were 25% apart, so any window separated the two counts.

    The drag floor is now 0.97, i.e. 3% below the ask rate, so ``math.ceil`` COLLAPSES
    the two counts for every window under ~13.6s — and this fixture's beat is a pinned
    10.0s clip, where BOTH counts are 25. Raising ``cfg.target`` does not help: the ask
    is computed from the BEAT's own window (``allotment.seconds + freeze_pad_seconds``),
    not from the timeline target, and reshaping the shared ``_CLIPS`` fixture to lengthen
    the beat would disturb the seventeen other tests that depend on it.

    So the integer-vs-integer proof is no longer available at this fixture, and pretending
    otherwise would leave ``assert 25 > 25`` and a negative assertion that is trivially
    true because ``floor == ask``. Instead this now proves the two things that ARE
    falsifiable, and states plainly that the third is gone:

    1. the prompt carries the count computed at ``_EXPAND_ASK_RATE`` (mutate that
       constant to the floor and the number in the message changes);
    2. the two constants are still ORDERED the right way round, which is the property
       the original integers were only ever a proxy for;
    3. and the two formulas still disagree wherever ``ceil`` does not collapse them, so
       the distinction is real arithmetic rather than a coincidence of one fixture.

    FAILS BEFORE: ``_expand_msg(original, word_floor)`` put the floor count in the prompt.
    """
    cfg = _cfg(target=10.0)
    lines = _underspoken_spine()
    demands = collect_demands(
        lines, clip_seconds=_CLIPS, wpm_band=cfg.wpm_band, wpm_natural=cfg.wpm_natural
    )
    allotments = solve_timeline(demands, target=cfg.target, tolerance_pct=cfg.tolerance_pct)

    # (2) The constants are ordered: the ask aims ABOVE the accept bar. This is the
    # property the old integer comparison was a proxy for, and it is what actually
    # matters — asking for exactly the floor count guarantees a clamp at the floor.
    assert _EXPAND_ASK_RATE > VO_RATE_DRAG_FLOOR, "the ask must aim above the accept bar"

    # (3) And the two formulas genuinely disagree wherever `ceil` does not collapse
    # them, so (2) is not a distinction without a difference. 40.0s: 100 vs 97.
    long_window = 40.0
    assert _natural_ask(long_window) > math.ceil(
        long_window * DEFAULT_WPM_NATURAL * VO_RATE_DRAG_FLOOR / 60.0
    )

    ask = _natural_ask(10.0)
    wm = _FakeWM([" ".join(f"w{i}" for i in range(ask + 1))])
    await _stage(wm)._expand_underspoken(lines, demands, allotments, cfg, clip_seconds=_CLIPS)

    # (1) The prompt carries the count taken at `_EXPAND_ASK_RATE`. Mutating that
    # constant to `VO_RATE_DRAG_FLOOR` changes this number and fails here.
    message = wm.calls[0]["message"]
    assert f"AT LEAST {ask} words" in message


@pytest.mark.asyncio
async def test_a_partial_expansion_buys_another_pass_up_to_the_bound() -> None:
    """2b. One shot left a beat short whenever the model under-delivered on the target.

    Models routinely return fewer words than asked for. With a single pass such a beat
    kept its partial text and stayed flagged, with no second chance — so the re-ask
    existed and still did not clear the defect it was written for.

    Three replies that each ADD words: the loop must keep going while progress is real,
    feed the CURRENT count back into each prompt (never the original's), and stop at
    ``_EXPAND_MAX_PASSES``.

    FAILS BEFORE: exactly one call was made and the 12-word reply was the final text.
    """
    cfg = _cfg(target=10.0)
    lines = _underspoken_spine()
    demands = collect_demands(
        lines, clip_seconds=_CLIPS, wpm_band=cfg.wpm_band, wpm_natural=cfg.wpm_natural
    )
    allotments = solve_timeline(demands, target=cfg.target, tolerance_pct=cfg.tolerance_pct)

    ask = _natural_ask(10.0)
    replies = [
        " ".join(f"w{i}" for i in range(12)),
        " ".join(f"w{i}" for i in range(18)),
        " ".join(f"w{i}" for i in range(ask)),
    ]
    wm = _FakeWM([*replies, " ".join(f"w{i}" for i in range(ask + 5))])

    new_lines, _, new_allotments, _notes = await _stage(wm)._expand_underspoken(
        lines, demands, allotments, cfg, clip_seconds=_CLIPS
    )

    assert len(wm.calls) == 3, "partial progress must buy another pass"
    assert len(wm.calls) <= _EXPAND_MAX_PASSES, "and never more than the bound"
    # Each prompt restates what the beat HAS NOW — 6, then 12, then 18 words.
    assert [c["message"].count("Beat (") for c in wm.calls] == [1, 1, 1]
    assert "Beat (6 words)" in wm.calls[0]["message"]
    assert "Beat (12 words)" in wm.calls[1]["message"]
    assert "Beat (18 words)" in wm.calls[2]["message"]
    # The LAST accepted reply is what ships, and it clears the flag.
    assert new_lines[0]["text"] == replies[2]
    assert not any(a.underspoken for a in new_allotments)


@pytest.mark.asyncio
async def test_a_pass_that_adds_nothing_stops_the_loop() -> None:
    """The loop cannot cycle: no progress ends the beat, and the best-so-far is kept.

    FAILS BEFORE: there was no loop to stop (and after 2b, a loop with no no-progress
    guard would burn all three calls on every beat a model stalls on).
    """
    cfg = _cfg(target=10.0)
    lines = _underspoken_spine()
    demands = collect_demands(
        lines, clip_seconds=_CLIPS, wpm_band=cfg.wpm_band, wpm_natural=cfg.wpm_natural
    )
    allotments = solve_timeline(demands, target=cfg.target, tolerance_pct=cfg.tolerance_pct)

    grown = " ".join(f"w{i}" for i in range(12))
    # Pass 2 returns the SAME length as pass 1 ⇒ no progress ⇒ stop, without a 3rd call.
    wm = _FakeWM([grown, " ".join(f"x{i}" for i in range(12)), "never reached"])

    new_lines, _, _, _notes = await _stage(wm)._expand_underspoken(
        lines, demands, allotments, cfg, clip_seconds=_CLIPS
    )

    assert len(wm.calls) == 2, "a no-progress pass must end the beat, not spend the budget"
    assert new_lines[0]["text"] == grown, "the best text so far is kept, not discarded"


# ---------------------------------------------------------------------------
# 3. The note now carries the seconds, not just the beat names.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 4. The RECORD must not contradict itself about the delivered rate.
#
# MEASURED DEFECT, live run 01a0a582-2531-7145-ba7d-a1295e92b232. One
# reconcile_timeline stage-output row carried BOTH:
#
#   lines[].vo_rate                  1.0383 / 1.0    / 1.0
#   timeline_manifest.beats[].vo_rate  0.80 / 0.80   / 0.80   <- the floor clamp
#
# for the same three clips, plus a note saying the re-ask had SUCCEEDED. 0.80 is
# exactly VO_RATE_NATURAL_FLOOR — the owner's "narration weirdly slows down"
# signature — so whichever array a reader opened decided the verdict on the
# headline complaint. Cause: the manifest is built inside `_relief_loop`, which
# runs BEFORE `_expand_underspoken` re-asks and RE-SOLVES; nothing then said the
# manifest rows had been superseded.
# ---------------------------------------------------------------------------


def _manifest_from(allotments: list[Any], demands: list[TimeDemand], target: float) -> Any:
    """The manifest exactly as ``_relief_loop`` builds it — from PRE-re-ask allotments."""
    from src.services.stages.shared.time_budget import (
        TimelineManifest,
        build_timeline_manifest,
    )

    return TimelineManifest.model_validate(
        build_timeline_manifest(
            demands,
            allotments,
            target_seconds=target,
            loop_iterations=1,
            converged=True,
            pacing_score=4.0,
        )
    )


# ---------------------------------------------------------------------------
# 5. …and the marker is actually WIRED into the stage, not merely available.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_names_the_superseded_rows_on_the_live_stage_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole path: relief loop builds the manifest, the re-ask supersedes a row, the
    stage output says so.

    Spine reproduces the live shape: a footage-PINNED beat with too few words for its
    measured clip (⇒ UNDERSPOKEN, rate clamped to 0.80) alongside an over-long VO beat
    that pushes Σmin past the target (⇒ INFEASIBLE, which ARMS the relief loop). The
    director is stubbed to propose nothing, so the manifest is built over the clamped
    rate; the beat-expander then returns a genuinely longer line and the re-solve lifts
    the rate off the floor — exactly the disagreement measured on run 01a0a582.

    FAILS BEFORE: ``execute`` never marked the manifest, so the stage output carried
    ``beats[].vo_rate == 0.80`` next to ``lines[].vo_rate > 0.80`` with nothing saying
    which was current.
    """
    # The SPINE word cap (``_REASK_TOTAL_WORD_RATIO``) would refuse this device's re-ask —
    # the spine already sits past 1.10x the body's word budget. The subject here is the
    # manifest wiring, not that cap (pinned in test_reconcile_footage_trim_and_cards.py),
    # so it is lifted for this test only.
    monkeypatch.setattr(
        "src.services.stages.demo_video.reconcile_timeline._REASK_TOTAL_WORD_RATIO", 100.0
    )
    from uuid import uuid4

    from src.services.runs.run_context import RunContext
    from src.services.stages.demo_video.reconcile_timeline import ReconcileTimelineStage

    # 26 words: past the 25-word natural-rate ask for the 10.0s clip, inside the beat's
    # window at the solved rate (10.0s + the 1.5s freeze-pad ceiling = 28 words at 150 wpm).
    # 40 used to be applied here; that is the run-01a0b007 overshoot and is now refused.
    long_text = " ".join(f"w{i}" for i in range(26))

    class _KeyedWM:
        """Keyed by ``worker_name``: no trim directives, a real expansion, a neutral score."""

        def __init__(self) -> None:
            self.calls: list[str] = []

        async def message(self, **kwargs: Any) -> Any:
            worker = str(kwargs.get("worker_name") or "")
            self.calls.append(worker)
            if worker == "pacing-critic":
                return SimpleNamespace(output={"overall_score": 4}, output_type="object")
            if worker == "beat-expander":
                return SimpleNamespace(output={"text": long_text}, output_type="object")
            # timing-director (structural pre-pass AND trim loop): propose nothing.
            return SimpleNamespace(output={"directives": []}, output_type="object")

    lines: list[dict[str, Any]] = [
        # Footage-pinned, 6 words against a measured 10.0s clip ⇒ underspoken at 0.80.
        {"id": "clip-1", "text": "One two three four five six", "seconds": None, "clip": "clip-1"},
        # 40 words ⇒ demand floor 40/170*60 = 14.12s; with the 10.0s clip, Σmin > 15.0.
        {"id": "clip-vo", "text": " ".join(["vo"] * 40), "seconds": None, "clip": None},
    ]
    wm = _KeyedWM()
    stage = ReconcileTimelineStage(client=cast("WMClient", wm))
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "direction": {
                "timing": {
                    "mode": "auto",
                    "relief": {"mode": "auto", "iterations": 2, "score_threshold": 4.0},
                }
            },
            "deliverable": {"target_duration_seconds": 15.0},
        },
        prior_run_outputs={
            "demo_video_gen": {
                "lines": lines,
                "bdd_job_id": "",
                "features_generated": 0,
                "package_url": "",
            },
            "demo_video_record": {
                "clips": [
                    {
                        "id": "clip-1",
                        "line_id": "clip-1",
                        "video_url": "https://minio.test/clip-1.mp4",
                        "seconds": 10.0,
                    }
                ]
            },
        },
    )

    out = await stage.handle(stage.input_model(), ctx)

    assert "beat-expander" in wm.calls, "precondition: the re-ask must have run"
    manifest = out.timeline_manifest
    assert manifest is not None, "precondition: the relief loop must have armed"
    snapshot = {beat.id: (beat.seconds, beat.vo_rate) for beat in manifest.beats}
    delivered = {line.id: (line.seconds, line.vo_rate) for line in out.lines}
    # PRECONDITION: the re-ask really moved clip-1's rate off the floor clamp.
    assert (delivered["clip-1"][1] or 0.0) > VO_RATE_DRAG_FLOOR

    named = set(manifest.run.beats_superseded_by_underspoken_reask)
    # clip-1 was never dragged below natural, before or after the re-ask, so its row
    # did not go stale and is not named (the marker's positive path is pinned in
    # `test_a_manifest_row_the_reask_invalidated_is_named_on_the_record`).
    assert delivered["clip-1"][1] == 1.0
    assert snapshot["clip-1"][1] == 1.0
    # … and since run 01a0b007 the beats[] delivery rate is REFRESHED, so the manifest and
    # the spine agree on vo_rate (``_refresh_beat_delivery``).
    assert {k: v[1] for k, v in snapshot.items()} == {
        k: v[1] for k, v in delivered.items() if k in snapshot
    }
    # `seconds` is deliberately NOT refreshed (relief sizing reads it), so ANY remaining
    # (seconds, vo_rate) disagreement must still be a NAMED row — the full pre-refresh
    # invariant, unchanged.
    assert _unnamed_disagreements(snapshot, delivered, named) == set()
    # …and that check can fail: an unnamed beat whose `seconds` disagrees is caught.
    unnamed = next(beat_id for beat_id in snapshot if beat_id not in named)
    seconds, rate = snapshot[unnamed]
    tampered = {**snapshot, unnamed: ((seconds or 0.0) + 1.0, rate)}
    assert _unnamed_disagreements(tampered, delivered, named) == {unnamed}


def _unnamed_disagreements(
    snapshot: dict[str, tuple[Any, Any]],
    delivered: dict[str, tuple[Any, Any]],
    named: set[str],
) -> set[str]:
    """Manifest rows whose (seconds, vo_rate) disagree with the spine and are NOT named."""
    return {
        beat_id
        for beat_id, snap in snapshot.items()
        if beat_id in delivered and delivered[beat_id] != snap and beat_id not in named
    }


# ---------------------------------------------------------------------------
# 6. …and the RUN-LEVEL budget numbers are refreshed, not just the beat rows.
# ---------------------------------------------------------------------------


def _band_cfg(*, target: float) -> _TimingConfig:
    """``_cfg`` with a REAL wpm band, so ``TimeDemand.min`` is eye-checkable.

    ``_cfg``'s ``(0.5, 2.0)`` band is a rate MULTIPLIER pair, but ``collect_demands``
    reads ``wpm_band`` as absolute words-per-minute — at 2.0 wpm a 6-word beat floors at
    180s, which makes any budget arithmetic unreadable. At ``wpm_max = 200`` the floor is
    exactly ``words x 0.3s``.
    """
    return _TimingConfig(
        mode="auto",
        target=target,
        tolerance_pct=10.0,
        wpm_band=(100.0, 200.0),
        wpm_natural=DEFAULT_WPM_NATURAL,
        freeze_pad_ceiling_seconds=DEFAULT_FREEZE_PAD_CEILING,
        vo_rate_drag_floor=VO_RATE_DRAG_FLOOR,
        relief_mode="auto",
        relief_iterations=2,
        relief_threshold=4.0,
    )


def _relief_loop_manifest(demands: list[TimeDemand], allotments: list[Any], cfg: Any) -> Any:
    """The manifest EXACTLY as ``_relief_loop`` builds it — overrun included."""
    from src.services.stages.shared.time_budget import (
        TimelineManifest,
        build_timeline_manifest,
    )

    return TimelineManifest.model_validate(
        build_timeline_manifest(
            demands,
            allotments,
            target_seconds=cfg.target,
            loop_iterations=1,
            converged=True,
            pacing_score=4.0,
            overrun_seconds=max(0.0, sum(d.min for d in demands) - cfg.target),
        )
    )


@pytest.mark.asyncio
async def test_handle_returns_a_manifest_carrying_the_post_reask_overrun(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """…and the refresh is WIRED into the stage, not merely available to a unit test.

    The two tests above call ``_refresh_run_budget`` directly, so deleting its call site in
    ``execute`` leaves them both green. This one goes through :meth:`handle` and never names
    the function: it asserts only what a caller of the stage can see — the
    ``timeline_manifest`` that comes OUT — so the wiring is what is under test.

    THE SPINE, chosen so the re-ask lands on an ELASTIC beat. ``wpm = {100, 150, 200}`` ⇒
    ``TimeDemand.min = words x 0.3s`` and a beat's ``max`` is ``words x 0.6s``.

    * ``clip-x`` is footage-PINNED to a 2.0s clip carrying 40 words ⇒ UNSAYABLE, which is
      what ARMS the relief loop (and therefore builds a manifest at all) on a timeline that
      is nowhere near infeasible. Its own demand is pinned at 2.0s, so it contributes
      nothing to the movement being measured — it is the arming device, not the subject.
    * ``beat-a`` / ``beat-b`` are ELASTIC, 20 words each ⇒ ``Σmin = 6.0 + 6.0 + 2.0 =
      14.0s`` against a 25.0s order, so BEFORE the re-ask ``overrun_seconds == 0.0`` and
      NO beat is infeasible. The 11.0s of slack water-fills both to their 12.0s ``max``,
      where 20 words of natural speech (``20 / 150 x 60 = 8.0s``) need rate ``8/12 = 0.67``
      — below the drag floor ⇒ both are UNDERSPOKEN and the re-ask fires on them.
    * The ``beat-expander`` returns 40 words for each ⇒ AFTER, ``Σmin = 12.0 + 12.0 + 2.0 =
      26.0s`` against the same 25.0s order: the run is 1.0s OVER and both elastic beats are
      now infeasible.

    FAILS BEFORE: with the ``_refresh_run_budget`` call removed from ``execute``, the
    manifest handed back still says ``overrun_seconds == 0.0`` / ``infeasible_beat_ids ==
    []`` — the PRE-re-ask timeline, which no longer exists. ``runs/structural_relief.py``
    sizes a PAID child render's cut from exactly those numbers.
    """
    # The SPINE word cap (``_REASK_TOTAL_WORD_RATIO``) would refuse this device's re-ask —
    # the spine already sits past 1.10x the body's word budget. The subject here is the
    # manifest wiring, not that cap (pinned in test_reconcile_footage_trim_and_cards.py),
    # so it is lifted for this test only.
    monkeypatch.setattr(
        "src.services.stages.demo_video.reconcile_timeline._REASK_TOTAL_WORD_RATIO", 100.0
    )
    from uuid import uuid4

    from src.services.runs.run_context import RunContext
    from src.services.stages.demo_video.reconcile_timeline import ReconcileTimelineStage

    forty = " ".join(f"w{i}" for i in range(40))

    class _KeyedWM:
        """Keyed by ``worker_name``: a real expansion, no trim directives, a neutral score."""

        def __init__(self) -> None:
            self.calls: list[str] = []

        async def message(self, **kwargs: Any) -> Any:
            worker = str(kwargs.get("worker_name") or "")
            self.calls.append(worker)
            if worker == "pacing-critic":
                return SimpleNamespace(output={"overall_score": 4}, output_type="object")
            if worker == "beat-expander":
                return SimpleNamespace(output={"text": forty}, output_type="object")
            # timing-director / beat-rewriter: propose nothing, shorten nothing.
            return SimpleNamespace(output={"directives": []}, output_type="object")

    lines: list[dict[str, Any]] = [
        {"id": "beat-a", "text": " ".join(f"a{i}" for i in range(20)), "seconds": None},
        {"id": "beat-b", "text": " ".join(f"b{i}" for i in range(20)), "seconds": None},
        {
            "id": "clip-x",
            "text": " ".join(f"x{i}" for i in range(40)),
            "seconds": None,
            "clip": "clip-x",
        },
    ]
    clip_seconds = {"clip-x": 2.0}

    # PRECONDITION, computed rather than assumed: the timeline the relief loop builds its
    # manifest over is UNDER budget, so a stale ``overrun_seconds`` is exactly 0.0.
    before = collect_demands(
        lines, clip_seconds=clip_seconds, wpm_band=(100.0, 200.0), wpm_natural=150.0
    )
    assert sum(d.min for d in before) == pytest.approx(14.0)
    assert max(0.0, sum(d.min for d in before) - 25.0) == pytest.approx(0.0)

    wm = _KeyedWM()
    stage = ReconcileTimelineStage(client=cast("WMClient", wm))
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "direction": {
                "timing": {
                    "mode": "auto",
                    "wpm": {"min": 100, "natural": 150, "max": 200},
                    # 40 words need 16.0s at 150 wpm against an 11.5s allotment; the
                    # re-ask admits a reply only inside allotment + freeze-pad ceiling, so
                    # 5.0s keeps this device admissible (clip-x stays unsayable, verified).
                    "freeze_pad_ceiling_seconds": 5.0,
                    "relief": {"mode": "auto", "iterations": 2, "score_threshold": 4.0},
                }
            },
            "deliverable": {"target_duration_seconds": 25.0},
        },
        prior_run_outputs={
            "demo_video_gen": {
                "lines": lines,
                "bdd_job_id": "",
                "features_generated": 0,
                "package_url": "",
            },
            "demo_video_record": {
                "clips": [
                    {
                        "id": "clip-x",
                        "line_id": "clip-x",
                        "video_url": "https://minio.test/clip-x.mp4",
                        "seconds": 2.0,
                    }
                ]
            },
        },
    )

    out = await stage.handle(stage.input_model(), ctx)

    assert "beat-expander" in wm.calls, "precondition: the re-ask must have run"
    manifest = out.timeline_manifest
    assert manifest is not None, "precondition: the relief loop must have armed"
    # PRECONDITION: the re-ask really did lengthen the ELASTIC beats (not the pinned one).
    delivered_words = {ln.id: len(str(ln.text).split()) for ln in out.lines}
    assert delivered_words["beat-a"] == 40
    assert delivered_words["beat-b"] == 40

    # THE ASSERTION. The manifest handed to the caller carries the POST-re-ask budget.
    assert manifest.run.overrun_seconds == pytest.approx(1.0), (
        "the manifest handed out of handle() still carries the PRE-re-ask overrun — "
        "structural_relief would size a paid child render's cut from a timeline that no "
        "longer exists"
    )
    assert manifest.run.infeasible_beat_ids == ["beat-a", "beat-b"]
    # …and the CLASSIFICATION the spawn gate reads is post-re-ask too. Before the re-ask
    # Σmin 14.0 <= 25.0 ⇒ the relief loop classified ``converged``; after it the spine is
    # 1.0s over with only the 2.0s clip protected ⇒ ``progress``. A stale ``converged``
    # here is the wiring being absent.
    assert manifest.run.convergence_outcome == "progress", (
        "the manifest handed out of handle() still carries the PRE-re-ask "
        "convergence_outcome — the paid spawn gate reads a timeline that no longer exists"
    )


# ---------------------------------------------------------------------------
# convergence_outcome is a spawn GATE — it must be post-re-ask in BOTH directions.
#
# #1042 re-measures the speech rate over the re-asked text and re-collects EVERY demand
# under the new wpm band, so a CLAIM-BEARING (protected) beat's ``min`` moves too, and
# with it the protected floor ``_classify_convergence`` tests. The spine below makes that
# arithmetic eye-checkable: one protected 40-word claim beat, one elastic 60-word beat,
# a 25.0s order.
#
#   * band max 200 wpm ⇒ min = words x 0.3s: claim 12.0s, elastic 18.0s ⇒ Σmin 30.0 > 25,
#     protected floor 12.0 <= 25 ⇒ ``progress`` (in-place trimming can still converge).
#   * band max  80 wpm ⇒ min = words x 0.75s: claim 30.0s > 25 on its own ⇒
#     ``cant_converge`` (only a gen re-run that drops protected beats can help).
# ---------------------------------------------------------------------------


def _protected_floor_spine() -> list[dict[str, object]]:
    return [
        {
            "id": "claim",
            "text": " ".join(f"c{i}" for i in range(40)),
            "seconds": None,
            "claim_status": "grounded",
        },
        {"id": "elastic", "text": " ".join(f"e{i}" for i in range(60)), "seconds": None},
    ]


def _manifest_classified_under(
    lines: list[dict[str, object]],
    wpm_max: float,
    *,
    outcome: Literal["relief_failed"] | None = None,
) -> tuple[Any, _TimingConfig]:
    """The manifest ``_relief_loop`` would build under a ``(100, wpm_max)`` band."""
    from src.services.stages.demo_video.reconcile_timeline import _classify_convergence
    from src.services.stages.shared.time_budget import (
        TimelineManifest,
        build_timeline_manifest,
    )

    cfg = dataclasses.replace(_band_cfg(target=25.0), wpm_band=(100.0, wpm_max))
    demands = collect_demands(
        lines, clip_seconds={}, wpm_band=cfg.wpm_band, wpm_natural=cfg.wpm_natural
    )
    allotments = solve_timeline(demands, target=cfg.target, tolerance_pct=cfg.tolerance_pct)
    manifest = TimelineManifest.model_validate(
        build_timeline_manifest(
            demands,
            allotments,
            target_seconds=cfg.target,
            loop_iterations=1,
            converged=False,
            pacing_score=4.0,
            convergence_outcome=outcome or _classify_convergence(lines, demands, cfg),
            overrun_seconds=max(0.0, sum(d.min for d in demands) - cfg.target),
        )
    )
    return manifest, cfg


def _remeasured_demands(lines: list[dict[str, object]], wpm_max: float) -> list[TimeDemand]:
    return collect_demands(lines, clip_seconds={}, wpm_band=(100.0, wpm_max), wpm_natural=150.0)


def _spawn_verdict(outcome: str, overrun: float) -> str:
    from src.services.runs.structural_relief_policy import decide_structural_relief

    return decide_structural_relief(
        convergence_outcome=outcome,
        structural_relief_requested=False,
        overrun_seconds=overrun,
        prior_overrun_seconds=None,
        pass_number=0,
    )


def _note_ctx() -> Any:
    from uuid import uuid4

    from src.services.runs.run_context import RunContext

    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={},
        prior_run_outputs={},
    )


_NOTE_HEAD = "STRUCTURAL RELIEF NEEDED"


@pytest.mark.asyncio
async def test_a_beat_whose_window_holds_fewer_words_than_the_ask_is_not_re_asked() -> None:
    """``freeze_pad_ceiling_seconds: 0``, a 10.3s clip at 136.06 wpm: the natural-rate ask
    is ceil(10.3 x 136.06 / 60) = 24 words but the window holds floor(23.36) = 23. A
    prompt of "AT LEAST 24 … AT MOST 23" cannot be honoured, so the beat is skipped: no WM
    call, original text kept — the same outcome as an unusable reply."""
    import dataclasses as _dc
    import math as _math

    wpm = 136.06
    cfg = _dc.replace(
        _cfg(target=10.3),
        wpm_natural=wpm,
        wpm_band=(wpm * 130 / 150, wpm * 170 / 150),
        freeze_pad_ceiling_seconds=0.0,
    )
    clips = {"clip-1": 10.3}
    lines = _underspoken_spine()
    demands = collect_demands(
        lines, clip_seconds=clips, wpm_band=cfg.wpm_band, wpm_natural=cfg.wpm_natural
    )
    allotments = solve_timeline(
        demands,
        target=cfg.target,
        tolerance_pct=cfg.tolerance_pct,
        wpm_natural=wpm,
        freeze_pad_ceiling_seconds=0.0,
    )
    assert any(a.underspoken for a in allotments), "precondition: the beat is flagged"
    assert _math.ceil(10.3 * wpm / 60.0) == 24
    assert _math.floor(10.3 * wpm / 60.0) == 23

    wm = _FakeWM([" ".join(f"w{i}" for i in range(24))])
    new_lines, _, _, notes = await _stage(wm)._expand_underspoken(
        lines, demands, allotments, cfg, clip_seconds=clips
    )
    assert wm.calls == [], "a self-contradicting ask must not be sent"
    assert new_lines[0]["text"] == lines[0]["text"]
    assert notes and "nothing usable" in notes[0]


def test_expand_ask_rate_is_the_narration_budget_rate() -> None:
    """The re-ask rate and the narration budget rate are ONE quantity, not two equal ones.

    The generator is ordered ``window * wpm * NARRATION_BUDGET_RATE / 60`` words per
    beat; the underspoken re-ask tops a short beat up to the same arithmetic at
    ``_EXPAND_ASK_RATE``. If the budget rate moves and the ask does not, the re-ask
    chases a count the generator was never told to write (and vice versa). Both are 1.0
    today, so an equality check cannot catch a restated literal — identity can: a
    ``_EXPAND_ASK_RATE = 1.0`` in reconcile_timeline is a different float object from
    the one defined in time_budget, and this goes red.
    """
    assert _EXPAND_ASK_RATE is NARRATION_BUDGET_RATE, (
        "_EXPAND_ASK_RATE must be imported from time_budget.NARRATION_BUDGET_RATE, "
        "never restated as a literal"
    )


@pytest.mark.asyncio
async def test_one_beats_wm_outage_keeps_the_rewrites_already_paid_for() -> None:
    """A WM call that raised on beat 2 used to escape the whole per-beat loop into the
    wrapper's catch-all, which returned the ORIGINAL spine — discarding beat 1's rewrite,
    a call already made and billed. Now the failure costs beat 2 alone, beat 1 keeps its
    lengthened text, and the note names the beat that was not re-asked.
    """

    class _SecondCallBoom:
        def __init__(self, first: str) -> None:
            self.first = first
            self.calls = 0

        async def message(self, **kwargs: Any) -> Any:
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(output={"text": self.first})
            raise RuntimeError("worker_management unreachable")

    cfg = _cfg(target=20.0)
    lines: list[dict[str, object]] = [
        {"id": "clip-1", "text": "One two three four five six", "seconds": 10.0},
        {"id": "clip-2", "text": "Seven eight nine ten eleven twelve", "seconds": 10.0},
    ]
    clips = {"clip-1": 10.0, "clip-2": 10.0}
    demands = collect_demands(
        lines, clip_seconds=clips, wpm_band=cfg.wpm_band, wpm_natural=cfg.wpm_natural
    )
    allotments = solve_timeline(demands, target=cfg.target, tolerance_pct=cfg.tolerance_pct)
    assert all(a.underspoken for a in allotments), "precondition: both beats start flagged"

    long_text = " ".join(f"w{i}" for i in range(26))
    stage = ReconcileTimelineStage()
    stage._wm = cast("WMClient", _SecondCallBoom(long_text))
    out_lines, _d, _a, notes = await stage._maybe_expand_underspoken(
        lines, demands, allotments, cfg, clip_seconds=clips
    )

    assert out_lines[0]["text"] == long_text, "beat 1's paid rewrite must survive"
    assert out_lines[1]["text"] == lines[1]["text"]
    assert notes and "clip-2" in notes[0] and "NOT re-asked" in notes[0]


# --- grounding: run 01a156a6 narrated "refactor the component tree" over a TodoMVC clip ---

_TODO_GHERKIN = (
    "Scenario: Add a todo and mark it as completed\n"
    '  When I type "Pay bills" into the input field\n'
    "  And I press the Enter key\n"
    '  And I click the checkbox next to the todo item "Pay bills"\n'
    '  Then the todo item "Pay bills" is marked as completed'
)


def _grounded_spine() -> list[dict[str, object]]:
    return [
        {
            "id": "clip-1",
            "text": "Next we add a todo and mark it done",
            "seconds": 10.0,
            "scenario_name": "Add a todo and mark it as completed",
            "gherkin": _TODO_GHERKIN,
        }
    ]


@pytest.mark.asyncio
async def test_an_ungrounded_expansion_is_refused_and_the_next_pass_is_used() -> None:
    """FAILS BEFORE: the first (invented) reply was applied — it was longer, so it won."""
    cfg = _cfg(target=10.0)
    lines = _grounded_spine()
    demands = collect_demands(
        lines, clip_seconds=_CLIPS, wpm_band=cfg.wpm_band, wpm_natural=cfg.wpm_natural
    )
    allotments = solve_timeline(demands, target=cfg.target, tolerance_pct=cfg.tolerance_pct)
    assert any(a.underspoken for a in allotments), "precondition: the beat starts flagged"

    invented = (
        "Next we refactor the entire component tree to eliminate redundant re-renders "
        "throughout the application structure for better debugging performance"
    )
    grounded = (
        'Next we type "Pay bills" into the input field and press Enter, then click '
        "the checkbox next to the todo item so it is marked completed."
    )
    assert 25 <= len(grounded.split()) <= 27, "inside the ask..ceiling window"
    wm = _FakeWM([invented, grounded])
    new_lines, _, _, _ = await _stage(wm)._expand_underspoken(
        lines, demands, allotments, cfg, clip_seconds=_CLIPS
    )

    assert len(wm.calls) == 2, "the refused reply spends a pass; the next one is asked"
    assert '"Pay bills"' in wm.calls[0]["message"], "the recording grounds the prompt"
    assert new_lines[0]["text"] == grounded


@pytest.mark.asyncio
async def test_the_expander_is_handed_the_beats_unnarrated_steps() -> None:
    cfg = _cfg(target=10.0)
    lines = _grounded_spine()
    lines[0]["text"] = 'Next we type "Pay bills" into the input field'
    demands = collect_demands(
        lines, clip_seconds=_CLIPS, wpm_band=cfg.wpm_band, wpm_natural=cfg.wpm_natural
    )
    allotments = solve_timeline(demands, target=cfg.target, tolerance_pct=cfg.tolerance_pct)
    assert any(a.underspoken for a in allotments), "precondition: the beat starts flagged"
    wm = _FakeWM([""])

    await _stage(wm)._expand_underspoken(lines, demands, allotments, cfg, clip_seconds=_CLIPS)

    message = wm.calls[0]["message"]
    assert '- click the checkbox next to the todo item "Pay bills"' in message
    assert '- type "Pay bills" into the input field' not in message, "already narrated"


@pytest.mark.asyncio
async def test_the_reask_asks_for_the_beats_share_of_the_order_when_larger() -> None:
    """gen's survivor re-budget (`narration_words_per_beat`) raises the ask, inside the window."""
    cfg = _cfg(target=10.0)
    lines = _underspoken_spine()
    demands = collect_demands(
        lines, clip_seconds=_CLIPS, wpm_band=cfg.wpm_band, wpm_natural=cfg.wpm_natural
    )
    allotments = solve_timeline(demands, target=cfg.target, tolerance_pct=cfg.tolerance_pct)
    share = _natural_ask(10.0) + 2
    wm = _FakeWM([" ".join(f"w{i}" for i in range(share))])

    await _stage(wm)._expand_underspoken(
        lines, demands, allotments, cfg, clip_seconds=_CLIPS, order_word_budgets={"clip-1": share}
    )

    assert f"AT LEAST {share} words" in wm.calls[0]["message"]
