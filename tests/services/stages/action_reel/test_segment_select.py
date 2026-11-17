"""action_reel segment_select — does the editor know how long the reel is?

The bug these tests exist for was silent and green: a 20 s reel was requested,
13.5 s of drone footage was probed correctly, and the model returned 2.4 s from
ONE of the two clips. Nothing failed — the stage never told the model the target
length, and "prefer a few strong segments over many weak ones" is exactly the
right instruction for an editor who does not know a reel has to be filled.

So the assertions here are on the MESSAGE THE MODEL RECEIVES, not on the
segments that come back: what the model does with the brief is the model's
business, whether the brief reaches it at all is this stage's. The second half
covers the other side of that contract — a run with no requested length must
still send byte-for-byte what it sent before — plus the validation floor, which
is what keeps a prompt change from widening what reaches the cut planner.
"""

from __future__ import annotations

import itertools
import json
import math
import re
from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import uuid7

import pytest
from w2v_common.clients import WMResponse

from src.services.runs.run_context import RunContext
from src.services.stages.action_reel.segment_select import (
    _COVERAGE_FACTOR,
    _MAX_SEGMENT_SECONDS,
    _MAX_TOKENS_PER_SEGMENT,
    _MIN_SEGMENT_SECONDS,
    _NEAR_DUPLICATE_GAP_SECONDS,
    _REASK_COVERAGE_MULTIPLE,
    _SCENE_CUTS_PER_SOURCE,
    _SEGMENTS_PER_SOURCE,
    _SELECT_ANSWER_TOKENS,
    _SELECT_BATCH_SOURCES,
    _SELECT_MAX_TOKENS,
    SegmentSelectInput,
    SegmentSelectStage,
)
from src.services.stages.action_reel.target_duration import target_seconds
from src.services.stages.base import StagePreconditionError
from src.services.stages.llm_budget import MIN_OUTPUT_TOKEN_BUDGET

_SOURCES: list[dict[str, Any]] = [
    {
        "id": "src-1",
        "duration_s": 6.006,
        "scene_boundaries": [2.0, 4.5],
        "motion_timeline": [{"t": 0.5, "motion": 0.8}, {"t": 3.0, "motion": 0.4}],
        "overall_score": 0.72,
    },
    {
        "id": "src-2",
        "duration_s": 7.5075,
        "scene_boundaries": [3.1],
        "motion_timeline": [{"t": 1.0, "motion": 0.9}],
        "overall_score": 0.65,
    },
]


def _segments() -> list[dict[str, Any]]:
    """A draw the validator accepts whole — so a test that fails, fails on the
    thing it is about rather than on the stage's empty-draw retry."""
    return [
        {"source_id": "src-1", "in_seconds": 1.0, "out_seconds": 4.0, "score": 0.8},
        {"source_id": "src-2", "in_seconds": 0.5, "out_seconds": 5.0, "score": 0.7},
    ]


def _wm_response(segments: list[dict[str, Any]]) -> WMResponse:
    return WMResponse(
        output={"segments": segments},
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )


def _sources_in(message: str) -> set[str]:
    """Which clips a given prompt actually showed the model.

    The stage batches the corpus (``_SELECT_BATCH_SOURCES``), so a fake that
    answers every call with the whole draw would hand the SAME segments back once
    per batch — turning every multi-clip test into a duplicate-detection test.
    Reading the ids out of the message is also the only way a test can assert
    that a call carried ONLY its batch."""
    return set(re.findall(r'"source_id": "([^"]+)"', message))


def _batched_wm(*draws: list[dict[str, Any]], repeat_last: bool) -> AsyncMock:
    """A WM that plays each DRAW across however many batch calls the stage makes.

    A "draw" is one full pass over the corpus. Each call is answered with only
    the segments naming a clip that call was given; a call naming a clip already
    served in the current draw starts the next draw (the batches partition the
    corpus, so a repeat can only mean the stage went round again — the re-ask).
    """
    state: dict[str, Any] = {"draw": -1, "seen": set[str]()}

    def _respond(**kwargs: Any) -> WMResponse:
        message = kwargs["message"]
        ids = _sources_in(message)
        assert ids, "a batch call must carry at least one source"
        # The stage labels a call only when it actually split the corpus; an
        # unlabelled call is the whole corpus in one batch.
        label = re.search(r"This is batch (\d+) of (\d+) of", message)
        index, total = (int(label[1]), int(label[2])) if label else (1, 1)
        if index == 1:
            state["draw"] += 1
            state["seen"] = set()
        state["seen"] |= ids
        draw_index = state["draw"]
        if draw_index >= len(draws):
            if not repeat_last:
                raise AssertionError(
                    f"WM called for draw {draw_index + 1}; only {len(draws)} supplied"
                )
            draw_index = len(draws) - 1
        draw = draws[draw_index]
        served = [seg for seg in draw if seg["source_id"] in ids]
        if index == total:
            # Segments naming a clip that is in NO batch — the fabricated
            # ``source_id`` the validator is supposed to drop. They belong to the
            # draw, not to a batch, so the last call of the pass carries them and
            # they are delivered exactly once.
            served += [seg for seg in draw if seg["source_id"] not in state["seen"]]
        return _wm_response(served)

    fake = AsyncMock()
    fake.message.side_effect = _respond
    return fake


def _fake_wm(segments: list[dict[str, Any]] | None = None) -> AsyncMock:
    return _batched_wm(_segments() if segments is None else segments, repeat_last=True)


def _fake_wm_sequence(*draws: list[dict[str, Any]]) -> AsyncMock:
    """A WM that answers each DRAW (not each call) with the next list. The
    short-draw re-ask is a second PAID pass, so every test about it has to be
    able to say what the model returned the first time and what the second."""
    return _batched_wm(*draws, repeat_last=False)


def _ctx(
    *,
    config: dict[str, Any] | None = None,
    sources: list[dict[str, Any]] | None = None,
) -> RunContext:
    return RunContext(
        run_id=uuid7(),
        deliverable_id=uuid7(),
        config_snapshot=config or {},
        prior_run_outputs={"shot_analysis": {"sources": _SOURCES if sources is None else sources}},
    )


def _sent_message(fake: AsyncMock) -> str:
    message = fake.message.await_args.kwargs["message"]
    assert isinstance(message, str)
    return message


# --- the brief reaches the model --------------------------------------------


async def test_the_target_length_and_the_available_footage_reach_the_model() -> None:
    """THE regression test. Both numbers matter: the target alone would read as
    an impossible quota against a 13.5 s corpus, and the total alone says nothing
    about how much of it to keep."""
    fake = _fake_wm()

    await SegmentSelectStage(client=fake).handle(
        SegmentSelectInput(), _ctx(config={"target_duration_seconds": 20})
    )

    message = _sent_message(fake)
    assert "20.0 seconds" in message
    # 6.006 + 7.5075, as stated to the model.
    assert "13.5 seconds of source footage" in message
    assert "2 clip(s)" in message


async def test_the_ask_is_capped_at_the_footage_that_actually_exists() -> None:
    """2x a 20 s target is 40 s of selection against 13.5 s of footage. Asking
    for a number the corpus cannot produce teaches the model that the brief is
    fiction, which is worse than not stating one."""
    fake = _fake_wm()

    await SegmentSelectStage(client=fake).handle(
        SegmentSelectInput(), _ctx(config={"target_duration_seconds": 20})
    )

    assert "Select at least 13.5 seconds" in _sent_message(fake)


async def test_a_comfortable_corpus_is_asked_for_twice_the_target() -> None:
    """The other side of the cap: with footage to spare, the ask is the coverage
    factor, not the target — a selection sized to the target can only come out
    short once the planner drops what does not fit the grid."""
    fake = _fake_wm()
    long_sources = [{**src, "duration_s": 60.0} for src in _SOURCES]

    await SegmentSelectStage(client=fake).handle(
        SegmentSelectInput(),
        _ctx(config={"target_duration_seconds": 20}, sources=long_sources),
    )

    assert "Select at least 40.0 seconds" in _sent_message(fake)


async def test_the_editor_is_told_to_draw_from_every_clip() -> None:
    """The live failure used one of two clips and reported success. The coverage
    instruction is part of the system prompt, so assert it is actually sent."""
    fake = _fake_wm()

    await SegmentSelectStage(client=fake).handle(
        SegmentSelectInput(), _ctx(config={"target_duration_seconds": 20})
    )

    prompt = fake.message.await_args.kwargs["system_prompt"]
    assert "EVERY source clip" in prompt
    # The floors the validator enforces must be the ones the model is given —
    # the prompt spells them out, so this is what keeps the two in step.
    assert f"at least {_MIN_SEGMENT_SECONDS} seconds" in prompt
    assert f"at most {_MAX_SEGMENT_SECONDS} seconds" in prompt
    # The over-selection instruction is the other half of the same fix and must
    # not be traded away for the variety instruction added beside it.
    assert "materially MORE total footage" in prompt


async def test_the_editor_is_told_that_two_of_the_same_shot_is_a_defect() -> None:
    """The second live failure: several near-identical static windows of one
    kiosk, every one individually valid. The structural guard below is the
    backstop; asking for variety is what makes the guard have nothing to do."""
    fake = _fake_wm()

    await SegmentSelectStage(client=fake).handle(
        SegmentSelectInput(), _ctx(config={"target_duration_seconds": 20})
    )

    prompt = fake.message.await_args.kwargs["system_prompt"]
    assert "VARIETY" in prompt
    assert "differ in subject, framing or motion character" in prompt
    assert "read on screen as the SAME shot" in prompt
    assert "a defect" in prompt
    # ...and the half of that paragraph that stops it becoming a budget of one
    # pick per clip, which is how the 10-clip run returned 10 segments.
    assert "never a budget of one pick per clip" in prompt
    assert "THE QUOTA WINS" in prompt


async def test_no_requested_length_sends_the_message_unchanged() -> None:
    """A run that never asked for a length must be byte-identical to today: no
    brief, no invented default, no second paragraph."""
    fake = _fake_wm()

    await SegmentSelectStage(client=fake).handle(SegmentSelectInput(), _ctx())

    message = _sent_message(fake)
    assert message.startswith("SOURCE CLIPS (one JSON object per clip):\n")
    assert "BRIEF" not in message
    # The brief is appended after a blank line; there is no other one.
    assert "\n\n" not in message
    stage = SegmentSelectStage(client=fake)
    assert message == stage._build_message(_SOURCES)


# --- the resolver -----------------------------------------------------------


@pytest.mark.parametrize(
    ("config", "expected"),
    [pytest.param({"target_duration_seconds": 20}, 20.0, id="config")],
)
def test_the_resolver_honours_config_then_deliverable_then_brief(
    config: dict[str, Any], expected: float
) -> None:
    assert target_seconds(_ctx(config=config)) == expected


@pytest.mark.parametrize("config", [{}])
def test_an_unset_or_zero_target_resolves_to_none(config: dict[str, Any]) -> None:
    """``None`` is "use every selected segment", which is a real editorial
    outcome — never a 0-second reel and never an invented default."""
    assert target_seconds(_ctx(config=config)) is None


# --- validation is unchanged ------------------------------------------------


async def test_out_of_range_windows_are_still_dropped_with_a_note() -> None:
    """The prompt now asks for MORE material; the floor that keeps a glitch out
    of the reel has to survive that. An unknown source, a sub-0.4s flash and an
    inverted window all die here, and the reviewer is told why."""
    fake = _fake_wm(
        [
            {"source_id": "ghost", "in_seconds": 0.0, "out_seconds": 3.0, "score": 0.9},
            {"source_id": "src-1", "in_seconds": 1.0, "out_seconds": 1.2, "score": 0.9},
            {"source_id": "src-1", "in_seconds": 4.0, "out_seconds": 2.0, "score": 0.9},
            {"source_id": "src-2", "in_seconds": 1.0, "out_seconds": 5.0, "score": 0.6},
        ]
    )

    out = await SegmentSelectStage(client=fake).handle(
        SegmentSelectInput(), _ctx(config={"target_duration_seconds": 20})
    )

    assert [s.source_id for s in out.segments] == ["src-2"]
    assert any("unknown source_id" in note for note in out.notes)
    assert sum("unusable window" in note for note in out.notes) == 2


async def test_a_window_past_the_end_of_its_source_is_clamped_not_invented() -> None:
    """Asking for twice the target is an invitation to over-reach past a clip's
    duration; the clamp is what keeps that from becoming a compositor 422."""
    fake = _fake_wm([{"source_id": "src-1", "in_seconds": 2.0, "out_seconds": 99.0, "score": 0.9}])

    out = await SegmentSelectStage(client=fake).handle(
        SegmentSelectInput(), _ctx(config={"target_duration_seconds": 20})
    )

    assert out.segments[0].out_seconds == pytest.approx(6.006)
    assert out.segments[0].seconds == pytest.approx(4.006)


# --- the variety guard ------------------------------------------------------
#
# The live 10-clip reel picked several near-identical static windows of one
# kiosk. Every one of them was individually valid, so nothing above catches it
# and nothing downstream compares two windows to each other. These tests are on
# the OUTPUT, not the message: unlike the brief, this is a promise the stage
# keeps itself whatever the model returns.


async def test_overlapping_windows_from_one_source_lose_the_weaker_one() -> None:
    """The literal live defect. The stronger window survives, the count is
    structured so a reviewer cannot skim past it, and the note names the drop."""
    fake = _fake_wm(
        [
            {"source_id": "src-1", "in_seconds": 1.0, "out_seconds": 4.0, "score": 0.9},
            {"source_id": "src-1", "in_seconds": 2.0, "out_seconds": 5.0, "score": 0.5},
            {"source_id": "src-2", "in_seconds": 0.5, "out_seconds": 5.0, "score": 0.7},
        ]
    )

    out = await SegmentSelectStage(client=fake).handle(SegmentSelectInput(), _ctx())

    assert [(s.source_id, s.in_seconds) for s in out.segments] == [("src-1", 1.0), ("src-2", 0.5)]
    assert out.near_duplicates_dropped == 1
    dropped_notes = [n for n in out.notes if "near-duplicate" in n]
    assert len(dropped_notes) == 1
    assert "src-1" in dropped_notes[0]
    assert "2.00s-5.00s" in dropped_notes[0]


async def test_near_adjacent_windows_from_one_source_read_as_one_shot() -> None:
    """Not overlapping — but half a second of one continuous camera move cannot
    have carried the subject anywhere new, so the second window is the same shot
    played again. The gap is a named constant; assert against IT, not against a
    number retyped here."""
    gap = _NEAR_DUPLICATE_GAP_SECONDS
    fake = _fake_wm(
        [
            {"source_id": "src-1", "in_seconds": 0.5, "out_seconds": 2.0, "score": 0.9},
            # Starts inside the gap that follows the first window's out-point.
            {"source_id": "src-1", "in_seconds": 2.0 + gap / 2, "out_seconds": 4.5, "score": 0.4},
            {"source_id": "src-2", "in_seconds": 0.5, "out_seconds": 5.0, "score": 0.7},
        ]
    )

    out = await SegmentSelectStage(client=fake).handle(SegmentSelectInput(), _ctx())

    assert [(s.source_id, s.in_seconds) for s in out.segments] == [("src-1", 0.5), ("src-2", 0.5)]
    assert out.near_duplicates_dropped == 1


async def test_windows_from_different_sources_at_the_same_time_are_both_kept() -> None:
    """Two clips at the same second of their OWN footage are two cameras. That
    is the variety this guard exists to protect, so a timestamp collision across
    sources must not trip it."""
    fake = _fake_wm(
        [
            {"source_id": "src-1", "in_seconds": 1.0, "out_seconds": 4.0, "score": 0.9},
            {"source_id": "src-2", "in_seconds": 1.0, "out_seconds": 4.0, "score": 0.8},
        ]
    )

    out = await SegmentSelectStage(client=fake).handle(SegmentSelectInput(), _ctx())

    assert [s.source_id for s in out.segments] == ["src-1", "src-2"]
    assert out.near_duplicates_dropped == 0
    assert not [n for n in out.notes if "near-duplicate" in n]


async def test_the_guard_never_starves_the_cut_planners_coverage() -> None:
    """Trading a repeated shot for a reel that runs SHORT is not a fix — the
    planner still has to be able to FILL the target. Against a 12 s target the
    floor is 12 s; the two unique windows are 9 s, so the higher-scoring of the
    two remaining duplicates comes back and the weakest still goes."""
    fake = _fake_wm(
        [
            {"source_id": "src-1", "in_seconds": 0.0, "out_seconds": 3.0, "score": 0.9},
            {"source_id": "src-1", "in_seconds": 0.2, "out_seconds": 3.2, "score": 0.8},
            {"source_id": "src-1", "in_seconds": 0.4, "out_seconds": 3.4, "score": 0.5},
            {"source_id": "src-2", "in_seconds": 0.0, "out_seconds": 6.0, "score": 0.7},
        ]
    )

    out = await SegmentSelectStage(client=fake).handle(
        SegmentSelectInput(), _ctx(config={"target_duration_seconds": 12})
    )

    # The 0.8 duplicate is re-admitted; the 0.5 one is not.
    assert sorted(s.in_seconds for s in out.segments) == [0.0, 0.0, 0.2]
    assert out.near_duplicates_dropped == 1
    assert out.near_duplicates_kept_for_coverage == 1
    kept_note = next(n for n in out.notes if n.startswith("KEPT"))
    assert "KEPT 1 near-duplicate segment(s)" in kept_note
    assert "12.0s the cut planner needs to fill the reel" in kept_note
    assert sum(s.seconds for s in out.segments) >= 12.0


# --- B1: the guard has to bite at its DESIGNED operating point ---------------


def _wide_corpus_sources() -> list[dict[str, Any]]:
    """Ten 12 s clips — a real action-reel corpus, not the two-clip fixture."""
    return [
        {
            "id": f"src-{n:02d}",
            "duration_s": 12.0,
            "scene_boundaries": [],
            "motion_timeline": [{"t": 1.0, "motion": 0.8}],
            "overall_score": 0.7,
        }
        for n in range(1, 11)
    ]


def _obedient_draw() -> list[dict[str, Any]]:
    """Exactly what the BRIEF asks for: ~2x a 45 s target (90 s over 20 windows),
    drawn from every clip — with a duplicate pair on each, 0.5 s apart."""
    draw: list[dict[str, Any]] = []
    for n in range(1, 11):
        source = f"src-{n:02d}"
        draw.append({"source_id": source, "in_seconds": 0.0, "out_seconds": 4.5, "score": 0.9})
        # 5.0 - 4.5 = 0.5 s after the first window's out-point: the same shot.
        draw.append({"source_id": source, "in_seconds": 5.0, "out_seconds": 9.5, "score": 0.6})
    return draw


async def test_the_guard_bites_when_the_model_obeys_the_brief() -> None:
    """THE regression for an INERT guard.

    The floor used to be ``_COVERAGE_FACTOR`` x target — the same number the
    brief asks the model for. So a model that OBEYED the brief produced a
    selection whose unique windows were always under the floor, every duplicate
    was re-admitted, and ``near_duplicates_dropped`` was 0 on exactly the runs
    the guard was written for. It only ever bit when the model over-delivered.

    45 s target, ten 12 s clips, 20 windows totalling 90 s, one duplicate pair
    per clip. Ten duplicates must actually leave.
    """
    fake = _fake_wm(_obedient_draw())

    out = await SegmentSelectStage(client=fake).handle(
        SegmentSelectInput(),
        _ctx(config={"target_duration_seconds": 45}, sources=_wide_corpus_sources()),
    )

    assert out.near_duplicates_dropped == 10
    assert out.near_duplicates_kept_for_coverage == 0
    assert [s.source_id for s in out.segments] == [f"src-{n:02d}" for n in range(1, 11)]
    assert all(s.in_seconds == 0.0 for s in out.segments)

    # Why the old floor was inert, spelled out rather than implied: the surviving
    # unique windows are 45.0 s, comfortably over a 1x floor and just as
    # comfortably UNDER the 2x one — which is what re-admitted all ten.
    unique_seconds = sum(s.seconds for s in out.segments)
    assert unique_seconds == pytest.approx(45.0)
    assert unique_seconds < 45.0 * _COVERAGE_FACTOR
    # ...and it still fills the reel it was de-duplicated for.
    assert unique_seconds >= 45.0


# --- B2: the counts must distinguish "none existed" from "kept anyway" -------


async def test_duplicates_kept_for_coverage_are_counted_separately() -> None:
    """``near_duplicates_dropped == 0`` is not evidence of a varied selection.

    The rescue re-admits duplicates, and it used to do so while still reporting
    a drop count of 0 — indistinguishable from a draw that had no duplicates at
    all, so a reviewer filtering on ``== 0`` got precisely the repeated-shot reel
    this stage exists to surface. The two counts are only honest as a pair.
    """
    # Every window on one source, all mutually near-duplicate, against a target
    # far larger than they add up to: the floor re-admits all of them.
    fake = _fake_wm(
        [
            {"source_id": "src-1", "in_seconds": 0.0, "out_seconds": 3.0, "score": 0.9},
            {"source_id": "src-1", "in_seconds": 0.2, "out_seconds": 3.2, "score": 0.8},
            {"source_id": "src-1", "in_seconds": 0.4, "out_seconds": 3.4, "score": 0.7},
        ]
    )

    out = await SegmentSelectStage(client=fake).handle(
        SegmentSelectInput(), _ctx(config={"target_duration_seconds": 13})
    )

    assert len(out.segments) == 3  # nothing left; the floor kept them all
    assert out.near_duplicates_dropped == 0
    # ...and THIS is what stops that 0 from reading as "the draw was varied".
    assert out.near_duplicates_kept_for_coverage == 2


# --- B3: the guard drops; it does not re-cut the edit ------------------------


async def test_the_guard_preserves_the_models_order() -> None:
    """The guard used to interleave survivors round-robin across sources.

    ``cut_plan`` derives ``section_change`` from "does the NEXT segment change
    source" and prefers a downbeat there, so a perfectly interleaved list makes
    that true on nearly every shot instead of occasionally — silently changing
    the beat placement of every action reel. Nobody asked for that. Which
    segments reach the planner is this stage's call; the ORDER is the edit.
    """
    fake = _fake_wm(
        [
            # Two from src-1 back to back, then two from src-2. A round-robin
            # walk emits src-1, src-2, src-1, src-2 — this asserts it does not.
            {"source_id": "src-1", "in_seconds": 0.0, "out_seconds": 2.0, "score": 0.4},
            {"source_id": "src-1", "in_seconds": 4.0, "out_seconds": 6.0, "score": 0.9},
            {"source_id": "src-2", "in_seconds": 0.0, "out_seconds": 2.0, "score": 0.3},
            {"source_id": "src-2", "in_seconds": 4.0, "out_seconds": 6.0, "score": 0.8},
        ]
    )

    out = await SegmentSelectStage(client=fake).handle(SegmentSelectInput(), _ctx())

    assert out.near_duplicates_dropped == 0
    assert [(s.source_id, s.in_seconds) for s in out.segments] == [
        ("src-1", 0.0),
        ("src-1", 4.0),
        ("src-2", 0.0),
        ("src-2", 4.0),
    ]


async def test_a_survivor_keeps_its_place_even_when_its_twin_scored_lower() -> None:
    """Dropping is by score; EMITTING is by the model's order. A weak-then-strong
    pair must not reorder the segments around it just because the strong one won
    the collision."""
    fake = _fake_wm(
        [
            {"source_id": "src-2", "in_seconds": 0.0, "out_seconds": 5.0, "score": 0.3},
            {"source_id": "src-1", "in_seconds": 0.0, "out_seconds": 3.0, "score": 0.2},
            {"source_id": "src-1", "in_seconds": 0.5, "out_seconds": 3.5, "score": 0.95},
        ]
    )

    out = await SegmentSelectStage(client=fake).handle(SegmentSelectInput(), _ctx())

    assert out.near_duplicates_dropped == 1
    # src-2 stays FIRST — it is where the model put it — and the surviving src-1
    # window stays where IT was, third in the draw and so second here.
    assert [(s.source_id, s.in_seconds) for s in out.segments] == [
        ("src-2", 0.0),
        ("src-1", 0.5),
    ]


# --- B4: determinism ---------------------------------------------------------


def _canonical(out_segments: list[Any]) -> list[tuple[str, float, float]]:
    return sorted((s.source_id, s.in_seconds, s.out_seconds) for s in out_segments)


async def test_the_kept_set_does_not_depend_on_the_order_the_model_returned() -> None:
    """720 permutations of one draw used to yield 6 distinct outputs, because the
    walk was round-robin over a ``dict`` and followed INSERTION order.

    Two invariants, and they are different things. The kept SET must be identical
    under every permutation — that is the selection decision, and it may not
    depend on draw order. The emitted ORDER must be exactly the input's own order
    filtered to that set — that is B3's promise, and it is a deterministic
    function of the input rather than of a dict's iteration.
    """
    draw = [
        {"source_id": "src-1", "in_seconds": 0.0, "out_seconds": 3.0, "score": 0.9},
        # Near-duplicate of the one above; the weaker of the pair.
        {"source_id": "src-1", "in_seconds": 0.2, "out_seconds": 3.2, "score": 0.5},
        {"source_id": "src-2", "in_seconds": 0.0, "out_seconds": 4.0, "score": 0.7},
        # 5.5 - 4.0 = 1.5 s clear of the window above: NOT a duplicate.
        {"source_id": "src-2", "in_seconds": 5.5, "out_seconds": 7.5, "score": 0.6},
    ]
    expected_kept = [("src-1", 0.0, 3.0), ("src-2", 0.0, 4.0), ("src-2", 5.5, 7.5)]

    for permutation in itertools.permutations(draw):
        ordering = list(permutation)
        fake = _fake_wm(ordering)

        out = await SegmentSelectStage(client=fake).handle(SegmentSelectInput(), _ctx())

        assert _canonical(out.segments) == expected_kept, ordering
        assert out.near_duplicates_dropped == 1, ordering
        # The emitted order is this permutation's order, filtered — never a
        # re-shuffle, and never dependent on which source was seen first.
        kept = {(s.source_id, s.in_seconds) for s in out.segments}
        assert [(s.source_id, s.in_seconds) for s in out.segments] == [
            (str(seg["source_id"]), float(seg["in_seconds"]))
            for seg in ordering
            if (str(seg["source_id"]), float(seg["in_seconds"])) in kept
        ], ordering


async def test_a_score_tie_is_broken_deterministically_not_by_draw_order() -> None:
    """With tied scores the KEPT SET itself used to change with the input order
    (``['a','c']`` on one permutation, ``['b','c']`` on another). The tie-break
    is explicit — score desc, then ``in_seconds`` asc — so the earlier window of
    a tied colliding pair wins under every permutation."""
    draw = [
        {"source_id": "src-1", "in_seconds": 0.0, "out_seconds": 3.0, "score": 0.7},
        # Same source, same score, collides with the one above.
        {"source_id": "src-1", "in_seconds": 0.5, "out_seconds": 3.5, "score": 0.7},
        {"source_id": "src-2", "in_seconds": 0.0, "out_seconds": 3.0, "score": 0.9},
    ]

    for permutation in itertools.permutations(draw):
        fake = _fake_wm(list(permutation))

        out = await SegmentSelectStage(client=fake).handle(SegmentSelectInput(), _ctx())

        # in_seconds ASC breaks the tie: 0.0 survives, 0.5 goes, always.
        assert _canonical(out.segments) == [
            ("src-1", 0.0, 3.0),
            ("src-2", 0.0, 3.0),
        ], list(permutation)
        assert out.near_duplicates_dropped == 1, list(permutation)


# --- C1: a SHORT draw is a failure, and something has to notice ---------------
#
# The other half of the 10-clip defect. The stage asked for 80.0 s of a 97.1 s
# corpus, got 18.8 s — one window per clip — and said nothing: `_attempt`
# re-prompted only on an EMPTY draw, so the deficit first appeared a stage later
# as `cut_plan.shortfall_seconds: 21.6`, on top of a plan already built from it.
# These tests are on the CALLS and the NOTES: how much the model returns is the
# model's business, noticing that it under-filled the brief is this stage's.


def _one_window_per_clip(seconds: float) -> list[dict[str, Any]]:
    """The live failure's shape: exactly one pick from each of the ten clips."""
    return [
        {"source_id": f"src-{n:02d}", "in_seconds": 0.0, "out_seconds": seconds, "score": 0.8}
        for n in range(1, 11)
    ]


def _two_windows_per_clip() -> list[dict[str, Any]]:
    """100 s over ten clips: 6 s at the head, 4 s at the tail, 2 s apart — clear
    of ``_NEAR_DUPLICATE_GAP_SECONDS``, so the variety guard has nothing to do
    and the totals asserted below are the model's draw, not the guard's."""
    draw: list[dict[str, Any]] = []
    for n in range(1, 11):
        source = f"src-{n:02d}"
        draw.append({"source_id": source, "in_seconds": 0.0, "out_seconds": 6.0, "score": 0.9})
        draw.append({"source_id": source, "in_seconds": 8.0, "out_seconds": 12.0, "score": 0.7})
    return draw


def _thin_corpus_sources() -> list[dict[str, Any]]:
    """Three 4 s clips — 12 s of footage against any reel worth making. There is
    no second answer to be had here, however the question is phrased."""
    return [
        {
            "id": f"src-{n}",
            "duration_s": 4.0,
            "scene_boundaries": [],
            "motion_timeline": [{"t": 1.0, "motion": 0.8}],
            "overall_score": 0.7,
        }
        for n in range(1, 4)
    ]


async def test_a_corpus_that_cannot_fill_the_ask_is_never_re_asked() -> None:
    """The guard that stops a pointless second PAID call.

    Three 4 s clips: the brief is already capped at all 12 s that exist, and the
    model kept 2 s of it. That is short — but there is under one shot's worth of
    untouched footage left, so a second call can only buy the same answer at
    twice the price. It gets the note instead.
    """
    fake = _fake_wm_sequence(
        [{"source_id": "src-1", "in_seconds": 0.0, "out_seconds": 2.0, "score": 0.9}]
    )

    out = await SegmentSelectStage(client=fake).handle(
        SegmentSelectInput(),
        _ctx(config={"target_duration_seconds": 40}, sources=_thin_corpus_sources()),
    )

    # Three clips, two per call: two calls, ONE draw — no re-ask.
    assert fake.message.await_count == 2
    short_note = next(n for n in out.notes if n.startswith("SHORT SELECTION"))
    assert "2.0s" in short_note
    assert "12.0s" in short_note


async def test_a_draw_that_already_meets_the_brief_is_not_re_asked() -> None:
    """The re-ask is for a MATERIAL shortfall, not for every draw under the ask.
    100 s against a 90 s brief is a full selection; a second call there would be
    spending the run's money on slack the reel does not need."""
    fake = _fake_wm_sequence(_two_windows_per_clip())

    out = await SegmentSelectStage(client=fake).handle(
        SegmentSelectInput(),
        _ctx(config={"target_duration_seconds": 45}, sources=_wide_corpus_sources()),
    )

    # Five calls: ONE draw over ten clips, not a re-ask.
    assert fake.message.await_count == 5
    assert not [n for n in out.notes if n.startswith("SHORT SELECTION")]
    assert not [n for n in out.notes if n.startswith("re-asked")]


async def test_a_re_ask_that_comes_back_worse_keeps_the_first_draw() -> None:
    """A model told to add material can lose some instead. Letting the second
    answer overwrite the first would make this guard a regression: the run would
    end up with LESS footage than the version without it."""
    first = _one_window_per_clip(3.0)
    fake = _fake_wm_sequence(
        first,
        [{"source_id": "src-01", "in_seconds": 0.0, "out_seconds": 6.0, "score": 0.9}],
    )

    out = await SegmentSelectStage(client=fake).handle(
        SegmentSelectInput(),
        _ctx(config={"target_duration_seconds": 45}, sources=_wide_corpus_sources()),
    )

    assert fake.message.await_count == 10
    assert [(s.source_id, s.in_seconds, s.out_seconds) for s in out.segments] == [
        (str(seg["source_id"]), float(seg["in_seconds"]), float(seg["out_seconds"]))
        for seg in first
    ]
    assert sum(s.seconds for s in out.segments) == pytest.approx(30.0)
    assert any("first draw was kept" in n for n in out.notes)


async def test_a_still_short_selection_says_so_with_the_numbers() -> None:
    """The mechanism that would have made the live defect visible HERE instead of
    in `cut_plan`. The re-ask improved the draw and it is still under the brief,
    so the stage says what it kept, what was asked for, and what exists."""
    fake = _fake_wm_sequence(_one_window_per_clip(3.0), _one_window_per_clip(4.0))

    out = await SegmentSelectStage(client=fake).handle(
        SegmentSelectInput(),
        _ctx(config={"target_duration_seconds": 45}, sources=_wide_corpus_sources()),
    )

    # Ten clips at two per call, twice over: the re-ask is a second full pass.
    assert fake.message.await_count == 10
    short_note = next(n for n in out.notes if n.startswith("SHORT SELECTION"))
    # ⚠️ UPDATED: this assertion USED TO REQUIRE "90.0s" — the 2x the brief asks
    # for — and was therefore pinning the defect, not the contract. The note is
    # read by a HUMAN at the review pause, and quoting the ask made it call a draw
    # that comfortably fills the reel "SHORT". It now quotes what the REEL NEEDS.
    # 40 s kept against a 45 s reel out of 120 s of footage.
    assert "40.0s" in short_note
    assert "45.0s" in short_note
    assert "90.0s" not in short_note
    assert "120.0s" in short_note
    assert any("second draw was kept" in n for n in out.notes)


def test_the_brief_is_byte_identical_without_a_shortfall_hint() -> None:
    """The re-ask parameter must be invisible on the path every first call takes.
    The brief text and ``_COVERAGE_FACTOR`` are agreed BY HAND — the tests above
    pin that agreement, and they only mean anything if the default message is
    still the message."""
    stage = SegmentSelectStage(client=_fake_wm())

    assert stage._build_message(_SOURCES, 20.0, shortfall_hint=None) == stage._build_message(
        _SOURCES, 20.0
    )
    assert stage._build_message(_SOURCES, shortfall_hint=None) == stage._build_message(_SOURCES)
    # ...and the hint is APPENDED, never a rewrite of the brief it follows.
    hinted = stage._build_message(_SOURCES, 20.0, shortfall_hint="CORRECTION — too short.")
    assert hinted == stage._build_message(_SOURCES, 20.0) + "\n\nCORRECTION — too short."


# --- D: the prompt and the token budget both have to be BOUNDED --------------
#
# The live 4-clip run (01a016c1, 18.0 / 8.5 / 9.0 / 3.0 s of drone footage) died
# here with a PermanentUpstreamError: "Model token limit (6000) exceeded before
# any response was generated".
#
# ⚠️ THIS BLOCK USED TO EXPLAIN THAT WRONG. It said WM "RESERVED 6000 output
# tokens against a small self-hosted context window", and the test below asserted
# ``_SELECT_MAX_TOKENS < 6000`` — a gate that ENFORCED the harmful direction. The
# model is served at ``--max-model-len=131072``; there is no small window. The
# 422 comes from ``pydantic_ai/_agent_graph.py:1100-1113`` AFTER generation, when
# ``finish_reason == 'length'`` and every part is a ``ThinkingPart``: the
# reasoning model spent the entire budget thinking and never emitted the JSON.
# Measured live 2026-08-19 on one prompt with only max_tokens varying:
# 500 -> 422, **1760 -> 422**, 10000 -> fine. So the budget must be a FLOOR.
# Two bounds, two failure modes: the output budget and the prompt (batching).


def _corpus(count: int, *, scene_cuts: list[float] | None = None) -> list[dict[str, Any]]:
    return [
        {
            "id": f"src-{n:02d}",
            "duration_s": 12.0,
            "scene_boundaries": list(scene_cuts or []),
            "motion_timeline": [{"t": 1.0, "motion": 0.8}],
            "overall_score": 0.7,
        }
        for n in range(1, count + 1)
    ]


async def test_every_worker_call_carries_an_explicit_token_budget() -> None:
    """THE regression for the live 422.

    Omitting ``max_tokens`` is not "use a sensible default": it leaves WM's 6000,
    which is itself an OBSERVED production failure on six workers. And the budget
    must leave the model room to THINK — the 422 text ("Model token limit (N)
    exceeded before any response was generated") is raised post-hoc by pydantic-ai
    after the model burned N on reasoning without emitting content, NOT as a
    reservation against a context window. So a budget sized to the OUTPUT alone
    reproduces the outage with a smaller N, which is exactly what E2E run
    32224735897 measured at N=1760. The assertion is therefore that the kwarg is
    SENT on every call, with the stage's own budget, and that the budget clears
    the floor rather than sitting under it.
    """
    fake = _fake_wm(_segments())

    await SegmentSelectStage(client=fake).handle(
        SegmentSelectInput(), _ctx(config={"target_duration_seconds": 20})
    )

    assert fake.message.await_args_list
    for call in fake.message.await_args_list:
        assert call.kwargs["max_tokens"] == _SELECT_MAX_TOKENS
    # ...and the ANSWER estimate is DERIVED from the batch size, so raising one
    # raises the other. A hand-typed number here is how the two drift back apart.
    assert _SELECT_ANSWER_TOKENS == _SELECT_BATCH_SOURCES * _SEGMENTS_PER_SOURCE * (
        _MAX_TOKENS_PER_SEGMENT
    )
    # THE inverted assertion. The budget must clear the floor — reasoning is
    # charged against it, so answer-sized is broken. 1760 (PR #851's value) is
    # measured to 422 in production; anything below MIN_OUTPUT_TOKEN_BUDGET is
    # strictly worse than passing nothing at all.
    assert _SELECT_MAX_TOKENS >= MIN_OUTPUT_TOKEN_BUDGET
    assert _SELECT_MAX_TOKENS > _SELECT_ANSWER_TOKENS
    # Both historically-failing values, named. 1760 was #851's constant and 422'd
    # in prod on 2026-08-19; 6000 is WM's own default and 422'd on six workers.
    assert _SELECT_MAX_TOKENS not in (1760, 6000)
    # ⚠️ NO hardcoded upper bound here. #854 asserted `<= 8500` from a token rate
    # measured on an idle box running a model that has since been retired, and a
    # router deadline the cluster overrides. The ceiling is DERIVED from the
    # configured wall clock in stages/llm_budget.py and is asserted there, against
    # its inputs — so a stale number cannot be frozen into this file again.


async def test_the_corpus_is_split_into_batches_and_each_call_sees_only_its_own() -> None:
    """The other half. A token budget shrinks the RESERVATION; the prompt still
    grows with every clip added, and the whole corpus went into one message.

    Five clips at two per call is three calls, the batches partition the corpus,
    and no call may show the model a clip belonging to another batch — a call
    that saw everything would defeat the split while still passing a count.
    """
    sources = _corpus(5)
    fake = _fake_wm(
        [
            {"source_id": src["id"], "in_seconds": 0.0, "out_seconds": 4.0, "score": 0.8}
            for src in sources
        ]
    )

    await SegmentSelectStage(client=fake).handle(SegmentSelectInput(), _ctx(sources=sources))

    assert fake.message.await_count == math.ceil(len(sources) / _SELECT_BATCH_SOURCES) == 3
    batches = [_sources_in(call.kwargs["message"]) for call in fake.message.await_args_list]
    assert batches == [{"src-01", "src-02"}, {"src-03", "src-04"}, {"src-05"}]
    assert set().union(*batches) == {src["id"] for src in sources}


async def test_the_merged_selection_is_not_grouped_by_batch() -> None:
    """Concatenating the batch draws would group the reel BY SOURCE.

    ``cut_plan`` reads a ``source_id`` change between adjacent segments as a
    section change, so a list that runs batch 1's picks then batch 2's plays each
    clip's material in one clump and loses most of the section changes — a real
    editorial regression that no count or duration assertion would catch.

    ⚠️ UPDATED literal. This test used to assert a global score-DESC order over
    the RAW scores, i.e. it pinned the defect that scores from independent model
    calls are comparable. They are not (see ``_merge_batch_draws`` and
    ``test_a_batch_that_self_scores_low_is_not_pushed_out_of_the_reel`` below).
    The sort is still a score-DESC sort — over scores normalised PER BATCH — so
    the change here is only in the tail: batch 1's 0.4 and batch 2's 0.5 are each
    their own batch's worst and now tie at 0.0, where the raw sort ranked 0.5
    above 0.4 across a boundary the two scores never shared. Still neither the
    concatenation order nor grouped by batch, which is what this test is about.
    """
    fake = _fake_wm(
        [
            {"source_id": "src-01", "in_seconds": 0.0, "out_seconds": 4.0, "score": 0.4},
            {"source_id": "src-02", "in_seconds": 0.0, "out_seconds": 4.0, "score": 0.9},
            {"source_id": "src-03", "in_seconds": 0.0, "out_seconds": 4.0, "score": 0.5},
            {"source_id": "src-04", "in_seconds": 0.0, "out_seconds": 4.0, "score": 0.8},
        ]
    )

    out = await SegmentSelectStage(client=fake).handle(
        SegmentSelectInput(), _ctx(sources=_corpus(4))
    )

    order = [s.source_id for s in out.segments]
    # Batch 1 is (src-01: 0.4, src-02: 0.9) and batch 2 is (src-03: 0.5,
    # src-04: 0.8). Normalised per batch: each batch's best goes to 1.0 and each
    # batch's worst to 0.0, and the stable sort breaks the two ties by batch order.
    assert order == ["src-02", "src-04", "src-01", "src-03"]
    # ...which is emphatically NOT the batch-concatenated order, the thing a
    # naive merge produces and the thing this test exists to fail on.
    assert order != ["src-01", "src-02", "src-03", "src-04"]
    # Batch 1's two clips are not adjacent to each other: the selection is
    # interleaved across batches rather than clumped by the batch it came from.
    assert abs(order.index("src-01") - order.index("src-02")) > 1


def test_a_long_cut_list_is_thinned_before_it_reaches_the_prompt() -> None:
    """``scene_cuts`` was the one unbounded input left in the message.

    The motion curve has been downsampled since this stage shipped; the cut list
    beside it went in whole, one entry per boundary the analyzer found, which on
    a long busy clip is hundreds of numbers per source — the prompt-side half of
    the failure ``max_tokens`` does not touch.
    """
    cuts = [round(n * 0.1, 2) for n in range(1, 401)]
    stage = SegmentSelectStage(client=_fake_wm())

    message = stage._build_message(_corpus(1, scene_cuts=cuts))

    sent = json.loads(message.split("\n", 1)[1])["scene_cuts"]
    assert len(sent) == _SCENE_CUTS_PER_SOURCE
    # Evenly strided, not head-truncated: head truncation would hide every cut
    # after the first seconds, which on a 40 s cut list is nearly all of them.
    assert sent[0] == cuts[0]
    assert sent[-1] > cuts[len(cuts) // 2]
    assert sent != cuts[:_SCENE_CUTS_PER_SOURCE]


# --- the corpus an analyzer failure leaves behind ---------------------------


_MEASURED_AND_UNMEASURED: list[dict[str, Any]] = [
    _SOURCES[0],
    # What `shot_analysis` emits for a source the analyzer timed out on: a real
    # duration from the upload-time probe, and no measurements at all.
    {
        "id": "src-2",
        "duration_s": 7.5075,
        "scene_boundaries": [],
        "motion_timeline": [],
        "overall_score": 0.0,
        "error": "analysis job did not finish within 540.0s",
    },
]


async def test_an_unmeasured_source_still_reaches_the_editor() -> None:
    """THE regression. An analyzer failure is MISSING MEASUREMENT, not a missing
    clip: the errored source was filtered straight out of the corpus, so ten
    uploaded clips with eight analyzer timeouts produced a reel cut from two —
    five green gates, `unfetchable_segments` and `unplaced_segments` both 0, and
    nothing anywhere saying why the reel was thin.

    It must be OFFERED, and offered honestly: marked unmeasured rather than
    carrying a quality of 0.0, which is what a measured, genuinely dead clip
    looks like.
    """
    fake = _fake_wm()

    out = await SegmentSelectStage(client=fake).handle(
        SegmentSelectInput(),
        _ctx(config={"target_duration_seconds": 20}, sources=_MEASURED_AND_UNMEASURED),
    )

    message = _sent_message(fake)
    # The clip the analyzer lost is in the prompt...
    assert _sources_in(message) == {"src-1", "src-2"}
    # ...as UNMEASURED, with no fabricated score beside it. A "quality": 0.0 for
    # something nobody measured tells the model the opposite of the truth.
    unmeasured = json.loads(message.split("\n")[2])
    assert unmeasured["source_id"] == "src-2"
    assert unmeasured["unmeasured"] is True
    assert "quality" not in unmeasured
    assert "motion" not in unmeasured
    assert 'marked "unmeasured": true' in message
    assert "not a score of zero" in message
    # ...its footage counts toward what the brief may ask for (6.0 + 7.5)...
    assert "13.5 seconds of source footage" in message
    assert "2 clip(s)" in message
    # ...the editor could and did cut from it...
    assert "src-2" in {seg.source_id for seg in out.segments}
    # ...and the reviewer is told, at the gate, how much of the shoot was never
    # measured — the thing that made the thin reel unreadable.
    assert any("1 of 2 sources could not be measured" in note for note in out.notes)


async def test_an_unmeasured_source_with_no_duration_is_dropped_and_counted() -> None:
    """The one source that genuinely cannot be offered: no measurements AND no
    known length, so every window named against it clamps to [0, 0] and dies in
    the validator. Dropping it is right; dropping it silently is the bug in a
    smaller costume."""
    sources = [
        _SOURCES[0],
        {"id": "src-2", "duration_s": 0.0, "error": "no playable url"},
    ]
    fake = _fake_wm([{"source_id": "src-1", "in_seconds": 1.0, "out_seconds": 4.0, "score": 0.8}])

    out = await SegmentSelectStage(client=fake).handle(SegmentSelectInput(), _ctx(sources=sources))

    assert _sources_in(_sent_message(fake)) == {"src-1"}
    assert any("no known duration" in note for note in out.notes)


async def test_a_corpus_with_nothing_measured_is_still_a_precondition_failure() -> None:
    """The invariant `shot_analysis` enforces at its own exit, held again here
    because this stage can run against a REPLAYED prior output. Duration-only is
    honest for SOME of the corpus and a coin toss for ALL of it."""
    sources = [
        {"id": "src-1", "duration_s": 6.0, "error": "analyzer unreachable"},
        {"id": "src-2", "duration_s": 7.5, "error": "analyzer unreachable"},
    ]

    with pytest.raises(StagePreconditionError, match="every source is unmeasured"):
        await SegmentSelectStage(client=_fake_wm()).handle(
            SegmentSelectInput(), _ctx(sources=sources)
        )


async def test_a_fully_measured_corpus_sends_the_message_unchanged() -> None:
    """The other side of the contract: nothing about the unmeasured path may show
    up in the prompt of a run where every source was measured."""
    fake = _fake_wm()

    await SegmentSelectStage(client=fake).handle(SegmentSelectInput(), _ctx())

    message = _sent_message(fake)
    assert "unmeasured" not in message
    assert message == SegmentSelectStage(client=fake)._build_message(_SOURCES)


def test_every_segment_must_carry_its_rationale():
    """`rationale` is required, and that is enforceable where `minItems` is not.

    Required PROPERTIES are expressible in the CFG vLLM builds from this schema, so
    guided decoding actually enforces them. `rationale` is what the review gate shows
    the operator, and an optional field is one the model can drop to save tokens.

    The sibling constraint — `minItems` on `segments`, to forbid the measured
    `{"segments": []}` draw — is deliberately NOT here. worker-management strips array
    bounds before building the grammar while still validating against the unstripped
    schema, so it would be invisible to generation and mandatory at validation. See
    `tests/test_output_schema_bounds.py`, which bans the whole family, and the note in
    `segment_select._OUTPUT_SCHEMA`. The empty draw is caught by the PARSER instead —
    see `test_segment_select_empty_draw_wire.py`.
    """
    from src.services.stages.action_reel.segment_select import _OUTPUT_SCHEMA

    # `_OUTPUT_SCHEMA` is typed `dict[str, object]`, so every level has to be re-typed
    # on the way down. `cast` rather than `isinstance`: narrowing gives
    # `dict[Unknown, Unknown]`, which `ty` then refuses to subscript.
    def _obj(node: object, key: str) -> dict[str, object]:
        assert isinstance(node, dict), f"{key}: expected a mapping"
        return cast(dict[str, object], node)

    properties = _obj(_OUTPUT_SCHEMA["properties"], "properties")
    segments = _obj(properties["segments"], "segments")
    items = _obj(segments["items"], "items")

    required = items["required"]
    assert isinstance(required, list)
    assert "rationale" in required

    assert "minItems" not in segments, (
        "minItems is stripped by worker-management before grammar construction but "
        "still validated against — see tests/test_output_schema_bounds.py"
    )


async def test_the_model_is_told_not_to_think_before_it_answers() -> None:
    """The stage MUST send ``enable_reasoning=False``, and this is a fence, not a
    preference.

    Measured through worker-management on the production path (Job
    ``wmprobe-segsel``, 2026-08-20) with this stage's real prompt and schema at
    ``max_tokens=8000``: with reasoning ON the call ran to the stage's own 950 s
    deadline and returned nothing - twice, under BOTH candidate output schemas.
    With reasoning OFF the same call answered in 34-67 s carrying real segments,
    every time. So this keyword is the difference between a stage that completes
    and one that cannot, and dropping it degrades silently: every unit test here
    passes without it, because a stubbed client answers instantly either way.

    ``w2v_common`` >= 0.61.0 is a hard requirement — worker-management's
    ``MessageRequest`` is ``extra="forbid"``, so an older client cannot send the
    field and a WM without it would 422 the call outright.
    """
    fake = _fake_wm()

    await SegmentSelectStage(client=fake).handle(
        SegmentSelectInput(), _ctx(config={"target_duration_seconds": 20})
    )

    assert fake.message.await_args.kwargs["enable_reasoning"] is False


# --- E: the draw is judged against what the REEL NEEDS, not against the ask ---
#
# The re-ask and its reviewer-facing note used to be measured against
# ``requested_floor`` = min(target x _COVERAGE_FACTOR, available) — the ~2x the
# BRIEF asks the model for. On a real corpus that is nearly all the footage in the
# run: 10 clips totalling 97.1 s against a 45 s reel puts the ask at
# min(90.0, 97.1) = 90.0 s, i.e. 92.7% of everything uploaded. The old re-ask bar
# (0.75 x the ask = 67.5 s) therefore sat ABOVE any healthy editorial draw, so a
# 50 s selection — which fills the 45 s reel with slack to spare — bought a whole
# second pass (5 more worker-management calls against a slow, flaky vLLM) and told
# the human at the review pause it was SHORT. Asking for headroom is fine; judging
# the answer against the ask is the defect.


def _real_corpus_sources() -> list[dict[str, Any]]:
    """The measured shape: ten clips, 97.1 s of footage in total."""
    return [
        {
            "id": f"src-{n:02d}",
            "duration_s": 9.71,
            "scene_boundaries": [],
            "motion_timeline": [{"t": 1.0, "motion": 0.8}],
            "overall_score": 0.7,
        }
        for n in range(1, 11)
    ]


async def test_a_50s_draw_against_a_45s_reel_is_not_reported_as_short() -> None:
    """THE regression for the false shortfall, at the measured numbers.

    97.1 s of footage, a 45 s reel, a 50 s draw — one 5 s window from each of the
    ten clips. That covers the reel with better than 10% to spare, and the human
    at the review pause was told:

        SHORT SELECTION: 50.0s of segments against the 90.0s the brief asked for

    because the note quoted ``requested_floor`` = min(45 x 2, 97.1) = 90.0, i.e.
    92.7% of every second uploaded. A note that cries wolf on a healthy edit is
    worse than no note, because the reviewer learns to skip the one note that has
    to be believed.

    ⚠️ This draw DOES still buy the corrective re-ask: 50.0 < 45.0 x
    ``_REASK_COVERAGE_MULTIPLE`` = 56.25. That is the chosen bar, not an oversight
    — see the test below for where the second paid pass stops.
    """
    fake = _fake_wm_sequence(_one_window_per_clip(5.0), _one_window_per_clip(5.0))

    out = await SegmentSelectStage(client=fake).handle(
        SegmentSelectInput(),
        _ctx(config={"target_duration_seconds": 45}, sources=_real_corpus_sources()),
    )

    assert sum(s.seconds for s in out.segments) == pytest.approx(50.0)
    assert not [n for n in out.notes if n.startswith("SHORT SELECTION")]
    # Spelled out rather than implied: the number this draw used to be measured
    # against really was nearly the whole corpus.
    assert min(45.0 * _COVERAGE_FACTOR, 97.1) == pytest.approx(90.0)


async def test_a_healthy_draw_that_fills_the_reel_is_neither_re_asked_nor_called_short() -> None:
    """The cost half: a draw with real headroom must not buy a second paid pass.

    Same corpus, same 45 s reel, a 60 s draw. Against what the reel NEEDS that is
    a third more than required, so there is nothing to buy. Against the old
    yardstick it was still "materially short" — 60.0 < 0.75 x 90.0 = 67.5 — so the
    stage went back for a whole extra pass, i.e. ``_SELECT_BATCH_CONCURRENCY`` = 5
    more worker-management calls against a slow, flaky vLLM, on a corpus that had
    already answered the brief.

    ``_fake_wm_sequence`` supplies exactly ONE draw, so a re-ask does not merely
    fail an await-count assertion: the fake raises.
    """
    fake = _fake_wm_sequence(_one_window_per_clip(6.0))

    out = await SegmentSelectStage(client=fake).handle(
        SegmentSelectInput(),
        _ctx(config={"target_duration_seconds": 45}, sources=_real_corpus_sources()),
    )

    # Ten clips, two per call: ONE pass of five calls. Not ten.
    assert fake.message.await_count == 5
    assert sum(s.seconds for s in out.segments) == pytest.approx(60.0)
    assert not [n for n in out.notes if n.startswith("re-asked")]
    assert not [n for n in out.notes if n.startswith("SHORT SELECTION")]
    # The draw clears what the reel needs plus dedupe headroom, and is under what
    # the old bar demanded — which is exactly the band that used to pay twice.
    assert 45.0 * _REASK_COVERAGE_MULTIPLE <= 60.0
    assert 60.0 < 0.75 * 90.0


async def test_a_genuinely_short_draw_is_re_asked_and_the_note_names_the_reel() -> None:
    """The other side: the bar must still catch a draw that cannot fill the reel.

    20 s against a 45 s reel with 97.1 s of footage untouched is the failure the
    re-ask exists for, and both draws come back short — so the note fires, and it
    names 45, the seconds the REEL needs. Never 90, the 2x the brief asked for:
    that note is what a human reads at the pause (``default_pause_after=True``),
    and a number a reviewer cannot act on is worse than no note at all.
    """
    fake = _fake_wm_sequence(_one_window_per_clip(2.0), _one_window_per_clip(2.0))

    out = await SegmentSelectStage(client=fake).handle(
        SegmentSelectInput(),
        _ctx(config={"target_duration_seconds": 45}, sources=_real_corpus_sources()),
    )

    # Two full passes over ten clips at two per call.
    assert fake.message.await_count == 10
    short_note = next(n for n in out.notes if n.startswith("SHORT SELECTION"))
    assert "20.0s" in short_note
    assert "45.0s" in short_note
    assert "90.0s" not in short_note
    assert "97.1s" in short_note


async def test_the_live_2026_08_20_run_re_asks_and_then_says_nothing() -> None:
    """The measured production run, both halves, as one regression.

    10 clips / 97.1 s / a 45 s reel. The live notes on the unfixed code read:

        re-asked the editor once: the first draw returned 45.4s against the 90.0s
        the brief asked for; the second returned 65.4s and the second draw was kept
        SHORT SELECTION: 65.4s of segments against the 90.0s the brief asked for,
        out of 97.1s of available footage — cut_plan will report the remainder as
        shortfall; ...

    ``cut_plan`` then reported ``shortfall_seconds: 0.0`` on a 47.08 s plan for a
    45 s target. The note shown to a HUMAN at a paused gate predicted a shortfall
    that did not happen — it was simply false, and it was false on a run that
    worked.

    Both halves are pinned here because they must move independently: the re-ask
    on 45.4 s is CORRECT and must survive (45.4 < 45.0 x 1.25 = 56.25, and the
    variety guard dropped 12 near-duplicates out of that draw, which is the margin
    the 25% exists for), while the note on 65.4 s must be gone (65.4 >= 45.0).
    A fix that silenced the note by also killing the re-ask would pass half of
    this test and be wrong.
    """
    fake = _fake_wm_sequence(_one_window_per_clip(4.54), _one_window_per_clip(6.54))

    out = await SegmentSelectStage(client=fake).handle(
        SegmentSelectInput(),
        _ctx(config={"target_duration_seconds": 45}, sources=_real_corpus_sources()),
    )

    # The re-ask fired: two full passes over ten clips at two per call.
    assert fake.message.await_count == 10
    assert sum(s.seconds for s in out.segments) == pytest.approx(65.4)
    reask_note = next(n for n in out.notes if n.startswith("re-asked"))
    assert "45.4s" in reask_note
    assert "65.4s" in reask_note
    # The re-ask note still quotes the BRIEF's own number, because that is the
    # instruction the model was actually given — only the DECISION and the
    # reviewer-facing verdict moved to what the reel needs.
    assert "90.0s" in reask_note
    # ...and the false prediction is gone.
    assert not [n for n in out.notes if n.startswith("SHORT SELECTION")]


# --- F: scores from independent model calls are NOT comparable ---------------


# --- the band where a stride is 1 and "even striding" was head truncation ----
