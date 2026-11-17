"""The capture path must MEASURE its own seconds-per-scenario, and name short clips.

The scenario count used to be sized by ``gen._SECONDS_PER_SCENARIO`` (4.0), a figure
extrapolated from two hand-observed samples on one E2E run; it is gone, and the count
is now sized from a MEASURED yield. Nothing in the capture path ever compared a
harvested clip against a real budget either: a capture yielding 0.8s/clip
produced a film 5x underfull and the first thing that noticed was
``reconcile_timeline``'s GROSSLY-UNDERFULL raise three stages later — which only fires
below ``demo_timeline_underfull_fail_ratio`` of the target, so a film 20-49% short
shipped silently.

Every assertion here is on an OBSERVED VALUE (a parsed number out of the emitted log
line, or the text of a note that carries the measured seconds), never on a mock's call
shape.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from src.services.stages.demo_video.record import DemoVideoRecordStage

from .test_stages import _ctx, _passing_analyzer, _qa_settings

_TEST_TIMEOUT_S = 30.0


def _ftr_client(clips: list[dict[str, Any]]) -> AsyncMock:
    client = AsyncMock()
    client.start_execution.return_value = "ftr-yield"
    client.wait_for_execution.return_value = {}
    client.get_job_result.return_value = {
        "scenario_clips": clips,
        "summary": {"passed": len(clips), "failed": 0},
    }
    client.get_clip_video.return_value = b"mp4"
    return client


async def _record(
    clips: list[dict[str, Any]], extra: dict[str, Any] | None = None, **settings_overrides: Any
):
    """Run the real stage over ``clips`` and return its output.

    ``extra`` is merged into the run's ``config_snapshot`` (e.g. an explicit
    ``direction.timing.target_seconds`` or a ``deliverable.target_duration_seconds``)
    so tests can exercise the ordered-target ladder the way ``reconcile_timeline``
    would resolve it.
    """
    client = _ftr_client(clips)
    gen = {
        "bdd_job_id": "bdd-yield",
        "lines": [{"id": str(c["clip_key"])} for c in clips],
    }
    stage = DemoVideoRecordStage(client=client, analyzer=_passing_analyzer())
    with (
        patch(
            "src.services.stages.demo_video.record.get_settings",
            return_value=_qa_settings(**settings_overrides),
        ),
        patch(
            "src.services.stages.demo_video.record.store_capture_clip",
            new=AsyncMock(side_effect=lambda **kw: f"https://minio/{kw['clip_id']}.mp4"),
        ),
    ):
        return await asyncio.wait_for(
            stage.handle(
                stage.input_model(),
                _ctx(
                    {"target_url": "https://app.example"},
                    prior={"demo_video_gen": gen},
                    extra=extra,
                ),
            ),
            timeout=_TEST_TIMEOUT_S,
        )


def _yield_line(caplog: pytest.LogCaptureFixture) -> str:
    lines = [r.getMessage() for r in caplog.records if "scenario yield" in r.getMessage()]
    assert len(lines) == 1, f"expected exactly one yield line, got {lines}"
    return lines[0]


def _num(pattern: str, text: str) -> float:
    match = re.search(pattern, text)
    assert match is not None, f"{pattern!r} not found in {text!r}"
    return float(match.group(1))


@pytest.mark.asyncio
async def test_yield_line_reports_the_measured_mean_min_max_and_total(caplog):
    """A short capture is MEASURED and every figure on the log line is observed.

    The line used to also carry a delta against an assumed 4.0s/scenario. That
    comparison is gone with the constant: measuring a real yield against a fiction
    produced a delta that meant nothing. The measurement itself remains.
    """
    caplog.set_level(logging.INFO, logger="src.services.stages.demo_video.record")
    await _record(
        [
            {"clip_key": "clip-1", "duration_ms": 800},
            {"clip_key": "clip-2", "duration_ms": 1200},
        ]
    )

    line = _yield_line(caplog)
    assert _num(r"mean=([0-9.]+)s/clip", line) == pytest.approx(1.0)
    assert _num(r"total=([0-9.]+)s", line) == pytest.approx(2.0)
    assert _num(r"min=([0-9.]+)s", line) == pytest.approx(0.8)
    assert _num(r"max=([0-9.]+)s", line) == pytest.approx(1.2)
    # ...and the fiction is NOT on the line any more.
    assert "assumed" not in line


@pytest.mark.asyncio
async def test_short_clips_are_attributed_at_harvest_with_their_measured_seconds(caplog):
    """Each clip below the floor is named in ``notes`` with what it actually recorded."""
    caplog.set_level(logging.INFO, logger="src.services.stages.demo_video.record")
    # 8.0s ordered across 2 clips at the 0.5 ratio floors each clip at 2.0s. The floor
    # is a share of what was ORDERED — there is no floor without an order (see
    # `test_short_clip_floor_is_zero_with_no_ordered_target_rather_than_fabricated`),
    # so this test supplies one instead of leaning on the deleted 0.5 x 4.0 fallback.
    out = await _record(
        [
            {"clip_key": "clip-1", "duration_ms": 800},
            {"clip_key": "clip-2", "duration_ms": 4500},
        ],
        extra={"direction": {"timing": {"target_seconds": 8.0}}},
    )

    short_notes = [n for n in out.notes if "below the" in n and "floor" in n]
    assert len(short_notes) == 1, out.notes
    note = short_notes[0]
    assert "clip-1" in note and "clip-2" not in note
    assert _num(r"recorded ([0-9.]+)s", note) == pytest.approx(0.8)
    assert _num(r"below the ([0-9.]+)s floor", note) == pytest.approx(2.0)
    assert _num(r"short by roughly ([0-9.]+)s", note) == pytest.approx(3.2)
    # The healthy clip is still harvested — this attributes, it does not discard.
    assert sorted(c.id for c in out.clips) == ["clip-1", "clip-2"]


@pytest.mark.asyncio
async def test_every_clip_short_adds_the_unreachable_target_summary(caplog):
    caplog.set_level(logging.INFO, logger="src.services.stages.demo_video.record")
    out = await _record(
        [
            {"clip_key": "clip-1", "duration_ms": 800},
            {"clip_key": "clip-2", "duration_ms": 900},
        ],
        extra={"direction": {"timing": {"target_seconds": 8.0}}},
    )

    summary = [n for n in out.notes if "EVERY measured clip" in n]
    assert len(summary) == 1, out.notes
    assert _num(r"EVERY measured clip \(([0-9]+)\)", summary[0]) == 2
    assert _num(r"yields ([0-9.]+)s/clip", summary[0]) == pytest.approx(0.85)


@pytest.mark.asyncio
async def test_clips_comfortably_over_the_ordered_share_produce_no_short_notes(caplog):
    """Clips above the ordered per-clip share must stay silent (2.0s floor, 4.2s+ clips)."""
    caplog.set_level(logging.INFO, logger="src.services.stages.demo_video.record")
    out = await _record(
        [
            {"clip_key": "clip-1", "duration_ms": 4160},
            {"clip_key": "clip-2", "duration_ms": 4400},
        ],
        extra={"direction": {"timing": {"target_seconds": 8.0}}},
    )

    assert [n for n in out.notes if "floor" in n] == []
    assert _num(r"mean=([0-9.]+)s/clip", _yield_line(caplog)) == pytest.approx(4.28)


@pytest.mark.asyncio
async def test_the_floor_moves_with_the_underfull_ratio_it_is_derived_from(caplog):
    """Same 2.5s clip: silent at the default 0.5 ratio, flagged at 0.8."""
    caplog.set_level(logging.INFO, logger="src.services.stages.demo_video.record")
    clip = [{"clip_key": "clip-1", "duration_ms": 2500}]
    # One clip against a 4.0s order: floor = ratio x 4.0.
    ordered = {"direction": {"timing": {"target_seconds": 4.0}}}

    lenient = await _record(clip, extra=ordered, demo_timeline_underfull_fail_ratio=0.5)
    assert [n for n in lenient.notes if "floor" in n] == []

    strict = await _record(clip, extra=ordered, demo_timeline_underfull_fail_ratio=0.8)
    strict_notes = [n for n in strict.notes if "recorded" in n and "floor" in n]
    assert len(strict_notes) == 1, strict.notes
    assert _num(r"below the ([0-9.]+)s floor", strict_notes[0]) == pytest.approx(3.2)

    # Ratio 0 disables the run-level gate; it must disable this one too.
    off = await _record(clip, extra=ordered, demo_timeline_underfull_fail_ratio=0.0)
    assert [n for n in off.notes if "floor" in n] == []


@pytest.mark.asyncio
async def test_an_unmeasured_clip_is_never_called_short(caplog):
    """``duration_ms=0`` is "nobody measured it", not "a zero-second clip".

    FTR races the Playwright webm and reports 0; with per-clip QA returning no probe,
    ``_stamp_durations`` cannot repair it. A guard firing here would fail runs on
    absent evidence — so the clip is excluded from both the mean and the floor, and
    the log says so out loud.
    """
    caplog.set_level(logging.INFO, logger="src.services.stages.demo_video.record")
    out = await _record(
        [
            {"clip_key": "clip-1", "duration_ms": 0},
            {"clip_key": "clip-2", "duration_ms": 0},
        ]
    )

    assert [n for n in out.notes if "floor" in n] == []
    line = _yield_line(caplog)
    assert "UNMEASURED" in line
    assert _num(r"0 of ([0-9]+) clip", line) == 2


@pytest.mark.asyncio
async def test_the_mean_ignores_unmeasured_clips_rather_than_counting_them_as_zero(caplog):
    """One 4.0s clip beside two unmeasured ones yields 4.0, not 1.33."""
    caplog.set_level(logging.INFO, logger="src.services.stages.demo_video.record")
    out = await _record(
        [
            {"clip_key": "clip-1", "duration_ms": 4000},
            {"clip_key": "clip-2", "duration_ms": 0},
            {"clip_key": "clip-3", "duration_ms": 0},
        ]
    )

    line = _yield_line(caplog)
    assert _num(r"mean=([0-9.]+)s/clip", line) == pytest.approx(4.0)
    assert _num(r"over ([0-9]+) of", line) == 1
    assert _num(r"of ([0-9]+) clip", line) == 3
    assert [n for n in out.notes if "floor" in n] == []


# ── the floor must track the ORDERED duration, not always the scenario count ──
#
# The floor must be a share of what was ORDERED. Measured live on
# ``demo_video_calibration`` (orders 30s, ``max_scenarios_per_feature: 3``): the old
# formula floored a clip at 0.5 x 4.0 = 2.0s (0.5 x the ``_SECONDS_PER_SCENARIO``
# constant) while ``reconcile_timeline`` actually required 0.5 x 30.0 = 15.0s.


@pytest.mark.asyncio
async def test_explicit_ordered_target_raises_the_per_clip_floor_end_to_end(caplog):
    """Same 4.5s clips: silent under the old 2.0s floor, flagged under the real 5.0s one."""
    caplog.set_level(logging.INFO, logger="src.services.stages.demo_video.record")
    out = await _record(
        [
            {"clip_key": "clip-1", "duration_ms": 4500},
            {"clip_key": "clip-2", "duration_ms": 4500},
            {"clip_key": "clip-3", "duration_ms": 4500},
        ],
        extra={"direction": {"timing": {"target_seconds": 30.0}}},
    )

    short_notes = [n for n in out.notes if n.startswith("clip ") and "below the" in n]
    assert len(short_notes) == 3, out.notes
    for note in short_notes:
        assert _num(r"below the ([0-9.]+)s floor", note) == pytest.approx(5.0)


@pytest.mark.asyncio
async def test_whole_capture_note_fires_when_total_footage_cannot_reach_the_order(caplog):
    """Real measured shape 01a0496f: 1 clip, 1.76s, ordered 30s.

    ``reconcile_timeline`` will certainly GROSSLY-UNDERFULL-fail this run three stages
    later; the note says so now, quoting both the measured total and the number
    ``reconcile_timeline`` will actually apply.
    """
    caplog.set_level(logging.INFO, logger="src.services.stages.demo_video.record")
    out = await _record(
        [{"clip_key": "clip-1", "duration_ms": 1760}],
        extra={"direction": {"timing": {"target_seconds": 30.0}}},
    )

    summary = [n for n in out.notes if "cannot satisfy reconcile_timeline" in n]
    assert len(summary) == 1, out.notes
    note = summary[0]
    assert _num(r"capture yielded ([0-9.]+)s", note) == pytest.approx(1.76)
    assert _num(r"the ([0-9.]+)s order needs", note) == pytest.approx(30.0)
    assert _num(r">= ([0-9.]+)s at the", note) == pytest.approx(15.0)


@pytest.mark.asyncio
async def test_whole_capture_note_is_silent_when_the_order_is_reachable(caplog):
    """16.0s of footage against a 30s order clears the 50% floor (15.0s) — no note."""
    caplog.set_level(logging.INFO, logger="src.services.stages.demo_video.record")
    out = await _record(
        [
            {"clip_key": "clip-1", "duration_ms": 8000},
            {"clip_key": "clip-2", "duration_ms": 8000},
        ],
        extra={"direction": {"timing": {"target_seconds": 30.0}}},
    )

    assert [n for n in out.notes if "cannot satisfy reconcile_timeline" in n] == []
    assert [n for n in out.notes if "s floor" in n] == []


@pytest.mark.asyncio
async def test_whole_capture_note_uses_the_gates_own_pad_credit_arithmetic(caplog):
    """The predictor must not re-derive the floor with a SECOND formula.

    ``reconcile_timeline`` credits freeze pad against the floor, capped at
    ``MAX_PAD_CREDIT_RATIO`` (10%) of target. 13.50s against a 30.00s order is 45% —
    under the bare 50% floor (15.00s), so a bare-sum predictor claims "this run cannot
    satisfy reconcile_timeline". That claim would be FALSE: the gate credits up to 3.00s
    of pad, reaching 16.50s ≥ 15.00s, and passes. Two formulas for one quantity is
    exactly how live run 01a059a2 shipped (there the divergence ran the other way: the
    predictor was right and the gate passed the run anyway).

    So: no note here, and the note must still fire once even the maximum credited pad
    cannot reach the floor.
    """
    caplog.set_level(logging.INFO, logger="src.services.stages.demo_video.record")
    out = await _record(
        [
            {"clip_key": "clip-1", "duration_ms": 6750},
            {"clip_key": "clip-2", "duration_ms": 6750},
        ],
        extra={"direction": {"timing": {"target_seconds": 30.0}}},
    )
    assert sum([6.75, 6.75]) < 0.5 * 30.0, "must sit BELOW the bare floor, or this is vacuous"
    assert [n for n in out.notes if "cannot satisfy reconcile_timeline" in n] == [], out.notes


@pytest.mark.asyncio
async def test_whole_capture_note_fires_on_the_live_01a059a2_shape(caplog):
    """Live run 01a059a2: 1.40s captured, 5.00s ordered — landed ``completed`` anyway.

    Even the most generous pad the gate will ever credit (10% of 5.00s = 0.50s) reaches
    only 1.90s against the 2.50s floor, so the run provably cannot satisfy the gate —
    and now the gate agrees.
    """
    caplog.set_level(logging.INFO, logger="src.services.stages.demo_video.record")
    out = await _record(
        [{"clip_key": "clip-1", "duration_ms": 1400}],
        extra={
            # The live run's own ladder rung: no explicit `target_seconds`, the
            # deliverable's authored 5s target.
            "direction": {"timing": {"mode": "auto"}},
            "deliverable": {"target_duration_seconds": 5},
        },
    )

    summary = [n for n in out.notes if "cannot satisfy reconcile_timeline" in n]
    assert len(summary) == 1, out.notes
    note = summary[0]
    assert _num(r"capture yielded ([0-9.]+)s", note) == pytest.approx(1.40)
    assert _num(r">= ([0-9.]+)s at the", note) == pytest.approx(2.50)
    assert _num(r"only reaches ([0-9.]+)s", note) == pytest.approx(1.90)


# ── audit findings (unshipped defects fixed alongside #922) ──


@pytest.mark.asyncio
async def test_malformed_ordered_target_degrades_instead_of_crashing_the_stage(caplog):
    """A non-numeric ``target_seconds`` must not escape as an uncaught ValueError.

    ``_resolve_timing_config`` does a bare ``float(raw_target)``; a config typo (or a
    bad LLM-authored override) makes that raise ``ValueError``, which the original
    ``except StageSkippedError`` did not catch — crashing the whole harvest stage over
    a value ``reconcile_timeline`` would otherwise report clearly three stages later.
    This stage is NOTE-only: it must degrade to "no ordered target", not raise.
    """
    caplog.set_level(logging.INFO, logger="src.services.stages.demo_video.record")
    out = await _record(
        [{"clip_key": "clip-1", "duration_ms": 800}],
        extra={"direction": {"timing": {"target_seconds": "not-a-number"}}},
    )
    # No crash, and no FABRICATED floor either. This used to assert the note fired at
    # 0.5 x 4.0 = 2.0s — a threshold built from `_SECONDS_PER_SCENARIO`, which measured
    # nothing, reported to the operator in the same words as a real one. A malformed
    # target is UNMEASURED: the stage degrades to "no ordered target" and stays silent
    # about a floor it cannot compute.
    assert [n for n in out.notes if "below the" in n and "floor" in n] == []
    assert out.clips, "the harvest itself must still succeed"


@pytest.mark.asyncio
async def test_per_clip_note_quotes_the_ordered_share_not_the_scenario_constant(caplog):
    """With an ordered target, the per-clip note must not cite the unrelated 4.0s
    scenario constant or print a negative "short by" delta.

    Same shape as ``test_explicit_ordered_target_raises_the_per_clip_floor_end_to_end``
    (30s ordered / 3 clips ⇒ 5.0s floor), but this asserts the NOTE TEXT, not just the
    floor value: the "assumed" figure quoted must be the 10.00s/clip share of the
    order (30.0 / 3), not the 4.0s scenario constant, and the shortfall must be
    positive (5.0 - 4.5 = 0.5s short), never the nonsensical 4.0 - 4.5 = -0.5s the old
    text produced.
    """
    caplog.set_level(logging.INFO, logger="src.services.stages.demo_video.record")
    out = await _record(
        [
            {"clip_key": "clip-1", "duration_ms": 4500},
            {"clip_key": "clip-2", "duration_ms": 4500},
            {"clip_key": "clip-3", "duration_ms": 4500},
        ],
        extra={"direction": {"timing": {"target_seconds": 30.0}}},
    )

    short_notes = [n for n in out.notes if n.startswith("clip ") and "below the" in n]
    assert len(short_notes) == 3, out.notes
    for note in short_notes:
        assert "4.0s/scenario assumed" not in note, note
        assert _num(r"([0-9.]+)s/clip share of the", note) == pytest.approx(10.0)
        assert _num(r"short by roughly ([0-9.]+)s", note) == pytest.approx(5.5)
