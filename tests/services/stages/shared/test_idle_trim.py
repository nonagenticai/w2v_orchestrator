"""Idle-span trimming of recorded screen footage (``shared/idle_trim`` + the record pass).

Covers the version-robust freezedetect parser, the span-shortening arithmetic, the
two coverage floors (per-clip narration, per-film ordered body), the fail-open paths,
the critic hand-off, and — when ffmpeg is installed — one real detect + cut over a
synthetic clip with a frozen middle.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from src.services.stages.demo_video import record as record_mod
from src.services.stages.demo_video.footage_critique import DemoVideoFootageCritiqueStage
from src.services.stages.demo_video.record import DemoVideoClip, MotionSample
from src.services.stages.shared import idle_trim
from src.services.stages.shared.footage_critique import FootageCritiqueStage
from src.services.stages.shared.idle_trim import FrozenSpan

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext


def _ctx(**kw: Any) -> RunContext:
    return cast("RunContext", SimpleNamespace(**kw))


# ── parser ────────────────────────────────────────────────────────────────────

# ffmpeg 4.4 (local): the filter's own log context, integer seconds.
_STDERR_44 = """\
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'f.mp4':
  Duration: 00:00:07.00, start: 0.000000, bitrate: 41 kb/s
[freezedetect @ 0x7fe80ca0e380] lavfi.freezedetect.freeze_start: 2
[freezedetect @ 0x7fe80ca0e380] lavfi.freezedetect.freeze_duration: 3
[freezedetect @ 0x7fe80ca0e380] lavfi.freezedetect.freeze_end: 5
frame=  175 fps=0.0 q=-0.0 Lsize=N/A time=00:00:07.00 bitrate=N/A speed= 350x
"""

# ffmpeg 6.x/7.x: the parsed-filter context, float seconds, two spans.
_STDERR_7 = """\
[Parsed_freezedetect_0 @ 0x55d0c1a2b3c0] lavfi.freezedetect.freeze_start: 1.04
[Parsed_freezedetect_0 @ 0x55d0c1a2b3c0] lavfi.freezedetect.freeze_duration: 2.52
[Parsed_freezedetect_0 @ 0x55d0c1a2b3c0] lavfi.freezedetect.freeze_end: 3.56
[Parsed_freezedetect_0 @ 0x55d0c1a2b3c0] lavfi.freezedetect.freeze_start: 6.2
[Parsed_freezedetect_0 @ 0x55d0c1a2b3c0] lavfi.freezedetect.freeze_duration: 1.8
[Parsed_freezedetect_0 @ 0x55d0c1a2b3c0] lavfi.freezedetect.freeze_end: 8
[out#0/null @ 0x55d0c1a2c000] video:0KiB audio:0KiB
"""

# ``metadata=mode=print`` form: ``key=value``.
_STDERR_PRINT = """\
[Parsed_metadata_1 @ 0x7fe5d37118c0] frame:50   pts:50   pts_time:2
[Parsed_metadata_1 @ 0x7fe5d37118c0] lavfi.freezedetect.freeze_start=2
[Parsed_metadata_1 @ 0x7fe5d37118c0] lavfi.freezedetect.freeze_duration=3
[Parsed_metadata_1 @ 0x7fe5d37118c0] lavfi.freezedetect.freeze_end=5
"""

# A freeze still open at EOF: only ``freeze_start`` (verified on 4.4).
_STDERR_OPEN_TAIL = "[freezedetect @ 0x7fa800707880] lavfi.freezedetect.freeze_start: 2\n"


@pytest.mark.parametrize(
    ("stderr", "duration", "expected"),
    [(_STDERR_44, 7.0, [FrozenSpan(2.0, 5.0)])],
    ids=["ffmpeg-4.4"],
)
def test_parser_reads_every_ffmpeg_shape(
    stderr: str, duration: float, expected: list[FrozenSpan]
) -> None:
    assert idle_trim.parse_freezedetect(stderr, duration=duration) == expected


def test_parser_open_tail_without_known_duration_is_dropped() -> None:
    # An unbounded span cannot be measured; a guess would be a fabricated number.
    assert idle_trim.parse_freezedetect(_STDERR_OPEN_TAIL, duration=None) == []


def test_parser_falls_back_to_start_plus_duration_and_clamps() -> None:
    stderr = (
        "lavfi.freezedetect.freeze_start: 1.5e0\n"
        "lavfi.freezedetect.freeze_duration: 10\n"
        "lavfi.freezedetect.freeze_start: 0.5\n"  # re-opened before any end
    )
    # First span closes at start+duration (clamped to the 6s clip); the second is open
    # at EOF and closes at the clip end; the overlap is merged into one span.
    assert idle_trim.parse_freezedetect(stderr, duration=6.0) == [FrozenSpan(0.5, 6.0)]


def test_parser_ignores_noise_and_stray_end() -> None:
    stderr = "garbage\nlavfi.freezedetect.freeze_end: 3\nframe=1 fps=0\n"
    assert idle_trim.parse_freezedetect(stderr, duration=4.0) == []
    assert idle_trim.parse_freezedetect("", duration=4.0) == []


# ── span-shortening math ──────────────────────────────────────────────────────


def test_plan_cuts_keeps_a_hold_and_spends_budget_longest_first() -> None:
    spans = [FrozenSpan(1.0, 3.0), FrozenSpan(5.0, 10.0)]
    cuts, protected = idle_trim.plan_cuts(spans, budget=100.0, hold=0.6)
    assert protected == 0
    assert cuts == [(1.6, 3.0), (5.6, 10.0)]
    # Budget smaller than the reducible total: the longest span is shortened first.
    cuts, _ = idle_trim.plan_cuts(spans, budget=3.0, hold=0.6)
    assert cuts == [(5.6, 8.6)]


def test_plan_cuts_protects_a_span_holding_a_recorded_action() -> None:
    spans = [FrozenSpan(1.0, 4.0), FrozenSpan(6.0, 9.0)]
    cuts, protected = idle_trim.plan_cuts(spans, budget=100.0, protected_times=[7.5])
    assert protected == 1
    assert cuts == [(1.6, 4.0)]


def test_plan_cuts_ignores_spans_too_short_to_cut() -> None:
    cuts, _ = idle_trim.plan_cuts([FrozenSpan(0.0, 0.7)], budget=100.0, hold=0.6)
    assert cuts == []


def test_kept_segments_and_remap_are_consistent() -> None:
    cuts = [(1.6, 3.0), (5.6, 10.0)]
    assert idle_trim.kept_segments(cuts, 12.0) == [(0.0, 1.6), (3.0, 5.6), (10.0, 12.0)]
    assert idle_trim.remap_time(1.0, cuts) == pytest.approx(1.0)
    assert idle_trim.remap_time(2.0, cuts) == pytest.approx(1.6)  # inside a cut: collapses
    assert idle_trim.remap_time(4.0, cuts) == pytest.approx(2.6)
    assert idle_trim.remap_time(11.0, cuts) == pytest.approx(11.0 - 1.4 - 4.4)
    assert idle_trim.frozen_after([FrozenSpan(1.0, 3.0), FrozenSpan(5.0, 10.0)], cuts) == (
        pytest.approx(1.2)
    )


def test_concat_filter_names_every_segment() -> None:
    graph = idle_trim.concat_filter([(0.0, 1.6), (3.0, 5.6)])
    assert "trim=start=0.000:end=1.600" in graph
    assert "trim=start=3.000:end=5.600" in graph
    assert graph.endswith("[s0][s1]concat=n=2:v=1:a=0[out]")


# ── coverage floors ───────────────────────────────────────────────────────────


def test_clip_floor_is_narration_with_margin_never_below_two_seconds() -> None:
    assert idle_trim.clip_floor_seconds(0.0) == idle_trim.MIN_KEEP_SECONDS
    assert idle_trim.clip_floor_seconds(10.0) == pytest.approx(11.0)
    # 26 words at 130 wpm = 12.0s
    assert idle_trim.estimated_narration_seconds(["word " * 26], 130.0) == pytest.approx(12.0)


def test_allocate_caps_per_clip_then_scales_to_the_film_floor() -> None:
    budgets = idle_trim.allocate_trim_budgets(
        {"a": 6.0, "b": 4.0}, {"a": 2.0, "b": 10.0}, global_room=None
    )
    assert budgets == {"a": 2.0, "b": 4.0}
    # Kept total may only drop 3s below the footage total: 6s of wanted cuts scale by 0.5.
    budgets = idle_trim.allocate_trim_budgets(
        {"a": 6.0, "b": 4.0}, {"a": 2.0, "b": 10.0}, global_room=3.0
    )
    assert budgets == {"a": pytest.approx(1.0), "b": pytest.approx(2.0)}
    # Footage already under the order ⇒ nothing may be cut.
    assert idle_trim.allocate_trim_budgets({"a": 6.0}, {"a": 6.0}, global_room=-1.0) == {"a": 0.0}


def test_blocked_status_names_the_binding_floor() -> None:
    assert idle_trim.blocked_status(4.0, 0.2, 0.2) == idle_trim.STATUS_CLIP_FLOOR
    assert idle_trim.blocked_status(4.0, 2.0, 0.0) == idle_trim.STATUS_FILM_FLOOR
    assert idle_trim.blocked_status(0.4, 2.0, 0.4) == idle_trim.STATUS_NO_IDLE


# ── the record-stage pass (ffmpeg stubbed) ────────────────────────────────────


def _clip(cid: str = "clip-1", seconds: float = 12.0, **kw: Any) -> DemoVideoClip:
    return DemoVideoClip(
        id=cid,
        line_id=cid,
        video_url=f"https://minio.test/run/{cid}.mp4",
        seconds=seconds,
        **kw,
    )


class _Stubs:
    def __init__(self, spans: dict[str, list[FrozenSpan]]) -> None:
        self.spans = spans
        self.cut_calls: list[tuple[str, list[tuple[float, float]]]] = []
        self.stored: list[str] = []
        self.detect_error: Exception | None = None
        self.cut_error: Exception | None = None
        self.timeouts: list[tuple[str, float | None]] = []

    async def detect(
        self, path: Path, *, usable: float, timeout_s: float | None = None
    ) -> list[FrozenSpan]:
        self.timeouts.append(("detect", timeout_s))
        if self.detect_error:
            raise self.detect_error
        return self.spans.get(path.stem, [])

    async def cut(
        self,
        src: Path,
        dst: Path,
        segments: list[tuple[float, float]],
        *,
        timeout_s: float | None = None,
    ) -> None:
        self.timeouts.append(("cut", timeout_s))
        if self.cut_error:
            raise self.cut_error
        self.cut_calls.append((src.stem, list(segments)))
        dst.write_bytes(b"trimmed")

    async def probe(self, path: Path, *, timeout_s: float | None = None) -> float | None:
        self.timeouts.append(("probe", timeout_s))
        return None  # force the arithmetic fallback

    async def store(self, **kw: Any) -> str:
        self.stored.append(kw["clip_id"])
        return f"https://minio.test/run/{kw['clip_id']}.mp4"


@pytest.fixture
def stubs(monkeypatch: pytest.MonkeyPatch) -> _Stubs:
    s = _Stubs({})
    monkeypatch.setattr(idle_trim, "detect_frozen_spans", s.detect)
    monkeypatch.setattr(idle_trim, "cut_segments", s.cut)
    monkeypatch.setattr(idle_trim, "probe_seconds", s.probe)
    monkeypatch.setattr(record_mod, "store_capture_clip", s.store)
    return s


def _run(clips: list[DemoVideoClip], lines: list[dict[str, Any]], **kw: Any) -> float:
    return asyncio.run(
        record_mod._trim_idle_footage(
            clips,
            {c.id: (b"bytes", "mp4") for c in clips},
            lines=lines,
            wpm=kw.pop("wpm", 130.0),
            ordered_target=kw.pop("ordered_target", None),
            run_id="run-1",
            stage_name="demo_video_record",
            **kw,
        )
    )


def test_pass_trims_remaps_and_records(stubs: _Stubs) -> None:
    stubs.spans = {"clip-1": [FrozenSpan(2.0, 7.0)]}
    clip = _clip(
        focus_track=[{"t": 1.0}, {"t": 8.0}],
        cursor_track=[{"t": 9.0, "x": 1}],
        motion_timeline=[MotionSample(t=1.0, motion=0.2), MotionSample(t=4.0, motion=0.0)],
    )
    removed = _run([clip], [{"id": "clip-1", "text": "short line"}])
    assert removed == pytest.approx(4.4)
    assert stubs.cut_calls == [("clip-1", [(0.0, 2.6), (7.0, 12.0)])]
    assert stubs.stored == ["clip-1-idle"]
    assert clip.video_url.endswith("clip-1-idle.mp4")
    assert clip.seconds == pytest.approx(7.6)
    assert clip.recorded_seconds is None
    assert clip.focus_track == [{"t": 1.0}, {"t": 3.6}]
    assert clip.cursor_track == [{"t": 4.6, "x": 1}]
    assert [s.t for s in clip.motion_timeline or []] == [1.0]  # the frozen sample is gone
    rec = clip.idle_trim
    assert rec is not None
    assert rec.status == idle_trim.STATUS_TRIMMED
    assert rec.frozen_seconds_before == pytest.approx(5.0)
    assert rec.frozen_seconds_after == pytest.approx(0.6)
    assert rec.idle_fraction_before == pytest.approx(5.0 / 12.0, abs=1e-4)
    assert rec.idle_fraction_after == pytest.approx(0.6 / 7.6, abs=1e-4)
    assert rec.spans_shortened == 1


def test_narration_floor_caps_the_trim(stubs: _Stubs) -> None:
    # 22 words at 130 wpm ≈ 10.15s; x1.10 ≈ 11.17s floor on a 12s clip ⇒ ≤0.83s may go.
    stubs.spans = {"clip-1": [FrozenSpan(2.0, 7.0)]}
    clip = _clip()
    _run([clip], [{"id": "clip-1", "text": "word " * 22}])
    assert clip.seconds >= 11.16
    rec = clip.idle_trim
    assert rec is not None
    assert rec.removed_seconds <= 12.0 - rec.floor_seconds + 1e-6


def test_narration_floor_above_clip_leaves_it_untouched(stubs: _Stubs) -> None:
    stubs.spans = {"clip-1": [FrozenSpan(2.0, 7.0)]}
    clip = _clip()
    _run([clip], [{"id": "clip-1", "text": "word " * 40}])
    assert stubs.cut_calls == []
    assert clip.seconds == 12.0
    assert clip.idle_trim is not None
    assert clip.idle_trim.status == idle_trim.STATUS_CLIP_FLOOR


def test_ordered_body_floor_caps_the_whole_film(stubs: _Stubs) -> None:
    stubs.spans = {"clip-1": [FrozenSpan(2.0, 7.0)], "clip-2": [FrozenSpan(1.0, 6.0)]}
    clips = [_clip("clip-1"), _clip("clip-2")]
    lines = [{"id": "clip-1", "text": "a"}, {"id": "clip-2", "text": "b"}]
    removed = _run(clips, lines, ordered_target=22.0)  # 24s of footage, only 2s may go
    assert removed <= 2.0 + 1e-6
    assert sum(c.seconds for c in clips) >= 22.0 - 1e-6


def test_unplayed_clip_creates_no_film_room(stubs: _Stubs) -> None:
    # A split's second pane has no line over it: it is never played, so its 12s must
    # not count toward the ordered body. Played footage = clip-1's 12s against an order
    # of 12s => zero room, nothing may be cut (counting the pane would allow 12s).
    stubs.spans = {"clip-1": [FrozenSpan(2.0, 7.0)], "pane-2": [FrozenSpan(2.0, 7.0)]}
    played, pane = _clip("clip-1"), _clip("pane-2")
    removed = _run([played, pane], [{"id": "clip-1", "text": "x"}], ordered_target=12.0)
    assert removed == 0.0
    assert stubs.cut_calls == []
    assert played.seconds == 12.0
    assert played.idle_trim is not None
    assert played.idle_trim.status == idle_trim.STATUS_FILM_FLOOR


# Run 01a15b57-9403 (showcase): three clips, gen's words per clip, the voice MEASURED at
# 176.56 wpm, ordered 30s. Each clip has idle and room of its own at the measured rate,
# but played footage (29.76s) is already under the order, so the per-film floor binds.
_RUN_WPM = 176.55581589032246
_RUN_CLIPS = (("c1", 10.32, 24), ("c2", 10.96, 24), ("c3", 8.48, 8))


def _run_clips(scale: float = 1.0) -> tuple[list[DemoVideoClip], list[dict[str, Any]]]:
    clips = [_clip(cid, round(sec * scale, 3)) for cid, sec, _ in _RUN_CLIPS]
    lines = [{"id": cid, "text": "word " * words} for cid, _, words in _RUN_CLIPS]
    return clips, lines


def test_run_01a15b57_reports_film_floor_and_cuts_nothing(stubs: _Stubs) -> None:
    clips, lines = _run_clips()
    stubs.spans = {c.id: [FrozenSpan(1.0, c.seconds - 0.5)] for c in clips}
    removed = _run(clips, lines, wpm=_RUN_WPM, ordered_target=30.0)
    assert removed == 0.0
    assert stubs.cut_calls == []
    for clip, (_, seconds, words) in zip(clips, _RUN_CLIPS, strict=True):
        rec = clip.idle_trim
        assert rec is not None
        assert rec.status == idle_trim.STATUS_FILM_FLOOR, clip.id
        assert clip.seconds == seconds
        # The floor is the MEASURED narration x1.10 and sits under the clip (the 130-wpm
        # clamp put c1's at 12.185s on a 10.32s clip).
        expected = max(2.0, words / _RUN_WPM * 60.0 * idle_trim.NARRATION_MARGIN)
        assert rec.floor_seconds == pytest.approx(expected, abs=1e-3)
        assert rec.floor_seconds < seconds


def test_footage_over_the_order_trims_down_to_it_never_under_a_clip_floor(
    stubs: _Stubs,
) -> None:
    # Same voice and words, but 40s of played footage against the 30s order.
    clips, lines = _run_clips(scale=40.0 / 29.76)
    stubs.spans = {c.id: [FrozenSpan(1.0, c.seconds - 0.5)] for c in clips}
    floors = {
        cid: max(2.0, words / _RUN_WPM * 60.0 * idle_trim.NARRATION_MARGIN)
        for cid, _, words in _RUN_CLIPS
    }
    removed = _run(clips, lines, wpm=_RUN_WPM, ordered_target=30.0)
    assert removed > 0.0
    # plan_cuts rounds each cut's endpoints to the millisecond: <=1ms per clip.
    assert sum(c.seconds for c in clips) >= 30.0 - 1e-3 * len(clips)
    for clip in clips:
        rec = clip.idle_trim
        assert rec is not None
        assert rec.status == idle_trim.STATUS_TRIMMED, clip.id
        assert rec.removed_seconds > 0.0
        assert clip.seconds >= floors[clip.id] - 1e-6


def test_authored_window_counts_only_its_window(stubs: _Stubs) -> None:
    # clip-2 plays only its 0-4s window: played = 12 + 4 = 16 vs order 15 => 1s room.
    stubs.spans = {"clip-1": [FrozenSpan(2.0, 9.0)]}
    clips = [_clip("clip-1"), _clip("clip-2")]
    lines = [
        {"id": "clip-1", "text": "x"},
        {"id": "clip-2", "text": "y", "clip_out_seconds": 4.0},
    ]
    removed = _run(clips, lines, ordered_target=15.0)
    assert removed <= 1.0 + 1e-6


def test_played_seconds_rules() -> None:
    clip = _clip(seconds=12.0)
    assert record_mod._played_seconds(clip, []) == 0.0
    assert record_mod._played_seconds(clip, [{"id": "a"}]) == 12.0
    assert record_mod._played_seconds(clip, [{"id": "a", "clip_out_seconds": 30.0}]) == 12.0
    lead = {"id": "a", "clip_lead_seconds": 3.0, "clip_out_seconds": 8.0}
    assert record_mod._played_seconds(clip, [lead]) == 5.0
    assert record_mod._played_seconds(clip, [{"id": "a", "clip_anchor": "mystery"}]) == 0.0


class _FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now


def test_total_budget_is_hard_remaining_time_bounds_each_call(
    stubs: _Stubs, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _FakeClock()
    monkeypatch.setattr(record_mod, "time", clock)
    stubs.spans = {"clip-1": [FrozenSpan(2.0, 7.0)], "clip-2": [FrozenSpan(2.0, 7.0)]}
    real_detect, real_cut = stubs.detect, stubs.cut

    async def detect(path: Path, *, usable: float, timeout_s: float | None = None) -> list:
        clock.now += 50.0  # two detects spend 100s of the 120s budget
        return await real_detect(path, usable=usable, timeout_s=timeout_s)

    async def cut(src: Path, dst: Path, segments: list, *, timeout_s: float | None = None) -> None:
        await real_cut(src, dst, segments, timeout_s=timeout_s)
        assert timeout_s is not None
        clock.now += timeout_s  # a cut that runs to its ceiling eats the whole remainder
        raise idle_trim.IdleTrimError(f"ffmpeg exceeded {timeout_s:.0f}s")

    monkeypatch.setattr(idle_trim, "detect_frozen_spans", detect)
    monkeypatch.setattr(idle_trim, "cut_segments", cut)
    clips = [_clip("clip-1"), _clip("clip-2")]
    lines = [{"id": "clip-1", "text": "x"}, {"id": "clip-2", "text": "y"}]
    removed = _run(clips, lines, budget_s=120.0)

    assert removed == 0.0
    # Detects were handed the remaining budget, not the bare 90s ceiling...
    assert [t for k, t in stubs.timeouts if k == "detect"] == [120.0, 70.0]
    # ...and the cut got the 20s left, not CUT_TIMEOUT_S (180s).
    assert [t for k, t in stubs.timeouts if k == "cut"] == [pytest.approx(20.0)]
    assert clock.now == pytest.approx(1120.0)  # the pass ended AT the cap
    # Fail-open: the interrupted clip is as harvested; the next is skipped unrun.
    assert all(c.seconds == 12.0 and c.video_url.endswith(f"{c.id}.mp4") for c in clips)
    assert clips[0].idle_trim is not None
    assert clips[0].idle_trim.status == idle_trim.STATUS_CUT_FAILED
    assert clips[1].idle_trim is not None
    assert clips[1].idle_trim.status == idle_trim.STATUS_BUDGET_EXHAUSTED


def test_bounded_tightens_and_refuses_a_spent_budget() -> None:
    assert idle_trim._bounded(180.0, None) == 180.0
    assert idle_trim._bounded(180.0, 7.5) == 7.5
    assert idle_trim._bounded(30.0, 99.0) == 30.0
    with pytest.raises(idle_trim.IdleTrimError):
        idle_trim._bounded(30.0, 0.0)


def test_short_clip_floor_is_respected(stubs: _Stubs) -> None:
    stubs.spans = {"clip-1": [FrozenSpan(2.0, 7.0)]}
    clip = _clip()
    _run([clip], [{"id": "clip-1", "text": "x"}], min_clip_seconds=10.0)
    assert clip.seconds == pytest.approx(10.0)
    assert clip.idle_trim is not None
    assert clip.idle_trim.floor_seconds == 10.0


def test_lines_sharing_a_recording_sum_their_narration(stubs: _Stubs) -> None:
    stubs.spans = {"clip-1": [FrozenSpan(2.0, 7.0)]}
    clip = _clip()
    lines = [
        {"id": "l1", "clip": "clip-1", "text": "word " * 11},
        {"id": "l2", "clip": "clip-1", "text": "word " * 11},
    ]
    _run([clip], lines)
    assert clip.idle_trim is not None
    assert clip.idle_trim.floor_seconds == pytest.approx(22 / 130 * 60 * 1.1, abs=1e-3)


def test_detect_failure_is_fail_open(stubs: _Stubs) -> None:
    stubs.detect_error = idle_trim.IdleTrimError("ffmpeg exited 1: invalid data")
    clip = _clip()
    before = clip.model_dump(exclude={"idle_trim"})
    assert _run([clip], [{"id": "clip-1", "text": "x"}]) == 0.0
    assert clip.model_dump(exclude={"idle_trim"}) == before
    assert clip.idle_trim is not None
    assert clip.idle_trim.status == idle_trim.STATUS_DETECT_FAILED
    assert "invalid data" in clip.idle_trim.detail


def test_cut_failure_is_fail_open(stubs: _Stubs) -> None:
    stubs.spans = {"clip-1": [FrozenSpan(2.0, 7.0)]}
    stubs.cut_error = idle_trim.IdleTrimError("ffmpeg exceeded 180s")
    clip = _clip(focus_track=[{"t": 8.0}])
    _run([clip], [{"id": "clip-1", "text": "x"}])
    assert clip.seconds == 12.0
    assert clip.focus_track == [{"t": 8.0}]
    assert clip.video_url.endswith("clip-1.mp4")
    assert clip.idle_trim is not None
    assert clip.idle_trim.status == idle_trim.STATUS_CUT_FAILED


def test_store_failure_is_fail_open(stubs: _Stubs, monkeypatch: pytest.MonkeyPatch) -> None:
    async def _boom(**_kw: Any) -> str:
        raise RuntimeError("minio down")

    monkeypatch.setattr(record_mod, "store_capture_clip", _boom)
    stubs.spans = {"clip-1": [FrozenSpan(2.0, 7.0)]}
    clip = _clip()
    assert _run([clip], [{"id": "clip-1", "text": "x"}]) == 0.0
    assert clip.seconds == 12.0
    assert clip.idle_trim is not None
    assert clip.idle_trim.status == idle_trim.STATUS_CUT_FAILED


def test_ineligible_clips_are_skipped(stubs: _Stubs) -> None:
    stubs.spans = {"clip-1": [FrozenSpan(2.0, 7.0)], "clip-2": [FrozenSpan(2.0, 7.0)]}
    authored = _clip("clip-1")
    unbound = _clip("clip-2")  # e.g. a split's second pane: no line plays over it
    _run([authored, unbound], [{"id": "clip-1", "text": "x", "clip_out_seconds": 9.0}])
    assert stubs.cut_calls == []
    assert authored.idle_trim is not None
    assert authored.idle_trim.status == idle_trim.STATUS_SKIPPED
    assert unbound.idle_trim is not None
    assert unbound.idle_trim.status == idle_trim.STATUS_SKIPPED


def test_no_idle_is_recorded_not_cut(stubs: _Stubs) -> None:
    clip = _clip()
    _run([clip], [{"id": "clip-1", "text": "x"}])
    assert stubs.cut_calls == []
    assert clip.idle_trim is not None
    assert clip.idle_trim.status == idle_trim.STATUS_NO_IDLE
    assert clip.idle_trim.idle_fraction_before == 0.0


def test_idle_pass_can_be_disabled() -> None:
    off = _ctx(config_snapshot={"record": {"idle_trim": False}})
    on = _ctx(config_snapshot={})
    assert record_mod._idle_trim_enabled(off) is False
    assert record_mod._idle_trim_enabled(on) is True


# ── critic hand-off ───────────────────────────────────────────────────────────


def test_critic_rows_carry_the_idle_measurement() -> None:
    ctx = _ctx(
        prior_run_outputs={},
        prior_output=lambda name: {
            "demo_video_record": {
                "clips": [
                    {
                        "id": "clip-1",
                        "idle_trim": {
                            "status": "trimmed",
                            "idle_fraction_before": 0.42,
                            "idle_fraction_after": 0.08,
                            "removed_seconds": 4.4,
                            "detail": "not shown to the critic",
                        },
                    }
                ]
            },
            "demo_video_gen": {
                "lines": [{"id": "l1", "clip": "clip-1"}, {"id": "l2", "clip": "clip-9"}]
            },
        }.get(name),
    )
    stage = DemoVideoFootageCritiqueStage.__new__(DemoVideoFootageCritiqueStage)
    idle = stage.idle_trim_by_shot(ctx)
    assert idle == {
        "l1": {
            "status": "trimmed",
            "idle_fraction_before": 0.42,
            "idle_fraction_after": 0.08,
            "removed_seconds": 4.4,
        }
    }
    base = FootageCritiqueStage.__new__(FootageCritiqueStage)
    rows = base._measure(
        None,
        [],
        [{"shot_id": "l1", "t0": 0.0, "t1": 3.0}, {"shot_id": "l2", "t0": 3.0, "t1": 6.0}],
        [{"line_id": "l1"}, {"line_id": "l2"}],
        idle_by_shot=idle,
    )
    assert rows[0]["idle_trim"]["idle_fraction_after"] == 0.08
    assert "idle_trim" not in rows[1]


# ── real ffmpeg (skipped when absent) ─────────────────────────────────────────

_HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg/ffprobe not installed")
def test_real_ffmpeg_detects_and_cuts_a_frozen_middle(tmp_path: Path) -> None:
    src = tmp_path / "clip.mp4"
    # 2s of moving testsrc, held frozen for 4s (tpad clone), then 2s more motion.
    subprocess.run(  # noqa: S603 — fixed args, test-only
        [  # noqa: S607
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=25:duration=2",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=25:duration=2",
            "-filter_complex",
            "[0:v]tpad=stop_mode=clone:stop_duration=4[a];[a][1:v]concat=n=2:v=1:a=0[v]",
            "-map",
            "[v]",
            "-pix_fmt",
            "yuv420p",
            str(src),
        ],
        check=True,
    )

    async def _go() -> tuple[list[FrozenSpan], float | None]:
        spans = await idle_trim.detect_frozen_spans(src, usable=8.0)
        cuts, _ = idle_trim.plan_cuts(spans, budget=100.0)
        dst = tmp_path / "out.mp4"
        await idle_trim.cut_segments(src, dst, idle_trim.kept_segments(cuts, 8.0))
        return spans, await idle_trim.probe_seconds(dst)

    spans, seconds = asyncio.run(_go())
    assert len(spans) == 1
    # The frozen run starts on the last moving frame (~1.96-2.0s) and ends at 6.0s.
    assert spans[0].start == pytest.approx(2.0, abs=0.1)
    assert spans[0].end == pytest.approx(6.0, abs=0.1)
    assert seconds is not None
    # 8s - (4s frozen - 0.6s hold) ≈ 4.6s
    assert seconds == pytest.approx(4.6, abs=0.15)
