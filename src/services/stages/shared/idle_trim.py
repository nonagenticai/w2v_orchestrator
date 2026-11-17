"""Idle-span trimming for recorded screen footage — shorten frozen stretches BEFORE compile.

WHY. A screen recording of a web app is mostly a page sitting still: the scenario
clicks, the app settles, Playwright waits on the next assertion, and the recording
keeps rolling over a frame that does not change. The footage critic
(``shared/footage_critique.py``) then scores those shots low on ``boring_cut`` /
``app_demoed``, the min-across-shots score lands under the 4.0 bar and the run ships
``footage_below_bar``. A re-render cannot fix that — the nudges only touch framing —
and nothing in the orchestrator ever looked for the idle itself. The one place in the
estate that did is ``video_production_service`` ``video_editor.remove_dead_time``
(ffmpeg ``freezedetect``), which is the pattern borrowed here.

WHAT. For each recorded clip: run ``freezedetect`` over the usable footage, and
shorten every frozen span to a short HOLD by CUTTING frames out of it (a trim, never a
freeze-frame: the kept frames are real frames, and a frozen span's frames are by
definition interchangeable). The clip is re-encoded with the kept segments
concatenated, and every time-indexed by-product of the recording (focus/cursor
tracks, motion curve) is remapped onto the new axis with :func:`remap_time`.

COVERAGE IS PROTECTED, NOT ASSUMED. A film is exactly its narration and the compiler
freeze-pads the LAST frame of a footage beat whose narration outruns it — so a trim
that left footage shorter than its narration would just move the idle from the middle
of the shot to a frozen TAIL. The trim is therefore budgeted, per clip and per film:

* per clip: never below ``max(MIN_KEEP_SECONDS, estimated narration x margin)`` of the
  lines the clip backs (:func:`clip_floor_seconds`);
* per film: the kept total never drops below the ordered narration body
  (:func:`allocate_trim_budgets`), so a film that was on its order is not made short.

FAIL-OPEN. Any detection, cut, probe or storage error leaves the clip exactly as it was
and is recorded on the clip's :class:`ClipIdleTrim` (``status``) and logged. This is a
quality pass; it must never fail the record stage.

PARSER. ffmpeg's freezedetect writes frame metadata as log lines, and the shape is not
identical across the three ffmpeg versions this code meets (local 4.4, CI 6.1.1, prod
7.1.x): the logging context differs (``[freezedetect @ 0x..]`` vs
``[Parsed_freezedetect_0 @ 0x..]``), ``metadata=print`` uses ``key=value`` where the
filter's own log uses ``key: value``, numbers may be integers or floats, and a freeze
still open at EOF emits ONLY ``freeze_start`` (verified on 4.4). The parser matches the
``lavfi.freezedetect.freeze_(start|duration|end)`` keys and nothing else.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from src.services.stages.base import StageOutput

log = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
#: Noise tolerance for "nothing changed". ffmpeg's own freezedetect default. It is the
#: mean absolute frame difference as a fraction of full scale (-60 dB = 0.001), which is
#: far above the codec noise on a static UI (an H.264/VP8 encode of a still page measures
#: well under it) and below any real UI change of note — a route change, a modal, a
#: scroll moves whole regions of the viewport. A few-pixel change (a blinking caret, one
#: typed glyph) CAN sit under it, which is why spans holding a recorded focus/cursor event
#: are protected (:func:`plan_cuts`) rather than trusted to the pixel threshold alone.
FREEZE_NOISE_DB: Final = -60.0
#: Minimum frozen duration to report. Shorter stillness is a natural beat — the viewer
#: reading what just changed — and is left alone; 1.5s is past a comfortable read of a
#: settled UI change and comfortably under the 5s Playwright expect waits that produce
#: the long idle stretches this exists for.
FREEZE_MIN_SECONDS: Final = 1.5
#: What a shortened frozen span keeps: enough for the eye to register the settled state
#: after the action that produced it, short enough not to read as a stall. Inside the
#: brief's 0.5-0.8s band.
HOLD_SECONDS: Final = 0.6
#: Absolute floor on a trimmed clip — the same 2.0s floor ``reconcile_timeline`` applies
#: to a footage trim (``max(narration, 60% clip, 2.0s)``).
MIN_KEEP_SECONDS: Final = 2.0
#: Headroom over the estimated narration: the script is still rewritten downstream
#: (script_doctor, reconcile_timeline's re-ask), so the estimate can grow. Same 1.10 ratio
#: gen applies to its whole-script word ceiling.
NARRATION_MARGIN: Final = 1.10
#: A single cut shorter than this is not worth a segment boundary.
MIN_CUT_SECONDS: Final = 0.25
#: A whole clip whose total cut is below this is left untouched (no re-encode).
MIN_CLIP_SAVING_SECONDS: Final = 0.5

DETECT_TIMEOUT_S: Final = 90.0
CUT_TIMEOUT_S: Final = 180.0
PROBE_TIMEOUT_S: Final = 30.0

# ── Status values carried on ClipIdleTrim.status ──────────────────────────────
STATUS_TRIMMED: Final = "trimmed"
STATUS_NO_IDLE: Final = "no_idle"
#: Idle found, but THIS clip's own floor (its narration x margin, or the short-clip
#: floor) is at/above what the clip could lose — the clip itself has no room.
STATUS_CLIP_FLOOR: Final = "clip_floor"
#: Idle found and the clip had room of its own, but the PER-FILM floor (played footage
#: must stay >= the ordered body) left the film no room — the budget was scaled away.
#: Expected and correct on a film already at/under its order; not a defect.
STATUS_FILM_FLOOR: Final = "film_floor"
STATUS_PROTECTED: Final = "protected"  # every idle span held a recorded action
STATUS_SKIPPED: Final = "skipped"  # not eligible (authored window, no line, unmeasured)
STATUS_DETECT_FAILED: Final = "detect_failed"
STATUS_CUT_FAILED: Final = "cut_failed"
STATUS_BUDGET_EXHAUSTED: Final = "budget_exhausted"

_FREEZE_KEY_RE = re.compile(
    r"lavfi\.freezedetect\.freeze_(start|duration|end)\s*[:=]\s*"
    r"(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


class IdleTrimError(RuntimeError):
    """ffmpeg/ffprobe could not complete a detect / cut / probe."""


class ClipIdleTrim(StageOutput):
    """What the idle pass measured and did for ONE recorded clip.

    ``*_before`` are measured by freezedetect on the usable footage; ``*_after`` are
    COMPUTED from the cuts actually applied (the clip is not re-detected). Fractions are
    of the clip's usable length before/after. ``status`` is one of the ``STATUS_*``
    values; ``detail`` carries the reason on a fail-open or skip.
    """

    status: str
    seconds_before: float = 0.0
    seconds_after: float = 0.0
    frozen_seconds_before: float = 0.0
    frozen_seconds_after: float = 0.0
    removed_seconds: float = 0.0
    idle_fraction_before: float = 0.0
    idle_fraction_after: float = 0.0
    spans_detected: int = 0
    spans_shortened: int = 0
    spans_protected: int = 0
    floor_seconds: float = 0.0
    detail: str = ""


@dataclass(frozen=True)
class FrozenSpan:
    """``[start, end)`` seconds over which freezedetect saw no change."""

    start: float
    end: float

    @property
    def seconds(self) -> float:
        return max(0.0, self.end - self.start)


# ── Pure helpers ──────────────────────────────────────────────────────────────
def _freeze_events(stderr: str) -> list[tuple[str, float]]:
    """``(key, value)`` for every freezedetect metadata key in ``stderr``, in order."""
    events: list[tuple[str, float]] = []
    for match in _FREEZE_KEY_RE.finditer(stderr or ""):
        try:
            value = float(match.group(2))
        except ValueError:
            continue
        if math.isfinite(value):
            events.append((match.group(1), value))
    return events


def parse_freezedetect(stderr: str, *, duration: float | None = None) -> list[FrozenSpan]:
    """Frozen spans from freezedetect's log, version-robust (see module docstring).

    Pairs each ``freeze_start`` with the next ``freeze_end``; failing that, with
    ``start + freeze_duration``; failing both (a freeze still open at EOF), with
    ``duration`` when known — else the span is dropped, since an unbounded span cannot
    be measured. Spans are clamped to ``[0, duration]``, sorted, and non-positive ones
    dropped; overlapping spans are merged. Unparseable lines are ignored.
    """
    raw: list[tuple[float, float | None]] = []
    start: float | None = None
    span_len: float | None = None
    for key, value in [*_freeze_events(stderr), ("start", math.nan)]:
        if key == "duration":
            span_len = value
            continue
        if start is not None:
            # A ``start`` (or the sentinel) closes an open span; an ``end`` closes it with
            # its own value.
            if key == "end":
                raw.append((start, value))
            else:
                raw.append((start, start + span_len if span_len is not None else duration))
        start = value if key == "start" and math.isfinite(value) else None
        span_len = None

    out: list[FrozenSpan] = []
    for raw_lo, raw_hi in sorted((s, e) for s, e in raw if e is not None):
        lo = max(0.0, raw_lo)
        hi = raw_hi if duration is None else min(raw_hi, duration)
        if hi - lo <= 0.0:
            continue
        if out and lo <= out[-1].end:
            # Overlap (only a malformed log produces one): merge, so no frame is cut twice.
            out[-1] = FrozenSpan(out[-1].start, max(out[-1].end, hi))
        else:
            out.append(FrozenSpan(lo, hi))
    return out


def estimated_narration_seconds(texts: Iterable[str], wpm: float) -> float:
    """Spoken seconds for ``texts`` at ``wpm`` (whitespace word count)."""
    if wpm <= 0:
        return 0.0
    words = sum(len(str(t).split()) for t in texts)
    return words / wpm * 60.0


def clip_floor_seconds(narration_seconds: float) -> float:
    """The shortest a clip may be trimmed to: its narration (with margin), never < 2s."""
    return max(MIN_KEEP_SECONDS, narration_seconds * NARRATION_MARGIN)


def reducible_seconds(spans: Sequence[FrozenSpan], hold: float = HOLD_SECONDS) -> float:
    """Seconds that shortening every span to ``hold`` would remove."""
    return sum(max(0.0, s.seconds - hold) for s in spans)


def allocate_trim_budgets(
    reducible_by_clip: Mapping[str, float],
    room_by_clip: Mapping[str, float],
    global_room: float | None,
) -> dict[str, float]:
    """Per-clip trim budget under the per-clip floor AND the film-level floor.

    Each clip may lose ``min(reducible, room)``; when the sum exceeds ``global_room``
    (kept total must stay >= the ordered body), every clip's budget is scaled by the
    same factor so no single beat absorbs the whole shortfall. ``global_room=None``
    means nothing was ordered — only the per-clip floor applies.
    """
    budgets = {
        cid: max(0.0, min(reducible_by_clip.get(cid, 0.0), room_by_clip.get(cid, 0.0)))
        for cid in reducible_by_clip
    }
    total = sum(budgets.values())
    if global_room is not None and total > 0.0:
        allowed = max(0.0, global_room)
        if total > allowed:
            scale = allowed / total
            budgets = {cid: b * scale for cid, b in budgets.items()}
    return budgets


def blocked_status(reducible: float, room: float, budget: float) -> str:
    """WHY a clip with cuttable idle was left uncut — which floor bound it.

    ``reducible`` is the idle the clip could lose, ``room`` what its own floor allows,
    ``budget`` what it got after the film-level scaling. The clip's own floor binds
    first: when ``room`` is below both the idle and the minimum worthwhile saving, the
    clip had nothing to give (``clip_floor``). Otherwise, when the film scaling took the
    budget under the minimum saving, the per-film floor bound (``film_floor``). Neither
    ⇒ the idle itself was too little to be worth a re-encode (``no_idle``).
    """
    if room < reducible and room < MIN_CLIP_SAVING_SECONDS:
        return STATUS_CLIP_FLOOR
    if budget < min(reducible, room) and budget < MIN_CLIP_SAVING_SECONDS:
        return STATUS_FILM_FLOOR
    return STATUS_NO_IDLE


def plan_cuts(
    spans: Sequence[FrozenSpan],
    budget: float,
    *,
    hold: float = HOLD_SECONDS,
    protected_times: Iterable[float] = (),
) -> tuple[list[tuple[float, float]], int]:
    """Intervals to REMOVE (sorted, disjoint) and how many spans were protected.

    Each span keeps its first ``hold`` seconds (the settled result of the action before
    it) and loses up to ``span - hold`` after that. A span with a recorded action time
    inside its would-be cut is PROTECTED and left whole — the pixel threshold can miss a
    small change (one typed glyph), the action log cannot. The budget goes to the
    longest spans first; a cut under :data:`MIN_CUT_SECONDS` is dropped.
    """
    times = sorted(protected_times)
    protected = 0
    candidates: list[FrozenSpan] = []
    for span in spans:
        if span.seconds - hold < MIN_CUT_SECONDS:
            continue
        lo, hi = span.start + hold, span.end
        if any(lo <= t < hi for t in times):
            protected += 1
            continue
        candidates.append(span)
    remaining = max(0.0, budget)
    cuts: list[tuple[float, float]] = []
    for span in sorted(candidates, key=lambda s: (-s.seconds, s.start)):
        if remaining < MIN_CUT_SECONDS:
            break
        cut = min(span.seconds - hold, remaining)
        if cut < MIN_CUT_SECONDS:
            continue
        lo = span.start + hold
        cuts.append((round(lo, 3), round(lo + cut, 3)))
        remaining -= cut
    cuts.sort()
    return cuts, protected


def kept_segments(cuts: Sequence[tuple[float, float]], usable: float) -> list[tuple[float, float]]:
    """The complement of ``cuts`` over ``[0, usable]`` — the segments to concatenate."""
    out: list[tuple[float, float]] = []
    pos = 0.0
    for lo, hi in cuts:
        if lo > pos:
            out.append((pos, min(lo, usable)))
        pos = max(pos, hi)
    if pos < usable:
        out.append((pos, usable))
    return [(a, b) for a, b in out if b - a > 1e-3]


def remap_time(t: float, cuts: Sequence[tuple[float, float]]) -> float:
    """``t`` on the original clip axis → the trimmed axis.

    Time inside a removed interval collapses onto the cut point.
    """
    shift = 0.0
    for lo, hi in cuts:
        if t >= hi:
            shift += hi - lo
        elif t > lo:
            return lo - shift
        else:
            break
    return t - shift


def frozen_after(spans: Sequence[FrozenSpan], cuts: Sequence[tuple[float, float]]) -> float:
    """Frozen seconds that remain once ``cuts`` are applied (cuts lie inside spans)."""
    removed = sum(hi - lo for lo, hi in cuts)
    return max(0.0, sum(s.seconds for s in spans) - removed)


def remap_track(
    events: Sequence[Mapping[str, object]] | None, cuts: Sequence[tuple[float, float]]
) -> list[dict[str, object]] | None:
    """Remap every event's numeric ``t``; non-numeric ``t`` passes through untouched."""
    if events is None:
        return None
    out: list[dict[str, object]] = []
    for event in events:
        row = dict(event)
        t = row.get("t")
        if isinstance(t, (int, float)) and not isinstance(t, bool):
            row["t"] = round(remap_time(float(t), cuts), 3)
        out.append(row)
    return out


def track_times(*tracks: Sequence[Mapping[str, object]] | None) -> list[float]:
    """Every numeric ``t`` across ``tracks`` — the protected action times."""
    times: list[float] = []
    for track in tracks:
        for event in track or []:
            t = event.get("t")
            if isinstance(t, (int, float)) and not isinstance(t, bool):
                times.append(float(t))
    return times


# ── ffmpeg ────────────────────────────────────────────────────────────────────
def _tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise IdleTrimError(f"{name} is not installed")
    return path


async def _run(cmd: list[str], timeout_s: float) -> tuple[bytes, bytes]:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except TimeoutError as exc:
        raise IdleTrimError(f"{Path(cmd[0]).name} exceeded {timeout_s:.0f}s") from exc
    finally:
        # Timeout AND cancellation: the child must never outlive its coroutine.
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
    if proc.returncode != 0:
        tail = err.decode(errors="replace").strip()[-300:]
        raise IdleTrimError(f"{Path(cmd[0]).name} exited {proc.returncode}: {tail}")
    return out, err


def _bounded(default_s: float, timeout_s: float | None) -> float:
    """The per-call ceiling, tightened to ``timeout_s`` (the caller's remaining budget)."""
    if timeout_s is None:
        return default_s
    if timeout_s <= 0:
        raise IdleTrimError("idle-trim time budget exhausted")
    return min(default_s, timeout_s)


async def probe_seconds(path: Path, *, timeout_s: float | None = None) -> float | None:
    """Container duration in seconds, or ``None`` when ffprobe cannot tell.

    ``timeout_s`` (the caller's remaining budget) tightens :data:`PROBE_TIMEOUT_S`.
    """
    out, _ = await _run(
        [
            _tool("ffprobe"),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        _bounded(PROBE_TIMEOUT_S, timeout_s),
    )
    try:
        value = float(out.decode().strip())
    except ValueError:
        return None
    return value if math.isfinite(value) and value > 0 else None


async def detect_frozen_spans(
    path: Path,
    *,
    usable: float,
    noise_db: float = FREEZE_NOISE_DB,
    min_seconds: float = FREEZE_MIN_SECONDS,
    timeout_s: float | None = None,
) -> list[FrozenSpan]:
    """freezedetect over the first ``usable`` seconds of ``path``.

    ``timeout_s`` (the caller's remaining budget) tightens :data:`DETECT_TIMEOUT_S`.
    """
    _, err = await _run(
        [
            _tool("ffmpeg"),
            "-hide_banner",
            "-nostats",
            "-t",
            f"{usable:.3f}",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-vf",
            f"freezedetect=n={noise_db:g}dB:d={min_seconds:g}",
            "-an",
            "-f",
            "null",
            "-",
        ],
        _bounded(DETECT_TIMEOUT_S, timeout_s),
    )
    return parse_freezedetect(err.decode(errors="replace"), duration=usable)


def concat_filter(segments: Sequence[tuple[float, float]]) -> str:
    """``filter_complex`` that trims ``segments`` out of input 0 and concatenates them."""
    parts = [
        f"[0:v]trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS[s{i}]"
        for i, (a, b) in enumerate(segments)
    ]
    labels = "".join(f"[s{i}]" for i in range(len(segments)))
    parts.append(f"{labels}concat=n={len(segments)}:v=1:a=0[out]")
    return ";".join(parts)


async def cut_segments(
    src: Path,
    dst: Path,
    segments: Sequence[tuple[float, float]],
    *,
    timeout_s: float | None = None,
) -> None:
    """Re-encode ``src`` to ``dst`` (H.264 MP4) keeping only ``segments``.

    ``timeout_s`` (the caller's remaining budget) tightens :data:`CUT_TIMEOUT_S`.
    """
    if not segments:
        raise IdleTrimError("no segments to keep")
    await _run(
        [
            _tool("ffmpeg"),
            "-hide_banner",
            "-nostats",
            "-y",
            "-i",
            str(src),
            "-filter_complex",
            concat_filter(segments),
            "-map",
            "[out]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(dst),
        ],
        _bounded(CUT_TIMEOUT_S, timeout_s),
    )
