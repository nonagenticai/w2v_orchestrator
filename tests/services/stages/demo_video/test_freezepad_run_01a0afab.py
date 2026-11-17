"""Run ``01a0afab-24a3-7745-b8b9-9625966f31ca`` (showcase, 30s order), replayed on its real numbers.

Stored ``reconcile_timeline`` output: ``resolved_wpm = 111.71826309133743`` (basis
``script_sample``), and three footage beats:

    beat    words  clip    freeze_pad_seconds     vo_rate
    clip-1  18     9.64    0.027174999999999727   1.0
    clip-4  23     10.28   1.5                    1.048599438785135
    clip-5  17     8.84    0.29010972222222264    1.0

Stored ``demo_video_narrate`` output: clip-4 kept at ``speed 0.97``, ``9.874958s``.

Two questions this file answers with arithmetic, not assumption:

* **Is clip-4's 1.5 a real pad or the ceiling constant leaking through?** It is the
  ceiling BINDING on a real computed gap: 23 words / 111.718 wpm = 12.352s estimated,
  minus the 10.28s clip = 2.072s, capped to 1.5. The two neighbours (0.027, 0.290) come
  out of the SAME formula bit-for-bit, which a leaked default could not produce.
* **Was the voice sped up?** Only in the PLAN. ``vo_rate`` 1.0486 is the first-pass
  speed chosen from that estimate; narrate measured the line, re-synthesized it, and
  KEPT the 0.97 clip. The delivered voice is slower than natural.

The defect that was real: the freeze-pad advisory stated the ESTIMATE as a measured
overrun ("carry narration that overruns their recorded clip"), with no numbers, while
the same run's narrate output measured the line SHORTER than its clip.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from src.clients.schemas.tts import TTSSynthesis
from src.services.runs.run_context import RunContext
from src.services.stages.demo_video.narrate import DemoVideoNarrateStage
from src.services.stages.demo_video.reconcile_timeline import _TimingConfig
from src.services.stages.shared.time_budget import (
    DEFAULT_FREEZE_PAD_CEILING,
    VO_RATE_DRAG_FLOOR,
    TimeDemand,
)
from src.services.stages.shared.time_budget import reconcile_timeline as solve_timeline

_RUN_WPM = 111.71826309133743
_TARGET = 30.0

# (id, words, clip seconds, stored freeze_pad_seconds, stored vo_rate)
_BEATS = [
    ("clip-1", 18, 9.64, 0.027174999999999727, 1.0),
    ("clip-4", 23, 10.28, 1.5, 1.048599438785135),
    ("clip-5", 17, 8.84, 0.29010972222222264, 1.0),
]


def _footage(beat_id: str, words: int, clip: float) -> TimeDemand:
    return TimeDemand(
        id=beat_id,
        kind="footage",
        min=clip,
        ideal=clip,
        max=clip,
        elasticity=0.0,
        pinned=True,
        word_count=words,
    )


def _cfg() -> _TimingConfig:
    return _TimingConfig(
        mode="auto",
        target=_TARGET,
        tolerance_pct=2.0,
        wpm_band=(0.5, 2.0),
        wpm_natural=_RUN_WPM,
        freeze_pad_ceiling_seconds=DEFAULT_FREEZE_PAD_CEILING,
        vo_rate_drag_floor=VO_RATE_DRAG_FLOOR,
        relief_mode="off",
        relief_iterations=0,
        relief_threshold=0.0,
    )


def _demands() -> list[TimeDemand]:
    return [_footage(beat_id, words, clip) for beat_id, words, clip, _, _ in _BEATS]


def test_the_solver_reproduces_every_stored_pad_and_rate_of_the_run() -> None:
    """Replay: the stored per-beat values are exactly this formula's output."""
    allotments = {
        a.id: a
        for a in solve_timeline(
            _demands(),
            target=_TARGET,
            wpm_natural=_RUN_WPM,
            freeze_pad_ceiling_seconds=DEFAULT_FREEZE_PAD_CEILING,
        )
    }
    for beat_id, _, clip, pad, rate in _BEATS:
        assert allotments[beat_id].seconds == clip
        assert allotments[beat_id].freeze_pad_seconds == pytest.approx(pad, abs=1e-9), beat_id
        assert allotments[beat_id].vo_rate == pytest.approx(rate, abs=1e-9), beat_id


def test_clip4_pad_of_1_5_is_the_ceiling_binding_on_a_real_2_07s_gap() -> None:
    """1.5 lands on ``DEFAULT_FREEZE_PAD_CEILING`` because the gap EXCEEDS it — not
    because a default leaked. The unclamped gap is 2.072s; the neighbours prove the
    same formula yields non-ceiling values when the gap is small."""
    estimated = 23 / _RUN_WPM * 60.0
    assert estimated == pytest.approx(12.352, abs=1e-3)
    assert estimated - 10.28 == pytest.approx(2.072, abs=1e-3)
    assert estimated - 10.28 > DEFAULT_FREEZE_PAD_CEILING  # ⇒ the cap binds

    pads = {
        a.id: a.freeze_pad_seconds
        for a in solve_timeline(
            _demands(),
            target=_TARGET,
            wpm_natural=_RUN_WPM,
            freeze_pad_ceiling_seconds=DEFAULT_FREEZE_PAD_CEILING,
        )
    }
    assert pads["clip-4"] == DEFAULT_FREEZE_PAD_CEILING
    assert pads["clip-1"] < 0.1
    assert pads["clip-5"] < 0.5


def _narrate_ctx(line: dict[str, Any]) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"voice": {"voice_id": "af_heart"}},
        prior_run_outputs={"reconcile_timeline": {"lines": [line]}},
    )


async def test_clip4_plan_speedup_is_not_what_narrate_delivers() -> None:
    """Defect B, measured: NOT a delivered defect.

    Natural (speed 1.0) duration of clip-4's real audio = 9.874958 x 0.97 ≈ 9.579s. The
    first pass synthesizes at the plan's 1.0486; the measured clip misses the 11.78s
    fit window by >5%. On the live run narrate re-synthesized at the 0.97 drag floor and
    KEPT that clip — slower than natural, 9.87s: the owner's "narration slows down".
    Now the correction relaxes the voice to natural (1.0) and no further; the picture
    holds the remainder.
    """
    natural = 9.874958333333334 * 0.97
    requested: list[float | None] = []

    async def _synthesize(**kwargs: Any) -> TTSSynthesis:
        speed = kwargs.get("speed")
        requested.append(speed)
        return TTSSynthesis(
            audio_bytes=b"\x01",
            sample_rate=24000,
            duration_seconds=natural / float(speed if speed is not None else 1.0),
            provider="kokoro",
            voice="af_heart",
        )

    client = AsyncMock()
    client.synthesize.side_effect = _synthesize
    line = {
        "id": "clip-4",
        "text": " ".join(["word"] * 23),
        "seconds": 10.28,
        "freeze_pad_seconds": 1.5,
        "hold_seconds": 0.0,
        "vo_rate": 1.048599438785135,
    }
    stage = DemoVideoNarrateStage(client=client)
    with (
        patch(
            "src.services.stages.primitives.narration.find_existing_clip",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "src.services.stages.primitives.narration.store_tts_clip",
            new=AsyncMock(side_effect=lambda **kw: f"https://minio.test/{kw['clip_id']}.wav"),
        ),
    ):
        out = await stage.handle(stage.input_model(), _narrate_ctx(line))

    assert requested == [pytest.approx(1.048599438785135), pytest.approx(1.0)]
    kept = out.clips[0]
    assert kept.speed == pytest.approx(1.0)  # natural — never slower
    assert (kept.speed or 0.0) > VO_RATE_DRAG_FLOOR
    assert kept.seconds == pytest.approx(natural)
    assert kept.seconds < 10.28  # still shorter than the footage; the picture holds it
