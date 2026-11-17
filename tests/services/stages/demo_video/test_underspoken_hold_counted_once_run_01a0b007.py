"""Run ``01a0b007-9c2c-716a-883f-beefd14f4fd5`` (demo_video, 30s order), replayed on its real numbers.

Stored ``reconcile_timeline`` line clip-5: footage beat, ``seconds = min = max = 9.56``,
16 words, ``vo_rate 0.97``, ``freeze_pad_seconds 0.0``, ``hold_seconds 1.5``.

Stored ``demo_video_narrate`` clip-5: kept at ``speed 1.468673676012461``, ``7.937958s``,
with the note "needs speed 1.469 (over 1.3) to fit its 8.06s allotment; its 16 words
already deliver 84.7 wpm over 11.34s of speech in a 11.91s clip (0.57s subtracted as
fixed silence)".

Stored ``demo_video_compile`` shot_timeline clip-5: ``26.233 -> 37.300`` = 11.067s.

Two defects, one hold:

* **Counted twice.** The hold is a slice OF the allotment (``time_budget.underspoken_hold``;
  narrate subtracts it from the voice's window). ``shot_assembly`` then ADDED it on top of
  the 9.56s footage ⇒ 11.06s shot, a dead frame past the footage.
* **A plan estimate overrode a measurement.** The line was planned underspoken but
  MEASURED long (11.91s at 0.97 against a 9.56s beat). Fitting it to ``9.56 - 1.5`` sped
  it to 1.469 purely to open a silent hold; the full window needed 1.224 (both under the
  ``1/speed`` model narrate then used; under the calibrated ``speed ** -k`` the window needs
  ~1.428 and the hold would have asked ~1.936).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from src.clients.schemas.tts import TTSSynthesis
from src.services.runs.run_context import RunContext
from src.services.stages.demo_video.narrate import DemoVideoNarrateStage
from src.services.stages.generic.line_fit import SPEED_EXPONENT
from src.services.stages.shot_assembly import (
    UNDERSPOKEN_HOLD_KEY,
    assemble_shots,
    drain_fabricated_hold_notes,
)

_FOOTAGE = 9.56
_HOLD = 1.5
_DELIVERED = 7.9379583333333334
_DELIVERED_SPEED = 1.468673676012461
_ONE_FRAME = 1 / 30
# The real clip's split, from the stored narrate note: 11.91s clip at speed 0.97 of which
# 0.57s is the backend's fixed silence ⇒ speech at speed 1.0 = 11.34 x 0.97 ** k under
# kokoro's calibrated response ``speech(s) = speech(1) * s ** -k`` (line_fit).
_K = SPEED_EXPONENT
_FIRST_PASS_SECONDS = 11.91
_FIXED_SILENCE = 0.57
_NATURAL_SPEECH = (_FIRST_PASS_SECONDS - _FIXED_SILENCE) * 0.97**_K


def _speech_at(speed: float, natural: float = _NATURAL_SPEECH) -> float:
    return natural * speed**-_K


def _clip5_line() -> dict[str, Any]:
    return {
        "id": "clip-5",
        "text": " ".join(["word"] * 16),
        "seconds": _FOOTAGE,
        "min_seconds": _FOOTAGE,
        "max_seconds": _FOOTAGE,
        "vo_rate": 0.97,
        "freeze_pad_seconds": 0.0,
        "hold_seconds": _HOLD,
    }


def _assemble(narration: float, *, bound: tuple[float, float], hold: float) -> dict[str, Any]:
    shots = assemble_shots(
        storyboard_shots=[{"id": "s-clip-5", "line_id": "clip-5", "seconds": _FOOTAGE}],
        images=[],
        narration_clips=[
            {"line_id": "clip-5", "artifact_url": "http://a/clip.wav", "seconds": narration}
        ],
        seconds_by_line={"clip-5": _FOOTAGE},
        min_max_by_line={"clip-5": bound},
        hold_seconds_by_line={"clip-5": hold},
    )
    return shots[0]


# ── shot_assembly: the hold is counted ONCE ────────────────────────────────────────────


def test_replay_the_delivered_shot_is_the_footage_not_footage_plus_hold() -> None:
    shot = _assemble(_DELIVERED, bound=(_FOOTAGE, _FOOTAGE), hold=_HOLD)

    assert shot["seconds"] <= _FOOTAGE + _ONE_FRAME, (
        f"shot {shot['seconds']:.3f}s runs past its {_FOOTAGE}s footage — the hold was "
        "added on top of a duration that already contains it (live: 11.067s)"
    )
    assert shot["seconds"] == pytest.approx(_FOOTAGE)
    # The footage already covers speech end + hold (7.938 + 1.5 = 9.438 < 9.56): no frame
    # was held, so no hold may be reported as applied.
    assert shot.get(UNDERSPOKEN_HOLD_KEY) is None
    assert drain_fabricated_hold_notes([shot]) == []


def test_a_voice_fit_to_exactly_allotment_minus_hold_lands_on_the_allotment() -> None:
    """Narrate's half of the contract: voice = 9.56 - 1.5 = 8.06s. The picture must end
    at the allotment, not 1.5s past it."""
    shot = _assemble(_FOOTAGE - _HOLD, bound=(_FOOTAGE, _FOOTAGE), hold=_HOLD)
    assert shot["seconds"] == pytest.approx(_FOOTAGE)


def test_the_hold_is_capped_at_the_allotment_it_is_a_slice_of() -> None:
    """A non-footage beat whose floor sits below its allotment: the held picture ends at
    speech end + hold, and never past the planned allotment."""
    shots = assemble_shots(
        storyboard_shots=[{"id": "s-b", "line_id": "b", "seconds": 5.0}],
        images=[],
        narration_clips=[{"line_id": "b", "artifact_url": "http://a/b.wav", "seconds": 4.2}],
        seconds_by_line={"b": 5.0},
        min_max_by_line={"b": (3.0, 6.0)},
        hold_seconds_by_line={"b": 1.5},
    )
    assert shots[0]["seconds"] == pytest.approx(5.0)  # min(4.2 + 1.5, 5.0)
    assert shots[0][UNDERSPOKEN_HOLD_KEY] == pytest.approx(0.8)


def test_a_line_longer_than_its_footage_still_extends_the_shot() -> None:
    """No regression of the narrated-picture floor: narration past the clip (inside the
    freeze-pad window) still declares the shot's length."""
    shot = _assemble(10.8, bound=(_FOOTAGE, _FOOTAGE + 1.5), hold=0.0)
    assert shot["seconds"] == pytest.approx(10.8)
    assert shot["max_seconds"] == pytest.approx(_FOOTAGE + 1.5)


# ── narrate: the hold never makes a line FASTER ────────────────────────────────────────


async def _narrate(line: dict[str, Any], natural_speech: float, pause: float) -> tuple:
    requested: list[float | None] = []

    async def _synthesize(**kwargs: Any) -> TTSSynthesis:
        speed = float(kwargs.get("speed") or 1.0)
        requested.append(speed)
        speech = _speech_at(speed, natural_speech)
        return TTSSynthesis(
            audio_bytes=b"\x01",
            sample_rate=24000,
            duration_seconds=speech + pause,
            provider="kokoro",
            voice="af_heart",
            speech_seconds=speech,
            pause_seconds=pause,
        )

    client = AsyncMock()
    client.synthesize.side_effect = _synthesize
    stage = DemoVideoNarrateStage(client=client)
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"voice": {"voice_id": "af_heart"}},
        prior_run_outputs={"reconcile_timeline": {"lines": [line]}},
    )
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
        out = await stage.handle(stage.input_model(), ctx)
    return requested, out.clips[0], out


async def test_replay_the_first_pass_matches_the_live_clip() -> None:
    """Guard the replay itself: the mock reproduces the live 11.91s first pass."""
    requested, _, _ = await _narrate(_clip5_line(), _NATURAL_SPEECH, _FIXED_SILENCE)
    assert requested[0] == pytest.approx(0.97)
    first_pass = _FIXED_SILENCE + _speech_at(float(requested[0] or 1.0))
    assert first_pass == pytest.approx(_FIRST_PASS_SECONDS)
    # And the live 1.469 is exactly what fitting to 9.56 - 1.5 produced under the
    # ``1/speed`` model narrate used at the time — the defect.
    live = (_FIRST_PASS_SECONDS - _FIXED_SILENCE) * 0.97 / (_FOOTAGE - _HOLD - _FIXED_SILENCE)
    assert live == pytest.approx(_DELIVERED_SPEED, abs=1e-3)


async def test_replay_a_measured_long_line_is_not_sped_up_to_open_the_hold() -> None:
    requested, kept, _ = await _narrate(_clip5_line(), _NATURAL_SPEECH, _FIXED_SILENCE)

    full_window = (_NATURAL_SPEECH / (_FOOTAGE - _FIXED_SILENCE)) ** (1 / _K)
    assert len(requested) == 2
    assert requested[1] == pytest.approx(full_window), (
        f"re-synthesized at {requested[1]:.3f}; the 9.56s picture needs only "
        f"{full_window:.3f} — the hold drove the rest (live: 1.469)"
    )
    # What fitting the voice into ``9.56 - 1.5`` would ask under the same calibrated
    # model (~1.936). Under ``speed ** -k`` the full window itself already needs ~1.428
    # (the ``1/speed`` model said 1.224), so the discriminator is "not the hold's
    # speed", not a fixed 1.3.
    hold_driven = (_NATURAL_SPEECH / (_FOOTAGE - _HOLD - _FIXED_SILENCE)) ** (1 / _K)
    assert requested[1] < hold_driven - 0.1
    # Fits the picture exactly: the speed-up that remains is the footage's, not the hold's.
    assert kept.seconds == pytest.approx(_FOOTAGE)
    assert kept.seconds <= _FOOTAGE + _ONE_FRAME


async def test_a_line_measured_inside_the_hold_range_is_kept_as_is() -> None:
    """8.76s at 0.97 sits in [8.06, 9.56]: it fits the picture, so it must not be sped
    up toward 8.06 (pre-fix: re-synthesized at ~1.054)."""
    requested, kept, _ = await _narrate(_clip5_line(), 8.5, 0.0)
    assert requested == [pytest.approx(0.97)]
    assert kept.seconds == pytest.approx(_speech_at(0.97, 8.5))


async def test_a_genuinely_underspoken_line_is_never_sped_up() -> None:
    """Measured SHORT of allotment - hold: the voice is not dragged and not sped up."""
    requested, kept, out = await _narrate(_clip5_line(), 6.0, 0.0)
    assert all(speed is not None and speed <= 1.0 for speed in requested), requested
    assert (kept.speed or 0.0) <= 1.0
    assert kept.seconds < _FOOTAGE - _HOLD
    # The short line is corrected against allotment - hold (8.06s), NOT accepted as its
    # own target: the drag it would need is below the floor, so it must be REPORTED —
    # a note naming the 8.06s target and a structured floor-clamp degradation. Clamping
    # only from above (`min(measured, window)`) keeps the clip silently.
    spoken_target = f"{_FOOTAGE - _HOLD:.2f}s allotment"
    assert any(spoken_target in note for note in out.notes), out.notes
    assert any(d["code"] == "narration_rate_floor_clamped" for d in out.degradations), (
        out.degradations
    )
