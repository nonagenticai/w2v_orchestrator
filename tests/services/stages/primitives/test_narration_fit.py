"""NarrationStageBase.fit_target_for — measured-duration enforcement.

``reconcile_timeline`` stamps an allotment per line from a words-per-minute
ESTIMATE. Narration was then synthesized once at the estimated rate and whatever
came back was accepted, leaving the compiler to clone the last frame over the
overrun (a 30 s order shipped as 83.07 s).

The synthesize response carries the REAL duration, and ``speed`` is applied
natively — but kokoro's length answers it as ``speed ** -k`` with the calibrated
``k = SPEED_EXPONENT`` (0.60, ``line_fit``), NOT ``1/speed``. So the exactly-correct
speed for allotment ``A`` after measuring ``d`` at speed ``s`` is
``s* = s * (d / A) ** (1/k)`` — one step, no iteration.

The base hook returns ``None``, which must keep every stage that does not
override it (promo / lesson / course / audiobook) byte-identical: exactly ONE
synthesize call and no ``notes`` key.
"""

from __future__ import annotations

from typing import Any, ClassVar
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from src.clients.schemas.tts import TTSSynthesis
from src.services.runs.run_context import RunContext
from src.services.stages.base import (
    ArtifactUrl,
    StageInput,
    StageOutput,
    StageOutputWithNotes,
)
from src.services.stages.generic.line_fit import SPEED_EXPONENT
from src.services.stages.primitives.narration import NarrationStageBase
from src.services.stages.shared.time_budget import VO_RATE_DRAG_FLOOR

# Natural length of the fixture line at speed 1.0. Every mocked synthesize returns
# it under kokoro's CALIBRATED response — ``natural * speed ** -SPEED_EXPONENT`` —
# the model the correction arithmetic is built on, so a test that asserts a
# resulting duration is asserting the formula and not a hardcoded number.
_NATURAL_SECONDS = 10.0
_K = SPEED_EXPONENT


def _kokoro_seconds(natural: float, speed: float) -> float:
    """Length of ``natural`` seconds of speech re-synthesized at ``speed``."""
    return natural * speed**-_K


class _Clip(StageOutput):
    id: str
    line_id: str
    artifact_url: ArtifactUrl
    seconds: float


class _In(StageInput):
    pass


class _Out(StageOutput):
    clips: list[_Clip]


class _OutWithNotes(StageOutputWithNotes):
    clips: list[_Clip]


class _PlainNarrateStage(NarrationStageBase[_In, _Out]):
    """A narrate stage that overrides NEITHER hook (promo / lesson / audiobook)."""

    name = "plain_narrate"
    input_model = _In
    output_model = _Out
    source_stage = "plain_gen"
    source_list_key = "lines"
    clip_model = _Clip


class _FittedNarrateStage(NarrationStageBase[_In, _OutWithNotes]):
    """demo_video's shape: an allotment per line, plus the planned rate estimate."""

    name = "fitted_narrate"
    input_model = _In
    output_model = _OutWithNotes
    source_stage = "plain_gen"
    source_list_key = "lines"
    clip_model = _Clip
    natural_seconds: ClassVar[float] = _NATURAL_SECONDS

    def fit_target_for(self, item: dict, ctx: RunContext) -> float | None:
        allotment = item.get("seconds")
        return float(allotment) if allotment else None

    def speed_for(self, item: dict, ctx: RunContext) -> float | None:
        rate = item.get("vo_rate")
        return float(rate) if rate else None


def _tts(natural: float = _NATURAL_SECONDS) -> AsyncMock:
    """A TTS client whose duration answers speed as kokoro does (``speed ** -k``)."""

    async def _synthesize(**kwargs: Any) -> TTSSynthesis:
        speed = float(kwargs.get("speed", 1.0))
        return TTSSynthesis(
            audio_bytes=b"\x01",
            sample_rate=24000,
            duration_seconds=_kokoro_seconds(natural, speed),
            provider="kokoro",
            voice=str(kwargs.get("voice") or "af_heart"),
        )

    client = AsyncMock()
    client.synthesize.side_effect = _synthesize
    return client


def _recording_tts(natural: float = _NATURAL_SECONDS) -> tuple[AsyncMock, list[float | None]]:
    """A TTS client that RECORDS every speed it was asked for.

    Returns ``(client, requested_speeds)``. The returned duration is DERIVED from
    the requested speed (``natural * speed ** -k``), so a test asserting the final
    duration is asserting that the calibrated model was used correctly — not
    merely that a call happened. Assert against the record and the resulting
    clip, never against ``call_args_list`` positions: an extra, entirely correct
    upstream call would shift every index and say nothing about behaviour.
    """
    requested: list[float | None] = []

    async def _synthesize(**kwargs: Any) -> TTSSynthesis:
        speed = kwargs.get("speed")
        requested.append(speed)
        return TTSSynthesis(
            audio_bytes=b"\x01",
            sample_rate=24000,
            duration_seconds=_kokoro_seconds(natural, float(speed if speed is not None else 1.0)),
            provider="kokoro",
            voice=str(kwargs.get("voice") or "af_heart"),
        )

    client = AsyncMock()
    client.synthesize.side_effect = _synthesize
    return client, requested


def _ctx(lines: list[dict]) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"voice": {"voice_id": "af_heart"}},
        prior_run_outputs={"plain_gen": {"lines": lines}},
    )


def _storage_patches():
    return (
        patch(
            "src.services.stages.primitives.narration.find_existing_clip",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "src.services.stages.primitives.narration.store_tts_clip",
            new=AsyncMock(side_effect=lambda **kw: f"https://minio.test/{kw['clip_id']}.wav"),
        ),
    )


async def _run(stage: NarrationStageBase, lines: list[dict]) -> Any:
    finder, storer = _storage_patches()
    with finder, storer:
        return await stage.handle(stage.input_model(), _ctx(lines))


async def test_no_override_makes_exactly_one_call_and_carries_no_notes() -> None:
    """The byte-identity guarantee for promo / lesson / course / audiobook.

    The line carries a `seconds` field that WOULD be an allotment for a stage that
    opted in — the default hook must ignore it entirely: one synthesize call, the
    measured duration accepted verbatim, and no `notes` key on the output.
    """
    client = _tts()
    stage = _PlainNarrateStage(client=client)

    out = await _run(stage, [{"id": "l1", "text": "Hello.", "seconds": 3.0}])

    assert client.synthesize.await_count == 1
    assert "speed" not in client.synthesize.call_args.kwargs
    # Accepted verbatim — NOT squeezed toward the 3.0 in the line.
    assert out.clips[0].seconds == _NATURAL_SECONDS
    assert "notes" not in out.model_dump()


async def test_within_tolerance_does_not_pay_for_a_second_synthesis() -> None:
    """A 4% miss is inside the ±5% band ⇒ accept, one call only."""
    client = _tts()
    stage = _FittedNarrateStage(client=client)

    # 10.0s measured against a 9.62s allotment => 3.95% over, inside tolerance.
    out = await _run(stage, [{"id": "l1", "text": "Hello.", "seconds": 9.62}])

    assert client.synthesize.await_count == 1
    assert out.clips[0].seconds == _NATURAL_SECONDS
    assert out.notes == []


async def test_overrun_resynthesizes_once_at_the_computed_speed() -> None:
    """s* = s * (d / A) ** (1/k), asserted numerically, including a non-1.0 first speed.

    First pass at the planned vo_rate 1.25 measures 10.0 * 1.25**-0.6 = 8.747s
    against a 7.0s allotment, so s* = 1.25 * (8.747/7.0) ** (1/0.6) = (10/7) ** (1/0.6)
    = 1.812. The old ``1/speed`` model asked 1.25 * 8.747/7.0 = 1.562, which kokoro
    delivers at 10.0 * 1.562**-0.6 = 7.65s — 9% LONG, outside the band. Ignoring `s`
    altogether would be worse still. The second call must carry 1.812.
    """
    client = _tts()
    stage = _FittedNarrateStage(client=client)

    out = await _run(stage, [{"id": "l1", "text": "Hello.", "seconds": 7.0, "vo_rate": 1.25}])

    assert client.synthesize.await_count == 2
    speeds = [c.kwargs.get("speed") for c in client.synthesize.call_args_list]
    assert speeds == [1.25, pytest.approx((10.0 / 7.0) ** (1 / _K))]
    # The corrected clip's MEASURED duration is what the clip reports.
    assert out.clips[0].seconds == pytest.approx(7.0)
    # A clean fit inside the band is not a degradation.
    assert out.notes == []


async def test_an_underrun_comes_in_short_rather_than_stretching_the_voice() -> None:
    """⭐ THIS TEST PREVIOUSLY ASSERTED THE DEFECT WAS CORRECT BEHAVIOUR.

    It was named ``test_underrun_is_stretched_onto_its_allotment``, and its docstring
    read: *"The correction is symmetric — a SHORT line is stretched, not left short ...
    10.0s of natural audio against a 12.5s allotment must come back as a 12.5s clip,
    which is only reachable by asking for speed 0.8."* It asserted
    ``out.clips[0].seconds == 12.5``, ``0.8 in requested``, and ``out.notes == []``.

    Every one of those is the owner's complaint written down as an expectation. Asking
    for 0.8 IS the 25% uniform stretch of phonemes and pauses that makes narration
    "weirdly slow down", and asserting ``notes == []`` pinned it as raising NOTHING.

    The wave-13 decision reverses the policy: the ordered duration is a target with a
    real under-delivery allowance, and the residual is the PICTURE's to hold. So the
    same fixture must now deliver a SHORT clip — at natural speed, never slowed — and
    must SAY SO.
    """
    client, requested = _recording_tts()
    stage = _FittedNarrateStage(client=client)

    out = await _run(stage, [{"id": "l1", "text": "Hello.", "seconds": 12.5}])

    # THE property, inverted: the clip does NOT fill its allotment, by design.
    # 10.0s at natural speed; the remaining 2.5s is held by the picture.
    assert out.clips[0].seconds == pytest.approx(_NATURAL_SECONDS)
    assert out.clips[0].seconds < 12.5
    # The voice was never slowed below natural, and 0.8 was never asked for.
    assert 0.8 not in requested
    assert all(s is None or s >= 1.0 for s in requested)
    assert all(s is None or 0.5 <= s <= 2.0 for s in requested)
    # Bounded work — corrected at most once, never a convergence loop.
    assert len(requested) <= 2
    # ⛔ AND IT IS NOT SILENT. `notes == []` was the assertion that made the defect
    # invisible; a short deliverable is a reportable fact.
    assert any("drag floor" in n for n in out.notes), out.notes


async def test_clamp_pins_at_the_band_edge_notes_it_and_still_succeeds() -> None:
    """A line too long for its slot at ANY legal rate: clamp, record, do NOT fail.

    10.0s against a 4.0s allotment wants s* = 2.5 ** (1/0.6) = 4.605, outside the
    0.5-2.0 band the backend accepts. The request must go out at 2.0 (which strictly
    reduces the overrun, 10.0s -> 10.0 * 2.0**-0.6 = 6.60s), the shortfall must be
    recorded on the notes channel,
    and the stage must still return an output.
    """
    client, requested = _recording_tts()
    stage = _FittedNarrateStage(client=client)

    out = await _run(stage, [{"id": "l1", "text": "Hello.", "seconds": 4.0}])

    # Nothing illegal was ever asked of the backend — the clamp bound at 2.0.
    assert all(s is None or 0.5 <= s <= 2.0 for s in requested)
    assert max(s for s in requested if s is not None) == 2.0
    # BEST ACHIEVABLE: 10.0s of audio at the 2.0 ceiling is 6.60s. Still over the
    # 4.0s allotment (the line genuinely does not fit), but strictly better than
    # the 10.0s that would otherwise have been freeze-padded over.
    assert out.clips[0].seconds == pytest.approx(_kokoro_seconds(_NATURAL_SECONDS, 2.0))
    assert 4.0 < out.clips[0].seconds < _NATURAL_SECONDS
    assert len(requested) <= 2
    # ...and the shortfall is RECORDED rather than silently shipped.
    assert len(out.notes) == 1
    note = out.notes[0]
    assert "l1" in note
    assert "4.00s" in note  # the target
    assert "6.60s" in note  # what was achieved
    assert "4.605" in note  # the speed it would have needed


async def test_already_at_the_band_edge_skips_the_pointless_second_call() -> None:
    """When the planned rate is ALREADY the clamp value, re-synthesizing at the
    same speed produces the same content-addressed clip — skip the call, keep the
    record that the line does not fit."""
    client = _tts()
    stage = _FittedNarrateStage(client=client)

    out = await _run(stage, [{"id": "l1", "text": "Hello.", "seconds": 2.0, "vo_rate": 2.0}])

    assert client.synthesize.await_count == 1
    assert out.clips[0].seconds == pytest.approx(_kokoro_seconds(_NATURAL_SECONDS, 2.0))
    assert len(out.notes) == 1
    assert "l1" in out.notes[0]


async def test_second_synthesis_failure_keeps_the_first_clip() -> None:
    """Correction must never fail a paid narrate: if the re-synthesis raises, the
    already-stored first clip is kept and the miss is noted."""
    client = _tts()
    calls: list[float | None] = []

    async def _synthesize(**kwargs: Any) -> TTSSynthesis:
        speed = kwargs.get("speed")
        calls.append(speed)
        if len(calls) > 1:
            raise RuntimeError("tts_backend 503")
        return TTSSynthesis(
            audio_bytes=b"\x01",
            sample_rate=24000,
            duration_seconds=_NATURAL_SECONDS,
            provider="kokoro",
            voice="af_heart",
        )

    client.synthesize.side_effect = _synthesize
    stage = _FittedNarrateStage(client=client)

    out = await _run(stage, [{"id": "l1", "text": "Hello.", "seconds": 8.0}])

    assert calls == [None, pytest.approx(1.25 ** (1 / _K))]
    # The FIRST clip survives — it is synthesized, stored and paid for.
    assert out.clips[0].seconds == _NATURAL_SECONDS
    assert len(out.notes) == 1
    assert "failed" in out.notes[0]


# ══ pause_seconds_for — SSML break silence is FIXED, not scaled by speed ═════
#
# A directed line's SSML `<break>` is rendered by the backend as a literal
# `np.zeros(...)` — never scaled by `speed`. `_fit_speed` used to fold that
# fixed silence into the same measured/target ratio as the speech that DOES
# scale, so a beat whose pause is a non-trivial fraction of its allotment
# mis-attributed the break and missed target on the SECOND (corrected)
# synthesis — a miss its undirected neighbour, same words and same target,
# never had.

_D1 = 6.0  # natural speech duration at speed 1.0 — IDENTICAL for both lines below
_TARGET = 5.0
_PAUSE = 1.0


class _PauseAwareFittedNarrateStage(_FittedNarrateStage):
    """Mirrors demo_video_narrate.pause_seconds_for: a per-item `pause_seconds`
    field stands in for a matched voice_director directive's `pause_before_ms`,
    already converted to seconds — exactly what the real hook returns."""

    def pause_seconds_for(self, item: dict, ctx: RunContext) -> float:
        return float(item.get("pause_seconds") or 0.0)


def _tts_with_per_text_pause(pauses: dict[str, float], natural: float = _D1) -> AsyncMock:
    """A TTS client whose duration is `natural * speed**-k + pauses[text]` — the
    speech portion scales with `speed`, the pause does not, exactly mirroring
    how `tts_backend` renders a `<break>` as fixed silence outside `speed`'s
    reach."""

    async def _synthesize(**kwargs: Any) -> TTSSynthesis:
        speed = float(kwargs.get("speed") or 1.0)
        text = str(kwargs.get("text") or "")
        pause = pauses.get(text, 0.0)
        return TTSSynthesis(
            audio_bytes=b"\x01",
            sample_rate=24000,
            duration_seconds=_kokoro_seconds(natural, speed) + pause,
            provider="kokoro",
            voice=str(kwargs.get("voice") or "af_heart"),
        )

    client = AsyncMock()
    client.synthesize.side_effect = _synthesize
    return client


async def test_a_directed_pause_is_no_longer_mis_attributed_to_the_speech() -> None:
    """Two lines, same natural speech length (`_D1`) and the same `_TARGET`
    allotment: one carries a 1.0s SSML pause via `pause_seconds_for`, the other
    none.

    Both must land within the pipeline's own ±5% `_FIT_TOLERANCE_RATIO` band of
    `_TARGET` after correction. Before the pause subtraction, the whole 7.0s clip
    went into the ratio and the paused line missed its band while its undirected
    neighbour did not. Now both land on 5.0 exactly:
    `s* = ((7.0-1.0)/(5.0-1.0)) ** (1/k)` ⇒ `6.0 * s**-k + 1.0 = 6.0/1.5 + 1.0 = 5.0`.
    """
    plain_text = "Six seconds of plain narration text right here."
    paused_text = "Six seconds of paused narration text right here."
    client = _tts_with_per_text_pause({plain_text: 0.0, paused_text: _PAUSE})
    stage = _PauseAwareFittedNarrateStage(client=client)

    out = await _run(
        stage,
        [
            {"id": "plain", "text": plain_text, "seconds": _TARGET},
            {
                "id": "paused",
                "text": paused_text,
                "seconds": _TARGET,
                "pause_seconds": _PAUSE,
            },
        ],
    )

    by_id = {c.line_id: c for c in out.clips}
    plain_error = abs(by_id["plain"].seconds - _TARGET) / _TARGET
    paused_error = abs(by_id["paused"].seconds - _TARGET) / _TARGET
    assert plain_error <= 0.05, by_id["plain"].seconds
    assert paused_error <= 0.05, by_id["paused"].seconds


async def test_provenance_is_forwarded_to_the_clip_sidecar() -> None:
    """voice / provider / speed / word_count reach store_tts_clip, so the sidecar
    can answer "how fast does this voice actually talk?" from history."""
    client = _tts()
    stage = _FittedNarrateStage(client=client)
    storer = AsyncMock(side_effect=lambda **kw: f"https://minio.test/{kw['clip_id']}.wav")
    finder = patch(
        "src.services.stages.primitives.narration.find_existing_clip",
        new=AsyncMock(return_value=None),
    )
    with (
        finder,
        patch("src.services.stages.primitives.narration.store_tts_clip", new=storer),
    ):
        await stage.handle(
            stage.input_model(),
            _ctx([{"id": "l1", "text": "One two three four."}]),
        )

    kwargs = storer.call_args.kwargs
    assert kwargs["voice"] == "af_heart"
    assert kwargs["provider"] == "kokoro"
    assert kwargs["speed"] is None
    assert kwargs["word_count"] == 4


# ══ tts_backend's OWN pause_seconds — the silence nothing upstream can see ═══
#
# A `<break>` the orchestrator authored is not the only fixed silence in a clip.
# kokoro splices an inter-phrase pacing gap between every sentence/clause piece
# whether or not the line carries a directive. Three clips delivered by run
# 01a09e7d-c6da-7450-a7cf-91c43a4aecf3 measured 42.0% / 44.1% / 48.8% silence
# while every one of them carried `pause_before_ms=0` — so `pause_seconds_for`
# returned 0.0 and the authored-break subtraction subtracted nothing, leaving
# `_fit_speed` to divide ~45% unscalable silence into the same ratio as the
# speech. The solver then over-drove the SPEECH to make up the difference: those
# clips delivered 237.5 / 293.9 / 250.2 wpm over their speech-only portion
# against a ~150 wpm natural rate.
#
# `tts_backend` now reports the split (`SynthesizeResponse.speech_seconds` /
# `.pause_seconds`); these tests pin that the orchestrator consumes it.

_SPEECH_AT_1X = 6.0  # speech-only length at speed 1.0 — scales with `speed`
_PACING_GAP = 3.0  # kokoro's own inter-phrase silence — does NOT scale
_GAP_TARGET = 7.0


def _tts_with_service_pacing(
    *, report: bool, speech_at_1x: float = _SPEECH_AT_1X, gap: float = _PACING_GAP
) -> AsyncMock:
    """A backend whose clip is ``speech_at_1x * speed ** -k + gap``.

    The gap is FIXED — it is silence spliced in as literal zeros, so `speed`
    never touches it. `report` chooses whether this backend is one that has
    shipped the `speech_seconds`/`pause_seconds` split (True) or one that
    predates it and omits both (False, i.e. they arrive as None).
    """

    async def _synthesize(**kwargs: Any) -> TTSSynthesis:
        speed = float(kwargs.get("speed") or 1.0)
        speech = _kokoro_seconds(speech_at_1x, speed)
        return TTSSynthesis(
            audio_bytes=b"\x01",
            sample_rate=24000,
            duration_seconds=speech + gap,
            provider="kokoro",
            voice=str(kwargs.get("voice") or "af_heart"),
            speech_seconds=speech if report else None,
            pause_seconds=gap if report else None,
        )

    client = AsyncMock()
    client.synthesize.side_effect = _synthesize
    return client


async def test_service_reported_pacing_silence_is_excluded_from_the_fit_ratio() -> None:
    """A clip that is 33% kokoro pacing silence lands ON its allotment.

    Physics: 6.0s of speech at 1.0x plus a fixed 3.0s gap ⇒ a 9.0s clip for a
    7.0s allotment.

    Without the split the ratio sees only the total: `s* = (9.0/7.0) ** (1/k)`, which
    delivers `6.0 * (7/9) + 3.0 = 7.667s` — 9.5% OVER target, outside the ±5%
    `_FIT_TOLERANCE_RATIO` band (the speech was compressed to absorb silence that
    never moved).

    With it the 3.0s is subtracted from BOTH sides: `s* = (6.0/4.0) ** (1/k) = 1.966`,
    delivering `6.0 * s**-k + 3.0 = 4.0 + 3.0 = 7.0s` — exactly the allotment.
    """
    client = _tts_with_service_pacing(report=True)
    stage = _FittedNarrateStage(client=client)

    out = await _run(
        stage, [{"id": "paced", "text": "Some narration text.", "seconds": _GAP_TARGET}]
    )

    delivered = out.clips[0].seconds
    assert abs(delivered - _GAP_TARGET) / _GAP_TARGET <= 0.05, delivered
    assert delivered == pytest.approx(7.0)


async def test_a_backend_that_omits_the_split_keeps_the_old_behaviour() -> None:
    """The SAME physics with a backend that does not report the split.

    This is the compatibility arm: `None` must mean "unknown — use the old
    path", never "zero silence". So this line is expected to MISS, exactly as it
    did before the field existed, rather than being silently corrected by a
    value the backend never sent. It also proves the previous test passes
    BECAUSE of the reported fields and not for some unrelated reason — the only
    difference between the two is whether the backend populated them.
    """
    client = _tts_with_service_pacing(report=False)
    stage = _FittedNarrateStage(client=client)

    out = await _run(
        stage, [{"id": "paced", "text": "Some narration text.", "seconds": _GAP_TARGET}]
    )

    delivered = out.clips[0].seconds
    # 6.0 * (7.0/9.0) + 3.0 — the un-subtracted ratio, i.e. today's behaviour.
    assert delivered == pytest.approx(7.6667, abs=1e-3)
    assert abs(delivered - _GAP_TARGET) / _GAP_TARGET > 0.05, delivered


async def test_an_over_threshold_correction_notes_the_rate_measured_over_speech() -> None:
    """The honest note states wpm over the SPEECH portion, not total duration.

    That distinction is the whole finding: a clip that is ~45% silence reads as
    an unremarkable rate over its total duration and as a 1.6x-2.0x overrun over
    its speech. Here 30 words are delivered over 3.0s of speech inside a 9.0s
    clip — 600.0 wpm, where the total-duration reading would say 200.0.

    Advisory only: the clip is still kept and the stage still succeeds. Asserting
    the NUMBER (not merely that a note exists) is what makes this fail if the
    note is ever computed from `seconds` again.
    """
    # 3.0s of speech at 1.0x + 6.0s of pacing = a 9.0s clip for a 4.0s allotment.
    # fixed=6.0 exceeds the 4.0 target, so the guard sends the plain ratio:
    # s* = (9.0/4.0) ** (1/k) = 3.86 > _RATE_DEFECT_SPEED_THRESHOLD.
    client = _tts_with_service_pacing(report=True, speech_at_1x=3.0, gap=6.0)
    stage = _FittedNarrateStage(client=client)
    text = " ".join(f"word{i}" for i in range(30))

    out = await _run(stage, [{"id": "fast", "text": text, "seconds": 4.0}])

    rate_notes = [n for n in out.notes if "wpm" in n]
    assert len(rate_notes) == 1, out.notes
    note = rate_notes[0]
    assert "600.0 wpm" in note, note
    assert "over 3.00s of speech" in note, note
    assert "in a 9.00s clip" in note, note


# ---------------------------------------------------------------------------
# A/V quality wave 10, item 2c — the SLOW direction had no structured signal.
#
# `_RATE_DEFECT_SPEED_THRESHOLD` (1.3) fires only when a line is RUSHED to fit.
# A line DRAGGED to the 0.80 natural floor — the owner's "the narration weirdly
# slows down", and simultaneously why the beat ships short — raised prose on
# `notes` and nothing a query could bind to. These pin the structured entry.
# ---------------------------------------------------------------------------


async def test_a_clamp_at_the_low_floor_is_recorded_as_a_degradation() -> None:
    """10.0s of audio for a 20.0s allotment wants speed 0.5**(1/k) = 0.315 — below the drag floor.

    The line is kept at natural speed (never slowed — see ``_slow_floor``), the clip
    delivers 10.00s, and 10.00s of the allotment is covered by neither speech nor the
    stretch. That is a
    defect in the artifact, so it belongs on the STRUCTURED channel with the uncovered
    seconds, not only in prose.

    ⚠️ Was "clamp binds at 0.80 / delivers 12.5s / 7.5s uncovered". The uncovered
    figure GREW because the voice is no longer permitted to absorb 2.2s of it by
    dragging — the hole is the same size, it is now counted instead of hidden.

    FAILS BEFORE: `out.degradations` was empty — `_fit_to_target` had no degradations
    sink at all and only the fast side (>1.3) raised anything beyond a note.
    """
    client, requested = _recording_tts()
    stage = _FittedNarrateStage(client=client)

    out = await _run(stage, [{"id": "slow-1", "text": "Hello.", "seconds": 20.0}])

    # Kept as synthesized at natural speed: no second, slower synthesis was paid for.
    assert requested == [None]
    assert out.clips[0].seconds == pytest.approx(_NATURAL_SECONDS)

    # Filter by code: this fixture's stub premaster also records `audio_unnormalized`,
    # which is a different, unrelated fact about the same clip.
    drags = [d for d in out.degradations if d["code"] == "narration_rate_floor_clamped"]
    assert len(drags) == 1, out.degradations
    detail = drags[0]["detail"]
    assert "line slow-1" in detail
    # The uncovered seconds are the number the note was missing: 20.00 - 10.00.
    assert "10.00s of the allotment" in detail, detail
    assert f"speed {0.5 ** (1 / _K):.3f}" in detail, detail


async def test_a_line_already_pinned_at_the_floor_still_records_the_drag() -> None:
    """The no-second-synthesis path degrades too — it is the SAME defect, cheaper.

    A line planned at the drag floor measures 12.5s against a 20.0s allotment: the
    exact speed is 0.606, the correction is already pinned at the floor, so no second
    synthesize is paid for. The clip is just as short, so the entry must still be
    recorded.

    ⚠️ The fixture's ``vo_rate`` moved 0.80 -> ``VO_RATE_DRAG_FLOOR``. It has to: the
    test's whole premise is "already pinned at the EDGE", and 0.80 is no longer the
    edge. Left at 0.80 this test silently stopped testing the pinned path — a second
    synthesize was paid for and the `await_count == 1` assertion caught it. That is the
    test doing its job; it is not a regression.

    FAILS BEFORE: this branch returned early with a note and nothing else.
    """
    client = _tts()
    stage = _FittedNarrateStage(client=client)

    out = await _run(
        stage,
        [{"id": "slow-2", "text": "Hello.", "seconds": 20.0, "vo_rate": VO_RATE_DRAG_FLOOR}],
    )

    assert client.synthesize.await_count == 1, "pinned at the edge ⇒ no second call"
    drags = [d for d in out.degradations if d["code"] == "narration_rate_floor_clamped"]
    assert len(drags) == 1, out.degradations
    # DERIVED from the clip actually delivered, not a pasted literal: the uncovered
    # seconds are the allotment minus what the kept clip occupies. A literal here would
    # silently stop matching the moment the floor or the stub's duration moves, which is
    # exactly what happened to the "7.50s" this replaces.
    uncovered = 20.0 - out.clips[0].seconds
    assert uncovered > 0.0, "the point of the fixture is that the clip is SHORT"
    assert f"{uncovered:.2f}s of the allotment" in drags[0]["detail"], drags[0]["detail"]


async def test_a_fast_clamp_does_not_raise_the_slow_code() -> None:
    """The control. A line RUSHED to the 2.0 ceiling is a different defect.

    Without this, a degradation emitted on every clamp in either direction would pass the
    two tests above while saying nothing about the slow direction at all.
    """
    client, _requested = _recording_tts()
    stage = _FittedNarrateStage(client=client)

    out = await _run(stage, [{"id": "fast-1", "text": "Hello.", "seconds": 4.0}])

    assert any("clamped to 2.00" in n for n in out.notes), out.notes
    assert [d for d in out.degradations if d["code"] == "narration_rate_floor_clamped"] == []


# ---------------------------------------------------------------------------
# Never slower than the planned rate: a SHORT clip is kept, the picture holds.
# ---------------------------------------------------------------------------


async def test_a_short_clip_is_never_refit_below_its_planned_rate() -> None:
    """10.0s at the planned 1.0 against a 12.5s allotment: the line FITS. It is kept
    as synthesized — one call, speed 1.0 — never re-synthesized slower."""
    client, requested = _recording_tts()
    stage = _FittedNarrateStage(client=client)

    out = await _run(stage, [{"id": "l1", "text": "Hello.", "seconds": 12.5, "vo_rate": 1.0}])

    assert requested == [1.0]
    assert out.clips[0].seconds == pytest.approx(_NATURAL_SECONDS)


async def test_a_short_clip_planned_fast_relaxes_to_natural_not_below() -> None:
    """Planned at 1.25 (8.0s) against a 12.5s allotment wants 0.8: the correction
    may relax the voice back to natural, never past it."""
    client, requested = _recording_tts()
    stage = _FittedNarrateStage(client=client)

    out = await _run(stage, [{"id": "l1", "text": "Hello.", "seconds": 12.5, "vo_rate": 1.25}])

    assert requested == [1.25, 1.0]
    assert out.clips[0].seconds == pytest.approx(_NATURAL_SECONDS)


async def test_a_long_clip_is_still_sped_up() -> None:
    """Control: the overrun side is untouched — 10.0s into 8.0s refits at 1.25**(1/k)."""
    client, requested = _recording_tts()
    stage = _FittedNarrateStage(client=client)

    out = await _run(stage, [{"id": "l1", "text": "Hello.", "seconds": 8.0, "vo_rate": 1.0}])

    assert requested == [1.0, pytest.approx(1.25 ** (1 / _K))]
    assert out.clips[0].seconds == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# ONE model of the voice: `_fit_speed` uses line_fit's calibrated k, not 1/speed.
# ---------------------------------------------------------------------------
