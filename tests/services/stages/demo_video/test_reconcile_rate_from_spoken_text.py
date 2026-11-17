"""The solver's rate must be measured on the text that will be SPOKEN, not on the draft.

Run ``01a0afab-24a3-7745-b8b9-9625966f31ca``. ``demo_video_gen`` sampled the RAW bdd
draft, whose every line embedded a quoted URL (``… application at
"https://demo.playwright.dev/todomvc"``). ``split()`` counts that URL as ONE word;
kokoro spells it out — measured on mini2 (kokoro-torch, af_heart, speed 1.0) the line is
6.406s with the URL and 3.074s without it. The draft therefore sampled at 111.7 wpm.
``script_doctor`` then rewrote every line WITHOUT the URL, and ``reconcile_timeline``
reused 111.7 as if it measured those lines: clip-4 (23 words, 10.28s clip) estimated at
12.35s ⇒ a 1.5s freeze pad ⇒ an "overruns" advisory, and narrate dragged the voice to
the floor to fill the padded window (``narration_rate_floor_clamped``).

The fake TTS below charges time per SPOKEN character with the URL's separators voiced,
so the URL costs time exactly the way it does live. Numbers are derived from the fake,
never hardcoded rates.
"""

from __future__ import annotations

import re
from typing import Any, cast
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from src.clients.schemas.bdd_playwright import BddJobResult
from src.clients.schemas.tts import TTSSynthesis
from src.clients.tts_client import TTSClient
from src.services.runs.run_context import RunContext
from src.services.stages.demo_video.gen import DemoVideoGenInput, DemoVideoGenStage
from src.services.stages.demo_video.narrate import DemoVideoNarrateStage
from src.services.stages.demo_video.reconcile_timeline import (
    ReconcileTimelineStage,
    _overpad_note,
    _TimingConfig,
)
from src.services.stages.demo_video.spine import resolve_plan_output
from src.services.stages.shared.degradation import Degradation
from src.services.stages.shared.speakable_text import speakable_text
from src.services.stages.shared.speech_rate import (
    CALIBRATION_TEXT,
    SCRIPT_SAMPLE,
    clear_measured_wpm_cache,
    spoken_text_digests,
)
from src.services.stages.shared.time_budget import (
    DEFAULT_FREEZE_PAD_CEILING,
    VO_RATE_DRAG_FLOOR,
    TimeDemand,
)
from src.services.stages.shared.time_budget import reconcile_timeline as solve_timeline

_URL = "https://demo.playwright.dev/todomvc"
_TARGET = 30.0

# The run's stored `demo_video_gen` lines (the bdd draft) and `script_doctor` rewrites,
# verbatim from `stage_outputs`.
_DRAFT = {
    "clip-1": f'First, I am on the React TodoMVC application at "{_URL}". Next, I enter '
    '"Buy groceries" into the todo input field.',
    "clip-4": f'First, I am on the React TodoMVC application at "{_URL}". Next, I enter '
    '"Send email" into the todo input field.',
    "clip-5": f'First, I am on the React TodoMVC application at "{_URL}".',
}
_DOCTORED = {
    "clip-1": 'I open the React TodoMVC demo and type "Buy groceries" into the input field.',
    "clip-4": 'Next, I enter "Send email" into the same input field.',
    "clip-5": "Finally, I return to the main TodoMVC application view.",
}
# clip-4 as it was finally narrated (reconcile's re-ask): 23 words.
_CLIP4_SPOKEN = (
    'Next, I carefully type the phrase "Send email" directly into the same input field '
    "that had been used previously for this particular task."
)
_CLIP4_CLIP_SECONDS = 10.28
# The run's stale plan for clip-4 (stored reconcile_timeline output).
_STALE_WPM = 111.71826309133743

_SECONDS_PER_SPOKEN_CHAR = 0.07
_URL_RE = re.compile(r"https?://[^\s\"]+")


def _spoken(text: str) -> str:
    """What kokoro actually voices: a URL's separators are read out as words."""

    def _voice(match: re.Match[str]) -> str:
        url = match.group(0).replace("://", " colon slash slash ")
        return url.replace("/", " slash ").replace(".", " dot ")

    return _URL_RE.sub(_voice, text)


def _natural_seconds(text: str) -> float:
    return len(_spoken(text)) * _SECONDS_PER_SPOKEN_CHAR


def _wpm_of(texts: list[str]) -> float:
    return sum(len(t.split()) for t in texts) / (sum(_natural_seconds(t) for t in texts) / 60.0)


class _SpokenCharTTS:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def synthesize(self, *, text: str, speed: float | None = None, **_: Any) -> Any:
        self.calls.append(text)
        if text == CALIBRATION_TEXT:
            duration = len(text.split()) * 60.0 / 175.2
        else:
            duration = _natural_seconds(text) / float(speed or 1.0)
        return type("_Synth", (), {"duration_seconds": duration})()


class _DownTTS:
    async def synthesize(self, **_: Any) -> Any:
        raise RuntimeError("tts /synthesize returned 503: license_unavailable")


@pytest.fixture(autouse=True)
def _clean_wpm_cache() -> Any:
    clear_measured_wpm_cache()
    yield
    clear_measured_wpm_cache()


def _bdd_client() -> AsyncMock:
    client = AsyncMock()
    client.start_demo_video.return_value = "bdd-job-1"
    client.wait_for_job.return_value = {}
    client.get_job_result.return_value = BddJobResult.model_validate(
        {
            "job_id": "bdd-job-1",
            "status": "completed",
            "features_generated": 1,
            "narration_script": {
                "parts": [
                    {
                        "part_id": part_id,
                        "scenario_name": part_id,
                        "text": text,
                        "original_gherkin": f"Scenario: {part_id}",
                        "estimated_duration_seconds": 3.0,
                    }
                    for part_id, text in _DRAFT.items()
                ]
            },
        }
    )
    return client


async def _gen_then_doctor(tts: object) -> tuple[RunContext, dict[str, Any]]:
    """Run the real gen stage on the draft, then lay the doctored rewrite over it."""
    config = {
        "brief": {"target_url": "https://demo.playwright.dev/todomvc"},
        "deliverable": {"target_duration_seconds": 60.0},
    }
    gen_ctx = RunContext(
        run_id=uuid4(), deliverable_id=uuid4(), config_snapshot=config, prior_run_outputs={}
    )
    gen = (
        await DemoVideoGenStage(client=_bdd_client(), tts_client=cast("TTSClient", tts)).handle(
            DemoVideoGenInput(), gen_ctx
        )
    ).model_dump()
    doctored = {
        **gen,
        "lines": [{**line, "text": _DOCTORED[str(line["id"])]} for line in gen["lines"]],
    }
    ctx = RunContext(
        run_id=gen_ctx.run_id,
        deliverable_id=gen_ctx.deliverable_id,
        config_snapshot=config,
        prior_run_outputs={"demo_video_gen": gen, "script_doctor": doctored},
    )
    return ctx, gen


def _clip4_allotment(wpm: float) -> Any:
    demand = TimeDemand(
        id="clip-4",
        kind="footage",
        min=_CLIP4_CLIP_SECONDS,
        ideal=_CLIP4_CLIP_SECONDS,
        max=_CLIP4_CLIP_SECONDS,
        elasticity=0.0,
        pinned=True,
        word_count=len(_CLIP4_SPOKEN.split()),
    )
    (allotment,) = solve_timeline(
        [demand],
        target=_TARGET,
        wpm_natural=wpm,
        freeze_pad_ceiling_seconds=DEFAULT_FREEZE_PAD_CEILING,
    )
    cfg = _TimingConfig(
        mode="auto",
        target=_TARGET,
        tolerance_pct=2.0,
        wpm_band=(0.5, 2.0),
        wpm_natural=wpm,
        freeze_pad_ceiling_seconds=DEFAULT_FREEZE_PAD_CEILING,
        vo_rate_drag_floor=VO_RATE_DRAG_FLOOR,
        relief_mode="off",
        relief_iterations=0,
        relief_threshold=0.0,
    )
    return allotment, _overpad_note([demand], cfg)


async def _narrate_codes(allotment: Any) -> list[str]:
    async def _synthesize(**kwargs: Any) -> TTSSynthesis:
        speed = kwargs.get("speed")
        return TTSSynthesis(
            audio_bytes=b"\x01",
            sample_rate=24000,
            duration_seconds=_natural_seconds(_CLIP4_SPOKEN) / float(speed or 1.0),
            provider="kokoro",
            voice="af_heart",
        )

    client = AsyncMock()
    client.synthesize.side_effect = _synthesize
    line = {
        "id": "clip-4",
        "text": _CLIP4_SPOKEN,
        "seconds": allotment.seconds,
        "freeze_pad_seconds": allotment.freeze_pad_seconds,
        "hold_seconds": allotment.hold_seconds,
        "vo_rate": allotment.vo_rate,
    }
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"voice": {"voice_id": "af_heart"}},
        prior_run_outputs={"reconcile_timeline": {"lines": [line]}},
    )
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
        out = await stage.handle(stage.input_model(), ctx)
    return [str(d["code"]) for d in out.degradations]


async def test_reconcile_resamples_the_doctored_text_instead_of_reusing_the_draft_rate() -> None:
    ctx, gen = await _gen_then_doctor(_SpokenCharTTS())

    draft_wpm = _wpm_of(list(_DRAFT.values()))
    spoken_wpm = _wpm_of(list(_DOCTORED.values()))
    # Precondition: the fake reproduces the defect's shape — the URL-laden draft samples
    # far slower than the doctored lines.
    assert draft_wpm < 0.75 * spoken_wpm

    # gen WROTE the digests of exactly what it sampled, beside the draft rate. Since run
    # 01a0b007 gen makes bdd's text speakable BEFORE sampling (`speakable_text`), so what
    # it sampled is the URL-free draft — still different text from the doctored lines.
    speakable_draft = [speakable_text(text) for text in _DRAFT.values()]
    assert speakable_draft != list(_DRAFT.values())
    assert gen["resolved_wpm_basis"] == SCRIPT_SAMPLE
    assert gen["resolved_wpm"] == pytest.approx(_wpm_of(speakable_draft), rel=1e-6)
    assert gen["resolved_wpm_text_digests"] == spoken_text_digests(speakable_draft)

    plan = resolve_plan_output(ctx)
    solver_tts = _SpokenCharTTS()
    degradations: list[Degradation] = []
    record: dict[str, object] = {}
    solver_wpm = await ReconcileTimelineStage(
        tts_client=cast("TTSClient", solver_tts)
    )._effective_natural_wpm(plan["lines"], ctx, degradations, plan, record)

    assert solver_wpm == pytest.approx(spoken_wpm, rel=1e-6)
    assert solver_wpm != pytest.approx(gen["resolved_wpm"], rel=1e-3)
    # It measured the text narrate will speak — and never the draft.
    assert set(solver_tts.calls) <= set(_DOCTORED.values())
    assert solver_tts.calls
    assert degradations == []
    # …and records the rate it actually used, with the text it measured.
    assert record["resolved_wpm"] == solver_wpm
    assert record["resolved_wpm_basis"] == SCRIPT_SAMPLE
    assert record["resolved_wpm_text_digests"] == spoken_text_digests(_DOCTORED.values())


async def test_clip4_pad_advisory_and_floor_clamp_both_vanish_at_the_spoken_rate() -> None:
    ctx, _ = await _gen_then_doctor(_SpokenCharTTS())
    plan = resolve_plan_output(ctx)
    solver_wpm = await ReconcileTimelineStage(
        tts_client=cast("TTSClient", _SpokenCharTTS())
    )._effective_natural_wpm(plan["lines"], ctx, [], plan)
    assert solver_wpm is not None

    # The run's stale plan, replayed: 12.35s estimate, 1.5s pad, "overruns" advisory,
    # and narrate clamps at the floor. Proves this harness can see the defect at all.
    stale, stale_note = _clip4_allotment(_STALE_WPM)
    assert len(_CLIP4_SPOKEN.split()) / _STALE_WPM * 60.0 == pytest.approx(12.352, abs=1e-3)
    assert stale.freeze_pad_seconds == DEFAULT_FREEZE_PAD_CEILING
    assert stale_note is not None and "clip-4" in stale_note
    assert "narration_rate_floor_clamped" in await _narrate_codes(stale)

    fixed, fixed_note = _clip4_allotment(solver_wpm)
    estimate = len(_CLIP4_SPOKEN.split()) / solver_wpm * 60.0
    assert estimate < _CLIP4_CLIP_SECONDS
    assert estimate != pytest.approx(12.35, abs=0.5)
    assert fixed.freeze_pad_seconds == 0.0
    assert fixed_note is None
    assert "narration_rate_floor_clamped" not in await _narrate_codes(fixed)


async def test_unchanged_text_still_reuses_the_stored_rate_without_a_probe() -> None:
    tts = _SpokenCharTTS()
    ctx, gen = await _gen_then_doctor(tts)
    same = RunContext(
        run_id=ctx.run_id,
        deliverable_id=ctx.deliverable_id,
        config_snapshot=ctx.config_snapshot,
        # A subset of gen's own lines (a dropped beat) is still the sampled text.
        prior_run_outputs={"demo_video_gen": {**gen, "lines": gen["lines"][:2]}},
    )
    plan = resolve_plan_output(same)
    solver_tts = _SpokenCharTTS()
    record: dict[str, object] = {}
    solver_wpm = await ReconcileTimelineStage(
        tts_client=cast("TTSClient", solver_tts)
    )._effective_natural_wpm(plan["lines"], same, [], plan, record)
    assert solver_wpm == gen["resolved_wpm"]
    assert solver_tts.calls == []
    assert record == {}


async def test_a_failed_resample_never_falls_back_to_the_stale_draft_rate() -> None:
    ctx, gen = await _gen_then_doctor(_SpokenCharTTS())
    plan = resolve_plan_output(ctx)
    degradations: list[Degradation] = []
    record: dict[str, object] = {}
    solver_wpm = await ReconcileTimelineStage(
        tts_client=cast("TTSClient", _DownTTS())
    )._effective_natural_wpm(plan["lines"], ctx, degradations, plan, record)

    assert solver_wpm != gen["resolved_wpm"]
    assert record["resolved_wpm"] != gen["resolved_wpm"]
    assert record["resolved_wpm_basis"] != SCRIPT_SAMPLE
    assert record["resolved_wpm_reason"] is not None
    # No digests: nothing may later reuse the fallback as a measurement of these lines.
    assert record["resolved_wpm_text_digests"] is None
    assert [d["code"] for d in degradations] == ["speech_rate_unusable"]
    assert "reconcile_timeline" in degradations[0]["detail"]


async def test_the_stage_output_persists_the_rate_it_solved_at_not_gens() -> None:
    """End to end through ``handle``: the re-sampled rate and its digests are WRITTEN to
    ``reconcile_timeline``'s output, which every later stage reads via
    ``resolve_plan_output`` — not gen's draft rate riding along on the ``**gen`` spread."""
    ctx, gen = await _gen_then_doctor(_SpokenCharTTS())
    run_ctx = RunContext(
        run_id=ctx.run_id,
        deliverable_id=ctx.deliverable_id,
        config_snapshot={
            **ctx.config_snapshot,
            "direction": {"timing": {"mode": "auto"}},
            # Sized to this 3-line spine so the apply path (not the underfull refusal) runs.
            "deliverable": {"target_duration_seconds": 15.0},
        },
        prior_run_outputs={**ctx.prior_run_outputs, "demo_video_record": {"clips": []}},
    )
    stage = ReconcileTimelineStage(tts_client=cast("TTSClient", _SpokenCharTTS()))
    out = await stage.handle(stage.input_model(), run_ctx)

    assert out.resolved_wpm == pytest.approx(_wpm_of(list(_DOCTORED.values())), rel=1e-6)
    assert out.resolved_wpm != pytest.approx(gen["resolved_wpm"], rel=1e-3)
    assert out.resolved_wpm_text_digests == spoken_text_digests(_DOCTORED.values())
