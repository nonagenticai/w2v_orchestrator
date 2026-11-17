"""AV quality wave 5 — the ONE invariant this wave exists to establish.

PR #992 added Phase B (``resolve_script_sample_wpm``, ``speech_rate.py``) and wired
it into ``demo_video_reconcile_timeline`` ONLY. ``demo_video_gen``'s word budget kept
asking bdd for a script sized at the Phase-A calibration-passage rate (175.2 wpm)
while the solver judged that same script at the Phase-B measured content rate
(119.4 wpm) — two different numbers for the same script, which produced an even
worse regression than the one Phase B was meant to fix ("UNDERFULL — content fills
only 29.08s", clips pinned at the 0.545 floor). PR #998 reverted it.

This test is exactly the one that would have caught it: it stubs the TTS client to
answer the CALIBRATION_TEXT probe and the drafted script's own lines at
DELIBERATELY DIFFERENT rates, runs ``demo_video_gen`` end to end, and asserts that
the wpm it stamps on its output (the single-owner rate — see
``resolve_effective_wpm`` in ``speech_rate.py``) is BIT-IDENTICAL to the wpm
``demo_video_reconcile_timeline`` resolves for that same script, and that neither is
the calibration-passage rate.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.clients.schemas.bdd_playwright import BddJobResult
from src.clients.tts_client import TTSClient
from src.services.runs.run_context import RunContext
from src.services.stages.demo_video.gen import DemoVideoGenInput, DemoVideoGenStage
from src.services.stages.demo_video.reconcile_timeline import ReconcileTimelineStage
from src.services.stages.shared.degradation import Degradation
from src.services.stages.shared.speech_rate import (
    CALIBRATION_TEXT,
    SCRIPT_SAMPLE,
    clear_measured_wpm_cache,
)

# Deliberately far apart and far from DEFAULT_WPM_NATURAL (150.0) so a test that
# accidentally reads the wrong rung, or falls back to the constant, cannot pass by
# coincidence.
_CALIBRATION_WPM = 175.2
_CONTENT_WPM = 100.0

_URL = "https://app.example"
_ORDERED_SECONDS = 60.0

# Two lines (>= speech_rate.MIN_SAMPLE_LINES), each at or above
# speech_rate.MIN_SAMPLE_AVG_WORDS_PER_LINE, so Phase B has enough REPRESENTATIVE
# measurements to land rather than falling back to the prior with
# SAMPLE_TOO_FEW_MEASURED or SAMPLE_LINES_TOO_SHORT.
_LINES = [
    {
        "part_id": "clip-1",
        "text": "First we open the dashboard, sign in, and review the account settings.",
    },
    {
        "part_id": "clip-2",
        "text": "Then we review the settings page in detail, checking every option carefully.",
    },
]


@pytest.fixture(autouse=True)
def _clean_wpm_cache() -> Any:
    clear_measured_wpm_cache()
    yield
    clear_measured_wpm_cache()


class _RateAwareTTS:
    """Answers the calibration probe and real script lines at DIFFERENT rates.

    Ignores ``voice``/``provider``/``speed`` (irrelevant here) and keys purely off
    the text: exactly ``CALIBRATION_TEXT`` gets the calibration rate, anything else
    (the drafted lines) gets the content rate — modelling the live defect exactly
    (a fixed passage measures faster than real narration).
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def synthesize(self, *, text: str, **_: Any) -> Any:
        self.calls.append(text)
        words = len(text.split())
        wpm = _CALIBRATION_WPM if text == CALIBRATION_TEXT else _CONTENT_WPM
        duration = words * 60.0 / wpm
        return type("_Synth", (), {"duration_seconds": duration})()


def _tts() -> TTSClient:
    return cast("TTSClient", _RateAwareTTS())


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
                        "part_id": part["part_id"],
                        "scenario_name": part["part_id"],
                        "text": part["text"],
                        "original_gherkin": f"Scenario: {part['part_id']}",
                        "estimated_duration_seconds": 3.0,
                    }
                    for part in _LINES
                ]
            },
        }
    )
    return client


def _gen_ctx() -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "brief": {"target_url": _URL},
            "deliverable": {"target_duration_seconds": _ORDERED_SECONDS},
        },
        prior_run_outputs={},
    )


@pytest.mark.asyncio
async def test_gen_and_reconcile_agree_on_the_same_measured_wpm() -> None:
    gen_ctx = _gen_ctx()
    gen_stage = DemoVideoGenStage(client=_bdd_client(), tts_client=_tts())
    gen_output = await gen_stage.handle(DemoVideoGenInput(), gen_ctx)

    # Phase B must have LANDED (not fallen back to the prior) — otherwise the rest
    # of this test would trivially pass by both sides sharing a fallback default.
    assert gen_output.resolved_wpm_measured is True
    assert gen_output.resolved_wpm_basis == SCRIPT_SAMPLE
    assert gen_output.resolved_wpm == pytest.approx(_CONTENT_WPM, rel=1e-6)
    # Not the calibration-passage rate — proves it measured the SCRIPT, not the probe.
    assert gen_output.resolved_wpm != pytest.approx(_CALIBRATION_WPM, rel=1e-3)

    # `resolve_plan_output` reads this via `ctx.prior_output(...)` on a plain dict —
    # exactly the shape the DAG executor persists.
    gen_dict = gen_output.model_dump()

    # A FRESH probe for the solver half, so "reconcile made zero extra calls" proves
    # it read the STORED value rather than re-measuring.
    solver_tts = _RateAwareTTS()
    reconcile_ctx = RunContext(
        run_id=gen_ctx.run_id,
        deliverable_id=gen_ctx.deliverable_id,
        config_snapshot=gen_ctx.config_snapshot,
        prior_run_outputs={"demo_video_gen": gen_dict},
    )
    reconcile_stage = ReconcileTimelineStage(tts_client=cast("TTSClient", solver_tts))
    solver_degradations: list[Degradation] = []
    # Private method call is deliberate: it is the exact function that supplies
    # `wpm_natural` to the solver, without standing up the full WM-calling loop.
    solver_wpm = await reconcile_stage._effective_natural_wpm(
        gen_dict["lines"], reconcile_ctx, solver_degradations, gen_dict
    )

    # ⛔ THE INVARIANT: bit-identical, and traceable to the SAME measurement.
    assert solver_wpm == gen_output.resolved_wpm
    assert solver_wpm != pytest.approx(_CALIBRATION_WPM, rel=1e-3)
    assert solver_wpm != pytest.approx(150.0, rel=1e-3)  # DEFAULT_WPM_NATURAL
    # And it cost no extra probe — read the stored rate, never re-measured it.
    assert solver_tts.calls == []


class _DownTTS:
    """tts_backend answering 503 to everything — the probe never measures."""

    async def synthesize(self, **_: Any) -> Any:
        raise RuntimeError("tts /synthesize returned 503: license_unavailable")


@pytest.mark.asyncio
async def test_an_unmeasured_budget_rate_is_recorded_once_not_silent() -> None:
    """A failed speech-rate probe budgets narration at the default and used to
    complete CLEAN (`resolved_wpm_measured: false`, no degradation). The stage that
    budgets from it must record the fallback with its reason, and the solver's
    single-voice read-back must not record it a second time."""
    gen_ctx = _gen_ctx()
    gen_output = await DemoVideoGenStage(
        client=_bdd_client(), tts_client=cast("TTSClient", _DownTTS())
    ).handle(DemoVideoGenInput(), gen_ctx)

    assert gen_output.resolved_wpm_measured is False
    unusable = [d for d in gen_output.degradations if d["code"] == "speech_rate_unusable"]
    assert len(unusable) == 1
    assert "demo_video_gen" in unusable[0]["detail"]
    assert f"reason={gen_output.resolved_wpm_reason}" in unusable[0]["detail"]

    gen_dict = gen_output.model_dump()
    reconcile_ctx = RunContext(
        run_id=gen_ctx.run_id,
        deliverable_id=gen_ctx.deliverable_id,
        config_snapshot=gen_ctx.config_snapshot,
        prior_run_outputs={"demo_video_gen": gen_dict},
    )
    solver_degradations: list[Degradation] = []
    await ReconcileTimelineStage(tts_client=cast("TTSClient", _DownTTS()))._effective_natural_wpm(
        gen_dict["lines"], reconcile_ctx, solver_degradations, gen_dict
    )
    assert solver_degradations == []


# ══ the two-term split reaches the STAGE OUTPUT, beside the rate it decomposes ══
#
# `speech_rate` computed `speech_seconds_split` / `pause_seconds_split` /
# `approx_pieces_split` / `lines_split` and dropped all four at a DIAGNOSTIC-ONLY
# log line. `resolved_wpm` is already persisted here and read back by
# `reconcile_timeline`; the decomposition of that same measurement now rides with
# it, so the evidence a future two-term fit needs accumulates on the run record.
#
# ⛔ NOTHING BRANCHES ON THEM, and this test pins that too: `resolved_wpm` and the
# solver's answer are asserted unchanged.


class _SplitRateAwareTTS(_RateAwareTTS):
    """`_RateAwareTTS` whose script-line results also report the backend split.

    A fixed 30% of each script line's duration is pause. The calibration probe
    reports NO split — it is Phase A, and pooling it would mix the passage into a
    measurement of the script.
    """

    async def synthesize(self, *, text: str, **kwargs: Any) -> Any:
        result = await super().synthesize(text=text, **kwargs)
        if text != CALIBRATION_TEXT:
            total = float(result.duration_seconds)
            result.pause_seconds = total * 0.30
            result.speech_seconds = total * 0.70
        return result


@pytest.mark.asyncio
async def test_gen_persists_the_two_term_split_beside_the_rate() -> None:
    fake = _SplitRateAwareTTS()
    gen_output = await DemoVideoGenStage(
        client=_bdd_client(), tts_client=cast("TTSClient", fake)
    ).handle(DemoVideoGenInput(), _gen_ctx())

    # Phase B LANDED — otherwise the split below would trivially be the zeroed
    # "no sample ran" record and this test would pass having asserted nothing.
    assert gen_output.resolved_wpm_basis == SCRIPT_SAMPLE
    assert gen_output.resolved_wpm == pytest.approx(_CONTENT_WPM, rel=1e-6)

    sampled_seconds = sum(len(line["text"].split()) for line in _LINES) * 60.0 / _CONTENT_WPM
    assert gen_output.lines_split == len(_LINES)
    assert gen_output.speech_seconds_split == pytest.approx(sampled_seconds * 0.70)
    assert gen_output.pause_seconds_split == pytest.approx(sampled_seconds * 0.30)
    # One sentence-terminating '.' per line ⇒ one approximate piece each.
    assert gen_output.approx_pieces_split == len(_LINES)

    # …and it survives the JSON round-trip the DAG executor persists, which is the
    # only form a later query or a future fit will ever see.
    assert gen_output.model_dump()["pause_seconds_split"] == pytest.approx(sampled_seconds * 0.30)


@pytest.mark.asyncio
async def test_a_backend_with_no_split_persists_zero_lines_split() -> None:
    """`lines_split == 0` is "no evidence", NOT "this voice inserts no padding".

    The seconds are zero either way; this field is the only thing that tells the
    two apart on the persisted record. And the rate is unaffected — the whole
    point of the tier is that a backend that cannot answer still budgets normally.
    """
    gen_output = await DemoVideoGenStage(client=_bdd_client(), tts_client=_tts()).handle(
        DemoVideoGenInput(), _gen_ctx()
    )

    assert gen_output.resolved_wpm == pytest.approx(_CONTENT_WPM, rel=1e-6)
    assert gen_output.lines_split == 0
    assert gen_output.speech_seconds_split == 0.0
    assert gen_output.pause_seconds_split == 0.0
    assert gen_output.approx_pieces_split == 0
