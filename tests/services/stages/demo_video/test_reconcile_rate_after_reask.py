"""The underspoken re-ask rewrites narration AFTER the rate was measured — replay run 01a0b007.

Acceptance run ``01a0b007-9c2c-716a-883f-beefd14f4fd5`` (30 s order, 3 footage beats).
``reconcile_timeline`` correctly re-sampled script_doctor's lines at 136.06 wpm, then the
underspoken re-ask rewrote all three:

* clip-5 came back carrying a literal URL (16 "words") and was delivered at 82.3 wpm;
* clip-1 went from 14 to 41 words against a 14.56 s clip — the stage's own FREEZE-PAD
  ADVISORY fired on the text its re-ask had just written;
* the persisted ``resolved_wpm_text_digests`` described the PRE-re-ask text, and
  ``timeline_manifest.beats[].vo_rate`` read 0.97 on all three beats.

The fake TTS charges per SPOKEN character with a URL's separators voiced, so a URL costs
real time. Every rate below is derived from the fake, never hardcoded.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

import pytest

from src.services.runs.run_context import RunContext
from src.services.stages.demo_video.reconcile_timeline import ReconcileTimelineStage
from src.services.stages.shared.speakable_text import speakable_text
from src.services.stages.shared.speech_rate import (
    CALIBRATION_TEXT,
    SCRIPT_SAMPLE,
    clear_measured_wpm_cache,
    spoken_text_digests,
)
from src.services.stages.shared.time_budget import (
    DEFAULT_FREEZE_PAD_CEILING,
    collect_demands,
)
from src.services.stages.shared.time_budget import reconcile_timeline as solve_timeline

if TYPE_CHECKING:
    from src.clients.tts_client import TTSClient
    from src.services.llm import WMClient

_URL = "https://demo.playwright.dev/todomvc"
_TARGET = 30.0
_CLIPS = {"clip-1": 14.56, "clip-4": 9.88, "clip-5": 9.56}

# script_doctor's lines for the run, verbatim from `stage_outputs`.
_DOCTORED = {
    "clip-1": 'I open the React TodoMVC app and type "Buy groceries" into the input field.',
    "clip-4": 'Next, I enter "Pay bills" into the todo input field.',
    "clip-5": f'Finally, I am on the React TodoMVC application at "{_URL}".',
}
# The re-ask's replies on the run, verbatim.
_REASK_CLIP1_41_WORDS = (
    'I open the React TodoMVC application on my screen and carefully type the phrase "Buy '
    'groceries" into the designated input field, taking my time to ensure each character is '
    "entered correctly before preparing to add this new task to the list."
)
_REASK_CLIP4 = (
    'Next, I carefully type the phrase "Pay bills" directly into the designated todo input '
    "field on the screen, ensuring each character is entered precisely."
)
_REASK_CLIP5_URL = (
    f'Finally, I am now on the React TodoMVC application, which is located at the URL "{_URL}".'
)
# A URL-free rewrite, deliberately SHORTER than the URL reply: with URL enforcement removed
# the URL reply is applied first and this one is then refused as not-longer, so the URL
# survives — which is what makes the enforcement mutation observable.
_REASK_CLIP5_CLEAN = (
    "Finally, I am back on the React TodoMVC demo application, ready for the next step."
)
# A draft rate gen stored (the run's value), over text that is NOT what reconcile sees.
_DRAFT_WPM = 108.36478295060748

# Chosen so script_doctor's lines — as reconcile now READS them, with clip-5's URL clause
# made speakable at its entry boundary — sample at the run's measured 136.06 wpm, so the
# re-ask bound is exercised at the run's rate.
_SECONDS_PER_SPOKEN_CHAR = 0.0811
_URL_RE = re.compile(r"https?://[^\s\"]+")
_TEST_URL_RE = re.compile(r"https?://|www\.|\b[a-z0-9-]+\.[a-z]{2,}/", re.IGNORECASE)


def _spoken(text: str) -> str:
    def _voice(match: re.Match[str]) -> str:
        url = match.group(0).replace("://", " colon slash slash ")
        return url.replace("/", " slash ").replace(".", " dot ")

    return _URL_RE.sub(_voice, text)


def _natural_seconds(text: str) -> float:
    return len(_spoken(text)) * _SECONDS_PER_SPOKEN_CHAR


def _wpm_of(texts: list[str]) -> float:
    return sum(len(t.split()) for t in texts) / (sum(_natural_seconds(t) for t in texts) / 60.0)


class _SpokenCharTTS:
    def __init__(self, fail_on: frozenset[str] = frozenset()) -> None:
        self.calls: list[str] = []
        self._fail_on = fail_on

    async def synthesize(self, *, text: str, speed: float | None = None, **_: Any) -> Any:
        self.calls.append(text)
        if text in self._fail_on:
            raise RuntimeError("tts /synthesize returned 503: license_unavailable")
        if text == CALIBRATION_TEXT:
            duration = len(text.split()) * 60.0 / 175.2
        else:
            duration = _natural_seconds(text) / float(speed or 1.0)
        return SimpleNamespace(duration_seconds=duration)


class _RunWM:
    """The run's WM, keyed by worker and (for the expander) by beat."""

    def __init__(self) -> None:
        self.expander_calls: list[str] = []
        self._replies = {
            "clip-1": [_REASK_CLIP1_41_WORDS],
            "clip-4": [_REASK_CLIP4],
            "clip-5": [_REASK_CLIP5_URL, _REASK_CLIP5_CLEAN],
        }

    async def message(self, **kwargs: Any) -> Any:
        worker = str(kwargs.get("worker_name") or "")
        if worker == "pacing-critic":
            return SimpleNamespace(output={"overall_score": 4}, output_type="object")
        if worker == "beat-expander":
            msg = str(kwargs["message"])
            self.expander_calls.append(msg)
            beat = (
                "clip-1" if "Buy groceries" in msg else "clip-4" if "Pay bills" in msg else "clip-5"
            )
            queue = self._replies[beat]
            return SimpleNamespace(output={"text": queue.pop(0) if queue else ""})
        return SimpleNamespace(output={"directives": []}, output_type="object")


@pytest.fixture(autouse=True)
def _clean_wpm_cache() -> Any:
    clear_measured_wpm_cache()
    yield
    clear_measured_wpm_cache()


def _ctx() -> RunContext:
    lines = [
        {"id": beat_id, "text": text, "seconds": None, "clip": beat_id}
        for beat_id, text in _DOCTORED.items()
    ]
    gen = {
        "lines": lines,
        "bdd_job_id": "",
        "features_generated": 0,
        "package_url": "",
        "resolved_wpm": _DRAFT_WPM,
        "resolved_wpm_measured": True,
        "resolved_wpm_reason": None,
        "resolved_wpm_basis": SCRIPT_SAMPLE,
        "resolved_wpm_text_digests": spoken_text_digests(["a draft that was rewritten"]),
    }
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "direction": {
                "timing": {
                    "mode": "auto",
                    "relief": {"mode": "auto", "iterations": 2, "score_threshold": 4.0},
                }
            },
            "deliverable": {"target_duration_seconds": _TARGET},
        },
        prior_run_outputs={
            "demo_video_gen": gen,
            "demo_video_record": {
                "clips": [
                    {
                        "id": beat_id,
                        "line_id": beat_id,
                        "video_url": f"https://minio.test/{beat_id}.mp4",
                        "seconds": seconds,
                    }
                    for beat_id, seconds in _CLIPS.items()
                ]
            },
        },
    )


async def test_replay_01a0b007_rate_digests_urls_bound_and_manifest_describe_the_final_text() -> (
    None
):
    tts = _SpokenCharTTS()
    wm = _RunWM()
    stage = ReconcileTimelineStage(client=cast("WMClient", wm), tts_client=cast("TTSClient", tts))
    out = await stage.handle(stage.input_model(), _ctx())

    doctored_wpm = _wpm_of([speakable_text(text) for text in _DOCTORED.values()])
    # PRECONDITIONS: the harness reproduces the run's shape.
    assert doctored_wpm == pytest.approx(136.06, abs=0.1)
    assert wm.expander_calls, "the re-ask must have run"
    ask_window_words = (_CLIPS["clip-1"] + DEFAULT_FREEZE_PAD_CEILING) * doctored_wpm / 60.0
    assert len(_REASK_CLIP1_41_WORDS.split()) > ask_window_words, (
        "precondition: 41 words overrun clip-1's window at the solved rate"
    )

    final = {line.id: str(line.text) for line in out.lines}
    final_texts = [final[beat_id] for beat_id in _DOCTORED]
    assert final_texts != list(_DOCTORED.values()), "precondition: the re-ask rewrote text"

    # (1) the rate is measured on the FINAL text.
    final_wpm = _wpm_of(final_texts)
    assert final_wpm != pytest.approx(doctored_wpm, rel=1e-3)
    assert out.resolved_wpm == pytest.approx(final_wpm, rel=1e-6)
    assert out.resolved_wpm_basis == SCRIPT_SAMPLE

    # (2) the digests match the final lines.
    assert out.resolved_wpm_text_digests == spoken_text_digests(final_texts)

    # (3) no URL survives into spoken text.
    for beat_id, text in final.items():
        assert not _TEST_URL_RE.search(text), f"{beat_id} still speaks a URL: {text!r}"

    # (4) the over-budget rewrite is rejected: clip-1 keeps script_doctor's line.
    assert final["clip-1"] == _DOCTORED["clip-1"]
    assert final["clip-4"] == _REASK_CLIP4

    # (5) vo_rate — on the spine AND the manifest — is the solve of the final text.
    assert out.timeline_manifest is not None
    demands = collect_demands(
        [{"id": k, "text": v, "seconds": None, "clip": k} for k, v in final.items()],
        clip_seconds=_CLIPS,
        wpm_band=(130.0 * final_wpm / 150.0, 170.0 * final_wpm / 150.0),
        wpm_natural=final_wpm,
    )
    expected = {
        a.id: a.vo_rate
        for a in solve_timeline(
            demands,
            target=_TARGET,
            wpm_natural=final_wpm,
            freeze_pad_ceiling_seconds=DEFAULT_FREEZE_PAD_CEILING,
        )
    }
    delivered = {line.id: line.vo_rate for line in out.lines}
    manifest_rates = {beat.id: beat.vo_rate for beat in out.timeline_manifest.beats}
    assert delivered == pytest.approx(expected)
    assert manifest_rates == pytest.approx(expected)


async def test_a_failed_post_reask_measurement_is_honest_never_the_pre_reask_rate() -> None:
    """The re-measure fails on the rewritten lines ⇒ #1041's honest shape, not 136 wpm."""
    tts = _SpokenCharTTS(fail_on=frozenset({_REASK_CLIP4, _REASK_CLIP5_CLEAN}))
    stage = ReconcileTimelineStage(
        client=cast("WMClient", _RunWM()), tts_client=cast("TTSClient", tts)
    )
    out = await stage.handle(stage.input_model(), _ctx())

    assert _REASK_CLIP4 in [line.text for line in out.lines], "precondition: text rewritten"
    assert out.resolved_wpm != pytest.approx(
        _wpm_of([speakable_text(text) for text in _DOCTORED.values()]), rel=1e-3
    )
    assert out.resolved_wpm_text_digests is None
    assert out.resolved_wpm_reason is not None
    assert "speech_rate_unusable" in [d["code"] for d in out.degradations]
