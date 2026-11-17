"""No URL in model-generated narration reaches the voice — replay run 01a0b007's clip-5.

script_doctor's accepted clip-5 was ``Finally, I am on the React TodoMVC application at
"https://demo.playwright.dev/todomvc".`` (verbatim from ``stage_outputs``). The voice spells
the URL out; every budget priced it as one word. The fake TTS below charges per SPOKEN
character with the URL's separators voiced, so a URL costs real time here too.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from src.clients.schemas.bdd_playwright import BddJobResult
from src.clients.schemas.tts import TTSSynthesis
from src.services.runs.run_context import RunContext
from src.services.stages.demo_video.gen import DemoVideoGenInput, DemoVideoGenStage
from src.services.stages.demo_video.narrate import DemoVideoNarrateStage
from src.services.stages.demo_video.reconcile_timeline import ReconcileTimelineStage
from src.services.stages.shared.speakable_text import (
    contains_url,
    speakable_text,
)
from src.services.stages.shared.speech_rate import (
    CALIBRATION_TEXT,
    clear_measured_wpm_cache,
    spoken_text_digests,
)

if TYPE_CHECKING:
    from src.clients.tts_client import TTSClient

_URL = "https://demo.playwright.dev/todomvc"
_SD_CLIP5 = f'Finally, I am on the React TodoMVC application at "{_URL}".'
_SD_CLIP5_SPEAKABLE = "Finally, I am on the React TodoMVC application."
_LINES = {
    "clip-1": 'I open the React TodoMVC app and type "Buy groceries" into the input field.',
    "clip-4": 'Next, I enter "Pay bills" into the todo input field.',
    "clip-5": "Finally, I return to the main TodoMVC application view.",
}
_CLIPS = {"clip-1": 14.56, "clip-4": 9.88, "clip-5": 9.56}
_SECONDS_PER_SPOKEN_CHAR = 0.0612
_URL_RE = re.compile(r"https?://[^\s\"]+")
_TEST_URL_RE = re.compile(r"https?://|www\.|\b[a-z0-9-]+\.[a-z]{2,}/", re.IGNORECASE)


def _natural_seconds(text: str) -> float:
    def _voice(match: re.Match[str]) -> str:
        url = match.group(0).replace("://", " colon slash slash ")
        return url.replace("/", " slash ").replace(".", " dot ")

    return len(_URL_RE.sub(_voice, text)) * _SECONDS_PER_SPOKEN_CHAR


def _wpm_of(texts: list[str]) -> float:
    return sum(len(t.split()) for t in texts) / (sum(_natural_seconds(t) for t in texts) / 60.0)


class _SpokenCharTTS:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def synthesize(self, *, text: str, speed: float | None = None, **_: Any) -> Any:
        self.calls.append(text)
        if text == CALIBRATION_TEXT:
            return SimpleNamespace(duration_seconds=len(text.split()) * 60.0 / 175.2)
        return SimpleNamespace(duration_seconds=_natural_seconds(text) / float(speed or 1.0))


@pytest.fixture(autouse=True)
def _clean_wpm_cache() -> Any:
    clear_measured_wpm_cache()
    yield
    clear_measured_wpm_cache()


async def _narrate_tts_inputs(plan: dict[str, Any]) -> list[str]:
    heard: list[str] = []

    async def _synthesize(**kwargs: Any) -> TTSSynthesis:
        heard.append(str(kwargs["text"]))
        return TTSSynthesis(
            audio_bytes=b"\x01",
            sample_rate=24000,
            duration_seconds=_natural_seconds(str(kwargs["text"])),
            provider="kokoro",
            voice="af_heart",
        )

    client = AsyncMock()
    client.synthesize.side_effect = _synthesize
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"voice": {"voice_id": "af_heart"}},
        prior_run_outputs={"reconcile_timeline": plan},
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
        await stage.handle(stage.input_model(), ctx)
    return heard


def _reconcile_ctx(
    writer: str, lines: dict[str, str], brief: dict[str, Any] | None = None
) -> RunContext:
    draft = [{"id": k, "text": v, "seconds": None, "clip": k} for k, v in _LINES.items()]
    gen = {
        "lines": draft,
        "bdd_job_id": "",
        "features_generated": 0,
        "package_url": "",
        "resolved_wpm": 108.36,
        "resolved_wpm_measured": True,
        "resolved_wpm_basis": "script_sample",
        "resolved_wpm_text_digests": spoken_text_digests(_LINES.values()),
    }
    written = [{**line, "text": lines[str(line["id"])]} for line in draft]
    config: dict[str, Any] = {
        "direction": {"timing": {"mode": "auto"}},
        "deliverable": {"target_duration_seconds": 35.0},
    }
    if brief is not None:
        config["brief"] = brief
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot=config,
        prior_run_outputs={
            "demo_video_gen": gen,
            # Every plan mutator re-emits gen's shape with only the text revised.
            writer: {**gen, "lines": written},
            "demo_video_record": {
                "clips": [
                    {"id": k, "line_id": k, "video_url": f"https://m/{k}.mp4", "seconds": s}
                    for k, s in _CLIPS.items()
                ]
            },
        },
    )


@pytest.mark.parametrize("writer", ["script_doctor"])
async def test_a_generated_url_never_reaches_the_rate_or_narrates_tts(writer: str) -> None:
    """Run 01a0b007's clip-5, written by ANY model stage upstream of reconcile. No URL is
    measured, budgeted or heard — one boundary, not one call site per writer."""
    ctx = _reconcile_ctx(writer, {**_LINES, "clip-5": _SD_CLIP5})
    texts = [*list(_LINES.values())[:2], _SD_CLIP5_SPEAKABLE]
    tts = _SpokenCharTTS()
    stage = ReconcileTimelineStage(tts_client=cast("TTSClient", tts))
    out = await stage.handle(stage.input_model(), ctx)

    heard = await _narrate_tts_inputs(out.model_dump())
    assert heard, "precondition: narrate synthesized the spine"
    for text in heard:
        assert not _TEST_URL_RE.search(text), f"narrate's TTS was handed a URL: {text!r}"
    assert [line.id for line in out.lines] == list(_LINES), "no line is dropped"

    # The rate was sampled and solved on the URL-free text.
    assert not any(_TEST_URL_RE.search(call) for call in tts.calls), tts.calls
    assert out.resolved_wpm == pytest.approx(_wpm_of(texts), rel=1e-6)
    assert out.resolved_wpm_text_digests == spoken_text_digests(texts)


@pytest.mark.parametrize("authored_text", ["Sign up at https://acme.com/start today."])
async def test_an_unlocked_owner_authored_line_keeps_its_url(authored_text: str) -> None:
    """brief.lines ⇒ authored run. ``locked`` defaults to False and is NOT the signal: a
    model's rewrite (script_doctor) of an authored line is still never rewritten here."""
    brief = {
        "package_url": "https://minio.test/pkg.zip",
        "lines": [{"id": k, "text": v, "locked": False} for k, v in _LINES.items()],
    }
    ctx = _reconcile_ctx("script_doctor", {**_LINES, "clip-5": authored_text}, brief)
    stage = ReconcileTimelineStage(tts_client=cast("TTSClient", _SpokenCharTTS()))
    out = await stage.handle(stage.input_model(), ctx)
    assert {line.id: line.text for line in out.lines}["clip-5"] == authored_text


async def test_gen_makes_bdds_url_line_speakable_before_sampling_the_rate() -> None:
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
                        "text": _SD_CLIP5 if part_id == "clip-5" else text,
                        "original_gherkin": f"Scenario: {part_id}",
                        "estimated_duration_seconds": 3.0,
                    }
                    for part_id, text in _LINES.items()
                ]
            },
        }
    )
    tts = _SpokenCharTTS()
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "brief": {"target_url": _URL},
            "deliverable": {"target_duration_seconds": 60.0},
        },
        prior_run_outputs={},
    )
    out = await DemoVideoGenStage(client=client, tts_client=cast("TTSClient", tts)).handle(
        DemoVideoGenInput(), ctx
    )
    texts = [str(line.text) for line in out.lines]
    # The Phase-B sample never synthesized a URL, and the stored rate is the URL-free text's.
    assert not any(_TEST_URL_RE.search(call) for call in tts.calls), tts.calls
    assert out.resolved_wpm == pytest.approx(_wpm_of(texts), rel=1e-6)
    assert _SD_CLIP5_SPEAKABLE in texts


async def test_owner_authored_brief_lines_keep_their_url() -> None:
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "brief": {
                "package_url": "https://minio.test/pkg.zip",
                "lines": [{"id": "clip-1", "text": f"Visit {_URL} today."}],
            },
        },
        prior_run_outputs={},
    )
    out = await DemoVideoGenStage(
        client=AsyncMock(), tts_client=cast("TTSClient", _SpokenCharTTS())
    ).handle(DemoVideoGenInput(), ctx)
    assert out.lines[0].text == f"Visit {_URL} today."


@pytest.mark.parametrize(
    ("raw", "speakable"),
    [(_SD_CLIP5, _SD_CLIP5_SPEAKABLE)],
)
def test_speakable_text_rewrites_scheme_and_www_urls(raw: str, speakable: str) -> None:
    assert speakable_text(raw) == speakable
    assert not contains_url(speakable)
    assert speakable_text(speakable) == speakable


@pytest.mark.parametrize(
    "prose",
    ["The page redirected to /checkout/success."],
)
def test_prose_that_only_looks_like_a_path_is_left_unchanged(prose: str) -> None:
    assert not contains_url(prose)
    assert speakable_text(prose) == prose
