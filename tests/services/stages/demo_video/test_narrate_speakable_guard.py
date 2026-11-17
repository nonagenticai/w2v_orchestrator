"""narrate's final speakable guard — the boundary that covers a run reconcile SKIPPED.

``reconcile_timeline._speakable_generated_spine`` only runs when reconcile runs; a run
with no ``direction.timing`` block skips it, and script_doctor's clip-5 of run
``01a0b007`` (``…application at "https://demo.playwright.dev/todomvc".``) would then reach
TTS verbatim. These replay that skip path end to end: reconcile is really invoked and
really skips, then narrate reads the plan exactly as the DAG hands it.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from src.clients.schemas.tts import TTSSynthesis
from src.services.runs.run_context import RunContext
from src.services.stages.demo_video.narrate import DemoVideoNarrateStage
from src.services.stages.demo_video.reconcile_timeline import ReconcileTimelineStage
from src.services.stages.errors import StageSkippedError
from src.services.stages.shared.degradation import URL_REWRITTEN_BEFORE_TTS

_URL = "https://demo.playwright.dev/todomvc"
_SD_CLIP5 = f'Finally, I am on the React TodoMVC application at "{_URL}".'
_LINES = {
    "clip-1": 'I open the React TodoMVC app and type "Buy groceries" into the input field.',
    "clip-4": 'Next, I enter "Pay bills" into the todo input field.',
    "clip-5": "Finally, I return to the main TodoMVC application view.",
}
_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)


def _ctx(brief: dict[str, Any] | None = None) -> RunContext:
    draft = [{"id": k, "text": v, "clip": k} for k, v in _LINES.items()]
    gen = {"lines": draft, "bdd_job_id": "", "features_generated": 0, "package_url": ""}
    doctored = [{**line, "text": _SD_CLIP5} if line["id"] == "clip-5" else line for line in draft]
    # NO `direction.timing` block ⇒ reconcile_timeline skips.
    config: dict[str, Any] = {"voice": {"voice_id": "af_heart"}}
    if brief is not None:
        config["brief"] = brief
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot=config,
        prior_run_outputs={"demo_video_gen": gen, "script_doctor": {**gen, "lines": doctored}},
    )


async def _narrate(ctx: RunContext) -> tuple[dict[str, str], Any]:
    heard: dict[str, str] = {}

    async def _synthesize(**kwargs: Any) -> TTSSynthesis:
        text = str(kwargs["text"])
        # Keyed by the first SPOKEN word — SSML tags stripped so lines never collide.
        heard[re.sub(r"<[^>]+>", "", text).split()[0]] = text
        return TTSSynthesis(
            audio_bytes=b"\x01",
            sample_rate=24000,
            duration_seconds=len(text.split()) * 0.4,
            provider="kokoro",
            voice="af_heart",
        )

    client = AsyncMock()
    client.synthesize.side_effect = _synthesize
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
    return heard, out


async def _assert_reconcile_skips(ctx: RunContext) -> None:
    stage = ReconcileTimelineStage(tts_client=AsyncMock())
    with pytest.raises(StageSkippedError):
        await stage.handle(stage.input_model(), ctx)


async def test_reconcile_skipped_generated_url_never_reaches_narrates_tts() -> None:
    ctx = _ctx()
    await _assert_reconcile_skips(ctx)

    heard, out = await _narrate(ctx)

    assert len(heard) == len(_LINES), "precondition: every line was synthesized"
    for text in heard.values():
        assert not _URL_RE.search(text), f"narrate's TTS was handed a URL: {text!r}"
    assert heard["Finally,"] == "Finally, I am on the React TodoMVC application."
    # Recorded STRUCTURED, naming the rewritten line and only that line.
    codes = [d for d in out.degradations if d["code"] == URL_REWRITTEN_BEFORE_TTS]
    assert len(codes) == 1, out.degradations
    assert "clip-5" in codes[0]["detail"], codes
    assert "clip-1" not in codes[0]["detail"] and "clip-4" not in codes[0]["detail"]
    assert "pre-rewrite text" in codes[0]["detail"]
    # No prose note carrying the old, inaccurate "measured on the URL text" claim.
    assert not any("URL" in note for note in out.notes), out.notes


async def test_owner_authored_brief_lines_keep_their_url_at_narrate() -> None:
    brief = {
        "package_url": "https://minio.test/pkg.zip",
        "lines": [{"id": k, "text": v, "locked": False} for k, v in _LINES.items()],
    }
    ctx = _ctx(brief)
    await _assert_reconcile_skips(ctx)

    heard, out = await _narrate(ctx)

    assert heard["Finally,"] == _SD_CLIP5
    assert not any("URL" in note for note in out.notes), out.notes
    assert not [d for d in out.degradations if d["code"] == URL_REWRITTEN_BEFORE_TTS]


# ══ emphasis on a rewritten line ════════════════════════════════════════════════
# voice_director indexes the UNREWRITTEN `text.split()`. After the URL rewrite those
# positions name different words, so a rewritten generated line must send NO
# emphasis; an unrewritten line keeps its directive's emphasis.


def _ssml_on() -> Any:
    return patch(
        "src.services.stages.demo_video.narrate.get_settings",
        return_value=SimpleNamespace(narration_ssml_enabled=True),
    )


def _with_directives(ctx: RunContext) -> RunContext:
    # Index 6 of clip-5's UNREWRITTEN text is the quoted URL; index 1 of clip-1 is "open".
    ctx.prior_run_outputs["voice_director"] = {
        "directives": [
            {"line_id": "clip-5", "pause_before_ms": 300, "emphasis_word_indices": [6]},
            {"line_id": "clip-1", "pause_before_ms": 300, "emphasis_word_indices": [1]},
        ]
    }
    return ctx


async def test_url_rewritten_generated_line_sends_no_emphasis_to_tts() -> None:
    ctx = _with_directives(_ctx())
    await _assert_reconcile_skips(ctx)

    with _ssml_on():
        heard, _ = await _narrate(ctx)

    sent = [t for t in heard.values() if "TodoMVC application" in t and "Finally," in t]
    assert len(sent) == 1, heard
    assert "<emphasis" not in sent[0], f"rewritten line carried emphasis: {sent[0]!r}"
    # The position-free pause survives the rewrite.
    assert sent[0].startswith('<speak><break time="300ms"/>'), sent[0]
    assert not _URL_RE.search(sent[0])


async def test_unrewritten_generated_line_keeps_its_emphasis() -> None:
    ctx = _with_directives(_ctx())
    await _assert_reconcile_skips(ctx)

    with _ssml_on():
        heard, _ = await _narrate(ctx)

    sent = [t for t in heard.values() if "Buy groceries" in t]
    assert len(sent) == 1, heard
    assert '<emphasis level="strong">open</emphasis>' in sent[0], sent[0]
