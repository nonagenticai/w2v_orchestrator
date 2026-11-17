"""CompileStageBase gates word-timed captions on burned_in OR overlay, and sends
the chosen mode to the compiler as ``captions_render`` (ORCH-4.2, overlay).

Wiring tests for the overlay soft-caption mode. Before overlay, the gate only
admitted ``captions_mode=burned_in``; overlay reuses the SAME mode-agnostic cue
build and lets the compiler decide burn-vs-mux off ``captions_render``. These pin:

1. overlay is admitted by the gate ⇒ cues are built and sent, with
   ``captions_render="overlay"`` on the compile call;
2. burned_in still builds+sends cues with ``captions_render="burned_in"``;
3. a mode outside {burned_in, overlay} (``none``) builds NO cues (``captions`` is
   None) — the flag alone never turns captions on.

No DB: storage, the ASR client, and the cue builder are stubbed, so the assertion
is purely on the kwargs handed to the compile client.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from src.services import tts_storage
from src.services.stages.base import ArtifactUrl, StageInput, StageOutput
from src.services.stages.primitives import compile_stage as compile_stage_mod
from src.services.stages.primitives.compile_stage import CompileStageBase
from src.services.stages.shared.step_ledger import NullStepLedger
from src.services.word_timed_captions import ClipRef

pytestmark = pytest.mark.asyncio


class _Input(StageInput):
    pass


class _Output(StageOutput):
    artifact_url: ArtifactUrl
    duration_seconds: float


class _Stage(CompileStageBase[_Input, _Output]):
    name = "test_compile_captions_render"
    input_model = _Input
    output_model = _Output

    def __init__(self, client: Any) -> None:
        super().__init__(client=client)

    def build_shots(self, ctx: Any) -> list[dict[str, Any]]:
        return [{"id": "s1", "line_id": "l1", "seconds": 2.0, "caption": "static"}]

    def build_output(self, result: dict[str, Any], ctx: Any) -> _Output:
        return _Output(
            artifact_url=str(result["artifact_url"]),
            duration_seconds=float(result["duration_seconds"]),
        )

    def compile_tags(self, ctx: Any) -> dict[str, str]:
        return {}  # avoid touching ctx.run_id in this unit test

    def caption_clip_refs(self, ctx: Any) -> dict[str, ClipRef]:
        return {
            "l1": ClipRef(
                line_id="l1",
                text="hi",
                narrate_stage="narrate",
                clip_filename="c.wav",
                seconds=2.0,
            )
        }


class _FakeStorage:
    async def resign_presigned_url(self, url: str) -> str:
        return url


_FAKE_CAPS = [{"shot_index": 0, "cues": [{"text": "hi", "start": 0.0, "end": 1.0}]}]


class _SpyAsr:
    """Stands in for TranscriptionWorkerClient; records that it was closed so the
    caption path is proven not to leak an httpx pool."""

    def __init__(self) -> None:
        self.closed = 0

    async def close(self) -> None:  # BaseServiceClient teardown is close(), not aclose()
        self.closed += 1


async def _run_stage(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    karaoke: bool = False,
    mood: str | None = None,
    stage_cls: type[_Stage] = _Stage,
) -> dict[str, Any]:
    """Run the compile tail with the caption kill-switch ON and ``captions_mode``
    set to ``mode``; return the kwargs handed to the compile client. ``karaoke``
    sets the PIPELINE_CAPTIONS_KARAOKE flag; ``mood`` seeds the run-level mood token."""
    captured: dict[str, Any] = {}

    async def _compile(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"artifact_url": "http://m/out.mp4", "duration_seconds": 2.0}

    client = AsyncMock()
    client.compile = AsyncMock(side_effect=_compile)

    # Kill-switch ON (captions_default_on_showcase); grade flag OFF (read on the
    # same handle path). Patch only the compile_stage module's get_settings — the
    # downstream helpers keep the real settings.
    fake_settings = SimpleNamespace(
        pipeline=SimpleNamespace(
            captions_default_on_showcase=True,
            grade_default_on_showcase=False,
            captions_karaoke=karaoke,
        )
    )
    monkeypatch.setattr(compile_stage_mod, "get_settings", lambda: fake_settings)

    # Stub the cue build + its collaborators so no ASR/storage/DB is touched.
    async def _fake_build(**_kwargs: Any) -> list[dict[str, Any]]:
        return _FAKE_CAPS

    async def _fake_get_storage() -> Any:
        return _FakeStorage()

    monkeypatch.setattr(compile_stage_mod, "build_shot_captions", _fake_build)
    monkeypatch.setattr(compile_stage_mod, "_get_storage", _fake_get_storage)
    asr_spy = _SpyAsr()
    monkeypatch.setattr(compile_stage_mod, "TranscriptionWorkerClient", lambda: asr_spy)

    tts_storage.set_storage_for_tests(cast(Any, _FakeStorage()))
    try:
        await stage_cls(client).handle(
            _Input(),
            ctx=cast(
                Any,
                SimpleNamespace(
                    config_snapshot={
                        "deliverable": {"captions_mode": mode},
                        **({"mood": mood} if mood is not None else {}),
                    },
                    prior_output=lambda _stage: None,
                    run_id="00000000-0000-0000-0000-000000000000",
                    render_idempotency_discriminator="",
                    step_ledger=lambda _stage: NullStepLedger(),
                ),
            ),
        )
    finally:
        tts_storage.set_storage_for_tests(None)
    captured["_asr"] = asr_spy
    return captured


async def test_overlay_is_admitted_and_mode_is_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    """overlay builds cues (same path as burned_in) and rides captions_render."""
    captured = await _run_stage("overlay", monkeypatch)
    assert captured["captions"] == _FAKE_CAPS
    assert captured["captions_render"] == "overlay"


async def test_burned_in_still_builds_and_sends_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = await _run_stage("burned_in", monkeypatch)
    assert captured["captions"] == _FAKE_CAPS
    assert captured["captions_render"] == "burned_in"


async def test_none_mode_builds_no_cues(monkeypatch: pytest.MonkeyPatch) -> None:
    """The flag alone never turns captions on — a none-mode deliverable sends no
    cues, so the compiler (which drops captions_render without cues) burns nothing."""
    captured = await _run_stage("none", monkeypatch)
    assert captured["captions"] is None


async def test_no_mood_forwards_none_caption_mood(monkeypatch: pytest.MonkeyPatch) -> None:
    """No mood in the snapshot ⇒ ``caption_mood`` is None (compiler uses defaults)."""
    captured = await _run_stage("burned_in", monkeypatch)
    assert captured["caption_mood"] is None


async def test_karaoke_flag_is_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    """PIPELINE_CAPTIONS_KARAOKE flows to the compile call as ``captions_karaoke``;
    the default is False so captions stay plain unless the flag is flipped."""
    on = await _run_stage("burned_in", monkeypatch, karaoke=True)
    assert on["captions_karaoke"] is True
    off = await _run_stage("burned_in", monkeypatch, karaoke=False)
    assert off["captions_karaoke"] is False


class _SilentStage(_Stage):
    """A captions-only stage: no narration clip refs, only shot-spanning cues."""

    def build_shots(self, ctx: Any) -> list[dict[str, Any]]:
        return [{"id": "s1", "line_id": "l1", "seconds": 2.0}]

    def caption_clip_refs(self, ctx: Any) -> dict[str, ClipRef]:
        return {}

    def static_caption_cues(self, ctx: Any, shots: list[dict[str, Any]]) -> Any:
        return _STATIC_CAPS


_STATIC_CAPS = [{"shot_index": 0, "cues": [{"text": "hi", "start": 0.0, "end": 2.0, "words": []}]}]


async def test_overlay_without_clip_refs_ships_the_static_cues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A captions-only overlay film has nothing to word-time; its static cues still
    reach the compiler, rendered as the soft overlay (never burned in)."""
    captured = await _run_stage("overlay", monkeypatch, stage_cls=_SilentStage)
    assert captured["captions"] == _STATIC_CAPS
    assert captured["captions_render"] == "overlay"


async def test_burned_in_never_consults_the_static_cues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = await _run_stage("burned_in", monkeypatch, stage_cls=_SilentStage)
    assert captured["captions"] is None
