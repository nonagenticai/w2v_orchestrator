"""CompileStageBase.handle re-signs shot asset URLs before the compile call.

Wiring test for the shared compile tail: a minimal concrete subclass builds one
shot carrying a stale presigned URL; the mocked video-compile client must receive
the *re-signed* URL (proving ``refresh_shot_asset_urls`` runs between build_shots
and the POST). Storage is injected via ``set_storage_for_tests``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import uuid7

import pytest

from src.services import tts_storage
from src.services.stages.base import ArtifactUrl, StageInput, StageOutput
from src.services.stages.primitives.compile_stage import CompileStageBase


def _fake_ctx(config_snapshot: dict[str, Any]) -> Any:
    """A RunContext stand-in for the compile tail: config + the prior-output reader.

    The real RunContext always exposes ``prior_output``; a fake that omits it would
    make every opt-in tail read (title cards, the A18 disclaimer card) blow up rather
    than resolve to "not requested".
    """
    return SimpleNamespace(
        run_id=uuid7(),
        render_idempotency_discriminator="",
        config_snapshot=config_snapshot,
        prior_output=lambda _stage: None,
    )


class _Input(StageInput):
    pass


class _Output(StageOutput):
    artifact_url: ArtifactUrl
    duration_seconds: float


class _Stage(CompileStageBase[_Input, _Output]):
    name = "test_compile"
    input_model = _Input
    output_model = _Output

    def ken_burns_local_only(self, ctx: Any) -> bool:
        # Pins the video_production RECORDER path, which the base no longer takes.
        return False

    def __init__(self, client: Any, shots: list[dict[str, Any]]) -> None:
        super().__init__(client=client)
        self._shots = shots

    def build_shots(self, ctx: Any) -> list[dict[str, Any]]:
        return self._shots

    def build_output(self, result: dict[str, Any], ctx: Any) -> _Output:
        return _Output(
            artifact_url=str(result["artifact_url"]),
            duration_seconds=float(result["duration_seconds"]),
        )

    def compile_tags(self, ctx: Any) -> dict[str, str]:
        return {}  # avoid touching ctx.config_snapshot/run_id in this unit test


class _FakeStorage:
    async def resign_presigned_url(self, url: str) -> str:
        return f"{url}&fresh"


@pytest.mark.asyncio
async def test_handle_resigns_shot_urls_before_compile() -> None:
    captured: dict[str, Any] = {}

    async def _compile(
        *,
        shots: list[dict[str, Any]],
        tags: dict[str, str],
        music: dict[str, Any] | None = None,
        ambient: list[dict[str, Any]] | None = None,
        mastering: dict[str, Any] | None = None,
        finish: dict[str, Any] | None = None,
        captions: list[dict[str, Any]] | None = None,
        captions_render: str | None = None,
        caption_mood: str | None = None,
        captions_karaoke: bool = False,
        caption_look: dict[str, object] | None = None,
        cadence: int | None = None,
        emit_shot_timeline: bool = False,
        timeout: float = 1200.0,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        captured["shots"] = shots
        return {
            "artifact_url": "http://m/w2v-compiled-videos/out.mp4?sig=x",
            "duration_seconds": 5.0,
        }

    client = AsyncMock()
    client.compile = AsyncMock(side_effect=_compile)
    shots = [
        {
            "id": "s1",
            "line_id": "l1",
            "seconds": 1.0,
            "image_url": "http://m/w2v-orchestrator/i.png?sig=old",
        }
    ]
    stage = _Stage(client, shots)

    # Minimal fakes: _FakeStorage only implements resign; ctx needs an empty
    # config_snapshot (no title_card block → the opt-in injection is a no-op) plus
    # the prior_output the real RunContext always exposes (no stage output ⇒ every
    # opt-in tail read, e.g. the A18 disclaimer card, resolves to None).
    # build_shots/build_output ignore ctx, compile_tags is overridden. cast past
    # the _StorageLike shape — same convention as test_storage_object_exists.
    tts_storage.set_storage_for_tests(cast(Any, _FakeStorage()))
    try:
        out = await stage.handle(_Input(), ctx=_fake_ctx({}))
    finally:
        tts_storage.set_storage_for_tests(None)

    # The client saw the re-signed URL, not the stale one. TWO ``&fresh`` suffixes:
    # the tail re-signs once BEFORE apply_ken_burns (so its own fetch gets a live
    # signature) and once after (so the clips it just stored are re-signed too).
    assert (
        captured["shots"][0]["image_url"] == "http://m/w2v-orchestrator/i.png?sig=old&fresh&fresh"
    )
    assert str(out.artifact_url).startswith("http")
    assert out.duration_seconds == 5.0


@pytest.mark.asyncio
async def test_ken_burns_receives_freshly_resigned_urls() -> None:
    """THE DEFECT: ``apply_ken_burns`` ran BEFORE the only re-sign, so on any
    paused/iterated run its own ``_fetch_own_asset`` hit an expired signature,
    403'd, and the per-shot blanket ``except Exception`` soft-failed EVERY shot —
    motion silently did nothing on exactly the runs that need it. The tail must
    hand ken_burns already-refreshed URLs, and re-sign again afterwards so the
    clips ken_burns itself stored also get a fresh TTL.
    """
    from src.services.stages.primitives import compile_stage as cs_mod

    seen_by_ken_burns: list[str] = []

    async def _fake_ken_burns(
        ctx: Any, shots: list[dict[str, Any]], **_kw: Any
    ) -> list[dict[str, Any]]:
        seen_by_ken_burns.extend(str(s["image_url"]) for s in shots)
        # Stand in for a recorded clip: a URL that exists only AFTER this pass, so
        # the second re-sign is the only thing that can freshen it.
        return [{**s, "video_url": "http://m/w2v-orchestrator/kb.mp4?sig=minted"} for s in shots]

    captured: dict[str, Any] = {}

    async def _compile(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"artifact_url": "http://m/out.mp4", "duration_seconds": 1.0}

    client = AsyncMock()
    client.compile = AsyncMock(side_effect=_compile)
    stage = _Stage(
        client,
        [
            {
                "id": "s1",
                "line_id": "l1",
                "seconds": 1.0,
                "image_url": "http://m/w2v-orchestrator/i.png?sig=old",
            }
        ],
    )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cs_mod, "apply_ken_burns", _fake_ken_burns)
        tts_storage.set_storage_for_tests(cast(Any, _FakeStorage()))
        try:
            await stage.handle(_Input(), ctx=_fake_ctx({}))
        finally:
            tts_storage.set_storage_for_tests(None)

    # Ordering proof: ken_burns saw a URL that had ALREADY been through re-sign.
    assert seen_by_ken_burns == ["http://m/w2v-orchestrator/i.png?sig=old&fresh"]
    # And the clip ken_burns minted afterwards was still re-signed before compile.
    assert captured["shots"][0]["video_url"] == "http://m/w2v-orchestrator/kb.mp4?sig=minted&fresh"


@pytest.mark.asyncio
async def test_handle_passes_stable_idempotency_key() -> None:
    """The compile submit carries f"{run_id}:{stage}" — stable across a resumable
    -timeout re-dispatch so the compiler dedupes a re-submit instead of running a
    second paid render."""
    captured: dict[str, Any] = {}

    async def _compile(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"artifact_url": "http://m/out.mp4", "duration_seconds": 1.0}

    client = AsyncMock()
    client.compile = AsyncMock(side_effect=_compile)
    stage = _Stage(client, [{"id": "s1", "line_id": "l1", "seconds": 1.0}])
    ctx = _fake_ctx({})
    tts_storage.set_storage_for_tests(cast(Any, _FakeStorage()))
    try:
        await stage.handle(_Input(), ctx=ctx)
    finally:
        tts_storage.set_storage_for_tests(None)
    assert captured["idempotency_key"] == f"{ctx.run_id}:test_compile"


@pytest.mark.asyncio
async def test_handle_splices_title_cards_around_shots(monkeypatch) -> None:
    """An opt-in intro/outro card brackets build_shots() output before compile."""
    from src.services.stages.primitives import compile_stage as cs_mod

    intro = {"id": "ti", "line_id": "__title_intro__", "seconds": 3.0, "video_url": "http://i"}
    outro = {"id": "to", "line_id": "__title_outro__", "seconds": 3.0, "video_url": "http://o"}
    monkeypatch.setattr(cs_mod, "render_title_card_plan", AsyncMock(return_value=(intro, outro)))

    captured: dict[str, Any] = {}

    async def _compile(
        *,
        shots: list[dict[str, Any]],
        tags: dict[str, str],
        music: dict[str, Any] | None = None,
        ambient: list[dict[str, Any]] | None = None,
        mastering: dict[str, Any] | None = None,
        finish: dict[str, Any] | None = None,
        captions: list[dict[str, Any]] | None = None,
        captions_render: str | None = None,
        caption_mood: str | None = None,
        captions_karaoke: bool = False,
        caption_look: dict[str, object] | None = None,
        cadence: int | None = None,
        emit_shot_timeline: bool = False,
        timeout: float = 1200.0,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        captured["shots"] = shots
        return {"artifact_url": "http://m/out.mp4", "duration_seconds": 9.0}

    client = AsyncMock()
    client.compile = AsyncMock(side_effect=_compile)
    body = [{"id": "s1", "line_id": "l1", "seconds": 2.0}]
    stage = _Stage(client, body)

    tts_storage.set_storage_for_tests(cast(Any, _FakeStorage()))
    try:
        await stage.handle(
            _Input(),
            ctx=_fake_ctx({"title_card": {"enabled": True}}),
        )
    finally:
        tts_storage.set_storage_for_tests(None)

    assert [s["line_id"] for s in captured["shots"]] == ["__title_intro__", "l1", "__title_outro__"]
