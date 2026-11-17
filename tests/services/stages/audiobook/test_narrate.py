"""audiobook_narrate stage — every chunk TTS'd + persisted, with linkage."""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest

from src.clients.schemas.tts import TTSSynthesis
from src.clients.schemas.tts_gateway import BatchStatusResult, SubmitBatchResult
from src.services import tts_storage
from src.services.runs.run_context import RunContext
from src.services.stages.audiobook.narrate import AudiobookNarrateStage
from src.services.stages.base import StagePreconditionError


class _FakeStorage:
    async def initialize(self) -> None:
        pass

    async def store_stage_artifact(
        self, job_id: str, stage_name: str, data: bytes, filename: str
    ) -> str:
        return f"artifacts/{job_id}/{stage_name}/{filename}"

    async def get_presigned_url(
        self, path: str, expires: int | None = None, bucket: str | None = None
    ) -> str | None:
        return f"https://minio.test/{path}"


@pytest.fixture
def fake_storage():
    tts_storage.set_storage_for_tests(_FakeStorage())
    yield
    tts_storage.set_storage_for_tests(None)


def _plan_ctx() -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"voice": {"voice_id": "af_heart", "speed": 1.1}},
        prior_run_outputs={
            "audiobook_plan": {
                "chapters": [
                    {
                        "id": "ch1",
                        "title": "One",
                        "sequence": 0,
                        "chunks": [
                            {"id": "k1", "text": "Alpha.", "sequence": 0},
                            {"id": "k2", "text": "Beta.", "sequence": 1},
                        ],
                    },
                    {
                        "id": "ch2",
                        "title": "Two",
                        "sequence": 1,
                        "chunks": [{"id": "k3", "text": "Gamma.", "sequence": 0}],
                    },
                ],
                "total_chunks": 3,
            }
        },
    )


async def test_narrate_emits_clip_per_chunk_with_linkage(fake_storage: None) -> None:
    fake = AsyncMock()
    fake.synthesize.return_value = TTSSynthesis(
        audio_bytes=b"\x01",
        sample_rate=0,
        duration_seconds=4.0,
        provider="",
        voice="",
        format="wav",
    )
    stage = AudiobookNarrateStage(client=fake)
    out = await stage.run({}, ctx=_plan_ctx())

    assert len(out.clips) == 3
    # every clip carries chapter + chunk linkage for stitch ordering
    by_chunk = {c.chunk_id: c for c in out.clips}
    assert by_chunk["k1"].chapter_id == "ch1" and by_chunk["k1"].chapter_sequence == 0
    assert by_chunk["k3"].chapter_id == "ch2" and by_chunk["k3"].chapter_sequence == 1
    assert all(c.artifact_url.startswith("https://minio.test/") for c in out.clips)
    # voice_id + speed forwarded to TTS
    _, kwargs = fake.synthesize.call_args
    assert kwargs["voice"] == "af_heart"
    assert kwargs["speed"] == 1.1


async def test_narrate_missing_plan_raises(fake_storage: None) -> None:
    stage = AudiobookNarrateStage(client=AsyncMock())
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})
    with pytest.raises(StagePreconditionError, match="no plan/segment output"):
        await stage.run({}, ctx=ctx)


def _cast_ctx() -> RunContext:
    ctx = _plan_ctx()
    # Voice acting present: per-chunk voice/speed for k1,k2; k3 left uncast.
    return RunContext(
        run_id=ctx.run_id,
        deliverable_id=ctx.deliverable_id,
        config_snapshot=ctx.config_snapshot,
        prior_run_outputs={
            **ctx.prior_run_outputs,
            "audiobook_cast": {
                "narrator_voice_id": "af_heart",
                "performances": [
                    {"chunk_id": "k1", "voice_id": "af_bella", "speed": 1.3},
                    {"chunk_id": "k2", "voice_id": "am_adam", "speed": 0.9},
                ],
            },
        },
    )


async def test_narrate_via_gateway_assembles_clips(monkeypatch, fake_storage: None) -> None:
    from src.config import clear_settings_cache

    monkeypatch.setenv("DOWNSTREAM_AUDIOBOOK_NARRATE_VIA_GATEWAY", "true")
    clear_settings_cache()

    gw = AsyncMock()
    gw.submit_batch.return_value = SubmitBatchResult.model_validate(
        {
            "batch_id": "b",
            "jobs": [
                {"ref": "k1", "idempotency_key": "key1", "job_id": "job1", "status": "queued"},
                {"ref": "k2", "idempotency_key": "key2", "job_id": "job2", "status": "queued"},
                {"ref": "k3", "idempotency_key": "key3", "job_id": "job3", "status": "queued"},
            ],
        }
    )
    gw.get_batch.return_value = BatchStatusResult.model_validate(
        {
            "batch_id": "b",
            "jobs": [
                {
                    "idempotency_key": "key1",
                    "job_id": "job1",
                    "status": "done",
                    "result_url": "http://m/1.wav",
                    "duration_seconds": 3.0,
                },
                {
                    "idempotency_key": "key2",
                    "job_id": "job2",
                    "status": "done",
                    "result_url": "http://m/2.wav",
                    "duration_seconds": 4.0,
                },
                {
                    "idempotency_key": "key3",
                    "job_id": "job3",
                    "status": "done",
                    "result_url": "http://m/3.wav",
                    "duration_seconds": 5.0,
                },
            ],
        }
    )
    monkeypatch.setattr(
        "src.services.stages.audiobook.narrate.TTSGatewayClient", lambda *a, **k: gw
    )
    direct = AsyncMock()
    try:
        out = await AudiobookNarrateStage(client=direct).run({}, ctx=_plan_ctx())
    finally:
        clear_settings_cache()

    assert len(out.clips) == 3
    by_chunk = {c.chunk_id: c for c in out.clips}
    # gateway result URLs + preserved chapter linkage
    assert by_chunk["k1"].artifact_url == "http://m/1.wav" and by_chunk["k1"].chapter_id == "ch1"
    assert by_chunk["k3"].artifact_url == "http://m/3.wav" and by_chunk["k3"].chapter_id == "ch2"
    assert by_chunk["k3"].seconds == 5.0
    # the direct synchronous TTS path was NOT used
    direct.synthesize.assert_not_called()
    gw.submit_batch.assert_awaited_once()


async def test_narrate_via_gateway_survives_transient_poll_blip(
    monkeypatch, fake_storage: None
) -> None:
    """A transient transport error on one poll must not tear down narrate — the
    batch keeps draining on the workers, so the loop re-polls and completes."""
    import httpx

    from src.config import clear_settings_cache

    monkeypatch.setenv("DOWNSTREAM_AUDIOBOOK_NARRATE_VIA_GATEWAY", "true")
    clear_settings_cache()

    gw = AsyncMock()
    gw.submit_batch.return_value = SubmitBatchResult.model_validate(
        {
            "batch_id": "b",
            "jobs": [
                {"ref": "k1", "idempotency_key": "key1", "job_id": "job1", "status": "queued"},
                {"ref": "k2", "idempotency_key": "key2", "job_id": "job2", "status": "queued"},
                {"ref": "k3", "idempotency_key": "key3", "job_id": "job3", "status": "queued"},
            ],
        }
    )
    done = BatchStatusResult.model_validate(
        {
            "batch_id": "b",
            "jobs": [
                {
                    "idempotency_key": f"key{i}",
                    "job_id": f"job{i}",
                    "status": "done",
                    "result_url": f"http://m/{i}.wav",
                    "duration_seconds": float(i),
                }
                for i in (1, 2, 3)
            ],
        }
    )
    # first poll blips with a ReadError; the loop must swallow it and re-poll
    gw.get_batch.side_effect = [httpx.ReadError("connection reset"), done]
    monkeypatch.setattr(
        "src.services.stages.audiobook.narrate.TTSGatewayClient", lambda *a, **k: gw
    )
    # keep the post-blip backoff sleep from slowing the test
    monkeypatch.setattr("src.services.stages.audiobook.narrate.asyncio.sleep", AsyncMock())
    try:
        out = await AudiobookNarrateStage(client=AsyncMock()).run({}, ctx=_plan_ctx())
    finally:
        clear_settings_cache()

    assert len(out.clips) == 3  # completed despite the blip
    assert gw.get_batch.await_count == 2  # re-polled after the error


async def test_narrate_uses_per_chunk_cast_voice(fake_storage: None) -> None:
    fake = AsyncMock()
    fake.synthesize.return_value = TTSSynthesis(
        audio_bytes=b"\x01",
        sample_rate=0,
        duration_seconds=4.0,
        provider="",
        voice="",
        format="wav",
    )
    await AudiobookNarrateStage(client=fake).run({}, ctx=_cast_ctx())

    by_text = {c.kwargs["text"]: c.kwargs for c in fake.synthesize.call_args_list}
    # cast voices applied per chunk
    assert by_text["Alpha."]["voice"] == "af_bella" and by_text["Alpha."]["speed"] == 1.3
    assert by_text["Beta."]["voice"] == "am_adam" and by_text["Beta."]["speed"] == 0.9
    # k3 (uncast) falls back to the project voice/speed
    assert by_text["Gamma."]["voice"] == "af_heart" and by_text["Gamma."]["speed"] == 1.1


# ── _lower_line helper (shared by direct + gateway paths, #22) ────────────────


def _gw_fake(n: int = 3) -> AsyncMock:
    gw = AsyncMock()
    gw.submit_batch.return_value = SubmitBatchResult.model_validate(
        {
            "batch_id": "b",
            "jobs": [
                {
                    "ref": f"k{i}",
                    "idempotency_key": f"key{i}",
                    "job_id": f"job{i}",
                    "status": "queued",
                }
                for i in range(1, n + 1)
            ],
        }
    )
    gw.get_batch.return_value = BatchStatusResult.model_validate(
        {
            "batch_id": "b",
            "jobs": [
                {
                    "idempotency_key": f"key{i}",
                    "job_id": f"job{i}",
                    "status": "done",
                    "result_url": f"http://m/{i}.wav",
                    "duration_seconds": float(i),
                }
                for i in range(1, n + 1)
            ],
        }
    )
    return gw


async def test_narrate_via_gateway_closes_client(monkeypatch, fake_storage: None) -> None:
    # The gateway path constructs a TTSGatewayClient per call; on the long-lived
    # multi-replica orchestrator pod it MUST be closed or every gateway narrate
    # leaks an httpx AsyncClient + its connection pool.
    from src.config import clear_settings_cache

    monkeypatch.setenv("DOWNSTREAM_AUDIOBOOK_NARRATE_VIA_GATEWAY", "true")
    clear_settings_cache()
    gw = _gw_fake()
    monkeypatch.setattr(
        "src.services.stages.audiobook.narrate.TTSGatewayClient", lambda *a, **k: gw
    )
    try:
        await AudiobookNarrateStage(client=AsyncMock()).run({}, ctx=_plan_ctx())
    finally:
        clear_settings_cache()
    # BaseServiceClient teardown is close(), NOT aclose() (has aclose == False). The
    # earlier AsyncMock auto-created .aclose and hid that the real method is close.
    gw.close.assert_awaited()


def _direct_ctx(directives: list[dict]) -> RunContext:
    ctx = _plan_ctx()
    return RunContext(
        run_id=ctx.run_id,
        deliverable_id=ctx.deliverable_id,
        config_snapshot=ctx.config_snapshot,
        prior_run_outputs={**ctx.prior_run_outputs, "audiobook_direct": {"directives": directives}},
    )


# ── #22 gateway-path SSML parity ─────────────────────────────────────────────
async def test_narrate_via_gateway_lowers_ssml_when_enabled(
    monkeypatch, fake_storage: None
) -> None:
    from src.config import clear_settings_cache

    monkeypatch.setenv("DOWNSTREAM_AUDIOBOOK_NARRATE_VIA_GATEWAY", "true")
    monkeypatch.setenv("NARRATION_SSML_ENABLED", "true")
    clear_settings_cache()
    gw = _gw_fake()
    monkeypatch.setattr(
        "src.services.stages.audiobook.narrate.TTSGatewayClient", lambda *a, **k: gw
    )
    # k1 gets a leading pause + emphasis (directed); k2/k3 undirected (plain).
    ctx = _direct_ctx([{"chunk_id": "k1", "pause_before_ms": 300, "emphasis_word_indices": [0]}])
    try:
        await AudiobookNarrateStage(client=AsyncMock()).run({}, ctx=ctx)
    finally:
        clear_settings_cache()

    submitted = {c["ref"]: c["text"] for c in gw.submit_batch.call_args.args[1]}
    # Directed chunk carries a <speak> document on the WIRE to the gateway.
    assert submitted["k1"].startswith("<speak>") and '<break time="300ms"/>' in submitted["k1"]
    assert "<emphasis" in submitted["k1"]
    # Undirected chunks stay byte-identical (plain text ⇒ stable idempotency key).
    assert submitted["k2"] == "Beta." and submitted["k3"] == "Gamma."


async def test_narrate_via_gateway_plain_when_ssml_disabled(
    monkeypatch, fake_storage: None
) -> None:
    from src.config import clear_settings_cache

    monkeypatch.setenv("DOWNSTREAM_AUDIOBOOK_NARRATE_VIA_GATEWAY", "true")
    monkeypatch.delenv("NARRATION_SSML_ENABLED", raising=False)
    clear_settings_cache()
    gw = _gw_fake()
    monkeypatch.setattr(
        "src.services.stages.audiobook.narrate.TTSGatewayClient", lambda *a, **k: gw
    )
    # Even with a directive present, the flag-off path submits RAW text (byte-identical).
    ctx = _direct_ctx([{"chunk_id": "k1", "pause_before_ms": 300, "emphasis_word_indices": [0]}])
    try:
        await AudiobookNarrateStage(client=AsyncMock()).run({}, ctx=ctx)
    finally:
        clear_settings_cache()

    submitted = {c["ref"]: c["text"] for c in gw.submit_batch.call_args.args[1]}
    assert submitted == {"k1": "Alpha.", "k2": "Beta.", "k3": "Gamma."}


# ── #6 resilient gateway submit ──────────────────────────────────────────────
async def test_narrate_via_gateway_survives_transient_submit_blip(
    monkeypatch, fake_storage: None
) -> None:
    import httpx

    from src.config import clear_settings_cache

    monkeypatch.setenv("DOWNSTREAM_AUDIOBOOK_NARRATE_VIA_GATEWAY", "true")
    clear_settings_cache()
    gw = _gw_fake()
    ok = gw.submit_batch.return_value
    # First submit blips; the resilient submit retries and succeeds.
    gw.submit_batch.return_value = None
    gw.submit_batch.side_effect = [httpx.ReadError("connection reset"), ok]
    monkeypatch.setattr(
        "src.services.stages.audiobook.narrate.TTSGatewayClient", lambda *a, **k: gw
    )
    monkeypatch.setattr("src.services.stages.audiobook.narrate.asyncio.sleep", AsyncMock())
    try:
        out = await AudiobookNarrateStage(client=AsyncMock()).run({}, ctx=_plan_ctx())
    finally:
        clear_settings_cache()

    assert len(out.clips) == 3  # completed despite the submit blip
    assert gw.submit_batch.await_count == 2  # retried after the error


async def test_narrate_via_gateway_submit_exhausted_raises(monkeypatch, fake_storage: None) -> None:
    import httpx

    from src.config import clear_settings_cache
    from src.services.stages.audiobook.narrate import _GATEWAY_SUBMIT_RETRIES

    monkeypatch.setenv("DOWNSTREAM_AUDIOBOOK_NARRATE_VIA_GATEWAY", "true")
    clear_settings_cache()
    gw = _gw_fake()
    gw.submit_batch.return_value = None
    gw.submit_batch.side_effect = httpx.ReadError("gateway down")
    monkeypatch.setattr(
        "src.services.stages.audiobook.narrate.TTSGatewayClient", lambda *a, **k: gw
    )
    monkeypatch.setattr("src.services.stages.audiobook.narrate.asyncio.sleep", AsyncMock())
    try:
        with pytest.raises(StagePreconditionError, match="gateway submit failed"):
            await AudiobookNarrateStage(client=AsyncMock()).run({}, ctx=_plan_ctx())
    finally:
        clear_settings_cache()
    assert gw.submit_batch.await_count == _GATEWAY_SUBMIT_RETRIES  # bounded, then raised


async def test_narrate_via_gateway_submit_invalid_url_is_handled(
    monkeypatch, fake_storage: None
) -> None:
    """A misconfigured gateway base URL raises httpx.InvalidURL — which inherits
    Exception, NOT HTTPError — so it must be caught, not escape raw."""
    import httpx

    from src.config import clear_settings_cache

    monkeypatch.setenv("DOWNSTREAM_AUDIOBOOK_NARRATE_VIA_GATEWAY", "true")
    clear_settings_cache()
    gw = _gw_fake()
    gw.submit_batch.return_value = None
    gw.submit_batch.side_effect = httpx.InvalidURL("no host in URL")
    monkeypatch.setattr(
        "src.services.stages.audiobook.narrate.TTSGatewayClient", lambda *a, **k: gw
    )
    monkeypatch.setattr("src.services.stages.audiobook.narrate.asyncio.sleep", AsyncMock())
    try:
        with pytest.raises(StagePreconditionError, match="gateway submit failed"):
            await AudiobookNarrateStage(client=AsyncMock()).run({}, ctx=_plan_ctx())
    finally:
        clear_settings_cache()
    # Permanent config error: no point retrying a URL that cannot become valid.
    assert gw.submit_batch.await_count == 1


def _status_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://tts-gateway/api/tts/batches")
    return httpx.HTTPStatusError(
        f"{status}", request=request, response=httpx.Response(status, request=request)
    )


async def test_narrate_via_gateway_submit_permanent_4xx_fails_fast(
    monkeypatch, fake_storage: None
) -> None:
    """A 422 (bad chunk payload) is deterministic — retrying burns ~6s of sleeps
    and fails anyway. It must be attempted exactly once."""
    from src.config import clear_settings_cache

    monkeypatch.setenv("DOWNSTREAM_AUDIOBOOK_NARRATE_VIA_GATEWAY", "true")
    clear_settings_cache()
    gw = _gw_fake()
    gw.submit_batch.return_value = None
    gw.submit_batch.side_effect = _status_error(422)
    monkeypatch.setattr(
        "src.services.stages.audiobook.narrate.TTSGatewayClient", lambda *a, **k: gw
    )
    sleep = AsyncMock()
    monkeypatch.setattr("src.services.stages.audiobook.narrate.asyncio.sleep", sleep)
    try:
        with pytest.raises(StagePreconditionError, match="gateway submit failed"):
            await AudiobookNarrateStage(client=AsyncMock()).run({}, ctx=_plan_ctx())
    finally:
        clear_settings_cache()
    assert gw.submit_batch.await_count == 1
    assert sleep.await_count == 0  # no backoff burned on a permanent error


async def test_narrate_via_gateway_submit_retries_5xx(monkeypatch, fake_storage: None) -> None:
    """A 503 IS transient (gateway rolling/overloaded) — it must be retried."""
    from src.config import clear_settings_cache

    monkeypatch.setenv("DOWNSTREAM_AUDIOBOOK_NARRATE_VIA_GATEWAY", "true")
    clear_settings_cache()
    gw = _gw_fake()
    ok = gw.submit_batch.return_value
    gw.submit_batch.return_value = None
    gw.submit_batch.side_effect = [_status_error(503), ok]
    monkeypatch.setattr(
        "src.services.stages.audiobook.narrate.TTSGatewayClient", lambda *a, **k: gw
    )
    monkeypatch.setattr("src.services.stages.audiobook.narrate.asyncio.sleep", AsyncMock())
    try:
        out = await AudiobookNarrateStage(client=AsyncMock()).run({}, ctx=_plan_ctx())
    finally:
        clear_settings_cache()

    assert len(out.clips) == 3
    assert gw.submit_batch.await_count > 1


# ── an UNMEASURED gateway duration must survive the client boundary (tts_gateway#38)


def test_gateway_job_status_carries_duration_unmeasured() -> None:
    """The hand-mirrored client schema must KEEP the gateway's unmeasured flag.

    tts_gateway#38 stopped fabricating ``duration_seconds=0.0`` for audio it could
    not measure: it now persists SQL NULL and sets ``duration_unmeasured``. Pydantic
    ``extra="ignore"`` drops an undeclared field SILENTLY — no error, no log — so
    without the field here the producer-side fix is inert.
    """
    from src.clients.schemas.tts_gateway import GatewayJobStatus

    job = GatewayJobStatus.model_validate(
        {
            "job_id": "job1",
            "idempotency_key": "key1",
            "status": "done",
            "result_url": "http://m/1.wav",
            "duration_seconds": None,
            "duration_unmeasured": True,
        }
    )
    assert job.duration_unmeasured is True
    assert job.duration_seconds is None
    # and a measured job keeps the legacy shape (default False)
    measured = GatewayJobStatus.model_validate(
        {
            "job_id": "job2",
            "idempotency_key": "key2",
            "status": "done",
            "result_url": "http://m/2.wav",
            "duration_seconds": 3.0,
        }
    )
    assert measured.duration_unmeasured is False


async def test_narrate_does_not_fabricate_a_zero_for_an_unmeasured_clip(
    monkeypatch, fake_storage: None
) -> None:
    """An unmeasured gateway duration must NOT collapse to 0.0.

    The audio is fine and is the valuable thing, so the clip ships — but the GAP
    has to be expressible (``seconds=None`` + ``duration_unmeasured``) and LOUD (a
    stage note, which api/routes/runs/_shared.py turns into a `degraded` run).
    A fabricated 0.0 would 422 the compiler (``AudioClip.seconds`` is ``gt=0``)
    and misplace every downstream m4b chapter marker.
    """
    from src.config import clear_settings_cache

    monkeypatch.setenv("DOWNSTREAM_AUDIOBOOK_NARRATE_VIA_GATEWAY", "true")
    clear_settings_cache()

    gw = AsyncMock()
    gw.submit_batch.return_value = SubmitBatchResult.model_validate(
        {
            "batch_id": "b",
            "jobs": [
                {"ref": "k1", "idempotency_key": "key1", "job_id": "job1", "status": "queued"},
                {"ref": "k2", "idempotency_key": "key2", "job_id": "job2", "status": "queued"},
                {"ref": "k3", "idempotency_key": "key3", "job_id": "job3", "status": "queued"},
            ],
        }
    )
    gw.get_batch.return_value = BatchStatusResult.model_validate(
        {
            "batch_id": "b",
            "jobs": [
                {
                    "idempotency_key": "key1",
                    "job_id": "job1",
                    "status": "done",
                    "result_url": "http://m/1.wav",
                    # the gateway could not measure this one
                    "duration_seconds": None,
                    "duration_unmeasured": True,
                },
                {
                    "idempotency_key": "key2",
                    "job_id": "job2",
                    "status": "done",
                    "result_url": "http://m/2.wav",
                    "duration_seconds": 4.0,
                },
                {
                    "idempotency_key": "key3",
                    "job_id": "job3",
                    "status": "done",
                    "result_url": "http://m/3.wav",
                    "duration_seconds": 5.0,
                },
            ],
        }
    )
    monkeypatch.setattr(
        "src.services.stages.audiobook.narrate.TTSGatewayClient", lambda *a, **k: gw
    )
    try:
        out = await AudiobookNarrateStage(client=AsyncMock()).run({}, ctx=_plan_ctx())
    finally:
        clear_settings_cache()

    by_chunk = {c.chunk_id: c for c in out.clips}
    # the audio still ships — dropping it would be worse than a misplaced marker
    assert by_chunk["k1"].artifact_url == "http://m/1.wav"
    # ...but its length is UNKNOWN, not zero
    assert by_chunk["k1"].seconds is None
    assert by_chunk["k1"].duration_unmeasured is True
    # measured neighbours are untouched
    assert by_chunk["k2"].seconds == 4.0 and by_chunk["k2"].duration_unmeasured is False
    assert by_chunk["k3"].seconds == 5.0
    # and the gap is LOUD: a note ⇒ the run reports `degraded`
    assert out.notes, "an unmeasured clip must degrade the run, not pass silently"
    assert any("k1" in n for n in out.notes)


async def test_narrate_via_gateway_measured_run_emits_no_notes(
    monkeypatch, fake_storage: None
) -> None:
    """Control: a fully-measured batch must stay byte-identical — NO note, so a
    healthy run is not falsely reported `degraded`."""
    from src.config import clear_settings_cache

    monkeypatch.setenv("DOWNSTREAM_AUDIOBOOK_NARRATE_VIA_GATEWAY", "true")
    clear_settings_cache()

    gw = AsyncMock()
    gw.submit_batch.return_value = SubmitBatchResult.model_validate(
        {
            "batch_id": "b",
            "jobs": [
                {"ref": "k1", "idempotency_key": "key1", "job_id": "job1", "status": "queued"},
                {"ref": "k2", "idempotency_key": "key2", "job_id": "job2", "status": "queued"},
                {"ref": "k3", "idempotency_key": "key3", "job_id": "job3", "status": "queued"},
            ],
        }
    )
    gw.get_batch.return_value = BatchStatusResult.model_validate(
        {
            "batch_id": "b",
            "jobs": [
                {
                    "idempotency_key": f"key{i}",
                    "job_id": f"job{i}",
                    "status": "done",
                    "result_url": f"http://m/{i}.wav",
                    "duration_seconds": float(i),
                }
                for i in (1, 2, 3)
            ],
        }
    )
    monkeypatch.setattr(
        "src.services.stages.audiobook.narrate.TTSGatewayClient", lambda *a, **k: gw
    )
    try:
        out = await AudiobookNarrateStage(client=AsyncMock()).run({}, ctx=_plan_ctx())
    finally:
        clear_settings_cache()

    assert out.notes == []
    assert all(c.seconds is not None and c.duration_unmeasured is False for c in out.clips)


async def _narrate_with_failed_job(monkeypatch, error: str):
    from src.config import clear_settings_cache

    monkeypatch.setenv("DOWNSTREAM_AUDIOBOOK_NARRATE_VIA_GATEWAY", "true")
    clear_settings_cache()
    gw = AsyncMock()
    gw.submit_batch.return_value = SubmitBatchResult.model_validate(
        {
            "batch_id": "b",
            "jobs": [
                {
                    "ref": f"k{i}",
                    "idempotency_key": f"key{i}",
                    "job_id": f"job{i}",
                    "status": "queued",
                }
                for i in (1, 2, 3)
            ],
        }
    )
    gw.get_batch.return_value = BatchStatusResult.model_validate(
        {
            "batch_id": "b",
            "jobs": [
                {"idempotency_key": "key1", "job_id": "job1", "status": "failed", "error": error},
                {"idempotency_key": "key2", "job_id": "job2", "status": "failed", "error": error},
                {"idempotency_key": "key3", "job_id": "job3", "status": "failed", "error": error},
            ],
        }
    )
    monkeypatch.setattr(
        "src.services.stages.audiobook.narrate.TTSGatewayClient", lambda *a, **k: gw
    )
    try:
        return await AudiobookNarrateStage(client=AsyncMock()).run({}, ctx=_plan_ctx())
    finally:
        clear_settings_cache()


# The worker's exact SynthConfigError text (tts_gateway worker/worker.py, #46).
_WORKER_402_ERROR = (
    "synth refused with HTTP 402 (customer configuration): the tenant must configure "
    "its own TTS provider before use — tenant must use its own configured tts provider"
)


async def test_gateway_job_failed_on_402_surfaces_byo_provider_required(
    monkeypatch, fake_storage: None
) -> None:
    """#1106: a job the worker failed TERMINALLY on modality_gateway's 402 is the
    tenant's byo_required refusal — structured, not the worker's prose."""
    from src.models.failure_class import CUSTOMER_INPUT, attribute_exception, failure_detail_for
    from src.services.providers.errors import SlotNotConfiguredError

    with pytest.raises(SlotNotConfiguredError) as info:
        await _narrate_with_failed_job(monkeypatch, _WORKER_402_ERROR)

    assert attribute_exception(info.value) == CUSTOMER_INPUT
    assert failure_detail_for(info.value) == {
        "code": "byo_provider_required",
        "modality": "tts",
        "slots": "tts",
        "message": "Configure your own TTS provider to continue",
    }
    assert info.value.upstream_detail == _WORKER_402_ERROR


async def test_an_ordinary_gateway_job_failure_stays_a_precondition_error(
    monkeypatch, fake_storage: None
) -> None:
    from src.services.providers.errors import SlotNotConfiguredError

    with pytest.raises(StagePreconditionError, match="failed on the gateway") as info:
        await _narrate_with_failed_job(monkeypatch, "kokoro crashed: HTTP 500")
    assert not isinstance(info.value, SlotNotConfiguredError)


def test_vendored_worker_402_prefix_matches_the_worker_text() -> None:
    from src.clients.tts_gateway_client import BYO_REFUSAL_ERROR_PREFIX, is_byo_refusal

    assert _WORKER_402_ERROR.startswith(BYO_REFUSAL_ERROR_PREFIX)
    assert is_byo_refusal(_WORKER_402_ERROR)
    assert not is_byo_refusal(None) and not is_byo_refusal("HTTP 402 elsewhere")


def test_narrate_stage_timeout_outlasts_the_gateway_poll_budget() -> None:
    """At 3600s the stage timeout killed a full novel mid-poll while the gateway
    budget (7200s) still had an hour to run."""
    from src.services.blueprints.audiobook import AUDIOBOOK_BLUEPRINT
    from src.services.stages.audiobook.narrate import _GATEWAY_POLL_BUDGET_SECONDS

    spec = next(s for s in AUDIOBOOK_BLUEPRINT.stages if s.name == "audiobook_narrate")
    assert spec.default_timeout_seconds is not None
    assert spec.default_timeout_seconds > _GATEWAY_POLL_BUDGET_SECONDS + 120


async def test_narrate_off_cast_uses_catalog_checked_narrator_and_clamped_speed(
    fake_storage: None,
) -> None:
    """voice_acting off: cast emits no performances but a CATALOG-CHECKED
    narrator_voice_id. Narrate must use it (not the raw, possibly unknown project
    voice) and clamp speed into [0.5, 2.0] — the gateway ChunkIn 422s outside it,
    non-retryably, failing the whole book."""
    base = _plan_ctx()
    ctx = RunContext(
        run_id=base.run_id,
        deliverable_id=base.deliverable_id,
        config_snapshot={"voice": {"voice_id": "not_a_kokoro_voice", "speed": 3.5}},
        prior_run_outputs={
            **base.prior_run_outputs,
            "audiobook_cast": {"casting": [], "performances": [], "narrator_voice_id": "af_heart"},
            "audiobook_direct": {"directives": [{"chunk_id": "k2", "rate_hint": 0.1}]},
        },
    )
    fake = AsyncMock()
    fake.synthesize.return_value = TTSSynthesis(
        audio_bytes=b"\x01",
        sample_rate=0,
        duration_seconds=4.0,
        provider="",
        voice="",
        format="wav",
    )
    out = await AudiobookNarrateStage(client=fake).run({}, ctx=ctx)

    by_chunk = {c.chunk_id: c for c in out.clips}
    assert {c.voice for c in out.clips} == {"af_heart"}
    assert by_chunk["k1"].speed == 2.0  # project speed 3.5 clamped
    assert by_chunk["k2"].speed == 0.5  # director rate_hint 0.1 clamped
