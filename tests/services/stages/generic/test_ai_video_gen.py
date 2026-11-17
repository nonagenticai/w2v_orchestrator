"""Tests for the ai_video_gen stage handler.

Sibling of test_ai_image_gen. Covers the opt-in skip contract
(StagePreconditionError) and the downstream-response gate
(StageDownstreamResponseError) added alongside the skipped-accounting work.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from w2v_common.stages import StageRetriesExhaustedError

from src.clients.schemas.ai_content_generation import (
    AcgVideoGenerationResult,
    AcgVideoJobStatus,
    AcgVideoProviderInfo,
    AcgVideoProviderList,
)
from src.services.runs.run_context import RunContext
from src.services.stages.base import (
    StageDownstreamResponseError,
    StageSkippedError,
)
from src.services.stages.generic.ai_video_gen import AiVideoGenStage


def _http_error(status: int) -> httpx.HTTPStatusError:
    """A real ACG HTTP failure, as ``BaseServiceClient._request`` raises it.

    ``stages/upstream.classify`` reads the status off the response, so the status
    is what decides permanent-vs-transient — not the exception type.
    """
    request = httpx.Request("GET", "https://acg.test/api/content/videos/providers")
    response = httpx.Response(status, request=request, json={"error": {"message": "nope"}})
    return httpx.HTTPStatusError(f"{status}", request=request, response=response)


def _providers(*names: str, available: bool = True) -> AcgVideoProviderList:
    """Build a provider-registry listing as ACG's /providers route returns it."""
    return AcgVideoProviderList(
        providers=[AcgVideoProviderInfo(name=n, available=available) for n in names],
        default_provider=names[0] if names else None,
    )


def _make_client(
    *,
    job_id: str = "job-1",
    status: str = "completed",
    result: AcgVideoGenerationResult | None = None,
    provider_list: AcgVideoProviderList | None = None,
    provider_probe_error: BaseException | None = None,
) -> AsyncMock:
    """Mock the ACG client's real two-step video contract.

    ``wait_for_generation`` returns the THIN status envelope (no video_url);
    ``get_generation_result`` returns the FAT result (video_url + duration). This
    mirrors the real service — the previous single-fat-dict mock hid a prod bug
    where ``ai_video_gen`` read ``video_url`` off the status poll (where it never
    exists) and always tripped its own downstream-response gate.

    ``list_video_providers`` is the capability probe the stage runs BEFORE
    submitting; by default it reports a registered ``stub`` provider so the
    pre-existing happy-path tests keep exercising the generation path.
    """
    client = AsyncMock()
    client.list_video_providers = AsyncMock(
        side_effect=provider_probe_error,
        return_value=provider_list if provider_list is not None else _providers("stub"),
    )
    client.start_generation = AsyncMock(return_value=job_id)
    client.wait_for_generation = AsyncMock(
        return_value=AcgVideoJobStatus(
            job_id=job_id,
            status=status,
            provider="stub",
            result_url=f"/api/content/videos/jobs/{job_id}/result",
        )
    )
    client.get_generation_result = AsyncMock(
        return_value=result
        or AcgVideoGenerationResult(
            job_id=job_id,
            status=status,
            provider="stub",
            video_url="https://minio.test/buckets/video/asset-1.mp4",
            actual_duration_s=5.0,
        )
    )
    return client


def _ctx(cfg: dict) -> RunContext:
    return RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot=cfg)


@pytest.mark.asyncio
async def test_happy_path_single_prompt() -> None:
    client = _make_client()
    stage = AiVideoGenStage(client=client)
    ctx = _ctx({"ai_video_gen": {"prompt": "a calm sea"}})

    out = await stage.run({}, ctx=ctx)

    assert len(out.videos) == 1
    vid = out.videos[0]
    assert vid.prompt == "a calm sea"
    # video results carry no asset_id (image-only concept)
    assert vid.asset_id is None
    # video_url + duration come from the RESULT fetch, not the status poll
    assert vid.video_url is not None
    assert vid.video_url.endswith("/asset-1.mp4")
    assert vid.duration_s == 5.0
    assert vid.provider_used == "stub"
    assert vid.status == "completed"
    assert vid.id

    # The submit carries a per-item idempotency key so a wait/get retry re-uses the
    # ORIGINAL paid job instead of double-billing (see test_generative_gen_idempotency).
    submit = client.start_generation.await_args
    assert submit.args == ("a calm sea", {})
    assert submit.kwargs["idempotency_key"].endswith(":ai_video_gen:" + vid.id)
    # Prove the two-step contract: status poll THEN result fetch for the job.
    client.wait_for_generation.assert_awaited_once()
    client.get_generation_result.assert_awaited_once_with("job-1")


@pytest.mark.asyncio
async def test_multiple_prompts() -> None:
    client = _make_client()
    stage = AiVideoGenStage(client=client)
    ctx = _ctx({"ai_video_gen": {"prompts": ["a", "b"]}})

    out = await stage.run({}, ctx=ctx)

    assert [v.prompt for v in out.videos] == ["a", "b"]
    assert client.start_generation.await_count == 2


@pytest.mark.asyncio
async def test_config_forwarded() -> None:
    # The run pins provider="kling", so kling must be in the registry listing or
    # the capability gate would (correctly) skip before forwarding anything.
    client = _make_client(provider_list=_providers("kling", "stub"))
    stage = AiVideoGenStage(client=client)
    ctx = _ctx(
        {
            "ai_video_gen": {
                "prompt": "x",
                "provider": "kling",
                "duration_s": 8.0,
                "width": 1280,
                "height": 720,
                "image_url": "https://minio.test/ref.png",
            }
        }
    )

    await stage.run({}, ctx=ctx)

    config = client.start_generation.await_args.args[1]
    assert config["provider"] == "kling"
    assert config["duration_s"] == 8.0
    assert config["width"] == 1280
    assert config["height"] == 720
    assert config["image_url"] == "https://minio.test/ref.png"


@pytest.mark.asyncio
async def test_non_completed_terminal_status_skips_result_fetch_and_fails_loud() -> None:
    # poll_job returns for completed/partial/completed_with_errors, but the
    # /result route 409s unless status=="completed". A non-completed terminal
    # status must NOT fetch the result (no usable asset) and must fail loud.
    client = _make_client(status="partial")
    stage = AiVideoGenStage(client=client)
    ctx = _ctx({"ai_video_gen": {"prompt": "x"}})

    with pytest.raises(StageDownstreamResponseError):
        await stage.run({}, ctx=ctx)
    client.get_generation_result.assert_not_awaited()


@pytest.mark.asyncio
async def test_skips_when_ai_video_gen_block_absent() -> None:
    # Opt-in: without an ai_video_gen block the stage raises
    # StageSkippedError so the DAG marks it 'skipped' and the run still
    # completes, rather than completing green with zero clips.
    client = _make_client()
    stage = AiVideoGenStage(client=client)
    ctx = _ctx({"title": "Pluto", "brief": {"angle": "exploration"}})

    with pytest.raises(StageSkippedError):
        await stage.run({}, ctx=ctx)
    client.start_generation.assert_not_awaited()


@pytest.mark.asyncio
async def test_skips_when_config_snapshot_is_empty() -> None:
    client = _make_client()
    stage = AiVideoGenStage(client=client)
    ctx = _ctx({})

    with pytest.raises(StageSkippedError):
        await stage.run({}, ctx=ctx)
    client.start_generation.assert_not_awaited()


# ---------------------------------------------------------------------------
# Capability gate (WS-3): opting in != being able to serve it.
#
# Live prod evidence: 1,309 ai_video_gen rows skipped on the config-presence
# guard, ZERO completions, and no request has ever reached ACG. There is no
# hardware for local generation, so a run that DOES set an ai_video_gen block
# must not dispatch a job that can never be served. The load-bearing assertion
# in every test below is `start_generation.assert_not_awaited()`.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_existing_no_config_skip_reason_is_unchanged() -> None:
    # The pre-existing opt-in skip must keep its verbatim reason string: 1,309
    # live rows carry it, and the probe must not run before it (no config block
    # means no HTTP call of ANY kind).
    client = _make_client()
    stage = AiVideoGenStage(client=client)

    with pytest.raises(StageSkippedError) as exc:
        await stage.run({}, ctx=_ctx({"title": "Pluto"}))

    assert exc.value.reason == (
        "ai_video_gen: no ai_video_gen config block (opt-in stage not requested)"
    )
    client.list_video_providers.assert_not_awaited()
    client.start_generation.assert_not_awaited()


@pytest.mark.asyncio
async def test_proceeds_when_a_provider_is_available() -> None:
    client = _make_client(provider_list=_providers("kling"))
    stage = AiVideoGenStage(client=client)

    out = await stage.run({}, ctx=_ctx({"ai_video_gen": {"prompt": "a comet"}}))

    client.list_video_providers.assert_awaited_once()
    client.start_generation.assert_awaited_once()
    assert len(out.videos) == 1


@pytest.mark.asyncio
async def test_skips_when_registry_is_empty_and_never_calls_acg() -> None:
    # The live case: ACG is up and healthy but registered no provider (no working
    # cloud key, no local GPU). Skip cleanly; the DAG proceeds.
    client = _make_client(provider_list=_providers())
    stage = AiVideoGenStage(client=client)

    with pytest.raises(StageSkippedError) as exc:
        await stage.run({}, ctx=_ctx({"ai_video_gen": {"prompt": "a comet"}}))

    assert exc.value.reason == "ai_video_gen: no capable provider configured"
    client.start_generation.assert_not_awaited()
    assert client.start_generation.await_count == 0
    client.wait_for_generation.assert_not_awaited()
    client.get_generation_result.assert_not_awaited()


@pytest.mark.asyncio
async def test_skips_when_every_listed_provider_is_unavailable() -> None:
    # Registered but not introspectable => available=False. Same verdict.
    client = _make_client(provider_list=_providers("kling", "veo", available=False))
    stage = AiVideoGenStage(client=client)

    with pytest.raises(StageSkippedError) as exc:
        await stage.run({}, ctx=_ctx({"ai_video_gen": {"prompt": "a comet"}}))

    assert exc.value.reason == "ai_video_gen: no capable provider configured"
    assert client.start_generation.await_count == 0


@pytest.mark.asyncio
async def test_skips_when_pinned_provider_and_fallbacks_all_absent() -> None:
    # The run pinned sora + seedance; only stub registered. Submitting would burn
    # the full job/poll/timeout budget only to fail.
    client = _make_client(provider_list=_providers("stub"))
    stage = AiVideoGenStage(client=client)
    cfg = {"ai_video_gen": {"prompt": "x", "provider": "sora", "fallback_providers": ["seedance"]}}

    with pytest.raises(StageSkippedError) as exc:
        await stage.run({}, ctx=_ctx(cfg))

    assert exc.value.reason.startswith("ai_video_gen: no capable provider configured")
    assert "sora" in exc.value.reason and "seedance" in exc.value.reason
    assert client.start_generation.await_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [404, 401, 403, 422])
async def test_permanent_probe_failure_skips_clean(status: int) -> None:
    # PERMANENT (upstream.PERMANENT_STATUSES): a definite structural answer — this
    # ACG build has no providers route / rejects us. No retry can change it, so skip
    # clean, with a reason distinct from "registry empty" so the two stay diagnosable.
    client = _make_client(provider_probe_error=_http_error(status))
    stage = AiVideoGenStage(client=client)

    with pytest.raises(StageSkippedError) as exc:
        await stage.run({}, ctx=_ctx({"ai_video_gen": {"prompt": "x"}}))

    assert "provider registry is unreadable" in exc.value.reason
    assert exc.value.reason != "ai_video_gen: no capable provider configured"
    assert client.start_generation.await_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boom",
    [_http_error(503)],
)
async def test_transient_probe_failure_raises_after_retry(boom: BaseException) -> None:
    # TRANSIENT: a probe timeout / 503 is NOT evidence the service is unservable.
    # Swallowing it into a StageSkippedError would TERMINALLY drop the most expensive
    # stage of the run on one blip, with no failure signal and no resume. House
    # convention (stages/upstream.py + llm_budget.LLM_RETRY_ON): retry, then fail loud.
    client = _make_client(provider_probe_error=boom)
    stage = AiVideoGenStage(client=client)

    with pytest.raises(StageRetriesExhaustedError):
        await stage.run({}, ctx=_ctx({"ai_video_gen": {"prompt": "x"}}))

    # Retried (retries=1 => 2 attempts) before giving up.
    assert client.list_video_providers.await_count == 2
    assert client.start_generation.await_count == 0


@pytest.mark.asyncio
async def test_gate_runs_before_prompt_resolution() -> None:
    # An unservable run skips on the capability reason, not the no-prompts one —
    # ordering proves no work is done on behalf of a request that cannot run.
    client = _make_client(provider_list=_providers())
    stage = AiVideoGenStage(client=client)

    with pytest.raises(StageSkippedError) as exc:
        await stage.run({}, ctx=_ctx({"ai_video_gen": {"prompts": []}}))

    assert exc.value.reason == "ai_video_gen: no capable provider configured"


@pytest.mark.asyncio
async def test_raises_when_provider_returns_no_usable_url() -> None:
    for bad in (None, "None", "null", ""):
        # Completed job, but the RESULT carries an empty/None video_url — the
        # post-gather gate must reject it (a provider can 200 with a dead asset).
        client = _make_client(
            result=AcgVideoGenerationResult(
                job_id="j", status="completed", provider="stub", video_url=bad
            )
        )
        stage = AiVideoGenStage(client=client)
        ctx = _ctx({"ai_video_gen": {"prompt": "x"}})
        with pytest.raises(StageDownstreamResponseError):
            await stage.run({}, ctx=ctx)


@pytest.mark.asyncio
async def test_republishes_acg_relative_url(monkeypatch: pytest.MonkeyPatch) -> None:
    # ACG returns a license-gated, ACG-relative download path the anonymous
    # compiler can't fetch; the stage downloads it and re-publishes to run
    # storage, substituting the fetchable URL (mirror ai_image_gen).
    client = _make_client(
        result=AcgVideoGenerationResult(
            job_id="j",
            status="completed",
            provider="stub",
            video_url="/api/content/videos/jobs/j/download",
        )
    )
    client.download_generated_video = AsyncMock(return_value=b"\x00\x01mp4")
    published: list[bytes] = []

    async def _fake_store(*, run_id, stage_name, video_id, video_bytes, video_format):
        published.append(video_bytes)
        assert video_format == "mp4"
        return f"https://minio.test/runs/{run_id}/ai_video_gen/{video_id}.mp4"

    monkeypatch.setattr(
        "src.services.stages.generic.ai_video_gen.store_generated_video", _fake_store
    )
    stage = AiVideoGenStage(client=client)
    ctx = _ctx({"ai_video_gen": {"prompt": "x"}})

    out = await stage.run({}, ctx=ctx)

    client.download_generated_video.assert_awaited_once_with("/api/content/videos/jobs/j/download")
    assert published == [b"\x00\x01mp4"]
    assert out.videos[0].video_url is not None
    assert out.videos[0].video_url.startswith("https://minio.test/runs/")


@pytest.mark.asyncio
async def test_absolute_url_passes_through_without_republish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client()  # absolute https video_url
    client.download_generated_video = AsyncMock()

    async def _boom(**_kwargs):
        raise AssertionError("store_generated_video must not be called for absolute URLs")

    monkeypatch.setattr("src.services.stages.generic.ai_video_gen.store_generated_video", _boom)
    stage = AiVideoGenStage(client=client)
    ctx = _ctx({"ai_video_gen": {"prompt": "x"}})

    out = await stage.run({}, ctx=ctx)

    client.download_generated_video.assert_not_awaited()
    assert out.videos[0].video_url == "https://minio.test/buckets/video/asset-1.mp4"


@pytest.mark.asyncio
async def test_client_failure_propagates() -> None:
    # A provider IS available (gate passes) but the generation call itself blows
    # up — that must still propagate, not be swallowed into a skip.
    client = _make_client()
    client.start_generation = AsyncMock(side_effect=RuntimeError("acg down"))
    stage = AiVideoGenStage(client=client)
    ctx = _ctx({"ai_video_gen": {"prompt": "x"}})

    with pytest.raises(RuntimeError, match="acg down"):
        await stage.run({}, ctx=ctx)


def test_name_and_io_models() -> None:
    from src.services.stages.generic.ai_video_gen import AiVideoGenInput, AiVideoGenOutput

    assert AiVideoGenStage.name == "ai_video_gen"
    assert AiVideoGenStage.input_model is AiVideoGenInput
    assert AiVideoGenStage.output_model is AiVideoGenOutput


class _FakeStorageHit:
    """Storage whose objects always exist — every find_existing_media is a hit."""

    async def stage_artifact_url(self, run_id: str, stage_name: str, filename: str) -> str:
        return f"https://minio.test/w2v-orchestrator/artifacts/{run_id}/{stage_name}/{filename}"

    async def object_exists(self, url: str) -> bool:
        return True


@pytest.mark.asyncio
async def test_reuses_already_stored_video_and_skips_acg() -> None:
    """Resume/retry idempotency: an already-stored clip is reused without a new
    (expensive) ACG generation."""
    from src.services.tts_storage import set_storage_for_tests

    client = _make_client()
    set_storage_for_tests(cast("Any", _FakeStorageHit()))
    try:
        stage = AiVideoGenStage(client=client)
        out = await stage.run({}, ctx=_ctx({"ai_video_gen": {"prompt": "a drone flyover"}}))

        assert len(out.videos) == 1
        assert out.videos[0].status == "reused"
        url = out.videos[0].video_url
        assert url is not None and url.endswith(".mp4")
        client.start_generation.assert_not_awaited()
        client.wait_for_generation.assert_not_awaited()
    finally:
        set_storage_for_tests(None)


@pytest.mark.asyncio
async def test_resumed_run_reuses_stored_media_even_with_an_empty_registry() -> None:
    """D1 REGRESSION: the capability gate must not discard already-paid-for assets.

    A resumed run whose clips were all generated and stored on an earlier attempt
    must hand them back even if ACG has since lost every provider. The gate exists
    to stop work being DISPATCHED; there is nothing to dispatch here. Skipping would
    silently throw away the most expensive artifacts in the pipeline.
    """
    from src.services.tts_storage import set_storage_for_tests

    client = _make_client(provider_list=_providers())  # registry EMPTY
    set_storage_for_tests(cast("Any", _FakeStorageHit()))
    try:
        stage = AiVideoGenStage(client=client)
        out = await stage.run({}, ctx=_ctx({"ai_video_gen": {"prompts": ["a drone", "a comet"]}}))

        assert [v.status for v in out.videos] == ["reused", "reused"]
        assert all(v.video_url and v.video_url.endswith(".mp4") for v in out.videos)
        # Nothing to dispatch => the gate is not even consulted, and ACG is untouched.
        client.list_video_providers.assert_not_awaited()
        client.start_generation.assert_not_awaited()
    finally:
        set_storage_for_tests(None)


class _FakeStoragePartialHit:
    """Storage where exactly one clip of the run already exists."""

    def __init__(self, hit_suffix: str) -> None:
        self._hit_suffix = hit_suffix
        self.probed: list[str] = []

    async def stage_artifact_url(self, run_id: str, stage_name: str, filename: str) -> str:
        return f"https://minio.test/w2v-orchestrator/artifacts/{run_id}/{stage_name}/{filename}"

    async def object_exists(self, url: str) -> bool:
        self.probed.append(url)
        return url.endswith(self._hit_suffix)


@pytest.mark.asyncio
async def test_partially_resumed_run_still_gates_the_outstanding_work() -> None:
    # The other half of D1: if ANY prompt still needs generating, the gate applies —
    # a partially-resumed run must not be allowed to dispatch into an empty registry.
    from src.services.stages.generic.ai_video_gen import AiVideoGenStage as _Stage
    from src.services.tts_storage import deterministic_media_id, set_storage_for_tests

    hit_id = deterministic_media_id("a drone", "{}")
    storage = _FakeStoragePartialHit(f"{hit_id}.mp4")
    client = _make_client(provider_list=_providers())
    set_storage_for_tests(cast("Any", storage))
    try:
        stage = _Stage(client=client)
        with pytest.raises(StageSkippedError) as exc:
            await stage.run({}, ctx=_ctx({"ai_video_gen": {"prompts": ["a drone", "a comet"]}}))

        assert exc.value.reason == "ai_video_gen: no capable provider configured"
        client.list_video_providers.assert_awaited_once()
        client.start_generation.assert_not_awaited()
    finally:
        set_storage_for_tests(None)


async def test_republish_pass_has_an_overall_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    """#1111: downloads run in sequence and each may busy-wait 300s; the whole pass
    is bounded by ``_REPUBLISH_TIMEOUT_S`` and what misses it keeps its original URL."""
    import asyncio

    from src.services.stages.generic import ai_video_gen as mod

    started: list[str] = []

    async def _stuck(url: str) -> bytes:
        started.append(url)
        await asyncio.Event().wait()  # a lane that never frees up
        raise AssertionError("unreachable")

    client = _make_client()
    client.download_generated_video = AsyncMock(side_effect=_stuck)
    store = AsyncMock(return_value="https://minio.test/never")
    monkeypatch.setattr(mod, "store_generated_video", store)
    monkeypatch.setattr(mod, "_REPUBLISH_TIMEOUT_S", 0.2)
    stage = AiVideoGenStage(client=client)
    videos = [
        mod.GeneratedVideo(
            id=f"v{i}",
            prompt="p",
            asset_id=None,
            video_url=f"/api/content/videos/jobs/j{i}/download",
            duration_s=None,
            provider_used=None,
            status="completed",
        )
        for i in range(3)
    ]

    out = await asyncio.wait_for(stage._republish_relative(videos, run_id="r1"), timeout=5)

    assert [v.video_url for v in out] == [v.video_url for v in videos]  # originals kept
    assert len(started) == 1  # the deadline was spent on the first; the rest never started
    store.assert_not_awaited()
