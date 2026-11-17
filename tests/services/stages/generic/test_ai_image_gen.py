"""Tests for the ai_image_gen stage handler."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest

from src.clients.ai_content_generation_client import AIContentGenerationClient
from src.clients.schemas.ai_content_generation import AcgConceptImageJob
from src.services.runs.run_context import RunContext
from src.services.stages.base import (
    StageDownstreamResponseError,
    StageSkippedError,
)
from src.services.stages.generic import ai_image_gen
from src.services.stages.generic.ai_image_gen import AiImageGenStage


def _make_client(*, job_id: str = "job-1", result: AcgConceptImageJob | None = None) -> AsyncMock:
    client = AsyncMock()
    client.create_concept_image = AsyncMock(return_value=job_id)
    client.wait_for_concept_image = AsyncMock(
        return_value=result
        or AcgConceptImageJob(
            job_id=job_id,
            asset_id="asset-1",
            status="completed",
            provider="local_image",
            image_url="https://minio.test/buckets/concept/asset-1.png",
        )
    )
    return client


def _ctx(cfg: dict) -> RunContext:
    return RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot=cfg)


@pytest.mark.asyncio
async def test_happy_path_single_prompt() -> None:
    client = _make_client()
    stage = AiImageGenStage(client=client)
    ctx = _ctx({"ai_image_gen": {"prompt": "a serene mountain"}})

    out = await stage.run({}, ctx=ctx)

    assert len(out.images) == 1
    img = out.images[0]
    assert img.prompt == "a serene mountain"
    assert img.asset_id == "asset-1"
    assert img.image_url is not None
    assert img.image_url.endswith("/asset-1.png")
    assert img.provider_used == "local_image"
    assert img.status == "completed"
    assert img.id  # uuid generated

    # The submit carries a per-item idempotency key so a wait retry re-uses the
    # ORIGINAL paid job instead of double-billing (see test_generative_gen_idempotency).
    submit = client.create_concept_image.await_args
    # An unauthored run on the default (local_image) provider now requests a
    # frame-shaped still: the square default was pillarboxed by ken_burns' `contain`,
    # leaving ~44% of the delivered frame width as blurred filler.
    assert submit.args == ("a serene mountain", {"size": ai_image_gen.DEFAULT_LOCAL_IMAGE_SIZE})
    assert submit.kwargs["idempotency_key"].endswith(":ai_image_gen:" + img.id)


@pytest.mark.asyncio
async def test_multiple_prompts() -> None:
    client = _make_client()
    stage = AiImageGenStage(client=client)
    ctx = _ctx({"ai_image_gen": {"prompts": ["a", "b", "c"]}})

    out = await stage.run({}, ctx=ctx)

    assert [img.prompt for img in out.images] == ["a", "b", "c"]
    assert client.create_concept_image.await_count == 3


@pytest.mark.asyncio
async def test_provider_and_size_forwarded_in_config() -> None:
    client = _make_client()
    stage = AiImageGenStage(client=client)
    ctx = _ctx(
        {
            "ai_image_gen": {
                "prompt": "x",
                "provider": "local_image",
                "size": "1024x1024",
                "style": "editorial",
                "negative_prompt": "blurry",
                "extra_params": {"steps": 8},
            }
        }
    )

    await stage.run({}, ctx=ctx)

    call = client.create_concept_image.await_args
    assert call.args[0] == "x"
    config = call.args[1]
    assert config["provider"] == "local_image"
    assert config["size"] == "1024x1024"
    assert config["style"] == "editorial"
    assert config["negative_prompt"] == "blurry"
    assert config["extra_params"] == {"steps": 8}


@pytest.mark.asyncio
async def test_falls_back_to_brief_derived_prompt_when_opted_in() -> None:
    # Brief-derived fallback only applies when the caller has explicitly
    # opted into the stage (ai_image_gen block present, even if empty of
    # prompts) AND given non-trivial config like a style override.
    client = _make_client()
    stage = AiImageGenStage(client=client)
    ctx = _ctx(
        {
            "ai_image_gen": {"style": "editorial"},
            "title": "Pluto",
            "brief": {"angle": "exploration"},
        }
    )

    out = await stage.run({}, ctx=ctx)

    assert len(out.images) == 1
    assert out.images[0].prompt == "Pluto. exploration"


@pytest.mark.asyncio
async def test_skips_when_ai_image_gen_block_absent() -> None:
    # Opt-in: without an ai_image_gen block the stage must NOT dispatch, even if
    # title/brief are present. It raises StagePreconditionError so the DAG marks
    # it 'skipped' (and the run still completes) rather than completing green
    # with zero images — the old silent-success lie.
    client = _make_client()
    stage = AiImageGenStage(client=client)
    ctx = _ctx({"title": "Pluto", "brief": {"angle": "exploration"}})

    with pytest.raises(StageSkippedError):
        await stage.run({}, ctx=ctx)
    client.create_concept_image.assert_not_awaited()


@pytest.mark.asyncio
async def test_skips_when_config_snapshot_is_empty() -> None:
    client = _make_client()
    stage = AiImageGenStage(client=client)
    ctx = _ctx({})

    with pytest.raises(StageSkippedError):
        await stage.run({}, ctx=ctx)
    client.create_concept_image.assert_not_awaited()


@pytest.mark.asyncio
async def test_raises_when_provider_returns_no_usable_url() -> None:
    # A provider can 200 with a failed payload carrying no/`"None"` url. The old
    # code stored that as a dead image and completed green; now it must fail loud.
    for bad in (None, "None", "null", ""):
        client = _make_client(
            result=AcgConceptImageJob(
                job_id="j",
                asset_id="a",
                status="completed",
                provider="local_image",
                image_url=bad,
            )
        )
        stage = AiImageGenStage(client=client)
        ctx = _ctx({"ai_image_gen": {"prompt": "x"}})
        with pytest.raises(StageDownstreamResponseError):
            await stage.run({}, ctx=ctx)


@pytest.mark.asyncio
async def test_client_failure_propagates() -> None:
    client = AsyncMock()
    client.create_concept_image = AsyncMock(side_effect=RuntimeError("acg down"))
    stage = AiImageGenStage(client=client)
    ctx = _ctx({"ai_image_gen": {"prompt": "x"}})

    with pytest.raises(RuntimeError, match="acg down"):
        await stage.run({}, ctx=ctx)


def test_name_and_io_models() -> None:
    assert AiImageGenStage.name == "ai_image_gen"
    from src.services.stages.generic.ai_image_gen import AiImageGenInput, AiImageGenOutput

    assert AiImageGenStage.input_model is AiImageGenInput
    assert AiImageGenStage.output_model is AiImageGenOutput


@pytest.mark.asyncio
async def test_relative_image_url_is_republished_to_fetchable_store(monkeypatch) -> None:
    """ACG returns a license-gated relative download URL the compiler can't fetch;
    the stage downloads the bytes and re-publishes to a fetchable store URL."""
    client = _make_client(
        result=AcgConceptImageJob(
            job_id="job-1",
            asset_id="asset-1",
            status="completed",
            provider="local_image",
            image_url="/api/concept-images/assets/asset-1/download",
        )
    )
    client.download_concept_image = AsyncMock(return_value=b"\x89PNGfake")
    captured = {}

    async def _fake_store(*, run_id, stage_name, image_id, image_bytes, image_format):
        captured["bytes"] = image_bytes
        captured["stage"] = stage_name
        return "https://minio.test/w2v-orchestrator/artifacts/run/ai_image_gen/x.png"

    monkeypatch.setattr(
        "src.services.stages.generic.ai_image_gen.store_generated_image", _fake_store
    )
    stage = AiImageGenStage(client=client)
    out = await stage.run({}, ctx=_ctx({"ai_image_gen": {"prompt": "a node graph"}}))

    assert out.images[0].image_url == (
        "https://minio.test/w2v-orchestrator/artifacts/run/ai_image_gen/x.png"
    )
    client.download_concept_image.assert_awaited_once_with(
        "/api/concept-images/assets/asset-1/download"
    )
    assert captured["bytes"] == b"\x89PNGfake"
    assert captured["stage"] == "ai_image_gen"


class _FakeStorageHit:
    """Storage whose objects always exist — every find_existing_media is a hit."""

    async def stage_artifact_url(self, run_id: str, stage_name: str, filename: str) -> str:
        return f"https://minio.test/w2v-orchestrator/artifacts/{run_id}/{stage_name}/{filename}"

    async def object_exists(self, url: str) -> bool:
        return True


@pytest.mark.asyncio
async def test_reuses_already_stored_image_and_skips_acg() -> None:
    """Resume/retry idempotency: when the item's deterministic media object is
    already stored, the stage reuses it and does NOT re-hit ACG (no regeneration,
    no re-download)."""
    from src.services.tts_storage import set_storage_for_tests

    client = _make_client()
    set_storage_for_tests(cast("Any", _FakeStorageHit()))
    try:
        stage = AiImageGenStage(client=client)
        out = await stage.run({}, ctx=_ctx({"ai_image_gen": {"prompt": "a serene mountain"}}))

        assert len(out.images) == 1
        assert out.images[0].status == "reused"
        url = out.images[0].image_url
        assert url is not None and url.endswith(".png")
        client.create_concept_image.assert_not_awaited()
        client.wait_for_concept_image.assert_not_awaited()
    finally:
        set_storage_for_tests(None)


@pytest.mark.asyncio
async def test_acg_402_for_a_byo_required_tenant_is_byo_provider_required_not_retried(
    monkeypatch,
) -> None:
    """#1106 end to end: the REAL ACG client against a mocked ACG whose local_image
    job failed on modality_gateway's 402. The stage must fail ONCE (no paid retry)
    with the structured refusal the run API surfaces — not a RuntimeError."""
    import httpx

    from src.clients.ai_content_generation_client import AIContentGenerationClient
    from src.models.failure_class import CUSTOMER_INPUT, attribute_exception, failure_detail_for
    from src.services.providers.errors import SlotNotConfiguredError

    submits: list[httpx.Request] = []

    def acg(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/api/concept-images":
            submits.append(request)
            return httpx.Response(202, json={"job_id": "job-402", "status": "queued"})
        if request.url.path == "/api/concept-images/jobs/job-402":
            return httpx.Response(
                200,
                json={
                    "job_id": "job-402",
                    "status": "failed",
                    "error": "local_image: backend returned 402",
                },
            )
        return httpx.Response(404)

    client = AIContentGenerationClient(base_url="http://acg.test", license_token="svc")
    client.set_transport(httpx.MockTransport(acg))
    monkeypatch.setattr(ai_image_gen, "find_existing_media", AsyncMock(return_value=None))
    stage = AiImageGenStage(client=client)

    with pytest.raises(SlotNotConfiguredError) as info:
        await stage.run({}, ctx=_ctx({"ai_image_gen": {"prompt": "x"}}))

    assert len(submits) == 1, "a byo refusal must not be retried (each retry is a paid submit)"
    assert attribute_exception(info.value) == CUSTOMER_INPUT
    assert failure_detail_for(info.value) == {
        "code": "byo_provider_required",
        "modality": "image",
        "slots": "image",
        "message": "Configure your own image provider to continue",
    }


def test_vendored_acg_402_marker_matches_local_image_wording() -> None:
    from src.clients.ai_content_generation_client import IMAGE_BYO_REFUSAL_MARKER

    # ACG local_image: f"local_image: backend returned {status}" with status=402.
    assert "local_image: backend returned " + str(402) == IMAGE_BYO_REFUSAL_MARKER


# --- queue wait vs render time ------------------------------------------------
#
# The self-hosted image backend renders ONE image at a time, so a job's poll
# spends most of its life QUEUED behind other runs' images. ACG reports that
# queue as ``processing`` too (it marks the job before taking the single
# provider slot), so the deadline must cover queue + render. These drive the
# REAL ACG client poll on a fake clock: every poll sleep advances it instead of
# waiting.


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now


def _install_fake_clock(monkeypatch: pytest.MonkeyPatch) -> _FakeClock:
    import asyncio
    import types

    from w2v_common.clients import base as client_base

    clock = _FakeClock()
    real_sleep = asyncio.sleep

    async def _sleep(delay: float, *args: Any, **kwargs: Any) -> None:
        clock.now += float(delay)
        await real_sleep(0)

    monkeypatch.setattr(client_base, "time", types.SimpleNamespace(monotonic=clock.monotonic))
    monkeypatch.setattr(client_base.asyncio, "sleep", _sleep)
    return clock


def _acg_queued_until(
    clock: _FakeClock,
    *,
    done_at: float | None,
    submits: list[httpx.Request],
    cancels: list[tuple[int, httpx.Request]] | None = None,
    cancel_status: int = 200,
) -> Callable[[httpx.Request], httpx.Response]:

    def acg(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/api/concept-images":
            submits.append(request)
            return httpx.Response(202, json={"job_id": "job-q", "status": "pending"})
        if request.method == "POST" and request.url.path == "/api/concept-images/jobs/job-q/cancel":
            if cancels is not None:
                # Record how many submits preceded the cancel: it must follow the LAST.
                cancels.append((len(submits), request))
            if cancel_status == 200:
                return httpx.Response(200, json={"job_id": "job-q", "status": "cancelled"})
            return httpx.Response(cancel_status, json={"detail": "Job cannot be cancelled"})
        if request.url.path == "/api/concept-images/jobs/job-q":
            if done_at is not None and clock.now >= done_at:
                return httpx.Response(
                    200,
                    json={
                        "job_id": "job-q",
                        "asset_id": "asset-q",
                        "status": "completed",
                        "provider": "local_image",
                        "image_url": "https://minio.test/buckets/concept/asset-q.png",
                    },
                )
            return httpx.Response(200, json={"job_id": "job-q", "status": "processing"})
        return httpx.Response(404)

    return acg


async def _real_client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> AIContentGenerationClient:
    """The REAL ACG client on a mock transport, KEEPING its baked auth headers and
    identity event hooks (``set_transport`` drops both, so a header assertion
    through it proves nothing; same technique as test_pr2_clients_auth)."""
    client = AIContentGenerationClient(base_url="http://acg.test", license_token="svc")
    real = await client._get_client()
    await real.aclose()
    client._client = httpx.AsyncClient(
        base_url=client.base_url,
        timeout=client.timeout,
        headers=dict(real.headers),
        event_hooks=real._event_hooks,
        transport=httpx.MockTransport(handler),
        follow_redirects=False,
    )
    return client


@pytest.mark.asyncio
async def test_long_queue_wait_then_completes_does_not_time_out(monkeypatch) -> None:
    """A job queued well past the old 180 s render budget (300 s here: 9 images
    ahead on a one-at-a-time backend) still completes inside the per-item budget,
    on ONE submit — no timeout, no retry, no second GPU job."""
    clock = _install_fake_clock(monkeypatch)
    submits: list[httpx.Request] = []
    client = await _real_client(_acg_queued_until(clock, done_at=300.0, submits=submits))
    monkeypatch.setattr(ai_image_gen, "find_existing_media", AsyncMock(return_value=None))

    out = await AiImageGenStage(client=client).run(
        {}, ctx=_ctx({"ai_image_gen": {"prompt": "queued"}})
    )

    assert clock.now >= 300.0
    assert [img.asset_id for img in out.images] == ["asset-q"]
    assert len(submits) == 1


@pytest.mark.asyncio
async def test_job_that_never_finishes_waits_the_whole_budget_then_cancels_once(
    monkeypatch,
) -> None:
    """A job that never completes is bounded by the per-item budget (not 180 s).
    Running out of that deadline is FINAL: no re-submit (it would only wait again
    on the same job), and exactly one best-effort cancel frees its GPU queue spot."""
    from w2v_common.auth.identity_context import set_identity_context
    from w2v_common.stages.errors import StageError

    from src.services import concept_images

    set_identity_context(str(uuid4()), "lic-1", "sub-1")
    clock = _install_fake_clock(monkeypatch)
    submits: list[httpx.Request] = []
    cancels: list[tuple[int, httpx.Request]] = []
    client = await _real_client(
        _acg_queued_until(clock, done_at=None, submits=submits, cancels=cancels)
    )
    monkeypatch.setattr(ai_image_gen, "find_existing_media", AsyncMock(return_value=None))

    with pytest.raises((StageError, TimeoutError)):
        await AiImageGenStage(client=client).run(
            {}, ctx=_ctx({"ai_image_gen": {"prompt": "never"}})
        )

    assert len(submits) == 1
    assert [n for n, _ in cancels] == [1]
    # The cancel carries the S2S identity like every other ACG call.
    for header in ("X-Tenant-Id", "X-License-Id", "X-Subject-Id"):
        assert cancels[0][1].headers.get(header) == submits[-1].headers.get(header)
        assert cancels[0][1].headers.get(header)
    # The wait outlived a whole deep queue, not the old 180 s render budget.
    deadline = concept_images.DEFAULT_BUDGET_S - concept_images.DEADLINE_MARGIN_S
    assert clock.now >= deadline > 500.0


@pytest.mark.asyncio
async def test_failed_attempt_is_retried_with_the_same_idempotency_key(monkeypatch) -> None:
    """A transient failure (the job reports ``failed``) is retried inside the same
    budget with the SAME Idempotency-Key, and a success is never cancelled."""
    clock = _install_fake_clock(monkeypatch)
    submits: list[httpx.Request] = []
    cancels: list[tuple[int, httpx.Request]] = []
    queued = _acg_queued_until(clock, done_at=0.0, submits=submits, cancels=cancels)

    def acg(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/concept-images/jobs/job-q" and len(submits) == 1:
            return httpx.Response(200, json={"job_id": "job-q", "status": "failed", "error": "x"})
        return queued(request)

    client = await _real_client(acg)
    monkeypatch.setattr(ai_image_gen, "find_existing_media", AsyncMock(return_value=None))

    out = await AiImageGenStage(client=client).run(
        {}, ctx=_ctx({"ai_image_gen": {"prompt": "flaky"}})
    )

    assert [img.asset_id for img in out.images] == ["asset-q"]
    assert len(submits) == 2
    keys = {r.headers.get("Idempotency-Key") for r in submits}
    assert len(keys) == 1 and None not in keys
    assert cancels == []


def test_stage_budget_is_the_shared_helper_budget() -> None:
    from src.services import concept_images

    # The helper stops waiting DEADLINE_MARGIN_S before stage_gather's timeout,
    # and the deadline is far above one render (~20-30 s): it covers the queue.
    assert ai_image_gen._PER_ITEM_TIMEOUT_SECONDS == concept_images.DEFAULT_BUDGET_S
    assert concept_images.DEFAULT_BUDGET_S - concept_images.DEADLINE_MARGIN_S >= 3 * 180.0


@pytest.mark.asyncio
async def test_authored_job_timeout_is_capped_at_the_whole_budget() -> None:
    from src.services import concept_images

    client = _make_client()
    await AiImageGenStage(client=client).run(
        {}, ctx=_ctx({"ai_image_gen": {"prompt": "x", "job_timeout_s": 5000}})
    )
    timeout = client.wait_for_concept_image.await_args.kwargs["timeout"]
    deadline = concept_images.DEFAULT_BUDGET_S - concept_images.DEADLINE_MARGIN_S
    assert deadline - 5.0 < timeout <= deadline


@pytest.mark.parametrize("cancel_status", [409])
@pytest.mark.asyncio
async def test_final_timeout_cancel_failure_never_masks_the_timeout(
    monkeypatch, cancel_status: int
) -> None:
    """409 (already finished) is fine, and even a 500 from the cancel must leave
    the stage failing on its TIMEOUT, not on the best-effort cancel."""
    from w2v_common.stages.errors import StageError

    clock = _install_fake_clock(monkeypatch)
    submits: list[httpx.Request] = []
    cancels: list[tuple[int, httpx.Request]] = []
    client = await _real_client(
        _acg_queued_until(
            clock, done_at=None, submits=submits, cancels=cancels, cancel_status=cancel_status
        )
    )
    monkeypatch.setattr(ai_image_gen, "find_existing_media", AsyncMock(return_value=None))

    with pytest.raises((StageError, TimeoutError)) as info:
        await AiImageGenStage(client=client).run(
            {}, ctx=_ctx({"ai_image_gen": {"prompt": "never"}})
        )

    assert not isinstance(info.value, httpx.HTTPStatusError)
    assert len(cancels) == 1


@pytest.mark.parametrize(("status", "expected"), [(200, True)])
@pytest.mark.asyncio
async def test_cancel_concept_image_maps_409_to_false(status: int, expected: bool) -> None:
    seen: list[httpx.Request] = []

    def acg(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if status == 200:
            return httpx.Response(200, json={"job_id": "j", "status": "cancelled"})
        return httpx.Response(status, json={"detail": "Job cannot be cancelled"})

    client = await _real_client(acg)
    assert await client.cancel_concept_image("j") is expected
    assert [(r.method, r.url.path) for r in seen] == [("POST", "/api/concept-images/jobs/j/cancel")]
