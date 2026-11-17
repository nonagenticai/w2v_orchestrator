"""ai_image_gen sends a frame-shaped default size only when none is authored."""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.clients.schemas.ai_content_generation import AcgConceptImageJob
from src.services.runs.run_context import RunContext
from src.services.stages.generic.ai_image_gen import (
    DEFAULT_LOCAL_IMAGE_SIZE,
    AiImageGenStage,
)


def _client() -> AsyncMock:
    client = AsyncMock()
    client.create_concept_image = AsyncMock(return_value="job-1")
    client.wait_for_concept_image = AsyncMock(
        return_value=AcgConceptImageJob(
            job_id="job-1",
            asset_id="asset-1",
            status="completed",
            provider="local_image",
            image_url="https://minio.test/buckets/concept/asset-1.png",
        )
    )
    return client


async def _sent_config(block: dict) -> dict:
    client = _client()
    ctx = RunContext(
        run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={"ai_image_gen": block}
    )
    await AiImageGenStage(client=client).run({}, ctx=ctx)
    return client.create_concept_image.await_args.args[1]


def test_default_is_the_valid_shape_closest_to_16_9() -> None:
    w, h = (int(v) for v in DEFAULT_LOCAL_IMAGE_SIZE.split("x"))
    assert (w, h) == (512, 320)
    assert w % 64 == 0 and h % 64 == 0  # image_gen_backend multiple_of=64
    assert w * h <= 512 * 512  # no latency regression vs the old square


def test_the_default_survives_the_live_device_cap_unchanged() -> None:
    """⛔ THE GUARD THAT STOPS THIS BEING RE-RAISED HERE. This was attempted and
    reverted (2026-09-11).

    ``IMAGE_GEN_MAX_DIM=512`` is live on mini1, from the gitops ConfigMap
    ``image-gen-backend-env`` (``the deployment manifests
    external-services/image-gen-backend.yaml:76``). ``flux_service._aspect_clamp``
    scales the LONG edge down to it, so a larger request is RESHAPED, not honoured:
    896x512 comes back 512x293 — 150016 px against today's 163840, a net LOSS of
    real detail. Raising this constant without raising the ConfigMap cap first
    therefore degrades stills while appearing to improve them.
    """
    cap = 512  # IMAGE_GEN_MAX_DIM, live on mini1 — see the ConfigMap above.
    w, h = (int(v) for v in DEFAULT_LOCAL_IMAGE_SIZE.split("x"))

    # The ask must pass the clamp untouched — no reshaping.
    assert max(w, h) <= cap

    scale = cap / max(w, h)
    assert (round(w * scale), round(h * scale)) == (w, h)
    # And it must beat what any over-cap ask would be reshaped into.
    assert w * h >= 512 * 293


@pytest.mark.asyncio
async def test_unauthored_size_sends_default() -> None:
    config = await _sent_config({"prompt": "x"})
    assert config["size"] == "512x320"


@pytest.mark.asyncio
async def test_unauthored_size_with_local_provider_sends_default() -> None:
    config = await _sent_config({"prompt": "x", "provider": "local_image"})
    assert config["size"] == "512x320"


@pytest.mark.asyncio
async def test_authored_size_wins() -> None:
    config = await _sent_config({"prompt": "x", "size": "1280x720"})
    assert config["size"] == "1280x720"


@pytest.mark.asyncio
async def test_cloud_provider_keeps_its_own_default() -> None:
    config = await _sent_config({"prompt": "x", "provider": "openai_images"})
    assert "size" not in config
