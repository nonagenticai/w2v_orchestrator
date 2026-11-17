"""Paid generative submits must NOT double-bill on a per-item wait/get retry.

``ai_video_gen`` / ``ai_image_gen`` run each prompt through
``stage_gather(..., per_item_retries=1)``. The paid provider submit
(``start_generation`` / ``create_concept_image``) and the wait/get live in the
SAME retried callable, so a transient failure in the WAIT re-runs the WHOLE
callable — re-submitting a SECOND paid render for the same prompt.

The fix passes a per-item ``idempotency_key`` on the submit so the provider
(ACG) returns the ORIGINAL job on the re-submit and bills nothing. These tests
model ACG's server-side Idempotency-Key dedup with a fake client: a repeat key
returns the cached job WITHOUT counting a new paid submit; a ``None`` key can
never dedupe (that is the broken path). The RED/GREEN signal is therefore the
count of PAID submits, not the raw mock call count.
"""

from __future__ import annotations

from typing import cast
from uuid import uuid4

import pytest

from src.clients.ai_content_generation_client import AIContentGenerationClient
from src.clients.schemas.ai_content_generation import (
    AcgConceptImageJob,
    AcgVideoGenerationResult,
    AcgVideoJobStatus,
    AcgVideoProviderInfo,
    AcgVideoProviderList,
)
from src.services.runs.run_context import RunContext
from src.services.stages.generic.ai_image_gen import AiImageGenStage
from src.services.stages.generic.ai_video_gen import AiVideoGenStage


class _BlipError(RuntimeError):
    """A transient wait failure — retried by stage_gather's default retry_on."""


class _FakeVideoClient:
    """Models ACG's Idempotency-Key dedup for the video generate endpoint."""

    def __init__(self) -> None:
        self.paid_submits = 0  # NEW paid provider renders actually billed
        self._by_key: dict[str, str] = {}
        self._wait_calls = 0

    async def start_generation(
        self, prompt: str, config: dict | None = None, *, idempotency_key: str | None = None
    ) -> str:
        # Server-side dedup: a repeat key returns the ORIGINAL job, bills nothing.
        # A missing key can never dedupe → every re-submit is a fresh paid render.
        if idempotency_key is not None and idempotency_key in self._by_key:
            return self._by_key[idempotency_key]
        self.paid_submits += 1
        job_id = f"vid-job-{self.paid_submits}"
        if idempotency_key is not None:
            self._by_key[idempotency_key] = job_id
        return job_id

    async def wait_for_generation(self, job_id: str, timeout: float = 300.0) -> AcgVideoJobStatus:
        self._wait_calls += 1
        if self._wait_calls == 1:
            raise _BlipError("transient poll blip after the job already started")
        return AcgVideoJobStatus(job_id=job_id, status="completed", provider="stub")

    async def get_generation_result(self, job_id: str) -> AcgVideoGenerationResult:
        return AcgVideoGenerationResult(
            job_id=job_id,
            status="completed",
            provider="stub",
            video_url="https://minio.test/run/clip.mp4",
            actual_duration_s=5.0,
        )

    async def list_video_providers(self) -> AcgVideoProviderList:
        return AcgVideoProviderList(providers=[AcgVideoProviderInfo(name="stub", available=True)])


class _FakeImageClient:
    """Models ACG's Idempotency-Key dedup for the concept-image endpoint."""

    def __init__(self) -> None:
        self.paid_submits = 0
        self._by_key: dict[str, str] = {}
        self._wait_calls = 0

    async def create_concept_image(
        self, prompt: str, config: dict | None = None, *, idempotency_key: str | None = None
    ) -> str:
        if idempotency_key is not None and idempotency_key in self._by_key:
            return self._by_key[idempotency_key]
        self.paid_submits += 1
        job_id = f"img-job-{self.paid_submits}"
        if idempotency_key is not None:
            self._by_key[idempotency_key] = job_id
        return job_id

    async def wait_for_concept_image(
        self, job_id: str, timeout: float = 300.0
    ) -> AcgConceptImageJob:
        self._wait_calls += 1
        if self._wait_calls == 1:
            raise _BlipError("transient poll blip after the job already started")
        return AcgConceptImageJob(
            job_id=job_id,
            status="completed",
            provider="stub",
            asset_id="asset-1",
            image_url="https://minio.test/run/img.png",
        )


@pytest.mark.asyncio
async def test_ai_video_gen_does_not_double_bill_on_wait_retry(monkeypatch: pytest.MonkeyPatch):
    # No prior stored media → the stage actually dispatches a generation.
    monkeypatch.setattr(
        "src.services.stages.generic.ai_video_gen.find_existing_media",
        _async_none,
    )
    client = _FakeVideoClient()
    stage = AiVideoGenStage(client=cast("AIContentGenerationClient", client))
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"ai_video_gen": {"prompts": ["a cat surfing"]}},
    )

    out = await stage.run({}, ctx=ctx)

    assert [v.status for v in out.videos] == ["completed"]
    # THE ASSERTION: the transient first-wait failure retried the callable, but
    # the idempotency key made the re-submit dedupe → exactly ONE paid render.
    # RED on pre-fix code (no key passed): paid_submits == 2 (double-bill).
    assert client.paid_submits == 1


@pytest.mark.asyncio
async def test_ai_image_gen_does_not_double_bill_on_wait_retry(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "src.services.stages.generic.ai_image_gen.find_existing_media",
        _async_none,
    )
    client = _FakeImageClient()
    stage = AiImageGenStage(client=cast("AIContentGenerationClient", client))
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"ai_image_gen": {"prompts": ["a neon skyline"]}},
    )

    out = await stage.run({}, ctx=ctx)

    assert [i.status for i in out.images] == ["completed"]
    assert client.paid_submits == 1


async def _async_none(*_args: object, **_kwargs: object) -> None:
    """find_existing_media stub: always a cache miss → the stage generates."""
    return
