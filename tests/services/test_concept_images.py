"""``generate_concept_image``: budget-sized deadline, same-key retry, final cancel."""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest

from src.clients.schemas.ai_content_generation import AcgConceptImageJob
from src.services import concept_images
from src.services.concept_images import generate_concept_image
from src.services.providers.errors import SlotNotConfiguredError

_DONE = AcgConceptImageJob(job_id="j1", status="completed", image_url="https://x/y.png")


def _client(*, wait: list[object]) -> AsyncMock:
    client = AsyncMock()
    client.create_concept_image = AsyncMock(return_value="j1")
    client.wait_for_concept_image = AsyncMock(side_effect=wait)
    client.cancel_concept_image = AsyncMock(return_value=True)
    return client


@pytest.mark.asyncio
async def test_deadline_is_the_callers_budget_minus_the_margin() -> None:
    client = _client(wait=[_DONE])
    out = await generate_concept_image(client, "p", {}, idempotency_key="k", budget_s=300.0)
    assert out is _DONE
    timeout = client.wait_for_concept_image.await_args.kwargs["timeout"]
    assert (
        300.0 - concept_images.DEADLINE_MARGIN_S - 1.0
        < timeout
        <= (300.0 - concept_images.DEADLINE_MARGIN_S)
    )
    client.cancel_concept_image.assert_not_awaited()


@pytest.mark.asyncio
async def test_transient_failure_retries_with_the_same_key_and_no_cancel() -> None:
    client = _client(wait=[httpx.ConnectError("boom"), _DONE])
    out = await generate_concept_image(client, "p", {"size": "1"}, idempotency_key="k")
    assert out is _DONE
    keys = [c.kwargs["idempotency_key"] for c in client.create_concept_image.await_args_list]
    assert keys == ["k", "k"]
    client.cancel_concept_image.assert_not_awaited()


@pytest.mark.asyncio
async def test_failure_on_every_attempt_cancels_once_after_the_final_one() -> None:
    client = _client(wait=[RuntimeError("failed"), RuntimeError("failed")])
    with pytest.raises(RuntimeError):
        await generate_concept_image(client, "p", {}, idempotency_key="k")
    assert client.create_concept_image.await_count == 2
    client.cancel_concept_image.assert_awaited_once_with("j1")


@pytest.mark.asyncio
async def test_timeout_is_final_and_cancels_once() -> None:
    client = _client(wait=[TimeoutError("deadline")])
    with pytest.raises(TimeoutError):
        await generate_concept_image(client, "p", {}, idempotency_key="k")
    assert client.create_concept_image.await_count == 1
    client.cancel_concept_image.assert_awaited_once_with("j1")


@pytest.mark.asyncio
async def test_permanent_refusal_is_not_retried_or_cancelled() -> None:
    client = _client(wait=[SlotNotConfiguredError("image")])
    with pytest.raises(SlotNotConfiguredError):
        await generate_concept_image(client, "p", {}, idempotency_key="k")
    assert client.create_concept_image.await_count == 1
    client.cancel_concept_image.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failing_cancel_never_replaces_the_original_error() -> None:
    client = _client(wait=[TimeoutError("deadline")])
    client.cancel_concept_image = AsyncMock(side_effect=httpx.ConnectError("down"))
    with pytest.raises(TimeoutError):
        await generate_concept_image(client, "p", {}, idempotency_key="k")


@pytest.mark.asyncio
async def test_no_retry_when_the_shared_deadline_is_spent(monkeypatch) -> None:
    clock = iter([0.0, 10.0, 590.0, 590.0, 590.0])
    monkeypatch.setattr(concept_images, "_now", lambda: next(clock))
    client = _client(wait=[RuntimeError("failed"), _DONE])
    with pytest.raises(RuntimeError):
        await generate_concept_image(client, "p", {}, idempotency_key="k", budget_s=600.0)
    assert client.create_concept_image.await_count == 1
    client.cancel_concept_image.assert_awaited_once_with("j1")


@pytest.mark.parametrize("budget", [0.0])
@pytest.mark.asyncio
async def test_a_budget_that_cannot_cover_the_margin_is_refused(budget: float) -> None:
    with pytest.raises(ValueError, match="budget"):
        await generate_concept_image(
            _client(wait=[_DONE]), "p", {}, idempotency_key="k", budget_s=budget
        )
