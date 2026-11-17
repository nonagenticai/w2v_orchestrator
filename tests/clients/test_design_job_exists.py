"""``design_job_exists`` — the deliverable probe for blueprints whose artifact is
an html_render design job rather than an object we stored.

Tri-state contract, same as ``StorageService.object_exists``: only ``False`` is
safe to fail a run on. It deliberately fetches the HTML (not the job row): the
job row outliving its artifact is exactly the loss this must catch, and this is
the same upstream path the user's Download HTML button proxies.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.clients.design_client import design_job_exists


def _status_error(code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "http://html-render/api/designs/jobs/j/html")
    return httpx.HTTPStatusError(
        f"design returned {code}", request=request, response=httpx.Response(code, request=request)
    )


def _patch(**kwargs):
    return patch(
        "src.clients.design_client.DesignClient.get_design_html", AsyncMock(**kwargs)
    ), patch("src.clients.design_client.DesignClient.close", AsyncMock())


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [404])
async def test_gone_is_definitively_absent(code: int) -> None:
    html, close = _patch(side_effect=_status_error(code))
    with html, close:
        assert await design_job_exists("job-1") is False


@pytest.mark.asyncio
async def test_served_html_is_present() -> None:
    html, close = _patch(return_value="<html></html>")
    with html, close:
        assert await design_job_exists("job-1") is True


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [401])
async def test_a_status_that_is_about_us_is_indeterminate(code: int) -> None:
    """An auth or upstream problem must never fail a finished run."""
    html, close = _patch(side_effect=_status_error(code))
    with html, close:
        assert await design_job_exists("job-1") is None


@pytest.mark.asyncio
async def test_transport_error_is_indeterminate() -> None:
    html, close = _patch(side_effect=httpx.ConnectError("no route"))
    with html, close:
        assert await design_job_exists("job-1") is None


@pytest.mark.asyncio
async def test_the_client_is_closed_after_the_probe_even_when_it_errors() -> None:
    """Observable property: the connection is released on the FAILURE path too.

    A probe that leaks an httpx client per finished run is a slow resource leak in
    the finalizer — the path that matters is the one that raises.
    """
    events: list[str] = []

    async def _html(_self: object, job_id: str) -> str:
        events.append(f"fetch:{job_id}")
        raise _status_error(404)

    async def _close(_self: object) -> None:
        events.append("close")

    with (
        patch("src.clients.design_client.DesignClient.get_design_html", _html),
        patch("src.clients.design_client.DesignClient.close", _close),
    ):
        verdict = await design_job_exists("job-1")

    assert verdict is False
    assert events == ["fetch:job-1", "close"]
