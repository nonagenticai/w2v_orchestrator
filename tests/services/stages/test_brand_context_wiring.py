"""The redesign + promo_html stages forward extracted brand/site_analysis to start_design."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.clients.schemas.design import DesignJobResult, DesignJobStatus
from src.services.runs.run_context import RunContext
from src.services.stages.promo.html_segments import PromoHtmlStage
from src.services.stages.website_review.redesign import WebsiteRedesignStage

_BRAND = {"primary_color": "#0a1f44", "secondary_color": "#e8b923", "name": "Acme"}
_SITE = {"visual_identity": {"primary_color": "#0a1f44"}}


def _patch_brand(monkeypatch: pytest.MonkeyPatch, module: str, brand: Any, site: Any) -> None:
    async def _resolved(*_a: Any, **_k: Any) -> tuple[Any, Any]:
        return brand, site

    monkeypatch.setattr(f"{module}.resolve_brand_context", _resolved)


@pytest.mark.asyncio
async def test_redesign_forwards_brand(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_brand(monkeypatch, "src.services.stages.website_review.redesign", _BRAND, _SITE)
    fake = AsyncMock()
    fake.start_design.return_value = "rd-1"
    fake.wait_for_design.return_value = DesignJobStatus(job_id="rd-1", status="completed")
    fake.get_design_result.return_value = DesignJobResult(
        job_id="rd-1",
        url="https://acme.test",
        scope="redesign",
        style="modern",
        screenshots={"desktop": "https://minio.test/after.png"},
    )
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"brief": {}},
        prior_run_outputs={"website_critique": {"url": "https://acme.test", "findings": []}},
    )

    await WebsiteRedesignStage(client=fake).run({}, ctx=ctx)

    _, kwargs = fake.start_design.call_args
    assert kwargs["brand"] == _BRAND
    assert kwargs["site_analysis"] == _SITE


@pytest.mark.asyncio
async def test_promo_html_forwards_brand(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_brand(monkeypatch, "src.services.stages.promo.html_segments", _BRAND, _SITE)
    fake = AsyncMock()
    fake.start_design.return_value = "ph-1"
    fake.wait_for_design.return_value = DesignJobStatus(job_id="ph-1", status="completed")
    fake.get_design_result.return_value = DesignJobResult(
        job_id="ph-1",
        url="https://acme.test",
        scope="hero",
        style="modern",
        screenshots={"desktop": "https://minio.test/hero.png"},
    )
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"brief": {"source_url": "https://acme.test"}},
    )

    out = await PromoHtmlStage(client=fake).run({}, ctx=ctx)

    assert out.used_html is True
    _, kwargs = fake.start_design.call_args
    assert kwargs["brand"] == _BRAND
    assert kwargs["site_analysis"] == _SITE


@pytest.mark.asyncio
async def test_promo_html_soft_fails_when_brand_resolver_used_but_design_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Even with brand resolved, a design failure must still soft-fail (0 segments).
    _patch_brand(monkeypatch, "src.services.stages.promo.html_segments", _BRAND, _SITE)
    fake = AsyncMock()
    fake.start_design.side_effect = RuntimeError("design down")
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"brief": {"source_url": "https://acme.test"}},
    )

    out = await PromoHtmlStage(client=fake).run({}, ctx=ctx)
    assert out.used_html is False
    assert out.segments == []
    assert any("design failed" in n for n in out.notes)
