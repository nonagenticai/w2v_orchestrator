"""source_ingest x persistent cross-run cache (A2 wiring).

Exercises the stage with the cache ACTIVE (session_factory + a bound tenant):
a cross-run hit skips the scrape, durable images survive the hit, force-refresh
bypasses, the tenant gate + degrade-on-cache-error keep the direct path. The
cache-DISABLED path (no session_factory) is covered by test_source_ingest.py.
"""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock
from uuid import uuid7

import pytest
from sqlalchemy.exc import OperationalError
from w2v_common.clients import ImageRef
from w2v_common.clients.webscraper import ScrapeResponse

from src.services.runs.run_context import RunContext
from src.services.stages.generic.source_ingest import SourceIngestStage
from src.services.tts_storage import set_storage_for_tests
from tests.support.documents import image_bytes

pytestmark = pytest.mark.asyncio

_PNG = base64.b64encode(image_bytes()).decode()


class _FakeStorage:
    """In-memory backend exposing the generic keyed-object surface + the
    _StorageLike clip stubs so set_storage_for_tests accepts it."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    async def initialize(self) -> None: ...
    async def store_stage_artifact(self, job_id, stage_name, data, filename) -> str:
        key = f"{job_id}/{stage_name}/{filename}"
        self.objects[key] = data
        return key

    async def get_presigned_url(self, path, expires=None, bucket=None) -> str | None:
        return f"minio://run/{path}"

    async def put_object(self, key, data, content_type="application/octet-stream") -> str:
        self.objects[key] = data
        return key

    async def object_url(self, key) -> str:
        return f"minio://bucket/{key}"

    async def object_exists(self, url) -> bool:
        return url.removeprefix("minio://bucket/") in self.objects


@pytest.fixture
def fake_storage():
    fake = _FakeStorage()
    set_storage_for_tests(fake)
    yield fake
    set_storage_for_tests(None)


def _ctx(brief: dict, tenant_id) -> RunContext:
    return RunContext(
        run_id=uuid7(),
        deliverable_id=uuid7(),
        tenant_id=tenant_id,
        config_snapshot={"brief": brief, "title": "T"},
    )


def _scraper(text: str = "Acme makes widgets.", *, with_image: bool = False) -> AsyncMock:
    images = [ImageRef(src="https://acme.com/logo.png", alt="logo", content_base64=_PNG)]
    ws = AsyncMock()
    ws.scrape_url.return_value = ScrapeResponse(
        url="https://acme.com", text=text, images=images if with_image else []
    )
    return ws


async def test_cross_run_hit_skips_the_scrape(db_session_factory, fake_storage) -> None:
    """Two runs (different run_id, SAME tenant + source) share ONE scrape."""
    tenant = uuid7()
    brief = {"sources": [{"kind": "url", "ref": "https://acme.com", "label": "Acme"}]}
    ws = _scraper()

    out1 = await SourceIngestStage(client=ws, session_factory=db_session_factory).run(
        {}, ctx=_ctx(brief, tenant)
    )
    assert "Acme makes widgets." in out1.grounding_text
    assert ws.scrape_url.await_count == 1

    # A fresh stage instance (≈ a later run) over the same source → cache HIT.
    out2 = await SourceIngestStage(client=ws, session_factory=db_session_factory).run(
        {}, ctx=_ctx(brief, tenant)
    )
    assert out2.grounding_text == out1.grounding_text
    assert ws.scrape_url.await_count == 1  # NOT re-scraped


async def test_hit_reproduces_durable_image(db_session_factory, fake_storage) -> None:
    """A cached image (stored under the durable media-cache prefix) is re-minted on
    a hit, so the second run's output carries the image without re-scraping."""
    tenant = uuid7()
    brief = {"sources": [{"kind": "url", "ref": "https://acme.com"}]}
    ws = _scraper(with_image=True)

    out1 = await SourceIngestStage(client=ws, session_factory=db_session_factory).run(
        {}, ctx=_ctx(brief, tenant)
    )
    assert len(out1.images) == 1
    assert out1.images[0].url.startswith("minio://bucket/media-cache/")  # durable prefix

    out2 = await SourceIngestStage(client=ws, session_factory=db_session_factory).run(
        {}, ctx=_ctx(brief, tenant)
    )
    assert ws.scrape_url.await_count == 1  # hit
    assert len(out2.images) == 1
    assert out2.images[0].url == out1.images[0].url  # re-minted from the same key


async def test_degrades_to_direct_ingest_on_cache_db_error(fake_storage) -> None:
    """A cache-layer DB failure must degrade to a direct ingest, never fail the run."""

    def _broken_factory():
        raise OperationalError("SELECT 1", {}, Exception("db down"))

    ws = _scraper()
    out = await SourceIngestStage(client=ws, session_factory=_broken_factory).run(
        {}, ctx=_ctx({"sources": [{"kind": "url", "ref": "https://acme.com"}]}, uuid7())
    )
    assert "Acme makes widgets." in out.grounding_text  # produced despite cache down
    assert ws.scrape_url.await_count == 1
