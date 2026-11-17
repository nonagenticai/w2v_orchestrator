"""source_ingest — multi-source grounding: webscraper scrape + in-process document parse."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from w2v_common.clients import ImageRef
from w2v_common.clients.webscraper import ScrapeResponse

from src.services.documents.extract import ExtractedDocument, ExtractedImage
from src.services.runs.run_context import RunContext
from src.services.stages.base import StageSkippedError
from src.services.stages.generic.source_ingest import SourceIngestStage, grounding_prompt_block
from tests.support import documents as fixtures


def _ctx(brief: dict) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"brief": brief, "title": "T"},
    )


def _page(url: str = "https://acme.com", **over: Any) -> ScrapeResponse:
    return ScrapeResponse(url=url, **over)


def _scraper(page: ScrapeResponse | None = None) -> AsyncMock:
    ws = AsyncMock()
    ws.scrape_url.return_value = page or _page(text="Acme makes great widgets.")
    return ws


@pytest.fixture
def no_ocr(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin OCR off regardless of the developer's env: the ``ocr`` slot resolves to
    the platform default (a local_allowed tenant with no OCR connection), which is
    off while ``W2V_PROVIDER_OCR_URL`` is empty."""
    from src.clients import ocr_client

    monkeypatch.delenv("W2V_PROVIDER_OCR_URL", raising=False)

    async def _platform(settings: Any) -> Any:
        return ocr_client.platform_target(settings)

    monkeypatch.setattr(ocr_client, "resolve_ocr_target", _platform)


@pytest.mark.asyncio
async def test_skips_when_no_sources():
    stage = SourceIngestStage(client=AsyncMock())
    with pytest.raises(StageSkippedError):
        await stage.run({}, ctx=_ctx({}))


@pytest.mark.asyncio
async def test_reuses_precomputed_course_ingest_without_refetch():
    """A lesson child handed a course-level ingest must reuse it verbatim and
    never re-scrape (the course-level ingest cache fast path)."""
    ws = _scraper()
    precomputed = {
        "grounding_text": "# Source: Acme\nAcme makes great widgets.",
        "references": ["ref-a"],
        "images": [],
        "pinned_media": [],
        "notes": ["scraped 1 source"],
        "sources_ingested": 1,
    }
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        # brief.sources present too — proves the short-circuit precedes ingest.
        config_snapshot={
            "brief": {"sources": [{"kind": "url", "ref": "https://acme.com"}]},
            "precomputed_source_ingest": precomputed,
        },
    )
    out = await SourceIngestStage(client=ws).run({}, ctx=ctx)
    ws.scrape_url.assert_not_awaited()
    assert out.sources_ingested == 1
    assert "Acme makes great widgets." in out.grounding_text
    assert any("reused course-level source_ingest" in n for n in out.notes)


@pytest.mark.asyncio
async def test_ignores_empty_precomputed_ingest():
    """An empty/skipped course ingest payload must NOT short-circuit — the child
    still grounds itself from its own brief.sources."""
    ws = _scraper(_page(text="Fresh content."))
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "brief": {"sources": [{"kind": "url", "ref": "https://acme.com"}]},
            "precomputed_source_ingest": {},  # skipped course ingest → {}
        },
    )
    out = await SourceIngestStage(client=ws).run({}, ctx=ctx)
    ws.scrape_url.assert_awaited_once()
    assert "Fresh content." in out.grounding_text


@pytest.mark.asyncio
async def test_scrapes_url_sources_into_grounding():
    ws = _scraper()
    out = await SourceIngestStage(client=ws).run(
        {}, ctx=_ctx({"sources": [{"kind": "url", "ref": "https://acme.com", "label": "Acme"}]})
    )
    assert "Acme makes great widgets." in out.grounding_text
    assert "# Source: Acme" in out.grounding_text
    assert out.sources_ingested == 1
    ws.scrape_url.assert_awaited_once()
    args, kwargs = ws.scrape_url.call_args
    assert args == ("https://acme.com",)
    assert kwargs["download_images"] is True
    assert kwargs["timeout"] == 180.0
    assert kwargs["auth_config"] is None


@pytest.mark.asyncio
async def test_scrape_prefers_markdown_over_plain_text():
    ws = _scraper(_page(text="Nav Home plain body", markdown="# Acme\n\nClean markdown body."))
    out = await SourceIngestStage(client=ws).run(
        {}, ctx=_ctx({"sources": [{"kind": "url", "ref": "https://acme.com"}]})
    )
    assert "Clean markdown body." in out.grounding_text
    assert "Nav Home plain body" not in out.grounding_text


@pytest.mark.asyncio
async def test_failed_page_status_is_a_visible_soft_fail():
    """webscraper answers an unscrapable page with HTTP 200 + status=failed, which
    the client maps to ``error``. That must NOT read as an empty success."""
    ws = _scraper(_page(text="", error="HTTP 403 from origin"))
    out = await SourceIngestStage(client=ws).run(
        {}, ctx=_ctx({"sources": [{"kind": "url", "ref": "https://acme.com"}]})
    )
    assert out.grounding_text == ""
    assert any("scrape failed" in n and "HTTP 403 from origin" in n for n in out.notes)


@pytest.mark.asyncio
async def test_scraped_images_are_stored_and_icons_dropped(monkeypatch):
    import base64

    big = base64.b64encode(fixtures.image_bytes()).decode()
    icon = base64.b64encode(fixtures.image_bytes(16, 16)).decode()
    ws = _scraper(
        _page(
            text="Body.",
            images=[
                ImageRef(src="https://acme.com/hero.png", alt="hero", content_base64=big),
                ImageRef(src="https://acme.com/icon.png", alt="icon", content_base64=icon),
                ImageRef(src="https://acme.com/lazy.png", alt="not downloaded"),
            ],
        )
    )
    store = AsyncMock(return_value="http://minio/img.png")
    monkeypatch.setattr("src.services.stages.generic.source_ingest.store_capture_image", store)
    out = await SourceIngestStage(client=ws).run(
        {}, ctx=_ctx({"sources": [{"kind": "url", "ref": "https://acme.com"}]})
    )
    assert [(i.url, i.alt_text) for i in out.images] == [("http://minio/img.png", "hero")]
    assert store.await_args is not None
    assert store.await_args.kwargs["image_format"] == "png"


@pytest.mark.asyncio
async def test_webscraper_client_is_built_lazily(monkeypatch):
    """No url source → no WebscraperClient (it refuses construction without a
    licence, and a document-only run must not fail on that)."""

    def _boom(*_a: object, **_k: object) -> None:
        raise AssertionError("WebscraperClient must not be built")

    monkeypatch.setattr("src.services.stages.generic.source_ingest.WebscraperClient", _boom)
    monkeypatch.setattr(
        "src.services.stages.generic.source_ingest.fetch_document",
        AsyncMock(return_value=b"Plain text notes about widgets."),
    )
    out = await SourceIngestStage().run(
        {}, ctx=_ctx({"sources": [{"kind": "document", "ref": "documents/t/notes.txt"}]})
    )
    assert "Plain text notes about widgets." in out.grounding_text


@pytest.mark.asyncio
async def test_ingests_url_without_evidence_of_a_usable_capture():
    ws = _scraper()
    out = await SourceIngestStage(client=ws).run(
        {},
        ctx=_ctx(
            {
                "source_url": "https://acme.com",
                "sources": [{"kind": "url", "ref": "https://acme.com"}],
            }
        ),
    )
    ws.scrape_url.assert_awaited_once()
    assert out.sources_ingested == 1


@pytest.mark.asyncio
async def test_ingests_pdf_document_in_process(monkeypatch, no_ocr):
    pdf = fixtures.text_pdf(
        ["Chapter one about widgets.", "Chapter two about gadgets."],
        jpeg=fixtures.image_bytes(300, 300, "JPEG"),
    )
    monkeypatch.setattr(
        "src.services.stages.generic.source_ingest.fetch_document", AsyncMock(return_value=pdf)
    )
    store = AsyncMock(return_value="http://minio/doc-img.jpg")
    monkeypatch.setattr("src.services.stages.generic.source_ingest.store_capture_image", store)
    out = await SourceIngestStage(client=AsyncMock()).run(
        {},
        ctx=_ctx({"sources": [{"kind": "document", "ref": "documents/t/x.pdf", "label": "Book"}]}),
    )
    assert "Chapter one about widgets." in out.grounding_text
    assert "# Source: Book" in out.grounding_text
    assert [i.url for i in out.images] == ["http://minio/doc-img.jpg"]
    assert store.await_args is not None
    assert store.await_args.kwargs["image_format"] == "jpg"
    assert out.sources_ingested == 1


@pytest.mark.asyncio
async def test_ingests_docx_document_by_sniffing_not_label(monkeypatch):
    """The label carries no extension; the bytes decide the parser."""
    monkeypatch.setattr(
        "src.services.stages.generic.source_ingest.fetch_document",
        AsyncMock(return_value=fixtures.docx(["Docx body text."], heading="Intro")),
    )
    out = await SourceIngestStage(client=AsyncMock()).run(
        {},
        ctx=_ctx({"sources": [{"kind": "document", "ref": "documents/t/blob", "label": "Brief"}]}),
    )
    assert "Docx body text." in out.grounding_text


@pytest.mark.asyncio
async def test_scanned_pdf_without_ocr_is_a_visible_soft_fail(monkeypatch, no_ocr):
    """A text-less PDF with OCR off contributes nothing and SAYS why — never a
    silent empty grounding block."""
    monkeypatch.setattr(
        "src.services.stages.generic.source_ingest.fetch_document",
        AsyncMock(return_value=fixtures.scanned_pdf(2)),
    )
    out = await SourceIngestStage(client=AsyncMock()).run(
        {}, ctx=_ctx({"sources": [{"kind": "document", "ref": "documents/t/scan.pdf"}]})
    )
    assert out.grounding_text == ""
    assert out.sources_ingested == 0
    assert any("document parse failed" in n and "OCR is not configured" in n for n in out.notes)


@pytest.mark.asyncio
async def test_unsupported_document_soft_fails(monkeypatch):
    monkeypatch.setattr(
        "src.services.stages.generic.source_ingest.fetch_document",
        AsyncMock(return_value=b"\x00\x01\x02 not a document"),
    )
    out = await SourceIngestStage(client=AsyncMock()).run(
        {}, ctx=_ctx({"sources": [{"kind": "document", "ref": "documents/t/x.bin"}]})
    )
    assert out.grounding_text == ""
    assert any("document parse failed" in n and "unsupported" in n for n in out.notes)


@pytest.mark.asyncio
async def test_extraction_notes_reach_the_run(monkeypatch):
    monkeypatch.setattr(
        "src.services.stages.generic.source_ingest.fetch_document",
        AsyncMock(return_value=b"%PDF"),
    )
    monkeypatch.setattr(
        "src.services.stages.generic.source_ingest.extract_document",
        AsyncMock(
            return_value=ExtractedDocument(
                text="Partial text.", notes=["x.pdf: 3 page(s) ... were not OCR'd"]
            )
        ),
    )
    out = await SourceIngestStage(client=AsyncMock()).run(
        {}, ctx=_ctx({"sources": [{"kind": "document", "ref": "documents/t/x.pdf"}]})
    )
    assert "Partial text." in out.grounding_text
    assert "x.pdf: 3 page(s) ... were not OCR'd" in out.notes


@pytest.mark.asyncio
async def test_media_pin_unreachable_is_soft(monkeypatch):
    """A pin whose fetch genuinely FAILS leaves composite_url + kind empty + a
    note, but the ref is still recorded and the run is not failed."""
    monkeypatch.setattr(
        "src.services.stages.generic.source_ingest.republish_image",
        AsyncMock(side_effect=RuntimeError("egress blocked")),
    )
    monkeypatch.setattr(
        "src.services.stages.generic.source_ingest.republish_video",
        AsyncMock(side_effect=RuntimeError("egress blocked")),
    )
    stage = SourceIngestStage(client=AsyncMock())
    out = await stage.run(
        {}, ctx=_ctx({"sources": [{"kind": "media", "ref": "https://v/page.html", "label": "x"}]})
    )
    assert out.pinned_media == [
        {
            "ref": "https://v/page.html",
            "label": "x",
            "composite_url": "",
            "kind": "",
            "composite_status": "failed",
            "composite_detail": "egress blocked",
        }
    ]
    assert any("not composited" in n for n in out.notes)


@pytest.mark.asyncio
async def test_media_pin_youtube_watch_page_is_classified_not_fetched(monkeypatch):
    """Prod 01a09f82: a pinned YouTube result is a watch PAGE (text/html), not a
    clip. It is classified up front — no fetch at all — and recorded without a
    failure-looking note (a note degrades the run)."""
    image = AsyncMock(side_effect=AssertionError("must not fetch a watch page"))
    video = AsyncMock(side_effect=AssertionError("must not fetch a watch page"))
    monkeypatch.setattr("src.services.stages.generic.source_ingest.republish_image", image)
    monkeypatch.setattr("src.services.stages.generic.source_ingest.republish_video", video)
    stage = SourceIngestStage(client=AsyncMock())
    ref = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    out = await stage.run({}, ctx=_ctx({"sources": [{"kind": "media", "ref": ref, "label": "yt"}]}))
    pin = out.pinned_media[0]
    assert pin["composite_status"] == "not_compositable"
    assert "youtube.com watch page" in str(pin["composite_detail"])
    assert pin["composite_url"] == ""
    assert not any("not composited" in n for n in out.notes)
    image.assert_not_awaited()
    video.assert_not_awaited()


@pytest.mark.asyncio
async def test_media_pin_html_answer_is_classified_not_a_failure(monkeypatch):
    """An unknown host that answers text/html to both probes is NOT media: recorded as
    not_compositable, no note."""
    from src.services.visual_media import NotMediaContentError

    url = "https://example.org/some-page"
    monkeypatch.setattr(
        "src.services.stages.generic.source_ingest.republish_image",
        AsyncMock(side_effect=NotMediaContentError(url, "text/html", expected="image")),
    )
    monkeypatch.setattr(
        "src.services.stages.generic.source_ingest.republish_video",
        AsyncMock(side_effect=NotMediaContentError(url, "text/html", expected="video")),
    )
    stage = SourceIngestStage(client=AsyncMock())
    out = await stage.run({}, ctx=_ctx({"sources": [{"kind": "media", "ref": url, "label": "p"}]}))
    assert out.pinned_media[0]["composite_status"] == "not_compositable"
    assert "text/html" in str(out.pinned_media[0]["composite_detail"])
    assert not any("not composited" in n for n in out.notes)


@pytest.mark.asyncio
async def test_scrape_failure_is_soft(monkeypatch):
    ws = AsyncMock()
    ws.scrape_url.side_effect = RuntimeError("boom")
    stage = SourceIngestStage(client=ws)
    out = await stage.run({}, ctx=_ctx({"sources": [{"kind": "url", "ref": "https://acme.com"}]}))
    assert out.grounding_text == ""
    assert any("scrape failed" in n for n in out.notes)


@pytest.mark.asyncio
async def test_republishes_document_images(monkeypatch):
    raw = fixtures.image_bytes()
    monkeypatch.setattr(
        "src.services.stages.generic.source_ingest.fetch_document",
        AsyncMock(return_value=b"doc"),
    )
    monkeypatch.setattr(
        "src.services.stages.generic.source_ingest.extract_document",
        AsyncMock(
            return_value=ExtractedDocument(
                text="x",
                images=[ExtractedImage(data=raw, mime_type="image/png", caption="a chart")],
            )
        ),
    )
    monkeypatch.setattr(
        "src.services.stages.generic.source_ingest.store_capture_image",
        AsyncMock(return_value="http://minio/img.png"),
    )
    out = await SourceIngestStage(client=AsyncMock()).run(
        {}, ctx=_ctx({"sources": [{"kind": "document", "ref": "documents/t/x.pdf"}]})
    )
    assert out.images and out.images[0].url == "http://minio/img.png"
    assert out.images[0].caption == "a chart"


@pytest.mark.asyncio
async def test_failed_image_publish_drops_the_image(monkeypatch):
    """No dead-URL fallback any more (KAG's download path is gone): an image whose
    publish fails is dropped, the text still grounds."""
    monkeypatch.setattr(
        "src.services.stages.generic.source_ingest.fetch_document",
        AsyncMock(return_value=b"doc"),
    )
    monkeypatch.setattr(
        "src.services.stages.generic.source_ingest.extract_document",
        AsyncMock(
            return_value=ExtractedDocument(
                text="Text survives.",
                images=[ExtractedImage(data=b"x", mime_type="image/png")],
            )
        ),
    )
    monkeypatch.setattr(
        "src.services.stages.generic.source_ingest.store_capture_image",
        AsyncMock(side_effect=RuntimeError("minio down")),
    )
    out = await SourceIngestStage(client=AsyncMock()).run(
        {}, ctx=_ctx({"sources": [{"kind": "document", "ref": "documents/t/x.pdf"}]})
    )
    assert "Text survives." in out.grounding_text
    assert out.images == []


def test_grounding_prompt_block_empty_when_no_output():
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})
    assert grounding_prompt_block(ctx) == ""


def test_grounding_prompt_block_includes_text():
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={},
        prior_run_outputs={"source_ingest": {"grounding_text": "Real facts here."}},
    )
    block = grounding_prompt_block(ctx)
    assert "Real facts here." in block
    assert "SOURCE MATERIAL" in block


# --------------------------------------------------------------------------- #
# Deferring to site_capture is only safe when site_capture actually produced text.
#
# site_capture's page scrape is best-effort — every failure becomes ``page_text
# = ""`` plus a note. Skipping the URL unconditionally turned that ONE soft
# failure into a total loss of grounding: ``promo_script`` reads
# ``site_capture.page_text``, everything else reads ``grounding_text``, and both
# were empty. Live on 2026-08-10 that shipped a promo with an invented script.
# --------------------------------------------------------------------------- #


def _ctx_with_capture(brief: dict, page_text: str) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"brief": brief, "title": "T"},
        prior_run_outputs={"site_capture": {"page_text": page_text, "notes": [], "clips": []}},
    )


_SITE = "https://example.com/landing"
_BRIEF = {"source_url": _SITE, "sources": [{"kind": "url", "ref": _SITE}]}


@pytest.mark.asyncio
async def test_skips_site_url_when_site_capture_got_text():
    """CONTROL: the de-duplication that motivated the skip must still happen."""
    ws = AsyncMock()
    stage = SourceIngestStage(client=ws)
    out = await stage.run(
        {}, ctx=_ctx_with_capture(_BRIEF, "The real rendered page content. " * 12)
    )
    ws.scrape_url.assert_not_awaited()
    assert out.sources_ingested == 1
    assert "The real rendered page content." in out.grounding_text


@pytest.mark.asyncio
async def test_reingests_site_url_when_site_capture_produced_no_text():
    """The regression: a failed capture must NOT also suppress the fallback."""
    ws = _scraper(_page(url=_SITE, text="Recovered page body about widgets."))
    stage = SourceIngestStage(client=ws)
    out = await stage.run({}, ctx=_ctx_with_capture(_BRIEF, ""))
    ws.scrape_url.assert_awaited()
    assert "Recovered page body about widgets." in out.grounding_text
    assert out.sources_ingested == 1


@pytest.mark.asyncio
async def test_reingests_a_thin_capture_instead_of_skipping_grounding():
    ws = _scraper(_page(url=_SITE, text="Recovered rendered content."))
    context = _ctx_with_capture(_BRIEF, "App title only")
    context.prior_run_outputs["site_capture"]["page_text_status"] = "thin"
    out = await SourceIngestStage(client=ws).run({}, ctx=context)
    ws.scrape_url.assert_awaited_once()
    assert "Recovered rendered content." in out.grounding_text


# --------------------------------------------------------------------------- #
# The grounding block is PASTED into a worker prompt, so it cannot be allowed to
# consume that worker's whole budget. WM answers an over-budget prompt with a 500
# reading "Model token limit (6000) exceeded before any response was generated",
# which stages/upstream.py classifies PERMANENT — the stage fails outright.
# --------------------------------------------------------------------------- #


def test_grounding_prompt_block_is_bounded_well_below_a_worker_budget():
    """A maximal stored grounding text must not blow the smallest worker budget.

    4 000 tokens is the smallest budget observed in the fleet; at a conservative
    3.5 chars/token that is ~14 000 chars for prompt AND answer together. The
    block alone must leave most of that free.
    """
    from src.services.stages.generic.source_ingest import _GROUNDING_MAX_CHARS

    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={},
        prior_run_outputs={"source_ingest": {"grounding_text": "x" * _GROUNDING_MAX_CHARS}},
    )
    block = grounding_prompt_block(ctx)
    smallest_worker_budget_chars = 4000 * 3.5
    assert len(block) < smallest_worker_budget_chars * 0.6
    # ...and it must still carry real grounding, not be truncated to nothing.
    assert len(block) > 4000


def test_grounding_prompt_block_respects_an_explicit_smaller_budget():
    """A caller with a large preamble can shrink its own share."""
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={},
        prior_run_outputs={"source_ingest": {"grounding_text": "y" * 20000}},
    )
    assert len(grounding_prompt_block(ctx, max_chars=500)) < len(grounding_prompt_block(ctx))
    assert "y" * 500 in grounding_prompt_block(ctx, max_chars=500)
