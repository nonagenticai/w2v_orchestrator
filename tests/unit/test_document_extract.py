"""In-process document extraction (KAG removal) — against REAL generated files."""

from __future__ import annotations

import base64
import io
import json
from typing import Any

import httpx
import pytest
from pydantic import SecretStr
from pypdf import PdfReader

from src.clients.ocr_client import OcrClient
from src.config import ProviderOcrSettings
from src.services.documents import archive_guard
from src.services.documents import extract as extract_mod
from src.services.documents.extract import (
    MAX_IMAGES,
    DocumentExtractError,
    detect_kind,
    extract_document,
)
from tests.support import documents as fixtures

pytestmark = pytest.mark.asyncio

_OFF = OcrClient(ProviderOcrSettings())


async def _no_sleep(_s: float) -> None:
    return None


def _ocr_settings() -> ProviderOcrSettings:
    return ProviderOcrSettings(
        W2V_PROVIDER_OCR_URL="http://ocr.test:9131",
        W2V_PROVIDER_OCR_API_KEY=SecretStr("k-test"),  # pragma: allowlist secret
        W2V_PROVIDER_OCR_MODEL="PaddleOCR-VL-1.5-bf16",
    )


class _FakeOcr:
    """A provider_contracts /v1/ocr fake: one markdown line per page of the sent PDF."""

    def __init__(self, status: int = 200) -> None:
        self.status = status
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        if self.status != 200:
            return httpx.Response(
                self.status, json={"error": {"message": "boom", "type": "auth", "code": None}}
            )
        url = body["document"]["document_url"]
        pdf = base64.b64decode(url.split(",", 1)[1])
        n = len(PdfReader(io.BytesIO(pdf)).pages)
        pages = [
            {"index": i, "markdown": f"OCR text of sent page {i}", "images": []} for i in range(n)
        ]
        return httpx.Response(200, json={"model": body["model"], "pages": pages})


def _ocr(fake: _FakeOcr) -> OcrClient:
    return OcrClient(_ocr_settings(), transport=httpx.MockTransport(fake), sleep=_no_sleep)


# ─── kinds ─────────────────────────────────────────────────────────────────────


async def test_detects_kinds_from_bytes_not_names() -> None:
    assert detect_kind(fixtures.text_pdf(["x"]), "mislabelled.txt") == "pdf"
    assert detect_kind(fixtures.docx(["x"]), "noext") == "docx"
    assert detect_kind(fixtures.epub(["x"]), "book.docx") == "epub"
    assert detect_kind(b"# Title\nbody", "notes.md") == "text"
    assert detect_kind("plain é text".encode(), "noext") == "text"
    with pytest.raises(DocumentExtractError, match="unsupported"):
        detect_kind(b"\x00\x01binary", "blob.bin")


async def test_empty_bytes_raise() -> None:
    with pytest.raises(DocumentExtractError, match="empty"):
        await extract_document(b"", "x.pdf", ocr=_OFF)


# ─── pdf ───────────────────────────────────────────────────────────────────────


async def test_pdf_text_layer_and_embedded_image() -> None:
    data = fixtures.text_pdf(
        ["First page about widgets.", "Second page about gadgets."],
        jpeg=fixtures.image_bytes(300, 300, "JPEG"),
    )
    doc = await extract_document(data, "deck.pdf", ocr=_OFF)
    assert "First page about widgets." in doc.text
    assert "Second page about gadgets." in doc.text
    assert doc.text.index("widgets") < doc.text.index("gadgets")
    assert len(doc.images) == 1
    assert doc.images[0].mime_type == "image/jpeg"
    assert doc.notes == []


async def test_password_protected_pdf_raises() -> None:
    with pytest.raises(DocumentExtractError, match="password-protected"):
        await extract_document(fixtures.encrypted_pdf(), "locked.pdf", ocr=_OFF)


async def test_corrupt_pdf_raises() -> None:
    with pytest.raises(DocumentExtractError, match="not a readable PDF"):
        await extract_document(b"%PDF-1.4\ngarbage with no xref", "broken.pdf", ocr=_OFF)


async def test_scanned_pdf_without_ocr_raises_naming_the_setting() -> None:
    """No silent empty document: a text-less PDF with OCR off is a loud error."""
    with pytest.raises(DocumentExtractError, match="W2V_PROVIDER_OCR_URL"):
        await extract_document(fixtures.scanned_pdf(2), "scan.pdf", ocr=_OFF)


async def test_mixed_pdf_without_ocr_keeps_text_and_notes_the_gap() -> None:
    data = fixtures.text_pdf(["A real text page with enough words.", None])
    doc = await extract_document(data, "mixed.pdf", ocr=_OFF)
    assert "A real text page" in doc.text
    assert doc.notes == [
        "mixed.pdf: 1 page(s) without a text layer were not OCR'd (OCR is not configured)"
    ]


async def test_scanned_pdf_is_ocrd_page_by_page() -> None:
    fake = _FakeOcr()
    doc = await extract_document(fixtures.scanned_pdf(3), "scan.pdf", ocr=_ocr(fake))
    assert doc.text.count("OCR text of sent page") == 3
    assert len(fake.requests) == 1
    sent = fake.requests[0]
    assert sent["model"] == "PaddleOCR-VL-1.5-bf16"
    assert str(sent["document"]["document_url"]).startswith("data:application/pdf;base64,")


async def test_only_textless_pages_are_sent_to_ocr() -> None:
    """Pages with a text layer stay pypdf text; only the blank page goes to OCR."""
    fake = _FakeOcr()
    data = fixtures.text_pdf(["Page one has a proper text layer.", None, "Page three too, plenty."])
    doc = await extract_document(data, "mixed.pdf", ocr=_ocr(fake))
    sent = fake.requests[0]["document"]["document_url"]
    sub = PdfReader(io.BytesIO(base64.b64decode(str(sent).split(",", 1)[1])))
    assert len(sub.pages) == 1
    parts = doc.text.split("\n\n")
    assert parts == [
        "Page one has a proper text layer.",
        "OCR text of sent page 0",
        "Page three too, plenty.",
    ]


async def test_ocr_batches_large_scans(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(extract_mod, "OCR_BATCH_PAGES", 2)
    fake = _FakeOcr()
    doc = await extract_document(fixtures.scanned_pdf(5), "scan.pdf", ocr=_ocr(fake))
    assert [
        len(
            PdfReader(
                io.BytesIO(base64.b64decode(str(r["document"]["document_url"]).split(",", 1)[1]))
            ).pages
        )
        for r in fake.requests
    ] == [2, 2, 1]
    assert doc.text.count("OCR text") == 5


async def test_ocr_page_cap_is_noted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(extract_mod, "MAX_OCR_PAGES", 2)
    doc = await extract_document(fixtures.scanned_pdf(3), "scan.pdf", ocr=_ocr(_FakeOcr()))
    assert doc.text.count("OCR text") == 2
    assert any("only the first 2 of 3" in n for n in doc.notes)


async def test_ocr_failure_on_textless_pdf_raises() -> None:
    with pytest.raises(DocumentExtractError, match=r"OCR failed: .*HTTP 401"):
        await extract_document(fixtures.scanned_pdf(1), "scan.pdf", ocr=_ocr(_FakeOcr(401)))


async def test_ocr_failure_on_mixed_pdf_keeps_text_and_notes_it() -> None:
    data = fixtures.text_pdf(["Real text survives the OCR outage.", None])
    doc = await extract_document(data, "mixed.pdf", ocr=_ocr(_FakeOcr(401)))
    assert "Real text survives" in doc.text
    assert any("OCR of 1 scanned page(s) failed" in n for n in doc.notes)


async def test_pdf_page_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(extract_mod, "MAX_PDF_PAGES", 2)
    doc = await extract_document(
        fixtures.text_pdf(["Page one text.", "Page two text.", "Page three text."]),
        "long.pdf",
        ocr=_OFF,
    )
    assert "Page three" not in doc.text
    assert any("first 2 of 3 pages" in n for n in doc.notes)


# ─── docx / epub / text ────────────────────────────────────────────────────────


async def test_docx_text_heading_and_image() -> None:
    data = fixtures.docx(
        ["Widgets are great.", "Gadgets too."], heading="Overview", image=fixtures.image_bytes()
    )
    doc = await extract_document(data, "brief.docx", ocr=_OFF)
    assert doc.text.startswith("## Overview")
    assert "Widgets are great." in doc.text
    assert len(doc.images) == 1
    assert doc.images[0].mime_type == "image/png"


async def test_docx_small_images_are_dropped() -> None:
    data = fixtures.docx(["Body text."], image=fixtures.image_bytes(40, 40))
    doc = await extract_document(data, "brief.docx", ocr=_OFF)
    assert doc.images == []


async def test_epub_chapters_and_image() -> None:
    data = fixtures.epub(["Once upon a time.", "The end."], image=fixtures.image_bytes())
    doc = await extract_document(data, "book.epub", ocr=_OFF)
    assert "Once upon a time." in doc.text
    assert "The end." in doc.text
    assert len(doc.images) == 1


async def test_markdown_is_kept_verbatim() -> None:
    doc = await extract_document(b"# Title\n\nSee https://acme.com now.", "n.md", ocr=_OFF)
    # Grounding keeps URLs as written (the audiobook TTS normaliser is NOT applied).
    assert doc.text == "# Title\n\nSee https://acme.com now."


async def test_empty_docx_raises() -> None:
    with pytest.raises(DocumentExtractError, match="no extractable text"):
        await extract_document(fixtures.docx([]), "empty.docx", ocr=_OFF)


async def test_non_web_image_formats_are_reencoded_to_png() -> None:
    """The stage stores png/jpg/webp only, so a GIF/BMP is re-encoded, not dropped."""
    from src.services.documents.extract import normalise_image

    gif = normalise_image(fixtures.image_bytes(fmt="GIF"))
    assert gif is not None
    assert gif.mime_type == "image/png"
    assert gif.data.startswith(b"\x89PNG")


async def test_pdf_image_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(extract_mod, "MAX_IMAGES", 0)
    data = fixtures.text_pdf(["Some text here."], jpeg=fixtures.image_bytes(300, 300, "JPEG"))
    doc = await extract_document(data, "deck.pdf", ocr=_OFF)
    assert doc.images == []
    assert MAX_IMAGES == 12


# ─── hostile input: decompression bombs ────────────────────────────────────────

_MIB = 1 << 20


async def test_docx_bomb_over_the_total_cap_is_rejected_before_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(archive_guard, "MAX_ARCHIVE_UNCOMPRESSED_BYTES", _MIB)
    monkeypatch.setattr(archive_guard, "RATIO_CHECK_MIN_BYTES", 1 << 40)  # isolate the total cap
    bomb = fixtures.pad_zip_member(fixtures.docx(["hello"]), "word/document.xml", 2 * _MIB)
    with pytest.raises(DocumentExtractError, match="uncompressed"):
        await extract_document(bomb, "bomb.docx")


async def test_docx_member_with_extreme_compression_ratio_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(archive_guard, "RATIO_CHECK_MIN_BYTES", _MIB)
    bomb = fixtures.pad_zip_member(fixtures.docx(["hello"]), "word/document.xml", 2 * _MIB)
    with pytest.raises(DocumentExtractError, match="compression ratio"):
        await extract_document(bomb, "bomb.docx")


async def test_epub_bomb_is_rejected_on_the_magic_sniff_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(archive_guard, "RATIO_CHECK_MIN_BYTES", _MIB)
    bomb = fixtures.pad_zip_member(fixtures.epub(["chapter"]), "EPUB/ch1.xhtml", 2 * _MIB)
    assert b"mimetypeapplication/epub+zip" in bomb[:128]  # the early-return sniff path
    with pytest.raises(DocumentExtractError, match="compression ratio"):
        await extract_document(bomb, "bomb.epub")


async def test_zip_with_absurd_member_count_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(archive_guard, "MAX_ARCHIVE_MEMBERS", 40)
    data = fixtures.zip_with_members(fixtures.docx(["hello"]), 50)
    with pytest.raises(DocumentExtractError, match="members"):
        await extract_document(data, "many.docx")


async def test_pdf_stops_at_the_decoded_content_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """pypdf bounds each stream (75 MB) but not their SUM across pages."""
    monkeypatch.setattr(extract_mod, "MAX_PDF_CONTENT_BYTES", 3 * _MIB)
    data = fixtures.padded_content_pdf([2 * _MIB] * 4)
    doc = await extract_document(data, "dense.pdf", ocr=_OFF)
    assert "Page 1 text" in doc.text
    assert "Page 4 text" not in doc.text
    assert any("decoded content budget" in n for n in doc.notes)


async def test_pdf_oversized_content_stream_is_skipped_and_noted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(extract_mod, "MAX_PDF_STREAM_BYTES", _MIB)
    data = fixtures.padded_content_pdf([0, 2 * _MIB, 0])
    doc = await extract_document(data, "mixed.pdf", ocr=_OFF)
    assert "Page 1 text" in doc.text
    assert "Page 2 text" not in doc.text
    assert "Page 3 text" in doc.text
    assert any("page 2" in n and "exceeds" in n for n in doc.notes)


async def test_pdf_with_only_oversized_streams_fails_naming_the_cap_and_skips_ocr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(extract_mod, "MAX_PDF_STREAM_BYTES", _MIB)
    fake = _FakeOcr()
    with pytest.raises(DocumentExtractError, match="exceeds 1 MiB"):
        await extract_document(fixtures.padded_content_pdf([2 * _MIB]), "huge.pdf", ocr=_ocr(fake))
    assert fake.requests == []  # an oversized page is not "scanned" — never shipped to OCR
