"""In-process document extraction for ``source_ingest`` document sources.

Replaces KAG's ``/ingest/documents/sync`` (retired with KAG). KAG's own parse was
pure-Python CPU code with no OCR, so this is a like-for-like move into the
orchestrator plus OCR for scanned PDFs:

* **pdf** — ``pypdf``: per-page ``extract_text()``; embedded images via
  ``page.images``. A page whose text layer is empty/negligible is OCR'd through
  :class:`~src.clients.ocr_client.OcrClient` (provider_contracts ``/v1/ocr``,
  mlx_api for now). Those pages are copied into a small sub-PDF with ``pypdf``'s
  writer and sent as a ``data:application/pdf`` document — the OCR contract
  accepts PDFs natively and the PROVIDER renders the pages (mlx_api uses
  pypdfium2), so the orchestrator needs no renderer dependency.
* **docx** — python-docx paragraphs (headings as ``## ``) + table cells; images
  from the document part's image relationships.
* **epub** — EbookLib ``ITEM_DOCUMENT`` bodies (nav skipped); ``ITEM_IMAGE`` images.
* **txt / md / markdown** — UTF-8 text verbatim.

The kind is sniffed from the bytes first (``%PDF-``, zip members) and only falls
back to the filename extension for plain text, so a mislabelled upload is still
parsed by the right parser.

Failure is LOUD: an unsupported/corrupt/password-protected document, or one that
yields no text, raises :class:`DocumentExtractError` — never an empty success. A
PDF with no text layer and no OCR configured raises naming the missing setting.
A partially-recovered document (some pages OCR'd, some not) succeeds and says so
in ``notes``.

Bounded against hostile input: zip containers pass
:func:`~src.services.documents.archive_guard.check_zip_archive` (decompression
bombs) before any parser sees them; parsing runs in a worker thread, at most
:data:`MAX_PDF_PAGES` pages are read, each decoded PDF stream is capped at
:data:`MAX_PDF_STREAM_BYTES` and the pages' decoded content in sum at
:data:`MAX_PDF_CONTENT_BYTES`, text extraction stops once
:data:`TEXT_BUDGET_CHARS` are collected (the stage keeps 12 000 per source),
at most :data:`MAX_OCR_PAGES` pages are OCR'd and :data:`MAX_IMAGES` images kept.
"""

from __future__ import annotations

import asyncio
import io
import logging
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal

from PIL import Image

from src.clients.ocr_client import OcrClient, OcrError, OcrTarget
from src.services.documents.archive_guard import UnsafeArchiveError, check_zip_archive

if TYPE_CHECKING:
    from docx.document import Document as DocxDocument
    from pypdf import Configuration, PageObject, PdfReader

log = logging.getLogger(__name__)

DocumentKind = Literal["pdf", "docx", "epub", "text"]

#: Pages read from a PDF at most (a hostile 100k-page PDF cannot pin a thread).
MAX_PDF_PAGES: Final = 500
#: Any ONE decoded PDF stream (content, image, xform) at most. pypdf's own
#: per-stream default is 75 MB; tightened because a page's text never needs more.
MAX_PDF_STREAM_BYTES: Final = 32 * 1024 * 1024
#: Decoded page-content bytes summed over the document at most. pypdf bounds each
#: stream but not their SUM: 40 pages x 70 MB of deflated whitespace (2.9 MB file)
#: took 220 s of CPU and 1.0 GB RSS, and a 25 MiB upload fits ~350 such pages.
MAX_PDF_CONTENT_BYTES: Final = 128 * 1024 * 1024
#: Stop extracting once this much text is collected — 4x the stage's per-source cap.
TEXT_BUDGET_CHARS: Final = 48_000
#: A page with fewer non-whitespace chars than this has no usable text layer.
MIN_PAGE_TEXT_CHARS: Final = 16
#: Scanned pages OCR'd per document at most, and per OCR request.
MAX_OCR_PAGES: Final = 40
OCR_BATCH_PAGES: Final = 8
#: Images kept per document (the stage's own cap), and candidates examined.
MAX_IMAGES: Final = 12
_MAX_IMAGE_CANDIDATES: Final = 64
#: Icons, bullets and spacer GIFs are not illustrative: drop small images.
MIN_IMAGE_SIDE_PX: Final = 128
MIN_IMAGE_BYTES: Final = 4 * 1024

_TEXT_EXTENSIONS: Final = frozenset({"txt", "md", "markdown"})
_KEEP_FORMATS: Final = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}


class DocumentExtractError(RuntimeError):
    """The document could not be turned into any grounding text."""


@dataclass(frozen=True, slots=True)
class ExtractedImage:
    """An illustrative image from the document, as raw bytes of ``mime_type``."""

    data: bytes
    mime_type: str
    caption: str = ""
    alt_text: str = ""


@dataclass(slots=True)
class ExtractedDocument:
    """What a document contributes to grounding."""

    text: str
    references: list[str] = field(default_factory=list)
    images: list[ExtractedImage] = field(default_factory=list)
    #: Partial-extraction facts the run must show (e.g. pages left un-OCR'd).
    notes: list[str] = field(default_factory=list)


# ─── Kind detection ────────────────────────────────────────────────────────────


def _ext(filename: str) -> str:
    _, dot, ext = filename.lower().rpartition(".")
    return ext if dot else ""


def _zip_kind(data: bytes, filename: str) -> DocumentKind:
    if b"mimetypeapplication/epub+zip" in data[:128]:
        return "epub"
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = set(zf.namelist())
    except zipfile.BadZipFile as exc:
        raise DocumentExtractError(f"{filename!r} is a corrupt zip container") from exc
    if "word/document.xml" in names:
        return "docx"
    if "META-INF/container.xml" in names:
        return "epub"
    raise DocumentExtractError(f"{filename!r} is a zip archive, not a docx or epub")


def _is_utf8_text(data: bytes) -> bool:
    if b"\x00" in data[:1024]:
        return False
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def detect_kind(data: bytes, filename: str) -> DocumentKind:
    """Sniff the document kind from its bytes; the extension decides only for text."""
    if b"%PDF-" in data[:1024]:
        return "pdf"
    if data[:4] == b"PK\x03\x04":
        try:
            check_zip_archive(data, filename)
        except UnsafeArchiveError as exc:
            raise DocumentExtractError(str(exc)) from exc
        return _zip_kind(data, filename)
    if _ext(filename) in _TEXT_EXTENSIONS or _is_utf8_text(data):
        return "text"
    raise DocumentExtractError(
        f"unsupported document type for {filename!r} (expected pdf, docx, epub, txt or md)"
    )


# ─── Images ────────────────────────────────────────────────────────────────────


def normalise_image(raw: bytes, *, alt_text: str = "") -> ExtractedImage | None:
    """``raw`` as a PNG/JPEG/WEBP image if it is big enough to illustrate, else None."""
    if len(raw) < MIN_IMAGE_BYTES:
        return None
    try:
        with Image.open(io.BytesIO(raw)) as img:
            if min(img.size) < MIN_IMAGE_SIDE_PX:
                return None
            fmt = img.format or ""
            if fmt in _KEEP_FORMATS:
                return ExtractedImage(data=raw, mime_type=_KEEP_FORMATS[fmt], alt_text=alt_text)
            out = io.BytesIO()
            converted = img if img.mode in {"RGB", "RGBA", "L", "LA"} else img.convert("RGBA")
            converted.save(out, format="PNG")
            return ExtractedImage(data=out.getvalue(), mime_type="image/png", alt_text=alt_text)
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        log.info("document image skipped (undecodable): %s", type(exc).__name__)
        return None


# ─── PDF ───────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class _PdfParse:
    reader: PdfReader
    page_texts: list[str]
    images: list[ExtractedImage]
    notes: list[str]
    #: Pages skipped because a content stream exceeds MAX_PDF_STREAM_BYTES. They
    #: are not "scanned": never OCR'd (the sub-PDF would carry the same stream).
    oversized: list[int] = field(default_factory=list)

    def blank_pages(self) -> list[int]:
        return [
            i
            for i, text in enumerate(self.page_texts)
            if i not in self.oversized and len("".join(text.split())) < MIN_PAGE_TEXT_CHARS
        ]


def _open_pdf(data: bytes, filename: str) -> PdfReader:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise DocumentExtractError(
                f"{filename!r} is a password-protected PDF; upload an unprotected copy"
            )
        _ = len(reader.pages)  # forces the page tree; a broken xref raises here
    except DocumentExtractError:
        raise
    except (PdfReadError, ValueError, KeyError, TypeError, OSError) as exc:
        raise DocumentExtractError(f"{filename!r} is not a readable PDF: {exc}") from exc
    return reader


def _page_content_bytes(page: PageObject, idx: int) -> int | None:
    """Decoded size of ``page``'s content stream(s); None when one exceeds the cap.

    Costs one extra inflate per page (bounded by the stream cap) in exchange for
    knowing the size BEFORE ``extract_text`` parses it into Python objects.
    """
    from pypdf.errors import LimitReachedError

    try:
        contents = page.get_contents()
        return len(contents.get_data()) if contents is not None else 0
    except LimitReachedError:
        return None
    except Exception as exc:  # noqa: BLE001 -- extract_text will skip it the same way
        log.info("pdf page %d content decode failed: %s", idx, type(exc).__name__)
        return 0


def _pdf_limits() -> Configuration:
    """The current pypdf ``Configuration`` with every decoded stream capped."""
    from pypdf import get_configuration

    cap = MAX_PDF_STREAM_BYTES
    return get_configuration().with_overwrites(
        zlib_maximum_output_length=cap,
        lzw_maximum_output_length=cap,
        run_length_maximum_output_length=cap,
        brotli_maximum_output_length=cap,
        jbig2_maximum_output_length=cap,
        array_based_stream_maximum_output_length=cap,
        image_maximum_buffer_size=cap,
    )


def _page_text(page: PageObject, idx: int) -> str:
    try:
        return (page.extract_text() or "").strip()
    except Exception as exc:  # noqa: BLE001 -- one bad page must not lose the document
        log.info("pdf page %d text extraction failed: %s", idx, type(exc).__name__)
        return ""


def _page_images(page: PageObject, idx: int, *, budget: int) -> tuple[list[ExtractedImage], int]:
    """Up to ``budget`` candidate images of ``page``; returns (kept, examined)."""
    try:
        page_images = list(page.images)
    except Exception as exc:  # noqa: BLE001 -- an undecodable image stream is skippable
        log.info("pdf page %d image listing failed: %s", idx, type(exc).__name__)
        return [], 0
    kept: list[ExtractedImage] = []
    examined = 0
    for pdf_image in page_images[:budget]:
        examined += 1
        try:
            raw = pdf_image.data
        except Exception:  # noqa: BLE001,S112 -- skip this one image, keep the rest
            continue
        image = normalise_image(raw)
        if image is not None:
            kept.append(image)
    return kept, examined


def _parse_pdf(data: bytes, filename: str) -> _PdfParse:
    from pypdf import apply_configuration

    # The ContextVar override lives only in this worker thread's context copy.
    with apply_configuration(_pdf_limits()):
        return _parse_pdf_bounded(data, filename)


def _parse_pdf_bounded(data: bytes, filename: str) -> _PdfParse:
    reader = _open_pdf(data, filename)
    notes: list[str] = []
    total = len(reader.pages)
    last = min(total, MAX_PDF_PAGES)
    if total > MAX_PDF_PAGES:
        notes.append(f"{filename}: only the first {MAX_PDF_PAGES} of {total} pages were read")
    page_texts: list[str] = []
    images: list[ExtractedImage] = []
    examined = 0
    collected = 0
    decoded = 0
    oversized: list[int] = []
    mib = 1024 * 1024
    for idx in range(last):
        page = reader.pages[idx]
        size = _page_content_bytes(page, idx)
        if size is None:
            notes.append(
                f"{filename}: page {idx + 1} has a content stream that exceeds "
                f"{MAX_PDF_STREAM_BYTES // mib} MiB decoded; its text was skipped"
            )
            page_texts.append("")
            oversized.append(idx)
            continue
        decoded += size
        if decoded > MAX_PDF_CONTENT_BYTES:
            notes.append(
                f"{filename}: text extraction stopped before page {idx + 1} (decoded content "
                f"budget of {MAX_PDF_CONTENT_BYTES // mib} MiB reached)"
            )
            break
        text = _page_text(page, idx)
        page_texts.append(text)
        collected += len(text)
        if len(images) < MAX_IMAGES and examined < _MAX_IMAGE_CANDIDATES:
            kept, seen = _page_images(page, idx, budget=_MAX_IMAGE_CANDIDATES - examined)
            images.extend(kept[: MAX_IMAGES - len(images)])
            examined += seen
        if collected >= TEXT_BUDGET_CHARS and idx + 1 < last:
            notes.append(
                f"{filename}: text extraction stopped after page {idx + 1} "
                f"(grounding budget of {TEXT_BUDGET_CHARS} chars reached)"
            )
            break
    if oversized and len(oversized) == len(page_texts):
        raise DocumentExtractError(
            f"{filename!r}: every page read has a content stream that exceeds "
            f"{MAX_PDF_STREAM_BYTES // mib} MiB decoded, so no text was extracted"
        )
    return _PdfParse(
        reader=reader, page_texts=page_texts, images=images, notes=notes, oversized=oversized
    )


def _sub_pdf(reader: PdfReader, pages: list[int]) -> bytes:
    """A PDF holding only ``pages`` of ``reader`` (in order) — the OCR payload."""
    from pypdf import PdfWriter

    writer = PdfWriter()
    for idx in pages:
        writer.add_page(reader.pages[idx])
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


async def _ocr_pages(
    parsed: _PdfParse, pages: list[int], ocr: OcrClient, target: OcrTarget
) -> dict[int, str]:
    """OCR ``pages`` in batches; returns ``{page_index: text}`` for pages with text."""
    recovered: dict[int, str] = {}
    for start in range(0, len(pages), OCR_BATCH_PAGES):
        batch = pages[start : start + OCR_BATCH_PAGES]
        payload = await asyncio.to_thread(_sub_pdf, parsed.reader, batch)
        for page in await ocr.ocr_pdf(payload, target=target):
            if 0 <= page.index < len(batch) and page.markdown.strip():
                recovered[batch[page.index]] = page.markdown.strip()
    return recovered


def _no_ocr_notes(parsed: _PdfParse, filename: str) -> list[str]:
    """OCR is off: fail a text-less PDF loudly; note pages left without text."""
    empty = [i for i, t in enumerate(parsed.page_texts) if not t.strip()]
    if len(empty) == len(parsed.page_texts):
        raise DocumentExtractError(
            f"{filename!r} has no text layer (a scanned or image-only PDF) and OCR "
            "is not configured (W2V_PROVIDER_OCR_URL is empty), so no text could "
            "be extracted"
        )
    if not empty:
        return []
    return [
        f"{filename}: {len(empty)} page(s) without a text layer were not OCR'd "
        "(OCR is not configured)"
    ]


async def _apply_ocr(
    parsed: _PdfParse, texts: list[str], filename: str, ocr: OcrClient, target: OcrTarget
) -> list[str]:
    """OCR the pages with a negligible text layer into ``texts``; returns notes."""
    candidates = parsed.blank_pages()
    if not candidates:
        return []
    notes: list[str] = []
    to_ocr = candidates[:MAX_OCR_PAGES]
    if len(candidates) > MAX_OCR_PAGES:
        notes.append(
            f"{filename}: only the first {MAX_OCR_PAGES} of {len(candidates)} pages "
            "without a text layer were OCR'd"
        )
    try:
        recovered = await _ocr_pages(parsed, to_ocr, ocr, target)
    except OcrError as exc:
        if not any(t.strip() for t in texts):
            raise DocumentExtractError(
                f"{filename!r} has no text layer and OCR failed: {exc}"
            ) from exc
        return [*notes, f"{filename}: OCR of {len(to_ocr)} scanned page(s) failed: {exc}"]
    for idx, text in recovered.items():
        if len(text) > len(texts[idx]):
            texts[idx] = text
    log.info("pdf %s: OCR recovered text on %d/%d page(s)", filename, len(recovered), len(to_ocr))
    return notes


async def _extract_pdf(data: bytes, filename: str, ocr: OcrClient) -> ExtractedDocument:
    parsed = await asyncio.to_thread(_parse_pdf, data, filename)
    texts = list(parsed.page_texts)
    # The ``ocr`` slot is resolved only when a page actually needs OCR, so a PDF
    # with a full text layer never depends on the tenant's OCR provider / posture.
    target = await ocr.target() if parsed.blank_pages() else None
    if target is not None:
        ocr_notes = await _apply_ocr(parsed, texts, filename, ocr, target)
    else:
        ocr_notes = _no_ocr_notes(parsed, filename)
    text = "\n\n".join(t for t in texts if t.strip()).strip()
    if not text:
        raise DocumentExtractError(f"{filename!r} yielded no text (not even after OCR)")
    return ExtractedDocument(text=text, images=parsed.images, notes=[*parsed.notes, *ocr_notes])


# ─── docx / epub / text ────────────────────────────────────────────────────────


def _docx_lines(document: DocxDocument) -> list[str]:
    lines: list[str] = []
    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = (para.style.name if para.style else "") or ""
        lines.append(f"## {text}" if style.startswith(("Heading", "Title")) else text)
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                lines.append(" | ".join(dict.fromkeys(cells)))
    return lines


def _docx_images(document: DocxDocument) -> list[ExtractedImage]:
    images: list[ExtractedImage] = []
    for part in list(document.part.related_parts.values())[:_MAX_IMAGE_CANDIDATES]:
        if len(images) >= MAX_IMAGES:
            break
        if str(getattr(part, "content_type", "")).startswith("image/"):
            kept = normalise_image(part.blob)
            if kept is not None:
                images.append(kept)
    return images


def _extract_docx(data: bytes, filename: str) -> ExtractedDocument:
    import docx  # python-docx

    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:
        raise DocumentExtractError(f"{filename!r} is not a readable docx: {exc}") from exc
    return ExtractedDocument(
        text="\n\n".join(_docx_lines(document)).strip(), images=_docx_images(document)
    )


def _extract_epub(data: bytes, filename: str) -> ExtractedDocument:
    import ebooklib
    from bs4 import BeautifulSoup
    from ebooklib import epub

    try:
        book = epub.read_epub(io.BytesIO(data))
    except Exception as exc:
        raise DocumentExtractError(f"{filename!r} is not a readable epub: {exc}") from exc
    parts: list[str] = []
    collected = 0
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        if isinstance(item, epub.EpubNav):
            continue
        text = BeautifulSoup(item.get_content(), "html.parser").get_text("\n", strip=True)
        if text:
            parts.append(text)
            collected += len(text)
        if collected >= TEXT_BUDGET_CHARS:
            break
    images: list[ExtractedImage] = []
    for item in list(book.get_items_of_type(ebooklib.ITEM_IMAGE))[:_MAX_IMAGE_CANDIDATES]:
        if len(images) >= MAX_IMAGES:
            break
        kept = normalise_image(item.get_content())
        if kept is not None:
            images.append(kept)
    return ExtractedDocument(text="\n\n".join(parts).strip(), images=images)


def _extract_text(data: bytes, filename: str) -> ExtractedDocument:
    return ExtractedDocument(text=data.decode("utf-8-sig", errors="replace").strip())


_SYNC_EXTRACTORS: dict[str, Callable[[bytes, str], ExtractedDocument]] = {
    "docx": _extract_docx,
    "epub": _extract_epub,
    "text": _extract_text,
}


async def extract_document(
    data: bytes, filename: str, *, ocr: OcrClient | None = None
) -> ExtractedDocument:
    """Extract grounding text (+ illustrative images) from an uploaded document.

    ``ocr`` defaults to :meth:`OcrClient.for_bound_run` — the bound run's ``ocr``
    provider slot (tenant connection or the platform ``W2V_PROVIDER_OCR_URL``, per
    the tenant's ``byo_mode``; platform OCR is off when that URL is empty). Raises :class:`DocumentExtractError` when
    no text can be extracted, naming why.
    """
    if not data:
        raise DocumentExtractError(f"{filename!r} is empty")
    kind = detect_kind(data, filename)
    if kind == "pdf":
        return await _extract_pdf(
            data, filename, ocr if ocr is not None else OcrClient.for_bound_run()
        )
    doc = await asyncio.to_thread(_SYNC_EXTRACTORS[kind], data, filename)
    if not doc.text:
        raise DocumentExtractError(f"{filename!r} contains no extractable text")
    return doc
