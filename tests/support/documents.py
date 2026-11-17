"""Generate real document fixtures (pdf / docx / epub) at test time.

No binary fixtures are checked in: each builder writes a genuine file the
production parsers read (pypdf, python-docx, EbookLib), so a parser upgrade that
changes behaviour is caught against real bytes, not a mock.
"""

from __future__ import annotations

import io

from PIL import Image


def image_bytes(width: int = 200, height: int = 200, fmt: str = "PNG") -> bytes:
    """A noisy (so not trivially compressible, i.e. > 4 KiB) image of ``fmt``."""
    img = Image.effect_noise((width, height), 64).convert("RGB")
    out = io.BytesIO()
    img.save(out, format=fmt)
    return out.getvalue()


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def text_pdf(pages: list[str | None], *, jpeg: bytes | None = None) -> bytes:
    """A PDF whose page i carries ``pages[i]`` as a real text layer.

    ``None`` makes that page carry NO text (a "scanned" page). ``jpeg`` (raw JPEG
    bytes) is drawn as an image XObject on the first page.
    """
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    catalog = add(b"")  # placeholder, filled below
    pages_obj = add(b"")
    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    image_obj = None
    if jpeg is not None:
        with Image.open(io.BytesIO(jpeg)) as im:
            w, h = im.size
        image_obj = add(
            f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} "
            f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode "
            f"/Length {len(jpeg)} >>\nstream\n".encode()
            + jpeg
            + b"\nendstream"
        )
    kids: list[int] = []
    for i, text in enumerate(pages):
        ops = []
        if text:
            y = 760
            for line in text.splitlines() or [""]:
                ops.append(f"BT /F1 12 Tf 50 {y} Td ({_escape(line)}) Tj ET")
                y -= 16
        xobj = ""
        if image_obj is not None and i == 0:
            ops.append("q 200 0 0 200 50 300 cm /Im1 Do Q")
            xobj = f" /XObject << /Im1 {image_obj} 0 R >>"
        stream = "\n".join(ops).encode()
        content = add(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        page = add(
            f"<< /Type /Page /Parent {pages_obj} 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font} 0 R >>{xobj} >> "
            f"/Contents {content} 0 R >>".encode()
        )
        kids.append(page)
    objects[catalog - 1] = f"<< /Type /Catalog /Pages {pages_obj} 0 R >>".encode()
    objects[pages_obj - 1] = (
        f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in kids)}] /Count {len(kids)} >>"
    ).encode()

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for n, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(f"{n} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog} 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n".encode()
    )
    return out.getvalue()


def scanned_pdf(n_pages: int = 2) -> bytes:
    """An image-only PDF (no text layer on any page), as a scanner produces."""
    frames = [Image.effect_noise((300, 400), 32).convert("RGB") for _ in range(n_pages)]
    out = io.BytesIO()
    frames[0].save(out, format="PDF", save_all=True, append_images=frames[1:])
    return out.getvalue()


def encrypted_pdf(password: str = "s3cret") -> bytes:  # noqa: S107 -- test fixture password
    """A text PDF protected by a non-empty USER password."""
    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(text_pdf(["Top secret body."]))))
    writer.encrypt(user_password=password, owner_password=password)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def docx(paragraphs: list[str], *, heading: str | None = None, image: bytes | None = None) -> bytes:
    """A .docx with an optional heading, paragraphs and one inline image."""
    import docx as python_docx
    from docx.shared import Inches

    document = python_docx.Document()
    if heading:
        document.add_heading(heading, level=1)
    for para in paragraphs:
        document.add_paragraph(para)
    if image is not None:
        document.add_picture(io.BytesIO(image), width=Inches(2))
    out = io.BytesIO()
    document.save(out)
    return out.getvalue()


def epub(chapters: list[str], *, image: bytes | None = None) -> bytes:
    """An .epub with one XHTML document per chapter and an optional image."""
    import tempfile
    from pathlib import Path

    from ebooklib import epub as ebook

    book = ebook.EpubBook()
    book.set_identifier("test-book")
    book.set_title("Test Book")
    book.set_language("en")
    items = []
    for i, body in enumerate(chapters, start=1):
        item = ebook.EpubHtml(title=f"Chapter {i}", file_name=f"ch{i}.xhtml", lang="en")
        item.content = f"<html><body><h1>Chapter {i}</h1><p>{body}</p></body></html>"
        book.add_item(item)
        items.append(item)
    if image is not None:
        book.add_item(
            ebook.EpubItem(
                uid="img1", file_name="images/pic.png", media_type="image/png", content=image
            )
        )
    book.toc = list(items)
    book.spine = ["nav", *items]
    book.add_item(ebook.EpubNcx())
    book.add_item(ebook.EpubNav())
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "book.epub"
        ebook.write_epub(str(path), book)
        return path.read_bytes()


def pad_zip_member(archive: bytes, member: str, pad_bytes: int) -> bytes:
    """``archive`` with ``pad_bytes`` trailing spaces appended to ``member``.

    Trailing whitespace after an XML root is still well-formed, so a padded
    ``word/document.xml`` stays a readable docx — a decompression bomb whose
    inflated size is ``pad_bytes`` and whose compressed size is ~1/1000 of that.
    """
    import zipfile

    src = zipfile.ZipFile(io.BytesIO(archive))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as dst:
        for info in src.infolist():
            data = src.read(info.filename)
            if info.filename != member:
                dst.writestr(info, data)
                continue
            with dst.open(info.filename, "w", force_zip64=True) as fh:
                fh.write(data)
                chunk = b" " * (1 << 16)
                for _ in range(pad_bytes // len(chunk)):
                    fh.write(chunk)
    return out.getvalue()


def zip_with_members(base: bytes, extra_members: int) -> bytes:
    """``base`` (a zip) plus ``extra_members`` tiny junk members."""
    import zipfile

    out = io.BytesIO(base)
    with zipfile.ZipFile(out, "a") as dst:
        for i in range(extra_members):
            dst.writestr(f"junk/{i}.txt", b"x")
    return out.getvalue()


def padded_content_pdf(pads: list[int]) -> bytes:
    """A PDF with one page per ``pads`` entry; page N's FlateDecode content stream
    is a one-line ``Page N text`` op followed by ``pads[N-1]`` bytes of whitespace.
    """
    pages = len(pads)
    import zlib

    objs: list[bytes] = [b"<< /Type /Catalog /Pages 2 0 R >>"]
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(pages))
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {pages} >>".encode())
    font = 3 + 2 * pages
    for i in range(pages):
        comp = zlib.compressobj(9)
        body = comp.compress(f"BT /F1 12 Tf 72 720 Td (Page {i + 1} text) Tj ET\n".encode())
        chunk = b" " * (1 << 16)
        body += b"".join(comp.compress(chunk) for _ in range(pads[i] // len(chunk)))
        body += comp.flush()
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {4 + 2 * i} 0 R "
            f"/Resources << /Font << /F1 {font} 0 R >> >> >>".encode()
        )
        objs.append(
            f"<< /Length {len(body)} /Filter /FlateDecode >>\nstream\n".encode()
            + body
            + b"\nendstream"
        )
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = io.BytesIO()
    out.write(b"%PDF-1.7\n")
    offsets: list[int] = []
    for n, obj in enumerate(objs, 1):
        offsets.append(out.tell())
        out.write(f"{n} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode())
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(
        f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    return out.getvalue()
