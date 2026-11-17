"""Pure-function tests for manuscript parsing + TTS chunking (no I/O)."""

from __future__ import annotations

import io

import pytest

from src.services.documents import archive_guard
from src.services.stages.audiobook.parsing import (
    DEFAULT_MAX_CHUNK_CHARS,
    ManuscriptParseError,
    chunk_text,
    normalize_markdown_for_tts,
    parse_manuscript,
)
from tests.support import documents as fixtures


def test_txt_splits_on_chapter_headings() -> None:
    text = (
        "Chapter 1\nThe beginning of it all.\n\n"
        "Chapter 2\nThe middle stretched on.\n\n"
        "Chapter 3\nAnd then the end."
    )
    m = parse_manuscript(text.encode(), "book.txt")
    assert [c.title for c in m.chapters] == ["Chapter 1", "Chapter 2", "Chapter 3"]
    assert [c.sequence for c in m.chapters] == [0, 1, 2]
    assert "beginning" in m.chapters[0].text


def test_txt_markdown_headings() -> None:
    text = "# Prologue\nOnce upon a time.\n\n## Closing\nThe end."
    m = parse_manuscript(text.encode(), "book.txt")
    assert [c.title for c in m.chapters] == ["Prologue", "Closing"]


def test_txt_without_headings_is_one_chapter() -> None:
    text = "Just a flat block of prose with no headings at all."
    m = parse_manuscript(text.encode(), "story.txt", fallback_title="My Story")
    assert len(m.chapters) == 1
    assert m.chapters[0].title == "My Story"
    assert m.chapters[0].text == text


def test_single_chapter_mode_merges() -> None:
    text = "Chapter 1\nAlpha.\n\nChapter 2\nBeta."
    m = parse_manuscript(text.encode(), "book.txt", single_chapter=True, fallback_title="T")
    assert len(m.chapters) == 1
    assert "Alpha." in m.chapters[0].text and "Beta." in m.chapters[0].text


def test_empty_manuscript_raises() -> None:
    with pytest.raises(ManuscriptParseError):
        parse_manuscript(b"   \n\n  ", "empty.txt")


def test_unsupported_extension_raises() -> None:
    with pytest.raises(ManuscriptParseError, match="unsupported"):
        parse_manuscript(b"x", "book.pdf")


def test_docx_round_trip() -> None:
    import docx

    document = docx.Document()
    document.add_heading("Chapter One", level=1)
    document.add_paragraph("The body of chapter one.")
    document.add_heading("Chapter Two", level=1)
    document.add_paragraph("The body of chapter two.")
    buf = io.BytesIO()
    document.save(buf)

    m = parse_manuscript(buf.getvalue(), "book.docx")
    assert [c.title for c in m.chapters] == ["Chapter One", "Chapter Two"]
    assert "chapter one" in m.chapters[0].text.lower()


def test_epub_round_trip(tmp_path) -> None:
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_title("My Book")
    book.add_author("Jane Author")
    c1 = epub.EpubHtml(title="Intro", file_name="c1.xhtml")
    c1.content = "<h1>Intro</h1><p>Hello world from chapter one.</p>"
    c2 = epub.EpubHtml(title="Outro", file_name="c2.xhtml")
    c2.content = "<h1>Outro</h1><p>Goodbye from chapter two.</p>"
    book.add_item(c1)
    book.add_item(c2)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = [c1, c2]
    path = tmp_path / "book.epub"
    epub.write_epub(str(path), book)

    m = parse_manuscript(path.read_bytes(), "book.epub")
    assert m.title == "My Book"
    assert m.author == "Jane Author"
    assert [c.title for c in m.chapters] == ["Intro", "Outro"]
    assert "Hello world" in m.chapters[0].text


def test_lexicon_applies_to_txt() -> None:
    text = "Chapter 1\nKoster met the CFO at the café."
    m = parse_manuscript(text.encode(), "book.txt", lexicon={"Koster": "KOSS-ter", "café": "cafay"})
    assert "KOSS-ter" in m.chapters[0].text
    assert "cafay" in m.chapters[0].text
    assert "Koster" not in m.chapters[0].text


def test_lexicon_applies_to_docx() -> None:
    import docx

    document = docx.Document()
    document.add_heading("Chapter One", level=1)
    document.add_paragraph("Merqi spoke with de Jong.")
    buf = io.BytesIO()
    document.save(buf)

    m = parse_manuscript(
        buf.getvalue(), "book.docx", lexicon={"Merqi": "MER-kee", "de Jong": "duh YONG"}
    )
    assert "MER-kee" in m.chapters[0].text
    assert "duh YONG" in m.chapters[0].text


def test_lexicon_applies_to_epub(tmp_path) -> None:
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_title("My Book")
    c1 = epub.EpubHtml(title="Intro", file_name="c1.xhtml")
    c1.content = "<h1>Intro</h1><p>Doorn walked past Meridian tower.</p>"
    book.add_item(c1)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = [c1]
    path = tmp_path / "book.epub"
    epub.write_epub(str(path), book)

    m = parse_manuscript(
        path.read_bytes(), "book.epub", lexicon={"Doorn": "Dorn", "Meridian": "muh-RID-ee-un"}
    )
    assert "Dorn" in m.chapters[0].text
    assert "muh-RID-ee-un" in m.chapters[0].text


def test_markdown_strip_still_applies_all_formats() -> None:
    # Markdown-strip is now centralized; a **bold** run narrates as plain prose,
    # and lexicon runs after strip so it sees the clean word.
    text = "Chapter 1\n**Koster** paused."
    m = parse_manuscript(text.encode(), "book.txt", lexicon={"Koster": "KOSS-ter"})
    assert "**" not in m.chapters[0].text
    assert "KOSS-ter" in m.chapters[0].text


def test_chunk_text_packs_sentences() -> None:
    text = "One sentence. Two sentence. Three sentence. Four."
    chunks = chunk_text(text, max_chars=25)
    assert chunks  # something produced
    assert all(len(c) <= 25 for c in chunks)
    # Round-trips the words in order.
    assert " ".join(chunks).replace("  ", " ").split() == text.split()


def test_chunk_text_hard_splits_long_sentence() -> None:
    long_sentence = "word " * 100  # 500 chars, no sentence breaks
    chunks = chunk_text(long_sentence.strip(), max_chars=50)
    assert len(chunks) > 1
    assert all(len(c) <= 50 for c in chunks)


def test_chunk_text_default_cap_under_tts_limit() -> None:
    assert DEFAULT_MAX_CHUNK_CHARS <= 10_000
    chunks = chunk_text("A. " * 5000, max_chars=DEFAULT_MAX_CHUNK_CHARS)
    assert all(len(c) <= DEFAULT_MAX_CHUNK_CHARS for c in chunks)


def test_chunk_text_empty() -> None:
    assert chunk_text("   \n\t ") == []


# --- normalize_markdown_for_tts ---------------------------------------------


def test_normalize_strips_emphasis_keeping_words() -> None:
    assert normalize_markdown_for_tts("She said *cancel* and **stop** now.") == (
        "She said cancel and stop now."
    )


def test_normalize_keeps_punctuation_trapped_in_span() -> None:
    # The book traps trailing punctuation inside spans (``*withdraw.*``).
    assert normalize_markdown_for_tts("*Critical,* *amen,* *withdraw.*") == (
        "Critical, amen, withdraw."
    )


def test_normalize_strips_inline_code() -> None:
    assert normalize_markdown_for_tts("Click `export compliance evidence` now.") == (
        "Click export compliance evidence now."
    )


def test_normalize_rewrites_links_and_images() -> None:
    assert normalize_markdown_for_tts("See [the docs](https://x.io/a) here.") == (
        "See the docs here."
    )
    assert normalize_markdown_for_tts("![a diagram](img.png)") == "a diagram"


def test_normalize_drops_horizontal_rules() -> None:
    out = normalize_markdown_for_tts("Scene one.\n\n---\n\nScene two.")
    assert "---" not in out
    assert "Scene one." in out and "Scene two." in out


def test_normalize_speaks_bare_urls() -> None:
    out = normalize_markdown_for_tts("Visit https://commonpractice.nonagentic.ai/ today.")
    assert "http" not in out
    assert "commonpractice dot nonagentic dot ai" in out


def test_normalize_leaves_technical_underscores_alone() -> None:
    # ``_`` emphasis is guarded to word boundaries so snake_case survives.
    assert normalize_markdown_for_tts("call read_claim_form once") == "call read_claim_form once"


def test_normalize_idempotent_on_clean_prose() -> None:
    prose = "Just a flat block of prose with no headings at all."
    assert normalize_markdown_for_tts(prose) == prose


def test_parse_manuscript_strips_markdown_from_body() -> None:
    text = (
        "## Chapter 1 — Everything Is Important\n"
        "Brian said *cancel* and clicked `read-claim-form`.\n\n"
        "---\n\n"
        "See https://commonpractice.nonagentic.ai/ for more.\n"
    )
    m = parse_manuscript(text.encode(), "book.txt")
    assert len(m.chapters) == 1
    assert m.chapters[0].title == "Chapter 1 — Everything Is Important"
    body = m.chapters[0].text
    assert "*" not in body
    assert "`" not in body
    assert "---" not in body
    assert "http" not in body
    assert "cancel" in body and "read-claim-form" in body


def test_docx_manuscript_bomb_is_rejected_before_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(archive_guard, "MAX_ARCHIVE_UNCOMPRESSED_BYTES", 1 << 20)
    monkeypatch.setattr(archive_guard, "RATIO_CHECK_MIN_BYTES", 1 << 40)
    bomb = fixtures.pad_zip_member(fixtures.docx(["hello"]), "word/document.xml", 2 << 20)
    with pytest.raises(ManuscriptParseError, match="uncompressed"):
        parse_manuscript(bomb, "book.docx")


def test_epub_manuscript_bomb_is_rejected_before_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(archive_guard, "RATIO_CHECK_MIN_BYTES", 1 << 20)
    bomb = fixtures.pad_zip_member(fixtures.epub(["chapter"]), "EPUB/ch1.xhtml", 2 << 20)
    with pytest.raises(ManuscriptParseError, match="compression ratio"):
        parse_manuscript(bomb, "book.epub")
