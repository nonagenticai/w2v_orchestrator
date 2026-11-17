"""Manuscript parsing + TTS chunking for the audiobook pipeline.

Pure functions (no I/O) so they're trivially unit-testable: bytes + filename
in, structured chapters out; and a chapter's text → TTS-safe chunks. Format is
dispatched on the filename extension:

* ``.txt``  — markdown/plain; chapters split on heading lines.
* ``.docx`` — python-docx; chapters split on ``Heading`` styled paragraphs.
* ``.epub`` — EbookLib; one chapter per spine document.

Chapter detection is best-effort: if nothing chapter-like is found (or
``single`` mode is forced) the whole manuscript becomes one chapter.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

from src.services.documents.archive_guard import UnsafeArchiveError, check_zip_archive
from src.services.stages.audiobook.pronunciation import apply_lexicon

# TTS backend hard-caps a request at 10k chars; stay well under so a chunk plus
# any provider-side markup never trips it, and so each clip is a sane length.
DEFAULT_MAX_CHUNK_CHARS = 2000

# A heading line in plain-text / markdown manuscripts: markdown ATX headings
# (``#``/``##``/``###``) or a "Chapter N" / "CHAPTER N" / "Part N" line.
_HEADING_RE = re.compile(
    r"^\s*(?:#{1,3}\s+(?P<md>.+?)\s*$|(?P<kw>(?:chapter|part|book)\s+[\divxlc]+\b.*?)\s*$)",
    re.IGNORECASE,
)
# Sentence boundary: punctuation followed by whitespace. Good enough for
# greedy chunk packing; we never need linguistically perfect segmentation.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# --- Markdown → spoken-text normalization -----------------------------------
# The TTS backend narrates whatever text we hand it verbatim, so raw markdown
# syntax is voiced literally ("asterisk", "dash dash dash", "h-t-t-p-s colon
# slash slash"). These patterns strip the markup while keeping the words, so a
# markdown manuscript narrates as prose. Applied to the ``.txt``/markdown path
# only — ``.docx``/``.epub`` already arrive as plain text.

# A whole-line horizontal rule / scene break: 3+ of ``-``, ``*`` or ``_``.
_HR_LINE_RE = re.compile(r"^\s*([-*_])\1{2,}\s*$")
# A residual ATX heading marker left in the body (``_HEADING_RE`` only consumes
# ``#{1,3}``; a ``####`` line would otherwise reach TTS as "hash hash …").
_HEADING_MARK_RE = re.compile(r"^\s*#{1,6}\s+")
# A leading blockquote marker.
_BLOCKQUOTE_RE = re.compile(r"^\s*>\s?")
# Images/links → their visible text. Image first (``![alt](url)``) so the ``!``
# is not left stranded by the link pass.
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
# Inline code span → its contents (drop the backticks).
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
# Emphasis. Bold before italic; ``*`` never appears mid-word so it is safe to
# strip unguarded, but ``_`` does (snake_case), so guard it to word boundaries.
_BOLD_STAR_RE = re.compile(r"\*\*([^*]+)\*\*")
_ITALIC_STAR_RE = re.compile(r"\*([^*]+)\*")
_BOLD_US_RE = re.compile(r"(?<![\w`])__([^_]+)__(?![\w`])")
_ITALIC_US_RE = re.compile(r"(?<![\w`])_([^_]+)_(?![\w`])")
# A bare URL left in prose.
_BARE_URL_RE = re.compile(r"https?://[^\s)>\]]+")


def _speak_url(match: re.Match[str]) -> str:
    """Rewrite a bare URL to a spoken-ish form: drop the scheme, say the dots.

    ``https://commonpractice.nonagentic.ai/`` → ``commonpractice dot nonagentic
    dot ai`` — so TTS reads a domain, not "h-t-t-p-s colon slash slash".
    """
    url = match.group(0)
    url = re.sub(r"^https?://", "", url).strip("/")
    return url.replace("/", " slash ").replace(".", " dot ")


def normalize_markdown_for_tts(text: str) -> str:
    """Strip markdown syntax so a manuscript narrates as prose, not markup.

    Pure and side-effect free. Removes horizontal-rule scene breaks, residual
    heading/blockquote markers, and inline emphasis/code/link/image syntax while
    keeping the visible words; rewrites bare URLs to a spoken form. Text with no
    markdown constructs is returned unchanged (idempotent on clean prose).
    """
    lines: list[str] = []
    for line in text.splitlines():
        if _HR_LINE_RE.match(line):
            # Drop the rule entirely; the surrounding paragraph break becomes a
            # natural clip boundary (silence) instead of a voiced "dash dash".
            lines.append("")
            continue
        line = _HEADING_MARK_RE.sub("", line)
        line = _BLOCKQUOTE_RE.sub("", line)
        lines.append(line)
    out = "\n".join(lines)

    out = _IMAGE_RE.sub(r"\1", out)
    out = _LINK_RE.sub(r"\1", out)
    out = _INLINE_CODE_RE.sub(r"\1", out)
    out = _BOLD_STAR_RE.sub(r"\1", out)
    out = _ITALIC_STAR_RE.sub(r"\1", out)
    out = _BOLD_US_RE.sub(r"\1", out)
    out = _ITALIC_US_RE.sub(r"\1", out)
    return _BARE_URL_RE.sub(_speak_url, out)


@dataclass
class ParsedChapter:
    title: str
    text: str
    sequence: int


@dataclass
class ParsedManuscript:
    title: str
    author: str
    chapters: list[ParsedChapter] = field(default_factory=list)


class ManuscriptParseError(RuntimeError):
    """The uploaded manuscript could not be parsed into any narratable text."""


def _ext(filename: str) -> str:
    _, _, ext = filename.lower().rpartition(".")
    return ext


def _guard_zip(data: bytes, filename: str) -> None:
    """Decompression-bomb guard before python-docx/EbookLib inflate anything."""
    try:
        check_zip_archive(data, filename)
    except UnsafeArchiveError as exc:
        raise ManuscriptParseError(str(exc)) from exc


def parse_manuscript(
    data: bytes,
    filename: str,
    *,
    single_chapter: bool = False,
    fallback_title: str = "Audiobook",
    lexicon: dict[str, str] | None = None,
) -> ParsedManuscript:
    """Parse manuscript bytes into a title/author + ordered chapters.

    ``single_chapter`` forces one chapter regardless of detected headings.
    ``lexicon`` is an optional per-book ``{written: spoken}`` pronunciation map
    (see :mod:`.pronunciation`) applied to every chapter's body.
    Raises ``ManuscriptParseError`` when no narratable text survives parsing.
    """
    ext = _ext(filename)
    if ext in {"docx", "epub"}:
        _guard_zip(data, filename)
    if ext == "txt":
        manuscript = _parse_txt(data, fallback_title)
    elif ext == "docx":
        manuscript = _parse_docx(data, fallback_title)
    elif ext == "epub":
        manuscript = _parse_epub(data, fallback_title)
    else:
        raise ManuscriptParseError(f"unsupported manuscript extension: {ext!r}")

    if single_chapter and manuscript.chapters:
        merged = "\n\n".join(c.text for c in manuscript.chapters if c.text.strip())
        manuscript.chapters = [ParsedChapter(title=manuscript.title, text=merged, sequence=0)]

    # Normalize every chapter body once, for all formats: strip markdown markup
    # (safe on plain prose — emphasis/URL patterns are word-boundary guarded),
    # then apply the per-book pronunciation lexicon. Markdown-strip must run
    # first so lexicon regexes see clean words, not ``**Koster**``. Both passes
    # preserve quote characters, so the downstream voice-acting span splitter is
    # undisturbed. The normalized text flows into the content-addressed clip id,
    # so any lexicon change re-synthesizes automatically.
    for chapter in manuscript.chapters:
        text = normalize_markdown_for_tts(chapter.text)
        if lexicon:
            text = apply_lexicon(text, lexicon)
        chapter.text = text

    manuscript.chapters = [c for c in manuscript.chapters if c.text.strip()]
    for i, chapter in enumerate(manuscript.chapters):
        chapter.sequence = i
    if not manuscript.chapters:
        raise ManuscriptParseError("manuscript contained no narratable text")
    return manuscript


def _parse_txt(data: bytes, fallback_title: str) -> ParsedManuscript:
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    chapters: list[ParsedChapter] = []
    current_title = fallback_title
    current_body: list[str] = []

    def _flush() -> None:
        if current_body and "".join(current_body).strip():
            chapters.append(
                ParsedChapter(title=current_title, text="\n".join(current_body).strip(), sequence=0)
            )

    seen_heading = False
    for line in lines:
        match = _HEADING_RE.match(line)
        if match:
            _flush()
            current_body = []
            current_title = (match.group("md") or match.group("kw") or fallback_title).strip()
            seen_heading = True
        else:
            current_body.append(line)
    _flush()

    if not seen_heading:
        # No headings at all → whole file is one chapter.
        body = text.strip()
        chapters = [ParsedChapter(title=fallback_title, text=body, sequence=0)] if body else []
    # Markdown-strip + lexicon are applied centrally in ``parse_manuscript`` for
    # every format, so the body is returned verbatim here.
    return ParsedManuscript(title=fallback_title, author="", chapters=chapters)


def _parse_docx(data: bytes, fallback_title: str) -> ParsedManuscript:
    try:
        import docx  # python-docx
    except ImportError as exc:  # pragma: no cover -- prod always has it
        raise ManuscriptParseError("python-docx is required to parse .docx manuscripts") from exc

    document = docx.Document(io.BytesIO(data))
    props = document.core_properties
    title = (props.title or "").strip() or fallback_title
    author = (props.author or "").strip()

    chapters: list[ParsedChapter] = []
    current_title = title
    current_body: list[str] = []

    def _flush() -> None:
        if current_body and "".join(current_body).strip():
            chapters.append(
                ParsedChapter(title=current_title, text="\n".join(current_body).strip(), sequence=0)
            )

    seen_heading = False
    for para in document.paragraphs:
        style_name = (para.style.name if para.style else "") or ""
        text = para.text.strip()
        if style_name.startswith("Heading") and text:
            _flush()
            current_body = []
            current_title = text
            seen_heading = True
        elif text:
            current_body.append(text)
    _flush()

    if not seen_heading:
        body = "\n".join(p.text.strip() for p in document.paragraphs if p.text.strip())
        chapters = [ParsedChapter(title=title, text=body, sequence=0)] if body.strip() else []
    return ParsedManuscript(title=title, author=author, chapters=chapters)


def _parse_epub(data: bytes, fallback_title: str) -> ParsedManuscript:
    try:
        import ebooklib
        from ebooklib import epub
    except ImportError as exc:  # pragma: no cover -- prod always has it
        raise ManuscriptParseError("EbookLib is required to parse .epub manuscripts") from exc
    from bs4 import BeautifulSoup

    book = epub.read_epub(io.BytesIO(data))

    def _meta(name: str) -> str:
        values = book.get_metadata("DC", name)
        return str(values[0][0]).strip() if values else ""

    title = _meta("title") or fallback_title
    author = _meta("creator")

    chapters: list[ParsedChapter] = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        # The EPUB nav document (table of contents) is also an ITEM_DOCUMENT;
        # it is not narratable prose, so skip it.
        if isinstance(item, epub.EpubNav):
            continue
        soup = BeautifulSoup(item.get_content(), "html.parser")
        heading = soup.find(["h1", "h2", "h3"])
        chapter_title = heading.get_text(strip=True) if heading else f"Chapter {len(chapters) + 1}"
        body = soup.get_text(separator="\n").strip()
        if body:
            chapters.append(ParsedChapter(title=chapter_title or title, text=body, sequence=0))
    return ParsedManuscript(title=title, author=author, chapters=chapters)


def chunk_text(text: str, max_chars: int = DEFAULT_MAX_CHUNK_CHARS) -> list[str]:
    """Greedily pack sentences into chunks no longer than ``max_chars``.

    A single sentence longer than ``max_chars`` is hard-split on whitespace so
    the TTS request never exceeds the backend cap. Whitespace is normalised so
    the spoken output has no stray blank-line gaps.
    """
    normalized = re.sub(r"[ \t]+", " ", text).strip()
    if not normalized:
        return []
    sentences = _split_sentences(normalized, max_chars)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= max_chars:
            current = f"{current} {sentence}"
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


def _split_sentences(text: str, max_chars: int) -> list[str]:
    out: list[str] = []
    for sentence in _SENTENCE_RE.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= max_chars:
            out.append(sentence)
        else:
            out.extend(_hard_split(sentence, max_chars))
    return out


def _hard_split(sentence: str, max_chars: int) -> list[str]:
    words = sentence.split(" ")
    pieces: list[str] = []
    current = ""
    for word in words:
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= max_chars:
            current = f"{current} {word}"
        else:
            pieces.append(current)
            current = word
    if current:
        pieces.append(current)
    return pieces


# --- Speaker-span segmentation (multi-voice / voice-acting) ------------------
# A double-quoted run is spoken dialogue; everything outside is narration. We key
# only on DOUBLE quotes (straight or curly) — single quotes collide with
# apostrophes/contractions in English prose and would mis-split constantly. The
# quote characters themselves are dropped from the dialogue text so TTS never
# voices "quote". Speaker attribution is a separate (LLM) concern in the segment
# stage; this split is purely mechanical and deterministic.
_SPEECH_QUOTE_RE = re.compile(r'["“]([^"”“]+?)["”]')
# Below this many characters a same-speaker fragment is merged into its neighbour
# so TTS never renders a choppy 2-3 word clip. A genuinely short interjection
# between DIFFERENT speakers is preserved (see ``coalesce_units``).
MIN_SPAN_CHARS = 120


@dataclass
class RawSpan:
    """One mechanically-detected span: dialogue (was double-quoted) or narration."""

    text: str
    is_dialogue: bool


def split_spans(text: str) -> list[RawSpan]:
    """Split prose into ordered narration / dialogue spans on double-quote pairs.

    Quote characters are stripped from the dialogue text. Whitespace inside a
    span is normalised (matches ``chunk_text``); empty spans are dropped. A
    passage with no double quotes returns a single narration span, so plain
    narration is unaffected.
    """
    normalized = re.sub(r"[ \t]+", " ", text)
    spans: list[RawSpan] = []
    pos = 0
    for m in _SPEECH_QUOTE_RE.finditer(normalized):
        pre = normalized[pos : m.start()].strip()
        if pre:
            spans.append(RawSpan(text=pre, is_dialogue=False))
        quote = m.group(1).strip()
        if quote:
            spans.append(RawSpan(text=quote, is_dialogue=True))
        pos = m.end()
    tail = normalized[pos:].strip()
    if tail:
        spans.append(RawSpan(text=tail, is_dialogue=False))
    return spans


def split_long_span(text: str, max_chars: int = DEFAULT_MAX_CHUNK_CHARS) -> list[str]:
    """Split one single-speaker span into TTS-safe pieces (reuses the chunker).

    A character's long monologue must not exceed the backend cap; sentence-pack
    it exactly like ``chunk_text`` so each piece stays one voice.
    """
    return chunk_text(text, max_chars=max_chars)
