"""Deterministic, LLM-free PDF **board** export for a project's storyboard.

Pure functions only — no DB, no LLM, no network, no filesystem: the only I/O is
an in-memory ``io.BytesIO``. The route layer (``api/routes/storyboard.py``)
fetches the panels + scenes via the repositories, resolves each panel's frame
bytes (SSRF-guarded — see the route), and hands both here.

Layout: a landscape-A4 grid of panel cards, :data:`COLUMNS` per row, with a
scene header before each run of panels belonging to the same scene. Each card
carries the frame (the resolved image, or a deliberate "no frame yet"
placeholder), the ``shot_key``, the camera line, the duration, and the
action/dialogue text.

Field set is deliberately the SAME as ``csv_export.CSV_COLUMNS`` so the board and
the shot list never disagree about what a panel contains; ordering is the SAME
``(scene.position, panel.position, panel.created_at)`` used by ``csv_export`` and
``edl_export`` (unscened panels sort LAST under ``"Unassigned"``).

Image bytes are untrusted input: a frame that reportlab cannot decode degrades
to the placeholder rather than raising, because a board missing one thumbnail is
useful and a 500 is not.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING
from uuid import UUID

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Flowable,
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.doctemplate import BaseDocTemplate

from src.database.models import StoryboardPanel, StoryboardScene
from src.services.storyboard.csv_export import _render_list, scene_label
from src.services.storyboard.edl_export import _ordered_panels

if TYPE_CHECKING:  # pragma: no cover - typing only
    from reportlab.pdfgen.canvas import Canvas

log = logging.getLogger(__name__)

# ─── Geometry ────────────────────────────────────────────────────────────────
PAGE_SIZE: tuple[float, float] = landscape(A4)
MARGIN = 12 * mm
COLUMNS = 3
GUTTER = 4 * mm
CARD_WIDTH = (PAGE_SIZE[0] - 2 * MARGIN - (COLUMNS - 1) * GUTTER) / COLUMNS
# 16:9 frame well; the image is letterboxed INSIDE it so a 4:3 or vertical panel
# still lines up with its neighbours on the grid.
FRAME_WIDTH = CARD_WIDTH - 6 * mm
FRAME_HEIGHT = FRAME_WIDTH * 9.0 / 16.0

_INK = colors.HexColor("#18181B")
_MUTED = colors.HexColor("#71717A")
_HAIRLINE = colors.HexColor("#D4D4D8")
_WELL = colors.HexColor("#F4F4F5")

_TITLE_STYLE = ParagraphStyle(
    "BoardTitle", fontName="Helvetica-Bold", fontSize=15, leading=19, textColor=_INK
)
_SCENE_STYLE = ParagraphStyle(
    "BoardScene",
    fontName="Helvetica-Bold",
    fontSize=10.5,
    leading=13,
    textColor=_INK,
    spaceBefore=2,
    spaceAfter=2,
)
_SHOT_STYLE = ParagraphStyle(
    "BoardShot", fontName="Helvetica-Bold", fontSize=8.5, leading=10.5, textColor=_INK
)
_META_STYLE = ParagraphStyle(
    "BoardMeta", fontName="Helvetica", fontSize=6.8, leading=8.6, textColor=_MUTED
)
_BODY_STYLE = ParagraphStyle(
    "BoardBody", fontName="Helvetica", fontSize=7.2, leading=9.2, textColor=_INK
)
_PLACEHOLDER_STYLE = ParagraphStyle(
    "BoardPlaceholder",
    fontName="Helvetica-Oblique",
    fontSize=7.5,
    leading=10,
    textColor=_MUTED,
    alignment=TA_CENTER,
)

PLACEHOLDER_CAPTION = "no frame yet"
#: A panel that HAS a frame we could not fetch (expired signature, dead object,
#: SSRF refusal). Deliberately distinct from ``PLACEHOLDER_CAPTION``: captioning
#: both "no frame yet" tells a director a shot they know was rendered never got
#: made, which is a false statement about their own board — the export silently
#: 200s either way, so the caption is the ONLY signal they get.
FRAME_UNAVAILABLE_CAPTION = "frame unavailable"


def _escape(value: object) -> str:
    """Escape a value for reportlab's mini-HTML paragraph parser.

    ``None``/empty → ``""``. ``&``/``<``/``>`` would otherwise be read as markup
    and either vanish or raise — panel text is author-supplied, so it must never
    be able to break the render.
    """
    if value is None:
        return ""
    text = str(value)
    if not text.strip():
        return ""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _duration_text(panel: StoryboardPanel) -> str:
    """``"3.5s"``, or an em dash when the panel is un-timed."""
    if panel.duration_seconds is None:
        return "—"
    return f"{panel.duration_seconds:g}s"


def _camera_line(panel: StoryboardPanel) -> str:
    """``"MS · eye_level · dolly · 35mm"`` — the set camera fields, dot-joined."""
    parts = [
        _escape(panel.shot_size),
        _escape(panel.angle),
        _escape(panel.movement),
        _escape(panel.lens),
    ]
    return " · ".join(p for p in parts if p)


def _labelled(label: str, value: str) -> str:
    """One ``<b>Label</b> value`` mini-HTML run, or ``""`` when unset."""
    return f"<b>{label}</b> {value}" if value else ""


def panel_order(panels: list[StoryboardPanel], scenes: list[StoryboardScene]) -> list[str]:
    """Shot keys in board order — the export ordering, exposed for assertions.

    The PDF itself is opaque to a test, so the ordering contract (shared with the
    CSV/EDL/FCPXML exports) is checkable here rather than by parsing bytes.
    """
    return [panel.shot_key for panel, _label in _ordered_panels(panels, scenes)]


class _PlaceholderFrame(Flowable):
    """A deliberate empty-frame well: dashed hairline box, tinted fill, caption.

    Panels with no visual are the COMMON case today (image generation is not
    built yet), so this is the default path — it has to read as "not shot yet",
    not as a broken image.
    """

    def __init__(self, width: float, height: float, caption: str = PLACEHOLDER_CAPTION) -> None:
        super().__init__()
        self.width = width
        self.height = height
        self.caption = caption

    def wrap(self, availWidth: float, availHeight: float) -> tuple[float, float]:  # noqa: N803
        return self.width, self.height

    def draw(self) -> None:
        canvas: Canvas = self.canv
        canvas.saveState()
        canvas.setFillColor(_WELL)
        canvas.rect(0, 0, self.width, self.height, stroke=0, fill=1)
        canvas.setStrokeColor(_HAIRLINE)
        canvas.setLineWidth(0.6)
        canvas.setDash(3, 2)
        canvas.rect(0, 0, self.width, self.height, stroke=1, fill=0)
        canvas.setDash()
        canvas.setFillColor(_MUTED)
        canvas.setFont("Helvetica-Oblique", 7.5)
        canvas.drawCentredString(self.width / 2.0, self.height / 2.0 - 3, self.caption)
        canvas.restoreState()


def _frame_flowable(
    data: bytes | None,
    width: float,
    height: float,
    caption: str = PLACEHOLDER_CAPTION,
) -> Flowable:
    """The card's picture: a letterboxed :class:`Image`, else the placeholder.

    ANY decode failure (truncated download, HTML error page served as an image,
    an unsupported codec) degrades to the placeholder — untrusted bytes must not
    be able to fail the export. Undecodable bytes mean we HAD a frame, so that
    path reports ``FRAME_UNAVAILABLE_CAPTION`` rather than "no frame yet".
    """
    if data:
        try:
            intrinsic_w, intrinsic_h = ImageReader(io.BytesIO(data)).getSize()
            if intrinsic_w > 0 and intrinsic_h > 0:
                scale = min(width / intrinsic_w, height / intrinsic_h)
                # A FRESH stream, and ``lazy=0``: reportlab's ``Image`` wants a
                # file-like (an ``ImageReader`` raises), and a lazy Image would
                # re-read the already-consumed buffer at draw time.
                image = Image(
                    io.BytesIO(data),
                    width=intrinsic_w * scale,
                    height=intrinsic_h * scale,
                    lazy=0,
                )
                image.hAlign = "CENTER"
                return image
        except Exception:  # untrusted bytes; degrade, never raise
            log.warning("storyboard pdf: undecodable frame bytes; using placeholder", exc_info=True)
            return _PlaceholderFrame(width, height, FRAME_UNAVAILABLE_CAPTION)
    return _PlaceholderFrame(width, height, caption)


def _card(
    panel: StoryboardPanel, label: str, image: bytes | None, *, unavailable: bool = False
) -> Table:
    """One panel card: frame well + shot header + the ``csv_export`` field set."""
    rows: list[list[Flowable]] = [
        [
            _frame_flowable(
                image,
                FRAME_WIDTH,
                FRAME_HEIGHT,
                FRAME_UNAVAILABLE_CAPTION if unavailable else PLACEHOLDER_CAPTION,
            )
        ],
        [
            Paragraph(
                f"{_escape(panel.shot_key) or '—'} "
                f'<font size="7" color="#71717A">· {_duration_text(panel)}</font>',
                _SHOT_STYLE,
            )
        ],
    ]

    camera = _camera_line(panel)
    if camera:
        rows.append([Paragraph(camera, _META_STYLE)])

    for text in (
        _escape(panel.action_description),
        _labelled("VO:", _escape(panel.vo_text)),
        _labelled("Dialogue:", _escape(panel.dialogue)),
    ):
        if text:
            rows.append([Paragraph(text, _BODY_STYLE)])

    footer_bits = [
        _labelled("SFX:", _escape(panel.sfx_cue)),
        _labelled("Music:", _escape(panel.music_cue)),
        _labelled("Cast:", _escape(_render_list(panel.characters))),
        _labelled("Props:", _escape(_render_list(panel.props))),
        _labelled("Style:", _escape(panel.art_style)),
        _labelled("Aspect:", _escape(panel.aspect_ratio)),
        _labelled("Out:", _escape(panel.transition_out)),
    ]
    footer = "<br/>".join(bit for bit in footer_bits if bit)
    if footer:
        rows.append([Paragraph(footer, _META_STYLE)])

    rows.append(
        [
            Paragraph(
                f"{_escape(label)} · #{panel.position + 1} · {_escape(panel.status)}",
                _META_STYLE,
            )
        ]
    )

    card = Table(rows, colWidths=[CARD_WIDTH])
    card.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.6, _HAIRLINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 0), (0, 0), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 1.4 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.4 * mm),
                ("TOPPADDING", (0, 0), (0, 0), 3 * mm),
                ("LINEABOVE", (0, -1), (-1, -1), 0.4, _HAIRLINE),
            ]
        )
    )
    return card


def _card_grid(cards: list[Table]) -> Table:
    """Lay a scene's cards out :data:`COLUMNS`-across, top-aligned, gutter-spaced."""
    grid_rows: list[list[Flowable | str]] = []
    for start in range(0, len(cards), COLUMNS):
        chunk: list[Flowable | str] = list(cards[start : start + COLUMNS])
        chunk.extend([""] * (COLUMNS - len(chunk)))
        grid_rows.append(chunk)

    grid = Table(grid_rows, colWidths=[CARD_WIDTH + GUTTER] * COLUMNS)
    grid.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), GUTTER),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), GUTTER),
            ]
        )
    )
    return grid


def _scene_header(label: str, count: int) -> Paragraph:
    plural = "panel" if count == 1 else "panels"
    return Paragraph(
        f"{_escape(label) or 'Unassigned'} "
        f'<font size="8" color="#71717A">· {count} {plural}</font>',
        _SCENE_STYLE,
    )


def _page_footer(project_name: str) -> Callable[[Canvas, BaseDocTemplate], None]:
    """``onPage`` callback stamping the project name + page number."""

    def _draw(canvas: Canvas, doc: BaseDocTemplate) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(_MUTED)
        canvas.drawString(MARGIN, MARGIN * 0.55, project_name[:120])
        canvas.drawRightString(PAGE_SIZE[0] - MARGIN, MARGIN * 0.55, f"{canvas.getPageNumber()}")
        canvas.restoreState()

    return _draw


def build_pdf(
    panels: list[StoryboardPanel],
    scenes: list[StoryboardScene],
    *,
    project_name: str = "Storyboard",
    images: dict[UUID, bytes] | None = None,
    unavailable: set[UUID] | None = None,
) -> bytes:
    """Render the storyboard as a printable PDF board.

    ``images`` maps ``panel.id`` → already-resolved frame bytes; the ROUTE
    resolves them (SSRF-guarded, size-capped) so this stays pure. A panel absent
    from the map — or whose bytes do not decode — gets the placeholder well.

    An empty board still produces a valid one-page PDF (title + an explicit
    "no panels" line), matching ``build_csv``'s header-only behaviour.
    """
    resolved: dict[UUID, bytes] = images or {}
    # Panels that HAD a frame url the route could not fetch. Distinct from
    # "absent from `images`", which is the honest "never generated" case.
    missing: set[UUID] = unavailable or set()
    ordered = _ordered_panels(panels, scenes)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=PAGE_SIZE,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
        title=f"Storyboard — {project_name}",
        author="w2v",
        subject="Storyboard board export",
    )

    story: list[Flowable] = [
        Paragraph(_escape(project_name) or "Storyboard", _TITLE_STYLE),
        Paragraph(
            f"Storyboard board · {len(ordered)} panels · {len(scenes)} scenes",
            _META_STYLE,
        ),
        Spacer(1, 4 * mm),
    ]

    if not ordered:
        story.append(Paragraph("This storyboard has no panels yet.", _BODY_STYLE))
    else:
        # Group CONSECUTIVE panels by scene label — the list is already in scene
        # order, so a run boundary is a scene boundary.
        groups: list[tuple[str, list[tuple[StoryboardPanel, str]]]] = []
        for panel, label in ordered:
            if not groups or groups[-1][0] != label:
                groups.append((label, []))
            groups[-1][1].append((panel, label))

        for index, (label, members) in enumerate(groups):
            if index:
                story.append(Spacer(1, 3 * mm))
            cards = [
                _card(panel, lbl, resolved.get(panel.id), unavailable=panel.id in missing)
                for panel, lbl in members
            ]
            header = _scene_header(label, len(members))
            # Glue the header to its FIRST row of cards only, so a scene title never
            # strands at the bottom of a page while a long scene still flows across
            # pages (a KeepTogether over the whole grid would not fit).
            story.append(KeepTogether([header, _card_grid(cards[:COLUMNS])]))
            if len(cards) > COLUMNS:
                story.append(_card_grid(cards[COLUMNS:]))

    footer = _page_footer(project_name or "Storyboard")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()


__all__ = [
    "COLUMNS",
    "FRAME_UNAVAILABLE_CAPTION",
    "PLACEHOLDER_CAPTION",
    "build_pdf",
    "panel_order",
    "scene_label",
]
