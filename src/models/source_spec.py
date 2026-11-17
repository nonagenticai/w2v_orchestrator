"""SourceSpec — the universal multi-source brief model.

A project of *any* type can ground itself in a combination of sources — a
website, a PDF/book, an uploaded doc, plus media pinned from the media-
discovery. The wizard writes these into ``brief["sources"]`` as a list of
``{kind, ref, label?, auth_info?}`` mappings; the ``source_ingest`` stage reads
them back, scrapes/parses each, and produces a grounding bundle the
downstream LLM + visual stages consume.

The brief itself is a free-form ``dict`` (no orchestrator-side schema), so this
parser is deliberately tolerant: unknown keys are ignored and malformed entries
are skipped rather than failing the run.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

SourceKind = Literal["url", "document", "media"]
_VALID_KINDS: frozenset[str] = frozenset({"url", "document", "media"})


class SourceSpec(BaseModel):
    """One source attached to a project brief.

    - ``url``      — a web page; ``source_ingest`` scrapes it via webscraper.
    - ``document`` — an uploaded file; ``ref`` is the opaque storage ref from
      ``POST /api/uploads/document``; ``source_ingest`` parses it in-process.
    - ``media``    — a media item pinned from media-discovery; ``ref`` is its
      fetchable URL/id, passed through to the visual stages (not re-scraped).
    """

    kind: SourceKind
    ref: str
    label: str = ""
    # Per-source auth (url kind only): the FE's auth_info shape, forwarded to
    # webscraper as ``auth_config``. Opaque here.
    auth_info: dict[str, object] | None = None


def parse_sources(brief: dict[str, Any]) -> list[SourceSpec]:
    """Extract the typed ``sources`` list from a free-form brief, skipping junk."""
    raw = brief.get("sources")
    if not isinstance(raw, list):
        return []
    out: list[SourceSpec] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        kind = item.get("kind")
        ref = item.get("ref")
        if kind not in _VALID_KINDS or not isinstance(ref, str) or not ref.strip():
            continue
        auth = item.get("auth_info")
        out.append(
            SourceSpec(
                kind=kind,  # type: ignore[arg-type]  # guarded by _VALID_KINDS above
                ref=ref.strip(),
                label=str(item.get("label") or ""),
                auth_info=auth if isinstance(auth, dict) and auth else None,
            )
        )
    return out
