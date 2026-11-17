"""Summarize stage results for lightweight DB storage.

Full stage results (e.g. crawl HTML) can be many megabytes.  This module
strips bulky fields so that only metadata suitable for the JSONB
``artifact_uris`` column is persisted.  The complete result is stored
separately in object storage (MinIO).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, cast

_MAX_STRING_LEN = 5000
_STRIP_KEYS = frozenset({"html"})


def summarize_stage_result(stage_name: str, result: dict[str, Any]) -> dict[str, Any]:
    """Return a lightweight summary of *result* suitable for JSONB storage.

    The original *result* dict is **not** mutated.
    """
    handler = _STAGE_HANDLERS.get(stage_name, _summarize_generic)
    return handler(result)


# ── Per-stage handlers ───────────────────────────────────────────


def _summarize_crawl(result: dict[str, Any]) -> dict[str, Any]:
    """Strip html/text from each page, keep metadata."""
    summary: dict[str, Any] = {
        "status": result.get("status", "completed"),
        "domain": result.get("domain", ""),
        "raw": result.get("raw", {}),
    }

    pages = result.get("pages", [])
    summary["page_count"] = len(pages)
    summary["pages"] = [
        {
            "url": p.get("url", ""),
            "title": p.get("title", ""),
            "page_id": p.get("page_id", ""),
            "word_count": p.get("word_count", 0),
            "depth": p.get("depth", 0),
            "images": p.get("images", []),
        }
        for p in pages
        if isinstance(p, dict)
    ]

    link_graph = result.get("link_graph", {})
    summary["link_graph_nodes"] = len(link_graph.get("nodes", []))

    return summary


def _summarize_analyze(result: dict[str, Any]) -> dict[str, Any]:
    """Keep job_id, site_analysis, per-page summaries (no HTML)."""
    summary: dict[str, Any] = {
        "status": result.get("status", "completed"),
        "job_id": result.get("job_id", ""),
        "site_analysis": result.get("site_analysis", {}),
        "raw": result.get("raw", {}),
    }

    pages = result.get("pages", [])
    summary["page_count"] = len(pages)
    summary["pages"] = [
        {
            k: v
            for k, v in p.items()
            if k not in _STRIP_KEYS and not (isinstance(v, str) and len(v) > _MAX_STRING_LEN)
        }
        for p in pages
        if isinstance(p, dict)
    ]

    return summary


def _summarize_ingest(result: dict[str, Any]) -> dict[str, Any]:
    """Strip large text from ingested pages/documents."""
    summary: dict[str, Any] = {
        "status": result.get("status", "completed"),
        "raw": result.get("raw", {}),
    }

    pages = result.get("pages", [])
    summary["page_count"] = len(pages)
    summary["pages"] = [
        {
            k: v
            for k, v in p.items()
            if k not in _STRIP_KEYS and not (isinstance(v, str) and len(v) > _MAX_STRING_LEN)
        }
        for p in pages
        if isinstance(p, dict)
    ]

    documents = result.get("documents", [])
    summary["document_count"] = len(documents)
    summary["documents"] = [
        {
            k: v
            for k, v in d.items()
            if k not in _STRIP_KEYS and not (isinstance(v, str) and len(v) > _MAX_STRING_LEN)
        }
        for d in documents
        if isinstance(d, dict)
    ]

    return summary


def _summarize_generic(result: dict[str, Any]) -> dict[str, Any]:
    """Generic fallback: recursively strip ``html`` keys and long strings."""
    serialized = json.dumps(result, default=str)
    if len(serialized) < 50_000:
        # Small enough — pass through as-is.
        return dict(result)

    return cast("dict[str, Any]", _strip_deep(result))


def _strip_deep(obj: object) -> object:
    """Recursively strip bulky values from a nested structure."""
    if isinstance(obj, dict):
        return {k: _strip_deep(v) for k, v in obj.items() if k not in _STRIP_KEYS}
    if isinstance(obj, list):
        return [_strip_deep(item) for item in obj]
    if isinstance(obj, str) and len(obj) > _MAX_STRING_LEN:
        return obj[:200] + f"... [{len(obj)} chars truncated]"
    return obj


# Stages that need special handling; everything else uses _summarize_generic.
_STAGE_HANDLERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "crawl": _summarize_crawl,
    "analyze": _summarize_analyze,
    "ingest": _summarize_ingest,
}
