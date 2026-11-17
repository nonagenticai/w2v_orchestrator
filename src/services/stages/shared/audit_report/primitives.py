"""Shared building blocks for the per-type audit report templates.

Each audit type (``seo.py`` / ``cro.py`` / ``marketing.py``) composes these into
its own document. Nothing here guesses the audit type: the caller chose it from
the run's ``project_type``.

Two rules every primitive follows:

* **No silent passes.** A section whose data is absent renders
  :func:`not_measured` ("Not measured — <reason>"), never an empty section and
  never a 100 bar. A scan written before a field existed therefore still renders.
* **AI status is read, never assumed.** :func:`ai_state` reads the scan's ``ai``
  block. A scan with no ``ai`` block is reported as "not reported", never as
  "complete" or "not requested", because an absent status is not an observation.

All untrusted text goes through :func:`html.escape`.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from html import escape
from typing import cast
from urllib.parse import urlparse

type Scan = Mapping[str, object]
type Row = dict[str, object]

# Per-type accent token (§6: a distinct accent per audit type). Status colours
# (``tone``) are semantic, not brand, and stay fixed.
TYPE_ACCENTS: dict[str, str] = {
    "seo_audit": "#0f766e",  # teal
    "cro_audit": "#c2410c",  # burnt orange
    "marketing_scorecard": "#6d28d9",  # violet
}

_GOOD = "#16a34a"
_WARN = "#d97706"
_BAD = "#dc2626"

# --- untyped-JSON narrowing -------------------------------------------------


def as_dict(value: object) -> Row:
    """Narrow an untyped JSON value to a dict (``{}`` if it is not one)."""
    return cast("Row", value) if isinstance(value, dict) else {}


def dict_rows(value: object) -> list[Row]:
    """Just the dict rows of an untyped JSON list (``[]`` if not a list)."""
    if not isinstance(value, list):
        return []
    return [cast("Row", r) for r in value if isinstance(r, dict)]


def str_list(value: object) -> list[str]:
    """The non-empty string items of an untyped JSON list."""
    if not isinstance(value, list):
        return []
    return [str(v) for v in value if v is not None and str(v) != ""]


def opt_num(value: object) -> float | None:
    """A finite number, or ``None`` when absent / null / non-numeric."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(cast("str | float | int", value))
    except TypeError, ValueError:
        return None
    return out if math.isfinite(out) else None


def num(value: object) -> float:
    """A number, ``0.0`` when absent. Only for values that are always present."""
    out = opt_num(value)
    return 0.0 if out is None else out


def txt(value: object) -> str:
    """Escaped display text for an untyped scalar (``""`` for ``None``)."""
    return "" if value is None else escape(str(value))


def measured_fact(facts: Row | None, key: str, *, missing: str) -> tuple[Row | None, str]:
    """A fact block and, when it is not usable, why.

    analysis_service marks a fact it could not produce as
    ``{"measured": false, "reason": "..."}``. That, an absent key, and an absent
    ``facts`` block all return ``(None, reason)`` — never an empty dict that a
    section might read as "measured, nothing found".
    """
    if facts is None:
        return None, missing
    raw = facts.get(key)
    if not isinstance(raw, dict):
        return None, missing
    block = cast("Row", raw)
    if block.get("measured") is False:
        return None, str(block.get("reason") or missing)
    return block, ""


def check_status(checks: list[Row] | None, check_id: str) -> str | None:
    """The status of one catalogued check (``pass``/``fail``/``not_run``), or ``None``."""
    for c in checks or []:
        if str(c.get("check_id") or c.get("id") or "") == check_id:
            return str(c.get("status") or "")
    return None


def host_of(url: str) -> str:
    """The URL's host, filename-safe (``"report"`` when unparseable).

    Restricted to ``[a-z0-9.-]`` because it lands in a Content-Disposition header.
    """
    try:
        host = urlparse(url).hostname if url else None
    except ValueError:
        host = None
    safe = re.sub(r"[^a-z0-9.-]", "", (host or "").lower())
    return safe or "report"


# --- colour -----------------------------------------------------------------


def tone(score: float) -> str:
    """80/50 banding on the rounded score (same as the frontend scorecards)."""
    s = round(score)
    if s >= 80:
        return _GOOD
    if s >= 50:
        return _WARN
    return _BAD


# --- document skeleton ------------------------------------------------------

_CSS = """
  @page { size: A4; margin: 16mm; }
  * { box-sizing: border-box; }
  body { font-family: var(--font); color: var(--on-surface); background: var(--surface);
         margin: 0; padding: 0; font-size: var(--t-body); line-height: 1.5; }
  header.band { display: flex; align-items: center; gap: 24px;
                border-bottom: 4px solid var(--type-accent); padding-bottom: 16px; margin-bottom: 12px; }
  .ring { width: 96px; height: 96px; border-radius: 50%; flex: 0 0 auto; display: flex;
          align-items: center; justify-content: center; flex-direction: column; color: #fff; }
  .ring .score { font-size: var(--t-h1); font-weight: 700; line-height: 1; }
  .ring .of { font-size: var(--t-caption); opacity: .85; }
  .band-main { flex: 1; }
  .band-main h1 { font-size: var(--t-h1); margin: 0 0 2px; color: var(--type-accent); }
  .band-main .question { margin: 0 0 4px; color: var(--on-surface-muted); font-style: italic; }
  .band-main .url { color: var(--on-surface-muted); font-size: var(--t-caption); word-break: break-all; }
  .grade { display: inline-block; margin-top: 6px; padding: 2px 12px; border-radius: 999px;
           color: #fff; font-weight: 600; font-size: var(--t-kicker); }
  .ai-status { margin: 0 0 16px; padding: 6px 12px; border-radius: 6px; font-size: var(--t-kicker);
               background: var(--surface-2); border-left: 4px solid var(--on-surface-muted); }
  .ai-status.complete { border-left-color: #16a34a; }
  .ai-status.partial { border-left-color: #d97706; }
  .ai-status.unavailable, .ai-status.unreported { border-left-color: #dc2626; }
  h2 { font-size: var(--t-h2); margin: 20px 0 10px; color: var(--on-surface);
       border-bottom: 1px solid var(--surface-2); padding-bottom: 4px; }
  h3 { font-size: var(--t-body); margin: 14px 0 6px; color: var(--on-surface-muted);
       text-transform: uppercase; letter-spacing: .04em; }
  section { margin-bottom: 18px; }
  section.major { break-before: page; }
  .nm { padding: 8px 12px; border-radius: 6px; background: #e5e7eb; color: #374151;
        font-size: var(--t-kicker); margin: 6px 0; }
  .ai-na { padding: 8px 12px; border-radius: 6px; background: #fef3c7; color: #78350f;
           font-size: var(--t-kicker); margin: 6px 0; }
  .bad-fact { color: #b91c1c; font-weight: 600; }
  .row { display: grid; grid-template-columns: 180px 1fr 48px; align-items: center;
         gap: var(--gutter); margin-bottom: 6px; }
  .cat { font-size: var(--t-kicker); color: var(--on-surface-muted); text-transform: capitalize; }
  .track { height: 8px; border-radius: 999px; background: var(--surface-2); overflow: hidden; }
  .fill { display: block; height: 100%; border-radius: 999px; }
  .val { font-size: var(--t-caption); color: var(--on-surface-muted); text-align: right;
         font-variant-numeric: tabular-nums; }
  table { width: 100%; border-collapse: collapse; font-size: var(--t-kicker); margin: 6px 0; }
  th { text-align: left; color: var(--on-surface-muted); font-weight: 600;
       border-bottom: 2px solid var(--surface-2); padding: 4px 6px; }
  td { border-bottom: 1px solid var(--surface-2); padding: 4px 6px; vertical-align: top; }
  tr { break-inside: avoid; }
  .st-pass { color: #15803d; font-weight: 700; }
  .st-fail { color: #b91c1c; font-weight: 700; }
  .st-not_run { color: #6b7280; font-weight: 700; }
  .rank { background: var(--type-accent); color: #fff; font-size: var(--t-caption);
          font-weight: 700; border-radius: 999px; padding: 1px 8px; white-space: nowrap; }
  .sev { color: var(--on-surface-muted); text-transform: uppercase; font-size: var(--t-caption); }
  .ground { font-size: var(--t-caption); border-radius: 999px; padding: 1px 8px; font-weight: 600; }
  .ground.ok { background: #16a34a; color: #fff; }
  .ground.warn { background: #d97706; color: #fff; }
  .rec { margin-top: 4px; padding-top: 4px; border-top: 1px dashed var(--surface-2); }
  .rec .fix { margin: 0 0 4px; }
  .rec .why { margin: 0 0 4px; color: var(--on-surface-muted); }
  .rec .eg { margin: 0; padding: 6px; background: var(--surface-2); border-radius: 6px;
             font-size: var(--t-caption); white-space: pre-wrap; }
  .lead { font-size: var(--t-h2); line-height: 1.4; margin: 0 0 6px; }
  .muted { color: var(--on-surface-muted); }
  .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
  .tile { border: 1px solid var(--surface-2); border-radius: 8px; padding: 8px 10px;
          font-size: var(--t-kicker); break-inside: avoid; }
  .tile.on { border-left: 4px solid #16a34a; }
  .tile.off { border-left: 4px solid #dc2626; }
  .tile.na { border-left: 4px solid #9ca3af; }
  .two { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .card { border: 1px solid var(--surface-2); border-radius: 8px; padding: 10px 12px;
          margin-bottom: 8px; break-inside: avoid; }
  .card h3 { margin-top: 0; }
  blockquote { margin: 6px 0; padding: 6px 12px; border-left: 4px solid var(--type-accent);
               background: var(--surface-2); }
  .strip { display: flex; flex-wrap: wrap; gap: 6px; margin: 6px 0 10px; }
  .chip { border-radius: 999px; padding: 2px 10px; font-size: var(--t-caption); font-weight: 600;
          background: var(--surface-2); }
  .chip.on { background: #dcfce7; color: #166534; }
  .chip.off { background: #fee2e2; color: #991b1b; }
  .chip.na { background: #e5e7eb; color: #374151; }
  .verdict { display: inline-block; margin-top: 6px; padding: 4px 14px; border-radius: 6px;
             font-weight: 800; letter-spacing: .06em; color: #fff; }
  .callout { font-size: var(--t-h2); font-weight: 700; margin: 4px 0 12px; }
  .dial { display: inline-block; min-width: 44px; text-align: center; border-radius: 999px;
          padding: 2px 8px; color: #fff; font-weight: 700; }
  ul.plain { margin: 4px 0; padding-left: 18px; }
"""


def document(*, title: str, tok: Mapping[str, str], accent: str, body: str) -> str:
    """Wrap the rendered body into a self-contained HTML document."""
    return (
        "<!doctype html>\n"
        f'<html><head><meta charset="utf-8"><title>{escape(title)}</title><style>\n'
        f"{root_vars(tok, accent)}\n{_CSS}</style></head><body>\n{body}\n</body></html>"
    )


def root_vars(tok: Mapping[str, str], accent: str) -> str:
    """The ``:root`` custom-property block from a flat design-token map + type accent."""
    return (
        ":root{"
        f"--surface:{tok['surface']};"
        f"--surface-2:{tok['surface_2']};"
        f"--on-surface:{tok['on_surface']};"
        f"--on-surface-muted:{tok['on_surface_muted']};"
        f"--accent:{tok['accent']};"
        f"--accent-2:{tok['accent_2']};"
        f"--accent-soft:{tok['accent_soft']};"
        f"--type-accent:{accent};"
        f"--font:{tok['font_family']}, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;"
        f"--t-display:{tok['type_display']}px;"
        f"--t-h1:{tok['type_h1']}px;"
        f"--t-h2:{tok['type_h2']}px;"
        f"--t-body:{tok['type_body']}px;"
        f"--t-kicker:{tok['type_kicker']}px;"
        f"--t-caption:{tok['type_caption']}px;"
        f"--gutter:{tok['grid_gutter']}px;"
        f"--margin:{tok['grid_margin']}px;"
        "}"
    )


def header_band(*, title: str, question: str, scan: Scan, extra: str = "") -> str:
    """Score ring + per-type title + the reader question + URL + grade."""
    score = num(scan.get("overall_score"))
    ring_tone = tone(score)
    grade = txt(scan.get("grade"))
    grade_html = (
        f'<span class="grade" style="background:{ring_tone}">Grade {grade}</span>' if grade else ""
    )
    return (
        '<header class="band">'
        f'<div class="ring" style="background:{ring_tone}"><span class="score">{round(score)}</span>'
        '<span class="of">/ 100</span></div>'
        '<div class="band-main">'
        f"<h1>{escape(title)}</h1>"
        f'<p class="question">{escape(question)}</p>'
        f'<div class="url">{txt(scan_url(scan))}</div>'
        f"{grade_html}{extra}"
        "</div></header>"
    )


def scan_url(scan: Scan) -> str:
    """The audited URL: the ORCH scan's ``url``, else the AS response's ``source``."""
    return str(scan.get("url") or scan.get("source") or "")


def section(key: str, title: str, body: str, *, major: bool = False) -> str:
    """A titled report section carrying a stable ``data-section`` hook."""
    cls = ' class="major"' if major else ""
    return f'<section data-section="{escape(key)}"{cls}><h2>{escape(title)}</h2>{body}</section>'


def not_measured(reason: str) -> str:
    """The grey "Not measured — <reason>" box (§2.0)."""
    return f'<div class="nm">Not measured — {escape(reason)}</div>'


def ai_unavailable(reason: str) -> str:
    """Per-card "AI analysis unavailable: <reason>" box (§3.3)."""
    return f'<div class="ai-na">AI analysis unavailable: {escape(reason)}</div>'


def bar(label: str, score: float) -> str:
    """One labelled score bar (only for a MEASURED score — see :func:`score_row`)."""
    pct = max(0, min(100, round(score)))
    return (
        '<div class="row">'
        f'<span class="cat">{escape(label)}</span>'
        f'<span class="track"><span class="fill" style="width:{pct}%;background:{tone(score)}"></span></span>'
        f'<span class="val">{pct}</span></div>'
    )


def score_row(label: str, score: float | None, *, note: str = "") -> str:
    """A bar for a measured score, or a not-measured box (never a bar) when ``None``."""
    if score is None:
        why = note or "this scan did not score it"
        return f'<div class="row"><span class="cat">{escape(label)}</span>{not_measured(why)}<span></span></div>'
    return bar(label, score)


def flag_chip(label: str, value: object, *, na_reason: str = "not measured") -> str:
    """A ✓ / ✗ / not-measured chip for a boolean-ish fact."""
    if value is None:
        return f'<span class="chip na" title="{escape(na_reason)}">— {escape(label)}</span>'
    if value:
        return f'<span class="chip on">✓ {escape(label)}</span>'
    return f'<span class="chip off">✗ {escape(label)}</span>'


def tile(label: str, value: object, *, detail: str = "") -> str:
    """An inventory tile: present ✓ / missing ✗ / not measured."""
    if value is None:
        cls, mark = "na", "— not measured"
    elif value:
        cls, mark = "on", "✓ present"
    else:
        cls, mark = "off", "✗ missing"
    extra = f'<div class="muted">{detail}</div>' if detail else ""
    return f'<div class="tile {cls}"><strong>{escape(label)}</strong><div>{mark}</div>{extra}</div>'


def dial(score: object) -> str:
    """A small score pill (0-100, or 0-10 scaled up) for AI judgments."""
    v = opt_num(score)
    if v is None:
        return '<span class="dial" style="background:#9ca3af">—</span>'
    scaled = v * 10 if v <= 10 else v
    return f'<span class="dial" style="background:{tone(scaled)}">{round(scaled)}</span>'


# --- AI status (§3.2) -------------------------------------------------------

_UNAVAILABLE_LAYER_STATES = frozenset({"failed", "empty"})


@dataclass(frozen=True)
class AiState:
    """What the scan says about its AI layers.

    ``status`` is one of ``not_requested`` / ``complete`` / ``partial`` /
    ``unavailable`` (the §3.2 contract), or ``unreported`` for a scan that carries
    no ``ai`` block at all (written before the contract existed).
    """

    status: str
    layers: dict[str, tuple[str, str]] = field(default_factory=dict)
    reason: str = ""

    @property
    def requested(self) -> bool:
        return self.status != "not_requested"

    def layer_problem(self, *layers: str) -> str | None:
        """The reason the first named layer that did not produce output, or ``None``."""
        for name in layers:
            st, why = self.layers.get(name, ("", ""))
            if st == "failed":
                return why or "layer failed"
            if st == "skipped":
                return f"skipped — {why}" if why else "skipped"
            if st == "empty":
                return why or "the model returned nothing usable"
        if self.status == "unavailable":
            return self.reason or "no AI layer produced a result"
        return None


def ai_state(scan: Scan) -> AiState:
    """Read the §3.2 ``ai`` block; a missing block is ``unreported``, never complete."""
    raw = scan.get("ai")
    if not isinstance(raw, dict):
        return AiState(status="unreported")
    ai = cast("Row", raw)
    layers: dict[str, tuple[str, str]] = {}
    for layer in dict_rows(ai.get("layers")):
        name = str(layer.get("layer") or "")
        if name:
            layers[name] = (str(layer.get("status") or ""), str(layer.get("reason") or ""))
    status = str(ai.get("status") or "")
    if ai.get("requested") is False:
        status = "not_requested"
    if status not in {"not_requested", "complete", "partial", "unavailable"}:
        status = "unreported"
    return AiState(status=status, layers=layers, reason=str(ai.get("reason") or ""))


def _failed_summary(state: AiState) -> str:
    """Failed layers grouped by reason: ``"WM 401 Unauthorized (recommendations, critic)"``."""
    by_reason: dict[str, list[str]] = {}
    for name, (st, why) in state.layers.items():
        if st in _UNAVAILABLE_LAYER_STATES:
            by_reason.setdefault(why or st, []).append(name)
    return "; ".join(f"{why} ({', '.join(names)})" for why, names in by_reason.items())


def ai_status_line(state: AiState) -> str:
    """The always-present page-1 AI status line (§2.0)."""
    if state.status == "not_requested":
        text = "AI analysis: not requested"
    elif state.status == "complete":
        text = "AI analysis: complete"
    elif state.status == "partial":
        bad = sum(1 for st, _ in state.layers.values() if st in _UNAVAILABLE_LAYER_STATES)
        text = f"AI analysis: partial — {bad} of {len(state.layers)} layers unavailable"
        detail = _failed_summary(state)
        if detail:
            text += f" ({detail})"
    elif state.status == "unavailable":
        text = f"AI analysis unavailable: {_failed_summary(state) or state.reason or 'no reason reported'}"
    else:
        text = "AI analysis: not reported — this scan predates AI status reporting"
    return f'<p class="ai-status {state.status}" data-ai-status="{state.status}">{escape(text)}</p>'


# --- findings / checks ------------------------------------------------------


def findings_of(scan: Scan) -> list[Row]:
    """The full ``findings[]`` (§4 X-G2), else the legacy ``quick_wins``+``strategic`` refs.

    De-duplicated by ``check_id`` and ordered by ``rank`` (unranked last, stable).
    """
    rows = dict_rows(scan.get("findings"))
    if not rows:
        rows = dict_rows(scan.get("quick_wins")) + dict_rows(scan.get("strategic"))
    seen: set[str] = set()
    out: list[Row] = []
    for r in rows:
        cid = str(r.get("check_id") or "")
        if cid and cid in seen:
            continue
        if cid:
            seen.add(cid)
        out.append(r)

    def _key(r: Row) -> tuple[int, float]:
        rank = r.get("rank")
        return (0, float(rank)) if isinstance(rank, int) else (1, 0.0)

    return sorted(out, key=_key)


def checks_of(scan: Scan) -> list[Row] | None:
    """The §4 X-G1 check inventory, or ``None`` when the scan carries none."""
    raw = scan.get("checks")
    if not isinstance(raw, list):
        return None
    return dict_rows(raw)


def in_bucket(row: Row, key: str, values: Iterable[str]) -> bool:
    return str(row.get(key) or "") in set(values)


def grounding_map(scan: Scan) -> dict[str, Row]:
    return {
        str(g.get("check_id") or ""): g
        for g in dict_rows(scan.get("grounding"))
        if g.get("check_id")
    }


def grounding_badge(check_id: str, grounding: Mapping[str, Row], *, has_rec: bool) -> str:
    """``unverified`` always shows; ``grounded`` only when there is prose to vouch for."""
    g = grounding.get(check_id)
    if g is None:
        return ""
    if g.get("supported"):
        return '<span class="ground ok">grounded</span>' if has_rec else ""
    return '<span class="ground warn">unverified</span>'


def recommendation(rec: object) -> str:
    """A finding's AI fix prose block; ``""`` when none was written."""
    r = as_dict(rec)
    fix = txt(r.get("fix_prose"))
    if not fix:
        return ""
    parts = [f'<p class="fix">{fix}</p>']
    why = txt(r.get("rationale"))
    if why:
        parts.append(f'<p class="why">{why}</p>')
    eg = txt(r.get("example"))
    if eg:
        parts.append(f'<pre class="eg">{eg}</pre>')
    return f'<div class="rec">{"".join(parts)}</div>'


def rank_badge(row: Row) -> str:
    rank = row.get("rank")
    return f'<span class="rank">#{rank}</span>' if isinstance(rank, int) else ""


def deterministic_fix(row: Row) -> str:
    """The scanner's own fix text (§4 X-G2 ``fix``), not AI prose."""
    return txt(row.get("fix") or row.get("description"))


def findings_table(rows: Sequence[Row], grounding: Mapping[str, Row]) -> str:
    """Rank · issue · severity · effort · deterministic fix, AI prose beneath the row."""
    if not rows:
        return ""
    body: list[str] = []
    for r in rows:
        title = txt(r.get("title") or r.get("check_id"))
        if not title:
            continue
        rec_html = recommendation(r.get("recommendation"))
        badge = grounding_badge(str(r.get("check_id") or ""), grounding, has_rec=bool(rec_html))
        fix = deterministic_fix(r) or '<span class="muted">—</span>'
        body.append(
            "<tr>"
            f"<td>{rank_badge(r)}</td>"
            f"<td><strong>{title}</strong> {badge}{rec_html}</td>"
            f'<td><span class="sev">{txt(r.get("severity"))}</span></td>'
            f"<td>{txt(r.get('effort')) or '—'}</td>"
            f"<td>{fix}</td>"
            "</tr>"
        )
    if not body:
        return ""
    return (
        '<table class="findings"><thead><tr><th>Rank</th><th>Issue</th><th>Severity</th>'
        f"<th>Effort</th><th>Fix</th></tr></thead><tbody>{''.join(body)}</tbody></table>"
    )


_STATUS_MARK = {"pass": "✓ pass", "fail": "✗ fail", "not_run": "— not run"}


def checklist(
    checks: list[Row] | None,
    *,
    key: str,
    buckets: Iterable[str],
    findings: Sequence[Row],
    not_reported: str = "the check inventory is not in this scan",
) -> str:
    """Check · status · measured · target · fix for one bucket of the inventory.

    ``checks is None`` (an older scan) falls back to the failing findings in the
    bucket and says that passing checks are not reported, so a reader never takes
    a short list for a clean bill.
    """
    wanted = set(buckets)
    by_id = {str(f.get("check_id") or ""): f for f in findings}
    if checks is None:
        failing = [f for f in findings if str(f.get(key) or "") in wanted]
        rows = "".join(
            "<tr>"
            f"<td>{txt(f.get('title') or f.get('check_id'))}</td>"
            '<td class="st-fail">✗ fail</td><td>—</td><td>—</td>'
            f"<td>{deterministic_fix(f) or '—'}</td></tr>"
            for f in failing
        )
        table = (
            '<table class="checklist"><thead><tr><th>Check</th><th>Status</th><th>Measured</th>'
            f"<th>Target</th><th>Fix</th></tr></thead><tbody>{rows}</tbody></table>"
            if rows
            else ""
        )
        return table + not_measured(f"passing checks — {not_reported}")
    mine = [c for c in checks if str(c.get(key) or "") in wanted]
    if not mine:
        return not_measured("no checks of this kind ran on this page")
    rows_html: list[str] = []
    for c in mine:
        cid = str(c.get("id") or c.get("check_id") or "")
        status = str(c.get("status") or "not_run")
        if status not in _STATUS_MARK:
            status = "not_run"
        f = by_id.get(cid, {})
        title = txt(c.get("title") or f.get("title") or cid)
        fix = deterministic_fix(f) if status == "fail" else ""
        rows_html.append(
            "<tr>"
            f"<td>{title}</td>"
            f'<td class="st-{status}">{_STATUS_MARK[status]}</td>'
            f"<td>{txt(c.get('measured')) or '—'}</td>"
            f"<td>{txt(c.get('target')) or '—'}</td>"
            f"<td>{fix or '—'}</td></tr>"
        )
    return (
        '<table class="checklist"><thead><tr><th>Check</th><th>Status</th><th>Measured</th>'
        f"<th>Target</th><th>Fix</th></tr></thead><tbody>{''.join(rows_html)}</tbody></table>"
    )


def check_counts(checks: list[Row] | None, *, key: str, bucket: str) -> tuple[int, int, int] | None:
    """(run, passed, failed) for one bucket, or ``None`` with no inventory."""
    if checks is None:
        return None
    mine = [c for c in checks if str(c.get(key) or "") == bucket]
    ran = [c for c in mine if c.get("status") in {"pass", "fail"}]
    passed = sum(1 for c in ran if c.get("status") == "pass")
    return len(ran), passed, len(ran) - passed


def unbucketed_findings(rows: Sequence[Row], *, key: str, grounding: Mapping[str, Row]) -> str:
    """Findings that name no bucket — rendered, never silently dropped."""
    orphans = [f for f in rows if not f.get(key)]
    if not orphans:
        return ""
    return f"<h3>Other findings</h3>{findings_table(orphans, grounding)}"


def all_passed(n: int) -> str:
    """ "All <n> checks passed." with the right plural."""
    return f"<p>All {n} check{'' if n == 1 else 's'} passed.</p>"


def weight_text(value: object) -> str:
    w = opt_num(value)
    return "—" if w is None else f"{w:.2f}"


def bucket_fix_list(
    *,
    bucket: str,
    key: str,
    findings: Sequence[Row],
    checks: list[Row] | None,
    grounding: Mapping[str, Row],
) -> str:
    """One bucket's failing findings, or "All <n> checks passed" / not-measured."""
    rows = [f for f in findings if str(f.get(key) or "") == bucket]
    if rows:
        return findings_table(rows, grounding)
    counts = check_counts(checks, key=key, bucket=bucket)
    if counts is None:
        return not_measured("no failing checks reported, and this scan has no check inventory")
    ran, passed, _ = counts
    if ran == 0:
        return not_measured("no checks of this kind ran on this page")
    return all_passed(passed)


# --- enrichment sections ----------------------------------------------------


def exec_summary_block(scan: Scan, findings: Sequence[Row]) -> str:
    """Executive summary + id-locked roadmap; ``""`` when the scan carries none."""
    es = as_dict(scan.get("exec_summary"))
    summary = txt(es.get("summary"))
    if not summary:
        return ""
    title_by_id = {
        str(f.get("check_id")): str(f.get("title"))
        for f in findings
        if f.get("check_id") and f.get("title")
    }
    so_what = txt(es.get("so_what"))
    parts = [f'<p class="lead">{summary}</p>']
    if so_what:
        parts.append(f'<p class="muted">{so_what}</p>')
    seen: set[str] = set()
    lis: list[str] = []
    for cid in str_list(es.get("roadmap")):
        # Defense-in-depth id-lock: render only ids that resolve to a real finding.
        if cid in seen or cid not in title_by_id:
            continue
        seen.add(cid)
        lis.append(f"<li>{escape(title_by_id[cid])}</li>")
    if lis:
        parts.append(f'<h3>Do these first</h3><ol class="roadmap">{"".join(lis)}</ol>')
    return "".join(parts)


def projection_block(scan: Scan) -> str:
    """Before → after from fixing the modelled findings; ``""`` when absent."""
    p = as_dict(scan.get("projection"))
    if not p:
        return ""
    base, proj, delta = num(p.get("baseline")), num(p.get("projected")), num(p.get("delta"))
    removed = str_list(p.get("removed"))
    return (
        f"<p>Fixing the top {len(removed)} finding(s): "
        f"<strong>{round(base)} &rarr; {round(proj)}</strong> "
        f'<span class="rank">+{round(delta)} pts</span></p>'
        f"{bar('Today', base)}{bar('After fixes', proj)}"
    )


def critique_card(scan: Scan, state: AiState) -> str:
    """The "Report quality" card from the report critic."""
    c = as_dict(scan.get("critique"))
    if not c:
        problem = state.layer_problem("critic")
        body = ai_unavailable(problem or "the report critic returned no verdict")
    else:
        generic = str_list(c.get("generic_check_ids"))
        ungrounded = str_list(c.get("ungrounded_check_ids"))
        body = (
            f"<p>Critic score {dial(c.get('score'))}</p>"
            f"<p class='muted'>Generic advice flagged: {len(generic)} · "
            f"ungrounded claims flagged: {len(ungrounded)}</p>"
        )
    return f'<div class="card" data-card="report-quality"><h3>Report quality</h3>{body}</div>'


def has_ai_content(scan: Scan) -> bool:
    """Whether a legacy (``unreported``) scan carries any AI-written content."""
    if scan.get("exec_summary") or scan.get("critique"):
        return True
    if dict_rows(scan.get("llm_judgments")) or dict_rows(scan.get("llm_assessments")):
        return True
    if dict_rows(as_dict(scan.get("ai")).get("annotations")):
        return True
    return any(as_dict(f.get("recommendation")).get("fix_prose") for f in findings_of(scan))


def show_ai_section(scan: Scan, state: AiState) -> bool:
    """AI section rendered when AI was requested (incl. failed); omitted when off."""
    if state.status == "not_requested":
        return False
    if state.status == "unreported":
        return has_ai_content(scan)
    return True


def methodology(
    scan: Scan,
    *,
    rows: Sequence[Row],
    key: str,
    extra_cols: Mapping[str, Mapping[str, str]] | None = None,
    note: str = "",
) -> str:
    """Weights table, checks-run count, ``scored_at``, source URL, coverage (§2.0)."""
    extra_cols = extra_cols or {}
    col_names = list(extra_cols)
    head = "".join(f"<th>{escape(c)}</th>" for c in col_names)
    body: list[str] = []
    for r in rows:
        name = str(r.get(key) or "")
        if not name:
            continue
        w = opt_num(r.get("weight"))
        s = opt_num(r.get("score"))
        cells = "".join(f"<td>{escape(extra_cols[c].get(name, '—'))}</td>" for c in col_names)
        body.append(
            f"<tr><td>{escape(name)}</td><td>{'—' if w is None else f'{w:.2f}'}</td>"
            f"<td>{'not measured' if s is None else round(s)}</td>{cells}</tr>"
        )
    table = (
        f'<table class="weights"><thead><tr><th>Component</th><th>Weight</th><th>Score</th>{head}'
        f"</tr></thead><tbody>{''.join(body)}</tbody></table>"
        if body
        else not_measured("no component weights in this scan")
    )
    checks = checks_of(scan)
    ran = (
        "not reported"
        if checks is None
        else str(len([c for c in checks if c.get("status") in {"pass", "fail"}]))
    )
    coverage = opt_num(scan.get("coverage"))
    cov = "not reported" if coverage is None else f"{round(coverage * 100)}%"
    scored_at = txt(scan.get("scored_at")) or "not recorded"
    facts = (
        '<ul class="plain">'
        f"<li>Source URL: {txt(scan_url(scan)) or 'not recorded'}</li>"
        f"<li>Checks run: {ran}</li>"
        f"<li>Rubric coverage: {cov}</li>"
        f"<li data-scored-at>Scored at: {scored_at}</li>"
        "</ul>"
    )
    extra = f'<p class="muted">{escape(note)}</p>' if note else ""
    return section("methodology", "Methodology", table + facts + extra, major=True)


# --- charts -----------------------------------------------------------------


def svg_radar(axes: Sequence[tuple[str, float | None]]) -> str:
    """A 0-100 radar (inline SVG). ``None`` axes plot at 0 and are labelled not measured."""
    n = len(axes)
    cx, cy, r = 160.0, 150.0, 105.0

    def pt(i: int, v: float) -> tuple[float, float]:
        ang = -math.pi / 2 + 2 * math.pi * i / n
        return cx + r * v / 100 * math.cos(ang), cy + r * v / 100 * math.sin(ang)

    rings = "".join(
        '<polygon fill="none" stroke="#d1d5db" stroke-width="1" points="'
        + " ".join(f"{x:.1f},{y:.1f}" for x, y in (pt(i, level) for i in range(n)))
        + '"/>'
        for level in (25, 50, 75, 100)
    )
    spokes = "".join(
        f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{pt(i, 100)[0]:.1f}" y2="{pt(i, 100)[1]:.1f}" stroke="#e5e7eb"/>'
        for i in range(n)
    )
    shape = " ".join(
        f"{x:.1f},{y:.1f}" for x, y in (pt(i, v or 0.0) for i, (_, v) in enumerate(axes))
    )
    labels: list[str] = []
    for i, (name, v) in enumerate(axes):
        lx, ly = pt(i, 122)
        anchor = "middle" if abs(lx - cx) < 8 else ("start" if lx > cx else "end")
        val = "n/m" if v is None else str(round(v))
        labels.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="11" text-anchor="{anchor}" '
            f'style="fill:var(--on-surface-muted)">{escape(name)} {val}</text>'
        )
    return (
        '<svg data-chart="radar" viewBox="-50 0 420 300" width="420" height="300" role="img" '
        'aria-label="Dimension radar">'
        f"{rings}{spokes}"
        f'<polygon points="{shape}" style="fill:var(--type-accent);fill-opacity:0.3;'
        'stroke:var(--type-accent);stroke-width:2"/>'
        f"{''.join(labels)}</svg>"
    )


def svg_funnel(stages: Sequence[tuple[str, float | None, str]], friction: float | None) -> str:
    """A journey funnel: one trapezoid per stage, width ∝ score, tech friction band.

    ``stages`` items are (stage label, score or ``None`` = not measured, detail).
    """
    row_h, gap, max_w, min_w, cx = 40, 6, 420.0, 80.0, 240.0
    parts = [
        '<defs><pattern id="nm-hatch" width="8" height="8" patternUnits="userSpaceOnUse" '
        'patternTransform="rotate(45)"><rect width="8" height="8" fill="#e5e7eb"/>'
        '<line x1="0" y1="0" x2="0" y2="8" stroke="#9ca3af" stroke-width="3"/></pattern></defs>'
    ]
    for i, (label, score, detail) in enumerate(stages):
        w = min_w + (max_w - min_w) * ((score or 0.0) / 100)
        w2 = w * 0.92
        y = i * (row_h + gap)
        pts = f"{cx - w / 2:.1f},{y} {cx + w / 2:.1f},{y} {cx + w2 / 2:.1f},{y + row_h} {cx - w2 / 2:.1f},{y + row_h}"
        fill = "url(#nm-hatch)" if score is None else tone(score)
        val = "not measured" if score is None else f"{round(score)}"
        parts.append(f'<polygon points="{pts}" fill="{fill}"/>')
        parts.append(
            f'<text x="{cx:.1f}" y="{y + 25}" font-size="13" text-anchor="middle" '
            f'fill="{"#111827" if score is None else "#ffffff"}" font-weight="700">'
            f"{escape(label)} · {escape(val)}</text>"
        )
        if detail:
            parts.append(
                f'<text x="{cx + max_w / 2 + 12:.1f}" y="{y + 25}" font-size="11" style="fill:var(--on-surface-muted)">'
                f"{escape(detail)}</text>"
            )
    total_h = len(stages) * (row_h + gap)
    band_x = 6
    if friction is None:
        parts.append(
            f'<rect x="{band_x}" y="0" width="14" height="{total_h - gap}" fill="url(#nm-hatch)"/>'
        )
        band_label = "Tech friction: not measured"
    else:
        opacity = max(0.15, min(1.0, (100 - friction) / 100))
        parts.append(
            f'<rect x="{band_x}" y="0" width="14" height="{total_h - gap}" fill="#dc2626" '
            f'fill-opacity="{opacity:.2f}"/>'
        )
        band_label = f"Tech friction overlay: {round(friction)}/100"
    parts.append(
        f'<text x="{band_x}" y="{total_h + 12}" font-size="11" style="fill:var(--on-surface-muted)">{escape(band_label)}</text>'
    )
    return (
        f'<svg data-chart="funnel" viewBox="0 0 640 {total_h + 20}" width="640" '
        f'height="{total_h + 20}" role="img" aria-label="On-page journey funnel">'
        f"{''.join(parts)}</svg>"
    )
