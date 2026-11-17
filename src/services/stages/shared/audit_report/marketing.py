"""Marketing scorecard report — "Go-to-market scorecard" (design §2.3).

Reader: a founder / CMO / exec sponsor. Page 1 is a one-pager: a 6-axis radar,
the weights table, and three moves. Three of the six dimensions (content,
conversion, seo) are borrowed from the CRO and SEO engines (F18), so the
scorecard shows each as one reference row with a pointer to that report and
**never lists their cro-/seo- findings**. Its body is the ``mkt-*`` checks only.
"""

from __future__ import annotations

from collections.abc import Mapping
from html import escape

from src.services.stages.shared.audit_report.primitives import (
    TYPE_ACCENTS,
    AiState,
    Row,
    Scan,
    ai_state,
    ai_status_line,
    ai_unavailable,
    as_dict,
    checklist,
    checks_of,
    critique_card,
    dial,
    dict_rows,
    document,
    exec_summary_block,
    findings_of,
    findings_table,
    flag_chip,
    grounding_map,
    header_band,
    methodology,
    not_measured,
    opt_num,
    rank_badge,
    score_row,
    section,
    show_ai_section,
    str_list,
    svg_radar,
    tile,
    txt,
)

TITLE = "Go-to-market scorecard"
QUESTION = (
    "How strong is this page as a go-to-market asset compared with what a credible "
    "competitor shows, and where should we invest?"
)
_KEY = "dimension"
_OWN = ("competitive", "brand", "growth")
_BORROWED = ("content", "conversion", "seo")
_AXES = ("content", "conversion", "seo", "competitive", "brand", "growth")
SOURCE_ENGINE = {
    "content": "CRO scanner",
    "conversion": "CRO scanner",
    "seo": "SEO scanner",
    "competitive": "Marketing scanner",
    "brand": "Marketing scanner",
    "growth": "Marketing scanner",
}
_ENGINE_LABEL = {"cro": "CRO scanner", "seo": "SEO scanner", "marketing": "Marketing scanner"}
_REFERENCE = {"content": "CRO audit", "conversion": "CRO audit", "seo": "SEO audit"}


def _facts(scan: Scan) -> Row | None:
    raw = scan.get("facts")
    return as_dict(raw) if isinstance(raw, dict) else None


def _dims(scan: Scan) -> dict[str, Row]:
    return {str(d.get(_KEY)): d for d in dict_rows(scan.get("dimensions")) if d.get(_KEY)}


def _dim_score(d: Row | None) -> float | None:
    if d is None or d.get("measured") is False:
        return None
    return opt_num(d.get("score"))


def _own_finding(f: Row) -> bool:
    """A finding the scorecard may list: an own dimension, never a cro-/seo- check."""
    cid = str(f.get("check_id") or "")
    if cid.startswith(("cro-", "seo-")):
        return False
    return str(f.get(_KEY) or "") in _OWN


def _one_pager(scan: Scan, dims: Mapping[str, Row], ctx: _Ctx, state: AiState) -> str:
    if dims:
        axes = [(name, _dim_score(dims.get(name))) for name in _AXES]
        axes += [(n, _dim_score(d)) for n, d in dims.items() if n not in _AXES]
        radar = svg_radar(axes)
    else:
        radar = not_measured("this scan carries no dimension scores")
    rows: list[str] = []
    sources = {
        str(r.get(_KEY)): r for r in dict_rows(as_dict(scan.get("facts")).get("dimension_sources"))
    }
    for name in list(_AXES) + [n for n in dims if n not in _AXES]:
        d = dims.get(name)
        if d is None:
            continue
        src = sources.get(name, {})
        w = opt_num(d.get("weight"))
        s = _dim_score(d)
        reported = opt_num(src.get("weighted_points"))
        pts = (
            f"{reported:.1f}"
            if reported is not None
            else ("—" if w is None or s is None else f"{w * s:.1f}")
        )
        engine = _ENGINE_LABEL.get(
            str(src.get("source_engine") or ""), SOURCE_ENGINE.get(name, "—")
        )
        rows.append(
            f"<tr><td>{escape(name)}</td><td>{'—' if w is None else f'{w:.2f}'}</td>"
            f"<td>{'not measured' if s is None else round(s)}</td><td>{pts}</td>"
            f"<td>{escape(engine)}</td></tr>"
        )
    weights = (
        '<table class="weights-onepager"><thead><tr><th>Dimension</th><th>Weight</th><th>Score</th>'
        f"<th>Weighted points</th><th>Source engine</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
        if rows
        else ""
    )
    moves = ctx.own_findings[:3]
    moves_html = (
        '<ol class="moves">'
        + "".join(
            f"<li>{rank_badge(m)} <strong>{txt(m.get('title'))}</strong> <span class='muted'>({txt(m.get(_KEY))})</span></li>"
            for m in moves
        )
        + "</ol>"
        if moves
        else not_measured("no brand / competitive / growth gaps were reported")
    )
    narrative = exec_summary_block(scan, ctx.own_findings)
    if not narrative:
        if state.status == "not_requested":
            narrative = '<p class="muted">AI analysis: not requested</p>'
        elif state.status == "unreported":
            narrative = '<p class="muted">AI analysis: not reported for this scan</p>'
        else:
            narrative = ai_unavailable(
                state.layer_problem("exec_summary") or "the narrative was not produced"
            )
    body = (
        f'<div class="two"><div>{radar}</div><div>{weights}</div></div>'
        f"<h3>3 moves</h3>{moves_html}"
        f'<div class="card" data-card="narrative"><h3>Narrative</h3>{narrative}</div>'
    )
    return section("one-pager", "Executive one-pager", body)


def _own_section(
    key: str, title: str, dim: str, dims: Mapping[str, Row], facts_html: str, ctx: _Ctx
) -> str:
    d = dims.get(dim)
    head = (
        score_row(dim, _dim_score(d), note=str((d or {}).get("note") or ""))
        if d
        else not_measured(f"no {dim} score in this scan")
    )
    rows = [f for f in ctx.own_findings if str(f.get(_KEY) or "") == dim]
    gaps = findings_table(rows, ctx.grounding) if rows else ""
    body = (
        head + facts_html + ctx.checklist(dim) + (f"<h3>Gaps to close</h3>{gaps}" if gaps else "")
    )
    return section(key, title, body, major=True)


def _present(value: object) -> object:
    """A ``{present, url}`` fact (or a legacy bare bool) as a present/missing flag."""
    if isinstance(value, dict):
        return as_dict(value).get("present")
    return value


def _card_fields_tile(value: object) -> str:
    """The share-card meta tags: present when all four are declared."""
    if not isinstance(value, dict):
        return tile("Social card fields", None if value is None else bool(value))
    fields = as_dict(value)
    on = [k for k, v in fields.items() if v]
    return tile(
        "Social card fields",
        len(on) == len(fields) and bool(fields),
        detail=escape(f"{len(on)} of {len(fields)} tags"),
    )


def _not_measured_reason(facts: Row | None, key: str) -> str:
    """The reason of a ``{measured: false, reason}`` fact, or ``""``."""
    block = as_dict(facts.get(key)) if facts else {}
    return str(block.get("reason") or "") if block.get("measured") is False else ""


def _positioning(facts: Row | None, dims: Mapping[str, Row], ctx: _Ctx) -> str:
    pos = as_dict(facts.get("positioning")) if facts else {}
    if not pos:
        html = not_measured("positioning copy not in this scan")
    else:
        fold = str(pos.get("fold_copy_excerpt") or "")
        signals = str_list(pos.get("reputation_signals"))
        html = (
            (
                f"<blockquote>{escape(fold)}</blockquote>"
                if fold
                else '<p class="bad-fact">No above-the-fold copy found</p>'
            )
            + '<div class="strip">'
            # The positioning block was measured, so a null claim means none was found.
            + flag_chip("differentiation claim", bool(pos.get("differentiation_claim")))
            + flag_chip("comparison page", bool(pos.get("comparison_page")))
            + "</div>"
            + (
                f"<p class='muted'>Reputation signals: {escape(', '.join(signals))}</p>"
                if signals
                else ""
            )
        )
    html += not_measured(
        _not_measured_reason(facts, "competitive_benchmark")
        or "No competitor pages analysed — add competitor URLs"
    )
    return _own_section("positioning", "Positioning & competitive", "competitive", dims, html, ctx)


def _brand(facts: Row | None, dims: Mapping[str, Row], ctx: _Ctx) -> str:
    b = as_dict(facts.get("brand")) if facts else {}
    if not b:
        html = not_measured("brand-asset inventory not in this scan")
    else:
        og = str(b.get("og_image_url") or "")
        html = (
            '<div class="grid">'
            + tile("Logo", _present(b.get("logo")))
            + tile("Social share image", bool(og) if "og_image_url" in b else None)
            + tile("Favicon", _present(b.get("favicon")))
            + tile("About page link", _present(b.get("about_link")))
            + tile("Contact", _present(b.get("contact")))
            + _card_fields_tile(b.get("social_card_fields"))
            + "</div>"
        )
        if og.startswith(("https://", "http://")):
            html += f'<img src="{escape(og)}" alt="Social share image" style="max-width:240px;margin-top:8px">'
    return _own_section("brand", "Brand presence", "brand", dims, html, ctx)


def _channels(facts: Row | None, dims: Mapping[str, Row], ctx: _Ctx) -> str:
    ch = as_dict(facts.get("channels")) if facts else {}
    if not ch:
        html = not_measured("owned-channel inventory not in this scan")
    else:
        social = dict_rows(ch.get("social_profiles"))
        vendors = str_list(ch.get("analytics_vendors"))
        html = (
            '<div class="grid">'
            + tile(
                "Owned social",
                bool(social) if "social_profiles" in ch else None,
                detail=escape(", ".join(str(s.get("platform") or "") for s in social)),
            )
            + tile("Email capture", ch.get("email_capture"))
            + tile(
                "Analytics",
                bool(vendors) if "analytics_vendors" in ch else None,
                detail=escape(", ".join(vendors)),
            )
            + tile("Pricing visible", ch.get("pricing_visible"))
            + tile("Referral", ch.get("referral"))
            + tile("Retention hook", ch.get("retention_hook"))
            + "</div>"
            + f"<p class='muted'>Consent banner: {'yes' if ch.get('consent') else 'no' if 'consent' in ch else 'not measured'}</p>"
        )
    html += not_measured(
        _not_measured_reason(facts, "channel_performance")
        or "Channel performance needs analytics/ad-account data — not available to a page scan"
    )
    return _own_section("channels", "Owned channels & growth", "growth", dims, html, ctx)


def _reference(scan: Scan, dims: Mapping[str, Row]) -> str:
    related = as_dict(scan.get("related_reports"))
    rows: list[str] = []
    for name in _BORROWED:
        d = dims.get(name)
        report = _REFERENCE[name]
        rtype = "cro_audit" if report == "CRO audit" else "seo_audit"
        sib = as_dict(related.get(rtype))
        pointer = (
            f"Details: {report} of {txt(sib.get('completed_at'))}"
            if sib.get("completed_at")
            else f"Run {'an' if report.startswith('SEO') else 'a'} {report} for the breakdown"
        )
        w = opt_num(d.get("weight")) if d else None
        rows.append(
            f"<div>{score_row(name, _dim_score(d), note='not in this scan')}"
            f"<p class='muted'>weight {'—' if w is None else f'{w:.2f}'} · {pointer}</p></div>"
        )
    return section("reference", "Conversion & discoverability", "".join(rows))


def _ai_section(scan: Scan, state: AiState) -> str:
    if not show_ai_section(scan, state):
        return ""
    items = dict_rows(scan.get("llm_assessments"))
    if items:
        cards = "".join(
            f'<div class="card" data-card="assess-{txt(a.get("element"))}"><h3>{txt(a.get("element"))}</h3>'
            f"<p>Score {dial(a.get('score'))} {txt(a.get('assessment'))}</p>"
            f"<blockquote>{txt(a.get('suggestion')) or '—'}</blockquote></div>"
            for a in items
        )
    else:
        cards = ai_unavailable(
            state.layer_problem("micro_judge") or "no copy assessment was returned"
        )
    return section(
        "ai-analysis", "Copy assessment (AI)", cards + critique_card(scan, state), major=True
    )


class _Ctx:
    def __init__(self, scan: Scan) -> None:
        self.own_findings = [f for f in findings_of(scan) if _own_finding(f)]
        self.checks = checks_of(scan)
        if self.checks is not None:
            self.checks = [
                c
                for c in self.checks
                if not str(c.get("check_id") or c.get("id") or "").startswith(("cro-", "seo-"))
            ]
        self.grounding = grounding_map(scan)

    def checklist(self, dim: str) -> str:
        return checklist(self.checks, key=_KEY, buckets=(dim,), findings=self.own_findings)


def render(scan: Scan, tok: Mapping[str, str]) -> str:
    """Render a marketing scorecard scan to a self-contained HTML document."""
    state = ai_state(scan)
    facts = _facts(scan)
    dims = _dims(scan)
    ctx = _Ctx(scan)
    body = "".join(
        [
            header_band(title=TITLE, question=QUESTION, scan=scan),
            ai_status_line(state),
            _one_pager(scan, dims, ctx, state),
            _positioning(facts, dims, ctx),
            _brand(facts, dims, ctx),
            _channels(facts, dims, ctx),
            _reference(scan, dims),
            _ai_section(scan, state),
            methodology(
                scan,
                rows=dict_rows(scan.get("dimensions")),
                key=_KEY,
                extra_cols={"Source engine": SOURCE_ENGINE},
                note=(
                    "content, conversion and seo are scored by the CRO and SEO engines and shown here "
                    "as reference rows only; their findings live in those reports."
                ),
            ),
        ]
    )
    return document(title=TITLE, tok=tok, accent=TYPE_ACCENTS["marketing_scorecard"], body=body)
