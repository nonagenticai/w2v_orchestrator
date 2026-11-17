"""SEO audit report — "Search & AI discoverability audit" (design §2.1).

Reader: a web developer or the SEO owner who files tickets for developers. The
form is a technical checklist: pass/fail per check, the measured value next to
the target, and a fix per row. Only ``seo-*`` findings appear here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
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
    bucket_fix_list,
    check_status,
    checklist,
    checks_of,
    critique_card,
    dict_rows,
    document,
    exec_summary_block,
    findings_of,
    flag_chip,
    grounding_map,
    header_band,
    measured_fact,
    methodology,
    not_measured,
    opt_num,
    projection_block,
    scan_url,
    score_row,
    section,
    show_ai_section,
    str_list,
    txt,
    unbucketed_findings,
)

TITLE = "Search & AI discoverability audit"
QUESTION = (
    "Can search and AI engines find, understand, and rank this page, "
    "and what exactly is blocking that?"
)
_KEY = "group"
# Display order of the SEO groups (AS ``SCN/seo/scoring.py`` GROUP_WEIGHTS) plus
# the §4 SEO-G1 performance group, which is reported but not yet measured.
_GROUP_ORDER = (
    "on_page",
    "technical",
    "content",
    "keyword",
    "schema",
    "eeat",
    "internal_linking",
    "snippet",
    "geo",
    "performance",
)
_PERF_NOT_WIRED = "No performance measurement on the audit path — Lighthouse/CWV not wired"


def _facts(scan: Scan) -> Row | None:
    raw = scan.get("facts")
    return as_dict(raw) if isinstance(raw, dict) else None


def _groups(scan: Scan) -> dict[str, Row]:
    return {str(g.get("group")): g for g in dict_rows(scan.get("groups")) if g.get("group")}


def _group_score(g: Row) -> float | None:
    """A group's score, or ``None`` when the scan says it was not measured."""
    if g.get("measured") is False:
        return None
    return opt_num(g.get("score"))


def _indexable_badge(facts: Row | None) -> str:
    idx = as_dict(facts.get("indexable")) if facts else {}
    value = idx.get("value")
    if not isinstance(value, bool):
        return '<span class="verdict" style="background:#6b7280">INDEXABILITY NOT MEASURED</span>'
    reasons = str_list(idx.get("reasons")) or str_list([idx.get("reason")])
    checked = str_list(idx.get("checked"))
    reason = escape("; ".join(reasons)) or (
        escape(f"checked: {', '.join(checked)}") if checked else ""
    )
    reason_html = f' <span class="muted">{reason}</span>' if reason else ""
    if value:
        return f'<span class="verdict" style="background:#15803d">INDEXABLE</span>{reason_html}'
    return f'<span class="verdict" style="background:#b91c1c">NOT INDEXABLE</span>{reason_html}'


def _cover(scan: Scan, groups: Mapping[str, Row]) -> str:
    parts: list[str] = []
    coverage = opt_num(scan.get("coverage"))
    unmeasured = str_list(scan.get("unmeasured_groups")) or [
        name for name, g in groups.items() if _group_score(g) is None
    ]
    if coverage is not None and coverage < 1:
        names = ", ".join(unmeasured) or "some groups"
        parts.append(
            f'<p class="muted">Score covers {round(coverage * 100)}% of the rubric; '
            f"{escape(names)} not measured.</p>"
        )
    rows: list[str] = []
    for name in _GROUP_ORDER:
        g = groups.get(name)
        if name == "performance" and g is None:
            rows.append(score_row(name, None, note=_PERF_NOT_WIRED))
            continue
        if g is None:
            continue
        rows.append(score_row(name, _group_score(g), note=str(g.get("note") or "")))
    for name, g in groups.items():
        if name not in _GROUP_ORDER:
            rows.append(score_row(name, _group_score(g), note=str(g.get("note") or "")))
    parts.append("".join(rows) if groups else not_measured("this scan carries no group scores"))
    return section("verdict", "Discoverability verdict", "".join(parts))


def _technical(scan: Scan, facts: Row | None, ctx: _Ctx) -> str:
    if facts is None:
        strip = not_measured("this scan carries no measured technical facts")
    else:
        robots, robots_why = measured_fact(
            facts, "robots", missing="robots.txt not fetched on this path"
        )
        sitemap, sitemap_why = measured_fact(
            facts, "sitemap", missing="sitemap not fetched on this path"
        )
        canonical = facts.get("canonical")
        meta_robots = facts.get("meta_robots")
        url = scan_url(scan)
        https = url.startswith("https://") if url else None
        # The facts block is present, so a null meta_robots means "no tag" (index by
        # default), not "not measured".
        meta_ok = "noindex" not in str(meta_robots or "").lower()
        viewport = check_status(ctx.checks, "seo-viewport-missing")
        strip = (
            '<div class="strip">'
            f"{flag_chip('robots.txt', None if robots is None else robots.get('fetched'), na_reason=robots_why)}"
            f"{flag_chip('sitemap', None if sitemap is None else sitemap.get('found'), na_reason=sitemap_why)}"
            f"{flag_chip('canonical', bool(canonical))}"
            f"{flag_chip('meta-robots ok', meta_ok)}"
            f"{flag_chip('HTTPS', https)}"
            f"{flag_chip('viewport', None if viewport in (None, 'not_run') else viewport == 'pass')}"
            "</div>"
        )
        for why in dict.fromkeys(w for w in (robots_why, sitemap_why) if w):
            strip += not_measured(why)
        if robots is not None and robots.get("search_engines_blocked"):
            strip += '<p class="bad-fact">robots.txt blocks search engines</p>'
        if canonical:
            strip += f'<p class="muted">Canonical: {txt(canonical)}</p>'
    return section(
        "indexability",
        "Indexability & technical",
        strip + ctx.checklist("technical"),
        major=True,
    )


def _gauge(label: str, length: int | None, lo: int, hi: int) -> str:
    if length is None:
        return f"<p>{escape(label)}: length not measured</p>"
    ok = lo <= length <= hi
    cls = "st-pass" if ok else "st-fail"
    return f'<p>{escape(label)}: <span class="{cls}">{length} chars</span> <span class="muted">(target {lo} to {hi})</span></p>'


def _on_page(scan: Scan, facts: Row | None, ctx: _Ctx) -> str:
    if facts is None:
        serp = not_measured("title / meta description text not in this scan")
        outline = ""
    else:
        title = as_dict(facts.get("title"))
        meta = as_dict(facts.get("meta_description"))
        t_text = str(title.get("text") or "")
        m_text = str(meta.get("text") or "")
        t_len = opt_num(title.get("length"))
        m_len = opt_num(meta.get("length"))
        t_html = (
            f'<div style="color:#1a0dab;font-size:18px">{escape(t_text[:60])}{"…" if len(t_text) > 60 else ""}</div>'
            if t_text
            else '<div class="bad-fact">No &lt;title&gt; found</div>'
        )
        m_html = (
            f'<div style="color:#4d5156">{escape(m_text[:160])}{"…" if len(m_text) > 160 else ""}</div>'
            if m_text
            else '<div class="bad-fact">No meta description found</div>'
        )
        serp = (
            '<div class="card serp" style="background:#ffffff">'
            f'<div style="color:#006621;font-size:12px">{txt(scan_url(scan))}</div>{t_html}{m_html}</div>'
            f"{_gauge('Title', None if t_len is None else int(t_len), 50, 60) if t_text else ''}"
            f"{_gauge('Meta description', None if m_len is None else int(m_len), 150, 160) if m_text else ''}"
        )
        h1s = str_list(facts.get("h1"))
        heads = dict_rows(facts.get("heading_outline"))
        lis = "".join(
            f'<li style="margin-left:{max(0, int(opt_num(h.get("level")) or 1) - 1) * 14}px">'
            f"H{int(opt_num(h.get('level')) or 1)} {txt(h.get('text'))}</li>"
            for h in heads
        )
        h1_line = (
            f"<p>H1: {', '.join(escape(h) for h in h1s)}</p>"
            if h1s
            else '<p class="bad-fact">No H1 found</p>'
        )
        images = as_dict(facts.get("images"))
        img_line = (
            f"<p>Images: {txt(images.get('total'))} total, {txt(images.get('missing_alt'))} missing alt text</p>"
            if images
            else not_measured("image alt-text counts not in this scan")
        )
        outline = (
            f"{h1_line}<h3>Heading outline</h3>"
            + (
                f'<ul class="plain">{lis}</ul>'
                if lis
                else not_measured("heading outline not in this scan")
            )
            + img_line
        )
    body = f'<div data-section="serp-preview"><h3>Search result preview</h3>{serp}</div>{outline}'
    return section("on-page", "On-page & meta", body + ctx.checklist("on_page"), major=True)


def _schema(facts: Row | None, ctx: _Ctx) -> str:
    sch = as_dict(facts.get("schema")) if facts else {}
    if not sch:
        body = not_measured("structured-data coverage not in this scan")
    else:
        present = set(str_list(sch.get("types_present")))
        missing = set(str_list(sch.get("missing")))
        incomplete = {str(k): str_list(v) for k, v in as_dict(sch.get("incomplete")).items()}
        recommended = str_list(sch.get("recommended")) or sorted(present | missing)
        industry = str(sch.get("industry") or "")
        note = (
            f"<p class='muted'>Detected industry: {escape(industry)}</p>"
            if industry
            else "<p class='muted'>Industry not detected — showing the universal set.</p>"
        )
        rows = "".join(
            "<tr>"
            f"<td>{escape(t)}</td>"
            f"<td>{'<span class=st-pass>✓</span>' if t in present else '<span class=st-fail>✗</span>'}</td>"
            f"<td>{'—' if t not in present else ('<span class=st-fail>incomplete</span>' if t in incomplete else '<span class=st-pass>complete</span>')}</td>"
            f"<td>{escape(', '.join(incomplete.get(t, []))) or '—'}</td>"
            "</tr>"
            for t in recommended
        )
        body = note + (
            '<table class="schema-matrix"><thead><tr><th>Type</th><th>Present</th><th>Complete</th>'
            f"<th>Missing properties</th></tr></thead><tbody>{rows}</tbody></table>"
            if rows
            else not_measured("no recommended schema types for this page")
        )
    return section("structured-data", "Structured data", body + ctx.checklist("schema"), major=True)


def _content(facts: Row | None, ctx: _Ctx) -> str:
    parts: list[str] = []
    if facts is None:
        parts.append(not_measured("keyword and readability measurements not in this scan"))
    else:
        kw = str(facts.get("primary_keyword") or "")
        if not kw:
            parts.append(not_measured("no dominant keyword found"))
        else:
            place = facts.get("keyword_placement")
            if isinstance(place, dict):
                cells = "".join(
                    f"<td>{'—' if place.get(k) is None else ('✓' if place.get(k) else '✗')}</td>"
                    for k in ("title", "h1", "intro", "meta")
                )
                parts.append(
                    '<table class="kw-grid"><thead><tr><th>Keyword</th><th>Title</th><th>H1</th>'
                    f"<th>Intro</th><th>Meta</th></tr></thead><tbody><tr><td>{escape(kw)}</td>{cells}</tr></tbody></table>"
                )
            else:
                parts.append(f"<p>Primary keyword: <strong>{escape(kw)}</strong></p>")
                parts.append(not_measured("keyword placement grid — see the Keyword checks below"))
        top = str_list(facts.get("top_keywords"))
        if top:
            parts.append(f"<p class='muted'>Top keywords: {escape(', '.join(top))}</p>")
        rd, rd_why = measured_fact(facts, "readability", missing="readability not in this scan")
        parts.append(
            f"<p>Readability: grade {txt(rd.get('grade'))} · Flesch {txt(rd.get('flesch'))} · {txt(rd.get('words'))} words</p>"
            if rd is not None
            else not_measured(rd_why)
        )
    parts.append("<h3>Content</h3>" + ctx.checklist("content"))
    parts.append("<h3>Keyword</h3>" + ctx.checklist("keyword"))
    parts.append("<h3>E-E-A-T signals</h3>" + ctx.checklist("eeat"))
    return section("content-keyword", "Content & keywords", "".join(parts), major=True)


def _links(facts: Row | None, groups: Mapping[str, Row], ctx: _Ctx) -> str:
    parts: list[str] = []
    links, links_why = measured_fact(facts, "links", missing="link counts not in this scan")
    if links is not None:
        parts.append(
            f"<p>Internal links {txt(links.get('internal'))} · external {txt(links.get('external'))} · "
            f"generic anchors {txt(links.get('generic_anchors'))}</p>"
        )
    else:
        parts.append(not_measured(links_why))
    g = groups.get("internal_linking", {})
    crawl, crawl_why = measured_fact(
        facts, "crawl", missing="site-wide linking not measured (cross-page crawl off)"
    )
    if g.get("measured") is False:
        parts.append(not_measured(str(g.get("note") or "site-wide linking crawl failed")))
    elif crawl is not None:
        broken = _link_items(crawl.get("broken"))
        orphans = _link_items(crawl.get("orphans"))
        parts.append(f"<p>Crawled {txt(crawl.get('pages'))} pages.</p>")
        parts.append(
            "<h3>Broken links</h3>"
            + (
                '<ul class="plain">' + "".join(f"<li>{escape(b)}</li>" for b in broken) + "</ul>"
                if broken
                else "<p>None found.</p>"
            )
        )
        parts.append(
            "<h3>Orphan pages</h3>"
            + (
                '<ul class="plain">' + "".join(f"<li>{escape(o)}</li>" for o in orphans) + "</ul>"
                if orphans
                else "<p>None found.</p>"
            )
        )
    else:
        parts.append(not_measured(crawl_why))
    parts.append(ctx.checklist("internal_linking"))
    return section("links", "Links", "".join(parts))


def _geo(facts: Row | None, ctx: _Ctx) -> str:
    extra = ""
    if facts is not None:
        robots, _ = measured_fact(facts, "robots", missing="")
        llms, llms_why = measured_fact(facts, "llms_txt", missing="llms.txt not fetched")
        bots = str_list(robots.get("ai_bots_blocked")) if robots is not None else []
        extra = (
            '<div class="strip">'
            f"{flag_chip('llms.txt', None if llms is None else llms.get('found'), na_reason=llms_why)}"
            f"{flag_chip('AI bots allowed', None if robots is None else not bots, na_reason='robots.txt not fetched')}"
            "</div>"
        )
        if bots:
            extra += f"<p class='muted'>Blocked AI crawlers: {escape(', '.join(bots))}</p>"
        date = facts.get("content_date")
        extra += f"<p class='muted'>Content date: {txt(date) or 'not found'}</p>"
    else:
        extra = not_measured("robots/llms.txt live facts not in this scan")
    body = (
        extra
        + '<div class="two"><div><h3>Featured snippet</h3>'
        + ctx.checklist("snippet")
        + "</div><div><h3>AI answer engines</h3>"
        + ctx.checklist("geo")
        + "</div></div>"
    )
    return section("geo", "Snippet & AI-answer readiness", body)


def _performance(facts: Row | None, groups: Mapping[str, Row]) -> str:
    perf, perf_why = measured_fact(facts, "performance", missing="")
    g = groups.get("performance")
    if perf is None:
        note = str(g.get("note") or "") if g else ""
        return section(
            "performance",
            "Performance",
            not_measured(perf_why or note or _PERF_NOT_WIRED),
            major=True,
        )

    def cwv(label: str, value: float | None, good: float, poor: float, unit: str) -> str:
        if value is None:
            return f"<div class='tile na'><strong>{label}</strong>{not_measured('lab Core Web Vitals not wired')}</div>"
        band, cls = (
            ("good", "on")
            if value <= good
            else ("poor", "off")
            if value > poor
            else ("needs work", "na")
        )
        return f"<div class='tile {cls}'><strong>{label}</strong><div>{value:g}{unit} — {band}</div></div>"

    gauges = (
        '<div class="grid">'
        + cwv("LCP", opt_num(perf.get("lcp_ms")), 2500, 4000, " ms")
        + cwv("CLS", opt_num(perf.get("cls")), 0.1, 0.25, "")
        + cwv("TBT", opt_num(perf.get("tbt_ms")), 200, 600, " ms")
        + "</div>"
    )
    blocking = str_list(perf.get("render_blocking"))
    proxies = (
        f"<p class='muted'>Measurement source: {txt(perf.get('source')) or 'unknown'}</p>"
        f"<p>Page weight: {txt(perf.get('page_weight_kb')) or '—'} KB · images without dimensions: "
        f"{txt(perf.get('images_without_dimensions')) or '—'} · render-blocking resources: {len(blocking)}</p>"
    )
    return section("performance", "Performance", gauges + proxies, major=True)


def _fix_list(ctx: _Ctx, groups: Mapping[str, Row]) -> str:
    names = [n for n in _GROUP_ORDER if n in groups] + [n for n in groups if n not in _GROUP_ORDER]
    for f in ctx.findings:
        g = str(f.get(_KEY) or "")
        if g and g not in names:
            names.append(g)
    other = unbucketed_findings(ctx.findings, key=_KEY, grounding=ctx.grounding)
    if not names and not other:
        return section(
            "fix-list",
            "Fix list by group",
            not_measured("no groups or findings in this scan"),
            major=True,
        )
    parts = [
        f"<h3>{escape(n)}</h3>"
        + bucket_fix_list(
            bucket=n, key=_KEY, findings=ctx.findings, checks=ctx.checks, grounding=ctx.grounding
        )
        for n in names
    ]
    return section("fix-list", "Fix list by group", "".join(parts) + other, major=True)


def _ai_section(scan: Scan, state: AiState) -> str:
    if not show_ai_section(scan, state):
        return ""
    ann = dict_rows(as_dict(scan.get("ai")).get("annotations"))
    by_kind = {str(a.get("kind") or ""): a for a in ann}

    def card(
        kind: str,
        title: str,
        layer: str,
        body_of: Callable[[Row], str],
        not_run_hint: str = "",
    ) -> str:
        a = by_kind.get(kind)
        if a is not None:
            d = as_dict(a.get("data"))
            summary = txt(a.get("summary"))
            body = body_of(d) + (f"<p class='muted'>{summary}</p>" if summary else "")
        elif (problem := state.layer_problem(layer)) is not None:
            body = ai_unavailable(problem)
        elif layer not in state.layers:
            body = f"<p class='muted'>Not run for this scan{escape(not_run_hint)}.</p>"
        else:
            body = ai_unavailable("no result reported for this analysis")
        return f'<div class="card" data-card="{kind}"><h3>{escape(title)}</h3>{body}</div>'

    intent = card(
        "search_intent",
        "Search intent",
        "seo_intent",
        lambda a: (
            f"<p>Intent: <strong>{txt(a.get('intent'))}</strong> · keyword: {txt(a.get('keyword'))}</p>"
        ),
    )
    eeat = card(
        "eeat_depth",
        "E-E-A-T depth",
        "seo_eeat",
        lambda a: (
            f"<p>Experience {txt(a.get('experience'))} · Expertise {txt(a.get('expertise'))}</p>"
            f"<p class='muted'>{txt(a.get('evidence'))}</p>"
        ),
    )
    gap_hint = " — add competitor URLs to compare content"
    gaps = card(
        "content_gap",
        "Competitor content gaps",
        "content_gap",
        lambda a: (
            '<ul class="plain">'
            + "".join(f"<li>{escape(t)}</li>" for t in str_list(a.get("topics") or a.get("gaps")))
            + "</ul>"
        ),
        gap_hint,
    )
    body = intent + eeat + gaps + critique_card(scan, state)
    return section("ai-analysis", "AI analysis — search", body, major=True)


def _link_items(value: object) -> list[str]:
    """Crawl list entries as display text (plain URLs, or ``{url: ...}`` rows)."""
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, dict):
            row = as_dict(item)
            out.append(str(row.get("url") or row.get("href") or row))
        elif item:
            out.append(str(item))
    return out


class _Ctx:
    """Per-render lookups shared by the section builders."""

    def __init__(self, scan: Scan) -> None:
        self.findings = [
            f
            for f in findings_of(scan)
            if not str(f.get("check_id") or "").startswith(("cro-", "mkt-"))
        ]
        self.checks = checks_of(scan)
        self.grounding = grounding_map(scan)

    def checklist(self, group: str) -> str:
        return checklist(self.checks, key=_KEY, buckets=(group,), findings=self.findings)


def render(scan: Scan, tok: Mapping[str, str]) -> str:
    """Render an SEO audit scan to a self-contained HTML document."""
    state = ai_state(scan)
    facts = _facts(scan)
    groups = _groups(scan)
    ctx = _Ctx(scan)
    exec_html = exec_summary_block(scan, ctx.findings)
    exec_sec = section("exec-summary", "Search verdict in brief", exec_html) if exec_html else ""
    proj = projection_block(scan)
    body = "".join(
        [
            header_band(
                title=TITLE,
                question=QUESTION,
                scan=scan,
                extra=f"<div>{_indexable_badge(facts)}</div>",
            ),
            ai_status_line(state),
            exec_sec,
            _cover(scan, groups),
            _technical(scan, facts, ctx),
            _on_page(scan, facts, ctx),
            _schema(facts, ctx),
            _content(facts, ctx),
            _links(facts, groups, ctx),
            _geo(facts, ctx),
            _performance(facts, groups),
            _fix_list(ctx, groups),
            _ai_section(scan, state),
            section("projection", "Score projection", proj) if proj else "",
            methodology(scan, rows=dict_rows(scan.get("groups")), key=_KEY),
        ]
    )
    return document(title=TITLE, tok=tok, accent=TYPE_ACCENTS["seo_audit"], body=body)
