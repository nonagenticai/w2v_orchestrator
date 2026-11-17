"""CRO audit report — "Conversion audit" (design §2.2).

Reader: a growth/product marketer or the landing-page owner. The scanner sees one
page, so the "funnel" is the on-page journey (a design mapping, §2.2):

``Attention (hero) → Understanding (valueprop) → Trust (social + objection) →
Action (cta) → Commitment (form) → Fallback (footer)``, with ``tech`` drawn as a
friction overlay. The body is inventories (CTAs, form fields, trust tiles) and a
ranked experiment backlog. Only ``cro-*`` findings appear here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape

from src.services.stages.shared.audit_report.primitives import (
    TYPE_ACCENTS,
    AiState,
    Row,
    Scan,
    ai_state,
    ai_status_line,
    ai_unavailable,
    all_passed,
    as_dict,
    check_counts,
    checklist,
    checks_of,
    critique_card,
    deterministic_fix,
    dial,
    dict_rows,
    document,
    exec_summary_block,
    findings_of,
    findings_table,
    grounding_map,
    header_band,
    methodology,
    not_measured,
    opt_num,
    projection_block,
    rank_badge,
    score_row,
    section,
    show_ai_section,
    svg_funnel,
    tile,
    txt,
    unbucketed_findings,
    weight_text,
)

TITLE = "Conversion audit"
QUESTION = "Where does a visitor drop off on this page, and which change should we test first?"
_KEY = "section"

# (stage, the CRO sections that make it up) — the §2.2 journey mapping.
JOURNEY: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Attention", ("hero",)),
    ("Understanding", ("valueprop",)),
    ("Trust", ("social", "objection")),
    ("Action", ("cta",)),
    ("Commitment", ("form",)),
    ("Fallback", ("footer",)),
)
_TECH = "tech"


def _facts(scan: Scan) -> Row | None:
    raw = scan.get("facts")
    return as_dict(raw) if isinstance(raw, dict) else None


def _subsections(scan: Scan) -> dict[str, Row]:
    return {str(s.get(_KEY)): s for s in dict_rows(scan.get("subsections")) if s.get(_KEY)}


def _sec_score(s: Row | None) -> float | None:
    if s is None or s.get("measured") is False:
        return None
    return opt_num(s.get("score"))


def _journey_fact(facts: Row | None) -> Row | None:
    raw = facts.get("journey") if facts else None
    return as_dict(raw) if isinstance(raw, dict) and raw.get("stages") else None


def _stage_scores(
    subs: Mapping[str, Row], facts: Row | None
) -> list[tuple[str, float | None, str]]:
    """(stage, weighted score or ``None`` = not measured, weakest-section detail).

    analysis_service computes the journey itself (``facts.journey``); that is used
    when present. Older scans fall back to the same mapping over ``subsections``.
    """
    journey = _journey_fact(facts)
    if journey is not None:
        rows: list[tuple[str, float | None, str]] = []
        for st in dict_rows(journey.get("stages")):
            label = str(st.get("label") or st.get("stage") or "")
            score = None if st.get("measured") is False else opt_num(st.get("score"))
            per = {
                str(k): v
                for k, v in as_dict(st.get("section_scores")).items()
                if opt_num(v) is not None
            }
            detail = ""
            if per:
                name = min(per, key=lambda k: opt_num(per[k]) or 0.0)
                detail = f"{name} {round(opt_num(per[name]) or 0.0)}/100"
            rows.append((label, score, detail))
        return rows
    out: list[tuple[str, float | None, str]] = []
    for stage, names in JOURNEY:
        scored = [
            (n, _sec_score(subs.get(n)), opt_num(subs.get(n, {}).get("weight")) or 1.0)
            for n in names
        ]
        measured = [(n, s, w) for n, s, w in scored if s is not None]
        if not measured:
            out.append((stage, None, ""))
            continue
        total_w = sum(w for _, _, w in measured)
        score = sum(s * w for _, s, w in measured) / total_w
        weakest = min(measured, key=lambda t: t[1])
        out.append((stage, score, f"{weakest[0]} {round(weakest[1])}/100"))
    return out


def _yes_no(value: object) -> str:
    return ("yes" if value else "no") if isinstance(value, bool) else "not measured"


def _cover(scan: Scan, facts: Row | None, stages: Sequence[tuple[str, float | None, str]]) -> str:
    measured = [(st, s, d) for st, s, d in stages if s is not None]
    if measured:
        stage, _, detail = min(measured, key=lambda t: t[1] or 0.0)
        callout = f'<p class="callout">Biggest leak: {escape(stage)} — {escape(detail)}</p>'
    else:
        callout = not_measured("no journey stage was scored in this scan")
    shot = str(facts.get("screenshot_url") or "") if facts else ""
    img = (
        f'<img src="{escape(shot)}" alt="Page screenshot" style="max-width:100%;max-height:320px;border:1px solid #e5e7eb">'
        if shot.startswith(("https://", "http://"))
        else ""
    )
    return section("verdict", "Conversion verdict", callout + img)


def _journey(
    subs: Mapping[str, Row], facts: Row | None, stages: Sequence[tuple[str, float | None, str]]
) -> str:
    if not subs and not stages:
        return section(
            "journey", "Visitor journey", not_measured("this scan carries no section scores")
        )
    journey = _journey_fact(facts)
    friction = (
        opt_num(as_dict(journey.get("friction")).get("score"))
        if journey is not None
        else _sec_score(subs.get(_TECH))
    )
    legend = (
        '<p class="muted">Width and colour follow each stage\'s score; hatched = not measured. '
        "The red band on the left is technical friction, which affects every stage.</p>"
    )
    unmapped = [n for n in subs if n != _TECH and all(n not in names for _, names in JOURNEY)]
    extra = (
        f"<p class='muted'>Sections outside the journey map: {escape(', '.join(unmapped))}</p>"
        if unmapped
        else ""
    )
    if all(score is None for _, score, _ in stages):
        extra += not_measured("no journey stage was scored in this scan")
    return section("journey", "Visitor journey", svg_funnel(stages, friction) + legend + extra)


def _above_fold(facts: Row | None, ctx: _Ctx) -> str:
    parts: list[str] = []
    hero = as_dict(facts.get("hero")) if facts else {}
    vp = as_dict(facts.get("value_prop")) if facts else {}
    if not hero:
        parts.append(not_measured("hero facts (H1, above-the-fold CTA) not in this scan"))
    else:
        h1 = str(hero.get("h1_text") or "")
        parts.append(
            f"<blockquote><strong>{escape(h1)}</strong></blockquote>"
            if h1
            else '<p class="bad-fact">No H1 on the page</p>'
        )
        above = hero.get("has_cta_above_fold")
        fold = (
            ("yes" if above else "no")
            if isinstance(above, bool)
            else "not measurable from a page fetch (no layout)"
        )
        before_h2 = hero.get("cta_before_first_h2")
        h2_line = (
            f" · CTA before the first H2: {'yes' if before_h2 else 'no'}"
            if isinstance(before_h2, bool)
            else ""
        )
        parts.append(
            f"<p class='muted'>H1 length {txt(hero.get('h1_length')) or '—'} chars · CTA above the fold: "
            f"{fold}{h2_line} · nav links: {txt(hero.get('nav_link_count')) or '—'}</p>"
        )
    if not vp:
        parts.append(not_measured("value-proposition copy not in this scan"))
    else:
        lead = str(vp.get("lead_text") or "")
        parts.append(
            f"<blockquote>{escape(lead)}</blockquote>"
            if lead
            else '<p class="bad-fact">No lead paragraph found</p>'
        )
        parts.append(
            f"<p class='muted'>Benefit bullets {txt(vp.get('bullets')) or '—'} · quantified benefit: "
            f"{_yes_no(vp.get('quantified_benefit'))} · reading grade {txt(vp.get('reading_grade')) or 'not measured'}</p>"
        )
    parts.append(ctx.checklist(("hero", "valueprop")))
    return section("above-fold", "Above the fold", "".join(parts), major=True)


def _ctas(facts: Row | None, ctx: _Ctx) -> str:
    if facts is None or "ctas" not in facts:
        body = not_measured("the CTA inventory is not in this scan")
    else:
        ctas = dict_rows(facts.get("ctas"))
        if not ctas:
            body = '<p class="bad-fact">No call-to-action found on this page.</p>'
        else:
            summary = as_dict(facts.get("cta_summary"))
            total = int(opt_num(summary.get("count")) or len(ctas))
            generic = (
                int(opt_num(summary.get("generic_count")) or 0)
                if summary
                else sum(1 for c in ctas if c.get("generic"))
            )
            rows = "".join(
                "<tr>"
                f"<td>{int(opt_num(c.get('position_index')) or 0) + 1}</td><td><strong>{txt(c.get('text'))}</strong></td>"
                f"<td>{txt(c.get('tag'))}</td><td>{txt(c.get('href_kind'))}</td>"
                f"<td>{'✓' if c.get('first_person') else '—'}</td>"
                f"<td>{'⚠ generic' if c.get('generic') else '—'}</td>"
                f"<td>{'✓' if c.get('has_microcopy') else '—'}</td></tr>"
                for c in ctas
            )
            body = (
                f"<p>{total} CTAs · {generic} generic ({round(100 * generic / max(total, 1))}%)"
                + (f" · showing the first {len(ctas)}" if summary.get("truncated") else "")
                + "</p>"
                '<table class="cta-inventory"><thead><tr><th>#</th><th>Text</th><th>Tag</th><th>Target</th>'
                f"<th>First person</th><th>Generic</th><th>Microcopy</th></tr></thead><tbody>{rows}</tbody></table>"
            )
    return section("cta-inventory", "CTA inventory", body + ctx.checklist(("cta",)), major=True)


def _form(facts: Row | None, ctx: _Ctx) -> str:
    form = as_dict(facts.get("form")) if facts else {}
    if not form:
        body = not_measured("form facts not in this scan")
    elif not form.get("present"):
        body = not_measured("No conversion form on this page")
    else:
        count = int(opt_num(form.get("field_count")) or 0)
        meter_score = max(0.0, 100.0 - max(0, count - 3) * 15.0)
        fields = dict_rows(form.get("fields"))
        rows = "".join(
            "<tr>"
            f"<td>{txt(f.get('type'))}</td><td>{txt(f.get('label')) or (txt(f.get('name')) + ' <span class=bad-fact>(no label)</span>' if f.get('name') else '<span class=bad-fact>no label</span>')}</td>"
            f"<td>{'required' if f.get('required') else 'optional'}</td>"
            f"<td>{txt(f.get('autocomplete')) or '—'}</td></tr>"
            for f in fields
        )
        table = (
            '<table class="form-fields"><thead><tr><th>Type</th><th>Label</th><th>Required</th>'
            f"<th>Autocomplete</th></tr></thead><tbody>{rows}</tbody></table>"
            if rows
            else not_measured("per-field detail not in this scan")
        )
        body = (
            f"<p>{count} fields · {txt(form.get('required_count')) or '—'} required · "
            f"multi-step: {'yes' if form.get('multistep') else 'no'} · placeholder-only labels: "
            f"{txt(form.get('placeholder_only_labels')) or '0'}</p>"
            + score_row("Friction meter", meter_score)
            + table
        )
    return section("form-friction", "Form friction", body + ctx.checklist(("form",)), major=True)


_TRUST_TILES: tuple[tuple[str, str], ...] = (
    ("testimonials", "Testimonials"),
    ("named_testimonials", "Named testimonials"),
    ("logos", "Customer logos"),
    ("ratings", "Ratings"),
    ("metrics", "Proof metrics"),
    ("faq", "FAQ"),
    ("risk_reversal", "Risk reversal"),
    ("security_assurance", "Security assurance"),
    ("contact", "Contact"),
)


def _trust(facts: Row | None, ctx: _Ctx) -> str:
    trust = as_dict(facts.get("trust")) if facts else {}
    if not trust:
        tiles = not_measured("trust-signal inventory not in this scan")
    else:
        tiles = (
            '<div class="grid">'
            + "".join(
                tile(
                    label,
                    trust.get(k),
                    detail=txt(trust.get(k))
                    if isinstance(trust.get(k), int) and not isinstance(trust.get(k), bool)
                    else "",
                )
                for k, label in _TRUST_TILES
            )
            + "</div>"
        )
    return section(
        "trust", "Trust & objections", tiles + ctx.checklist(("social", "objection")), major=True
    )


def _tech(ctx: _Ctx) -> str:
    body = ctx.checklist((_TECH,)) + not_measured(
        "Core Web Vitals on this path — see the SEO audit's Performance section"
    )
    return section("tech-friction", "Technical friction", body)


def _drilldown(subs: Mapping[str, Row], ctx: _Ctx) -> str:
    names = list(subs) + [
        str(f.get(_KEY)) for f in ctx.findings if f.get(_KEY) and str(f.get(_KEY)) not in subs
    ]
    other = unbucketed_findings(ctx.findings, key=_KEY, grounding=ctx.grounding)
    if not names and not other:
        return section(
            "drilldown", "Section drilldown", not_measured("no sections in this scan"), major=True
        )
    parts: list[str] = []
    for n in dict.fromkeys(names):
        s = subs.get(n)
        counts = check_counts(ctx.checks, key=_KEY, bucket=n)
        count_line = (
            f"<p class='muted'>weight {weight_text(s.get('weight')) if s else '—'} · checks run {counts[0]}, "
            f"passed {counts[1]}, failed {counts[2]}</p>"
            if counts
            else f"<p class='muted'>weight {weight_text(s.get('weight')) if s else '—'} · check inventory not reported</p>"
        )
        rows = [f for f in ctx.findings if str(f.get(_KEY) or "") == n]
        if rows:
            detail = findings_table(rows, ctx.grounding)
        elif counts and counts[0]:
            detail = all_passed(counts[1])
        elif counts is not None:
            detail = not_measured("no checks of this kind ran on this page")
        else:
            detail = not_measured(
                "no failing checks reported, and this scan has no check inventory"
            )
        parts.append(
            f"<h3>{escape(n)}</h3>{score_row(n, _sec_score(s), note=str((s or {}).get('note') or ''))}{count_line}{detail}"
        )
    return section("drilldown", "Section drilldown", "".join(parts) + other, major=True)


def _experiments(scan: Scan, ctx: _Ctx) -> str:
    buckets = dict_rows(scan.get("quick_wins")) + dict_rows(scan.get("strategic"))
    ids = {str(b.get("check_id") or "") for b in buckets}
    rows = [f for f in ctx.findings if str(f.get("check_id") or "") in ids] if ids else []
    if not rows:
        rows = buckets
    if not rows:
        return section(
            "experiments",
            "Prioritised experiments",
            not_measured("no prioritised findings in this scan"),
            major=True,
        )
    body: list[str] = []
    for f in rows:
        exp = as_dict(f.get("experiment"))
        hypothesis = (
            txt(exp.get("hypothesis")) or '<span class="muted">hypothesis: not authored</span>'
        )
        variant = (
            txt(as_dict(f.get("recommendation")).get("example")) or txt(exp.get("variant")) or "—"
        )
        body.append(
            "<tr>"
            f"<td>{rank_badge(f)}</td><td>{hypothesis}</td>"
            f"<td><strong>{txt(f.get('title'))}</strong><br>{deterministic_fix(f) or '—'}</td>"
            f"<td>{txt(exp.get('primary_metric')) or '—'}</td><td>{txt(f.get('effort')) or '—'}</td>"
            f"<td>{variant}</td></tr>"
        )
    table = (
        '<table class="experiments"><thead><tr><th>Rank</th><th>Hypothesis</th><th>Change</th>'
        f"<th>Metric</th><th>Effort</th><th>Variant B</th></tr></thead><tbody>{''.join(body)}</tbody></table>"
    )
    return section("experiments", "Prioritised experiments", table, major=True)


# (judgment element, the label, where the actual page copy lives in ``facts``)
_JUDGED: tuple[tuple[str, str, tuple[str, str]], ...] = (
    ("headline", "Headline", ("hero", "h1_text")),
    ("value_prop", "Value proposition", ("value_prop", "lead_text")),
    ("testimonial", "Testimonial", ("trust", "testimonial_text")),
)


def _ai_section(scan: Scan, facts: Row | None, state: AiState) -> str:
    if not show_ai_section(scan, state):
        return ""
    judgments = {str(j.get("element") or ""): j for j in dict_rows(scan.get("llm_judgments"))}
    cards: list[str] = []
    for element, label, (fact_group, fact_key) in _JUDGED:
        actual = str(as_dict(facts.get(fact_group)).get(fact_key) or "") if facts else ""
        j = judgments.pop(element, None)
        if j is None:
            problem = state.layer_problem("micro_judge")
            body = (
                ai_unavailable(problem)
                if problem
                else f"<p class='muted'>No {escape(label.lower())} on page to judge.</p>"
            )
        else:
            body = (
                f"<p>Score {dial(j.get('score'))} {txt(j.get('assessment'))}</p>"
                '<div class="two">'
                f"<div><h3>On the page</h3><blockquote>{escape(actual) or '<span class=muted>not captured</span>'}</blockquote></div>"
                f"<div><h3>Suggested</h3><blockquote>{txt(j.get('suggestion')) or '—'}</blockquote></div></div>"
            )
        cards.append(
            f'<div class="card" data-card="judge-{escape(element)}"><h3>{escape(label)}</h3>{body}</div>'
        )
    for element, j in judgments.items():  # any element the judge added beyond the three
        cards.append(
            f'<div class="card" data-card="judge-{escape(element)}"><h3>{escape(element)}</h3>'
            f"<p>Score {dial(j.get('score'))} {txt(j.get('assessment'))}</p>"
            f"<blockquote>{txt(j.get('suggestion')) or '—'}</blockquote></div>"
        )
    return section(
        "ai-analysis",
        "AI analysis — copy before / after",
        "".join(cards) + critique_card(scan, state),
        major=True,
    )


class _Ctx:
    def __init__(self, scan: Scan) -> None:
        self.findings = [
            f
            for f in findings_of(scan)
            if not str(f.get("check_id") or "").startswith(("seo-", "mkt-"))
        ]
        self.checks = checks_of(scan)
        self.grounding = grounding_map(scan)

    def checklist(self, sections: tuple[str, ...]) -> str:
        return checklist(self.checks, key=_KEY, buckets=sections, findings=self.findings)


def render(scan: Scan, tok: Mapping[str, str]) -> str:
    """Render a CRO audit scan to a self-contained HTML document."""
    state = ai_state(scan)
    facts = _facts(scan)
    subs = _subsections(scan)
    stages = _stage_scores(subs, facts)
    ctx = _Ctx(scan)
    exec_html = exec_summary_block(scan, ctx.findings)
    proj = projection_block(scan)
    body = "".join(
        [
            header_band(title=TITLE, question=QUESTION, scan=scan),
            ai_status_line(state),
            section("exec-summary", "Conversion story in brief", exec_html) if exec_html else "",
            _cover(scan, facts, stages),
            _journey(subs, facts, stages),
            _above_fold(facts, ctx),
            _ctas(facts, ctx),
            _form(facts, ctx),
            _trust(facts, ctx),
            _tech(ctx),
            _drilldown(subs, ctx),
            _experiments(scan, ctx),
            section("projection", "Projected lift", proj) if proj else "",
            _ai_section(scan, facts, state),
            methodology(scan, rows=dict_rows(scan.get("subsections")), key=_KEY),
        ]
    )
    return document(title=TITLE, tok=tok, accent=TYPE_ACCENTS["cro_audit"], body=body)
