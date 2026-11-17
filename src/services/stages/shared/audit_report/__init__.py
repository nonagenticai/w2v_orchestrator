"""Per-type audit report templates (seo_audit / cro_audit / marketing_scorecard).

The on-demand audit-report PDF route turns a run's ``<project_type>_scan`` output
into a self-contained, inline-CSS HTML page that html_render_service prints to a
PDF. Each audit type has its OWN template (design
``reports/ui-review-2026-10-17/audit-templates-design.md`` §2): a different
title, accent, reader question, section set, and signature visual.

The type is chosen **explicitly** from the run's ``project_type`` via
:data:`RENDERERS` — never inferred from which payload keys happen to be present.
An unknown type raises; there is no default.

Older scan outputs (written before the ``facts`` / ``checks`` / ``findings`` /
``ai`` fields) still render: every section whose data is missing shows
"Not measured — <reason>".
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from src.services.stages.shared.audit_report import cro, marketing, seo
from src.services.stages.shared.audit_report.primitives import Scan
from src.services.stages.shared.design_system import DEFAULT_TOKENS, as_params

type Renderer = Callable[[Scan, Mapping[str, str]], str]

# Keyed by project_type; locked to ``audit_family.AUDIT_PROJECT_TYPES`` by test.
RENDERERS: dict[str, Renderer] = {
    "seo_audit": seo.render,
    "cro_audit": cro.render,
    "marketing_scorecard": marketing.render,
}


class UnknownAuditTypeError(ValueError):
    """Raised for a ``project_type`` with no report template."""


def build_report_html(
    project_type: str, scan: Mapping[str, object] | None, *, tokens: Mapping[str, str] | None = None
) -> str:
    """Render ``scan`` with the template for ``project_type``.

    ``tokens`` is the flat C23 design-token map (``design_system.as_params``);
    omitted, the shipped :data:`DEFAULT_TOKENS` brand is used.
    """
    renderer = RENDERERS.get(project_type)
    if renderer is None:
        raise UnknownAuditTypeError(f"no audit report template for project_type {project_type!r}")
    tok = tokens if tokens is not None else as_params(DEFAULT_TOKENS)
    return renderer(scan or {}, tok)


__all__ = ["RENDERERS", "UnknownAuditTypeError", "build_report_html"]
