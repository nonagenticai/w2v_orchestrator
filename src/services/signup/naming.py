"""The workspace an approval founds: its name and the tenant slug.

Name = the company, else the requester's name (``~/.claude/plans/w2v-early-access.md``),
else the email's local part. The slug is the licences registry name and the
``tenants.name`` of the new tenant (same as ``POST /api/admin/provision``'s
``tenant_slug``: ``^[a-z0-9-]{2,40}$``); candidates ``<base>``, ``<base>-2`` …
``<base>-10`` are tried in order (fr ``services/signup/naming.ts``).
"""

from __future__ import annotations

import re
import unicodedata

_NON_SLUG = re.compile(r"[^a-z0-9]+")
#: ≤ 36 characters, leaving room for a ``-10`` suffix inside the 40-char rule.
SLUG_BASE_MAX = 36
SLUG_CANDIDATES = 10


def default_tenant_name(*, email: str, name: str | None, company: str | None) -> str:
    for value in (company, name):
        if value and value.strip():
            return value.strip()
    return email.split("@", 1)[0] or "workspace"


def slugify(name: str) -> str:
    folded = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    slug = _NON_SLUG.sub("-", folded.lower()).strip("-")[:SLUG_BASE_MAX].rstrip("-")
    if len(slug) >= 2:
        return slug
    return f"{slug}-ws" if slug else "workspace"


def slug_candidates(base: str, count: int = SLUG_CANDIDATES) -> list[str]:
    return [base, *(f"{base}-{i}" for i in range(2, count + 1))]
