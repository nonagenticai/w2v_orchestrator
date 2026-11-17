"""Backfill: blank every plaintext login secret stored in a brief or run snapshot

Measured on the primary 2026-10-02, db ``w2v_orchestrator``: 31 ``projects.brief``
and 31 ``runs.config_snapshot`` rows held a tenant's login password in plaintext —
29 promo projects under ``brief.auth_info.password`` (June, one tenant) and the two
showcase-spike projects (``01a0fc53``, ``01a0fc54``) plus their runs (``01a0fc54-39cc``,
``01a0fc55``) under ``brief.auth_config.password``. Session cookies are credentials
too (``cookie_header`` / ``cookies``, replayed by the recorder) — 0 stored rows carry
one today (counted 2026-10-02), but the same fields are blanked. ``GET /api/projects/{id}`` and
``GET /api/runs/{id}`` returned them verbatim.

From this release new secrets are sealed before they are written
(``src/services/auth_secrets.py`` + the mapper listeners in ``models.py``). The rows
already written are not re-encrypted here — the migration has no business holding the
key, and a credential that sat readable must be ROTATED, not preserved. So every
plaintext secret field under an ``auth_config`` / ``auth_info`` block (dict or list,
at any depth) is REMOVED. A run re-recorded from such a project needs its login
supplied again; that is the intended cost.

Sealed values (``{"$sealed": ...}``) are left alone, so a second application changes
0 rows. RLS on these tables is ENABLEd, not FORCEd, and alembic connects as the owner,
so every tenant's rows are visible without touching ``row_security``. The walker is a
FROZEN copy of ``auth_secrets``' (this repo's migration-freeze convention).

Revision ID: 0069_redact_brief_login_secrets
Revises: 0068_sweep_stranded_stages
Create Date: 2026-10-02
"""

from __future__ import annotations

import json
import logging
from typing import Any

from alembic import op
from sqlalchemy import text

revision = "0069_redact_brief_login_secrets"
down_revision = "0068_sweep_stranded_stages"
branch_labels = None
depends_on = None

log = logging.getLogger("alembic.runtime.migration")

_AUTH_BLOCK_KEYS = frozenset({"auth_config", "auth_info"})
_SECRET_FIELDS = frozenset(
    {
        "password",
        "token",
        "access_token",
        "refresh_token",
        "client_secret",
        "api_key",
        "secret",
        "authorization",
        "headers",
        "cookie_header",
        "cookies",
        "cookie",
    }
)
# Cheap pre-filter; the walker decides. Literal statements per table (no f-string SQL).
_CANDIDATE = (
    '"(password|token|access_token|refresh_token|client_secret|api_key|secret'
    '|authorization|headers|cookie_header|cookies|cookie)"'
)
_TARGETS = (
    (
        "projects",
        "SELECT id, brief FROM projects WHERE brief::text ~ :pat",
        "UPDATE projects SET brief = CAST(:doc AS jsonb) WHERE id = :id",
    ),
    (
        "runs",
        "SELECT id, config_snapshot FROM runs WHERE config_snapshot::text ~ :pat",
        "UPDATE runs SET config_snapshot = CAST(:doc AS jsonb) WHERE id = :id",
    ),
)


def _is_plaintext(value: object) -> bool:
    """Any non-empty value that is not already sealed — strings AND cookie lists."""
    sealed = isinstance(value, dict) and isinstance(value.get("$sealed"), str)
    return value not in (None, "", [], {}) and not sealed


def _blank_entry(entry: object) -> bool:
    """Drop the plaintext secret fields of ONE login dict; True if any was dropped."""
    if not isinstance(entry, dict):
        return False
    plain = [f for f in _SECRET_FIELDS & entry.keys() if _is_plaintext(entry[f])]
    for field in plain:
        del entry[field]
    return bool(plain)


def _blank(doc: Any) -> bool:
    """Remove plaintext secrets under auth blocks IN PLACE; True if anything changed."""
    if isinstance(doc, list):
        return any([_blank(item) for item in doc])  # list, not generator: visit them all
    if not isinstance(doc, dict):
        return False
    changed = False
    for key, value in doc.items():
        if key in _AUTH_BLOCK_KEYS:
            entries = value if isinstance(value, list) else [value]
            changed = any([_blank_entry(e) for e in entries]) or changed
        else:
            changed = _blank(value) or changed
    return changed


def blank_rows(bind: Any) -> dict[str, int]:
    """Blank every affected row; return ``{table: rows_changed}``. Re-runnable."""
    counts: dict[str, int] = {}
    for table, select_sql, update_sql in _TARGETS:
        rows = bind.execute(text(select_sql), {"pat": _CANDIDATE}).all()
        changed = 0
        for row_id, doc in rows:
            if isinstance(doc, dict) and _blank(doc):
                bind.execute(text(update_sql), {"doc": json.dumps(doc), "id": row_id})
                changed += 1
        counts[table] = changed
    return counts


def upgrade() -> None:
    counts = blank_rows(op.get_bind())
    log.info("0069: blanked plaintext login secrets in %s", counts)


def downgrade() -> None:
    # Forward-only: a removed plaintext secret is not recoverable, by design.
    pass
