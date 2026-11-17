"""Backfill: declare ``signs_into`` on every existing single-sign-on login, once.

#1163 made single sign-on EXPLICIT: a login signs into its own sign-in page's origin
unless it declares ``signs_into``. Logins saved before that declare nothing, so a
project like ``01a174a6`` (sign-in page on ``auth.nonagentic.ai``, app on
``commonpractice-app.nonagentic.ai``) would fail capture until its owner re-saved the
login — and declaring the site is a login change that clears the saved password.
Adding a field must not cost the tenant anything (George, 2026-10-28).

So this records, ONCE, as data: for each SINGLE login block (``auth_info`` /
``auth_config`` that is a dict, at any depth — ``sources[].auth_info`` included) with
no ``origin`` and no ``signs_into``, whose ``login_url`` origin differs from the ONE
site its enclosing dict binds it to (``target_url`` / ``source_url`` / ``url`` /
``ref``, all on one https origin), it sets ``signs_into`` to that site's origin.
That is exactly the site such a login was entered beside. Nothing is inferred at
runtime after this; a login list (one login per origin, keyed by its own) is left
alone, and an ambiguous or absent site is left alone.

The sealed password is NOT touched — no secret field is read or written — so the
backfilled login keeps it. Only a later user change to ``signs_into`` (or any other
binding field) clears it, through ``auth_secrets.restore_redacted``. Both
``projects.brief`` and ``runs.config_snapshot`` are backfilled, so a paused run
resumed from its own snapshot signs in too. Re-runnable: a backfilled login now
declares ``signs_into`` and is skipped. Only counts are logged, never a value.

The origin parser is a FROZEN copy of ``auth_secrets._strict_host_port`` (this repo's
migration-freeze convention). RLS on these tables is ENABLEd, not FORCEd, and alembic
connects as the owner, so every tenant's rows are visible.

Chained behind 0105 (#1164), which follows 0104 (#1165) and 0103 (#1160). It has
no schema dependency on any of them.

Revision ID: 0106_backfill_login_signs_into
Revises: 0105_run_usage_mirror_unique
Create Date: 2026-10-28
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from alembic import op
from sqlalchemy import text

revision = "0106_backfill_login_signs_into"
down_revision = "0105_run_usage_mirror_unique"
branch_labels = None
depends_on = None

log = logging.getLogger("alembic.runtime.migration")

_AUTH_BLOCK_KEYS = frozenset({"auth_config", "auth_info"})
_BOUND_URL_KEYS = ("target_url", "source_url", "url", "ref")
_LDH_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
_NUMERIC_LABEL = re.compile(r"(?:0x[0-9a-f]*|[0-9]+)")
_BATCH = 500
#: ``(table, json column)``. Rows are selected by a jsonb path filter (only documents
#: that carry a ``login_url`` key at any depth), in id-keyset batches of ``_BATCH``,
#: so the migration never loads every matching row into memory at once.
_TARGETS = (("projects", "brief"), ("runs", "config_snapshot"))


def _origin(raw: object) -> str | None:
    """``https://host[:port]`` of a strict https URL, else None (frozen copy)."""
    if not isinstance(raw, str) or not raw.startswith("https://") or "\\" in raw:
        return None
    if any(not (0x21 <= ord(c) <= 0x7E) for c in raw):
        return None
    authority = re.split(r"[/?#]", raw[len("https://") :], maxsplit=1)[0]
    if "@" in authority or "%" in authority:
        return None
    host, sep, port_text = authority.partition(":")
    port = 443
    if sep:
        if not port_text.isdigit() or not 0 < int(port_text) <= 65535:
            return None
        port = int(port_text)
    host = host.lower()
    labels = host.split(".")
    if len(host) > 253 or not all(_LDH_LABEL.fullmatch(label) for label in labels):
        return None
    if _NUMERIC_LABEL.fullmatch(labels[-1]):
        return None
    return f"https://{host}" if port == 443 else f"https://{host}:{port}"


def _bound_site(context: dict) -> str | None:
    origins = {_origin(context[k]) for k in _BOUND_URL_KEYS if context.get(k)}
    return origins.pop() if len(origins) == 1 else None


def _backfill(doc: Any) -> int:
    """Declare ``signs_into`` on qualifying single logins IN PLACE; return how many."""
    if isinstance(doc, list):
        return sum(_backfill(item) for item in doc)
    if not isinstance(doc, dict):
        return 0
    count = 0
    for key, value in doc.items():
        if key in _AUTH_BLOCK_KEYS:
            if (
                isinstance(value, dict)
                and not value.get("origin")
                and not value.get("signs_into")
                and (login := _origin(value.get("login_url"))) is not None
                and (site := _bound_site(doc)) is not None
                and login != site
            ):
                value["signs_into"] = site
                count += 1
        else:
            count += _backfill(value)
    return count


def backfill_rows(bind: Any) -> dict[str, int]:
    """Backfill every affected row; return ``{table: logins_declared}``. Re-runnable."""
    counts: dict[str, int] = {}
    for table, column in _TARGETS:
        select_sql = text(
            f"SELECT id, {column} FROM {table} "  # noqa: S608 -- constants, not input
            f"WHERE jsonb_path_exists({column}, '$.**.login_url') "
            "AND (CAST(:after AS uuid) IS NULL OR id > CAST(:after AS uuid)) "
            "ORDER BY id LIMIT :batch"
        )
        update_sql = text(
            f"UPDATE {table} SET {column} = CAST(:doc AS jsonb) WHERE id = :id"  # noqa: S608
        )
        declared = 0
        after = None
        while rows := bind.execute(select_sql, {"after": after, "batch": _BATCH}).all():
            for row_id, doc in rows:
                if isinstance(doc, dict) and (n := _backfill(doc)):
                    bind.execute(update_sql, {"doc": json.dumps(doc), "id": row_id})
                    declared += n
            after = rows[-1][0]
        counts[table] = declared
    return counts


def upgrade() -> None:
    counts = backfill_rows(op.get_bind())
    log.info("0106: declared signs_into on existing single-sign-on logins: %s", counts)


def downgrade() -> None:
    # Forward-only (pre-prod rule). Removing the declaration would re-break capture.
    pass
