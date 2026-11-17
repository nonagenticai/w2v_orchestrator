"""Re-key ONE Studio user from one Keycloak ``sub`` to another, in one transaction.

Why: realm-per-app cutover. Studio users move from Keycloak realm ``portal`` to
realm ``w2v``. A user who MOVES keeps their id and needs nothing here. A user who
is COPIED (fr-test, …) gets a NEW Keycloak id in the new realm — and the
orchestrator keys its data on the sub: there is no users table, the sub sits in
plain columns (``tenant_members.subject_id``, ``projects.owner_subject_id``,
``run_usage.subject_id``, …). This script moves all of them. Port of
commonpractice ``backend/scripts/remap-user-sub.ts`` (CP #481) and fr's (#1629).

Run it inside the orchestrator pod with the MIGRATION credentials (the table
owner ``w2v_service`` — ``DATABASE_URL``; never ``RUNTIME_DATABASE_URL``)::

    python -m scripts.remap_user_sub --from <old sub> --to <new sub>            # dry run
    python -m scripts.remap_user_sub --from <old sub> --to <new sub> --apply    # remap
    python -m scripts.remap_user_sub --from <old> --to <new> --resolve-preexisting \\
        --email <addr> [--moved-after <ISO>] [--apply]

WHICH COLUMNS. Every column whose name looks like it could hold a person
(``CANDIDATE_NAME_RE``) is in exactly one of two lists:

* ``SUB_COLUMNS`` — hold a sub; rewritten. ``uuid`` columns match by value,
  ``text`` columns match the lowercase sub string exactly (so
  ``signup_approvals.decider_actor = 'telegram:<id>'`` never matches);
* ``NOT_SUB_COLUMNS`` — person-shaped name, value is not a sub.

Before anything else the run asks the LIVE database for every text/uuid/array
column whose name is person-shaped and REFUSES if one is in neither list, or if a
listed column is missing. ``tests/scripts/test_remap_user_sub.py`` fails the same
way in CI against BOTH the models schema and the alembic chain, so a new
sub-bearing column cannot be added without classifying it here.

There is no append-only trigger on any sub-bearing table (unlike fr), so the
audit tables are re-keyed too: they record the same person.

json/jsonb is NOT rewritten (``runs.config_snapshot``, ``request_audit.detail``,
…): the run COUNTS rows whose json text contains ``--from`` and prints them.

RLS. The runtime role is subject to row-level security; under it every count and
the post-check would silently see only one tenant's rows and report success. The
run therefore REFUSES unless the connected role owns every RLS table (and none is
``FORCE``d) or is superuser / ``BYPASSRLS``.

PRE-EXISTING ``--to`` (the user signed in with the new sub before this ran —
``GET /api/auth/me`` upserts a ``tenant_members`` row and every request writes
``request_audit``). Without ``--resolve-preexisting`` the run refuses and prints
what references ``--to``. With it, and only when it is provably a stub of the
SAME PERSON:

* ``--email`` is required, every email on record for ``--from`` AND ``--to``
  (``tenant_members.email``, ``provisioning_audit.target_email``,
  ``tenant_invites.email`` of an accepted invite) equals it case-insensitively,
  and at least one such email exists;
* ``--to`` is referenced ONLY in ``STUB_COLUMNS`` (``tenant_members``,
  ``request_audit``, ``admin_audit_log``), each row created at/after
  ``--moved-after`` (default ``now() - 24h``). Anything else ⇒ refused.

The same transaction then DELETEs the ``--to`` ``tenant_members`` rows that would
collide with a ``--from`` membership (same tenant — the primary key), keeps every
other ``--to`` row, and re-keys as usual. Still a dry run unless ``--apply``.

WHAT ``--apply`` DOES, IN ONE TRANSACTION (no session-level ``SET`` — the
orchestrator may sit behind PgBouncer in transaction pooling):

1. classify + RLS guard + count (as the dry run); refuse on any problem;
2. delete colliding ``--to`` membership stubs (if ``--resolve-preexisting``);
3. UPDATE every ``SUB_COLUMNS`` column from ``--from`` to ``--to``;
4. re-count: any row still on ``--from``, or a ``--to`` count that is not exactly
   (moved + kept), raises ⇒ ROLLBACK.

The dry run runs step 1 inside a ``READ ONLY`` transaction. Idempotent: ``--from``
unreferenced and ``--to`` referenced ⇒ "already remapped", exit 0.

Exit: 0 ok · 1 usage / refused / rolled back (nothing written).
"""

from __future__ import annotations

import argparse
import datetime
import os
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

from sqlalchemy import Connection, create_engine, text


class Col(NamedTuple):
    table: str
    column: str

    @property
    def key(self) -> str:
        return f"{self.table}.{self.column}"


#: Columns holding a Keycloak sub — rewritten by ``--apply``.
SUB_COLUMNS: tuple[Col, ...] = (
    Col("admin_audit_log", "actor_subject_id"),
    Col("projects", "owner_subject_id"),
    Col("provisioning_audit", "actor_subject_id"),
    Col("provisioning_audit", "target_subject_id"),
    Col("request_audit", "actor_subject_id"),
    Col("run_usage", "subject_id"),
    Col("service_runs", "subject_id"),
    Col("signup_approvals", "decided_by"),  # text: str(sub) of the super_admin decider
    Col("signup_approvals", "decider_actor"),  # text: sub | 'telegram:<id>' | 'system:…'
    Col("storyboard_branches", "actor_subject"),  # text: str(sub)
    Col("storyboard_panel_events", "actor_subject_id"),
    Col("storyboard_panels", "approved_by"),  # text: str(sub)
    Col("storyboard_snapshots", "actor_subject"),  # text: str(sub)
    Col("tenant_api_keys", "created_by"),
    Col("tenant_connection_versions", "created_by"),
    Col("tenant_connections", "created_by"),
    Col("tenant_connections", "updated_by"),
    Col("tenant_defaults", "updated_by"),
    Col("tenant_invites", "accepted_subject_id"),
    Col("tenant_invites", "invited_by"),
    Col("tenant_members", "invited_by"),
    Col("tenant_members", "subject_id"),
    Col("tenant_offboarding", "decided_by"),
    Col("tenant_offboarding", "requested_by"),
    Col("tenant_sound_assets", "rights_attested_by"),
)

#: Person-shaped names whose values are NOT a Keycloak sub. ``table.column`` → why.
NOT_SUB_COLUMNS: dict[str, str] = {
    "request_audit.claim_tenant_id": "the tenant claimed by the token, not a person",
    "signup_approvals.approving_claim": "random lease token of an in-flight approve",
    "source_ingest_cache.claimed_by": "orchestrator instance id holding the ingest lease",
}

#: Postgres regex over column names: what "looks like it could hold a person".
CANDIDATE_NAME_RE = (
    "(^|_)(user|users|owner|actor|assignee|approver|requester|author|creator|reviewer|"
    "decider|recipient|member|subject|sub|principal|grantee|holder|initiator|operator|"
    "invitee|inviter|caller|account|person|identity)(_|$)|_by($|_)|(^|_)kc_|keycloak|claim"
)


@dataclass(frozen=True, slots=True)
class Stub:
    """Where a ``--to`` stub may live (``--resolve-preexisting``). With ``collide_on``,
    ``--to`` rows that would collide with a ``--from`` row on that column are deleted."""

    col: Col
    created_at: str = "created_at"
    collide_on: str | None = None


STUB_COLUMNS: tuple[Stub, ...] = (
    Stub(Col("tenant_members", "subject_id"), collide_on="tenant_id"),
    Stub(Col("request_audit", "actor_subject_id")),
    Stub(Col("admin_audit_log", "actor_subject_id")),
)

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+$")


class RemapRefusedError(Exception):
    """Refused before writing anything (or rolled back)."""


@dataclass(frozen=True, slots=True)
class Resolved:
    col: Col
    kind: str  # 'uuid' | 'text'


@dataclass(slots=True)
class Preexisting:
    refs: dict[str, int]
    emails_from: list[str]
    emails_to: list[str]
    moved_after: str
    collisions: int
    blockers: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RemapPlan:
    state: str  # 'ready' | 'already_remapped'
    from_counts: dict[str, int]
    total: int
    json: dict[str, int]
    preexisting: Preexisting | None = None


@dataclass(frozen=True, slots=True)
class Options:
    resolve_preexisting: bool = False
    email: str | None = None
    moved_after: str | None = None


def _ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _schema(conn: Connection) -> str:
    return str(conn.execute(text("SELECT current_schema()")).scalar_one())


def _qual(schema: str, table: str) -> str:
    return f"{_ident(schema)}.{_ident(table)}"


def candidate_columns(conn: Connection, schema: str | None = None) -> list[str]:
    """Every person-shaped text/uuid/array column of a base table in ``schema``."""
    schema = schema or _schema(conn)
    rows = conn.execute(
        text(
            """
            SELECT c.table_name || '.' || c.column_name
              FROM information_schema.columns c
              JOIN information_schema.tables t
                ON t.table_schema = c.table_schema AND t.table_name = c.table_name
             WHERE c.table_schema = :schema AND t.table_type = 'BASE TABLE'
               AND c.udt_name IN ('text', 'uuid', 'varchar', 'bpchar', '_uuid', '_text')
               AND c.column_name ~ :re
             ORDER BY 1
            """
        ),
        {"schema": schema, "re": CANDIDATE_NAME_RE},
    )
    return [str(r[0]) for r in rows]


def unclassified_columns(conn: Connection, schema: str | None = None) -> list[str]:
    """Person-shaped columns in the live schema that neither list classifies."""
    listed = {c.key for c in SUB_COLUMNS} | set(NOT_SUB_COLUMNS)
    return [k for k in candidate_columns(conn, schema) if k not in listed]


def resolve_columns(conn: Connection, schema: str | None = None) -> list[Resolved]:
    """``SUB_COLUMNS`` with their live type; refuses if one is missing or unhandled."""
    schema = schema or _schema(conn)
    out: list[Resolved] = []
    for c in SUB_COLUMNS:
        udt = conn.execute(
            text(
                "SELECT udt_name::text FROM information_schema.columns "
                "WHERE table_schema = :s AND table_name = :t AND column_name = :c"
            ),
            {"s": schema, "t": c.table, "c": c.column},
        ).scalar_one_or_none()
        if udt is None:
            raise RemapRefusedError(
                f"expected column {c.key} is missing — update scripts/remap_user_sub.py"
            )
        if udt == "uuid":
            out.append(Resolved(c, "uuid"))
        elif udt in ("text", "varchar"):
            out.append(Resolved(c, "text"))
        else:
            raise RemapRefusedError(f"column {c.key} has unhandled type {udt}")
    return out


def rls_blind_tables(conn: Connection, schema: str | None = None) -> list[str]:
    """RLS tables whose rows the CONNECTED role may not all see (counts would lie)."""
    schema = schema or _schema(conn)
    privileged = conn.execute(
        text("SELECT rolsuper OR rolbypassrls FROM pg_roles WHERE rolname = current_user")
    ).scalar_one()
    if privileged:
        return []
    rows = conn.execute(
        text(
            """
            SELECT c.relname
              FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
             WHERE n.nspname = :schema AND c.relkind IN ('r', 'p') AND c.relrowsecurity
               AND (c.relforcerowsecurity
                    OR NOT pg_has_role(current_user, c.relowner, 'USAGE'))
             ORDER BY 1
            """
        ),
        {"schema": schema},
    )
    return [str(r[0]) for r in rows]


def _match(r: Resolved) -> str:
    col = _ident(r.col.column)
    return f"{col} = CAST(:sub AS uuid)" if r.kind == "uuid" else f"lower({col}) = :sub"


def _count(conn: Connection, schema: str, r: Resolved, sub: str) -> int:
    sql = f"SELECT count(*) FROM {_qual(schema, r.col.table)} WHERE {_match(r)}"  # noqa: S608 — identifiers quoted, value bound
    return int(conn.execute(text(sql), {"sub": sub}).scalar_one())


def _counts(conn: Connection, schema: str, cols: Sequence[Resolved], sub: str) -> dict[str, int]:
    return {r.col.key: _count(conn, schema, r, sub) for r in cols}


def json_mentions(conn: Connection, sub: str, schema: str | None = None) -> dict[str, int]:
    """json/jsonb columns whose text contains ``sub`` (reported, never rewritten)."""
    schema = schema or _schema(conn)
    cols = conn.execute(
        text(
            """
            SELECT c.table_name::text, c.column_name::text
              FROM information_schema.columns c
              JOIN information_schema.tables t
                ON t.table_schema = c.table_schema AND t.table_name = c.table_name
             WHERE c.table_schema = :schema AND t.table_type = 'BASE TABLE'
               AND c.udt_name IN ('json', 'jsonb')
             ORDER BY 1, 2
            """
        ),
        {"schema": schema},
    ).all()
    out: dict[str, int] = {}
    for table, column in cols:
        sql = (
            f"SELECT count(*) FROM {_qual(schema, table)} "  # noqa: S608 — quoted identifiers
            f"WHERE strpos(lower({_ident(column)}::text), :sub) > 0"
        )
        n = int(conn.execute(text(sql), {"sub": sub}).scalar_one())
        if n:
            out[f"{table}.{column}"] = n
    return out


def emails_on_record(conn: Connection, sub: str, schema: str | None = None) -> list[str]:
    """Every email the orchestrator has on record for a sub (lowercased, distinct)."""
    schema = schema or _schema(conn)
    tm = _qual(schema, "tenant_members")
    pa = _qual(schema, "provisioning_audit")
    ti = _qual(schema, "tenant_invites")
    sql = f"""
        SELECT DISTINCT e FROM (
          SELECT lower(email) AS e FROM {tm} WHERE subject_id = CAST(:sub AS uuid)
          UNION ALL
          SELECT lower(target_email) FROM {pa} WHERE target_subject_id = CAST(:sub AS uuid)
          UNION ALL
          SELECT lower(email) FROM {ti} WHERE accepted_subject_id = CAST(:sub AS uuid)
        ) x WHERE e IS NOT NULL AND e <> '' ORDER BY 1
    """  # noqa: S608 — quoted identifiers, value bound
    return [str(r[0]) for r in conn.execute(text(sql), {"sub": sub})]


def _collisions(schema: str, stub: Stub) -> str:
    """SQL (params :from_, :to) selecting --to stub rows that collide with a --from row."""
    assert stub.collide_on is not None  # noqa: S101 — internal invariant
    t = _qual(schema, stub.col.table)
    sub, on = _ident(stub.col.column), _ident(stub.collide_on)
    return (
        f"FROM {t} x WHERE x.{sub} = CAST(:to AS uuid) AND EXISTS ("  # noqa: S608
        f"SELECT 1 FROM {t} y WHERE y.{sub} = CAST(:from_ AS uuid) AND y.{on} = x.{on})"
    )


def _inspect_preexisting(
    conn: Connection, schema: str, from_: str, to: str, refs: dict[str, int], o: Options
) -> Preexisting:
    stub_keys = {s.col.key for s in STUB_COLUMNS}
    blockers: list[str] = []
    other = {k: n for k, n in refs.items() if k not in stub_keys}
    if other:
        blockers.append(f"--to is referenced outside the stub tables: {_fmt(other)}")
    cutoff = conn.execute(
        text(
            "SELECT to_char(COALESCE(CAST(:m AS timestamptz), now() - interval '24 hours') "
            "AT TIME ZONE 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SS\"Z\"')"
        ),
        {"m": o.moved_after},
    ).scalar_one()
    collisions = 0
    for s in STUB_COLUMNS:
        if refs.get(s.col.key, 0) == 0:
            continue
        sql = (
            f"SELECT count(*) FROM {_qual(schema, s.col.table)} "  # noqa: S608
            f"WHERE {_ident(s.col.column)} = CAST(:to AS uuid) "
            f"AND {_ident(s.created_at)} < CAST(:cut AS timestamptz)"
        )
        old = int(conn.execute(text(sql), {"to": to, "cut": cutoff}).scalar_one())
        if old:
            blockers.append(
                f"{s.col.key}: {old} row(s) for --to created before --moved-after {cutoff}"
            )
        if s.collide_on:
            q = f"SELECT count(*) {_collisions(schema, s)}"
            collisions += int(conn.execute(text(q), {"to": to, "from_": from_}).scalar_one())
    emails_from = emails_on_record(conn, from_, schema)
    emails_to = emails_on_record(conn, to, schema)
    want = (o.email or "").strip().lower()
    if not want:
        blockers.append("--email <addr> is required with --resolve-preexisting")
    else:
        found = emails_from + emails_to
        if not found:
            blockers.append(
                "no email on record for --from or --to — cannot prove they are one person"
            )
        wrong = sorted({e for e in found if e != want})
        if wrong:
            blockers.append(
                f"email(s) on record differ from --email: {', '.join(wrong)} — a DIFFERENT person"
            )
    return Preexisting(refs, emails_from, emails_to, str(cutoff), collisions, blockers)


def preexisting_message(from_: str, to: str, p: Preexisting) -> str:
    head = (
        f"--to {to} is already referenced ({_fmt(p.refs)}) — the user used Studio with the "
        "new sub before this ran."
    )
    if p.blockers:
        return f"{head} NOT a removable stub: {'; '.join(p.blockers)}. Merge by hand, then re-run."
    return (
        f"{head} Plan (one transaction): 1) DELETE {p.collisions} --to tenant_members row(s) "
        f"colliding with a --from membership (all --to rows created after {p.moved_after}, "
        f"same email; other --to rows are kept); 2) re-key {from_} → {to}. "
        "Run with --resolve-preexisting for this dry run, add --apply to do it."
    )


def _fmt(counts: dict[str, int]) -> str:
    return ", ".join(f"{k}={n}" for k, n in counts.items())


def _preflight(
    conn: Connection, from_: str, to: str, o: Options
) -> tuple[RemapPlan, str, list[Resolved], dict[str, int]]:
    schema = _schema(conn)
    unclassified = unclassified_columns(conn, schema)
    if unclassified:
        raise RemapRefusedError(
            f"unclassified person-shaped column(s): {', '.join(unclassified)} — add each to "
            "SUB_COLUMNS or NOT_SUB_COLUMNS in scripts/remap_user_sub.py"
        )
    blind = rls_blind_tables(conn, schema)
    if blind:
        raise RemapRefusedError(
            f"connected role is subject to row-level security on {', '.join(blind)} — counts "
            "would silently cover one tenant. Connect as the table owner (DATABASE_URL)."
        )
    cols = resolve_columns(conn, schema)
    from_counts = _counts(conn, schema, cols, from_)
    to_counts = _counts(conn, schema, cols, to)
    to_refs = {k: n for k, n in to_counts.items() if n}
    total = sum(from_counts.values())
    js = json_mentions(conn, from_, schema)
    if total == 0 and to_refs:
        return RemapPlan("already_remapped", from_counts, 0, js), schema, cols, to_counts
    if total == 0:
        raise RemapRefusedError(
            f"nothing references --from {from_} (and nothing --to) — wrong sub?"
        )
    pre: Preexisting | None = None
    if to_refs:
        pre = _inspect_preexisting(conn, schema, from_, to, to_refs, o)
        if not o.resolve_preexisting or pre.blockers:
            raise RemapRefusedError(preexisting_message(from_, to, pre))
    return RemapPlan("ready", from_counts, total, js, pre), schema, cols, to_counts


def plan_remap(conn: Connection, from_: str, to: str, o: Options | None = None) -> RemapPlan:
    """Dry run: per-column counts. Writes nothing (the CLI also makes the tx READ ONLY)."""
    return _preflight(conn, from_, to, o or Options())[0]


def apply_remap(conn: Connection, from_: str, to: str, o: Options | None = None) -> RemapPlan:
    """The whole re-key inside the CALLER's transaction; any raise must roll it back."""
    plan, schema, cols, to_before = _preflight(conn, from_, to, o or Options())
    if plan.state == "already_remapped":
        return plan
    deleted: dict[str, int] = {}
    if plan.preexisting:
        for s in STUB_COLUMNS:
            if s.collide_on and to_before.get(s.col.key, 0):
                sql = (
                    f"DELETE FROM {_qual(schema, s.col.table)} t USING (SELECT x.ctid AS c "  # noqa: S608
                    f"{_collisions(schema, s)}) d WHERE t.ctid = d.c"
                )
                res = conn.execute(text(sql), {"to": to, "from_": from_})
                deleted[s.col.key] = int(res.rowcount)
    for r in cols:
        if not plan.from_counts[r.col.key]:
            continue
        col = _ident(r.col.column)
        new = "CAST(:to AS uuid)" if r.kind == "uuid" else ":to"
        sql = f"UPDATE {_qual(schema, r.col.table)} SET {col} = {new} WHERE {_match(r)}"  # noqa: S608
        conn.execute(text(sql), {"sub": from_, "to": to})
    left = {k: n for k, n in _counts(conn, schema, cols, from_).items() if n}
    if left:
        raise RemapRefusedError(f"post-check: --from still referenced ({_fmt(left)}) — rolled back")
    after = _counts(conn, schema, cols, to)
    wrong = {
        k: n
        for k, n in after.items()
        if n != plan.from_counts[k] + to_before.get(k, 0) - deleted.get(k, 0)
    }
    if wrong:
        raise RemapRefusedError(f"post-check: --to row counts off ({_fmt(wrong)}) — rolled back")
    return plan


@dataclass(frozen=True, slots=True)
class Args:
    from_: str
    to: str
    apply: bool
    options: Options


def parse_args(argv: Sequence[str]) -> Args:
    p = argparse.ArgumentParser(prog="python -m scripts.remap_user_sub", exit_on_error=False)
    p.add_argument("--from", dest="from_", required=True)
    p.add_argument("--to", required=True)
    p.add_argument("--apply", action="store_true")
    p.add_argument("--resolve-preexisting", action="store_true")
    p.add_argument("--email")
    p.add_argument("--moved-after")
    try:
        ns = p.parse_args(list(argv))
    except (argparse.ArgumentError, SystemExit) as exc:
        raise RemapRefusedError(f"bad arguments: {exc}") from exc
    from_, to = ns.from_.strip().lower(), ns.to.strip().lower()
    if not _UUID_RE.match(from_) or not _UUID_RE.match(to):
        raise RemapRefusedError("--from and --to must both be UUIDs (Keycloak subs)")
    if from_ == to:
        raise RemapRefusedError("--from and --to are the same")
    moved_after: str | None = None
    if ns.moved_after is not None:
        try:
            dt = datetime.datetime.fromisoformat(ns.moved_after.strip())
        except ValueError as exc:
            raise RemapRefusedError(
                "--moved-after must be an ISO timestamp (the realm-move run time)"
            ) from exc
        if dt.tzinfo is None:
            raise RemapRefusedError("--moved-after needs a timezone (e.g. 2026-10-22T10:00Z)")
        moved_after = dt.astimezone(datetime.UTC).isoformat()
    email: str | None = None
    if ns.email is not None:
        email = ns.email.strip().lower()
        if not _EMAIL_RE.match(email):
            raise RemapRefusedError("--email must be an email address")
    return Args(from_, to, ns.apply, Options(ns.resolve_preexisting, email, moved_after))


def report(from_: str, to: str, plan: RemapPlan, apply: bool) -> list[str]:
    out: list[str] = []
    if plan.state == "already_remapped":
        out.append(f"[remap] {from_} → {to}: already remapped (no column references --from).")
    else:
        if plan.preexisting:
            out.append(
                f"[remap] pre-existing --to: {preexisting_message(from_, to, plan.preexisting)}"
            )
        out.extend(f"[remap] {k}: {n}" for k, n in plan.from_counts.items())
    out.extend(
        f"[remap] json {k}: {n} row(s) mention --from — NOT rewritten" for k, n in plan.json.items()
    )
    if plan.state == "ready":
        tail = "COMMITTED" if apply else "DRY RUN — nothing written. Re-run with --apply."
        out.append(
            f"[remap] {from_} → {to} · {plan.total} row(s) in {len(SUB_COLUMNS)} columns · {tail}"
        )
    return out


def _sync_url(url: str) -> str:
    return re.sub(r"^postgres(ql)?(\+\w+)?://", "postgresql+psycopg://", url)


def run(url: str, args: Args) -> list[str]:
    """Open ONE transaction on ``url`` and dry-run or apply. Raises on refusal."""
    engine = create_engine(_sync_url(url), poolclass=None)
    try:
        with engine.begin() as conn:
            if args.apply:
                plan = apply_remap(conn, args.from_, args.to, args.options)
            else:
                conn.execute(text("SET TRANSACTION READ ONLY"))
                plan = plan_remap(conn, args.from_, args.to, args.options)
    finally:
        engine.dispose()
    return report(args.from_, args.to, plan, args.apply)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(sys.argv[1:] if argv is None else argv)
    except RemapRefusedError as exc:
        print(f"[remap] {exc}", file=sys.stderr)
        print(
            "usage: python -m scripts.remap_user_sub --from <old sub> --to <new sub> "
            "[--resolve-preexisting --email <addr> [--moved-after <ISO>]] [--apply]",
            file=sys.stderr,
        )
        return 1
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        print("[remap] DATABASE_URL (the migration/owner URL) is not set", file=sys.stderr)
        return 1
    try:
        lines = run(url, args)
    except RemapRefusedError as exc:
        print(f"[remap] REFUSED/ROLLED BACK — nothing written: {exc}", file=sys.stderr)
        return 1
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
