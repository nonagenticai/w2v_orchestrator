"""``scripts/remap_user_sub.py`` against a REAL Postgres.

* every person-shaped column is classified in BOTH schema authorities — the models
  (``create_all``, what this suite runs on) and the alembic chain (what PROD runs
  on) — so a NEW sub-bearing column added without being listed FAILS here, and a
  column added on the fly is shown to trip the same guard;
* dry run counts per column and writes nothing; ``--apply`` re-keys uuid and text
  columns in one transaction, leaves non-matching text (``telegram:<id>``) alone,
  and a re-run is "already remapped";
* the post-check really rolls back: a trigger that swallows one UPDATE leaves
  ``--from`` referenced, and NOTHING is written;
* a pre-existing ``--to`` refuses; ``--resolve-preexisting`` removes only a
  same-email post-move membership stub;
* the RLS guard refuses a role that row-level security would blind.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import UUID, uuid7

import pytest
from sqlalchemy import Connection, text
from sqlalchemy.exc import ProgrammingError

from scripts import remap_user_sub as rus
from scripts.remap_user_sub import (
    CANDIDATE_NAME_RE,
    NOT_SUB_COLUMNS,
    STUB_COLUMNS,
    SUB_COLUMNS,
    Options,
    RemapRefusedError,
    apply_remap,
    parse_args,
    plan_remap,
    report,
    resolve_columns,
    rls_blind_tables,
    unclassified_columns,
)
from tests.conftest import require_createrole

EMAIL = "fr-test@nonagentic.test"


@pytest.fixture
def subs() -> tuple[str, str, str, UUID]:
    return str(uuid7()), str(uuid7()), str(uuid7()), uuid7()


async def _sync(db_engine: Any, fn: Callable[..., Any], *args: Any) -> Any:
    """Run ``fn(conn, *args)`` in ONE committed transaction (rolled back on raise)."""
    async with db_engine.begin() as conn:
        return await conn.run_sync(lambda c: fn(c, *args))


def _seed(conn: Connection, frm: str, other: str, tenant: UUID) -> UUID:
    t = str(tenant)
    conn.execute(text("INSERT INTO tenants (id) VALUES (:t)"), {"t": t})
    conn.execute(
        text(
            "INSERT INTO tenant_members (subject_id, tenant_id, email, invited_by, roles) "
            "VALUES (:f, :t, :e, :o, '[]'), (:o, :t, 'other@x.test', :f, '[]')"
        ),
        {"f": frm, "o": other, "t": t, "e": EMAIL},
    )
    project = uuid7()
    conn.execute(
        text(
            "INSERT INTO projects (id, tenant_id, owner_subject_id, type, title, visibility, brief, "
            "brand, voice, theme) VALUES (:p, :t, :f, 'generic', 'remap', 'private', "
            "CAST(:b AS jsonb), '{}', '{}', '{}')"
        ),
        {"p": str(project), "t": t, "f": frm, "b": f'{{"by": "{frm}"}}'},
    )
    conn.execute(
        text(
            "INSERT INTO signup_approvals (source, name, email, status, decided_at, decided_by, "
            "decider_actor) VALUES ('web', 'a', 'a@x.test', 'rejected', now(), :f, :f), "
            "('web', 'b', 'b@x.test', 'rejected', now(), NULL, 'telegram:42')"
        ),
        {"f": frm},
    )
    conn.execute(
        text(
            "INSERT INTO request_audit (id, event, method, path, status_code, actor_subject_id, "
            "detail) VALUES (:i, 'mutation', 'POST', '/api/projects', 201, :f, '{}')"
        ),
        {"i": str(uuid7()), "f": frm},
    )
    conn.execute(
        text(
            "INSERT INTO tenant_invites (id, tenant_id, token_hash, email, expires_at, status, "
            "invited_by, accepted_subject_id) VALUES (:i, :t, :h, :e, now() + interval '1 day', "
            "'accepted', :f, :f)"
        ),
        {"i": str(uuid7()), "t": t, "h": b"\x01" * 32, "e": EMAIL, "f": frm},
    )
    conn.execute(
        text("INSERT INTO tenant_defaults (tenant_id, updated_by) VALUES (:t, :f)"),
        {"t": t, "f": frm},
    )
    return project


SEEDED = {
    "tenant_members.subject_id": 1,
    "tenant_members.invited_by": 1,
    "projects.owner_subject_id": 1,
    "signup_approvals.decided_by": 1,
    "signup_approvals.decider_actor": 1,
    "request_audit.actor_subject_id": 1,
    "tenant_invites.invited_by": 1,
    "tenant_invites.accepted_subject_id": 1,
    "tenant_defaults.updated_by": 1,
}


def _refs(conn: Connection, sub: str) -> dict[str, int]:
    schema = str(conn.execute(text("SELECT current_schema()")).scalar_one())
    return {k: n for k, n in rus._counts(conn, schema, resolve_columns(conn), sub).items() if n}


# ─── classification guard ────────────────────────────────────────────────


async def test_every_person_shaped_column_is_classified_in_the_models_schema(db_engine) -> None:
    assert await _sync(db_engine, unclassified_columns) == []


def test_every_person_shaped_column_is_classified_in_the_alembic_schema(parity_schemas) -> None:
    # PROD runs the migration chain, not create_all: a column added only by a
    # migration must fail here too.
    with parity_schemas.engine.connect() as conn:
        assert unclassified_columns(conn, parity_schemas.alembic) == []
        assert {r.col.key for r in resolve_columns(conn, parity_schemas.alembic)} == {
            c.key for c in SUB_COLUMNS
        }


async def test_an_unlisted_sub_column_trips_the_guard_and_refuses_the_run(db_engine, subs) -> None:
    frm, to, other, tenant = subs
    await _sync(db_engine, _seed, frm, other, tenant)
    async with db_engine.connect() as conn:
        tx = await conn.begin()
        try:
            await conn.execute(text("ALTER TABLE projects ADD COLUMN reviewer_subject_id uuid"))
            assert await conn.run_sync(unclassified_columns) == ["projects.reviewer_subject_id"]
            with pytest.raises(RemapRefusedError, match=r"unclassified.*reviewer_subject_id"):
                await conn.run_sync(lambda c: plan_remap(c, frm, to))
        finally:
            await tx.rollback()


def test_lists_are_disjoint_and_not_sub_entries_are_real_candidates() -> None:
    import re

    sub = [c.key for c in SUB_COLUMNS]
    assert len(set(sub)) == len(sub)
    assert not set(sub) & set(NOT_SUB_COLUMNS)
    pat = re.compile(CANDIDATE_NAME_RE)
    for key in [*sub, *NOT_SUB_COLUMNS]:
        assert pat.search(key.split(".")[1]), f"{key} is not matched by CANDIDATE_NAME_RE"
    for s in STUB_COLUMNS:
        assert s.col.key in sub


# ─── args + report ───────────────────────────────────────────────────────


def test_parse_args_validates_subs_email_and_moved_after(subs) -> None:
    frm, to, _, _ = subs
    with pytest.raises(RemapRefusedError, match="UUIDs"):
        parse_args(["--from", "x", "--to", to])
    with pytest.raises(RemapRefusedError, match="same"):
        parse_args(["--from", frm, "--to", frm])
    with pytest.raises(RemapRefusedError, match="moved-after"):
        parse_args(["--from", frm, "--to", to, "--moved-after", "nope"])
    with pytest.raises(RemapRefusedError, match="timezone"):
        parse_args(["--from", frm, "--to", to, "--moved-after", "2026-10-22T10:00"])
    with pytest.raises(RemapRefusedError, match="email"):
        parse_args(["--from", frm, "--to", to, "--email", "nope"])
    with pytest.raises(RemapRefusedError, match="bad arguments"):
        parse_args(["--to", to])
    a = parse_args(
        [
            "--from",
            frm.upper(),
            "--to",
            to,
            "--apply",
            "--resolve-preexisting",
            "--email",
            "A@B.io",
            "--moved-after",
            "2026-10-22T12:00:00+02:00",
        ]
    )
    assert (a.from_, a.to, a.apply) == (frm, to, True)
    assert a.options == Options(True, "a@b.io", "2026-10-22T10:00:00+00:00")
    assert parse_args(["--from", frm, "--to", to]).options == Options()


# ─── dry run + apply ─────────────────────────────────────────────────────


async def test_dry_run_counts_per_column_and_writes_nothing(db_engine, subs) -> None:
    frm, to, other, tenant = subs
    project = await _sync(db_engine, _seed, frm, other, tenant)
    plan = await _sync(db_engine, plan_remap, frm, to)
    assert plan.state == "ready"
    assert {k: n for k, n in plan.from_counts.items() if n} == SEEDED
    assert plan.total == 9
    assert plan.json == {"projects.brief": 1}
    lines = report(frm, to, plan, apply=False)
    assert "DRY RUN" in lines[-1]
    assert any("json projects.brief: 1" in line for line in lines)
    owner = await _sync(
        db_engine,
        lambda c: c.execute(
            text("SELECT owner_subject_id::text FROM projects WHERE id = :p"), {"p": str(project)}
        ).scalar_one(),
    )
    assert owner == frm
    assert await _sync(db_engine, _refs, to) == {}


async def test_apply_rekeys_uuid_and_text_columns_and_rerun_is_already_remapped(
    db_engine, subs
) -> None:
    frm, to, other, tenant = subs
    await _sync(db_engine, _seed, frm, other, tenant)
    plan = await _sync(db_engine, apply_remap, frm, to)
    assert plan.state == "ready"
    assert "COMMITTED" in report(frm, to, plan, apply=True)[-1]
    assert await _sync(db_engine, _refs, frm) == {}
    assert await _sync(db_engine, _refs, to) == SEEDED
    # the OTHER member's invited_by moved; a Telegram decider never matches
    rows = await _sync(
        db_engine,
        lambda c: c.execute(
            text("SELECT subject_id::text, invited_by::text FROM tenant_members ORDER BY 1")
        ).all(),
    )
    assert sorted(map(tuple, rows)) == sorted([(to, other), (other, to)])
    tg = await _sync(
        db_engine,
        lambda c: c.execute(
            text("SELECT count(*) FROM signup_approvals WHERE decider_actor = 'telegram:42'")
        ).scalar_one(),
    )
    assert tg == 1
    again = await _sync(db_engine, apply_remap, frm, to)
    assert again.state == "already_remapped"
    assert "already remapped" in report(frm, to, again, apply=True)[0]


async def test_unknown_from_is_refused(db_engine, subs) -> None:
    frm, to, _, _ = subs
    with pytest.raises(RemapRefusedError, match="wrong sub"):
        await _sync(db_engine, plan_remap, frm, to)


async def test_post_check_failure_rolls_back_everything(db_engine, subs) -> None:
    frm, to, other, tenant = subs
    project = await _sync(db_engine, _seed, frm, other, tenant)
    fn = f"remap_swallow_{uuid7().hex[:10]}"
    await _sync(
        db_engine,
        lambda c: c.execute(
            text(
                f"CREATE FUNCTION {fn}() RETURNS trigger LANGUAGE plpgsql AS "
                "$$ BEGIN RETURN NULL; END $$; "
                f"CREATE TRIGGER {fn} BEFORE UPDATE ON tenant_defaults "
                f"FOR EACH ROW EXECUTE FUNCTION {fn}()"
            )
        ),
    )
    try:
        with pytest.raises(RemapRefusedError, match=r"post-check.*tenant_defaults\.updated_by"):
            await _sync(db_engine, apply_remap, frm, to)
    finally:
        await _sync(
            db_engine,
            lambda c: c.execute(
                text(f"DROP TRIGGER {fn} ON tenant_defaults; DROP FUNCTION {fn}()")
            ),
        )
    assert await _sync(db_engine, _refs, frm) == SEEDED
    owner = await _sync(
        db_engine,
        lambda c: c.execute(
            text("SELECT owner_subject_id::text FROM projects WHERE id = :p"), {"p": str(project)}
        ).scalar_one(),
    )
    assert owner == frm


# ─── pre-existing --to ───────────────────────────────────────────────────


def _seed_stub(conn: Connection, to: str, tenant: UUID, *, email: str = EMAIL) -> None:
    conn.execute(
        text(
            "INSERT INTO tenant_members (subject_id, tenant_id, email, roles) "
            "VALUES (:s, :t, :e, '[]')"
        ),
        {"s": to, "t": str(tenant), "e": email},
    )
    conn.execute(
        text(
            "INSERT INTO request_audit (id, event, method, path, status_code, actor_subject_id, "
            "detail) VALUES (:i, 'read', 'GET', '/api/auth/me', 200, :s, '{}')"
        ),
        {"i": str(uuid7()), "s": to},
    )


async def test_preexisting_to_is_refused_without_the_flag(db_engine, subs) -> None:
    frm, to, other, tenant = subs
    await _sync(db_engine, _seed, frm, other, tenant)
    await _sync(db_engine, _seed_stub, to, tenant)
    with pytest.raises(RemapRefusedError, match=r"already referenced.*Plan \(one transaction\)"):
        await _sync(db_engine, plan_remap, frm, to, Options(email=EMAIL))


async def test_resolve_preexisting_deletes_the_colliding_stub_and_rekeys(db_engine, subs) -> None:
    frm, to, other, tenant = subs
    await _sync(db_engine, _seed, frm, other, tenant)
    await _sync(db_engine, _seed_stub, to, tenant)
    o = Options(resolve_preexisting=True, email=EMAIL)
    dry = await _sync(db_engine, plan_remap, frm, to, o)
    assert dry.preexisting is not None and dry.preexisting.collisions == 1
    assert any("DELETE 1 --to tenant_members" in line for line in report(frm, to, dry, False))
    await _sync(db_engine, apply_remap, frm, to, o)
    assert await _sync(db_engine, _refs, frm) == {}
    after = await _sync(db_engine, _refs, to)
    assert after == {**SEEDED, "request_audit.actor_subject_id": 2}
    # the surviving membership is the ORIGINAL one (it carries invited_by = other)
    inv = await _sync(
        db_engine,
        lambda c: c.execute(
            text("SELECT invited_by::text FROM tenant_members WHERE subject_id = :s"), {"s": to}
        ).scalar_one(),
    )
    assert inv == other


@pytest.mark.parametrize(
    ("setup", "options", "blocker"),
    [("project", Options(True, EMAIL), "outside the stub tables")],
)
async def test_resolve_preexisting_refuses_anything_but_a_same_person_post_move_stub(
    db_engine, subs, setup: str, options: Options, blocker: str
) -> None:
    frm, to, other, tenant = subs
    await _sync(db_engine, _seed, frm, other, tenant)
    await _sync(db_engine, _seed_stub, to, tenant)
    if setup == "project":
        await _sync(
            db_engine,
            lambda c: c.execute(
                text(
                    "INSERT INTO projects (id, tenant_id, owner_subject_id, type, title, "
                    "visibility, brief, brand, voice, theme) "
                    "VALUES (:p, :t, :s, 'generic', 'mine', 'private', '{}', '{}', '{}', '{}')"
                ),
                {"p": str(uuid7()), "t": str(tenant), "s": to},
            ),
        )
    if setup == "old_stub":
        await _sync(
            db_engine,
            lambda c: c.execute(
                text(
                    "UPDATE tenant_members SET created_at = now() - interval '3 days' "
                    "WHERE subject_id = :s"
                ),
                {"s": to},
            ),
        )
    with pytest.raises(RemapRefusedError, match="NOT a removable stub") as exc:
        await _sync(db_engine, apply_remap, frm, to, options)
    assert blocker in str(exc.value)
    assert await _sync(db_engine, _refs, frm) == SEEDED


# ─── RLS guard ───────────────────────────────────────────────────────────


async def test_rls_guard_refuses_a_role_that_row_level_security_would_blind(
    db_engine, subs
) -> None:
    frm, to, other, tenant = subs
    await _sync(db_engine, _seed, frm, other, tenant)
    role = f"remap_probe_{uuid7().hex[:12]}"
    async with db_engine.connect() as conn:
        schema = (await conn.execute(text("SELECT current_schema()"))).scalar_one()
        try:
            await conn.execute(text(f'CREATE ROLE "{role}" NOLOGIN NOSUPERUSER NOBYPASSRLS'))
        except ProgrammingError as exc:
            await conn.rollback()
            require_createrole(exc)
        await conn.execute(text(f'GRANT "{role}" TO CURRENT_USER'))
        await conn.execute(text(f'GRANT USAGE ON SCHEMA "{schema}" TO "{role}"'))
        await conn.execute(text(f'GRANT SELECT ON ALL TABLES IN SCHEMA "{schema}" TO "{role}"'))
        await conn.commit()
    try:
        async with db_engine.connect() as conn:
            await conn.begin()
            await conn.execute(text(f'SET LOCAL ROLE "{role}"'))
            blind = await conn.run_sync(rls_blind_tables)
            assert "projects" in blind and "run_usage" in blind
            assert "tenant_members" not in blind  # no RLS there
            with pytest.raises(RemapRefusedError, match="row-level security"):
                await conn.run_sync(lambda c: plan_remap(c, frm, to))
            await conn.rollback()
        assert await _sync(db_engine, rls_blind_tables) == []  # the owner sees everything
    finally:
        async with db_engine.connect() as conn:
            await conn.execute(text(f'REVOKE ALL ON ALL TABLES IN SCHEMA "{schema}" FROM "{role}"'))
            await conn.execute(text(f'REVOKE USAGE ON SCHEMA "{schema}" FROM "{role}"'))
            await conn.execute(text(f'DROP ROLE "{role}"'))
            await conn.commit()


# ─── CLI ─────────────────────────────────────────────────────────────────


async def test_main_dry_run_then_apply_then_refusal(db_engine, subs, monkeypatch, capsys) -> None:
    from tests.conftest import TEST_DATABASE_URL

    frm, to, other, tenant = subs
    await _sync(db_engine, _seed, frm, other, tenant)
    monkeypatch.setenv("DATABASE_URL", TEST_DATABASE_URL)
    assert rus.main(["--from", frm, "--to", to]) == 0
    assert "DRY RUN" in capsys.readouterr().out
    assert await _sync(db_engine, _refs, to) == {}
    assert rus.main(["--from", frm, "--to", to, "--apply"]) == 0
    assert "COMMITTED" in capsys.readouterr().out
    assert await _sync(db_engine, _refs, frm) == {}
    assert rus.main(["--from", str(uuid7()), "--to", str(uuid7())]) == 1
    assert "REFUSED/ROLLED BACK" in capsys.readouterr().err
    assert rus.main(["--from", "x", "--to", to]) == 1
    assert "usage:" in capsys.readouterr().err
    monkeypatch.delenv("DATABASE_URL")
    assert rus.main(["--from", frm, "--to", to]) == 1
    assert "DATABASE_URL" in capsys.readouterr().err


def test_sync_url_normalises_driver() -> None:
    assert rus._sync_url("postgresql://u@h/d") == "postgresql+psycopg://u@h/d"
    assert rus._sync_url("postgres://u@h/d") == "postgresql+psycopg://u@h/d"
