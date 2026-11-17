"""``scripts/backfill_tenancy.py`` — the Keycloak-aware half of the P2 backfill.

Two layers: ``plan_backfill`` is pure (every rule pinned here without I/O), and
``run`` is exercised against the real test database with a FAKE identity reader,
so dry-run-writes-nothing and apply-writes-exactly-the-plan are proven on rows,
not on mocks.
"""

from __future__ import annotations

from uuid import UUID, uuid7

import pytest
from sqlalchemy import select

from scripts.backfill_tenancy import (
    KcIdentity,
    MemberRow,
    desired_role,
    identity_from_kc_user,
    licensed_from_file,
    plan_backfill,
    run,
)
from src.database.models import Tenant, TenantMember
from src.database.repositories.admin_repository import AdminRepository
from src.database.repositories.members_repository import MembersRepository

HOME, OTHER, LIC_T = uuid7(), uuid7(), uuid7()
LICENCE = uuid7()


def _row(sub: UUID, tenant: UUID, role: str = "member", status: str = "active") -> MemberRow:
    return MemberRow(sub, tenant, role, status, f"{sub}@x.test", None)


def _kc(home: UUID | None, *roles: str, found: bool = True) -> KcIdentity:
    return KcIdentity(found, home, frozenset(roles), None if home else "no_tenant_attribute")


# ─── pure planning ───────────────────────────────────────────────────────


def test_super_admin_scope_switch_rows_are_removed_and_home_role_is_member() -> None:
    # super_admin alone is NOT tenant_admin: its authority is the realm role at
    # request time. A home row that claims tenant_admin is demoted to member.
    sa = uuid7()
    plan = plan_backfill(
        [_row(sa, HOME, "tenant_admin"), _row(sa, OTHER, "tenant_admin")],
        {sa: _kc(HOME, "super_admin")},
        {},
        {HOME: None, OTHER: None},
    )
    assert [(r.tenant_id, r.reason) for r in plan.removals] == [(OTHER, "super_admin_scope_switch")]
    assert [(c.tenant_id, c.old, c.new) for c in plan.role_changes] == [
        (HOME, "tenant_admin", "member")
    ]
    assert plan.additions == [] and plan.tenant_creates == []


def test_home_role_comes_from_the_live_tenant_admin_realm_role_only() -> None:
    both, sa_only, ta = uuid7(), uuid7(), uuid7()
    plan = plan_backfill(
        [_row(both, HOME, "member"), _row(sa_only, HOME, "member"), _row(ta, HOME, "member")],
        {
            both: _kc(HOME, "super_admin", "tenant_admin"),
            sa_only: _kc(HOME, "super_admin"),
            ta: _kc(HOME, "tenant_admin"),
        },
        {},
        {HOME: None},
    )
    assert sorted((c.subject_id, c.new) for c in plan.role_changes) == sorted(
        [(both, "tenant_admin"), (ta, "tenant_admin")]
    )
    assert desired_role(["super_admin"]) == "member"
    assert desired_role(["super_admin", "tenant_admin"]) == "tenant_admin"
    assert desired_role(["member"]) == "member"


def test_demoted_admin_and_moved_member() -> None:
    demoted, moved = uuid7(), uuid7()
    plan = plan_backfill(
        [_row(demoted, HOME, "tenant_admin"), _row(moved, OTHER)],
        {demoted: _kc(HOME, "member"), moved: _kc(HOME, "member")},
        {},
        {HOME: None, OTHER: None},
    )
    assert [(c.subject_id, c.new) for c in plan.role_changes] == [(demoted, "member")]
    assert [(r.subject_id, r.reason) for r in plan.removals] == [(moved, "claim_tenant_mismatch")]
    # The moved member gets a home membership; HOME already has a tenants row.
    assert [(a.subject_id, a.tenant_id, a.role, a.email) for a in plan.additions] == [
        (moved, HOME, "member", f"{moved}@x.test")
    ]
    assert plan.tenant_creates == []


def test_missing_or_ambiguous_identities_are_skipped_never_guessed() -> None:
    gone, unattributed = uuid7(), uuid7()
    plan = plan_backfill(
        [_row(gone, HOME), _row(unattributed, OTHER)],
        {gone: KcIdentity(False), unattributed: _kc(None, "super_admin")},
        {},
        {HOME: None, OTHER: None},
    )
    assert {(s.subject_id, s.reason) for s in plan.skips} == {
        (gone, "kc_user_missing"),
        (unattributed, "no_tenant_attribute"),
    }
    assert plan.is_noop


def test_removed_home_row_is_not_revived_and_already_removed_rows_are_left() -> None:
    sub = uuid7()
    plan = plan_backfill(
        [_row(sub, HOME, status="removed"), _row(sub, OTHER, status="removed")],
        {sub: _kc(HOME, "tenant_admin")},
        {},
        {HOME: None, OTHER: None},
    )
    assert plan.kept_removed == [(sub, HOME)]
    assert plan.removals == [] and plan.role_changes == [] and plan.additions == []


def test_licensed_tenants_get_rows_and_missing_licence_refs() -> None:
    sub, new_home = uuid7(), uuid7()
    plan = plan_backfill(
        [_row(sub, OTHER)],
        {sub: _kc(new_home, "member")},
        {LIC_T: LICENCE, HOME: LICENCE, OTHER: LICENCE},
        {HOME: None, OTHER: uuid7()},  # OTHER already has a ref: left alone
    )
    creates = {(t.tenant_id, t.licence_ref) for t in plan.tenant_creates}
    assert creates == {(LIC_T, LICENCE), (new_home, None)}
    assert plan.licence_refs == [(HOME, LICENCE)]


def test_identity_from_kc_user_parses_the_tenant_attribute() -> None:
    t = uuid7()
    assert identity_from_kc_user({"attributes": {"tenant_id": [str(t)]}}, ["member"]) == KcIdentity(
        True, t, frozenset({"member"})
    )
    assert identity_from_kc_user({}, []).tenant_problem == "no_tenant_attribute"
    assert (
        identity_from_kc_user({"attributes": {"tenant_id": ["nope"]}}, []).tenant_problem
        == "invalid_tenant_attribute"
    )
    two = {"attributes": {"tenant_id": [str(t), str(uuid7())]}}
    assert identity_from_kc_user(two, []).tenant_problem == "multiple_tenant_attributes"


def test_licensed_from_file(tmp_path) -> None:
    path = tmp_path / "licensed.json"
    path.write_text(f'{{"{LIC_T}": "{LICENCE}"}}')
    assert licensed_from_file(path) == {LIC_T: LICENCE}
    path.write_text("[]")
    with pytest.raises(ValueError, match="JSON object"):
        licensed_from_file(path)


# ─── run() against the real database ─────────────────────────────────────


async def _seed(session) -> dict[str, UUID]:
    ids = {"home": uuid7(), "other": uuid7(), "sa": uuid7(), "mem": uuid7(), "gone": uuid7()}
    admin, members = AdminRepository(session), MembersRepository(session)
    await admin.ensure_tenant(ids["home"], created_via="backfill")
    await admin.ensure_tenant(ids["other"], created_via="backfill")
    for sub, tenant, roles in [
        (ids["sa"], ids["home"], ["super_admin"]),
        (ids["sa"], ids["other"], ["super_admin"]),
        (ids["mem"], ids["home"], ["member"]),
        (ids["gone"], ids["other"], ["member"]),
    ]:
        await members.record_seen(
            subject_id=sub, tenant_id=tenant, email=None, display_name=None, roles=roles
        )
    await session.commit()
    return ids


async def _state(session_factory) -> tuple[dict[tuple[UUID, UUID], tuple[str, str]], set[UUID]]:
    async with session_factory() as s:
        members = (
            await s.execute(
                select(
                    TenantMember.subject_id,
                    TenantMember.tenant_id,
                    TenantMember.role,
                    TenantMember.status,
                )
            )
        ).all()
        tenants = set((await s.execute(select(Tenant.id))).scalars().all())
    return {(m[0], m[1]): (m[2], m[3]) for m in members}, tenants


async def test_run_dry_run_writes_nothing_and_apply_writes_the_plan(db_session_factory) -> None:
    async with db_session_factory() as s:
        ids = await _seed(s)
    licensed_tenant = uuid7()
    identities = {
        ids["sa"]: _kc(ids["home"], "super_admin"),
        ids["mem"]: _kc(ids["home"], "member", "tenant_admin"),
        ids["gone"]: KcIdentity(False),
    }

    async def reader(subject_id: UUID) -> KcIdentity:
        return identities[subject_id]

    before = await _state(db_session_factory)
    out: list[str] = []
    rc = await run(
        db_session_factory, reader, {licensed_tenant: LICENCE}, apply=False, out=out.append
    )
    assert rc == 0
    assert await _state(db_session_factory) == before, "dry run must not write"
    assert out[-1] == "DRY RUN — nothing written."
    assert any(line.startswith("REMOVE") and "super_admin_scope_switch" in line for line in out)
    assert any(line.startswith("ROLE") for line in out)
    assert any(line.startswith(f"TENANT+  {licensed_tenant}") for line in out)
    assert any(line.startswith(f"SKIP     {ids['gone']}: kc_user_missing") for line in out)

    out.clear()
    rc = await run(
        db_session_factory, reader, {licensed_tenant: LICENCE}, apply=True, out=out.append
    )
    assert rc == 0 and out[-1] == "APPLIED (committed)."
    members, tenants = await _state(db_session_factory)
    assert members == {
        # super_admin-only: home row stays member (realm role is the authority).
        (ids["sa"], ids["home"]): ("member", "active"),
        (ids["sa"], ids["other"]): ("member", "removed"),
        # live tenant_admin realm role → promoted in the home tenant.
        (ids["mem"], ids["home"]): ("tenant_admin", "active"),
        (ids["gone"], ids["other"]): ("member", "active"),  # skipped, untouched
    }
    assert licensed_tenant in tenants
    async with db_session_factory() as s:
        row = await AdminRepository(s).get_tenant(licensed_tenant)
        assert row is not None and (row.created_via, row.licence_ref) == ("backfill", LICENCE)

    # Idempotent: a second apply has nothing to do.
    out.clear()
    assert (
        await run(
            db_session_factory, reader, {licensed_tenant: LICENCE}, apply=True, out=out.append
        )
        == 0
    )
    assert out[-1] == "APPLY: nothing to change — already converged."
    assert "role_changes=0 removals=0 additions=0 tenants_created=0 licence_refs=0" in out[-2]


async def test_run_with_an_empty_roster_is_a_failure_not_a_clean_estate(
    db_session_factory,
) -> None:
    async def reader(subject_id: UUID) -> KcIdentity:  # pragma: no cover - never called
        raise AssertionError(subject_id)

    out: list[str] = []
    assert await run(db_session_factory, reader, {}, apply=True, out=out.append) == 1
    assert "SCANNED ZERO" in out[0]
