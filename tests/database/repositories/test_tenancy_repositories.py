"""P2 tenancy repositories against REAL postgres (members, invites, API keys, tenants).

Each test starts from TRUNCATEd tables (``db_engine``). The cross-tenant cases are
the point: the invite and API-key tables carry no RLS by design, so the tenant
scoping lives in these WHERE clauses and nowhere else.
"""

from __future__ import annotations

import datetime
from uuid import UUID, uuid7

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError

from src.auth.api_key_format import hash_api_key, parse_api_key
from src.database.models import Tenant, TenantApiKey, TenantMember
from src.database.repositories.admin_repository import AdminRepository, TenantCreatedVia
from src.database.repositories.api_keys_repository import ApiKeysRepository
from src.database.repositories.invites_repository import (
    InvitesRepository,
    hash_invite_token,
    new_invite_token,
)
from src.database.repositories.members_repository import MembersRepository

pytestmark = pytest.mark.asyncio


def _some[T](value: T | None) -> T:
    assert value is not None
    return value


_NOW = datetime.datetime.now(datetime.UTC)


async def _tenant(session, created_via: TenantCreatedVia = "super_admin") -> UUID:
    tenant_id = uuid7()
    assert await AdminRepository(session).ensure_tenant(tenant_id, created_via=created_via)
    return tenant_id


# ─── members ─────────────────────────────────────────────────────────────


async def test_record_seen_insert_gets_member_active_defaults_and_never_touches_role(
    db_session,
) -> None:
    repo = MembersRepository(db_session)
    sub, tenant = uuid7(), uuid7()
    await repo.record_seen(
        subject_id=sub, tenant_id=tenant, email="a@x.test", display_name=None, roles=["member"]
    )
    row = await repo.get_member(sub, tenant)
    assert row is not None
    assert (row.role, row.status, row.removed_at) == ("member", "active", None)

    await repo.set_role(sub, tenant, "tenant_admin")
    assert await repo.remove_member(sub, tenant)
    # A login after removal refreshes the snapshot but neither re-grants nor revives.
    await repo.record_seen(
        subject_id=sub, tenant_id=tenant, email=None, display_name="A", roles=["tenant_admin"]
    )
    await db_session.commit()
    db_session.expire_all()
    row = await repo.get_member(sub, tenant)
    assert row is not None
    assert (row.role, row.status, row.email, row.display_name) == (
        "tenant_admin",
        "removed",
        "a@x.test",
        "A",
    )
    assert row.removed_at is not None


async def test_resolvable_memberships_require_a_registered_non_suspended_tenant(
    db_session,
) -> None:
    """A membership resolves only into a tenant w2v provisioned: no ``tenants``
    row → not resolvable (the request path then skips require_known_tenant)."""
    repo = MembersRepository(db_session)
    admin = AdminRepository(db_session)
    sub = uuid7()
    active, lapsed, suspended = await _tenant(db_session), await _tenant(db_session), uuid7()
    unregistered = uuid7()
    assert await admin.ensure_tenant(suspended, created_via="super_admin")
    await db_session.execute(
        text("UPDATE tenants SET status = 'licence_lapsed' WHERE id = :t"), {"t": lapsed}
    )
    await db_session.execute(
        text("UPDATE tenants SET status = 'suspended' WHERE id = :t"), {"t": suspended}
    )
    for t in (active, lapsed, suspended, unregistered):
        await repo.add_member(subject_id=sub, tenant_id=t, role="member")
    await db_session.commit()

    resolvable = {r.tenant_id for r in await repo.list_resolvable_memberships(sub)}

    assert resolvable == {active, lapsed}
    assert await repo.tenant_accepts_lazy_members(active)
    assert await repo.tenant_accepts_lazy_members(lapsed)
    assert not await repo.tenant_accepts_lazy_members(suspended)
    assert not await repo.tenant_accepts_lazy_members(unregistered)
    assert (
        await db_session.execute(select(Tenant).where(Tenant.id == unregistered))
    ).first() is None, "the check must never create a tenant row"


async def test_membership_lifecycle_add_role_remove_readd(db_session) -> None:
    repo = MembersRepository(db_session)
    sub, inviter = uuid7(), uuid7()
    t1, t2 = await _tenant(db_session), await _tenant(db_session)

    await repo.add_member(subject_id=sub, tenant_id=t1, role="member", invited_by=inviter)
    await repo.add_member(subject_id=sub, tenant_id=t2, role="tenant_admin", email="s@x.test")
    await db_session.execute(
        text(
            "UPDATE tenant_members SET last_seen_at = now() + interval '1 minute' WHERE tenant_id = :t"
        ),
        {"t": t2},
    )
    active = await repo.list_active_memberships(sub)
    assert [m.tenant_id for m in active] == [t2, t1], "most recently seen first"
    assert await repo.count_active_admins(t2) == 1
    assert await repo.count_active_admins(t1) == 0

    assert await repo.set_role(sub, t1, "tenant_admin")
    assert await repo.remove_member(sub, t1)
    assert not await repo.remove_member(sub, t1), "second removal is a no-op"
    assert not await repo.set_role(sub, t1, "member"), "a removed member is not re-roled"
    assert await repo.get_active_membership(sub, t1) is None
    assert [m.tenant_id for m in await repo.list_active_memberships(sub)] == [t2]
    assert await repo.count_active_admins(t1) == 0

    # Re-adding revives the same row: removed_at cleared, invited_by kept.
    await repo.add_member(subject_id=sub, tenant_id=t1, role="member")
    await db_session.commit()
    db_session.expire_all()
    row = await repo.get_active_membership(sub, t1)
    assert row is not None
    assert (row.role, row.status, row.removed_at, row.invited_by) == (
        "member",
        "active",
        None,
        inviter,
    )


async def test_removed_status_without_removed_at_is_refused(db_session) -> None:
    sub, tenant = uuid7(), uuid7()
    await MembersRepository(db_session).add_member(subject_id=sub, tenant_id=tenant, role="member")
    with pytest.raises(IntegrityError):
        await db_session.execute(
            text("UPDATE tenant_members SET status = 'removed' WHERE subject_id = :s"), {"s": sub}
        )


async def test_list_members_carries_role_and_status(db_session) -> None:
    repo = MembersRepository(db_session)
    tenant = uuid7()
    await repo.add_member(subject_id=uuid7(), tenant_id=tenant, role="tenant_admin")
    rows, total = await repo.list_members(tenant, limit=10, offset=0)
    assert total == 1
    assert (rows[0].role, rows[0].status) == ("tenant_admin", "active")


# ─── invites ─────────────────────────────────────────────────────────────


async def test_invite_lifecycle_and_tenant_scoping(db_session) -> None:
    repo = InvitesRepository(db_session)
    t1, t2 = await _tenant(db_session), await _tenant(db_session)
    token, token_hash = new_invite_token()
    assert hash_invite_token(token) == token_hash and len(token_hash) == 32

    invite = await repo.create(
        tenant_id=t1,
        email="  New.User@Example.TEST ",
        role="member",
        token_hash=token_hash,
        expires_at=_NOW + datetime.timedelta(days=7),
        invited_by=uuid7(),
    )
    assert invite.email == "New.User@Example.TEST"
    assert _some(await repo.get_pending_by_token_hash(token_hash)).id == invite.id
    assert (
        _some(await repo.get_pending_for_tenant_email(t1, "new.user@example.test")).id == invite.id
    )
    assert [i.id for i in await repo.list_pending_for_email("NEW.USER@example.test")] == [invite.id]

    # Another tenant can neither see, revoke nor resend it.
    assert await repo.get(invite.id, t2) is None
    assert not await repo.revoke(invite.id, t2)
    _, other_hash = new_invite_token()
    assert not await repo.resend(invite.id, t2, token_hash=other_hash, expires_at=_NOW)

    # Resend rotates the token: the old link stops working.
    new_token, new_hash = new_invite_token()
    assert await repo.resend(
        invite.id, t1, token_hash=new_hash, expires_at=_NOW + datetime.timedelta(days=7)
    )
    assert await repo.get_pending_by_token_hash(token_hash) is None
    assert _some(await repo.get_pending_by_token_hash(hash_invite_token(new_token))).id == invite.id

    sub = uuid7()
    assert await repo.mark_accepted(invite.id, sub)
    assert not await repo.mark_accepted(invite.id, uuid7()), "accept is once-only"
    assert await repo.get_pending_by_token_hash(new_hash) is None
    invite_id = invite.id
    await db_session.commit()
    db_session.expire_all()
    settled = await repo.get(invite_id, t1)
    assert settled is not None
    assert (settled.status, settled.accepted_subject_id) == ("accepted", sub)
    assert settled.accepted_at is not None and settled.resent_at is not None


async def test_one_pending_invite_per_tenant_email_case_insensitive(db_session) -> None:
    repo = InvitesRepository(db_session)
    t1, t2 = await _tenant(db_session), await _tenant(db_session)
    expires = _NOW + datetime.timedelta(days=1)
    first = await repo.create(
        tenant_id=t1,
        email="dup@x.test",
        role="member",
        token_hash=new_invite_token()[1],
        expires_at=expires,
        invited_by=None,
    )
    # Same email in ANOTHER tenant is fine.
    await repo.create(
        tenant_id=t2,
        email="DUP@x.test",
        role="member",
        token_hash=new_invite_token()[1],
        expires_at=expires,
        invited_by=None,
    )
    first_id = first.id
    await db_session.commit()
    with pytest.raises(IntegrityError):
        await repo.create(
            tenant_id=t1,
            email="Dup@X.test",
            role="tenant_admin",
            token_hash=new_invite_token()[1],
            expires_at=expires,
            invited_by=None,
        )
    await db_session.rollback()
    # Once the first is revoked a new pending invite is allowed.
    assert await repo.revoke(first_id, t1)
    await repo.create(
        tenant_id=t1,
        email="dup@x.test",
        role="member",
        token_hash=new_invite_token()[1],
        expires_at=expires,
        invited_by=None,
    )
    assert len(await repo.list_for_tenant(t1, status=None)) == 2
    assert len(await repo.list_for_tenant(t1)) == 1


async def test_expired_invites_are_not_served_and_expire_stale_flips_them(db_session) -> None:
    repo = InvitesRepository(db_session)
    tenant = await _tenant(db_session)
    token_hash = new_invite_token()[1]
    await repo.create(
        tenant_id=tenant,
        email="late@x.test",
        role="member",
        token_hash=token_hash,
        expires_at=_NOW - datetime.timedelta(minutes=1),
        invited_by=None,
    )
    assert await repo.get_pending_by_token_hash(token_hash) is None
    assert await repo.list_pending_for_email("late@x.test") == []
    assert await repo.expire_stale() == 1
    assert await repo.expire_stale() == 0
    assert [i.status for i in await repo.list_for_tenant(tenant, status="expired")] == ["expired"]


# ─── API keys ────────────────────────────────────────────────────────────


async def test_api_key_create_resolve_revoke(db_session) -> None:
    repo = ApiKeysRepository(db_session)
    t1, t2 = await _tenant(db_session), await _tenant(db_session)
    row, full_key = await repo.create(tenant_id=t1, name="ci", created_by=uuid7())

    parsed = parse_api_key(full_key)
    assert parsed is not None and parsed.prefix == row.prefix
    assert row.key_hash == hash_api_key(full_key)
    stored = (
        await db_session.execute(select(TenantApiKey).where(TenantApiKey.id == row.id))
    ).scalar_one()
    assert parsed.secret.encode() not in stored.key_hash, "secret is never stored"

    resolved = await repo.resolve(full_key)
    assert resolved is not None and resolved.tenant_id == t1

    # Same prefix, wrong secret → no match (hash compare, not prefix trust).
    forged = f"w2v_sk_{row.prefix}_{'A' * 43}"
    assert await repo.resolve(forged) is None
    assert await repo.resolve("w2v_sk_garbage") is None

    assert [k.id for k in await repo.list_for_tenant(t1)] == [row.id]
    assert await repo.list_for_tenant(t2) == []
    assert not await repo.revoke(row.id, t2), "another tenant cannot revoke it"
    await repo.touch_last_used(row.id)
    assert await repo.revoke(row.id, t1)
    assert not await repo.revoke(row.id, t1)
    assert await repo.resolve(full_key) is None
    assert await repo.list_for_tenant(t1) == []
    assert len(await repo.list_for_tenant(t1, include_revoked=True)) == 1


async def test_expired_api_key_does_not_resolve(db_session) -> None:
    repo = ApiKeysRepository(db_session)
    tenant = await _tenant(db_session)
    _, full_key = await repo.create(
        tenant_id=tenant,
        name="short",
        created_by=None,
        expires_at=_NOW + datetime.timedelta(hours=1),
    )
    assert await repo.resolve(full_key, now=_NOW) is not None
    assert await repo.resolve(full_key, now=_NOW + datetime.timedelta(hours=2)) is None


# ─── tenants (admin repository) ──────────────────────────────────────────


async def test_ensure_tenant_is_insert_only_and_records_provenance(db_session) -> None:
    repo = AdminRepository(db_session)
    tenant, licence, partner = uuid7(), uuid7(), uuid7()
    assert await repo.ensure_tenant(
        tenant,
        created_via="partner",
        name="Acme",
        licence_ref=licence,
        partner_licence_id=partner,
    )
    assert not await repo.ensure_tenant(tenant, created_via="backfill", name="Other")
    row = await repo.get_tenant(tenant)
    assert row is not None
    assert (row.name, row.created_via, row.licence_ref, row.partner_licence_id, row.status) == (
        "Acme",
        "partner",
        licence,
        partner,
        "active",
    )
    assert row.licence_checked_at is not None


async def test_upsert_tenant_sets_created_via_on_insert_only(db_session) -> None:
    repo = AdminRepository(db_session)
    fresh, plain = uuid7(), uuid7()
    await repo.upsert_tenant(fresh, name="n", created_via="partner")
    await repo.upsert_tenant(fresh, name="renamed", created_via="backfill")
    await repo.upsert_tenant(plain, name="p")
    rows = {
        r.id: r.created_via
        for r in (await db_session.execute(select(Tenant).where(Tenant.id.in_([fresh, plain]))))
        .scalars()
        .all()
    }
    assert rows == {fresh: "partner", plain: "super_admin"}


async def test_record_licence_check_lapses_recovers_and_never_unsuspends(db_session) -> None:
    repo = AdminRepository(db_session)
    live, suspended, licence = uuid7(), uuid7(), uuid7()
    await repo.ensure_tenant(live, created_via="backfill")
    await repo.upsert_tenant(suspended, status="suspended")

    assert await repo.record_licence_check(live, entitled=False)
    assert _some(await repo.get_tenant(live)).status == "licence_lapsed"
    assert await repo.record_licence_check(live, entitled=True, licence_ref=licence)
    db_session.expire_all()
    row = _some(await repo.get_tenant(live))
    assert (row.status, row.licence_ref) == ("active", licence)
    assert row.licence_checked_at is not None

    assert await repo.record_licence_check(suspended, entitled=True)
    assert await repo.record_licence_check(suspended, entitled=False)
    db_session.expire_all()
    assert _some(await repo.get_tenant(suspended)).status == "suspended"
    assert not await repo.record_licence_check(uuid7(), entitled=True), "no row → False"


async def test_member_table_has_no_rls(db_session) -> None:
    """Membership/invite lookups are cross-tenant by design (blueprint §2). API keys
    are NOT any more (0090): the table is RLS-isolated and the one pre-tenant
    lookup goes through the ``resolve_api_key`` SECURITY DEFINER resolver."""
    flags = (
        await db_session.execute(
            text(
                "SELECT relname, relrowsecurity FROM pg_class WHERE relname IN "
                "('tenant_members','tenant_invites','tenant_api_keys') "
                "AND relnamespace = to_regnamespace(current_schema())"
            )
        )
    ).all()
    assert sorted(flags) == [
        ("tenant_api_keys", True),
        ("tenant_invites", False),
        ("tenant_members", False),
    ]
    assert TenantMember.__table__.c.role.server_default is not None
