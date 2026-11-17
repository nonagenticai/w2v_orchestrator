"""P2 tenancy backfill: make ``tenant_members`` + ``tenants`` match Keycloak + licences.

Migration 0072 did the pure-SQL part (role from the ``roles`` snapshot, a tenants
row per member tenant). What SQL cannot know is each subject's HOME tenant — the
Keycloak ``tenant_id`` attribute — so ``tenant_members`` still carries the
super_admin scope-switch pollution (``record_seen`` wrote the EFFECTIVE tenant),
and the role snapshot may be stale. This script fixes that, per subject:

* read the Keycloak user (``tenant_id`` attribute + EFFECTIVE realm roles);
* desired role in the home tenant = ``tenant_admin`` iff the live realm roles
  hold ``tenant_admin``, else ``member`` — the same mapping as migration 0072.
  ``super_admin`` alone maps to ``member``: super_admin authority comes from the
  realm role at request time, never from a membership row;
* home-tenant row: correct its ``role``; missing → ADD it (the cutover gate is
  "every active user has a membership");
* rows in any OTHER tenant → ``removed`` (``super_admin_scope_switch`` for a
  super_admin, ``claim_tenant_mismatch`` otherwise — under claim mode that
  subject could not reach that tenant either, so removal preserves today's
  access exactly);
* a row already ``removed`` is never reactivated (a tenant_admin may have
  removed it after cutover — re-running must not undo that).

Subjects that Keycloak no longer knows, or that carry no/invalid/multiple
``tenant_id`` attributes, are SKIPPED and listed — never guessed at.

Licensed tenants: every tenant holding an effectively-active ``w2v`` licence gets
a ``tenants`` row (``created_via='backfill'``, ``licence_ref``). Source is the
licenses service's ``/internal/licenses`` (needs ``service-internal`` on the
``w2v-backend`` SA — a 403 ABORTS the run before any write, it is never read as
"no licences"), or ``--licensed-tenants-file`` (JSON ``{tenant_id: licence_id}``).

Dry run is the DEFAULT and prints the full diff; ``--apply`` writes everything in
ONE transaction. Run inside the orchestrator pod (it needs the pod's env)::

    python -m scripts.backfill_tenancy                 # dry run, diff report
    python -m scripts.backfill_tenancy --apply

Exit codes: 0 ok · 1 nothing to scan (empty ``tenant_members`` — a vacuous run is
a failure, not a success) · 2 a source (Keycloak / licenses) could not be read.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.database.models import Tenant, TenantMember
from src.database.repositories.admin_repository import AdminRepository
from src.database.repositories.members_repository import MembersRepository

W2V_APP_KEY = "w2v"


# ─── inputs ──────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class MemberRow:
    subject_id: UUID
    tenant_id: UUID
    role: str
    status: str
    email: str | None = None
    display_name: str | None = None


@dataclass(frozen=True, slots=True)
class KcIdentity:
    """What Keycloak says about one subject. ``found=False`` = no such user."""

    found: bool
    home_tenant: UUID | None = None
    realm_roles: frozenset[str] = frozenset()
    #: Why ``home_tenant`` is None for a found user (missing / invalid / multiple).
    tenant_problem: str | None = None


IdentityReader = Callable[[UUID], Awaitable[KcIdentity]]


# ─── plan ────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class RoleChange:
    subject_id: UUID
    tenant_id: UUID
    old: str
    new: str


@dataclass(frozen=True, slots=True)
class Removal:
    subject_id: UUID
    tenant_id: UUID
    reason: str


@dataclass(frozen=True, slots=True)
class Addition:
    subject_id: UUID
    tenant_id: UUID
    role: str
    email: str | None
    display_name: str | None


@dataclass(frozen=True, slots=True)
class TenantCreate:
    tenant_id: UUID
    licence_ref: UUID | None


@dataclass(frozen=True, slots=True)
class Skip:
    subject_id: UUID
    reason: str


@dataclass
class Plan:
    members_scanned: int = 0
    subjects: int = 0
    role_changes: list[RoleChange] = field(default_factory=list)
    removals: list[Removal] = field(default_factory=list)
    additions: list[Addition] = field(default_factory=list)
    tenant_creates: list[TenantCreate] = field(default_factory=list)
    #: existing tenants rows that get a ``licence_ref`` (they had none).
    licence_refs: list[tuple[UUID, UUID]] = field(default_factory=list)
    skips: list[Skip] = field(default_factory=list)
    #: removed rows in the home tenant that this run deliberately did NOT revive.
    kept_removed: list[tuple[UUID, UUID]] = field(default_factory=list)

    @property
    def is_noop(self) -> bool:
        return not (
            self.role_changes
            or self.removals
            or self.additions
            or self.tenant_creates
            or self.licence_refs
        )


def desired_role(realm_roles: Iterable[str]) -> str:
    """Tenant-scoped role from the LIVE realm roles. Only an explicit
    ``tenant_admin`` grants ``tenant_admin``; ``super_admin`` does not (its
    authority is the realm role itself, checked per request)."""
    return "tenant_admin" if "tenant_admin" in set(realm_roles) else "member"


def _plan_subject(
    plan: Plan, subject_id: UUID, rows: list[MemberRow], ident: KcIdentity
) -> UUID | None:
    """Append one subject's changes to ``plan``. Returns the home tenant when a
    membership must be ADDED there (so the caller ensures its tenants row)."""
    if not ident.found:
        plan.skips.append(Skip(subject_id, "kc_user_missing"))
        return None
    if ident.home_tenant is None:
        plan.skips.append(Skip(subject_id, ident.tenant_problem or "no_tenant_attribute"))
        return None
    home = ident.home_tenant
    role = desired_role(ident.realm_roles)
    reason = (
        "super_admin_scope_switch"
        if "super_admin" in ident.realm_roles
        else "claim_tenant_mismatch"
    )
    plan.removals.extend(
        Removal(subject_id, r.tenant_id, reason)
        for r in rows
        if r.tenant_id != home and r.status == "active"
    )
    home_row = next((r for r in rows if r.tenant_id == home), None)
    if home_row is None:
        named = next((r for r in rows if r.email or r.display_name), None)
        email = named.email if named else None
        display_name = named.display_name if named else None
        plan.additions.append(Addition(subject_id, home, role, email, display_name))
        return home
    if home_row.status != "active":
        plan.kept_removed.append((subject_id, home))
    elif home_row.role != role:
        plan.role_changes.append(RoleChange(subject_id, home, home_row.role, role))
    return None


def plan_backfill(
    members: list[MemberRow],
    identities: dict[UUID, KcIdentity],
    licensed: dict[UUID, UUID],
    existing_tenants: dict[UUID, UUID | None],
) -> Plan:
    """Pure: compute every change from the four inputs. No I/O.

    ``identities`` must hold an entry for every subject in ``members``;
    ``licensed`` maps tenant → licence id; ``existing_tenants`` maps tenant →
    its current ``licence_ref``.
    """
    plan = Plan(members_scanned=len(members))
    by_subject: dict[UUID, list[MemberRow]] = defaultdict(list)
    for row in members:
        by_subject[row.subject_id].append(row)
    plan.subjects = len(by_subject)

    needed_tenants: set[UUID] = set()
    for subject_id in sorted(by_subject, key=str):
        rows = sorted(by_subject[subject_id], key=lambda r: str(r.tenant_id))
        added_home = _plan_subject(plan, subject_id, rows, identities[subject_id])
        if added_home is not None:
            needed_tenants.add(added_home)

    for tenant_id in sorted(needed_tenants | set(licensed), key=str):
        if tenant_id not in existing_tenants:
            plan.tenant_creates.append(TenantCreate(tenant_id, licensed.get(tenant_id)))
        elif tenant_id in licensed and existing_tenants[tenant_id] is None:
            plan.licence_refs.append((tenant_id, licensed[tenant_id]))
    return plan


_OUTCOME = {
    "dry_run": "DRY RUN — nothing written.",
    "applied": "APPLIED (committed).",
    "converged": "APPLY: nothing to change — already converged.",
}


def render_report(plan: Plan, *, outcome: str) -> list[str]:
    lines = [
        f"ROLE     {c.subject_id} @ {c.tenant_id}: {c.old} -> {c.new}" for c in plan.role_changes
    ]
    lines += [f"REMOVE   {r.subject_id} @ {r.tenant_id}: {r.reason}" for r in plan.removals]
    lines += [f"ADD      {a.subject_id} @ {a.tenant_id}: {a.role}" for a in plan.additions]
    lines += [f"TENANT+  {t.tenant_id} licence_ref={t.licence_ref}" for t in plan.tenant_creates]
    lines += [f"LICENCE  {t} licence_ref={lic}" for t, lic in plan.licence_refs]
    lines += [f"KEPT-RM  {s} @ {t}: removed member not reactivated" for s, t in plan.kept_removed]
    lines += [f"SKIP     {s.subject_id}: {s.reason}" for s in plan.skips]
    lines.append(
        f"summary: members_scanned={plan.members_scanned} subjects={plan.subjects} "
        f"role_changes={len(plan.role_changes)} removals={len(plan.removals)} "
        f"additions={len(plan.additions)} tenants_created={len(plan.tenant_creates)} "
        f"licence_refs={len(plan.licence_refs)} kept_removed={len(plan.kept_removed)} "
        f"skipped={len(plan.skips)}"
    )
    lines.append(_OUTCOME[outcome])
    return lines


# ─── DB ──────────────────────────────────────────────────────────────────


async def load_members(session: AsyncSession) -> list[MemberRow]:
    rows = (
        await session.execute(
            select(
                TenantMember.subject_id,
                TenantMember.tenant_id,
                TenantMember.role,
                TenantMember.status,
                TenantMember.email,
                TenantMember.display_name,
            )
        )
    ).all()
    return [MemberRow(*r) for r in rows]


async def load_tenants(session: AsyncSession) -> dict[UUID, UUID | None]:
    rows = (await session.execute(select(Tenant.id, Tenant.licence_ref))).all()
    return {r[0]: r[1] for r in rows}


async def apply_plan(session: AsyncSession, plan: Plan) -> None:
    """Write the plan through the repositories. No commit — caller decides."""
    admin = AdminRepository(session)
    members = MembersRepository(session)
    for t in plan.tenant_creates:
        await admin.ensure_tenant(t.tenant_id, created_via="backfill", licence_ref=t.licence_ref)
    for tenant_id, licence_id in plan.licence_refs:
        await admin.record_licence_check(tenant_id, entitled=True, licence_ref=licence_id)
    for a in plan.additions:
        await members.add_member(
            subject_id=a.subject_id,
            tenant_id=a.tenant_id,
            role="tenant_admin" if a.role == "tenant_admin" else "member",
            email=a.email,
            display_name=a.display_name,
        )
    for c in plan.role_changes:
        changed = await members.set_role(
            c.subject_id, c.tenant_id, "tenant_admin" if c.new == "tenant_admin" else "member"
        )
        if not changed:
            raise RuntimeError(f"role change raced: {c.subject_id} @ {c.tenant_id}")
    for r in plan.removals:
        if not await members.remove_member(r.subject_id, r.tenant_id):
            raise RuntimeError(f"removal raced: {r.subject_id} @ {r.tenant_id}")


async def run(
    session_maker: async_sessionmaker[AsyncSession],
    read_identity: IdentityReader,
    licensed: dict[UUID, UUID],
    *,
    apply: bool,
    out: Callable[[str], None] = print,
) -> int:
    # Snapshot first, in a short read transaction: the Keycloak reads below are
    # one HTTP round-trip per subject and must not hold a transaction open.
    async with session_maker() as session:
        members = await load_members(session)
        tenants = await load_tenants(session)
    if not members:
        out(
            "SCANNED ZERO tenant_members rows. A backfill that sees nothing is a "
            "FAILURE, not a clean estate — check DATABASE_URL points at the "
            "orchestrator database."
        )
        return 1
    identities: dict[UUID, KcIdentity] = {}
    for subject_id in sorted({m.subject_id for m in members}, key=str):
        identities[subject_id] = await read_identity(subject_id)
    plan = plan_backfill(members, identities, licensed, tenants)
    outcome = "converged" if apply else "dry_run"
    if apply and not plan.is_noop:
        # One transaction. set_role/remove_member only match ACTIVE rows and raise
        # if a row moved since the snapshot, which rolls the whole apply back.
        async with session_maker() as session:
            await apply_plan(session, plan)
            await session.commit()
        outcome = "applied"
    for line in render_report(plan, outcome=outcome):
        out(line)
    return 0


# ─── live sources ────────────────────────────────────────────────────────


def identity_from_kc_user(user: Mapping[str, object], realm_roles: Iterable[str]) -> KcIdentity:
    attrs = user.get("attributes")
    values = attrs.get("tenant_id") if isinstance(attrs, dict) else None
    values = (
        [v for v in values if isinstance(v, str) and v.strip()] if isinstance(values, list) else []
    )
    roles = frozenset(realm_roles)
    if not values:
        return KcIdentity(True, None, roles, "no_tenant_attribute")
    if len(set(values)) > 1:
        return KcIdentity(True, None, roles, "multiple_tenant_attributes")
    try:
        return KcIdentity(True, UUID(values[0].strip()), roles)
    except ValueError:
        return KcIdentity(True, None, roles, "invalid_tenant_attribute")


def licensed_from_file(path: Path) -> dict[UUID, UUID]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("--licensed-tenants-file must hold a JSON object {tenant_id: licence_id}")
    return {UUID(str(t)): UUID(str(lic)) for t, lic in data.items()}


async def _licensed_from_service() -> dict[UUID, UUID]:
    """Every tenant's live w2v licence, from licences' cross-tenant listing.

    The ONE remaining ``/internal/*`` licences call in this repo, deliberately kept
    HERE: this is a one-shot, in-cluster operator backfill run inside the pod with
    the ``w2v-backend`` SA's ``service-internal`` role. The service itself verifies
    licences by introspection (``src/services/tenant_licence.py``) because an
    off-cluster ingress refuses ``/internal/*``; this script never runs off-cluster."""
    import httpx
    from w2v_common.discovery import service_url

    from src.clients.service_auth import get_service_authenticator

    base = service_url("licenses").rstrip("/")
    headers = {"Authorization": f"Bearer {await get_service_authenticator().token()}"}
    async with httpx.AsyncClient(timeout=30.0, headers=headers) as http:
        resp = await http.get(f"{base}/internal/apps")
        resp.raise_for_status()
        apps = [
            a
            for a in resp.json()
            if isinstance(a, dict) and a.get("key") == W2V_APP_KEY and not a.get("revoked")
        ]
        if len(apps) != 1:
            raise RuntimeError(f"expected exactly one live {W2V_APP_KEY!r} app, found {len(apps)}")
        resp = await http.get(
            f"{base}/internal/licenses",
            params={"app_id": apps[0]["id"], "status": "active", "effective": "true"},
        )
        resp.raise_for_status()
        rows = resp.json()
    if not isinstance(rows, list):
        raise RuntimeError("licences /internal/licenses did not answer a JSON list")
    licensed: dict[UUID, tuple[str, UUID]] = {}
    for lic in rows:
        if lic.get("effective_status") != "active":
            continue
        tenant, lic_id = UUID(str(lic["tenant_id"])), UUID(str(lic["id"]))
        # Several active licences per tenant (one per user) is normal; keep the
        # most recently started one as the tenant's reference.
        key = lic.get("starts_at") or ""
        if tenant not in licensed or key > licensed[tenant][0]:
            licensed[tenant] = (key, lic_id)
    return {t: lic for t, (_, lic) in licensed.items()}


def _kc_identity_reader() -> IdentityReader:
    from src.clients.keycloak_admin import get_keycloak_admin

    # Users live in KEYCLOAK_USER_REALM (default KEYCLOAK_REALM): same admin client as approve.
    kc = get_keycloak_admin()

    async def read(subject_id: UUID) -> KcIdentity:
        user = await kc.get_user(str(subject_id))
        if user is None:
            return KcIdentity(False)
        names = await kc.get_user_effective_realm_roles(str(subject_id))
        return identity_from_kc_user({"attributes": user.attributes or {}}, names)

    return read


async def _main(args: argparse.Namespace) -> int:
    from src.database.session import get_session_maker

    try:
        licensed = (
            licensed_from_file(args.licensed_tenants_file)
            if args.licensed_tenants_file
            else await _licensed_from_service()
        )
        reader = _kc_identity_reader()
        return await run(await get_session_maker(), reader, licensed, apply=args.apply)
    except Exception as exc:  # noqa: BLE001 - CLI boundary: report, exit 2, write nothing
        print(f"ABORTED, nothing written: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--apply", action="store_true", help="write the changes (default: dry run)")
    parser.add_argument(
        "--licensed-tenants-file",
        type=Path,
        help="JSON {tenant_id: licence_id}; replaces the licenses-service lookup",
    )
    raise SystemExit(asyncio.run(_main(parser.parse_args())))
