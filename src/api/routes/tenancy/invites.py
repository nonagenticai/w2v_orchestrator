"""Tenant invites (P2 WP4a).

Tenant-admin surface (scoped to the EFFECTIVE tenant):

- POST   /api/tenant/invites               invite an email with a role → one-time link
- GET    /api/tenant/invites               pending (or all) invites
- POST   /api/tenant/invites/{id}/resend   rotate the token + expiry (old link dies)
- DELETE /api/tenant/invites/{id}          revoke

Invitee surface (any authenticated JWT caller, membership or not):

- POST   /api/invites/accept {token}       join the invite's tenant (the emailed link)
- POST   /api/invites/{id}/accept          join a pending invite listed in
                                           ``/api/auth/me`` ``pending_invites``

``/api/auth/me`` only LISTS pending invites; joining a tenant is always this
explicit call (owner decision — no auto-accept on login).

The invite token is 32 random bytes (urlsafe); only its SHA-256 is stored and
the raw token appears only in the create/resend response (inside
``invite_url``). Identity: w2v holds no Keycloak write power. Provisioning's
identity invite API (``src/clients/identity_inviter.py``) finds the invitee's
account by email or creates it unverified WITHOUT attributes, credentials or
roles (w2v membership is w2v's own table), and sends the Keycloak action email
the account needs: ``UPDATE_PASSWORD`` (+ ``VERIFY_EMAIL``) when it has no
password, ``VERIFY_EMAIL`` when only unverified, none when it is ready. The
email carries NO invite token: it redirects to the SPA root, which lists the
signed-in user's pending invites (``/api/auth/me``) and accepts by id with the
verified email. The token-bearing ``invite_url`` is only in the admin's
response (copyable); a resend rotates it (the old LINK dies), while every
identity email ever sent stays valid.

None of that is observable to the inviting tenant_admin: the realm is shared
with fr/kag, so "does this address have an account, is it verified" must not
leak through an invite response — and w2v never learns it: provisioning answers
every accepted invite with the same ``202 {"accepted": true}``. The response
reports only ``email_delivery: "requested" | "failed"`` and its shape never
varies.
(``tenant_invites.kc_user_created`` is no longer written; it stays false.)

Ordering on create: the invite row is inserted FIRST (pending,
``kc_user_created=false``) — a concurrent duplicate hits the partial unique
index and is answered with the existing invite before Keycloak is touched —
then the row + audit COMMIT, and only then is the identity invite requested
(account + email in one provisioning call). An identity failure after the
commit leaves the invite standing and answers ``email_delivery: "failed"``.

Acceptance requires a VERIFIED email equal (case-insensitively) to the invite's
email: the token proves the link was received, the verified email proves the
Keycloak account that redeems it belongs to the invitee. The by-id path has no
token, so the verified email is the whole proof — and an invite addressed to
someone else answers 404 there (an id must not confirm another person's invite).
"""

from __future__ import annotations

import datetime
import logging
from typing import Literal
from urllib.parse import urlencode
from uuid import UUID

from fastapi import APIRouter, Query, Request, Response, status
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.routes._common import (
    LICENSE_GATE_ONLY,
    AuditRepoDep,
    IdentityInviterDep,
    MembersRepoDep,
    SessionDep,
    SubjectIdDep,
    TenantAdminDep,
)
from src.api.routes.tenancy.errors import TenancyError
from src.api.schemas.tenancy import (
    EmailDelivery,
    InviteAccept,
    InviteAccepted,
    InviteCreate,
    InviteCreated,
    InviteList,
    InviteOut,
)
from src.clients.identity_inviter import IdentityInviteError, IdentityInviter, tenant_context
from src.config import get_settings
from src.database.models import Tenant, TenantInvite, TenantMember
from src.database.repositories.invites_repository import (
    InvitesRepository,
    hash_invite_token,
    new_invite_token,
)
from src.database.repositories.members_repository import MembersRepository

router = APIRouter(prefix="/api/tenant/invites", tags=["tenancy"])
accept_router = APIRouter(prefix="/api/invites", tags=["tenancy"])

_log = logging.getLogger("w2v_orchestrator.tenancy")

INVITE_TTL = datetime.timedelta(days=7)

INVITE_NOT_FOUND = "INVITE_NOT_FOUND"
INVITE_EMAIL_MISMATCH = "INVITE_EMAIL_MISMATCH"
INVITE_GONE = "INVITE_GONE"
EMAIL_NOT_VERIFIED = "EMAIL_NOT_VERIFIED"


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def invite_url(token: str) -> str:
    base = get_settings().provisioning.w2v_frontend_url.rstrip("/")
    return f"{base}/invite/accept?{urlencode({'token': token})}"


def _out(inv: TenantInvite) -> InviteOut:
    return InviteOut(
        id=inv.id,
        email=inv.email,
        role=inv.role,  # ty: ignore[invalid-argument-type]  # why: DB CHECK constrains it to the Literal
        status=inv.status,  # ty: ignore[invalid-argument-type]  # why: DB CHECK constrains it to the Literal
        expires_at=inv.expires_at,
        created_at=inv.created_at,
        resent_at=inv.resent_at,
        invited_by=inv.invited_by,
    )


def _error(code: int, error: str, message: str) -> TenancyError:
    return TenancyError(code, error, message)


async def _ensure_tenant_row(session: AsyncSession, tenant_id: UUID) -> None:
    """``tenant_invites.tenant_id`` is a FK to ``tenants``. A pre-P2 tenant may
    still lack its registry row (the backfill creates them); register it rather
    than 500 on the FK. Never touches an existing row."""
    await session.execute(
        pg_insert(Tenant)
        .values(id=tenant_id, created_via="backfill")
        .on_conflict_do_nothing(index_elements=[Tenant.id])
    )


async def _request_identity_email(
    inviter: IdentityInviter, inv: TenantInvite, actor: UUID, *, op: str
) -> EmailDelivery:
    """Ask provisioning for the invitee's identity + action email. Called only
    AFTER the invite is committed, so a failure never loses the invite: it is
    logged and reported as ``"failed"`` (the admin-copyable ``invite_url`` still
    works, and a resend retries). Success is ``"requested"`` — provisioning's
    answer is a uniform 202, so w2v never knows (or reports) whether an email
    actually went out.

    The email carries NO invite token: its redirect is the SPA root (the
    inviter's default), whose layout lists the signed-in user's pending invites
    (``/api/auth/me`` → ``POST /api/invites/{id}/accept`` by verified email). So
    an older email stays valid across resends."""
    try:
        await inviter.invite(inv.email, actor=actor, context=tenant_context(inv.tenant_id))
    except IdentityInviteError as exc:
        _log.warning(
            "invite %s: identity invite failed invite=%s code=%s upstream=%s retry_after=%s",
            op,
            inv.id,
            exc.code,
            exc.upstream_status,
            exc.retry_after,
        )
        return "failed"
    return "requested"


def _created(inv: TenantInvite, token: str, delivery: EmailDelivery) -> InviteCreated:
    return InviteCreated(
        **_out(inv).model_dump(), invite_url=invite_url(token), email_delivery=delivery
    )


def _pending_conflict(pending: TenantInvite | None) -> TenancyError:
    return TenancyError(
        409,
        "INVITE_PENDING",
        "A pending invite exists for this email; resend it instead.",
        {"invite_id": str(pending.id)} if pending is not None else None,
    )


# ─── tenant-admin surface ────────────────────────────────────────────────


@router.post(
    "",
    response_model=InviteCreated,
    status_code=status.HTTP_201_CREATED,
    operation_id="tenant.invites.create",
    dependencies=LICENSE_GATE_ONLY,
)
async def create_invite(
    body: InviteCreate,
    tenant_id: TenantAdminDep,
    actor: SubjectIdDep,
    audit: AuditRepoDep,
    session: SessionDep,
    inviter: IdentityInviterDep,
) -> InviteCreated:
    email = body.email
    already = (
        await session.execute(
            select(func.count())
            .select_from(TenantMember)
            .where(
                TenantMember.tenant_id == tenant_id,
                TenantMember.status == "active",
                func.lower(TenantMember.email) == email.lower(),
            )
        )
    ).scalar_one()
    if already:
        raise _error(409, "ALREADY_MEMBER", "This email is already an active member.")
    invites = InvitesRepository(session)
    pending = await invites.get_pending_for_tenant_email(tenant_id, email)
    if pending is not None:
        raise _pending_conflict(pending)

    # 1. Claim the (tenant, email) pending slot BEFORE Keycloak. A concurrent
    #    create for the same email blocks on the partial unique index until we
    #    finish, then loses here — without ever creating a Keycloak user.
    token, token_hash = new_invite_token()
    await _ensure_tenant_row(session, tenant_id)
    try:
        async with session.begin_nested():
            inv = await invites.create(
                tenant_id=tenant_id,
                email=email,
                role=body.role,
                token_hash=token_hash,
                expires_at=_now() + INVITE_TTL,
                invited_by=actor,
                kc_user_created=False,
            )
    except IntegrityError as exc:  # a concurrent invite for the same email won
        existing = await invites.get_pending_for_tenant_email(tenant_id, email)
        conflict = _pending_conflict(existing)  # read .id BEFORE rollback expires it
        await session.rollback()
        raise conflict from exc

    await audit.record(
        actor_subject_id=actor,
        action="member.invite",
        target_tenant_id=tenant_id,
        detail={"invite_id": str(inv.id), "email": email, "role": body.role},
    )
    await session.commit()
    await session.refresh(inv)
    # 2. Only now, with the invite durable: the identity + action email.
    delivery = await _request_identity_email(inviter, inv, actor, op="create")
    return _created(inv, token, delivery)


@router.get(
    "",
    response_model=InviteList,
    operation_id="tenant.invites.list",
    dependencies=LICENSE_GATE_ONLY,
)
async def list_invites(
    tenant_id: TenantAdminDep,
    session: SessionDep,
    invite_status: Literal["pending", "accepted", "revoked", "expired", "all"] = Query(
        "pending", alias="status"
    ),
) -> InviteList:
    rows = await InvitesRepository(session).list_for_tenant(
        tenant_id, status=None if invite_status == "all" else invite_status
    )
    return InviteList(items=[_out(r) for r in rows])


@router.post(
    "/{invite_id}/resend",
    response_model=InviteCreated,
    operation_id="tenant.invites.resend",
    dependencies=LICENSE_GATE_ONLY,
)
async def resend_invite(
    invite_id: UUID,
    tenant_id: TenantAdminDep,
    actor: SubjectIdDep,
    audit: AuditRepoDep,
    session: SessionDep,
    inviter: IdentityInviterDep,
) -> InviteCreated:
    invites = InvitesRepository(session)
    inv = await invites.get(invite_id, tenant_id)
    if inv is None:
        raise _error(404, INVITE_NOT_FOUND, "invite not found in this workspace")
    token, token_hash = new_invite_token()
    if not await invites.resend(
        invite_id, tenant_id, token_hash=token_hash, expires_at=_now() + INVITE_TTL
    ):
        raise _error(409, "INVITE_NOT_PENDING", f"invite is {inv.status}, not pending")
    await audit.record(
        actor_subject_id=actor,
        action="member.invite",
        target_tenant_id=tenant_id,
        detail={"invite_id": str(invite_id), "email": inv.email, "resend": True},
    )
    await session.commit()
    await session.refresh(inv)
    delivery = await _request_identity_email(inviter, inv, actor, op="resend")
    return _created(inv, token, delivery)


@router.delete(
    "/{invite_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    operation_id="tenant.invites.revoke",
    dependencies=LICENSE_GATE_ONLY,
)
async def revoke_invite(
    invite_id: UUID,
    tenant_id: TenantAdminDep,
    actor: SubjectIdDep,
    audit: AuditRepoDep,
    session: SessionDep,
) -> Response:
    invites = InvitesRepository(session)
    inv = await invites.get(invite_id, tenant_id)
    if inv is None:
        raise _error(404, INVITE_NOT_FOUND, "invite not found in this workspace")
    if not await invites.revoke(invite_id, tenant_id):
        raise _error(409, "INVITE_NOT_PENDING", f"invite is {inv.status}, not pending")
    await audit.record(
        actor_subject_id=actor,
        action="invite.revoke",
        target_tenant_id=tenant_id,
        detail={"invite_id": str(invite_id), "email": inv.email},
    )
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ─── invitee surface ─────────────────────────────────────────────────────


def verified_email(request: Request) -> str | None:
    """The caller's email iff Keycloak marked it verified, else None."""
    email = getattr(request.state, "email", None)
    if not email or getattr(request.state, "email_verified", False) is not True:
        return None
    return str(email).strip()


async def accept_invite_row(
    session: AsyncSession,
    inv: TenantInvite,
    subject_id: UUID,
    *,
    display_name: str | None,
) -> tuple[str, bool] | None:
    """Bind ``subject_id`` to a PENDING, unexpired invite whose email the caller
    has already proven. Returns ``(role, already_member)``, or ``None`` when the
    invite is no longer pending (a concurrent revoke/accept won) — in which case
    NO membership is written. An existing ACTIVE membership keeps its role (an
    invite never promotes or demotes); a removed one is reactivated with the
    invite's role. No commit: the pending→accepted flip and the membership
    write land in the caller's one transaction, or neither does.

    The flip goes FIRST and gates the membership: ``mark_accepted`` is a
    conditional ``UPDATE … WHERE status='pending'``, so it is the single
    authority on whether this caller won the invite."""
    if not await InvitesRepository(session).mark_accepted(inv.id, subject_id):
        return None
    members = MembersRepository(session)
    current = await members.get_active_membership(subject_id, inv.tenant_id)
    if current is None:
        await members.add_member(
            subject_id=subject_id,
            tenant_id=inv.tenant_id,
            role=inv.role,  # ty: ignore[invalid-argument-type]  # why: DB CHECK constrains it to the Literal
            invited_by=inv.invited_by,
            email=inv.email,
            display_name=display_name,
        )
        return inv.role, False
    return current.role, True


async def _accept_locked(
    session: AsyncSession,
    request: Request,
    inv: TenantInvite,
    subject_id: UUID,
    members: MembersRepository,
) -> InviteAccepted:
    """Accept ``inv`` (already row-locked FOR UPDATE, email already proven) for
    ``subject_id``: idempotent replay by the same identity, 410 when it is no
    longer pending/unexpired, else the ``mark_accepted``-gated membership
    write. Commits on success."""
    tenant_name = (
        await session.execute(select(Tenant.name).where(Tenant.id == inv.tenant_id))
    ).scalar_one_or_none()

    if inv.status == "accepted" and inv.accepted_subject_id == subject_id:
        # Idempotent replay by the same identity.
        current = await members.get_active_membership(subject_id, inv.tenant_id)
        if current is not None:
            return InviteAccepted(
                tenant_id=inv.tenant_id,
                tenant_name=tenant_name,
                role=current.role,  # ty: ignore[invalid-argument-type]  # why: DB CHECK
                already_member=True,
            )
        raise _error(410, INVITE_GONE, "This invite was used, and the membership has ended.")
    if inv.status != "pending" or inv.expires_at <= _now():
        raise _error(410, INVITE_GONE, f"This invite is no longer valid ({_gone_reason(inv)}).")

    accepted = await accept_invite_row(
        session, inv, subject_id, display_name=getattr(request.state, "name", None)
    )
    if accepted is None:  # unreachable while the FOR UPDATE holds; never trust it
        await session.rollback()
        raise _error(410, INVITE_GONE, "This invite is no longer valid.")
    role, already = accepted
    await session.commit()
    _log.info("invite accepted invite=%s subject=%s tenant=%s", inv.id, subject_id, inv.tenant_id)
    return InviteAccepted(
        tenant_id=inv.tenant_id,
        tenant_name=tenant_name,
        role=role,  # ty: ignore[invalid-argument-type]  # why: DB CHECK constrains it to the Literal
        already_member=already,
    )


@accept_router.post("/accept", response_model=InviteAccepted, operation_id="invites.accept")
async def accept_invite(
    body: InviteAccept,
    request: Request,
    subject_id: SubjectIdDep,
    session: SessionDep,
    members: MembersRepoDep,
) -> InviteAccepted:
    inv = (
        await session.execute(
            select(TenantInvite)
            .where(TenantInvite.token_hash == hash_invite_token(body.token))
            .with_for_update()
        )
    ).scalar_one_or_none()
    if inv is None:
        raise _error(404, INVITE_NOT_FOUND, "This invite link is not valid.")
    email = verified_email(request)
    if email is None:
        raise _error(403, EMAIL_NOT_VERIFIED, "Verify your email address to accept invites.")
    if email.lower() != inv.email.strip().lower():
        raise _error(
            403, INVITE_EMAIL_MISMATCH, "This invite was sent to a different email address."
        )
    return await _accept_locked(session, request, inv, subject_id, members)


@accept_router.post(
    "/{invite_id}/accept", response_model=InviteAccepted, operation_id="invites.accept_by_id"
)
async def accept_invite_by_id(
    invite_id: UUID,
    request: Request,
    subject_id: SubjectIdDep,
    session: SessionDep,
    members: MembersRepoDep,
) -> InviteAccepted:
    """Join an invite listed in ``/api/auth/me`` ``pending_invites``. The
    caller's VERIFIED email must equal the invite's; any other invite id —
    unknown, or addressed to someone else — is the same 404."""
    email = verified_email(request)
    if email is None:
        raise _error(403, EMAIL_NOT_VERIFIED, "Verify your email address to accept invites.")
    inv = (
        await session.execute(
            select(TenantInvite).where(TenantInvite.id == invite_id).with_for_update()
        )
    ).scalar_one_or_none()
    if inv is None or email.lower() != inv.email.strip().lower():
        await session.rollback()  # release the row lock on someone else's invite
        raise _error(404, INVITE_NOT_FOUND, "invite not found")
    return await _accept_locked(session, request, inv, subject_id, members)


def _gone_reason(inv: TenantInvite) -> str:
    if inv.status == "pending":
        return "expired"
    return inv.status
