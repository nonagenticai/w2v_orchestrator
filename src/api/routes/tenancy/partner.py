"""Partner tenant creation (P2 tenancy, WP4b): ``POST /api/tenants``.

LICENCE LANE ONLY. The caller authenticates with a w2v licence token
(w2v_common introspects it and stashes the verdict on ``request.state.license``)
and is a partner iff that licence is

* ACTIVE (``derive_active_state`` → an entitled state — the same derivation the
  licence gate uses, so expiry handling cannot drift),
* a ``w2v`` licence, and
* listed in the operator allowlist ``W2V_PARTNER_LICENSE_IDS`` (owner decision:
  not every customer licence may mint tenants).

Anything else — a Keycloak JWT, a tenant API key, an inactive/foreign/unlisted
licence — is 403. The tenant id is NOT taken from the caller's licence; it is
the body's ``tenant_id`` (idempotency key) or a fresh one.

Because the partner CHOOSES ``tenant_id``, this is a takeover surface. The
owner rule (``tenant_provisioning.create_tenant``): a partner may create an id
with no registry row and NO projects/members, replay one it already holds
(``partner_licence_id`` = its licence → 200 + the existing pending invite), or
adopt an empty ``created_via='backfill'`` row with no partner. Anything else —
another creator's tenant, or a licensed tenant that has data but no registry
row — is 409 ``TENANT_ID_UNAVAILABLE``. First creation / adoption is 201.

Licence — by INTROSPECTION (``src/services/tenant_licence.py``; no the licences internal routes
call, which an off-cluster ingress refuses):

* ``licence_token`` given → introspected: it must be ACTIVE, of THIS
  ``tenant_id`` and of app ``w2v``, else 422 ``LICENCE_TOKEN_MISMATCH``. Holding
  the token PROVES the licence, so any allow-listed partner may then claim a
  licensed (but otherwise empty) id. It is stored SEALED and its licence becomes
  ``licence_ref``. A replay with a DIFFERENT valid token replaces the stored one
  through the rotation path (lease + CAS: 409 ``licence_busy`` /
  ``licence_changed_concurrently``); the same token again is a no-op. A replaced
  token is dropped, never revoked (licences has no single-token revoke).
* no token → ``GET /licenses/active`` (P9 review blocker still applies): the
  licences service records no issuer/owner on a licence, so a THIRD-PARTY partner
  cannot be shown to own a live w2v licence on a tenant id — such an id is 409
  ``TENANT_ID_UNAVAILABLE`` (except its own replay). A FIRST-PARTY partner
  (``W2V_PARTNER_FIRST_PARTY_LICENSE_IDS``) MAY claim it.
* no token, ANY partner → after the w2v row is committed, the tenant's own
  ``w2v`` bundle is ENSURED in licences (``license_provisioning``, held by
  ``svc:<tenant>:w2v``; idempotent ``exists`` / ``adopt`` / ``insert``) and its
  w2v licence becomes ``licence_ref`` — a created tenant is never left with a
  null ``licence_ref`` (live 2026-10-15: one was, and was unusable with
  ``W2V_LICENSE_AUTOISSUE_ENABLED=false``). A refused app is 409
  ``LICENCE_REFUSED`` (``details.reason``, e.g. ``member_licence`` /
  ``several_candidates``); the row stays and a replay retries the ensure. This
  uses licences' ``/internal`` bundle-ensure route (as ``/api/admin/provision``
  does), so it needs an in-cluster licences reach.

Licences unreachable → 503 ``LICENCES_UNAVAILABLE`` + Retry-After (could-not-check
is never a yes); a definite 4xx refusal from licences or its gateway → 502
``LICENCES_REFUSED``, no Retry-After (a configuration fault).

``PUT /api/tenants/{tenant_id}/licence-token`` ``{licence_token}`` rotates / hands
over the stored token: the partner bound to the tenant, or a super_admin. Same
verification, lease + CAS, audit ``tenant.licence_token_set`` (never the token).

Licences REGISTRATION (licences migration 050 — tenants are owned per app): right
before the w2v row is created (create, replay and redrive alike) the tenant is
registered in licences with ``PUT /tenants/{tenant_id}`` ``{name}`` as the
w2v-backend SA, so licences records w2v as its owning app (the owner is derived
from the SA, never sent). The licences name is the slug of ``name`` (licences'
``^[a-z0-9][a-z0-9-]*$``; ``tenant-<id>`` when nothing slugs). 201 = w2v now owns
it; 200 = already w2v's, or another app's tenant on which w2v holds a grant (e.g.
a merqi tenant after merqi ensured its ``w2v`` bundle) — unchanged. Refusals:
409 ``TENANT_OWNED_BY_OTHER_APP`` (another app owns the id and has not granted
w2v), 409 ``LICENCES_TENANT_NAME_TAKEN`` (w2v already owns a different licences
tenant with that name), 502 ``LICENCES_REFUSED`` (a 403 — this SA is not an
owning app / lacks the role), 503 ``LICENCES_UNAVAILABLE``. Nothing is created
in w2v when registration fails.

The response never carries the invitee's Keycloak user id. ``admin_email`` is
optional (P9-4): absent → no invite, the tenant is run by the partner's keys.

P9-3 — the partner also MINTS MACHINE KEYS for the tenants it manages:

* ``POST /api/tenants/{tenant_id}/api-keys`` ``{name, scopes}`` → 201 with the
  ``w2v_sk_…`` secret, returned ONCE (never listed, never recoverable).
* ``DELETE /api/tenants/{tenant_id}/api-keys/{key_id}`` → 204; only keys THIS
  partner minted (a tenant_admin's key is a 404 to the partner).

Gate, in order: an allow-listed active partner licence (else 403); the tenant
exists and is not deleted (else **404** ``TENANT_NOT_FOUND``); its
``partner_licence_id`` is the caller's (else **403** ``TENANT_NOT_PARTNER_BOUND``).
404-vs-403 is deliberate: tenant EXISTENCE is not a secret from a partner —
``POST /api/tenants`` already answers 409 ``TENANT_ID_UNAVAILABLE`` for any taken
id — so distinguishing "no such tenant" from "not yours" leaks nothing new and
tells the partner's operator which fix applies (provision vs. super_admin bind).

Idempotency is by ``name``: a second mint while a LIVE key of that name exists
(any creator) is 409 ``API_KEY_NAME_EXISTS``. Only when THIS partner minted that
key does ``error.details`` carry its METADATA (id, prefix, scopes, created_at,
created_via) — never a secret; a key of anyone else (a tenant_admin's) is a bare
409. A caller that lost the secret revokes ``details.key_id`` and mints again.
The tenant row is held ``FOR SHARE`` while minting, so an offboarding delete
cannot interleave with a mint.
The partial unique index ``uq_tenant_api_keys_partner_live_name`` closes the
concurrent-mint race (IntegrityError → the same 409). Audit ``apikey.create`` /
``apikey.revoke``, actor = the partner licence's subject.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from uuid import UUID, uuid7

from fastapi import APIRouter, HTTPException, Request, Response, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.routes._common import (
    SUPER_ADMIN,
    W2V_APP_KEY,
    AuditRepoDep,
    IdentityInviterDep,
    LicenceKeyringDep,
    LicensesClientDep,
    SessionDep,
    _roles,
)
from src.api.routes.tenancy.api_keys import api_key_summary
from src.api.routes.tenancy.errors import (
    TenancyError,
    identity_error,
    licences_refused,
    tenant_not_onboarded,
)
from src.api.routes.tenancy.licence_token import (
    install_token,
    keyring_unavailable,
    licences_unavailable,
    verify_licence_token,
)
from src.api.schemas.tenancy_keys import (
    ApiKeyCreated,
    PartnerApiKeyCreate,
    PartnerTenantCreate,
    PartnerTenantResult,
    TenantLicenceTokenResult,
    TenantLicenceTokenSet,
)
from src.clients.identity_inviter import IdentityInviteError
from src.clients.licenses_client import (
    LicencesTenantNameTakenError,
    LicencesTenantOwnedByOtherAppError,
)
from src.config import get_settings
from src.database.models import Tenant, TenantApiKey
from src.database.repositories.admin_repository import AdminRepository
from src.database.repositories.api_keys_repository import ApiKeysRepository
from src.database.session import bind_session_tenant
from src.services.license_provisioning import (
    W2V_LICENCE_APP_KEY,
    TenantLicenceRefusedError,
    ensure_active_license,
    record_ensured_licence_ref,
)
from src.services.license_status import ENTITLED_STATES, derive_active_state
from src.services.tenant_licence import (
    FallbackVerdict,
    LicencesRefusedError,
    LicencesTenantNotOnboardedError,
    LicencesUnavailableError,
    TokenVerdict,
    licences_error,
    tenant_active_licence,
)
from src.services.tenant_licence_install import Installed
from src.services.tenant_provisioning import TenantConflictError, create_tenant, invite_user

router = APIRouter(prefix="/api/tenants", tags=["tenancy"])

_log = logging.getLogger("w2v_orchestrator.tenancy")

TENANT_ID_UNAVAILABLE = "TENANT_ID_UNAVAILABLE"
TENANT_OWNED_BY_OTHER_APP = "TENANT_OWNED_BY_OTHER_APP"
LICENCES_TENANT_NAME_TAKEN = "LICENCES_TENANT_NAME_TAKEN"
TENANT_NOT_FOUND = "TENANT_NOT_FOUND"
TENANT_NOT_PARTNER_BOUND = "TENANT_NOT_PARTNER_BOUND"
API_KEY_NAME_EXISTS = "API_KEY_NAME_EXISTS"  # pragma: allowlist secret — an error code
API_KEY_NOT_FOUND = "API_KEY_NOT_FOUND"  # pragma: allowlist secret — an error code


def _uuid(raw: object) -> UUID | None:
    try:
        return UUID(str(raw)) if raw else None
    except ValueError:
        return None


def partner_licence(request: Request) -> tuple[UUID, UUID | None]:
    """``(licence_id, licence_subject)`` of an allow-listed, active w2v partner
    licence on this request — else 403. Fail-closed on every unknown shape."""
    detail = getattr(request.state, "license", None)
    if not isinstance(detail, dict) or not detail.get("active"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "partner licence required")
    app = detail.get("app")
    licence_id = _uuid(detail.get("license_id"))
    if not isinstance(app, dict) or app.get("key") != W2V_APP_KEY or licence_id is None:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "partner licence required")
    if derive_active_state(detail, datetime.now(UTC)).state not in ENTITLED_STATES:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "partner licence is not active")
    if licence_id not in get_settings().w2v_partner_license_ids:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "licence is not a partner licence")
    return licence_id, _uuid(detail.get("subject_id"))


@router.post(
    "",
    response_model=PartnerTenantResult,
    status_code=status.HTTP_201_CREATED,
    operation_id="tenants.partner_create",
    responses={200: {"model": PartnerTenantResult, "description": "idempotent replay"}},
)
async def partner_create_tenant(
    request: Request,
    response: Response,
    body: PartnerTenantCreate,
    inviter: IdentityInviterDep,
    licences: LicensesClientDep,
    keyring: LicenceKeyringDep,
    audit: AuditRepoDep,
    session: SessionDep,
) -> PartnerTenantResult:
    licence_id, licence_subject = partner_licence(request)
    tenant_id = body.tenant_id or uuid7()
    first_party = licence_id in get_settings().w2v_partner_first_party_license_ids
    existing = await AdminRepository(session).get_tenant(tenant_id)
    replay = existing is not None and existing.partner_licence_id == licence_id
    token = body.licence_token.get_secret_value() if body.licence_token is not None else None
    # Every licences call happens BEFORE create_tenant takes the tenant lock —
    # or after its commit (the licence ensure below).
    verdict, _ = await _create_licence_check(
        licences, keyring, tenant_id, token, first_party=first_party, replay=replay
    )
    await register_in_licences(licences, tenant_id, body.name)
    try:
        creation = await create_tenant(
            session,
            tenant_id=tenant_id,
            name=body.name,
            created_via="partner",
            actor_subject_id=licence_subject,
            partner_licence_id=licence_id,
        )
    except TenantConflictError as exc:
        await session.rollback()
        raise TenancyError(
            409, TENANT_ID_UNAVAILABLE, "This tenant id is not available to this partner."
        ) from exc
    invite_id, already_member = None, False
    if body.admin_email is not None:  # P9-4: no email → no human, no invite
        try:
            invite = await invite_user(
                session,
                inviter,
                tenant_id=tenant_id,
                email=body.admin_email,
                role="tenant_admin",
                invited_by=licence_subject,
            )
        except IdentityInviteError as exc:
            await session.rollback()
            raise identity_error(exc) from exc
        invite_id, already_member = invite.invite_id, invite.already_member
    licence_ref = creation.tenant.licence_ref
    name = creation.tenant.name
    await session.commit()
    if token is None:
        # No token: ensure the tenant's own w2v bundle (held by svc:<tenant>:w2v,
        # idempotent — exists/adopt/insert) and record its w2v licence as
        # licence_ref, AFTER the commit so no tenant lock spans the licences call.
        # Never a silent null: a refusal is a typed error, and a replay retries
        # the ensure (the registry row from this call stays; it is idempotent).
        licence_ref = await _ensure_licence_ref(session, licences, tenant_id)
        await session.commit()
    elif verdict is not None:
        # Create, replay and redrive all go through the rotation path: a NEW token
        # replaces the stored one under lease + CAS; the same token is a no-op.
        installed = await install_token(
            session,
            keyring,
            tenant_id,
            token,
            verdict,
            on_swapped=_audit_token_set(audit, tenant_id, licence_subject, "partner"),
        )
        licence_ref = installed.licence_ref
    if not creation.created:
        response.status_code = status.HTTP_200_OK
    return PartnerTenantResult(
        tenant_id=tenant_id,
        name=name,
        created=creation.created,
        invite_id=invite_id,
        already_member=already_member,
        licence_ref=licence_ref,
    )


LICENCE_REFUSED = "LICENCE_REFUSED"


async def _ensure_licence_ref(
    session: AsyncSession, licences: LicensesClientDep, tenant_id: UUID
) -> UUID:
    """:func:`ensure_tenant_licence_ref`, with licences' answers mapped: an app
    refused → 409 ``LICENCE_REFUSED`` (``details.reason`` = licences' code, e.g.
    ``member_licence``); not onboarded → 403; a definite 4xx → 502
    ``LICENCES_REFUSED``; unreachable → 503 ``LICENCES_UNAVAILABLE``."""
    try:
        licence_id = await ensure_active_license(
            licences, app_key=W2V_LICENCE_APP_KEY, tenant_id=tenant_id
        )
    except TenantLicenceRefusedError as exc:
        _log.error(
            "partner create: licences refused tenant %s its %s licence: %s",
            tenant_id,
            exc.app_key,
            exc.reason,
        )
        raise TenancyError(
            409,
            LICENCE_REFUSED,
            f"The licences service refused this tenant's {exc.app_key!r} licence "
            f"({exc.reason}); the tenant exists but is not entitled. An operator must "
            "resolve the refusal, then replay this request.",
            details={"tenant_id": str(tenant_id), "app": exc.app_key, "reason": exc.reason},
        ) from exc
    except Exception as exc:
        failure = licences_error(exc)
        if isinstance(failure, LicencesTenantNotOnboardedError):
            raise tenant_not_onboarded() from exc
        if isinstance(failure, LicencesRefusedError):
            raise licences_refused(failure.status_code) from exc
        raise licences_unavailable(
            "Could not ensure this tenant's Studio licence with the licences service; retry."
        ) from exc
    try:
        return await record_ensured_licence_ref(session, tenant_id, licence_id)
    except TenantLicenceRefusedError as exc:
        raise TenancyError(
            502, LICENCE_REFUSED, "The licences service returned a malformed licence id."
        ) from exc


_NON_SLUG = re.compile(r"[^a-z0-9]+")


def licences_tenant_name(name: str, tenant_id: UUID) -> str:
    """The licences registry name for a partner tenant: ``name`` ASCII-folded and
    slugged to licences' ``^[a-z0-9][a-z0-9-]*$`` (max 255); ``tenant-<id>`` when
    nothing usable is left (e.g. a non-Latin name)."""
    folded = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    slug = _NON_SLUG.sub("-", folded.lower()).strip("-")[:255].rstrip("-")
    return slug or f"tenant-{tenant_id}"


async def register_in_licences(licences: LicensesClientDep, tenant_id: UUID, name: str) -> None:
    """Register ``tenant_id`` in licences as a w2v-owned tenant (idempotent) —
    before anything is written in w2v. See the module docstring for the answers."""
    licences_name = licences_tenant_name(name, tenant_id)
    try:
        await licences.register_tenant(tenant_id, name=licences_name)
    except LicencesTenantOwnedByOtherAppError as exc:
        _log.error(
            "licences: tenant %s is owned by another app and w2v holds no grant — "
            "partner create refused, not retried",
            tenant_id,
        )
        raise TenancyError(
            409,
            TENANT_OWNED_BY_OTHER_APP,
            "This tenant id belongs to another app in the licences service, which has "
            "not granted w2v access to it.",
        ) from exc
    except LicencesTenantNameTakenError as exc:
        raise TenancyError(
            409,
            LICENCES_TENANT_NAME_TAKEN,
            "w2v already has a different tenant with this name in the licences "
            "service; choose another name.",
            details={"licences_name": licences_name},
        ) from exc
    except Exception as exc:
        failure = licences_error(exc)
        if isinstance(failure, LicencesTenantNotOnboardedError):
            raise tenant_not_onboarded() from exc
        if isinstance(failure, LicencesRefusedError):
            raise licences_refused(failure.status_code) from exc
        raise licences_unavailable(
            "Could not register this tenant with the licences service; retry."
        ) from exc


async def _create_licence_check(
    licences: LicensesClientDep,
    keyring: LicenceKeyringDep,
    tenant_id: UUID,
    token: str | None,
    *,
    first_party: bool,
    replay: bool,
) -> tuple[TokenVerdict | None, UUID | None]:
    """``(verdict, fallback_ref)`` for a create: the token's ENTITLED verdict when
    one was sent (422 / 503 / 502 otherwise); else the fallback, which 409s a
    third party off a licensed id and yields a first party's single licence."""
    if token is not None:
        if not keyring.configured:
            raise keyring_unavailable()
        return await verify_licence_token(licences, tenant_id, token), None
    if not first_party and replay:
        return None, None
    fallback = await _active_licence(licences, tenant_id)
    if fallback.entitled and not first_party and not replay:
        raise TenancyError(
            409, TENANT_ID_UNAVAILABLE, "This tenant id is not available to this partner."
        )
    return None, fallback.licence_ref if first_party else None


def _audit_token_set(
    audit: AuditRepoDep, tenant_id: UUID, actor: UUID | None, by: str
) -> Callable[[Installed], Awaitable[None]]:
    """``on_swapped`` hook: one ``tenant.licence_token_set`` row in the swap's
    transaction. Records the licence ids — never the token."""

    async def record(installed: Installed) -> None:
        await audit.record(
            actor_subject_id=actor,
            action="tenant.licence_token_set",
            target_tenant_id=tenant_id,
            detail={
                "by": by,
                "licence_ref": str(installed.licence_ref) if installed.licence_ref else None,
                "previous_licence_ref": (
                    str(installed.previous_licence_ref) if installed.previous_licence_ref else None
                ),
            },
        )

    return record


async def _active_licence(licences: LicensesClientDep, tenant_id: UUID) -> FallbackVerdict:
    """The no-token fallback; 502 ``LICENCES_REFUSED`` when licences (or its
    gateway) definitely refused the lookup, 503 when it cannot answer."""
    try:
        return await tenant_active_licence(licences, tenant_id)
    except LicencesTenantNotOnboardedError as exc:
        raise tenant_not_onboarded() from exc
    except LicencesRefusedError as exc:
        raise licences_refused(exc.status_code) from exc
    except LicencesUnavailableError as exc:
        raise licences_unavailable(
            "Could not check this tenant id against the licences service; retry."
        ) from exc


@router.put(
    "/{tenant_id}/licence-token",
    response_model=TenantLicenceTokenResult,
    operation_id="tenants.licence_token.set",
)
async def set_tenant_licence_token(
    tenant_id: UUID,
    body: TenantLicenceTokenSet,
    request: Request,
    licences: LicensesClientDep,
    keyring: LicenceKeyringDep,
    audit: AuditRepoDep,
    session: SessionDep,
) -> TenantLicenceTokenResult:
    """Rotate / hand over the tenant's stored licence token (CP
    ``changeTenantLicence`` minus mint/revoke — the partner owns those). Caller:
    the partner bound to the tenant (else 404 / 403 as for its keys), or a
    super_admin. Verify (422 ``LICENCE_TOKEN_MISMATCH``) → lease (409
    ``licence_busy``) → CAS on the old blob (409 ``licence_changed_concurrently``)
    → audit, committed before the response."""
    if SUPER_ADMIN in _roles(request):
        actor_raw, by = getattr(request.state, "subject_id", None), "super_admin"
        tenant = (
            await session.execute(select(Tenant).where(Tenant.id == tenant_id))
        ).scalar_one_or_none()
        if tenant is None or tenant.status == "deleted":
            raise TenancyError(404, TENANT_NOT_FOUND, "No such tenant.")
        actor = _uuid(actor_raw)
    else:
        licence_id, actor = partner_licence(request)
        by = "partner"
        await _partner_bound_tenant(session, tenant_id, licence_id)
    await session.rollback()  # no row lock held across the licences call
    if not keyring.configured:
        raise keyring_unavailable()
    token = body.licence_token.get_secret_value()
    verdict = await verify_licence_token(licences, tenant_id, token)
    installed = await install_token(
        session,
        keyring,
        tenant_id,
        token,
        verdict,
        on_swapped=_audit_token_set(audit, tenant_id, actor, by),
    )
    return TenantLicenceTokenResult(
        tenant_id=tenant_id,
        licence_ref=installed.licence_ref,
        changed=installed.changed,
        licence_token_set_at=installed.set_at,
        licence_token_expires_at=verdict.token_expires_at if installed.changed else None,
    )


# ─── P9-3: partner-minted machine keys ───────────────────────────────────


async def _partner_bound_tenant(session: AsyncSession, tenant_id: UUID, licence_id: UUID) -> Tenant:
    """The tenant, if it is bound to THIS partner licence. 404 when it does not
    exist (or was deleted); 403 when another party manages it (see module
    docstring for why existence is not hidden)."""
    tenant = (
        await session.execute(
            select(Tenant).where(Tenant.id == tenant_id).with_for_update(read=True)
        )
    ).scalar_one_or_none()
    if tenant is None or tenant.status == "deleted":
        raise TenancyError(404, TENANT_NOT_FOUND, "No such tenant.")
    if tenant.partner_licence_id != licence_id:
        raise TenancyError(
            403, TENANT_NOT_PARTNER_BOUND, "This tenant is not managed by this partner."
        )
    return tenant


def _name_taken(row: TenantApiKey, licence_id: UUID) -> TenancyError:
    if row.partner_licence_id != licence_id:
        # Someone else's key (a tenant_admin's): its existence answers the call,
        # its identity is none of the partner's business.
        return TenancyError(
            409, API_KEY_NAME_EXISTS, "A live API key with this name already exists."
        )
    summary = api_key_summary(row)
    return TenancyError(
        409,
        API_KEY_NAME_EXISTS,
        "A live API key with this name already exists for this tenant. Its secret "
        "cannot be shown again: revoke it and mint a new one to rotate.",
        details={
            "key_id": str(summary.id),
            "name": summary.name,
            "prefix": summary.prefix,
            "scopes": summary.scopes,
            "created_at": summary.created_at.isoformat(),
            "created_via": summary.created_via,
        },
    )


@router.post(
    "/{tenant_id}/api-keys",
    response_model=ApiKeyCreated,
    status_code=status.HTTP_201_CREATED,
    operation_id="tenants.partner_api_keys.create",
)
async def partner_create_api_key(
    tenant_id: UUID,
    body: PartnerApiKeyCreate,
    request: Request,
    audit: AuditRepoDep,
    session: SessionDep,
) -> ApiKeyCreated:
    licence_id, licence_subject = partner_licence(request)
    await _partner_bound_tenant(session, tenant_id, licence_id)
    # tenant_api_keys is RLS-scoped (0090): act as the partner-bound TARGET tenant,
    # not the partner's own claim tenant the after_begin fallback would bind.
    await bind_session_tenant(session, tenant_id)
    keys = ApiKeysRepository(session)
    existing = await keys.live_by_name(tenant_id, body.name)
    if existing is not None:
        raise _name_taken(existing, licence_id)
    try:
        row, full_key = await keys.create(
            tenant_id=tenant_id,
            name=body.name,
            created_by=licence_subject,
            scopes=list(body.scopes),
            partner_licence_id=licence_id,
        )
    except IntegrityError as exc:
        # A concurrent mint of the same name won (uq_tenant_api_keys_partner_live_name).
        await session.rollback()
        winner = await keys.live_by_name(tenant_id, body.name)
        if winner is None:  # pragma: no cover - the winner was revoked in between
            raise
        raise _name_taken(winner, licence_id) from exc
    await session.refresh(row)  # server-default created_at
    await audit.record(
        actor_subject_id=licence_subject,
        action="apikey.create",
        target_tenant_id=tenant_id,
        # The public prefix only — never the secret.
        detail={
            "key_id": str(row.id),
            "name": row.name,
            "prefix": row.prefix,
            "scopes": list(row.scopes or []),
            "by": "partner",
            "partner_licence_id": str(licence_id),
        },
    )
    await session.commit()
    return ApiKeyCreated(**api_key_summary(row).model_dump(), key=full_key)


@router.delete(
    "/{tenant_id}/api-keys/{key_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    operation_id="tenants.partner_api_keys.revoke",
)
async def partner_revoke_api_key(
    tenant_id: UUID,
    key_id: UUID,
    request: Request,
    audit: AuditRepoDep,
    session: SessionDep,
) -> Response:
    licence_id, licence_subject = partner_licence(request)
    await _partner_bound_tenant(session, tenant_id, licence_id)
    # tenant_api_keys is RLS-scoped (0090): act as the partner-bound TARGET tenant,
    # not the partner's own claim tenant the after_begin fallback would bind.
    await bind_session_tenant(session, tenant_id)
    if not await ApiKeysRepository(session).revoke_partner_key(key_id, tenant_id, licence_id):
        raise TenancyError(404, API_KEY_NOT_FOUND, "API key not found.")
    await audit.record(
        actor_subject_id=licence_subject,
        action="apikey.revoke",
        target_tenant_id=tenant_id,
        detail={"key_id": str(key_id), "by": "partner", "partner_licence_id": str(licence_id)},
    )
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
