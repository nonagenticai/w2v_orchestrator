"""Early-access signup (v2) — w2v's half of the CommonPractice / fr flow.

Service doors (``door_router``, prefix ``/api/signup-approvals``):

- POST /api/signup-approvals/intake                 early-access-intake's SA → a pending row
- POST /api/signup-approvals/{id}/telegram-decision chat-bridge's SA relaying George's tap (C3)

super_admin (``admin_router``, prefix ``/api/admin/signup-approvals``):

- GET  /api/admin/signup-approvals                  the queue (?status=pending|approved|rejected|needs_attention)
                                                     + ``defaultLlmAccess`` (W2V_SIGNUP_LLM_ACCESS)
- GET  /api/admin/signup-approvals/{id}
- POST /api/admin/signup-approvals/invite           invite straight into early access (approves)
- POST /api/admin/signup-approvals/{id}/approve     (optional ``tenantName``, ``llmAccess``)
- POST /api/admin/signup-approvals/{id}/reject
- POST /api/admin/signup-approvals/{id}/retry       re-run an approved row's unfinished steps
- POST /api/admin/signup-approvals/{id}/remove      REMOVE an approved early-access workspace
                                                     (``services/signup/removal.py``; confirm = slug)
- POST /api/admin/signup-approvals/{id}/removal/retry    re-run a pending / blocked / failed removal
- POST /api/admin/signup-approvals/{id}/removal/resolve  blocked / failed → resolved_manually

⛔ Every route answers 503 ``{"error": "signup_not_configured"}`` unless
``W2V_SIGNUP_ENABLED=true`` (read per call). None of them is behind the tenant
gate: the service doors carry no tenant (they verify their own token —
``src/services/signup/service_token.py``; ``AuthMiddleware`` is bypassed for
exactly those two paths, ``src/auth/signup_doors.py``), and the super_admin
routes are how a tenant comes to exist (the same circularity as
``POST /api/admin/provision``).

Errors are flat ``{"error": <code>, "detail": <text>}`` (the contract the intake
handler and chat-bridge were written against, fr/CP shape), not the tenancy
envelope.
"""

from __future__ import annotations

import datetime
import logging
import re
from typing import Annotated, Final, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, FastAPI, Header, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.api.routes._common import (
    LicenceKeyringDep,
    LicensesClientDep,
    SignupKeycloakAdminDep,
    SubjectIdDep,
    SuperAdminDep,
)
from src.config import SIGNUP_LOCALES, current_signup_settings
from src.database.models import SignupApproval
from src.services.signup import core, removal
from src.services.signup.core import (
    AlreadyDecidedError,
    SignupActor,
    SignupDeps,
    SignupError,
)
from src.services.signup.naming import default_tenant_name
from src.services.signup.service_token import (
    SignupAuthError,
    SignupServiceCaller,
    require_decider_caller,
    require_intake_caller,
)
from src.services.signup.telegram import NewSignupNotice, notify_new_signup
from src.services.signup.text import multi_line, normalise_locale, single_line

_log = logging.getLogger("w2v_orchestrator.signup.routes")

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class SignupNotConfiguredError(Exception):
    pass


def require_signup_enabled() -> None:
    if not current_signup_settings().enabled:
        raise SignupNotConfiguredError


def _flat(status_code: int, error: str, detail: str, **extra: object) -> JSONResponse:
    return JSONResponse({"error": error, "detail": detail, **extra}, status_code=status_code)


async def _not_configured(_request: Request, _exc: Exception) -> JSONResponse:
    return _flat(
        503,
        "signup_not_configured",
        "Early-access signup is not enabled on this deployment (W2V_SIGNUP_ENABLED).",
    )


async def _auth_error(_request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, SignupAuthError)  # noqa: S101 — registered for it only
    return _flat(exc.status_code, exc.code, exc.message)


async def _signup_error(_request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, SignupError)  # noqa: S101 — registered for it only
    return _flat(exc.status_code, exc.code, exc.message, **exc.extra)


def register_signup_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(SignupNotConfiguredError, _not_configured)
    app.add_exception_handler(SignupAuthError, _auth_error)
    app.add_exception_handler(SignupError, _signup_error)


_ENABLED = [Depends(require_signup_enabled)]
door_router = APIRouter(prefix="/api/signup-approvals", tags=["signup"], dependencies=_ENABLED)
admin_router = APIRouter(
    prefix="/api/admin/signup-approvals", tags=["signup"], dependencies=_ENABLED
)


def _sf(request: Request) -> core.SessionFactory:
    return request.app.state.session_factory


def get_signup_deps(
    licenses: LicensesClientDep, identity: SignupKeycloakAdminDep, keyring: LicenceKeyringDep
) -> SignupDeps:
    return SignupDeps(licenses=licenses, identity=identity, keyring=keyring)


SignupDepsDep = Annotated[SignupDeps, Depends(get_signup_deps)]


async def intake_caller(
    authorization: Annotated[str | None, Header()] = None,
) -> SignupServiceCaller:
    return await require_intake_caller(authorization, current_signup_settings())


async def decider_caller(
    authorization: Annotated[str | None, Header()] = None,
) -> SignupServiceCaller:
    return await require_decider_caller(authorization, current_signup_settings())


# ─── request bodies ──────────────────────────────────────────────────────


def _single(value: str, limit: int, *, required: bool) -> str | None:
    if len(value) > limit * 4:
        raise ValueError(f"at most {limit} characters")
    v = single_line(value)
    if not v:
        if required:
            raise ValueError("must not be empty")
        return None
    if len(v) > limit:
        raise ValueError(f"at most {limit} characters")
    return v


def _email(value: str) -> str:
    v = value.strip()
    if len(v) > 320 or not _EMAIL.match(v):
        raise ValueError("not an email address")
    return v


class IntakeBody(BaseModel):
    """What early-access-intake forwards (``forwarded_body``): requester-typed
    text is normalised to one printable line (``message`` keeps newlines)."""

    model_config = ConfigDict(extra="forbid")
    email: str
    name: str
    company: str | None = None
    message: str | None = None
    source: Literal["web", "chat"]
    #: C6: lenient — an unknown / malformed tag is "not told" (→ en), never a 422.
    locale: str | None = Field(default=None, max_length=35)
    #: Accepted for contract compatibility (the intake verified it already).
    turnstileToken: str | None = Field(default=None, max_length=4096)  # noqa: N815

    _check_email = field_validator("email")(_email)

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        out = _single(v, 160, required=True)
        assert out is not None  # noqa: S101 — required
        return out

    @field_validator("company")
    @classmethod
    def _company(cls, v: str | None) -> str | None:
        return None if v is None else _single(v, 160, required=False)

    @field_validator("message")
    @classmethod
    def _message(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if len(v) > 8000:
            raise ValueError("at most 2000 characters")
        out = multi_line(v)
        if len(out) > 2000:
            raise ValueError("at most 2000 characters")
        return out or None


class DecisionBody(BaseModel):
    """Contract C3 — strict."""

    model_config = ConfigDict(extra="forbid", strict=True)
    action: Literal["approve", "reject"]
    telegramUserId: int = Field(gt=0, le=2**53 - 1)  # noqa: N815


#: The approver's per-person LLM access choice (CommonPractice's plan choice on
#: approve / invite): any of the three postures, like ``POST /api/admin/provision``
#: (owner decision 2026-10-22). Anything else is a 422.
#: Omitted = the deployment default (``W2V_SIGNUP_LLM_ACCESS``, ``local_allowed``).
#: The Telegram one-tap approve always passes ``local_allowed`` ("Platform models
#: allowed", owner choice) — never the deployment default.
LlmAccessChoice = Literal["local_allowed", "byo_required", "platform_only"]
#: What the Telegram one-tap approve gives a new tenant.
TELEGRAM_APPROVE_LLM_ACCESS: Final[LlmAccessChoice] = "local_allowed"


class ApproveBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tenantName: str | None = Field(default=None, min_length=2, max_length=120)  # noqa: N815
    llmAccess: LlmAccessChoice | None = None  # noqa: N815


class InviteBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: str
    name: str
    company: str | None = None
    tenantName: str | None = Field(default=None, min_length=2, max_length=120)  # noqa: N815
    #: The email's language (the form's picker). Strict: anything else is a 422.
    locale: Literal["ar", "de", "en", "es", "fr", "he", "nl", "pl"] | None = None
    llmAccess: LlmAccessChoice | None = None  # noqa: N815

    _check_email = field_validator("email")(_email)

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        out = _single(v, 160, required=True)
        assert out is not None  # noqa: S101 — required
        return out

    @field_validator("company")
    @classmethod
    def _company(cls, v: str | None) -> str | None:
        return None if v is None else _single(v, 160, required=False)


class LegacyIdentityBody(BaseModel):
    """The login a LEGACY row (approved before logins were stamped) created:
    its Keycloak user id and the operator's evidence (kept as a fingerprint only)."""

    model_config = ConfigDict(extra="forbid")
    user_id: UUID
    #: Free text — NEVER stored: only its sha256 + length + when
    #: (``removal.evidence_fingerprint``) lands on the row and in the audit.
    evidence: str = Field(min_length=3, max_length=2000)

    @field_validator("evidence")
    @classmethod
    def _evidence(cls, v: str) -> str:
        clean = v.strip()
        if len(clean) < 3:
            raise ValueError("evidence must not be blank")
        return clean


class RemoveBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    #: The workspace slug, typed back.
    confirm: str = Field(min_length=1, max_length=64)
    legacy_identity: LegacyIdentityBody | None = None
    #: Legacy only: delete the login even though it holds a credential.
    allow_credentialed_delete: bool = False


class RemovalRetryBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    legacy_identity: LegacyIdentityBody | None = None
    allow_credentialed_delete: bool | None = None


class RemovalResolveBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    note: str = Field(min_length=3, max_length=2000)
    acknowledge_live_access: bool = False


def _require_removal_worker() -> None:
    """A removal only ever runs on the signup tick: with the tick disabled
    (``W2V_SIGNUP_TICK_INTERVAL_SECONDS <= 0``) a start / retry would queue
    silently forever, so it is refused instead."""
    if current_signup_settings().tick_interval_seconds <= 0:
        raise SignupError(
            503,
            "removal_worker_disabled",
            "The signup tick that runs removals is disabled "
            "(W2V_SIGNUP_TICK_INTERVAL_SECONDS <= 0); nothing would run this removal.",
        )


def _legacy(body: LegacyIdentityBody | None) -> removal.LegacyIdentity | None:
    if body is None:
        return None
    return removal.LegacyIdentity(user_id=str(body.user_id), evidence=body.evidence.strip())


assert set(SIGNUP_LOCALES) == {"ar", "de", "en", "es", "fr", "he", "nl", "pl"}  # noqa: S101


# ─── views ───────────────────────────────────────────────────────────────


def _iso(value: datetime.datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def signup_view(row: SignupApproval) -> dict[str, object]:
    """The row as the onboarding screen shows it (no claim / lease internals)."""
    now = datetime.datetime.now(datetime.UTC)
    return {
        "id": str(row.id),
        "email": row.email,
        "name": row.name,
        "company": row.company,
        "message": row.message,
        "locale": row.locale,
        "source": row.source,
        "status": row.status,
        "created_at": _iso(row.created_at),
        "decided_at": _iso(row.decided_at),
        "decided_via": row.decided_via,
        "decider_actor": row.decider_actor,
        "approving": row.approving_until is not None and row.approving_until > now,
        "tenant_name": row.tenant_name,
        "tenant_slug": row.tenant_slug,
        "tenant_id": str(row.tenant_id) if row.tenant_id else None,
        "llm_access": row.llm_access,
        "suggested_tenant_name": row.tenant_name
        or default_tenant_name(email=row.email, name=row.name, company=row.company),
        "steps": row.steps or {},
        "steps_not_ok": core.steps_not_ok(row) if row.status == "approved" else [],
        "email_state": core.email_state(row),
        "steps_attempts": row.steps_attempts,
        "steps_next_attempt_at": _iso(row.steps_next_attempt_at),
        # The early-access removal (0100): None until a super_admin removes it.
        "removal": removal.removal_view(row),
        "has_identity_step": removal.has_identity_step(row),
        "needs_legacy_identity": row.status == "approved" and removal.needs_legacy_identity(row),
    }


def _approve_body(result: core.ApproveResult) -> dict[str, object]:
    row = result.row
    return {
        "approval": signup_view(row),
        "tenantId": str(row.tenant_id) if row.tenant_id else None,
        "tenant": {
            "id": str(row.tenant_id) if row.tenant_id else None,
            "name": row.tenant_name,
            "slug": row.tenant_slug,
        },
        "emailState": result.email_state,
        "stepsNotOk": result.not_ok,
    }


def _not_found(signup_id: object) -> SignupError:
    return SignupError(404, "approval_not_found", f"No signup approval {signup_id!s}")


def _uuid(raw: str) -> UUID:
    try:
        return UUID(raw)
    except ValueError:
        raise _not_found(raw) from None


# ─── intake (service account) ────────────────────────────────────────────


@door_router.post("/intake", status_code=202, operation_id="signup.intake")
async def signup_intake(
    request: Request,
    caller: Annotated[SignupServiceCaller, Depends(intake_caller)],
    body: IntakeBody,
) -> JSONResponse:
    """Queue a request (``202 {"status": "queued"}``); an address that already has
    a pending request OR an approved workspace (one per person) answers ``202
    {"status": "already_pending"}`` and sends nothing — the intake's contract, and
    it tells the public form nothing about the address.
    New rows post the Telegram notice (fire-and-forget)."""
    cfg = current_signup_settings()
    sf = _sf(request)
    pending = await core.pending_count(sf)
    if pending >= cfg.queue_max_pending:
        _log.warning("signup queue full pending=%s intake_client=%s", pending, caller.azp)
        return _flat(503, "signup_queue_full", "Sign-up requests are temporarily closed.")
    locale = normalise_locale(body.locale)
    new_id = await core.insert_request(
        sf,
        email=body.email,
        name=body.name,
        company=body.company,
        message=body.message,
        locale=locale,
        source=body.source,
        intake_client=caller.azp,
    )
    if new_id is None:
        return JSONResponse({"status": "already_pending"}, status_code=202)
    notify_new_signup(
        NewSignupNotice(
            id=str(new_id),
            source=body.source,
            pending_total=pending + 1,
            email=body.email,
            name=body.name,
            company=body.company,
            message=body.message,
            locale=locale,
        )
    )
    return JSONResponse({"status": "queued"}, status_code=202)


# ─── Telegram decision (chat-bridge service account, contract C3) ────────


def summary(text: str) -> str:
    """Plain text, ≤ 200 characters — shown in Telegram by chat-bridge."""
    flat = " ".join(text.split())
    return f"{flat[:199]}…" if len(flat) > 200 else flat


IN_PROGRESS_SUMMARY = "Another approval is in progress - try again in a minute."


def _already_decided(row: SignupApproval) -> JSONResponse:
    workspace = (
        f" (workspace {row.tenant_slug or row.tenant_id})"
        if row.status == "approved" and row.tenant_id
        else ""
    )
    body: dict[str, object] = {
        "status": "already_decided",
        "outcome": row.status,
        "summary": summary(f"Already {row.status}: {row.email}{workspace}."),
    }
    if row.status == "approved" and row.tenant_id:
        body["tenantId"] = str(row.tenant_id)
    return JSONResponse(body)


def _approved_summary(result: core.ApproveResult) -> str:
    row = result.row
    failed = [s for s in result.not_ok if s["status"] in ("failed", *core.UNCERTAIN_SEND)]
    if result.email_state == "sent":
        tail = "set-password email sent"
    elif failed:
        first = failed[0]
        tail = f"{first['step']} {first['status']} ({first['code']}), retried automatically"
    else:
        tail = "email follows once every step is ok"
    return summary(
        f'Approved: {row.email} -> workspace "{row.tenant_name}" ({row.tenant_slug}); {tail}.'
    )


@door_router.post("/{signup_id}/telegram-decision", operation_id="signup.telegram_decision")
async def signup_telegram_decision(
    signup_id: str,
    request: Request,
    caller: Annotated[SignupServiceCaller, Depends(decider_caller)],
    body: DecisionBody,
    deps: SignupDepsDep,
) -> JSONResponse:
    """``200 {status: approved|rejected|already_decided, tenantId?, summary}``,
    404 unknown id, 401/403 auth, ``502 {status: failed, summary}``, ``409
    {status: in_progress, summary}`` while another approve holds the row (a
    RETRYABLE answer: chat-bridge keeps the buttons). Idempotent: a decided row
    answers ``already_decided`` with its outcome."""
    sf = _sf(request)
    try:
        sid = UUID(signup_id)
    except ValueError:
        sid = None
    row = await core.load(sf, sid) if sid is not None else None
    if row is None:
        return _flat(404, "approval_not_found", f"No signup approval {signup_id}")
    if row.status != "pending":
        return _already_decided(row)
    actor = SignupActor(kind="telegram", telegram_user_id=body.telegramUserId, azp=caller.azp)
    try:
        if body.action == "reject":
            await core.reject(sf, row.id, actor)
            return JSONResponse(
                {"status": "rejected", "summary": summary(f"Rejected: {row.email}.")}
            )
        result = await core.approve(sf, deps, row.id, actor, llm_access=TELEGRAM_APPROVE_LLM_ACCESS)
        return JSONResponse(
            {
                "status": "approved",
                "tenantId": str(result.row.tenant_id),
                "summary": _approved_summary(result),
            }
        )
    except AlreadyDecidedError as exc:
        return _already_decided(exc.row)
    except SignupError as exc:
        if exc.code == "approval_in_progress":
            # NOT final: the concurrent approve may still fail and leave the row
            # pending. chat-bridge keeps the Approve / Reject buttons on any
            # non-200 other than 404 (chat-bridge #37), so this is a 409.
            return JSONResponse(
                {
                    "status": "in_progress",
                    "code": exc.code,
                    "summary": IN_PROGRESS_SUMMARY,
                },
                status_code=409,
            )
        code, message = exc.code, exc.message
    except Exception as exc:  # noqa: BLE001 — answered as 502, logged
        _log.exception("signup telegram decision failed id=%s", row.id)
        code, message = "internal_error", type(exc).__name__
    _log.warning(
        "signup telegram decision failed id=%s action=%s code=%s", row.id, body.action, code
    )
    return JSONResponse(
        {
            "status": "failed",
            "code": code,
            "summary": summary(f"Could not {body.action}: {code} - {message}"),
        },
        status_code=502,
    )


# ─── super_admin ─────────────────────────────────────────────────────────


def _admin_actor(subject: UUID) -> SignupActor:
    return SignupActor(kind="super_admin", subject_id=subject)


@admin_router.get("", operation_id="admin.signup_approvals.list")
async def list_signup_approvals(
    request: Request,
    _: SuperAdminDep,
    status: str | None = Query(None),
) -> JSONResponse:
    if status is not None and status not in core.LIST_FILTERS:
        return _flat(
            422,
            "invalid_status_filter",
            f"status must be one of {', '.join(core.LIST_FILTERS)}",
        )
    rows = await core.list_signups(_sf(request), status)
    # ``defaultLlmAccess``: what an approve / invite WITHOUT ``llmAccess`` gets
    # (``W2V_SIGNUP_LLM_ACCESS``), so the page pre-selects it and sends nothing
    # unless the operator changes it — one default for every door.
    return JSONResponse(
        {
            "items": [signup_view(r) for r in rows],
            "defaultLlmAccess": current_signup_settings().llm_access,
        }
    )


@admin_router.get("/{signup_id}", operation_id="admin.signup_approvals.get")
async def get_signup_approval(signup_id: str, request: Request, _: SuperAdminDep) -> JSONResponse:
    row = await core.load(_sf(request), _uuid(signup_id))
    if row is None:
        raise _not_found(signup_id)
    return JSONResponse({"data": signup_view(row)})


@admin_router.post("/invite", operation_id="admin.signup_approvals.invite")
async def invite_signup(
    request: Request,
    body: InviteBody,
    actor: SubjectIdDep,
    _: SuperAdminDep,
    deps: SignupDepsDep,
) -> JSONResponse:
    """Invite a person straight into early access: records a ``source='invite'``
    row — or reuses the address's PENDING row (its language overwritten by the
    picker) — and runs the SAME approve. No new-signup notice; the decision
    notice still goes. An address with an APPROVED row answers 409
    ``early_access_workspace_exists`` (one workspace per person)."""
    sf = _sf(request)
    new_id = await core.insert_request(
        sf,
        email=body.email,
        name=body.name,
        company=body.company,
        message=None,
        locale=body.locale,
        source="invite",
        intake_client=None,
    )
    signup_id = new_id
    if signup_id is None:
        signup_id = await core.pending_id_for_email(sf, body.email)
        if signup_id is None and (existing := await core.approved_for_email(sf, body.email)):
            raise SignupError(
                409,
                "early_access_workspace_exists",
                f"{existing.email} already has an early-access workspace "
                f'("{existing.tenant_name}", {existing.tenant_slug}); early access is one '
                "workspace per person. Invite them into that workspace instead.",
                {"signupId": str(existing.id), "tenantId": str(existing.tenant_id)},
            )
        if signup_id is None and (removing := await core.removing_for_email(sf, body.email)):
            raise SignupError(
                409,
                "early_access_removal_in_progress",
                f"{removing.email}'s early-access workspace is being removed "
                f"({removal.removal_state(removing)}); the address is free once that is done.",
                {"signupId": str(removing.id)},
            )
        if signup_id is None:
            raise SignupError(
                400,
                "approval_not_pending",
                "The pending signup for this address was just decided; retry.",
            )
        if body.locale:
            await core.set_locale(sf, signup_id, body.locale)
    _log.info("signup invite id=%s reused_pending=%s by=%s", signup_id, new_id is None, actor)
    result = await core.approve(
        sf,
        deps,
        signup_id,
        _admin_actor(actor),
        tenant_name=body.tenantName,
        llm_access=body.llmAccess,
        notify=True,
    )
    return JSONResponse({**_approve_body(result), "reusedPending": new_id is None})


@admin_router.post("/{signup_id}/approve", operation_id="admin.signup_approvals.approve")
async def approve_signup(
    signup_id: str,
    request: Request,
    actor: SubjectIdDep,
    _: SuperAdminDep,
    deps: SignupDepsDep,
    body: ApproveBody | None = None,
) -> JSONResponse:
    sid = _uuid(signup_id)
    result = await core.approve(
        _sf(request),
        deps,
        sid,
        _admin_actor(actor),
        tenant_name=body.tenantName if body else None,
        llm_access=body.llmAccess if body else None,
        notify=True,
    )
    return JSONResponse(_approve_body(result))


@admin_router.post("/{signup_id}/reject", operation_id="admin.signup_approvals.reject")
async def reject_signup(
    signup_id: str, request: Request, actor: SubjectIdDep, _: SuperAdminDep
) -> JSONResponse:
    row = await core.reject(_sf(request), _uuid(signup_id), _admin_actor(actor), notify=True)
    return JSONResponse({"approval": signup_view(row)})


@admin_router.post("/{signup_id}/retry", operation_id="admin.signup_approvals.retry")
async def retry_signup(
    signup_id: str,
    request: Request,
    actor: SubjectIdDep,
    _: SuperAdminDep,
    deps: SignupDepsDep,
) -> JSONResponse:
    row = await core.retry(_sf(request), deps, _uuid(signup_id), _admin_actor(actor))
    return JSONResponse({"approval": signup_view(row)})


@admin_router.post(
    "/{signup_id}/remove", operation_id="admin.signup_approvals.remove", status_code=202
)
async def remove_signup(
    signup_id: str,
    request: Request,
    body: RemoveBody,
    actor: SubjectIdDep,
    _: SuperAdminDep,
) -> JSONResponse:
    """Remove an APPROVED early-access workspace end to end. ``confirm`` must be
    the workspace slug. 202: the removal is ENQUEUED (due at once); the signup
    tick runs it — poll the row's ``removal`` for progress. 503
    ``removal_worker_disabled`` when that tick is off."""
    _require_removal_worker()
    row = await removal.start_removal(
        _sf(request),
        _uuid(signup_id),
        _admin_actor(actor),
        confirm=body.confirm,
        legacy_identity=_legacy(body.legacy_identity),
        allow_credentialed_delete=body.allow_credentialed_delete,
    )
    return JSONResponse({"approval": signup_view(row)}, status_code=202)


@admin_router.post(
    "/{signup_id}/removal/retry",
    operation_id="admin.signup_approvals.removal_retry",
    status_code=202,
)
async def retry_removal(
    signup_id: str,
    request: Request,
    actor: SubjectIdDep,
    _: SuperAdminDep,
    body: RemovalRetryBody | None = None,
) -> JSONResponse:
    """Re-run a pending / blocked / failed removal: 202, ENQUEUED like the remove
    (503 ``removal_worker_disabled`` when the tick is off)."""
    _require_removal_worker()
    row = await removal.retry_removal(
        _sf(request),
        _uuid(signup_id),
        _admin_actor(actor),
        legacy_identity=_legacy(body.legacy_identity) if body else None,
        allow_credentialed_delete=body.allow_credentialed_delete if body else None,
    )
    return JSONResponse({"approval": signup_view(row)}, status_code=202)


@admin_router.post(
    "/{signup_id}/removal/resolve", operation_id="admin.signup_approvals.removal_resolve"
)
async def resolve_removal(
    signup_id: str,
    request: Request,
    body: RemovalResolveBody,
    actor: SubjectIdDep,
    _: SuperAdminDep,
) -> JSONResponse:
    row = await removal.resolve_removal(
        _sf(request),
        _uuid(signup_id),
        _admin_actor(actor),
        note=body.note.strip(),
        acknowledge_live_access=body.acknowledge_live_access,
    )
    return JSONResponse({"approval": signup_view(row)})


SIGNUP_ROUTERS = (door_router, admin_router)
__all__ = [
    "SIGNUP_ROUTERS",
    "admin_router",
    "door_router",
    "register_signup_exception_handlers",
]
