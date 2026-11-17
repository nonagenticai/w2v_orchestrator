"""Early-access signup: approve, reject, the provisioning steps and the retry tick.

Port of the fr / CommonPractice early-access v2 core onto w2v's own onboarding
path (``~/.claude/plans/w2v-early-access.md``). The approve is SHARED by the
Telegram decision, the super_admin approve and the super_admin invite.

THE DECISION. ``approve`` takes the row's APPROVE CLAIM (``approving_until`` /
``approving_claim``: one conditional UPDATE on a pending row with a fresh token —
two approvers, a redelivered tap, a reject, can never race), reserves the
workspace name + slug (partial UNIQUE on the row), mints the ``tenant_id`` ONCE,
and records ``approved`` in the same token-guarded UPDATE. From then on the row
is approved whatever the provisioning does: a failed step is recorded and the
tick retries it — the decision is never re-asked.

THE STEPS, in order, each recorded on the row (``steps``), never inferred:

  identity      w2v MINTS the requester's realm user itself (George 2026-10-20:
                exactly like CommonPractice and fr — ``identity.py``): credential-
                less, unverified, firstName/lastName from the name (fr
                ``splitName``, so Keycloak shows no "Update account information"
                form), ``locale`` / ``signup_app=w2v`` / ``w2v_signup_id=[row id]``,
                re-read to confirm the stamp stuck. An address whose realm user
                does not carry THIS row's id is REFUSED (``identity_exists``),
                never adopted. FIRST, so a refusal lands before any tenant or
                licence is founded for a person this path cannot onboard — and
                so the ``membership`` step knows the person's Keycloak id.
  provision     the SAME shared core ``POST /api/admin/provision`` uses
                (``src/services/tenant_founding.py``): register the tenant in
                licences as w2v's own → the ``tenants`` row → the tenant-level
                ``w2v`` bundle licences + the LLM access posture (the approver's
                choice recorded on the row, ``llm_access``) → a dedicated
                sealed licence token. Then the GATE CHECK: the licence status the
                request gate itself reads (``get_license_status`` for app key
                ``w2v``, tenant-level) must be ENTITLED, or the step fails
                (fr #1616: issue exactly what your own gate checks).
  membership    SEATS the requester at approval, exactly like CommonPractice:
                a ``tenant_admin`` invite into the new tenant is created and
                ACCEPTED at once for the identity step's Keycloak user id (= the
                JWT ``sub`` they will sign in with), through the SAME
                ``accept_invite_row`` an invitee's own accept runs — so the
                membership row, its role and the invite's ``accepted`` record
                are exactly what an accepted invite leaves. Nothing is left to
                accept at first login.
  invite_email  LAST, only once every earlier step is ok: the seat is CHECKED,
                never re-made — a seat a super_admin removed after the membership
                step fails the step ``seat_removed`` and no email goes (also when
                the login was deleted too: no seat is re-made). ``seat_removed``
                is operator-terminal: the tick parks it, only a Retry
                re-evaluates it (only the membership step seats; it re-runs
                when a SEATED login is re-minted, see ``identity_gone``) — then Keycloak's ``execute-actions-email`` sends the
                ONE email to the identity step's user (still stamped with this
                row's id, ``locale`` + ``signup_app`` re-asserted; UPDATE_PASSWORD
                + VERIFY_EMAIL, client ``w2v-frontend``, lifespan 14 days) whose
                redirect is the APP ROOT ``{W2V_FRONTEND_URL}/`` (CP
                ``signup-email.ts``): the person is already seated, so the link
                only has to land them in the app, and a link that carries no
                per-send token is never invalidated by a later send (a Retry
                leaves every earlier email working). A read/write timeout records
                ``unknown`` (the email may have gone out) and is NEVER re-sent
                (Telegram is told ONCE per send attempt: ``operator_notified_at``)
                automatically — only the super_admin Retry does. ``in_flight`` is
                committed BEFORE the call, so a process killed mid-send leaves a
                marker the resume treats like ``unknown``.

Approve, the 5-minute tick and the super_admin Retry all take the row's STEPS
LEASE before running any step, so the email can never be sent twice in parallel.
The approve takes it IN the decision UPDATE (and puts the first tick attempt a
lease-length out), so the steps run as the approver, never as an anonymous tick;
the tick runs as the row's decider (``decider_of``).
The tick backs off 5 min doubling, capped at 6 h, at most 10 attempts.

Reject (pending only) marks the row: approve creates everything, so a pending row
has nothing to undo. A decided row answers ``already_decided``.
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from typing import Literal
from uuid import UUID, uuid7

from sqlalchemy import bindparam, func, or_, select, text, update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.api.routes.tenancy.errors import TenancyError
from src.api.routes.tenancy.invites import accept_invite_row
from src.api.routes.tenancy.partner import LICENCES_TENANT_NAME_TAKEN
from src.clients.keycloak_admin import KeycloakAdminClient, KeycloakAdminError
from src.clients.licenses_client import LicensesClient, LicensesClientError
from src.config import current_signup_settings, get_settings
from src.database.models import SignupApproval, Tenant, TenantInvite
from src.database.repositories.admin_audit_repository import AuditRepository
from src.database.repositories.invites_repository import InvitesRepository, new_invite_token
from src.database.repositories.members_repository import MembersRepository
from src.services.license_provisioning import W2V_LICENCE_APP_KEY, NewTenantLlmAccess
from src.services.license_status import (
    ENTITLED_STATES,
    get_license_status,
    invalidate_license_status,
)
from src.services.providers.crypto import Keyring
from src.services.signup.identity import (
    EMAIL_ACTIONS,
    EMAIL_LIFESPAN_SECONDS,
    IDENTITY_GONE,
    IdentityError,
    ensure_identity,
    prepare_email,
)
from src.services.signup.naming import default_tenant_name, slug_candidates, slugify
from src.services.signup.telegram import (
    STILL_PENDING_HEADLINE,
    DecisionNotice,
    EmailUncertainNotice,
    NewSignupNotice,
    notify_decision,
    notify_email_uncertain,
    notify_new_signup,
)
from src.services.tenant_founding import found_tenant, license_new_tenant

_log = logging.getLogger("w2v_orchestrator.signup")

SessionFactory = async_sessionmaker[AsyncSession]

STEPS: tuple[str, ...] = ("identity", "provision", "membership", "invite_email")
StepStatus = Literal["ok", "skipped", "failed", "waiting", "unknown", "in_flight"]
#: An email send whose outcome is not known: ``unknown`` (the call failed after the
#: request may have been processed) or ``in_flight`` (recorded BEFORE the call and
#: never replaced — the process died mid-send). Neither is ever re-sent
#: automatically; only the super_admin Retry re-sends.
UNCERTAIN_SEND: tuple[str, ...] = ("unknown", "in_flight")
#: invite_email found no active seat for the login it would mail: a super_admin
#: removed it after the membership step. Never re-seated by the email step.
SEAT_REMOVED = "seat_removed"

CLAIM_SECONDS = 5 * 60
STEPS_LEASE_SECONDS = 5 * 60
STEPS_MAX_ATTEMPTS = 10
STEPS_BACKOFF_BASE_SECONDS = 5 * 60
STEPS_BACKOFF_CAP_SECONDS = 6 * 60 * 60
TICK_BATCH = 20
_LEASE = datetime.timedelta(seconds=STEPS_LEASE_SECONDS)

# ─── errors / actors ─────────────────────────────────────────────────────


class SignupError(Exception):
    """A refusal of the signup core: HTTP status + a stable ``code``."""

    def __init__(
        self, status_code: int, code: str, message: str, extra: dict[str, object] | None = None
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.extra = extra or {}


class AlreadyDecidedError(SignupError):
    def __init__(self, row: SignupApproval) -> None:
        super().__init__(400, "approval_not_pending", f"Approval is already {row.status}")
        self.row = row


@dataclass(frozen=True, slots=True)
class SignupActor:
    kind: Literal["super_admin", "telegram", "system"]
    subject_id: UUID | None = None
    telegram_user_id: int | None = None
    azp: str | None = None

    @property
    def audit(self) -> str:
        if self.kind == "telegram":
            return f"telegram:{self.telegram_user_id}"
        if self.kind == "super_admin":
            return str(self.subject_id)
        return "system:signup-tick"

    @property
    def label(self) -> str:
        if self.kind == "telegram":
            return f"Telegram (user {self.telegram_user_id})"
        return "Studio onboarding screen" if self.kind == "super_admin" else "signup tick"

    @property
    def decided_via(self) -> str:
        return "telegram" if self.kind == "telegram" else "onboarding"

    def audit_detail(self) -> dict[str, object]:
        """Who acted, for audit ``detail``: ``actor_subject_id`` is a UUID column
        and a Telegram decider has none, so the actor string always rides here."""
        out: dict[str, object] = {"actor": self.audit}
        if self.azp:
            out["via"] = self.azp
        return out


SYSTEM_ACTOR = SignupActor(kind="system")


def decider_of(row: SignupApproval) -> SignupActor:
    """The actor who DECIDED ``row`` — the tick acts on their decision, so
    whatever it creates (the invite, its audit) is attributed to them, never to
    an anonymous system actor."""
    if row.decided_via == "telegram" and (row.decider_actor or "").startswith("telegram:"):
        raw = (row.decider_actor or "").split(":", 1)[1]
        if raw.isdigit():
            return SignupActor(kind="telegram", telegram_user_id=int(raw))
    if row.decided_by:
        try:
            return SignupActor(kind="super_admin", subject_id=UUID(row.decided_by))
        except ValueError:
            pass
    return SYSTEM_ACTOR


# ─── dependencies (injectable: tests pass fakes) ─────────────────────────


@dataclass(slots=True)
class SignupDeps:
    licenses: LicensesClient
    identity: KeycloakAdminClient
    keyring: Keyring


# ─── steps record ────────────────────────────────────────────────────────


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def steps_of(row: SignupApproval) -> dict[str, dict[str, object]]:
    return dict(row.steps or {})


def _done(rec: dict[str, object] | None) -> bool:
    return bool(rec) and rec.get("status") in ("ok", "skipped")  # type: ignore[union-attr]


def steps_complete(row: SignupApproval) -> bool:
    s = steps_of(row)
    return all(_done(s.get(k)) for k in STEPS)


def steps_not_ok(row: SignupApproval) -> list[dict[str, object]]:
    s = steps_of(row)
    return [
        {
            "step": k,
            "status": (s.get(k) or {}).get("status", "pending"),
            "code": (s.get(k) or {}).get("code"),
        }
        for k in STEPS
        if not _done(s.get(k))
    ]


def awaiting_operator(row: SignupApproval) -> bool:
    """The email step is OPERATOR-TERMINAL: its send is UNCERTAIN (``unknown`` /
    ``in_flight``), or the seat was removed (``seat_removed`` — a super_admin's
    deliberate act). The tick never re-runs it; only the super_admin Retry
    re-evaluates it."""
    rec = steps_of(row).get("invite_email") or {}
    if rec.get("status") in UNCERTAIN_SEND:
        return True
    return rec.get("status") == "failed" and rec.get("code") == SEAT_REMOVED


def email_state(row: SignupApproval) -> Literal["sent", "waiting"]:
    return "sent" if (steps_of(row).get("invite_email") or {}).get("status") == "ok" else "waiting"


def steps_backoff_seconds(attempts: int) -> int:
    exponent = min(max(attempts, 1) - 1, 10)
    return min(STEPS_BACKOFF_CAP_SECONDS, STEPS_BACKOFF_BASE_SECONDS * 2**exponent)


def _step(
    status: StepStatus,
    *,
    code: str | None = None,
    error: str | None = None,
    detail: dict[str, object] | None = None,
) -> dict[str, object]:
    rec: dict[str, object] = {"status": status, "at": _now().isoformat(), "code": code}
    if error:
        rec["error"] = error[:500]
    if detail:
        rec["detail"] = detail
    return rec


def _err_code(exc: Exception, fallback: str) -> str:
    if isinstance(exc, SignupError):
        return exc.code
    if isinstance(exc, TenancyError):
        return exc.code
    if isinstance(exc, IdentityError):
        return exc.code
    if isinstance(exc, KeycloakAdminError):
        return fallback
    if isinstance(exc, LicensesClientError):
        return exc.code or f"{fallback}_{exc.status_code or 'unreachable'}"
    status_code = getattr(exc, "status_code", None)
    return f"{fallback}_{status_code}" if isinstance(status_code, int) else fallback


def _err_text(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:500]


async def _patch_steps(
    s: AsyncSession, signup_id: UUID, patch: dict[str, dict[str, object]]
) -> None:
    """Merge ``patch`` (step → record) into the row's ``steps``. No commit."""
    await s.execute(
        update(SignupApproval)
        .where(SignupApproval.id == signup_id)
        .values(
            steps=SignupApproval.steps.op("||")(bindparam("patch", patch, type_=JSONB)),
            updated_at=func.now(),
        )
    )


async def _record_step(
    sf: SessionFactory, signup_id: UUID, step: str, rec: dict[str, object]
) -> None:
    """Unconditional on purpose: the effect exists whatever the lease state."""
    async with sf() as s:
        await _patch_steps(s, signup_id, {step: rec})
        await s.commit()


# ─── reads ───────────────────────────────────────────────────────────────


async def load(sf: SessionFactory, signup_id: UUID) -> SignupApproval | None:
    async with sf() as s:
        return await s.get(SignupApproval, signup_id, populate_existing=True)


async def _must_load(sf: SessionFactory, signup_id: UUID) -> SignupApproval:
    row = await load(sf, signup_id)
    if row is None:
        raise SignupError(404, "approval_not_found", f"No signup approval {signup_id!s}")
    return row


async def pending_count(sf: SessionFactory) -> int:
    async with sf() as s:
        return int(
            (
                await s.execute(
                    select(func.count())
                    .select_from(SignupApproval)
                    .where(SignupApproval.status == "pending")
                )
            ).scalar_one()
        )


_NOT_DONE = "(coalesce(steps->'{k}'->>'status', 'pending') NOT IN ('ok','skipped'))"
_NEEDS_ATTENTION = (
    "(status = 'approved' AND ("
    + " OR ".join(_NOT_DONE.format(k=k) for k in STEPS)
    + ")) OR (status = 'removed' AND removal->>'state' IN ('pending','blocked','failed'))"
)

LIST_FILTERS = ("pending", "approved", "rejected", "removed", "needs_attention")


async def list_signups(sf: SessionFactory, status: str | None) -> list[SignupApproval]:
    q = select(SignupApproval).order_by(SignupApproval.created_at.desc()).limit(500)
    if status == "needs_attention":
        q = q.where(text(_NEEDS_ATTENTION))
    elif status is not None:
        q = q.where(SignupApproval.status == status)
    async with sf() as s:
        return list((await s.execute(q)).scalars().all())


# ─── intake / invite rows ────────────────────────────────────────────────


async def insert_request(
    sf: SessionFactory,
    *,
    email: str,
    name: str,
    company: str | None,
    message: str | None,
    locale: str | None,
    source: str,
    intake_client: str | None,
) -> UUID | None:
    """Insert a PENDING row; None when the address already has an OPEN one — a
    pending request or an APPROVED workspace (``uq_signup_approvals_open_email``:
    early access is one workspace per person). A rejected row never blocks."""
    async with sf() as s:
        new_id = (
            await s.execute(
                pg_insert(SignupApproval)
                .values(
                    id=uuid7(),
                    email=email,
                    name=name,
                    company=company,
                    message=message,
                    locale=locale,
                    source=source,
                    intake_client=intake_client,
                )
                .on_conflict_do_nothing()
                .returning(SignupApproval.id)
            )
        ).scalar_one_or_none()
        await s.commit()
    return new_id


async def pending_id_for_email(sf: SessionFactory, email: str) -> UUID | None:
    async with sf() as s:
        return (
            await s.execute(
                select(SignupApproval.id).where(
                    SignupApproval.status == "pending",
                    func.lower(SignupApproval.email) == email.strip().lower(),
                )
            )
        ).scalar_one_or_none()


async def approved_for_email(sf: SessionFactory, email: str) -> SignupApproval | None:
    async with sf() as s:
        return (
            await s.execute(
                select(SignupApproval).where(
                    SignupApproval.status == "approved",
                    func.lower(SignupApproval.email) == email.strip().lower(),
                )
            )
        ).scalar_one_or_none()


async def removing_for_email(sf: SessionFactory, email: str) -> SignupApproval | None:
    """A REMOVED row whose cleanup is not finished: it still reserves the address
    (``uq_signup_approvals_open_email``, migration 0100)."""
    async with sf() as s:
        return (
            await s.execute(
                select(SignupApproval).where(
                    SignupApproval.status == "removed",
                    SignupApproval.removal["state"].astext.not_in(("done", "resolved_manually")),
                    func.lower(SignupApproval.email) == email.strip().lower(),
                )
            )
        ).scalar_one_or_none()


async def set_locale(sf: SessionFactory, signup_id: UUID, locale: str) -> None:
    async with sf() as s:
        await s.execute(
            update(SignupApproval)
            .where(SignupApproval.id == signup_id, SignupApproval.status == "pending")
            .values(locale=locale, updated_at=func.now())
        )
        await s.commit()


# ─── the approve claim + the decision ────────────────────────────────────


async def _claim(sf: SessionFactory, signup_id: UUID) -> UUID:
    token = uuid7()
    async with sf() as s:
        got = (
            await s.execute(
                update(SignupApproval)
                .where(
                    SignupApproval.id == signup_id,
                    SignupApproval.status == "pending",
                    or_(
                        SignupApproval.approving_until.is_(None),
                        SignupApproval.approving_until <= func.now(),
                    ),
                )
                .values(
                    approving_until=func.now() + datetime.timedelta(seconds=CLAIM_SECONDS),
                    approving_claim=token,
                )
                .returning(SignupApproval.id)
            )
        ).scalar_one_or_none()
        await s.commit()
    if got is not None:
        return token
    row = await _must_load(sf, signup_id)
    if row.status != "pending":
        raise AlreadyDecidedError(row)
    raise SignupError(
        409, "approval_in_progress", "This signup is being decided right now; reload in a moment."
    )


async def _release_claim(sf: SessionFactory, signup_id: UUID, token: UUID) -> None:
    try:
        async with sf() as s:
            await s.execute(
                update(SignupApproval)
                .where(SignupApproval.id == signup_id, SignupApproval.approving_claim == token)
                .values(approving_until=None, approving_claim=None)
            )
            await s.commit()
    except Exception:  # noqa: BLE001 — the claim expires on its own
        _log.warning("signup id=%s: approve claim release failed", signup_id, exc_info=True)


async def _taken_names(sf: SessionFactory, candidates: list[str]) -> set[str]:
    async with sf() as s:
        names = (
            await s.execute(
                select(func.lower(Tenant.name)).where(func.lower(Tenant.name).in_(candidates))
            )
        ).scalars()
        return {n for n in names if n}


async def _decide_approved(
    sf: SessionFactory,
    row: SignupApproval,
    token: UUID,
    actor: SignupActor,
    tenant_name: str | None,
    lease: UUID,
    llm_access: NewTenantLlmAccess,
) -> SignupApproval:
    """Reserve the workspace slug, mint the tenant id and record ``approved`` —
    one token-guarded UPDATE per slug candidate (the partial UNIQUE on
    ``tenant_slug`` arbitrates between concurrent approvals).

    The same UPDATE hands the approving request the STEPS LEASE (``lease``) and
    puts the first tick attempt one lease-length out: no tick on any replica can
    run the steps before the approver does (they would run as the system actor
    and the approver's reply would say "email follows"); if the approver dies, its
    lease expires and the tick picks the row up then."""
    name = (
        tenant_name
        or row.tenant_name
        or default_tenant_name(email=row.email, name=row.name, company=row.company)
    ).strip()
    candidates = slug_candidates(slugify(name))
    taken = await _taken_names(sf, candidates)
    tenant_id = row.tenant_id or uuid7()
    for slug in candidates:
        if slug in taken:
            continue
        try:
            async with sf() as s:
                updated = (
                    await s.execute(
                        update(SignupApproval)
                        .where(
                            SignupApproval.id == row.id,
                            SignupApproval.status == "pending",
                            SignupApproval.approving_claim == token,
                        )
                        .values(
                            status="approved",
                            decided_at=func.now(),
                            decided_by=str(actor.subject_id) if actor.subject_id else None,
                            decided_via=actor.decided_via,
                            decider_actor=actor.audit,
                            tenant_name=name,
                            tenant_slug=slug,
                            tenant_id=tenant_id,
                            llm_access=llm_access,
                            approving_until=None,
                            approving_claim=None,
                            steps_lease_token=lease,
                            steps_lease_until=func.now() + _LEASE,
                            steps_next_attempt_at=func.now() + _LEASE,
                            steps_attempts=0,
                            updated_at=func.now(),
                        )
                        .returning(SignupApproval)
                    )
                ).scalar_one_or_none()
                await s.commit()
        except IntegrityError:
            continue  # another signup reserved this slug — the next candidate
        if updated is None:
            now = await _must_load(sf, row.id)
            if now.status != "pending":
                raise AlreadyDecidedError(now)
            raise SignupError(409, "approval_claim_lost", "The approve claim was lost; retry.")
        return updated
    raise SignupError(
        409,
        "tenant_name_unavailable",
        f"No free workspace name for {name!r} (tried {candidates[0]} … {candidates[-1]}); "
        "approve on the onboarding screen with another name.",
    )


# ─── the steps lease ─────────────────────────────────────────────────────


async def _take_lease(sf: SessionFactory, signup_id: UUID) -> UUID | None:
    token = uuid7()
    async with sf() as s:
        got = (
            await s.execute(
                update(SignupApproval)
                .where(
                    SignupApproval.id == signup_id,
                    SignupApproval.status == "approved",
                    or_(
                        SignupApproval.steps_lease_until.is_(None),
                        SignupApproval.steps_lease_until <= func.now(),
                    ),
                )
                .values(
                    steps_lease_token=token,
                    steps_lease_until=func.now() + datetime.timedelta(seconds=STEPS_LEASE_SECONDS),
                )
                .returning(SignupApproval.id)
            )
        ).scalar_one_or_none()
        await s.commit()
    return token if got is not None else None


async def _renew_lease(sf: SessionFactory, signup_id: UUID, token: UUID) -> bool:
    async with sf() as s:
        got = (
            await s.execute(
                update(SignupApproval)
                .where(SignupApproval.id == signup_id, SignupApproval.steps_lease_token == token)
                .values(
                    steps_lease_until=func.now() + datetime.timedelta(seconds=STEPS_LEASE_SECONDS)
                )
                .returning(SignupApproval.id)
            )
        ).scalar_one_or_none()
        await s.commit()
    return got is not None


async def _hold_lease(sf: SessionFactory, signup_id: UUID, held: UUID | None) -> UUID | None:
    """The lease to run under: ``held`` (taken with the decision) if it is still
    ours, else a fresh one; None when another run holds it."""
    if held is None:
        return await _take_lease(sf, signup_id)
    return held if await _renew_lease(sf, signup_id, held) else None


async def _release_lease(sf: SessionFactory, signup_id: UUID, token: UUID) -> None:
    try:
        async with sf() as s:
            await s.execute(
                update(SignupApproval)
                .where(SignupApproval.id == signup_id, SignupApproval.steps_lease_token == token)
                .values(steps_lease_token=None, steps_lease_until=None)
            )
            await s.commit()
    except Exception:  # noqa: BLE001 — the lease expires on its own
        _log.warning("signup id=%s: steps lease release failed", signup_id, exc_info=True)


# ─── the steps ───────────────────────────────────────────────────────────


def app_root_url() -> str:
    """Where the welcome email returns the person: the APP ROOT (CP
    ``signup-email.ts``). They were seated at approval, so there is nothing to
    accept — and a link with no per-send token is never killed by a later send."""
    return get_settings().provisioning.w2v_frontend_url.rstrip("/") + "/"


def llm_access_of(row: SignupApproval) -> NewTenantLlmAccess:
    """The LLM access the approver chose (recorded with the decision); the
    deployment default (``W2V_SIGNUP_LLM_ACCESS``) for a row decided before the
    choice was recorded."""
    if row.llm_access == "platform_only":
        return "platform_only"
    if row.llm_access == "byo_required":
        return "byo_required"
    if row.llm_access == "local_allowed":
        return "local_allowed"
    return current_signup_settings().llm_access


def _app_keys() -> list[str]:
    """The apps whose tenant-level licences the step requires — always including
    the GATE key ``w2v`` (``_common.W2V_APP_KEY``)."""
    return list(dict.fromkeys([W2V_LICENCE_APP_KEY, *get_settings().provisioning.license_app_keys]))


async def _reserve_next_slug(
    sf: SessionFactory, row: SignupApproval, lease: UUID, exclude: set[str]
) -> SignupApproval:
    """After licences refused the reserved name (another w2v tenant has it): move
    this approved row to the next free candidate (nothing was created yet)."""
    base = slugify(row.tenant_name or row.tenant_slug or "workspace")
    candidates = slug_candidates(base)
    taken = await _taken_names(sf, candidates) | exclude
    for slug in candidates:
        if slug in taken:
            continue
        try:
            async with sf() as s:
                updated = (
                    await s.execute(
                        update(SignupApproval)
                        .where(
                            SignupApproval.id == row.id,
                            SignupApproval.steps_lease_token == lease,
                        )
                        .values(tenant_slug=slug, updated_at=func.now())
                        .returning(SignupApproval)
                    )
                ).scalar_one_or_none()
                await s.commit()
        except IntegrityError:
            continue
        if updated is None:
            raise SignupError(409, "steps_lease_lost", "The steps lease was lost.")
        return updated
    raise SignupError(409, "tenant_name_unavailable", f"No free workspace name for {base!r}.")


async def _provision(
    sf: SessionFactory, deps: SignupDeps, row: SignupApproval, lease: UUID, actor: SignupActor
) -> dict[str, object]:
    if not deps.keyring.configured:
        return _step(
            "failed",
            code="keyring_unavailable",
            error="No licence keyring is configured, so the tenant's licence token cannot be stored.",
        )
    assert row.tenant_id is not None and row.tenant_slug is not None  # noqa: S101 — CHECK
    tenant_id = row.tenant_id
    tried: set[str] = set()
    while True:
        try:
            async with sf() as s:
                await found_tenant(
                    s,
                    deps.licenses,
                    tenant_id=tenant_id,
                    name=str(row.tenant_slug),
                    actor_subject_id=actor.subject_id,
                )
                licensed = await license_new_tenant(
                    s,
                    deps.licenses,
                    deps.keyring,
                    tenant_id=tenant_id,
                    app_keys=_app_keys(),
                    llm_access=llm_access_of(row),
                    mint_tokens=False,
                )
                await s.commit()
            break
        except TenancyError as exc:
            if exc.code != LICENCES_TENANT_NAME_TAKEN or len(tried) >= 9:
                raise
            tried.add(str(row.tenant_slug))
            row = await _reserve_next_slug(sf, row, lease, tried)
    detail: dict[str, object] = {
        "tenant_id": str(tenant_id),
        "tenant_slug": row.tenant_slug,
        "licence_ref": str(licensed.licence_ref) if licensed.licence_ref else None,
        "license_ids": licensed.license_ids,
        "llm_access": licensed.llm_access.as_record(),
    }
    if licensed.llm_access.status != "ok":
        return _step(
            "failed",
            code="llm_access_failed",
            error=licensed.llm_access.detail or "the LLM access posture was not set",
            detail=detail,
        )
    # The GATE CHECK: the same resolution the request gate (require_license) runs,
    # tenant-level (no subject), for app key ``w2v``.
    invalidate_license_status(
        tenant_id=str(tenant_id), app_key=W2V_LICENCE_APP_KEY, subject_id=None
    )
    verdict = await get_license_status(
        client=deps.licenses,
        app_key=W2V_LICENCE_APP_KEY,
        tenant_id=str(tenant_id),
        subject_id=None,
        gate_enabled=True,
    )
    detail["licence_state"] = verdict.state
    if verdict.state not in ENTITLED_STATES:
        return _step(
            "failed",
            code="licence_not_entitled",
            error=f"the tenant's {W2V_LICENCE_APP_KEY!r} licence reads {verdict.state!r} at the gate",
            detail=detail,
        )
    return _step("ok", detail=detail)


async def _seat(
    s: AsyncSession, row: SignupApproval, user_id: str, actor: SignupActor
) -> tuple[UUID | None, bool]:
    """``(invite_id, already_member)`` — SEAT the requester (Keycloak user
    ``user_id``) as ``tenant_admin`` of the new tenant, the way CommonPractice
    does at approval: the ``tenant_admin`` invite is created (when absent) and
    ACCEPTED for ``user_id`` through ``accept_invite_row`` — the one rule an
    invitee's own accept runs (the pending→accepted flip gates the membership
    write; an existing active membership keeps its role). Idempotent: an active
    membership for ``user_id`` answers ``already_member``. No commit.

    Expiry: ``accept_invite_row`` binds only an UNEXPIRED invite and leaves that
    check to its caller. This is that check, made explicit: the seat is the
    approver's decision being carried out NOW, not a token the requester holds,
    so an already-pending invite for the address whose ``expires_at`` has passed
    is RE-ARMED (``expires_at`` = now + ``invite_ttl_days``, audited as
    ``member.invite`` with ``rearmed``) under the
    row lock before the accept — never accepted while expired."""
    assert row.tenant_id is not None  # noqa: S101 — approved rows carry it (CHECK)
    subject_id = UUID(user_id)
    if await MembersRepository(s).get_active_membership(subject_id, row.tenant_id):
        return None, True
    invites = InvitesRepository(s)
    pending = await invites.get_pending_for_tenant_email(row.tenant_id, row.email)
    if pending is None:
        _token, token_hash = new_invite_token()  # never sent: accepted right here
        ttl = datetime.timedelta(days=current_signup_settings().invite_ttl_days)
        pending = await invites.create(
            tenant_id=row.tenant_id,
            email=row.email,
            role="tenant_admin",
            token_hash=token_hash,
            expires_at=_now() + ttl,
            invited_by=actor.subject_id,
        )
        await AuditRepository(s).record(
            actor_subject_id=actor.subject_id,
            action="member.invite",
            target_tenant_id=row.tenant_id,
            detail={
                "invite_id": str(pending.id),
                "email": row.email,
                "role": "tenant_admin",
                "signup_id": str(row.id),
                "seated_subject_id": user_id,
                **actor.audit_detail(),
            },
        )
    invite = (
        await s.execute(select(TenantInvite).where(TenantInvite.id == pending.id).with_for_update())
    ).scalar_one()
    now = _now()
    if invite.expires_at <= now:
        expired_at = invite.expires_at
        invite.expires_at = now + datetime.timedelta(days=current_signup_settings().invite_ttl_days)
        await s.flush()
        await AuditRepository(s).record(
            actor_subject_id=actor.subject_id,
            action="member.invite",
            target_tenant_id=row.tenant_id,
            detail={
                "rearmed": True,
                "invite_id": str(invite.id),
                "email": row.email,
                "signup_id": str(row.id),
                "expired_at": expired_at.isoformat(),
                "expires_at": invite.expires_at.isoformat(),
                "seated_subject_id": user_id,
                **actor.audit_detail(),
            },
        )
    accepted = await accept_invite_row(s, invite, subject_id, display_name=row.name)
    if accepted is None:
        raise SignupError(409, "invite_not_pending", "the seat's invite was settled meanwhile")
    _role, already = accepted
    return invite.id, already


async def _membership(
    sf: SessionFactory, row: SignupApproval, actor: SignupActor
) -> dict[str, object]:
    user_id = _identity_user_id(row)
    if user_id is None:  # the gate runs this only after identity is ok
        return _step(
            "failed", code="identity_not_owned", error="no realm user recorded for this signup"
        )
    async with sf() as s:
        invite_id, already = await _seat(s, row, user_id, actor)
        await s.commit()
    if already:
        return _step(
            "ok", code="already_member", detail={"already_member": True, "subject_id": user_id}
        )
    return _step(
        "ok",
        detail={"invite_id": str(invite_id), "role": "tenant_admin", "subject_id": user_id},
    )


def _identity_user_id(row: SignupApproval) -> str | None:
    """The realm user the ``identity`` step minted (or adopted) for this row."""
    detail = (steps_of(row).get("identity") or {}).get("detail")
    user_id = detail.get("user_id") if isinstance(detail, dict) else None
    return user_id if isinstance(user_id, str) and user_id else None


async def _identity(deps: SignupDeps, row: SignupApproval) -> dict[str, object]:
    try:
        got = await ensure_identity(
            deps.identity,
            signup_id=str(row.id),
            email=row.email,
            name=row.name,
            locale=row.locale,
        )
    except IdentityError as exc:
        return _step("failed", code=exc.code, error=str(exc), detail=exc.detail)
    return _step("ok", detail={"user_id": got.user_id, "minted": got.minted})


async def _invite_email(
    sf: SessionFactory, deps: SignupDeps, row: SignupApproval, actor: SignupActor
) -> dict[str, object]:
    assert row.tenant_id is not None  # noqa: S101 — approved rows carry it (CHECK)
    if not get_settings().provisioning.w2v_frontend_url.strip():
        return _step(
            "failed",
            code="frontend_url_unset",
            error="W2V_FRONTEND_URL is not set: the email's landing link would be relative.",
        )
    user_id = _identity_user_id(row)
    if user_id is None:  # the gate runs this only after identity is ok
        return _step(
            "failed", code="identity_not_owned", error="no realm user recorded for this signup"
        )
    # Never write to, or mail, a login this signup did not mint; locale +
    # signup_app select the w2v-branded template in the person's language.
    try:
        attributes = await prepare_email(
            deps.identity, user_id=user_id, signup_id=str(row.id), locale=row.locale
        )
    except IdentityError as exc:
        if exc.code != IDENTITY_GONE:
            raise  # identity_not_owned: someone else's login — never touched, keeps refusing
        # Deleted out of band after the identity step (fr): forget the recorded
        # user so the identity step mints (or adopts) one again on the next
        # attempt / Retry. The seat bound to the deleted login can never sign in
        # again: end it, and reopen the membership step in the SAME transaction,
        # so the next run seats the new login through that step (its record then
        # names the new user) — invite_email itself never seats.
        # If there was NO active seat to end, a super_admin already removed it:
        # re-seating a new login would undo that decision. Record
        # ``seat_removed`` and leave identity + membership as they are — the
        # operator decides (Retry once they re-seat, or offboard).
        async with sf() as s:
            if not await MembersRepository(s).remove_member(UUID(user_id), row.tenant_id):
                await s.rollback()
                return _step(
                    "failed",
                    code=SEAT_REMOVED,
                    error="the person's seat in the workspace was removed and their login "
                    "was deleted; no welcome email was sent and the seat was not re-made",
                    detail={"subject_id": user_id, "identity_gone": True},
                )
            await _patch_steps(
                s,
                row.id,
                {
                    "identity": _step(
                        "failed",
                        code=IDENTITY_GONE,
                        error=str(exc),
                        detail={"gone_user_id": user_id},
                    ),
                    "membership": _step(
                        "waiting", code=IDENTITY_GONE, detail={"unseated_subject_id": user_id}
                    ),
                },
            )
            await s.commit()
        return _step("failed", code=IDENTITY_GONE, error=str(exc), detail={"user_id": user_id})
    client_id = get_settings().provisioning.identity_invite_client_id
    redirect_uri = app_root_url()
    async with sf() as s:
        # CHECK the seat, never re-make it: the gate ran this only after the
        # membership step seated ``user_id``, so a missing active membership now
        # means it was removed since (a super_admin's deliberate act) — mailing a
        # sign-in link into a workspace they were taken out of would be wrong, and
        # re-seating them would undo that decision.
        if await MembersRepository(s).get_active_membership(UUID(user_id), row.tenant_id) is None:
            return _step(
                "failed",
                code=SEAT_REMOVED,
                error="the person's seat in the workspace was removed after approval; "
                "no welcome email was sent",
                detail={"subject_id": user_id},
            )
        # ``in_flight`` is committed BEFORE the call: a process killed after
        # Keycloak sent but before ``ok`` lands leaves this marker, and a resumed
        # run treats it like ``unknown`` (never re-sent automatically). Only the
        # super_admin Retry re-sends — harmless now: the link carries no token,
        # so every earlier email keeps working.
        attempt = {
            "attempt_id": str(uuid7()),
            "actor": actor.audit,
            "user_id": user_id,
            "redirect_uri": redirect_uri,
        }
        await s.execute(
            update(SignupApproval)
            .where(SignupApproval.id == row.id)
            .values(
                steps=SignupApproval.steps.op("||")(
                    bindparam(
                        "patch",
                        {"invite_email": _step("in_flight", code="email_sending", detail=attempt)},
                        type_=JSONB,
                    )
                ),
                updated_at=func.now(),
            )
        )
        await s.commit()
    try:
        await deps.identity.execute_actions_email(
            user_id,
            list(EMAIL_ACTIONS),
            client_id=client_id,
            redirect_uri=redirect_uri,
            lifespan_seconds=EMAIL_LIFESPAN_SECONDS,
        )
    except KeycloakAdminError as exc:
        if not exc.uncertain:
            raise  # refused before anything was sent: failed, the tick retries it
        return _step("unknown", code="email_send_uncertain", error=_err_text(exc), detail=attempt)
    except Exception as exc:  # noqa: BLE001 — the request may have gone out
        return _step("unknown", code="email_send_uncertain", error=_err_text(exc), detail=attempt)
    return _step(
        "ok",
        detail={
            **attempt,
            "locale": row.locale or "en",
            "client_id": client_id,
            "attributes": attributes,
        },
    )


@dataclass(slots=True)
class StepsRun:
    ran: bool
    row: SignupApproval
    reason: str | None = None
    order: list[str] = field(default_factory=list)


def _gate(row: SignupApproval, step: str, *, manual: bool) -> Literal["run", "skip", "wait"]:
    """Whether ``step`` runs now: never once it landed; the email never while
    an earlier step is not ok, and an ``unknown`` send only on a manual Retry."""
    s = steps_of(row)
    if _done(s.get(step)):
        return "skip"
    if step == "invite_email" and awaiting_operator(row) and not manual:
        return "skip"
    if any(not _done(s.get(k)) for k in STEPS[: STEPS.index(step)]):
        return "wait"
    return "run"


async def _run_step(
    sf: SessionFactory,
    deps: SignupDeps,
    row: SignupApproval,
    step: str,
    lease: UUID,
    actor: SignupActor,
) -> dict[str, object]:
    try:
        if step == "identity":
            return await _identity(deps, row)
        if step == "provision":
            return await _provision(sf, deps, row, lease, actor)
        if step == "membership":
            return await _membership(sf, row, actor)
        return await _invite_email(sf, deps, row, actor)
    except Exception as exc:  # noqa: BLE001 — recorded on the row, retried by the tick
        _log.warning("signup id=%s step=%s failed: %s", row.id, step, _err_text(exc))
        return _step("failed", code=_err_code(exc, f"{step}_failed"), error=_err_text(exc))


async def run_steps(
    sf: SessionFactory,
    deps: SignupDeps,
    signup_id: UUID,
    *,
    actor: SignupActor | None = None,
    manual: bool = False,
    lease: UUID | None = None,
) -> StepsRun:
    """Run every step that has not landed, in order, under the steps lease. A
    failure is recorded and the run moves on (the email then waits). The email
    runs only when every earlier step is ok, never once it is ok, and an
    ``unknown`` / ``in_flight`` send only on a ``manual`` (super_admin) Retry.

    ``lease``: the steps lease the approve already holds (taken with the
    decision); otherwise one is taken here. ``actor`` None = the row's decider
    (the tick acts on their decision)."""
    first = await _must_load(sf, signup_id)
    if first.status != "approved":
        return StepsRun(ran=False, row=first, reason="not_approved")
    lease = await _hold_lease(sf, signup_id, lease)
    if lease is None:
        return StepsRun(ran=False, row=first, reason="lease_busy")
    if actor is None:
        actor = decider_of(first)
    order: list[str] = []
    try:
        for step in STEPS:
            row = await _must_load(sf, signup_id)
            if row.status != "approved":
                break
            gate = _gate(row, step, manual=manual)
            if gate == "skip":
                continue
            if gate == "wait":
                blocking = [
                    k for k in STEPS[: STEPS.index(step)] if not _done(steps_of(row).get(k))
                ]
                waiting = _step("waiting", code="waiting_for_steps", detail={"blocking": blocking})
                await _record_step(sf, signup_id, step, waiting)
                continue
            if not await _renew_lease(sf, signup_id, lease):
                break
            order.append(step)
            await _record_step(
                sf, signup_id, step, await _run_step(sf, deps, row, step, lease, actor)
            )
        await _notify_uncertain_email(sf, signup_id)
    finally:
        await _release_lease(sf, signup_id, lease)
    return StepsRun(ran=True, row=await _must_load(sf, signup_id), order=order)


async def _notify_uncertain_email(sf: SessionFactory, signup_id: UUID) -> None:
    """Tell Telegram ONCE per send attempt that the welcome email's send is
    uncertain (``unknown``, or an ``in_flight`` left by a killed process): the row
    parks for the super_admin Retry, so nobody would otherwise know. The mark
    (``operator_notified_at`` on the step) is taken with a conditional UPDATE on
    the same attempt, so later ticks and other replicas never repeat it; a Retry
    writes a fresh step record, and a new uncertain outcome is told again. Never
    raises: a notice failure must not fail the step."""
    try:
        row = await _must_load(sf, signup_id)
        rec = steps_of(row).get("invite_email") or {}
        status = rec.get("status")
        if status not in UNCERTAIN_SEND or rec.get("operator_notified_at"):
            return
        if not current_signup_settings().telegram_enabled:
            return
        email_rec = SignupApproval.steps["invite_email"]
        async with sf() as s:
            claimed = (
                await s.execute(
                    update(SignupApproval)
                    .where(
                        SignupApproval.id == signup_id,
                        email_rec["status"].astext == str(status),
                        email_rec["at"].astext == str(rec.get("at")),
                        ~email_rec.has_key("operator_notified_at"),
                    )
                    .values(
                        steps=func.jsonb_set(
                            SignupApproval.steps,
                            text("'{invite_email,operator_notified_at}'::text[]"),
                            func.to_jsonb(func.now()),
                        ),
                        updated_at=func.now(),
                    )
                    .returning(SignupApproval.id)
                )
            ).scalar_one_or_none()
            await s.commit()
        if claimed is None:
            return
        notify_email_uncertain(
            EmailUncertainNotice(id=str(row.id), status=str(status), email=row.email, name=row.name)
        )
        _log.info("signup id=%s invite_email %s: operator notified", signup_id, status)
    except Exception:  # noqa: BLE001 — a notice never fails the step or the tick
        _log.warning("signup id=%s: uncertain-email notice failed", signup_id, exc_info=True)


async def _finish_steps_run(
    sf: SessionFactory, row: SignupApproval, *, reset: bool
) -> SignupApproval:
    """Schedule the next tick attempt (or none): complete / awaiting the operator
    → none; else 5 min doubling capped at 6 h, at most 10 attempts."""
    complete = steps_complete(row) or awaiting_operator(row)
    attempts = 0 if reset else row.steps_attempts + 1
    if complete:
        values: dict[str, object] = {"steps_next_attempt_at": None}
    elif attempts >= STEPS_MAX_ATTEMPTS:
        values = {"steps_attempts": attempts, "steps_next_attempt_at": None}
    else:
        delay = steps_backoff_seconds(attempts + 1 if reset else attempts)
        values = {
            "steps_attempts": attempts,
            "steps_next_attempt_at": func.now() + datetime.timedelta(seconds=delay),
        }
    async with sf() as s:
        updated = (
            await s.execute(
                update(SignupApproval)
                .where(SignupApproval.id == row.id)
                .values(**values, updated_at=func.now())
                .returning(SignupApproval)
            )
        ).scalar_one_or_none()
        await s.commit()
    return updated or row


# ─── approve / reject / retry / tick ─────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ApproveResult:
    row: SignupApproval
    order: list[str]

    @property
    def email_state(self) -> str:
        return email_state(self.row)

    @property
    def not_ok(self) -> list[dict[str, object]]:
        return steps_not_ok(self.row)


async def approve(
    sf: SessionFactory,
    deps: SignupDeps,
    signup_id: UUID,
    actor: SignupActor,
    *,
    tenant_name: str | None = None,
    llm_access: NewTenantLlmAccess | None = None,
    notify: bool = False,
) -> ApproveResult:
    """Decide ``approved`` (under the approve claim), then run the steps.

    ``llm_access`` — the approver's per-person choice for the new tenant
    (CommonPractice's plan choice on approve / invite); None = the deployment
    default (the Telegram approve, like CP's Telegram path takes the default
    plan). Recorded on the row WITH the decision, so every retry provisions it.

    Raises :class:`AlreadyDecidedError` for a decided row, ``SignupError`` 404 /
    409 (``approval_in_progress``, ``tenant_name_unavailable``). A step failure
    does NOT raise: the row stays approved with that step recorded, and the tick
    retries it."""
    token = await _claim(sf, signup_id)
    lease = uuid7()
    try:
        row = await _must_load(sf, signup_id)
        decided = await _decide_approved(
            sf,
            row,
            token,
            actor,
            tenant_name,
            lease,
            llm_access or current_signup_settings().llm_access,
        )
    except AlreadyDecidedError:
        await _release_claim(sf, signup_id, token)
        raise
    except Exception as exc:
        await _release_claim(sf, signup_id, token)
        await _repost_if_still_pending(sf, signup_id, exc)
        raise
    await _release_claim(sf, signup_id, token)
    _log.info(
        "signup approved id=%s tenant=%s slug=%s by=%s",
        signup_id,
        decided.tenant_id,
        decided.tenant_slug,
        actor.audit,
    )
    run = await run_steps(sf, deps, signup_id, actor=actor, lease=lease)
    final = await _finish_steps_run(sf, run.row, reset=True) if run.ran else run.row
    if notify:
        notify_decision(
            DecisionNotice(
                id=str(final.id),
                outcome="approved",
                email=final.email,
                name=final.name,
                tenant_id=str(final.tenant_id) if final.tenant_id else None,
                tenant_name=final.tenant_name,
                tenant_slug=final.tenant_slug,
                decided_via=actor.label,
                email_state=email_state(final),
                not_ok=steps_not_ok(final),
            )
        )
    return ApproveResult(row=final, order=run.order)


async def _repost_if_still_pending(sf: SessionFactory, signup_id: UUID, exc: Exception) -> None:
    """An approve that failed BEFORE the decision leaves the row pending: post a
    fresh Approve / Reject message, so a pending row always has Telegram buttons
    (an earlier message's buttons may be gone). Never raises."""
    try:
        row = await load(sf, signup_id)
        if row is None or row.status != "pending":
            return
        code = _err_code(exc, "approve_failed")
        notify_new_signup(
            NewSignupNotice(
                id=str(row.id),
                source=row.source,
                pending_total=await pending_count(sf),
                email=row.email,
                name=row.name,
                company=row.company,
                message=row.message,
                locale=row.locale,
                headline=STILL_PENDING_HEADLINE,
                note=f"An approve failed ({code}); the request is still pending.",
            )
        )
    except Exception:  # noqa: BLE001 — a notice never masks the approve's own error
        _log.warning("signup id=%s: still-pending notice failed", signup_id, exc_info=True)


async def reject(
    sf: SessionFactory, signup_id: UUID, actor: SignupActor, *, notify: bool = False
) -> SignupApproval:
    """Pending → rejected. No side effects to undo: approve creates everything.
    Raises :class:`AlreadyDecidedError` for a decided row, 409 while an approve
    holds the claim, 404 for an unknown id."""
    async with sf() as s:
        updated = (
            await s.execute(
                update(SignupApproval)
                .where(
                    SignupApproval.id == signup_id,
                    SignupApproval.status == "pending",
                    or_(
                        SignupApproval.approving_until.is_(None),
                        SignupApproval.approving_until <= func.now(),
                    ),
                )
                .values(
                    status="rejected",
                    decided_at=func.now(),
                    decided_by=str(actor.subject_id) if actor.subject_id else None,
                    decided_via=actor.decided_via,
                    decider_actor=actor.audit,
                    updated_at=func.now(),
                )
                .returning(SignupApproval)
            )
        ).scalar_one_or_none()
        await s.commit()
    if updated is None:
        row = await _must_load(sf, signup_id)
        if row.status != "pending":
            raise AlreadyDecidedError(row)
        raise SignupError(
            409, "approval_in_progress", "This signup is being approved right now; reload."
        )
    _log.info("signup rejected id=%s by=%s", signup_id, actor.audit)
    if notify:
        notify_decision(
            DecisionNotice(
                id=str(updated.id),
                outcome="rejected",
                email=updated.email,
                name=updated.name,
                decided_via=actor.label,
            )
        )
    return updated


async def retry(
    sf: SessionFactory, deps: SignupDeps, signup_id: UUID, actor: SignupActor
) -> SignupApproval:
    """super_admin Retry of an approved row's unfinished steps (resets the
    backoff; also re-sends an ``unknown`` email)."""
    row = await _must_load(sf, signup_id)
    if row.status != "approved":
        raise SignupError(
            400,
            "approval_not_approved",
            f"Only an approved signup's steps can be retried (it is {row.status}).",
        )
    run = await run_steps(sf, deps, signup_id, actor=actor, manual=True)
    if not run.ran:
        raise SignupError(409, "steps_in_progress", "Its steps are running right now; reload.")
    return await _finish_steps_run(sf, run.row, reset=True)


async def run_tick(sf: SessionFactory, deps: SignupDeps, *, limit: int = TICK_BATCH) -> int:
    """One pass over DUE approved rows (``steps_next_attempt_at <= now()``). Safe
    with several replicas: each row is taken under its steps lease. Never raises
    per row. Returns how many rows ran."""
    async with sf() as s:
        due = list(
            (
                await s.execute(
                    select(SignupApproval.id)
                    .where(
                        SignupApproval.status == "approved",
                        SignupApproval.steps_next_attempt_at <= func.now(),
                    )
                    .order_by(SignupApproval.steps_next_attempt_at)
                    .limit(limit)
                )
            ).scalars()
        )
    ran = 0
    for signup_id in due:
        try:
            run = await run_steps(sf, deps, signup_id)
            if not run.ran:
                continue
            ran += 1
            await _finish_steps_run(sf, run.row, reset=False)
        except Exception:  # noqa: BLE001 — one row never stops the tick
            _log.warning("signup tick: row %s failed", signup_id, exc_info=True)
    return ran
