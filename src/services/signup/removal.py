"""Early-access REMOVAL: undo an APPROVED signup end to end (George 2026-10-20:
"Build proper offboarding first"; spec ``~/.claude/plans/w2v-early-access-offboarding-spec.md``).

CommonPractice and fr only clean up a pending row whose approve aborted; this
removes a FULLY approved early-access workspace. It reuses their machinery
(CommonPractice #477): one lease per row, every write guarded by the lease
token, each step recorded, ``ok`` steps skipped on a re-run, a bounded retry
tick, ``needs_attention``, a super_admin Retry / Resolve and a Telegram notice.

THE ACTION (:func:`start_removal`, ``POST /api/admin/signup-approvals/{id}/remove``,
super_admin): only an ``approved`` row whose steps lease is free, with the
workspace slug typed back as confirmation. ONE guarded UPDATE moves it to
``status='removed'`` with ``removal = {state: 'pending', attempts: 0, ...}``
(audited ``signup.remove``) and makes it due NOW: the answer is 202 and the
signup tick runs it (never inside the HTTP request — a run can outlast any proxy
timeout). A running removal renews its lease every minute. A removed row keeps the address and the slug RESERVED until the cleanup is
finished (``uq_signup_approvals_open_email``, migration 0100).

THE STEPS, in order (:data:`REMOVAL_STEPS`), each recorded in ``removal.steps``:

  blockers         re-checked EVERY run; any → ``blocked``, nothing is touched:
                   an active member other than the requester, a pending invite
                   for someone else, rows in any content table
                   (``row_counts`` minus :data:`ENGINE_TABLES`), an active run,
                   or ``tenant_id`` not equal to ``steps.provision.detail.tenant_id``.
                   An operator-named LEGACY login is proven read-only here
                   (:func:`_check_legacy`) before it counts as the requester;
                   a login that fails the proof blocks — and so does a named
                   login that does not exist while ANY login carries the row's
                   address. The operator's evidence is kept as a FINGERPRINT
                   only (sha256 + length + when; :func:`evidence_fingerprint`).
  close_access     suspend the tenant (audited like ``PATCH /api/admin/tenants``)
                   and revoke its pending invites — the emailed link dies at once.
  licences         list → revoke every non-revoked licence that is not a shared
                   DEPENDENCY licence (w2v's own SA, the tenant's owning app:
                   bundle-minted licences of other apps included) → re-list and
                   CONFIRM no own licence is live and every dependency licence
                   was released with them (licences releases orphaned
                   dependencies in the revoke's transaction). A dependency still
                   active gets ONE explicit owner revoke: 409
                   ``dependency_still_required`` by a licence on THIS tenant that
                   is not ours → ``blocked`` (operator); needed only by another
                   tenant's licence → kept (the archive retires it here); 403 →
                   ``pending`` ``dependency_release_unsupported``. Then the status
                   cache is invalidated and a FRESH gate read must say
                   ``revoked`` / ``none`` / ``expired``. Anything else (a 403, a
                   partial revoke, another verdict) stays ``pending``
                   (``licence_failed``).
  identity         retire the realm login this signup created (see
                   :func:`_identity`): delete it when it never got a credential,
                   else disable + mark it — only a login PROVEN to be this
                   row's (stamped ``w2v_signup_id``, or a legacy login passing
                   the evidence rule), and never one another w2v tenant uses.
                   A row with neither an identity step nor a named login looks
                   the address up first (:func:`_identity_unnamed`).
  export           skipped when a completed export exists, else one is run.
  tenant_delete    the SAME core ``DELETE /api/admin/tenants/{t}`` runs
                   (``tenant_offboarding.start_tenant_delete``), never ``force``:
                   the licences are revoked, the export exists. A later run
                   polls the job: done → ok, failed → started again (it resumes
                   from its progress), running → ``in_progress``. An open job
                   an admin FORCED is never resumed (``forced_job_exists``,
                   blocked). The delete no
                   longer erases this row (``TENANT_TABLES`` skips ``removed``).
  licences_archive ``PATCH /tenants/{id}/archive`` in licences.
  finalize         FIRST the tenant's export archive(s) (``exports/tenants/{t}/``,
                   which hold the signup's name / address / company / message and
                   the invite + audit addresses) are deleted and the deleted keys
                   recorded; storage unavailable or any key left → ``pending``
                   ``export_purge_failed`` (retried, never ``done``). Then
                   ``tenant_slug`` / ``tenant_name`` cleared (the slug in the
                   removal + the provision step → ``removed-<id prefix>``), the personal data
                   erased (``name=''``, ``company`` / ``message`` NULL, ``email =
                   erased+<id>@invalid``), ``state='done'``, Telegram, audit.

A ``resolved_manually`` removal erases the row's personal data like ``finalize``
(it does NOT delete an export archive: the operator closed it by hand). Its
note is kept as a FINGERPRINT only (sha256 + length), like the evidence.
Telegram notices never carry the name, the address or the real slug.

THE RETRY: the 5-minute signup tick runs DUE ``pending`` removals
(:func:`run_due_removals`). The claim counts the attempt; the backoff is 5 min
doubling capped at 6 h; after 10 attempts → ``failed``. ``blocked`` and
``failed`` wait for the super_admin: Retry (re-runs pending / blocked / failed)
or Resolve (``resolved_manually``, with a note and an explicit acknowledgement
that access may still be live — audited). Telegram is told on ``done``,
``blocked`` and ``failed``.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Literal, cast
from uuid import UUID, uuid7

from sqlalchemy import ColumnElement, func, or_, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.clients.keycloak_admin import KeycloakAdminClient, KeycloakAdminError
from src.clients.licenses_client import LicensesClient, LicensesClientError
from src.clients.schemas.keycloak import KeycloakUser
from src.clients.schemas.licenses import LicenseView
from src.database.models import (
    SignupApproval,
    Tenant,
    TenantInvite,
    TenantMember,
    TenantOffboarding,
)
from src.database.repositories.admin_audit_repository import AuditRepository
from src.services import tenant_offboarding as offboarding
from src.services.license_provisioning import W2V_LICENCE_APP_KEY
from src.services.license_status import get_license_status, invalidate_license_status
from src.services.signup.core import (
    STEPS_LEASE_SECONDS,
    STEPS_MAX_ATTEMPTS,
    SYSTEM_ACTOR,
    TICK_BATCH,
    SessionFactory,
    SignupActor,
    SignupError,
    steps_backoff_seconds,
    steps_of,
)
from src.services.signup.identity import SIGNUP_APP_ATTRIBUTE, SIGNUP_ID_ATTRIBUTE, belongs_to
from src.services.signup.telegram import SIGNUP_APP_KEY, RemovalNotice, notify_removal
from src.services.tenant_sound_library import tenant_has_active_runs

_log = logging.getLogger("w2v_orchestrator.signup.removal")

REMOVAL_STEPS: tuple[str, ...] = (
    "blockers",
    "close_access",
    "licences",
    "identity",
    "export",
    "tenant_delete",
    "licences_archive",
    "finalize",
)
RemovalState = Literal["pending", "blocked", "failed", "done", "resolved_manually"]
StepStatus = Literal["ok", "skipped", "pending", "in_progress", "blocked"]
#: States the tick runs; the super_admin Retry also runs blocked + failed.
TICK_STATES: tuple[str, ...] = ("pending",)
RETRY_STATES: tuple[str, ...] = ("pending", "blocked", "failed")
RESOLVE_STATES: tuple[str, ...] = ("blocked", "failed")
#: The address is free again once the removal reached one of these (0100 index).
FINISHED_STATES: tuple[str, ...] = ("done", "resolved_manually")

#: The tables approve / the platform itself writes for a fresh workspace: rows
#: here are NOT the person's work, so they never block a removal (the export
#: keeps them anyway). Every OTHER ``TENANT_TABLES`` table with a row blocks.
ENGINE_TABLES: frozenset[str] = frozenset(
    {
        "tenants",
        "tenant_members",  # checked on their own: another active member blocks
        "tenant_invites",  # checked on their own: a foreign pending invite blocks
        "provisioning_audit",
        "signup_approvals",
        "request_audit",
        "admin_audit_log",
        "tenant_offboarding",
        "tenant_defaults",
    }
)

#: The marker a retired-but-kept login carries (CommonPractice ``cp_rejected:<id>``).
REMOVED_MARKER_PREFIX = "w2v_removed:"
#: Legacy evidence: the login must be created no later than this after the
#: ``invite_email`` step recorded its send.
LEGACY_WINDOW_SLACK = datetime.timedelta(minutes=5)
#: Attributes a legacy login must NOT carry (it would belong to another flow).
LEGACY_FOREIGN_ATTRIBUTES: tuple[str, ...] = (
    SIGNUP_ID_ATTRIBUTE,
    "fr_signup_id",
    "tenant_id",
    # Set by other apps' membership / booking flows: a login carrying either is
    # (also) someone else's, never a plain early-access login of this row.
    "member_tenant_ids",
    "signup_book_url",
)
#: Gate verdicts that prove no w2v licence of the tenant authorizes any more.
DEAD_LICENCE_STATES: frozenset[str] = frozenset({"revoked", "none", "expired"})

# ─── codes ───────────────────────────────────────────────────────────────
# Blockers (removal.blockers[].code)
OTHER_MEMBER = "other_member"
FOREIGN_INVITE = "foreign_invite"
TENANT_CONTENT = "tenant_content"
ACTIVE_RUNS = "active_runs"
TENANT_MISMATCH = "tenant_mismatch"
TENANT_NOT_PROVISIONED = "tenant_not_provisioned"
# Step codes (removal.steps.<step>.code)
LICENCE_FAILED = "licence_failed"
#: A dependency licence of the tenant is still required by a licence on THIS
#: tenant that the removal does not own (licences 409): blocked, operator decides.
DEPENDENCY_STILL_REQUIRED = "dependency_still_required"
#: licences refused to release a dependency licence (403 — not upgraded yet):
#: pending, retried.
DEPENDENCY_RELEASE_UNSUPPORTED = "dependency_release_unsupported"
LICENCE_ARCHIVE_FAILED = "licence_archive_failed"
LEGACY_IDENTITY_MISMATCH = "legacy_identity_mismatch"
LEGACY_IDENTITY_WINDOW = "legacy_identity_window"
LEGACY_IDENTITY_FOREIGN = "legacy_identity_foreign"
IDENTITY_KEPT_IN_USE = "identity_kept_in_use"
IDENTITY_NOT_OWNED = "identity_not_owned"
EXPORT_FAILED = "export_failed"
#: finalize could not delete the tenant's export archive(s) (retried; never ``done``).
EXPORT_PURGE_FAILED = "export_purge_failed"
TENANT_DELETE_FAILED = "tenant_delete_failed"
TENANT_DELETE_BLOCKED = "tenant_delete_blocked"
STORAGE_UNAVAILABLE = "storage_unavailable"
IN_PROGRESS = "in_progress"
#: A delete job of the tenant was FORCED by an admin earlier: the removal never
#: resumes it (that would make the removal forced) — the operator decides.
FORCED_JOB_EXISTS = "forced_job_exists"

#: How many hex characters of the evidence fingerprint the onboarding screen shows.
EVIDENCE_PREFIX_CHARS = 12


def evidence_fingerprint(evidence: str, *, at: str | None = None) -> dict[str, object]:
    """What is kept of the operator's legacy evidence — on the row AND in the
    audit: its sha256, its length and when it was recorded. NEVER the text,
    not even a "redacted" copy: free text cannot be sanitised completely
    (URL-encoded ``token%3D…``, ``invite_token=``, JSON ``"token": "…"``, path
    tokens …), so none of it is stored."""
    return {
        "evidence_sha256": hashlib.sha256(evidence.encode()).hexdigest(),
        "evidence_length": len(evidence),
        "evidence_created_at": at or _iso(),
    }


def note_fingerprint(note: str) -> dict[str, object]:
    """What is kept of a Resolve note — on the row AND in the audit: its sha256
    and length, never the text (free text can carry the person's data)."""
    return {"note_sha256": hashlib.sha256(note.encode()).hexdigest(), "note_length": len(note)}


_STEP_FALLBACK_CODE: dict[str, str] = {
    "blockers": "blockers_failed",
    "close_access": "close_access_failed",
    "licences": LICENCE_FAILED,
    "identity": "identity_failed",
    "export": EXPORT_FAILED,
    "tenant_delete": TENANT_DELETE_FAILED,
    "licences_archive": LICENCE_ARCHIVE_FAILED,
    "finalize": "finalize_failed",
}

_LEASE = datetime.timedelta(seconds=STEPS_LEASE_SECONDS)


@dataclass(slots=True)
class RemovalDeps:
    licenses: LicensesClient
    identity: KeycloakAdminClient
    #: The own bucket (export + delete); None → those steps wait (``storage_unavailable``).
    storage: offboarding.OffboardingStorage | None


@dataclass(frozen=True, slots=True)
class LegacyIdentity:
    user_id: str
    evidence: str


class _LeaseLostError(RuntimeError):
    pass


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def _iso() -> str:
    return _now().isoformat()


def _rec(
    status: StepStatus,
    *,
    code: str | None = None,
    error: str | None = None,
    detail: dict[str, object] | None = None,
) -> dict[str, object]:
    rec: dict[str, object] = {"status": status, "at": _iso(), "code": code}
    if error:
        rec["error"] = error[:500]
    if detail:
        rec["detail"] = detail
    return rec


def _done(rec: object) -> bool:
    return isinstance(rec, dict) and rec.get("status") in ("ok", "skipped")


def removal_of(row: SignupApproval) -> dict[str, object]:
    return dict(row.removal or {})


def removal_state(row: SignupApproval) -> str | None:
    state = removal_of(row).get("state")
    return state if isinstance(state, str) else None


def _rsteps(removal: dict[str, object]) -> dict[str, dict[str, object]]:
    steps = removal.get("steps")
    return dict(cast(dict[str, dict[str, object]], steps)) if isinstance(steps, dict) else {}


def has_identity_step(row: SignupApproval) -> bool:
    """The row was approved by the identity-minting approve (#1132): its login is
    stamped ``w2v_signup_id``. A row approved before that is LEGACY."""
    return "identity" in steps_of(row)


def legacy_login_was_sent(row: SignupApproval) -> bool:
    """A legacy row's login exists only once its ``invite_email`` step (the
    provisioning identity invite) landed."""
    return (steps_of(row).get("invite_email") or {}).get("status") == "ok"


def needs_legacy_identity(row: SignupApproval) -> bool:
    return not has_identity_step(row) and legacy_login_was_sent(row)


def _err_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:500]


def _uuid_or_none(raw: object) -> UUID | None:
    try:
        return UUID(str(raw)) if raw else None
    except ValueError:
        return None


def removal_actor(row: SignupApproval) -> SignupActor:
    """Who asked for the removal — the tick acts on their decision."""
    subject = _uuid_or_none(removal_of(row).get("requested_by"))
    return SignupActor(kind="super_admin", subject_id=subject) if subject else SYSTEM_ACTOR


# ─── the action ──────────────────────────────────────────────────────────


async def _load(sf: SessionFactory, signup_id: UUID) -> SignupApproval:
    async with sf() as s:
        row = await s.get(SignupApproval, signup_id, populate_existing=True)
    if row is None:
        raise SignupError(404, "approval_not_found", f"No signup approval {signup_id!s}")
    return row


def _lease_free() -> ColumnElement[bool]:
    return or_(
        SignupApproval.steps_lease_until.is_(None),
        SignupApproval.steps_lease_until <= func.now(),
    )


def _legacy_record(legacy_identity: LegacyIdentity) -> dict[str, object]:
    """The named login + the evidence's FINGERPRINT (never its text)."""
    return {"user_id": legacy_identity.user_id, **evidence_fingerprint(legacy_identity.evidence)}


def _audit_fingerprint(legacy: dict[str, object] | None) -> dict[str, object]:
    """The audit's copy of the evidence: the same fingerprint, nothing else."""
    keys = ("evidence_sha256", "evidence_length", "evidence_created_at")
    return {f"legacy_{k}": (legacy or {}).get(k) for k in keys}


async def start_removal(
    sf: SessionFactory,
    signup_id: UUID,
    actor: SignupActor,
    *,
    confirm: str,
    legacy_identity: LegacyIdentity | None = None,
    allow_credentialed_delete: bool = False,
) -> SignupApproval:
    """``approved`` → ``removed`` (one guarded UPDATE, audited) and ENQUEUED: the
    row is due at once and the signup tick runs it (never inside the HTTP request
    — a run can outlast any proxy timeout). Raises :class:`SignupError` 404 / 409
    (``approval_not_approved``, ``removal_exists``, ``steps_in_progress``) / 422
    (``confirmation_mismatch``, ``legacy_identity_required``,
    ``legacy_identity_not_applicable``)."""
    row = await _load(sf, signup_id)
    if row.status == "removed":
        raise SignupError(409, "removal_exists", "This workspace's removal already exists.")
    if row.status != "approved":
        raise SignupError(
            409,
            "approval_not_approved",
            f"Only an approved early-access workspace can be removed (it is {row.status}).",
        )
    if confirm.strip() != (row.tenant_slug or ""):
        raise SignupError(
            422,
            "confirmation_mismatch",
            "Type the workspace slug exactly to confirm the removal.",
        )
    if legacy_identity is not None and has_identity_step(row):
        raise SignupError(
            422,
            "legacy_identity_not_applicable",
            "This signup minted its own login (identity step); no legacy login may be named.",
        )
    if legacy_identity is None and needs_legacy_identity(row):
        raise SignupError(
            422,
            "legacy_identity_required",
            "This signup was approved before logins were stamped: name the login "
            "(Keycloak user id) and the evidence that this signup created it.",
        )
    legacy = _legacy_record(legacy_identity) if legacy_identity else None
    removal: dict[str, object] = {
        "state": "pending",
        "attempts": 0,
        "requested_at": _iso(),
        "requested_by": str(actor.subject_id) if actor.subject_id else None,
        "requested_via": actor.audit,
        "tenant_id": str(row.tenant_id) if row.tenant_id else None,
        "tenant_slug": row.tenant_slug,
        "legacy_identity": legacy,
        "allow_credentialed_delete": allow_credentialed_delete,
        "steps": {},
        "blockers": [],
        "last_error": None,
    }
    async with sf() as s:
        got = (
            await s.execute(
                update(SignupApproval)
                .where(
                    SignupApproval.id == signup_id,
                    SignupApproval.status == "approved",
                    SignupApproval.tenant_slug == confirm.strip(),
                    _lease_free(),
                )
                .values(
                    status="removed",
                    removal=removal,
                    steps_lease_token=None,
                    steps_lease_until=None,
                    # Due NOW: the next signup tick claims and runs it.
                    steps_next_attempt_at=func.now(),
                    updated_at=func.now(),
                )
                .returning(SignupApproval)
            )
        ).scalar_one_or_none()
        if got is not None:
            await AuditRepository(s).record(
                actor_subject_id=actor.subject_id,
                action="signup.remove",
                target_tenant_id=None,  # survives the tenant delete (it erases its own audit)
                detail={
                    "phase": "requested",
                    "signup_id": str(signup_id),
                    "tenant_id": str(row.tenant_id) if row.tenant_id else None,
                    # The never-scrubbed audit keeps no company-derived slug.
                    "tenant_slug": slug_tombstone(signup_id),
                    "legacy_user_id": legacy_identity.user_id if legacy_identity else None,
                    **_audit_fingerprint(legacy),
                    "allow_credentialed_delete": allow_credentialed_delete,
                    **actor.audit_detail(),
                },
            )
        await s.commit()
    if got is None:
        now = await _load(sf, signup_id)
        if now.status != "approved":
            raise SignupError(409, "approval_not_approved", f"The signup is {now.status} now.")
        raise SignupError(
            409, "steps_in_progress", "Its provisioning steps are running right now; reload."
        )
    _log.info("signup removal requested (enqueued) id=%s by=%s", signup_id, actor.audit)
    return got


# ─── the lease ───────────────────────────────────────────────────────────


async def _claim(
    sf: SessionFactory, signup_id: UUID, *, manual: bool
) -> tuple[UUID, SignupApproval] | None:
    """Take the row's (free) lease for a run, COUNTING the attempt
    (CommonPractice: the claim is the attempt). A manual run starts the count
    again and re-opens a blocked / failed removal; the tick runs ``pending`` only."""
    token = uuid7()
    states = RETRY_STATES if manual else TICK_STATES
    attempts = text("1") if manual else text("coalesce((removal->>'attempts')::int, 0) + 1")
    guards: list[ColumnElement[bool]] = [
        SignupApproval.id == signup_id,
        SignupApproval.status == "removed",
        SignupApproval.removal["state"].astext.in_(states),
        _lease_free(),
    ]
    if not manual:
        # The backoff, re-checked IN the claim: a row the tick listed as due may
        # have been rescheduled since (another runner finished a pass).
        guards.append(SignupApproval.steps_next_attempt_at <= func.now())
    async with sf() as s:
        row = (
            await s.execute(
                update(SignupApproval)
                .where(*guards)
                .values(
                    removal=func.jsonb_set(
                        func.jsonb_set(
                            SignupApproval.removal,
                            text("'{attempts}'::text[]"),
                            func.to_jsonb(attempts),
                        ),
                        text("'{state}'::text[]"),
                        text("'\"pending\"'::jsonb"),
                    ),
                    steps_lease_token=token,
                    steps_lease_until=func.now() + _LEASE,
                    steps_next_attempt_at=func.now() + _LEASE,
                    updated_at=func.now(),
                )
                .returning(SignupApproval)
            )
        ).scalar_one_or_none()
        await s.commit()
    return (token, row) if row is not None else None


async def _write(
    sf: SessionFactory,
    signup_id: UUID,
    lease: UUID,
    removal: dict[str, object],
    **values: object,
) -> SignupApproval:
    """Store ``removal`` (+ any column ``values``) ONLY while ``lease`` is ours."""
    async with sf() as s:
        row = (
            await s.execute(
                update(SignupApproval)
                .where(SignupApproval.id == signup_id, SignupApproval.steps_lease_token == lease)
                .values(removal=removal, updated_at=func.now(), **values)
                .returning(SignupApproval)
            )
        ).scalar_one_or_none()
        await s.commit()
    if row is None:
        raise _LeaseLostError(f"signup {signup_id}: the removal lease was lost")
    return row


async def _release(sf: SessionFactory, signup_id: UUID, lease: UUID) -> None:
    try:
        async with sf() as s:
            await s.execute(
                update(SignupApproval)
                .where(SignupApproval.id == signup_id, SignupApproval.steps_lease_token == lease)
                .values(steps_lease_token=None, steps_lease_until=None)
            )
            await s.commit()
    except Exception:  # noqa: BLE001 — the lease expires on its own
        _log.warning("signup id=%s: removal lease release failed", signup_id, exc_info=True)


#: How often a running removal renews its lease (well inside the 5-min lease):
#: one long step (an export, a delete job, a slow licences call) never lets the
#: lease lapse and a second runner start the same removal.
LEASE_RENEW_SECONDS: float = STEPS_LEASE_SECONDS / 5


async def _renew(sf: SessionFactory, signup_id: UUID, lease: UUID) -> bool:
    """Extend the lease (and the crash fallback) while it is still ours."""
    async with sf() as s:
        got = (
            await s.execute(
                update(SignupApproval)
                .where(SignupApproval.id == signup_id, SignupApproval.steps_lease_token == lease)
                .values(
                    steps_lease_until=func.now() + _LEASE,
                    steps_next_attempt_at=func.now() + _LEASE,
                )
                .returning(SignupApproval.id)
            )
        ).scalar_one_or_none()
        await s.commit()
    return got is not None


async def _heartbeat(sf: SessionFactory, signup_id: UUID, lease: UUID, lost: asyncio.Event) -> None:
    """Renew the lease every :data:`LEASE_RENEW_SECONDS` until cancelled; a lost
    lease sets ``lost`` (the run stops before its next step)."""
    while True:
        await asyncio.sleep(LEASE_RENEW_SECONDS)
        try:
            if not await _renew(sf, signup_id, lease):
                lost.set()
                return
        except Exception:  # noqa: BLE001 — retried on the next beat; the lease has slack
            _log.warning("signup id=%s: removal lease renew failed", signup_id, exc_info=True)


# ─── the steps ───────────────────────────────────────────────────────────


@dataclass(slots=True)
class _Ctx:
    sf: SessionFactory
    deps: RemovalDeps
    row: SignupApproval
    actor: SignupActor

    @property
    def tenant_id(self) -> UUID:
        assert self.row.tenant_id is not None  # noqa: S101 — approved rows carry it (CHECK)
        return self.row.tenant_id

    @property
    def removal(self) -> dict[str, object]:
        return removal_of(self.row)


def _requester_subjects(row: SignupApproval) -> set[UUID]:
    """Every login the APPROVE itself recorded seating / creating: the identity
    step's user and the membership step's subject. An operator-named legacy
    login is NOT trusted here — :func:`_legacy_requester` verifies it first."""
    steps = steps_of(row)
    found: set[UUID] = set()
    for step in ("identity", "membership"):
        detail = (steps.get(step) or {}).get("detail")
        if isinstance(detail, dict):
            for key in ("user_id", "subject_id"):
                if (sub := _uuid_or_none(detail.get(key))) is not None:
                    found.add(sub)
    return found


def _legacy_named(row: SignupApproval) -> str | None:
    legacy = removal_of(row).get("legacy_identity")
    if isinstance(legacy, dict) and legacy.get("user_id"):
        return str(legacy["user_id"])
    return None


async def _check_legacy(
    ctx: _Ctx, user_id: str
) -> tuple[KeycloakUser | None, dict[str, object] | None, datetime.datetime | None]:
    """READ-ONLY proof that the operator-named ``user_id`` is this row's login:
    it is the address's one realm user (exact email lookup), created inside the
    approval window (``decided_at`` .. the ``invite_email`` step + 5 min), and it
    carries nothing another flow sets (foreign attributes, IdP links, groups,
    roles). Returns ``(user, None, created)`` when proven, ``(user, blocker,
    created)`` when not. A named login that does NOT exist is ``absent``
    (``(None, None, None)``) ONLY when no realm login carries the row's address
    either; any login for the address means the operator named the wrong id —
    the real login would stay live and the address be freed — so that is a
    ``legacy_identity_mismatch`` blocker ``(None, blocker, None)``."""
    kc = ctx.deps.identity
    row = ctx.row
    user = await kc.get_user(user_id)
    by_email = await kc.find_user_by_email(row.email)
    if user is None:
        if by_email is None:
            return None, None, None
        return (
            None,
            {
                "code": LEGACY_IDENTITY_MISMATCH,
                "message": f"the named login {user_id} does not exist, but the realm "
                f"login for the signup's address is {by_email.id}",
            },
            None,
        )
    if by_email is None or by_email.id != user_id:
        return (
            user,
            {
                "code": LEGACY_IDENTITY_MISMATCH,
                "message": f"the realm login for the signup's address is "
                f"{by_email.id if by_email else 'absent'}, not {user_id}",
            },
            None,
        )
    lower = row.decided_at
    upper = _parse_at((steps_of(row).get("invite_email") or {}).get("at"))
    created = (
        datetime.datetime.fromtimestamp(user.created_timestamp / 1000, datetime.UTC)
        if user.created_timestamp is not None
        else None
    )
    if (
        created is None
        or lower is None
        or upper is None
        or not (lower <= created <= upper + LEGACY_WINDOW_SLACK)
    ):
        return (
            user,
            {
                "code": LEGACY_IDENTITY_WINDOW,
                "message": f"the login was created at "
                f"{created.isoformat() if created else 'unknown'}, outside the approval "
                f"window {lower.isoformat() if lower else '?'} .. "
                f"{upper.isoformat() if upper else '?'} (+5 min)",
            },
            created,
        )
    foreign = await _legacy_foreign(kc, user, row.id)
    if foreign:
        return (
            user,
            {
                "code": LEGACY_IDENTITY_FOREIGN,
                "message": "the login carries what this signup never set: " + ", ".join(foreign),
            },
            created,
        )
    return user, None, created


async def _legacy_requester(ctx: _Ctx) -> tuple[UUID | None, dict[str, object] | None]:
    """The operator-named legacy login as a REQUESTER subject — only once it is
    proven this row's (:func:`_check_legacy`), or once the identity step already
    retired exactly that login (it was proven then, and may be gone now). A
    login that fails the proof is a BLOCKER: nothing is written."""
    user_id = _legacy_named(ctx.row)
    if user_id is None or has_identity_step(ctx.row):
        return None, None
    sub = _uuid_or_none(user_id)
    ident = _rsteps(ctx.removal).get("identity")
    detail = ident.get("detail") if isinstance(ident, dict) else None
    if _done(ident) and isinstance(detail, dict) and detail.get("user_id") == user_id:
        return sub, None
    user, problem, _created = await _check_legacy(ctx, user_id)
    if problem is not None:
        return None, problem
    return (sub if user is not None else None), None


async def _workspace_blockers(
    s: AsyncSession, ctx: _Ctx, requester: set[UUID]
) -> list[dict[str, object]]:
    """Other members, foreign invites, content, active runs (read-only)."""
    row = ctx.row
    blockers: list[dict[str, object]] = []
    members = (
        (
            await s.execute(
                select(TenantMember.subject_id).where(
                    TenantMember.tenant_id == ctx.tenant_id, TenantMember.status == "active"
                )
            )
        )
        .scalars()
        .all()
    )
    others = sorted(str(m) for m in members if m not in requester)
    if others:
        blockers.append(
            {
                "code": OTHER_MEMBER,
                "message": f"{len(others)} other active member(s) use this workspace",
                "subjects": others,
            }
        )
    foreign = (
        (
            await s.execute(
                select(TenantInvite.id).where(
                    TenantInvite.tenant_id == ctx.tenant_id,
                    TenantInvite.status == "pending",
                    func.lower(TenantInvite.email) != row.email.strip().lower(),
                )
            )
        )
        .scalars()
        .all()
    )
    if foreign:
        blockers.append(
            {
                "code": FOREIGN_INVITE,
                "message": f"{len(foreign)} pending invite(s) for someone else",
                "invites": [str(i) for i in foreign],
            }
        )
    counts = await offboarding.row_counts(s, ctx.tenant_id)
    content = {t: n for t, n in counts.items() if n and t not in ENGINE_TABLES}
    if content:
        blockers.append(
            {
                "code": TENANT_CONTENT,
                "message": "the workspace holds content: "
                + ", ".join(f"{t}={n}" for t, n in sorted(content.items())),
                "tables": content,
            }
        )
    if await tenant_has_active_runs(s, ctx.tenant_id):
        blockers.append({"code": ACTIVE_RUNS, "message": "a run of the workspace is active"})
    return blockers


async def _blockers(ctx: _Ctx) -> dict[str, object]:
    """Re-checked EVERY run, READ-ONLY: any blocker → ``blocked`` before a single
    write. An operator-named legacy login counts as the requester only once it
    is proven this row's; a login that fails the proof is itself a blocker."""
    row = ctx.row
    blockers: list[dict[str, object]] = []
    provisioned = (steps_of(row).get("provision") or {}).get("detail")
    recorded = provisioned.get("tenant_id") if isinstance(provisioned, dict) else None
    if recorded != str(row.tenant_id):
        blockers.append(
            {
                "code": TENANT_MISMATCH,
                "message": f"the row's tenant {row.tenant_id} is not the one its provision step "
                f"recorded ({recorded or 'none'})",
            }
        )
        return _rec("blocked", code=TENANT_MISMATCH, detail={"blockers": blockers})
    async with ctx.sf() as s:
        tenant = await s.get(Tenant, ctx.tenant_id)
        gone = tenant is not None and tenant.status == "deleted"
        await s.rollback()
    legacy_sub, legacy_problem = (None, None) if gone else await _legacy_requester(ctx)
    if legacy_problem is not None:
        blockers.append(legacy_problem)
    async with ctx.sf() as s:
        tenant = await s.get(Tenant, ctx.tenant_id)
        if tenant is None:
            blockers.append(
                {"code": TENANT_NOT_PROVISIONED, "message": "no tenants row exists for it"}
            )
            return _rec("blocked", code=TENANT_NOT_PROVISIONED, detail={"blockers": blockers})
        if tenant.status == "deleted":
            # Only this removal's own tenant_delete (or an earlier one) gets here:
            # nothing of the tenant is left to protect.
            return _rec("ok", code="tenant_deleted")
        requester = _requester_subjects(row)
        if legacy_sub is not None:
            requester.add(legacy_sub)
        blockers.extend(await _workspace_blockers(s, ctx, requester))
        await s.rollback()
    if blockers:
        return _rec("blocked", code=str(blockers[0]["code"]), detail={"blockers": blockers})
    return _rec("ok")


async def _close_access(ctx: _Ctx) -> dict[str, object]:
    async with ctx.sf() as s:
        tenant = (
            await s.execute(select(Tenant).where(Tenant.id == ctx.tenant_id).with_for_update())
        ).scalar_one()
        before = tenant.status
        suspended = before not in ("suspended", "deleted")
        if suspended:
            tenant.status = "suspended"
            tenant.updated_at = _now()
            await AuditRepository(s).record(
                actor_subject_id=ctx.actor.subject_id,
                action="tenant.update",
                target_tenant_id=ctx.tenant_id,
                detail={
                    "before": {"name": tenant.name, "status": before},
                    "after": {"name": tenant.name, "status": "suspended"},
                    "signup_id": str(ctx.row.id),
                    "reason": "signup.remove",
                    **ctx.actor.audit_detail(),
                },
            )
        revoked = (
            (
                await s.execute(
                    update(TenantInvite)
                    .where(
                        TenantInvite.tenant_id == ctx.tenant_id, TenantInvite.status == "pending"
                    )
                    .values(status="revoked")
                    .returning(TenantInvite.id)
                )
            )
            .scalars()
            .all()
        )
        await s.commit()
    # This pod stops admitting the tenant at once (others within the cache TTL).
    from src.api.routes import _common

    _common._known_tenant_cache.pop(ctx.tenant_id, None)
    return _rec(
        "ok",
        detail={
            "status_before": before,
            "suspended": suspended,
            "revoked_invites": [str(i) for i in revoked],
        },
    )


def _is_live(lic: object) -> bool:
    return not (getattr(lic, "revoked_at", None) or getattr(lic, "status", None) == "revoked")


def _is_dependency(lic: LicenseView) -> bool:
    """A shared DEPENDENCY licence (licences migration 059: provisioned by the
    bundle ensure only because a bundled app requires it; w2v may not revoke it
    while another licence needs it). ``is_dependency`` when the listing carries
    it; else the ``metadata.dependency_of`` marker the ensure writes. A licence a
    bundle ADOPTED carries ``bundle_id`` and is ours to revoke."""
    if lic.bundle_id:
        return False
    if lic.is_dependency is not None:
        return lic.is_dependency
    return bool((lic.metadata or {}).get("dependency_of"))


def _app_name(lic: LicenseView) -> str:
    if lic.app is not None:
        return lic.app.key or lic.app.name or lic.app.id
    return lic.app_id or lic.id


def _required_by(exc: LicensesClientError) -> list[dict[str, str | None]]:
    """The licences a 409 ``dependency_still_required`` names as still needing the
    dependency (``detail.required_by``: ``license_id`` / ``tenant_id`` / ``app``)."""
    rows = (exc.detail or {}).get("required_by")
    found: list[dict[str, str | None]] = []
    for raw in rows if isinstance(rows, list) else []:
        if not isinstance(raw, dict):
            continue
        item = cast(dict[str, object], raw)
        app = item.get("app_key") or item.get("app")
        if isinstance(app, dict):
            app = cast(dict[str, object], app).get("key")
        found.append(
            {
                "license_id": str(item["license_id"]) if item.get("license_id") else None,
                "tenant_id": str(item["tenant_id"]) if item.get("tenant_id") else None,
                "app": str(app) if app else None,
            }
        )
    return found


async def _release_one(
    ctx: _Ctx, dep: LicenseView, ours: set[str]
) -> tuple[str, dict[str, object]]:
    """One explicit owner revoke of a dependency licence → (outcome, record):
    ``released`` | ``kept`` (only another tenant's licence needs it) |
    ``still_required`` (a licence on THIS tenant that is not ours, or one
    licences did not name) | ``waiting`` (one of OUR dependencies needs it: a
    rerun releases it) | ``unsupported`` (403) | ``failed``."""
    tenant = str(ctx.tenant_id)
    row: dict[str, object] = {"license_id": dep.id, "app": _app_name(dep)}
    try:
        await ctx.deps.licenses.revoke_license(dep.id, tenant_id=tenant)
    except LicensesClientError as exc:
        row["status"] = exc.status_code
        if exc.status_code == 403:
            return "unsupported", {**row, "code": exc.code}
        if exc.status_code != 409 or exc.code != DEPENDENCY_STILL_REQUIRED:
            return "failed", {**row, "code": exc.code}
        requirers = _required_by(exc)
        row["required_by"] = requirers
        here = [r for r in requirers if r["tenant_id"] in (None, tenant)]
        if not requirers or any(r["license_id"] not in ours for r in here):
            return "still_required", row
        return ("waiting" if here else "kept"), row
    return "released", row


def _still_required_blocker(rows: list[dict[str, object]]) -> dict[str, object]:
    apps = sorted({str(r["app"]) for r in rows})
    requiring = sorted(
        {
            str(q["app"] or q["license_id"])
            for r in rows
            for q in cast(list[dict[str, object]], r["required_by"])
        }
    )
    return {
        "code": DEPENDENCY_STILL_REQUIRED,
        "message": f"dependency licence(s) {', '.join(apps)} still required by "
        f"{', '.join(requiring) or 'a licence licences did not name'} on this tenant "
        "that the removal does not own; an operator must decide",
        "apps": apps,
        "required_by": requiring,
        "licences": [r["license_id"] for r in rows],
    }


async def _release_dependencies(
    ctx: _Ctx, live_deps: list[LicenseView], ours: set[str]
) -> tuple[StepStatus, str | None, str | None, dict[str, object]]:
    """The cascade left dependency licences active: try the owner's explicit
    revoke ONCE per row (:func:`_release_one`). Returns (status, code, error,
    detail): ``ok`` when each was released or is kept only for another tenant's
    licence (the archive retires it for this tenant); ``blocked``
    ``dependency_still_required`` (operator); ``pending``
    ``dependency_release_unsupported`` (403, retried) or ``licence_failed``."""
    out: dict[str, list[dict[str, object]]] = {}
    for dep in live_deps:
        outcome, row = await _release_one(ctx, dep, ours)
        out.setdefault(outcome, []).append(row)
    detail: dict[str, object] = {
        "released_explicitly": [r["license_id"] for r in out.pop("released", [])]
    }
    if kept := out.pop("kept", []):
        detail["kept_for_other_tenants"] = kept
    detail.update(out)
    if rows := out.get("still_required"):
        blocker = _still_required_blocker(rows)
        detail["blockers"] = [blocker]
        return "blocked", DEPENDENCY_STILL_REQUIRED, str(blocker["message"]), detail
    if rows := out.get("unsupported"):
        return (
            "pending",
            DEPENDENCY_RELEASE_UNSUPPORTED,
            f"licences refused to release {len(rows)} dependency licence(s) (403): "
            "it does not support dependency release yet",
            detail,
        )
    if out:
        pending = sum(len(v) for v in out.values())
        return "pending", LICENCE_FAILED, f"{pending} dependency licence(s) not released", detail
    return "ok", None, None, detail


async def _confirm_dependencies(
    ctx: _Ctx, after: list[LicenseView], detail: dict[str, object]
) -> dict[str, object] | None:
    """Every dependency licence must be released once our own are revoked.
    ``None`` = confirmed; else the step record (``detail`` is filled in)."""
    live_deps = [lic for lic in after if _is_live(lic) and _is_dependency(lic)]
    if not live_deps:
        return None
    detail["dependencies_live_after"] = sorted(lic.id for lic in live_deps)
    status, code, error, sub = await _release_dependencies(
        ctx, live_deps, {lic.id for lic in after}
    )
    detail.update(sub)
    if status != "ok":
        return _rec(status, code=code, error=error, detail=detail)
    kept = {
        str(k["license_id"])
        for k in cast(list[dict[str, object]], sub.get("kept_for_other_tenants", []))
    }
    left = [
        lic.id
        for lic in await ctx.deps.licenses.list_tenant_licenses(tenant_id=str(ctx.tenant_id))
        if _is_live(lic) and _is_dependency(lic) and lic.id not in kept
    ]
    if left:
        detail["dependencies_still_live"] = left
        return _rec(
            "pending",
            code=LICENCE_FAILED,
            error=f"{len(left)} dependency licence(s) still live after the release",
            detail=detail,
        )
    return None


async def _licences(ctx: _Ctx) -> dict[str, object]:
    licences = ctx.deps.licenses
    tenant = str(ctx.tenant_id)
    listed = await licences.list_tenant_licenses(tenant_id=tenant)
    # Every listed licence must be THIS tenant's before a single revoke: a
    # listing that leaks another tenant's licence blocks, with zero writes.
    foreign = sorted(lic.id for lic in listed if lic.tenant_id != tenant)
    if foreign:
        blocker = {
            "code": TENANT_MISMATCH,
            "message": f"the licences listing for tenant {tenant} returned "
            f"{len(foreign)} licence(s) of another tenant; nothing was revoked",
            "licences": foreign,
        }
        return _rec(
            "blocked",
            code=TENANT_MISMATCH,
            error=str(blocker["message"]),
            detail={"listed": len(listed), "foreign": foreign, "blockers": [blocker]},
        )
    # Only the tenant's OWN licences are revoked; the shared DEPENDENCY licences
    # (w2v may not revoke them: 403 ``app_not_issuable_by_caller``) are released
    # by licences when the last licence requiring them goes.
    live = [lic.id for lic in listed if _is_live(lic) and not _is_dependency(lic)]
    revoked: list[str] = []
    released: list[str] = []
    refused: list[dict[str, object]] = []
    for license_id in live:
        try:
            released += await licences.revoke_license(license_id, tenant_id=tenant)
            revoked.append(license_id)
        except LicensesClientError as exc:
            refused.append({"license_id": license_id, "status": exc.status_code, "code": exc.code})
    detail: dict[str, object] = {
        "listed": len(listed),
        "already_revoked": sorted(lic.id for lic in listed if not _is_live(lic)),
        "revoked": revoked,
        "dependencies": sorted(lic.id for lic in listed if _is_dependency(lic)),
    }
    if released:
        detail["released_dependencies"] = sorted(set(released))
    if refused:
        detail["refused"] = refused
        return _rec(
            "pending",
            code=LICENCE_FAILED,
            error=f"licences refused {len(refused)} of {len(live)} revocations",
            detail=detail,
        )
    # CONFIRM the count: a fresh listing must hold no live licence of our own,
    # and every dependency must have been released with them.
    after = await licences.list_tenant_licenses(tenant_id=tenant)
    still_live = [lic.id for lic in after if _is_live(lic) and not _is_dependency(lic)]
    detail["live_after"] = len(still_live)
    if still_live:
        detail["still_live"] = still_live
        return _rec(
            "pending",
            code=LICENCE_FAILED,
            error=f"{len(still_live)} licence(s) still live after the revoke",
            detail=detail,
        )
    if (dep_rec := await _confirm_dependencies(ctx, after, detail)) is not None:
        return dep_rec
    # The GATE verdict, read FRESH (the status cache is evicted and bypassed).
    invalidate_license_status(tenant_id=tenant, app_key=W2V_LICENCE_APP_KEY, subject_id=None)
    verdict = await get_license_status(
        client=licences,
        app_key=W2V_LICENCE_APP_KEY,
        tenant_id=tenant,
        subject_id=None,
        gate_enabled=True,
        now=_now(),
    )
    detail["gate_state"] = verdict.state
    if verdict.state not in DEAD_LICENCE_STATES:
        return _rec(
            "pending",
            code=LICENCE_FAILED,
            error=f"the tenant's {W2V_LICENCE_APP_KEY!r} licence still reads {verdict.state!r}",
            detail=detail,
        )
    return _rec("ok", detail=detail)


async def _referenced_elsewhere(ctx: _Ctx, user_id: str) -> bool:
    """The login is used by ANOTHER w2v tenant: a membership row for its sub, or
    an invite there accepted by it / pending for the address."""
    sub = _uuid_or_none(user_id)
    async with ctx.sf() as s:
        member = (
            await s.execute(
                select(TenantMember.tenant_id)
                .where(TenantMember.subject_id == sub, TenantMember.tenant_id != ctx.tenant_id)
                .limit(1)
            )
        ).first()
        invite = (
            await s.execute(
                select(TenantInvite.id)
                .where(
                    TenantInvite.tenant_id != ctx.tenant_id,
                    or_(
                        TenantInvite.accepted_subject_id == sub,
                        (TenantInvite.status == "pending")
                        & (func.lower(TenantInvite.email) == ctx.row.email.strip().lower()),
                    ),
                )
                .limit(1)
            )
        ).first()
    return member is not None or invite is not None


def removed_marker(signup_id: UUID) -> str:
    return f"{REMOVED_MARKER_PREFIX}{signup_id}"


async def _retire(
    ctx: _Ctx, user_id: str, *, delete: bool, evidence: dict[str, object]
) -> dict[str, object]:
    """Delete the login, or disable + mark it (ONE read-modify-write, read back)."""
    kc = ctx.deps.identity
    if delete:
        gone = await kc.delete_user(user_id)
        result = "deleted" if gone else "absent"
    else:
        marker = removed_marker(ctx.row.id)
        stored = await kc.set_user_state(
            user_id, enabled=False, attributes={SIGNUP_APP_ATTRIBUTE: marker}
        )
        if stored is None:
            result = "absent"
        elif stored.enabled:
            raise KeycloakAdminError(f"set_user_state: login {user_id} is still enabled")
        else:
            kept = (stored.attributes or {}).get(SIGNUP_APP_ATTRIBUTE)
            result = "disabled" if kept == [marker] else "disabled_unmarked"
    async with ctx.sf() as s:
        await AuditRepository(s).record(
            actor_subject_id=ctx.actor.subject_id,
            action="signup.remove",
            target_tenant_id=None,
            detail={
                "phase": "identity",
                "signup_id": str(ctx.row.id),
                "user_id": user_id,
                "result": result,
                **evidence,
                **ctx.actor.audit_detail(),
            },
        )
        await s.commit()
    return _rec("ok", code=result, detail={"user_id": user_id, "result": result, **evidence})


async def _credential_state(kc: KeycloakAdminClient, user_id: str) -> tuple[int, int]:
    return await kc.count_credentials(user_id), await kc.count_federated_identities(user_id)


def _parse_at(raw: object) -> datetime.datetime | None:
    if not isinstance(raw, str):
        return None
    try:
        at = datetime.datetime.fromisoformat(raw)
    except ValueError:
        return None
    return at if at.tzinfo else at.replace(tzinfo=datetime.UTC)


async def _legacy_foreign(kc: KeycloakAdminClient, user: KeycloakUser, row_id: UUID) -> list[str]:
    """What marks ``user`` as NOT a plain early-access login of this row."""
    found: list[str] = []
    attrs = user.attributes or {}
    for name in LEGACY_FOREIGN_ATTRIBUTES:
        if any(v for v in attrs.get(name) or []):
            found.append(f"attribute:{name}")
    # ANY signup_app other than w2v's own (``w2v``, or this row's removed marker
    # from an earlier run of this very removal) belongs to another flow / app.
    own = {SIGNUP_APP_KEY, removed_marker(row_id)}
    apps = [v for v in attrs.get(SIGNUP_APP_ATTRIBUTE) or [] if v]
    if foreign_apps := sorted(v for v in apps if v not in own):
        found.append("attribute:signup_app=" + ",".join(foreign_apps))
    if await kc.count_federated_identities(user.id):
        found.append("federated_identities")
    if await kc.count_groups(user.id):
        found.append("groups")
    roles = await kc.get_user_role_mappings(user.id)
    realm = [r for r in roles.realm if not r.startswith("default-roles-")]
    if realm:
        found.append("realm_roles:" + ",".join(realm))
    clients = sorted(c for c, names in roles.clients.items() if names)
    if clients:
        found.append("client_roles:" + ",".join(clients))
    return found


async def _identity(ctx: _Ctx) -> dict[str, object]:
    """Retire the login this signup created. Shared precondition: no OTHER w2v
    tenant references it (``identity_kept_in_use`` otherwise — kept as is)."""
    if has_identity_step(ctx.row):
        return await _identity_stamped(ctx)
    return await _identity_legacy(ctx)


async def _identity_stamped(ctx: _Ctx) -> dict[str, object]:
    """The identity step's user, only while it carries THIS row's stamp
    (CommonPractice ``retireRejectedIdentity``): no credential and no IdP link →
    deleted; otherwise disabled + marked ``signup_app=w2v_removed:<id>``."""
    kc = ctx.deps.identity
    row = ctx.row
    detail = (steps_of(row).get("identity") or {}).get("detail")
    user_id = detail.get("user_id") if isinstance(detail, dict) else None
    if not isinstance(user_id, str) or not user_id:
        return _rec("skipped", code="no_identity", detail={"mode": "stamped"})
    base: dict[str, object] = {"user_id": user_id, "mode": "stamped"}
    user = await kc.get_user(user_id)
    if user is None:
        return _rec("ok", code="absent", detail=base)
    if not belongs_to(user, str(row.id)):
        # Someone else's login (e.g. ``identity_exists`` recorded the FOREIGN
        # user): never touched.
        return _rec("ok", code=IDENTITY_NOT_OWNED, detail=base)
    if await _referenced_elsewhere(ctx, user_id):
        return _rec("ok", code=IDENTITY_KEPT_IN_USE, detail=base)
    credentials, links = await _credential_state(kc, user_id)
    return await _retire(
        ctx,
        user_id,
        delete=credentials == 0 and links == 0,
        evidence={"mode": "stamped", "credentials": credentials, "idp_links": links},
    )


async def _identity_legacy(ctx: _Ctx) -> dict[str, object]:
    """A row approved before logins were stamped: the operator names the login,
    and it must be PROVEN this row's (else ``blocked``): it is the address's one
    realm user, created inside the approval window (``decided_at`` .. the
    ``invite_email`` step + 5 min), and carries nothing another flow sets. Then
    deleted when it holds no credential (or ``allow_credentialed_delete``),
    else disabled + marked."""
    kc = ctx.deps.identity
    removal = ctx.removal
    user_id = _legacy_named(ctx.row)
    if user_id is None:
        return await _identity_unnamed(ctx)
    base: dict[str, object] = {"user_id": user_id, "mode": "legacy"}
    # Re-proven here too (the blockers step proved it this run; the login may
    # have changed since): the same read-only rule.
    user, problem, created = await _check_legacy(ctx, user_id)
    if problem is not None:
        return _rec(
            "blocked",
            code=str(problem["code"]),
            error=str(problem["message"]),
            detail={**base, "blockers": [problem]},
        )
    if user is None:
        # Proven absent: neither the named login nor any login for the address.
        return _rec("ok", code="absent", detail=base)
    if await _referenced_elsewhere(ctx, user_id):
        return _rec("ok", code=IDENTITY_KEPT_IN_USE, detail=base)
    credentials = await kc.count_credentials(user_id)
    allow = bool(removal.get("allow_credentialed_delete"))
    return await _retire(
        ctx,
        user_id,
        delete=credentials == 0 or allow,
        evidence={
            "mode": "legacy",
            "credentials": credentials,
            "allow_credentialed_delete": allow,
            "created_at": created.isoformat() if created else None,
        },
    )


def _created(user: KeycloakUser) -> datetime.datetime | None:
    if user.created_timestamp is None:
        return None
    return datetime.datetime.fromtimestamp(user.created_timestamp / 1000, datetime.UTC)


def _approval_window(
    row: SignupApproval,
) -> tuple[datetime.datetime | None, datetime.datetime | None]:
    """``decided_at`` .. the LATEST step the approve recorded (+ the slack is
    the caller's): a login created in there could be one this approve made."""
    ats = [
        at
        for rec in steps_of(row).values()
        if isinstance(rec, dict) and (at := _parse_at(rec.get("at"))) is not None
    ]
    return row.decided_at, (max(ats) if ats else None)


async def _identity_unnamed(ctx: _Ctx) -> dict[str, object]:
    """No identity step, no login named (the ``invite_email`` step never landed,
    so none was required). A login may still exist — e.g. the invite created it
    and then failed to send. LOOK IT UP by the row's address before recording
    ``no_identity``: a login this signup could have created — stamped for THIS
    row, or created inside the approval window and carrying no foreign marker —
    is retired by the same rules as a stamped one (kept when another w2v tenant
    uses it; deleted only with no credential and no IdP link, else disabled +
    marked). Any other login for the address is someone else's: never touched."""
    kc = ctx.deps.identity
    row = ctx.row
    user = await kc.find_user_by_email(row.email)
    if user is None:
        return _rec("skipped", code="no_identity", detail={"mode": "unnamed", "looked_up": True})
    base: dict[str, object] = {"user_id": user.id, "mode": "unnamed"}
    stamped = belongs_to(user, str(row.id))
    created = _created(user)
    if not stamped:
        lower, upper = _approval_window(row)
        if (
            created is None
            or lower is None
            or upper is None
            or not (lower <= created <= upper + LEGACY_WINDOW_SLACK)
        ):
            return _rec(
                "ok", code=IDENTITY_NOT_OWNED, detail={**base, "why": "outside_approval_window"}
            )
        if foreign := await _legacy_foreign(kc, user, row.id):
            return _rec("ok", code=IDENTITY_NOT_OWNED, detail={**base, "foreign": foreign})
    if await _referenced_elsewhere(ctx, user.id):
        return _rec("ok", code=IDENTITY_KEPT_IN_USE, detail=base)
    credentials, links = await _credential_state(kc, user.id)
    return await _retire(
        ctx,
        user.id,
        delete=credentials == 0 and links == 0,
        evidence={
            "mode": "unnamed",
            "stamped": stamped,
            "credentials": credentials,
            "idp_links": links,
            "created_at": created.isoformat() if created else None,
        },
    )


async def _job_status(sf: SessionFactory, job_id: UUID) -> TenantOffboarding | None:
    async with sf() as s:
        return await s.get(TenantOffboarding, job_id, populate_existing=True)


async def _export(ctx: _Ctx) -> dict[str, object]:
    async with ctx.sf() as s:
        if await offboarding.has_completed_export(s, ctx.tenant_id):
            return _rec("skipped", code="export_exists")
    storage = ctx.deps.storage
    if storage is None:
        return _rec("pending", code=STORAGE_UNAVAILABLE, error="artifact storage is unavailable")
    async with ctx.sf() as s:
        tenant = (
            await s.execute(select(Tenant).where(Tenant.id == ctx.tenant_id).with_for_update())
        ).scalar_one()
        try:
            job = await offboarding.start_tenant_export(s, tenant, ctx.actor.subject_id)
        except offboarding.StartRefusedError as exc:
            await s.rollback()
            if exc.code == "EXPORT_IN_PROGRESS":
                return _rec("in_progress", code=IN_PROGRESS, detail=exc.details)
            return _rec("pending", code=EXPORT_FAILED, error=f"{exc.code}: {exc.message}")
        job_id = job.id
    await offboarding.execute_job(ctx.sf, storage, job_id, "export")
    done = await _job_status(ctx.sf, job_id)
    if done is not None and done.status == "done":
        return _rec("ok", detail={"job_id": str(job_id)})
    if done is not None and done.status == "failed":
        return _rec("pending", code=EXPORT_FAILED, error=done.error, detail={"job_id": str(job_id)})
    return _rec("in_progress", code=IN_PROGRESS, detail={"job_id": str(job_id)})


async def _tenant_delete(ctx: _Ctx) -> dict[str, object]:
    storage = ctx.deps.storage
    if storage is None:
        return _rec("pending", code=STORAGE_UNAVAILABLE, error="artifact storage is unavailable")
    async with ctx.sf() as s:
        tenant = (
            await s.execute(select(Tenant).where(Tenant.id == ctx.tenant_id).with_for_update())
        ).scalar_one()
        try:
            started = await offboarding.start_tenant_delete(
                s,
                ctx.deps.licenses,
                tenant,
                ctx.actor.subject_id,
                force=False,
                refuse_forced=True,
            )
        except offboarding.StartRefusedError as exc:
            await s.rollback()
            if exc.code == "FORCED_JOB_EXISTS":
                blocker = {
                    "code": FORCED_JOB_EXISTS,
                    "message": "an earlier admin delete of this tenant was FORCED; the removal "
                    "never resumes a forced job — finish or cancel it in the tenant admin, "
                    "then Retry (or Resolve)",
                    **exc.details,
                }
                return _rec(
                    "blocked",
                    code=FORCED_JOB_EXISTS,
                    error=exc.message,
                    detail={**exc.details, "blockers": [blocker]},
                )
            if exc.code == "DELETE_IN_PROGRESS":
                return _rec("in_progress", code=IN_PROGRESS, detail=exc.details)
            if exc.code == "ALREADY_DELETED":
                return _rec("ok", code="already_deleted")
            code = (
                TENANT_DELETE_BLOCKED if exc.code == "OFFBOARDING_BLOCKED" else TENANT_DELETE_FAILED
            )
            return _rec(
                "pending", code=code, error=f"{exc.code}: {exc.message}", detail=exc.details
            )
        job_id, force = started.job.id, started.job.force
        if not started.started:
            return _rec("ok", detail={"job_id": str(job_id), "force": force})
    await offboarding.execute_job(ctx.sf, storage, job_id, "delete")
    done = await _job_status(ctx.sf, job_id)
    detail: dict[str, object] = {"job_id": str(job_id), "force": force}
    if done is not None and done.status == "done":
        return _rec("ok", detail=detail)
    if done is not None and done.status == "failed":
        # The next run starts it again: it resumes from its recorded progress.
        return _rec("pending", code=TENANT_DELETE_FAILED, error=done.error, detail=detail)
    return _rec("in_progress", code=IN_PROGRESS, detail=detail)


async def _licences_archive(ctx: _Ctx) -> dict[str, object]:
    archived_at = await ctx.deps.licenses.archive_tenant(str(ctx.tenant_id))
    return _rec("ok", detail={"archived_at": archived_at})


def erased_email(signup_id: UUID) -> str:
    return f"erased+{signup_id}@invalid"


def _scrub(value: object, needle: str, replacement: str) -> object:
    """Every occurrence of ``needle`` (case-insensitive, also INSIDE a longer
    string: evidence, notes, error text) → ``replacement``."""
    if isinstance(value, dict):
        return {k: _scrub(v, needle, replacement) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(v, needle, replacement) for v in value]
    if isinstance(value, str) and needle:
        return re.sub(re.escape(needle), replacement, value, flags=re.IGNORECASE)
    return value


def slug_tombstone(signup_id: UUID) -> str:
    """What replaces the (company-derived) workspace slug once the data is erased."""
    return f"removed-{str(signup_id)[:8]}"


def _tombstone_slug(
    removal: dict[str, object], steps: dict[str, object], tomb: str
) -> tuple[dict[str, object], dict[str, object]]:
    """``removal.tenant_slug`` and ``steps.provision.detail.tenant_slug`` → the
    tombstone; ``tenant_id`` (the referential key) is kept everywhere."""
    removal = {**removal, "tenant_slug": tomb}
    provision = steps.get("provision")
    if isinstance(provision, dict):
        detail = provision.get("detail")
        if isinstance(detail, dict) and "tenant_slug" in detail:
            steps = {**steps, "provision": {**provision, "detail": {**detail, "tenant_slug": tomb}}}
    return removal, steps


def _erasure(row: SignupApproval, removal: dict[str, object]) -> dict[str, object]:
    """The column values that erase the requester's personal data (finalize and
    resolve alike): the removal / steps JSON scrubbed of the address, and the
    company-derived workspace slug replaced by :func:`slug_tombstone` (in the
    removal and in the provision step's detail). Only those FIELDS: a substring
    replace of a short slug could hit a UUID value and break the references."""
    needle = row.email.strip().lower()
    erased = erased_email(row.id)
    tomb = slug_tombstone(row.id)
    removal_, steps_ = _tombstone_slug(removal, dict(row.steps or {}), tomb)
    return {
        "removal": _scrub(removal_, needle, erased),
        "steps": _scrub(steps_, needle, erased),
        "tenant_slug": None,
        "tenant_name": None,
        "name": "",
        "company": None,
        "message": None,
        "email": erased,
    }


ERASED_FIELDS: tuple[str, ...] = ("name", "email", "company", "message")


async def _purge_exports(ctx: _Ctx) -> dict[str, object]:
    """finalize's first half: delete the tenant's export archive(s)
    (``exports/tenants/{t}/``) — they hold the signup's name, address, company,
    message and the invite / audit addresses, so "personal data erased" is only
    true once they are gone. Runs only after ``tenant_delete`` landed (the steps
    run in order). Storage unavailable, a refused delete, a key that survives or
    any error → ``pending`` ``export_purge_failed`` (retried), never ``done``."""
    storage = ctx.deps.storage
    prefix = offboarding.export_prefix(ctx.tenant_id)
    if storage is None:
        return _rec(
            "pending",
            code=EXPORT_PURGE_FAILED,
            error="artifact storage is unavailable: the export archive is not deleted yet",
            detail={"export_prefix": prefix},
        )
    try:
        keys = sorted(await storage.list_object_keys(prefix))
        if keys and not await storage.delete_object_prefix(prefix):
            return _rec(
                "pending",
                code=EXPORT_PURGE_FAILED,
                error="the storage refused to delete the export archive",
                detail={"export_prefix": prefix, "keys": keys},
            )
        left = sorted(await storage.list_object_keys(prefix))
    except Exception as exc:  # noqa: BLE001 — recorded on the row, retried
        return _rec("pending", code=EXPORT_PURGE_FAILED, error=_err_text(exc))
    if left:
        return _rec(
            "pending",
            code=EXPORT_PURGE_FAILED,
            error=f"{len(left)} export object(s) survived the delete",
            detail={"export_prefix": prefix, "left": left},
        )
    return _rec("ok", detail={"export_prefix": prefix, "export_keys_deleted": keys})


_STEP_FNS = {
    "blockers": _blockers,
    "close_access": _close_access,
    "licences": _licences,
    "identity": _identity,
    "export": _export,
    "tenant_delete": _tenant_delete,
    "licences_archive": _licences_archive,
}


# ─── the run ─────────────────────────────────────────────────────────────


@dataclass(slots=True)
class RemovalRun:
    ran: bool
    row: SignupApproval
    reason: str | None = None
    order: list[str] = field(default_factory=list)


async def _run_one(ctx: _Ctx, step: str) -> dict[str, object]:
    try:
        return await _STEP_FNS[step](ctx)
    except _LeaseLostError:
        raise
    except Exception as exc:  # noqa: BLE001 — recorded on the row, retried
        _log.warning("signup removal id=%s step=%s failed: %s", ctx.row.id, step, _err_text(exc))
        code = _STEP_FALLBACK_CODE[step]
        return _rec("pending", code=code, error=_err_text(exc))


async def _finalize(
    sf: SessionFactory,
    row: SignupApproval,
    lease: UUID,
    removal: dict[str, object],
    actor: SignupActor,
    *,
    purged: dict[str, object],
) -> SignupApproval:
    steps = _rsteps(removal)
    steps["finalize"] = _rec("ok", detail={"erased": list(ERASED_FIELDS), **purged})
    final = {**removal, "steps": steps, "state": "done", "finished_at": _iso()}
    async with sf() as s:
        done = (
            await s.execute(
                update(SignupApproval)
                .where(SignupApproval.id == row.id, SignupApproval.steps_lease_token == lease)
                .values(
                    **_erasure(row, final),
                    steps_next_attempt_at=None,
                    updated_at=func.now(),
                )
                .returning(SignupApproval)
            )
        ).scalar_one_or_none()
        if done is None:
            await s.rollback()
            raise _LeaseLostError(f"signup {row.id}: the removal lease was lost")
        await AuditRepository(s).record(
            actor_subject_id=actor.subject_id,
            action="signup.remove",
            target_tenant_id=None,
            detail={
                "phase": "completed",
                "signup_id": str(row.id),
                "tenant_id": str(row.tenant_id) if row.tenant_id else None,
                # The never-scrubbed audit keeps no company-derived slug.
                "tenant_slug": slug_tombstone(row.id),
                "export_keys_deleted": purged.get("export_keys_deleted", []),
                **actor.audit_detail(),
            },
        )
        await s.commit()
    return done


async def run_removal(
    sf: SessionFactory,
    deps: RemovalDeps,
    signup_id: UUID,
    *,
    actor: SignupActor | None = None,
    manual: bool = False,
) -> RemovalRun:
    """Run every removal step that has not landed, in order, under the row's lease
    (renewed every :data:`LEASE_RENEW_SECONDS` while the run lasts). The first
    step that does not land ends the run (each depends on the ones before it).
    Then the next attempt is scheduled (or ``failed`` after 10)."""
    first = await _load(sf, signup_id)
    if first.status != "removed":
        return RemovalRun(ran=False, row=first, reason="not_removed")
    claimed = await _claim(sf, signup_id, manual=manual)
    if claimed is None:
        return RemovalRun(ran=False, row=first, reason="lease_busy_or_not_runnable")
    lease, row = claimed
    order: list[str] = []
    lost = asyncio.Event()
    beat = asyncio.create_task(_heartbeat(sf, signup_id, lease, lost))
    try:
        row = await _run_steps(sf, deps, row, lease, actor or removal_actor(row), order, lost)
        await _schedule(sf, signup_id, lease, row)
    except _LeaseLostError:
        _log.warning("signup removal id=%s: lease lost mid-run", signup_id)
    finally:
        beat.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await beat
        await _release(sf, signup_id, lease)
    final = await _load(sf, signup_id)
    _notify_transition(removal_state(first), final)
    return RemovalRun(ran=True, row=final, order=order)


async def _run_steps(
    sf: SessionFactory,
    deps: RemovalDeps,
    row: SignupApproval,
    lease: UUID,
    actor: SignupActor,
    order: list[str],
    lost: asyncio.Event,
) -> SignupApproval:
    for step in REMOVAL_STEPS:
        removal = removal_of(row)
        steps = _rsteps(removal)
        if step != "blockers" and _done(steps.get(step)):
            continue
        # Between steps: the lease must still be ours (renewed, not lapsed).
        if lost.is_set() or not await _renew(sf, row.id, lease):
            raise _LeaseLostError(f"signup {row.id}: the removal lease was lost")
        order.append(step)
        if step == "finalize":
            purge = await _purge_exports(_Ctx(sf=sf, deps=deps, row=row, actor=actor))
            if purge.get("status") != "ok":
                steps["finalize"] = purge
                removal = {
                    **removal,
                    "steps": steps,
                    "last_error": {
                        "step": step,
                        "code": purge.get("code"),
                        "error": purge.get("error"),
                    },
                }
                return await _write(sf, row.id, lease, removal)
            detail = purge.get("detail")
            return await _finalize(
                sf, row, lease, removal, actor, purged=detail if isinstance(detail, dict) else {}
            )
        rec = await _run_one(_Ctx(sf=sf, deps=deps, row=row, actor=actor), step)
        steps[step] = rec
        removal = {**removal, "steps": steps}
        status = rec.get("status")
        if status == "blocked":
            detail = rec.get("detail")
            removal["blockers"] = detail.get("blockers", []) if isinstance(detail, dict) else []
            removal["state"] = "blocked"
            removal["last_error"] = {"step": step, "code": rec.get("code")}
            return await _write(sf, row.id, lease, removal, steps_next_attempt_at=None)
        if status in ("pending", "in_progress"):
            removal["last_error"] = {
                "step": step,
                "code": rec.get("code"),
                "error": rec.get("error"),
            }
            return await _write(sf, row.id, lease, removal)
        if step == "blockers":
            removal["blockers"] = []
        row = await _write(sf, row.id, lease, removal)
    return row


async def _schedule(
    sf: SessionFactory, signup_id: UUID, lease: UUID, row: SignupApproval
) -> SignupApproval:
    """A ``pending`` removal is due again after the backoff, or ``failed`` once
    the 10th attempt did not finish it. Other states wait for nobody."""
    removal = removal_of(row)
    if removal.get("state") != "pending":
        return await _write(sf, signup_id, lease, removal, steps_next_attempt_at=None)
    attempts = int(cast(int, removal.get("attempts") or 0))
    if attempts >= STEPS_MAX_ATTEMPTS:
        return await _write(
            sf, signup_id, lease, {**removal, "state": "failed"}, steps_next_attempt_at=None
        )
    delay = datetime.timedelta(seconds=steps_backoff_seconds(attempts))
    return await _write(sf, signup_id, lease, removal, steps_next_attempt_at=func.now() + delay)


def _notify_transition(before: str | None, row: SignupApproval) -> None:
    """Telegram on reaching done / blocked / failed (never raises)."""
    try:
        removal = removal_of(row)
        state = removal.get("state")
        if state not in ("done", "blocked", "failed") or (state == before and state != "blocked"):
            return
        ident = _rsteps(removal).get("identity") or {}
        notify_removal(
            RemovalNotice(
                id=str(row.id),
                state=str(state),
                # Never the company-derived slug, the name or the address: a
                # Telegram message cannot be erased later. The row id + the
                # tombstone + the Open-in-w2v link identify the workspace.
                tenant_slug=slug_tombstone(row.id),
                tenant_id=str(row.tenant_id) if row.tenant_id else None,
                blockers=cast(list[dict[str, object]], removal.get("blockers") or []),
                last_error=cast(dict[str, object] | None, removal.get("last_error")),
                attempts=int(cast(int, removal.get("attempts") or 0)),
                identity=cast(str | None, ident.get("code")) if isinstance(ident, dict) else None,
            )
        )
    except Exception:  # noqa: BLE001 — a notice never fails the removal
        _log.warning("signup removal id=%s: notice failed", row.id, exc_info=True)


# ─── retry / resolve / tick ──────────────────────────────────────────────


async def retry_removal(
    sf: SessionFactory,
    signup_id: UUID,
    actor: SignupActor,
    *,
    legacy_identity: LegacyIdentity | None = None,
    allow_credentialed_delete: bool | None = None,
) -> SignupApproval:
    """super_admin Retry of a pending / blocked / failed removal: ENQUEUED (state
    ``pending``, the count starts again, due at once — the tick runs it).
    ``legacy_identity`` / ``allow_credentialed_delete`` may be corrected while
    the identity step has not landed."""
    row = await _load(sf, signup_id)
    if row.status != "removed":
        raise SignupError(409, "removal_not_found", "This signup has no removal to retry.")
    state = removal_state(row)
    if state not in RETRY_STATES:
        raise SignupError(
            409, "removal_not_retryable", f"The removal is {state}; there is nothing to retry."
        )
    if legacy_identity is not None or allow_credentialed_delete is not None:
        row = await _amend_identity(sf, row, actor, legacy_identity, allow_credentialed_delete)
    async with sf() as s:
        got = (
            await s.execute(
                update(SignupApproval)
                .where(
                    SignupApproval.id == signup_id,
                    SignupApproval.status == "removed",
                    SignupApproval.removal["state"].astext.in_(RETRY_STATES),
                    _lease_free(),
                )
                .values(
                    removal=func.jsonb_set(
                        func.jsonb_set(
                            SignupApproval.removal,
                            text("'{attempts}'::text[]"),
                            text("'0'::jsonb"),
                        ),
                        text("'{state}'::text[]"),
                        text("'\"pending\"'::jsonb"),
                    ),
                    steps_next_attempt_at=func.now(),
                    updated_at=func.now(),
                )
                .returning(SignupApproval)
            )
        ).scalar_one_or_none()
        await s.commit()
    if got is None:
        raise SignupError(409, "removal_in_progress", "The removal is running right now; reload.")
    _log.info("signup removal retry (enqueued) id=%s by=%s", signup_id, actor.audit)
    return got


def _unchanged(row: SignupApproval) -> ColumnElement[bool]:
    """Optimistic concurrency: the row is still exactly the version ``row`` read
    (every removal write bumps ``updated_at``)."""
    return SignupApproval.updated_at == row.updated_at


async def _amend_identity(
    sf: SessionFactory,
    row: SignupApproval,
    actor: SignupActor,
    legacy_identity: LegacyIdentity | None,
    allow_credentialed_delete: bool | None,
) -> SignupApproval:
    removal = removal_of(row)
    if _done(_rsteps(removal).get("identity")):
        raise SignupError(
            409, "identity_step_done", "The login was already retired; it cannot be changed."
        )
    if legacy_identity is not None and has_identity_step(row):
        raise SignupError(
            422,
            "legacy_identity_not_applicable",
            "This signup minted its own login (identity step); no legacy login may be named.",
        )
    patch: dict[str, object] = {}
    if legacy_identity is not None:
        patch["legacy_identity"] = _legacy_record(legacy_identity)
    if allow_credentialed_delete is not None:
        patch["allow_credentialed_delete"] = allow_credentialed_delete
    async with sf() as s:
        got = (
            await s.execute(
                update(SignupApproval)
                .where(
                    SignupApproval.id == row.id,
                    SignupApproval.status == "removed",
                    _lease_free(),
                    _unchanged(row),
                )
                .values(removal={**removal, **patch}, updated_at=func.now())
                .returning(SignupApproval)
            )
        ).scalar_one_or_none()
        if got is None:
            await s.rollback()
            raise SignupError(
                409, "removal_in_progress", "The removal changed or is running right now; reload."
            )
        legacy = cast(dict[str, object] | None, patch.get("legacy_identity"))
        await AuditRepository(s).record(
            actor_subject_id=actor.subject_id,
            action="signup.remove",
            target_tenant_id=None,
            detail={
                "phase": "identity_amended",
                "signup_id": str(row.id),
                "legacy_user_id": legacy_identity.user_id if legacy_identity else None,
                **_audit_fingerprint(legacy),
                "allow_credentialed_delete": allow_credentialed_delete,
                **actor.audit_detail(),
            },
        )
        await s.commit()
    return got


async def resolve_removal(
    sf: SessionFactory,
    signup_id: UUID,
    actor: SignupActor,
    *,
    note: str,
    acknowledge_live_access: bool,
) -> SignupApproval:
    """A blocked / failed removal → ``resolved_manually`` (the operator finished
    it by hand, or decided not to). Needs an explicit acknowledgement that
    access may still be live; audited. Erases the personal data exactly like
    ``finalize`` and frees the address. Guarded by the version it read: a
    removal that changed meanwhile answers 409 ``removal_in_progress``."""
    if acknowledge_live_access is not True:
        raise SignupError(
            422,
            "acknowledgement_required",
            "Confirm that the workspace, its licences or the login may still be live.",
        )
    row = await _load(sf, signup_id)
    removal = removal_of(row)
    if row.status != "removed" or removal.get("state") not in RESOLVE_STATES:
        raise SignupError(
            409,
            "removal_not_resolvable",
            f"Only a blocked or failed removal can be resolved (it is {removal.get('state')}).",
        )
    fingerprint = note_fingerprint(note)
    resolved = {
        **removal,
        "state": "resolved_manually",
        "resolved": {"at": _iso(), "by": actor.audit, **fingerprint},
        "finished_at": _iso(),
    }
    async with sf() as s:
        got = (
            await s.execute(
                update(SignupApproval)
                .where(
                    SignupApproval.id == signup_id,
                    SignupApproval.status == "removed",
                    SignupApproval.removal["state"].astext.in_(RESOLVE_STATES),
                    _lease_free(),
                    _unchanged(row),
                )
                .values(
                    **_erasure(row, resolved),
                    steps_next_attempt_at=None,
                    updated_at=func.now(),
                )
                .returning(SignupApproval)
            )
        ).scalar_one_or_none()
        if got is None:
            await s.rollback()
            raise SignupError(409, "removal_in_progress", "The removal changed meanwhile; reload.")
        await AuditRepository(s).record(
            actor_subject_id=actor.subject_id,
            action="signup.remove",
            target_tenant_id=None,
            detail={
                "phase": "resolved_manually",
                "signup_id": str(signup_id),
                "state_before": removal.get("state"),
                "tenant_id": str(row.tenant_id) if row.tenant_id else None,
                **fingerprint,
                "erased": list(ERASED_FIELDS),
                "acknowledge_live_access": True,
                **actor.audit_detail(),
            },
        )
        await s.commit()
    return got


async def run_due_removals(
    sf: SessionFactory, deps: RemovalDeps, *, limit: int = TICK_BATCH
) -> int:
    """One pass over DUE pending removals. Never raises per row."""
    async with sf() as s:
        due = list(
            (
                await s.execute(
                    select(SignupApproval.id)
                    .where(
                        SignupApproval.status == "removed",
                        SignupApproval.removal["state"].astext == "pending",
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
            run = await run_removal(sf, deps, signup_id)
            ran += int(run.ran)
        except Exception:  # noqa: BLE001 — one row never stops the tick
            _log.warning("signup removal tick: row %s failed", signup_id, exc_info=True)
    return ran


#: A ``pending`` removal due for longer than this, and not running, is STALE:
#: the signup tick is not picking it up (disabled, crashing, or the pod is down).
STALE_AFTER = datetime.timedelta(minutes=15)


def _legacy_view(legacy: object) -> dict[str, object] | None:
    """The named login + the evidence fingerprint's PREFIX (never the text)."""
    if not isinstance(legacy, dict):
        return None
    digest = legacy.get("evidence_sha256")
    return {
        "user_id": legacy.get("user_id"),
        "evidence_sha256_prefix": digest[:EVIDENCE_PREFIX_CHARS]
        if isinstance(digest, str)
        else None,
        "evidence_length": legacy.get("evidence_length"),
        "evidence_created_at": legacy.get("evidence_created_at"),
    }


def is_stale(row: SignupApproval, *, now: datetime.datetime | None = None) -> bool:
    """Queued (``pending``, not running) and due for more than :data:`STALE_AFTER`."""
    if row.status != "removed" or removal_state(row) != "pending":
        return False
    now = now or _now()
    running = row.steps_lease_until is not None and row.steps_lease_until > now
    due = row.steps_next_attempt_at
    return not running and due is not None and due <= now - STALE_AFTER


def removal_view(row: SignupApproval) -> dict[str, object] | None:
    """The removal as the onboarding screen shows it."""
    if row.removal is None:
        return None
    removal = removal_of(row)
    steps = _rsteps(removal)
    return {
        "state": removal.get("state"),
        "attempts": removal.get("attempts", 0),
        "requested_at": removal.get("requested_at"),
        "requested_via": removal.get("requested_via"),
        "tenant_slug": removal.get("tenant_slug"),
        "steps": [{"step": k, **(steps.get(k) or {"status": "waiting"})} for k in REMOVAL_STEPS],
        "blockers": removal.get("blockers") or [],
        "last_error": removal.get("last_error"),
        "legacy_identity": _legacy_view(removal.get("legacy_identity")),
        "allow_credentialed_delete": bool(removal.get("allow_credentialed_delete")),
        "resolved": removal.get("resolved"),
        "finished_at": removal.get("finished_at"),
        "next_attempt_at": (
            row.steps_next_attempt_at.isoformat() if row.steps_next_attempt_at else None
        ),
        # A background pass holds the lease right now (the page shows "running").
        "running": bool(
            row.steps_lease_token is not None
            and row.steps_lease_until is not None
            and row.steps_lease_until > _now()
        ),
        # Queued > 15 min and nobody runs it: the signup tick is not working.
        "stale": is_stale(row),
    }


def needs_attention(row: SignupApproval) -> bool:
    return row.status == "removed" and removal_state(row) in RETRY_STATES


__all__ = [
    "ENGINE_TABLES",
    "REMOVAL_STEPS",
    "LegacyIdentity",
    "RemovalDeps",
    "RemovalRun",
    "needs_attention",
    "needs_legacy_identity",
    "removal_view",
    "resolve_removal",
    "retry_removal",
    "run_due_removals",
    "run_removal",
    "start_removal",
]
