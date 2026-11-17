"""Tenant licence verification by INTROSPECTION (the commonpractice model).

An off-cluster deployment reaches licences only through its ingress, which
refuses the licences internal routes (400 ``missing_app_key``). It admits exactly two licences
routes for this service: ``POST /license/introspect`` (public — the token in the
body is the credential) and ``GET /licenses/active`` (the deployment's
service-account). So:

1. **Stored token.** A tenant's own w2v licence token is stored SEALED on its
   ``tenants`` row (``licence_token_ciphertext``; provider :class:`Keyring`, AAD
   ``("tenant-licence", tenant_id)``). Entitled iff its introspection is
   ``active`` ∧ ``tenant_id == tenant`` ∧ ``app.key == "w2v"``; its
   ``license_id`` is the tenant's ``licence_ref``. Never returned, never logged.
2. **Fallback, no token stored** (legacy tenants; control_service-created ones
   until it sends a token): ``GET /licenses/active?tenant_id=T&app_key=w2v`` —
   entitled yes/no; a ``licence_ref`` only when ``candidate_count == 1`` (with
   more, the returned id is the service's deterministic PICK, not provably "the"
   licence).

Distinct outcomes, because they need different operators: ``unreadable`` (the
stored blob will not open — a rotated/missing key or a corrupt row; standing, it
never recovers by retrying), ``unavailable`` (5xx / transport — retryable),
``refused`` (a definite 4xx from licences or its gateway — a configuration fault
of THIS deployment), ``inactive`` (``active:false``; licences' ``reason`` passed
through) and ``mismatch`` (an active token of another tenant or another app).
Could-not-check is never a verdict.
"""

from __future__ import annotations

import datetime
import http
import logging
import re
from dataclasses import dataclass
from typing import Literal, Protocol
from uuid import UUID

from src.clients.licenses_client import LicensesClientError, TenantNotOnboardedError
from src.clients.schemas.licenses import ActiveLicenseView
from src.services.providers.crypto import (
    Keyring,
    KeyringNotConfiguredError,
    SecretDecryptError,
    tenant_licence_aad,
)

_log = logging.getLogger("w2v_orchestrator.tenant_licence")

W2V_APP_KEY = "w2v"

#: :attr:`Entitlement.state` when licences says the tenant is not onboarded to w2v.
NOT_ONBOARDED_STATE = "not_onboarded"


class LicenceIntrospection(Protocol):
    """The two licences reads this module makes (both gateway-admitted)."""

    async def introspect(self, token: str) -> ActiveLicenseView: ...

    async def get_active_license(
        self, *, tenant_id: UUID | str, app_key: str
    ) -> ActiveLicenseView | None: ...


# ─── could-not-check ─────────────────────────────────────────────────────


class LicencesUnavailableError(RuntimeError):
    """licences could not answer (5xx, transport, a shape we cannot trust)."""


class LicencesRefusedError(LicencesUnavailableError):
    """licences — or the ingress gateway in front of it — DEFINITELY refused the
    call: a 4xx other than the retryable 408/425/429. A configuration fault of
    THIS deployment (missing realm role, a route the ingress does not admit for
    this token, a bad request), not a transient outage: retrying will not help.

    A subclass of :class:`LicencesUnavailableError` on purpose: every
    ``except LicencesUnavailableError`` keeps its fail-closed "could not check"
    behaviour; a caller that wants to say *refused* instead of *retry* catches
    this first. ``excerpt`` is a bounded, token-redacted slice of the upstream
    error (safe to log; never put in a response)."""

    def __init__(self, status_code: int, excerpt: str) -> None:
        super().__init__(f"licences refused the call: HTTP {status_code}")
        self.status_code = status_code
        self.excerpt = excerpt


class LicencesTenantNotOnboardedError(LicencesRefusedError):
    """licences answered 403 ``tenant_not_accessible``: the tenant is owned by
    another app that granted w2v no access — NOT ONBOARDED to w2v. A definite,
    non-retryable answer (hence a :class:`LicencesRefusedError`, so every existing
    ``except`` keeps refusing without a retry), but about the TENANT, not a
    configuration fault of this deployment: callers that map refusals to a 502
    catch this first and answer 403 ``TENANT_NOT_ONBOARDED``."""

    def __init__(self) -> None:
        super().__init__(403, "tenant_not_accessible")


class LicenceTokenUnreadableError(RuntimeError):
    """The tenant's stored licence token cannot be opened (unknown / missing key,
    wrong AAD, tampered or malformed blob). STANDING — needs an operator."""


#: 4xx statuses that mean "try again later", not "no".
_RETRYABLE_4XX = frozenset({408, 425, 429})
_EXCERPT_MAX = 300
_JWT_RE = re.compile(r"eyJ[\w-]+\.[\w-]+\.[\w-]*")
_BEARER_RE = re.compile(r"(?i)\bbearer\s+\S+")
#: Licence tokens are OPAQUE urlsafe strings (no ``eyJ`` / ``Bearer`` marker) —
#: ``secrets.token_urlsafe(32)``, 43 characters — so any urlsafe- or base64-ish
#: run of 40+ characters is treated as a secret. 40 keeps what an operator needs
#: to diagnose: error codes (``subject_tenant_mismatch_for_licence`` is 35).
_OPAQUE_MIN = 40
_OPAQUE_RE = re.compile(rf"[A-Za-z0-9_\-+/]{{{_OPAQUE_MIN},}}={{0,2}}")
#: Canonical UUIDs are ids, never tokens: kept even inside a long run such as
#: ``/licenses/<uuid>`` (46 characters). Only the run's NON-UUID pieces of 40+
#: characters are redacted, so a token glued to a UUID is still caught.
_UUID_SPLIT_RE = re.compile(r"([0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12})")


def _redact_opaque(match: re.Match[str]) -> str:
    # The UUIDs come back as their own (36-char) pieces, so they are never redacted.
    pieces = _UUID_SPLIT_RE.split(match.group(0))
    return "".join("<redacted>" if len(p) >= _OPAQUE_MIN else p for p in pieces)


def _redacted_excerpt(text: str) -> str:
    text = _BEARER_RE.sub("Bearer <redacted>", _JWT_RE.sub("<jwt>", text))
    return _OPAQUE_RE.sub(_redact_opaque, text)[:_EXCERPT_MAX]


def _status_only(status_code: int) -> str:
    try:
        return f"HTTP {status_code} {http.HTTPStatus(status_code).phrase}"
    except ValueError:
        return f"HTTP {status_code}"


def licences_error(exc: Exception, *, token_in_request: bool = False) -> LicencesUnavailableError:
    """Classify a failed licences call. A definite 4xx refusal (from licences or
    its gateway) → :class:`LicencesRefusedError`, logged with the status and a
    bounded, redacted excerpt; everything else (5xx, transport, bad shape) →
    :class:`LicencesUnavailableError`. A 403 ``tenant_not_accessible`` →
    :class:`LicencesTenantNotOnboardedError` (a refusal about the tenant).

    ``token_in_request`` (the introspect call — the token is the request body):
    the excerpt is the status + reason phrase ONLY, never any of the body."""
    if isinstance(exc, TenantNotOnboardedError):
        _log.info("licences: tenant not onboarded to w2v (403 tenant_not_accessible)")
        return LicencesTenantNotOnboardedError()
    status_code = exc.status_code if isinstance(exc, LicensesClientError) else None
    if status_code is not None and 400 <= status_code < 500 and status_code not in _RETRYABLE_4XX:
        excerpt = _status_only(status_code) if token_in_request else _redacted_excerpt(str(exc))
        refused = LicencesRefusedError(status_code, excerpt)
        _log.warning(
            "licences REFUSED the call (HTTP %d) — a configuration fault, not an outage: %s",
            status_code,
            refused.excerpt,
        )
        return refused
    return LicencesUnavailableError(type(exc).__name__)


# ─── sealing ─────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SealedLicenceToken:
    ciphertext: bytes
    key_id: str

    def __repr__(self) -> str:  # never render the envelope
        return f"SealedLicenceToken(key_id={self.key_id!r})"


def seal_licence_token(keyring: Keyring, tenant_id: UUID, token: str) -> SealedLicenceToken:
    """Seal ``token`` for ``tenant_id``. Raises :class:`KeyringNotConfiguredError`
    (→ 503) when the deployment has no keyring."""
    ciphertext, kid = keyring.encrypt_with_aad(token, tenant_licence_aad(tenant_id))
    return SealedLicenceToken(ciphertext=ciphertext, key_id=kid)


def open_licence_token(keyring: Keyring, tenant_id: UUID, ciphertext: bytes) -> str:
    """The stored token, or :class:`LicenceTokenUnreadableError` — an empty keyring
    with a stored blob is unreadable too (it needs the same operator)."""
    try:
        return keyring.decrypt_with_aad(ciphertext, tenant_licence_aad(tenant_id))
    except (SecretDecryptError, KeyringNotConfiguredError) as exc:
        raise LicenceTokenUnreadableError(str(exc)) from None


# ─── verdicts ────────────────────────────────────────────────────────────

TokenOutcome = Literal["entitled", "inactive", "mismatch"]


@dataclass(frozen=True, slots=True)
class TokenVerdict:
    """The introspection of ONE token, judged for ONE tenant."""

    outcome: TokenOutcome
    #: The token's licence (set iff ``entitled``).
    licence_id: UUID | None = None
    #: ``inactive``: licences' ``reason`` (or ``expired``); ``mismatch``: ``tenant`` | ``app``.
    reason: str | None = None
    #: The TOKEN's own expiry (introspection ``token_expires_at``; None = perpetual).
    token_expires_at: datetime.datetime | None = None

    @property
    def entitled(self) -> bool:
        return self.outcome == "entitled"


def _uuid(raw: object) -> UUID | None:
    try:
        return UUID(str(raw)) if raw else None
    except ValueError:
        return None


def _instant(raw: str | None) -> datetime.datetime | None:
    if not raw:
        return None
    try:
        when = datetime.datetime.fromisoformat(raw)
    except ValueError:
        return None
    return when if when.tzinfo is not None else when.replace(tzinfo=datetime.UTC)


def _expired(expires_at: str | None) -> bool:
    when = _instant(expires_at)
    return when is not None and when <= datetime.datetime.now(datetime.UTC)


def judge_introspection(view: ActiveLicenseView, tenant_id: UUID) -> TokenVerdict:
    """Entitled iff active ∧ unexpired ∧ ``tenant_id`` ∧ ``app.key == w2v``.
    Raises :class:`LicencesUnavailableError` on an ACTIVE answer with no usable
    licence id (a contract break is could-not-check, never a yes)."""
    if not view.active:
        return TokenVerdict("inactive", reason=view.reason or "inactive")
    if _expired(view.expires_at):
        return TokenVerdict("inactive", reason="expired")
    if _uuid(view.tenant_id) != tenant_id:
        return TokenVerdict("mismatch", reason="tenant")
    if view.app is None or view.app.key != W2V_APP_KEY:
        return TokenVerdict("mismatch", reason="app")
    licence_id = _uuid(view.license_id)
    if licence_id is None:
        raise LicencesUnavailableError("introspection answered active without a licence id")
    return TokenVerdict(
        "entitled", licence_id=licence_id, token_expires_at=_instant(view.token_expires_at)
    )


async def introspect_for_tenant(
    client: LicenceIntrospection, token: str, tenant_id: UUID
) -> TokenVerdict:
    """Introspect ``token`` and judge it for ``tenant_id``. Raises
    :class:`LicencesUnavailableError` / :class:`LicencesRefusedError` when
    licences cannot answer."""
    try:
        view = await client.introspect(token)
    except Exception as exc:  # noqa: BLE001 — classified + re-raised; cause dropped below
        # ``from None``: the cause's message may carry the upstream body (the token).
        raise licences_error(exc, token_in_request=True) from None
    return judge_introspection(view, tenant_id)


@dataclass(frozen=True, slots=True)
class FallbackVerdict:
    """``GET /licenses/active`` for (tenant, w2v)."""

    entitled: bool
    #: Set only when the tenant holds EXACTLY one active w2v licence.
    licence_ref: UUID | None = None
    candidate_count: int | None = None
    #: licences answered 403 ``tenant_not_accessible``: not onboarded to w2v.
    not_onboarded: bool = False


async def tenant_active_licence(client: LicenceIntrospection, tenant_id: UUID) -> FallbackVerdict:
    """The no-stored-token fallback. Raises :class:`LicencesUnavailableError` /
    :class:`LicencesRefusedError` when licences cannot answer, or answers about
    another tenant / app (a contract break, never a verdict)."""
    try:
        view = await client.get_active_license(tenant_id=tenant_id, app_key=W2V_APP_KEY)
    except TenantNotOnboardedError:
        # 403 ``tenant_not_accessible``: another app owns the tenant and has not
        # granted w2v — it is not onboarded to w2v, so w2v sees no licence there.
        return FallbackVerdict(entitled=False, not_onboarded=True)
    except Exception as exc:
        raise licences_error(exc) from exc
    if view is None:
        return FallbackVerdict(entitled=False, candidate_count=0)
    if view.conflict:  # legacy 409 multi-active: entitled, no single licence
        return FallbackVerdict(entitled=True)
    if not view.active:
        return FallbackVerdict(entitled=False)
    if (view.tenant_id is not None and _uuid(view.tenant_id) != tenant_id) or (
        view.app is not None and view.app.key != W2V_APP_KEY
    ):
        _log.warning("licences /licenses/active answered for another tenant/app — untrusted")
        raise LicencesUnavailableError("/licenses/active answered for another tenant or app")
    single = view.candidate_count == 1
    return FallbackVerdict(
        entitled=True,
        licence_ref=_uuid(view.license_id) if single else None,
        candidate_count=view.candidate_count,
    )


@dataclass(frozen=True, slots=True)
class Entitlement:
    """A tenant's entitlement, from whichever source applies."""

    source: Literal["token", "fallback"]
    entitled: bool
    #: The licence to record as ``licence_ref`` (None = leave the ref as it is).
    licence_ref: UUID | None
    #: ``entitled`` | ``inactive`` | ``mismatch`` | ``none`` (fallback, no licence)
    #: | ``not_onboarded`` (fallback: licences 403 ``tenant_not_accessible``).
    state: str
    reason: str | None = None


#: Tenants already warned (this process) that their stored token expired.
_expired_warned: set[UUID] = set()


def stored_token_expired(
    tenant_id: UUID, expires_at: datetime.datetime | None, *, site: str
) -> bool:
    """True when the stored token's own expiry has passed — it then counts as NO
    token (the caller falls back to ``/licenses/active``). WARNING once per tenant
    per process; every occurrence is counted."""
    if expires_at is None or expires_at > datetime.datetime.now(datetime.UTC):
        return False
    from src.observability.metrics import tenant_licence_token_expired_total

    tenant_licence_token_expired_total.labels(site=site).inc()
    if tenant_id not in _expired_warned:
        _expired_warned.add(tenant_id)
        _log.warning(
            "tenant %s: stored licence token expired at %s — falling back to "
            "/licenses/active; re-mint it and PUT /api/tenants/{id}/licence-token",
            tenant_id,
            expires_at.isoformat(),
        )
    return True


async def tenant_entitlement(
    client: LicenceIntrospection,
    keyring: Keyring,
    tenant_id: UUID,
    ciphertext: bytes | None,
    *,
    token_expires_at: datetime.datetime | None = None,
    site: str = "sweep",
) -> Entitlement:
    """Stored (unexpired) token → introspection; none stored, or expired → the
    fallback. Raises :class:`LicenceTokenUnreadableError` /
    :class:`LicencesUnavailableError` (incl. :class:`LicencesRefusedError`) when
    no verdict can be reached."""
    if ciphertext is not None and not stored_token_expired(tenant_id, token_expires_at, site=site):
        token = open_licence_token(keyring, tenant_id, ciphertext)
        verdict = await introspect_for_tenant(client, token, tenant_id)
        return Entitlement(
            source="token",
            entitled=verdict.entitled,
            licence_ref=verdict.licence_id,
            state=verdict.outcome,
            reason=verdict.reason,
        )
    fallback = await tenant_active_licence(client, tenant_id)
    return Entitlement(
        source="fallback",
        entitled=fallback.entitled,
        licence_ref=fallback.licence_ref,
        state=(
            "entitled"
            if fallback.entitled
            else NOT_ONBOARDED_STATE
            if fallback.not_onboarded
            else "none"
        ),
    )
