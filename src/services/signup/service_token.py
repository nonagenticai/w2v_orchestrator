"""The two service-account doors of early-access signup.

  intake   POST /api/signup-approvals/intake                  early-access-intake's SA
                                                              (client + role ``w2v-signup-intake``)
  decider  POST /api/signup-approvals/{id}/telegram-decision  chat-bridge's SA (client
                                                              ``chat-bridge-signup-decider``, role
                                                              ``w2v-signup-decide``; contract C3)

A verified token is admitted only when ALL hold (same four conditions as CP / fr):

1. ``azp`` is one of the door's configured clients — a realm role can be mapped onto
   the wrong client by mistake; these doors write rows / found tenants;
2. ``aud`` contains that same ``azp`` (gitops gives each client an ``<id>-audience``
   mapper);
3. the door's realm role;
4. NOT ``super_admin`` — an operator acts on the super_admin routes, never through a
   service door.

⛔ VERIFIED HERE, NOT BY ``AuthMiddleware``: the w2v_common verifier pins
``aud`` to ``KEYCLOAK_AUDIENCES`` (``w2v-backend``) and never looks at ``azp``, so
these paths bypass it (``src/auth/signup_doors.py``) and verify the signature
themselves with w2v_common's :class:`KeycloakJWTValidator` (its cached JWKS,
RS256 only, ``exp``/``iss``/``sub`` REQUIRED — a signed token without ``exp``
never expires), the issuer pinned per door (``W2V_SIGNUP_INTAKE_ISSUER`` /
``W2V_SIGNUP_DECIDER_ISSUER``, default the shared portal pin), the JWKS taken
from that pinned issuer's realm, and its own audience check off (condition 2
replaces it).

401 ``unauthorized`` for a missing / malformed / unverifiable / expired token;
403 ``signup_intake_forbidden`` / ``signup_decider_forbidden`` for a verified
token that fails a condition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from w2v_common.auth.keycloak import KeycloakJWTValidator

from src.config import SignupSettings, get_settings

SUPER_ADMIN = "super_admin"


class SignupAuthError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class SignupServiceCaller:
    #: The service account's subject — audit metadata only.
    subject: str
    azp: str


class TokenValidator(Protocol):
    async def validate(self, token: str) -> tuple[dict[str, object] | None, str | None]: ...


_validators: dict[tuple[str, str, str], KeycloakJWTValidator] = {}
_override: TokenValidator | None = None


def set_signup_token_validator_for_tests(validator: TokenValidator | None) -> None:
    """⛔ TESTS ONLY: verify against a local key instead of the realm JWKS."""
    global _override
    _override = validator


def jwks_realm_for_issuer(issuer: str, default_realm: str) -> str:
    """The realm whose JWKS verifies ``issuer``'s tokens.

    Taken from the pinned issuer (``.../realms/<realm>``), so a door pinned to
    ``secure-apps`` fetches secure-apps keys even though ``KEYCLOAK_REALM`` (the
    provisioning admin realm) is ``portal``. Falls back to ``default_realm`` only
    for an issuer that names no realm. For today's portal pin this is ``portal``,
    i.e. byte-identical to before.
    """
    tail = issuer.strip().rstrip("/").rsplit("/realms/", 1)
    if len(tail) == 2 and tail[1] and "/" not in tail[1]:
        return tail[1]
    return default_realm


def _validator(issuer: str) -> TokenValidator | None:
    if _override is not None:
        return _override
    prov = get_settings().provisioning
    if not prov.keycloak_url or not issuer:
        return None
    realm = jwks_realm_for_issuer(issuer, prov.keycloak_realm)
    key = (prov.keycloak_url, realm, issuer)
    validator = _validators.get(key)
    if validator is None:
        # One per (JWKS url, realm, issuer): the JWKS cache lives on the instance.
        validator = KeycloakJWTValidator(
            keycloak_url=prov.keycloak_url,
            realm=realm,
            audiences=[],
            issuer=issuer,
        )
        _validators[key] = validator
    return validator


def bearer_token(authorization: str | None) -> str:
    parts = (authorization or "").strip().split()
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1]:
        raise SignupAuthError(401, "unauthorized", "A bearer token is required.")
    return parts[1]


def _realm_roles(claims: dict[str, object]) -> list[str]:
    access = claims.get("realm_access")
    roles = access.get("roles") if isinstance(access, dict) else None
    return [r for r in roles if isinstance(r, str)] if isinstance(roles, list) else []


def _audiences(claims: dict[str, object]) -> list[str]:
    aud = claims.get("aud")
    if isinstance(aud, str):
        return [aud]
    return [a for a in aud if isinstance(a, str)] if isinstance(aud, list) else []


async def _verify(authorization: str | None, issuer: str) -> dict[str, object]:
    token = bearer_token(authorization)
    issuer = issuer.strip().rstrip("/")
    validator = _validator(issuer)
    if validator is None:
        raise SignupAuthError(401, "unauthorized", "Token issuer is not configured.")
    claims, _reason = await validator.validate(token)
    if claims is None:
        raise SignupAuthError(401, "unauthorized", "Invalid or expired token.")
    # Belt and braces: the validator pins the issuer and requires exp/iss/sub.
    if str(claims.get("iss", "")).rstrip("/") != issuer or not claims.get("sub"):
        raise SignupAuthError(401, "unauthorized", "Invalid or expired token.")
    if not isinstance(claims.get("exp"), (int, float)):
        raise SignupAuthError(401, "unauthorized", "Invalid or expired token.")
    return claims


async def _require(
    authorization: str | None,
    *,
    issuer: str,
    allowed_azp: list[str],
    role: str,
    code: str,
    what: str,
) -> SignupServiceCaller:
    claims = await _verify(authorization, issuer)
    roles = _realm_roles(claims)
    if SUPER_ADMIN in roles:
        raise SignupAuthError(
            403,
            code,
            f"The {what} accepts only its dedicated service-account token, never an operator.",
        )
    azp = claims.get("azp") if isinstance(claims.get("azp"), str) else ""
    if not azp or azp not in allowed_azp:
        raise SignupAuthError(403, code, f"This client is not allowed to use the {what}.")
    if azp not in _audiences(claims):
        raise SignupAuthError(403, code, f"The token is not addressed to the {what}.")
    if not role or role not in roles:
        raise SignupAuthError(403, code, f"Role {role!r} is required for the {what}.")
    return SignupServiceCaller(subject=str(claims["sub"]), azp=str(azp))


async def require_intake_caller(
    authorization: str | None, cfg: SignupSettings
) -> SignupServiceCaller:
    return await _require(
        authorization,
        issuer=cfg.intake_token_issuer,
        allowed_azp=cfg.intake_azps,
        role=cfg.intake_role.strip(),
        code="signup_intake_forbidden",
        what="signup intake",
    )


async def require_decider_caller(
    authorization: str | None, cfg: SignupSettings
) -> SignupServiceCaller:
    return await _require(
        authorization,
        issuer=cfg.decider_token_issuer,
        allowed_azp=cfg.decider_azps,
        role=cfg.decider_role.strip(),
        code="signup_decider_forbidden",
        what="signup decision door",
    )
