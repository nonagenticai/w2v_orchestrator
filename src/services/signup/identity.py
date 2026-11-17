"""The Keycloak half of an early-access approval (fr ``services/signup/identity.ts``,
CommonPractice ``services/provisioning.ts``).

George 2026-10-20: w2v early access works EXACTLY like CommonPractice and fr.
w2v MINTS the requester's realm user itself, credential-less, with
``firstName`` / ``lastName`` from the request's name (fr ``splitName``): a user
created without them (provisioning's identity invite) makes Keycloak show an
"Update account information" form right after the password.

OWNERSHIP IS AN ATTRIBUTE, NOT A HEURISTIC. The user is stamped
``w2v_signup_id = [<signup row id>]``. The approve adopts an existing realm user
ONLY when it carries exactly this row's id AND is still untouched (no credential,
no identity-provider link, no federation, email unverified — fr; a crash
between create and record);
any other user with that address — someone's portal login, another app's
early-access identity (the realm is shared) — is REFUSED (``identity_exists``),
never adopted, and the super_admin sees it on /admin/onboarding. The attribute
must be declared in the realm's user profile (admin view/edit) or Keycloak drops
it SILENTLY — so the step re-reads the user right after create/adopt and refuses
(``identity_attr_dropped``) before any tenant or licence exists, deleting a user
it just minted so the address is free again.

The email (``prepare_email`` + the admin client's ``execute_actions_email``) is
sent ONLY to a user still carrying this row's stamp (``identity_not_owned``
otherwise), after making sure ``locale`` + ``signup_app`` are set (they select
the w2v-branded template and its language; set at create, re-asserted here,
non-fatal like fr).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.clients.keycloak_admin import KeycloakAdminClient, KeycloakAdminError
from src.clients.schemas.keycloak import KeycloakUser
from src.services.signup.telegram import SIGNUP_APP_KEY

_log = logging.getLogger("w2v_orchestrator.signup.identity")

#: The realm-user attribute that marks an identity as minted by THIS signup row.
SIGNUP_ID_ATTRIBUTE = "w2v_signup_id"
#: Selects the app-branded Keycloak email template (declared in the realm profile).
SIGNUP_APP_ATTRIBUTE = "signup_app"
#: The one email's required actions (fr / CP).
EMAIL_ACTIONS: tuple[str, ...] = ("UPDATE_PASSWORD", "VERIFY_EMAIL")
#: How long the email's link works (fr ``emailLifespanSeconds``).
EMAIL_LIFESPAN_SECONDS = 14 * 24 * 3600
DEFAULT_LOCALE = "en"

# Step codes (recorded on ``steps.identity`` / ``steps.invite_email``).
IDENTITY_EXISTS = "identity_exists"
IDENTITY_FAILED = "identity_failed"
IDENTITY_ATTR_DROPPED = "identity_attr_dropped"
IDENTITY_NOT_OWNED = "identity_not_owned"
#: The recorded realm user no longer exists (deleted out of band): the identity
#: step is forgotten and runs again on the next attempt (fr: "forget it and mint
#: again").
IDENTITY_GONE = "identity_gone"


class IdentityError(Exception):
    """A refusal of the identity half: a stable step ``code`` + detail."""

    def __init__(self, code: str, message: str, detail: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.detail = detail or {}


def split_name(name: str | None, email: str) -> dict[str, str]:
    """fr ``splitName``: trim, split on whitespace, first word → ``firstName``,
    the rest (joined by single spaces) → ``lastName``. A one-word name repeats
    as ``lastName`` and an empty one uses the email's local part for both:
    Keycloak's VERIFY_PROFILE wants BOTH, or it shows the "Update account
    information" form after the password."""
    parts = (name or "").split()
    if not parts:
        address = email.strip().lower()
        local = address.split("@", 1)[0] or address
        return {"firstName": local, "lastName": local}
    if len(parts) == 1:
        return {"firstName": parts[0], "lastName": parts[0]}
    return {"firstName": parts[0], "lastName": " ".join(parts[1:])}


def belongs_to(user: KeycloakUser | None, signup_id: str) -> bool:
    """True only when ``user`` carries exactly ``w2v_signup_id = [signup_id]``."""
    values = ((user.attributes if user else None) or {}).get(SIGNUP_ID_ATTRIBUTE)
    return isinstance(values, list) and values == [signup_id]


def user_representation(
    *, email: str, name: str | None, locale: str, signup_id: str
) -> dict[str, object]:
    """The create body: enabled, UNVERIFIED (the email's VERIFY_EMAIL completes
    it), no credential (only the inbox owner can set one, via the link)."""
    address = email.strip().lower()
    return {
        "username": address,
        "email": address,
        "emailVerified": False,
        "enabled": True,
        **split_name(name, address),
        "attributes": {
            "locale": [locale],
            SIGNUP_APP_ATTRIBUTE: [SIGNUP_APP_KEY],
            SIGNUP_ID_ATTRIBUTE: [signup_id],
        },
    }


@dataclass(frozen=True, slots=True)
class Identity:
    user_id: str
    #: True when THIS call created the user; False when it adopted the one an
    #: earlier attempt of the same row created.
    minted: bool


async def _adoptable(kc: KeycloakAdminClient, user: KeycloakUser, signup_id: str) -> bool:
    """fr: adopt ONLY what an earlier attempt of THIS row minted (a crash between
    create and record) — stamped with this row's id AND still untouched by its
    owner: no credential, no identity-provider link, no user-storage federation,
    email not verified. Freshness alone proves nothing (the realm is shared)."""
    return (
        belongs_to(user, signup_id)
        and not user.federation_link
        and not user.email_verified
        and await kc.count_credentials(user.id) == 0
        and await kc.count_federated_identities(user.id) == 0
    )


async def ensure_identity(
    kc: KeycloakAdminClient, *, signup_id: str, email: str, name: str | None, locale: str | None
) -> Identity:
    """Mint the row's realm user, or adopt the one an earlier attempt of THIS row
    minted. Raises :class:`IdentityError` (``identity_exists`` /
    ``identity_failed`` / ``identity_attr_dropped``) or
    :class:`KeycloakAdminError` (Keycloak failed)."""
    rep = user_representation(
        email=email, name=name, locale=locale or DEFAULT_LOCALE, signup_id=signup_id
    )
    created = await kc.create_user(rep)
    if created is not None:
        user_id, minted = created, True
    else:
        found = await kc.find_user_by_email(email)
        if found is None:
            raise IdentityError(
                IDENTITY_FAILED, "Keycloak reported the address taken but no user carries it."
            )
        if not await _adoptable(kc, found, signup_id):
            raise IdentityError(
                IDENTITY_EXISTS,
                f"{email} already has a login that this approval did not create; it is never "
                "adopted. Onboard this person another way (e.g. invite them into a workspace).",
                {"user_id": found.id},
            )
        user_id, minted = found.id, False
    # RE-READ before anything irreversible: a realm whose user profile does not
    # declare w2v_signup_id drops it SILENTLY, and the email's ownership check
    # would then fail after the tenant was founded. Refuse here instead.
    stored = await kc.get_user(user_id)
    if not belongs_to(stored, signup_id):
        deleted = False
        if minted:
            try:
                deleted = await kc.delete_user(user_id)
            except KeycloakAdminError:
                _log.warning("signup %s: could not delete unstamped login %s", signup_id, user_id)
        raise IdentityError(
            IDENTITY_ATTR_DROPPED,
            f"Keycloak did not keep {SIGNUP_ID_ATTRIBUTE}={signup_id} on login {user_id}: "
            f"declare {SIGNUP_ID_ATTRIBUTE} in the realm user profile (admin view/edit). "
            + ("The login was deleted." if deleted else f"Login {user_id} was NOT deleted."),
            {"user_id": user_id, "deleted": deleted},
        )
    return Identity(user_id=user_id, minted=minted)


async def prepare_email(
    kc: KeycloakAdminClient, *, user_id: str, signup_id: str, locale: str | None
) -> bool:
    """Before the one email: refuse a user that does not carry this row's stamp
    (``identity_not_owned``; a user that is gone: ``identity_gone``), then make sure ``locale`` + ``signup_app`` are set.
    Returns whether the attributes are in place (a failed write is NOT fatal:
    Keycloak then sends its generic email in the realm language — fr)."""
    user = await kc.get_user(user_id)
    if user is None:
        raise IdentityError(
            IDENTITY_GONE,
            f"login {user_id} no longer exists in Keycloak (deleted after the identity step); "
            "the identity step runs again on the next attempt.",
            {"user_id": user_id},
        )
    if not belongs_to(user, signup_id):
        raise IdentityError(
            IDENTITY_NOT_OWNED,
            f"login {user_id} does not carry {SIGNUP_ID_ATTRIBUTE}={signup_id}; "
            "this approval never writes to or mails it.",
            {"user_id": user_id},
        )
    want = {"locale": locale or DEFAULT_LOCALE, SIGNUP_APP_ATTRIBUTE: SIGNUP_APP_KEY}
    have = user.attributes or {}
    missing: dict[str, str | None] = {k: v for k, v in want.items() if have.get(k) != [v]}
    if not missing:
        return True
    try:
        await kc.update_user_attributes(user_id, missing)
    except KeycloakAdminError as exc:
        _log.warning("signup %s: email attributes not set on %s: %s", signup_id, user_id, exc)
        return False
    return True
