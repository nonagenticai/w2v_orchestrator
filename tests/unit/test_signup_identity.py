"""The early-access identity (``src/services/signup/identity.py``): fr splitName,
the create body, ownership by stamp, the re-read, and the pre-email check."""

from __future__ import annotations

from typing import Any, cast

import pytest

from src.clients.keycloak_admin import KeycloakAdminClient, KeycloakAdminError
from src.clients.schemas.keycloak import KeycloakUser
from src.services.signup import identity as ident

pytestmark = pytest.mark.asyncio

SID = "0199a000-0000-7000-8000-000000000001"


@pytest.mark.parametrize(
    ("name", "expected"),
    [("Ann Early", {"firstName": "Ann", "lastName": "Early"})],
)
async def test_split_name_is_fr_split_name(name: str | None, expected: dict[str, str]) -> None:
    assert ident.split_name(name, " Ann.Early@Acme.Example ") == expected


async def test_every_create_body_carries_both_names() -> None:
    for name in ("Cher", "", None, "Ann Early"):
        rep = ident.user_representation(email="c@x.test", name=name, locale="en", signup_id=SID)
        assert rep["firstName"] and rep["lastName"], name


async def test_create_body_is_credential_less_unverified_and_stamped() -> None:
    rep = ident.user_representation(
        email=" Ann@Acme.Example ", name="Anna van der Berg", locale="nl", signup_id=SID
    )
    assert rep == {
        "username": "ann@acme.example",
        "email": "ann@acme.example",
        "emailVerified": False,
        "enabled": True,
        "firstName": "Anna",
        "lastName": "van der Berg",
        "attributes": {"locale": ["nl"], "signup_app": ["w2v"], "w2v_signup_id": [SID]},
    }
    single = ident.user_representation(email="c@x.test", name="Cher", locale="en", signup_id=SID)
    assert (single["firstName"], single["lastName"]) == ("Cher", "Cher")
    assert "credentials" not in rep and "requiredActions" not in rep


async def test_belongs_to_needs_exactly_this_rows_id() -> None:
    def user(attrs: dict[str, list[str]] | None) -> KeycloakUser:
        return KeycloakUser(id="u", attributes=attrs)

    assert ident.belongs_to(user({"w2v_signup_id": [SID]}), SID)
    assert not ident.belongs_to(user({"w2v_signup_id": [SID, "other"]}), SID)
    assert not ident.belongs_to(user({"w2v_signup_id": ["other"]}), SID)
    assert not ident.belongs_to(user({"fr_signup_id": [SID]}), SID)
    assert not ident.belongs_to(user(None), SID)
    assert not ident.belongs_to(None, SID)


class Kc:
    def __init__(self, existing: dict[str, Any] | None = None, *, keep: bool = True) -> None:
        self.existing = existing
        self.keep = keep
        self.users: dict[str, dict[str, Any]] = {}
        self.deleted: list[str] = []
        self.writes: list[dict[str, str | None]] = []
        self.write_error = False
        self.credentials: dict[str, int] = {}
        self.links: dict[str, int] = {}

    async def create_user(self, rep: dict[str, Any]) -> str | None:
        if self.existing is not None:
            return None
        attrs = dict(rep["attributes"])
        if not self.keep:
            attrs.pop("w2v_signup_id")
        self.users["new"] = {"id": "new", "attributes": attrs}
        return "new"

    async def find_user_by_email(self, _email: str) -> KeycloakUser | None:
        if self.existing is None:
            return None
        self.users[self.existing["id"]] = self.existing
        return KeycloakUser.model_validate(self.existing)

    async def get_user(self, uid: str) -> KeycloakUser | None:
        u = self.users.get(uid)
        return KeycloakUser.model_validate(u) if u else None

    async def count_credentials(self, uid: str) -> int:
        return self.credentials.get(uid, 0)

    async def count_federated_identities(self, uid: str) -> int:
        return self.links.get(uid, 0)

    async def delete_user(self, uid: str) -> bool:
        self.deleted.append(uid)
        return self.users.pop(uid, None) is not None

    async def update_user_attributes(self, _uid: str, attrs: dict[str, str | None]) -> None:
        if self.write_error:
            raise KeycloakAdminError("down", status_code=503)
        self.writes.append(attrs)


def _kc(kc: Kc) -> KeycloakAdminClient:
    return cast("KeycloakAdminClient", kc)


async def _ensure(kc: Kc) -> ident.Identity:
    return await ident.ensure_identity(
        _kc(kc), signup_id=SID, email="a@b.test", name="Ann Early", locale=None
    )


async def test_mint_then_reread() -> None:
    kc = Kc()
    assert await _ensure(kc) == ident.Identity(user_id="new", minted=True)
    assert kc.users["new"]["attributes"]["locale"] == ["en"], "no row locale → en"


async def test_adopt_only_our_own_stamp() -> None:
    kc = Kc({"id": "old", "attributes": {"w2v_signup_id": [SID]}})
    assert await _ensure(kc) == ident.Identity(user_id="old", minted=False)


@pytest.mark.parametrize(
    "touched",
    ["credential"],
)
async def test_a_stamped_user_its_owner_already_used_is_never_adopted(touched: str) -> None:
    """fr: stamped with this row's id is NOT enough — a user that has a
    credential, an IdP link, a federation link or a verified email is someone's
    live login now, and is refused (identity_exists), never touched."""
    existing: dict[str, Any] = {"id": "old", "attributes": {"w2v_signup_id": [SID]}}
    if touched == "federation":
        existing["federationLink"] = "ldap-1"
    if touched == "verified":
        existing["emailVerified"] = True
    kc = Kc(existing)
    if touched == "credential":
        kc.credentials["old"] = 1
    if touched == "idp_link":
        kc.links["old"] = 1
    with pytest.raises(ident.IdentityError) as info:
        await _ensure(kc)
    assert info.value.code == ident.IDENTITY_EXISTS
    assert kc.deleted == [] and kc.writes == []


@pytest.mark.parametrize("attrs", [{}])
async def test_a_foreign_existing_user_is_refused_never_touched(attrs: Any) -> None:
    kc = Kc({"id": "old", "attributes": attrs})
    with pytest.raises(ident.IdentityError) as info:
        await _ensure(kc)
    assert info.value.code == ident.IDENTITY_EXISTS
    assert kc.deleted == [] and kc.writes == []


async def test_a_dropped_stamp_deletes_only_what_this_call_minted() -> None:
    kc = Kc(keep=False)
    with pytest.raises(ident.IdentityError) as info:
        await _ensure(kc)
    assert info.value.code == ident.IDENTITY_ATTR_DROPPED
    assert kc.deleted == ["new"] and info.value.detail["deleted"] is True


async def test_prepare_email_refuses_a_user_without_our_stamp() -> None:
    kc = Kc()
    kc.users["u"] = {"id": "u", "attributes": {"w2v_signup_id": ["x"]}}
    with pytest.raises(ident.IdentityError) as info:
        await ident.prepare_email(_kc(kc), user_id="u", signup_id=SID, locale="de")
    assert info.value.code == ident.IDENTITY_NOT_OWNED
    with pytest.raises(ident.IdentityError) as gone:
        await ident.prepare_email(_kc(kc), user_id="gone", signup_id=SID, locale="de")
    assert gone.value.code == ident.IDENTITY_GONE, "a deleted user is gone, not someone else's"
    assert kc.writes == []


async def test_prepare_email_sets_only_missing_attributes_and_never_fails_on_them() -> None:
    kc = Kc()
    kc.users["u"] = {"id": "u", "attributes": {"w2v_signup_id": [SID], "locale": ["en"]}}
    assert await ident.prepare_email(_kc(kc), user_id="u", signup_id=SID, locale="de")
    assert kc.writes == [{"locale": "de", "signup_app": "w2v"}]
    kc.users["u"]["attributes"].update({"locale": ["de"], "signup_app": ["w2v"]})
    assert await ident.prepare_email(_kc(kc), user_id="u", signup_id=SID, locale="de")
    assert len(kc.writes) == 1, "nothing to write"
    kc.users["u"]["attributes"].pop("signup_app")
    kc.write_error = True
    assert await ident.prepare_email(_kc(kc), user_id="u", signup_id=SID, locale="de") is False
