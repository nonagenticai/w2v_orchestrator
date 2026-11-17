"""Login secrets in briefs: sealed at rest, redacted on read, unsealed only to dispatch."""

from __future__ import annotations

import copy
import json

import pytest
from cryptography.fernet import Fernet
from pydantic import SecretStr

from src.config import get_settings
from src.services.auth_secrets import (
    REDACTED,
    AuthSecretError,
    auth_for_origin,
    canonical_origin,
    has_plaintext_secret,
    login_needs_reentry,
    redact_auth_secrets,
    restore_redacted,
    seal_auth_secrets,
    strict_login_url,
    unseal_auth,
    validate_auth_block,
    validate_brief_logins,
)

# Fake credentials (not real secrets).
_KC = "kc-pw"  # pragma: allowlist secret
_TW = "tw-pw"  # pragma: allowlist secret
_SRC = "src-pw"  # pragma: allowlist secret


@pytest.fixture
def key(monkeypatch: pytest.MonkeyPatch) -> str:
    k = Fernet.generate_key().decode()
    monkeypatch.setattr(get_settings().auth_secrets, "auth_config_secret_keys", SecretStr(k))
    return k


def _logins(block: object) -> list[dict]:
    assert isinstance(block, list)
    return block


def _one(entry: dict | None) -> dict:
    assert entry is not None
    return entry


def _brief() -> dict:
    return {
        "target_url": "https://marketing.example/strategy",
        "auth_config": [
            {
                "login_url": "https://marketing.example/strategy",
                "username": "kc",
                "password": _KC,
            },
            {
                "origin": "https://twenty.example",
                "username": "tw",
                "password": _TW,
            },
        ],
        "sources": [
            {
                "kind": "url",
                "ref": "https://a.example",
                "auth_info": {"password": _SRC},
            }
        ],
    }


def test_seal_leaves_no_plaintext_anywhere_and_unseal_restores(key: str) -> None:
    sealed = seal_auth_secrets(_brief())
    text = str(sealed)
    for secret in (_KC, _TW, _SRC):
        assert secret not in text
    assert not has_plaintext_secret(sealed)
    assert unseal_auth(sealed["auth_config"]) == _brief()["auth_config"]
    assert unseal_auth(sealed["sources"][0]["auth_info"]) == {"password": _SRC}


def test_seal_is_idempotent(key: str) -> None:
    once = seal_auth_secrets(_brief())
    token = once["auth_config"][0]["password"]
    assert seal_auth_secrets(once)["auth_config"][0]["password"] == token


def test_seal_without_a_key_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(get_settings().auth_secrets, "auth_config_secret_keys", SecretStr(""))
    with pytest.raises(AuthSecretError) as exc:
        seal_auth_secrets(_brief())
    assert _KC not in str(exc.value)


def test_a_brief_without_secrets_needs_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(get_settings().auth_secrets, "auth_config_secret_keys", SecretStr(""))
    brief = {"auth_config": {"login_url": "https://x.example", "sso_button": "Keycloak"}}
    assert seal_auth_secrets(brief) == brief


def test_unseal_with_the_wrong_key_refuses(key: str, monkeypatch: pytest.MonkeyPatch) -> None:
    sealed = seal_auth_secrets(_brief())
    monkeypatch.setattr(
        get_settings().auth_secrets,
        "auth_config_secret_keys",
        SecretStr(Fernet.generate_key().decode()),
    )
    with pytest.raises(AuthSecretError):
        unseal_auth(sealed["auth_config"])


def test_rotation_old_key_still_opens(key: str, monkeypatch: pytest.MonkeyPatch) -> None:
    sealed = seal_auth_secrets(_brief())
    new = Fernet.generate_key().decode()
    monkeypatch.setattr(
        get_settings().auth_secrets, "auth_config_secret_keys", SecretStr(f"{new},{key}")
    )
    assert _logins(unseal_auth(sealed["auth_config"]))[1]["password"] == _TW


def test_redact_hides_plaintext_and_sealed_alike(key: str) -> None:
    plain = redact_auth_secrets(_brief())
    sealed = redact_auth_secrets(seal_auth_secrets(_brief()))
    for out in (plain, sealed):
        assert out["auth_config"][0]["password"] == REDACTED
        assert out["sources"][0]["auth_info"]["password"] == REDACTED
        assert out["auth_config"][1]["username"] == "tw"


def test_restore_treats_a_reordered_login_list_as_a_change(key: str) -> None:
    stored = seal_auth_secrets(_brief())
    echoed = redact_auth_secrets(stored)
    echoed["auth_config"].reverse()  # list order is content: not identical ⇒ drop all
    restored = seal_auth_secrets(restore_redacted(echoed, stored))
    assert all("password" not in e for e in restored["auth_config"])


def test_validate_rejects_duplicate_or_originless_logins() -> None:
    validate_auth_block({"username": "single-dict-ok"})
    with pytest.raises(ValueError, match="twice"):
        validate_auth_block([{"login_url": "https://a.example/x"}, {"origin": "https://A.example"}])
    with pytest.raises(ValueError, match="origin"):
        validate_auth_block([{"username": "u"}])


def test_auth_for_origin_picks_the_matching_login() -> None:
    block = _brief()["auth_config"]
    assert _one(auth_for_origin(block, "https://twenty.example/objects"))["username"] == "tw"
    # Never "the first login": no exact origin match ⇒ no login at all.
    assert auth_for_origin(block, "https://unknown.example") is None
    assert auth_for_origin({"username": "no-origin", "password": _KC}, "https://x.example") is None
    assert auth_for_origin(block, "https://attacker.example\\@twenty.example/") is None


_VICTIM_PW = "hunter2"  # pragma: allowlist secret


def _single(login: dict, target: str = "https://victim.example") -> dict:
    return {"target_url": target, "auth_config": login}


@pytest.mark.parametrize(
    ("new_login", "new_target"),
    [
        # the audit repro: one login, its page repointed at the attacker
        ({"login_url": "https://attacker.example/login"}, "https://victim.example"),
        # the `origin` LABEL still says victim, but the page FTR opens is attacker's
        (
            {"origin": "https://victim.example", "login_url": "https://attacker.example/x"},
            "https://victim.example",
        ),
        # userinfo: the browser connects to attacker.example
        ({"login_url": "https://victim.example@attacker.example/login"}, "https://victim.example"),
        # same host, different port / scheme
        ({"login_url": "https://victim.example:8443/login"}, "https://victim.example"),
        ({"login_url": "http://victim.example/login"}, "https://victim.example"),
        # no login_url: FTR falls back to target_url, which moved
        ({}, "https://attacker.example"),
    ],
)
def test_restore_drops_the_secret_when_the_login_origin_changes(
    key: str, new_login: dict, new_target: str
) -> None:
    stored_login = (
        {"login_url": "https://victim.example/login", "password": _VICTIM_PW}
        if new_login
        else {"password": _VICTIM_PW}
    )
    stored = seal_auth_secrets(_single(stored_login))
    echoed = _single({**new_login, "username": "u", "password": dict(REDACTED)}, new_target)
    restored = seal_auth_secrets(restore_redacted(echoed, stored))
    opened = unseal_auth(restored["auth_config"])
    assert isinstance(opened, dict)
    assert "password" not in opened
    assert _VICTIM_PW not in str(opened)


def test_restore_keeps_the_secret_only_for_an_identical_brief(key: str) -> None:
    stored = seal_auth_secrets(
        {
            "target_url": "https://victim.example",
            "auth_info": {"username": "u", "password": _VICTIM_PW},
            "auth_config": {"login_url": "https://victim.example/l", "password": _KC},
        }
    )
    echoed = redact_auth_secrets(stored)
    # Same content, different key order: still identical, and each secret goes back
    # to ITS entry (paired by JSON path, not walk order).
    echoed = {k: echoed[k] for k in ("auth_config", "auth_info", "target_url")}
    restored = seal_auth_secrets(restore_redacted(echoed, stored))
    assert unseal_auth(restored["auth_info"]) == {"username": "u", "password": _VICTIM_PW}
    opened = unseal_auth(restored["auth_config"])
    assert isinstance(opened, dict)
    assert opened["password"] == _KC


def _stored_login() -> dict:
    return seal_auth_secrets(
        {
            "target_url": "https://victim.example",
            "auth_config": {
                "login_url": "https://victim.example/login",
                "username": "u",
                "password": _VICTIM_PW,
            },
        }
    )


@pytest.mark.parametrize(
    "edit",
    [
        pytest.param(lambda b: b.update(title_hint="renamed"), id="unrelated-key-added"),
        # Steering the CAPTURE is not steering the login: dispatch releases the login
        # only to its bound origin (see test_site_capture's off-origin tests).
        pytest.param(
            lambda b: b.update(site_capture={"paths": ["https://attacker.example/x"]}),
            id="site_capture.paths-absolute",
        ),
        pytest.param(
            lambda b: b.update(site_capture={"paths": [{"path": "//attacker.example"}]}),
            id="site_capture.paths-dict-scheme-relative",
        ),
        # Same bound origin, so the same site.
        pytest.param(lambda b: b.update(target_url="https://victim.example/"), id="trailing-slash"),
    ],
)
def test_an_edit_that_leaves_the_login_alone_keeps_the_password(key: str, edit) -> None:
    """George 2026-10-28: a brief edit must not silently drop the saved password."""
    stored = _stored_login()
    echoed = redact_auth_secrets(stored)
    edit(echoed)
    restored = seal_auth_secrets(restore_redacted(echoed, stored))
    opened = unseal_auth(restored["auth_config"])
    assert isinstance(opened, dict)
    assert opened["password"] == _VICTIM_PW
    assert login_needs_reentry(restored) == []
    assert _VICTIM_PW not in json.dumps(redact_auth_secrets(restored))  # still never returned


@pytest.mark.parametrize(
    "edit",
    [
        pytest.param(
            lambda b: b["auth_config"].update(login_url="https://victim.example/signin"),
            id="login-url-path",
        ),
        pytest.param(
            lambda b: b["auth_config"].update(login_url="https://attacker.example/login"),
            id="login-url-origin",
        ),
        pytest.param(lambda b: b["auth_config"].update(username="someone-else"), id="username"),
        pytest.param(
            lambda b: b["auth_config"].update(password_selector="#search"), id="password-selector"
        ),
        pytest.param(lambda b: b.update(target_url="https://attacker.example"), id="bound-site"),
    ],
)
def test_changing_the_login_drops_the_password_and_says_so(key: str, edit) -> None:
    stored = _stored_login()
    echoed = redact_auth_secrets(stored)
    edit(echoed)
    restored = seal_auth_secrets(restore_redacted(echoed, stored))
    assert "password" not in restored["auth_config"]
    assert _VICTIM_PW not in str(unseal_auth(restored["auth_config"]))
    assert login_needs_reentry(restored) == ["auth_config"]
    # Reported to the client: a plain, non-secret flag on the login itself.
    shown = redact_auth_secrets(restored)
    assert shown["auth_config"]["needs_reentry"] is True
    assert _VICTIM_PW not in json.dumps(shown)
    # Saving again without re-entering keeps asking; typing a password clears it.
    again = seal_auth_secrets(restore_redacted(redact_auth_secrets(restored), restored))
    assert again["auth_config"]["needs_reentry"] is True
    typed = redact_auth_secrets(restored)
    typed["auth_config"]["password"] = "fresh-pw"  # pragma: allowlist secret
    cleared = seal_auth_secrets(restore_redacted(typed, restored))
    assert "needs_reentry" not in cleared["auth_config"]


def test_a_newly_typed_password_after_a_login_change_is_kept(key: str) -> None:
    stored = _stored_login()
    echoed = redact_auth_secrets(stored)
    echoed["auth_config"]["login_url"] = "https://victim.example/signin"
    echoed["auth_config"]["password"] = "new-pw"  # pragma: allowlist secret
    restored = seal_auth_secrets(restore_redacted(echoed, stored))
    opened = unseal_auth(restored["auth_config"])
    assert isinstance(opened, dict)
    assert opened["password"] == "new-pw"  # pragma: allowlist secret
    assert login_needs_reentry(restored) == []


def test_auditor_repro_source_url_added_beside_auth_info_drops_the_secret(key: str) -> None:
    """website_review: auth_info bound to target_url=victim; a PATCH ADDS
    source_url=attacker (site_capture logs in there) and echoes REDACTED."""
    stored = seal_auth_secrets(
        {
            "target_url": "https://victim.example",
            "auth_info": {"username": "u", "password": _VICTIM_PW},
        }
    )
    echoed = redact_auth_secrets(stored)
    echoed["source_url"] = "https://attacker.example"
    restored = seal_auth_secrets(restore_redacted(echoed, stored))
    assert "password" not in restored["auth_info"]
    assert _VICTIM_PW not in str(unseal_auth(restored["auth_info"]))


_VICTIM_PKG = "package_ref:tenant-a/victim-suite@sha256:" + "a" * 64


@pytest.mark.parametrize(
    "edit",
    [
        pytest.param(
            lambda b: b.update(package_url="package_ref:tenant-a/mine@sha256:" + "b" * 64),
            id="package_url-swapped",
        ),
        pytest.param(lambda b: b.pop("package_url"), id="package_url-removed"),
        pytest.param(lambda b: b.update(login_label="gm"), id="login_label"),
        # Ambiguous before AND after (two distinct origins either way): the collapsed
        # context origin is None on both sides, so only the per-key origins see it.
        pytest.param(
            lambda b: b.update(target_url="https://attacker.example"),
            id="ambiguous-target-swapped",
        ),
    ],
)
def test_changing_the_code_that_receives_the_login_drops_the_password(key: str, edit) -> None:
    """Auditor repro: demo_video_record sends the WHOLE unsealed auth_config to the
    brief's package_url. A tenant member who swaps package_url for their own
    uploaded package and echoes REDACTED must NOT get the saved password restored."""
    stored = seal_auth_secrets(
        {
            "target_url": "https://victim.example",
            "source_url": "https://docs.victim-docs.example",
            "package_url": _VICTIM_PKG,
            "login_label": "requester",
            "auth_config": {
                "login_url": "https://victim.example/login",
                "username": "u",
                "label": "requester",
                "password": _VICTIM_PW,
            },
        }
    )
    echoed = redact_auth_secrets(stored)
    edit(echoed)
    restored = seal_auth_secrets(restore_redacted(echoed, stored))
    assert "password" not in restored["auth_config"]
    assert _VICTIM_PW not in str(unseal_auth(restored["auth_config"]))
    assert login_needs_reentry(restored) == ["auth_config"]
    # Control: the same brief echoed UNCHANGED keeps the password.
    kept = seal_auth_secrets(restore_redacted(redact_auth_secrets(stored), stored))
    opened = unseal_auth(kept["auth_config"])
    assert isinstance(opened, dict) and opened["password"] == _VICTIM_PW


# The re-audit repro: urlsplit reads host twenty.example, a browser (WHATWG treats
# "\" as "/") navigates to attacker.example.
_BACKSLASH = "https://attacker.example\\@twenty.example/"


def test_restore_refuses_the_backslash_parser_differential(key: str) -> None:
    stored = seal_auth_secrets(
        _single({"login_url": "https://twenty.example/", "password": _VICTIM_PW})
    )
    for echoed in (
        _single({"login_url": _BACKSLASH, "password": dict(REDACTED)}),
        # same trick through the target_url fallback (no login_url)
        _single({"password": dict(REDACTED)}, target=_BACKSLASH),
    ):
        restored = seal_auth_secrets(restore_redacted(echoed, stored))
        assert "password" not in restored["auth_config"]
        assert _VICTIM_PW not in str(unseal_auth(restored["auth_config"]))


@pytest.mark.parametrize(
    "url",
    [
        _BACKSLASH,
        "https://attacker.example%5C@twenty.example/",
        "https://attacker.example%40twenty.example/",
        "https://user@twenty.example/",
        "https://twenty.example /x",
        "https://twenty.example/\tx",
        "http://twenty.example/",
        "javascript:alert(1)",
        "https:///nohost",
    ],
)
def test_strict_login_url_rejects_parser_differentials(url: str) -> None:
    with pytest.raises(ValueError):
        strict_login_url(url, "login_url")
    assert canonical_origin(url) is None


def test_canonical_origin_is_one_shape() -> None:
    assert canonical_origin("https://Twenty.Example/a?b#c") == "https://twenty.example"
    assert canonical_origin("https://twenty.example:443/") == "https://twenty.example"
    assert canonical_origin("https://twenty.example:8443/") == "https://twenty.example:8443"


@pytest.mark.parametrize(
    "brief",
    [
        {"target_url": "https://ok.example", "auth_config": {"login_url": _BACKSLASH}},
        {"target_url": _BACKSLASH, "auth_config": {"username": "u"}},
        {"target_url": "https://ok.example", "auth_info": {"login_url": "http://ok.example"}},
        {"sources": [{"ref": _BACKSLASH, "auth_info": {"username": "u"}}]},
        {
            "target_url": "https://ok.example",
            "auth_config": {"origin": "https://ok.example", "login_url": "https://evil.example"},
        },
    ],
)
def test_validate_brief_logins_rejects_unsafe_login_urls(brief: dict) -> None:
    with pytest.raises(ValueError):
        validate_brief_logins(brief)


def test_validate_brief_logins_leaves_login_free_briefs_alone() -> None:
    # No login in the brief ⇒ an http target stays legal (strictness is login-bound).
    assert validate_brief_logins({"target_url": "http://plain.example", "angle": "x"}) is None


def test_an_originless_login_binds_only_to_its_single_context_origin() -> None:
    login = {"type": "bearer", "token": _KC}
    one = {"target_url": "https://victim.example", "auth_info": login}
    assert auth_for_origin(login, "https://victim.example/x", one) == login
    assert auth_for_origin(login, "https://attacker.example", one) is None
    two = {**one, "source_url": "https://attacker.example"}
    # Two bound origins: ambiguous ⇒ bound to nothing, and refused at the door.
    assert auth_for_origin(login, "https://victim.example", two) is None
    with pytest.raises(ValueError, match="exactly one target"):
        validate_brief_logins(two)


_COOKIE = "session=abc123secret"  # pragma: allowlist secret


def test_cookies_are_sealed_redacted_and_reopened(key: str) -> None:
    jar = [{"name": "session", "value": "abc123secret", "domain": "victim.example"}]
    brief = {
        "target_url": "https://victim.example",
        "auth_info": {"type": "cookies", "cookies": jar, "cookie_header": _COOKIE},
    }
    assert has_plaintext_secret(brief)
    sealed = seal_auth_secrets(copy.deepcopy(brief))
    assert "abc123secret" not in json.dumps(sealed)
    assert not has_plaintext_secret(sealed)
    shown = redact_auth_secrets(sealed)
    assert shown["auth_info"]["cookies"] == REDACTED
    assert shown["auth_info"]["cookie_header"] == REDACTED
    opened = unseal_auth(sealed["auth_info"])
    assert isinstance(opened, dict)
    assert opened["cookies"] == jar  # a list survives the round-trip as a list
    assert opened["cookie_header"] == _COOKIE


_IDP_URL = "https://auth.example/realms/app/protocol/openid-connect/auth?client_id=app"


def test_a_single_sign_on_login_signs_into_the_site_it_declares() -> None:
    """Prod run 01a17e6f: login_url on the identity provider, the app as source_url.

    The site a login signs into is DECLARED (``signs_into``), never inferred from
    origins differing; the declaration itself is never forwarded to the recorder."""
    login = {
        "type": "basic",
        "username": "u",
        "password": _KC,
        "login_url": _IDP_URL,
        "signs_into": "https://app.example",
    }
    brief = {"source_url": "https://app.example/", "auth_info": login}
    sent = {k: v for k, v in login.items() if k != "signs_into"}
    assert auth_for_origin(login, "https://app.example/submit", brief) == sent
    # Never released for the identity provider itself, nor for any other site.
    assert auth_for_origin(login, "https://auth.example/", brief) is None
    assert auth_for_origin(login, "https://attacker.example/", brief) is None
    # A label selects it only when it carries that label.
    assert auth_for_origin(login, "https://app.example/", brief, label="gm") is None


def test_without_a_declaration_a_login_signs_into_its_own_sign_in_page_only() -> None:
    """No inference: the same login WITHOUT ``signs_into`` is not released for the app."""
    login = {"type": "basic", "username": "u", "password": _KC, "login_url": _IDP_URL}
    brief = {"source_url": "https://app.example/", "auth_info": login}
    assert auth_for_origin(login, "https://app.example/submit", brief) is None
    assert auth_for_origin(login, "https://auth.example/x", brief) == login


def test_signs_into_must_be_a_strict_https_origin() -> None:
    with pytest.raises(ValueError, match="signs_into"):
        validate_auth_block({"login_url": _IDP_URL, "signs_into": "http://app.example"})
    with pytest.raises(ValueError, match="signs_into"):
        validate_auth_block({"login_url": _IDP_URL, "signs_into": "https://a.example\\@b.example"})


def test_an_explicit_origin_still_decides_which_site_a_login_is_for() -> None:
    login = {"username": "u", "password": _KC, "origin": "https://auth.example"}
    brief = {"source_url": "https://app.example/", "auth_info": login}
    assert auth_for_origin(login, "https://app.example/", brief) is None
    assert auth_for_origin(login, "https://auth.example/x", brief) == login


def test_a_list_of_logins_is_still_keyed_by_each_login_origin() -> None:
    block = [{"username": "u", "password": _KC, "login_url": _IDP_URL}]
    brief = {"target_url": "https://app.example/", "auth_config": block}
    assert auth_for_origin(block, "https://app.example/", brief) is None
    assert auth_for_origin(block, "https://auth.example/", brief) == block[0]
