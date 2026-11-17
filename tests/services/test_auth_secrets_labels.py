"""Same-origin logins told apart by ``label``: a login is keyed ``(origin, label)``.

The recording that needs this signs ONE app (chat.raisemytickets.com) in twice — as the
requester and as the approving manager — in a single run. Origin-only uniqueness made
that brief impossible to submit.
"""

from __future__ import annotations

import pytest

from src.api.schemas.projects import ShowcaseBrief
from src.services.auth_secrets import (
    REDACTED,
    auth_for_origin,
    redact_auth_secrets,
    seal_auth_secrets,
    validate_auth_block,
)

_CHAT = "https://chat.example.test"
_PW_A = "pw-a"  # pragma: allowlist secret
_PW_B = "pw-b"  # pragma: allowlist secret


def _pair() -> list[dict]:
    return [
        {"origin": _CHAT, "label": "requester", "username": "req", "password": _PW_A},
        {"origin": _CHAT, "label": "gm", "username": "mgr", "password": _PW_B},
    ]


def test_same_origin_with_distinct_labels_is_accepted() -> None:
    block = _pair()
    validate_auth_block(block)  # raises ValueError if the pair were refused
    assert {(e["origin"], e["label"]) for e in block} == {(_CHAT, "requester"), (_CHAT, "gm")}


def test_same_origin_same_label_is_rejected() -> None:
    block = _pair()
    block[1]["label"] = "requester"
    with pytest.raises(ValueError, match="'requester' twice"):
        validate_auth_block(block)


def test_same_origin_without_labels_is_still_rejected() -> None:
    with pytest.raises(ValueError, match="twice"):
        validate_auth_block([{"origin": _CHAT}, {"login_url": f"{_CHAT}/login"}])


def test_an_unlabelled_login_may_not_share_an_origin_with_a_labelled_one() -> None:
    with pytest.raises(ValueError, match="needs a label"):
        validate_auth_block([{"origin": _CHAT, "label": "gm"}, {"origin": _CHAT}])


@pytest.mark.parametrize("bad", ["GM"])
def test_label_must_be_a_slug(bad: object) -> None:
    with pytest.raises(ValueError, match="label"):
        validate_auth_block([{"origin": _CHAT, "label": bad}])


def test_a_single_labelled_login_per_origin_is_fine() -> None:
    block = [{"origin": _CHAT, "label": "gm"}, {"origin": "https://other.example"}]
    validate_auth_block(block)  # raises ValueError if refused
    assert (auth_for_origin(block, "https://other.example") or {}).get("label") is None
    assert (auth_for_origin(block, _CHAT, label="gm") or {}).get("label") == "gm"


def test_auth_for_origin_selects_by_label() -> None:
    block = _pair()
    assert (auth_for_origin(block, _CHAT, label="gm") or {})["username"] == "mgr"
    assert (auth_for_origin(block, _CHAT, label="requester") or {})["username"] == "req"
    assert auth_for_origin(block, _CHAT, label="nobody") is None


def test_auth_for_origin_never_guesses_between_same_origin_logins() -> None:
    # Two logins on the origin and no label asked for: picking either is a guess.
    assert auth_for_origin(_pair(), _CHAT) is None
    # A lone login on the origin is still returned without a label (backward compatible).
    assert auth_for_origin([_pair()[0]], _CHAT) is not None


def test_label_is_not_a_secret_and_survives_redaction() -> None:
    redacted = redact_auth_secrets({"auth_config": _pair()})["auth_config"]
    assert [e["label"] for e in redacted] == ["requester", "gm"]
    assert all(e["password"] == REDACTED for e in redacted)


def test_label_survives_sealing(monkeypatch: pytest.MonkeyPatch) -> None:
    from cryptography.fernet import Fernet
    from pydantic import SecretStr

    from src.config import get_settings

    monkeypatch.setattr(
        get_settings().auth_secrets,
        "auth_config_secret_keys",
        SecretStr(Fernet.generate_key().decode()),
    )
    sealed = seal_auth_secrets({"auth_config": _pair()})["auth_config"]
    assert [e["label"] for e in sealed] == ["requester", "gm"]
    assert all("$sealed" in e["password"] for e in sealed)


def test_the_brief_door_accepts_same_origin_labelled_logins() -> None:
    brief = ShowcaseBrief.model_validate({"target_url": _CHAT, "auth_config": _pair()})
    assert brief.auth_config is not None and len(brief.auth_config) == 2


def test_the_brief_door_rejects_same_origin_unlabelled_logins() -> None:
    with pytest.raises(ValueError, match="twice"):
        ShowcaseBrief.model_validate(
            {"target_url": _CHAT, "auth_config": [{"origin": _CHAT}, {"origin": _CHAT}]}
        )
