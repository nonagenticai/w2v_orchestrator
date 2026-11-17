"""A brief's login password is sealed at rest and never returned by any read route.

Measured 2026-10-02: the showcase spike's ``brief.auth_config.password`` sat in
plaintext in ``projects.brief`` AND ``runs.config_snapshot``, and both
``GET /api/projects/{id}`` and ``GET /api/runs/{id}`` returned it verbatim.
"""

from __future__ import annotations

import logging
from uuid import UUID

import pytest
from cryptography.fernet import Fernet
from httpx import AsyncClient
from pydantic import SecretStr
from sqlalchemy import text

from src.config import get_settings
from src.services.auth_secrets import REDACTED, unseal_auth

pytestmark = pytest.mark.asyncio

_KC_PW = "kc-Secret-0001"
_TW_PW = "tw-Secret-0002"
_ROTATED = "tw-rotated"  # pragma: allowlist secret


@pytest.fixture
def secret_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        get_settings().auth_secrets,
        "auth_config_secret_keys",
        SecretStr(Fernet.generate_key().decode()),
    )


def _project() -> dict:
    return {
        "type": "showcase",
        "title": "multi-login showcase",
        "brief": {
            "target_url": "https://marketing.example/strategy",
            "auth_config": [
                {
                    "type": "form",
                    "login_url": "https://marketing.example/strategy",
                    "username": "kc-user",
                    "password": _KC_PW,
                },
                {
                    "origin": "https://twenty.example",
                    "login_url": "https://twenty.example/",
                    "username": "tw@example.com",
                    "password": _TW_PW,
                },
            ],
        },
        "deliverables": [
            {"name": "d", "format": "16x9", "captions_mode": "none", "target_duration_seconds": 30}
        ],
    }


async def _raw(db_session_factory, sql: str, **params) -> str:
    async with db_session_factory() as s:
        return str((await s.execute(text(sql), params)).scalar_one())


async def test_no_read_route_returns_the_password_and_the_rows_are_sealed(
    client: AsyncClient, db_session_factory, secret_key, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    created = await client.post("/api/projects", json=_project())
    assert created.status_code == 201, created.text
    pid = created.json()["id"]

    detail = await client.get(f"/api/projects/{pid}")
    listing = await client.get("/api/projects")
    for resp in (created, detail, listing):
        assert resp.status_code == 200 or resp.status_code == 201
        assert _KC_PW not in resp.text and _TW_PW not in resp.text
    logins = detail.json()["brief"]["auth_config"]
    assert [e["password"] for e in logins] == [REDACTED, REDACTED]
    assert logins[1]["username"] == "tw@example.com"  # only the secret is hidden

    row = await _raw(
        db_session_factory, "SELECT brief::text FROM projects WHERE id = :id", id=UUID(pid)
    )
    assert _KC_PW not in row and _TW_PW not in row and "$sealed" in row

    deliv = detail.json()["deliverables"][0]["id"]
    run = await client.post(f"/api/deliverables/{deliv}/runs", json={"pause_after_stages": []})
    assert run.status_code == 201, run.text
    rid = run.json()["id"]
    run_detail = await client.get(f"/api/runs/{rid}")
    assert run_detail.status_code == 200, run_detail.text
    assert _KC_PW not in run_detail.text and _TW_PW not in run_detail.text
    snap_logins = run_detail.json()["config_snapshot"]["brief"]["auth_config"]
    assert [e["password"] for e in snap_logins] == [REDACTED, REDACTED]

    snap = await _raw(
        db_session_factory, "SELECT config_snapshot::text FROM runs WHERE id = :id", id=UUID(rid)
    )
    assert _KC_PW not in snap and _TW_PW not in snap and "$sealed" in snap

    assert _KC_PW not in caplog.text and _TW_PW not in caplog.text


async def test_patch_echoing_redacted_keeps_the_stored_secret(
    client: AsyncClient, db_session_factory, secret_key
) -> None:
    created = await client.post("/api/projects", json=_project())
    assert created.status_code == 201, created.text
    pid = created.json()["id"]
    brief = (await client.get(f"/api/projects/{pid}")).json()["brief"]
    brief["auth_config"][1]["password"] = _ROTATED
    r = await client.patch(f"/api/projects/{pid}", json={"brief": brief})
    assert r.status_code == 200, r.text
    assert _ROTATED not in r.text

    async with db_session_factory() as s:
        stored = (
            await s.execute(text("SELECT brief FROM projects WHERE id = :id"), {"id": UUID(pid)})
        ).scalar_one()
    plain = unseal_auth(stored["auth_config"])
    assert isinstance(plain, list)
    assert [e["password"] for e in plain] == [_KC_PW, _ROTATED]


async def test_no_key_configured_refuses_rather_than_storing_plaintext(
    client: AsyncClient, db_session_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(get_settings().auth_secrets, "auth_config_secret_keys", SecretStr(""))
    r = await client.post("/api/projects", json=_project())
    assert r.status_code == 503, r.text
    assert _KC_PW not in r.text
    n = await _raw(
        db_session_factory,
        "SELECT count(*) FROM projects WHERE brief::text LIKE :p",
        p=f"%{_KC_PW}%",
    )
    assert n == "0"


async def test_duplicate_login_origin_is_rejected_without_echoing_the_password(
    app, client: AsyncClient, secret_key
) -> None:
    # The production envelope (w2v_common, wired by create_app) drops pydantic's
    # `input`; the bare test app does not have it, so install it here — the default
    # FastAPI 422 echoes the whole body, password included.
    from w2v_common.asgi.error_envelope import register_error_envelope_handlers

    register_error_envelope_handlers(app)
    body = _project()
    body["brief"]["auth_config"][1]["origin"] = "https://marketing.example"
    r = await client.post("/api/projects", json=body)
    assert r.status_code == 422, r.text
    assert _KC_PW not in r.text and _TW_PW not in r.text


async def test_patch_repointing_a_login_never_carries_the_stored_password(
    client: AsyncClient, db_session_factory, secret_key
) -> None:
    created = await client.post("/api/projects", json=_project())
    assert created.status_code == 201, created.text
    pid = created.json()["id"]
    brief = (await client.get(f"/api/projects/{pid}")).json()["brief"]
    # Echo REDACTED but repoint the Twenty login at another origin.
    brief["auth_config"][1]["login_url"] = "https://attacker.example/login"
    brief["auth_config"][1].pop("origin")
    r = await client.patch(f"/api/projects/{pid}", json={"brief": brief})
    assert r.status_code == 200, r.text

    async with db_session_factory() as s:
        stored = (
            await s.execute(text("SELECT brief FROM projects WHERE id = :id"), {"id": UUID(pid)})
        ).scalar_one()
    plain = unseal_auth(stored["auth_config"])
    assert isinstance(plain, list)
    # Only the REPOINTED login loses its secret (it was approved for twenty.example);
    # the untouched Keycloak login keeps its own (George 2026-10-28).
    assert "password" not in plain[1]
    assert _TW_PW not in str(plain)
    assert plain[0]["password"] == _KC_PW
    # ...and the API TELLS the client which login needs its password re-entered.
    shown = (await client.get(f"/api/projects/{pid}")).json()["brief"]["auth_config"]
    assert shown[1]["needs_reentry"] is True
    assert "needs_reentry" not in shown[0]
    assert _KC_PW not in str(shown) and _TW_PW not in str(shown)


async def test_patch_editing_an_unrelated_brief_field_keeps_every_saved_password(
    client: AsyncClient, db_session_factory, secret_key
) -> None:
    """The Brief page sends the WHOLE brief on every save. Editing anything that is
    not the login (here the target page's path) used to drop every saved password."""
    created = await client.post("/api/projects", json=_project())
    assert created.status_code == 201, created.text
    pid = created.json()["id"]
    brief = (await client.get(f"/api/projects/{pid}")).json()["brief"]
    assert _KC_PW not in str(brief) and _TW_PW not in str(brief)
    brief["target_url"] = "https://marketing.example/strategy/q4"
    brief["instructions"] = "Walk through the Q4 board."
    r = await client.patch(f"/api/projects/{pid}", json={"brief": brief})
    assert r.status_code == 200, r.text
    assert _KC_PW not in r.text and _TW_PW not in r.text

    async with db_session_factory() as s:
        stored = (
            await s.execute(text("SELECT brief FROM projects WHERE id = :id"), {"id": UUID(pid)})
        ).scalar_one()
    plain = unseal_auth(stored["auth_config"])
    assert isinstance(plain, list)
    assert [e["password"] for e in plain] == [_KC_PW, _TW_PW]


async def test_backslash_login_url_is_rejected_at_the_door(
    app, client: AsyncClient, db_session_factory, secret_key
) -> None:
    from w2v_common.asgi.error_envelope import register_error_envelope_handlers

    register_error_envelope_handlers(app)
    created = await client.post("/api/projects", json=_project())
    assert created.status_code == 201, created.text
    pid = created.json()["id"]
    brief = (await client.get(f"/api/projects/{pid}")).json()["brief"]
    brief["auth_config"][1].pop("origin")
    brief["auth_config"][1]["login_url"] = "https://attacker.example\\@twenty.example/"
    r = await client.patch(f"/api/projects/{pid}", json={"brief": brief})
    assert r.status_code == 422, r.text

    async with db_session_factory() as s:
        stored = (
            await s.execute(text("SELECT brief FROM projects WHERE id = :id"), {"id": UUID(pid)})
        ).scalar_one()
    assert "attacker" not in str(stored)


async def test_session_cookies_are_sealed_and_never_returned(
    client: AsyncClient, db_session_factory, secret_key
) -> None:
    cookie = "sess-cookie-0003"  # pragma: allowlist secret
    body = {
        "type": "website_review",
        "title": "cookie login",
        "brief": {
            "target_url": "https://victim.example/",
            "auth_info": {
                "type": "cookies",
                "cookies": [{"name": "session", "value": cookie}],
                "cookie_header": f"session={cookie}",
            },
        },
        "deliverables": [
            {"name": "d", "format": "16x9", "captions_mode": "none", "target_duration_seconds": 60}
        ],
    }
    created = await client.post("/api/projects", json=body)
    assert created.status_code == 201, created.text
    pid = created.json()["id"]
    detail = await client.get(f"/api/projects/{pid}")
    assert cookie not in created.text and cookie not in detail.text
    shown = detail.json()["brief"]["auth_info"]
    assert shown["cookies"] == REDACTED and shown["cookie_header"] == REDACTED
    row = await _raw(
        db_session_factory, "SELECT brief::text FROM projects WHERE id = :id", id=UUID(pid)
    )
    assert cookie not in row and "$sealed" in row
