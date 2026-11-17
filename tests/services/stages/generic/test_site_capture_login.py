"""site_capture uses the project's saved sign-in on every run, or fails the stage loudly.

Prod run 01a17e6f (2026-10-27): the brief carried a sealed single-sign-on login
(``login_url`` on the identity provider, ``source_url`` on the app). The login was
keyed by its ``login_url`` origin, so it bound to the identity provider, no capture
got it, the app was recorded signed out, and the run "completed" with a video of a
login screen and a thin-text warning.
"""

from __future__ import annotations

import copy
import io
import zipfile
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet
from pydantic import SecretStr

from src.config import get_settings
from src.models.failure_class import CUSTOMER_INPUT, attribute_exception, failure_detail_for
from src.services.auth_secrets import seal_auth_secrets
from src.services.runs.run_context import RunContext
from src.services.stages.generic.site_capture import SiteCaptureLoginError, SiteCaptureStage

_STORE = "src.services.stages.generic.site_capture.store_capture_clip"
_STORE_IMG = "src.services.stages.generic.site_capture.store_capture_image"
_PW = "pw-not-real"  # pragma: allowlist secret
_IDP = (
    "https://auth.example.com/realms/app/protocol/openid-connect/auth"
    "?client_id=app-frontend&redirect_uri=https%3A%2F%2Fapp.example.com%2F"
)
_SSO_LOGIN = {
    "type": "basic",
    "username": "u@example.com",
    "password": _PW,
    "login_url": _IDP,
    "signs_into": "https://app.example.com",
}
#: What the recorder receives: never the orchestrator-only declaration.
_SENT = {k: v for k, v in _SSO_LOGIN.items() if k != "signs_into"}


@pytest.fixture
def key(monkeypatch: pytest.MonkeyPatch) -> str:
    k = Fernet.generate_key().decode()
    monkeypatch.setattr(get_settings().auth_secrets, "auth_config_secret_keys", SecretStr(k))
    return k


def _zip(ids: list[str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for sid in ids:
            zf.writestr(f"{sid}.png", b"png")
    return buf.getvalue()


def _client(ids: list[str]) -> AsyncMock:
    client = AsyncMock()
    client.capture_screens = AsyncMock(return_value="job-s")
    client.record_url = AsyncMock(return_value="job-c")
    client.wait_for_content_recording = AsyncMock(
        return_value={"status": "completed", "duration_seconds": 3.0}
    )
    client.get_recording_video = AsyncMock(return_value=(_zip(ids), "application/zip", None))
    return client


def _shotlist_brief(auth: dict | None) -> dict:
    brief: dict = {
        "source_url": "https://app.example.com/",
        "shotlist": [
            {"kind": "url", "url": "https://app.example.com/", "narration": "home"},
            {"kind": "url", "url": "https://app.example.com/submit", "narration": "submit"},
        ],
    }
    if auth is not None:
        brief["auth_info"] = auth
    return brief


def _ctx(brief: dict) -> RunContext:
    return RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={"brief": brief})


@pytest.mark.asyncio
@pytest.mark.usefixtures("key")
async def test_a_sealed_single_sign_on_login_reaches_the_recorder() -> None:
    """The prod shape, sealed at rest exactly as runs.config_snapshot holds it."""
    brief = seal_auth_secrets(copy.deepcopy(_shotlist_brief(_SSO_LOGIN)))
    assert _PW not in str(brief)  # sealed, as in prod
    client = _client(["shot-000", "shot-001"])
    with patch(_STORE_IMG, new=AsyncMock(return_value="https://minio/s.png")):
        out = await SiteCaptureStage(client=client).run({}, ctx=_ctx(brief))

    call = client.capture_screens.await_args
    assert call is not None
    assert call.kwargs["auth_info"] == _SENT  # unsealed, login_url kept for the form
    assert out.used_real_footage is True


@pytest.mark.asyncio
async def test_a_login_for_another_site_fails_the_stage_instead_of_recording_signed_out() -> None:
    login = {
        **_SENT,
        "origin": "https://other.example.com",
        "login_url": "https://other.example.com/login",
    }
    client = _client(["shot-000", "shot-001"])
    with pytest.raises(SiteCaptureLoginError) as err:
        await SiteCaptureStage(client=client).run({}, ctx=_ctx(_shotlist_brief(login)))

    client.capture_screens.assert_not_awaited()  # nothing recorded signed out
    msg = str(err.value)
    assert msg.startswith("Couldn't sign in to app.example.com")
    assert "Brief page" in msg
    assert "rerun this step" in msg
    assert _PW not in msg
    assert attribute_exception(err.value) == CUSTOMER_INPUT
    assert failure_detail_for(err.value) == {
        "code": "site_login_failed",
        "site": "app.example.com",
        "reason": err.value.reason,
        "message": msg,
    }
    assert 'set "Signs into" to app.example.com' in err.value.reason


@pytest.mark.asyncio
async def test_a_rejected_sign_in_fails_the_stage_with_a_user_message() -> None:
    """The recorder reports the sign-in did not work: a clear stage failure, not slides."""
    client = _client([])
    client.wait_for_content_recording = AsyncMock(
        side_effect=RuntimeError(
            "video_production job job-s failed: site_login_failed: still on the sign-in page"
        )
    )
    with pytest.raises(SiteCaptureLoginError, match="did not accept the saved login"):
        await SiteCaptureStage(client=client).run({}, ctx=_ctx(_shotlist_brief(_SSO_LOGIN)))


@pytest.mark.asyncio
async def test_a_rejected_sign_in_on_the_clip_path_fails_the_stage() -> None:
    client = _client([])
    client.wait_for_content_recording = AsyncMock(
        side_effect=RuntimeError("video_production job job-c failed: site_login_failed: x")
    )
    brief = {"source_url": "https://app.example.com/", "auth_info": _SSO_LOGIN}
    with (
        patch(_STORE, new=AsyncMock(return_value="https://minio/c.mp4")),
        pytest.raises(SiteCaptureLoginError, match=r"^Couldn't sign in to app\.example\.com"),
    ):
        await SiteCaptureStage(client=client).run({}, ctx=_ctx(brief))


@pytest.mark.asyncio
async def test_a_login_applying_to_no_clip_path_fails_the_stage() -> None:
    client = _client([])
    login = {"type": "bearer", "token": "t", "origin": "https://other.example.com"}
    brief = {"source_url": "https://app.example.com/", "auth_info": login}
    with pytest.raises(SiteCaptureLoginError, match="the saved sign-in is for a different site"):
        await SiteCaptureStage(client=client).run({}, ctx=_ctx(brief))
    client.record_url.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_saved_login_still_records_a_public_site() -> None:
    client = _client(["shot-000", "shot-001"])
    with patch(_STORE_IMG, new=AsyncMock(return_value="https://minio/s.png")):
        out = await SiteCaptureStage(client=client).run({}, ctx=_ctx(_shotlist_brief(None)))
    call = client.capture_screens.await_args
    assert call is not None
    assert call.kwargs["auth_info"] is None
    assert out.used_real_footage is True


@pytest.mark.asyncio
@pytest.mark.usefixtures("key")
async def test_capture_after_an_unrelated_brief_edit_still_signs_in() -> None:
    """The Brief round-trip George hit: GET (redacted) → edit a shot → PATCH."""
    from src.services.auth_secrets import redact_auth_secrets, restore_redacted

    stored = seal_auth_secrets(copy.deepcopy(_shotlist_brief(_SSO_LOGIN)))
    echoed = redact_auth_secrets(stored)
    assert _PW not in str(echoed)  # the client never sees the secret
    echoed["shotlist"][1]["narration"] = "Report a bottleneck in two sentences."
    saved = seal_auth_secrets(restore_redacted(echoed, stored))

    client = _client(["shot-000", "shot-001"])
    with patch(_STORE_IMG, new=AsyncMock(return_value="https://minio/s.png")):
        await SiteCaptureStage(client=client).run({}, ctx=_ctx(saved))
    call = client.capture_screens.await_args
    assert call is not None
    assert call.kwargs["auth_info"]["password"] == _PW


@pytest.mark.asyncio
@pytest.mark.usefixtures("key")
async def test_a_changed_login_address_blocks_capture_until_the_password_is_re_entered() -> None:
    from src.services.auth_secrets import redact_auth_secrets, restore_redacted

    stored = seal_auth_secrets(copy.deepcopy(_shotlist_brief(_SSO_LOGIN)))
    echoed = redact_auth_secrets(stored)
    echoed["auth_info"]["login_url"] = (
        "https://auth.example.com/realms/other/protocol/openid-connect/auth"
    )
    saved = seal_auth_secrets(restore_redacted(echoed, stored))
    assert "password" not in saved["auth_info"]

    client = _client(["shot-000", "shot-001"])
    with pytest.raises(SiteCaptureLoginError, match="re-enter the password"):
        await SiteCaptureStage(client=client).run({}, ctx=_ctx(saved))
    client.capture_screens.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_sso_login_that_does_not_declare_its_site_fails_with_what_to_set() -> None:
    """Nothing is inferred: without ``signs_into`` the sign-in is for auth.example."""
    login = {k: v for k, v in _SSO_LOGIN.items() if k != "signs_into"}
    client = _client(["shot-000", "shot-001"])
    with pytest.raises(SiteCaptureLoginError) as err:
        await SiteCaptureStage(client=client).run({}, ctx=_ctx(_shotlist_brief(login)))
    client.capture_screens.assert_not_awaited()
    detail = failure_detail_for(err.value)
    assert detail is not None
    assert detail["site"] == "app.example.com"
    assert "auth.example.com" in detail["reason"]
    assert "Signs into" in detail["reason"]


def test_the_run_api_carries_site_and_reason_to_the_frontend() -> None:
    """``RunDetail.failure_detail`` is a pydantic model: a key it does not declare is
    dropped on the way out, so the run page could only parse ``message`` prose."""
    from src.api.schemas.runs import RunFailureDetail

    err = SiteCaptureLoginError("https://app.example.com/x", "the sign-in page did not accept it")
    detail = failure_detail_for(err)
    assert detail is not None
    shown = RunFailureDetail.model_validate(detail).model_dump(mode="json")
    assert shown["site"] == "app.example.com"
    assert shown["reason"] == "the sign-in page did not accept it"
    assert shown["code"] == "site_login_failed"
