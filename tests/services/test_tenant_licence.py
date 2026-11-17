"""``src/services/tenant_licence.py`` — the introspection verdict, the fallback,
and the sealed-token envelope. DB-free."""

from __future__ import annotations

import datetime
import secrets
from uuid import uuid7

import pytest

from src.clients.licenses_client import LicensesClientError, TenantNotOnboardedError
from src.clients.schemas.licenses import ActiveAppRef, ActiveLicenseView
from src.services import tenant_licence as tl
from src.services.providers.crypto import Keyring
from tests.api.routes._p9_licences import TEST_KEYRING, P9Licences


def _view(**kw: object) -> ActiveLicenseView:
    base: dict[str, object] = {"active": True, "app": ActiveAppRef(key="w2v")}
    return ActiveLicenseView.model_validate(base | kw)


def test_entitled_only_for_an_active_w2v_token_of_this_tenant() -> None:
    tenant, licence = uuid7(), uuid7()
    ok = tl.judge_introspection(_view(tenant_id=str(tenant), license_id=str(licence)), tenant)
    assert ok.entitled and ok.licence_id == licence


def test_a_token_of_another_tenant_is_a_mismatch() -> None:
    tenant = uuid7()
    v = tl.judge_introspection(_view(tenant_id=str(uuid7()), license_id=str(uuid7())), tenant)
    assert (v.outcome, v.reason, v.licence_id) == ("mismatch", "tenant", None)


def test_a_token_of_another_app_is_a_mismatch() -> None:
    tenant = uuid7()
    view = _view(tenant_id=str(tenant), license_id=str(uuid7()), app=ActiveAppRef(key="fr"))
    v = tl.judge_introspection(view, tenant)
    assert (v.outcome, v.reason) == ("mismatch", "app")
    no_app = _view(tenant_id=str(tenant), license_id=str(uuid7()), app=None)
    assert tl.judge_introspection(no_app, tenant).reason == "app"


def test_inactive_passes_the_reason_through_and_expiry_is_inactive() -> None:
    tenant = uuid7()
    v = tl.judge_introspection(ActiveLicenseView(active=False, reason="license_revoked"), tenant)
    assert (v.outcome, v.reason) == ("inactive", "license_revoked")
    past = (datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=1)).isoformat()
    v = tl.judge_introspection(
        _view(tenant_id=str(tenant), license_id=str(uuid7()), expires_at=past), tenant
    )
    assert (v.outcome, v.reason) == ("inactive", "expired")


def test_active_without_a_licence_id_is_could_not_check() -> None:
    tenant = uuid7()
    with pytest.raises(tl.LicencesUnavailableError):
        tl.judge_introspection(_view(tenant_id=str(tenant), license_id=None), tenant)


@pytest.mark.asyncio
async def test_fallback_ref_only_for_a_single_candidate() -> None:
    tenant, a, b = uuid7(), uuid7(), uuid7()
    one = await tl.tenant_active_licence(P9Licences({tenant: [a]}), tenant)
    assert one.entitled and one.licence_ref == a and one.candidate_count == 1
    many = await tl.tenant_active_licence(P9Licences({tenant: [a, b]}), tenant)
    assert many.entitled and many.licence_ref is None and many.candidate_count == 2
    none = await tl.tenant_active_licence(P9Licences(), tenant)
    assert not none.entitled and none.licence_ref is None


@pytest.mark.asyncio
async def test_fallback_answer_for_another_tenant_is_untrusted() -> None:
    class Wrong(P9Licences):
        async def get_active_license(self, *, tenant_id, app_key):  # type: ignore[no-untyped-def,override]
            return _view(tenant_id=str(uuid7()), license_id=str(uuid7()), candidate_count=1)

    with pytest.raises(tl.LicencesUnavailableError):
        await tl.tenant_active_licence(Wrong(), uuid7())


@pytest.mark.asyncio
async def test_failures_classify_refused_vs_unavailable() -> None:
    tenant = uuid7()
    with pytest.raises(tl.LicencesRefusedError):
        await tl.introspect_for_tenant(P9Licences(refuse=400), "t", tenant)
    with pytest.raises(tl.LicencesUnavailableError) as info:
        await tl.introspect_for_tenant(P9Licences(down=True), "t", tenant)
    assert not isinstance(info.value, tl.LicencesRefusedError)


def test_tenant_not_accessible_classifies_as_not_onboarded_not_a_config_fault() -> None:
    """G7: a definite, non-retryable refusal (so every ``except LicencesRefusedError``
    still refuses without a retry) — but its own type, so callers answer 403
    ``TENANT_NOT_ONBOARDED`` instead of 502 ``LICENCES_REFUSED``."""
    err = tl.licences_error(
        TenantNotOnboardedError("x returned 403", status_code=403, code="tenant_not_accessible")
    )
    assert isinstance(err, tl.LicencesTenantNotOnboardedError)
    assert isinstance(err, tl.LicencesRefusedError)
    other = tl.licences_error(LicensesClientError("x returned 403", status_code=403))
    assert not isinstance(other, tl.LicencesTenantNotOnboardedError)


@pytest.mark.asyncio
async def test_fallback_entitlement_says_not_onboarded() -> None:
    tenant = uuid7()
    lic = P9Licences()
    lic.not_onboarded.add(tenant)
    ent = await tl.tenant_entitlement(lic, TEST_KEYRING, tenant, None)
    assert (ent.entitled, ent.state) == (False, "not_onboarded")


def test_sealed_token_opens_only_for_its_tenant_and_keyring() -> None:
    tenant = uuid7()
    sealed = tl.seal_licence_token(TEST_KEYRING, tenant, "lic_tok_secret")
    assert b"lic_tok_secret" not in sealed.ciphertext and "lic_tok" not in repr(sealed)
    assert tl.open_licence_token(TEST_KEYRING, tenant, sealed.ciphertext) == "lic_tok_secret"
    with pytest.raises(tl.LicenceTokenUnreadableError):
        tl.open_licence_token(TEST_KEYRING, uuid7(), sealed.ciphertext)  # other tenant's AAD
    with pytest.raises(tl.LicenceTokenUnreadableError):
        tl.open_licence_token(Keyring([("other", bytes(32))]), tenant, sealed.ciphertext)
    with pytest.raises(tl.LicenceTokenUnreadableError):
        tl.open_licence_token(Keyring([]), tenant, sealed.ciphertext)  # no keyring at all


@pytest.mark.asyncio
async def test_entitlement_uses_the_token_when_stored_else_the_fallback() -> None:
    tenant, by_token, by_fallback = uuid7(), uuid7(), uuid7()
    lic = P9Licences({tenant: [by_fallback]})
    sealed = tl.seal_licence_token(TEST_KEYRING, tenant, lic.token(tenant, by_token))
    ent = await tl.tenant_entitlement(lic, TEST_KEYRING, tenant, sealed.ciphertext)
    assert (ent.source, ent.entitled, ent.licence_ref) == ("token", True, by_token)
    ent = await tl.tenant_entitlement(lic, TEST_KEYRING, tenant, None)
    assert (ent.source, ent.entitled, ent.licence_ref) == ("fallback", True, by_fallback)
    expired = datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=1)
    ent = await tl.tenant_entitlement(
        lic, TEST_KEYRING, tenant, sealed.ciphertext, token_expires_at=expired
    )
    assert ent.source == "fallback", "an expired stored token counts as none"


# ─── review fix 4: an opaque licence token never reaches a log line ──────────

#: Licence tokens are opaque urlsafe strings — no ``eyJ`` / ``Bearer`` marker —
#: shaped like ``secrets.token_urlsafe(32)`` (43 characters).
#: Generated at import, never a literal: secret scanners (gitleaks licence-token rule) flag any
#: committed 43-char urlsafe string, fake or not.
_OPAQUE = secrets.token_urlsafe(32)
assert len(_OPAQUE) == 43


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 403, 422])
async def test_introspect_4xx_echoing_the_token_logs_status_and_reason_only(
    caplog, status: int
) -> None:
    """The REAL client + classifier: a gateway 4xx that echoes the request body
    (the token) leaves nothing of the body in the logs, the excerpt or the error."""
    import httpx

    from src.clients.licenses_client import LicensesClient

    class _Auth:
        async def token(self) -> str:
            return "sa"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status, text=f'{{"error":"bad_token","echo":{request.content.decode()}}}'
        )

    client = LicensesClient(
        authenticator=_Auth(),
        base_url="http://licenses:8544",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        introspect_url="http://licenses:8544/license/introspect",
    )
    caplog.set_level("DEBUG")
    with pytest.raises(tl.LicencesRefusedError) as info:
        await tl.introspect_for_tenant(client, _OPAQUE, uuid7())
    err = info.value
    assert err.status_code == status
    assert err.excerpt.startswith(f"HTTP {status} "), err.excerpt
    assert "bad_token" not in err.excerpt, "no body excerpt at all on the introspect call"
    assert err.__cause__ is None and err.__suppress_context__
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert f"HTTP {status}" in logged, logged
    assert _OPAQUE not in logged and _OPAQUE not in err.excerpt and _OPAQUE not in str(err)


def test_other_licences_4xx_redacts_opaque_runs_but_keeps_the_error_code(caplog) -> None:
    from src.clients.licenses_client import LicensesClientError

    caplog.set_level("WARNING", logger="w2v_orchestrator.tenant_licence")
    b64ish = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVphYmNk+/=="  # pragma: allowlist secret
    exc = LicensesClientError(
        f'get_active_license returned 400: {{"error":"missing_app_key","t":"{_OPAQUE}",'
        f'"b":"{b64ish}"}}',
        status_code=400,
    )
    err = tl.licences_error(exc)
    assert isinstance(err, tl.LicencesRefusedError)
    assert "missing_app_key" in err.excerpt, "short codes survive"
    logged = "\n".join(r.getMessage() for r in caplog.records)
    for secret in (_OPAQUE, b64ish, b64ish[:20]):
        assert secret not in err.excerpt and secret not in logged


def test_redaction_catches_real_tokens_but_keeps_uuids_and_error_codes(caplog) -> None:
    """The threshold sits between the longest diagnostic we must keep and the
    shortest token we must catch: a real ``token_urlsafe(32)`` (43) is redacted —
    even glued to a UUID — while a canonical UUID (36, also inside a long path
    run) and a 35-char error code survive."""
    import secrets

    from src.clients.licenses_client import LicensesClientError

    token = secrets.token_urlsafe(32)
    licence = "0199d5a2-7c3e-7b41-9f0a-1c2d3e4f5a6b"
    code = "subject_tenant_mismatch_for_licence"
    caplog.set_level("WARNING", logger="w2v_orchestrator.tenant_licence")
    exc = LicensesClientError(
        f'get_active_license returned 403: {{"error":"{code}","licence_id":"{licence}",'
        f'"path":"/licenses/{licence}","t":"{token}","glued":"{token}/{licence}"}}',
        status_code=403,
    )
    err = tl.licences_error(exc)
    assert isinstance(err, tl.LicencesRefusedError)
    logged = "\n".join(r.getMessage() for r in caplog.records)
    for text in (err.excerpt, logged):
        assert token not in text, text
        assert code in text, text
        assert f'"licence_id":"{licence}"' in text, text
        assert f'"path":"/licenses/{licence}"' in text, "a UUID inside a long run is kept"
        assert f'"glued":"<redacted>{licence}"' in text, text
    # the boundary: 39 survives, 40 does not
    kept, cut = "a" * 39, "b" * 40
    edge = tl.licences_error(LicensesClientError(f"x 400: {kept} {cut}", status_code=400))
    assert isinstance(edge, tl.LicencesRefusedError)
    assert kept in edge.excerpt and cut not in edge.excerpt, edge.excerpt
