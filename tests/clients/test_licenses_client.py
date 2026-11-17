"""LicensesClient — mint + token surface (Phase C-2)."""

from __future__ import annotations

import json
from uuid import uuid4

import httpx
import pytest

from src.clients.licenses_client import (
    LicencesAccessRefusedError,
    LicencesTenantNameTakenError,
    LicencesTenantOwnedByOtherAppError,
    LicensesClient,
    LicensesClientError,
    RegisteredTenant,
    TenantNotOnboardedError,
)

pytestmark = pytest.mark.asyncio


class _Auth:
    async def token(self) -> str:
        return "tok"


def _lic(handler) -> LicensesClient:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return LicensesClient(authenticator=_Auth(), base_url="http://licenses:8544", client=client)


async def test_list_apps_sends_bearer_and_tenant() -> None:
    tenant = uuid4()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/apps"
        assert request.headers["authorization"] == "Bearer tok"
        assert request.headers["x-tenant-id"] == str(tenant)
        return httpx.Response(200, json=[{"id": "a1", "key": "w2v", "name": "W2V"}])

    apps = await _lic(handler).list_apps(tenant_id=tenant)
    assert apps[0].key == "w2v"


async def test_create_license_posts_to_subject() -> None:
    tenant, subject = uuid4(), uuid4()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == f"/licenses/user/{subject}"
        assert request.headers["x-tenant-id"] == str(tenant)
        return httpx.Response(201, json={"id": "lic-9"})

    lic = await _lic(handler).create_license(
        subject_id=subject, app_id="a1", plan_id="p1", tenant_id=tenant
    )
    assert lic.id == "lic-9"


async def test_get_active_license_200_returns_detail() -> None:
    tenant = uuid4()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/licenses/active"
        assert request.url.params["tenant_id"] == str(tenant)
        assert request.url.params["app_key"] == "w2v"
        assert request.headers["x-tenant-id"] == str(tenant)
        return httpx.Response(
            200, json={"status": "active", "expires_at": "2027-01-01T00:00:00+00:00"}
        )

    detail = await _lic(handler).get_active_license(tenant_id=tenant, app_key="w2v")
    assert detail is not None and detail.status == "active"


async def test_get_active_license_404_returns_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "no_active_license"})

    assert await _lic(handler).get_active_license(tenant_id=uuid4(), app_key="w2v") is None


async def test_get_active_license_409_returns_conflict() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"error": "multiple_active_licenses"})

    detail = await _lic(handler).get_active_license(tenant_id=uuid4(), app_key="w2v")
    assert detail is not None and detail.conflict is True


async def test_get_active_license_500_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with pytest.raises(LicensesClientError):
        await _lic(handler).get_active_license(tenant_id=uuid4(), app_key="w2v")


async def test_list_user_licenses_gets_subject_path() -> None:
    tenant, subject = uuid4(), uuid4()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == f"/licenses/user/{subject}"
        assert request.headers["x-tenant-id"] == str(tenant)
        return httpx.Response(200, json=[{"id": "lic-1", "app_id": "a1", "revoked": False}])

    rows = await _lic(handler).list_user_licenses(subject, tenant_id=tenant)
    assert rows[0].id == "lic-1"


async def test_list_user_licenses_raises_on_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with pytest.raises(LicensesClientError):
        await _lic(handler).list_user_licenses(uuid4(), tenant_id=uuid4())


async def test_regenerate_token_returns_raw() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/licenses/lic-9/regenerate-token"
        return httpx.Response(200, json={"token": "RAW-TOKEN", "message": "ok"})

    assert await _lic(handler).regenerate_token("lic-9", tenant_id=uuid4()) == "RAW-TOKEN"


async def test_regenerate_token_missing_token_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": "ok"})

    with pytest.raises(LicensesClientError, match="missing token"):
        await _lic(handler).regenerate_token("lic-9", tenant_id=uuid4())


async def test_non_200_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="forbidden")

    with pytest.raises(LicensesClientError, match="403"):
        await _lic(handler).list_apps(tenant_id=uuid4())


async def test_introspect_posts_the_token_without_service_credentials() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "active": True,
                "tenant_id": "t1",
                "license_id": "l1",
                "app": {"key": "w2v"},
                "token_expires_at": "2027-01-01T00:00:00+00:00",
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    lic = LicensesClient(
        authenticator=_Auth(),
        base_url="http://licenses:8544",
        client=client,
        introspect_url="https://licenses.test/license/introspect",
    )
    view = await lic.introspect("lic_tok_secret")
    assert view.active and view.license_id == "l1" and view.token_expires_at
    [req] = seen
    assert str(req.url) == "https://licenses.test/license/introspect"
    assert req.content == b'{"token":"lic_tok_secret"}'
    assert "authorization" not in req.headers, "the token in the body is the credential"


async def test_introspect_defaults_to_the_w2v_common_url(monkeypatch) -> None:
    import w2v_common.auth.middleware as mw

    monkeypatch.setattr(mw, "LICENSE_INTROSPECT_URL", "http://gw.test/license/introspect")
    urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(200, json={"active": False, "reason": "token_not_found"})

    view = await _lic(handler).introspect("t")
    assert urls == ["http://gw.test/license/introspect"]
    assert not view.active and view.reason == "token_not_found"


@pytest.mark.parametrize("status", [400, 401, 500, 503])
async def test_introspect_non_200_raises_with_status_and_no_token(status: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        # An upstream that ECHOES the request body (the token) in its error.
        return httpx.Response(status, text=f"missing_app_key {request.content.decode()}")

    with pytest.raises(LicensesClientError) as info:
        await _lic(handler).introspect("lic_tok_secret")
    assert info.value.status_code == status
    assert "lic_tok_secret" not in str(info.value)
    assert str(info.value).startswith(f"introspect returned {status}")


async def test_create_token_posts_an_api_token_and_keeps_its_expiry() -> None:
    tenant = uuid4()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST" and request.url.path == "/licenses/l1/tokens"
        assert request.url.params["token_type"] == "api"
        assert request.headers["x-tenant-id"] == str(tenant)
        return httpx.Response(201, json={"token": "raw", "expires_at": "2027-10-13T00:00:00Z"})

    minted = await _lic(handler).create_token("l1", tenant_id=tenant)
    assert minted.token == "raw" and minted.expires_at == "2027-10-13T00:00:00Z"
    assert "raw" not in repr(minted)


async def test_create_token_refusal_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"detail": {"error": "license_not_active"}})

    with pytest.raises(LicensesClientError, match="409"):
        await _lic(handler).create_token("l1", tenant_id=uuid4())


# ─── licences migration 050: tenants owned per app ──────────────────────


def _refusal(code: str) -> dict[str, object]:
    return {"detail": {"error": code, "message": "refused"}}


@pytest.mark.parametrize(("status", "created"), [(201, True), (200, False)])
async def test_register_tenant_puts_the_name_without_a_tenant_header(
    status: int, created: bool
) -> None:
    tenant = uuid4()
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, json={"id": str(tenant), "name": "stored-name"})

    out = await _lic(handler).register_tenant(tenant, name="acme")

    assert out == RegisteredTenant(tenant_id=str(tenant), name="stored-name", created=created)
    (req,) = seen
    assert req.method == "PUT"
    assert req.url.path == f"/tenants/{tenant}"
    assert req.headers["authorization"] == "Bearer tok"
    # The owner is derived by licences from the SA — never sent, no retarget header.
    assert "x-tenant-id" not in req.headers
    assert req.read() == b'{"name":"acme"}'


@pytest.mark.parametrize(
    ("status", "body", "exc_type", "code"),
    [
        (
            409,
            _refusal("tenant_owned_by_other_app"),
            LicencesTenantOwnedByOtherAppError,
            "tenant_owned_by_other_app",
        ),
        (409, _refusal("tenant_name_taken"), LicencesTenantNameTakenError, "tenant_name_taken"),
        (403, _refusal("unknown_api_client"), LicencesAccessRefusedError, "unknown_api_client"),
        (403, {"detail": "Role 'license-admin' required"}, LicensesClientError, None),
        (409, {"detail": "name already belongs to tenant x"}, LicensesClientError, None),
    ],
)
async def test_register_tenant_refusals_are_typed(
    status: int, body: object, exc_type: type[LicensesClientError], code: str | None
) -> None:
    lic = _lic(lambda _r: httpx.Response(status, json=body))
    with pytest.raises(LicensesClientError) as info:
        await lic.register_tenant(uuid4(), name="acme")
    assert type(info.value) is exc_type
    assert info.value.status_code == status
    assert info.value.code == code


async def test_tenant_not_accessible_is_a_typed_not_onboarded_error() -> None:
    lic = _lic(lambda _r: httpx.Response(403, json=_refusal("tenant_not_accessible")))
    with pytest.raises(TenantNotOnboardedError) as info:
        await lic.get_active_license(tenant_id=uuid4(), app_key="w2v")
    assert info.value.status_code == 403
    assert info.value.code == "tenant_not_accessible"


@pytest.mark.parametrize("code", ["app_not_issuable_by_caller", "unknown_api_client"])
async def test_mint_ownership_refusals_are_typed_access_errors(code: str) -> None:
    lic = _lic(lambda _r: httpx.Response(403, json=_refusal(code)))
    with pytest.raises(LicencesAccessRefusedError) as info:
        await lic.create_license(subject_id=uuid4(), app_id="a1", plan_id="p1", tenant_id=uuid4())
    assert info.value.code == code
    # The long-standing message shape is kept (callers log it; tests match it).
    assert str(info.value).startswith("create_license returned 403:")


async def test_a_plain_403_stays_a_generic_authz_error() -> None:
    """A role refusal without a licences code is NOT "not onboarded": the gate's
    misconfiguration path (inventory fallback + loud log) must still see it."""
    lic = _lic(lambda _r: httpx.Response(403, json={"detail": "Role required"}))
    with pytest.raises(LicensesClientError) as info:
        await lic.get_active_license(tenant_id=uuid4(), app_key="w2v")
    assert type(info.value) is LicensesClientError
    assert info.value.status_code == 403


# ─── tenant-level licences: POST /internal/tenants/{id}/bundles/{bundle}/ensure ──


def _ensure_body(tenant: object, *, refused: list[dict[str, str]] | None = None) -> dict:
    return {
        "tenant_id": str(tenant),
        "bundle": "w2v",
        "budget_currency": None,
        "apps": {
            "w2v": {
                "decision": "insert",
                "license_id": "lic-w2v",
                "plan": "standard",
                "metering": "token_and_quota",
            }
        },
        "refused": refused or [],
    }


async def test_ensure_tenant_bundle_posts_the_plans_to_the_bundle_route() -> None:
    tenant = uuid4()
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["tenant"] = request.headers["x-tenant-id"]
        seen["body"] = request.read()
        return httpx.Response(200, json=_ensure_body(tenant))

    out = await _lic(handler).ensure_tenant_bundle(tenant, "w2v", plans=["p2", "standard"])
    assert seen["method"] == "POST"
    assert seen["path"] == f"/internal/tenants/{tenant}/bundles/w2v/ensure"
    assert seen["tenant"] == str(tenant)
    assert seen["body"] == b'{"plans":["p2","standard"]}'
    assert out.apps["w2v"].license_id == "lic-w2v"
    assert out.refused == []


async def test_ensure_tenant_bundle_409_with_refusals_is_a_result_not_an_error() -> None:
    tenant = uuid4()
    body = _ensure_body(tenant, refused=[{"app": "w2v-image", "reason": "member_licence"}])
    out = await _lic(lambda _r: httpx.Response(409, json=body)).ensure_tenant_bundle(
        tenant, "w2v", plans=["standard"]
    )
    assert [(r.app, r.reason) for r in out.refused] == [("w2v-image", "member_licence")]
    assert out.apps["w2v"].license_id == "lic-w2v"


@pytest.mark.parametrize(
    ("status", "body", "exc_type"),
    [
        (409, {"detail": {"error": "tenant_archived", "message": "x"}}, LicensesClientError),
        (403, _refusal("tenant_owner_only"), LicencesAccessRefusedError),
        (404, {"detail": {"error": "tenant_not_found", "message": "x"}}, LicensesClientError),
    ],
)
async def test_ensure_tenant_bundle_whole_call_refusals_raise(
    status: int, body: object, exc_type: type[LicensesClientError]
) -> None:
    lic = _lic(lambda _r: httpx.Response(status, json=body))
    with pytest.raises(LicensesClientError) as info:
        await lic.ensure_tenant_bundle(uuid4(), "w2v", plans=["standard"])
    assert type(info.value) is exc_type
    assert info.value.status_code == status


# ─── the ensure's own timeout (live 2026-10-15) ───────────────────────────────
# licences #275: the ensure also provisions dependency licences, 15.8 s live —
# past the 15 s budget every licences call shared, so the client gave up on a
# request licences completed. The ensure gets its own budget; nothing else does.


def _read_timeout(request: httpx.Request) -> float:
    return request.extensions["timeout"]["read"]


async def test_the_ensure_uses_its_own_long_timeout_and_other_calls_do_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production path (no injected client): one AsyncClient per call."""
    import src.clients.licenses_client as mod

    seen: dict[str, float] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen[request.url.path] = _read_timeout(request)
        if request.url.path.endswith("/ensure"):
            return httpx.Response(200, json=_ensure_body(tenant))
        return httpx.Response(200, json=[])

    real = httpx.AsyncClient

    def _client(*, timeout: float) -> httpx.AsyncClient:
        return real(transport=httpx.MockTransport(handler), timeout=timeout)

    monkeypatch.setattr(mod.httpx, "AsyncClient", _client)
    tenant = uuid4()
    lic = LicensesClient(authenticator=_Auth(), base_url="http://licenses:8544")

    await lic.ensure_tenant_bundle(tenant, "w2v", plans=["standard"])
    await lic.list_apps(tenant_id=tenant)
    await lic.list_user_licenses(f"svc:{tenant}:w2v", tenant_id=tenant)

    assert mod.ENSURE_TIMEOUT_SECONDS == 90.0
    assert seen[f"/internal/tenants/{tenant}/bundles/w2v/ensure"] == 90.0
    assert seen["/apps"] == 15.0
    assert seen[f"/licenses/user/svc:{tenant}:w2v"] == 15.0


async def test_the_ensure_timeout_is_configurable_and_overrides_an_injected_client() -> None:
    tenant = uuid4()
    seen: dict[str, float] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen[request.url.path] = _read_timeout(request)
        if request.url.path.endswith("/ensure"):
            return httpx.Response(200, json=_ensure_body(tenant))
        return httpx.Response(200, json=[])

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=7.0)
    lic = LicensesClient(
        authenticator=_Auth(),
        base_url="http://licenses:8544",
        client=client,
        ensure_timeout_seconds=42.0,
    )
    await lic.ensure_tenant_bundle(tenant, "w2v", plans=["standard"])
    await lic.list_apps(tenant_id=tenant)

    assert seen[f"/internal/tenants/{tenant}/bundles/w2v/ensure"] == 42.0
    assert seen["/apps"] == 7.0  # the injected client's own budget, untouched


async def test_the_factory_wires_the_ensure_timeout_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.api.routes import _common
    from src.config import get_settings

    monkeypatch.setenv("W2V_LICENCES_ENSURE_TIMEOUT_SECONDS", "123")
    get_settings.cache_clear()
    try:
        lic = _common.get_licenses_client()
    finally:
        monkeypatch.delenv("W2V_LICENCES_ENSURE_TIMEOUT_SECONDS")
        get_settings.cache_clear()
    assert lic._ensure_timeout == 123.0
    assert get_settings().licences_ensure_timeout_seconds == 90.0


# ─── early-access removal: list / revoke / archive ───────────────────────


async def test_list_tenant_licenses_sends_the_tenant_header() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET" and request.url.path == "/licenses"
        assert request.headers["x-tenant-id"] == "t-1"
        return httpx.Response(
            200, json=[{"id": "l1", "status": "active"}, {"id": "l2", "revoked_at": "x"}]
        )

    rows = await _lic(handler).list_tenant_licenses(tenant_id="t-1")
    assert [(r.id, r.revoked_at) for r in rows] == [("l1", None), ("l2", "x")]
    with pytest.raises(LicensesClientError) as exc:
        await _lic(lambda r: httpx.Response(403, json={})).list_tenant_licenses(tenant_id="t")
    assert exc.value.status_code == 403


async def test_revoke_license_posts_the_id_and_refuses_a_false_answer() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"revoked": True, "message": "ok"})

    await _lic(handler).revoke_license("l1", tenant_id="t-1")
    [req] = seen
    assert req.method == "POST" and req.url.path == "/license/revoke"
    assert req.headers["x-tenant-id"] == "t-1"
    assert json.loads(req.content) == {"license_id": "l1"}
    # licences' "License not found" (unknown, or another tenant's) is NOT a revoke.
    with pytest.raises(LicensesClientError) as exc:
        await _lic(
            lambda r: httpx.Response(200, json={"revoked": False, "message": "License not found"})
        ).revoke_license("l1", tenant_id="t")
    assert exc.value.code == "licence_not_revoked"
    with pytest.raises(LicensesClientError) as exc:
        await _lic(lambda r: httpx.Response(403, json={})).revoke_license("l1", tenant_id="t")
    assert exc.value.status_code == 403


async def test_revoke_license_returns_released_dependencies_and_the_409_detail() -> None:
    body = {"revoked": True, "released_dependency_ids": ["d1", "d2"]}
    lic = _lic(lambda r: httpx.Response(200, json=body))
    assert await lic.revoke_license("l1", tenant_id="t") == ["d1", "d2"]
    # an older licences names none
    lic = _lic(lambda r: httpx.Response(200, json={"revoked": True}))
    assert await lic.revoke_license("l1", tenant_id="t") == []
    detail = {
        "error": "dependency_still_required",
        "message": "x",
        "required_by": [{"license_id": "l9", "tenant_id": "t", "app_key": "kag"}],
    }
    with pytest.raises(LicensesClientError) as exc:
        await _lic(lambda r: httpx.Response(409, json={"detail": detail})).revoke_license(
            "d1", tenant_id="t"
        )
    assert exc.value.code == "dependency_still_required"
    assert exc.value.detail == detail


async def test_list_tenant_licenses_reads_is_dependency() -> None:
    rows = await _lic(
        lambda r: httpx.Response(
            200,
            json=[
                {"id": "d1", "is_dependency": True, "metadata": {"dependency_of": "w2v"}},
                {"id": "l1"},
            ],
        )
    ).list_tenant_licenses(tenant_id="t")
    assert [(r.id, r.is_dependency) for r in rows] == [("d1", True), ("l1", None)]


async def test_archive_tenant_patches_archived_true() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"id": "t-1", "archived_at": "2026-10-21T00:00:00Z"})

    assert await _lic(handler).archive_tenant("t-1") == "2026-10-21T00:00:00Z"
    [req] = seen
    assert req.method == "PATCH" and req.url.path == "/tenants/t-1/archive"
    assert json.loads(req.content) == {"archived": True}
    with pytest.raises(LicensesClientError):
        await _lic(lambda r: httpx.Response(404, json={})).archive_tenant("t-1")
