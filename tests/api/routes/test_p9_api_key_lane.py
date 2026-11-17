"""P9 — a tenant API key on the REAL request path: billable, and never forwarded.

Production stack as in ``test_tenancy_api_keys.py``: w2v_common's
``AuthMiddleware`` (``create_app``) with the registered ``w2v_sk_`` verifier and
the orchestrator's ``TenancyMiddleware``; real ``tenants`` / ``tenant_api_keys`` /
``projects`` rows. Only analysis (an ``httpx`` MockTransport on ``_proxy``) and the
licences service (an entitled stub) are fake.

* P9-6: the key lane binds ``license_id = tenants.licence_ref`` — the run snapshot
  ``build_config_snapshot`` builds carries it, and so does the outbound
  ``X-License-Id``; a tenant WITHOUT a ref is refused 503 ``license_unavailable``
  by the licence gate (never unattributable spend).
* P9-7: ``/api/brand/*`` + ``GET /api/critique/compare`` proxy to analysis; the
  compare query string reaches analysis; the ``w2v_sk_`` key NEVER leaves the
  orchestrator (outbound auth = the service-account licence + delegated tenant).

Non-vacuity: the snapshot test fails on ``main`` (the lane bound ``license_id=None``
→ ``""``), the no-ref test fails without ``_require_attributable_api_key``, and the
leak test inspects EVERY outbound header value, not just ``Authorization``.
"""

from __future__ import annotations

import ast
import datetime
import json
import types
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
from fastapi import APIRouter, Depends, FastAPI
from fastapi.routing import APIRoute
from httpx import ASGITransport, AsyncClient
from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from w2v_common.asgi import create_app
from w2v_common.auth import keycloak as keycloak_mod
from w2v_common.auth import register_api_key_verifier
from w2v_common.auth.identity_context import get_license_id_from_context

from src.api.routes import _common
from src.api.routes._common import TENANT_GATE, SubjectIdDep, TenantIdDep, require_license
from src.api.routes.brand import router as brand_router
from src.api.routes.critique import router as critique_router
from src.api.routes.tenancy.errors import register_tenancy_exception_handlers
from src.auth.tenancy_middleware import install_tenancy_middleware, make_api_key_verifier
from src.config import clear_settings_cache, get_settings
from src.database.models import Tenant
from src.database.repositories.api_keys_repository import ApiKeysRepository
from src.database.repositories.project_repository import ProjectRepository
from src.middleware.proxy_body_limit import PROXY_BODY_LIMIT_BYTES, ProxyBodyLimitMiddleware
from src.observability.metrics import api_key_unattributable_total
from src.services.runs.run_setup import build_config_snapshot
from src.services.tenant_licence import seal_licence_token
from tests.api.routes._p9_licences import TEST_KEYRING, P9Licences, override_licence_keyring

pytestmark = pytest.mark.asyncio

_SA_TOKEN = "test-downstream-license-token"  # conftest's DOWNSTREAM_LICENSE_TOKEN
_PUBLIC_PEER = ("203.0.113.9", 50000)


@dataclass
class Harness:
    app: FastAPI
    factory: async_sessionmaker[AsyncSession]
    sent: list[dict[str, Any]]
    timeouts: list[httpx.Timeout]
    licences: P9Licences

    async def call(self, method: str, path: str, key: str, json_body: Any = None) -> httpx.Response:
        transport = ASGITransport(app=self.app, client=_PUBLIC_PEER)
        async with AsyncClient(transport=transport, base_url="http://testserver") as c:
            return await c.request(
                method, path, json=json_body, headers={"Authorization": f"Bearer {key}"}
            )

    async def tenant_with_key(
        self,
        licence_ref: uuid.UUID | None,
        *,
        live: bool = True,
        token: Callable[[uuid.UUID], str] | None = None,
        sealed_for_other_tenant: bool = False,
    ) -> tuple[uuid.UUID, str]:
        """A tenant + key; ``licence_ref`` is registered as the tenant's LIVE w2v
        licence with the licences fake unless ``live=False``. ``token(tid)`` mints
        the raw licence token stored (sealed) on the tenant row."""
        tid = uuid.uuid7()
        if licence_ref is not None and live:
            self.licences.live[tid] = [licence_ref]
        values: dict[str, Any] = {
            "id": tid,
            "name": f"t-{tid}",
            "status": "active",
            "licence_ref": licence_ref,
        }
        if token is not None:
            sealed = seal_licence_token(
                TEST_KEYRING, uuid.uuid7() if sealed_for_other_tenant else tid, token(tid)
            )
            values |= {
                "licence_token_ciphertext": sealed.ciphertext,
                "licence_token_kid": sealed.key_id,
                "licence_token_set_at": datetime.datetime.now(datetime.UTC),
            }
        async with self.factory() as s:
            await s.execute(insert(Tenant).values(**values))
            _row, key = await ApiKeysRepository(s).create(
                tenant_id=tid, name="merqi-marketing", created_by=None
            )
            await s.commit()
        return tid, key


def _analysis_stub(monkeypatch, sent: list[dict[str, Any]], timeouts: list[httpx.Timeout]) -> None:
    """Every request the proxy sends to analysis is recorded and answered 200."""

    def _handler(request: httpx.Request) -> httpx.Response:
        body = request.content.decode() if request.content else ""
        sent.append(
            {
                "path": request.url.path,
                "params": dict(request.url.params),
                "headers": dict(request.headers),
                "json": json.loads(body) if body else None,
            }
        )
        return httpx.Response(200, json={"ok": True})

    real = httpx.AsyncClient

    class _Stub:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.timeout: httpx.Timeout = kwargs["timeout"]
            self._inner = real(transport=httpx.MockTransport(_handler), timeout=self.timeout)

        async def __aenter__(self) -> _Stub:
            await self._inner.__aenter__()
            return self

        async def __aexit__(self, *exc: Any) -> None:
            await self._inner.__aexit__(*exc)

        async def get(self, url, *, params=None, headers=None):  # type: ignore[no-untyped-def]
            timeouts.append(self.timeout)
            return await self._inner.get(url, params=params, headers=headers)

        async def post(self, url, *, json=None, headers=None):  # type: ignore[no-untyped-def]
            timeouts.append(self.timeout)
            return await self._inner.post(url, json=json, headers=headers)

    import src.api.routes._proxy as proxy_mod

    monkeypatch.setattr(proxy_mod.httpx, "AsyncClient", _Stub)


@pytest.fixture
def harness(monkeypatch, db_session_factory) -> Iterator[Harness]:
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("KEYCLOAK_URL", "https://keycloak.test")
    monkeypatch.setenv("KEYCLOAK_REALM", "portal")
    monkeypatch.setenv("KEYCLOAK_AUDIENCES", "w2v-frontend")
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    monkeypatch.setenv("W2V_TENANCY_SOURCE", "membership")
    monkeypatch.setenv("AUTH_JWT_TENANT_REQUIRED", "false")
    clear_settings_cache()
    # No JWT in these tests; JWKS is never fetched. Keep it offline regardless.
    monkeypatch.setattr(
        keycloak_mod,
        "httpx",
        types.SimpleNamespace(
            AsyncClient=lambda **kw: httpx.AsyncClient(
                transport=httpx.MockTransport(lambda r: httpx.Response(404)), **kw
            ),
            HTTPError=httpx.HTTPError,
        ),
    )
    keycloak_mod.reset_jwt_validator()
    register_api_key_verifier(make_api_key_verifier(db_session_factory))

    probe = APIRouter(prefix="/api/probe", dependencies=[Depends(require_license)])

    @probe.post("/snapshot")
    async def snapshot(tenant_id: TenantIdDep, subject: SubjectIdDep) -> dict[str, Any]:
        """What a run started by this caller would persist (``RunEngine.start_run``
        → ``build_config_snapshot``), plus the ambient licence binding."""
        async with db_session_factory() as s:
            proj = await ProjectRepository(s).create_with_deliverables(
                tenant_id=tenant_id,
                owner_subject_id=subject,
                type_="promo",
                title="p",
                brief={},
                brand={},
                voice={},
                theme={},
                visibility="tenant",
                deliverables=[{"name": "d", "format": "16x9", "captions_mode": "none"}],
            )
            await s.commit()
            snap = await build_config_snapshot(
                s,
                proj=proj,
                deliv=proj.deliverables[0],
                tenant_id=tenant_id,
                subject_id=subject,
                config_overrides=None,
            )
        return {"snapshot_license_id": snap["license_id"], "ctx": get_license_id_from_context()}

    app = create_app(get_settings(), routers=[probe], cors_enabled=False)
    # Mounted exactly as src/asgi.py mounts them: behind the full tenant gate.
    app.include_router(brand_router, dependencies=TENANT_GATE)
    app.include_router(critique_router, dependencies=TENANT_GATE)
    install_tenancy_middleware(app)
    register_tenancy_exception_handlers(app)  # as src/asgi.py does
    app.state.session_factory = db_session_factory
    licences = P9Licences()
    app.dependency_overrides[_common.get_licenses_client] = lambda: licences
    override_licence_keyring(app)
    app.add_middleware(ProxyBodyLimitMiddleware)
    _common.clear_api_key_licence_cache()
    sent: list[dict[str, Any]] = []
    timeouts: list[httpx.Timeout] = []
    _analysis_stub(monkeypatch, sent, timeouts)
    yield Harness(
        app=app, factory=db_session_factory, sent=sent, timeouts=timeouts, licences=licences
    )
    _common.clear_api_key_licence_cache()

    register_api_key_verifier(None)
    keycloak_mod.reset_jwt_validator()
    clear_settings_cache()


# ─── P9-6: billable ───────────────────────────────────────────────────────


async def test_api_key_run_snapshot_carries_the_tenant_licence(harness: Harness) -> None:
    licence = uuid.uuid7()
    _tid, key = await harness.tenant_with_key(licence)
    r = await harness.call("POST", "/api/probe/snapshot", key)
    assert r.status_code == 200, r.text
    assert r.json() == {"snapshot_license_id": str(licence), "ctx": str(licence)}


async def test_api_key_of_a_tenant_without_licence_ref_is_refused(harness: Harness) -> None:
    tid, key = await harness.tenant_with_key(None)
    harness.licences.live[tid] = [uuid.uuid7()]  # entitled, but no ref recorded
    r = await harness.call("POST", "/api/probe/snapshot", key)
    assert r.status_code == 503, r.text
    assert "license_unavailable" in r.text
    assert r.headers.get("retry-after") == "60"
    # …and nothing went downstream either.
    r = await harness.call("POST", "/api/brand/icp-from-url", key, {"url": "https://a.test"})
    assert r.status_code == 503, r.text
    assert harness.sent == []


# ─── P9-7: proxied, key never forwarded ───────────────────────────────────

_ROUTES: list[tuple[str, str, dict | None, str]] = [
    ("POST", "/api/brand/from-url", {"url": "https://acme.test"}, "/api/brand/from-url"),
    ("POST", "/api/brand/extract", {"analysis": {"x": 1}}, "/api/brand/extract"),
    ("POST", "/api/brand/icp-from-url", {"url": "https://acme.test"}, "/api/brand/icp-from-url"),
    (
        "GET",
        "/api/critique/compare?job_id_1=job-a&job_id_2=job-b",
        None,
        "/api/critique/compare",
    ),
]


@pytest.mark.parametrize(
    ("method", "path", "body", "upstream"),
    [("POST", "/api/brand/from-url", {"url": "https://acme.test"}, "/api/brand/from-url")],
    ids=lambda v: str(v),
)
async def test_key_caller_is_proxied_without_the_key(
    harness: Harness, method: str, path: str, body: dict | None, upstream: str
) -> None:
    licence = uuid.uuid7()
    tid, key = await harness.tenant_with_key(licence)
    r = await harness.call(method, path, key, body)
    assert r.status_code == 200, r.text
    [sent] = harness.sent
    assert sent["path"] == upstream
    assert sent["json"] == body, "body forwarded verbatim"
    headers = sent["headers"]
    leaked = {k: v for k, v in headers.items() if "w2v_sk_" in v or key in v}
    assert not leaked, f"tenant API key forwarded downstream: {sorted(leaked)}"
    assert headers["authorization"] == f"Bearer {_SA_TOKEN}"
    assert headers["x-tenant-id"] == f"Bearer {_SA_TOKEN}"
    assert headers["x-effective-tenant-id"] == str(tid)
    assert headers["x-license-id"] == str(licence), "downstream spend bills the tenant licence"


async def test_compare_forwards_its_query_and_is_not_swallowed_by_job_id(
    harness: Harness,
) -> None:
    _tid, key = await harness.tenant_with_key(uuid.uuid7())
    r = await harness.call("GET", "/api/critique/compare?job_id_1=a1&job_id_2=b2", key)
    assert r.status_code == 200, r.text
    [sent] = harness.sent
    assert sent["path"] == "/api/critique/compare"
    assert sent["params"] == {"job_id_1": "a1", "job_id_2": "b2"}
    missing = await harness.call("GET", "/api/critique/compare?job_id_1=a1", key)
    assert missing.status_code == 422, "both ids are required — never proxied as a job id"
    assert len(harness.sent) == 1


async def test_icp_and_extract_get_the_slow_timeout(harness: Harness) -> None:
    _tid, key = await harness.tenant_with_key(uuid.uuid7())
    for path in ("/api/brand/icp-from-url", "/api/brand/extract", "/api/brand/from-url"):
        assert (await harness.call("POST", path, key, {"url": "https://a.test"})).status_code == 200
    assert [t.read for t in harness.timeouts] == [120.0, 120.0, 60.0]


async def test_compare_is_declared_before_the_job_id_catch_all() -> None:
    paths = [r.path for r in critique_router.routes if isinstance(r, APIRoute)]
    assert paths.index("/api/critique/compare") < paths.index("/api/critique/{job_id}")


# ─── review fix 2: only a LIVE licence of the tenant is ever billed ────────
# Verified by INTROSPECTING the tenant's stored token, else GET /licenses/active.


def _refused(r: httpx.Response, status_code: int, code: str, state: str | None = None) -> None:
    assert r.status_code == status_code, r.text
    assert code in r.text, r.text
    if state is not None:
        assert f"{state}" in r.text, r.text


@pytest.mark.parametrize(
    ("case", "status_code", "code", "state"),
    [("token-other-tenant", 402, "licence_inactive", "tenant")],
)
async def test_api_key_whose_licence_ref_is_not_live_is_never_billed(
    harness: Harness, case: str, status_code: int, code: str, state: str | None
) -> None:
    ref = uuid.uuid7()
    lic = harness.licences
    token: Callable[[uuid.UUID], str] | None = {
        "token-other-tenant": lambda tid: lic.token(uuid.uuid7(), ref),
        "token-other-app": lambda tid: lic.token(tid, ref, app="fr"),
        "token-inactive": lambda tid: lic.token(tid, ref, active=False, reason="license_revoked"),
        "token-other-licence": lambda tid: lic.token(tid, uuid.uuid7()),
        "token-unreadable": lambda tid: lic.token(tid, ref),
        "introspect-down": lambda tid: lic.token(tid, ref),
        "introspect-refused": lambda tid: lic.token(tid, ref),
    }.get(case)
    tid, key = await harness.tenant_with_key(
        ref, token=token, sealed_for_other_tenant=case == "token-unreadable"
    )
    if case == "introspect-down":
        lic.introspect_down = True
    elif case == "fallback-stale-ref":
        lic.live[tid] = [uuid.uuid7()]  # ONE live licence, and it is not the ref
    elif case == "fallback-not-onboarded":
        lic.not_onboarded.add(tid)
    elif case == "introspect-refused":
        lic.introspect_refuse = 400
    r = await harness.call("POST", "/api/brand/icp-from-url", key, {"url": "https://a.test"})
    _refused(r, status_code, code, state)
    if case == "introspect-down":
        assert r.headers.get("retry-after") == "60"
    else:
        assert "retry-after" not in r.headers
    assert harness.sent == [], "nothing was spent on a licence that is not live"
    snap = await harness.call("POST", "/api/probe/snapshot", key)
    assert snap.status_code == status_code, snap.text


async def test_api_key_with_a_valid_stored_token_is_billed_without_the_fallback(
    harness: Harness,
) -> None:
    ref = uuid.uuid7()
    _tid, key = await harness.tenant_with_key(
        ref, token=lambda tid: harness.licences.token(tid, ref)
    )
    r = await harness.call("POST", "/api/probe/snapshot", key)
    assert r.status_code == 200, r.text
    assert r.json()["snapshot_license_id"] == str(ref)
    assert harness.licences.introspect_calls == 1


@pytest.mark.parametrize(
    "case",
    ["ref-revoked-two-others-live"],
)
async def test_fallback_that_cannot_single_out_the_ref_fails_closed(
    harness: Harness, case: str
) -> None:
    """With >1 active licences ``/licenses/active`` returns a PICK, so the ref
    cannot be proven live and the tenant's own — even when it happens to be one
    of the candidates. Fail closed like a tenant with no ref (P9-6): the same 503
    ``license_unavailable`` envelope + Retry-After, nothing spent, no X-License-Id
    — but its own state / metric reason ``ref_ambiguous``, so an operator can tell
    "store the token" apart from "no ref yet". The first case is the review's
    revoked L1 with two other live licences."""
    ref = uuid.uuid7()
    tid, key = await harness.tenant_with_key(ref)
    if case == "ref-revoked-two-others-live":
        harness.licences.live[tid] = [uuid.uuid7(), uuid.uuid7()]
    elif case == "ref-live-among-several":
        harness.licences.live[tid] = [uuid.uuid7(), ref]
    else:
        harness.licences.conflict.add(tid)
    # The no-ref tenant's refusal is the reference shape.
    no_ref_tid, no_ref_key = await harness.tenant_with_key(None)
    harness.licences.live[no_ref_tid] = [uuid.uuid7()]
    counted = api_key_unattributable_total.labels
    no_ref0, amb0 = counted("no_ref")._value.get(), counted("ref_ambiguous")._value.get()
    no_ref = await harness.call("POST", "/api/probe/snapshot", no_ref_key)
    assert no_ref.status_code == 503, no_ref.text
    assert counted("no_ref")._value.get() == no_ref0 + 1

    def detail_of(r: httpx.Response) -> dict[str, Any]:
        # the w2v_common envelope stringifies the detail
        return ast.literal_eval(r.json()["error"]["message"])

    def shape(r: httpx.Response) -> tuple[Any, ...]:
        err, detail = r.json()["error"], detail_of(r)
        return err["code"], err["status"], detail["error"], sorted(detail)

    assert detail_of(no_ref)["state"] == "unattributable"

    for path, body in (
        ("/api/brand/icp-from-url", {"url": "https://a.test"}),
        ("/api/probe/snapshot", None),
    ):
        r = await harness.call("POST", path, key, body)
        assert r.status_code == 503, r.text
        assert shape(r) == shape(no_ref), (r.text, no_ref.text)
        assert detail_of(r)["error"] == "license_unavailable", r.text
        assert detail_of(r)["state"] == "ref_ambiguous", r.text
        assert r.headers.get("retry-after") == "60"
    assert harness.sent == [], "nothing billed to a ref nobody verified"
    assert counted("ref_ambiguous")._value.get() == amb0 + 2
    assert counted("no_ref")._value.get() == no_ref0 + 1, "ambiguity never counts as no_ref"


async def test_fallback_with_several_licences_is_admitted_when_the_stored_token_names_the_ref(
    harness: Harness,
) -> None:
    ref = uuid.uuid7()
    tid, key = await harness.tenant_with_key(
        ref, token=lambda tid: harness.licences.token(tid, ref)
    )
    harness.licences.live[tid] = [uuid.uuid7(), ref]
    r = await harness.call("POST", "/api/brand/icp-from-url", key, {"url": "https://a.test"})
    assert r.status_code == 200, r.text
    assert harness.sent[0]["headers"]["x-license-id"] == str(ref)


async def test_fallback_single_candidate_equal_to_the_ref_is_admitted(harness: Harness) -> None:
    ref = uuid.uuid7()
    _tid, key = await harness.tenant_with_key(ref)  # live == [ref], no token stored
    r = await harness.call("POST", "/api/brand/icp-from-url", key, {"url": "https://a.test"})
    assert r.status_code == 200, r.text
    assert harness.licences.introspect_calls == 0, "no token: the fallback decided"
    assert harness.sent[0]["headers"]["x-license-id"] == str(ref)


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch) -> _Clock:
    """Drive both verification caches off a fake clock (same TTLs as the gate)."""
    from cachetools import TTLCache

    from src.services.license_status import _STALE_TTL_SECONDS, _STATUS_TTL_SECONDS

    c = _Clock()
    monkeypatch.setattr(
        _common, "_api_key_licence_ok", TTLCache(maxsize=8, ttl=_STATUS_TTL_SECONDS, timer=c)
    )
    monkeypatch.setattr(
        _common, "_api_key_licence_stale", TTLCache(maxsize=8, ttl=_STALE_TTL_SECONDS, timer=c)
    )
    return c


async def test_revoked_after_a_good_call_is_refused_once_the_fresh_ttl_lapses(
    harness: Harness, clock: _Clock
) -> None:
    ref = uuid.uuid7()
    raw: list[str] = []

    def mint(tid: uuid.UUID) -> str:
        raw.append(harness.licences.token(tid, ref))
        return raw[0]

    _tid, key = await harness.tenant_with_key(ref, token=mint)
    assert (await harness.call("POST", "/api/probe/snapshot", key)).status_code == 200
    calls = harness.licences.introspect_calls
    assert (await harness.call("POST", "/api/probe/snapshot", key)).status_code == 200
    assert harness.licences.introspect_calls == calls, "positive verdict cached within the TTL"
    harness.licences.revoke(raw[0])
    clock.now += 61  # fresh verdict expired; the stale one must NOT mask a revocation
    r = await harness.call("POST", "/api/probe/snapshot", key)
    assert r.status_code == 402 and "licence_inactive" in r.text, r.text
    harness.licences.introspect_down = True  # …and the confirmed refusal evicted last-known-good
    assert (await harness.call("POST", "/api/probe/snapshot", key)).status_code == 402


async def test_licences_outage_serves_last_known_good_only_within_the_gate_window(
    harness: Harness, clock: _Clock
) -> None:
    ref = uuid.uuid7()
    _tid, key = await harness.tenant_with_key(
        ref, token=lambda tid: harness.licences.token(tid, ref)
    )
    assert (await harness.call("POST", "/api/probe/snapshot", key)).status_code == 200
    harness.licences.introspect_down = True
    clock.now += 61  # fresh expired, stale (900 s) still warm
    within = await harness.call("POST", "/api/probe/snapshot", key)
    assert within.status_code == 200, within.text
    clock.now += 900  # beyond the stale window
    beyond = await harness.call("POST", "/api/probe/snapshot", key)
    assert beyond.status_code == 402 and "licence_unavailable" in beyond.text, beyond.text


async def test_verification_caches_are_bounded_like_the_gate() -> None:
    from src.services.license_status import _CACHE_MAXSIZE

    assert _common._api_key_licence_ok.maxsize == _CACHE_MAXSIZE
    assert _common._api_key_licence_stale.maxsize == _CACHE_MAXSIZE


# ─── review fix 3: body cap on the proxy routes ────────────────────────────


async def test_proxy_body_over_the_cap_is_413_declared_or_chunked(harness: Harness) -> None:
    _tid, key = await harness.tenant_with_key(uuid.uuid7())
    big = {"url": "https://a.test", "pad": "x" * (PROXY_BODY_LIMIT_BYTES + 1)}
    declared = await harness.call("POST", "/api/brand/from-url", key, big)
    assert declared.status_code == 413, declared.text

    payload = json.dumps(big).encode()

    async def chunks():  # no Content-Length → chunked transfer
        for i in range(0, len(payload), 64 * 1024):
            yield payload[i : i + 64 * 1024]

    transport = ASGITransport(app=harness.app, client=_PUBLIC_PEER)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        chunked = await c.post(
            "/api/brand/from-url",
            content=chunks(),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
    assert chunked.status_code == 413, chunked.text
    assert harness.sent == [], "an over-cap body never reaches analysis"
    ok = await harness.call("POST", "/api/brand/from-url", key, {"url": "https://a.test"})
    assert ok.status_code == 200, "a normal body is untouched"


# ─── P9-7 review: critique cap is 10 MiB, but nothing is buffered pre-auth ──


def _metered(payload: bytes, step: int = 64 * 1024) -> tuple[Any, list[int]]:
    """A chunked (no Content-Length) body that records how many bytes the app
    actually pulled — ASGITransport feeds ``receive`` from it lazily."""
    pulled = [0]

    async def gen():
        for i in range(0, len(payload), step):
            chunk = payload[i : i + step]
            pulled[0] += len(chunk)
            yield chunk

    return gen(), pulled


async def _post_raw(app: FastAPI, path: str, content: Any, key: str | None) -> httpx.Response:
    headers = {"Content-Type": "application/json"}
    if key is not None:
        headers["Authorization"] = f"Bearer {key}"
    transport = ASGITransport(app=app, client=_PUBLIC_PEER)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        return await c.post(path, content=content, headers=headers)


async def test_anonymous_oversized_critique_is_refused_without_buffering(
    harness: Harness,
) -> None:
    """The real stack (Auth is a BaseHTTPMiddleware INSIDE the cap middleware):
    an anonymous 12 MiB chunked POST is answered by Auth having read at most
    the 256 KiB pre-auth bound — not the 10 MiB critique allowance."""
    payload = json.dumps({"html": "x" * (12 * 1024 * 1024)}).encode()
    body, pulled = _metered(payload)
    r = await _post_raw(harness.app, "/api/critique", body, key=None)
    assert r.status_code in (401, 403), r.text
    assert pulled[0] <= PROXY_BODY_LIMIT_BYTES, f"pre-auth read {pulled[0]} bytes"
    assert harness.sent == []


async def test_authenticated_critique_html_1mb_is_forwarded(harness: Harness) -> None:
    _tid, key = await harness.tenant_with_key(uuid.uuid7())
    html = "<html><body>" + "x" * (1024 * 1024) + "</body></html>"
    body, _ = _metered(json.dumps({"html": html, "categories": ["seo"]}).encode())
    r = await _post_raw(harness.app, "/api/critique", body, key)
    assert r.status_code == 200, r.text
    assert harness.sent[-1]["path"] == "/api/critique"
    assert harness.sent[-1]["json"]["html"] == html


async def test_authenticated_critique_over_the_cap_is_413_streamed_and_declared(
    harness: Harness,
) -> None:
    from src.middleware.proxy_body_limit import PROXY_BODY_LIMIT_OVERRIDES

    cap = PROXY_BODY_LIMIT_OVERRIDES["/api/critique"]
    _tid, key = await harness.tenant_with_key(uuid.uuid7())
    payload = json.dumps({"html": "x" * (cap + 1)}).encode()

    body, pulled = _metered(payload)
    chunked = await _post_raw(harness.app, "/api/critique", body, key)
    assert chunked.status_code == 413, chunked.text
    assert chunked.json()["error"]["code"] == "REQUEST_BODY_TOO_LARGE"
    # Reading stops at the first chunk past the cap.
    assert pulled[0] <= cap + 64 * 1024, pulled[0]

    declared = await harness.call("POST", "/api/critique", key, {"html": "x" * (cap + 1)})
    assert declared.status_code == 413, declared.text
    assert harness.sent == [], "an over-cap body never reaches analysis"
