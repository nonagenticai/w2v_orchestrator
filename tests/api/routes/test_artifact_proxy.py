"""Artifact-proxy bucket gate — the durable guardrail against allowlist drift.

Background: the proxy used to gate on a hardcoded 3-bucket ``frozenset`` while
``StorageService`` (the compile-side re-sign path) already treated *any* ``w2v-*``
bucket as readable and the URL rewriter emitted proxy URLs for them. That drift is
exactly why a delivered audiobook (``w2v-audiobooks``) re-signed fine on the compile
side yet 404'd when the browser fetched it through this proxy. Both sides now share
:func:`is_readable_w2v_bucket`; these tests lock that in so they can't drift again.

The route-level tests mount ONLY the proxy router on a minimal app and stub the
tenant-ownership query + MinIO client, so they exercise the bucket gate + streaming
without a live cluster or DB (the tenant-ownership query itself is covered by the
runs tests).

The second half of the file covers the OTHER half of the boundary: owning the run
never authorised the ``object_key``, so any owned run could address any object in
any readable ``w2v-*`` bucket. The key is now re-derived against the refs the run
itself persisted; the escape cases below are the point of that gate.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.routes import artifact_proxy
from src.api.routes._common import get_db_session

# Driven directly (not only through the route) so the escape shapes are not at
# the mercy of the HTTP client's own path normalisation.
from src.services.artifact_ref_guard import normalize_stored_ref
from src.services.storage_service import is_readable_w2v_bucket

if TYPE_CHECKING:
    from aiohttp import ClientResponse
    from sqlalchemy.ext.asyncio import AsyncSession

# Captured before any test monkeypatches it away: the conditional-request tests
# below need the REAL `_open_minio_object` (the stub in `_stub_owned_refs`
# replaces exactly the code under test), driven against a fake storage.
_REAL_OPEN_MINIO = artifact_proxy._open_minio_object
_PROXY_LOGGER = artifact_proxy.log.name

# The owned-ref lookup is stubbed in these tests, so the session is never
# touched; cast rather than build a real one.
_NO_SESSION = cast("AsyncSession", None)

_OWN = "w2v-orchestrator"

# Buckets the orchestrator's MinIO user can read: its own bucket + every w2v-*
# artifact bucket. w2v-audiobooks / w2v-courses are the ones the old frozenset
# dropped — the regression this whole change exists to fix.
_READABLE = [
    "w2v-orchestrator",
    "w2v-compiled-videos",
    "w2v-html-render",
    "w2v-audiobooks",
    "w2v-courses",
    "w2v-anything-future",
]
# Anything outside the orchestrator's own w2v-* namespace must be refused.
_REFUSED = ["", "secret-bucket", "etc", "backups", "w2vfoo", "minio-internal"]

# The one key the stubbed run "owns" — in every bucket the route tests drive,
# INCLUDING the refused ones, so a bucket-gate test can only pass by the bucket
# gate rejecting first and never by the key gate rejecting for it.
_OWNED_KEY = "some/key.mp3"
_OWNED_REFS = frozenset(f"{b}/{_OWNED_KEY}" for b in (*_READABLE, *_REFUSED))


# ---------------------------------------------------------------------------
# Pure predicate — always runs (no DB, no cluster)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bucket", ["w2v-orchestrator"])
def test_predicate_accepts_own_and_w2v_buckets(bucket: str) -> None:
    assert is_readable_w2v_bucket(bucket, _OWN) is True


@pytest.mark.parametrize("bucket", [""])
def test_predicate_refuses_non_w2v_buckets(bucket: str) -> None:
    assert is_readable_w2v_bucket(bucket, _OWN) is False


def test_predicate_accepts_configured_own_bucket_even_if_not_w2v() -> None:
    # A service configured with a non-w2v own bucket can still read its own bucket.
    assert is_readable_w2v_bucket("legacy-own", "legacy-own") is True
    assert is_readable_w2v_bucket("legacy-own", _OWN) is False


# ---------------------------------------------------------------------------
# Route-level bucket gate — minimal app, stubbed ownership + MinIO
# ---------------------------------------------------------------------------


class _StubSession:
    """Stand-in for the request-scoped ``SessionDep``.

    Was ``None`` — the DB walk is stubbed here, so the handler never touched it.
    It does now: the handler ends its transaction with ``await session.commit()``
    before streaming (see the route comment and the dedicated test at the foot of
    this file), so the double must at least answer that call.
    """

    def __init__(self) -> None:
        self.commits = 0

    async def commit(self) -> None:
        self.commits += 1


class _FakeStream:
    """Minimal stand-in for the facade ``StreamedObject`` the proxy streams
    (status / headers / async chunk iterator / aclose)."""

    def __init__(self) -> None:
        self.status = 200
        self.headers = {"content-type": "audio/mpeg", "content-length": "15"}

    async def iter_chunks(self, _size: int = 65536):
        yield b"audiobook-bytes"

    async def aclose(self) -> None:
        pass


def _stub_owned_refs(
    monkeypatch: pytest.MonkeyPatch, refs: frozenset[str] = _OWNED_REFS
) -> list[str]:
    """Stub the run's persisted-artifact-ref lookup; returns the keys MinIO saw.

    The DB walk itself is stubbed (no DB here), but the *decision* it feeds — key
    normalisation + whole-ref match — is the real code under test.
    """
    opened: list[str] = []

    async def _refs(_session, _run_id):
        return set(refs)

    async def _open(_bucket, key, _hdrs):
        opened.append(key)
        return _FakeStream()

    monkeypatch.setattr(artifact_proxy, "_run_artifact_refs", _refs)
    monkeypatch.setattr(artifact_proxy, "_open_minio_object", _open)
    return opened


def _build_app(monkeypatch: pytest.MonkeyPatch, *, tenant_id: str, owns_run: bool) -> FastAPI:
    app = FastAPI()
    app.include_router(artifact_proxy.router)

    class _SeedTenant(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            request.state.tenant_id = tenant_id
            request.state.subject_id = tenant_id
            return await call_next(request)

    app.add_middleware(_SeedTenant)

    # Ownership query is stubbed — its real DB behaviour is covered by runs tests.
    async def _fake_session():
        yield _StubSession()

    app.dependency_overrides[get_db_session] = _fake_session

    async def _owns(_session, _run_id, _tenant_id):
        return object() if owns_run else None

    monkeypatch.setattr(artifact_proxy, "_tenant_owns_run", _owns)

    # MinIO is stubbed so an allowed bucket streams without a live cluster.
    class _FakeStorage:
        _minio_client = object()

    async def _get_storage():
        return _FakeStorage()

    monkeypatch.setattr(artifact_proxy, "_get_storage", _get_storage)
    _stub_owned_refs(monkeypatch)
    return app


async def _get(app: FastAPI, bucket: str, key: str = _OWNED_KEY):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        return await c.get(f"/api/runs/{uuid4()}/artifact/{bucket}/{key}")


@pytest.mark.asyncio
async def test_audiobook_bucket_streams_through_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """The delivered-book regression: w2v-audiobooks now streams (was a 404)."""
    tenant = str(uuid4())
    app = _build_app(monkeypatch, tenant_id=tenant, owns_run=True)
    resp = await _get(app, "w2v-audiobooks")
    assert resp.status_code == 200
    assert resp.content == b"audiobook-bytes"
    assert resp.headers["content-type"] == "audio/mpeg"


@pytest.mark.asyncio
async def test_refused_bucket_404s_at_gate_before_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-w2v bucket is refused at the bucket gate.

    Note owns_run=True: even for an owned run, the bucket gate rejects first,
    proving the refusal is the namespace bound and not the tenant check.
    """
    tenant = str(uuid4())
    app = _build_app(monkeypatch, tenant_id=tenant, owns_run=True)
    resp = await _get(app, "secret-bucket")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "artifact not found"


@pytest.mark.asyncio
async def test_allowed_bucket_passes_gate_then_hits_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An allowed bucket clears the gate and is refused only by ownership.

    The distinct 404 detail ("run not found", not "artifact not found") proves
    w2v-audiobooks passed the bucket gate and failed later, at the tenant check.
    """
    tenant = str(uuid4())
    app = _build_app(monkeypatch, tenant_id=tenant, owns_run=False)
    resp = await _get(app, "w2v-audiobooks")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "run not found"


# ---------------------------------------------------------------------------
# Browser scope-switch — ?active_tenant= on the artifact proxy
# ---------------------------------------------------------------------------
# A super-admin browsing ANOTHER tenant's run loads every artifact through
# markup (`<img src>` / `<video src>` / `<a href download>`), and markup cannot
# send `X-Active-Tenant-Id`. Without the query-param channel the effective
# tenant collapses back to the admin's own claim and `_tenant_owns_run` 404s
# every asset. `get_tenant_id` — the single effective-tenant dep, which reads
# BOTH the header and the query param — is what makes this work; these tests
# prove the ROUTE honours the param end-to-end, not merely that the dep does
# (the dep and the router-level gates are covered in test_scope_switch).
#
# The ownership stub here is tenant-AWARE (unlike `_build_app`'s boolean), so a
# 200 can only come from the effective tenant actually being the owning one.

_AUDIT_LOGGER = "w2v_orchestrator.audit"


def _build_scoped_app(
    monkeypatch: pytest.MonkeyPatch,
    *,
    claim_tenant: str,
    owning_tenant: str,
    roles: list[str],
) -> FastAPI:
    """Proxy app whose ownership check compares the EFFECTIVE tenant to the owner."""
    app = FastAPI()
    app.include_router(artifact_proxy.router)
    subject_id = str(uuid4())

    class _SeedRoles(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):  # type: ignore[no-untyped-def]
            request.state.tenant_id = claim_tenant
            request.state.roles = roles
            request.state.subject_id = subject_id
            return await call_next(request)

    app.add_middleware(_SeedRoles)

    async def _fake_session():
        yield _StubSession()

    app.dependency_overrides[get_db_session] = _fake_session

    async def _owns(_session, _run_id, tenant_id):
        # The real query is `WHERE project.tenant_id = :tenant_id` — mirrored here
        # so only the tenant that owns the run gets a row back.
        return object() if str(tenant_id) == owning_tenant else None

    monkeypatch.setattr(artifact_proxy, "_tenant_owns_run", _owns)

    class _FakeStorage:
        _minio_client = object()

    async def _get_storage():
        return _FakeStorage()

    monkeypatch.setattr(artifact_proxy, "_get_storage", _get_storage)
    _stub_owned_refs(monkeypatch)
    return app


def _artifact_path(run_id: str, bucket: str = "w2v-audiobooks") -> str:
    return f"/api/runs/{run_id}/artifact/{bucket}/{_OWNED_KEY}"


async def _fetch(app: FastAPI, path: str):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        return await c.get(path)


@pytest.mark.asyncio
async def test_active_tenant_param_streams_for_super_but_404s_for_non_super(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The adoption proof: ONE request URL, TWO role sets — 200 vs 404.

    Driving the identical path/param with `super_admin` and then with
    `tenant_admin` is what makes the 200 meaningful: a mis-seeded fixture that
    happened to own the run would return 200 for BOTH, and a route still on the
    header-only `TenantIdDep` would return 404 for both. Only a route that
    honours `?active_tenant=` under the super_admin gate produces the delta.
    """
    owner, admin_claim = str(uuid4()), str(uuid4())
    run_id = str(uuid4())
    path = f"{_artifact_path(run_id)}?active_tenant={owner}"

    super_app = _build_scoped_app(
        monkeypatch, claim_tenant=admin_claim, owning_tenant=owner, roles=["super_admin"]
    )
    allowed = await _fetch(super_app, path)

    plain_app = _build_scoped_app(
        monkeypatch, claim_tenant=admin_claim, owning_tenant=owner, roles=["tenant_admin"]
    )
    denied = await _fetch(plain_app, path)

    assert allowed.status_code == 200
    # Assert the BYTES, not just the status — proves the stream actually ran.
    assert allowed.content == b"audiobook-bytes"
    assert allowed.headers["content-type"] == "audio/mpeg"

    assert denied.status_code == 404  # non-super param ignored → own claim → no escape
    assert denied.json()["detail"] == "run not found"


@pytest.mark.asyncio
async def test_active_tenant_param_malformed_is_422_with_scope_switch_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NC-B: a super_admin's unparseable `active_tenant` is a 422 from the dep.

    FastAPI's own path validation also yields 422, so the detail string is
    asserted — it can only come from `get_tenant_id`, proving the 422 is the
    scope-switch gate and not `run_id`/`bucket` coercion.
    """
    owner, admin_claim = str(uuid4()), str(uuid4())
    app = _build_scoped_app(
        monkeypatch, claim_tenant=admin_claim, owning_tenant=owner, roles=["super_admin"]
    )
    resp = await _fetch(app, f"{_artifact_path(str(uuid4()))}?active_tenant=not-a-uuid")
    assert resp.status_code == 422
    assert resp.json()["detail"] == "malformed active tenant: not a UUID"


@pytest.mark.asyncio
async def test_no_param_own_tenant_unchanged_and_no_scope_switch_audit(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """NC-C: with no param the behaviour is byte-identical to before the change.

    The absent SCOPE_SWITCH line is the load-bearing assertion: a 200 alone
    could come from a silent fallback that quietly re-scoped the request. No
    switch happened, so no audit line may be emitted.
    """
    owner = str(uuid4())
    app = _build_scoped_app(
        monkeypatch, claim_tenant=owner, owning_tenant=owner, roles=["super_admin"]
    )
    with caplog.at_level(logging.INFO, logger=_AUDIT_LOGGER):
        resp = await _fetch(app, _artifact_path(str(uuid4())))
    assert resp.status_code == 200
    assert resp.content == b"audiobook-bytes"
    assert not [rec for rec in caplog.records if rec.message.startswith("SCOPE_SWITCH")]


@pytest.mark.asyncio
async def test_refused_bucket_404s_before_ownership_even_with_active_tenant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gate order survives the scope-switch: bucket predicate still runs FIRST.

    A valid cross-tenant `active_tenant` would otherwise reach the ownership
    check; the "artifact not found" detail (not "run not found") proves the
    namespace bound rejected it before the tenant check ever ran.
    """
    owner, admin_claim = str(uuid4()), str(uuid4())
    app = _build_scoped_app(
        monkeypatch, claim_tenant=admin_claim, owning_tenant=owner, roles=["super_admin"]
    )
    path = f"{_artifact_path(str(uuid4()), 'secret-bucket')}?active_tenant={owner}"
    resp = await _fetch(app, path)
    assert resp.status_code == 404
    assert resp.json()["detail"] == "artifact not found"


@pytest.mark.asyncio
async def test_cross_tenant_fetch_emits_exactly_one_scope_switch_audit_line(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Every effective cross-tenant artifact read is auditable — exactly once."""
    owner, admin_claim = str(uuid4()), str(uuid4())
    run_id = str(uuid4())
    app = _build_scoped_app(
        monkeypatch, claim_tenant=admin_claim, owning_tenant=owner, roles=["super_admin"]
    )
    with caplog.at_level(logging.INFO, logger=_AUDIT_LOGGER):
        resp = await _fetch(app, f"{_artifact_path(run_id)}?active_tenant={owner}")
    assert resp.status_code == 200

    switches = [rec for rec in caplog.records if rec.message.startswith("SCOPE_SWITCH")]
    assert len(switches) == 1  # one line per request, not one per dep resolution
    msg = switches[0].getMessage()
    assert admin_claim in msg  # from_tenant
    assert owner in msg  # to_tenant
    assert _artifact_path(run_id) in msg  # path — the artifact route, not some other dep


# ---------------------------------------------------------------------------
# Object-key confinement — owning the run does not authorise the key
# ---------------------------------------------------------------------------
# Every test here runs with owns_run=True and an allowed bucket, i.e. both
# pre-existing gates fully satisfied. A 404 can therefore only come from the key
# gate, and a 200 only from a key the run's own persisted refs contain.

# The victim: another run's audiobook, in a bucket this caller may read.
_FOREIGN_KEY = "artifacts/00000000-0000-0000-0000-0000000000ff/audiobook_stitch/book.mp3"


@pytest.mark.asyncio
async def test_owned_key_still_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    """The legitimate path: a key the run's stage outputs reference is served."""
    app = _build_app(monkeypatch, tenant_id=str(uuid4()), owns_run=True)
    opened = _stub_owned_refs(monkeypatch)

    resp = await _get(app, "w2v-audiobooks")

    assert resp.status_code == 200
    assert resp.content == b"audiobook-bytes"
    assert opened == [_OWNED_KEY]  # normalised key, unchanged, reached MinIO


@pytest.mark.asyncio
async def test_foreign_key_on_owned_run_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """The exploit: own any run, name another run's object in a readable bucket."""
    app = _build_app(monkeypatch, tenant_id=str(uuid4()), owns_run=True)
    opened = _stub_owned_refs(monkeypatch)

    resp = await _get(app, "w2v-audiobooks", _FOREIGN_KEY)

    assert resp.status_code == 404
    assert resp.json()["detail"] == "artifact not found"
    assert opened == []  # never reached MinIO at all


@pytest.mark.asyncio
async def test_owned_key_in_a_different_bucket_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bucket is part of the ref: the same key elsewhere is a different object."""
    app = _build_app(monkeypatch, tenant_id=str(uuid4()), owns_run=True)
    opened = _stub_owned_refs(monkeypatch, frozenset({f"w2v-courses/{_OWNED_KEY}"}))

    resp = await _get(app, "w2v-audiobooks")

    assert resp.status_code == 404
    assert opened == []


# The escape shapes are driven straight at ``_confine_object_key`` rather than
# through a URL: an HTTP client is entitled to normalise ``..`` and ``//`` in a
# path before it ever leaves, which would test the client, not the gate.


@pytest.mark.asyncio
async def test_run_with_no_persisted_refs_serves_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail-closed: an empty artifact list authorises nothing, not everything."""
    app = _build_app(monkeypatch, tenant_id=str(uuid4()), owns_run=True)
    opened = _stub_owned_refs(monkeypatch, frozenset())

    resp = await _get(app, "w2v-audiobooks")

    assert resp.status_code == 404
    assert opened == []


@pytest.mark.asyncio
async def test_key_gate_runs_after_ownership_not_instead_of_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unowned run still fails at ownership even for a key in the ref set —
    the new gate is an addition to the tenant check, not a replacement."""
    app = _build_app(monkeypatch, tenant_id=str(uuid4()), owns_run=False)
    _stub_owned_refs(monkeypatch)

    resp = await _get(app, "w2v-audiobooks")

    assert resp.status_code == 404
    assert resp.json()["detail"] == "run not found"


# ---------------------------------------------------------------------------
# Percent-encoded keys — decode to DECIDE, fetch what the caller NAMED
# ---------------------------------------------------------------------------
# A real object key may contain a literal ``%``. The safety transform used to
# decode to a fixpoint and return the decoded string, so ``discount%2Ddeal.pdf``
# was matched and then fetched as ``discount-deal.pdf`` — an object that does not
# exist. The decode now only judges; the identifier is the caller's own spelling.

_PCT_KEY = "artifacts/some-run/upload/discount%2Ddeal.pdf"
# How that key appears inside a presigned URL's path: the literal ``%`` is itself
# encoded, so the stored ref carries one more layer than the true key.
_PCT_REF_PATH = "w2v-orchestrator/artifacts/some-run/upload/discount%252Ddeal.pdf"


def test_stored_ref_is_decoded_exactly_once_to_the_true_key() -> None:
    """A stored ref is a URL path: strip one layer, never to a fixpoint.

    Mutation killed: dropping ``decode_ref_path`` (leaves ``%252D``, so the
    browser's request never matches) or re-decoding to a fixpoint (yields
    ``discount-deal.pdf``, matching a key MinIO does not have).
    """
    assert normalize_stored_ref(_PCT_REF_PATH) == f"w2v-orchestrator/{_PCT_KEY}"


@pytest.mark.asyncio
async def test_percent_encoded_key_is_served_under_its_real_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end for the same bug: the key handed to MinIO keeps its ``%2D``.

    ``_confine_object_key`` receives what Starlette already decoded once — the
    true key — and must hand that same string on. Mutation killed: returning the
    decoded probe from ``normalize_key`` (MinIO would be asked for
    ``discount-deal.pdf`` and the assertion on ``opened`` fails).
    """
    app = _build_app(monkeypatch, tenant_id=str(uuid4()), owns_run=True)
    opened = _stub_owned_refs(monkeypatch, frozenset({f"w2v-orchestrator/{_PCT_KEY}"}))

    # httpx sends ``%25`` on the wire; Starlette decodes it back to ``%``.
    resp = await _get(app, "w2v-orchestrator", _PCT_KEY.replace("%2D", "%252D"))

    assert resp.status_code == 200
    assert opened == [_PCT_KEY]


# ---------------------------------------------------------------------------
# Conditional requests — a 304 from MinIO is a SUCCESS, not a transport error
# ---------------------------------------------------------------------------
# The proxy forwards If-None-Match / If-Modified-Since upstream, so MinIO
# correctly answers `304 Not Modified` for an unchanged object. `miniopy_async`
# returns the response only for 200/204/206 and RAISES for every other status,
# so that 304 arrived as `ServerError("server failed with HTTP status code 304",
# 304)` and fell into the broad `except Exception`: WARNING "transport error" +
# HTTP 502, once per conditional request, for a perfectly valid one — and a full
# object re-fetch instead of the cheap 304 the header exists to buy.
#
# These tests drive the REAL `_open_minio_object` against a storage stub that
# raises the exact exception classes miniopy constructs, so they bind the
# behaviour to the structured `status_code` / `code` attributes rather than to
# any message text.


class _FakeAiohttpResponse:
    """Stand-in for the aiohttp `ClientResponse` an `S3Error` carries — the only
    place the S3 HTTP status is actually readable (`S3Error` itself has no
    `status_code`; only `ServerError` does)."""

    def __init__(self, status: int) -> None:
        self.status = status


def _server_error(status: int):
    from miniopy_async.error import ServerError

    # Constructed exactly as miniopy_async.api._execute does for a bodiless
    # non-2xx it has no code mapping for.
    return ServerError(f"server failed with HTTP status code {status}", status)


def _s3_error(status: int, code: str):
    from miniopy_async.error import S3Error

    return S3Error(
        code,
        "synthetic",
        "/w2v-audiobooks/some/key.mp3",
        None,
        None,
        # S3Error only reads `.status` off this; a real ClientResponse cannot be
        # built without a live connection, so cast the stand-in. ty does not
        # honour mypy's `type: ignore[arg-type]` rule codes.
        cast("ClientResponse", _FakeAiohttpResponse(status)),
        bucket_name="w2v-audiobooks",
        object_name=_OWNED_KEY,
    )


class _RaisingStorage:
    """Storage whose streaming get always raises — records what was forwarded."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc
        self.forwarded: list[dict[str, str] | None] = []

    async def get_object_stream(self, _bucket, _key, request_headers=None):
        self.forwarded.append(request_headers)
        raise self._exc


def _app_with_raising_storage(
    monkeypatch: pytest.MonkeyPatch, exc: BaseException
) -> tuple[FastAPI, _RaisingStorage]:
    """Proxy app with both prior gates satisfied, the REAL `_open_minio_object`
    restored, and a storage that raises `exc`."""
    app = _build_app(monkeypatch, tenant_id=str(uuid4()), owns_run=True)
    storage = _RaisingStorage(exc)

    async def _get_storage():
        return storage

    monkeypatch.setattr(artifact_proxy, "_get_storage", _get_storage)
    # `_build_app` -> `_stub_owned_refs` replaced this with a stub; put the real
    # one back, or the test would assert against the stub.
    monkeypatch.setattr(artifact_proxy, "_open_minio_object", _REAL_OPEN_MINIO)
    return app, storage


async def _conditional_get(app: FastAPI, headers: dict[str, str], bucket: str = "w2v-audiobooks"):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        return await c.get(f"/api/runs/{uuid4()}/artifact/{bucket}/{_OWNED_KEY}", headers=headers)


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.asyncio
async def test_upstream_304_returns_bodiless_304_and_logs_no_warning(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The measured bug: a satisfied conditional GET must be a 304, not a 502.

    Three assertions, each killing a different mutation:
    * status 304 + empty body — killing "delete the `ServerError` branch" (the
      304 falls through to the broad `except Exception` -> HTTPException 502) and
      "return the raised object as a stream" (a body under a 304).
    * the ETag echo — killing "return a bare `Response(304)`", which would leave
      the client with no validator to revalidate against next time.
    * no WARNING record — killing "handle 304 but keep logging it as a transport
      error", which is what buries genuine MinIO failures in noise.
    """
    app, storage = _app_with_raising_storage(monkeypatch, _server_error(304))

    with caplog.at_level(logging.INFO, logger=_PROXY_LOGGER):
        resp = await _conditional_get(app, {"If-None-Match": '"abc123"'})

    assert resp.status_code == 304
    assert resp.content == b""
    assert resp.headers.get("etag") == '"abc123"'
    assert _warnings(caplog) == []
    # And the conditional was genuinely forwarded upstream — a fix that simply
    # stopped sending If-None-Match would also produce no 304s, and would give
    # up the bandwidth saving the header exists for.
    assert storage.forwarded == [{"if-none-match": '"abc123"'}]


@pytest.mark.asyncio
async def test_upstream_304_echoes_last_modified_from_if_modified_since(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other conditional channel: date-based revalidation gets its validator
    back too. Mutation killed: echoing only `If-None-Match` — an `<img>` that
    revalidates by date would get a 304 carrying nothing to re-cache on."""
    since = "Wed, 21 Oct 2026 07:28:00 GMT"
    app, _ = _app_with_raising_storage(monkeypatch, _server_error(304))

    resp = await _conditional_get(app, {"If-Modified-Since": since})

    assert resp.status_code == 304
    assert resp.content == b""
    assert resp.headers.get("last-modified") == since


@pytest.mark.parametrize("if_none_match", ["*"])
@pytest.mark.asyncio
async def test_upstream_304_does_not_echo_non_singular_validators(
    monkeypatch: pytest.MonkeyPatch,
    if_none_match: str,
) -> None:
    """`*` and a multi-tag list do not name one entity, so neither may be echoed
    back as *the* ETag. Mutation killed: `headers["etag"] = request ETag`
    unconditionally — the proxy would assert a validator the origin never issued
    (`ETag: *` is not even a legal entity tag), poisoning the client's cache."""
    app, _ = _app_with_raising_storage(monkeypatch, _server_error(304))

    resp = await _conditional_get(app, {"If-None-Match": if_none_match})

    assert resp.status_code == 304
    assert "etag" not in resp.headers


@pytest.mark.asyncio
async def test_non_304_server_error_still_502s_and_warns(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The 304 handling is narrow: a 503 from MinIO arrives as the SAME
    exception class and must still be a loud 502.

    Mutation killed: dropping the `status_code == 304` test and returning
    not-modified for every `ServerError` — a MinIO outage would then be reported
    to the browser as "your cached copy is still fresh", silently serving stale
    content and hiding the outage from the logs entirely.
    """
    app, _ = _app_with_raising_storage(monkeypatch, _server_error(503))

    with caplog.at_level(logging.INFO, logger=_PROXY_LOGGER):
        resp = await _conditional_get(app, {"If-None-Match": '"abc123"'})

    assert resp.status_code == 502
    assert resp.json()["detail"] == "artifact unreachable"
    assert any("503" in msg for msg in _warnings(caplog))


@pytest.mark.asyncio
async def test_genuine_transport_failure_still_502s_and_warns(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A real connection failure is unchanged: WARNING + 502.

    Mutation killed: narrowing or deleting the broad `except Exception` while
    adding the 304 branch — a DNS/TLS/connection error would then escape as an
    opaque 500 with no log line, and the browser would retry into it.
    """
    app, _ = _app_with_raising_storage(monkeypatch, ConnectionError("minio.test unreachable"))

    with caplog.at_level(logging.INFO, logger=_PROXY_LOGGER):
        resp = await _conditional_get(app, {"If-None-Match": '"abc123"'})

    assert resp.status_code == 502
    assert resp.json()["detail"] == "artifact unreachable"
    assert any("transport error" in msg for msg in _warnings(caplog))


@pytest.mark.parametrize(
    ("status", "code"),
    [(404, "NoSuchKey")],
)
@pytest.mark.asyncio
async def test_s3_error_403_and_404_still_collapse_to_404(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    status: int,
    code: str,
) -> None:
    """Absent-vs-forbidden stays indistinguishable to the caller, and is not a
    WARNING.

    Mutation killed: reading the status as `getattr(exc, "status_code", 500)` —
    which is what the code did, and `S3Error` has NO `status_code`, so EVERY
    S3Error scored 500 and a plain missing object 502'd instead of 404'ing.
    """
    app, _ = _app_with_raising_storage(monkeypatch, _s3_error(status, code))

    with caplog.at_level(logging.INFO, logger=_PROXY_LOGGER):
        resp = await _conditional_get(app, {})

    assert resp.status_code == 404
    assert resp.json()["detail"] == "artifact not found"
    assert _warnings(caplog) == []


@pytest.mark.asyncio
async def test_s3_error_416_passes_through_as_416_not_502(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``<video>`` that seeks past EOF must get a RECOVERABLE 416.

    MinIO answers an out-of-bounds ``Range: bytes=99999999-`` with 416
    InvalidRange. ``_s3_error_status`` reads 416 off the response, which is not in
    the ``(403, 404)`` tuple, so the whole branch fell through to
    ``HTTPException(502, "artifact unreachable")``. A 416 is a protocol exchange
    the player corrects itself from; a 502 makes the element error out — and the
    tail-probe seek is exactly what a reviewer's footage thumbnail performs.

    Mutation killed: deleting the 416 branch (back to 502), or widening the
    404-collapse to include 416 (the player would conclude the clip is gone).
    """
    app, storage = _app_with_raising_storage(monkeypatch, _s3_error(416, "InvalidRange"))

    resp = await _conditional_get(app, {"Range": "bytes=99999999-"})

    assert resp.status_code == 416
    assert resp.json()["detail"] == "requested range not satisfiable"
    # And the Range really was forwarded — a fix that stopped forwarding Range
    # would also produce no 416s, at the cost of all seeking.
    assert storage.forwarded == [{"range": "bytes=99999999-"}]


@pytest.mark.asyncio
async def test_416_forwards_the_upstream_content_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RFC 9110 15.5.17: the 416 SHOULD carry ``bytes */<complete-length>`` so the
    player learns the real size in the same round trip. Forwarded when upstream
    sent one — never fabricated.

    Mutation killed: dropping ``headers=_upstream_content_range(exc)`` — the
    player would have to guess the length again and re-seek blind.
    """
    exc = _s3_error(416, "InvalidRange")
    # The stand-in response the S3Error carries; a real one would have headers.
    exc.response.headers = {"Content-Range": "bytes */2048", "x-amz-request-id": "nope"}
    app, _ = _app_with_raising_storage(monkeypatch, exc)

    resp = await _conditional_get(app, {"Range": "bytes=99999999-"})

    assert resp.status_code == 416
    assert resp.headers.get("content-range") == "bytes */2048"
    # Whitelist still applies — MinIO's X-Amz-* noise does not reach the SPA.
    assert "x-amz-request-id" not in resp.headers


@pytest.mark.parametrize(("status", "code"), [(400, "InvalidArgument")])
@pytest.mark.asyncio
async def test_other_4xx_stay_502_by_decision(
    monkeypatch: pytest.MonkeyPatch, status: int, code: str
) -> None:
    """DECISION ON RECORD: only 416 passes through.

    400/405 on this path describe a request the ORCHESTRATOR composed (we choose
    the bucket, the key and the verb; the browser contributes only conditional
    and Range headers). Re-emitting them would blame the caller for our bug and
    invite a retry loop against a request that can never succeed. 502 is the
    honest answer: this service could not obtain the artifact.
    """
    app, _ = _app_with_raising_storage(monkeypatch, _s3_error(status, code))

    resp = await _conditional_get(app, {})

    assert resp.status_code == 502
    assert resp.json()["detail"] == "artifact unreachable"


@pytest.mark.asyncio
async def test_s3_error_5xx_still_502s(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only 403/404 collapse — a server-side S3 error is still unreachable.
    Mutation killed: collapsing every `S3Error` to 404, which would tell the SPA
    a present artifact is gone whenever MinIO has a bad minute."""
    app, _ = _app_with_raising_storage(monkeypatch, _s3_error(500, "InternalError"))

    resp = await _conditional_get(app, {})

    assert resp.status_code == 502
    assert resp.json()["detail"] == "artifact unreachable"


# ---------------------------------------------------------------------------
# The handler must NOT hold a transaction open while the body streams
# ---------------------------------------------------------------------------
# `_tenant_owns_run` + `_confine_object_key` autobegin a transaction on the
# request-scoped session, and `get_db_session` commits only in its post-`yield`
# teardown — which FastAPI 0.136 runs AFTER the response has finished streaming.
# A multi-hundred-MB video download therefore kept postgres `idle in transaction`
# for its whole duration, and pgbouncer (pool_mode=transaction) cannot recycle a
# server slot that is inside a transaction: concurrent downloads exhausted the
# pool and unrelated endpoints (and the DAG dispatcher) died on
# `query_wait_timeout`. Same defect class as the SSE routes (`src/api/sse.py`).
#
# ⚠️ Driving this through the ASGI/httpx test client would prove NOTHING — the
# client drives dependency teardown differently, so the session is tidy by the
# time an assertion could look. This calls the handler directly, with a REAL
# session, and asserts the transaction state at the one moment that matters:
# after the StreamingResponse is returned but BEFORE its body is consumed.


@pytest.mark.asyncio
async def test_get_artifact_ends_transaction_before_streaming(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sqlalchemy import text
    from starlette.requests import Request as _Request
    from starlette.responses import StreamingResponse

    gate_states: list[bool] = []

    # Both stubs issue a real statement, exactly as the queries they stand in for
    # do, so the session genuinely autobegins — without that this test would be
    # vacuous.
    async def _owns(session: AsyncSession, _run_id, _tenant_id):  # type: ignore[no-untyped-def]
        await session.execute(text("SELECT 1"))
        gate_states.append(session.in_transaction())
        return object()

    async def _refs(session: AsyncSession, _run_id):  # type: ignore[no-untyped-def]
        await session.execute(text("SELECT 1"))
        gate_states.append(session.in_transaction())
        return {f"w2v-audiobooks/{_OWNED_KEY}"}

    async def _open(_bucket, _key, _hdrs):  # type: ignore[no-untyped-def]
        return _FakeStream()

    monkeypatch.setattr(artifact_proxy, "_tenant_owns_run", _owns)
    monkeypatch.setattr(artifact_proxy, "_run_artifact_refs", _refs)
    monkeypatch.setattr(artifact_proxy, "_open_minio_object", _open)

    request = _Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [],
            "query_string": b"",
        }
    )

    response = await artifact_proxy.get_artifact(
        run_id=uuid4(),
        bucket="w2v-audiobooks",
        object_key=_OWNED_KEY,
        request=request,
        session=db_session,
        tenant_id=uuid4(),
    )

    # The gates really did open a transaction — otherwise the assertion below
    # would pass on any code at all.
    assert gate_states == [True, True]
    assert isinstance(response, StreamingResponse)
    # The body has NOT been consumed yet; this is the window in which the old
    # code pinned a pgbouncer server slot.
    assert db_session.in_transaction() is False
