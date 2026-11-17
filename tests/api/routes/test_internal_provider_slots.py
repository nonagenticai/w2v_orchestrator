"""``POST /api/internal/provider-slots/resolve`` — satellites read a run's resolved slot.

The route logic under test: cell-lane-only auth, run ownership (404), the run's
PINNED version (not the tenant's live one), purged → 409 ``provider_unavailable``
(never a platform fallback), the uniform ``byo_mode`` policy, the response shape
(secret for tenant, none for deployment), ``no-store``, and that no secret reaches
the logs. The resolver's storage is the resolver test's fakes; the run lookup is a
fake ``_tenant_owns_run`` (its real query is covered by the runs routes).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest
from fastapi import FastAPI, Request, Response
from httpx import ASGITransport, AsyncClient
from w2v_common.auth.identity_context import clear_identity_context, set_identity_context
from w2v_common.tenant import TenantSource

from src.api.routes import internal_provider_slots as route
from src.api.routes._common import get_db_session, get_tenant_id
from src.observability.metrics import provider_slot_resolve_total
from src.services.providers.byo_posture import ByoMode, StaticByoPosture
from src.services.providers.resolver import PROVIDER_SLOTS_KEY, SlotRef
from tests.services.providers.test_resolver import (
    CONN,
    DEC,
    OTHER_TENANT,
    PLAINTEXT,
    TENANT,
    FakeRepo,
    Rec,
    policy,
)

PATH = "/api/internal/provider-slots/resolve"
RUN = UUID("0199aaaa-0000-7000-8000-000000000001")
CALLER = "html-render-service"


def _pin(rec: Rec) -> dict[str, Any]:
    ref = SlotRef(
        slot=rec.slot,
        source="tenant",
        kind=rec.kind,
        connection_id=rec.connection_id,
        version=rec.version,
    )
    return {PROVIDER_SLOTS_KEY: {rec.slot: ref.to_dict()}}


def _app(
    *,
    repo: FakeRepo,
    mode: ByoMode = "local_allowed",
    runs: dict[tuple[UUID, UUID], dict[str, Any]] | None = None,
    lane: str | None = "cell",
    source: TenantSource | None = TenantSource.CELL_JWT,
    monkeypatch: pytest.MonkeyPatch,
    allowed: str = CALLER,
    run_claim: str | None = None,
) -> FastAPI:
    from src.config import clear_settings_cache

    monkeypatch.setenv("W2V_PROVIDER_SLOT_CALLERS", allowed)
    clear_settings_cache()
    app = FastAPI()

    @app.middleware("http")
    async def _cell_lane(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # What w2v_common's AuthMiddleware sets for a verified tenant cell token.
        request.state.tenant_id = str(TENANT)
        if lane is not None:
            request.state.cell_auth_lane = lane
            request.state.cell_scope = "tenant"
            request.state.cell_caller = CALLER
            request.state.cell_claims = SimpleNamespace(run=run_claim)
        set_identity_context(TENANT, None, None, tenant_source=source)
        try:
            return await call_next(request)
        finally:
            clear_identity_context()

    async def _session() -> AsyncIterator[object]:
        yield object()

    app.dependency_overrides[get_db_session] = _session
    app.dependency_overrides[get_tenant_id] = lambda: TENANT
    app.dependency_overrides[route.get_resolve_byo_posture] = lambda: StaticByoPosture(mode)
    monkeypatch.setattr(
        route,
        "build_resolver_deps",
        lambda _s, _settings: SimpleNamespace(repo=repo, decryptor=DEC, policy=policy()),
    )
    owned = runs or {}

    async def _owns(_session: object, run_id: UUID, tenant_id: UUID) -> Any:
        snap = owned.get((run_id, tenant_id))
        return None if snap is None else SimpleNamespace(config_snapshot=snap)

    monkeypatch.setattr(route, "_tenant_owns_run", _owns)
    app.include_router(route.router)
    return app


async def _post(app: FastAPI, body: dict[str, Any]) -> Any:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        return await c.post(PATH, json=body)


def _count(**labels: str) -> float:
    return provider_slot_resolve_total.labels(**labels)._value.get()


async def test_pinned_version_wins_after_the_tenant_edits(monkeypatch) -> None:
    v1 = Rec(1).sealed()
    v2 = replace(Rec(2), model="edited-later").sealed()
    app = _app(
        repo=FakeRepo(active=v2, versions=[v1]),
        runs={(RUN, TENANT): _pin(v1)},
        monkeypatch=monkeypatch,
    )
    before = _count(slot="llm", source="tenant", caller=CALLER, outcome="ok")
    r = await _post(app, {"slot": "llm", "run_id": str(RUN)})
    assert r.status_code == 200, r.text
    assert r.headers["cache-control"] == "no-store"
    assert r.json() == {
        "source": "tenant",
        "slot": "llm",
        "kind": "openai-compatible",
        "base_url": "https://1.1.1.1/v1",
        "model": "gpt-x",
        "options": {},
        "secret": PLAINTEXT,
        "ref": {"connection_id": str(CONN), "version": 1},
    }
    assert _count(slot="llm", source="tenant", caller=CALLER, outcome="ok") == before + 1
    # Without run_id: the tenant's LIVE connection (v2).
    live = await _post(app, {"slot": "llm", "run_id": None})
    assert live.status_code == 200 and live.json()["ref"]["version"] == 2


async def test_a_run_of_another_tenant_is_404(monkeypatch) -> None:
    app = _app(repo=FakeRepo(), runs={(RUN, OTHER_TENANT): _pin(Rec(1))}, monkeypatch=monkeypatch)
    r = await _post(app, {"slot": "llm", "run_id": str(RUN)})
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "run_not_found"
    assert r.headers["cache-control"] == "no-store"


@pytest.mark.parametrize(
    ("lane", "source"),
    [(None, TenantSource.LICENSE_TOKEN)],
)
async def test_any_lane_but_the_cell_lane_is_403(
    monkeypatch, lane: str | None, source: TenantSource | None
) -> None:
    app = _app(
        repo=FakeRepo(active=Rec(1).sealed()), lane=lane, source=source, monkeypatch=monkeypatch
    )
    r = await _post(app, {"slot": "llm", "run_id": None})
    assert r.status_code == 403
    assert r.json()["detail"]["code"] == "cell_lane_required"
    assert PLAINTEXT not in r.text


async def test_a_purged_pinned_version_is_provider_unavailable_never_the_platform(
    monkeypatch,
) -> None:
    v1 = replace(Rec(1), purged_at=datetime(2026, 10, 1, tzinfo=UTC))
    app = _app(
        repo=FakeRepo(active=Rec(2).sealed(), versions=[v1]),
        runs={(RUN, TENANT): _pin(v1)},
        monkeypatch=monkeypatch,
    )
    r = await _post(app, {"slot": "llm", "run_id": str(RUN)})
    assert r.status_code == 409, r.text
    assert r.json()["detail"]["code"] == "provider_unavailable"
    assert "purged" in r.json()["detail"]["message"]


async def test_byo_required_with_only_the_platform_is_refused(monkeypatch) -> None:
    app = _app(repo=FakeRepo(), mode="byo_required", monkeypatch=monkeypatch)
    r = await _post(app, {"slot": "tts", "run_id": None})
    assert r.status_code == 403
    assert r.json()["detail"] == {
        "code": "byo_provider_required",
        "modality": "tts",
        "slots": "tts",
        "message": "Configure your own TTS provider to continue",
    }


async def test_platform_only_answers_the_platform_without_a_secret(monkeypatch) -> None:
    app = _app(repo=FakeRepo(active=Rec(1).sealed()), mode="platform_only", monkeypatch=monkeypatch)
    r = await _post(app, {"slot": "llm", "run_id": None})
    assert r.status_code == 200, r.text
    assert r.json() == {
        "source": "deployment",
        "slot": "llm",
        "kind": "nonagentic-wm",
        "base_url": None,
        "model": None,
        "options": {},
        "secret": None,
        "ref": None,
    }


async def test_an_unknown_or_grant_slot_is_422(monkeypatch) -> None:
    app = _app(repo=FakeRepo(), monkeypatch=monkeypatch)
    for slot in ("nope", "publish.youtube"):
        r = await _post(app, {"slot": slot, "run_id": None})
        assert r.status_code == 422 and r.json()["detail"]["code"] == "unknown_slot"


async def test_the_secret_never_reaches_the_logs(monkeypatch, caplog) -> None:
    v1 = Rec(1).sealed()
    app = _app(
        repo=FakeRepo(versions=[v1]), runs={(RUN, TENANT): _pin(v1)}, monkeypatch=monkeypatch
    )
    with caplog.at_level(logging.DEBUG):
        r = await _post(app, {"slot": "llm", "run_id": str(RUN)})
    assert r.status_code == 200 and r.json()["secret"] == PLAINTEXT
    assert caplog.records, "the resolve is logged (slot/source/caller)"
    assert PLAINTEXT not in caplog.text


async def test_a_cell_caller_outside_the_endpoint_allowlist_is_403(monkeypatch) -> None:
    """CELL_ALLOWED_CALLERS is not enough: the answer can carry a decrypted secret."""
    app = _app(repo=FakeRepo(active=Rec(1).sealed()), allowed="", monkeypatch=monkeypatch)
    r = await _post(app, {"slot": "llm", "run_id": None})
    assert r.status_code == 403 and r.json()["detail"]["code"] == "caller_not_allowed"
    assert PLAINTEXT not in r.text
    app = _app(repo=FakeRepo(active=Rec(1).sealed()), allowed="other-svc", monkeypatch=monkeypatch)
    r = await _post(app, {"slot": "llm", "run_id": None})
    assert r.status_code == 403 and r.json()["detail"]["code"] == "caller_not_allowed"


@pytest.mark.parametrize("body_run", [None])
async def test_a_run_scoped_token_may_only_read_its_own_run(
    monkeypatch, body_run: str | None
) -> None:
    v1 = Rec(1).sealed()
    app = _app(
        repo=FakeRepo(active=v1),
        runs={(RUN, TENANT): _pin(v1)},
        run_claim=str(RUN),
        monkeypatch=monkeypatch,
    )
    r = await _post(app, {"slot": "llm", "run_id": body_run})
    assert r.status_code == 403 and r.json()["detail"]["code"] == "run_mismatch"
    assert PLAINTEXT not in r.text
    ok = await _post(app, {"slot": "llm", "run_id": str(RUN)})
    assert ok.status_code == 200 and ok.json()["ref"]["version"] == 1


@pytest.mark.parametrize("raw", ["*"])
def test_a_wildcard_caller_list_is_refused_at_startup(monkeypatch, raw: str) -> None:
    from pydantic import ValidationError

    from src.config import ProviderSlotResolveSettings

    monkeypatch.setenv("W2V_PROVIDER_SLOT_CALLERS", raw)
    with pytest.raises(ValidationError, match="wildcard"):
        ProviderSlotResolveSettings()


def test_the_caller_list_parses_explicit_names(monkeypatch) -> None:
    from src.config import ProviderSlotResolveSettings

    monkeypatch.setenv("W2V_PROVIDER_SLOT_CALLERS", " html-render-service , content-service ")
    assert ProviderSlotResolveSettings().provider_slot_callers == {
        "html-render-service",
        "content-service",
    }
