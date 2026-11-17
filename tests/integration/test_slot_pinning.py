"""P3 WP6 gate — a run's ``llm`` slot is pinned at start and survives a provider switch.

End to end over the REAL code paths, with only the external providers faked:

* the tenant's connection is written through the REAL ``/api/tenant/connections``
  router (real RS256 auth + tenancy middleware, real ``HttpProviderTester`` hitting the
  fake provider's ``/models``, real AES-GCM keyring from ``PROVIDER_SECRET_KEYS``);
* runs are started by the REAL ``RunEngine.start_run`` (``build_config_snapshot`` →
  ``pin_slots``) and executed by the REAL ``RunDispatcher`` (``bind_run_provider_slots``),
  over a two-stage blueprint whose stages call the orchestrator's REAL
  ``src.services.llm.dispatch.WMClient.message`` — the single LLM injection point every
  production stage uses;
* the tenant path runs the REAL resolver (decrypt of the pinned version), the REAL
  in-process pydantic-ai client and the REAL ``run_usage`` metering; the only seam is
  ``in_process.inner_transport_factory`` (the socket), answered by fake
  OpenAI-compatible servers A and B that record host / path / model / key / prompts;
* the platform path (no tenant slot) is the REAL ``LLM_BACKEND=vllm`` client the suite
  runs (prod's backend), over the same socket seam: the fake world also answers the
  platform vLLM host, and the REAL ``run_usage`` write records its ``local_gpu`` rows;
* the secret GC is the REAL ``purge_unpinned_provider_secrets`` SECURITY DEFINER
  function; the grace window is crossed by back-dating timestamps, because the
  function clamps its cutoff to ``now() - 1h`` server-side.

Resume mirrors ``POST /api/runs/{id}/resume`` (``lifecycle.resume_run``): strip the
consumed checkpoint, ``mark_resumed``, commit, ``dispatcher.dispatch``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, ClassVar

import httpx
import httpx2
import pytest
from sqlalchemy import select, text
from w2v_common.auth.identity_context import clear_identity_context

from src.api.routes.tenancy.connections import get_provider_tester
from src.auth.run_providers import clear_run_provider_slots
from src.config import clear_settings_cache
from src.database.models import RunUsage, TenantConnection, TenantConnectionVersion
from src.database.repositories.run_repository import RunRepository
from src.services.blueprints.base import DAGBlueprint, StageSpec, handler
from src.services.blueprints.registry import BlueprintRegistry
from src.services.llm import dispatch, in_process
from src.services.providers.errors import SlotUnavailableError
from src.services.providers.gc import purge_unpinned_secrets
from src.services.providers.resolver import SlotRef
from src.services.providers.settings import get_provider_policy
from src.services.providers.tester import HttpProviderTester
from src.services.stages.base import StageHandler, StageInput, StageOutput
from tests.api.routes._tenancy_harness import (  # noqa: F401 — fixtures
    Harness,
    harness_fixture,
    signing_key_fixture,
)
from tests.integration._dispatch_harness import _create_project
from tests.support.platform_llm import CHAT_PATH, VLLM_TEST_BASE_URL

pytestmark = pytest.mark.asyncio

HOST_A = "llm-a.gate.test"
HOST_B = "llm-b.gate.test"
BASE_A = f"https://{HOST_A}/v1"
BASE_B = f"https://{HOST_B}/v1"
KEY_A = "sk-gate-A-0f1e2d3c4b5a"  # pragma: allowlist secret - fake fixture key
KEY_B = "sk-gate-B-9a8b7c6d5e4f"  # pragma: allowlist secret - fake fixture key
#: The platform vLLM the suite's ``LLM_BACKEND=vllm`` points at (tests/conftest.py).
PLATFORM_HOST = httpx.URL(VLLM_TEST_BASE_URL).host
PLATFORM_TOKEN = "test-vllm-licence-token"  # pragma: allowlist secret - conftest fake
WORKER = "script-doctor-critique"
SYSTEM = "P3-GATE-SYSTEM-PROMPT"
FIRST, SECOND = "gate_first", "gate_second"
_BRIEF = {"angle": "growth", "audience": "saas", "target_duration_seconds": 30}


# ─── the fake outside world ──────────────────────────────────────────────


@dataclass
class ProviderCall:
    host: str
    method: str
    path: str
    authorization: str | None
    model: str | None = None
    system: str = ""
    user: str = ""


@dataclass
class World:
    """Fake providers A/B + the platform vLLM, each recording what reached it."""

    provider_calls: list[ProviderCall] = field(default_factory=list)
    platform_calls: list[ProviderCall] = field(default_factory=list)

    # -- providers (OpenAI-compatible) ------------------------------------
    def _answer(self, host: str, method: str, path: str, auth: str | None, raw: bytes) -> tuple:
        if host == PLATFORM_HOST:
            return self._answer_platform(method, path, auth, raw)
        if host not in {HOST_A, HOST_B}:
            return 599, {"error": f"unexpected host {host}"}
        if method == "GET" and path == "/v1/models":
            self.provider_calls.append(ProviderCall(host, method, path, auth))
            return 200, {"object": "list", "data": [{"id": f"model-{host[4]}", "object": "model"}]}
        if method == "POST" and path == "/v1/chat/completions":
            body = json.loads(raw or b"{}")
            msgs = body.get("messages") or []
            system = "\n".join(str(m["content"]) for m in msgs if m["role"] == "system")
            user = "\n".join(str(m["content"]) for m in msgs if m["role"] == "user")
            self.provider_calls.append(
                ProviderCall(host, method, path, auth, body.get("model"), system, user)
            )
            n = len(self.provider_calls)
            return 200, {
                "id": f"chatcmpl-{n}",
                "object": "chat.completion",
                "created": 0,
                "model": body.get("model"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": f"answer from {host}"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 13, "completion_tokens": 5, "total_tokens": 18},
            }
        return 404, {"error": f"no route {method} {path}"}

    def _answer_platform(self, method: str, path: str, auth: str | None, raw: bytes) -> tuple:
        if (method, path) != ("POST", CHAT_PATH):
            return 404, {"error": f"no platform route {method} {path}"}
        body = json.loads(raw or b"{}")
        msgs = body.get("messages") or []
        self.platform_calls.append(
            ProviderCall(
                PLATFORM_HOST,
                method,
                path,
                auth,
                body.get("model"),
                "\n".join(str(m["content"]) for m in msgs if m["role"] == "system"),
                "\n".join(str(m["content"]) for m in msgs if m["role"] == "user"),
            )
        )
        return 200, {
            "id": f"chatcmpl-platform-{len(self.platform_calls)}",
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "answer from platform"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 17, "completion_tokens": 4, "total_tokens": 21},
        }

    def tester_transport(self) -> httpx.MockTransport:
        """For the REAL ``HttpProviderTester`` (httpx)."""

        def _h(request: httpx.Request) -> httpx.Response:
            status, body = self._answer(
                request.url.host,
                request.method,
                request.url.path,
                request.headers.get("authorization"),
                request.content,
            )
            return httpx.Response(status, json=body)

        return httpx.MockTransport(_h)

    def llm_transport(self) -> httpx2.AsyncBaseTransport:
        """For the REAL in-process client (httpx2), behind its ``PinnedTransport``."""

        def _h(request: httpx2.Request) -> httpx2.Response:
            status, body = self._answer(
                request.url.host,
                request.method,
                request.url.path,
                request.headers.get("authorization"),
                request.content,
            )
            return httpx2.Response(status, json=body)

        return httpx2.MockTransport(_h)

    def provider_chats(self, host: str | None = None) -> list[ProviderCall]:
        return [
            c
            for c in self.provider_calls
            if c.path.endswith("/chat/completions") and (host is None or c.host == host)
        ]


_WORLD: World | None = None


def _world() -> World:
    assert _WORLD is not None, "gate world not installed"
    return _WORLD


def _marker(stage: str, run_id: object) -> str:
    return f"[{stage}|{run_id}]"


# ─── a minimal two-stage blueprint whose stages make one LLM call each ───


class _GateIn(StageInput):
    pass


class _GateOut(StageOutput):
    answer: str


class _LLMStage(StageHandler[_GateIn, _GateOut]):
    """One ``message()`` through the orchestrator's WMClient — what every stage does."""

    name: ClassVar[str]
    input_model = _GateIn
    output_model = _GateOut

    async def handle(self, payload: _GateIn, ctx: Any) -> _GateOut:
        wm = dispatch.WMClient(timeout=30.0)
        try:
            resp = await wm.message(
                WORKER,
                f"critique {_marker(self.name, ctx.run_id)}",
                system_prompt=SYSTEM,
                model_override="platform-model",
                work_tier="high",
            )
        finally:
            await wm.aclose()
        return _GateOut(answer=str(resp.output))


class _First(_LLMStage):
    name = FIRST


class _Second(_LLMStage):
    name = SECOND


_BLUEPRINT = DAGBlueprint(
    id="p3_slot_gate_v1",
    project_type="generic",
    stages=[
        StageSpec(name=FIRST, handler_factory=handler(_First), default_timeout_seconds=30),
        StageSpec(
            name=SECOND,
            handler_factory=handler(_Second),
            depends_on=(FIRST,),
            default_timeout_seconds=30,
        ),
    ],
)


def _registry() -> BlueprintRegistry:
    reg = BlueprintRegistry()
    reg.register(_BLUEPRINT)
    return reg


# ─── fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def world(
    monkeypatch: pytest.MonkeyPatch, harness: Harness, db_session_factory: Any
) -> Iterator[World]:
    global _WORLD
    w = World()
    _WORLD = w
    key = base64.b64encode(os.urandom(32)).decode()
    monkeypatch.setenv("PROVIDER_SECRET_KEYS", f"gk1:{key}")
    monkeypatch.setenv("PROVIDER_SSRF_MODE", "private_ok")
    monkeypatch.setenv("PROVIDER_EGRESS_ALLOWLIST", f"{HOST_A},{HOST_B}")
    clear_settings_cache()
    get_provider_policy.cache_clear()

    # The resolver + metering open their own sessions via ``get_db_session``.
    @asynccontextmanager
    async def _session() -> AsyncIterator[Any]:
        async with db_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    monkeypatch.setattr("src.database.session.get_db_session", _session)
    monkeypatch.setattr("src.services.llm.metering.get_db_session", _session)
    monkeypatch.setattr(in_process, "inner_transport_factory", w.llm_transport)
    # The REAL tester, with only its socket swapped for the fake providers.
    harness.app.dependency_overrides[get_provider_tester] = lambda: HttpProviderTester(
        get_provider_policy(), transport=w.tester_transport()
    )
    clear_run_provider_slots()
    yield w
    _WORLD = None
    clear_run_provider_slots()
    clear_identity_context()
    get_provider_policy.cache_clear()
    clear_settings_cache()


@dataclass
class Runs:
    factory: Any
    dispatcher: Any
    engine: Any

    async def start(self, tenant: uuid.UUID, *, pause_after_first: bool) -> uuid.UUID:
        subject = uuid.uuid7()
        _project, deliverable = await _create_project(
            self.factory, tenant=tenant, subject=subject, type_="generic", brief=_BRIEF
        )
        run = await self.engine.start_run(
            deliverable_id=deliverable,
            tenant_id=tenant,
            subject_id=subject,
            parent_run_id=None,
            pause_after_stages=[FIRST] if pause_after_first else [],
        )
        return run.id

    async def wait(self, run_id: uuid.UUID, status: str) -> Any:
        last = None
        for _ in range(300):
            async with self.factory() as s:
                last = await RunRepository(s).get(run_id)
            if last is not None and last.status in {"completed", "failed", "paused"}:
                break
            await asyncio.sleep(0.05)
        assert last is not None
        assert last.status == status, f"run {run_id}: {last.status} (error={last.error!r})"
        return last

    async def resume(self, run_id: uuid.UUID) -> None:
        """What ``POST /api/runs/{id}/resume`` does (``lifecycle.resume_run``)."""
        async with self.factory() as s:
            assert await RunRepository(s).mark_resumed(run_id, pause_after_stages=[])
            await s.commit()
        await self.dispatcher.dispatch(run_id)


@pytest.fixture
async def runs(db_session_factory: Any, redis_client: Any) -> AsyncIterator[Runs]:
    from src.services.run_dispatcher import RunDispatcher
    from src.services.run_engine import RunEngine

    reg = _registry()
    dispatcher = RunDispatcher(session_factory=db_session_factory, registry=reg, redis=redis_client)
    engine = RunEngine(
        session_factory=db_session_factory, registry=reg, dispatch=dispatcher.dispatch
    )
    dispatcher.bind_run_engine(engine)
    try:
        yield Runs(db_session_factory, dispatcher, engine)
    finally:
        await dispatcher.shutdown()


# ─── helpers ─────────────────────────────────────────────────────────────


async def _admin(h: Harness) -> tuple[uuid.UUID, str]:
    tenant = await h.tenant()
    sub = uuid.uuid7()
    await h.member(sub, tenant, "tenant_admin", email=f"admin-{sub}@acme.test")
    return tenant, h.token(sub)


async def _put_llm(h: Harness, tok: str, *, base: str, model: str, key: str, expected: int) -> None:
    r = await h.call(
        "PUT",
        "/api/tenant/connections/llm",
        tok,
        json_body={
            "kind": "openai-compatible",
            "base_url": base,
            "model": model,
            "secret": key,
            "expected_version": expected,
        },
    )
    assert r.status_code == 200, r.text


async def _pin(factory: Any, run_id: uuid.UUID) -> dict[str, Any]:
    async with factory() as s:
        run = await RunRepository(s).get(run_id)
    assert run is not None
    return dict(run.config_snapshot["provider_slots"]["llm"])


async def _version_row(factory: Any, tenant: uuid.UUID, version: int) -> TenantConnectionVersion:
    async with factory() as s:
        return (
            await s.execute(
                select(TenantConnectionVersion)
                .join(
                    TenantConnection, TenantConnection.id == TenantConnectionVersion.connection_id
                )
                .where(
                    TenantConnection.tenant_id == tenant, TenantConnectionVersion.version == version
                )
            )
        ).scalar_one()


async def _gc(factory: Any) -> int:
    async with factory() as s:
        n = await purge_unpinned_secrets(s)
        await s.commit()
        return n


async def _backdate(factory: Any, sql: str, **params: object) -> None:
    async with factory() as s:
        result = await s.execute(text(sql), params)
        assert result.rowcount == 1, sql
        await s.commit()


def _chats_for(world: World, host: str, run_id: uuid.UUID) -> list[str]:
    """Which stages of ``run_id`` reached ``host`` (in call order)."""
    out = []
    for c in world.provider_chats(host):
        for stage in (FIRST, SECOND):
            if _marker(stage, run_id) in c.user:
                out.append(stage)
    return out


def _platform_stages(world: World, run_id: uuid.UUID) -> list[str]:
    return [
        stage
        for c in world.platform_calls
        for stage in (FIRST, SECOND)
        if _marker(stage, run_id) in c.user
    ]


# ─── 1. no slot → the platform LLM ───────────────────────────────────────


async def test_tenant_without_a_slot_reaches_the_platform_llm(
    world: World, harness: Harness, runs: Runs
) -> None:
    tenant, _tok = await _admin(harness)
    run_id = await runs.start(tenant, pause_after_first=False)
    await runs.wait(run_id, "completed")

    pin = await _pin(harness.factory, run_id)
    assert (pin["source"], pin["kind"]) == ("deployment", "nonagentic-wm")
    assert world.provider_calls == [], "no tenant provider may be contacted"
    assert _platform_stages(world, run_id) == [FIRST, SECOND]

    for call in world.platform_calls:
        # The platform's static licence, never a tenant key; its served model, not
        # the caller's (ignored) ``model_override``; the stage's system prompt.
        assert (call.method, call.path) == ("POST", CHAT_PATH)
        assert call.authorization == f"Bearer {PLATFORM_TOKEN}"
        assert call.model == "local-llm"
        assert SYSTEM in call.system
    async with harness.factory() as s:
        usage = (await s.execute(select(RunUsage).where(RunUsage.run_id == run_id))).scalars().all()
    # vLLM-direct skips billing, so the orchestrator writes the ledger row itself.
    assert sorted(r.stage_name for r in usage) == [FIRST, SECOND]
    assert {r.provider for r in usage} == {"vllm"}


# ─── 2-4. pin across a switch, GC, delete ────────────────────────────────


async def test_run_keeps_its_pinned_provider_across_switch_gc_and_delete(
    world: World, harness: Harness, runs: Runs
) -> None:
    h = harness
    tenant, tok = await _admin(h)

    # ── PUT llm=A through the API: the real tester hits A's /models with A's key.
    await _put_llm(h, tok, base=BASE_A, model="model-a", key=KEY_A, expected=0)
    assert [(c.host, c.method, c.path, c.authorization) for c in world.provider_calls] == [
        (HOST_A, "GET", "/v1/models", f"Bearer {KEY_A}")
    ]
    v1 = await _version_row(h.factory, tenant, 1)
    assert v1.secret_ciphertext is not None and KEY_A.encode() not in bytes(v1.secret_ciphertext)

    # ── R1 pins A, runs its first stage on A, pauses.
    r1 = await runs.start(tenant, pause_after_first=True)
    await runs.wait(r1, "paused")
    pin1 = await _pin(h.factory, r1)
    assert (pin1["source"], pin1["version"]) == ("tenant", 1)
    assert _chats_for(world, HOST_A, r1) == [FIRST]

    # ── PUT llm=B (tester hits B), then R2 pins B; R2 pauses too (kept live for GC).
    await _put_llm(h, tok, base=BASE_B, model="model-b", key=KEY_B, expected=1)
    assert world.provider_calls[-1].host == HOST_B
    assert world.provider_calls[-1].path == "/v1/models"
    r2 = await runs.start(tenant, pause_after_first=True)
    await runs.wait(r2, "paused")
    pin2 = await _pin(h.factory, r2)
    assert (pin2["source"], pin2["version"], pin2["connection_id"]) == (
        "tenant",
        2,
        pin1["connection_id"],
    )
    assert _chats_for(world, HOST_B, r2) == [FIRST]

    # ── resume R1 → its next call still hits A (the pinned version), not B.
    await runs.resume(r1)
    await runs.wait(r1, "completed")
    assert _chats_for(world, HOST_A, r1) == [FIRST, SECOND]
    assert _chats_for(world, HOST_B, r1) == [], "a resumed run must not follow the switch"

    # What reached A was A's pinned config: model, decrypted key, system prompt.
    for c in world.provider_chats(HOST_A):
        assert (c.model, c.authorization) == ("model-a", f"Bearer {KEY_A}")
        assert SYSTEM in c.system
    for c in world.provider_chats(HOST_B):
        assert (c.model, c.authorization) == ("model-b", f"Bearer {KEY_B}")
        assert SYSTEM in c.system
    assert world.platform_calls == [], "a tenant-slot run must never reach the platform LLM"

    # Real metering: one tenant_byo row per in-process call, on the pinned version.
    async with h.factory() as s:
        rows = (await s.execute(select(RunUsage).where(RunUsage.run_id == r1))).scalars().all()
    assert sorted(r.stage_name for r in rows) == [FIRST, SECOND]
    metas = [r.metadata_ or {} for r in rows]
    assert {(m["funding"], m["connection_version"]) for m in metas} == {("tenant_byo", 1)}

    # ── GC. A (v1) was superseded by the second PUT; push it past the grace window.
    await _backdate(
        h.factory,
        "UPDATE tenant_connection_versions SET superseded_at = now() - interval '2 hours' "
        "WHERE connection_id = :cid AND version = 1 AND superseded_at IS NOT NULL",
        cid=uuid.UUID(pin1["connection_id"]),
    )
    # R1 is terminal but only just: its pin is still honoured (recently-terminal window).
    assert await _gc(h.factory) == 0
    assert (await _version_row(h.factory, tenant, 1)).secret_ciphertext is not None

    # (The still-PAUSED half of the rule is the next test.) Past the grace window too:
    await _backdate(
        h.factory,
        "UPDATE runs SET completed_at = now() - interval '2 hours', "
        "updated_at = now() - interval '2 hours' WHERE id = :rid",
        rid=r1,
    )
    assert await _gc(h.factory) == 1, "A's secret is purged once no live run pins it"
    a = await _version_row(h.factory, tenant, 1)
    assert a.secret_ciphertext is None and a.purged_at is not None
    b = await _version_row(h.factory, tenant, 2)
    assert b.secret_ciphertext is not None and b.purged_at is None
    clear_identity_context()
    with pytest.raises(SlotUnavailableError):
        await dispatch.resolve_pinned_llm(tenant, SlotRef.from_dict("llm", pin1))

    # R2 is unaffected: resumed after the purge, its second call still hits B.
    await runs.resume(r2)
    await runs.wait(r2, "completed")
    assert _chats_for(world, HOST_B, r2) == [FIRST, SECOND]
    assert _chats_for(world, HOST_A, r2) == []

    # ── DELETE the slot → a new run goes back to the platform LLM.
    r = await h.call("DELETE", "/api/tenant/connections/llm", tok)
    assert r.status_code == 204, r.text
    provider_calls_before = len(world.provider_calls)
    r3 = await runs.start(tenant, pause_after_first=False)
    await runs.wait(r3, "completed")
    assert (await _pin(h.factory, r3))["source"] == "deployment"
    assert _platform_stages(world, r3) == [FIRST, SECOND]
    assert len(world.provider_calls) == provider_calls_before


async def test_gc_keeps_a_superseded_secret_while_a_paused_run_pins_it(
    world: World, harness: Harness, runs: Runs
) -> None:
    """The paused-run half of the GC rule, with a control that proves it can purge."""
    h = harness
    tenant, tok = await _admin(h)
    await _put_llm(h, tok, base=BASE_A, model="model-a", key=KEY_A, expected=0)
    r1 = await runs.start(tenant, pause_after_first=True)
    await runs.wait(r1, "paused")
    pin1 = await _pin(h.factory, r1)
    await _put_llm(h, tok, base=BASE_B, model="model-b", key=KEY_B, expected=1)
    await _backdate(
        h.factory,
        "UPDATE tenant_connection_versions SET superseded_at = now() - interval '2 hours' "
        "WHERE connection_id = :cid AND version = 1 AND superseded_at IS NOT NULL",
        cid=uuid.UUID(pin1["connection_id"]),
    )
    # Make the paused run look old too: "paused" must protect regardless of age.
    await _backdate(
        h.factory,
        "UPDATE runs SET updated_at = now() - interval '2 hours' WHERE id = :rid",
        rid=r1,
    )

    assert await _gc(h.factory) == 0, "a PAUSED run's pinned secret must survive GC"
    assert (await _version_row(h.factory, tenant, 1)).secret_ciphertext is not None

    # The paused run resumes next "week" on the exact credential it started with.
    await runs.resume(r1)
    await runs.wait(r1, "completed")
    assert _chats_for(world, HOST_A, r1) == [FIRST, SECOND]
    assert _chats_for(world, HOST_B, r1) == []

    # Control: same version, same cutoff, once the run is terminal + past grace → purged.
    await _backdate(
        h.factory,
        "UPDATE runs SET completed_at = now() - interval '2 hours', "
        "updated_at = now() - interval '2 hours' WHERE id = :rid",
        rid=r1,
    )
    assert await _gc(h.factory) == 1
    assert (await _version_row(h.factory, tenant, 1)).secret_ciphertext is None
