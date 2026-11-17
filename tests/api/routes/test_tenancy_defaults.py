"""P6 ORCH-1 — tenant defaults on the REAL request path.

Real RS256 JWT → w2v_common ``AuthMiddleware`` → ``TenancyMiddleware`` (membership
mode) → the REAL ``/api/tenant/defaults`` router → real ``tenant_defaults`` /
``admin_audit_log`` rows in Postgres. Only the JWKS socket is faked.

Non-vacuity: every refusal asserts nothing was persisted AND is paired with a control
request through the same call shape that does persist.
"""

from __future__ import annotations

import uuid
from typing import Any, cast

import pytest
from sqlalchemy import func, select

from src.api.routes.tenancy.defaults import put_tenant_defaults
from src.api.schemas.tenant_defaults import TenantDefaultsPut
from src.config import get_settings
from src.database.models import AdminAuditLog, Tenant, TenantDefaults
from src.database.repositories.admin_audit_repository import AuditRepository
from src.database.repositories.connections_repository import ConnectionsRepository
from src.services.stages.audiobook.voice_catalog import VOICE_CATALOG
from tests.api.routes._tenancy_harness import (  # noqa: F401 — fixtures
    Harness,
    harness_fixture,
    signing_key_fixture,
)

pytestmark = pytest.mark.asyncio

PATH = "/api/tenant/defaults"


async def _user(h: Harness, tenant: uuid.UUID, role: str = "tenant_admin") -> str:
    sub = uuid.uuid7()
    await h.member(sub, tenant, role, email=f"{role}-{sub}@acme.test")
    return h.token(sub)


def _doc(tenant: uuid.UUID, expected: int, **over: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "expected_version": expected,
        "voice": {"voice_id": "bf_emma", "speed": 1.1},
        "brand": {
            "primary_color": "#112233",
            "secondary_color": "#abc",
            "font_family": "Inter",
        },
        "captions": {"mode": "burned_in", "karaoke": True},
        "voice_direction": {"tone": "warm"},
        "retention_mode": "auto",
        "model_tier": {"timing_director": "tier-high"},
    }
    body.update(over)
    return body


async def _row(h: Harness, tenant: uuid.UUID) -> TenantDefaults | None:
    async with h.factory() as s:
        return (
            await s.execute(select(TenantDefaults).where(TenantDefaults.tenant_id == tenant))
        ).scalar_one_or_none()


async def _audits(h: Harness, tenant: uuid.UUID) -> list[AdminAuditLog]:
    async with h.factory() as s:
        return list(
            (
                await s.execute(
                    select(AdminAuditLog)
                    .where(
                        AdminAuditLog.target_tenant_id == tenant,
                        AdminAuditLog.action == "defaults.update",
                    )
                    .order_by(AdminAuditLog.created_at)
                )
            )
            .scalars()
            .all()
        )


# ─── read ────────────────────────────────────────────────────────────────


async def test_get_without_a_row_reports_version_zero_and_the_deployment_defaults(
    harness: Harness,
) -> None:
    tenant = await harness.tenant()
    member = await _user(harness, tenant, "member")
    r = await harness.call("GET", PATH, member)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["version"] == 0
    for field in ("voice", "brand", "captions", "voice_direction", "retention_mode"):
        assert body[field] is None, field
    pipeline = get_settings().pipeline
    assert body["deployment"] == {
        "voice_id": None,
        "captions_mode": "none",
        "captions_karaoke": pipeline.captions_karaoke,
        "voice_direction_tone": "confident" if pipeline.voice_direction_default_enabled else None,
        "retention_mode": "critique_only" if pipeline.retention_default_enabled else None,
        "timing_director_model_tier": pipeline.timing_director_model_tier,
    }
    assert await _row(harness, tenant) is None, "a GET must not create a row"


# ─── write: round trip, versions, audit ──────────────────────────────────


async def test_put_round_trip_versions_and_audit(harness: Harness) -> None:
    tenant = await harness.tenant()
    admin = await _user(harness, tenant)
    member = await _user(harness, tenant, "member")

    r = await harness.call("PUT", PATH, admin, json_body=_doc(tenant, 0))
    assert r.status_code == 200, r.text
    created = r.json()
    assert created["version"] == 1
    assert created["voice"] == {"voice_id": "bf_emma", "speed": 1.1}
    assert created["captions"] == {"mode": "burned_in", "karaoke": True}
    assert created["model_tier"] == {"timing_director": "tier-high"}
    assert created["retention_mode"] == "auto"

    # Any member reads exactly what the admin saved.
    r = await harness.call("GET", PATH, member)
    assert r.status_code == 200, r.text
    read = r.json()
    assert {k: v for k, v in read.items() if k != "updated_at"} == {
        k: v for k, v in created.items() if k != "updated_at"
    }

    # A PUT replaces the document: omitted blocks are cleared; null fields inside a
    # block are dropped; a block left empty is stored as SQL NULL.
    r = await harness.call(
        "PUT",
        PATH,
        admin,
        json_body={
            "expected_version": 1,
            "voice": {"voice_id": "am_adam", "speed": None},
            "brand": {"primary_color": None},
            "retention_mode": "off",
        },
    )
    assert r.status_code == 200, r.text
    second = r.json()
    assert second["version"] == 2
    assert second["voice"] == {"voice_id": "am_adam", "speed": None}
    assert second["brand"] is None
    assert second["captions"] is None
    assert second["retention_mode"] == "off"
    row = await _row(harness, tenant)
    assert row is not None
    assert row.version == 2
    assert row.voice == {"voice_id": "am_adam"}
    assert row.brand is None and row.captions is None and row.model_tier is None

    audits = await _audits(harness, tenant)
    assert [a.detail["version"] for a in audits] == [1, 2]
    assert audits[0].actor_subject_id is not None
    changes = cast("dict[str, Any]", audits[1].detail["changes"])
    assert set(changes) == {
        "voice",
        "brand",
        "captions",
        "voice_direction",
        "model_tier",
        "retention_mode",
    }
    assert changes["voice"] == {
        "from": {"voice_id": "bf_emma", "speed": 1.1},
        "to": {"voice_id": "am_adam"},
    }
    assert changes["retention_mode"] == {"from": "auto", "to": "off"}


async def test_captions_mode_accepts_the_burned_alias(harness: Harness) -> None:
    tenant = await harness.tenant()
    admin = await _user(harness, tenant)
    r = await harness.call(
        "PUT", PATH, admin, json_body={"expected_version": 0, "captions": {"mode": "burned"}}
    )
    assert r.status_code == 200, r.text
    assert r.json()["captions"] == {"mode": "burned_in", "karaoke": None}


async def test_stale_or_premature_version_is_409_and_changes_nothing(harness: Harness) -> None:
    tenant = await harness.tenant()
    admin = await _user(harness, tenant)

    r = await harness.call("PUT", PATH, admin, json_body=_doc(tenant, 1))
    assert r.status_code == 409, r.text
    assert r.json()["error"]["code"] == "VERSION_CONFLICT"
    assert await _row(harness, tenant) is None

    r = await harness.call("PUT", PATH, admin, json_body=_doc(tenant, 0))
    assert r.status_code == 200, r.text
    # A second editor still holding version 0 loses.
    r = await harness.call(
        "PUT", PATH, admin, json_body=_doc(tenant, 0, retention_mode="critique_only")
    )
    assert r.status_code == 409, r.text
    assert r.json()["error"]["details"] == {"expected": 0, "current": 1}
    row = await _row(harness, tenant)
    assert row is not None and row.version == 1 and row.retention_mode == "auto"
    assert len(await _audits(harness, tenant)) == 1


# ─── authz ───────────────────────────────────────────────────────────────


async def test_member_may_read_but_not_write(harness: Harness) -> None:
    tenant = await harness.tenant()
    member = await _user(harness, tenant, "member")
    r = await harness.call("PUT", PATH, member, json_body=_doc(tenant, 0))
    assert r.status_code == 403, r.text
    assert await _row(harness, tenant) is None
    assert await _audits(harness, tenant) == []
    r = await harness.call("GET", PATH, member)
    assert r.status_code == 200, r.text
    # Control: the same body as an admin persists.
    admin = await _user(harness, tenant)
    r = await harness.call("PUT", PATH, admin, json_body=_doc(tenant, 0))
    assert r.status_code == 200, r.text


async def test_non_member_is_refused(harness: Harness) -> None:
    tenant = await harness.tenant()
    stranger = harness.token(uuid.uuid7())
    for method, body in (("GET", None), ("PUT", _doc(tenant, 0))):
        r = await harness.call(method, PATH, stranger, json_body=body)
        assert r.status_code in (401, 403), (method, r.status_code, r.text)
    assert await _row(harness, tenant) is None


async def test_tenant_a_cannot_read_or_write_tenant_b(harness: Harness) -> None:
    a, b = await harness.tenant(), await harness.tenant()
    admin_a, admin_b = await _user(harness, a), await _user(harness, b)
    member_a = await _user(harness, a, "member")
    r = await harness.call("PUT", PATH, admin_b, json_body=_doc(b, 0))
    assert r.status_code == 200, r.text

    # A sees its own (empty) defaults, never B's.
    for tok in (admin_a, member_a):
        r = await harness.call("GET", PATH, tok)
        assert r.status_code == 200, r.text
        assert r.json()["version"] == 0
        assert r.json()["voice"] is None
    # A tenant_admin / member cannot scope-switch into B (super_admin only).
    for method, tok, body in (
        ("GET", member_a, None),
        ("GET", admin_a, None),
        ("PUT", admin_a, _doc(b, 1, retention_mode="off")),
    ):
        r = await harness.call(
            method, PATH, tok, json_body=body, headers={"X-Active-Tenant-Id": str(b)}
        )
        assert r.status_code == 403, (method, r.text)
    # A's own write creates A's row and leaves B untouched.
    r = await harness.call("PUT", PATH, admin_a, json_body=_doc(a, 0, retention_mode="off"))
    assert r.status_code == 200, r.text
    row_b = await _row(harness, b)
    assert row_b is not None and row_b.version == 1 and row_b.retention_mode == "auto"
    assert len(await _audits(harness, b)) == 1


# ─── validation ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("over", "code"),
    [({"voice": {"voice_id": "x" * 129}}, None)],
)
async def test_invalid_documents_are_422_and_persist_nothing(
    harness: Harness, over: dict[str, Any], code: str | None
) -> None:
    tenant = await harness.tenant()
    admin = await _user(harness, tenant)
    r = await harness.call("PUT", PATH, admin, json_body={"expected_version": 0, **over})
    assert r.status_code == 422, r.text
    if code is not None:
        assert r.json()["error"]["code"] == code
    assert await _row(harness, tenant) is None


async def test_unknown_voice_message_names_the_catalog(harness: Harness) -> None:
    tenant = await harness.tenant()
    admin = await _user(harness, tenant)
    r = await harness.call(
        "PUT", PATH, admin, json_body=_doc(tenant, 0, voice={"voice_id": "nova"})
    )
    assert r.status_code == 422, r.text
    err = r.json()["error"]
    assert err["code"] == "UNKNOWN_VOICE"
    assert "'nova'" in err["message"] and f"{len(VOICE_CATALOG)} catalog voices" in err["message"]
    assert await _row(harness, tenant) is None
    # Control: the same call with a catalog voice persists.
    r = await harness.call(
        "PUT", PATH, admin, json_body=_doc(tenant, 0, voice={"voice_id": "af_heart"})
    )
    assert r.status_code == 200, r.text


async def test_a_tenant_tts_provider_accepts_its_own_voice_namespace(harness: Harness) -> None:
    """With an active tenant ``tts`` connection the voice is the provider's (not
    enumerable), so a non-kokoro id is accepted; another tenant without one is refused."""
    tenant = await harness.tenant()
    other = await harness.tenant()
    async with harness.factory() as s:
        await ConnectionsRepository(s).create_version(
            tenant_id=tenant,
            slot="tts",
            kind="openai-compatible",
            base_url="https://tts.example.com/v1",
            model="kokoro",
            options={"voice": "nova"},
            secret=None,
        )
        await s.commit()
    body = _doc(tenant, 0, voice={"voice_id": "nova"})
    r = await harness.call("PUT", PATH, await _user(harness, tenant), json_body=body)
    assert r.status_code == 200, r.text
    assert r.json()["voice"]["voice_id"] == "nova"
    r = await harness.call(
        "PUT",
        PATH,
        await _user(harness, other),
        json_body=_doc(other, 0, voice={"voice_id": "nova"}),
    )
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "UNKNOWN_VOICE"


async def test_put_is_committed_before_the_response_is_built(db_session_factory) -> None:
    """Called DIRECTLY (a test client runs the teardown commit before returning, so it
    proves nothing — repo CLAUDE.md): the row + its audit must be visible to another
    connection while the handler's session is still open."""
    tenant = uuid.uuid7()
    async with db_session_factory() as s:
        s.add(Tenant(id=tenant, name=f"durable-{tenant}"))
        await s.commit()
    async with db_session_factory() as session:
        view = await put_tenant_defaults(
            body=TenantDefaultsPut.model_validate(_doc(tenant, 0)),
            tenant_id=tenant,
            actor=uuid.uuid4(),
            audit=AuditRepository(session),
            session=session,
        )
        assert view.version == 1
        async with db_session_factory() as probe:
            rows = (
                await probe.execute(
                    select(func.count())
                    .select_from(TenantDefaults)
                    .where(TenantDefaults.tenant_id == tenant)
                )
            ).scalar_one()
            actions = (
                (
                    await probe.execute(
                        select(AdminAuditLog.action).where(AdminAuditLog.target_tenant_id == tenant)
                    )
                )
                .scalars()
                .all()
            )
    assert rows == 1, "PUT /api/tenant/defaults returned before committing"
    assert list(actions) == ["defaults.update"]
