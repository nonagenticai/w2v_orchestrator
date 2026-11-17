"""Early-access REMOVAL (``src/services/signup/removal.py``): remove an APPROVED
signup end to end — blockers, close access, licences, the login, export, the
tenant delete (never forced), the licences archive, the erasure.

Real test postgres for every row; the approve really runs (identity, provision,
membership, email) against the in-memory licences / Keycloak fakes of
``test_signup_approvals``, extended here with the removal's surface (list /
revoke / archive; set_user_state / groups / role mappings). The own bucket is the
in-memory fake of ``test_tenancy_offboarding``. Telegram never leaves the process.
"""

from __future__ import annotations

import asyncio
import datetime
import hashlib
from collections.abc import Iterator
from typing import Any, cast
from uuid import UUID, uuid7

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import func, select, text, update
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.routes import _common
from src.api.routes.signup_approvals import (
    admin_router,
    door_router,
    register_signup_exception_handlers,
)
from src.api.routes.tenancy.errors import register_tenancy_exception_handlers
from src.clients.keycloak_admin import KeycloakAdminClient
from src.clients.licenses_client import LicensesClient, LicensesClientError
from src.clients.schemas.keycloak import KeycloakRoleMappings, KeycloakUser
from src.clients.schemas.licenses import AppInfoView, LicenseView
from src.database.models import (
    AdminAuditLog,
    Project,
    SignupApproval,
    Tenant,
    TenantInvite,
    TenantOffboarding,
)
from src.database.repositories.members_repository import MembersRepository
from src.lifecycle import signup_tick
from src.services import tts_storage
from src.services.license_status import get_license_status
from src.services.signup import core, removal
from tests.api.routes._p9_licences import TEST_KEYRING, override_licence_keyring
from tests.api.routes.test_signup_approvals import (  # noqa: F401 — fixture
    FakeKeycloak,
    FakeLicences,
    Harness,
    signup_env,
)
from tests.api.routes.test_tenancy_offboarding import _MemStorage

pytestmark = pytest.mark.asyncio

EMAIL = "ann@acme.example"


class RLicences(FakeLicences):
    """+ the removal's licences surface. ``revoke_fail[id] = n`` refuses that
    licence's revoke (403) n times; ``live_after_revoke`` keeps the gate entitled."""

    def __init__(self) -> None:
        super().__init__()
        self.revoked: set[str] = set()
        self.revoke_calls: list[str] = []
        self.revoke_fail: dict[str, int] = {}
        self.archived: dict[str, str] = {}
        self.list_error: Exception | None = None
        self.archive_error: Exception | None = None
        self.live_after_revoke = False
        self.list_gate: asyncio.Event | None = None
        #: Extra rows the listing returns as they are (e.g. another tenant's).
        self.extra_listed: list[LicenseView] = []
        #: Shared DEPENDENCY licences (licences migration 059), a subset of ``holders``.
        self.dependencies: set[str] = set()
        #: The listing carries ``is_dependency`` (else only ``metadata.dependency_of``).
        self.expose_is_dependency = True
        #: licences releases a tenant's dependencies when its last own licence goes.
        self.cascade = False
        #: An explicit revoke of a dependency raises this (popped per call when a list).
        self.dep_revoke_error: dict[str, Exception] = {}

    async def list_tenant_licenses(self, *, tenant_id) -> list[LicenseView]:  # type: ignore[no-untyped-def]
        if self.list_gate is not None:
            await self.list_gate.wait()
        if self.list_error is not None:
            raise self.list_error
        return [
            LicenseView(
                id=lid,
                tenant_id=t,
                status="revoked" if lid in self.revoked else "active",
                revoked_at="2026-10-21T00:00:00Z" if lid in self.revoked else None,
                app=AppInfoView(id=f"app-{a}", key=a, name=a),
                is_dependency=(lid in self.dependencies) if self.expose_is_dependency else None,
                metadata={"dependency_of": "w2v"} if lid in self.dependencies else None,
            )
            for lid, (t, a, _h) in self.holders.items()
            if t == str(tenant_id)
        ] + list(self.extra_listed)

    async def revoke_license(self, license_id: str, *, tenant_id) -> list[str]:  # type: ignore[no-untyped-def]
        self.revoke_calls.append(license_id)
        left = self.revoke_fail.get(license_id, 0)
        if left:
            self.revoke_fail[license_id] = left - 1
            raise LicensesClientError("revoke_license returned 403", status_code=403)
        if license_id in self.dependencies and license_id in self.dep_revoke_error:
            raise self.dep_revoke_error[license_id]
        self.revoked.add(license_id)
        tenant = str(tenant_id)
        mine = [lid for lid, (t, _a, _h) in self.holders.items() if t == tenant]
        if not self.cascade or license_id in self.dependencies:
            return []
        if any(lid not in self.revoked and lid not in self.dependencies for lid in mine):
            return []
        released = [lid for lid in mine if lid in self.dependencies and lid not in self.revoked]
        self.revoked.update(released)
        return released

    async def archive_tenant(self, tenant_id) -> str | None:  # type: ignore[no-untyped-def]
        if self.archive_error is not None:
            raise self.archive_error
        self.archived.setdefault(str(tenant_id), "2026-10-21T00:00:00Z")
        return self.archived[str(tenant_id)]

    async def get_active_license(self, *, tenant_id, app_key: str):  # type: ignore[no-untyped-def]
        got = await super().get_active_license(tenant_id=tenant_id, app_key=app_key)
        if got is None or self.live_after_revoke or got.license_id not in self.revoked:
            return got
        return None


class RKeycloak(FakeKeycloak):
    """+ set_user_state (one RMW, read back), groups, direct role mappings."""

    def __init__(self) -> None:
        super().__init__()
        self.state_writes: list[tuple[str, bool, dict[str, str | None]]] = []

    async def set_user_state(
        self, user_id: str, *, enabled: bool, attributes: dict[str, str | None]
    ) -> KeycloakUser | None:
        u = self.users.get(user_id)
        if u is None:
            return None
        self.state_writes.append((user_id, enabled, attributes))
        u["enabled"] = enabled
        for k, v in attributes.items():
            if v is None:
                u["attributes"].pop(k, None)
            elif k not in self.drop_attrs:
                u["attributes"][k] = [v]
        return KeycloakUser.model_validate(u)

    async def count_groups(self, user_id: str) -> int:
        return len(self.users[user_id].get("groups") or [])

    async def get_user_role_mappings(self, user_id: str) -> KeycloakRoleMappings:
        u = self.users[user_id]
        return KeycloakRoleMappings(
            realm=u.get("realm_roles", ["default-roles-portal"]),
            clients=u.get("client_roles", {}),
        )


class RHarness(Harness):
    lic: RLicences
    idp: RKeycloak
    storage: _MemStorage

    @property
    def rdeps(self) -> removal.RemovalDeps:
        return removal.RemovalDeps(
            licenses=cast("LicensesClient", self.lic),
            identity=cast("KeycloakAdminClient", self.idp),
            storage=cast(Any, self.storage),
        )


@pytest.fixture
def rh(db_session_factory, signup_env) -> Iterator[RHarness]:  # noqa: F811
    lic, idp, mem = RLicences(), RKeycloak(), _MemStorage()
    tts_storage.set_storage_for_tests(cast(Any, mem))
    app = FastAPI()
    app.state.session_factory = db_session_factory
    app.include_router(door_router)
    app.include_router(admin_router)
    register_signup_exception_handlers(app)
    register_tenancy_exception_handlers(app)
    harness = RHarness(app, db_session_factory, lic, idp, signup_env)
    harness.storage = mem

    class _Seed(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):  # type: ignore[no-untyped-def]
            request.state.tenant_id = str(uuid7())
            request.state.subject_id = str(harness.admin_subject)
            request.state.roles = ["super_admin"]
            return await call_next(request)

    app.add_middleware(_Seed)
    app.dependency_overrides[_common.get_signup_keycloak_admin] = lambda: idp
    app.dependency_overrides[_common.get_licenses_client] = lambda: lic
    override_licence_keyring(app, TEST_KEYRING)
    yield harness
    tts_storage.set_storage_for_tests(None)


# ─── helpers ─────────────────────────────────────────────────────────────


async def approved(rh: RHarness, email: str = EMAIL) -> SignupApproval:
    assert (await rh.intake(email)).status_code == 202
    sid = await rh.pending_id(email)
    async with rh.client() as c:
        r = await c.post(f"/api/admin/signup-approvals/{sid}/approve", json={})
    assert r.status_code == 200, r.text
    row = await rh.row(sid)
    assert row.status == "approved" and core.steps_complete(row), row.steps
    return row


async def drive(rh: RHarness) -> int:
    """One signup-tick pass over the due removals (the remove / retry only enqueue)."""
    return await removal.run_due_removals(rh.sf, rh.rdeps)


async def remove(
    rh: RHarness, row: SignupApproval, *, run: bool = True, **body: Any
) -> httpx.Response:
    """POST the remove (202: enqueued), then — unless ``run=False`` — one tick pass."""
    payload = {"confirm": row.tenant_slug, **body}
    async with rh.client() as c:
        r = await c.post(f"/api/admin/signup-approvals/{row.id}/remove", json=payload)
    if r.status_code == 202 and run:
        await drive(rh)
    return r


async def retry(rh: RHarness, row: SignupApproval, body: Any = None) -> httpx.Response:
    """POST the removal Retry (202: enqueued), then one tick pass."""
    async with rh.client() as c:
        r = await c.post(f"/api/admin/signup-approvals/{row.id}/removal/retry", json=body)
    if r.status_code == 202:
        await drive(rh)
    return r


async def rview(rh: RHarness, row: SignupApproval) -> dict[str, Any]:
    """The removal as the onboarding screen shows it NOW (after the tick ran)."""
    return cast(dict[str, Any], removal.removal_view(await rh.row(row.id)))


async def post(rh: RHarness, path: str, body: Any = None) -> httpx.Response:
    async with rh.client() as c:
        return await c.post(path, json=body)


def state(row: SignupApproval) -> str | None:
    return removal.removal_state(row)


def rstep(row: SignupApproval, step: str) -> dict[str, Any]:
    return cast(dict[str, Any], (row.removal or {}).get("steps", {}).get(step) or {})


def user_of(row: SignupApproval) -> str:
    return str(row.steps["identity"]["detail"]["user_id"])


async def tenant(rh: RHarness, tenant_id: UUID | None) -> Tenant:
    async with rh.sf() as s:
        got = await s.get(Tenant, tenant_id, populate_existing=True)
    assert got is not None
    return got


async def jobs(rh: RHarness, tenant_id: UUID | None, kind: str) -> list[TenantOffboarding]:
    async with rh.sf() as s:
        return list(
            (
                await s.execute(
                    select(TenantOffboarding).where(
                        TenantOffboarding.tenant_id == tenant_id, TenantOffboarding.kind == kind
                    )
                )
            ).scalars()
        )


async def make_due(rh: RHarness, sid: UUID) -> None:
    async with rh.sf() as s:
        await s.execute(
            update(SignupApproval)
            .where(SignupApproval.id == sid)
            .values(steps_next_attempt_at=func.now() - datetime.timedelta(seconds=1))
        )
        await s.commit()


async def removal_notices(rh: RHarness) -> list[str]:
    from src.services.signup import telegram as tg

    await tg.drain_telegram_for_tests()
    return [str(p["text"]) for _u, p in rh.sent.messages if "workspace removal" in str(p["text"])]


def tenant_licences(rh: RHarness, row: SignupApproval) -> set[str]:
    return {lid for lid, (t, _a, _h) in rh.lic.holders.items() if t == str(row.tenant_id)}


# ─── the whole removal ───────────────────────────────────────────────────


async def test_removal_undoes_the_approve_and_erases_the_row(rh: RHarness) -> None:
    row = await approved(rh)
    uid = user_of(row)
    # A bundle-minted licence of another app (licences #275, NOT a dependency) is revoked too.
    rh.lic.holders["dep-workers"] = (str(row.tenant_id), "workers", f"svc:{row.tenant_id}:workers")
    licences = tenant_licences(rh, row)
    assert len(licences) == 3

    r = await remove(rh, row)
    assert r.status_code == 202, r.text
    body = {"removal": await rview(rh, row)}
    assert body["removal"]["state"] == "done", body["removal"]
    done = await rh.row(row.id)

    # every step ran, in order, and landed
    assert [s["step"] for s in body["removal"]["steps"]] == list(removal.REMOVAL_STEPS)
    assert {s["step"]: s["status"] for s in body["removal"]["steps"]} == {
        "blockers": "ok",
        "close_access": "ok",
        "licences": "ok",
        "identity": "ok",
        "export": "ok",
        "tenant_delete": "ok",
        "licences_archive": "ok",
        "finalize": "ok",
    }
    # licences: all revoked, confirmed, archived
    assert rh.lic.revoked == licences
    assert rstep(done, "licences")["detail"]["live_after"] == 0
    assert rstep(done, "licences")["detail"]["gate_state"] in removal.DEAD_LICENCE_STATES
    assert rh.lic.archived == {str(row.tenant_id): "2026-10-21T00:00:00Z"}
    # the login: credential-less → deleted
    assert uid not in rh.idp.users and rh.idp.deleted[-1] == uid
    assert rstep(done, "identity")["code"] == "deleted"
    # the tenant: tombstoned through the ordinary delete job, never forced
    assert (await tenant(rh, row.tenant_id)).status == "deleted"
    [delete_job] = await jobs(rh, row.tenant_id, "delete")
    assert delete_job.status == "done" and delete_job.force is False
    assert (delete_job.progress or {}).get("forced_over") == []
    [export_job] = await jobs(rh, row.tenant_id, "export")
    assert export_job.status == "done"
    # the row SURVIVED the tenant delete and its personal data is erased
    assert done.status == "removed"
    assert done.email == f"erased+{row.id}@invalid"
    assert done.name == "" and done.company is None and done.message is None
    assert done.tenant_slug is None and done.tenant_name is None
    assert EMAIL not in str(done.steps) and EMAIL not in str(done.removal)
    assert done.steps_next_attempt_at is None and done.steps_lease_token is None
    # the address and the slug are free again
    assert (await rh.intake(EMAIL)).status_code == 202
    assert await core.pending_id_for_email(rh.sf, EMAIL) is not None
    # Telegram + the audit trail (target-less: it survives the tenant delete)
    [notice] = await removal_notices(rh)
    assert "DONE" in notice and EMAIL not in notice
    assert "login deleted" in notice
    async with rh.sf() as s:
        phases = (
            (
                await s.execute(
                    select(AdminAuditLog.detail["phase"].astext)
                    .where(AdminAuditLog.action == "signup.remove")
                    .order_by(AdminAuditLog.created_at)
                )
            )
            .scalars()
            .all()
        )
    assert phases == ["requested", "identity", "completed"]


async def test_close_access_suspends_and_revokes_pending_invites(rh: RHarness) -> None:
    row = await approved(rh)
    # an extra pending invite FOR THE REQUESTER's own address (not a blocker)
    async with rh.sf() as s:
        await s.execute(
            update(TenantInvite)
            .where(TenantInvite.tenant_id == row.tenant_id)
            .values(status="pending", accepted_subject_id=None, accepted_at=None)
        )
        await s.commit()
    rh.lic.archive_error = LicensesClientError("archive_tenant returned 503", status_code=503)
    r = await remove(rh, row)
    got = await rh.row(row.id)
    assert state(got) == "pending", r.text
    assert rstep(got, "close_access")["detail"]["suspended"] is True
    assert rstep(got, "close_access")["detail"]["revoked_invites"]
    assert rstep(got, "licences_archive")["code"] == removal.LICENCE_ARCHIVE_FAILED
    assert got.removal is not None and got.removal["last_error"]["step"] == "licences_archive"
    # retry → archive lands → done
    rh.lic.archive_error = None
    r = await retry(rh, row)
    assert r.status_code == 202 and state(await rh.row(row.id)) == "done"


# ─── blockers ────────────────────────────────────────────────────────────


async def _untouched(rh: RHarness, row: SignupApproval) -> None:
    got = await rh.row(row.id)
    assert state(got) == "blocked"
    assert (await tenant(rh, row.tenant_id)).status == "active"
    assert rh.lic.revoke_calls == [] and rh.lic.archived == {}
    assert user_of(row) in rh.idp.users and rh.idp.deleted == []
    assert await jobs(rh, row.tenant_id, "delete") == []
    assert set(cast(dict, got.removal)["steps"]) == {"blockers"}


async def test_another_member_blocks_and_nothing_is_touched(rh: RHarness) -> None:
    row = await approved(rh)
    async with rh.sf() as s:
        await MembersRepository(s).add_member(
            subject_id=uuid7(), tenant_id=cast(UUID, row.tenant_id), role="member"
        )
        await s.commit()
    r = await remove(rh, row)
    assert r.status_code == 202
    view = await rview(rh, row)
    assert [b["code"] for b in view["blockers"]] == [removal.OTHER_MEMBER]
    await _untouched(rh, row)
    [notice] = await removal_notices(rh)
    assert "BLOCKED" in notice and removal.OTHER_MEMBER in notice


async def test_a_project_blocks_and_a_retry_after_it_is_gone_finishes(rh: RHarness) -> None:
    row = await approved(rh)
    project_id = uuid7()
    async with rh.sf() as s:
        s.add(
            Project(
                id=project_id,
                tenant_id=row.tenant_id,
                owner_subject_id=UUID(user_of(row)),
                visibility="tenant",
                type="generic",
                title="work",
            )
        )
        await s.commit()
    r = await remove(rh, row)
    assert [b["code"] for b in (await rview(rh, row))["blockers"]] == [removal.TENANT_CONTENT]
    await _untouched(rh, row)
    # the tick never runs a blocked removal
    await make_due(rh, row.id)
    assert await removal.run_due_removals(rh.sf, rh.rdeps) == 0
    async with rh.sf() as s:
        await s.execute(text("DELETE FROM projects WHERE id = :p"), {"p": project_id})
        await s.commit()
    r = await retry(rh, row)
    assert r.status_code == 202 and state(await rh.row(row.id)) == "done"


async def test_a_pending_invite_for_someone_else_blocks(rh: RHarness) -> None:
    row = await approved(rh)
    async with rh.sf() as s:
        s.add(
            TenantInvite(
                id=uuid7(),
                tenant_id=row.tenant_id,
                email="bob@else.example",
                role="member",
                token_hash=b"x" * 32,
                expires_at=datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=1),
            )
        )
        await s.commit()
    r = await remove(rh, row)
    assert r.status_code == 202, r.text
    assert [b["code"] for b in (await rview(rh, row))["blockers"]] == [removal.FOREIGN_INVITE]
    await _untouched(rh, row)


async def test_a_tenant_other_than_the_provisioned_one_blocks(rh: RHarness) -> None:
    row = await approved(rh)
    steps = dict(row.steps)
    steps["provision"] = {**steps["provision"], "detail": {"tenant_id": str(uuid7())}}
    async with rh.sf() as s:
        await s.execute(
            update(SignupApproval).where(SignupApproval.id == row.id).values(steps=steps)
        )
        await s.commit()
    r = await remove(rh, row)
    assert r.status_code == 202, r.text
    assert (await rview(rh, row))["blockers"][0]["code"] == removal.TENANT_MISMATCH
    await _untouched(rh, row)


# ─── licences ────────────────────────────────────────────────────────────


async def test_a_partial_403_revoke_stays_pending_and_the_retry_finishes_it(
    rh: RHarness,
) -> None:
    row = await approved(rh)
    first, second = sorted(tenant_licences(rh, row))
    rh.lic.revoke_fail[second] = 1
    await remove(rh, row)
    got = await rh.row(row.id)
    assert state(got) == "pending"
    rec = rstep(got, "licences")
    assert rec["status"] == "pending" and rec["code"] == removal.LICENCE_FAILED
    assert rec["detail"]["refused"] == [{"license_id": second, "status": 403, "code": None}]
    assert rh.lic.revoked == {first}
    # nothing after the licences step ran
    assert set(cast(dict, got.removal)["steps"]) == {"blockers", "close_access", "licences"}
    assert got.steps_next_attempt_at is not None
    # the TICK picks it up when due and finishes it (only the remaining one is revoked)
    await make_due(rh, row.id)
    assert await removal.run_due_removals(rh.sf, rh.rdeps) == 1
    got = await rh.row(row.id)
    assert state(got) == "done"
    assert rh.lic.revoked == {first, second}
    assert rh.lic.revoke_calls.count(first) == 1


async def test_the_gate_verdict_is_read_fresh_not_from_the_cache(rh: RHarness) -> None:
    row = await approved(rh)
    # The request gate cached "active" for this tenant (no ``now``: the cache is used).
    cached = await get_license_status(
        client=cast(Any, rh.lic),
        app_key="w2v",
        tenant_id=str(row.tenant_id),
        subject_id=None,
        gate_enabled=True,
    )
    assert cached.state == "active"
    await remove(rh, row)
    got = await rh.row(row.id)
    assert rstep(got, "licences")["status"] == "ok"
    assert rstep(got, "licences")["detail"]["gate_state"] == "none"


async def test_a_gate_that_still_reads_active_keeps_the_step_pending(rh: RHarness) -> None:
    row = await approved(rh)
    rh.lic.live_after_revoke = True
    await remove(rh, row)
    got = await rh.row(row.id)
    rec = rstep(got, "licences")
    assert rec["status"] == "pending" and rec["code"] == removal.LICENCE_FAILED
    assert rec["detail"]["gate_state"] == "active"
    assert user_of(row) in rh.idp.users  # the identity step never ran


# ─── the tenant delete ───────────────────────────────────────────────────


async def test_the_delete_runs_unforced_and_a_failed_job_is_resumed(rh: RHarness) -> None:
    row = await approved(rh)
    tid = str(row.tenant_id)
    rh.storage.objects[f"tenants/{tid}/upload.bin"] = b"x"
    rh.storage.fail_prefix_once = f"tenants/{tid}/"
    await remove(rh, row)
    got = await rh.row(row.id)
    rec = rstep(got, "tenant_delete")
    assert rec["status"] == "pending" and rec["code"] == removal.TENANT_DELETE_FAILED
    [job] = await jobs(rh, row.tenant_id, "delete")
    assert job.status == "failed" and job.force is False
    assert (job.progress or {}).get("rows_deleted") is True
    # the row was NOT deleted with the tenant's rows
    assert got.status == "removed" and got.email == EMAIL
    r = await retry(rh, row)
    assert r.status_code == 202 and state(await rh.row(row.id)) == "done"
    [job] = await jobs(rh, row.tenant_id, "delete")  # the SAME job, resumed
    assert job.status == "done" and job.force is False
    assert rh.storage.list_object_keys  # sanity
    assert not [k for k in rh.storage.objects if k.startswith(f"tenants/{tid}/")]


async def test_an_existing_export_is_reused(rh: RHarness) -> None:
    row = await approved(rh)
    async with rh.sf() as s:
        s.add(
            TenantOffboarding(
                id=uuid7(),
                tenant_id=row.tenant_id,
                kind="export",
                status="done",
                artifact_key=f"exports/tenants/{row.tenant_id}/x.zip",
                progress={},
            )
        )
        await s.commit()
    await remove(rh, row)
    got = await rh.row(row.id)
    assert rstep(got, "export")["status"] == "skipped"
    assert len(await jobs(rh, row.tenant_id, "export")) == 1


# ─── the stamped login ───────────────────────────────────────────────────


async def test_a_credentialed_stamped_login_is_disabled_and_marked(rh: RHarness) -> None:
    row = await approved(rh)
    uid = user_of(row)
    rh.idp.users[uid]["credentials"] = [{"type": "password"}]
    await remove(rh, row)
    got = await rh.row(row.id)
    assert state(got) == "done"
    assert rstep(got, "identity")["code"] == "disabled"
    user = rh.idp.users[uid]
    assert user["enabled"] is False
    assert user["attributes"]["signup_app"] == [f"w2v_removed:{row.id}"]
    assert user["attributes"]["w2v_signup_id"] == [str(row.id)]  # every other attribute kept
    assert rh.idp.deleted == []


async def test_a_dropped_marker_reads_disabled_unmarked(rh: RHarness) -> None:
    row = await approved(rh)
    uid = user_of(row)
    rh.idp.users[uid]["federatedIdentities"] = [{"identityProvider": "google"}]
    rh.idp.drop_attrs = {"signup_app"}
    await remove(rh, row)
    got = await rh.row(row.id)
    assert rstep(got, "identity")["code"] == "disabled_unmarked"
    assert rh.idp.users[uid]["enabled"] is False


async def test_a_login_that_lost_the_stamp_is_never_touched(rh: RHarness) -> None:
    row = await approved(rh)
    uid = user_of(row)
    rh.idp.users[uid]["attributes"]["w2v_signup_id"] = [str(uuid7())]
    await remove(rh, row)
    got = await rh.row(row.id)
    assert state(got) == "done"
    assert rstep(got, "identity")["code"] == removal.IDENTITY_NOT_OWNED
    assert uid in rh.idp.users and rh.idp.users[uid].get("enabled", True) is True
    assert rh.idp.deleted == [] and rh.idp.state_writes == []


async def test_a_login_another_tenant_uses_is_kept(rh: RHarness) -> None:
    row = await approved(rh)
    uid = user_of(row)
    other = await approved(rh, "bob@other.example")
    async with rh.sf() as s:
        await MembersRepository(s).add_member(
            subject_id=UUID(uid), tenant_id=cast(UUID, other.tenant_id), role="member"
        )
        await s.commit()
    await remove(rh, row)
    got = await rh.row(row.id)
    assert state(got) == "done"
    assert rstep(got, "identity")["code"] == removal.IDENTITY_KEPT_IN_USE
    assert uid in rh.idp.users and rh.idp.deleted == []


async def test_an_already_deleted_login_is_absent(rh: RHarness) -> None:
    row = await approved(rh)
    del rh.idp.users[user_of(row)]
    await remove(rh, row)
    assert rstep(await rh.row(row.id), "identity")["code"] == "absent"


# ─── the legacy login (approved before logins were stamped) ──────────────


async def legacy(
    rh: RHarness, *, created_offset: datetime.timedelta | None = None
) -> tuple[SignupApproval, str]:
    """An approved row exactly like the two live test workspaces: no identity
    step, an unstamped realm user the provisioning invite created."""
    row = await approved(rh)
    uid = user_of(row)
    steps = dict(row.steps)
    steps.pop("identity")
    async with rh.sf() as s:
        await s.execute(
            update(SignupApproval).where(SignupApproval.id == row.id).values(steps=steps)
        )
        await s.commit()
    user = rh.idp.users[uid]
    user["attributes"].pop("w2v_signup_id")
    decided = cast(datetime.datetime, row.decided_at)
    created = decided + (created_offset or datetime.timedelta(seconds=1))
    user["createdTimestamp"] = int(created.timestamp() * 1000)
    return await rh.row(row.id), uid


def legacy_body(uid: str, **extra: Any) -> dict[str, Any]:
    return {"legacy_identity": {"user_id": uid, "evidence": "provisioning log line"}, **extra}


async def test_a_legacy_row_requires_the_login_to_be_named(rh: RHarness) -> None:
    row, _uid = await legacy(rh)
    r = await remove(rh, row)
    assert r.status_code == 422 and r.json()["error"] == "legacy_identity_required"
    assert (await rh.row(row.id)).status == "approved"
    # …and a stamped row refuses a legacy login
    stamped = await approved(rh, "carl@else.example")
    r = await remove(rh, stamped, **legacy_body(user_of(stamped)))
    assert r.status_code == 422 and r.json()["error"] == "legacy_identity_not_applicable"


async def test_a_legacy_login_without_a_credential_is_deleted(rh: RHarness) -> None:
    row, uid = await legacy(rh)
    r = await remove(rh, row, **legacy_body(uid))
    assert (await rview(rh, row))["state"] == "done", r.text
    got = await rh.row(row.id)
    assert rstep(got, "identity")["code"] == "deleted"
    assert uid not in rh.idp.users
    async with rh.sf() as s:
        detail = (
            await s.execute(
                select(AdminAuditLog.detail).where(
                    AdminAuditLog.action == "signup.remove",
                    AdminAuditLog.detail["phase"].astext == "requested",
                )
            )
        ).scalar_one()
    assert detail["legacy_evidence_sha256"] == hashlib.sha256(b"provisioning log line").hexdigest()
    assert detail["legacy_evidence_length"] == len("provisioning log line")
    assert detail["legacy_evidence_created_at"]
    assert "provisioning log line" not in str(detail)


async def test_a_credentialed_legacy_login_without_the_flag_is_disabled(rh: RHarness) -> None:
    row, uid = await legacy(rh)
    rh.idp.users[uid]["credentials"] = [{"type": "password"}]
    await remove(rh, row, **legacy_body(uid))
    got = await rh.row(row.id)
    assert state(got) == "done"
    assert rstep(got, "identity")["code"] == "disabled"
    assert rh.idp.users[uid]["enabled"] is False


async def test_a_credentialed_legacy_login_with_the_flag_is_deleted(rh: RHarness) -> None:
    row, uid = await legacy(rh)
    rh.idp.users[uid]["credentials"] = [{"type": "password"}]
    await remove(rh, row, **legacy_body(uid, allow_credentialed_delete=True))
    assert rstep(await rh.row(row.id), "identity")["code"] == "deleted"
    assert uid not in rh.idp.users


async def _legacy_blocked(rh: RHarness, row: SignupApproval, uid: str, code: str) -> None:
    """A legacy login that fails the proof blocks in the BLOCKERS step — before
    a single write: the tenant is still active, no licence was revoked, the
    login is untouched and no other step ran."""
    got = await rh.row(row.id)
    assert state(got) == "blocked"
    assert rstep(got, "blockers")["code"] == code
    assert cast(dict, got.removal)["blockers"][0]["code"] == code
    assert uid in rh.idp.users and rh.idp.deleted == [] and rh.idp.state_writes == []
    assert set(cast(dict, got.removal)["steps"]) == {"blockers"}
    assert (await tenant(rh, row.tenant_id)).status == "active"
    assert rh.lic.revoke_calls == [] and rh.lic.archived == {}
    assert await jobs(rh, row.tenant_id, "delete") == []


async def test_a_legacy_id_that_is_not_the_addresss_login_blocks(rh: RHarness) -> None:
    row, uid = await legacy(rh)
    stranger = rh.idp.seed("someone@else.example")
    await remove(rh, row, **legacy_body(stranger))
    await _legacy_blocked(rh, row, uid, removal.LEGACY_IDENTITY_MISMATCH)
    assert stranger in rh.idp.users


async def test_a_legacy_login_created_outside_the_window_blocks(rh: RHarness) -> None:
    row, uid = await legacy(rh, created_offset=-datetime.timedelta(days=1))
    r = await remove(rh, row, **legacy_body(uid))
    assert r.status_code == 202, r.text
    assert (await rview(rh, row))["state"] == "blocked"
    await _legacy_blocked(rh, row, uid, removal.LEGACY_IDENTITY_WINDOW)


@pytest.mark.parametrize(
    "foreign",
    [{"attributes": {"fr_signup_id": ["x"]}}],
)
async def test_a_legacy_login_carrying_foreign_marks_blocks(rh: RHarness, foreign) -> None:
    row, uid = await legacy(rh)
    user = rh.idp.users[uid]
    for key, value in foreign.items():
        if key == "attributes":
            user["attributes"].update(value)
        else:
            user[key] = value
    r = await remove(rh, row, **legacy_body(uid))
    assert r.status_code == 202, r.text
    assert (await rview(rh, row))["blockers"][0]["code"] == removal.LEGACY_IDENTITY_FOREIGN
    await _legacy_blocked(rh, row, uid, removal.LEGACY_IDENTITY_FOREIGN)


async def test_a_blocked_legacy_identity_can_be_corrected_on_retry(rh: RHarness) -> None:
    row, uid = await legacy(rh)
    stranger = rh.idp.seed("someone@else.example")
    await remove(rh, row, **legacy_body(stranger))
    assert state(await rh.row(row.id)) == "blocked"
    r = await retry(rh, row, legacy_body(uid))
    assert r.status_code == 202 and state(await rh.row(row.id)) == "done"
    assert uid not in rh.idp.users and stranger in rh.idp.users


# ─── the action's refusals ───────────────────────────────────────────────


async def test_the_slug_must_be_typed_back_and_only_approved_rows_are_removed(
    rh: RHarness,
) -> None:
    row = await approved(rh)
    r = await post(rh, f"/api/admin/signup-approvals/{row.id}/remove", {"confirm": "nope"})
    assert r.status_code == 422 and r.json()["error"] == "confirmation_mismatch"
    assert (await rh.row(row.id)).status == "approved"
    assert (await rh.intake("pat@x.example")).status_code == 202
    pending = await rh.pending_id("pat@x.example")
    r = await post(rh, f"/api/admin/signup-approvals/{pending}/remove", {"confirm": "x"})
    assert r.status_code == 409 and r.json()["error"] == "approval_not_approved"
    assert (await remove(rh, row)).status_code == 202
    r = await remove(rh, await rh.row(row.id))
    assert r.status_code in (409, 422)


async def test_a_held_steps_lease_refuses_the_remove(rh: RHarness) -> None:
    row = await approved(rh)
    async with rh.sf() as s:
        await s.execute(
            update(SignupApproval)
            .where(SignupApproval.id == row.id)
            .values(
                steps_lease_token=uuid7(), steps_lease_until=func.now() + text("interval '5 min'")
            )
        )
        await s.commit()
    r = await remove(rh, row)
    assert r.status_code == 409 and r.json()["error"] == "steps_in_progress"
    assert (await rh.row(row.id)).status == "approved"


# ─── lease, cap, resolve, reservation ────────────────────────────────────


async def test_two_runners_never_run_one_removal_at_once(rh: RHarness) -> None:
    row = await approved(rh)
    rh.lic.list_error = LicensesClientError("down", status_code=503)
    await remove(rh, row)
    assert state(await rh.row(row.id)) == "pending"
    rh.lic.list_error = None
    rh.lic.list_gate = asyncio.Event()
    first = asyncio.create_task(removal.run_removal(rh.sf, rh.rdeps, row.id, manual=True))
    for _ in range(50):  # wait until the first runner is parked inside the licences step
        await asyncio.sleep(0.02)
        if (await rh.row(row.id)).steps_lease_token is not None and not first.done():
            break
    try:
        # Without the lease the second runner would enter the step and park too.
        second = await asyncio.wait_for(
            removal.run_removal(rh.sf, rh.rdeps, row.id, manual=True), timeout=10
        )
    except TimeoutError:
        rh.lic.list_gate.set()
        await first
        raise
    assert second.ran is False
    r = await post(rh, f"/api/admin/signup-approvals/{row.id}/removal/retry")
    assert r.status_code == 409 and r.json()["error"] == "removal_in_progress"
    rh.lic.list_gate.set()
    done = await first
    assert done.ran is True and state(done.row) == "done"


async def test_the_cap_fails_it_and_resolve_needs_the_acknowledgement(rh: RHarness) -> None:
    row = await approved(rh)
    rh.lic.list_error = LicensesClientError("list returned 403", status_code=403)
    await remove(rh, row)
    for _ in range(core.STEPS_MAX_ATTEMPTS - 1):
        assert state(await rh.row(row.id)) == "pending"
        await make_due(rh, row.id)
        assert await removal.run_due_removals(rh.sf, rh.rdeps) == 1
    got = await rh.row(row.id)
    assert state(got) == "failed"
    assert cast(dict, got.removal)["attempts"] == core.STEPS_MAX_ATTEMPTS
    assert got.steps_next_attempt_at is None
    await make_due(rh, row.id)
    assert await removal.run_due_removals(rh.sf, rh.rdeps) == 0
    [notice] = await removal_notices(rh)
    assert "FAILED" in notice
    # still listed for attention, address still reserved
    async with rh.client() as c:
        listed = (await c.get("/api/admin/signup-approvals?status=needs_attention")).json()
    assert [i["id"] for i in listed["items"]] == [str(row.id)]
    assert (await rh.intake(EMAIL)).json() == {"status": "already_pending"}
    # resolve: the acknowledgement is required
    url = f"/api/admin/signup-approvals/{row.id}/removal/resolve"
    r = await post(rh, url, {"note": "done by hand"})
    assert r.status_code == 422 and r.json()["error"] == "acknowledgement_required"
    r = await post(rh, url, {"note": "done by hand", "acknowledge_live_access": True})
    assert r.status_code == 200
    assert (await rview(rh, row))["state"] == "resolved_manually"
    resolved = (await rview(rh, row))["resolved"]
    assert "note" not in resolved
    assert resolved["note_sha256"] == hashlib.sha256(b"done by hand").hexdigest()
    assert resolved["note_length"] == len("done by hand")
    # a resolved removal is final, and frees the address
    r = await post(rh, url, {"note": "again", "acknowledge_live_access": True})
    assert r.status_code == 409 and r.json()["error"] == "removal_not_resolvable"
    assert (await rh.intake(EMAIL)).json() == {"status": "queued"}


async def test_a_pending_removal_cannot_be_resolved(rh: RHarness) -> None:
    row = await approved(rh)
    rh.lic.list_error = LicensesClientError("down", status_code=503)
    await remove(rh, row)
    r = await post(
        rh,
        f"/api/admin/signup-approvals/{row.id}/removal/resolve",
        {"note": "x" * 5, "acknowledge_live_access": True},
    )
    assert r.status_code == 409 and r.json()["error"] == "removal_not_resolvable"


async def test_the_address_is_reserved_until_the_removal_is_done(rh: RHarness) -> None:
    row = await approved(rh)
    rh.lic.list_error = LicensesClientError("down", status_code=503)
    await remove(rh, row)
    assert state(await rh.row(row.id)) == "pending"
    assert (await rh.intake(EMAIL)).json() == {"status": "already_pending"}
    r = await post(
        rh,
        "/api/admin/signup-approvals/invite",
        {"email": EMAIL, "name": "Ann Example"},
    )
    assert r.status_code == 409 and r.json()["error"] == "early_access_removal_in_progress"
    async with rh.sf() as s:
        open_rows = (
            await s.execute(
                select(func.count())
                .select_from(SignupApproval)
                .where(func.lower(SignupApproval.email) == EMAIL)
            )
        ).scalar_one()
    assert open_rows == 1
    rh.lic.list_error = None
    await make_due(rh, row.id)
    assert await removal.run_due_removals(rh.sf, rh.rdeps) == 1
    assert state(await rh.row(row.id)) == "done"
    assert (await rh.intake(EMAIL)).json() == {"status": "queued"}


async def test_the_signup_tick_also_runs_due_removals(rh: RHarness, monkeypatch) -> None:
    row = await approved(rh)
    rh.lic.list_error = LicensesClientError("down", status_code=503)
    await remove(rh, row)
    rh.lic.list_error = None
    await make_due(rh, row.id)

    async def deps() -> removal.RemovalDeps:
        return rh.rdeps

    monkeypatch.setattr(signup_tick, "tick_deps", lambda: rh.deps)
    monkeypatch.setattr(signup_tick, "removal_deps", deps)
    assert await signup_tick.signup_tick_once(rh.sf) == 1
    assert state(await rh.row(row.id)) == "done"


async def test_the_view_exposes_the_removal_and_the_legacy_need(rh: RHarness) -> None:
    row, _uid = await legacy(rh)
    async with rh.client() as c:
        view = (await c.get(f"/api/admin/signup-approvals/{row.id}")).json()["data"]
    assert view["removal"] is None
    assert view["has_identity_step"] is False and view["needs_legacy_identity"] is True
    stamped = await approved(rh, "carl@else.example")
    async with rh.client() as c:
        view = (await c.get(f"/api/admin/signup-approvals/{stamped.id}")).json()["data"]
    assert view["has_identity_step"] is True and view["needs_legacy_identity"] is False
    async with rh.client() as c:
        removed = (await c.get("/api/admin/signup-approvals?status=removed")).json()["items"]
    assert removed == []


# ─── review fixes (each fails without its fix) ───────────────────────────


async def test_fix1_a_pasted_member_sub_as_legacy_login_blocks_before_any_write(
    rh: RHarness,
) -> None:
    """Defect 1: an operator-named legacy login is NOT the requester until it is
    proven (read-only, in the blockers step). Pasting ANOTHER member's sub must
    not hide that member from ``other_member`` — and nothing may be written."""
    row, uid = await legacy(rh)
    member = rh.idp.seed("colleague@acme.example")
    async with rh.sf() as s:
        await MembersRepository(s).add_member(
            subject_id=UUID(member), tenant_id=cast(UUID, row.tenant_id), role="member"
        )
        await s.commit()
    await remove(rh, row, **legacy_body(member))
    got = await rh.row(row.id)
    codes = [b["code"] for b in cast(dict, got.removal)["blockers"]]
    assert codes == [removal.LEGACY_IDENTITY_MISMATCH, removal.OTHER_MEMBER]
    await _legacy_blocked(rh, row, uid, removal.LEGACY_IDENTITY_MISMATCH)
    assert member in rh.idp.users


async def test_fix1_a_retired_legacy_login_still_counts_as_the_requester_on_a_rerun(
    rh: RHarness,
) -> None:
    row, uid = await legacy(rh)
    real_storage = rh.storage
    rh.storage = cast(Any, None)  # export waits: storage_unavailable
    await remove(rh, row, **legacy_body(uid))
    got = await rh.row(row.id)
    assert state(got) == "pending" and rstep(got, "identity")["code"] == "deleted"
    assert uid not in rh.idp.users
    rh.storage = real_storage
    assert (await retry(rh, row)).status_code == 202
    assert state(await rh.row(row.id)) == "done"


async def test_fix2_resolve_erases_the_personal_data(rh: RHarness) -> None:
    row = await approved(rh)
    rh.lic.list_error = LicensesClientError("down", status_code=503)
    async with rh.sf() as s:
        await MembersRepository(s).add_member(
            subject_id=uuid7(), tenant_id=cast(UUID, row.tenant_id), role="member"
        )
        await s.commit()
    await remove(rh, row)
    assert state(await rh.row(row.id)) == "blocked"
    r = await post(
        rh,
        f"/api/admin/signup-approvals/{row.id}/removal/resolve",
        {"note": f"talked to {EMAIL.upper()}, kept", "acknowledge_live_access": True},
    )
    assert r.status_code == 200, r.text
    got = await rh.row(row.id)
    assert state(got) == "resolved_manually"
    assert got.email == f"erased+{row.id}@invalid"
    assert got.name == "" and got.company is None and got.message is None
    assert got.tenant_slug is None and got.tenant_name is None
    assert EMAIL not in str(got.removal).lower() and EMAIL not in str(got.steps).lower()
    async with rh.sf() as s:
        audit = (
            await s.execute(
                select(AdminAuditLog.detail).where(
                    AdminAuditLog.detail["phase"].astext == "resolved_manually"
                )
            )
        ).scalar_one()
    assert EMAIL not in str(audit).lower()


async def test_fix2_a_finished_removal_row_goes_with_a_later_tenant_delete(
    rh: RHarness,
) -> None:
    """Only a removal still RUNNING keeps its row out of the tenant delete; a
    resolved_manually one (tenant still alive) is deleted with the tenant."""
    from src.services import tenant_offboarding as offboarding

    row = await approved(rh)
    rh.lic.list_error = LicensesClientError("down", status_code=503)
    await remove(rh, row)
    assert state(await rh.row(row.id)) == "pending"

    async def counted() -> int:
        async with rh.sf() as s:
            n = (await offboarding.row_counts(s, cast(UUID, row.tenant_id)))["signup_approvals"]
            await s.rollback()
        return n

    assert await counted() == 0
    async with rh.sf() as s:
        await s.execute(
            update(SignupApproval)
            .where(SignupApproval.id == row.id)
            .values(
                removal=func.jsonb_set(
                    SignupApproval.removal,
                    text("'{state}'::text[]"),
                    text("'\"resolved_manually\"'::jsonb"),
                )
            )
        )
        await s.commit()
    assert await counted() == 1


async def test_fix3_the_done_notice_says_what_happened_to_the_login(rh: RHarness) -> None:
    row = await approved(rh)
    uid = user_of(row)
    other = await approved(rh, "bob@other.example")
    async with rh.sf() as s:
        await MembersRepository(s).add_member(
            subject_id=UUID(uid), tenant_id=cast(UUID, other.tenant_id), role="member"
        )
        await s.commit()
    await remove(rh, row)
    assert rstep(await rh.row(row.id), "identity")["code"] == removal.IDENTITY_KEPT_IN_USE
    [notice] = await removal_notices(rh)
    assert "DONE" in notice and "login KEPT" in notice
    assert "retired" not in notice and "login deleted" not in notice


async def test_fix4_resolve_refuses_a_removal_that_changed_after_it_was_read(
    rh: RHarness, monkeypatch
) -> None:
    row = await approved(rh)
    rh.lic.list_error = LicensesClientError("down", status_code=503)
    async with rh.sf() as s:
        await MembersRepository(s).add_member(
            subject_id=uuid7(), tenant_id=cast(UUID, row.tenant_id), role="member"
        )
        await s.commit()
    await remove(rh, row)
    assert state(await rh.row(row.id)) == "blocked"
    real = removal._load

    async def stale(sf, sid):  # type: ignore[no-untyped-def]
        snapshot = await real(sf, sid)
        async with rh.sf() as s:  # a concurrent writer lands after the read
            await s.execute(
                update(SignupApproval)
                .where(SignupApproval.id == sid)
                .values(
                    removal=func.jsonb_set(
                        SignupApproval.removal, text("'{marker}'::text[]"), text("'\"x\"'::jsonb")
                    ),
                    updated_at=func.now(),
                )
            )
            await s.commit()
        return snapshot

    monkeypatch.setattr(removal, "_load", stale)
    r = await post(
        rh,
        f"/api/admin/signup-approvals/{row.id}/removal/resolve",
        {"note": "by hand", "acknowledge_live_access": True},
    )
    assert r.status_code == 409 and r.json()["error"] == "removal_in_progress"
    got = await real(rh.sf, row.id)
    assert state(got) == "blocked" and cast(dict, got.removal)["marker"] == "x"
    assert got.email == EMAIL


async def test_fix4_amend_refuses_a_removal_that_changed_after_it_was_read(
    rh: RHarness,
) -> None:
    row, uid = await legacy(rh)
    stranger = rh.idp.seed("someone@else.example")
    await remove(rh, row, **legacy_body(stranger))
    snapshot = await rh.row(row.id)
    assert state(snapshot) == "blocked"
    async with rh.sf() as s:
        await s.execute(
            update(SignupApproval)
            .where(SignupApproval.id == row.id)
            .values(
                removal=func.jsonb_set(
                    SignupApproval.removal, text("'{marker}'::text[]"), text("'\"x\"'::jsonb")
                ),
                updated_at=func.now(),
            )
        )
        await s.commit()
    with pytest.raises(core.SignupError) as err:
        await removal._amend_identity(
            rh.sf, snapshot, core.SYSTEM_ACTOR, removal.LegacyIdentity(uid, "log line"), None
        )
    assert err.value.status_code == 409 and err.value.code == "removal_in_progress"
    got = await rh.row(row.id)
    assert cast(dict, got.removal)["marker"] == "x"
    assert cast(dict, got.removal)["legacy_identity"]["user_id"] == stranger


async def test_fix5_the_remove_only_enqueues_and_the_tick_runs_it(rh: RHarness) -> None:
    row = await approved(rh)
    r = await remove(rh, row, run=False)
    assert r.status_code == 202, r.text
    assert r.json()["approval"]["removal"]["state"] == "pending"
    assert r.json()["approval"]["removal"]["running"] is False
    got = await rh.row(row.id)
    assert state(got) == "pending" and cast(dict, got.removal)["steps"] == {}
    assert got.steps_lease_token is None and got.steps_next_attempt_at is not None
    assert (await tenant(rh, row.tenant_id)).status == "active"
    assert rh.lic.revoke_calls == []
    assert await drive(rh) == 1
    got = await rh.row(row.id)
    assert state(got) == "done" and cast(dict, got.removal)["attempts"] == 1


async def test_fix5_a_retry_only_enqueues(rh: RHarness) -> None:
    row = await approved(rh)
    rh.lic.list_error = LicensesClientError("down", status_code=503)
    await remove(rh, row)
    rh.lic.list_error = None
    async with rh.client() as c:
        r = await c.post(f"/api/admin/signup-approvals/{row.id}/removal/retry")
    assert r.status_code == 202
    got = await rh.row(row.id)
    assert state(got) == "pending" and cast(dict, got.removal)["attempts"] == 0
    assert rstep(got, "licences")["status"] == "pending"  # nothing ran yet
    assert await drive(rh) == 1
    assert state(await rh.row(row.id)) == "done"


async def test_fix5_a_long_step_keeps_renewing_the_lease(rh: RHarness, monkeypatch) -> None:
    monkeypatch.setattr(removal, "LEASE_RENEW_SECONDS", 0.05)
    row = await approved(rh)
    await remove(rh, row, run=False)
    rh.lic.list_gate = asyncio.Event()
    task = asyncio.create_task(drive(rh))
    try:
        for _ in range(100):  # parked inside the licences step
            await asyncio.sleep(0.02)
            got = await rh.row(row.id)
            if "close_access" in cast(dict, got.removal)["steps"]:
                break
        assert (await rview(rh, row))["running"] is True
        first = cast(datetime.datetime, (await rh.row(row.id)).steps_lease_until)
        await asyncio.sleep(0.4)
        later = cast(datetime.datetime, (await rh.row(row.id)).steps_lease_until)
        assert later > first
    finally:
        rh.lic.list_gate.set()
        await task
    assert state(await rh.row(row.id)) == "done"


async def test_fix6_a_forced_admin_job_is_never_resumed(rh: RHarness) -> None:
    row = await approved(rh)
    forced = uuid7()
    async with rh.sf() as s:
        s.add(
            TenantOffboarding(
                id=forced,
                tenant_id=row.tenant_id,
                kind="delete",
                status="failed",
                force=True,
                progress={},
            )
        )
        await s.commit()
    await remove(rh, row)
    got = await rh.row(row.id)
    assert state(got) == "blocked"
    rec = rstep(got, "tenant_delete")
    assert rec["status"] == "blocked" and rec["code"] == removal.FORCED_JOB_EXISTS
    assert cast(dict, got.removal)["blockers"][0]["code"] == removal.FORCED_JOB_EXISTS
    [job] = await jobs(rh, row.tenant_id, "delete")
    assert job.id == forced and job.status == "failed"
    assert (await tenant(rh, row.tenant_id)).status == "suspended"


#: The second review's inputs no sanitiser catches completely.
REVIEWER_SECRETS: tuple[str, ...] = (
    "https://kc.example/realms/portal/login-actions/action-token?key%3DSECRETURLENC1",
    "token%3DSECRETURLENC2",
    "invite_token=SECRETINVITE3",
    '{"token": "SECRETJSON4"}',
    "https://kc.example/invite/SECRETPATH5/accept",
    "Authorization: Bearer eyJhbGciOi.eyJzdWIi.SECRETJWT6",
)
SECRET_NEEDLES: tuple[str, ...] = (
    "secreturlenc1",
    "secreturlenc2",
    "secretinvite3",
    "secretjson4",
    "secretpath5",
    "secretjwt6",
    "eyjhbgcioi",
)


async def _everything_stored(rh: RHarness, row: SignupApproval) -> str:
    """Every DB row the removal could have written to, flattened, lower-cased."""
    async with rh.sf() as s:
        parts: list[str] = []
        for table in ("signup_approvals", "admin_audit_log", "request_audit"):
            rows = (await s.execute(text(f"SELECT to_jsonb(t)::text FROM {table} t"))).scalars()  # noqa: S608
            parts.extend(rows)
    async with rh.client() as c:
        parts.append((await c.get(f"/api/admin/signup-approvals/{row.id}")).text)
    return "\n".join(parts).lower()


@pytest.mark.parametrize(
    "raw", ["https://kc.example/realms/portal/login-actions/action-token?key%3DSECRETURLENC1"]
)
async def test_fix8_evidence_is_stored_as_a_fingerprint_only(rh: RHarness, raw: str) -> None:
    row, uid = await legacy(rh)
    evidence = f"invite to {EMAIL.upper()} — {raw}"
    r = await remove(rh, row, run=False, legacy_identity={"user_id": uid, "evidence": evidence})
    assert r.status_code == 202, r.text
    stored = cast(dict, (await rh.row(row.id)).removal)["legacy_identity"]
    assert set(stored) == {"user_id", "evidence_sha256", "evidence_length", "evidence_created_at"}
    assert stored["evidence_sha256"] == hashlib.sha256(evidence.encode()).hexdigest()
    assert stored["evidence_length"] == len(evidence)
    view = r.json()["approval"]["removal"]["legacy_identity"]
    assert view["evidence_sha256_prefix"] == stored["evidence_sha256"][:12]
    assert "evidence" not in view and "evidence_sha256" not in view
    flat = await _everything_stored(rh, row)
    for needle in SECRET_NEEDLES:
        assert needle not in flat, needle
    # …the whole run, and an amend on Retry, never store it either
    await drive(rh)
    assert state(await rh.row(row.id)) == "done"
    flat = await _everything_stored(rh, row)
    for needle in (*SECRET_NEEDLES, EMAIL):
        assert needle not in flat, needle


async def test_fix8_an_amended_evidence_is_a_fingerprint_too(rh: RHarness) -> None:
    row, uid = await legacy(rh)
    stranger = rh.idp.seed("someone@else.example")
    await remove(rh, row, **legacy_body(stranger))
    assert state(await rh.row(row.id)) == "blocked"
    evidence = " ".join(REVIEWER_SECRETS)
    r = await retry(rh, row, {"legacy_identity": {"user_id": uid, "evidence": evidence}})
    assert r.status_code == 202, r.text
    assert state(await rh.row(row.id)) == "done"
    flat = await _everything_stored(rh, row)
    for needle in SECRET_NEEDLES:
        assert needle not in flat, needle
    async with rh.sf() as s:
        amended = (
            await s.execute(
                select(AdminAuditLog.detail).where(
                    AdminAuditLog.detail["phase"].astext == "identity_amended"
                )
            )
        ).scalar_one()
    assert amended["legacy_evidence_sha256"] == hashlib.sha256(evidence.encode()).hexdigest()


async def test_fix8_no_sanitiser_is_left() -> None:
    assert not hasattr(removal, "sanitise_evidence")
    assert not hasattr(removal, "audit_evidence")


# ─── a named legacy login that does not exist ────────────────────────────


async def test_fix9_an_absent_named_login_blocks_while_the_address_has_a_login(
    rh: RHarness,
) -> None:
    row, uid = await legacy(rh)
    ghost = str(uuid7())
    await remove(rh, row, **legacy_body(ghost))
    await _legacy_blocked(rh, row, uid, removal.LEGACY_IDENTITY_MISMATCH)
    assert rh.idp.users[uid].get("enabled", True) is not False


async def test_fix9_the_identity_step_itself_refuses_an_absent_named_login(
    rh: RHarness, monkeypatch
) -> None:
    """Even past the blockers (the login vanished in between, or a future
    blockers change), the identity step never records ``absent`` while the
    address still has a login."""
    row, uid = await legacy(rh)
    ghost = str(uuid7())

    async def trusted(ctx):  # type: ignore[no-untyped-def]
        return UUID(uid), None

    monkeypatch.setattr(removal, "_legacy_requester", trusted)
    await remove(rh, row, **legacy_body(ghost))
    got = await rh.row(row.id)
    assert state(got) == "blocked"
    rec = rstep(got, "identity")
    assert rec["status"] == "blocked" and rec["code"] == removal.LEGACY_IDENTITY_MISMATCH
    assert uid in rh.idp.users and rh.idp.deleted == [] and rh.idp.state_writes == []
    assert "tenant_delete" not in cast(dict, got.removal)["steps"]
    assert got.email == EMAIL


async def test_fix9_an_absent_named_login_with_no_login_for_the_address_is_absent(
    rh: RHarness,
) -> None:
    row, uid = await legacy(rh)
    del rh.idp.users[uid]
    await remove(rh, row, **legacy_body(str(uuid7())))
    got = await rh.row(row.id)
    assert state(got) == "done"
    assert rstep(got, "identity")["code"] == "absent"


# ─── the removal worker ──────────────────────────────────────────────────


async def test_fix10_removals_run_even_when_the_steps_pass_fails(rh: RHarness, monkeypatch) -> None:
    from src.observability.metrics import db_errors_total

    row = await approved(rh)
    await remove(rh, row, run=False)

    async def boom(*_a, **_k):  # type: ignore[no-untyped-def]
        raise RuntimeError("steps pass down")

    async def deps() -> removal.RemovalDeps:
        return rh.rdeps

    monkeypatch.setattr(core, "run_tick", boom)
    monkeypatch.setattr(signup_tick, "tick_deps", lambda: rh.deps)
    monkeypatch.setattr(signup_tick, "removal_deps", deps)
    before = db_errors_total.labels(op="signup_tick_steps")._value.get()
    assert await signup_tick.signup_tick_once(rh.sf) == 1
    assert state(await rh.row(row.id)) == "done"
    assert db_errors_total.labels(op="signup_tick_steps")._value.get() == before + 1


async def test_fix10_a_failing_removal_pass_is_counted_and_never_raises(
    rh: RHarness, monkeypatch
) -> None:
    from src.observability.metrics import db_errors_total

    async def broken() -> removal.RemovalDeps:
        raise RuntimeError("storage down")

    monkeypatch.setattr(signup_tick, "tick_deps", lambda: rh.deps)
    monkeypatch.setattr(signup_tick, "removal_deps", broken)
    before = db_errors_total.labels(op="signup_tick_removals")._value.get()
    assert await signup_tick.signup_tick_once(rh.sf) == 0
    assert db_errors_total.labels(op="signup_tick_removals")._value.get() == before + 1


async def test_fix10_start_and_retry_refuse_when_the_tick_is_disabled(
    rh: RHarness, monkeypatch
) -> None:
    row = await approved(rh)
    blocked = await approved(rh, "bob@other.example")
    rh.lic.list_error = LicensesClientError("down", status_code=503)
    await remove(rh, blocked)
    rh.lic.list_error = None
    monkeypatch.setenv("W2V_SIGNUP_TICK_INTERVAL_SECONDS", "0")
    r = await remove(rh, row, run=False)
    assert r.status_code == 503 and r.json()["error"] == "removal_worker_disabled"
    assert (await rh.row(row.id)).status == "approved"
    before = await rh.row(blocked.id)
    async with rh.client() as c:
        r = await c.post(f"/api/admin/signup-approvals/{blocked.id}/removal/retry")
    assert r.status_code == 503 and r.json()["error"] == "removal_worker_disabled"
    assert (await rh.row(blocked.id)).updated_at == before.updated_at


async def test_fix10_a_removal_queued_for_over_15_minutes_is_stale(rh: RHarness) -> None:
    row = await approved(rh)
    r = await remove(rh, row, run=False)
    assert r.json()["approval"]["removal"]["stale"] is False

    async def due_ago(minutes: int) -> None:
        async with rh.sf() as s:
            await s.execute(
                update(SignupApproval)
                .where(SignupApproval.id == row.id)
                .values(steps_next_attempt_at=func.now() - datetime.timedelta(minutes=minutes))
            )
            await s.commit()

    await due_ago(14)
    assert (await rview(rh, row))["stale"] is False
    await due_ago(16)
    assert (await rview(rh, row))["stale"] is True
    async with rh.client() as c:
        view = (await c.get(f"/api/admin/signup-approvals/{row.id}")).json()["data"]
    assert view["removal"]["stale"] is True
    await drive(rh)
    assert (await rview(rh, row))["stale"] is False  # done


# ─── the slug tombstone ──────────────────────────────────────────────────


async def test_fix11_finalize_tombstones_the_company_derived_slug(rh: RHarness) -> None:
    row = await approved(rh)
    slug = cast(str, row.tenant_slug)
    assert row.steps["provision"]["detail"]["tenant_slug"] == slug
    await remove(rh, row)
    done = await rh.row(row.id)
    assert state(done) == "done"
    tomb = f"removed-{str(row.id)[:8]}"
    assert cast(dict, done.removal)["tenant_slug"] == tomb
    assert done.steps["provision"]["detail"]["tenant_slug"] == tomb
    # referential integrity: the tenant id is kept everywhere
    assert done.tenant_id == row.tenant_id
    assert done.steps["provision"]["detail"]["tenant_id"] == str(row.tenant_id)
    assert cast(dict, done.removal)["tenant_id"] == str(row.tenant_id)
    assert (await rview(rh, row))["tenant_slug"] == tomb
    async with rh.sf() as s:
        audits = list(
            (
                await s.execute(
                    select(AdminAuditLog.detail).where(AdminAuditLog.action == "signup.remove")
                )
            ).scalars()
        )
    assert audits and all(d.get("tenant_slug") in (None, tomb) for d in audits)


async def test_fix11_resolve_tombstones_the_slug_too(rh: RHarness) -> None:
    row = await approved(rh)
    async with rh.sf() as s:
        await MembersRepository(s).add_member(
            subject_id=uuid7(), tenant_id=cast(UUID, row.tenant_id), role="member"
        )
        await s.commit()
    await remove(rh, row)
    assert state(await rh.row(row.id)) == "blocked"
    r = await post(
        rh,
        f"/api/admin/signup-approvals/{row.id}/removal/resolve",
        {"note": "kept by hand", "acknowledge_live_access": True},
    )
    assert r.status_code == 200, r.text
    got = await rh.row(row.id)
    tomb = f"removed-{str(row.id)[:8]}"
    assert cast(dict, got.removal)["tenant_slug"] == tomb
    assert got.steps["provision"]["detail"]["tenant_slug"] == tomb
    assert got.tenant_slug is None and got.tenant_id == row.tenant_id


async def test_fix12_a_w2v_signup_app_is_not_foreign(rh: RHarness) -> None:
    row, uid = await legacy(rh)
    rh.idp.users[uid]["attributes"]["signup_app"] = ["w2v"]
    await remove(rh, row, **legacy_body(uid))
    assert state(await rh.row(row.id)) == "done"


# ─── third review ────────────────────────────────────────────────────────


def export_keys(rh: RHarness, row: SignupApproval) -> list[str]:
    from src.services import tenant_offboarding as offboarding

    prefix = offboarding.export_prefix(cast(UUID, row.tenant_id))
    return sorted(k for k in rh.storage.objects if k.startswith(prefix))


async def test_r3_finalize_deletes_the_export_archive(rh: RHarness) -> None:
    """The export archive holds the signup's name / address / company: finalize
    deletes it (and records the keys) before claiming "personal data erased"."""
    row = await approved(rh)
    await remove(rh, row)
    done = await rh.row(row.id)
    assert state(done) == "done"
    [export_job] = await jobs(rh, row.tenant_id, "export")
    from src.services import tenant_offboarding as offboarding

    key = offboarding.export_key(cast(UUID, row.tenant_id), export_job.id)
    assert export_keys(rh, row) == []
    assert rstep(done, "finalize")["detail"]["export_keys_deleted"] == [key]
    [notice] = await removal_notices(rh)
    assert "export archive deleted" in notice and "personal data erased" in notice
    async with rh.sf() as s:
        completed = (
            await s.execute(
                select(AdminAuditLog.detail).where(
                    AdminAuditLog.action == "signup.remove",
                    AdminAuditLog.detail["phase"].astext == "completed",
                )
            )
        ).scalar_one()
    assert completed["export_keys_deleted"] == [key]


async def test_r3_an_export_purge_failure_is_retried_never_done(rh: RHarness) -> None:
    from src.services import tenant_offboarding as offboarding

    row = await approved(rh)
    prefix = offboarding.export_prefix(cast(UUID, row.tenant_id))
    # 1) the storage refuses the delete once → pending, nothing erased
    rh.storage.fail_prefix_once = prefix
    await remove(rh, row)
    got = await rh.row(row.id)
    assert state(got) == "pending"
    assert rstep(got, "finalize")["code"] == removal.EXPORT_PURGE_FAILED
    assert cast(dict, got.removal)["last_error"]["code"] == removal.EXPORT_PURGE_FAILED
    assert got.email == EMAIL and len(export_keys(rh, row)) == 1
    assert rstep(got, "tenant_delete")["status"] == "ok"
    # 2) storage unavailable → still pending
    real = rh.storage
    cast(Any, rh).storage = None
    await make_due(rh, row.id)
    await drive(rh)
    got = await rh.row(row.id)
    assert state(got) == "pending" and got.email == EMAIL
    assert rstep(got, "finalize")["code"] == removal.EXPORT_PURGE_FAILED
    assert await removal_notices(rh) == []
    # 3) storage back → done, archive gone
    rh.storage = real
    await make_due(rh, row.id)
    await drive(rh)
    got = await rh.row(row.id)
    assert state(got) == "done" and export_keys(rh, row) == []
    assert got.email == f"erased+{row.id}@invalid"


async def test_r3_blocked_and_failed_notices_carry_no_personal_data(rh: RHarness) -> None:
    row = await approved(rh)
    async with rh.sf() as s:
        await MembersRepository(s).add_member(
            subject_id=uuid7(), tenant_id=cast(UUID, row.tenant_id), role="member"
        )
        await s.commit()
    await remove(rh, row)
    assert state(await rh.row(row.id)) == "blocked"
    from src.services.signup import telegram as tg

    await tg.drain_telegram_for_tests()
    [payload] = [p for _u, p in rh.sent.messages if "workspace removal" in str(p["text"])]
    text_ = str(payload["text"])
    assert "BLOCKED" in text_ and str(row.id) in text_
    assert f"removed-{str(row.id)[:8]}" in text_
    assert EMAIL not in text_ and "Ann Example" not in text_
    assert str(row.tenant_slug) not in text_
    assert "Open in Studio" in str(payload.get("reply_markup"))
    # the dataclass cannot carry them at all
    assert not {"email", "name"} & set(tg.RemovalNotice.__dataclass_fields__)


async def test_r3_licences_of_another_tenant_block_with_zero_revokes(rh: RHarness) -> None:
    row = await approved(rh)
    rh.lic.extra_listed = [LicenseView(id="foreign-lic", tenant_id=str(uuid7()), status="active")]
    await remove(rh, row)
    got = await rh.row(row.id)
    assert state(got) == "blocked"
    assert rstep(got, "licences")["code"] == removal.TENANT_MISMATCH
    assert cast(dict, got.removal)["blockers"][0]["licences"] == ["foreign-lic"]
    assert rh.lic.revoke_calls == [] and rh.lic.revoked == set()
    assert await jobs(rh, row.tenant_id, "delete") == []


async def test_r3_the_claim_rechecks_the_backoff(rh: RHarness) -> None:
    row = await approved(rh)
    rh.lic.list_error = LicensesClientError("down", status_code=503)
    await remove(rh, row)
    got = await rh.row(row.id)
    assert state(got) == "pending"
    assert got.steps_next_attempt_at is not None
    assert got.steps_next_attempt_at > datetime.datetime.now(datetime.UTC)
    attempts = cast(dict, got.removal)["attempts"]
    # A runner that listed it as due earlier must NOT claim it inside the backoff.
    run = await removal.run_removal(rh.sf, rh.rdeps, row.id)
    assert run.ran is False and run.reason == "lease_busy_or_not_runnable"
    assert cast(dict, (await rh.row(row.id)).removal)["attempts"] == attempts
    await make_due(rh, row.id)
    assert (await removal.run_removal(rh.sf, rh.rdeps, row.id)).ran is True


async def unsent(
    rh: RHarness, *, created_offset: datetime.timedelta | None = None
) -> tuple[SignupApproval, str]:
    """No identity step AND the invite_email step never landed: no login has to
    be named — but one may exist (the invite created it, then failed to send)."""
    row, uid = await legacy(rh, created_offset=created_offset)
    steps = dict(row.steps)
    steps["invite_email"] = {**steps["invite_email"], "status": "failed"}
    async with rh.sf() as s:
        await s.execute(
            update(SignupApproval).where(SignupApproval.id == row.id).values(steps=steps)
        )
        await s.commit()
    got = await rh.row(row.id)
    assert not removal.needs_legacy_identity(got)
    return got, uid


async def test_r3_an_unnamed_login_made_in_the_approval_window_is_retired(rh: RHarness) -> None:
    row, uid = await unsent(rh)
    r = await remove(rh, row)
    assert r.status_code == 202, r.text
    got = await rh.row(row.id)
    assert state(got) == "done"
    assert rstep(got, "identity")["code"] == "deleted"
    assert rstep(got, "identity")["detail"]["mode"] == "unnamed"
    assert uid not in rh.idp.users


async def test_r3_an_unnamed_credentialed_login_is_disabled(rh: RHarness) -> None:
    row, uid = await unsent(rh)
    rh.idp.users[uid]["credentials"] = [{"type": "password"}]
    await remove(rh, row)
    got = await rh.row(row.id)
    assert rstep(got, "identity")["code"] == "disabled"
    assert rh.idp.users[uid]["enabled"] is False


async def test_r3_an_unnamed_login_stamped_for_this_row_is_retired(rh: RHarness) -> None:
    row, uid = await unsent(rh, created_offset=-datetime.timedelta(days=3))
    rh.idp.users[uid]["attributes"]["w2v_signup_id"] = [str(row.id)]
    await remove(rh, row)
    assert rstep(await rh.row(row.id), "identity")["code"] == "deleted"
    assert uid not in rh.idp.users


@pytest.mark.parametrize("why", ["window"])
async def test_r3_an_unnamed_login_that_is_not_this_signups_is_kept(rh: RHarness, why) -> None:
    offset = -datetime.timedelta(days=3) if why == "window" else None
    row, uid = await unsent(rh, created_offset=offset)
    if why == "foreign":
        rh.idp.users[uid]["attributes"]["member_tenant_ids"] = [str(uuid7())]
    await remove(rh, row)
    got = await rh.row(row.id)
    assert state(got) == "done"
    assert rstep(got, "identity")["code"] == removal.IDENTITY_NOT_OWNED
    assert uid in rh.idp.users and rh.idp.deleted == [] and rh.idp.state_writes == []


async def test_r3_no_login_for_the_address_is_no_identity(rh: RHarness) -> None:
    row, uid = await unsent(rh)
    del rh.idp.users[uid]
    await remove(rh, row)
    got = await rh.row(row.id)
    assert rstep(got, "identity")["code"] == "no_identity"
    assert rstep(got, "identity")["detail"]["looked_up"] is True


# ─── shared DEPENDENCY licences ──────────────────────────────────────────

DEP_APPS = (
    "kag",
    "workers",
    "evidence-review",
    "mlx-backend",
    "mlx-ingestion-backend",
    "inference-router",
)


def add_dependencies(rh: RHarness, row: SignupApproval) -> list[str]:
    """The 6 dependency licences the w2v bundle ensure creates live."""
    ids = []
    for app in DEP_APPS:
        lid = f"dep-{app}"
        rh.lic.holders[lid] = (str(row.tenant_id), app, f"svc:{row.tenant_id}:{app}")
        rh.lic.dependencies.add(lid)
        ids.append(lid)
    return ids


def still_required(*required_by: dict[str, str]) -> LicensesClientError:
    return LicensesClientError(
        "revoke_license returned 409",
        status_code=409,
        code=removal.DEPENDENCY_STILL_REQUIRED,
        detail={"error": removal.DEPENDENCY_STILL_REQUIRED, "required_by": list(required_by)},
    )


async def test_dependencies_released_by_the_cascade_are_never_revoked_by_w2v(
    rh: RHarness,
) -> None:
    row = await approved(rh)
    own = tenant_licences(rh, row)
    deps = add_dependencies(rh, row)
    rh.lic.cascade = True
    # w2v may not revoke a dependency: an explicit revoke would be refused.
    for lid in deps:
        rh.lic.dep_revoke_error[lid] = LicensesClientError("403", status_code=403)
    await remove(rh, row)
    got = await rh.row(row.id)
    assert state(got) == "done", got.removal
    assert set(rh.lic.revoke_calls) == own  # never a dependency
    detail = rstep(got, "licences")["detail"]
    assert detail["dependencies"] == sorted(deps)
    assert detail["released_dependencies"] == sorted(deps)
    assert "dependencies_live_after" not in detail
    assert rh.lic.revoked == own | set(deps)
    # the archive ran, so introspect answers tenant_archived for the dependencies too
    assert rh.lic.archived == {str(row.tenant_id): "2026-10-21T00:00:00Z"}


@pytest.mark.parametrize("expose", [True])
async def test_a_dependency_left_active_gets_one_explicit_owner_revoke(
    rh: RHarness, expose: bool
) -> None:
    row = await approved(rh)
    own = tenant_licences(rh, row)
    deps = add_dependencies(rh, row)
    rh.lic.expose_is_dependency = expose  # else only metadata.dependency_of marks them
    await remove(rh, row)
    got = await rh.row(row.id)
    assert state(got) == "done", got.removal
    # own licences first, then each dependency exactly once
    assert set(rh.lic.revoke_calls[: len(own)]) == own
    assert sorted(rh.lic.revoke_calls[len(own) :]) == sorted(deps)
    detail = rstep(got, "licences")["detail"]
    assert sorted(detail["released_explicitly"]) == sorted(deps)
    assert detail["dependencies_live_after"] == sorted(deps)


async def test_a_dependency_still_required_on_this_tenant_blocks(rh: RHarness) -> None:
    row = await approved(rh)
    deps = add_dependencies(rh, row)
    tid = str(row.tenant_id)
    rh.lic.dep_revoke_error["dep-kag"] = still_required(
        {"license_id": "lic-not-ours", "tenant_id": tid, "app_key": "evidence-portal"}
    )
    await remove(rh, row)
    got = await rh.row(row.id)
    assert state(got) == "blocked"
    rec = rstep(got, "licences")
    assert (rec["status"], rec["code"]) == ("blocked", removal.DEPENDENCY_STILL_REQUIRED)
    [blocker] = cast(dict, got.removal)["blockers"]
    assert blocker["code"] == removal.DEPENDENCY_STILL_REQUIRED
    assert blocker["apps"] == ["kag"] and blocker["required_by"] == ["evidence-portal"]
    assert cast(dict, got.removal)["last_error"]["code"] == removal.DEPENDENCY_STILL_REQUIRED
    # the other dependencies were released; nothing after the licences step ran
    assert rh.lic.revoked >= set(deps) - {"dep-kag"} and "dep-kag" not in rh.lic.revoked
    assert set(cast(dict, got.removal)["steps"]) == {"blockers", "close_access", "licences"}
    assert got.steps_next_attempt_at is None


@pytest.mark.parametrize("who", ["unnamed"])
async def test_who_still_requires_a_dependency_decides(rh: RHarness, who: str) -> None:
    row = await approved(rh)
    add_dependencies(rh, row)
    required_by = {
        "unnamed": [],
        "other_tenant": [{"license_id": "x", "tenant_id": str(uuid7()), "app_key": "kag"}],
        "ours": [{"license_id": "dep-kag", "tenant_id": str(row.tenant_id), "app_key": "kag"}],
    }[who]
    rh.lic.dep_revoke_error["dep-workers"] = still_required(*required_by)
    await remove(rh, row)
    got = await rh.row(row.id)
    rec = rstep(got, "licences")
    if who == "unnamed":  # licences did not say who: the operator decides
        assert state(got) == "blocked" and rec["code"] == removal.DEPENDENCY_STILL_REQUIRED
    elif who == "other_tenant":  # kept for another tenant; the archive retires it here
        assert state(got) == "done", got.removal
        [kept] = rec["detail"]["kept_for_other_tenants"]
        assert kept["license_id"] == "dep-workers"
    else:  # one of OUR dependencies needs it: retried, not blocked
        assert state(got) == "pending"
        assert (rec["status"], rec["code"]) == ("pending", removal.LICENCE_FAILED)
        assert [w["license_id"] for w in rec["detail"]["waiting"]] == ["dep-workers"]


async def test_a_licences_that_cannot_release_dependencies_is_retryable(rh: RHarness) -> None:
    row = await approved(rh)
    deps = add_dependencies(rh, row)
    for lid in deps:  # the live 403 app_not_issuable_by_caller
        rh.lic.dep_revoke_error[lid] = LicensesClientError(
            "revoke_license returned 403", status_code=403, code="app_not_issuable_by_caller"
        )
    await remove(rh, row)
    got = await rh.row(row.id)
    assert state(got) == "pending" and got.steps_next_attempt_at is not None
    rec = rstep(got, "licences")
    assert (rec["status"], rec["code"]) == ("pending", removal.DEPENDENCY_RELEASE_UNSUPPORTED)
    assert len(rec["detail"]["unsupported"]) == len(deps)
    assert cast(dict, got.removal)["last_error"]["code"] == removal.DEPENDENCY_RELEASE_UNSUPPORTED
    # licences upgraded (the cascade) → the next due run finishes it
    rh.lic.dep_revoke_error.clear()
    await make_due(rh, row.id)
    assert await removal.run_due_removals(rh.sf, rh.rdeps) == 1
    assert state(await rh.row(row.id)) == "done"


async def test_a_failed_removal_can_be_retried_once_licences_is_upgraded(rh: RHarness) -> None:
    row = await approved(rh)
    deps = add_dependencies(rh, row)
    for lid in deps:
        rh.lic.dep_revoke_error[lid] = LicensesClientError("403", status_code=403)
    await remove(rh, row)
    for _ in range(core.STEPS_MAX_ATTEMPTS - 1):
        await make_due(rh, row.id)
        assert await removal.run_due_removals(rh.sf, rh.rdeps) == 1
    got = await rh.row(row.id)
    assert state(got) == "failed"
    assert cast(dict, got.removal)["last_error"]["code"] == removal.DEPENDENCY_RELEASE_UNSUPPORTED
    # the tick never runs a failed removal
    await make_due(rh, row.id)
    assert await removal.run_due_removals(rh.sf, rh.rdeps) == 0
    # licences now releases dependencies; the super_admin Retry finishes it
    rh.lic.dep_revoke_error.clear()
    r = await retry(rh, row)
    assert r.status_code == 202, r.text
    got = await rh.row(row.id)
    assert state(got) == "done", got.removal
    assert rh.lic.revoked >= set(deps)
