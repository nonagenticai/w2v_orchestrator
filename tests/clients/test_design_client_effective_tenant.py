"""The design job html_render creates must be stamped with the RUN's tenant.

THE DEFECT. ``DesignClient`` authenticates with the orchestrator's
service-account licence (``X-Tenant-Id: Bearer <token>``). html_render
introspects that to the SA tenant and writes THAT onto the ``redesign_jobs``
row, then scopes every read of the row by it — so a customer's render was
stored under ``01a1113b-674f-72b0-a84b-dc746d39afca`` and the requesting
tenant's own fetch 404'd forever.

NON-VACUITY. These tests do NOT assert on a stub's captured kwargs. They run
the REAL ``httpx`` request pipeline (``MockTransport``) and assert on the bytes
that actually leave the process: the outbound headers html_render's
AuthMiddleware will read. Delete the ``_get_client`` override in
``design_client`` and ``test_the_submit_hop_names_the_run_tenant`` fails.
"""

from __future__ import annotations

import uuid

import httpx
import pytest
from w2v_common.auth.identity_context import (
    clear_identity_context,
    set_identity_context,
)
from w2v_common.tenant import TenantSource, UnvouchedTenantError

from src.clients.design_client import DesignClient

_RUN_TENANT = uuid.UUID("e4f70a83-81ee-429a-abca-6999e3a0d24f")
#: A header VALUE, never used to authenticate against anything.
_SA_TOKEN = "sa-license-token-not-a-jwt"


@pytest.fixture(autouse=True)
def _clean_identity():
    clear_identity_context()
    yield
    clear_identity_context()


async def _client(sink: list[httpx.Request]) -> DesignClient:
    """A real, fully-wired client whose only fake part is the socket.

    Deliberately NOT ``BaseServiceClient.set_transport``: that seam builds a
    BARE ``httpx.AsyncClient`` with no default headers and no ``event_hooks``,
    so a test using it cannot see the outbound identity headers at all — which
    is precisely what is under test. Building the production client and then
    swapping only its transport keeps every header and both shared hooks.
    """
    client = DesignClient(base_url="http://html-render.test:6319", license_token=_SA_TOKEN)

    def handler(request: httpx.Request) -> httpx.Response:
        sink.append(request)
        return httpx.Response(200, json={"job_id": "job-1", "status": "pending"})

    inner = await client._get_client()
    inner._transport = httpx.MockTransport(handler)
    return client


@pytest.mark.asyncio
async def test_the_submit_hop_names_the_run_tenant() -> None:
    """The write path. This is the whole bug: the job row's tenant comes from here."""
    set_identity_context(
        tenant_id=_RUN_TENANT,
        license_id="lic",
        subject_id="sub",
        tenant_source=TenantSource.LICENSE_TOKEN,
    )
    sent: list[httpx.Request] = []
    client = await _client(sent)
    try:
        await client.start_design(url="https://example.com", scope="redesign")
    finally:
        await client.close()

    assert sent, "no request left the client"
    request = sent[0]
    assert request.url.path == "/api/designs"
    # The row html_render inserts is scoped to THIS value.
    assert request.headers["X-Effective-Tenant-Id"] == str(_RUN_TENANT)
    # …and the SA token it authenticates us with is untouched.
    assert request.headers["X-Tenant-Id"] == f"Bearer {_SA_TOKEN}"


@pytest.mark.asyncio
async def test_the_read_hop_is_scoped_to_the_same_tenant() -> None:
    """A row written under the run tenant is only readable under it — same header."""
    set_identity_context(
        tenant_id=_RUN_TENANT,
        license_id="lic",
        subject_id="sub",
        tenant_source=TenantSource.LICENSE_TOKEN,
    )
    sent: list[httpx.Request] = []
    client = await _client(sent)
    try:
        await client.get_design_status("job-1")
    finally:
        await client.close()

    assert sent[0].headers["X-Effective-Tenant-Id"] == str(_RUN_TENANT)


@pytest.mark.asyncio
async def test_an_unvouched_tenant_is_never_delegated() -> None:
    """Fail closed. A tenant of unrecorded provenance must not be acted as."""
    set_identity_context(tenant_id=_RUN_TENANT, license_id="lic", subject_id="sub")
    sent: list[httpx.Request] = []
    client = await _client(sent)
    try:
        with pytest.raises(UnvouchedTenantError):
            await client.start_design(url="https://example.com")
    finally:
        await client.close()
    assert not sent, "an unvouched hop must not reach the wire"


@pytest.mark.asyncio
async def test_a_tenantless_call_still_goes_out_on_the_service_account() -> None:
    """``design_job_exists`` runs outside a run task — it must keep working."""
    sent: list[httpx.Request] = []
    client = await _client(sent)
    try:
        await client.get_design_status("job-1")
    finally:
        await client.close()

    assert "X-Effective-Tenant-Id" not in sent[0].headers
    assert sent[0].headers["X-Tenant-Id"] == f"Bearer {_SA_TOKEN}"
