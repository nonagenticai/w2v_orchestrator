"""``ensure_tenant_licences`` / ``ensure_active_license``: the tenant's w2v
licence is TENANT-level, held by ``svc:<tenant>:w2v`` — never by a person.

The fake (``tests/_tenant_licences_fake.py``) models licences' bundle ensure and
records the HOLDER of every licence, so these tests fail if the helper goes back
to minting ``POST /licenses/user/{person}``.
"""

from __future__ import annotations

from typing import cast
from uuid import UUID, uuid4

import httpx
import pytest

from src.clients.licenses_client import LicensesClient, LicensesClientError
from src.clients.schemas.licenses import PlanView
from src.services.license_provisioning import (
    NOT_CONFIRMED_AFTER_TIMEOUT,
    TenantLicenceRefusedError,
    ensure_active_license,
    ensure_tenant_licences,
    tenant_licence_subject,
)
from tests._tenant_licences_fake import TenantLicencesFake

pytestmark = pytest.mark.asyncio


def _client(fake: TenantLicencesFake) -> LicensesClient:
    return cast(LicensesClient, fake)


async def test_the_tenant_licence_is_held_by_the_tenant_service_subject() -> None:
    fake, tenant = TenantLicencesFake(), uuid4()
    license_id = await ensure_active_license(_client(fake), app_key="w2v", tenant_id=tenant)
    assert fake.holder_of(license_id) == f"svc:{tenant}:w2v"
    # Every licence licences now holds for this tenant is tenant-level.
    assert {h for (_t, _a, h) in fake.holders.values()} == {
        f"svc:{tenant}:w2v",
        f"svc:{tenant}:w2v-image",
    }
    [call] = fake.ensure_calls
    assert call["bundle"] == "w2v" and call["tenant_id"] == str(tenant)


async def test_ensure_is_idempotent_and_never_mints_twice() -> None:
    fake, tenant = TenantLicencesFake(), uuid4()
    first = await ensure_tenant_licences(_client(fake), tenant_id=tenant)
    second = await ensure_tenant_licences(_client(fake), tenant_id=tenant)
    assert first.license_ids == second.license_ids
    assert len(fake.holders) == 2  # w2v + w2v-image, once


async def test_plan_preference_is_the_w2v_plans_then_licences_default() -> None:
    fake = TenantLicencesFake()  # p1 revoked, p2 live
    await ensure_tenant_licences(_client(fake), tenant_id=uuid4())
    assert fake.ensure_calls[0]["plans"] == ["p2", "standard"]


async def test_plan_preference_never_duplicates_standard() -> None:
    fake = TenantLicencesFake(
        plans=[
            PlanView(id="a", app_id="app-w2v", code="standard", name="S"),
            PlanView(id="b", app_id="app-w2v", code="perpetual", name="P"),
        ]
    )
    await ensure_tenant_licences(_client(fake), tenant_id=uuid4())
    assert fake.ensure_calls[0]["plans"] == ["standard", "perpetual"]


async def test_a_legacy_person_held_licence_is_refused_not_reused() -> None:
    """A tenant still holding a member's w2v licence: licences refuses
    ``member_licence`` and w2v raises — it never adopts or re-mints for a person."""
    fake, tenant = TenantLicencesFake(), uuid4()
    await fake.create_license(subject_id=uuid4(), app_id="app-w2v", plan_id="p", tenant_id=tenant)
    with pytest.raises(TenantLicenceRefusedError) as info:
        await ensure_active_license(_client(fake), app_key="w2v", tenant_id=str(tenant))
    assert info.value.reason == "member_licence"
    assert info.value.app_key == "w2v"


async def test_an_app_outside_the_bundle_is_refused() -> None:
    fake = TenantLicencesFake(bundle_apps=("w2v",))
    with pytest.raises(TenantLicenceRefusedError) as info:
        await ensure_active_license(_client(fake), app_key="w2v-image", tenant_id=uuid4())
    assert info.value.reason == "not_in_bundle"


async def test_an_upstream_failure_propagates() -> None:
    fake = TenantLicencesFake()
    fake.ensure_error = LicensesClientError("licences down", status_code=503)
    with pytest.raises(LicensesClientError):
        await ensure_tenant_licences(_client(fake), tenant_id=uuid4())


async def test_tenant_licence_subject_is_canonical() -> None:
    tenant = UUID("0195F48B-D000-7000-8000-0000000000AA")
    assert tenant_licence_subject(str(tenant).upper(), "w2v") == (
        "svc:0195f48b-d000-7000-8000-0000000000aa:w2v"
    )


# ─── a client timeout on the ensure (live 2026-10-15) ─────────────────────────
# licences #275 made the ensure provision dependency licences too: 15.8 s
# server-side, committed, while the 15 s client budget raised ReadTimeout and
# every caller reported a failure for a tenant that was in fact licensed.


async def test_a_timed_out_ensure_that_licences_finished_is_confirmed_by_a_reread() -> None:
    fake, tenant = TenantLicencesFake(), uuid4()
    fake.ensure_error_after_commit = httpx.ReadTimeout("read timed out")

    ensured = await ensure_tenant_licences(_client(fake), tenant_id=tenant)

    assert set(ensured.license_ids) == {"w2v", "w2v-image"}
    for key, lid in ensured.license_ids.items():
        assert fake.holder_of(lid) == f"svc:{tenant}:{key}"
    assert ensured.refused == {}
    # The re-read asked for the tenant-level HOLDERS, never a person.
    assert fake.reread_calls == [f"svc:{tenant}:w2v", f"svc:{tenant}:w2v-image"]
    [_call] = fake.ensure_calls  # re-read, never a second ensure


async def test_a_timed_out_ensure_returns_the_w2v_licence_for_the_licence_ref() -> None:
    fake, tenant = TenantLicencesFake(), uuid4()
    fake.ensure_error_after_commit = httpx.ReadTimeout("read timed out")

    license_id = await ensure_active_license(_client(fake), app_key="w2v", tenant_id=tenant)

    assert fake.holder_of(license_id) == f"svc:{tenant}:w2v"


async def test_a_timed_out_ensure_that_wrote_nothing_still_raises_the_timeout() -> None:
    fake, tenant = TenantLicencesFake(), uuid4()
    fake.ensure_error = httpx.ReadTimeout("read timed out")

    with pytest.raises(httpx.ReadTimeout):
        await ensure_tenant_licences(_client(fake), tenant_id=tenant)
    assert fake.reread_calls == [f"svc:{tenant}:w2v", f"svc:{tenant}:w2v-image"]


async def test_a_reread_that_finds_only_a_person_held_licence_is_not_a_confirmation() -> None:
    fake, tenant = TenantLicencesFake(), uuid4()
    fake.holders["person-lic"] = (str(tenant), "w2v", "person-sub")
    fake.ensure_error = httpx.ReadTimeout("read timed out")

    with pytest.raises(httpx.ReadTimeout):
        await ensure_tenant_licences(_client(fake), tenant_id=tenant)


async def test_a_failing_reread_reports_the_original_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake, tenant = TenantLicencesFake(), uuid4()
    fake.ensure_error = httpx.ReadTimeout("read timed out")

    async def _down(subject_id, *, tenant_id):  # type: ignore[no-untyped-def]
        raise LicensesClientError("list_user_licenses returned 503", status_code=503)

    monkeypatch.setattr(fake, "list_user_licenses", _down)
    with pytest.raises(httpx.ReadTimeout):
        await ensure_tenant_licences(_client(fake), tenant_id=tenant)


async def test_a_reread_missing_a_bundle_app_records_it_as_not_confirmed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake, tenant = TenantLicencesFake(), uuid4()
    fake.ensure_error_after_commit = httpx.ReadTimeout("read timed out")
    real = fake.list_user_licenses

    async def _no_image(subject_id, *, tenant_id):  # type: ignore[no-untyped-def]
        return (
            []
            if str(subject_id).endswith(":w2v-image")
            else await real(subject_id, tenant_id=tenant_id)
        )

    monkeypatch.setattr(fake, "list_user_licenses", _no_image)
    ensured = await ensure_tenant_licences(_client(fake), tenant_id=tenant)

    assert set(ensured.license_ids) == {"w2v"}
    assert ensured.refused == {"w2v-image": NOT_CONFIRMED_AFTER_TIMEOUT}


async def test_a_licences_404_for_the_holder_on_reread_means_not_finished(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake, tenant = TenantLicencesFake(), uuid4()
    fake.ensure_error = httpx.ReadTimeout("read timed out")

    async def _unknown(subject_id, *, tenant_id):  # type: ignore[no-untyped-def]
        raise LicensesClientError("list_user_licenses returned 404", status_code=404)

    monkeypatch.setattr(fake, "list_user_licenses", _unknown)
    with pytest.raises(httpx.ReadTimeout):
        await ensure_tenant_licences(_client(fake), tenant_id=tenant)
