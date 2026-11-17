"""Tenant licence refresh by introspection (``src/services/licence_refresh.py``).

Real Postgres ``tenants`` rows; the licences service is the ``P9Licences`` fake of
the two gateway-admitted routes. The test DB is shared, so every assertion is
about the tenants this test created, never about whole-table counts.
"""

from __future__ import annotations

import asyncio
import datetime
from uuid import UUID, uuid7

import pytest
from sqlalchemy import insert, select, update

from src.clients.licenses_client import LicensesClientError
from src.database.models import Tenant
from src.observability.metrics import (
    licence_refresh_untrusted_total,
    tenant_licence_token_expiring,
)
from src.services import licence_refresh, tenant_licence
from src.services.licence_refresh import refresh_tenant_licences
from src.services.tenant_licence import seal_licence_token
from tests.api.routes._p9_licences import TEST_KEYRING, P9Licences

pytestmark = pytest.mark.asyncio

_NOW = datetime.datetime.now(datetime.UTC)


async def _tenant(
    factory,
    status: str,
    licence_ref: UUID | None,
    *,
    token: str | None = None,
    sealed_for: UUID | None = None,
    expires: datetime.datetime | None = None,
    partner: UUID | None = None,
) -> UUID:
    tid = uuid7()
    values: dict[str, object] = {
        "id": tid,
        "name": "t",
        "status": status,
        "licence_ref": licence_ref,
        "partner_licence_id": partner,
    }
    if token is not None:
        sealed = seal_licence_token(TEST_KEYRING, sealed_for or tid, token)
        values |= {
            "licence_token_ciphertext": sealed.ciphertext,
            "licence_token_kid": sealed.key_id,
            "licence_token_set_at": _NOW,
            "licence_token_expires_at": expires,
        }
    async with factory() as s:
        await s.execute(insert(Tenant).values(**values))
        await s.commit()
    return tid


async def _row(factory, tid: UUID) -> tuple[str, UUID | None, object]:
    async with factory() as s:
        r = (
            await s.execute(
                select(Tenant.status, Tenant.licence_ref, Tenant.licence_checked_at).where(
                    Tenant.id == tid
                )
            )
        ).one()
    return r.status, r.licence_ref, r.licence_checked_at


def _untrusted(reason: str) -> float:
    return licence_refresh_untrusted_total.labels(reason=reason)._value.get()


async def _sweep(factory, client: P9Licences) -> licence_refresh.RefreshResult:
    async with factory() as s:
        return await refresh_tenant_licences(s, client, TEST_KEYRING)


async def test_sweep_with_mixed_tenants(db_session_factory) -> None:
    """Token tenants are judged by introspection, the rest by /licenses/active;
    each verdict is the tenant's own."""
    lic = P9Licences()
    tok_ok_lic, tok_new_lic = uuid7(), uuid7()
    # token tenants (the token must be minted for the tenant id the row gets)
    tok_ok = await _tenant(db_session_factory, "licence_lapsed", None)
    async with db_session_factory() as s:  # re-seal now that the id is known
        sealed = seal_licence_token(TEST_KEYRING, tok_ok, lic.token(tok_ok, tok_new_lic))
        await s.execute(
            update(Tenant)
            .where(Tenant.id == tok_ok)
            .values(
                licence_token_ciphertext=sealed.ciphertext,
                licence_token_kid=sealed.key_id,
                licence_token_set_at=_NOW,
                licence_ref=tok_ok_lic,
            )
        )
        await s.commit()
    tok_inactive_raw = lic.token(uuid7(), uuid7(), active=False, reason="license_revoked")
    tok_inactive = await _tenant(db_session_factory, "active", uuid7(), token=tok_inactive_raw)
    foreign_raw = lic.token(uuid7(), uuid7())  # an active token of ANOTHER tenant
    tok_foreign = await _tenant(db_session_factory, "active", uuid7(), token=foreign_raw)
    unreadable = await _tenant(
        db_session_factory, "active", uuid7(), token="x", sealed_for=uuid7()
    )  # sealed under another tenant's AAD → will not open
    # fallback tenants
    single_old, single_new = uuid7(), uuid7()
    fb_single = await _tenant(db_session_factory, "active", single_old)
    fb_multi_ref = uuid7()
    fb_multi = await _tenant(db_session_factory, "licence_lapsed", fb_multi_ref)
    fb_none = await _tenant(db_session_factory, "active", uuid7())
    partner_ref = uuid7()
    fb_partner = await _tenant(db_session_factory, "active", partner_ref, partner=uuid7())
    suspended = await _tenant(db_session_factory, "suspended", uuid7())
    unlicensed = await _tenant(db_session_factory, "active", None)
    lic.live = {
        fb_single: [single_new],
        fb_multi: [uuid7(), uuid7()],
        fb_partner: [uuid7()],
    }
    unreadable0 = _untrusted("unreadable")

    result = await _sweep(db_session_factory, lic)

    assert (await _row(db_session_factory, tok_ok))[:2] == ("active", tok_new_lic)
    assert tok_ok in result.recovered
    assert (await _row(db_session_factory, tok_inactive))[0] == "licence_lapsed"
    assert (await _row(db_session_factory, tok_foreign))[0] == "licence_lapsed"
    assert (await _row(db_session_factory, unreadable))[:1] == ("active",)
    assert (await _row(db_session_factory, unreadable))[2] is None, "unreadable: untouched"
    assert _untrusted("unreadable") == unreadable0 + 1
    assert (await _row(db_session_factory, fb_single))[:2] == ("active", single_new)
    assert (await _row(db_session_factory, fb_multi))[:2] == ("active", fb_multi_ref), (
        ">1 candidate: entitled, ref kept"
    )
    assert (await _row(db_session_factory, fb_none))[0] == "licence_lapsed"
    assert (await _row(db_session_factory, fb_partner))[:2] == ("active", partner_ref), (
        "a partner tenant's ref is never switched by the fallback"
    )
    assert (await _row(db_session_factory, suspended))[0] == "suspended"
    assert (await _row(db_session_factory, unlicensed))[:2] == ("active", None)
    assert (await _row(db_session_factory, unlicensed))[2] is None, "not swept at all"
    assert {tok_inactive, tok_foreign, fb_none} <= set(result.lapsed)


@pytest.mark.parametrize("client", [P9Licences(down=True)])
async def test_licences_unavailable_or_refused_changes_nothing(db_session_factory, client) -> None:
    tok = await _tenant(db_session_factory, "active", uuid7(), token="lic_tok_x")
    fb = await _tenant(db_session_factory, "licence_lapsed", uuid7())
    before = {t: await _row(db_session_factory, t) for t in (tok, fb)}
    reason = "unavailable" if client.down else "refused"
    count0 = _untrusted(reason)
    result = await _sweep(db_session_factory, client)
    for t, row in before.items():
        assert await _row(db_session_factory, t) == row
    assert result.untrusted >= 2 and not ({tok, fb} & set(result.lapsed + result.recovered))
    assert _untrusted(reason) >= count0 + 2


async def test_expired_stored_token_falls_back_and_warns_once(db_session_factory, caplog) -> None:
    licence = uuid7()
    tid = await _tenant(
        db_session_factory,
        "active",
        licence,
        token="lic_tok_expired",
        expires=_NOW - datetime.timedelta(days=1),
    )
    lic = P9Licences({tid: [licence]})
    caplog.set_level("WARNING", logger="w2v_orchestrator.tenant_licence")
    await _sweep(db_session_factory, lic)
    await _sweep(db_session_factory, lic)
    assert lic.introspect_calls == 0, "an expired token is never introspected"
    assert (await _row(db_session_factory, tid))[:2] == ("active", licence)
    warned = [
        r for r in caplog.records if str(tid) in r.getMessage() and "expired" in r.getMessage()
    ]
    assert len(warned) == 1, "WARNING once per tenant per process"
    assert tid in tenant_licence._expired_warned


async def test_expired_token_fallback_never_rewrites_the_token_owned_ref(
    db_session_factory,
) -> None:
    """A tenant whose stored token EXPIRED is judged by the fallback, and a single
    /licenses/active candidate that differs from its ref is NOT written: while any
    token is stored only the token path changes ``licence_ref``. Status and
    ``licence_checked_at`` still move (here lapsed → active)."""
    owned, other = uuid7(), uuid7()
    tid = await _tenant(
        db_session_factory,
        "licence_lapsed",
        owned,
        token="lic_tok_expired",
        expires=_NOW - datetime.timedelta(days=1),
    )
    lic = P9Licences({tid: [other]})
    result = await _sweep(db_session_factory, lic)
    status, ref, checked_at = await _row(db_session_factory, tid)
    assert lic.introspect_calls == 0
    assert ref == owned, "the fallback must not switch a token-owned licence_ref"
    assert status == "active" and tid in result.recovered
    assert checked_at is not None


class _RacingLicences(P9Licences):
    """Runs ``during`` INSIDE the sweep's licences call — i.e. after the sweep
    read the row and before it writes its verdict (the review's interleaving)."""

    def __init__(self, during, **kw) -> None:
        super().__init__(**kw)
        self.during = during

    async def introspect(self, token: str):
        await self.during()
        return await super().introspect(token)

    async def get_active_license(self, *, tenant_id, app_key):
        await self.during()
        return await super().get_active_license(tenant_id=tenant_id, app_key=app_key)


async def test_sweep_never_undoes_a_rotation_that_lands_mid_check(db_session_factory) -> None:
    """Old token revoked → the sweep's verdict is "lapse". A rotation stores a NEW
    verified token + ref while the sweep is in its introspect call: the rotation
    must win (status active, the new ref), and the sweep counts ``raced``."""
    from src.database.repositories.tenant_licence_repository import TenantLicenceRepository

    old_ref, new_ref = uuid7(), uuid7()
    tid_box: list[UUID] = []

    async def rotate() -> None:
        tid = tid_box[0]
        sealed = seal_licence_token(TEST_KEYRING, tid, lic.token(tid, new_ref))
        async with db_session_factory() as s:
            await TenantLicenceRepository(s).store_token(
                tid,
                ciphertext=sealed.ciphertext,
                key_id=sealed.key_id,
                licence_ref=new_ref,
                token_expires_at=None,
            )
            await s.commit()

    lic = _RacingLicences(rotate)
    revoked = lic.token(uuid7(), old_ref, active=False, reason="license_revoked")
    tid = await _tenant(db_session_factory, "active", old_ref, token=revoked)
    tid_box.append(tid)
    raced0 = _untrusted("raced")

    result = await _sweep(db_session_factory, lic)

    assert (await _row(db_session_factory, tid))[:2] == ("active", new_ref)
    assert tid not in result.lapsed
    assert _untrusted("raced") == raced0 + 1


async def test_sweep_never_overwrites_a_ref_bound_mid_check(db_session_factory) -> None:
    """Fallback tenant: a super_admin bind changes ``licence_ref`` while the sweep
    is in ``/licenses/active`` — the sweep's single-candidate ref must not
    overwrite it."""
    bound = uuid7()
    tid_box: list[UUID] = []

    async def bind() -> None:
        async with db_session_factory() as s:
            await s.execute(update(Tenant).where(Tenant.id == tid_box[0]).values(licence_ref=bound))
            await s.commit()

    lic = _RacingLicences(bind)
    tid = await _tenant(db_session_factory, "active", uuid7())
    tid_box.append(tid)
    lic.live[tid] = [uuid7()]  # one candidate, not the bound ref

    await _sweep(db_session_factory, lic)

    assert (await _row(db_session_factory, tid))[:2] == ("active", bound)


async def test_sweep_skips_a_tenant_whose_licence_change_lease_is_held(
    db_session_factory,
) -> None:
    from src.database.repositories.tenant_licence_repository import TenantLicenceRepository

    tid = await _tenant(db_session_factory, "active", uuid7())  # no live licence → would lapse
    async with db_session_factory() as s:
        assert await TenantLicenceRepository(s).acquire_lease(tid, uuid7())
        await s.commit()
    await _sweep(db_session_factory, P9Licences())
    status, _ref, checked = await _row(db_session_factory, tid)
    assert (status, checked) == ("active", None), "a held lease: the sweep writes nothing"


async def test_sweep_still_writes_when_nothing_changed(db_session_factory) -> None:
    """Control for the CAS: an untouched row IS updated."""
    tid = await _tenant(db_session_factory, "active", uuid7())
    await _sweep(db_session_factory, P9Licences())
    status, _ref, checked = await _row(db_session_factory, tid)
    assert status == "licence_lapsed" and checked is not None


async def test_sweep_sets_the_expiring_soon_gauge(db_session_factory) -> None:
    await _tenant(
        db_session_factory,
        "active",
        uuid7(),
        token="lic_tok_soon",
        expires=_NOW + datetime.timedelta(days=5),
    )
    await _sweep(db_session_factory, P9Licences())
    assert tenant_licence_token_expiring._value.get() >= 1


async def test_refused_excerpt_is_bounded_and_token_free(caplog) -> None:
    caplog.set_level("WARNING", logger="w2v_orchestrator.tenant_licence")
    gateway_400 = LicensesClientError(
        "introspect returned 400: "
        '{"error":"missing_app_key","hint":"Bearer eyJhbGc.eyJzdWI.c2ln"}' + "x" * 600,
        status_code=400,
    )
    err = tenant_licence.licences_error(gateway_400)
    assert isinstance(err, tenant_licence.LicencesRefusedError)
    assert "missing_app_key" in err.excerpt and len(err.excerpt) <= 300
    assert "eyJ" not in err.excerpt
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "HTTP 400" in logged and "eyJ" not in logged
    assert str(err) == "licences refused the call: HTTP 400"
    for retryable in (408, 425, 429, 500, 503):
        e = tenant_licence.licences_error(LicensesClientError("x", status_code=retryable))
        assert not isinstance(e, tenant_licence.LicencesRefusedError)
    assert not isinstance(
        tenant_licence.licences_error(OSError("down")), tenant_licence.LicencesRefusedError
    )


async def test_loop_survives_a_failing_sweep_and_disabled_interval(
    db_session_factory, monkeypatch
) -> None:
    assert licence_refresh.start_licence_refresh_task(db_session_factory, 0) is None
    calls: list[int] = []

    async def _boom(session, client, keyring):  # type: ignore[no-untyped-def]
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("sweep exploded")
        raise asyncio.CancelledError

    monkeypatch.setattr(licence_refresh, "refresh_tenant_licences", _boom)
    with pytest.raises(asyncio.CancelledError):
        await licence_refresh.run_licence_refresh_loop(
            db_session_factory, P9Licences, 0, lambda: TEST_KEYRING
        )
    assert len(calls) == 2, "a failed sweep must not kill the loop"


async def test_loop_interval_is_jittered(db_session_factory, monkeypatch) -> None:
    slept: list[float] = []

    async def _sweep_stub(session, client, keyring):  # type: ignore[no-untyped-def]
        if len(slept) >= 20:
            raise asyncio.CancelledError

    async def _sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(licence_refresh, "refresh_tenant_licences", _sweep_stub)
    monkeypatch.setattr(licence_refresh.asyncio, "sleep", _sleep)
    with pytest.raises(asyncio.CancelledError):
        await licence_refresh.run_licence_refresh_loop(
            db_session_factory, P9Licences, 100, lambda: TEST_KEYRING
        )
    assert all(90 <= v <= 110 for v in slept), slept
    assert len(set(slept)) > 1, "every pod must not sleep the exact same interval"
