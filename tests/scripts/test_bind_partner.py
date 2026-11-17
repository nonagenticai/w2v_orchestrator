"""``scripts/bind_partner.py`` — the operator CLI for the P9 partner bind.

``run`` is driven against the real test database with the licences fake, so
"dry run writes nothing", "apply binds + audits as operator-cli", idempotence
and refusals are proven on rows. The CLI shares
``src.services.tenant_partner_bind.bind_tenant_to_partner`` with the super_admin
route; the route's own behaviour stays pinned in
``tests/api/routes/test_p9_partner_keys.py``.
"""

from __future__ import annotations

import argparse
from typing import Any, cast
from uuid import UUID, uuid4, uuid7

import pytest
from sqlalchemy import insert, select

from scripts import bind_partner
from scripts.bind_partner import OPERATOR_CLI, run
from src.database.models import AdminAuditLog, Project, Tenant
from tests.api.routes._p9_licences import TEST_KEYRING, P9Licences
from tests.api.routes.test_tenancy_partner import (  # noqa: F401 — autouse allow-list fixture
    _OTHER_PARTNER,
    _PARTNER,
    _allowlist,
)

pytestmark = pytest.mark.asyncio


async def _occupied(factory) -> UUID:
    """A tenant with data and NO registry row — merqi's dogfood shape."""
    tid = uuid7()
    async with factory() as s:
        s.add(
            Project(
                id=uuid7(),
                tenant_id=tid,
                owner_subject_id=uuid7(),
                visibility="private",
                type="promo",
                title="dogfood data",
            )
        )
        await s.commit()
    return tid


async def _tenant(factory, tid: UUID) -> Tenant | None:
    async with factory() as s:
        return (await s.execute(select(Tenant).where(Tenant.id == tid))).scalar_one_or_none()


async def _audit(factory, tid: UUID) -> list[tuple[str, UUID | None, dict]]:
    async with factory() as s:
        rows = await s.execute(
            select(AdminAuditLog.action, AdminAuditLog.actor_subject_id, AdminAuditLog.detail)
            .where(AdminAuditLog.target_tenant_id == tid)
            .order_by(AdminAuditLog.created_at)
        )
        return [(r.action, r.actor_subject_id, r.detail) for r in rows]


async def _run(factory, lic: P9Licences, tid: UUID, *, token=None, apply=False, partner=_PARTNER):
    lines: list[str] = []
    code = await run(
        factory,
        cast("Any", lic),
        TEST_KEYRING,
        tenant_id=tid,
        partner_licence_id=partner,
        token=token,
        apply=apply,
        out=lines.append,
    )
    return code, "\n".join(lines)


async def test_dry_run_reports_the_bind_and_writes_nothing(db_session_factory) -> None:
    tid = await _occupied(db_session_factory)
    code, out = await _run(db_session_factory, P9Licences(), tid)
    assert code == 0, out
    assert "would bind" in out and "would create the tenants row" in out
    assert "DRY RUN — nothing written." in out
    assert await _tenant(db_session_factory, tid) is None, "dry run kept the registry row"
    assert await _audit(db_session_factory, tid) == []


async def test_dry_run_with_a_token_verifies_but_stores_nothing(db_session_factory) -> None:
    tid, licence = await _occupied(db_session_factory), uuid7()
    lic = P9Licences()
    token = lic.token(tid, licence)
    code, out = await _run(db_session_factory, lic, tid, token=token)
    assert code == 0, out
    assert lic.introspect_calls == 1, "the token is verified in a dry run too"
    assert f"licence_ref={licence}" in out and token not in out
    assert await _tenant(db_session_factory, tid) is None
    assert await _audit(db_session_factory, tid) == []


async def test_apply_binds_and_audits_as_operator_cli(db_session_factory) -> None:
    tid = await _occupied(db_session_factory)
    code, out = await _run(db_session_factory, P9Licences(), tid, apply=True)
    assert code == 0, out
    assert "APPLIED (committed)." in out
    tenant = await _tenant(db_session_factory, tid)
    assert tenant is not None
    assert tenant.partner_licence_id == _PARTNER and tenant.created_via == "super_admin"
    assert await _audit(db_session_factory, tid) == [
        (
            "tenant.partner_bind",
            None,  # no fabricated super_admin subject
            {
                "partner_licence_id": str(_PARTNER),
                "created_registry_row": True,
                "by": OPERATOR_CLI,
            },
        )
    ]


async def test_apply_is_idempotent(db_session_factory) -> None:
    tid = await _occupied(db_session_factory)
    assert (await _run(db_session_factory, P9Licences(), tid, apply=True))[0] == 0
    code, out = await _run(db_session_factory, P9Licences(), tid, apply=True)
    assert code == 0, out
    assert "already bound to this partner" in out and "bound=False" in out
    assert [a for a, _, _ in await _audit(db_session_factory, tid)] == ["tenant.partner_bind"]
    # and a dry run of the bound tenant is a clean no-op
    code, out = await _run(db_session_factory, P9Licences(), tid)
    assert code == 0 and "bound=False" in out, out


async def test_apply_with_token_stores_it_sealed_and_never_prints_it(db_session_factory) -> None:
    tid, licence = await _occupied(db_session_factory), uuid7()
    lic = P9Licences()
    token = lic.token(tid, licence)
    code, out = await _run(db_session_factory, lic, tid, token=token, apply=True)
    assert code == 0, out
    assert "TOKEN" in out and "stored" in out and token not in out
    tenant = await _tenant(db_session_factory, tid)
    assert tenant is not None and tenant.licence_ref == licence
    assert tenant.licence_token_ciphertext is not None
    assert token.encode() not in bytes(tenant.licence_token_ciphertext)
    audit = await _audit(db_session_factory, tid)
    assert [(a, actor) for a, actor, _ in audit] == [
        ("tenant.partner_bind", None),
        ("tenant.licence_token_set", None),
    ]
    assert audit[1][2]["by"] == OPERATOR_CLI and audit[1][2]["licence_ref"] == str(licence)
    assert token not in str(audit)
    # Same token again: idempotent — no new rows, token unchanged.
    code, out = await _run(db_session_factory, lic, tid, token=token, apply=True)
    assert code == 0 and "unchanged" in out and token not in out, out
    assert len(await _audit(db_session_factory, tid)) == 2


@pytest.mark.parametrize("apply", [False])
async def test_a_token_that_is_not_the_tenants_licence_is_refused(
    db_session_factory, apply: bool
) -> None:
    tid = await _occupied(db_session_factory)
    lic = P9Licences()
    token = lic.token(uuid7(), uuid7())  # another tenant's licence
    code, out = await _run(db_session_factory, lic, tid, token=token, apply=apply)
    assert code == 1, out
    assert "REFUSED 422 LICENCE_TOKEN_MISMATCH" in out and "'reason': 'tenant'" in out
    assert token not in out
    assert await _tenant(db_session_factory, tid) is None, "a refused bind writes nothing"
    assert await _audit(db_session_factory, tid) == []


async def test_a_non_partner_licence_is_refused(db_session_factory) -> None:
    tid = await _occupied(db_session_factory)
    code, out = await _run(db_session_factory, P9Licences(), tid, apply=True, partner=uuid4())
    assert code == 1 and "REFUSED 422 NOT_A_PARTNER_LICENCE" in out, out
    assert await _tenant(db_session_factory, tid) is None


async def test_unknown_tenant_and_another_partners_tenant_are_refused(db_session_factory) -> None:
    code, out = await _run(db_session_factory, P9Licences(), uuid7(), apply=True)
    assert code == 1 and "REFUSED 404 TENANT_NOT_FOUND" in out, out

    tid = uuid7()
    async with db_session_factory() as s:
        await s.execute(
            insert(Tenant).values(
                id=tid, name="t", created_via="partner", partner_licence_id=_OTHER_PARTNER
            )
        )
        await s.commit()
    for apply in (False, True):
        code, out = await _run(db_session_factory, P9Licences(), tid, apply=apply)
        assert code == 1 and "REFUSED 409 TENANT_BOUND_TO_OTHER_PARTNER" in out, out
    tenant = await _tenant(db_session_factory, tid)
    assert tenant is not None and tenant.partner_licence_id == _OTHER_PARTNER


async def test_licences_down_is_refused_and_writes_nothing(db_session_factory) -> None:
    tid = await _occupied(db_session_factory)
    code, out = await _run(
        db_session_factory, P9Licences(down=True), tid, token="lic_tok_whatever", apply=True
    )
    assert code == 1 and "REFUSED 503 LICENCES_UNAVAILABLE" in out, out
    assert await _tenant(db_session_factory, tid) is None


# ─── CLI surface: the token comes from an env var, never argv, never output ─


async def test_the_cli_has_no_token_or_licence_id_argument() -> None:
    opts = {a for action in bind_partner._parser()._actions for a in action.option_strings}
    assert "--licence-token-env" in opts
    assert not {o for o in opts if o != "--licence-token-env" and "token" in o}
    assert "--licence-id" not in opts, "retired by #1101: a bare licence id proves nothing"


async def test_main_reads_the_token_from_env_and_redacts_it_on_abort(monkeypatch, capsys) -> None:
    secret = "lic_tok_" + "s3cr3t" * 4  # pragma: allowlist secret
    monkeypatch.setenv("BIND_TOKEN_UNDER_TEST", secret)

    def boom() -> object:
        raise RuntimeError(f"upstream echoed {secret}")

    async def no_db() -> object:
        return object()

    monkeypatch.setattr("src.database.session.get_session_maker", no_db)
    monkeypatch.setattr("src.api.routes._common.get_licenses_client", boom)
    args = argparse.Namespace(
        tenant_id=uuid7(),
        partner_licence_id=_PARTNER,
        licence_token_env="BIND_TOKEN_UNDER_TEST",
        apply=True,
    )
    assert await bind_partner._main(args) == 2
    captured = capsys.readouterr()
    assert "ABORTED, nothing written" in captured.err
    assert secret not in captured.out + captured.err and "***" in captured.err


async def test_main_aborts_on_an_unset_token_env(monkeypatch, capsys) -> None:
    monkeypatch.delenv("BIND_TOKEN_UNSET", raising=False)
    args = argparse.Namespace(
        tenant_id=uuid7(),
        partner_licence_id=_PARTNER,
        licence_token_env="BIND_TOKEN_UNSET",
        apply=False,
    )
    assert await bind_partner._main(args) == 2
    assert "BIND_TOKEN_UNSET" in capsys.readouterr().err
