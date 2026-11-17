"""Phase D RLS — the ``after_begin`` GUC re-bind listener (Bug A regression).

``get_run`` commits the request session mid-handler (``RunUsageReconciler.reconcile``
runs a read-through billing reconcile and commits), which drops the
transaction-scoped ``app.current_tenant`` GUC; the *next* RLS-gated read
(``_project_id_for_deliverable``) then runs unbound and, under enforce, returns 0
rows → a spurious ``404 "run not found"`` on a run the caller genuinely owns
(observed live as ``200 running → 404 → 200 completed``).

``install_rls_tenant_rebind`` registers an ``after_begin`` listener that re-binds
the GUC from the identity ContextVar on every ``BEGIN``, so any intra-request (or
background) commit self-heals. These tests prove the GUC survives a commit under
enforce, and that the listener is inert off-enforce / with no tenant in context.

We assert the GUC value directly rather than row visibility: the test DB user
OWNS the tables, and RLS ``ENABLE`` (not ``FORCE``) never binds the owner, so
every row is visible regardless of the GUC — only the GUC value itself proves the
re-bind fired.
"""

from __future__ import annotations

from uuid import UUID, uuid7

import pytest
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError
from w2v_common.auth.identity_context import clear_identity_context, set_identity_context

from src.api.routes._common import apply_tenant_rls
from src.config import clear_settings_cache
from src.database.models import Deliverable, Project, Run
from src.database.rls_events import install_rls_tenant_rebind
from tests.conftest import require_createrole

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _reset_settings_and_context():
    clear_settings_cache()
    clear_identity_context()
    yield
    clear_settings_cache()
    clear_identity_context()


async def _guc(session) -> str | None:
    return (
        await session.execute(text("SELECT current_setting('app.current_tenant', true)"))
    ).scalar_one()


async def test_after_begin_rebinds_tenant_guc_across_commit(db_session_factory, monkeypatch):
    """The core regression: the GUC is still bound after a mid-handler commit."""
    monkeypatch.setenv("TENANT_RLS_MODE", "enforce")
    clear_settings_cache()
    install_rls_tenant_rebind()  # idempotent — already installed by asgi in prod

    tenant = uuid7()
    set_identity_context(str(tenant), None, None)

    async with db_session_factory() as session:
        # First transaction: after_begin binds the GUC before the statement runs.
        assert await _guc(session) == str(tenant)
        # The commit that used to drop the GUC (RunUsageReconciler inside get_run).
        await session.commit()
        # Next transaction: the listener re-binds → still scoped, not '' → no 404.
        assert await _guc(session) == str(tenant)


async def test_listener_inert_when_not_enforce(db_session_factory, monkeypatch):
    monkeypatch.setenv("TENANT_RLS_MODE", "off")
    clear_settings_cache()
    install_rls_tenant_rebind()
    set_identity_context(str(uuid7()), None, None)

    async with db_session_factory() as session:
        # Off enforce → listener no-ops → GUC never set on this connection.
        assert not await _guc(session)


async def test_listener_inert_when_no_tenant_in_context(db_session_factory, monkeypatch):
    monkeypatch.setenv("TENANT_RLS_MODE", "enforce")
    clear_settings_cache()
    install_rls_tenant_rebind()
    clear_identity_context()  # no tenant in context (e.g. a pre-auth/background session)

    async with db_session_factory() as session:
        assert not await _guc(session)


# ── Row-visibility reproduction under a REAL restricted role ──────────────────
# The GUC-value tests above prove the listener re-binds ``app.current_tenant``,
# but the test DB user OWNS the tables (RLS is ``ENABLE`` not ``FORCE``), so the
# owner bypasses every policy and every row is visible regardless of the GUC.
# That is exactly why the read-after-write regression escaped the unit suite and
# was only caught by the post-promote E2E: prod runs as the non-owner,
# ``NOBYPASSRLS`` runtime role ``w2v_orch_app`` where the GUC actually gates
# rows. This test reproduces the E2E path directly by ``SET LOCAL ROLE`` into a
# throwaway non-owner role — mirrors ``test_rls_isolation.py``. Must run under CI
# (``require_createrole`` fails loudly if CREATEROLE is ever missing); skips only
# for a developer connected as the restricted runtime role, same as that test.


async def _seed_run_tree(session, tenant_id: UUID) -> UUID:
    """Insert one project→deliverable→run tree for ``tenant_id`` (as the owner).
    Returns the run id — the row ``get_run``'s post-reconcile-commit read fetches.
    """
    project = Project(
        id=uuid7(),
        tenant_id=tenant_id,
        owner_subject_id=uuid7(),
        visibility="private",
        type="promo",
        title=f"proj-{tenant_id}",
    )
    deliverable = Deliverable(id=uuid7(), project_id=project.id, name="d1", format="16x9")
    run = Run(id=uuid7(), deliverable_id=deliverable.id, blueprint_id="promo_v1")
    session.add_all([project, deliverable, run])
    await session.commit()
    return run.id


async def _run_tree_visible(session, run_id: UUID) -> bool:
    """The ``_tenant_owns_run`` / ``_project_id_for_deliverable`` shape: does the
    RLS-gated run→deliverable→project join return the row for the bound tenant?"""
    found = (
        await session.execute(
            text(
                "SELECT r.id FROM runs r "
                "JOIN deliverables d ON d.id = r.deliverable_id "
                "JOIN projects p ON p.id = d.project_id "
                "WHERE r.id = :rid"
            ),
            {"rid": str(run_id)},
        )
    ).scalar_one_or_none()
    return found is not None


@pytest.fixture
async def restricted_role(db_session_factory):
    """A throwaway ``NOBYPASSRLS`` non-owner role to ``SET ROLE`` into.

    The test DB user OWNS the tables and RLS is ``ENABLE`` (not ``FORCE``), so the
    owner bypasses every policy — only a non-owner role makes row visibility
    actually depend on ``app.current_tenant``. Mirrors the prod runtime posture
    (``w2v_orch_app``). Skips only for a developer already connected as a
    restricted role; under CI ``require_createrole`` fails loudly instead.
    """
    role = f"rls_rebind_probe_{uuid7().hex[:12]}"
    async with db_session_factory() as owner:
        schema = (await owner.execute(text("SELECT current_schema()"))).scalar_one()
        try:
            await owner.execute(text(f'CREATE ROLE "{role}" NOLOGIN NOSUPERUSER NOBYPASSRLS'))
        except ProgrammingError as exc:  # insufficient_privilege → role lacks CREATEROLE
            await owner.rollback()
            require_createrole(exc)  # fail under CI, skip for a locally-restricted role
        await owner.execute(text(f'GRANT "{role}" TO CURRENT_USER'))
        await owner.execute(text(f'GRANT USAGE ON SCHEMA "{schema}" TO "{role}"'))
        await owner.execute(text(f'GRANT SELECT ON ALL TABLES IN SCHEMA "{schema}" TO "{role}"'))
        await owner.commit()
    try:
        yield role
    finally:
        async with db_session_factory() as owner:
            teardown_schema = (await owner.execute(text("SELECT current_schema()"))).scalar_one()
            # Explicitly revoke every grant before DROP ROLE. DROP OWNED BY alone
            # can leave schema/table privileges when the grantor is a non-superuser
            # (the shared-schema local path), tripping DependentObjectsStillExist.
            for stmt in (
                f'REVOKE ALL ON ALL TABLES IN SCHEMA "{teardown_schema}" FROM "{role}"',
                f'REVOKE ALL ON SCHEMA "{teardown_schema}" FROM "{role}"',
                f'REVOKE "{role}" FROM CURRENT_USER',
                f'DROP OWNED BY "{role}"',
                f'DROP ROLE IF EXISTS "{role}"',
            ):
                try:
                    await owner.execute(text(stmt))
                    await owner.commit()
                except ProgrammingError:
                    await owner.rollback()


async def test_owned_run_stays_visible_after_commit_under_restricted_role(
    db_session_factory, restricted_role, monkeypatch
):
    """The actual read-after-write regression, reproduced under a role RLS gates.

    ``get_run`` binds the tenant GUC once (router dep), reads the run, then the
    cost read-through commits mid-handler (``RunUsageReconciler.reconcile``). That
    commit ends the transaction the ``is_local`` GUC lived in; the subsequent
    RLS-gated run-tree read (``_project_id_for_deliverable``) then runs unbound
    and — under a non-owner ``NOBYPASSRLS`` role + enforce — returns 0 rows → a
    spurious ``404 "run not found"`` on a run the caller owns (the E2E's
    ``200 running → 404 → 200`` flap that made the post-promote validator time
    out). The ``after_begin`` re-bind listener must keep the row visible across
    that commit.
    """
    monkeypatch.setenv("TENANT_RLS_MODE", "enforce")
    clear_settings_cache()
    install_rls_tenant_rebind()

    tenant = uuid7()
    async with db_session_factory() as owner:
        run_id = await _seed_run_tree(owner, tenant)

    # The identity ContextVar is what AuthMiddleware sets per request and what
    # the after_begin listener reads to re-bind the GUC.
    set_identity_context(str(tenant), None, None)
    async with db_session_factory() as s:
        # SESSION-scoped role (not SET LOCAL): a SET LOCAL ROLE reverts on the
        # mid-handler commit, so the post-commit read would run as the RLS-
        # bypassing OWNER and hide the bug. SET ROLE survives the commit — the
        # non-owner runtime posture the prod pod actually has across the reconcile.
        await s.execute(text(f'SET ROLE "{restricted_role}"'))
        # Simulate the request path's ONE explicit bind (the apply_tenant_rls
        # router dep) so the first read is bound regardless of the listener —
        # exactly as get_run's first read (_tenant_owns_run) is. This isolates
        # the regression to the POST-COMMIT read, which is the actual bug.
        await s.execute(
            text("SELECT set_config('app.current_tenant', :t, true)"), {"t": str(tenant)}
        )
        assert await _run_tree_visible(s, run_id), "run must be visible on the bound read"
        # The mid-handler commit the reconciler issues — drops the is_local GUC.
        await s.commit()
        # The post-commit RLS-gated read (_project_id_for_deliverable). The one
        # explicit bind above is now gone; only the after_begin listener re-binds
        # it. Without the listener this returns None → 404; with it, still visible.
        assert await _run_tree_visible(s, run_id), (
            "owned run must stay visible after the intra-request reconcile commit "
            "(regression: unbound GUC → 0 rows → spurious 404 run not found)"
        )
        await s.execute(text("RESET ROLE"))
        await s.rollback()


# ── Super-admin scope-switch: the effective tenant must survive EVERY commit ───


async def test_scope_switched_tenant_survives_two_sequential_commits(
    db_session_factory, restricted_role, monkeypatch
):
    """A ``super_admin`` acting into another tenant must keep seeing THAT tenant.

    ``get_tenant_id`` (``src/api/routes/_common.py``) honours ``X-Active-Tenant-Id``
    for a ``super_admin``, so the EFFECTIVE tenant differs from the token's CLAIM
    tenant — and ``apply_tenant_rls`` binds the effective one. That bind is
    ``SET LOCAL``, so it dies at the first COMMIT; the ``after_begin`` listener used
    to re-bind unconditionally from the identity ContextVar, which holds the CLAIM
    tenant. From transaction 2 onward the request therefore ran under the ADMIN's
    OWN tenant: fail-closed (no cross-tenant leak) but silently WRONG — every
    RLS-gated read matched 0 rows and handlers returned 200/204 with empty data.

    The listener now prefers ``session.info[SESSION_TENANT_KEY]`` (recorded by
    ``bind_session_tenant``), so the effective tenant survives arbitrarily many
    commits. Two sequential commits are asserted because one commit alone cannot
    distinguish "re-bound correctly" from "re-bound to the claim and got lucky".
    """
    monkeypatch.setenv("TENANT_RLS_MODE", "enforce")
    clear_settings_cache()
    install_rls_tenant_rebind()

    admin_tenant = uuid7()  # the super_admin's OWN claim tenant
    target_tenant = uuid7()  # X-Active-Tenant-Id scope-switch target
    async with db_session_factory() as owner:
        target_run = await _seed_run_tree(owner, target_tenant)
        admin_run = await _seed_run_tree(owner, admin_tenant)

    # AuthMiddleware sets the CLAIM tenant in the identity ContextVar; the
    # scope-switch lives only in the header, i.e. only in the effective tenant.
    set_identity_context(str(admin_tenant), None, None)

    async with db_session_factory() as s:
        await s.execute(text(f'SET ROLE "{restricted_role}"'))
        # Exactly what the router dep does for a scope-switched super_admin.
        await apply_tenant_rls(tenant_id=target_tenant, session=s)

        for label, commits in (("bound read", 0), ("after commit #1", 1), ("after commit #2", 2)):
            assert await _run_tree_visible(s, target_run), (
                f"scope-switched super_admin lost the target tenant's rows {label} "
                f"({commits} intra-request commits) — the listener re-bound the CLAIM tenant"
            )
            assert not await _run_tree_visible(s, admin_run), (
                f"the admin's OWN-tenant rows must never be visible {label} while "
                "scope-switched (the effective tenant is the only one in scope)"
            )
            assert await _guc(s) == str(target_tenant)
            await s.commit()

        await s.execute(text("RESET ROLE"))
        await s.rollback()
