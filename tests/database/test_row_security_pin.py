"""Regression: a leaked ``row_security = off`` must not take the orchestrator down.

Observed live on 2026-08-20 in the ``w2v`` namespace: the reconciler loop
crash-looped and ``GET /api/runs/{id}/status`` 500'd on the same pods, every time
with::

    psycopg.errors.InsufficientPrivilege:
      query would be affected by row-level security policy for table "runs"

That error is NOT what an unbound ``app.current_tenant`` produces — an unbound GUC
makes the policy predicate false, which yields ``0 rows`` / ``UPDATE 0``, silently.
The 42501 raise has exactly one cause: ``row_security = off`` on the session
(PostgreSQL docs: "When set to off, queries fail which would otherwise apply at
least one policy"). ``row_security`` is a plain USERSET session GUC, and the
orchestrator reaches Postgres through PgBouncer in ``pool_mode = transaction``,
which does not run a reset query between transactions — so a session-level ``SET
row_security = off`` from ANY client sharing the pool (``pg_dump`` opens with
precisely that statement) sticks on the server connection and poisons every
subsequent borrower.

The fix pins ``row_security`` back ON transaction-locally at the head of every
transaction, from the ``after_begin`` listener (``src/database/rls_events.py``)
and ONLY from there — ``after_begin`` fires on the first statement of the
transaction, so a second copy folded into ``bind_session_tenant``'s own statement
would be unreachable. The same statement reads the value the pooled connection
arrived with (``ROW_SECURITY_PIN_SQL`` / ``BIND_TENANT_DRIVER_SQL`` in
``src/database/session.py``) and WARNS when it was not ``on``, so self-healing
does not delete the only diagnosis available to the >=7 sibling w2v services that
share this pool under enforce WITHOUT a pin.

These tests run under a throwaway non-owner ``NOBYPASSRLS`` role — the prod
runtime posture (``w2v_orch_app``) — with ``TENANT_RLS_MODE=enforce``. That is
what makes them exercise the real enforce path: the test DB user OWNS the tables
and RLS is ``ENABLE`` (not ``FORCE``), so as the owner every policy is bypassed,
``row_security = off`` is a no-op, and the whole class of bug is invisible.
``test_poisoned_row_security_off_reproduces_the_live_error`` is the vacuity guard
for exactly that: it asserts the harness really does reproduce the production
error before the other tests assert it is gone.
"""

from __future__ import annotations

import logging
from uuid import UUID, uuid7

import pytest
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError
from w2v_common.auth.identity_context import clear_identity_context

from src.config import clear_settings_cache
from src.database.models import Deliverable, Project, Run
from src.database.rls_events import install_rls_tenant_rebind
from src.database.session import (
    _ROW_SECURITY_WARN_INTERVAL_S,
    bind_run_tenant,
    bind_session_tenant,
    reset_row_security_warn_throttle,
    warn_if_row_security_was_leaked,
)
from tests.conftest import require_createrole

pytestmark = pytest.mark.asyncio

_LIVE_ERROR = "query would be affected by row-level security policy"
#: The alarm the pin must NOT delete. See ``warn_if_row_security_was_leaked``.
_ALARM = "row_security was 'off' on a pooled server connection"


@pytest.fixture(autouse=True)
def _reset_settings_and_context():
    clear_settings_cache()
    clear_identity_context()
    # The leaked-GUC warning is throttled process-globally; without this reset the
    # second test in the file would find the throttle armed by the first and see no
    # log line, so the alarm assertions would pass or fail by test ORDER.
    reset_row_security_warn_throttle()
    yield
    clear_settings_cache()
    clear_identity_context()
    reset_row_security_warn_throttle()


async def _unpoison(session) -> None:
    """Undo ``_poison`` DURABLY.

    ``SET`` is TRANSACTIONAL in Postgres: issued inside the transaction SQLAlchemy
    autobegins, it is REVERTED by a rollback. The original version of this cleanup
    ended in ``await s.rollback()``, which therefore handed the connection back to
    the pool still ``SET ROLE``d and still ``row_security = off`` — a no-op cleanup
    that only looked harmless because ``db_engine`` is per-test and disposed
    (``tests/conftest.py``). Make ``db_engine`` session-scoped and it poisons the
    whole suite. So COMMIT it. Do NOT "optimise" this back to a rollback.
    """
    await session.rollback()  # discard whatever state the assertions left behind
    await session.execute(text("SET row_security = on"))
    await session.execute(text("RESET ROLE"))
    await session.commit()  # load-bearing — a rollback would revert both SETs
    # Mutation guard for the RESET ROLE half: a fresh transaction on the SAME
    # connection must no longer be the probe role. Swap the commit above back to a
    # rollback and this fails. The row_security half cannot be asserted here (the
    # after_begin pin sets it transaction-locally, so any read in a later
    # transaction says 'on' either way) — it is proven by
    # ``test_cleanup_of_row_security_survives_the_transaction`` with the listener
    # inert.
    who = (await session.execute(text("SELECT current_user"))).scalar_one()
    assert not str(who).startswith("rls_rowsec_probe_"), (
        f"cleanup did not survive: connection returns to the pool as {who!r}"
    )
    await session.commit()


async def _seed_run_tree(session, tenant_id: UUID) -> UUID:
    project = Project(
        id=uuid7(),
        tenant_id=tenant_id,
        owner_subject_id=uuid7(),
        visibility="private",
        type="promo",
        title=f"proj-{tenant_id}",
    )
    deliverable = Deliverable(id=uuid7(), project_id=project.id, name="d1", format="16x9")
    run = Run(id=uuid7(), deliverable_id=deliverable.id, blueprint_id="promo_v1", status="running")
    session.add_all([project, deliverable, run])
    await session.commit()
    return run.id


async def _run_visible(session, run_id: UUID) -> bool:
    found = (
        await session.execute(text("SELECT id FROM runs WHERE id = :rid"), {"rid": str(run_id)})
    ).scalar_one_or_none()
    return found is not None


async def _fail_run(session, run_id: UUID) -> int:
    """The reconciler's ``_resolve_orphan_run`` stale-orphan UPDATE, verbatim shape."""
    result = await session.execute(
        text("UPDATE runs SET status = 'failed', error = :e WHERE id = :rid"),
        {"e": "orphan run exceeded reconciler max-age", "rid": str(run_id)},
    )
    return result.rowcount


@pytest.fixture
async def restricted_role(db_session_factory):
    """A throwaway ``NOBYPASSRLS`` non-owner role to ``SET ROLE`` into.

    Mirrors ``tests/database/test_rls_rebind_listener.py``'s fixture, plus the
    ``UPDATE`` and SECURITY-DEFINER ``EXECUTE`` grants prod's ``w2v_orch_app``
    carries — this file exercises the reconciler's WRITE and the dispatcher's
    ``run_tenant()`` bootstrap, not just reads.
    """
    role = f"rls_rowsec_probe_{uuid7().hex[:12]}"
    async with db_session_factory() as owner:
        schema = (await owner.execute(text("SELECT current_schema()"))).scalar_one()
        try:
            await owner.execute(text(f'CREATE ROLE "{role}" NOLOGIN NOSUPERUSER NOBYPASSRLS'))
        except ProgrammingError as exc:  # insufficient_privilege → role lacks CREATEROLE
            await owner.rollback()
            require_createrole(exc)
        await owner.execute(text(f'GRANT "{role}" TO CURRENT_USER'))
        await owner.execute(text(f'GRANT USAGE ON SCHEMA "{schema}" TO "{role}"'))
        await owner.execute(
            text(f'GRANT SELECT, UPDATE ON ALL TABLES IN SCHEMA "{schema}" TO "{role}"')
        )
        for fn in ("run_tenant(uuid)", "orphan_runs()"):
            await owner.execute(text(f'GRANT EXECUTE ON FUNCTION {fn} TO "{role}"'))
        await owner.commit()
    try:
        yield role
    finally:
        async with db_session_factory() as owner:
            teardown_schema = (await owner.execute(text("SELECT current_schema()"))).scalar_one()
            for stmt in (
                f'REVOKE ALL ON ALL TABLES IN SCHEMA "{teardown_schema}" FROM "{role}"',
                f'REVOKE ALL ON FUNCTION run_tenant(uuid) FROM "{role}"',
                f'REVOKE ALL ON FUNCTION orphan_runs() FROM "{role}"',
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


async def _poison(session, role: str) -> None:
    """Put the session in the exact state a poisoned PgBouncer server connection is
    in: acting as the restricted runtime role, with ``row_security`` switched off at
    SESSION scope. The COMMIT is load-bearing — it ends the transaction the ``SET``s
    were issued in (they survive it, a ``SET LOCAL`` would not), so every assertion
    below runs in a FRESH transaction whose ``after_begin`` fires with the poison
    already in place, exactly as a borrowed pooled connection does.
    """
    await session.execute(text(f'SET ROLE "{role}"'))
    await session.execute(text("SET row_security = off"))
    await session.commit()


async def test_poisoned_row_security_off_reproduces_the_live_error(
    db_session_factory, restricted_role, monkeypatch
):
    """Vacuity guard: without the pin, the harness raises the PRODUCTION error.

    ``TENANT_RLS_MODE`` is left ``off`` so the ``after_begin`` listener (and with
    it the pin) is inert — this is the un-defended baseline. If this test ever
    stops raising, the other three prove nothing and this file must be re-derived.
    """
    monkeypatch.setenv("TENANT_RLS_MODE", "off")
    clear_settings_cache()

    tenant = uuid7()
    async with db_session_factory() as owner:
        run_id = await _seed_run_tree(owner, tenant)

    async with db_session_factory() as s:
        await _poison(s, restricted_role)
        with pytest.raises(ProgrammingError) as exc_info:
            await _run_visible(s, run_id)
        assert _LIVE_ERROR in str(exc_info.value)
        await _unpoison(s)


async def test_reconciler_write_survives_a_poisoned_row_security_off(
    db_session_factory, restricted_role, monkeypatch, caplog
):
    """``_resolve_orphan_run``'s stale-orphan UPDATE, on a poisoned connection.

    The reconciler binds the orphan row's tenant (``bind_session_tenant``) and then
    UPDATEs ``runs`` — for a tenant that is NOT in any ambient context (the scan is
    cross-tenant, sourced from the ``orphan_runs()`` SECURITY DEFINER resolver, and
    the reconciler task has no identity ContextVar at all). Without the pin this is
    the 42501 that crash-looped the loop; with it the write lands.
    """
    monkeypatch.setenv("TENANT_RLS_MODE", "enforce")
    clear_settings_cache()
    install_rls_tenant_rebind()

    tenant, other_tenant = uuid7(), uuid7()
    async with db_session_factory() as owner:
        run_id = await _seed_run_tree(owner, tenant)
        other_run_id = await _seed_run_tree(owner, other_tenant)

    caplog.set_level(logging.WARNING, logger="src.database.session")
    async with db_session_factory() as s:
        await _poison(s, restricted_role)
        await bind_session_tenant(s, tenant)
        assert await _run_visible(s, run_id), "bound tenant's orphan must be readable"
        assert await _fail_run(s, run_id) == 1, (
            "the reconciler's stale-orphan UPDATE must affect the row "
            "(regression: 42501 query would be affected by row-level security policy)"
        )
        # Pinning row_security ON is the STRICT value: it means "apply the policy".
        # Prove it did not become a bypass — another tenant's run stays invisible
        # and unwritable even though we are on the same poisoned connection.
        assert not await _run_visible(s, other_run_id)
        assert await _fail_run(s, other_run_id) == 0
        # The pin must not DELETE THE ALARM. >=7 sibling services share this
        # PgBouncer pool under enforce with no pin and are still 500ing; the
        # orchestrator is the only one that can name the cause, so it must.
        assert any(_ALARM in r.getMessage() for r in caplog.records), (
            "self-healing silently would erase the only diagnosis of a leaked "
            "row_security=off on the shared pgbouncer pool"
        )
        await _unpoison(s)


async def test_dispatcher_resume_survives_a_poisoned_row_security_off(
    db_session_factory, restricted_role, monkeypatch
):
    """The resumed ``RunDispatcher._run`` task, on a poisoned connection.

    Its very first statement is ``bind_run_tenant``'s ``SELECT run_tenant(:rid)`` —
    a SECURITY DEFINER function that reads ``runs``, so it raises 42501 on a
    poisoned connection *before* any tenant can be bound. Fixing only the
    reconciler would resume the run straight back into the same death; this asserts
    the bootstrap AND a subsequent write both work.
    """
    monkeypatch.setenv("TENANT_RLS_MODE", "enforce")
    clear_settings_cache()
    install_rls_tenant_rebind()

    tenant = uuid7()
    async with db_session_factory() as owner:
        run_id = await _seed_run_tree(owner, tenant)

    async with db_session_factory() as s:
        await _poison(s, restricted_role)
        await bind_run_tenant(s, run_id)  # the dispatcher's RLS bootstrap
        assert await _run_visible(s, run_id), "dispatcher must read its own run row"
        assert await _fail_run(s, run_id) == 1, "dispatcher's status write must land"
        await _unpoison(s)


async def test_untenanted_orphan_scan_survives_a_poisoned_row_security_off(
    db_session_factory, restricted_role, monkeypatch, caplog
):
    """``_reconcile_orphan_runs``'s cross-tenant scan has NO tenant to bind.

    It runs with nothing in ``session.info`` and no identity ContextVar, so the
    ``after_begin`` listener's no-tenant branch is the only thing that can pin
    ``row_security`` for it. Covers the branch the tenant-bound tests above never
    reach.

    Note ``orphan_runs()`` itself is SECURITY DEFINER and executes as the table
    OWNER, who bypasses RLS before ``row_security`` is ever consulted — it survives
    a poisoned connection unaided (asserted here so that stays true, but it carries
    no teeth). The teeth are the direct, unbound read of ``runs`` on the same
    transaction: under enforce that must return NO ROWS (the policy predicate is
    false with an unbound tenant), never RAISE. Raising is what took the loop down.

    ⚠️ This test asserts 0 rows where the pre-pin code RAISED — i.e. it converts a
    loud failure into a quiet one — so it owes an explicit justification:

    1. 0 rows is the CORRECT semantics for an unbound read. An unbound
       ``app.current_tenant`` makes the RLS predicate false; every other unbound
       read in the service returns 0 rows. The 42501 raise was never the tenancy
       signal — it fires on a poisoned connection regardless of whether a tenant is
       bound, so it said "the pool is poisoned", not "this scan lost its tenant".
    2. The signal is NOT lost: it moved to where it is actionable. The same
       statement that pins the GUC reads the value the connection arrived with and
       logs a WARNING naming pgbouncer + the unpinned sibling services — asserted
       below. That is strictly MORE information than the 42501 ever carried.
    3. ``bind_run_tenant``'s fail-loud contract ("must not execute silently against
       0 rows", ``session.py``) is untouched and NOT in conflict: it governs a run
       whose tenant cannot be RESOLVED, and it still raises ``RuntimeError``. This
       scan is cross-tenant BY DESIGN and has no tenant to resolve.
    """
    monkeypatch.setenv("TENANT_RLS_MODE", "enforce")
    clear_settings_cache()
    install_rls_tenant_rebind()

    tenant = uuid7()
    async with db_session_factory() as owner:
        run_id = await _seed_run_tree(owner, tenant)

    caplog.set_level(logging.WARNING, logger="src.database.session")
    async with db_session_factory() as s:
        await _poison(s, restricted_role)
        rows = (await s.execute(text("SELECT run_id, tenant_id FROM orphan_runs()"))).all()
        assert run_id in {r.run_id for r in rows}, (
            "the reconciler's untenanted cross-tenant orphan scan must survive a "
            "poisoned row_security=off connection"
        )
        assert not await _run_visible(s, run_id), (
            "an unbound read must be RLS-scoped to zero rows, not a 42501 raise"
        )
        # Point 2 of the docstring, enforced: the 0-row result is only acceptable
        # because the WARNING carries the diagnosis. If this assertion ever goes,
        # the branch really has gone silent and the 0-row assertion above must be
        # re-litigated rather than kept.
        assert any(_ALARM in r.getMessage() for r in caplog.records), (
            "the no-tenant branch returns 0 rows instead of raising; the WARNING is "
            "the ONLY remaining signal that the connection was poisoned"
        )
        await _unpoison(s)


async def test_a_clean_connection_raises_no_alarm(
    db_session_factory, restricted_role, monkeypatch, caplog
):
    """The alarm must be silent on the overwhelmingly common healthy path.

    Every transaction in the service runs the probe, so a warning that fired on a
    clean connection would be one line per transaction forever — the operator would
    filter it out and the real one with it.
    """
    monkeypatch.setenv("TENANT_RLS_MODE", "enforce")
    clear_settings_cache()
    install_rls_tenant_rebind()

    tenant = uuid7()
    async with db_session_factory() as owner:
        run_id = await _seed_run_tree(owner, tenant)

    caplog.set_level(logging.WARNING, logger="src.database.session")
    async with db_session_factory() as s:
        # SET ROLE only — row_security is left at its 'on' default.
        await s.execute(text(f'SET ROLE "{restricted_role}"'))
        await s.commit()
        await bind_session_tenant(s, tenant)
        assert await _run_visible(s, run_id)
        assert not any(_ALARM in r.getMessage() for r in caplog.records), (
            "the leaked-GUC warning fired on a connection that was never poisoned"
        )
        await _unpoison(s)


async def test_leaked_guc_warning_is_throttled_and_reports_what_it_suppressed(caplog):
    """A poisoned pooled connection is re-borrowed by EVERY transaction until
    PgBouncer retires it, so the alarm has to be throttled or it is a log flood.

    Asserts the three properties that make the throttle honest: the first
    occurrence is loud, the immediately following ones are swallowed, and the count
    of swallowed ones is carried into the next line that does get through.
    """
    reset_row_security_warn_throttle()
    caplog.set_level(logging.WARNING, logger="src.database.session")

    assert warn_if_row_security_was_leaked("on") is False, "'on' is not a leak"
    assert warn_if_row_security_was_leaked(None) is False, "NULL is not a leak"
    assert caplog.records == []

    assert warn_if_row_security_was_leaked("off") is True
    assert _ALARM in caplog.records[-1].getMessage()
    assert "pool_mode=transaction" in caplog.records[-1].getMessage()
    assert "0 further occurrence(s) suppressed" in caplog.records[-1].getMessage()

    for _ in range(5):
        assert warn_if_row_security_was_leaked("off") is False, "throttle must swallow"
    assert len(caplog.records) == 1

    # Age the throttle past its window and the next one gets through, carrying the
    # count of what was swallowed.
    import src.database.session as session_mod

    last, suppressed = session_mod._row_security_warn_state
    session_mod._row_security_warn_state = (last - _ROW_SECURITY_WARN_INTERVAL_S - 1, suppressed)
    assert warn_if_row_security_was_leaked("off") is True
    assert "5 further occurrence(s) suppressed" in caplog.records[-1].getMessage()
    reset_row_security_warn_throttle()


async def test_cleanup_of_row_security_survives_the_transaction(
    db_session_factory, restricted_role, monkeypatch
):
    """``_unpoison`` must COMMIT, because ``SET`` is transactional.

    ``TENANT_RLS_MODE`` is left ``off`` so the ``after_begin`` pin is inert and
    ``current_setting('row_security')`` reports the CONNECTION's real state rather
    than this transaction's local pin. Turn ``_unpoison``'s final ``commit()`` back
    into a ``rollback()`` and this reads ``'off'``: the connection goes back to the
    pool still poisoned, which is exactly the no-op cleanup this file used to ship.
    """
    monkeypatch.setenv("TENANT_RLS_MODE", "off")
    clear_settings_cache()

    async with db_session_factory() as s:
        await _poison(s, restricted_role)
        await _unpoison(s)
        assert (await s.execute(text("SELECT current_setting('row_security')"))).scalar_one() == (
            "on"
        ), "SET is transactional — the cleanup must be committed, not rolled back"
        await s.commit()
