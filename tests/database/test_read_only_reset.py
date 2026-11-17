"""A leaked ``default_transaction_read_only = on`` must not break our writes.

The sibling of ``test_row_security_pin.py``, for the OTHER GUC PgBouncer's
``server_reset_query`` does not scrub. Under ``pool_mode = transaction`` the reset
covers only ``RESET row_security``, so ANY session-level ``SET`` a pool-sharing
client leaves behind sticks on that server connection and every later borrower
inherits it — permanently, and on that ONE backend, which is why it presents as a
flaky application bug rather than an outage.

Live symptom: ``psycopg.errors.ReadOnlySqlTransaction: cannot execute UPDATE in a
read-only transaction``, ~15/hr/replica on 2026-08-23, on the preview-proxy
sweeper's bookkeeping write — landing AFTER the transcode and the MinIO upload had
already succeeded, so the pod burned the whole expensive job and then could not
record that it had done it.

These tests run against a REAL postgres because the entire question is one of
Postgres semantics: ``SET LOCAL transaction_read_only = off`` is legal only before
the transaction's first snapshot, and no mock can tell you that. That ordering is
the reason the reset is a separate statement issued first in the ``after_begin``
listener rather than a third ``set_config`` folded into the pin/bind SELECT.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from sqlalchemy import event, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DBAPIError
from w2v_common.auth.identity_context import clear_identity_context

from src.config import clear_settings_cache
from src.database.rls_events import install_rls_tenant_rebind
from src.database.session import (
    _ROW_SECURITY_WARN_INTERVAL_S,
    RESET_READ_WRITE_SQL,
    backend_is_in_recovery,
    note_read_only_reset_rejected,
    reset_backend_recovery_state,
    reset_row_security_warn_throttle,
    set_backend_in_recovery,
    warn_if_read_only_was_leaked,
)

pytestmark = pytest.mark.asyncio

#: The exact wire error a poisoned backend produces. Asserted verbatim so this file
#: stays tied to the live symptom rather than to our paraphrase of it.
_LIVE_ERROR = "cannot execute UPDATE in a read-only transaction"
#: The alarm the reset must NOT delete — matched on the POISONED value.
_ALARM = "default_transaction_read_only was 'on' on a pooled server connection"
#: Value-INDEPENDENT marker for "the read-only alarm fired at all". The healthy-path
#: test must use this, not ``_ALARM``: an alarm that misclassifies a healthy 'off'
#: as a leak logs "was 'off' …", which ``_ALARM`` does not match, so asserting
#: ``_ALARM`` there would pass with the classification broken.
_ALARM_ANY = "on a pooled server connection and has been overridden to read-write"


@pytest.fixture(autouse=True)
def _reset_settings_and_context():
    clear_settings_cache()
    clear_identity_context()
    reset_row_security_warn_throttle()
    # Process-global, like the throttles: a test that leaves the standby latch set
    # would silently disable the reset for every later test in the session.
    reset_backend_recovery_state()
    yield
    clear_settings_cache()
    clear_identity_context()
    reset_row_security_warn_throttle()
    reset_backend_recovery_state()


@pytest.fixture
async def probe_table(db_session_factory):
    """A tiny non-RLS table to write to.

    Deliberately NOT one of the RLS-gated business tables: this file is about the
    read-only poison ALONE, and mixing in tenant policies would let an RLS failure
    masquerade as a read-only failure (and vice versa).
    """
    name = "_ro_reset_probe"
    async with db_session_factory() as owner:
        await owner.execute(text(f"CREATE TABLE IF NOT EXISTS {name} (id int primary key, v int)"))
        await owner.execute(text(f"TRUNCATE {name}"))
        await owner.execute(text(f"INSERT INTO {name} VALUES (1, 0)"))  # noqa: S608
        await owner.commit()
    yield name
    async with db_session_factory() as owner:
        await owner.execute(text(f"DROP TABLE IF EXISTS {name}"))
        await owner.commit()


async def _poison(session) -> None:
    """Put the session in the exact state a poisoned PgBouncer backend is in.

    The COMMIT is load-bearing: it ends the transaction the ``SET`` was issued in
    (a session-level ``SET`` survives that, a ``SET LOCAL`` would not), so every
    assertion below runs in a FRESH transaction whose ``after_begin`` fires with
    the poison already in place — exactly as a borrowed pooled connection does.
    """
    await session.execute(text("SET default_transaction_read_only = on"))
    await session.commit()


async def _unpoison(session) -> None:
    """Undo ``_poison`` DURABLY. ``SET`` is transactional, so a rollback here would
    hand the connection back to the pool still read-only."""
    await session.rollback()
    await session.execute(text("SET default_transaction_read_only = off"))
    await session.commit()


async def _bump(session, table: str) -> int:
    result = await session.execute(text(f"UPDATE {table} SET v = v + 1 WHERE id = 1"))  # noqa: S608
    return result.rowcount


async def test_poison_really_does_break_writes_without_the_listener(
    db_session_factory, probe_table, monkeypatch
):
    """The CONTROL. Proves the poison is real and reproduces the live error.

    Without this, every other assertion in the file could pass against a database
    that was never poisoned at all. ``TENANT_RLS_MODE`` is left ``off`` so the
    ``after_begin`` listener — and with it the reset — is inert.
    """
    monkeypatch.setenv("TENANT_RLS_MODE", "off")
    clear_settings_cache()
    install_rls_tenant_rebind()

    async with db_session_factory() as s:
        await _poison(s)
        with pytest.raises(DBAPIError) as exc_info:
            await _bump(s, probe_table)
        assert _LIVE_ERROR in str(exc_info.value)
        await _unpoison(s)


async def test_a_write_survives_a_poisoned_read_only_backend(
    db_session_factory, probe_table, monkeypatch, caplog
):
    """The fix. Same poison, listener enforcing → the UPDATE lands.

    This is the preview-proxy sweeper's bookkeeping write: the work is already done
    and paid for, and all that is left is to record it.
    """
    monkeypatch.setenv("TENANT_RLS_MODE", "enforce")
    clear_settings_cache()
    install_rls_tenant_rebind()

    caplog.set_level(logging.WARNING, logger="src.database.session")
    async with db_session_factory() as s:
        await _poison(s)
        assert await _bump(s, probe_table) == 1, (
            f"the bookkeeping write must land on a poisoned backend (regression: {_LIVE_ERROR})"
        )
        await s.commit()

        # The reset must not DELETE THE ALARM. Sibling w2v services share this
        # PgBouncer pool and have no reset of their own; if we self-healed silently
        # the one service that can NAME the root cause would say nothing.
        assert any(_ALARM in r.getMessage() for r in caplog.records), (
            "self-healing silently erases the only diagnosis of a leaked "
            "default_transaction_read_only=on on the shared pgbouncer pool"
        )
        await _unpoison(s)


async def test_the_reset_is_transaction_local_and_never_leaks_back(
    db_session_factory, probe_table, monkeypatch
):
    """``SET LOCAL``, not ``SET``. The fix must not become a leak of our own on the
    shared pool: after our transaction commits, the session-level setting the
    poisoning client left behind is still exactly as we found it."""
    monkeypatch.setenv("TENANT_RLS_MODE", "enforce")
    clear_settings_cache()
    install_rls_tenant_rebind()

    async with db_session_factory() as s:
        await _poison(s)
        assert await _bump(s, probe_table) == 1
        await s.commit()

        # Now make the listener INERT and try again on the SAME connection. The
        # poison must still be there: our reset was only ever meant to cover the one
        # transaction that needed it.
        #
        # Asserting ``current_setting('default_transaction_read_only') == 'on'`` here
        # instead is VACUOUS — a session-scoped ``SET transaction_read_only = off``
        # would not change ``default_transaction_read_only`` either, so that
        # assertion passes with the bug present. Only re-attempting the write with
        # the listener disabled can tell the two apart.
        monkeypatch.setenv("TENANT_RLS_MODE", "off")
        clear_settings_cache()

        with pytest.raises(DBAPIError) as exc_info:
            await _bump(s, probe_table)
        assert _LIVE_ERROR in str(exc_info.value), (
            "the reset escaped its transaction and durably un-poisoned the pooled "
            "connection — a session-level SET here makes this service a POISONER of "
            "the shared pool (every later borrower silently inherits read-write) "
            "rather than a victim that heals itself"
        )
        await _unpoison(s)


async def test_a_healthy_backend_writes_and_raises_no_alarm(
    db_session_factory, probe_table, monkeypatch, caplog
):
    """The unpoisoned path. The reset must be a no-op that neither breaks the write
    nor cries wolf — an alarm that fires on every healthy transaction is noise, and
    noise is how the real one gets ignored."""
    monkeypatch.setenv("TENANT_RLS_MODE", "enforce")
    clear_settings_cache()
    install_rls_tenant_rebind()

    caplog.set_level(logging.WARNING, logger="src.database.session")
    async with db_session_factory() as s:
        assert await _bump(s, probe_table) == 1
        await s.commit()

    assert not any(_ALARM_ANY in r.getMessage() for r in caplog.records)


def test_the_alarm_classifies_and_throttles(monkeypatch):
    """Unit-level truth table for the warn helper, asserted on its RETURN value so
    the decision is pinned rather than log plumbing."""
    reset_row_security_warn_throttle()
    assert warn_if_read_only_was_leaked("off") is False, "'off' is not a leak"
    assert warn_if_read_only_was_leaked(None) is False, "NULL is not a leak"

    assert warn_if_read_only_was_leaked("on") is True
    # A poisoned backend is borrowed by every transaction until PgBouncer retires
    # it, so an unthrottled alarm would be thousands of lines per minute.
    assert warn_if_read_only_was_leaked("on") is False, "throttle must swallow"

    # ...and it must open again once the window has passed. Jump the clock the
    # module reads rather than sleeping ``_ROW_SECURITY_WARN_INTERVAL_S`` seconds.
    from src.database import session as session_mod

    monkeypatch.setattr(
        session_mod.time, "monotonic", lambda: 10_000_000.0 + _ROW_SECURITY_WARN_INTERVAL_S
    )
    assert warn_if_read_only_was_leaked("on") is True


def test_the_two_alarms_do_not_silence_each_other():
    """Independent throttles. A shared one would let whichever poison fired first
    mute the other for a minute — hiding half the diagnosis in exactly the incident
    where both matter."""
    from src.database.session import warn_if_row_security_was_leaked

    reset_row_security_warn_throttle()
    assert warn_if_row_security_was_leaked("off") is True
    assert warn_if_read_only_was_leaked("on") is True, (
        "the row_security alarm swallowed the read-only alarm — shared throttle"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# A STANDBY MUST DEGRADE, NOT DIE
#
# ``SET LOCAL transaction_read_only = off`` is rejected during recovery (0A000
# "cannot set transaction read-write mode during recovery"). It is the FIRST
# statement of every transaction, so on a standby an unconditional reset turns
# "reads work, writes fail" into "every transaction dies at BEGIN" — health probe
# included. This fleet has a real standby: postgres.test answers on the write
# PRIMARY while the standby node runs a read-only standby on its own :5432.
#
# ⚠️ The RECOVERY CONDITION here is SIMULATED — the latch these tests set is the
# same one the startup probe writes, but no live standby was reachable from the
# development box, so "the server rejects the statement" is modelled, not measured.
# What IS measured against real postgres 18: that with the latch set the reset is
# not issued at all (the poisoned write goes back to failing), and that reads keep
# working in that state.
# ═══════════════════════════════════════════════════════════════════════════════


@contextmanager
def _statements_seen() -> Iterator[list[str]]:
    """Every SQL string the engine sends, for the duration of the block.

    The assertion has to be "the reset was NOT SENT", not "the read succeeded": on
    the CI primary the reset succeeds harmlessly, so a success-only assertion is
    VACUOUS — it passes with the gate deleted. Only the wire tells the two apart.
    """
    seen: list[str] = []

    def _record(conn, cursor, statement: str, params, context, executemany) -> None:
        seen.append(statement)

    event.listen(Engine, "before_cursor_execute", _record)
    try:
        yield seen
    finally:
        event.remove(Engine, "before_cursor_execute", _record)


async def test_a_standby_still_serves_reads_and_is_never_sent_the_reset(
    db_session_factory, probe_table, monkeypatch
):
    """The BLOCKER this gate exists for. With the backend in recovery the listener
    must not issue the reset — and a pure read must still complete."""
    monkeypatch.setenv("TENANT_RLS_MODE", "enforce")
    clear_settings_cache()
    install_rls_tenant_rebind()
    set_backend_in_recovery(True)

    with _statements_seen() as seen:
        async with db_session_factory() as s:
            value = (await s.execute(text(f"SELECT v FROM {probe_table} WHERE id = 1"))).scalar()  # noqa: S608
            await s.commit()

    assert value == 0
    assert RESET_READ_WRITE_SQL not in seen, (
        "the read-write reset was sent to a backend in recovery. Postgres rejects it "
        "with 0A000 there, and it is the transaction's FIRST statement — so on a real "
        "standby this read, and every other transaction including the health probe, "
        "dies at BEGIN. That is a total outage where the service used to degrade."
    )


async def test_the_reset_is_skipped_when_the_backend_is_in_recovery(
    db_session_factory, probe_table, monkeypatch
):
    """The other half: prove the SKIP is real and not merely harmless.

    Same poison as ``test_a_write_survives_a_poisoned_read_only_backend``, but with
    the recovery latch set — the write must go back to failing, which can only
    happen if the reset was never issued. Asserting "no exception" alone would pass
    with the gate wired backwards.
    """
    monkeypatch.setenv("TENANT_RLS_MODE", "enforce")
    clear_settings_cache()
    install_rls_tenant_rebind()
    set_backend_in_recovery(True)

    async with db_session_factory() as s:
        await _poison(s)
        with pytest.raises(DBAPIError) as exc_info:
            await _bump(s, probe_table)
        assert _LIVE_ERROR in str(exc_info.value), (
            "the reset ran anyway — on a real standby this statement raises 0A000 "
            "and kills the transaction instead"
        )
        await _unpoison(s)


async def test_the_reset_is_still_issued_on_a_primary(db_session_factory, probe_table, monkeypatch):
    """The gate must not have disabled the fix everywhere. Default state (no
    standby latched) → the poisoned write still lands."""
    monkeypatch.setenv("TENANT_RLS_MODE", "enforce")
    clear_settings_cache()
    install_rls_tenant_rebind()

    assert backend_is_in_recovery() is False, "the default must be 'assume primary'"
    with _statements_seen() as seen:
        async with db_session_factory() as s:
            await _poison(s)
            assert await _bump(s, probe_table) == 1
            await s.commit()
            await _unpoison(s)

    assert RESET_READ_WRITE_SQL in seen, "the gate disabled the fix on a primary too"


def test_a_recovery_rejection_latches_but_an_unrelated_error_does_not():
    """The failover safety net: the probe answers for the connection the engine was
    built on, so a DSN that STARTS answering from a standby later can only be caught
    from the rejection itself. One transaction pays; the rest degrade.

    SIMULATED: the 0A000 rejection is constructed, not obtained from a live standby.
    """
    from psycopg.errors import FeatureNotSupported
    from psycopg.errors import ProgrammingError as PgProgrammingError

    stmt = "SET LOCAL transaction_read_only = off"
    # ``FeatureNotSupported`` carries sqlstate 0A000 as a CLASS attribute (verified),
    # which is what the latch matches on — the message text is localised and must
    # not be the discriminator.
    recovery = DBAPIError(
        stmt, None, FeatureNotSupported("cannot set transaction read-write mode during recovery")
    )
    unrelated = DBAPIError(stmt, None, PgProgrammingError("syntax error"))

    assert note_read_only_reset_rejected(unrelated) is False
    assert backend_is_in_recovery() is False, "an unrelated error must not disable the reset"

    assert note_read_only_reset_rejected(recovery) is True
    assert backend_is_in_recovery() is True, (
        "the latch did not engage — every later transaction would die on the same "
        "rejection instead of degrading to read-only"
    )


class _FakeEngineRaising:
    """An engine whose ``connect()`` misbehaves in a way the probe never listed.

    Returning a bare coroutine (rather than an ``AsyncConnection``, which IS an async
    context manager) is not hypothetical: it is exactly what a mocked/adapted engine
    hands back, and it makes ``async with`` raise ``TypeError`` — a class no
    connectivity-flavoured except-tuple contains.
    """

    def __init__(self, exc: BaseException | None = None) -> None:
        self.exc = exc
        self.connect_calls = 0

    def connect(self):
        self.connect_calls += 1
        if self.exc is not None:
            raise self.exc

        async def _not_a_context_manager():
            return None

        return _not_a_context_manager()


async def test_get_engine_survives_a_probe_that_blows_up(monkeypatch):
    """End-to-end shape of the regression: a probe failure must not propagate out of
    ``get_engine`` and take the whole service's startup with it."""
    import src.database.connection as conn_module
    from src.database.connection import get_engine

    reset_backend_recovery_state()
    engine = _FakeEngineRaising()

    async def _fake_create(*_args, **_kwargs):
        return engine

    monkeypatch.setattr(conn_module, "create_engine_with_retry", _fake_create)
    monkeypatch.setattr(conn_module, "_engine", None)
    try:
        assert await get_engine() is engine
    finally:
        conn_module._engine = None
    assert engine.connect_calls == 1, "the probe did not run; this asserts nothing about it"


class _FakeStandbyEngine:
    """An engine whose backend answers ``pg_is_in_recovery() = t``.

    The live-postgres probe test can only assert ``False`` (the CI database is a
    primary), so it passes identically against a probe hard-wired to ``False``. This
    is the other half: it fails unless the probe writes what the backend REPORTED.
    """

    def __init__(self) -> None:
        self.sql: list[str] = []

    def connect(self):
        engine = self

        class _Conn:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                return False

            async def execute(self, stmt):
                engine.sql.append(str(stmt))

                class _Result:
                    def scalar(self):
                        return True

                return _Result()

        return _Conn()
