"""Unit tests for the preview-proxy sweeper (``src/lifecycle/preview_proxy.py``).

DELIBERATELY UNMARKED. ``pyproject.toml``'s addopts are
``-m 'not integration and not cross_service_monorepo'`` and CI never re-selects
the excluded markers, so a load-bearing assertion behind ``@pytest.mark.integration``
would never run anywhere. The first half is DB-free (mock session, same idiom as
``tests/unit/test_asgi_reconciler.py``); the DURABILITY half at the bottom needs a
real postgres, because "the counter reached the DATABASE before ffmpeg started" is
precisely the thing a mock session cannot answer — it is unmarked too, exactly like
``tests/database/*``, and runs on the ordinary ``db_session_factory`` fixture.

What is pinned:

* a pending row gets a proxy, goes ``ready``, and has ``preview_key`` written;
* a row that is no longer ``pending`` at claim time is SKIPPED, not rebuilt;
* a failing transcode counts an attempt and stays ``pending``, then writes
  ``failed`` + ``error`` on the last attempt;
* the advisory lock keeps two workers from both building one row;
* the module semaphore bounds how many transcodes run at once;
* the staging directory is reclaimed even when the transcode raises;
* the object key is EXACTLY ``_source_prefix(...) + "preview.mp4"``;
* **the attempt counter and the lease are COMMITTED before ``_build_proxy`` runs**,
  read back through an INDEPENDENT connection — the fix for the 2026-08-22 defect
  where a pod SIGKILLed mid-transcode wrote nothing and the row was re-picked
  forever;
* a row already at ``max_attempts`` is retired at CLAIM time, without transcoding;
* every terminal write clears the lease;
* ``pending_video_sources`` hides a row leased into the future and re-offers one
  whose lease has expired.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4, uuid7

import pytest

from src.config import get_settings
from src.lifecycle.preview_proxy import (
    PendingSource,
    PreviewProxyError,
    _lease_seconds,
    _preview_key,
    _process_source,
    _scan_pending,
    sweep_pending_previews,
)

pytestmark = pytest.mark.asyncio

#: The lease timestamp the fake claim hands back — the worker's claim token.
_CLAIM_LEASE = datetime(2026, 8, 23, 12, 0, tzinfo=UTC)


def _source(*, attempts: int = 0) -> PendingSource:
    return PendingSource(
        source_id=uuid4(),
        tenant_id=uuid4(),
        project_id=uuid4(),
        object_key="tenants/t/projects/p/video-sources/s/original.mov",
        attempts=attempts,
    )


class _FakeSession:
    """Records every statement + params the worker issues.

    Not an ``AsyncMock`` with a side_effect list: the assertions here are about
    WHICH statement ran with WHICH values (did it write ``'ready'``? did the
    attempts counter go up?), and a positional side_effect list would let a
    reordering pass silently.
    """

    def __init__(
        self,
        *,
        locked: bool = True,
        status: str | None = "pending",
        claimable: bool = True,
        terminal_rowcount: int = 1,
    ) -> None:
        self._locked = locked
        self._status = status
        self._claimable = claimable
        #: Rows the FENCED terminal write reports. 0 models "this worker's lease
        #: expired and someone else re-claimed the row", which is the only way the
        #: ownership fence can be observed from outside the database.
        self._terminal_rowcount = terminal_rowcount
        self.statements: list[tuple[str, dict[str, object]]] = []
        self.commits = 0

    async def execute(self, stmt: object, params: dict[str, object] | None = None) -> MagicMock:
        sql = str(stmt)
        self.statements.append((sql, dict(params or {})))
        result = MagicMock()
        if "pg_try_advisory_xact_lock" in sql:
            result.scalar = MagicMock(return_value=self._locked)
        elif "SELECT preview_status" in sql:
            result.scalar_one_or_none = MagicMock(return_value=self._status)
        elif "preview_leased_until = now()" in sql:
            # The durable claim's ``RETURNING id, preview_leased_until``. Explicit
            # rather than leaning on MagicMock's auto-truthiness, so
            # ``claimable=False`` can model the sibling-won-the-row case instead of
            # being unreachable. The returned lease is the worker's claim token and
            # must reach every terminal write as :lease.
            claim_row = MagicMock()
            claim_row.preview_leased_until = _CLAIM_LEASE
            result.one_or_none = MagicMock(return_value=claim_row if self._claimable else None)
        elif sql.startswith("UPDATE"):
            result.rowcount = self._terminal_rowcount
        return result

    async def commit(self) -> None:
        self.commits += 1

    def writes(self) -> list[tuple[str, dict[str, object]]]:
        return [(sql, p) for sql, p in self.statements if sql.startswith("UPDATE")]


def _fake_storage(*, size: int | None = 1_000) -> MagicMock:
    """A storage double with the full ``_FileStorage`` trio.

    ``object_size`` is the metadata HEAD the disk-budget check calls BEFORE the
    download; a double without it makes ``_build_proxy`` fail on an un-awaitable
    MagicMock rather than on the behaviour under test.
    """
    storage = MagicMock()
    storage.object_size = AsyncMock(return_value=size)
    storage.get_file = AsyncMock(return_value=True)
    storage.put_file = AsyncMock(return_value="k")
    return storage


def _factory(session: _FakeSession):
    @asynccontextmanager
    async def _f():
        yield session

    return _f


async def test_pending_row_gets_a_proxy_and_goes_ready(monkeypatch) -> None:
    session = _FakeSession()
    src = _source()
    monkeypatch.setattr(
        "src.lifecycle.preview_proxy._build_proxy",
        AsyncMock(return_value="tenants/t/projects/p/video-sources/s/preview.mp4"),
    )

    outcome = await _process_source(_factory(session), src, max_attempts=3, timeout_seconds=900)

    assert outcome == "ready"
    writes = session.writes()
    # Two writes now: the durable claim (counter + lease, committed BEFORE the
    # transcode) and the terminal flip to ready.
    assert len(writes) == 2
    claim_sql, claim_params = writes[0]
    assert "preview_leased_until = now()" in claim_sql
    assert claim_params["attempts"] == 1
    assert claim_params["lease_seconds"] == _lease_seconds(
        900, get_settings().pipeline.preview_proxy_transfer_timeout_seconds
    )
    sql, params = writes[1]
    assert "preview_status = 'ready'" in sql
    assert "preview_key = :key" in sql
    assert "preview_leased_until = NULL" in sql
    assert params["key"] == "tenants/t/projects/p/video-sources/s/preview.mp4"
    assert params["attempts"] == 1
    assert session.commits == 2


async def test_row_that_is_no_longer_pending_is_skipped(monkeypatch) -> None:
    session = _FakeSession(status="ready")
    build = AsyncMock(return_value="k")
    monkeypatch.setattr("src.lifecycle.preview_proxy._build_proxy", build)

    outcome = await _process_source(
        _factory(session), _source(), max_attempts=3, timeout_seconds=900
    )

    assert outcome == "skipped"
    build.assert_not_awaited()
    assert session.writes() == []


async def test_failing_transcode_counts_an_attempt_and_stays_pending(monkeypatch) -> None:
    session = _FakeSession()
    monkeypatch.setattr(
        "src.lifecycle.preview_proxy._build_proxy",
        AsyncMock(side_effect=PreviewProxyError("ffmpeg exited 1: boom")),
    )

    outcome = await _process_source(
        _factory(session), _source(attempts=0), max_attempts=3, timeout_seconds=900
    )

    assert outcome == "retry"
    sql, params = session.writes()[-1]
    # "preview_status = " — the claim statement READS preview_status in its WHERE
    # clause, so a bare substring check would now pass vacuously.
    assert "preview_status = " not in sql  # still pending — only the counter moved
    assert "preview_leased_until = NULL" in sql
    assert params["attempts"] == 1
    assert params["error"] == "ffmpeg exited 1: boom"


async def test_last_attempt_writes_failed_and_the_error(monkeypatch) -> None:
    session = _FakeSession()
    monkeypatch.setattr(
        "src.lifecycle.preview_proxy._build_proxy",
        AsyncMock(side_effect=PreviewProxyError("ffmpeg exited 1: boom")),
    )

    outcome = await _process_source(
        _factory(session), _source(attempts=2), max_attempts=3, timeout_seconds=900
    )

    assert outcome == "failed"
    sql, params = session.writes()[-1]
    assert "preview_status = 'failed'" in sql
    assert "preview_leased_until = NULL" in sql
    assert params["attempts"] == 3
    assert params["error"] == "ffmpeg exited 1: boom"


async def test_advisory_lock_stops_a_second_worker_claiming_the_same_row(monkeypatch) -> None:
    """Two workers, one row: exactly one builds and the other reports contention.

    The fake honours real mutual exclusion (a shared held-set keyed by source id)
    rather than being handed a scripted True/False, so this fails if the worker
    ignores the lock result.
    """
    held: set[str] = set()
    builds: list[str] = []

    class _LockingSession(_FakeSession):
        async def execute(self, stmt, params=None):  # type: ignore[no-untyped-def]
            sql = str(stmt)
            if "pg_try_advisory_xact_lock" in sql:
                key = str((params or {})["k"])
                self._locked = key not in held
                if self._locked:
                    held.add(key)
            return await super().execute(stmt, params)

    async def _build(source, timeout_seconds):  # type: ignore[no-untyped-def]
        builds.append(str(source.source_id))
        await asyncio.sleep(0)
        return "key"

    monkeypatch.setattr("src.lifecycle.preview_proxy._build_proxy", _build)
    src = _source()
    first, second = _LockingSession(), _LockingSession()

    outcomes = await asyncio.gather(
        _process_source(_factory(first), src, max_attempts=3, timeout_seconds=900),
        _process_source(_factory(second), src, max_attempts=3, timeout_seconds=900),
    )

    assert sorted(outcomes) == ["contended", "ready"]
    assert len(builds) == 1


async def test_semaphore_bounds_concurrent_transcodes(monkeypatch) -> None:
    """Four claimable rows, never more than ``_CONCURRENCY`` transcodes in flight.

    Asserts the ceiling AND that it is actually reached — a build that never
    overlapped would satisfy ``<= 2`` vacuously.
    """
    from src.lifecycle.preview_proxy import _CONCURRENCY

    live = 0
    peak = 0

    async def _build(source, timeout_seconds):  # type: ignore[no-untyped-def]
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        for _ in range(5):
            await asyncio.sleep(0)
        live -= 1
        return "key"

    monkeypatch.setattr("src.lifecycle.preview_proxy._build_proxy", _build)
    sources = [_source() for _ in range(4)]
    monkeypatch.setattr(
        "src.lifecycle.preview_proxy._scan_pending", AsyncMock(return_value=sources)
    )
    monkeypatch.setattr(
        "src.lifecycle.preview_proxy._process_source",
        _process_source,  # keep the real one; only its dependencies are faked
    )

    sessions = [_FakeSession() for _ in sources]
    factories = iter(sessions)

    @asynccontextmanager
    async def _next_session():
        yield next(factories)

    tally = await sweep_pending_previews(
        cast("Any", _next_session), batch_size=4, max_attempts=3, timeout_seconds=900
    )

    assert tally.ready == 4
    assert peak == _CONCURRENCY == 2


async def test_staging_dir_is_removed_even_when_the_transcode_raises(
    monkeypatch, tmp_path: Path
) -> None:
    from src.lifecycle import preview_proxy

    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "source").write_bytes(b"not really a movie")

    monkeypatch.setattr(preview_proxy, "_make_staging_dir", lambda: str(staging))

    storage = _fake_storage()
    monkeypatch.setattr("src.services.tts_storage._get_storage", AsyncMock(return_value=storage))
    monkeypatch.setattr(
        preview_proxy,
        "_transcode",
        AsyncMock(side_effect=PreviewProxyError("ffmpeg exited 1")),
    )

    with pytest.raises(PreviewProxyError):
        await preview_proxy._build_proxy(_source(), 900)

    assert not os.path.exists(staging)
    storage.put_file.assert_not_awaited()


async def test_preview_key_is_exactly_the_upload_prefix_plus_preview_mp4() -> None:
    """The proxy MUST land inside the same tenant/project-confined tree as the
    original — that prefix is both the tenant boundary and the key the gitops
    minio-lifecycle CronJob declares never-expired."""
    from src.api.routes.uploads_video import _source_prefix

    src = _source()

    assert _preview_key(src) == (
        _source_prefix(src.tenant_id, src.project_id, src.source_id) + "preview.mp4"
    )


async def test_sweeper_is_off_by_default() -> None:
    """The feature ships disabled — see PipelineSettings.preview_proxy_enabled."""
    from src.config import get_settings
    from src.lifecycle.preview_proxy import _start_preview_proxy_task

    assert get_settings().pipeline.preview_proxy_enabled is False
    assert _start_preview_proxy_task(MagicMock()) is None


# ---------------------------------------------------------------------------
# DURABILITY — real postgres, because a mock session cannot answer the question
# ---------------------------------------------------------------------------
#
# The defect these pin: ``attempts = source.attempts + 1`` was a PYTHON LOCAL that
# only reached the database AFTER ``_build_proxy`` returned or raised — i.e. after
# up to 900 s of ffmpeg — and the row's only claim was a transaction-scoped
# advisory lock. Kill the POD mid-transcode (OOMKill of PID 1, node drain, node
# reboot) and nothing was written at all: the lock died with the connection, the
# row stayed ``pending`` with its counter unchanged, and the resolver re-offered it
# every tick FOREVER, with ``max_attempts`` — enforced only on the caught-exception
# path — unable to engage. No exception is raised in that scenario, so no
# assertion about the exception path can catch it.
#
# ⚠️ Reading the row back through the WORKER'S OWN session would prove nothing: it
# would see its own uncommitted write and pass against the broken code. Every probe
# below therefore opens an INDEPENDENT session, which can only see committed data.


async def _seed_source(
    db_session_factory,  # type: ignore[no-untyped-def]
    *,
    attempts: int = 0,
    status: str = "pending",
    leased_until: object = None,
) -> PendingSource:
    """Commit a project + one ``video_sources`` row and describe it as the resolver would."""
    from src.database.models import VideoSource
    from src.database.repositories.project_repository import ProjectRepository

    tenant_id, subject_id = uuid4(), uuid4()
    source_id = uuid7()
    object_key = f"tenants/{tenant_id}/projects/x/video-sources/{source_id}/original.mov"
    async with db_session_factory() as session:
        project = await ProjectRepository(session).create_with_deliverables(
            tenant_id=tenant_id,
            owner_subject_id=subject_id,
            type_="generic",
            title="preview-lease-probe",
            brief={},
            brand={},
            voice={},
            theme={},
            visibility="tenant",
            parent_project_id=None,
            deliverables=[{"name": "d1", "format": "16x9"}],
        )
        session.add(
            VideoSource(
                id=source_id,
                project_id=project.id,
                filename="original.mov",
                object_key=object_key,
                content_type="video/quicktime",
                bytes=1234,
                preview_status=status,
                probe={"preview_worker": {"attempts": attempts}} if attempts else {},
                preview_leased_until=leased_until,
            )
        )
        await session.commit()
        project_id = project.id

    return PendingSource(
        source_id=source_id,
        tenant_id=tenant_id,
        project_id=project_id,
        object_key=object_key,
        attempts=attempts,
    )


async def _read_row(db_session_factory, source_id) -> Any:  # type: ignore[no-untyped-def]
    """Read one ``video_sources`` row through a connection that never shared the
    worker's transaction — the only reading that can prove durability."""
    from sqlalchemy import text as sa_text

    async with db_session_factory() as probe:
        return (
            await probe.execute(
                sa_text(
                    "SELECT preview_status, preview_leased_until, error, "
                    "CASE WHEN jsonb_typeof(probe->'preview_worker'->'attempts') = 'number' "
                    "THEN (probe->'preview_worker'->>'attempts')::int ELSE 0 END AS attempts "
                    "FROM video_sources WHERE id = :i"
                ),
                {"i": str(source_id)},
            )
        ).one()


async def test_the_attempt_and_lease_are_committed_before_the_transcode_starts(
    db_session_factory, monkeypatch
) -> None:
    """MUTANT: move the claim's ``session.commit()`` below ``_build_proxy``.

    THE test for the pod-death class. ``_build_proxy`` is replaced by a probe that
    reads the row on an INDEPENDENT connection at exactly the moment ffmpeg would
    be running; if the counter is still 0 or the lease is still NULL there, a pod
    killed at that instant loses the attempt and the row is re-picked forever.
    """
    src = await _seed_source(db_session_factory)
    seen: dict[str, Any] = {}

    async def _probe_build(source: PendingSource, timeout_seconds: int) -> str:
        row = await _read_row(db_session_factory, source.source_id)
        seen["attempts"] = row.attempts
        seen["lease"] = row.preview_leased_until
        return "tenants/t/projects/p/video-sources/s/preview.mp4"

    monkeypatch.setattr("src.lifecycle.preview_proxy._build_proxy", _probe_build)

    outcome = await _process_source(db_session_factory, src, max_attempts=3, timeout_seconds=900)

    assert outcome == "ready"
    assert seen["attempts"] == 1, (
        "the attempt counter had not reached the database when the transcode began; "
        "a pod killed here loses the attempt and the row is re-picked forever"
    )
    assert seen["lease"] is not None, "the row was not leased while its transcode ran"
    assert seen["lease"] > datetime.now(UTC), "the lease was already expired at claim time"
    # And the lease outlives the transcode budget — a lease shorter than the ffmpeg
    # timeout re-offers a row that is still actively being transcoded.
    assert seen["lease"] > datetime.now(UTC) + timedelta(seconds=900)


async def test_success_clears_the_lease_and_marks_ready(db_session_factory, monkeypatch) -> None:
    """MUTANT: drop ``preview_leased_until = NULL`` from ``_MARK_READY_SQL``."""
    src = await _seed_source(db_session_factory)
    monkeypatch.setattr(
        "src.lifecycle.preview_proxy._build_proxy", AsyncMock(return_value="k/preview.mp4")
    )

    outcome = await _process_source(db_session_factory, src, max_attempts=3, timeout_seconds=900)

    assert outcome == "ready"
    row = await _read_row(db_session_factory, src.source_id)
    assert row.preview_status == "ready"
    assert row.attempts == 1
    assert row.preview_leased_until is None, "a finished row must never read as leased"


async def test_caught_failure_below_the_cap_leaves_pending_with_the_lease_cleared(
    db_session_factory, monkeypatch
) -> None:
    """MUTANT: drop ``preview_leased_until = NULL`` from ``_MARK_RETRY_SQL``.

    A retryable row that kept its lease would sit out the rest of the transcode
    budget before any tick could look at it again.
    """
    src = await _seed_source(db_session_factory)
    monkeypatch.setattr(
        "src.lifecycle.preview_proxy._build_proxy",
        AsyncMock(side_effect=PreviewProxyError("ffmpeg exited 1: boom")),
    )

    outcome = await _process_source(db_session_factory, src, max_attempts=3, timeout_seconds=900)

    assert outcome == "retry"
    row = await _read_row(db_session_factory, src.source_id)
    assert row.preview_status == "pending"
    assert row.attempts == 1
    assert row.error == "ffmpeg exited 1: boom"
    assert row.preview_leased_until is None


async def test_a_row_at_the_cap_is_retired_at_claim_time_without_transcoding(
    db_session_factory, monkeypatch
) -> None:
    """MUTANT: delete the ``source.attempts >= max_attempts`` branch.

    This is the terminating condition for the pod-death loop: those attempts ended
    with no exception, so the exception-path cap can never see them. Without this
    branch the row transcodes again — burning a bounded slot — every single tick.
    """
    src = await _seed_source(db_session_factory, attempts=3)
    build = AsyncMock(return_value="k")
    monkeypatch.setattr("src.lifecycle.preview_proxy._build_proxy", build)

    outcome = await _process_source(db_session_factory, src, max_attempts=3, timeout_seconds=900)

    assert outcome == "failed"
    build.assert_not_awaited()
    row = await _read_row(db_session_factory, src.source_id)
    assert row.preview_status == "failed"
    assert row.attempts == 3, "a row that never transcoded must not be charged an attempt"
    assert row.preview_leased_until is None
    assert "exhausted 3 attempts" in (row.error or "")
    assert "without recording an outcome" in (row.error or "")


async def test_the_resolver_hides_a_leased_row_and_re_offers_an_expired_one(
    db_session_factory,
) -> None:
    """MUTANT: delete the ``preview_leased_until`` predicate from the resolver.

    Both halves matter. Without the hide, a row is handed to the next tick while
    its transcode is still running. Without the re-offer, a dead pod's row is
    stranded until a human notices — the lease expiry IS the crash-recovery path,
    and it is why this fix needs no reaper.
    """
    now = datetime.now(UTC)
    leased = await _seed_source(db_session_factory, leased_until=now + timedelta(minutes=30))
    expired = await _seed_source(db_session_factory, leased_until=now - timedelta(minutes=30))
    unleased = await _seed_source(db_session_factory)

    offered = {row.source_id for row in await _scan_pending(db_session_factory, 50)}

    assert leased.source_id not in offered, (
        "a row whose transcode is still running was re-offered to the next tick"
    )
    assert expired.source_id in offered, (
        "an expired lease was not re-offered; a pod that died mid-transcode strands "
        "its row forever, because nothing else reaps it"
    )
    assert unleased.source_id in offered


# ═══════════════════════════════════════════════════════════════════════════════
# OWNERSHIP FENCE on the terminal writes (2026-08-23)
#
# The claim re-asserted ``pending AND not-leased``; the three terminal writes
# predicated on ``id`` alone. A worker whose lease expired mid-transcode could
# therefore still land its verdict on a row another worker had since claimed and
# finished — overwriting 'ready' with 'failed' while ``preview_key`` still pointed
# at a working proxy, and (via the retry write) CLEARING the winner's live lease
# and inviting a third concurrent transcode of the same source.
# ═══════════════════════════════════════════════════════════════════════════════


async def test_every_terminal_write_carries_the_lease_fence(monkeypatch) -> None:
    """READY, RETRY and FAILED must all predicate on the lease this worker holds.

    Asserts the SQL and the bound value together: a fence whose ``:lease`` param
    never reaches the statement is not a fence.
    """
    seen: list[tuple[str, dict[str, object]]] = []

    async def _run(build: AsyncMock, *, attempts: int, max_attempts: int) -> None:
        session = _FakeSession()
        monkeypatch.setattr("src.lifecycle.preview_proxy._build_proxy", build)
        await _process_source(
            _factory(session),
            _source(attempts=attempts),
            max_attempts=max_attempts,
            timeout_seconds=900,
        )
        seen.append(session.writes()[-1])

    # ready
    await _run(AsyncMock(return_value="k"), attempts=0, max_attempts=3)
    # retry (fails, attempts left)
    await _run(AsyncMock(side_effect=PreviewProxyError("boom")), attempts=0, max_attempts=3)
    # failed (fails, last attempt)
    await _run(AsyncMock(side_effect=PreviewProxyError("boom")), attempts=2, max_attempts=3)

    assert len(seen) == 3
    for sql, params in seen:
        assert "preview_leased_until = :lease" in sql, (
            f"terminal write is not fenced on the claim's lease: {sql}"
        )
        assert params["lease"] == _CLAIM_LEASE


async def test_a_stale_worker_cannot_overwrite_the_winners_ready(monkeypatch) -> None:
    """The exact production scenario: A's lease expired, B finished 'ready', A's
    ffmpeg then failed. A's 'failed' write must affect 0 rows and be REPORTED,
    never silently swallowed and never applied."""
    session = _FakeSession(terminal_rowcount=0)
    monkeypatch.setattr(
        "src.lifecycle.preview_proxy._build_proxy",
        AsyncMock(side_effect=PreviewProxyError("ffmpeg exited 1")),
    )

    outcome = await _process_source(
        _factory(session), _source(attempts=0), max_attempts=3, timeout_seconds=900
    )

    assert outcome == "lost_lease", "a fenced-off verdict must not report as retry/failed"


async def test_a_stale_worker_that_succeeds_also_loses(monkeypatch) -> None:
    """Symmetry: even a successful transcode from an expired claim must not write.
    Its proxy is byte-identical (``_preview_key`` is deterministic), so discarding
    the bookkeeping loses nothing but the wasted CPU."""
    session = _FakeSession(terminal_rowcount=0)
    monkeypatch.setattr("src.lifecycle.preview_proxy._build_proxy", AsyncMock(return_value="k"))

    outcome = await _process_source(
        _factory(session), _source(), max_attempts=3, timeout_seconds=900
    )

    assert outcome == "lost_lease"


async def test_lost_lease_is_tallied_separately_from_skipped(monkeypatch) -> None:
    """``skipped`` means "never started the work"; ``lost_lease`` means "did the
    whole transcode and threw the verdict away". Folding them together would hide
    the only signal that the lease is under-sized."""
    session = _FakeSession(terminal_rowcount=0)
    monkeypatch.setattr("src.lifecycle.preview_proxy._build_proxy", AsyncMock(return_value="k"))
    monkeypatch.setattr(
        "src.lifecycle.preview_proxy._scan_pending", AsyncMock(return_value=[_source()])
    )

    tally = await sweep_pending_previews(
        _factory(session), batch_size=10, max_attempts=3, timeout_seconds=900
    )

    assert tally.lost_lease == 1
    assert tally.skipped == 0
    assert tally.ready == 0


async def test_exhausted_at_claim_retirement_is_fenced_on_pending_and_lease(
    monkeypatch,
) -> None:
    """The claim-time retirement runs before any lease of ours exists, so it uses
    the CLAIM's predicate instead — it must never retire a row a live worker is
    mid-transcode on."""
    session = _FakeSession()
    build = AsyncMock()
    monkeypatch.setattr("src.lifecycle.preview_proxy._build_proxy", build)

    outcome = await _process_source(
        _factory(session), _source(attempts=3), max_attempts=3, timeout_seconds=900
    )

    assert outcome == "failed"
    build.assert_not_awaited()
    sql, _ = session.writes()[-1]
    assert "preview_status = 'pending'" in sql, "retirement is not fenced on pending"
    assert "preview_leased_until IS NULL OR preview_leased_until < now()" in sql, (
        "retirement is not fenced on the lease being free"
    )


async def test_a_fenced_off_retirement_does_not_report_failed(monkeypatch) -> None:
    """The fence exists to be LOST sometimes, and a lost retirement wrote nothing.

    Reporting ``failed`` (and logging "giving up on source=…") when the UPDATE
    matched 0 rows asserts a retirement that never happened and books a
    ``tally.failed`` for a row somebody else owns. Every other terminal write in
    this module already checks its rowcount; this one did not.
    """
    session = _FakeSession(terminal_rowcount=0)
    build = AsyncMock()
    monkeypatch.setattr("src.lifecycle.preview_proxy._build_proxy", build)

    outcome = await _process_source(
        _factory(session), _source(attempts=3), max_attempts=3, timeout_seconds=900
    )

    build.assert_not_awaited()
    assert outcome == "skipped", (
        "a retirement the fence rejected reported as 'failed' — the sweep claims to "
        "have retired a row it never wrote to"
    )


async def test_a_fenced_off_retirement_is_not_tallied_as_failed(monkeypatch) -> None:
    """The counter half. ``skipped`` means 'never started the work', which is
    exactly what a lost claim-time race is; ``failed`` would inflate the one number
    an operator reads as 'previews that will never play'."""
    session = _FakeSession(terminal_rowcount=0)
    monkeypatch.setattr("src.lifecycle.preview_proxy._build_proxy", AsyncMock())
    monkeypatch.setattr(
        "src.lifecycle.preview_proxy._scan_pending",
        AsyncMock(return_value=[_source(attempts=3)]),
    )

    tally = await sweep_pending_previews(
        _factory(session), batch_size=10, max_attempts=3, timeout_seconds=900
    )

    assert tally.failed == 0
    assert tally.skipped == 1
    assert tally.lost_lease == 0, "no lease was held and no transcode was burned"


# ═══════════════════════════════════════════════════════════════════════════════
# DISK BUDGET + TRANSFER BOUNDS (2026-08-23)
# ═══════════════════════════════════════════════════════════════════════════════


async def test_oversized_source_is_refused_before_a_single_byte_moves(monkeypatch) -> None:
    """A source larger than the disk budget can NEVER succeed — it would ENOSPC the
    pod's /tmp, take the other concurrent job down with it, and burn every attempt.
    It must be refused on the metadata HEAD, with a reason an operator can act on,
    and the download must not be started."""
    from src.lifecycle import preview_proxy

    budget = get_settings().pipeline.preview_proxy_max_source_bytes
    storage = _fake_storage(size=budget + 1)
    monkeypatch.setattr("src.services.tts_storage._get_storage", AsyncMock(return_value=storage))
    transcode = AsyncMock()
    monkeypatch.setattr(preview_proxy, "_transcode", transcode)

    with pytest.raises(PreviewProxyError) as excinfo:
        await preview_proxy._build_proxy(_source(), 900)

    storage.get_file.assert_not_awaited(), "the oversized object was downloaded anyway"
    transcode.assert_not_awaited()
    assert str(budget) in str(excinfo.value)
    assert "preview_proxy_max_source_bytes" in str(excinfo.value)


async def test_a_source_exactly_at_the_budget_is_still_built(monkeypatch) -> None:
    """Boundary: the budget is inclusive. A strict `>=` here would refuse a clip
    that fits, which is the same class of bug in the other direction."""
    from src.lifecycle import preview_proxy

    budget = get_settings().pipeline.preview_proxy_max_source_bytes
    storage = _fake_storage(size=budget)
    monkeypatch.setattr("src.services.tts_storage._get_storage", AsyncMock(return_value=storage))
    monkeypatch.setattr(preview_proxy, "_transcode", AsyncMock())

    key = await preview_proxy._build_proxy(_source(), 900)

    assert key.endswith("preview.mp4")
    storage.get_file.assert_awaited()


async def test_a_missing_object_is_named_from_the_metadata_head(monkeypatch) -> None:
    """``object_size`` returning None is absence — it must raise the operator-facing
    'missing from storage', not fall through into a comparison against None."""
    from src.lifecycle import preview_proxy

    storage = _fake_storage(size=None)
    monkeypatch.setattr("src.services.tts_storage._get_storage", AsyncMock(return_value=storage))

    with pytest.raises(PreviewProxyError, match="missing from storage"):
        await preview_proxy._build_proxy(_source(), 900)

    storage.get_file.assert_not_awaited()


async def test_a_stalled_download_cannot_outlive_its_bound(monkeypatch) -> None:
    """``fget_object`` has no timeout of its own, so a stalled transfer used to hang
    the worker indefinitely — past the row's lease, which is what makes a second
    worker re-claim it. The bound turns that into a normal counted attempt."""
    from src.lifecycle import preview_proxy

    storage = _fake_storage()

    async def _never_returns(key: str, local_path: str) -> bool:
        await asyncio.sleep(3600)
        return True

    storage.get_file = AsyncMock(side_effect=_never_returns)
    monkeypatch.setattr("src.services.tts_storage._get_storage", AsyncMock(return_value=storage))
    monkeypatch.setattr(
        get_settings().pipeline, "preview_proxy_transfer_timeout_seconds", 1, raising=True
    )

    # ⚠️ The outer guard must be MUCH larger than the configured transfer bound, and
    # the ELAPSED time is the real assertion. An earlier version wrapped the call in
    # `wait_for(..., timeout=10)` and asserted only `pytest.raises(TimeoutError)` —
    # which passed with the production timeout DELETED, because the test's own guard
    # raises the same TimeoutError. Measuring which bound fired is what makes this
    # test about the code instead of about itself.
    loop = asyncio.get_running_loop()
    started = loop.time()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(preview_proxy._build_proxy(_source(), 900), timeout=30)
    elapsed = loop.time() - started

    assert elapsed < 10, (
        f"the transfer completed in {elapsed:.1f}s — only the test's own 30s guard "
        "fired, so the production transfer bound is missing"
    )


async def test_a_stalled_upload_cannot_outlive_its_bound(monkeypatch) -> None:
    """Same bound on the OTHER side of ffmpeg. The upload is the transfer that was
    hanging AFTER the expensive work was already done."""
    from src.lifecycle import preview_proxy

    storage = _fake_storage()

    async def _never_returns(key: str, local_path: str, content_type: str = "") -> str:
        await asyncio.sleep(3600)
        return "k"

    storage.put_file = AsyncMock(side_effect=_never_returns)
    monkeypatch.setattr("src.services.tts_storage._get_storage", AsyncMock(return_value=storage))
    monkeypatch.setattr(preview_proxy, "_transcode", AsyncMock())
    monkeypatch.setattr(
        get_settings().pipeline, "preview_proxy_transfer_timeout_seconds", 1, raising=True
    )

    # ⚠️ The outer guard must be MUCH larger than the configured transfer bound, and
    # the ELAPSED time is the real assertion. An earlier version wrapped the call in
    # `wait_for(..., timeout=10)` and asserted only `pytest.raises(TimeoutError)` —
    # which passed with the production timeout DELETED, because the test's own guard
    # raises the same TimeoutError. Measuring which bound fired is what makes this
    # test about the code instead of about itself.
    loop = asyncio.get_running_loop()
    started = loop.time()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(preview_proxy._build_proxy(_source(), 900), timeout=30)
    elapsed = loop.time() - started

    assert elapsed < 10, (
        f"the transfer completed in {elapsed:.1f}s — only the test's own 30s guard "
        "fired, so the production transfer bound is missing"
    )


async def test_the_lease_covers_both_bounded_transfers_plus_ffmpeg(monkeypatch) -> None:
    """The lease must be strictly longer than this worker's worst-case runtime, or
    the ownership fence fires on ordinary slow-but-healthy work instead of on
    genuine lease loss. A flat margin smaller than the transfer budget did exactly
    that."""
    transfer = get_settings().pipeline.preview_proxy_transfer_timeout_seconds

    assert _lease_seconds(900, transfer) >= 900 + 2 * transfer


# ═══════════════════════════════════════════════════════════════════════════════
# `video_sources.error` IS A USER-VISIBLE STRING (2026-08-23)
#
# footage_probe interpolates row.error verbatim into a stage note and the frontend
# renders that note verbatim in the review gate. The generic branch of `reason` was
# `f"{type(exc).__name__}: {exc}"` — unbounded and unscrubbed — and `_build_proxy`
# talks to MinIO, so a transport exception routinely carries a presigned URL whose
# SigV4 query string IS a live credential.
# ═══════════════════════════════════════════════════════════════════════════════

_PRESIGNED = (
    "https://minio.test:9000/w2v/tenants/t/original.mov"
    "?X-Amz-Algorithm=AWS4-HMAC-SHA256"
    "&X-Amz-Credential=AKIAIOSFODNN7EXAMPLE%2F20260823%2Fus-east-1%2Fs3%2Faws4_request"  # pragma: allowlist secret
    "&X-Amz-Signature=deadbeefcafe1234567890abcdef"  # pragma: allowlist secret
)


async def _reason_written_for(monkeypatch, exc: Exception) -> str:
    """Drive a real failing attempt and return the text bound to the error column."""
    session = _FakeSession()
    monkeypatch.setattr("src.lifecycle.preview_proxy._build_proxy", AsyncMock(side_effect=exc))
    await _process_source(_factory(session), _source(), max_attempts=3, timeout_seconds=900)
    _, params = session.writes()[-1]
    return str(params["error"])


async def test_a_presigned_url_credential_never_reaches_the_error_column(monkeypatch) -> None:
    """SigV4 puts the access key id and the signature in the query string. Neither
    may survive into a column the review gate renders into a browser."""
    written = await _reason_written_for(
        monkeypatch, RuntimeError(f"S3 operation failed on {_PRESIGNED}")
    )

    assert "AKIAIOSFODNN7EXAMPLE" not in written, (  # pragma: allowlist secret
        "presigned ACCESS KEY ID reached the UI string"
    )
    assert "deadbeefcafe1234567890abcdef" not in written, (  # pragma: allowlist secret
        "presigned SIGNATURE reached the UI"
    )
    assert "[redacted]" in written
    # Still diagnosable: the type and the endpoint survive.
    assert "RuntimeError" in written
    assert "minio.test" in written


async def test_a_dsn_password_never_reaches_the_error_column(monkeypatch) -> None:
    """The other credential shape this code path can raise — a driver error echoing
    its DSN. Covered by the reused `safe_exc_message`, asserted here because THIS
    call site is the one that reaches a browser."""
    written = await _reason_written_for(
        monkeypatch,
        RuntimeError(
            "connect failed postgresql://w2v:hunter2@db:5432/w2v"  # pragma: allowlist secret
        ),
    )

    assert "hunter2" not in written, (  # pragma: allowlist secret
        "DSN password reached the UI string"
    )
    assert "[redacted]" in written


async def test_a_huge_exception_message_is_bounded(monkeypatch) -> None:
    """An unbounded repr becomes an unbounded row AND an unbounded DOM string."""
    written = await _reason_written_for(monkeypatch, RuntimeError("A" * 20_000))

    assert len(written) < 600, f"error column text was {len(written)} chars — unbounded"


async def test_an_empty_stringing_exception_still_says_something(monkeypatch) -> None:
    """`str(TimeoutError())` is EMPTY. Scrubbing must not turn a real failure into a
    blank cell in the review gate — the reused helper falls back to `repr`."""
    written = await _reason_written_for(monkeypatch, TimeoutError())

    # ⚠️ Asserting only `"TimeoutError" in written` is VACUOUS: the type name is
    # prepended unconditionally, so it passes with the repr fallback deleted. The
    # real assertion is that something survives AFTER the type prefix.
    _, _, body = written.partition(": ")
    assert body.strip(), (
        f"the error column would render as a bare type prefix ({written!r}) — "
        "str(TimeoutError()) is empty, so the repr fallback is load-bearing"
    )
    assert "TimeoutError" in body


async def test_the_ffmpeg_operator_message_is_passed_through_verbatim(monkeypatch) -> None:
    """The considered decision that stays: argv and paths are excluded at the raise
    site and the stderr tail is capped there, so a PreviewProxyError is passed
    through VERBATIM. An operator must still get the actionable ffmpeg diagnosis."""
    written = await _reason_written_for(
        monkeypatch, PreviewProxyError("ffmpeg exited 1: moov atom not found")
    )

    assert written == "ffmpeg exited 1: moov atom not found", (
        "the ffmpeg operator message was mangled by the scrubber"
    )
