"""Sub-DAG primitives unit tests — pubsub + DB-poll behavior with mocked deps."""

from __future__ import annotations

import asyncio
import contextlib
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from redis._parsers.socket import SERVER_CLOSED_CONNECTION_ERROR
from redis.exceptions import AuthenticationError, ResponseError
from redis.exceptions import ConnectionError as RedisConnectionError

from src.services.runs import sub_dag
from src.services.runs.sub_dag import ChildRunFailed, await_runs


class _FakePubSub:
    """Minimal pubsub stub: returns queued messages then None."""

    def __init__(self, messages: list[dict]) -> None:
        self._messages = list(messages)
        self.subscribed: list[str] = []
        self.unsubscribed: list[str] = []
        self.closed = False

    async def subscribe(self, *channels: str) -> None:
        self.subscribed.extend(channels)

    async def unsubscribe(self, *channels: str) -> None:
        self.unsubscribed.extend(channels)

    async def aclose(self) -> None:
        self.closed = True

    async def get_message(self, *, ignore_subscribe_messages: bool, timeout: float) -> dict | None:
        if self._messages:
            return self._messages.pop(0)
        await asyncio.sleep(0)
        return None


def _make_redis(messages: list[dict]) -> SimpleNamespace:
    pubsub = _FakePubSub(messages)
    return SimpleNamespace(pubsub=lambda: pubsub, _pubsub=pubsub)


@contextlib.contextmanager
def _patch_get(get: AsyncMock):
    """Patch the per-run ``RunRepository.get`` AND the batched ``statuses`` the
    poll loop uses, both driven by the same per-id ``get`` fake. ``statuses``
    omits ids whose ``get`` returns ``None`` (the unreadable/missing row)."""

    async def _statuses(run_ids):
        out = {}
        for rid in run_ids:
            run = await get(rid)
            if run is not None:
                out[rid] = str(run.status)
        return out

    with (
        patch.object(sub_dag.RunRepository, "get", new=get),
        patch.object(sub_dag.RunRepository, "statuses", new=AsyncMock(side_effect=_statuses)),
    ):
        yield get


def _session_factory() -> contextlib.AbstractAsyncContextManager:
    @contextlib.asynccontextmanager
    async def _cm():
        yield object()  # session not actually used; RunRepository is patched

    return _cm()


@pytest.mark.asyncio
async def test_await_runs_completes_on_pubsub_event() -> None:
    rid = uuid4()
    redis = _make_redis([{"data": json.dumps({"event": "run_completed", "run_id": str(rid)})}])
    fake_run = SimpleNamespace(status="pending", error=None)
    with _patch_get(AsyncMock(return_value=fake_run)):
        result = await await_runs(
            run_ids=[rid],
            session_factory=_session_factory,
            redis=redis,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            poll_interval_s=0.05,
            timeout_s=2,
        )
    assert result == {rid: "completed"}
    assert redis._pubsub.subscribed == [f"runs:{rid}"]
    assert redis._pubsub.unsubscribed == [f"runs:{rid}"]
    assert redis._pubsub.closed is True


@pytest.mark.asyncio
async def test_await_runs_picks_up_pre_terminated_run_via_initial_db_poll() -> None:
    rid = uuid4()
    redis = _make_redis([])  # no pub/sub messages
    fake_run = SimpleNamespace(status="completed", error=None)
    with _patch_get(AsyncMock(return_value=fake_run)):
        result = await await_runs(
            run_ids=[rid],
            session_factory=_session_factory,
            redis=redis,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            poll_interval_s=0.05,
            timeout_s=2,
        )
    assert result == {rid: "completed"}


@pytest.mark.asyncio
async def test_await_runs_raises_on_failed_child_when_fail_on_any() -> None:
    rid = uuid4()
    redis = _make_redis(
        [
            {
                "data": json.dumps(
                    {
                        "event": "run_failed",
                        "run_id": str(rid),
                        "error": "boom",
                    }
                )
            }
        ]
    )
    fake_run = SimpleNamespace(status="failed", error="boom")
    with _patch_get(AsyncMock(return_value=fake_run)):
        with pytest.raises(ChildRunFailed) as exc_info:
            await await_runs(
                run_ids=[rid],
                session_factory=_session_factory,
                redis=redis,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
                fail_on_any=True,
                poll_interval_s=0.05,
                timeout_s=2,
            )
    assert exc_info.value.run_id == rid
    assert exc_info.value.status == "failed"
    assert exc_info.value.error == "boom"


@pytest.mark.asyncio
async def test_await_runs_returns_failure_status_when_fail_on_any_false() -> None:
    rid = uuid4()
    redis = _make_redis([{"data": json.dumps({"event": "run_failed", "run_id": str(rid)})}])
    fake_run = SimpleNamespace(status="failed", error=None)
    with _patch_get(AsyncMock(return_value=fake_run)):
        result = await await_runs(
            run_ids=[rid],
            session_factory=_session_factory,
            redis=redis,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            fail_on_any=False,
            poll_interval_s=0.05,
            timeout_s=2,
        )
    assert result == {rid: "failed"}


@pytest.mark.asyncio
async def test_await_runs_times_out_when_no_terminal_signal() -> None:
    rid = uuid4()
    redis = _make_redis([])
    fake_run = SimpleNamespace(status="running", error=None)
    with _patch_get(AsyncMock(return_value=fake_run)):
        with pytest.raises(TimeoutError):
            await await_runs(
                run_ids=[rid],
                session_factory=_session_factory,
                redis=redis,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
                poll_interval_s=0.05,
                timeout_s=0.2,
            )


# --- child-wait bound derived from the parent run's remaining budget ---------


class _FakeRunRepo:
    """Recording stand-in for RunRepository, keyed by run_id.

    A missing key returns ``None`` — i.e. the row a failed dispatch never wrote.
    """

    rows: ClassVar[dict] = {}

    def __init__(self, session: object) -> None:
        self._session = session

    async def get(self, run_id: object) -> object:
        return _FakeRunRepo.rows.get(run_id)

    async def statuses(self, run_ids: list) -> dict:
        return {
            rid: str(row.status)
            for rid in run_ids
            if (row := _FakeRunRepo.rows.get(rid)) is not None
        }


@contextlib.contextmanager
def _patched_runs(rows: dict):
    prev = _FakeRunRepo.rows
    _FakeRunRepo.rows = rows
    with patch.object(sub_dag, "RunRepository", _FakeRunRepo):
        try:
            yield
        finally:
            _FakeRunRepo.rows = prev


def _settings(run_deadline_seconds: int) -> SimpleNamespace:
    return SimpleNamespace(pipeline=SimpleNamespace(run_deadline_seconds=run_deadline_seconds))


def _run_row(status: str, *, started_ago_s: float | None = None) -> SimpleNamespace:
    started_at = (
        None if started_ago_s is None else datetime.now(UTC) - timedelta(seconds=started_ago_s)
    )
    return SimpleNamespace(status=status, error=None, started_at=started_at)


@pytest.mark.asyncio
async def test_child_wait_is_the_parents_remaining_budget_minus_the_margin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = uuid4()
    monkeypatch.setattr(sub_dag, "get_settings", lambda: _settings(14400))
    with _patched_runs({parent: _run_row("running", started_ago_s=12000)}):
        bound, source = await sub_dag._derive_child_wait_s(parent, _session_factory, None)
    # 14400 - 12000 elapsed = 2400 remaining, minus the 120s safety margin.
    assert 2270.0 < bound < 2290.0
    assert bound < 7200.0  # the constant this replaced
    assert "remaining=" in source


@pytest.mark.asyncio
async def test_child_wait_never_exceeds_the_fallback_even_on_a_fresh_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = uuid4()
    monkeypatch.setattr(sub_dag, "get_settings", lambda: _settings(14400))
    with _patched_runs({parent: _run_row("running", started_ago_s=1.0)}):
        bound, _ = await sub_dag._derive_child_wait_s(parent, _session_factory, None)
    assert bound == sub_dag._FALLBACK_CHILD_WAIT_S
    assert sub_dag._FALLBACK_CHILD_WAIT_S < 5400.0  # below the E2E scenario deadline
    assert sub_dag._FALLBACK_CHILD_WAIT_S < 14400.0  # below run_deadline_seconds


@pytest.mark.asyncio
async def test_fallback_when_the_parent_is_unknown_or_the_deadline_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[float, str]] = []
    monkeypatch.setattr(sub_dag, "get_settings", lambda: _settings(14400))
    with _patched_runs({}):
        observed.append(await sub_dag._derive_child_wait_s(None, _session_factory, None))
        missing = uuid4()
        observed.append(await sub_dag._derive_child_wait_s(missing, _session_factory, None))
    monkeypatch.setattr(sub_dag, "get_settings", lambda: _settings(0))
    with _patched_runs({}):
        observed.append(await sub_dag._derive_child_wait_s(uuid4(), _session_factory, None))
    assert [b for b, _ in observed] == [sub_dag._FALLBACK_CHILD_WAIT_S] * 3
    assert [s for _, s in observed] == [
        "fallback:no-parent-run-id",
        "fallback:no-started-at",
        "fallback:run-deadline-disabled",
    ]


@pytest.mark.asyncio
async def test_await_runs_default_bound_fires_from_the_parent_budget_not_7200(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No explicit timeout_s: the wait must end on the PARENT's remaining budget.

    With the old fixed 7200s default this call parks for two hours, so the outer
    ``wait_for`` is what ends it — and its TimeoutError carries no message.
    """
    parent, child = uuid4(), uuid4()
    monkeypatch.setattr(sub_dag, "get_settings", lambda: _settings(600))
    monkeypatch.setattr(sub_dag, "_MIN_CHILD_WAIT_S", 0.05)
    redis = _make_redis([])
    with _patched_runs(
        {
            parent: _run_row("running", started_ago_s=480),  # 120s remaining == the margin
            child: _run_row("running", started_ago_s=10),
        }
    ):
        with pytest.raises(TimeoutError) as exc_info:
            await asyncio.wait_for(
                await_runs(
                    run_ids=[child],
                    session_factory=_session_factory,
                    redis=redis,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
                    parent_run_id=parent,
                    poll_interval_s=0.01,
                ),
                timeout=5.0,
            )
    message = str(exc_info.value)
    assert "sub_dag await_runs timed out" in message
    assert "bound=remaining=120s of 600s" in message
    assert "7200" not in message


@pytest.mark.asyncio
async def test_timeout_message_separates_never_started_from_in_progress() -> None:
    never_started, in_progress = uuid4(), uuid4()
    redis = _make_redis([])
    with _patched_runs({in_progress: _run_row("running", started_ago_s=5)}):
        with pytest.raises(TimeoutError) as exc_info:
            await asyncio.wait_for(
                await_runs(
                    run_ids=[never_started, in_progress],
                    session_factory=_session_factory,
                    redis=redis,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
                    poll_interval_s=0.01,
                    timeout_s=0.2,
                ),
                timeout=5.0,
            )
    message = str(exc_info.value)
    assert f"'{never_started}': 'missing'" in message
    assert f"'{in_progress}': 'running'" in message
    never_part, _, progress_part = message.partition("in_progress=")
    assert str(never_started) in never_part
    assert str(never_started) not in progress_part
    assert str(in_progress) in progress_part
    assert "bound=explicit" in message


# ─── redis pubsub reconnect: surviving a redis restart mid-child-wait ────────
#
# E2E attempt #11 (run 33864937267) scored 38/39; the single failure was
# `course_v1`, whose `course_compile_lessons` stage output read
# `failed, ConnectionError: Connection closed by server.` — redis-py's
# `SERVER_CLOSED_CONNECTION_ERROR`, raised out of an UNGUARDED
# `pubsub.get_message()` when the single-replica `Recreate`-strategy redis pod
# rolled while the course was waiting on its lesson children. Every child was
# still rendering; only the observer died, and it took the course down with it.
#
# The outage in these tests therefore starts AFTER the wait is underway. An
# outage present at t0 would only exercise the initial-connect ladder and would
# never drive the `get_message` path that actually failed in production —
# green, and blind to the bug.

_OUTAGE_START = 1005.0


class _FakeClock:
    """A monotonic clock driven by the patched `asyncio.sleep` and by idle polls.

    Real time makes these assertions untestable — a 300-second budget cannot be
    waited out in a unit test — and an attempt-COUNT assertion alone would NOT
    pin the property that matters: the budget is denominated in SECONDS,
    because what it must outlast is a pod rollout measured in seconds.
    """

    def __init__(self) -> None:
        self.now = 1000.0
        #: Every delay the reconnect ladder actually slept. Asserting on THIS
        #: rather than on elapsed time is what separates "the ladder was never
        #: entered" from "the loop's own idle polls moved the clock" — the two
        #: are indistinguishable by wall time and only one of them is the
        #: property under test.
        self.sleeps: list[float] = []

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _FakeClock:
    """Put sub_dag's clock and every sleep it performs onto fake time."""
    c = _FakeClock()
    monkeypatch.setattr(sub_dag, "_now", lambda: c.now)
    real_sleep = asyncio.sleep

    async def _fake_sleep(delay: float, *args: object, **kwargs: object) -> None:
        if delay:
            # Zero-delay yields (the fakes' own `asyncio.sleep(0)`) are
            # cooperative scheduling, not budget. Only real waits are budget.
            c.sleeps.append(delay)
        c.advance(delay)
        await real_sleep(0)  # still yield, so the loop stays cooperative

    monkeypatch.setattr(sub_dag.asyncio, "sleep", _fake_sleep)
    return c


class _FlakyPubSub:
    """A pubsub that is DOWN for a window of fake seconds.

    The outage is expressed in clock time, not in a call count, so a test can
    make it shorter or longer than the reconnect budget and mean it.
    """

    def __init__(self, clock: _FakeClock, *, outage: tuple[float, float], error: Exception) -> None:
        self._clock = clock
        self._outage = outage
        self._error = error
        self.subscribed: list[str] = []
        self.unsubscribed: list[str] = []
        self.closed = False

    def _down(self) -> bool:
        return self._outage[0] <= self._clock.now < self._outage[1]

    async def subscribe(self, *channels: str) -> None:
        if self._down():
            raise self._error
        self.subscribed.extend(channels)

    async def unsubscribe(self, *channels: str) -> None:
        self.unsubscribed.extend(channels)

    async def aclose(self) -> None:
        self.closed = True

    async def get_message(self, *, ignore_subscribe_messages: bool, timeout: float) -> dict | None:
        if self._down():
            raise self._error
        self._clock.advance(timeout)  # an idle poll costs its timeout
        await asyncio.sleep(0)
        return None


class _FlakyRedis:
    """Hands out a FRESH `_FlakyPubSub` per `.pubsub()` — like the real client,
    whose dead connection is evicted and replaced on reconnect."""

    def __init__(self, clock: _FakeClock, *, outage: tuple[float, float], error: Exception) -> None:
        self._clock = clock
        self._outage = outage
        self._error = error
        self.pubsubs: list[_FlakyPubSub] = []

    def pubsub(self) -> _FlakyPubSub:
        ps = _FlakyPubSub(self._clock, outage=self._outage, error=self._error)
        self.pubsubs.append(ps)
        return ps


def _server_closed() -> Exception:
    """The EXACT error E2E #11 died on."""
    return RedisConnectionError(SERVER_CLOSED_CONNECTION_ERROR)


def _run_completing_at(clock: _FakeClock, when: float) -> AsyncMock:
    """A `RunRepository.get` whose child is `running` until `when`, then `completed`.

    The child terminalises on its OWN schedule, indifferent to the observer.
    Setting `when` inside the outage window is what proves the post-reconnect DB
    re-read closes the gap: the `run_completed` publish happened while nobody
    was subscribed, so an event-only recovery would wait forever.
    """

    async def _get(_rid: object) -> SimpleNamespace:
        status = "completed" if clock.now >= when else "running"
        return SimpleNamespace(status=status, error=None)

    return AsyncMock(side_effect=_get)


@pytest.mark.asyncio
async def test_reconnects_when_redis_restarts_shorter_than_the_budget(
    clock: _FakeClock,
) -> None:
    # A 75-second redis Recreate roll — the measured worst case — beginning
    # after the wait is already underway. 75s sits well inside the 300s ladder,
    # so the stage must SURVIVE it and complete on the child's real status.
    rid = uuid4()
    outage = (_OUTAGE_START, _OUTAGE_START + 75.0)
    redis = _FlakyRedis(clock, outage=outage, error=_server_closed())
    # The child completes DURING the outage: its `run_completed` was published
    # to nobody, so only the post-reconnect DB re-read can ever see it.
    completes_at = _OUTAGE_START + 40.0
    started = clock.now
    with _patch_get(_run_completing_at(clock, completes_at)):
        result = await await_runs(
            run_ids=[rid],
            session_factory=_session_factory,
            redis=redis,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            poll_interval_s=2.0,
            timeout_s=4800.0,
        )
    assert result == {rid: "completed"}
    # The FIRST pubsub connected fine — i.e. the outage really did start
    # mid-wait, on `get_message`, which is the production failure path.
    assert redis.pubsubs[0].subscribed == [f"runs:{rid}"]
    # It really did reconnect: a second pubsub was created and re-subscribed.
    assert len(redis.pubsubs) >= 2
    assert redis.pubsubs[-1].subscribed == [f"runs:{rid}"]
    # The dead connection was released rather than leaked.
    assert redis.pubsubs[0].closed is True
    # And it spent real (fake) seconds waiting the outage out — a non-trivial
    # failure window, not an instant re-call that would pass on a no-op.
    assert clock.now >= outage[1]
    assert clock.now - started >= 75.0
    # The ladder was genuinely entered and climbed — the exact prefix of the
    # real schedule needed to cover a 75s roll, not a lucky single re-call.
    assert clock.sleeps == list(sub_dag._PUBSUB_RECONNECT_BACKOFF[: len(clock.sleeps)])
    assert sum(clock.sleeps) >= 75.0 - 2.0


@pytest.mark.asyncio
async def test_reconnect_budget_is_bounded_and_still_fails_when_redis_stays_down(
    clock: _FakeClock,
) -> None:
    # Redis never comes back. The stage must still FAIL — loudly, naming the
    # connection error — after a BOUNDED wait, not hang forever.
    rid = uuid4()
    redis = _FlakyRedis(clock, outage=(_OUTAGE_START, 1e18), error=_server_closed())
    started = clock.now
    with _patch_get(_run_completing_at(clock, 1e18)):
        with pytest.raises(RedisConnectionError) as exc_info:
            await await_runs(
                run_ids=[rid],
                session_factory=_session_factory,
                redis=redis,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
                poll_interval_s=2.0,
                timeout_s=4800.0,  # far larger than the ladder: the LADDER binds
            )
    assert str(exc_info.value) == SERVER_CLOSED_CONNECTION_ERROR
    # One live pubsub, then 24 connect attempts — 23 delays summing to 300s.
    assert len(sub_dag._PUBSUB_RECONNECT_BACKOFF) + 1 == 24
    assert len(redis.pubsubs) == 25
    # Bounded in SECONDS: the 300s ladder plus only the few idle polls before
    # the outage began, and nowhere near the 4800s child-wait — so the failure
    # is reported as a redis outage, not laundered into a TimeoutError.
    assert clock.sleeps == list(sub_dag._PUBSUB_RECONNECT_BACKOFF)
    assert sum(clock.sleeps) == pytest.approx(300.0)
    elapsed = clock.now - started
    assert 300.0 < elapsed < 320.0


@pytest.mark.asyncio
async def test_reconnect_budget_never_outlives_the_child_wait_deadline(
    clock: _FakeClock,
) -> None:
    # The child-wait deadline stays AUTHORITATIVE. With only ~25s of budget left
    # when redis drops, the ladder must not sleep past it and launder a clear
    # connection failure into an opaque timeout — it gives up early, still
    # naming redis.
    rid = uuid4()
    redis = _FlakyRedis(clock, outage=(_OUTAGE_START, 1e18), error=_server_closed())
    started = clock.now
    with _patch_get(_run_completing_at(clock, 1e18)):
        with pytest.raises(RedisConnectionError):
            await await_runs(
                run_ids=[rid],
                session_factory=_session_factory,
                redis=redis,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
                poll_interval_s=2.0,
                timeout_s=30.0,
            )
    # Stopped INSIDE the 30s deadline, having spent a fraction of the 300s ladder.
    assert clock.now - started < 30.0
    assert len(redis.pubsubs) < 25
    # It climbed a STRICT PREFIX of the ladder and stopped because the deadline
    # said so, not because the ladder ran out.
    assert 0 < len(clock.sleeps) < len(sub_dag._PUBSUB_RECONNECT_BACKOFF)
    assert clock.sleeps == list(sub_dag._PUBSUB_RECONNECT_BACKOFF[: len(clock.sleeps)])


@pytest.mark.asyncio
async def test_a_response_error_fails_immediately_and_is_not_retried(
    clock: _FakeClock,
) -> None:
    # The 5xx analogue. A healthy redis REFUSING the command is a real defect;
    # retrying it into the 300s connection budget would mask it and turn a fast
    # failure into a slow one. It must escape on the FIRST raise.
    rid = uuid4()
    redis = _FlakyRedis(
        clock, outage=(_OUTAGE_START, 1e18), error=ResponseError("ERR unknown command")
    )
    with _patch_get(_run_completing_at(clock, 1e18)):
        with pytest.raises(ResponseError):
            await await_runs(
                run_ids=[rid],
                session_factory=_session_factory,
                redis=redis,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
                poll_interval_s=2.0,
                timeout_s=4800.0,
            )
    assert len(redis.pubsubs) == 1  # never reconnected
    assert clock.sleeps == []  # and slept NONE of the 300s ladder


@pytest.mark.asyncio
async def test_an_auth_error_fails_immediately_despite_subclassing_connectionerror(
    clock: _FakeClock,
) -> None:
    # `AuthenticationError` SUBCLASSES redis-py's `ConnectionError`, so a naive
    # `except ConnectionError` would swallow a rejected credential and hammer a
    # healthy server for 300s. It must be excluded by name.
    assert issubclass(AuthenticationError, RedisConnectionError)
    rid = uuid4()
    redis = _FlakyRedis(
        clock, outage=(_OUTAGE_START, 1e18), error=AuthenticationError("bad password")
    )
    with _patch_get(_run_completing_at(clock, 1e18)):
        with pytest.raises(AuthenticationError):
            await await_runs(
                run_ids=[rid],
                session_factory=_session_factory,
                redis=redis,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
                poll_interval_s=2.0,
                timeout_s=4800.0,
            )
    assert len(redis.pubsubs) == 1
    assert clock.sleeps == []


@pytest.mark.asyncio
async def test_reconnect_replays_no_paid_work_so_it_cannot_double_bill(
    clock: _FakeClock,
) -> None:
    # The idempotency argument, asserted rather than left in prose. Unlike an
    # HTTP POST, this retry replays no work: `await_runs` is a pure OBSERVER —
    # it receives child run ids that were already dispatched and never creates
    # one — so surviving a reconnect must leave the observed set byte-identical
    # and bill nothing. A lesson render is PAID; double-spawning one would be
    # far worse than the bug being fixed.
    kids = [uuid4(), uuid4()]
    redis = _FlakyRedis(clock, outage=(_OUTAGE_START, _OUTAGE_START + 40.0), error=_server_closed())
    with _patch_get(_run_completing_at(clock, _OUTAGE_START + 20.0)) as get:
        result = await await_runs(
            run_ids=kids,
            session_factory=_session_factory,
            redis=redis,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            poll_interval_s=2.0,
            timeout_s=4800.0,
        )
    assert len(redis.pubsubs) >= 2  # a reconnect really happened
    assert set(result) == set(kids)  # exactly the children handed in, no more
    # Every DB touch is a READ of a child we were given — no spawn, no write.
    assert get.await_count > 0
    assert {c.args[0] for c in get.await_args_list} == set(kids)
    # Each reconnect re-subscribes to exactly the same channels, never more.
    for ps in redis.pubsubs:
        assert ps.subscribed in ([], [f"runs:{k}" for k in kids])


@pytest.mark.asyncio
async def test_every_discarded_connect_attempt_is_released(clock: _FakeClock) -> None:
    # redis-py checks a connection OUT of the pool inside `subscribe` and keeps
    # it on the PubSub even when the command then fails. A ladder that discards
    # 24 failed attempts without releasing them exhausts the pool BEFORE the
    # outage ends, so the reconnect it exists to perform can never succeed —
    # the fix would defeat itself under exactly the outage it targets.
    rid = uuid4()
    redis = _FlakyRedis(clock, outage=(_OUTAGE_START, 1e18), error=_server_closed())
    with _patch_get(_run_completing_at(clock, 1e18)):
        with pytest.raises(RedisConnectionError):
            await await_runs(
                run_ids=[rid],
                session_factory=_session_factory,
                redis=redis,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
                poll_interval_s=2.0,
                timeout_s=4800.0,
            )
    assert len(redis.pubsubs) == 25  # 1 live + a full 24-attempt ladder
    assert all(ps.closed for ps in redis.pubsubs)  # not one connection leaked


@pytest.mark.asyncio
async def test_a_pubsub_is_never_released_twice(clock: _FakeClock) -> None:
    # When the reconnect itself exhausts its budget, the dead pubsub has
    # already been released — the `finally` must not release it again.
    # redis-py's `unsubscribe` RE-ACQUIRES a connection from the pool when it
    # no longer holds one, so a double release would dial redis a second time
    # purely to unsubscribe a subscription that is already gone.
    rid = uuid4()
    redis = _FlakyRedis(clock, outage=(_OUTAGE_START, 1e18), error=_server_closed())
    with _patch_get(_run_completing_at(clock, 1e18)):
        with pytest.raises(RedisConnectionError):
            await await_runs(
                run_ids=[rid],
                session_factory=_session_factory,
                redis=redis,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
                poll_interval_s=2.0,
                timeout_s=4800.0,
            )
    for ps in redis.pubsubs:
        assert ps.unsubscribed == [f"runs:{rid}"]  # exactly once, never twice


@pytest.mark.asyncio
async def test_each_poll_reads_every_open_child_in_one_query() -> None:
    """The DB poll is batched: one ``statuses`` call per refresh covering every
    open child — never a per-child ``get`` (N SELECTs per 2s tick)."""
    kids = [uuid4(), uuid4(), uuid4()]
    redis = _make_redis([])
    batched = AsyncMock(return_value=dict.fromkeys(kids, "completed"))
    per_child = AsyncMock(side_effect=AssertionError("per-child get in the poll loop"))
    with (
        patch.object(sub_dag.RunRepository, "statuses", new=batched),
        patch.object(sub_dag.RunRepository, "get", new=per_child),
    ):
        result = await await_runs(
            run_ids=kids,
            session_factory=_session_factory,
            redis=redis,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            poll_interval_s=0.05,
            timeout_s=2,
        )
    assert result == dict.fromkeys(kids, "completed")
    assert batched.await_count == 1
    assert set(batched.await_args_list[0].args[0]) == set(kids)
    per_child.assert_not_awaited()


@pytest.mark.asyncio
async def test_child_wait_cap_scales_with_the_number_of_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 5-lesson course on a fresh run must not be capped at the single-child 4800s.

    The lessons share one GPU pool and largely render in series, so the cap grows
    by a per-child allowance; one child keeps the old single-child cap.
    """
    parent = uuid4()
    monkeypatch.setattr(sub_dag, "get_settings", lambda: _settings(14400))
    with _patched_runs({parent: _run_row("running", started_ago_s=1.0)}):
        one, _ = await sub_dag._derive_child_wait_s(parent, _session_factory, None, 1)
        three, _ = await sub_dag._derive_child_wait_s(parent, _session_factory, None, 3)
        five, source = await sub_dag._derive_child_wait_s(parent, _session_factory, None, 5)
    assert one == sub_dag._FALLBACK_CHILD_WAIT_S
    assert three == sub_dag._FALLBACK_CHILD_WAIT_S + 2 * sub_dag._PER_EXTRA_CHILD_WAIT_S
    assert three > sub_dag._FALLBACK_CHILD_WAIT_S
    # 5 children would be 4800 + 4*1800 = 12000 uncapped; the remaining budget
    # (14400 - ~1s - 120s margin) is above that, so the scaled cap applies.
    assert five == 12000.0
    assert "cap=12000s for 5 children" in source


@pytest.mark.asyncio
async def test_scaled_child_wait_cap_is_still_bounded_by_the_remaining_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = uuid4()
    monkeypatch.setattr(sub_dag, "get_settings", lambda: _settings(14400))
    with _patched_runs({parent: _run_row("running", started_ago_s=8000)}):
        bound, _ = await sub_dag._derive_child_wait_s(parent, _session_factory, None, 10)
    # 14400 - 8000 = 6400 remaining, minus the 120s margin — NOT the 21000s cap.
    assert 6270.0 < bound < 6290.0
