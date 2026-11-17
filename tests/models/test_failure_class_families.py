"""``attribute_exception`` against the failure families MEASURED on prod.

Companion to ``test_failure_class_attribution.py``, which proves the allowlist's
REFUSALS. This file proves the opposite half — that the families which carry a
structural signal now get attributed — and it re-proves the refusal for the
families that do NOT, because those are the ones a "raise the coverage number"
change would quietly eat.

The taxonomy is the 47 armed-era (2026-09-02 → 09-10) failed runs, whose
``failure_class`` was ``unknown`` for 45 of them:

    n=20  bare ``TimeoutError``                     → stays unknown (ambiguous)
    n=15  ``StageTimeoutError`` per-attempt budget  → stays unknown (ambiguous)
    n=12  ``reconcile_timeline`` bare RuntimeError  → stays unknown (unstamped)
    n=11  transport disconnect outside ``guarded``  → EXTERNAL   (fixed here)
    n=8   ``StageProducedNothingError``             → OUR_DEFECT (fixed here)
    n=3   WM ``TransientUpstreamError`` on 422      → stays unknown (ambiguous)
          ``ChildRunError`` cascade                 → INHERITS the child's

Every "stays unknown" case below is load-bearing, not filler: it is the guard
that stops the next coverage push from attributing a timeout by guessing.
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import AsyncIterator

import httpx
import pytest
from w2v_common.stages import StageRetriesExhaustedError, StageTimeoutError

from src.models.failure_class import (
    EXTERNAL,
    OUR_DEFECT,
    UNKNOWN,
    attribute_exception,
)
from src.services.runs import sub_dag
from src.services.runs.sub_dag import ChildRunError
from src.services.stages.base import StageProducedNothingError


def _wrapped(inner: BaseException) -> BaseException:
    """``inner`` as ``stage_call`` delivers it to the persistence site.

    ``w2v_common.stages.execution.stage_call`` raises
    ``StageRetriesExhaustedError(...) from exc``, and ``dag_executor._run_one``
    re-raises bare, so the chain the dispatcher attributes on is exactly this.
    """
    try:
        raise StageRetriesExhaustedError(
            stage_name="voice_director", attempts=2, last_error=inner
        ) from inner
    except StageRetriesExhaustedError as exc:
        return exc


# ── The families that now classify ───────────────────────────────────────────


@pytest.mark.parametrize(
    "inner",
    [
        httpx.ReadError("peer closed connection"),
        httpx.RemoteProtocolError("server disconnected without sending a response"),
        httpx.ConnectError("connection refused"),
        ConnectionResetError("connection reset by peer"),
    ],
    ids=["read_error", "remote_protocol", "connect_error", "builtin_conn_reset"],
)
def test_a_transport_disconnect_outside_guarded_records_external(
    inner: BaseException,
) -> None:
    """The n=11 family: a dependency never delivered a body.

    ``upstream.guarded`` would have stamped these ``external`` via
    ``classify_transport`` — but only the calls that go THROUGH it do, and these
    reached the persistence site raw. Attribution must not be a function of which
    call site the failure came through, so the type carries the same verdict.
    """
    assert attribute_exception(_wrapped(inner)) == EXTERNAL


def test_an_unguarded_5xx_is_attributed_from_the_status_it_still_carries() -> None:
    """A raw ``HTTPStatusError`` never passed through ``guarded``, so it has no
    stamp — but it is still carrying the response, and 5xx has exactly one
    reading: a dependency failed. Same map ``guarded`` would have used."""
    request = httpx.Request("POST", "http://worker-management:8887/workers/x/message")
    response = httpx.Response(503, request=request, text="upstream unavailable")
    raw = httpx.HTTPStatusError("503", request=request, response=response)
    assert attribute_exception(_wrapped(raw)) == EXTERNAL


def test_a_stage_that_reported_success_and_produced_nothing_is_our_defect() -> None:
    """The n=8 family. No customer input and no upstream service can make a stage
    of OURS claim success while producing nothing for a field it declared
    mandatory — if a downstream failure emptied it, the stage swallowed that
    failure, and the swallowing is the defect."""
    exc = StageProducedNothingError("demo_video_slides", ["slides"])
    assert attribute_exception(exc) == OUR_DEFECT


def test_a_child_run_cascade_inherits_the_childs_attribution() -> None:
    """A cascade has no fault of its own. Attributing ``ChildRunError`` on its own
    type would file every sub-DAG failure ``unknown`` however well the child was
    attributed."""
    exc = ChildRunError(
        run_id=uuid.uuid4(),
        status="failed",
        error="TypeError: 'NoneType' object is not subscriptable",
        failure_class=OUR_DEFECT,
    )
    assert attribute_exception(exc) == OUR_DEFECT


def test_an_unattributed_child_does_not_veto_a_definite_cause() -> None:
    """A child that failed before the column was armed carries ``unknown``. That
    must be SKIPPED, not accepted — otherwise a legacy child poisons a parent
    whose own chain names a transport failure."""
    child = ChildRunError(run_id=uuid.uuid4(), status="failed", error="legacy row")
    assert child.failure_class == UNKNOWN
    try:
        raise child from httpx.ConnectError("connection refused")
    except ChildRunError as exc:
        assert attribute_exception(exc) == EXTERNAL


# ── The families that deliberately stay ``unknown`` ──────────────────────────


@pytest.mark.parametrize(
    "inner,why",
    [
        (
            TimeoutError(),
            "bare zero-arg TimeoutError: our budget being too small and a "
            "dependency hanging produce the IDENTICAL object (n=20)",
        ),
        (
            StageTimeoutError("stage 'promo_render': all 2 attempts timed out"),
            "the per-attempt stage budget expiring says nothing about whose "
            "fault the slowness was (n=15)",
        ),
        (
            httpx.ReadTimeout("timed out"),
            "a TransportError that is ALSO a timeout — excluded from the "
            "external allowlist for the same reason as the two above",
        ),
        (
            httpx.PoolTimeout("pool timeout"),
            "our own connection-pool ceiling, reached under OUR concurrency",
        ),
        (
            RuntimeError(
                "reconcile_timeline: timeline GROSSLY UNDERFULL — 1.40s of the "
                "5.00s target (28.0%) is below the 60% floor"
            ),
            "a bare RuntimeError from stage code: could be our planner or a "
            "thin brief, and prose matching is banned (n=12)",
        ),
        (
            ValueError("model returned no usable shots"),
            "ValueError is the re-prompt signal AND pydantic's validation base",
        ),
    ],
    ids=[
        "bare_timeout",
        "stage_timeout",
        "httpx_read_timeout",
        "httpx_pool_timeout",
        "reconcile_underfull",
        "value_error",
    ],
)
def test_an_ambiguous_family_is_still_refused(inner: BaseException, why: str) -> None:
    assert attribute_exception(_wrapped(inner)) == UNKNOWN, why


def test_an_unguarded_ambiguous_4xx_does_not_veto_a_transport_cause() -> None:
    """422 is ambiguous (WM answers it for a contract fault AND a lost model
    draw), so the status pass must fall THROUGH rather than returning its
    ``unknown`` — a transport failure deeper in the same chain still wins."""
    request = httpx.Request("POST", "http://worker-management:8887/workers/x/message")
    response = httpx.Response(422, request=request, text="{}")
    try:
        raise httpx.HTTPStatusError("422", request=request, response=response) from httpx.ReadError(
            "peer closed connection"
        )
    except httpx.HTTPStatusError as exc:
        assert attribute_exception(exc) == EXTERNAL


def test_a_bad_url_scheme_is_not_blamed_on_the_dependency() -> None:
    """``UnsupportedProtocol`` is an ``httpx.TransportError`` — so the external
    allowlist would sweep it up — but a URL THIS process built with a scheme
    httpx cannot speak is our configuration, not a dependency failing. It is
    subtracted explicitly, and lands in ``unknown`` rather than ``external``."""
    assert attribute_exception(_wrapped(httpx.UnsupportedProtocol("unknown scheme"))) == UNKNOWN


def test_a_programming_error_outranks_a_transport_cause() -> None:
    """A ``KeyError`` raised while handling a dependency's answer is OUR bug: the
    dependency answered and we mishandled it. The our_defect pass must therefore
    run BEFORE the external pass, not after."""
    try:
        raise KeyError("shots") from httpx.ReadError("peer closed connection")
    except KeyError as exc:
        assert attribute_exception(exc) == OUR_DEFECT


# ── The wiring that reads the child's column ─────────────────────────────────


async def test_raise_on_non_completed_reads_the_childs_persisted_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The inheritance above is only real if the raise site actually READS the
    column. This exercises ``_raise_on_non_completed``'s own repository call —
    without it, ``ChildRunError``'s parameter is a field nothing ever fills.
    """

    class _Row:
        error = "httpx.ConnectError: connection refused"
        failure_class = EXTERNAL

    class _Repo:
        def __init__(self, session: object) -> None:
            self._session = session

        async def get(self, run_id: uuid.UUID) -> _Row:
            return _Row()

    @contextlib.asynccontextmanager
    async def _sessions() -> AsyncIterator[object]:
        yield object()

    async def _noop_bind(session: object, tenant_id: uuid.UUID | None) -> None:
        return None

    monkeypatch.setattr(sub_dag, "RunRepository", _Repo)
    monkeypatch.setattr(sub_dag, "bind_session_tenant", _noop_bind)

    child_id = uuid.uuid4()
    with pytest.raises(ChildRunError) as caught:
        await sub_dag._raise_on_non_completed({child_id: "failed"}, _sessions, None)

    assert caught.value.failure_class == EXTERNAL
    assert attribute_exception(caught.value) == EXTERNAL


async def test_a_child_row_with_a_null_class_stays_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-arming child row has ``failure_class = NULL``. The coercion through
    ``FAILURE_CLASSES`` must land that on ``UNKNOWN`` rather than letting a raw
    ``None`` reach the attribute and be compared against the vocabulary."""

    class _Row:
        error = "legacy failure"
        failure_class = None

    class _Repo:
        def __init__(self, session: object) -> None: ...

        async def get(self, run_id: uuid.UUID) -> _Row:
            return _Row()

    @contextlib.asynccontextmanager
    async def _sessions() -> AsyncIterator[object]:
        yield object()

    async def _noop_bind(session: object, tenant_id: uuid.UUID | None) -> None:
        return None

    monkeypatch.setattr(sub_dag, "RunRepository", _Repo)
    monkeypatch.setattr(sub_dag, "bind_session_tenant", _noop_bind)

    with pytest.raises(ChildRunError) as caught:
        await sub_dag._raise_on_non_completed({uuid.uuid4(): "failed"}, _sessions, None)

    assert caught.value.failure_class == UNKNOWN
