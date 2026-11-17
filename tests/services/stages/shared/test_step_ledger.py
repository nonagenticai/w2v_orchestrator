"""Unit tests for the step ledger helper — DB-free, over an in-memory store.

The real table / RLS / upsert are covered in
``tests/database/repositories/test_stage_step_repository.py``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import enum
import hashlib
import logging
from pathlib import PurePosixPath
from uuid import UUID, uuid7

import pytest
from pydantic import BaseModel

from src.services.runs.run_context import RunContext
from src.services.stages.shared import step_ledger as mod
from src.services.stages.shared.step_ledger import (
    MAX_RESULT_BYTES,
    NullStepLedger,
    StepLedger,
    StoredStep,
    hash_inputs,
    step_ledger_factory,
)
from tests.support.reentry import (
    InMemoryStepStore,
    Outbound,
    SimulatedPodDeath,
    ctx_with_ledger,
    run_with_reentry,
)

RUN_ID = uuid7()
STAGE = "narrate"


class _Counted[T]:
    """A zero-arg coroutine factory that counts its invocations."""

    def __init__(self, value: T, raises: BaseException | None = None) -> None:
        self.calls = 0
        self._value = value
        self._raises = raises

    async def __call__(self) -> T:
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return self._value


def _ledger(store: InMemoryStepStore, stage: str = STAGE) -> StepLedger:
    return StepLedger(store, RUN_ID, stage)


def _events(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The ``step_ledger.<event>`` token of every ledger log record, in order."""
    return [r.getMessage().split(" ", 1)[0] for r in caplog.records if r.name == mod.__name__]


# ── once(): the four contract outcomes ───────────────────────────────────────


async def test_miss_calls_fn_stores_result_and_returns_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger=mod.__name__)
    store = InMemoryStepStore()
    fn = _Counted({"audio": "s3://a.wav", "seconds": 4.2})

    out = await _ledger(store).once("scene-1", "h1", fn)

    assert out == {"audio": "s3://a.wav", "seconds": 4.2}
    assert fn.calls == 1
    assert store.rows[(RUN_ID, STAGE, "scene-1")] == StoredStep(
        input_hash="h1", result={"value": {"audio": "s3://a.wav", "seconds": 4.2}}
    )
    assert _events(caplog) == ["step_ledger.miss"]


async def test_hit_returns_stored_result_without_calling_fn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = InMemoryStepStore()
    await _ledger(store).once("scene-1", "h1", _Counted({"audio": "first"}))
    caplog.set_level(logging.INFO, logger=mod.__name__)
    caplog.clear()

    # A FRESH ledger over the same store = the re-entered stage on a new pod.
    second = _Counted({"audio": "SECOND — must never be produced"})
    out = await _ledger(store).once("scene-1", "h1", second)

    assert out == {"audio": "first"}
    assert second.calls == 0
    assert len(store.puts) == 1, "a hit must not rewrite the row"
    assert _events(caplog) == ["step_ledger.hit"]
    assert caplog.records[-1].levelno == logging.INFO


async def test_stale_hash_calls_fn_and_replaces_the_row(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = InMemoryStepStore()
    await _ledger(store).once("scene-1", "old-hash", _Counted("old"))
    caplog.set_level(logging.INFO, logger=mod.__name__)
    caplog.clear()

    fn = _Counted("new")
    out = await _ledger(store).once("scene-1", "new-hash", fn)

    assert out == "new"
    assert fn.calls == 1
    assert store.rows[(RUN_ID, STAGE, "scene-1")] == StoredStep("new-hash", {"value": "new"})
    assert len(store.rows) == 1
    assert _events(caplog) == ["step_ledger.stale"]


async def test_fn_raising_propagates_and_stores_nothing() -> None:
    store = InMemoryStepStore()
    boom = RuntimeError("tts 503")

    with pytest.raises(RuntimeError) as caught:
        await _ledger(store).once("scene-1", "h1", _Counted(None, raises=boom))

    assert caught.value is boom
    assert store.rows == {}
    assert store.puts == []

    # …and the next attempt is a plain miss, not a poisoned row.
    fn = _Counted("ok")
    assert await _ledger(store).once("scene-1", "h1", fn) == "ok"
    assert fn.calls == 1


async def test_cancellation_propagates_and_stores_nothing() -> None:
    """A drain cancels the stage task; the ledger must not swallow or record it."""
    store = InMemoryStepStore()

    with pytest.raises(asyncio.CancelledError):
        await _ledger(store).once("scene-1", "h1", _Counted(None, raises=asyncio.CancelledError()))

    assert store.rows == {}


# ── keys are scoped ──────────────────────────────────────────────────────────


async def test_rows_are_scoped_by_run_stage_and_step_key() -> None:
    store = InMemoryStepStore()
    await StepLedger(store, RUN_ID, "narrate").once("k", "h", _Counted("narrate/k"))

    other_step = _Counted("narrate/k2")
    other_stage = _Counted("compile/k")
    other_run = _Counted("run2/narrate/k")
    assert await StepLedger(store, RUN_ID, "narrate").once("k2", "h", other_step) == "narrate/k2"
    assert await StepLedger(store, RUN_ID, "compile").once("k", "h", other_stage) == "compile/k"
    assert await StepLedger(store, uuid7(), "narrate").once("k", "h", other_run) == "run2/narrate/k"

    assert (other_step.calls, other_stage.calls, other_run.calls) == (1, 1, 1)
    assert len(store.rows) == 4


# ── dump / load ──────────────────────────────────────────────────────────────


class _Clip(BaseModel):
    url: str
    seconds: float


# ── key scope: which re-executions hit (module docstring § "Key scope") ───────
#
# What moves on each re-execution is pinned where it happens — ``attempts`` /
# ``interruptions`` in ``tests/services/runs/test_dag_executor.py`` (timeout ⇒
# bump, cancellation ⇒ pending, any other exception ⇒ terminal even with resume
# wired) and the write-once ``mark_*`` guards in the repository tests. These three
# pin the LEDGER's side: the key is ``run_id``, and no stage-row counter is in it.


async def test_case_a_hard_pod_death_redispatch_hits() -> None:
    """Nothing on the stage row moves; the re-dispatch is the same run ⇒ hit."""
    store = InMemoryStepStore()
    await _ledger(store).once("draw", "h", _Counted("paid"))

    after_death = _Counted("paid again")
    assert await _ledger(store).once("draw", "h", after_death) == "paid"
    assert after_death.calls == 0


@pytest.mark.parametrize("moved", ["interruptions"], ids=["drain"])
async def test_case_b_drain_or_resumable_timeout_hits(moved: str) -> None:
    """A drain bumps ``interruptions``, a resumable timeout ``attempts`` — same run.

    Neither counter is part of the key, so both re-entries reuse what was paid for.
    That is the reason rows are NOT scoped by ``attempts``: it moves here too.
    """
    store = InMemoryStepStore()
    counters = {"attempts": 0, "interruptions": 0}
    await _ledger(store).once("draw", "h", _Counted("paid"))

    counters[moved] += 1  # what DAGExecutor records before the reconciler resumes
    resumed = _Counted("paid again")
    assert await _ledger(store).once("draw", "h", resumed) == "paid"
    assert resumed.calls == 0
    assert all(key == (RUN_ID, STAGE, "draw") for key in store.rows)


async def test_case_c_a_failed_then_retried_stage_misses() -> None:
    """A failed stage is terminal for its run; its retry is a CHILD run ⇒ miss.

    ``regenerate`` starts a new run, so the retry carries a new ``run_id`` and can
    never be served the replies that led to the failure — with the parent's rows
    still in the table.
    """
    store = InMemoryStepStore()
    await _ledger(store).once("draw", "h", _Counted("reply that failed the stage"))

    retry = _Counted("fresh reply")
    child_run = uuid7()
    assert await StepLedger(store, child_run, STAGE).once("draw", "h", retry) == "fresh reply"
    assert retry.calls == 1
    assert set(store.rows) == {(RUN_ID, STAGE, "draw"), (child_run, STAGE, "draw")}


async def test_dump_and_load_round_trip_a_model() -> None:
    store = InMemoryStepStore()
    clip = _Clip(url="s3://clip.mp4", seconds=3.5)

    first = await _ledger(store).once(
        "render", "h", _Counted(clip), dump=lambda c: c.model_dump(mode="json"), load=_load_clip
    )
    assert first is clip, "a miss returns fn's own object, not a round-tripped copy"
    assert store.rows[(RUN_ID, STAGE, "render")].result == {
        "value": {"url": "s3://clip.mp4", "seconds": 3.5}
    }

    again = _Counted(clip)
    second = await _ledger(store).once(
        "render", "h", again, dump=lambda c: c.model_dump(mode="json"), load=_load_clip
    )
    assert second == clip
    assert isinstance(second, _Clip)
    assert again.calls == 0


def _load_clip(raw: object) -> _Clip:
    return _Clip.model_validate(raw)


async def test_a_none_result_is_a_real_stored_result_not_a_missing_row() -> None:
    """``None`` is a legitimate sub-step result; it must hit on re-entry."""
    store = InMemoryStepStore()
    await _ledger(store).once("probe", "h", _Counted(None))

    fn = _Counted("must not run")
    assert await _ledger(store).once("probe", "h", fn) is None
    assert fn.calls == 0


async def test_unloadable_stored_value_falls_through_to_fn_and_is_replaced(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """E.g. the result model changed shape between the image that wrote the row and
    the image re-entering the stage."""
    store = InMemoryStepStore()
    store.rows[(RUN_ID, STAGE, "render")] = StoredStep("h", {"value": {"legacy": True}})
    caplog.set_level(logging.INFO, logger=mod.__name__)
    clip = _Clip(url="s3://new.mp4", seconds=1.0)
    fn = _Counted(clip)

    out = await _ledger(store).once(
        "render", "h", fn, dump=lambda c: c.model_dump(mode="json"), load=_load_clip
    )

    assert out is clip
    assert fn.calls == 1
    assert store.rows[(RUN_ID, STAGE, "render")].result == {
        "value": {"url": "s3://new.mp4", "seconds": 1.0}
    }
    assert _events(caplog) == ["step_ledger.unreadable"]
    assert caplog.records[-1].levelno == logging.WARNING


async def test_a_row_that_is_not_an_envelope_is_treated_as_unreadable() -> None:
    store = InMemoryStepStore()
    store.rows[(RUN_ID, STAGE, "k")] = StoredStep("h", ["not", "an", "envelope"])
    fn = _Counted("fresh")

    assert await _ledger(store).once("k", "h", fn) == "fresh"
    assert fn.calls == 1


# ── the ledger never fails the stage ─────────────────────────────────────────


async def test_over_cap_result_is_returned_but_not_stored(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger=mod.__name__)
    store = InMemoryStepStore()
    big = "x" * (MAX_RESULT_BYTES + 1)

    out = await _ledger(store).once("blob", "h", _Counted(big))

    assert out == big
    assert store.rows == {}
    assert store.puts == []
    assert _events(caplog) == ["step_ledger.miss", "step_ledger.oversize"]
    assert caplog.records[-1].levelno == logging.WARNING


async def test_cap_is_measured_in_utf8_bytes_and_is_inclusive() -> None:
    store = InMemoryStepStore()
    # A JSON string of n ASCII chars encodes to n + 2 bytes (the quotes).
    exactly_at_cap = "x" * (MAX_RESULT_BYTES - 2)
    await _ledger(store).once("at-cap", "h", _Counted(exactly_at_cap))
    assert (RUN_ID, STAGE, "at-cap") in store.rows

    # Same character count, but 3 bytes each in UTF-8 → far over the cap.
    multibyte = "€" * (MAX_RESULT_BYTES - 2)
    await _ledger(store).once("multibyte", "h", _Counted(multibyte))
    assert (RUN_ID, STAGE, "multibyte") not in store.rows


async def test_unserializable_result_is_returned_but_not_stored(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger=mod.__name__)
    store = InMemoryStepStore()
    not_json = object()

    out = await _ledger(store).once("k", "h", _Counted(not_json))

    assert out is not_json
    assert store.rows == {}
    assert _events(caplog) == ["step_ledger.miss", "step_ledger.dump_failed"]


async def test_nan_result_is_not_stored() -> None:
    """jsonb rejects NaN; it must be refused before it reaches the database."""
    store = InMemoryStepStore()

    out = await _ledger(store).once("k", "h", _Counted({"score": float("nan")}))

    assert out["score"] != out["score"]  # still NaN, still returned
    assert store.puts == []


async def test_dump_raising_returns_the_result_and_stores_nothing() -> None:
    store = InMemoryStepStore()

    def _bad_dump(_value: str) -> str:
        raise ValueError("dump bug")

    assert await _ledger(store).once("k", "h", _Counted("result"), dump=_bad_dump) == "result"
    assert store.rows == {}


async def test_db_read_error_falls_through_to_fn(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger=mod.__name__)
    store = InMemoryStepStore()
    store.fail_get = ConnectionError("pgbouncer gone")
    fn = _Counted("computed")

    out = await _ledger(store).once("k", "h", fn)

    assert out == "computed"
    assert fn.calls == 1
    # The write is still attempted — the read outage may have been transient.
    assert store.rows[(RUN_ID, STAGE, "k")] == StoredStep("h", {"value": "computed"})
    assert _events(caplog) == ["step_ledger.read_failed", "step_ledger.miss"]


async def test_db_write_error_still_returns_the_result(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger=mod.__name__)
    store = InMemoryStepStore()
    store.fail_put = ConnectionError("pgbouncer gone")
    fn = _Counted("computed")

    out = await _ledger(store).once("k", "h", fn)

    assert out == "computed"
    assert fn.calls == 1
    assert store.rows == {}
    assert _events(caplog) == ["step_ledger.miss", "step_ledger.write_failed"]


async def test_total_db_outage_degrades_to_no_ledger_at_all() -> None:
    store = InMemoryStepStore()
    store.fail_get = store.fail_put = ConnectionError("down")
    fn = _Counted("v")

    assert await _ledger(store).once("k", "h", fn) == "v"
    assert await _ledger(store).once("k", "h", fn) == "v"
    assert fn.calls == 2


# ── NullStepLedger + ctx wiring ──────────────────────────────────────────────


async def test_null_ledger_always_calls_fn() -> None:
    ledger = NullStepLedger("narrate")
    fn = _Counted("v")

    assert await ledger.once("k", "h", fn) == "v"
    assert await ledger.once("k", "h", fn) == "v"

    assert fn.calls == 2
    assert ledger.run_id is None
    assert isinstance(ledger, StepLedger), "handlers type against StepLedger only"


def _bare_ctx() -> RunContext:
    return RunContext(run_id=uuid7(), deliverable_id=uuid7(), config_snapshot={})


async def test_ctx_without_a_factory_hands_out_a_null_ledger() -> None:
    """Every pre-existing ``RunContext(...)`` call site keeps working unchanged."""
    ctx = _bare_ctx()
    assert ctx.step_ledger_factory is None

    ledger = ctx.step_ledger("narrate")

    assert isinstance(ledger, NullStepLedger)
    assert ledger.stage_name == "narrate"
    fn = _Counted(1)
    await ledger.once("k", "h", fn)
    await ledger.once("k", "h", fn)
    assert fn.calls == 2


async def test_ctx_with_a_factory_hands_out_a_ledger_bound_to_run_and_stage() -> None:
    store = InMemoryStepStore()
    ctx = ctx_with_ledger(_bare_ctx(), store)

    ledger = ctx.step_ledger("narrate")
    await ledger.once("k", "h", _Counted("v"))

    assert not isinstance(ledger, NullStepLedger)
    assert (ledger.run_id, ledger.stage_name) == (ctx.run_id, "narrate")
    assert list(store.rows) == [(ctx.run_id, "narrate", "k")]


def test_the_factory_survives_replace_and_does_not_affect_equality() -> None:
    """``dag_executor`` rebuilds ctx with ``replace(ctx, prior_run_outputs=…)`` before
    every stage; the ledger must ride along, and must not make two otherwise-equal
    contexts compare unequal."""
    store = InMemoryStepStore()
    bare = _bare_ctx()
    wired = ctx_with_ledger(bare, store)

    rebuilt = dataclasses.replace(wired, prior_run_outputs={"a": {"x": 1}})

    assert rebuilt.step_ledger_factory is wired.step_ledger_factory
    assert wired == bare
    assert "step_ledger_factory" not in repr(wired)


class _FakeSessions:
    """Session-factory stand-in recording how many sessions were opened."""

    def __init__(self) -> None:
        self.opened = 0

    def __call__(self) -> _FakeSessions:
        self.opened += 1
        return self

    async def __aenter__(self) -> object:
        raise ConnectionError("no database in a unit test")

    async def __aexit__(self, *exc: object) -> bool:
        return False


async def test_dispatcher_factory_binds_run_and_survives_an_unreachable_db() -> None:
    """``step_ledger_factory`` is what ``RunDispatcher`` wires; with the DB down the
    ledger it builds must still run the step."""
    sessions = _FakeSessions()
    run_id = uuid7()
    make = step_ledger_factory(sessions, run_id=run_id, tenant_id=uuid7())  # ty: ignore[invalid-argument-type]

    ledger = make("compile")
    fn = _Counted("v")
    out = await ledger.once("k", "h", fn)

    assert (ledger.run_id, ledger.stage_name) == (run_id, "compile")
    assert out == "v"
    assert fn.calls == 1
    assert sessions.opened == 2, "one session for the read, one for the write"


# ── hash_inputs ──────────────────────────────────────────────────────────────


class _Mode(enum.Enum):
    FAST = "fast"


@dataclasses.dataclass(frozen=True)
class _Shot:
    index: int
    prompt: str


def test_hash_inputs_is_a_stable_sha256_hex_digest() -> None:
    digest = hash_inputs("prompt", {"b": 2, "a": 1}, 3)

    # Pinned to literal digests of the canonical JSON: if these move, every row
    # already stored in prod silently stops hitting after the deploy.
    assert digest == hashlib.sha256(b'["prompt",{"a":1,"b":2},3]').hexdigest()
    assert hash_inputs() == hashlib.sha256(b"[]").hexdigest()


def test_hash_inputs_is_order_sensitive_across_parts_only() -> None:
    assert hash_inputs("a", "b") != hash_inputs("b", "a")
    assert hash_inputs({"x": 1, "y": [1, 2]}) == hash_inputs({"y": [1, 2], "x": 1})
    assert hash_inputs([1, 2]) != hash_inputs([2, 1])
    assert hash_inputs({3, 1, 2}) == hash_inputs({2, 3, 1})
    assert hash_inputs(frozenset({"b", "a"})) == hash_inputs({"a", "b"})


def test_hash_inputs_distinguishes_values_and_boundaries() -> None:
    assert hash_inputs("ab", "c") != hash_inputs("a", "bc")
    assert hash_inputs("1") != hash_inputs(1)
    assert hash_inputs(None) != hash_inputs()
    assert hash_inputs({"model": "a"}) != hash_inputs({"model": "b"})


def test_hash_inputs_canonicalizes_common_non_json_types() -> None:
    uid = UUID("01890000-0000-7000-8000-000000000001")
    when = datetime.datetime(2026, 10, 6, 12, 0, tzinfo=datetime.UTC)

    assert hash_inputs(uid) == hash_inputs(str(uid))
    assert hash_inputs(_Clip(url="u", seconds=1.0)) == hash_inputs({"url": "u", "seconds": 1.0})
    assert hash_inputs(_Shot(1, "p")) == hash_inputs({"index": 1, "prompt": "p"})
    assert hash_inputs(_Mode.FAST) == hash_inputs("fast")
    assert hash_inputs(PurePosixPath("a/b")) == hash_inputs("a/b")
    assert hash_inputs(when) == hash_inputs(when.isoformat())
    assert hash_inputs((1, 2)) == hash_inputs([1, 2])
    assert hash_inputs(b"abc") == hash_inputs(bytearray(b"abc"))
    assert hash_inputs(b"abc") != hash_inputs(b"abd")
    assert hash_inputs({"nested": [_Shot(1, "p")]}) == hash_inputs(
        {"nested": [{"index": 1, "prompt": "p"}]}
    )


def test_hash_inputs_refuses_what_it_cannot_hash_stably() -> None:
    """A default ``repr`` carries a memory address — hashing it would differ on every
    re-entry and make the ledger a silent no-op. Fail at the call site instead."""
    with pytest.raises(TypeError, match="cannot canonicalize"):
        hash_inputs(object())
    with pytest.raises(ValueError, match="Out of range float"):
        hash_inputs(float("nan"))


# ── the re-entry harness itself (tests/support/reentry.py) ───────────────────


class _ThreeSceneStage:
    """Minimal handler shape: one LLM call, then one TTS call per scene."""

    name = "narrate"

    def __init__(self, llm, tts, *, wrapped: bool = True) -> None:
        self._llm = llm
        self._tts = tts
        self._wrapped = wrapped

    async def run(self, _inputs: dict, ctx: RunContext) -> dict:
        ledger = ctx.step_ledger(self.name) if self._wrapped else NullStepLedger(self.name)
        script = await ledger.once("script", hash_inputs("brief-v1"), lambda: self._llm("brief-v1"))
        clips = [
            await ledger.once(
                f"tts:{scene}", hash_inputs(scene, script), lambda scene=scene: self._tts(scene)
            )
            for scene in ("s1", "s2", "s3")
        ]
        return {"script": script, "clips": clips}


def _stage(out: Outbound, *, wrapped: bool = True) -> _ThreeSceneStage:
    return _ThreeSceneStage(
        out.fake("llm.draft", "the script"),
        out.fake("tts.synthesize", lambda scene: f"s3://{scene}.wav"),
        wrapped=wrapped,
    )


async def test_reentry_reuses_every_finished_sub_step() -> None:
    out = Outbound()
    stage = _stage(out)

    result = await run_with_reentry(
        lambda ctx: stage.run({}, ctx), ctx=_bare_ctx(), outbound=out, die_after=2
    )

    assert result.first_attempt_calls == {"llm.draft": 1, "tts.synthesize": 1}
    assert result.second_attempt_calls == {"tts.synthesize": 2}
    result.assert_calls({"llm.draft": 1, "tts.synthesize": 3})
    assert result.output == {
        "script": "the script",
        "clips": ["s3://s1.wav", "s3://s2.wav", "s3://s3.wav"],
    }
    assert result.store.step_keys("narrate") == {"script", "tts:s1", "tts:s2", "tts:s3"}


async def test_reentry_harness_catches_an_unwrapped_stage() -> None:
    """The harness must be able to FAIL: the same stage without the ledger repeats
    its paid calls, and ``assert_calls`` says so."""
    out = Outbound()
    stage = _stage(out, wrapped=False)

    result = await run_with_reentry(
        lambda ctx: stage.run({}, ctx), ctx=_bare_ctx(), outbound=out, die_after=2
    )

    assert result.total_calls == {"llm.draft": 2, "tts.synthesize": 4}
    with pytest.raises(AssertionError, match="REPEATED a finished, paid call"):
        result.assert_calls({"llm.draft": 1, "tts.synthesize": 3})


async def test_reentry_harness_fails_when_the_stage_never_dies() -> None:
    out = Outbound()
    stage = _stage(out)

    with pytest.raises(AssertionError, match="finished without dying"):
        await run_with_reentry(
            lambda ctx: stage.run({}, ctx), ctx=_bare_ctx(), outbound=out, die_after=99
        )


async def test_death_on_the_first_call_leaves_the_ledger_empty() -> None:
    out = Outbound()
    stage = _stage(out)
    store = InMemoryStepStore()
    ctx = ctx_with_ledger(_bare_ctx(), store)

    out.arm(0)
    with pytest.raises(SimulatedPodDeath):
        await stage.run({}, ctx)

    assert out.total == 0
    assert store.rows == {}
