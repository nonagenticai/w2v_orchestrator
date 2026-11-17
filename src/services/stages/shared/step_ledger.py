"""Step ledger — reuse a stage's FINISHED sub-steps when the stage is re-entered.

**The problem.** A stage is one ``handler.run({}, ctx)`` call and its output is
persisted only when that call returns. A pod death half-way (every merge rolls the
``replicas: 1`` orchestrator) makes the reconciler re-dispatch the run, and the
handler restarts FROM THE TOP: every paid call it had already finished — LLM, TTS,
render, scrape — is made, and billed, again.

**The fix.** Wrap each expensive sub-step in :meth:`StepLedger.once`. The first
time it runs, its result is written to ``stage_step_results`` (migration 0071),
keyed ``(run_id, stage_name, step_key)``. On re-entry the stored result is returned
and the call is not made.

Usage (the whole API a stage needs)
-----------------------------------
::

    from src.services.stages.shared.step_ledger import hash_inputs

    ledger = ctx.step_ledger(self.name)            # StepLedger; never None

    script = await ledger.once(
        "draft_script",                            # step_key: stable, unique in the stage
        hash_inputs(brief, model_name, prompt_v),  # EVERYTHING the result depends on
        lambda: self._llm.draft(brief),            # zero-arg coroutine factory
        dump=lambda s: s.model_dump(mode="json"),  # T -> JSON   (omit if T is JSON)
        load=Script.model_validate,                # JSON -> T   (omit if T is JSON)
    )

``ctx.step_ledger(stage_name)`` returns a real ledger when the run is DB-backed
(the dispatcher wires it) and a :class:`NullStepLedger` — which simply calls
``fn`` — everywhere else (unit tests, dry runs, any hand-built ``RunContext``).
Handler code is identical in both cases.

Contract of ``once(step_key, input_hash, fn, *, dump=None, load=None) -> T``
----------------------------------------------------------------------------
* stored row, SAME ``input_hash``  → ``load(stored)`` is returned, ``fn`` is NOT
  called. Logged ``step_ledger.hit``.
* no row                           → ``fn()`` is awaited, ``dump(result)`` stored,
  the ORIGINAL result returned. Logged ``step_ledger.miss``.
* stored row, DIFFERENT hash       → as a miss, and the row is replaced. Logged
  ``step_ledger.stale``.
* ``fn`` raises                    → the exception propagates, NOTHING is stored.

**The ledger can never fail a stage.** Every ledger-side problem degrades to
"call ``fn`` as if there were no ledger" and is logged at warning:

* read fails (DB down, RLS, timeout)   → ``fn`` is called (``step_ledger.read_failed``)
* ``load`` raises on a stored value    → ``fn`` is called, row replaced
  (``step_ledger.unreadable`` — e.g. the result model changed shape between images)
* ``dump`` raises / result not JSON    → result returned, not stored
  (``step_ledger.dump_failed``)
* payload over :data:`MAX_RESULT_BYTES` → result returned, not stored
  (``step_ledger.oversize``)
* write fails                          → result returned (``step_ledger.write_failed``)

Only ``Exception`` is absorbed; ``asyncio.CancelledError`` (a drain) propagates.

Rules for callers — read before wrapping a call site
----------------------------------------------------
1. **``input_hash`` must cover every input that changes the result** — prompt
   text, model name, prompt/template version, upstream stage outputs, asset urls.
   Use :func:`hash_inputs`; do not hand-roll hashing. An under-specified hash
   silently serves a stale result; an over-specified one (a timestamp, a fresh
   uuid) never hits. Do NOT put ``run_id`` / ``stage_name`` in it — they are the key.
   **Never hash a secret.** ``input_hash`` is stored in the clear; a digest of a
   password is an offline guessing oracle. Hash the login's non-secret identity
   (``site_capture._login_identity``).
2. **``step_key`` is unique within the stage and stable across re-entries.** For a
   loop, put the stable item identity in it (``f"narrate:{scene_id}"``), never an
   index into a list whose order can change.
3. **Store the RESULT, not a handle to an expiring thing** — a job id or an
   object key, not a presigned URL minted for this step alone.
   One exception, and why: a stored stage OUTPUT model that itself carries a
   presigned asset url (``site_capture``'s ``SiteClip`` / ``SiteScreen``) is
   stored as it is. That exact url is what
   ``stage_outputs.output`` persists for the same run under the same RLS, so the
   row adds no exposure, and it goes stale there identically on any paused run —
   which is why consumers re-sign from the url's object path
   (``shot_presign.refresh_shot_asset_urls``) and never trust the signature.
   Rebuilding it on load would need an async ``load`` and a way to re-record when
   the object is gone; neither exists.
4. **Keep payloads small.** Over 256 KiB the result is simply not stored. Store a
   reference (object key, job id) to anything large.
5. **A hit returns ``load(stored)``, a miss returns ``fn``'s own object.** With no
   ``dump``/``load`` the value must survive a JSON round trip unchanged (tuples
   come back as lists, non-str dict keys as strings). Pass both for pydantic
   models.
6. **Only wrap calls that are safe to SKIP.** A sub-step whose side effect later
   code depends on in-process (a populated cache, an open handle) is not skipped
   safely.
7. **Not for in-flight work.** The ledger records FINISHED sub-steps. A submitted
   job that is still running at pod death needs a re-attach handle written BEFORE
   the wait — that is ``design_job_stash`` and the compile idempotency key, not
   this. (Wrapping "submit" alone, storing the job id, and polling outside the
   wrapper composes fine.)
8. **No in-process dedupe.** Two concurrent ``once`` calls with the same key both
   call ``fn``; the last write wins. Give concurrent siblings distinct keys.
9. **Rows live as long as the run** (``ON DELETE CASCADE``) and are NOT cleared
   when the stage completes — only when a user restamp sends the stage back to
   ``pending``. See "Key scope" below for why that is safe.

Key scope — why ``run_id`` alone is the epoch
---------------------------------------------
A row must be REUSED when the same stage execution is re-entered, and must NOT be
when a stage that failed is retried (replaying the stored replies would fail it
identically, forever). The stage row's two counters do not draw that line; the
run id does. Verified against the code, 2026-10-06:

==========================================  ==========  ===============  ==========
re-execution                                 attempts    interruptions    run_id
==========================================  ==========  ===============  ==========
(a) hard pod death, reconciler re-dispatch   unchanged   unchanged        same
(b) graceful drain / pause / cancel          unchanged   +1               same
(b) resumable timeout                        +1          unchanged        same
(c) stage FAILED, then retried               unchanged   unchanged        **new**
==========================================  ==========  ===============  ==========

* (a) runs no handler code at all — the row is simply left ``running``.
* (b) ``DAGExecutor._run_one``: ``except CancelledError`` → ``note_interrupted``
  (``increment_interruptions``); ``except _RESUMABLE_TIMEOUT_EXC`` →
  ``bump_attempts`` (``increment_attempts``). Both then ``mark_pending``.
* (c) any other exception → ``mark_failed`` on the stage row (terminal,
  write-once: every ``mark_*`` writer is guarded ``status IN ('pending',
  'running')``) and ``RunRepository.mark_failed`` on the run (terminal:
  ``mark_started`` is guarded the same way). The only two ``dispatch()`` callers
  are the reconciler (``running`` orphans) and ``resume_run`` (``paused``), so a
  failed stage is NEVER re-executed inside its run. The retry is
  ``POST /runs/{id}/regenerate`` — a CHILD run, a new ``run_id``, an empty ledger.

So NEITHER counter moves only on (c) — ``attempts`` moves on a (b) that must hit,
and nothing on the stage row moves on (c) at all — and scoping rows by one would
make a resumable timeout re-pay every finished sub-step for no gain. Do not add
one. If a same-run "retry this failed stage" path is ever introduced, it must
bump a dedicated epoch and that epoch must join the key here.

The one USER-initiated same-run re-execution clears its rows. ``POST
…/elements/{id}/restamp`` (``_reset_downstream_stages`` in
``src/api/routes/runs/stages.py``) resets every DOWNSTREAM stage row on a paused run,
and ``resume_run`` then re-runs them under the same ``run_id``. The user asked for
those stages to be done again, so the route deletes their ledger rows
(``StageStepRepository.delete_for_stage``) in the same transaction as the reset —
a re-run must not depend on every downstream ``input_hash`` happening to cover the
restamped content. The restamped stage itself stays ``completed`` and keeps its rows.

Narrow residue: the ledger is not the only memory. video_production's idempotency
cache still answers the BASE key + an identical body with the ORIGINAL job id for
24 h, so a ``site_capture`` part re-run after a restamp gets its old recording back
— and if that job had failed, the part soft-fails instead of being recorded again.

It is the only such path (verified 2026-10-06): the sole other writer of
``status='pending'`` onto an existing stage row is
``StageOutputRepository.mark_pending``, guarded ``status == 'running'`` and called
only from the dispatcher's drain/timeout reset — case (b), which must keep its rows.
Every other user "redo" (regenerate, reorder-clips, apply-edits, renarrate-clip,
regenerate-clip) spawns a CHILD run. ``tests/api/routes/
test_restamp_clears_step_ledger.py`` pins both halves: the delete, and the closed
set of pending-reset writers. A NEW path that un-finishes a stage inside its run
must clear that stage's rows too, or that test fails.

Vocabulary: this is a "step ledger". "Checkpoint" means a user pause gate in this
codebase (``runs.pause_after_stages``) — never use it for this.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import logging
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from decimal import Decimal
from enum import Enum
from pathlib import PurePath
from typing import Protocol, cast
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.repositories.stage_step_repository import StageStepRepository
from src.database.session import bind_session_tenant

log = logging.getLogger(__name__)

type JsonValue = bool | int | float | str | list[JsonValue] | dict[str, JsonValue] | None

SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]

#: Largest result payload the ledger will store, measured as UTF-8 bytes of compact
#: JSON. Over this, ``once`` returns the result WITHOUT storing it. The ledger is for
#: answers and references, not blobs; a wide JSONB row is also TOASTed and re-read
#: on every re-entry.
MAX_RESULT_BYTES = 256 * 1024

#: Key of the envelope every stored payload is wrapped in — ``{"value": <dumped>}``.
#: It keeps a sub-step that legitimately returned ``None`` distinguishable from "no
#: row" and keeps the NOT NULL column independent of JSON-null vs SQL-NULL handling.
_ENVELOPE_KEY = "value"


# ── hashing ──────────────────────────────────────────────────────────────────


def _canonical(value: object) -> object:
    """``json.dumps`` ``default=`` hook: a STABLE JSON form for common non-JSON types.

    Raises ``TypeError`` for anything else — deliberately no ``repr()`` fallback: a
    default ``repr`` embeds a memory address, which would make the hash differ on
    every re-entry and turn the ledger into a silent no-op.
    """
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, UUID | PurePath | Decimal):
        return str(value)
    if isinstance(value, datetime.datetime | datetime.date | datetime.time):
        return value.isoformat()
    if isinstance(value, bytes | bytearray | memoryview):
        return {"__bytes_sha256__": hashlib.sha256(bytes(value)).hexdigest()}
    if isinstance(value, set | frozenset):
        return sorted(
            (json.loads(_canonical_json(item)) for item in value),
            key=_canonical_json,
        )
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(
        f"hash_inputs: cannot canonicalize {type(value).__name__!r}; pass JSON-able "
        "values, pydantic models, dataclasses, enums, UUIDs, paths, datetimes, bytes or sets"
    )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=_canonical,
    )


def hash_inputs(*parts: object) -> str:
    """Stable sha256 hex digest over the canonical JSON of ``parts``.

    Order-sensitive across ``parts`` (``hash_inputs(a, b) != hash_inputs(b, a)``),
    order-INsensitive within dicts and sets. Accepts JSON values plus pydantic
    models, dataclasses, enums, UUIDs, paths, ``Decimal``, datetimes, bytes (hashed)
    and sets. Raises ``TypeError`` on anything else and ``ValueError`` on NaN/inf —
    at the call site, before any paid call, so an unhashable input is a loud coding
    error rather than a ledger that never hits.

    Identical across processes and Python versions for the same inputs (no reliance
    on ``hash()`` / ``PYTHONHASHSEED``).
    """
    return hashlib.sha256(_canonical_json(list(parts)).encode("utf-8")).hexdigest()


# ── storage seam ─────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True, slots=True)
class StoredStep:
    """What a store hands back for one sub-step: its hash and the raw stored payload."""

    input_hash: str
    result: object


class StepStore(Protocol):
    """Where ledger rows live. Production: :class:`SessionStepStore`. Tests: a dict."""

    async def get(self, run_id: UUID, stage_name: str, step_key: str) -> StoredStep | None: ...

    async def put(
        self, run_id: UUID, stage_name: str, step_key: str, input_hash: str, result: JsonValue
    ) -> None: ...


class SessionStepStore:
    """:class:`StepStore` over ``stage_step_results``.

    One short session per operation — a stage runs for minutes and must not pin a
    pooled connection (or hold a transaction open) between sub-steps. The tenant is
    bound the same way the sibling stage bookkeeping (``design_job_stash``) binds
    it: ``bind_session_tenant(session, tenant_id)`` before the first statement. No
    ``SET row_security``, no bypass.
    """

    def __init__(self, sessions: SessionFactory, tenant_id: UUID | None) -> None:
        self._sessions = sessions
        self._tenant_id = tenant_id

    async def get(self, run_id: UUID, stage_name: str, step_key: str) -> StoredStep | None:
        async with self._sessions() as session:
            await bind_session_tenant(session, self._tenant_id)
            row = await StageStepRepository(session).get(run_id, stage_name, step_key)
            if row is None:
                return None
            return StoredStep(input_hash=row.input_hash, result=row.result)

    async def put(
        self, run_id: UUID, stage_name: str, step_key: str, input_hash: str, result: JsonValue
    ) -> None:
        async with self._sessions() as session:
            await bind_session_tenant(session, self._tenant_id)
            await StageStepRepository(session).put(run_id, stage_name, step_key, input_hash, result)
            await session.commit()


# ── the ledger ───────────────────────────────────────────────────────────────


class StepLedger:
    """The step ledger of ONE stage of ONE run. Obtain it with ``ctx.step_ledger(name)``.

    Cheap to construct and stateless — everything durable is in the store — so a
    fresh instance on every (re-)entry is the expected use.
    """

    def __init__(self, store: StepStore, run_id: UUID, stage_name: str) -> None:
        self._store = store
        self._run_id = run_id
        #: ``None`` only on a :class:`NullStepLedger`.
        self.run_id: UUID | None = run_id
        self.stage_name = stage_name

    async def once[T](
        self,
        step_key: str,
        input_hash: str,
        fn: Callable[[], Awaitable[T]],
        *,
        dump: Callable[[T], JsonValue] | None = None,
        load: Callable[[JsonValue], T] | None = None,
    ) -> T:
        """Return the stored result of this sub-step, or run ``fn`` and store it.

        ``dump``/``load`` default to identity (``None``); see the module docstring
        for the full contract and the caller rules.
        """
        outcome = "miss"
        stored = await self._read(step_key)
        if stored is not None:
            if stored.input_hash != input_hash:
                outcome = "stale"
            else:
                try:
                    value = self._unwrap(stored.result, load)
                except Exception:  # an unreadable row must not fail the stage
                    outcome = "unreadable"
                    log.warning(
                        "step_ledger.unreadable run_id=%s stage=%s step=%s — stored result "
                        "could not be loaded; running the step again",
                        self.run_id,
                        self.stage_name,
                        step_key,
                        exc_info=True,
                    )
                else:
                    log.info(
                        "step_ledger.hit run_id=%s stage=%s step=%s",
                        self.run_id,
                        self.stage_name,
                        step_key,
                    )
                    return value
        if outcome != "unreadable":
            log.info(
                "step_ledger.%s run_id=%s stage=%s step=%s",
                outcome,
                self.run_id,
                self.stage_name,
                step_key,
            )
        result = await fn()
        await self._write(step_key, input_hash, result, dump)
        return result

    @staticmethod
    def _unwrap[T](raw: object, load: Callable[[JsonValue], T] | None) -> T:
        if not isinstance(raw, dict) or _ENVELOPE_KEY not in raw:
            raise TypeError("stored step result is not a step-ledger envelope")
        payload = cast("JsonValue", raw[_ENVELOPE_KEY])
        return cast("T", payload) if load is None else load(payload)

    async def _read(self, step_key: str) -> StoredStep | None:
        try:
            return await self._store.get(self._run_id, self.stage_name, step_key)
        except Exception:  # a ledger read must never fail the stage
            log.warning(
                "step_ledger.read_failed run_id=%s stage=%s step=%s — running the step "
                "as if nothing were stored",
                self.run_id,
                self.stage_name,
                step_key,
                exc_info=True,
            )
            return None

    async def _write[T](
        self,
        step_key: str,
        input_hash: str,
        result: T,
        dump: Callable[[T], JsonValue] | None,
    ) -> None:
        try:
            payload = cast("JsonValue", result) if dump is None else dump(result)
            # allow_nan=False: NaN/Infinity are not JSON and jsonb rejects them.
            size = len(
                json.dumps(
                    payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
                ).encode("utf-8")
            )
        except Exception:  # a result the ledger cannot serialize must not fail the stage
            log.warning(
                "step_ledger.dump_failed run_id=%s stage=%s step=%s — result NOT stored; "
                "a re-entry will repeat this step",
                self.run_id,
                self.stage_name,
                step_key,
                exc_info=True,
            )
            return
        if size > MAX_RESULT_BYTES:
            log.warning(
                "step_ledger.oversize run_id=%s stage=%s step=%s bytes=%d cap=%d — result "
                "NOT stored; a re-entry will repeat this step (store a reference instead)",
                self.run_id,
                self.stage_name,
                step_key,
                size,
                MAX_RESULT_BYTES,
            )
            return
        try:
            await self._store.put(
                self._run_id, self.stage_name, step_key, input_hash, {_ENVELOPE_KEY: payload}
            )
        except Exception:  # a ledger write must never fail the stage
            log.warning(
                "step_ledger.write_failed run_id=%s stage=%s step=%s — result NOT stored; "
                "a re-entry will repeat this step",
                self.run_id,
                self.stage_name,
                step_key,
                exc_info=True,
            )


class NullStepLedger(StepLedger):
    """A ledger with no store: ``once`` always calls ``fn`` and records nothing.

    What ``ctx.step_ledger(...)`` returns when the context has no DB-backed run
    behind it (unit tests, dry runs, hand-built contexts), so handler code never
    branches on "is there a ledger".
    """

    def __init__(self, stage_name: str = "") -> None:  # no store, no run: skip super()
        self.run_id = None
        self.stage_name = stage_name

    async def once[T](
        self,
        step_key: str,
        input_hash: str,
        fn: Callable[[], Awaitable[T]],
        *,
        dump: Callable[[T], JsonValue] | None = None,
        load: Callable[[JsonValue], T] | None = None,
    ) -> T:
        return await fn()


StepLedgerFactory = Callable[[str], StepLedger]


def step_ledger_factory(
    sessions: SessionFactory, *, run_id: UUID, tenant_id: UUID | None
) -> StepLedgerFactory:
    """The ``RunContext.step_ledger_factory`` for a DB-backed run (dispatcher wiring).

    Returns ``stage_name -> StepLedger`` bound to this run and tenant over one shared
    :class:`SessionStepStore`.
    """
    store = SessionStepStore(sessions, tenant_id)

    def _for_stage(stage_name: str) -> StepLedger:
        return StepLedger(store, run_id, stage_name)

    return _for_stage
