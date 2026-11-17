"""Name the phase a stage is currently inside, so a timeout can say WHERE it died.

The defect this exists for: ``CompileStageBase.handle`` does ~120 lines of asset
work (title cards, two re-sign passes, ken burns, sound) BEFORE it reaches
``stage_call``'s guarded region. On 2026-09-03/04 that block burned 2500 s three
times over — 126 minutes — and produced a zero-argument ``TimeoutError``: one log
line, zero DB writes, and ``failure_class='unknown'``. The block was dark, so the
outage was diagnosed four times and misattributed three of them (a compiler
rollout, the compiler itself, an E2E budget) before anyone reached the real cause,
which was a leaked recording slot in a DOWNSTREAM service.

The lesson encoded here is `w2v-fail-open-on-a-measurement-gap`: the gap must be
made EXPRESSIBLE before it can be made loud. A phase name is that expression.

Two consumers, deliberately decoupled through a ContextVar rather than a
parameter, because the writer (a stage) and the reader (an outer middleware) are
separated by the whole middleware stack and cannot pass anything to each other:

* writer — ``async with phase("ken_burns"):`` inside a stage's ``handle``
* reader — ``current_phase()`` in ``TimeoutMiddleware``, to name the phase in the
  message it raises

ContextVar semantics are what make this correct under concurrency: ``asyncio``
copies the context per task, so two stages running concurrently each see their own
phase, and a nested phase restores its parent on exit via the reset token.

A THIRD, opt-in consumer added alongside the ``lesson_compile`` regression
investigation (2026-09): ``reset_phase_totals`` / ``phase_totals`` let a caller
accumulate each named phase's elapsed seconds for the current task, so a stage
can persist WHERE its wall-clock went instead of that information dying in logs
a pruner deletes. Same ContextVar-per-task isolation as ``current_phase`` — a
``reset_phase_totals()`` call only affects the current task's own copied
context, so concurrently-running stages (or child tasks spawned via
``asyncio.gather``/``TaskGroup``) never clobber each other's totals. A caller
that never calls ``reset_phase_totals()`` gets an always-empty dict back from
``phase_totals()`` — accumulation is opt-in and adds zero cost to every other
``phase()`` user.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar

from w2v_common.stages import StageTimeoutError

logger = logging.getLogger(__name__)


class PhaseDeadlineError(StageTimeoutError, TimeoutError):
    """An aggregate phase deadline expired — messaged, phase-named, attributable.

    Raised instead of a bare ``TimeoutError`` so the failure survives
    ``attribute_exception`` (``src/models/failure_class.py``) as something other
    than ``unknown``. That function walks the cause chain for an explicitly
    stamped ``failure_class`` and otherwise falls through to ``unknown`` — its
    docstring rules out prose matching and any third pass — so an unstamped
    timeout is, by construction, unattributable. ``external`` is the honest
    class: a block that overran its budget waiting on downstream recording work
    is not a defect in this repo's code.

    Subclasses ``StageTimeoutError`` so ``dag_executor._RESUMABLE_TIMEOUT_EXC``
    still treats it as a resumable timeout, and the builtin ``TimeoutError`` so
    existing ``except TimeoutError`` handlers keep working.
    """

    failure_class = "external"


# The phase currently executing in THIS task's context, or None outside any phase.
_current_phase: ContextVar[str | None] = ContextVar("current_phase", default=None)

# Accumulated elapsed-seconds per phase name for THIS task's context, or None
# when no caller has opted in via `reset_phase_totals()`. `None` (not `{}`) is
# the default so `phase()` can tell "nobody is recording" apart from "recording,
# nothing has completed yet" without an extra ContextVar.
_phase_totals: ContextVar[dict[str, float] | None] = ContextVar("phase_totals", default=None)


def current_phase() -> str | None:
    """The innermost phase name for this task, or ``None`` when outside a phase.

    Read by ``TimeoutMiddleware`` when it converts an anonymous ``TimeoutError``
    into a messaged, attributable one. Never raises: a missing phase is reported
    as ``None``, which the caller renders as "unknown phase" rather than failing
    the failure path.
    """
    return _current_phase.get()


def reset_phase_totals() -> None:
    """Start (or restart) accumulating per-phase elapsed seconds for this task.

    Call once before the sequence of ``phase(...)`` blocks whose timings you
    want to read back later via ``phase_totals()``. Safe to call again (e.g. on
    a retry) — it simply replaces the accumulator with a fresh empty dict for
    the CURRENT task's context; a concurrently-running task (its own copied
    context) is unaffected, matching ``current_phase``'s isolation.
    """
    _phase_totals.set({})


def phase_totals() -> dict[str, float]:
    """Snapshot of elapsed seconds per phase name accumulated since the last
    ``reset_phase_totals()`` call in this task's context.

    Returns ``{}`` if ``reset_phase_totals()`` was never called — recording is
    opt-in, so a caller that does not ask for it pays nothing and gets nothing.
    A copy is returned so the caller's dict cannot be mutated by a later phase.
    """
    totals = _phase_totals.get()
    return dict(totals) if totals is not None else {}


@asynccontextmanager
async def phase(name: str) -> AsyncIterator[None]:
    """Mark the enclosed block as phase ``name``, logging entry and exit.

    Emits a line on entry and a line on exit carrying the elapsed seconds — on the
    error path too, so a phase that dies is the LAST thing in the log rather than
    silently absent. That asymmetry is the whole point: before this, the 126-minute
    stall produced exactly one log line and it was about something else.
    """
    token = _current_phase.set(name)
    started = time.monotonic()
    logger.info("phase.start name=%s", name)
    try:
        yield
    except BaseException:
        # Deliberately BaseException: a phase killed by asyncio.CancelledError or a
        # TimeoutError from an outer deadline is EXACTLY the case this log line
        # exists for, and neither is an Exception subclass in 3.11+.
        logger.warning("phase.failed name=%s elapsed=%.1fs", name, time.monotonic() - started)
        raise
    else:
        elapsed = time.monotonic() - started
        logger.info("phase.done name=%s elapsed=%.1fs", name, elapsed)
        # Accumulate on the SUCCESS path only, mirroring `phase.done` above: a
        # phase that raised has no meaningful elapsed-to-completion, and its
        # caller (see `_run_presubmit`'s `except TimeoutError`) already re-raises
        # rather than building an output, so there is nowhere to persist it
        # anyway. `+=` rather than `=` because a phase name can legitimately run
        # more than once per recording window (none does today, but nothing here
        # assumes otherwise).
        totals = _phase_totals.get()
        if totals is not None:
            totals[name] = totals.get(name, 0.0) + elapsed
    finally:
        _current_phase.reset(token)
