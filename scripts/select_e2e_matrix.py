#!/usr/bin/env python3
"""Choose which E2E scenarios the nightly runs, so no scenario can sit unexecuted.

WHY THIS EXISTS — a measured gap, not a hypothetical.

``matrix.json`` carries 34 scenarios. Six are ``nightly: true``. The other 28 are
"dispatch-only", which in practice means **never**: a workflow that only fires on
``workflow_dispatch`` runs when a human remembers it, and nobody remembers 28 of them.
That is how a spec sits on main for days, lint-clean and compile-clean, having never
executed a single assertion — see memory ``feedback_green_e2e_that_never_ran``. Three
customer-startable blueprints (``showcase_v1``, ``web_design_v1``,
``marketing_scorecard_v1``) have no scenario at all; the ones that DO have a scenario
deserve better than a file nobody runs.

WHY NOT JUST RUN ALL 34 NIGHTLY

Because that was tried and reverted. The cron was disabled entirely on 2026-07-16 after
the full fan-out saturated pgbouncer and produced noisy reds, and re-enabled on
2026-07-22 only once it was narrowed to the curated six. Restoring the full fan-out
would re-buy a regression that has already been paid for once.

THE SHAPE: CORE + A DETERMINISTIC ROTATING SLICE

Every nightly runs the curated core (unchanged) **plus** ``slice_size`` scenarios taken
from the rest, chosen by day-of-year. The window advances by ``slice_size`` each day, so
with 28 pooled scenarios and a slice of 3 every scenario runs at least once every ~10
days, and the per-night job count goes 6 → 9 rather than 6 → 34.

Deterministic on purpose: given the date you can say exactly which scenarios ran, and a
re-run of the same night reproduces the same set. Nothing is sampled at random.

OPTING OUT. A scenario may set ``"rotation": false`` in ``matrix.json`` to stay out of
the pool. That is for a scenario that is genuinely not runnable unattended (needs a
manual fixture, costs a fortune), **not** for one that is merely failing — a failing
scenario is the signal this whole mechanism exists to surface. Anything opted out must
carry a ``rotation_reason``.

NON-VACUITY. A selector that returns nothing is a bug, not a quiet pass: the nightly
would go green having run zero jobs, which is the ``skip and continue`` → silent-green
failure class. So the schedule path asserts (a) the result is non-empty, and (b) every
``nightly: true`` scenario is present. Both raise rather than warn.

EXPECTED CONSEQUENCE, STATED UP FRONT. The 28 pooled scenarios have not run in months.
The first rotation cycles will very likely go RED, and that is the mechanism working:
each red is a scenario that was already broken and was not telling anyone. Fix or
``rotation: false`` + reason — do not silence the selector.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict


class Scenario(TypedDict, total=False):
    """One row of ``matrix.json``. Only the keys this selector reads — plus the two
    the matrix invariant test asserts on — are declared; the rows are passed through
    to the workflow verbatim, so unknown keys survive."""

    label: str
    scenario: str
    nightly: bool
    rotation: bool
    rotation_reason: str
    # Not read by the selector; declared so the deadline-vs-GitHub-timeout invariant
    # in tests/test_e2e_matrix_selection.py can type-check what it reads.
    deadline: int
    timeout_minutes: int


# How many pooled scenarios ride along with the curated core on each nightly. 3 puts the
# 28-scenario pool on a ~10-day cycle at a per-night cost of 9 jobs against the matrix's
# `max-parallel: 4` — i.e. roughly one extra wave, not a fan-out.
DEFAULT_SLICE_SIZE = 3


class EmptySelectionError(RuntimeError):
    """The selector produced no jobs. A nightly that runs zero scenarios reports
    success, which is indistinguishable from a nightly that ran and passed."""


class MissingCoreScenarioError(RuntimeError):
    """A ``nightly: true`` scenario was dropped from a scheduled selection. The
    rotation is additive; it must never displace the curated core."""


def assert_core_preserved(core_labels: set[str], selected_labels: set[str]) -> None:
    """The rotation is ADDITIVE — it may never displace a curated nightly scenario.

    Split out as its own function on purpose. Correct ``select`` code cannot reach the
    violating state, so a test driving ``select`` can only ever assert the happy path and
    would prove nothing about the guard. Calling this directly with a fabricated
    ``selected_labels`` is how the guard is watched FIRING, which is the only way it counts
    as a gate rather than an assumption."""
    missing = core_labels - selected_labels
    if missing:
        raise MissingCoreScenarioError(
            f"rotation dropped curated nightly scenarios: {sorted(missing)}"
        )


def _is_core(row: Scenario) -> bool:
    return bool(row.get("nightly"))


def _in_pool(row: Scenario) -> bool:
    """Pooled = not core, and not explicitly opted out."""
    return not _is_core(row) and row.get("rotation", True) is not False


def rotating_slice(pool: Sequence[Scenario], day_of_year: int, slice_size: int) -> list[Scenario]:
    """``slice_size`` consecutive pool entries, starting at a window that advances by
    ``slice_size`` per day. Wraps, so the pool is covered in ``ceil(len/slice)`` days."""
    if not pool or slice_size <= 0:
        return []
    size = min(slice_size, len(pool))
    start = (day_of_year * slice_size) % len(pool)
    return [pool[(start + offset) % len(pool)] for offset in range(size)]


def select(
    rows: Sequence[Scenario],
    *,
    event_name: str,
    label: str = "",
    day_of_year: int = 1,
    slice_size: int = DEFAULT_SLICE_SIZE,
) -> list[Scenario]:
    """The scenarios this trigger should run.

    ``workflow_dispatch`` with a label runs exactly that label (a targeted proof must not
    fan out the heavy render matrix) — and a label is the ONLY way to reach a
    ``rotation: false`` row. ``schedule`` runs core + rotating slice. Every other trigger
    with no label runs every row that has not opted out with ``rotation: false``: those
    rows are paid, hand-launched proof scenarios, and an unlabelled dispatch must not buy
    them as a side effect of "run everything"."""
    if label:
        return [row for row in rows if row.get("label") == label]
    if event_name != "schedule":
        return [row for row in rows if row.get("rotation", True) is not False]

    core = [row for row in rows if _is_core(row)]
    pool = [row for row in rows if _in_pool(row)]
    selected = core + rotating_slice(pool, day_of_year, slice_size)

    # De-duplicate by label while preserving order — a pool shorter than the slice can
    # otherwise repeat a row, which would schedule the same scenario twice in one night.
    seen: set[str] = set()
    deduped: list[Scenario] = []
    for row in selected:
        key = str(row.get("label", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    if not deduped:
        raise EmptySelectionError(
            "scheduled selection is empty — the nightly would run zero scenarios and report success"
        )
    assert_core_preserved({str(row.get("label", "")) for row in core}, seen)
    return deduped


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--event-name", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--day-of-year", type=int, required=True)
    parser.add_argument("--slice-size", type=int, default=DEFAULT_SLICE_SIZE)
    args = parser.parse_args(argv)

    rows: list[Scenario] = json.loads(args.matrix.read_text())
    chosen = select(
        rows,
        event_name=args.event_name,
        label=args.label,
        day_of_year=args.day_of_year,
        slice_size=args.slice_size,
    )
    print(json.dumps(chosen, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
