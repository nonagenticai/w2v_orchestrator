"""Unit coverage for ``runs.run_setup.reconcile_blueprint_stage_rows``.

Stage rows are seeded once, at run creation. A stage added to a blueprint later
has no row on older runs, and every ``mark_*`` writer is a bare ``UPDATE`` — so
it silently no-ops and ``decide_run_terminal_state``'s set-equality against
``topological_order()`` can never hold. The helper seeds those rows.

Pinned here:
* nothing missing → the rows are returned unchanged and the repo is NOT touched
  (the zero-extra-query guarantee for the overwhelmingly common case).
* a missing stage → ``insert_pending_bulk`` called exactly once with exactly the
  missing names, then the rows are re-read.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import cast
from uuid import UUID, uuid7

import pytest

from src.database.models import StageOutput
from src.database.repositories.stage_output_repository import StageOutputRepository
from src.services.blueprints.base import DAGBlueprint

pytestmark = pytest.mark.asyncio


class _StubStageRepo(StageOutputRepository):
    """Records calls; ``list_for_run`` returns the post-insert row set.

    Subclasses the real repository so the helper's signature is exercised as
    written — a structural stub would let a signature drift pass unnoticed.
    """

    def __init__(self, rows_after_insert: list[StageOutput]) -> None:
        self._rows_after_insert = rows_after_insert
        self.insert_calls: list[tuple[UUID, list[str]]] = []
        self.insert_skipped_calls: list[tuple[UUID, list[tuple[str, str]]]] = []
        self.list_for_run_calls = 0

    async def insert_pending_bulk(self, run_id: UUID, stage_names: Sequence[str]) -> None:
        self.insert_calls.append((run_id, list(stage_names)))

    async def insert_skipped_bulk(self, run_id: UUID, entries: Sequence[tuple[str, str]]) -> None:
        self.insert_skipped_calls.append((run_id, list(entries)))

    async def list_for_run(self, run_id: UUID) -> list[StageOutput]:
        self.list_for_run_calls += 1
        return list(self._rows_after_insert)


def _row(name: str, status: str = "completed") -> StageOutput:
    return StageOutput(id=uuid7(), run_id=uuid7(), stage_name=name, status=status)


def _blueprint(
    order: list[str], depends_on: dict[str, tuple[str, ...]] | None = None
) -> DAGBlueprint:
    """Stub with no real dependency edges unless ``depends_on`` says otherwise —
    so ``descendants`` returns empty for every stage by default, matching these
    tests' existing (dependency-free) fixtures exactly."""
    edges = depends_on or {}

    def descendants(name: str) -> set[str]:
        # Same transitive-forward-reachability semantics as the real
        # ``DAGBlueprint.descendants`` (dep -> dependents), computed from the
        # inverse of the ``depends_on`` map supplied here.
        adj: dict[str, list[str]] = {}
        for stage, deps in edges.items():
            for dep in deps:
                adj.setdefault(dep, []).append(stage)
        seen: set[str] = set()
        queue = list(adj.get(name, []))
        while queue:
            n = queue.pop()
            if n in seen:
                continue
            seen.add(n)
            queue.extend(adj.get(n, []))
        return seen

    return cast(
        "DAGBlueprint",
        SimpleNamespace(topological_order=lambda: list(order), descendants=descendants),
    )


async def test_no_missing_rows_issues_zero_queries() -> None:
    """Every blueprint stage already has a row → return them as-is without
    touching the repository at all. WITHOUT the zero-query guard this would
    still fire an insert + a re-read on every dispatch of every healthy run."""
    from src.services.runs.run_setup import reconcile_blueprint_stage_rows

    run_id = uuid7()
    rows = [_row("s1"), _row("s2", "pending")]
    repo = _StubStageRepo(rows_after_insert=[])

    result = await reconcile_blueprint_stage_rows(
        repo,
        run_id=run_id,
        blueprint=_blueprint(["s1", "s2"]),
        stage_rows=rows,
    )

    assert result == rows
    assert repo.insert_calls == []
    assert repo.list_for_run_calls == 0


async def test_missing_row_is_inserted_once_and_rows_reread() -> None:
    """A blueprint that gained ``s3`` after the run was created → exactly the
    missing name is inserted, once, and the reconciled rows are re-read.
    WITHOUT the fix the caller keeps the two-row set, ``s3`` stays invisible to
    the completion gate, and the run never leaves ``running``."""
    from src.services.runs.run_setup import reconcile_blueprint_stage_rows

    run_id = uuid7()
    rows = [_row("s1"), _row("s2")]
    reconciled = [*rows, _row("s3", "pending")]
    repo = _StubStageRepo(rows_after_insert=reconciled)

    result = await reconcile_blueprint_stage_rows(
        repo,
        run_id=run_id,
        blueprint=_blueprint(["s1", "s2", "s3"]),
        stage_rows=rows,
    )

    assert repo.insert_calls == [(run_id, ["s3"])]
    assert repo.list_for_run_calls == 1
    assert result == reconciled


async def test_only_the_missing_names_are_inserted() -> None:
    """Pre-existing rows are never re-inserted — only the genuine gaps, in
    topological order. WITHOUT the name-diff this would re-insert every stage
    and rely on the ON CONFLICT clause to absorb it."""
    from src.services.runs.run_setup import reconcile_blueprint_stage_rows

    run_id = uuid7()
    rows = [_row("s2")]
    repo = _StubStageRepo(rows_after_insert=rows)

    await reconcile_blueprint_stage_rows(
        repo,
        run_id=run_id,
        blueprint=_blueprint(["s1", "s2", "s3"]),
        stage_rows=rows,
    )

    assert repo.insert_calls == [(run_id, ["s1", "s3"])]


# ---------------------------------------------------------------------------
# Retrofit-skew: a missing stage must not seed 'pending' when it sits UPSTREAM
# of a dependent that already left 'pending' on THIS run's frozen blueprint.
# Prod incident: PR #1035 added `website_review_slide_plan` upstream of
# `website_review_slides`; an in-flight run with `slides` already COMPLETED got
# `slide_plan` retrofitted as 'pending' and EXECUTED — orphaned LLM spend.
# ---------------------------------------------------------------------------


async def test_a_new_upstream_stage_with_completed_dependent_is_skipped_not_pending() -> None:
    """(a) the prod case: new stage upstream of a COMPLETED dependent -> 'skipped'
    with a recorded reason, never 'pending', never inserted via insert_pending_bulk
    (so it can never be dispatched/executed)."""
    from src.services.runs.run_setup import reconcile_blueprint_stage_rows

    run_id = uuid7()
    rows = [_row("slides", "completed")]
    repo = _StubStageRepo(rows_after_insert=[*rows, _row("slide_plan", "skipped")])

    result = await reconcile_blueprint_stage_rows(
        repo,
        run_id=run_id,
        blueprint=_blueprint(["slide_plan", "slides"], depends_on={"slides": ("slide_plan",)}),
        stage_rows=rows,
    )

    assert repo.insert_calls == []
    assert len(repo.insert_skipped_calls) == 1
    (skip_run_id, entries) = repo.insert_skipped_calls[0]
    assert skip_run_id == run_id
    assert [name for name, _reason in entries] == ["slide_plan"]
    _name, reason = entries[0]
    assert "slide_plan" in reason
    assert "slides" in reason
    assert repo.list_for_run_calls == 1
    assert result == repo._rows_after_insert


async def test_b_new_upstream_stage_with_running_dependent_is_skipped() -> None:
    """(b) 'started' is not just 'completed' — a RUNNING dependent must also
    block the retrofit, since its execution decision (running without the new
    stage's output) is already made."""
    from src.services.runs.run_setup import reconcile_blueprint_stage_rows

    run_id = uuid7()
    rows = [_row("slides", "running")]
    repo = _StubStageRepo(rows_after_insert=rows)

    await reconcile_blueprint_stage_rows(
        repo,
        run_id=run_id,
        blueprint=_blueprint(["slide_plan", "slides"], depends_on={"slides": ("slide_plan",)}),
        stage_rows=rows,
    )

    assert repo.insert_calls == []
    assert [name for name, _ in repo.insert_skipped_calls[0][1]] == ["slide_plan"]


async def test_c_new_downstream_only_stage_on_old_run_still_seeds_pending() -> None:
    """(c) a genuinely downstream-only new stage (nothing depends on it) must keep
    seeding 'pending' exactly as before, regardless of how far along its
    upstream/ancestor stages are — this is the ORIGINAL docstring's scenario and
    must not regress."""
    from src.services.runs.run_setup import reconcile_blueprint_stage_rows

    run_id = uuid7()
    rows = [_row("slides", "completed")]
    repo = _StubStageRepo(rows_after_insert=[*rows, _row("post_audit", "pending")])

    await reconcile_blueprint_stage_rows(
        repo,
        run_id=run_id,
        blueprint=_blueprint(["slides", "post_audit"], depends_on={"post_audit": ("slides",)}),
        stage_rows=rows,
    )

    assert repo.insert_skipped_calls == []
    assert repo.insert_calls == [(run_id, ["post_audit"])]


async def test_d_mixed_started_and_pending_dependents_skips_the_new_stage_only() -> None:
    """(d) new stage S has two direct dependents: A already COMPLETED, B still
    PENDING. S must be skipped (ANY started dependent blocks it) — and B, which
    already has its own row, must be left completely untouched: the executor's
    existing 'a skipped dependency satisfies eligibility' rule (every stage can
    raise StageSkippedError at runtime, so every downstream reader already
    tolerates it) lets B proceed exactly as if S had legitimately skipped
    itself. This function must not touch B's row at all -- it only classifies
    the MISSING stage."""
    from src.services.runs.run_setup import reconcile_blueprint_stage_rows

    run_id = uuid7()
    rows = [_row("a", "completed"), _row("b", "pending")]
    repo = _StubStageRepo(rows_after_insert=[*rows, _row("s", "skipped")])

    await reconcile_blueprint_stage_rows(
        repo,
        run_id=run_id,
        blueprint=_blueprint(["s", "a", "b"], depends_on={"a": ("s",), "b": ("s",)}),
        stage_rows=rows,
    )

    assert repo.insert_calls == []
    assert [name for name, _ in repo.insert_skipped_calls[0][1]] == ["s"]
    # b's pre-existing row was never passed to any insert path.
    for _run_id, names in repo.insert_calls:
        assert "b" not in names
    for _run_id, entries in repo.insert_skipped_calls:
        assert "b" not in [n for n, _ in entries]


async def test_e_brand_new_run_with_every_row_present_is_unaffected() -> None:
    """(e) a brand-new run dispatched against the blueprint it was created under
    has a row for every declared stage already -> zero missing, zero inserts of
    either kind, the existing zero-query fast path is preserved."""
    from src.services.runs.run_setup import reconcile_blueprint_stage_rows

    run_id = uuid7()
    rows = [_row("slide_plan", "pending"), _row("slides", "pending")]
    repo = _StubStageRepo(rows_after_insert=[])

    result = await reconcile_blueprint_stage_rows(
        repo,
        run_id=run_id,
        blueprint=_blueprint(["slide_plan", "slides"], depends_on={"slides": ("slide_plan",)}),
        stage_rows=rows,
    )

    assert result == rows
    assert repo.insert_calls == []
    assert repo.insert_skipped_calls == []
    assert repo.list_for_run_calls == 0
