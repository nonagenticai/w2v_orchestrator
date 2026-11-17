"""A user restamp un-finishes downstream stages — their step-ledger rows must go.

The step ledger (``src/services/stages/shared/step_ledger.py``) reuses a stage's
finished sub-steps when the SAME execution is re-entered after a pod death. A
restamp is the opposite request: the user edited an upstream element and wants the
downstream stages done again, under the same ``run_id``. A stored sub-step whose
input hash does not happen to cover the edited content would otherwise be replayed
into the "re-run".

Two halves:

* behaviour — the restamp route deletes exactly the downstream stages' rows, in
  the transaction that resets them;
* a closed-set guard — the restamp is the ONLY code that sends a finished stage
  back to ``pending`` inside a run. If another appears, it must clear the ledger
  too, and this file is where that is decided.
"""

from __future__ import annotations

import ast
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import text

from src.database.repositories.stage_step_repository import StageStepRepository
from tests.api.routes.test_runs import _make_project, _seed_run_with_full_dag, _stop_run

_SRC = Path(__file__).resolve().parents[3] / "src"

_STAGES = ("brief_intake", "script_draft", "storyboard_draft", "video_render")


async def _ledger_keys(session, run_id: UUID) -> set[tuple[str, str]]:
    rows = await session.execute(
        text("SELECT stage_name, step_key FROM stage_step_results WHERE run_id = :r"),
        {"r": run_id},
    )
    return {(stage, key) for stage, key in rows}


async def _seed_ledger(session, run_id: UUID) -> None:
    repo = StageStepRepository(session)
    for stage in _STAGES:
        await repo.put(run_id, stage, "draw:1", "h-1", {"value": f"{stage}-1"})
        await repo.put(run_id, stage, "draw:2", "h-2", {"value": f"{stage}-2"})
    await session.commit()


# ── behaviour ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_restamp_deletes_the_ledger_rows_of_every_downstream_stage(
    client, db_session
) -> None:
    _, deliv = await _make_project(client)
    rid = UUID(await _seed_run_with_full_dag(db_session, deliv, [{"id": "L1", "text": "old"}]))
    await _stop_run(db_session, rid)
    other = UUID(await _seed_run_with_full_dag(db_session, deliv, [{"id": "L1", "text": "old"}]))
    await _stop_run(db_session, other)
    await _seed_ledger(db_session, rid)
    await _seed_ledger(db_session, other)

    r = await client.post(
        f"/api/runs/{rid}/stages/script_draft/elements/L1/restamp",
        json={"value": {"text": "new"}},
    )
    assert r.status_code == 200, r.text

    # Downstream of script_draft (storyboard_draft, video_render) is forgotten; the
    # restamped stage and everything upstream stay completed and keep their rows.
    assert await _ledger_keys(db_session, rid) == {
        ("brief_intake", "draw:1"),
        ("brief_intake", "draw:2"),
        ("script_draft", "draw:1"),
        ("script_draft", "draw:2"),
    }
    # Another run of the same tenant is untouched: the delete is keyed by run.
    assert len(await _ledger_keys(db_session, other)) == len(_STAGES) * 2


@pytest.mark.asyncio
async def test_a_refused_restamp_deletes_nothing(client, db_session) -> None:
    """An unknown element 404s before the reset — and before the ledger delete."""
    _, deliv = await _make_project(client)
    rid = UUID(await _seed_run_with_full_dag(db_session, deliv, [{"id": "L1", "text": "old"}]))
    await _stop_run(db_session, rid)
    await _seed_ledger(db_session, rid)

    r = await client.post(
        f"/api/runs/{rid}/stages/script_draft/elements/NOPE/restamp",
        json={"value": {"text": "new"}},
    )
    assert r.status_code == 404, r.text
    assert len(await _ledger_keys(db_session, rid)) == len(_STAGES) * 2


@pytest.mark.asyncio
async def test_another_tenants_restamp_cannot_delete_ledger_rows(
    client, other_tenant_client, db_session
) -> None:
    _, deliv = await _make_project(client)
    rid = UUID(await _seed_run_with_full_dag(db_session, deliv, [{"id": "L1", "text": "old"}]))
    await _seed_ledger(db_session, rid)

    r = await other_tenant_client.post(
        f"/api/runs/{rid}/stages/script_draft/elements/L1/restamp",
        json={"value": {"text": "new"}},
    )
    assert r.status_code == 404, r.text
    assert len(await _ledger_keys(db_session, rid)) == len(_STAGES) * 2


@pytest.mark.asyncio
async def test_delete_for_stage_is_scoped_to_one_stage_of_one_run(client, db_session) -> None:
    _, deliv = await _make_project(client)
    rid = UUID(await _seed_run_with_full_dag(db_session, deliv, [{"id": "L1", "text": "old"}]))
    await _seed_ledger(db_session, rid)

    deleted = await StageStepRepository(db_session).delete_for_stage(rid, "video_render")
    await db_session.commit()

    assert deleted == 2
    remaining = await _ledger_keys(db_session, rid)
    assert {stage for stage, _ in remaining} == set(_STAGES) - {"video_render"}
    assert await StageStepRepository(db_session).delete_for_stage(rid, "video_render") == 0


# ── closed set of "back to pending" writers ──────────────────────────────────


def _root_call(node: ast.expr) -> ast.Call | None:
    """The innermost call of a fluent chain: ``update(X).where(..).values(..)`` → ``update(X)``."""
    found: ast.Call | None = None
    current: ast.expr | None = node
    while current is not None:
        if isinstance(current, ast.Call):
            found = current
            current = current.func
        elif isinstance(current, ast.Attribute):
            current = current.value
        else:
            current = None
    return found


def _is_stage_output_update(call: ast.Call | None) -> bool:
    return (
        call is not None
        and isinstance(call.func, ast.Name)
        and call.func.id == "update"
        and len(call.args) == 1
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == "StageOutput"
    )


def _sets_status_pending(call: ast.Call) -> bool:
    return (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "values"
        and any(
            kw.arg == "status"
            and isinstance(kw.value, ast.Constant)
            and kw.value.value == "pending"
            for kw in call.keywords
        )
    )


def _enclosing_functions(tree: ast.AST) -> dict[int, str]:
    """``id(node) -> name of the innermost function containing it``."""
    owner: dict[int, str] = {}

    def visit(node: ast.AST, current: str) -> None:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            current = node.name
        owner[id(node)] = current
        for child in ast.iter_child_nodes(node):
            visit(child, current)

    visit(tree, "<module>")
    return owner


def _python_sources() -> list[Path]:
    return [p for p in sorted(_SRC.rglob("*.py")) if "migrations/versions" not in p.as_posix()]


def _pending_reset_writers() -> set[tuple[str, str]]:
    """Every ``update(StageOutput)…values(status="pending")`` as ``(file, function)``."""
    writers: set[tuple[str, str]] = set()
    for path in _python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        owner = _enclosing_functions(tree)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and _sets_status_pending(node)
                and _is_stage_output_update(_root_call(node))
            ):
                writers.add((path.relative_to(_SRC).as_posix(), owner[id(node)]))
    return writers


def _callers_of(method: str) -> set[str]:
    callers: set[str] = set()
    for path in _python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == method
            for node in ast.walk(tree)
        ):
            callers.add(path.relative_to(_SRC).as_posix())
    return callers


def test_the_set_of_writers_that_send_a_stage_back_to_pending_is_closed() -> None:
    """Exactly two statements reset an existing stage row to ``pending``.

    * ``mark_pending`` — ``status == 'running'`` only, i.e. an interrupted execution
      (drain / pause / resumable timeout). Its ledger rows MUST survive: reusing
      them is the feature.
    * ``_reset_downstream_stages`` (the element restamp) — a user un-finishing
      downstream stages. It MUST
      clear their ledger rows (asserted below and by the behaviour tests above).

    Adding a third writer fails here on purpose. Decide which kind it is: an
    interrupted execution keeps its rows; anything that re-runs a FINISHED stage in
    the same run calls ``StageStepRepository.delete_for_stage`` in its transaction.
    Then extend this set and the "Key scope" section of the step-ledger docstring.
    """
    assert _pending_reset_writers() == {
        ("database/repositories/stage_output_repository.py", "mark_pending"),
        ("api/routes/runs/stages.py", "_reset_downstream_stages"),
    }


def test_no_raw_sql_resets_stage_rows() -> None:
    """The AST guard above only sees SQLAlchemy Core; keep raw SQL out of the picture."""
    offenders = [
        path.relative_to(_SRC).as_posix()
        for path in _python_sources()
        if "UPDATE stage_outputs" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_mark_pending_is_reached_only_from_the_dispatchers_interruption_reset() -> None:
    assert _callers_of("mark_pending") == {"services/run_dispatcher.py"}


def test_the_ledger_is_cleared_only_by_the_restamp() -> None:
    """…and nothing on the reconciler / drain / timeout side ever deletes rows."""
    assert _callers_of("delete_for_stage") == {"api/routes/runs/stages.py"}


def test_mark_pending_still_resets_running_rows_only() -> None:
    """The premise of the closed set: ``mark_pending`` cannot un-finish a stage."""
    source = (_SRC / "database/repositories/stage_output_repository.py").read_text("utf-8")
    tree = ast.parse(source)
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "mark_pending"
    )
    # Drop the docstring so prose about the guard cannot satisfy the assertion.
    body = ast.Module(body=fn.body[1:], type_ignores=[])
    assert "StageOutput.status == 'running'" in ast.unparse(body)
