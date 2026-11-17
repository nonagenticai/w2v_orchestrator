"""The run-deadline anchor: where the 2h budget is measured FROM.

These pin the helper directly. The wiring that feeds it was wrong on first
write — it passed the ``Run`` ORM row and read ``started_at`` after the
dispatcher's session had closed, which raises ``DetachedInstanceError`` for any
column not already materialised. ``started_at`` is written by ``mark_started``'s
bare UPDATE, so it is precisely such a column. 24 tests went red on it, every
in-process DAG among them.

The anchor must be the run's FIRST slice, matching ``mark_started``'s
``COALESCE(started_at, now())`` — measuring from the latest resume would hand a
run a fresh 2h on every reconciler re-dispatch, i.e. no bound at all.
"""

from __future__ import annotations

import pytest

from src.config import get_settings
from src.services.run_dispatcher import _run_deadline_at


@pytest.fixture(autouse=True)
def _reset_settings():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _set_budget(monkeypatch: pytest.MonkeyPatch, seconds: int) -> None:
    monkeypatch.setenv("PIPELINE_RUN_DEADLINE_SECONDS", str(seconds))
    get_settings.cache_clear()


def test_helper_takes_a_value_not_an_orm_row() -> None:
    """Guard the regression directly: a row-like object must NOT be accepted.

    The first implementation took ``Run`` and reached for ``.started_at`` outside
    the session. Passing an object here would raise ``TypeError`` on the
    arithmetic rather than silently working, but the contract is stated in the
    signature so a future edit cannot quietly reintroduce the coupling.
    """
    import inspect

    param, paused = inspect.signature(_run_deadline_at).parameters.values()
    assert param.name == "started_at"
    assert "datetime" in str(param.annotation)
    # The pause credit (migration 0094) is a plain int value too, never the row.
    assert paused.name == "paused_seconds" and "int" in str(paused.annotation)
