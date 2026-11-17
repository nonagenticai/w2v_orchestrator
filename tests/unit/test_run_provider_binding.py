"""P3 WP3 — the dispatcher binds a run's PINNED provider slots + model tiers.

A stage deep inside a run asks ``current_run_slot("llm")`` /
``current_run_model_tier(...)``; the dispatcher binds both from the run snapshot
next to ``bind_run_attribution``. A legacy run binds "nothing pinned" so callers
fall back to live resolution / the env tier.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.auth.run_providers import (
    bind_run_provider_slots,
    clear_run_provider_slots,
    current_run_model_tier,
    current_run_slot,
)
from src.config import get_settings
from src.services.providers import SlotRef, SlotUnavailableError
from src.services.stages.demo_video.reconcile_timeline import ReconcileTimelineStage
from tests.unit.test_run_dispatcher_failure import _drive_dispatcher

CONN = uuid4()
PINS = {
    "provider_slots": {
        "llm": {"source": "tenant", "kind": "anthropic", "connection_id": str(CONN), "version": 4},
        "tts": {"source": "deployment", "kind": "platform", "connection_id": None},
    },
    "model_tier": {"timing_director": "tier-high"},
}


@pytest.fixture(autouse=True)
def _unbound() -> Iterator[None]:
    clear_run_provider_slots()
    yield
    clear_run_provider_slots()


@pytest.mark.asyncio
async def test_dispatcher_binds_the_snapshots_pins_for_the_stage() -> None:
    observed: dict[str, object] = {}

    async def _stage(*_args: object, **_kwargs: object) -> None:
        observed["llm"] = current_run_slot("llm")
        observed["tts"] = current_run_slot("tts")
        observed["tier"] = current_run_model_tier("timing_director", default="env-tier")
        raise RuntimeError("stop after observing")

    with patch("src.services.run_dispatcher.RunDispatcher._recompute_project", new=AsyncMock()):
        await _drive_dispatcher(
            executor_side_effect=_stage, project_id=uuid4(), snapshot_extra=PINS
        )

    llm = observed["llm"]
    assert isinstance(llm, SlotRef) and llm.source == "tenant"
    assert (llm.connection_id, llm.version) == (CONN, 4)
    tts = observed["tts"]
    assert isinstance(tts, SlotRef) and tts.source == "deployment"
    assert observed["tier"] == "tier-high"
    # Bound inside the run task only.
    assert current_run_slot("llm") is None


@pytest.mark.asyncio
async def test_legacy_run_binds_nothing_pinned() -> None:
    observed: dict[str, object] = {}

    async def _stage(*_args: object, **_kwargs: object) -> None:
        observed["llm"] = current_run_slot("llm")
        observed["tier"] = current_run_model_tier("timing_director", default="env-tier")
        raise RuntimeError("stop")

    with patch("src.services.run_dispatcher.RunDispatcher._recompute_project", new=AsyncMock()):
        await _drive_dispatcher(executor_side_effect=_stage, project_id=uuid4(), snapshot_extra={})
    assert observed == {"llm": None, "tier": "env-tier"}


def test_a_present_none_tier_is_a_pin_not_a_fallback() -> None:
    bind_run_provider_slots({"model_tier": {"timing_director": None}})
    assert current_run_model_tier("timing_director", default="env-tier") is None


def test_malformed_pin_fails_closed() -> None:
    bind_run_provider_slots({"provider_slots": {"llm": {"source": "tenant", "kind": "x"}}})
    with pytest.raises(SlotUnavailableError):
        current_run_slot("llm")


def _stage() -> ReconcileTimelineStage:
    return ReconcileTimelineStage(client=MagicMock())


def test_reconcile_reads_the_tier_from_the_run_snapshot(monkeypatch) -> None:
    monkeypatch.setattr(get_settings().pipeline, "timing_director_model_tier", "tier-medium")
    stage = _stage()

    def _in_run(snapshot: dict) -> str | None:
        bind_run_provider_slots(snapshot)
        return stage._model_tier

    # Pinned at run start → the snapshot wins over a later env flip.
    assert contextvars.copy_context().run(_in_run, PINS) == "tier-high"
    # Legacy run (no model_tier) → env fallback.
    assert contextvars.copy_context().run(_in_run, {}) == "tier-medium"
    # Unbound (unit tests, dry runs) → env.
    assert stage._model_tier == "tier-medium"
