"""Unit tests for build_config_snapshot's ORCH-3.3 showcase grade-default lift.

Hermetic like ``test_run_setup_direction``: the two DB touches in
build_config_snapshot (verbatim + storyboard) are stubbed to ``[]`` and the
``get_settings`` call is patched, so the headline guarantee — the whole-film
colorist default is showcase-only, flag-gated, and never clobbers an author's
``finish.grade`` — runs without a live postgres (it can never silently skip).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, patch
from uuid import uuid4, uuid7

import pytest

from src.services.runs.run_setup import build_config_snapshot

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

_VERBATIM = "src.services.runs.run_setup.VerbatimItemRepository"
_STORYBOARD = "src.services.runs.run_setup.StoryboardPanelRepository"
_SETTINGS = "src.services.runs.run_setup.get_settings"


def _fake_session() -> AsyncSession:
    return cast("AsyncSession", object())


def _proj(*, type_: str, brief: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid7(),
        title="A Showcase",
        type=type_,
        brief=brief or {},
        brand={},
        voice={},
        theme={},
    )


def _deliv() -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid7(), name="d", format="16x9", captions_mode="none", target_duration_seconds=None
    )


async def _build(*, type_: str, brief: dict, flag: bool) -> dict:
    # sound_default_on_showcase is read by the sibling ORCH-4.1 sound-lift that
    # runs before the grade-lift; stub it off so it stays a no-op for these tests.
    settings = SimpleNamespace(
        pipeline=SimpleNamespace(
            grade_default_on_showcase=flag,
            sound_default_on_showcase=False,
            # WS-1 opt-in-stage defaults: off, so this file keeps asserting the grade
            # lift in isolation (the injection has its own file,
            # test_run_setup_stage_defaults.py).
            ship_gate_default_enabled=False,
            footage_direction_default_enabled=False,
            footage_critique_default_enabled=False,
            retention_default_enabled=False,
            timing_default_enabled=False,
            voice_direction_default_enabled=False,
            claim_finish_default_enabled=False,
            script_doctor_default_enabled=False,
            # Isolate the grade feature under test from the universal provenance
            # ledger (a non-intent showcase run would otherwise record one).
            provenance_ledger_universal_enabled=False,
            # P3: the run pins this tier into ``model_tier`` (default None).
            timing_director_model_tier=None,
        )
    )
    with (
        patch(_VERBATIM) as verbatim_cls,
        patch(_STORYBOARD) as storyboard_cls,
        patch(_SETTINGS, return_value=settings),
    ):
        verbatim_cls.return_value.list_for_project = AsyncMock(return_value=[])
        storyboard_cls.return_value.list_for_project = AsyncMock(return_value=[])
        return await build_config_snapshot(
            _fake_session(),
            proj=_proj(type_=type_, brief=brief),
            deliv=_deliv(),
            tenant_id=uuid4(),
            subject_id=uuid4(),
            config_overrides=None,
        )


@pytest.mark.asyncio
async def test_flag_off_showcase_injects_nothing() -> None:
    """Flag off ⇒ no finish block injected (byte-identical to before)."""
    snap = await _build(type_="showcase", brief={}, flag=False)
    assert "finish" not in snap


@pytest.mark.asyncio
async def test_flag_on_non_showcase_untouched() -> None:
    """Flag on but a non-showcase project ⇒ no finish block injected."""
    snap = await _build(type_="promo", brief={}, flag=True)
    assert "finish" not in snap


@pytest.mark.asyncio
async def test_flag_on_showcase_injects_enabled_grade() -> None:
    """Flag on + showcase + no brief finish ⇒ an enabled finish block with grade."""
    snap = await _build(type_="showcase", brief={}, flag=True)
    assert snap["finish"] == {"enabled": True, "grade": {}}


@pytest.mark.asyncio
async def test_author_finish_grade_is_never_clobbered() -> None:
    """A brief that already authored finish.grade wins outright."""
    brief = {"finish": {"enabled": True, "grade": {"look": "bw"}}}
    snap = await _build(type_="showcase", brief=brief, flag=True)
    assert snap["finish"]["grade"] == {"look": "bw"}


@pytest.mark.asyncio
async def test_explicit_disabled_finish_stays_disabled() -> None:
    """A brief finish block with enabled=False but no grade gets a grade key added,
    but its explicit enabled=False is respected (the grade stays inert)."""
    brief = {"finish": {"enabled": False, "letterbox_color": "#123456"}}
    snap = await _build(type_="showcase", brief=brief, flag=True)
    assert snap["finish"]["enabled"] is False
    assert snap["finish"]["grade"] == {}
    assert snap["finish"]["letterbox_color"] == "#123456"


# P3: build_config_snapshot pins provider slots; these tests run on a sentinel session.
pytestmark = pytest.mark.usefixtures("stub_provider_pins")
