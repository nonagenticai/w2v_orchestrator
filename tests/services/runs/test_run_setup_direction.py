"""Unit tests for build_config_snapshot's direction + reproducibility lift.

The lift logic under test is pure; the only DB touch in build_config_snapshot
is the incidental verbatim-items lookup, which we stub to `[]` so the headline
guarantee — presence-is-opt-in: a brief WITHOUT the new blocks yields a
byte-identical snapshot (no new keys, no master_seed) — runs hermetically here
AND in CI (no live postgres required, so it can never silently skip).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, patch
from uuid import uuid4, uuid7

import pytest

from src.config import get_settings
from src.services.runs.run_setup import build_config_snapshot

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

# Patch target: VerbatimItemRepository as imported into run_setup's namespace.
_REPO = "src.services.runs.run_setup.VerbatimItemRepository"

#: WS-1 kill-switches. build_config_snapshot now seeds `direction.{retention,timing,
#: footage}` (and siblings) for the seven opt-in stages that never ran in prod; these
#: tests assert the brief LIFT in isolation, so turn the injection off here. Its own
#: coverage lives in test_run_setup_stage_defaults.py.
_WS1_FLAGS = (
    "ship_gate_default_enabled",
    "footage_direction_default_enabled",
    "footage_critique_default_enabled",
    "retention_default_enabled",
    "timing_default_enabled",
    "voice_direction_default_enabled",
    "claim_finish_default_enabled",
)


@pytest.fixture(autouse=True)
def _ws1_stage_defaults_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable every WS-1 default-injection flag on the cached settings singleton
    (``get_settings`` is ``lru_cache``d, so this is the same object run_setup reads)."""
    pipeline = get_settings().pipeline
    for flag in _WS1_FLAGS:
        monkeypatch.setattr(pipeline, flag, False)


def _fake_session() -> AsyncSession:
    """A sentinel session; unused once the repositories are patched."""
    return cast("AsyncSession", object())


@pytest.fixture(autouse=True)
def _stub_storyboard_repo() -> object:
    """build_config_snapshot gained a second DB touch with the Phase-0 storyboard
    seed (StoryboardPanelRepository); these tests exercise only the pure direction/
    reproducibility lift, so stub it to [] like the verbatim lookup — otherwise the
    sentinel ``object()`` session hits a real ``.execute``."""
    with patch("src.services.runs.run_setup.StoryboardPanelRepository") as repo_cls:
        repo_cls.return_value.list_for_project = AsyncMock(return_value=[])
        yield


def _proj(*, brief: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid7(),
        title="A Promo",
        type="promo",
        brief=brief or {},
        brand={},
        voice={},
        theme={},
    )


def _deliv() -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid7(),
        name="d",
        format="16x9",
        captions_mode="none",
        target_duration_seconds=None,
    )


async def _build(brief: dict) -> dict:
    """Run build_config_snapshot with the verbatim DB lookup stubbed to []."""
    with patch(_REPO) as repo_cls:
        repo_cls.return_value.list_for_project = AsyncMock(return_value=[])
        return await build_config_snapshot(
            _fake_session(),
            proj=_proj(brief=brief),
            deliv=_deliv(),
            tenant_id=uuid4(),
            subject_id=uuid4(),
            config_overrides=None,
        )


@pytest.mark.asyncio
async def test_absent_direction_and_reproducibility_byte_identical() -> None:
    """A brief with neither block adds neither key: the snapshot carries no
    `direction`, `reproducibility`, or `master_seed` anywhere."""
    snap = await _build({})
    assert "direction" not in snap
    assert "reproducibility" not in snap
    blob = json.dumps(snap, sort_keys=True, default=str)
    assert "reproducibility" not in blob
    assert "master_seed" not in blob
    assert "direction" not in blob


@pytest.mark.asyncio
async def test_brief_direction_lifts_into_snapshot() -> None:
    cfg = {"tone": "energetic", "pacing": "fast"}
    snap = await _build({"direction": cfg})
    assert snap["direction"] == cfg


@pytest.mark.asyncio
async def test_brief_direction_footage_survives_the_lift() -> None:
    """ORCH-3.2: direction is lifted WHOLE, so the nested `footage` block (the
    directed-footage gate read by site_capture + the promo/website_review compile)
    rides into config_snapshot for free — no dedicated lift key needed."""
    footage = {"auto_zoom": True, "show_clicks": True}
    snap = await _build({"direction": {"footage": footage}})
    assert snap["direction"]["footage"] == footage


@pytest.mark.asyncio
async def test_reproducibility_mode_off_gets_no_master_seed() -> None:
    """A reproducibility block with mode=off lifts verbatim but is NOT stamped."""
    snap = await _build({"reproducibility": {"mode": "off"}})
    assert snap["reproducibility"] == {"mode": "off"}
    assert "master_seed" not in snap["reproducibility"]


@pytest.mark.asyncio
async def test_reproducibility_active_stamps_deterministic_master_seed() -> None:
    """mode != off ⇒ a deterministic master_seed is stamped; a second build for
    the same project yields the same seed (keyed on project_id)."""
    # Fix the project id so both builds key on the same value.
    pid = uuid7()

    async def _build_fixed() -> dict:
        with patch(_REPO) as repo_cls:
            repo_cls.return_value.list_for_project = AsyncMock(return_value=[])
            proj = _proj(brief={"reproducibility": {"mode": "on"}})
            proj.id = pid
            return await build_config_snapshot(
                _fake_session(),
                proj=proj,
                deliv=_deliv(),
                tenant_id=uuid4(),
                subject_id=uuid4(),
                config_overrides=None,
            )

    snap = await _build_fixed()
    seed = snap["reproducibility"]["master_seed"]
    assert isinstance(seed, int)
    assert 0 <= seed < 2**31
    assert snap["reproducibility"]["mode"] == "on"

    snap2 = await _build_fixed()
    assert snap2["reproducibility"]["master_seed"] == seed


@pytest.mark.asyncio
async def test_reproducibility_stamp_does_not_mutate_source_brief() -> None:
    """The lift copies the block before stamping, so the project brief's dict is
    never mutated with a master_seed."""
    brief = {"reproducibility": {"mode": "on"}}
    with patch(_REPO) as repo_cls:
        repo_cls.return_value.list_for_project = AsyncMock(return_value=[])
        await build_config_snapshot(
            _fake_session(),
            proj=_proj(brief=brief),
            deliv=_deliv(),
            tenant_id=uuid4(),
            subject_id=uuid4(),
            config_overrides=None,
        )
    assert "master_seed" not in brief["reproducibility"]


# P3: build_config_snapshot pins provider slots; these tests run on a sentinel session.
pytestmark = pytest.mark.usefixtures("stub_provider_pins")
