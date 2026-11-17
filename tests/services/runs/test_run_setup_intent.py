"""Unit tests for build_config_snapshot's A0 authoring-front-door intent lift.

The lift logic (``_apply_intent_plan`` → ``resolve_plan``) is pure; the only DB
touch in build_config_snapshot is the incidental verbatim/storyboard lookups,
stubbed to ``[]`` so these run hermetically here AND in CI (no live postgres, so
they can never silently skip — a skipped test is not a test).

Headline guarantees under test:
- A brief WITHOUT an ``intent`` block never runs the behavioural brief-merge
  (no ``lines``/knobs merged, no ``intent`` key) — the delivered video is
  byte-identical. It DOES record an audit-only ``plan_document`` provenance
  ledger for demo_video/showcase runs (universal-ledger follow-up), gated by
  ``provenance_ledger_universal_enabled`` and scoped to those types.
- intent + package ⇒ footage-bound ``lines`` materialized + ``plan_document`` set.
- The make-or-break rule: a showcase intent with NO package never materializes
  authored ``lines`` (they would die stage-deep at demo_video_record).
- Per-knob override: an authored ``sound_effects`` gain survives the merge.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, patch
from uuid import uuid4, uuid7

import pytest

from src.services.runs.run_setup import build_config_snapshot

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

_REPO = "src.services.runs.run_setup.VerbatimItemRepository"


def _fake_session() -> AsyncSession:
    return cast("AsyncSession", object())


@pytest.fixture(autouse=True)
def _stub_storyboard_repo() -> object:
    with patch("src.services.runs.run_setup.StoryboardPanelRepository") as repo_cls:
        repo_cls.return_value.list_for_project = AsyncMock(return_value=[])
        yield


def _proj(*, brief: dict[str, Any], type_: str = "showcase") -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid7(),
        title="A Showcase",
        type=type_,
        brief=brief,
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


async def _build(brief: dict[str, Any], *, type_: str = "showcase") -> dict[str, Any]:
    with patch(_REPO) as repo_cls:
        repo_cls.return_value.list_for_project = AsyncMock(return_value=[])
        return await build_config_snapshot(
            _fake_session(),
            proj=_proj(brief=brief, type_=type_),
            deliv=_deliv(),
            tenant_id=uuid4(),
            subject_id=uuid4(),
            config_overrides=None,
        )


def _intent(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "type": "showcase",
        "title": "A Showcase",
        "target_url": "https://example.com",
        "preset": "default",
    }
    base.update(over)
    return base


@pytest.mark.asyncio
async def test_absent_intent_records_audit_ledger_but_brief_byte_identical() -> None:
    """A demo_video/showcase brief with no `intent` records an AUDIT-ONLY
    `plan_document` ledger — but NEVER runs the behavioural brief-merge, so the
    brief (and thus the delivered video) is byte-identical: no `intent` key, no
    merged `lines`/`package_url`, and the authored brief is untouched."""
    snap = await _build({})
    # Audit ledger present + populated (this is what lights up the manifest card).
    assert snap["plan_document"]["provenance"]["resolution"]["source"] == "default"
    # But the behavioural bridge never fired: the brief is deep-equal to the
    # authored input (no merged knobs of any kind), and no intent key leaked.
    assert snap["brief"] == {}
    assert "intent" not in snap


@pytest.mark.asyncio
async def test_universal_ledger_kill_switch_off_is_byte_identical() -> None:
    """With `provenance_ledger_universal_enabled` OFF, a non-intent run records
    no `plan_document` at all — fully byte-identical snapshot."""
    with patch("src.services.runs.run_setup.get_settings") as gs:
        gs.return_value.pipeline.planning_from_intent_enabled = True
        gs.return_value.pipeline.provenance_ledger_universal_enabled = False
        gs.return_value.pipeline.sound_default_on_showcase = False
        snap = await _build({})
    assert "plan_document" not in snap
    blob = json.dumps(snap, sort_keys=True, default=str)
    assert "plan_document" not in blob


@pytest.mark.asyncio
async def test_non_ledger_type_skips_universal_ledger() -> None:
    """The universal ledger is scoped to types whose blueprints carry
    render_manifest (demo_video/showcase). A `generic` run records nothing."""
    snap = await _build({}, type_="generic")
    assert "plan_document" not in snap


@pytest.mark.asyncio
async def test_intent_with_package_materializes_lines_and_plan() -> None:
    """intent + package ⇒ footage-bound lines materialized into the brief +
    a plan_document audit sidecar with per-knob provenance."""
    snap = await _build(
        {
            "intent": _intent(
                package="minio://w2v/pkg.zip",
                must_show=["home", "editor"],
                must_say=["Meet the editor.", "Ship in one click."],
            )
        }
    )
    lines = snap["brief"]["lines"]
    assert [line["id"] for line in lines] == ["clip-1", "clip-2"]
    assert [line["text"] for line in lines] == ["Meet the editor.", "Ship in one click."]
    assert snap["brief"]["package_url"] == "minio://w2v/pkg.zip"
    assert snap["plan_document"]["input_hash"]
    assert snap["plan_document"]["provenance"]["music_gain_db"]["source"] == "default"


@pytest.mark.asyncio
async def test_showcase_intent_without_package_never_materializes_lines() -> None:
    """The make-or-break rule: no footage ⇒ no authored `lines` (which would die
    stage-deep at demo_video_record). The plan still records provenance."""
    snap = await _build({"intent": _intent(must_show=["home"], must_say=["Hello."])})
    assert not snap["brief"].get("lines")
    assert not snap["brief"].get("package_url")
    assert snap["plan_document"]["lines"] == []


@pytest.mark.asyncio
async def test_authored_gain_wins_over_computed() -> None:
    """A hand-authored sound_effects gain survives the intent merge (per-knob
    override), and the plan records it as an `override`."""
    snap = await _build(
        {
            "intent": _intent(package="minio://w2v/pkg.zip", must_show=["home"]),
            "sound_effects": {"music_gain_db": -22},
        }
    )
    assert snap["sound_effects"]["music_gain_db"] == -22
    assert snap["plan_document"]["provenance"]["music_gain_db"]["source"] == "override"


@pytest.mark.asyncio
async def test_absent_intent_kill_switch_off_still_byte_identical() -> None:
    """With the intent kill-switch OFF, an intent block is ignored entirely — no
    behavioural brief-merge. (The universal audit ledger is also disabled here so
    this stays a focused byte-identical assertion of the intent path.)"""
    with patch("src.services.runs.run_setup.get_settings") as gs:
        gs.return_value.pipeline.planning_from_intent_enabled = False
        gs.return_value.pipeline.provenance_ledger_universal_enabled = False
        gs.return_value.pipeline.sound_default_on_showcase = False
        snap = await _build({"intent": _intent(package="minio://w2v/pkg.zip", must_show=["home"])})
    assert "plan_document" not in snap
    assert not snap["brief"].get("lines")


# P3: build_config_snapshot pins provider slots; these tests run on a sentinel session.
pytestmark = pytest.mark.usefixtures("stub_provider_pins")
